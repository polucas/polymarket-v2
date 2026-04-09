import json
import httpx
import structlog
from typing import List, Optional
from datetime import datetime, timezone

from src.config import Settings
from src.models import Market, OrderBook, OrderBookLevel
from src.pipelines.market_classifier import classify_market_type

# CLOB SDK — only required for live trading; imported at module level for testability.
try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import ApiCreds, OrderArgs
except ImportError:  # pragma: no cover
    ClobClient = None  # type: ignore[assignment]
    ApiCreds = None  # type: ignore[assignment]
    OrderArgs = None  # type: ignore[assignment]

log = structlog.get_logger()

GAMMA_API_BASE = "https://gamma-api.polymarket.com"
CLOB_API_BASE = "https://clob.polymarket.com"


class PolymarketClient:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._timeout = httpx.Timeout(15.0, connect=5.0)

    async def get_active_markets(self, tier: int, tier1_fee_rate: float = 0.0, tier2_fee_rate: float = 0.04) -> List[Market]:
        """Get active markets filtered by tier criteria.
        Tier 1: resolution 15m-7d, liquidity > $5K
        Tier 2: crypto markets, 15-min resolution
        """
        all_raw = []
        page_size = self._settings.MARKET_PAGE_SIZE
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                for page in range(self._settings.MARKET_FETCH_PAGES):
                    params = {
                        "active": "true",
                        "closed": "false",
                        "limit": page_size,
                        "offset": page * page_size,
                        "order": "volume24hr",
                        "ascending": "false",
                    }
                    resp = await client.get(f"{GAMMA_API_BASE}/markets", params=params)
                    if resp.status_code == 429:
                        log.warning("polymarket_rate_limited", page=page)
                        break  # Keep what we have so far instead of returning empty
                    resp.raise_for_status()
                    page_markets = resp.json()
                    all_raw.extend(page_markets)
                    log.debug("market_page_fetched", page=page, count=len(page_markets))
                    if len(page_markets) < page_size:
                        break  # No more pages
        except httpx.TimeoutException:
            log.warning("polymarket_timeout")
            if not all_raw:
                return []
        except Exception as e:
            log.error("polymarket_fetch_failed", error=str(e))
            if not all_raw:
                return []

        markets = []
        now = datetime.now(timezone.utc)
        total_parsed = 0
        filtered_resolution = 0
        filtered_liquidity = 0
        filtered_price_range = 0
        for m in all_raw:
            try:
                # Parse prices
                outcomes = m.get("outcomes", [])
                outcomePrices = m.get("outcomePrices", "")
                if isinstance(outcomePrices, str) and outcomePrices:
                    prices = json.loads(outcomePrices)
                elif isinstance(outcomePrices, list):
                    prices = outcomePrices
                else:
                    prices = [0.5, 0.5]

                yes_price = float(prices[0]) if len(prices) > 0 else 0.5
                no_price = float(prices[1]) if len(prices) > 1 else 1 - yes_price

                # Resolution time
                end_date = m.get("endDate") or m.get("end_date_iso")
                resolution_time = None
                hours_to_resolution = 0.0
                if end_date:
                    try:
                        resolution_time = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                        hours_to_resolution = max(0, (resolution_time - now).total_seconds() / 3600)
                    except (ValueError, TypeError):
                        pass

                question = m.get("question", "")
                market_type = classify_market_type(question)

                volume_24h = float(m.get("volume24hr", 0) or 0)
                liquidity = float(m.get("liquidity", 0) or 0)

                # Parse keywords from question
                keywords = [w.lower() for w in question.split() if len(w) > 3]

                # Check if resolved
                resolved = m.get("closed", False) or m.get("resolved", False)
                resolution = None
                if resolved:
                    # Gamma API uses outcomePrices, not resolutionPrices
                    op_raw_r = m.get("outcomePrices", "")
                    if isinstance(op_raw_r, str) and op_raw_r:
                        try:
                            op_r = json.loads(op_raw_r)
                        except (json.JSONDecodeError, ValueError):
                            op_r = None
                    elif isinstance(op_raw_r, list):
                        op_r = op_raw_r
                    else:
                        op_r = None
                    if op_r and len(op_r) >= 2:
                        try:
                            yes_price_r = float(op_r[0])
                            if yes_price_r > 0.5:
                                resolution = "YES"
                            elif yes_price_r < 0.5:
                                resolution = "NO"
                        except (ValueError, TypeError):
                            pass

                # Extract CLOB token IDs
                clob_token_ids_raw = m.get("clobTokenIds", "[]")
                if isinstance(clob_token_ids_raw, str):
                    clob_tokens = json.loads(clob_token_ids_raw)
                else:
                    clob_tokens = clob_token_ids_raw or []
                clob_token_id_yes = str(clob_tokens[0]) if len(clob_tokens) > 0 else ""
                clob_token_id_no = str(clob_tokens[1]) if len(clob_tokens) > 1 else ""

                market = Market(
                    market_id=str(m.get("id", m.get("condition_id", ""))),
                    question=question,
                    yes_price=yes_price,
                    no_price=no_price,
                    resolution_time=resolution_time,
                    hours_to_resolution=hours_to_resolution,
                    volume_24h=volume_24h,
                    liquidity=liquidity,
                    market_type=market_type,
                    fee_rate=tier1_fee_rate if tier == 1 else tier2_fee_rate,
                    keywords=keywords[:10],
                    resolved=resolved,
                    resolution=resolution,
                    clob_token_id_yes=clob_token_id_yes,
                    clob_token_id_no=clob_token_id_no,
                )

                total_parsed += 1

                # Apply tier filters
                if tier == 1:
                    if hours_to_resolution < 0.25 or hours_to_resolution > 168:
                        filtered_resolution += 1
                        continue
                    if liquidity < 5000:
                        filtered_liquidity += 1
                        continue
                elif tier == 2:
                    if market_type != "crypto_15m":
                        continue

                # Price range filter — extreme prices create dangerous leverage
                if yes_price > self._settings.MAX_TRADEABLE_PRICE or yes_price < self._settings.MIN_TRADEABLE_PRICE:
                    filtered_price_range += 1
                    continue

                markets.append(market)
            except Exception as e:
                log.warning("market_parse_error", error=str(e), market_id=m.get("id"))
                continue

        log.info("market_filter_results",
                 tier=tier,
                 total_from_api=total_parsed,
                 total_raw=len(all_raw),
                 passed=len(markets),
                 filtered_resolution=filtered_resolution,
                 filtered_liquidity=filtered_liquidity,
                 filtered_price_range=filtered_price_range)
        return markets

    async def get_orderbook(self, token_id: str, market_id: str = "") -> OrderBook:
        """Get top 5 bid/ask levels."""
        if not token_id:
            log.warning("orderbook_no_token_id", market_id=market_id)
            return OrderBook(market_id=market_id)
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(
                    f"{CLOB_API_BASE}/book",
                    params={"token_id": token_id},
                )
                resp.raise_for_status()
                data = resp.json()
                raw_bids = sorted(data.get("bids", []), key=lambda b: float(b.get("price", 0)), reverse=True)
                raw_asks = sorted(data.get("asks", []), key=lambda a: float(a.get("price", 0)))
                bids = [OrderBookLevel(price=float(b.get("price", 0)), size=float(b.get("size", 0)))
                        for b in raw_bids[:5]]
                asks = [OrderBookLevel(price=float(a.get("price", 0)), size=float(a.get("size", 0)))
                        for a in raw_asks[:5]]
                return OrderBook(market_id=market_id, bids=bids, asks=asks, timestamp=datetime.now(timezone.utc))
        except Exception as e:
            log.warning("orderbook_fetch_failed", market_id=market_id, error=str(e))
            return OrderBook(market_id=market_id)

    async def get_market(self, market_id: str) -> Optional[Market]:
        """Get single market including resolution status."""
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(f"{GAMMA_API_BASE}/markets/{market_id}")
                resp.raise_for_status()
                m = resp.json()

                outcomes = m.get("outcomes", [])
                outcomePrices = m.get("outcomePrices", "")
                if isinstance(outcomePrices, str) and outcomePrices:
                    prices = json.loads(outcomePrices)
                elif isinstance(outcomePrices, list):
                    prices = outcomePrices
                else:
                    prices = [0.5, 0.5]

                yes_price = float(prices[0]) if len(prices) > 0 else 0.5
                no_price = float(prices[1]) if len(prices) > 1 else 1 - yes_price

                resolved = m.get("closed", False) or m.get("resolved", False)
                resolution = None
                if resolved:
                    # Gamma API uses outcomePrices (a JSON-encoded list like ["1","0"] or
                    # ["0","1"]) — there is no resolutionPrices field (fixed in commit 65b8caa
                    # for backtest data_ingestion; mirrored here).
                    op_raw = m.get("outcomePrices", "")
                    if isinstance(op_raw, str) and op_raw:
                        try:
                            op = json.loads(op_raw)
                        except (json.JSONDecodeError, ValueError):
                            op = None
                    elif isinstance(op_raw, list):
                        op = op_raw
                    else:
                        op = None
                    if op and len(op) >= 2:
                        try:
                            yes_price_resolved = float(op[0])
                            if yes_price_resolved > 0.5:
                                resolution = "YES"
                            elif yes_price_resolved < 0.5:
                                resolution = "NO"
                            # 0.5 = voided/cancelled, leave as None
                        except (ValueError, TypeError):
                            pass

                question = m.get("question", "")
                return Market(
                    market_id=str(m.get("id", m.get("condition_id", ""))),
                    question=question,
                    yes_price=yes_price,
                    no_price=no_price,
                    market_type=classify_market_type(question),
                    resolved=resolved,
                    resolution=resolution,
                    keywords=[w.lower() for w in question.split() if len(w) > 3][:10],
                )
        except Exception as e:
            log.error("market_get_failed", market_id=market_id, error=str(e))
            return None

    async def place_order(
        self,
        clob_token_id: str,
        price: float,
        size: float,
        side: str = "BUY",
    ) -> dict:
        """Place a BUY order for a specific ERC-1155 outcome token via CLOB API.

        To buy YES shares, pass the market's clob_token_id_yes.
        To buy NO shares, pass the market's clob_token_id_no.
        Side is always "BUY" when opening a position — the direction is encoded
        by which token_id you target.

        Only executes in ENVIRONMENT=live. Paper mode returns a rejected response.
        """
        if self._settings.ENVIRONMENT != "live":
            log.warning("place_order_called_in_paper_mode")
            return {"status": "rejected", "reason": "paper_mode"}

        try:
            # Level 2 auth: host + private key (L1) + API creds (L2).
            # funder is required for proxy/smart wallets; leave None for EOA.
            funder = self._settings.POLYMARKET_FUNDER_ADDRESS or None
            clob = ClobClient(
                host=CLOB_API_BASE,
                key=self._settings.POLYMARKET_PRIVATE_KEY,
                chain_id=137,
                creds=ApiCreds(
                    api_key=self._settings.POLYMARKET_API_KEY,
                    api_secret=self._settings.POLYMARKET_SECRET,
                    api_passphrase=self._settings.POLYMARKET_PASSPHRASE,
                ),
                funder=funder,
            )
            # OrderArgs uses token_id (snake_case), not tokenID.
            # side is always "BUY" — the token_id encodes YES vs NO direction.
            order_args = OrderArgs(
                token_id=clob_token_id,
                price=price,
                size=size,
                side=side.upper(),
            )
            order = clob.create_and_post_order(order_args)
            return {"status": "submitted", "order": order}
        except Exception as e:
            log.error("order_failed", token_id=clob_token_id, error=str(e))
            return {"status": "error", "error": str(e)}
