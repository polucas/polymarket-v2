import json
import httpx
import structlog
from typing import List, Optional
from datetime import datetime, timezone

from src.config import Settings
from src.models import Market, OrderBook

log = structlog.get_logger()

GAMMA_API_BASE = "https://gamma-api.polymarket.com"
CLOB_API_BASE = "https://clob.polymarket.com"

MARKET_TYPE_KEYWORDS = {
    "political": ["president", "election", "congress", "senate", "vote", "political", "trump", "biden", "governor", "democrat", "republican"],
    "economic": ["gdp", "inflation", "fed", "interest rate", "unemployment", "economy", "recession", "jobs", "cpi", "fomc"],
    "crypto_15m": ["bitcoin", "btc", "ethereum", "eth", "crypto", "solana", "sol"],
    "sports": ["nba", "nfl", "mlb", "nhl", "soccer", "football", "basketball", "baseball", "championship", "super bowl"],
    "cultural": ["oscar", "grammy", "emmy", "movie", "album", "show", "celebrity", "entertainment"],
    "regulatory": ["sec", "regulation", "law", "ban", "approve", "fda", "ruling", "court"],
}


class PolymarketClient:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._timeout = httpx.Timeout(15.0, connect=5.0)

    def _classify_market_type(self, question: str, tags: list = None) -> str:
        q_lower = question.lower()
        for mtype, keywords in MARKET_TYPE_KEYWORDS.items():
            if any(kw in q_lower for kw in keywords):
                return mtype
        return "political"  # default

    async def get_active_markets(self, tier: int, tier1_fee_rate: float = 0.0, tier2_fee_rate: float = 0.04) -> List[Market]:
        """Get active markets filtered by tier criteria.
        Tier 1: resolution 15m-7d, liquidity > $5K
        Tier 2: crypto markets, 15-min resolution
        """
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                params = {"active": "true", "closed": "false", "limit": 100}
                resp = await client.get(f"{GAMMA_API_BASE}/markets", params=params)
                if resp.status_code == 429:
                    log.warning("polymarket_rate_limited")
                    return []
                resp.raise_for_status()
                raw_markets = resp.json()
        except httpx.TimeoutException:
            log.warning("polymarket_timeout")
            return []
        except Exception as e:
            log.error("polymarket_fetch_failed", error=str(e))
            return []

        markets = []
        now = datetime.now(timezone.utc)
        total_parsed = 0
        filtered_resolution = 0
        filtered_liquidity = 0
        for m in raw_markets:
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
                market_type = self._classify_market_type(question, m.get("tags", []))

                volume_24h = float(m.get("volume24hr", 0) or 0)
                liquidity = float(m.get("liquidity", 0) or 0)

                # Parse keywords from question
                keywords = [w.lower() for w in question.split() if len(w) > 3]

                # Check if resolved
                resolved = m.get("closed", False) or m.get("resolved", False)
                resolution = None
                if resolved:
                    res_prices = m.get("resolutionPrices", {})
                    if res_prices:
                        resolution = "YES" if float(res_prices.get("0", 0)) > 0.5 else "NO"

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

                markets.append(market)
            except Exception as e:
                log.warning("market_parse_error", error=str(e), market_id=m.get("id"))
                continue

        log.info("market_filter_results",
                 tier=tier,
                 total_from_api=total_parsed,
                 passed=len(markets),
                 filtered_resolution=filtered_resolution,
                 filtered_liquidity=filtered_liquidity)
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
                bids = [float(b.get("size", 0)) for b in (data.get("bids", []))[:5]]
                asks = [float(a.get("size", 0)) for a in (data.get("asks", []))[:5]]
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
                    res_prices = m.get("resolutionPrices", {})
                    if res_prices:
                        resolution = "YES" if float(res_prices.get("0", 0)) > 0.5 else "NO"

                question = m.get("question", "")
                return Market(
                    market_id=str(m.get("id", m.get("condition_id", ""))),
                    question=question,
                    yes_price=yes_price,
                    no_price=no_price,
                    market_type=self._classify_market_type(question),
                    resolved=resolved,
                    resolution=resolution,
                    keywords=[w.lower() for w in question.split() if len(w) > 3][:10],
                )
        except Exception as e:
            log.error("market_get_failed", market_id=market_id, error=str(e))
            return None

    async def place_order(self, market_id: str, side: str, price: float, size: float) -> dict:
        """Place order via CLOB API. Only for live mode."""
        if self._settings.ENVIRONMENT != "live":
            log.warning("place_order_called_in_paper_mode")
            return {"status": "rejected", "reason": "paper_mode"}

        try:
            from py_clob_client.client import ClobClient
            clob = ClobClient(
                host=CLOB_API_BASE,
                key=self._settings.POLYMARKET_API_KEY,
                chain_id=137,
            )
            order = clob.create_and_post_order({
                "tokenID": market_id,
                "side": "BUY" if "YES" in side.upper() else "SELL",
                "price": price,
                "size": size,
            })
            return {"status": "submitted", "order": order}
        except Exception as e:
            log.error("order_failed", market_id=market_id, error=str(e))
            return {"status": "error", "error": str(e)}
