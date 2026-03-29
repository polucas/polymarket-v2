from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import structlog

from src.backtest.clock import Clock
from src.models import Market, OrderBook, OrderBookLevel, Signal
from src.pipelines.signal_classifier import classify_source_tier, classify_info_type, SOURCE_TIER_CREDIBILITY

log = structlog.get_logger()

DEFAULT_DOMAINS = ["reuters.com", "apnews.com", "bbc.com", "bloomberg.com", "coindesk.com"]

# --- MARKET TYPE KEYWORDS (copied from polymarket.py, no dep) ---
MARKET_TYPE_KEYWORDS = {
    "political": ["president", "election", "congress", "senate", "vote", "political", "trump", "biden", "governor", "democrat", "republican"],
    "economic": ["gdp", "inflation", "fed", "interest rate", "unemployment", "economy", "recession", "jobs", "cpi", "fomc"],
    "crypto_15m": ["bitcoin", "btc", "ethereum", "eth", "crypto", "solana", "sol"],
    "sports": ["nba", "nfl", "mlb", "nhl", "soccer", "football", "basketball", "baseball", "championship", "super bowl"],
    "cultural": ["oscar", "grammy", "emmy", "movie", "album", "show", "celebrity", "entertainment"],
    "regulatory": ["sec", "regulation", "law", "ban", "approve", "fda", "ruling", "court"],
}


def _classify_market_type(question: str) -> str:
    q_lower = question.lower()
    for mtype, keywords in MARKET_TYPE_KEYWORDS.items():
        if any(kw in q_lower for kw in keywords):
            return mtype
    return "political"


class BacktestPolymarketClient:
    """Reads from backtest_data.db instead of calling the real Polymarket API."""

    def __init__(self, settings, db_path: str):
        self._settings = settings
        self._db_path = db_path

    def _conn(self):
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _get_baseline(self, market_id: str) -> float:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT baseline_price FROM historical_markets WHERE market_id = ?",
                (market_id,),
            ).fetchone()
        return float(row["baseline_price"]) if row and row["baseline_price"] is not None else 0.5

    async def get_active_markets(self, tier: int, tier1_fee_rate: float = 0.0, tier2_fee_rate: float = 0.04) -> List[Market]:
        now = Clock.utcnow()
        now_iso = now.isoformat()
        markets = []

        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM historical_markets
                   WHERE (created_at IS NULL OR created_at <= ?)
                     AND resolution_datetime IS NOT NULL
                     AND resolution_datetime > ?""",
                (now_iso, now_iso),
            ).fetchall()

        for row in rows:
            try:
                resolution_time = datetime.fromisoformat(
                    str(row["resolution_datetime"]).replace("Z", "+00:00")
                )
                hours_to_resolution = max(0, (resolution_time - now).total_seconds() / 3600)

                # Tier 1 resolution filter: 0.25h to 168h
                if tier == 1:
                    if hours_to_resolution < 0.25 or hours_to_resolution > 168:
                        continue

                p = float(row["baseline_price"] or 0.5)
                # Price range filter
                min_p = getattr(self._settings, "MIN_TRADEABLE_PRICE", 0.05)
                max_p = getattr(self._settings, "MAX_TRADEABLE_PRICE", 0.95)
                if p < min_p or p > max_p:
                    continue

                question = row["question"] or ""
                fee_rate = tier1_fee_rate if tier == 1 else tier2_fee_rate
                keywords = [w.lower() for w in question.split() if len(w) > 3][:10]

                markets.append(Market(
                    market_id=row["market_id"],
                    question=question,
                    yes_price=p,
                    no_price=round(1.0 - p, 4),
                    resolution_time=resolution_time,
                    hours_to_resolution=hours_to_resolution,
                    volume_24h=10_000.0,  # synthetic — historical volume not available
                    liquidity=10_000.0,   # liquidity filter skipped in backtest
                    market_type=row["market_type"] or _classify_market_type(question),
                    fee_rate=fee_rate,
                    keywords=keywords,
                    resolved=False,
                    clob_token_id_yes=row["market_id"],
                    clob_token_id_no=row["market_id"] + "_no",
                ))
            except Exception as e:
                log.warning("backtest_market_parse_error", error=str(e))
                continue

        log.info("backtest_active_markets", tier=tier, count=len(markets), clock=now_iso)
        return markets

    async def get_orderbook(self, token_id: str, market_id: str = "") -> OrderBook:
        """Tiered synthetic orderbook — forces kelly_size_vwap() to do real math.

        3 ask levels with decaying liquidity:
          Level 1: p+0.02, $50   (thin — best ask almost always crossed)
          Level 2: p+0.04, $150  (moderate)
          Level 3: p+0.08, $500  (deep)
        Buying $160 requires crossing L1 into L2; VWAP ≈ p+0.034.
        """
        p = self._get_baseline(market_id or token_id)
        p = max(0.05, min(0.95, p))
        return OrderBook(
            market_id=market_id,
            bids=[
                OrderBookLevel(price=round(p - 0.02, 4), size=50.0),
                OrderBookLevel(price=round(p - 0.04, 4), size=150.0),
                OrderBookLevel(price=round(p - 0.08, 4), size=500.0),
            ],
            asks=[
                OrderBookLevel(price=round(p + 0.02, 4), size=50.0),
                OrderBookLevel(price=round(p + 0.04, 4), size=150.0),
                OrderBookLevel(price=round(p + 0.08, 4), size=500.0),
            ],
            timestamp=Clock.utcnow(),
        )

    async def get_market(self, market_id: str) -> Optional[Market]:
        now = Clock.utcnow()
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM historical_markets WHERE market_id = ?",
                (market_id,),
            ).fetchone()
        if row is None:
            return None

        resolution_time = None
        resolved = False
        resolution = None
        if row["resolution_datetime"]:
            try:
                resolution_time = datetime.fromisoformat(
                    str(row["resolution_datetime"]).replace("Z", "+00:00")
                )
                if now >= resolution_time:
                    resolved = True
                    resolution = row["actual_outcome"]  # "YES" or "NO"
            except (ValueError, TypeError):
                pass

        p = float(row["baseline_price"] or 0.5)
        question = row["question"] or ""
        return Market(
            market_id=row["market_id"],
            question=question,
            yes_price=p,
            no_price=round(1.0 - p, 4),
            resolution_time=resolution_time,
            hours_to_resolution=max(0, (resolution_time - now).total_seconds() / 3600) if resolution_time and not resolved else 0.0,
            market_type=row["market_type"] or _classify_market_type(question),
            resolved=resolved,
            resolution=resolution,
            keywords=[w.lower() for w in question.split() if len(w) > 3][:10],
            clob_token_id_yes=row["market_id"],
            clob_token_id_no=row["market_id"] + "_no",
        )

    async def place_order(self, market_id: str, side: str, price: float, size: float) -> dict:
        return {"status": "rejected", "reason": "backtest_mode"}


class BacktestRSSPipeline:
    """Reads from historical_news table with a 30-min window ending at Clock.utcnow()."""

    def __init__(self, db_path: str):
        self._db_path = db_path

    def _build_signals_from_rows(self, rows) -> List[Signal]:
        """Shared Signal construction logic used by both sync and async paths."""
        signals = []
        for row in rows:
            try:
                pub_dt = datetime.fromisoformat(str(row["published_at"]).replace("Z", "+00:00"))
                source_tier = classify_source_tier({
                    "source_type": "rss",
                    "domain": row["source_domain"],
                })
                signals.append(Signal(
                    source="rss",
                    source_tier=source_tier,
                    info_type=classify_info_type(source_tier),
                    content=row["headline"],
                    credibility=SOURCE_TIER_CREDIBILITY.get(source_tier, 0.30),
                    author=row["source_domain"],
                    followers=0,
                    engagement=0,
                    timestamp=pub_dt,
                    headline_only=True,
                ))
            except Exception as e:
                log.warning("backtest_rss_parse_error", error=str(e))
                continue
        return signals

    def consume_signals(self) -> List[Signal]:
        """Direct sync DB query — no event loop needed.

        Called synchronously by Scheduler.run_tier1_scan() (line 241 of scheduler.py).
        Uses sqlite3 directly to avoid any asyncio complications.
        """
        now = Clock.utcnow()
        window_start = (now - timedelta(minutes=30)).isoformat()
        window_end = now.isoformat()

        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT * FROM historical_news WHERE published_at >= ? AND published_at <= ? LIMIT 50",
                (window_start, window_end),
            ).fetchall()
        finally:
            conn.close()

        return self._build_signals_from_rows(rows)

    async def get_breaking_news(self) -> List[Signal]:
        """Async variant used by Tier 2 scan."""
        now = Clock.utcnow()
        window_start = (now - timedelta(minutes=30)).isoformat()
        window_end = now.isoformat()

        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT * FROM historical_news WHERE published_at >= ? AND published_at <= ? LIMIT 50",
                (window_start, window_end),
            ).fetchall()
        finally:
            conn.close()

        return self._build_signals_from_rows(rows)

    async def poll_and_accumulate(self) -> None:
        pass  # No-op in backtest — consume_signals() queries directly


class BacktestGrokClient:
    """Wraps the real GrokClient with a SQLite-backed prompt cache.

    Cache key: SHA256 of the context string.
    On cache hit: returns stored JSON immediately (free, deterministic).
    On cache miss: calls real Grok API, stores result.
    This ensures backtest evaluates actual LLM reasoning, not a random distribution.
    """

    def __init__(self, real_grok_client, cache_db_path: str):
        self._real_grok = real_grok_client
        self._cache_db_path = cache_db_path
        self._init_cache_db()
        self._hits = 0
        self._misses = 0

    def _init_cache_db(self) -> None:
        from pathlib import Path
        Path(self._cache_db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._cache_db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS grok_cache (
                prompt_hash TEXT PRIMARY KEY,
                market_id TEXT,
                response_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()

    def _cache_get(self, prompt_hash: str) -> Optional[dict]:
        conn = sqlite3.connect(self._cache_db_path)
        try:
            row = conn.execute(
                "SELECT response_json FROM grok_cache WHERE prompt_hash = ?",
                (prompt_hash,),
            ).fetchone()
            if row:
                return json.loads(row[0])
            return None
        finally:
            conn.close()

    def _cache_put(self, prompt_hash: str, market_id: str, result: dict) -> None:
        conn = sqlite3.connect(self._cache_db_path)
        try:
            conn.execute(
                "INSERT OR REPLACE INTO grok_cache (prompt_hash, market_id, response_json, created_at) VALUES (?, ?, ?, ?)",
                (prompt_hash, market_id, json.dumps(result), datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()
        finally:
            conn.close()

    async def call_grok_with_retry(self, context: str, market_id: str) -> Optional[dict]:
        prompt_hash = hashlib.sha256(context.encode()).hexdigest()
        cached = self._cache_get(prompt_hash)
        if cached is not None:
            self._hits += 1
            log.debug("grok_cache_hit", market_id=market_id, hits=self._hits)
            return cached

        self._misses += 1
        log.debug("grok_cache_miss", market_id=market_id, misses=self._misses)
        result = await self._real_grok.call_grok_with_retry(context, market_id)
        if result is not None:
            self._cache_put(prompt_hash, market_id, result)
        return result

    @property
    def cache_stats(self) -> dict:
        return {"hits": self._hits, "misses": self._misses}


class BacktestTwitterPipeline:
    """Queries historical_news with keyword filtering as a Twitter proxy."""

    def __init__(self, db_path: str):
        self._db_path = db_path

    def _conn(self):
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    async def get_signals_for_market(self, keywords: List[str]) -> List[Signal]:
        if not keywords:
            return []

        now = Clock.utcnow()
        window_start = (now - timedelta(minutes=30)).isoformat()
        window_end = now.isoformat()

        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM historical_news WHERE published_at >= ? AND published_at <= ? LIMIT 100",
                (window_start, window_end),
            ).fetchall()

        signals = []
        kw_lower = [kw.lower() for kw in keywords[:5]]
        for row in rows:
            headline_lower = str(row["headline"]).lower()
            if not any(kw in headline_lower for kw in kw_lower):
                continue
            try:
                pub_dt = datetime.fromisoformat(str(row["published_at"]).replace("Z", "+00:00"))
                signals.append(Signal(
                    source="twitter",
                    source_tier="S6",  # GDELT URLs are not real tweets — conservative tier
                    info_type=classify_info_type("S6"),
                    content=row["headline"],
                    credibility=SOURCE_TIER_CREDIBILITY.get("S6", 0.30),
                    author=row["source_domain"],
                    followers=1000,
                    engagement=10,
                    timestamp=pub_dt,
                    headline_only=True,
                ))
                if len(signals) >= 5:
                    break
            except Exception as e:
                log.warning("backtest_twitter_parse_error", error=str(e))
                continue
        return signals
