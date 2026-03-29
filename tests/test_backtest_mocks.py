"""Tests for BacktestPolymarketClient, BacktestRSSPipeline, BacktestGrokClient, BacktestTwitterPipeline."""
from __future__ import annotations

import json
import sqlite3
import tempfile
from datetime import datetime, timezone, timedelta
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.backtest.clock import Clock
from src.backtest.mocks import (
    BacktestPolymarketClient,
    BacktestRSSPipeline,
    BacktestGrokClient,
    BacktestTwitterPipeline,
)
from src.models import OrderBook, OrderBookLevel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_clock():
    Clock.reset()
    yield
    Clock.reset()


@pytest.fixture
def backtest_db():
    """In-memory-backed temp SQLite file with test data."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE historical_markets (
            market_id TEXT PRIMARY KEY,
            question TEXT,
            market_type TEXT,
            created_at TEXT,
            resolution_datetime TEXT,
            actual_outcome TEXT,
            baseline_price REAL
        )
    """)
    conn.execute("""
        CREATE TABLE historical_news (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            published_at TEXT,
            source_domain TEXT,
            headline TEXT
        )
    """)

    # Markets
    conn.execute("INSERT INTO historical_markets VALUES (?,?,?,?,?,?,?)", (
        "mkt-1", "Will Bitcoin reach $100K?", "crypto_15m",
        "2025-01-01T00:00:00+00:00",
        "2025-06-01T12:00:00+00:00",
        "YES", 0.60,
    ))
    conn.execute("INSERT INTO historical_markets VALUES (?,?,?,?,?,?,?)", (
        "mkt-extreme", "Will this unlikely event happen?", "political",
        "2025-01-01T00:00:00+00:00",
        "2025-06-01T12:00:00+00:00",
        "NO", 0.02,  # below MIN_TRADEABLE_PRICE — should be filtered
    ))
    conn.execute("INSERT INTO historical_markets VALUES (?,?,?,?,?,?,?)", (
        "mkt-future", "Will something happen in 2027?", "economic",
        "2026-01-01T00:00:00+00:00",  # created_at in the future
        "2027-06-01T12:00:00+00:00",
        None, 0.50,
    ))

    # News
    now_str = datetime(2025, 3, 15, 10, 0, 0, tzinfo=timezone.utc).isoformat()
    old_str = datetime(2025, 3, 15, 9, 0, 0, tzinfo=timezone.utc).isoformat()  # 60 min ago — outside 30-min window
    conn.execute("INSERT INTO historical_news (published_at, source_domain, headline) VALUES (?,?,?)",
                 (now_str, "reuters.com", "https://reuters.com/bitcoin-price-update"))
    conn.execute("INSERT INTO historical_news (published_at, source_domain, headline) VALUES (?,?,?)",
                 (old_str, "apnews.com", "https://apnews.com/old-news-story"))

    conn.commit()
    conn.close()
    yield db_path


@pytest.fixture
def mock_settings():
    s = MagicMock()
    s.MIN_TRADEABLE_PRICE = 0.05
    s.MAX_TRADEABLE_PRICE = 0.95
    s.TIER1_FEE_RATE = 0.0
    return s


# ---------------------------------------------------------------------------
# BacktestPolymarketClient
# ---------------------------------------------------------------------------

class TestBacktestPolymarketClient:
    @pytest.mark.asyncio
    async def test_returns_markets_in_valid_window(self, backtest_db, mock_settings):
        """mkt-1 should appear when clock is within 168h of resolution_datetime (2025-06-01).
        Set clock to 2025-05-29 so hours_to_resolution ≈ 60h, safely inside the 0.25h-168h window.
        """
        Clock.set_time(datetime(2025, 5, 29, 0, 0, 0, tzinfo=timezone.utc))
        client = BacktestPolymarketClient(mock_settings, backtest_db)
        markets = await client.get_active_markets(tier=1, tier1_fee_rate=0.0, tier2_fee_rate=0.04)
        ids = [m.market_id for m in markets]
        assert "mkt-1" in ids

    @pytest.mark.asyncio
    async def test_excludes_market_before_created_at(self, backtest_db, mock_settings):
        """mkt-future has created_at=2026-01-01 so it must not appear in 2025."""
        Clock.set_time(datetime(2025, 6, 1, 0, 0, 0, tzinfo=timezone.utc))
        client = BacktestPolymarketClient(mock_settings, backtest_db)
        markets = await client.get_active_markets(tier=1, tier1_fee_rate=0.0, tier2_fee_rate=0.04)
        ids = [m.market_id for m in markets]
        assert "mkt-future" not in ids

    @pytest.mark.asyncio
    async def test_excludes_extreme_price_market(self, backtest_db, mock_settings):
        """mkt-extreme has baseline_price=0.02 < MIN_TRADEABLE_PRICE=0.05 — filtered."""
        Clock.set_time(datetime(2025, 3, 15, tzinfo=timezone.utc))
        client = BacktestPolymarketClient(mock_settings, backtest_db)
        markets = await client.get_active_markets(tier=1, tier1_fee_rate=0.0, tier2_fee_rate=0.04)
        ids = [m.market_id for m in markets]
        assert "mkt-extreme" not in ids

    @pytest.mark.asyncio
    async def test_orderbook_has_3_tiered_levels(self, backtest_db, mock_settings):
        """Synthetic orderbook: 3 ask levels with sizes [50, 150, 500]."""
        Clock.set_time(datetime(2025, 3, 15, tzinfo=timezone.utc))
        client = BacktestPolymarketClient(mock_settings, backtest_db)
        ob = await client.get_orderbook("mkt-1", "mkt-1")
        assert isinstance(ob, OrderBook)
        assert len(ob.asks) == 3
        assert len(ob.bids) == 3
        # Ask sizes: 50, 150, 500
        ask_sizes = [a.size for a in ob.asks]
        assert ask_sizes == [50.0, 150.0, 500.0]
        # Ask prices strictly ascending
        ask_prices = [a.price for a in ob.asks]
        assert ask_prices == sorted(ask_prices)
        # Bid prices strictly descending
        bid_prices = [b.price for b in ob.bids]
        assert bid_prices == sorted(bid_prices, reverse=True)

    @pytest.mark.asyncio
    async def test_orderbook_centered_on_baseline(self, backtest_db, mock_settings):
        """Best ask = baseline + 0.02, best bid = baseline - 0.02."""
        Clock.set_time(datetime(2025, 3, 15, tzinfo=timezone.utc))
        client = BacktestPolymarketClient(mock_settings, backtest_db)
        ob = await client.get_orderbook("mkt-1", "mkt-1")
        assert abs(ob.best_ask - (0.60 + 0.02)) < 0.001
        assert abs(ob.best_bid - (0.60 - 0.02)) < 0.001

    @pytest.mark.asyncio
    async def test_get_market_not_resolved_before_resolution_time(self, backtest_db, mock_settings):
        Clock.set_time(datetime(2025, 3, 15, tzinfo=timezone.utc))
        client = BacktestPolymarketClient(mock_settings, backtest_db)
        market = await client.get_market("mkt-1")
        assert market is not None
        assert market.resolved is False

    @pytest.mark.asyncio
    async def test_get_market_resolved_after_resolution_time(self, backtest_db, mock_settings):
        """After resolution_datetime, market appears resolved with actual_outcome."""
        Clock.set_time(datetime(2025, 7, 1, tzinfo=timezone.utc))  # after 2025-06-01
        client = BacktestPolymarketClient(mock_settings, backtest_db)
        market = await client.get_market("mkt-1")
        assert market.resolved is True
        assert market.resolution == "YES"


# ---------------------------------------------------------------------------
# BacktestRSSPipeline
# ---------------------------------------------------------------------------

class TestBacktestRSSPipeline:
    def test_consume_signals_in_window(self, backtest_db):
        """News published at 10:00 should appear when clock is at 10:15 (within 30-min window)."""
        Clock.set_time(datetime(2025, 3, 15, 10, 15, tzinfo=timezone.utc))
        pipeline = BacktestRSSPipeline(backtest_db)
        signals = pipeline.consume_signals()
        contents = [s.content for s in signals]
        assert any("reuters.com/bitcoin" in c for c in contents)

    def test_consume_signals_excludes_old_news(self, backtest_db):
        """News published at 09:00 should NOT appear when clock is at 10:15 (61 min ago > 30 min window)."""
        Clock.set_time(datetime(2025, 3, 15, 10, 15, tzinfo=timezone.utc))
        pipeline = BacktestRSSPipeline(backtest_db)
        signals = pipeline.consume_signals()
        contents = [s.content for s in signals]
        assert not any("apnews.com/old-news" in c for c in contents)

    def test_poll_and_accumulate_is_noop(self, backtest_db):
        """poll_and_accumulate() should return without error."""
        pipeline = BacktestRSSPipeline(backtest_db)
        asyncio_loop = __import__('asyncio')
        asyncio_loop.get_event_loop().run_until_complete(pipeline.poll_and_accumulate())


# ---------------------------------------------------------------------------
# BacktestGrokClient
# ---------------------------------------------------------------------------

class TestBacktestGrokClient:
    @pytest.mark.asyncio
    async def test_cache_miss_calls_real_grok(self):
        """First call with new context calls real API."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            cache_db = f.name

        real_grok = MagicMock()
        real_grok.call_grok_with_retry = AsyncMock(return_value={
            "estimated_probability": 0.70,
            "confidence": 0.80,
            "reasoning": "test",
        })
        client = BacktestGrokClient(real_grok, cache_db)
        result = await client.call_grok_with_retry("unique context string", "mkt-test")
        assert result is not None
        assert result["estimated_probability"] == 0.70
        real_grok.call_grok_with_retry.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_hit_returns_cached_result(self):
        """Second call with same context returns cached result without calling real API."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            cache_db = f.name

        real_grok = MagicMock()
        real_grok.call_grok_with_retry = AsyncMock(return_value={
            "estimated_probability": 0.65,
            "confidence": 0.75,
            "reasoning": "cached test",
        })
        client = BacktestGrokClient(real_grok, cache_db)

        context = "same context for two calls"
        result1 = await client.call_grok_with_retry(context, "mkt-a")
        result2 = await client.call_grok_with_retry(context, "mkt-a")

        assert result1 == result2
        real_grok.call_grok_with_retry.assert_called_once()  # not twice
        assert client.cache_stats["hits"] == 1
        assert client.cache_stats["misses"] == 1

    @pytest.mark.asyncio
    async def test_different_contexts_are_separate_cache_entries(self):
        """Different prompt strings produce separate cache entries, both call real API."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            cache_db = f.name

        real_grok = MagicMock()
        real_grok.call_grok_with_retry = AsyncMock(return_value={"estimated_probability": 0.5, "confidence": 0.6, "reasoning": ""})
        client = BacktestGrokClient(real_grok, cache_db)

        await client.call_grok_with_retry("context A", "mkt-a")
        await client.call_grok_with_retry("context B", "mkt-b")

        assert real_grok.call_grok_with_retry.call_count == 2
        assert client.cache_stats["misses"] == 2
        assert client.cache_stats["hits"] == 0


# ---------------------------------------------------------------------------
# BacktestTwitterPipeline
# ---------------------------------------------------------------------------

class TestBacktestTwitterPipeline:
    def test_keyword_match_returns_signals(self, backtest_db):
        """Headlines containing 'bitcoin' match keyword 'bitcoin'."""
        Clock.set_time(datetime(2025, 3, 15, 10, 15, tzinfo=timezone.utc))
        pipeline = BacktestTwitterPipeline(backtest_db)

        import asyncio
        signals = asyncio.get_event_loop().run_until_complete(
            pipeline.get_signals_for_market(["bitcoin"])
        )
        assert len(signals) >= 1
        assert all(s.source == "twitter" for s in signals)
        assert all(s.source_tier == "S6" for s in signals)

    def test_no_keyword_match_returns_empty(self, backtest_db):
        """Keywords with no match return an empty list."""
        Clock.set_time(datetime(2025, 3, 15, 10, 15, tzinfo=timezone.utc))
        pipeline = BacktestTwitterPipeline(backtest_db)

        import asyncio
        signals = asyncio.get_event_loop().run_until_complete(
            pipeline.get_signals_for_market(["completely_unrelated_keyword_xyz"])
        )
        assert signals == []

    def test_empty_keywords_returns_empty(self, backtest_db):
        Clock.set_time(datetime(2025, 3, 15, 10, 15, tzinfo=timezone.utc))
        pipeline = BacktestTwitterPipeline(backtest_db)

        import asyncio
        signals = asyncio.get_event_loop().run_until_complete(
            pipeline.get_signals_for_market([])
        )
        assert signals == []
