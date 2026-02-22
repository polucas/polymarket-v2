"""Tests for skip record creation on silent early returns in _process_market."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import MonkModeConfig, Settings
from src.models import Market, Portfolio
from src.scheduler import Scheduler


def _build_scheduler() -> Scheduler:
    settings = MagicMock(spec=Settings)
    settings.TIER1_SCAN_INTERVAL_MINUTES = 15
    settings.TIER2_SCAN_INTERVAL_MINUTES = 3
    settings.TIER1_MIN_EDGE = 0.04
    settings.TIER2_MIN_EDGE = 0.05
    settings.TIER1_FEE_RATE = 0.0
    settings.TIER2_FEE_RATE = 0.04
    settings.KELLY_FRACTION = 0.25
    settings.MAX_POSITION_PCT = 0.08
    settings.ENVIRONMENT = "paper"
    settings.TIER1_DAILY_CAP = 5
    settings.TIER2_DAILY_CAP = 3
    settings.DAILY_LOSS_LIMIT_PCT = 0.05
    settings.WEEKLY_LOSS_LIMIT_PCT = 0.10
    settings.CONSECUTIVE_LOSS_COOLDOWN = 3
    settings.COOLDOWN_DURATION_HOURS = 2.0
    settings.DAILY_API_BUDGET_USD = 8.0
    settings.MAX_TOTAL_EXPOSURE_PCT = 0.30

    db = AsyncMock()
    polymarket = AsyncMock()
    twitter = AsyncMock()
    rss = AsyncMock()
    grok = AsyncMock()
    calibration_mgr = MagicMock()
    market_type_mgr = MagicMock()
    signal_tracker_mgr = MagicMock()

    with patch.object(MonkModeConfig, "from_settings", return_value=MonkModeConfig()):
        scheduler = Scheduler(
            settings=settings, db=db, polymarket=polymarket,
            twitter=twitter, rss=rss, grok=grok,
            calibration_mgr=calibration_mgr,
            market_type_mgr=market_type_mgr,
            signal_tracker_mgr=signal_tracker_mgr,
        )
    return scheduler


def _make_market(**overrides) -> Market:
    defaults = dict(
        market_id="test-market-001",
        question="Will X happen?",
        market_type="political",
        yes_price=0.50,
        no_price=0.50,
        liquidity=10000.0,
        volume_24h=5000.0,
        hours_to_resolution=12.0,
        resolution_time=datetime.now(timezone.utc),
        fee_rate=0.0,
    )
    defaults.update(overrides)
    return Market(**defaults)


class TestGrokFailureSkipRecord:
    """When Grok returns None, a SKIP record with reason 'grok_failed' must be saved."""

    @pytest.mark.asyncio
    async def test_grok_failure_saves_skip_record(self):
        scheduler = _build_scheduler()
        market = _make_market()

        scheduler._market_type_mgr.should_disable.return_value = False
        scheduler._grok.call_grok_with_retry = AsyncMock(return_value=None)
        scheduler._twitter.get_signals_for_market = AsyncMock(return_value=[])
        scheduler._polymarket.get_orderbook = AsyncMock(
            return_value=MagicMock(bids=[], asks=[])
        )

        candidates = []

        with patch("src.scheduler.extract_keywords", return_value=["test"]):
            with patch("src.scheduler.build_grok_context", return_value="ctx"):
                await scheduler._process_market(
                    market=market, rss_signals=[], scan_mode="active",
                    candidates=candidates, all_skips=[],
                    today_trades=[], experiment_run="test-run", tier=1,
                )

        scheduler._db.save_trade.assert_called_once()
        saved = scheduler._db.save_trade.call_args[0][0]
        assert saved.action == "SKIP"
        assert saved.skip_reason == "grok_failed"
        assert len(candidates) == 0


class TestPositionTooSmallSkipRecord:
    """When Kelly sizing produces position < $1, a SKIP record must be saved."""

    @pytest.mark.asyncio
    async def test_tiny_position_saves_skip_record(self):
        scheduler = _build_scheduler()
        market = _make_market()

        scheduler._market_type_mgr.should_disable.return_value = False
        scheduler._grok.call_grok_with_retry = AsyncMock(return_value={
            "estimated_probability": 0.51,
            "confidence": 0.60,
            "reasoning": "test",
            "signal_info_types": [],
        })
        scheduler._twitter.get_signals_for_market = AsyncMock(return_value=[])
        scheduler._polymarket.get_orderbook = AsyncMock(
            return_value=MagicMock(bids=[], asks=[])
        )

        candidates = []

        with patch("src.scheduler.extract_keywords", return_value=["test"]):
            with patch("src.scheduler.build_grok_context", return_value="ctx"):
                with patch("src.scheduler.adjust_prediction", return_value=(0.505, 0.58, 0.0)):
                    with patch("src.scheduler.calculate_edge", return_value=0.05):
                        with patch("src.scheduler.determine_side", return_value="BUY_YES"):
                            with patch("src.scheduler.kelly_size", return_value=0.50):
                                await scheduler._process_market(
                                    market=market, rss_signals=[], scan_mode="active",
                                    candidates=candidates, all_skips=[],
                                    today_trades=[], experiment_run="test-run", tier=1,
                                )

        scheduler._db.save_trade.assert_called_once()
        saved = scheduler._db.save_trade.call_args[0][0]
        assert saved.action == "SKIP"
        assert "position_too_small" in saved.skip_reason
        assert len(candidates) == 0


class TestMarketTypeDisabledSkipRecord:
    """When a market type is disabled by learning, a SKIP record must be saved."""

    @pytest.mark.asyncio
    async def test_disabled_market_type_saves_skip_record(self):
        scheduler = _build_scheduler()
        market = _make_market(market_type="crypto_15min")

        scheduler._market_type_mgr.should_disable.return_value = True

        candidates = []

        await scheduler._process_market(
            market=market, rss_signals=[], scan_mode="active",
            candidates=candidates, all_skips=[],
            today_trades=[], experiment_run="test-run", tier=1,
        )

        scheduler._db.save_trade.assert_called_once()
        saved = scheduler._db.save_trade.call_args[0][0]
        assert saved.action == "SKIP"
        assert saved.skip_reason == "market_type_disabled"
        assert len(candidates) == 0
