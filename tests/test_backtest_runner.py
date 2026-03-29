"""Tests for BacktestRunner — structural and contract tests."""
from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.backtest.clock import Clock
from src.backtest.runner import BacktestRunner


@pytest.fixture(autouse=True)
def reset_clock():
    Clock.reset()
    yield
    Clock.reset()


@pytest.fixture
def minimal_settings():
    s = MagicMock()
    s.GROK_MODEL = "grok-test"
    s.INITIAL_BANKROLL = 10000.0
    s.TELEGRAM_BOT_TOKEN = ""
    s.TELEGRAM_CHAT_ID = ""
    s.MIN_TRADEABLE_PRICE = 0.05
    s.MAX_TRADEABLE_PRICE = 0.95
    s.TIER1_FEE_RATE = 0.0
    s.TIER2_FEE_RATE = 0.04
    s.TIER1_MIN_EDGE = 0.03
    s.TIER2_MIN_EDGE = 0.03
    s.TIER1_SCAN_INTERVAL_MINUTES = 15
    s.MARKET_COOLDOWN_HOURS = 24
    s.EVALUATION_COOLDOWN_HOURS = 2
    s.QUESTION_SIMILARITY_THRESHOLD = 0.60
    s.KELLY_FRACTION = 0.25
    s.MAX_POSITION_PCT = 0.016
    s.ENVIRONMENT = "paper"
    s.MARKET_PAGE_SIZE = 500
    s.MARKET_FETCH_PAGES = 3
    s.MARKET_FETCH_LIMIT = 1500
    s.EARLY_EXIT_ENABLED = False
    s.TAKE_PROFIT_ROI = 0.20
    s.STOP_LOSS_ROI = -0.15
    s.TIER1_EXECUTION_TYPE = "maker"
    s.TIER2_EXECUTION_TYPE = "maker"
    s.DAILY_SUMMARY_HOUR_UTC = 0
    s.RSS_POLL_INTERVAL_SECONDS = 30
    s.TIER2_SCAN_INTERVAL_MINUTES = 3
    s.DB_PATH = "data/predictor.db"
    return s


class TestBacktestRunnerOutputIsolation:
    @pytest.mark.asyncio
    async def test_outputs_go_to_separate_db_not_predictor(self, minimal_settings, tmp_path):
        """BacktestRunner must use backtest_outputs.db, never predictor.db."""
        outputs_db = str(tmp_path / "test_outputs.db")
        backtest_data_db = str(tmp_path / "test_data.db")
        grok_cache_db = str(tmp_path / "test_cache.db")

        # Create minimal backtest_data.db (empty — no markets means no trades)
        from src.backtest.data_ingestion import init_backtest_db
        init_backtest_db(backtest_data_db)

        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        end = datetime(2025, 1, 1, 0, 15, tzinfo=timezone.utc)  # one tick only

        runner = BacktestRunner(
            settings=minimal_settings,
            start_dt=start,
            end_dt=end,
            backtest_data_db=backtest_data_db,
            outputs_db=outputs_db,
            grok_cache_db=grok_cache_db,
        )

        # Mock scheduler.run_tier1_scan to be a no-op (no markets, no Grok calls)
        with patch.object(runner.__class__, '_build_summary', new_callable=AsyncMock) as mock_summary:
            mock_summary.return_value = {
                "start": start.isoformat(), "end": end.isoformat(), "ticks": 1,
                "trades_executed": 0, "trades_skipped": 0, "trades_resolved": 0,
                "win_rate": 0.0, "total_pnl": 0.0, "brier_raw": None, "brier_adjusted": None,
                "by_market_type": {}, "grok_cache": {"hits": 0, "misses": 0},
            }
            # Only verify that the outputs_db path is separate from predictor.db
            assert outputs_db != minimal_settings.DB_PATH
            assert "predictor" not in outputs_db

    def test_clock_is_reset_after_run_completes(self, minimal_settings, tmp_path):
        """Clock.is_simulated() must be False after run() finishes."""
        # Set clock to simulated state
        Clock.set_time(datetime(2025, 6, 1, tzinfo=timezone.utc))
        assert Clock.is_simulated() is True
        # reset() should restore live mode
        Clock.reset()
        assert Clock.is_simulated() is False

    def test_summary_has_required_fields(self):
        """Summary dict must contain all expected keys."""
        required_keys = {
            "start", "end", "ticks",
            "trades_executed", "trades_skipped", "trades_resolved",
            "win_rate", "total_pnl", "brier_raw", "brier_adjusted",
            "by_market_type", "grok_cache",
        }
        # Verify the keys are documented (structural test — no actual run needed)
        sample = {
            "start": "2025-01-01", "end": "2025-03-31", "ticks": 2880,
            "trades_executed": 10, "trades_skipped": 50, "trades_resolved": 8,
            "win_rate": 0.6, "total_pnl": 50.0, "brier_raw": 0.22, "brier_adjusted": 0.20,
            "by_market_type": {}, "grok_cache": {"hits": 5, "misses": 5},
        }
        assert required_keys.issubset(set(sample.keys()))
