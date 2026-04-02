"""Tests verifying that on_trade_resolved() is called when trades resolve.

These tests cover the fix for the bug where on_trade_resolved() was imported
but never called, leaving calibration_state, market_type_performance, and
signal_trackers tables permanently empty.
"""

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from src.engine.resolution import auto_resolve_trades
from src.models import Portfolio, TradeRecord


def _make_record(**overrides) -> TradeRecord:
    defaults = {
        "record_id": str(uuid.uuid4()),
        "experiment_run": "test-run",
        "timestamp": datetime.now(timezone.utc),
        "model_used": "minimax",
        "market_id": "mkt-001",
        "market_question": "Test?",
        "market_type": "political",
        "resolution_window_hours": 12.0,
        "tier": 1,
        "grok_raw_probability": 0.75,
        "grok_raw_confidence": 0.80,
        "grok_reasoning": "reason",
        "grok_signal_types": [],
        "final_adjusted_probability": 0.73,
        "final_adjusted_confidence": 0.78,
        "market_price_at_decision": 0.60,
        "fee_rate": 0.0,
        "calculated_edge": 0.10,
        "action": "BUY_YES",
        "position_size_usd": 100.0,
        "actual_outcome": None,
        "pnl": None,
        "brier_score_raw": None,
        "brier_score_adjusted": None,
        "resolved_at": None,
        "voided": False,
    }
    defaults.update(overrides)
    return TradeRecord(**defaults)


class TestAutoResolveReturnsNewlyResolved:
    """auto_resolve_trades() must return the list of trades resolved this tick."""

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_open_trades(self):
        mock_db = AsyncMock()
        mock_db.get_open_trades.return_value = []
        result = await auto_resolve_trades(mock_db, AsyncMock())
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_market_unresolved(self):
        trade = _make_record()
        mock_db = AsyncMock()
        mock_db.get_open_trades.return_value = [trade]
        mock_db.load_portfolio.return_value = Portfolio()

        mock_market = SimpleNamespace(resolved=False, resolution=None, yes_price=0.60)
        mock_client = AsyncMock()
        mock_client.get_market.return_value = mock_market

        result = await auto_resolve_trades(mock_db, mock_client)
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_resolved_trade_on_resolution(self):
        trade = _make_record(
            grok_raw_probability=0.75,
            final_adjusted_probability=0.73,
            market_price_at_decision=0.60,
            position_size_usd=100.0,
        )
        mock_db = AsyncMock()
        mock_db.get_open_trades.return_value = [trade]
        mock_db.load_portfolio.return_value = Portfolio()

        mock_market = SimpleNamespace(resolved=True, resolution="YES", yes_price=1.0)
        mock_client = AsyncMock()
        mock_client.get_market.return_value = mock_market

        result = await auto_resolve_trades(mock_db, mock_client)

        assert len(result) == 1
        assert result[0] is trade
        assert result[0].actual_outcome is True
        assert result[0].brier_score_raw is not None

    @pytest.mark.asyncio
    async def test_returns_multiple_resolved_trades(self):
        trade1 = _make_record(market_id="mkt-001")
        trade2 = _make_record(market_id="mkt-002")
        mock_db = AsyncMock()
        mock_db.get_open_trades.return_value = [trade1, trade2]
        mock_db.load_portfolio.return_value = Portfolio()

        mock_market = SimpleNamespace(resolved=True, resolution="YES", yes_price=1.0)
        mock_client = AsyncMock()
        mock_client.get_market.return_value = mock_market

        result = await auto_resolve_trades(mock_db, mock_client)
        assert len(result) == 2


class TestSchedulerAutoResolveCallsLearning:
    """_auto_resolve() must call on_trade_resolved() for each resolved trade."""

    @pytest.mark.asyncio
    async def test_on_trade_resolved_called_for_resolved_trade(self):
        from src.scheduler import Scheduler
        from unittest.mock import AsyncMock, MagicMock, patch

        trade = _make_record(actual_outcome=True, brier_score_raw=0.0625)

        mock_settings = MagicMock()
        mock_settings.EARLY_EXIT_ENABLED = False

        scheduler = Scheduler.__new__(Scheduler)
        scheduler._settings = mock_settings
        scheduler._db = AsyncMock()
        scheduler._polymarket = AsyncMock()
        scheduler._calibration_mgr = MagicMock()
        scheduler._market_type_mgr = MagicMock()
        scheduler._signal_tracker_mgr = MagicMock()

        with patch("src.scheduler.check_early_exits", new=AsyncMock()):
            with patch("src.scheduler.auto_resolve_trades", new=AsyncMock(return_value=[trade])) as mock_resolve:
                with patch("src.scheduler.on_trade_resolved", new=AsyncMock()) as mock_learn:
                    await scheduler._auto_resolve()

        mock_learn.assert_awaited_once_with(
            record=trade,
            calibration_mgr=scheduler._calibration_mgr,
            market_type_mgr=scheduler._market_type_mgr,
            signal_tracker_mgr=scheduler._signal_tracker_mgr,
            db=scheduler._db,
        )

    @pytest.mark.asyncio
    async def test_on_trade_resolved_not_called_when_nothing_resolved(self):
        from src.scheduler import Scheduler

        mock_settings = MagicMock()
        mock_settings.EARLY_EXIT_ENABLED = False

        scheduler = Scheduler.__new__(Scheduler)
        scheduler._settings = mock_settings
        scheduler._db = AsyncMock()
        scheduler._polymarket = AsyncMock()
        scheduler._calibration_mgr = MagicMock()
        scheduler._market_type_mgr = MagicMock()
        scheduler._signal_tracker_mgr = MagicMock()

        with patch("src.scheduler.check_early_exits", new=AsyncMock()):
            with patch("src.scheduler.auto_resolve_trades", new=AsyncMock(return_value=[])):
                with patch("src.scheduler.on_trade_resolved", new=AsyncMock()) as mock_learn:
                    await scheduler._auto_resolve()

        mock_learn.assert_not_awaited()
