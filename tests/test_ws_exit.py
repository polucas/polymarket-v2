"""Tests for RealTimeExitManager with mocked WebSocket."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.engine.ws_exit import RealTimeExitManager
from src.models import Portfolio, TradeRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trade(
    *,
    market_id: str = "market-1",
    action: str = "BUY_YES",
    entry_price: float = 0.50,
    position_size: float = 100.0,
    clob_token_id_yes: str = "token-yes-1",
    clob_token_id_no: str = "token-no-1",
) -> TradeRecord:
    return TradeRecord(
        record_id="rec-1",
        experiment_run="run-1",
        timestamp=datetime.now(timezone.utc),
        model_used="grok-test",
        market_id=market_id,
        market_question="Will X happen?",
        market_type="political",
        resolution_window_hours=24.0,
        tier=1,
        grok_raw_probability=0.65,
        grok_raw_confidence=0.80,
        grok_reasoning="test",
        grok_signal_types=[],
        action=action,
        market_price_at_decision=entry_price,
        position_size_usd=position_size,
        clob_token_id_yes=clob_token_id_yes,
        clob_token_id_no=clob_token_id_no,
    )


def _make_portfolio(cash: float = 9900.0) -> Portfolio:
    return Portfolio(
        cash_balance=cash,
        total_pnl=0.0,
        total_equity=cash,
        peak_equity=cash,
    )


def _build_manager(trades=None, portfolio=None):
    """Build a RealTimeExitManager with mocked dependencies."""
    db = AsyncMock()
    db.get_open_trades.return_value = trades or []
    db.load_portfolio.return_value = portfolio or _make_portfolio()

    polymarket = AsyncMock()

    settings = MagicMock()
    settings.EARLY_EXIT_ENABLED = True
    settings.TAKE_PROFIT_ROI = 0.20
    settings.STOP_LOSS_ROI = -0.15
    settings.ENVIRONMENT = "paper"
    settings.TELEGRAM_BOT_TOKEN = ""
    settings.TELEGRAM_CHAT_ID = ""

    mgr = RealTimeExitManager(db=db, polymarket_client=polymarket, settings=settings)
    return mgr


# ---------------------------------------------------------------------------
# Tests: Position tracking
# ---------------------------------------------------------------------------


class TestRefreshPositions:
    @pytest.mark.asyncio
    async def test_loads_positions_by_token_id(self):
        trade = _make_trade(clob_token_id_yes="tok-yes-abc")
        mgr = _build_manager(trades=[trade])

        await mgr._refresh_positions()

        assert "tok-yes-abc" in mgr._active_positions
        assert mgr._active_positions["tok-yes-abc"] is trade

    @pytest.mark.asyncio
    async def test_skips_trades_without_token_id(self):
        trade = _make_trade(clob_token_id_yes="")
        mgr = _build_manager(trades=[trade])

        await mgr._refresh_positions()

        assert len(mgr._active_positions) == 0

    @pytest.mark.asyncio
    async def test_empty_open_trades(self):
        mgr = _build_manager(trades=[])

        await mgr._refresh_positions()

        assert len(mgr._active_positions) == 0


# ---------------------------------------------------------------------------
# Tests: Message handling — take profit
# ---------------------------------------------------------------------------


class TestTakeProfit:
    @pytest.mark.asyncio
    async def test_take_profit_triggered_on_high_bid(self):
        """Best bid at $0.65 when entry was $0.50 → ROI = 30% → TP fires."""
        trade = _make_trade(entry_price=0.50)
        mgr = _build_manager(trades=[trade], portfolio=_make_portfolio())
        mgr._active_positions = {trade.clob_token_id_yes: trade}

        ws_message = [{
            "event_type": "book",
            "asset_id": trade.clob_token_id_yes,
            "bids": [{"price": "0.65", "size": "100"}],
            "asks": [{"price": "0.66", "size": "100"}],
        }]

        with patch("src.engine.ws_exit.send_alert", new_callable=AsyncMock):
            await mgr._handle_message(ws_message)

        mgr.db.update_trade.assert_called_once()
        updated_trade = mgr.db.update_trade.call_args[0][0]
        assert updated_trade.exit_type == "take_profit"
        assert updated_trade.exit_price == 0.65
        assert updated_trade.pnl is not None
        assert updated_trade.pnl > 0

    @pytest.mark.asyncio
    async def test_take_profit_not_triggered_below_threshold(self):
        """Best bid at $0.55 when entry was $0.50 → ROI = 10% → no TP."""
        trade = _make_trade(entry_price=0.50)
        mgr = _build_manager(trades=[trade])
        mgr._active_positions = {trade.clob_token_id_yes: trade}

        ws_message = [{
            "event_type": "book",
            "asset_id": trade.clob_token_id_yes,
            "bids": [{"price": "0.55", "size": "100"}],
            "asks": [],
        }]

        await mgr._handle_message(ws_message)

        mgr.db.update_trade.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: Message handling — stop loss
# ---------------------------------------------------------------------------


class TestStopLoss:
    @pytest.mark.asyncio
    async def test_stop_loss_triggered_on_low_bid(self):
        """Best bid at $0.40 when entry was $0.50 → ROI = -20% → SL fires."""
        trade = _make_trade(entry_price=0.50)
        mgr = _build_manager(trades=[trade], portfolio=_make_portfolio())
        mgr._active_positions = {trade.clob_token_id_yes: trade}

        ws_message = [{
            "event_type": "book",
            "asset_id": trade.clob_token_id_yes,
            "bids": [{"price": "0.40", "size": "100"}],
            "asks": [],
        }]

        with patch("src.engine.ws_exit.send_alert", new_callable=AsyncMock):
            await mgr._handle_message(ws_message)

        mgr.db.update_trade.assert_called_once()
        updated_trade = mgr.db.update_trade.call_args[0][0]
        assert updated_trade.exit_type == "stop_loss"
        assert updated_trade.exit_price == 0.40
        assert updated_trade.pnl < 0

    @pytest.mark.asyncio
    async def test_stop_loss_not_triggered_above_threshold(self):
        """Best bid at $0.46 when entry was $0.50 → ROI = -8% → no SL."""
        trade = _make_trade(entry_price=0.50)
        mgr = _build_manager(trades=[trade])
        mgr._active_positions = {trade.clob_token_id_yes: trade}

        ws_message = [{
            "event_type": "book",
            "asset_id": trade.clob_token_id_yes,
            "bids": [{"price": "0.46", "size": "100"}],
            "asks": [],
        }]

        await mgr._handle_message(ws_message)

        mgr.db.update_trade.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: BUY_NO handling
# ---------------------------------------------------------------------------


class TestBuyNo:
    @pytest.mark.asyncio
    async def test_buy_no_take_profit(self):
        """BUY_NO with entry $0.50 (NO at $0.50). YES bid drops to $0.30
        → NO value = $0.70 → ROI = (0.70/0.50 - 1) = 40% → TP fires.
        Note: calculate_unrealized_roi takes yes_price as input."""
        trade = _make_trade(action="BUY_NO", entry_price=0.50)
        mgr = _build_manager(trades=[trade], portfolio=_make_portfolio())
        mgr._active_positions = {trade.clob_token_id_yes: trade}

        ws_message = [{
            "event_type": "book",
            "asset_id": trade.clob_token_id_yes,
            "bids": [{"price": "0.30", "size": "100"}],
            "asks": [],
        }]

        with patch("src.engine.ws_exit.send_alert", new_callable=AsyncMock):
            await mgr._handle_message(ws_message)

        mgr.db.update_trade.assert_called_once()
        updated = mgr.db.update_trade.call_args[0][0]
        assert updated.exit_type == "take_profit"
        assert updated.pnl > 0


# ---------------------------------------------------------------------------
# Tests: Double-fire prevention
# ---------------------------------------------------------------------------


class TestDoubleFire:
    @pytest.mark.asyncio
    async def test_position_removed_after_exit(self):
        """After triggering an exit, the position should be removed from tracking."""
        trade = _make_trade(entry_price=0.50)
        mgr = _build_manager(trades=[trade], portfolio=_make_portfolio())
        mgr._active_positions = {trade.clob_token_id_yes: trade}

        ws_message = [{
            "event_type": "book",
            "asset_id": trade.clob_token_id_yes,
            "bids": [{"price": "0.65", "size": "100"}],
            "asks": [],
        }]

        with patch("src.engine.ws_exit.send_alert", new_callable=AsyncMock):
            await mgr._handle_message(ws_message)

        # Position should be gone
        assert trade.clob_token_id_yes not in mgr._active_positions

        # Second message should be ignored
        mgr.db.update_trade.reset_mock()
        await mgr._handle_message(ws_message)
        mgr.db.update_trade.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: Message format edge cases
# ---------------------------------------------------------------------------


class TestMessageEdgeCases:
    @pytest.mark.asyncio
    async def test_ignores_unknown_token_id(self):
        """Messages for tokens we don't track should be silently ignored."""
        mgr = _build_manager()
        mgr._active_positions = {}

        ws_message = [{
            "event_type": "book",
            "asset_id": "unknown-token",
            "bids": [{"price": "0.50", "size": "100"}],
            "asks": [],
        }]

        await mgr._handle_message(ws_message)
        mgr.db.update_trade.assert_not_called()

    @pytest.mark.asyncio
    async def test_ignores_message_without_bids(self):
        """Messages with empty bids should be ignored."""
        trade = _make_trade()
        mgr = _build_manager(trades=[trade])
        mgr._active_positions = {trade.clob_token_id_yes: trade}

        ws_message = [{
            "event_type": "book",
            "asset_id": trade.clob_token_id_yes,
            "bids": [],
            "asks": [{"price": "0.55", "size": "100"}],
        }]

        await mgr._handle_message(ws_message)
        mgr.db.update_trade.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_non_list_message(self):
        """Single dict message (not wrapped in list) should be handled."""
        trade = _make_trade(entry_price=0.50)
        mgr = _build_manager(trades=[trade], portfolio=_make_portfolio())
        mgr._active_positions = {trade.clob_token_id_yes: trade}

        ws_message = {
            "event_type": "book",
            "asset_id": trade.clob_token_id_yes,
            "bids": [{"price": "0.65", "size": "100"}],
            "asks": [],
        }

        with patch("src.engine.ws_exit.send_alert", new_callable=AsyncMock):
            await mgr._handle_message(ws_message)

        mgr.db.update_trade.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: Portfolio update
# ---------------------------------------------------------------------------


class TestPortfolioUpdate:
    @pytest.mark.asyncio
    async def test_portfolio_updated_on_exit(self):
        """Portfolio cash_balance and total_pnl should be updated after exit."""
        trade = _make_trade(entry_price=0.50, position_size=100.0)
        portfolio = _make_portfolio(cash=9900.0)
        mgr = _build_manager(trades=[trade], portfolio=portfolio)
        mgr._active_positions = {trade.clob_token_id_yes: trade}

        ws_message = [{
            "event_type": "book",
            "asset_id": trade.clob_token_id_yes,
            "bids": [{"price": "0.65", "size": "100"}],
            "asks": [],
        }]

        with patch("src.engine.ws_exit.send_alert", new_callable=AsyncMock):
            await mgr._handle_message(ws_message)

        mgr.db.save_portfolio.assert_called_once()
        saved_portfolio = mgr.db.save_portfolio.call_args[0][0]
        # PnL = 100 * (0.65/0.50 - 1) = 30
        assert saved_portfolio.total_pnl == pytest.approx(30.0)
        assert saved_portfolio.cash_balance == pytest.approx(9900.0 + 100.0 + 30.0)


# ---------------------------------------------------------------------------
# Tests: Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_start_disabled_when_early_exit_off(self):
        """If EARLY_EXIT_ENABLED is False, start() should be a no-op."""
        mgr = _build_manager()
        mgr.settings.EARLY_EXIT_ENABLED = False

        await mgr.start()

        assert mgr._task is None
        assert mgr._running is False

    @pytest.mark.asyncio
    async def test_stop_cancels_task(self):
        """stop() should cancel the background task."""
        mgr = _build_manager()
        mgr._running = True

        # Create a real asyncio task that we can cancel
        async def _noop():
            await asyncio.sleep(3600)

        mgr._task = asyncio.create_task(_noop())

        await mgr.stop()

        assert mgr._running is False
        assert mgr._task.cancelled()
