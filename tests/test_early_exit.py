import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from src.engine.resolution import (
    calculate_early_exit_pnl,
    calculate_unrealized_roi,
    check_early_exits,
)
from src.models import TradeRecord


def _make_trade(action="BUY_YES", entry_price=0.50, size=100.0) -> TradeRecord:
    return TradeRecord(
        record_id="test-001",
        experiment_run="exp-001",
        timestamp=datetime.now(timezone.utc),
        model_used="test-model",
        market_id="market-001",
        market_question="Will X happen?",
        market_type="event",
        resolution_window_hours=24.0,
        tier=1,
        grok_raw_probability=0.6,
        grok_raw_confidence=0.7,
        grok_reasoning="test",
        grok_signal_types=[],
        action=action,
        position_size_usd=size,
        market_price_at_decision=entry_price,
    )


class TestCalculateEarlyExitPnl:
    def test_buy_yes_price_increase(self):
        """BUY_YES profits when price increases."""
        trade = _make_trade(action="BUY_YES", entry_price=0.50, size=100)
        pnl = calculate_early_exit_pnl(trade, exit_price=0.70)
        # 100 * (0.70/0.50 - 1) = 100 * 0.40 = 40
        assert abs(pnl - 40.0) < 0.01

    def test_buy_yes_price_decrease(self):
        """BUY_YES loses when price decreases."""
        trade = _make_trade(action="BUY_YES", entry_price=0.50, size=100)
        pnl = calculate_early_exit_pnl(trade, exit_price=0.40)
        # 100 * (0.40/0.50 - 1) = 100 * (-0.20) = -20
        assert abs(pnl - (-20.0)) < 0.01

    def test_buy_no_price_decrease(self):
        """BUY_NO profits when YES price decreases (NO price increases)."""
        trade = _make_trade(action="BUY_NO", entry_price=0.50, size=100)
        pnl = calculate_early_exit_pnl(trade, exit_price=0.30)
        # 100 * ((1-0.30)/(1-0.50) - 1) = 100 * (0.70/0.50 - 1) = 100 * 0.40 = 40
        assert abs(pnl - 40.0) < 0.01

    def test_buy_no_price_increase(self):
        """BUY_NO loses when YES price increases."""
        trade = _make_trade(action="BUY_NO", entry_price=0.50, size=100)
        pnl = calculate_early_exit_pnl(trade, exit_price=0.70)
        # 100 * ((1-0.70)/(1-0.50) - 1) = 100 * (0.30/0.50 - 1) = 100 * (-0.40) = -40
        assert abs(pnl - (-40.0)) < 0.01

    def test_skip_action_returns_zero(self):
        """SKIP trades return 0 PnL."""
        trade = _make_trade(action="SKIP")
        pnl = calculate_early_exit_pnl(trade, exit_price=0.70)
        assert pnl == 0.0


class TestCalculateUnrealizedRoi:
    def test_positive_roi(self):
        trade = _make_trade(action="BUY_YES", entry_price=0.50, size=100)
        roi = calculate_unrealized_roi(trade, current_price=0.60)
        # PnL = 100 * (0.60/0.50 - 1) = 20, ROI = 20/100 = 0.20
        assert abs(roi - 0.20) < 0.01

    def test_negative_roi(self):
        trade = _make_trade(action="BUY_YES", entry_price=0.50, size=100)
        roi = calculate_unrealized_roi(trade, current_price=0.40)
        # PnL = 100 * (0.40/0.50 - 1) = -20, ROI = -20/100 = -0.20
        assert abs(roi - (-0.20)) < 0.01

    def test_zero_position_size(self):
        trade = _make_trade(size=0.0)
        roi = calculate_unrealized_roi(trade, current_price=0.60)
        assert roi == 0.0


class TestCheckEarlyExits:
    @pytest.mark.asyncio
    async def test_take_profit_triggered(self):
        """Trade exits when ROI exceeds take-profit threshold."""
        trade = _make_trade(action="BUY_YES", entry_price=0.50, size=100)

        # Mock market with price at 0.65 (ROI = 30% > 20% threshold)
        mock_market = MagicMock()
        mock_market.resolved = False
        mock_market.yes_price = 0.65

        mock_db = AsyncMock()
        mock_db.get_open_trades = AsyncMock(return_value=[trade])
        mock_db.load_portfolio = AsyncMock(return_value=MagicMock(
            total_pnl=0, cash_balance=9900, open_positions=[
                MagicMock(market_id="market-001", current_value=100)
            ], total_equity=10000, peak_equity=10000, max_drawdown=0))
        mock_db.update_trade = AsyncMock()
        mock_db.save_portfolio = AsyncMock()

        mock_poly = AsyncMock()
        mock_poly.get_market = AsyncMock(return_value=mock_market)

        settings = MagicMock()
        settings.EARLY_EXIT_ENABLED = True
        settings.TAKE_PROFIT_ROI = 0.20
        settings.STOP_LOSS_ROI = -0.15

        await check_early_exits(mock_db, mock_poly, settings)

        mock_db.update_trade.assert_called_once()
        updated_trade = mock_db.update_trade.call_args[0][0]
        assert updated_trade.exit_type == "take_profit"
        assert updated_trade.exit_price == 0.65
        assert updated_trade.pnl > 0

    @pytest.mark.asyncio
    async def test_stop_loss_triggered(self):
        """Trade exits when ROI drops below stop-loss threshold."""
        trade = _make_trade(action="BUY_YES", entry_price=0.50, size=100)

        mock_market = MagicMock()
        mock_market.resolved = False
        mock_market.yes_price = 0.40  # ROI = -20% < -15% threshold

        mock_db = AsyncMock()
        mock_db.get_open_trades = AsyncMock(return_value=[trade])
        mock_db.load_portfolio = AsyncMock(return_value=MagicMock(
            total_pnl=0, cash_balance=9900, open_positions=[
                MagicMock(market_id="market-001", current_value=100)
            ], total_equity=10000, peak_equity=10000, max_drawdown=0))
        mock_db.update_trade = AsyncMock()
        mock_db.save_portfolio = AsyncMock()

        mock_poly = AsyncMock()
        mock_poly.get_market = AsyncMock(return_value=mock_market)

        settings = MagicMock()
        settings.EARLY_EXIT_ENABLED = True
        settings.TAKE_PROFIT_ROI = 0.20
        settings.STOP_LOSS_ROI = -0.15

        await check_early_exits(mock_db, mock_poly, settings)

        mock_db.update_trade.assert_called_once()
        updated_trade = mock_db.update_trade.call_args[0][0]
        assert updated_trade.exit_type == "stop_loss"
        assert updated_trade.pnl < 0

    @pytest.mark.asyncio
    async def test_no_exit_within_thresholds(self):
        """No exit when ROI is between thresholds."""
        trade = _make_trade(action="BUY_YES", entry_price=0.50, size=100)

        mock_market = MagicMock()
        mock_market.resolved = False
        mock_market.yes_price = 0.55  # ROI = 10%, within thresholds

        mock_db = AsyncMock()
        mock_db.get_open_trades = AsyncMock(return_value=[trade])
        mock_db.load_portfolio = AsyncMock(return_value=MagicMock(
            total_pnl=0, cash_balance=9900, open_positions=[], total_equity=10000,
            peak_equity=10000, max_drawdown=0))

        mock_poly = AsyncMock()
        mock_poly.get_market = AsyncMock(return_value=mock_market)

        settings = MagicMock()
        settings.EARLY_EXIT_ENABLED = True
        settings.TAKE_PROFIT_ROI = 0.20
        settings.STOP_LOSS_ROI = -0.15

        await check_early_exits(mock_db, mock_poly, settings)

        mock_db.update_trade.assert_not_called()

    @pytest.mark.asyncio
    async def test_resolved_market_skipped(self):
        """Resolved markets are skipped — let normal resolution handle them."""
        trade = _make_trade(action="BUY_YES", entry_price=0.50, size=100)

        mock_market = MagicMock()
        mock_market.resolved = True
        mock_market.yes_price = 0.90

        mock_db = AsyncMock()
        mock_db.get_open_trades = AsyncMock(return_value=[trade])
        mock_db.load_portfolio = AsyncMock(return_value=MagicMock(
            total_pnl=0, cash_balance=9900, open_positions=[], total_equity=10000,
            peak_equity=10000, max_drawdown=0))

        mock_poly = AsyncMock()
        mock_poly.get_market = AsyncMock(return_value=mock_market)

        settings = MagicMock()
        settings.EARLY_EXIT_ENABLED = True
        settings.TAKE_PROFIT_ROI = 0.20
        settings.STOP_LOSS_ROI = -0.15

        await check_early_exits(mock_db, mock_poly, settings)

        mock_db.update_trade.assert_not_called()

    @pytest.mark.asyncio
    async def test_feature_flag_disabled(self):
        """When EARLY_EXIT_ENABLED is False, no checks are performed."""
        mock_db = AsyncMock()
        mock_poly = AsyncMock()
        settings = MagicMock()
        settings.EARLY_EXIT_ENABLED = False

        await check_early_exits(mock_db, mock_poly, settings)

        mock_db.get_open_trades.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_open_trades(self):
        """No error when there are no open trades."""
        mock_db = AsyncMock()
        mock_db.get_open_trades = AsyncMock(return_value=[])
        mock_poly = AsyncMock()
        settings = MagicMock()
        settings.EARLY_EXIT_ENABLED = True

        await check_early_exits(mock_db, mock_poly, settings)

        mock_db.update_trade.assert_not_called()
