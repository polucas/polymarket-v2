"""Tests for src.engine.execution – simulate_execution and execute_trade."""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.engine.execution import simulate_execution, execute_trade
from src.models import (
    ExecutionResult,
    Market,
    Portfolio,
    Position,
    TradeCandidate,
    TradeRecord,
)


class TestTakerSlippage:
    """Taker slippage = 0.005 + 0.01 * min(size / max(depth, 1), 1.0)."""

    def test_small_order_slippage(self):
        """size=100, depth=5000 -> 0.005 + 0.01 * (100/5000) = 0.005 + 0.0002 = 0.0052."""
        result = simulate_execution(
            side="BUY_YES",
            price=0.60,
            size_usd=100,
            execution_type="taker",
            orderbook_depth=5000,
        )
        expected_slippage = 0.005 + 0.01 * (100 / 5000)
        assert abs(result.slippage - expected_slippage) < 1e-9
        assert abs(result.slippage - 0.0052) < 1e-9

    def test_large_order_slippage_capped(self):
        """size=10000, depth=5000 -> min(10000/5000, 1.0) = 1.0 -> 0.005 + 0.01 = 0.015."""
        result = simulate_execution(
            side="BUY_YES",
            price=0.60,
            size_usd=10000,
            execution_type="taker",
            orderbook_depth=5000,
        )
        expected_slippage = 0.005 + 0.01 * 1.0
        assert abs(result.slippage - expected_slippage) < 1e-9
        assert abs(result.slippage - 0.015) < 1e-9


class TestTakerExecutedPrice:
    """YES side adds slippage, NO side subtracts slippage."""

    def test_yes_taker_price_increase(self):
        """price=0.60, slippage=~0.01 -> executed_price = 0.61."""
        # Force slippage = 0.01 by choosing size/depth so that
        # 0.005 + 0.01 * min(size/depth, 1) = 0.01
        # => 0.01 * min(size/depth, 1) = 0.005 => size/depth = 0.5
        result = simulate_execution(
            side="BUY_YES",
            price=0.60,
            size_usd=2500,
            execution_type="taker",
            orderbook_depth=5000,
        )
        expected_slippage = 0.005 + 0.01 * 0.5
        assert abs(result.slippage - 0.01) < 1e-9
        assert abs(result.executed_price - 0.61) < 1e-9

    def test_no_taker_price_decrease(self):
        """price=0.60, slippage=0.01 -> executed_price = 0.59."""
        result = simulate_execution(
            side="BUY_NO",
            price=0.60,
            size_usd=2500,
            execution_type="taker",
            orderbook_depth=5000,
        )
        expected_slippage = 0.005 + 0.01 * 0.5
        assert abs(result.slippage - 0.01) < 1e-9
        assert abs(result.executed_price - 0.59) < 1e-9


class TestTakerFill:
    """Taker orders always fill."""

    def test_fill_probability_always_one(self):
        result = simulate_execution(
            side="BUY_YES",
            price=0.60,
            size_usd=100,
            execution_type="taker",
            orderbook_depth=5000,
        )
        assert result.fill_probability == 1.0

    def test_filled_always_true(self):
        result = simulate_execution(
            side="BUY_NO",
            price=0.40,
            size_usd=500,
            execution_type="taker",
            orderbook_depth=1000,
        )
        assert result.filled is True


class TestMakerExecution:
    """Maker: slippage=0, fill_probability = 0.4 + 0.4*(1-abs(price-0.5))."""

    def test_maker_slippage_always_zero(self):
        with patch("src.engine.execution.random.random", return_value=0.0):
            result = simulate_execution(
                side="BUY_YES",
                price=0.60,
                size_usd=100,
                execution_type="maker",
                orderbook_depth=5000,
            )
        assert result.slippage == 0.0
        assert result.executed_price == 0.60

    def test_maker_fill_probability_at_050(self):
        """price=0.50 -> fill_prob = 0.4 + 0.4*(1-0) = 0.80."""
        with patch("src.engine.execution.random.random", return_value=0.0):
            result = simulate_execution(
                side="BUY_YES",
                price=0.50,
                size_usd=100,
                execution_type="maker",
                orderbook_depth=5000,
            )
        assert abs(result.fill_probability - 0.80) < 1e-9

    def test_maker_fill_probability_at_090(self):
        """price=0.90 -> fill_prob = 0.4 + 0.4*(1-0.4) = 0.4 + 0.24 = 0.64."""
        with patch("src.engine.execution.random.random", return_value=0.0):
            result = simulate_execution(
                side="BUY_YES",
                price=0.90,
                size_usd=100,
                execution_type="maker",
                orderbook_depth=5000,
            )
        assert abs(result.fill_probability - 0.64) < 1e-9

    def test_maker_filled_when_random_below_threshold(self):
        """random() = 0.5 < fill_prob=0.80 -> filled=True."""
        with patch("src.engine.execution.random.random", return_value=0.5):
            result = simulate_execution(
                side="BUY_YES",
                price=0.50,
                size_usd=100,
                execution_type="maker",
                orderbook_depth=5000,
            )
        assert result.filled is True

    def test_maker_not_filled_when_random_above_threshold(self):
        """random() = 0.85 >= fill_prob=0.80 -> filled=False."""
        with patch("src.engine.execution.random.random", return_value=0.85):
            result = simulate_execution(
                side="BUY_YES",
                price=0.50,
                size_usd=100,
                execution_type="maker",
                orderbook_depth=5000,
            )
        assert result.filled is False


class TestPriceClamping:
    """Executed price clamped to [0.01, 0.99]."""

    def test_price_clamped_low(self):
        """BUY_NO at price=0.02 with slippage should not go below 0.01."""
        result = simulate_execution(
            side="BUY_NO",
            price=0.02,
            size_usd=10000,
            execution_type="taker",
            orderbook_depth=5000,
        )
        # slippage = 0.015 -> 0.02 - 0.015 = 0.005 -> clamped to 0.01
        assert result.executed_price == 0.01

    def test_price_clamped_high(self):
        """BUY_YES at price=0.98 with slippage should not exceed 0.99."""
        result = simulate_execution(
            side="BUY_YES",
            price=0.98,
            size_usd=10000,
            execution_type="taker",
            orderbook_depth=5000,
        )
        # slippage = 0.015 -> 0.98 + 0.015 = 0.995 -> clamped to 0.99
        assert result.executed_price == 0.99


# ---------------------------------------------------------------------------
# Test 11: Maker fill_probability at price=0.10
# ---------------------------------------------------------------------------


class TestMakerFillProbabilityExtreme:
    """Maker fill probability at extreme prices."""

    def test_maker_fill_probability_at_010(self):
        """Test 11: price=0.10 -> fill_prob = 0.4 + 0.4*(1-|0.10-0.5|) = 0.4 + 0.4*(1-0.4) = 0.4 + 0.24 = 0.64."""
        with patch("src.engine.execution.random.random", return_value=0.0):
            result = simulate_execution(
                side="BUY_YES",
                price=0.10,
                size_usd=100,
                execution_type="maker",
                orderbook_depth=5000,
            )
        assert abs(result.fill_probability - 0.64) < 1e-9


# ---------------------------------------------------------------------------
# Test 14-17: execute_trade tests
# ---------------------------------------------------------------------------


def _make_candidate(**overrides) -> TradeCandidate:
    """Build a TradeCandidate with sensible defaults."""
    market = overrides.pop("market", None) or Market(
        market_id="mkt-test-001",
        question="Will X happen?",
        yes_price=0.60,
        no_price=0.40,
        market_type="political",
    )
    defaults = {
        "market": market,
        "adjusted_probability": 0.73,
        "adjusted_confidence": 0.78,
        "calculated_edge": 0.11,
        "score": 0.05,
        "position_size": 200.0,
        "side": "BUY_YES",
        "resolution_hours": 12.0,
        "fee_rate": 0.02,
        "market_price": 0.60,
        "kelly_fraction_used": 0.25,
        "orderbook_depth": 5000.0,
        "tier": 1,
        "grok_raw_probability": 0.75,
        "grok_raw_confidence": 0.80,
        "grok_reasoning": "test reasoning",
        "grok_signal_types": [],
    }
    defaults.update(overrides)
    return TradeCandidate(**defaults)


class TestExecuteTradePaper:
    """Tests 14-15: execute_trade in paper mode."""

    @pytest.mark.asyncio
    async def test_execute_trade_paper_mode_returns_trade_record(self):
        """Test 14: execute_trade in paper mode calls simulate_execution
        and returns a TradeRecord with correct fields."""
        candidate = _make_candidate()
        portfolio = Portfolio(cash_balance=5000.0, open_positions=[])
        mock_db = AsyncMock()
        mock_client = AsyncMock()

        record = await execute_trade(
            candidate=candidate,
            portfolio=portfolio,
            db=mock_db,
            polymarket_client=mock_client,
            environment="paper",
            experiment_run="test-run",
            model_used="grok-3-fast",
        )

        assert record is not None
        assert isinstance(record, TradeRecord)
        assert record.market_id == "mkt-test-001"
        assert record.action == "BUY_YES"
        assert record.position_size_usd == 200.0
        assert record.market_price_at_decision == 0.60
        assert record.experiment_run == "test-run"
        assert record.model_used == "grok-3-fast"
        assert record.grok_raw_probability == 0.75
        # Verify db.save_trade was called
        mock_db.save_trade.assert_awaited_once()
        # Verify polymarket_client.place_order was NOT called (paper mode)
        mock_client.place_order.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_execute_trade_paper_mode_unfilled_maker_returns_none(self):
        """Test 15: execute_trade paper mode with unfilled maker order returns None."""
        candidate = _make_candidate(tier=2)  # tier 2 -> maker execution
        portfolio = Portfolio(cash_balance=5000.0, open_positions=[])
        mock_db = AsyncMock()
        mock_client = AsyncMock()

        # Force the random value above fill probability so maker order is not filled
        with patch("src.engine.execution.random.random", return_value=0.999):
            record = await execute_trade(
                candidate=candidate,
                portfolio=portfolio,
                db=mock_db,
                polymarket_client=mock_client,
                environment="paper",
                experiment_run="test-run",
            )

        assert record is None
        # db.save_trade should NOT have been called since order was not filled
        mock_db.save_trade.assert_not_awaited()


class TestExecuteTradeLive:
    """Test 16: execute_trade in live mode."""

    @pytest.mark.asyncio
    async def test_execute_trade_live_mode_calls_place_order(self):
        """Test 16: execute_trade live mode calls polymarket_client.place_order."""
        candidate = _make_candidate()
        portfolio = Portfolio(cash_balance=5000.0, open_positions=[])
        mock_db = AsyncMock()
        mock_client = AsyncMock()
        mock_client.place_order.return_value = {"status": "ok"}

        record = await execute_trade(
            candidate=candidate,
            portfolio=portfolio,
            db=mock_db,
            polymarket_client=mock_client,
            environment="live",
            experiment_run="test-run",
        )

        assert record is not None
        mock_client.place_order.assert_awaited_once_with(
            market_id="mkt-test-001",
            side="BUY_YES",
            price=0.60,
            size=200.0,
        )
        mock_db.save_trade.assert_awaited_once()


class TestExecuteTradePortfolioUpdate:
    """Test 17: execute_trade updates portfolio."""

    @pytest.mark.asyncio
    async def test_execute_trade_updates_portfolio(self):
        """Test 17: After execute_trade, cash is decreased by position_size
        and a Position is added to open_positions."""
        candidate = _make_candidate(position_size=200.0)
        portfolio = Portfolio(cash_balance=5000.0, open_positions=[])
        mock_db = AsyncMock()
        mock_client = AsyncMock()

        record = await execute_trade(
            candidate=candidate,
            portfolio=portfolio,
            db=mock_db,
            polymarket_client=mock_client,
            environment="paper",
            experiment_run="test-run",
        )

        assert record is not None
        # Cash should be decreased by position_size
        assert abs(portfolio.cash_balance - 4800.0) < 1e-9
        # A new position should be added
        assert len(portfolio.open_positions) == 1
        pos = portfolio.open_positions[0]
        assert pos.market_id == "mkt-test-001"
        assert pos.side == "BUY_YES"
        assert pos.size_usd == 200.0
        # save_portfolio should have been called
        mock_db.save_portfolio.assert_awaited_once()
