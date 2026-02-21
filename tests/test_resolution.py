"""Tests for src.engine.resolution – calculate_pnl, calculate_hypothetical_pnl,
auto_resolve_trades, and update_unrealized_adverse_moves."""

import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.engine.resolution import (
    calculate_pnl,
    calculate_hypothetical_pnl,
    auto_resolve_trades,
    update_unrealized_adverse_moves,
)
from src.models import Portfolio, Position, TradeRecord


def _make_record(**overrides) -> TradeRecord:
    """Build a TradeRecord with sensible defaults for PnL tests."""
    defaults = {
        "record_id": str(uuid.uuid4()),
        "experiment_run": "test-run",
        "timestamp": datetime.now(timezone.utc),
        "model_used": "grok-3-fast",
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
        "fee_rate": 0.02,
        "calculated_edge": 0.10,
        "action": "BUY_YES",
        "position_size_usd": 200.0,
    }
    defaults.update(overrides)
    return TradeRecord(**defaults)


class TestBuyYesPnl:
    """BUY_YES: win = size*(1-price) - size*fee, lose = -size."""

    def test_buy_yes_outcome_yes_positive_pnl(self):
        """BUY_YES + YES outcome -> profit."""
        record = _make_record(
            action="BUY_YES",
            position_size_usd=200.0,
            market_price_at_decision=0.60,
            fee_rate=0.02,
        )
        pnl = calculate_pnl(record, outcome=True)
        # 200 * (1 - 0.60) - 200 * 0.02 = 80 - 4 = 76
        expected = 200.0 * (1.0 - 0.60) - (200.0 * 0.02)
        assert abs(pnl - expected) < 1e-9
        assert pnl > 0

    def test_buy_yes_outcome_no_full_loss(self):
        """BUY_YES + NO outcome -> lose entire position."""
        record = _make_record(
            action="BUY_YES",
            position_size_usd=200.0,
        )
        pnl = calculate_pnl(record, outcome=False)
        assert pnl == -200.0


class TestBuyNoPnl:
    """BUY_NO: win = size*price - size*fee, lose = -size."""

    def test_buy_no_outcome_no_positive_pnl(self):
        """BUY_NO + NO outcome -> profit."""
        record = _make_record(
            action="BUY_NO",
            position_size_usd=200.0,
            market_price_at_decision=0.60,
            fee_rate=0.02,
        )
        pnl = calculate_pnl(record, outcome=False)
        # 200 * 0.60 - 200 * 0.02 = 120 - 4 = 116
        expected = 200.0 * 0.60 - (200.0 * 0.02)
        assert abs(pnl - expected) < 1e-9
        assert pnl > 0

    def test_buy_no_outcome_yes_full_loss(self):
        """BUY_NO + YES outcome -> lose entire position."""
        record = _make_record(
            action="BUY_NO",
            position_size_usd=200.0,
        )
        pnl = calculate_pnl(record, outcome=True)
        assert pnl == -200.0


class TestEdgeCases:
    """Edge cases for calculate_pnl."""

    def test_unknown_action_returns_zero(self):
        record = _make_record(action="SKIP")
        pnl = calculate_pnl(record, outcome=True)
        assert pnl == 0.0

    def test_zero_fee_rate(self):
        record = _make_record(
            action="BUY_YES",
            position_size_usd=100.0,
            market_price_at_decision=0.50,
            fee_rate=0.0,
        )
        pnl = calculate_pnl(record, outcome=True)
        # 100 * 0.50 - 0 = 50
        assert abs(pnl - 50.0) < 1e-9


class TestHypotheticalPnl:
    """calculate_hypothetical_pnl for skipped trades."""

    def test_hypothetical_no_outcome_returns_zero(self):
        """Unresolved trade -> hypothetical PnL is 0."""
        record = _make_record(actual_outcome=None)
        assert calculate_hypothetical_pnl(record) == 0.0

    def test_hypothetical_with_outcome_delegates_to_calculate_pnl(self):
        """Resolved trade -> uses calculate_pnl with stored outcome."""
        record = _make_record(
            action="BUY_YES",
            position_size_usd=200.0,
            market_price_at_decision=0.60,
            fee_rate=0.02,
            actual_outcome=True,
        )
        result = calculate_hypothetical_pnl(record)
        expected = calculate_pnl(record, True)
        assert abs(result - expected) < 1e-9

    def test_hypothetical_losing_trade(self):
        """Hypothetical for a losing BUY_YES where outcome=False."""
        record = _make_record(
            action="BUY_YES",
            position_size_usd=150.0,
            actual_outcome=False,
        )
        result = calculate_hypothetical_pnl(record)
        assert result == -150.0


# ---------------------------------------------------------------------------
# Test 21: Unresolved market keeps trade open
# ---------------------------------------------------------------------------


class TestAutoResolveUnresolved:
    """Test 21: auto_resolve_trades with unresolved market keeps trade open."""

    @pytest.mark.asyncio
    async def test_unresolved_market_keeps_trade_open(self):
        """Test 21: If the market is not resolved and not crypto_15m past its window,
        auto_resolve_trades should not change the trade."""
        trade = _make_record(
            action="BUY_YES",
            market_type="political",
            position_size_usd=200.0,
            actual_outcome=None,
            pnl=None,
        )

        mock_db = AsyncMock()
        mock_db.get_open_trades.return_value = [trade]
        mock_db.load_portfolio.return_value = Portfolio()

        # Unresolved market
        mock_market = SimpleNamespace(
            resolved=False,
            resolution=None,
            yes_price=0.60,
        )
        mock_client = AsyncMock()
        mock_client.get_market.return_value = mock_market

        await auto_resolve_trades(mock_db, mock_client)

        # update_trade should NOT have been called
        mock_db.update_trade.assert_not_awaited()
        # Trade should remain unresolved
        assert trade.actual_outcome is None
        assert trade.pnl is None


# ---------------------------------------------------------------------------
# Test 22: Brier scores calculated on resolution
# ---------------------------------------------------------------------------


class TestBrierScoreCalculation:
    """Test 22: Brier scores are correctly calculated when a trade resolves."""

    @pytest.mark.asyncio
    async def test_brier_scores_calculated_on_resolution(self):
        """Test 22: brier_raw = (grok_raw_prob - actual)^2,
        brier_adjusted = (adjusted_prob - actual)^2."""
        trade = _make_record(
            action="BUY_YES",
            market_type="political",
            grok_raw_probability=0.75,
            final_adjusted_probability=0.73,
            market_price_at_decision=0.60,
            position_size_usd=200.0,
            fee_rate=0.02,
        )
        # Ensure trade is open (no outcome yet)
        trade.actual_outcome = None
        trade.pnl = None

        mock_db = AsyncMock()
        mock_db.get_open_trades.return_value = [trade]
        mock_db.load_portfolio.return_value = Portfolio(
            cash_balance=4800.0,
            total_equity=5000.0,
            peak_equity=5000.0,
            open_positions=[Position(
                market_id="mkt-001",
                side="BUY_YES",
                entry_price=0.60,
                size_usd=200.0,
                current_value=200.0,
            )],
        )

        # Market resolved YES
        mock_market = SimpleNamespace(
            resolved=True,
            resolution="YES",
            yes_price=1.0,
        )
        mock_client = AsyncMock()
        mock_client.get_market.return_value = mock_market

        await auto_resolve_trades(mock_db, mock_client)

        # Verify brier scores
        # outcome=YES -> actual_val = 1.0
        expected_brier_raw = (0.75 - 1.0) ** 2  # 0.0625
        expected_brier_adjusted = (0.73 - 1.0) ** 2  # 0.0729
        assert trade.actual_outcome is True
        assert abs(trade.brier_score_raw - expected_brier_raw) < 1e-9
        assert abs(trade.brier_score_adjusted - expected_brier_adjusted) < 1e-9
        mock_db.update_trade.assert_awaited_once()


# ---------------------------------------------------------------------------
# Test 23: Voided trade excluded from auto_resolve_trades
# ---------------------------------------------------------------------------


class TestVoidedTradeExcluded:
    """Test 23: Voided trades are excluded by get_open_trades query."""

    @pytest.mark.asyncio
    async def test_voided_trade_excluded_from_auto_resolve(self):
        """Test 23: A voided trade should not be returned by get_open_trades,
        so auto_resolve_trades should not process it.
        (The DB query filters voided=FALSE, so we simulate empty result.)"""
        mock_db = AsyncMock()
        # get_open_trades filters out voided trades at the DB level
        mock_db.get_open_trades.return_value = []

        mock_client = AsyncMock()

        await auto_resolve_trades(mock_db, mock_client)

        # No markets should have been queried
        mock_client.get_market.assert_not_awaited()
        mock_db.update_trade.assert_not_awaited()


# ---------------------------------------------------------------------------
# Tests 25-27: update_unrealized_adverse_moves
# ---------------------------------------------------------------------------


class TestUpdateUnrealizedAdverseMoves:
    """Tests 25-27: update_unrealized_adverse_moves calculates adverse moves correctly."""

    @pytest.mark.asyncio
    async def test_buy_yes_adverse_move(self):
        """Test 25: BUY_YES at 0.60, current price 0.48 -> adverse_move = 0.12."""
        trade = _make_record(
            action="BUY_YES",
            market_price_at_decision=0.60,
            actual_outcome=None,
        )

        mock_db = AsyncMock()
        mock_db.get_open_trades.return_value = [trade]

        mock_market = SimpleNamespace(yes_price=0.48)
        mock_client = AsyncMock()
        mock_client.get_market.return_value = mock_market

        await update_unrealized_adverse_moves(mock_db, mock_client)

        # adverse_move = max(0, 0.60 - 0.48) = 0.12
        # 0.12 > 0.10 threshold -> should update
        assert abs(trade.unrealized_adverse_move - 0.12) < 1e-9
        mock_db.update_trade.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_buy_yes_favorable_move(self):
        """Test 26: BUY_YES at 0.60, current price 0.65 -> adverse_move = 0 (favorable)."""
        trade = _make_record(
            action="BUY_YES",
            market_price_at_decision=0.60,
            actual_outcome=None,
        )

        mock_db = AsyncMock()
        mock_db.get_open_trades.return_value = [trade]

        mock_market = SimpleNamespace(yes_price=0.65)
        mock_client = AsyncMock()
        mock_client.get_market.return_value = mock_market

        await update_unrealized_adverse_moves(mock_db, mock_client)

        # adverse_move = max(0, 0.60 - 0.65) = 0 -> below threshold, not updated
        mock_db.update_trade.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_buy_no_adverse_move(self):
        """Test 27: BUY_NO at 0.40, current price 0.52 -> adverse_move = 0.12."""
        trade = _make_record(
            action="BUY_NO",
            market_price_at_decision=0.40,
            actual_outcome=None,
        )

        mock_db = AsyncMock()
        mock_db.get_open_trades.return_value = [trade]

        mock_market = SimpleNamespace(yes_price=0.52)
        mock_client = AsyncMock()
        mock_client.get_market.return_value = mock_market

        await update_unrealized_adverse_moves(mock_db, mock_client)

        # adverse_move = max(0, 0.52 - 0.40) = 0.12
        # 0.12 > 0.10 threshold -> should update
        assert abs(trade.unrealized_adverse_move - 0.12) < 1e-9
        mock_db.update_trade.assert_awaited_once()
