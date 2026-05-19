"""Tests for src.engine.resolution – calculate_pnl, calculate_hypothetical_pnl,
auto_resolve_trades, and update_unrealized_adverse_moves."""

import logging
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import structlog
from structlog.testing import capture_logs

from src.engine.resolution import (
    calculate_pnl,
    calculate_hypothetical_pnl,
    auto_resolve_trades,
    check_early_exits,
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


# ---------------------------------------------------------------------------
# Tests 28-30: unrealized_adverse_triggered log on threshold crossing
# ---------------------------------------------------------------------------


class TestUnrealizedAdverseTriggeredLog:
    """Tests 28-30: structured log emitted only on fresh threshold crossing."""

    @pytest.mark.asyncio
    async def test_fresh_crossing_emits_log(self):
        """Test 28: prior adverse 0.05 (below threshold), current 15% adverse
        -> log 'unrealized_adverse_triggered' emitted once with correct fields."""
        trade = _make_record(
            action="BUY_YES",
            market_price_at_decision=0.50,
            actual_outcome=None,
        )
        trade.unrealized_adverse_move = 0.05  # below threshold

        mock_db = AsyncMock()
        mock_db.get_open_trades.return_value = [trade]

        # current_price=0.35 -> adverse_move = max(0, 0.50-0.35) = 0.15 > 0.10
        mock_market = SimpleNamespace(yes_price=0.35)
        mock_client = AsyncMock()
        mock_client.get_market.return_value = mock_market

        with capture_logs() as log_output:
            await update_unrealized_adverse_moves(mock_db, mock_client)

        triggered = [e for e in log_output if e.get("event") == "unrealized_adverse_triggered"]
        assert len(triggered) == 1
        rec = triggered[0]
        assert abs(rec["adverse_pct"] - 0.15) < 1e-4
        assert rec["threshold"] == 0.10
        assert rec["market_id"] == trade.market_id

    @pytest.mark.asyncio
    async def test_small_adverse_no_log(self):
        """Test 29: 2% adverse move (below 10% threshold) -> no log emitted."""
        trade = _make_record(
            action="BUY_YES",
            market_price_at_decision=0.50,
            actual_outcome=None,
        )
        trade.unrealized_adverse_move = 0.0

        mock_db = AsyncMock()
        mock_db.get_open_trades.return_value = [trade]

        # current_price=0.48 -> adverse_move = max(0, 0.50-0.48) = 0.02 < 0.10
        mock_market = SimpleNamespace(yes_price=0.48)
        mock_client = AsyncMock()
        mock_client.get_market.return_value = mock_market

        with capture_logs() as log_output:
            await update_unrealized_adverse_moves(mock_db, mock_client)

        triggered = [e for e in log_output if e.get("event") == "unrealized_adverse_triggered"]
        assert len(triggered) == 0

    @pytest.mark.asyncio
    async def test_already_above_threshold_no_new_log(self):
        """Test 30: prior adverse 0.15 (already above threshold), current 20% adverse
        -> no new log emitted (not a fresh crossing)."""
        trade = _make_record(
            action="BUY_YES",
            market_price_at_decision=0.50,
            actual_outcome=None,
        )
        trade.unrealized_adverse_move = 0.15  # already above 0.10

        mock_db = AsyncMock()
        mock_db.get_open_trades.return_value = [trade]

        # current_price=0.30 -> adverse_move = max(0, 0.50-0.30) = 0.20 > 0.10
        # but prior was already > 0.10 -> no fresh crossing
        mock_market = SimpleNamespace(yes_price=0.30)
        mock_client = AsyncMock()
        mock_client.get_market.return_value = mock_market

        with capture_logs() as log_output:
            await update_unrealized_adverse_moves(mock_db, mock_client)

        triggered = [e for e in log_output if e.get("event") == "unrealized_adverse_triggered"]
        assert len(triggered) == 0


# ---------------------------------------------------------------------------
# Test 31: resolution_skipped_unresolved log for non-crypto unresolved market
# ---------------------------------------------------------------------------


class TestResolutionSkippedUnresolvedLog:
    """Test 31: resolution_skipped_unresolved log emitted for political trade past resolution_datetime."""

    @pytest.mark.asyncio
    async def test_skipped_unresolved_log_emitted(self):
        """Political trade 48h past resolution_datetime; market not resolved.
        Expect log 'resolution_skipped_unresolved' once with correct fields.
        update_trade must NOT be called. actual_outcome stays None."""
        now = datetime(2026, 4, 19, 12, 0, 0, tzinfo=timezone.utc)
        res_dt = now - timedelta(hours=48)

        trade = _make_record(
            market_type="political",
            action="BUY_YES",
            position_size_usd=100.0,
            actual_outcome=None,
            pnl=None,
        )
        trade.resolution_datetime = res_dt

        mock_db = AsyncMock()
        mock_db.get_open_trades.return_value = [trade]
        mock_db.load_portfolio.return_value = Portfolio()

        mock_market = SimpleNamespace(
            resolved=False,
            resolution=None,
            yes_price=0.8,
        )
        mock_client = AsyncMock()
        mock_client.get_market.return_value = mock_market

        with patch("src.engine.resolution.Clock") as mock_clock, \
             capture_logs() as log_output:
            mock_clock.utcnow.return_value = now
            await auto_resolve_trades(mock_db, mock_client)

        skipped = [e for e in log_output if e.get("event") == "resolution_skipped_unresolved"]
        assert len(skipped) == 1, f"Expected 1 log entry, got {len(skipped)}: {log_output}"
        rec = skipped[0]
        assert rec["market_id"] == trade.market_id
        assert rec["market_type"] == "political"
        assert rec["polymarket_resolved_flag"] is False
        assert abs(rec["polymarket_yes_price"] - 0.8) < 1e-9
        assert abs(rec["hours_past_resolution"] - 48.0) < 0.1

        mock_db.update_trade.assert_not_awaited()
        assert trade.actual_outcome is None


# ---------------------------------------------------------------------------
# Test 32: resolution_fallback_crypto_price log for crypto_15m past window
# ---------------------------------------------------------------------------


class TestResolutionFallbackCryptoPriceLog:
    """Test 32: resolution_fallback_crypto_price log emitted for crypto_15m past window."""

    @pytest.mark.asyncio
    async def test_fallback_crypto_price_log_emitted(self):
        """crypto_15m trade 45min old with 0.25h window (15min) -> past window.
        Market unresolved, yes_price=0.7 -> inferred YES.
        Expect log 'resolution_fallback_crypto_price' with inferred_outcome=1.
        actual_outcome=True written."""
        now = datetime(2026, 4, 19, 12, 0, 0, tzinfo=timezone.utc)
        trade_ts = now - timedelta(minutes=45)

        trade = _make_record(
            market_type="crypto_15m",
            action="BUY_YES",
            position_size_usd=100.0,
            resolution_window_hours=0.25,
            actual_outcome=None,
            pnl=None,
        )
        trade.timestamp = trade_ts

        mock_db = AsyncMock()
        mock_db.get_open_trades.return_value = [trade]
        mock_db.load_portfolio.return_value = Portfolio()
        mock_db.save_portfolio.return_value = None

        mock_market = SimpleNamespace(
            resolved=False,
            resolution=None,
            yes_price=0.7,
        )
        mock_client = AsyncMock()
        mock_client.get_market.return_value = mock_market

        with patch("src.engine.resolution.Clock") as mock_clock, \
             capture_logs() as log_output:
            mock_clock.utcnow.return_value = now
            await auto_resolve_trades(mock_db, mock_client)

        fallback = [e for e in log_output if e.get("event") == "resolution_fallback_crypto_price"]
        assert len(fallback) == 1, f"Expected 1 log entry, got {len(fallback)}: {log_output}"
        rec = fallback[0]
        assert rec["market_id"] == trade.market_id
        assert rec["inferred_outcome"] == 1
        assert abs(rec["polymarket_yes_price"] - 0.7) < 1e-9
        # hours_past_resolution: 45min - 15min window = 30min = 0.5h
        assert abs(rec["hours_past_resolution"] - 0.5) < 0.05

        assert trade.actual_outcome is True
        mock_db.update_trade.assert_awaited_once()


# ---------------------------------------------------------------------------
# Dual-label tests: check_early_exits writes trade_profitable + pnl_brier_*
# ---------------------------------------------------------------------------


def _make_settings_mock(take_profit=0.20, stop_loss=-0.15, enabled=True):
    s = MagicMock()
    s.EARLY_EXIT_ENABLED = enabled
    s.TAKE_PROFIT_ROI = take_profit
    s.STOP_LOSS_ROI = stop_loss
    return s


class TestCheckEarlyExitsDualLabel:
    """check_early_exits writes trade_profitable and pnl Brier scores on exit."""

    @pytest.mark.asyncio
    async def test_check_early_exits_writes_trade_profitable_tp(self):
        """TP threshold crossed -> trade_profitable=1, pnl_brier_raw/adjusted set."""
        # entry 0.50, current 0.65 → ROI ~30% → TP
        trade = _make_record(
            action="BUY_YES",
            market_price_at_decision=0.50,
            position_size_usd=100.0,
            grok_raw_probability=0.70,
            final_adjusted_probability=0.68,
            fee_rate=0.0,
            actual_outcome=None,
            pnl=None,
        )

        mock_db = AsyncMock()
        mock_db.get_open_trades.return_value = [trade]
        mock_db.load_portfolio.return_value = Portfolio(
            cash_balance=9900.0, total_equity=10000.0, peak_equity=10000.0
        )

        mock_market = SimpleNamespace(resolved=False, resolution=None, yes_price=0.65)
        mock_client = AsyncMock()
        mock_client.get_market.return_value = mock_market

        settings = _make_settings_mock()

        await check_early_exits(mock_db, mock_client, settings)

        mock_db.update_trade.assert_awaited_once()
        assert trade.exit_type == "take_profit"
        assert trade.pnl > 0
        assert trade.trade_profitable == 1
        assert trade.pnl_brier_raw is not None
        assert trade.pnl_brier_adjusted is not None
        assert 0.0 <= trade.pnl_brier_raw <= 1.0
        assert 0.0 <= trade.pnl_brier_adjusted <= 1.0
        # actual_outcome must NOT be set (event hasn't resolved)
        assert trade.actual_outcome is None

    @pytest.mark.asyncio
    async def test_check_early_exits_writes_trade_profitable_sl(self):
        """SL threshold crossed -> trade_profitable=0, pnl_brier_raw/adjusted set."""
        # entry 0.50, current 0.40 → ROI -20% → SL
        trade = _make_record(
            action="BUY_YES",
            market_price_at_decision=0.50,
            position_size_usd=100.0,
            grok_raw_probability=0.70,
            final_adjusted_probability=0.68,
            fee_rate=0.0,
            actual_outcome=None,
            pnl=None,
        )

        mock_db = AsyncMock()
        mock_db.get_open_trades.return_value = [trade]
        mock_db.load_portfolio.return_value = Portfolio(
            cash_balance=9900.0, total_equity=10000.0, peak_equity=10000.0
        )

        mock_market = SimpleNamespace(resolved=False, resolution=None, yes_price=0.40)
        mock_client = AsyncMock()
        mock_client.get_market.return_value = mock_market

        settings = _make_settings_mock()

        await check_early_exits(mock_db, mock_client, settings)

        mock_db.update_trade.assert_awaited_once()
        assert trade.exit_type == "stop_loss"
        assert trade.pnl < 0
        assert trade.trade_profitable == 0
        assert trade.pnl_brier_raw is not None
        assert trade.pnl_brier_adjusted is not None
        assert 0.0 <= trade.pnl_brier_raw <= 1.0
        assert 0.0 <= trade.pnl_brier_adjusted <= 1.0
        assert trade.actual_outcome is None


class TestAutoResolveDualLabel:
    """auto_resolve_trades writes trade_profitable + pnl Brier scores on natural resolution."""

    @pytest.mark.asyncio
    async def test_auto_resolve_writes_trade_profitable(self):
        """Naturally resolved trade gets trade_profitable from pnl sign, plus pnl_brier_*."""
        trade = _make_record(
            action="BUY_YES",
            market_type="political",
            grok_raw_probability=0.75,
            final_adjusted_probability=0.73,
            market_price_at_decision=0.60,
            position_size_usd=200.0,
            fee_rate=0.02,
            actual_outcome=None,
            pnl=None,
        )

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

        # Market resolved YES → pnl > 0 → trade_profitable=1
        mock_market = SimpleNamespace(resolved=True, resolution="YES", yes_price=1.0)
        mock_client = AsyncMock()
        mock_client.get_market.return_value = mock_market

        await auto_resolve_trades(mock_db, mock_client)

        mock_db.update_trade.assert_awaited_once()
        assert trade.actual_outcome is True
        assert trade.pnl is not None
        assert trade.pnl > 0
        assert trade.trade_profitable == 1
        assert trade.pnl_brier_raw is not None
        assert trade.pnl_brier_adjusted is not None
        assert 0.0 <= trade.pnl_brier_raw <= 1.0
        assert 0.0 <= trade.pnl_brier_adjusted <= 1.0
        # Legacy brier scores also set
        assert trade.brier_score_raw is not None
        assert trade.brier_score_adjusted is not None


class TestCheckEarlyExitsSkipsAlreadyExitedTrades:
    """check_early_exits skips trades that already have exit_type set."""

    @pytest.mark.asyncio
    async def test_check_early_exits_skips_already_exited_trades(self):
        """Trade A (exit_type='take_profit') must be skipped; Trade B (exit_type=None)
        must have get_market called. Since B's price doesn't trigger TP/SL,
        no update_trade calls should occur."""
        trade_a = _make_record(
            action="BUY_YES",
            market_id="mkt-exited",
            market_price_at_decision=0.50,
            position_size_usd=100.0,
            actual_outcome=None,
            pnl=15.0,
        )
        trade_a.exit_type = "take_profit"

        trade_b = _make_record(
            action="BUY_YES",
            market_id="mkt-open",
            market_price_at_decision=0.50,
            position_size_usd=100.0,
            actual_outcome=None,
            pnl=None,
        )
        # trade_b.exit_type is None (default)

        mock_db = AsyncMock()
        mock_db.get_open_trades.return_value = [trade_a, trade_b]
        mock_db.load_portfolio.return_value = Portfolio(
            cash_balance=9900.0, total_equity=10000.0, peak_equity=10000.0
        )

        # Return a market that does NOT trigger TP/SL (current price == entry price → ROI=0)
        mock_market = SimpleNamespace(resolved=False, resolution=None, yes_price=0.50)
        mock_client = AsyncMock()
        mock_client.get_market.return_value = mock_market

        settings = _make_settings_mock()

        await check_early_exits(mock_db, mock_client, settings)

        # Only trade_b should have triggered a get_market call
        assert mock_client.get_market.call_count == 1
        assert mock_client.get_market.call_args[0][0] == "mkt-open"

        # Neither trade triggered TP/SL → no update_trade
        mock_db.update_trade.assert_not_awaited()


class TestNaturalResolutionAfterTpExit:
    """After TP/SL exit, auto_resolve_trades can still write actual_outcome."""

    @pytest.mark.asyncio
    async def test_natural_resolution_after_tp_exit(self):
        """A trade with exit_type='take_profit' and actual_outcome=NULL is returned
        by get_open_trades (filter removal) and auto_resolve can set actual_outcome."""
        trade = _make_record(
            action="BUY_YES",
            market_type="political",
            grok_raw_probability=0.75,
            final_adjusted_probability=0.73,
            market_price_at_decision=0.60,
            position_size_usd=100.0,
            fee_rate=0.0,
            actual_outcome=None,
            pnl=15.0,      # already has pnl from TP exit
        )
        trade.exit_type = "take_profit"
        trade.exit_price = 0.65

        mock_db = AsyncMock()
        # Filter removal: get_open_trades now returns exited trades without actual_outcome
        mock_db.get_open_trades.return_value = [trade]
        mock_db.load_portfolio.return_value = Portfolio()

        # Market has now resolved YES
        mock_market = SimpleNamespace(resolved=True, resolution="YES", yes_price=1.0)
        mock_client = AsyncMock()
        mock_client.get_market.return_value = mock_market

        await auto_resolve_trades(mock_db, mock_client)

        mock_db.update_trade.assert_awaited_once()
        assert trade.actual_outcome is True


# ---------------------------------------------------------------------------
# F2 Tests: already_exited guard in auto_resolve_trades
# ---------------------------------------------------------------------------


class TestAutoResolveSkipsPortfolioForEarlyExited:
    """auto_resolve_trades: when exit_type is set, only write brier/outcome; skip pnl + portfolio."""

    @pytest.mark.asyncio
    async def test_skips_portfolio_update_for_early_exited(self):
        """Trade with exit_type='stop_loss' and pnl=-15 already set.
        Resolution writes actual_outcome + brier scores but leaves pnl and portfolio unchanged."""
        trade = _make_record(
            action="BUY_YES",
            market_type="political",
            grok_raw_probability=0.75,
            final_adjusted_probability=0.73,
            market_price_at_decision=0.60,
            position_size_usd=100.0,
            fee_rate=0.0,
            actual_outcome=None,
            pnl=-15.0,
        )
        trade.exit_type = "stop_loss"

        portfolio = Portfolio(cash_balance=10000.0, total_equity=10000.0, peak_equity=10000.0)

        mock_db = AsyncMock()
        mock_db.get_open_trades.return_value = [trade]
        mock_db.load_portfolio.return_value = portfolio

        mock_market = SimpleNamespace(resolved=True, resolution="YES", yes_price=1.0)
        mock_client = AsyncMock()
        mock_client.get_market.return_value = mock_market

        result = await auto_resolve_trades(mock_db, mock_client)

        # update_trade must be called once with actual_outcome + brier scores
        mock_db.update_trade.assert_awaited_once()
        assert trade.actual_outcome is True
        assert trade.brier_score_raw is not None
        assert isinstance(trade.brier_score_raw, float)
        assert trade.resolved_at is not None

        # pnl must NOT be overwritten by resolution
        assert trade.pnl == -15.0

        # portfolio save must NOT be called for already-exited trade
        mock_db.save_portfolio.assert_not_awaited()

        # trade appears in returned list
        assert trade in result


class TestAutoResolveUpdatesPortfolioForUnexited:
    """auto_resolve_trades: unexited trade gets full pnl calc + portfolio update."""

    @pytest.mark.asyncio
    async def test_updates_portfolio_for_unexited(self):
        """Trade with exit_type=None gets pnl calculated and portfolio saved."""
        trade = _make_record(
            action="BUY_YES",
            market_type="political",
            grok_raw_probability=0.75,
            final_adjusted_probability=0.73,
            market_price_at_decision=0.60,
            position_size_usd=100.0,
            fee_rate=0.0,
            actual_outcome=None,
            pnl=None,
        )
        # exit_type is None by default in _make_record (TradeRecord default)

        portfolio = Portfolio(
            cash_balance=9900.0,
            total_equity=10000.0,
            peak_equity=10000.0,
            open_positions=[Position(
                market_id="mkt-001",
                side="BUY_YES",
                entry_price=0.60,
                size_usd=100.0,
                current_value=100.0,
            )],
        )

        mock_db = AsyncMock()
        mock_db.get_open_trades.return_value = [trade]
        mock_db.load_portfolio.return_value = portfolio

        mock_market = SimpleNamespace(resolved=True, resolution="YES", yes_price=1.0)
        mock_client = AsyncMock()
        mock_client.get_market.return_value = mock_market

        initial_cash = portfolio.cash_balance

        await auto_resolve_trades(mock_db, mock_client)

        # pnl should be calculated (BUY_YES, YES outcome, entry=0.60, size=100, fee=0)
        # pnl = 100*(1-0.60) - 0 = 40
        assert trade.pnl is not None
        assert trade.pnl == pytest.approx(40.0)

        # trade_profitable and pnl_brier_* should be set
        assert trade.trade_profitable == 1
        assert trade.pnl_brier_raw is not None
        assert trade.pnl_brier_adjusted is not None

        # portfolio should have been saved
        mock_db.save_portfolio.assert_awaited_once()

        # cash_balance should increase
        saved_portfolio = mock_db.save_portfolio.call_args[0][0]
        assert saved_portfolio.cash_balance > initial_cash
