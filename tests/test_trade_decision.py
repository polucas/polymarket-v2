"""Tests for src.engine.trade_decision -- edge calculation, Kelly sizing,
Monk Mode checks, and scan mode logic."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone, timedelta

import pytest

from src.config import MonkModeConfig
from src.models import Market, OrderBook, OrderBookLevel, Portfolio, Position, TradeCandidate, TradeRecord
from src.engine.trade_decision import (
    calculate_edge,
    calculate_spread_adjusted_edge,
    compute_vwap,
    determine_side,
    kelly_size,
    kelly_size_vwap,
    check_monk_mode,
    get_scan_mode,
)


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------

def _market(**overrides) -> Market:
    defaults = dict(
        market_id=str(uuid.uuid4()),
        question="Will X happen?",
        yes_price=0.60,
        no_price=0.40,
        market_type="political",
    )
    defaults.update(overrides)
    return Market(**defaults)


def _candidate(**overrides) -> TradeCandidate:
    m = overrides.pop("market", None) or _market()
    defaults = dict(
        market=m,
        adjusted_probability=0.75,
        adjusted_confidence=0.80,
        calculated_edge=0.10,
        resolution_hours=2.0,
        position_size=200.0,
        side="BUY_YES",
        tier=1,
    )
    defaults.update(overrides)
    return TradeCandidate(**defaults)


def _trade_record(**overrides) -> TradeRecord:
    now = datetime.now(timezone.utc)
    defaults = dict(
        record_id=str(uuid.uuid4()),
        experiment_run="test-run",
        timestamp=now,
        model_used="grok-3-fast",
        market_id=str(uuid.uuid4()),
        market_question="Test?",
        market_type="political",
        resolution_window_hours=12.0,
        tier=1,
        grok_raw_probability=0.75,
        grok_raw_confidence=0.80,
        grok_reasoning="reason",
        grok_signal_types=[],
        action="BUY_YES",
    )
    defaults.update(overrides)
    return TradeRecord(**defaults)


def _monk_config(**overrides) -> MonkModeConfig:
    defaults = dict(
        tier1_daily_trade_cap=5,
        tier2_daily_trade_cap=3,
        daily_loss_limit_pct=0.05,
        weekly_loss_limit_pct=0.10,
        consecutive_loss_cooldown=3,
        cooldown_duration_hours=2.0,
        daily_api_budget_usd=8.0,
        max_position_pct=0.08,
        max_total_exposure_pct=0.30,
        kelly_fraction=0.25,
    )
    defaults.update(overrides)
    return MonkModeConfig(**defaults)


def _portfolio(**overrides) -> Portfolio:
    defaults = dict(
        cash_balance=5000.0,
        total_equity=5000.0,
        total_pnl=0.0,
        peak_equity=5000.0,
        max_drawdown=0.0,
        open_positions=[],
    )
    defaults.update(overrides)
    return Portfolio(**defaults)


# ---------------------------------------------------------------------------
# 1. calculate_edge
# ---------------------------------------------------------------------------

class TestCalculateEdge:

    def test_positive_edge(self):
        """prob=0.75, price=0.60, fee=0.02 -> |0.75-0.60| - 0.02 = 0.13"""
        result = calculate_edge(0.75, 0.60, 0.02)
        assert pytest.approx(result, abs=1e-9) == 0.13

    def test_negative_edge(self):
        """prob=0.60, price=0.60, fee=0.02 -> |0.60-0.60| - 0.02 = -0.02"""
        result = calculate_edge(0.60, 0.60, 0.02)
        assert pytest.approx(result, abs=1e-9) == -0.02


# ---------------------------------------------------------------------------
# 2. determine_side
# ---------------------------------------------------------------------------

class TestDetermineSide:

    def test_buy_yes(self):
        """prob > price -> BUY_YES"""
        assert determine_side(0.75, 0.60) == "BUY_YES"

    def test_buy_no(self):
        """prob < price -> BUY_NO"""
        assert determine_side(0.40, 0.60) == "BUY_NO"

    def test_skip(self):
        """prob == price -> SKIP"""
        assert determine_side(0.60, 0.60) == "SKIP"


# ---------------------------------------------------------------------------
# 3. kelly_size -- BUY_YES
# ---------------------------------------------------------------------------

class TestKellySizeBuyYes:

    def test_capped_at_max_position(self):
        """prob=0.80, price=0.60, bankroll=5000
        f* = (0.80-0.60)/(1-0.60) = 0.50
        quarter Kelly = 0.50*0.25 = 0.125
        raw size = 0.125*5000 = 625
        USD cap = 0.08*5000 = 400
        notional cap: max_payout = 400/0.60 = 666.7 > 400
        -> position = 400 * 0.60 = 240
        """
        result = kelly_size(0.80, 0.60, "BUY_YES", 5000.0,
                            kelly_fraction=0.25, max_position_pct=0.08)
        assert pytest.approx(result, abs=1e-2) == 240.0

    def test_small_edge(self):
        """prob=0.65, price=0.60, bankroll=5000
        f* = (0.65-0.60)/(1-0.60) = 0.125
        quarter = 0.125*0.25 = 0.03125
        size = 0.03125*5000 = 156.25
        cap = 400 (not hit)
        """
        result = kelly_size(0.65, 0.60, "BUY_YES", 5000.0,
                            kelly_fraction=0.25, max_position_pct=0.08)
        assert pytest.approx(result, abs=1e-2) == 156.25

    def test_no_edge_returns_zero(self):
        """prob=0.55 <= price=0.60 -> 0.0"""
        result = kelly_size(0.55, 0.60, "BUY_YES", 5000.0)
        assert result == 0.0


# ---------------------------------------------------------------------------
# 4. kelly_size -- BUY_NO
# ---------------------------------------------------------------------------

class TestKellySizeBuyNo:

    def test_capped_at_max_position(self):
        """prob=0.30, price=0.60, bankroll=5000
        f* = (0.60-0.30)/0.60 = 0.50
        quarter = 0.50*0.25 = 0.125
        raw = 625
        USD cap = 400
        notional cap: max_payout = 400/0.40 = 1000 > 400
        -> position = 400 * 0.40 = 160
        """
        result = kelly_size(0.30, 0.60, "BUY_NO", 5000.0,
                            kelly_fraction=0.25, max_position_pct=0.08)
        assert pytest.approx(result, abs=1e-2) == 160.0

    def test_moderate_edge(self):
        """prob=0.55, price=0.60, bankroll=5000
        f* = (0.60-0.55)/0.60 = 0.08333...
        quarter = 0.08333*0.25 = 0.02083...
        size = 0.02083*5000 = 104.1666...
        """
        result = kelly_size(0.55, 0.60, "BUY_NO", 5000.0,
                            kelly_fraction=0.25, max_position_pct=0.08)
        assert pytest.approx(result, abs=0.1) == 104.17

    def test_no_edge_returns_zero(self):
        """prob=0.65 >= price=0.60 -> 0.0"""
        result = kelly_size(0.65, 0.60, "BUY_NO", 5000.0)
        assert result == 0.0


# ---------------------------------------------------------------------------
# 5. check_monk_mode
# ---------------------------------------------------------------------------

class TestCheckMonkMode:

    def test_tier1_daily_cap_reached(self):
        """5 executed tier-1 trades today -> blocked."""
        config = _monk_config()
        signal = _candidate(tier=1, position_size=100.0)
        portfolio = _portfolio()
        today = [_trade_record(tier=1, action="BUY_YES") for _ in range(5)]
        allowed, reason = check_monk_mode(config, signal, portfolio, today, [], 0.0)
        assert allowed is False
        assert reason == "tier1_daily_cap_reached"

    def test_daily_loss_limit(self):
        """Daily PnL < -5% equity -> blocked."""
        config = _monk_config()
        signal = _candidate(tier=1, position_size=100.0)
        portfolio = _portfolio(total_equity=5000.0)
        # -300 PnL on 5000 equity = -6%
        today = [_trade_record(tier=2, action="BUY_YES", pnl=-300.0)]
        allowed, reason = check_monk_mode(config, signal, portfolio, today, [], 0.0)
        assert allowed is False
        assert reason == "daily_loss_limit"

    def test_weekly_loss_limit(self):
        """Weekly PnL < -10% equity -> blocked."""
        config = _monk_config()
        signal = _candidate(tier=1, position_size=100.0)
        portfolio = _portfolio(total_equity=5000.0)
        today: list[TradeRecord] = []
        # -600 PnL on 5000 equity = -12%
        week = [_trade_record(tier=1, action="BUY_YES", pnl=-600.0)]
        allowed, reason = check_monk_mode(config, signal, portfolio, today, week, 0.0)
        assert allowed is False
        assert reason == "weekly_loss_limit"

    def test_consecutive_losses_cooldown(self):
        """3 consecutive losing trades -> blocked."""
        config = _monk_config(consecutive_loss_cooldown=3)
        signal = _candidate(tier=1, position_size=100.0)
        portfolio = _portfolio(total_equity=5000.0)
        now = datetime.now(timezone.utc)
        today = [
            _trade_record(tier=1, action="BUY_YES", pnl=-10.0,
                          timestamp=now - timedelta(minutes=i))
            for i in range(3)
        ]
        allowed, reason = check_monk_mode(config, signal, portfolio, today, [], 0.0)
        assert allowed is False
        assert reason is not None
        assert "consecutive_adverse" in reason

    def test_max_total_exposure(self):
        """Total exposure > 30% equity -> blocked."""
        config = _monk_config()
        signal = _candidate(tier=1, position_size=200.0)
        positions = [
            Position(market_id="m1", side="BUY_YES", entry_price=0.6, size_usd=800.0),
            Position(market_id="m2", side="BUY_YES", entry_price=0.5, size_usd=800.0),
        ]
        # existing exposure = 1600, + 200 = 1800 > 30% of 5000 = 1500
        portfolio = _portfolio(total_equity=5000.0, open_positions=positions)
        allowed, reason = check_monk_mode(config, signal, portfolio, [], [], 0.0)
        assert allowed is False
        assert reason == "max_total_exposure"

    def test_api_budget_exceeded(self):
        """API spend >= $8 -> blocked."""
        config = _monk_config(daily_api_budget_usd=8.0)
        signal = _candidate(tier=1, position_size=100.0)
        portfolio = _portfolio()
        allowed, reason = check_monk_mode(config, signal, portfolio, [], [], 8.0)
        assert allowed is False
        assert reason == "api_budget_exceeded"

    def test_all_checks_pass(self):
        """All constraints satisfied -> allowed."""
        config = _monk_config()
        signal = _candidate(tier=1, position_size=100.0)
        portfolio = _portfolio(total_equity=5000.0)
        allowed, reason = check_monk_mode(config, signal, portfolio, [], [], 0.0)
        assert allowed is True
        assert reason is None


# ---------------------------------------------------------------------------
# 6. get_scan_mode
# ---------------------------------------------------------------------------

class TestGetScanMode:

    def test_observe_only_when_cap_reached(self):
        """5+ tier-1 executed trades -> observe_only."""
        config = _monk_config(tier1_daily_trade_cap=5)
        today = [_trade_record(tier=1, action="BUY_YES") for _ in range(5)]
        assert get_scan_mode(today, config) == "observe_only"

    def test_active_with_headroom(self):
        """4 executed + 2 skipped -> still active (skips don't count)."""
        config = _monk_config(tier1_daily_trade_cap=5)
        executed = [_trade_record(tier=1, action="BUY_YES") for _ in range(4)]
        skipped = [_trade_record(tier=1, action="SKIP") for _ in range(2)]
        assert get_scan_mode(executed + skipped, config) == "active"

    def test_active_with_zero_tier1_trades(self):
        """0 tier-1 executed trades -> active."""
        config = _monk_config(tier1_daily_trade_cap=5)
        assert get_scan_mode([], config) == "active"


# ---------------------------------------------------------------------------
# 7. Additional calculate_edge tests
# ---------------------------------------------------------------------------

class TestCalculateEdgeAdditional:

    def test_small_gap_with_fee(self):
        """adjusted_prob=0.55, market_price=0.60, fee=0.02
        edge = |0.55-0.60| - 0.02 = 0.05 - 0.02 = 0.03
        """
        result = calculate_edge(0.55, 0.60, 0.02)
        assert pytest.approx(result, abs=1e-9) == 0.03


# ---------------------------------------------------------------------------
# 8. Additional kelly_size edge cases
# ---------------------------------------------------------------------------

class TestKellySizeEdgeCases:

    def test_kelly_fraction_zero(self):
        """kelly_fraction=0.0 -> fractional Kelly = 0 -> returns 0.0."""
        result = kelly_size(0.80, 0.60, "BUY_YES", 5000.0,
                            kelly_fraction=0.0, max_position_pct=0.08)
        assert result == 0.0

    def test_max_position_pct_zero(self):
        """max_position_pct=0.0 -> cap = 0 -> returns 0.0."""
        result = kelly_size(0.80, 0.60, "BUY_YES", 5000.0,
                            kelly_fraction=0.25, max_position_pct=0.0)
        assert result == 0.0


# ---------------------------------------------------------------------------
# Notional exposure cap tests
# ---------------------------------------------------------------------------

class TestNotionalExposureCap:

    def test_buy_no_extreme_price_capped(self):
        """BUY_NO at YES=0.95 (NO=0.05). Without notional cap, $400 buys
        8000 shares with max payout $8000. Cap should reduce position so
        max payout <= max_position_pct * bankroll = $400."""
        result = kelly_size(0.30, 0.95, "BUY_NO", 5000.0,
                            kelly_fraction=0.25, max_position_pct=0.08)
        # max_position = 0.08 * 5000 = 400
        # At NO price 0.05: max_payout = position / 0.05
        # Cap: position = 400 * 0.05 = 20
        assert result == pytest.approx(20.0, abs=1e-2)

    def test_buy_yes_extreme_price_capped(self):
        """BUY_YES at YES=0.05. Without notional cap, $400 buys
        8000 shares with max payout $8000. Cap should reduce position."""
        result = kelly_size(0.70, 0.05, "BUY_YES", 5000.0,
                            kelly_fraction=0.25, max_position_pct=0.08)
        # max_position = 400
        # At YES price 0.05: max_payout = position / 0.05
        # Cap: position = 400 * 0.05 = 20
        assert result == pytest.approx(20.0, abs=1e-2)

    def test_midrange_price_not_capped(self):
        """BUY_YES at YES=0.50. max_payout = position / 0.50 = 2x position.
        With max_position=400, max_payout=800 > 400 -> cap applies but
        position = 400 * 0.50 = 200. However the raw kelly is smaller."""
        result = kelly_size(0.65, 0.60, "BUY_YES", 5000.0,
                            kelly_fraction=0.25, max_position_pct=0.08)
        # f* = (0.65-0.60)/(1-0.60) = 0.125
        # quarter Kelly = 0.125*0.25 = 0.03125
        # raw = 0.03125*5000 = 156.25
        # max_position = 400, so position = 156.25 (not hit)
        # max_payout = 156.25 / 0.60 = 260.4 < 400 -> no notional cap
        assert result == pytest.approx(156.25, abs=1e-2)

    def test_buy_no_midrange_not_affected(self):
        """BUY_NO at YES=0.60 (NO=0.40). max_payout = position / 0.40 = 2.5x.
        Small position won't hit cap."""
        result = kelly_size(0.30, 0.60, "BUY_NO", 5000.0,
                            kelly_fraction=0.25, max_position_pct=0.08)
        # f* = (0.60-0.30)/0.60 = 0.50
        # quarter = 0.125, raw = 625
        # capped at 400 (USD cap)
        # max_payout = 400 / 0.40 = 1000 > 400 -> notional cap fires
        # position = 400 * 0.40 = 160
        assert result == pytest.approx(160.0, abs=1e-2)

    def test_notional_cap_max_payout_bounded(self):
        """Verify the invariant: max payout never exceeds max_position_pct * bankroll."""
        for price in [0.05, 0.10, 0.20, 0.50, 0.80, 0.90, 0.95]:
            bankroll = 10000.0
            max_pct = 0.016
            max_position = max_pct * bankroll  # $160
            # BUY_NO
            size_no = kelly_size(0.01, price, "BUY_NO", bankroll,
                                 kelly_fraction=0.25, max_position_pct=max_pct)
            if size_no > 0 and (1.0 - price) > 0:
                max_payout_no = size_no / (1.0 - price)
                assert max_payout_no <= max_position + 0.01, (
                    f"BUY_NO at {price}: payout {max_payout_no} > cap {max_position}")
            # BUY_YES
            size_yes = kelly_size(0.99, price, "BUY_YES", bankroll,
                                  kelly_fraction=0.25, max_position_pct=max_pct)
            if size_yes > 0 and price > 0:
                max_payout_yes = size_yes / price
                assert max_payout_yes <= max_position + 0.01, (
                    f"BUY_YES at {price}: payout {max_payout_yes} > cap {max_position}")


# ---------------------------------------------------------------------------
# 9. Additional check_monk_mode tests
# ---------------------------------------------------------------------------

class TestCheckMonkModeAdditional:

    def test_tier1_with_4_trades_passes(self):
        """Tier 1 with 4 executed trades -> passes (cap is 5, allowed=True)."""
        config = _monk_config(tier1_daily_trade_cap=5)
        signal = _candidate(tier=1, position_size=100.0)
        portfolio = _portfolio(total_equity=5000.0)
        today = [_trade_record(tier=1, action="BUY_YES") for _ in range(4)]
        allowed, reason = check_monk_mode(config, signal, portfolio, today, [], 0.0)
        assert allowed is True
        assert reason is None

    def test_tier2_with_3_trades_blocked(self):
        """Tier 2 with 3 executed trades -> False, tier2_daily_cap_reached
        (cap is 3, so 3 >= 3 triggers block)."""
        config = _monk_config(tier2_daily_trade_cap=3)
        signal = _candidate(tier=2, position_size=100.0)
        portfolio = _portfolio(total_equity=5000.0)
        today = [_trade_record(tier=2, action="BUY_YES") for _ in range(3)]
        allowed, reason = check_monk_mode(config, signal, portfolio, today, [], 0.0)
        assert allowed is False
        assert reason == "tier2_daily_cap_reached"

    def test_daily_loss_below_limit_passes(self):
        """Today PnL = -$240 with equity $5000 -> -4.8% loss.
        daily_loss_limit_pct=0.05 (5%), so -4.8% > -5% -> passes."""
        config = _monk_config(daily_loss_limit_pct=0.05)
        signal = _candidate(tier=1, position_size=100.0)
        portfolio = _portfolio(total_equity=5000.0)
        today = [_trade_record(tier=1, action="BUY_YES", pnl=-240.0)]
        allowed, reason = check_monk_mode(config, signal, portfolio, today, [], 0.0)
        assert allowed is True
        assert reason is None

    def test_consecutive_losses_plus_unrealized_adverse(self):
        """2 consecutive losses + 1 unrealized adverse move >10% = 3 adverse total
        -> blocked (consecutive_loss_cooldown=3)."""
        config = _monk_config(consecutive_loss_cooldown=3)
        signal = _candidate(tier=1, position_size=100.0)
        portfolio = _portfolio(total_equity=5000.0)
        now = datetime.now(timezone.utc)
        today = [
            # Most recent: unrealized adverse move > 10%
            _trade_record(tier=1, action="BUY_YES", pnl=None,
                          unrealized_adverse_move=0.15,
                          timestamp=now - timedelta(minutes=1)),
            # Second: losing trade
            _trade_record(tier=1, action="BUY_YES", pnl=-20.0,
                          timestamp=now - timedelta(minutes=2)),
            # Third: losing trade
            _trade_record(tier=1, action="BUY_YES", pnl=-15.0,
                          timestamp=now - timedelta(minutes=3)),
        ]
        allowed, reason = check_monk_mode(config, signal, portfolio, today, [], 0.0)
        assert allowed is False
        assert reason is not None
        assert "consecutive_adverse" in reason

    def test_two_consecutive_losses_passes(self):
        """2 consecutive losses only -> passes (below threshold of 3)."""
        config = _monk_config(consecutive_loss_cooldown=3)
        signal = _candidate(tier=1, position_size=100.0)
        portfolio = _portfolio(total_equity=5000.0)
        now = datetime.now(timezone.utc)
        today = [
            _trade_record(tier=1, action="BUY_YES", pnl=-20.0,
                          timestamp=now - timedelta(minutes=1)),
            _trade_record(tier=1, action="BUY_YES", pnl=-15.0,
                          timestamp=now - timedelta(minutes=2)),
        ]
        allowed, reason = check_monk_mode(config, signal, portfolio, today, [], 0.0)
        assert allowed is True
        assert reason is None

    def test_cooldown_expired_old_losses_passes(self):
        """Cooldown expired (only old losses, with a winning trade breaking the
        consecutive streak) -> passes."""
        config = _monk_config(consecutive_loss_cooldown=3)
        signal = _candidate(tier=1, position_size=100.0)
        portfolio = _portfolio(total_equity=5000.0)
        now = datetime.now(timezone.utc)
        today = [
            # Most recent: a winning trade breaks the consecutive streak
            _trade_record(tier=1, action="BUY_YES", pnl=50.0,
                          timestamp=now - timedelta(minutes=1)),
            # Older: 3 losses (but the win above breaks the sequence)
            _trade_record(tier=1, action="BUY_YES", pnl=-20.0,
                          timestamp=now - timedelta(hours=3)),
            _trade_record(tier=1, action="BUY_YES", pnl=-15.0,
                          timestamp=now - timedelta(hours=4)),
            _trade_record(tier=1, action="BUY_YES", pnl=-10.0,
                          timestamp=now - timedelta(hours=5)),
        ]
        allowed, reason = check_monk_mode(config, signal, portfolio, today, [], 0.0)
        assert allowed is True
        assert reason is None


# ---------------------------------------------------------------------------
# 10. calculate_spread_adjusted_edge
# ---------------------------------------------------------------------------

class TestCalculateSpreadAdjustedEdge:
    def test_buy_yes_uses_ask_price(self):
        """BUY_YES edge should use best_ask, not market_price."""
        edge = calculate_spread_adjusted_edge(
            adjusted_prob=0.70, market_price=0.50, fee_rate=0.0,
            side="BUY_YES", best_bid=0.48, best_ask=0.55,
        )
        # Edge should be 0.70 - 0.55 = 0.15 (not 0.70 - 0.50 = 0.20)
        assert abs(edge - 0.15) < 1e-9

    def test_buy_no_uses_bid_price(self):
        """BUY_NO edge should use best_bid."""
        edge = calculate_spread_adjusted_edge(
            adjusted_prob=0.30, market_price=0.50, fee_rate=0.0,
            side="BUY_NO", best_bid=0.45, best_ask=0.55,
        )
        # Edge should be |0.30 - 0.45| = 0.15 (not |0.30 - 0.50| = 0.20)
        assert abs(edge - 0.15) < 1e-9

    def test_fallback_to_market_price_when_no_orderbook(self):
        """Falls back to market_price when orderbook is empty."""
        edge = calculate_spread_adjusted_edge(
            adjusted_prob=0.70, market_price=0.50, fee_rate=0.0,
            side="BUY_YES", best_bid=None, best_ask=None,
        )
        assert abs(edge - 0.20) < 1e-9

    def test_fee_rate_subtracted(self):
        """Fee rate is subtracted from edge."""
        edge = calculate_spread_adjusted_edge(
            adjusted_prob=0.70, market_price=0.50, fee_rate=0.02,
            side="BUY_YES", best_bid=None, best_ask=None,
        )
        assert abs(edge - 0.18) < 1e-9


# ---------------------------------------------------------------------------
# 11. compute_vwap
# ---------------------------------------------------------------------------

class TestComputeVWAP:
    def test_single_level_full_fill(self):
        """VWAP equals level price when fully filled at one level."""
        levels = [OrderBookLevel(price=0.50, size=200)]
        vwap, filled = compute_vwap(levels, 100)
        assert abs(vwap - 0.50) < 1e-9
        assert abs(filled - 100) < 1e-9

    def test_multiple_levels(self):
        """VWAP walks multiple levels; filled equals requested USD."""
        levels = [
            OrderBookLevel(price=0.50, size=100),  # $50 available (100 shares * $0.50)
            OrderBookLevel(price=0.55, size=100),  # $55 available (100 shares * $0.55)
        ]
        vwap, filled = compute_vwap(levels, 80)
        # level 1: $50 spent, 100 shares at $0.50
        # level 2: $30 spent, 30/0.55 = 54.545 shares at $0.55
        # VWAP = $80 / (100 + 54.545) = 0.5176...
        assert filled == 80
        expected_vwap = 80 / (100 + 30 / 0.55)
        assert abs(vwap - expected_vwap) < 1e-6

    def test_partial_fill_insufficient_depth(self):
        """When orderbook is too thin, only fills what's available."""
        levels = [OrderBookLevel(price=0.50, size=40)]  # Only $20 available
        vwap, filled = compute_vwap(levels, 100)
        assert abs(filled - 20) < 1e-9

    def test_empty_levels(self):
        """Empty orderbook returns zero."""
        vwap, filled = compute_vwap([], 100)
        assert vwap == 0.0
        assert filled == 0.0


# ---------------------------------------------------------------------------
# 12. kelly_size_vwap
# ---------------------------------------------------------------------------

class TestKellySizeVWAP:
    def _make_orderbook(self, ask_levels=None, bid_levels=None):
        return OrderBook(
            market_id="test",
            bids=[OrderBookLevel(p, s) for p, s in (bid_levels or [])],
            asks=[OrderBookLevel(p, s) for p, s in (ask_levels or [])],
        )

    def test_deep_book_returns_full_kelly(self):
        """Deep orderbook returns full Kelly size."""
        ob = self._make_orderbook(ask_levels=[(0.50, 10000)])
        size, vwap = kelly_size_vwap(
            adjusted_prob=0.70, market_price=0.50, side="BUY_YES",
            bankroll=10000, orderbook=ob,
            kelly_fraction=0.25, max_position_pct=0.016, fee_rate=0.0,
        )
        assert size > 0

    def test_thin_book_caps_size(self):
        """Thin orderbook caps position at available depth."""
        ob = self._make_orderbook(ask_levels=[(0.50, 10)])  # Only $5 available
        size, vwap = kelly_size_vwap(
            adjusted_prob=0.70, market_price=0.50, side="BUY_YES",
            bankroll=10000, orderbook=ob,
            kelly_fraction=0.25, max_position_pct=0.016, fee_rate=0.0,
        )
        assert size <= 5.0  # Can't exceed available depth

    def test_empty_orderbook_falls_back(self):
        """Empty orderbook falls back to standard Kelly."""
        ob = self._make_orderbook()
        size, vwap = kelly_size_vwap(
            adjusted_prob=0.70, market_price=0.50, side="BUY_YES",
            bankroll=10000, orderbook=ob,
            kelly_fraction=0.25, max_position_pct=0.016, fee_rate=0.0,
        )
        # Should fall back to standard kelly_size behavior
        assert size > 0  # kelly_size returns > 0 for this edge
        assert vwap == 0.50  # market_price as fallback

    def test_no_edge_returns_zero(self):
        """When probability <= market_price for BUY_YES, returns 0."""
        ob = self._make_orderbook(ask_levels=[(0.50, 10000)])
        size, vwap = kelly_size_vwap(
            adjusted_prob=0.40, market_price=0.50, side="BUY_YES",
            bankroll=10000, orderbook=ob,
        )
        assert size == 0.0
