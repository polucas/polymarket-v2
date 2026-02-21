"""Tests for MarketTypeManager (learning layer 2)."""

import pytest

from src.learning.market_type import MarketTypeManager


# ---------------------------------------------------------------------------
# update_market_type
# ---------------------------------------------------------------------------


class TestUpdateMarketType:
    """Tests 11-13: MarketTypeManager.update_market_type."""

    def test_executed_trade_increments_total_trades_and_appends_brier(self, sample_trade_record):
        """Test 11: An executed (non-SKIP) trade increments total_trades
        and appends brier_score_adjusted to brier_scores."""
        mgr = MarketTypeManager()
        record = sample_trade_record(
            market_type="political",
            action="BUY_YES",
            actual_outcome=True,
            brier_score_adjusted=0.12,
            voided=False,
        )
        mgr.update_market_type(record)

        perf = mgr.performances["political"]
        assert perf.total_trades == 1
        assert perf.brier_scores == [0.12]

    def test_skipped_trade_increments_total_observed(self, sample_trade_record):
        """Test 12: A skipped trade increments total_observed (not total_trades PnL)."""
        mgr = MarketTypeManager()
        record = sample_trade_record(
            market_type="crypto_15m",
            action="SKIP",
            actual_outcome=True,
            brier_score_adjusted=0.20,
            pnl=None,
            voided=False,
        )
        mgr.update_market_type(record, counterfactual_pnl=5.0)

        perf = mgr.performances["crypto_15m"]
        assert perf.total_trades == 1
        assert perf.total_observed == 1
        assert perf.counterfactual_pnl == 5.0
        # brier_score_adjusted is still appended (even for skips)
        assert perf.brier_scores == [0.20]

    def test_uses_brier_score_adjusted_not_raw(self, sample_trade_record):
        """Test 13: update_market_type appends brier_score_adjusted, not brier_score_raw."""
        mgr = MarketTypeManager()
        record = sample_trade_record(
            market_type="sports",
            action="BUY_YES",
            actual_outcome=True,
            brier_score_raw=0.30,
            brier_score_adjusted=0.18,
            voided=False,
        )
        mgr.update_market_type(record)

        perf = mgr.performances["sports"]
        # Must contain brier_score_adjusted, not brier_score_raw
        assert perf.brier_scores == [0.18]


# ---------------------------------------------------------------------------
# get_edge_adjustment
# ---------------------------------------------------------------------------


class TestGetEdgeAdjustment:
    """Tests 14-15: MarketTypeManager.get_edge_adjustment."""

    def test_unknown_market_type_returns_zero(self):
        """Test 14: get_edge_adjustment for unknown market type -> 0.0."""
        mgr = MarketTypeManager()
        assert mgr.get_edge_adjustment("nonexistent") == 0.0

    def test_below_15_trades_returns_zero(self, sample_trade_record):
        """Test 15: get_edge_adjustment returns 0.0 when total_trades < 15."""
        mgr = MarketTypeManager()
        for i in range(14):
            record = sample_trade_record(
                record_id=f"rec-{i}",
                market_type="political",
                action="BUY_YES",
                actual_outcome=True,
                brier_score_adjusted=0.35,  # high brier, but not enough trades
                voided=False,
            )
            mgr.update_market_type(record)

        perf = mgr.performances["political"]
        assert perf.total_trades == 14
        assert mgr.get_edge_adjustment("political") == 0.0


# ---------------------------------------------------------------------------
# dampen_on_swap
# ---------------------------------------------------------------------------


class TestDampenOnSwap:
    """Test 16: dampen_on_swap truncates to last 15 brier scores."""

    def test_dampen_on_swap_truncates_to_last_15(self, sample_trade_record):
        """Test 16: After dampen_on_swap, each market type keeps only the last 15 Brier scores."""
        mgr = MarketTypeManager()
        for i in range(30):
            record = sample_trade_record(
                record_id=f"rec-{i}",
                market_type="political",
                action="BUY_YES",
                actual_outcome=True,
                brier_score_adjusted=float(i),  # 0.0, 1.0, ..., 29.0
                voided=False,
            )
            mgr.update_market_type(record)

        perf = mgr.performances["political"]
        assert len(perf.brier_scores) == 30

        mgr.dampen_on_swap()

        assert len(perf.brier_scores) == 15
        # Should keep the LAST 15: values 15.0 through 29.0
        assert perf.brier_scores == [float(i) for i in range(15, 30)]


# ---------------------------------------------------------------------------
# Tests 20-22: get_edge_adjustment with various avg_brier values
# ---------------------------------------------------------------------------


class TestEdgeAdjustmentValues:
    """Tests 20-22: edge_adjustment depends on avg_brier and total_trades >= 15."""

    def test_high_brier_returns_005(self, sample_trade_record):
        """Test 20: Market type with 20 trades, avg_brier=0.32 -> edge_adjustment = 0.05.
        All brier scores set to 0.32 so the weighted avg equals 0.32."""
        mgr = MarketTypeManager()
        for i in range(20):
            record = sample_trade_record(
                record_id=f"rec-{i}",
                market_type="sports",
                action="BUY_YES",
                actual_outcome=True,
                brier_score_adjusted=0.32,
                voided=False,
            )
            mgr.update_market_type(record)

        perf = mgr.performances["sports"]
        assert perf.total_trades == 20
        # When all scores are the same, weighted avg = that value
        assert abs(perf.avg_brier - 0.32) < 0.01
        assert mgr.get_edge_adjustment("sports") == 0.05

    def test_medium_brier_returns_003(self, sample_trade_record):
        """Test 21: Market type with 20 trades, avg_brier=0.27 -> edge_adjustment = 0.03.
        All brier scores set to 0.27 so the weighted avg equals 0.27."""
        mgr = MarketTypeManager()
        for i in range(20):
            record = sample_trade_record(
                record_id=f"rec-{i}",
                market_type="economic",
                action="BUY_YES",
                actual_outcome=True,
                brier_score_adjusted=0.27,
                voided=False,
            )
            mgr.update_market_type(record)

        perf = mgr.performances["economic"]
        assert perf.total_trades == 20
        assert abs(perf.avg_brier - 0.27) < 0.01
        assert mgr.get_edge_adjustment("economic") == 0.03

    def test_low_brier_returns_zero(self, sample_trade_record):
        """Test 22: Market type with 20 trades, avg_brier=0.18 -> edge_adjustment = 0.0.
        0.18 <= 0.20, so no adjustment needed."""
        mgr = MarketTypeManager()
        for i in range(20):
            record = sample_trade_record(
                record_id=f"rec-{i}",
                market_type="crypto_15m",
                action="BUY_YES",
                actual_outcome=True,
                brier_score_adjusted=0.18,
                voided=False,
            )
            mgr.update_market_type(record)

        perf = mgr.performances["crypto_15m"]
        assert perf.total_trades == 20
        assert abs(perf.avg_brier - 0.18) < 0.01
        assert mgr.get_edge_adjustment("crypto_15m") == 0.0


# ---------------------------------------------------------------------------
# Test 24: dampen_on_swap with < 15 scores leaves them unchanged
# ---------------------------------------------------------------------------


class TestDampenOnSwapUnchanged:
    """Test 24: dampen_on_swap with fewer than 15 Brier scores leaves them unchanged."""

    def test_dampen_on_swap_under_15_unchanged(self, sample_trade_record):
        """Test 24: Market type with 10 Brier scores -> dampen_on_swap keeps all 10."""
        mgr = MarketTypeManager()
        for i in range(10):
            record = sample_trade_record(
                record_id=f"rec-{i}",
                market_type="political",
                action="BUY_YES",
                actual_outcome=True,
                brier_score_adjusted=0.10 + i * 0.01,
                voided=False,
            )
            mgr.update_market_type(record)

        perf = mgr.performances["political"]
        assert len(perf.brier_scores) == 10
        original_scores = list(perf.brier_scores)

        mgr.dampen_on_swap()

        # Since 10 < 15, dampen_on_swap's [-15:] keeps all scores unchanged
        assert len(perf.brier_scores) == 10
        assert perf.brier_scores == original_scores
