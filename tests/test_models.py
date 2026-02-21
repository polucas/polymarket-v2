import pytest
from src.models import (
    CalibrationBucket, MarketTypePerformance, SignalTracker,
    TradeRecord, SOURCE_TIER_CREDIBILITY, CALIBRATION_BUCKET_RANGES,
)
from datetime import datetime, timezone


class TestCalibrationBucket:
    def test_expected_accuracy(self):
        b = CalibrationBucket((0.70, 0.80), alpha=10, beta=2)
        assert abs(b.expected_accuracy - 10/12) < 1e-6

    def test_sample_count_zero(self):
        b = CalibrationBucket((0.50, 0.60), alpha=1, beta=1)
        assert b.sample_count == 0

    def test_sample_count_eight(self):
        b = CalibrationBucket((0.50, 0.60), alpha=6, beta=4)
        assert b.sample_count == 8

    def test_update_correct(self):
        b = CalibrationBucket((0.50, 0.60), alpha=1.0, beta=1.0)
        b.update(was_correct=True, recency_weight=1.0)
        assert b.alpha == 2.0
        assert b.beta == 1.0

    def test_update_incorrect(self):
        b = CalibrationBucket((0.50, 0.60), alpha=1.0, beta=1.0)
        b.update(was_correct=False, recency_weight=1.0)
        assert b.alpha == 1.0
        assert b.beta == 2.0

    def test_update_with_recency_weight(self):
        b = CalibrationBucket((0.50, 0.60), alpha=1.0, beta=1.0)
        b.update(was_correct=True, recency_weight=0.5)
        assert b.alpha == 1.5
        assert b.beta == 1.0

    def test_correction_zero_low_samples(self):
        b = CalibrationBucket((0.60, 0.70), alpha=3, beta=3)
        assert b.sample_count < 10
        assert b.get_correction() == 0.0

    def test_correction_positive_underconfident(self):
        # expected_accuracy > midpoint -> positive correction
        b = CalibrationBucket((0.60, 0.70), alpha=20, beta=5)
        assert b.sample_count >= 10
        assert b.get_correction() > 0.0

    def test_correction_negative_overconfident(self):
        # expected_accuracy < midpoint -> negative correction
        b = CalibrationBucket((0.60, 0.70), alpha=5, beta=20)
        assert b.sample_count >= 10
        assert b.get_correction() < 0.0

    def test_uncertainty_uses_scipy(self):
        b = CalibrationBucket((0.50, 0.60), alpha=10, beta=10)
        assert b.uncertainty > 0


class TestMarketTypePerformance:
    def test_avg_brier_empty(self):
        m = MarketTypePerformance(market_type="political")
        assert m.avg_brier == 0.25

    def test_avg_brier_single(self):
        m = MarketTypePerformance(market_type="political", brier_scores=[0.15])
        assert abs(m.avg_brier - 0.15) < 1e-6

    def test_avg_brier_exponential_decay(self):
        m = MarketTypePerformance(market_type="political", brier_scores=[0.10, 0.20, 0.30])
        # weights: [0.95^2, 0.95^1, 0.95^0] = [0.9025, 0.95, 1.0]
        expected = (0.10*0.9025 + 0.20*0.95 + 0.30*1.0) / (0.9025 + 0.95 + 1.0)
        assert abs(m.avg_brier - expected) < 1e-6

    def test_edge_adjustment_low_trades(self):
        m = MarketTypePerformance(market_type="political", total_trades=10)
        assert m.edge_adjustment == 0.0

    def test_edge_adjustment_high_brier(self):
        m = MarketTypePerformance(market_type="political", total_trades=20, brier_scores=[0.35]*20)
        assert m.edge_adjustment == 0.05

    def test_edge_adjustment_medium_brier(self):
        m = MarketTypePerformance(market_type="political", total_trades=20, brier_scores=[0.27]*20)
        assert m.edge_adjustment == 0.03

    def test_edge_adjustment_low_medium_brier(self):
        m = MarketTypePerformance(market_type="political", total_trades=20, brier_scores=[0.22]*20)
        assert m.edge_adjustment == 0.01

    def test_edge_adjustment_good_brier(self):
        m = MarketTypePerformance(market_type="political", total_trades=20, brier_scores=[0.15]*20)
        assert m.edge_adjustment == 0.0

    def test_should_disable_true(self):
        m = MarketTypePerformance(market_type="political", total_trades=30, total_pnl=-5.0)
        assert m.should_disable is True

    def test_should_disable_false_low_trades(self):
        m = MarketTypePerformance(market_type="political", total_trades=20, total_pnl=-10.0)
        assert m.should_disable is False


class TestSignalTracker:
    def test_lift_insufficient_present(self):
        t = SignalTracker("S2", "I2", "political", present_in_winning_trades=2, present_in_losing_trades=1,
                          absent_in_winning_trades=10, absent_in_losing_trades=10)
        assert t.lift == 1.0

    def test_lift_insufficient_absent(self):
        t = SignalTracker("S2", "I2", "political", present_in_winning_trades=10, present_in_losing_trades=10,
                          absent_in_winning_trades=2, absent_in_losing_trades=1)
        assert t.lift == 1.0

    def test_lift_calculation(self):
        t = SignalTracker("S2", "I2", "political", present_in_winning_trades=8, present_in_losing_trades=2,
                          absent_in_winning_trades=5, absent_in_losing_trades=5)
        assert abs(t.lift - 1.6) < 1e-6

    def test_weight_normal(self):
        t = SignalTracker("S2", "I2", "political", present_in_winning_trades=8, present_in_losing_trades=2,
                          absent_in_winning_trades=5, absent_in_losing_trades=5)
        # lift = 1.6, raw = 1 + 0.6*0.3 = 1.18
        assert abs(t.weight - 1.18) < 1e-6

    def test_weight_clamped_high(self):
        t = SignalTracker("S2", "I2", "political", present_in_winning_trades=10, present_in_losing_trades=0,
                          absent_in_winning_trades=1, absent_in_losing_trades=9)
        assert t.weight == 1.2

    def test_weight_clamped_low(self):
        t = SignalTracker("S2", "I2", "political", present_in_winning_trades=1, present_in_losing_trades=9,
                          absent_in_winning_trades=9, absent_in_losing_trades=1)
        assert t.weight == 0.8

    def test_weight_insufficient_data(self):
        t = SignalTracker("S2", "I2", "political")
        assert t.weight == 1.0


class TestSourceTierCredibility:
    def test_all_tiers(self):
        assert SOURCE_TIER_CREDIBILITY["S1"] == 0.95
        assert SOURCE_TIER_CREDIBILITY["S2"] == 0.90
        assert SOURCE_TIER_CREDIBILITY["S3"] == 0.80
        assert SOURCE_TIER_CREDIBILITY["S4"] == 0.65
        assert SOURCE_TIER_CREDIBILITY["S5"] == 0.70
        assert SOURCE_TIER_CREDIBILITY["S6"] == 0.30

    def test_bucket_ranges(self):
        assert len(CALIBRATION_BUCKET_RANGES) == 6
        assert CALIBRATION_BUCKET_RANGES[0] == (0.50, 0.60)
        assert CALIBRATION_BUCKET_RANGES[-1] == (0.95, 1.00)
