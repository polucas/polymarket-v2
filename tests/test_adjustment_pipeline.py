"""Tests for adjust_prediction (5-step adjustment pipeline)."""

from datetime import datetime, timezone

import pytest

from src.learning.adjustment import adjust_prediction
from src.learning.calibration import CalibrationManager
from src.learning.market_type import MarketTypeManager
from src.learning.signal_tracker import SignalTrackerManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh_managers():
    """Return fresh (empty) manager instances."""
    return CalibrationManager(), MarketTypeManager(), SignalTrackerManager()


# ---------------------------------------------------------------------------
# Fresh state
# ---------------------------------------------------------------------------


class TestFreshState:
    """Test 22: With no prior learning data, probabilities pass through unchanged."""

    def test_fresh_state_no_change(self):
        """Test 22: Fresh managers (no calibration/signal data) ->
        probabilities and confidence unchanged (no correction, no shrinkage, no edge)."""
        cal, mkt, sig = _fresh_managers()

        adj_prob, adj_conf, extra_edge = adjust_prediction(
            grok_probability=0.72,
            grok_confidence=0.80,
            market_type="political",
            signal_tags=[],
            calibration_mgr=cal,
            market_type_mgr=mkt,
            signal_tracker_mgr=sig,
        )

        # No calibration correction (fresh -> sample_count=0 -> correction=0.0)
        # No signal tags -> no signal weight adjustment
        # No shrinkage (sample_count < 10)
        # No market type edge (unknown type)
        assert adj_prob == 0.72
        assert adj_conf == 0.80
        assert extra_edge == 0.0


# ---------------------------------------------------------------------------
# Clamping
# ---------------------------------------------------------------------------


class TestClamping:
    """Tests 23-24: Adjusted confidence and probability are clamped."""

    def test_adjusted_confidence_clamped_to_range(self):
        """Test 23: Adjusted confidence is clamped to [0.50, 0.99]."""
        cal, mkt, sig = _fresh_managers()

        # Force a large positive calibration correction by manipulating bucket
        bucket = cal.find_bucket(0.97)
        # Make bucket have > 10 samples and high expected_accuracy
        bucket.alpha = 50.0  # expected_accuracy = 50/(50+1) ~ 0.98
        bucket.beta = 1.0
        # sample_count = (50+1) - 2 = 49

        _, adj_conf, _ = adjust_prediction(
            grok_probability=0.95,
            grok_confidence=0.97,
            market_type="political",
            signal_tags=[],
            calibration_mgr=cal,
            market_type_mgr=mkt,
            signal_tracker_mgr=sig,
        )

        assert adj_conf >= 0.50
        assert adj_conf <= 0.99

    def test_adjusted_probability_clamped_to_range(self):
        """Test 24: Adjusted probability is clamped to [0.01, 0.99]."""
        cal, mkt, sig = _fresh_managers()

        # Set up a bucket with enough samples so shrinkage applies
        bucket = cal.find_bucket(0.55)
        bucket.alpha = 50.0
        bucket.beta = 2.0
        # sample_count = (50+2) - 2 = 50, so shrinkage will be applied

        adj_prob, _, _ = adjust_prediction(
            grok_probability=0.99,  # extreme probability
            grok_confidence=0.55,
            market_type="political",
            signal_tags=[],
            calibration_mgr=cal,
            market_type_mgr=mkt,
            signal_tracker_mgr=sig,
        )

        assert adj_prob >= 0.01
        assert adj_prob <= 0.99


# ---------------------------------------------------------------------------
# Signal tags
# ---------------------------------------------------------------------------


class TestSignalTags:
    """Test 25: No signal tags -> no signal weight adjustment."""

    def test_no_signal_tags_no_weight_adjustment(self):
        """Test 25: When signal_tags is empty, signal weight step is skipped entirely."""
        cal, mkt, sig = _fresh_managers()

        _, adj_conf_no_tags, _ = adjust_prediction(
            grok_probability=0.72,
            grok_confidence=0.80,
            market_type="political",
            signal_tags=[],
            calibration_mgr=cal,
            market_type_mgr=mkt,
            signal_tracker_mgr=sig,
        )

        # With unknown signal weights (all 1.0), the weight adjustment is
        # (avg_weight - 1.0) * 0.1 = 0. But with no tags, the step is skipped.
        # Either way, confidence should be the same as the base.
        _, adj_conf_with_tags, _ = adjust_prediction(
            grok_probability=0.72,
            grok_confidence=0.80,
            market_type="political",
            signal_tags=[{"source_tier": "S6", "info_type": "I5"}],
            calibration_mgr=cal,
            market_type_mgr=mkt,
            signal_tracker_mgr=sig,
        )

        # With fresh tracker: weight=1.0, so (1.0 - 1.0)*0.1 = 0.
        # Both should yield same confidence in this case.
        assert adj_conf_no_tags == adj_conf_with_tags


# ---------------------------------------------------------------------------
# Market type edge
# ---------------------------------------------------------------------------


class TestMarketTypeEdge:
    """Test 26: Market type with high brier -> extra_edge > 0."""

    def test_high_brier_yields_positive_extra_edge(self, sample_trade_record):
        """Test 26: When avg_brier > 0.30 and total_trades >= 15, extra_edge > 0."""
        cal, mkt, sig = _fresh_managers()

        # Feed 20 trades with high brier scores (> 0.30)
        for i in range(20):
            record = sample_trade_record(
                record_id=f"rec-{i}",
                market_type="sports",
                action="BUY_YES",
                actual_outcome=True,
                brier_score_adjusted=0.40,  # bad calibration
                voided=False,
            )
            mkt.update_market_type(record)

        perf = mkt.performances["sports"]
        assert perf.total_trades == 20
        assert perf.avg_brier > 0.30

        _, _, extra_edge = adjust_prediction(
            grok_probability=0.72,
            grok_confidence=0.80,
            market_type="sports",
            signal_tags=[],
            calibration_mgr=cal,
            market_type_mgr=mkt,
            signal_tracker_mgr=sig,
        )

        assert extra_edge > 0.0
        # With avg_brier > 0.30 and >= 15 trades, edge_adjustment = 0.05
        assert extra_edge == 0.05


# ---------------------------------------------------------------------------
# Calibration correction (confidence)
# ---------------------------------------------------------------------------


class TestCalibrationCorrection:
    """Tests 31-32: Calibration correction adjusts confidence."""

    def test_positive_calibration_correction(self):
        """Test 31: Calibration correction of +0.05 -> confidence increases by 0.05.
        Set up a CalibrationManager with a bucket where get_correction returns +0.05.
        """
        from unittest.mock import patch

        cal, mkt, sig = _fresh_managers()

        # We need get_correction to return +0.05 for the given confidence
        with patch.object(cal, "get_correction", return_value=0.05):
            adj_prob, adj_conf, extra_edge = adjust_prediction(
                grok_probability=0.72,
                grok_confidence=0.75,
                market_type="political",
                signal_tags=[],
                calibration_mgr=cal,
                market_type_mgr=mkt,
                signal_tracker_mgr=sig,
            )

        # Confidence should increase by 0.05 (no other adjustments apply)
        assert adj_conf == pytest.approx(0.80, abs=1e-9)

    def test_negative_calibration_correction(self):
        """Test 32: Calibration correction of -0.05 -> confidence decreases by 0.05."""
        from unittest.mock import patch

        cal, mkt, sig = _fresh_managers()

        with patch.object(cal, "get_correction", return_value=-0.05):
            adj_prob, adj_conf, extra_edge = adjust_prediction(
                grok_probability=0.72,
                grok_confidence=0.75,
                market_type="political",
                signal_tags=[],
                calibration_mgr=cal,
                market_type_mgr=mkt,
                signal_tracker_mgr=sig,
            )

        # Confidence should decrease by 0.05 (no other adjustments apply)
        assert adj_conf == pytest.approx(0.70, abs=1e-9)


# ---------------------------------------------------------------------------
# Signal weight adjustments (confidence)
# ---------------------------------------------------------------------------


class TestSignalWeightAdjustments:
    """Tests 34-35: Signal weights adjust confidence."""

    def test_single_signal_weight_above_one(self):
        """Test 34: Single signal with weight 1.15 -> confidence += (1.15 - 1.0) * 0.1 = +0.015."""
        from unittest.mock import patch

        cal, mkt, sig = _fresh_managers()

        with patch.object(sig, "get_signal_weight", return_value=1.15):
            adj_prob, adj_conf, extra_edge = adjust_prediction(
                grok_probability=0.72,
                grok_confidence=0.80,
                market_type="political",
                signal_tags=[{"source_tier": "S2", "info_type": "I2"}],
                calibration_mgr=cal,
                market_type_mgr=mkt,
                signal_tracker_mgr=sig,
            )

        # Expected: 0.80 + (1.15 - 1.0) * 0.1 = 0.80 + 0.015 = 0.815
        assert adj_conf == pytest.approx(0.815, abs=1e-9)

    def test_two_signals_average_weight(self):
        """Test 35: Two signals with weights [1.10, 1.20] -> avg=1.15, same adjustment."""
        from unittest.mock import patch

        cal, mkt, sig = _fresh_managers()

        # Return 1.10 for the first call, 1.20 for the second call
        with patch.object(sig, "get_signal_weight", side_effect=[1.10, 1.20]):
            adj_prob, adj_conf, extra_edge = adjust_prediction(
                grok_probability=0.72,
                grok_confidence=0.80,
                market_type="political",
                signal_tags=[
                    {"source_tier": "S1", "info_type": "I1"},
                    {"source_tier": "S3", "info_type": "I3"},
                ],
                calibration_mgr=cal,
                market_type_mgr=mkt,
                signal_tracker_mgr=sig,
            )

        # avg_weight = (1.10 + 1.20) / 2 = 1.15
        # Expected: 0.80 + (1.15 - 1.0) * 0.1 = 0.815
        assert adj_conf == pytest.approx(0.815, abs=1e-9)


# ---------------------------------------------------------------------------
# Probability shrinkage
# ---------------------------------------------------------------------------


class TestProbabilityShrinkage:
    """Tests 37-40: Probability shrinkage toward 0.50."""

    def test_shrinkage_overconfident_bucket(self):
        """Test 37: Shrinkage when overconfident (expected_accuracy < midpoint).
        adjusted = 0.5 + (grok_prob - 0.5) * shrinkage_factor
        where shrinkage_factor = expected_accuracy / bucket_midpoint < 1.
        This pulls probability TOWARD 0.50.
        """
        cal, mkt, sig = _fresh_managers()

        # Bucket for confidence 0.75 is (0.70, 0.80), midpoint = 0.75
        bucket = cal.find_bucket(0.75)
        # Set expected_accuracy < midpoint -> shrinkage_factor < 1
        # expected_accuracy = alpha / (alpha + beta)
        # Want expected_accuracy = 0.60  => alpha/(alpha+beta) = 0.60
        # e.g. alpha = 12.0, beta = 8.0 => 12/20 = 0.60, sample_count = 18 (>= 10)
        bucket.alpha = 12.0
        bucket.beta = 8.0
        assert bucket.sample_count >= 10

        adj_prob, _, _ = adjust_prediction(
            grok_probability=0.80,
            grok_confidence=0.75,
            market_type="political",
            signal_tags=[],
            calibration_mgr=cal,
            market_type_mgr=mkt,
            signal_tracker_mgr=sig,
        )

        # shrinkage_factor = 0.60 / 0.75 = 0.80
        # adjusted = 0.5 + (0.80 - 0.5) * 0.80 = 0.5 + 0.24 = 0.74
        assert adj_prob == pytest.approx(0.74, abs=1e-9)
        # Shrunk toward 0.50 compared to original 0.80
        assert adj_prob < 0.80

    def test_shrinkage_underconfident_bucket(self):
        """Test 38: Shrinkage when underconfident (expected_accuracy > midpoint).
        shrinkage_factor > 1 expands probability AWAY from 0.50.
        """
        cal, mkt, sig = _fresh_managers()

        bucket = cal.find_bucket(0.75)
        # expected_accuracy = 0.90 => alpha/(alpha+beta) = 0.90
        # e.g. alpha = 18.0, beta = 2.0 => 18/20 = 0.90, sample_count = 18
        bucket.alpha = 18.0
        bucket.beta = 2.0
        assert bucket.sample_count >= 10

        adj_prob, _, _ = adjust_prediction(
            grok_probability=0.70,
            grok_confidence=0.75,
            market_type="political",
            signal_tags=[],
            calibration_mgr=cal,
            market_type_mgr=mkt,
            signal_tracker_mgr=sig,
        )

        # shrinkage_factor = 0.90 / 0.75 = 1.20
        # adjusted = 0.5 + (0.70 - 0.5) * 1.20 = 0.5 + 0.24 = 0.74
        assert adj_prob == pytest.approx(0.74, abs=1e-9)
        # Expanded away from 0.50 compared to original 0.70
        assert adj_prob > 0.70

    def test_shrinkage_symmetric_around_half(self):
        """Test 39: CRITICAL: Shrinkage works correctly on BOTH sides of 0.50.
        - p=0.20 (below 0.50) with overconfident bucket: shrinks TOWARD 0.50 (increases)
        - p=0.80 (above 0.50) with overconfident bucket: shrinks TOWARD 0.50 (decreases)
        """
        cal, mkt, sig = _fresh_managers()

        # Use bucket for confidence 0.75, set overconfident (expected_accuracy < midpoint)
        bucket = cal.find_bucket(0.75)
        bucket.alpha = 12.0
        bucket.beta = 8.0  # expected_accuracy = 0.60, midpoint = 0.75
        assert bucket.sample_count >= 10

        # Case A: p=0.20 (below 0.50)
        adj_prob_low, _, _ = adjust_prediction(
            grok_probability=0.20,
            grok_confidence=0.75,
            market_type="political",
            signal_tags=[],
            calibration_mgr=cal,
            market_type_mgr=mkt,
            signal_tracker_mgr=sig,
        )

        # Case B: p=0.80 (above 0.50) -- re-use same bucket since confidence is the same
        adj_prob_high, _, _ = adjust_prediction(
            grok_probability=0.80,
            grok_confidence=0.75,
            market_type="political",
            signal_tags=[],
            calibration_mgr=cal,
            market_type_mgr=mkt,
            signal_tracker_mgr=sig,
        )

        # shrinkage_factor = 0.60 / 0.75 = 0.80
        # For p=0.20: adjusted = 0.5 + (0.20 - 0.5) * 0.80 = 0.5 - 0.24 = 0.26
        # For p=0.80: adjusted = 0.5 + (0.80 - 0.5) * 0.80 = 0.5 + 0.24 = 0.74

        # p=0.20 shrinks TOWARD 0.50 -> increases
        assert adj_prob_low > 0.20
        assert adj_prob_low == pytest.approx(0.26, abs=1e-9)

        # p=0.80 shrinks TOWARD 0.50 -> decreases
        assert adj_prob_high < 0.80
        assert adj_prob_high == pytest.approx(0.74, abs=1e-9)

        # Both moved closer to 0.50
        assert abs(adj_prob_low - 0.50) < abs(0.20 - 0.50)
        assert abs(adj_prob_high - 0.50) < abs(0.80 - 0.50)

    def test_no_calibration_data_skips_shrinkage(self):
        """Test 40: No calibration data (< 10 samples) -> probability unchanged
        (shrinkage skipped).
        """
        cal, mkt, sig = _fresh_managers()

        # Bucket is fresh: alpha=1, beta=1, sample_count = 0 (< 10)
        bucket = cal.find_bucket(0.75)
        assert bucket.sample_count < 10

        adj_prob, _, _ = adjust_prediction(
            grok_probability=0.80,
            grok_confidence=0.75,
            market_type="political",
            signal_tags=[],
            calibration_mgr=cal,
            market_type_mgr=mkt,
            signal_tracker_mgr=sig,
        )

        # Probability should be unchanged -- shrinkage skipped
        assert adj_prob == 0.80


# ---------------------------------------------------------------------------
# Market type edge - new market type
# ---------------------------------------------------------------------------


class TestNewMarketTypeEdge:
    """Test 43: New market type with < 15 trades -> extra_edge = 0.0."""

    def test_new_market_type_no_edge(self, sample_trade_record):
        """Test 43: New market type (< 15 trades) -> extra_edge = 0.0."""
        cal, mkt, sig = _fresh_managers()

        # Feed only 10 trades (< 15 threshold)
        for i in range(10):
            record = sample_trade_record(
                record_id=f"rec-{i}",
                market_type="crypto_15m",
                action="BUY_YES",
                actual_outcome=True,
                brier_score_adjusted=0.40,
                voided=False,
            )
            mkt.update_market_type(record)

        perf = mkt.performances["crypto_15m"]
        assert perf.total_trades == 10
        assert perf.total_trades < 15

        _, _, extra_edge = adjust_prediction(
            grok_probability=0.72,
            grok_confidence=0.80,
            market_type="crypto_15m",
            signal_tags=[],
            calibration_mgr=cal,
            market_type_mgr=mkt,
            signal_tracker_mgr=sig,
        )

        assert extra_edge == 0.0


# ---------------------------------------------------------------------------
# Temporal confidence decay / boost (Step 5)
# ---------------------------------------------------------------------------


class TestTemporalDecayBoost:
    """Tests 44-47: Temporal confidence adjustments based on signal ages."""

    def test_recent_i1_signal_boosts_confidence(self):
        """Test 44: I1 signal < 30 min old -> confidence *= 1.05 (temporal boost)."""
        from datetime import timedelta

        cal, mkt, sig = _fresh_managers()

        now = datetime.now(timezone.utc)
        recent_ts = (now - timedelta(minutes=10)).isoformat()

        _, adj_conf, _ = adjust_prediction(
            grok_probability=0.72,
            grok_confidence=0.80,
            market_type="political",
            signal_tags=[
                {"source_tier": "S1", "info_type": "I1", "timestamp": recent_ts},
            ],
            calibration_mgr=cal,
            market_type_mgr=mkt,
            signal_tracker_mgr=sig,
        )

        # 0.80 * 1.05 = 0.84
        assert adj_conf == pytest.approx(0.84, abs=1e-9)

    def test_all_signals_over_2h_decay(self):
        """Test 45: All signals > 2h old -> confidence decays by temporal_decay formula.
        decay = max(0.85, 1.0 - 0.05 * (max_age_hours - 1.0))
        For 2.5h: decay = max(0.85, 1.0 - 0.05 * 1.5) = max(0.85, 0.925) = 0.925
        """
        from datetime import timedelta

        cal, mkt, sig = _fresh_managers()

        now = datetime.now(timezone.utc)
        old_ts = (now - timedelta(hours=2, minutes=30)).isoformat()

        _, adj_conf, _ = adjust_prediction(
            grok_probability=0.72,
            grok_confidence=0.80,
            market_type="political",
            signal_tags=[
                {"source_tier": "S2", "info_type": "I2", "timestamp": old_ts},
            ],
            calibration_mgr=cal,
            market_type_mgr=mkt,
            signal_tracker_mgr=sig,
        )

        # decay = max(0.85, 1.0 - 0.05 * (2.5 - 1.0)) = max(0.85, 0.925) = 0.925
        # 0.80 * 0.925 = 0.74
        assert adj_conf == pytest.approx(0.80 * 0.925, abs=0.01)

    def test_all_signals_over_4h_decay_floor(self):
        """Test 46: All signals > 4h old -> confidence *= 0.85 (floor).
        decay = max(0.85, 1.0 - 0.05 * (max_age_hours - 1.0))
        For 5h: decay = max(0.85, 1.0 - 0.05 * 4.0) = max(0.85, 0.80) = 0.85
        """
        from datetime import timedelta

        cal, mkt, sig = _fresh_managers()

        now = datetime.now(timezone.utc)
        very_old_ts = (now - timedelta(hours=5)).isoformat()

        _, adj_conf, _ = adjust_prediction(
            grok_probability=0.72,
            grok_confidence=0.80,
            market_type="political",
            signal_tags=[
                {"source_tier": "S2", "info_type": "I2", "timestamp": very_old_ts},
            ],
            calibration_mgr=cal,
            market_type_mgr=mkt,
            signal_tracker_mgr=sig,
        )

        # decay = max(0.85, 1.0 - 0.05 * 4.0) = max(0.85, 0.80) = 0.85
        # 0.80 * 0.85 = 0.68 -- but clamped to >= 0.50
        assert adj_conf == pytest.approx(0.80 * 0.85, abs=1e-9)

    def test_no_signal_timestamps_default_behavior(self):
        """Test 47: No signal timestamps -> default behavior (no temporal adjustment)."""
        cal, mkt, sig = _fresh_managers()

        # Signals without timestamps
        _, adj_conf_no_ts, _ = adjust_prediction(
            grok_probability=0.72,
            grok_confidence=0.80,
            market_type="political",
            signal_tags=[
                {"source_tier": "S2", "info_type": "I2"},  # no timestamp field
            ],
            calibration_mgr=cal,
            market_type_mgr=mkt,
            signal_tracker_mgr=sig,
        )

        # With no timestamps: has_recent_i1=False, max_age_hours=0.0
        # The elif branch (max_age_hours > 1.0) is False, so no decay applied.
        # Confidence should remain at base (+ any signal weight adjustment, which
        # is 0 for fresh tracker).
        assert adj_conf_no_ts == pytest.approx(0.80, abs=1e-9)
