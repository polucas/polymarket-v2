"""Tests for CalibrationManager (learning layer 1)."""

import uuid
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio

from src.learning.calibration import CalibrationManager
from src.models import CALIBRATION_BUCKET_RANGES


# ---------------------------------------------------------------------------
# Bucket lookup
# ---------------------------------------------------------------------------


class TestFindBucket:
    """Tests 1-3: CalibrationManager.find_bucket returns the correct bucket."""

    def test_find_bucket_055(self):
        """Test 1: 0.55 -> (0.50, 0.60)."""
        mgr = CalibrationManager()
        bucket = mgr.find_bucket(0.55)
        assert bucket is not None
        assert bucket.bucket_range == (0.50, 0.60)

    def test_find_bucket_065(self):
        """Test 2: 0.65 -> (0.60, 0.70)."""
        mgr = CalibrationManager()
        bucket = mgr.find_bucket(0.65)
        assert bucket is not None
        assert bucket.bucket_range == (0.60, 0.70)

    def test_find_bucket_097(self):
        """Test 3: 0.97 -> (0.95, 1.00)."""
        mgr = CalibrationManager()
        bucket = mgr.find_bucket(0.97)
        assert bucket is not None
        assert bucket.bucket_range == (0.95, 1.00)


# ---------------------------------------------------------------------------
# get_correction
# ---------------------------------------------------------------------------


class TestGetCorrection:
    """Tests 4-5: CalibrationManager.get_correction."""

    def test_fresh_bucket_returns_zero(self):
        """Test 4: Fresh bucket (alpha=1, beta=1, sample_count=0) -> 0.0."""
        mgr = CalibrationManager()
        correction = mgr.get_correction(0.75)
        assert correction == 0.0

    def test_below_min_samples_returns_zero(self):
        """Test 5: Bucket with 9 samples (< 10 minimum) -> 0.0."""
        mgr = CalibrationManager()
        bucket = mgr.find_bucket(0.75)
        assert bucket is not None
        # Add 9 samples: alpha starts at 1, adding 9 correct -> alpha=10, beta=1.
        # sample_count = (10 + 1) - 2 = 9.
        for _ in range(9):
            bucket.update(was_correct=True, recency_weight=1.0)
        assert bucket.sample_count == 9
        correction = mgr.get_correction(0.75)
        assert correction == 0.0


# ---------------------------------------------------------------------------
# update_calibration
# ---------------------------------------------------------------------------


class TestUpdateCalibration:
    """Tests 6-7: update_calibration uses RAW probability, not adjusted."""

    def test_uses_raw_confidence_for_bucket(self, sample_trade_record):
        """Test 6: update_calibration uses grok_raw_confidence for bucket lookup,
        NOT final_adjusted_confidence."""
        mgr = CalibrationManager()
        record = sample_trade_record(
            grok_raw_confidence=0.75,       # -> bucket (0.70, 0.80)
            final_adjusted_confidence=0.85,  # -> would be (0.80, 0.90) if used
            grok_raw_probability=0.80,
            actual_outcome=True,
            voided=False,
        )
        mgr.update_calibration(record)

        # The (0.70, 0.80) bucket should have been updated, not (0.80, 0.90)
        bucket_70_80 = mgr.find_bucket(0.75)
        bucket_80_90 = mgr.find_bucket(0.85)
        assert bucket_70_80.alpha > 1.0  # was updated
        assert bucket_80_90.alpha == 1.0  # untouched

    def test_raw_probability_determines_bucket_not_adjusted(self, sample_trade_record):
        """Test 7 (CRITICAL): Trade with raw=0.75, adjusted=0.85.
        Verify the update goes to bucket (0.70, 0.80) based on raw_confidence,
        and correctness is determined by raw_probability (0.75 > 0.5 => predicts YES).
        """
        mgr = CalibrationManager()
        record = sample_trade_record(
            grok_raw_probability=0.75,
            grok_raw_confidence=0.75,       # bucket (0.70, 0.80)
            final_adjusted_probability=0.85,
            final_adjusted_confidence=0.85,  # bucket (0.80, 0.90) -- NOT used
            actual_outcome=True,            # YES outcome
            voided=False,
        )
        mgr.update_calibration(record)

        # raw_predicted_yes = 0.75 > 0.5 => True; actual_outcome = True => was_correct = True
        bucket = mgr.find_bucket(0.75)
        assert bucket.bucket_range == (0.70, 0.80)
        assert bucket.alpha > 1.0  # correct prediction increments alpha

        # (0.80, 0.90) bucket should be untouched
        wrong_bucket = mgr.find_bucket(0.85)
        assert wrong_bucket.alpha == 1.0
        assert wrong_bucket.beta == 1.0


# ---------------------------------------------------------------------------
# Recency weighting
# ---------------------------------------------------------------------------


class TestRecencyWeight:
    """Test 8: Recency weight is applied as 0.95^days."""

    def test_recency_weight_applied(self, sample_trade_record):
        """Test 8: A trade from 10 days ago gets weight 0.95^10."""
        mgr = CalibrationManager()
        ten_days_ago = datetime.now(timezone.utc) - timedelta(days=10)
        record = sample_trade_record(
            grok_raw_confidence=0.75,
            grok_raw_probability=0.80,
            actual_outcome=True,
            timestamp=ten_days_ago,
            voided=False,
        )
        mgr.update_calibration(record)

        bucket = mgr.find_bucket(0.75)
        expected_weight = 0.95 ** 10
        # alpha should be 1.0 + weight (correct prediction)
        assert abs(bucket.alpha - (1.0 + expected_weight)) < 0.01


# ---------------------------------------------------------------------------
# reset_to_priors
# ---------------------------------------------------------------------------


class TestResetToPriors:
    """Test 9: reset_to_priors sets all buckets back to alpha=1, beta=1."""

    def test_reset_to_priors(self, sample_trade_record):
        """Test 9: After updating, reset_to_priors restores alpha=1, beta=1."""
        mgr = CalibrationManager()
        record = sample_trade_record(
            grok_raw_confidence=0.75,
            grok_raw_probability=0.80,
            actual_outcome=True,
            voided=False,
        )
        mgr.update_calibration(record)

        # Confirm something was changed
        bucket = mgr.find_bucket(0.75)
        assert bucket.alpha > 1.0

        mgr.reset_to_priors()

        for b in mgr.buckets:
            assert b.alpha == 1.0, f"Bucket {b.bucket_range} alpha not reset"
            assert b.beta == 1.0, f"Bucket {b.bucket_range} beta not reset"


# ---------------------------------------------------------------------------
# Voided trades
# ---------------------------------------------------------------------------


class TestVoidedTrades:
    """Test 10: Voided trades are skipped by update_calibration."""

    def test_voided_trade_skipped(self, sample_trade_record):
        """Test 10: A voided trade should not update any bucket."""
        mgr = CalibrationManager()
        record = sample_trade_record(
            grok_raw_confidence=0.75,
            grok_raw_probability=0.80,
            actual_outcome=True,
            voided=True,
        )
        mgr.update_calibration(record)

        # All buckets should be untouched
        for b in mgr.buckets:
            assert b.alpha == 1.0
            assert b.beta == 1.0


# ---------------------------------------------------------------------------
# Additional bucket boundary tests
# ---------------------------------------------------------------------------


class TestFindBucketBoundaries:
    """Tests for bucket boundary values."""

    def test_find_bucket_095_boundary(self):
        """Test 3 (boundary): 0.95 is exactly the lower bound of (0.95, 1.00).
        find_bucket uses <= confidence < upper, so 0.95 falls in (0.95, 1.00)."""
        mgr = CalibrationManager()
        bucket = mgr.find_bucket(0.95)
        assert bucket is not None
        assert bucket.bucket_range == (0.95, 1.00)

    def test_find_bucket_050_boundary(self):
        """Test 5 (boundary): 0.50 is exactly the lower bound of (0.50, 0.60).
        find_bucket uses <= confidence < upper, so 0.50 falls in (0.50, 0.60)."""
        mgr = CalibrationManager()
        bucket = mgr.find_bucket(0.50)
        assert bucket is not None
        assert bucket.bucket_range == (0.50, 0.60)


# ---------------------------------------------------------------------------
# Test 8: Positive correction when expected_accuracy > midpoint
# ---------------------------------------------------------------------------


class TestPositiveCorrection:
    """Test 8: Bucket with 15+ samples, expected_accuracy > midpoint -> positive correction."""

    def test_positive_correction_when_accuracy_above_midpoint(self):
        """Test 8: A bucket with enough samples where expected_accuracy exceeds
        the midpoint should produce a positive correction."""
        mgr = CalibrationManager()
        bucket = mgr.find_bucket(0.75)
        assert bucket is not None
        assert bucket.bucket_range == (0.70, 0.80)

        # Add 15 correct predictions to push expected_accuracy above midpoint.
        # After 15 correct: alpha = 1 + 15 = 16, beta = 1
        # expected_accuracy = 16 / 17 ≈ 0.941
        # midpoint = (0.70 + 0.80) / 2 = 0.75
        # So expected_accuracy (0.941) > midpoint (0.75) -> positive correction
        for _ in range(15):
            bucket.update(was_correct=True, recency_weight=1.0)

        assert bucket.sample_count >= 10  # (16 + 1 - 2) = 15
        correction = bucket.get_correction()
        assert correction > 0.0


# ---------------------------------------------------------------------------
# Test 10 (additional): update_calibration increments beta for incorrect prediction
# ---------------------------------------------------------------------------


class TestUpdateCalibrationBetaIncrement:
    """Test 10: update_calibration with outcome=False for a YES-predicting raw prob
    increments beta in the correct bucket."""

    def test_incorrect_prediction_increments_beta(self, sample_trade_record):
        """Test 10: Trade with grok_raw_probability=0.75 (predicts YES),
        outcome=False -> prediction is wrong -> beta incremented in bucket (0.70, 0.80)."""
        mgr = CalibrationManager()
        record = sample_trade_record(
            grok_raw_probability=0.75,
            grok_raw_confidence=0.75,  # bucket (0.70, 0.80)
            actual_outcome=False,      # YES prediction was wrong
            voided=False,
        )
        mgr.update_calibration(record)

        bucket = mgr.find_bucket(0.75)
        assert bucket.bucket_range == (0.70, 0.80)
        # raw_predicted_yes = 0.75 > 0.5 => True; actual_outcome = False => was_correct = False
        # beta should be incremented (alpha stays at 1.0)
        assert bucket.alpha == 1.0
        assert bucket.beta > 1.0


# ---------------------------------------------------------------------------
# Test 14: After reset_to_priors, all get_correction() returns 0.0
# ---------------------------------------------------------------------------


class TestResetToPriorsCorrection:
    """Test 14: After reset_to_priors, all get_correction() calls return 0.0."""

    def test_all_corrections_zero_after_reset(self, sample_trade_record):
        """Test 14: After populating buckets and then resetting to priors,
        every bucket should return correction=0.0."""
        mgr = CalibrationManager()

        # Add many samples to multiple buckets to ensure non-zero corrections
        for conf in [0.55, 0.65, 0.75, 0.85, 0.92]:
            for i in range(20):
                record = sample_trade_record(
                    record_id=f"rec-{conf}-{i}",
                    grok_raw_probability=conf + 0.02,
                    grok_raw_confidence=conf,
                    actual_outcome=True,
                    voided=False,
                )
                mgr.update_calibration(record)

        mgr.reset_to_priors()

        # After reset, all buckets should be back to alpha=1, beta=1
        # which means sample_count < 10, so get_correction returns 0.0
        for br_low, br_high in CALIBRATION_BUCKET_RANGES:
            mid = (br_low + br_high) / 2
            assert mgr.get_correction(mid) == 0.0
