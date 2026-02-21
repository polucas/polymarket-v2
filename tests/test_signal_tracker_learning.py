"""Tests for SignalTrackerManager (learning layer 3)."""

import pytest

from src.learning.signal_tracker import SignalTrackerManager


# ---------------------------------------------------------------------------
# update_signal_trackers
# ---------------------------------------------------------------------------


class TestUpdateSignalTrackers:
    """Tests 17-19: SignalTrackerManager.update_signal_trackers."""

    def test_correct_outcome_present_signal_increments_present_winning(self, sample_trade_record):
        """Test 17: Correct outcome with present signal -> present_in_winning_trades incremented."""
        mgr = SignalTrackerManager()
        record = sample_trade_record(
            market_type="political",
            grok_signal_types=[{"source_tier": "S2", "info_type": "I2"}],
            final_adjusted_probability=0.80,  # > 0.5 => predicts YES
            actual_outcome=True,              # YES => correct
            voided=False,
        )
        mgr.update_signal_trackers(record)

        tracker = mgr.trackers[("S2", "I2", "political")]
        assert tracker.present_in_winning_trades == 1
        assert tracker.present_in_losing_trades == 0

    def test_incorrect_outcome_present_signal_increments_present_losing(self, sample_trade_record):
        """Test 18: Incorrect outcome with present signal -> present_in_losing_trades incremented."""
        mgr = SignalTrackerManager()
        record = sample_trade_record(
            market_type="political",
            grok_signal_types=[{"source_tier": "S3", "info_type": "I1"}],
            final_adjusted_probability=0.80,  # > 0.5 => predicts YES
            actual_outcome=False,             # NO => incorrect
            voided=False,
        )
        mgr.update_signal_trackers(record)

        tracker = mgr.trackers[("S3", "I1", "political")]
        assert tracker.present_in_losing_trades == 1
        assert tracker.present_in_winning_trades == 0

    def test_uses_adjusted_correctness(self, sample_trade_record):
        """Test 19: Correctness is determined by final_adjusted_probability (> 0.5),
        NOT grok_raw_probability."""
        mgr = SignalTrackerManager()
        # Raw says NO (0.40), but adjusted says YES (0.60)
        record = sample_trade_record(
            market_type="political",
            grok_signal_types=[{"source_tier": "S1", "info_type": "I3"}],
            grok_raw_probability=0.40,
            final_adjusted_probability=0.60,  # adjusted > 0.5 => predicts YES
            actual_outcome=True,              # YES => correct (per adjusted)
            voided=False,
        )
        mgr.update_signal_trackers(record)

        tracker = mgr.trackers[("S1", "I3", "political")]
        # If raw were used, prediction would be NO (0.40 < 0.5) and outcome=True
        # would mean incorrect. But since adjusted is used, it's correct.
        assert tracker.present_in_winning_trades == 1
        assert tracker.present_in_losing_trades == 0


# ---------------------------------------------------------------------------
# get_signal_weight
# ---------------------------------------------------------------------------


class TestGetSignalWeight:
    """Tests 20-21: SignalTrackerManager.get_signal_weight."""

    def test_unknown_combo_returns_1_0(self):
        """Test 20: get_signal_weight for unknown (source_tier, info_type, market_type) -> 1.0."""
        mgr = SignalTrackerManager()
        weight = mgr.get_signal_weight("S1", "I1", "political")
        assert weight == 1.0

    def test_known_combo_returns_computed_weight(self):
        """Test 21: get_signal_weight for known combo returns the tracker's weight property."""
        mgr = SignalTrackerManager()
        tracker = mgr._ensure("S2", "I2", "political")

        # Set enough data so lift calculation is meaningful
        # Need total_present >= 5 and total_absent >= 5
        tracker.present_in_winning_trades = 8
        tracker.present_in_losing_trades = 2  # total_present = 10, win_rate = 0.8
        tracker.absent_in_winning_trades = 5
        tracker.absent_in_losing_trades = 5   # total_absent = 10, win_rate = 0.5

        # lift = 0.8 / 0.5 = 1.6
        # raw_weight = 1.0 + (1.6 - 1.0) * 0.3 = 1.0 + 0.18 = 1.18
        # clamped to [0.8, 1.2] => 1.18
        expected_weight = tracker.weight
        result = mgr.get_signal_weight("S2", "I2", "political")
        assert result == expected_weight
        assert abs(result - 1.18) < 0.01
