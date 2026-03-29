"""Tests for the backtest Clock singleton."""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from src.backtest.clock import Clock


@pytest.fixture(autouse=True)
def reset_clock():
    """Ensure Clock is in live mode before and after every test."""
    Clock.reset()
    yield
    Clock.reset()


class TestClockLiveMode:
    def test_live_mode_returns_real_time(self):
        """When not simulated, utcnow() returns approximately current wall time."""
        before = datetime.now(timezone.utc)
        t = Clock.utcnow()
        after = datetime.now(timezone.utc)
        assert before <= t <= after

    def test_is_simulated_false_in_live_mode(self):
        assert Clock.is_simulated() is False


class TestClockSimulatedMode:
    def test_set_time_and_read(self):
        dt = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        Clock.set_time(dt)
        assert Clock.utcnow() == dt

    def test_is_simulated_true_after_set(self):
        Clock.set_time(datetime(2025, 1, 1, tzinfo=timezone.utc))
        assert Clock.is_simulated() is True

    def test_naive_datetime_gets_utc(self):
        """Naive datetime passed to set_time gets UTC tzinfo attached."""
        naive = datetime(2025, 6, 1, 10, 0, 0)
        Clock.set_time(naive)
        result = Clock.utcnow()
        assert result.tzinfo is not None
        assert result == datetime(2025, 6, 1, 10, 0, 0, tzinfo=timezone.utc)

    def test_advance_increments_time(self):
        start = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        Clock.set_time(start)
        Clock.advance(15)
        expected = start + timedelta(minutes=15)
        assert Clock.utcnow() == expected

    def test_advance_multiple_steps(self):
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        Clock.set_time(start)
        Clock.advance(15)
        Clock.advance(15)
        Clock.advance(15)
        assert Clock.utcnow() == start + timedelta(minutes=45)

    def test_reset_returns_to_live_mode(self):
        Clock.set_time(datetime(2025, 1, 1, tzinfo=timezone.utc))
        assert Clock.is_simulated() is True
        Clock.reset()
        assert Clock.is_simulated() is False
        # After reset, should return real time again
        now_real = datetime.now(timezone.utc)
        t = Clock.utcnow()
        assert abs((t - now_real).total_seconds()) < 5  # within 5s

    def test_advance_from_unset_clock(self):
        """advance() when not yet set should start from real time."""
        before = datetime.now(timezone.utc)
        Clock.advance(60)
        after = Clock.utcnow()
        # Should be approximately before + 60 minutes
        assert after >= before + timedelta(minutes=59)

    def test_time_is_frozen_without_advance(self):
        """utcnow() returns same value if not advanced."""
        dt = datetime(2025, 3, 15, 9, 0, 0, tzinfo=timezone.utc)
        Clock.set_time(dt)
        t1 = Clock.utcnow()
        t2 = Clock.utcnow()
        assert t1 == t2
