"""Tests for Scheduler.should_activate_tier2 with fully mocked external services."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models import Signal
from src.config import Settings, MonkModeConfig
from src.scheduler import Scheduler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_signal(
    content: str = "bitcoin is rising",
    source_tier: str = "S3",
    followers: int = 50_000,
    **kwargs,
) -> Signal:
    defaults = dict(
        source="rss",
        source_tier=source_tier,
        info_type=None,
        content=content,
        credibility=0.80,
        author="TestAuthor",
        followers=followers,
        engagement=100,
        timestamp=datetime.now(timezone.utc),
        headline_only=False,
    )
    defaults.update(kwargs)
    return Signal(**defaults)


def _build_scheduler() -> Scheduler:
    """Build a Scheduler with every external dependency mocked out."""
    settings = MagicMock(spec=Settings)
    settings.TIER1_SCAN_INTERVAL_MINUTES = 15
    settings.TIER2_SCAN_INTERVAL_MINUTES = 3
    settings.TIER1_MIN_EDGE = 0.04
    settings.TIER2_MIN_EDGE = 0.05
    settings.TIER1_FEE_RATE = 0.02
    settings.TIER2_FEE_RATE = 0.04
    settings.KELLY_FRACTION = 0.25
    settings.MAX_POSITION_PCT = 0.08
    settings.ENVIRONMENT = "paper"
    settings.TIER1_DAILY_CAP = 5
    settings.TIER2_DAILY_CAP = 3
    settings.DAILY_LOSS_LIMIT_PCT = 0.05
    settings.WEEKLY_LOSS_LIMIT_PCT = 0.10
    settings.CONSECUTIVE_LOSS_COOLDOWN = 3
    settings.COOLDOWN_DURATION_HOURS = 2.0
    settings.DAILY_API_BUDGET_USD = 8.0
    settings.MAX_TOTAL_EXPOSURE_PCT = 0.30

    db = AsyncMock()
    polymarket = AsyncMock()
    twitter = AsyncMock()
    rss = AsyncMock()
    grok = AsyncMock()
    calibration_mgr = MagicMock()
    market_type_mgr = MagicMock()
    signal_tracker_mgr = MagicMock()

    # MonkModeConfig.from_settings reads attributes from Settings -- patch it
    with patch.object(MonkModeConfig, "from_settings", return_value=MonkModeConfig()):
        scheduler = Scheduler(
            settings=settings,
            db=db,
            polymarket=polymarket,
            twitter=twitter,
            rss=rss,
            grok=grok,
            calibration_mgr=calibration_mgr,
            market_type_mgr=market_type_mgr,
            signal_tracker_mgr=signal_tracker_mgr,
        )
    return scheduler


# ---------------------------------------------------------------------------
# Tests: should_activate_tier2
# ---------------------------------------------------------------------------


class TestShouldActivateTier2:
    """Tier-2 activation requires 2+ crypto-relevant signals with at least
    one from S1/S2 or 100K+ followers."""

    def test_true_with_two_crypto_signals_and_s1_source(self):
        """Two crypto signals, one from S1 => activate."""
        scheduler = _build_scheduler()
        signals = [
            _make_signal(content="bitcoin breaks 100k", source_tier="S1", followers=5000),
            _make_signal(content="eth rally continues", source_tier="S4", followers=2000),
        ]
        assert scheduler.should_activate_tier2(signals) is True

    def test_true_with_two_crypto_signals_and_s2_source(self):
        """Two crypto signals, one from S2 => activate."""
        scheduler = _build_scheduler()
        signals = [
            _make_signal(content="crypto market pump", source_tier="S2", followers=3000),
            _make_signal(content="solana new high", source_tier="S5", followers=1000),
        ]
        assert scheduler.should_activate_tier2(signals) is True

    def test_true_with_two_crypto_signals_and_100k_followers(self):
        """Two crypto signals, one with 100K+ followers => activate."""
        scheduler = _build_scheduler()
        signals = [
            _make_signal(content="btc pump", source_tier="S4", followers=150_000),
            _make_signal(content="ethereum flipping", source_tier="S4", followers=500),
        ]
        assert scheduler.should_activate_tier2(signals) is True

    def test_true_with_exactly_100k_followers(self):
        """Boundary: exactly 100,000 followers should activate."""
        scheduler = _build_scheduler()
        signals = [
            _make_signal(content="btc analysis", source_tier="S6", followers=100_000),
            _make_signal(content="crypto crash incoming", source_tier="S6", followers=50),
        ]
        assert scheduler.should_activate_tier2(signals) is True

    def test_false_with_only_one_crypto_signal(self):
        """Only 1 crypto-relevant signal => do NOT activate."""
        scheduler = _build_scheduler()
        signals = [
            _make_signal(content="bitcoin halving coming", source_tier="S1", followers=200_000),
            _make_signal(content="weather forecast looks nice", source_tier="S1", followers=200_000),
        ]
        assert scheduler.should_activate_tier2(signals) is False

    def test_false_with_zero_crypto_signals(self):
        """No crypto signals at all => do NOT activate."""
        scheduler = _build_scheduler()
        signals = [
            _make_signal(content="stock market rally", source_tier="S1", followers=500_000),
            _make_signal(content="political news update", source_tier="S2", followers=300_000),
        ]
        assert scheduler.should_activate_tier2(signals) is False

    def test_false_with_two_crypto_signals_all_s6_and_low_followers(self):
        """Two crypto signals but all S6 and < 100K followers => no authority."""
        scheduler = _build_scheduler()
        signals = [
            _make_signal(content="btc price moving", source_tier="S6", followers=5_000),
            _make_signal(content="ethereum update news", source_tier="S6", followers=80_000),
        ]
        assert scheduler.should_activate_tier2(signals) is False

    def test_false_with_empty_signals(self):
        """Empty signals list => do NOT activate."""
        scheduler = _build_scheduler()
        assert scheduler.should_activate_tier2([]) is False

    def test_mixed_signals_only_two_crypto_qualify(self):
        """Mix of crypto and non-crypto signals; only crypto ones count."""
        scheduler = _build_scheduler()
        signals = [
            _make_signal(content="bitcoin surge", source_tier="S2", followers=1000),
            _make_signal(content="the economy is doing well", source_tier="S1", followers=500_000),
            _make_signal(content="solana ecosystem growing", source_tier="S4", followers=2000),
        ]
        assert scheduler.should_activate_tier2(signals) is True

    def test_three_crypto_signals_with_authority(self):
        """3 crypto signals with S1 authority => activate."""
        scheduler = _build_scheduler()
        signals = [
            _make_signal(content="btc going up", source_tier="S1", followers=10),
            _make_signal(content="eth going up", source_tier="S5", followers=10),
            _make_signal(content="sol going up", source_tier="S6", followers=10),
        ]
        assert scheduler.should_activate_tier2(signals) is True
