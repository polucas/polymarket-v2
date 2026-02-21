"""Tests for Scheduler.should_activate_tier2 with fully mocked external services."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models import Portfolio, Signal
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

# ---------------------------------------------------------------------------
# Tests: Daily summary alert
# ---------------------------------------------------------------------------


class TestDailySummaryJob:
    """Verify _send_daily_summary calls send_alert with a Daily Summary message."""

    @pytest.mark.asyncio
    async def test_send_daily_summary_calls_send_alert(self):
        scheduler = _build_scheduler()
        scheduler._db.get_today_trades.return_value = []
        scheduler._db.load_portfolio.return_value = Portfolio()

        with patch("src.scheduler.send_alert", new_callable=AsyncMock) as mock_alert:
            await scheduler._send_daily_summary()

            mock_alert.assert_called_once()
            message = mock_alert.call_args[0][0]
            assert "Daily Summary" in message


# ---------------------------------------------------------------------------
# Tests: Stale scan alert
# ---------------------------------------------------------------------------


class TestStaleScanAlert:
    """Verify _check_stale_scan fires alert only when last_scan_completed is >30 min ago."""

    @pytest.mark.asyncio
    async def test_alert_sent_when_scan_is_stale(self):
        """If last_scan_completed is >30 min ago, send_alert should be called."""
        scheduler = _build_scheduler()
        scheduler.last_scan_completed = datetime.now(timezone.utc) - timedelta(minutes=45)

        with patch("src.scheduler.send_alert", new_callable=AsyncMock) as mock_alert:
            await scheduler._check_stale_scan()

            mock_alert.assert_called_once()
            message = mock_alert.call_args[0][0]
            assert "STALE SCAN" in message

    @pytest.mark.asyncio
    async def test_no_alert_when_scan_is_recent(self):
        """If last_scan_completed is <30 min ago, send_alert should NOT be called."""
        scheduler = _build_scheduler()
        scheduler.last_scan_completed = datetime.now(timezone.utc) - timedelta(minutes=10)

        with patch("src.scheduler.send_alert", new_callable=AsyncMock) as mock_alert:
            await scheduler._check_stale_scan()

            mock_alert.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_alert_when_last_scan_is_none(self):
        """If last_scan_completed is None (still initializing), no alert."""
        scheduler = _build_scheduler()
        scheduler.last_scan_completed = None

        with patch("src.scheduler.send_alert", new_callable=AsyncMock) as mock_alert:
            await scheduler._check_stale_scan()

            mock_alert.assert_not_called()

# ---------------------------------------------------------------------------
# Tests: Tier 2 activation / deactivation alerts
# ---------------------------------------------------------------------------


class TestTier2Alerts:
    """Verify _activate_tier2 and _deactivate_tier2 send correct alerts."""

    @pytest.mark.asyncio
    async def test_activate_tier2_sends_activated_alert(self):
        scheduler = _build_scheduler()

        with patch("src.scheduler.send_alert", new_callable=AsyncMock) as mock_alert:
            await scheduler._activate_tier2()

            mock_alert.assert_called_once()
            message = mock_alert.call_args[0][0]
            assert "ACTIVATED" in message

    @pytest.mark.asyncio
    async def test_deactivate_tier2_sends_deactivated_alert(self):
        scheduler = _build_scheduler()
        # Must be active first so deactivation proceeds
        scheduler._tier2_active = True

        with patch("src.scheduler.send_alert", new_callable=AsyncMock) as mock_alert:
            await scheduler._deactivate_tier2()

            mock_alert.assert_called_once()
            message = mock_alert.call_args[0][0]
            assert "DEACTIVATED" in message

    @pytest.mark.asyncio
    async def test_activate_tier2_noop_when_already_active(self):
        """Calling _activate_tier2 when already active should not send alert."""
        scheduler = _build_scheduler()
        scheduler._tier2_active = True

        with patch("src.scheduler.send_alert", new_callable=AsyncMock) as mock_alert:
            await scheduler._activate_tier2()

            mock_alert.assert_not_called()

    @pytest.mark.asyncio
    async def test_deactivate_tier2_noop_when_already_inactive(self):
        """Calling _deactivate_tier2 when already inactive should not send alert."""
        scheduler = _build_scheduler()
        scheduler._tier2_active = False

        with patch("src.scheduler.send_alert", new_callable=AsyncMock) as mock_alert:
            await scheduler._deactivate_tier2()

            mock_alert.assert_not_called()

# ---------------------------------------------------------------------------
# Tests: Error alert wiring in _auto_resolve
# ---------------------------------------------------------------------------


class TestErrorAlertWiring:
    """Verify that when _auto_resolve raises, an error alert is sent."""

    @pytest.mark.asyncio
    async def test_auto_resolve_error_sends_alert(self):
        scheduler = _build_scheduler()

        with patch(
            "src.scheduler.auto_resolve_trades",
            new_callable=AsyncMock,
            side_effect=RuntimeError("db connection lost"),
        ), patch("src.scheduler.send_alert", new_callable=AsyncMock) as mock_alert:
            await scheduler._auto_resolve()

            mock_alert.assert_called_once()
            message = mock_alert.call_args[0][0]
            assert "ERROR" in message
            assert "db connection lost" in message


# ---------------------------------------------------------------------------
# Tests: Observe-only alert fires once, not twice
# ---------------------------------------------------------------------------


class TestObserveOnlyAlert:
    """Verify the observe-only alert fires once per day, not on every scan."""

    @pytest.mark.asyncio
    async def test_observe_only_alert_fires_once(self):
        scheduler = _build_scheduler()

        # Build a fake trade that counts as tier-1 executed (action != "SKIP")
        fake_trade = MagicMock()
        fake_trade.tier = 1
        fake_trade.action = "BUY"

        # Return enough trades to exceed the cap so get_scan_mode returns "observe_only"
        trades = [fake_trade] * 10
        scheduler._db.get_today_trades.return_value = trades
        scheduler._db.get_week_trades.return_value = []
        scheduler._db.get_today_api_spend.return_value = 0.0
        scheduler._db.load_portfolio.return_value = Portfolio()

        # Mock away everything else in run_tier1_scan that we don't care about
        scheduler._polymarket.get_active_markets.return_value = []
        scheduler._rss.get_breaking_news.return_value = []

        with patch(
            "src.scheduler.get_scan_mode", return_value="observe_only"
        ), patch(
            "src.scheduler.send_alert", new_callable=AsyncMock
        ) as mock_alert, patch(
            "src.scheduler.get_current_experiment", new_callable=AsyncMock, return_value=None
        ):
            # First scan: alert should fire
            await scheduler.run_tier1_scan()
            assert mock_alert.call_count == 1
            message = mock_alert.call_args[0][0]
            assert "OBSERVE-ONLY" in message

            # Second scan: same day, alert should NOT fire again
            mock_alert.reset_mock()
            await scheduler.run_tier1_scan()
            assert mock_alert.call_count == 0
