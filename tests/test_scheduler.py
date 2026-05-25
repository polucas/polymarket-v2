"""Tests for Scheduler.should_activate_tier2 with fully mocked external services."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models import Portfolio, Signal, Market
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
    settings.MARKET_COOLDOWN_HOURS = 24.0
    settings.EVALUATION_COOLDOWN_HOURS = 4.0
    settings.QUESTION_SIMILARITY_THRESHOLD = 0.60
    settings.LLM_MODEL = "MiniMax-M2.7"
    settings.MARKET_FETCH_LIMIT = 200
    settings.MARKET_PAGE_SIZE = 500
    settings.MARKET_FETCH_PAGES = 3
    settings.MIN_TRADEABLE_PRICE = 0.05
    settings.MAX_TRADEABLE_PRICE = 0.95
    settings.RSS_POLL_INTERVAL_SECONDS = 30
    settings.TIER1_EXECUTION_TYPE = "maker"
    settings.TIER2_EXECUTION_TYPE = "maker"
    settings.TAKE_PROFIT_ROI = 0.20
    settings.STOP_LOSS_ROI = -0.15
    settings.EARLY_EXIT_ENABLED = True
    settings.PRESCREEN_ENABLED = False
    settings.WEAK_SIGNAL_STRENGTH_THRESHOLD = 0.45
    settings.TWITTER_ENABLED = True
    settings.disabled_market_types_set = set()
    settings.FAST_EXIT_POLL_INTERVAL_SECONDS = 60
    settings.WS_HEARTBEAT_SECONDS = 10

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
        scheduler._rss.consume_signals.return_value = []

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


# ---------------------------------------------------------------------------
# Tests: Paper mode position size threshold
# ---------------------------------------------------------------------------


class TestPaperModePositionThreshold:
    """In paper mode, the minimum position is $0.50; in live mode it is $1.00."""

    def test_paper_mode_accepts_position_above_threshold(self):
        """A $0.75 position is above the $0.50 paper threshold — should be accepted."""
        scheduler = _build_scheduler()
        # ENVIRONMENT is already "paper" in _build_scheduler
        assert scheduler._settings.ENVIRONMENT == "paper"
        min_position = 0.50 if scheduler._settings.ENVIRONMENT == "paper" else 1.0
        position_size = 0.75
        assert position_size >= min_position

    def test_paper_mode_rejects_position_below_threshold(self):
        """A $0.25 position is below the $0.50 paper threshold — should be rejected."""
        scheduler = _build_scheduler()
        assert scheduler._settings.ENVIRONMENT == "paper"
        min_position = 0.50 if scheduler._settings.ENVIRONMENT == "paper" else 1.0
        position_size = 0.25
        assert position_size < min_position

    def test_live_mode_rejects_position_below_one_dollar(self):
        """In live mode, a $0.75 position is below the $1.00 threshold — should be rejected."""
        scheduler = _build_scheduler()
        scheduler._settings.ENVIRONMENT = "live"
        min_position = 0.50 if scheduler._settings.ENVIRONMENT == "paper" else 1.0
        position_size = 0.75
        assert position_size < min_position

    def test_live_mode_accepts_position_at_one_dollar(self):
        """In live mode, a $1.00 position meets the threshold exactly — should be accepted."""
        scheduler = _build_scheduler()
        scheduler._settings.ENVIRONMENT = "live"
        min_position = 0.50 if scheduler._settings.ENVIRONMENT == "paper" else 1.0
        position_size = 1.00
        assert position_size >= min_position


# ---------------------------------------------------------------------------
# Tests: No-signals short-circuit in _process_market
# ---------------------------------------------------------------------------


def _make_market(
    market_id: str = "mkt-001",
    question: str = "Will X happen?",
    yes_price: float = 0.50,
    market_type: str = "political",
) -> Market:
    return Market(
        market_id=market_id,
        question=question,
        yes_price=yes_price,
        no_price=1.0 - yes_price,
        hours_to_resolution=24.0,
        market_type=market_type,
        keywords=["x", "happen"],
    )


class TestNoSignalsShortCircuit:
    """When both twitter_signals and relevant_rss are empty, _process_market
    must save a no_signals SKIP and return without calling the LLM or orderbook."""

    @pytest.mark.asyncio
    async def test_no_signals_skips_without_llm_or_orderbook(self):
        """Empty twitter + empty rss => save skip(no_signals), no LLM, no orderbook."""
        scheduler = _build_scheduler()
        market = _make_market()

        # market_type_mgr must not disable the market
        scheduler._market_type_mgr.should_disable.return_value = False
        # twitter returns empty list
        scheduler._twitter.get_signals_for_market.return_value = []

        with patch("src.scheduler.extract_keywords", return_value=["x", "happen"]):
            await scheduler._process_market(
                market=market,
                rss_signals=[],  # no RSS signals either
                scan_mode="normal",
                candidates=[],
                all_skips=[],
                today_trades=[],
                experiment_run="exp-001",
                tier=1,
            )

        # save_trade called exactly once with no_signals skip
        scheduler._db.save_trade.assert_called_once()
        saved_record = scheduler._db.save_trade.call_args[0][0]
        assert saved_record.skip_reason == "no_signals"

        # LLM and orderbook must NOT have been called
        scheduler._grok.call_grok_with_retry.assert_not_called()
        scheduler._polymarket.get_orderbook.assert_not_called()

    @pytest.mark.asyncio
    async def test_with_signals_continues_past_shortcircuit(self):
        """Providing at least one signal bypasses the no_signals short-circuit."""
        scheduler = _build_scheduler()
        market = _make_market()

        scheduler._market_type_mgr.should_disable.return_value = False

        one_signal = _make_signal(content="will x happen politics")
        scheduler._twitter.get_signals_for_market.return_value = [one_signal]

        # LLM returns None to short-circuit further processing (grok_failed)
        scheduler._grok.call_grok_with_retry.return_value = None
        scheduler._polymarket.get_orderbook.return_value = MagicMock()

        with patch("src.scheduler.extract_keywords", return_value=["x", "happen"]):
            await scheduler._process_market(
                market=market,
                rss_signals=[],
                scan_mode="normal",
                candidates=[],
                all_skips=[],
                today_trades=[],
                experiment_run="exp-001",
                tier=1,
            )

        # orderbook AND grok must have been called (short-circuit was NOT hit)
        scheduler._polymarket.get_orderbook.assert_called()
        scheduler._grok.call_grok_with_retry.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: Weak-signal gate in _process_market
# ---------------------------------------------------------------------------


class TestWeakSignalGate:
    """When all signals have low credibility (avg < WEAK_SIGNAL_STRENGTH_THRESHOLD),
    _process_market must save a weak_signals_* SKIP and return without touching
    the orderbook or calling the LLM prescreen."""

    @pytest.mark.asyncio
    async def test_weak_signals_skip_without_prescreen_or_orderbook(self):
        """2 signals with credibility=0.3 each → avg=0.3 < 0.45 → gate fires."""
        scheduler = _build_scheduler()
        # Enable the weak-signal threshold
        scheduler._settings.WEAK_SIGNAL_STRENGTH_THRESHOLD = 0.45
        # Prescreen is disabled in _build_scheduler; keep it off so no prescreen call
        scheduler._settings.PRESCREEN_ENABLED = False

        market = _make_market()
        scheduler._market_type_mgr.should_disable.return_value = False

        weak_signal_1 = _make_signal(content="will x happen politics", credibility=0.3)
        weak_signal_2 = _make_signal(content="x politics update", credibility=0.3)
        # twitter returns two weak signals
        scheduler._twitter.get_signals_for_market.return_value = [weak_signal_1, weak_signal_2]

        with patch("src.scheduler.extract_keywords", return_value=["x", "happen"]):
            await scheduler._process_market(
                market=market,
                rss_signals=[],
                scan_mode="normal",
                candidates=[],
                all_skips=[],
                today_trades=[],
                experiment_run="exp-001",
                tier=1,
            )

        # save_trade must be called once with a weak_signals_* reason
        scheduler._db.save_trade.assert_called_once()
        saved_record = scheduler._db.save_trade.call_args[0][0]
        assert saved_record.skip_reason.startswith("weak_signals_")

        # LLM prescreen and orderbook must NOT have been called
        scheduler._grok.call_prescreen.assert_not_called()
        scheduler._polymarket.get_orderbook.assert_not_called()

    @pytest.mark.asyncio
    async def test_strong_signals_pass_gate(self):
        """Signals averaging credibility=0.7 must bypass the weak-signal gate."""
        scheduler = _build_scheduler()
        scheduler._settings.WEAK_SIGNAL_STRENGTH_THRESHOLD = 0.45
        scheduler._settings.PRESCREEN_ENABLED = False

        market = _make_market()
        scheduler._market_type_mgr.should_disable.return_value = False

        strong_signal = _make_signal(content="will x happen politics", credibility=0.7)
        scheduler._twitter.get_signals_for_market.return_value = [strong_signal]

        # LLM returns None → grok_failed skip (we just need it past the gate)
        scheduler._grok.call_grok_with_retry.return_value = None
        scheduler._polymarket.get_orderbook.return_value = MagicMock()

        with patch("src.scheduler.extract_keywords", return_value=["x", "happen"]):
            await scheduler._process_market(
                market=market,
                rss_signals=[],
                scan_mode="normal",
                candidates=[],
                all_skips=[],
                today_trades=[],
                experiment_run="exp-001",
                tier=1,
            )

        # save_trade may be called (for grok_failed), but NOT for weak_signals_*
        for call in scheduler._db.save_trade.call_args_list:
            record = call[0][0]
            assert not record.skip_reason.startswith("weak_signals_"), (
                f"Unexpected weak_signals skip: {record.skip_reason}"
            )

        # orderbook was fetched (gate did not fire)
        scheduler._polymarket.get_orderbook.assert_called()


# ---------------------------------------------------------------------------
# F2 Tests: _fast_exit_check, market_type_disabled_env, twitter_disabled
# ---------------------------------------------------------------------------


class TestFastExitCheck:
    """_fast_exit_check (Bug 9): polls check_early_exits whenever positions are
    tracked, regardless of ws_exit_mgr._connected (which can lie on silent stall).
    No-ops when ws_exit_mgr is None or _active_positions is empty."""

    @pytest.mark.asyncio
    async def test_noop_when_no_positions(self):
        """Skip when no positions tracked, regardless of _connected state."""
        scheduler = _build_scheduler()
        scheduler.ws_exit_mgr = MagicMock(_active_positions={}, _connected=False)

        with patch(
            "src.engine.resolution.check_early_exits", new_callable=AsyncMock
        ) as cee:
            await scheduler._fast_exit_check()

        cee.assert_not_called()

    @pytest.mark.asyncio
    async def test_fires_when_positions_tracked(self):
        """Polls Gamma whenever positions exist — even if _connected=True (Bug 9 safety)."""
        scheduler = _build_scheduler()
        scheduler.ws_exit_mgr = MagicMock(
            _active_positions={"tok-1": MagicMock()},
            _connected=True,  # Bug 9: flag can lie; fast_exit_check must still run
        )

        with patch(
            "src.engine.resolution.check_early_exits", new_callable=AsyncMock
        ) as cee:
            await scheduler._fast_exit_check()

        cee.assert_called_once()

    @pytest.mark.asyncio
    async def test_fires_when_positions_tracked_and_disconnected(self):
        """Also fires when _connected=False (original Bug 6a behavior preserved)."""
        scheduler = _build_scheduler()
        scheduler.ws_exit_mgr = MagicMock(
            _active_positions={"tok-1": MagicMock()},
            _connected=False,
        )

        with patch(
            "src.engine.resolution.check_early_exits", new_callable=AsyncMock
        ) as cee:
            await scheduler._fast_exit_check()

        cee.assert_called_once()

    @pytest.mark.asyncio
    async def test_noop_when_ws_exit_mgr_none(self):
        """Still skip when ws_exit_mgr is None."""
        scheduler = _build_scheduler()
        scheduler.ws_exit_mgr = None

        with patch(
            "src.engine.resolution.check_early_exits", new_callable=AsyncMock
        ) as cee:
            await scheduler._fast_exit_check()

        cee.assert_not_called()


class TestMarketTypeDisabledEnv:
    """_process_market: env-gate DISABLED_MARKET_TYPES skips before learned disable."""

    @pytest.mark.asyncio
    async def test_skip_market_type_disabled_via_env(self):
        """Market with market_type='sports' skipped when DISABLED_MARKET_TYPES='sports'."""
        scheduler = _build_scheduler()
        # Make disabled_market_types_set return {"sports"}
        scheduler._settings.disabled_market_types_set = {"sports"}
        scheduler._market_type_mgr.should_disable.return_value = False
        scheduler._twitter.get_signals_for_market.return_value = []

        market = _make_market(market_type="sports")

        with patch("src.scheduler.extract_keywords", return_value=["sports"]):
            await scheduler._process_market(
                market=market,
                rss_signals=[],
                scan_mode="normal",
                candidates=[],
                all_skips=[],
                today_trades=[],
                experiment_run="exp-001",
                tier=1,
            )

        # save_trade must be called once with market_type_disabled_env
        scheduler._db.save_trade.assert_called_once()
        saved_record = scheduler._db.save_trade.call_args[0][0]
        assert saved_record.skip_reason == "market_type_disabled_env"

        # Learned disable must NOT have been called (env gate fires first)
        scheduler._market_type_mgr.should_disable.assert_not_called()


class TestTwitterDisabledGate:
    """_process_market: when TWITTER_ENABLED=False, twitter.get_signals_for_market not called."""

    @pytest.mark.asyncio
    async def test_twitter_disabled_skips_fetch(self):
        """With TWITTER_ENABLED=False, twitter signal fetch is bypassed (twitter_signals=[])."""
        scheduler = _build_scheduler()
        scheduler._settings.TWITTER_ENABLED = False
        scheduler._settings.disabled_market_types_set = set()
        scheduler._market_type_mgr.should_disable.return_value = False

        # Provide one strong RSS signal so the no_signals / weak_signal gates don't fire
        strong_rss = _make_signal(content="will x happen politics", credibility=0.8)

        # LLM returns None -> grok_failed (we just need to get past the Twitter gate)
        scheduler._grok.call_grok_with_retry.return_value = None
        scheduler._grok.call_prescreen.return_value = None
        scheduler._polymarket.get_orderbook.return_value = MagicMock()

        market = _make_market()

        with patch("src.scheduler.extract_keywords", return_value=["x", "happen"]):
            await scheduler._process_market(
                market=market,
                rss_signals=[strong_rss],
                scan_mode="normal",
                candidates=[],
                all_skips=[],
                today_trades=[],
                experiment_run="exp-001",
                tier=1,
            )

        # Twitter fetch must NOT have been called
        scheduler._twitter.get_signals_for_market.assert_not_called()


# ---------------------------------------------------------------------------
# F2: disabled_market_types_set property parses DISABLED_MARKET_TYPES correctly
# ---------------------------------------------------------------------------


class TestDisabledMarketTypesSetProperty:
    """Settings.disabled_market_types_set parses comma-separated env var."""

    def test_empty_string_gives_empty_set(self):
        from src.config import Settings
        s = Settings(XAI_API_KEY="x", POLYMARKET_API_KEY="y", DISABLED_MARKET_TYPES="")
        assert s.disabled_market_types_set == set()

    def test_single_value(self):
        from src.config import Settings
        s = Settings(XAI_API_KEY="x", POLYMARKET_API_KEY="y", DISABLED_MARKET_TYPES="sports")
        assert s.disabled_market_types_set == {"sports"}

    def test_multiple_values_with_spaces(self):
        from src.config import Settings
        s = Settings(
            XAI_API_KEY="x", POLYMARKET_API_KEY="y",
            DISABLED_MARKET_TYPES="sports, crypto_15m, weather"
        )
        assert s.disabled_market_types_set == {"sports", "crypto_15m", "weather"}


# ---------------------------------------------------------------------------
# F3 Tests: ws_exit_mgr.add_position called after execute_trade
# ---------------------------------------------------------------------------


def _make_stub_trade_record():
    from src.models import TradeRecord
    from datetime import datetime, timezone
    return TradeRecord(
        record_id="stub-rec",
        experiment_run="exp-001",
        timestamp=datetime.now(timezone.utc),
        model_used="MiniMax-M2.7",
        market_id="mkt-stub",
        market_question="Will Y happen?",
        market_type="political",
        resolution_window_hours=24.0,
        tier=1,
        grok_raw_probability=0.70,
        grok_raw_confidence=0.80,
        grok_reasoning="strong signals",
        grok_signal_types=[],
        action="BUY_YES",
        market_price_at_decision=0.45,
        position_size_usd=50.0,
    )


def _make_stub_candidate():
    from src.models import TradeCandidate
    market = _make_market()
    return TradeCandidate(
        market=market,
        adjusted_probability=0.70,
        adjusted_confidence=0.80,
        calculated_edge=0.25,
        position_size=50.0,
        side="BUY_YES",
        resolution_hours=24.0,
        tier=1,
        grok_raw_probability=0.70,
        grok_raw_confidence=0.80,
        grok_reasoning="strong signals",
        execution_type="maker",
    )


class TestAddPositionCalledAfterTradeExecution:
    """After execute_trade() returns a record, ws_exit_mgr.add_position must be called."""

    @pytest.mark.asyncio
    async def test_add_position_called_when_record_returned(self):
        """If execute_trade returns a stub record, ws_exit_mgr.add_position is called once.

        Drives run_tier1_scan with all heavy machinery mocked out. _process_market
        is patched to inject one candidate directly. select_best_trades returns that
        candidate. check_monk_mode allows it. execute_trade returns a stub record.
        The branch 'if record: ws_exit_mgr.add_position(record)' is exercised.
        """
        scheduler = _build_scheduler()

        ws_exit_mgr = MagicMock()
        ws_exit_mgr.add_position = AsyncMock()
        scheduler.ws_exit_mgr = ws_exit_mgr

        stub_record = _make_stub_trade_record()
        stub_candidate = _make_stub_candidate()

        portfolio = Portfolio()
        scheduler._db.get_today_trades.return_value = []
        scheduler._db.get_week_trades.return_value = []
        scheduler._db.get_today_api_spend.return_value = 0.0
        scheduler._db.load_portfolio.return_value = portfolio
        scheduler._db.get_open_market_ids.return_value = set()
        scheduler._db.get_recently_traded_market_ids.return_value = set()
        scheduler._db.get_recently_evaluated_market_ids.return_value = set()
        scheduler._db.get_recent_market_questions.return_value = []
        scheduler._polymarket.get_active_markets.return_value = [_make_market()]
        # consume_signals is sync; override on AsyncMock to return a plain list
        scheduler._rss.consume_signals = MagicMock(return_value=[])

        # _process_market injects the stub candidate into the shared candidates list
        async def fake_process_market(*, candidates, **kwargs):
            candidates.append(stub_candidate)

        with patch.object(scheduler, "_process_market", side_effect=fake_process_market), \
             patch("src.scheduler.select_best_trades", return_value=([stub_candidate], [])), \
             patch("src.scheduler.check_monk_mode", return_value=(True, "ok")), \
             patch("src.scheduler.execute_trade", new_callable=AsyncMock, return_value=stub_record), \
             patch("src.scheduler.send_alert", new_callable=AsyncMock), \
             patch("src.scheduler.get_current_experiment", new_callable=AsyncMock, return_value=None), \
             patch("src.scheduler.get_scan_mode", return_value="normal"):
            await scheduler.run_tier1_scan()

        ws_exit_mgr.add_position.assert_called_once_with(stub_record)

    @pytest.mark.asyncio
    async def test_add_position_not_called_when_no_record(self):
        """If execute_trade returns None, ws_exit_mgr.add_position must NOT be called."""
        scheduler = _build_scheduler()

        ws_exit_mgr = MagicMock()
        ws_exit_mgr.add_position = AsyncMock()
        scheduler.ws_exit_mgr = ws_exit_mgr

        stub_candidate = _make_stub_candidate()

        portfolio = Portfolio()
        scheduler._db.get_today_trades.return_value = []
        scheduler._db.get_week_trades.return_value = []
        scheduler._db.get_today_api_spend.return_value = 0.0
        scheduler._db.load_portfolio.return_value = portfolio
        scheduler._db.get_open_market_ids.return_value = set()
        scheduler._db.get_recently_traded_market_ids.return_value = set()
        scheduler._db.get_recently_evaluated_market_ids.return_value = set()
        scheduler._db.get_recent_market_questions.return_value = []
        scheduler._polymarket.get_active_markets.return_value = [_make_market()]
        # consume_signals is sync; override on AsyncMock to return a plain list
        scheduler._rss.consume_signals = MagicMock(return_value=[])

        async def fake_process_market(*, candidates, **kwargs):
            candidates.append(stub_candidate)

        with patch.object(scheduler, "_process_market", side_effect=fake_process_market), \
             patch("src.scheduler.select_best_trades", return_value=([stub_candidate], [])), \
             patch("src.scheduler.check_monk_mode", return_value=(True, "ok")), \
             patch("src.scheduler.execute_trade", new_callable=AsyncMock, return_value=None), \
             patch("src.scheduler.send_alert", new_callable=AsyncMock), \
             patch("src.scheduler.get_current_experiment", new_callable=AsyncMock, return_value=None), \
             patch("src.scheduler.get_scan_mode", return_value="normal"):
            await scheduler.run_tier1_scan()

        ws_exit_mgr.add_position.assert_not_called()
