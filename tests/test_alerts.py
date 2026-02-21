"""Tests for src/alerts.py."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.alerts import (
    TELEGRAM_API,
    format_daily_summary,
    format_error_alert,
    format_lifecycle_alert,
    format_monk_mode_alert,
    format_observe_only_alert,
    format_stale_scan_alert,
    format_tier2_alert,
    format_trade_alert,
    send_alert,
)
from src.config import Settings
from src.models import Portfolio, Position, TradeRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_settings(token: str = "tok123", chat_id: str = "42") -> Settings:
    return Settings(
        XAI_API_KEY="t",
        TWITTER_API_KEY="t",
        TELEGRAM_BOT_TOKEN=token,
        TELEGRAM_CHAT_ID=chat_id,
    )


def _make_trade(**overrides) -> TradeRecord:
    from datetime import datetime, timezone
    defaults = dict(
        record_id="r1",
        experiment_run="run1",
        timestamp=datetime.now(timezone.utc),
        model_used="grok-3-fast",
        market_id="m1",
        market_question="Will BTC hit 100k?",
        market_type="crypto",
        resolution_window_hours=24.0,
        tier=1,
        grok_raw_probability=0.70,
        grok_raw_confidence=0.80,
        grok_reasoning="reasoning",
        grok_signal_types=[],
        final_adjusted_probability=0.72,
        final_adjusted_confidence=0.78,
        market_price_at_decision=0.60,
        calculated_edge=0.10,
        trade_score=0.05,
        action="BUY_YES",
        position_size_usd=200.0,
    )
    defaults.update(overrides)
    return TradeRecord(**defaults)


# ---------------------------------------------------------------------------
# send_alert
# ---------------------------------------------------------------------------

class TestSendAlert:
    @pytest.mark.asyncio
    async def test_sends_post_when_token_set(self):
        """With a valid token and chat_id, send_alert POSTs to the Telegram API."""
        settings = _make_settings(token="bot-token", chat_id="12345")
        mock_response = MagicMock()
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("src.alerts.httpx.AsyncClient", return_value=mock_client):
            await send_alert("hello", settings)

        mock_client.post.assert_awaited_once_with(
            f"{TELEGRAM_API}/botbot-token/sendMessage",
            json={
                "chat_id": "12345",
                "text": "hello",
                "parse_mode": "HTML",
            },
        )

    @pytest.mark.asyncio
    async def test_noop_when_token_empty(self):
        """With empty token, send_alert returns immediately without making a request."""
        settings = _make_settings(token="", chat_id="12345")
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("src.alerts.httpx.AsyncClient", return_value=mock_client):
            await send_alert("hello", settings)

        mock_client.post.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_noop_when_chat_id_empty(self):
        """With empty chat_id, send_alert returns immediately without making a request."""
        settings = _make_settings(token="tok", chat_id="")
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("src.alerts.httpx.AsyncClient", return_value=mock_client):
            await send_alert("hello", settings)

        mock_client.post.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_telegram_api_error_no_exception(self):
        """When the Telegram API call raises, send_alert swallows the exception."""
        settings = _make_settings(token="tok", chat_id="42")
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.side_effect = httpx.ConnectError("connection refused")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("src.alerts.httpx.AsyncClient", return_value=mock_client):
            # Should not raise
            await send_alert("hello", settings)


# ---------------------------------------------------------------------------
# format_trade_alert
# ---------------------------------------------------------------------------

class TestFormatTradeAlert:
    def test_contains_market_question_action_edge(self):
        """format_trade_alert output includes market_question, action-derived label, and edge."""
        trade = _make_trade(
            market_question="Will ETH merge happen?",
            action="BUY_YES",
            calculated_edge=0.123,
        )
        result = format_trade_alert(trade)
        assert "Will ETH merge happen?" in result
        assert "BUY" in result  # emoji label for non-SKIP
        assert "0.123" in result

    def test_skip_action_label(self):
        """SKIP trades produce a 'SKIP' label in the alert."""
        trade = _make_trade(action="SKIP")
        result = format_trade_alert(trade)
        assert "SKIP" in result


# ---------------------------------------------------------------------------
# format_daily_summary
# ---------------------------------------------------------------------------

class TestFormatDailySummary:
    def test_includes_trade_counts_and_pnl(self):
        """format_daily_summary shows executed/skipped counts and PnL."""
        trades = [
            _make_trade(action="BUY_YES", actual_outcome=True, pnl=50.0),
            _make_trade(action="BUY_NO", actual_outcome=False, pnl=-20.0),
            _make_trade(action="SKIP", actual_outcome=None, pnl=None),
        ]
        portfolio = Portfolio(
            cash_balance=4800.0,
            total_equity=5030.0,
            open_positions=[],
        )
        result = format_daily_summary(trades, portfolio)
        assert "Executed: 2" in result
        assert "Skipped: 1" in result
        assert "Resolved: 2" in result
        # total_pnl = 50 + (-20) = 30
        assert "+30.00" in result
        assert "5,030.00" in result
        assert "4,800.00" in result


# ---------------------------------------------------------------------------
# format_error_alert
# ---------------------------------------------------------------------------

class TestFormatErrorAlert:
    def test_includes_error_message(self):
        """format_error_alert includes the error text."""
        result = format_error_alert("Something went wrong")
        assert "ERROR" in result
        assert "Something went wrong" in result

    def test_truncates_to_500_chars(self):
        """format_error_alert truncates long error messages to 500 characters."""
        long_msg = "x" * 1000
        result = format_error_alert(long_msg)
        # The format is "<b>ERROR</b>\n{error[:500]}"
        # So the error portion should be exactly 500 'x' chars
        assert "x" * 500 in result
        assert "x" * 501 not in result

# ---------------------------------------------------------------------------
# format_observe_only_alert
# ---------------------------------------------------------------------------

class TestFormatObserveOnlyAlert:
    def test_contains_observe_only_heading(self):
        """format_observe_only_alert includes the OBSERVE-ONLY heading."""
        result = format_observe_only_alert(executed=5, cap=5)
        assert "OBSERVE-ONLY" in result

    def test_shows_executed_and_cap(self):
        """format_observe_only_alert displays the executed/cap ratio."""
        result = format_observe_only_alert(executed=3, cap=10)
        assert "3/10" in result

    def test_returns_html_bold(self):
        """format_observe_only_alert wraps heading in HTML bold tags."""
        result = format_observe_only_alert(executed=1, cap=5)
        assert "<b>OBSERVE-ONLY</b>" in result

    def test_mentions_observe_only_mode(self):
        """format_observe_only_alert mentions switching to observe-only mode."""
        result = format_observe_only_alert(executed=5, cap=5)
        assert "observe-only mode" in result


# ---------------------------------------------------------------------------
# format_monk_mode_alert
# ---------------------------------------------------------------------------

class TestFormatMonkModeAlert:
    def test_contains_monk_mode_heading(self):
        """format_monk_mode_alert includes the MONK MODE heading."""
        result = format_monk_mode_alert("daily_loss_limit")
        assert "MONK MODE" in result

    def test_daily_loss_limit_reason(self):
        """format_monk_mode_alert maps daily_loss_limit to a human-readable label."""
        result = format_monk_mode_alert("daily_loss_limit")
        assert "Daily loss limit reached" in result

    def test_weekly_loss_limit_reason(self):
        """format_monk_mode_alert maps weekly_loss_limit to a human-readable label."""
        result = format_monk_mode_alert("weekly_loss_limit")
        assert "Weekly loss limit reached" in result

    def test_max_total_exposure_reason(self):
        """format_monk_mode_alert maps max_total_exposure to a human-readable label."""
        result = format_monk_mode_alert("max_total_exposure")
        assert "Max total exposure reached" in result

    def test_api_budget_exceeded_reason(self):
        """format_monk_mode_alert maps api_budget_exceeded to a human-readable label."""
        result = format_monk_mode_alert("api_budget_exceeded")
        assert "Daily API budget exceeded" in result

    def test_consecutive_adverse_reason(self):
        """format_monk_mode_alert handles consecutive_adverse_3 with count."""
        result = format_monk_mode_alert("consecutive_adverse_3")
        assert "Consecutive adverse" in result
        assert "3" in result

    def test_tier1_daily_cap_reason(self):
        """format_monk_mode_alert handles tier daily cap reasons."""
        result = format_monk_mode_alert("tier1_daily_cap")
        assert "daily cap" in result.lower()

    def test_trade_blocked_message(self):
        """format_monk_mode_alert includes trade blocked message."""
        result = format_monk_mode_alert("daily_loss_limit")
        assert "Trade blocked" in result

    def test_unknown_reason_falls_through(self):
        """format_monk_mode_alert uses raw reason if not in known labels."""
        result = format_monk_mode_alert("some_unknown_reason")
        assert "some_unknown_reason" in result


# ---------------------------------------------------------------------------
# format_tier2_alert
# ---------------------------------------------------------------------------

class TestFormatTier2Alert:
    def test_activated_message(self):
        """format_tier2_alert returns TIER 2 ACTIVATED when active=True."""
        result = format_tier2_alert(active=True)
        assert "TIER 2 ACTIVATED" in result

    def test_deactivated_message(self):
        """format_tier2_alert returns TIER 2 DEACTIVATED when active=False."""
        result = format_tier2_alert(active=False)
        assert "TIER 2 DEACTIVATED" in result

    def test_activated_mentions_crypto(self):
        """format_tier2_alert activated message mentions crypto news."""
        result = format_tier2_alert(active=True)
        assert "Crypto news" in result or "crypto" in result.lower()

    def test_deactivated_mentions_stopping(self):
        """format_tier2_alert deactivated message mentions stopping Tier 2."""
        result = format_tier2_alert(active=False)
        assert "Stopping Tier 2" in result or "stopping" in result.lower()


# ---------------------------------------------------------------------------
# format_lifecycle_alert
# ---------------------------------------------------------------------------

class TestFormatLifecycleAlert:
    def test_startup_event(self):
        """format_lifecycle_alert shows BOT STARTUP for startup event."""
        result = format_lifecycle_alert(event="startup", environment="production")
        assert "BOT STARTUP" in result

    def test_shutdown_event(self):
        """format_lifecycle_alert shows BOT SHUTDOWN for shutdown event."""
        result = format_lifecycle_alert(event="shutdown", environment="staging")
        assert "BOT SHUTDOWN" in result

    def test_includes_environment(self):
        """format_lifecycle_alert includes the environment/mode in output."""
        result = format_lifecycle_alert(event="startup", environment="production")
        assert "production" in result

    def test_event_uppercased(self):
        """format_lifecycle_alert uppercases the event string."""
        result = format_lifecycle_alert(event="restart", environment="dev")
        assert "BOT RESTART" in result

    def test_mode_label(self):
        """format_lifecycle_alert includes Mode label."""
        result = format_lifecycle_alert(event="startup", environment="paper")
        assert "Mode: paper" in result


# ---------------------------------------------------------------------------
# format_stale_scan_alert
# ---------------------------------------------------------------------------

class TestFormatStaleScanAlert:
    def test_contains_stale_scan_heading(self):
        """format_stale_scan_alert includes the STALE SCAN WARNING heading."""
        result = format_stale_scan_alert(minutes_since=15.0)
        assert "STALE SCAN WARNING" in result

    def test_shows_minutes(self):
        """format_stale_scan_alert displays the number of minutes."""
        result = format_stale_scan_alert(minutes_since=15.0)
        assert "15" in result

    def test_rounds_minutes(self):
        """format_stale_scan_alert rounds fractional minutes to whole number."""
        result = format_stale_scan_alert(minutes_since=7.6)
        assert "8" in result

    def test_check_bot_health_message(self):
        """format_stale_scan_alert includes a message about checking bot health."""
        result = format_stale_scan_alert(minutes_since=20.0)
        assert "Check bot health" in result

    def test_html_bold_heading(self):
        """format_stale_scan_alert wraps heading in HTML bold tags."""
        result = format_stale_scan_alert(minutes_since=5.0)
        assert "<b>STALE SCAN WARNING</b>" in result
