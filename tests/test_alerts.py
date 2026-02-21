"""Tests for src/alerts.py."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.alerts import (
    TELEGRAM_API,
    format_daily_summary,
    format_error_alert,
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
