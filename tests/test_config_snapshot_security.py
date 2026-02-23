"""Tests for config snapshot secret redaction."""
from __future__ import annotations

import pytest
from src.config import Settings, _SECRET_FIELDS


class TestSafeConfig:
    def test_safe_config_redacts_all_secrets(self):
        settings = Settings(
            XAI_API_KEY="xai-secret-123",
            TWITTER_API_KEY="tw-secret-456",
            POLYMARKET_API_KEY="pm-key",
            POLYMARKET_SECRET="pm-secret",
            POLYMARKET_PASSPHRASE="pm-pass",
            TELEGRAM_BOT_TOKEN="tg-token-789",
            TELEGRAM_CHAT_ID="-100123",
            _env_file=None,
        )
        safe = settings.safe_config()
        for field in _SECRET_FIELDS:
            assert safe[field] == "***REDACTED***", f"{field} was not redacted"

    def test_safe_config_preserves_non_secrets(self):
        settings = Settings(
            XAI_API_KEY="test",
            TWITTER_API_KEY="test",
            _env_file=None,
        )
        safe = settings.safe_config()
        assert safe["ENVIRONMENT"] == "paper"
        assert safe["DB_PATH"] == "data/predictor.db"
        assert safe["TIER1_FEE_RATE"] == 0.0
        assert safe["INITIAL_BANKROLL"] == 10000.0

    def test_safe_config_preserves_telegram_chat_id(self):
        settings = Settings(
            XAI_API_KEY="test",
            TWITTER_API_KEY="test",
            TELEGRAM_CHAT_ID="-100999",
            _env_file=None,
        )
        safe = settings.safe_config()
        assert safe["TELEGRAM_CHAT_ID"] == "-100999"  # NOT redacted

    def test_secret_fields_is_frozen(self):
        assert isinstance(_SECRET_FIELDS, frozenset)
        assert len(_SECRET_FIELDS) == 6
