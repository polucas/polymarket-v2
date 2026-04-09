"""Tests for fee rate configuration and edge calculation impact."""
from __future__ import annotations

import pytest

from src.config import Settings
from src.engine.trade_decision import calculate_edge
from src.pipelines.market_classifier import MARKET_TYPE_FEES, get_fee_rate


class TestTier1FeeRateDefault:
    """Verify TIER1_FEE_RATE defaults to 0.0 (fallback for unmapped types)."""

    def test_settings_default_tier1_fee_is_zero(self):
        """Settings class should default TIER1_FEE_RATE to 0.0."""
        settings = Settings(
            XAI_API_KEY="test",
            TWITTER_API_KEY="test",
            _env_file=None,
        )
        assert settings.TIER1_FEE_RATE == 0.0

    def test_settings_tier2_fee_unchanged(self):
        """TIER2_FEE_RATE should remain 0.04 (not affected by fix)."""
        settings = Settings(
            XAI_API_KEY="test",
            TWITTER_API_KEY="test",
            _env_file=None,
        )
        assert settings.TIER2_FEE_RATE == 0.04


class TestMarketTypeFees:
    """Verify MARKET_TYPE_FEES and get_fee_rate() reflect Q1 2026 fee schedule."""

    def test_political_fee_is_zero(self):
        """Political markets are still fee-free."""
        assert MARKET_TYPE_FEES["political"] == 0.0

    def test_geopolitical_fee_is_zero(self):
        """Geopolitical markets are still fee-free."""
        assert MARKET_TYPE_FEES["geopolitical"] == 0.0

    def test_crypto_15m_fee_nonzero(self):
        """Crypto markets now have a real taker fee."""
        assert get_fee_rate("crypto_15m") > 0.0

    def test_crypto_15m_fee_value(self):
        """Crypto peak taker fee is 1.56%."""
        assert get_fee_rate("crypto_15m") == pytest.approx(0.0156)

    def test_get_fee_rate_returns_correct_value(self):
        """get_fee_rate returns the MARKET_TYPE_FEES entry for known types."""
        assert get_fee_rate("political") == 0.0
        assert get_fee_rate("sports") == pytest.approx(0.02)
        assert get_fee_rate("unknown") == pytest.approx(0.02)

    def test_get_fee_rate_unmapped_uses_default(self):
        """An unmapped market type uses the caller-supplied default."""
        assert get_fee_rate("brand_new_type", default=0.03) == pytest.approx(0.03)

    def test_get_fee_rate_unmapped_uses_builtin_default(self):
        """An unmapped market type falls back to 0.02 if no default is given."""
        assert get_fee_rate("brand_new_type") == pytest.approx(0.02)


class TestEdgeCalculationWithZeroFee:
    """Verify edge calculation with correct 0% Tier 1 fee."""

    def test_edge_with_zero_fee(self):
        """Edge should NOT have phantom fee subtracted for Tier 1."""
        edge = calculate_edge(adjusted_prob=0.55, market_price=0.50, fee_rate=0.0)
        assert edge == pytest.approx(0.05, abs=1e-9)

    def test_edge_with_old_incorrect_fee(self):
        """Demonstrate the bug: 2% fee reduces a 5% edge to 3%."""
        edge = calculate_edge(adjusted_prob=0.55, market_price=0.50, fee_rate=0.02)
        assert edge == pytest.approx(0.03, abs=1e-9)

    def test_edge_threshold_pass_with_zero_fee(self):
        """A 5% raw edge should pass the 4% threshold with 0% fee."""
        edge = calculate_edge(adjusted_prob=0.55, market_price=0.50, fee_rate=0.0)
        assert edge >= 0.04

    def test_edge_threshold_fail_with_phantom_fee(self):
        """Same 5% raw edge fails the 4% threshold with phantom 2% fee."""
        edge = calculate_edge(adjusted_prob=0.55, market_price=0.50, fee_rate=0.02)
        assert edge < 0.04

    def test_tier2_fee_still_applied(self):
        """Tier 2 edge calculation should still subtract 4% fee."""
        edge = calculate_edge(adjusted_prob=0.60, market_price=0.50, fee_rate=0.04)
        assert edge == pytest.approx(0.06, abs=1e-9)


class TestEnvExampleDefaults:
    """Verify .env.example has correct defaults."""

    def test_env_example_tier1_fee_rate(self):
        with open(".env.example") as f:
            content = f.read()
        assert "TIER1_FEE_RATE=0.0" in content

    def test_env_example_initial_bankroll(self):
        with open(".env.example") as f:
            content = f.read()
        assert "INITIAL_BANKROLL=2000.0" in content
