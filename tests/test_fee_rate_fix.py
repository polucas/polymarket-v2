"""Tests for TIER1_FEE_RATE correction and edge calculation impact."""
from __future__ import annotations

import pytest

from src.config import Settings
from src.engine.trade_decision import calculate_edge


class TestTier1FeeRateDefault:
    """Verify TIER1_FEE_RATE defaults to 0.0 (fee-free markets)."""

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
