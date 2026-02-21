import pytest
from src.config import Settings, MonkModeConfig


class TestSettings:
    def test_defaults(self):
        s = Settings(XAI_API_KEY="test", TWITTER_API_KEY="test")
        assert s.ENVIRONMENT == "paper"
        assert s.LOG_LEVEL == "INFO"
        assert s.DB_PATH == "data/predictor.db"

    def test_numeric_types(self):
        s = Settings(XAI_API_KEY="t", TWITTER_API_KEY="t")
        assert isinstance(s.TIER1_MIN_EDGE, float)
        assert isinstance(s.TIER1_DAILY_CAP, int)
        assert isinstance(s.KELLY_FRACTION, float)

    def test_kelly_default(self):
        s = Settings(XAI_API_KEY="t", TWITTER_API_KEY="t")
        assert s.KELLY_FRACTION == 0.25

    def test_max_cluster_default(self):
        s = Settings(XAI_API_KEY="t", TWITTER_API_KEY="t")
        assert s.MAX_CLUSTER_EXPOSURE_PCT == 0.12

    def test_initial_bankroll_default(self):
        s = Settings(XAI_API_KEY="t", TWITTER_API_KEY="t")
        assert s.INITIAL_BANKROLL == 2000.0


class TestMonkModeConfig:
    def test_from_settings(self):
        s = Settings(XAI_API_KEY="t", TWITTER_API_KEY="t")
        m = MonkModeConfig.from_settings(s)
        assert m.tier1_daily_trade_cap == 5
        assert m.tier2_daily_trade_cap == 3
        assert m.daily_loss_limit_pct == 0.05
        assert m.weekly_loss_limit_pct == 0.10
        assert m.consecutive_loss_cooldown == 3
        assert m.cooldown_duration_hours == 2.0
        assert m.daily_api_budget_usd == 8.0
        assert m.max_position_pct == 0.08
        assert m.max_total_exposure_pct == 0.30
        assert m.kelly_fraction == 0.25
