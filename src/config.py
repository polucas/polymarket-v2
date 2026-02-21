from dataclasses import dataclass
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # API Keys
    XAI_API_KEY: str = ""
    TWITTER_API_KEY: str = ""
    POLYMARKET_API_KEY: str = ""
    POLYMARKET_SECRET: str = ""
    POLYMARKET_PASSPHRASE: str = ""
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""

    # Environment
    ENVIRONMENT: str = "paper"  # "paper" or "live"
    DB_PATH: str = "data/predictor.db"
    LOG_LEVEL: str = "INFO"

    # Tier 1 Config
    TIER1_SCAN_INTERVAL_MINUTES: int = 15
    TIER1_MIN_EDGE: float = 0.04
    TIER1_DAILY_CAP: int = 5
    TIER1_FEE_RATE: float = 0.02

    # Tier 2 Config
    TIER2_SCAN_INTERVAL_MINUTES: int = 3
    TIER2_MIN_EDGE: float = 0.05
    TIER2_DAILY_CAP: int = 3
    TIER2_FEE_RATE: float = 0.04

    # Monk Mode
    DAILY_LOSS_LIMIT_PCT: float = 0.05
    WEEKLY_LOSS_LIMIT_PCT: float = 0.10
    CONSECUTIVE_LOSS_COOLDOWN: int = 3
    COOLDOWN_DURATION_HOURS: float = 2.0
    DAILY_API_BUDGET_USD: float = 8.0
    MAX_POSITION_PCT: float = 0.08
    MAX_TOTAL_EXPOSURE_PCT: float = 0.30
    KELLY_FRACTION: float = 0.25
    MAX_CLUSTER_EXPOSURE_PCT: float = 0.12

    # Alerts
    DAILY_SUMMARY_HOUR_UTC: int = 0  # Hour of day (UTC) to send daily summary

    # Initial Bankroll
    INITIAL_BANKROLL: float = 2000.0

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@dataclass
class MonkModeConfig:
    tier1_daily_trade_cap: int = 5
    tier2_daily_trade_cap: int = 3
    daily_loss_limit_pct: float = 0.05
    weekly_loss_limit_pct: float = 0.10
    consecutive_loss_cooldown: int = 3
    cooldown_duration_hours: float = 2.0
    daily_api_budget_usd: float = 8.0
    max_position_pct: float = 0.08
    max_total_exposure_pct: float = 0.30
    kelly_fraction: float = 0.25

    @classmethod
    def from_settings(cls, s: Settings) -> "MonkModeConfig":
        return cls(
            tier1_daily_trade_cap=s.TIER1_DAILY_CAP,
            tier2_daily_trade_cap=s.TIER2_DAILY_CAP,
            daily_loss_limit_pct=s.DAILY_LOSS_LIMIT_PCT,
            weekly_loss_limit_pct=s.WEEKLY_LOSS_LIMIT_PCT,
            consecutive_loss_cooldown=s.CONSECUTIVE_LOSS_COOLDOWN,
            cooldown_duration_hours=s.COOLDOWN_DURATION_HOURS,
            daily_api_budget_usd=s.DAILY_API_BUDGET_USD,
            max_position_pct=s.MAX_POSITION_PCT,
            max_total_exposure_pct=s.MAX_TOTAL_EXPOSURE_PCT,
            kelly_fraction=s.KELLY_FRACTION,
        )


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
