from dataclasses import dataclass
from pydantic_settings import BaseSettings, SettingsConfigDict

_SECRET_FIELDS = frozenset({
    "XAI_API_KEY", "MINIMAX_API_KEY", "TWITTER_API_KEY", "POLYMARKET_API_KEY",
    "POLYMARKET_SECRET", "POLYMARKET_PASSPHRASE", "TELEGRAM_BOT_TOKEN",
    "POLYMARKET_PRIVATE_KEY", "POLYMARKET_FUNDER_ADDRESS",
})


class Settings(BaseSettings):
    # API Keys
    XAI_API_KEY: str = ""
    MINIMAX_API_KEY: str = ""
    TWITTER_API_KEY: str = ""
    POLYMARKET_API_KEY: str = ""
    POLYMARKET_SECRET: str = ""      # L2 ApiCreds api_secret — used by ClobClient for CLOB authentication
    POLYMARKET_PASSPHRASE: str = ""  # L2 ApiCreds api_passphrase — used by ClobClient for CLOB authentication
    POLYMARKET_PRIVATE_KEY: str = ""          # Wallet private key for signing CLOB orders (0x-prefixed hex)
    POLYMARKET_FUNDER_ADDRESS: str = ""       # Proxy/funder wallet address (for smart wallets)
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""

    # Environment
    ENVIRONMENT: str = "paper"  # "paper" or "live"
    DB_PATH: str = "data/predictor.db"
    LOG_LEVEL: str = "INFO"
    LLM_MODEL: str = "MiniMax-M2.7"

    # RSS Polling
    RSS_POLL_INTERVAL_SECONDS: int = 30

    # Tier 1 Config
    TIER1_SCAN_INTERVAL_MINUTES: int = 15
    TIER1_MIN_EDGE: float = 0.03
    TIER1_DAILY_CAP: int = 20
    TIER1_FEE_RATE: float = 0.0
    MARKET_FETCH_LIMIT: int = 200
    MARKET_PAGE_SIZE: int = 100          # Markets per API page (Polymarket Gamma caps at 100, larger values silently truncated)
    MARKET_FETCH_PAGES: int = 10         # Number of offset-paginated pages to fetch (100 * 10 = 1000 markets/scan)
    TIER1_EXECUTION_TYPE: str = "maker"

    # Tier 2 Config
    TIER2_SCAN_INTERVAL_MINUTES: int = 3
    TIER2_MIN_EDGE: float = 0.05
    TIER2_DAILY_CAP: int = 3
    TIER2_FEE_RATE: float = 0.04
    TIER2_EXECUTION_TYPE: str = "maker"

    # Early Exit
    TAKE_PROFIT_ROI: float = 0.20
    STOP_LOSS_ROI: float = -0.15
    EARLY_EXIT_ENABLED: bool = True
    FAST_EXIT_POLL_INTERVAL_SECONDS: int = 60
    WS_HEARTBEAT_SECONDS: int = 10
    MIN_BID_LIQUIDITY_USD: float = 5.0  # Skip WS exit if best-bid liquidity below this (Bug 7b — flash empty-book guard)

    # Monk Mode
    DAILY_LOSS_LIMIT_PCT: float = 0.05
    WEEKLY_LOSS_LIMIT_PCT: float = 0.10
    CONSECUTIVE_LOSS_COOLDOWN: int = 3
    COOLDOWN_DURATION_HOURS: float = 2.0
    DAILY_API_BUDGET_USD: float = 15.0
    MAX_POSITION_PCT: float = 0.016
    MAX_TOTAL_EXPOSURE_PCT: float = 0.30
    KELLY_FRACTION: float = 0.25
    MAX_CLUSTER_EXPOSURE_PCT: float = 0.12

    # Market Price Filter
    MIN_TRADEABLE_PRICE: float = 0.05   # Skip markets with YES < 5%
    MAX_TRADEABLE_PRICE: float = 0.95   # Skip markets with YES > 95%

    # Duplicate Prevention
    MARKET_COOLDOWN_HOURS: float = 24.0
    EVALUATION_COOLDOWN_HOURS: float = 4.0  # Skip re-calling Grok on same market within 4 hours
    QUESTION_SIMILARITY_THRESHOLD: float = 0.60

    # Pre-screen: cheap LLM gate before Twitter API call
    PRESCREEN_ENABLED: bool = True
    PRESCREEN_MIN_EDGE: float = 0.05        # raw |prob - price| gate (intentionally loose)
    PRESCREEN_MIN_CONFIDENCE: float = 0.25   # LLM confidence gate (loose — filters only very low confidence)
    PRESCREEN_MAX_TOKENS: int = 500          # small response budget
    PRESCREEN_ANCHORING_MODE: str = "independent"  # "independent" (default) | "anchored" (rollback lever, uses full-eval SYSTEM_PROMPT)
    WEAK_SIGNAL_STRENGTH_THRESHOLD: float = 0.45  # avg credibility below this → skip before orderbook + LLM

    # Alerts
    DAILY_SUMMARY_HOUR_UTC: int = 0  # Hour of day (UTC) to send daily summary

    # Initial Bankroll
    INITIAL_BANKROLL: float = 10000.0
    MIN_HOURS_TO_RESOLUTION: float = 0.5
    DISABLED_MARKET_TYPES: str = ""
    TWITTER_ENABLED: bool = True

    @property
    def disabled_market_types_set(self) -> set:
        return {t.strip() for t in self.DISABLED_MARKET_TYPES.split(",") if t.strip()}

    def safe_config(self) -> dict:
        """Return config dict with secret fields redacted for DB storage."""
        import json as _json
        data = _json.loads(self.model_dump_json())
        for key in _SECRET_FIELDS:
            if key in data:
                data[key] = "***REDACTED***"
        return data

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
