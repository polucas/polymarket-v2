from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import aiosqlite

SCHEMA_VERSION = 2

MIGRATIONS: dict[int, list[str]] = {
    1: [
        # --- Schema version tracking ---
        """CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL DEFAULT (datetime('now'))
        )""",
        # --- Experiment runs ---
        """CREATE TABLE IF NOT EXISTS experiment_runs (
            run_id TEXT PRIMARY KEY,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            config_snapshot TEXT NOT NULL,
            description TEXT,
            model_used TEXT NOT NULL,
            include_in_learning BOOLEAN DEFAULT TRUE,
            total_trades INTEGER DEFAULT 0,
            total_pnl REAL DEFAULT 0.0,
            avg_brier REAL DEFAULT 0.0,
            sharpe_ratio REAL DEFAULT 0.0
        )""",
        # --- Model swaps ---
        """CREATE TABLE IF NOT EXISTS model_swaps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            old_model TEXT NOT NULL,
            new_model TEXT NOT NULL,
            reason TEXT,
            experiment_run_started TEXT REFERENCES experiment_runs(run_id)
        )""",
        # --- Trade records ---
        """CREATE TABLE IF NOT EXISTS trade_records (
            record_id TEXT PRIMARY KEY,
            experiment_run TEXT NOT NULL REFERENCES experiment_runs(run_id),
            timestamp TEXT NOT NULL,
            model_used TEXT NOT NULL,

            market_id TEXT NOT NULL,
            market_question TEXT NOT NULL,
            market_type TEXT NOT NULL,
            resolution_window_hours REAL,
            tier INTEGER NOT NULL,

            grok_raw_probability REAL NOT NULL,
            grok_raw_confidence REAL NOT NULL,
            grok_reasoning TEXT,
            grok_signal_types TEXT,
            headline_only_signal BOOLEAN DEFAULT FALSE,

            calibration_adjustment REAL DEFAULT 0,
            market_type_adjustment REAL DEFAULT 0,
            signal_weight_adjustment REAL DEFAULT 0,
            final_adjusted_probability REAL NOT NULL,
            final_adjusted_confidence REAL NOT NULL,

            market_price_at_decision REAL NOT NULL,
            orderbook_depth_usd REAL,
            fee_rate REAL NOT NULL,
            calculated_edge REAL NOT NULL,
            trade_score REAL,

            action TEXT NOT NULL,
            skip_reason TEXT,
            position_size_usd REAL DEFAULT 0,
            kelly_fraction_used REAL DEFAULT 0,
            market_cluster_id TEXT,

            actual_outcome BOOLEAN,
            pnl REAL,
            brier_score_raw REAL,
            brier_score_adjusted REAL,
            resolved_at TEXT,
            unrealized_adverse_move REAL,

            voided BOOLEAN DEFAULT FALSE,
            void_reason TEXT
        )""",
        # --- Trade record indexes ---
        "CREATE INDEX IF NOT EXISTS idx_trades_market_type ON trade_records(market_type)",
        "CREATE INDEX IF NOT EXISTS idx_trades_experiment ON trade_records(experiment_run)",
        "CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trade_records(timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_trades_model ON trade_records(model_used)",
        "CREATE INDEX IF NOT EXISTS idx_trades_unresolved ON trade_records(actual_outcome) WHERE actual_outcome IS NULL",
        "CREATE INDEX IF NOT EXISTS idx_trades_headline ON trade_records(headline_only_signal) WHERE headline_only_signal = TRUE",
        # --- Calibration state ---
        """CREATE TABLE IF NOT EXISTS calibration_state (
            bucket_range TEXT PRIMARY KEY,
            alpha REAL NOT NULL,
            beta REAL NOT NULL,
            updated_at TEXT NOT NULL
        )""",
        # --- Market type performance ---
        """CREATE TABLE IF NOT EXISTS market_type_performance (
            market_type TEXT PRIMARY KEY,
            total_trades INTEGER DEFAULT 0,
            total_pnl REAL DEFAULT 0.0,
            brier_scores TEXT,
            total_observed INTEGER DEFAULT 0,
            counterfactual_pnl REAL DEFAULT 0.0,
            updated_at TEXT NOT NULL
        )""",
        # --- Signal trackers ---
        """CREATE TABLE IF NOT EXISTS signal_trackers (
            source_tier TEXT NOT NULL,
            info_type TEXT NOT NULL,
            market_type TEXT NOT NULL,
            present_winning INTEGER DEFAULT 0,
            present_losing INTEGER DEFAULT 0,
            absent_winning INTEGER DEFAULT 0,
            absent_losing INTEGER DEFAULT 0,
            last_updated TEXT,
            PRIMARY KEY (source_tier, info_type, market_type)
        )""",
        # --- Portfolio ---
        """CREATE TABLE IF NOT EXISTS portfolio (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            cash_balance REAL NOT NULL,
            total_equity REAL NOT NULL,
            total_pnl REAL NOT NULL,
            peak_equity REAL NOT NULL,
            max_drawdown REAL NOT NULL,
            updated_at TEXT NOT NULL
        )""",
        # --- API costs ---
        """CREATE TABLE IF NOT EXISTS api_costs (
            date TEXT NOT NULL,
            service TEXT NOT NULL,
            calls INTEGER DEFAULT 0,
            tokens_in INTEGER DEFAULT 0,
            tokens_out INTEGER DEFAULT 0,
            cost_usd REAL DEFAULT 0.0,
            PRIMARY KEY (date, service)
        )""",
        # --- Daily mode log ---
        """CREATE TABLE IF NOT EXISTS daily_mode_log (
            date TEXT PRIMARY KEY,
            observe_only_triggered_at TEXT,
            trades_before_observe INTEGER DEFAULT 0,
            grok_calls_saved INTEGER DEFAULT 0,
            cost_saved_usd REAL DEFAULT 0.0
        )""",
        # --- Parse failures ---
        """CREATE TABLE IF NOT EXISTS parse_failures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            market_id TEXT NOT NULL,
            timestamp TEXT NOT NULL DEFAULT (datetime('now')),
            date TEXT NOT NULL DEFAULT (date('now'))
        )""",
    ],
    2: [
        "ALTER TABLE trade_records ADD COLUMN resolution_datetime TEXT",
    ],
}


async def get_current_version(conn: aiosqlite.Connection) -> int:
    try:
        cursor = await conn.execute(
            "SELECT MAX(version) FROM schema_version"
        )
        row = await cursor.fetchone()
        return row[0] if row and row[0] is not None else 0
    except Exception:
        return 0


async def run_migrations(db) -> None:
    """Apply pending migrations. `db` is a Database instance."""
    conn = db._conn
    current = await get_current_version(conn)

    for version in sorted(MIGRATIONS.keys()):
        if version <= current:
            continue
        for stmt in MIGRATIONS[version]:
            await conn.execute(stmt)
        await conn.execute(
            "INSERT INTO schema_version (version) VALUES (?)", (version,)
        )
        await conn.commit()
