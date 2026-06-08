"""Cash drift monitoring — startup check + periodic gauge.

Computes drift = actual_cash - (INITIAL_BANKROLL - sum_entries + sum_exits).
Used by main.py at startup and by scheduler.py every 30 minutes.
"""
from __future__ import annotations

import structlog

from src.config import Settings
from src.db.sqlite import Database

log = structlog.get_logger()


async def compute_drift(db: Database, settings: Settings) -> dict:
    """Compute current drift snapshot. Returns dict with all fields needed for logging and persistence."""
    async with db._conn.execute("SELECT cash_balance FROM portfolio WHERE id = 1") as cur:
        row = await cur.fetchone()
    actual_cash = float(row[0]) if row else 0.0

    async with db._conn.execute("""
        SELECT
            COALESCE(SUM(position_size_usd), 0) AS deducted,
            COALESCE(SUM(CASE WHEN actual_outcome IS NOT NULL OR exit_type IS NOT NULL
                              THEN position_size_usd + pnl ELSE 0 END), 0) AS credited,
            COUNT(*) AS n_entries,
            SUM(CASE WHEN actual_outcome IS NOT NULL OR exit_type IS NOT NULL THEN 1 ELSE 0 END) AS n_exits,
            COALESCE(SUM(CASE WHEN actual_outcome IS NULL AND exit_type IS NULL THEN position_size_usd ELSE 0 END), 0) AS locked
        FROM trade_records
        WHERE action != 'SKIP' AND voided = 0
    """) as cur:
        sums = await cur.fetchone()

    deducted = float(sums[0] or 0.0)
    credited = float(sums[1] or 0.0)
    n_entries = int(sums[2] or 0)
    n_exits = int(sums[3] or 0)
    locked = float(sums[4] or 0.0)

    initial = float(settings.INITIAL_BANKROLL)
    expected = initial - deducted + credited
    drift = actual_cash - expected

    return {
        "actual_cash": round(actual_cash, 4),
        "expected_cash": round(expected, 4),
        "drift": round(drift, 4),
        "n_entries": n_entries,
        "n_exits": n_exits,
        "locked_replay": round(locked, 4),
    }


async def check_cash_drift(db: Database, settings: Settings) -> dict:
    """Compute drift and log at appropriate level. Used at startup. Returns the snapshot dict."""
    snap = await compute_drift(db, settings)
    drift = snap["drift"]
    if abs(drift) >= 50.0:
        log.error("cash_drift_alarm", **snap)
    elif abs(drift) >= 5.0:
        log.warning("cash_drift_growing", **snap)
    else:
        log.info("cash_drift_ok", actual_cash=snap["actual_cash"], drift=snap["drift"])
    return snap


async def check_cash_drift_periodic(db: Database, settings: Settings) -> None:
    """Periodic 30-min drift check: log + persist row + emit growth event if drift increased."""
    snap = await compute_drift(db, settings)

    # Read previous row to detect growth
    async with db._conn.execute(
        "SELECT drift, timestamp FROM drift_history ORDER BY id DESC LIMIT 1"
    ) as cur:
        prev_row = await cur.fetchone()

    # Persist
    await db._conn.execute(
        """INSERT INTO drift_history
           (actual_cash, expected_cash, drift, n_entries, n_exits, locked_replay)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (snap["actual_cash"], snap["expected_cash"], snap["drift"],
         snap["n_entries"], snap["n_exits"], snap["locked_replay"]),
    )
    await db._conn.commit()

    # Log at level
    drift = snap["drift"]
    if abs(drift) >= 50.0:
        log.error("cash_drift_alarm", **snap)
    elif abs(drift) >= 5.0:
        log.warning("cash_drift_growing", **snap)
    else:
        log.info("cash_drift_ok", actual_cash=snap["actual_cash"], drift=snap["drift"])

    # Growth event
    if prev_row is not None:
        prev_drift = float(prev_row[0])
        d_drift = drift - prev_drift
        if abs(d_drift) > 1.0:
            log.warning(
                "cash_drift_growth_event",
                previous_drift=round(prev_drift, 4),
                current_drift=round(drift, 4),
                delta=round(d_drift, 4),
                previous_timestamp=prev_row[1],
            )
