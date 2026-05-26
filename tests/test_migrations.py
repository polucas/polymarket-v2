"""Tests for src/db/migrations.py — schema versioning and backfill logic."""
from __future__ import annotations

import uuid

import aiosqlite
import pytest
import pytest_asyncio

from src.db.migrations import MIGRATIONS, SCHEMA_VERSION, run_migrations
from src.db.sqlite import Database


# ---------------------------------------------------------------------------
# Helper: apply a specific migration version to a raw aiosqlite connection
# ---------------------------------------------------------------------------


async def _apply_migrations_up_to(conn: aiosqlite.Connection, max_version: int) -> None:
    """Apply all migrations up to and including max_version."""
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA foreign_keys = ON")
    for version in sorted(MIGRATIONS.keys()):
        if version > max_version:
            break
        for stmt in MIGRATIONS[version]:
            await conn.execute(stmt)
        await conn.execute(
            "INSERT OR IGNORE INTO schema_version (version) VALUES (?)", (version,)
        )
    await conn.commit()


# ---------------------------------------------------------------------------
# Test: SCHEMA_VERSION constant is 7
# ---------------------------------------------------------------------------


def test_schema_version_is_8():
    """SCHEMA_VERSION must be 8 after F8 price snapshot migration."""
    assert SCHEMA_VERSION == 8


# ---------------------------------------------------------------------------
# Test: v7 backfill sets trade_profitable correctly from pnl sign
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_v7_backfill_sets_trade_profitable():
    """Apply v6 schema, insert 3 exited trade rows, then apply v7.
    Assert backfill populates trade_profitable: pnl>0→1, pnl<=0→0, pnl==0→0.
    """
    async with aiosqlite.connect(":memory:") as conn:
        # Build DB at v6 schema
        await _apply_migrations_up_to(conn, 6)

        # Insert parent experiment run (required by FK constraint)
        await conn.execute(
            "INSERT INTO experiment_runs "
            "(run_id, started_at, config_snapshot, model_used) "
            "VALUES (?, ?, ?, ?)",
            ("exp-backfill", "2026-01-01T00:00:00", "{}", "test-model"),
        )

        # Insert 3 trade rows with pnl set and exit_type (so backfill applies)
        trades = [
            ("trade-profit",    "BUY_YES",  10.0, "take_profit"),
            ("trade-loss",      "BUY_YES",  -5.0, "stop_loss"),
            ("trade-zero",      "BUY_YES",   0.0, "take_profit"),
        ]
        for tid, action, pnl, exit_type in trades:
            await conn.execute(
                """INSERT INTO trade_records
                   (record_id, experiment_run, timestamp, model_used,
                    market_id, market_question, market_type, resolution_window_hours, tier,
                    grok_raw_probability, grok_raw_confidence,
                    final_adjusted_probability, final_adjusted_confidence,
                    market_price_at_decision, fee_rate, calculated_edge,
                    action, position_size_usd, pnl, exit_type)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (tid, "exp-backfill", "2026-04-01T12:00:00", "test-model",
                 f"mkt-{tid}", "Test?", "political", 12.0, 1,
                 0.70, 0.80, 0.68, 0.78, 0.60, 0.0, 0.08,
                 action, 100.0, pnl, exit_type),
            )
        await conn.commit()

        # Now apply v7 migration (columns + backfill)
        for stmt in MIGRATIONS[7]:
            await conn.execute(stmt)
        await conn.commit()

        # Verify backfill results
        cursor = await conn.execute(
            "SELECT record_id, trade_profitable FROM trade_records ORDER BY record_id"
        )
        rows = {r[0]: r[1] for r in await cursor.fetchall()}

    assert rows["trade-profit"] == 1,  "pnl > 0 should set trade_profitable=1"
    assert rows["trade-loss"]   == 0,  "pnl < 0 should set trade_profitable=0"
    assert rows["trade-zero"]   == 0,  "pnl = 0 should set trade_profitable=0 (pnl<=0 branch)"


@pytest.mark.asyncio
async def test_v7_backfill_ignores_skip_records():
    """SKIP trade records must NOT receive trade_profitable from backfill."""
    async with aiosqlite.connect(":memory:") as conn:
        await _apply_migrations_up_to(conn, 6)

        await conn.execute(
            "INSERT INTO experiment_runs "
            "(run_id, started_at, config_snapshot, model_used) "
            "VALUES (?, ?, ?, ?)",
            ("exp-skip", "2026-01-01T00:00:00", "{}", "test-model"),
        )

        # SKIP record with pnl (unusual but defensive)
        await conn.execute(
            """INSERT INTO trade_records
               (record_id, experiment_run, timestamp, model_used,
                market_id, market_question, market_type, resolution_window_hours, tier,
                grok_raw_probability, grok_raw_confidence,
                final_adjusted_probability, final_adjusted_confidence,
                market_price_at_decision, fee_rate, calculated_edge,
                action, position_size_usd, pnl, exit_type, skip_reason)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("trade-skip", "exp-skip", "2026-04-01T12:00:00", "test-model",
             "mkt-skip", "Test?", "political", 12.0, 1,
             0.70, 0.80, 0.68, 0.78, 0.60, 0.0, 0.08,
             "SKIP", 0.0, 10.0, "take_profit", "low_edge"),
        )
        await conn.commit()

        for stmt in MIGRATIONS[7]:
            await conn.execute(stmt)
        await conn.commit()

        cursor = await conn.execute(
            "SELECT trade_profitable FROM trade_records WHERE record_id='trade-skip'"
        )
        row = await cursor.fetchone()

    assert row[0] is None, "SKIP records must NOT receive trade_profitable backfill"


@pytest.mark.asyncio
async def test_v7_adds_daily_review_columns():
    """v7 migration adds win_rate_pnl, avg_pnl_brier_raw, avg_pnl_brier_adjusted, pnl_resolved_count."""
    async with aiosqlite.connect(":memory:") as conn:
        await _apply_migrations_up_to(conn, 7)

        # Verify columns exist by inserting a row with those fields
        await conn.execute(
            """INSERT INTO daily_reviews
               (review_date, timestamp, win_rate_pnl, avg_pnl_brier_raw,
                avg_pnl_brier_adjusted, pnl_resolved_count,
                brier_by_market_type, calibration_drift, signal_effectiveness,
                skip_reason_distribution, top_performing_types, worst_performing_types,
                llm_recommendations)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("2026-04-28", "2026-04-28T23:00:00", 0.75, 0.08, 0.07, 5,
             "{}", "{}", "{}", "{}", "[]", "[]", "[]"),
        )
        await conn.commit()

        cursor = await conn.execute(
            "SELECT win_rate_pnl, avg_pnl_brier_raw, avg_pnl_brier_adjusted, pnl_resolved_count "
            "FROM daily_reviews WHERE review_date='2026-04-28'"
        )
        row = await cursor.fetchone()

    assert abs(row[0] - 0.75) < 1e-9
    assert abs(row[1] - 0.08) < 1e-9
    assert abs(row[2] - 0.07) < 1e-9
    assert row[3] == 5


@pytest.mark.asyncio
async def test_full_migration_is_idempotent():
    """run_migrations twice on same in-memory DB must not raise."""
    db = await Database.init(":memory:")
    try:
        await run_migrations(db)
        await run_migrations(db)  # second run should be a no-op
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_migration_v8_creates_snapshot_table():
    """Migration 8 creates trade_price_snapshots table with required columns and indexes."""
    async with aiosqlite.connect(":memory:") as conn:
        # Apply all migrations up to v8
        await _apply_migrations_up_to(conn, 8)

        # Check table exists
        cursor = await conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='trade_price_snapshots'"
        )
        row = await cursor.fetchone()
        assert row is not None, "trade_price_snapshots table should exist after v8 migration"

        # Check required columns via PRAGMA
        cursor = await conn.execute("PRAGMA table_info(trade_price_snapshots)")
        col_rows = await cursor.fetchall()
        col_names = {r[1] for r in col_rows}
        required_cols = {"id", "trade_record_id", "timestamp", "best_bid", "roi", "source"}
        assert required_cols.issubset(col_names), (
            f"Missing columns: {required_cols - col_names}"
        )

        # Check indexes exist
        cursor = await conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='trade_price_snapshots'"
        )
        idx_rows = await cursor.fetchall()
        idx_names = {r[0] for r in idx_rows}
        assert "idx_snapshots_trade" in idx_names
        assert "idx_snapshots_timestamp" in idx_names
