import uuid
from datetime import datetime, timezone

import pytest
import pytest_asyncio
from structlog.testing import capture_logs

from src.config import Settings
from src.learning.drift_monitor import (
    check_cash_drift,
    check_cash_drift_periodic,
    compute_drift,
)
from src.models import ExperimentRun, Portfolio


pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture(autouse=True)
async def _seed_experiment(db):
    """Insert the default experiment run so trade_records FK is satisfied."""
    run = ExperimentRun(
        run_id="test-run-001",
        started_at=datetime.now(timezone.utc),
        model_used="grok-3-fast",
        description="Seed experiment for drift monitor tests",
    )
    await db.save_experiment(run)


def _settings(**overrides):
    defaults = dict(XAI_API_KEY="test", TWITTER_API_KEY="test", INITIAL_BANKROLL=10000.0)
    defaults.update(overrides)
    return Settings(**defaults)


async def _seed_resolved_trade(db, sample_trade_record, *, position_size_usd: float, pnl: float):
    """Insert a resolved (exited) trade contributing position_size + pnl back to cash."""
    record = sample_trade_record(
        record_id=str(uuid.uuid4()),
        action="BUY_YES",
        position_size_usd=position_size_usd,
        actual_outcome=True,
        pnl=pnl,
        resolved_at=datetime.now(timezone.utc),
    )
    await db.save_trade(record)
    return record


class TestComputeDrift:
    async def test_zero_drift_when_cash_matches_replay(self, db, sample_trade_record):
        """expected_cash = initial - deducted + credited; drift = actual - expected."""
        settings = _settings(INITIAL_BANKROLL=10000.0)

        position_size = 200.0
        pnl = 50.0
        await _seed_resolved_trade(db, sample_trade_record, position_size_usd=position_size, pnl=pnl)

        # actual cash = initial - deducted + credited (matches replay exactly)
        expected_cash = 10000.0 - position_size + (position_size + pnl)
        portfolio = Portfolio(cash_balance=expected_cash, total_equity=expected_cash, total_pnl=pnl)
        await db.save_portfolio(portfolio)

        snap = await compute_drift(db, settings)

        assert snap["actual_cash"] == pytest.approx(expected_cash)
        assert snap["expected_cash"] == pytest.approx(expected_cash)
        assert snap["drift"] == pytest.approx(0.0, abs=1e-6)
        assert snap["n_entries"] == 1
        assert snap["n_exits"] == 1
        assert snap["locked_replay"] == pytest.approx(0.0)

    async def test_nonzero_drift_when_cash_diverges(self, db, sample_trade_record):
        settings = _settings(INITIAL_BANKROLL=10000.0)

        position_size = 200.0
        pnl = 50.0
        await _seed_resolved_trade(db, sample_trade_record, position_size_usd=position_size, pnl=pnl)

        expected_cash = 10000.0 - position_size + (position_size + pnl)
        actual_cash = expected_cash + 25.0  # introduce drift
        portfolio = Portfolio(cash_balance=actual_cash, total_equity=actual_cash, total_pnl=pnl)
        await db.save_portfolio(portfolio)

        snap = await compute_drift(db, settings)
        assert snap["drift"] == pytest.approx(25.0)


class TestCheckCashDriftLogging:
    async def test_logs_info_when_drift_small(self, db, sample_trade_record):
        settings = _settings(INITIAL_BANKROLL=10000.0)
        portfolio = Portfolio(cash_balance=10000.0, total_equity=10000.0, total_pnl=0.0)
        await db.save_portfolio(portfolio)

        with capture_logs() as cap:
            await check_cash_drift(db, settings)

        events = [e["event"] for e in cap]
        assert "cash_drift_ok" in events
        assert "cash_drift_growing" not in events
        assert "cash_drift_alarm" not in events

    async def test_logs_warning_when_drift_moderate(self, db, sample_trade_record):
        settings = _settings(INITIAL_BANKROLL=10000.0)
        # No trades -> expected_cash = INITIAL_BANKROLL; actual diverges by $10
        portfolio = Portfolio(cash_balance=10010.0, total_equity=10010.0, total_pnl=0.0)
        await db.save_portfolio(portfolio)

        with capture_logs() as cap:
            await check_cash_drift(db, settings)

        events = [e["event"] for e in cap]
        assert "cash_drift_growing" in events
        assert "cash_drift_alarm" not in events

    async def test_logs_error_when_drift_large(self, db, sample_trade_record):
        settings = _settings(INITIAL_BANKROLL=10000.0)
        portfolio = Portfolio(cash_balance=10100.0, total_equity=10100.0, total_pnl=0.0)
        await db.save_portfolio(portfolio)

        with capture_logs() as cap:
            await check_cash_drift(db, settings)

        events = [e["event"] for e in cap]
        assert "cash_drift_alarm" in events


class TestCheckCashDriftPeriodic:
    async def test_persists_row_to_drift_history(self, db, sample_trade_record):
        settings = _settings(INITIAL_BANKROLL=10000.0)
        portfolio = Portfolio(cash_balance=10000.0, total_equity=10000.0, total_pnl=0.0)
        await db.save_portfolio(portfolio)

        await check_cash_drift_periodic(db, settings)

        async with db._conn.execute("SELECT COUNT(*) FROM drift_history") as cur:
            row = await cur.fetchone()
        assert row[0] == 1

        async with db._conn.execute(
            "SELECT actual_cash, expected_cash, drift, n_entries, n_exits, locked_replay FROM drift_history ORDER BY id DESC LIMIT 1"
        ) as cur:
            persisted = await cur.fetchone()
        assert persisted[0] == pytest.approx(10000.0)
        assert persisted[1] == pytest.approx(10000.0)
        assert persisted[2] == pytest.approx(0.0)

    async def test_emits_growth_event_when_drift_changes_significantly(self, db, sample_trade_record):
        settings = _settings(INITIAL_BANKROLL=10000.0)

        # First run: drift = 0
        portfolio = Portfolio(cash_balance=10000.0, total_equity=10000.0, total_pnl=0.0)
        await db.save_portfolio(portfolio)
        await check_cash_drift_periodic(db, settings)

        # Second run: drift jumps by > $1
        portfolio2 = Portfolio(cash_balance=10005.0, total_equity=10005.0, total_pnl=0.0)
        await db.save_portfolio(portfolio2)

        with capture_logs() as cap:
            await check_cash_drift_periodic(db, settings)

        growth_events = [e for e in cap if e.get("event") == "cash_drift_growth_event"]
        assert len(growth_events) == 1
        assert growth_events[0]["delta"] == pytest.approx(5.0)

        async with db._conn.execute("SELECT COUNT(*) FROM drift_history") as cur:
            row = await cur.fetchone()
        assert row[0] == 2

    async def test_no_growth_event_when_drift_stable(self, db, sample_trade_record):
        settings = _settings(INITIAL_BANKROLL=10000.0)

        portfolio = Portfolio(cash_balance=10000.0, total_equity=10000.0, total_pnl=0.0)
        await db.save_portfolio(portfolio)
        await check_cash_drift_periodic(db, settings)

        # Second run with same drift (no change)
        with capture_logs() as cap:
            await check_cash_drift_periodic(db, settings)

        growth_events = [e for e in cap if e.get("event") == "cash_drift_growth_event"]
        assert len(growth_events) == 0
