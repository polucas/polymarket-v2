import pytest
import pytest_asyncio
from datetime import datetime, timedelta, timezone
from src.db.sqlite import Database
from src.db.migrations import run_migrations
from src.models import (
    CalibrationBucket, MarketTypePerformance, SignalTracker,
    ExperimentRun, ModelSwapEvent, Portfolio, CALIBRATION_BUCKET_RANGES,
)


@pytest_asyncio.fixture(autouse=True)
async def _seed_experiment(db):
    """Insert the default experiment run so trade_records FK is satisfied."""
    run = ExperimentRun(
        run_id="test-run-001",
        started_at=datetime.now(timezone.utc),
        model_used="grok-3-fast",
        description="Seed experiment for tests",
    )
    await db.save_experiment(run)


@pytest.mark.asyncio
class TestSchema:
    async def test_migrations_create_tables(self, db):
        cursor = await db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        rows = await cursor.fetchall()
        tables = {row[0] for row in rows}
        expected = {
            "schema_version", "experiment_runs", "model_swaps", "trade_records",
            "calibration_state", "market_type_performance", "signal_trackers",
            "portfolio", "api_costs", "parse_failures",
        }
        assert expected.issubset(tables)

    async def test_migrations_idempotent(self, db):
        # Running again should not error
        await run_migrations(db)


@pytest.mark.asyncio
class TestTradeRecordCRUD:
    async def test_save_and_get(self, db, sample_trade_record):
        record = sample_trade_record()
        await db.save_trade(record)
        loaded = await db.get_trade(record.record_id)
        assert loaded is not None
        assert loaded.record_id == record.record_id
        assert loaded.market_id == record.market_id
        assert loaded.grok_raw_probability == record.grok_raw_probability
        assert loaded.action == record.action

    async def test_get_open_trades_excludes_resolved(self, db, sample_trade_record):
        open_trade = sample_trade_record(action="BUY_YES")
        resolved_trade = sample_trade_record(action="BUY_YES", actual_outcome=True, pnl=50.0)
        await db.save_trade(open_trade)
        await db.save_trade(resolved_trade)
        opens = await db.get_open_trades()
        assert len(opens) == 1
        assert opens[0].record_id == open_trade.record_id

    async def test_get_open_trades_excludes_voided(self, db, sample_trade_record):
        voided_trade = sample_trade_record(action="BUY_YES", voided=True, void_reason="test")
        await db.save_trade(voided_trade)
        opens = await db.get_open_trades()
        assert len(opens) == 0

    async def test_get_open_trades_excludes_skips(self, db, sample_trade_record):
        skip_trade = sample_trade_record(action="SKIP", skip_reason="low_edge")
        await db.save_trade(skip_trade)
        opens = await db.get_open_trades()
        assert len(opens) == 0

    async def test_update_trade(self, db, sample_trade_record):
        record = sample_trade_record()
        await db.save_trade(record)
        record.actual_outcome = True
        record.pnl = 100.0
        record.brier_score_raw = 0.0625
        record.brier_score_adjusted = 0.0729
        record.resolved_at = datetime.now(timezone.utc)
        await db.update_trade(record)
        loaded = await db.get_trade(record.record_id)
        assert loaded.actual_outcome == True  # noqa: E712  (SQLite returns int 1)
        assert loaded.pnl == 100.0

    async def test_count_today_trades(self, db, sample_trade_record):
        await db.save_trade(sample_trade_record(action="BUY_YES"))
        await db.save_trade(sample_trade_record(action="BUY_NO"))
        await db.save_trade(sample_trade_record(action="SKIP"))
        count = await db.count_today_trades()
        assert count == 2  # SKIPs excluded

    async def test_count_open_trades(self, db, sample_trade_record):
        await db.save_trade(sample_trade_record(action="BUY_YES"))
        await db.save_trade(sample_trade_record(action="BUY_YES", actual_outcome=True))
        count = await db.count_open_trades()
        assert count == 1


@pytest.mark.asyncio
class TestCalibrationPersistence:
    async def test_save_load_roundtrip(self, db):
        from src.learning.calibration import CalibrationManager
        mgr = CalibrationManager()
        mgr.buckets[0].alpha = 10.0
        mgr.buckets[0].beta = 5.0
        await mgr.save(db)
        mgr2 = CalibrationManager()
        await mgr2.load(db)
        assert mgr2.buckets[0].alpha == 10.0
        assert mgr2.buckets[0].beta == 5.0


@pytest.mark.asyncio
class TestMarketTypePersistence:
    async def test_save_load_roundtrip(self, db):
        from src.learning.market_type import MarketTypeManager
        mgr = MarketTypeManager()
        mgr._ensure("political")
        mgr.performances["political"].total_trades = 10
        mgr.performances["political"].brier_scores = [0.1, 0.2, 0.3]
        await mgr.save(db)
        mgr2 = MarketTypeManager()
        await mgr2.load(db)
        assert "political" in mgr2.performances
        assert mgr2.performances["political"].total_trades == 10
        assert mgr2.performances["political"].brier_scores == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
class TestSignalTrackerPersistence:
    async def test_save_load_roundtrip(self, db):
        from src.learning.signal_tracker import SignalTrackerManager
        mgr = SignalTrackerManager()
        mgr._ensure("S2", "I2", "political")
        mgr.trackers[("S2", "I2", "political")].present_in_winning_trades = 8
        await mgr.save(db)
        mgr2 = SignalTrackerManager()
        await mgr2.load(db)
        assert ("S2", "I2", "political") in mgr2.trackers
        assert mgr2.trackers[("S2", "I2", "political")].present_in_winning_trades == 8


@pytest.mark.asyncio
class TestPortfolio:
    async def test_save_load(self, db):
        p = Portfolio(cash_balance=4800.0, total_equity=5200.0, total_pnl=200.0)
        await db.save_portfolio(p)
        loaded = await db.load_portfolio()
        assert loaded.cash_balance == 4800.0
        assert loaded.total_equity == 5200.0

    async def test_update_overwrites(self, db):
        p = Portfolio(cash_balance=5000.0)
        await db.save_portfolio(p)
        p.cash_balance = 4500.0
        await db.save_portfolio(p)
        loaded = await db.load_portfolio()
        assert loaded.cash_balance == 4500.0


@pytest.mark.asyncio
class TestAPICosts:
    async def test_increment_new(self, db):
        await db.increment_api_cost("grok", tokens_in=1000, tokens_out=200)
        spend = await db.get_today_api_spend()
        assert spend > 0

    async def test_increment_existing(self, db):
        await db.increment_api_cost("grok", tokens_in=1000)
        await db.increment_api_cost("grok", tokens_in=1000)
        spend = await db.get_today_api_spend()
        # Should have accumulated two calls
        assert spend > 0


@pytest.mark.asyncio
class TestExperimentRuns:
    async def test_save_and_get_current(self, db):
        run = ExperimentRun(
            run_id="exp-001",
            started_at=datetime.now(timezone.utc),
            model_used="grok-3-fast",
            description="Test experiment",
        )
        await db.save_experiment(run)
        current = await db.get_current_experiment()
        assert current is not None
        assert current.run_id in ("exp-001", "test-run-001")

    async def test_end_experiment(self, db):
        run = ExperimentRun(
            run_id="exp-002",
            started_at=datetime.now(timezone.utc),
            model_used="grok-3-fast",
        )
        await db.save_experiment(run)
        # End both the seed experiment and exp-002
        await db.end_experiment("test-run-001", {"total_trades": 0, "total_pnl": 0.0})
        await db.end_experiment("exp-002", {"total_trades": 10, "total_pnl": 50.0})
        current = await db.get_current_experiment()
        assert current is None  # ended, no active experiment
