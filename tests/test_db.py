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

    async def test_get_open_market_ids(self, db, sample_trade_record):
        # Two open trades on different markets
        await db.save_trade(sample_trade_record(action="BUY_YES", market_id="market_open_1"))
        await db.save_trade(sample_trade_record(action="BUY_NO", market_id="market_open_2"))
        # Resolved trade — should NOT appear
        await db.save_trade(sample_trade_record(action="BUY_YES", market_id="market_resolved", actual_outcome=True, pnl=10.0))
        # Voided trade — should NOT appear
        await db.save_trade(sample_trade_record(action="BUY_YES", market_id="market_voided", voided=True, void_reason="test"))
        # SKIP — should NOT appear
        await db.save_trade(sample_trade_record(action="SKIP", market_id="market_skip", skip_reason="low_edge"))

        result = await db.get_open_market_ids()
        assert result == {"market_open_1", "market_open_2"}

    async def test_get_open_market_ids_empty(self, db):
        result = await db.get_open_market_ids()
        assert result == set()


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
class TestPortfolioInit:
    async def test_init_portfolio_if_missing_creates_row(self, db):
        await db.init_portfolio_if_missing(10000.0)
        loaded = await db.load_portfolio()
        assert loaded.cash_balance == 10000.0
        assert loaded.total_equity == 10000.0
        assert loaded.peak_equity == 10000.0
        assert loaded.total_pnl == 0.0
        assert loaded.max_drawdown == 0.0

    async def test_init_portfolio_if_missing_is_noop(self, db):
        custom = Portfolio(cash_balance=7777.0, total_equity=8000.0, total_pnl=223.0, peak_equity=8500.0, max_drawdown=0.05)
        await db.save_portfolio(custom)
        await db.init_portfolio_if_missing(10000.0)
        loaded = await db.load_portfolio()
        assert loaded.cash_balance == 7777.0
        assert loaded.total_equity == 8000.0
        assert loaded.total_pnl == 223.0
        assert loaded.peak_equity == 8500.0
        assert loaded.max_drawdown == 0.05


@pytest.mark.asyncio
class TestAPICosts:
    async def test_increment_new(self, db):
        await db.increment_api_cost("minimax", tokens_in=1000, tokens_out=200)
        spend = await db.get_today_api_spend()
        assert spend > 0

    async def test_increment_existing(self, db):
        await db.increment_api_cost("minimax", tokens_in=1000)
        await db.increment_api_cost("minimax", tokens_in=1000)
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


# ---------------------------------------------------------------------------
# Dual-label: new DB tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestDualLabelDB:
    """Tests for Phase 1 dual-label fields in trade_records and daily_reviews."""

    async def test_get_open_trades_includes_exited(self, db, sample_trade_record):
        """get_open_trades now returns trades with exit_type set (filter removal).
        Both open-and-not-exited AND TP/SL-exited-but-unresolved should appear."""
        trade_open = sample_trade_record(action="BUY_YES", actual_outcome=None)
        trade_tp = sample_trade_record(
            action="BUY_YES",
            actual_outcome=None,
            pnl=15.0,
            exit_type="take_profit",
            exit_price=0.65,
        )
        await db.save_trade(trade_open)
        await db.save_trade(trade_tp)

        opens = await db.get_open_trades()
        ids = {t.record_id for t in opens}
        assert trade_open.record_id in ids
        assert trade_tp.record_id in ids

    async def test_save_and_load_trade_with_dual_labels(self, db, sample_trade_record):
        """Round-trip trade with trade_profitable, pnl_brier_raw, pnl_brier_adjusted."""
        record = sample_trade_record(
            action="BUY_YES",
            pnl=30.0,
            exit_type="take_profit",
            exit_price=0.65,
            trade_profitable=1,
            pnl_brier_raw=0.04,
            pnl_brier_adjusted=0.03,
        )
        await db.save_trade(record)
        loaded = await db.get_trade(record.record_id)

        assert loaded is not None
        assert loaded.trade_profitable == 1
        assert loaded.pnl_brier_raw == pytest.approx(0.04)
        assert loaded.pnl_brier_adjusted == pytest.approx(0.03)

    async def test_update_trade_persists_dual_labels(self, db, sample_trade_record):
        """update_trade correctly persists all three dual-label fields."""
        record = sample_trade_record(action="BUY_YES")
        await db.save_trade(record)

        # Simulate what resolution/ws_exit does after an exit
        record.pnl = -10.0
        record.exit_type = "stop_loss"
        record.exit_price = 0.42
        record.trade_profitable = 0
        record.pnl_brier_raw = 0.49
        record.pnl_brier_adjusted = 0.46
        await db.update_trade(record)

        loaded = await db.get_trade(record.record_id)
        assert loaded.trade_profitable == 0
        assert loaded.pnl_brier_raw == pytest.approx(0.49)
        assert loaded.pnl_brier_adjusted == pytest.approx(0.46)

    async def test_save_and_load_daily_review_with_pnl_metrics(self, db):
        """Round-trip DailyReview with all 4 new pnl-metric fields populated."""
        from src.models import DailyReview

        review = DailyReview(
            review_date="2026-04-28",
            timestamp=datetime.now(timezone.utc),
            trade_count=10,
            skip_count=5,
            resolved_count=8,
            win_rate=0.625,
            roi_pct=8.0,
            total_pnl=40.0,
            avg_brier_raw=0.12,
            avg_brier_adjusted=0.11,
            health_status="HEALTHY",
            experiment_run="test-run-001",
            win_rate_pnl=0.70,
            avg_pnl_brier_raw=0.09,
            avg_pnl_brier_adjusted=0.08,
            pnl_resolved_count=7,
        )
        await db.save_daily_review(review)
        loaded = await db.get_daily_review("2026-04-28")

        assert loaded is not None
        assert loaded.win_rate_pnl == pytest.approx(0.70)
        assert loaded.avg_pnl_brier_raw == pytest.approx(0.09)
        assert loaded.avg_pnl_brier_adjusted == pytest.approx(0.08)
        assert loaded.pnl_resolved_count == 7
