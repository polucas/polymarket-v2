"""Tests for handle_model_swap and void_trade."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from src.learning.adjustment import on_trade_resolved
from src.learning.calibration import CalibrationManager
from src.learning.market_type import MarketTypeManager
from src.learning.model_swap import handle_model_swap, recalculate_learning_from_scratch, void_trade
from src.learning.signal_tracker import SignalTrackerManager
from src.models import ExperimentRun, ModelSwapEvent


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _seed_learning(db, sample_trade_record, cal, mkt, sig):
    """Seed all three managers with some trade data and persist a trade to the DB."""
    record = sample_trade_record(
        record_id="seed-001",
        market_type="political",
        grok_raw_probability=0.80,
        grok_raw_confidence=0.75,
        final_adjusted_probability=0.78,
        final_adjusted_confidence=0.74,
        grok_signal_types=[{"source_tier": "S2", "info_type": "I2"}],
        action="BUY_YES",
        actual_outcome=True,
        brier_score_raw=0.04,
        brier_score_adjusted=0.05,
        voided=False,
    )

    # Update all managers
    cal.update_calibration(record)
    mkt.update_market_type(record)
    sig.update_signal_trackers(record)

    # Persist the trade
    await db.save_trade(record)
    return record


# ---------------------------------------------------------------------------
# handle_model_swap
# ---------------------------------------------------------------------------


class TestHandleModelSwap:
    """Tests 27-29: handle_model_swap resets/dampens/preserves layers.

    We patch save_model_swap and start_experiment to avoid FK ordering issues
    in the DB (the swap event references the new experiment run before it's
    created). The tests focus on verifying the reset/dampen/preserve logic
    of the three learning layers.
    """

    @pytest.mark.asyncio
    @patch("src.learning.model_swap.start_experiment", new_callable=AsyncMock)
    async def test_model_swap_resets_calibration(
        self, mock_start_exp, db, sample_trade_record
    ):
        """Test 27: Model swap resets calibration (all alpha=1, beta=1)."""
        cal = CalibrationManager()
        mkt = MarketTypeManager()
        sig = SignalTrackerManager()

        await _seed_learning(db, sample_trade_record, cal, mkt, sig)

        # Verify calibration was updated
        bucket = cal.find_bucket(0.75)
        assert bucket.alpha > 1.0

        # Patch save_model_swap to skip the FK-constrained INSERT
        db.save_model_swap = AsyncMock()

        await handle_model_swap(
            old_model="grok-3-fast",
            new_model="grok-3-v2",
            reason="test swap",
            calibration_mgr=cal,
            market_type_mgr=mkt,
            db=db,
        )

        # All calibration buckets should be reset
        for b in cal.buckets:
            assert b.alpha == 1.0, f"Bucket {b.bucket_range} alpha not reset after swap"
            assert b.beta == 1.0, f"Bucket {b.bucket_range} beta not reset after swap"

    @pytest.mark.asyncio
    @patch("src.learning.model_swap.start_experiment", new_callable=AsyncMock)
    async def test_model_swap_dampens_market_type(
        self, mock_start_exp, db, sample_trade_record
    ):
        """Test 28: Model swap dampens market-type (keeps last 15 brier scores)."""
        cal = CalibrationManager()
        mkt = MarketTypeManager()
        sig = SignalTrackerManager()

        # Feed 25 trades to market type
        for i in range(25):
            record = sample_trade_record(
                record_id=f"mkt-{i}",
                market_type="political",
                action="BUY_YES",
                actual_outcome=True,
                brier_score_adjusted=float(i) / 100.0,
                voided=False,
            )
            mkt.update_market_type(record)
            await db.save_trade(record)

        assert len(mkt.performances["political"].brier_scores) == 25

        db.save_model_swap = AsyncMock()

        await handle_model_swap(
            old_model="grok-3-fast",
            new_model="grok-3-v2",
            reason="test swap",
            calibration_mgr=cal,
            market_type_mgr=mkt,
            db=db,
        )

        # Should keep only last 15
        assert len(mkt.performances["political"].brier_scores) == 15

    @pytest.mark.asyncio
    @patch("src.learning.model_swap.start_experiment", new_callable=AsyncMock)
    async def test_model_swap_preserves_signal_trackers(
        self, mock_start_exp, db, sample_trade_record
    ):
        """Test 29: Model swap preserves signal trackers (no reset/dampen)."""
        cal = CalibrationManager()
        mkt = MarketTypeManager()
        sig = SignalTrackerManager()

        await _seed_learning(db, sample_trade_record, cal, mkt, sig)

        # Snapshot signal tracker state before swap
        tracker_before = sig.trackers.get(("S2", "I2", "political"))
        assert tracker_before is not None
        present_winning_before = tracker_before.present_in_winning_trades

        db.save_model_swap = AsyncMock()

        await handle_model_swap(
            old_model="grok-3-fast",
            new_model="grok-3-v2",
            reason="test swap",
            calibration_mgr=cal,
            market_type_mgr=mkt,
            db=db,
        )

        # Signal trackers should be untouched
        tracker_after = sig.trackers.get(("S2", "I2", "political"))
        assert tracker_after is not None
        assert tracker_after.present_in_winning_trades == present_winning_before


# ---------------------------------------------------------------------------
# void_trade
# ---------------------------------------------------------------------------


class TestVoidTrade:
    """Tests 30-31: void_trade and on_trade_resolved voided handling."""

    @pytest.mark.asyncio
    async def test_void_trade_marks_voided(self, db, sample_trade_record):
        """Test 30: void_trade marks the trade record as voided=True."""
        cal = CalibrationManager()
        mkt = MarketTypeManager()
        sig = SignalTrackerManager()

        record = await _seed_learning(db, sample_trade_record, cal, mkt, sig)

        await void_trade(
            trade_id="seed-001",
            reason="test void",
            db=db,
            calibration_mgr=cal,
            market_type_mgr=mkt,
            signal_tracker_mgr=sig,
        )

        # Verify the trade in DB is now voided
        updated = await db.get_trade("seed-001")
        assert updated is not None
        assert updated.voided is True

    @pytest.mark.asyncio
    async def test_on_trade_resolved_skips_voided(self, db, sample_trade_record):
        """Test 31: on_trade_resolved skips voided trades entirely."""
        cal = CalibrationManager()
        mkt = MarketTypeManager()
        sig = SignalTrackerManager()

        record = sample_trade_record(
            record_id="voided-001",
            market_type="political",
            grok_raw_probability=0.80,
            grok_raw_confidence=0.75,
            final_adjusted_probability=0.78,
            final_adjusted_confidence=0.74,
            grok_signal_types=[{"source_tier": "S2", "info_type": "I2"}],
            action="BUY_YES",
            actual_outcome=True,
            voided=True,  # voided!
        )
        await db.save_trade(record)

        await on_trade_resolved(
            record=record,
            calibration_mgr=cal,
            market_type_mgr=mkt,
            signal_tracker_mgr=sig,
            db=db,
        )

        # Calibration should not have been touched
        for b in cal.buckets:
            assert b.alpha == 1.0
            assert b.beta == 1.0

        # Market type should have no entries
        assert len(mkt.performances) == 0

        # Signal trackers should have no entries
        assert len(sig.trackers) == 0


# ---------------------------------------------------------------------------
# handle_model_swap DB interactions
# ---------------------------------------------------------------------------


class TestHandleModelSwapDBInteractions:
    """Tests 51-52: Verify handle_model_swap creates DB records."""

    @pytest.mark.asyncio
    @patch("src.learning.model_swap.start_experiment", new_callable=AsyncMock)
    async def test_model_swap_starts_new_experiment(
        self, mock_start_exp, db, sample_trade_record
    ):
        """Test 51: handle_model_swap: New experiment run created in DB
        (verify start_experiment called)."""
        cal = CalibrationManager()
        mkt = MarketTypeManager()

        db.save_model_swap = AsyncMock()

        await handle_model_swap(
            old_model="grok-3-fast",
            new_model="grok-3-v2",
            reason="test swap",
            calibration_mgr=cal,
            market_type_mgr=mkt,
            db=db,
        )

        # start_experiment should have been called exactly once
        mock_start_exp.assert_called_once()
        call_kwargs = mock_start_exp.call_args
        # Verify it was called with new_model info
        assert call_kwargs.kwargs["model"] == "grok-3-v2"
        assert "grok-3-fast" in call_kwargs.kwargs["description"]
        assert "grok-3-v2" in call_kwargs.kwargs["description"]
        assert call_kwargs.kwargs["config"] == {
            "old_model": "grok-3-fast",
            "new_model": "grok-3-v2",
        }

    @pytest.mark.asyncio
    @patch("src.learning.model_swap.start_experiment", new_callable=AsyncMock)
    async def test_model_swap_saves_event_to_db(
        self, mock_start_exp, db, sample_trade_record
    ):
        """Test 52: handle_model_swap: ModelSwapEvent saved to DB
        (verify db.save_model_swap called with correct event)."""
        cal = CalibrationManager()
        mkt = MarketTypeManager()

        db.save_model_swap = AsyncMock()

        await handle_model_swap(
            old_model="grok-3-fast",
            new_model="grok-3-v2",
            reason="upgrade available",
            calibration_mgr=cal,
            market_type_mgr=mkt,
            db=db,
        )

        # save_model_swap should have been called once with a ModelSwapEvent
        db.save_model_swap.assert_called_once()
        event = db.save_model_swap.call_args[0][0]
        assert isinstance(event, ModelSwapEvent)
        assert event.old_model == "grok-3-fast"
        assert event.new_model == "grok-3-v2"
        assert event.reason == "upgrade available"
        assert event.experiment_run_started.startswith("exp_grok-3-v2_")


# ---------------------------------------------------------------------------
# recalculate_learning_from_scratch
# ---------------------------------------------------------------------------


class TestRecalculateLearning:
    """Test 55: recalculate_learning_from_scratch rebuilds all three layers."""

    @pytest.mark.asyncio
    async def test_recalculate_rebuilds_from_non_voided_trades(
        self, db, sample_trade_record
    ):
        """Test 55: recalculate_learning_from_scratch: rebuilds all three layers
        from non-voided trades."""
        cal = CalibrationManager()
        mkt = MarketTypeManager()
        sig = SignalTrackerManager()

        # Seed two resolved trades (one voided, one not)
        good_record = sample_trade_record(
            record_id="good-001",
            market_type="political",
            grok_raw_probability=0.80,
            grok_raw_confidence=0.75,
            final_adjusted_probability=0.78,
            final_adjusted_confidence=0.74,
            grok_signal_types=[{"source_tier": "S2", "info_type": "I2"}],
            action="BUY_YES",
            actual_outcome=True,
            brier_score_raw=0.04,
            brier_score_adjusted=0.05,
            voided=False,
        )
        voided_record = sample_trade_record(
            record_id="voided-001",
            market_type="political",
            grok_raw_probability=0.60,
            grok_raw_confidence=0.65,
            final_adjusted_probability=0.58,
            final_adjusted_confidence=0.63,
            grok_signal_types=[{"source_tier": "S3", "info_type": "I3"}],
            action="BUY_YES",
            actual_outcome=False,
            brier_score_raw=0.36,
            brier_score_adjusted=0.34,
            voided=True,
            void_reason="test void",
        )

        await db.save_trade(good_record)
        await db.save_trade(voided_record)

        # Initially populate managers with both trades
        cal.update_calibration(good_record)
        cal.update_calibration(voided_record)  # voided -> skipped by update_calibration
        mkt.update_market_type(good_record)
        sig.update_signal_trackers(good_record)

        # Verify there is data before recalculation
        assert len(sig.trackers) > 0

        # Recalculate from scratch
        await recalculate_learning_from_scratch(db, cal, mkt, sig)

        # After recalculation, only the good (non-voided) record should be used.
        # Calibration: bucket for 0.75 should have been updated with good_record
        bucket = cal.find_bucket(0.75)
        # The good record was correct (predicted yes at 0.80, actual=True)
        # so alpha should have been incremented
        assert bucket.alpha > 1.0

        # Market type: should have 1 trade
        assert mkt.performances["political"].total_trades == 1

        # Signal trackers: only S2/I2 combo from non-voided trade
        assert ("S2", "I2", "political") in sig.trackers


# ---------------------------------------------------------------------------
# on_trade_resolved - Brier scores and layer updates
# ---------------------------------------------------------------------------


class TestOnTradeResolved:
    """Tests 56-61: on_trade_resolved calculates Brier scores and updates all layers."""

    @pytest.mark.asyncio
    async def test_brier_score_raw_calculated(self, db, sample_trade_record):
        """Test 56: on_trade_resolved: brier_score_raw calculated as
        (grok_raw_probability - actual)^2."""
        cal = CalibrationManager()
        mkt = MarketTypeManager()
        sig = SignalTrackerManager()

        record = sample_trade_record(
            record_id="brier-raw-001",
            market_type="political",
            grok_raw_probability=0.80,
            grok_raw_confidence=0.75,
            final_adjusted_probability=0.78,
            final_adjusted_confidence=0.74,
            grok_signal_types=[{"source_tier": "S2", "info_type": "I2"}],
            action="BUY_YES",
            actual_outcome=True,
            brier_score_raw=None,  # Not pre-calculated
            brier_score_adjusted=None,
            voided=False,
        )
        await db.save_trade(record)

        with patch.object(cal, "save", new_callable=AsyncMock), \
             patch.object(mkt, "save", new_callable=AsyncMock), \
             patch.object(sig, "save", new_callable=AsyncMock), \
             patch("src.learning.adjustment.db", db, create=True):
            await on_trade_resolved(
                record=record,
                calibration_mgr=cal,
                market_type_mgr=mkt,
                signal_tracker_mgr=sig,
                db=db,
            )

        # actual=True -> actual_val=1.0
        # brier_raw = (0.80 - 1.0)^2 = 0.04
        assert record.brier_score_raw == pytest.approx(0.04, abs=1e-9)

    @pytest.mark.asyncio
    async def test_brier_score_adjusted_calculated(self, db, sample_trade_record):
        """Test 57: on_trade_resolved: brier_score_adjusted calculated as
        (final_adjusted_probability - actual)^2."""
        cal = CalibrationManager()
        mkt = MarketTypeManager()
        sig = SignalTrackerManager()

        record = sample_trade_record(
            record_id="brier-adj-001",
            market_type="political",
            grok_raw_probability=0.80,
            grok_raw_confidence=0.75,
            final_adjusted_probability=0.70,
            final_adjusted_confidence=0.74,
            grok_signal_types=[{"source_tier": "S2", "info_type": "I2"}],
            action="BUY_YES",
            actual_outcome=False,
            brier_score_raw=None,
            brier_score_adjusted=None,
            voided=False,
        )
        await db.save_trade(record)

        with patch.object(cal, "save", new_callable=AsyncMock), \
             patch.object(mkt, "save", new_callable=AsyncMock), \
             patch.object(sig, "save", new_callable=AsyncMock):
            await on_trade_resolved(
                record=record,
                calibration_mgr=cal,
                market_type_mgr=mkt,
                signal_tracker_mgr=sig,
                db=db,
            )

        # actual=False -> actual_val=0.0
        # brier_adjusted = (0.70 - 0.0)^2 = 0.49
        assert record.brier_score_adjusted == pytest.approx(0.49, abs=1e-9)

    @pytest.mark.asyncio
    async def test_calibration_updated_with_raw_correctness(
        self, db, sample_trade_record
    ):
        """Test 58: on_trade_resolved: calibration updated with RAW correctness
        (verify calibration_mgr.update_calibration called)."""
        cal = CalibrationManager()
        mkt = MarketTypeManager()
        sig = SignalTrackerManager()

        record = sample_trade_record(
            record_id="cal-update-001",
            market_type="political",
            grok_raw_probability=0.80,
            grok_raw_confidence=0.75,
            final_adjusted_probability=0.78,
            final_adjusted_confidence=0.74,
            grok_signal_types=[{"source_tier": "S2", "info_type": "I2"}],
            action="BUY_YES",
            actual_outcome=True,
            brier_score_raw=0.04,
            brier_score_adjusted=0.05,
            voided=False,
        )
        await db.save_trade(record)

        with patch.object(cal, "update_calibration", wraps=cal.update_calibration) as mock_update, \
             patch.object(cal, "save", new_callable=AsyncMock), \
             patch.object(mkt, "save", new_callable=AsyncMock), \
             patch.object(sig, "save", new_callable=AsyncMock):
            await on_trade_resolved(
                record=record,
                calibration_mgr=cal,
                market_type_mgr=mkt,
                signal_tracker_mgr=sig,
                db=db,
            )

        # update_calibration should have been called with the record
        mock_update.assert_called_once_with(record)

        # Verify the bucket for raw confidence 0.75 was updated
        bucket = cal.find_bucket(0.75)
        # Raw prediction: 0.80 > 0.5 predicted YES, actual=True -> was_correct
        assert bucket.alpha > 1.0

    @pytest.mark.asyncio
    async def test_market_type_updated_with_adjusted_brier(
        self, db, sample_trade_record
    ):
        """Test 59: on_trade_resolved: market-type updated with ADJUSTED Brier
        (verify market_type_mgr.update_market_type called)."""
        cal = CalibrationManager()
        mkt = MarketTypeManager()
        sig = SignalTrackerManager()

        record = sample_trade_record(
            record_id="mkt-update-001",
            market_type="political",
            grok_raw_probability=0.80,
            grok_raw_confidence=0.75,
            final_adjusted_probability=0.78,
            final_adjusted_confidence=0.74,
            grok_signal_types=[{"source_tier": "S2", "info_type": "I2"}],
            action="BUY_YES",
            actual_outcome=True,
            brier_score_raw=0.04,
            brier_score_adjusted=0.05,
            voided=False,
        )
        await db.save_trade(record)

        with patch.object(mkt, "update_market_type", wraps=mkt.update_market_type) as mock_update, \
             patch.object(cal, "save", new_callable=AsyncMock), \
             patch.object(mkt, "save", new_callable=AsyncMock), \
             patch.object(sig, "save", new_callable=AsyncMock):
            await on_trade_resolved(
                record=record,
                calibration_mgr=cal,
                market_type_mgr=mkt,
                signal_tracker_mgr=sig,
                db=db,
            )

        # update_market_type should have been called
        mock_update.assert_called_once()

        # Verify the brier score used is the ADJUSTED one (0.05)
        perf = mkt.performances["political"]
        assert perf.total_trades == 1
        assert perf.brier_scores == [0.05]

    @pytest.mark.asyncio
    async def test_signal_trackers_updated(self, db, sample_trade_record):
        """Test 60: on_trade_resolved: signal trackers updated
        (verify signal_tracker_mgr.update_signal_trackers called)."""
        cal = CalibrationManager()
        mkt = MarketTypeManager()
        sig = SignalTrackerManager()

        record = sample_trade_record(
            record_id="sig-update-001",
            market_type="political",
            grok_raw_probability=0.80,
            grok_raw_confidence=0.75,
            final_adjusted_probability=0.78,
            final_adjusted_confidence=0.74,
            grok_signal_types=[{"source_tier": "S2", "info_type": "I2"}],
            action="BUY_YES",
            actual_outcome=True,
            brier_score_raw=0.04,
            brier_score_adjusted=0.05,
            voided=False,
        )
        await db.save_trade(record)

        with patch.object(sig, "update_signal_trackers", wraps=sig.update_signal_trackers) as mock_update, \
             patch.object(cal, "save", new_callable=AsyncMock), \
             patch.object(mkt, "save", new_callable=AsyncMock), \
             patch.object(sig, "save", new_callable=AsyncMock):
            await on_trade_resolved(
                record=record,
                calibration_mgr=cal,
                market_type_mgr=mkt,
                signal_tracker_mgr=sig,
                db=db,
            )

        # update_signal_trackers should have been called with the record
        mock_update.assert_called_once_with(record)

        # Verify the signal tracker was created for S2/I2/political
        tracker = sig.trackers.get(("S2", "I2", "political"))
        assert tracker is not None
        # adjusted prediction: 0.78 > 0.5 -> predicted YES, actual=True -> correct
        assert tracker.present_in_winning_trades == 1

    @pytest.mark.asyncio
    async def test_all_changes_persisted_to_db(self, db, sample_trade_record):
        """Test 61: on_trade_resolved: all changes persisted to DB
        (verify save calls)."""
        cal = CalibrationManager()
        mkt = MarketTypeManager()
        sig = SignalTrackerManager()

        record = sample_trade_record(
            record_id="persist-001",
            market_type="political",
            grok_raw_probability=0.80,
            grok_raw_confidence=0.75,
            final_adjusted_probability=0.78,
            final_adjusted_confidence=0.74,
            grok_signal_types=[{"source_tier": "S2", "info_type": "I2"}],
            action="BUY_YES",
            actual_outcome=True,
            brier_score_raw=0.04,
            brier_score_adjusted=0.05,
            voided=False,
        )
        await db.save_trade(record)

        with patch.object(cal, "save", new_callable=AsyncMock) as mock_cal_save, \
             patch.object(mkt, "save", new_callable=AsyncMock) as mock_mkt_save, \
             patch.object(sig, "save", new_callable=AsyncMock) as mock_sig_save, \
             patch.object(db, "update_trade", new_callable=AsyncMock) as mock_update_trade:
            await on_trade_resolved(
                record=record,
                calibration_mgr=cal,
                market_type_mgr=mkt,
                signal_tracker_mgr=sig,
                db=db,
            )

        # All three managers should have saved to DB
        mock_cal_save.assert_called_once_with(db)
        mock_mkt_save.assert_called_once_with(db)
        mock_sig_save.assert_called_once_with(db)

        # Trade record should have been updated in DB
        mock_update_trade.assert_called_once_with(record)
