"""Tests for experiment auto-creation on startup and FK constraint validation."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import pytest
import pytest_asyncio

from src.db.migrations import run_migrations
from src.db.sqlite import Database
from src.learning.experiments import get_current_experiment, start_experiment
from src.models import TradeRecord


@pytest_asyncio.fixture
async def db():
    database = await Database.init(":memory:")
    await run_migrations(database)
    yield database
    await database.close()


class TestExperimentAutoCreation:
    """Verify that startup creates an experiment run if none exists."""

    @pytest.mark.asyncio
    async def test_no_experiment_exists_initially(self, db):
        """Fresh DB should have no active experiment."""
        experiment = await get_current_experiment(db)
        assert experiment is None

    @pytest.mark.asyncio
    async def test_create_experiment_on_empty_db(self, db):
        """Creating an experiment on empty DB should succeed and be retrievable."""
        run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        await start_experiment(
            run_id=run_id,
            description="Auto-created on startup",
            config={"test": True},
            model="grok-3-fast",
            db=db,
        )

        experiment = await get_current_experiment(db)
        assert experiment is not None
        assert experiment.run_id == run_id
        assert experiment.model_used == "grok-3-fast"
        assert experiment.ended_at is None

    @pytest.mark.asyncio
    async def test_existing_experiment_not_duplicated(self, db):
        """When an active experiment already exists, check returns it (no duplicate needed)."""
        await start_experiment(
            run_id="existing-run",
            description="Pre-existing",
            config={},
            model="grok-3-fast",
            db=db,
        )

        existing = await get_current_experiment(db)
        assert existing is not None
        assert existing.run_id == "existing-run"

    @pytest.mark.asyncio
    async def test_trade_saves_with_valid_experiment(self, db):
        """Trade records save successfully when a valid experiment run exists."""
        await start_experiment(
            run_id="valid-run",
            description="test",
            config={},
            model="grok-3-fast",
            db=db,
        )

        record = TradeRecord(
            record_id=str(uuid.uuid4()),
            experiment_run="valid-run",
            timestamp=datetime.now(timezone.utc),
            model_used="grok-3-fast",
            market_id="market-001",
            market_question="Will X happen?",
            market_type="political",
            resolution_window_hours=12.0,
            tier=1,
            grok_raw_probability=0.60,
            grok_raw_confidence=0.70,
            grok_reasoning="test reasoning",
            grok_signal_types=[],
            final_adjusted_probability=0.58,
            final_adjusted_confidence=0.68,
            market_price_at_decision=0.50,
            fee_rate=0.0,
            calculated_edge=0.08,
            action="BUY_YES",
        )
        await db.save_trade(record)

        saved = await db.get_trade(record.record_id)
        assert saved is not None
        assert saved.experiment_run == "valid-run"

    @pytest.mark.asyncio
    async def test_trade_fails_with_invalid_experiment(self, db):
        """Trade records MUST fail when experiment_run references non-existent run (FK constraint)."""
        record = TradeRecord(
            record_id=str(uuid.uuid4()),
            experiment_run="nonexistent-run",
            timestamp=datetime.now(timezone.utc),
            model_used="grok-3-fast",
            market_id="market-001",
            market_question="Will X happen?",
            market_type="political",
            resolution_window_hours=12.0,
            tier=1,
            grok_raw_probability=0.60,
            grok_raw_confidence=0.70,
            grok_reasoning="test reasoning",
            grok_signal_types=[],
            final_adjusted_probability=0.58,
            final_adjusted_confidence=0.68,
            market_price_at_decision=0.50,
            fee_rate=0.0,
            calculated_edge=0.08,
            action="SKIP",
        )
        with pytest.raises(Exception):
            await db.save_trade(record)
