from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from src.db.sqlite import Database
from src.models import ExperimentRun


async def start_experiment(
    run_id: str,
    description: str,
    config: dict,
    model: str,
    db: Database,
) -> None:
    """Start a new experiment run."""
    run = ExperimentRun(
        run_id=run_id,
        started_at=datetime.now(timezone.utc),
        config_snapshot=config,
        description=description,
        model_used=model,
        include_in_learning=True,
    )
    await db.save_experiment(run)


async def end_experiment(run_id: str, stats: dict, db: Database) -> None:
    """End an experiment run with final stats."""
    await db.end_experiment(run_id, stats)


async def get_current_experiment(db: Database) -> Optional[ExperimentRun]:
    """Get the currently active experiment run."""
    return await db.get_current_experiment()
