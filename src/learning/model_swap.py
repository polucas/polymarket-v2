from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

import structlog

from src.db.sqlite import Database
from src.learning.calibration import CalibrationManager
from src.learning.experiments import start_experiment
from src.learning.market_type import MarketTypeManager
from src.learning.signal_tracker import SignalTrackerManager
from src.models import ModelSwapEvent

log = structlog.get_logger()


async def handle_model_swap(
    old_model: str,
    new_model: str,
    reason: str,
    calibration_mgr: CalibrationManager,
    market_type_mgr: MarketTypeManager,
    db: Database,
) -> None:
    """Execute model swap protocol.
    1. Save ModelSwapEvent
    2. Start new experiment
    3. RESET calibration to priors
    4. DAMPEN market-type (keep last 15 Brier scores)
    5. PRESERVE signal trackers (no action)
    """
    now = datetime.now(timezone.utc)
    run_id = f"exp_{new_model}_{now.strftime('%Y%m%d_%H%M%S')}"

    # 1. Save event
    event = ModelSwapEvent(
        timestamp=now,
        old_model=old_model,
        new_model=new_model,
        reason=reason,
        experiment_run_started=run_id,
    )
    await db.save_model_swap(event)

    # 2. Start new experiment
    await start_experiment(
        run_id=run_id,
        description=f"Model swap: {old_model} -> {new_model}. Reason: {reason}",
        config={"old_model": old_model, "new_model": new_model},
        model=new_model,
        db=db,
    )

    # 3. RESET calibration
    calibration_mgr.reset_to_priors()
    await calibration_mgr.save(db)

    # 4. DAMPEN market-type
    market_type_mgr.dampen_on_swap()
    await market_type_mgr.save(db)

    # 5. Signal trackers are PRESERVED (no action needed)

    log.info("model_swap_complete",
             old_model=old_model,
             new_model=new_model,
             experiment_run=run_id)


async def void_trade(
    trade_id: str,
    reason: str,
    db: Database,
    calibration_mgr: CalibrationManager,
    market_type_mgr: MarketTypeManager,
    signal_tracker_mgr: SignalTrackerManager,
) -> None:
    """Void a trade and recalculate all learning from scratch."""
    record = await db.get_trade(trade_id)
    if record is None:
        log.error("void_trade_not_found", trade_id=trade_id)
        return

    record.voided = True
    record.void_reason = reason
    await db.update_trade(record)

    # Recalculate learning from scratch
    await recalculate_learning_from_scratch(db, calibration_mgr, market_type_mgr, signal_tracker_mgr)

    log.info("trade_voided", trade_id=trade_id, reason=reason)


async def recalculate_learning_from_scratch(
    db: Database,
    calibration_mgr: CalibrationManager,
    market_type_mgr: MarketTypeManager,
    signal_tracker_mgr: SignalTrackerManager,
) -> None:
    """Reload all non-voided resolved trades and rebuild all three learning layers."""
    # Reset all
    calibration_mgr.reset_to_priors()
    market_type_mgr.performances.clear()
    signal_tracker_mgr.trackers.clear()

    # Reload all resolved, non-voided trades
    trades = await db.get_all_resolved_trades(include_voided=False)

    for trade in trades:
        if trade.actual_outcome is None:
            continue
        calibration_mgr.update_calibration(trade)
        market_type_mgr.update_market_type(trade)
        signal_tracker_mgr.update_signal_trackers(trade)

    # Persist
    await calibration_mgr.save(db)
    await market_type_mgr.save(db)
    await signal_tracker_mgr.save(db)

    log.info("learning_recalculated", trades_processed=len(trades))
