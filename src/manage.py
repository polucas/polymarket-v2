from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from datetime import datetime, timezone

import structlog

log = structlog.get_logger()


async def _init_deps():
    """Initialize database and learning managers."""
    from src.config import get_settings
    from src.db.sqlite import Database
    from src.db.migrations import run_migrations
    from src.learning.calibration import CalibrationManager
    from src.learning.market_type import MarketTypeManager
    from src.learning.signal_tracker import SignalTrackerManager

    settings = get_settings()
    db = await Database.init(settings.DB_PATH)
    await run_migrations(db)

    cal = CalibrationManager()
    await cal.load(db)
    mt = MarketTypeManager()
    await mt.load(db)
    st = SignalTrackerManager()
    await st.load(db)

    return db, cal, mt, st


async def cmd_model_swap(args):
    db, cal, mt, st = await _init_deps()
    try:
        from src.learning.model_swap import handle_model_swap
        await handle_model_swap(args.old_model, args.new_model, args.reason, cal, mt, db)
        print(f"Model swap complete: {args.old_model} -> {args.new_model}")
    finally:
        await db.close()


async def cmd_void_trade(args):
    db, cal, mt, st = await _init_deps()
    try:
        from src.learning.model_swap import void_trade
        await void_trade(args.trade_id, args.reason, db, cal, mt, st)
        print(f"Trade voided: {args.trade_id}")
    finally:
        await db.close()


async def cmd_start_experiment(args):
    db, cal, mt, st = await _init_deps()
    try:
        from src.learning.experiments import start_experiment
        run_id = f"exp_{args.model}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        await start_experiment(run_id, args.description, {}, args.model, db)
        print(f"Experiment started: {run_id}")
    finally:
        await db.close()


async def cmd_end_experiment(args):
    db, cal, mt, st = await _init_deps()
    try:
        from src.learning.experiments import end_experiment
        await end_experiment(args.run_id, {}, db)
        print(f"Experiment ended: {args.run_id}")
    finally:
        await db.close()


async def cmd_recalculate_learning(args):
    db, cal, mt, st = await _init_deps()
    try:
        from src.learning.model_swap import recalculate_learning_from_scratch
        await recalculate_learning_from_scratch(db, cal, mt, st)
        print("Learning recalculated from scratch.")
    finally:
        await db.close()


def main():
    parser = argparse.ArgumentParser(description="Polymarket v2 Management CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    # model_swap
    p_swap = sub.add_parser("model_swap", help="Swap the active model")
    p_swap.add_argument("--old-model", required=True)
    p_swap.add_argument("--new-model", required=True)
    p_swap.add_argument("--reason", required=True)

    # void_trade
    p_void = sub.add_parser("void_trade", help="Void a trade")
    p_void.add_argument("--trade-id", required=True)
    p_void.add_argument("--reason", required=True)

    # start_experiment
    p_start = sub.add_parser("start_experiment", help="Start a new experiment")
    p_start.add_argument("--description", required=True)
    p_start.add_argument("--model", required=True)

    # end_experiment
    p_end = sub.add_parser("end_experiment", help="End an experiment")
    p_end.add_argument("--run-id", required=True)

    # recalculate_learning
    sub.add_parser("recalculate_learning", help="Recalculate all learning from scratch")

    args = parser.parse_args()

    cmd_map = {
        "model_swap": cmd_model_swap,
        "void_trade": cmd_void_trade,
        "start_experiment": cmd_start_experiment,
        "end_experiment": cmd_end_experiment,
        "recalculate_learning": cmd_recalculate_learning,
    }

    asyncio.run(cmd_map[args.command](args))


if __name__ == "__main__":
    main()
