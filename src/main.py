from __future__ import annotations

import json
import os
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import logging

import structlog
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from src.alerts import format_lifecycle_alert, send_alert
from src.config import Settings, get_settings
from src.db.migrations import run_migrations
from src.db.sqlite import Database
from src.engine.grok_client import GrokClient
from src.learning.calibration import CalibrationManager
from src.learning.market_type import MarketTypeManager
from src.learning.signal_tracker import SignalTrackerManager
from src.pipelines.polymarket import PolymarketClient
from src.pipelines.rss import RSSPipeline
from src.pipelines.twitter import TwitterDataPipeline
from src.learning.experiments import get_current_experiment, start_experiment
from src.scheduler import Scheduler

# ---------------------------------------------------------------------------
# Structlog configuration
# ---------------------------------------------------------------------------

_LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

# Structlog → stdlib logging bridge
_shared_processors = [
    structlog.contextvars.merge_contextvars,
    structlog.processors.add_log_level,
    structlog.processors.TimeStamper(fmt="iso"),
    structlog.processors.StackInfoRenderer(),
    structlog.processors.format_exc_info,
    structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
]

structlog.configure(
    processors=_shared_processors,
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

# JSON formatter for both handlers
_formatter = structlog.stdlib.ProcessorFormatter(
    processor=structlog.processors.JSONRenderer(),
)

# File handler
_log_dir = os.path.dirname(get_settings().DB_PATH) or "data"
os.makedirs(_log_dir, exist_ok=True)
_file_handler = logging.FileHandler(os.path.join(_log_dir, "bot.log"))
_file_handler.setFormatter(_formatter)

# Stdout handler
_stream_handler = logging.StreamHandler(sys.stdout)
_stream_handler.setFormatter(_formatter)

# Configure root logger
root_logger = logging.getLogger()
root_logger.handlers.clear()
root_logger.addHandler(_file_handler)
root_logger.addHandler(_stream_handler)
root_logger.setLevel(
    _LOG_LEVELS.get(get_settings().LOG_LEVEL.upper(), logging.INFO)
)

log = structlog.get_logger()

# ---------------------------------------------------------------------------
# App state (populated during lifespan)
# ---------------------------------------------------------------------------

_app_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown logic."""
    settings = get_settings()
    log.info("starting_up", environment=settings.ENVIRONMENT)

    # Ensure data directory exists
    db_dir = os.path.dirname(settings.DB_PATH)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    # Database
    db = await Database.init(settings.DB_PATH)
    await run_migrations(db)

    # Ensure an active experiment run exists
    experiment = await get_current_experiment(db)
    if experiment is None:
        run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        await start_experiment(
            run_id=run_id,
            description="Auto-created on startup",
            config=settings.safe_config(),
            model="grok-3-fast",
            db=db,
        )
        log.info("experiment_created", run_id=run_id)
    else:
        log.info("experiment_exists", run_id=experiment.run_id)

    # Learning managers
    calibration_mgr = CalibrationManager()
    await calibration_mgr.load(db)
    market_type_mgr = MarketTypeManager()
    await market_type_mgr.load(db)
    signal_tracker_mgr = SignalTrackerManager()
    await signal_tracker_mgr.load(db)

    # Clients
    polymarket = PolymarketClient(settings)
    twitter = TwitterDataPipeline(settings)
    rss = RSSPipeline()
    grok = GrokClient(settings, db)

    # Scheduler
    scheduler = Scheduler(
        settings=settings,
        db=db,
        polymarket=polymarket,
        twitter=twitter,
        rss=rss,
        grok=grok,
        calibration_mgr=calibration_mgr,
        market_type_mgr=market_type_mgr,
        signal_tracker_mgr=signal_tracker_mgr,
    )
    scheduler.start()

    _app_state.update(
        db=db,
        settings=settings,
        scheduler=scheduler,
        portfolio=await db.load_portfolio(),
        started_at=time.time(),
    )

    log.info("startup_complete")
    await send_alert(format_lifecycle_alert("STARTED", settings.ENVIRONMENT), settings)
    yield

    # Shutdown
    log.info("shutting_down")
    await send_alert(format_lifecycle_alert("STOPPING", settings.ENVIRONMENT), settings)
    scheduler.stop()
    await db.close()
    log.info("shutdown_complete")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(title="Polymarket Predictor v2", lifespan=lifespan)


@app.get("/health")
async def health():
    """Health check endpoint.
    200 if last scan <30 min ago, 503 if stale.
    """
    scheduler: Scheduler = _app_state.get("scheduler")
    settings: Settings = _app_state.get("settings")
    db: Database = _app_state.get("db")
    started_at: float = _app_state.get("started_at", time.time())

    now = datetime.now(timezone.utc)
    last_scan = scheduler.last_scan_completed if scheduler else None
    minutes_since = None
    stale = True

    if last_scan:
        minutes_since = (now - last_scan).total_seconds() / 60
        stale = minutes_since > 30

    open_trades = await db.count_open_trades() if db else 0
    today_trades = await db.count_today_trades() if db else 0
    uptime_hours = (time.time() - started_at) / 3600

    mode = "paper"
    if settings:
        mode = settings.ENVIRONMENT

    body = {
        "status": "ok" if not stale else "stale",
        "last_scan_completed": last_scan.isoformat() if last_scan else None,
        "minutes_since_scan": round(minutes_since, 1) if minutes_since is not None else None,
        "mode": mode,
        "open_trades": open_trades,
        "today_trades": today_trades,
        "uptime_hours": round(uptime_hours, 2),
    }

    status_code = 200 if not stale else 503
    return JSONResponse(content=body, status_code=status_code)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=False)
