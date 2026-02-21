"""Tests for the /health endpoint logic.

Because the health endpoint reads from the module-level ``_app_state`` dict
and depends on ``Scheduler.last_scan_completed``, we test by patching
``_app_state`` directly and using ``httpx.AsyncClient`` with the FastAPI app.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx

from src.main import app, _app_state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_app_state(
    last_scan: datetime | None = None,
    environment: str = "paper",
    open_trades: int = 0,
    today_trades: int = 0,
) -> dict:
    """Build a mock _app_state dict."""
    scheduler = MagicMock()
    scheduler.last_scan_completed = last_scan

    settings = MagicMock()
    settings.ENVIRONMENT = environment

    db = AsyncMock()
    db.count_open_trades = AsyncMock(return_value=open_trades)
    db.count_today_trades = AsyncMock(return_value=today_trades)

    return {
        "scheduler": scheduler,
        "settings": settings,
        "db": db,
        "portfolio": MagicMock(),
        "started_at": time.time() - 3600,  # 1 hour ago
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_response_has_all_expected_fields():
    """Response body must contain every documented field."""
    now = datetime.now(timezone.utc)
    state = _patch_app_state(last_scan=now)

    with patch.dict("src.main._app_state", state, clear=True):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            resp = await client.get("/health")

    body = resp.json()
    expected_keys = {
        "status",
        "last_scan_completed",
        "minutes_since_scan",
        "mode",
        "open_trades",
        "today_trades",
        "uptime_hours",
    }
    assert expected_keys == set(body.keys())


@pytest.mark.asyncio
async def test_health_status_ok_when_recent_scan():
    """Status is 'ok' (HTTP 200) when last scan was less than 30 min ago."""
    recent = datetime.now(timezone.utc) - timedelta(minutes=5)
    state = _patch_app_state(last_scan=recent)

    with patch.dict("src.main._app_state", state, clear=True):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            resp = await client.get("/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["minutes_since_scan"] is not None
    assert body["minutes_since_scan"] < 30


@pytest.mark.asyncio
async def test_health_status_stale_when_no_scan():
    """Status is 'stale' (HTTP 503) when no scan has ever completed."""
    state = _patch_app_state(last_scan=None)

    with patch.dict("src.main._app_state", state, clear=True):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            resp = await client.get("/health")

    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "stale"
    assert body["last_scan_completed"] is None
    assert body["minutes_since_scan"] is None


@pytest.mark.asyncio
async def test_health_status_stale_when_scan_older_than_30_min():
    """Status is 'stale' (HTTP 503) when last scan was more than 30 min ago."""
    old = datetime.now(timezone.utc) - timedelta(minutes=45)
    state = _patch_app_state(last_scan=old)

    with patch.dict("src.main._app_state", state, clear=True):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            resp = await client.get("/health")

    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "stale"
    assert body["minutes_since_scan"] > 30


@pytest.mark.asyncio
async def test_health_reflects_environment_mode():
    """The 'mode' field reflects the Settings.ENVIRONMENT value."""
    now = datetime.now(timezone.utc)
    state = _patch_app_state(last_scan=now, environment="live")

    with patch.dict("src.main._app_state", state, clear=True):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            resp = await client.get("/health")

    assert resp.json()["mode"] == "live"


@pytest.mark.asyncio
async def test_health_includes_trade_counts():
    """open_trades and today_trades are populated from the database."""
    now = datetime.now(timezone.utc)
    state = _patch_app_state(last_scan=now, open_trades=3, today_trades=7)

    with patch.dict("src.main._app_state", state, clear=True):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            resp = await client.get("/health")

    body = resp.json()
    assert body["open_trades"] == 3
    assert body["today_trades"] == 7
