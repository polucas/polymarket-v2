"""Tests for daily self-check loop (src/learning/self_check.py)."""

from __future__ import annotations

import os
import tempfile
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from src.db.migrations import run_migrations
from src.db.sqlite import Database
from src.learning.calibration import CalibrationManager
from src.learning.market_type import MarketTypeManager
from src.learning.self_check import (
    _build_llm_prompt,
    _gather_metrics,
    _write_markdown,
    format_self_check_alert,
    run_daily_self_check,
)
from src.learning.signal_tracker import SignalTrackerManager
from src.models import DailyReview, ExperimentRun


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def db():
    """In-memory SQLite database with schema applied."""
    database = await Database.init(":memory:")
    await run_migrations(database)
    yield database
    await database.close()


@pytest_asyncio.fixture
async def db_with_experiment(db):
    """DB with a seeded experiment run for FK compliance."""
    run = ExperimentRun(
        run_id="test-run-001",
        started_at=datetime.now(timezone.utc),
        model_used="grok-3-fast",
        description="Test experiment",
    )
    await db.save_experiment(run)
    return db


@pytest.fixture
def sample_trade_record():
    """Factory for TradeRecord with sensible defaults."""
    from src.models import TradeRecord

    def _make(**overrides):
        defaults = {
            "record_id": str(uuid.uuid4()),
            "experiment_run": "test-run-001",
            "timestamp": datetime.now(timezone.utc),
            "model_used": "grok-3-fast",
            "market_id": "market-001",
            "market_question": "Will X happen?",
            "market_type": "political",
            "resolution_window_hours": 12.0,
            "tier": 1,
            "grok_raw_probability": 0.75,
            "grok_raw_confidence": 0.80,
            "grok_reasoning": "Test reasoning",
            "grok_signal_types": [],
            "calibration_adjustment": 0.0,
            "market_type_adjustment": 0.0,
            "signal_weight_adjustment": 0.0,
            "final_adjusted_probability": 0.73,
            "final_adjusted_confidence": 0.78,
            "market_price_at_decision": 0.60,
            "orderbook_depth_usd": 5000.0,
            "fee_rate": 0.0,
            "calculated_edge": 0.11,
            "trade_score": 0.05,
            "action": "BUY_YES",
            "skip_reason": None,
            "position_size_usd": 100.0,
            "kelly_fraction_used": 0.25,
            "actual_outcome": None,
            "pnl": None,
            "brier_score_raw": None,
            "brier_score_adjusted": None,
            "resolved_at": None,
            "unrealized_adverse_move": None,
            "voided": False,
            "void_reason": None,
        }
        defaults.update(overrides)
        return TradeRecord(**defaults)

    return _make


# ---------------------------------------------------------------------------
# Test 1: gather_metrics on empty day
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gather_metrics_empty_day(db_with_experiment):
    """No trades on the review date → all zeroes / None."""
    metrics = await _gather_metrics(db_with_experiment, "2099-01-01")
    assert metrics["trade_count"] == 0
    assert metrics["skip_count"] == 0
    assert metrics["resolved_count"] == 0
    assert metrics["wins"] == 0
    assert metrics["total_pnl"] == 0.0
    assert metrics["avg_brier_raw"] is None
    assert metrics["avg_brier_adjusted"] is None
    assert metrics["win_rate"] is None
    assert metrics["roi_pct"] is None
    assert metrics["skip_reason_distribution"] == {}
    assert metrics["brier_by_market_type"] == {}
    assert metrics["top_performing_types"] == []
    assert metrics["worst_performing_types"] == []


# ---------------------------------------------------------------------------
# Test 2: gather_metrics with trades
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gather_metrics_with_trades(db_with_experiment, sample_trade_record):
    """Insert sample trades; verify aggregated stats are correct."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # 1 executed trade, resolved win
    await db_with_experiment.save_trade(
        sample_trade_record(
            action="BUY_YES",
            actual_outcome=True,
            pnl=20.0,
            brier_score_raw=0.05,
            brier_score_adjusted=0.06,
            position_size_usd=100.0,
            market_type="political",
        )
    )
    # 1 executed trade, resolved loss
    await db_with_experiment.save_trade(
        sample_trade_record(
            action="BUY_YES",
            actual_outcome=False,
            pnl=-10.0,
            brier_score_raw=0.25,
            brier_score_adjusted=0.27,
            position_size_usd=100.0,
            market_type="political",
        )
    )
    # 1 skip
    await db_with_experiment.save_trade(
        sample_trade_record(action="SKIP", skip_reason="low_edge_0.0200")
    )

    metrics = await _gather_metrics(db_with_experiment, today)
    assert metrics["trade_count"] == 2
    assert metrics["skip_count"] == 1
    assert metrics["resolved_count"] == 2
    assert metrics["wins"] == 1
    assert abs(metrics["total_pnl"] - 10.0) < 0.01
    assert metrics["win_rate"] == pytest.approx(0.5)
    assert metrics["roi_pct"] == pytest.approx(5.0)  # 10 / 200 * 100
    assert "political" in metrics["brier_by_market_type"]
    assert "low_edge_0.0200" in metrics["skip_reason_distribution"]


# ---------------------------------------------------------------------------
# Test 3: build_llm_prompt contains all required sections
# ---------------------------------------------------------------------------


def test_build_llm_prompt():
    """The generated prompt includes all key sections."""
    metrics = {
        "trade_count": 5,
        "skip_count": 12,
        "resolved_count": 3,
        "total_pnl": 15.50,
        "avg_brier_raw": 0.12,
        "avg_brier_adjusted": 0.14,
        "win_rate": 0.667,
        "roi_pct": 7.75,
        "brier_by_market_type": {
            "political": {"avg_brier_raw": 0.10, "pnl": 20.0, "count": 2}
        },
        "skip_reason_distribution": {"low_edge": 8, "grok_failed": 4},
        "top_performing_types": ["political"],
        "worst_performing_types": [],
    }
    prompt = _build_llm_prompt(metrics, "2026-03-19")

    assert "2026-03-19" in prompt
    assert "Executed trades: 5" in prompt
    assert "Skipped evaluations: 12" in prompt
    assert "Win rate:" in prompt
    assert "66.7%" in prompt
    assert "Avg Brier (raw): 0.120" in prompt
    assert "political" in prompt
    assert "low_edge" in prompt
    assert "grok_failed" in prompt
    assert "INSTRUCTIONS" in prompt
    assert '"insights"' in prompt
    assert '"health_status"' in prompt


# ---------------------------------------------------------------------------
# Test 4: format_self_check_alert produces valid Telegram message
# ---------------------------------------------------------------------------


def test_format_self_check_alert():
    """Telegram alert message includes all key fields."""
    review = DailyReview(
        review_date="2026-03-19",
        timestamp=datetime.now(timezone.utc),
        trade_count=5,
        skip_count=10,
        resolved_count=3,
        win_rate=0.667,
        roi_pct=7.5,
        total_pnl=15.0,
        avg_brier_raw=0.12,
        avg_brier_adjusted=0.14,
        health_status="HEALTHY",
        llm_insights="System performing well with strong edge detection.",
        llm_recommendations=["Reduce exposure to crypto markets", "Improve skip reasons logging"],
    )
    msg = format_self_check_alert(review)

    assert "DAILY SELF-CHECK" in msg
    assert "2026-03-19" in msg
    assert "HEALTHY" in msg
    assert "5 executed" in msg
    assert "10 skipped" in msg
    assert "3 resolved" in msg
    assert "67%" in msg
    assert "7.5%" in msg
    assert "$+15.00" in msg
    assert "0.120" in msg
    assert "0.140" in msg
    assert "System performing well" in msg
    assert "Reduce exposure to crypto markets" in msg
    assert "Improve skip reasons logging" in msg


def test_format_self_check_alert_none_values():
    """Alert handles None win_rate/roi_pct gracefully."""
    review = DailyReview(
        review_date="2026-03-19",
        timestamp=datetime.now(timezone.utc),
        health_status="UNKNOWN",
    )
    msg = format_self_check_alert(review)
    assert "N/A" in msg
    assert "UNKNOWN" in msg


# ---------------------------------------------------------------------------
# Test 5: write_markdown creates file with expected content
# ---------------------------------------------------------------------------


def test_write_markdown():
    """write_markdown creates a .md file with all sections."""
    review = DailyReview(
        review_date="2026-03-19",
        timestamp=datetime.now(timezone.utc),
        trade_count=3,
        skip_count=7,
        resolved_count=2,
        win_rate=0.5,
        roi_pct=5.0,
        total_pnl=10.0,
        avg_brier_raw=0.15,
        avg_brier_adjusted=0.17,
        health_status="CAUTION",
        brier_by_market_type={
            "political": {"avg_brier_raw": 0.15, "pnl": 10.0, "count": 2}
        },
        skip_reason_distribution={"low_edge": 5, "grok_failed": 2},
        llm_insights="Some pattern observed.",
        llm_recommendations=["Do this", "Do that"],
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("src.learning.self_check.REVIEW_DIR", tmpdir):
            _write_markdown(review)

        md_path = os.path.join(tmpdir, "2026-03-19.md")
        assert os.path.exists(md_path)

        content = open(md_path).read()
        assert "# Daily Review — 2026-03-19" in content
        assert "CAUTION" in content
        assert "| Executed trades | 3 |" in content
        assert "50.0%" in content
        assert "political" in content
        assert "low_edge" in content
        assert "Some pattern observed." in content
        assert "Do this" in content
        assert "Do that" in content


# ---------------------------------------------------------------------------
# Test 6: run_daily_self_check — Grok failure saves metrics-only review
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_daily_self_check_grok_failure(db_with_experiment):
    """When Grok fails, a review with health_status=UNKNOWN is still saved."""
    calibration_mgr = CalibrationManager()
    market_type_mgr = MarketTypeManager()
    signal_tracker_mgr = SignalTrackerManager()

    mock_grok = MagicMock()
    mock_grok.complete = AsyncMock(side_effect=Exception("API unavailable"))

    mock_settings = MagicMock()

    with patch("src.learning.self_check.REVIEW_DIR", tempfile.mkdtemp()):
        review = await run_daily_self_check(
            db_with_experiment,
            mock_grok,
            calibration_mgr,
            market_type_mgr,
            signal_tracker_mgr,
            mock_settings,
        )

    assert review.health_status == "UNKNOWN"
    assert review.llm_insights == ""
    assert review.llm_recommendations == []
    # Verify it was saved to DB
    loaded = await db_with_experiment.get_daily_review(review.review_date)
    assert loaded is not None
    assert loaded.health_status == "UNKNOWN"


# ---------------------------------------------------------------------------
# Test 7: daily_review_db_round_trip
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_daily_review_db_round_trip(db_with_experiment):
    """Save a DailyReview and load it back; verify data integrity."""
    review = DailyReview(
        review_date="2026-03-15",
        timestamp=datetime.now(timezone.utc),
        trade_count=8,
        skip_count=22,
        resolved_count=5,
        win_rate=0.6,
        roi_pct=12.5,
        total_pnl=100.0,
        avg_brier_raw=0.11,
        avg_brier_adjusted=0.13,
        brier_by_market_type={"political": {"avg_brier_raw": 0.11, "pnl": 100.0, "count": 5}},
        calibration_drift={"0.50-0.60": 0.02, "0.70-0.80": -0.01},
        signal_effectiveness={"('S2', 'I2', 'political')": 1.05},
        skip_reason_distribution={"low_edge": 15, "grok_failed": 7},
        top_performing_types=["political"],
        worst_performing_types=["crypto_15m"],
        llm_insights="The bot showed good performance on political markets.",
        llm_recommendations=["Increase exposure to political markets", "Review crypto thresholds"],
        health_status="HEALTHY",
        experiment_run="test-run-001",
    )

    await db_with_experiment.save_daily_review(review)
    loaded = await db_with_experiment.get_daily_review("2026-03-15")

    assert loaded is not None
    assert loaded.review_date == "2026-03-15"
    assert loaded.trade_count == 8
    assert loaded.skip_count == 22
    assert loaded.resolved_count == 5
    assert loaded.win_rate == pytest.approx(0.6)
    assert loaded.roi_pct == pytest.approx(12.5)
    assert loaded.total_pnl == pytest.approx(100.0)
    assert loaded.avg_brier_raw == pytest.approx(0.11)
    assert loaded.avg_brier_adjusted == pytest.approx(0.13)
    assert loaded.brier_by_market_type == {"political": {"avg_brier_raw": 0.11, "pnl": 100.0, "count": 5}}
    assert loaded.calibration_drift == {"0.50-0.60": 0.02, "0.70-0.80": -0.01}
    assert loaded.signal_effectiveness == {"('S2', 'I2', 'political')": 1.05}
    assert loaded.skip_reason_distribution == {"low_edge": 15, "grok_failed": 7}
    assert loaded.top_performing_types == ["political"]
    assert loaded.worst_performing_types == ["crypto_15m"]
    assert loaded.llm_insights == "The bot showed good performance on political markets."
    assert loaded.llm_recommendations == ["Increase exposure to political markets", "Review crypto thresholds"]
    assert loaded.health_status == "HEALTHY"
    assert loaded.experiment_run == "test-run-001"


@pytest.mark.asyncio
async def test_get_recent_reviews_returns_multiple(db_with_experiment):
    """get_recent_reviews returns all reviews within the window."""
    # Use recent dates (within the last 30-day window) so the cutoff query finds them
    from datetime import date, timedelta
    today = date.today()
    dates_to_insert = [
        (today - timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range(3)
    ]
    for review_date in dates_to_insert:
        review = DailyReview(
            review_date=review_date,
            timestamp=datetime.now(timezone.utc),
            health_status="HEALTHY",
            experiment_run="test-run-001",
        )
        await db_with_experiment.save_daily_review(review)

    reviews = await db_with_experiment.get_recent_reviews(days=30)
    assert len(reviews) == 3
    # Should be in descending order
    review_dates = [r.review_date for r in reviews]
    assert review_dates == sorted(review_dates, reverse=True)


@pytest.mark.asyncio
async def test_gather_metrics_includes_pnl(db_with_experiment, sample_trade_record):
    """_gather_metrics computes PnL-based metrics from trade_profitable column.

    Insert 3 trades with trade_profitable set (2 profitable, 1 not).
    Verify win_rate_pnl == 2/3, pnl_resolved_count == 3, and Brier averages in [0, 1].
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Trade 1: profitable (TP exit, no actual_outcome)
    await db_with_experiment.save_trade(
        sample_trade_record(
            action="BUY_YES",
            actual_outcome=None,
            pnl=20.0,
            exit_type="take_profit",
            exit_price=0.70,
            trade_profitable=1,
            pnl_brier_raw=0.09,
            pnl_brier_adjusted=0.08,
            position_size_usd=100.0,
        )
    )
    # Trade 2: profitable (natural resolution)
    await db_with_experiment.save_trade(
        sample_trade_record(
            action="BUY_YES",
            actual_outcome=True,
            pnl=30.0,
            brier_score_raw=0.05,
            brier_score_adjusted=0.06,
            trade_profitable=1,
            pnl_brier_raw=0.04,
            pnl_brier_adjusted=0.03,
            position_size_usd=100.0,
        )
    )
    # Trade 3: loss (SL exit)
    await db_with_experiment.save_trade(
        sample_trade_record(
            action="BUY_YES",
            actual_outcome=None,
            pnl=-15.0,
            exit_type="stop_loss",
            exit_price=0.40,
            trade_profitable=0,
            pnl_brier_raw=0.49,
            pnl_brier_adjusted=0.46,
            position_size_usd=100.0,
        )
    )

    metrics = await _gather_metrics(db_with_experiment, today)

    assert metrics["pnl_resolved_count"] == 3
    assert metrics["win_rate_pnl"] == pytest.approx(2 / 3)
    assert metrics["avg_pnl_brier_raw"] is not None
    assert metrics["avg_pnl_brier_adjusted"] is not None
    # Average of [0.09, 0.04, 0.49] = 0.62/3 ≈ 0.2067
    assert metrics["avg_pnl_brier_raw"] == pytest.approx((0.09 + 0.04 + 0.49) / 3, rel=1e-4)
    assert 0.0 <= metrics["avg_pnl_brier_raw"] <= 1.0
    assert 0.0 <= metrics["avg_pnl_brier_adjusted"] <= 1.0


@pytest.mark.asyncio
async def test_get_daily_review_not_found(db_with_experiment):
    """get_daily_review returns None for a date with no review."""
    result = await db_with_experiment.get_daily_review("2099-12-31")
    assert result is None


@pytest.mark.asyncio
async def test_get_period_trade_stats_empty(db_with_experiment):
    """get_period_trade_stats on a date with no trades returns all zeroes."""
    stats = await db_with_experiment.get_period_trade_stats("2099-01-01", "2099-01-01")
    assert stats["trade_count"] == 0
    assert stats["skip_count"] == 0
    assert stats["resolved_count"] == 0
    assert stats["total_pnl"] == 0.0
    assert stats["avg_brier_raw"] is None


@pytest.mark.asyncio
async def test_run_daily_self_check_with_grok_success(db_with_experiment):
    """When Grok returns valid JSON, the review includes parsed insights."""
    calibration_mgr = CalibrationManager()
    market_type_mgr = MarketTypeManager()
    signal_tracker_mgr = SignalTrackerManager()

    mock_grok = MagicMock()
    mock_grok.complete = AsyncMock(
        return_value='{"insights": "Good day overall.", "recommendations": ["Keep calibrating"], "health_status": "HEALTHY"}'
    )

    mock_settings = MagicMock()

    with patch("src.learning.self_check.REVIEW_DIR", tempfile.mkdtemp()):
        review = await run_daily_self_check(
            db_with_experiment,
            mock_grok,
            calibration_mgr,
            market_type_mgr,
            signal_tracker_mgr,
            mock_settings,
        )

    assert review.health_status == "HEALTHY"
    assert review.llm_insights == "Good day overall."
    assert review.llm_recommendations == ["Keep calibrating"]
    # Verify saved
    loaded = await db_with_experiment.get_daily_review(review.review_date)
    assert loaded is not None
    assert loaded.health_status == "HEALTHY"
    assert loaded.llm_insights == "Good day overall."


# ---------------------------------------------------------------------------
# Test: run_daily_self_check uses json_object response_format + max_tokens=4000
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_self_check_uses_json_mode_and_higher_max_tokens(db_with_experiment):
    """grok.complete must be called with max_tokens=4000 and response_format={'type':'json_object'}."""
    calibration_mgr = CalibrationManager()
    market_type_mgr = MarketTypeManager()
    signal_tracker_mgr = SignalTrackerManager()

    mock_grok = MagicMock()
    mock_grok.complete = AsyncMock(
        return_value='{"insights": "x", "recommendations": ["y"], "health_status": "HEALTHY"}'
    )

    mock_settings = MagicMock()

    with patch("src.learning.self_check.REVIEW_DIR", tempfile.mkdtemp()):
        await run_daily_self_check(
            db_with_experiment,
            mock_grok,
            calibration_mgr,
            market_type_mgr,
            signal_tracker_mgr,
            mock_settings,
        )

    assert mock_grok.complete.call_count >= 1
    call_kwargs = mock_grok.complete.call_args.kwargs
    assert call_kwargs["max_tokens"] == 4000
    assert call_kwargs["response_format"] == {"type": "json_object"}


# ---------------------------------------------------------------------------
# Test: run_daily_self_check retries on parse failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_self_check_retries_on_parse_failure(db_with_experiment):
    """First call returns unparseable output; second call returns valid JSON.
    Checks: 2 calls made, review populated from second response, warning logged."""
    calibration_mgr = CalibrationManager()
    market_type_mgr = MarketTypeManager()
    signal_tracker_mgr = SignalTrackerManager()

    mock_grok = MagicMock()
    mock_grok.complete = AsyncMock(
        side_effect=[
            "<think>truncated",
            '{"insights": "ok", "recommendations": ["x"], "health_status": "HEALTHY"}',
        ]
    )

    mock_settings = MagicMock()

    with patch("src.learning.self_check.REVIEW_DIR", tempfile.mkdtemp()), \
         patch("src.learning.self_check.asyncio.sleep", new_callable=AsyncMock) as mock_sleep, \
         patch("src.learning.self_check.log") as mock_log:
        review = await run_daily_self_check(
            db_with_experiment,
            mock_grok,
            calibration_mgr,
            market_type_mgr,
            signal_tracker_mgr,
            mock_settings,
        )

    # Two attempts made
    assert mock_grok.complete.call_count == 2

    # Review populated from the successful second attempt
    assert review.llm_insights == "ok"
    assert review.health_status == "HEALTHY"

    # Sleep called once between attempts
    mock_sleep.assert_awaited_once_with(1.0)

    # Warning logged with raw_preview containing the failed response text
    warning_calls = mock_log.warning.call_args_list
    assert len(warning_calls) >= 1
    first_warning = warning_calls[0]
    # structlog uses positional arg for event name
    assert first_warning[0][0] == "self_check_grok_parse_failed"
    assert "raw_preview" in first_warning[1]
    assert "truncated" in first_warning[1]["raw_preview"]


# ---------------------------------------------------------------------------
# Test: calibration_drift suppressed when bucket has n < 10 observations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_self_check_drift_suppressed_when_n_lt_10(db_with_experiment):
    """Buckets with fewer than 10 observations produce string 'insufficient (n=X)';
    buckets with >=10 produce a float drift value."""
    calibration_mgr = CalibrationManager()
    market_type_mgr = MarketTypeManager()
    signal_tracker_mgr = SignalTrackerManager()

    # bucket[0]: alpha=1.0, beta=1.0 → n = int(round(1+1-2)) = 0 → suppressed
    calibration_mgr.buckets[0].alpha = 1.0
    calibration_mgr.buckets[0].beta = 1.0

    # bucket[1]: alpha=7.0, beta=5.0 → n = int(round(7+5-2)) = 10 → NOT suppressed
    calibration_mgr.buckets[1].alpha = 7.0
    calibration_mgr.buckets[1].beta = 5.0

    mock_grok = MagicMock()
    mock_grok.complete = AsyncMock(
        return_value='{"insights": "fine", "recommendations": [], "health_status": "HEALTHY"}'
    )

    mock_settings = MagicMock()

    with patch("src.learning.self_check.REVIEW_DIR", tempfile.mkdtemp()):
        review = await run_daily_self_check(
            db_with_experiment,
            mock_grok,
            calibration_mgr,
            market_type_mgr,
            signal_tracker_mgr,
            mock_settings,
        )

    drift = review.calibration_drift

    # Build expected keys from bucket ranges
    b0 = calibration_mgr.buckets[0]
    b1 = calibration_mgr.buckets[1]
    key0 = f"{b0.bucket_range[0]:.2f}-{b0.bucket_range[1]:.2f}"
    key1 = f"{b1.bucket_range[0]:.2f}-{b1.bucket_range[1]:.2f}"

    assert key0 in drift
    assert isinstance(drift[key0], str)
    assert drift[key0].startswith("insufficient")

    assert key1 in drift
    assert isinstance(drift[key1], float)
