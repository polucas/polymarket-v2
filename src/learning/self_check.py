"""Daily self-check loop — Karpathy 'Autoresearch' inspired.

Gathers performance metrics, calls Grok for analysis, persists findings.
Does NOT auto-implement changes — only documents recommendations.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone, timedelta
from typing import Optional

import structlog

from src.db.sqlite import Database
from src.engine.grok_client import GrokClient, parse_json_safe
from src.learning.calibration import CalibrationManager
from src.learning.market_type import MarketTypeManager
from src.learning.signal_tracker import SignalTrackerManager
from src.models import DailyReview

log = structlog.get_logger()

REVIEW_DIR = "data/daily_reviews"


async def _gather_metrics(db: Database, review_date: str) -> dict:
    """Gather all performance metrics for the given date."""
    stats = await db.get_period_trade_stats(review_date, review_date)

    # Skip reason distribution
    cursor = await db._conn.execute(
        "SELECT skip_reason, COUNT(*) FROM trade_records "
        "WHERE date(timestamp) = ? AND action = 'SKIP' AND skip_reason IS NOT NULL "
        "GROUP BY skip_reason ORDER BY COUNT(*) DESC",
        (review_date,),
    )
    skip_rows = await cursor.fetchall()
    skip_distribution = {r[0]: r[1] for r in skip_rows}

    # Brier by market type
    cursor = await db._conn.execute(
        "SELECT market_type, AVG(brier_score_raw), AVG(brier_score_adjusted), COUNT(*), SUM(pnl) "
        "FROM trade_records "
        "WHERE date(timestamp) = ? AND action != 'SKIP' AND actual_outcome IS NOT NULL AND voided = FALSE "
        "GROUP BY market_type",
        (review_date,),
    )
    mt_rows = await cursor.fetchall()
    brier_by_type = {}
    top_types = []
    worst_types = []
    for r in mt_rows:
        mtype = r[0] or "unknown"
        brier_by_type[mtype] = {
            "avg_brier_raw": round(r[1], 4) if r[1] is not None else None,
            "avg_brier_adjusted": round(r[2], 4) if r[2] is not None else None,
            "count": r[3],
            "pnl": round(r[4], 2) if r[4] is not None else 0,
        }
        if r[4] is not None and r[4] > 0:
            top_types.append(mtype)
        elif r[4] is not None and r[4] < 0:
            worst_types.append(mtype)

    # Win rate
    win_rate = None
    total_resolved = stats["total_resolved"]
    if total_resolved and total_resolved > 0:
        win_rate = stats["wins"] / total_resolved

    # ROI — total PnL / total amount invested
    total_invested = 0.0
    cursor = await db._conn.execute(
        "SELECT SUM(position_size_usd) FROM trade_records "
        "WHERE date(timestamp) = ? AND action != 'SKIP' AND voided = FALSE",
        (review_date,),
    )
    inv_row = await cursor.fetchone()
    if inv_row and inv_row[0]:
        total_invested = inv_row[0]
    roi_pct = (stats["total_pnl"] / total_invested * 100) if total_invested > 0 else None

    return {
        **stats,
        "win_rate": win_rate,
        "roi_pct": roi_pct,
        "skip_reason_distribution": skip_distribution,
        "brier_by_market_type": brier_by_type,
        "top_performing_types": top_types,
        "worst_performing_types": worst_types,
    }


def _build_llm_prompt(metrics: dict, review_date: str) -> str:
    """Build the Grok prompt for daily analysis."""
    win_rate_str = f"{metrics['win_rate']:.1%}" if metrics['win_rate'] is not None else "N/A"
    roi_str = f"{metrics['roi_pct']:.1f}%" if metrics['roi_pct'] is not None else "N/A"
    brier_raw_str = f"{metrics['avg_brier_raw']:.3f}" if metrics['avg_brier_raw'] is not None else "N/A"
    brier_adj_str = f"{metrics['avg_brier_adjusted']:.3f}" if metrics['avg_brier_adjusted'] is not None else "N/A"

    # Format market type table
    mt_lines = []
    for mtype, data in metrics.get("brier_by_market_type", {}).items():
        mt_lines.append(
            f"  {mtype}: Brier={data.get('avg_brier_raw', 'N/A')}, "
            f"PnL=${data.get('pnl', 0):.2f}, Trades={data.get('count', 0)}"
        )
    mt_table = "\n".join(mt_lines) if mt_lines else "  No resolved trades by market type today."

    # Format skip reasons
    skip_lines = []
    for reason, count in metrics.get("skip_reason_distribution", {}).items():
        skip_lines.append(f"  {reason}: {count}")
    skip_table = "\n".join(skip_lines) if skip_lines else "  No skips recorded."

    return f"""DAILY PERFORMANCE REVIEW — {review_date}

Today's Metrics:
  Executed trades: {metrics['trade_count']}
  Skipped evaluations: {metrics['skip_count']}
  Resolved trades: {metrics['resolved_count']}
  Win rate: {win_rate_str}
  Total PnL: ${metrics['total_pnl']:+.2f}
  ROI: {roi_str}
  Avg Brier (raw): {brier_raw_str}
  Avg Brier (adjusted): {brier_adj_str}

Market Type Breakdown:
{mt_table}

Skip Reason Distribution:
{skip_table}

Top performing types: {metrics.get('top_performing_types', [])}
Worst performing types: {metrics.get('worst_performing_types', [])}

INSTRUCTIONS:
You are analyzing the daily performance of an automated prediction market trading bot.
1. Identify the 3 most important patterns or observations from today's data
2. Flag any concerning trends (poor calibration, consistently losing market types, too many skips of one type)
3. Suggest 2-3 specific, actionable improvements. Do NOT suggest implementation — only what SHOULD change.
4. Rate overall system health as one of: HEALTHY, CAUTION, CONCERN

Respond ONLY with valid JSON in this exact format:
{{"insights": "Your 3 key observations here as a paragraph", "recommendations": ["Recommendation 1", "Recommendation 2"], "health_status": "HEALTHY"}}"""


def _write_markdown(review: DailyReview) -> None:
    """Write daily review as a human-readable markdown file."""
    os.makedirs(REVIEW_DIR, exist_ok=True)
    path = os.path.join(REVIEW_DIR, f"{review.review_date}.md")

    lines = [
        f"# Daily Review — {review.review_date}",
        f"**Health Status: {review.health_status}**\n",
        "## Metrics",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Executed trades | {review.trade_count} |",
        f"| Skipped evaluations | {review.skip_count} |",
        f"| Resolved trades | {review.resolved_count} |",
        f"| Win rate | {f'{review.win_rate:.1%}' if review.win_rate is not None else 'N/A'} |",
        f"| Total PnL | ${review.total_pnl:+.2f} |",
        f"| ROI | {f'{review.roi_pct:.1f}%' if review.roi_pct is not None else 'N/A'} |",
        f"| Avg Brier (raw) | {f'{review.avg_brier_raw:.3f}' if review.avg_brier_raw is not None else 'N/A'} |",
        f"| Avg Brier (adjusted) | {f'{review.avg_brier_adjusted:.3f}' if review.avg_brier_adjusted is not None else 'N/A'} |",
        "",
    ]

    if review.brier_by_market_type:
        lines.append("## Market Type Breakdown")
        lines.append("| Type | Brier (raw) | PnL | Trades |")
        lines.append("|------|-------------|-----|--------|")
        for mtype, data in review.brier_by_market_type.items():
            brier = data.get("avg_brier_raw", "N/A")
            pnl = f"${data.get('pnl', 0):.2f}"
            count = data.get("count", 0)
            lines.append(f"| {mtype} | {brier} | {pnl} | {count} |")
        lines.append("")

    if review.skip_reason_distribution:
        lines.append("## Skip Reasons")
        for reason, count in review.skip_reason_distribution.items():
            lines.append(f"- **{reason}**: {count}")
        lines.append("")

    if review.llm_insights:
        lines.append("## Insights")
        lines.append(review.llm_insights)
        lines.append("")

    if review.llm_recommendations:
        lines.append("## Recommendations")
        for rec in review.llm_recommendations:
            lines.append(f"- {rec}")
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    log.info("daily_review_markdown_written", path=path)


def format_self_check_alert(review: DailyReview) -> str:
    """Format daily self-check for Telegram."""
    win_str = f"{review.win_rate:.0%}" if review.win_rate is not None else "N/A"
    roi_str = f"{review.roi_pct:.1f}%" if review.roi_pct is not None else "N/A"
    brier_raw = f"{review.avg_brier_raw:.3f}" if review.avg_brier_raw is not None else "N/A"
    brier_adj = f"{review.avg_brier_adjusted:.3f}" if review.avg_brier_adjusted is not None else "N/A"

    text = (
        f"<b>DAILY SELF-CHECK — {review.review_date}</b>\n"
        f"Health: <b>{review.health_status}</b>\n\n"
        f"Trades: {review.trade_count} executed, {review.skip_count} skipped, {review.resolved_count} resolved\n"
        f"Win Rate: {win_str} | ROI: {roi_str}\n"
        f"PnL: ${review.total_pnl:+.2f}\n"
        f"Brier: {brier_raw} (raw) / {brier_adj} (adj)\n"
    )

    if review.llm_insights:
        text += f"\n<b>Insights:</b>\n{review.llm_insights}\n"

    if review.llm_recommendations:
        text += "\n<b>Recommendations:</b>\n"
        for rec in review.llm_recommendations:
            text += f"• {rec}\n"

    return text


async def run_daily_self_check(
    db: Database,
    grok: GrokClient,
    calibration_mgr: CalibrationManager,
    market_type_mgr: MarketTypeManager,
    signal_tracker_mgr: SignalTrackerManager,
    settings,
) -> DailyReview:
    """Run the daily self-check analysis."""
    from src.learning.experiments import get_current_experiment

    review_date = (datetime.now(timezone.utc) - timedelta(hours=1)).strftime("%Y-%m-%d")
    log.info("daily_self_check_start", review_date=review_date)

    # Step 1: Gather metrics
    metrics = await _gather_metrics(db, review_date)

    # Step 2: Get calibration drift from manager
    calibration_drift = {}
    for bucket in calibration_mgr.buckets:
        key = f"{bucket.bucket_range[0]:.2f}-{bucket.bucket_range[1]:.2f}"
        midpoint = (bucket.bucket_range[0] + bucket.bucket_range[1]) / 2
        expected = bucket.alpha / (bucket.alpha + bucket.beta)
        calibration_drift[key] = round(expected - midpoint, 4)
    metrics["calibration_drift"] = calibration_drift

    # Step 3: Get signal effectiveness from manager
    signal_effectiveness = {}
    for key, tracker in signal_tracker_mgr.trackers.items():
        total = tracker.present_in_winning_trades + tracker.present_in_losing_trades
        if total >= 5:
            signal_effectiveness[str(key)] = round(tracker.weight, 3)
    metrics["signal_effectiveness"] = signal_effectiveness

    # Step 4: Get experiment run
    experiment = await get_current_experiment(db)
    experiment_run = experiment.run_id if experiment else ""

    # Step 5: Call Grok for analysis
    llm_insights = ""
    llm_recommendations: list = []
    health_status = "UNKNOWN"

    prompt = _build_llm_prompt(metrics, review_date)
    try:
        raw_response = await grok.complete(prompt, max_tokens=1000)
        parsed = parse_json_safe(raw_response)
        if parsed:
            llm_insights = parsed.get("insights", "")
            llm_recommendations = parsed.get("recommendations", [])
            health_status = parsed.get("health_status", "UNKNOWN")
            if not isinstance(llm_recommendations, list):
                llm_recommendations = [str(llm_recommendations)]
        else:
            log.warning("self_check_grok_parse_failed")
            health_status = "UNKNOWN"
    except Exception as e:
        log.error("self_check_grok_error", error=str(e))
        health_status = "UNKNOWN"

    # Step 6: Build review
    review = DailyReview(
        review_date=review_date,
        timestamp=datetime.now(timezone.utc),
        trade_count=metrics["trade_count"],
        skip_count=metrics["skip_count"],
        resolved_count=metrics["resolved_count"],
        win_rate=metrics["win_rate"],
        roi_pct=metrics["roi_pct"],
        total_pnl=metrics["total_pnl"],
        avg_brier_raw=metrics["avg_brier_raw"],
        avg_brier_adjusted=metrics["avg_brier_adjusted"],
        brier_by_market_type=metrics.get("brier_by_market_type", {}),
        calibration_drift=calibration_drift,
        signal_effectiveness=signal_effectiveness,
        skip_reason_distribution=metrics.get("skip_reason_distribution", {}),
        top_performing_types=metrics.get("top_performing_types", []),
        worst_performing_types=metrics.get("worst_performing_types", []),
        llm_insights=llm_insights,
        llm_recommendations=llm_recommendations,
        health_status=health_status,
        experiment_run=experiment_run,
    )

    # Step 7: Persist
    await db.save_daily_review(review)
    _write_markdown(review)

    log.info("daily_self_check_complete", review_date=review_date, health=health_status)
    return review
