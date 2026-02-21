from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional, Tuple

import structlog

from src.db.sqlite import Database
from src.learning.calibration import CalibrationManager
from src.learning.market_type import MarketTypeManager
from src.learning.signal_tracker import SignalTrackerManager
from src.models import TradeRecord

log = structlog.get_logger()


def adjust_prediction(
    grok_probability: float,
    grok_confidence: float,
    market_type: str,
    signal_tags: List[dict],
    calibration_mgr: CalibrationManager,
    market_type_mgr: MarketTypeManager,
    signal_tracker_mgr: SignalTrackerManager,
) -> Tuple[float, float, float]:
    """5-step adjustment pipeline.
    Returns (adjusted_probability, adjusted_confidence, extra_edge_penalty).
    """
    adjusted_confidence = grok_confidence
    adjusted_probability = grok_probability

    # Step 1: Bayesian calibration (confidence)
    cal_correction = calibration_mgr.get_correction(grok_confidence)
    adjusted_confidence = max(0.50, min(0.99, adjusted_confidence + cal_correction))

    # Step 2: Signal-type weighting (confidence)
    if signal_tags:
        weights = []
        for tag in signal_tags:
            st = tag.get("source_tier", "S6")
            it = tag.get("info_type", "I5")
            w = signal_tracker_mgr.get_signal_weight(st, it, market_type)
            weights.append(w)
        if weights:
            avg_weight = sum(weights) / len(weights)
            adjusted_confidence += (avg_weight - 1.0) * 0.1
            adjusted_confidence = max(0.50, min(0.99, adjusted_confidence))

    # Step 3: Probability shrinkage
    bucket = calibration_mgr.find_bucket(grok_confidence)
    if bucket and bucket.sample_count >= 10:
        bucket_midpoint = (bucket.bucket_range[0] + bucket.bucket_range[1]) / 2
        if bucket_midpoint > 0:
            shrinkage_factor = bucket.expected_accuracy / bucket_midpoint
            adjusted_probability = 0.5 + (grok_probability - 0.5) * shrinkage_factor
            adjusted_probability = max(0.01, min(0.99, adjusted_probability))

    # Step 4: Market-type edge penalty
    extra_edge = market_type_mgr.get_edge_adjustment(market_type)

    # Step 5: Temporal confidence decay
    # Check signal ages - applied per signal
    now = datetime.now(timezone.utc)
    has_recent_i1 = False
    max_age_hours = 0.0
    for tag in signal_tags:
        ts_str = tag.get("timestamp")
        if ts_str:
            try:
                ts = datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
                age_hours = (now - ts).total_seconds() / 3600
                max_age_hours = max(max_age_hours, age_hours)
                if tag.get("info_type") == "I1" and age_hours < 0.5:
                    has_recent_i1 = True
            except (ValueError, TypeError):
                pass

    if has_recent_i1:
        adjusted_confidence *= 1.05
        adjusted_confidence = min(0.99, adjusted_confidence)
    elif max_age_hours > 1.0:
        decay = max(0.85, 1.0 - 0.05 * (max_age_hours - 1.0))
        adjusted_confidence *= decay
        adjusted_confidence = max(0.50, adjusted_confidence)

    return adjusted_probability, adjusted_confidence, extra_edge


async def on_trade_resolved(
    record: TradeRecord,
    calibration_mgr: CalibrationManager,
    market_type_mgr: MarketTypeManager,
    signal_tracker_mgr: SignalTrackerManager,
    db: Database,
) -> None:
    """Handle a resolved trade: update all learning layers and persist."""
    if record.voided:
        return

    if record.actual_outcome is None:
        return

    # Calculate Brier scores if not already set
    actual_val = 1.0 if record.actual_outcome else 0.0
    if record.brier_score_raw is None:
        record.brier_score_raw = (record.grok_raw_probability - actual_val) ** 2
    if record.brier_score_adjusted is None:
        record.brier_score_adjusted = (record.final_adjusted_probability - actual_val) ** 2

    # Layer 1: Calibration (uses RAW probability/confidence)
    calibration_mgr.update_calibration(record)

    # Layer 2: Market-type performance (uses ADJUSTED Brier score)
    from src.engine.resolution import calculate_hypothetical_pnl
    counterfactual = calculate_hypothetical_pnl(record) if record.action == "SKIP" else 0.0
    market_type_mgr.update_market_type(record, counterfactual_pnl=counterfactual)

    # Layer 3: Signal tracker (uses ADJUSTED correctness)
    signal_tracker_mgr.update_signal_trackers(record)

    # Persist all
    await calibration_mgr.save(db)
    await market_type_mgr.save(db)
    await signal_tracker_mgr.save(db)
    await db.update_trade(record)

    log.info("learning_updated",
             market_id=record.market_id,
             brier_raw=record.brier_score_raw,
             brier_adjusted=record.brier_score_adjusted)
