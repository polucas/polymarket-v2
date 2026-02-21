from __future__ import annotations

from typing import Dict, List, Tuple

import structlog

from src.config import get_settings
from src.models import TradeCandidate, Position

log = structlog.get_logger()


def _keyword_overlap(kw1: List[str], kw2: List[str]) -> float:
    """Jaccard similarity of lowered keyword sets."""
    s1 = set(w.lower() for w in kw1)
    s2 = set(w.lower() for w in kw2)
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def detect_market_clusters(candidates: List[TradeCandidate]) -> Dict[str, str]:
    """Detect correlated market clusters.
    Group by market_type, then within same category: sort by resolution_time.
    Markets within 1h resolution window + 50% keyword Jaccard overlap = same cluster.
    Returns: {market_id: cluster_id}
    """
    if not candidates:
        return {}

    # Group by market type
    by_type: Dict[str, List[TradeCandidate]] = {}
    for c in candidates:
        mtype = c.market.market_type
        by_type.setdefault(mtype, []).append(c)

    clusters: Dict[str, str] = {}
    cluster_counter = 0

    for mtype, group in by_type.items():
        # Sort by resolution time
        group.sort(key=lambda c: c.resolution_hours)

        assigned: Dict[int, str] = {}  # index -> cluster_id
        for i, c1 in enumerate(group):
            if i in assigned:
                continue
            cluster_counter += 1
            cid = f"cluster_{cluster_counter}"
            clusters[c1.market.market_id] = cid
            assigned[i] = cid

            for j in range(i + 1, len(group)):
                if j in assigned:
                    continue
                c2 = group[j]
                # Within 1h resolution window
                if abs(c1.resolution_hours - c2.resolution_hours) <= 1.0:
                    # 50% keyword overlap
                    if _keyword_overlap(c1.market.keywords, c2.market.keywords) >= 0.50:
                        clusters[c2.market.market_id] = cid
                        assigned[j] = cid

    return clusters


def check_cluster_exposure(
    candidate: TradeCandidate,
    cluster_id: str,
    open_positions: List[Position],
    pending: List[TradeCandidate],
    clusters: Dict[str, str],
    bankroll: float,
) -> bool:
    """Check if adding this trade would exceed cluster exposure limit.
    Returns True if within limit, False if would exceed.
    """
    settings = get_settings()
    max_cluster_pct = settings.MAX_CLUSTER_EXPOSURE_PCT

    # Sum existing exposure for same cluster
    existing = 0.0
    for pos in open_positions:
        if clusters.get(pos.market_id) == cluster_id:
            existing += pos.size_usd

    # Sum pending exposure
    for p in pending:
        if clusters.get(p.market.market_id) == cluster_id:
            existing += p.position_size

    total = existing + candidate.position_size
    return total <= max_cluster_pct * bankroll


def select_best_trades(
    candidates: List[TradeCandidate],
    remaining_cap: int,
    open_positions: List[Position],
    bankroll: float,
) -> Tuple[List[TradeCandidate], List[TradeCandidate]]:
    """Score, rank, and select best trades within caps and cluster limits.

    score = edge * adjusted_confidence * (1.0 / max(resolution_hours, 0.5))

    Returns: (to_execute, to_skip)
    """
    if not candidates:
        return [], []

    # Score each candidate
    for c in candidates:
        c.score = c.calculated_edge * c.adjusted_confidence * (1.0 / max(c.resolution_hours, 0.5))

    # Sort by score descending
    ranked = sorted(candidates, key=lambda c: c.score, reverse=True)

    # Detect clusters
    clusters = detect_market_clusters(candidates)

    to_execute: List[TradeCandidate] = []
    to_skip: List[TradeCandidate] = []

    for c in ranked:
        c.market_cluster_id = clusters.get(c.market.market_id)

        if len(to_execute) >= remaining_cap:
            c.skip_reason = "daily_cap_reached"
            to_skip.append(c)
            continue

        if c.market_cluster_id and not check_cluster_exposure(
            c, c.market_cluster_id, open_positions, to_execute, clusters, bankroll
        ):
            c.skip_reason = "cluster_exposure_limit"
            to_skip.append(c)
            continue

        to_execute.append(c)

    return to_execute, to_skip
