"""TP threshold sensitivity analysis.

Mirrors sl_analysis.py — SL is held at None (current experimental state).
For each resolved trade since --since date, replay snapshot price evolution
and compute what realized PnL would have been at different TP thresholds.

Usage:
    python scripts/tp_analysis.py --since 2026-05-26
    python scripts/tp_analysis.py --since 2026-05-26 --thresholds none,0.10,0.15,0.20,0.25,0.30,0.40,0.50
"""
import argparse
import asyncio
import os
import sys
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import get_settings
from src.db.sqlite import Database
from src.engine.resolution import calculate_early_exit_pnl
from src.models import TradeRecord


def parse_thresholds(s: str) -> List[Optional[float]]:
    out = []
    for tok in s.split(","):
        tok = tok.strip().lower()
        if tok in ("none", "off", ""):
            out.append(None)
        else:
            out.append(float(tok))
    return out


def simulate(trade: TradeRecord, snapshots, tp_threshold: Optional[float]) -> dict:
    """Walk snapshots; first ROI to hit tp_threshold triggers a simulated exit.
    SL is held at None (no SL firing) — matches current experimental state.
    If no snapshot hits tp_threshold, return the actual final pnl (natural resolution case)."""
    for snap in snapshots:
        roi = snap["roi"]
        if tp_threshold is not None and roi >= tp_threshold:
            pnl = calculate_early_exit_pnl(trade, snap["best_bid"])
            return {"exit_type": "take_profit", "exit_price": snap["best_bid"], "pnl": pnl, "exit_ts": snap["timestamp"]}
    # Held to resolution — actual pnl
    return {"exit_type": "natural", "exit_price": trade.exit_price, "pnl": trade.pnl or 0.0, "exit_ts": None}


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--since", required=True, help="ISO date or datetime")
    p.add_argument("--thresholds", default="none,0.10,0.15,0.20,0.25,0.30,0.40,0.50",
                   help="comma-separated TP thresholds; use 'none' for no-TP (natural resolution only)")
    args = p.parse_args()
    thresholds = parse_thresholds(args.thresholds)

    settings = get_settings()
    db = await Database.init(settings.DB_PATH)

    # Pull all resolved trades since --since with their snapshots
    cursor = await db._conn.execute(
        """SELECT * FROM trade_records
           WHERE action != 'SKIP' AND voided = FALSE
             AND timestamp >= ?
             AND (actual_outcome IS NOT NULL OR exit_type IS NOT NULL)
           ORDER BY timestamp""",
        (args.since,),
    )
    rows = await cursor.fetchall()
    trades = [db._row_to_trade(r) for r in rows]
    print(f"Trades to analyze: {len(trades)}")

    # Build snapshot dict
    cursor = await db._conn.execute(
        "SELECT trade_record_id, timestamp, best_bid, roi, source FROM trade_price_snapshots "
        "ORDER BY trade_record_id, timestamp"
    )
    snaprows = await cursor.fetchall()
    snaps_by_trade = {}
    for r in snaprows:
        tid = r["trade_record_id"]
        snaps_by_trade.setdefault(tid, []).append({
            "timestamp": r["timestamp"],
            "best_bid": r["best_bid"],
            "roi": r["roi"],
            "source": r["source"],
        })

    print(f"Snapshots loaded: {sum(len(v) for v in snaps_by_trade.values())} across {len(snaps_by_trade)} trades\n")

    # Run simulations
    results_by_threshold = {}
    for thresh in thresholds:
        results = []
        for trade in trades:
            snaps = snaps_by_trade.get(trade.record_id, [])
            r = simulate(trade, snaps, tp_threshold=thresh)
            r["trade_id"] = trade.record_id[:8]
            r["entry"] = trade.market_price_at_decision
            r["size"] = trade.position_size_usd
            results.append(r)
        results_by_threshold[thresh] = results

    # Report
    print(f"{'TP':<10}{'N':>5}{'wins':>6}{'win%':>7}{'sum_pnl':>11}{'avg_pnl':>10}{'min':>10}{'max':>10}")
    print("-" * 70)
    for thresh, results in results_by_threshold.items():
        label = "none" if thresh is None else f"+{thresh:.2%}"
        n = len(results)
        wins = sum(1 for r in results if r["pnl"] > 0)
        sum_pnl = sum(r["pnl"] for r in results)
        avg_pnl = sum_pnl / n if n else 0.0
        min_pnl = min((r["pnl"] for r in results), default=0.0)
        max_pnl = max((r["pnl"] for r in results), default=0.0)
        win_pct = wins / n * 100 if n else 0.0
        print(f"{label:<10}{n:>5}{wins:>6}{win_pct:>6.1f}%{sum_pnl:>+11.2f}{avg_pnl:>+10.2f}{min_pnl:>+10.2f}{max_pnl:>+10.2f}")

    await db.close()


if __name__ == "__main__":
    asyncio.run(main())
