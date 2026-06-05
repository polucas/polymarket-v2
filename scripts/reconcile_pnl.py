"""Reconcile stored vs cost-based correct PnL for natural-resolution trades.

For each closed trade resolved via natural resolution (exit_type IS NULL,
actual_outcome IS NOT NULL), compute the correct cost-based PnL and compare
to the stored value. Reports per-trade diffs and total drift attributable
to the formula bug.

READ-ONLY. Does not modify the DB.

Usage:
    python scripts/reconcile_pnl.py
    python scripts/reconcile_pnl.py --since 2026-05-19
    python scripts/reconcile_pnl.py --top 20         # show top 20 worst diffs
"""
import argparse
import asyncio
import os
import sys
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import get_settings
from src.db.sqlite import Database


def correct_pnl(action: str, position_size_usd: float, entry: float, outcome: bool, fee_rate: float) -> float:
    """Cost-based PnL — matches the fixed calculate_pnl.

    Inlined deliberately so this script works both before and after the
    calculate_pnl fix is deployed to production.
    """
    pos = position_size_usd
    fee = pos * fee_rate
    if action == "BUY_YES":
        if outcome:
            if entry <= 0:
                return 0.0
            return pos * (1.0 - entry) / entry - fee
        return -pos
    if action == "BUY_NO":
        if not outcome:
            if entry >= 1.0:
                return 0.0
            return pos * entry / (1.0 - entry) - fee
        return -pos
    return 0.0


async def main() -> None:
    p = argparse.ArgumentParser(
        description="Reconcile stored vs cost-based PnL for natural-resolution trades."
    )
    p.add_argument(
        "--since",
        default=None,
        help="Only consider trades at or after this ISO date/datetime (e.g. 2026-05-19).",
    )
    p.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of worst per-trade diffs to display (default: 20).",
    )
    args = p.parse_args()

    settings = get_settings()
    db = await Database.init(settings.DB_PATH)

    since = args.since  # None or string — SQLite will handle either fine

    async with db._conn.execute(
        """SELECT record_id, action, market_price_at_decision, position_size_usd,
                  fee_rate, actual_outcome, pnl
           FROM trade_records
           WHERE action != 'SKIP'
             AND exit_type IS NULL
             AND actual_outcome IS NOT NULL
             AND voided = 0
             AND (? IS NULL OR timestamp >= ?)""",
        (since, since),
    ) as cursor:
        rows = await cursor.fetchall()

    await db.close()

    if not rows:
        print("No natural-resolution closed trades found.")
        return

    # ------------------------------------------------------------------ #
    # Compute diffs                                                        #
    # ------------------------------------------------------------------ #
    results = []
    for r in rows:
        record_id: str = r["record_id"]
        action: str = r["action"]
        entry: float = float(r["market_price_at_decision"] or 0.0)
        pos: float = float(r["position_size_usd"] or 0.0)
        fee_rate: float = float(r["fee_rate"] or 0.0)
        # actual_outcome stored as INTEGER (1/0) in SQLite
        outcome: bool = bool(r["actual_outcome"])
        stored: float = float(r["pnl"] or 0.0)

        corrected = correct_pnl(action, pos, entry, outcome, fee_rate)
        diff = corrected - stored

        results.append({
            "record_id": record_id,
            "action": action,
            "entry": entry,
            "pos": pos,
            "outcome": outcome,
            "stored_pnl": stored,
            "correct_pnl": corrected,
            "diff": diff,
        })

    # ------------------------------------------------------------------ #
    # Aggregate metrics                                                    #
    # ------------------------------------------------------------------ #
    n_total = len(results)
    winners = [r for r in results if r["stored_pnl"] > 0]
    losers  = [r for r in results if r["stored_pnl"] <= 0]

    stored_winner_sum  = sum(r["stored_pnl"]  for r in winners)
    correct_winner_sum = sum(r["correct_pnl"] for r in winners)
    total_drift        = correct_winner_sum - stored_winner_sum

    drift_pct = (total_drift / stored_winner_sum * 100) if stored_winner_sum else 0.0

    # ------------------------------------------------------------------ #
    # Table 1 — per-trade diffs (top N by abs diff)                       #
    # ------------------------------------------------------------------ #
    # Only winners can have a positive drift; include losers in case
    # the formula also affects them (they won't, but show them if large).
    sorted_results = sorted(results, key=lambda x: abs(x["diff"]), reverse=True)
    top_n = sorted_results[: args.top]

    hdr = f"{'record_id':<18}{'action':<9}{'entry':>7}{'pos':>8}{'outcome':<9}{'stored_pnl':>12}{'correct_pnl':>13}{'diff':>9}{'diff%':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in top_n:
        outcome_label = "YES" if r["outcome"] else "NO"
        diff_pct = (r["diff"] / r["stored_pnl"] * 100) if r["stored_pnl"] else float("nan")
        diff_pct_str = f"{diff_pct:+.1f}%" if r["stored_pnl"] != 0 else "  n/a"
        print(
            f"{r['record_id'][:16]:<18}"
            f"{r['action']:<9}"
            f"{r['entry']:>7.3f}"
            f"{r['pos']:>8.2f}"
            f"{outcome_label:<9}"
            f"{r['stored_pnl']:>+12.2f}"
            f"{r['correct_pnl']:>+13.2f}"
            f"{r['diff']:>+9.2f}"
            f"{diff_pct_str:>8}"
        )

    # ------------------------------------------------------------------ #
    # Table 2 — aggregate summary                                         #
    # ------------------------------------------------------------------ #
    print()
    print(f"Natural-resolution trades (closed, voided=0):   N total = {n_total}")
    if args.since:
        print(f"  (filtered: since {args.since})")
    print(f"  Winners (stored pnl > 0):                     N = {len(winners)}")
    print(f"  Losers  (stored pnl <= 0):                    N = {len(losers)}")
    print()
    print(f"Stored pnl sum  (winners):                      ${stored_winner_sum:>10.2f}")
    print(f"Correct pnl sum (winners):                      ${correct_winner_sum:>10.2f}")
    print(f"TOTAL DRIFT ATTRIBUTABLE TO FORMULA BUG:        ${total_drift:>+10.2f}  (correct - stored)")
    print()
    print(f"% drift vs total stored winning pnl:            {drift_pct:+.1f}%")


if __name__ == "__main__":
    asyncio.run(main())
