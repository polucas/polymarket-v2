"""Reconcile stored vs cost-based correct PnL for early-exit (TP/SL) trades.

Pre-`8338c32`, `calculate_early_exit_pnl()` did NOT subtract fees on winning
exits. Historical TP/SL winners therefore have stored pnl = GROSS. This script
recomputes the cost-based net pnl (matching the post-fix formula) and reports
(or, with --apply, backfills) the drift.

READ-ONLY by default. Pass --apply to mutate the DB.

Usage:
    python scripts/reconcile_early_exit_pnl.py
    python scripts/reconcile_early_exit_pnl.py --since 2026-05-19
    python scripts/reconcile_early_exit_pnl.py --top 20         # show top 20 worst diffs
    python scripts/reconcile_early_exit_pnl.py --apply          # backfill (atomic)
"""
import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import get_settings
from src.db.sqlite import Database


def correct_early_exit_pnl(action: str, pos: float, entry: float, exit_price: float, fee_rate: float) -> float:
    """Cost-based early-exit pnl with fee on wins (matches calculate_early_exit_pnl post-8338c32).

    Inlined deliberately so this script works both before and after the
    calculate_early_exit_pnl fix is deployed to production.
    """
    fee = pos * fee_rate
    if action == "BUY_YES":
        if entry <= 0:
            return 0.0
        gross = pos * (exit_price / entry - 1.0)
        return gross - fee if gross > 0 else gross
    if action == "BUY_NO":
        if entry >= 1.0:
            return 0.0
        gross = pos * ((1.0 - exit_price) / (1.0 - entry) - 1.0)
        return gross - fee if gross > 0 else gross
    return 0.0


async def main() -> None:
    p = argparse.ArgumentParser(
        description="Reconcile stored vs cost-based PnL for early-exit (TP/SL) trades."
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
    p.add_argument(
        "--apply",
        action="store_true",
        help="MUTATE the DB: update trade_records.pnl for affected trades and "
             "adjust portfolio.total_pnl by total_diff. "
             "Atomic transaction. Idempotent. Does NOT touch cash_balance/total_equity "
             "(cash was credited gross historically; positions are closed).",
    )
    args = p.parse_args()

    settings = get_settings()
    db = await Database.init(settings.DB_PATH)

    since = args.since  # None or string — SQLite will handle either fine

    async with db._conn.execute(
        """SELECT record_id, action, market_price_at_decision, exit_price,
                  position_size_usd, fee_rate, exit_type, pnl
           FROM trade_records
           WHERE exit_type IS NOT NULL
             AND voided = 0
             AND (? IS NULL OR timestamp >= ?)""",
        (since, since),
    ) as cursor:
        rows = await cursor.fetchall()

    if not rows:
        await db.close()
        print("No early-exit (TP/SL) trades found.")
        return

    # ------------------------------------------------------------------ #
    # Compute diffs                                                        #
    # ------------------------------------------------------------------ #
    results = []
    for r in rows:
        record_id: str = r["record_id"]
        action: str = r["action"]
        entry: float = float(r["market_price_at_decision"] or 0.0)
        exit_price: float = float(r["exit_price"] or 0.0)
        pos: float = float(r["position_size_usd"] or 0.0)
        fee_rate: float = float(r["fee_rate"] or 0.0)
        exit_type: str = r["exit_type"]
        stored: float = float(r["pnl"] or 0.0)

        corrected = correct_early_exit_pnl(action, pos, entry, exit_price, fee_rate)
        diff = corrected - stored

        results.append({
            "record_id": record_id,
            "action": action,
            "entry": entry,
            "exit_price": exit_price,
            "pos": pos,
            "exit_type": exit_type,
            "stored_pnl": stored,
            "correct_pnl": corrected,
            "diff": diff,
        })

    # ------------------------------------------------------------------ #
    # Aggregate metrics                                                    #
    # ------------------------------------------------------------------ #
    n_total = len(results)
    take_profits = [r for r in results if r["exit_type"] == "take_profit"]
    stop_losses  = [r for r in results if r["exit_type"] == "stop_loss"]

    stored_sum  = sum(r["stored_pnl"]  for r in results)
    correct_sum = sum(r["correct_pnl"] for r in results)
    total_diff  = correct_sum - stored_sum

    # ------------------------------------------------------------------ #
    # Apply mutation (only if --apply)                                     #
    # ------------------------------------------------------------------ #
    if args.apply:
        if abs(total_diff) < 1e-6:
            print()
            print("Already reconciled — no-op.")
            await db.close()
            return

        # Trades whose pnl needs updating (any nonzero diff)
        to_update = [r for r in results if abs(r["diff"]) >= 1e-9]

        try:
            await db._conn.execute("BEGIN IMMEDIATE")
            for r in to_update:
                await db._conn.execute(
                    "UPDATE trade_records SET pnl = ? WHERE record_id = ?",
                    (r["correct_pnl"], r["record_id"]),
                )
            # Portfolio adjustment (id=1 by convention). Do NOT touch
            # cash_balance/total_equity — cash was credited gross historically
            # and positions are already closed; only the recorded pnl drifted.
            await db._conn.execute(
                """UPDATE portfolio
                   SET total_pnl  = total_pnl + ?,
                       updated_at = datetime('now')
                   WHERE id = 1""",
                (total_diff,),
            )
            await db._conn.commit()
        except Exception:
            await db._conn.rollback()
            await db.close()
            raise

        print()
        print(f"APPLIED: {len(to_update)} trade pnls updated; "
              f"portfolio.total_pnl adjusted by ${total_diff:+.2f}")
        await db.close()
        return

    # ------------------------------------------------------------------ #
    # Table 1 — per-trade diffs (top N by abs diff)                       #
    # ------------------------------------------------------------------ #
    sorted_results = sorted(results, key=lambda x: abs(x["diff"]), reverse=True)
    top_n = sorted_results[: args.top]

    hdr = f"{'record_id':<18}{'action':<9}{'exit_type':<13}{'entry':>7}{'exit':>7}{'pos':>8}{'stored_pnl':>12}{'correct_pnl':>13}{'diff':>9}{'diff%':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in top_n:
        diff_pct = (r["diff"] / r["stored_pnl"] * 100) if r["stored_pnl"] else float("nan")
        diff_pct_str = f"{diff_pct:+.1f}%" if r["stored_pnl"] != 0 else "  n/a"
        print(
            f"{r['record_id'][:16]:<18}"
            f"{r['action']:<9}"
            f"{r['exit_type']:<13}"
            f"{r['entry']:>7.3f}"
            f"{r['exit_price']:>7.3f}"
            f"{r['pos']:>8.2f}"
            f"{r['stored_pnl']:>+12.2f}"
            f"{r['correct_pnl']:>+13.2f}"
            f"{r['diff']:>+9.2f}"
            f"{diff_pct_str:>8}"
        )

    # ------------------------------------------------------------------ #
    # Table 2 — aggregate summary                                         #
    # ------------------------------------------------------------------ #
    print()
    print(f"Early-exit trades (closed, voided=0):    N total = {n_total}")
    if args.since:
        print(f"  (filtered: since {args.since})")
    print(f"  Take profit:                            N = {len(take_profits)}")
    print(f"  Stop loss:                              N = {len(stop_losses)}")
    print()
    print(f"Stored pnl sum (early exits):            $ {stored_sum:>10.2f}")
    print(f"Correct pnl sum (early exits):           $ {correct_sum:>10.2f}")
    print(f"TOTAL DIFF (correct - stored):           $ {total_diff:>+10.2f}  (will be NEGATIVE if rows are pre-fix gross)")

    await db.close()


if __name__ == "__main__":
    asyncio.run(main())
