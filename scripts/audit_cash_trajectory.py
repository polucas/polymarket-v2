"""Replay cash trajectory from trade_records and compare to portfolio.cash_balance.

Starting from INITIAL_BANKROLL, replays every entry (cash -= position_size_usd)
and every exit (cash += position_size_usd + pnl) in chronological order, then
compares the replayed final cash to the actual `portfolio.cash_balance` row.

If the replay matches actual cash, the drift is NOT in the entry/exit cash flow
recorded in trade_records — it must be external (init bankroll mismatch, manual
DB edits, voided-trade cash leakage, or pre-existing residue from before the
audited window). If it diverges, the delta pinpoints how much of the drift is
explained by the recorded trade flow itself.

READ-ONLY. Does not modify the DB.

Usage:
    python scripts/audit_cash_trajectory.py
    python scripts/audit_cash_trajectory.py --since 2026-05-19
    python scripts/audit_cash_trajectory.py --anomalies
"""
import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import get_settings
from src.db.sqlite import Database


async def main() -> None:
    p = argparse.ArgumentParser(
        description="Replay cash trajectory from trade_records and compare to actual portfolio.cash_balance"
    )
    p.add_argument("--since", default=None, help="Only include trades at/after this ISO date/datetime.")
    p.add_argument("--anomalies", action="store_true", help="Print anomalous trade rows.")
    args = p.parse_args()

    settings = get_settings()
    db = await Database.init(settings.DB_PATH)

    # Get portfolio current state
    async with db._conn.execute("SELECT cash_balance, total_pnl, total_equity FROM portfolio WHERE id=1") as cur:
        portfolio_row = await cur.fetchone()

    # Get all non-skip, non-voided trades
    sql = """SELECT record_id, action, market_id, timestamp, resolved_at,
                    position_size_usd, pnl, exit_type, actual_outcome, voided, fee_rate
             FROM trade_records
             WHERE action != 'SKIP' AND voided = 0"""
    params = ()
    if args.since:
        sql += " AND timestamp >= ?"
        params = (args.since,)
    sql += " ORDER BY timestamp ASC"

    async with db._conn.execute(sql, params) as cur:
        rows = await cur.fetchall()

    # Build event timeline (entry + exit events)
    events = []
    for r in rows:
        events.append(("ENTRY", r["timestamp"], dict(r)))
        if r["exit_type"] or r["actual_outcome"] is not None:
            exit_ts = r["resolved_at"] or r["timestamp"]
            events.append(("EXIT", exit_ts, dict(r)))
    events.sort(key=lambda e: e[1] or "")

    INITIAL = settings.INITIAL_BANKROLL  # 10000
    cash = INITIAL
    n_entries = 0
    n_exits = 0
    sum_entry_pos = 0.0
    sum_exit_credit = 0.0
    open_set = set()

    for kind, ts, r in events:
        pos = float(r["position_size_usd"] or 0)
        pnl = float(r["pnl"] or 0)
        if kind == "ENTRY":
            cash -= pos
            sum_entry_pos += pos
            n_entries += 1
            open_set.add(r["record_id"])
        else:
            credit = pos + pnl
            cash += credit
            sum_exit_credit += credit
            n_exits += 1
            open_set.discard(r["record_id"])

    # Compute locked from open_set
    locked_replay = 0.0
    for r_dict in (dict(x) for x in rows):
        if r_dict["record_id"] in open_set:
            locked_replay += float(r_dict["position_size_usd"] or 0)

    actual_cash = float(portfolio_row["cash_balance"])
    delta = cash - actual_cash

    print("=== CASH TRAJECTORY REPLAY ===")
    print(f"Initial bankroll:           ${INITIAL:>12.2f}")
    print(f"Total entries:              {n_entries} (sum positions deducted ${sum_entry_pos:.2f})")
    print(f"Total exits:                {n_exits} (sum credited ${sum_exit_credit:.2f})")
    print(f"Open positions (replay):    {len(open_set)} (locked ${locked_replay:.2f})")
    print()
    print(f"Replayed final cash:        ${cash:>12.2f}")
    print(f"Actual cash:                ${actual_cash:>12.2f}")
    print(f"DELTA (replay - actual):    ${delta:>+12.2f}")
    print()
    if abs(delta) < 0.01:
        print("VERDICT: Cash trajectory is INTERNALLY CONSISTENT with trade records.")
        print("         Drift source is OUTSIDE the trade flow (init bankroll, external mutation, or pre-existing residue).")
    else:
        print(f"VERDICT: Cash math diverges by ${delta:+.2f}.")
        print("         Investigate which event(s) caused the mismatch.")

    # Anomalies
    if args.anomalies:
        print()
        print("=== ANOMALIES ===")
        for r_orig in rows:
            r = dict(r_orig)
            issues = []
            if r["position_size_usd"] is None or r["position_size_usd"] <= 0:
                issues.append(f"pos={r['position_size_usd']}")
            if (r["exit_type"] is not None or r["actual_outcome"] is not None) and r["pnl"] is None:
                issues.append("closed but pnl=NULL")
            if r["resolved_at"] and r["timestamp"] and r["resolved_at"] < r["timestamp"]:
                issues.append("resolved before timestamp")
            if issues:
                print(f"  {r['record_id'][:18]} {r['action']:<8} {' | '.join(issues)}")

    await db.close()


if __name__ == "__main__":
    asyncio.run(main())
