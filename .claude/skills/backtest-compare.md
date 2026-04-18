---
name: backtest-compare
description: Run the backtest twice (baseline vs one env-var override) and report WR / ROI / Brier delta with a Welch's t-test.
trigger: backtest compare, compare backtest, ab backtest, /backtest-compare
---

## Purpose

Underpins the §2 biweekly shadow evaluator. Given a date range and a single `KEY=VALUE` override, run the historical backtest twice and report the statistical significance of the delta.

## Inputs

- `$1` — `start_date` (YYYY-MM-DD), required.
- `$2` — `end_date` (YYYY-MM-DD), required.
- `$3` — `override`, single `KEY=VALUE` pair (e.g., `TIER1_MIN_EDGE=0.04`). Required.
- `$4` — `max_markets`, int, default `5000`.

## Steps

1. Verify `data/backtest_data.db` already covers the date range:
   ```bash
   sqlite3 data/backtest_data.db "SELECT MIN(created_at), MAX(resolution_datetime) FROM historical_markets;"
   ```
   If the range isn't covered, confirm with the user before ingesting (30+ min):
   ```bash
   python -m src.manage run_backtest --start-date $1 --end-date $2 --ingest --max-markets $4
   ```
2. Baseline run (no override):
   ```bash
   python -m src.manage run_backtest --start-date $1 --end-date $2 --max-markets $4
   ```
   Capture the `run_id` printed by the runner (format: `exp_<model>_<YYYYMMDD_HHMMSS>`).
3. Override run:
   ```bash
   $3 python -m src.manage run_backtest --start-date $1 --end-date $2 --max-markets $4
   ```
   Capture the second `run_id`.
4. Aggregate both runs from `data/backtest_outputs.db`:
   ```sql
   SELECT
     COUNT(*) AS n,
     AVG(CASE WHEN pnl>0 THEN 1.0 ELSE 0 END) AS wr,
     SUM(pnl)/SUM(position_size_usd) AS roi,
     AVG((final_adjusted_probability - actual_outcome)*(final_adjusted_probability - actual_outcome)) AS brier
   FROM trade_records
   WHERE experiment_run=? AND actual_outcome IS NOT NULL;
   ```
5. Welch's t-test on per-trade `pnl`:
   ```python
   import sqlite3
   from scipy.stats import ttest_ind
   conn = sqlite3.connect('data/backtest_outputs.db')
   a = [r[0] for r in conn.execute("SELECT pnl FROM trade_records WHERE experiment_run=? AND actual_outcome IS NOT NULL", (baseline_id,)).fetchall()]
   b = [r[0] for r in conn.execute("SELECT pnl FROM trade_records WHERE experiment_run=? AND actual_outcome IS NOT NULL", (override_id,)).fetchall()]
   t, p = ttest_ind(a, b, equal_var=False)
   ```

## Output format

Markdown with five sections:

- **Baseline** — `n`, `WR %`, `ROI %`, `Brier` (4 decimals).
- **Override** (`$3`) — same four fields.
- **Delta** — signed diffs (`override − baseline`).
- **p-value** — Welch's t-test on per-trade PnL.
- **Verdict** — `significant improvement` / `significant regression` / `noise` at α=0.05.

## Error handling

- Backtest data not ingested for the range → STOP, confirm with user before running the 30-min ingest.
- Either run has `n < 30` → print "UNDERPOWERED: n={n}, treat verdict as indicative only".
- `scipy` missing locally → `pip install scipy` before step 5.
- `run_id` not captured (runner output changed) → grep `data/backtest_outputs.db` for rows `WHERE started_at > <now-5min>` in `experiment_runs`.

## Related

- `doc-audit` — run before comparing; ensures baseline config is internally consistent.
- `learning-status` — read the calibration / market-type state that fed both runs.
- §2 biweekly loop (future `hypotheses` table) — this skill is the comparison engine for hypothesis testing.

## Unresolved

- Backtest ingest table confirmed as `historical_markets` (not `market_snapshots`). Date range check uses `MIN(created_at), MAX(resolution_datetime)`.
- `trade_records.experiment_run` is the FK column (no `_id` suffix).
