---
name: learning-status
description: Snapshot of calibration buckets, per-market-type performance, and signal-tracker lift to ground tuning decisions.
trigger: learning status, calibration snapshot, market type brier, signal lift, /learning-status
---

## Purpose

Before tuning per-market-type `MIN_EDGE` (§1b) or resetting calibration, get the current state of all three learning layers from the VPS.

## Inputs

None.

## Steps

1. Calibration buckets (Beta distribution alpha/beta per confidence bucket):
   ```bash
   ssh root@49.13.159.52 "cd /root/polymarket-v2 && sqlite3 data/predictor.db \"SELECT bucket_range, alpha, beta, (alpha*1.0/(alpha+beta)) AS mean_observed, updated_at FROM calibration_state ORDER BY bucket_range;\""
   ```
   Derive "bias" as `mean_observed − bucket_midpoint` in the skill's post-processing (not in SQL).
2. Per-market-type performance (table name is `market_type_performance`, not `market_type_state`):
   ```bash
   ssh root@49.13.159.52 "cd /root/polymarket-v2 && sqlite3 data/predictor.db \"SELECT market_type, total_trades, total_pnl, total_observed, counterfactual_pnl, brier_scores FROM market_type_performance ORDER BY total_trades DESC;\""
   ```
   `brier_scores` is a JSON array of floats; compute `avg(brier)` in post-processing.
3. Signal-tracker lift (table name is `signal_trackers`, not `signal_tracker_state`). Counts are stored, lift is derived:
   ```bash
   ssh root@49.13.159.52 "cd /root/polymarket-v2 && sqlite3 data/predictor.db \"SELECT source_tier, info_type, market_type, present_winning, present_losing, absent_winning, absent_losing, (present_winning + present_losing) AS n_present FROM signal_trackers WHERE (present_winning + present_losing) >= 5 ORDER BY n_present DESC LIMIT 40;\""
   ```
   Lift formula (compute in post-processing):
   `lift = (p_win_when_present / p_win_when_absent)` where
   `p_win_when_present = present_winning / max(1, present_winning+present_losing)` and
   `p_win_when_absent = absent_winning / max(1, absent_winning+absent_losing)`.
4. Raw Brier by market_type sanity check:
   ```bash
   ssh root@49.13.159.52 "cd /root/polymarket-v2 && sqlite3 data/predictor.db \"SELECT market_type, COUNT(*), AVG((final_adjusted_probability - actual_outcome)*(final_adjusted_probability - actual_outcome)) FROM trade_records WHERE action!='SKIP' AND actual_outcome IS NOT NULL GROUP BY market_type;\""
   ```

## Output format

Four markdown tables, in this order: **Calibration**, **Market-type performance**, **Signal-tracker lift (top 15 by |lift − 1|)**, **Brier by market_type (raw)**.

Row-level flags:

- Calibration bias `> 0.05` (absolute) → `⚠ drift`.
- Market-type Brier `> 0.25` → `⚠ poor`.
- Signal lift `> 1.30` with `n_present ≥ 10` → `✓ signal working`.
- Signal lift `< 0.70` with `n_present ≥ 10` → `⚠ anti-signal`.

End with a one-sentence recommendation (e.g., "reset `low` calibration bucket" / "raise sports MIN_EDGE to 0.05").

## Error handling

- Any of the three learning tables empty → "no resolved trades yet — system has not learned anything; skip flags".
- `brier_scores` JSON parse error → fall back to the raw-trades query (step 4) for that row.
- Column-name mismatch → re-check `src/db/migrations.py` and update this skill.

## Related

- `backtest-compare` — validate any tuning change suggested by this skill.
- `live-readiness` — reuses table 4 (raw Brier by type) as its performance sanity check.

## Unresolved

- Real table names differ from the original plan: `market_type_performance` (not `_state`), `signal_trackers` (not `signal_tracker_state`). Verified against [src/db/migrations.py](../../src/db/migrations.py).
- Calibration stores Beta `alpha`/`beta` per `bucket_range`, not `mean_predicted`/`mean_actual`. Bias derived in post-processing.
- `market_type_performance.brier_scores` is a JSON blob; the skill parses it client-side.
- `signal_trackers` stores four win/loss counts; `lift` and `n_trades` are computed, not stored.
