---
name: db-diagnostics
description: Parameterized SQL recipes against the VPS predictor.db — skip breakdown, cash reconciliation, prescreen failures, open positions.
trigger: db diagnostics, skip breakdown, cash reconcile, prescreen failures, open positions, /db-diagnostics
---

## Purpose

Named SQL recipes for the four investigations that come up most often on the Polymarket v2 bot. Each recipe is a single SSH + sqlite3 invocation against `/root/polymarket-v2/data/predictor.db` on `root@49.13.159.52`.

## Inputs

- `$1` — `recipe`, one of: `skip-breakdown` | `cash-reconcile` | `prescreen-failures` | `open-positions`.
- `$2` — `hours`, integer, default `48`. Used by recipes with a time window (`skip-breakdown`, `prescreen-failures`); ignored otherwise.

## Steps

Pick the recipe from `$1` and run the matching block. All queries hit `trade_records` unless noted.

### `skip-breakdown`

```bash
ssh root@49.13.159.52 "cd /root/polymarket-v2 && sqlite3 data/predictor.db \"SELECT skip_reason, COUNT(*) FROM trade_records WHERE action='SKIP' AND timestamp > datetime('now','-$2 hours') GROUP BY skip_reason ORDER BY 2 DESC LIMIT 20;\""
```

### `cash-reconcile`

Run three queries, then the `/health` snapshot, then compute the §0d math.

1. Resolved trades aggregate:
   ```sql
   SELECT COUNT(*), SUM(pnl), SUM(position_size_usd)
   FROM trade_records WHERE action!='SKIP' AND actual_outcome IS NOT NULL;
   ```
2. Unresolved / locked positions:
   ```sql
   SELECT COUNT(*), SUM(position_size_usd)
   FROM trade_records WHERE action!='SKIP' AND actual_outcome IS NULL AND exit_type IS NULL AND voided=0;
   ```
3. Early-exit (TP/SL) PnL. `exit_type` values are `take_profit` / `stop_loss`:
   ```sql
   SELECT COUNT(*), SUM(pnl)
   FROM trade_records WHERE exit_type IN ('take_profit','stop_loss');
   ```
4. `curl -s http://localhost:8000/health` → not in the JSON body directly; cash balance needs the portfolio table:
   ```sql
   SELECT cash_balance, total_equity FROM portfolio WHERE id=1;
   ```
5. Expected cash = `INITIAL_BANKROLL (10000) − (query2.sum_position_size) + (query1.sum_pnl) + (query3.sum_pnl)`.

### `prescreen-failures`

```sql
SELECT market_id, COUNT(*)
FROM trade_records
WHERE skip_reason LIKE 'prescreen_parse_failed%'
  AND timestamp > datetime('now','-$2 hours')
GROUP BY market_id ORDER BY 2 DESC LIMIT 20;
```

### `open-positions`

```sql
SELECT record_id, market_id, action, position_size_usd, market_price_at_decision, timestamp
FROM trade_records
WHERE action!='SKIP' AND actual_outcome IS NULL AND exit_type IS NULL AND voided=0
ORDER BY timestamp DESC;
```

## Output format

- `skip-breakdown` — two-column markdown table (`skip_reason`, `count`) sorted desc.
- `cash-reconcile` — bullets with each subquery's result, then a verdict line: `MATCH (delta $<1)` or `DRIFT $X — investigate hypotheses 1-5 in docs/NEXT_STEPS.md §0d`.
- `prescreen-failures` — two-column table (`market_id`, `fail_count`), then a one-line hint: "Feed top offenders into `prescreen-debug`."
- `open-positions` — full table with all 6 columns; summarize count + total `position_size_usd` at the bottom.

## Error handling

- `database is locked` → retry once after 2 seconds, then report the failure.
- Missing table (e.g., `no such table: portfolio`) → report "schema mismatch — check `src/db/migrations.py` SCHEMA_VERSION vs VPS state".
- SSH timeout → report unreachable, do not retry.

## Related

- `skip-reason-analyzer` — deeper per-reason sample + fix category (built on `skip-breakdown`).
- `prescreen-debug` — run on each top offender from `prescreen-failures`.
- `vps-health-check` — consume 24h counts when those look anomalous.

## Unresolved

- `exit_type` values in the code use `take_profit` / `stop_loss` (verified in `src/alerts.py:121` and `src/engine/ws_exit.py`). The plan's original `'tp'`/`'sl'` sentinels were wrong; recipe `cash-reconcile` uses the real values above.
- `/health` endpoint does not expose `cash_balance` directly (verified in [src/main.py:231-240](../../src/main.py#L231-L240)); recipe queries `portfolio` table instead.
