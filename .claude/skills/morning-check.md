---
name: morning-check
description: Daily post-Round-2 monitoring routine in one shot — health, skip reasons, cash reconcile, today's review, with alarm thresholds and env-only rollback hints.
trigger: morning check, daily check, morning routine, daily standup, /morning-check
---

## Purpose

Consolidates the §0a daily routine from `docs/NEXT_STEPS.md` into a single command. Replaces running `/vps-health-check` + `/skip-reason-analyzer 24` + `/db-diagnostics cash-reconcile` + `/reviews/{today}` manually. Applies alarm thresholds from `project_daily_routine.md` and names the exact `.env` rollback lever for each alarm (no redeploy).

## Inputs

None. Always scans the last 24h.

## Steps

1. **Service + 24h activity** (reuse `vps-health-check` logic):
   ```bash
   ssh root@49.13.159.52 'systemctl is-active polymarket'
   ssh root@49.13.159.52 'curl -s http://localhost:8000/health | python3 -m json.tool'
   ssh root@49.13.159.52 "cd /root/polymarket-v2 && sqlite3 data/predictor.db \"SELECT action, COUNT(*) FROM trade_records WHERE timestamp > datetime('now','-24 hours') GROUP BY action;\""
   ```
   Expect `BUY_YES + BUY_NO` non-zero (unless quiet news day).

2. **Top skip reasons 24h** (reuse `skip-reason-analyzer` logic):
   ```bash
   ssh root@49.13.159.52 "cd /root/polymarket-v2 && sqlite3 data/predictor.db \"SELECT skip_reason, COUNT(*) FROM trade_records WHERE action='SKIP' AND timestamp > datetime('now','-24 hours') GROUP BY skip_reason ORDER BY 2 DESC LIMIT 5;\""
   ```
   Also grep bot.log for prescreen parse failures (not recorded as SKIP rows):
   ```bash
   ssh root@49.13.159.52 "grep -c prescreen_parse_failed /root/polymarket-v2/data/bot.log"
   ```
   Compare to yesterday's reading — alarm on growth > 20/24h.

3. **Cash reconcile** (reuse `db-diagnostics cash-reconcile` logic):
   ```bash
   ssh root@49.13.159.52 "cd /root/polymarket-v2 && sqlite3 data/predictor.db \"SELECT cash_balance, total_equity FROM portfolio WHERE id=1;\""
   ssh root@49.13.159.52 "cd /root/polymarket-v2 && sqlite3 data/predictor.db \"SELECT COUNT(*), SUM(pnl), SUM(position_size_usd) FROM trade_records WHERE action!='SKIP' AND actual_outcome IS NOT NULL;\""
   ssh root@49.13.159.52 "cd /root/polymarket-v2 && sqlite3 data/predictor.db \"SELECT COUNT(*), SUM(position_size_usd) FROM trade_records WHERE action!='SKIP' AND actual_outcome IS NULL AND exit_type IS NULL AND voided=0;\""
   ssh root@49.13.159.52 "cd /root/polymarket-v2 && sqlite3 data/predictor.db \"SELECT COUNT(*), SUM(pnl) FROM trade_records WHERE exit_type IN ('take_profit','stop_loss');\""
   ```
   Expected cash = `10000 − locked_position_size + resolved_pnl + early_exit_pnl`. Alarm DRIFT > $50.

4. **Label coverage** (post-Phase-1 deploy: every exited trade must have trade_profitable set):
   ```bash
   ssh root@49.13.159.52 "cd /root/polymarket-v2 && sqlite3 data/predictor.db \"SELECT SUM(CASE WHEN trade_profitable IS NOT NULL THEN 1 ELSE 0 END) AS pnl_labeled, SUM(CASE WHEN actual_outcome IS NOT NULL THEN 1 ELSE 0 END) AS outcome_labeled, COUNT(*) AS total_non_skip FROM trade_records WHERE action != 'SKIP';\""
   ```
   Alarm if `pnl_labeled < total_non_skip` (post-deploy should be 100%). A gap means the exit-path label write regressed.

5. **Yesterday's daily review** (self-check writes PREVIOUS day's report at 06:15 UTC):
   ```bash
   YESTERDAY=$(date -u -d 'yesterday' +%Y-%m-%d)
   ssh root@49.13.159.52 "curl -s http://localhost:8000/reviews/$YESTERDAY"
   ```
   Extract `trades_executed`, `win_rate`, `roi_pct`, `total_pnl`, `health_status`, `llm_insights` (first 200 chars), `llm_recommendations`. If endpoint 404s → note "review missing for $YESTERDAY — check daily_reviews table". Note: `win_rate` and Brier are NULL when all trades exited via TP/SL (expected — early-exit trades have no `actual_outcome`).

## Alarm thresholds (from `project_daily_routine.md`)

Apply these checks to step 2's output. Each alarm names its env-only rollback lever (edit `/root/polymarket-v2/.env` on VPS + `systemctl restart polymarket` — NO redeploy).

| Condition | Interpretation | Lever |
|---|---|---|
| `no_direction > 200/24h` | LLM echoing market price en masse (confirmed 2026-04-19: `grok_raw_prob == market_price` exactly on samples). Not a gate regression — SYSTEM_PROMPT anchoring is deliberate. | `PRESCREEN_MIN_CONFIDENCE` 0.25 → 0.30 (cuts pre-LLM) OR revise SYSTEM_PROMPT anchoring (code change). |
| `prescreen_parse_failed` growth `> 100/24h` in bot.log | Round 5 (2026-04-20) wires `PRESCREEN_MAX_TOKENS` env var (was dead code at hardcoded 300), adds `PrescreenResult` pydantic schema, enables MiniMax JSON-mode. Pre-Round-5 baseline ~800/24h; post-deploy expected <100/24h. | Pre-deploy: alarm expected, safe to ignore until deploy completes. Post-deploy + 24h: if still >100/24h, inspect `raw_preview` field on failure logs. Real fix already in; further drift = schema mismatch (model output drift). |
| `similar_to_*` `> 50/24h` on any single `market_id` | Extractor unification regressed on `src/scheduler.py:474` — investigate. | Possibly raise `QUESTION_SIMILARITY_THRESHOLD` 0.60 → 0.65 if samples are true near-duplicates. |
| `weak_signals_*` total `> 200/24h` | Gate too aggressive. | `WEAK_SIGNAL_STRENGTH_THRESHOLD` 0.45 → 0.35 |
| `weak_signals_*` total `< 10/24h` AND `no_direction > 50/24h` | Gate too lax. | `WEAK_SIGNAL_STRENGTH_THRESHOLD` 0.45 → 0.55 |
| Cash DRIFT `> $50` | Portfolio init or early-exit crediting regressed. | Investigate, don't tune env. |
| `pnl_labeled < total_non_skip` | Exit-path label write regressed. | Investigate ws_exit.py / resolution.py — code change, not env. |
| `consecutive_adverse_*` `> 100/24h` sustained 48h AND `grep unrealized_adverse_triggered` shows real adverse moves | Threshold tuning discussion. | Possibly raise adverse threshold 0.10 → 0.15 in code (separate PR, not env). |

## Output format

Single markdown report with five sections:

1. **Service** — one line: `active + uptime Xh` or flag.
2. **24h activity** — table: action | count; plus execution rate.
3. **Skip reasons 24h** — table: reason | count; plus `prescreen_parse_failed` bot.log count.
4. **Cash reconcile** — bullets: cash_balance, total_equity, locked_position_size, resolved_pnl, early_exit_pnl, expected_cash, delta. Verdict: `MATCH` / `DRIFT $X`.
5. **Daily review** — `trades_executed`, `win_rate`, `mean_edge`, `mean_confidence`, or `not yet written`.

Then an **Alarms** section: every triggered alarm from the table above, with the named lever. If none, print `✓ no alarms`.

End with a one-line verdict: `GREEN` (no alarms, cash MATCH, service active) / `YELLOW` (any alarm triggered) / `RED` (service inactive, SSH down, or DRIFT > $50).

## Error handling

- SSH unreachable → `RED — VPS unreachable`, halt.
- Service inactive → `RED — polymarket not active`, keep pulling DB/bot.log via separate SSH attempts (they may still work).
- `/reviews/$YESTERDAY` 404 → check `sqlite3 daily_reviews WHERE review_date='$YESTERDAY'` directly; if also missing, log failure and move on.
- `sqlite3` locked → retry once after 2s.

## Related

- `vps-health-check` — lighter one-off check, used post-deploy.
- `skip-reason-analyzer` — deeper per-reason sample when this skill flags a skip-reason alarm.
- `db-diagnostics` — individual recipes when one section needs a drill-down.

## Unresolved

- Weekly rolling 7-day-vs-prior-7-day comparison (WR drop >5pp, mean edge drop >0.01) is NOT in this skill — it's a Monday-only activity. Kept in `project_daily_routine.md` under "Weekly rolling" for now. Consider a sibling `weekly-check` skill if the comparison becomes recurring.
- `no_direction > 50/24h` lever says "investigate code" because the Round 2 fix was a pre-LLM gate, not an env flag. If the gate itself needs disabling, set `WEAK_SIGNAL_STRENGTH_THRESHOLD=0.0` in `.env` to short-circuit the gate (accept no_direction volume), but that's a regression, not a fix.
