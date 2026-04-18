---
name: skip-reason-analyzer
description: Top 3 skip_reasons in the last N hours, 5-market sample per reason, matched with a suggested fix category.
trigger: skip analyzer, analyze skips, top skip reasons, /skip-reason-analyzer
---

## Purpose

Automates the §0c investigation loop: instead of hand-crafting the SQL every morning, name the top 3 skip reasons over a time window, sample 5 markets per reason, and tag each with the right fix lane.

## Inputs

- `$1` — `hours`, integer, default `48`.

## Steps

1. Top 3 skip reasons in the window:
   ```bash
   ssh root@49.13.159.52 "cd /root/polymarket-v2 && sqlite3 data/predictor.db \"SELECT skip_reason, COUNT(*) FROM trade_records WHERE action='SKIP' AND timestamp > datetime('now','-$1 hours') GROUP BY skip_reason ORDER BY 2 DESC LIMIT 3;\""
   ```
   Also capture the total skip count for the window to compute percentages:
   ```sql
   SELECT COUNT(*) FROM trade_records WHERE action='SKIP' AND timestamp > datetime('now','-$1 hours');
   ```
2. For each of the top 3 reasons, draw 5 random samples (use `LIKE '<stem>%'` because reasons often have suffixes like `_-0.0200` or `_<market_id>`):
   ```sql
   SELECT market_id, substr(market_question,1,80), market_type, final_adjusted_probability, market_price_at_decision
   FROM trade_records
   WHERE skip_reason LIKE '<reason_stem>%' AND timestamp > datetime('now','-$1 hours')
   ORDER BY RANDOM() LIMIT 5;
   ```
3. Tag each reason with a fix category (stem match):
   | Stem | Fix category |
   |---|---|
   | `no_direction` | §1c prompt tuning — pull raw LLM responses via `trade-replay`. |
   | `low_edge_` | Expected — edge math working. Consider per-market-type `MIN_EDGE`. |
   | `market_cooldown` | Expected — 24h dedup is working. |
   | `similar_to_` | §1c threshold — verify samples are true near-duplicates before loosening. |
   | `consecutive_adverse_` | §0c Monk Mode cooldown — check if concentrated in one `market_type`. |
   | `grok_failed` | LLM API errors — grep `data/bot.log` for rate-limit / 5xx. |
   | `prescreen_filtered_` | Gate working. Run `doc-audit` if ratio > 40 % to confirm thresholds. |
   | `daily_loss_limit` / `weekly_loss_limit` / `max_exposure` | Monk Mode lockout — see `polymarket_system_v2.md` §9. |
   | `ranked_out` | Cluster detection dropped it — only a concern if frequent. |
4. Flag any reason accounting for more than **20 %** of total skips.

## Output format

Three sections (one per top reason). Each section:

- Header with `skip_reason`, count, percent of total.
- Markdown table of the 5 sampled markets (5 columns from step 2).
- One-paragraph **Suggested fix** from the table above.

Footer: flag any reason >20 % with a leading `⚠`.

## Error handling

- Fewer than 3 distinct reasons in window → report what exists, skip the missing rows.
- Zero skips in window → "quiet window — no skip activity in the last `$1` h".
- `timestamp` column has mixed timezones → sqlite `datetime('now','-N hours')` uses local time; the bot writes UTC. If results look shifted, use `datetime('now','-$1 hours','utc')`.

## Related

- `db-diagnostics` recipe `skip-breakdown` — the raw aggregate this skill builds on.
- `trade-replay` — drill into one sampled market.
- `prescreen-debug` — when `prescreen_filtered_` or `prescreen_parse_failed` dominates.

## Unresolved

- Additional reasons found in `src/engine/trade_decision.py`: `daily_loss_limit`, `weekly_loss_limit`, `max_exposure`, plus `ranked_out` from the ranker fallback. Added to the table above.
- Evaluation-cooldown skips are silent (no DB record) per `CLAUDE.md` — they never appear in the query results.
