---
name: prescreen-debug
description: Given a market_id, pull the raw MiniMax prescreen response from bot.log, classify the parse failure, and suggest a fix.
trigger: prescreen debug, debug prescreen, prescreen failure, /prescreen-debug
---

## Purpose

Tactical debugger for the recurring `prescreen_parse_failed` skips on the VPS (see `docs/NEXT_STEPS.md` §0b). Pulls the two most recent prescreen attempts for a single market, classifies *why* parsing failed, and names the fix category.

## Inputs

- `$1` — `market_id` (required). The Polymarket market id from the `trade_records.market_id` column.

## Steps

1. Pull recent prescreen log entries for the market (last 2 attempts + responses):
   ```bash
   ssh root@49.13.159.52 "grep -A 20 '\"$1\"' /root/polymarket-v2/data/bot.log | grep -i 'prescreen' | tail -40"
   ```
2. Pull market context from the DB:
   ```bash
   ssh root@49.13.159.52 "cd /root/polymarket-v2 && sqlite3 data/predictor.db \"SELECT market_question, market_type, resolution_datetime FROM trade_records WHERE market_id='$1' LIMIT 1;\""
   ```
3. Classify the failure against the raw response body from step 1. Check categories in this order:
   - `markdown_fence` — response contains triple backticks.
   - `truncation` — response ends without a closing `}` (likely hit `PRESCREEN_MAX_TOKENS=300`).
   - `prose_wrapper` — response starts with non-`{` text (prose paragraph before JSON).
   - `missing_field` — JSON parses but lacks any of `estimated_probability`, `confidence`, `reasoning` (the `REQUIRED_FIELDS` set in [src/engine/grok_client.py](../../src/engine/grok_client.py)).
   - `unicode` — `market_question` contains non-ASCII characters (emoji/smart quotes).
   - `unknown` — none of the above.
4. Map classification → fix option from `docs/NEXT_STEPS.md` §0b:
   - `markdown_fence` / `prose_wrapper` → **tighten prompt** in `call_prescreen()` (demand strict JSON, no markdown).
   - `truncation` → **raise `PRESCREEN_MAX_TOKENS` 300 → 500**.
   - `missing_field` → **strengthen JSON parse fallback** (strip fences, extract first `{...}` block).
   - `unicode` → **strengthen parser** + add Unicode normalization before prompt.
   - `unknown` → **skip-list cache** (if same market fails 3× in row, go straight to full eval without retry).

## Output format

Markdown report with four sections:

- **Market** — `market_question` (truncate to 120 chars) + `market_type` + `resolution_datetime`.
- **Raw responses** — up to 2 most recent prescreen response bodies in fenced code blocks.
- **Classification** — one of the 6 categories above, with the evidence (quoted snippet).
- **Suggested fix** — the matching fix option, with a one-line rationale.

## Error handling

- Log does not contain `market_id` → report "no recent prescreen activity for `$1` — market may be on the 2h evaluation cooldown (`EVALUATION_COOLDOWN_HOURS`) or was never hit by the scan".
- DB has no row for the `market_id` → report "market_id not in DB — verify the id".
- `bot.log` rotated/missing → report "log rotated, pull from archive under `data/bot.log.*` if retained".

## Related

- `db-diagnostics prescreen-failures $hours` — get the list of offenders first, then invoke this skill on each.
- `trade-replay` — full trade trace for a market that did pass prescreen but then failed downstream.

## Unresolved

- DB column verified as `resolution_datetime` ([src/db/migrations.py:160](../../src/db/migrations.py#L160)), not `resolution_date` as the original plan had.
- `call_prescreen()` confirmed in [src/engine/grok_client.py](../../src/engine/grok_client.py); `REQUIRED_FIELDS = {"estimated_probability","confidence","reasoning"}`.
