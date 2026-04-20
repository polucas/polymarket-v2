---
name: doc-audit
description: Cross-reference numeric constants and feature flags across .env, src/config.py, CLAUDE.md, polymarket_system_v2.md, and docs/NEXT_STEPS.md to flag drift.
trigger: doc audit, audit docs, config drift, /doc-audit
---

## Purpose

Biweekly (or pre-commit) sanity check that the five sources-of-truth agree on every load-bearing tunable. The last full alignment was commit `bdd01db`; any row flagged is a real drift to resolve before shipping config changes.

`docs/NEXT_STEPS.md` is included because monitoring tables + pending-lever sections frequently reference specific thresholds (e.g. `TIER1_MIN_EDGE=0.03`, `PRESCREEN_MIN_CONFIDENCE=0.25`), and those strings rot when the underlying default changes via `.env` tuning.

## Inputs

None. Runs locally in `/home/jedicelli/polymarket-v2/`.

## Steps

1. Extract defaults from `src/config.py`. Pydantic `Settings` uses `= Field(default=...)` and bare `= <literal>`:
   ```bash
   rg -n '^\s+[A-Z_]+\s*:\s*(int|float|bool|str).*=' src/config.py
   ```
2. Extract `.env` values:
   ```bash
   rg -n '^[A-Z_][A-Z0-9_]*=' .env
   ```
3. Extract backticked mentions in `CLAUDE.md`:
   ```bash
   rg -n '`[A-Z_]{3,}[A-Z0-9_]*`' CLAUDE.md
   ```
4. Extract backticked mentions in `polymarket_system_v2.md`:
   ```bash
   rg -n '`[A-Z_]{3,}[A-Z0-9_]*`' polymarket_system_v2.md
   ```
5. Extract backticked mentions in `docs/NEXT_STEPS.md`:
   ```bash
   rg -n '`[A-Z_]{3,}[A-Z0-9_]*`' docs/NEXT_STEPS.md
   ```
   Also capture numeric thresholds in the monitoring table (row heads like `sports 0.05`, `crypto_15m 0.04`, `political/etc 0.03` for §2b per-type MIN_EDGE), since those bare numbers matter even without a backticked name.
6. Build one row per constant; 5 columns: `config.py default`, `.env value`, `CLAUDE.md mention`, `v2.md mention`, `NEXT_STEPS.md mention`. Flag any row where two populated cells disagree (ignore cells marked "not mentioned").

Mandatory constants to include (even if not drifting):

`TIER1_MIN_EDGE`, `TIER2_MIN_EDGE`, `MAX_POSITION_PCT`, `KELLY_FRACTION`, `MIN_TRADEABLE_PRICE`, `MAX_TRADEABLE_PRICE`, `MARKET_PAGE_SIZE`, `MARKET_FETCH_PAGES`, `PRESCREEN_MIN_EDGE`, `PRESCREEN_MIN_CONFIDENCE`, `PRESCREEN_MAX_TOKENS`, `EVALUATION_COOLDOWN_HOURS`, `MARKET_COOLDOWN_HOURS`, `QUESTION_SIMILARITY_THRESHOLD`, `DAILY_API_BUDGET_USD`, `EARLY_EXIT_ENABLED`, `INITIAL_BANKROLL`.

## Output format

Single markdown table with the columns listed above. Below the table:

- Count of drifts (rows where at least two populated cells disagree).
- `✓ docs aligned as of <YYYY-MM-DD>` if zero drifts.
- For each drift, a one-sentence suggested resolution (which source is authoritative — usually `src/config.py` for defaults, `.env` for deployed values).

## Error handling

- Zero matches from step 1 → "parse failure: `src/config.py` schema changed (no `Field` / annotated defaults found), update this skill's regex".
- `.env` missing → note "no .env on dev machine; compare only 3 sources".
- A constant is mentioned in docs but absent from `src/config.py` → flag as "stale doc reference".
- `docs/NEXT_STEPS.md` references a threshold that conflicts with current `.env` (e.g. doc says `PRESCREEN_MIN_CONFIDENCE=0.25` but `.env` has `0.35`) → flag as "NEXT_STEPS stale — update after most recent Round tuning".

## Related

- Run before any commit that touches `src/config.py`, `.env`, `CLAUDE.md`, or `polymarket_system_v2.md`.
- `backtest-compare` — after resolving drift, validate that the chosen value still performs.

