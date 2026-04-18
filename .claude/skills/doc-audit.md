---
name: doc-audit
description: Cross-reference numeric constants and feature flags across .env, src/config.py, CLAUDE.md, and polymarket_system_v2.md to flag drift.
trigger: doc audit, audit docs, config drift, /doc-audit
---

## Purpose

Biweekly (or pre-commit) sanity check that the four sources-of-truth agree on every load-bearing tunable. The last full alignment was commit `bdd01db`; any row flagged is a real drift to resolve before shipping config changes.

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
5. Build one row per constant; 4 columns: `config.py default`, `.env value`, `CLAUDE.md mention`, `v2.md mention`. Flag any row where two populated cells disagree (ignore cells marked "not mentioned").

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

## Related

- Run before any commit that touches `src/config.py`, `.env`, `CLAUDE.md`, or `polymarket_system_v2.md`.
- `backtest-compare` — after resolving drift, validate that the chosen value still performs.

## Unresolved

- Prior subagent reported suspected drift: `PRESCREEN_MIN_CONFIDENCE` may be `0.35` in `config.py` vs `0.25` in `CLAUDE.md`. Verify on first invocation and either update `CLAUDE.md` or the default.
