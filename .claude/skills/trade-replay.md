---
name: trade-replay
description: Reconstruct signal → LLM → adjustment → decision for a single trade record, to debug why the bot acted (or skipped) the way it did.
trigger: trade replay, replay trade, why did it trade, /trade-replay
---

## Purpose

Given a `market_id` or `record_id`, rebuild the full decision trail: signals fed to the LLM, raw MiniMax response, orderbook snapshot, adjustment pipeline, final action. Primary use: debug recurring `no_direction` / `consecutive_adverse_*` / surprising BUY outcomes.

## Inputs

One of:

- `$1 = market_id=<id>` — find the most recent row for that market.
- `$1 = record_id=<uuid>` — exact record lookup.

## Steps

1. Fetch the full trade row:
   ```bash
   ssh root@49.13.159.52 "cd /root/polymarket-v2 && sqlite3 -header data/predictor.db \"SELECT * FROM trade_records WHERE <record_id='X' | market_id='X' ORDER BY timestamp DESC LIMIT 1>;\""
   ```
2. Extract key fields from the row (real column names per [src/db/migrations.py](../../src/db/migrations.py#L41-L86)):
   - Raw LLM: `grok_raw_probability`, `grok_raw_confidence`, `grok_reasoning`, `grok_signal_types`.
   - Adjustment intermediates (stored): `calibration_adjustment`, `signal_weight_adjustment`, `market_type_adjustment`.
   - Final: `final_adjusted_probability`, `final_adjusted_confidence`, `calculated_edge`, `trade_score`.
   - Market state: `market_price_at_decision`, `orderbook_depth_usd`, `spread_at_decision`, `vwap_price`, `fee_rate`.
   - Action: `action`, `skip_reason`, `position_size_usd`, `kelly_fraction_used`.
   - Outcome (if resolved): `actual_outcome`, `pnl`, `brier_score_raw`, `brier_score_adjusted`, `exit_type`, `exit_price`.
3. Parse `grok_signal_types` as JSON → list `source_tier`, `info_type`, `timestamp` per signal. Note: the DB stores this column but runtime `signal_tags` structure is richer (see `CLAUDE.md` signal_tags format).
4. Pull matching log lines from `bot.log`:
   ```bash
   ssh root@49.13.159.52 "grep -B 2 -A 30 '$1_value' /root/polymarket-v2/data/bot.log | tail -120"
   ```
   Filter for `_process_market` entries, prescreen response, full LLM response, edge calc, final decision.
5. Compute the adjustment trail using the stored intermediates + pipeline in [src/learning/adjustment.py](../../src/learning/adjustment.py) (5 steps: Bayesian calibration → signal-type weighting → probability shrinkage → market-type edge penalty → temporal confidence decay). Intermediates for steps 3 and 5 are *not* stored; compute largest-swing step as `max(|calibration_adjustment|, |signal_weight_adjustment|, |market_type_adjustment|, |final_adjusted_probability − grok_raw_probability|)` and name it.

## Output format

Markdown with six sections:

- **Trade summary** — action, size, `calculated_edge`, `skip_reason` (if SKIP), outcome (if resolved).
- **Signals used** — table of `source_tier` / `info_type` / `timestamp` from `grok_signal_types`.
- **LLM raw response** — `grok_reasoning` in a fenced block.
- **Orderbook at decision** — `market_price_at_decision`, `spread_at_decision`, `orderbook_depth_usd`, `vwap_price` (pulled from DB; if log has richer bid/ask, include it).
- **Adjustment trail** — `grok_raw_probability` → +`calibration_adjustment` → +`signal_weight_adjustment` → +`market_type_adjustment` → `final_adjusted_probability`, with each step diff as a signed number.
- **Verdict** — one sentence naming the driver: prompt quality / signal mix / calibration bias / market state / execution gating.

## Error handling

- `market_id` / `record_id` not found in DB → "not in DB — verify the id".
- Log file missing the market_id (rotated) → continue with DB-only trace, note "log data purged, no LLM response text".
- `grok_signal_types` is NULL or empty JSON → note "no signals recorded (headline-only or Tier 2)".

## Related

- `skip-reason-analyzer` — use to pick interesting markets, then replay each.
- `prescreen-debug` — subset of this skill for markets that failed at the prescreen gate.

## Unresolved

- There is no `signal_tags` DB column (contrary to earlier docs); the persisted column is `grok_signal_types` (TEXT/JSON). Runtime `signal_tags` built from signal objects per `CLAUDE.md` is not stored verbatim.
- Steps 3 (probability shrinkage) and 5 (temporal decay) intermediates are not stored per-trade — they are inferred from the diff between adjusted probability and the sum of the three stored adjustments.
