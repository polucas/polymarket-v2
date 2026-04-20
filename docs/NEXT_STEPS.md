# Next Steps — Polymarket v2

_Last updated: 2026-04-19 after Round 4 deploy (§2b per-type `MIN_EDGE` landed). Round 2 verification window still running through 2026-04-25._

## Monitoring summary (what to watch, at a glance)

| Window | Signal | Action on hit |
|---|---|---|
| Daily | `/morning-check` output | Named env-only lever per alarm row (see skill) |
| 24h post-deploy | First BUY has `spread_at_decision`, `vwap_price`, `orderbook_depth_usd` all non-zero | If any zero → §1b fix regressed |
| 24h post-deploy | First WS reconnect emits `ws_subscription_sent` | If never → §1g subscribe path broken |
| 24h post-Round-4 | Mean executed edge by `market_type` ≥ per-type floor (sports 0.05, crypto_15m 0.04, political/etc 0.03) | If any type below floor → §2b regressed |
| 48h (by 2026-04-21) | `resolution_skipped_unresolved` vs `resolution_fallback_crypto_price` counts by market_type | Decide §1d non-crypto price fallback |
| 7-day (2026-04-18 → 2026-04-25) | Round 2 baseline: WR ≥68%, mean edge ≥0.04, 12–15 trades/day | Lever table in `/morning-check` |
| Weekly (Mondays) | 7-day vs prior 7-day WR / mean edge | See §0 weekly rolling |

---

## 0. Active verification windows

### 0a. Round 2 baseline (through 2026-04-25)

Round 2 fixes (§0b token raise, §0c-1 weak-signals gate, §0c-2 similar_to extractor, §0c-3 adverse instrumentation) + Round 2 env tuning (`PRESCREEN_MAX_TOKENS=1000`, `WEAK_SIGNAL_STRENGTH_THRESHOLD=0.55`) all deployed.

**Goal:** 12–15 trades/day without WR <68% or mean edge <0.04.

**Daily:** `/morning-check` — consolidated skill (service + skip reasons + cash reconcile + today's review + alarm table with env-only levers). Drill into `/vps-health-check`, `/skip-reason-analyzer 24`, `/db-diagnostics cash-reconcile`, `/reviews/{today}` only when flagged.

**Weekly rolling (7d vs prior 7d):**
- WR drop >5pp → investigate market_type breakdown in `daily_reviews`.
- Mean edge drop >0.01 → investigate.
- Trade count rising but WR falling → revert last lever.
- `unrealized_adverse_triggered` counts per market_type → decide if `consecutive_loss_cooldown` needs tuning.

### 0b. Round 3 post-deploy checks (collapse once confirmed)

- **§1b orderbook persistence** — after first post-deploy BUY:
  ```bash
  ssh root@49.13.159.52 "cd /root/polymarket-v2 && sqlite3 data/predictor.db \"SELECT market_id, spread_at_decision, vwap_price, orderbook_depth_usd FROM trade_records WHERE action!='SKIP' ORDER BY timestamp DESC LIMIT 5;\""
  ```
  All three columns must be non-zero. If confirmed → remove this bullet.

- **§1g WS observability** — after first reconnect event:
  ```bash
  ssh root@49.13.159.52 "grep -cE 'ws_subscription_sent|ws_message_decode_error|ws_event_' /root/polymarket-v2/data/bot.log"
  ```
  Expect ≥1 `ws_subscription_sent` per reconnect. If positions are opened and WS never subscribes → investigate. Polling fallback (5min) still covers TP/SL in the meantime.

- **§1e config canonicalization** — deploy sequence ran; redundant `.env` lines removed. No ongoing monitoring.

- **§2b per-type `MIN_EDGE`** — after 24h:
  ```bash
  ssh root@49.13.159.52 "cd /root/polymarket-v2 && sqlite3 data/predictor.db \"SELECT market_type, COUNT(*), ROUND(AVG(edge),4) FROM trade_records WHERE action!='SKIP' AND timestamp > datetime('now','-24 hours') GROUP BY market_type;\""
  ssh root@49.13.159.52 "cd /root/polymarket-v2 && sqlite3 data/predictor.db \"SELECT market_type, COUNT(*) FROM trade_records WHERE action='SKIP' AND skip_reason LIKE 'low_edge%' AND timestamp > datetime('now','-24 hours') GROUP BY market_type;\""
  ssh root@49.13.159.52 "grep low_edge_skip /root/polymarket-v2/data/bot.log | tail -5 | jq '{market_type, min_edge, edge}'"
  ```
  Expect: sports mean edge ≥ 0.05, crypto_15m ≥ 0.04, political ≥ 0.03; more `low_edge` SKIPs on sports/esports than before. If confirmed → collapse this bullet.

---

## 1. Open investigations

### 1d. Resolution coverage — 48h data review (decision by 2026-04-21)

Round 3 instrumented both branches:
- `resolution_fallback_crypto_price` — crypto_15m auto-resolves from `market.yes_price > 0.5` when past window.
- `resolution_skipped_unresolved` — every other market_type hits `continue` with full payload logged (no `actual_outcome` written).

**Decision input:** after 48h, group log counts by `market_type`:
```bash
ssh root@49.13.159.52 "grep resolution_skipped_unresolved /root/polymarket-v2/data/bot.log | jq -r .market_type | sort | uniq -c"
```

**Options:**
- If most skipped trades eventually get `resolved=true` from Polymarket → leave behavior; Polymarket just lags.
- If `resolved=false` persists long past window for political/sports → add per-market-type price fallback (separate PR, not env).

**Priority:** high — learning system is the project's long-term edge. Calibration/Brier stays at prior until real resolutions flow.

### 1g. WS subscription ACK (deferred — observability landed this round)

Round 3 added `ws_subscription_sent` (log-only, no server ACK per spec) and structlog for all three silent-skip paths in `_handle_message()`. Reconnect loop + polling fallback unchanged.

**Open only if:** malformed-event log rate spikes (schema change) or a position opens with no `ws_subscription_sent` following. Otherwise no work needed.

### 1h. `no_direction` is LLM anchoring, not gate regression (resolved 2026-04-19)

Morning check 2026-04-19 showed `no_direction = 182/24h`, above the Round 2 alarm threshold of 50. Drilled in — `WEAK_SIGNAL_STRENGTH_THRESHOLD=0.55` deployed, but `grep weak_signals data/bot.log` returned 0 matches lifetime. Gate never fires for typical S1-S5 signal mix (credibility 0.65-0.95, well above 0.55). Sampled 10 `no_direction` trade_records — every row had `grok_raw_probability == market_price_at_decision` exactly (e.g. sports 0.505/0.505/0.505, crypto 0.820/0.820/0.820), with confidence 0.25-0.52. LLM is echoing market price per `SYSTEM_PROMPT` anchoring ("markets generally efficient; current price IS consensus"). `determine_side()` returns SKIP on exact equality → `no_direction`.

**Actions taken:**
- Raised `no_direction` alarm threshold 50 → 200/24h in [`.claude/skills/morning-check.md`](../../.claude/skills/morning-check.md) + memory [`project_daily_routine.md`](../../../.claude/projects/-home-jedicelli-polymarket-v2/memory/project_daily_routine.md).
- Documented that gate rarely applicable for current signal mix — not dead, but only trips on S6-dominated batches.

**Future consideration (NOT now):** If `no_direction` climbs past 300/24h, tighten `PRESCREEN_MIN_CONFIDENCE` 0.25 → 0.30 (§2a cost lever) or revise SYSTEM_PROMPT anchoring language (code change, separate PR).

### 1i. Prescreen JSON parse failures — chronic malformed LLM output (diagnosed 2026-04-20)

Morning check 2026-04-20 flagged `prescreen_parse_failed` growth +373/11h (~815/24h extrapolated) vs old alarm threshold `>20/24h`. Drilled in — LLM returns HTTP 200 every call; parse fails on ~50% of attempt-0 calls; most recover on attempt-1 retry; rest hit `prescreen_failed_passthrough` → full eval. Zero trade impact.

Token budget NOT the cause: tested progression 500 → 1000 → 1500 with no change in failure rate. Root cause is JSON format inconsistency from MiniMax model (wrong field names, malformed structure, or trailing content).

**Actions taken 2026-04-20:**
- `PRESCREEN_MAX_TOKENS` 1000 → 1500 (insurance only, ~$2/mo cost bump).
- Revised skill alarm threshold `>20/24h` → `>500/24h growth vs 7-day baseline` in [`.claude/skills/morning-check.md`](../../.claude/skills/morning-check.md) + memory.

**Queued code PR:** add pydantic schema validation + JSON-mode enforcement in `call_prescreen()` at `src/engine/grok_client.py`. Should reduce retry overhead (~50% of prescreen calls currently make 2 LLM requests). Estimated savings: ~400 unnecessary calls/24h × $0.0001 ≈ $1.20/mo + latency reduction.

**Priority:** low — system functions correctly via retry+passthrough. Ship with next batched code round.

---

## 2. Profitability Levers (start after 2026-04-25)

Defer until 7 days of post-Round-2 data.

### 2a. Cost
- Tighten `PRESCREEN_MIN_CONFIDENCE` 0.25 → 0.30 after 7-day filter-rate baseline. Expected ~$2/mo savings.
- Cheaper pre-screen model (MiniMax-M2.5 prescreen, M2.7 full eval). Expected $14/mo → ~$12/mo.

### 2b. Quality
- ~~Per-market-type `MIN_EDGE`: sports 5% (2% fee), crypto_15m 4% (1.56% fee), political/etc 3% (0% fee).~~ **Landed Round 4 (2026-04-19).** See §0b verification.
- Ensemble LLM on positions >$100: 2 calls, skip on disagreement >0.15. Expected Brier −0.01 to −0.02, +$1–2/mo.

### 2c. Volume
- Loosen `QUESTION_SIMILARITY_THRESHOLD` 0.60 → 0.65 only if Round 2 extractor-unification still false-positives.
- `no_direction` follow-up — if weak-signals gate reduces it <50/24h, stop. Else investigate prompt.

### Recommended order
1. ~~Per-market-type MIN_EDGE~~ — done (Round 4).
2. Cost levers (prescreen confidence bump, cheaper model) — waiting on 7-day filter-rate baseline.
3. Ensemble on high-stakes — needs volume baseline first.

---

## 3. Biweekly Improvement Loop (Deferred B)

~2–3 days dev. Lands after profitability tuning settled.

- Biweekly cadence (14d), manual approval gate, shadow-mode A/B.
- New DB tables: `hypotheses`, `biweekly_reviews`.
- CLI: `biweekly_review`, `propose_hypothesis`, `approve_hypothesis`, `merge_hypothesis`.
- Endpoint: `GET /experiments/compare?a=<run>&b=<run>` — Welch's t-test + Mann-Whitney.
- Safety: param whitelist, rollback on 3-day `daily_loss_limit` breach.

---

## 4. UI Dashboard (lowest priority)

Replaces raw Datasette browsing. Views: portfolio, trade history, calibration buckets, skip-reason breakdown, Brier by market_type, daily review archive. FastAPI+Jinja+HTMX OR Next.js/SvelteKit. Token/IP auth, stays on VPS.

**Prereq:** §1 + §2 + §3 settled first.
