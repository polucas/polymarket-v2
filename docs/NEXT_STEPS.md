# Next Steps — Polymarket v2

_Last updated: 2026-04-18 after Round 2 deploy (§0b, §0c-1/2/3 landed)._

## 0. Monitoring (post Round-2 verification window: 2026-04-18 → 2026-04-25)

Round 1 fixes (§0d portfolio init, §0c narrow empty-signals short-circuit, §0b validator soft-fields) + Round 2 fixes (§0b token raise, §0c-1 weak-signals gate, §0c-2 similar_to extractor unification, §0c-3 adverse instrumentation) are all deployed. Watch these metrics for 48h-7d to confirm no regression.

### 0a. Daily trade-count monitoring (ongoing)

**Goal:** grow daily trade count without degrading bet quality.

**Baseline after Round 2:** 22 executed / 48h (~11/day). Target 12-15/day without dropping WR <68% or mean edge <0.04.

**Morning routine:** `/morning-check` — single skill consolidating all four checks below plus the alarm-threshold table (see `.claude/skills/morning-check.md`). Alarms printed inline name the exact env-only lever. Individual skills remain for drill-downs:
- `/vps-health-check` — systemd + 24h action counts
- `/skip-reason-analyzer 24` — sampled markets per top reason + fix category
- `/db-diagnostics cash-reconcile` — DRIFT detail
- `/reviews/{today}` — raw daily self-check JSON

**Weekly rolling comparison (7-day vs prior 7-day):**
- WR drop >5pp → investigate
- Mean edge drop >0.01 → investigate
- Daily trade count rising but WR falling → revert last lever
- `unrealized_adverse_triggered` log events per market_type → decide if `consecutive_loss_cooldown` needs tuning

### 0b. `prescreen_parse_failed` — monitor post-fix

**Round 2 fix:** raised `PRESCREEN_MAX_TOKENS` 300 → 500 (`src/config.py`). Deterministic truncation on 8 repeat-offender markets should stop.

**Verify:**
```bash
ssh root@49.13.159.52 "grep -c prescreen_parse_failed /root/polymarket-v2/data/bot.log"
ssh root@49.13.159.52 "/root/polymarket-v2/venv/bin/python -m src.manage prescreen_debug --market-id 1994998"  # (if the CLI exists; else run prescreen-debug skill)
```

**Alarm:** if parse_failed count grows by >50/hour post-deploy, escalate:
1. Inspect raw MiniMax response for an offender (requires adding response logging to `call_prescreen`).
2. Consider `PRESCREEN_MAX_TOKENS` 500 → 700 or tighten prompt to force shorter JSON.
3. Skip-list cache (3 consecutive fails → skip pre-screen on that market).

### 0c. Skip-reason health — monitor post-Round-2

48h pre-Round-2 VPS state was 557 `no_direction`, 95 `consecutive_adverse_4`, 3 hot `similar_to_*` blockers (1943736/1928750/1987122).

**Round 2 fixes:**
- `no_direction`: weak-signals gate (`WEAK_SIGNAL_STRENGTH_THRESHOLD=0.45`) now skips pre-LLM when avg signal credibility is low. Expected post-deploy: `no_direction` <50/24h, `weak_signals_*` appears in breakdown.
- `similar_to_*`: unified `extract_keywords()` on both sides of Jaccard. Expected: hot-blocker counts drop sharply; false-positive blocks gone.
- `consecutive_adverse_4`: instrumentation only (`unrealized_adverse_triggered` structlog on 10% threshold crossing). No threshold change — collect 48h data then decide.

**Decision after 48h of data:**
- If `unrealized_adverse_triggered` logs show real adverse moves → leave `consecutive_loss_cooldown=3` alone.
- If most triggers are small temporary fluctuations (adverse bounces 10% then reverses) → consider raising adverse threshold 0.10 → 0.15 OR requiring sustained adverse for N minutes.
- If `weak_signals_*` dominates >200/24h → lower threshold to 0.35 via `.env`.


### 0d. Skip-reason health — monitor post-Round-2

- no_signals=303 dominant — not alarm (markets with zero Twitter+RSS signals, expected). Possible volume lever: broaden RSS feed list or Twitter query scope, but out of scope for monitoring window.


---

## 1. Open investigations (not yet addressed)

### 1b. Orderbook zero-fill persistence bug

**Finding:** Trade-replay shows `orderbook_depth_usd=0`, `spread_at_decision=0`, `vwap_price=0` stored for markets the log confirms had actual depth (CLOB HTTP 200). Data fetched but not persisted.

**Scope:** degrades `spread_aware_edge` accuracy — falls back to `market.yes_price` midpoint when orderbook columns are zero.

**Diagnostic:**
- `src/pipelines/polymarket.py` → `get_order_book()` return path.
- `src/db/sqlite.py` → `save_trade()` — confirm `spread_at_decision`, `orderbook_depth_usd`, `vwap_price` are extracted from the orderbook object and not left as zero default.

**Priority:** medium. Affects edge accuracy on every executed trade.

### 1d. Learning system at prior — no resolved trades

**Finding:** all calibration buckets at Beta(1,1). `market_type_performance` + `signal_trackers` empty. 22 early-exit trades have PnL but no `actual_outcome` (correctly excluded).

**Why:** either (a) auto-resolution job not finding settled markets and writing `actual_outcome`, or (b) all 22 positions expired worthless and `actual_outcome=0` was never recorded.

**Diagnostic:**
```bash
ssh root@49.13.159.52 "grep 'auto_resolve\|resolution' /root/polymarket-v2/data/bot.log | tail -50"
ssh root@49.13.159.52 "sqlite3 data/predictor.db 'SELECT record_id, market_id, resolution_datetime, exit_type, actual_outcome FROM trade_records WHERE action!=\"SKIP\" ORDER BY timestamp DESC;'"
```

**Priority:** high — learning system is the project's long-term edge. Without resolved trades, calibration/Brier never converges.

### 1e. Config drift — decision required

Two confirmed drifts between `config.py` defaults, deployed `.env`, and CLAUDE.md:
1. `PRESCREEN_MIN_CONFIDENCE`: `config.py` default = `0.35`, `.env` = `0.25`, docs say `0.25`. Which is canonical?
2. `EVALUATION_COOLDOWN_HOURS`: `config.py` default = `2.0`, `.env` = `4.0`, CLAUDE.md says "2h". Is 4h intentional?

**Action:** user picks canonical values; update the losing side (remove `.env` override OR update default + docs).

### 1g. WebSocket exit monitor reconnection

**Finding:** `vps-health-check` showed `ws_exit.connected=false`. Zero open positions now → no immediate risk. But verify reconnection logic before first live position.

**Diagnostic:** `src/engine/ws_exit.py` reconnection loop. Confirm `RealTimeExitManager` is started in lifespan and re-subscribes on disconnect.

**Priority:** low until first live position.

---

## 2. Profitability Levers (wait for Round 2 baseline)

Defer until 7 days of post-Round-2 data.

### 2a. Cost

- **Tighten `PRESCREEN_MIN_CONFIDENCE` 0.25 → 0.30**
  - Prereq: 7 days of filter-rate data.
  - Expected: ~$2/mo savings.
- **Cheaper pre-screen-only model** (MiniMax-M2.5 for prescreen, M2.7 for full eval).
  - Expected: $14/mo → ~$12/mo.

### 2b. Quality

- **Per-market-type `MIN_EDGE`:** sports 5% (covers 2% fee), crypto_15m 4% (covers 1.56% fee), political/etc 3% (0% fee).
- **Ensemble LLM on high-stakes trades** (position >$100): 2 calls, skip on disagreement >0.15.
  - Expected: Brier -0.01 to -0.02, +$1-2/mo cost.

### 2c. Volume

- **Loosen similarity threshold 0.60 → 0.65** — only if Round 2 extractor-unification leaves residual false positives.
- **`no_direction` follow-up** — if weak-signals gate reduces it to <50/24h, stop. If not, investigate prompt.

### Recommended order (after Round 2 baseline)

1. Per-market-type MIN_EDGE
2. Cost levers (prescreen confidence bump, cheaper model)
3. Ensemble on high-stakes (needs volume baseline first)

---

## 3. Biweekly Improvement Loop (Deferred B)

~2-3 days dev. Lands after profitability tuning settled.

**User preferences (unchanged):**
- Biweekly cadence (14d)
- Manual approval gate
- Shadow-mode A/B (log hypothesis configs alongside real trades)
- Structured JSON hypothesis on biweekly review + free-text daily

**New DB tables:** `hypotheses`, `biweekly_reviews`.
**CLI:** `biweekly_review`, `propose_hypothesis`, `approve_hypothesis`, `merge_hypothesis`.
**Endpoint:** `GET /experiments/compare?a=<run>&b=<run>` — Welch's t-test + Mann-Whitney.
**Safety:** param whitelist, rollback on 3-day `daily_loss_limit` breach.

---

## 4. UI Dashboard (lowest priority)

Visual dashboard replacing raw Datasette browsing.

**Views:** live portfolio, trade history, calibration buckets, skip-reason breakdown, Brier by market_type, daily review archive.
**Source:** read-only `data/predictor.db` + `/health` + `/reviews`.
**Stack options:** FastAPI + Jinja + HTMX (lightweight) OR separate Next.js/SvelteKit hitting new JSON endpoints.
**Auth:** token or IP allowlist; stays on VPS.

**Prereq:** §1 + §2 + §3 settled first.
