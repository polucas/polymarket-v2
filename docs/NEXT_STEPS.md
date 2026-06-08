# Next Steps — Polymarket v2

_Last updated: 2026-06-08 — Fresh DB after drift root-cause + atomic save fix. Drift monitoring Layers 1+2 shipped. SL/TP soak restarts from n=0._

## Current state

- **Mode:** paper, `ENVIRONMENT=paper`
- **DB:** fresh as of 2026-06-08 06:54 UTC (full wipe, schema v9, $10K bankroll)
- **Old session archive:** `data/predictor.db.session-end-1780901680` (29 MB, $792 drift)
- **Service:** active, fresh start, cash_drift_ok = 0.0 at boot
- **LLM provider:** MiMo / Xiaomi (`mimo-v2.5-pro`, base `https://api.xiaomimimo.com/v1`)
- **Active flags (`.env`):**
  - `STOP_LOSS_ENABLED=false` (SL A/B experiment — soak from n=0)
  - `TAKE_PROFIT_ENABLED=false` (TP A/B experiment — soak from n=0)
  - `TIER1_DAILY_CAP=50`
  - `PRESCREEN_MIN_EDGE=0.025`, `PRESCREEN_MIN_CONFIDENCE=0.30`
  - `MIN_HOURS_TO_RESOLUTION=0.25`, `MIN_MARKET_VOLUME_24H=20000.0`
  - `TWITTER_ENABLED=false`
  - `MARKET_FETCH_PAGES=10` (10× pagination fix held)
- **RSS:** 44 feeds, ~30 working from VPS (14 Cloudflare-blocked, see [project_vps_rss_ipblock](../.claude/projects/-home-jedicelli-polymarket-v2/memory/project_vps_rss_ipblock.md))
- **Drift monitoring:** startup check + 30min periodic gauge → `drift_history` table + structured `cash_mutation` log per trade
- **Atomic save:** `save_trade_with_portfolio` wraps entry cash deduction + trade record write in `BEGIN IMMEDIATE`/`COMMIT` (commit `7a45032`)

---

## Priority 1 — Drift observation (active, automatic)

Confirm Layers 1+2 monitoring stack works end-to-end. No action needed unless drift detected.

### What's already shipped (commit `1a72d05`)

- **Schema v9** — `drift_history` table (timestamp, actual_cash, expected_cash, drift, n_entries, n_exits, locked_replay)
- **APScheduler job** "Cash drift periodic check" runs every 30min → logs + persists row
- **Per-trade audit log** — `cash_mutation` (operation: ENTRY/EXIT) on every cash write with cash_before / cash_after / delta / trade_id
- **Startup check** in `src/main.py` lifespan — runs `check_cash_drift` after `init_portfolio_if_missing`

### Observation checklist

```bash
# 1. First periodic row landed (30min after boot)
ssh root@49.13.159.52 'cd /root/polymarket-v2 && sqlite3 data/predictor.db "SELECT COUNT(*), MIN(timestamp), MAX(timestamp), ROUND(MAX(ABS(drift)),2) FROM drift_history;"'

# 2. Drift trajectory
ssh root@49.13.159.52 'cd /root/polymarket-v2 && sqlite3 data/predictor.db "SELECT timestamp, ROUND(drift,2), n_entries, n_exits FROM drift_history ORDER BY id DESC LIMIT 20;"'

# 3. Find growth windows
ssh root@49.13.159.52 'cd /root/polymarket-v2 && sqlite3 data/predictor.db "SELECT timestamp, ROUND(drift,2), ROUND(drift - LAG(drift) OVER (ORDER BY timestamp), 4) AS d_drift FROM drift_history ORDER BY timestamp DESC LIMIT 20;"'

# 4. Per-trade audit
ssh root@49.13.159.52 'grep cash_mutation /root/polymarket-v2/data/bot.log | tail -10'

# 5. Growth events alarmed
ssh root@49.13.159.52 'grep cash_drift_growth_event /root/polymarket-v2/data/bot.log | tail -5'
```

### Alarm thresholds (built into `drift_monitor.check_cash_drift_periodic`)

| `|drift|` | Log level | Event name |
|---|---|---|
| `< 5.0` | info | `cash_drift_ok` |
| `5.0 - 49.99` | warning | `cash_drift_growing` |
| `>= 50.0` | error | `cash_drift_alarm` |
| `Δdrift > 1.0` between checks | warning | `cash_drift_growth_event` |

### If drift fires

1. Find first non-zero drift row in `drift_history` → timestamp T
2. Grep `cash_mutation` events in `bot.log` around T ± 5 min
3. Sum logged deltas; compare to actual `cash_balance` change between two `drift_history` rows
4. Orphan delta = culprit trade. Missing trade row = interrupt → grep `CancelledError` in same window
5. → escalate to **Priority 3** (build Layers 3-5)

---

## Priority 2 — no_signal audit + RSS feed expansion

Bot was wiped 2026-06-08; new no_signal volume needs to accumulate before audit is meaningful. **Target: run after 48-72h of fresh trading** (~500+ SKIP rows in `trade_records`).

### Run the existing skill

```bash
# Defaults: N=30 sampled markets, K=24h window, exclude sports/esports
/no-signal-rss-audit

# Or custom: more samples + longer window
/no-signal-rss-audit 50 72 sports,esports,crypto_15m
```

Skill ([.claude/skills/no-signal-rss-audit.md](../.claude/skills/no-signal-rss-audit.md)) pulls `no_signals` skip samples, runs a live RSS poll on VPS, checks per-market keyword overlap, clusters uncovered patterns, and recommends candidate feeds. Diagnostic-only — never modifies `rss_feeds.yaml`.

### Decision branch

| Skill verdict | Action |
|---|---|
| >70% of no_signals markets uncovered AND coverable | Add HIGH/MOSTLY-factuality feeds from skill recommendations. Update `config/rss_feeds.yaml` + `config/known_sources.yaml`. Restart bot. |
| >70% uncovered AND structurally unservable (e.g. local elections, tweet-count meta-markets) | Tighten upstream filter — bump `MIN_MARKET_VOLUME_24H` or extend `DISABLED_MARKET_TYPES`. |
| <30% uncovered | Skip. Bot is not signal-bound. |

### Constraints

- **VPS IP-block**: 14 feeds already blocked by Cloudflare ([project_vps_rss_ipblock](../.claude/projects/-home-jedicelli-polymarket-v2/memory/project_vps_rss_ipblock.md)). Recommend new feeds from publishers known to allow DC IPs (gov sites, EU institutions, smaller outlets).
- **Factuality bar**: HIGH or MOSTLY only per MBFC / AllSides / NewsGuard. CENTER bias preferred.
- **Signal cap**: full-eval 10, prescreen 5 ([context_builder.py:104,172](../src/pipelines/context_builder.py)) — no need to raise; gap is sources, not capacity.
- **Volume budget**: ~45 feeds OK. Beyond 60 risks 30s poll cycle overflow.

### Verification post-deploy

```bash
# Skip-reason distribution
ssh root@49.13.159.52 "cd /root/polymarket-v2 && sqlite3 data/predictor.db \"SELECT skip_reason, COUNT(*) FROM trade_records WHERE action='SKIP' AND timestamp > datetime('now','-24 hours') GROUP BY skip_reason ORDER BY 2 DESC LIMIT 10;\""

# Expected: no_signals proportion drops by 10-30pp
```

---

## Priority 3 — SL / TP soak experiments (restart from n=0)

Both flags off → ALL closures via natural resolution. Snapshots accumulate in `trade_price_snapshots`. Soak target: **n ≥ 20 held-to-resolution trades** per [feedback_statistical_significance](../.claude/projects/-home-jedicelli-polymarket-v2/memory/feedback_statistical_significance.md).

### Daily monitoring (`/morning-check`)

- Watch `cash_drift_ok` line in startup logs (drift=0 expected)
- Watch trade volume, exec rate, daily PnL
- Watch new `cash_drift_growth_event` warnings

### Decision points

| Metric | Pass | Tune | Block |
|---|---|---|---|
| Held-to-resolution count | ≥20 | 10-19 (wait) | bot not trading |
| Cash drift | `$0` | `<$10` | `>$50` |
| Daily realized PnL | net positive | even | net negative |

When n ≥ 20:

```bash
ssh root@49.13.159.52 'cd /root/polymarket-v2 && venv/bin/python3 scripts/sl_analysis.py --since 2026-06-08'
ssh root@49.13.159.52 'cd /root/polymarket-v2 && venv/bin/python3 scripts/tp_analysis.py --since 2026-06-08'
```

Decide: re-enable SL/TP at the best threshold OR keep disabled.

---

## Priority 4 — Layers 3-5 (build when drift recurs)

Deferred — implement only if drift > $10 detected via Priority 1 alarms. Plan in [/home/jedicelli/.claude/plans/lets-restart-our-work-enchanted-curry.md](../.claude/plans/lets-restart-our-work-enchanted-curry.md) covers Layers 1+2; extend for these.

### Layer 3 — `CancelledError` trap with stage marker

Wrap `execute_trade` body with explicit `try/except asyncio.CancelledError` that logs which stage was interrupted (pre-atomic-save, post-atomic-save). Today's atomic transaction makes this rare, but the marker would prove it.

### Layer 4 — Daily drift summary in `daily_reviews`

Append fields to `daily_reviews`: `drift_today`, `drift_total_lifetime`, `drift_delta_24h`. Surface in `/reviews/{date}` endpoint. Add Telegram alert when re-enabled.

### Layer 5 — `/diagnose-drift` skill

New `.claude/skills/diagnose-drift.md`. Reads first growth-event in `drift_history`, greps log around that timestamp, outputs ranked list of suspect events (`CancelledError`, mid-write exceptions, mismatched `cash_mutation` deltas vs actual cash change).

---

## Priority 5 — Phase C live launch (deferred)

Pre-flight checklist + risk-scaled launch unchanged from prior plan. **Blocked on:**

1. Drift monitoring stable for 7+ days at $0
2. SL/TP soak completes (n ≥ 20 per axis) and config decided
3. Polymarket L2 API creds + funding

### C1-C7 steps (unchanged, archive of original Phase C plan)

```bash
# C1. Verify POLYMARKET_FUNDER_ADDRESS (42 chars or empty)
ssh root@49.13.159.52 "grep '^POLYMARKET_FUNDER_ADDRESS=' /root/polymarket-v2/.env | awk -F= '{print length(\$2)}'"

# C2. Generate L2 API creds via polymarket.com web UI OR ClobClient.create_api_key()
# C3. Fund Polymarket wallet ($500 USDC.e on Polygon)
# C4. CLOB allowance: ssh root@49.13.159.52 "cd /root/polymarket-v2 && venv/bin/python3 scripts/setup_clob_allowance.py"
# C5. Smoke test: ssh root@49.13.159.52 "cd /root/polymarket-v2 && ENVIRONMENT=live venv/bin/python3 scripts/live_smoke_test.py"
# C6. Flip prod to live: sed ENVIRONMENT, INITIAL_BANKROLL=500, MAX_POSITION_PCT=0.02
# C7. First 24h watch
```

### Phase C exit criteria (post-live, 50+ trades or 14d)

| Metric | Pass | Tune | Block |
|---|---|---|---|
| Realized PnL | >0% | -3% to 0% | <-3% |
| Brier raw | <0.20 | 0.20-0.25 | >0.25 |
| Cash drift vs on-chain | 0% | <1% | >1% |
| Catastrophic SL fires | 0% | <5% | >5% |

---

## Priority 6 — Future improvements (post-Phase C)

### Twitter A/B (dev env)

`TWITTER_ENABLED=false` in prod. Hypothesis from prior session: net-negative on paper. Re-test via parallel dev env (`polymarket-dev.service` on port 8003, separate DB at `data-dev/`). Plan archived below.

### RSS — VPS IP block

14 feeds 403 from VPS (ESPN ×6, ap_top, dotesports, euractiv, euronews, europarl, polygon, sec_press, politico). Solutions require proxy or cloudscraper — out of scope. See [project_vps_rss_ipblock](../.claude/projects/-home-jedicelli-polymarket-v2/memory/project_vps_rss_ipblock.md). 30 working feeds = 263 signals/poll baseline.

### Cost levers (defer)

- Cheaper pre-screen model. MiMo may offer tiered models; investigate when budget binds.

### Ensemble LLM on high-stakes (defer)

Post-Phase-D when n≥50 per market type.

### `MIMO_API_KEY` cost calibration

`api_costs.cost_usd` shows $0 for `mimo` service rows — pricing constants not updated for xiaomimimo. Low priority; affects DAILY_API_BUDGET_USD gate. Update when budget enforcement matters.

---

## Parallel dev env (deferred until live)

Same architecture as before:

| Slot | Service | Port | DB | .env |
|---|---|---|---|---|
| Prod (live) | `polymarket.service` | 8000 | `data/predictor.db` | `/root/polymarket-v2/.env` |
| Dev (paper) | `polymarket-dev.service` | 8003 | `data-dev/predictor.db` | `/root/polymarket-v2/.env.dev` |

Needs `HTTP_PORT` env reading (small PR). Build after Phase C.

---

## Closed items (this session, 2026-05-19 → 2026-06-08)

| Round | Date | Items |
|---|---|---|
| Cash drift root cause + fix | 2026-06-08 | DONE: atomic `save_trade_with_portfolio` (commit `7a45032`). $792 historical drift cleared by full DB wipe. |
| Drift monitoring Layers 1+2 | 2026-06-08 | DONE: schema v9 `drift_history` table, 30min APScheduler gauge, `cash_mutation` audit log (commit `1a72d05`) |
| LLM provider swap MiniMax → MiMo | 2026-06-05 | DONE: configurable `LLM_BASE_URL` + `MIMO_API_KEY` fallback (commit `25bd784`) |
| `calculate_pnl` formula fix (cost-based) | 2026-06-05 | DONE: `pos*(1-entry)/entry` for BUY_YES wins (commit `8cc1b93`) + reconcile backfill |
| `calculate_early_exit_pnl` fee deduction | 2026-06-05 | DONE: fee on wins (commit `8338c32`) + backfill (commit `572002b`) |
| RSS httpx + feedparser UA | 2026-06-03 | DONE: `feedparser/6.0` UA, fixed 3-4 feeds; ESPN regression resolved via UA hotfix |
| Volume floor $20k | 2026-06-03 | DONE: `MIN_MARKET_VOLUME_24H` cuts 85% of markets pre-LLM |
| `/no-signal-rss-audit` skill | 2026-06-03 | DONE: diagnostic-only skill |
| RSS expansion 29 → 44 feeds | 2026-05-29 | DONE: weather, esports, intl politics, geopolitics, regulatory, cultural |
| Signal cap 7→10 / 3→5 | 2026-05-29 | DONE: prescreen + full eval expanded |
| SL disable + snapshot table | 2026-05-26 | DONE: `STOP_LOSS_ENABLED` flag + `trade_price_snapshots` table + `scripts/sl_analysis.py` (schema v8) |
| TP disable + tp_analysis | 2026-06-05 | DONE: `TAKE_PROFIT_ENABLED` flag + `scripts/tp_analysis.py` |
| Bug 5/6/7/8/9 round | 2026-05-19 → 2026-05-26 | DONE: WS reliability, thin-book guard, observe-only fix, silent stall detection |

---

## Long-term reminder

This system is paper-tested but **not yet validated in live**. First 50 live trades are the real validation. During paper soak: do not add features unless they're monitoring/diagnostics. Treat drift, cash math, label coverage as load-bearing invariants.

After 100 live trades and 4 weeks profitable → ensemble LLM, biweekly loop, dashboard, web_search, critical learning section.

---

## Prediction Arena comparison (icebox)

- Add `web_search()` tool to LLM context (single call per market at full-eval). MiMo may support tool-use API.
- Replicate "critical learning section" — surface 5 recent losing trades + reasoning in next prompt.
