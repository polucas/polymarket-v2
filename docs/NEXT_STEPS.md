# Next Steps — Polymarket v2

_Last updated: 2026-05-20 — Phase A acceleration active. DB reset clean. Bug 5/6 round shipped. Pagination fix shipped (10x market volume). Live launch (Phase C) blocked only on L2 API creds + funding._

## Current state

- **Mode:** paper, `ENVIRONMENT=paper`
- **DB:** fresh as of 2026-05-19 (full wipe, schema v7, $10K bankroll)
- **Phase A loosened gates active (2026-05-20):**
  - `DISABLED_MARKET_TYPES=sports,esports` (economic re-enabled after Bug 5/6 round; esports added 2026-05-20 as conceptually equivalent to sports)
  - `PRESCREEN_MIN_EDGE=0.025` (was 0.05)
  - `PRESCREEN_MIN_CONFIDENCE=0.25` (was 0.30)
  - `MIN_HOURS_TO_RESOLUTION=0.25` (was 0.5)
  - `TWITTER_ENABLED=false` (cold start, will A/B in dev env later)
  - `FAST_EXIT_POLL_INTERVAL_SECONDS=60`, `WS_HEARTBEAT_SECONDS=10` (Bug 6 fixes)
- **Pagination fix (2026-05-20):** `MARKET_PAGE_SIZE=100`, `MARKET_FETCH_PAGES=10` — Polymarket Gamma API silently caps `limit` at 100/page, so previous defaults (500/2) were fetching 100 markets/scan instead of 1000. After fix: ~1000 markets/scan = 10x candidate pool.
- **Service:** `active + enabled`, paper

## Monitoring summary

| Window | Signal | Action |
|---|---|---|
| Daily | `/morning-check` | Use named env-only lever per alarm row |
| Daily | `cash_balance` vs computed | DRIFT > $50 → halt + investigate (Bug 5 territory) |
| Daily | Label coverage (`pnl_labeled == total_non_skip`) | Mismatch → exit-path label regression |
| Phase A 5-day cumulative | `executed_trades`, sum_pnl, Brier raw | <20 trades → loosen more; bad PnL → fix before live |

---

## Phase A: paper acceleration (in flight, 2026-05-20 → 2026-05-25)

**Goal:** 20+ executed trades, real Brier signal, validate Bug 5/6 fixes under load.

### Watch list

```bash
# Daily morning
/morning-check

# Quick volume check
ssh root@49.13.159.52 "cd /root/polymarket-v2 && sqlite3 data/predictor.db \"SELECT action, COUNT(*) FROM trade_records WHERE timestamp > datetime('now','-24 hours') GROUP BY action;\""

# WS health
ssh root@49.13.159.52 "grep -c ws_exit_connected /root/polymarket-v2/data/bot.log"
```

### Phase A exit criteria (Day 5, 2026-05-25)

| Metric | Pass | Tune | Block |
|---|---|---|---|
| Executed trades | ≥20 | 5-19 | 0-4 |
| Brier raw | <0.20 | 0.20-0.25 | >0.25 |
| Cash drift | $0 | <$10 | >$50 |
| WS slippage (exits at 0.01) | 0% | <10% | >20% |

- **Pass:** proceed to Phase B/C
- **Tune:** further loosen (`PRESCREEN_MIN_EDGE` 0.025 → 0.015) or wait 3 more days
- **Block:** identify specific bug, fix, restart Phase A

---

## Phase B: decision gate (2026-05-25)

Pull metrics. Decide between:

1. **Go live** → Phase C
2. **Iterate paper** → tune one parameter, +3 days
3. **Bug found** → fix + restart Phase A

---

## Phase C: live launch ($500 starting bankroll, 2026-05-25+)

**Risk-scaled launch.** $500 = 1/20th of paper bankroll. Real-money signal at low blast radius.

### Pre-flight checklist (do these IN PARALLEL with Phase A soak)

#### C1. Fix `POLYMARKET_FUNDER_ADDRESS`

Current value is len=125, should be 42 (`0x` + 40 hex) or empty for EOA.

```bash
ssh root@49.13.159.52 "grep '^POLYMARKET_FUNDER_ADDRESS=' /root/polymarket-v2/.env"
# Inspect output. If wrong format, fix:
ssh root@49.13.159.52 "sed -i 's|^POLYMARKET_FUNDER_ADDRESS=.*|POLYMARKET_FUNDER_ADDRESS=0xYOUR_PROXY_OR_EMPTY|' /root/polymarket-v2/.env"
```

#### C2. Generate L2 API creds (interactive)

Three vars currently empty: `POLYMARKET_API_KEY`, `POLYMARKET_SECRET`, `POLYMARKET_PASSPHRASE`. Required for `ClobClient` Level-2 auth.

**Option A — Polymarket web UI** (recommended):
1. Visit polymarket.com → connect wallet → API keys section → Create new
2. Copy `key`, `secret`, `passphrase`
3. Append to `.env`:

```bash
ssh root@49.13.159.52 "cat >> /root/polymarket-v2/.env <<'EOF'

# L2 API credentials (2026-05-25)
POLYMARKET_API_KEY=<api_key>
POLYMARKET_SECRET=<secret>
POLYMARKET_PASSPHRASE=<passphrase>
EOF"
```

**Option B — Bootstrap via ClobClient.create_api_key()** (if UI unavailable):
```python
# One-off Python REPL on VPS with L1 key set:
from py_clob_client.client import ClobClient
c = ClobClient(host="https://clob.polymarket.com", chain_id=137, key=os.environ["POLYMARKET_PRIVATE_KEY"])
creds = c.create_or_derive_api_key()
print(creds.api_key, creds.api_secret, creds.api_passphrase)
# Copy outputs to .env
```

#### C3. Fund Polymarket wallet

1. Get wallet address:
```bash
ssh root@49.13.159.52 "cd /root/polymarket-v2 && venv/bin/python3 -c \"
from py_clob_client.client import ClobClient
import os
from dotenv import load_dotenv; load_dotenv('/root/polymarket-v2/.env')
c = ClobClient(host='https://clob.polymarket.com', chain_id=137, key=os.environ['POLYMARKET_PRIVATE_KEY'])
print('wallet:', c.get_address())
\""
```
2. Send **USDC.e on Polygon** to that address. Amount: $500 (live launch) + ~$5 in MATIC for gas if EOA.
3. Wait for confirmations.

#### C4. Run CLOB allowance setup

One-time approval for CLOB contract to spend USDC.

```bash
ssh root@49.13.159.52 "cd /root/polymarket-v2 && venv/bin/python3 scripts/setup_clob_allowance.py"
```

Expect output:
- `Wallet address: 0x...`
- `USDC contract:  0x...`
- `Response: <tx hash>`
- `Balance: 500.0` (or whatever was funded)

If balance shows 0 → funding didn't arrive yet, retry after 1 min.

#### C5. Run live smoke test

Places a $0.01 limit order far from market price (won't fill). Validates signing + API flow.

```bash
ssh root@49.13.159.52 "cd /root/polymarket-v2 && ENVIRONMENT=live venv/bin/python3 scripts/live_smoke_test.py"
```

Expect: `OK — smoke test passed.` + order ID printed.

**Manually cancel that test order on polymarket.com.** Currently no auto-cancel in the script.

#### C6. Flip prod to live

Switch service to live mode + reduce bankroll for risk-scaled launch.

```bash
ssh root@49.13.159.52 "systemctl stop polymarket && \
  sed -i 's/^ENVIRONMENT=.*/ENVIRONMENT=live/' /root/polymarket-v2/.env && \
  sed -i 's/^INITIAL_BANKROLL=.*/INITIAL_BANKROLL=500.0/' /root/polymarket-v2/.env && \
  cat >> /root/polymarket-v2/.env <<'EOF'

# Phase C launch overrides (2026-05-25)
MAX_POSITION_PCT=0.02
DAILY_LOSS_LIMIT_PCT=0.10
WEEKLY_LOSS_LIMIT_PCT=0.20
EOF
  systemctl start polymarket && sleep 8 && curl -s http://localhost:8000/health | python3 -m json.tool"
```

Confirms:
- `mode: live`
- `open_trades: 0`
- `uptime_hours > 0`

#### C7. First 24h watch

```bash
# Every 4h
ssh root@49.13.159.52 "curl -s http://localhost:8000/health"

# Real wallet balance check
ssh root@49.13.159.52 "cd /root/polymarket-v2 && venv/bin/python3 -c \"
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds, BalanceAllowanceParams, AssetType
import os; from dotenv import load_dotenv; load_dotenv('/root/polymarket-v2/.env')
c = ClobClient(host='https://clob.polymarket.com', chain_id=137,
  key=os.environ['POLYMARKET_PRIVATE_KEY'],
  creds=ApiCreds(api_key=os.environ['POLYMARKET_API_KEY'],
                 api_secret=os.environ['POLYMARKET_SECRET'],
                 api_passphrase=os.environ['POLYMARKET_PASSPHRASE']))
print(c.get_balance_allowance(BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)))
\""
```

**Kill switch:** if anything looks wrong (drift, error spam, unexpected positions), immediately:
```bash
ssh root@49.13.159.52 "systemctl stop polymarket && systemctl disable polymarket"
```

### Phase C exit criteria (after 50+ live trades or 14 days, whichever first)

| Metric | Pass | Tune | Block |
|---|---|---|---|
| Realized PnL | >0% | -3% to 0% | <-3% |
| Brier raw | <0.20 | 0.20-0.25 | >0.25 |
| Cash drift on-chain vs reported | 0% | <1% | >1% |
| Catastrophic SL fires (exit ~0.01) | 0% | <5% | >5% |

**Pass → Phase D scale-up.**

---

## Phase D: scale-up (Day 21+ live, ~2026-06-15)

- $500 → $2,000 (4x bankroll bump)
- Re-evaluate twitter via dev env (see below)
- Then $2K → $10K once 100+ live trades validated positive

---

## Parallel dev env on same VPS

**Goal:** test Twitter A/B, prompt changes, new gates without risking prod.

### Architecture

| Slot | Service | Port | DB | .env |
|---|---|---|---|---|
| Prod (live) | `polymarket.service` | 8000 | `data/predictor.db` | `/root/polymarket-v2/.env` |
| Dev (paper) | `polymarket-dev.service` | 8003 | `data-dev/predictor.db` | `/root/polymarket-v2/.env.dev` |
| Datasette | `datasette.service` | 8001 | mounts both | (default config) |

### Setup steps

```bash
# 1. Create dev workspace
ssh root@49.13.159.52 "mkdir -p /root/polymarket-dev && \
  cp -r /root/polymarket-v2/{src,scripts,config,tests,requirements.txt,metadata.json} /root/polymarket-dev/ && \
  cp -r /root/polymarket-v2/venv /root/polymarket-dev/venv && \
  mkdir /root/polymarket-dev/data"

# 2. Build dev .env (paper, separate API budget, twitter on)
ssh root@49.13.159.52 "cp /root/polymarket-v2/.env /root/polymarket-dev/.env && \
  sed -i 's/^ENVIRONMENT=.*/ENVIRONMENT=paper/' /root/polymarket-dev/.env && \
  sed -i 's/^TWITTER_ENABLED=.*/TWITTER_ENABLED=true/' /root/polymarket-dev/.env && \
  sed -i 's/^DAILY_API_BUDGET_USD=.*/DAILY_API_BUDGET_USD=5.0/' /root/polymarket-dev/.env && \
  cat >> /root/polymarket-dev/.env <<'EOF'

# Dev env override (paper sandbox)
DB_PATH=/root/polymarket-dev/data/predictor.db
HTTP_PORT=8003
EOF"
```

(Code change: `src/config.py` needs `DB_PATH` setting + `HTTP_PORT` reading. Confirm what exists today — may need a small PR to make port configurable. If not, run dev on port 8000 with prod off, only useful for one-off tests not parallel.)

```bash
# 3. Build dev systemd unit
ssh root@49.13.159.52 "cat > /etc/systemd/system/polymarket-dev.service <<'EOF'
[Unit]
Description=Polymarket Predictor — DEV sandbox
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/polymarket-dev
EnvironmentFile=/root/polymarket-dev/.env
ExecStart=/root/polymarket-dev/venv/bin/python -m src.main
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload"
```

### Risks + mitigations on shared VPS

| Risk | Mitigation |
|---|---|
| Shared MiniMax API quota | Lower `DAILY_API_BUDGET_USD=5` on dev (was 15) |
| Shared Twitter API (if re-funded) | Use a different `TWITTER_API_KEY` on dev only |
| CPU/RAM contention | Polymarket-v2 is async, low resource. 1GB free is fine for both. Monitor `htop`. |
| Dev crash takes down prod | None — separate systemd units, independent processes |
| Datasette serves wrong DB | Mount both: `--load-extension prod:data/predictor.db --load-extension dev:data-dev/predictor.db` |

### Workflow

1. Code change → push to `main`
2. Test in dev: `systemctl restart polymarket-dev`
3. Monitor `/health` at port 8003 + datasette
4. If clean after N days → enable on prod (`systemctl restart polymarket`)

---

## Closed items (archived)

| Round | Date | Items |
|---|---|---|
| §1j Phase 1 (dual-label) | 2026-04-28 | DONE |
| Bug 5 + 6 round | 2026-05-19 | DONE: `auto_resolve` early-exit guard, fast_exit_check, MIN_HOURS_TO_RESOLUTION, WS heartbeat, sports/economic env-disable, twitter env-disable |
| DB reset to clean state | 2026-05-19 | DONE: fresh schema v7, $10K bankroll, 0 trades |
| Phase A acceleration | 2026-05-20 | ACTIVE — running 5 days |
| Per-market-type MIN_EDGE | 2026-04-19 | DONE (Round 4) |
| Twitter dependency | 2026-05-05 | API depleted, kept paper running on RSS-only. Net positive PnL impact. Twitter re-evaluation deferred to dev env (post-Phase C). |

## Future improvements (post-Phase D scale-up)

### Twitter A/B (in dev env)

Hypothesis: Twitter signals are net-negative based on 19-day sample. Re-test with hybrid gate `TWITTER_MIN_CONFIDENCE_BOOST=0.1` (only execute Twitter-influenced trades if `final_adjusted_confidence >= base + 0.1`).

Run in dev env for 14 days. Compare PnL/Brier to prod (RSS-only). If dev wins → port hybrid gate to prod.

### Cost levers (§2a from old roadmap)

- Cheaper pre-screen model (MiniMax-M2.5 prescreen, M2.7 full eval). Add `PRESCREEN_LLM_MODEL` setting + route `call_prescreen()`. Expected $14/mo → $12/mo.

### Ensemble LLM on high-stakes (§2b)

Requires calibration data. Post-Phase-D when n≥50 per market type.

### Biweekly improvement loop (§3)

Auto-propose hypothesis, manual approval, shadow-mode A/B. Defer until 100+ live trades.

### UI dashboard (§4)

Replace Datasette. Defer. Datasette is sufficient.

---

## Long-term reminder

This system is paper-tested but **not yet validated in live**. First 50 live trades are the real validation. Don't add features during Phase C — only fix bugs.

After 100 live trades and 4 weeks profitable → consider ensemble LLM, biweekly loop, dashboard.

--- 

# Prediciton Arena Comparison

- Disable SL or widen to -25% on settlement-bound trades (PA finding: holders outperform early-exit)
- Add web_search() tool to LLM context (single search per market at full-eval stage). Cheap to add via MiniMax tool-use API if it supports.
- Replicate "critical learning section" — surface 5 recent losing trades + reasoning in next prompt