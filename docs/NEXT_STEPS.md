# Next Steps — Polymarket v2

§1j Phase 1 implementation in progress 2026-04-28.

_Last updated: 2026-04-28 — Round 2 window expired, §0b/§1d/§1h/§1i removed, §1j plan written (implementing next), §2 unblocked but still waiting on §1j._

## Monitoring summary (what to watch, at a glance)

| Window | Signal | Action on hit |
|---|---|---|
| Daily | `/morning-check` output | Named env-only lever per alarm row (see skill) |
| Weekly (Mondays) | 7-day vs prior 7-day trade count / PnL | WR/Brier unmeasurable until §1j lands — use PnL + trade count as proxy |
| Ongoing | `weak_signals_*` total in `/morning-check` | Structural at 929/24h; alarm if drops below 5/day (gate broken) or no trades for 48h |

---

## 1. Open — IMMEDIATE PRIORITY

### 1j. Dual-label calibration — PLAN WRITTEN, NOT IMPLEMENTED

**Problem:** 74 trades, zero calibration data. Every trade exits via WS TP/SL before natural resolution. `actual_outcome` never written. Learning system (calibration, Brier, signal tracker, adjustment pipeline) has been blind since 2026-04-16.

**Root cause confirmed (D1–D8 diagnostics 2026-04-25):**
- `ws_exit.py` writes `exit_type`/`pnl` but not `actual_outcome`, no `on_trade_resolved()` call.
- `get_open_trades()` filters `AND exit_type IS NULL` → TP/SL-exited trades permanently excluded from resolution scanner.
- All calibration bucket alphas = 1.0, betas = 1.0 (initialization priors, never updated).
- D6 Gamma API cross-check: synthetic label accuracy 50% overall (coin flip). BUY_NO+TP → 0/3 accuracy. Only BUY_YES+SL is defensible.

**Decision: dual-label approach**
- `trade_profitable` (pnl > 0) — fast label, available on every exit, used for interim calibration/Brier.
- `actual_outcome` — Polymarket ground truth, written by `resolution.py` when market resolves. Remove `AND exit_type IS NULL` filter so scanner continues checking post-TP/SL trades.
- Both labels coexist. `trade_profitable` gives signal immediately; `actual_outcome` provides ground truth when available.

**Plan:** `/home/jedicelli/.claude/plans/dual-label-calibration.md` — full Phase 1 + Phase 2 with all downstream changes (DB migration, ws_exit, resolution, sqlite, calibration, Brier, signal tracker, adjustment, self_check, alerts, datasette metadata, docs).

**Next action:** implement Phase 1 (schema + exit writes + filter removal + pnl-based metrics). No env changes needed.

---

## 2. Profitability Levers (unblocked since 2026-04-25, but §1j still first)

### 2a. Cost
- Tighten `PRESCREEN_MIN_CONFIDENCE` 0.25 → 0.30. Expected ~$2/mo savings.
- **Cheaper pre-screen model** (MiniMax-M2.5 prescreen, M2.7 full eval). Expected $14/mo → ~$12/mo. Combine with raising `PRESCREEN_MIN_EDGE` 0.03 → 0.05.
  - Implementation: add `PRESCREEN_LLM_MODEL` setting in `src/config.py`, route `call_prescreen()` in `grok_client.py` to use it.

### 2b. Quality
- ~~Per-market-type `MIN_EDGE`~~ — **done Round 4 (2026-04-19)**, confirmed 2026-04-24.
- Ensemble LLM on positions >$100: 2 calls, skip on disagreement >0.15. Needs calibration data first (§1j).

### 2c. Volume
- Loosen `QUESTION_SIMILARITY_THRESHOLD` 0.60 → 0.65 if false-positives confirmed by skip-reason samples.

### Recommended order
1. ~~Per-market-type MIN_EDGE~~ — done.
2. **§1j dual-label calibration** — implement now. Blocking all learning.
3. Cost levers (§2a) — after §1j deployed and 3-day signal confirmed.
4. Ensemble on high-stakes (§2b) — needs calibration data accumulating first.

---

## 3. Biweekly Improvement Loop (Deferred)

~2–3 days dev. Lands after §1j + §2 settled.

- Biweekly cadence (14d), manual approval gate, shadow-mode A/B.
- New DB tables: `hypotheses`, `biweekly_reviews`.
- CLI: `biweekly_review`, `propose_hypothesis`, `approve_hypothesis`, `merge_hypothesis`.
- Endpoint: `GET /experiments/compare?a=<run>&b=<run>` — Welch's t-test + Mann-Whitney.
- Safety: param whitelist, rollback on 3-day `daily_loss_limit` breach.

---

## 4. UI Dashboard (lowest priority)

Replaces raw Datasette browsing. Views: portfolio, trade history, calibration buckets, skip-reason breakdown, Brier by market_type, daily review archive. FastAPI+Jinja+HTMX OR Next.js/SvelteKit. Token/IP auth, stays on VPS.

**Prereq:** §1j + §2 + §3 settled first.
