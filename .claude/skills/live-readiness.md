---
name: live-readiness
description: Preflight checklist before flipping ENVIRONMENT=paper → live on the VPS. Gates every mutation behind explicit user confirmation.
trigger: live readiness, pre go-live, live preflight, /live-readiness
---

## Purpose

Required safety checklist before any transition to real-money trading, especially post the April 6, 2026 Polymarket exchange upgrade. Read-only checks run first; real-money mutations (allowance + smoke order) are gated on explicit user approval and cannot run if any cred is missing.

## Inputs

None.

## Steps

1. Verify the VPS is still in paper mode:
   ```bash
   ssh root@49.13.159.52 "grep ^ENVIRONMENT /root/polymarket-v2/.env"
   ```
   Must print `ENVIRONMENT=paper`. If it already reads `live`, STOP and report.
2. Confirm all four wallet creds are set (redact values):
   ```bash
   ssh root@49.13.159.52 "grep -E '^POLYMARKET_(PRIVATE_KEY|FUNDER_ADDRESS|SECRET|PASSPHRASE)=' /root/polymarket-v2/.env | sed 's/=.*/=<set>/'"
   ```
   Required: `POLYMARKET_PRIVATE_KEY`, `POLYMARKET_SECRET`, `POLYMARKET_PASSPHRASE`. Optional (EOA can skip): `POLYMARKET_FUNDER_ADDRESS`. Missing any required → HALT.
3. `py-clob-client` version (must be `~=0.34.0` post April 6 upgrade):
   ```bash
   ssh root@49.13.159.52 "cd /root/polymarket-v2 && venv/bin/pip show py-clob-client | grep Version"
   ```
4. Recent paper-mode performance (reuse the raw-Brier table from `learning-status` step 4):
   ```bash
   ssh root@49.13.159.52 "cd /root/polymarket-v2 && sqlite3 data/predictor.db \"SELECT market_type, COUNT(*), AVG((final_adjusted_probability - actual_outcome)*(final_adjusted_probability - actual_outcome)) FROM trade_records WHERE action!='SKIP' AND actual_outcome IS NOT NULL GROUP BY market_type;\""
   ```
   Flag any market_type with Brier > 0.25.
5. Fee sanity check:
   ```bash
   rg -n 'MARKET_TYPE_FEES' src/pipelines/market_classifier.py
   ```
   Confirm values in the dict match: crypto_15m ≈ 0.0156, sports/esports ≈ 0.02, political/geopolitical/economic/cultural/regulatory/weather = 0.0.
6. **STOP — user confirmation gate 1.** Prompt: "About to run `scripts/setup_clob_allowance.py` (one-time USDC allowance approval on-chain) and then `scripts/live_smoke_test.py` (places a real $0.01 limit order far from market). Continue?" If the user says no, end the skill.
7. Run allowance setup (no-op after the first successful run):
   ```bash
   ssh root@49.13.159.52 "cd /root/polymarket-v2 && venv/bin/python scripts/setup_clob_allowance.py"
   ```
8. Run smoke test. **Remind user verbally: it does NOT auto-cancel — they must manually cancel the order on [https://polymarket.com](https://polymarket.com) after this step.**
   ```bash
   ssh root@49.13.159.52 "cd /root/polymarket-v2 && venv/bin/python scripts/live_smoke_test.py"
   ```
   Non-zero exit → HALT, do not suggest the final flip.
9. **STOP — user confirmation gate 2.** Prompt: "Smoke test clean. Flip `ENVIRONMENT=paper → live` in `/root/polymarket-v2/.env` and run `deploy-update`?" End of skill; do not flip the env file automatically.

## Output format

A 9-row checklist table: `step | status (✓/✗/pending) | detail`. End with one of:

- `READY — awaiting user gate to flip env` (all steps 1-8 green).
- `NOT READY: <reason>` (any step failed or a cred is missing).

## Error handling

- Any required cred missing → HALT at step 2, refuse to proceed, list what's missing.
- `py-clob-client` version mismatch → HALT, ask user to update `requirements.txt` and run `deploy-update`.
- `scripts/setup_clob_allowance.py` non-zero → print stderr, HALT — do not run the smoke test.
- `scripts/live_smoke_test.py` non-zero → print stderr + transaction hash if any, HALT.
- User says no at either gate → exit gracefully with the state of the checklist.

## Related

- `deploy-update` — required follow-up after the user flips the env.
- `learning-status` — deeper look at paper-mode performance before committing to live.
- `vps-health-check` — run first to confirm the bot is healthy before touching credentials.
