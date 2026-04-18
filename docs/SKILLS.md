# Claude Skills — Quick Reference

Project-scoped skills live in [`../.claude/skills/`](../.claude/skills/). Invoke by trigger phrase or `/<skill-name>`. Read-only unless noted.

| # | Skill | Trigger | Input | Mutates? | One-line use |
|---|---|---|---|---|---|
| 1 | [vps-health-check](../.claude/skills/vps-health-check.md) | `/vps-health-check` | — | no | Morning status: systemd, log tail, `/health`, 24h counts. |
| 2 | [db-diagnostics](../.claude/skills/db-diagnostics.md) | `/db-diagnostics <recipe> [hours]` | recipe + hours | no | SQL recipes: `skip-breakdown`, `cash-reconcile`, `prescreen-failures`, `open-positions`. |
| 3 | [prescreen-debug](../.claude/skills/prescreen-debug.md) | `/prescreen-debug <market_id>` | market_id | no | Pull raw MiniMax response, classify parse failure, suggest fix. |
| 4 | [doc-audit](../.claude/skills/doc-audit.md) | `/doc-audit` | — | no | Flag drift across `.env`, `config.py`, `CLAUDE.md`, `polymarket_system_v2.md`. |
| 5 | [backtest-compare](../.claude/skills/backtest-compare.md) | `/backtest-compare <start> <end> <KEY=VAL> [max]` | dates + override | runs backtest | Baseline vs override — WR/ROI/Brier + Welch's t-test. |
| 6 | [deploy-update](../.claude/skills/deploy-update.md) | `/deploy-update [branch]` | branch | **yes — git push, VPS restart** | Push → VPS sync → pip install → restart → verify. Confirmation-gated. |
| 7 | [skip-reason-analyzer](../.claude/skills/skip-reason-analyzer.md) | `/skip-reason-analyzer [hours]` | hours | no | Top 3 skip reasons + 5 samples each + fix category. |
| 8 | [learning-status](../.claude/skills/learning-status.md) | `/learning-status` | — | no | Calibration + market-type + signal-tracker snapshot. |
| 9 | [trade-replay](../.claude/skills/trade-replay.md) | `/trade-replay market_id=X` or `record_id=X` | market_id or record_id | no | Signal → LLM → adjustment → decision trail for one trade. |
| 10 | [live-readiness](../.claude/skills/live-readiness.md) | `/live-readiness` | — | **yes — on-chain allowance + $0.01 order** | Preflight before flipping `ENVIRONMENT=live`. Gated twice. |

## Test order

Safe first:

1. `/vps-health-check` — fastest smoke test, proves SSH works.
2. `/db-diagnostics skip-breakdown 48` — confirms DB query + SSH round-trip.
3. `/prescreen-debug 1994998` — one of top 8 recurring offenders from [NEXT_STEPS.md §0b](./NEXT_STEPS.md).
4. `/doc-audit` — purely local, no VPS needed. Likely surfaces one drift (`PRESCREEN_MIN_CONFIDENCE`).
5. `/skip-reason-analyzer 48`
6. `/learning-status`
7. `/trade-replay market_id=<id from #3 or #5>`

Mutation skills — only when the real task requires them:

8. `/backtest-compare` — long-running, needs ingested data.
9. `/deploy-update` — only on real deploy.
10. `/live-readiness` — only when flipping to live trading.

## Fixing a broken skill

Each skill has an `## Unresolved` section at the bottom listing schema assumptions. If a step fails on first invocation:

1. Read the failing command output.
2. Edit the skill `.md` file directly. Correct SQL columns, flag names, or paths.
3. Re-invoke. No restart needed — skills reload per invocation.

## Conventions

- VPS: `root@49.13.159.52`, project path `/root/polymarket-v2`, systemd unit `polymarket`, bot port 8000.
- Positional args: `$1`, `$2`, etc.
- Mutation steps always print **CONFIRM BEFORE:** and wait for user approval.
