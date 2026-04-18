# Polymarket v2 — Claude Skills

Reusable skills for recurring workflows on this bot. Invoke by name (e.g. `/vps-health-check`) or via any of the trigger phrases listed in each file's frontmatter.

Each skill follows the same template: `Purpose → Inputs → Steps → Output format → Error handling → Related`. Read-only by default; mutations (git push, systemctl restart, on-chain calls) are gated behind explicit user confirmation.

## Priority order

The first three are the daily / per-investigation workhorses; the rest are tuned for specific scenarios.

| # | Skill | Cadence | When to run |
|---|---|---|---|
| 1 | [vps-health-check](./vps-health-check.md) | daily | Morning status: systemd, log tail, `/health`, 24h trade counts. |
| 2 | [db-diagnostics](./db-diagnostics.md) | every investigation | Four SQL recipes: skip-breakdown, cash-reconcile, prescreen-failures, open-positions. |
| 3 | [prescreen-debug](./prescreen-debug.md) | on demand | `prescreen_parse_failed` on a specific `market_id` — classify + suggest fix. |
| 4 | [doc-audit](./doc-audit.md) | biweekly / pre-commit | Cross-ref `.env` / `config.py` / `CLAUDE.md` / `polymarket_system_v2.md`. |
| 5 | [backtest-compare](./backtest-compare.md) | biweekly (§2 shadow) | Baseline vs `KEY=VALUE` override — WR/ROI/Brier delta + Welch's t-test. |
| 6 | [deploy-update](./deploy-update.md) | each deploy | Push → VPS reset → pip install → restart → verify. |
| 7 | [skip-reason-analyzer](./skip-reason-analyzer.md) | weekly / on demand | Top 3 skip reasons + 5-market sample + fix category. |
| 8 | [learning-status](./learning-status.md) | before tuning | Calibration / market-type / signal-tracker snapshot. |
| 9 | [trade-replay](./trade-replay.md) | debug a market | Full signal → LLM → adjustment → decision trail for one record. |
| 10 | [live-readiness](./live-readiness.md) | pre go-live | Checklist before flipping `ENVIRONMENT=paper → live`. |

## Conventions

- VPS: `root@49.13.159.52`, project path `/root/polymarket-v2`, systemd unit `polymarket`, bot port 8000.
- DB paths: live `data/predictor.db`, backtest `data/backtest_data.db` / `data/backtest_outputs.db`.
- Args are positional: `$1`, `$2`, ... — see each skill's **Inputs** section.
- Any skill reporting schema drift should be resolved by updating the skill file, not by guessing in prod.
