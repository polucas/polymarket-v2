---
name: vps-health-check
description: One-shot VPS status check for the Polymarket v2 bot — systemd unit, log tail, /health, 24h trade counts.
trigger: vps health, vps status, check vps, daily vps, /vps-health-check
---

## Purpose

Single-command VPS health snapshot used each morning or after any deploy. Confirms the bot is alive, the scheduler is ticking, and trades are flowing (or flags stagnation).

## Inputs

None.

## Steps

1. Systemd unit state + uptime:
   ```bash
   ssh root@49.13.159.52 'systemctl is-active polymarket && systemctl status polymarket --no-pager | head -30'
   ```
2. Log tail (last 50 lines):
   ```bash
   ssh root@49.13.159.52 'tail -50 /root/polymarket-v2/data/bot.log'
   ```
3. Health endpoint:
   ```bash
   ssh root@49.13.159.52 'curl -s http://localhost:8000/health | python3 -m json.tool'
   ```
4. 24h action counts:
   ```bash
   ssh root@49.13.159.52 "cd /root/polymarket-v2 && sqlite3 data/predictor.db \"SELECT action, COUNT(*) FROM trade_records WHERE timestamp > datetime('now','-24 hours') GROUP BY action;\""
   ```

## Output format

Produce a single markdown report with four sections:

- **Service** — active/inactive/failed + uptime (parsed from `systemctl status`).
- **Log tail** — last 50 lines in a fenced block; if only routine entries, collapse to a 1-line summary ("50 lines, no errors") and keep only ERROR/WARN lines.
- **Health endpoint** — pretty-printed JSON. Highlight `status != "ok"` (the endpoint returns `"ok"` or `"stale"`) or `minutes_since_scan > 30`. Note `ws_exit.connected` explicitly.
- **24h activity** — table of BUY_YES / BUY_NO / SKIP counts + `execution_rate = non_SKIP / total` as a percent.

End with a one-line verdict: `GREEN` (all healthy), `YELLOW` (stale scan or zero executions in 24h), or `RED` (service inactive or SSH failed).

## Error handling

- SSH timeout / unreachable → report "VPS unreachable" with the last successful timestamp if known, halt remaining steps.
- Systemd `inactive (dead)` or `failed` → flag RED and still pull log tail + DB counts (local file reads on VPS may still work via SSH even if service is down).
- Log tail empty → flag "no recent activity, scheduler may be frozen".
- Health returns HTTP 503 (stale) → keep going, highlight in output.
- `sqlite3` errors (locked DB) → retry once after 2 seconds.

## Related

- `deploy-update` — run this skill after any deploy to verify the service came back.
- `db-diagnostics` — drill into the 24h counts (e.g., skip-breakdown recipe) if execution rate looks wrong.
- `skip-reason-analyzer` — invoke when SKIP dominates the 24h table.
