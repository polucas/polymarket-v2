---
name: deploy-update
description: Push current branch to origin, sync on VPS, restart the polymarket systemd unit, verify startup.
trigger: deploy, deploy update, push to vps, restart bot, /deploy-update
---

## Purpose

Controlled deploy of the Polymarket v2 bot from the dev machine to the VPS. Each destructive step is gated on explicit user confirmation — the skill never pushes, resets, or restarts without a green light.

## Inputs

- `$1` — `branch`, optional. Defaults to current branch from `git rev-parse --abbrev-ref HEAD`.

## Steps

1. Local pre-flight:
   ```bash
   git status --short
   git rev-parse --abbrev-ref HEAD
   git log origin/$1..HEAD --oneline
   ```
   Abort if `git status` shows uncommitted changes.
2. **CONFIRM BEFORE step 3** — show the user the commit list and ask: "Push these commits and deploy to VPS?"
3. Push to origin:
   ```bash
   git push origin $1
   ```
4. **CONFIRM BEFORE step 5** when `$1 == main` — explicit "reset VPS to main?" check.
5. Sync VPS (hard reset — destroys local VPS changes, which should not exist):
   ```bash
   ssh root@49.13.159.52 "cd /root/polymarket-v2 && git fetch && git reset --hard origin/$1"
   ```
6. Install any new deps:
   ```bash
   ssh root@49.13.159.52 "cd /root/polymarket-v2 && source .venv/bin/activate && pip install -r requirements.txt --quiet"
   ```
   If pip fails, STOP — do not restart.
7. **CONFIRM BEFORE step 8** — "Restart polymarket systemd unit?"
8. Restart:
   ```bash
   ssh root@49.13.159.52 "systemctl restart polymarket"
   ```
9. Wait 15 seconds, then tail journal + check health:
   ```bash
   sleep 15
   ssh root@49.13.159.52 "journalctl -u polymarket --since '30 seconds ago' --no-pager | tail -60"
   ssh root@49.13.159.52 "curl -s http://localhost:8000/health"
   ```
10. Ask the user to confirm a Telegram startup alert fired (this skill cannot read Telegram).

## Output format

Markdown with six sections:

- **Commits shipped** — one-line per commit.
- **Pip install** — summary (new packages, if any).
- **Restart result** — systemd return code + first error line if non-zero.
- **Journal tail** — filtered to INFO/WARN/ERROR lines from scheduler startup.
- **Health** — `/health` JSON (must have `status == "ok"` and non-null `last_scan_completed` after one scan interval).
- **Telegram alert** — user-confirmed `YES` / `NO`.

End with a verdict line: `DEPLOY OK` / `DEPLOY FAILED: <reason>`.

## Error handling

- `git status` dirty → STOP, print diff summary, ask user to commit or stash.
- `git push` rejected → STOP, suggest `git pull --rebase` before retrying.
- `pip install` non-zero → STOP before restart, print stderr.
- Restart non-zero → run `journalctl -u polymarket -n 80 --no-pager` and abort.
- `/health` returns 503 or `status: "stale"` → wait 60s more, retry once, then STOP.

## Related

- `vps-health-check` — run immediately after deploy for a fuller snapshot.
- `live-readiness` — required before the first deploy with `ENVIRONMENT=live`.

## Unresolved

- `format_lifecycle_alert("STARTED", env)` in [src/alerts.py:106-108](../../src/alerts.py#L106-L108) renders `"BOT STARTED\nMode: ..."` for Telegram, but the bot uses structlog JSON for stdout — the literal string `BOT STARTED` may not appear in `journalctl`. The skill verifies startup via `/health` + user-confirmed Telegram, not a log grep.
