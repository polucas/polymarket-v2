from __future__ import annotations

from typing import List

import httpx
import structlog

from src.config import Settings
from src.models import Portfolio, TradeRecord

log = structlog.get_logger()

TELEGRAM_API = "https://api.telegram.org"


async def send_alert(message: str, settings: Settings) -> None:
    """POST to Telegram. No-op if TELEGRAM_BOT_TOKEN is empty."""
    if not settings.TELEGRAM_BOT_TOKEN or not settings.TELEGRAM_CHAT_ID:
        return
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            await client.post(
                f"{TELEGRAM_API}/bot{settings.TELEGRAM_BOT_TOKEN}/sendMessage",
                json={
                    "chat_id": settings.TELEGRAM_CHAT_ID,
                    "text": message,
                    "parse_mode": "HTML",
                },
            )
    except Exception as e:
        log.warning("telegram_alert_failed", error=str(e))


def format_trade_alert(record: TradeRecord) -> str:
    """Format a trade execution alert."""
    emoji = "BUY" if record.action != "SKIP" else "SKIP"
    return (
        f"<b>{emoji}: {record.market_question[:80]}</b>\n"
        f"Side: {record.action} | Edge: {record.calculated_edge:.3f}\n"
        f"Size: ${record.position_size_usd:.2f} | Price: {record.market_price_at_decision:.4f}\n"
        f"Prob: {record.final_adjusted_probability:.3f} (raw: {record.grok_raw_probability:.3f})\n"
        f"Conf: {record.final_adjusted_confidence:.3f} | Score: {record.trade_score:.4f}\n"
        f"Tier: {record.tier} | Type: {record.market_type}"
    )


def format_daily_summary(trades: List[TradeRecord], portfolio: Portfolio) -> str:
    """Format daily summary alert."""
    executed = [t for t in trades if t.action != "SKIP"]
    skipped = [t for t in trades if t.action == "SKIP"]
    resolved = [t for t in trades if t.actual_outcome is not None]
    total_pnl = sum(t.pnl or 0 for t in resolved)

    return (
        f"<b>Daily Summary</b>\n"
        f"Executed: {len(executed)} | Skipped: {len(skipped)} | Resolved: {len(resolved)}\n"
        f"Day PnL: ${total_pnl:+.2f}\n"
        f"Portfolio: ${portfolio.total_equity:,.2f} (cash: ${portfolio.cash_balance:,.2f})\n"
        f"Drawdown: {portfolio.max_drawdown:.1%} | Open: {len(portfolio.open_positions)}"
    )


def format_error_alert(error: str) -> str:
    """Format error alert."""
    return f"<b>ERROR</b>\n{error[:500]}"


def format_observe_only_alert(executed: int, cap: int) -> str:
    """Format observe-only mode notification."""
    return (
        f"<b>OBSERVE-ONLY</b>\n"
        f"Daily cap hit ({executed}/{cap}). Switching to observe-only mode."
    )


def format_monk_mode_alert(reason: str) -> str:
    """Format Monk Mode trigger alert."""
    labels = {
        "daily_loss_limit": "Daily loss limit reached",
        "weekly_loss_limit": "Weekly loss limit reached",
        "max_total_exposure": "Max total exposure reached",
        "api_budget_exceeded": "Daily API budget exceeded",
    }
    if reason.startswith("consecutive_adverse"):
        parts = reason.split("_")
        count = parts[-1] if parts[-1].isdigit() else "?"
        label = f"Consecutive adverse trades cooldown ({count} in a row)"
    elif "daily_cap" in reason:
        label = "Tier daily cap reached"
    else:
        label = labels.get(reason, reason)
    return (
        f"<b>MONK MODE</b>\n"
        f"{label}\n"
        f"Trade blocked. Waiting for conditions to improve."
    )


def format_tier2_alert(active: bool) -> str:
    """Format Tier 2 activation/deactivation alert."""
    if active:
        return "<b>TIER 2 ACTIVATED</b>\nCrypto news detected. Starting rapid scan cycle."
    return "<b>TIER 2 DEACTIVATED</b>\nNo new crypto signals for 30 min. Stopping Tier 2."


def format_lifecycle_alert(event: str, environment: str) -> str:
    """Format bot lifecycle (startup/shutdown) alert."""
    return f"<b>BOT {event.upper()}</b>\nMode: {environment}"


def format_stale_scan_alert(minutes_since: float) -> str:
    """Format stale scan health warning."""
    return (
        f"<b>STALE SCAN WARNING</b>\n"
        f"No scan completed in {minutes_since:.0f} minutes. Check bot health."
    )
