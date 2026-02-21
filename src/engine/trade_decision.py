from __future__ import annotations

from typing import List, Optional, Tuple

import structlog

from src.config import MonkModeConfig
from src.models import Portfolio, TradeCandidate, TradeRecord

log = structlog.get_logger()


def calculate_edge(adjusted_prob: float, market_price: float, fee_rate: float) -> float:
    """Calculate edge: abs(adjusted_prob - market_price) - fee_rate"""
    return abs(adjusted_prob - market_price) - fee_rate


def determine_side(adjusted_prob: float, market_price: float) -> str:
    """Determine trade side based on probability vs market price."""
    if adjusted_prob > market_price:
        return "BUY_YES"
    elif adjusted_prob < market_price:
        return "BUY_NO"
    return "SKIP"


def kelly_size(
    adjusted_prob: float,
    market_price: float,
    side: str,
    bankroll: float,
    kelly_fraction: float = 0.25,
    max_position_pct: float = 0.08,
) -> float:
    """Kelly criterion for binary markets.
    BUY_YES: f* = (prob - price) / (1 - price)
    BUY_NO:  f* = (price - prob) / price
    Apply quarter Kelly, cap at max_position_pct * bankroll.
    """
    if side == "BUY_YES":
        if adjusted_prob <= market_price:
            return 0.0
        f_star = (adjusted_prob - market_price) / (1 - market_price)
    elif side == "BUY_NO":
        if adjusted_prob >= market_price:
            return 0.0
        f_star = (market_price - adjusted_prob) / market_price
    else:
        return 0.0

    # Apply fractional Kelly
    position = f_star * kelly_fraction * bankroll
    # Cap at max position
    max_position = max_position_pct * bankroll
    return min(position, max_position)


def check_monk_mode(
    config: MonkModeConfig,
    trade_signal: TradeCandidate,
    portfolio: Portfolio,
    today_trades: List[TradeRecord],
    week_trades: List[TradeRecord],
    api_spend: float,
) -> Tuple[bool, Optional[str]]:
    """Check Monk Mode constraints. Returns (allowed, reason_if_blocked)."""

    # 1. Tier daily cap
    tier = trade_signal.tier
    tier_trades_today = [t for t in today_trades if t.tier == tier and t.action != "SKIP"]
    cap = config.tier1_daily_trade_cap if tier == 1 else config.tier2_daily_trade_cap
    if len(tier_trades_today) >= cap:
        return False, f"tier{tier}_daily_cap_reached"

    # 2. Daily loss limit (-5%)
    today_pnl = sum(t.pnl or 0 for t in today_trades if t.pnl is not None)
    if portfolio.total_equity > 0 and today_pnl / portfolio.total_equity < -config.daily_loss_limit_pct:
        return False, "daily_loss_limit"

    # 3. Weekly loss limit (-10%)
    week_pnl = sum(t.pnl or 0 for t in week_trades if t.pnl is not None)
    if portfolio.total_equity > 0 and week_pnl / portfolio.total_equity < -config.weekly_loss_limit_pct:
        return False, "weekly_loss_limit"

    # 4. Consecutive adverse (3 losses including unrealized adverse moves >10%)
    recent_trades = sorted(
        [t for t in today_trades if t.action != "SKIP"],
        key=lambda t: t.timestamp,
        reverse=True,
    )
    consecutive_adverse = 0
    for t in recent_trades[:config.consecutive_loss_cooldown + 2]:
        is_adverse = (t.pnl is not None and t.pnl < 0) or (
            t.unrealized_adverse_move is not None and t.unrealized_adverse_move > 0.10
        )
        if is_adverse:
            consecutive_adverse += 1
        else:
            break
    if consecutive_adverse >= config.consecutive_loss_cooldown:
        return False, f"consecutive_adverse_{consecutive_adverse}"

    # 5. Max total exposure (30%)
    total_exposure = sum(p.size_usd for p in portfolio.open_positions)
    if portfolio.total_equity > 0 and (total_exposure + trade_signal.position_size) / portfolio.total_equity > config.max_total_exposure_pct:
        return False, "max_total_exposure"

    # 6. API budget ($8/day)
    if api_spend >= config.daily_api_budget_usd:
        return False, "api_budget_exceeded"

    return True, None


def get_scan_mode(today_trades: List[TradeRecord], config: MonkModeConfig) -> str:
    """Return 'observe_only' if tier1 executed trades >= cap, else 'active'."""
    tier1_executed = [t for t in today_trades if t.tier == 1 and t.action != "SKIP"]
    if len(tier1_executed) >= config.tier1_daily_trade_cap:
        return "observe_only"
    return "active"
