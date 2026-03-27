from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

import structlog

from src.config import MonkModeConfig
from src.models import Portfolio, TradeCandidate, TradeRecord

if TYPE_CHECKING:
    from src.models import OrderBook, OrderBookLevel

log = structlog.get_logger()


def calculate_edge(adjusted_prob: float, market_price: float, fee_rate: float) -> float:
    """Calculate edge: abs(adjusted_prob - market_price) - fee_rate"""
    return abs(adjusted_prob - market_price) - fee_rate


def calculate_spread_adjusted_edge(
    adjusted_prob: float,
    market_price: float,
    fee_rate: float,
    side: str,
    best_bid: Optional[float] = None,
    best_ask: Optional[float] = None,
) -> float:
    """Edge using executable orderbook prices instead of midpoint.
    BUY_YES: edge vs best_ask (what you'd actually pay).
    BUY_NO: edge vs best_bid (determines NO price you'd pay).
    Falls back to market_price if orderbook is empty.
    """
    if side == "BUY_YES" and best_ask is not None:
        effective_price = best_ask
    elif side == "BUY_NO" and best_bid is not None:
        effective_price = best_bid
    else:
        effective_price = market_price
    return abs(adjusted_prob - effective_price) - fee_rate


def compute_vwap(
    levels: List[OrderBookLevel],
    target_size_usd: float,
) -> Tuple[float, float]:
    """Walk the orderbook levels, compute VWAP up to target_size_usd.

    For asks: levels should be sorted lowest-price first.
    For bids: levels should be sorted highest-price first.

    Returns (vwap_price, fillable_size_usd).
    VWAP = total_usd_spent / total_shares_bought.
    """
    if not levels or target_size_usd <= 0:
        return 0.0, 0.0

    filled_usd = 0.0
    total_shares = 0.0
    for level in levels:
        level_usd = level.size * level.price
        remaining = target_size_usd - filled_usd
        if remaining <= 0:
            break
        take_usd = min(level_usd, remaining)
        shares = take_usd / level.price if level.price > 0 else 0.0
        total_shares += shares
        filled_usd += take_usd

    vwap = filled_usd / total_shares if total_shares > 0 else 0.0
    return vwap, filled_usd


def kelly_size_vwap(
    adjusted_prob: float,
    market_price: float,
    side: str,
    bankroll: float,
    orderbook: OrderBook,
    kelly_fraction: float = 0.25,
    max_position_pct: float = 0.08,
    fee_rate: float = 0.0,
    min_edge: float = 0.0,
) -> Tuple[float, float]:
    """Kelly size capped by profitable orderbook depth.

    Returns (position_size_usd, vwap_price).
    Falls back to standard kelly_size if orderbook is empty.
    """
    # 1. Base Kelly size
    base_size = kelly_size(adjusted_prob, market_price, side, bankroll,
                           kelly_fraction, max_position_pct)
    if base_size == 0:
        return 0.0, market_price

    # 2. Get relevant orderbook side
    levels = orderbook.asks if side == "BUY_YES" else orderbook.bids
    if not levels:
        return base_size, market_price

    # 3. Compute VWAP at base_size
    vwap, fillable = compute_vwap(levels, base_size)
    if fillable == 0:
        return 0.0, market_price

    # 4. Check if VWAP still gives enough edge
    edge_at_vwap = abs(adjusted_prob - vwap) - fee_rate
    if edge_at_vwap >= min_edge:
        # Full size is profitable at VWAP
        final_size = min(base_size, fillable)
        final_vwap, _ = compute_vwap(levels, final_size)
        return final_size, final_vwap

    # 5. Binary search for max profitable size
    lo, hi = 0.0, base_size
    for _ in range(15):  # converges quickly
        mid = (lo + hi) / 2
        v, f = compute_vwap(levels, mid)
        if f == 0:
            hi = mid
            continue
        e = abs(adjusted_prob - v) - fee_rate
        if e >= min_edge:
            lo = mid
        else:
            hi = mid

    final_size = min(lo, fillable)
    if final_size <= 0:
        return 0.0, market_price
    final_vwap, _ = compute_vwap(levels, final_size)
    return final_size, final_vwap


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
    # Cap at max position (USD spent)
    max_position = max_position_pct * bankroll
    position = min(position, max_position)

    # Cap by notional exposure (max possible payout)
    # At extreme prices, small USD amounts buy huge share counts
    if side == "BUY_YES" and market_price > 0:
        max_payout = position / market_price
    elif side == "BUY_NO" and market_price < 1.0:
        max_payout = position / (1.0 - market_price)
    else:
        max_payout = position
    if max_payout > max_position:
        position = max_position * (market_price if side == "BUY_YES" else (1.0 - market_price))

    return position


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
