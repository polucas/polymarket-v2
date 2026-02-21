from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import structlog

from src.db.sqlite import Database
from src.models import Portfolio, TradeRecord

log = structlog.get_logger()


def calculate_pnl(record: TradeRecord, outcome: bool) -> float:
    """Calculate PnL for a resolved trade.
    outcome: True = YES, False = NO
    """
    if record.action == "BUY_YES":
        if outcome:
            return record.position_size_usd * (1.0 - record.market_price_at_decision) - (record.position_size_usd * record.fee_rate)
        else:
            return -record.position_size_usd
    elif record.action == "BUY_NO":
        if not outcome:
            return record.position_size_usd * record.market_price_at_decision - (record.position_size_usd * record.fee_rate)
        else:
            return -record.position_size_usd
    return 0.0


def calculate_hypothetical_pnl(record: TradeRecord) -> float:
    """For skipped trades: what would PnL be if trade was taken."""
    if record.actual_outcome is None:
        return 0.0
    return calculate_pnl(record, record.actual_outcome)


async def auto_resolve_trades(db: Database, polymarket_client) -> None:
    """Check and resolve open trades."""
    open_trades = await db.get_open_trades()
    if not open_trades:
        return

    portfolio = await db.load_portfolio()
    resolved_count = 0

    for trade in open_trades:
        try:
            market = await polymarket_client.get_market(trade.market_id)
            if market is None:
                continue

            if not market.resolved:
                # For crypto_15m: check if past expected resolution time
                if trade.market_type == "crypto_15m":
                    now = datetime.now(timezone.utc)
                    expected_resolution = trade.timestamp.replace(tzinfo=timezone.utc) if trade.timestamp.tzinfo is None else trade.timestamp
                    expected_resolution = expected_resolution + __import__('datetime').timedelta(hours=trade.resolution_window_hours)
                    if now < expected_resolution:
                        continue
                    # Past resolution time but market not resolved - check current price
                    outcome = market.yes_price > 0.5
                else:
                    continue
            else:
                outcome = market.resolution == "YES"

            # Calculate PnL
            pnl = calculate_pnl(trade, outcome)

            # Calculate Brier scores
            actual_val = 1.0 if outcome else 0.0
            brier_raw = (trade.grok_raw_probability - actual_val) ** 2
            brier_adjusted = (trade.final_adjusted_probability - actual_val) ** 2

            # Update trade record
            trade.actual_outcome = outcome
            trade.pnl = pnl
            trade.brier_score_raw = brier_raw
            trade.brier_score_adjusted = brier_adjusted
            trade.resolved_at = datetime.now(timezone.utc)

            await db.update_trade(trade)

            # Update portfolio
            portfolio.total_pnl += pnl
            portfolio.cash_balance += trade.position_size_usd + pnl
            portfolio.total_equity = portfolio.cash_balance + sum(
                p.current_value for p in portfolio.open_positions
                if p.market_id != trade.market_id
            )
            portfolio.open_positions = [
                p for p in portfolio.open_positions if p.market_id != trade.market_id
            ]
            if portfolio.total_equity > portfolio.peak_equity:
                portfolio.peak_equity = portfolio.total_equity
            drawdown = (portfolio.peak_equity - portfolio.total_equity) / portfolio.peak_equity if portfolio.peak_equity > 0 else 0
            portfolio.max_drawdown = max(portfolio.max_drawdown, drawdown)

            await db.save_portfolio(portfolio)
            resolved_count += 1

            log.info("trade_resolved",
                     market_id=trade.market_id,
                     outcome="YES" if outcome else "NO",
                     pnl=pnl,
                     brier_raw=brier_raw,
                     brier_adjusted=brier_adjusted)

        except Exception as e:
            log.error("resolution_error", market_id=trade.market_id, error=str(e))
            continue

    if resolved_count:
        log.info("resolution_cycle_complete", resolved=resolved_count)


async def update_unrealized_adverse_moves(db: Database, polymarket_client) -> None:
    """Track unrealized adverse moves for Monk Mode cooldown."""
    open_trades = await db.get_open_trades()

    for trade in open_trades:
        try:
            market = await polymarket_client.get_market(trade.market_id)
            if market is None:
                continue

            current_price = market.yes_price
            entry_price = trade.market_price_at_decision

            if trade.action == "BUY_YES":
                adverse_move = max(0, entry_price - current_price)
            elif trade.action == "BUY_NO":
                adverse_move = max(0, current_price - entry_price)
            else:
                continue

            if adverse_move > 0.10:
                trade.unrealized_adverse_move = adverse_move
                await db.update_trade(trade)
                log.warning("adverse_move_detected",
                           market_id=trade.market_id,
                           adverse_move=adverse_move)
        except Exception as e:
            log.warning("adverse_move_check_failed", market_id=trade.market_id, error=str(e))
