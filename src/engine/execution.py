from __future__ import annotations

import random
import uuid
from datetime import datetime, timezone
from typing import Optional

import structlog

from src.db.sqlite import Database
from src.models import (
    ExecutionResult,
    Portfolio,
    Position,
    TradeCandidate,
    TradeRecord,
)

log = structlog.get_logger()


def simulate_execution(
    side: str,
    price: float,
    size_usd: float,
    execution_type: str,
    orderbook_depth: float,
) -> ExecutionResult:
    """Simulate trade execution with realistic slippage.

    Taker: slippage = 0.005 + 0.01 * min(size_usd / max(orderbook_depth, 1), 1.0)
           YES price += slippage, NO price -= slippage, fill_probability = 1.0
    Maker: fill_probability = 0.4 + 0.4 * (1 - abs(price - 0.5))
           slippage = 0, executed_price = price
    """
    if execution_type == "maker":
        fill_probability = 0.4 + 0.4 * (1 - abs(price - 0.5))
        filled = random.random() < fill_probability
        return ExecutionResult(
            executed_price=price,
            slippage=0.0,
            fill_probability=fill_probability,
            filled=filled,
        )
    else:  # taker
        slippage = 0.005 + 0.01 * min(size_usd / max(orderbook_depth, 1), 1.0)
        if "YES" in side.upper():
            executed_price = price + slippage
        else:
            executed_price = price - slippage
        # Clamp to [0.01, 0.99]
        executed_price = max(0.01, min(0.99, executed_price))
        return ExecutionResult(
            executed_price=executed_price,
            slippage=slippage,
            fill_probability=1.0,
            filled=True,
        )


async def execute_trade(
    candidate: TradeCandidate,
    portfolio: Portfolio,
    db: Database,
    polymarket_client,
    environment: str,
    experiment_run: str = "",
    model_used: str = "grok-3-fast",
) -> Optional[TradeRecord]:
    """Execute a trade (paper or live) and create trade record.

    Returns None if order is not filled (maker orders).
    """
    market = candidate.market

    if environment == "paper":
        exec_type = "taker" if candidate.tier == 1 else "maker"
        result = simulate_execution(
            side=candidate.side,
            price=candidate.market_price,
            size_usd=candidate.position_size,
            execution_type=exec_type,
            orderbook_depth=candidate.orderbook_depth,
        )
    else:
        # Live execution
        try:
            order_result = await polymarket_client.place_order(
                market_id=market.market_id,
                side=candidate.side,
                price=candidate.market_price,
                size=candidate.position_size,
            )
            if order_result.get("status") == "error":
                log.error("live_order_failed", market_id=market.market_id, error=order_result.get("error"))
                return None
            result = ExecutionResult(
                executed_price=candidate.market_price,
                slippage=0.0,
                fill_probability=1.0,
                filled=True,
            )
        except Exception as e:
            log.error("live_execution_error", market_id=market.market_id, error=str(e))
            return None

    if not result.filled:
        log.info("order_not_filled", market_id=market.market_id, side=candidate.side)
        return None

    # Update portfolio
    portfolio.cash_balance -= candidate.position_size
    portfolio.open_positions.append(Position(
        market_id=market.market_id,
        side=candidate.side,
        entry_price=result.executed_price,
        size_usd=candidate.position_size,
        current_value=candidate.position_size,
        market_cluster_id=candidate.market_cluster_id,
    ))
    await db.save_portfolio(portfolio)

    # Create trade record
    now = datetime.now(timezone.utc)
    record = TradeRecord(
        record_id=str(uuid.uuid4()),
        experiment_run=experiment_run,
        timestamp=now,
        model_used=model_used,
        market_id=market.market_id,
        market_question=market.question,
        market_type=market.market_type,
        resolution_window_hours=candidate.resolution_hours,
        tier=candidate.tier,
        grok_raw_probability=candidate.grok_raw_probability,
        grok_raw_confidence=candidate.grok_raw_confidence,
        grok_reasoning=candidate.grok_reasoning,
        grok_signal_types=candidate.grok_signal_types,
        headline_only_signal=candidate.headline_only_signal,
        calibration_adjustment=candidate.calibration_adjustment,
        market_type_adjustment=candidate.market_type_adjustment,
        signal_weight_adjustment=candidate.signal_weight_adjustment,
        final_adjusted_probability=candidate.adjusted_probability,
        final_adjusted_confidence=candidate.adjusted_confidence,
        market_price_at_decision=candidate.market_price,
        orderbook_depth_usd=candidate.orderbook_depth,
        fee_rate=candidate.fee_rate,
        calculated_edge=candidate.calculated_edge,
        trade_score=candidate.score,
        action=candidate.side,
        position_size_usd=candidate.position_size,
        kelly_fraction_used=candidate.kelly_fraction_used,
        market_cluster_id=candidate.market_cluster_id,
    )

    await db.save_trade(record)
    log.info("trade_executed",
             market_id=market.market_id,
             side=candidate.side,
             size=candidate.position_size,
             price=result.executed_price,
             slippage=result.slippage)
    return record
