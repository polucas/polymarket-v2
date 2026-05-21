"""Real-time WebSocket exit manager for take-profit / stop-loss.

Subscribes to the Polymarket CLOB market channel for YES tokens of open
positions.  On each orderbook update the best bid is checked against TP/SL
thresholds.  When triggered, the exit is executed immediately instead of
waiting for the 5-minute polling loop.

The 5-minute ``check_early_exits()`` polling in the scheduler is kept as a
fallback for periods when the WebSocket is disconnected.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Dict, Optional

import aiohttp
import structlog

from src.alerts import format_early_exit_alert, send_alert
from src.engine.resolution import calculate_early_exit_pnl, calculate_unrealized_roi
from src.models import TradeRecord

log = structlog.get_logger()

WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

# How often (seconds) to re-query open positions and refresh subscriptions.
_REFRESH_INTERVAL = 60

# Maximum exponential backoff (seconds) between reconnect attempts.
_MAX_BACKOFF = 30


class RealTimeExitManager:
    """WebSocket-based real-time TP/SL monitor.

    Subscribes to Polymarket market channel for YES tokens of open positions.
    On book update: calculates ROI from best bid, triggers exit if threshold
    is crossed.
    """

    def __init__(self, db, polymarket_client, settings):
        self.db = db
        self.polymarket = polymarket_client
        self.settings = settings
        self._active_positions: Dict[str, TradeRecord] = {}  # token_id -> trade
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._connected = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if not self.settings.EARLY_EXIT_ENABLED:
            log.info("ws_exit_disabled")
            return
        self._running = True
        self._task = asyncio.create_task(self._listen_loop())
        log.info("ws_exit_manager_started")

    async def stop(self) -> None:
        self._running = False
        self._connected = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        log.info("ws_exit_manager_stopped")

    # ------------------------------------------------------------------
    # Position tracking
    # ------------------------------------------------------------------

    async def _refresh_positions(self) -> None:
        """Load open trades from DB, index by YES token_id.

        get_open_trades() returns both unexited and already-exited trades
        (Phase 1 design — auto_resolve_trades needs the latter for natural
        resolution). The WS exit path must skip already-exited trades to
        avoid re-firing TP/SL on positions that were already closed.
        """
        trades = await self.db.get_open_trades()
        self._active_positions = {}
        for t in trades:
            if t.exit_type is not None:
                continue
            if t.clob_token_id_yes:
                self._active_positions[t.clob_token_id_yes] = t

    async def add_position(self, trade: TradeRecord) -> None:
        """Register a newly-opened position immediately, bypassing the 60s refresh interval.
        Called by scheduler right after execute_trade() returns a record. Without this,
        WS doesn't see the new position until the next refresh cycle, leaving a window
        where price can move adversely with no exit monitoring (Bug 7a)."""
        if trade is None or trade.exit_type is not None:
            return
        if not trade.clob_token_id_yes:
            return
        self._active_positions[trade.clob_token_id_yes] = trade
        log.info(
            "ws_position_added",
            market_id=trade.market_id,
            token_id=trade.clob_token_id_yes,
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def _listen_loop(self) -> None:
        backoff = 1
        while self._running:
            try:
                await self._refresh_positions()
                token_ids = list(self._active_positions.keys())

                if not token_ids:
                    self._connected = False
                    await asyncio.sleep(30)
                    continue

                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(WS_URL, heartbeat=self.settings.WS_HEARTBEAT_SECONDS) as ws:
                        await self._subscribe(ws, token_ids)
                        self._connected = True
                        backoff = 1  # reset on successful connect
                        log.info("ws_exit_connected", positions=len(token_ids))

                        last_refresh = time.monotonic()

                        async for msg in ws:
                            if not self._running:
                                break

                            if msg.type == aiohttp.WSMsgType.TEXT:
                                try:
                                    data = json.loads(msg.data)
                                    await self._handle_message(data)
                                except json.JSONDecodeError as e:
                                    log.warning(
                                        "ws_message_decode_error",
                                        raw_data=msg.data[:200] if hasattr(msg, "data") else None,
                                        error=str(e),
                                    )
                            elif msg.type in (
                                aiohttp.WSMsgType.CLOSED,
                                aiohttp.WSMsgType.ERROR,
                            ):
                                break

                            # Periodically refresh subscriptions
                            if time.monotonic() - last_refresh > _REFRESH_INTERVAL:
                                await self._refresh_positions()
                                new_ids = list(self._active_positions.keys())
                                if set(new_ids) != set(token_ids):
                                    token_ids = new_ids
                                    await self._subscribe(ws, token_ids)
                                    log.info("ws_exit_resubscribed", positions=len(token_ids))
                                last_refresh = time.monotonic()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._connected = False
                log.warning("ws_exit_reconnecting", error=str(e), backoff=backoff)
                await asyncio.sleep(min(backoff, _MAX_BACKOFF))
                backoff = min(backoff * 2, _MAX_BACKOFF)

        self._connected = False

    # ------------------------------------------------------------------
    # Subscription
    # ------------------------------------------------------------------

    @staticmethod
    async def _subscribe(ws, token_ids: list) -> None:
        """Send subscription message for the given token IDs."""
        if not token_ids:
            return
        await ws.send_json({
            "assets_ids": token_ids,
            "type": "market",
        })
        log.info("ws_subscription_sent", token_count=len(token_ids), tokens=token_ids[:5])

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    async def _handle_message(self, data) -> None:
        """Process incoming WS message and check TP/SL thresholds."""
        # Polymarket market channel sends lists of events
        events = data if isinstance(data, list) else [data]

        for event in events:
            if not isinstance(event, dict):
                log.debug("ws_event_unrecognized", event_type=type(event).__name__)
                continue

            asset_id = event.get("asset_id", "")
            trade = self._active_positions.get(asset_id)
            if trade is None:
                log.debug("ws_event_asset_not_tracked", asset_id=asset_id)
                continue

            # Extract best bid — this is the price we'd get if selling
            bids = event.get("bids", [])
            if not bids:
                log.warning(
                    "ws_event_missing_fields",
                    event_type=event.get("event_type"),
                    asset_id=asset_id,
                )
                continue

            try:
                best_bid = float(bids[0]["price"]) if isinstance(bids[0], dict) else float(bids[0])
                best_bid_size = float(bids[0].get("size", "0")) if isinstance(bids[0], dict) else 0.0
            except (ValueError, IndexError, KeyError, TypeError):
                continue

            # Bug 7b: skip exit when best-bid liquidity is too thin to be a real price.
            # Flash empty-book or 1-share dust orders should not trigger SL/TP.
            best_bid_usd = best_bid * best_bid_size
            if best_bid_usd < self.settings.MIN_BID_LIQUIDITY_USD:
                log.warning(
                    "ws_thin_book_skip",
                    market_id=trade.market_id,
                    asset_id=asset_id,
                    best_bid=best_bid,
                    best_bid_size=best_bid_size,
                    best_bid_usd=best_bid_usd,
                    threshold=self.settings.MIN_BID_LIQUIDITY_USD,
                )
                continue

            # Use best_bid as current YES price for ROI calculation
            roi = calculate_unrealized_roi(trade, best_bid)

            if roi >= self.settings.TAKE_PROFIT_ROI:
                await self._trigger_exit(trade, best_bid, "take_profit")
            elif roi <= self.settings.STOP_LOSS_ROI:
                await self._trigger_exit(trade, best_bid, "stop_loss")

    # ------------------------------------------------------------------
    # Exit execution
    # ------------------------------------------------------------------

    async def _trigger_exit(self, trade: TradeRecord, best_bid: float, exit_type: str) -> None:
        """Execute an early exit immediately."""
        # Defensive: never re-fire on an already-exited trade
        if trade.exit_type is not None:
            self._active_positions.pop(trade.clob_token_id_yes, None)
            return
        # Remove from tracking to prevent double-fire
        self._active_positions.pop(trade.clob_token_id_yes, None)

        pnl = calculate_early_exit_pnl(trade, best_bid)
        roi = calculate_unrealized_roi(trade, best_bid)

        log.info(
            "ws_early_exit_triggered",
            market_id=trade.market_id,
            exit_type=exit_type,
            roi=f"{roi:.4f}",
            pnl=f"{pnl:.2f}",
            exit_price=best_bid,
            entry_price=trade.market_price_at_decision,
        )

        try:
            # Update trade record
            trade.exit_type = exit_type
            trade.exit_price = best_bid
            trade.pnl = pnl
            trade.resolved_at = datetime.now(timezone.utc)
            # Dual-label: write pnl-based label immediately on exit
            trade.trade_profitable = 1 if trade.pnl > 0 else 0
            predicted_raw = (
                trade.grok_raw_probability if trade.action == "BUY_YES"
                else 1 - trade.grok_raw_probability
            )
            trade.pnl_brier_raw = (predicted_raw - trade.trade_profitable) ** 2
            predicted_adj = (
                trade.final_adjusted_probability if trade.action == "BUY_YES"
                else 1 - trade.final_adjusted_probability
            )
            trade.pnl_brier_adjusted = (predicted_adj - trade.trade_profitable) ** 2
            await self.db.update_trade(trade)

            # Update portfolio
            portfolio = await self.db.load_portfolio()
            portfolio.total_pnl += pnl
            portfolio.cash_balance += trade.position_size_usd + pnl
            portfolio.open_positions = [
                p for p in portfolio.open_positions if p.market_id != trade.market_id
            ]
            portfolio.total_equity = portfolio.cash_balance + sum(
                p.current_value for p in portfolio.open_positions
            )
            if portfolio.total_equity > portfolio.peak_equity:
                portfolio.peak_equity = portfolio.total_equity
            drawdown = (
                (portfolio.peak_equity - portfolio.total_equity) / portfolio.peak_equity
                if portfolio.peak_equity > 0
                else 0
            )
            portfolio.max_drawdown = max(portfolio.max_drawdown, drawdown)
            await self.db.save_portfolio(portfolio)

            # Send Telegram alert
            await send_alert(format_early_exit_alert(trade), self.settings)

            # Live mode: place taker sell order for guaranteed fill
            if self.settings.ENVIRONMENT == "live":
                token_id = (
                    trade.clob_token_id_yes
                    if trade.action == "BUY_YES"
                    else trade.clob_token_id_no
                )
                await self.polymarket.place_order(
                    market_id=trade.market_id,
                    side="SELL",
                    price=best_bid,
                    size=trade.position_size_usd,
                )

        except Exception as e:
            log.error(
                "ws_exit_error",
                market_id=trade.market_id,
                exit_type=exit_type,
                error=str(e),
            )
