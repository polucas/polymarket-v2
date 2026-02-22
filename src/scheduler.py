from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import List, Optional

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from src.alerts import (
    format_error_alert,
    format_monk_mode_alert,
    format_observe_only_alert,
    format_stale_scan_alert,
    format_tier2_alert,
    format_trade_alert,
    send_alert,
)
from src.config import MonkModeConfig, Settings
from src.db.sqlite import Database
from src.engine.execution import execute_trade
from src.engine.grok_client import GrokClient
from src.engine.resolution import auto_resolve_trades, update_unrealized_adverse_moves
from src.engine.trade_decision import (
    calculate_edge,
    check_monk_mode,
    determine_side,
    get_scan_mode,
    kelly_size,
)
from src.engine.trade_ranker import select_best_trades
from src.learning.adjustment import adjust_prediction, on_trade_resolved
from src.learning.calibration import CalibrationManager
from src.learning.experiments import get_current_experiment
from src.learning.market_type import MarketTypeManager
from src.learning.signal_tracker import SignalTrackerManager
from src.models import Market, Signal, TradeCandidate, TradeRecord
from src.pipelines.context_builder import build_grok_context, extract_keywords
from src.pipelines.polymarket import PolymarketClient
from src.pipelines.rss import RSSPipeline
from src.pipelines.twitter import TwitterDataPipeline

log = structlog.get_logger()


class Scheduler:
    def __init__(
        self,
        settings: Settings,
        db: Database,
        polymarket: PolymarketClient,
        twitter: TwitterDataPipeline,
        rss: RSSPipeline,
        grok: GrokClient,
        calibration_mgr: CalibrationManager,
        market_type_mgr: MarketTypeManager,
        signal_tracker_mgr: SignalTrackerManager,
    ):
        self._settings = settings
        self._db = db
        self._polymarket = polymarket
        self._twitter = twitter
        self._rss = rss
        self._grok = grok
        self._calibration_mgr = calibration_mgr
        self._market_type_mgr = market_type_mgr
        self._signal_tracker_mgr = signal_tracker_mgr
        self._monk_config = MonkModeConfig.from_settings(settings)
        self._scheduler = AsyncIOScheduler()
        self.last_scan_completed: Optional[datetime] = None
        self._tier2_active = False
        self._tier2_last_signal: Optional[datetime] = None
        self._observe_only_alerted_today: bool = False
        self._observe_only_alert_date: Optional[str] = None

    def start(self) -> None:
        self._scheduler.add_job(
            self.run_tier1_scan,
            "interval",
            minutes=self._settings.TIER1_SCAN_INTERVAL_MINUTES,
            id="tier1_scan",
            max_instances=1,
        )
        self._scheduler.add_job(
            self._auto_resolve,
            "interval",
            minutes=5,
            id="auto_resolve",
            max_instances=1,
        )
        self._scheduler.add_job(
            self._update_adverse,
            "interval",
            minutes=10,
            id="adverse_moves",
            max_instances=1,
        )
        self._scheduler.add_job(
            self._send_daily_summary,
            "cron",
            hour=self._settings.DAILY_SUMMARY_HOUR_UTC,
            minute=0,
            id="daily_summary",
            max_instances=1,
        )
        self._scheduler.add_job(
            self._check_stale_scan,
            "interval",
            minutes=15,
            id="stale_check",
            max_instances=1,
        )
        self._scheduler.start()
        log.info("scheduler_started",
                 tier1_interval=self._settings.TIER1_SCAN_INTERVAL_MINUTES)

    def stop(self) -> None:
        self._scheduler.shutdown(wait=False)
        log.info("scheduler_stopped")

    # ------------------------------------------------------------------
    # Resolution helpers
    # ------------------------------------------------------------------

    async def _auto_resolve(self) -> None:
        try:
            await auto_resolve_trades(self._db, self._polymarket)
            # After resolution, trigger learning updates for newly resolved trades
            resolved = await self._db.get_all_resolved_trades()
            for r in resolved:
                if r.resolved_at and r.brier_score_raw is not None:
                    # Already processed
                    pass
        except Exception as e:
            log.error("auto_resolve_error", error=str(e))
            await send_alert(format_error_alert(f"Auto-resolve failed: {e}"), self._settings)

    async def _update_adverse(self) -> None:
        try:
            await update_unrealized_adverse_moves(self._db, self._polymarket)
        except Exception as e:
            log.error("adverse_update_error", error=str(e))
            await send_alert(format_error_alert(f"Adverse move update failed: {e}"), self._settings)

    # ------------------------------------------------------------------
    # Tier 2 activation
    # ------------------------------------------------------------------

    def should_activate_tier2(self, signals: List[Signal]) -> bool:
        """Check if Tier 2 should activate.
        Needs 2+ crypto-relevant signals, at least one from S1/S2 or 100K+ followers.
        """
        crypto_signals = [
            s for s in signals
            if any(kw in s.content.lower() for kw in
                   ["bitcoin", "btc", "ethereum", "eth", "crypto", "solana", "sol"])
        ]
        if len(crypto_signals) < 2:
            return False
        has_authority = any(
            s.source_tier in ("S1", "S2") or s.followers >= 100_000
            for s in crypto_signals
        )
        return has_authority

    # ------------------------------------------------------------------
    # Tier 1 scan
    # ------------------------------------------------------------------

    async def run_tier1_scan(self) -> None:
        log.info("tier1_scan_start")
        try:
            # Reset observe-only alert flag at start of new day
            today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if self._observe_only_alert_date != today_str:
                self._observe_only_alerted_today = False
                self._observe_only_alert_date = today_str

            today_trades = await self._db.get_today_trades()
            week_trades = await self._db.get_week_trades()
            api_spend = await self._db.get_today_api_spend()
            portfolio = await self._db.load_portfolio()

            scan_mode = get_scan_mode(today_trades, self._monk_config)
            if scan_mode == "observe_only" and not self._observe_only_alerted_today:
                self._observe_only_alerted_today = True
                tier1_executed = [t for t in today_trades if t.tier == 1 and t.action != "SKIP"]
                await send_alert(
                    format_observe_only_alert(len(tier1_executed), self._monk_config.tier1_daily_trade_cap),
                    self._settings,
                )
            log.info("scan_mode", mode=scan_mode)

            # Get experiment run ID
            experiment = await get_current_experiment(self._db)
            experiment_run = experiment.run_id if experiment else "default"

            # Get active markets
            markets = await self._polymarket.get_active_markets(
                tier=1,
                tier1_fee_rate=self._settings.TIER1_FEE_RATE,
                tier2_fee_rate=self._settings.TIER2_FEE_RATE,
            )
            if not markets:
                log.info("no_tier1_markets")
                self.last_scan_completed = datetime.now(timezone.utc)
                return

            # Collect all signals from RSS for breaking news
            rss_signals = await self._rss.get_breaking_news()

            # Check Tier 2 activation
            if self.should_activate_tier2(rss_signals) and not self._tier2_active:
                await self._activate_tier2()

            # Process each market
            candidates: List[TradeCandidate] = []
            all_skips: List[TradeRecord] = []

            for market in markets:
                try:
                    await self._process_market(
                        market=market,
                        rss_signals=rss_signals,
                        scan_mode=scan_mode,
                        candidates=candidates,
                        all_skips=all_skips,
                        today_trades=today_trades,
                        experiment_run=experiment_run,
                        tier=1,
                    )
                except Exception as e:
                    log.error("market_processing_error",
                              market_id=market.market_id, error=str(e))
                    continue

            # Rank and select
            if candidates:
                tier1_executed = [t for t in today_trades if t.tier == 1 and t.action != "SKIP"]
                remaining_cap = max(0, self._monk_config.tier1_daily_trade_cap - len(tier1_executed))

                to_execute, to_skip = select_best_trades(
                    candidates, remaining_cap, portfolio.open_positions, portfolio.total_equity,
                )

                # Execute trades
                monk_alerted_reasons: set = set()
                for candidate in to_execute:
                    allowed, reason = check_monk_mode(
                        self._monk_config, candidate, portfolio,
                        today_trades, week_trades, api_spend,
                    )
                    if not allowed:
                        candidate.skip_reason = reason
                        to_skip.append(candidate)
                        if reason not in monk_alerted_reasons:
                            monk_alerted_reasons.add(reason)
                            await send_alert(
                                format_monk_mode_alert(reason), self._settings
                            )
                        continue

                    record = await execute_trade(
                        candidate, portfolio, self._db, self._polymarket,
                        self._settings.ENVIRONMENT, experiment_run,
                    )
                    if record:
                        today_trades.append(record)
                        await send_alert(
                            format_trade_alert(record), self._settings
                        )

                # Record skips
                for skip in to_skip:
                    await self._record_skip(skip, experiment_run)

            self.last_scan_completed = datetime.now(timezone.utc)
            log.info("tier1_scan_complete",
                     markets_scanned=len(markets),
                     candidates=len(candidates))

        except Exception as e:
            log.error("tier1_scan_error", error=str(e))
            await send_alert(format_error_alert(f"Tier 1 scan failed: {e}"), self._settings)

    # ------------------------------------------------------------------
    # Tier 2 scan
    # ------------------------------------------------------------------

    async def run_tier2_scan(self) -> None:
        log.info("tier2_scan_start")
        try:
            today_trades = await self._db.get_today_trades()
            week_trades = await self._db.get_week_trades()
            api_spend = await self._db.get_today_api_spend()
            portfolio = await self._db.load_portfolio()

            experiment = await get_current_experiment(self._db)
            experiment_run = experiment.run_id if experiment else "default"

            markets = await self._polymarket.get_active_markets(
                tier=2,
                tier1_fee_rate=self._settings.TIER1_FEE_RATE,
                tier2_fee_rate=self._settings.TIER2_FEE_RATE,
            )
            if not markets:
                log.info("no_tier2_markets")
                return

            rss_signals = await self._rss.get_breaking_news()

            candidates: List[TradeCandidate] = []
            all_skips: List[TradeRecord] = []

            for market in markets:
                try:
                    await self._process_market(
                        market=market,
                        rss_signals=rss_signals,
                        scan_mode="active",
                        candidates=candidates,
                        all_skips=all_skips,
                        today_trades=today_trades,
                        experiment_run=experiment_run,
                        tier=2,
                    )
                except Exception as e:
                    log.error("tier2_market_error",
                              market_id=market.market_id, error=str(e))
                    continue

            if candidates:
                tier2_executed = [t for t in today_trades if t.tier == 2 and t.action != "SKIP"]
                remaining_cap = max(0, self._monk_config.tier2_daily_trade_cap - len(tier2_executed))

                to_execute, to_skip = select_best_trades(
                    candidates, remaining_cap, portfolio.open_positions, portfolio.total_equity,
                )

                monk_alerted_reasons: set = set()
                for candidate in to_execute:
                    allowed, reason = check_monk_mode(
                        self._monk_config, candidate, portfolio,
                        today_trades, week_trades, api_spend,
                    )
                    if not allowed:
                        candidate.skip_reason = reason
                        to_skip.append(candidate)
                        if reason not in monk_alerted_reasons:
                            monk_alerted_reasons.add(reason)
                            await send_alert(
                                format_monk_mode_alert(reason), self._settings
                            )
                        continue

                    record = await execute_trade(
                        candidate, portfolio, self._db, self._polymarket,
                        self._settings.ENVIRONMENT, experiment_run,
                    )
                    if record:
                        today_trades.append(record)
                        await send_alert(
                            format_trade_alert(record), self._settings
                        )

                for skip in to_skip:
                    await self._record_skip(skip, experiment_run)

            # Check deactivation: no new crypto signals for 30 min
            now = datetime.now(timezone.utc)
            if self._tier2_last_signal:
                minutes_since = (now - self._tier2_last_signal).total_seconds() / 60
                if minutes_since > 30:
                    await self._deactivate_tier2()

            log.info("tier2_scan_complete", markets_scanned=len(markets))

        except Exception as e:
            log.error("tier2_scan_error", error=str(e))
            await send_alert(format_error_alert(f"Tier 2 scan failed: {e}"), self._settings)

    # ------------------------------------------------------------------
    # Market processing (shared by both tiers)
    # ------------------------------------------------------------------

    async def _process_market(
        self,
        market: Market,
        rss_signals: List[Signal],
        scan_mode: str,
        candidates: List[TradeCandidate],
        all_skips: list,
        today_trades: List[TradeRecord],
        experiment_run: str,
        tier: int,
    ) -> None:
        # Check if market type disabled by learning
        if self._market_type_mgr.should_disable(market.market_type):
            log.info("market_type_disabled", market_type=market.market_type)
            skip_record = self._build_skip_record(market, "market_type_disabled", experiment_run, tier)
            await self._db.save_trade(skip_record)
            return

        # Extract keywords
        keywords = extract_keywords(market.market_id, market.question, market.market_type)

        # Fetch Twitter signals
        twitter_signals = await self._twitter.get_signals_for_market(keywords)

        # Filter RSS signals relevant to this market
        relevant_rss = [
            s for s in rss_signals
            if any(kw.lower() in s.content.lower() for kw in keywords[:5])
        ]

        # Update tier2 last signal time if crypto signals found
        if tier == 2 and (twitter_signals or relevant_rss):
            self._tier2_last_signal = datetime.now(timezone.utc)

        # Observe-only mode: record as SKIP
        if scan_mode == "observe_only":
            skip_record = self._build_skip_record(
                market, "observe_only", experiment_run, tier,
            )
            await self._db.save_trade(skip_record)
            return

        # Get orderbook
        orderbook = await self._polymarket.get_orderbook(market.clob_token_id_yes, market.market_id)

        # Build context and call Grok
        context = build_grok_context(market, twitter_signals, relevant_rss, orderbook)
        grok_result = await self._grok.call_grok_with_retry(context, market.market_id)

        if grok_result is None:
            skip_record = self._build_skip_record(market, "grok_failed", experiment_run, tier)
            await self._db.save_trade(skip_record)
            return

        grok_prob = grok_result["estimated_probability"]
        grok_conf = grok_result["confidence"]
        reasoning = grok_result.get("reasoning", "")
        signal_types = grok_result.get("signal_info_types", [])

        # Adjust prediction (5-step pipeline)
        adj_prob, adj_conf, extra_edge = adjust_prediction(
            grok_prob, grok_conf, market.market_type, signal_types,
            self._calibration_mgr, self._market_type_mgr, self._signal_tracker_mgr,
        )

        # Calculate edge and side
        fee_rate = self._settings.TIER1_FEE_RATE if tier == 1 else self._settings.TIER2_FEE_RATE
        edge = calculate_edge(adj_prob, market.yes_price, fee_rate) - extra_edge
        side = determine_side(adj_prob, market.yes_price)

        min_edge = self._settings.TIER1_MIN_EDGE if tier == 1 else self._settings.TIER2_MIN_EDGE

        if side == "SKIP" or edge < min_edge:
            skip_record = self._build_skip_record(
                market, f"low_edge_{edge:.4f}" if side != "SKIP" else "no_direction",
                experiment_run, tier,
                grok_prob=grok_prob, grok_conf=grok_conf,
                adj_prob=adj_prob, adj_conf=adj_conf,
                reasoning=reasoning, signal_types=signal_types,
            )
            await self._db.save_trade(skip_record)
            return

        # Kelly sizing
        position_size = kelly_size(
            adj_prob, market.yes_price, side,
            (await self._db.load_portfolio()).total_equity,
            self._settings.KELLY_FRACTION,
            self._settings.MAX_POSITION_PCT,
        )

        if position_size < 1.0:
            skip_record = self._build_skip_record(
                market, f"position_too_small_{position_size:.2f}", experiment_run, tier,
                grok_prob=grok_prob, grok_conf=grok_conf,
                adj_prob=adj_prob, adj_conf=adj_conf,
                reasoning=reasoning, signal_types=signal_types,
            )
            await self._db.save_trade(skip_record)
            return

        # Detect headline-only
        headline_only = all(s.headline_only for s in (twitter_signals + relevant_rss)) if (twitter_signals or relevant_rss) else False

        ob_depth = sum(orderbook.bids) + sum(orderbook.asks)

        candidate = TradeCandidate(
            market=market,
            adjusted_probability=adj_prob,
            adjusted_confidence=adj_conf,
            calculated_edge=edge,
            position_size=position_size,
            side=side,
            resolution_hours=market.hours_to_resolution,
            signal_tags=signal_types,
            fee_rate=fee_rate,
            market_price=market.yes_price,
            kelly_fraction_used=self._settings.KELLY_FRACTION,
            orderbook_depth=ob_depth,
            tier=tier,
            grok_raw_probability=grok_prob,
            grok_raw_confidence=grok_conf,
            grok_reasoning=reasoning,
            grok_signal_types=signal_types,
            headline_only_signal=headline_only,
            calibration_adjustment=adj_conf - grok_conf,
            market_type_adjustment=extra_edge,
            signal_weight_adjustment=0.0,
        )
        candidates.append(candidate)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_skip_record(
        self,
        market: Market,
        reason: str,
        experiment_run: str,
        tier: int,
        grok_prob: float = 0.0,
        grok_conf: float = 0.0,
        adj_prob: float = 0.0,
        adj_conf: float = 0.0,
        reasoning: str = "",
        signal_types: list = None,
    ) -> TradeRecord:
        return TradeRecord(
            record_id=str(uuid.uuid4()),
            experiment_run=experiment_run,
            timestamp=datetime.now(timezone.utc),
            model_used="grok-3-fast",
            market_id=market.market_id,
            market_question=market.question,
            market_type=market.market_type,
            resolution_window_hours=market.hours_to_resolution,
            tier=tier,
            grok_raw_probability=grok_prob,
            grok_raw_confidence=grok_conf,
            grok_reasoning=reasoning,
            grok_signal_types=signal_types or [],
            final_adjusted_probability=adj_prob,
            final_adjusted_confidence=adj_conf,
            market_price_at_decision=market.yes_price,
            fee_rate=market.fee_rate,
            action="SKIP",
            skip_reason=reason,
        )

    async def _record_skip(self, candidate: TradeCandidate, experiment_run: str) -> None:
        record = TradeRecord(
            record_id=str(uuid.uuid4()),
            experiment_run=experiment_run,
            timestamp=datetime.now(timezone.utc),
            model_used="grok-3-fast",
            market_id=candidate.market.market_id,
            market_question=candidate.market.question,
            market_type=candidate.market.market_type,
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
            action="SKIP",
            skip_reason=candidate.skip_reason or "ranked_out",
            position_size_usd=candidate.position_size,
            kelly_fraction_used=candidate.kelly_fraction_used,
            market_cluster_id=candidate.market_cluster_id,
        )
        await self._db.save_trade(record)

    async def _activate_tier2(self) -> None:
        if self._tier2_active:
            return
        self._tier2_active = True
        self._tier2_last_signal = datetime.now(timezone.utc)
        self._scheduler.add_job(
            self.run_tier2_scan,
            "interval",
            minutes=self._settings.TIER2_SCAN_INTERVAL_MINUTES,
            id="tier2_scan",
            max_instances=1,
        )
        log.info("tier2_activated")
        await send_alert(format_tier2_alert(active=True), self._settings)

    async def _deactivate_tier2(self) -> None:
        if not self._tier2_active:
            return
        self._tier2_active = False
        try:
            self._scheduler.remove_job("tier2_scan")
        except Exception:
            pass
        log.info("tier2_deactivated")
        await send_alert(format_tier2_alert(active=False), self._settings)

    # ------------------------------------------------------------------
    # Daily summary & stale scan
    # ------------------------------------------------------------------

    async def _send_daily_summary(self) -> None:
        """Send daily trade summary via Telegram."""
        try:
            from src.alerts import format_daily_summary
            today_trades = await self._db.get_today_trades()
            portfolio = await self._db.load_portfolio()
            message = format_daily_summary(today_trades, portfolio)
            await send_alert(message, self._settings)
            log.info("daily_summary_sent", trade_count=len(today_trades))
        except Exception as e:
            log.error("daily_summary_error", error=str(e))

    async def _check_stale_scan(self) -> None:
        """Alert if no scan has completed in >30 minutes."""
        try:
            if self.last_scan_completed is None:
                return  # Still initializing
            now = datetime.now(timezone.utc)
            minutes_since = (now - self.last_scan_completed).total_seconds() / 60
            if minutes_since > 30:
                await send_alert(
                    format_stale_scan_alert(minutes_since), self._settings
                )
        except Exception as e:
            log.error("stale_check_error", error=str(e))
