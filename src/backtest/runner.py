from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Optional

import structlog

from src.backtest.clock import Clock
from src.backtest.mocks import BacktestPolymarketClient, BacktestRSSPipeline, BacktestLLMClient, BacktestTwitterPipeline
from src.db.sqlite import Database
from src.db.migrations import run_migrations
from src.engine.grok_client import LLMClient
from src.engine.resolution import auto_resolve_trades
from src.learning.calibration import CalibrationManager
from src.learning.experiments import start_experiment
from src.learning.market_type import MarketTypeManager
from src.learning.signal_tracker import SignalTrackerManager
from src.models import Portfolio
from src.scheduler import Scheduler

log = structlog.get_logger()


class BacktestRunner:
    def __init__(
        self,
        settings,
        start_dt: datetime,
        end_dt: datetime,
        backtest_data_db: str = "data/backtest_data.db",
        outputs_db: str = "data/backtest_outputs.db",
        grok_cache_db: str = "data/backtest_grok_cache.db",
    ):
        self._settings = settings
        self._start_dt = start_dt if start_dt.tzinfo else start_dt.replace(tzinfo=timezone.utc)
        self._end_dt = end_dt if end_dt.tzinfo else end_dt.replace(tzinfo=timezone.utc)
        self._backtest_data_db = backtest_data_db
        self._outputs_db = outputs_db
        self._grok_cache_db = grok_cache_db

    async def run(self) -> dict:
        # --- Initialize outputs DB (isolated from predictor.db) ---
        db = await Database.init(self._outputs_db)
        await run_migrations(db)

        # Initialize portfolio in outputs DB
        portfolio = await db.load_portfolio()
        if portfolio.total_equity == 0:
            portfolio = Portfolio(
                cash_balance=self._settings.INITIAL_BANKROLL,
                total_equity=self._settings.INITIAL_BANKROLL,
                peak_equity=self._settings.INITIAL_BANKROLL,
            )
            await db.save_portfolio(portfolio)

        # Create experiment run (required by FK constraint on trade_records)
        run_id = f"backtest_{self._start_dt.strftime('%Y%m%d')}_{self._end_dt.strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}"
        await start_experiment(run_id, f"Backtest {self._start_dt.date()} -> {self._end_dt.date()}", {}, self._settings.LLM_MODEL, db)

        # Learning managers (fresh state for this backtest)
        calibration_mgr = CalibrationManager()
        await calibration_mgr.load(db)
        market_type_mgr = MarketTypeManager()
        await market_type_mgr.load(db)
        signal_tracker_mgr = SignalTrackerManager()
        await signal_tracker_mgr.load(db)

        # Mocked dependencies
        polymarket = BacktestPolymarketClient(self._settings, self._backtest_data_db)
        rss = BacktestRSSPipeline(self._backtest_data_db)
        twitter = BacktestTwitterPipeline(self._backtest_data_db)

        # Real LLM with cache — LLMClient requires (settings, db)
        real_llm = LLMClient(self._settings, db)
        grok = BacktestLLMClient(real_llm, self._grok_cache_db)

        # Suppress Telegram alerts
        self._settings.TELEGRAM_BOT_TOKEN = ""

        # Build scheduler with mocked deps (never call .start() — we drive the loop manually)
        scheduler = Scheduler(
            settings=self._settings,
            db=db,
            polymarket=polymarket,
            twitter=twitter,
            rss=rss,
            grok=grok,
            calibration_mgr=calibration_mgr,
            market_type_mgr=market_type_mgr,
            signal_tracker_mgr=signal_tracker_mgr,
        )

        # --- Tick loop ---
        Clock.set_time(self._start_dt)
        ticks = 0
        total_ticks = int((self._end_dt - self._start_dt).total_seconds() / (15 * 60)) + 1

        log.info("backtest_start",
                 start=self._start_dt.isoformat(),
                 end=self._end_dt.isoformat(),
                 ticks=total_ticks)

        while Clock.utcnow() <= self._end_dt:
            await scheduler.run_tier1_scan()
            await auto_resolve_trades(db, polymarket)
            Clock.advance(15)
            ticks += 1

            if ticks % 96 == 0:  # Log every 24h (96 x 15min)
                log.info("backtest_progress",
                         tick=ticks,
                         total=total_ticks,
                         clock=Clock.utcnow().isoformat(),
                         **grok.cache_stats)

        Clock.reset()

        # --- Build summary ---
        summary = await self._build_summary(db, ticks, grok.cache_stats)
        await db.close()

        self._print_summary(summary)
        return summary

    async def _build_summary(self, db: Database, ticks: int, cache_stats: dict) -> dict:
        # Fetch all trade records via direct query (Database has no get_all_trades())
        cursor = await db._conn.execute(
            "SELECT * FROM trade_records ORDER BY timestamp"
        )
        rows = await cursor.fetchall()
        all_trades = [db._row_to_trade(r) for r in rows]

        executed = [t for t in all_trades if t.action != "SKIP"]
        resolved = [t for t in executed if t.actual_outcome is not None]
        wins = [
            t for t in resolved
            if (t.action == "BUY_YES" and t.actual_outcome) or (t.action == "BUY_NO" and not t.actual_outcome)
        ]

        total_pnl = sum(t.pnl or 0 for t in resolved)
        win_rate = len(wins) / len(resolved) if resolved else 0.0
        brier_raw = sum(t.brier_score_raw or 0 for t in resolved) / len(resolved) if resolved else None
        brier_adj = sum(t.brier_score_adjusted or 0 for t in resolved) / len(resolved) if resolved else None

        # Per market type
        by_type: dict = {}
        for t in resolved:
            mt = t.market_type
            if mt not in by_type:
                by_type[mt] = {"trades": 0, "pnl": 0.0, "brier_sum": 0.0, "wins": 0}
            by_type[mt]["trades"] += 1
            by_type[mt]["pnl"] += t.pnl or 0
            by_type[mt]["brier_sum"] += t.brier_score_raw or 0
            if t in wins:
                by_type[mt]["wins"] += 1

        return {
            "start": self._start_dt.isoformat(),
            "end": self._end_dt.isoformat(),
            "ticks": ticks,
            "trades_executed": len(executed),
            "trades_skipped": len(all_trades) - len(executed),
            "trades_resolved": len(resolved),
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "brier_raw": brier_raw,
            "brier_adjusted": brier_adj,
            "by_market_type": by_type,
            "grok_cache": cache_stats,
        }

    def _print_summary(self, s: dict) -> None:
        print(f"\nBacktest: {s['start'][:10]} -> {s['end'][:10]} ({s['ticks']} ticks)")
        print(f"Trades executed: {s['trades_executed']:>5}   Win rate: {s['win_rate']:.1%}   Total PnL: ${s['total_pnl']:+.2f}")
        if s['brier_raw'] is not None:
            print(f"Brier raw: {s['brier_raw']:.3f}   Brier adjusted: {s['brier_adjusted']:.3f}")
        print("By market type:")
        for mt, v in s["by_market_type"].items():
            wr = v["wins"] / v["trades"] if v["trades"] else 0
            br = v["brier_sum"] / v["trades"] if v["trades"] else 0
            print(f"  {mt:<15} {v['trades']:>3} trades  ${v['pnl']:+8.2f}  WR {wr:.0%}  Brier {br:.3f}")
        c = s["grok_cache"]
        total = c["hits"] + c["misses"]
        hit_pct = c["hits"] / total if total else 0
        print(f"Grok cache: {c['hits']} hits / {c['misses']} misses ({hit_pct:.0%} hit rate)")
