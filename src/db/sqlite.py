from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import aiosqlite

from src.models import (
    CalibrationBucket,
    ExperimentRun,
    MarketTypePerformance,
    ModelSwapEvent,
    Portfolio,
    Position,
    SignalTracker,
    TradeRecord,
    CALIBRATION_BUCKET_RANGES,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


def _parse_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None


class Database:
    def __init__(self, conn: aiosqlite.Connection):
        self._conn = conn

    @classmethod
    async def init(cls, db_path: str) -> "Database":
        conn = await aiosqlite.connect(db_path, check_same_thread=False)
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA foreign_keys=ON")
        return cls(conn)

    async def close(self) -> None:
        await self._conn.close()

    # ------------------------------------------------------------------
    # Trade Records
    # ------------------------------------------------------------------

    async def save_trade(self, r: TradeRecord) -> None:
        await self._conn.execute(
            """INSERT INTO trade_records (
                record_id, experiment_run, timestamp, model_used,
                market_id, market_question, market_type, resolution_window_hours, resolution_datetime, tier,
                grok_raw_probability, grok_raw_confidence, grok_reasoning, grok_signal_types,
                headline_only_signal,
                calibration_adjustment, market_type_adjustment, signal_weight_adjustment,
                final_adjusted_probability, final_adjusted_confidence,
                market_price_at_decision, orderbook_depth_usd, fee_rate, calculated_edge, trade_score,
                action, skip_reason, position_size_usd, kelly_fraction_used, market_cluster_id,
                actual_outcome, pnl, brier_score_raw, brier_score_adjusted, resolved_at,
                unrealized_adverse_move, voided, void_reason
            ) VALUES (?,?,?,?, ?,?,?,?,?,?, ?,?,?,?,?, ?,?,?, ?,?, ?,?,?,?,?, ?,?,?,?,?, ?,?,?,?,?, ?,?,?)""",
            (
                r.record_id, r.experiment_run, _iso(r.timestamp), r.model_used,
                r.market_id, r.market_question, r.market_type, r.resolution_window_hours,
                _iso(r.resolution_datetime), r.tier,
                r.grok_raw_probability, r.grok_raw_confidence, r.grok_reasoning,
                json.dumps(r.grok_signal_types), r.headline_only_signal,
                r.calibration_adjustment, r.market_type_adjustment, r.signal_weight_adjustment,
                r.final_adjusted_probability, r.final_adjusted_confidence,
                r.market_price_at_decision, r.orderbook_depth_usd, r.fee_rate,
                r.calculated_edge, r.trade_score,
                r.action, r.skip_reason, r.position_size_usd, r.kelly_fraction_used,
                r.market_cluster_id,
                r.actual_outcome, r.pnl, r.brier_score_raw, r.brier_score_adjusted,
                _iso(r.resolved_at), r.unrealized_adverse_move, r.voided, r.void_reason,
            ),
        )
        await self._conn.commit()

    def _row_to_trade(self, row) -> TradeRecord:
        return TradeRecord(
            record_id=row["record_id"],
            experiment_run=row["experiment_run"],
            timestamp=_parse_dt(row["timestamp"]) or _utcnow(),
            model_used=row["model_used"],
            market_id=row["market_id"],
            market_question=row["market_question"],
            market_type=row["market_type"],
            resolution_window_hours=row["resolution_window_hours"] or 0.0,
            tier=row["tier"],
            grok_raw_probability=row["grok_raw_probability"],
            grok_raw_confidence=row["grok_raw_confidence"],
            grok_reasoning=row["grok_reasoning"] or "",
            grok_signal_types=json.loads(row["grok_signal_types"] or "[]"),
            headline_only_signal=bool(row["headline_only_signal"]),
            calibration_adjustment=row["calibration_adjustment"] or 0.0,
            market_type_adjustment=row["market_type_adjustment"] or 0.0,
            signal_weight_adjustment=row["signal_weight_adjustment"] or 0.0,
            final_adjusted_probability=row["final_adjusted_probability"],
            final_adjusted_confidence=row["final_adjusted_confidence"],
            market_price_at_decision=row["market_price_at_decision"],
            orderbook_depth_usd=row["orderbook_depth_usd"] or 0.0,
            fee_rate=row["fee_rate"],
            calculated_edge=row["calculated_edge"],
            trade_score=row["trade_score"] or 0.0,
            action=row["action"],
            skip_reason=row["skip_reason"],
            position_size_usd=row["position_size_usd"] or 0.0,
            kelly_fraction_used=row["kelly_fraction_used"] or 0.0,
            market_cluster_id=row["market_cluster_id"],
            actual_outcome=row["actual_outcome"] if row["actual_outcome"] is not None else None,
            pnl=row["pnl"],
            brier_score_raw=row["brier_score_raw"],
            brier_score_adjusted=row["brier_score_adjusted"],
            resolved_at=_parse_dt(row["resolved_at"]),
            resolution_datetime=_parse_dt(row["resolution_datetime"]),
            unrealized_adverse_move=row["unrealized_adverse_move"],
            voided=bool(row["voided"]),
            void_reason=row["void_reason"],
        )

    async def get_trade(self, record_id: str) -> Optional[TradeRecord]:
        cursor = await self._conn.execute(
            "SELECT * FROM trade_records WHERE record_id = ?", (record_id,)
        )
        row = await cursor.fetchone()
        return self._row_to_trade(row) if row else None

    async def get_open_trades(self) -> List[TradeRecord]:
        cursor = await self._conn.execute(
            "SELECT * FROM trade_records WHERE actual_outcome IS NULL AND voided = FALSE AND action != 'SKIP'"
        )
        rows = await cursor.fetchall()
        return [self._row_to_trade(r) for r in rows]

    async def get_today_trades(self) -> List[TradeRecord]:
        today = _utcnow().strftime("%Y-%m-%d")
        cursor = await self._conn.execute(
            "SELECT * FROM trade_records WHERE date(timestamp) = ? ORDER BY timestamp",
            (today,),
        )
        rows = await cursor.fetchall()
        return [self._row_to_trade(r) for r in rows]

    async def get_week_trades(self) -> List[TradeRecord]:
        week_ago = (_utcnow() - timedelta(days=7)).isoformat()
        cursor = await self._conn.execute(
            "SELECT * FROM trade_records WHERE timestamp >= ? ORDER BY timestamp",
            (week_ago,),
        )
        rows = await cursor.fetchall()
        return [self._row_to_trade(r) for r in rows]

    async def update_trade(self, r: TradeRecord) -> None:
        await self._conn.execute(
            """UPDATE trade_records SET
                actual_outcome=?, pnl=?, brier_score_raw=?, brier_score_adjusted=?,
                resolved_at=?, unrealized_adverse_move=?, voided=?, void_reason=?
            WHERE record_id=?""",
            (
                r.actual_outcome, r.pnl, r.brier_score_raw, r.brier_score_adjusted,
                _iso(r.resolved_at), r.unrealized_adverse_move, r.voided, r.void_reason,
                r.record_id,
            ),
        )
        await self._conn.commit()

    async def count_today_trades(self) -> int:
        today = _utcnow().strftime("%Y-%m-%d")
        cursor = await self._conn.execute(
            "SELECT COUNT(*) FROM trade_records WHERE date(timestamp) = ? AND action != 'SKIP'",
            (today,),
        )
        row = await cursor.fetchone()
        return row[0] if row else 0

    async def count_open_trades(self) -> int:
        cursor = await self._conn.execute(
            "SELECT COUNT(*) FROM trade_records WHERE actual_outcome IS NULL AND voided = FALSE AND action != 'SKIP'"
        )
        row = await cursor.fetchone()
        return row[0] if row else 0

    async def get_all_resolved_trades(self, include_voided: bool = False) -> List[TradeRecord]:
        sql = "SELECT * FROM trade_records WHERE actual_outcome IS NOT NULL"
        if not include_voided:
            sql += " AND voided = FALSE"
        sql += " ORDER BY timestamp"
        cursor = await self._conn.execute(sql)
        rows = await cursor.fetchall()
        return [self._row_to_trade(r) for r in rows]

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    async def load_calibration(self) -> List[CalibrationBucket]:
        buckets = []
        for br in CALIBRATION_BUCKET_RANGES:
            key = f"{br[0]}-{br[1]}"
            cursor = await self._conn.execute(
                "SELECT alpha, beta FROM calibration_state WHERE bucket_range = ?",
                (key,),
            )
            row = await cursor.fetchone()
            if row:
                buckets.append(CalibrationBucket(br, alpha=row["alpha"], beta=row["beta"]))
            else:
                buckets.append(CalibrationBucket(br))
        return buckets

    async def save_calibration(self, buckets: List[CalibrationBucket]) -> None:
        now = _utcnow().isoformat()
        for b in buckets:
            key = f"{b.bucket_range[0]}-{b.bucket_range[1]}"
            await self._conn.execute(
                """INSERT INTO calibration_state (bucket_range, alpha, beta, updated_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(bucket_range) DO UPDATE SET alpha=?, beta=?, updated_at=?""",
                (key, b.alpha, b.beta, now, b.alpha, b.beta, now),
            )
        await self._conn.commit()

    # ------------------------------------------------------------------
    # Market Type Performance
    # ------------------------------------------------------------------

    async def load_market_type_performance(self) -> Dict[str, MarketTypePerformance]:
        cursor = await self._conn.execute("SELECT * FROM market_type_performance")
        rows = await cursor.fetchall()
        result = {}
        for row in rows:
            scores = json.loads(row["brier_scores"] or "[]")
            result[row["market_type"]] = MarketTypePerformance(
                market_type=row["market_type"],
                total_trades=row["total_trades"],
                total_pnl=row["total_pnl"],
                brier_scores=scores,
                total_observed=row["total_observed"],
                counterfactual_pnl=row["counterfactual_pnl"],
            )
        return result

    async def save_market_type_performance(
        self, perfs: Dict[str, MarketTypePerformance]
    ) -> None:
        now = _utcnow().isoformat()
        for mtype, p in perfs.items():
            await self._conn.execute(
                """INSERT INTO market_type_performance
                   (market_type, total_trades, total_pnl, brier_scores, total_observed, counterfactual_pnl, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(market_type) DO UPDATE SET
                   total_trades=?, total_pnl=?, brier_scores=?, total_observed=?, counterfactual_pnl=?, updated_at=?""",
                (
                    mtype, p.total_trades, p.total_pnl, json.dumps(p.brier_scores),
                    p.total_observed, p.counterfactual_pnl, now,
                    p.total_trades, p.total_pnl, json.dumps(p.brier_scores),
                    p.total_observed, p.counterfactual_pnl, now,
                ),
            )
        await self._conn.commit()

    # ------------------------------------------------------------------
    # Signal Trackers
    # ------------------------------------------------------------------

    async def load_signal_trackers(self) -> Dict[Tuple[str, str, str], SignalTracker]:
        cursor = await self._conn.execute("SELECT * FROM signal_trackers")
        rows = await cursor.fetchall()
        result = {}
        for row in rows:
            key = (row["source_tier"], row["info_type"], row["market_type"])
            result[key] = SignalTracker(
                source_tier=row["source_tier"],
                info_type=row["info_type"],
                market_type=row["market_type"],
                present_in_winning_trades=row["present_winning"],
                present_in_losing_trades=row["present_losing"],
                absent_in_winning_trades=row["absent_winning"],
                absent_in_losing_trades=row["absent_losing"],
            )
        return result

    async def save_signal_trackers(
        self, trackers: Dict[Tuple[str, str, str], SignalTracker]
    ) -> None:
        now = _utcnow().isoformat()
        for _key, t in trackers.items():
            await self._conn.execute(
                """INSERT INTO signal_trackers
                   (source_tier, info_type, market_type, present_winning, present_losing,
                    absent_winning, absent_losing, last_updated)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(source_tier, info_type, market_type) DO UPDATE SET
                   present_winning=?, present_losing=?, absent_winning=?, absent_losing=?, last_updated=?""",
                (
                    t.source_tier, t.info_type, t.market_type,
                    t.present_in_winning_trades, t.present_in_losing_trades,
                    t.absent_in_winning_trades, t.absent_in_losing_trades, now,
                    t.present_in_winning_trades, t.present_in_losing_trades,
                    t.absent_in_winning_trades, t.absent_in_losing_trades, now,
                ),
            )
        await self._conn.commit()

    # ------------------------------------------------------------------
    # Experiment Runs
    # ------------------------------------------------------------------

    async def save_experiment(self, run: ExperimentRun) -> None:
        await self._conn.execute(
            """INSERT INTO experiment_runs
               (run_id, started_at, ended_at, config_snapshot, description, model_used,
                include_in_learning, total_trades, total_pnl, avg_brier, sharpe_ratio)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run.run_id, _iso(run.started_at), _iso(run.ended_at),
                json.dumps(run.config_snapshot), run.description, run.model_used,
                run.include_in_learning, run.total_trades, run.total_pnl,
                run.avg_brier, run.sharpe_ratio,
            ),
        )
        await self._conn.commit()

    async def get_current_experiment(self) -> Optional[ExperimentRun]:
        cursor = await self._conn.execute(
            "SELECT * FROM experiment_runs WHERE ended_at IS NULL ORDER BY started_at DESC LIMIT 1"
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return ExperimentRun(
            run_id=row["run_id"],
            started_at=_parse_dt(row["started_at"]) or _utcnow(),
            ended_at=_parse_dt(row["ended_at"]),
            config_snapshot=json.loads(row["config_snapshot"] or "{}"),
            description=row["description"] or "",
            model_used=row["model_used"],
            include_in_learning=bool(row["include_in_learning"]),
            total_trades=row["total_trades"],
            total_pnl=row["total_pnl"],
            avg_brier=row["avg_brier"],
            sharpe_ratio=row["sharpe_ratio"],
        )

    async def end_experiment(self, run_id: str, stats: dict) -> None:
        now = _utcnow().isoformat()
        await self._conn.execute(
            """UPDATE experiment_runs SET ended_at=?, total_trades=?, total_pnl=?,
               avg_brier=?, sharpe_ratio=? WHERE run_id=?""",
            (
                now, stats.get("total_trades", 0), stats.get("total_pnl", 0.0),
                stats.get("avg_brier", 0.0), stats.get("sharpe_ratio", 0.0), run_id,
            ),
        )
        await self._conn.commit()

    # ------------------------------------------------------------------
    # Portfolio
    # ------------------------------------------------------------------

    async def load_portfolio(self) -> Portfolio:
        cursor = await self._conn.execute("SELECT * FROM portfolio WHERE id = 1")
        row = await cursor.fetchone()
        if not row:
            return Portfolio()
        return Portfolio(
            cash_balance=row["cash_balance"],
            total_equity=row["total_equity"],
            total_pnl=row["total_pnl"],
            peak_equity=row["peak_equity"],
            max_drawdown=row["max_drawdown"],
        )

    async def save_portfolio(self, p: Portfolio) -> None:
        now = _utcnow().isoformat()
        await self._conn.execute(
            """INSERT INTO portfolio (id, cash_balance, total_equity, total_pnl, peak_equity, max_drawdown, updated_at)
               VALUES (1, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
               cash_balance=?, total_equity=?, total_pnl=?, peak_equity=?, max_drawdown=?, updated_at=?""",
            (
                p.cash_balance, p.total_equity, p.total_pnl, p.peak_equity, p.max_drawdown, now,
                p.cash_balance, p.total_equity, p.total_pnl, p.peak_equity, p.max_drawdown, now,
            ),
        )
        await self._conn.commit()

    # ------------------------------------------------------------------
    # API Costs
    # ------------------------------------------------------------------

    async def increment_api_cost(
        self, service: str, tokens_in: int = 0, tokens_out: int = 0
    ) -> None:
        today = _utcnow().strftime("%Y-%m-%d")
        # Estimate cost (rough pricing for Grok 4.1 Fast)
        cost = 0.0
        if service == "grok":
            cost = tokens_in * 0.000005 + tokens_out * 0.000025  # approx
        elif service == "twitter":
            cost = 0.0075  # per search
        await self._conn.execute(
            """INSERT INTO api_costs (date, service, calls, tokens_in, tokens_out, cost_usd)
               VALUES (?, ?, 1, ?, ?, ?)
               ON CONFLICT(date, service) DO UPDATE SET
               calls = calls + 1, tokens_in = tokens_in + ?, tokens_out = tokens_out + ?,
               cost_usd = cost_usd + ?""",
            (today, service, tokens_in, tokens_out, cost, tokens_in, tokens_out, cost),
        )
        await self._conn.commit()

    async def get_today_api_spend(self) -> float:
        today = _utcnow().strftime("%Y-%m-%d")
        cursor = await self._conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) FROM api_costs WHERE date = ?",
            (today,),
        )
        row = await cursor.fetchone()
        return row[0] if row else 0.0

    # ------------------------------------------------------------------
    # Parse Failures
    # ------------------------------------------------------------------

    async def record_parse_failure(self, market_id: str) -> None:
        await self._conn.execute(
            "INSERT INTO parse_failures (market_id) VALUES (?)", (market_id,)
        )
        await self._conn.commit()

    # ------------------------------------------------------------------
    # Model Swaps
    # ------------------------------------------------------------------

    async def save_model_swap(self, event: ModelSwapEvent) -> None:
        await self._conn.execute(
            """INSERT INTO model_swaps (timestamp, old_model, new_model, reason, experiment_run_started)
               VALUES (?, ?, ?, ?, ?)""",
            (
                _iso(event.timestamp), event.old_model, event.new_model,
                event.reason, event.experiment_run_started,
            ),
        )
        await self._conn.commit()
