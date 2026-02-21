from typing import Dict, Optional
from src.models import MarketTypePerformance, TradeRecord
from src.db.sqlite import Database


class MarketTypeManager:
    def __init__(self):
        self.performances: Dict[str, MarketTypePerformance] = {}
    
    def _ensure(self, market_type: str) -> MarketTypePerformance:
        if market_type not in self.performances:
            self.performances[market_type] = MarketTypePerformance(market_type=market_type)
        return self.performances[market_type]
    
    def update_market_type(self, record: TradeRecord, counterfactual_pnl: float = 0.0) -> None:
        """Update market-type stats with resolved trade. Uses ADJUSTED Brier score."""
        if record.actual_outcome is None or record.voided:
            return
        
        mtype = self._ensure(record.market_type)
        mtype.total_trades += 1
        
        if record.brier_score_adjusted is not None:
            mtype.brier_scores.append(record.brier_score_adjusted)
        
        if record.action != "SKIP":
            mtype.total_pnl += record.pnl or 0.0
        else:
            mtype.total_observed += 1
            mtype.counterfactual_pnl += counterfactual_pnl
    
    def get_edge_adjustment(self, market_type: str) -> float:
        perf = self.performances.get(market_type)
        return perf.edge_adjustment if perf else 0.0
    
    def should_disable(self, market_type: str) -> bool:
        perf = self.performances.get(market_type)
        return perf.should_disable if perf else False
    
    def dampen_on_swap(self) -> None:
        """On model swap: keep only last 15 Brier scores per market type."""
        for perf in self.performances.values():
            perf.brier_scores = perf.brier_scores[-15:]
    
    async def load(self, db: Database) -> None:
        self.performances = await db.load_market_type_performance()
    
    async def save(self, db: Database) -> None:
        await db.save_market_type_performance(self.performances)
