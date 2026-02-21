from typing import Dict, Set, Tuple
from src.models import SignalTracker, TradeRecord
from src.db.sqlite import Database


class SignalTrackerManager:
    def __init__(self):
        self.trackers: Dict[Tuple[str, str, str], SignalTracker] = {}
    
    def _ensure(self, source_tier: str, info_type: str, market_type: str) -> SignalTracker:
        key = (source_tier, info_type, market_type)
        if key not in self.trackers:
            self.trackers[key] = SignalTracker(source_tier=source_tier, info_type=info_type, market_type=market_type)
        return self.trackers[key]
    
    def get_all_observed_combos(self, market_type: str) -> Set[Tuple[str, str]]:
        """Get all (source_tier, info_type) combos ever observed for this market type."""
        return {(k[0], k[1]) for k in self.trackers if k[2] == market_type}
    
    def update_signal_trackers(self, record: TradeRecord) -> None:
        """Update signal trackers with resolved trade. Uses ADJUSTED correctness."""
        if record.actual_outcome is None or record.voided:
            return
        
        # Determine correctness using ADJUSTED probability (system-level accuracy)
        adjusted_predicted_yes = record.final_adjusted_probability > 0.5
        was_correct = adjusted_predicted_yes == record.actual_outcome
        
        # Get signal combos present in this trade
        present_combos: Set[Tuple[str, str]] = set()
        for tag in (record.grok_signal_types or []):
            st = tag.get("source_tier")
            it = tag.get("info_type")
            if st and it:
                present_combos.add((st, it))
        
        # Update all observed combos for this market type
        all_combos = self.get_all_observed_combos(record.market_type) | present_combos
        
        for combo in all_combos:
            tracker = self._ensure(combo[0], combo[1], record.market_type)
            present = combo in present_combos
            if present and was_correct:
                tracker.present_in_winning_trades += 1
            elif present and not was_correct:
                tracker.present_in_losing_trades += 1
            elif not present and was_correct:
                tracker.absent_in_winning_trades += 1
            else:
                tracker.absent_in_losing_trades += 1
    
    def get_signal_weight(self, source_tier: str, info_type: str, market_type: str) -> float:
        key = (source_tier, info_type, market_type)
        tracker = self.trackers.get(key)
        return tracker.weight if tracker else 1.0
    
    async def load(self, db: Database) -> None:
        self.trackers = await db.load_signal_trackers()
    
    async def save(self, db: Database) -> None:
        await db.save_signal_trackers(self.trackers)
