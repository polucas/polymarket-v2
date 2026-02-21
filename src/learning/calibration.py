from typing import List, Optional
from datetime import datetime, timezone
from src.models import CalibrationBucket, TradeRecord, CALIBRATION_BUCKET_RANGES
from src.db.sqlite import Database


class CalibrationManager:
    def __init__(self):
        self.buckets: List[CalibrationBucket] = [
            CalibrationBucket(br) for br in CALIBRATION_BUCKET_RANGES
        ]
    
    def find_bucket(self, confidence: float) -> Optional[CalibrationBucket]:
        for b in self.buckets:
            if b.bucket_range[0] <= confidence < b.bucket_range[1]:
                return b
        # Handle edge: confidence == 1.0 goes to last bucket
        if confidence >= self.buckets[-1].bucket_range[0]:
            return self.buckets[-1]
        return self.buckets[0]  # Below 0.50 -> first bucket
    
    def get_correction(self, confidence: float) -> float:
        bucket = self.find_bucket(confidence)
        return bucket.get_correction() if bucket else 0.0
    
    def update_calibration(self, record: TradeRecord) -> None:
        """Update calibration with resolved trade.
        
        CRITICAL: Uses grok_raw_confidence to find bucket and grok_raw_probability
        to determine correctness. NOT final_adjusted values. Using adjusted would 
        create a self-referencing loop.
        """
        if record.actual_outcome is None or record.voided:
            return
        
        bucket = self.find_bucket(record.grok_raw_confidence)
        if not bucket:
            return
        
        # Use RAW probability for correctness
        raw_predicted_yes = record.grok_raw_probability > 0.5
        was_correct = raw_predicted_yes == record.actual_outcome
        
        # Recency weight: more recent trades matter more
        now = datetime.now(timezone.utc)
        if record.timestamp.tzinfo is None:
            days_since = (now.replace(tzinfo=None) - record.timestamp).total_seconds() / 86400
        else:
            days_since = (now - record.timestamp).total_seconds() / 86400
        recency = 0.95 ** max(0, days_since)
        
        bucket.update(was_correct, recency_weight=recency)
    
    def reset_to_priors(self) -> None:
        for b in self.buckets:
            b.alpha = 1.0
            b.beta = 1.0
    
    async def load(self, db: Database) -> None:
        loaded = await db.load_calibration()
        for i, b in enumerate(loaded):
            if i < len(self.buckets):
                self.buckets[i] = b
    
    async def save(self, db: Database) -> None:
        await db.save_calibration(self.buckets)
