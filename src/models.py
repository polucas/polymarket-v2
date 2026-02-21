from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SOURCE_TIER_CREDIBILITY: Dict[str, float] = {
    "S1": 0.95,
    "S2": 0.90,
    "S3": 0.80,
    "S4": 0.65,
    "S5": 0.70,
    "S6": 0.30,
}

CALIBRATION_BUCKET_RANGES: List[Tuple[float, float]] = [
    (0.50, 0.60),
    (0.60, 0.70),
    (0.70, 0.80),
    (0.80, 0.90),
    (0.90, 0.95),
    (0.95, 1.00),
]

# ---------------------------------------------------------------------------
# Data pipeline models
# ---------------------------------------------------------------------------


@dataclass
class Signal:
    source: str  # "twitter", "rss", "market_data"
    source_tier: str  # S1-S6
    info_type: Optional[str]  # I1-I6 or None (assigned by Grok)
    content: str
    credibility: float
    author: str
    followers: int
    engagement: int
    timestamp: Optional[datetime] = None
    headline_only: bool = False


@dataclass
class Market:
    market_id: str
    question: str
    yes_price: float
    no_price: float
    resolution_time: Optional[datetime] = None
    hours_to_resolution: float = 0.0
    volume_24h: float = 0.0
    liquidity: float = 0.0
    market_type: str = ""  # political, economic, crypto_15m, sports, cultural, regulatory
    fee_rate: float = 0.02
    keywords: List[str] = field(default_factory=list)
    resolved: bool = False
    resolution: Optional[str] = None  # "YES" or "NO" when resolved


@dataclass
class OrderBook:
    market_id: str
    bids: List[float] = field(default_factory=list)  # top 5 bid sizes
    asks: List[float] = field(default_factory=list)  # top 5 ask sizes
    timestamp: Optional[datetime] = None


@dataclass
class TradeCandidate:
    market: Market
    adjusted_probability: float = 0.0
    adjusted_confidence: float = 0.0
    calculated_edge: float = 0.0
    score: float = 0.0
    position_size: float = 0.0
    side: str = ""  # BUY_YES, BUY_NO, SKIP
    skip_reason: Optional[str] = None
    market_cluster_id: Optional[str] = None
    resolution_hours: float = 0.0
    signal_tags: List[dict] = field(default_factory=list)
    fee_rate: float = 0.02
    market_price: float = 0.0
    kelly_fraction_used: float = 0.0
    orderbook_depth: float = 0.0
    tier: int = 1
    # Raw Grok outputs preserved for trade record
    grok_raw_probability: float = 0.0
    grok_raw_confidence: float = 0.0
    grok_reasoning: str = ""
    grok_signal_types: List[dict] = field(default_factory=list)
    headline_only_signal: bool = False
    calibration_adjustment: float = 0.0
    market_type_adjustment: float = 0.0
    signal_weight_adjustment: float = 0.0


@dataclass
class ExecutionResult:
    executed_price: float
    slippage: float
    fill_probability: float
    filled: bool


@dataclass
class Position:
    market_id: str
    side: str  # BUY_YES or BUY_NO
    entry_price: float
    size_usd: float
    current_value: float = 0.0
    market_cluster_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Trade record (persisted per trade, including skips)
# ---------------------------------------------------------------------------


@dataclass
class TradeRecord:
    record_id: str
    experiment_run: str
    timestamp: datetime
    model_used: str

    market_id: str
    market_question: str
    market_type: str
    resolution_window_hours: float
    tier: int

    grok_raw_probability: float
    grok_raw_confidence: float
    grok_reasoning: str
    grok_signal_types: List[dict]
    headline_only_signal: bool = False

    calibration_adjustment: float = 0.0
    market_type_adjustment: float = 0.0
    signal_weight_adjustment: float = 0.0
    final_adjusted_probability: float = 0.0
    final_adjusted_confidence: float = 0.0

    market_price_at_decision: float = 0.0
    orderbook_depth_usd: float = 0.0
    fee_rate: float = 0.02
    calculated_edge: float = 0.0
    trade_score: float = 0.0

    action: str = "SKIP"  # BUY_YES, BUY_NO, SKIP
    skip_reason: Optional[str] = None
    position_size_usd: float = 0.0
    kelly_fraction_used: float = 0.0
    market_cluster_id: Optional[str] = None

    actual_outcome: Optional[bool] = None
    pnl: Optional[float] = None
    brier_score_raw: Optional[float] = None
    brier_score_adjusted: Optional[float] = None
    resolved_at: Optional[datetime] = None
    unrealized_adverse_move: Optional[float] = None

    voided: bool = False
    void_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Learning models
# ---------------------------------------------------------------------------


@dataclass
class CalibrationBucket:
    bucket_range: Tuple[float, float]
    alpha: float = 1.0
    beta: float = 1.0

    @property
    def expected_accuracy(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def sample_count(self) -> int:
        return int(self.alpha + self.beta - 2)

    @property
    def uncertainty(self) -> float:
        from scipy.stats import beta as beta_dist

        low, high = beta_dist.ppf([0.025, 0.975], self.alpha, self.beta)
        return high - low

    def update(self, was_correct: bool, recency_weight: float = 1.0) -> None:
        if was_correct:
            self.alpha += recency_weight
        else:
            self.beta += recency_weight

    def get_correction(self) -> float:
        if self.sample_count < 10:
            return 0.0
        bucket_midpoint = (self.bucket_range[0] + self.bucket_range[1]) / 2
        correction = self.expected_accuracy - bucket_midpoint
        certainty = max(0, 1 - self.uncertainty * 2)
        return correction * certainty


@dataclass
class MarketTypePerformance:
    market_type: str
    total_trades: int = 0
    total_pnl: float = 0.0
    brier_scores: List[float] = field(default_factory=list)
    total_observed: int = 0
    counterfactual_pnl: float = 0.0

    @property
    def avg_brier(self) -> float:
        if not self.brier_scores:
            return 0.25
        weights = [0.95**i for i in range(len(self.brier_scores) - 1, -1, -1)]
        return sum(b * w for b, w in zip(self.brier_scores, weights)) / sum(weights)

    @property
    def edge_adjustment(self) -> float:
        if self.total_trades < 15:
            return 0.0
        if self.avg_brier > 0.30:
            return 0.05
        elif self.avg_brier > 0.25:
            return 0.03
        elif self.avg_brier > 0.20:
            return 0.01
        return 0.0

    @property
    def should_disable(self) -> bool:
        return self.total_trades >= 30 and self.total_pnl < -0.15 * abs(
            self.total_trades
        )


@dataclass
class SignalTracker:
    source_tier: str  # S1-S6
    info_type: str  # I1-I6
    market_type: str
    present_in_winning_trades: int = 0
    present_in_losing_trades: int = 0
    absent_in_winning_trades: int = 0
    absent_in_losing_trades: int = 0

    @property
    def lift(self) -> float:
        total_present = self.present_in_winning_trades + self.present_in_losing_trades
        total_absent = self.absent_in_winning_trades + self.absent_in_losing_trades
        if total_present < 5 or total_absent < 5:
            return 1.0
        win_rate_present = self.present_in_winning_trades / total_present
        win_rate_absent = self.absent_in_winning_trades / total_absent
        if win_rate_absent == 0:
            return 1.0
        return win_rate_present / win_rate_absent

    @property
    def weight(self) -> float:
        raw = 1.0 + (self.lift - 1.0) * 0.3
        return max(0.8, min(1.2, raw))


# ---------------------------------------------------------------------------
# Experiment tracking
# ---------------------------------------------------------------------------


@dataclass
class ExperimentRun:
    run_id: str
    started_at: datetime
    ended_at: Optional[datetime] = None
    config_snapshot: dict = field(default_factory=dict)
    description: str = ""
    model_used: str = ""
    include_in_learning: bool = True
    total_trades: int = 0
    total_pnl: float = 0.0
    avg_brier: float = 0.0
    sharpe_ratio: float = 0.0


@dataclass
class ModelSwapEvent:
    timestamp: datetime
    old_model: str
    new_model: str
    reason: str
    experiment_run_started: str


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------


@dataclass
class Portfolio:
    cash_balance: float = 2000.0
    total_equity: float = 2000.0
    total_pnl: float = 0.0
    peak_equity: float = 2000.0
    max_drawdown: float = 0.0
    open_positions: List[Position] = field(default_factory=list)
