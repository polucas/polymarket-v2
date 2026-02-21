import uuid
from datetime import datetime, timezone

import pytest
import pytest_asyncio

from src.db.migrations import run_migrations
from src.db.sqlite import Database
from src.models import Signal, TradeRecord


@pytest_asyncio.fixture
async def db():
    """In-memory SQLite database with schema applied."""
    database = await Database.init(":memory:")
    await run_migrations(database)
    yield database
    await database.close()


@pytest.fixture
def sample_trade_record():
    """Factory for TradeRecord with sensible defaults."""
    def _make(**overrides):
        defaults = {
            "record_id": str(uuid.uuid4()),
            "experiment_run": "test-run-001",
            "timestamp": datetime.now(timezone.utc),
            "model_used": "grok-3-fast",
            "market_id": "market-001",
            "market_question": "Will X happen?",
            "market_type": "political",
            "resolution_window_hours": 12.0,
            "tier": 1,
            "grok_raw_probability": 0.75,
            "grok_raw_confidence": 0.80,
            "grok_reasoning": "Test reasoning",
            "grok_signal_types": [{"source_tier": "S2", "info_type": "I2", "content": "test"}],
            "calibration_adjustment": 0.0,
            "market_type_adjustment": 0.0,
            "signal_weight_adjustment": 0.0,
            "final_adjusted_probability": 0.73,
            "final_adjusted_confidence": 0.78,
            "market_price_at_decision": 0.60,
            "orderbook_depth_usd": 5000.0,
            "fee_rate": 0.02,
            "calculated_edge": 0.11,
            "trade_score": 0.05,
            "action": "BUY_YES",
            "skip_reason": None,
            "position_size_usd": 200.0,
            "kelly_fraction_used": 0.25,
            "actual_outcome": None,
            "pnl": None,
            "brier_score_raw": None,
            "brier_score_adjusted": None,
            "resolved_at": None,
            "unrealized_adverse_move": None,
            "voided": False,
            "void_reason": None,
        }
        defaults.update(overrides)
        return TradeRecord(**defaults)
    return _make


@pytest.fixture
def sample_signal():
    """Factory for Signal objects."""
    def _make(**overrides):
        defaults = {
            "source": "twitter",
            "source_tier": "S3",
            "info_type": None,
            "content": "Breaking: test signal content",
            "credibility": 0.80,
            "author": "TestAuthor",
            "followers": 50000,
            "engagement": 100,
            "timestamp": datetime.now(timezone.utc),
            "headline_only": False,
        }
        defaults.update(overrides)
        return Signal(**defaults)
    return _make
