"""Tests for duplicate bet prevention: cooldown and question-similarity dedup."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from src.db.migrations import run_migrations
from src.db.sqlite import Database
from src.engine.trade_ranker import keyword_overlap
from src.models import ExperimentRun, Market, Signal, TradeRecord
from src.config import Settings, MonkModeConfig
from src.scheduler import Scheduler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _make_experiment(run_id: str = "test-run-001") -> ExperimentRun:
    return ExperimentRun(
        run_id=run_id,
        started_at=_utcnow(),
        config_snapshot={},
        description="test",
        model_used="grok-3-fast",
    )


def _make_trade_record(
    market_id: str = "market-001",
    market_question: str = "Will X happen?",
    market_type: str = "political",
    action: str = "BUY_YES",
    timestamp: datetime = None,
    experiment_run: str = "test-run-001",
) -> TradeRecord:
    return TradeRecord(
        record_id=str(uuid.uuid4()),
        experiment_run=experiment_run,
        timestamp=timestamp or _utcnow(),
        model_used="grok-3-fast",
        market_id=market_id,
        market_question=market_question,
        market_type=market_type,
        resolution_window_hours=12.0,
        tier=1,
        grok_raw_probability=0.75,
        grok_raw_confidence=0.80,
        grok_reasoning="Test reasoning",
        grok_signal_types=[],
        final_adjusted_probability=0.73,
        final_adjusted_confidence=0.78,
        market_price_at_decision=0.60,
        fee_rate=0.02,
        calculated_edge=0.11,
        action=action,
        skip_reason=None if action != "SKIP" else "test_skip",
        position_size_usd=200.0,
        kelly_fraction_used=0.25,
    )


def _make_market(
    market_id: str = "market-001",
    question: str = "Will X happen?",
    market_type: str = "political",
    keywords: list = None,
) -> Market:
    return Market(
        market_id=market_id,
        question=question,
        yes_price=0.60,
        no_price=0.40,
        market_type=market_type,
        keywords=keywords or ["trump", "election"],
    )


def _build_scheduler() -> Scheduler:
    """Build a Scheduler with every external dependency mocked out."""
    settings = MagicMock(spec=Settings)
    settings.TIER1_SCAN_INTERVAL_MINUTES = 15
    settings.TIER2_SCAN_INTERVAL_MINUTES = 3
    settings.TIER1_MIN_EDGE = 0.04
    settings.TIER2_MIN_EDGE = 0.05
    settings.TIER1_FEE_RATE = 0.0
    settings.TIER2_FEE_RATE = 0.04
    settings.KELLY_FRACTION = 0.25
    settings.MAX_POSITION_PCT = 0.016
    settings.ENVIRONMENT = "paper"
    settings.TIER1_DAILY_CAP = 20
    settings.TIER2_DAILY_CAP = 3
    settings.DAILY_LOSS_LIMIT_PCT = 0.05
    settings.WEEKLY_LOSS_LIMIT_PCT = 0.10
    settings.CONSECUTIVE_LOSS_COOLDOWN = 3
    settings.COOLDOWN_DURATION_HOURS = 2.0
    settings.DAILY_API_BUDGET_USD = 8.0
    settings.MAX_TOTAL_EXPOSURE_PCT = 0.30
    settings.MARKET_COOLDOWN_HOURS = 24.0
    settings.QUESTION_SIMILARITY_THRESHOLD = 0.60
    settings.GROK_MODEL = "grok-4.20-experimental-beta-0304-reasoning"
    settings.MARKET_FETCH_LIMIT = 200
    settings.MARKET_PAGE_SIZE = 500
    settings.MARKET_FETCH_PAGES = 3
    settings.MIN_TRADEABLE_PRICE = 0.05
    settings.MAX_TRADEABLE_PRICE = 0.95

    db = AsyncMock()
    polymarket = AsyncMock()
    twitter = AsyncMock()
    rss = AsyncMock()
    grok = AsyncMock()
    calibration_mgr = MagicMock()
    market_type_mgr = MagicMock()
    signal_tracker_mgr = MagicMock()

    with patch.object(MonkModeConfig, "from_settings", return_value=MonkModeConfig()):
        scheduler = Scheduler(
            settings=settings,
            db=db,
            polymarket=polymarket,
            twitter=twitter,
            rss=rss,
            grok=grok,
            calibration_mgr=calibration_mgr,
            market_type_mgr=market_type_mgr,
            signal_tracker_mgr=signal_tracker_mgr,
        )
    return scheduler


# ---------------------------------------------------------------------------
# 1. DB query: get_recently_traded_market_ids
# ---------------------------------------------------------------------------

class TestGetRecentlyTradedMarketIds:
    """Verify DB query returns the correct set of recently-traded market IDs."""

    @pytest.mark.asyncio
    async def test_returns_executed_markets_within_window(self, db):
        """Trades within the cooldown window appear in the result."""
        experiment = _make_experiment()
        await db.save_experiment(experiment)

        trade = _make_trade_record(
            market_id="market-abc",
            action="BUY_YES",
            timestamp=_utcnow() - timedelta(hours=6),
        )
        await db.save_trade(trade)

        result = await db.get_recently_traded_market_ids(cooldown_hours=24.0)
        assert "market-abc" in result

    @pytest.mark.asyncio
    async def test_excludes_skip_records(self, db):
        """SKIP records are not returned even if within the window."""
        experiment = _make_experiment()
        await db.save_experiment(experiment)

        skip_trade = _make_trade_record(
            market_id="market-skip",
            action="SKIP",
            timestamp=_utcnow() - timedelta(hours=1),
        )
        await db.save_trade(skip_trade)

        result = await db.get_recently_traded_market_ids(cooldown_hours=24.0)
        assert "market-skip" not in result

    @pytest.mark.asyncio
    async def test_excludes_trades_outside_window(self, db):
        """Trades older than the cooldown window are not returned."""
        experiment = _make_experiment()
        await db.save_experiment(experiment)

        old_trade = _make_trade_record(
            market_id="market-old",
            action="BUY_YES",
            timestamp=_utcnow() - timedelta(hours=30),
        )
        await db.save_trade(old_trade)

        result = await db.get_recently_traded_market_ids(cooldown_hours=24.0)
        assert "market-old" not in result

    @pytest.mark.asyncio
    async def test_returns_empty_set_when_no_trades(self, db):
        """Empty DB returns empty set."""
        result = await db.get_recently_traded_market_ids(cooldown_hours=24.0)
        assert result == set()

    @pytest.mark.asyncio
    async def test_deduplicates_same_market_multiple_trades(self, db):
        """Same market traded twice appears only once in the result."""
        experiment = _make_experiment()
        await db.save_experiment(experiment)

        for _ in range(2):
            trade = _make_trade_record(
                market_id="market-dup",
                action="BUY_YES",
                timestamp=_utcnow() - timedelta(hours=1),
            )
            await db.save_trade(trade)

        result = await db.get_recently_traded_market_ids(cooldown_hours=24.0)
        assert result == {"market-dup"}


# ---------------------------------------------------------------------------
# 2. Cooldown blocks a recently traded market
# ---------------------------------------------------------------------------

class TestCooldownBlocksRecentlyTraded:
    """A market traded 12h ago with a 24h cooldown must be blocked."""

    @pytest.mark.asyncio
    async def test_cooldown_blocks_recently_traded(self, db):
        experiment = _make_experiment()
        await db.save_experiment(experiment)

        trade = _make_trade_record(
            market_id="market-001",
            action="BUY_YES",
            timestamp=_utcnow() - timedelta(hours=12),
        )
        await db.save_trade(trade)

        recently_traded = await db.get_recently_traded_market_ids(cooldown_hours=24.0)
        assert "market-001" in recently_traded


# ---------------------------------------------------------------------------
# 3. Cooldown allows market after cooldown expiry
# ---------------------------------------------------------------------------

class TestCooldownAllowsAfterExpiry:
    """A market traded 30h ago with a 24h cooldown must NOT be blocked."""

    @pytest.mark.asyncio
    async def test_cooldown_allows_after_expiry(self, db):
        experiment = _make_experiment()
        await db.save_experiment(experiment)

        trade = _make_trade_record(
            market_id="market-001",
            action="BUY_YES",
            timestamp=_utcnow() - timedelta(hours=30),
        )
        await db.save_trade(trade)

        recently_traded = await db.get_recently_traded_market_ids(cooldown_hours=24.0)
        assert "market-001" not in recently_traded


# ---------------------------------------------------------------------------
# 4. _process_market: cooldown skip is saved when market in recently_traded_ids
# ---------------------------------------------------------------------------

class TestProcessMarketCooldownSkip:
    """When a market_id is in recently_traded_ids, _process_market must save a SKIP
    with reason 'market_cooldown' and return without calling Grok."""

    @pytest.mark.asyncio
    async def test_cooldown_skip_saved(self):
        scheduler = _build_scheduler()

        market = _make_market(market_id="market-cool")
        candidates = []
        all_skips = []

        scheduler._db.save_trade = AsyncMock()

        await scheduler._process_market(
            market=market,
            rss_signals=[],
            scan_mode="active",
            candidates=candidates,
            all_skips=all_skips,
            today_trades=[],
            experiment_run="test-run-001",
            tier=1,
            open_market_ids=frozenset(),
            recently_traded_ids={"market-cool"},
            recent_questions=[],
        )

        assert scheduler._db.save_trade.called
        saved_record = scheduler._db.save_trade.call_args[0][0]
        assert saved_record.action == "SKIP"
        assert saved_record.skip_reason == "market_cooldown"
        assert saved_record.market_id == "market-cool"
        # Grok should NOT have been called
        scheduler._grok.call_grok_with_retry.assert_not_called()
        assert candidates == []


# ---------------------------------------------------------------------------
# 5. _process_market: similar question is blocked
# ---------------------------------------------------------------------------

class TestSimilarQuestionBlocked:
    """High keyword overlap with a recently traded market of the same type → blocked."""

    @pytest.mark.asyncio
    async def test_similar_question_blocked(self):
        scheduler = _build_scheduler()

        # Market to evaluate: "Will Trump win the election?" with matching keywords
        market = _make_market(
            market_id="market-new",
            question="Will Trump win the election?",
            market_type="political",
            keywords=["trump", "election", "vote"],
        )
        candidates = []
        all_skips = []

        # Recent question with same type and high keyword overlap
        # Splitting "Will Trump win the election?" → words > 3 chars: ["will", "trump", "election"]
        # Actually: "Will"(4), "Trump"(5), "win"(3-skip), "the"(3-skip), "election"(8) → ["Will", "Trump", "election"]
        # After lower: ["will", "trump", "election"]
        # market.keywords: ["trump", "election", "vote"]
        # set_a = {"trump","election","vote"}, set_b = {"will","trump","election"}
        # intersection = {"trump","election"} = 2, union = {"trump","election","vote","will"} = 4
        # Jaccard = 2/4 = 0.50, which is >= 0.60? No — need higher overlap.
        # Use a question that gives higher overlap:
        # "Trump election vote outcome" → words > 3: ["Trump","election","vote","outcome"]
        # lower: {"trump","election","vote","outcome"}
        # intersection with {"trump","election","vote"} = {"trump","election","vote"} = 3
        # union = {"trump","election","vote","outcome"} = 4 → Jaccard = 3/4 = 0.75 >= 0.60 ✓
        recent_questions = [
            ("market-prev", "Trump election vote outcome", "political"),
        ]

        scheduler._db.save_trade = AsyncMock()
        scheduler._market_type_mgr.should_disable.return_value = False

        await scheduler._process_market(
            market=market,
            rss_signals=[],
            scan_mode="active",
            candidates=candidates,
            all_skips=all_skips,
            today_trades=[],
            experiment_run="test-run-001",
            tier=1,
            open_market_ids=frozenset(),
            recently_traded_ids=frozenset(),
            recent_questions=recent_questions,
        )

        assert scheduler._db.save_trade.called
        saved_record = scheduler._db.save_trade.call_args[0][0]
        assert saved_record.action == "SKIP"
        assert "similar_to_market-prev" in saved_record.skip_reason
        scheduler._grok.call_grok_with_retry.assert_not_called()
        assert candidates == []


# ---------------------------------------------------------------------------
# 6. _process_market: different question (low overlap) is allowed through
# ---------------------------------------------------------------------------

class TestDifferentQuestionAllowed:
    """Low keyword overlap with recent questions → market is NOT blocked by similarity check."""

    @pytest.mark.asyncio
    async def test_different_question_allowed(self):
        scheduler = _build_scheduler()

        market = _make_market(
            market_id="market-new",
            question="Will the Fed cut rates?",
            market_type="economic",
            keywords=["fed", "rates", "cut"],
        )
        candidates = []
        all_skips = []

        # Completely different question — no overlap with market keywords
        recent_questions = [
            ("market-prev", "Will Trump win the election?", "economic"),
        ]

        scheduler._db.save_trade = AsyncMock()
        scheduler._market_type_mgr.should_disable.return_value = False
        # Make Grok fail so we get a grok_failed skip (proving we passed the dedup checks)
        scheduler._grok.call_grok_with_retry = AsyncMock(return_value=None)
        scheduler._twitter.get_signals_for_market = AsyncMock(return_value=[])
        scheduler._polymarket.get_orderbook = AsyncMock(
            return_value=MagicMock(bids=[], asks=[])
        )

        await scheduler._process_market(
            market=market,
            rss_signals=[],
            scan_mode="active",
            candidates=candidates,
            all_skips=all_skips,
            today_trades=[],
            experiment_run="test-run-001",
            tier=1,
            open_market_ids=frozenset(),
            recently_traded_ids=frozenset(),
            recent_questions=recent_questions,
        )

        # Should reach Grok (not blocked by similarity)
        scheduler._grok.call_grok_with_retry.assert_called_once()

        # Saved skip should be grok_failed, not similar_to_*
        saved_record = scheduler._db.save_trade.call_args[0][0]
        assert saved_record.skip_reason == "grok_failed"


# ---------------------------------------------------------------------------
# 7. _process_market: same keywords but different market_type → NOT blocked
# ---------------------------------------------------------------------------

class TestDifferentMarketTypeNotCompared:
    """Even if keyword overlap is high, different market_type prevents similarity blocking."""

    @pytest.mark.asyncio
    async def test_different_market_type_not_compared(self):
        scheduler = _build_scheduler()

        market = _make_market(
            market_id="market-new",
            question="Will Trump win the election?",
            market_type="political",
            keywords=["trump", "election", "vote"],
        )
        candidates = []
        all_skips = []

        # Same question text but different market_type → should NOT trigger similarity block
        recent_questions = [
            ("market-prev", "Trump election vote outcome", "economic"),
        ]

        scheduler._db.save_trade = AsyncMock()
        scheduler._market_type_mgr.should_disable.return_value = False
        scheduler._grok.call_grok_with_retry = AsyncMock(return_value=None)
        scheduler._twitter.get_signals_for_market = AsyncMock(return_value=[])
        scheduler._polymarket.get_orderbook = AsyncMock(
            return_value=MagicMock(bids=[], asks=[])
        )

        await scheduler._process_market(
            market=market,
            rss_signals=[],
            scan_mode="active",
            candidates=candidates,
            all_skips=all_skips,
            today_trades=[],
            experiment_run="test-run-001",
            tier=1,
            open_market_ids=frozenset(),
            recently_traded_ids=frozenset(),
            recent_questions=recent_questions,
        )

        # Should reach Grok — different market_type means no comparison was made
        scheduler._grok.call_grok_with_retry.assert_called_once()

        # Skip reason should be grok_failed, not similar_to_*
        saved_record = scheduler._db.save_trade.call_args[0][0]
        assert saved_record.skip_reason == "grok_failed"


# ---------------------------------------------------------------------------
# 8. Unit test: keyword_overlap function
# ---------------------------------------------------------------------------

class TestKeywordOverlapFunction:
    """Unit tests for the public keyword_overlap() function."""

    def test_identical_lists_returns_one(self):
        assert keyword_overlap(["trump", "election"], ["trump", "election"]) == 1.0

    def test_disjoint_lists_returns_zero(self):
        assert keyword_overlap(["trump", "election"], ["fed", "rates"]) == 0.0

    def test_empty_list_returns_zero(self):
        assert keyword_overlap([], ["trump"]) == 0.0
        assert keyword_overlap(["trump"], []) == 0.0
        assert keyword_overlap([], []) == 0.0

    def test_partial_overlap_jaccard(self):
        # intersection={"trump"}, union={"trump","election","vote"} → 1/3
        result = keyword_overlap(["trump", "election"], ["trump", "vote"])
        assert abs(result - 1.0 / 3.0) < 1e-6

    def test_case_insensitive(self):
        # Uppercase and lowercase should match
        result = keyword_overlap(["Trump", "Election"], ["trump", "election"])
        assert result == 1.0

    def test_three_quarters_overlap(self):
        # intersection={"a","b","c"}, union={"a","b","c","d"} → 3/4
        result = keyword_overlap(["a", "b", "c"], ["a", "b", "c", "d"])
        assert abs(result - 0.75) < 1e-6

    def test_threshold_boundary_60_percent(self):
        # intersection={"trump","election","vote"}, union={"trump","election","vote","outcome"} → 3/4 = 0.75
        result = keyword_overlap(["trump", "election", "vote"], ["trump", "election", "vote", "outcome"])
        assert result >= 0.60

    def test_below_threshold_50_percent(self):
        # intersection={"fed","rates"}, union={"fed","rates","fomc","cut"} → 2/4 = 0.50
        result = keyword_overlap(["fed", "rates", "fomc"], ["fed", "rates", "cut"])
        assert result == 0.50
        assert result < 0.60


# ---------------------------------------------------------------------------
# 9. get_recent_market_questions DB query
# ---------------------------------------------------------------------------

class TestGetRecentMarketQuestions:
    """Verify get_recent_market_questions returns the correct tuples."""

    @pytest.mark.asyncio
    async def test_returns_recent_executed_trades(self, db):
        experiment = _make_experiment()
        await db.save_experiment(experiment)

        trade = _make_trade_record(
            market_id="market-q",
            market_question="Will X happen?",
            market_type="political",
            action="BUY_YES",
            timestamp=_utcnow() - timedelta(hours=6),
        )
        await db.save_trade(trade)

        result = await db.get_recent_market_questions(hours=24.0)
        market_ids = [r[0] for r in result]
        assert "market-q" in market_ids
        # Check tuple structure: (market_id, question, market_type)
        matching = [r for r in result if r[0] == "market-q"]
        assert matching[0][1] == "Will X happen?"
        assert matching[0][2] == "political"

    @pytest.mark.asyncio
    async def test_excludes_skip_records(self, db):
        experiment = _make_experiment()
        await db.save_experiment(experiment)

        skip = _make_trade_record(
            market_id="market-skip-q",
            action="SKIP",
            timestamp=_utcnow() - timedelta(hours=1),
        )
        await db.save_trade(skip)

        result = await db.get_recent_market_questions(hours=24.0)
        market_ids = [r[0] for r in result]
        assert "market-skip-q" not in market_ids

    @pytest.mark.asyncio
    async def test_excludes_old_trades(self, db):
        experiment = _make_experiment()
        await db.save_experiment(experiment)

        old = _make_trade_record(
            market_id="market-old-q",
            action="BUY_YES",
            timestamp=_utcnow() - timedelta(hours=30),
        )
        await db.save_trade(old)

        result = await db.get_recent_market_questions(hours=24.0)
        market_ids = [r[0] for r in result]
        assert "market-old-q" not in market_ids
