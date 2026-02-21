"""Tests for the Twitter data pipeline (TwitterDataPipeline).

Covers signal generation, filtering, deduplication, bot detection,
sorting, pagination, and error handling.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.config import Settings
from src.models import Signal, SOURCE_TIER_CREDIBILITY
from src.pipelines.twitter import TwitterDataPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tweet(
    text: str = "Bitcoin is surging today",
    screen_name: str = "CryptoAnalyst",
    followers_count: int = 5000,
    following_count: int = 200,
    engagement_score: int = 50,
    verified: bool = False,
    bio: str = "",
    created_at: str = "2026-02-21T12:00:00Z",
) -> dict:
    """Build a realistic raw tweet dict matching the Twitter API shape."""
    return {
        "text": text,
        "author": {
            "screen_name": screen_name,
            "name": screen_name,
            "followers_count": followers_count,
            "following_count": following_count,
            "friends_count": following_count,
            "verified": verified,
            "bio": bio,
            "description": bio,
        },
        "engagement_score": engagement_score,
        "created_at": created_at,
    }


def _build_pipeline() -> TwitterDataPipeline:
    """Create a TwitterDataPipeline with fake settings."""
    settings = Settings(TWITTER_API_KEY="test-api-key-123")
    return TwitterDataPipeline(settings)


def _mock_response(tweets: list, status_code: int = 200) -> httpx.Response:
    """Build a fake httpx.Response that returns the given tweets."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = {"tweets": tweets}
    resp.raise_for_status = MagicMock()
    return resp


# ---------------------------------------------------------------------------
# Test: get_signals_for_market returns List[Signal]
# ---------------------------------------------------------------------------


class TestGetSignalsForMarket:
    """Core happy-path tests for get_signals_for_market."""

    @pytest.mark.asyncio
    async def test_returns_list_of_signals(self):
        """get_signals_for_market(['bitcoin']) returns a list of Signal objects."""
        pipeline = _build_pipeline()
        tweets = [_make_tweet()]

        with patch.object(pipeline, "_search_tweets", new_callable=AsyncMock, return_value=tweets):
            with patch("src.pipelines.twitter.classify_source_tier", return_value="S6"):
                signals = await pipeline.get_signals_for_market(["bitcoin"])

        assert isinstance(signals, list)
        assert len(signals) >= 1
        assert all(isinstance(s, Signal) for s in signals)

    @pytest.mark.asyncio
    async def test_signal_fields(self):
        """Each Signal has source='twitter', source_tier in S1-S6, info_type=None."""
        pipeline = _build_pipeline()
        tweets = [_make_tweet(screen_name="analyst1", verified=True, bio="economist", followers_count=60000)]

        with patch.object(pipeline, "_search_tweets", new_callable=AsyncMock, return_value=tweets):
            with patch("src.pipelines.twitter.classify_source_tier", return_value="S3"):
                signals = await pipeline.get_signals_for_market(["bitcoin"])

        assert len(signals) == 1
        sig = signals[0]
        assert sig.source == "twitter"
        assert sig.source_tier in ("S1", "S2", "S3", "S4", "S5", "S6")
        assert sig.info_type is None


# ---------------------------------------------------------------------------
# Test: Follower filtering
# ---------------------------------------------------------------------------


class TestFollowerFiltering:
    """Tweets from accounts with < 1000 followers are excluded."""

    @pytest.mark.asyncio
    async def test_low_followers_excluded(self):
        """Author with 500 followers should be filtered out."""
        pipeline = _build_pipeline()
        tweets = [
            _make_tweet(screen_name="bigaccount", followers_count=5000, engagement_score=50),
            _make_tweet(screen_name="smallaccount", followers_count=500, engagement_score=50),
        ]

        with patch.object(pipeline, "_search_tweets", new_callable=AsyncMock, return_value=tweets):
            with patch("src.pipelines.twitter.classify_source_tier", return_value="S6"):
                signals = await pipeline.get_signals_for_market(["bitcoin"])

        authors = [s.author for s in signals]
        assert "bigaccount" in authors
        assert "smallaccount" not in authors

    @pytest.mark.asyncio
    async def test_exactly_1000_followers_included(self):
        """Author with exactly 1000 followers should pass the filter."""
        pipeline = _build_pipeline()
        tweets = [_make_tweet(screen_name="threshold", followers_count=1000, engagement_score=50)]

        with patch.object(pipeline, "_search_tweets", new_callable=AsyncMock, return_value=tweets):
            with patch("src.pipelines.twitter.classify_source_tier", return_value="S6"):
                signals = await pipeline.get_signals_for_market(["bitcoin"])

        assert len(signals) == 1
        assert signals[0].author == "threshold"


# ---------------------------------------------------------------------------
# Test: Engagement filtering
# ---------------------------------------------------------------------------


class TestEngagementFiltering:
    """Tweets with engagement_score < 10 are excluded."""

    @pytest.mark.asyncio
    async def test_low_engagement_excluded(self):
        """Tweet with engagement_score=5 should be filtered out."""
        pipeline = _build_pipeline()
        tweets = [
            _make_tweet(screen_name="popular", engagement_score=50),
            _make_tweet(screen_name="ignored", engagement_score=5),
        ]

        with patch.object(pipeline, "_search_tweets", new_callable=AsyncMock, return_value=tweets):
            with patch("src.pipelines.twitter.classify_source_tier", return_value="S6"):
                signals = await pipeline.get_signals_for_market(["bitcoin"])

        authors = [s.author for s in signals]
        assert "popular" in authors
        assert "ignored" not in authors

    @pytest.mark.asyncio
    async def test_exactly_10_engagement_included(self):
        """Tweet with exactly engagement_score=10 should pass the filter."""
        pipeline = _build_pipeline()
        tweets = [_make_tweet(screen_name="edge", engagement_score=10)]

        with patch.object(pipeline, "_search_tweets", new_callable=AsyncMock, return_value=tweets):
            with patch("src.pipelines.twitter.classify_source_tier", return_value="S6"):
                signals = await pipeline.get_signals_for_market(["bitcoin"])

        assert len(signals) == 1
        assert signals[0].author == "edge"


# ---------------------------------------------------------------------------
# Test: Bot account exclusion
# ---------------------------------------------------------------------------


class TestBotAccountExclusion:
    """Bot accounts are filtered from results."""

    @pytest.mark.asyncio
    async def test_bot_account_excluded_from_signals(self):
        """A tweet from a bot account should not appear in signals."""
        pipeline = _build_pipeline()
        tweets = [
            _make_tweet(screen_name="RealAnalyst", followers_count=5000),
            _make_tweet(screen_name="crypto_bot", followers_count=5000),
        ]

        with patch.object(pipeline, "_search_tweets", new_callable=AsyncMock, return_value=tweets):
            with patch("src.pipelines.twitter.classify_source_tier", return_value="S6"):
                signals = await pipeline.get_signals_for_market(["bitcoin"])

        authors = [s.author for s in signals]
        assert "RealAnalyst" in authors
        assert "crypto_bot" not in authors


# ---------------------------------------------------------------------------
# Test: _is_bot_account
# ---------------------------------------------------------------------------


class TestIsBotAccount:
    """Unit tests for the static _is_bot_account method."""

    def test_name_with_bot_keyword(self):
        """Screen name containing 'bot' -> True."""
        author = {"screen_name": "news_bot_daily", "followers_count": 5000, "following_count": 100}
        assert TwitterDataPipeline._is_bot_account(author) is True

    def test_name_with_autopost(self):
        """Screen name containing 'autopost' -> True."""
        author = {"screen_name": "autopost_crypto", "followers_count": 5000, "following_count": 100}
        assert TwitterDataPipeline._is_bot_account(author) is True

    def test_name_with_feed(self):
        """Screen name containing 'feed' -> True."""
        author = {"screen_name": "cryptofeedlive", "followers_count": 5000, "following_count": 100}
        assert TwitterDataPipeline._is_bot_account(author) is True

    def test_high_following_to_follower_ratio(self):
        """following/followers > 50 -> True."""
        author = {"screen_name": "normalname", "followers_count": 100, "following_count": 5100}
        assert TwitterDataPipeline._is_bot_account(author) is True

    def test_normal_account(self):
        """Normal account with regular name and ratio -> False."""
        author = {"screen_name": "JohnSmith", "followers_count": 10000, "following_count": 500}
        assert TwitterDataPipeline._is_bot_account(author) is False

    def test_empty_author_dict(self):
        """Empty dict should return False (no bot indicators)."""
        assert TwitterDataPipeline._is_bot_account({}) is False

    def test_following_ratio_exactly_at_threshold(self):
        """following/followers == 50 is NOT a bot (needs > 50)."""
        author = {"screen_name": "borderline", "followers_count": 100, "following_count": 5000}
        assert TwitterDataPipeline._is_bot_account(author) is False

    def test_following_ratio_just_over_threshold(self):
        """following/followers == 50.1 is a bot."""
        author = {"screen_name": "borderline", "followers_count": 100, "following_count": 5010}
        assert TwitterDataPipeline._is_bot_account(author) is True


# ---------------------------------------------------------------------------
# Test: Deduplication by content similarity
# ---------------------------------------------------------------------------


class TestDeduplication:
    """Tests for _deduplicate_by_content_similarity."""

    def test_high_overlap_deduplicates(self):
        """90% word overlap (>0.8 threshold) -> only one tweet kept."""
        # 10 shared words out of 11 total unique = 10/11 ~ 0.91 overlap
        tweets = [
            _make_tweet(text="bitcoin price surges to new all time high today in markets", screen_name="A"),
            _make_tweet(text="bitcoin price surges to new all time high today in trading", screen_name="B"),
        ]
        result = TwitterDataPipeline._deduplicate_by_content_similarity(tweets)
        assert len(result) == 1

    def test_low_overlap_keeps_both(self):
        """30% word overlap -> both tweets kept."""
        tweets = [
            _make_tweet(text="bitcoin price surges to new all time high today", screen_name="A"),
            _make_tweet(text="ethereum developers announce major protocol upgrade coming soon", screen_name="B"),
        ]
        result = TwitterDataPipeline._deduplicate_by_content_similarity(tweets)
        assert len(result) == 2

    def test_empty_list(self):
        """Empty input returns empty output."""
        assert TwitterDataPipeline._deduplicate_by_content_similarity([]) == []

    def test_single_tweet(self):
        """Single tweet is always kept."""
        tweets = [_make_tweet(text="hello world")]
        result = TwitterDataPipeline._deduplicate_by_content_similarity(tweets)
        assert len(result) == 1

    def test_identical_tweets_dedup(self):
        """Identical text -> only one kept."""
        tweets = [
            _make_tweet(text="exactly the same tweet", screen_name="A"),
            _make_tweet(text="exactly the same tweet", screen_name="B"),
        ]
        result = TwitterDataPipeline._deduplicate_by_content_similarity(tweets)
        assert len(result) == 1

    def test_empty_text_tweets_skipped(self):
        """Tweets with empty text are skipped (no words -> filtered)."""
        tweets = [
            _make_tweet(text="", screen_name="A"),
            _make_tweet(text="real content here", screen_name="B"),
        ]
        result = TwitterDataPipeline._deduplicate_by_content_similarity(tweets)
        assert len(result) == 1
        assert result[0]["author"]["screen_name"] == "B"

    @pytest.mark.asyncio
    async def test_dedup_integration_in_pipeline(self):
        """Dedup is applied within get_signals_for_market."""
        pipeline = _build_pipeline()
        # 10 shared words out of 11 total unique = ~0.91 overlap (>0.8 threshold)
        tweets = [
            _make_tweet(text="bitcoin price surges to new all time high today in markets", screen_name="A"),
            _make_tweet(text="bitcoin price surges to new all time high today in trading", screen_name="B"),
        ]

        with patch.object(pipeline, "_search_tweets", new_callable=AsyncMock, return_value=tweets):
            with patch("src.pipelines.twitter.classify_source_tier", return_value="S6"):
                signals = await pipeline.get_signals_for_market(["bitcoin"])

        assert len(signals) == 1


# ---------------------------------------------------------------------------
# Test: Signals sorted by credibility descending
# ---------------------------------------------------------------------------


class TestSortingByCredibility:
    """Signals are sorted by credibility in descending order."""

    @pytest.mark.asyncio
    async def test_sorted_descending(self):
        """Signals should come out ordered highest credibility first."""
        pipeline = _build_pipeline()
        tweets = [
            _make_tweet(screen_name="low_tier", text="unique tweet one about markets"),
            _make_tweet(screen_name="high_tier", text="unique tweet two about crypto analysis"),
            _make_tweet(screen_name="mid_tier", text="unique tweet three about economy trends"),
        ]

        tier_map = {
            "low_tier": "S6",   # 0.30
            "high_tier": "S2",  # 0.90
            "mid_tier": "S3",   # 0.80
        }

        def mock_classify(signal_dict):
            handle = signal_dict.get("account_handle", "").lstrip("@")
            return tier_map.get(handle, "S6")

        with patch.object(pipeline, "_search_tweets", new_callable=AsyncMock, return_value=tweets):
            with patch("src.pipelines.twitter.classify_source_tier", side_effect=mock_classify):
                signals = await pipeline.get_signals_for_market(["bitcoin"])

        assert len(signals) == 3
        creds = [s.credibility for s in signals]
        assert creds == sorted(creds, reverse=True)
        assert signals[0].author == "high_tier"
        assert signals[1].author == "mid_tier"
        assert signals[2].author == "low_tier"


# ---------------------------------------------------------------------------
# Test: Max 10 signals returned
# ---------------------------------------------------------------------------


class TestMaxSignalsLimit:
    """At most 10 signals are returned."""

    @pytest.mark.asyncio
    async def test_max_10_signals(self):
        """Even with 15 qualifying tweets, only 10 signals are returned."""
        pipeline = _build_pipeline()
        tweets = [
            _make_tweet(
                screen_name=f"user_{i}",
                text=f"completely unique tweet number {i} about topic {i * 7}",
                followers_count=5000,
                engagement_score=50,
            )
            for i in range(15)
        ]

        with patch.object(pipeline, "_search_tweets", new_callable=AsyncMock, return_value=tweets):
            with patch("src.pipelines.twitter.classify_source_tier", return_value="S6"):
                signals = await pipeline.get_signals_for_market(["bitcoin"])

        assert len(signals) <= 10


# ---------------------------------------------------------------------------
# Test: Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """API timeout and rate-limiting produce empty lists."""

    @pytest.mark.asyncio
    async def test_api_timeout_returns_empty_list(self):
        """httpx.TimeoutException -> empty list of signals."""
        pipeline = _build_pipeline()

        async def _raise_timeout(*args, **kwargs):
            raise httpx.TimeoutException("connection timed out")

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = _raise_timeout
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            signals = await pipeline.get_signals_for_market(["bitcoin"])

        assert signals == []

    @pytest.mark.asyncio
    async def test_api_429_returns_empty_list(self):
        """HTTP 429 rate limit -> empty list of signals."""
        pipeline = _build_pipeline()

        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 429

        async def _return_429(*args, **kwargs):
            return mock_resp

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = _return_429
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            signals = await pipeline.get_signals_for_market(["bitcoin"])

        assert signals == []


# ---------------------------------------------------------------------------
# Test: Empty keywords
# ---------------------------------------------------------------------------


class TestEmptyKeywords:
    """Empty keywords list returns empty signal list immediately."""

    @pytest.mark.asyncio
    async def test_empty_keywords_returns_empty(self):
        """get_signals_for_market([]) returns [] without calling API."""
        pipeline = _build_pipeline()

        with patch.object(pipeline, "_search_tweets", new_callable=AsyncMock) as mock_search:
            signals = await pipeline.get_signals_for_market([])

        assert signals == []
        mock_search.assert_not_called()
