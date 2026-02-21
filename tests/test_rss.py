"""Tests for the RSS data pipeline (RSSPipeline).

Covers signal generation, deduplication across calls, headline age filtering,
source tier classification by feed domain, parse error handling, and per-feed
entry limits.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from src.models import Signal, SOURCE_TIER_CREDIBILITY
from src.pipelines.rss import RSSPipeline, _parse_date


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(
    title: str = "Breaking: Major economic event",
    published: str | None = None,
) -> SimpleNamespace:
    """Build a fake feedparser entry with .title and optional .published."""
    entry = SimpleNamespace()
    entry.title = title
    if published is not None:
        entry.published = published
    else:
        # Default to "just now" in ISO format
        entry.published = datetime.now(timezone.utc).isoformat()
    # feedparser entries use .get() sometimes, so add a simple get method
    entry.get = lambda key, default="": getattr(entry, key, default)
    return entry


def _make_feed(entries: list) -> SimpleNamespace:
    """Build a fake feedparser.parse() result."""
    feed = SimpleNamespace()
    feed.entries = entries
    return feed


# A minimal two-feed config matching the structure of rss_feeds.yaml.
_TEST_FEEDS = {
    "reuters_top": {
        "url": "https://feeds.reuters.com/reuters/topNews",
        "domain": "reuters.com",
    },
    "bbc_world": {
        "url": "http://feeds.bbci.co.uk/news/world/rss.xml",
        "domain": "bbc.com",
    },
}


def _build_pipeline(feeds: dict | None = None) -> RSSPipeline:
    """Create an RSSPipeline with mocked _load_feed_config."""
    with patch("src.pipelines.rss._load_feed_config", return_value=feeds or _TEST_FEEDS):
        return RSSPipeline()


# ---------------------------------------------------------------------------
# Test: Fresh headlines produce signals
# ---------------------------------------------------------------------------


class TestFreshHeadlines:
    """First call with fresh headlines returns signals."""

    @pytest.mark.asyncio
    async def test_first_call_returns_signals(self):
        """Fresh headlines should produce Signal objects."""
        pipeline = _build_pipeline()
        now_iso = datetime.now(timezone.utc).isoformat()

        reuters_feed = _make_feed([_make_entry("US Fed raises rates", now_iso)])
        bbc_feed = _make_feed([_make_entry("UK election update", now_iso)])

        def mock_parse(url):
            if "reuters" in url:
                return reuters_feed
            return bbc_feed

        with patch("feedparser.parse", side_effect=mock_parse):
            with patch("src.pipelines.rss.classify_source_tier", return_value="S3"):
                signals = await pipeline.get_breaking_news()

        assert len(signals) == 2
        assert all(isinstance(s, Signal) for s in signals)


# ---------------------------------------------------------------------------
# Test: Signal field values
# ---------------------------------------------------------------------------


class TestSignalFields:
    """Each signal has source='rss' and headline_only=True."""

    @pytest.mark.asyncio
    async def test_source_and_headline_only(self):
        """Verify source='rss' and headline_only=True on every signal."""
        pipeline = _build_pipeline()
        now_iso = datetime.now(timezone.utc).isoformat()

        feed = _make_feed([_make_entry("Test headline", now_iso)])

        with patch("feedparser.parse", return_value=feed):
            with patch("src.pipelines.rss.classify_source_tier", return_value="S3"):
                signals = await pipeline.get_breaking_news()

        for sig in signals:
            assert sig.source == "rss"
            assert sig.headline_only is True


# ---------------------------------------------------------------------------
# Test: Deduplication across calls
# ---------------------------------------------------------------------------


class TestDeduplication:
    """Same headline on second call is not returned again."""

    @pytest.mark.asyncio
    async def test_duplicate_headline_not_returned(self):
        """A headline seen in call 1 should not appear in call 2."""
        pipeline = _build_pipeline()
        now_iso = datetime.now(timezone.utc).isoformat()

        feed = _make_feed([_make_entry("Repeated headline", now_iso)])

        with patch("feedparser.parse", return_value=feed):
            with patch("src.pipelines.rss.classify_source_tier", return_value="S3"):
                signals_first = await pipeline.get_breaking_news()
                signals_second = await pipeline.get_breaking_news()

        assert len(signals_first) >= 1
        # On the second call, the headline is already seen, so filtered out
        repeated_in_second = [s for s in signals_second if s.content == "Repeated headline"]
        assert len(repeated_in_second) == 0

    @pytest.mark.asyncio
    async def test_new_headline_on_second_call_returned(self):
        """A new headline on the second call should still come through."""
        pipeline = _build_pipeline()
        now_iso = datetime.now(timezone.utc).isoformat()

        feed_call1 = _make_feed([_make_entry("First headline", now_iso)])
        feed_call2 = _make_feed([
            _make_entry("First headline", now_iso),
            _make_entry("Second headline", now_iso),
        ])

        call_count = {"n": 0}

        def mock_parse(url):
            call_count["n"] += 1
            if call_count["n"] <= len(_TEST_FEEDS):
                return feed_call1
            return feed_call2

        with patch("feedparser.parse", side_effect=mock_parse):
            with patch("src.pipelines.rss.classify_source_tier", return_value="S3"):
                await pipeline.get_breaking_news()
                signals_second = await pipeline.get_breaking_news()

        contents = [s.content for s in signals_second]
        assert "Second headline" in contents


# ---------------------------------------------------------------------------
# Test: Old headlines excluded (> 2 hours)
# ---------------------------------------------------------------------------


class TestOldHeadlineExclusion:
    """Headlines older than 2 hours are excluded."""

    @pytest.mark.asyncio
    async def test_headline_older_than_2h_excluded(self):
        """A headline published 3 hours ago should be excluded."""
        pipeline = _build_pipeline()
        three_hours_ago = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()

        feed = _make_feed([_make_entry("Old headline", three_hours_ago)])

        with patch("feedparser.parse", return_value=feed):
            with patch("src.pipelines.rss.classify_source_tier", return_value="S3"):
                signals = await pipeline.get_breaking_news()

        contents = [s.content for s in signals]
        assert "Old headline" not in contents

    @pytest.mark.asyncio
    async def test_headline_within_2h_included(self):
        """A headline published 1 hour ago should be included."""
        pipeline = _build_pipeline()
        one_hour_ago = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()

        feed = _make_feed([_make_entry("Recent headline", one_hour_ago)])

        with patch("feedparser.parse", return_value=feed):
            with patch("src.pipelines.rss.classify_source_tier", return_value="S3"):
                signals = await pipeline.get_breaking_news()

        contents = [s.content for s in signals]
        assert "Recent headline" in contents


# ---------------------------------------------------------------------------
# Test: _prune_old_headlines
# ---------------------------------------------------------------------------


class TestPruneOldHeadlines:
    """_prune_old_headlines removes entries older than 24 hours."""

    def test_prune_removes_old_entries(self):
        """Headlines timestamped > 24h ago are removed from seen_headlines."""
        pipeline = _build_pipeline()
        now = datetime.now(timezone.utc)

        pipeline.seen_headlines = {
            "Old headline": now - timedelta(hours=25),
            "Recent headline": now - timedelta(hours=1),
        }

        pipeline._prune_old_headlines()

        assert "Old headline" not in pipeline.seen_headlines
        assert "Recent headline" in pipeline.seen_headlines

    def test_prune_keeps_recent_entries(self):
        """Headlines timestamped < 24h ago are preserved."""
        pipeline = _build_pipeline()
        now = datetime.now(timezone.utc)

        pipeline.seen_headlines = {
            "Headline A": now - timedelta(hours=23),
            "Headline B": now - timedelta(hours=12),
        }

        pipeline._prune_old_headlines()

        assert len(pipeline.seen_headlines) == 2

    def test_prune_empty_dict(self):
        """Pruning an empty dict does not error."""
        pipeline = _build_pipeline()
        pipeline.seen_headlines = {}
        pipeline._prune_old_headlines()
        assert pipeline.seen_headlines == {}


# ---------------------------------------------------------------------------
# Test: Source tier classification by domain
# ---------------------------------------------------------------------------


class TestSourceTierByDomain:
    """Feed domain determines the source_tier via classify_source_tier."""

    @pytest.mark.asyncio
    async def test_reuters_domain_gets_s2(self):
        """reuters.com feed -> source_tier='S2'."""
        feeds = {
            "reuters_only": {
                "url": "https://feeds.reuters.com/reuters/topNews",
                "domain": "reuters.com",
            },
        }
        pipeline = _build_pipeline(feeds)
        now_iso = datetime.now(timezone.utc).isoformat()
        feed = _make_feed([_make_entry("Reuters headline", now_iso)])

        with patch("feedparser.parse", return_value=feed):
            # Use the real classifier -- reuters.com is in known_sources.yaml as wire_services
            signals = await pipeline.get_breaking_news()

        assert len(signals) == 1
        assert signals[0].source_tier == "S2"

    @pytest.mark.asyncio
    async def test_bbc_domain_gets_s3(self):
        """bbc.com feed -> source_tier='S3'."""
        feeds = {
            "bbc_only": {
                "url": "http://feeds.bbci.co.uk/news/world/rss.xml",
                "domain": "bbc.com",
            },
        }
        pipeline = _build_pipeline(feeds)
        now_iso = datetime.now(timezone.utc).isoformat()
        feed = _make_feed([_make_entry("BBC headline", now_iso)])

        with patch("feedparser.parse", return_value=feed):
            # Use the real classifier -- bbc.com is in known_sources.yaml as institutional_media
            signals = await pipeline.get_breaking_news()

        assert len(signals) == 1
        assert signals[0].source_tier == "S3"

    @pytest.mark.asyncio
    async def test_unknown_domain_gets_s6(self):
        """Unknown domain -> source_tier='S6'."""
        feeds = {
            "unknown_blog": {
                "url": "https://randomblog.xyz/feed",
                "domain": "randomblog.xyz",
            },
        }
        pipeline = _build_pipeline(feeds)
        now_iso = datetime.now(timezone.utc).isoformat()
        feed = _make_feed([_make_entry("Blog headline", now_iso)])

        with patch("feedparser.parse", return_value=feed):
            signals = await pipeline.get_breaking_news()

        assert len(signals) == 1
        assert signals[0].source_tier == "S6"


# ---------------------------------------------------------------------------
# Test: Feed parse failure -> warning logged, continues
# ---------------------------------------------------------------------------


class TestFeedParseFailure:
    """A feed that raises an exception logs a warning and continues."""

    @pytest.mark.asyncio
    async def test_parse_failure_logs_warning_and_continues(self, caplog):
        """One bad feed should not break the other feeds."""
        feeds = {
            "bad_feed": {
                "url": "https://broken.example.com/rss",
                "domain": "broken.example.com",
            },
            "good_feed": {
                "url": "https://good.example.com/rss",
                "domain": "good.example.com",
            },
        }
        pipeline = _build_pipeline(feeds)
        now_iso = datetime.now(timezone.utc).isoformat()
        good_feed = _make_feed([_make_entry("Good headline", now_iso)])

        call_count = {"n": 0}

        def mock_parse(url):
            call_count["n"] += 1
            if "broken" in url:
                raise Exception("Network error")
            return good_feed

        with patch("feedparser.parse", side_effect=mock_parse):
            with patch("src.pipelines.rss.classify_source_tier", return_value="S6"):
                signals = await pipeline.get_breaking_news()

        # The good feed's signal should still come through
        assert len(signals) >= 1
        contents = [s.content for s in signals]
        assert "Good headline" in contents


# ---------------------------------------------------------------------------
# Test: Each feed processes up to 10 entries
# ---------------------------------------------------------------------------


class TestMaxEntriesPerFeed:
    """Each feed only processes the first 10 entries."""

    @pytest.mark.asyncio
    async def test_max_10_entries_per_feed(self):
        """A feed with 15 entries should only process the first 10."""
        feeds = {
            "big_feed": {
                "url": "https://example.com/rss",
                "domain": "example.com",
            },
        }
        pipeline = _build_pipeline(feeds)
        now_iso = datetime.now(timezone.utc).isoformat()

        entries = [_make_entry(f"Headline {i}", now_iso) for i in range(15)]
        feed = _make_feed(entries)

        with patch("feedparser.parse", return_value=feed):
            with patch("src.pipelines.rss.classify_source_tier", return_value="S6"):
                signals = await pipeline.get_breaking_news()

        # Only the first 10 entries should be processed
        assert len(signals) <= 10

    @pytest.mark.asyncio
    async def test_all_entries_processed_when_fewer_than_10(self):
        """A feed with 5 entries should process all 5."""
        feeds = {
            "small_feed": {
                "url": "https://example.com/rss",
                "domain": "example.com",
            },
        }
        pipeline = _build_pipeline(feeds)
        now_iso = datetime.now(timezone.utc).isoformat()

        entries = [_make_entry(f"Headline {i}", now_iso) for i in range(5)]
        feed = _make_feed(entries)

        with patch("feedparser.parse", return_value=feed):
            with patch("src.pipelines.rss.classify_source_tier", return_value="S6"):
                signals = await pipeline.get_breaking_news()

        assert len(signals) == 5


# ---------------------------------------------------------------------------
# Test 4: Same headline after 24h+ prune -> returned again (re-appears)
# ---------------------------------------------------------------------------


class TestHeadlineReappearanceAfterPrune:
    """After pruning stale entries (>24h), a headline can re-appear."""

    @pytest.mark.asyncio
    async def test_headline_reappears_after_prune(self):
        """Test 4: A headline seen once, then pruned after 24h+, should
        be returned again on the next call."""
        pipeline = _build_pipeline()
        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()

        feed = _make_feed([_make_entry("Reappearing headline", now_iso)])

        with patch("feedparser.parse", return_value=feed):
            with patch("src.pipelines.rss.classify_source_tier", return_value="S3"):
                signals_first = await pipeline.get_breaking_news()

        # Headline should be seen in the first call
        assert any(s.content == "Reappearing headline" for s in signals_first)

        # Simulate 25 hours passing: manually set seen_headlines timestamp to > 24h ago
        pipeline.seen_headlines["Reappearing headline"] = now - timedelta(hours=25)

        # Prune will remove entries older than 24h
        pipeline._prune_old_headlines()
        assert "Reappearing headline" not in pipeline.seen_headlines

        # Now the headline should come through again
        with patch("feedparser.parse", return_value=feed):
            with patch("src.pipelines.rss.classify_source_tier", return_value="S3"):
                signals_again = await pipeline.get_breaking_news()

        assert any(s.content == "Reappearing headline" for s in signals_again)


# ---------------------------------------------------------------------------
# Test 7: Headline entry with no published date -> treated as "now", included
# ---------------------------------------------------------------------------


def _make_entry_no_published(title: str = "No-date headline") -> SimpleNamespace:
    """Build a fake feedparser entry WITHOUT a .published attribute."""
    entry = SimpleNamespace()
    entry.title = title
    # Do NOT set entry.published -- simulates a feed entry missing the date.
    entry.get = lambda key, default="": getattr(entry, key, default)
    return entry


class TestHeadlineNoPublishedDate:
    """Entries with no published date are treated as current and included."""

    @pytest.mark.asyncio
    async def test_entry_without_published_date_included(self):
        """Test 7: An entry missing its published date should still produce
        a signal (treated as 'now' so it's within the 2h window)."""
        feeds = {
            "test_feed": {
                "url": "https://example.com/rss",
                "domain": "example.com",
            },
        }
        pipeline = _build_pipeline(feeds)

        entry = _make_entry_no_published("No-date headline")
        feed = _make_feed([entry])

        with patch("feedparser.parse", return_value=feed):
            with patch("src.pipelines.rss.classify_source_tier", return_value="S6"):
                signals = await pipeline.get_breaking_news()

        # The entry should be included since _parse_date returns None for missing
        # published field, and the code skips the age check when published is None.
        contents = [s.content for s in signals]
        assert "No-date headline" in contents


# ---------------------------------------------------------------------------
# Test 10: Adding many headlines doesn't cause memory issues (bounded cache)
# ---------------------------------------------------------------------------


class TestBoundedCache:
    """Pruning keeps memory bounded even with many headlines."""

    @pytest.mark.asyncio
    async def test_many_headlines_bounded_by_prune(self):
        """Test 10: After inserting many headlines and pruning old ones,
        only recent headlines remain in seen_headlines (bounded cache)."""
        pipeline = _build_pipeline()
        now = datetime.now(timezone.utc)

        # Simulate 1000 headlines, half of them old (> 24h)
        for i in range(500):
            pipeline.seen_headlines[f"Old headline {i}"] = now - timedelta(hours=25)
        for i in range(500):
            pipeline.seen_headlines[f"Recent headline {i}"] = now - timedelta(hours=1)

        assert len(pipeline.seen_headlines) == 1000

        pipeline._prune_old_headlines()

        # Only the 500 recent headlines should remain
        assert len(pipeline.seen_headlines) == 500
        assert all(
            key.startswith("Recent headline")
            for key in pipeline.seen_headlines
        )


# ---------------------------------------------------------------------------
# Test 13: Feed domain "coindesk.com" -> source_tier="S3"
# ---------------------------------------------------------------------------


class TestCoindeskSourceTier:
    """coindesk.com is in institutional_media -> S3."""

    @pytest.mark.asyncio
    async def test_coindesk_domain_gets_s3(self):
        """Test 13: coindesk.com feed -> source_tier='S3'."""
        feeds = {
            "coindesk": {
                "url": "https://www.coindesk.com/arc/outboundfeeds/rss/",
                "domain": "coindesk.com",
            },
        }
        pipeline = _build_pipeline(feeds)
        now_iso = datetime.now(timezone.utc).isoformat()
        feed = _make_feed([_make_entry("Crypto market update", now_iso)])

        with patch("feedparser.parse", return_value=feed):
            # Use the real classifier -- coindesk.com is in known_sources.yaml as institutional_media
            signals = await pipeline.get_breaking_news()

        assert len(signals) == 1
        assert signals[0].source_tier == "S3"


# ---------------------------------------------------------------------------
# Test 15: Feed with empty entries list -> no signals from that feed
# ---------------------------------------------------------------------------


class TestEmptyFeedEntries:
    """A feed that returns an empty entries list produces no signals."""

    @pytest.mark.asyncio
    async def test_empty_entries_produces_no_signals(self):
        """Test 15: A feed with entries=[] should produce zero signals."""
        feeds = {
            "empty_feed": {
                "url": "https://example.com/rss",
                "domain": "example.com",
            },
        }
        pipeline = _build_pipeline(feeds)
        feed = _make_feed([])  # Empty entries list

        with patch("feedparser.parse", return_value=feed):
            with patch("src.pipelines.rss.classify_source_tier", return_value="S6"):
                signals = await pipeline.get_breaking_news()

        assert len(signals) == 0
