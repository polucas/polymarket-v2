"""Tests for keyword extraction and Grok context building."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.models import Market, Signal, OrderBook
from src.pipelines.context_builder import (
    extract_keywords,
    build_grok_context,
    _keyword_cache,
    KEYWORD_SUPPLEMENTS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_market(**overrides) -> Market:
    defaults = dict(
        market_id="test-market-1",
        question="Will Donald Trump win the 2024 election?",
        yes_price=0.62,
        no_price=0.38,
        resolution_time=datetime(2024, 11, 6, tzinfo=timezone.utc),
        hours_to_resolution=12.0,
        volume_24h=50_000.0,
        liquidity=25_000.0,
        market_type="political",
        fee_rate=0.02,
        keywords=["donald", "trump", "election"],
    )
    defaults.update(overrides)
    return Market(**defaults)


def _make_signal(
    *,
    content: str = "Breaking: new poll shows lead.",
    credibility: float = 0.85,
    source: str = "twitter",
    source_tier: str = "S2",
    author: str = "pollster123",
    followers: int = 50_000,
    engagement: int = 1_200,
    headline_only: bool = False,
) -> Signal:
    return Signal(
        source=source,
        source_tier=source_tier,
        info_type=None,
        content=content,
        credibility=credibility,
        author=author,
        followers=followers,
        engagement=engagement,
        timestamp=datetime.now(timezone.utc),
        headline_only=headline_only,
    )


def _make_orderbook(
    market_id: str = "test-market-1",
    bids: list | None = None,
    asks: list | None = None,
) -> OrderBook:
    return OrderBook(
        market_id=market_id,
        bids=bids if bids is not None else [100.0, 80.0, 60.0],
        asks=asks if asks is not None else [90.0, 70.0, 50.0],
        timestamp=datetime.now(timezone.utc),
    )


@pytest.fixture(autouse=True)
def clear_keyword_cache():
    """Clear the module-level keyword cache before each test."""
    _keyword_cache.clear()
    yield
    _keyword_cache.clear()


# ---------------------------------------------------------------------------
# extract_keywords
# ---------------------------------------------------------------------------


class TestExtractKeywords:
    def test_named_entities_extracted(self):
        """Named entities like 'Donald Trump' are extracted from question."""
        keywords = extract_keywords(
            "m-1",
            "Will Donald Trump win the presidency?",
            "political",
        )
        # "Donald Trump" should be among the results
        assert any("Donald Trump" in kw for kw in keywords)

    def test_acronyms_extracted(self):
        """Uppercase acronyms like SEC and ETF are extracted."""
        keywords = extract_keywords(
            "m-2",
            "Will the SEC approve the Bitcoin ETF filing?",
            "regulatory",
        )
        found = {kw for kw in keywords}
        assert "SEC" in found or any("SEC" in kw for kw in keywords)
        assert "ETF" in found or any("ETF" in kw for kw in keywords)

    def test_cache_returns_same_result(self):
        """Same market_id returns cached result without recomputing."""
        first = extract_keywords("cached-1", "Will Donald Trump run?", "political")
        # Mutate the cache intentionally (to prove it's used)
        _keyword_cache["cached-1"] = ["CACHED_VALUE"]
        second = extract_keywords("cached-1", "Completely different question", "economic")
        assert second == ["CACHED_VALUE"]

    def test_supplements_added_when_few_regex_matches(self):
        """When regex finds < 2 entities, market type supplements are added."""
        # A question with no clear named entities or acronyms
        keywords = extract_keywords(
            "m-sparse",
            "will something happen soon here today",
            "economic",
        )
        # Should include supplements from KEYWORD_SUPPLEMENTS["economic"]
        economic_supps = KEYWORD_SUPPLEMENTS["economic"]
        overlap = set(keywords) & set(economic_supps)
        assert len(overlap) > 0, f"Expected economic supplements in {keywords}"

    def test_ticker_pattern_extracted(self):
        """#29 — 'Will BTC reach $100K?' -> extracts 'BTC' (ticker/acronym pattern)."""
        keywords = extract_keywords(
            "m-btc",
            "Will BTC reach $100K?",
            "crypto_15m",
        )
        assert any("BTC" in kw for kw in keywords)

    def test_fed_with_few_regex_matches_gets_supplements(self):
        """#31 — Question about 'Fed' with < 2 regex matches -> supplements added."""
        keywords = extract_keywords(
            "m-fed",
            "Will the Fed cut rates soon?",
            "economic",
        )
        # "Fed" alone is only 3 chars uppercase, may not match [A-Z]{2,6} or named entity.
        # With < 2 regex entities, economic supplements should be added.
        economic_supps = KEYWORD_SUPPLEMENTS["economic"]
        overlap = set(keywords) & set(economic_supps)
        assert len(overlap) > 0, f"Expected economic supplements in {keywords}"

    def test_keywords_list_reasonable_size(self):
        """#34 — Keywords list doesn't exceed a reasonable size (<=20 items)."""
        # Even a long question with many named entities should stay bounded
        keywords = extract_keywords(
            "m-long",
            "Will Donald Trump, Joe Biden, Barack Obama, Kamala Harris, "
            "Ron DeSantis, or Elon Musk win the election according to CNN, "
            "NBC, FOX, ABC, CBS, BBC, or MSNBC polls?",
            "political",
        )
        assert len(keywords) <= 20, f"Keywords list too large: {len(keywords)} items"


# ---------------------------------------------------------------------------
# build_grok_context
# ---------------------------------------------------------------------------


class TestBuildGrokContext:
    def test_output_contains_market_question(self):
        market = _make_market()
        ctx = build_grok_context(market, [], [], _make_orderbook())
        assert market.question in ctx

    def test_output_contains_yes_no_prices(self):
        market = _make_market(yes_price=0.62, no_price=0.38)
        ctx = build_grok_context(market, [], [], _make_orderbook())
        assert "0.6200" in ctx
        assert "0.3800" in ctx

    def test_signals_sorted_by_credibility_top_7(self):
        """Signals should be sorted by credibility descending, top 7 only."""
        market = _make_market()
        signals = [
            _make_signal(content=f"Signal {i}", credibility=0.1 * i, author=f"auth{i}")
            for i in range(1, 11)  # 10 signals
        ]
        ctx = build_grok_context(market, signals, [], _make_orderbook())

        # Highest credibility is 1.0 (i=10), lowest included is 0.4 (i=4)
        # The top 7: i=10,9,8,7,6,5,4
        assert "auth10" in ctx
        assert "auth9" in ctx
        assert "auth4" in ctx
        # i=3 (0.3 cred) should be excluded
        assert "auth3" not in ctx
        # i=1 (0.1 cred) should be excluded — use word boundary to avoid matching "auth10"
        assert "@auth1 " not in ctx and "@auth1)" not in ctx

    def test_output_contains_orderbook_depth_and_skew(self):
        """Output should contain orderbook depth dollar amount and skew value."""
        ob = _make_orderbook(bids=[200.0, 100.0], asks=[50.0, 50.0])
        # depth = 200+100+50+50 = 400, skew = (300-100)/400 = +0.50
        market = _make_market()
        ctx = build_grok_context(market, [], [], ob)
        assert "400" in ctx  # depth
        assert "+0.50" in ctx  # skew

    def test_output_contains_json_format_instructions(self):
        """Output should contain the JSON response format instructions."""
        market = _make_market()
        ctx = build_grok_context(market, [], [], _make_orderbook())
        assert "estimated_probability" in ctx
        assert "confidence" in ctx
        assert "reasoning" in ctx
        assert "signal_info_types" in ctx
        assert "JSON" in ctx

    def test_zero_signals_produces_valid_prompt(self):
        """When there are 0 signals, the context is still a valid non-empty prompt."""
        market = _make_market()
        ctx = build_grok_context(market, [], [], _make_orderbook())
        assert len(ctx) > 100
        assert "No signals available" in ctx
        assert market.question in ctx

    def test_mixed_twitter_and_rss_signals_merged(self):
        """Twitter and RSS signals are merged and sorted together."""
        market = _make_market()
        twitter_signals = [
            _make_signal(content="Twitter high", credibility=0.95, source="twitter", author="tw_top"),
            _make_signal(content="Twitter low", credibility=0.30, source="twitter", author="tw_low"),
        ]
        rss_signals = [
            _make_signal(content="RSS mid", credibility=0.70, source="rss", author="rss_mid"),
        ]
        ctx = build_grok_context(market, twitter_signals, rss_signals, _make_orderbook())

        # All three should appear (total <= 7)
        assert "tw_top" in ctx
        assert "tw_low" in ctx
        assert "rss_mid" in ctx

        # Verify ordering: tw_top (0.95) should come before rss_mid (0.70)
        pos_tw_top = ctx.index("tw_top")
        pos_rss_mid = ctx.index("rss_mid")
        pos_tw_low = ctx.index("tw_low")
        assert pos_tw_top < pos_rss_mid < pos_tw_low

    def test_output_contains_resolution_time(self):
        """#37 — Output string contains resolution time (hours_to_resolution)."""
        market = _make_market(hours_to_resolution=36.5)
        ctx = build_grok_context(market, [], [], _make_orderbook())
        assert "36.5" in ctx

    def test_output_contains_volume_and_liquidity(self):
        """#38 — Output string contains volume and liquidity dollar amounts."""
        market = _make_market(volume_24h=123_456.0, liquidity=78_900.0)
        ctx = build_grok_context(market, [], [], _make_orderbook())
        assert "$123,456" in ctx
        assert "$78,900" in ctx

    def test_signal_lines_show_source_tier_label(self):
        """#41 — Each signal line in context shows source tier label like '[S2|twitter]'."""
        market = _make_market()
        signals = [
            _make_signal(source="twitter", source_tier="S2", content="Tweet info", author="user_a"),
            _make_signal(source="rss", source_tier="S3", content="RSS info", author="user_b"),
        ]
        ctx = build_grok_context(market, signals, [], _make_orderbook())
        assert "[S2|twitter]" in ctx
        assert "[S3|rss]" in ctx

    def test_prompt_includes_info_type_classification_instructions(self):
        """#42 — Prompt includes info-type classification instructions text (I1-I5)."""
        market = _make_market()
        ctx = build_grok_context(market, [], [], _make_orderbook())
        assert "I1" in ctx
        assert "I2" in ctx
        assert "I3" in ctx
        assert "I4" in ctx
        assert "I5" in ctx
        # Verify descriptive labels are present
        assert "Verified fact" in ctx
        assert "Authoritative analysis" in ctx
        assert "Statistical" in ctx or "data-driven" in ctx
        assert "Market intelligence" in ctx
        assert "Rumor" in ctx or "speculation" in ctx

    def test_bid_depth_is_sum_of_bids(self):
        """#46 — bid_depth = sum of orderbook bids."""
        ob = _make_orderbook(bids=[100.0, 50.0, 25.0], asks=[10.0])
        market = _make_market()
        ctx = build_grok_context(market, [], [], ob)
        # total bids = 175, total asks = 10, depth = 185
        # The depth line should contain 185
        assert "185" in ctx

    def test_ask_depth_is_sum_of_asks(self):
        """#47 — ask_depth = sum of orderbook asks."""
        ob = _make_orderbook(bids=[10.0], asks=[200.0, 100.0, 50.0])
        market = _make_market()
        ctx = build_grok_context(market, [], [], ob)
        # total bids = 10, total asks = 350, depth = 360
        # skew = (10 - 350) / 360 = -340/360 = -0.9444...
        assert "360" in ctx

    def test_skew_with_empty_orderbook_no_division_by_zero(self):
        """#48 — Skew calculation with empty orderbook (division by zero protection)."""
        ob = _make_orderbook(bids=[], asks=[])
        market = _make_market()
        # Should not raise any exceptions
        ctx = build_grok_context(market, [], [], ob)
        assert ctx is not None
        assert len(ctx) > 0
        # With empty bids and asks, ob_depth=0, skew should be 0
        # Verify the context string is still valid and contains the market question
        assert market.question in ctx
