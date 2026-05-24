"""Tests for keyword extraction and Grok context building."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.models import Market, Signal, OrderBook, OrderBookLevel
from src.pipelines.context_builder import (
    extract_keywords,
    build_grok_context,
    build_prescreen_context,
    _format_signal_age,
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
    """Build an OrderBook from lists of sizes (for backward compat) or OrderBookLevel objects."""
    def _to_levels(values: list, is_bids: bool) -> list:
        if not values:
            return []
        if isinstance(values[0], OrderBookLevel):
            return values
        # plain float sizes — assign placeholder prices
        price_start = 0.50
        step = -0.01 if is_bids else 0.01
        return [OrderBookLevel(price=price_start + i * step, size=v) for i, v in enumerate(values)]

    raw_bids = bids if bids is not None else [100.0, 80.0, 60.0]
    raw_asks = asks if asks is not None else [90.0, 70.0, 50.0]
    return OrderBook(
        market_id=market_id,
        bids=_to_levels(raw_bids, is_bids=True),
        asks=_to_levels(raw_asks, is_bids=False),
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
# extract_keywords — F5 additions (single-word entity, prefix stripping, stopwords)
# ---------------------------------------------------------------------------


class TestExtractKeywordsF5:
    def test_single_word_team_extracted(self):
        """Single-word teams ≥6 chars (new regex) are extracted."""
        kw = extract_keywords("m1", "Canadiens vs. Hurricanes: O/U 5.5", "sports")
        assert "Hurricanes" in kw
        assert "Canadiens" in kw

    def test_will_prefix_stripped(self):
        """Leading 'Will' is stripped from multi-word entities by the prefix filter."""
        kw = extract_keywords("m2", "Will John Cornyn win the 2026 Texas Republican Primary?", "political")
        assert "John Cornyn" in kw
        assert "Will John Cornyn" not in kw

    def test_fc_filtered_from_acronyms(self):
        """'FC' is in the expanded stopword set and must not appear as a keyword."""
        kw = extract_keywords("m3", "Will Caracas FC win on 2026-05-21?", "sports")
        assert "FC" not in kw
        assert "Caracas" in kw

    def test_short_capitalized_word_not_overmatched(self):
        """Words shorter than 6 chars (Will/May) must not be captured by the single-word regex."""
        kw = extract_keywords("m4", "Will it rain in Madrid on May 25?", "weather")
        # Madrid is 6 chars — allowed
        assert "Madrid" in kw
        # 'Will' (4 chars) and 'May' (3 chars) are not ≥6 chars for single-word regex,
        # and 'Will' is stripped as a prefix from multi-word matches.
        assert "Will" not in kw
        assert "May" not in kw

    def test_multi_word_still_captured(self):
        """Regression: multi-word entities like 'Donald Trump' are still captured after F5 changes."""
        kw = extract_keywords("m5", "Will Donald Trump announce military action?", "political")
        assert "Donald Trump" in kw


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
            _make_signal(content=f"Signal {i} unique_{i}", credibility=0.1 * i, author=f"auth{i}")
            for i in range(1, 11)  # 10 signals
        ]
        ctx = build_grok_context(market, signals, [], _make_orderbook())

        # Highest credibility is 1.0 (i=10), lowest included is 0.4 (i=4)
        # The top 7: i=10,9,8,7,6,5,4
        assert "unique_10" in ctx
        assert "unique_9" in ctx
        assert "unique_4" in ctx
        # i=3 (0.3 cred) should be excluded
        assert "unique_3" not in ctx

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
        # signal_info_types removed from prompt: info_type is now assigned deterministically
        assert "signal_info_types" not in ctx

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
            _make_signal(content="Twitter high unique_tw_top", credibility=0.95, source="twitter", author="tw_top"),
            _make_signal(content="Twitter low unique_tw_low", credibility=0.30, source="twitter", author="tw_low"),
        ]
        rss_signals = [
            _make_signal(content="RSS mid unique_rss_mid", credibility=0.70, source="rss", author="rss_mid"),
        ]
        ctx = build_grok_context(market, twitter_signals, rss_signals, _make_orderbook())

        # All three should appear (total <= 7)
        assert "unique_tw_top" in ctx
        assert "unique_tw_low" in ctx
        assert "unique_rss_mid" in ctx

        # Verify ordering: tw_top (0.95) should come before rss_mid (0.70)
        pos_tw_top = ctx.index("unique_tw_top")
        pos_rss_mid = ctx.index("unique_rss_mid")
        pos_tw_low = ctx.index("unique_tw_low")
        assert pos_tw_top < pos_rss_mid < pos_tw_low

    def test_output_contains_resolution_time(self):
        """#37 — Output string contains resolution time (hours_to_resolution)."""
        market = _make_market(hours_to_resolution=36.5)
        ctx = build_grok_context(market, [], [], _make_orderbook())
        assert "36.5" in ctx

    def test_output_does_not_contain_volume_and_liquidity(self):
        """Volume/liquidity removed from prompt — irrelevant noise for probability estimation."""
        market = _make_market(volume_24h=123_456.0, liquidity=78_900.0)
        ctx = build_grok_context(market, [], [], _make_orderbook())
        assert "$123,456" not in ctx
        assert "$78,900" not in ctx

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

    def test_prompt_does_not_include_info_type_classification_instructions(self):
        """#42 — Prompt no longer asks Grok to classify I1-I5; this is now deterministic."""
        market = _make_market()
        ctx = build_grok_context(market, [], [], _make_orderbook())
        # info_type classification removed from prompt — assigned deterministically from source_tier
        assert "Verified fact" not in ctx
        assert "Authoritative analysis" not in ctx
        assert "signal_info_types" not in ctx

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

    def test_output_contains_market_type(self):
        """Prompt header includes market type for domain-aware reasoning."""
        market = _make_market(market_type="sports")
        ctx = build_grok_context(market, [], [], _make_orderbook())
        assert "MARKET ANALYSIS — sports" in ctx

    def test_output_contains_market_consensus_label(self):
        """YES price labeled as 'market consensus' for anchoring."""
        market = _make_market(yes_price=0.62)
        ctx = build_grok_context(market, [], [], _make_orderbook())
        assert "market consensus" in ctx

    def test_output_contains_adversarial_instructions(self):
        """Prompt asks LLM to consider why market price might be correct."""
        market = _make_market()
        ctx = build_grok_context(market, [], [], _make_orderbook())
        assert "might already be correct" in ctx
        assert "why market might be right" in ctx.lower() or "Counter" in ctx

    def test_signal_age_shown_in_output(self):
        """Signals display age (e.g., '45min ago') instead of @author."""
        from datetime import timedelta
        market = _make_market()
        sig = _make_signal(content="Breaking news")
        sig.timestamp = datetime.now(timezone.utc) - timedelta(minutes=45)
        ctx = build_grok_context(market, [sig], [], _make_orderbook())
        assert "45min ago" in ctx

    def test_no_signals_text(self):
        """When no signals, shows 'No signals available'."""
        market = _make_market()
        ctx = build_grok_context(market, [], [], _make_orderbook())
        assert "No signals available" in ctx


# ---------------------------------------------------------------------------
# _format_signal_age
# ---------------------------------------------------------------------------


class TestFormatSignalAge:
    def test_none_timestamp(self):
        sig = _make_signal()
        sig.timestamp = None
        assert _format_signal_age(sig) == "age unknown"

    def test_recent_signal(self):
        from datetime import timedelta
        sig = _make_signal()
        sig.timestamp = datetime.now(timezone.utc) - timedelta(minutes=30)
        assert "30min ago" in _format_signal_age(sig)

    def test_old_signal_hours(self):
        from datetime import timedelta
        sig = _make_signal()
        sig.timestamp = datetime.now(timezone.utc) - timedelta(hours=3)
        result = _format_signal_age(sig)
        assert "h ago" in result


# ---------------------------------------------------------------------------
# build_prescreen_context
# ---------------------------------------------------------------------------


class TestBuildPrescreenContext:
    def test_no_twitter_signals_in_prescreen(self):
        """Pre-screen context takes no Twitter signals parameter."""
        market = _make_market()
        ctx = build_prescreen_context(market, [], _make_orderbook())
        assert "FAST SCREEN" in ctx
        assert market.question in ctx

    def test_max_3_rss_signals(self):
        """Pre-screen caps at 3 RSS signals."""
        market = _make_market()
        signals = [
            _make_signal(content=f"RSS signal {i}", credibility=0.1 * i, source="rss")
            for i in range(1, 8)  # 7 signals
        ]
        ctx = build_prescreen_context(market, signals, _make_orderbook())
        # Count signal lines (numbered 1., 2., 3.)
        assert "  1." in ctx
        assert "  3." in ctx
        assert "  4." not in ctx

    def test_prescreen_contains_prices(self):
        market = _make_market(yes_price=0.75, no_price=0.25)
        ctx = build_prescreen_context(market, [], _make_orderbook())
        assert "0.7500" in ctx
        assert "0.2500" in ctx

    def test_prescreen_no_volume_liquidity(self):
        market = _make_market(volume_24h=999_999.0, liquidity=888_888.0)
        ctx = build_prescreen_context(market, [], _make_orderbook())
        assert "999,999" not in ctx
        assert "888,888" not in ctx

    def test_prescreen_context_does_not_anchor_prices_first(self):
        """Prescreen prompt leads with question + signals, not prices."""
        market = _make_market(yes_price=0.72)
        signals = [_make_signal(content="Fed cuts rates 50bps", credibility=0.85)]
        ob = _make_orderbook()
        ctx = build_prescreen_context(market, signals, ob)
        # Header reframes task
        assert "independent probability" in ctx.lower()
        assert "mispriced" not in ctx.lower()
        # Question appears before prices
        q_pos = ctx.find("Question:")
        price_pos = ctx.find("YES:")
        assert q_pos >= 0 and price_pos >= 0 and q_pos < price_pos
        # Signals appear before prices
        sig_pos = ctx.find("SIGNALS")
        assert sig_pos >= 0 and sig_pos < price_pos
        # Explicit de-anchoring instruction present
        assert "do not anchor" in ctx.lower()
        # Prices still present (audit requirement)
        assert "0.7200" in ctx


# ---------------------------------------------------------------------------
# extract_keywords — F6 additions (hyphenated entities, sports stopwords,
#                    league acronyms)
# ---------------------------------------------------------------------------


class TestExtractKeywordsF6:
    def test_hyphenated_entity_captured(self):
        """Hyphenated proper names like 'Counter-Strike' are captured."""
        kw = extract_keywords("m_hy1", "Counter-Strike: MIBR vs Legacy - Map 1 Winner", "esports")
        assert "Counter-Strike" in kw
        # Generic noise must be filtered by the expanded stopword set
        assert "Winner" not in kw
        assert "Map" not in kw

    def test_generic_sports_stopwords_filtered(self):
        """Generic esports words like 'Winner' and 'Game' are filtered out."""
        kw = extract_keywords("m_sw1", "LoL: Karmine Corp vs G2 Esports - Game 4 Winner", "esports")
        assert "Winner" not in kw
        assert "Game" not in kw
        # Real entity still captured
        assert any("Karmine" in k for k in kw)

    def test_league_acronyms_filtered(self):
        """League acronyms like NHL are filtered; team names are kept."""
        kw = extract_keywords("m_la1", "NHL Stanley Cup: Avalanche vs Hurricanes", "sports")
        assert "NHL" not in kw
        # Teams are ≥6-char single-word entities — must survive
        assert "Avalanche" in kw
        assert "Hurricanes" in kw

    def test_promotions_filtered_standalone(self):
        """Bare 'Promotions' token is filtered; real names in the question survive."""
        kw = extract_keywords("m_pr1", "Brand Risk Promotions 14: Johnny Manziel vs. Bob Menery", "sports")
        # Standalone "Promotions" must not appear as its own keyword
        assert "Promotions" not in kw
        # Real proper names must still come through
        assert any("Manziel" in k for k in kw)

    def test_regression_existing_extraction_preserved(self):
        """F5 cases are not broken by F6 stopword additions."""
        cases = [
            ("Will Donald Trump announce military action?", "political", "Donald Trump"),
            ("Knicks vs. Cavaliers: 1H O/U 109.5", "sports", "Knicks"),
            ("Spread: Avalanche (-1.5)", "sports", "Avalanche"),
            ("Will Carolina Hurricanes win on 2026-06-01?", "sports", "Hurricanes"),
        ]
        for q, mt, expected in cases:
            _keyword_cache.clear()
            kw = extract_keywords(f"m_reg_{expected}", q, mt)
            assert any(expected in k for k in kw), f"{q} missing '{expected}', got {kw}"
