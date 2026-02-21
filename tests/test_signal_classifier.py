"""Tests for the signal source-tier classifier (S1-S6).

Covers all tier classification paths, edge cases, and the
SOURCE_TIER_CREDIBILITY mapping.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.pipelines.signal_classifier import classify_source_tier, SOURCE_TIER_CREDIBILITY


# ---------------------------------------------------------------------------
# S1 -- Official government / institutional primary sources
# ---------------------------------------------------------------------------


class TestS1Classification:
    """S1: Official government / institutional sources."""

    def test_rss_federalreserve_gov(self):
        sig = {"source_type": "rss", "domain": "federalreserve.gov"}
        assert classify_source_tier(sig) == "S1"

    def test_rss_sec_gov(self):
        sig = {"source_type": "rss", "domain": "sec.gov"}
        assert classify_source_tier(sig) == "S1"

    def test_twitter_whitehouse(self):
        sig = {"source_type": "twitter", "account_handle": "@WhiteHouse"}
        assert classify_source_tier(sig) == "S1"

    def test_twitter_federalreserve(self):
        sig = {"source_type": "twitter", "account_handle": "@FederalReserve"}
        assert classify_source_tier(sig) == "S1"

    def test_rss_subdomain_of_official(self):
        """feeds.federalreserve.gov should still match S1."""
        sig = {"source_type": "rss", "domain": "feeds.federalreserve.gov"}
        assert classify_source_tier(sig) == "S1"


# ---------------------------------------------------------------------------
# S2 -- Wire services
# ---------------------------------------------------------------------------


class TestS2Classification:
    """S2: Wire services (Reuters, AP, AFP)."""

    def test_rss_reuters(self):
        sig = {"source_type": "rss", "domain": "reuters.com"}
        assert classify_source_tier(sig) == "S2"

    def test_rss_apnews(self):
        sig = {"source_type": "rss", "domain": "apnews.com"}
        assert classify_source_tier(sig) == "S2"

    def test_twitter_reuters(self):
        sig = {"source_type": "twitter", "account_handle": "@Reuters"}
        assert classify_source_tier(sig) == "S2"

    def test_twitter_ap(self):
        sig = {"source_type": "twitter", "account_handle": "@AP"}
        assert classify_source_tier(sig) == "S2"

    def test_rss_subdomain_of_reuters(self):
        sig = {"source_type": "rss", "domain": "feeds.reuters.com"}
        assert classify_source_tier(sig) == "S2"


# ---------------------------------------------------------------------------
# S3 -- Institutional media
# ---------------------------------------------------------------------------


class TestS3Classification:
    """S3: Institutional media (BBC, CoinDesk, NYT, CNBC ...)."""

    def test_rss_bbc(self):
        sig = {"source_type": "rss", "domain": "bbc.com"}
        assert classify_source_tier(sig) == "S3"

    def test_rss_coindesk(self):
        sig = {"source_type": "rss", "domain": "coindesk.com"}
        assert classify_source_tier(sig) == "S3"

    def test_twitter_nytimes(self):
        sig = {"source_type": "twitter", "account_handle": "@nytimes"}
        assert classify_source_tier(sig) == "S3"

    def test_twitter_cnbc(self):
        sig = {"source_type": "twitter", "account_handle": "@CNBC"}
        assert classify_source_tier(sig) == "S3"


# ---------------------------------------------------------------------------
# S4 -- Verified domain experts
# ---------------------------------------------------------------------------


class TestS4Classification:
    """S4: Verified account + >=50k followers + expert bio keyword."""

    def test_verified_expert_with_enough_followers(self):
        sig = {
            "source_type": "twitter",
            "account_handle": "@SomeExpert",
            "is_verified": True,
            "follower_count": 60_000,
            "bio": "Senior economist at think-tank",
        }
        assert classify_source_tier(sig) == "S4"

    def test_verified_expert_exactly_50k(self):
        sig = {
            "source_type": "twitter",
            "account_handle": "@Edge",
            "is_verified": True,
            "follower_count": 50_000,
            "bio": "Journalist covering politics",
        }
        assert classify_source_tier(sig) == "S4"

    def test_verified_expert_bio_with_slash_delimiter(self):
        """Bio 'journalist/editor' should be tokenised correctly."""
        sig = {
            "source_type": "twitter",
            "account_handle": "@SlashBio",
            "is_verified": True,
            "follower_count": 100_000,
            "bio": "journalist/editor",
        }
        assert classify_source_tier(sig) == "S4"


class TestS4FailCases:
    """Situations that look like S4 but should fall to S6."""

    def test_no_expert_keyword_in_bio(self):
        sig = {
            "source_type": "twitter",
            "account_handle": "@NoKeyword",
            "is_verified": True,
            "follower_count": 100_000,
            "bio": "Just a regular person",
        }
        assert classify_source_tier(sig) == "S6"

    def test_below_50k_followers(self):
        sig = {
            "source_type": "twitter",
            "account_handle": "@SmallAcc",
            "is_verified": True,
            "follower_count": 49_999,
            "bio": "Economist at university",
        }
        assert classify_source_tier(sig) == "S6"

    def test_not_verified(self):
        sig = {
            "source_type": "twitter",
            "account_handle": "@Unverified",
            "is_verified": False,
            "follower_count": 200_000,
            "bio": "Professor of economics",
        }
        assert classify_source_tier(sig) == "S6"

    def test_empty_bio(self):
        sig = {
            "source_type": "twitter",
            "account_handle": "@NoBio",
            "is_verified": True,
            "follower_count": 100_000,
            "bio": "",
        }
        assert classify_source_tier(sig) == "S6"


# ---------------------------------------------------------------------------
# S5 -- Market data
# ---------------------------------------------------------------------------


class TestS5Classification:
    """S5: source_type='market_data' always returns S5."""

    def test_market_data(self):
        sig = {"source_type": "market_data"}
        assert classify_source_tier(sig) == "S5"

    def test_market_data_with_extra_fields(self):
        sig = {
            "source_type": "market_data",
            "domain": "something.com",
            "account_handle": "@whatever",
        }
        assert classify_source_tier(sig) == "S5"

    def test_market_data_case_insensitive(self):
        sig = {"source_type": "Market_Data"}
        assert classify_source_tier(sig) == "S5"


# ---------------------------------------------------------------------------
# S6 -- Unknown / fallback
# ---------------------------------------------------------------------------


class TestS6Classification:
    """S6: Unknown domains, handles, or source types."""

    def test_unknown_rss_domain(self):
        sig = {"source_type": "rss", "domain": "randomblog.xyz"}
        assert classify_source_tier(sig) == "S6"

    def test_unknown_twitter_handle(self):
        sig = {"source_type": "twitter", "account_handle": "@randomuser123"}
        assert classify_source_tier(sig) == "S6"

    def test_unknown_source_type(self):
        sig = {"source_type": "telegram"}
        assert classify_source_tier(sig) == "S6"

    def test_empty_source_type(self):
        sig = {"source_type": ""}
        assert classify_source_tier(sig) == "S6"

    def test_missing_source_type(self):
        sig = {}
        assert classify_source_tier(sig) == "S6"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Case insensitivity, missing fields, whitespace."""

    def test_twitter_handle_case_insensitive(self):
        sig = {"source_type": "twitter", "account_handle": "@whitehouse"}
        assert classify_source_tier(sig) == "S1"

    def test_twitter_handle_uppercase(self):
        sig = {"source_type": "twitter", "account_handle": "@REUTERS"}
        assert classify_source_tier(sig) == "S2"

    def test_rss_domain_case_insensitive(self):
        sig = {"source_type": "rss", "domain": "FederalReserve.Gov"}
        assert classify_source_tier(sig) == "S1"

    def test_rss_domain_with_www_prefix(self):
        sig = {"source_type": "rss", "domain": "www.reuters.com"}
        assert classify_source_tier(sig) == "S2"

    def test_rss_domain_with_https_prefix(self):
        sig = {"source_type": "rss", "domain": "https://bbc.com"}
        assert classify_source_tier(sig) == "S3"

    def test_rss_domain_with_trailing_slash(self):
        sig = {"source_type": "rss", "domain": "reuters.com/"}
        assert classify_source_tier(sig) == "S2"

    def test_source_type_with_whitespace(self):
        sig = {"source_type": " twitter ", "account_handle": "@Reuters"}
        assert classify_source_tier(sig) == "S2"

    def test_rss_missing_domain(self):
        sig = {"source_type": "rss"}
        assert classify_source_tier(sig) == "S6"

    def test_rss_empty_domain(self):
        sig = {"source_type": "rss", "domain": ""}
        assert classify_source_tier(sig) == "S6"

    def test_twitter_missing_handle(self):
        sig = {"source_type": "twitter"}
        assert classify_source_tier(sig) == "S6"

    def test_twitter_handle_without_at_prefix(self):
        """Handle provided without '@' should still be matched."""
        sig = {"source_type": "twitter", "account_handle": "Reuters"}
        assert classify_source_tier(sig) == "S2"

    def test_follower_count_none(self):
        """follower_count=None should not crash S4 check."""
        sig = {
            "source_type": "twitter",
            "account_handle": "@SomeoneNew",
            "is_verified": True,
            "follower_count": None,
            "bio": "Professor of law",
        }
        assert classify_source_tier(sig) == "S6"


# ---------------------------------------------------------------------------
# SOURCE_TIER_CREDIBILITY mapping
# ---------------------------------------------------------------------------


class TestSourceTierCredibility:
    """Verify the credibility values for each tier."""

    def test_s1_credibility(self):
        assert SOURCE_TIER_CREDIBILITY["S1"] == 0.95

    def test_s2_credibility(self):
        assert SOURCE_TIER_CREDIBILITY["S2"] == 0.90

    def test_s3_credibility(self):
        assert SOURCE_TIER_CREDIBILITY["S3"] == 0.80

    def test_s4_credibility(self):
        assert SOURCE_TIER_CREDIBILITY["S4"] == 0.65

    def test_s5_credibility(self):
        assert SOURCE_TIER_CREDIBILITY["S5"] == 0.70

    def test_s6_credibility(self):
        assert SOURCE_TIER_CREDIBILITY["S6"] == 0.30

    def test_all_tiers_present(self):
        assert set(SOURCE_TIER_CREDIBILITY.keys()) == {"S1", "S2", "S3", "S4", "S5", "S6"}

    def test_s1_is_highest_credibility(self):
        assert SOURCE_TIER_CREDIBILITY["S1"] == max(SOURCE_TIER_CREDIBILITY.values())

    def test_s6_is_lowest_credibility(self):
        assert SOURCE_TIER_CREDIBILITY["S6"] == min(SOURCE_TIER_CREDIBILITY.values())
