"""Tests for src/pipelines/market_classifier.py.

Covers all 9 known market types, the "unknown" fallback, ordering-sensitive
cases, case insensitivity, and real backtest miss cases.
"""
from __future__ import annotations

import pytest
from src.pipelines.market_classifier import classify_market_type, MARKET_TYPE_KEYWORDS


# ---------------------------------------------------------------------------
# Basic per-type coverage
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "question, expected",
    [
        # crypto_15m
        ("Will Bitcoin BTC hit $100k by December?", "crypto_15m"),
        ("ETH ethereum price above $5,000?", "crypto_15m"),
        ("Will Solana SOL reach $300?", "crypto_15m"),
        # esports (must come before sports in iteration order)
        ("LoL: Sentinels vs Disguised - Game 2 Winner", "esports"),
        ("Honor of Kings: AG Super Play vs Qing Jiu Club - Game 3 Winner", "esports"),
        ("Valorant Champions: LOUD vs NRG Game 1", "esports"),
        # weather (must come before sports — "rain" should not fall to sports)
        ("Will a cozy rain occur on Friday?", "weather"),
        ("Will the highest temperature in Chicago be between 6-7°F on January 15?", "weather"),
        ("Will there be a hurricane landfall this week?", "weather"),
        # geopolitical (must come before political — "war" / "ceasefire" should not fall to political)
        ("Ukraine ceasefire agreement before July?", "geopolitical"),
        ("Will US impose sanctions on Iran this month?", "geopolitical"),
        ("Taiwan invasion risk by end of 2026?", "geopolitical"),
        # regulatory
        ("Will the SEC approve the new stablecoin regulation?", "regulatory"),
        ("FDA approval of new Alzheimer's drug by Q2?", "regulatory"),
        # economic
        ("Will the Fed raise interest rates in June?", "economic"),
        ("US GDP growth above 2% in Q1 2026?", "economic"),
        # political
        ("Will Trump win the 2024 election?", "political"),
        ("Jair Bolsonaro in jail by January 31?", "unknown"),  # no political keyword matches
        ("Will the Democrat win the Senate primary?", "political"),
        # sports (generic match patterns)
        ("CF Cruz Azul vs. Club Puebla: O/U 1.5", "sports"),
        ("FK Bodø/Glimt vs. Manchester City FC: O/U 3.5", "sports"),
        ("Will the NBA finals go to game 7?", "sports"),
        ("Super Bowl LIX winner?", "sports"),
        # cultural
        ("Will this movie win an Oscar?", "cultural"),
        ("Grammy Album of the Year 2026?", "cultural"),
        # unknown fallback
        ("Will the next major product launch be successful?", "unknown"),
        ("Will the color blue be popular next year?", "unknown"),
    ],
)
def test_classify_market_type(question: str, expected: str) -> None:
    assert classify_market_type(question) == expected


# ---------------------------------------------------------------------------
# Ordering-sensitive: esports before sports
# ---------------------------------------------------------------------------

def test_esports_before_sports_lol_vs() -> None:
    """'LoL' in esports list should win over 'vs' in sports list."""
    result = classify_market_type("LoL: Sentinels vs Disguised - Game 2 Winner")
    assert result == "esports"


def test_honor_of_kings_is_esports_not_sports() -> None:
    result = classify_market_type(
        "Honor of Kings vs LoL: O/U 2.5 maps?"
    )
    assert result == "esports"


def test_sports_vs_pattern_without_esports() -> None:
    """Generic 'vs.' with no esports keyword lands in sports."""
    result = classify_market_type("Cruz Azul vs. Club Puebla: O/U 1.5")
    assert result == "sports"


# ---------------------------------------------------------------------------
# Ordering-sensitive: geopolitical before political
# ---------------------------------------------------------------------------

def test_ukraine_ceasefire_is_geopolitical_not_political() -> None:
    """'ceasefire' keyword in geopolitical should win over any political match."""
    result = classify_market_type("Ukraine ceasefire agreement?")
    assert result == "geopolitical"


def test_war_is_geopolitical_not_political() -> None:
    result = classify_market_type("Russia-Ukraine war ends by 2026?")
    assert result == "geopolitical"


# ---------------------------------------------------------------------------
# Ordering-sensitive: weather before sports
# ---------------------------------------------------------------------------

def test_rain_is_weather_not_unknown() -> None:
    result = classify_market_type("Will it rain in London on Friday?")
    assert result == "weather"


# ---------------------------------------------------------------------------
# Case insensitivity
# ---------------------------------------------------------------------------

def test_case_insensitive_upper() -> None:
    assert classify_market_type("BITCOIN above $100k?") == "crypto_15m"


def test_case_insensitive_mixed() -> None:
    assert classify_market_type("Will THE FED raise INTEREST RATE?") == "economic"


# ---------------------------------------------------------------------------
# Substring-collision regressions: bare "sol"/"eth"/"btc"/"lol" must not match
# ---------------------------------------------------------------------------

def test_console_is_not_crypto() -> None:
    """'console' contains 'sol' — must not classify as crypto_15m."""
    assert classify_market_type("Will the next gaming console outsell PS5?") != "crypto_15m"


def test_ethics_is_not_crypto() -> None:
    """'ethics' contains 'eth' — must not classify as crypto_15m."""
    assert classify_market_type("Will the AI ethics bill pass committee?") != "crypto_15m"


def test_method_is_not_crypto() -> None:
    """'method' contains 'eth' — must not classify as crypto_15m."""
    assert classify_market_type("Will the new payment method launch in Q2?") != "crypto_15m"


def test_lollipop_is_not_esports() -> None:
    """'lollipop' contains 'lol' — must not classify as esports."""
    assert classify_market_type("Will Lollipop Records release a new album?") != "esports"


def test_dollar_btc_still_crypto() -> None:
    """$-prefixed tickers must still resolve to crypto_15m."""
    assert classify_market_type("Will $BTC close above $100k?") == "crypto_15m"
    assert classify_market_type("$ETH gas fees this week?") == "crypto_15m"
    assert classify_market_type("$SOL TVL ranking?") == "crypto_15m"


# ---------------------------------------------------------------------------
# MARKET_TYPE_KEYWORDS dict sanity
# ---------------------------------------------------------------------------

def test_all_expected_types_present() -> None:
    expected_types = {
        "crypto_15m", "esports", "weather", "geopolitical",
        "regulatory", "economic", "political", "sports", "cultural",
    }
    assert expected_types == set(MARKET_TYPE_KEYWORDS.keys())


def test_iteration_order() -> None:
    """esports must precede sports, geopolitical must precede both weather and political."""
    keys = list(MARKET_TYPE_KEYWORDS.keys())
    assert keys.index("esports") < keys.index("sports"), "esports must come before sports"
    assert keys.index("geopolitical") < keys.index("political"), "geopolitical must come before political"
    assert keys.index("geopolitical") < keys.index("weather"), "geopolitical must come before weather (ukraine contains 'rain')"
    assert keys.index("weather") < keys.index("sports"), "weather must come before sports"
    assert keys.index("crypto_15m") < keys.index("political"), "crypto_15m must come before political"
