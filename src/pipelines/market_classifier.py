"""Shared market type classification for Polymarket v2.

Single source of truth for the MARKET_TYPE_KEYWORDS dict and the
classify_market_type() function. Imported by polymarket.py, data_ingestion.py,
and mocks.py to avoid the drift that came from previously maintaining 3 copies.
"""
from __future__ import annotations


MARKET_TYPE_KEYWORDS: dict[str, list[str]] = {
    # Most specific / unambiguous types first to avoid false substring matches.
    "crypto_15m": [
        # Bare "eth"/"sol" removed — collide with "ethics", "method", "console",
        # "solid", etc. "btc" is kept (no realistic English-word collisions).
        # ETH/SOL ticker-only mentions must use the $-prefixed form to qualify.
        "bitcoin", "btc", "ethereum", "crypto", "solana",
        "$btc", "$eth", "$sol",
    ],
    "esports": [
        # Bare "lol" removed (matches "lollipop"). "lol:" and " lol " keep
        # the common Polymarket title patterns ("LoL: TeamA vs TeamB") working
        # without colliding with English words.
        "esports", "league of legends", "lol:", " lol ", "dota", "dota 2", "valorant",
        "counter-strike", "cs:go", "cs2", "honor of kings", "starcraft",
        "overwatch", "fortnite", "apex legends", "rocket league", "worlds",
    ],
    # geopolitical before weather: "ukraine" contains "rain" as a substring;
    # country/region keywords must be checked before the bare weather terms.
    "geopolitical": [
        "war", "ceasefire", "treaty", "sanction", "sanctions", "invasion",
        "border", "nato", "united nations", "un security council",
        "russia", "ukraine", "china", "taiwan", "israel", "gaza", "iran",
        "north korea", "peace deal", "diplomacy",
    ],
    "weather": [
        "temperature", "weather", "snowfall", "rainfall", "hurricane",
        "rainstorm", "snowstorm", "tornado", "heat wave", "celsius", "fahrenheit",
        "precipitation", "weather forecast",
        # standalone "rain " / " rain" patterns (with a trailing/leading space)
        # to avoid "ukraine" false-positive; note classify_market_type adds
        # spaces by checking q_lower which preserves the original spacing.
        " rain ", "raining", "rainy",
    ],
    "regulatory": [
        "sec", "regulation", "law", "ban", "approve", "fda", "ruling",
        "court", "lawsuit", "settlement", "subpoena",
    ],
    "economic": [
        "gdp", "inflation", "fed", "interest rate", "unemployment", "economy",
        "recession", "jobs", "cpi", "fomc", "earnings", "stocks", "s&p",
        "nasdaq", "dow",
    ],
    "political": [
        "president", "election", "congress", "senate", "vote", "political",
        "trump", "biden", "governor", "democrat", "republican", "primary",
        "campaign", "ballot",
    ],
    "sports": [
        # Leagues and sports
        "nba", "nfl", "mlb", "nhl", "epl", "ucl", "ufc", "mma", "tennis",
        "soccer", "football", "basketball", "baseball", "hockey", "golf",
        "championship", "super bowl", "world cup", "liga mx", "serie a",
        "premier league", "bundesliga", "la liga", "champions league",
        "playoff", "playoffs",
        # Game / match patterns
        "vs.", "vs ", " vs", "o/u", "over/under", "over under", "spread",
        "moneyline", "first half", "halftime",
        # Common scoring terms
        "goals", "points spread", "total goals", "total points",
    ],
    "cultural": [
        "oscar", "grammy", "emmy", "movie", "album", "show", "celebrity",
        "entertainment", "box office", "billboard",
    ],
}


def classify_market_type(question: str) -> str:
    """Classify a market question into one of the known market types.

    Returns the matching type name, or "unknown" if no keyword matches.
    Types are checked in iteration order of MARKET_TYPE_KEYWORDS — the
    first match wins. The "unknown" fallback is intentional: the learning
    system (MarketTypeManager, SignalTrackerManager) auto-creates entries
    for unknown types, and _DECAY_PARAMS falls back to _default.
    """
    q_lower = question.lower()
    for mtype, keywords in MARKET_TYPE_KEYWORDS.items():
        if any(kw in q_lower for kw in keywords):
            return mtype
    return "unknown"


# Per-market-type taker fee rates as of Q1 2026.
# Source: https://docs.polymarket.com/polymarket-learn/trading/fees
#
# Crypto markets: taker fees enabled Jan 2026, peak 1.56% at 50% probability.
# Sports markets: taker fees enabled Feb 2026 (NCAAB, Serie A first rollout).
# Political/economic/cultural/regulatory/weather/geopolitical: still 0%.
#
# These are conservative taker-side estimates. We use maker orders by default
# (TIER1_EXECUTION_TYPE=maker) which may earn rebates instead of paying fees,
# but maker/taker asymmetry is not yet modeled here. Treat as the cost ceiling.
MARKET_TYPE_FEES: dict[str, float] = {
    "political":    0.0,
    "geopolitical": 0.0,
    "economic":     0.0,
    "cultural":     0.0,
    "regulatory":   0.0,
    "weather":      0.0,
    "crypto_15m":   0.0156,  # peak 1.56% taker fee
    "sports":       0.02,    # placeholder — verify exact rate
    "esports":      0.02,    # assume same as sports
    "unknown":      0.02,    # conservative default
}


def get_fee_rate(market_type: str, default: float = 0.02) -> float:
    """Look up the taker fee rate for a given market type.

    Returns the MARKET_TYPE_FEES entry if present, else `default`.
    The default is intentionally conservative (2%) so that an unmapped
    type errs on the side of skipping marginal trades rather than
    taking them with underestimated costs.
    """
    return MARKET_TYPE_FEES.get(market_type, default)
