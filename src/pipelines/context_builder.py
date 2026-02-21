from __future__ import annotations

import re
from typing import Dict, List, Optional

import structlog

from src.models import Market, Signal, OrderBook

log = structlog.get_logger()

# Cache keywords per market
_keyword_cache: Dict[str, List[str]] = {}

ENTITY_PATTERNS = [
    re.compile(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\b'),  # Named entities (e.g., "Donald Trump")
    re.compile(r'\b[A-Z]{2,6}\b'),  # Acronyms (e.g., "GDP", "FBI")
    re.compile(r'\$[A-Z]{1,5}\b'),  # Tickers (e.g., "$BTC", "$AAPL")
]

KEYWORD_SUPPLEMENTS = {
    "political": ["election", "vote", "polls"],
    "economic": ["economy", "market", "federal reserve"],
    "crypto_15m": ["crypto", "bitcoin", "trading"],
    "sports": ["game", "match", "score"],
    "cultural": ["entertainment", "media"],
    "regulatory": ["regulation", "policy", "ruling"],
}


def extract_keywords(market_id: str, market_question: str, market_type: str) -> List[str]:
    """Extract search keywords from market question.
    Uses regex first, LLM fallback only if <2 entities found.
    """
    if market_id in _keyword_cache:
        return _keyword_cache[market_id]

    entities = set()
    for pattern in ENTITY_PATTERNS:
        matches = pattern.findall(market_question)
        for m in matches:
            cleaned = m.strip().strip("$")
            if len(cleaned) > 1 and cleaned.upper() not in {"THE", "AND", "FOR", "BUT", "NOT", "YES", "WILL", "BE", "BY", "IN", "ON", "AT", "TO"}:
                entities.add(cleaned)

    # Add market type supplements
    supplements = KEYWORD_SUPPLEMENTS.get(market_type, [])

    keywords = list(entities)
    if len(keywords) < 2:
        # Add supplements when regex doesn't find enough
        keywords.extend(supplements[:3])

    # Always include significant words from question as fallback
    if not keywords:
        words = [w for w in market_question.split() if len(w) > 4]
        keywords = words[:5]

    _keyword_cache[market_id] = keywords
    return keywords


def build_grok_context(
    market: Market,
    twitter_signals: List[Signal],
    rss_signals: List[Signal],
    orderbook: OrderBook,
) -> str:
    """Build the context prompt for Grok LLM call."""
    # Merge and sort signals by credibility, take top 7
    all_signals = sorted(
        twitter_signals + rss_signals,
        key=lambda s: s.credibility,
        reverse=True,
    )[:7]

    # Orderbook depth and skew
    total_bids = sum(orderbook.bids) if orderbook.bids else 0
    total_asks = sum(orderbook.asks) if orderbook.asks else 0
    ob_depth = total_bids + total_asks
    ob_skew = (total_bids - total_asks) / max(ob_depth, 1) if ob_depth > 0 else 0

    # Build signal text
    signal_lines = []
    for i, s in enumerate(all_signals, 1):
        source_label = f"[{s.source_tier}|{s.source}]"
        headline_tag = " [HEADLINE-ONLY]" if s.headline_only else ""
        signal_lines.append(f"  {i}. {source_label} @{s.author} (cred={s.credibility:.2f}): {s.content[:200]}{headline_tag}")

    signals_text = "\n".join(signal_lines) if signal_lines else "  No signals available."

    context = f"""MARKET ANALYSIS REQUEST

Market Question: {market.question}
Current YES price: {market.yes_price:.4f}
Current NO price: {market.no_price:.4f}
Resolution: {market.hours_to_resolution:.1f} hours
Volume (24h): ${market.volume_24h:,.0f}
Liquidity: ${market.liquidity:,.0f}
Orderbook depth: ${ob_depth:,.0f} (skew: {ob_skew:+.2f})

SIGNALS:
{signals_text}

INSTRUCTIONS:
1. Analyze the signals and market context
2. Classify each signal's information type:
   - I1: Verified fact (official announcement, confirmed event)
   - I2: Authoritative analysis (expert opinion, institutional report)
   - I3: Statistical/data-driven (polls, economic indicators)
   - I4: Market intelligence (order flow, whale movements)
   - I5: Rumor/speculation (unconfirmed reports, social media buzz)
3. Estimate the probability of YES outcome
4. Rate your confidence in the estimate

Respond with ONLY this JSON (no markdown, no extra text):
{{"estimated_probability": 0.XX, "confidence": 0.XX, "reasoning": "...", "signal_info_types": [{{"source_tier": "SX", "info_type": "IX", "content_summary": "..."}}]}}"""

    return context
