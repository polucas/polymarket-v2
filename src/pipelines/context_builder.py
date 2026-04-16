from __future__ import annotations

import re
from datetime import datetime, timezone
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
    "weather": ["temperature", "forecast", "precipitation"],
    "esports": ["gaming", "tournament", "match"],
    "geopolitical": ["international", "conflict", "diplomacy"],
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


def _format_signal_age(signal: Signal) -> str:
    """Format signal age as human-readable string."""
    if not signal.timestamp:
        return "age unknown"
    age_min = (datetime.now(timezone.utc) - signal.timestamp).total_seconds() / 60
    if age_min < 0:
        return "future?"
    if age_min < 120:
        return f"{int(age_min)}min ago"
    return f"{age_min / 60:.1f}h ago"


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
    total_bids = sum(l.size for l in orderbook.bids) if orderbook.bids else 0
    total_asks = sum(l.size for l in orderbook.asks) if orderbook.asks else 0
    ob_depth = total_bids + total_asks
    ob_skew = (total_bids - total_asks) / max(ob_depth, 1) if ob_depth > 0 else 0

    # Build signal text with age
    signal_lines = []
    for i, s in enumerate(all_signals, 1):
        source_label = f"[{s.source_tier}|{s.source}]"
        age_str = _format_signal_age(s)
        headline_tag = " [HEADLINE-ONLY]" if s.headline_only else ""
        signal_lines.append(
            f"  {i}. {source_label} {age_str} (cred={s.credibility:.2f}): "
            f"{s.content[:200]}{headline_tag}"
        )

    signals_text = "\n".join(signal_lines) if signal_lines else "  No signals available."

    try:
        spread_val = orderbook.spread
        spread_line = (f"\nSpread: {spread_val:.4f}" if spread_val and spread_val > 0 else "")
    except (TypeError, AttributeError):
        spread_line = ""

    market_type_label = market.market_type or "unknown"

    context = f"""MARKET ANALYSIS — {market_type_label}

Question: {market.question}
Current YES price: {market.yes_price:.4f} (market consensus)
Current NO price: {market.no_price:.4f}
Resolution: {market.hours_to_resolution:.1f} hours
Orderbook depth: ${ob_depth:,.0f} (skew: {ob_skew:+.2f}){spread_line}

SIGNALS:
{signals_text}

INSTRUCTIONS:
1. Consider why the market price ({market.yes_price:.4f}) might already be correct
2. Identify specific evidence from signals that suggests mispricing
3. If signals are stale, weak, or ambiguous, stay close to market price
4. Estimate true probability and your confidence

Return ONLY: {{"estimated_probability": 0.XX, "confidence": 0.XX, "reasoning": "Why market is wrong: [evidence]. Counter: [why market might be right]."}}"""

    return context


def build_prescreen_context(
    market: Market,
    rss_signals: List[Signal],
    orderbook: OrderBook,
) -> str:
    """Build a lightweight context for pre-screen LLM call (no Twitter signals)."""
    total_bids = sum(l.size for l in orderbook.bids) if orderbook.bids else 0
    total_asks = sum(l.size for l in orderbook.asks) if orderbook.asks else 0
    ob_depth = total_bids + total_asks
    ob_skew = (total_bids - total_asks) / max(ob_depth, 1) if ob_depth > 0 else 0

    # Top 3 RSS signals only
    top_rss = sorted(rss_signals, key=lambda s: s.credibility, reverse=True)[:3]
    signal_lines = []
    for i, s in enumerate(top_rss, 1):
        age_str = _format_signal_age(s)
        signal_lines.append(f"  {i}. [{s.source_tier}|{s.source}] {age_str}: {s.content[:150]}")
    signals_text = "\n".join(signal_lines) if signal_lines else "  No signals."

    return f"""QUICK SCREEN — is this market likely mispriced?
Question: {market.question}
YES: {market.yes_price:.4f}, NO: {market.no_price:.4f}
Resolution: {market.hours_to_resolution:.1f}h
OB depth: ${ob_depth:,.0f} (skew: {ob_skew:+.2f})

{signals_text}

Return ONLY: {{"estimated_probability": 0.XX, "confidence": 0.XX, "reasoning": "one sentence"}}"""
