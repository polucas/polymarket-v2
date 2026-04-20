from __future__ import annotations

import email.utils
from datetime import datetime, timezone
from typing import List

import httpx
import structlog

from src.config import Settings
from src.models import Signal
from src.pipelines.signal_classifier import classify_source_tier, classify_info_type, SOURCE_TIER_CREDIBILITY

log = structlog.get_logger()

BASE_URL = "https://api.twitterapi.io/twitter"


class TwitterDataPipeline:
    def __init__(self, settings: Settings):
        self._api_key = settings.TWITTER_API_KEY
        self._timeout = httpx.Timeout(15.0, connect=5.0)

    async def get_signals_for_market(self, keywords: List[str], max_tweets: int = 50) -> List[Signal]:
        if not keywords:
            return []

        query = " OR ".join(keywords)
        raw_tweets = await self._search_tweets(query, max_results=max_tweets)

        # Pre-filter: followers >= 1000, engagement >= 10, not a bot
        filtered = [
            t for t in raw_tweets
            if (t.get("author") or {}).get("followers", 0) >= 1000
            and self._compute_engagement(t) >= 10
            and not self._is_bot_account(t.get("author") or {})
        ]

        deduplicated = self._deduplicate_by_content_similarity(filtered)

        # Classify and score
        scored = []
        for tweet in deduplicated:
            author = tweet.get("author") or {}
            source_tier = classify_source_tier({
                "source_type": "twitter",
                "account_handle": f"@{author.get('userName', '')}",
                "is_verified": bool(author.get("isVerified") or author.get("isBlueVerified")),
                "follower_count": int(author.get("followers") or 0),
                "bio": author.get("description") or "",
            })
            credibility = SOURCE_TIER_CREDIBILITY.get(source_tier, 0.30)

            created_at = self._parse_twitter_date(tweet.get("createdAt", ""))
            if created_at is None:
                created_at = datetime.now(timezone.utc)

            scored.append((credibility, source_tier, tweet, created_at))

        scored.sort(key=lambda x: x[0], reverse=True)

        signals = []
        for cred, st, tw, ts in scored[:10]:
            author = tw.get("author") or {}
            signals.append(Signal(
                source="twitter",
                source_tier=st,
                info_type=classify_info_type(st),
                content=(tw.get("text", "") or "")[:280],
                credibility=cred,
                author=author.get("userName", ""),
                followers=int(author.get("followers") or 0),
                engagement=self._compute_engagement(tw),
                timestamp=ts,
                headline_only=False,
            ))

        log.info(
            "twitter_signals_fetched",
            query_keywords=len(keywords),
            raw_count=len(raw_tweets),
            filtered_count=len(filtered),
            deduplicated_count=len(deduplicated),
            signal_count=len(signals),
        )
        return signals

    async def _search_tweets(self, query: str, max_results: int = 50) -> list:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(
                    f"{BASE_URL}/tweet/advanced_search",
                    headers={"X-API-Key": self._api_key},
                    params={
                        "query": query,
                        "queryType": "Top",
                        "cursor": "",
                    },
                )
                if resp.status_code == 429:
                    log.warning("twitter_rate_limited")
                    return []
                resp.raise_for_status()
                data = resp.json()
                return (data.get("tweets") or data.get("data") or [])[:max_results]
        except httpx.TimeoutException:
            log.warning("twitter_timeout", query=query)
            return []
        except Exception as e:
            log.error("twitter_search_failed", error=str(e))
            return []

    @staticmethod
    def _compute_engagement(tweet: dict) -> int:
        """Sum likeCount + retweetCount + replyCount + quoteCount from current API schema."""
        return (
            int(tweet.get("likeCount") or 0)
            + int(tweet.get("retweetCount") or 0)
            + int(tweet.get("replyCount") or 0)
            + int(tweet.get("quoteCount") or 0)
        )

    @staticmethod
    def _parse_twitter_date(s: str) -> datetime | None:
        """Parse a Twitter date string into an aware datetime.

        Handles the Twitter legacy format "Mon Apr 20 17:05:24 +0000 2026"
        via email.utils.parsedate_to_datetime, and falls back to fromisoformat
        for ISO-8601 shapes (e.g. "2026-02-21T12:00:00Z").
        Returns None if both parsers fail.
        """
        if not s:
            return None
        try:
            return email.utils.parsedate_to_datetime(s)
        except Exception:
            pass
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        except Exception:
            pass
        return None

    @staticmethod
    def _is_bot_account(author: dict) -> bool:
        name = (author.get("userName") or author.get("name") or "").lower()
        # Heuristic bot indicators
        if any(p in name for p in ["bot", "_bot", "autopost", "feed"]):
            return True
        followers = int(author.get("followers") or 0)
        following = int(author.get("following") or 0)
        if followers > 0 and following > 0 and following / followers > 50:
            return True
        return False

    @staticmethod
    def _deduplicate_by_content_similarity(tweets: list) -> list:
        if not tweets:
            return []
        seen_sets = []
        result = []
        for tw in tweets:
            text = (tw.get("text") or "").lower()
            words = set(text.split())
            if not words:
                continue
            is_dup = False
            for seen in seen_sets:
                if not seen:
                    continue
                overlap = len(words & seen) / max(len(words | seen), 1)
                if overlap > 0.8:
                    is_dup = True
                    break
            if not is_dup:
                seen_sets.append(words)
                result.append(tw)
        return result
