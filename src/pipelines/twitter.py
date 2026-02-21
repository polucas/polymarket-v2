from __future__ import annotations

from datetime import datetime, timezone
from typing import List

import httpx
import structlog

from src.config import Settings
from src.models import Signal
from src.pipelines.signal_classifier import classify_source_tier, SOURCE_TIER_CREDIBILITY

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
        
        # Pre-filter
        filtered = [
            t for t in raw_tweets
            if t.get("author", {}).get("followers_count", 0) >= 1000
            and t.get("engagement_score", 0) >= 10
            and not self._is_bot_account(t.get("author", {}))
        ]
        
        deduplicated = self._deduplicate_by_content_similarity(filtered)
        
        # Classify and score
        scored = []
        for tweet in deduplicated:
            author = tweet.get("author", {})
            source_tier = classify_source_tier({
                "source_type": "twitter",
                "account_handle": f"@{author.get("screen_name", "")}",
                "is_verified": author.get("verified", False),
                "follower_count": author.get("followers_count", 0),
                "bio": author.get("bio", "") or author.get("description", "") or "",
            })
            credibility = SOURCE_TIER_CREDIBILITY.get(source_tier, 0.30)
            
            created_at = None
            if tweet.get("created_at"):
                try:
                    created_at = datetime.fromisoformat(tweet["created_at"].replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    created_at = datetime.now(timezone.utc)
            
            scored.append((credibility, source_tier, tweet, created_at))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        
        signals = []
        for cred, st, tw, ts in scored[:10]:
            author = tw.get("author", {})
            signals.append(Signal(
                source="twitter",
                source_tier=st,
                info_type=None,  # Assigned later by Grok
                content=(tw.get("text", "") or "")[:280],
                credibility=cred,
                author=author.get("screen_name", ""),
                followers=author.get("followers_count", 0),
                engagement=tw.get("engagement_score", 0),
                timestamp=ts,
                headline_only=False,
            ))
        return signals

    async def _search_tweets(self, query: str, max_results: int = 50) -> list:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(
                    f"{BASE_URL}/tweet/advanced_search",
                    headers={"X-API-Key": self._api_key},
                    params={
                        "query": query,
                        "queryType": "Latest",
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
    def _is_bot_account(author: dict) -> bool:
        name = (author.get("screen_name") or author.get("name") or "").lower()
        # Heuristic bot indicators
        if any(p in name for p in ["bot", "_bot", "autopost", "feed"]):
            return True
        followers = author.get("followers_count", 0)
        following = author.get("following_count", 0) or author.get("friends_count", 0)
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
