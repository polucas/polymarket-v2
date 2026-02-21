from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, List

import feedparser
import structlog
import yaml
from pathlib import Path

from src.models import Signal
from src.pipelines.signal_classifier import classify_source_tier, SOURCE_TIER_CREDIBILITY

log = structlog.get_logger()


def _load_feed_config() -> dict:
    config_path = Path(__file__).resolve().parents[2] / "config" / "rss_feeds.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f).get("feeds", {})


def _parse_date(date_str: str) -> datetime | None:
    if not date_str:
        return None
    try:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        pass
    # feedparser returns time_struct
    import time
    try:
        parsed = feedparser._parse_date(date_str)
        if parsed:
            return datetime(*parsed[:6], tzinfo=timezone.utc)
    except Exception:
        pass
    # Try email.utils for RFC 2822 dates
    from email.utils import parsedate_to_datetime
    try:
        return parsedate_to_datetime(date_str).replace(tzinfo=timezone.utc)
    except Exception:
        return None


class RSSPipeline:
    def __init__(self):
        self.seen_headlines: Dict[str, datetime] = {}
        self._feeds = _load_feed_config()

    def _prune_old_headlines(self) -> None:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        self.seen_headlines = {
            h: ts for h, ts in self.seen_headlines.items() if ts > cutoff
        }

    async def get_breaking_news(self) -> List[Signal]:
        self._prune_old_headlines()
        signals = []
        now = datetime.now(timezone.utc)

        for feed_name, cfg in self._feeds.items():
            try:
                feed = feedparser.parse(cfg["url"])
                for entry in feed.entries[:10]:
                    headline = entry.title.strip()
                    if headline in self.seen_headlines:
                        continue
                    self.seen_headlines[headline] = now

                    published = _parse_date(entry.get("published", ""))
                    if published and (now - published).total_seconds() > 7200:
                        continue  # Older than 2 hours

                    source_tier = classify_source_tier({
                        "source_type": "rss",
                        "domain": cfg["domain"],
                    })

                    signals.append(Signal(
                        source="rss",
                        source_tier=source_tier,
                        info_type=None,
                        content=headline,
                        credibility=SOURCE_TIER_CREDIBILITY.get(source_tier, 0.30),
                        author=feed_name,
                        followers=0,
                        engagement=0,
                        timestamp=published or now,
                        headline_only=True,
                    ))
            except Exception as e:
                log.warning("rss_feed_error", feed=feed_name, error=str(e))
                continue

        return signals
