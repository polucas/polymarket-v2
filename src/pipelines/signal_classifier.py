"""Signal source-tier classifier.

Classifies incoming signals into tiers S1-S6 based on the source metadata
and the known-sources registry at ``config/known_sources.yaml``.

Tier definitions
----------------
* **S1** -- Official government / institutional primary sources
* **S2** -- Wire services (Reuters, AP, AFP ...)
* **S3** -- Institutional media (BBC, CNN, NYT, WSJ ...)
* **S4** -- Verified domain experts (verified + 50 k followers + expert bio)
* **S5** -- Market data feeds
* **S6** -- Unverified / low-credibility / unknown
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Set

import structlog
import yaml

# Re-export the credibility mapping so callers can do:
#   from src.pipelines.signal_classifier import classify_source_tier, SOURCE_TIER_CREDIBILITY
from src.models import SOURCE_TIER_CREDIBILITY  # noqa: F401

log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Load known_sources.yaml once at module level
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "known_sources.yaml"


def _load_known_sources() -> Dict[str, Any]:
    """Read and parse the known-sources YAML config."""
    try:
        with open(_CONFIG_PATH, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
            if not isinstance(data, dict):
                log.error("known_sources_invalid_format", path=str(_CONFIG_PATH))
                return {}
            return data
    except FileNotFoundError:
        log.error("known_sources_not_found", path=str(_CONFIG_PATH))
        return {}
    except yaml.YAMLError as exc:
        log.error("known_sources_parse_error", path=str(_CONFIG_PATH), error=str(exc))
        return {}


_KNOWN: Dict[str, Any] = _load_known_sources()

# Pre-build lookup sets for O(1) membership tests.
# Twitter handles are normalised to lower-case for case-insensitive matching.

_OFFICIAL_TWITTER: Set[str] = {
    h.lower() for h in _KNOWN.get("official_sources", {}).get("twitter", [])
}
_WIRE_TWITTER: Set[str] = {
    h.lower() for h in _KNOWN.get("wire_services", {}).get("twitter", [])
}
_INSTITUTIONAL_TWITTER: Set[str] = {
    h.lower() for h in _KNOWN.get("institutional_media", {}).get("twitter", [])
}

_OFFICIAL_RSS: Set[str] = {
    d.lower() for d in _KNOWN.get("official_sources", {}).get("rss_domains", [])
}
_WIRE_RSS: Set[str] = {
    d.lower() for d in _KNOWN.get("wire_services", {}).get("rss_domains", [])
}
_INSTITUTIONAL_RSS: Set[str] = {
    d.lower() for d in _KNOWN.get("institutional_media", {}).get("rss_domains", [])
}

_EXPERT_BIO_KEYWORDS: Set[str] = {
    kw.lower() for kw in _KNOWN.get("expert_bio_keywords", [])
}

# Minimum follower threshold for S4 (verified expert) classification.
_S4_MIN_FOLLOWERS: int = 50_000


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_source_tier(signal: dict) -> str:
    """Classify a signal dict into a source tier (S1-S6).

    Parameters
    ----------
    signal : dict
        Expected keys (not all are required for every source type):

        * ``source_type`` -- ``"twitter"`` | ``"rss"`` | ``"market_data"``
        * ``domain`` -- RSS feed domain (e.g. ``"reuters.com"``)
        * ``account_handle`` -- Twitter handle including ``@`` prefix
        * ``is_verified`` -- bool, Twitter verification status
        * ``follower_count`` -- int, number of followers
        * ``bio`` -- str, user biography text

    Returns
    -------
    str
        One of ``"S1"``, ``"S2"``, ``"S3"``, ``"S4"``, ``"S5"``, ``"S6"``.
    """
    source_type = (signal.get("source_type") or "").lower().strip()

    # ---- Market data is always S5 ----
    if source_type == "market_data":
        return "S5"

    # ---- RSS classification ----
    if source_type == "rss":
        return _classify_rss(signal)

    # ---- Twitter classification ----
    if source_type == "twitter":
        return _classify_twitter(signal)

    # Unknown source type falls to lowest tier.
    log.warning("unknown_source_type", source_type=source_type)
    return "S6"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalise_domain(raw: str) -> str:
    """Strip protocol, ``www.`` prefix, and trailing slashes from a domain."""
    domain = raw.lower().strip()
    for prefix in ("https://", "http://", "www."):
        if domain.startswith(prefix):
            domain = domain[len(prefix):]
    return domain.rstrip("/")


def _classify_rss(signal: dict) -> str:
    domain = _normalise_domain(signal.get("domain") or "")
    if not domain:
        return "S6"

    if domain in _OFFICIAL_RSS:
        return "S1"
    if domain in _WIRE_RSS:
        return "S2"
    if domain in _INSTITUTIONAL_RSS:
        return "S3"

    # Also check if the domain *ends with* a known domain (e.g. "feeds.reuters.com"
    # should still match "reuters.com").
    for known in _OFFICIAL_RSS:
        if domain.endswith("." + known):
            return "S1"
    for known in _WIRE_RSS:
        if domain.endswith("." + known):
            return "S2"
    for known in _INSTITUTIONAL_RSS:
        if domain.endswith("." + known):
            return "S3"

    return "S6"


def _classify_twitter(signal: dict) -> str:
    handle = (signal.get("account_handle") or "").lower().strip()

    # Ensure the handle starts with '@' for consistent matching.
    if handle and not handle.startswith("@"):
        handle = f"@{handle}"

    if handle in _OFFICIAL_TWITTER:
        return "S1"
    if handle in _WIRE_TWITTER:
        return "S2"
    if handle in _INSTITUTIONAL_TWITTER:
        return "S3"

    # S4: verified account + >= 50 000 followers + expert bio keyword
    is_verified = bool(signal.get("is_verified"))
    follower_count = int(signal.get("follower_count") or 0)
    bio = (signal.get("bio") or "").lower()

    if (
        is_verified
        and follower_count >= _S4_MIN_FOLLOWERS
        and _bio_contains_expert_keyword(bio)
    ):
        return "S4"

    return "S6"


def _bio_contains_expert_keyword(bio: str) -> bool:
    """Return ``True`` if *bio* contains at least one expert keyword."""
    if not bio:
        return False
    # Split on common delimiters so "journalist/editor" is detected.
    bio_tokens = set(bio.replace("/", " ").replace("|", " ").replace(",", " ").split())
    return bool(bio_tokens & _EXPERT_BIO_KEYWORDS)
