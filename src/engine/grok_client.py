from __future__ import annotations

import asyncio
import json
import re
from typing import Optional

import httpx
import structlog

from src.config import Settings
from src.db.sqlite import Database

log = structlog.get_logger()

XAI_API_BASE = "https://api.x.ai/v1"
MAX_RETRIES = 2
REQUIRED_FIELDS = {"estimated_probability", "confidence", "reasoning", "signal_info_types"}


def parse_json_safe(raw: str) -> Optional[dict]:
    """Parse JSON with multiple fallback strategies."""
    text = raw.strip()
    # Direct parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    # Strip markdown fences
    fenced = re.sub(r'^```(?:json)?\s*\n?', '', text, flags=re.MULTILINE)
    fenced = re.sub(r'\n?```\s*$', '', fenced, flags=re.MULTILINE)
    try:
        return json.loads(fenced.strip())
    except (json.JSONDecodeError, ValueError):
        pass
    # Find first {...} block
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def _validate_grok_response(data: dict) -> Optional[dict]:
    """Validate and coerce Grok response fields."""
    missing = REQUIRED_FIELDS - set(data.keys())
    if missing:
        log.warning("grok_missing_fields", missing=list(missing))
        return None

    # Type coercion
    try:
        prob = float(data["estimated_probability"])
        conf = float(data["confidence"])
    except (ValueError, TypeError):
        log.warning("grok_invalid_types")
        return None

    if not (0 <= prob <= 1) or not (0 <= conf <= 1):
        log.warning("grok_out_of_range", prob=prob, conf=conf)
        return None

    data["estimated_probability"] = prob
    data["confidence"] = conf
    return data


class GrokClient:
    def __init__(self, settings: Settings, db: Database):
        self._api_key = settings.XAI_API_KEY
        self._db = db
        self._model = "grok-4-1-fast-reasoning"
        self._timeout = httpx.Timeout(30.0, connect=10.0)

    async def complete(self, prompt: str, max_tokens: int = 500) -> str:
        """Raw API call to xAI."""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{XAI_API_BASE}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.1,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            # Track API cost
            usage = data.get("usage", {})
            await self._db.increment_api_cost(
                "grok",
                tokens_in=usage.get("prompt_tokens", 0),
                tokens_out=usage.get("completion_tokens", 0),
            )
            return content

    async def call_grok_with_retry(self, context: str, market_id: str) -> Optional[dict]:
        """Call Grok with retry pipeline. MAX_RETRIES=2 (total 3 attempts)."""
        for attempt in range(MAX_RETRIES + 1):
            try:
                raw = await self.complete(context)
                parsed = parse_json_safe(raw)
                if parsed is None:
                    log.warning("grok_parse_failed", attempt=attempt, market_id=market_id)
                    if attempt < MAX_RETRIES:
                        await asyncio.sleep(1.0 * (attempt + 1))
                    continue

                validated = _validate_grok_response(parsed)
                if validated is None:
                    log.warning("grok_validation_failed", attempt=attempt, market_id=market_id)
                    if attempt < MAX_RETRIES:
                        await asyncio.sleep(1.0 * (attempt + 1))
                    continue

                return validated

            except httpx.HTTPStatusError as e:
                log.warning("grok_http_error", status=e.response.status_code, attempt=attempt)
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(1.0 * (attempt + 1))
            except Exception as e:
                log.error("grok_error", error=str(e), attempt=attempt)
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(1.0 * (attempt + 1))

        # All retries exhausted
        log.error("grok_all_retries_failed", market_id=market_id)
        await self._db.record_parse_failure(market_id)
        return None
