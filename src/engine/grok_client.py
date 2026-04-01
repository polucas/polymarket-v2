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

MINIMAX_API_BASE = "https://api.minimaxi.chat/v1"
MAX_RETRIES = 2
REQUIRED_FIELDS = {"estimated_probability", "confidence", "reasoning"}


def parse_json_safe(raw: str) -> Optional[dict]:
    """Parse JSON with multiple fallback strategies."""
    text = raw.strip()
    # Strip <think>...</think> blocks (reasoning models like MiniMax-M2.7)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
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
    # Find last {...} block (reasoning models may have JSON at end)
    matches = list(re.finditer(r'\{[^{}]*\}', text, re.DOTALL))
    for m in reversed(matches):
        try:
            return json.loads(m.group())
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def _validate_llm_response(data: dict) -> Optional[dict]:
    """Validate and coerce LLM response fields."""
    missing = REQUIRED_FIELDS - set(data.keys())
    if missing:
        log.warning("llm_missing_fields", missing=list(missing))
        return None

    # Type coercion
    try:
        prob = float(data["estimated_probability"])
        conf = float(data["confidence"])
    except (ValueError, TypeError):
        log.warning("llm_invalid_types")
        return None

    if not (0 <= prob <= 1) or not (0 <= conf <= 1):
        log.warning("llm_out_of_range", prob=prob, conf=conf)
        return None

    data["estimated_probability"] = prob
    data["confidence"] = conf
    return data


class LLMClient:
    def __init__(self, settings: Settings, db: Database):
        self._api_key = settings.MINIMAX_API_KEY
        self._db = db
        self._model = settings.LLM_MODEL
        self._timeout = httpx.Timeout(30.0, connect=10.0)

    async def complete(self, prompt: str, max_tokens: int = 2000) -> str:
        """Raw API call to MiniMax."""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{MINIMAX_API_BASE}/chat/completions",
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
                "minimax",
                tokens_in=usage.get("prompt_tokens", 0),
                tokens_out=usage.get("completion_tokens", 0),
            )
            return content

    async def call_grok_with_retry(self, context: str, market_id: str) -> Optional[dict]:
        """Call MiniMax with retry pipeline. MAX_RETRIES=2 (total 3 attempts)."""
        for attempt in range(MAX_RETRIES + 1):
            try:
                raw = await self.complete(context)
                parsed = parse_json_safe(raw)
                if parsed is None:
                    log.warning("llm_parse_failed", attempt=attempt, market_id=market_id)
                    if attempt < MAX_RETRIES:
                        await asyncio.sleep(1.0 * (attempt + 1))
                    continue

                validated = _validate_llm_response(parsed)
                if validated is None:
                    log.warning("llm_validation_failed", attempt=attempt, market_id=market_id)
                    if attempt < MAX_RETRIES:
                        await asyncio.sleep(1.0 * (attempt + 1))
                    continue

                return validated

            except httpx.HTTPStatusError as e:
                log.warning("llm_http_error", status=e.response.status_code, attempt=attempt)
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(1.0 * (attempt + 1))
            except Exception as e:
                log.error("llm_error", error=str(e), attempt=attempt)
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(1.0 * (attempt + 1))

        # All retries exhausted
        log.error("llm_all_retries_failed", market_id=market_id)
        await self._db.record_parse_failure(market_id)
        return None


# Backward-compat alias
GrokClient = LLMClient
