from __future__ import annotations

import asyncio
import json
import re
from typing import Optional

import httpx
import structlog
from pydantic import BaseModel, Field, ValidationError

from src.config import Settings
from src.db.sqlite import Database

log = structlog.get_logger()

MINIMAX_API_BASE = "https://api.minimaxi.chat/v1"
MAX_RETRIES = 2
REQUIRED_FIELDS = {"estimated_probability", "confidence", "reasoning"}


class PrescreenResult(BaseModel):
    estimated_probability: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(default=0.50, ge=0.0, le=1.0)
    reasoning: str = ""

    model_config = {"extra": "ignore"}

SYSTEM_PROMPT = """You are a prediction market analyst. Your job: estimate the true probability of market outcomes.

KEY PRINCIPLES:
- Markets are generally efficient. The current price IS the consensus probability. Only deviate when signals provide clear, specific evidence of mispricing.
- Extraordinary claims require extraordinary evidence. A 20%+ deviation from market price needs multiple strong, recent signals.
- HEADLINE-ONLY signals have no article body — discount them significantly.
- Older signals are less reliable. A 6-hour-old signal on a 2-hour market is nearly worthless.

CONFIDENCE SCALE (be precise):
- 0.25-0.40: Weak/stale signals, headline-only, single low-credibility source
- 0.40-0.60: Moderate evidence from 1-2 credible sources, but ambiguous
- 0.60-0.80: Strong evidence from multiple credible sources, clear direction
- 0.80-0.95: Very strong multi-source confirmation, breaking/official news
- >0.95: Reserved for officially confirmed outcomes only

OUTPUT FORMAT: Return ONLY valid JSON, no markdown or extra text."""


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
    """Validate and coerce LLM response fields.

    estimated_probability: required, must be coercible to float in [0, 1].
    confidence: optional, defaults to 0.50 if missing; must be in [0, 1] if present.
    reasoning: optional, defaults to "" if missing.
    """
    # --- estimated_probability: strict ---
    if "estimated_probability" not in data:
        log.warning("llm_missing_fields", missing=["estimated_probability"])
        return None
    try:
        prob = float(data["estimated_probability"])
    except (ValueError, TypeError):
        log.warning("llm_invalid_types", field="estimated_probability")
        return None
    if not (0 <= prob <= 1):
        log.warning("llm_out_of_range", field="estimated_probability", value=prob)
        return None
    data["estimated_probability"] = prob

    # --- confidence: optional, default 0.50 ---
    if "confidence" not in data:
        log.info("llm_default_confidence", market="unknown")
        data["confidence"] = 0.50
    else:
        try:
            conf = float(data["confidence"])
        except (ValueError, TypeError):
            log.warning("llm_invalid_types", field="confidence")
            return None
        if not (0 <= conf <= 1):
            log.warning("llm_out_of_range", field="confidence", value=conf)
            return None
        data["confidence"] = conf

    # --- reasoning: optional, default "" ---
    if "reasoning" not in data:
        log.info("llm_default_reasoning", market="unknown")
        data["reasoning"] = ""

    return data


class LLMClient:
    def __init__(self, settings: Settings, db: Database):
        self._api_key = settings.MINIMAX_API_KEY
        self._db = db
        self._model = settings.LLM_MODEL
        self._settings = settings
        self._timeout = httpx.Timeout(30.0, connect=10.0)

    async def complete(
        self,
        prompt: str,
        max_tokens: int = 2000,
        system_prompt: str = SYSTEM_PROMPT,
        response_format: Optional[dict] = None,
    ) -> str:
        """Raw API call to MiniMax."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.1,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{MINIMAX_API_BASE}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
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

    async def call_prescreen(self, context: str, market_id: str) -> Optional[dict]:
        """Cheap pre-screen LLM call. 1 retry, settings.PRESCREEN_MAX_TOKENS budget, fail-open."""
        for attempt in range(2):
            try:
                raw = await self.complete(
                    context,
                    max_tokens=self._settings.PRESCREEN_MAX_TOKENS,
                    response_format={"type": "json_object"},
                )
                try:
                    return PrescreenResult.model_validate_json(raw).model_dump()
                except ValidationError:
                    parsed = parse_json_safe(raw)
                    if parsed is not None:
                        try:
                            return PrescreenResult.model_validate(parsed).model_dump()
                        except ValidationError as ve:
                            log.warning(
                                "prescreen_parse_failed",
                                attempt=attempt,
                                market_id=market_id,
                                errors=str(ve)[:200],
                                raw_preview=raw[:300],
                            )
                    else:
                        log.warning(
                            "prescreen_parse_failed",
                            attempt=attempt,
                            market_id=market_id,
                            errors="no_json_recovered",
                            raw_preview=raw[:300],
                        )
                    if attempt == 0:
                        await asyncio.sleep(1.0)
                    continue
            except Exception as e:
                log.warning("prescreen_error", error=str(e), attempt=attempt, market_id=market_id)
                if attempt == 0:
                    await asyncio.sleep(1.0)

        log.info("prescreen_failed_passthrough", market_id=market_id)
        return None


# Backward-compat alias
GrokClient = LLMClient
