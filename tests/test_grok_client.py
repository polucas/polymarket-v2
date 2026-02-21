"""Tests for GrokClient JSON parsing, validation, and retry logic."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.engine.grok_client import (
    GrokClient,
    parse_json_safe,
    _validate_grok_response,
    REQUIRED_FIELDS,
    MAX_RETRIES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _valid_grok_dict(**overrides) -> dict:
    """Return a minimal valid Grok response dict."""
    base = {
        "estimated_probability": 0.72,
        "confidence": 0.85,
        "reasoning": "Strong bullish signal from official sources.",
        "signal_info_types": [
            {"source_tier": "S1", "info_type": "I1", "content_summary": "Official announcement."},
        ],
    }
    base.update(overrides)
    return base


def _make_xai_response(content: str, prompt_tokens: int = 100, completion_tokens: int = 50) -> dict:
    """Build a fake xAI API response body."""
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    }


def _mock_settings():
    s = MagicMock()
    s.XAI_API_KEY = "test-xai-key"
    return s


def _mock_db():
    db = AsyncMock()
    db.increment_api_cost = AsyncMock()
    db.record_parse_failure = AsyncMock()
    return db


# ---------------------------------------------------------------------------
# parse_json_safe
# ---------------------------------------------------------------------------


class TestParseJsonSafe:
    def test_valid_json_parsed(self):
        raw = json.dumps({"key": "value", "num": 42})
        result = parse_json_safe(raw)
        assert result == {"key": "value", "num": 42}

    def test_json_in_json_fences(self):
        raw = '```json\n{"key": "value"}\n```'
        result = parse_json_safe(raw)
        assert result == {"key": "value"}

    def test_json_in_plain_fences(self):
        raw = '```\n{"key": "value"}\n```'
        result = parse_json_safe(raw)
        assert result == {"key": "value"}

    def test_preamble_text_with_json(self):
        raw = 'Here is my analysis:\n{"estimated_probability": 0.75, "confidence": 0.80, "reasoning": "test", "signal_info_types": []}'
        result = parse_json_safe(raw)
        assert result is not None
        assert result["estimated_probability"] == 0.75

    def test_invalid_string_returns_none(self):
        result = parse_json_safe("This is not JSON at all.")
        assert result is None

    def test_empty_string_returns_none(self):
        result = parse_json_safe("")
        assert result is None

    def test_json_with_postamble_text(self):
        """#5 — JSON followed by conversational postamble -> parsed dict."""
        raw = '{"estimated_probability": 0.60, "confidence": 0.80, "reasoning": "ok", "signal_info_types": []} Hope this helps!'
        result = parse_json_safe(raw)
        assert result is not None
        assert result["estimated_probability"] == 0.60

    def test_nested_braces_in_json_values(self):
        """#7 — Nested braces inside JSON string values -> correctly parsed."""
        raw = json.dumps({
            "estimated_probability": 0.50,
            "confidence": 0.70,
            "reasoning": "The {market} showed {volatility}",
            "signal_info_types": [
                {"source_tier": "S1", "info_type": "I1", "content_summary": "Event {confirmed}"},
            ],
        })
        result = parse_json_safe(raw)
        assert result is not None
        assert "{market}" in result["reasoning"]
        assert result["signal_info_types"][0]["content_summary"] == "Event {confirmed}"

    def test_only_whitespace_returns_none(self):
        """#9 — Only whitespace -> None."""
        result = parse_json_safe("   \n\t  \n  ")
        assert result is None

    def test_json_with_trailing_comma(self):
        """#10 — Trailing comma (common LLM error) -> attempt parse, may return None."""
        # Standard json.loads rejects trailing commas; verify graceful handling
        raw = '{"estimated_probability": 0.5, "confidence": 0.7, "reasoning": "test", "signal_info_types": [],}'
        result = parse_json_safe(raw)
        # Python stdlib json rejects trailing commas, so expect None
        assert result is None


# ---------------------------------------------------------------------------
# _validate_grok_response  --  field validation
# ---------------------------------------------------------------------------


class TestValidateGrokResponse:
    def test_missing_required_fields_returns_none(self):
        incomplete = {"estimated_probability": 0.5, "confidence": 0.5}
        # Missing: reasoning, signal_info_types
        result = _validate_grok_response(incomplete)
        assert result is None

    def test_probability_above_1_returns_none(self):
        data = _valid_grok_dict(estimated_probability=1.5)
        result = _validate_grok_response(data)
        assert result is None

    def test_probability_as_string_coerced_to_float(self):
        data = _valid_grok_dict(estimated_probability="0.75", confidence="0.80")
        result = _validate_grok_response(data)
        assert result is not None
        assert isinstance(result["estimated_probability"], float)
        assert result["estimated_probability"] == pytest.approx(0.75)
        assert isinstance(result["confidence"], float)
        assert result["confidence"] == pytest.approx(0.80)

    def test_valid_response_passes(self):
        data = _valid_grok_dict()
        result = _validate_grok_response(data)
        assert result is not None
        assert result["estimated_probability"] == pytest.approx(0.72)

    def test_negative_probability_returns_none(self):
        data = _valid_grok_dict(estimated_probability=-0.1)
        result = _validate_grok_response(data)
        assert result is None

    def test_confidence_above_1_returns_none(self):
        data = _valid_grok_dict(confidence=1.01)
        result = _validate_grok_response(data)
        assert result is None

    def test_missing_only_estimated_probability(self):
        """#11 — Response missing only 'estimated_probability' -> validation fails."""
        data = _valid_grok_dict()
        del data["estimated_probability"]
        result = _validate_grok_response(data)
        assert result is None

    def test_missing_only_confidence(self):
        """#12 — Response missing only 'confidence' -> validation fails."""
        data = _valid_grok_dict()
        del data["confidence"]
        result = _validate_grok_response(data)
        assert result is None

    def test_missing_only_reasoning(self):
        """#13 — Response missing only 'reasoning' -> validation fails."""
        data = _valid_grok_dict()
        del data["reasoning"]
        result = _validate_grok_response(data)
        assert result is None

    def test_missing_only_signal_info_types(self):
        """#14 — Response missing only 'signal_info_types' -> validation fails."""
        data = _valid_grok_dict()
        del data["signal_info_types"]
        result = _validate_grok_response(data)
        assert result is None

    def test_confidence_negative_returns_none(self):
        """#16 — confidence = -0.1 (below 0.0) -> validation fails."""
        data = _valid_grok_dict(confidence=-0.1)
        result = _validate_grok_response(data)
        assert result is None


# ---------------------------------------------------------------------------
# call_grok_with_retry  --  retry logic
# ---------------------------------------------------------------------------


class TestCallGrokWithRetry:
    @pytest.mark.asyncio
    async def test_success_on_first_attempt(self):
        """Successful first attempt returns validated dict."""
        db = _mock_db()
        settings = _mock_settings()
        grok = GrokClient(settings, db)

        valid_json = json.dumps(_valid_grok_dict())
        api_resp = _make_xai_response(valid_json)

        mock_http_resp = MagicMock(spec=httpx.Response)
        mock_http_resp.status_code = 200
        mock_http_resp.json.return_value = api_resp
        mock_http_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_http_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("src.engine.grok_client.httpx.AsyncClient", return_value=mock_client):
            result = await grok.call_grok_with_retry("test context", "market-1")

        assert result is not None
        assert result["estimated_probability"] == pytest.approx(0.72)
        db.increment_api_cost.assert_awaited_once()
        db.record_parse_failure.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_failure_then_success_returns_second_attempt(self):
        """First attempt fails parsing, second succeeds."""
        db = _mock_db()
        settings = _mock_settings()
        grok = GrokClient(settings, db)

        # First call: invalid JSON content. Second call: valid.
        bad_content = "I can't answer that"
        good_content = json.dumps(_valid_grok_dict(estimated_probability=0.55))
        api_resp_bad = _make_xai_response(bad_content)
        api_resp_good = _make_xai_response(good_content)

        mock_resp_bad = MagicMock(spec=httpx.Response)
        mock_resp_bad.status_code = 200
        mock_resp_bad.json.return_value = api_resp_bad
        mock_resp_bad.raise_for_status = MagicMock()

        mock_resp_good = MagicMock(spec=httpx.Response)
        mock_resp_good.status_code = 200
        mock_resp_good.json.return_value = api_resp_good
        mock_resp_good.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=[mock_resp_bad, mock_resp_good])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("src.engine.grok_client.httpx.AsyncClient", return_value=mock_client), \
             patch("src.engine.grok_client.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await grok.call_grok_with_retry("ctx", "market-2")

        assert result is not None
        assert result["estimated_probability"] == pytest.approx(0.55)
        mock_sleep.assert_awaited()
        db.record_parse_failure.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_all_attempts_fail_returns_none(self):
        """All retry attempts fail -> returns None and records failure."""
        db = _mock_db()
        settings = _mock_settings()
        grok = GrokClient(settings, db)

        bad_content = "Sorry, can't process that."
        api_resp_bad = _make_xai_response(bad_content)

        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = api_resp_bad
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        # MAX_RETRIES + 1 = 3 total attempts
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("src.engine.grok_client.httpx.AsyncClient", return_value=mock_client), \
             patch("src.engine.grok_client.asyncio.sleep", new_callable=AsyncMock):
            result = await grok.call_grok_with_retry("ctx", "market-3")

        assert result is None
        db.record_parse_failure.assert_awaited_once_with("market-3")

    @pytest.mark.asyncio
    async def test_success_increments_api_cost(self):
        """On successful API call, db.increment_api_cost is called."""
        db = _mock_db()
        settings = _mock_settings()
        grok = GrokClient(settings, db)

        valid_json = json.dumps(_valid_grok_dict())
        api_resp = _make_xai_response(valid_json, prompt_tokens=200, completion_tokens=80)

        mock_http_resp = MagicMock(spec=httpx.Response)
        mock_http_resp.status_code = 200
        mock_http_resp.json.return_value = api_resp
        mock_http_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_http_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("src.engine.grok_client.httpx.AsyncClient", return_value=mock_client):
            await grok.call_grok_with_retry("ctx", "market-4")

        db.increment_api_cost.assert_awaited_once_with(
            "grok", tokens_in=200, tokens_out=80,
        )

    @pytest.mark.asyncio
    async def test_total_failure_records_parse_failure(self):
        """When all retries exhausted, db.record_parse_failure is called."""
        db = _mock_db()
        settings = _mock_settings()
        grok = GrokClient(settings, db)

        # Simulate HTTP errors on every attempt
        mock_client = AsyncMock()
        mock_http_resp = MagicMock(spec=httpx.Response)
        mock_http_resp.status_code = 500
        mock_http_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            message="500 Internal Server Error",
            request=MagicMock(),
            response=mock_http_resp,
        )
        mock_client.post = AsyncMock(return_value=mock_http_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("src.engine.grok_client.httpx.AsyncClient", return_value=mock_client), \
             patch("src.engine.grok_client.asyncio.sleep", new_callable=AsyncMock):
            result = await grok.call_grok_with_retry("ctx", "market-fail")

        assert result is None
        db.record_parse_failure.assert_awaited_once_with("market-fail")

    @pytest.mark.asyncio
    async def test_parse_failure_first_two_success_on_third(self):
        """#21 — Parse failure on 1st and 2nd, success on 3rd -> returns dict from 3rd."""
        db = _mock_db()
        settings = _mock_settings()
        grok = GrokClient(settings, db)

        bad_content_1 = "Not JSON at all"
        bad_content_2 = "Still not valid"
        good_content = json.dumps(_valid_grok_dict(estimated_probability=0.33))

        responses = []
        for content in [bad_content_1, bad_content_2, good_content]:
            api_resp = _make_xai_response(content)
            mock_resp = MagicMock(spec=httpx.Response)
            mock_resp.status_code = 200
            mock_resp.json.return_value = api_resp
            mock_resp.raise_for_status = MagicMock()
            responses.append(mock_resp)

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=responses)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("src.engine.grok_client.httpx.AsyncClient", return_value=mock_client), \
             patch("src.engine.grok_client.asyncio.sleep", new_callable=AsyncMock):
            result = await grok.call_grok_with_retry("ctx", "market-21")

        assert result is not None
        assert result["estimated_probability"] == pytest.approx(0.33)
        assert mock_client.post.await_count == 3
        db.record_parse_failure.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_api_error_first_success_on_second(self):
        """#23 — API error on 1st attempt, success on 2nd -> returns dict (error recovery)."""
        db = _mock_db()
        settings = _mock_settings()
        grok = GrokClient(settings, db)

        # First response: HTTP 500 error
        mock_resp_err = MagicMock(spec=httpx.Response)
        mock_resp_err.status_code = 500
        mock_resp_err.raise_for_status.side_effect = httpx.HTTPStatusError(
            message="500 Internal Server Error",
            request=MagicMock(),
            response=mock_resp_err,
        )

        # Second response: valid
        good_content = json.dumps(_valid_grok_dict(estimated_probability=0.88))
        api_resp_good = _make_xai_response(good_content)
        mock_resp_good = MagicMock(spec=httpx.Response)
        mock_resp_good.status_code = 200
        mock_resp_good.json.return_value = api_resp_good
        mock_resp_good.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=[mock_resp_err, mock_resp_good])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("src.engine.grok_client.httpx.AsyncClient", return_value=mock_client), \
             patch("src.engine.grok_client.asyncio.sleep", new_callable=AsyncMock):
            result = await grok.call_grok_with_retry("ctx", "market-23")

        assert result is not None
        assert result["estimated_probability"] == pytest.approx(0.88)
        db.record_parse_failure.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_linear_backoff_sleep_durations(self):
        """#24 — Linear backoff: asyncio.sleep called with 1.0 then 2.0 seconds."""
        db = _mock_db()
        settings = _mock_settings()
        grok = GrokClient(settings, db)

        bad_content = "not json"
        api_resp_bad = _make_xai_response(bad_content)

        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = api_resp_bad
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("src.engine.grok_client.httpx.AsyncClient", return_value=mock_client), \
             patch("src.engine.grok_client.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await grok.call_grok_with_retry("ctx", "market-24")

        assert result is None
        # With MAX_RETRIES=2, total 3 attempts:
        #   attempt 0 fails -> sleep(1.0 * (0+1)) = sleep(1.0)
        #   attempt 1 fails -> sleep(1.0 * (1+1)) = sleep(2.0)
        #   attempt 2 fails -> no sleep (last attempt)
        assert mock_sleep.await_count == 2
        calls = [c.args[0] for c in mock_sleep.await_args_list]
        assert calls == [1.0, 2.0]
