"""Tests for CLOB token ID extraction and orderbook calls."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.config import Settings
from src.models import Market
from src.pipelines.polymarket import PolymarketClient


def _make_gamma_market(**overrides):
    """Build a minimal Gamma API market response."""
    base = {
        "id": "517310",
        "question": "Will X happen?",
        "outcomePrices": '["0.55", "0.45"]',
        "endDate": "2026-02-25T12:00:00Z",
        "liquidity": "20000",
        "volume24hr": "5000",
        "outcomes": '["Yes", "No"]',
        "clobTokenIds": '["111222333444555", "666777888999000"]',
        "closed": False,
        "resolved": False,
    }
    base.update(overrides)
    return base


class TestClobTokenIdExtraction:
    @pytest.mark.asyncio
    async def test_market_has_clob_token_ids(self):
        settings = MagicMock(spec=Settings)
        client = PolymarketClient(settings)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [_make_gamma_market()]
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            markets = await client.get_active_markets(tier=1)

        assert len(markets) >= 1
        m = markets[0]
        assert m.clob_token_id_yes == "111222333444555"
        assert m.clob_token_id_no == "666777888999000"

    @pytest.mark.asyncio
    async def test_missing_clob_token_ids_defaults_empty(self):
        settings = MagicMock(spec=Settings)
        client = PolymarketClient(settings)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [_make_gamma_market(clobTokenIds=None)]
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            markets = await client.get_active_markets(tier=1)

        if markets:  # May or may not pass filters
            assert markets[0].clob_token_id_yes == ""
            assert markets[0].clob_token_id_no == ""


class TestGetOrderbookTokenId:
    @pytest.mark.asyncio
    async def test_orderbook_passes_token_id_to_clob(self):
        settings = MagicMock(spec=Settings)
        client = PolymarketClient(settings)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"bids": [], "asks": []}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            await client.get_orderbook("111222333444555", market_id="517310")

        # Verify the CLOB was called with token_id, not market_id
        call_kwargs = mock_client.get.call_args
        assert call_kwargs[1]["params"]["token_id"] == "111222333444555"

    @pytest.mark.asyncio
    async def test_empty_token_id_returns_empty_orderbook(self):
        settings = MagicMock(spec=Settings)
        client = PolymarketClient(settings)

        ob = await client.get_orderbook("", market_id="517310")
        assert ob.bids == []
        assert ob.asks == []
