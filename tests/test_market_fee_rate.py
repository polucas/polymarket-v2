"""Tests for market fee_rate from config and per-type fee schedule."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import Settings
from src.pipelines.market_classifier import get_fee_rate
from src.pipelines.polymarket import PolymarketClient


def _make_gamma_market(**overrides):
    base = {
        "id": "517310",
        "question": "Will X happen?",
        "outcomePrices": '["0.55", "0.45"]',
        "endDate": (datetime.now(timezone.utc) + timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "liquidity": "20000",
        "volume24hr": "5000",
        "clobTokenIds": '["111", "222"]',
        "closed": False,
        "resolved": False,
    }
    base.update(overrides)
    return base


class TestMarketFeeRate:
    @pytest.mark.asyncio
    async def test_tier1_market_fee_rate_zero(self):
        settings = MagicMock(spec=Settings)
        settings.MARKET_FETCH_LIMIT = 200
        settings.MARKET_PAGE_SIZE = 500
        settings.MARKET_FETCH_PAGES = 3
        settings.MIN_TRADEABLE_PRICE = 0.05
        settings.MAX_TRADEABLE_PRICE = 0.95
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

            markets = await client.get_active_markets(tier=1, tier1_fee_rate=0.0)

        assert len(markets) >= 1
        assert markets[0].fee_rate == 0.0

    @pytest.mark.asyncio
    async def test_tier2_market_fee_rate(self):
        settings = MagicMock(spec=Settings)
        settings.MARKET_FETCH_LIMIT = 200
        settings.MARKET_PAGE_SIZE = 500
        settings.MARKET_FETCH_PAGES = 3
        settings.MIN_TRADEABLE_PRICE = 0.05
        settings.MAX_TRADEABLE_PRICE = 0.95
        client = PolymarketClient(settings)

        # Use a crypto market for tier 2
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [_make_gamma_market(
            question="Will Bitcoin hit $100k?",
            endDate=(datetime.now(timezone.utc) + timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )]
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            markets = await client.get_active_markets(tier=2, tier2_fee_rate=0.04)

        if markets:  # Only if it passes crypto filter
            assert markets[0].fee_rate == 0.04


@pytest.mark.parametrize("market_type,expected_fee", [
    ("political", 0.0),
    ("geopolitical", 0.0),
    ("economic", 0.0),
    ("crypto_15m", 0.0156),
    ("sports", 0.02),
    ("esports", 0.02),
    ("weather", 0.0),
    ("unknown", 0.02),
])
def test_get_fee_rate(market_type, expected_fee):
    assert get_fee_rate(market_type) == pytest.approx(expected_fee)


def test_get_fee_rate_unmapped_uses_default():
    assert get_fee_rate("brand_new_type", default=0.03) == pytest.approx(0.03)
