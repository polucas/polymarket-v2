"""Tests for PolymarketClient with mocked HTTP responses."""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.config import Settings
from src.models import Market, OrderBook
from src.pipelines.polymarket import PolymarketClient, MARKET_TYPE_KEYWORDS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _make_raw_market(
    *,
    id: str = "market-1",
    question: str = "Will BTC hit $100k?",
    outcome_prices: list | None = None,
    end_date: str | None = None,
    hours_ahead: float = 6.0,
    liquidity: float = 10_000.0,
    volume_24h: float = 5_000.0,
    closed: bool = False,
    resolved: bool = False,
    resolution_prices: dict | None = None,
    tags: list | None = None,
) -> dict:
    if outcome_prices is None:
        outcome_prices = [0.65, 0.35]
    if end_date is None:
        future = _utcnow() + timedelta(hours=hours_ahead)
        end_date = future.isoformat()
    raw = {
        "id": id,
        "question": question,
        "outcomePrices": json.dumps(outcome_prices),
        "outcomes": ["Yes", "No"],
        "endDate": end_date,
        "liquidity": str(liquidity),
        "volume24hr": str(volume_24h),
        "closed": closed,
        "resolved": resolved,
        "tags": tags or [],
    }
    if resolution_prices is not None:
        raw["resolutionPrices"] = resolution_prices
    return raw


def _make_settings(**overrides) -> Settings:
    defaults = {
        "ENVIRONMENT": "paper",
        "XAI_API_KEY": "test-key",
        "POLYMARKET_API_KEY": "test-pm-key",
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _mock_response(status_code: int = 200, json_data=None) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data if json_data is not None else []
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            message=f"HTTP {status_code}",
            request=MagicMock(),
            response=resp,
        )
    return resp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def settings():
    return _make_settings()


@pytest.fixture
def client(settings):
    return PolymarketClient(settings)


# ---------------------------------------------------------------------------
# get_active_markets  --  Tier 1
# ---------------------------------------------------------------------------


class TestGetActiveMarketsTier1:
    @pytest.mark.asyncio
    async def test_returns_list_of_markets_with_correct_fields(self, client):
        """Tier 1 call returns List[Market] with all expected attributes."""
        raw = [
            _make_raw_market(
                id="abc-123",
                question="Will Trump win the election?",
                outcome_prices=[0.60, 0.40],
                hours_ahead=12.0,
                liquidity=20_000,
                volume_24h=8_000,
            ),
        ]
        mock_resp = _mock_response(json_data=raw)
        mock_client_instance = AsyncMock()
        mock_client_instance.get = AsyncMock(return_value=mock_resp)
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)

        with patch("src.pipelines.polymarket.httpx.AsyncClient", return_value=mock_client_instance):
            markets = await client.get_active_markets(tier=1)

        assert isinstance(markets, list)
        assert len(markets) == 1
        m = markets[0]
        assert isinstance(m, Market)
        assert m.market_id == "abc-123"
        assert m.question == "Will Trump win the election?"
        assert m.yes_price == pytest.approx(0.60)
        assert m.no_price == pytest.approx(0.40)
        assert m.market_type == "political"
        assert m.fee_rate == 0.02
        assert m.liquidity == 20_000.0
        assert m.volume_24h == 8_000.0
        assert m.resolved is False
        assert m.resolution is None
        assert isinstance(m.keywords, list)

    @pytest.mark.asyncio
    async def test_tier1_excludes_less_than_15min_resolution(self, client):
        """Markets resolving in <15 minutes are excluded from tier 1."""
        raw = [
            _make_raw_market(hours_ahead=0.1, liquidity=10_000),
        ]
        mock_resp = _mock_response(json_data=raw)
        mock_client_instance = AsyncMock()
        mock_client_instance.get = AsyncMock(return_value=mock_resp)
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)

        with patch("src.pipelines.polymarket.httpx.AsyncClient", return_value=mock_client_instance):
            markets = await client.get_active_markets(tier=1)

        assert len(markets) == 0

    @pytest.mark.asyncio
    async def test_tier1_excludes_more_than_7d_resolution(self, client):
        """Markets resolving in >7 days are excluded from tier 1."""
        raw = [
            _make_raw_market(hours_ahead=200.0, liquidity=10_000),
        ]
        mock_resp = _mock_response(json_data=raw)
        mock_client_instance = AsyncMock()
        mock_client_instance.get = AsyncMock(return_value=mock_resp)
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)

        with patch("src.pipelines.polymarket.httpx.AsyncClient", return_value=mock_client_instance):
            markets = await client.get_active_markets(tier=1)

        assert len(markets) == 0

    @pytest.mark.asyncio
    async def test_tier1_excludes_low_liquidity(self, client):
        """Markets with liquidity < $5K are excluded from tier 1."""
        raw = [
            _make_raw_market(hours_ahead=6.0, liquidity=4_999),
        ]
        mock_resp = _mock_response(json_data=raw)
        mock_client_instance = AsyncMock()
        mock_client_instance.get = AsyncMock(return_value=mock_resp)
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)

        with patch("src.pipelines.polymarket.httpx.AsyncClient", return_value=mock_client_instance):
            markets = await client.get_active_markets(tier=1)

        assert len(markets) == 0


# ---------------------------------------------------------------------------
# get_active_markets  --  Tier 2
# ---------------------------------------------------------------------------


class TestGetActiveMarketsTier2:
    @pytest.mark.asyncio
    async def test_tier2_returns_only_crypto_markets(self, client):
        """Tier 2 only returns markets classified as crypto_15m."""
        raw = [
            _make_raw_market(
                id="crypto-1",
                question="Will BTC hit $100k?",
                hours_ahead=6.0,
                liquidity=10_000,
            ),
            _make_raw_market(
                id="politics-1",
                question="Will Trump win the election?",
                hours_ahead=6.0,
                liquidity=10_000,
            ),
        ]
        mock_resp = _mock_response(json_data=raw)
        mock_client_instance = AsyncMock()
        mock_client_instance.get = AsyncMock(return_value=mock_resp)
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)

        with patch("src.pipelines.polymarket.httpx.AsyncClient", return_value=mock_client_instance):
            markets = await client.get_active_markets(tier=2)

        assert len(markets) == 1
        assert markets[0].market_id == "crypto-1"
        assert markets[0].market_type == "crypto_15m"
        assert markets[0].fee_rate == 0.04


# ---------------------------------------------------------------------------
# market_type classification
# ---------------------------------------------------------------------------


class TestMarketTypeClassification:
    @pytest.mark.parametrize(
        "question,expected_type",
        [
            ("Will Trump win the 2024 election?", "political"),
            ("Will the Fed raise interest rate?", "economic"),
            ("Will Bitcoin BTC hit $100k?", "crypto_15m"),
            ("Will the NBA finals go to game 7?", "sports"),
            ("Will this movie win an Oscar?", "cultural"),
            ("Will the SEC approve the ETF?", "regulatory"),
            ("Will a cozy rain occur on Friday?", "political"),  # default fallback
        ],
    )
    def test_classify_market_type(self, client, question, expected_type):
        result = client._classify_market_type(question)
        assert result == expected_type


# ---------------------------------------------------------------------------
# get_orderbook
# ---------------------------------------------------------------------------


class TestGetOrderbook:
    @pytest.mark.asyncio
    async def test_returns_orderbook_with_bids_asks(self, client):
        """get_orderbook returns OrderBook with correct bids and asks lists."""
        ob_data = {
            "bids": [
                {"price": "0.60", "size": "100"},
                {"price": "0.59", "size": "200"},
                {"price": "0.58", "size": "150"},
            ],
            "asks": [
                {"price": "0.62", "size": "80"},
                {"price": "0.63", "size": "120"},
            ],
        }
        mock_resp = _mock_response(json_data=ob_data)
        mock_client_instance = AsyncMock()
        mock_client_instance.get = AsyncMock(return_value=mock_resp)
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)

        with patch("src.pipelines.polymarket.httpx.AsyncClient", return_value=mock_client_instance):
            ob = await client.get_orderbook("token-xyz")

        assert isinstance(ob, OrderBook)
        assert ob.market_id == "token-xyz"
        assert ob.bids == [100.0, 200.0, 150.0]
        assert ob.asks == [80.0, 120.0]
        assert ob.timestamp is not None


# ---------------------------------------------------------------------------
# get_market
# ---------------------------------------------------------------------------


class TestGetMarket:
    @pytest.mark.asyncio
    async def test_returns_market_with_resolution_status(self, client):
        """get_market returns Market including resolved=True and resolution outcome."""
        raw = {
            "id": "m-resolved",
            "question": "Did Trump win?",
            "outcomePrices": json.dumps([1.0, 0.0]),
            "outcomes": ["Yes", "No"],
            "closed": True,
            "resolved": True,
            "resolutionPrices": {"0": "1.0", "1": "0.0"},
        }
        mock_resp = _mock_response(json_data=raw)
        mock_client_instance = AsyncMock()
        mock_client_instance.get = AsyncMock(return_value=mock_resp)
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)

        with patch("src.pipelines.polymarket.httpx.AsyncClient", return_value=mock_client_instance):
            m = await client.get_market("m-resolved")

        assert m is not None
        assert isinstance(m, Market)
        assert m.market_id == "m-resolved"
        assert m.resolved is True
        assert m.resolution == "YES"
        assert m.yes_price == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_timeout_returns_empty_list(self, client):
        """API timeout returns empty list instead of raising."""
        mock_client_instance = AsyncMock()
        mock_client_instance.get = AsyncMock(side_effect=httpx.TimeoutException("connection timed out"))
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)

        with patch("src.pipelines.polymarket.httpx.AsyncClient", return_value=mock_client_instance):
            result = await client.get_active_markets(tier=1)

        assert result == []

    @pytest.mark.asyncio
    async def test_429_returns_empty_list(self, client):
        """HTTP 429 rate limit returns empty list."""
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 429
        mock_resp.json.return_value = []
        # raise_for_status should NOT be called when 429 is detected early
        mock_resp.raise_for_status = MagicMock()

        mock_client_instance = AsyncMock()
        mock_client_instance.get = AsyncMock(return_value=mock_resp)
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)

        with patch("src.pipelines.polymarket.httpx.AsyncClient", return_value=mock_client_instance):
            result = await client.get_active_markets(tier=1)

        assert result == []


# ---------------------------------------------------------------------------
# place_order  --  paper mode
# ---------------------------------------------------------------------------


class TestPlaceOrder:
    @pytest.mark.asyncio
    async def test_paper_mode_rejected(self):
        """Paper mode place_order returns rejected status."""
        settings = _make_settings(ENVIRONMENT="paper")
        client = PolymarketClient(settings)
        result = await client.place_order("market-1", "BUY_YES", 0.60, 50.0)
        assert result["status"] == "rejected"
        assert result["reason"] == "paper_mode"
