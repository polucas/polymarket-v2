"""Tests for market filter logging."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import Settings
from src.pipelines.polymarket import PolymarketClient


def _make_gamma_market(**overrides):
    base = {
        "id": "1",
        "question": "Will X happen?",
        "outcomePrices": '["0.55", "0.45"]',
        "endDate": "2026-02-25T12:00:00Z",
        "liquidity": "20000",
        "volume24hr": "5000",
        "clobTokenIds": '["111", "222"]',
        "closed": False,
        "resolved": False,
    }
    base.update(overrides)
    return base


class TestMarketFilterLogging:
    @pytest.mark.asyncio
    async def test_filter_logging_counts(self):
        settings = MagicMock(spec=Settings)
        client = PolymarketClient(settings)

        markets_data = [
            _make_gamma_market(id="1", liquidity="20000", endDate="2026-02-25T12:00:00Z"),  # passes: within window, high liquidity
            _make_gamma_market(id="2", liquidity="100", endDate="2026-02-25T12:00:00Z"),     # filtered: within window but low liquidity
            _make_gamma_market(id="3", liquidity="20000", endDate="2030-01-01T12:00:00Z"),   # filtered: too far out (>168h)
        ]

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = markets_data
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            with patch("src.pipelines.polymarket.log") as mock_log:
                markets = await client.get_active_markets(tier=1)

                # Verify the filter log was called
                mock_log.info.assert_any_call(
                    "market_filter_results",
                    tier=1,
                    total_from_api=3,
                    passed=1,
                    filtered_resolution=1,
                    filtered_liquidity=1,
                )
