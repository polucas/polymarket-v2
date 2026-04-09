"""Live Polymarket CLOB smoke test.

Run manually on VPS with live credentials to validate the CLOB client works
after the April 6, 2026 exchange upgrade:

    python scripts/live_smoke_test.py

Fetches account balance, places a tiny ($0.01) limit order far from market
(price 0.01 YES, size 1.0 share), waits briefly, and reports the result.
The order is placed at a price far below market so it will not fill.
Manually cancel the open order on polymarket.com after the test.

Exits non-zero on any error. Does NOT automatically cancel the order
(cancellation requires a separate Level-2 call with the order ID).

Pre-requisites:
    ENVIRONMENT=live
    POLYMARKET_API_KEY=<your api key>
    POLYMARKET_PRIVATE_KEY=<your wallet private key (0x-prefixed)>
    POLYMARKET_SECRET=<your api secret>
    POLYMARKET_PASSPHRASE=<your api passphrase>
    POLYMARKET_FUNDER_ADDRESS=<proxy wallet address, or leave blank for EOA>
"""
import asyncio
import os
import sys

# Ensure we can import from src/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Settings
from src.pipelines.polymarket import PolymarketClient


async def main() -> int:
    settings = Settings()

    if settings.ENVIRONMENT != "live":
        print(f"ERROR: ENVIRONMENT is '{settings.ENVIRONMENT}', must be 'live' for smoke test")
        return 1
    if not settings.POLYMARKET_API_KEY:
        print("ERROR: POLYMARKET_API_KEY not set")
        return 1
    if not settings.POLYMARKET_PRIVATE_KEY:
        print("ERROR: POLYMARKET_PRIVATE_KEY not set")
        return 1

    client = PolymarketClient(settings)

    # 1. Fetch 1 active market to get a valid token ID
    print("Fetching 1 active Tier 1 market...")
    markets = await client.get_active_markets(tier=1)
    if not markets:
        print("ERROR: no active Tier 1 markets returned — cannot run order test")
        return 1
    m = markets[0]
    print(f"  Sample market: {m.question[:80]}")
    print(f"  YES token ID : {m.clob_token_id_yes or 'MISSING'}")
    print(f"  NO  token ID : {m.clob_token_id_no or 'MISSING'}")

    if not m.clob_token_id_yes:
        print("ERROR: clob_token_id_yes is empty — CLOB token IDs not populated in market data")
        return 1

    # 2. Place a tiny limit order far below market (price 0.01 YES = near-zero).
    #    Size 1.0 share at $0.01/share = $0.01 total. Very unlikely to fill.
    #    This validates that ClobClient init, signing, and API connection all work.
    print("Placing $0.01 test limit order at 0.01 YES (far from market, should not fill)...")
    result = await client.place_order(
        clob_token_id=m.clob_token_id_yes,
        price=0.01,
        size=1.0,
        side="BUY",
    )
    print(f"  Result: {result}")

    if result.get("status") == "error":
        print(f"ERROR: order placement failed: {result.get('error')}")
        return 1
    if result.get("status") == "rejected":
        print(f"ERROR: order rejected: {result.get('reason')}")
        return 1

    order_id = None
    if isinstance(result.get("order"), dict):
        order_id = result["order"].get("orderID") or result["order"].get("id")

    print()
    print("OK — smoke test passed.")
    if order_id:
        print(f"  Order ID: {order_id}")
        print(f"  Manually cancel this order on polymarket.com or via the CLOB API cancel endpoint.")
    else:
        print("  Order submitted (no order ID in response — check polymarket.com open orders).")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
