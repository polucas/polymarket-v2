# Polymarket v2 — Test Agent Specification

> **Purpose:** Each section below is a self-contained prompt for one test agent. Test agents are fully independent from developer agents to avoid cross-contamination. Each tests **contracts and interfaces**, not implementation details.
>
> **Test framework:** `pytest` + `pytest-asyncio`
> **Mocking:** `unittest.mock` (patch, MagicMock, AsyncMock)
> **Source of truth:** `polymarket_system_v2.md` in project root + interface contracts in `docs/TASKS.md`

---

## Execution DAG

```
Wave 1:  TEST-01                  (after DEV-01)
Wave 2:  TEST-02  TEST-05  TEST-06  TEST-07   (after their respective DEV agents)
Wave 3:  TEST-03  TEST-04  TEST-08  TEST-09   (after their respective DEV agents)
Wave 4:  TEST-10                               (after DEV-11 + DEV-12)
```

### Dependency Matrix

| Test Agent | Tests For | Can Start After |
|------------|-----------|-----------------|
| TEST-01 | DEV-01 (Foundation) | DEV-01 complete |
| TEST-02 | DEV-02 (Signal Classifier) | DEV-02 complete |
| TEST-03 | DEV-03 (Twitter) | DEV-03 complete |
| TEST-04 | DEV-04 (RSS) | DEV-04 complete |
| TEST-05 | DEV-05 (Polymarket) | DEV-05 complete |
| TEST-06 | DEV-06 (Grok + Context) | DEV-06 complete |
| TEST-07 | DEV-07 (Trade Engine) | DEV-07 complete |
| TEST-08 | DEV-08 (Execution) | DEV-08 complete |
| TEST-09 | DEV-09 + DEV-10 (Learning) | DEV-10 complete |
| TEST-10 | DEV-11 + DEV-12 (Integration) | DEV-11 + DEV-12 complete |

### Key Principle: Isolation

- Each test agent uses **mocks** for all external dependencies (APIs, file I/O)
- Each test agent uses **in-memory SQLite** for database tests
- Test agents never import from other test files (no shared test state)
- Tests validate the **contract** (function signature, return types, edge behavior) not the implementation

---

## TEST-01: Foundation Layer

**Agent ID:** `TEST-01-FOUNDATION`
**Tests For:** DEV-01 (config.py, models.py, db/sqlite.py, db/migrations.py)
**Complexity:** MEDIUM

### Files to Create

```
tests/__init__.py
tests/conftest.py
tests/test_config.py
tests/test_models.py
tests/test_db.py
```

### `tests/conftest.py` — Shared Fixtures

```python
import pytest
import pytest_asyncio
from src.db.sqlite import Database
from src.db.migrations import run_migrations
from src.models import *
from datetime import datetime, timedelta
import uuid

@pytest_asyncio.fixture
async def db():
    """In-memory SQLite database with schema applied."""
    database = await Database.init(":memory:")
    await run_migrations(database)
    yield database
    await database.close()

@pytest.fixture
def sample_trade_record():
    """Factory for TradeRecord with sensible defaults."""
    def _make(**overrides):
        defaults = {
            "record_id": str(uuid.uuid4()),
            "experiment_run": "test-run-001",
            "timestamp": datetime.utcnow(),
            "model_used": "grok-4-1-fast-reasoning",
            "market_id": "market-001",
            "market_question": "Will X happen?",
            "market_type": "political",
            "resolution_window_hours": 12.0,
            "resolution_datetime": None,  # Exact market resolution time (populated from Gamma API endDate)
            "tier": 1,
            "grok_raw_probability": 0.75,
            "grok_raw_confidence": 0.80,
            "grok_reasoning": "Test reasoning",
            "grok_signal_types": [{"source_tier": "S2", "info_type": "I2", "content": "test"}],
            "calibration_adjustment": 0.0,
            "market_type_adjustment": 0.0,
            "signal_weight_adjustment": 0.0,
            "final_adjusted_probability": 0.73,
            "final_adjusted_confidence": 0.78,
            "market_price_at_decision": 0.60,
            "orderbook_depth_usd": 5000.0,
            "fee_rate": 0.02,
            "calculated_edge": 0.11,
            "action": "BUY_YES",
            "skip_reason": None,
            "position_size_usd": 200.0,
            "kelly_fraction_used": 0.25,
            "actual_outcome": None,
            "pnl": None,
            "brier_score_raw": None,
            "brier_score_adjusted": None,
            "resolved_at": None,
            "unrealized_adverse_move": None,
            "voided": False,
            "void_reason": None,
        }
        defaults.update(overrides)
        return TradeRecord(**defaults)
    return _make

@pytest.fixture
def sample_signal():
    """Factory for Signal objects."""
    def _make(**overrides):
        defaults = {
            "source": "twitter",
            "source_tier": "S3",
            "info_type": None,
            "content": "Breaking: test signal content",
            "credibility": 0.80,
            "author": "TestAuthor",
            "followers": 50000,
            "engagement": 100,
            "timestamp": datetime.utcnow(),
            "headline_only": False,
        }
        defaults.update(overrides)
        return Signal(**defaults)
    return _make
```

### `tests/test_config.py`

```
Test cases:
1. Settings loads from environment variables (set via monkeypatch)
2. Settings has correct defaults for optional fields (ENVIRONMENT="paper", LOG_LEVEL="INFO")
3. All required fields (XAI_API_KEY, TWITTER_API_KEY) cause error if missing
4. MonkModeConfig correctly built from Settings values
5. Numeric config values have correct types (float, int)
6. KELLY_FRACTION default is 0.25
7. MAX_CLUSTER_EXPOSURE_PCT default is 0.12
```

### `tests/test_models.py`

```
CalibrationBucket tests:
1. expected_accuracy = alpha / (alpha + beta)
   - alpha=10, beta=2 -> expected_accuracy = 0.833...
2. sample_count = int(alpha + beta - 2)
   - alpha=1, beta=1 -> sample_count = 0
   - alpha=6, beta=4 -> sample_count = 8
3. update(was_correct=True) increments alpha by recency_weight
   update(was_correct=False) increments beta by recency_weight
4. get_correction() returns 0.0 when sample_count < 10
5. get_correction() with sufficient samples:
   - Bucket (0.60, 0.70), midpoint=0.65
   - If expected_accuracy=0.70, correction is positive (model underconfident)
   - If expected_accuracy=0.55, correction is negative (model overconfident)
   - Correction weighted by certainty = max(0, 1 - uncertainty * 2)
6. uncertainty uses scipy beta distribution 95% CI width

MarketTypePerformance tests:
7. avg_brier with empty brier_scores returns 0.25 (default)
8. avg_brier with single score returns that score
9. avg_brier with multiple scores applies 0.95 exponential decay:
   - Recent scores weighted more heavily
   - Verify with known values: [0.10, 0.20, 0.30] with weights [0.95^2, 0.95^1, 0.95^0]
10. edge_adjustment returns 0.0 when total_trades < 15
11. edge_adjustment returns 0.05 when avg_brier > 0.30 and 15+ trades
12. edge_adjustment returns 0.03 when 0.25 < avg_brier <= 0.30
13. edge_adjustment returns 0.01 when 0.20 < avg_brier <= 0.25
14. edge_adjustment returns 0.0 when avg_brier <= 0.20
15. should_disable: True when total_trades >= 30 AND total_pnl < -0.15 * abs(total_trades)
16. should_disable: False when < 30 trades

SignalTracker tests:
17. lift returns 1.0 when total_present < 5 (insufficient data)
18. lift returns 1.0 when total_absent < 5 (insufficient data)
19. lift calculation: win_rate_present / win_rate_absent
    - present: 8 wins, 2 losses (80% win rate)
    - absent: 5 wins, 5 losses (50% win rate)
    - lift = 0.80 / 0.50 = 1.60
20. weight = 1.0 + (lift - 1.0) * 0.3, clamped [0.8, 1.2]
    - lift=1.6 -> raw=1.18 -> weight=1.18
    - lift=3.0 -> raw=1.60 -> weight=1.20 (clamped)
    - lift=0.2 -> raw=0.76 -> weight=0.80 (clamped)
21. weight = 1.0 when insufficient data (lift=1.0)

TradeRecord tests:
22. All fields have correct types and defaults
23. voided defaults to False
24. skip_reason defaults to None
25. brier_score fields default to None (set on resolution)

SOURCE_TIER_CREDIBILITY tests:
26. Exact values: S1=0.95, S2=0.90, S3=0.80, S4=0.65, S5=0.70, S6=0.30
```

### `tests/test_db.py`

```
Schema tests:
1. run_migrations() creates all 9 tables without error
2. Running run_migrations() twice is idempotent
3. All expected tables exist: experiment_runs, model_swaps, trade_records,
   calibration_state, market_type_performance, signal_trackers, portfolio,
   api_costs, daily_mode_log

Trade Record CRUD:
4. save_trade() + get_trade() round-trips correctly (all fields preserved)
5. get_open_trades() returns only records where actual_outcome IS NULL AND voided = False
6. get_open_trades() excludes voided trades
7. get_today_trades() returns only trades from today (UTC)
8. get_week_trades() returns trades from last 7 days
9. update_trade() modifies existing record
10. count_today_trades() returns correct count
11. count_open_trades() returns correct count

Calibration persistence:
12. save_calibration() + load_calibration() round-trips all 6 buckets
13. Bucket alpha/beta values preserved exactly

Market Type Performance persistence:
14. save + load round-trip for multiple market types
15. brier_scores JSON array preserved correctly

Signal Tracker persistence:
16. save + load round-trip for (source_tier, info_type, market_type) key
17. All 4 counter fields preserved

Portfolio:
18. save + load portfolio (single row, id=1 constraint)
19. Portfolio update overwrites existing row

API Costs:
20. increment_api_cost() creates new row for new (date, service) pair
21. increment_api_cost() updates existing row (adds to counts)
22. get_today_api_spend() returns sum of today's costs

Experiment Runs:
23. save_experiment() + get_current_experiment() returns the active experiment
24. end_experiment() sets ended_at

Parse Failures:
25. record_parse_failure() increments failure count

Concurrent Access:
26. Two simultaneous writes to different tables don't conflict
```

### Mocking Strategy

No mocks needed — foundation layer uses in-memory SQLite for DB tests and environment variables (via monkeypatch) for config tests.

---

## TEST-02: Signal Classifier

**Agent ID:** `TEST-02-SIGNAL-CLASSIFIER`
**Tests For:** DEV-02 (signal_classifier.py)
**Complexity:** LOW

### Files to Create

```
tests/test_signal_classifier.py
```

### Test Cases

```
S1 (Official Primary):
1. RSS domain "federalreserve.gov" -> "S1"
2. RSS domain "sec.gov" -> "S1"
3. Twitter handle "@WhiteHouse" -> "S1"
4. Twitter handle "@FederalReserve" -> "S1"

S2 (Wire Service):
5. RSS domain "reuters.com" -> "S2"
6. RSS domain "apnews.com" -> "S2"
7. Twitter handle "@Reuters" -> "S2"
8. Twitter handle "@AP" -> "S2"

S3 (Institutional Media):
9. RSS domain "bbc.com" -> "S3"
10. RSS domain "coindesk.com" -> "S3"
11. Twitter handle "@nytimes" -> "S3"
12. Twitter handle "@CNBC" -> "S3"

S4 (Verified Expert):
13. Verified=True, followers=60000, bio="journalist at XYZ" -> "S4"
14. Verified=True, followers=100000, bio="economist, professor" -> "S4"
15. Verified=True, followers=60000, bio="I love cats" -> "S6" (no expert keyword)
16. Verified=True, followers=40000, bio="journalist" -> "S6" (below 50K threshold)
17. Verified=False, followers=200000, bio="analyst" -> "S6" (not verified)

S5 (Market Derived):
18. source_type="market_data" -> "S5" regardless of other fields
19. source_type="market_data" with all twitter fields populated -> still "S5"

S6 (Unverified Social):
20. Unknown RSS domain "randomblog.com" -> "S6"
21. Unknown Twitter handle "@randomuser" -> "S6"
22. Verified=True, followers=5000 -> "S6" (below threshold)

Edge cases:
23. Empty bio string with verified=True, followers=60000 -> "S6"
24. Missing follower_count key -> "S6" (should not crash)
25. Handle case sensitivity (e.g., "@whitehouse" vs "@WhiteHouse")
26. SOURCE_TIER_CREDIBILITY dict has all 6 tiers with correct values
```

### Mocking Strategy

No mocks needed — pure deterministic logic. May need to mock `yaml.safe_load()` if file path is hardcoded, or provide a test YAML fixture.

---

## TEST-03: Twitter Pipeline

**Agent ID:** `TEST-03-TWITTER`
**Tests For:** DEV-03 (twitter.py)
**Complexity:** MEDIUM

### Files to Create

```
tests/test_twitter.py
```

### Test Cases

```
Successful flow:
1. get_signals_for_market(["bitcoin"]) returns List[Signal]
2. Each Signal has: source="twitter", source_tier in S1-S6, info_type=None,
   content (str), credibility (float), author (str), followers (int), timestamp

Pre-filtering:
3. Tweets with followers < 1000 are excluded
4. Tweets with engagement < 10 are excluded
5. Bot accounts are excluded (implement test for _is_bot_account)
6. All tweets pass filter -> all returned (up to max)

Deduplication:
7. Two tweets with 90% word overlap -> only one signal returned
8. Two tweets with 30% word overlap -> both returned
9. Three identical tweets -> one signal returned

Source tier assignment:
10. Tweet from @Reuters -> Signal.source_tier == "S2"
11. Tweet from unknown account -> Signal.source_tier == "S6"

Ordering and limits:
12. Signals sorted by credibility descending
13. Maximum 10 signals returned even with 50 valid tweets

Error handling:
14. API timeout -> returns empty list, no exception raised
15. API returns 429 (rate limit) -> retries then returns empty list
16. Malformed API response -> returns empty list, logs warning
17. Empty keyword list -> returns empty list

API call structure:
18. Query built as " OR ".join(keywords)
19. Recency parameter set to "2h"
```

### Mocking Strategy

- **Mock `httpx.AsyncClient.get/post`** with canned JSON responses:
  ```python
  SAMPLE_TWEET_RESPONSE = {
      "tweets": [
          {"text": "Bitcoin surges past $100K", "author": {"screen_name": "CoinDesk",
           "verified": True, "followers_count": 500000, "bio": "crypto news"},
           "engagement_score": 5000, "created_at": "2026-02-21T10:00:00Z"},
          # ... more tweets
      ]
  }
  ```
- Mock `classify_source_tier` to return predictable values (or use real implementation if available)

---

## TEST-04: RSS Pipeline

**Agent ID:** `TEST-04-RSS`
**Tests For:** DEV-04 (rss.py)
**Complexity:** LOW

### Files to Create

```
tests/test_rss.py
```

### Test Cases

```
New headline detection:
1. First call with fresh headlines -> returns Signal for each
2. Each Signal has: source="rss", headline_only=True, source_tier assigned

Deduplication:
3. Same headline text on second call within 24h -> not returned
4. Same headline text after 24h+ -> returned again (pruned from seen)

Age filtering:
5. Headline published 30 minutes ago -> included
6. Headline published 3 hours ago -> excluded (>2h threshold)
7. Headline with no published date -> treated as "now", included

Bounded cache:
8. _prune_old_headlines() removes entries older than 24h
9. After pruning, seen_headlines dict size decreased
10. Adding 1000 headlines doesn't cause memory issues

Source tier assignment:
11. Feed domain "reuters.com" -> source_tier="S2"
12. Feed domain "bbc.com" -> source_tier="S3"
13. Feed domain "coindesk.com" -> source_tier="S3"

Error handling:
14. One feed fails to parse -> other feeds still processed, no crash
15. Feed returns empty entries list -> no signals from that feed
16. Network error on feed URL -> logs warning, continues

Multiple feeds:
17. Headlines from different feeds are all included (not deduplicated across feeds unless same text)
18. Each feed processes up to 10 entries
```

### Mocking Strategy

- **Mock `feedparser.parse()`** with canned feed data:
  ```python
  SAMPLE_FEED = MagicMock()
  SAMPLE_FEED.entries = [
      MagicMock(title="Fed holds rates steady", published="Sat, 21 Feb 2026 10:00:00 GMT"),
      MagicMock(title="Supreme Court ruling on...", published="Sat, 21 Feb 2026 09:30:00 GMT"),
  ]
  ```
- Mock `datetime.utcnow()` to control time-based tests

---

## TEST-05: Polymarket Client

**Agent ID:** `TEST-05-POLYMARKET`
**Tests For:** DEV-05 (polymarket.py)
**Complexity:** MEDIUM

### Files to Create

```
tests/test_polymarket.py
```

### Test Cases

```
Market discovery:
1. get_active_markets(tier=1) returns List[Market] with correct fields
2. Market objects have: market_id, question, yes_price, no_price, resolution_time,
   hours_to_resolution, volume_24h, liquidity, market_type, fee_rate
3. Tier 1 filter: markets with <1h resolution excluded
4. Tier 1 filter: markets with >24h resolution excluded
5. Tier 1 filter: markets with liquidity < $5K excluded
6. Tier 2 filter: only crypto markets returned
7. market_type correctly derived from API response

Orderbook:
8. get_orderbook() returns OrderBook with bids and asks lists
9. OrderBook bids/asks have correct number of levels (top 5)
10. Empty orderbook handled gracefully

Resolution:
11. get_market() for resolved market -> market.resolved == True
12. get_market() for unresolved market -> market.resolved == False

Error handling:
13. 429 rate limit -> retries with backoff
14. 500 server error -> retries then raises/returns None
15. Timeout -> raises/returns None with log
16. Invalid JSON response -> handled gracefully
```

### Mocking Strategy

- **Mock `httpx.AsyncClient`** for Gamma API responses:
  ```python
  SAMPLE_MARKETS_RESPONSE = [
      {"condition_id": "abc123", "question": "Will Fed cut rates?",
       "tokens": [{"outcome": "Yes", "price": 0.65}, {"outcome": "No", "price": 0.35}],
       "end_date_iso": "2026-02-22T18:00:00Z", "volume": 150000, "liquidity": 25000},
  ]
  ```
- **Mock `py_clob_client`** for CLOB API operations

---

## TEST-06: Grok Client + Context Builder

**Agent ID:** `TEST-06-GROK-CONTEXT`
**Tests For:** DEV-06 (grok_client.py, context_builder.py)
**Complexity:** MEDIUM-HIGH

### Files to Create

```
tests/test_grok_client.py
tests/test_context_builder.py
```

### `tests/test_grok_client.py`

```
parse_json_safe() tests:
1. Valid JSON string -> parsed dict
2. JSON wrapped in ```json ... ``` -> parsed dict
3. JSON wrapped in ``` ... ``` (no json tag) -> parsed dict
4. JSON with preamble text "Here is my analysis: {...}" -> parsed dict
5. JSON with postamble text "{...} Hope this helps!" -> parsed dict
6. Completely invalid string "not json at all" -> None
7. Nested braces in JSON values -> correctly parsed
8. Empty string -> None
9. Only whitespace -> None
10. JSON with trailing comma (common LLM error) -> attempt to parse, may fail gracefully

Field validation:
11. Response missing "estimated_probability" -> retry triggered
12. Response missing "confidence" -> retry triggered
13. Response missing "reasoning" -> retry triggered
14. Response missing "signal_info_types" -> retry triggered
15. probability = 1.5 (> 1.0) -> retry triggered
16. confidence = -0.1 (< 0.0) -> retry triggered
17. probability = "0.75" (string) -> coerced to float 0.75, accepted
18. confidence = "0.80" (string) -> coerced to float 0.80, accepted

Retry logic:
19. Success on first attempt -> returns parsed dict, no retry
20. Parse failure on 1st, success on 2nd -> returns dict from 2nd attempt
21. Parse failure on 1st and 2nd, success on 3rd -> returns dict from 3rd
22. All 3 attempts fail -> returns None
23. API error on 1st, success on 2nd -> returns dict (error recovery)

Backoff:
24. Linear backoff between retries: sleep called with 1.0 then 2.0 seconds

Side effects:
25. On success: db.increment_api_cost called with "grok" service
26. On total failure: db.record_parse_failure called with market_id
27. All attempts logged with structlog (warning for retries, error for total failure)
```

### `tests/test_context_builder.py`

```
extract_keywords() tests:
28. "Will Donald Trump sign the executive order?" -> extracts "Donald Trump" (named entity)
29. "Will BTC reach $100K?" -> extracts "BTC" (acronym)
30. "Will the SEC approve the ETF?" -> extracts "SEC", "ETF"
31. "Will the Fed cut rates?" -> extracts "Fed" + adds supplements ["federal reserve", "interest rate"]
32. Complex question with < 2 regex matches -> falls back to LLM
33. Cache hit: second call with same market_id returns cached result without LLM call
34. Keywords limited to max 5

build_grok_context() tests:
35. Output string contains market question
36. Output string contains current YES/NO prices
37. Output string contains resolution time
38. Output string contains volume and liquidity
39. Output string contains orderbook bid/ask depth and skew
40. Signals sorted by credibility, top 7 included
41. Each signal line shows source tier and credibility
42. Prompt includes info-type classification instructions (I1-I5)
43. Prompt requests JSON response format
44. With 10 signals, only top 7 appear in context
45. With 0 signals, signal section is empty but prompt is still valid

Orderbook calculations:
46. bid_depth = sum of top 5 bids
47. ask_depth = sum of top 5 asks
48. book_skew = bid_depth / ask_depth (protected from division by zero)
```

### Mocking Strategy

- **Mock `httpx.AsyncClient.post`** for Grok API calls:
  ```python
  VALID_GROK_RESPONSE = json.dumps({
      "estimated_probability": 0.75,
      "confidence": 0.80,
      "reasoning": "Based on wire reports...",
      "key_signals_used": ["Reuters: Fed signals hold"],
      "contradictions": [],
      "signal_info_types": {"Fed signals hold": "I2"}
  })
  ```
- **Mock `db.increment_api_cost`** and **`db.record_parse_failure`** to verify calls
- **Mock `asyncio.sleep`** to avoid actual delays in tests
- For context_builder: create Market, Signal, OrderBook fixtures with known values

---

## TEST-07: Trade Ranker + Decision Engine

**Agent ID:** `TEST-07-TRADE-ENGINE`
**Tests For:** DEV-07 (trade_ranker.py, trade_decision.py)
**Complexity:** HIGH

### Files to Create

```
tests/test_trade_ranker.py
tests/test_trade_decision.py
```

### `tests/test_trade_ranker.py`

```
Score calculation:
1. score = edge * confidence * (1.0 / max(resolution_hours, 0.5))
   - edge=0.10, confidence=0.80, resolution=2h -> score = 0.10 * 0.80 * 0.50 = 0.04
2. Faster resolution -> higher time_value:
   - 1h resolution: time_value = 1.0
   - 24h resolution: time_value = 0.0417
3. Resolution < 0.5h clamped: 0.25h -> time_value = 1.0/0.5 = 2.0

Ranking:
4. 3 candidates with scores [0.04, 0.08, 0.02] -> ranked [0.08, 0.04, 0.02]
5. With remaining_cap=2: top 2 executed, 3rd gets skip_reason="ranked_below_cutoff"
6. With remaining_cap=0: all get skip_reason="ranked_below_cutoff"

Cluster detection:
7. Two markets: same category, resolution within 30min, 60% keyword overlap -> same cluster
8. Two markets: same category, resolution within 30min, 20% keyword overlap -> different clusters
9. Two markets: different categories, resolution within 30min, 80% overlap -> different clusters (different category)
10. Two markets: same category, resolution 2h apart, 80% overlap -> different clusters (>1h gap)

_keyword_overlap (Jaccard):
11. ["trump", "election"] vs ["trump", "vote"] -> intersection=1, union=3 -> 0.333
12. ["fed", "rates", "fomc"] vs ["fed", "rates", "cut"] -> 2/4 = 0.50
13. Empty keyword list -> 0.0

Cluster exposure:
14. Cluster with $500 existing + $100 pending + $200 new candidate, bankroll=$5000:
    total=$800, limit=12%*5000=$600 -> EXCEEDS limit, returns False
15. Same but bankroll=$10000: limit=$1200 -> WITHIN limit, returns True
16. No existing exposure in cluster -> candidate always passes if small enough

Edge cases:
17. Zero candidates -> ([], [])
18. Single candidate within cap -> ([candidate], [])
19. All candidates in same cluster -> only first fits, rest skipped for cluster limit
```

### `tests/test_trade_decision.py`

```
calculate_edge():
20. adjusted_prob=0.75, market_price=0.60, fee=0.02 -> edge = 0.15 - 0.02 = 0.13
21. adjusted_prob=0.55, market_price=0.60, fee=0.02 -> edge = 0.05 - 0.02 = 0.03
22. adjusted_prob=0.60, market_price=0.60, fee=0.02 -> edge = 0.0 - 0.02 = -0.02 (negative)

determine_side():
23. adjusted_prob=0.75, market_price=0.60 -> "BUY_YES"
24. adjusted_prob=0.40, market_price=0.60 -> "BUY_NO"
25. adjusted_prob=0.60, market_price=0.60 -> "SKIP"

kelly_size() BUY_YES:
26. prob=0.80, price=0.60, bankroll=5000:
    f* = (0.80-0.60)/(1-0.60) = 0.50, quarter=0.125, size=$625, cap=1.844%*5000=$92.20
    -> returns $92.20 (capped)
27. prob=0.65, price=0.60, bankroll=5000:
    f* = (0.65-0.60)/(1-0.60) = 0.125, quarter=0.03125, size=$156.25
    -> returns $156.25 (under cap)
28. prob=0.55, price=0.60: prob <= price -> returns 0.0

kelly_size() BUY_NO:
29. prob=0.30, price=0.60, bankroll=5000:
    f* = (0.60-0.30)/0.60 = 0.50, quarter=0.125, size=$625, cap=$92.20
    -> returns $92.20 (capped)
30. prob=0.55, price=0.60, bankroll=5000:
    f* = (0.60-0.55)/0.60 = 0.0833, quarter=0.0208, size=$104.17
    -> returns $104.17
31. prob=0.65, price=0.60: prob >= price -> returns 0.0

kelly_size() edge cases:
32. kelly_fraction=0.0 -> returns 0.0
33. max_position_pct=0.0 -> returns 0.0

check_monk_mode():
34. Tier 1 with 20 executed trades today -> False, "tier1_daily_cap_reached"
35. Tier 1 with 19 executed trades -> passes this check
36. Tier 2 with 3 executed trades -> False, "tier2_daily_cap_reached"
37. Today PnL = -$260 with equity $5000 (5.2% loss) -> False, "daily_loss_limit"
38. Today PnL = -$240 (4.8% loss) -> passes this check
39. Week PnL = -$510 with equity $5000 (10.2% loss) -> False, "weekly_loss_limit"
40. 3 consecutive losses (all resolved with negative PnL) -> False, "cooldown_until_..."
41. 2 consecutive losses + 1 unrealized adverse move >10% = 3 adverse -> False, "cooldown_until_..."
42. 2 consecutive losses only -> passes (below threshold of 3)
43. Cooldown expired (>2 hours ago) -> passes
44. Total exposure at 29% + new 5% trade -> False, "max_exposure"
45. API spend = $8.50 -> False, "api_budget_exceeded"
46. All checks pass -> True, None

get_scan_mode():
47. 20+ tier1 executed trades -> "observe_only"
48. 19 tier1 executed + 2 tier1 skipped -> "active" (skips don't count)
49. 0 tier1 trades -> "active"
```

### Mocking Strategy

No mocks needed — all pure logic with dataclass inputs. Create TradeCandidate and Portfolio fixtures directly.

---

## TEST-08: Execution + Resolution

**Agent ID:** `TEST-08-EXECUTION`
**Tests For:** DEV-08 (execution.py, resolution.py)
**Complexity:** MEDIUM

### Files to Create

```
tests/test_execution.py
tests/test_resolution.py
```

### `tests/test_execution.py`

```
Taker execution simulation:
1. Slippage formula: size_usd=100, orderbook_depth=5000:
   size_ratio = 100/5000 = 0.02, slippage = 0.005 + 0.01*0.02 = 0.0052
2. Slippage with large order: size_usd=10000, depth=5000:
   size_ratio = 2.0 -> capped at 1.0, slippage = 0.005 + 0.01*1.0 = 0.015
3. YES taker: price=0.60, slippage=0.01 -> executed_price=0.606
4. NO taker: price=0.60, slippage=0.01 -> executed_price=0.594
5. Taker fill_probability always 1.0
6. Taker filled always True

Maker execution simulation:
7. Maker slippage always 0.0
8. Maker executed_price equals input price
9. Maker fill_probability at price=0.50: 0.4 + 0.4*(1-0) = 0.80
10. Maker fill_probability at price=0.90: 0.4 + 0.4*(1-0.4) = 0.64
11. Maker fill_probability at price=0.10: 0.4 + 0.4*(1-0.4) = 0.64

Price clamping:
12. executed_price > 0.99 -> clamped to 0.99
13. executed_price < 0.01 -> clamped to 0.01

execute_trade():
14. Paper mode: calls simulate_execution, returns TradeRecord
15. Paper mode, unfilled maker: returns None
16. Live mode: calls polymarket_client.place_order
17. Portfolio updated: cash decreased, position added
```

### `tests/test_resolution.py`

```
auto_resolve_trades():
18. Resolved market (outcome=YES): trade pnl calculated correctly
    - BUY_YES at 0.60, size $100: pnl = $100 * (1/0.60 - 1) = $66.67
19. Resolved market (outcome=NO): BUY_YES trade loses full stake
    - pnl = -$100
20. BUY_NO at 0.60, outcome=NO: pnl = $100 * (1/0.40 - 1) = $150
21. Unresolved market: trade remains open, no changes
22. Brier scores calculated on resolution:
    - brier_score_raw = (grok_raw_probability - actual)^2
    - brier_score_adjusted = (final_adjusted_probability - actual)^2
23. Voided trade excluded from resolution

calculate_hypothetical_pnl():
24. Skipped BUY_YES: what if we bought at market_price with position_size?
    Outcome YES: positive pnl. Outcome NO: negative pnl.

update_unrealized_adverse_moves():
25. BUY_YES at 0.60, current market price 0.48 -> adverse move = 0.12/0.60 = 0.20 (20%)
26. BUY_YES at 0.60, current market price 0.65 -> adverse move = 0 (price moved favorably)
27. BUY_NO at 0.40, current market price 0.52 -> adverse move = 0.12/0.40 = 0.30
```

### Mocking Strategy

- **Mock `random.random()`** for deterministic fill simulation
- **Mock `PolymarketClient.get_market()`** to return resolved/unresolved market objects
- **Mock `PolymarketClient.place_order()`** for live execution tests
- **Mock `Database`** for save/update calls

---

## TEST-09: Learning System (All Layers)

**Agent ID:** `TEST-09-LEARNING`
**Tests For:** DEV-09 (calibration.py, market_type.py, signal_tracker.py) + DEV-10 (adjustment.py, experiments.py, model_swap.py)
**Complexity:** HIGH — This is the most critical test suite

### Files to Create

```
tests/test_calibration.py
tests/test_market_type_learning.py
tests/test_signal_tracker_learning.py
tests/test_adjustment_pipeline.py
tests/test_model_swap.py
```

### `tests/test_calibration.py`

```
CalibrationManager:
1. find_bucket(0.55) -> bucket (0.50, 0.60)
2. find_bucket(0.65) -> bucket (0.60, 0.70)
3. find_bucket(0.95) -> bucket (0.90, 0.95) [note: 0.95 is start of last bucket]
4. find_bucket(0.97) -> bucket (0.95, 1.00)
5. find_bucket(0.50) -> bucket (0.50, 0.60) [boundary]

get_correction():
6. Fresh bucket (sample_count=0) -> correction = 0.0
7. Bucket with 9 samples -> correction = 0.0 (below 10 minimum)
8. Bucket with 15 samples, expected_accuracy > midpoint -> positive correction

update_calibration() CRITICAL CORRECTNESS:
9. Trade with grok_raw_probability=0.75 (predicted YES), outcome=True (correct):
   -> bucket (0.70, 0.80) alpha incremented
10. Trade with grok_raw_probability=0.75, outcome=False (incorrect):
    -> bucket (0.70, 0.80) beta incremented
11. CRITICAL: Uses grok_raw_probability, NOT final_adjusted_probability
    - Create trade where raw=0.75 (bucket 0.70-0.80) but adjusted=0.85 (bucket 0.80-0.90)
    - Verify update goes to bucket (0.70, 0.80), NOT (0.80, 0.90)
12. Recency weight applied: trade from 10 days ago -> weight = 0.95^10 = 0.5987

reset_to_priors():
13. After reset, all buckets have alpha=1.0, beta=1.0
14. After reset, all get_correction() returns 0.0
```

### `tests/test_market_type_learning.py`

```
MarketTypeManager:
15. update with executed trade: total_trades incremented, brier_score appended, pnl added
16. update with skipped trade: total_observed incremented, counterfactual_pnl updated
17. Uses brier_score_adjusted (NOT raw) for market-type tracking
    - Create trade where raw Brier=0.10 but adjusted Brier=0.25
    - Verify 0.25 is appended to brier_scores, not 0.10

get_edge_adjustment():
18. Unknown market type -> 0.0
19. Market type with < 15 trades -> 0.0
20. Market type with 20 trades, avg_brier=0.32 -> 0.05
21. Market type with 20 trades, avg_brier=0.27 -> 0.03
22. Market type with 20 trades, avg_brier=0.18 -> 0.0

dampen_on_swap():
23. Market type with 50 Brier scores -> truncated to last 15
24. Market type with 10 Brier scores -> unchanged (< 15)
```

### `tests/test_signal_tracker_learning.py`

```
SignalTrackerManager:
25. Trade with S2/I2 signal, correct outcome:
    - (S2, I2, market_type) tracker: present_winning += 1
    - All other observed combos for this market_type: absent_winning += 1
26. Trade with S2/I2 signal, incorrect outcome:
    - (S2, I2, market_type) tracker: present_losing += 1
27. Uses ADJUSTED correctness, not raw
    - Trade where raw prediction correct but adjusted prediction incorrect
    - Verify signal tracker counts the adjusted result (incorrect)

get_signal_weight():
28. Tracker with < 5 present samples -> returns 1.0 (no influence)
29. Tracker with sufficient data, lift=1.5 -> weight between 0.8 and 1.2
```

### `tests/test_adjustment_pipeline.py`

```
adjust_prediction() 5-step order:
30. With no calibration data (fresh buckets), no signal data, no market-type data:
    -> adjusted_probability = grok_probability (no adjustments active)
    -> adjusted_confidence = grok_confidence (no adjustments active)
    -> extra_edge = 0.0

Step 1 - Calibration:
31. Calibration correction of +0.05 -> confidence increases by 0.05
32. Calibration correction of -0.05 -> confidence decreases by 0.05
33. Adjusted confidence clamped to [0.50, 0.99]

Step 2 - Signal weighting:
34. Single signal with weight 1.15 -> confidence += (1.15 - 1.0) * 0.1 = +0.015
35. Two signals with weights [1.10, 1.20] -> avg=1.15, same as above
36. No signal tags -> no adjustment (skip step)

Step 3 - Probability shrinkage:
37. Overconfident model (expected_accuracy < midpoint):
    - p=0.80, shrinkage_factor=0.85 -> adjusted = 0.5 + 0.30*0.85 = 0.755 (shrunk toward 0.50)
38. Underconfident model (expected_accuracy > midpoint):
    - p=0.80, shrinkage_factor=1.10 -> adjusted = 0.5 + 0.30*1.10 = 0.83 (pushed from 0.50)
39. CRITICAL: Shrinkage works correctly on both sides of 0.50:
    - p=0.20 (below 0.50), overconfident: shrinks toward 0.50 (increases)
    - p=0.80 (above 0.50), overconfident: shrinks toward 0.50 (decreases)
40. No calibration data (< 10 samples) -> probability unchanged (trust Grok as-is)
41. Adjusted probability clamped to [0.01, 0.99]

Step 4 - Market-type edge penalty:
42. Market type with avg_brier > 0.30 -> extra_edge = 0.05
43. New market type (< 15 trades) -> extra_edge = 0.0

Step 5 - Temporal decay:
44. I1 signal < 30 min old -> confidence *= 1.05 (boost)
45. All signals > 2h old -> confidence *= max(0.85, 1.0 - 0.05*(2-1)) = 0.95
46. All signals > 4h old -> confidence *= max(0.85, 1.0 - 0.05*(4-1)) = 0.85 (floor)
47. No signal timestamps -> default age used (2.0h), decay applied
```

### `tests/test_model_swap.py`

```
handle_model_swap():
48. Calibration fully reset (all buckets alpha=1, beta=1)
49. Market-type dampened (brier_scores truncated to last 15)
50. Signal trackers NOT modified (all counts preserved)
51. New experiment run created in DB
52. ModelSwapEvent saved to DB

void_trade():
53. Trade marked as voided=True with reason
54. Voided trade excluded from on_trade_resolved (returns immediately)
55. recalculate_learning_from_scratch rebuilds all three layers from non-voided trades

on_trade_resolved():
56. Brier scores calculated: raw = (grok_raw_prob - actual)^2
57. Brier scores calculated: adjusted = (final_adjusted_prob - actual)^2
58. Calibration updated with RAW correctness
59. Market-type updated with ADJUSTED correctness
60. Signal trackers updated with ADJUSTED correctness
61. All changes persisted to DB
```

### Mocking Strategy

- **In-memory SQLite** with pre-seeded learning state (calibration buckets with known alpha/beta, market-type with known Brier scores)
- **TradeRecord fixtures** with specific raw/adjusted values to test correctness routing
- **No external mocks** — learning is pure math + DB persistence

---

## TEST-10: Integration + Scheduler + CLI

**Agent ID:** `TEST-10-INTEGRATION`
**Tests For:** DEV-11 (main.py, scheduler.py) + DEV-12 (manage.py, alerts.py)
**Complexity:** HIGH

### Files to Create

```
tests/test_scheduler.py
tests/test_health.py
tests/test_manage.py
tests/test_alerts.py
```

### `tests/test_scheduler.py`

```
Tier 1 scan pipeline (end-to-end with mocks):
1. Full pipeline: markets -> keywords -> signals -> context -> Grok -> adjust -> rank -> execute
   - Mock all external services, verify pipeline calls in correct order
2. Single market produces TradeRecord saved to DB

Observe-only mode:
3. When scan_mode="observe_only": Grok is NOT called
4. Observe-only: trades recorded as SKIP with skip_reason="daily_cap_observe_only"
5. Observe-only: RSS and Twitter still called (cheap signals preserved)

Tier 2 activation:
6. should_activate_tier2 returns True with 2+ crypto signals + S1 source
7. should_activate_tier2 returns True with 2+ crypto signals + 100K follower source
8. should_activate_tier2 returns False with 1 crypto signal
9. should_activate_tier2 returns False with 2 crypto signals but all S6 and < 100K followers

Error isolation:
10. Exception during one market's Grok call -> other markets still processed
11. Exception during keyword extraction -> that market skipped, others continue
12. Twitter API timeout -> scan continues with RSS signals only

Auto-resolution scheduling:
13. auto_resolve_trades called at expected interval
14. Resolved trades trigger on_trade_resolved learning update
```

### `tests/test_health.py`

```
Health endpoint:
15. GET /health after recent scan -> 200 with status="healthy"
16. GET /health with no scan in 31 minutes -> 503 with status="degraded"
17. Response includes: mode, open_trades, today_trades, uptime_hours
18. Response includes: last_scan_completed, minutes_since_scan
```

### `tests/test_manage.py`

```
CLI commands:
19. model_swap command calls handle_model_swap with correct args
20. void_trade command sets voided=True and triggers recalculate
21. start_experiment creates new experiment run
22. end_experiment sets ended_at on experiment
```

### `tests/test_alerts.py`

```
send_alert():
23. With TELEGRAM_BOT_TOKEN set: POST request sent to Telegram API
24. With empty TELEGRAM_BOT_TOKEN: no request sent (no-op)
25. format_trade_alert produces string with market question, action, edge, size
26. format_daily_summary includes trade count, PnL, Brier score
27. format_error_alert includes error message
28. Telegram API error -> logged but no exception raised
```

### Mocking Strategy

- **Mock ALL external services:**
  - `PolymarketClient`: return canned markets, orderbooks, resolution status
  - `TwitterDataPipeline`: return canned signals
  - `RSSPipeline`: return canned signals
  - `GrokClient`: return canned parsed responses
- **In-memory SQLite** for DB
- **Mock `httpx.AsyncClient`** for Telegram alerts
- **Mock `APScheduler`** to verify job registration without actual scheduling
- **FastAPI TestClient** for health endpoint tests

---

## Summary: Test Agent → Test File Mapping

| Test Agent | Test Files | Test Count (approx) |
|------------|-----------|---------------------|
| TEST-01 | conftest.py, test_config.py, test_models.py, test_db.py | ~26 |
| TEST-02 | test_signal_classifier.py | ~26 |
| TEST-03 | test_twitter.py | ~19 |
| TEST-04 | test_rss.py | ~18 |
| TEST-05 | test_polymarket.py | ~16 |
| TEST-06 | test_grok_client.py, test_context_builder.py | ~48 |
| TEST-07 | test_trade_ranker.py, test_trade_decision.py | ~49 |
| TEST-08 | test_execution.py, test_resolution.py | ~27 |
| TEST-09 | test_calibration.py, test_market_type_learning.py, test_signal_tracker_learning.py, test_adjustment_pipeline.py, test_model_swap.py | ~61 |
| TEST-10 | test_scheduler.py, test_health.py, test_manage.py, test_alerts.py | ~28 |
| **TOTAL** | **20 test files** | **~318 test cases** |

---

## Critical Test Priorities

If time is limited, these tests catch the highest-risk bugs:

1. **TEST-09 cases 11, 17, 27, 39** — RAW vs ADJUSTED correctness routing (silent learning corruption if wrong)
2. **TEST-07 cases 26-31** — Kelly sizing formula (direct financial impact)
3. **TEST-06 cases 1-10** — JSON parsing fallbacks (Grok returns malformed JSON 2-5% of the time)
4. **TEST-09 cases 37-41** — Probability shrinkage direction (overconfident shrinks toward 0.50, not away)
5. **TEST-07 cases 34-46** — Monk Mode enforcement (risk management)
6. **TEST-09 cases 48-51** — Model swap resets correct layers (calibration resets, signals preserved)
