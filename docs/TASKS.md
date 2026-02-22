# Polymarket v2 — Developer Agent Task Specification

> **Purpose:** Each section below is a self-contained prompt for one developer agent. Agents work on isolated modules with explicit interface contracts. Hand each section to a separate Claude Code agent.
>
> **Source of truth:** `polymarket_system_v2.md` (2,326 lines) in project root.
>
> **Python version:** 3.11+ | **Async:** All I/O is async (aiohttp/httpx/aiosqlite)

---

## Execution DAG

```
Wave 1:  DEV-01
Wave 2:  DEV-02  DEV-05  DEV-06  DEV-07  DEV-09          (parallel, all need only DEV-01)
Wave 3:  DEV-03  DEV-04  DEV-08  DEV-10  DEV-12          (parallel, see per-agent deps)
Wave 4:  DEV-11                                           (needs ALL dev agents)
```

### Dependency Matrix

| Agent | Hard Dependencies | Can Start After |
|-------|------------------|-----------------|
| DEV-01 | None | Immediately |
| DEV-02 | DEV-01 | DEV-01 |
| DEV-03 | DEV-01, DEV-02 | DEV-02 |
| DEV-04 | DEV-01, DEV-02 | DEV-02 |
| DEV-05 | DEV-01 | DEV-01 |
| DEV-06 | DEV-01 | DEV-01 |
| DEV-07 | DEV-01 | DEV-01 |
| DEV-08 | DEV-01, DEV-05 | DEV-05 |
| DEV-09 | DEV-01 | DEV-01 |
| DEV-10 | DEV-01, DEV-09 | DEV-09 |
| DEV-11 | ALL DEV agents | All complete |
| DEV-12 | DEV-01, DEV-10 | DEV-10 |

---

## DEV-01: Foundation — Config, Models, Database

**Agent ID:** `DEV-01-FOUNDATION`
**Complexity:** HIGH (largest agent, 13 files, defines all contracts)
**Estimated files:** 13

### Scope

Build the absolute foundation that every other agent imports from: configuration, all data models, database wrapper, schema migrations, YAML configs, and project scaffolding.

### Files to Create

```
src/__init__.py
src/config.py
src/models.py
src/db/__init__.py
src/db/sqlite.py
src/db/migrations.py
src/pipelines/__init__.py
src/engine/__init__.py
src/learning/__init__.py
config/known_sources.yaml
config/rss_feeds.yaml
requirements.txt
.env.example
.gitignore
```

### Detailed Specifications

#### `src/config.py`
Pydantic Settings class reading from `.env`:
```python
class Settings(BaseSettings):
    # API Keys
    XAI_API_KEY: str
    TWITTER_API_KEY: str
    POLYMARKET_API_KEY: str = ""
    POLYMARKET_SECRET: str = ""
    POLYMARKET_PASSPHRASE: str = ""
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""

    # Environment
    ENVIRONMENT: str = "paper"  # "paper" or "live"
    DB_PATH: str = "data/predictor.db"
    LOG_LEVEL: str = "INFO"

    # Tier 1 Config
    TIER1_SCAN_INTERVAL_MINUTES: int = 15
    TIER1_MIN_EDGE: float = 0.04
    TIER1_DAILY_CAP: int = 5
    TIER1_FEE_RATE: float = 0.02

    # Tier 2 Config
    TIER2_SCAN_INTERVAL_MINUTES: int = 3
    TIER2_MIN_EDGE: float = 0.05
    TIER2_DAILY_CAP: int = 3
    TIER2_FEE_RATE: float = 0.04

    # Monk Mode
    DAILY_LOSS_LIMIT_PCT: float = 0.05
    WEEKLY_LOSS_LIMIT_PCT: float = 0.10
    CONSECUTIVE_LOSS_COOLDOWN: int = 3
    COOLDOWN_DURATION_HOURS: float = 2.0
    DAILY_API_BUDGET_USD: float = 8.0
    MAX_POSITION_PCT: float = 0.08
    MAX_TOTAL_EXPOSURE_PCT: float = 0.30
    KELLY_FRACTION: float = 0.25
    MAX_CLUSTER_EXPOSURE_PCT: float = 0.12

    # Initial Bankroll
    INITIAL_BANKROLL: float = 5000.0

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
```

Also export a `MonkModeConfig` dataclass built from Settings (spec Section 9.1, lines 1106-1118).

#### `src/models.py`
All dataclasses. Each must match the spec exactly:

1. **`Signal`** — source, source_tier (S1-S6), info_type (I1-I6 or None), content, credibility, author, followers, engagement, timestamp, headline_only (bool)
2. **`Market`** — market_id, question, yes_price, no_price, resolution_time, hours_to_resolution, volume_24h, liquidity, market_type, fee_rate, keywords (List[str])
3. **`OrderBook`** — market_id, bids (List[float]), asks (List[float]), timestamp
4. **`TradeCandidate`** — extends with: adjusted_probability, adjusted_confidence, calculated_edge, score, position_size, side (BUY_YES/BUY_NO), skip_reason, market_cluster_id, resolution_hours
5. **`ExecutionResult`** — executed_price, slippage, fill_probability, filled (bool)
6. **`Position`** — market_id, side, entry_price, size_usd, current_value, market_cluster_id
7. **`TradeRecord`** — 35+ fields (spec Section 7.1, lines 749-792). Includes both `brier_score_raw` and `brier_score_adjusted`, `unrealized_adverse_move`, `voided`, `void_reason`
8. **`CalibrationBucket`** — bucket_range (Tuple[float,float]), alpha, beta. Properties: `expected_accuracy`, `sample_count`, `uncertainty` (scipy beta ppf), `get_correction()` (10-sample min, certainty-weighted). Method: `update(was_correct, recency_weight)`. (Spec lines 803-845)
9. **`MarketTypePerformance`** — market_type, total_trades, total_pnl, brier_scores (List[float]), total_observed, counterfactual_pnl. Properties: `avg_brier` (0.95 exponential decay), `edge_adjustment` (thresholds at 15+ trades: >0.30=0.05, >0.25=0.03, >0.20=0.01), `should_disable` (30+ trades, negative PnL). (Spec lines 852-880)
10. **`SignalTracker`** — source_tier, info_type, market_type, present_winning/losing, absent_winning/losing. Properties: `lift` (5-sample min), `weight` (clamped [0.8, 1.2]). (Spec lines 705-731)
11. **`ExperimentRun`** — run_id, started_at, ended_at, config_snapshot, description, model_used, include_in_learning, total_trades, total_pnl, avg_brier, sharpe_ratio
12. **`ModelSwapEvent`** — timestamp, old_model, new_model, reason, experiment_run_started
13. **`Portfolio`** — cash_balance, total_equity, total_pnl, peak_equity, max_drawdown, open_positions (List[Position])

Constants to export:
```python
SOURCE_TIER_CREDIBILITY = {"S1": 0.95, "S2": 0.90, "S3": 0.80, "S4": 0.65, "S5": 0.70, "S6": 0.30}

CALIBRATION_BUCKET_RANGES = [
    (0.50, 0.60), (0.60, 0.70), (0.70, 0.80),
    (0.80, 0.90), (0.90, 0.95), (0.95, 1.00),
]
```

#### `src/db/sqlite.py`
Async SQLite wrapper using `aiosqlite`:
```python
class Database:
    async def init(db_path: str) -> "Database"
    async def close()

    # Trade Records
    async def save_trade(record: TradeRecord)
    async def get_trade(record_id: str) -> Optional[TradeRecord]
    async def get_open_trades() -> List[TradeRecord]  # unresolved, non-voided
    async def get_today_trades() -> List[TradeRecord]
    async def get_week_trades() -> List[TradeRecord]
    async def update_trade(record: TradeRecord)
    async def count_today_trades() -> int
    async def count_open_trades() -> int

    # Calibration
    async def load_calibration() -> List[CalibrationBucket]
    async def save_calibration(buckets: List[CalibrationBucket])

    # Market Type Performance
    async def load_market_type_performance() -> Dict[str, MarketTypePerformance]
    async def save_market_type_performance(perfs: Dict[str, MarketTypePerformance])

    # Signal Trackers
    async def load_signal_trackers() -> Dict[Tuple[str,str,str], SignalTracker]
    async def save_signal_trackers(trackers: Dict[Tuple[str,str,str], SignalTracker])

    # Experiment Runs
    async def save_experiment(run: ExperimentRun)
    async def get_current_experiment() -> Optional[ExperimentRun]
    async def end_experiment(run_id: str, stats: dict)

    # Portfolio
    async def load_portfolio() -> Portfolio
    async def save_portfolio(portfolio: Portfolio)

    # API Costs
    async def increment_api_cost(service: str, tokens_in: int = 0, tokens_out: int = 0)
    async def get_today_api_spend() -> float

    # Parse Failures
    async def record_parse_failure(market_id: str)

    # Model Swaps
    async def save_model_swap(event: ModelSwapEvent)
```

#### `src/db/migrations.py`
Full DDL from spec Section 13 (lines 1752-1886). Schema versioning via a `schema_version` table. `run_migrations()` applies pending migrations idempotently.

All 9 tables: `experiment_runs`, `model_swaps`, `trade_records`, `calibration_state`, `market_type_performance`, `signal_trackers`, `portfolio`, `api_costs`, `daily_mode_log`.

All 7 indexes including partial indexes on `actual_outcome IS NULL` and `headline_only_signal = TRUE`.

#### `config/known_sources.yaml`
Source tier lists from spec Section 6.6 (lines 646-697):
```yaml
official_sources:
  twitter: ["@WhiteHouse", "@FederalReserve", "@SECGov", "@SecYellen", "@USTreasury"]
  rss_domains: ["federalreserve.gov", "sec.gov", "bls.gov", "whitehouse.gov", "supremecourt.gov", "treasury.gov"]

wire_services:
  twitter: ["@Reuters", "@AP", "@AFP", "@BNONews", "@business"]
  rss_domains: ["reuters.com", "apnews.com", "afp.com"]

institutional_media:
  twitter: ["@BBCBreaking", "@CNN", "@nytimes", "@WSJ", "@CNBC", "@CoinDesk", "@TheBlock__"]
  rss_domains: ["bbc.com", "nytimes.com", "wsj.com", "cnbc.com", "coindesk.com", "theblock.co"]

expert_bio_keywords:
  - journalist
  - reporter
  - editor
  - correspondent
  - analyst
  - researcher
  - professor
  - economist
  - senator
  - representative
  - minister
  - official
  - ceo
  - cto
  - founder
  - partner
  - director
  - crypto
  - blockchain
  - defi
```

#### `config/rss_feeds.yaml`
From spec Section 11.3 (lines 1511-1517):
```yaml
feeds:
  reuters_top:
    url: "https://feeds.reuters.com/reuters/topNews"
    domain: "reuters.com"
  reuters_business:
    url: "https://feeds.reuters.com/reuters/businessNews"
    domain: "reuters.com"
  ap_top:
    url: "https://rsshub.app/apnews/topics/apf-topnews"
    domain: "apnews.com"
  bbc_world:
    url: "http://feeds.bbci.co.uk/news/world/rss.xml"
    domain: "bbc.com"
  coindesk:
    url: "https://www.coindesk.com/arc/outboundfeeds/rss/"
    domain: "coindesk.com"
```

#### `requirements.txt`
From spec Section 1.5 (lines 116-143).

#### `.env.example`
Template with all Settings fields, values blank.

#### `.gitignore`
Standard Python + `.env`, `data/`, `__pycache__/`, etc.

### Output Interfaces (consumed by all other agents)
```python
from src.config import Settings, get_settings
from src.models import (Signal, Market, OrderBook, TradeCandidate, ExecutionResult,
                         Position, TradeRecord, CalibrationBucket, MarketTypePerformance,
                         SignalTracker, ExperimentRun, ModelSwapEvent, Portfolio,
                         SOURCE_TIER_CREDIBILITY, CALIBRATION_BUCKET_RANGES)
from src.db.sqlite import Database
from src.db.migrations import run_migrations
```

---

## DEV-02: Signal Classifier

**Agent ID:** `DEV-02-SIGNAL-CLASSIFIER`
**Complexity:** LOW
**Dependencies:** DEV-01

### Scope

Programmatic source tier classification (S1-S6). Pure deterministic logic, no external API calls.

### Files to Create

```
src/pipelines/signal_classifier.py
```

### Detailed Specification

Implement `classify_source_tier(signal: dict) -> str` exactly as specified in lines 644-698:

1. If `signal["source_type"] == "market_data"` -> return `"S5"`
2. If RSS: match `signal["domain"]` against known_sources.yaml tiers
3. If Twitter: match `signal["account_handle"]` against known_sources.yaml, then check S4 criteria (verified + 50K+ followers + expert bio keyword)
4. Default: `"S6"`

Load source lists from `config/known_sources.yaml` at module level (not per-call).

Also export `SOURCE_TIER_CREDIBILITY` dict (or re-export from models).

### Interface Contract

```python
def classify_source_tier(signal: dict) -> str
# signal keys: source_type ("twitter"|"rss"|"market_data"), domain?, account_handle?,
#              is_verified?, follower_count?, bio?
# Returns: "S1" | "S2" | "S3" | "S4" | "S5" | "S6"
```

---

## DEV-03: Twitter Pipeline

**Agent ID:** `DEV-03-TWITTER`
**Complexity:** MEDIUM
**Dependencies:** DEV-01, DEV-02

### Scope

TwitterAPI.io client for fetching social signals per market.

### Files to Create

```
src/pipelines/twitter.py
```

### Detailed Specification

Implement `TwitterDataPipeline` class (spec Section 11.2, lines 1457-1492):

- **Base URL:** `https://api.twitterapi.io/twitter`
- **`get_signals_for_market(keywords, max_tweets=50) -> List[Signal]`:**
  1. Search tweets with `" OR ".join(keywords)`, recency 2h, max_results
  2. Pre-filter: `followers_count >= 1000`, `engagement_score >= 10`, not bot
  3. Deduplicate by content similarity
  4. Classify source tier via `classify_source_tier()`
  5. Sort by credibility descending
  6. Return top 10 as `Signal` objects (info_type=None, assigned later by Grok)
- **`_is_bot_account(author) -> bool`:** Heuristic bot detection (e.g., account age < 30 days, username matches pattern, no profile picture)
- **`_deduplicate_by_content_similarity(tweets) -> List`:** Simple dedup (e.g., 80%+ word overlap = duplicate)
- **Error handling:** Timeout returns empty list, rate limit waits and retries, log all failures

Use `httpx.AsyncClient` for HTTP calls. API key from `Settings.TWITTER_API_KEY`.

### Interface Contract

```python
class TwitterDataPipeline:
    def __init__(self, settings: Settings): ...
    async def get_signals_for_market(self, keywords: List[str], max_tweets: int = 50) -> List[Signal]: ...
```

---

## DEV-04: RSS Pipeline

**Agent ID:** `DEV-04-RSS`
**Complexity:** LOW-MEDIUM
**Dependencies:** DEV-01, DEV-02

### Scope

RSS feed parsing with bounded headline cache and breaking news detection.

### Files to Create

```
src/pipelines/rss.py
```

### Detailed Specification

Implement `RSSPipeline` class (spec Section 11.3, lines 1510-1557):

- **`seen_headlines: Dict[str, datetime]`:** Bounded dict with 24h TTL (spec explicitly calls out memory leak prevention)
- **`_prune_old_headlines()`:** Remove entries older than 24h, called at start of each `get_breaking_news()`
- **`get_breaking_news() -> List[Signal]`:**
  1. Load feed configs from `config/rss_feeds.yaml`
  2. Parse each feed with `feedparser.parse()`
  3. Process top 10 entries per feed
  4. Skip already-seen headlines
  5. Skip entries older than 2 hours
  6. Classify source tier via `classify_source_tier()`
  7. Return all new signals with `headline_only=True`
- **Error handling:** Feed parse failure logs warning and continues to next feed (no crash)
- **Date parsing:** Handle multiple published date formats, treat missing as "now"

### Interface Contract

```python
class RSSPipeline:
    def __init__(self): ...
    async def get_breaking_news(self) -> List[Signal]: ...
```

---

## DEV-05: Polymarket Client

**Agent ID:** `DEV-05-POLYMARKET`
**Complexity:** MEDIUM-HIGH
**Dependencies:** DEV-01

### Scope

Polymarket Gamma API (market discovery) + CLOB API (orderbook, trading, resolution).

### Files to Create

```
src/pipelines/polymarket.py
```

### Detailed Specification

Implement `PolymarketClient` class:

- **Market Discovery (Gamma API):**
  - `get_active_markets(tier: int) -> List[Market]`
  - Tier 1 filters: resolution 1-24h, liquidity > $5K
  - Tier 2 filters: crypto markets, 15-min resolution
  - Map API response to `Market` dataclass including: market_id, question, yes_price, no_price, resolution_time, hours_to_resolution, volume_24h, liquidity, market_type, fee_rate
  - Derive `market_type` from market category/tags (political, economic, crypto_15m, sports, cultural, regulatory)

- **Orderbook (CLOB API):**
  - `get_orderbook(market_id: str) -> OrderBook`
  - Return top 5 bid/ask levels

- **Resolution:**
  - `get_market(market_id: str) -> Market` (includes resolution status)
  - Check if market has resolved and what the outcome was

- **Trading (CLOB API, live mode only):**
  - `place_order(market_id, side, price, size) -> dict`
  - Use `py-clob-client` SDK for order signing
  - Only called when `ENVIRONMENT=live`

- **Error handling:** 429 rate limits with exponential backoff, 500 errors with retry, timeouts

Use `httpx.AsyncClient` for Gamma API, `py-clob-client` for CLOB operations.

### Interface Contract

```python
class PolymarketClient:
    def __init__(self, settings: Settings): ...
    async def get_active_markets(self, tier: int) -> List[Market]: ...
    async def get_orderbook(self, market_id: str) -> OrderBook: ...
    async def get_market(self, market_id: str) -> Market: ...
    async def place_order(self, market_id: str, side: str, price: float, size: float) -> dict: ...
```

---

## DEV-06: Grok Client + Context Builder

**Agent ID:** `DEV-06-GROK-CONTEXT`
**Complexity:** HIGH
**Dependencies:** DEV-01

### Scope

Grok API wrapper with retry/JSON-parsing fallback, keyword extraction (regex + LLM), context prompt construction.

### Files to Create

```
src/engine/grok_client.py
src/pipelines/context_builder.py
```

### Detailed Specification

#### `src/engine/grok_client.py`

Implement the Grok API wrapper (spec Section 11.5, lines 1604-1688):

- **`GrokClient` class:**
  - `complete(prompt: str, max_tokens: int = 500) -> str` — Raw API call to xAI
  - `call_grok_with_retry(context: str, market_id: str) -> Optional[dict]` — Full retry pipeline

- **Retry logic (lines 1612-1660):**
  - `MAX_RETRIES = 2` (total 3 attempts)
  - On parse failure: log warning, continue to next attempt
  - On missing required fields: log warning, continue
  - On invalid probability/confidence (not in [0,1]): log warning, continue
  - Type coercion: string "0.75" -> float 0.75
  - Linear backoff between retries: `1.0 * (attempt + 1)` seconds
  - All retries exhausted: log error, call `db.record_parse_failure()`, return None
  - On success: call `db.increment_api_cost("grok", ...)`

- **`parse_json_safe(raw: str) -> Optional[dict]` (lines 1663-1688):**
  1. Direct `json.loads(raw.strip())`
  2. Strip markdown fences: `` ```json ... ``` `` or `` ``` ... ``` ``
  3. Find first `{...}` block via regex `r'\{.*\}'` with `re.DOTALL`
  4. Return None if all fail

- **`REQUIRED_FIELDS = {"estimated_probability", "confidence", "reasoning", "signal_info_types"}`**

#### `src/pipelines/context_builder.py`

- **`extract_keywords(market_id, market_question, market_type) -> List[str]` (lines 1388-1450):**
  1. Check `_keyword_cache` (Dict[str, List[str]])
  2. Regex extraction using `ENTITY_PATTERNS` (named entities, acronyms, tickers)
  3. Add `KEYWORD_SUPPLEMENTS` based on market_type
  4. If 2+ entities found -> use them (no LLM cost)
  5. Else -> LLM fallback via `grok_extract_keywords()` (costs ~$0.0002, cached per market)

- **`build_grok_context(market, twitter_signals, rss_signals, orderbook) -> str` (lines 1562-1598):**
  - Merge and sort all signals by credibility, take top 7
  - Format: market question, current prices, resolution time, volume, liquidity, orderbook depth/skew
  - Include signal list with source tier and credibility
  - Include info-type classification instructions (I1-I5)
  - Request JSON response format with specific keys

### Interface Contract

```python
class GrokClient:
    def __init__(self, settings: Settings, db: Database): ...
    async def call_grok_with_retry(self, context: str, market_id: str) -> Optional[dict]: ...

def build_grok_context(market: Market, twitter_signals: List[Signal],
                       rss_signals: List[Signal], orderbook: OrderBook) -> str: ...

def extract_keywords(market_id: str, market_question: str, market_type: str) -> List[str]: ...
# Note: extract_keywords may call grok internally for fallback. Pass GrokClient or make it async.
```

---

## DEV-07: Trade Ranker + Decision Engine

**Agent ID:** `DEV-07-TRADE-ENGINE`
**Complexity:** HIGH
**Dependencies:** DEV-01

### Scope

Trade scoring, ranking, correlated market detection, edge calculation, Kelly sizing, Monk Mode enforcement.

### Files to Create

```
src/engine/trade_ranker.py
src/engine/trade_decision.py
```

### Detailed Specification

#### `src/engine/trade_ranker.py` (spec Section 10.1-10.2, lines 1206-1321)

- **`select_best_trades(candidates, remaining_cap, open_positions, bankroll) -> Tuple[List[TradeCandidate], List[TradeCandidate]]`:**
  1. Score each candidate: `score = edge * adjusted_confidence * (1.0 / max(resolution_hours, 0.5))`
  2. Sort by score descending
  3. Detect market clusters
  4. Iterate ranked list: add to `to_execute` if within cap and cluster limit, else `to_skip` with reason

- **`detect_market_clusters(candidates) -> Dict[str, str]`:**
  - Group by market_type (category)
  - Within same category: sort by resolution_time
  - Markets within 1h resolution window + 50% keyword Jaccard overlap = same cluster

- **`_keyword_overlap(kw1, kw2) -> float`:** Jaccard similarity of lowered keyword sets

- **`check_cluster_exposure(candidate, cluster_id, open_positions, pending, clusters, bankroll) -> bool`:**
  - Sum existing + pending exposure for same cluster
  - Return `total <= MAX_CLUSTER_EXPOSURE_PCT * bankroll`

#### `src/engine/trade_decision.py` (spec Sections 9-10, lines 1101-1375)

- **`calculate_edge(adjusted_prob, market_price, fee_rate) -> float`:**
  - `abs(adjusted_prob - market_price) - fee_rate`

- **`kelly_size(adjusted_prob, market_price, side, bankroll, kelly_fraction=0.25, max_position_pct=0.08) -> float` (lines 1337-1375):**
  - BUY_YES: `f* = (prob - price) / (1 - price)`, return 0 if prob <= price
  - BUY_NO: `f* = (price - prob) / price`, return 0 if prob >= price
  - Apply quarter Kelly, cap at max_position_pct * bankroll

- **`determine_side(adjusted_prob, market_price) -> str`:**
  - If adjusted_prob > market_price: "BUY_YES"
  - If adjusted_prob < market_price: "BUY_NO"
  - Else: "SKIP"

- **`check_monk_mode(config, trade_signal, portfolio, today_trades, week_trades, api_spend) -> Tuple[bool, Optional[str]]` (lines 1123-1172):**
  - Check in order: tier daily cap, daily loss limit (-5%), weekly loss limit (-10%), consecutive adverse (3 losses including unrealized adverse moves >10%), max total exposure (30%), API budget ($8/day)

- **`get_scan_mode(today_trades, config) -> str` (lines 1179-1184):**
  - Return "observe_only" if tier1 executed trades >= cap, else "active"

### Interface Contract

```python
# trade_ranker.py
def select_best_trades(candidates: List[TradeCandidate], remaining_cap: int,
                       open_positions: List[Position], bankroll: float
                       ) -> Tuple[List[TradeCandidate], List[TradeCandidate]]: ...
def detect_market_clusters(candidates: List[TradeCandidate]) -> Dict[str, str]: ...

# trade_decision.py
def calculate_edge(adjusted_prob: float, market_price: float, fee_rate: float) -> float: ...
def kelly_size(adjusted_prob: float, market_price: float, side: str,
               bankroll: float, kelly_fraction: float = 0.25,
               max_position_pct: float = 0.08) -> float: ...
def determine_side(adjusted_prob: float, market_price: float) -> str: ...
def check_monk_mode(config: MonkModeConfig, trade_signal, portfolio: Portfolio,
                    today_trades: List[TradeRecord], week_trades: List[TradeRecord],
                    api_spend: float) -> Tuple[bool, Optional[str]]: ...
def get_scan_mode(today_trades: List[TradeRecord], config: MonkModeConfig) -> str: ...
```

---

## DEV-08: Execution + Resolution

**Agent ID:** `DEV-08-EXECUTION`
**Complexity:** MEDIUM
**Dependencies:** DEV-01, DEV-05

### Scope

Paper trading simulation with realistic slippage, live order execution, auto-resolution loop, unrealized adverse move tracking.

### Files to Create

```
src/engine/execution.py
src/engine/resolution.py
```

### Detailed Specification

#### `src/engine/execution.py` (spec Section 12.1, lines 1698-1717)

- **`simulate_execution(side, price, size_usd, execution_type, orderbook_depth) -> ExecutionResult`:**
  - Taker: `slippage = 0.005 + 0.01 * min(size_usd / max(orderbook_depth, 1), 1.0)`, YES price += slippage, NO price -= slippage, fill_probability = 1.0
  - Maker: `fill_probability = 0.4 + 0.4 * (1 - abs(price - 0.5))`, slippage = 0, executed_price = price
  - Clamp `executed_price` to [0.01, 0.99]
  - `filled = random.random() < fill_probability`

- **`execute_trade(candidate, portfolio, db, polymarket_client, environment) -> Optional[TradeRecord]`:**
  - If paper: call `simulate_execution()`
  - If live: call `polymarket_client.place_order()`
  - If not filled (maker): return None (no trade record for unfilled orders)
  - Update portfolio: deduct cash, add position
  - Create and save TradeRecord

#### `src/engine/resolution.py` (spec Section 12.2, lines 1722-1734)

- **`auto_resolve_trades(db, polymarket_client)`:**
  - Get all open trades from DB
  - For each: check if market resolved via `polymarket_client.get_market()`
  - For event markets: `market.resolved == True` -> resolve with outcome
  - For crypto_15m: check if past expected resolution time -> resolve against strike
  - On resolution: calculate PnL, set `brier_score_raw` and `brier_score_adjusted`, update portfolio

- **`calculate_hypothetical_pnl(record) -> float`:**
  - For skipped trades: what would PnL be if trade was taken at the recorded price/size

- **`update_unrealized_adverse_moves(db, polymarket_client)`:**
  - For open trades: check current market price vs entry
  - If moved >10% against position, update `unrealized_adverse_move` field (used by Monk Mode cooldown)

### Interface Contract

```python
# execution.py
def simulate_execution(side: str, price: float, size_usd: float,
                       execution_type: str, orderbook_depth: float) -> ExecutionResult: ...
async def execute_trade(candidate: TradeCandidate, portfolio: Portfolio,
                        db: Database, polymarket_client, environment: str
                        ) -> Optional[TradeRecord]: ...

# resolution.py
async def auto_resolve_trades(db: Database, polymarket_client) -> None: ...
def calculate_hypothetical_pnl(record: TradeRecord) -> float: ...
async def update_unrealized_adverse_moves(db: Database, polymarket_client) -> None: ...
```

---

## DEV-09: Learning Core — Calibration, Market-Type, Signal Tracker

**Agent ID:** `DEV-09-LEARNING-CORE`
**Complexity:** HIGH
**Dependencies:** DEV-01

### Scope

All three learning layers with their persistence logic. This is the statistical core of the system.

### Files to Create

```
src/learning/calibration.py
src/learning/market_type.py
src/learning/signal_tracker.py
```

### Detailed Specification

#### `src/learning/calibration.py` (spec Section 7.2, lines 796-845)

**`CalibrationManager` class:**
- Owns the 6 `CalibrationBucket` instances
- `find_bucket(confidence: float) -> CalibrationBucket` — match confidence to bucket range
- `get_correction(confidence: float) -> float` — delegate to bucket's `get_correction()`
- `update_calibration(record: TradeRecord)`:
  - **CRITICAL: Use RAW probability for feedback, NOT adjusted** (spec lines 969-982)
  - `raw_predicted_yes = record.grok_raw_probability > 0.5`
  - `was_correct = raw_predicted_yes == record.actual_outcome`
  - `recency = 0.95 ** days_since(record.timestamp)`
  - `bucket.update(was_correct, recency_weight=recency)`
- `reset_to_priors()` — set all buckets to alpha=1.0, beta=1.0
- `load(db) / save(db)` — persist to/from `calibration_state` table

#### `src/learning/market_type.py` (spec Section 7.3, lines 847-880)

**`MarketTypeManager` class:**
- Dict of `MarketTypePerformance` objects keyed by market_type
- `update_market_type(record: TradeRecord)`:
  - Uses ADJUSTED Brier score (not raw)
  - `mtype.total_trades += 1`
  - `mtype.brier_scores.append(record.brier_score_adjusted)`
  - If not skipped: `mtype.total_pnl += record.pnl`
  - If skipped: `mtype.total_observed += 1`, `mtype.counterfactual_pnl += hypothetical_pnl`
- `get_edge_adjustment(market_type) -> float` — delegate to performance's `edge_adjustment` property
- `should_disable(market_type) -> bool`
- `dampen_on_swap()` — keep only last 15 Brier scores per market type
- `load(db) / save(db)` — persist to/from `market_type_performance` table

#### `src/learning/signal_tracker.py` (spec Section 6.7, lines 704-731; Section 7.6, lines 991-1001)

**`SignalTrackerManager` class:**
- Dict of `SignalTracker` objects keyed by (source_tier, info_type, market_type) tuple
- `update_signal_trackers(record: TradeRecord)`:
  - Uses ADJUSTED correctness (not raw)
  - For all observed (source_tier, info_type) combos for this market_type:
    - If combo was present in trade's signals AND trade was correct: increment `present_winning`
    - Present + incorrect: `present_losing`
    - Absent + correct: `absent_winning`
    - Absent + incorrect: `absent_losing`
- `get_signal_weight(source_tier, info_type, market_type) -> float` — delegate to tracker's `weight`
- `load(db) / save(db)` — persist to/from `signal_trackers` table

### Interface Contract

```python
class CalibrationManager:
    def __init__(self): ...
    def find_bucket(self, confidence: float) -> CalibrationBucket: ...
    def get_correction(self, confidence: float) -> float: ...
    def update_calibration(self, record: TradeRecord) -> None: ...
    def reset_to_priors(self) -> None: ...
    async def load(self, db: Database) -> None: ...
    async def save(self, db: Database) -> None: ...

class MarketTypeManager:
    def __init__(self): ...
    def update_market_type(self, record: TradeRecord) -> None: ...
    def get_edge_adjustment(self, market_type: str) -> float: ...
    def should_disable(self, market_type: str) -> bool: ...
    def dampen_on_swap(self) -> None: ...
    async def load(self, db: Database) -> None: ...
    async def save(self, db: Database) -> None: ...

class SignalTrackerManager:
    def __init__(self): ...
    def update_signal_trackers(self, record: TradeRecord) -> None: ...
    def get_signal_weight(self, source_tier: str, info_type: str, market_type: str) -> float: ...
    async def load(self, db: Database) -> None: ...
    async def save(self, db: Database) -> None: ...
```

---

## DEV-10: Learning Pipeline + Experiments + Model Swap

**Agent ID:** `DEV-10-LEARNING-PIPELINE`
**Complexity:** MEDIUM-HIGH
**Dependencies:** DEV-01, DEV-09

### Scope

Combined adjustment pipeline (5-step), experiment management, model swap protocol, void mechanism, the `on_trade_resolved` handler.

### Files to Create

```
src/learning/adjustment.py
src/learning/experiments.py
src/learning/model_swap.py
```

### Detailed Specification

#### `src/learning/adjustment.py` (spec Section 7.5, lines 888-956)

**`adjust_prediction()` — the 5-step pipeline in exact order:**

1. **Bayesian calibration** (confidence): `adjusted_confidence = confidence + calibration_correction` (clamped [0.50, 0.99])
2. **Signal-type weighting** (confidence): Average weights for all signal tags, adjust confidence by `(avg_weight - 1.0) * 0.1`
3. **Probability shrinkage**: `shrinkage_factor = expected_accuracy / bucket_midpoint`, `adjusted_probability = 0.5 + (grok_probability - 0.5) * shrinkage_factor` (only if 10+ samples). Clamped [0.01, 0.99]
4. **Market-type edge penalty**: Return `extra_edge` from market-type performance
5. **Temporal confidence decay**:
   - I1 signal <30min old: boost `* 1.05`
   - All signals >1h old: decay `max(0.85, 1.0 - 0.05 * (age_hours - 1.0))`

**`on_trade_resolved(record: TradeRecord, calibration_mgr, market_type_mgr, signal_tracker_mgr, db)`:**
- If voided: return immediately
- Calculate `brier_score_raw = (grok_raw_probability - actual)^2`
- Calculate `brier_score_adjusted = (final_adjusted_probability - actual)^2`
- Call `calibration_mgr.update_calibration(record)` (uses RAW)
- Call `market_type_mgr.update_market_type(record)` (uses ADJUSTED)
- Call `signal_tracker_mgr.update_signal_trackers(record)` (uses ADJUSTED)
- Persist all to DB

#### `src/learning/experiments.py` (spec Section 7.7, lines 1011-1027)

- `start_experiment(run_id, description, config, model, db)`
- `end_experiment(run_id, stats, db)`
- `get_current_experiment(db) -> Optional[ExperimentRun]`

#### `src/learning/model_swap.py` (spec Section 8.3, lines 1058-1080)

**`handle_model_swap(old_model, new_model, reason, calibration_mgr, market_type_mgr, db)`:**
1. Save ModelSwapEvent to DB
2. Start new experiment
3. **RESET** calibration: `calibration_mgr.reset_to_priors()`
4. **DAMPEN** market-type: `market_type_mgr.dampen_on_swap()`
5. **PRESERVE** signal trackers (no action needed)

**`void_trade(trade_id, reason, db, calibration_mgr, market_type_mgr, signal_tracker_mgr)`:**
1. Set `record.voided = True`, `record.void_reason = reason`
2. Call `recalculate_learning_from_scratch()` — reload all non-voided resolved trades, rebuild all three learning layers

### Interface Contract

```python
# adjustment.py
def adjust_prediction(grok_probability: float, grok_confidence: float,
                      market_type: str, signal_tags: List[dict],
                      calibration_mgr: CalibrationManager,
                      market_type_mgr: MarketTypeManager,
                      signal_tracker_mgr: SignalTrackerManager
                      ) -> Tuple[float, float, float]:
    """Returns (adjusted_probability, adjusted_confidence, extra_edge)"""

async def on_trade_resolved(record: TradeRecord, calibration_mgr, market_type_mgr,
                            signal_tracker_mgr, db: Database) -> None: ...

# experiments.py
async def start_experiment(run_id, description, config, model, db) -> None: ...
async def end_experiment(run_id, stats, db) -> None: ...
async def get_current_experiment(db) -> Optional[ExperimentRun]: ...

# model_swap.py
async def handle_model_swap(old_model, new_model, reason,
                            calibration_mgr, market_type_mgr, db) -> None: ...
async def void_trade(trade_id, reason, db, calibration_mgr,
                     market_type_mgr, signal_tracker_mgr) -> None: ...
```

---

## DEV-11: Scheduler + Main Entry Point

**Agent ID:** `DEV-11-ORCHESTRATION`
**Complexity:** HIGH
**Dependencies:** ALL DEV agents

### Scope

Main FastAPI app, APScheduler setup, the full scan loop for both tiers, observe-only mode, health endpoint, structlog configuration.

### Files to Create

```
src/main.py
src/scheduler.py
```

### Detailed Specification

#### `src/main.py`

- FastAPI app creation with lifespan (startup/shutdown)
- **Startup:** Init DB, run migrations, load learning state, init all clients (Twitter, RSS, Polymarket, Grok), start scheduler
- **Shutdown:** Close DB, stop scheduler
- **Structlog configuration** (spec Section 15.1, lines 1973-1996): ISO timestamps, JSON renderer, log levels: INFO for scan cycles/decisions, WARNING for parse failures/retries, ERROR for API failures
- **Health endpoint** (spec Section 15.2, lines 2009-2039): `GET /health`, 200 if last scan <30min ago, 503 if stale. Returns: status, last_scan_completed, minutes_since_scan, mode, open_trades, today_trades, uptime_hours

#### `src/scheduler.py`

- **APScheduler jobs:**
  - Tier 1 scan: every 15 min
  - Auto-resolution: every 5 min
  - Unrealized adverse move update: every 10 min

- **`run_tier1_scan()` — the full pipeline:**
  1. Check scan mode (active vs observe_only)
  2. Get active markets from Polymarket
  3. For each market:
     a. Extract keywords
     b. Fetch Twitter signals + RSS signals
     c. If observe_only: record as SKIP, continue
     d. Build Grok context
     e. Call Grok with retry
     f. If Grok fails: skip market
     g. Adjust prediction (5-step pipeline)
     h. Calculate edge, determine side
     i. If edge below threshold: record as SKIP
     j. Create TradeCandidate
  4. Rank all candidates
  5. For each to_execute: check Monk Mode, execute trade
  6. Record all trades (executed + skipped)
  7. Update last_scan_completed timestamp
  8. Log cycle summary

- **`should_activate_tier2(signals) -> bool`** (spec Section 5.4, lines 574-587):
  - Needs 2+ crypto-relevant signals
  - At least one from S1/S2 or 100K+ followers

- **`run_tier2_scan()`:** Similar to Tier 1 but:
  - Only crypto markets
  - Maker order execution
  - Deactivates when no new crypto signals for 30 min

- **Error isolation:** Single market failure must not crash the scan cycle. Wrap each market in try/except, log error, continue.

### Interface Contract

```python
# main.py
app: FastAPI  # The application instance

# scheduler.py
class Scheduler:
    def __init__(self, settings, db, polymarket, twitter, rss, grok,
                 calibration_mgr, market_type_mgr, signal_tracker_mgr): ...
    async def run_tier1_scan(self) -> None: ...
    async def run_tier2_scan(self) -> None: ...
    def should_activate_tier2(self, signals: List[Signal]) -> bool: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
```

---

## DEV-12: CLI + Alerts

**Agent ID:** `DEV-12-CLI-DASHBOARD`
**Complexity:** LOW-MEDIUM
**Dependencies:** DEV-01, DEV-10

### Scope

Management CLI for operations (model swap, void, experiments), Telegram alerting.

### Files to Create

```
src/manage.py
src/alerts.py
```

### Detailed Specification

#### `src/manage.py`

CLI tool using `argparse` or `click`:

```bash
python -m src.manage model_swap --old-model "grok-4-1-fast-reasoning" --new-model "grok-5-fast" --reason "..."
python -m src.manage void_trade --trade-id "uuid" --reason "..."
python -m src.manage start_experiment --description "..." --model "grok-4-1-fast-reasoning"
python -m src.manage end_experiment --run-id "..."
python -m src.manage recalculate_learning
```

Each command initializes DB, loads learning state, performs operation, saves, and exits.

#### `src/alerts.py` (spec Section 16.4, lines 2152-2167)

```python
async def send_alert(message: str, settings: Settings) -> None:
    """POST to Telegram. No-op if TELEGRAM_BOT_TOKEN is empty."""
```

Alert helper functions:
- `format_trade_alert(record: TradeRecord) -> str`
- `format_daily_summary(trades: List[TradeRecord], portfolio: Portfolio) -> str`
- `format_error_alert(error: str) -> str`

### Interface Contract

```python
# manage.py — CLI entry point, no programmatic interface needed

# alerts.py
async def send_alert(message: str, settings: Settings) -> None: ...
def format_trade_alert(record: TradeRecord) -> str: ...
def format_daily_summary(trades: List[TradeRecord], portfolio: Portfolio) -> str: ...
def format_error_alert(error: str) -> str: ...
```

---

## Summary: Agent → File Mapping

| Agent | Files | Count |
|-------|-------|-------|
| DEV-01 | config.py, models.py, db/sqlite.py, db/migrations.py, YAMLs, requirements.txt, .env.example, .gitignore, __init__.py files | 13 |
| DEV-02 | pipelines/signal_classifier.py | 1 |
| DEV-03 | pipelines/twitter.py | 1 |
| DEV-04 | pipelines/rss.py | 1 |
| DEV-05 | pipelines/polymarket.py | 1 |
| DEV-06 | engine/grok_client.py, pipelines/context_builder.py | 2 |
| DEV-07 | engine/trade_ranker.py, engine/trade_decision.py | 2 |
| DEV-08 | engine/execution.py, engine/resolution.py | 2 |
| DEV-09 | learning/calibration.py, learning/market_type.py, learning/signal_tracker.py | 3 |
| DEV-10 | learning/adjustment.py, learning/experiments.py, learning/model_swap.py | 3 |
| DEV-11 | main.py, scheduler.py | 2 |
| DEV-12 | manage.py, alerts.py | 2 |
| **TOTAL** | | **33** |
