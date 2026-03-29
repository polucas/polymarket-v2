# CLAUDE.md — Polymarket v2

## What This Is

Automated prediction market trading bot targeting Polymarket. Detects news events, estimates probabilities via Grok LLM, compares to market prices, and trades when it finds an edge. Two tiers: Tier 1 (news-driven event markets, 15min-7d resolution) and Tier 2 (crypto, 15min resolution, trigger-based).

**Design doc:** `polymarket_system_v2.md` (v2.7, 2500+ lines) — the single source of truth for all system behavior. Consult it for rationale behind any design decision.

## Commands

```bash
# Install
pip install -r requirements.txt

# Run bot
python -m src.main                    # FastAPI on 0.0.0.0:8000

# Tests
pytest                                # All tests (~535 cases)
pytest tests/test_config.py           # Single file
pytest -k "calibration"               # By keyword

# CLI management
python -m src.manage model_swap --old-model "grok-4.20-experimental-beta-0304-reasoning" --new-model "..." --reason "..."
python -m src.manage void_trade --trade-id "<uuid>" --reason "..."
python -m src.manage start_experiment --description "..." --model "grok-4.20-experimental-beta-0304-reasoning"
python -m src.manage end_experiment --run-id "<uuid>"
python -m src.manage recalculate_learning
```

## Project Structure

```
src/
├── main.py                  # FastAPI app + lifespan (experiment auto-creation, file logging, startup/shutdown alerts)
├── config.py                # Pydantic Settings (40+ env vars) + MonkModeConfig
├── models.py                # 22 dataclasses (Signal, Market, OrderBookLevel, TradeRecord, Portfolio, DailyReview, etc.)
├── alerts.py                # Telegram alerting (10 alert types, incl. early exit)
├── scheduler.py             # APScheduler: Tier 1 (15min), RSS (30s), resolution (5min), early exit, daily summary, self-check
├── manage.py                # CLI: model_swap, void_trade, experiments
├── db/
│   ├── sqlite.py            # Async SQLite wrapper (aiosqlite, 10 tables)
│   └── migrations.py        # DDL schema, idempotent migrations (v6)
├── pipelines/
│   ├── signal_classifier.py # Deterministic S1-S6 source tier + I1-I5 info type classification
│   ├── twitter.py           # TwitterAPI.io client
│   ├── rss.py               # RSS feed parser + 24h headline dedup + 30s signal accumulator
│   ├── polymarket.py        # Gamma API (markets) + CLOB API (trading)
│   └── context_builder.py   # Keyword extraction + Grok context formatting
├── engine/
│   ├── grok_client.py       # xAI API wrapper, 3-attempt retry, JSON parse fallback
│   ├── trade_decision.py    # Spread-aware edge calc, VWAP Kelly sizing, Monk Mode
│   ├── trade_ranker.py      # Score = edge x confidence x time_value, cluster detection
│   ├── execution.py         # Paper simulation (maker/taker) + live CLOB orders
│   ├── ws_exit.py           # Real-time WebSocket TP/SL monitor (Polymarket market channel)
│   └── resolution.py        # Auto-resolve trades, early exit (TP/SL fallback), Brier score calculation
└── learning/
    ├── calibration.py       # 6-bucket Bayesian calibration (uses RAW predictions)
    ├── market_type.py       # Per-market-type Brier tracking with exponential decay
    ├── signal_tracker.py    # (source_tier x info_type x market_type) lift calculation
    ├── adjustment.py        # 5-step pipeline: calibrate -> signal weight -> shrink -> penalty -> decay
    ├── experiments.py       # Experiment run management
    ├── self_check.py        # Daily self-check loop (Karpathy Autoresearch inspired)
    └── model_swap.py        # Model swap protocol (reset/dampen/preserve)
config/
├── known_sources.yaml       # S1-S6 tier mappings (handles, domains, keywords)
└── rss_feeds.yaml           # RSS feed URLs (Reuters, AP, BBC, CoinDesk)
tests/                       # 26 test files mirroring src/ structure
docs/
├── TASKS.md                 # 12 dev agent specs (DEV-01 through DEV-12)
└── TESTS.md                 # 10 test agent specs with execution DAG
```

## Architecture Essentials

**Async throughout** — all I/O uses async/await (aiosqlite, httpx, APScheduler AsyncIOScheduler).

**Learning system critical invariant:** Calibration uses RAW Grok probability, never adjusted. Market-type and signal trackers use ADJUSTED. Mixing these up causes self-referencing convergence. See `polymarket_system_v2.md` Section 7.

**Adjustment pipeline order** (in `src/learning/adjustment.py`):
1. Bayesian calibration correction (model-specific)
2. Signal-type weighting (model-independent)
3. Probability shrinkage toward 0.50
4. Market-type edge penalty
5. Temporal confidence decay

**Risk management (Monk Mode):** Daily loss -5%, weekly -10%, consecutive adverse cooldown (3+), max exposure 30%, API budget $15/day. All enforced in `src/engine/trade_decision.py`.

**Duplicate bet prevention:** 24h cooldown after trading a market (`MARKET_COOLDOWN_HOURS`), plus Jaccard keyword-overlap similarity check at 60% threshold (`QUESTION_SIMILARITY_THRESHOLD`) to block near-duplicate questions. Additionally, a 2h evaluation cooldown (`EVALUATION_COOLDOWN_HOURS`) prevents re-calling Grok on the same market within 2 hours (silent skip, no DB record). Skip reasons: `market_cooldown`, `similar_to_{id}`.

**Daily self-check (Autoresearch):** Runs 15 min after nightly summary. Gathers metrics (win rate, ROI, Brier by type, calibration drift, skip reasons), calls Grok for analysis, persists to `daily_reviews` table + `data/daily_reviews/*.md`, sends Telegram alert. Does NOT auto-implement changes.

**Orderbook re-fetch after Grok:** The orderbook is fetched once before Grok is called, then re-fetched after `call_grok_with_retry()` returns. This prevents adverse selection: if latency bots sweep the book during the ~2-4s Grok call, the edge recalculation uses fresh prices. If the book moved against the prediction, edge will be negative and the trade is skipped. See `scheduler.py:_process_market()`.

**Deterministic signal classification:** `info_type` (I1-I5) is assigned deterministically at signal creation time via `classify_info_type(source_tier)` in `src/pipelines/signal_classifier.py`. S1→I1, S2/S3→I2, S4→I3, S5→I4, S6→I5. This replaces the previous approach of asking Grok to classify signal types (which introduced subjective variance and lacked timestamps). Signal tags now include real publication timestamps, making temporal confidence decay actually operational.

**Temporal confidence decay (market-type specific):** Step 5 of `adjust_prediction()` decays confidence based on signal age. Rates vary by market type — crypto signals decay at 0.05/min (floor 0.40, grace 1min), while political/regulatory signals decay at 0.01-0.02/hr (floor 0.80-0.85, grace 1-2h). See `_DECAY_PARAMS` in `src/learning/adjustment.py`. Decay only fires when signal timestamps are present in `signal_tags`.

**Spread-aware edge:** Edge uses best ask price (BUY_YES) or best bid price (BUY_NO) from the CLOB orderbook, not the Gamma API midpoint. Falls back to `market.yes_price` if orderbook is empty. Implemented in `calculate_spread_adjusted_edge()`.

**VWAP Kelly sizing:** Kelly position size is capped by profitable orderbook depth. `kelly_size_vwap()` walks the orderbook levels, computes volume-weighted average price, and binary-searches for the maximum size where VWAP still gives edge above `TIER1_MIN_EDGE`. Prevents oversizing into thin books.

**Extreme price guards:** Two layers prevent leverage explosions at extreme prices (e.g., BUY_NO at YES=0.9995 buying millions of shares for a few dollars):
1. **Market price filter** (`MIN_TRADEABLE_PRICE=0.05`, `MAX_TRADEABLE_PRICE=0.95`): Markets where one side is <5% are skipped during market fetch. Applied in `get_active_markets()` after resolution and liquidity filters.
2. **Notional exposure cap** in `kelly_size()`: Max possible payout (position / share_price) is capped at `MAX_POSITION_PCT * bankroll`. A single trade can never win more than the position cap. This prevents leverage from extreme prices (buying at $0.05/share) from creating outsized payouts.

**Maker execution:** Both Tier 1 and Tier 2 use maker (limit) orders (`TIER1_EXECUTION_TYPE=maker`). Paper mode simulates fill probability 40-80% with zero slippage. This acknowledges that LLM inference latency (2-4s) makes taker sniping non-viable.

**Early exit (TP/SL):** Real-time WebSocket monitoring (`src/engine/ws_exit.py`) subscribes to Polymarket's market channel for YES tokens of open positions. On book update, best bid is checked against TP (+20% ROI) and SL (-15% ROI) thresholds for instant exit. The 5-min polling loop (`check_early_exits()`) is kept as fallback when WS is disconnected. Early-exited trades have PnL but no `actual_outcome` — excluded from Brier/calibration metrics. Feature flag: `EARLY_EXIT_ENABLED`.

**RSS signal accumulator:** RSS feeds poll independently every 30s (`RSS_POLL_INTERVAL_SECONDS`) via `poll_and_accumulate()`. Tier 1 scan consumes accumulated signals via `consume_signals()`. Tier 2 still calls `get_breaking_news()` directly.

**Trade scoring:** `edge x adjusted_confidence x (1.0 / max(resolution_hours, 0.5))`. Candidates are ranked per scan cycle, not first-come-first-served. Cluster detection prevents overexposure to correlated markets.

## Key Configuration

- **Model:** `GROK_MODEL` env var, default `grok-4.20-experimental-beta-0304-reasoning` (configurable in `.env`)
- **Bankroll:** $10,000 (set in `config.py` INITIAL_BANKROLL). Max bet $160 via `MAX_POSITION_PCT=0.016` applied to live `portfolio.total_equity`.
- **Tier 1:** 15-min scan interval, resolution 15min-7d, 20 trades/day cap, 0% fee, 3% min edge
- **Tier 2:** 2-3 min scan (only during active news window), 15-min resolution, 3 trades/day cap
- **Market fetch:** 3 pages x 500 markets sorted by volume (`MARKET_PAGE_SIZE=500`, `MARKET_FETCH_PAGES=3`) = up to 1,500 most active markets per scan
- **Price filter:** Skip markets outside 5%-95% YES range (`MIN_TRADEABLE_PRICE=0.05`, `MAX_TRADEABLE_PRICE=0.95`)
- **API budget:** $15/day (`DAILY_API_BUDGET_USD=15.0`)
- **Duplicate prevention:** 24h market cooldown, 2h evaluation cooldown, 60% question similarity threshold
- **Environment:** `ENVIRONMENT=paper` (start here) or `live`
- **DB:** SQLite at `data/predictor.db` (WAL mode)
- **RSS polling:** 30s independent cycle (`RSS_POLL_INTERVAL_SECONDS=30`)
- **Execution:** Both tiers use maker orders (`TIER1_EXECUTION_TYPE=maker`, `TIER2_EXECUTION_TYPE=maker`)
- **Early exit:** Take-profit at +20% ROI, stop-loss at -15% ROI (`EARLY_EXIT_ENABLED=true`)
- **Dashboard:** `/health` (status), `/reviews` and `/reviews/{date}` (daily self-check reports)

## Conventions

- **Testing:** pytest + pytest-asyncio. Fixtures in `tests/conftest.py` (`db`, `sample_trade_record()`, `sample_signal()`). All external APIs mocked.
- **Logging:** structlog with JSON renderer, ISO timestamps. Use `log = structlog.get_logger()` at module level. Logs go to both stdout and `data/bot.log` (file handler added at startup).
- **Config:** All settings via Pydantic `Settings` class reading from `.env`. Never hardcode credentials.
- **Type hints** on all function signatures. Dataclasses for domain objects (not dicts).
- **Error handling in scheduler:** Each scan/job catches exceptions, logs them, and sends Telegram error alert. Individual market failures don't abort the scan cycle.

## Things to Watch Out For

- **Experiment runs required:** The `trade_records` table has a FK constraint on `experiment_runs(run_id)` with `PRAGMA foreign_keys=ON`. If no experiment run exists, all `save_trade()` calls fail silently. The lifespan auto-creates one on startup, but if you're writing tests or scripts that insert trade records, ensure an experiment run exists first.
- The `known_sources.yaml` maps Twitter handles and domains to source tiers (S1-S6). If adding new sources, classify them correctly — S1 is official/institutional, S6 is anonymous/low-credibility.
- Polymarket CLOB API credentials (API_KEY, SECRET, PASSPHRASE) are only needed for live trading, not paper mode.
- The `seen_headlines` dict in RSS pipeline uses a bounded 24h TTL to prevent memory leaks. Don't remove that bound.
- Kelly sizing uses quarter Kelly (`KELLY_FRACTION=0.25`), not full Kelly. This is intentional conservatism.
- Resolution window for Tier 1 is 15min to 7 days (was 1-24h, changed because that range was empty on Polymarket).
- **TIER1_FEE_RATE must be 0.0** for fee-free markets. A non-zero value creates a phantom fee that eats into edge calculations and causes valid trades to fall below the minimum edge threshold.
- **Skip records:** The scheduler saves SKIP trade records for all early returns (grok failure, position too small, market type disabled, low edge, market cooldown, similar question). These are essential for debugging "why didn't it trade?" — always check the `skip_reason` column.
- **GROK_MODEL env var:** Model name is no longer hardcoded. Set `GROK_MODEL` in `.env` to change models. After changing, run `python -m src.manage model_swap` to reset calibration properly.
- **Paper mode relaxations:** Min position threshold is $0.50 in paper mode vs $1.00 in live. `TIER1_MIN_EDGE` defaults to 0.03 (can increase for live).
- **Daily reviews:** Written to both DB (`daily_reviews` table) and `data/daily_reviews/YYYY-MM-DD.md`. Check `/reviews` endpoint or the markdown files for daily performance analysis.
- **OrderBookLevel vs plain floats:** `OrderBook.bids` and `OrderBook.asks` are `List[OrderBookLevel]` (price + size), NOT plain floats. Bids are sorted descending by price, asks ascending, before slicing top 5 from the CLOB API. Tests using mock OrderBooks must use `OrderBookLevel` objects or mock the `spread`/`total_depth` properties.
- **Evaluation cooldown:** Markets that received a Grok evaluation (including low_edge SKIPs) are silently skipped for 2 hours (`EVALUATION_COOLDOWN_HOURS`). No DB record is written for these skips to avoid bloating the `trade_records` table. This prevents wasteful repeated LLM calls on the same market when few markets pass the resolution filter.
- **Early-exited trades have no actual_outcome:** They have PnL and `exit_type` set but `actual_outcome` is NULL. The learning system (calibration, Brier) correctly skips them. The `get_open_trades()` query filters `AND exit_type IS NULL` to avoid re-processing.
- **VWAP returns price, not 1.0:** `compute_vwap()` returns `total_usd / total_shares` — the volume-weighted average *price* per share, not a USD ratio. VWAP is used in `kelly_size_vwap()` to cap position size at the depth where the trade remains profitable.
- **RSS accumulator lock:** `poll_and_accumulate()` uses an `asyncio.Lock` to prevent race conditions with `consume_signals()`. The lock is per-instance, not global.
- **WebSocket exit manager:** `RealTimeExitManager` in `src/engine/ws_exit.py` subscribes to YES tokens of open positions on the Polymarket market channel (`wss://ws-subscriptions-clob.polymarket.com/ws/market`). It calculates ROI from the best bid (sell price) and triggers instant TP/SL exits. The 5-min polling fallback in `check_early_exits()` catches anything missed during WS disconnects. Positions are tracked by `clob_token_id_yes` (stored in TradeRecord since migration v6).
- **Pagination settings for mocked tests:** Tests that mock `Settings` with `spec=Settings` must set `MARKET_PAGE_SIZE`, `MARKET_FETCH_PAGES`, `MIN_TRADEABLE_PRICE`, and `MAX_TRADEABLE_PRICE` in addition to `MARKET_FETCH_LIMIT` for any test calling `get_active_markets()`.
- **signal_tags format:** `signal_tags` passed to `adjust_prediction()` are now built from signal objects (not from Grok's response). Format: `[{"source_tier": "SX", "info_type": "IX", "timestamp": "ISO8601|None"}]`. `info_type` is assigned by `classify_info_type(source_tier)` at creation. Timestamps from RSS/Twitter publication time enable temporal decay in Step 5.
- **Grok response no longer includes signal_info_types:** `REQUIRED_FIELDS` in `grok_client.py` is `{"estimated_probability", "confidence", "reasoning"}`. Grok is not asked to classify signals — this is done deterministically. Tests that build Grok response dicts should NOT include `signal_info_types`.
