# CLAUDE.md — Polymarket v2

## What This Is

Automated prediction market trading bot targeting Polymarket. Detects news events, estimates probabilities via Grok LLM, compares to market prices, and trades when it finds an edge. Two tiers: Tier 1 (news-driven event markets, 15min-7d resolution) and Tier 2 (crypto, 15min resolution, trigger-based).

**Design doc:** `polymarket_system_v2.md` (v2.5, 2300+ lines) — the single source of truth for all system behavior. Consult it for rationale behind any design decision.

## Commands

```bash
# Install
pip install -r requirements.txt

# Run bot
python -m src.main                    # FastAPI on 0.0.0.0:8000

# Tests
pytest                                # All tests (~445 cases)
pytest tests/test_config.py           # Single file
pytest -k "calibration"               # By keyword

# CLI management
python -m src.manage model_swap --old-model "grok-4-1-fast-reasoning" --new-model "..." --reason "..."
python -m src.manage void_trade --trade-id "<uuid>" --reason "..."
python -m src.manage start_experiment --description "..." --model "grok-4-1-fast-reasoning"
python -m src.manage end_experiment --run-id "<uuid>"
python -m src.manage recalculate_learning
```

## Project Structure

```
src/
├── main.py                  # FastAPI app + lifespan (experiment auto-creation, file logging, startup/shutdown alerts)
├── config.py                # Pydantic Settings (40+ env vars) + MonkModeConfig
├── models.py                # 19 dataclasses (Signal, Market, TradeRecord, Portfolio, etc.)
├── alerts.py                # Telegram alerting (9 alert types)
├── scheduler.py             # APScheduler: Tier 1 (15min), resolution (5min), daily summary, skip records for audit trail
├── manage.py                # CLI: model_swap, void_trade, experiments
├── db/
│   ├── sqlite.py            # Async SQLite wrapper (aiosqlite, 9 tables)
│   └── migrations.py        # DDL schema, idempotent migrations
├── pipelines/
│   ├── signal_classifier.py # Deterministic S1-S6 source tier classification
│   ├── twitter.py           # TwitterAPI.io client
│   ├── rss.py               # RSS feed parser + 24h headline dedup
│   ├── polymarket.py        # Gamma API (markets) + CLOB API (trading)
│   └── context_builder.py   # Keyword extraction + Grok context formatting
├── engine/
│   ├── grok_client.py       # xAI API wrapper, 3-attempt retry, JSON parse fallback
│   ├── trade_decision.py    # Edge calc, Kelly sizing (quarter Kelly), Monk Mode
│   ├── trade_ranker.py      # Score = edge x confidence x time_value, cluster detection
│   ├── execution.py         # Paper simulation (slippage model) + live CLOB orders
│   └── resolution.py        # Auto-resolve trades, Brier score calculation
└── learning/
    ├── calibration.py       # 6-bucket Bayesian calibration (uses RAW predictions)
    ├── market_type.py       # Per-market-type Brier tracking with exponential decay
    ├── signal_tracker.py    # (source_tier x info_type x market_type) lift calculation
    ├── adjustment.py        # 5-step pipeline: calibrate -> signal weight -> shrink -> penalty -> decay
    ├── experiments.py       # Experiment run management
    └── model_swap.py        # Model swap protocol (reset/dampen/preserve)
config/
├── known_sources.yaml       # S1-S6 tier mappings (handles, domains, keywords)
└── rss_feeds.yaml           # RSS feed URLs (Reuters, AP, BBC, CoinDesk)
tests/                       # 22 test files mirroring src/ structure
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

**Risk management (Monk Mode):** Daily loss -5%, weekly -10%, consecutive adverse cooldown (3+), max exposure 30%, API budget $8/day. All enforced in `src/engine/trade_decision.py`.

**Trade scoring:** `edge x adjusted_confidence x (1.0 / max(resolution_hours, 0.5))`. Candidates are ranked per scan cycle, not first-come-first-served. Cluster detection prevents overexposure to correlated markets.

## Key Configuration

- **Bankroll:** $2,000 (set in `config.py` INITIAL_BANKROLL)
- **Tier 1:** 15-min scan interval, resolution 15min-7d, 5 trades/day cap, 0% fee
- **Tier 2:** 2-3 min scan (only during active news window), 15-min resolution, 3 trades/day cap
- **Environment:** `ENVIRONMENT=paper` (start here) or `live`
- **DB:** SQLite at `data/predictor.db` (WAL mode)

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
- **Skip records:** The scheduler saves SKIP trade records for all early returns (grok failure, position too small, market type disabled, low edge). These are essential for debugging "why didn't it trade?" — always check the `skip_reason` column.
