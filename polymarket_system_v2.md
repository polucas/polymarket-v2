# Polymarket Prediction System v2.3 — Complete Project Bible

> **Version:** 2.3 | **Last updated:** 2026-02-21
> **Status:** Pre-implementation (design complete, paper trading not started)
> **Bankroll:** $5,000 | **Monthly OpEx target:** <$30
> 
> **v2.3 changes (critical review fixes):**
> - Fixed calibration feedback loop — now uses raw probability, not adjusted (prevented self-referencing convergence)
> - Fixed probability adjustment — shrinkage toward 0.50 instead of broken linear shift
> - Fixed Kelly formula — proper binary market Kelly (was using naive f*=edge)
> - Added market-to-signal keyword extraction (Section 11.1)
> - Added correlated market detection (Section 10.2)
> - Implemented temporal confidence decay (was documented but missing)
> - Enforced weekly loss limit (was in config but dead code)
> - Added Grok JSON parse error handling with retry (Section 11.5)
> - Fixed seen_headlines memory leak (bounded dict with 24h TTL)
> - Fixed cooldown for slow-resolving trades (tracks unrealized adverse moves)
> - Track both raw and adjusted Brier scores (go-live uses adjusted)
> - Corrected Tier 2 cost estimate (was 5x too low)
> - Added structured logging strategy (Section 15.1)
> - Added health check endpoint (Section 15.2)

---

## Table of Contents

0. [Project Overview & Realistic Expectations](#0-project-overview--realistic-expectations)
1. [Prerequisites & Accounts](#1-prerequisites--accounts)
2. [Infrastructure Setup (VPS)](#2-infrastructure-setup-vps)
3. [Decisions Locked](#3-decisions-locked)
4. [System Architecture](#4-system-architecture)
5. [Tier Configuration](#5-tier-configuration)
6. [Signal Classification (Two-Dimensional)](#6-signal-classification-two-dimensional)
7. [Learning System](#7-learning-system)
8. [Model Swap Protocol](#8-model-swap-protocol)
9. [Monk Mode v2](#9-monk-mode-v2)
10. [Trade Decision Engine](#10-trade-decision-engine)
11. [Data Pipeline](#11-data-pipeline)
12. [Paper Trading](#12-paper-trading)
13. [SQLite Schema](#13-sqlite-schema)
14. [Cost Budget](#14-cost-budget)
15. [Deployment & File Structure](#15-deployment--file-structure)
16. [Dashboard & Monitoring](#16-dashboard--monitoring)
17. [Implementation Priority](#17-implementation-priority)
18. [Operational Runbooks](#18-operational-runbooks)

---

## 0. Project Overview & Realistic Expectations

This system is an automated prediction market trading bot targeting Polymarket. It detects news events, estimates probabilities using an LLM, compares them to market prices, and trades when it finds an edge.

**What this is:** A disciplined, low-cost experiment to see if LLM-assisted prediction can generate alpha on Polymarket event markets, with a learning system that improves over time.

**What this is NOT:** A guaranteed money printer. Only 7.6% of Polymarket participants are profitable. The system must prove itself in paper trading before any real capital is risked.

**Realistic expectations:**
- Paper trading will take 4-8 weeks (200+ trades minimum)
- Most signal types will show zero edge initially
- Monthly OpEx of ~$26 (VPS + API costs) runs regardless of performance
- The learning system needs 100+ resolved trades per market type before its adjustments become reliable
- If adjusted Brier score is consistently >0.30 after 200 trades, the system does not have an edge and should not go live

---

## 1. Prerequisites & Accounts

Everything you need before writing a single line of code.

### 1.1 Accounts to Create

| Account | URL | What For | Cost | Setup Notes |
|---------|-----|----------|------|-------------|
| **Polymarket** | https://polymarket.com | Trading (paper then live) | Free to create; funded via USDC on Polygon | Need wallet (MetaMask/Rabby). KYC may be required for withdrawals. |
| **xAI (Grok API)** | https://console.x.ai | LLM for probability estimation | Pay-as-you-go (~$6.42/mo) | API key needed. Model: `grok-4.1-fast`. |
| **TwitterAPI.io** | https://twitterapi.io | Social signal ingestion | Pay-as-you-go (~$15.30/mo) | API key needed. Unofficial Twitter API proxy. |
| **Hetzner Cloud** | https://console.hetzner.cloud | VPS hosting (24/7 operation) | €4.35/mo (CX22) | EU datacenter. See Section 2 for setup. |
| **GitHub** | https://github.com | Code repository + CI | Free | Private repo recommended. |
| **Cloudflare** (optional) | https://cloudflare.com | Tunnel for dashboard access | Free tier | Avoid exposing VPS ports directly. |

### 1.2 API Keys to Generate

After creating accounts, generate and securely store these keys:

```
XAI_API_KEY=xai-xxxxxxxxxxxxxxxxxxxx        # From console.x.ai → API Keys
TWITTER_API_KEY=xxxxxxxxxxxxxxxxxx           # From twitterapi.io → Dashboard
POLYMARKET_API_KEY=xxxxxxxxxxxxxxxx          # From Polymarket CLOB API docs
POLYMARKET_SECRET=xxxxxxxxxxxxxxxx           # CLOB trading credential
POLYMARKET_PASSPHRASE=xxxxxxxxxxxxxxxx       # CLOB trading credential
```

**Polymarket API credentials:** Polymarket uses a CLOB (Central Limit Order Book) API. You need to generate API credentials from within your Polymarket account. Documentation: https://docs.polymarket.com/#clob-api

### 1.3 Wallet Setup (for Live Trading — Not Needed for Paper)

- Install MetaMask or Rabby browser extension
- Create a wallet on Polygon network
- Fund with USDC (bridged from Ethereum or direct purchase)
- The Polymarket CLOB API uses this wallet to sign orders
- **DO NOT fund the wallet until paper trading is complete and profitable**

### 1.4 Local Development Requirements

| Tool | Version | Install |
|------|---------|---------|
| Python | 3.11+ | `pyenv install 3.11` or system package |
| pip | Latest | Bundled with Python |
| Git | 2.x+ | System package |
| SSH client | Any | For VPS access |
| Docker (optional) | 24+ | For local testing with docker-compose |

### 1.5 Python Dependencies

```
# Core
fastapi>=0.100.0
uvicorn>=0.23.0
apscheduler>=3.10.0
httpx>=0.24.0
aiohttp>=3.9.0
feedparser>=6.0.0
pydantic>=2.0.0
pydantic-settings>=2.0.0

# Database
aiosqlite>=0.19.0

# Math/Stats
scipy>=1.11.0
numpy>=1.24.0

# Monitoring
datasette>=0.64.0
datasette-auth-passwords>=0.4.0

# Utilities
python-dotenv>=1.0.0
structlog>=23.1.0

# Polymarket
py-clob-client>=0.1.0    # Polymarket CLOB SDK
```

---

## 2. Infrastructure Setup (VPS)

### 2.1 Why a VPS is Required

The system must run 24/7 because:
- Polymarket markets resolve at any hour. Missed resolution windows = missed data for learning.
- News breaks at any time. The value of news-driven trades degrades in minutes.
- RSS feeds must be polled every 5 minutes, Twitter every 15 minutes.
- The dashboard must be accessible at any time without "waking up" a local machine.

A laptop or local machine fails these requirements — sleep mode, network changes, power outages all break the system.

### 2.2 VPS Selection: Hetzner CX22

| Spec | Value | Why Sufficient |
|------|-------|----------------|
| Provider | Hetzner Cloud | EU datacenter (Falkenstein/Helsinki), excellent price/performance |
| Plan | CX22 | Cheapest option that comfortably runs the workload |
| vCPU | 2 (shared) | Python + SQLite + Datasette = minimal CPU |
| RAM | 4 GB | Python process ~200MB, SQLite in-memory cache ~100MB, Datasette ~100MB |
| Storage | 40 GB SSD | SQLite DB will stay <1GB for years. Plenty for logs. |
| Bandwidth | 20 TB/mo | Far exceeds needs (~5 GB/month of API traffic) |
| Cost | €4.35/mo (~$4.70) | Annual: ~$56 |
| Location | Falkenstein (DE) or Helsinki (FI) | EU-based, decent latency to Polymarket infra |

**Alternative if Hetzner is unavailable in your region:** DigitalOcean Basic Droplet ($6/mo, 1 vCPU, 1GB RAM — tighter but workable) or Oracle Cloud Free Tier (free ARM instance — 4 vCPU, 24GB RAM, but less reliable availability).

### 2.3 VPS Setup — Step by Step

#### Step 1: Create the Server

1. Go to https://console.hetzner.cloud
2. Create a new project (e.g., "polymarket-bot")
3. Add a server:
   - Location: Falkenstein (cheapest) or Helsinki
   - OS: Ubuntu 24.04
   - Type: Shared vCPU → CX22
   - SSH key: Add your public key (see Step 2 if you don't have one)
   - Name: `polymarket-prod`
4. Click Create. Note the IP address.

#### Step 2: SSH Key Setup (if needed)

```bash
# On your local machine
ssh-keygen -t ed25519 -C "polymarket-bot"
# Accept defaults. Public key is at ~/.ssh/id_ed25519.pub
cat ~/.ssh/id_ed25519.pub
# Copy this and paste into Hetzner SSH Keys section
```

#### Step 3: Initial Server Security

```bash
# SSH into server
ssh root@YOUR_SERVER_IP

# Update system
apt update && apt upgrade -y

# Create non-root user
adduser botuser
usermod -aG sudo botuser

# Copy SSH key to new user
mkdir -p /home/botuser/.ssh
cp ~/.ssh/authorized_keys /home/botuser/.ssh/
chown -R botuser:botuser /home/botuser/.ssh
chmod 700 /home/botuser/.ssh
chmod 600 /home/botuser/.ssh/authorized_keys

# Disable root SSH login and password auth
sed -i 's/PermitRootLogin yes/PermitRootLogin no/' /etc/ssh/sshd_config
sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
systemctl restart sshd

# Set up firewall
ufw allow OpenSSH
ufw allow 8001/tcp    # Datasette dashboard
ufw enable

# Install fail2ban
apt install -y fail2ban
systemctl enable fail2ban
```

#### Step 4: Install Python & Dependencies

```bash
# Switch to botuser
su - botuser

# Install Python 3.11 + tools
sudo apt install -y python3.11 python3.11-venv python3-pip git

# Create project directory
mkdir -p ~/polymarket-bot
cd ~/polymarket-bot

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies (from requirements.txt — see Section 1.5)
pip install -r requirements.txt
```

#### Step 5: Environment Variables

```bash
# Create .env file (NEVER commit this to git)
cat > ~/polymarket-bot/.env << 'EOF'
# LLM
XAI_API_KEY=xai-your-key-here

# Twitter
TWITTER_API_KEY=your-key-here

# Polymarket
POLYMARKET_API_KEY=your-key-here
POLYMARKET_SECRET=your-secret-here
POLYMARKET_PASSPHRASE=your-passphrase-here

# System
ENVIRONMENT=paper          # "paper" or "live" — START WITH PAPER
LOG_LEVEL=INFO
DASHBOARD_PORT=8001
DASHBOARD_PASSWORD_HASH=   # Generate: python -c "from datasette_auth_passwords import hash_password; print(hash_password('your-password'))"

# Alerts (optional)
TELEGRAM_BOT_TOKEN=        # For trade notifications
TELEGRAM_CHAT_ID=
EOF

chmod 600 ~/polymarket-bot/.env
```

#### Step 6: Systemd Services (Auto-Start on Boot)

```bash
# Main bot service
sudo cat > /etc/systemd/system/polymarket-bot.service << 'EOF'
[Unit]
Description=Polymarket Prediction Bot
After=network.target

[Service]
Type=simple
User=botuser
WorkingDirectory=/home/botuser/polymarket-bot
Environment=PATH=/home/botuser/polymarket-bot/venv/bin:/usr/bin
EnvironmentFile=/home/botuser/polymarket-bot/.env
ExecStart=/home/botuser/polymarket-bot/venv/bin/python -m src.main
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Dashboard service (Datasette)
sudo cat > /etc/systemd/system/polymarket-dashboard.service << 'EOF'
[Unit]
Description=Polymarket Dashboard (Datasette)
After=network.target

[Service]
Type=simple
User=botuser
WorkingDirectory=/home/botuser/polymarket-bot
Environment=PATH=/home/botuser/polymarket-bot/venv/bin:/usr/bin
ExecStart=/home/botuser/polymarket-bot/venv/bin/datasette /home/botuser/polymarket-bot/data/predictor.db --host 0.0.0.0 --port 8001 --setting sql_time_limit_ms 5000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable services
sudo systemctl daemon-reload
sudo systemctl enable polymarket-bot
sudo systemctl enable polymarket-dashboard

# Start services
sudo systemctl start polymarket-bot
sudo systemctl start polymarket-dashboard
```

#### Step 7: Verify Everything Works

```bash
# Check bot is running
sudo systemctl status polymarket-bot

# Check dashboard is accessible
curl http://localhost:8001

# From your local machine, access dashboard
# http://YOUR_SERVER_IP:8001

# View bot logs
sudo journalctl -u polymarket-bot -f
```

#### Step 8: Optional — Cloudflare Tunnel (HTTPS + Auth)

If you don't want to expose port 8001 directly:

```bash
# Install cloudflared
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /usr/local/bin/cloudflared
chmod +x /usr/local/bin/cloudflared

# Login to Cloudflare
cloudflared tunnel login

# Create tunnel
cloudflared tunnel create polymarket-dashboard

# Configure
cat > ~/.cloudflared/config.yml << EOF
tunnel: YOUR_TUNNEL_ID
credentials-file: /home/botuser/.cloudflared/YOUR_TUNNEL_ID.json

ingress:
  - hostname: dashboard.yourdomain.com
    service: http://localhost:8001
  - service: http_status:404
EOF

# Run as service
sudo cloudflared service install
sudo systemctl start cloudflared
```

Now dashboard is at `https://dashboard.yourdomain.com` with Cloudflare's security.

### 2.4 VPS Maintenance

| Task | Frequency | Command |
|------|-----------|---------|
| System updates | Weekly | `sudo apt update && sudo apt upgrade -y` |
| Reboot (for kernel updates) | Monthly | `sudo reboot` (bot auto-restarts via systemd) |
| Check disk usage | Monthly | `df -h` (alert if >80%) |
| Backup SQLite DB | Daily (cron) | `cp data/predictor.db data/backups/predictor_$(date +%F).db` |
| Rotate old backups | Weekly | Keep last 30 daily backups |
| Review logs for errors | Weekly | `sudo journalctl -u polymarket-bot --since "7 days ago" --priority err` |

**Automated backup cron:**

```bash
# Add to botuser's crontab: crontab -e
0 3 * * * cp /home/botuser/polymarket-bot/data/predictor.db /home/botuser/polymarket-bot/data/backups/predictor_$(date +\%F).db
0 4 * * 0 find /home/botuser/polymarket-bot/data/backups/ -name "*.db" -mtime +30 -delete
```

---

## 3. Decisions Locked

| Decision | Choice | Rationale |
|----------|--------|-----------|
| LLM | Grok 4.1 Fast ($0.20/$0.50 per M tokens) | Best reasoning-per-dollar, no refusal on financial predictions |
| Social data | TwitterAPI.io ($0.15/1K tweets) | Full control over pre-filtering, cheaper than Grok X Search |
| News data | RSS feeds (AP, Reuters, Bloomberg) | Free, real-time, headline-level sufficient as trigger (see 11.3) |
| Strategy | Dual-tier (News Sniper + Crypto Opportunist) | Maximizes edge per dollar of OpEx |
| DB | SQLite | Zero-config, handles concurrency, no JSON race conditions |
| Order execution | Maker + Taker | Maker for rebates on crypto; taker for speed on news events |
| Paper trading first | Mandatory | Minimum 200 trades before live capital |
| Signal taxonomy | Two-dimensional (Source Tier × Info Type) | Avoids mixed taxonomy, enables granular learning per market type |
| Model swap policy | Per-layer reset strategy | Calibration resets, market-type dampens, signals preserved |
| VPS | Hetzner CX22 (€4.35/mo) | 24/7 operation, EU-based, auto-restart via systemd |
| Dashboard | Datasette (Phase 1) → React (Phase 3+) | Zero-code SQL explorer first; custom UI only if profitable |

---

## 4. System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        SCHEDULER (APScheduler)                    │
│  Tier 1: every 15 min   │   Tier 2: trigger-based (2-3 min)     │
└──────────┬───────────────┴──────────────┬────────────────────────┘
           │                              │
           ▼                              ▼
┌─────────────────────┐      ┌─────────────────────────┐
│   MARKET SCANNER    │      │   NEWS DETECTOR          │
│   (Polymarket API)  │      │   (RSS + TwitterAPI.io)  │
│                     │      │                          │
│ Gamma API: markets  │      │ Checks every 5 min:      │
│ CLOB WS: orderbook  │      │  - RSS feeds (free)      │
│                     │      │  - Twitter search (paid)  │
│ Filters:            │      │                          │
│  - Resolution 1-24h │      │ If breaking news found → │
│  - Liquidity > $5K  │      │   activate Tier 2 scan   │
│  - Fee-free markets │      │   for 30 min window      │
└──────────┬──────────┘      └────────────┬────────────┘
           │                              │
           ▼                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                     DATA PIPELINE (Python)                        │
│                                                                   │
│  1. For each candidate market:                                    │
│     a. Pull tweets via TwitterAPI.io (keyword search)             │
│     b. Pre-filter: verified accounts, >1K followers, <2h old     │
│     c. Classify source tier (S1-S6) — programmatic               │
│     d. Deduplicate (same event → single summary)                  │
│     e. Pull RSS headlines, classify source tier                   │
│     f. Get order book snapshot from CLOB → classify as S5/I6     │
│                                                                   │
│  2. Compress into structured context (~800-1500 tokens):          │
│     {market_question, current_price, resolution_time,             │
│      top_signals_with_source_tiers, orderbook_summary, fee_rate}  │
│                                                                   │
│  Output: MarketContext object per market                           │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                   GROK 4.1 FAST (Prediction Engine)               │
│                                                                   │
│  Single API call per market. Receives MarketContext.               │
│  Returns structured JSON:                                         │
│    {estimated_probability, confidence, reasoning,                  │
│     key_signals_used, contradictions,                              │
│     signal_info_types: {"signal_name": "I1|I2|I3|I4|I5"}}       │
│                                                                   │
│  Grok classifies information types (I1-I5) per signal.           │
│  Source tiers (S1-S6) are pre-classified before Grok sees them.  │
│                                                                   │
│  IMPORTANT: Grok outputs RAW probability.                         │
│  Learning system adjusts it AFTER Grok returns.                   │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                    LEARNING ADJUSTMENT LAYER                      │
│                                                                   │
│  Inputs: Grok's raw_probability + raw_confidence                  │
│  Applies (in order):                                              │
│    1. Bayesian calibration correction (MODEL-SPECIFIC)            │
│    2. Signal-type weighting (MODEL-INDEPENDENT)                   │
│    3. Probability shrinkage toward 0.50 (calibration-derived)     │
│    4. Market-type performance adjustment (PARTIALLY model-dep.)   │
│    5. Temporal confidence decay (signal freshness)                 │
│  Output: adjusted_probability, adjusted_confidence                │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                   TRADE DECISION ENGINE (with Ranking)            │
│                                                                   │
│  1. Collect ALL candidates from this scan cycle                   │
│  2. Score each: edge × confidence × time_value                    │
│  3. Rank by score — NOT first-come-first-served                  │
│  4. For top N (respecting daily cap):                             │
│     a. Subtract fee (if applicable)                               │
│     b. Check Monk Mode constraints                                │
│     c. Kelly sizing → execute (paper or live)                     │
│  5. Remaining → log as SKIP (still tracked for learning)         │
│  6. If daily cap already hit → observe-only mode (no Grok calls) │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                    EXECUTION + RESOLUTION                          │
│                                                                   │
│  Paper mode: simulate with slippage                               │
│  Live mode: Polymarket CLOB API                                   │
│    - Tier 1 (event markets): taker orders (speed matters)         │
│    - Tier 2 (crypto): maker orders (rebate matters)               │
│                                                                   │
│  Auto-resolution:                                                 │
│    - Crypto 15-min: check price at resolution time                │
│    - Event markets: poll Polymarket for resolution status          │
│  All outcomes → Learning DB                                       │
└──────────────────────────────────────────────────────────────────┘
```

---

## 5. Tier Configuration

### 5.1 Tier 1: News Sniper (Primary — 80% of capital)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Scan frequency | Every 15 min | Mispricings persist minutes-hours on event markets |
| Market types | Political, regulatory, economic, cultural, sports | Fee-free on Polymarket |
| Resolution window | 1-24 hours | Sub-1h markets reprice before 15-min scan detects them (see 5.3) |
| Fee rate | 0% (fee-free markets) | No taker fees on non-crypto, non-sports markets |
| Min edge threshold | 4% (high confidence) / 7% (moderate) | Minimum quality bar — ranking handles prioritization above this |
| Min confidence | 65% (adjusted) | Lower start, let Bayesian calibration correct |
| Execution | Taker orders | Speed > rebate for news-driven moves |
| Max position | 8% of bankroll | Conservative for event uncertainty |
| Daily trade cap | 5 | Quality over quantity — enforced via ranking, not first-come |

### 5.2 Tier 2: Crypto Opportunist (Secondary — 20% of capital)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Scan frequency | Every 2-3 min, ONLY during active news window | Don't compete with latency bots during quiet periods |
| Market types | BTC/ETH/SOL 15-min price predictions | Only crypto markets |
| Resolution window | 15 min | Ultra-short |
| Fee rate | Max 1.56% taker / rebate as maker | Corrected from original 3.15% |
| Min edge threshold | 5% after fees | Higher bar due to fees and competition |
| Min confidence | 70% (adjusted) | Slightly higher for adversarial environment |
| Execution | Maker orders (limit) | Earn rebates, avoid taker fees |
| Max position | 5% of bankroll | Smaller due to higher risk |
| Daily trade cap | 3 | Most days should be 0 |
| Activation trigger | News Detector finds crypto-relevant breaking news | Don't run blind |

### 5.3 Why 1h Minimum Resolution (Not 15 min) for Tier 1

Markets resolving in 15-60 minutes on non-crypto events are almost certainly already priced correctly by the time the 15-min scan detects them. If a Fed decision is announced at 2:00 PM and a market resolves at 2:15 PM, the scan might not catch it before resolution. And if it does, the price has already moved to 0.95+ because faster participants have acted.

The 1h minimum gives the system time to: detect news → pull signals → call Grok → calculate edge → execute. That pipeline takes 1-3 minutes. The remaining 57+ minutes is for the position to have value.

**Revisit condition:** If WebSocket-based real-time news detection is added (not in v1), sub-1h markets become viable. Track skipped markets with `resolution_window_too_short` to quantify missed opportunities.

### 5.4 Tier 2 Activation Logic

Tier 2 is OFF by default. It activates for 30 minutes when:

```python
def should_activate_tier2(news_signals: List[Signal]) -> bool:
    crypto_signals = [s for s in news_signals if s.is_crypto_relevant]
    
    if len(crypto_signals) < 2:
        return False
    
    has_authoritative = any(
        s.source_tier in ("S1", "S2") or s.follower_count >= 100_000
        for s in crypto_signals
    )
    
    return has_authoritative
```

---

## 6. Signal Classification (Two-Dimensional)

### 6.1 Why the Original 9-Category Taxonomy Failed

The original flat list had three fatal flaws:

1. **Mixed taxonomy**: Some categories described the *source* (wire_news) while others described *content type* (orderbook_imbalance). A Reuters tweet about a whale trade matched 3 categories simultaneously.
2. **Missing critical categories**: No slot for official data releases (Fed decisions, CPI), court rulings, or regulatory announcements — the most deterministic signals.
3. **"verified_insider" was unclassifiable programmatically**: Either empty (never triggered) or misclassified (anyone confident gets tagged) → corrupts learning data.

### 6.2 Solution: Two Independent Dimensions

Every signal gets TWO tags: one source tier (WHO said it) + one information type (WHAT kind of claim). This creates a 3D learning matrix: **source × info_type × market_type**.

### 6.3 Dimension 1: Source Credibility Tiers (Classified Programmatically)

| Tier | Name | Classification Rule | Starting Credibility |
|------|------|---------------------|---------------------|
| S1 | `official_primary` | Government agency, court, central bank, exchange, scheduled data release. URL or account matches known official list. | 0.95 |
| S2 | `wire_service` | Reuters, AP, AFP, Bloomberg terminal flash. Identified by RSS domain or known wire accounts. | 0.90 |
| S3 | `institutional_media` | BBC, NYT, WSJ, CNBC, CoinDesk, The Block, etc. Identified by RSS domain or verified org accounts. | 0.80 |
| S4 | `verified_expert` | Twitter verified + >50K followers + bio contains expert keywords (journalist, analyst, CEO, senator, etc.). | 0.65 |
| S5 | `market_derived` | Polymarket orderbook data, whale trade on-chain, volume spike, cross-market divergence. No human source — purely market data. | 0.70 |
| S6 | `unverified_social` | Everything else. Verified <50K, unverified high-engagement, anonymous accounts. | 0.30 |

### 6.4 Dimension 2: Information Types (Classified by LLM)

| Type | Name | Description | Example |
|------|------|-------------|---------|
| I1 | `deterministic_outcome` | Event already occurred or officially decided. Market should resolve to 0 or 1. | "Fed holds rates steady" for "Will Fed cut?" market |
| I2 | `strong_directional` | Credible evidence strongly shifting probability. Court filing, official statement, leaked doc. | "White House confirms veto of bill" |
| I3 | `weak_directional` | Evidence modestly shifting probability. Polls, analyst opinions, unnamed sources. | "Sources say negotiations stalling" |
| I4 | `sentiment_shift` | Social/market mood change without specific factual claim. | Bitcoin fear index drops to 15 |
| I5 | `contradictory` | Signal conflicts with other signals or market price. | Wire says deal done, official denies it |
| I6 | `price_action_only` | No news — just market data showing unusual movement. | $200K buy at 0.55 when market was at 0.50 |

### 6.5 Why Two Dimensions Matter

Source tier is classified **programmatically** (account handle / domain matching — no ambiguity). Information type is classified **by the LLM** (semantic judgment — "expected to hold" vs "holds" requires content understanding). This plays to each system's strengths.

The two-tag system enables granular learning:

| Scenario | Source | Info Type | What System Learns |
|----------|--------|-----------|-------------------|
| Fed announces rate hold | S1 | I1 | S1 × I1 on political = near-certain signal |
| Reuters: "Sources say deal stalling" | S2 | I3 | S2 × I3 on political = moderate signal |
| @elonmusk tweets about crypto | S4 | I4 | S4 × I4 on crypto = track if it moves markets |
| $200K whale buy on Polymarket | S5 | I6 | S5 × I6 on political = whale conviction signal |
| Random account: "BREAKING: Trump to sign EO" | S6 | I2 | S6 × I2 = strong claim, weak source — track accuracy |

With 6 × 6 × ~5 market types = 180 cells. Most will be sparse. System requires 5 samples per cell before activating lift calculations — empty cells don't influence decisions.

### 6.6 Source Classification Implementation

```python
OFFICIAL_SOURCES = {
    "twitter": {"@WhiteHouse", "@FederalReserve", "@SECGov",
                "@SecYellen", "@USTreasury"},
    "rss_domains": {"federalreserve.gov", "sec.gov", "bls.gov",
                    "whitehouse.gov", "supremecourt.gov", "treasury.gov"}
}

WIRE_SERVICES = {
    "twitter": {"@Reuters", "@AP", "@AFP", "@BNONews", "@business"},
    "rss_domains": {"reuters.com", "apnews.com", "afp.com"}
}

INSTITUTIONAL_MEDIA = {
    "twitter": {"@BBCBreaking", "@CNN", "@nytimes", "@WSJ",
                "@CNBC", "@CoinDesk", "@TheBlock__"},
    "rss_domains": {"bbc.com", "nytimes.com", "wsj.com",
                    "cnbc.com", "coindesk.com", "theblock.co"}
}

EXPERT_BIO_KEYWORDS = [
    "journalist", "reporter", "editor", "correspondent",
    "analyst", "researcher", "professor", "economist",
    "senator", "representative", "minister", "official",
    "ceo", "cto", "founder", "partner", "director",
    "crypto", "blockchain", "defi"
]


def classify_source_tier(signal: dict) -> str:
    if signal["source_type"] == "market_data":
        return "S5"

    if signal["source_type"] == "rss":
        domain = signal["domain"]
        if domain in OFFICIAL_SOURCES["rss_domains"]: return "S1"
        if domain in WIRE_SERVICES["rss_domains"]: return "S2"
        if domain in INSTITUTIONAL_MEDIA["rss_domains"]: return "S3"
        return "S6"

    if signal["source_type"] == "twitter":
        handle = signal["account_handle"]
        if handle in OFFICIAL_SOURCES["twitter"]: return "S1"
        if handle in WIRE_SERVICES["twitter"]: return "S2"
        if handle in INSTITUTIONAL_MEDIA["twitter"]: return "S3"

        if signal.get("is_verified") and signal.get("follower_count", 0) >= 50000:
            bio = signal.get("bio", "").lower()
            if any(kw in bio for kw in EXPERT_BIO_KEYWORDS):
                return "S4"
        return "S6"

    return "S6"
```

These lists are maintained in `config/known_sources.yaml` and reviewed monthly (see Runbook 18.2).

### 6.7 Signal Tracker (Learning)

```python
@dataclass
class SignalTracker:
    source_tier: str                  # S1-S6
    info_type: str                    # I1-I6
    market_type: str                  # political, crypto_15m, sports, etc.
    present_in_winning_trades: int = 0
    present_in_losing_trades: int = 0
    absent_in_winning_trades: int = 0
    absent_in_losing_trades: int = 0

    @property
    def lift(self) -> float:
        total_present = self.present_in_winning_trades + self.present_in_losing_trades
        total_absent = self.absent_in_winning_trades + self.absent_in_losing_trades
        if total_present < 5 or total_absent < 5:
            return 1.0
        win_rate_present = self.present_in_winning_trades / total_present
        win_rate_absent = self.absent_in_winning_trades / total_absent
        if win_rate_absent == 0:
            return 1.0
        return win_rate_present / win_rate_absent

    @property
    def weight(self) -> float:
        raw = 1.0 + (self.lift - 1.0) * 0.3
        return max(0.8, min(1.2, raw))
```

### 6.8 Known Weaknesses (Monitor During Paper Trading)

1. **Static account lists.** New high-credibility accounts won't be captured until list update. Review monthly.
2. **50K follower threshold for S4 is arbitrary.** Track S6 signals with expert-like lift → signal to lower threshold.
3. **LLM info-type classification adds dependency.** Sample 10% of trades, manually verify. Target <15% error rate.
4. **3D matrix sparsity.** 180 cells, many empty. System handles via 5-sample minimum for lift activation.

---

## 7. Learning System

Three layers, each operating on different timescales.

### 7.1 What Gets Recorded (Every Trade, Including Skipped)

```python
@dataclass
class TradeRecord:
    record_id: str                    # UUID
    experiment_run: str               # e.g. "grok-4.1-fast_20260220"
    timestamp: datetime
    model_used: str                   # "grok-4.1-fast"
    
    market_id: str
    market_question: str
    market_type: str                  # "political", "crypto_15m", "economic", etc.
    resolution_window_hours: float
    tier: int                         # 1 or 2
    
    grok_raw_probability: float
    grok_raw_confidence: float
    grok_reasoning: str
    grok_signal_types: List[dict]     # [{"source_tier": "S2", "info_type": "I2", "content": "..."}]
    
    calibration_adjustment: float
    market_type_adjustment: float
    signal_weight_adjustment: float
    final_adjusted_probability: float
    final_adjusted_confidence: float
    
    market_price_at_decision: float
    orderbook_depth_usd: float
    fee_rate: float
    calculated_edge: float
    
    action: str                       # "BUY_YES", "BUY_NO", "SKIP"
    skip_reason: Optional[str]        # "edge_below_threshold", "daily_cap", "observe_only", etc.
    position_size_usd: float
    kelly_fraction_used: float
    
    actual_outcome: Optional[bool]
    pnl: Optional[float]
    brier_score_raw: Optional[float]        # Measures Grok's raw accuracy
    brier_score_adjusted: Optional[float]   # Measures full system accuracy (go-live criterion)
    resolved_at: Optional[datetime]
    unrealized_adverse_move: Optional[float] # For cooldown on unresolved trades
    
    voided: bool = False
    void_reason: Optional[str] = None
```

**Critical: Record SKIPPED trades too.** This gives counterfactual data: "Would I have made money if my threshold was lower?" Without this, learning is biased to the subset of trades taken.

### 7.2 Layer 1: Bayesian Calibration

**Problem:** When Grok says "80% probability," it might be right only 60% of the time.

**Model dependency: FULL** — resets on model swap (Section 8).

```python
@dataclass
class CalibrationBucket:
    bucket_range: Tuple[float, float]
    alpha: float = 1.0
    beta: float = 1.0
    
    @property
    def expected_accuracy(self) -> float:
        return self.alpha / (self.alpha + self.beta)
    
    @property
    def sample_count(self) -> int:
        return int(self.alpha + self.beta - 2)
    
    @property
    def uncertainty(self) -> float:
        from scipy.stats import beta as beta_dist
        low, high = beta_dist.ppf([0.025, 0.975], self.alpha, self.beta)
        return high - low
    
    def update(self, was_correct: bool, recency_weight: float = 1.0):
        if was_correct:
            self.alpha += recency_weight
        else:
            self.beta += recency_weight
    
    def get_correction(self) -> float:
        if self.sample_count < 10:
            return 0.0
        bucket_midpoint = (self.bucket_range[0] + self.bucket_range[1]) / 2
        correction = self.expected_accuracy - bucket_midpoint
        certainty = max(0, 1 - self.uncertainty * 2)
        return correction * certainty

CALIBRATION_BUCKETS = [
    CalibrationBucket((0.50, 0.60)),
    CalibrationBucket((0.60, 0.70)),
    CalibrationBucket((0.70, 0.80)),
    CalibrationBucket((0.80, 0.90)),
    CalibrationBucket((0.90, 0.95)),
    CalibrationBucket((0.95, 1.00)),
]
```

### 7.3 Layer 2: Market-Type Performance

**Model dependency: PARTIAL** — dampened on model swap.

```python
@dataclass
class MarketTypePerformance:
    market_type: str
    total_trades: int = 0
    total_pnl: float = 0.0
    brier_scores: List[float] = field(default_factory=list)
    total_observed: int = 0
    counterfactual_pnl: float = 0.0
    
    @property
    def avg_brier(self) -> float:
        if not self.brier_scores:
            return 0.25
        weights = [0.95 ** i for i in range(len(self.brier_scores) - 1, -1, -1)]
        return sum(b * w for b, w in zip(self.brier_scores, weights)) / sum(weights)
    
    @property
    def edge_adjustment(self) -> float:
        if self.total_trades < 15:
            return 0.0
        if self.avg_brier > 0.30: return 0.05
        elif self.avg_brier > 0.25: return 0.03
        elif self.avg_brier > 0.20: return 0.01
        else: return 0.0
    
    @property 
    def should_disable(self) -> bool:
        return self.total_trades >= 30 and self.total_pnl < -0.15 * abs(self.total_trades)
```

### 7.4 Layer 3: Signal-Type Weighting

**Model dependency: NONE** — preserved on model swap. See Section 6 for full taxonomy and tracker implementation.

### 7.5 Combined Adjustment Pipeline

```python
def adjust_prediction(
    grok_probability: float,
    grok_confidence: float,
    market_type: str,
    signal_tags: List[dict],
    calibration_buckets: List[CalibrationBucket],
    market_type_performance: Dict[str, MarketTypePerformance],
    signal_trackers: Dict[Tuple[str, str, str], SignalTracker],
) -> Tuple[float, float, float]:
    
    # Step 1: Bayesian calibration
    bucket = find_bucket(grok_confidence, calibration_buckets)
    calibration_correction = bucket.get_correction()
    adjusted_confidence = max(0.50, min(0.99, grok_confidence + calibration_correction))
    
    # Step 2: Signal-type weighting
    if signal_tags:
        weights = []
        for tag in signal_tags:
            key = (tag["source_tier"], tag["info_type"], market_type)
            tracker = signal_trackers.get(key)
            if tracker:
                weights.append(tracker.weight)
        if weights:
            avg_w = sum(weights) / len(weights)
            adjusted_confidence = max(0.50, min(0.99, adjusted_confidence + (avg_w - 1.0) * 0.1))
    
    # Step 3: Adjust probability via SHRINKAGE toward 0.50
    # Overconfidence (correction < 0) → shrink probability toward 0.50 (less extreme)
    # Underconfidence (correction > 0) → push probability away from 0.50 (more extreme)
    # This correctly handles both sides of 0.50:
    #   p=0.80, overconfident → shrinks to 0.76 (toward 0.50) ✓
    #   p=0.20, overconfident → shrinks to 0.24 (toward 0.50) ✓
    bucket_midpoint = (bucket.bucket_range[0] + bucket.bucket_range[1]) / 2
    if bucket_midpoint > 0 and bucket.sample_count >= 10:
        shrinkage_factor = bucket.expected_accuracy / bucket_midpoint
        adjusted_probability = 0.5 + (grok_probability - 0.5) * shrinkage_factor
        adjusted_probability = max(0.01, min(0.99, adjusted_probability))
    else:
        adjusted_probability = grok_probability  # No data yet, trust Grok as-is
    
    # Step 4: Market-type edge penalty
    mtype_perf = market_type_performance.get(market_type)
    extra_edge = mtype_perf.edge_adjustment if mtype_perf else 0.0
    
    # Step 5: Temporal confidence decay
    # Confidence should decrease as time passes without new signals arriving.
    # A market with no fresh signals (<30 min old) is relying on stale information
    # and the confidence should reflect that. Conversely, I1 (deterministic) signals
    # very close to resolution are MORE reliable, not less.
    if signal_tags:
        freshest_signal_age_hours = min(
            (datetime.utcnow() - tag.get("timestamp", datetime.utcnow())).total_seconds() / 3600
            for tag in signal_tags
        ) if any(tag.get("timestamp") for tag in signal_tags) else 2.0
        
        has_deterministic = any(tag.get("info_type") == "I1" for tag in signal_tags)
        
        if has_deterministic and freshest_signal_age_hours < 0.5:
            # Recent deterministic signal — boost confidence slightly
            adjusted_confidence = min(0.99, adjusted_confidence * 1.05)
        elif freshest_signal_age_hours > 1.0:
            # All signals are >1h old — decay confidence
            decay = max(0.85, 1.0 - 0.05 * (freshest_signal_age_hours - 1.0))
            adjusted_confidence = max(0.50, adjusted_confidence * decay)
    
    return adjusted_probability, adjusted_confidence, extra_edge
```

### 7.6 Learning Update Flow (After Every Resolution)

```python
def on_trade_resolved(record: TradeRecord):
    if record.voided:
        return
    
    actual = 1.0 if record.actual_outcome else 0.0
    record.brier_score_raw = (record.grok_raw_probability - actual) ** 2
    record.brier_score_adjusted = (record.final_adjusted_probability - actual) ** 2
    
    # IMPORTANT: Use RAW probability for calibration feedback.
    # Using adjusted would create a self-referencing loop where calibration
    # converges to a fixed point unrelated to Grok's actual accuracy.
    raw_predicted_yes = record.grok_raw_probability > 0.5
    was_correct_raw = raw_predicted_yes == record.actual_outcome
    
    # For market-type and signal trackers, use ADJUSTED (system-level accuracy)
    adjusted_predicted_yes = record.final_adjusted_probability > 0.5
    was_correct_adjusted = adjusted_predicted_yes == record.actual_outcome
    
    # Layer 1: Calibration (uses RAW correctness — measures Grok, not system)
    bucket = find_bucket(record.grok_raw_confidence, CALIBRATION_BUCKETS)
    recency = 0.95 ** days_since(record.timestamp)
    bucket.update(was_correct_raw, recency_weight=recency)
    
    # Layer 2: Market-type (uses ADJUSTED correctness — measures system)
    mtype = market_type_performance[record.market_type]
    mtype.total_trades += 1
    mtype.brier_scores.append(record.brier_score_adjusted)
    if record.action != "SKIP":
        mtype.total_pnl += record.pnl or 0.0
    
    # Layer 3: Signal trackers (uses ADJUSTED — measures system-level value of signals)
    present_combos = {(t["source_tier"], t["info_type"]) for t in record.grok_signal_types}
    for combo in get_all_observed_combos(record.market_type):
        key = (combo[0], combo[1], record.market_type)
        tracker = signal_trackers.setdefault(key, SignalTracker(*key))
        present = combo in present_combos
        if present and was_correct_adjusted:     tracker.present_in_winning_trades += 1
        elif present and not was_correct_adjusted: tracker.present_in_losing_trades += 1
        elif not present and was_correct_adjusted: tracker.absent_in_winning_trades += 1
        else:                                      tracker.absent_in_losing_trades += 1
    
    # Counterfactual
    if record.action == "SKIP":
        mtype.total_observed += 1
        mtype.counterfactual_pnl += calculate_hypothetical_pnl(record)
    
    # Persist
    db.save_all()
```

### 7.7 Experiment Runs

```python
@dataclass
class ExperimentRun:
    run_id: str
    started_at: datetime
    ended_at: Optional[datetime]
    config_snapshot: dict
    description: str
    model_used: str
    include_in_learning: bool = True
    total_trades: int = 0
    total_pnl: float = 0.0
    avg_brier: float = 0.0
    sharpe_ratio: float = 0.0
```

---

## 8. Model Swap Protocol

### 8.1 When to Swap

- New model released with claimed reasoning improvements (e.g., Grok 5 drops)
- Current model shows degrading calibration (Brier trending >0.30 over 50+ trades)
- Price change makes another model better value

### 8.2 Pre-Swap Validation

**DO NOT swap on release day.** Wait 1-2 weeks for community benchmarks and stability.

Optional parallel test: for 20-30 markets, send identical context to both old and new model. Compare:
- If raw probabilities within ±5% on >80% of markets → low-risk swap
- If significant divergence → investigate before swapping

### 8.3 Execute the Swap

```bash
python manage.py model_swap \
  --old-model "grok-4.1-fast" \
  --new-model "grok-5-fast" \
  --reason "Grok 5 released, 30% reasoning improvement claimed, 2 weeks stable"
```

This triggers `handle_model_swap()`:

```python
def handle_model_swap(old_model: str, new_model: str, reason: str):
    swap = ModelSwapEvent(datetime.utcnow(), old_model, new_model, reason,
        f"{new_model}_{datetime.utcnow().strftime('%Y%m%d')}")
    db.save_model_swap(swap)
    
    start_experiment(
        run_id=swap.experiment_run_started,
        description=f"Model swap: {old_model} → {new_model}. {reason}",
        config=current_config, model=new_model,
    )
    
    # RESET calibration (FULL model dependency)
    for bucket in CALIBRATION_BUCKETS:
        bucket.alpha = 1.0
        bucket.beta = 1.0
    
    # DAMPEN market-type (PARTIAL dependency)
    for mtype in market_type_performance.values():
        mtype.brier_scores = mtype.brier_scores[-15:]
    
    # PRESERVE signal trackers (NO dependency)
```

| Layer | Dependency | Action |
|-------|-----------|--------|
| Bayesian Calibration | **FULL** | Reset to priors |
| Market-Type Performance | **PARTIAL** | Keep last 15 Brier scores, reset disable flags |
| Signal-Type Weighting | **NONE** | Preserve fully |

### 8.4 Post-Swap Monitoring (First 30 Trades)

- System is automatically conservative (no calibration adjustments active — priors return 0 correction)
- Watch Grok's raw probability vs market price — are edges similar to before?
- Watch for JSON parsing failures (new model may format differently)
- After 30 trades: is new model's Brier tracking better or worse than old at same count?

### 8.5 Rollback Trigger

If Brier score >0.35 after 30 trades with new model → rollback to old model. Same swap procedure in reverse.

---

## 9. Monk Mode v2

### 9.1 Configuration

```python
@dataclass
class MonkModeConfig:
    tier1_daily_trade_cap: int = 5
    tier2_daily_trade_cap: int = 3
    daily_loss_limit_pct: float = 0.05   # -5% → stop for day
    weekly_loss_limit_pct: float = 0.10  # -10% → stop for week
    consecutive_loss_cooldown: int = 3   # 3 losses → 2h pause
    cooldown_duration_hours: float = 2.0
    daily_api_budget_usd: float = 8.0
    max_position_pct: float = 0.08       # 8% per trade
    max_total_exposure_pct: float = 0.30 # 30% total
    kelly_fraction: float = 0.25         # Quarter Kelly
```

### 9.2 Enforcement

```python
def check_monk_mode(config, trade_signal, portfolio, today_trades, week_trades, api_spend) -> Tuple[bool, Optional[str]]:
    tier = trade_signal.tier
    tier_trades = [t for t in today_trades if t.tier == tier and t.action != "SKIP"]
    
    cap = config.tier1_daily_trade_cap if tier == 1 else config.tier2_daily_trade_cap
    if len(tier_trades) >= cap:
        return False, f"tier{tier}_daily_cap_reached"
    
    today_pnl = sum(t.pnl or 0 for t in today_trades if t.pnl is not None)
    if today_pnl < -config.daily_loss_limit_pct * portfolio.total_equity:
        return False, "daily_loss_limit"
    
    # Weekly loss limit (prevents sequence of -4.9% days slipping through daily limit)
    week_pnl = sum(t.pnl or 0 for t in week_trades if t.pnl is not None)
    if week_pnl < -config.weekly_loss_limit_pct * portfolio.total_equity:
        return False, "weekly_loss_limit"
    
    # Consecutive loss cooldown
    # Problem: Tier 1 trades on 12-24h markets won't have pnl filled yet.
    # Solution: Count both (a) resolved losses AND (b) unrealized adverse moves
    # (market price moved >10% against entry) as "loss signals."
    recent = sorted(
        [t for t in today_trades if t.action != "SKIP"],
        key=lambda t: t.timestamp, reverse=True
    )
    consecutive_adverse = 0
    for t in recent:
        if t.pnl is not None and t.pnl < 0:
            consecutive_adverse += 1
        elif t.pnl is None and t.unrealized_adverse_move and t.unrealized_adverse_move > 0.10:
            # Market moved >10% against us (unrealized) — count as adverse signal
            consecutive_adverse += 1
        else:
            break  # Streak broken
    
    if consecutive_adverse >= config.consecutive_loss_cooldown:
        cooldown_end = recent[0].timestamp + timedelta(hours=config.cooldown_duration_hours)
        if datetime.utcnow() < cooldown_end:
            return False, f"cooldown_until_{cooldown_end}"
    
    # Exposure limit
    open_exposure = sum(p.current_value for p in portfolio.open_positions)
    if open_exposure + trade_signal.position_size > config.max_total_exposure_pct * portfolio.total_equity:
        return False, "max_exposure"
    
    if api_spend > config.daily_api_budget_usd:
        return False, "api_budget_exceeded"
    
    return True, None
```

### 9.3 Observe-Only Mode (After Daily Cap Hit)

Once the daily trade cap is reached, the system switches to observe-only:

```python
def get_scan_mode(today_trades: List[TradeRecord], config: MonkModeConfig) -> str:
    tier1_executed = len([t for t in today_trades if t.tier == 1 and t.action != "SKIP"])
    
    if tier1_executed >= config.tier1_daily_trade_cap:
        return "observe_only"
    return "active"

# In the scan loop:
if scan_mode == "observe_only":
    # Still pull RSS headlines (free)
    # Still pull Twitter signals (cheap — $0.0075/search)
    # DON'T call Grok (saves ~$0.0004/call × remaining scans)
    # Record all as SKIP with skip_reason="daily_cap_observe_only"
    # Use market price movement post-skip for counterfactual learning
```

**Cost savings:** If cap hit by noon, saves ~$0.10/day (~$3/month) in Grok calls while preserving the learning data stream.

---

## 10. Trade Decision Engine

### 10.1 Trade Ranking (Not First-Come-First-Served)

The scan cycle evaluates all candidate markets simultaneously. Instead of executing the first market that passes the threshold, collect all candidates and rank them:

```python
def select_best_trades(
    candidates: List[TradeCandidate],
    remaining_cap: int,
    open_positions: List[Position],
    bankroll: float,
) -> Tuple[List[TradeCandidate], List[TradeCandidate]]:
    """
    Returns: (to_execute, to_skip)
    Ranks by expected value score. Checks cluster exposure limits.
    """
    
    for c in candidates:
        # Score = edge × confidence × time_value
        # Faster resolution = faster capital recycling = higher effective return
        time_value = 1.0 / max(c.resolution_hours, 0.5)
        c.score = c.calculated_edge * c.adjusted_confidence * time_value
    
    ranked = sorted(candidates, key=lambda c: c.score, reverse=True)
    
    # Detect correlated markets
    clusters = detect_market_clusters(candidates)
    
    to_execute = []
    to_skip = []
    
    for c in ranked:
        if len(to_execute) >= remaining_cap:
            c.skip_reason = "ranked_below_cutoff"
            to_skip.append(c)
        elif not check_cluster_exposure(c, clusters.get(c.market_id, c.market_id),
                                         open_positions, to_execute, clusters, bankroll):
            c.skip_reason = "cluster_exposure_limit"
            to_skip.append(c)
        else:
            c.market_cluster_id = clusters.get(c.market_id)
            to_execute.append(c)
    
    return to_execute, to_skip
```

**Why this matters:** If you have a 12% edge / 85% confidence political market resolving in 2h and a 5% edge / 66% confidence sports market resolving in 18h, the political one scores ~10x higher. Without ranking, whichever the scanner found first would get the slot.

### 10.2 Correlated Market Detection

Polymarket frequently lists multiple markets about the same underlying event: "Will the Fed cut rates?", "Will the Fed cut by 50bps?", "Will the Fed hold rates?" — all resolve on the same announcement. Without correlation detection, the system could take 8% positions on all three, concentrating 24% of bankroll on a single event.

```python
from collections import defaultdict

def detect_market_clusters(candidates: List[TradeCandidate]) -> Dict[str, str]:
    """
    Assign cluster_id to correlated markets. Markets sharing a cluster
    have their combined exposure checked against a single-event limit.
    
    Returns: {market_id: cluster_id}
    """
    clusters = {}
    
    # Method 1: Resolution time + category proximity
    # Markets resolving within 1h of each other in the same category = likely correlated
    by_category = defaultdict(list)
    for c in candidates:
        by_category[c.market_type].append(c)
    
    cluster_id_counter = 0
    for category, markets in by_category.items():
        markets_sorted = sorted(markets, key=lambda m: m.resolution_time)
        for i, m1 in enumerate(markets_sorted):
            if m1.market_id in clusters:
                continue
            cluster_id = f"cluster_{cluster_id_counter}"
            clusters[m1.market_id] = cluster_id
            for m2 in markets_sorted[i+1:]:
                if m2.market_id in clusters:
                    continue
                time_diff = abs((m2.resolution_time - m1.resolution_time).total_seconds())
                if time_diff <= 3600:  # Within 1 hour
                    keyword_overlap = _keyword_overlap(m1.keywords, m2.keywords)
                    if keyword_overlap >= 0.5:  # 50%+ shared keywords
                        clusters[m2.market_id] = cluster_id
            cluster_id_counter += 1
    
    return clusters


def _keyword_overlap(kw1: List[str], kw2: List[str]) -> float:
    """Jaccard similarity of keyword sets."""
    s1, s2 = set(k.lower() for k in kw1), set(k.lower() for k in kw2)
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


MAX_CLUSTER_EXPOSURE_PCT = 0.12  # Max 12% of bankroll on correlated markets

def check_cluster_exposure(
    candidate: TradeCandidate,
    cluster_id: str,
    open_positions: List[Position],
    pending_executions: List[TradeCandidate],
    clusters: Dict[str, str],
    bankroll: float,
) -> bool:
    """Returns True if adding this trade keeps cluster exposure within limits."""
    same_cluster_ids = {mid for mid, cid in clusters.items() if cid == cluster_id}
    
    existing_exposure = sum(
        p.current_value for p in open_positions if p.market_id in same_cluster_ids
    )
    pending_exposure = sum(
        c.position_size for c in pending_executions if c.market_id in same_cluster_ids
    )
    
    total = existing_exposure + pending_exposure + candidate.position_size
    return total <= MAX_CLUSTER_EXPOSURE_PCT * bankroll
```

The `select_best_trades()` function checks cluster limits before adding each trade to the execution list. If a trade would breach the cluster limit, it's skipped even if its score is high.

### 10.3 Edge Calculation

```python
def calculate_edge(adjusted_prob: float, market_price: float, fee_rate: float) -> float:
    """Net edge after fees."""
    raw_edge = abs(adjusted_prob - market_price)
    return raw_edge - fee_rate
```

### 10.4 Kelly Sizing

```python
def kelly_size(adjusted_prob: float, market_price: float, side: str,
               bankroll: float, kelly_fraction: float = 0.25, 
               max_position_pct: float = 0.08) -> float:
    """
    Correct Kelly for binary prediction markets.
    
    Buying YES at price p: payout = 1/p on win, lose stake on loss.
      f* = (prob * (1 - price) - (1 - prob) * price) / (1 - price)
      Simplified: f* = (prob - price) / (1 - price)
    
    Buying NO at price p: equivalent to buying YES at (1-p).
      f* = ((1 - prob) - (1 - price)) / price
      Simplified: f* = (price - prob) / price
    
    Example impact vs naive "f* = edge":
      Buy YES at 0.90, true prob 0.95:
        Naive:   f* = 0.05 (5%)
        Correct: f* = (0.95 - 0.90) / (1 - 0.90) = 0.50 (50%)
      The market pays 10:1 on a 5% edge — Kelly wants heavy exposure.
      Quarter Kelly + 8% cap still limits actual position to 8%, but the
      RANKING of trades changes: high-price YES bets rank much higher.
    """
    if side == "BUY_YES":
        if adjusted_prob <= market_price:
            return 0.0
        full_kelly = (adjusted_prob - market_price) / (1 - market_price)
    elif side == "BUY_NO":
        if adjusted_prob >= market_price:
            return 0.0
        full_kelly = (market_price - adjusted_prob) / market_price
    else:
        return 0.0
    
    quarter_kelly = full_kelly * kelly_fraction
    position = quarter_kelly * bankroll
    max_position = max_position_pct * bankroll
    
    return min(position, max_position)
```

---

## 11. Data Pipeline

### 11.1 Market-to-Signal Keyword Extraction

Before pulling signals for a market, the system must extract search keywords from the market question. This is the core matching problem — bad keywords produce irrelevant signals.

**Hybrid approach (regex first, LLM fallback):**

```python
import re

# Common entity patterns in Polymarket questions
ENTITY_PATTERNS = [
    r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',          # "Donald Trump", "Elon Musk"
    r'\b[A-Z]{2,}\b',                           # "BTC", "ETH", "FBI", "SEC"
    r'\$[A-Z]{1,5}\b',                          # "$BTC", "$AAPL"
    r'\b(?:Fed|ECB|SEC|FDA|WHO|NATO|EU)\b',     # Known acronyms
]

# Market type → supplementary keyword map
KEYWORD_SUPPLEMENTS = {
    "fed_rate":    ["federal reserve", "interest rate", "FOMC"],
    "crypto":      ["bitcoin", "ethereum", "crypto"],
    "election":    ["election", "polls", "vote"],
    "scotus":      ["supreme court", "ruling", "SCOTUS"],
    "regulation":  ["regulation", "executive order", "legislation"],
}

# Cache: market_id → keywords (generate once per market, reuse across scans)
_keyword_cache: Dict[str, List[str]] = {}


def extract_keywords(market_id: str, market_question: str, market_type: str) -> List[str]:
    """Extract 3-5 search keywords from market question. Cached per market_id."""
    if market_id in _keyword_cache:
        return _keyword_cache[market_id]
    
    # Step 1: Regex extraction
    entities = set()
    for pattern in ENTITY_PATTERNS:
        entities.update(re.findall(pattern, market_question))
    
    # Step 2: Add market-type supplements
    for key, supplements in KEYWORD_SUPPLEMENTS.items():
        if key in market_type or any(s.lower() in market_question.lower() for s in supplements):
            entities.update(supplements[:2])
    
    # Step 3: If regex found 2+ entities, use them (no LLM cost)
    if len(entities) >= 2:
        keywords = list(entities)[:5]
        _keyword_cache[market_id] = keywords
        return keywords
    
    # Step 4: LLM fallback for complex/ambiguous questions
    # Costs ~$0.0002 per call, cached so only called once per market
    keywords = await grok_extract_keywords(market_question)
    _keyword_cache[market_id] = keywords
    return keywords


async def grok_extract_keywords(question: str) -> List[str]:
    """Ask Grok for search keywords. Only called when regex fails."""
    prompt = f"""Extract 3-5 Twitter/news search keywords for this prediction market question.
Return ONLY a JSON array of strings. No explanation.

Question: "{question}"

Example: ["Trump", "executive order", "immigration"]"""
    
    response = await grok_client.complete(prompt, max_tokens=50)
    return json.loads(response)  # Wrapped in parse_json_safe() in production
```

**Cost impact:** Regex handles ~70% of markets (named entities are usually explicit). The remaining 30% need one Grok call each — but results are cached per market_id, so each market only costs $0.0002 *once*, not per scan cycle. Negligible monthly impact (~$0.50).

### 11.2 TwitterAPI.io Integration

```python
class TwitterDataPipeline:
    BASE_URL = "https://api.twitterapi.io/twitter"
    
    async def get_signals_for_market(self, keywords: List[str], max_tweets: int = 50) -> List[Signal]:
        raw_tweets = await self._search_tweets(
            query=" OR ".join(keywords), max_results=max_tweets, recency="2h")
        
        # Pre-filter (no LLM cost)
        filtered = [t for t in raw_tweets 
                    if t.author.followers_count >= 1000 
                    and t.engagement_score >= 10
                    and not self._is_bot_account(t.author)]
        
        deduplicated = self._deduplicate_by_content_similarity(filtered)
        
        # Classify source tier (programmatic)
        scored = []
        for tweet in deduplicated:
            source_tier = classify_source_tier({
                "source_type": "twitter",
                "account_handle": f"@{tweet.author.screen_name}",
                "is_verified": tweet.author.verified,
                "follower_count": tweet.author.followers_count,
                "bio": tweet.author.bio or "",
            })
            scored.append((SOURCE_TIER_CREDIBILITY[source_tier], source_tier, tweet))
        
        scored.sort(reverse=True)
        return [Signal(source="twitter", source_tier=st, info_type=None,
                       content=tw.text[:280], credibility=cred,
                       author=tw.author.screen_name, followers=tw.author.followers_count,
                       engagement=tw.engagement_score, timestamp=tw.created_at)
                for cred, st, tw in scored[:10]]

SOURCE_TIER_CREDIBILITY = {"S1": 0.95, "S2": 0.90, "S3": 0.80, "S4": 0.65, "S5": 0.70, "S6": 0.30}
```

### 11.3 RSS News Pipeline — Design Rationale

**Why RSS headlines are sufficient (and a €50/month news API is not justified yet):**

RSS gives headlines + sometimes a 1-2 sentence summary. This is a real limitation. But the system doesn't need to understand articles — it needs to detect that *something happened* and match it to a market. Headlines are sufficient >90% of the time:

- "Fed holds rates at 5.25%" → contains everything needed for "Will Fed cut?" market
- "Supreme Court rules 6-3 in favor of..." → deterministic resolution signal
- "Trump signs executive order on..." → direct market resolution

Where headlines fail: nuanced signals like "Sources say negotiations stalling" where the full article reveals *which* sources, how confident the journalist is. But Twitter signals from verified journalists (S4) fill this gap — you're already paying for that data.

**RSS has no rate limits.** Public RSS feeds are just XML endpoints. Reuters, AP, BBC update every few minutes. The only practical limitation: Bloomberg gates full content behind paywalls, but the headline still works as a trigger.

**Upgrade path:** Track a field `headline_only_signal: bool` on trades. If after paper trading >15% of losing trades had headline-only signals where the full article would have changed the decision, *then* a news API pays for itself. Until that's proven, it nearly triples OpEx for unproven value.

```python
RSS_FEEDS = {
    "reuters_top":      {"url": "https://feeds.reuters.com/reuters/topNews",       "domain": "reuters.com"},
    "reuters_business": {"url": "https://feeds.reuters.com/reuters/businessNews",   "domain": "reuters.com"},
    "ap_top":           {"url": "https://rsshub.app/apnews/topics/apf-topnews",    "domain": "apnews.com"},
    "bbc_world":        {"url": "http://feeds.bbci.co.uk/news/world/rss.xml",      "domain": "bbc.com"},
    "coindesk":         {"url": "https://www.coindesk.com/arc/outboundfeeds/rss/",  "domain": "coindesk.com"},
}

class RSSPipeline:
    def __init__(self):
        # Use dict with timestamps instead of unbounded set (prevents memory leak)
        # Entries older than 24h are pruned on each scan
        self.seen_headlines: Dict[str, datetime] = {}
    
    def _prune_old_headlines(self):
        cutoff = datetime.utcnow() - timedelta(hours=24)
        self.seen_headlines = {
            h: ts for h, ts in self.seen_headlines.items() if ts > cutoff
        }
    
    async def get_breaking_news(self) -> List[Signal]:
        self._prune_old_headlines()
        signals = []
        for feed_name, cfg in RSS_FEEDS.items():
            try:
                feed = feedparser.parse(cfg["url"])
                for entry in feed.entries[:10]:
                    headline = entry.title.strip()
                    if headline in self.seen_headlines:
                        continue
                    self.seen_headlines[headline] = datetime.utcnow()
                    
                    published = parse_date(entry.get("published", ""))
                    if published and (datetime.utcnow() - published).total_seconds() > 7200:
                        continue
                    
                    source_tier = classify_source_tier({"source_type": "rss", "domain": cfg["domain"]})
                    signals.append(Signal(
                        source="rss", source_tier=source_tier, info_type=None,
                        content=headline, credibility=SOURCE_TIER_CREDIBILITY[source_tier],
                        author=feed_name, followers=0, engagement=0, timestamp=published,
                        headline_only=True,
                    ))
            except Exception:
                continue
        return signals
```

### 11.4 Context Compression (Grok Prompt)

```python
def build_grok_context(market: Market, twitter_signals: List[Signal],
                       rss_signals: List[Signal], orderbook: OrderBook) -> str:
    all_signals = sorted(twitter_signals + rss_signals, key=lambda s: s.credibility, reverse=True)[:7]
    
    signal_text = "\n".join([
        f"- [Source: {s.source_tier}] ({s.credibility:.0%}) {s.content}"
        for s in all_signals
    ])
    
    bid_depth = sum(orderbook.bids[:5])
    ask_depth = sum(orderbook.asks[:5])
    book_skew = bid_depth / max(ask_depth, 0.01)
    
    return f"""MARKET: {market.question}
CURRENT PRICE: YES={market.yes_price:.3f} NO={market.no_price:.3f}
RESOLUTION: {market.resolution_time.isoformat()} ({market.hours_to_resolution:.1f}h from now)
VOLUME 24H: ${market.volume_24h:,.0f}
LIQUIDITY: ${market.liquidity:,.0f}
ORDER BOOK: Bid depth ${bid_depth:,.0f} / Ask depth ${ask_depth:,.0f} (skew {book_skew:.2f}x)

SIGNALS (ranked by credibility, source tier in brackets):
{signal_text}

TASK: Estimate the TRUE probability that this market resolves YES.
Consider signal credibility, potential contradictions, and orderbook state.
Be conservative — overconfidence destroys capital.

For each signal you rely on, classify its information type:
- I1: deterministic_outcome (event already happened / officially decided)
- I2: strong_directional (credible evidence strongly shifting probability)
- I3: weak_directional (polls, opinions, unnamed sources)
- I4: sentiment_shift (mood change, no specific factual claim)
- I5: contradictory (conflicts with other signals or market price)

Respond ONLY with valid JSON:
{{"estimated_probability": 0.XX, "confidence": 0.XX, "reasoning": "...", "key_signals_used": ["..."], "contradictions": ["..."], "signal_info_types": {{"signal_description": "I1|I2|I3|I4|I5"}}}}"""
```

### 11.5 Grok Response Parsing & Error Handling

LLMs frequently return malformed JSON. Expect 2-5% of calls to fail. Without handling, those markets silently get no analysis.

```python
import json
import re

MAX_RETRIES = 2
REQUIRED_FIELDS = {"estimated_probability", "confidence", "reasoning", "signal_info_types"}


async def call_grok_with_retry(context: str, market_id: str) -> Optional[dict]:
    """
    Call Grok with structured retry and fallback parsing.
    Returns parsed dict or None (skip this market this cycle).
    """
    for attempt in range(MAX_RETRIES + 1):
        try:
            raw_response = await grok_client.complete(context, max_tokens=500)
            parsed = parse_json_safe(raw_response)
            
            if not parsed:
                log.warning("grok_parse_failed", market_id=market_id, attempt=attempt,
                            raw_preview=raw_response[:200])
                continue
            
            # Validate required fields exist and have correct types
            missing = REQUIRED_FIELDS - set(parsed.keys())
            if missing:
                log.warning("grok_missing_fields", market_id=market_id, missing=list(missing))
                continue
            
            if not (0.0 <= float(parsed["estimated_probability"]) <= 1.0):
                log.warning("grok_invalid_probability", market_id=market_id,
                            value=parsed["estimated_probability"])
                continue
            
            if not (0.0 <= float(parsed["confidence"]) <= 1.0):
                log.warning("grok_invalid_confidence", market_id=market_id,
                            value=parsed["confidence"])
                continue
            
            # Coerce types (LLMs sometimes return "0.75" as string)
            parsed["estimated_probability"] = float(parsed["estimated_probability"])
            parsed["confidence"] = float(parsed["confidence"])
            
            # Track success
            db.increment_api_cost("grok", tokens_in=len(context) // 4,
                                  tokens_out=len(raw_response) // 4)
            return parsed
            
        except Exception as e:
            log.error("grok_call_failed", market_id=market_id, attempt=attempt, error=str(e))
            if attempt < MAX_RETRIES:
                await asyncio.sleep(1.0 * (attempt + 1))  # Linear backoff
    
    # All retries exhausted — skip this market
    log.error("grok_all_retries_failed", market_id=market_id)
    db.record_parse_failure(market_id)
    return None


def parse_json_safe(raw: str) -> Optional[dict]:
    """Extract JSON from LLM output with common fallbacks."""
    # Try direct parse first
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass
    
    # Strip markdown code fences: ```json ... ``` or ``` ... ```
    cleaned = re.sub(r'^```(?:json)?\s*', '', raw.strip())
    cleaned = re.sub(r'\s*```$', '', cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # Find first { ... } block (handles preamble/postamble text)
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    
    return None
```

**Monitoring:** Track parse failure rate in `api_costs` table. If >10% of Grok calls fail, the prompt needs adjustment (likely the model is wrapping in markdown or adding explanatory text). Alert via Telegram if failure rate exceeds threshold.

---

## 12. Paper Trading

### 12.1 Realistic Execution Simulation

```python
def simulate_execution(side: str, price: float, size_usd: float,
                       execution_type: str, orderbook_depth: float) -> ExecutionResult:
    if execution_type == "taker":
        size_ratio = size_usd / max(orderbook_depth, 1)
        slippage = 0.005 + 0.01 * min(size_ratio, 1.0)
        executed_price = price * (1 + slippage) if side == "YES" else price * (1 - slippage)
        fill_probability = 1.0
    elif execution_type == "maker":
        price_distance = abs(price - 0.5)
        fill_probability = 0.4 + 0.4 * (1 - price_distance)
        executed_price = price
        slippage = 0.0
    
    return ExecutionResult(
        executed_price=min(0.99, max(0.01, executed_price)),
        slippage=slippage, fill_probability=fill_probability,
        filled=random.random() < fill_probability,
    )
```

### 12.2 Auto-Resolution

```python
async def auto_resolve_trades(db: Database):
    """Run every 5 minutes."""
    for trade in db.get_open_trades():
        market = await polymarket_client.get_market(trade.market_id)
        if market.resolved:
            resolve_trade(trade, market.resolution == "YES")
            on_trade_resolved(trade)
        elif market.market_type == "crypto_15m":
            if datetime.utcnow() > trade.expected_resolution_time:
                actual_price = await get_crypto_price_at(trade.crypto_asset, trade.expected_resolution_time)
                resolve_trade(trade, actual_price > trade.strike_price)
                on_trade_resolved(trade)
```

### 12.3 Void Mechanism

```python
def void_trade(trade_id: str, reason: str):
    """Exclude from learning without deleting. Use for bugs, API garbage, wrong resolutions."""
    record = db.get_record(trade_id)
    record.voided = True
    record.void_reason = reason
    db.save_record(record)
    recalculate_learning_from_scratch()
```

---

## 13. SQLite Schema

```sql
CREATE TABLE experiment_runs (
    run_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    config_snapshot TEXT NOT NULL,  -- JSON
    description TEXT,
    model_used TEXT NOT NULL,
    include_in_learning BOOLEAN DEFAULT TRUE,
    total_trades INTEGER DEFAULT 0,
    total_pnl REAL DEFAULT 0.0,
    avg_brier REAL DEFAULT 0.0,
    sharpe_ratio REAL DEFAULT 0.0
);

CREATE TABLE model_swaps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    old_model TEXT NOT NULL,
    new_model TEXT NOT NULL,
    reason TEXT,
    experiment_run_started TEXT REFERENCES experiment_runs(run_id)
);

CREATE TABLE trade_records (
    record_id TEXT PRIMARY KEY,
    experiment_run TEXT NOT NULL REFERENCES experiment_runs(run_id),
    timestamp TEXT NOT NULL,
    model_used TEXT NOT NULL,
    
    market_id TEXT NOT NULL,
    market_question TEXT NOT NULL,
    market_type TEXT NOT NULL,
    resolution_window_hours REAL,
    tier INTEGER NOT NULL,
    
    grok_raw_probability REAL NOT NULL,
    grok_raw_confidence REAL NOT NULL,
    grok_reasoning TEXT,
    grok_signal_types TEXT,       -- JSON: [{source_tier, info_type, content}]
    headline_only_signal BOOLEAN DEFAULT FALSE,
    
    calibration_adjustment REAL DEFAULT 0,
    market_type_adjustment REAL DEFAULT 0,
    signal_weight_adjustment REAL DEFAULT 0,
    final_adjusted_probability REAL NOT NULL,
    final_adjusted_confidence REAL NOT NULL,
    
    market_price_at_decision REAL NOT NULL,
    orderbook_depth_usd REAL,
    fee_rate REAL NOT NULL,
    calculated_edge REAL NOT NULL,
    trade_score REAL,             -- Ranking score (edge × confidence × time_value)
    
    action TEXT NOT NULL,         -- BUY_YES, BUY_NO, SKIP
    skip_reason TEXT,             -- edge_below_threshold, daily_cap, ranked_below_cutoff, cluster_exposure_limit, observe_only
    position_size_usd REAL DEFAULT 0,
    kelly_fraction_used REAL DEFAULT 0,
    market_cluster_id TEXT,       -- Correlated market group (Section 10.2)
    
    actual_outcome BOOLEAN,
    pnl REAL,
    brier_score_raw REAL,         -- (grok_raw_probability - actual)^2 — measures Grok
    brier_score_adjusted REAL,    -- (final_adjusted_probability - actual)^2 — measures system
    resolved_at TEXT,
    unrealized_adverse_move REAL, -- For cooldown on unresolved trades (Section 9.2)
    
    voided BOOLEAN DEFAULT FALSE,
    void_reason TEXT
);

CREATE INDEX idx_trades_market_type ON trade_records(market_type);
CREATE INDEX idx_trades_experiment ON trade_records(experiment_run);
CREATE INDEX idx_trades_timestamp ON trade_records(timestamp);
CREATE INDEX idx_trades_model ON trade_records(model_used);
CREATE INDEX idx_trades_unresolved ON trade_records(actual_outcome) WHERE actual_outcome IS NULL;
CREATE INDEX idx_trades_headline ON trade_records(headline_only_signal) WHERE headline_only_signal = TRUE;

CREATE TABLE calibration_state (
    bucket_range TEXT PRIMARY KEY,
    alpha REAL NOT NULL,
    beta REAL NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE market_type_performance (
    market_type TEXT PRIMARY KEY,
    total_trades INTEGER DEFAULT 0,
    total_pnl REAL DEFAULT 0.0,
    brier_scores TEXT,            -- JSON array
    total_observed INTEGER DEFAULT 0,
    counterfactual_pnl REAL DEFAULT 0.0,
    updated_at TEXT NOT NULL
);

CREATE TABLE signal_trackers (
    source_tier TEXT NOT NULL,
    info_type TEXT NOT NULL,
    market_type TEXT NOT NULL,
    present_winning INTEGER DEFAULT 0,
    present_losing INTEGER DEFAULT 0,
    absent_winning INTEGER DEFAULT 0,
    absent_losing INTEGER DEFAULT 0,
    last_updated TEXT,
    PRIMARY KEY (source_tier, info_type, market_type)
);

CREATE TABLE portfolio (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    cash_balance REAL NOT NULL,
    total_equity REAL NOT NULL,
    total_pnl REAL NOT NULL,
    peak_equity REAL NOT NULL,
    max_drawdown REAL NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE api_costs (
    date TEXT NOT NULL,
    service TEXT NOT NULL,
    calls INTEGER DEFAULT 0,
    tokens_in INTEGER DEFAULT 0,
    tokens_out INTEGER DEFAULT 0,
    cost_usd REAL DEFAULT 0.0,
    PRIMARY KEY (date, service)
);

CREATE TABLE daily_mode_log (
    date TEXT PRIMARY KEY,
    observe_only_triggered_at TEXT,  -- NULL if cap never hit
    trades_before_observe INTEGER DEFAULT 0,
    grok_calls_saved INTEGER DEFAULT 0,
    cost_saved_usd REAL DEFAULT 0.0
);
```

---

## 14. Cost Budget

### 14.1 Monthly Operating Costs

| Component | Daily calls | Cost/call | Daily | Monthly |
|-----------|-------------|-----------|-------|---------|
| Grok 4.1 Fast (Tier 1) | ~480 | ~$0.0004 | $0.19 | $5.70 |
| Grok 4.1 Fast (Tier 2) | ~75 per activation × ~0.43/day avg | ~$0.0004 | $0.013 | $3.60 |
| Grok (keyword extraction) | ~20 (cache miss) | ~$0.0002 | $0.004 | $0.12 |
| TwitterAPI.io (Tier 1) | ~48 searches | ~$0.0075 | $0.36 | $10.80 |
| TwitterAPI.io (Tier 2) | ~15 per activation × ~0.43/day avg | ~$0.0075 | $0.048 | $1.44 |
| RSS feeds | ~288 | Free | $0.00 | $0.00 |
| Polymarket API | ~500 | Free | $0.00 | $0.00 |
| **Hetzner VPS (CX22)** | — | — | — | **$4.70** |
| **TOTAL** | | | | **$26.36** |

*Tier 2 estimate: each activation runs 30 min at 2-3 min intervals = ~10-15 scans × ~5 crypto markets = ~75 Grok calls. Tier 2 activates ~3x/week on average = 0.43/day. If activation frequency is higher in volatile periods, monthly Tier 2 costs could reach $6-8.*

### 14.2 Cost as % of Bankroll

With $5,000 bankroll: monthly OpEx = **0.53%** of bankroll. System needs **0.6% monthly return** to break even on costs. This is achievable but not trivial. In volatile crypto periods where Tier 2 activates frequently, costs could reach ~$32/month (0.64%).

### 14.3 Observe-Only Savings

If daily cap is hit by noon on average: saves ~$0.10/day (~$3/month) in Grok calls. Marginal but good discipline.

---

## 15. Deployment & File Structure

```
polymarket-predictor-v2/
├── src/
│   ├── main.py                  # FastAPI + APScheduler entry point
│   ├── config.py                # Pydantic settings (reads .env)
│   ├── models.py                # All dataclasses
│   ├── scheduler.py             # Tier 1/2 scheduling + observe-only mode
│   ├── pipelines/
│   │   ├── twitter.py           # TwitterAPI.io client
│   │   ├── rss.py               # RSS feed parser
│   │   ├── polymarket.py        # Polymarket CLOB + Gamma client
│   │   ├── signal_classifier.py # Source tier classification (S1-S6)
│   │   └── context_builder.py   # Compresses signals → Grok prompt
│   ├── engine/
│   │   ├── grok_client.py       # Grok API wrapper
│   │   ├── trade_ranker.py      # Score + rank candidates per scan cycle
│   │   ├── trade_decision.py    # Edge calc + Monk Mode check
│   │   ├── execution.py         # Paper + live execution
│   │   └── resolution.py        # Auto-resolution
│   ├── learning/
│   │   ├── calibration.py       # Bayesian calibration buckets
│   │   ├── market_type.py       # Market-type performance
│   │   ├── signal_tracker.py    # 2D signal-type weighting
│   │   ├── adjustment.py        # Combined adjustment pipeline
│   │   ├── experiments.py       # Experiment run management
│   │   └── model_swap.py        # Model swap protocol
│   ├── db/
│   │   ├── sqlite.py            # SQLite wrapper
│   │   └── migrations.py        # Schema versioning
│   └── manage.py                # CLI: model_swap, void_trade, start_experiment, etc.
├── config/
│   ├── known_sources.yaml       # S1-S4 account/domain lists
│   └── rss_feeds.yaml           # RSS feed URLs and domains
├── data/
│   ├── predictor.db             # SQLite database
│   └── backups/                 # Daily DB snapshots
├── tests/
│   ├── test_calibration.py
│   ├── test_signal_classifier.py
│   ├── test_trade_ranker.py
│   ├── test_model_swap.py
│   ├── test_execution.py
│   └── test_learning.py
├── requirements.txt
├── .env                         # API keys (NEVER commit)
├── .gitignore
└── README.md
```

### 15.1 Structured Logging

All components use `structlog` for machine-readable, queryable logs. This is essential for debugging during paper trading — "why did it skip that market?" should be answerable from logs in real-time, not only by querying SQLite after the fact.

```python
import structlog

log = structlog.get_logger()

# Configure once at startup
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.BoundLogger,
    logger_factory=structlog.PrintLoggerFactory(),
)

# Usage throughout codebase:
log.info("scan_cycle_start", tier=1, mode="active", markets_found=12)
log.info("grok_called", market_id="abc123", tokens_in=850, tokens_out=200, latency_ms=1200)
log.info("trade_decision", market_id="abc123", edge=0.08, score=0.12, action="BUY_YES", rank=2)
log.warning("grok_parse_failed", market_id="abc123", attempt=1, raw_preview="```json...")
log.error("api_timeout", service="twitterapi.io", endpoint="/search", timeout_sec=10)
log.info("scan_cycle_end", tier=1, duration_sec=45, trades_executed=1, trades_skipped=3)
```

**Log levels:**
- `INFO`: Every scan cycle start/end, every trade decision, every resolution
- `WARNING`: Parse failures, API retries, approaching limits (budget, cap)
- `ERROR`: API failures after all retries, unexpected exceptions

Logs go to `journald` via systemd. View with: `sudo journalctl -u polymarket-bot -f --output=json-pretty`

### 15.2 Health Check Endpoint

The bot runs with `Restart=always`, but systemd only detects process crashes — not zombie states where the process is alive but stuck. A health endpoint enables external monitoring.

```python
from fastapi import FastAPI
from datetime import datetime, timedelta

app = FastAPI()

# Updated by scheduler after each completed scan cycle
_last_scan_completed: Optional[datetime] = None
_current_mode: str = "initializing"

@app.get("/health")
async def health():
    """Returns 200 if bot is actively scanning. Returns 503 if stale."""
    now = datetime.utcnow()
    stale = _last_scan_completed is None or (now - _last_scan_completed) > timedelta(minutes=30)
    
    status = {
        "status": "degraded" if stale else "healthy",
        "last_scan_completed": _last_scan_completed.isoformat() if _last_scan_completed else None,
        "minutes_since_scan": round((now - _last_scan_completed).total_seconds() / 60, 1) if _last_scan_completed else None,
        "mode": _current_mode,  # "active", "observe_only", "initializing"
        "open_trades": db.count_open_trades(),
        "today_trades": db.count_today_trades(),
        "uptime_hours": round((now - _startup_time).total_seconds() / 3600, 1),
    }
    
    if stale:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=503, content=status)
    return status
```

**Monitoring:** Set up a simple cron on a separate machine (or use UptimeRobot free tier) to ping `http://VPS_IP:8001/health` every 5 minutes. Alert via Telegram if 503 returned.

---

## 16. Dashboard & Monitoring

### 16.1 Phase 1: Datasette (Paper Trading Period)

Datasette turns the SQLite database into a browsable web interface with zero custom code.

**Setup:**
```bash
pip install datasette datasette-auth-passwords
datasette predictor.db --host 0.0.0.0 --port 8001
```

**What you get:**
- Full SQL query interface on all tables
- Saved queries as bookmarks (canned views for common needs)
- JSON API for any query (useful for future automation)
- Basic auth via `datasette-auth-passwords` plugin

**Essential saved queries:**

```sql
-- Portfolio overview
SELECT cash_balance, total_equity, total_pnl, peak_equity, max_drawdown,
       ROUND(max_drawdown / peak_equity * 100, 1) as drawdown_pct
FROM portfolio;

-- Today's trades
SELECT timestamp, market_question, market_type, action, calculated_edge,
       trade_score, position_size_usd, pnl, skip_reason
FROM trade_records WHERE date(timestamp) = date('now') ORDER BY timestamp DESC;

-- Brier score by market type — RAW vs ADJUSTED (last 30 days)
-- Raw = Grok accuracy. Adjusted = full system accuracy. 
-- Go-live criterion uses ADJUSTED. Gap between them shows learning system impact.
SELECT market_type, COUNT(*) as trades,
       ROUND(AVG(brier_score_raw), 3) as avg_brier_raw,
       ROUND(AVG(brier_score_adjusted), 3) as avg_brier_adjusted,
       ROUND(AVG(brier_score_raw) - AVG(brier_score_adjusted), 3) as learning_improvement,
       ROUND(SUM(pnl), 2) as total_pnl
FROM trade_records
WHERE brier_score_raw IS NOT NULL AND timestamp > datetime('now', '-30 days')
GROUP BY market_type;

-- Calibration bucket state
SELECT bucket_range, ROUND(alpha / (alpha + beta), 3) as expected_accuracy,
       CAST(alpha + beta - 2 AS INTEGER) as sample_count
FROM calibration_state;

-- Signal tracker lift (most useful combinations)
SELECT source_tier, info_type, market_type,
       present_winning + present_losing as total_present,
       CASE WHEN (present_winning + present_losing) >= 5 AND (absent_winning + absent_losing) >= 5
            THEN ROUND(CAST(present_winning AS REAL) / (present_winning + present_losing) /
                       (CAST(absent_winning AS REAL) / (absent_winning + absent_losing)), 2)
            ELSE NULL END as lift
FROM signal_trackers
WHERE present_winning + present_losing >= 5
ORDER BY lift DESC;

-- Daily API costs
SELECT date, service, calls, ROUND(cost_usd, 4) as cost
FROM api_costs ORDER BY date DESC, service;

-- Counterfactual: would skipped trades have been profitable?
SELECT market_type, COUNT(*) as skipped,
       ROUND(SUM(counterfactual_pnl), 2) as missed_pnl
FROM market_type_performance WHERE total_observed > 0;

-- Model comparison
SELECT run_id, model_used, total_trades, ROUND(total_pnl, 2) as pnl,
       ROUND(avg_brier, 3) as brier
FROM experiment_runs ORDER BY started_at DESC;

-- Headline-only signal analysis (does full article matter?)
SELECT headline_only_signal, COUNT(*) as trades,
       ROUND(AVG(brier_score_adjusted), 3) as avg_brier,
       ROUND(SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as win_pct
FROM trade_records WHERE brier_score_adjusted IS NOT NULL
GROUP BY headline_only_signal;
```

### 16.2 Phase 3+: Custom React Dashboard (Only If Profitable)

Build only after the system proves profitable in paper trading. Uses the same FastAPI backend, adds a React SPA:

| View | Data |
|------|------|
| Portfolio | Cash, equity, PnL chart, drawdown, open positions |
| Trade log | All trades with filters (tier/market_type/model/action) |
| Learning | Calibration heatmap, market-type Brier chart, signal lift matrix |
| Model comparison | Side-by-side experiment runs |
| Cost tracker | Daily spend by service, observe-only savings |
| Counterfactual | "What if" analysis on skipped trades |

### 16.3 Hosting

Both Datasette and the future React dashboard run on the same Hetzner VPS as the bot. No additional hosting cost.

**Access options:**
- Direct: `http://VPS_IP:8001` (basic auth required)
- Cloudflare Tunnel: `https://dashboard.yourdomain.com` (free, adds HTTPS + DDoS protection)
- Phone-friendly: Datasette's default UI is mobile-responsive

### 16.4 Alerting (Optional)

Telegram bot for real-time notifications:

```python
async def send_alert(message: str):
    """Send to Telegram. Use for: trades executed, daily summary, errors."""
    if not TELEGRAM_BOT_TOKEN:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    await httpx.AsyncClient().post(url, json={
        "chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"
    })

# Alert triggers:
# - Trade executed: "🟢 BUY YES on 'Will Fed cut rates?' at 0.35 | Edge: 12% | Size: $400"
# - Daily summary: "📊 Day: 3 trades, +$47.20 PnL, Brier: 0.18"
# - Error: "🔴 Grok API timeout — skipping scan cycle"
# - Observe-only: "⏸️ Daily cap hit (5/5). Switching to observe-only."
```

---

## 17. Implementation Priority

### Phase 1: Foundation (Week 1-2)
1. VPS setup (Section 2) — Hetzner CX22, Ubuntu, firewall, user, systemd
2. SQLite schema + all dataclasses
3. Structured logging setup (Section 15.1)
4. Health check endpoint (Section 15.2)
5. Polymarket client (Gamma API + CLOB)
6. Market-to-signal keyword extraction (Section 11.1)
7. TwitterAPI.io client + pre-filter
8. Source tier classifier (S1-S6) + `known_sources.yaml`
9. RSS pipeline with source tier classification + bounded headline cache
10. Context builder (with info-type classification prompt)
11. Grok 4.1 Fast client (JSON parsing with retry — Section 11.5)
12. Basic Tier 1 scanner (every 15 min)
13. Correlated market detection (Section 10.2)
14. Trade ranker (score + rank per cycle, cluster limit check)
15. Paper trade execution with slippage simulation
16. Auto-resolution

### Phase 2: Learning (Week 3-4)
17. Trade recording (including skips, observe-only, ranked_below_cutoff, cluster info)
18. Bayesian calibration buckets (with RAW correctness feedback — not adjusted)
19. Market-type performance tracking (with ADJUSTED Brier)
20. 2D signal-type weighting (source × info_type × market_type)
21. Combined adjustment pipeline (shrinkage + temporal decay)
22. Experiment run management (with model_used)
23. Void mechanism
24. Observe-only mode logic

### Phase 3: Tier 2 + Polish (Week 5-6)
25. News Detector (RSS trigger → Tier 2 activation)
26. Tier 2 crypto scanner
27. Maker order execution simulation
28. Monk Mode v2 (all limits including weekly + unrealized adverse cooldown)
29. API cost tracking + parse failure rate monitoring
30. Model swap CLI + protocol
31. Datasette dashboard setup + saved queries (with dual Brier views)
32. Telegram alerting (optional)

### Phase 4: Paper Trading (Week 7-12)
33. Run minimum 200 trades across both tiers
34. Validate info-type classification accuracy (sample 10%, target <15% error)
35. Check headline_only_signal correlation with losses (RSS upgrade decision)
36. Review learning convergence — are calibration buckets stabilizing?
37. Check signal tracker sparsity — which cells have enough data?
38. Verify Grok parse failure rate <5% (if higher, adjust prompt)
39. Verify raw Brier > adjusted Brier (learning system is helping)
40. Monthly: update `known_sources.yaml`, check S4 threshold

### Phase 5: Go Live (Week 13+)
41. If adjusted Brier < 0.25 AND positive paper PnL → switch to live with 20% of bankroll ($1,000)
42. Fund Polygon wallet with USDC
43. Change `ENVIRONMENT=live` in .env, restart bot
44. Monitor first 20 live trades closely — watch for slippage discrepancy vs paper
45. Gradually increase allocation: 20% → 50% → 80% over 4 weeks if profitable
46. If live adjusted Brier >0.30 or drawdown >10%, pause and investigate

---

## 18. Operational Runbooks

### 18.1 Daily Checklist (2 minutes)

```
□ Check Telegram for overnight alerts (if enabled)
□ Open Datasette → run "Today's trades" query
□ Check portfolio equity vs yesterday
□ Verify bot is running: sudo systemctl status polymarket-bot
□ Check API costs: run "Daily API costs" query
```

### 18.2 Monthly Review (30 minutes)

```
□ Update known_sources.yaml — any new official/wire/media accounts?
□ Review S4 threshold: any S6 signals showing expert-like lift? Lower to 40K?
□ Run "Brier by market type" query — any market types to disable?
□ Run "Signal tracker lift" query — any surprising findings?
□ Run "Headline-only analysis" — is RSS-only correlated with losses?
□ Check VPS disk usage: df -h (alert if >80%)
□ Run system updates: sudo apt update && sudo apt upgrade -y
□ Review total monthly cost vs budget ($26.42 target)
□ Run "Model comparison" query — is current model still performing?
□ Review counterfactual data — are we missing profitable skipped trades?
□ Delete old DB backups: find data/backups/ -mtime +30 -delete
```

### 18.3 Model Swap Runbook

```
1. □ Wait 1-2 weeks after new model release for community benchmarks
2. □ Optional: parallel test on 20-30 markets (compare raw probabilities)
3. □ Run: python manage.py model_swap --old-model X --new-model Y --reason "..."
4. □ Verify new experiment run created in DB
5. □ Verify calibration buckets reset (all alpha=1, beta=1)
6. □ Monitor first 30 trades:
   - JSON parsing errors? (adjust Grok client if needed)
   - Edge sizes similar to before?
   - Brier score trajectory?
7. □ After 30 trades: compare Brier with old model at same count
8. □ If Brier >0.35 → rollback: python manage.py model_swap --old-model Y --new-model X
```

### 18.4 Go-Live Checklist

```
1. □ Paper trading complete: 200+ trades
2. □ Adjusted Brier score < 0.25 (measures full system, not just Grok)
3. □ Raw Brier improving → adjusted Brier (learning system is helping, not hurting)
4. □ Positive cumulative PnL on paper
4. □ Calibration buckets have stabilized (uncertainty < 0.15 for main buckets)
5. □ No market type with should_disable = True
6. □ Wallet funded with USDC on Polygon
7. □ POLYMARKET_API_KEY, SECRET, PASSPHRASE set in .env
8. □ Change ENVIRONMENT=live in .env
9. □ Start with 20% of bankroll ($1,000)
10. □ Restart bot: sudo systemctl restart polymarket-bot
11. □ Verify first live trade executes correctly
12. □ Set Telegram alerts to high priority for first week
13. □ Review daily for first 2 weeks
```

### 18.5 Emergency Stop

```bash
# Stop all trading immediately
sudo systemctl stop polymarket-bot

# Or, less drastic: switch to paper mode
# Edit .env: ENVIRONMENT=paper
sudo systemctl restart polymarket-bot

# Check open positions (they'll remain on Polymarket until resolved)
# Use Datasette: SELECT * FROM trade_records WHERE actual_outcome IS NULL AND action != 'SKIP'
```

### 18.6 Backup & Recovery

```bash
# Manual backup
cp data/predictor.db data/backups/predictor_manual_$(date +%F_%H%M).db

# Restore from backup
sudo systemctl stop polymarket-bot
cp data/backups/predictor_2026-03-15.db data/predictor.db
sudo systemctl start polymarket-bot

# Full system recovery (new VPS)
# 1. Set up new VPS (Section 2)
# 2. Clone repo from GitHub
# 3. Copy .env from secure backup
# 4. Copy latest predictor.db backup
# 5. Start services
```
