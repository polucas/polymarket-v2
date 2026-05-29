---
name: no-signal-rss-audit
description: Diagnostic audit of markets stuck at skip_reason=no_signals. Pulls a sample, runs a Python coverage check against live RSS signals, clusters uncovered markets by question pattern, and proposes candidate feed searches. Read-only — never modifies rss_feeds.yaml.
trigger: no signal audit, rss audit, no_signals audit, rss coverage, /no-signal-rss-audit
---

## Purpose

Answers "why are markets skipping with `no_signals`?" in one command. Pulls a random sample of `no_signals` markets from the DB, runs a live RSS poll on the VPS to see which headlines would match each market's keywords, clusters the uncovered markets by topic pattern, and recommends feed search guidance for each cluster. Purely diagnostic — never writes to config.

## Inputs

- `N` — number of markets to sample (default `30`)
- `K` — hours lookback window (default `24`)
- `EXCLUDE_TYPES` — comma-separated market types to exclude (default `sports,esports`)

Parse from user invocation positionally: `/no-signal-rss-audit 50 48 sports,esports,crypto_15m`.
If fewer than 3 args are given, use defaults for the missing ones.

## Steps

1. **Pull N no_signals markets from DB.**

   Substitute `K`, `N`, and the excluded types into the query. For the `NOT IN` clause, each excluded type must be a quoted string:

   ```bash
   ssh root@49.13.159.52 "cd /root/polymarket-v2 && sqlite3 -separator '|' data/predictor.db \
   \"SELECT market_id, substr(market_question,1,100), market_type, ROUND(volume_24h,2) \
   FROM trade_records \
   WHERE skip_reason='no_signals' \
     AND timestamp > datetime('now','-K hours') \
     AND market_type NOT IN ('sports','esports') \
   ORDER BY RANDOM() \
   LIMIT N;\""
   ```

   Also capture the total `no_signals` count in the window for the header:

   ```bash
   ssh root@49.13.159.52 "cd /root/polymarket-v2 && sqlite3 data/predictor.db \
   \"SELECT COUNT(*) FROM trade_records \
   WHERE skip_reason='no_signals' AND timestamp > datetime('now','-K hours');\""
   ```

   If the sample returns 0 rows, print "No `no_signals` skips in the last K hours (after exclusions)" and stop.

2. **Run Python diagnostic on VPS.**

   Write the script locally to `/tmp/no_signal_audit.py`, then `scp` it and run it:

   ```python
   # /tmp/no_signal_audit.py
   # Usage: python3 no_signal_audit.py "<pipe-separated rows from step 1>"
   import asyncio, sys, json

   ROWS_RAW = sys.argv[1]  # pipe/newline-separated rows from step 1

   async def main():
       import os, sys
       sys.path.insert(0, "/root/polymarket-v2")
       os.chdir("/root/polymarket-v2")

       from src.config import Settings
       from src.pipelines.rss import RSSPipeline
       from src.pipelines.context_builder import extract_keywords

       settings = Settings()
       rss = RSSPipeline(settings=settings)
       await rss.poll_and_accumulate()
       signals = rss._cached_signals

       # Per-feed signal yield
       feed_counts = {}
       for s in signals:
           feed_counts[s.author] = feed_counts.get(s.author, 0) + 1

       # Parse sampled markets
       markets = []
       for line in ROWS_RAW.strip().split("\n"):
           parts = line.strip().split("|")
           if len(parts) < 2:
               continue
           market_id = parts[0].strip()
           question = parts[1].strip()
           market_type = parts[2].strip() if len(parts) > 2 else "unknown"
           markets.append((market_id, question, market_type))

       results = []
       for (mid, question, mtype) in markets:
           keywords = extract_keywords(mid, question, mtype)
           matches = []
           for s in signals:
               content_lower = s.content.lower()
               for kw in keywords:
                   if kw.lower() in content_lower:
                       matches.append({"feed": s.author, "headline": s.content[:80], "kw": kw})
                       break  # one match per signal is enough
           results.append({
               "market_id": mid,
               "question": question[:80],
               "market_type": mtype,
               "keywords": keywords[:8],
               "match_count": len(matches),
               "top_3_matches": matches[:3],
           })

       print(json.dumps({
           "feed_counts": feed_counts,
           "total_signals": len(signals),
           "results": results,
       }, indent=2))

   asyncio.run(main())
   ```

   Run it on the VPS, passing the DB rows as an argument:

   ```bash
   # scp the script
   scp /tmp/no_signal_audit.py root@49.13.159.52:/tmp/no_signal_audit.py

   # Run, piping the DB output as stdin via argument
   ssh root@49.13.159.52 "cd /root/polymarket-v2 && python3 /tmp/no_signal_audit.py '<rows_from_step1>'"
   ```

   Capture the JSON output for analysis in steps 3 and 4.

3. **Build per-feed health table.**

   From the `feed_counts` dict in the JSON output, classify each feed:

   | Status | Condition |
   |---|---|
   | `working` | ≥ 3 signals in current poll |
   | `sparse` | 1–2 signals |
   | `broken` | 0 signals (feed in config but no output) |

   To find feeds that are configured but produced zero signals, compare `feed_counts` keys against the known feed list:

   ```bash
   ssh root@49.13.159.52 "cd /root/polymarket-v2 && python3 -c \
   \"import yaml; feeds=yaml.safe_load(open('config/rss_feeds.yaml')).get('feeds',{}); \
   [print(k) for k in feeds.keys()]\""
   ```

   Any feed name in the config but absent from `feed_counts` is `broken` (0 signals).

   Alarm threshold: if **more than 3 feeds** show `broken` simultaneously, flag as `RSS_HEALTH: DEGRADED` — likely a network or config issue rather than coverage gap.

4. **Cluster uncovered markets and propose feeds.**

   From the `results` array, identify markets with `match_count == 0` (uncovered).

   Cluster uncovered markets using these heuristics (apply in order, a market can match multiple):
   - **Country/region cluster**: extract country names or demonyms from the question (e.g. "South Korea", "Korean", "Brazil", "Nigerian"). Group by country.
   - **Platform/entity cluster**: look for recurring named entities (e.g. "Elon", "Tesla", "FIFA", "IMF"). Group by entity.
   - **Metric cluster**: look for recurring measurement patterns: "how many", "tweet count", "transit", "rainfall", "temperature". Group as "metrics cluster".
   - **Catch-all**: remaining uncovered markets with no shared pattern → "misc uncovered".

   For each cluster, produce a candidate feed search block:

   ```
   Cluster: <name>
   Markets: <count>
   Coverage gap: no current feed covers <topic>
   Search guidance: <one-sentence description of what kind of feed to look for>
   Evaluation note: before adding, verify source bias/reliability via MBFC (mediabiasfactcheck.com)
                    or AllSides. Do NOT add feeds from sources rated "Low Factual Reporting".
   ```

   The skill must NOT recommend specific URLs. Only provide search guidance such as:
   - "Search for official government press release feeds for [country]"
   - "Look for established wire service feeds covering [region] politics"
   - "Search MBFC for RSS feeds from centre/centre-right business outlets covering [entity]"

## Output format

Markdown report with five sections:

**1. Header**
```
no_signals RSS Audit — K=K hours | Sample N=N (TOTAL_COUNT total in window) | Excluded: EXCLUDE_TYPES
RSS poll captured TOTAL_SIGNALS signals across FEED_COUNT feeds
```

**2. Per-feed health**

Table: `feed name | signals this poll | status`

If `RSS_HEALTH: DEGRADED` alarm fires, print it as a bold warning above the table.

**3. Coverage matrix**

Table: `cluster | markets in cluster | market count uncovered | search guidance`

Then: coverage rate = `(N - uncovered_count) / N * 100`

Alarm threshold: if **coverage rate < 30%** (more than 70% of sampled markets uncovered), print:

> WARNING: system-level filter gap — fewer than 30% of no_signals markets match any current RSS feed. Consider broadening feed coverage or reviewing whether market types should be filtered upstream.

**4. Sample uncovered markets**

Show up to 10 uncovered markets (market_id | question | type | keywords extracted). This helps spot extraction failures (e.g. keywords that are too generic to ever match headlines).

**5. Verdict and next steps**

One-line verdict:
- `GREEN` — coverage >= 70%, all feeds working
- `YELLOW` — coverage 30–70% or 1–3 broken feeds
- `RED` — coverage < 30% or > 3 broken feeds or RSS_HEALTH DEGRADED

Numbered next-steps list. Example structure:
1. Fix broken feeds (if any) — re-check URLs in `config/rss_feeds.yaml`.
2. For each cluster with >3 uncovered markets — investigate candidate feeds using the search guidance above.
3. After identifying candidate feeds, validate them manually and add to `config/rss_feeds.yaml` (requires code change + deploy).
4. Re-run `/no-signal-rss-audit` after any feed additions to confirm coverage improvement.

## Error handling

- SSH unreachable → abort, print "ERROR: VPS unreachable at root@49.13.159.52".
- Step 1 returns 0 rows → "No `no_signals` skips in the last K hours (after exclusions). Feed coverage looks complete for this window."
- Python script import error on VPS (e.g. missing module) → print the traceback and note "VPS environment issue — run `pip install -r requirements.txt` on VPS".
- JSON parse failure from script output → print raw output and note "Diagnostic script returned non-JSON; inspect manually".
- `sqlite3` locked → retry once after 2s.
- Feed config file not found → "config/rss_feeds.yaml missing on VPS — re-deploy or git pull".

## Related

- `skip-reason-analyzer` — this skill goes deeper on one specific skip reason (`no_signals`); `skip-reason-analyzer` gives the breadth view across all skip reasons.
- `vps-health-check` — confirms the service is running before this audit is meaningful.
- `deploy-update` — use after manually adding feeds to `config/rss_feeds.yaml` and updating `rss_feeds.yaml` in git.
- `morning-check` — will surface `no_signals` as a top skip reason, which is the trigger to run this audit.
