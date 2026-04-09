from __future__ import annotations

import io
import sqlite3
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
import structlog

log = structlog.get_logger()

GAMMA_API_BASE = "https://gamma-api.polymarket.com"
GDELT_MASTERFILELIST = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
DEFAULT_DOMAINS = ["reuters.com", "apnews.com", "bbc.com", "bloomberg.com", "coindesk.com"]

from src.pipelines.market_classifier import classify_market_type


def init_backtest_db(db_path: str) -> None:
    """Create backtest_data.db with historical_markets and historical_news tables."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS historical_markets (
            market_id TEXT PRIMARY KEY,
            question TEXT NOT NULL,
            market_type TEXT NOT NULL,
            created_at TEXT,
            resolution_datetime TEXT,
            actual_outcome TEXT,
            baseline_price REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS historical_news (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            published_at TEXT NOT NULL,
            source_domain TEXT NOT NULL,
            headline TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_news_dedup
        ON historical_news(published_at, source_domain, headline)
    """)
    conn.commit()
    conn.close()


async def scrape_polymarket_markets(db_path: str, max_markets: int = 5000) -> int:
    """Fetch closed markets from Polymarket Gamma API and save to historical_markets."""
    import json

    init_backtest_db(db_path)
    conn = sqlite3.connect(db_path)
    timeout = httpx.Timeout(15.0, connect=5.0)
    inserted = 0
    offset = 0
    page_size = 500

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            while inserted < max_markets:
                params = {
                    "closed": "true",
                    "active": "false",
                    "limit": page_size,
                    "offset": offset,
                    "order": "volume",
                    "ascending": "false",
                }
                try:
                    resp = await client.get(f"{GAMMA_API_BASE}/markets", params=params)
                    if resp.status_code == 429:
                        log.warning("polymarket_rate_limited_during_scrape", offset=offset)
                        break
                    resp.raise_for_status()
                    page = resp.json()
                except Exception as e:
                    log.error("scrape_page_error", offset=offset, error=str(e))
                    break

                if not page:
                    break

                for m in page:
                    try:
                        # Use neutral baseline price (0.50) for all historical markets.
                        # The Gamma API's outcomePrices for closed markets reflects the
                        # FINAL price (near 1.0 or 0.0), which leaks the outcome to the
                        # LLM and inflates win rates. A neutral price forces the LLM to
                        # predict from the question + news context alone.
                        baseline_price = 0.50

                        # actual_outcome: derived from outcomePrices on resolved markets.
                        # Resolved markets have outcomePrices like ["1","0"] (YES won)
                        # or ["0","1"] (NO won). Voided markets show ["0.5","0.5"].
                        actual_outcome = None
                        outcomePrices = m.get("outcomePrices", "")
                        if isinstance(outcomePrices, str) and outcomePrices:
                            try:
                                op = json.loads(outcomePrices)
                            except (json.JSONDecodeError, ValueError):
                                op = None
                        elif isinstance(outcomePrices, list):
                            op = outcomePrices
                        else:
                            op = None
                        if op and len(op) >= 2:
                            try:
                                yes_price = float(op[0])
                                if yes_price > 0.5:
                                    actual_outcome = "YES"
                                elif yes_price < 0.5:
                                    actual_outcome = "NO"
                                # 0.5 = voided/cancelled, leave as None
                            except (ValueError, TypeError):
                                pass

                        # Dates
                        resolution_datetime = m.get("endDate") or m.get("end_date_iso")
                        created_at = m.get("createdAt") or m.get("created_at")

                        question = m.get("question", "")
                        market_id = str(m.get("id", m.get("condition_id", "")))
                        market_type = classify_market_type(question)

                        conn.execute(
                            """INSERT OR IGNORE INTO historical_markets
                               (market_id, question, market_type, created_at, resolution_datetime, actual_outcome, baseline_price)
                               VALUES (?, ?, ?, ?, ?, ?, ?)""",
                            (market_id, question, market_type, created_at, resolution_datetime, actual_outcome, baseline_price),
                        )
                        inserted += 1
                    except Exception as e:
                        log.warning("market_parse_error", error=str(e))
                        continue

                conn.commit()

                if offset % 2500 == 0:
                    log.info("scrape_progress", inserted=inserted, offset=offset)

                offset += page_size
                if len(page) < page_size:
                    break
    finally:
        conn.close()

    log.info("scrape_complete", total_inserted=inserted)
    return inserted


async def download_gdelt_news(
    db_path: str,
    start_date: str,
    end_date: str,
    domains: list[str] | None = None,
) -> int:
    """Download GDELT v2 GKG files for date range and save to historical_news."""
    if domains is None:
        domains = DEFAULT_DOMAINS

    init_backtest_db(db_path)
    conn = sqlite3.connect(db_path)

    start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    timeout = httpx.Timeout(60.0, connect=10.0)
    total_inserted = 0

    try:
        # Fetch masterfilelist
        async with httpx.AsyncClient(timeout=timeout) as client:
            log.info("fetching_gdelt_masterfilelist")
            resp = await client.get(GDELT_MASTERFILELIST)
            resp.raise_for_status()
            masterfilelist = resp.text

        # Parse and filter URLs
        gkg_urls = []
        for line in masterfilelist.splitlines():
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            url = parts[2]
            if not url.endswith(".gkg.csv.zip"):
                continue
            # Extract timestamp from filename: .../20250115120000.gkg.csv.zip
            fname = url.split("/")[-1]
            ts_str = fname.split(".")[0]
            if len(ts_str) != 14:
                continue
            try:
                file_dt = datetime.strptime(ts_str, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            if start_dt <= file_dt <= end_dt:
                gkg_urls.append((file_dt, url))

        total_files = len(gkg_urls)
        log.info("gdelt_files_to_download", count=total_files, start=start_date, end=end_date)

        if total_files > 500:
            log.warning(
                "gdelt_large_download",
                files=total_files,
                tip="Use --domains to limit download size (~5MB/day with default domains vs 200-400MB/day unfiltered)",
            )

        async with httpx.AsyncClient(timeout=timeout) as client:
            for i, (file_dt, url) in enumerate(gkg_urls):
                try:
                    resp = await client.get(url)
                    resp.raise_for_status()
                    raw = resp.content

                    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                        csv_name = zf.namelist()[0]
                        with zf.open(csv_name) as csv_bytes:
                            reader = io.TextIOWrapper(csv_bytes, encoding="utf-8", errors="replace")
                            batch = []
                            for line in reader:
                                cols = line.split("\t")
                                if len(cols) < 5:
                                    continue
                                date_col = cols[1].strip()
                                source_common = cols[3].strip()
                                doc_id = cols[4].strip()

                                # Domain filter
                                if domains and not any(d.lower() in source_common.lower() for d in domains):
                                    continue

                                # Parse published_at
                                if len(date_col) != 14:
                                    continue
                                try:
                                    pub_dt = datetime.strptime(date_col, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
                                except ValueError:
                                    continue

                                batch.append((pub_dt.isoformat(), source_common, doc_id))

                            if batch:
                                conn.executemany(
                                    "INSERT OR IGNORE INTO historical_news (published_at, source_domain, headline) VALUES (?, ?, ?)",
                                    batch,
                                )
                                conn.commit()
                                total_inserted += len(batch)

                    if (i + 1) % 50 == 0:
                        log.info("gdelt_progress", processed=i + 1, total=total_files, rows_inserted=total_inserted)

                except Exception as e:
                    log.warning("gdelt_file_error", url=url, error=str(e))
                    continue

    finally:
        conn.close()

    log.info("gdelt_download_complete", total_inserted=total_inserted)
    return total_inserted
