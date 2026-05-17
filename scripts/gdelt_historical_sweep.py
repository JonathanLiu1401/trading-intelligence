"""GDELT v2 DOC API — historical monthly sweep.

Iterates every (month_window × query_group) combination from START_DATE to
now, inserting up to 250 articles per call directly into ArticleStore. The
checkpoint file lets you Ctrl-C and resume without re-fetching already-done
windows. Designed to run alongside the live daemon without conflicts (uses the
same ArticleStore write-lock).

Usage:
    python scripts/gdelt_historical_sweep.py          # 2013 → now
    python scripts/gdelt_historical_sweep.py 2018 01  # custom start year/month
"""
from __future__ import annotations

import hashlib
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from collectors.gdelt_collector import QUERY_GROUPS  # reuse canonical list
from storage.article_store import ArticleStore, _get_db_path

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
MAX_RECORDS = 250
REQUEST_TIMEOUT = 25
MAX_WORKERS = 1           # GDELT hard limit: 1 req per 5s globally
REQ_INTERVAL_S = 5.1      # just over their stated 5s limit
CHECKPOINT_PATH = BASE_DIR / "data" / "gdelt_sweep_checkpoint.json"

# Use YEARLY windows to minimise total tasks while maximising coverage.
# 2015-2025 = 11 years × 252 queries = 2,772 tasks (vs 40K monthly).
# GDELT v2 DOC API coverage starts Feb 2015 — earlier windows return nothing.
SWEEP_START = (2015, 1)
USE_YEARLY_WINDOWS = True

# Parse seendate: "20240115T143000Z" or "20240115143000"
def _parse_gdelt_date(raw: str) -> str:
    raw = (raw or "").replace("T", "").replace("Z", "").strip()
    if len(raw) >= 14:
        try:
            dt = datetime.strptime(raw[:14], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
            return dt.isoformat()
        except ValueError:
            pass
    if len(raw) >= 8:
        try:
            dt = datetime.strptime(raw[:8], "%Y%m%d").replace(tzinfo=timezone.utc)
            return dt.isoformat()
        except ValueError:
            pass
    return ""


def _task_id(query: str, start: str) -> str:
    return hashlib.md5(f"{query}|{start}".encode()).hexdigest()


def _load_checkpoint() -> set[str]:
    if CHECKPOINT_PATH.exists():
        try:
            return set(json.loads(CHECKPOINT_PATH.read_text()).get("done", []))
        except Exception:
            pass
    return set()


_ckpt_lock = threading.Lock()
_done_ids: set[str] = set()


def _mark_done(tid: str):
    with _ckpt_lock:
        _done_ids.add(tid)
        if len(_done_ids) % 200 == 0:
            CHECKPOINT_PATH.write_text(json.dumps({"done": list(_done_ids)}))


def _flush_checkpoint():
    with _ckpt_lock:
        CHECKPOINT_PATH.write_text(json.dumps({"done": list(_done_ids)}))


def _fetch_window(query: str, start_dt: str, end_dt: str) -> list[dict]:
    """Fetch one (query, window) slice. Returns article dicts."""
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": MAX_RECORDS,
        "startdatetime": start_dt,
        "enddatetime": end_dt,
        "sourcelang": "english",
        "sort": "DateDesc",
    }
    try:
        r = requests.get(GDELT_URL, params=params, timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            return []
        data = r.json()
        return data.get("articles") or []
    except Exception:
        return []


def _build_windows(start_year: int, start_month: int) -> list[tuple[str, str, str]]:
    """Build (gdelt_start, gdelt_end, label) windows.

    If USE_YEARLY_WINDOWS: one window per year (minimises task count while
    maximising GDELT API efficiency — still returns 250 results per query).
    Monthly otherwise (more granular but 12× more API calls).
    """
    import calendar
    now = datetime.now(timezone.utc)
    windows = []
    if USE_YEARLY_WINDOWS:
        for y in range(start_year, now.year + 1):
            last_day = calendar.monthrange(y, 12)[1]
            end_m = min(now.month, 12) if y == now.year else 12
            last_day = calendar.monthrange(y, end_m)[1]
            start_str = f"{y:04d}0101000000"
            end_str = f"{y:04d}{end_m:02d}{last_day:02d}235959"
            windows.append((start_str, end_str, str(y)))
    else:
        y, m = start_year, start_month
        while (y, m) <= (now.year, now.month):
            ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
            last_day = calendar.monthrange(y, m)[1]
            start_str = f"{y:04d}{m:02d}01000000"
            end_str = f"{y:04d}{m:02d}{last_day:02d}235959"
            windows.append((start_str, end_str, f"{y:04d}-{m:02d}"))
            y, m = ny, nm
    return windows


def run(start_year: int = 2013, start_month: int = 1):
    global _done_ids
    _done_ids = _load_checkpoint()
    print(f"[gdelt_sweep] checkpoint: {len(_done_ids)} tasks already done")

    store = ArticleStore()
    windows = _build_windows(start_year, start_month)
    queries = QUERY_GROUPS

    # Build task list: (query, start, end, label)
    tasks = []
    for (start_str, end_str, label) in windows:
        for q in queries:
            tid = _task_id(q, start_str)
            if tid not in _done_ids:
                tasks.append((q, start_str, end_str, label, tid))

    total = len(tasks)
    done_count = 0
    inserted_total = 0
    start_time = time.time()
    print(f"[gdelt_sweep] {total} tasks to run ({len(windows)} months × {len(queries)} queries)")
    print(f"[gdelt_sweep] writing to {_get_db_path()}")

    lock = threading.Lock()
    _last_req_time = [0.0]  # mutable for closure capture

    def worker(task):
        nonlocal done_count, inserted_total
        q, start_str, end_str, label, tid = task
        # Global rate limit: 1 req / REQ_INTERVAL_S
        with lock:
            now_t = time.time()
            wait = REQ_INTERVAL_S - (now_t - _last_req_time[0])
            if wait > 0:
                time.sleep(wait)
            _last_req_time[0] = time.time()
        articles = _fetch_window(q, start_str, end_str)

        to_insert = []
        for a in articles:
            url = a.get("url") or ""
            title = a.get("title") or ""
            if not url or not title:
                continue
            domain = a.get("domain") or ""
            published = _parse_gdelt_date(a.get("seendate") or "")
            to_insert.append({
                "link": url,
                "title": title,
                "source": f"gdelt_historical/{domain}" if domain else "gdelt_historical",
                "published": published,
                "summary": "",
                "_relevance_score": 3.0,
            })

        inserted = store.insert_batch(to_insert) if to_insert else 0
        _mark_done(tid)

        with lock:
            done_count += 1
            inserted_total += inserted
            if done_count % 100 == 0:
                elapsed = time.time() - start_time
                rate = done_count / elapsed * 60
                remaining = (total - done_count) / (done_count / elapsed) if done_count else 0
                print(
                    f"[gdelt_sweep] {done_count}/{total} done | "
                    f"+{inserted_total} articles | "
                    f"{rate:.0f} tasks/min | "
                    f"ETA {remaining/60:.1f}h"
                )
        return inserted

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(worker, t) for t in tasks]
        try:
            for f in as_completed(futures):
                f.result()
        except KeyboardInterrupt:
            print("\n[gdelt_sweep] interrupted — saving checkpoint")

    _flush_checkpoint()
    elapsed = time.time() - start_time
    print(f"\n[gdelt_sweep] DONE — {inserted_total} new articles in {elapsed/60:.1f} min")


if __name__ == "__main__":
    y = int(sys.argv[1]) if len(sys.argv) > 1 else 2013
    m = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    run(y, m)
