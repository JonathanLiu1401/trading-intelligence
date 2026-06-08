"""Memory-safe ArticleNet aggregation daemon for the Mac launchd service.

This process intentionally runs only article collectors plus the optional
ArticleNet trainer. It does not import or start scorer, alerting, dashboard,
paper-trading, or backtest workers. The full Linux daemon remains in daemon.py;
this file is the constrained Mac runtime used to keep ArticleNet collecting
and retraining without exhausting memory.
"""
from __future__ import annotations

import fcntl
import os
import signal
import sys
import threading
import time
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
sys.path.insert(0, str(BASE_DIR))

from collectors.benzinga_analyst_collector import collect_benzinga_analyst
from collectors.financial_blogs_collector import collect_financial_blogs
from collectors.gdelt_collector import collect_gdelt
from collectors.globenewswire_collector import collect_globenewswire
from collectors.google_news import collect_google_news
from collectors.investment_research_blogs_collector import (
    collect_investment_research_blogs,
)
from collectors.market_movers import collect_market_movers
from collectors.prnewswire_collector import collect as collect_prnewswire
from collectors.rss_collector import collect_rss
from collectors.seekingalpha_collector import collect_seekingalpha
from collectors.yahoo_ticker_rss import collect_yahoo_ticker_rss
from collectors.yahoo_trending_tickers import collect_yahoo_trending
from collectors import source_health
from core.logger import get_logger
from storage.article_store import ArticleStore
from triage.heuristic_scorer import score_article as _heuristic_score_article

log = get_logger("article_aggregation_daemon")

_running = True
_store_lock = threading.Lock()
_worker_last_ok: dict[str, float] = {}


def _env_seconds(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _requested_workers() -> set[str]:
    raw = os.environ.get("DIGITAL_INTERN_WORKERS", "")
    return {name.strip() for name in raw.split(",") if name.strip()}


def _sleep(seconds: float) -> None:
    deadline = time.monotonic() + max(0.0, seconds)
    while _running and time.monotonic() < deadline:
        time.sleep(min(1.0, deadline - time.monotonic()))


def _handle_signal(_sig, _frame) -> None:
    global _running
    _running = False


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)


def _acquire_singleton_lock() -> int:
    lock_path = BASE_DIR / "data" / "article_aggregation.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log.error("[article_aggregation] another instance already holds the lock; exiting")
        os.close(fd)
        sys.exit(1)
    os.ftruncate(fd, 0)
    os.write(fd, str(os.getpid()).encode())
    return fd


def _ingest(store: ArticleStore, articles: list[dict], source_tag: str) -> int:
    for art in articles:
        if "_relevance_score" in art:
            continue
        result = _heuristic_score_article(
            art.get("title", ""),
            art.get("summary", ""),
            art.get("source", ""),
            art.get("published", ""),
        )
        art["_relevance_score"] = result["score"]
        art["_score_detail"] = result
    relevant = [a for a in articles if a.get("_relevance_score", 0) >= 0.5]
    with _store_lock:
        inserted = store.insert_batch(relevant)
    if inserted:
        log.info("[%s] +%d new articles from %d collected", source_tag, inserted, len(articles))
    return inserted


def _collector_worker(
    name: str,
    collect_fn,
    store: ArticleStore,
    interval: float,
    source_tag: str | None = None,
) -> None:
    log.info("[%s_worker] started interval=%ss", name, interval)
    source_key = source_tag or name
    while _running:
        started = time.monotonic()
        try:
            articles = collect_fn() or []
            inserted = _ingest(store, articles, source_key)
            try:
                source_health.record_result(source_key, len(articles))
            except Exception as he:
                log.warning("[%s_worker] source_health error: %s", name, he)
            _worker_last_ok[name] = time.time()
            log.debug(
                "[%s] cycle ok collected=%d inserted=%d elapsed=%.1fs",
                name,
                len(articles),
                inserted,
                time.monotonic() - started,
            )
        except Exception as e:
            log.warning("[%s_worker] error: %s", name, e)
        _sleep(interval)


def _stats_worker(store: ArticleStore) -> None:
    interval = _env_seconds("STATS_INTERVAL", 60)
    while _running:
        try:
            with _store_lock:
                stats = store.stats()
            log.info(
                "[stats] total=%s urgent=%s unscored=%s low_kw=%s db=%sMB",
                stats.get("total"),
                stats.get("urgent"),
                stats.get("unscored"),
                stats.get("below_threshold", 0),
                stats.get("db_mb"),
            )
            _worker_last_ok["stats"] = time.time()
        except Exception as e:
            log.warning("[stats_worker] error: %s", e)
        _sleep(interval)


def _purge_worker(store: ArticleStore) -> None:
    reap_interval = _env_seconds("URGENT_REAP_INTERVAL", 3600)
    purge_interval = _env_seconds("PURGE_INTERVAL", 6 * 3600)
    last_reap = 0.0
    last_purge = time.monotonic()
    while _running:
        now = time.monotonic()
        try:
            if now - last_reap >= reap_interval:
                with _store_lock:
                    n = store.reap_stale_urgent()
                if n:
                    log.info("[purge] reaped %d stale urgent row(s)", n)
                last_reap = now
            if now - last_purge >= purge_interval:
                with _store_lock:
                    store.purge_old()
                log.info("[purge] purge_old complete")
                last_purge = now
            _worker_last_ok["purge"] = time.time()
        except Exception as e:
            log.warning("[purge_worker] error: %s", e)
        _sleep(300)


def main() -> None:
    lock_fd = _acquire_singleton_lock()
    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    log.info(" ARTICLENET AGGREGATION DAEMON — STARTING")
    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    store = ArticleStore()
    log.info("Store ready: %s", store.stats())

    requested = _requested_workers()
    workers = [
        ("gdelt", collect_gdelt, _env_seconds("GDELT_INTERVAL", 1800), "gdelt"),
        ("rss", collect_rss, _env_seconds("RSS_INTERVAL", 300), "rss"),
        ("google_news", collect_google_news, _env_seconds("GOOGLE_NEWS_INTERVAL", 300), "google_news"),
        (
            "yahoo_ticker_rss",
            collect_yahoo_ticker_rss,
            _env_seconds("YAHOO_TICKER_RSS_INTERVAL", 600),
            "yahoo_ticker_rss",
        ),
        ("market_movers", collect_market_movers, _env_seconds("MARKET_MOVERS_INTERVAL", 600), "market_movers"),
        ("yahoo_trending", collect_yahoo_trending, _env_seconds("YAHOO_TRENDING_INTERVAL", 600), "yahoo_trending"),
        (
            "benzinga_analyst",
            collect_benzinga_analyst,
            _env_seconds("BENZINGA_INTERVAL", 300),
            "benzinga_analyst",
        ),
        (
            "globenewswire",
            collect_globenewswire,
            _env_seconds("GLOBENEWSWIRE_INTERVAL", 300),
            "globenewswire",
        ),
        ("prnewswire", collect_prnewswire, _env_seconds("PRNEWSWIRE_INTERVAL", 300), "prnewswire"),
        ("seekingalpha", collect_seekingalpha, _env_seconds("SEEKINGALPHA_INTERVAL", 600), "seekingalpha"),
        (
            "financial_blogs",
            collect_financial_blogs,
            _env_seconds("FINANCIAL_BLOGS_INTERVAL", 600),
            "financial_blogs",
        ),
        (
            "investment_research_blogs",
            collect_investment_research_blogs,
            _env_seconds("INVESTMENT_RESEARCH_BLOGS_INTERVAL", 900),
            "investment_research_blogs",
        ),
    ]

    known_workers = {name for name, *_ in workers} | {"stats", "purge"}
    unknown_workers = sorted(requested - known_workers)
    if unknown_workers:
        log.warning("[article_aggregation] unknown DIGITAL_INTERN_WORKERS ignored: %s", unknown_workers)
    if requested:
        workers = [w for w in workers if w[0] in requested]
        log.info("[article_aggregation] worker allowlist active: %s", sorted(requested & known_workers))

    threads: list[threading.Thread] = []
    for name, collect_fn, interval, source_tag in workers:
        t = threading.Thread(
            target=_collector_worker,
            args=(name, collect_fn, store, interval, source_tag),
            name=name,
            daemon=True,
        )
        t.start()
        threads.append(t)

    utility_workers = [("stats", _stats_worker), ("purge", _purge_worker)]
    if requested:
        utility_workers = [(name, fn) for name, fn in utility_workers if name in requested]
    for name, fn in utility_workers:
        t = threading.Thread(target=fn, args=(store,), name=name, daemon=True)
        t.start()
        threads.append(t)

    log.info("[article_aggregation] all %d workers running", len(threads))
    try:
        while _running:
            _sleep(5)
    finally:
        try:
            os.close(lock_fd)
        except Exception:
            pass
        log.info("[article_aggregation] shutdown complete")


if __name__ == "__main__":
    main()
