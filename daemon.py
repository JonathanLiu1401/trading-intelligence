"""
Digital Intern — Maximum-throughput continuous intelligence daemon.

Architecture: independent worker threads, each running their own infinite loop.
No global sleep. Workers run as fast as their sources allow.

Workers:
  W1  gdelt_worker       — full GDELT sweep via collect_gdelt() every 10min
  W2  rss_worker         — re-polls all RSS feeds every 60s
  W3  web_worker         — scrapes 60+ financial sites every 90s
  W4  reddit_worker      — re-polls Reddit every 90s
  W5  ticker_worker      — re-fetches yfinance news every 120s
  W6  scorer_worker      — NN-first urgency scoring; Sonnet only for uncertain articles
  W7  alert_worker       — fires Discord alert whenever urgent items appear
  W8  heartbeat_worker   — posts full Opus briefing every 5h
  W9  purge_worker       — cleans old data every 6h
  W10 ml_trainer_worker  — retrains ArticleNet hourly on accumulated LLM labels
  W11 price_alert_worker — alerts on >3% portfolio moves every 5min
"""
import os
import sys
import time
import signal
import threading
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
sys.path.insert(0, str(BASE_DIR))

# Central structured logger — must import before any other local module
from core.logger import get_logger, record_metric

from collectors.rss_collector import collect_rss
from collectors.gdelt_collector import collect_gdelt, QUERY_GROUPS, _fetch_query
from collectors.ticker_news import collect_ticker_news
from collectors.reddit_collector import collect_reddit
from collectors.web_scraper import scrape_web
from collectors.stock_data import get_stock_data
from collectors.earnings_calendar import get_earnings
from collectors.options_monitor import get_options_data, format_options_block
from collectors.portfolio_pnl import get_portfolio_pnl, format_pnl_block
from collectors import source_health
from triage.heuristic_scorer import score_article as _heuristic_score_article
from analysis.claude_analyst import analyze
from notifier.discord_notifier import send as discord_send
from storage.article_store import ArticleStore
from watchers.urgency_scorer import score_batch
from watchers.alert_agent import send_urgent_alert
from ml.inference import triage_articles
from ml.trainer import train as ml_train

# ── Config ──────────────────────────────────────────────────────────────────
HEARTBEAT_INTERVAL  = 5 * 3600   # 5h
RSS_INTERVAL        = 60          # re-poll RSS every 60s
WEB_INTERVAL        = 90          # scrape web every 90s
REDDIT_INTERVAL     = 90          # re-poll Reddit every 90s
TICKER_INTERVAL     = 120         # re-fetch ticker news every 120s
SCORE_INTERVAL      = 30          # run scoring pass every 30s
ALERT_CHECK         = 20          # check for urgent alerts every 20s
PURGE_INTERVAL      = 6 * 3600   # purge old data every 6h
GDELT_INTERVAL      = 600         # full GDELT sweep every 10min
ML_TRAIN_INTERVAL   = 3600        # retrain ArticleNet every hour
PRICE_ALERT_INTERVAL = 300        # check portfolio prices every 5min
PRICE_ALERT_THRESHOLD = 3.0       # alert when |%| move >= this
WORKER_HEALTH_STALE_SECS = 15 * 60  # mark worker stale in heartbeat if no success in this many seconds

PORTFOLIO_TICKERS = ("LITE", "MU", "MSFT", "AXTI", "ORCL", "TSEM", "QBTS")

log = get_logger("daemon")

_running = True
_store_lock = threading.Lock()
_worker_last_ok: dict[str, float] = {}
_last_prices: dict[str, float] = {}

def _handle_signal(sig, frame):
    global _running
    log.info(f"Signal {sig} — shutting down")
    _running = False

signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT,  _handle_signal)


def _ingest(store: ArticleStore, articles: list, source_tag: str) -> int:
    """Score and insert articles into store. Returns new count."""
    for art in articles:
        result = _heuristic_score_article(
            art.get("title", ""), art.get("summary", ""),
            art.get("source", ""), art.get("published", ""),
        )
        art["_relevance_score"] = result["score"]
        art["_score_detail"] = result
    # filter obvious noise before inserting (heuristic score < 1.5)
    relevant = [a for a in articles if a["_relevance_score"] >= 1.5]
    with _store_lock:
        inserted = store.insert_batch(relevant)
    if inserted:
        log.info(f"[{source_tag}] +{inserted} new articles (from {len(articles)} collected)")
    return inserted


# ── Worker W1: GDELT — full sweep, per-query health tracking ────────────────
def gdelt_worker(store: ArticleStore):
    log.info("[gdelt_worker] started")
    while _running:
        try:
            # Full aggregate sweep (handles cross-query dedup + seen_articles cache)
            articles = collect_gdelt()
            _ingest(store, articles, "gdelt")
            # Per-query health tracking — count articles per query in this sweep
            counts: dict[str, int] = {}
            for art in articles:
                q = art.get("_query") or ""
                if q:
                    counts[q] = counts.get(q, 0) + 1
            for query in QUERY_GROUPS:
                new = counts.get(query, 0)
                try:
                    source_health.record_result(f"gdelt:{query[:20]}", new)
                except Exception as he:
                    log.warning(f"[gdelt_worker] source_health error: {he}")
            _worker_last_ok["gdelt"] = time.time()
        except Exception as e:
            log.warning(f"[gdelt_worker] error: {e}")
        _sleep(GDELT_INTERVAL)


# ── Worker W2: RSS — re-poll every 60s ──────────────────────────────────────
def rss_worker(store: ArticleStore):
    log.info("[rss_worker] started")
    while _running:
        try:
            articles = collect_rss()
            _ingest(store, articles, "rss")
            try:
                source_health.record_result("rss", len(articles))
            except Exception as he:
                log.warning(f"[rss_worker] source_health error: {he}")
            _worker_last_ok["rss"] = time.time()
        except Exception as e:
            log.warning(f"[rss_worker] error: {e}")
        _sleep(RSS_INTERVAL)


# ── Worker W3: Web scraper — 60+ sites every 90s ────────────────────────────
def web_worker(store: ArticleStore):
    log.info("[web_worker] started")
    while _running:
        try:
            articles = scrape_web()
            _ingest(store, articles, "web")
            try:
                source_health.record_result("web", len(articles))
            except Exception as he:
                log.warning(f"[web_worker] source_health error: {he}")
        except Exception as e:
            log.warning(f"[web_worker] error: {e}")
        _sleep(WEB_INTERVAL)


# ── Worker W4: Reddit — re-poll every 90s ────────────────────────────────────
def reddit_worker(store: ArticleStore):
    log.info("[reddit_worker] started")
    while _running:
        try:
            articles = collect_reddit()
            _ingest(store, articles, "reddit")
            try:
                source_health.record_result("reddit", len(articles))
            except Exception as he:
                log.warning(f"[reddit_worker] source_health error: {he}")
        except Exception as e:
            log.warning(f"[reddit_worker] error: {e}")
        _sleep(REDDIT_INTERVAL)


# ── Worker W5: Ticker news — re-fetch every 120s ─────────────────────────────
def ticker_worker(store: ArticleStore):
    log.info("[ticker_worker] started")
    while _running:
        try:
            articles = collect_ticker_news()
            _ingest(store, articles, "ticker")
        except Exception as e:
            log.warning(f"[ticker_worker] error: {e}")
        _sleep(TICKER_INTERVAL)


# ── Worker W6: NN-first scorer — NN handles bulk, Sonnet only for grey zone ──
def scorer_worker(store: ArticleStore):
    log.info("[scorer_worker] started (NN-first mode)")
    from ml.embedder import get_embedder
    while _running:
        try:
            with _store_lock:
                unscored = store.get_unscored(limit=500, min_kw=1.5)

            if unscored:
                log.info(f"[scorer] Scoring {len(unscored)} articles...")
                buckets = triage_articles(unscored)

                batch = []
                for art, sc in buckets["confident"]:
                    aid = art.get("_id")
                    if aid:
                        is_urgent = sc.urgency >= 8.0
                        score = max(sc.relevance, sc.urgency, 0.01)
                        batch.append((aid, score, 1 if is_urgent else 0))
                for art, sc in buckets["noise"]:
                    aid = art.get("_id")
                    if aid:
                        batch.append((aid, max(sc.relevance, 0.01), 0))
                if batch:
                    store.update_ai_scores_batch(batch)

                llm_candidates = [art for art, _ in buckets["uncertain"]]
                record_metric("scorer.nn_bypass_rate",
                              1.0 - len(llm_candidates) / max(len(unscored), 1),
                              {"total": len(unscored), "to_llm": len(llm_candidates)})

                urgent = 0
                if llm_candidates:
                    urgent = score_batch(llm_candidates, store)
                log.info(f"[scorer] Done: {urgent} urgent from {len(llm_candidates)} LLM candidates")

            _worker_last_ok["scorer"] = time.time()

        except Exception as e:
            log.warning(f"[scorer_worker] error: {e}")
        _sleep(SCORE_INTERVAL)


# ── Worker W11: Portfolio price alerts — every 5min ─────────────────────────
def price_alert_worker(store: ArticleStore):
    log.info("[price_alert_worker] started")
    while _running:
        try:
            data = get_stock_data()
            by_ticker = {row["ticker"]: row for row in data.get("equities", [])}
            for tkr in PORTFOLIO_TICKERS:
                row = by_ticker.get(tkr)
                if not row:
                    continue
                price = row.get("price")
                if price is None:
                    continue
                prev = _last_prices.get(tkr)
                if prev is None:
                    _last_prices[tkr] = price
                    continue
                pct = (price - prev) / prev * 100.0 if prev else 0.0
                if abs(pct) >= PRICE_ALERT_THRESHOLD:
                    sign = "+" if pct >= 0 else ""
                    msg = (f"📈 PRICE ALERT: {tkr} {sign}{pct:.1f}% to "
                           f"${price:.2f} (from ${prev:.2f}, "
                           f"{PRICE_ALERT_INTERVAL // 60}min ago)")
                    log.info(f"[price_alert] {msg}")
                    discord_send(msg, is_alert=True)
                _last_prices[tkr] = price
            _worker_last_ok["price_alert"] = time.time()
        except Exception as e:
            log.warning(f"[price_alert_worker] error: {e}")
        _sleep(PRICE_ALERT_INTERVAL)


# ── Worker W10: ML trainer — retrains ArticleNet hourly ─────────────────────
def ml_trainer_worker(store: ArticleStore):
    log.info("[ml_trainer] started")
    # Bootstrap on first run
    _sleep(30)  # let collectors gather some data first
    try:
        log.info("[ml_trainer] Running initial bootstrap training...")
        metrics = ml_train(store)
        log.info(f"[ml_trainer] Bootstrap done: {metrics}")
        record_metric("ml.train.loss", metrics.get("final_loss", 0),
                      {"n": metrics.get("n", 0), "phase": "bootstrap"})
    except Exception as e:
        log.warning(f"[ml_trainer] Bootstrap error: {e}")

    while _running:
        _sleep(ML_TRAIN_INTERVAL)
        try:
            log.info("[ml_trainer] Retraining on accumulated labels...")
            metrics = ml_train(store)
            log.info(f"[ml_trainer] Retrain: n={metrics.get('n')} "
                     f"loss={metrics.get('final_loss', 0):.4f} "
                     f"elapsed={metrics.get('elapsed_s', 0):.0f}s")
            record_metric("ml.train.loss", metrics.get("final_loss", 0),
                          {"n": metrics.get("n", 0), "phase": "retrain"})
        except Exception as e:
            log.warning(f"[ml_trainer] Retrain error: {e}")


# ── Worker W7: Alert dispatcher — checks every 20s ──────────────────────────
def alert_worker(store: ArticleStore):
    log.info("[alert_worker] started")
    while _running:
        try:
            with _store_lock:
                urgent = store.get_unalerted_urgent()
            if urgent:
                log.info(f"[alert] {len(urgent)} urgent items → dispatching")
                send_urgent_alert(urgent, store)
        except Exception as e:
            log.warning(f"[alert_worker] error: {e}")
        _sleep(ALERT_CHECK)


# ── Worker W8: Heartbeat briefing every 5h ──────────────────────────────────
def heartbeat_worker(store: ArticleStore):
    log.info("[heartbeat_worker] started")
    last = 0.0  # trigger immediately on start

    while _running:
        now = time.time()
        if now - last >= HEARTBEAT_INTERVAL:
            try:
                log.info("[heartbeat] Generating Opus 4.7 briefing...")
                with _store_lock:
                    top = store.get_top_for_briefing(hours=5, limit=50)

                stocks   = get_stock_data()
                earnings = get_earnings()
                opts     = get_options_data()
                opts_blk = format_options_block(opts)

                try:
                    pnl = get_portfolio_pnl()
                except Exception as pe:
                    log.warning(f"[heartbeat] portfolio P&L error: {pe}")
                    pnl = None
                if pnl is not None:
                    top = [{
                        "title": "PORTFOLIO P&L SNAPSHOT",
                        "source": "portfolio",
                        "summary": format_pnl_block(pnl),
                        "ai_score": 10,
                    }] + top

                if opts_blk and opts_blk != "N/A":
                    top = [{"title": "OPTIONS SNAPSHOT", "source": "options_monitor",
                             "summary": opts_blk, "ai_score": 10}] + top

                briefing = analyze(top, stocks, earnings)
                health_line = _build_health_line(store)
                message = briefing.rstrip() + "\n\n" + health_line
                ok = discord_send(message)
                log.info(f"[heartbeat] {'sent' if ok else 'FAILED'} ({len(message)} chars)")
                last = time.time()
            except Exception as e:
                log.error(f"[heartbeat] error: {e}")
                last = time.time() - HEARTBEAT_INTERVAL + 300  # retry in 5min
        _sleep(30)


# ── Worker W9: Purge old data every 6h ──────────────────────────────────────
def purge_worker(store: ArticleStore):
    log.info("[purge_worker] started")
    while _running:
        _sleep(PURGE_INTERVAL)
        try:
            with _store_lock:
                store.purge_old()
            stats = store.stats()
            log.info(f"[purge] DB: {stats['total']} articles, {stats['db_mb']}MB")
        except Exception as e:
            log.warning(f"[purge_worker] error: {e}")


# ── Stats reporter — every 60s ───────────────────────────────────────────────
def stats_worker(store: ArticleStore):
    while _running:
        _sleep(60)
        try:
            s = store.stats()
            log.info(f"[stats] total={s['total']} urgent={s['urgent']} "
                     f"unscored={s['unscored']} db={s['db_mb']}MB")
        except Exception:
            pass


def _sleep(seconds: float):
    """Interruptible sleep — checks _running every 0.5s."""
    deadline = time.time() + seconds
    while _running and time.time() < deadline:
        time.sleep(0.5)


def _build_health_line(store: ArticleStore) -> str:
    now = time.time()
    tracked = ("gdelt", "rss", "scorer", "price_alert")
    parts = []
    for name in tracked:
        last_ok = _worker_last_ok.get(name)
        ok = last_ok is not None and (now - last_ok) < WORKER_HEALTH_STALE_SECS
        parts.append(f"{name}={'✓' if ok else '✗'}")
    try:
        day = store.stats_since(24)
        suffix = f" [{day['total']} articles today, {day['urgent']} urgent]"
    except Exception:
        suffix = ""
    return "⚙ Workers: " + " ".join(parts) + suffix


def main():
    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    log.info(" DIGITAL INTERN DAEMON — STARTING")
    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    store = ArticleStore()
    log.info(f"Store ready: {store.stats()}")

    workers = [
        ("gdelt",       gdelt_worker),
        ("rss",         rss_worker),
        ("web",         web_worker),
        ("reddit",      reddit_worker),
        ("ticker",      ticker_worker),
        ("scorer",      scorer_worker),
        ("alert",       alert_worker),
        ("heartbeat",   heartbeat_worker),
        ("purge",       purge_worker),
        ("stats",       stats_worker),
        ("ml_trainer",  ml_trainer_worker),
        ("price_alert", price_alert_worker),
    ]

    threads = []
    for name, fn in workers:
        t = threading.Thread(target=fn, args=(store,), name=name, daemon=True)
        t.start()
        threads.append(t)
        log.info(f"[daemon] Worker '{name}' started")

    log.info(f"[daemon] All {len(threads)} workers running — max throughput mode")

    # Main thread just keeps process alive and monitors workers
    while _running:
        time.sleep(5)
        for t in threads:
            if not t.is_alive():
                log.error(f"[daemon] Worker '{t.name}' died — respawning")
                fn = next(f for n, f in workers if n == t.name)
                new_t = threading.Thread(target=fn, args=(store,), name=t.name, daemon=True)
                new_t.start()
                threads[threads.index(t)] = new_t

    log.info("[daemon] Shutdown complete")


if __name__ == "__main__":
    main()
