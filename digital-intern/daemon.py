"""
Digital Intern — Maximum-throughput continuous intelligence daemon.

Architecture: independent worker threads, each running their own infinite loop.
No global sleep. Workers run as fast as their sources allow.

Workers (intervals below track the *_INTERVAL constants in the Config block):
  W1  gdelt_worker       — full GDELT sweep via collect_gdelt() every 10min
  W2  rss_worker         — re-polls all RSS feeds every 30s
  W3  web_worker         — scrapes 100+ financial sites every 60s
  W4  reddit_worker      — re-polls Reddit every 45s
  W5  ticker_worker      — re-fetches yfinance news every 60s
  W6  scorer_worker      — NN-first urgency scoring; Sonnet only for uncertain articles
  W7  alert_worker       — fires Discord alert whenever urgent items appear
  W8  heartbeat_worker   — posts full Opus briefing every 5h
  W9  purge_worker       — cleans old data every 6h
  W10 ml_trainer_worker  — retrains ArticleNet every 3min on accumulated LLM labels
  W11 price_alert_worker — alerts on >3% portfolio moves every 5min
  W12 continuous_trainer_worker — lightweight 40-epoch GPU pass every 2min to keep RTX 3060 hot
  W12b recursive_labeler_worker — three-tier Claude labeling (Sonnet+Opus) every 4h
  W13 sec_edgar_worker   — SEC 8-K RSS feed for portfolio/watchlist tickers every 5min
  W14 google_news_worker — round-robin Google News RSS per portfolio ticker every 2min
  W15 portfolio_pl_worker — writes data/portfolio_pl.json every 5min via yfinance
  W16 sentiment_trends_worker — writes data/sentiment_trends.json every 10min
  W17 web_server_worker  — Flask dashboard bound to 0.0.0.0:8080
"""
import json
import logging
import os
import re
import sys
import time
import signal
import tempfile
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
from collectors.gdelt_collector import collect_gdelt, QUERY_GROUPS
from collectors.ticker_news import collect_ticker_news
from collectors.reddit_collector import collect_reddit
from collectors.web_scraper import scrape_web
from collectors.stock_data import get_stock_data
from collectors.earnings_calendar import get_earnings
from collectors.options_monitor import get_options_data, format_options_block
from collectors.portfolio_pnl import get_portfolio_pnl, format_pnl_block, write_pl_snapshot
from collectors.sec_edgar import collect_sec_edgar, collect_sec_edgar_fulltext
from collectors.google_news import collect_google_news
from collectors.nitter_collector import collect_nitter
from collectors.substack_collector import collect_substack
from collectors.finnhub_collector import collect_finnhub
from collectors.alphavantage_collector import collect_alphavantage
from collectors.polygon_collector import collect_polygon
from collectors.massive_collector import collect_massive
from collectors.newsapi_collector import collect_newsapi
from collectors.yahoo_ticker_rss import collect_yahoo_ticker_rss
from collectors.wikipedia_collector import collect_wikipedia
from collectors.macro_calendar_collector import collect_macro_calendar
from collectors.congress_trades_collector import collect_congress_trades
from collectors.finra_short_volume import collect_finra_short_volume
from collectors.market_movers import collect_market_movers
from collectors import source_health
from core.backoff import Backoff
from triage.heuristic_scorer import score_article as _heuristic_score_article
from analysis.claude_analyst import analyze
from notifier.discord_notifier import send as discord_send
from storage.article_store import ArticleStore
from watchers.urgency_scorer import score_batch, BATCH_SIZE as URGENCY_BATCH_SIZE
from watchers.alert_agent import send_urgent_alert
from ml.inference import score_articles
from ml.sentiment_trends import write_trends as write_score_trends
from ml.trainer import train as ml_train
from ml.trainer import train_continuous
from ml.recursive_labeler import run_recursive_labeling
from core.retrain_guard import should_alert as _ml_retrain_should_alert
from core.retrain_guard import alert_message as _ml_retrain_alert_message

# ── Config ──────────────────────────────────────────────────────────────────
HEARTBEAT_INTERVAL  = 5 * 3600   # 5h
RSS_INTERVAL        = 30          # re-poll RSS every 30s (collector is parallelized)
WEB_INTERVAL        = 60          # scrape web every 60s
REDDIT_INTERVAL     = 45          # re-poll Reddit every 45s
TICKER_INTERVAL     = 60          # re-fetch ticker news every 60s
SCORE_INTERVAL      = 30          # run scoring pass every 30s
ALERT_CHECK         = 20          # check for urgent alerts every 20s
PURGE_INTERVAL      = 6 * 3600   # purge old data every 6h
GDELT_INTERVAL      = 600         # full GDELT sweep every 10min
ML_TRAIN_INTERVAL   = 1800        # full ArticleNet retrain every 30 min (GPU)
CONTINUOUS_TRAIN_INTERVAL = 600   # lightweight GPU pass every 10 min
RECURSIVE_LABEL_INTERVAL = 4 * 3600  # recursive Claude labeling every 4h
PRICE_ALERT_INTERVAL = 300        # check portfolio prices every 5min
PRICE_ALERT_THRESHOLD = 3.0       # alert when |%| move >= this
WORKER_HEALTH_STALE_SECS = 15 * 60  # mark worker stale in heartbeat if no success in this many seconds
SEC_EDGAR_INTERVAL  = 300         # SEC 8-K RSS sweep every 5min
SEC_EDGAR_FT_INTERVAL = 900       # SEC full-text per-ticker every 15min
GOOGLE_NEWS_INTERVAL = 120        # Google News per-ticker pass every 2min
NITTER_INTERVAL     = 180         # Nitter twitter mirror every 3min
SUBSTACK_INTERVAL   = 600         # Substack newsletters every 10min
FINNHUB_INTERVAL    = 300         # Finnhub per-ticker company news every 5min
ALPHAVANTAGE_INTERVAL = 1800      # AlphaVantage NEWS_SENTIMENT every 30min (free=25/day)
POLYGON_INTERVAL    = 600         # Polygon news per-ticker every 10min (free=5/min)
MASSIVE_INTERVAL    = 600         # Massive.com news per-ticker every 10min
NEWSAPI_INTERVAL    = 1500        # NewsAPI keyword search every 25min (free=100/day)
YAHOO_TICKER_RSS_INTERVAL = 240   # Yahoo per-ticker RSS every 4min
WIKIPEDIA_INTERVAL  = 600         # Wikipedia recent-changes filter every 10min
MACRO_CALENDAR_INTERVAL = 3600    # FOMC/BLS macro event calendar — once per hour
FINRA_SHORT_INTERVAL    = 3600    # FINRA RegSHO short volume — once per hour (daily file)
CONGRESS_TRADES_INTERVAL = 3600   # Congressional trading disclosures — once per hour
MARKET_MOVERS_INTERVAL  = 300     # Yahoo Finance gainers/losers/most-active every 5min
PORTFOLIO_PL_INTERVAL = 300       # rewrite portfolio_pl.json every 5min
SENTIMENT_TRENDS_INTERVAL = 600   # rewrite sentiment_trends.json every 10min
EXPORT_INTERVAL     = 30 * 60     # training-data export to USB every 30min
WEB_SERVER_PORT     = int(os.environ.get("WEB_SERVER_PORT", "8080"))
WEB_SERVER_HOST     = os.environ.get("WEB_SERVER_HOST", "0.0.0.0")

# Active portfolio + watchlist tickers used for price alerts and relevance boosts.
# Keep in sync with config/portfolio.json (positions + sector_watchlist).
PORTFOLIO_TICKERS = (
    "LITE", "LNOK", "MUU", "DRAM", "SNDU",
    "MU", "MSFT", "AXTI", "ORCL", "TSEM", "QBTS", "NVDA",
)

log = get_logger("daemon")

_running = True
_store_lock = threading.Lock()
_worker_last_ok: dict[str, float] = {}
_last_prices: dict[str, float] = {}

# ── Supervisor state (one entry per worker) ─────────────────────────────────
# Tracks crash history, current health state, and pending respawn timing so
# the main supervisor loop can react smarter than "restart immediately".
CRASH_WINDOW_SECS = 300           # crash counts evaluated over this window
DEGRADED_THRESHOLD = 3            # 3+ crashes in window → degraded
DISABLED_THRESHOLD = 10           # 10+ crashes in window → disabled
DEGRADED_BACKOFF_SECS = 60        # respawn delay while degraded
DISABLED_DURATION_SECS = 30 * 60  # how long a disabled worker stays off
OOM_BACKOFF_SECS = 30             # respawn delay after a MemoryError
HEALTH_REPORT_INTERVAL_SECS = 300 # write per-worker health entry every 5min

_supervisor_lock = threading.Lock()
_worker_crashes: dict[str, list[float]] = {}        # timestamps of recent crashes
_worker_state: dict[str, str] = {}                  # "ok" | "degraded" | "disabled"
_worker_disabled_until: dict[str, float] = {}       # epoch when re-enable is allowed
_worker_respawn_at: dict[str, float] = {}           # epoch when respawn should fire
_worker_last_exception: dict[str, str] = {}         # last exception class name
_worker_total_crashes: dict[str, int] = {}          # cumulative crash count since daemon start

SUPERVISOR_STATE_PATH = BASE_DIR / "logs" / "supervisor_state.json"

# Names of all worker threads — kept in sync with `workers` in main(). Used
# by the health reporter so dashboards know what to expect even when a worker
# has never logged anything yet.
ALL_WORKERS = (
    "gdelt", "rss", "web", "reddit", "ticker", "sec_edgar", "sec_edgar_ft",
    "google_news", "nitter", "substack",
    "finnhub", "alphavantage", "polygon", "massive", "newsapi",
    "yahoo_ticker_rss", "market_movers", "wikipedia", "macro_calendar",
    "scorer", "alert", "heartbeat", "purge", "stats",
    "ml_trainer", "continuous_trainer", "recursive_labeler", "price_alert",
    "portfolio_pl", "sentiment_trends", "export", "web_server",
)
# If all of these are stale at once the dashboard shows the CRITICAL banner.
CORE_WORKERS = ("rss", "web", "reddit", "scorer")

# Expected seconds between consecutive `_worker_last_ok` pings for each worker,
# i.e. its natural poll cadence. Liveness is judged relative to THIS, not a
# single global window: a fixed 15min deadline wrongly flagged every slow
# worker DEAD forever (alphavantage 30min, newsapi 25min, recursive_labeler 4h)
# — pure log/dashboard noise that also inflated the `dead=N` rollup. Workers
# absent from this map (e.g. web_server, which never pings) fall back to the
# floor, preserving the previous behaviour for them.
WORKER_POLL_INTERVAL_SECS = {
    "gdelt": GDELT_INTERVAL, "rss": RSS_INTERVAL, "web": WEB_INTERVAL,
    "reddit": REDDIT_INTERVAL, "ticker": TICKER_INTERVAL,
    "sec_edgar": SEC_EDGAR_INTERVAL, "sec_edgar_ft": SEC_EDGAR_FT_INTERVAL,
    "google_news": GOOGLE_NEWS_INTERVAL, "nitter": NITTER_INTERVAL,
    "substack": SUBSTACK_INTERVAL, "finnhub": FINNHUB_INTERVAL,
    "alphavantage": ALPHAVANTAGE_INTERVAL, "polygon": POLYGON_INTERVAL,
    "massive": MASSIVE_INTERVAL, "newsapi": NEWSAPI_INTERVAL,
    "yahoo_ticker_rss": YAHOO_TICKER_RSS_INTERVAL,
    "market_movers": MARKET_MOVERS_INTERVAL,
    "wikipedia": WIKIPEDIA_INTERVAL, "macro_calendar": MACRO_CALENDAR_INTERVAL,
    "scorer": SCORE_INTERVAL,
    "alert": ALERT_CHECK, "heartbeat": 60, "purge": 300, "stats": 60,
    "ml_trainer": ML_TRAIN_INTERVAL,
    "continuous_trainer": CONTINUOUS_TRAIN_INTERVAL,
    "recursive_labeler": RECURSIVE_LABEL_INTERVAL,
    "price_alert": PRICE_ALERT_INTERVAL,
    "portfolio_pl": PORTFOLIO_PL_INTERVAL,
    "sentiment_trends": SENTIMENT_TRENDS_INTERVAL,
    "export": EXPORT_INTERVAL,
}
# A worker is DEAD only after missing well over one full cycle, so a single
# slow upstream poll never trips it. Floor keeps the old 15min minimum so a
# fast worker (30s cadence) silent for 15min is still correctly flagged.
LIVENESS_MULTIPLIER  = 2.5
LIVENESS_FLOOR_SECS  = 15 * 60


def _worker_liveness_deadline(name: str) -> float:
    """Max seconds a worker may go without a success ping before it is DEAD.

    Scales with the worker's own poll cadence (``LIVENESS_MULTIPLIER`` cycles)
    but never drops below ``LIVENESS_FLOOR_SECS``."""
    interval = WORKER_POLL_INTERVAL_SECS.get(name)
    if interval is None:
        return float(LIVENESS_FLOOR_SECS)
    return max(float(LIVENESS_FLOOR_SECS), LIVENESS_MULTIPLIER * float(interval))

_shutdown_signal = 0  # set by _handle_signal, logged from the main thread

def _handle_signal(sig, frame):
    # Async-signal-safe ONLY. Do NOT call logging here: the signal can land
    # mid-write inside a log handler's BufferedWriter, and re-entering logging
    # from the handler raises "RuntimeError: reentrant call inside
    # <_io.BufferedWriter>", which crashed the shutdown path and pushed the
    # service onto the SIGTERM-timeout -> SIGKILL escalation (orphaned flock
    # holder + sqlite corruption risk). Only touch module globals here; the
    # main supervisor thread logs the signal once it observes the flag.
    global _running, _shutdown_signal
    _shutdown_signal = sig
    _running = False

signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT,  _handle_signal)


# ── Memory / GPU OOM handling ───────────────────────────────────────────────
def _handle_memory_error(worker_name: str) -> None:
    """Free GPU cache (best-effort) and log an OOM event with the worker tag.

    Called by the inner exception handlers in GPU-touching workers (scorer,
    ml_trainer, continuous_trainer) so the worker keeps running, and by the
    wrapper as a fallback if the exception escapes."""
    log.error(f"[{worker_name}] MemoryError — clearing GPU cache, backing off {OOM_BACKOFF_SECS}s")
    _worker_last_exception[worker_name] = "MemoryError"
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        log.warning(f"[{worker_name}] torch.cuda.empty_cache() failed: {e}")
    # Brief pause before the caller's loop resumes; the supervisor uses the
    # same OOM_BACKOFF if the worker thread exits entirely.
    _sleep(OOM_BACKOFF_SECS)


def _wrap_worker(name: str, fn):
    """Wrap a worker entry-point to capture escaped exceptions for the supervisor.

    Each worker has its own inner ``except`` handler so most failures never
    reach this wrapper. When something *does* escape, we record the exception
    class so the supervisor can apply the right respawn policy (OOM backoff
    vs normal). The wrapper does NOT re-raise — letting the thread exit is
    what triggers the supervisor's respawn path."""
    def _runner(store: ArticleStore):
        try:
            fn(store)
        except MemoryError:
            log.error(f"[{name}] thread exited with MemoryError")
            _worker_last_exception[name] = "MemoryError"
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        except Exception as e:
            log.error(f"[{name}] thread exited with {type(e).__name__}: {e}", exc_info=True)
            _worker_last_exception[name] = type(e).__name__
    _runner.__name__ = f"wrapped_{name}"
    return _runner


# ── Supervisor crash tracking & state transitions ───────────────────────────
def _record_crash(name: str) -> tuple[int, str, str]:
    """Record a crash for ``name``; return (count_in_window, old_state, new_state).

    State machine:
      - >= DISABLED_THRESHOLD crashes in window → "disabled" (off for 30min)
      - >= DEGRADED_THRESHOLD crashes in window → "degraded" (slow respawn)
      - otherwise → "ok"
    Discord alerts are emitted by the caller, only on transitions."""
    now = time.time()
    with _supervisor_lock:
        crashes = _worker_crashes.setdefault(name, [])
        crashes.append(now)
        # keep only crashes within the rolling window
        crashes[:] = [t for t in crashes if now - t < CRASH_WINDOW_SECS]
        count = len(crashes)
        _worker_total_crashes[name] = _worker_total_crashes.get(name, 0) + 1

        old_state = _worker_state.get(name, "ok")
        if count >= DISABLED_THRESHOLD:
            new_state = "disabled"
            _worker_disabled_until[name] = now + DISABLED_DURATION_SECS
        elif count >= DEGRADED_THRESHOLD:
            new_state = "degraded"
        else:
            new_state = "ok"
        _worker_state[name] = new_state
    return count, old_state, new_state


def _notify_state_transition(name: str, old: str, new: str, count: int) -> None:
    """Send Discord alert on degraded/disabled transitions. Throttled by state
    transitions — the caller only invokes this when old != new."""
    if new == "degraded":
        msg = (f"⚠️ Worker `{name}` degraded — {count} crashes in last "
               f"{CRASH_WINDOW_SECS // 60}min. Increasing backoff to "
               f"{DEGRADED_BACKOFF_SECS}s.")
    elif new == "disabled":
        msg = (f"🛑 Worker `{name}` disabled — {count} crashes in last "
               f"{CRASH_WINDOW_SECS // 60}min. Disabled for "
               f"{DISABLED_DURATION_SECS // 60}min.")
    elif old in ("degraded", "disabled") and new == "ok":
        msg = f"✅ Worker `{name}` recovered (state: {old} → ok)."
    else:
        return
    try:
        discord_send(msg, is_alert=True)
    except Exception as e:
        log.warning(f"[supervisor] Discord notification failed: {e}")


def _compute_respawn_delay(name: str, now: float) -> float:
    """How long the supervisor should wait before respawning ``name``.

    Read the three supervisor maps under ``_supervisor_lock`` so we never
    observe a torn state (e.g., ``_worker_state`` updated to 'disabled' but
    ``_worker_disabled_until`` not yet written by the concurrent
    ``_record_crash`` call). The lock is held by ``_record_crash`` for the
    duration of its updates; reading without it can race.
    """
    with _supervisor_lock:
        disabled_until = _worker_disabled_until.get(name, 0.0)
        last_exc = _worker_last_exception.get(name)
        state = _worker_state.get(name)
    if disabled_until > now:
        return disabled_until - now
    if last_exc == "MemoryError":
        return OOM_BACKOFF_SECS
    if state == "degraded":
        return DEGRADED_BACKOFF_SECS
    return 1.0  # tiny pause so we don't hot-loop on a flapping worker


def _worker_health_snapshot(now: float | None = None) -> dict:
    """Compute the current health view of all workers — used by both the
    structured log entries and the JSON snapshot for the dashboard."""
    if now is None:
        now = time.time()
    workers = []
    workers_ok = 0
    workers_dead = 0
    for name in ALL_WORKERS:
        with _supervisor_lock:
            crashes = list(_worker_crashes.get(name, []))
            state = _worker_state.get(name, "ok")
            disabled_until = _worker_disabled_until.get(name, 0.0)
            total_crashes = _worker_total_crashes.get(name, 0)
            last_exc = _worker_last_exception.get(name, "")
        crashes_5m = sum(1 for t in crashes if now - t < CRASH_WINDOW_SECS)
        last_ok = _worker_last_ok.get(name)
        age_s = (now - last_ok) if last_ok else None
        # "alive" semantics for the rollup: state != disabled AND we've seen a
        # success ping within this worker's own liveness deadline (scaled to
        # its poll cadence), or we have no ping yet but state is ok.
        if state == "disabled":
            alive = False
        elif age_s is None:
            alive = state == "ok"
        else:
            alive = age_s < _worker_liveness_deadline(name)
        if alive:
            workers_ok += 1
        else:
            workers_dead += 1
        workers.append({
            "name": name,
            "state": state,
            "alive": alive,
            "crashes_5m": crashes_5m,
            "total_crashes": total_crashes,
            "last_exception": last_exc,
            "last_ok_age_s": age_s,
            "disabled_until": disabled_until if disabled_until > now else 0,
        })
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "workers_ok": workers_ok,
        "workers_dead": workers_dead,
        "workers": workers,
    }


def _write_supervisor_state(snapshot: dict) -> None:
    """Atomically dump the supervisor state for the dashboard to read."""
    try:
        SUPERVISOR_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        # tempfile + rename = atomic from the reader's perspective
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=str(SUPERVISOR_STATE_PATH.parent),
            prefix=".supervisor_state.", suffix=".tmp", delete=False,
        ) as tf:
            json.dump(snapshot, tf, ensure_ascii=False)
            tmp_path = Path(tf.name)
        os.replace(tmp_path, SUPERVISOR_STATE_PATH)
    except Exception as e:
        log.warning(f"[supervisor] failed to write supervisor_state.json: {e}")


def _worker_health_report() -> None:
    """Emit one [worker] alive entry per worker plus a rollup; persist a JSON
    snapshot. Called every HEALTH_REPORT_INTERVAL_SECS from the main loop."""
    snapshot = _worker_health_snapshot()
    log.info(
        f"[daemon] health_report ok={snapshot['workers_ok']} "
        f"dead={snapshot['workers_dead']}",
        extra={
            "workers_ok": snapshot["workers_ok"],
            "workers_dead": snapshot["workers_dead"],
            "event": "health_report",
        },
    )
    for w in snapshot["workers"]:
        # Reflect the *computed* liveness, not the raw supervisor state. A
        # worker can have state=ok yet be counted in dead= because it has not
        # pinged success in >15min; logging "alive state=ok" for it made the
        # dead workers impossible to identify from the logs.
        is_alive = w.get("alive", True)
        age_s = w.get("last_ok_age_s")
        age_txt = f"{age_s:.0f}s" if isinstance(age_s, (int, float)) else "n/a"
        liveness = "alive" if is_alive else "DEAD"
        msg = (
            f"[{w['name']}] {liveness} state={w['state']} "
            f"crashes_5m={w['crashes_5m']} last_ok={age_txt}"
        )
        extra = {
            "event": "worker_alive" if is_alive else "worker_dead",
            "worker": w["name"],
            "alive": is_alive,
            "state": w["state"],
            "crashes_5m": w["crashes_5m"],
            "total_crashes": w["total_crashes"],
            "last_ok_age_s": age_s,
        }
        # WARNING for dead workers so they surface above the INFO heartbeat
        # noise; INFO keeps healthy pings visible in the rotated logs.
        if is_alive:
            log.info(msg, extra=extra)
        else:
            log.warning(msg, extra=extra)
    _write_supervisor_state(snapshot)


def _ingest(store: ArticleStore, articles: list, source_tag: str) -> int:
    """Score and insert articles into store. Returns new count."""
    for art in articles:
        result = _heuristic_score_article(
            art.get("title", ""), art.get("summary", ""),
            art.get("source", ""), art.get("published", ""),
        )
        art["_relevance_score"] = result["score"]
        art["_score_detail"] = result
    # filter obvious noise before inserting (heuristic score < 0.5)
    relevant = [a for a in articles if a["_relevance_score"] >= 0.5]
    with _store_lock:
        inserted = store.insert_batch(relevant)
    if inserted:
        log.info(f"[{source_tag}] +{inserted} new articles (from {len(articles)} collected)")
    return inserted


# ── Worker W1: GDELT — full sweep, per-query health tracking ────────────────
def gdelt_worker(store: ArticleStore):
    log.info("[gdelt_worker] started")
    bo = Backoff("gdelt", base=5.0, cap=300.0)
    while _running:
        try:
            # Full aggregate sweep (handles cross-query dedup + seen_articles cache)
            articles = collect_gdelt()
            _ingest(store, articles, "gdelt")
            # Aggregate health only. Per-query `gdelt:<query>` keys were
            # meaningless under cross-query/persistent dedup: collect_gdelt()
            # returns only net-new articles, so almost every query nets 0 per
            # sweep and tripped the 3-failure disable threshold while GDELT
            # was perfectly healthy — burying real down-sources in the hourly
            # alert. Track one "gdelt" key like every other worker.
            try:
                source_health.record_result("gdelt", len(articles))
            except Exception as he:
                log.warning(f"[gdelt_worker] source_health error: {he}")
            # Per-query breakdown kept as debug-only observability (which
            # queries are still productive) without feeding the disable logic.
            if log.isEnabledFor(logging.DEBUG):
                counts: dict[str, int] = {}
                for art in articles:
                    q = art.get("_query") or ""
                    if q:
                        counts[q] = counts.get(q, 0) + 1
                if counts:
                    top = sorted(counts.items(), key=lambda kv: -kv[1])[:8]
                    log.debug("[gdelt_worker] per-query new: "
                              + ", ".join(f"{q[:20]}={n}" for q, n in top))
            _worker_last_ok["gdelt"] = time.time()
            bo.reset()
        except Exception as e:
            log.warning(f"[gdelt_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(GDELT_INTERVAL)


# ── Worker W2: RSS — re-poll every 60s ──────────────────────────────────────
def rss_worker(store: ArticleStore):
    log.info("[rss_worker] started")
    bo = Backoff("rss", base=5.0, cap=300.0)
    while _running:
        try:
            articles = collect_rss()
            _ingest(store, articles, "rss")
            try:
                source_health.record_result("rss", len(articles))
            except Exception as he:
                log.warning(f"[rss_worker] source_health error: {he}")
            _worker_last_ok["rss"] = time.time()
            bo.reset()
        except Exception as e:
            log.warning(f"[rss_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(RSS_INTERVAL)


# ── Worker W3: Web scraper — 60+ sites every 90s ────────────────────────────
def web_worker(store: ArticleStore):
    log.info("[web_worker] started")
    bo = Backoff("web", base=5.0, cap=300.0)
    while _running:
        try:
            articles = scrape_web()
            _ingest(store, articles, "web")
            try:
                source_health.record_result("web", len(articles))
            except Exception as he:
                log.warning(f"[web_worker] source_health error: {he}")
            _worker_last_ok["web"] = time.time()
            bo.reset()
        except Exception as e:
            log.warning(f"[web_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(WEB_INTERVAL)


# ── Worker W4: Reddit — re-poll every 90s ────────────────────────────────────
def reddit_worker(store: ArticleStore):
    log.info("[reddit_worker] started")
    bo = Backoff("reddit", base=5.0, cap=300.0)
    while _running:
        try:
            articles = collect_reddit()
            _ingest(store, articles, "reddit")
            try:
                source_health.record_result("reddit", len(articles))
            except Exception as he:
                log.warning(f"[reddit_worker] source_health error: {he}")
            _worker_last_ok["reddit"] = time.time()
            bo.reset()
        except Exception as e:
            log.warning(f"[reddit_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(REDDIT_INTERVAL)


# ── Worker W5: Ticker news — re-fetch every 120s ─────────────────────────────
def ticker_worker(store: ArticleStore):
    log.info("[ticker_worker] started")
    bo = Backoff("ticker", base=5.0, cap=300.0)
    while _running:
        try:
            articles = collect_ticker_news()
            _ingest(store, articles, "ticker")
            _worker_last_ok["ticker"] = time.time()
            bo.reset()
        except Exception as e:
            log.warning(f"[ticker_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(TICKER_INTERVAL)


# ── Worker: SEC EDGAR 8-K — every 5min ──────────────────────────────────────
def sec_edgar_worker(store: ArticleStore):
    log.info("[sec_edgar_worker] started")
    bo = Backoff("sec_edgar", base=5.0, cap=300.0)
    while _running:
        try:
            articles = collect_sec_edgar()
            _ingest(store, articles, "sec_edgar")
            try:
                source_health.record_result("sec_edgar", len(articles))
            except Exception as he:
                log.warning(f"[sec_edgar_worker] source_health error: {he}")
            _worker_last_ok["sec_edgar"] = time.time()
            bo.reset()
        except Exception as e:
            log.warning(f"[sec_edgar_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(SEC_EDGAR_INTERVAL)


# ── Worker: Google News per-ticker — every 5min ─────────────────────────────
def google_news_worker(store: ArticleStore):
    log.info("[google_news_worker] started")
    bo = Backoff("google_news", base=5.0, cap=300.0)
    while _running:
        try:
            articles = collect_google_news()
            _ingest(store, articles, "google_news")
            try:
                source_health.record_result("google_news", len(articles))
            except Exception as he:
                log.warning(f"[google_news_worker] source_health error: {he}")
            _worker_last_ok["google_news"] = time.time()
            bo.reset()
        except Exception as e:
            log.warning(f"[google_news_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(GOOGLE_NEWS_INTERVAL)


# ── Worker: SEC EDGAR full-text per-ticker — every 15min ────────────────────
def sec_edgar_ft_worker(store: ArticleStore):
    log.info("[sec_edgar_ft_worker] started")
    bo = Backoff("sec_edgar_ft", base=10.0, cap=600.0)
    while _running:
        try:
            articles = collect_sec_edgar_fulltext()
            _ingest(store, articles, "sec_edgar_ft")
            try:
                source_health.record_result("sec_edgar_ft", len(articles))
            except Exception as he:
                log.warning(f"[sec_edgar_ft_worker] source_health error: {he}")
            _worker_last_ok["sec_edgar_ft"] = time.time()
            bo.reset()
        except Exception as e:
            log.warning(f"[sec_edgar_ft_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(SEC_EDGAR_FT_INTERVAL)


# ── Worker: Nitter / Twitter — every 3min ───────────────────────────────────
def nitter_worker(store: ArticleStore):
    log.info("[nitter_worker] started")
    bo = Backoff("nitter", base=10.0, cap=600.0)
    while _running:
        try:
            articles = collect_nitter()
            _ingest(store, articles, "nitter")
            try:
                source_health.record_result("nitter", len(articles))
            except Exception as he:
                log.warning(f"[nitter_worker] source_health error: {he}")
            _worker_last_ok["nitter"] = time.time()
            bo.reset()
        except Exception as e:
            log.warning(f"[nitter_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(NITTER_INTERVAL)


# ── Worker: Substack newsletters — every 10min ──────────────────────────────
def substack_worker(store: ArticleStore):
    log.info("[substack_worker] started")
    bo = Backoff("substack", base=10.0, cap=600.0)
    while _running:
        try:
            articles = collect_substack()
            _ingest(store, articles, "substack")
            try:
                source_health.record_result("substack", len(articles))
            except Exception as he:
                log.warning(f"[substack_worker] source_health error: {he}")
            _worker_last_ok["substack"] = time.time()
            bo.reset()
        except Exception as e:
            log.warning(f"[substack_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(SUBSTACK_INTERVAL)


# ── Worker: Finnhub per-ticker company news — every 5min ────────────────────
def finnhub_worker(store: ArticleStore):
    log.info("[finnhub_worker] started")
    bo = Backoff("finnhub", base=10.0, cap=600.0)
    while _running:
        try:
            articles = collect_finnhub()
            _ingest(store, articles, "finnhub")
            try:
                source_health.record_result("finnhub", len(articles))
            except Exception as he:
                log.warning(f"[finnhub_worker] source_health error: {he}")
            _worker_last_ok["finnhub"] = time.time()
            log.debug(f"[finnhub] cycle ok ({len(articles)} new)")
            bo.reset()
        except Exception as e:
            log.warning(f"[finnhub_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(FINNHUB_INTERVAL)


# ── Worker: Alpha Vantage NEWS_SENTIMENT — every 30min (quota=25/day) ───────
def alphavantage_worker(store: ArticleStore):
    log.info("[alphavantage_worker] started")
    bo = Backoff("alphavantage", base=30.0, cap=1800.0)
    while _running:
        try:
            articles = collect_alphavantage()
            _ingest(store, articles, "alphavantage")
            try:
                source_health.record_result("alphavantage", len(articles))
            except Exception as he:
                log.warning(f"[alphavantage_worker] source_health error: {he}")
            _worker_last_ok["alphavantage"] = time.time()
            log.debug(f"[alphavantage] cycle ok ({len(articles)} new)")
            bo.reset()
        except Exception as e:
            log.warning(f"[alphavantage_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(ALPHAVANTAGE_INTERVAL)


# ── Worker: Polygon news — every 10min ──────────────────────────────────────
def polygon_worker(store: ArticleStore):
    log.info("[polygon_worker] started")
    bo = Backoff("polygon", base=15.0, cap=900.0)
    while _running:
        try:
            articles = collect_polygon()
            _ingest(store, articles, "polygon")
            try:
                source_health.record_result("polygon", len(articles))
            except Exception as he:
                log.warning(f"[polygon_worker] source_health error: {he}")
            _worker_last_ok["polygon"] = time.time()
            log.debug(f"[polygon] cycle ok ({len(articles)} new)")
            bo.reset()
        except Exception as e:
            log.warning(f"[polygon_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(POLYGON_INTERVAL)


# ── Worker: Massive.com news — every 10min ─────────────────────────────────
def massive_worker(store: ArticleStore):
    log.info("[massive_worker] started")
    bo = Backoff("massive", base=15.0, cap=900.0)
    while _running:
        try:
            articles = collect_massive()
            _ingest(store, articles, "massive")
            try:
                source_health.record_result("massive", len(articles))
            except Exception as he:
                log.warning(f"[massive_worker] source_health error: {he}")
            _worker_last_ok["massive"] = time.time()
            log.debug(f"[massive] cycle ok ({len(articles)} new)")
            bo.reset()
        except Exception as e:
            log.warning(f"[massive_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(MASSIVE_INTERVAL)


# ── Worker: NewsAPI keyword search — every 25min (quota=100/day) ────────────
def newsapi_worker(store: ArticleStore):
    log.info("[newsapi_worker] started")
    bo = Backoff("newsapi", base=30.0, cap=1800.0)
    while _running:
        try:
            articles = collect_newsapi()
            _ingest(store, articles, "newsapi")
            try:
                source_health.record_result("newsapi", len(articles))
            except Exception as he:
                log.warning(f"[newsapi_worker] source_health error: {he}")
            _worker_last_ok["newsapi"] = time.time()
            log.debug(f"[newsapi] cycle ok ({len(articles)} new)")
            bo.reset()
        except Exception as e:
            log.warning(f"[newsapi_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(NEWSAPI_INTERVAL)


# ── Worker: Yahoo Finance per-ticker RSS — every 4min ───────────────────────
def yahoo_ticker_rss_worker(store: ArticleStore):
    log.info("[yahoo_ticker_rss_worker] started")
    bo = Backoff("yahoo_ticker_rss", base=10.0, cap=600.0)
    while _running:
        try:
            articles = collect_yahoo_ticker_rss()
            _ingest(store, articles, "yahoo_ticker_rss")
            try:
                source_health.record_result("yahoo_ticker_rss", len(articles))
            except Exception as he:
                log.warning(f"[yahoo_ticker_rss_worker] source_health error: {he}")
            _worker_last_ok["yahoo_ticker_rss"] = time.time()
            log.debug(f"[yahoo_ticker_rss] cycle ok ({len(articles)} new)")
            bo.reset()
        except Exception as e:
            log.warning(f"[yahoo_ticker_rss_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(YAHOO_TICKER_RSS_INTERVAL)


# ── Worker: Yahoo Finance market movers — every 5min ────────────────────────
def market_movers_worker(store: ArticleStore):
    log.info("[market_movers_worker] started")
    bo = Backoff("market_movers", base=10.0, cap=600.0)
    while _running:
        try:
            articles = collect_market_movers()
            _ingest(store, articles, "market_movers")
            try:
                source_health.record_result("market_movers", len(articles))
            except Exception as he:
                log.warning(f"[market_movers_worker] source_health error: {he}")
            _worker_last_ok["market_movers"] = time.time()
            log.debug(f"[market_movers] cycle ok ({len(articles)} new)")
            bo.reset()
        except Exception as e:
            log.warning(f"[market_movers_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(MARKET_MOVERS_INTERVAL)


# ── Worker: Wikipedia recent-changes filter — every 10min ───────────────────
def wikipedia_worker(store: ArticleStore):
    log.info("[wikipedia_worker] started")
    bo = Backoff("wikipedia", base=10.0, cap=600.0)
    while _running:
        try:
            articles = collect_wikipedia()
            _ingest(store, articles, "wikipedia")
            try:
                source_health.record_result("wikipedia", len(articles))
            except Exception as he:
                log.warning(f"[wikipedia_worker] source_health error: {he}")
            _worker_last_ok["wikipedia"] = time.time()
            log.debug(f"[wikipedia] cycle ok ({len(articles)} new)")
            bo.reset()
        except Exception as e:
            log.warning(f"[wikipedia_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(WIKIPEDIA_INTERVAL)


# ── Worker: Macro economic calendar — every 1h ──────────────────────────────
def macro_calendar_worker(store: ArticleStore):
    log.info("[macro_calendar_worker] started")
    bo = Backoff("macro_calendar", base=60.0, cap=900.0)
    while _running:
        try:
            articles = collect_macro_calendar()
            _ingest(store, articles, "macro_calendar")
            try:
                source_health.record_result("macro_calendar", len(articles))
            except Exception as he:
                log.warning(f"[macro_calendar_worker] source_health error: {he}")
            _worker_last_ok["macro_calendar"] = time.time()
            log.debug(f"[macro_calendar] cycle ok ({len(articles)} new events)")
            bo.reset()
        except Exception as e:
            log.warning(f"[macro_calendar_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(MACRO_CALENDAR_INTERVAL)


# ── Worker: FINRA RegSHO short volume — every 1h ────────────────────────────
def finra_short_worker(store: ArticleStore):
    log.info("[finra_short_worker] started")
    bo = Backoff("finra_short", base=60.0, cap=900.0)
    while _running:
        try:
            articles = collect_finra_short_volume()
            _ingest(store, articles, "finra_short_volume")
            try:
                source_health.record_result("finra_short_volume", len(articles))
            except Exception as he:
                log.warning(f"[finra_short_worker] source_health error: {he}")
            _worker_last_ok["finra_short_volume"] = time.time()
            log.debug(f"[finra_short] cycle ok ({len(articles)} new rows)")
            bo.reset()
        except Exception as e:
            log.warning(f"[finra_short_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(FINRA_SHORT_INTERVAL)


# ── Worker: Congressional trading disclosures — every 1h ────────────────────
def congress_trades_worker(store: ArticleStore):
    log.info("[congress_trades_worker] started")
    bo = Backoff("congress_trades", base=60.0, cap=900.0)
    while _running:
        try:
            articles = collect_congress_trades()
            _ingest(store, articles, "congress_trades")
            try:
                source_health.record_result("congress_trades", len(articles))
            except Exception as he:
                log.warning(f"[congress_trades_worker] source_health error: {he}")
            _worker_last_ok["congress_trades"] = time.time()
            log.debug(f"[congress_trades] cycle ok ({len(articles)} new rows)")
            bo.reset()
        except Exception as e:
            log.warning(f"[congress_trades_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(CONGRESS_TRADES_INTERVAL)


# ── Worker: Portfolio P/L snapshot — every 5min ─────────────────────────────
def portfolio_pl_worker(store: ArticleStore):
    log.info("[portfolio_pl_worker] started")
    bo = Backoff("portfolio_pl", base=5.0, cap=300.0)
    while _running:
        try:
            snap = write_pl_snapshot()
            if snap is not None:
                s = snap.get("summary", {})
                log.info(f"[portfolio_pl] grand_value=${s.get('grand_value', 0):.2f} "
                         f"pnl=${s.get('grand_pnl', 0):+.2f} "
                         f"({s.get('grand_pnl_pct', 0):+.2f}%)")
                _worker_last_ok["portfolio_pl"] = time.time()
                bo.reset()
            else:
                log.warning("[portfolio_pl] snapshot returned None")
                bo.sleep(lambda: _running)
                continue
        except Exception as e:
            log.warning(f"[portfolio_pl_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(PORTFOLIO_PL_INTERVAL)


# ── Worker: Sentiment / score trends — every 10min ──────────────────────────
def sentiment_trends_worker(store: ArticleStore):
    log.info("[sentiment_trends_worker] started")
    while _running:
        try:
            with _store_lock:
                data = write_score_trends(store)
            n = len(data.get("tickers", {}))
            log.info(f"[sentiment_trends] wrote {n} tickers ({data.get('window_hours')}h window)")
            _worker_last_ok["sentiment_trends"] = time.time()
        except Exception as e:
            log.warning(f"[sentiment_trends_worker] error: {e}")
        _sleep(SENTIMENT_TRENDS_INTERVAL)


# ── Worker: Periodic training-data export to USB drive ──────────────────────
def export_worker(store: ArticleStore):
    log.info("[export_worker] started")
    _sleep(60)  # let collectors warm up first
    while _running:
        try:
            from scripts.export_training_data import export_all
            result = export_all()
            log.info(f"[export] exported {result['db_count']} signals to USB drive "
                     f"(json={result['json_count']})")
            _worker_last_ok["export"] = time.time()
        except Exception as e:
            log.warning(f"[export_worker] error: {e}")
        _sleep(EXPORT_INTERVAL)


# ── Worker: Public Flask web server — bind 0.0.0.0:8080 ─────────────────────
def _port_is_free(host: str, port: int) -> bool:
    import socket as _sock
    s = _sock.socket(_sock.AF_INET, _sock.SOCK_STREAM)
    # Match werkzeug's BaseWSGIServer, which sets SO_REUSEADDR before binding.
    # Without this, the probe fails on sockets stuck in TIME_WAIT after a
    # systemd restart even though werkzeug itself would bind successfully —
    # causing spurious "port busy" warnings and a 5–60s startup delay.
    s.setsockopt(_sock.SOL_SOCKET, _sock.SO_REUSEADDR, 1)
    try:
        s.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def _describe_port_holder(port: int) -> str:
    """Best-effort lookup of which PID/cmd is listening on `port`. Never raises."""
    try:
        import psutil
        try:
            conns = psutil.net_connections(kind="inet")
        except (psutil.AccessDenied, PermissionError):
            return "holder=unknown (psutil access denied — daemon lacks CAP_NET_ADMIN)"
        for c in conns:
            if c.laddr and c.laddr.port == port and c.status == psutil.CONN_LISTEN:
                pid = c.pid
                if not pid:
                    return f"holder=pid?? (listen on :{port}, no pid visible)"
                try:
                    p = psutil.Process(pid)
                    cmd = " ".join(p.cmdline())[:160]
                    return f"holder=pid={pid} name={p.name()} cmd={cmd}"
                except Exception:
                    return f"holder=pid={pid}"
        return f"holder=unknown (no LISTEN on :{port} found by psutil)"
    except Exception as e:
        return f"holder_lookup_failed: {e}"


def web_server_worker(store: ArticleStore):
    # Wait up to 60s for the port to be free before trying to bind.
    # This prevents crash-loops when a duplicate daemon holds the port briefly.
    bound = False

    # Fast-path: if the holder is a *sibling* daemon.py (different PID, same
    # script), the 12 short attempts will never succeed — that's a persistent
    # duplicate-process condition that needs operator action, not retries.
    # Skip the noisy poll and go straight to the long backoff with a single
    # WARNING per cycle instead of 13. This cuts ~700 lines/day of log noise
    # under the dual-systemd-unit failure mode while preserving the original
    # 60s retry loop for genuine transient TIME_WAIT contention.
    sibling_holder = None
    if not _port_is_free(WEB_SERVER_HOST, WEB_SERVER_PORT):
        _holder_desc = _describe_port_holder(WEB_SERVER_PORT)
        if "daemon.py" in _holder_desc:
            try:
                _holder_pid = int(_holder_desc.split("pid=")[1].split()[0])
            except Exception:
                _holder_pid = None
            # Treat any daemon.py holder as a sibling (including the defensive
            # "my own pid" case — shouldn't happen post-singleton-lock, but
            # backing off is safer than crashing).
            if _holder_pid != os.getpid():
                sibling_holder = _holder_desc

    if sibling_holder is not None:
        log.warning(
            f"[web_server_worker] port {WEB_SERVER_PORT} held by sibling daemon — "
            f"skipping retry loop, backing off 5min. {sibling_holder}"
        )
    else:
        for _attempt in range(12):
            if _port_is_free(WEB_SERVER_HOST, WEB_SERVER_PORT):
                bound = True
                break
            log.warning(
                f"[web_server_worker] port {WEB_SERVER_PORT} busy — waiting 5s (attempt {_attempt+1}/12)"
            )
            _sleep(5)
            if not _running:
                return

    if not bound:
        # Don't let Werkzeug crash with "Address already in use" — back off and
        # poll for the port to free up, then let the supervisor respawn us.
        if sibling_holder is None:
            holder = _describe_port_holder(WEB_SERVER_PORT)
            log.error(
                f"[web_server_worker] port {WEB_SERVER_PORT} still busy after 60s — "
                f"backing off 5min before retry. {holder}"
            )
        # In sibling case, the WARNING above already explained the situation;
        # don't add a second ERROR line per cycle.
        # Poll every 15s during the backoff so we recover quickly if the port
        # frees up (e.g. stale daemon exits) without waiting the full 5min.
        backoff_deadline = time.time() + 300
        while _running and time.time() < backoff_deadline:
            _sleep(15)
            if not _running:
                return
            if _port_is_free(WEB_SERVER_HOST, WEB_SERVER_PORT):
                log.info(f"[web_server_worker] port {WEB_SERVER_PORT} freed — resuming")
                bound = True
                break
        if not bound:
            return  # let supervisor respawn

    log.info(f"[web_server_worker] starting Flask on {WEB_SERVER_HOST}:{WEB_SERVER_PORT}")
    try:
        from dashboard.web_server import run_server
        run_server(store, host=WEB_SERVER_HOST, port=WEB_SERVER_PORT)
        # run_server is expected to block until shutdown. A return without
        # exception means Werkzeug exited (typically due to socket close from
        # a duplicate process binding :8080). Surface it so the supervisor's
        # crash counter and audit logs reflect a real failure instead of an
        # untraceable clean_return.
        if _running:
            raise RuntimeError(
                f"run_server returned unexpectedly (port {WEB_SERVER_PORT} "
                "may be held by another process)"
            )
    except Exception as e:
        log.error(f"[web_server_worker] crashed: {e}")
        # Sleep before letting the supervisor respawn so we don't hot-loop on
        # a bind error.
        _sleep(30)


# ── Worker W6: NN-first scorer — NN handles bulk, Sonnet only for grey zone ──
def scorer_worker(store: ArticleStore):
    log.info("[scorer_worker] started (NN-first mode)")
    while _running:
        try:
            with _store_lock:
                unscored = store.get_unscored(limit=1000, min_kw=0.0)
            if not unscored:
                _worker_last_ok["scorer"] = time.time()
                _sleep(SCORE_INTERVAL)
                continue
            scores = score_articles(unscored)
            batch = []
            llm_candidates = []
            ts_updates = []
            for art, sc in zip(unscored, scores):
                aid = art.get("_id")
                if not aid:
                    continue
                # Persist time_sensitivity for every article the model
                # produced a real prediction for. rel_std==99 is the
                # sentinel score_articles returns when the model isn't
                # fitted; skip those so we don't pollute the column.
                if sc.rel_std < 99:
                    ts_updates.append((aid, sc.time_sensitivity))
                ml_score = max(sc.relevance, sc.urgency)
                # LLM zone — only Sonnet for narrow uncertain band on ML score
                if 3.8 <= ml_score <= 4.3 and not sc.needs_llm:
                    # within ML zone — also send to LLM
                    llm_candidates.append(art)
                    continue
                if sc.needs_llm:
                    llm_candidates.append(art)
                    continue
                is_urgent = 1 if sc.urgency >= 8.0 else 0
                final = max(sc.relevance, sc.urgency, 0.01)
                batch.append((aid, final, is_urgent))
            if batch:
                # Model predictions go to ml_score (NOT ai_score). Writing them
                # to ai_score with score_source='llm' would re-feed the model's
                # own outputs into the trainer's ground-truth pool — the exact
                # label-feedback loop the recent fix was meant to break.
                # See storage/article_store.py::score_pending for the canonical
                # ML-vs-LLM separation; this worker has to match it.
                store.update_ml_scores_batch(batch)
            if ts_updates:
                store.update_time_sensitivity_batch(ts_updates)
            llm_urgent = 0
            if llm_candidates:
                # score_batch sends its whole input to Sonnet as a single
                # prompt. At cold start the model is unfitted so every article
                # is needs_llm — passing the full backlog (up to 1000) in one
                # call produces a prompt Sonnet can't parse. Chunk it.
                # score_batch returns the count of *urgent* items it found.
                for j in range(0, len(llm_candidates), URGENCY_BATCH_SIZE):
                    llm_urgent += score_batch(
                        llm_candidates[j:j + URGENCY_BATCH_SIZE], store
                    )
            with _store_lock:
                remaining = store.count_unscored(min_kw=0.0)
            log.info(f"[scorer] batch={len(unscored)} scored={len(batch)} "
                     f"llm_sent={len(llm_candidates)} llm_urgent={llm_urgent} "
                     f"remaining={remaining}")
            _worker_last_ok["scorer"] = time.time()
            # Loop immediately only when the ML path actually scored a batch
            # (guaranteed forward progress) AND work remains. If this cycle
            # produced no ML scores — model unfitted, or the whole backlog
            # routed to Sonnet — sleep SCORE_INTERVAL. Otherwise a failing or
            # empty LLM path leaves `remaining` unchanged and spins this worker
            # in a tight zero-delay loop hammering the Claude CLI. This mirrors
            # the explicit no-progress guard in ArticleStore.score_pending.
            if remaining == 0 or not batch:
                _sleep(SCORE_INTERVAL)
            # else: ML made progress and work remains — loop immediately
        except MemoryError:
            _handle_memory_error("scorer")
        except Exception as e:
            log.warning(f"[scorer_worker] error: {e}")
            _sleep(SCORE_INTERVAL)


# ── Worker W11: Portfolio price alerts — every 5min ─────────────────────────
def price_alert_worker(store: ArticleStore):
    log.info("[price_alert_worker] started")
    bo = Backoff("price_alert", base=5.0, cap=300.0)
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
            log.debug(f"[price_alert] checked {len(PORTFOLIO_TICKERS)} tickers")
            bo.reset()
        except Exception as e:
            log.warning(f"[price_alert_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(PRICE_ALERT_INTERVAL)


# ── Worker W10: ML trainer — retrains ArticleNet hourly ─────────────────────
def ml_trainer_worker(store: ArticleStore):
    log.info("[ml_trainer] started")
    # Bootstrap on first run
    _sleep(300)  # let collectors gather some data first
    if not _running:
        return
    try:
        log.info("[ml_trainer] Running initial bootstrap training...")
        metrics = ml_train(store)
        log.info(f"[ml_trainer] Bootstrap done: {metrics}")
        record_metric("ml.train.loss", metrics.get("final_loss", 0),
                      {"n": metrics.get("n", 0), "phase": "bootstrap"})
    except MemoryError:
        _handle_memory_error("ml_trainer")
    except Exception as e:
        log.warning(f"[ml_trainer] Bootstrap error: {e}")

    # Back-to-back retrain failures. A silent UnboundLocalError once kept
    # ArticleNet from retraining for a whole daemon lifetime while these logs
    # stayed at WARNING (invisible to the ERROR/CRITICAL healthcheck grep).
    # Escalate to Discord once the failures persist so it can't recur silently.
    consec_fail = 0
    while _running:
        _sleep(ML_TRAIN_INTERVAL)
        try:
            log.info("[ml_trainer] Retraining on accumulated labels...")
            metrics = ml_train(store)
            val = metrics.get("val_loss")
            val_str = f"{val:.4f}" if isinstance(val, (int, float)) else "n/a"
            log.info(f"[ml_trainer] Retrain: n={metrics.get('n')} "
                     f"loss={metrics.get('final_loss', 0):.4f} "
                     f"val_loss={val_str} "
                     f"elapsed={metrics.get('elapsed_s', 0):.0f}s")
            record_metric("ml.train.loss", metrics.get("final_loss", 0),
                          {"n": metrics.get("n", 0), "phase": "retrain"})
            _worker_last_ok["ml_trainer"] = time.time()
            consec_fail = 0
        except MemoryError:
            _handle_memory_error("ml_trainer")
        except Exception as e:
            consec_fail += 1
            log.warning(f"[ml_trainer] Retrain error (#{consec_fail}): {e}")
            if _ml_retrain_should_alert(consec_fail):
                try:
                    discord_send(_ml_retrain_alert_message(consec_fail, str(e)),
                                 is_alert=True)
                except Exception as alert_err:
                    log.warning(f"[ml_trainer] failed to send stuck alert: {alert_err}")


# ── Worker W12: Continuous GPU pass — keeps RTX 3060 hot ────────────────────
def continuous_trainer_worker(store: ArticleStore):
    log.info("[continuous_trainer] started")
    _sleep(300)  # let full trainer bootstrap first

    # A single pass (data-load + 40-epoch fit) can legitimately run >900s
    # under GPU contention — longer than the supervisor's staleness deadline.
    # Without an in-flight ping the worker only reports healthy when the pass
    # *returns*, so a slow-but-working pass gets false-flagged DEAD. Hand
    # train_continuous a heartbeat it calls after data-load and per-epoch.
    def _heartbeat():
        _worker_last_ok["continuous_trainer"] = time.time()

    while _running:
        try:
            metrics = train_continuous(store, heartbeat=_heartbeat)
            if metrics.get("status") == "skipped":
                log.debug(f"[continuous_trainer] skipped: {metrics.get('reason')} n={metrics.get('n')}")
            else:
                loss = metrics.get("loss")
                loss_str = f"{loss:.4f}" if loss is not None else "n/a"
                val = metrics.get("val_loss")
                val_str = f"{val:.4f}" if isinstance(val, (int, float)) else "n/a"
                best_tag = " new_best" if metrics.get("new_best") else ""
                log.info(f"[continuous_trainer] n={metrics.get('n')} "
                         f"loss={loss_str} val_loss={val_str}{best_tag} "
                         f"gpu_mem_mb={metrics.get('gpu_mem_mb')} "
                         f"elapsed_s={metrics.get('elapsed_s')}")
            # A skipped pass (busy trainer / too few samples) is still a healthy
            # heartbeat — without this ping, quiet-data periods would flap the
            # worker into a stale state.
            _worker_last_ok["continuous_trainer"] = time.time()
        except MemoryError:
            _handle_memory_error("continuous_trainer")
        except Exception as e:
            log.warning(f"[continuous_trainer] error: {e}")
        _sleep(CONTINUOUS_TRAIN_INTERVAL)


# ── Worker: Recursive Claude labeling — every 4h ────────────────────────────
def recursive_labeler_worker(store: ArticleStore):
    log.info("[recursive_labeler] worker started")
    # Stagger first run so we don't compete with the bootstrap ML trainer.
    _sleep(15 * 60)
    while _running:
        try:
            log.info("[recursive_labeler] starting pipeline run")
            summary = run_recursive_labeling(store)
            for r in summary.get("rounds", []):
                log.info(f"[recursive_labeler] round={r['name']} "
                         f"labeled={r['labeled']} requested={r['requested']} "
                         f"failures={r['failures']} elapsed={r['elapsed_s']}s")
            log.info(f"[recursive_labeler] pipeline done total_labeled="
                     f"{summary.get('total_labeled', 0)} "
                     f"elapsed={summary.get('elapsed_s', 0)}s")
            _worker_last_ok["recursive_labeler"] = time.time()
        except MemoryError:
            _handle_memory_error("recursive_labeler")
        except Exception as e:
            log.warning(f"[recursive_labeler] error: {e}")
        _sleep(RECURSIVE_LABEL_INTERVAL)


# ── Worker W7: Alert dispatcher — checks every 20s ──────────────────────────
def alert_worker(store: ArticleStore):
    log.info("[alert_worker] started")
    _last_ping = 0.0
    while _running:
        try:
            with _store_lock:
                urgent = store.get_unalerted_urgent()
            if urgent:
                log.info(f"[alert] {len(urgent)} urgent items → dispatching")
                send_urgent_alert(urgent, store)
            elif time.time() - _last_ping >= 300:
                log.debug("[alert] idle — no urgent items")
                _last_ping = time.time()
            _worker_last_ok["alert"] = time.time()
        except Exception as e:
            log.warning(f"[alert_worker] error: {e}")
        _sleep(ALERT_CHECK)


# ── Worker W8: Heartbeat briefing every 5h ──────────────────────────────────
def _extract_briefing_labels(briefing_text: str, articles: list) -> list[dict]:
    """Parse Opus briefing to extract article-level quality labels.

    Returns one entry per source article with ``in_briefing=True`` when the
    title prefix appears in the briefing prose. The 12-char minimum on the
    prefix prevents short generic titles ("Stocks", "Markets") from
    false-matching — it mirrors trainer._fetch_briefing_samples.
    """
    labels = []
    bt_lower = briefing_text.lower()
    for art in articles:
        title = art.get("title", "")
        url = art.get("url") or art.get("link", "")
        if not url:
            continue  # synthetic entries (P&L, options snapshot) have no url
        prefix = title[:40].lower()
        in_briefing = len(prefix) >= 12 and prefix in bt_lower
        labels.append({
            "url": url,
            "title": title,
            "in_briefing": in_briefing,
        })
    return labels


# Briefing-cadence resilience across daemon restarts. The daemon restarts far
# more often than the 5h HEARTBEAT_INTERVAL under the documented OOM-restart
# churn (observed: hundreds of starts/day in the rotated logs; actual briefing
# gaps of 30-40h in the `briefings` table vs the 5h target). heartbeat_worker
# used to reset its clock to ``time.time()`` on every start, so any restart
# < 5h after launch silently pushed the next briefing out another full
# interval — the analyst's scheduled digest was being starved for 30+h at a
# time. We seed the clock from the last *persisted* briefing instead so the
# cadence survives restarts. NOTE: save_briefing() runs even when the Discord
# POST fails (the extracted labels are training signal regardless), so a DB ts
# means "briefing generated", not strictly "delivered" — a Discord outage now
# costs at most one skipped 5-min retry instead of many starved briefings, an
# intentional and strictly better trade than the restart-churn starvation.
HEARTBEAT_RESTART_WARMUP_SECS = 120

# Adaptive briefing coverage. When restart-churn / DB-contention starves the
# 5h cadence (observed live: 31.9h and 41.2h gaps in the `briefings` table),
# a recovered briefing used to pull only the last 5h of articles and gave the
# consuming analyst no signal they had been dark — the "last 5h" framing
# silently understated a 30h+ coverage hole. We widen the article lookback to
# cover the real gap (hard-capped at the 24h published-staleness ceiling
# get_top_for_briefing already enforces, so no new stale-news risk) and prepend
# a one-line warning when materially overdue. Healthy cadence is unchanged.
BRIEFING_GAP_WARN_HOURS = 7.0          # warn when the last briefing is older than this
BRIEFING_MAX_LOOKBACK_HOURS = 24       # == get_top_for_briefing's pub-staleness ceiling


def _briefing_gap_hours(prev_ts: str | None, now: datetime | None = None) -> float | None:
    """Hours since the last persisted briefing ``prev_ts``.

    Returns ``None`` when there is no prior briefing, the ts is unparseable,
    or it is in the future (clock skew / bad row) — every ``None`` path
    degrades to the original 5h behaviour. A naive ts is assumed UTC, same
    convention as ``_initial_heartbeat_last`` and the rest of the pipeline."""
    if not prev_ts:
        return None
    try:
        dt = datetime.fromisoformat(prev_ts.strip().replace("Z", "+00:00"))
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    now = now or datetime.now(timezone.utc)
    gap = (now - dt).total_seconds() / 3600.0
    return gap if gap >= 0 else None


def _briefing_lookback_hours(gap_hours: float | None) -> int:
    """Article-lookback window (hours) for the next briefing.

    On a healthy cadence (gap unknown or within the 5h target) this is the
    unchanged 5h window — the common path has zero behaviour change. When
    overdue it widens to span the real gap so a late briefing isn't
    artificially narrowed to 5h, clamped to ``BRIEFING_MAX_LOOKBACK_HOURS``
    (the same 24h ceiling get_top_for_briefing already enforces via the
    published-staleness filter — no new stale-news exposure)."""
    target_h = HEARTBEAT_INTERVAL / 3600.0
    if gap_hours is None or gap_hours <= target_h:
        return int(target_h)
    return int(max(target_h, min(round(gap_hours), BRIEFING_MAX_LOOKBACK_HOURS)))


def _coverage_gap_banner(gap_hours: float | None) -> str:
    """One-line analyst warning prepended to a materially-overdue briefing.

    Empty string on a healthy cadence (or unknown gap) so an on-time briefing
    stays clean. Discord-only — never part of the saved briefing text, so it
    can't reach the trainer's title-prefix label scan (same discipline as the
    source-health line)."""
    if gap_hours is None or gap_hours < BRIEFING_GAP_WARN_HOURS:
        return ""
    target_h = HEARTBEAT_INTERVAL // 3600
    return (
        f"⚠ COVERAGE GAP: first briefing in {gap_hours:.1f}h "
        f"(target {target_h}h) — this digest spans the backlog, "
        f"not the usual {target_h}h window"
    )


def _initial_heartbeat_last(store: ArticleStore, now: float | None = None) -> float:
    """Seed heartbeat_worker's ``last`` clock from the most recent persisted
    briefing so a daemon restart does not reset the 5h cadence.

    Returns an epoch such that:
      - no briefing row / unparseable / future ts → ``now`` (original
        behaviour: wait a full HEARTBEAT_INTERVAL before the first briefing);
      - last briefing < 5h ago → that ts (worker waits only the remainder);
      - last briefing ≥ 5h ago (overdue) → clamped so the next briefing fires
        after a short warm-up, not instantly (let collectors/scorer catch up
        post-restart) and not a full interval later.
    """
    if now is None:
        now = time.time()
    try:
        rows = store.get_briefings_for_training(limit=1)
    except Exception:
        return now
    if not rows:
        return now
    ts_raw = (rows[0].get("ts") or "").strip()
    try:
        dt = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
    except Exception:
        return now
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    last_epoch = dt.timestamp()
    # Overdue → fire after a brief warm-up, not instantly. Future ts (clock
    # skew / bad row) → clamp to now (degrade to the original behaviour).
    earliest = now - HEARTBEAT_INTERVAL + HEARTBEAT_RESTART_WARMUP_SECS
    return min(now, max(earliest, last_epoch))


def heartbeat_worker(store: ArticleStore):
    log.info("[heartbeat_worker] started")
    # Seed from the last persisted briefing so restart-churn cannot starve the
    # 5h cadence (see _initial_heartbeat_last). Falls back to "wait a full
    # interval" when there is no prior briefing or the ts is unusable.
    last = _initial_heartbeat_last(store)
    _worker_last_ok["heartbeat"] = time.time()

    while _running:
        now = time.time()
        # Keep the worker visibly alive between the 5h fires so the dashboard
        # health view doesn't tag it as stale on quiet days.
        _worker_last_ok["heartbeat"] = now
        if now - last >= HEARTBEAT_INTERVAL:
            try:
                log.info("[heartbeat] Generating Opus 4.7 briefing...")
                # Gap since the LAST persisted briefing (before we save this
                # one). Drives an adaptive lookback so a restart-starved
                # briefing covers the real backlog instead of a stale 5h
                # window, and a one-line analyst warning when overdue.
                try:
                    _prev = store.get_briefings_for_training(limit=1)
                    _prev_ts = _prev[0].get("ts") if _prev else None
                except Exception:
                    _prev_ts = None
                gap_h = _briefing_gap_hours(_prev_ts)
                lookback_h = _briefing_lookback_hours(gap_h)
                with _store_lock:
                    top = store.get_top_for_briefing(hours=lookback_h, limit=50)
                source_articles = list(top)  # keep originals (with urls) for labeling

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
                # analyze() returns the placeholder string "[analyst] No
                # response from Claude." when claude_call yields nothing.
                # Posting that to Discord (and saving it as a briefing the
                # trainer will later scan) is pure noise — treat it like the
                # exception path and retry in 5min.
                if not briefing or briefing.startswith("[analyst]"):
                    log.warning("[heartbeat] empty/placeholder briefing — skipping post; retry in 5min")
                    last = time.time() - HEARTBEAT_INTERVAL + 300
                    _sleep(30)
                    continue

                health_line = _build_health_line(store)
                # Book-coverage map of THIS digest's articles (source_articles
                # is the pre-snapshot real-article list). Discord-only, like
                # health_line / banner — never folded into the saved
                # `briefing` text, so it can't reach the trainer's
                # title-prefix label scan.
                coverage_line = _format_portfolio_coverage(source_articles)
                # Coverage-gap banner is Discord-only — NOT folded into the
                # saved `briefing` text, so it can't reach the trainer's
                # title-prefix label scan (same discipline as health_line).
                banner = _coverage_gap_banner(gap_h)
                message = (
                    (banner + "\n\n" if banner else "")
                    + briefing.rstrip() + "\n\n" + health_line
                    + ("\n" + coverage_line if coverage_line else "")
                )
                if banner:
                    log.warning(
                        f"[heartbeat] overdue briefing: gap={gap_h:.1f}h "
                        f"lookback={lookback_h}h (cadence starved)"
                    )
                ok = discord_send(message)
                log.info(f"[heartbeat] {'sent' if ok else 'FAILED'} ({len(message)} chars)")

                # Feed the briefing into the ML training pipeline as labeled data.
                # Always run — the briefing is valid training signal even when
                # the Discord webhook is down (the labels feed ArticleNet, not Discord).
                try:
                    labels = _extract_briefing_labels(briefing, source_articles)
                    boosted = store.update_scores_from_labels(labels)
                    ts = datetime.now(timezone.utc).isoformat()
                    store.save_briefing(ts, briefing, len(source_articles))
                    in_briefing = sum(1 for l in labels if l["in_briefing"])
                    log.info(f"[heartbeat] training labels: {in_briefing}/{len(labels)} "
                             f"in briefing, boosted {boosted} ai_scores")
                except Exception as le:
                    log.warning(f"[heartbeat] label extraction failed: {le}")

                if ok:
                    last = time.time()
                else:
                    # Discord delivery failed — don't skip the next 5h. Mirror the
                    # exception path so we retry in ~5min once the webhook is back.
                    last = time.time() - HEARTBEAT_INTERVAL + 300
            except Exception as e:
                log.error(f"[heartbeat] error: {e}")
                last = time.time() - HEARTBEAT_INTERVAL + 300  # retry in 5min
        _sleep(30)


def _purge_worker_startup_reap(store) -> int:
    """One-shot reap on worker startup. ``ArticleStore.reap_stale_urgent`` is
    otherwise only called inside ``purge_old`` which fires on a 6h cadence; an
    operator restart cycle SHORTER than 6h (the documented stale-manual-daemon
    case — memory ``di-stale-manual-daemon``) means the reaper never gets a
    chance to clean accumulated phantom ``urgency=1`` rows.
    Live evidence (2026-05-18→19): 26 rows stuck at ``urgency=1`` since
    2026-05-13 — 6 days — inflating the dashboard urgent tile and re-fetched/
    re-decompressed every cycle by the alert worker. Calling reap once at
    purge_worker startup makes the cleanup deterministic per daemon run.
    Idempotent + cheap (one UPDATE with the indexed ``urgency=1`` filter +
    a first_seen comparison) and identically invariant-safe to the existing
    in-purge_old call: only ``urgency`` is mutated, never ai_score /
    ml_score / score_source / synthetic rows. Best-effort — any exception is
    logged and swallowed so a transient store error never blocks the worker
    from starting the 5-min health-ping loop."""
    try:
        n = store.reap_stale_urgent()
        if n:
            log.info(
                f"[purge_worker] startup-reaped {n} phantom urgency=1 row(s) "
                f"(aged out of the alert worker's 24h window between daemon "
                f"runs — never alerted, now demoted to urgency=0)"
            )
        return n
    except Exception as e:
        log.warning(f"[purge_worker] startup reap failed: {e}")
        return 0


# ── Worker W9: Purge old data every 6h ──────────────────────────────────────
def purge_worker(store: ArticleStore):
    log.info("[purge_worker] started")
    # Startup reap: catch up phantom urgency=1 rows that aged out between
    # daemon runs. The full purge cadence is 6h; the operator restart cycle is
    # often shorter (memory ``di-stale-manual-daemon``), so without this the
    # reaper would never fire on a short-lived daemon and phantom rows
    # accumulate indefinitely. See ``_purge_worker_startup_reap``.
    _purge_worker_startup_reap(store)
    # Pin liveness independently of the 6h purge cadence so the health
    # snapshot doesn't mark this worker stale between fires. Without this
    # split, ``min(PURGE_INTERVAL, 300)`` was capping the loop at 5 min and
    # firing ``purge_old`` 72× more often than intended.
    last_purge = time.time()
    while _running:
        _worker_last_ok["purge"] = time.time()
        _sleep(300)
        if time.time() - last_purge < PURGE_INTERVAL:
            continue
        try:
            with _store_lock:
                store.purge_old()
            # Sweep away legacy per-query gdelt:* health keys (now recorded
            # as a single aggregate "gdelt" key). Idempotent: 0 rows after
            # the first pass.
            try:
                removed = source_health.delete_sources("gdelt:")
                if removed:
                    log.info(f"[purge] dropped {removed} legacy gdelt:* health keys")
            except Exception as he:
                log.warning(f"[purge_worker] source_health cleanup error: {he}")
            stats = store.stats()
            log.info(f"[purge] DB: {stats['total']} articles, {stats['db_mb']}MB")
            last_purge = time.time()
            _worker_last_ok["purge"] = time.time()
        except Exception as e:
            log.warning(f"[purge_worker] error: {e}")


# ── Source-health line formatter (pure, unit-tested) ─────────────────────────
SOURCE_HEALTH_FULL_DUMP_SECS = 3600


def _format_source_health_line(
    disabled, stale, down, last_down_sig, last_full_dump, now
):
    """Build the [source_health] log line + decide level and new state.

    Pure function so the steady-state log-bloat behaviour can be unit-tested
    without threads or a live logger.

    Returns ``(line, level, new_last_down_sig, new_last_full_dump)`` where
    ``level`` is one of ``"warning"`` / ``"info"``.

    Behaviour:
      * Down-set changed -> WARNING with counts + delta (newly_down /
        recovered) only. The full ``list=`` is appended ONLY when this is the
        first observation (``last_down_sig is None``) or the hourly safety-net
        has elapsed.
      * Down-set unchanged -> concise INFO ``(unchanged)``, UNLESS the hourly
        safety net has elapsed, in which case a WARNING with the full list is
        emitted so the audit can always reconstruct state.
    """
    down = list(down)
    down_sig = tuple(down)
    n = f"disabled={len(disabled)} stale={len(stale)} down={len(down)}"
    new_full_dump = last_full_dump
    full_due = (now - last_full_dump) >= SOURCE_HEALTH_FULL_DUMP_SECS

    if down_sig != last_down_sig:
        prev = set(last_down_sig or ())
        cur = set(down)
        newly = sorted(cur - prev)
        recovered = sorted(prev - cur)
        parts = [f"[source_health] {n}"]
        if newly:
            parts.append(f"newly_down={newly}")
        if recovered:
            parts.append(f"recovered={recovered}")
        first_seen = last_down_sig is None
        if first_seen or full_due:
            parts.append(f"list={down}")
            new_full_dump = now
        line = " ".join(parts)
        return line, "warning", down_sig, new_full_dump

    # Unchanged down-set.
    if full_due:
        new_full_dump = now
        return (
            f"[source_health] {n} (unchanged) list={down}",
            "warning",
            down_sig,
            new_full_dump,
        )
    return (
        f"[source_health] {n} (unchanged)",
        "info",
        down_sig,
        new_full_dump,
    )


# ── Stats reporter — every 60s, but only emit when changed or 5min elapsed ───
def stats_worker(store: ArticleStore):
    last_sig = None
    last_emit = 0.0
    last_down_sig = None
    last_full_dump = 0.0
    HEARTBEAT_SECS = 300
    while _running:
        _sleep(60)
        gpu_info = ""
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=2,
            )
            gpu_info = f" gpu={result.stdout.strip()}" if result.returncode == 0 else ""
        except Exception:
            pass
        try:
            s = store.stats()
            sig = (s["total"], s["urgent"], s["unscored"], s.get("below_threshold", 0))
            now = time.time()
            if sig != last_sig or (now - last_emit) >= HEARTBEAT_SECS:
                log.info(f"[stats] total={s['total']} urgent={s['urgent']} "
                         f"unscored={s['unscored']} low_kw={s.get('below_threshold', 0)} "
                         f"db={s['db_mb']}MB{gpu_info}")
                try:
                    disabled = source_health.get_disabled_sources()
                    stale = source_health.get_stale_sources()
                    if disabled or stale:
                        down = sorted(set(disabled) | set(stale))
                        line, level, last_down_sig, last_full_dump = (
                            _format_source_health_line(
                                disabled, stale, down,
                                last_down_sig, last_full_dump, now,
                            )
                        )
                        getattr(log, level)(line)
                    elif last_down_sig is not None:
                        log.warning("[source_health] all sources recovered")
                        last_down_sig = None
                except Exception as he:
                    log.debug(f"[stats_worker] source_health probe failed: {he}")
                last_sig = sig
                last_emit = now
            _worker_last_ok["stats"] = now
        except Exception as e:
            # Don't blanket-swallow — a stats query failure is usually a DB
            # lock or schema issue worth seeing, even if non-fatal.
            log.debug(f"[stats_worker] error: {e}")


def _sleep(seconds: float):
    """Interruptible sleep — checks _running every 0.5s.

    For sleeps longer than 60s, emits a ``[<worker_name>] alive`` debug log
    every 60s so health dashboards stay green even when the worker is
    intentionally idle (heartbeat 5h, purge 6h, ml_trainer 3min, etc.). The
    worker tag is taken from the current thread's name."""
    deadline = time.time() + seconds
    name = threading.current_thread().name
    emit_pings = seconds > 60 and name and name != "MainThread"
    next_ping = time.time() + 60 if emit_pings else float("inf")
    while _running and time.time() < deadline:
        time.sleep(0.5)
        if emit_pings and time.time() >= next_ping:
            log.debug(f"[{name}] alive")
            next_ping = time.time() + 60


def _format_source_health_summary(
    disabled, stale, max_names: int = 4, max_chars: int = 110
) -> str:
    """Compact 'collectors down' line for the heartbeat briefing.

    A disabled SEC-EDGAR / wire collector means the 5h digest is silently
    missing that source's news — the briefing prose alone never reveals the
    blind spot, and the existing health line only reports four worker threads'
    liveness, not source_health's disabled/stale set. Surfacing it lets the
    analyst know which part of their coverage universe went dark.

    Pure + deterministic for unit testing: ``disabled`` (hard-down collectors)
    are de-duplicated against ``stale`` (collectors with no fresh data in the
    staleness window), each set is sorted, the union is listed disabled-first
    (the harder failure), then truncated to ``max_names`` with a ``+N``
    overflow marker and a hard ``max_chars`` cap. Returns "" when nothing is
    down so a healthy briefing stays clean.
    """
    d = sorted(set(disabled or ()))
    s = sorted(set(stale or ()) - set(d))
    total = len(d) + len(s)
    if total == 0:
        return ""
    names = d + s  # disabled first — the harder failure
    shown = names[:max_names]
    label = ", ".join(shown)
    overflow = total - len(shown)
    if overflow > 0:
        label += f", +{overflow}"
    line = f"⚠ Sources down ({total}): {label}"
    if len(line) > max_chars:
        line = line[: max_chars - 1].rstrip(", ") + "…"
    return line


_COVERAGE_TICKER_CACHE: dict[tuple[str, ...], "re.Pattern"] = {}


def _format_portfolio_coverage(
    articles, tickers=PORTFOLIO_TICKERS, max_chars: int = 150
) -> str:
    """One-line book-coverage map of the 5h digest for the analyst.

    The Opus briefing prose never states which of the operator's *positions*
    the digest actually touches. A 5h window with zero mentions of a held
    name (AXTI/QBTS/SNDU are thin-coverage) is a silent blind spot — the
    analyst cannot tell "nothing happened" from "the pipeline missed it" and
    has money at risk either way. This converts that into one explicit signal.

    Pure + deterministic for unit testing (mirrors
    ``_format_source_health_summary``). Matching reuses the established
    case-sensitive word-boundary convention from ``ml.features._LIVE_RE``
    (financial copy writes tickers uppercase; ``\\bMU\\b`` won't match inside
    "MUSEUM" and "MUU" stays distinct from "MU"). ``covered`` preserves the
    ``tickers`` order so the line is stable cycle-to-cycle; the ``silent``
    tail is truncated with a ``+N`` overflow marker under a hard
    ``max_chars`` cap. Returns "" only when there are no articles to assess
    (degrade clean, same as the other briefing-augmentation helpers).

    Discord-only — the caller appends it to the posted message, NEVER folds
    it into the saved ``briefing`` text, so it cannot reach the trainer's
    title-prefix label scan (same discipline as the source-health line and
    the coverage-gap banner). Read-only: touches no articles row, no
    ai_score/ml_score/score_source — all four load-bearing invariants intact.
    """
    tickers = tuple(tickers)
    if not articles or not tickers:
        return ""
    pat = _COVERAGE_TICKER_CACHE.get(tickers)
    if pat is None:
        # Longest-first so the alternation prefers \bMUU\b over \bMU\b.
        alts = sorted({t for t in tickers if t}, key=len, reverse=True)
        pat = re.compile(r"\b(?:" + "|".join(re.escape(t) for t in alts) + r")\b")
        _COVERAGE_TICKER_CACHE[tickers] = pat
    seen: set[str] = set()
    for a in articles:
        blob = f"{a.get('title') or ''} {a.get('summary') or ''}"
        if not blob.strip():
            continue
        for m in pat.findall(blob):
            seen.add(m)
    n = len(tickers)
    covered = [t for t in tickers if t in seen]   # stable: tickers order
    silent = [t for t in tickers if t not in seen]
    head = "·".join(covered) if covered else "none"
    line = f"📊 Book in digest: {head} ({len(covered)}/{n})"
    if silent:
        line += " — silent: " + " ".join(silent)
    if len(line) > max_chars:
        # Trim the silent tail token-wise, leaving a +N overflow marker.
        if " — silent: " in line:
            prefix, tail = line.split(" — silent: ", 1)
            toks = tail.split(" ")
            kept: list[str] = []
            for tok in toks:
                trial = prefix + " — silent: " + " ".join(kept + [tok])
                if len(trial) + 4 > max_chars:
                    break
                kept.append(tok)
            overflow = len(silent) - len(kept)
            line = prefix + " — silent: " + " ".join(kept)
            if overflow > 0:
                line += f" +{overflow}"
        else:
            line = line[: max_chars - 1].rstrip() + "…"
    return line


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
    base = "⚙ Workers: " + " ".join(parts) + suffix
    # Append the source-health blind-spot line so a disabled collector is
    # visible in the Discord briefing, not just the daemon log.
    try:
        sh = _format_source_health_summary(
            source_health.get_disabled_sources(),
            source_health.get_stale_sources(),
        )
    except Exception:
        sh = ""
    return base + ("\n" + sh if sh else "")


def _acquire_singleton_lock():
    """Prevent duplicate daemon instances.

    Why: an orphaned daemon.py from a prior session can keep port 8080 bound,
    causing the systemd-managed instance to log repeated port-busy warnings
    and never serve the dashboard. fcntl.flock on a pidfile gives us a
    kernel-enforced singleton that releases automatically on process exit.

    The file is opened without O_TRUNC so a racing second daemon can still
    read the holder's pid for the diagnostic log line. The previous "w" open
    truncated the file before the flock check, wiping the holder pid the
    moment a second instance tried to start.
    """
    import fcntl
    lock_path = BASE_DIR / "data" / "daemon.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    # O_CLOEXEC is critical: ml/trainer.py runs training in a multiprocessing
    # "spawn" child (fork+exec). Without close-on-exec, that child inherits and
    # holds this flock. When the daemon main process hard-exits via os._exit(0)
    # on SIGTERM, the kernel does NOT release the flock until the orphaned
    # spawn child also exits — so the next instance sees the lock "held by
    # pid=<dead main>", SIGTERMs a corpse, then stalls 20-30s polling the
    # flock. That stall is a primary driver of the restart flap.
    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            holder = os.read(fd, 64).decode(errors="replace").strip() or "unknown"
        except Exception:
            holder = "unknown"
        log.warning(
            f"[daemon] Another daemon instance is already running (lock held by pid={holder}). "
            f"Sending SIGTERM and waiting for it to exit."
        )
        # Actively terminate the holder so we don't block indefinitely.
        # (systemd restart only kills its own MainPID; orphans from prior
        # crash cycles are outside the cgroup and survive restart.)
        try:
            holder_pid = int(holder)
            os.kill(holder_pid, signal.SIGTERM)
        except (ValueError, ProcessLookupError, PermissionError):
            pass  # already dead or not ours

        # Poll the flock for up to 30s for the graceful release; if the
        # holder is still alive after that, force-kill it then acquire.
        acquired = False
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except BlockingIOError:
                time.sleep(0.5)
        if not acquired:
            try:
                os.kill(int(holder), signal.SIGKILL)
                log.warning(f"[daemon] Holder pid={holder} did not exit in 30s; sent SIGKILL.")
            except (ValueError, ProcessLookupError, PermissionError):
                pass
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
            except Exception as e:
                os.close(fd)
                log.error(f"[daemon] Blocking flock failed after SIGKILL: {e}; exiting.")
                sys.exit(1)
        log.info(f"[daemon] Previous holder pid={holder} exited; this instance is now primary.")
    # Seek to 0 before truncate+write: the diagnostic read above advances the
    # fd offset, and os.ftruncate does not reset it. Without this seek, the
    # subsequent write lands past the truncated end, leaving leading NUL bytes
    # that corrupt the pid string future instances read for diagnostics.
    os.lseek(fd, 0, os.SEEK_SET)
    os.ftruncate(fd, 0)
    os.write(fd, str(os.getpid()).encode())
    # Keep fd open for the process lifetime; the kernel releases the flock
    # on the final close (i.e., process exit).
    globals()["_singleton_lock_fd"] = fd


def main():
    _acquire_singleton_lock()

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
        ("sec_edgar",   sec_edgar_worker),
        ("sec_edgar_ft", sec_edgar_ft_worker),
        ("google_news", google_news_worker),
        ("nitter",      nitter_worker),
        ("substack",    substack_worker),
        ("finnhub",     finnhub_worker),
        ("alphavantage", alphavantage_worker),
        ("polygon",     polygon_worker),
        ("massive",     massive_worker),
        ("newsapi",     newsapi_worker),
        ("yahoo_ticker_rss", yahoo_ticker_rss_worker),
        ("market_movers", market_movers_worker),
        ("wikipedia",   wikipedia_worker),
        ("macro_calendar", macro_calendar_worker),
        ("finra_short",   finra_short_worker),
        ("congress_trades", congress_trades_worker),
        ("scorer",      scorer_worker),
        ("alert",       alert_worker),
        ("heartbeat",   heartbeat_worker),
        ("purge",       purge_worker),
        ("stats",       stats_worker),
        ("ml_trainer",  ml_trainer_worker),
        ("continuous_trainer", continuous_trainer_worker),
        ("recursive_labeler", recursive_labeler_worker),
        ("price_alert", price_alert_worker),
        ("portfolio_pl",    portfolio_pl_worker),
        ("sentiment_trends", sentiment_trends_worker),
        ("export",      export_worker),
        ("web_server",  web_server_worker),
    ]

    # Map name → fn for lookup during respawn
    worker_map = {name: fn for name, fn in workers}

    threads: list[threading.Thread] = []
    for name, fn in workers:
        wrapped = _wrap_worker(name, fn)
        t = threading.Thread(target=wrapped, args=(store,), name=name, daemon=True)
        t.start()
        threads.append(t)
        _worker_state[name] = "ok"
        log.info(f"[daemon] Worker '{name}' started")

    log.info(f"[daemon] All {len(threads)} workers running — max throughput mode")

    # ── Supervisor loop ──────────────────────────────────────────────────────
    # Smart respawn: per-worker crash counter, degraded/disabled states, OOM
    # backoff, Discord alerts on state transitions only, periodic health
    # report writes to structured.jsonl + supervisor_state.json snapshot.
    last_health_report = 0.0
    last_recovery_sweep = 0.0
    RECOVERY_SWEEP_SECS = 60  # how often to age out stale degraded states
    while _running:
        time.sleep(5)
        if not _running:
            # Graceful shutdown in progress — worker threads are exiting cleanly
            # as their own loops observe _running=False. Logging them as "died"
            # at ERROR pollutes the error count for hourly audits.
            # Safe to log here: normal thread context, not a signal handler.
            log.info(f"Signal {_shutdown_signal} — shutting down")
            break
        now = time.time()

        # Recovery sweep: a worker that crashes its way into "degraded" then
        # stops crashing has no path back to "ok" — _record_crash only fires
        # on new crashes. Without this sweep, the dashboard shows degraded
        # forever. Once the rolling crash window has elapsed with zero new
        # crashes AND the thread is currently alive, transition back to ok.
        if now - last_recovery_sweep >= RECOVERY_SWEEP_SECS:
            for t in threads:
                if not t.is_alive():
                    continue
                name = t.name
                with _supervisor_lock:
                    state = _worker_state.get(name, "ok")
                    crashes = _worker_crashes.get(name, [])
                    recent = [x for x in crashes if now - x < CRASH_WINDOW_SECS]
                if state == "degraded" and not recent:
                    with _supervisor_lock:
                        _worker_state[name] = "ok"
                        _worker_crashes[name] = []
                    _notify_state_transition(name, "degraded", "ok", 0)
            last_recovery_sweep = now
        for i, t in enumerate(threads):
            name = t.name
            if t.is_alive():
                continue
            # Already detected dead and waiting for respawn window?
            respawn_at = _worker_respawn_at.get(name)
            if respawn_at is not None:
                if now < respawn_at:
                    continue
                _worker_respawn_at.pop(name, None)
            else:
                # First detection of this crash: record + schedule respawn
                count, old_state, new_state = _record_crash(name)
                if old_state != new_state:
                    _notify_state_transition(name, old_state, new_state, count)
                delay = _compute_respawn_delay(name, now)
                _worker_respawn_at[name] = now + delay
                last_exc = _worker_last_exception.get(name, "")
                # A thread that exited without recording an exception returned
                # cleanly — that's abnormal for a long-running worker, but it
                # isn't a crash with a stack trace. Log at WARNING so the
                # hourly audit's ERROR/CRITICAL count reflects real failures.
                msg = (
                    f"[daemon] Worker '{name}' exited "
                    f"(crashes_5m={count}, state={new_state}, "
                    f"last_exc={last_exc or 'clean_return'}) — "
                    f"respawning in {delay:.0f}s"
                )
                extra = {
                    "event": "worker_died",
                    "worker": name,
                    "crashes_5m": count,
                    "state": new_state,
                    "respawn_in_s": delay,
                    "last_exc": last_exc or "clean_return",
                }
                if last_exc:
                    log.error(msg, extra=extra)
                else:
                    log.warning(msg, extra=extra)
                continue

            # Respawn now
            fn = worker_map[name]
            wrapped = _wrap_worker(name, fn)
            new_t = threading.Thread(target=wrapped, args=(store,), name=name, daemon=True)
            new_t.start()
            threads[i] = new_t
            _worker_last_exception.pop(name, None)
            # If we passed through a disabled window cleanly, reset state to ok
            disabled_until = _worker_disabled_until.get(name, 0.0)
            if disabled_until and now >= disabled_until:
                old_state = _worker_state.get(name, "disabled")
                _worker_state[name] = "ok"
                _worker_disabled_until.pop(name, None)
                with _supervisor_lock:
                    _worker_crashes[name] = []
                if old_state != "ok":
                    _notify_state_transition(name, old_state, "ok", 0)
            log.info(
                f"[daemon] Worker '{name}' respawned",
                extra={"event": "worker_respawned", "worker": name},
            )

        # Periodic health report (every 5 minutes)
        if now - last_health_report >= HEALTH_REPORT_INTERVAL_SECS:
            try:
                _worker_health_report()
            except Exception as e:
                log.warning(f"[supervisor] health report error: {e}")
            last_health_report = now

    log.info("[daemon] Shutdown complete")

    # Hard-exit instead of falling through into normal interpreter teardown.
    # Every worker runs as a daemon thread, and the ML trainers spend most of
    # their time inside torch/CUDA C-extension code. If a training pass is in
    # flight when SIGTERM lands, Py_FinalizeEx blocks trying to reclaim the GIL
    # that the C loop still holds — the process then sits idle until systemd's
    # TimeoutStopSec elapses and escalates to SIGKILL (observed: code=killed
    # status=9, "Failed with result 'timeout'"). Flushing the log handlers and
    # calling os._exit guarantees a sub-second, clean-looking stop every time.
    import logging
    logging.shutdown()
    os._exit(0)


if __name__ == "__main__":
    main()
