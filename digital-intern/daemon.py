"""
Digital Intern — Maximum-throughput continuous intelligence daemon.

Architecture: independent worker threads, each running their own infinite loop.
No global sleep. Workers run as fast as their sources allow.

Workers (intervals below track the *_INTERVAL constants in the Config block):
  W1  gdelt_worker       — full GDELT sweep via collect_gdelt() every 10min
  W2  rss_worker         — re-polls all RSS feeds every 30s
  W3  web_worker         — scrapes 100+ financial sites every 60s
  W4  reddit_worker      — re-polls Reddit every 45s
  W4b stocktwits_worker  — StockTwits trending stream every 90s
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
from collectors.stocktwits_collector import collect_stocktwits
from collectors.stocktwits_sentiment import collect_stocktwits_sentiment
from collectors.web_scraper import scrape_web
from collectors.stock_data import get_stock_data, _fetch_one
from collectors.earnings_calendar import get_earnings
from collectors.options_monitor import get_options_data, format_options_block
from collectors.cboe_unusual_options import collect_cboe_unusual_options
from collectors.portfolio_pnl import get_portfolio_pnl, format_pnl_block, write_pl_snapshot
from collectors.sec_edgar import collect_sec_edgar, collect_sec_edgar_fulltext
from collectors.sec_activist_collector import collect as collect_sec_activist
from collectors.google_news import collect_google_news
from collectors.nitter_collector import collect_nitter
from collectors.substack_collector import collect_substack
from collectors.finnhub_collector import collect_finnhub
from collectors.alphavantage_collector import collect_alphavantage
from collectors.polygon_collector import collect_polygon
from collectors.massive_collector import collect_massive
from collectors.newsapi_collector import collect_newsapi
from collectors.yahoo_ticker_rss import collect_yahoo_ticker_rss
from collectors.short_interest_collector import collect_short_interest
from collectors.wikipedia_collector import collect_wikipedia
from collectors.wikipedia_pageviews import collect_wikipedia_pageviews
from collectors.macro_calendar_collector import collect_macro_calendar
from collectors.tic_foreign_holdings import collect_tic
from collectors.congress_trades_collector import collect_congress_trades
from collectors.finra_short_volume import collect_finra_short_volume
from collectors.market_movers import collect_market_movers
from collectors.yahoo_trending_tickers import collect_yahoo_trending
from collectors.fear_greed_collector import collect_fear_greed
from collectors.crypto_fear_greed_collector import collect_crypto_fear_greed
from collectors.binance_funding_collector import collect_crypto_funding
from collectors.earnings_surprise_collector import collect_earnings_surprises
from collectors.polymarket_collector import collect as collect_polymarket
from collectors.manifold_collector import collect_manifold
from collectors.collector_rate_monitor import collect_rate_alerts
from collectors.yield_curve_collector import collect_yield_curve
from collectors.g10_sovereign_yields import collect_g10_yields
from collectors.fred_collector import collect_fred
from collectors.cftc_cot_collector import collect_cftc_cot
from collectors.vix_term_structure import collect as collect_vix_ts
from collectors.dxy_collector import collect as collect_dxy
from collectors.commodity_futures_collector import collect as collect_commodity_futures
from collectors.openinsider_cluster import collect as collect_insider_cluster
from collectors.sector_etf_momentum import collect as collect_sector_etf
from collectors.cisa_kev_collector import collect_cisa_kev
from collectors.benzinga_analyst_collector import collect_benzinga_analyst
from collectors.ftc_doj_collector import collect_ftc_doj
from collectors.eia_collector import collect_eia
from collectors.fed_press_collector import collect_fed_press
from collectors.ecb_press_collector import collect_ecb_press
from collectors.boj_press_collector import collect_boj_press
from collectors.boe_press_collector import collect_boe_press
from collectors.whitehouse_collector import collect_whitehouse
from collectors.g10_central_banks_collector import collect_g10_central_banks
from collectors.global_regulators_collector import collect_global_regulators
from collectors.imf_bis_worldbank_collector import collect_imf_bis_worldbank
from collectors.un_news_collector import collect_un_news
from collectors.globenewswire_collector import collect_globenewswire
from collectors.short_seller_collector import collect_short_sellers
from collectors.financial_blogs_collector import collect_financial_blogs
from collectors.hackernews_collector import collect_hackernews
from collectors.market_breadth_collector import collect_market_breadth
from collectors.sec_xbrl_financials import collect_sec_xbrl_financials
from collectors.usgs_earthquake_collector import collect_usgs_earthquakes
from collectors.forex_factory_calendar import collect as collect_forex_factory_cal
from collectors.sec_13f_collector import collect_13f_filings
from collectors.sec_insider_form4 import collect_sec_form4
from collectors.nasdaq_halts_collector import collect_nasdaq_halts
from collectors.fda_collector import collect_fda
from collectors.usaspending_contracts_collector import collect_usaspending_contracts
from collectors.unusual_volume_collector import collect_unusual_volume
from collectors.short_squeeze_monitor import collect_short_squeeze
from collectors.twse_semiconductor import collect_twse_semiconductor
from collectors.nasdaq_ipo_calendar import collect_nasdaq_ipo
from collectors.nasdaq_earnings_calendar import collect as collect_nasdaq_earnings
from collectors.putcall_ratio_collector import collect_putcall_ratio
from collectors.bls_collector import collect_bls
from collectors.bea_collector import collect_bea
from collectors.federal_register_collector import collect_federal_register
from collectors import source_health
from core.backoff import Backoff
from triage.heuristic_scorer import score_article as _heuristic_score_article
from analysis.claude_analyst import analyze
from notifier.discord_notifier import send as discord_send
from storage.article_store import ArticleStore
from watchers.urgency_scorer import score_batch, BATCH_SIZE as URGENCY_BATCH_SIZE
from watchers.alert_agent import send_urgent_alert
from ml.inference import score_articles
from ml.features import LIVE_PORTFOLIO_TICKERS
from ml.sentiment_trends import write_trends as write_score_trends
from ml.trainer import train as ml_train
from ml.trainer import train_continuous
from ml.recursive_labeler import run_recursive_labeling
from core.retrain_guard import should_alert as _ml_retrain_should_alert
from core.retrain_guard import alert_message as _ml_retrain_alert_message
from core.retrain_guard import is_retrain_failure as _ml_retrain_is_failure

# ── Config ──────────────────────────────────────────────────────────────────
HEARTBEAT_INTERVAL  = 5 * 3600   # 5h
RSS_INTERVAL        = 30          # re-poll RSS every 30s (collector is parallelized)
WEB_INTERVAL        = 60          # scrape web every 60s
REDDIT_INTERVAL     = 45          # re-poll Reddit every 45s
STOCKTWITS_INTERVAL = 90          # re-poll StockTwits trending every 90s
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
SEC_ACTIVIST_INTERVAL = 600       # SEC activist/M&A special filings every 10min
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
WIKI_PAGEVIEWS_INTERVAL = 3600    # Wikipedia pageview z-score surge alerts once per hour
MACRO_CALENDAR_INTERVAL = 3600    # FOMC/BLS macro event calendar — once per hour
TIC_INTERVAL            = 21600   # TIC foreign Treasury holdings — 6h (monthly release)
FOREX_FACTORY_CAL_INTERVAL = 3600  # Forex Factory economic calendar — once per hour
FINRA_SHORT_INTERVAL    = 3600    # FINRA RegSHO short volume — once per hour (daily file)
CONGRESS_TRADES_INTERVAL = 3600   # Congressional trading disclosures — once per hour
CBOE_UNUSUAL_OPTIONS_INTERVAL = 900  # CBOE unusual options flow — every 15min
CISA_KEV_INTERVAL       = 3600    # CISA Known Exploited Vulnerabilities catalog — once per hour
MARKET_MOVERS_INTERVAL  = 300     # Yahoo Finance gainers/losers/most-active every 5min
YAHOO_TRENDING_INTERVAL = 300     # Yahoo Finance trending tickers (retail attention) every 5min
FEAR_GREED_INTERVAL     = 600     # CNN Fear & Greed Index every 10min
CRYPTO_FEAR_GREED_INTERVAL = 1800  # Crypto Fear & Greed (alternative.me) every 30min
CRYPTO_FUNDING_INTERVAL   = 1800  # OKX perpetual funding rates every 30min
EARNINGS_SURPRISE_INTERVAL = 900  # EPS beat/miss scanner every 15min
POLYMARKET_INTERVAL     = 900     # Polymarket prediction markets every 15min
MANIFOLD_INTERVAL       = 1800    # Manifold Markets prediction markets every 30min
RATE_MONITOR_INTERVAL   = 3600    # per-collector silence detector — hourly
YIELD_CURVE_INTERVAL    = 3600    # 10Y-2Y spread monitor every 1h (FRED daily)
G10_YIELDS_INTERVAL     = 3600    # G10 sovereign yields every 1h (FRED daily/monthly)
FRED_MACRO_INTERVAL     = 3600    # FRED macro series (claims, M2, rig count, etc.) — hourly
COT_INTERVAL            = 6 * 3600  # CFTC COT report — weekly release, check 6-hourly
TWSE_INTERVAL           = 3600    # Taiwan Stock Exchange semis — hourly (market open 09:00–13:30 TW)
SHORT_INTEREST_INTERVAL = 21600   # highshortinterest.com every 6h (data updates ~2/month)
NASDAQ_IPO_INTERVAL     = 3600    # Nasdaq IPO calendar - hourly
NASDAQ_EARNINGS_INTERVAL = 3600  # Nasdaq earnings calendar - hourly
PUTCALL_RATIO_INTERVAL  = 600     # Market put/call ratio - every 10min
VIX_TS_INTERVAL         = 600     # VIX term structure snapshot every 10min
DXY_INTERVAL            = 600     # DXY + major-pair FX snapshot every 10min
INSIDER_CLUSTER_INTERVAL = 600    # EDGAR Form 4 cluster-buy scan every 10min
SECTOR_ETF_INTERVAL     = 600     # Sector ETF momentum snapshot every 10min
COMMODITY_FUTURES_INTERVAL = 600  # Commodity futures price monitor every 10min
BENZINGA_INTERVAL       = 300     # Benzinga analyst-ratings RSS sweep every 5min
FTC_DOJ_INTERVAL        = 1800    # FTC + DOJ ATR press releases — every 30min
EIA_INTERVAL            = 1800    # EIA Today in Energy + press releases — every 30min
FED_PRESS_INTERVAL      = 1800    # Federal Reserve press / speech / testimony RSS — every 30min
ECB_PRESS_INTERVAL      = 1800    # ECB press releases RSS — every 30min
BOJ_PRESS_INTERVAL      = 1800    # Bank of Japan press / speech / MPM RSS — every 30min
BOE_PRESS_INTERVAL      = 1800    # Bank of England press / publications RSS — every 30min
WHITEHOUSE_INTERVAL     = 1800    # White House executive orders / proclamations / briefings — every 30min
G10_CB_INTERVAL         = 1800    # Bank of Canada + RBA press / speeches RSS — every 30min
GLOBAL_REG_INTERVAL     = 1800    # FSB, FCA, Fed research notes/papers — every 30min
UN_NEWS_INTERVAL        = 1800    # UN News economic/climate/regional feeds — every 30min
BIS_INTERVAL            = 1800    # BIS press releases, speeches, research — every 30min
FED_REG_INTERVAL        = 1800    # Federal Register BIS/OFAC/FTC/FCC/NIST rules — every 30min
BLS_INTERVAL            = 3600    # BLS macro series (CPI, unemployment, payrolls) — once per hour
BEA_INTERVAL            = 3600    # BEA macro releases (GDP, trade, personal income) — once per hour
GLOBENEWSWIRE_INTERVAL  = 600     # GlobeNewswire financial press releases (8 subject feeds) — every 10min
SHORT_SELLER_INTERVAL   = 1800    # Short-seller research reports (rare, high-priority) — every 30min
FINANCIAL_BLOGS_INTERVAL = 600    # InvestorPlace, Motley Fool, Nasdaq RSS — every 10min
HACKERNEWS_INTERVAL     = 300     # Hacker News front-page + finance/business stories — every 5min
MARKET_BREADTH_INTERVAL = 3600    # Finviz market breadth (% above MAs, new highs/lows) — hourly
USASPENDING_INTERVAL    = 3600    # USASpending.gov federal contract awards — hourly (new awards rare)
SEC_XBRL_INTERVAL       = 6 * 3600  # SEC XBRL quarterly financials — every 6h (filings rare)
SEC_13F_INTERVAL        = 1800      # SEC 13F institutional holdings — every 30min (quarterly season)
SEC_FORM4_INTERVAL      = 300       # SEC Form 4 insider transactions (portfolio tickers) — every 5min
USGS_QUAKE_INTERVAL     = 1800    # USGS M≥5 earthquake feed every 30min (insurance/semis/energy catalyst)
NASDAQ_HALTS_INTERVAL   = 120     # NASDAQ/UTP trading halt+resume feed every 2min
FDA_INTERVAL            = 1800    # FDA press releases + MedWatch safety alerts — every 30min
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
    "gdelt", "rss", "web", "reddit", "ticker", "sec_edgar", "sec_edgar_ft", "sec_xbrl",
    "google_news", "nitter", "substack",
    "finnhub", "alphavantage", "polygon", "massive", "newsapi",
    "yahoo_ticker_rss", "market_movers", "yahoo_trending", "wikipedia", "wiki_pageviews", "macro_calendar", "tic", "short_interest",
    "fed_press", "ecb_press", "boj_press", "boe_press", "eia", "bls", "bea", "g10_cb", "global_reg", "whitehouse",
    "usgs_quake", "fda",
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
    "reddit": REDDIT_INTERVAL, "stocktwits": STOCKTWITS_INTERVAL, "ticker": TICKER_INTERVAL,
    "sec_edgar": SEC_EDGAR_INTERVAL, "sec_edgar_ft": SEC_EDGAR_FT_INTERVAL,
    "sec_activist": SEC_ACTIVIST_INTERVAL,
    "google_news": GOOGLE_NEWS_INTERVAL, "nitter": NITTER_INTERVAL,
    "substack": SUBSTACK_INTERVAL, "finnhub": FINNHUB_INTERVAL,
    "alphavantage": ALPHAVANTAGE_INTERVAL, "polygon": POLYGON_INTERVAL,
    "massive": MASSIVE_INTERVAL, "newsapi": NEWSAPI_INTERVAL,
    "yahoo_ticker_rss": YAHOO_TICKER_RSS_INTERVAL,
    "market_movers": MARKET_MOVERS_INTERVAL,
    "yahoo_trending": YAHOO_TRENDING_INTERVAL,
    "unusual_volume": MARKET_MOVERS_INTERVAL,
    "short_squeeze":  MARKET_MOVERS_INTERVAL,
    "fear_greed": FEAR_GREED_INTERVAL,
    "crypto_fear_greed": CRYPTO_FEAR_GREED_INTERVAL,
    "crypto_funding":   CRYPTO_FUNDING_INTERVAL,
    "earnings_surprise": EARNINGS_SURPRISE_INTERVAL,
    "polymarket": POLYMARKET_INTERVAL,
    "rate_monitor": RATE_MONITOR_INTERVAL,
    "yield_curve": YIELD_CURVE_INTERVAL,
    "g10_yields": G10_YIELDS_INTERVAL,
    "fred_macro": FRED_MACRO_INTERVAL,
    "short_interest": SHORT_INTEREST_INTERVAL,
    "wikipedia": WIKIPEDIA_INTERVAL, "wiki_pageviews": WIKI_PAGEVIEWS_INTERVAL, "macro_calendar": MACRO_CALENDAR_INTERVAL, "tic": TIC_INTERVAL,
    "cisa_kev": CISA_KEV_INTERVAL,
    "benzinga_analyst": BENZINGA_INTERVAL,
    "ftc_doj": FTC_DOJ_INTERVAL,
    "fed_press": FED_PRESS_INTERVAL,
    "eia": EIA_INTERVAL,
    "ecb_press": ECB_PRESS_INTERVAL,
    "boj_press": BOJ_PRESS_INTERVAL,
    "boe_press": BOE_PRESS_INTERVAL,
    "whitehouse": WHITEHOUSE_INTERVAL,
    "g10_cb": G10_CB_INTERVAL,
    "global_reg": GLOBAL_REG_INTERVAL,
    "bis": BIS_INTERVAL,
    "federal_register": FED_REG_INTERVAL,
    "bls": BLS_INTERVAL,
    "bea": BEA_INTERVAL,
    "globenewswire": GLOBENEWSWIRE_INTERVAL,
    "short_seller": SHORT_SELLER_INTERVAL,
    "financial_blogs": FINANCIAL_BLOGS_INTERVAL,
    "hackernews": HACKERNEWS_INTERVAL,
    "usaspending": USASPENDING_INTERVAL,
    "market_breadth": MARKET_BREADTH_INTERVAL,
    "sec_xbrl": SEC_XBRL_INTERVAL,
    "sec_13f": SEC_13F_INTERVAL,
    "sec_form4": SEC_FORM4_INTERVAL,
    "usgs_quake": USGS_QUAKE_INTERVAL,
    "nasdaq_halts": NASDAQ_HALTS_INTERVAL,
    "nasdaq_ipo":      NASDAQ_IPO_INTERVAL,
    "nasdaq_earnings": NASDAQ_EARNINGS_INTERVAL,
    "putcall_ratio":   PUTCALL_RATIO_INTERVAL,
    "fda": FDA_INTERVAL,
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
    """Score and insert articles into store. Returns new count.

    An article that already carries ``_relevance_score`` is treated as
    pre-scored by its collector — the heuristic scorer is NOT re-run on it.
    Synthetic operations alerts (the ``collector_rate_monitor`` SILENT
    notifications, in particular) carry no portfolio tickers / financial
    keywords, so ``_heuristic_score_article`` returns 0.0 on them and the
    0.5 noise gate below silently dropped every one — the operations-alert
    path was inert in production. The opt-in pre-scoring contract lets a
    collector deliberately set its own ``kw_score`` (the same convention
    ``vix_ts`` / ``dxy`` / ``sector_etf`` use via their direct-write
    pattern) without bypassing this central insert path. Existing
    collectors that DON'T set ``_relevance_score`` are byte-unchanged —
    their articles are still heuristic-scored and 0.5-noise-gated exactly
    as before. Backtest isolation is untouched (`store.insert_batch` is
    the same call; the ``_LIVE_ONLY_CLAUSE`` read filter in the store is
    where invariant #1 is enforced, not here)."""
    for art in articles:
        if "_relevance_score" in art:
            continue  # collector pre-scored — honor its score, skip heuristic
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


# ── Worker W4b: StockTwits trending — re-poll every 90s ──────────────────────
def stocktwits_worker(store: ArticleStore):
    log.info("[stocktwits_worker] started")
    bo = Backoff("stocktwits", base=5.0, cap=300.0)
    while _running:
        try:
            articles = collect_stocktwits()
            _ingest(store, articles, "stocktwits")
            try:
                source_health.record_result("stocktwits", len(articles))
            except Exception as he:
                log.warning(f"[stocktwits_worker] source_health error: {he}")
            _worker_last_ok["stocktwits"] = time.time()
            bo.reset()
        except Exception as e:
            log.warning(f"[stocktwits_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(STOCKTWITS_INTERVAL)


def stocktwits_sentiment_worker(store: ArticleStore):
    log.info("[stocktwits_sentiment_worker] started")
    bo = Backoff("stocktwits_sentiment", base=10.0, cap=600.0)
    while _running:
        try:
            articles = collect_stocktwits_sentiment()
            _ingest(store, articles, "stocktwits/sentiment")
            try:
                source_health.record_result("stocktwits_sentiment", len(articles))
            except Exception as he:
                log.warning(f"[stocktwits_sentiment_worker] source_health error: {he}")
            _worker_last_ok["stocktwits_sentiment"] = time.time()
            bo.reset()
        except Exception as e:
            log.warning(f"[stocktwits_sentiment_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(300)  # re-scan every 5 min; internal cursor prevents per-ticker spam


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


# ── Worker: Benzinga analyst ratings — every 5min ───────────────────────────
def benzinga_analyst_worker(store: ArticleStore):
    log.info("[benzinga_analyst_worker] started")
    bo = Backoff("benzinga_analyst", base=5.0, cap=300.0)
    while _running:
        try:
            articles = collect_benzinga_analyst()
            _ingest(store, articles, "benzinga_analyst")
            try:
                source_health.record_result("benzinga_analyst", len(articles))
            except Exception as he:
                log.warning(f"[benzinga_analyst_worker] source_health error: {he}")
            _worker_last_ok["benzinga_analyst"] = time.time()
            bo.reset()
        except Exception as e:
            log.warning(f"[benzinga_analyst_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(BENZINGA_INTERVAL)


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


# ── Worker: SEC Activist / M&A — every 10min ────────────────────────────────
def sec_activist_worker(store: ArticleStore):
    log.info("[sec_activist_worker] started")
    bo = Backoff("sec_activist", base=10.0, cap=600.0)
    while _running:
        try:
            articles = collect_sec_activist()
            _ingest(store, articles, "sec_activist")
            try:
                source_health.record_result("sec_activist", len(articles))
            except Exception as he:
                log.warning(f"[sec_activist_worker] source_health error: {he}")
            _worker_last_ok["sec_activist"] = time.time()
            bo.reset()
        except Exception as e:
            log.warning(f"[sec_activist_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(SEC_ACTIVIST_INTERVAL)


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


# ── Worker: Yahoo Finance trending tickers — every 5min ─────────────────────
def yahoo_trending_worker(store: ArticleStore):
    log.info("[yahoo_trending_worker] started")
    bo = Backoff("yahoo_trending", base=10.0, cap=600.0)
    while _running:
        try:
            articles = collect_yahoo_trending()
            _ingest(store, articles, "yahoo_trending")
            try:
                source_health.record_result("yahoo_trending", len(articles))
            except Exception as he:
                log.warning(f"[yahoo_trending_worker] source_health error: {he}")
            _worker_last_ok["yahoo_trending"] = time.time()
            log.debug(f"[yahoo_trending] cycle ok ({len(articles)} new)")
            bo.reset()
        except Exception as e:
            log.warning(f"[yahoo_trending_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(YAHOO_TRENDING_INTERVAL)


# ── Worker: Unusual Volume screener — every 5min ────────────────────────────
def unusual_volume_worker(store: ArticleStore):
    log.info("[unusual_volume_worker] started")
    bo = Backoff("unusual_volume", base=10.0, cap=600.0)
    while _running:
        try:
            articles = collect_unusual_volume()
            _ingest(store, articles, "unusual_volume")
            try:
                source_health.record_result("unusual_volume", len(articles))
            except Exception as he:
                log.warning(f"[unusual_volume_worker] source_health error: {he}")
            _worker_last_ok["unusual_volume"] = time.time()
            log.debug(f"[unusual_volume] cycle ok ({len(articles)} new)")
            bo.reset()
        except Exception as e:
            log.warning(f"[unusual_volume_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(MARKET_MOVERS_INTERVAL)


def short_squeeze_worker(store: ArticleStore):
    log.info("[short_squeeze_worker] started")
    bo = Backoff("short_squeeze", base=10.0, cap=600.0)
    while _running:
        try:
            articles = collect_short_squeeze()
            _ingest(store, articles, "short_squeeze")
            try:
                source_health.record_result("short_squeeze", len(articles))
            except Exception as he:
                log.warning(f"[short_squeeze_worker] source_health error: {he}")
            _worker_last_ok["short_squeeze"] = time.time()
            log.debug(f"[short_squeeze] cycle ok ({len(articles)} new)")
            bo.reset()
        except Exception as e:
            log.warning(f"[short_squeeze_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(MARKET_MOVERS_INTERVAL)


# ── Worker: CNN Fear & Greed Index — every 10min ────────────────────────────
def fear_greed_worker(store: ArticleStore):
    log.info("[fear_greed_worker] started")
    bo = Backoff("fear_greed", base=30.0, cap=600.0)
    while _running:
        try:
            articles = collect_fear_greed()
            _ingest(store, articles, "fear_greed")
            try:
                source_health.record_result("fear_greed", len(articles))
            except Exception as he:
                log.warning(f"[fear_greed_worker] source_health error: {he}")
            _worker_last_ok["fear_greed"] = time.time()
            log.debug(f"[fear_greed] cycle ok ({len(articles)} new)")
            bo.reset()
        except Exception as e:
            log.warning(f"[fear_greed_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(FEAR_GREED_INTERVAL)


# ── Worker: TWSE Semiconductor pre-market tracker — every 1h ────────────────
def twse_semiconductor_worker(store: ArticleStore):
    log.info("[twse_semiconductor_worker] started")
    bo = Backoff("twse_semiconductor", base=60.0, cap=1800.0)
    while _running:
        try:
            articles = collect_twse_semiconductor()
            _ingest(store, articles, "twse_semiconductor")
            try:
                source_health.record_result("twse_semiconductor", len(articles))
            except Exception as he:
                log.warning(f"[twse_semiconductor_worker] source_health error: {he}")
            _worker_last_ok["twse_semiconductor"] = time.time()
            log.debug(f"[twse_semiconductor] cycle ok ({len(articles)} new)")
            bo.reset()
        except Exception as e:
            log.warning(f"[twse_semiconductor_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(TWSE_INTERVAL)


# ── Worker: Nasdaq IPO calendar — every 1h ──────────────────────────────────
def nasdaq_ipo_worker(store: ArticleStore):
    log.info("[nasdaq_ipo_worker] started")
    bo = Backoff("nasdaq_ipo", base=60.0, cap=1800.0)
    while _running:
        try:
            articles = collect_nasdaq_ipo()
            _ingest(store, articles, "nasdaq_ipo")
            try:
                source_health.record_result("nasdaq_ipo", len(articles))
            except Exception as he:
                log.warning(f"[nasdaq_ipo_worker] source_health error: {he}")
            _worker_last_ok["nasdaq_ipo"] = time.time()
            log.debug(f"[nasdaq_ipo] cycle ok ({len(articles)} new)")
            bo.reset()
        except Exception as e:
            log.warning(f"[nasdaq_ipo_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(NASDAQ_IPO_INTERVAL)


# ── Worker: NASDAQ Earnings Calendar — broad market, hourly ─────────────────
def nasdaq_earnings_worker(store: ArticleStore):
    log.info("[nasdaq_earnings_worker] started")
    bo = Backoff("nasdaq_earnings", base=60.0, cap=1800.0)
    while _running:
        try:
            articles = collect_nasdaq_earnings()
            _ingest(store, articles, "nasdaq_earnings")
            try:
                source_health.record_result("nasdaq_earnings", len(articles))
            except Exception as he:
                log.warning(f"[nasdaq_earnings_worker] source_health error: {he}")
            _worker_last_ok["nasdaq_earnings"] = time.time()
            log.debug(f"[nasdaq_earnings] cycle ok ({len(articles)} new)")
            bo.reset()
        except Exception as e:
            log.warning(f"[nasdaq_earnings_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(NASDAQ_EARNINGS_INTERVAL)


# ── Worker: Market put/call ratio — every 10min ─────────────────────────────
def putcall_ratio_worker(store: ArticleStore):
    log.info("[putcall_ratio_worker] started")
    bo = Backoff("putcall_ratio", base=60.0, cap=1800.0)
    while _running:
        try:
            articles = collect_putcall_ratio()
            _ingest(store, articles, "putcall_ratio")
            try:
                source_health.record_result("putcall_ratio", len(articles))
            except Exception as he:
                log.warning(f"[putcall_ratio_worker] source_health error: {he}")
            _worker_last_ok["putcall_ratio"] = time.time()
            log.debug(f"[putcall_ratio] cycle ok ({len(articles)} new)")
            bo.reset()
        except Exception as e:
            log.warning(f"[putcall_ratio_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(PUTCALL_RATIO_INTERVAL)


# ── Worker: Crypto Fear & Greed Index — every 30min ─────────────────────────
def crypto_fear_greed_worker(store: ArticleStore):
    log.info("[crypto_fear_greed_worker] started")
    bo = Backoff("crypto_fear_greed", base=60.0, cap=900.0)
    while _running:
        try:
            articles = collect_crypto_fear_greed()
            _ingest(store, articles, "crypto_fear_greed")
            try:
                source_health.record_result("crypto_fear_greed", len(articles))
            except Exception as he:
                log.warning(f"[crypto_fear_greed_worker] source_health error: {he}")
            _worker_last_ok["crypto_fear_greed"] = time.time()
            log.debug(f"[crypto_fear_greed] cycle ok ({len(articles)} new)")
            bo.reset()
        except Exception as e:
            log.warning(f"[crypto_fear_greed_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(CRYPTO_FEAR_GREED_INTERVAL)


# ── Worker: OKX Perpetual Funding Rates — every 30min ────────────────────────
def crypto_funding_worker(store: ArticleStore):
    log.info("[crypto_funding_worker] started")
    bo = Backoff("crypto_funding", base=60.0, cap=900.0)
    while _running:
        try:
            articles = collect_crypto_funding()
            _ingest(store, articles, "crypto_funding")
            try:
                source_health.record_result("crypto_funding", len(articles))
            except Exception as he:
                log.warning(f"[crypto_funding_worker] source_health error: {he}")
            _worker_last_ok["crypto_funding"] = time.time()
            log.debug(f"[crypto_funding] cycle ok ({len(articles)} new)")
            bo.reset()
        except Exception as e:
            log.warning(f"[crypto_funding_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(CRYPTO_FUNDING_INTERVAL)


def earnings_surprise_worker(store: ArticleStore):
    log.info("[earnings_surprise_worker] started")
    bo = Backoff("earnings_surprise", base=60.0, cap=900.0)
    while _running:
        try:
            articles = collect_earnings_surprises()
            _ingest(store, articles, "earnings_surprise")
            try:
                source_health.record_result("earnings_surprise", len(articles))
            except Exception as he:
                log.warning(f"[earnings_surprise_worker] source_health error: {he}")
            _worker_last_ok["earnings_surprise"] = time.time()
            log.debug(f"[earnings_surprise] cycle ok ({len(articles)} new)")
            bo.reset()
        except Exception as e:
            log.warning(f"[earnings_surprise_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(EARNINGS_SURPRISE_INTERVAL)


def polymarket_worker(store: ArticleStore):
    log.info("[polymarket_worker] started")
    bo = Backoff("polymarket", base=30.0, cap=600.0)
    while _running:
        try:
            articles = collect_polymarket()
            _ingest(store, articles, "polymarket")
            try:
                source_health.record_result("polymarket", len(articles))
            except Exception as he:
                log.warning(f"[polymarket_worker] source_health error: {he}")
            _worker_last_ok["polymarket"] = time.time()
            log.debug(f"[polymarket] cycle ok ({len(articles)} new)")
            bo.reset()
        except Exception as e:
            log.warning(f"[polymarket_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(POLYMARKET_INTERVAL)


# ── Worker: Manifold Markets prediction markets — every 30min ─────────────────
def manifold_worker(store: ArticleStore):
    log.info("[manifold_worker] started")
    bo = Backoff("manifold", base=30.0, cap=600.0)
    while _running:
        try:
            articles = collect_manifold()
            _ingest(store, articles, "manifold")
            try:
                source_health.record_result("manifold", len(articles))
            except Exception as he:
                log.warning(f"[manifold_worker] source_health error: {he}")
            _worker_last_ok["manifold"] = time.time()
            bo.reset()
        except Exception as e:
            log.warning(f"[manifold_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(MANIFOLD_INTERVAL)


# ── Worker: per-collector silence detector — every 1h ────────────────────────
def rate_monitor_worker(store: ArticleStore):
    log.info("[rate_monitor_worker] started")
    bo = Backoff("rate_monitor", base=60.0, cap=1800.0)
    while _running:
        try:
            articles = collect_rate_alerts()
            _ingest(store, articles, "collector_monitor")
            try:
                source_health.record_result("collector_monitor", len(articles))
            except Exception as he:
                log.warning(f"[rate_monitor_worker] source_health error: {he}")
            _worker_last_ok["rate_monitor"] = time.time()
            log.debug(f"[rate_monitor] cycle ok ({len(articles)} silent alerts)")
            bo.reset()
        except Exception as e:
            log.warning(f"[rate_monitor_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(RATE_MONITOR_INTERVAL)


# ── Worker: 10Y-2Y yield-curve inversion monitor — every 1h ─────────────────
def yield_curve_worker(store: ArticleStore):
    log.info("[yield_curve_worker] started")
    bo = Backoff("yield_curve", base=60.0, cap=1800.0)
    while _running:
        try:
            articles = collect_yield_curve()
            _ingest(store, articles, "yield_curve")
            try:
                source_health.record_result("yield_curve", len(articles))
            except Exception as he:
                log.warning(f"[yield_curve_worker] source_health error: {he}")
            _worker_last_ok["yield_curve"] = time.time()
            log.debug(f"[yield_curve] cycle ok ({len(articles)} new)")
            bo.reset()
        except Exception as e:
            log.warning(f"[yield_curve_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(YIELD_CURVE_INTERVAL)


def g10_yields_worker(store: ArticleStore):
    log.info("[g10_yields_worker] started")
    bo = Backoff("g10_yields", base=60.0, cap=1800.0)
    while _running:
        try:
            articles = collect_g10_yields()
            _ingest(store, articles, "g10_yields")
            try:
                source_health.record_result("g10_yields", len(articles))
            except Exception as he:
                log.warning(f"[g10_yields_worker] source_health error: {he}")
            _worker_last_ok["g10_yields"] = time.time()
            log.debug(f"[g10_yields] cycle ok ({len(articles)} new)")
            bo.reset()
        except Exception as e:
            log.warning(f"[g10_yields_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(G10_YIELDS_INTERVAL)


def fred_macro_worker(store: ArticleStore):
    log.info("[fred_macro_worker] started")
    bo = Backoff("fred_macro", base=60.0, cap=1800.0)
    while _running:
        try:
            articles = collect_fred()
            _ingest(store, articles, "fred_macro")
            try:
                source_health.record_result("fred_macro", len(articles))
            except Exception as he:
                log.warning(f"[fred_macro_worker] source_health error: {he}")
            _worker_last_ok["fred_macro"] = time.time()
            log.debug(f"[fred_macro] cycle ok ({len(articles)} new)")
            bo.reset()
        except Exception as e:
            log.warning(f"[fred_macro_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(FRED_MACRO_INTERVAL)


def cftc_cot_worker(store: ArticleStore):
    log.info("[cftc_cot_worker] started")
    bo = Backoff("cftc_cot", base=300.0, cap=3600.0)
    while _running:
        try:
            articles = collect_cftc_cot()
            _ingest(store, articles, "cftc_cot")
            try:
                source_health.record_result("cftc_cot", len(articles))
            except Exception as he:
                log.warning(f"[cftc_cot_worker] source_health error: {he}")
            _worker_last_ok["cftc_cot"] = time.time()
            log.debug(f"[cftc_cot] cycle ok ({len(articles)} new)")
            bo.reset()
        except Exception as e:
            log.warning(f"[cftc_cot_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(COT_INTERVAL)


# ── Worker: High short interest monitor — every 6h ──────────────────────────
def short_interest_worker(store: ArticleStore):
    log.info("[short_interest_worker] started")
    bo = Backoff("short_interest", base=60.0, cap=3600.0)
    while _running:
        try:
            articles = collect_short_interest()
            _ingest(store, articles, "short_interest")
            try:
                source_health.record_result("short_interest", len(articles))
            except Exception as he:
                log.warning(f"[short_interest_worker] source_health error: {he}")
            _worker_last_ok["short_interest"] = time.time()
            log.debug(f"[short_interest] cycle ok ({len(articles)} new)")
            bo.reset()
        except Exception as e:
            log.warning(f"[short_interest_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(SHORT_INTEREST_INTERVAL)


# ── Worker: VIX term structure snapshot — every 10min ───────────────────────
def vix_ts_worker(store: ArticleStore):
    log.info("[vix_ts_worker] started")
    bo = Backoff("vix_ts", base=30.0, cap=600.0)
    while _running:
        try:
            # collector inserts directly; count emitted rows
            articles = collect_vix_ts()
            n = len(articles)
            try:
                source_health.record_result("vix_ts", n)
            except Exception as he:
                log.warning(f"[vix_ts_worker] source_health error: {he}")
            _worker_last_ok["vix_ts"] = time.time()
            if n:
                log.info(f"[vix_ts] emitted {n} article(s)")
            bo.reset()
        except Exception as e:
            log.warning(f"[vix_ts_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(VIX_TS_INTERVAL)


# ── Worker: DXY + major-pair FX snapshot — every 10min ─────────────────────
def dxy_worker(store: ArticleStore):
    log.info("[dxy_worker] started")
    bo = Backoff("dxy", base=30.0, cap=600.0)
    while _running:
        try:
            articles = collect_dxy()
            n = len(articles)
            try:
                source_health.record_result("dxy", n)
            except Exception as he:
                log.warning(f"[dxy_worker] source_health error: {he}")
            _worker_last_ok["dxy"] = time.time()
            if n:
                log.info(f"[dxy] emitted {n} article(s)")
            bo.reset()
        except Exception as e:
            log.warning(f"[dxy_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(DXY_INTERVAL)


# ── Worker: Commodity futures price monitor — every 10min ───────────────────
def commodity_futures_worker(store: ArticleStore):
    log.info("[commodity_futures_worker] started")
    bo = Backoff("commodity_futures", base=30.0, cap=600.0)
    while _running:
        try:
            articles = collect_commodity_futures()
            n = len(articles)
            try:
                source_health.record_result("commodity_futures", n)
            except Exception as he:
                log.warning(f"[commodity_futures_worker] source_health error: {he}")
            _worker_last_ok["commodity_futures"] = time.time()
            if n:
                log.info(f"[commodity_futures] emitted {n} article(s)")
            bo.reset()
        except Exception as e:
            log.warning(f"[commodity_futures_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(COMMODITY_FUTURES_INTERVAL)


# ── Worker: EDGAR Form 4 insider cluster-buy detection — every 10min ────────
def insider_cluster_worker(store: ArticleStore):
    log.info("[insider_cluster_worker] started")
    bo = Backoff("insider_cluster", base=30.0, cap=600.0)
    while _running:
        try:
            articles = collect_insider_cluster()
            n = len(articles)
            if n:
                _ingest(store, articles, "insider_cluster")
                log.info(f"[insider_cluster] emitted {n} cluster alert(s)")
            try:
                source_health.record_result("insider_cluster", n)
            except Exception as he:
                log.warning(f"[insider_cluster_worker] source_health error: {he}")
            _worker_last_ok["insider_cluster"] = time.time()
            bo.reset()
        except Exception as e:
            log.warning(f"[insider_cluster_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(INSIDER_CLUSTER_INTERVAL)


# ── Worker: Sector ETF momentum / rotation — every 10min ────────────────────
def sector_etf_worker(store: ArticleStore):
    log.info("[sector_etf_worker] started")
    bo = Backoff("sector_etf", base=30.0, cap=600.0)
    while _running:
        try:
            articles = collect_sector_etf()
            n = len(articles)
            try:
                source_health.record_result("sector_etf", n)
            except Exception as he:
                log.warning(f"[sector_etf_worker] source_health error: {he}")
            _worker_last_ok["sector_etf"] = time.time()
            if n:
                log.info(f"[sector_etf] emitted {n} article(s)")
            bo.reset()
        except Exception as e:
            log.warning(f"[sector_etf_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(SECTOR_ETF_INTERVAL)


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


# ── Worker: Wikipedia pageview surge alerts — every 1h ──────────────────────
def wiki_pageviews_worker(store: ArticleStore):
    log.info("[wiki_pageviews_worker] started")
    bo = Backoff("wiki_pageviews", base=60.0, cap=1800.0)
    while _running:
        try:
            articles = collect_wikipedia_pageviews()
            _ingest(store, articles, "wiki_pageviews")
            try:
                source_health.record_result("wiki_pageviews", len(articles))
            except Exception as he:
                log.warning(f"[wiki_pageviews_worker] source_health error: {he}")
            _worker_last_ok["wiki_pageviews"] = time.time()
            log.debug(f"[wiki_pageviews] cycle ok ({len(articles)} new)")
            bo.reset()
        except Exception as e:
            log.warning(f"[wiki_pageviews_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(WIKI_PAGEVIEWS_INTERVAL)


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


def tic_worker(store: ArticleStore):
    log.info("[tic_worker] started")
    bo = Backoff("tic", base=120.0, cap=1800.0)
    while _running:
        try:
            articles = collect_tic()
            _ingest(store, articles, "tic")
            try:
                source_health.record_result("tic_foreign_holdings", len(articles))
            except Exception as he:
                log.warning(f"[tic_worker] source_health error: {he}")
            _worker_last_ok["tic"] = time.time()
            log.debug(f"[tic] cycle ok ({len(articles)} new articles)")
            bo.reset()
        except Exception as e:
            log.warning(f"[tic_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(TIC_INTERVAL)


def forex_factory_cal_worker(store: ArticleStore):
    log.info("[forex_factory_cal_worker] started")
    bo = Backoff("forex_factory_cal", base=120.0, cap=1800.0)
    while _running:
        try:
            articles = collect_forex_factory_cal()
            _ingest(store, articles, "forex_factory_calendar")
            try:
                source_health.record_result("forex_factory_calendar", len(articles))
            except Exception as he:
                log.warning(f"[forex_factory_cal_worker] source_health error: {he}")
            _worker_last_ok["forex_factory_cal"] = time.time()
            log.debug(f"[forex_factory_cal] cycle ok ({len(articles)} new events)")
            bo.reset()
        except Exception as e:
            log.warning(f"[forex_factory_cal_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(FOREX_FACTORY_CAL_INTERVAL)


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
def cboe_unusual_options_worker(store: ArticleStore):
    log.info("[cboe_unusual_options_worker] started")
    bo = Backoff("cboe_unusual_options", base=60.0, cap=900.0)
    while _running:
        try:
            articles = collect_cboe_unusual_options()
            _ingest(store, articles, "cboe_unusual_options")
            try:
                source_health.record_result("cboe_unusual_options", len(articles))
            except Exception as he:
                log.warning(f"[cboe_unusual_options_worker] source_health error: {he}")
            _worker_last_ok["cboe_unusual_options"] = time.time()
            log.debug(f"[cboe_unusual_options] cycle ok ({len(articles)} new rows)")
            bo.reset()
        except Exception as e:
            log.warning(f"[cboe_unusual_options_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(CBOE_UNUSUAL_OPTIONS_INTERVAL)


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


# ── Worker: CISA Known Exploited Vulnerabilities — every 1h ────────────────
def cisa_kev_worker(store: ArticleStore):
    log.info("[cisa_kev_worker] started")
    bo = Backoff("cisa_kev", base=60.0, cap=900.0)
    while _running:
        try:
            articles = collect_cisa_kev()
            _ingest(store, articles, "cisa_kev")
            try:
                source_health.record_result("cisa_kev", len(articles))
            except Exception as he:
                log.warning(f"[cisa_kev_worker] source_health error: {he}")
            _worker_last_ok["cisa_kev"] = time.time()
            log.debug(f"[cisa_kev] cycle ok ({len(articles)} new rows)")
            bo.reset()
        except Exception as e:
            log.warning(f"[cisa_kev_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(CISA_KEV_INTERVAL)


# ── Worker: Federal Reserve press / speech / testimony — every 30min ──────
# These four central-bank press workers all follow the cisa_kev pattern: pull
# the collector (which handles its own dedup via seen_articles.db), hand the
# fresh entries to _ingest() so the heuristic scorer / insert_batch path runs
# verbatim, and ping source_health. Press releases are low-volume (handfuls
# per day), so 30min is high enough to avoid worsening the chronic
# insert_batch lock contention and low enough to catch a same-hour FOMC /
# ECB / BoJ decision before the next briefing cycle.
def ftc_doj_worker(store: ArticleStore):
    log.info("[ftc_doj_worker] started")
    bo = Backoff("ftc_doj", base=60.0, cap=900.0)
    while _running:
        try:
            articles = collect_ftc_doj()
            _ingest(store, articles, "ftc_doj")
            try:
                source_health.record_result("ftc_doj", len(articles))
            except Exception as he:
                log.warning(f"[ftc_doj_worker] source_health error: {he}")
            _worker_last_ok["ftc_doj"] = time.time()
            log.debug(f"[ftc_doj] cycle ok ({len(articles)} new rows)")
            bo.reset()
        except Exception as e:
            log.warning(f"[ftc_doj_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(FTC_DOJ_INTERVAL)


def eia_worker(store: ArticleStore):
    log.info("[eia_worker] started")
    bo = Backoff("eia", base=60.0, cap=900.0)
    while _running:
        try:
            articles = collect_eia()
            _ingest(store, articles, "eia")
            try:
                source_health.record_result("eia", len(articles))
            except Exception as he:
                log.warning(f"[eia_worker] source_health error: {he}")
            _worker_last_ok["eia"] = time.time()
            log.debug(f"[eia] cycle ok ({len(articles)} new rows)")
            bo.reset()
        except Exception as e:
            log.warning(f"[eia_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(EIA_INTERVAL)


def fed_press_worker(store: ArticleStore):
    log.info("[fed_press_worker] started")
    bo = Backoff("fed_press", base=60.0, cap=900.0)
    while _running:
        try:
            articles = collect_fed_press()
            _ingest(store, articles, "fed_press")
            try:
                source_health.record_result("fed_press", len(articles))
            except Exception as he:
                log.warning(f"[fed_press_worker] source_health error: {he}")
            _worker_last_ok["fed_press"] = time.time()
            log.debug(f"[fed_press] cycle ok ({len(articles)} new rows)")
            bo.reset()
        except Exception as e:
            log.warning(f"[fed_press_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(FED_PRESS_INTERVAL)


def ecb_press_worker(store: ArticleStore):
    log.info("[ecb_press_worker] started")
    bo = Backoff("ecb_press", base=60.0, cap=900.0)
    while _running:
        try:
            articles = collect_ecb_press()
            _ingest(store, articles, "ecb_press")
            try:
                source_health.record_result("ecb_press", len(articles))
            except Exception as he:
                log.warning(f"[ecb_press_worker] source_health error: {he}")
            _worker_last_ok["ecb_press"] = time.time()
            log.debug(f"[ecb_press] cycle ok ({len(articles)} new rows)")
            bo.reset()
        except Exception as e:
            log.warning(f"[ecb_press_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(ECB_PRESS_INTERVAL)


def boj_press_worker(store: ArticleStore):
    log.info("[boj_press_worker] started")
    bo = Backoff("boj_press", base=60.0, cap=900.0)
    while _running:
        try:
            articles = collect_boj_press()
            _ingest(store, articles, "boj_press")
            try:
                source_health.record_result("boj_press", len(articles))
            except Exception as he:
                log.warning(f"[boj_press_worker] source_health error: {he}")
            _worker_last_ok["boj_press"] = time.time()
            log.debug(f"[boj_press] cycle ok ({len(articles)} new rows)")
            bo.reset()
        except Exception as e:
            log.warning(f"[boj_press_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(BOJ_PRESS_INTERVAL)


def boe_press_worker(store: ArticleStore):
    log.info("[boe_press_worker] started")
    bo = Backoff("boe_press", base=60.0, cap=900.0)
    while _running:
        try:
            articles = collect_boe_press()
            _ingest(store, articles, "boe_press")
            try:
                source_health.record_result("boe_press", len(articles))
            except Exception as he:
                log.warning(f"[boe_press_worker] source_health error: {he}")
            _worker_last_ok["boe_press"] = time.time()
            log.debug(f"[boe_press] cycle ok ({len(articles)} new rows)")
            bo.reset()
        except Exception as e:
            log.warning(f"[boe_press_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(BOE_PRESS_INTERVAL)


def bls_worker(store: ArticleStore):
    log.info("[bls_worker] started")
    bo = Backoff("bls", base=120.0, cap=1800.0)
    while _running:
        try:
            articles = collect_bls()
            _ingest(store, articles, "bls")
            try:
                source_health.record_result("bls", len(articles))
            except Exception as he:
                log.warning(f"[bls_worker] source_health error: {he}")
            _worker_last_ok["bls"] = time.time()
            log.debug(f"[bls] cycle ok ({len(articles)} new rows)")
            bo.reset()
        except Exception as e:
            log.warning(f"[bls_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(BLS_INTERVAL)


def bea_worker(store: ArticleStore):
    log.info("[bea_worker] started")
    bo = Backoff("bea", base=120.0, cap=1800.0)
    while _running:
        try:
            articles = collect_bea()
            _ingest(store, articles, "bea")
            try:
                source_health.record_result("bea", len(articles))
            except Exception as he:
                log.warning(f"[bea_worker] source_health error: {he}")
            _worker_last_ok["bea"] = time.time()
            log.debug(f"[bea] cycle ok ({len(articles)} releases)")
            bo.reset()
        except Exception as e:
            log.warning(f"[bea_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(BEA_INTERVAL)


def federal_register_worker(store: ArticleStore):
    log.info("[federal_register_worker] started")
    bo = Backoff("federal_register", base=120.0, cap=1800.0)
    while _running:
        try:
            articles = collect_federal_register()
            _ingest(store, articles, "federal_register")
            try:
                source_health.record_result("federal_register", len(articles))
            except Exception as he:
                log.warning(f"[federal_register_worker] source_health error: {he}")
            _worker_last_ok["federal_register"] = time.time()
            log.debug(f"[federal_register] cycle ok ({len(articles)} new regulatory actions)")
            bo.reset()
        except Exception as e:
            log.warning(f"[federal_register_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(FED_REG_INTERVAL)


def whitehouse_worker(store: ArticleStore):
    log.info("[whitehouse_worker] started")
    bo = Backoff("whitehouse", base=60.0, cap=900.0)
    while _running:
        try:
            articles = collect_whitehouse()
            _ingest(store, articles, "whitehouse")
            try:
                source_health.record_result("whitehouse", len(articles))
            except Exception as he:
                log.warning(f"[whitehouse_worker] source_health error: {he}")
            _worker_last_ok["whitehouse"] = time.time()
            log.debug(f"[whitehouse] cycle ok ({len(articles)} new rows)")
            bo.reset()
        except Exception as e:
            log.warning(f"[whitehouse_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(WHITEHOUSE_INTERVAL)


def g10_cb_worker(store: ArticleStore):
    log.info("[g10_cb_worker] started")
    bo = Backoff("g10_cb", base=60.0, cap=900.0)
    while _running:
        try:
            articles = collect_g10_central_banks()
            _ingest(store, articles, "g10_cb")
            try:
                source_health.record_result("g10_cb", len(articles))
            except Exception as he:
                log.warning(f"[g10_cb_worker] source_health error: {he}")
            _worker_last_ok["g10_cb"] = time.time()
            log.debug(f"[g10_cb] cycle ok ({len(articles)} new rows)")
            bo.reset()
        except Exception as e:
            log.warning(f"[g10_cb_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(G10_CB_INTERVAL)


# ── Worker: USGS M≥5 earthquake feed — every 30min ───────────────────────────
# Seismic events with magnitude ≥5 in populated regions are a recurring
# insurance / semiconductor-supply-chain / energy-infrastructure catalyst no
# other collector covers. The USGS feed itself updates every minute, but
# M≥5 quakes are rare enough (~3/day worldwide) that 30min is plenty fast
# to land an event in the next briefing without worsening writer contention.
# The collector applies its own M≥5.0 floor and dedups by USGS event_id, so
# magnitude revisions of the same quake do NOT re-emit.
def usgs_quake_worker(store: ArticleStore):
    log.info("[usgs_quake_worker] started")
    bo = Backoff("usgs_quake", base=60.0, cap=900.0)
    while _running:
        try:
            articles = collect_usgs_earthquakes()
            _ingest(store, articles, "usgs_quake")
            try:
                source_health.record_result("usgs_quake", len(articles))
            except Exception as he:
                log.warning(f"[usgs_quake_worker] source_health error: {he}")
            _worker_last_ok["usgs_quake"] = time.time()
            log.debug(f"[usgs_quake] cycle ok ({len(articles)} new rows)")
            bo.reset()
        except Exception as e:
            log.warning(f"[usgs_quake_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(USGS_QUAKE_INTERVAL)


# ── Worker: NASDAQ/UTP trading halts — every 2min ────────────────────────────
def nasdaq_halts_worker(store: ArticleStore):
    log.info("[nasdaq_halts_worker] started")
    bo = Backoff("nasdaq_halts", base=30.0, cap=600.0)
    while _running:
        try:
            articles = collect_nasdaq_halts()
            _ingest(store, articles, "nasdaq_halts")
            try:
                source_health.record_result("nasdaq_halts", len(articles))
            except Exception as he:
                log.warning(f"[nasdaq_halts_worker] source_health error: {he}")
            _worker_last_ok["nasdaq_halts"] = time.time()
            log.debug(f"[nasdaq_halts] cycle ok ({len(articles)} new rows)")
            bo.reset()
        except Exception as e:
            log.warning(f"[nasdaq_halts_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(NASDAQ_HALTS_INTERVAL)


# ── Worker: FDA press releases + MedWatch safety alerts — every 30min ────────
def fda_worker(store: ArticleStore):
    log.info("[fda_worker] started")
    bo = Backoff("fda", base=60.0, cap=900.0)
    while _running:
        try:
            articles = collect_fda()
            _ingest(store, articles, "fda")
            try:
                source_health.record_result("fda", len(articles))
            except Exception as he:
                log.warning(f"[fda_worker] source_health error: {he}")
            _worker_last_ok["fda"] = time.time()
            log.debug(f"[fda] cycle ok ({len(articles)} new rows)")
            bo.reset()
        except Exception as e:
            log.warning(f"[fda_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(FDA_INTERVAL)


# ── Worker: Global financial regulators — every 30min ────────────────────────
def global_reg_worker(store: ArticleStore):
    log.info("[global_reg_worker] started")
    bo = Backoff("global_reg", base=60.0, cap=900.0)
    while _running:
        try:
            articles = collect_global_regulators()
            _ingest(store, articles, "global_reg")
            try:
                source_health.record_result("global_reg", len(articles))
            except Exception as he:
                log.warning(f"[global_reg_worker] source_health error: {he}")
            _worker_last_ok["global_reg"] = time.time()
            log.debug(f"[global_reg] cycle ok ({len(articles)} new rows)")
            bo.reset()
        except Exception as e:
            log.warning(f"[global_reg_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(GLOBAL_REG_INTERVAL)


# ── Worker: BIS press releases, speeches, research — every 30min ─────────────
def bis_worker(store: ArticleStore):
    log.info("[bis_worker] started")
    bo = Backoff("bis", base=60.0, cap=900.0)
    while _running:
        try:
            articles = collect_imf_bis_worldbank()
            _ingest(store, articles, "bis")
            try:
                source_health.record_result("bis", len(articles))
            except Exception as he:
                log.warning(f"[bis_worker] source_health error: {he}")
            _worker_last_ok["bis"] = time.time()
            log.debug(f"[bis] cycle ok ({len(articles)} new rows)")
            bo.reset()
        except Exception as e:
            log.warning(f"[bis_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(BIS_INTERVAL)



# ── Worker: UN News — economic/climate/regional RSS feeds — every 30min ──────
def un_news_worker(store: ArticleStore):
    log.info("[un_news_worker] started")
    bo = Backoff("un_news", base=60.0, cap=900.0)
    while True:
        try:
            articles = collect_un_news()
            _ingest(store, articles, "un_news")
            try:
                source_health.record_result("un_news", len(articles))
            except Exception as he:
                log.warning(f"[un_news_worker] source_health error: {he}")
            _worker_last_ok["un_news"] = time.time()
            log.debug(f"[un_news] cycle ok ({len(articles)} new rows)")
            bo.reset()
        except Exception as e:
            log.warning(f"[un_news_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.advance()
        _sleep(UN_NEWS_INTERVAL)


# ── Worker: GlobeNewswire financial press releases — every 10min ─────────────
def globenewswire_worker(store: ArticleStore):
    log.info("[globenewswire_worker] started")
    bo = Backoff("globenewswire", base=60.0, cap=900.0)
    while _running:
        try:
            articles = collect_globenewswire()
            _ingest(store, articles, "globenewswire")
            try:
                source_health.record_result("globenewswire", len(articles))
            except Exception as he:
                log.warning(f"[globenewswire_worker] source_health error: {he}")
            _worker_last_ok["globenewswire"] = time.time()
            log.debug(f"[globenewswire] cycle ok ({len(articles)} new rows)")
            bo.reset()
        except Exception as e:
            log.warning(f"[globenewswire_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(GLOBENEWSWIRE_INTERVAL)


# ── Worker: Short-seller research reports — every 30min ─────────────────────
def short_seller_worker(store: ArticleStore):
    log.info("[short_seller_worker] started")
    bo = Backoff("short_seller", base=120.0, cap=1800.0)
    while _running:
        try:
            articles = collect_short_sellers()
            _ingest(store, articles, "short_seller")
            try:
                source_health.record_result("short_seller", len(articles))
            except Exception as he:
                log.warning(f"[short_seller_worker] source_health error: {he}")
            _worker_last_ok["short_seller"] = time.time()
            log.debug(f"[short_seller] cycle ok ({len(articles)} new rows)")
            bo.reset()
        except Exception as e:
            log.warning(f"[short_seller_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(SHORT_SELLER_INTERVAL)


# ── Worker: InvestorPlace / Motley Fool / Nasdaq RSS — every 10min ───────────
def financial_blogs_worker(store: ArticleStore):
    log.info("[financial_blogs_worker] started")
    bo = Backoff("financial_blogs", base=60.0, cap=900.0)
    while _running:
        try:
            articles = collect_financial_blogs()
            _ingest(store, articles, "financial_blogs")
            try:
                source_health.record_result("financial_blogs", len(articles))
            except Exception as he:
                log.warning(f"[financial_blogs_worker] source_health error: {he}")
            _worker_last_ok["financial_blogs"] = time.time()
            log.debug(f"[financial_blogs] cycle ok ({len(articles)} new rows)")
            bo.reset()
        except Exception as e:
            log.warning(f"[financial_blogs_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(FINANCIAL_BLOGS_INTERVAL)


# ── Worker: SEC XBRL financial facts — every 6h ─────────────────────────────
def sec_xbrl_worker(store: ArticleStore):
    log.info("[sec_xbrl_worker] started")
    bo = Backoff("sec_xbrl", base=60.0, cap=1800.0)
    while _running:
        try:
            articles = collect_sec_xbrl_financials()
            _ingest(store, articles, "sec_xbrl")
            try:
                source_health.record_result("sec_xbrl", len(articles))
            except Exception as he:
                log.warning(f"[sec_xbrl_worker] source_health error: {he}")
            _worker_last_ok["sec_xbrl"] = time.time()
            log.debug(f"[sec_xbrl] cycle ok ({len(articles)} new rows)")
            bo.reset()
        except Exception as e:
            log.warning(f"[sec_xbrl_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(SEC_XBRL_INTERVAL)


# ── Worker: SEC 13F institutional holdings — every 30min ────────────────────
def sec_13f_worker(store: ArticleStore):
    log.info("[sec_13f_worker] started")
    bo = Backoff("sec_13f", base=60.0, cap=900.0)
    while _running:
        try:
            articles = collect_13f_filings()
            _ingest(store, articles, "sec_13f")
            try:
                source_health.record_result("sec_13f", len(articles))
            except Exception as he:
                log.warning(f"[sec_13f_worker] source_health error: {he}")
            _worker_last_ok["sec_13f"] = time.time()
            bo.reset()
        except Exception as e:
            log.warning(f"[sec_13f_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(SEC_13F_INTERVAL)


# ── Worker: SEC Form 4 insider transactions (portfolio tickers) — every 5min ─
def sec_form4_worker(store: ArticleStore):
    log.info("[sec_form4_worker] started")
    bo = Backoff("sec_form4", base=30.0, cap=600.0)
    while _running:
        try:
            articles = collect_sec_form4()
            _ingest(store, articles, "sec_form4")
            try:
                source_health.record_result("sec_form4", len(articles))
            except Exception as he:
                log.warning(f"[sec_form4_worker] source_health error: {he}")
            _worker_last_ok["sec_form4"] = time.time()
            bo.reset()
        except Exception as e:
            log.warning(f"[sec_form4_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(SEC_FORM4_INTERVAL)


# ── Worker: Hacker News front-page + finance/business stories — every 5min ──
def hackernews_worker(store: ArticleStore):
    log.info("[hackernews_worker] started")
    bo = Backoff("hackernews", base=60.0, cap=900.0)
    while _running:
        try:
            articles = collect_hackernews()
            _ingest(store, articles, "hackernews")
            try:
                source_health.record_result("hackernews", len(articles))
            except Exception as he:
                log.warning(f"[hackernews_worker] source_health error: {he}")
            _worker_last_ok["hackernews"] = time.time()
            log.debug(f"[hackernews] cycle ok ({len(articles)} new rows)")
            bo.reset()
        except Exception as e:
            log.warning(f"[hackernews_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(HACKERNEWS_INTERVAL)


# ── Worker: USASpending.gov federal contract awards — hourly ─────────────────
def usaspending_worker(store: ArticleStore):
    log.info("[usaspending_worker] started")
    bo = Backoff("usaspending", base=60.0, cap=900.0)
    while _running:
        try:
            articles = collect_usaspending_contracts()
            _ingest(store, articles, "usaspending_contracts")
            try:
                source_health.record_result("usaspending_contracts", len(articles))
            except Exception as he:
                log.warning(f"[usaspending_worker] source_health error: {he}")
            _worker_last_ok["usaspending"] = time.time()
            log.debug(f"[usaspending] cycle ok ({len(articles)} new rows)")
            bo.reset()
        except Exception as e:
            log.warning(f"[usaspending_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(USASPENDING_INTERVAL)


# ── Worker: Market breadth (Finviz screener) — hourly ───────────────────────
def market_breadth_worker(store: ArticleStore):
    log.info("[market_breadth_worker] started")
    bo = Backoff("market_breadth", base=120.0, cap=1800.0)
    while _running:
        try:
            articles = collect_market_breadth()
            _ingest(store, articles, "market_breadth")
            try:
                source_health.record_result("market_breadth", len(articles))
            except Exception as he:
                log.warning(f"[market_breadth_worker] source_health error: {he}")
            _worker_last_ok["market_breadth"] = time.time()
            log.debug(f"[market_breadth] cycle ok ({len(articles)} new rows)")
            bo.reset()
        except Exception as e:
            log.warning(f"[market_breadth_worker] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(lambda: _running)
            continue
        _sleep(MARKET_BREADTH_INTERVAL)


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
def _price_alert_universe() -> list[str]:
    """Tickers the price-alert worker monitors for >=3% moves.

    The union of the static ``PORTFOLIO_TICKERS`` tuple with the live
    ``ml.features.LIVE_PORTFOLIO_TICKERS`` set (which reads positions, option
    underlyings and sector_watchlist out of ``config/portfolio.json``). The
    union — never just the static tuple — is what guarantees an open position
    added in the trading UI is never silently dropped from price alerting,
    even though the static tuple is frozen for cross-module ``_BOOK_TICKERS``
    parity. Sorted for a deterministic, test-pinnable order."""
    return sorted(set(PORTFOLIO_TICKERS) | set(LIVE_PORTFOLIO_TICKERS))


_PORTFOLIO_JSON_PATH = BASE_DIR / "config" / "portfolio.json"


def _load_held_positions() -> dict[str, dict]:
    """``{TICKER: {qty, avg_cost, type}}`` for open positions in
    config/portfolio.json. Best-effort — any error (missing / corrupt file,
    unexpected shape) degrades to ``{}`` so a price alert still fires, just
    without the held-position context line."""
    out: dict[str, dict] = {}
    try:
        with open(_PORTFOLIO_JSON_PATH, "r", encoding="utf-8") as f:
            pf = json.load(f)
        for pos in pf.get("positions", []) or []:
            tkr = ((pos or {}).get("ticker") or "").strip().upper()
            if tkr:
                out[tkr] = {
                    "qty": pos.get("qty"),
                    "avg_cost": pos.get("avg_cost"),
                    "type": pos.get("type", ""),
                }
    except Exception:
        pass
    return out


def _fmt_qty(qty) -> str:
    """Render a (possibly fractional) share quantity compactly: ``14`` /
    ``3.615`` / ``4.7095`` — never ``3.6150000001`` float dust."""
    try:
        q = float(qty)
    except (TypeError, ValueError):
        return "?"
    return str(int(q)) if q == int(q) else f"{q:.6g}"


def _price_alert_position_line(ticker: str, price: float,
                               positions: dict[str, dict]) -> str:
    """Held-position context for a price alert — the analyst persona's
    "is this MY money, and where does the move leave me vs cost?" question.

    Returns "" when ``ticker`` is watchlist-only (not an open position) or
    its avg_cost is missing/non-positive, so a non-held mover's alert stays
    a clean one-liner. Pure and deterministic for unit testing."""
    pos = positions.get((ticker or "").upper())
    if not pos:
        return ""
    try:
        avg = float(pos.get("avg_cost"))
    except (TypeError, ValueError):
        return ""
    if avg <= 0 or price is None:
        return ""
    pnl_pct = (price - avg) / avg * 100.0
    side = "above" if pnl_pct >= 0 else "below"
    return (f"💼 HELD POSITION: {_fmt_qty(pos.get('qty'))} @ ${avg:.2f} avg "
            f"— now {abs(pnl_pct):.1f}% {side} cost basis")


def _price_alert_news_line(store, ticker: str) -> str:
    """Recent-news-catalyst context for a price alert: how many live
    articles mentioned ``ticker`` in the last 60min. Tells the analyst at a
    glance whether a 3% move has a news catalyst (act) or is technical/quiet
    (watch). Best-effort — any store failure degrades to "" so the alert
    still fires. Reuses the canonical ``ticker_mention_velocity`` primitive
    (already ``_LIVE_ONLY_CLAUSE``-scoped, so synthetic backtest rows can
    never inflate the count)."""
    if store is None:
        return ""
    try:
        rows = store.ticker_mention_velocity([ticker], window_min=60)
    except Exception:
        return ""
    for r in rows or []:
        if isinstance(r, dict) and r.get("ticker") == ticker:
            try:
                recent = int(r.get("recent") or 0)
            except (TypeError, ValueError):
                recent = 0
            if recent >= 1:
                return (f"📰 {recent} live article(s) mention {ticker} in the "
                        f"last 60min — likely news catalyst")
            return ""
    return ""


def price_alert_worker(store: ArticleStore):
    log.info("[price_alert_worker] started")
    bo = Backoff("price_alert", base=5.0, cap=300.0)
    while _running:
        try:
            data = get_stock_data()
            by_ticker = {row["ticker"]: row for row in data.get("equities", [])}
            # get_stock_data() is driven by config/watchlist.json, which lags
            # config/portfolio.json — actual open positions absent from the
            # watchlist (live 2026-05-21: GOOG/COHR/NVDL/LNOK/MUU were held in
            # portfolio.json but not in watchlist.json) would get NO 3% price
            # alert at all: a silent blind spot on names the analyst has real
            # money in. Cover every live held/watched name: union the static
            # PORTFOLIO_TICKERS with ml.features.LIVE_PORTFOLIO_TICKERS (the
            # SSOT that reads positions + option underlyings + sector_watchlist
            # from config/portfolio.json), then fetch any the watchlist sweep
            # missed directly via stock_data._fetch_one.
            alert_universe = _price_alert_universe()
            # Held-position context (avg cost, P&L-vs-cost) for the alert —
            # reloaded each cycle so a fresh fill is reflected without a restart.
            held_positions = _load_held_positions()
            for tkr in alert_universe:
                if tkr not in by_ticker:
                    row = _fetch_one(tkr)
                    if row:
                        by_ticker[tkr] = row
            for tkr in alert_universe:
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
                    headline = (f"📈 PRICE ALERT: {tkr} {sign}{pct:.1f}% to "
                                f"${price:.2f} (from ${prev:.2f}, "
                                f"{PRICE_ALERT_INTERVAL // 60}min ago)")
                    # Enrich: held-position context (is this MY money, where
                    # vs cost) + recent-news-catalyst context. Both degrade to
                    # "" so a non-held / quiet mover stays a clean one-liner.
                    extra = [
                        ln for ln in (
                            _price_alert_position_line(tkr, price, held_positions),
                            _price_alert_news_line(store, tkr),
                        ) if ln
                    ]
                    msg = "\n".join([headline, *extra])
                    log.info(f"[price_alert] {headline}")
                    discord_send(msg, is_alert=True)
                _last_prices[tkr] = price
            _worker_last_ok["price_alert"] = time.time()
            log.debug(f"[price_alert] checked {len(alert_universe)} tickers")
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
    #
    # CRITICAL: ml_train()/train() catches every internal error and RETURNS a
    # status dict ({"status": "error", "reason": "subprocess_timeout"}, ...)
    # instead of raising. The try/except below therefore never observes the
    # most common real failure mode — so a returned-error cycle must be
    # classified explicitly (``_ml_retrain_is_failure``) and counted, or a
    # trainer that times out every cycle escalates nothing (observed live
    # 2026-05-22: subprocess_timeout after 659.5s, consec_fail stuck at 0).
    consec_fail = 0
    while _running:
        _sleep(ML_TRAIN_INTERVAL)
        failed = False
        last_err = ""
        try:
            log.info("[ml_trainer] Retraining on accumulated labels...")
            metrics = ml_train(store)
            if _ml_retrain_is_failure(metrics):
                failed = True
                last_err = (
                    (metrics.get("reason") or metrics.get("status") or "unknown")
                    if isinstance(metrics, dict) else str(metrics)
                )
                log.warning(f"[ml_trainer] Retrain failed: {metrics}")
            else:
                val = metrics.get("val_loss")
                val_str = f"{val:.4f}" if isinstance(val, (int, float)) else "n/a"
                fl = metrics.get("final_loss")
                fl_str = f"{fl:.4f}" if isinstance(fl, (int, float)) else "n/a"
                log.info(f"[ml_trainer] Retrain: status={metrics.get('status')} "
                         f"n={metrics.get('n')} loss={fl_str} "
                         f"val_loss={val_str} "
                         f"elapsed={metrics.get('elapsed_s', 0):.0f}s")
                # Only a real completed cycle records a loss metric — a
                # "skipped" no-op has no final_loss and recording 0 would
                # plant a misleading perfect-loss point in the metrics log.
                if metrics.get("status") == "ok":
                    record_metric("ml.train.loss", metrics.get("final_loss", 0),
                                  {"n": metrics.get("n", 0), "phase": "retrain"})
                _worker_last_ok["ml_trainer"] = time.time()
                consec_fail = 0
        except MemoryError:
            # GPU OOM has its own backoff path and is not counted toward the
            # consecutive-failure escalation (unchanged from prior behaviour).
            _handle_memory_error("ml_trainer")
        except Exception as e:
            failed = True
            last_err = str(e)
            log.warning(f"[ml_trainer] Retrain exception: {e}")
        if failed:
            consec_fail += 1
            log.warning(f"[ml_trainer] retrain failure #{consec_fail} — {last_err}")
            if _ml_retrain_should_alert(consec_fail):
                try:
                    discord_send(
                        _ml_retrain_alert_message(consec_fail, last_err),
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
                #
                # Pass the LIVE held/watched union (positions + option
                # underlyings + sector_watchlist from config/portfolio.json,
                # ∪ the static PORTFOLIO_TICKERS fallback) instead of letting
                # the function default to the frozen static tuple. The static
                # tuple alone was silently dropping positions added in the
                # trading UI (GOOG/COHR/NVDL on 2026-05-23) — the briefing's
                # coverage line then read 12/12 covered when the analyst's
                # *actual* book had 3 silent names with money at risk. The
                # function default stays as the static tuple for back-compat
                # with the unit tests that pin it; this caller — the live
                # heartbeat path — uses the union.
                coverage_line = _format_portfolio_coverage(
                    source_articles, tickers=_price_alert_universe()
                )
                # Label-calibration line: of urgency>=1 rows in the last 5h,
                # what fraction carried a real LLM ground-truth label vs only
                # a model self-prediction? Silent on healthy/quiet windows,
                # emits only on Sonnet-dark or majority-unverified-storm —
                # exact dashboard-verdict parity. Discord-only, same
                # discipline as the other augmentation lines.
                calibration_line = _format_label_calibration(store)
                # Training-pool composition: parallel to calibration_line but
                # for the TRAINING corpus rather than the short-horizon urgent
                # stream. Silent on healthy windows; emits on extreme
                # synthetic-dominance / Claude-label-dark conditions. Same
                # Discord-only / never-folded-into-saved-briefing discipline.
                training_pool_line = _format_training_pool_composition()
                # Coverage-gap banner is Discord-only — NOT folded into the
                # saved `briefing` text, so it can't reach the trainer's
                # title-prefix label scan (same discipline as health_line).
                banner = _coverage_gap_banner(gap_h)
                message = (
                    (banner + "\n\n" if banner else "")
                    + briefing.rstrip() + "\n\n" + health_line
                    + ("\n" + coverage_line if coverage_line else "")
                    + ("\n" + calibration_line if calibration_line else "")
                    + ("\n" + training_pool_line if training_pool_line else "")
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
# Cheap urgency=1 reaper cadence BETWEEN full 6h purges. The full purge is
# heavy (DELETE + WAL TRUNCATE) and only fires every 6h, so without an
# in-between cadence a urgency=1 row that aged out of the alerter's 24h
# fetch window can linger un-demoted for up to ~6 hours past the cutoff —
# invisible to the alert worker (push lost) yet still inflating the
# dashboard's urgent tile and the ``overdue`` count in
# ``urgent_queue_health``. Live evidence (2026-05-23 16:30Z): 22 of 81
# queued urgency=1 rows were >24h old (some 29-30h old), never alerted,
# awaiting the next purge_old fire. ``reap_stale_urgent`` is idempotent +
# cheap (one indexed UPDATE) so running it hourly costs nothing while
# shrinking the worst-case stuck-urgent-row lifetime from ~30h to ~25h.
URGENT_REAP_INTERVAL = 3600


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
    # Startup reap counts as the most recent reap; pin ``last_reap`` to now so
    # the hourly cadence below doesn't immediately re-fire on the first tick.
    last_reap = time.time()
    while _running:
        _worker_last_ok["purge"] = time.time()
        _sleep(300)
        # Hourly cheap reap BETWEEN full 6h purges. Without this, a
        # urgency=1 row that aged past the alerter's 24h fetch window stays
        # urgency=1 until the next purge_old fires — up to ~6 additional
        # hours of invisible inflation on the urgent tile. The reap is one
        # indexed UPDATE so even a no-op cycle costs essentially nothing.
        if time.time() - last_reap >= URGENT_REAP_INTERVAL:
            try:
                with _store_lock:
                    n = store.reap_stale_urgent()
                if n:
                    log.info(
                        f"[purge_worker] hourly-reaped {n} aged urgency=1 "
                        f"row(s) (first_seen older than the alerter's 24h "
                        f"window — never pushed, now demoted to urgency=0)"
                    )
                last_reap = time.time()
            except Exception as e:
                log.warning(f"[purge_worker] hourly reap failed: {e}")
        if time.time() - last_purge < PURGE_INTERVAL:
            continue
        try:
            with _store_lock:
                store.purge_old()
            # purge_old internally calls reap_stale_urgent — sync ``last_reap``
            # to ``now`` so we don't redundantly re-fire the cheap reap a few
            # minutes later.
            last_reap = time.time()
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


def _format_label_calibration(
    store: ArticleStore, hours: int = 5, max_chars: int = 150
) -> str:
    """One-line urgent-row calibration signal for the heartbeat briefing.

    The 5h digest already surfaces source-health and book-coverage; the
    aggregate that was missing is *how much of this window's urgent stream
    carried a real LLM ground-truth label* vs only an unverified
    model-self-prediction. The per-row "[unverified — model-only urgent]"
    alert tag (see ``ArticleStore.get_unalerted_urgent``'s ``_llm_vetted``
    key) hedges individual pushes, but nothing exposed the cohort fact —
    when the Sonnet ``urgency_scorer`` is dark / quota-throttled / flooring
    everything to noise the standalone-push channel becomes single-headed
    and the analyst should know.

    Mirrors the silence-on-healthy discipline of
    ``_format_source_health_summary`` / ``_format_portfolio_coverage`` (and
    the chat enrichment blocks' ``_event_readiness`` / ``_macro_calendar``
    precedent): a quiet or healthy window emits ``""`` so the briefing
    stays clean; only an actionable calibration miss emits a line. Verdict
    thresholds mirror the ``/api/urgent-label-split`` endpoint
    (``dashboard/web_server.py``) byte-for-byte so the briefing surface
    cannot drift from the dashboard verdict.

    Read-only — composes ``ArticleStore.urgency_label_split`` (a single
    GROUP BY SELECT with ``_LIVE_ONLY_CLAUSE``, no writes, synthetic rows
    excluded). No ``ai_score`` / ``ml_score`` / ``score_source`` / urgency
    mutation. All four load-bearing invariants intact by construction.
    Discord-only — the caller appends to the posted message, NEVER folds
    into the saved ``briefing`` text, so the trainer's title-prefix label
    scan cannot reach it (same discipline as the source-health line, the
    coverage line, and the coverage-gap banner).
    """
    try:
        data = store.urgency_label_split(hours=hours)
    except Exception:
        return ""  # best-effort — a metric outage must never block a briefing
    total = int(data.get("total") or 0)
    if total == 0:
        return ""  # quiet window — nothing actionable to say
    try:
        llm_fraction = float(data.get("llm_fraction") or 0.0)
    except (TypeError, ValueError):
        return ""
    # Same verdict ladder as dashboard /api/urgent-label-split — keep aligned.
    if llm_fraction == 0.0 and total >= 3:
        verdict = "storm"
    elif llm_fraction < 0.5 and total >= 5:
        verdict = "mostly_unverified"
    else:
        return ""  # healthy — silent
    by_src = data.get("by_source") or {}
    ml_n = int(by_src.get("ml") or 0)
    pct = int(round(llm_fraction * 100))
    if verdict == "storm":
        # 0% LLM-vetted means the Sonnet urgency_scorer hasn't labeled a
        # single urgent row in the window — most likely dark / quota / wedged.
        line = (
            f"🔬 Urgent calibration: 0% LLM-vetted last {hours}h "
            f"({ml_n}/{total} ML-only) — Sonnet scorer dark"
        )
    else:
        line = (
            f"🔬 Urgent calibration: {pct}% LLM-vetted last {hours}h "
            f"({ml_n}/{total} ML-only)"
        )
    if len(line) > max_chars:
        line = line[: max_chars - 1].rstrip() + "…"
    return line


def _format_training_pool_composition(
    audit_fn=None, max_chars: int = 200
) -> str:
    """One-line training-pool composition signal for the heartbeat briefing.

    The 5h digest already carries an ``Urgent calibration`` line for the
    short-horizon urgent stream's LLM-vetted fraction
    (``_format_label_calibration``); this is the parallel signal for the
    *training* corpus that ArticleNet itself is fit against. Live evidence
    (2026-05-20 articles.db): the strong-label pool sits at 96.5%
    synthetic backtest/opus rows vs 3.5% real Claude-tagged labels (with
    Sonnet quota chronically throttling urgency_scorer). Synthetic IS
    legitimate training signal per CLAUDE.md §5, but the analyst persona
    "react to events affecting MY positions" cares whether the model's
    relevance/urgency head is mostly remembering replayed-trade outcomes
    vs. learning from fresh Claude judgments — that question was never
    answerable from any consumed product. The new
    ``synthetic_fraction_of_strong`` / ``llm_fraction_of_strong`` audit
    fields make it answerable; this surfaces it in the briefing.

    Mirrors the silence-on-healthy discipline of
    ``_format_label_calibration`` / ``_format_source_health_summary`` /
    ``_format_portfolio_coverage``: emit only when the composition is
    extreme. Two thresholds:

      * ``llm_fraction < 0.05`` — Claude labels effectively absent
        (urgency_scorer dark or quota-floored to almost nothing); the
        strong pool is learning ~entirely from backtest replay outcomes.
      * ``synthetic_fraction >= 0.85`` — synthetic-dominant; the model's
        signal is still mostly the paper-trader's replay outcomes, even
        if some Claude labels are present.

    Otherwise silent — a healthy mix (>=15% Claude-tagged labels) needs
    no analyst-facing callout.

    Read-only by construction: ``ml.label_audit.audit`` issues four
    ``COUNT(*)`` queries with ``_LIVE_ONLY_CLAUSE`` /
    ``STRONG_LABEL_WHERE`` only; this helper opens a fresh ``mode=ro``
    connection via ``label_audit._RoStore`` (NEVER the daemon's shared
    ``self.conn`` — the documented cursor-collision hazard, same
    discipline as ``analysis.claude_analyst._collect_macro_calendar_events``).
    No ``ai_score`` / ``ml_score`` / ``score_source`` / urgency mutation,
    synthetic rows already separated from live rows by the audit's own
    bucket logic. All four load-bearing invariants intact by construction.

    Discord-only — the caller appends to the posted message, NEVER folds
    into the saved ``briefing`` text, so the trainer's title-prefix
    label scan cannot reach it (same discipline as ``_build_health_line``
    / ``_format_portfolio_coverage`` / ``_format_label_calibration``).

    ``audit_fn`` is an optional injectable returning the same dict shape
    as ``label_audit.audit`` — purely for testability so the helper can
    be exercised against a controlled in-memory bucket distribution
    without touching the real DB. Production callers omit it; the helper
    opens its own ``mode=ro`` connection.
    """
    try:
        if audit_fn is None:
            from ml import label_audit
            from storage.article_store import _get_db_path
            store = label_audit._RoStore(_get_db_path())
            try:
                data = label_audit.audit(store)
            finally:
                store.close()
        else:
            data = audit_fn()
    except Exception:
        return ""  # best-effort — a metric outage must never block a briefing
    if not isinstance(data, dict):
        return ""
    sp = data.get("strong_pool") or {}
    try:
        total = int(sp.get("total") or 0)
    except (TypeError, ValueError):
        return ""
    # A very small pool can swing dramatically on a single label arrival —
    # the composition number would be analyst-noise, not signal. Threshold
    # tuned to the size at which the synthetic/Claude ratio is a stable
    # estimate (well below the live ~530k strong-pool size).
    if total < 100:
        return ""
    try:
        synth_frac = float(data.get("synthetic_fraction_of_strong") or 0.0)
        llm_frac = float(data.get("llm_fraction_of_strong") or 0.0)
    except (TypeError, ValueError):
        return ""
    try:
        synth_n = int(sp.get("synthetic_backtest_opus") or 0)
        llm_n = (int(sp.get("llm") or 0) + int(sp.get("briefing_boost") or 0))
    except (TypeError, ValueError):
        return ""
    pct = int(round(llm_frac * 100))
    if llm_frac < 0.05:
        line = (
            f"🧪 Training pool: only {pct}% Claude-tagged labels "
            f"({llm_n} LLM vs {synth_n} synthetic) — "
            f"model learns mostly from backtest replay"
        )
    elif synth_frac >= 0.85:
        line = (
            f"🧪 Training pool: {pct}% Claude-tagged labels "
            f"({llm_n} LLM vs {synth_n} synthetic) — "
            f"synthetic-dominant"
        )
    else:
        return ""  # healthy mix — silent
    if len(line) > max_chars:
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
        ("stocktwits",  stocktwits_worker),
        ("stocktwits_sentiment", stocktwits_sentiment_worker),
        ("ticker",      ticker_worker),
        ("sec_edgar",   sec_edgar_worker),
        ("sec_edgar_ft", sec_edgar_ft_worker),
        ("sec_activist", sec_activist_worker),
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
        ("yahoo_trending", yahoo_trending_worker),
        ("unusual_volume", unusual_volume_worker),
        ("short_squeeze",  short_squeeze_worker),
        ("short_interest", short_interest_worker),
        ("fear_greed",  fear_greed_worker),
        ("crypto_fear_greed", crypto_fear_greed_worker),
        ("crypto_funding",   crypto_funding_worker),
        ("earnings_surprise", earnings_surprise_worker),
        ("polymarket",  polymarket_worker),
        ("manifold",    manifold_worker),
        ("rate_monitor", rate_monitor_worker),
        ("yield_curve", yield_curve_worker),
        ("g10_yields",  g10_yields_worker),
        ("fred_macro",  fred_macro_worker),
        ("cftc_cot",    cftc_cot_worker),
        ("vix_ts",      vix_ts_worker),
        ("dxy",         dxy_worker),
        ("commodity_futures", commodity_futures_worker),
        ("sector_etf",  sector_etf_worker),
        ("wikipedia",   wikipedia_worker),
        ("wiki_pageviews", wiki_pageviews_worker),
        ("macro_calendar", macro_calendar_worker),
        ("tic",            tic_worker),
        ("forex_factory_cal", forex_factory_cal_worker),
        ("finra_short",   finra_short_worker),
        ("congress_trades", congress_trades_worker),
        ("cboe_unusual_options", cboe_unusual_options_worker),
        ("cisa_kev",    cisa_kev_worker),
        ("insider_cluster", insider_cluster_worker),
        ("benzinga_analyst", benzinga_analyst_worker),
        ("ftc_doj",     ftc_doj_worker),
        ("eia",         eia_worker),
        ("fed_press",   fed_press_worker),
        ("ecb_press",   ecb_press_worker),
        ("boj_press",   boj_press_worker),
        ("boe_press",   boe_press_worker),
        ("bls",         bls_worker),
        ("bea",         bea_worker),
        ("federal_register", federal_register_worker),
        ("whitehouse",  whitehouse_worker),
        ("g10_cb",      g10_cb_worker),
        ("global_reg",  global_reg_worker),
        ("bis",         bis_worker),
        ("un_news",     un_news_worker),
        ("globenewswire", globenewswire_worker),
        ("short_seller", short_seller_worker),
        ("financial_blogs", financial_blogs_worker),
        ("hackernews",     hackernews_worker),
        ("usaspending",    usaspending_worker),
        ("market_breadth", market_breadth_worker),
        ("sec_xbrl",    sec_xbrl_worker),
        ("sec_13f",     sec_13f_worker),
        ("sec_form4",   sec_form4_worker),
        ("usgs_quake",  usgs_quake_worker),
        ("nasdaq_halts", nasdaq_halts_worker),
        ("twse_semiconductor", twse_semiconductor_worker),
        ("nasdaq_ipo",   nasdaq_ipo_worker),
        ("nasdaq_earnings", nasdaq_earnings_worker),
        ("putcall_ratio", putcall_ratio_worker),
        ("fda",         fda_worker),
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
