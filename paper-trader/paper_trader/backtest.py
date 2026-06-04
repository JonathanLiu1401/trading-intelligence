"""Backtesting engine — runs N independent year-long simulations.

Each run starts with $1000, samples every 5th NYSE trading day,
fetches historical news from GDELT, scores with a keyword heuristic,
and asks Opus 4.7 for trading decisions. Stocks-only (no options —
yfinance has no historical option prices). Stop-loss / take-profit
are checked daily between sampled decisions using cached closes.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import shutil
import sqlite3
import subprocess
import threading
import time
import urllib.parse
import urllib.request

import requests
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import yfinance as yf

from .strategy import MODEL
from .llm_adapter import call_llm as _llm_call

ROOT = Path(__file__).resolve().parent.parent
BACKTEST_DB = ROOT / "backtest.db"
CACHE_DIR = ROOT / "data" / "backtest_cache"
GDELT_CACHE = CACHE_DIR / "gdelt"
# Legacy cache path — kept for migration; new caches are per-window.
PRICE_CACHE_PATH = CACHE_DIR / "prices.json"

INITIAL_CASH = 1000.0
SAMPLE_EVERY_N_DAYS = 1         # daily decisions — 1 per trading day
MAX_DECISIONS_PER_DAY = 10     # intraday loop: up to N ml_decide calls per trading day
GDELT_RATE_LIMIT_S = 5.5       # GDELT actual limit is ~1 req/5s; use 5.5 for safety
GDELT_MAX_RECORDS = 100
GDELT_RETRY_BACKOFF_S = 20.0  # reduced; 30s was too conservative
GDELT_WARM_WORKERS = 1        # single worker — parallel workers share rate-limit lock and deadlock
GDELT_MAX_WARM_REQUESTS = 150  # cap per warm cycle — full window warming takes hours; not worth it
OPUS_TIMEOUT_S = 150
# Concurrency for claude subprocesses now lives in paper_trader.llm_adapter
# (`_CLAUDE_SEM`) so it is shared across all callers that route through
# `call_llm`. The orphaned module-level definition was removed in Task 3.

WATCHLIST = [
    # Core US large-cap + semis (kept from v1 watchlist)
    "SPY", "QQQ", "NVDA", "AMD", "MU", "LITE", "AMAT", "LRCX",
    "SMH", "TSM", "INTC", "QCOM", "AAPL", "MSFT", "META", "GOOGL",
    "AMZN", "TSLA", "CRM", "SNOW", "BTC-USD", "GC=F",
    # Power semis / GaN / SiC
    "NVTS", "MPWR", "WOLF", "STM", "MCHP", "AMBA", "SWKS", "QRVO",
    # AI infrastructure / accelerators
    "SMCI", "AVGO", "ARM", "MRVL", "CDNS", "SNPS", "AEHR", "COHU",
    # Quantum computing
    "IONQ", "RGTI", "QUBT", "ARQQ", "QMCO",
    # High-momentum speculative / space / autonomy / nuclear
    "RKLB", "LUNR", "AST", "ACHR", "JOBY", "OKLO", "NNE",
    # Global / ADR
    "BABA", "ASML", "SAP", "NVO", "TM", "SONY", "HSBC", "BP", "RIO", "BHP",
    # US financials
    "GS", "JPM", "BAC", "BRK-B",
    # Energy / healthcare / payments
    "XOM", "CVX", "LLY", "UNH", "V", "MA",
    # Fintech / crypto-adjacent / speculative
    "SHOP", "SQ", "COIN", "MSTR", "PLTR", "RIVN", "NIO", "ARKK",
    # AI software / voice / defense tech
    "SOUN", "BBAI", "ASTS", "BFLY", "PRCT",
    # Macro / commodity ETFs
    "TLT", "GLD", "SLV", "USO", "UNG",
    # Leveraged ETFs — 3x Bull
    "TQQQ", "UPRO", "SPXL", "UDOW", "URTY",        # index 3x
    "SOXL", "TECL", "FNGU", "CURE", "LABU",         # sector 3x (semis/tech/health/bio)
    "NAIL", "WANT", "DFEN", "MIDU", "TNA",           # housing/China/defense/mid/small 3x
    "DPST", "FAS", "HIBL", "UTSL",                   # banks/financials/high-beta/utilities 3x
    # Leveraged ETFs — 2x Bull
    "QLD", "SSO", "MVV", "SAA", "UWM",               # index 2x
    "NVDU", "NVDX", "MSFU", "AMZU", "GOOGU", "METAU", # single-stock 2x (NVDA/MSFT/AMZN/GOOG/META)
    "TSLT", "AAPLU", "CONL", "TSLL",                 # Tesla/Apple/Coinbase 2x
    "LNOK", "SMCI2X", "PLTU",                        # Nokia/SMCI/Palantir 2x
    "USD", "ROM", "UXI", "UYG",                      # tech/industrial/financial 2x
    # Leveraged ETFs — Bear / Inverse (for hedging)
    "SQQQ", "SPXS", "SDOW", "SRTY",                  # index 3x inverse
    "SOXS", "TECS", "FNGD",                          # sector 3x inverse
    "TZA", "FAZ", "HIBS",                            # small/financial/high-beta inverse
    # Crypto/commodity leveraged
    "BITX", "BITU", "ETHU",                          # crypto 2x
    "BOIL", "UNG", "UCO", "AGQ",                     # nat gas/oil/silver 2x
    # Market structure / sector rotation gauges
    "^VIX", "XLK", "XLE", "XLF", "XLV", "XLI",
]

# Subset of the watchlist for which we compute heavier technical indicators
# (RSI/MACD/MA crossover/volume/52w proximity). Top 10 most-traded large caps
# plus the index proxies.
QUANT_SIGNAL_TICKERS = [
    # Tech / semis — high-growth mega-caps
    "SPY", "QQQ", "NVDA", "AMD", "MU", "TSM", "AAPL", "MSFT", "META",
    "GOOGL", "TSLA", "CRM", "SNOW",
    # Power semis / AI infra — new high-growth additions
    "NVTS", "MPWR", "AVGO", "ARM", "MRVL", "SMCI",
    # Quantum computing — high-volatility names where momentum signals matter
    "IONQ", "RGTI", "QUBT",
    # Momentum speculative — space / voice AI
    "RKLB", "SOUN", "AST",
    # Speculative growth / crypto-adjacent
    "PLTR", "COIN", "MSTR",
    # Single-stock 2x leveraged
    "NVDU", "MSFU", "AMZU", "TSLL", "TSLT", "BITU",
    # Leveraged index / sector ETFs — RSI/MACD computed so they score as buy candidates
    "TQQQ", "SOXL", "UPRO", "SPXL", "TECL", "UDOW", "URTY", "FNGU", "LABU", "FAS",
    # Consumer / retail
    "AMZN", "SHOP",
]

# Leveraged ETFs that can receive elevated conviction when signals are very strong.
# Coverage audit: every WATCHLIST ticker that the inline section comments classify
# as a "Leveraged ETF — Bull" (3x bull / 2x bull / crypto 2x / commodity 2x) MUST
# appear here so the `_ml_decide` conviction-cap arm (0.40 in bull/sideways) fires
# uniformly across the documented set. Prior to the audit 18 watchlist
# leveraged-bull tickers were missing — WANT/MIDU/TNA/UTSL (3x), SAA/UWM/GOOGU/
# METAU/AAPLU/CONL/SMCI2X/PLTU/USD/ROM/UXI/UYG (2x), and BITX/ETHU (crypto 2x)
# — silently capped at the regular 0.25 conviction so the documented
# leveraged-vehicle thesis (CLAUDE.md §3) was unreachable for those names.
# Inverse leveraged ETFs (SQQQ/SPXS/SOXS/TECS/FNGD/etc.) are intentionally
# EXCLUDED: the cap-arm gate is `regime in ("bull", "sideways")`, where buying a
# leveraged-SHORT vehicle is a counter-thesis trade that should NOT receive the
# elevated bull-conviction cap. The `WATCHLIST_LEVERAGED_BULL` enumeration below
# pins the coverage so a future watchlist addition surfaces in `_ml_decide`
# behaviour rather than silently degrading to the regular cap.
_LEVERAGED_ETFS = {
    # 3x bull
    "SOXL", "TQQQ", "UPRO", "SPXL", "UDOW", "URTY", "TECL", "FNGU", "CURE", "LABU",
    "NAIL", "DFEN", "DPST", "FAS", "HIBL", "WANT", "MIDU", "TNA", "UTSL",
    # 2x bull (index / single-stock)
    "QLD", "SSO", "MVV", "SAA", "UWM",
    "NVDU", "NVDX", "MSFU", "AMZU", "GOOGU", "METAU", "TSLT", "TSLL",
    "AAPLU", "CONL", "SMCI2X", "PLTU", "LNOK",
    "USD", "ROM", "UXI", "UYG",
    # Crypto / commodity 2x
    "BITU", "BITX", "ETHU", "BOIL", "UCO", "AGQ",
}
# LNOK is a thin OTC name and yfinance often returns nothing → omitted from default fetch.

# ─────────────────────────── trading personas ───────────────────────────
# Each parallel run gets a distinct style so the 10 runs do not converge on
# identical trades when fed the same news. Keyed by run_id; callers should map
# arbitrary run_ids onto 1..10 with ((run_id - 1) % 10) + 1 so continuous
# cycling stays inside the dict.
PERSONAS: dict[int, dict[str, str]] = {
    1: {
        "name": "Value Investor",
        "style": (
            "You are a deep-value investor in the Buffett / Graham tradition. "
            "Hunt for undervalued cash-flow machines: low P/E, low P/B, high free cash "
            "flow yield, durable competitive moats, healthy balance sheets. Be skeptical "
            "of hype; prefer boring mature businesses trading below intrinsic value. "
            "Avoid momentum chases and unprofitable speculative names entirely. Hold for "
            "the thesis to play out — patience is the edge."
        ),
    },
    2: {
        "name": "Momentum Trader",
        "style": (
            "You are a price-momentum trader. Buy what is already going up. Earnings "
            "beats with raised guidance are gold — pile in. Chase breakouts, ride trends, "
            "respect the tape. Strength begets strength; weakness begets weakness. Cut "
            "losers fast, let winners run. Avoid contrarian bottom-fishing; never catch "
            "falling knives. Stops are tight; size scales with conviction in the trend."
        ),
    },
    3: {
        "name": "Contrarian",
        "style": (
            "You are a contrarian investor. Buy fear, sell greed. When headlines scream "
            "panic and quality names get dumped in indiscriminate selloffs, you step in. "
            "When everyone is euphoric and price targets are being raised in unison, you "
            "trim. Look for oversold quality — strong businesses under temporary clouds. "
            "Ignore momentum; trust mean reversion. Comfortable being early and lonely."
        ),
    },
    4: {
        "name": "Global Macro",
        "style": (
            "You are a macro-aware growth rotator. Translate macro inflection points into "
            "leveraged equity positions — pile into SOXL/TQQQ/TECL when rates fall and tech "
            "re-rates, rotate into BITU/COIN/MSTR on dollar weakness and risk-on shifts, "
            "buy LABU/CURE on healthcare innovation cycles. The thesis is always expressed "
            "through growth instruments — never bonds, gold, or commodities. Macro analysis "
            "is the edge; leveraged ETFs on high-growth sectors are the vehicle."
        ),
    },
    5: {
        "name": "Growth at a Reasonable Price (GARP)",
        "style": (
            "You are a GARP investor — Peter Lynch / Terry Smith style. Find high-quality "
            "compounders growing revenue >15% with sane multiples and improving margins. "
            "Avoid pure value traps AND avoid bubble-multiple growth. The sweet spot is "
            "underappreciated quality growth: AI infrastructure, healthcare innovators, "
            "high-ROIC consumer brands. Size moderately and hold for the compounding."
        ),
    },
    6: {
        "name": "Quant / Event-Driven",
        "style": (
            "You are a pure-signal quant. Trade the news, not the story. Earnings beats, "
            "guidance revisions, FDA approvals, M&A leaks, regulatory catalysts — react "
            "fast and unemotionally. Treat each decision as a signal-weighted bet. "
            "No narratives, no loyalty to tickers, just probabilistic edge on catalysts. "
            "Set tight stop-losses; close positions when the catalyst is priced in."
        ),
    },
    7: {
        "name": "Sector Rotator",
        "style": (
            "You are a sector rotator. Capital flows between growth sectors as the macro "
            "cycle turns — semis on AI capex, tech on disinflation, financials when curves "
            "steepen, biotech on rate cuts. Use leveraged sector ETFs (SOXL, TECL, FAS, "
            "LABU, FNGU) and growth leaders to express rotation views. Always be long "
            "*something*; cash is the absence of a thesis. Rotate aggressively when the "
            "regime changes — there is always a growth sector to lever."
        ),
    },
    8: {
        "name": "Small / Mid Cap Hunter",
        "style": (
            "You are a small/mid cap specialist. Mega-caps are crowded and efficiently "
            "priced; your edge is in names below ~$50B market cap that institutions "
            "overlook. Hunt for hidden compounders, niche category leaders, post-IPO "
            "orphans. Avoid SPY/QQQ/NVDA/AAPL/MSFT/GOOGL/AMZN/META unless they are part "
            "of a hedge. Prefer LITE, MU (mid-cap when undervalued), RIVN, NIO, COIN, "
            "PLTR, MSTR, SHOP, SQ. Concentrate; small caps reward conviction."
        ),
    },
    9: {
        "name": "ESG / Thematic",
        "style": (
            "You are a thematic investor riding mega-trends. Clean energy transition, AI "
            "infrastructure (compute, power, cooling), semiconductor sovereignty, GLP-1 "
            "healthcare revolution, electrification. Buy the picks-and-shovels: NVDA/AMD/"
            "TSM/ASML for AI compute, LLY for GLP-1, RIVN/NIO/TSLA for EVs, ARKK for "
            "innovation beta. Ignore quarter-to-quarter noise; the trend is 5+ years."
        ),
    },
    10: {
        "name": "Pure Speculator",
        "style": (
            "You are a high-conviction speculator. Asymmetric payoffs only — small "
            "downside, massive upside. Concentrated bets, no diversification cult. "
            "When you see asymmetric setup (BTC-USD on macro shifts, MSTR/COIN/BITU as "
            "crypto leverage, MU on memory super-cycles, biotech catalysts, leveraged "
            "equity plays via SOXL/TQQQ on AI capex inflections) — go big. 100% "
            "position sizing is fine. Cash between trades, full send when the setup "
            "is right. No half-measures."
        ),
    },
}


def persona_for(run_id: int) -> dict[str, str]:
    """Map any run_id to one of the 10 personas (cycles after 10)."""
    key = ((int(run_id) - 1) % len(PERSONAS)) + 1
    return PERSONAS[key]


# Per-persona sector boosts applied to ticker_scores before final pick.
# Leveraged ETFs get strong boosts in aggressive personas — they amplify gains
# and are the primary vehicle for outperformance in high-conviction setups.
_PERSONA_BOOSTS: dict[int, dict[str, float]] = {
    1: {"MSFT": 1.5, "GOOGL": 1.5, "AMZN": 1.5, "JPM": 1.0},                # Value → growth compounders
    2: {"SOXL": 4.0, "TQQQ": 3.5, "UPRO": 3.0, "SPXL": 2.5,
        "TECL": 2.0, "FNGU": 2.0, "QQQ": 1.5},                              # Momentum → 3x ETFs
    3: {"COIN": 1.5, "PLTR": 2.0, "MSTR": 1.5, "AMD": 1.5, "RIVN": 1.0},    # Contrarian → beaten-down growth
    4: {"SOXL": 2.5, "TQQQ": 2.0, "BITU": 2.0, "NVDA": 1.5, "AMD": 1.5},    # Global Macro → leveraged equities
    5: {"LLY": 2.0, "UNH": 1.5, "AMZN": 1.5, "MSFT": 1.5,
        "CURE": 1.5, "LABU": 1.0},                                           # GARP
    6: {"SOXL": 2.0, "TQQQ": 2.0, "FAS": 2.0, "LABU": 1.5,
        "XLF": 1.5, "XLV": 1.5},                                             # Quant/Event → lev ETFs
    7: {"FAS": 2.5, "DFEN": 2.0, "LABU": 2.0, "BOIL": 1.5,
        "XLE": 2.0, "XLF": 2.0, "XLI": 1.5},                               # Sector Rotator
    8: {"COIN": 1.5, "PLTR": 1.5, "SHOP": 1.5, "SQ": 1.5,
        "MSTR": 1.5, "BITU": 1.0},                                           # Small/Mid Cap
    9: {"SOXL": 2.5, "TQQQ": 2.0, "LLY": 2.0, "DFEN": 1.5,
        "NVO": 1.5, "CURE": 1.5},                                            # ESG/Thematic
    10: {"SOXL": 3.5, "TQQQ": 3.0, "UPRO": 2.5, "BTC-USD": 2.5,
         "BITU": 2.0, "COIN": 2.0, "MSTR": 2.0},                            # Speculator → max leverage
}

KEYWORD_GROUPS = [
    # Tech / semis — core signals
    "stock market earnings semiconductor",
    "NVDA AMD Micron earnings revenue",
    "SP500 market rally selloff",
    "Federal Reserve interest rates inflation",
    # Leveraged ETF catalysts
    "SOXL TQQQ UPRO leveraged ETF rally",
    "semiconductor chip AI earnings beat",
    "Nasdaq tech rally surge breakout",
    # Sector rotation signals
    "earnings revenue profit loss guidance",
    "commodity oil gold copper energy",
    "cryptocurrency bitcoin ethereum blockchain",
    "bank earnings financial sector rate",
    "healthcare pharma biotech FDA approval",
    "defense spending military contract",
    # Macro / global
    "global markets central bank interest rates",
    "currency forex dollar euro yen",
    "inflation CPI jobs employment report",
    "China trade tariff economic data",
    "European stocks DAX FTSE earnings",
    "Asian markets Nikkei Hang Seng",
]

# Heuristic scorer lexicon
BUY_PHRASES = [
    "beat earnings", "earnings beat", "revenue beat", "guidance raised",
    "raised guidance", "record revenue", "strong demand", "supply shortage",
    "upgrade", "outperform", "all-time high", "rally", "surge", "soar",
    "buy rating", "price target raised", "expansion", "breakthrough",
]
SELL_PHRASES = [
    "miss earnings", "earnings miss", "guidance cut", "cut guidance",
    "layoffs", "inventory correction", "downgrade", "underperform",
    "recession", "selloff", "plunge", "tumble", "sell rating",
    "price target cut", "bankruptcy", "fraud", "investigation", "crash",
]
SEMIS_TICKERS = {"NVDA", "AMD", "MU", "LRCX", "AMAT", "TSM", "INTC", "ASML", "KLAC", "MRVL", "SMH", "SOXX"}


# ─────────────────────────── store ───────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS backtest_runs (
    run_id              INTEGER PRIMARY KEY,
    seed                INTEGER NOT NULL,
    start_date          TEXT NOT NULL,
    end_date            TEXT NOT NULL,
    start_value         REAL NOT NULL,
    final_value         REAL NOT NULL DEFAULT 0,
    total_return_pct    REAL NOT NULL DEFAULT 0,
    spy_return_pct      REAL NOT NULL DEFAULT 0,
    vs_spy_pct          REAL NOT NULL DEFAULT 0,
    n_trades            INTEGER NOT NULL DEFAULT 0,
    n_decisions         INTEGER NOT NULL DEFAULT 0,
    status              TEXT NOT NULL,
    started_at          TEXT NOT NULL,
    completed_at        TEXT,
    equity_curve_json   TEXT NOT NULL DEFAULT '[]',
    notes               TEXT
);

CREATE TABLE IF NOT EXISTS backtest_trades (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL,
    sim_date    TEXT NOT NULL,
    ticker      TEXT NOT NULL,
    action      TEXT NOT NULL,
    qty         REAL NOT NULL,
    price       REAL NOT NULL,
    value       REAL NOT NULL,
    reason      TEXT
);
CREATE INDEX IF NOT EXISTS idx_bt_trades_run ON backtest_trades(run_id);

CREATE TABLE IF NOT EXISTS backtest_decisions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER NOT NULL,
    sim_date        TEXT NOT NULL,
    action          TEXT,
    ticker          TEXT,
    qty             REAL,
    confidence      REAL,
    reasoning       TEXT,
    status          TEXT,
    detail          TEXT,
    cash            REAL,
    total_value     REAL,
    signal_count    INTEGER
);
CREATE INDEX IF NOT EXISTS idx_bt_dec_run ON backtest_decisions(run_id);
"""


class BacktestStore:
    def __init__(self, path: Path | None = None):
        # Resolve BACKTEST_DB at call time, not at def time. A
        # `path: Path = BACKTEST_DB` default binds the module global's value
        # when backtest.py is imported, so a later
        # `monkeypatch.setattr(bt, "BACKTEST_DB", tmp)` (conftest test
        # isolation) had no effect — every BacktestStore()/BacktestEngine()
        # silently connected to the real persistent backtest.db, polluting it
        # across tests and producing order-dependent flaky failures.
        if path is None:
            path = BACKTEST_DB
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), timeout=30, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(SCHEMA)
        self.conn.commit()
        self._lock = threading.Lock()
        self._migrate()

    def _migrate(self) -> None:
        """Idempotent schema migrations for columns added after initial deploy."""
        migrations = [
            "ALTER TABLE backtest_runs ADD COLUMN model_id TEXT NOT NULL DEFAULT 'ml_quant'",
            "ALTER TABLE backtest_runs ADD COLUMN hf_errors INT NOT NULL DEFAULT 0",
        ]
        with self._lock:
            for sql in migrations:
                try:
                    self.conn.execute(sql)
                    self.conn.commit()
                except Exception:
                    pass  # column already exists — idempotent

    def upsert_run(self, run_id: int, seed: int, status: str,
                   start: date, end: date, model_id: str = "ml_quant") -> None:
        with self._lock:
            existing = self.conn.execute(
                "SELECT run_id FROM backtest_runs WHERE run_id=?", (run_id,)
            ).fetchone()
            now = datetime.now(timezone.utc).isoformat()
            if existing:
                # Update model_id too so re-running a row under a different
                # model (e.g. retry on a different LLM) reflects honestly in
                # the persisted record rather than the original default.
                self.conn.execute(
                    "UPDATE backtest_runs SET status=?, model_id=? WHERE run_id=?",
                    (status, model_id, run_id),
                )
            else:
                self.conn.execute(
                    "INSERT INTO backtest_runs (run_id, seed, start_date, end_date, "
                    "start_value, status, started_at, model_id) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (run_id, seed, start.isoformat(), end.isoformat(),
                     INITIAL_CASH, status, now, model_id),
                )
            self.conn.commit()

    def finalize_run(self, run_id: int, final_value: float, spy_return_pct: float,
                     n_trades: int, n_decisions: int, equity_curve: list,
                     status: str = "complete", notes: str = "") -> None:
        total_return_pct = (final_value - INITIAL_CASH) / INITIAL_CASH * 100
        vs_spy = total_return_pct - spy_return_pct
        with self._lock:
            self.conn.execute(
                "UPDATE backtest_runs SET final_value=?, total_return_pct=?, "
                "spy_return_pct=?, vs_spy_pct=?, n_trades=?, n_decisions=?, "
                "equity_curve_json=?, status=?, completed_at=?, notes=? WHERE run_id=?",
                (final_value, total_return_pct, spy_return_pct, vs_spy,
                 n_trades, n_decisions, json.dumps(equity_curve), status,
                 datetime.now(timezone.utc).isoformat(), notes, run_id),
            )
            self.conn.commit()

    def update_partial_progress(self, run_id: int, current_value: float,
                                n_trades: int, n_decisions: int,
                                equity_curve: list) -> None:
        """Push in-progress equity curve + counters so the dashboard can render
        partial state while a run is still executing."""
        pct = (current_value - INITIAL_CASH) / INITIAL_CASH * 100
        with self._lock:
            self.conn.execute(
                "UPDATE backtest_runs SET final_value=?, total_return_pct=?, "
                "n_trades=?, n_decisions=?, equity_curve_json=? WHERE run_id=?",
                (current_value, pct, n_trades, n_decisions,
                 json.dumps(equity_curve), run_id),
            )
            self.conn.commit()

    def record_trade(self, run_id: int, sim_date: str, ticker: str, action: str,
                     qty: float, price: float, reason: str) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO backtest_trades (run_id, sim_date, ticker, action, qty, "
                "price, value, reason) VALUES (?,?,?,?,?,?,?,?)",
                (run_id, sim_date, ticker, action, qty, price, qty * price, reason),
            )
            self.conn.commit()

    def record_decision(self, run_id: int, sim_date: str, decision: dict | None,
                        status: str, detail: str, cash: float, total_value: float,
                        signal_count: int) -> None:
        d = decision or {}
        with self._lock:
            self.conn.execute(
                "INSERT INTO backtest_decisions (run_id, sim_date, action, ticker, qty, "
                "confidence, reasoning, status, detail, cash, total_value, signal_count) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (run_id, sim_date, d.get("action"), d.get("ticker"), d.get("qty"),
                 d.get("confidence"), d.get("reasoning"), status, detail, cash,
                 total_value, signal_count),
            )
            self.conn.commit()

    def all_runs(self, include_curves: bool = False) -> list[dict]:
        # Serialise through the lock like every other method — the connection
        # is shared (check_same_thread=False) and an unlocked read interleaved
        # with a run thread's write corrupts cursor state.
        with self._lock:
            if include_curves:
                rows = self.conn.execute(
                    "SELECT * FROM backtest_runs ORDER BY run_id ASC"
                ).fetchall()
            else:
                # Exclude equity_curve_json (25MB total) when not needed —
                # reading and discarding it was the main cause of 9s load times.
                rows = self.conn.execute(
                    "SELECT run_id, seed, start_date, end_date, start_value,"
                    " final_value, total_return_pct, spy_return_pct,"
                    " vs_spy_pct, n_trades, n_decisions, status,"
                    " started_at, completed_at, notes, model_id"
                    " FROM backtest_runs ORDER BY run_id ASC"
                ).fetchall()
        out = []
        from datetime import date as _date
        for r in rows:
            d = dict(r)
            eq_json = d.pop("equity_curve_json", None) or "[]"
            if include_curves:
                try:
                    d["equity_curve"] = json.loads(eq_json)
                except Exception:
                    d["equity_curve"] = []
            # Compute duration and annualized return from stored dates.
            try:
                s = _date.fromisoformat(d["start_date"])
                e = _date.fromisoformat(d["end_date"])
                d["duration_days"] = (e - s).days
                years = d["duration_days"] / 365.25
                if years > 0 and d.get("start_value") and d.get("final_value"):
                    growth = d["final_value"] / d["start_value"]
                    d["annualized_return_pct"] = round((growth ** (1.0 / years) - 1.0) * 100.0, 3)
                else:
                    d["annualized_return_pct"] = None
            except Exception:
                d["duration_days"] = None
                d["annualized_return_pct"] = None
            out.append(d)
        return out

    def run_curves(self, run_ids: list[int]) -> dict[int, list]:
        """Return equity_curve lists for specific run_ids (lightweight lookup)."""
        if not run_ids:
            return {}
        # Dedupe and batch the IN-clause: a single query with hundreds of
        # bound params risks SQLite's host-parameter limit (999 on older
        # builds), so chunk it. Each chunk takes the lock independently,
        # mirroring the short read-lock idiom used by other read methods.
        unique_ids = list(dict.fromkeys(run_ids))
        rows: list = []
        BATCH = 500
        for i in range(0, len(unique_ids), BATCH):
            chunk = unique_ids[i:i + BATCH]
            placeholders = ",".join("?" * len(chunk))
            with self._lock:
                rows.extend(self.conn.execute(
                    f"SELECT run_id, start_date, start_value, equity_curve_json "
                    f"FROM backtest_runs WHERE run_id IN ({placeholders})",
                    chunk,
                ).fetchall())
        out = {}
        from datetime import date as _date
        for row in rows:
            rid, start_date_str, start_val, eq_json = row
            try:
                raw = json.loads(eq_json or "[]")
            except Exception:
                raw = []
            try:
                start_d = _date.fromisoformat(start_date_str) if start_date_str else None
            except Exception:
                start_d = None
            sv = float(start_val or 1000.0)
            normalized = []
            for p in raw:
                v = float(p.get("value") or 0.0)
                day_idx = None
                if start_d and p.get("date"):
                    try:
                        day_idx = (_date.fromisoformat(p["date"]) - start_d).days
                    except Exception:
                        pass
                normalized.append({
                    "date": p.get("date"),
                    "day_index": day_idx,
                    "value": v,
                    "value_pct": round((v / sv - 1.0) * 100.0, 3) if sv else 0.0,
                })
            out[rid] = normalized
        return out

    def run_detail(self, run_id: int) -> dict | None:
        # All three reads under one lock — the shared connection is used by
        # concurrent run threads, so interleaved execute() calls would corrupt
        # cursor state and mix rows between queries.
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM backtest_runs WHERE run_id=?", (run_id,)
            ).fetchone()
            if not row:
                return None
            trades = self.conn.execute(
                "SELECT * FROM backtest_trades WHERE run_id=? ORDER BY sim_date ASC, id ASC",
                (run_id,),
            ).fetchall()
            decisions = self.conn.execute(
                "SELECT * FROM backtest_decisions WHERE run_id=? ORDER BY sim_date ASC, id ASC",
                (run_id,),
            ).fetchall()
        d = dict(row)
        try:
            d["equity_curve"] = json.loads(d.pop("equity_curve_json") or "[]")
        except Exception:
            d["equity_curve"] = []
        d["trades"] = [dict(t) for t in trades]
        d["decisions"] = [dict(x) for x in decisions]
        return d


# ─────────────────────────── price cache ───────────────────────────

class PriceCache:
    """Loads OHLCV history for all watchlist tickers once. Lookups by date."""

    def __init__(self, tickers: list[str], start: date, end: date):
        self.tickers = tickers
        self.start = start
        self.end = end
        # ticker -> {iso_date: close}
        self.prices: dict[str, dict[str, float]] = {}
        self.trading_days: list[date] = []
        self._load()

    @property
    def cache_path(self) -> Path:
        """Per-window cache file. Variable windows would otherwise collide on one file."""
        return CACHE_DIR / f"prices_{self.start.isoformat()}_{self.end.isoformat()}.json"

    def _load(self) -> None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        # Try per-window cache first, then fall back to legacy single-file cache
        # so existing 2025-05-01 → 2026-05-13 caches still work after the refactor.
        for cache_path in (self.cache_path, PRICE_CACHE_PATH):
            if not cache_path.exists():
                continue
            try:
                cached = json.loads(cache_path.read_text())
                meta = cached.get("_meta", {})
                if (meta.get("start") == self.start.isoformat()
                        and meta.get("end") == self.end.isoformat()
                        and set(meta.get("tickers", [])) >= set(self.tickers)):
                    candidate = {k: v for k, v in cached.items() if k != "_meta"}
                    # Benchmark-integrity guard. yfinance intermittently fails
                    # to return SPY at cache-build time; the old code persisted
                    # (and then re-accepted) a payload with an EMPTY SPY series
                    # because SPY was still listed in _meta.tickers. Verified
                    # live: 34/177 per-window caches are poisoned this way.
                    # Accepting one makes _build_trading_days silently fall back
                    # to another ticker's calendar and returns_pct("SPY",…)
                    # return 0.0 → vs_spy_pct is fabricated (== total_return),
                    # which then poisons the live trader's _ml_is_qualified
                    # median-alpha gate (CLAUDE.md §15) every cycle this window
                    # is redrawn. SPY has data back to its 1993 inception
                    # (== EARLIEST_WINDOW_START), so an empty SPY series is
                    # ALWAYS a transient fetch failure, never a real
                    # data-availability gap. Reject the poisoned cache and fall
                    # through to a fresh download (which self-heals the file on
                    # success; see the write-side guard below).
                    if "SPY" in self.tickers and not candidate.get("SPY"):
                        print(f"[price_cache] {cache_path.name} has an empty "
                              f"SPY series — rejecting poisoned cache, "
                              f"re-downloading")
                        continue
                    self.prices = candidate
                    self._build_trading_days()
                    print(f"[price_cache] loaded {len(self.prices)} tickers from "
                          f"{cache_path.name} ({len(self.trading_days)} trading days)")
                    return
            except Exception as e:
                print(f"[price_cache] cache read failed ({cache_path.name}): {e}")

        print(f"[price_cache] downloading {len(self.tickers)} tickers "
              f"{self.start} → {self.end} from yfinance…")
        # Pad end by +1 day because yfinance end is exclusive.
        end_pad = (self.end + timedelta(days=2)).isoformat()
        for t in self.tickers:
            try:
                hist = yf.Ticker(t).history(start=self.start.isoformat(),
                                            end=end_pad, auto_adjust=False)
                if hist.empty:
                    print(f"[price_cache]   {t}: no data")
                    self.prices[t] = {}
                    continue
                series: dict[str, float] = {}
                for ts, row in hist.iterrows():
                    iso = ts.date().isoformat()
                    close = row.get("Close")
                    if close is None or close != close:  # NaN check
                        continue
                    # Treat 0/negative closes as missing rather than poisoning
                    # the cache. yfinance can return 0.0 on a halted /
                    # illiquid intraday row (and the bulk path Agent 1 just
                    # patched in `market.get_prices` had the same class of
                    # bug). A $0 cached close would propagate into
                    # `returns_pct` as `(end - 0) / 0` (DivisionByZero — but
                    # we hit `if not s` first which is True for 0.0 and
                    # returns a *fabricated* 0.0 outcome) and into
                    # `_compute_decision_outcomes` as a flat-return training
                    # row. `_buy(price=0)` would have notional=0. SPY 0-close
                    # days would still appear in `_build_trading_days` since
                    # it iterates `spy.keys()` — making them sampled
                    # decisions with degenerate marks. Filter at the storage
                    # boundary so neither the cache file NOR the in-memory
                    # series ever carry a poisoned tick.
                    close_f = float(close)
                    if close_f <= 0:
                        continue
                    series[iso] = close_f
                self.prices[t] = series
                print(f"[price_cache]   {t}: {len(series)} rows")
            except Exception as e:
                print(f"[price_cache]   {t} failed: {e}")
                self.prices[t] = {}

        self._build_trading_days()
        # Write-side benchmark-integrity guard (paired with the cache-read
        # guard above). If SPY was requested but the download yielded an
        # empty series, this run hit the same transient yfinance failure.
        # Persisting it would re-poison the per-window cache permanently —
        # every future redraw of this window would fabricate vs_spy_pct.
        # Skip the write so the next draw retries the download fresh. The
        # run still completes off the fallback-ticker calendar built above;
        # run_one then writes the honest `benchmark_unavailable` note
        # (locked by test_integration_backtest.py::TestBenchmarkUnavailableNote).
        if "SPY" in self.tickers and not self.prices.get("SPY"):
            print(f"[price_cache] SPY series empty after download for "
                  f"{self.start} → {self.end} — NOT caching (transient "
                  f"yfinance failure; retry on next draw rather than "
                  f"poisoning the cache)")
            return
        payload = {"_meta": {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "tickers": list(self.prices.keys()),
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }}
        payload.update(self.prices)
        out_path = self.cache_path
        # Atomic write — `path.write_text` is NOT atomic: a process kill (OOM
        # / SIGKILL) mid-write leaves a truncated/torn JSON file. The next
        # cache load then fails `json.loads`, falls through to the download
        # path, and silently re-pays the (hundreds of MB, dozens of tickers)
        # yfinance refetch on every subsequent run for this window. Worse,
        # the legacy-cache fallback path (`prices.json`) would *also* be
        # consulted on torn-file failure and could accept a stale cross-
        # window payload (the `_meta` mismatch is the only guard there).
        # Mirrors the atomic-write idiom already used by `train_scorer`
        # (scorer.pkl.tmp), the outcomes-file trim, `_persist_volume_cache_for_window`,
        # and the validation persister — all of which document the same
        # class of "process kill mid-write would corrupt the artifact"
        # failure. The CACHE_DIR is the same filesystem as the destination
        # so `Path.replace` is genuinely atomic.
        tmp = out_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload))
        tmp.replace(out_path)
        print(f"[price_cache] saved → {out_path} "
              f"({len(self.trading_days)} trading days)")

    def _build_trading_days(self) -> None:
        spy = self.prices.get("SPY") or {}
        if not spy:
            # Pick the DENSEST non-empty series as the calendar proxy. The
            # prior "first non-empty" fallback could land on a thin/foreign
            # ticker (LNOK / a sparse ADR) and produce a sparse trading_days
            # calendar that silently skipped real NYSE days for the WHOLE
            # backtest — every sampled decision day, the SL/TP scan, and the
            # equity curve all run off this calendar. Density (len of close
            # map) is a safe proxy: SPY-like ETFs (QQQ, NVDA, etc.) all have
            # near-full NYSE coverage; thin names have much shorter series.
            best_n = 0
            for t, series in self.prices.items():
                if series and len(series) > best_n:
                    spy = series
                    best_n = len(series)
        days = sorted(date.fromisoformat(d) for d in spy.keys()
                      if self.start <= date.fromisoformat(d) <= self.end)
        self.trading_days = days

    def price_on(self, ticker: str, d: date) -> float | None:
        """Close on `d` if available; else most recent prior close.

        0/negative cached values (legacy caches built before the storage-side
        filter landed) are treated as missing and walked back over — the same
        defensive contract `market.get_price` already follows. The walk-back
        therefore skips poisoned ticks rather than returning a $0 mark.
        """
        series = self.prices.get(ticker)
        if not series:
            return None
        iso = d.isoformat()
        v = series.get(iso)
        if v is not None and v > 0:
            return v
        # walk back up to 7 days
        for delta in range(1, 8):
            prior = (d - timedelta(days=delta)).isoformat()
            v = series.get(prior)
            if v is not None and v > 0:
                return v
        return None

    def resolved_close_date(self, ticker: str, d: date) -> date | None:
        """The actual date `price_on(ticker, d)` resolves to (after up-to-7-day
        walk-back), or None if no close was found in that window.

        Honesty helper for outcome computation. ``price_on`` returns a float but
        hides whether the value came from the requested date or a walk-back
        fallback — so an outcome-side caller cannot detect when two ``price_on``
        lookups for sim_d / end_d both walked back to the SAME prior close
        (e.g. a ticker whose last trade was before sim_d on a thin/foreign
        calendar). When that happens ``returns_pct`` is 0.0 by construction —
        a fabricated flat outcome that poisons the DecisionScorer training set
        and looks indistinguishable from a real flat 5-day window. This method
        exposes the resolution so callers can refuse a collision instead.
        Pure read; uses the same 7-day window as ``price_on`` so semantics stay
        in lockstep (a change to ``price_on``'s window MUST update this too) —
        same 0/negative-as-missing filter so the resolved-date and the
        returned price agree on which day was usable.
        """
        series = self.prices.get(ticker)
        if not series:
            return None
        v = series.get(d.isoformat())
        if v is not None and v > 0:
            return d
        for delta in range(1, 8):
            prior_d = d - timedelta(days=delta)
            v = series.get(prior_d.isoformat())
            if v is not None and v > 0:
                return prior_d
        return None

    def returns_pct(self, ticker: str, start_d: date, end_d: date) -> float:
        s = self.price_on(ticker, start_d)
        e = self.price_on(ticker, end_d)
        if not s or not e:
            return 0.0
        return (e - s) / s * 100


# ─────────────────────────── technical indicators ───────────────────────────

# Volume series cache: keyed by (ticker, start_iso, end_iso) so different
# backtest windows don't collide. Persisted per-window on disk too.
# A bare ticker key (as used by the original module global) would silently
# return stale data when reused across windows — vol_ratio would compute
# against an unrelated window's history.
_VOLUME_CACHE: dict[tuple[str, str, str], dict[str, float]] = {}
_VOLUME_CACHE_LOCK = threading.Lock()
# Serializes the tmp open→write→replace in `_persist_volume_cache_for_window`
# so two backtest threads that both fetched a fresh volume series in the same
# window can't concurrently `open(..., 'w')` the SAME `.json.tmp` (O_TRUNC,
# interleaved writes → torn JSON under canonical via `.replace`). The cache
# snapshot itself runs under `_VOLUME_CACHE_LOCK` above; this is a SEPARATE
# (write-only) lock so a cache reader/writer is never blocked by an in-flight
# disk write. See `_persist_volume_cache_for_window` for the full rationale.
_VOLUME_PERSIST_LOCK = threading.Lock()
# Track which (start, end) windows we've already loaded from disk into memory.
# Uses an OrderedDict to preserve insertion order so we can evict the oldest
# window when bounded growth hits the cap. The continuous loop picks a fresh
# random window every cycle, so an unbounded set silently leaked tens of MB
# of volume series across a day of running (~30 tickers × ~250 daily volumes
# × 8 bytes ≈ 60 KB / window; ~144 cycles / 24 h ≈ 8.6 MB / day; ~60 MB / week).
# Cap at the most recent N windows — well above any realistic in-flight set
# but bounded so the cache cannot dominate process RSS.
from collections import OrderedDict as _OrderedDict
_VOLUME_CACHE_DISK_LOADED: "_OrderedDict[tuple[str, str], bool]" = _OrderedDict()
_VOLUME_CACHE_MAX_WINDOWS = 16


def _volume_cache_path(start: date, end: date) -> Path:
    return CACHE_DIR / f"volumes_{start.isoformat()}_{end.isoformat()}.json"


def _evict_oldest_volume_windows_locked() -> int:
    """Evict the oldest cached (start, end) windows when the in-memory map
    exceeds the configured cap. Returns count of windows evicted.

    Caller MUST hold `_VOLUME_CACHE_LOCK`. The eviction drops every per-ticker
    entry whose (start_iso, end_iso) matches the evicted window so memory is
    actually reclaimed (not just the bookkeeping set). On-disk caches remain
    untouched — the next visit to that window pays one disk read to refill.
    """
    evicted = 0
    while len(_VOLUME_CACHE_DISK_LOADED) > _VOLUME_CACHE_MAX_WINDOWS:
        (old_start, old_end), _ = _VOLUME_CACHE_DISK_LOADED.popitem(last=False)
        stale_keys = [k for k in _VOLUME_CACHE
                      if k[1] == old_start and k[2] == old_end]
        for k in stale_keys:
            _VOLUME_CACHE.pop(k, None)
        evicted += 1
    return evicted


def _load_volume_cache_for_window(start: date, end: date) -> None:
    """Load the on-disk per-window volume cache into memory. Idempotent.

    No legacy fallback: an unrelated window's series silently seeds the new
    cache key, which then short-circuits the network refetch — `vol_ratio`
    then computes against irrelevant dates and returns None. Better to start
    empty and pay one yfinance fetch.
    """
    key = (start.isoformat(), end.isoformat())
    with _VOLUME_CACHE_LOCK:
        if key in _VOLUME_CACHE_DISK_LOADED:
            # Touch on access so an actively-used window doesn't get evicted
            # under bounded-LRU semantics.
            _VOLUME_CACHE_DISK_LOADED.move_to_end(key)
            return
        path = _volume_cache_path(start, end)
        loaded: dict[str, dict[str, float]] = {}
        if path.exists():
            try:
                raw = json.loads(path.read_text())
                # Type guard. `json.loads` raises only on syntax errors —
                # a syntactically valid but type-wrong payload (e.g. a list
                # `[1,2,3]` written by some external tool, or a truncated dict
                # that happened to land on a closed bracket) returned a
                # non-dict here. The next line then crashed on `.items()`,
                # raising AttributeError out through `_load_volume_cache_for_window`
                # → `_ensure_volume_for` → `_compute_technical_indicators`,
                # killing the run thread mid-cycle. Mirror the GDELT /
                # AlphaVantage cache guards: narrow to a dict of dicts so
                # a single bad nested entry drops just that ticker rather
                # than the whole window's volume series.
                if isinstance(raw, dict):
                    loaded = {tk: s for tk, s in raw.items()
                              if isinstance(s, dict)}
            except Exception:
                loaded = {}
        for ticker, series in loaded.items():
            _VOLUME_CACHE[(ticker, key[0], key[1])] = series
        _VOLUME_CACHE_DISK_LOADED[key] = True
        _evict_oldest_volume_windows_locked()


def _persist_volume_cache_for_window(start: date, end: date) -> None:
    key = (start.isoformat(), end.isoformat())
    try:
        path = _volume_cache_path(start, end)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Snapshot the shared cache UNDER the lock. This runs in a backtest
        # run thread (up to RUNS_PER_CYCLE in parallel); other threads call
        # `_VOLUME_CACHE[...] = series` concurrently. Iterating the live dict
        # here raced with those writes and raised
        # `RuntimeError: dictionary changed size during iteration` — caught by
        # the except below, so the cache silently never persisted under
        # parallel runs (every run re-fetched volumes from yfinance). Copy
        # under the lock, then do file IO outside it (don't hold the lock
        # across disk writes).
        with _VOLUME_CACHE_LOCK:
            flat = {ticker: series
                    for (ticker, s, e), series in _VOLUME_CACHE.items()
                    if (s, e) == key}
        # Atomic write — `path.write_text` is NOT atomic: a process kill (OOM
        # / SIGKILL) mid-write leaves a truncated/torn JSON file. The next
        # `_load_volume_cache_for_window` then fails `json.loads`, falls back
        # to an empty dict, and the bookkeeping marks the window "loaded";
        # subsequent vol_ratio computations re-fetch from yfinance on every
        # single decision for the whole window. Worse, a CONCURRENT loader
        # (the disk-load helper in another thread) can read a partially-
        # written file. Mirrors the atomic-write idiom already used by
        # `train_scorer` (scorer.pkl.tmp), the outcomes-file trim, and the
        # validation persister — all of which document the same class of
        # "a process kill mid-write would corrupt the artifact" failure.
        #
        # Serialize the tmp open→write→replace under a separate persist
        # lock. Two run threads that each fetched a volume series in the
        # same window both reach this persist concurrently; with the
        # shared `".json.tmp"` filename they would `open(..., 'w')`
        # (O_TRUNC) the SAME file in parallel and their writes can
        # interleave at the OS level → a torn JSON tmp could land under
        # `path` via `replace`. The serialization is a no-op in the common
        # single-window case (the snapshot above happened under the
        # cache lock, releasing it before this short write); only
        # concurrent persists for the SAME window contend, and they
        # write the same snapshot anyway so last-writer-wins is correct.
        tmp = path.with_suffix(".json.tmp")
        with _VOLUME_PERSIST_LOCK:
            tmp.write_text(json.dumps(flat))
            tmp.replace(path)
    except Exception as e:
        print(f"[volume_cache] persist failed: {e}")


def _ensure_volume_for(ticker: str, start: date, end: date) -> dict[str, float]:
    """Lazily fetch a volume series for `ticker` covering [start, end]. Cached on disk."""
    _load_volume_cache_for_window(start, end)
    cache_key = (ticker, start.isoformat(), end.isoformat())
    # Use `in` rather than truthiness: a previously-failed ticker is stored as
    # {} which is falsy, so a truthiness check would retry the network call on
    # every invocation. Across many decisions × runs this added measurable cost.
    if cache_key in _VOLUME_CACHE:
        return _VOLUME_CACHE[cache_key]
    try:
        end_pad = (end + timedelta(days=2)).isoformat()
        hist = yf.Ticker(ticker).history(start=start.isoformat(),
                                         end=end_pad, auto_adjust=False)
        series: dict[str, float] = {}
        if hist is not None and not hist.empty:
            for ts, row in hist.iterrows():
                vol = row.get("Volume")
                if vol is None or vol != vol:
                    continue
                series[ts.date().isoformat()] = float(vol)
        with _VOLUME_CACHE_LOCK:
            _VOLUME_CACHE[cache_key] = series
        _persist_volume_cache_for_window(start, end)
        return series
    except Exception as e:
        print(f"[volume_cache] {ticker} fetch failed: {e}")
        with _VOLUME_CACHE_LOCK:
            _VOLUME_CACHE[cache_key] = {}
        return {}


def _series_up_to(prices: "PriceCache", ticker: str, sim_date: date,
                  max_points: int = 260) -> list[tuple[date, float]]:
    """Return (date, close) tuples for `ticker` <= sim_date, oldest first, capped at max_points."""
    series = prices.prices.get(ticker) or {}
    if not series:
        return []
    iso = sim_date.isoformat()
    pairs = [(date.fromisoformat(d), v) for d, v in series.items() if d <= iso]
    pairs.sort(key=lambda x: x[0])
    return pairs[-max_points:]


def _ema(values: list[float], period: int) -> list[float]:
    if len(values) < period:
        return []
    k = 2.0 / (period + 1)
    out: list[float] = []
    seed = sum(values[:period]) / period
    out.append(seed)
    for v in values[period:]:
        out.append(v * k + out[-1] * (1 - k))
    return out


def _rsi(closes: list[float], period: int = 14) -> float | None:
    if len(closes) <= period:
        return None
    # Reject non-finite closes (NaN / ±Inf from a poisoned price cache or a
    # bad yfinance row). The Wilder smoothing chain below propagates NaN
    # silently — every downstream comparison (`avg_l == 0`, `100/(1+rs)`)
    # collapses to NaN, which then passes _ml_decide's `isinstance(rsi,
    # (int, float))` guard and feeds the negative branch (`adj -= 0.5`
    # because `NaN > 0` is False). Returning None instead skips the quant
    # adjustment honestly so a single poisoned close doesn't silently
    # penalise every name. Mirrors the discipline `_macd` already follows.
    for c in closes:
        if c is None or not math.isfinite(c):
            return None
    gains, losses = 0.0, 0.0
    # initial averages over first `period` deltas
    for i in range(1, period + 1):
        diff = closes[i] - closes[i - 1]
        if diff >= 0:
            gains += diff
        else:
            losses -= diff
    avg_g = gains / period
    avg_l = losses / period
    # Wilder smoothing for the rest
    for i in range(period + 1, len(closes)):
        diff = closes[i] - closes[i - 1]
        g = diff if diff > 0 else 0.0
        l = -diff if diff < 0 else 0.0
        avg_g = (avg_g * (period - 1) + g) / period
        avg_l = (avg_l * (period - 1) + l) / period
    if avg_l == 0:
        # Strict all-up returns 100; a perfectly FLAT series (no gain AND no
        # loss across `period` deltas) returned 100.0 too — a spurious
        # "severely overbought" signal that fed a -1.5 conviction penalty
        # in `_ml_decide` for any flat name. RSI is undefined at zero
        # variance; the textbook neutral reading is 50.
        return 100.0 if avg_g > 0 else 50.0
    rs = avg_g / avg_l
    return 100.0 - (100.0 / (1.0 + rs))


def _macd(closes: list[float]) -> tuple[str, float, float] | None:
    """Return (label, macd, signal). label is 'bullish'/'bearish'/'flat'.

    Non-finite closes (NaN/±Inf from a poisoned price cache) yield None
    rather than propagating silently — same defensive contract as `_rsi`.
    The label uses an epsilon tolerance so a steady-state linear trend
    (where the macd line and signal line converge to the same value and
    only floating-point roundoff separates them) is reported as ``"flat"``
    instead of fabricating a bullish/bearish verdict from EMA
    accumulation order. The threshold is scaled by the magnitudes
    involved so real crossovers — even small ones — stay
    classified correctly.
    """
    if len(closes) < 35:
        return None
    for c in closes:
        if c is None or not math.isfinite(c):
            return None
    ema12 = _ema(closes, 12)
    ema26 = _ema(closes, 26)
    if not ema12 or not ema26:
        return None
    # align: ema26 starts 14 points later than ema12 (offset of 26-12=14)
    offset = len(ema12) - len(ema26)
    macd_line = [ema12[i + offset] - ema26[i] for i in range(len(ema26))]
    if len(macd_line) < 9:
        return None
    signal_line = _ema(macd_line, 9)
    if not signal_line:
        return None
    m = macd_line[-1]
    s = signal_line[-1]
    # Epsilon-tolerant comparison so a steady-state linear trend (m ≈ s
    # at machine precision) reports "flat" rather than flapping to
    # bullish/bearish from EMA accumulation roundoff. Tolerance scales
    # with the magnitudes involved so real crossovers (m - s well above
    # the noise floor) are unaffected.
    tol = 1e-9 * max(abs(m), abs(s), 1.0)
    diff = m - s
    if diff > tol:
        label = "bullish"
    elif diff < -tol:
        label = "bearish"
    else:
        label = "flat"
    return (label, m, s)


def _compute_technical_indicators(ticker: str, sim_date: date,
                                  prices: "PriceCache") -> dict | None:
    """RSI/MACD/MA crossover/volume ratio/52w proximity computed from cached closes.

    Returns None if there isn't enough history for the ticker at sim_date.

    Returns both uppercase keys (RSI, MACD label, MA_cross) for the prompt-building
    path and lowercase numeric keys (rsi, macd_signal, bb_position, mom_5d, mom_20d,
    wk52_pos) that _ml_decide and the DecisionScorer consume. Prior versions only
    emitted uppercase, so _ml_decide silently saw null momentum/BB and a string MACD
    label that failed isinstance checks — most quant features no-op'd.
    """
    pairs = _series_up_to(prices, ticker, sim_date, max_points=300)
    if len(pairs) < 60:
        return None
    closes = [p[1] for p in pairs]
    last = closes[-1]

    rsi = _rsi(closes, 14)

    macd_res = _macd(closes)
    macd_label = macd_res[0] if macd_res else None

    # MACD signal-line value (numeric) — used by _ml_decide and the scorer.
    # _macd() already computes and returns it as element [2]; reuse it instead
    # of recomputing the full EMA chain.
    macd_signal_val: float | None = macd_res[2] if macd_res else None

    # Enhanced MACD features (12/26/9 + EMA200 filter — mirrors strategy.py's
    # live get_quant_signals_live so the live trader and the DecisionScorer
    # see the same shape). Recomputing the full EMA chain here costs ~0; the
    # alternative would be to plumb the values out of _macd(), but _macd()
    # already collapses to a 3-tuple and bolting on 3 more values risks
    # breaking external consumers.
    macd_hist: float | None = None
    hist_cross_up: bool = False
    macd_below_zero_cross: bool = False
    try:
        if len(closes) >= 35:
            e12 = _ema(closes, 12)
            e26 = _ema(closes, 26)
            if e12 and e26:
                offset = len(e12) - len(e26)
                macd_line = [e12[i + offset] - e26[i] for i in range(len(e26))]
                if len(macd_line) >= 9:
                    sig = _ema(macd_line, 9)
                    if sig:
                        sig_len = len(sig)
                        ml_aligned = macd_line[-sig_len:]
                        macd_hist = ml_aligned[-1] - sig[-1]
                        if len(ml_aligned) >= 2 and len(sig) >= 2:
                            hist_prev = ml_aligned[-2] - sig[-2]
                            hist_curr = ml_aligned[-1] - sig[-1]
                            hist_cross_up = bool(
                                hist_prev < 0 and hist_curr > 0
                            )
                            macd_below_zero_cross = bool(
                                hist_cross_up and ml_aligned[-1] < 0
                            )
    except Exception:
        macd_hist = None
        hist_cross_up = False
        macd_below_zero_cross = False

    # 200-day EMA filter — confirms long-term trend; preferred long entries
    # require ema200_above=True (the textbook MACD-strategy filter).
    ema200_above: bool | None = None
    try:
        if len(closes) >= 200:
            e200 = _ema(closes, 200)
            if e200:
                ema200_above = bool(closes[-1] > e200[-1])
    except Exception:
        ema200_above = None

    ma_cross = None
    if len(closes) >= 200:
        ma50 = sum(closes[-50:]) / 50
        ma200 = sum(closes[-200:]) / 200
        ma_cross = "golden" if ma50 > ma200 else "death"
    elif len(closes) >= 50:
        ma50 = sum(closes[-50:]) / 50
        ma_cross = "above50" if last > ma50 else "below50"

    # Bollinger Band position over 20 days (clamped to ±2)
    bb_position: float | None = None
    if len(closes) >= 20:
        window20 = closes[-20:]
        sma20 = sum(window20) / 20
        sd20 = _stdev(window20)
        if sd20 > 0:
            raw = (last - sma20) / (2 * sd20)
            bb_position = max(-2.0, min(2.0, raw))

    # 5- and 20-day momentum %
    mom_5d: float | None = None
    if len(closes) >= 6 and closes[-6] > 0:
        mom_5d = (last - closes[-6]) / closes[-6] * 100
    mom_20d: float | None = None
    if len(closes) >= 21 and closes[-21] > 0:
        mom_20d = (last - closes[-21]) / closes[-21] * 100

    hi_52 = max(closes[-252:]) if len(closes) >= 252 else max(closes)
    lo_52 = min(closes[-252:]) if len(closes) >= 252 else min(closes)
    pct_from_52h = (last - hi_52) / hi_52 * 100 if hi_52 else 0.0
    pct_from_52l = (last - lo_52) / lo_52 * 100 if lo_52 else 0.0

    # 52-week position 0 (low) .. 1 (high)
    wk52_pos: float | None = None
    if hi_52 > lo_52:
        wk52_pos = (last - lo_52) / (hi_52 - lo_52)

    vol_ratio: float | None = None
    try:
        # volumes cover [start, end] for the ticker — use the PriceCache's
        # window (offset back 400d for warm-up) so different backtest windows
        # don't share a single cached series.
        vols = _ensure_volume_for(
            ticker,
            prices.start - timedelta(days=400),
            prices.end,
        )
        if vols:
            iso = sim_date.isoformat()
            # find sim_date volume + last 20 trading-day window
            vdates = sorted(d for d in vols.keys() if d <= iso)
            if len(vdates) >= 21:
                today_v = vols[vdates[-1]]
                prior20 = [vols[d] for d in vdates[-21:-1]]
                avg20 = sum(prior20) / len(prior20)
                if avg20 > 0:
                    vol_ratio = today_v / avg20
    except Exception:
        vol_ratio = None

    return {
        # Legacy uppercase keys — _build_prompt and any external readers rely on these.
        "RSI": round(rsi, 1) if rsi is not None else None,
        "MACD": macd_label,
        "MA_cross": ma_cross,
        "pct_from_52h": round(pct_from_52h, 1),
        "pct_from_52l": round(pct_from_52l, 1),
        # Lowercase numeric keys consumed by _ml_decide, _compute_decision_outcomes,
        # _format_quant_signals_block, and DecisionScorer features.
        "rsi": round(rsi, 2) if rsi is not None else None,
        "macd_signal": round(macd_signal_val, 4) if macd_signal_val is not None else None,
        "bb_position": round(bb_position, 2) if bb_position is not None else None,
        "mom_5d": round(mom_5d, 2) if mom_5d is not None else None,
        "mom_20d": round(mom_20d, 2) if mom_20d is not None else None,
        "wk52_pos": round(wk52_pos, 2) if wk52_pos is not None else None,
        "vol_ratio": round(vol_ratio, 2) if vol_ratio is not None else None,
        # Enhanced MACD signals (mirrors strategy.py get_quant_signals_live).
        # Consumed by DecisionScorer.build_features so the scorer learns the
        # MACD-zero-cross + EMA200-filter setup.
        "macd_hist": round(macd_hist, 4) if macd_hist is not None else None,
        "hist_cross_up": hist_cross_up,
        "macd_below_zero_cross": macd_below_zero_cross,
        "ema200_above": ema200_above,
    }


def _get_quant_signals(sim_date: date, tickers: list[str],
                       prices: "PriceCache") -> dict[str, dict]:
    """Compute technical indicators for each ticker at sim_date.

    Returns a dict {ticker: {RSI, MACD, MA_cross, vol_ratio, pct_from_52h, pct_from_52l}}.
    Tickers with insufficient history are omitted."""
    out: dict[str, dict] = {}
    for t in tickers:
        try:
            ind = _compute_technical_indicators(t, sim_date, prices)
            if ind is not None:
                out[t] = ind
        except Exception as e:
            print(f"[quant] {t} indicator compute failed: {e}")
    return out


def _stdev(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return (sum((v - mean) ** 2 for v in values) / n) ** 0.5


def _format_quant_signals_block(quant: dict[str, dict]) -> str:
    """Render the per-spec QUANTITATIVE SIGNALS table for the Claude prompt."""
    if not quant:
        return ""
    header = ("=== QUANTITATIVE SIGNALS ===\n"
              "Ticker | RSI | MACD | BB-pos | Mom-5d% | Mom-20d% | Vol-ratio | 52wk-pos")
    lines = [header]
    for tk in sorted(quant.keys()):
        q = quant[tk]
        def _f(key, default="N/A"):
            v = q.get(key)
            return default if v is None else v
        rsi = _f("rsi")
        macd = _f("macd_signal")
        bbp = _f("bb_position")
        m5 = q.get("mom_5d")
        m20 = q.get("mom_20d")
        vr = q.get("vol_ratio")
        w52 = _f("wk52_pos")
        m5_s = f"{m5:+.1f}%" if isinstance(m5, (int, float)) else "N/A"
        m20_s = f"{m20:+.1f}%" if isinstance(m20, (int, float)) else "N/A"
        vr_s = f"{vr:.2f}x" if isinstance(vr, (int, float)) else "N/A"
        lines.append(f"{tk:<6} | {rsi} | {macd} | {bbp} | {m5_s} | {m20_s} | {vr_s} | {w52}")
    return "\n".join(lines)


def _market_regime(sim_date: date, prices: "PriceCache") -> str:
    """Bull/bear/sideways via SPY 50/200 MA + slope."""
    pairs = _series_up_to(prices, "SPY", sim_date, max_points=260)
    if len(pairs) < 200:
        return "unknown"
    closes = [p[1] for p in pairs]
    last = closes[-1]
    ma50 = sum(closes[-50:]) / 50
    ma200 = sum(closes[-200:]) / 200
    if last > ma50 > ma200:
        return "bull"
    if last < ma50 < ma200:
        return "bear"
    return "sideways"


def _sector_rotation(sim_date: date, prices: "PriceCache",
                     lookback_days: int = 21) -> list[tuple[str, float]]:
    """Trailing ~1 month total return for sector ETFs, sorted descending."""
    sectors = ["XLK", "XLE", "XLF", "XLV", "XLI"]
    results: list[tuple[str, float]] = []
    for s in sectors:
        pairs = _series_up_to(prices, s, sim_date, max_points=lookback_days + 5)
        if len(pairs) < 2:
            continue
        start = pairs[0][1]
        end = pairs[-1][1]
        if start <= 0:
            continue
        results.append((s, (end - start) / start * 100))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def _vix_level(sim_date: date, prices: "PriceCache") -> float | None:
    return prices.price_on("^VIX", sim_date)


# ─────────────────────────── GDELT fetcher ───────────────────────────

class GDELTFetcher:
    """Cached GDELT fetcher using the gdeltdoc library (alex9smith/gdelt-doc-api).

    Thread-safe: a class-level lock serializes outbound GDELT requests so 10
    parallel run threads don't all hit the 5s rate limit simultaneously."""

    def __init__(self):
        GDELT_CACHE.mkdir(parents=True, exist_ok=True)
        from gdeltdoc import GdeltDoc
        self._client = GdeltDoc()
        self._request_lock = threading.Lock()
        self._last_request_ts = 0.0

    def _cache_key(self, d: date, keywords: str) -> Path:
        slug = hashlib.md5(keywords.encode()).hexdigest()[:8]
        return GDELT_CACHE / f"{d.isoformat()}_{slug}.json"

    def fetch(self, d: date, keywords: str) -> list[dict]:
        path = self._cache_key(d, keywords)
        if path.exists():
            try:
                cached = json.loads(path.read_text())
                # Type guard. A corrupt cache file (truncated mid-write,
                # external editor saving a dict / number / string instead of
                # the expected list) silently passed through here as the
                # function return, then crashed downstream in `_fetch_signals`
                # at `for a in articles: a.get(...)`. Iterating a string yields
                # single-char strings whose `.get` raises AttributeError; a
                # number isn't iterable; a dict iterates its keys (strings).
                # Any of those kills the run thread mid-cycle. Mirror the
                # `_load_volume_cache_for_window` type guard discipline:
                # narrow to a list of dicts (drop non-dict entries so a single
                # bad row doesn't poison the whole day's signals).
                if isinstance(cached, list):
                    return [a for a in cached if isinstance(a, dict)]
            except Exception:
                pass

        from gdeltdoc import Filters
        from gdeltdoc.errors import RateLimitError
        start_str = d.strftime("%Y-%m-%d")
        end_str = (d + timedelta(days=1)).strftime("%Y-%m-%d")

        articles: list[dict] = []
        success = False
        permanent = False
        for attempt in range(3):
            err: str | None = None
            with self._request_lock:
                elapsed = time.time() - self._last_request_ts
                if elapsed < GDELT_RATE_LIMIT_S:
                    time.sleep(GDELT_RATE_LIMIT_S - elapsed)
                try:
                    f = Filters(keyword=keywords, start_date=start_str, end_date=end_str)
                    df = self._client.article_search(f)
                    self._last_request_ts = time.time()
                    if df is not None and not df.empty:
                        keep = [c for c in ["title", "url", "domain", "seendate"]
                                if c in df.columns]
                        articles = df[keep].rename(columns={"domain": "source"}).to_dict("records")
                    success = True
                except RateLimitError:
                    self._last_request_ts = time.time()
                    err = "rate-limited"
                except Exception as e:
                    self._last_request_ts = time.time()
                    err = f"{type(e).__name__}: {e}"
                    # GDELT DOC 2.0 only indexes ~2017-onward. A pre-coverage
                    # date raises a deterministic ValueError ("The query was
                    # not valid … Invalid query start date") that can NEVER
                    # succeed on retry. The continuous loop picks windows back
                    # to 1993, so without this short-circuit every such
                    # (date,keyword) burned 20+40+60s of backoff AND was never
                    # cached, so it was re-attempted every cycle for hours
                    # (see continuous.log). Treat it as permanent: stop
                    # retrying and negative-cache an empty result so the
                    # warm-cache exists()-filter and the tier-3 disk lookup
                    # skip it forever after.
                    _m = str(e).lower()
                    if "not valid" in _m or "invalid query" in _m:
                        permanent = True
            if success or permanent:
                break
            backoff = GDELT_RETRY_BACKOFF_S * (attempt + 1)
            print(f"[gdelt] {err} {d} {keywords[:30]!r} "
                  f"attempt {attempt+1}/3 — sleeping {backoff:.0f}s")
            time.sleep(backoff)

        # Cache on success (a legitimately-empty result for a covered date is
        # the correct answer and SHOULD be cached) OR on a permanent coverage
        # error (an empty list IS the correct, immutable answer for a date
        # GDELT will never index). NEVER cache a transient failure — that
        # would poison a temporarily rate-limited/disconnected date for the
        # rest of the loop's lifetime.
        if success or permanent:
            # Atomic write — a kill mid-`path.write_text` left a torn GDELT
            # cache file that the next read silently re-fetched (5s rate
            # limit), permanently degrading the warm cache for that
            # (date, keyword). Same atomic idiom as the per-window
            # PriceCache and the volume-cache persister. Never raises.
            _atomic_write_json(path, articles)
        if permanent:
            print(f"[gdelt] permanent: {d} outside GDELT coverage — "
                  f"cached empty, no retry")
        return articles


# ─────────────────────────── Alpha Vantage news fetcher ───────────────────────────

AV_CACHE_DIR = CACHE_DIR / "alphavantage"
AV_QUOTA_PATH = CACHE_DIR / "av_quota.json"
AV_MAX_DAILY = 22  # stay under 25/day limit with margin


def _atomic_write_json(path: Path, payload) -> None:
    """Atomic JSON write via tmp file + ``Path.replace``.

    Mirrors the atomic-write idiom already used by ``train_scorer``
    (scorer.pkl.tmp), the per-window PriceCache write, the volume-cache
    persister, the outcomes-file trim, and the validation persister — every
    one of which documents the same class of "process kill mid-write would
    corrupt the artifact" failure (OOM/SIGKILL leaves a torn JSON file). The
    GDELT cache, the Alpha Vantage cache, and the AV cross-restart quota
    tracker were the last JSON writers in this file still using bare
    ``path.write_text(json.dumps(...))``: a kill mid-write left a corrupt
    file, and on the next load the per-file ``except Exception: pass`` guard
    silently treated the corruption as "no cache" — re-fetching with quota /
    rate-limit burn (the AV quota guarantee is *cross-restart* per the
    AV_QUOTA_PATH comment, so a torn quota file silently reset to 0 calls).
    The tmp file lives in the same directory so ``Path.replace`` is genuinely
    atomic on POSIX. Never raises — any IO/serialization failure degrades
    the same way the legacy bare-write would have (the call site already
    runs inside best-effort try/except blocks, e.g. ``GDELTFetcher.fetch``,
    ``AlphaVantageNewsFetcher.fetch``, ``_inc_quota``).
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload))
        tmp.replace(path)
    except Exception as e:
        print(f"[atomic_json] write to {path} failed: {e}")


class AlphaVantageNewsFetcher:
    """Disk-cached Alpha Vantage NEWS_SENTIMENT fetcher.

    Extremely conservative: max 22 calls/day tracked across restarts, skip
    when quota is exhausted. All results persisted to disk so backtest reruns
    are free. Gracefully disabled when ALPHA_VANTAGE_KEY is unset.
    """
    _lock = threading.Lock()

    def __init__(self):
        AV_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._key = os.environ.get("ALPHA_VANTAGE_KEY", "").strip()

    def _quota(self) -> dict:
        try:
            if AV_QUOTA_PATH.exists():
                q = json.loads(AV_QUOTA_PATH.read_text())
                if q.get("date") == date.today().isoformat():
                    return q
        except Exception:
            pass
        return {"date": date.today().isoformat(), "calls": 0}

    def _inc_quota(self):
        with self._lock:
            q = self._quota()
            q["calls"] += 1
            # Atomic write — the AV_QUOTA_PATH comment explicitly promises
            # cross-restart quota tracking, but the bare `write_text` was
            # NOT atomic: a kill mid-write left a torn JSON file, the next
            # `_quota` read raised, the `except Exception: pass` fell back
            # to a fresh `{"date": today, "calls": 0}` — silently resetting
            # the AV quota counter to 0 and reopening the cap (25/day) for
            # arbitrary refetches. Same atomic idiom as `train_scorer` /
            # PriceCache / volume cache / GDELT cache.
            _atomic_write_json(AV_QUOTA_PATH, q)

    def _cache_path(self, ticker: str, d: date) -> Path:
        return AV_CACHE_DIR / f"{d.isoformat()}_{ticker}.json"

    def fetch(self, tickers: list[str], d: date) -> list[dict]:
        if not self._key:
            return []
        # AV NEWS_SENTIMENT without time_from/time_to returns the latest news for
        # the ticker, not news as-of `d`. For historical backtest dates this is
        # forward-leakage (training on news that didn't exist yet). Constrain the
        # query window to a tight band around `d`.
        articles: list[dict] = []
        time_from = d.strftime("%Y%m%dT0000")
        time_to = (d + timedelta(days=1)).strftime("%Y%m%dT0000")
        for tk in tickers:
            path = self._cache_path(tk, d)
            if path.exists():
                try:
                    cached = json.loads(path.read_text())
                    # Type guard mirrors `GDELTFetcher.fetch` / the volume
                    # cache loader: a corrupt AV cache file (truncated
                    # mid-write, external editor saving a dict / number
                    # instead of a list) silently extended `articles` with
                    # whatever the raw value iterated to (dict→keys as
                    # strings, number→TypeError) and crashed downstream at
                    # `a.get("url")` in `_fetch_signals`. Narrow to a list of
                    # dicts so a single bad row drops just that row, not the
                    # whole ticker's news for the day.
                    if isinstance(cached, list):
                        articles.extend(a for a in cached if isinstance(a, dict))
                    continue
                except Exception:
                    pass
            with self._lock:
                q = self._quota()
                if q["calls"] >= AV_MAX_DAILY:
                    continue
            try:
                resp = requests.get(
                    "https://www.alphavantage.co/query",
                    params={"function": "NEWS_SENTIMENT", "tickers": tk,
                            "limit": 50, "time_from": time_from,
                            "time_to": time_to, "apikey": self._key},
                    timeout=12,
                )
                data = resp.json()
                feed = data.get("feed", [])
                items = [{"title": a.get("title", ""), "url": a.get("url", ""),
                          "source": a.get("source", "")}
                         for a in feed if a.get("title")]
                # Atomic write — a kill mid-`write_text` left a torn AV cache
                # file; the next read's `except Exception: pass` silently
                # treated it as "no cache" and re-fetched (charging the
                # 22/day quota for a name that should have been free).
                _atomic_write_json(path, items)
                articles.extend(items)
                self._inc_quota()
                time.sleep(1.2)  # AV rate-limit buffer
            except Exception as e:
                print(f"[av_news] {tk} {d}: {e}")
        return articles


# ─────────────────────────── heuristic scorer ───────────────────────────

_TICKER_RE = re.compile(r"\b([A-Z]{2,5})\b")
_NOT_TICKERS = {
    "AI", "AND", "FOR", "THE", "WITH", "FROM", "AFTER", "INTO", "HAVE", "WILL",
    "MAY", "JUNE", "JULY", "AUG", "SEPT", "OCT", "NOV", "DEC", "CEO", "ETF",
    "USA", "USD", "GDP", "CPI", "OPEC", "FED", "FOMC", "PMI", "ISM", "WHO",
    "NEW", "OLD", "ALL", "YES", "ITS", "OUR", "ONE", "TWO",
}


def _extract_tickers(text: str) -> set[str]:
    out = set()
    for m in re.finditer(r"\$([A-Z]{1,5})\b", text or ""):
        out.add(m.group(1))
    for m in _TICKER_RE.finditer(text or ""):
        tok = m.group(1)
        if tok in _NOT_TICKERS:
            continue
        out.add(tok)
    return out


def score_article(article: dict) -> tuple[float, list[str]]:
    """Return (score 0..5, tickers). Pure keyword heuristic."""
    title = (article.get("title") or "")
    body = title.lower()
    pos = sum(1 for p in BUY_PHRASES if p in body)
    neg = sum(1 for p in SELL_PHRASES if p in body)
    tickers = _extract_tickers(title)
    semis_boost = 0.5 if tickers & SEMIS_TICKERS else 0.0
    score = 2.5 + pos * 0.5 - neg * 0.5 + semis_boost
    return max(0.0, min(5.0, score)), sorted(tickers)


# ─────────────────────────── ML/quant decision engine ─────────────────
_BULLISH_WORDS = {
    "beat", "beats", "surge", "surges", "rally", "rallies", "upgrade", "upgraded",
    "record", "breakout", "buy", "outperform", "strong", "growth", "profit",
    "dividend", "bullish", "higher", "raise", "raised", "exceed",
    # Explicit acquisition variants. The legacy ``"acqui"`` stem matched
    # ``acquire/acquires/acquired/acquiring/acquisition/acquisitions`` via
    # ``startswith`` — but the same ``startswith`` rule silently matched
    # ``mission`` against ``miss`` (bearish) and ``cute`` against ``cut``
    # (bearish), poisoning per-article sentiment scoring (verified live:
    # ``"NVDA upgrade mission critical AI"`` scored 0.0 instead of bullish
    # because ``upgrade`` and ``mission→miss`` cancelled out). The
    # ``_article_sentiment`` rewrite below replaces ``startswith`` with a
    # word-boundary regex that allows only a closed safe-suffix set — which
    # cannot match ``acqui + sition``, so the stem must be enumerated.
    "acquire", "acquires", "acquired", "acquiring",
    "acquisition", "acquisitions",
    # Same explicit-variant rationale as the bearish-side ``cutting`` /
    # ``selling`` entries: the regex tail ``ying`` isn't in the safe-suffix
    # set, so ``buy→buying`` and ``rally→rallying`` would silently lose
    # coverage. Enumerating the headline-common forms keeps semantics
    # equivalent to the prior startswith approach for these high-frequency
    # tokens without re-opening the false-positive class for unrelated
    # stems.
    "buying", "rallying",
}
_BEARISH_WORDS = {
    "miss", "misses", "plunge", "plunges", "downgrade", "downgraded", "cut",
    "cuts", "layoff", "layoffs", "loss", "losses", "warning", "shortfall",
    "selloff", "sell", "underperform", "weak", "decline", "declines", "crash",
    "lower", "reduce", "reduced", "concern", "concerns", "risk",
    # Explicit double-consonant verbal nouns / participles whose ending lies
    # outside ``_SENTIMENT_SAFE_SUFFIX_RE``. ``cut→cutting``,
    # ``sell→selling``, ``buy→buying`` would naturally fall to the regex word
    # boundary the safe-suffix rewrite enforces (``\bcut(s|es|ed|ing|...)?\b``
    # rejects ``cutting`` because ``ting`` isn't a recognised inflection).
    # Enumerating the common production variants explicitly preserves
    # coverage on the small handful of headlines that actually appear
    # (``Fed cutting rates``, ``Hedge fund selling``) — without re-opening
    # the ``mission/cute/missile`` false-positive class the regex closes.
    "cutting", "selling",
}

# Safe suffixes allowed to follow a sentiment stem at a word boundary. The
# closed set is the textbook English inflection tail — adding ``-s/-es/-ed/-ing``
# captures the common verb forms (``misses``/``missed``/``missing``),
# ``-er/-ers/-est`` covers comparative/superlative (``stronger``/``strongest``
# / ``buyer``/``buyers``), ``-y/-ies`` covers ``rally/rallies``. Critically,
# this list does NOT include the tails that produced the documented false
# positives: ``"ion"`` (``mission`` against ``miss``), ``"ile"``
# (``missile`` against ``miss``), ``"e"`` (``cute`` against ``cut``), or
# ``"ert"`` (``concert`` would be against ``conc`` if such a stem existed —
# defense in depth). So a future stem added to either word set CANNOT trip
# the same class of bug unless its own true derivatives genuinely use one
# of these endings, in which case the match is legitimate.
_SENTIMENT_SAFE_SUFFIX_RE = r"(?:s|es|ed|ing|er|ers|est|ly|y|ies|d)?"


def _build_sentiment_regex(words: set[str]) -> "re.Pattern[str]":
    """Build one compiled regex that matches any of ``words`` at a word
    boundary, optionally followed by one of the ``_SENTIMENT_SAFE_SUFFIX_RE``
    English-inflection tails.

    Sorts alternatives longest-first because Python's ``re`` is leftmost-FIRST
    (not leftmost-longest): in the alternation ``(beat|beats)`` the engine
    commits to ``beat`` even when the input is ``beats``, then backtracks
    onto the optional suffix to recover the longer match. Longest-first
    alternation makes the recovery unnecessary and the intent obvious.
    """
    sorted_words = sorted(words, key=len, reverse=True)
    return re.compile(
        r"\b(?:" + "|".join(re.escape(w) for w in sorted_words) + r")"
        + _SENTIMENT_SAFE_SUFFIX_RE + r"\b"
    )


_BULLISH_RE = _build_sentiment_regex(_BULLISH_WORDS)
_BEARISH_RE = _build_sentiment_regex(_BEARISH_WORDS)
_WORD_TO_TICKER: dict[str, str] = {
    # Tech / semis — map bullish tech headlines straight to leveraged ETFs
    "nvidia": "NVDA", "amd": "AMD", "apple": "AAPL", "microsoft": "MSFT",
    "amazon": "AMZN", "google": "GOOGL", "alphabet": "GOOGL", "meta": "META",
    "tesla": "TSLA", "intel": "INTC", "micron": "MU",
    # `broadcom` previously mapped to AVGO, which is NOT in WATCHLIST — so the
    # entry was dead (filtered out by the `if tk not in WATCHLIST: continue`
    # guard in `_ml_decide`). Redirect to SOXL, the 3x semis ETF that IS in
    # WATCHLIST, so Broadcom-headline sentiment actually contributes to the
    # semis tracker the same way other semi keywords ("semiconductor", "chip")
    # do. Avoids silently dropping a major semi name's news.
    "broadcom": "SOXL",
    "qualcomm": "QCOM", "spy": "SPY", "qqq": "QQQ",
    "semiconductor": "SOXL", "chip": "SOXL", "chips": "SOXL",
    "nasdaq": "TQQQ",          # nasdaq headline → 3x Nasdaq
    "nasdaq rally": "TQQQ", "tech rally": "TQQQ", "tech surge": "TQQQ",
    "s&p rally": "UPRO", "sp500 rally": "UPRO", "bull market": "UPRO",
    "ai": "TQQQ", "artificial intelligence": "TQQQ",
    "semiconductor surge": "SOXL", "chip rally": "SOXL", "semi rally": "SOXL",
    "nvidia surge": "SOXL", "nvidia rally": "SOXL",
    # Energy
    "exxon": "XOM", "chevron": "CVX", "oil": "USO", "crude": "USO",
    "natural gas": "UNG", "energy": "XLE", "opec": "USO", "petroleum": "USO",
    "lng": "UNG", "shale": "XOM",
    # Financials / banks
    "goldman": "GS", "jpmorgan": "JPM", "jp morgan": "JPM", "bank of america": "BAC",
    "federal reserve": "TLT", "fed rate": "TLT", "interest rate": "TLT",
    "treasury": "TLT", "yields": "TLT", "bonds": "TLT", "banking": "FAS",
    "financials": "XLF",
    # Healthcare / pharma
    "eli lilly": "LLY", "lilly": "LLY", "ozempic": "LLY", "wegovy": "LLY",
    "glp-1": "LLY", "unitedhealth": "UNH", "healthcare": "XLV",
    "pharma": "XLV", "biotech": "LABU", "fda": "LABU",
    # Commodities / macro
    "gold": "GLD", "silver": "SLV", "inflation": "GLD", "copper": "XOM",
    "commodities": "GLD", "precious metals": "GLD",
    # Defense / industrials
    "defense": "DFEN", "military": "DFEN", "lockheed": "DFEN", "boeing": "DFEN",
    "raytheon": "DFEN", "industrial": "XLI",
    # China / global
    "alibaba": "BABA", "china": "BABA", "taiwan": "TSM", "asml": "ASML",
    "novo nordisk": "NVO", "toyota": "TM",
    # Crypto
    "bitcoin": "BTC-USD", "crypto": "COIN", "coinbase": "COIN",
}

# Word-boundary regex patterns for `_WORD_TO_TICKER` lookup. A naive
# `keyword in title_lower` substring match — the old form — false-positively
# mapped short keys to their tickers on irrelevant articles in the SAME
# pattern strategy.py was previously broken in (see `strategy._WORD_TO_TICKER_LIVE_PATTERNS`
# and `tests/test_ml_live_opinion.TestKeywordSubstringFalsePositives`):
#
#   * "ai" → TQQQ matched "rain" / "Spain" / "training" / "captain" / "blockchain"
#   * "gold" → GLD matched "Goldman" (very common in finance headlines)
#   * "intel" → INTC matched "intelligence" (and "artificial intelligence",
#     double-counted with the "ai" → TQQQ map)
#   * "oil" → USO matched "spoiled" / "coil"
#
# Every such false positive silently boosted an unrelated ticker's `ticker_scores`
# weight on every article containing the substring — distorting which ticker
# wins the per-day buy/sell pick AND poisoning the `decision_outcomes.jsonl`
# training corpus that retrains the DecisionScorer (the gate the live trader
# eventually relies on). The fix mirrors the strategy.py side exactly: compile
# `\bkw\b` once at module import and match via `Pattern.search`. Multi-word
# keys ("nasdaq rally", "natural gas") still match because `\b` sits between
# any word/non-word transition — including the space between the two tokens.
_WORD_TO_TICKER_PATTERNS: dict[str, "re.Pattern[str]"] = {
    kw: re.compile(rf"\b{re.escape(kw)}\b") for kw in _WORD_TO_TICKER
}


def _article_sentiment(title: str) -> float:
    """Return -1..+1 based on bullish/bearish keyword count in title.

    Uses word-boundary regex matching against the compiled bullish/bearish
    stem regexes (see ``_build_sentiment_regex``). The prior ``startswith``
    implementation silently flagged ``mission`` as bearish (against ``miss``),
    ``cute`` as bearish (against ``cut``), and ``missile`` as bearish — verified
    live: ``"NVDA upgrade mission critical AI"`` scored 0.0 instead of bullish
    because the upgrade vote was cancelled by ``mission→miss``. Because
    ``_ml_decide`` multiplies per-article ``raw_score * sentiment`` directly
    into the per-ticker score, those false negatives poisoned both the daily
    decision and the ``decision_outcomes.jsonl`` row that retrains the
    DecisionScorer the gate eventually relies on.

    Matches are deduplicated via ``set`` so a title that repeats the same
    bullish word twice (``"beats beats earnings"``) counts as one vote per
    sentiment side — preserving the original set-based dedup semantics.
    """
    if not title:
        return 0.0
    low = title.lower()
    bull = len(set(_BULLISH_RE.findall(low)))
    bear = len(set(_BEARISH_RE.findall(low)))
    total = bull + bear
    if total == 0:
        return 0.0
    return (bull - bear) / total


# ─────────────────────────── decision scorer singleton ───────────────────────────
# Lazy-loaded so import errors or missing files never crash the backtest.

_DECISION_SCORER = None
_DECISION_SCORER_LOCK = threading.Lock()


def _get_decision_scorer():
    global _DECISION_SCORER
    if _DECISION_SCORER is not None:
        return _DECISION_SCORER
    with _DECISION_SCORER_LOCK:
        if _DECISION_SCORER is not None:
            return _DECISION_SCORER
        try:
            from .ml.decision_scorer import DecisionScorer as _DS
            _DECISION_SCORER = _DS()
        except Exception as exc:
            print(f"[decision_scorer] init failed ({exc}) — running without scorer")

            class _Dummy:
                is_trained = False

                def predict(self, **kw):
                    return 0.0

            _DECISION_SCORER = _Dummy()
    return _DECISION_SCORER


# ─────────────────────────── gate kill-switch ───────────────────────────
# Why this exists: the conviction gate (#5) modulates BUY sizing by ×0.6 /
# ×0.85 / ×1.0 / ×1.15 / ×1.3 on the deployed scorer's prediction once
# n_train >= 500. But the deployed scorer's persisted OOS BUY rank-IC is
# ~0 in production (`scorer_skill_log.jsonl` running median ≈ -0.01 over
# trailing cycles; `baseline_compare` reports MLP_WORSE_THAN_TRIVIAL).
# Under that condition the gate's per-arm reallocation is variance with no
# compensating realized edge — `gate_pnl_skill_log` already trends a
# GATE_INEFFECTIVE / GATE_SUBTRACTS_RETURN verdict in OOS. This kill-switch
# reads the same per-cycle ledger and short-circuits the gate's conviction
# modulation when trailing OOS BUY rank-IC is statistically at noise.
#
# Behaviour matrix:
#   - Ledger absent / unreadable / fewer than _GATE_SKILL_MIN_CYCLES rows
#     with parseable `oos_buy_ic`: the kill-switch DEFAULTS to gate-active
#     (preserves invariant #5 semantics for the first few hundred cycles
#     after a fresh start, and never disables the gate due to a transient
#     ledger fault).
#   - Median oos_buy_ic < _GATE_SKILL_IC_TOLERANCE (signed): the modulation
#     block is short-circuited — conviction is left untouched. This covers
#     BOTH near-zero noise (|IC|≈0) AND persistently NEGATIVE (anti-predictive)
#     skill. Live audit on 2026-05-28 showed the trailing-20 oos_buy_ic
#     median sitting at -0.06 — strictly anti-predictive — while the gate
#     stayed active because |IC|=0.06 > tolerance. The gate's arms
#     (pred < -10 → ×0.6, pred > 10 → ×1.3) assume POSITIVE rank-IC; under
#     negative IC the modulation directionality is INVERTED vs reality, so
#     the gate actively HURTS sized return rather than abstaining. The fix
#     requires demonstrated positive skill above tolerance: anything else
#     (noise OR anti-skill) → kill. Reasoning string still surfaces
#     `scorer=X%(gate-killed,no-skill)` so `_parse_gate_decision` and the
#     dashboard see a single, well-known abstention marker for either case.
#   - Median oos_buy_ic >= tolerance (strictly positive skill): gate stays
#     active. Existing ×0.6/×0.85/×1.15/×1.3 modulation is applied unchanged.
#
# Cached for 1h (TTL) so the JSONL read happens at most once per hour
# regardless of the per-decision cadence — mirrors the live trader's
# `_ml_qualify_cache` pattern in `strategy.py` (CLAUDE.md §15).
_GATE_SKILL_LOG_PATH = ROOT / "data" / "scorer_skill_log.jsonl"
_GATE_SKILL_MIN_CYCLES = 20      # min trailing cycles with parseable oos_buy_ic
# Signed median BUY rank-IC below this ⇒ no demonstrated positive skill →
# kill the gate. Covers BOTH near-zero noise AND persistently negative
# (anti-predictive) skill, since the gate's per-arm sizing assumes positive
# rank-IC and inverts under anti-skill.
_GATE_SKILL_IC_TOLERANCE = 0.03
_GATE_SKILL_KILL_SWITCH_TTL_S = 3600.0  # recheck every hour
_gate_skill_cache: tuple[bool, str, float] | None = None
_GATE_SKILL_CACHE_LOCK = threading.Lock()


def _reset_gate_skill_cache() -> None:
    """Test seam: clear the kill-switch cache so a fresh re-evaluation runs.
    Production callers never need this — the TTL handles staleness."""
    global _gate_skill_cache
    with _GATE_SKILL_CACHE_LOCK:
        _gate_skill_cache = None


def _should_gate_modulate_conviction() -> tuple[bool, str]:
    """Return ``(gate_active, reason)``.

    Reads the trailing ``_GATE_SKILL_MIN_CYCLES`` rows of
    ``scorer_skill_log.jsonl`` and computes the **signed median** of
    ``oos_buy_ic``. When that median is below ``_GATE_SKILL_IC_TOLERANCE``,
    the gate's modulation has no realized economic edge — either at noise
    (|IC|≈0, the original case) or strictly anti-predictive (median < 0,
    the case live data caught on 2026-05-28: trailing-20 median = -0.06).
    The gate's per-arm sizing assumes POSITIVE rank-IC; under anti-skill
    the directionality is inverted vs realized returns, so leaving the
    gate active actively subtracts return rather than abstaining. Return
    ``(False, …)`` in both cases to short-circuit the modulation in
    ``_ml_decide``.

    Defaults to ``(True, …)`` on any fault (missing ledger, unparseable
    rows, fewer than the minimum trailing cycles with parseable
    ``oos_buy_ic``). That preserves invariant #5 semantics during fresh
    starts and never disables the gate due to a transient ledger fault.

    Cached for ``_GATE_SKILL_KILL_SWITCH_TTL_S`` (1h) — the per-decision
    cadence is up to 10 calls per sim_date and the ledger updates only
    once per backtest cycle, so re-reading per-decision wastes IO and
    blocks the run thread.
    """
    global _gate_skill_cache
    now = time.time()
    if _gate_skill_cache is not None:
        gate, reason, ts = _gate_skill_cache
        if now - ts < _GATE_SKILL_KILL_SWITCH_TTL_S:
            return gate, reason

    def _set(result: tuple[bool, str]) -> tuple[bool, str]:
        global _gate_skill_cache
        with _GATE_SKILL_CACHE_LOCK:
            _gate_skill_cache = (result[0], result[1], now)
        return result

    try:
        if not _GATE_SKILL_LOG_PATH.exists():
            return _set((True, "skill ledger missing — default gate-active"))
        # Tail the file: only the last ~MIN_CYCLES rows matter, but the ledger
        # is capped at SCORER_SKILL_LOG_KEEP=2000 rows by the continuous loop,
        # so reading the whole file is bounded. Use a deque to avoid O(n²)
        # over splitlines if the cap ever loosens.
        from collections import deque as _deque
        with _GATE_SKILL_LOG_PATH.open("r") as fh:
            tail = list(
                _deque((ln for ln in fh if ln.strip()),
                       maxlen=_GATE_SKILL_MIN_CYCLES * 3)
            )
        ics: list[float] = []
        for ln in tail:
            try:
                row = json.loads(ln)
            except Exception:
                continue
            v = row.get("oos_buy_ic")
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(fv):
                continue
            ics.append(fv)
        if len(ics) < _GATE_SKILL_MIN_CYCLES:
            return _set((True,
                         f"only {len(ics)}/{_GATE_SKILL_MIN_CYCLES} valid "
                         f"oos_buy_ic rows — default gate-active"))
        # Keep only the most-recent MIN_CYCLES rows (the tail deque may have
        # held more if rows were repeated; the deque is FIFO so the right
        # slice is already the most recent).
        recent = ics[-_GATE_SKILL_MIN_CYCLES:]
        # Median is robust to a single outlier cycle (sklearn convergence
        # warning, a rare regime). A trimmed mean would also work but
        # median is the textbook "is the central tendency at noise" reading.
        recent_sorted = sorted(recent)
        mid = len(recent_sorted) // 2
        if len(recent_sorted) % 2 == 1:
            median_ic = recent_sorted[mid]
        else:
            median_ic = (recent_sorted[mid - 1] + recent_sorted[mid]) / 2.0
        # Signed threshold (not |median|): the gate is killed for BOTH
        # near-zero noise AND persistently anti-predictive (median < 0)
        # skill. The arms (pred<-10 → ×0.6, pred>+10 → ×1.3) assume
        # positive rank-IC; under negative IC the modulation is
        # directionally inverted relative to realized returns, so an
        # active gate subtracts return rather than just being inert. The
        # previous `abs(median_ic)` guard only protected against the
        # noise case — live audit on 2026-05-28 caught the gate firing on
        # trailing-20 median = -0.06 (anti-skill); see the module-level
        # comment above for the full rationale.
        if median_ic < _GATE_SKILL_IC_TOLERANCE:
            return _set((False,
                         f"median oos_buy_ic={median_ic:+.3f} over last "
                         f"{len(recent)} cycles is below "
                         f"+{_GATE_SKILL_IC_TOLERANCE} (noise or "
                         f"anti-predictive) — gate killed"))
        return _set((True,
                     f"median oos_buy_ic={median_ic:+.3f} over last "
                     f"{len(recent)} cycles — gate active"))
    except Exception as exc:
        return _set((True, f"kill-switch read error ({exc}) — default "
                     f"gate-active"))


def _ml_decide(
    sim_date: date,
    portfolio: "SimPortfolio",
    articles: list[dict],
    prices: "PriceCache",
    run_id: int,
    rng: random.Random,
    exclude_tickers: set | None = None,
) -> dict:
    """Pure ML + quant decision — no Claude call.

    Scores every watchlist ticker via ML article scores weighted by sentiment,
    then adjusts with RSI/MACD/momentum. Returns action/ticker/qty dict.
    """
    # 1. Build ticker sentiment scores from articles
    ticker_scores: dict[str, float] = {}
    ticker_article_count: dict[str, int] = {}
    ticker_max_urgency: dict[str, float] = {}
    for a in articles:
        # `.get("score", 0.0)` only defaults on a MISSING key — a present-but-
        # None score (a malformed article dict) reaches float(None) and raises
        # TypeError, which is uncaught here and kills the whole run thread
        # mid-cycle (the run is recorded "failed" with no decisions). `or 0.0`
        # coerces a None/0/"" score to 0.0, which the `< 1.0` guard then skips
        # as no-signal. No behaviour change for any real float score.
        raw_score = float(a.get("score") or 0.0)
        if raw_score < 1.0:
            continue
        sentiment = _article_sentiment(a.get("title", ""))
        # `.get("tickers", [])` only defaults on a MISSING key — a present-but-
        # None value (`"tickers": null` in a malformed article dict) returns
        # None, and `list(None)` raises the SAME uncaught TypeError as the
        # `score` case above, killing the whole run thread mid-cycle. `or []`
        # coerces None/""/0 to the empty list. Identical behaviour for every
        # real list value (a non-empty list is truthy; an empty list stays []).
        tickers = list(a.get("tickers") or [])
        title_lower = (a.get("title") or "").lower()
        # Word-boundary match via pre-compiled patterns — mirrors strategy.py's
        # `_WORD_TO_TICKER_LIVE_PATTERNS` so the backtest signal extractor and
        # the live trader treat the same keyword inputs identically. Naive
        # `word in title_lower` substring matching false-positively aliased
        # "ai" → TQQQ on "training" / "rain" / "Spain" / "blockchain" / etc.,
        # silently inflating unrelated tickers' scores and poisoning the
        # decision_outcomes corpus that retrains the DecisionScorer. See the
        # `_WORD_TO_TICKER_PATTERNS` block for the full rationale.
        for word, sym in _WORD_TO_TICKER.items():
            if sym in tickers:
                continue
            pat = _WORD_TO_TICKER_PATTERNS.get(word)
            if pat is not None and pat.search(title_lower):
                tickers.append(sym)
        try:
            a_urg = float(a.get("urgency", 0.0) or 0.0)
        except (TypeError, ValueError):
            a_urg = 0.0
        for tk in tickers:
            if tk not in WATCHLIST:
                continue
            ticker_scores[tk] = ticker_scores.get(tk, 0.0) + raw_score * sentiment
            ticker_article_count[tk] = ticker_article_count.get(tk, 0) + 1
            if a_urg > ticker_max_urgency.get(tk, 0.0):
                ticker_max_urgency[tk] = a_urg

    # 2. Quant signal adjustments
    quant_tickers = sorted(set(QUANT_SIGNAL_TICKERS) | set(portfolio.positions.keys()))
    quant = _get_quant_signals(sim_date, quant_tickers, prices)
    for tk, q in quant.items():
        adj = 0.0
        # Use only numeric fields. The legacy "MACD" key is a string label
        # ("bullish"/"bearish") used only for prompt display; falling through
        # via `or` when macd_signal==0.0 would feed a string into
        # `isinstance(macd, (int, float))` checks and silently no-op.
        rsi = q.get("rsi")
        macd = q.get("macd_signal")
        mom5 = q.get("mom_5d")
        mom20 = q.get("mom_20d")
        bb = q.get("bb_position")
        if isinstance(rsi, (int, float)):
            if rsi < 33:
                adj += 1.5
            elif rsi < 45:
                adj += 0.5
            elif rsi > 67:
                adj -= 1.5
            elif rsi > 55:
                adj -= 0.5
        if isinstance(macd, (int, float)):
            adj += 0.5 if macd > 0 else -0.5
        if isinstance(mom5, (int, float)):
            adj += min(1.0, max(-1.0, mom5 / 3.0))
        if isinstance(mom20, (int, float)):
            adj += min(0.5, max(-0.5, mom20 / 10.0))
        if isinstance(bb, (int, float)):
            adj -= bb * 0.5  # mean-reversion nudge
        ticker_scores[tk] = ticker_scores.get(tk, 0.0) + adj

    # 3. Market regime dampener
    regime = _market_regime(sim_date, prices)
    # "unknown" (insufficient SPY history) gets a neutral 1.0 — early backtest
    # days previously fell into the bear bucket (0.3), silently dampening every
    # signal for the first ~200 trading days of each run.
    if regime == "bull":
        regime_mult = 1.0
    elif regime == "sideways":
        regime_mult = 0.6
    elif regime == "bear":
        regime_mult = 0.3
    else:  # "unknown"
        regime_mult = 1.0

    total_val = portfolio.total_value(prices, sim_date)

    _excl = exclude_tickers or set()

    # 4. Sell: worst-scoring held position with negative signal
    sell_ticker = None
    worst_score = -0.8
    for tk in portfolio.positions:
        if tk in _excl:
            continue
        s = ticker_scores.get(tk, 0.0) * regime_mult
        if s < worst_score:
            worst_score = s
            sell_ticker = tk

    # 5. Persona-seeded bias — each persona boosts its preferred sector tickers
    # so runs explore different parts of the market, not just tech/semis.
    persona_idx = ((run_id - 1) % len(PERSONAS)) + 1

    for tk, boost in _PERSONA_BOOSTS.get(persona_idx, {}).items():
        if tk in WATCHLIST and tk not in _excl:
            ticker_scores[tk] = ticker_scores.get(tk, 0.0) + boost

    # Per-persona buy threshold — applied BEFORE selection so it actually changes
    # which tickers qualify as buys. Earlier versions adjusted best_score *after*
    # the selection loop, which only shifted conviction (and in the wrong direction).
    if persona_idx in (2, 10):
        buy_threshold = 0.85   # MOMENTUM / SPECULATOR — lower bar
    elif persona_idx in (1, 5):
        buy_threshold = 1.15   # VALUE / GARP — higher bar
    else:
        buy_threshold = 1.0

    # Buy pick: highest-scoring watchlist ticker above threshold (after persona boosts)
    buy_ticker = None
    best_score = buy_threshold
    for tk, s in ticker_scores.items():
        if tk in _excl:
            continue
        adj_s = s * regime_mult
        if adj_s > best_score and prices.price_on(tk, sim_date):
            best_score = adj_s
            buy_ticker = tk

    # Sector concentration guard: if portfolio is >60% single-stock tech, penalise those buys.
    # Leveraged ETFs (SOXL, TQQQ, TECL, etc.) are excluded from this penalty — they are
    # diversified within their sector and are the intended aggressive vehicles.
    tech_tickers = {"NVDA", "AMD", "MU", "INTC", "QCOM", "AAPL", "MSFT", "META",
                    "GOOGL", "SMH", "TSM", "QQQ"}
    if total_val > 0:
        tech_val = sum(
            portfolio.positions[t]["qty"] * (prices.price_on(t, sim_date) or 0)
            for t in portfolio.positions if t in tech_tickers
        )
        if tech_val / total_val > 0.60 and buy_ticker in tech_tickers:
            # Penalise — try the next-best non-tech ticker
            for tk, s in sorted(ticker_scores.items(), key=lambda x: x[1] * regime_mult, reverse=True):
                if tk in _excl or tk in tech_tickers:
                    continue
                adj_s = s * regime_mult
                if adj_s > 0.5 and prices.price_on(tk, sim_date):
                    buy_ticker = tk
                    best_score = adj_s
                    break

    # Populate quant for buy_ticker BEFORE any gate evaluates it. A sentiment-
    # only buy_ticker (outside QUANT_SIGNAL_TICKERS ∪ portfolio.positions)
    # silently bypassed both the 52-week-high gate below AND the CONTRARIAN
    # RSI flip a few lines later, because `quant.get(buy_ticker, {})` returned
    # an empty dict — `wk52_pos`/`rsi` lookups read None and the isinstance
    # guards failed open. Same lazy fetch the later scorer-feature parity
    # block already used (and which is now redundant — kept idempotent via
    # the `not in quant` check below). Empirically ~21% of all BUYs in the
    # live decision_outcomes.jsonl tail are sentiment-only picks (XLF/XLV/XLI/
    # ARKK/BTC-USD/NVDU/AMZU/METAU/CONL), so this gap silently disarmed the
    # bubble gate for the leveraged-single-stock 2x names where a 52w peak
    # buy is most dangerous.
    if buy_ticker and buy_ticker not in quant:
        _extra_q = _get_quant_signals(sim_date, [buy_ticker], prices)
        if buy_ticker in _extra_q:
            quant[buy_ticker] = _extra_q[buy_ticker]

    # 52-week-high gate: suppress buys when the ticker is at an extended high.
    # Prevents buying into bubble peaks where news clusters at market tops
    # (e.g. dot-com 2000) causing underperformance vs shuffled baselines.
    if buy_ticker:
        _w52 = quant.get(buy_ticker, {}).get("wk52_pos")
        if isinstance(_w52, (int, float)) and _w52 > 0.80:
            _peak_penalty = (_w52 - 0.80) * 20.0
            if best_score - _peak_penalty < buy_threshold:
                buy_ticker = None

    # Track the score that actually triggered the sell. Default is the worst-held
    # score from step 4; the CONTRARIAN swap (below) overrides with best_score
    # because that's what tagged the ticker as overbought in the first place.
    sell_score = worst_score
    if persona_idx == 3:       # CONTRARIAN — flip overbought buy to sell
        q = quant.get(buy_ticker or "", {})
        rsi_v = q.get("rsi")
        # Can only SELL what we own. Swapping to a non-held ticker just
        # produces a BLOCKED SELL and loses the buy intent for no reason.
        if (buy_ticker and isinstance(rsi_v, (int, float)) and rsi_v > 65
                and portfolio.positions.get(buy_ticker)):
            sell_ticker, buy_ticker = buy_ticker, None
            sell_score = best_score

    if sell_ticker and portfolio.positions.get(sell_ticker):
        pos = portfolio.positions[sell_ticker]
        sell_qty = round(pos["qty"] * 0.5, 4)
        s_news_count = ticker_article_count.get(sell_ticker, 0)
        s_news_urg = ticker_max_urgency.get(sell_ticker, 0.0)
        return {
            "action": "SELL", "ticker": sell_ticker, "qty": sell_qty,
            "reasoning": (
                f"ML+quant: {sell_ticker} score={sell_score:.2f} regime={regime} "
                f"RSI={quant.get(sell_ticker, {}).get('rsi', 'N/A')} "
                f"news_count={s_news_count} news_urg={s_news_urg:.1f} — reducing"
            ),
        }

    if buy_ticker:
        price = prices.price_on(buy_ticker, sim_date) or 1.0
        # Leveraged ETFs get elevated max conviction in bull/sideways markets — they
        # are the primary vehicle for aggressive outperformance. Cap is 40% of portfolio
        # vs 25% for regular tickers.
        if buy_ticker in _LEVERAGED_ETFS and regime in ("bull", "sideways"):
            conviction = min(0.40, best_score / 15.0)
        else:
            conviction = min(0.25, best_score / 20.0)

        # DecisionScorer nudge: only modulate conviction once the model has seen
        # enough real outcomes (≥500 records). With fewer, it is too noisy to gate.
        # Training/inference feature parity for sentiment-only buy_tickers is
        # now guaranteed by the lazy `_get_quant_signals` fetch a few blocks
        # above (executed BEFORE the wk52 gate so the gate doesn't no-op on
        # empty quant). The scorer feature vector inherits the populated
        # `quant[buy_ticker]` entry by construction.
        q_buy = quant.get(buy_ticker, {})
        buy_news_count = ticker_article_count.get(buy_ticker, 0)
        buy_news_urg = ticker_max_urgency.get(buy_ticker, 0.0)
        _scorer = _get_decision_scorer()
        _feat = dict(
            ml_score=best_score,
            rsi=q_buy.get("rsi"),
            macd=q_buy.get("macd_signal"),
            mom5=q_buy.get("mom_5d"),
            mom20=q_buy.get("mom_20d"),
            regime_mult=regime_mult,
            ticker=buy_ticker,
            vol_ratio=q_buy.get("vol_ratio"),
            bb_pos=q_buy.get("bb_position"),
            news_urgency=buy_news_urg if buy_news_count else None,
            news_article_count=float(buy_news_count) if buy_news_count else None,
            # Enhanced MACD / EMA200 features — `build_features` accepts these
            # as inputs to the 17-th/18-th/19-th feature slots of the deployed
            # scorer, but until this hook landed `_ml_decide` did NOT pass them
            # in, so the inference call defaulted them to None → 0.0 and the
            # corresponding model weights (verified directly via the deployed
            # pickle's `coefs_[0]`: mean |w| = EXACTLY 0.000000 for all three)
            # were dead-trained on always-zero inputs. Pairing this with the
            # matching `_compute_decision_outcomes` capture closes the loop so
            # the next retrain can actually learn these features. None when
            # `_compute_technical_indicators` had insufficient history for the
            # buy ticker (same fall-through as the sibling rsi/macd lookups
            # already use).
            ema200_above=q_buy.get("ema200_above"),
            hist_cross_up=q_buy.get("hist_cross_up"),
            macd_below_zero_cross=q_buy.get("macd_below_zero_cross"),
        )
        # Prefer predict_with_meta so we can see the scorer's own trust flag.
        # The MLP head is unbounded; AGENTS.md documents it extrapolating to
        # nonsense off-distribution (observed -89% then +32% for the SAME
        # LITE vector across two retrain cycles). predict() clamps to
        # ±PRED_CLAMP_PCT so a fabricated -89 surfaces as exactly -50 — which
        # still lands in the `p < -10 → ×0.6` arm and silently halves
        # conviction on pure extrapolation noise. `off_distribution` is True
        # exactly when the raw output exceeded the empirical label support
        # (or the predict call failed / went non-finite); in that case the
        # point estimate carries no information, so we leave the quant-derived
        # conviction untouched rather than modulate on noise. Fall back to the
        # plain predict() scalar for any scorer without the meta sibling (the
        # _Dummy init-failure stub and predict-only test fakes) — treated as
        # in-distribution so existing gate behaviour is unchanged there.
        _pwm = getattr(_scorer, "predict_with_meta", None)
        if callable(_pwm):
            _meta = _pwm(**_feat)
            scorer_pred = float(_meta.get("pred", 0.0))
            scorer_off_dist = bool(_meta.get("off_distribution", False))
        else:
            scorer_pred = float(_scorer.predict(**_feat))
            scorer_off_dist = False
        _scorer_n = getattr(_scorer, "_n_train", 0)
        # Per-cycle no-skill kill-switch (2026-05-25 feature). When the
        # trailing-OOS BUY rank-IC is statistically at noise the gate's
        # per-arm sizing reallocation is variance with no compensating
        # realized edge — see `_should_gate_modulate_conviction` for the
        # full economic rationale and CLAUDE.md §6 / §15 for the supporting
        # diagnostics that already trend this state. Defaults to gate-active
        # on any fault (missing ledger, parse error, insufficient trailing
        # cycles) so invariant #5 semantics are preserved during fresh
        # starts and never disabled by a transient read fault.
        _gate_modulate_active, _gate_kill_reason = (
            _should_gate_modulate_conviction())
        if (_scorer.is_trained and _scorer_n >= 500 and not scorer_off_dist
                and _gate_modulate_active):
            # Scorer adjusts conviction only — never cancels the trade.
            # The LLM already chose this ticker via ML score + quant analysis.
            # Blocking based on 5d forward-return predictions sabotages leveraged
            # ETF strategies: SOXL/TQQQ have noisy 5d windows but strong 3-12 month
            # returns. A HOLD block here was the root cause of the oscillation.
            if scorer_pred < -10.0:
                conviction *= 0.6   # strong headwind — reduce but still buy
            elif scorer_pred < 0.0:
                conviction *= 0.85  # mild headwind — small reduction
            elif scorer_pred > 10.0:
                conviction = min(conviction * 1.3, 0.95)  # strong tailwind
            elif scorer_pred > 5.0:
                conviction = min(conviction * 1.15, 0.95)  # mild tailwind

        buy_notional = min(total_val * conviction, portfolio.cash * 0.95)
        qty = round(buy_notional / price, 4)
        if qty < 0.01:
            return {"action": "HOLD", "ticker": buy_ticker, "qty": 0,
                    "reasoning": f"ML score={best_score:.2f} but notional too small"}
        if not _scorer.is_trained:
            scorer_note = ""
        elif _scorer_n < 500:
            # Scorer trained but sub-gate (n_train < 500, invariant #5). The
            # gate doesn't act on the prediction, so the reasoning must not
            # advertise `scorer=X%` as if it had — `_parse_gate_decision`'s
            # contract is that the token appears ONLY when the gate acted
            # (or explicitly abstained via off-distribution). Emitting it
            # sub-gate poisons `decision_outcomes.gate_scorer_pred` with a
            # value the gate never touched, which leaks into `gate_pnl` /
            # `gate_audit` diagnostics. Suppress like the untrained case.
            scorer_note = ""
        elif scorer_off_dist:
            # Surfaced so the dashboard / a reading quant can see the gate
            # deliberately abstained on an off-distribution extrapolation
            # rather than silently treating a clamped ±50 as a real signal.
            scorer_note = f" scorer={scorer_pred:+.1f}%(off-dist,gate-skipped)"
        elif not _gate_modulate_active:
            # No-skill kill-switch fired: the scorer's trailing OOS BUY
            # rank-IC is statistically at noise, so the gate's
            # ×0.6/×0.85/×1.15/×1.3 modulation was short-circuited
            # (conviction left untouched). Surfaced via the
            # `(gate-killed,no-skill)` marker so the dashboard / a reading
            # quant can distinguish a data-driven abstention (kill-switch)
            # from an architectural one (off-distribution).
            # `_parse_gate_decision` was extended in run_continuous_backtests.py
            # to detect `(gate-killed` AS WELL AS `(off-dist` and set
            # ``gate_off_dist=True`` for either marker, so downstream
            # ``gate_pnl`` / ``gate_arm_historical`` analyzers correctly drop
            # both abstention types — the gate did NOT act in either case,
            # so the bucket assignment cannot be attributed to scorer skill.
            scorer_note = f" scorer={scorer_pred:+.1f}%(gate-killed,no-skill)"
        else:
            scorer_note = f" scorer={scorer_pred:+.1f}%"
        return {
            "action": "BUY", "ticker": buy_ticker, "qty": qty,
            "stop_loss": round(price * 0.92, 2),
            "take_profit": round(price * 1.15, 2),
            "reasoning": (
                f"ML+quant: {buy_ticker} score={best_score:.2f} regime={regime} "
                f"RSI={q_buy.get('rsi', 'N/A')} "
                f"news_count={buy_news_count} news_urg={buy_news_urg:.1f} "
                f"conviction={conviction:.0%}{scorer_note}"
            ),
        }

    return {"action": "HOLD", "ticker": "", "qty": 0,
            "reasoning": f"ML+quant: no high-conviction signal {sim_date} regime={regime}"}


# ─────────────────────────── portfolio sim ───────────────────────────

@dataclass
class SimPortfolio:
    cash: float = INITIAL_CASH
    # ticker -> {qty, avg_cost, stop_loss, take_profit, peak_pct}
    positions: dict[str, dict] = field(default_factory=dict)

    def total_value(self, prices: PriceCache, d: date) -> float:
        v = self.cash
        for ticker, p in self.positions.items():
            px = prices.price_on(ticker, d) or p["avg_cost"]
            v += px * p["qty"]
        return v

    def open_value(self, prices: PriceCache, d: date) -> float:
        v = 0.0
        for ticker, p in self.positions.items():
            px = prices.price_on(ticker, d) or p["avg_cost"]
            v += px * p["qty"]
        return v


def _buy(portfolio: SimPortfolio, ticker: str, qty: float, price: float,
         stop_loss: float | None, take_profit: float | None) -> None:
    notional = qty * price
    portfolio.cash -= notional
    existing = portfolio.positions.get(ticker)
    if existing:
        new_qty = existing["qty"] + qty
        blended = (existing["qty"] * existing["avg_cost"] + qty * price) / new_qty
        existing["qty"] = new_qty
        existing["avg_cost"] = blended
        # Truthiness check would silently drop an explicit `stop_loss=0.0` (or
        # `take_profit=0.0`) update without overwriting the prior value — the
        # new-position branch below stores 0.0 unconditionally via the dict
        # literal, so accumulating into an existing position with the same
        # 0.0 would silently diverge from a fresh open. `is not None` matches
        # the (None-aware, zero-preserving) semantics every other check on
        # these fields uses (`_execute_decision`'s `isinstance(..., (int, float))`,
        # `_enforce_risk_exits`'s `if sl and ...` is intentional — there 0.0
        # means "no real stop" — but for assignment we honor explicit zero).
        if stop_loss is not None:
            existing["stop_loss"] = stop_loss
        if take_profit is not None:
            existing["take_profit"] = take_profit
    else:
        portfolio.positions[ticker] = {
            "qty": qty, "avg_cost": price,
            "stop_loss": stop_loss, "take_profit": take_profit,
        }


def _sell(portfolio: SimPortfolio, ticker: str, qty: float, price: float) -> float:
    pos = portfolio.positions.get(ticker)
    if not pos:
        return 0.0
    qty = min(qty, pos["qty"])
    proceeds = qty * price
    portfolio.cash += proceeds
    pos["qty"] -= qty
    if pos["qty"] <= 1e-6:
        del portfolio.positions[ticker]
    return proceeds


def _enforce_risk_exits(portfolio: SimPortfolio, prices: PriceCache,
                        from_day: date, to_day: date, run_id: int,
                        store: BacktestStore) -> int:
    """Honor only explicit stop_loss / take_profit from Opus. No default risk exits."""
    n = 0
    if not portfolio.positions:
        return 0
    # Membership of `prices.trading_days` (a list, up to ~2500 entries for a
    # 10-year continuous-loop window) is tested once per calendar day in the
    # scan below. `cur not in <list>` is O(len) — for long windows the
    # continuous loop ran this list-scan tens of millions of times per run.
    # Snapshot to a set once: O(1) membership, identical result.
    trading_days_set = set(prices.trading_days)
    cur = from_day + timedelta(days=1)
    while cur <= to_day:
        if not portfolio.positions:
            break
        if cur not in trading_days_set:
            cur += timedelta(days=1)
            continue
        for ticker in list(portfolio.positions.keys()):
            pos = portfolio.positions[ticker]
            px = prices.price_on(ticker, cur)
            if px is None:
                continue
            sl = pos.get("stop_loss")
            tp = pos.get("take_profit")
            if sl and px <= sl:
                qty = pos["qty"]
                _sell(portfolio, ticker, qty, px)
                store.record_trade(run_id, cur.isoformat(), ticker, "SELL", qty, px,
                                   f"stop-loss @ {sl} (close {px:.2f})")
                n += 1
            elif tp and px >= tp:
                qty = pos["qty"]
                _sell(portfolio, ticker, qty, px)
                store.record_trade(run_id, cur.isoformat(), ticker, "SELL", qty, px,
                                   f"take-profit @ {tp} (close {px:.2f})")
                n += 1
        cur += timedelta(days=1)
    return n


# ─────────────────────────── Opus call ───────────────────────────

SYSTEM_PROMPT = """You are managing a paper trading portfolio with $1000 starting capital.
This is a HISTORICAL backtest — you are deciding trades for a specific past date based on news available at that date.
Your ONLY goal is maximum profit over a 1-year horizon. You have complete freedom over position
sizing, risk, and timing. There are NO enforced limits. You can:
- Put 100% of portfolio into one trade if you have high conviction
- Go all-in on a single ticker
- Let losers run if you expect reversal

THINK LIKE A HEDGE FUND MANAGER WHO WANTS ASYMMETRIC RETURNS.
Small, safe trades will not outperform. Take calculated risks.
High conviction = large size. Low conviction = stay cash.

Trade US stocks ONLY (no options or futures in this backtest).

LEVERAGE INSTRUMENTS AVAILABLE:
- Leveraged ETFs 3x Bull: TQQQ (QQQ), UPRO/SPXL (SPY), UDOW (Dow), URTY (Russell), SOXL (semis), TECL (tech), FNGU (tech FANGs), CURE (healthcare), LABU (biotech), NAIL (homebuilders), DPST (banks), FAS (financials), DFEN (defense), TNA (small-cap), UTSL (utilities)
- Leveraged ETFs 2x Bull: QLD (QQQ 2x), SSO (SPY 2x), NVDU (NVDA), MSFU (MSFT), AMZU (AMZN), GOOGU (GOOG), METAU (META), TSLL (TSLA), CONL (COIN), LNOK (Nokia), BITU (BTC), ETHU (ETH)
- Leveraged ETFs Bear/Hedge: SQQQ/SPXS (3x short index), SOXS (3x short semis), TECS (3x short tech), FNGD (3x short FANGs)
- Crypto leveraged: BITX (2x BTC), BITU (2x BTC), ETHU (2x ETH)
- For high-conviction directional trades, consider 2-3x leveraged ETFs instead of the underlying
- For options-equivalent exposure: buy deep ITM LEAPS calls (delta >0.80) to simulate leveraged long
- Risk: leveraged ETFs decay in sideways markets; best for strong trending moves only

POSITION SIZING GUIDANCE (committee should consider):
- High conviction (RSI+MACD+MA all aligned): up to 40% portfolio
- Medium conviction (2/3 signals aligned): 15-25%
- Low conviction / leveraged ETF: max 10%
- Never go 100% into one leveraged ETF (decay risk)

Respond with a SINGLE JSON object — no prose, no markdown fences. Schema:

{
  "action": "BUY" | "SELL" | "HOLD",
  "ticker": "NVDA",
  "qty": 0.5,
  "confidence": 0.85,
  "reasoning": "1-3 sentences why",
  "stop_loss": 850.0,       // optional — only honored if set
  "take_profit": 950.0      // optional — only honored if set
}

- For SELL, ticker must match an open position.
- Fractional shares are allowed (qty can be e.g. 0.5).
- If you set stop_loss / take_profit, they will fire on daily closes.

Return JSON ONLY.
"""


def _claude_call(prompt: str, retries: int = 1) -> str | None:
    """Thin delegation to llm_adapter.call_llm.

    The `retries` parameter is kept for backward compatibility with any
    existing callers, but the unified adapter owns retry policy internally.
    """
    return _llm_call(MODEL, prompt)


def _parse_decision(raw: str | None) -> dict | None:
    if not raw:
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception as e:
        print(f"[backtest] JSON parse failed: {e} raw={text[:200]!r}")
        return None


COMMITTEE_BRIEF = """You are a trading committee of 10 traders managing a single $1000 paper trading portfolio.
Each committee member has a distinct style. For this trading day, each member silently proposes
a trade. The committee then VOTES and executes the single highest-conviction consensus trade.

THE 10 COMMITTEE MEMBERS:
1. VALUE      — P/E, fundamentals, undervalued cash-flow machines, durable moats
2. MOMENTUM   — Buys what is going up; earnings beats + raised guidance; ride trends
3. CONTRARIAN — Buys fear, sells greed; oversold quality; mean reversion
4. MACRO      — Rates, FX, geopolitics expressed through leveraged equities (SOXL/TQQQ/BITU)
5. GARP       — Growth at reasonable price; quality compounders with sane multiples
6. QUANT      — Pure signal/catalyst reaction; news-driven, unemotional
7. ROTATOR    — Sector rotation by macro cycle; XLE/XLK/XLF/SMH/ARKK
8. SMALLCAP   — Gems outside mega-caps; LITE, MU, RIVN, NIO, COIN, PLTR, MSTR, SHOP, SQ
9. ESG/THEME  — AI infra, clean energy, semis, GLP-1, EVs; picks-and-shovels
10. SPECULATOR — Concentrated asymmetric bets; full-size when setup is right

PROCESS:
  (a) Each member proposes one trade (BUY/SELL/HOLD).
  (b) Members vote — weighted by conviction and by how well the proposal fits today's signals.
  (c) Output the SINGLE consensus trade as JSON.

The reasoning field MUST briefly list each member's proposal then state the consensus, e.g.:
"VALUE: BUY HSBC. MOMENTUM: BUY NVDA. CONTRARIAN: HOLD. MACRO: BUY SOXL. GARP: BUY LLY.
QUANT: BUY NVDA. ROTATOR: BUY SMH. SMALLCAP: BUY LITE. ESG: BUY NVDA. SPECULATOR: BUY MSTR.
Consensus: BUY NVDA (4 votes + highest conviction on AI compute catalyst)."
"""


def _build_prompt(run_id: int, seed: int, sim_date: date, portfolio: SimPortfolio,
                  top_articles: list[dict], prices: PriceCache,
                  extra_quant_block: str = "") -> str:
    pos_lines = []
    for ticker, p in portfolio.positions.items():
        px = prices.price_on(ticker, sim_date) or p["avg_cost"]
        pl_pct = (px - p["avg_cost"]) / p["avg_cost"] * 100
        pos_lines.append(f"  {ticker}: qty={p['qty']} avg=${p['avg_cost']:.2f} "
                         f"now=${px:.2f} P/L={pl_pct:+.1f}%")

    art_lines = []
    for a in top_articles:
        tickers = a.get("tickers", [])
        t_str = f" tickers={','.join(tickers[:5])}" if tickers else ""
        art_lines.append(f"  [{a['score']:.1f}] {a['title'][:140]}{t_str}")

    px_lines = []
    for t in WATCHLIST:
        if t.startswith("^"):
            continue  # index gauges shown elsewhere
        p = prices.price_on(t, sim_date)
        px_lines.append(f"  {t}: ${p:.2f}" if p else f"  {t}: N/A")

    # Technical signals for held positions + top watchlist names.
    quant_tickers = sorted(set(QUANT_SIGNAL_TICKERS) | set(portfolio.positions.keys()))
    quant_sigs = _get_quant_signals(sim_date, quant_tickers, prices)
    quant_lines = []
    for tk in sorted(quant_sigs.keys()):
        q = quant_sigs[tk]
        quant_lines.append(
            f"  {tk}: RSI={q.get('RSI')}  MACD={q.get('MACD')}  "
            f"MA={q.get('MA_cross')}  vol_ratio={q.get('vol_ratio')}  "
            f"52h={q.get('pct_from_52h')}%  52l={q.get('pct_from_52l')}%"
        )

    vix = _vix_level(sim_date, prices)
    regime = _market_regime(sim_date, prices)
    rotation = _sector_rotation(sim_date, prices)
    rot_str = ", ".join(f"{t} {p:+.1f}%" for t, p in rotation) if rotation else "n/a"
    vix_str = f"{vix:.2f}" if vix is not None else "N/A"

    total = portfolio.total_value(prices, sim_date)

    return f"""{SYSTEM_PROMPT}

---
{COMMITTEE_BRIEF}

---
SIMULATION CONTEXT:
You are committee instance #{run_id} with random seed {seed}. Other parallel committees
see different slices of news and may reach different consensus trades — that's expected.

SIMULATED DATE: {sim_date.isoformat()}

PORTFOLIO:
  cash: ${portfolio.cash:.2f}
  total value: ${total:.2f}
  positions:
{chr(10).join(pos_lines) if pos_lines else '  (none)'}

WATCHLIST CLOSES on {sim_date.isoformat()}:
{chr(10).join(px_lines)}

MARKET STRUCTURE on {sim_date.isoformat()}:
  VIX: {vix_str}
  Market regime: {regime} (SPY vs 50/200 MA)
  Sector rotation (~21d return): {rot_str}

TECHNICAL SIGNALS (positions + top watchlist):
{chr(10).join(quant_lines) if quant_lines else '  (no quant signals — insufficient history)'}

{extra_quant_block or ''}

TOP NEWS SIGNALS for {sim_date.isoformat()} (score 0..5 from keyword heuristic):
{chr(10).join(art_lines) if art_lines else '  (no signals)'}

Return JSON only — a SINGLE consensus decision, no per-member objects.
"""


# ─────────────────────────── engine ───────────────────────────

@dataclass
class BacktestRun:
    run_id: int
    seed: int
    start_date: str           # ISO date — no default; engine must supply window
    end_date: str             # ISO date — no default; engine must supply window
    start_value: float = INITIAL_CASH
    final_value: float = 0.0
    total_return_pct: float = 0.0
    spy_return_pct: float = 0.0
    vs_spy_pct: float = 0.0
    n_trades: int = 0
    n_decisions: int = 0
    bubble_gate_suppressions: int = 0
    status: str = "pending"
    trades: list[dict] = field(default_factory=list)
    decisions: list[dict] = field(default_factory=list)
    equity_curve: list[dict] = field(default_factory=list)


LOCAL_ARTICLES_DB = Path(__file__).resolve().parent.parent.parent / "digital-intern" / "data" / "articles.db"


def _parse_yf_news_item(item: dict) -> tuple[str, str, float | None]:
    """Extract ``(title, url, pub_ts)`` from one ``yf.Ticker(...).news`` entry.

    yfinance changed its news payload in late 2024:

      * **Legacy** (pre-late-2024) — flat top-level keys::

            {"title": "...", "link": "...",
             "providerPublishTime": <unix-ts>}

      * **Current** — nested under ``content``::

            {"id": "...",
             "content": {"title": "...",
                         "canonicalUrl": {"url": "..."},
                         "clickThroughUrl": {"url": "..."},
                         "pubDate": "2026-05-24T01:50:00Z"}}

    The prior backtest reader only looked at the legacy top-level keys, so
    every current-schema entry came back as ``title=""`` and was dropped at
    the ``if not title: continue`` guard — Tier 2 of ``_fetch_signals`` went
    dark for every recent sim_date. This helper reads the current schema
    first (where it exists) and falls back to legacy fields, so both shapes
    work without a schema-version branch at the call site.

    Returns a 3-tuple of ``(title:str, url:str, pub_ts:float|None)``:

      * ``title`` — empty string when absent in BOTH shapes (the caller's
        ``if not title: continue`` is the documented filter).
      * ``url`` — empty string when neither ``canonicalUrl.url`` /
        ``clickThroughUrl.url`` (new) nor ``link`` (legacy) is present.
        Empty-URL items still flow through (the caller skips dedup on "").
      * ``pub_ts`` — Unix epoch seconds (float) when parseable, else
        ``None``. The new schema uses an ISO-8601 ``pubDate`` string; the
        legacy schema used ``providerPublishTime`` as a Unix timestamp. We
        try both; an unparseable date degrades to ``None`` (the caller's
        forward-leakage guard only filters when ``pub_ts`` is a number).

    Pure, total — never raises (a per-item parse failure is the caller's
    cue to skip that item, mirroring the existing GDELT / volume / AV cache
    type-guard discipline).
    """
    content = item.get("content") if isinstance(item, dict) else None
    if not isinstance(content, dict):
        content = {}

    # title: prefer new-schema, fall back to legacy top-level
    title = content.get("title") or item.get("title") or ""
    if not isinstance(title, str):
        title = str(title)

    # url: try canonicalUrl.url, then clickThroughUrl.url, then legacy link.
    # Both new-schema URL fields are dicts; either may be None.
    url = ""
    for key in ("canonicalUrl", "clickThroughUrl"):
        candidate = content.get(key)
        if isinstance(candidate, dict):
            u = candidate.get("url")
            if isinstance(u, str) and u:
                url = u
                break
    if not url:
        legacy_link = item.get("link") or content.get("link")
        if isinstance(legacy_link, str):
            url = legacy_link

    # pub_ts: legacy providerPublishTime is already epoch-seconds; new-schema
    # pubDate is an ISO-8601 string we parse to a UTC epoch float. Either may
    # be missing.
    pub_ts: float | None = None
    legacy_ts = item.get("providerPublishTime")
    if isinstance(legacy_ts, (int, float)):
        pub_ts = float(legacy_ts)
    else:
        pub_date_iso = content.get("pubDate") or content.get("displayTime")
        if isinstance(pub_date_iso, str) and pub_date_iso:
            try:
                # Normalize the trailing 'Z' (RFC 3339) to '+00:00' for
                # `fromisoformat`. Python <3.11 doesn't accept the literal
                # 'Z' suffix; manual strip is portable across 3.10+.
                iso = pub_date_iso.rstrip("Z")
                if iso != pub_date_iso:
                    iso = iso + "+00:00"
                pub_ts = datetime.fromisoformat(iso).timestamp()
            except (TypeError, ValueError):
                pub_ts = None
    return title, url, pub_ts


class BacktestEngine:
    _VALID_MODEL_PREFIXES = ("ml_quant", "claude-", "gpt-", "hf/")
    # Class-level default so callers that bypass __init__ via
    # `BacktestEngine.__new__(...)` (the canonical no-network test pattern in
    # tests/test_integration_backtest.py and tests/test_model_rankings.py)
    # still see a sensible model_id for run_one → upsert_run wiring.
    model_id: str = "ml_quant"

    def __init__(self, start: date | None = None, end: date | None = None,
                 model_id: str = "ml_quant"):
        # Standalone runs (e.g. `python3 run_backtests.py`) get a sane default
        # equal to the pre-refactor hardcoded window so the one-shot launcher
        # keeps working without arg-plumbing changes. Continuous loop overrides.
        if not (model_id == "ml_quant" or model_id.startswith(("claude-", "hf/"))):
            raise ValueError(
                f"Invalid model_id {model_id!r}. Must start with one of "
                f"{self._VALID_MODEL_PREFIXES}"
            )
        self.model_id = model_id
        self.start = start or date(2025, 5, 1)
        self.end = end or date(2026, 5, 13)
        self.store = BacktestStore()
        self.prices = PriceCache(WATCHLIST, self.start, self.end)
        self.gdelt = GDELTFetcher()
        self.av_news = AlphaVantageNewsFetcher()
        # Pre-load local articles DB into memory keyed by ISO date string.
        # This is instant for all subsequent lookups — no network, no disk per-date.
        self._local_news: dict[str, list[dict]] = self._load_local_articles()
        if not self.prices.trading_days:
            raise RuntimeError("PriceCache has no trading days — yfinance fetch failed")

    def refresh_local_articles(self) -> int:
        """Reload local articles from disk and atomically swap the in-memory map.

        The continuous loop reuses one BacktestEngine for many hours; without
        this, `_local_news` is frozen at engine startup and progressively misses
        every article written after that point. Returns total article count
        after refresh (or current count on failure)."""
        fresh = self._load_local_articles()
        if fresh:
            self._local_news = fresh
        return sum(len(v) for v in self._local_news.values())

    def _load_local_articles(self) -> dict[str, list[dict]]:
        """Load entire articles.db into memory, keyed by published/first_seen date (YYYY-MM-DD).

        Returns empty dict if DB is unavailable — callers fall back gracefully.
        SEC EDGAR cache files (populated by historical_collector) are merged in
        regardless of whether the articles DB exists, so pre-2015 windows can
        still see SEC filings as signals.
        """
        result: dict[str, list[dict]] = {}
        conn = None
        try:
            import sqlite3 as _sqlite3, zlib as _zlib
            if not LOCAL_ARTICLES_DB.exists():
                # Skip DB load but still merge SEC cache below.
                raise FileNotFoundError("articles.db not present — SEC-only mode")
            conn = _sqlite3.connect(f"file:{LOCAL_ARTICLES_DB}?mode=ro", uri=True, timeout=5.0)
            # Filter out backtest-injected rows so the engine doesn't read its own
            # past decisions as future signals (training contamination). Mirrors
            # the live-only clause used by paper_trader/signals.py.
            rows = conn.execute(
                "SELECT title, url, source, published, first_seen, ai_score, "
                "kw_score, full_text, urgency "
                "FROM articles WHERE title IS NOT NULL AND title != '' "
                "AND (url IS NULL OR url NOT LIKE 'backtest://%') "
                "AND (source IS NULL OR (source NOT LIKE 'backtest_%' "
                "AND source NOT LIKE 'opus_annotation%'))"
            )
            # Hindsight label filter: an article whose `first_seen` lags
            # `published` by more than this many days was almost certainly
            # collected retroactively (e.g. by backfill_news.py / GDELT) and
            # then labeled by Claude with knowledge of what happened after
            # publication. Trusting that `ai_score` for a historical backtest
            # is silent lookahead — fall back to the keyword baseline instead.
            LABEL_STALENESS_DAYS = 60
            win_start = self.start - timedelta(days=30)
            win_end = self.end + timedelta(days=30)
            from email.utils import parsedate_to_datetime as _parse_822

            def _parse_day(raw):
                if not raw:
                    return None
                try:
                    dt = _parse_822(raw)
                    if dt is not None:
                        return dt.date()
                except Exception:
                    pass
                try:
                    return date.fromisoformat(str(raw)[:10])
                except Exception:
                    return None

            scanned = 0
            for (title, url, source, published, first_seen, ai_score,
                 kw_score, full_text, urgency) in rows:
                scanned += 1
                pub_d = _parse_day(published)
                if pub_d is None:
                    continue
                if pub_d < win_start or pub_d > win_end:
                    continue
                day_str = pub_d.isoformat()
                # Decode full text for richer scoring
                snippet = ""
                if full_text:
                    try:
                        snippet = _zlib.decompress(full_text).decode("utf-8", errors="replace")[:300]
                    except Exception:
                        pass

                seen_d = _parse_day(first_seen)
                # NULL first_seen means "we don't know when this was labeled"
                # (rows that pre-date the column). Don't punish those — keep the
                # legacy `ai_score or kw_score` fallback. Only mark as
                # contaminated when we can prove the lag.
                hindsight_contaminated = False
                if seen_d is not None:
                    days_lag = (seen_d - pub_d).days
                    hindsight_contaminated = days_lag > LABEL_STALENESS_DAYS

                # Distinguish None (unscored) from 0.0 (legitimately-zero ML
                # score). The prior `ai_score or kw_score or 0` form collapsed
                # both — so an article ArticleNet had scored as flat-zero (a
                # real "this carries no signal" verdict) silently inherited the
                # heuristic baseline `kw_score` (typically 2.5 from
                # score_article's BUY_PHRASES/SELL_PHRASES net of zero hits).
                # That promoted "scored 0" content above other articles, biasing
                # the backtest's per-ticker sentiment scores. Explicit None
                # checks keep the legitimate-zero verdict intact.
                if hindsight_contaminated:
                    score = float(kw_score) if kw_score is not None else 0.0
                elif ai_score is not None:
                    score = float(ai_score)
                elif kw_score is not None:
                    score = float(kw_score)
                else:
                    score = 0.0

                # `urgency` is also computed by ArticleNet at first_seen time. A
                # hindsight-contaminated row was scored AFTER what the article
                # described actually happened, so its urgency is forward-looking
                # too — keeping it would leak that knowledge into the
                # `news_urgency` scorer feature (via _ml_decide → reasoning →
                # _compute_decision_outcomes → train_scorer). Mirror the score
                # filter: zero out urgency for the same reason kw_score replaces
                # ai_score here.
                if hindsight_contaminated:
                    urg_v = 0.0
                else:
                    try:
                        urg_v = float(urgency) if urgency is not None else 0.0
                    except (TypeError, ValueError):
                        urg_v = 0.0
                result.setdefault(day_str, []).append({
                    "title": title, "url": url or "",
                    "source": source or "", "score": score, "snippet": snippet,
                    "urgency": urg_v,
                    "hindsight_contaminated": hindsight_contaminated,
                })
            print(f"[local_news] scanned {scanned} rows, loaded {sum(len(v) for v in result.values())} articles across {len(result)} days from local DB")
        except FileNotFoundError:
            # Expected for pre-2015 backtest windows that pre-date the digital-intern
            # articles DB — SEC merge below still runs.
            pass
        except Exception as e:
            print(f"[local_news] DB load failed (using quant-only mode): {e}")
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

        # Merge any cached SEC EDGAR filings overlapping this engine's window.
        # historical_collector.fetch_sec_historical writes per-(ticker, start, end)
        # JSON files; we union all matching files into _local_news so pre-2015
        # windows (which have no GDELT, no live articles.db rows) still see SEC
        # 8-K/10-Q/10-K as signals.
        try:
            sec_added = self._merge_sec_cache(result)
            if sec_added:
                print(f"[local_news] merged {sec_added} SEC filings from disk cache")
        except Exception as e:
            print(f"[local_news] SEC merge failed: {e}")

        return result

    def _merge_sec_cache(self, result: dict[str, list[dict]]) -> int:
        """Union SEC EDGAR cache files overlapping the engine's window into result.

        Each file is named `{TICKER}_{start}_{end}.json`. We accept any file whose
        recorded window has *some* overlap with this engine's window, since the
        filings inside carry their own `published` date and we filter per-filing.
        """
        sec_dir = CACHE_DIR / "sec_edgar"
        if not sec_dir.exists():
            return 0
        added = 0
        win_start = self.start
        win_end = self.end
        seen_urls: set[str] = set()
        for f in sec_dir.iterdir():
            if not f.name.endswith(".json"):
                continue
            # Filename: TICKER_YYYY-MM-DD_YYYY-MM-DD.json. Parse loosely; if it
            # doesn't match, fall through and just trust per-filing dates below.
            try:
                entries = json.loads(f.read_text())
            except Exception:
                continue
            # Type guard mirrors the GDELT / AlphaVantage / volume cache
            # loaders. A corrupt SEC cache file (truncated, externally edited)
            # could deserialize to a dict / number / string, then crash the
            # whole engine init at `e.get("published")` because iterating a
            # string yields chars (no .get) and iterating a number raises
            # TypeError. Skip non-list payloads; per-entry isinstance check
            # below already protects against mixed-type contents.
            if not isinstance(entries, list):
                continue
            for e in entries:
                if not isinstance(e, dict):
                    continue
                pub = e.get("published") or ""
                if len(pub) < 10:
                    continue
                pub_d = pub[:10]
                try:
                    pd = date.fromisoformat(pub_d)
                except ValueError:
                    continue
                if pd < win_start or pd > win_end:
                    continue
                url = e.get("url") or ""
                if url and url in seen_urls:
                    continue
                if url:
                    seen_urls.add(url)
                title = e.get("title") or ""
                if not title:
                    continue
                result.setdefault(pub_d, []).append({
                    "title": title,
                    "url": url,
                    "source": e.get("source") or "SEC",
                    # SEC filings without Claude labels carry a modest baseline
                    # signal — they're material disclosures, not noise.
                    "score": 2.5,
                    "snippet": (e.get("full_text") or "")[:300],
                    "urgency": 0.0,
                })
                added += 1
        return added

    def _sampled_days(self) -> list[date]:
        days = self.prices.trading_days
        return days[::SAMPLE_EVERY_N_DAYS]

    def _fetch_signals(self, d: date, seed: int, rng: random.Random,
                       portfolio: "SimPortfolio | None" = None) -> list[dict]:
        """Three-tier news fetch — fast local DB first, network only as fallback.

        Tier 1 (instant):  local articles.db pre-loaded into memory at engine init.
        Tier 2 (fast):     yfinance ticker.news — no key, no rate limit, recent only.
        Tier 3 (slow/opt): GDELT disk cache — only if local DB returned 0 articles
                           AND the cache file already exists (no outbound call).
        Alpha Vantage:     quota-guarded, disk-cached, fetched only when quota allows.
        """
        articles: list[dict] = []
        seen_urls: set[str] = set()

        # ── Tier 1: local articles DB (in-memory, zero latency) ──────────────
        # Skip dedup for empty URLs — multiple URL-less articles must not all
        # collapse to the same seen_urls bucket (only the first would survive).
        day_str = d.isoformat()
        for a in self._local_news.get(day_str, []):
            url = a.get("url", "")
            if url and url in seen_urls:
                continue
            if url:
                seen_urls.add(url)
            _, tickers = score_article({"title": a["title"], "url": url})
            articles.append({"title": a["title"], "url": url,
                             "score": a["score"], "tickers": tickers,
                             "urgency": a.get("urgency", 0.0)})

        # ── Tier 2: yfinance recent news (no rate limit, no API key) ─────────
        if d >= date.today() - timedelta(days=30):
            for a in self._fetch_yf_news(list(QUANT_SIGNAL_TICKERS), d):
                url = a.get("url", "")
                if url and url in seen_urls:
                    continue
                if url:
                    seen_urls.add(url)
                articles.append(a)

        # ── Alpha Vantage (quota-guarded, disk-cached) ────────────────────────
        # Only fetch for recent dates. AV's NEWS_SENTIMENT historically returned
        # latest news regardless of date (before the time_from/time_to fix), so
        # cached files for old sim_dates may contain forward-leaking current news.
        # Restrict to the last 14 days so any contamination is bounded to recent
        # backtest windows that already overlap with live data.
        if d >= date.today() - timedelta(days=14):
            if portfolio is not None:
                av_tickers = list(portfolio.positions.keys())[:2] + ["NVDA", "SPY"]
            else:
                av_tickers = ["NVDA", "SPY"]
            for a in self.av_news.fetch(list(dict.fromkeys(av_tickers))[:4], d):
                url = a.get("url", "")
                if url and url in seen_urls:
                    continue
                if url:
                    seen_urls.add(url)
                score, tickers = score_article(a)
                articles.append({"title": a["title"], "url": url,
                                 "score": score, "tickers": tickers})

        # ── Tier 3: GDELT — only from existing disk cache, no outbound calls ──
        # Skip entirely if local DB already gave us articles (fast path).
        if not articles:
            idxs = rng.sample(range(len(KEYWORD_GROUPS)), 2)
            for i in idxs:
                kw = KEYWORD_GROUPS[i]
                cache_path = self.gdelt._cache_key(d, kw)
                if not cache_path.exists():
                    continue  # skip — no cache, no network call
                for a in self.gdelt.fetch(d, kw):
                    url = a.get("url", "")
                    if not a.get("title"):
                        continue
                    if url and url in seen_urls:
                        continue
                    if url:
                        seen_urls.add(url)
                    score, tickers = score_article(a)
                    articles.append({"title": a["title"], "url": url,
                                     "score": score, "tickers": tickers})

        # ── Rank and sample ───────────────────────────────────────────────────
        articles.sort(key=lambda x: x["score"], reverse=True)
        top10 = articles[:10]
        if len(top10) <= 5:
            return top10
        return sorted(rng.sample(top10, 5), key=lambda x: x["score"], reverse=True)

    def _execute_decision(self, run_id: int, sim_date: date, decision: dict,
                          portfolio: SimPortfolio) -> tuple[str, str]:
        action = (decision.get("action") or "HOLD").upper()
        if action == "HOLD":
            return "HOLD", decision.get("reasoning", "")
        ticker = (decision.get("ticker") or "").upper()
        try:
            qty = float(decision.get("qty") or 0)
        except (TypeError, ValueError):
            return "BLOCKED", "bad qty"
        if not ticker:
            return "BLOCKED", "no ticker"
        if qty <= 0:
            return "BLOCKED", "qty must be > 0"

        price = self.prices.price_on(ticker, sim_date)
        if price is None or price <= 0:
            return "BLOCKED", f"no price for {ticker} on {sim_date}"

        if action == "BUY":
            # Bubble gate: suppress buys when price is more than 2x SMA200.
            # Skip the gate if fewer than 200 prior closes are available.
            pairs = _series_up_to(self.prices, ticker, sim_date, max_points=220)
            if len(pairs) >= 200:
                last200 = [v for _, v in pairs[-200:]]
                sma200 = sum(last200) / 200.0
                if price > 2.0 * sma200:
                    return (
                        "BUBBLE_GATE_SUPPRESSED",
                        f"bubble_gate_suppressed: {ticker} price={price:.2f} > 2.0x sma200={sma200:.2f}",
                    )
            notional = qty * price
            if portfolio.cash - notional < 0:
                return "BLOCKED", f"insufficient cash (have ${portfolio.cash:.2f}, need ${notional:.2f})"
            sl = decision.get("stop_loss")
            tp = decision.get("take_profit")
            _buy(portfolio, ticker, qty, price,
                 float(sl) if isinstance(sl, (int, float)) else None,
                 float(tp) if isinstance(tp, (int, float)) else None)
            self.store.record_trade(run_id, sim_date.isoformat(), ticker, "BUY", qty,
                                    price, decision.get("reasoning", "")[:200])
            return "FILLED", f"BUY {qty} {ticker} @ {price:.2f}"

        if action == "SELL":
            pos = portfolio.positions.get(ticker)
            if not pos:
                return "BLOCKED", f"no open position in {ticker}"
            sell_qty = min(qty, pos["qty"])
            _sell(portfolio, ticker, sell_qty, price)
            self.store.record_trade(run_id, sim_date.isoformat(), ticker, "SELL",
                                    sell_qty, price,
                                    decision.get("reasoning", "")[:200])
            return "FILLED", f"SELL {sell_qty} {ticker} @ {price:.2f}"

        return "BLOCKED", f"unsupported action {action}"

    def run_one(self, run_id: int, seed: int | None = None) -> BacktestRun:
        if seed is None:
            seed = int.from_bytes(os.urandom(4), "big") ^ (run_id * 1337)
        rng = random.Random(seed)
        self.store.upsert_run(run_id, seed, "running",
                              start=self.start, end=self.end,
                              model_id=self.model_id)
        print(f"\n══════ RUN {run_id}  seed={seed} window={self.start}→{self.end} "
              f"model={self.model_id} ══════")

        portfolio = SimPortfolio()
        equity_curve: list[dict] = []
        n_trades = 0
        n_bubble_suppressed = 0
        n_decisions = 0
        prev_sample = self.prices.trading_days[0] - timedelta(days=1)

        sampled = self._sampled_days()
        if sampled and sampled[-1] != self.prices.trading_days[-1]:
            sampled.append(self.prices.trading_days[-1])
        print(f"[run {run_id}] {len(sampled)} sample days")

        for idx, sim_date in enumerate(sampled):
            # daily SL/TP scan since previous sample
            exits = _enforce_risk_exits(portfolio, self.prices, prev_sample,
                                        sim_date, run_id, self.store)
            n_trades += exits
            prev_sample = sim_date

            # fetch & score (once per day — signals don't change intraday)
            signals = self._fetch_signals(sim_date, seed, rng, portfolio)

            # Intraday loop: up to MAX_DECISIONS_PER_DAY ml_decide calls per day
            # for the ml_quant path. LLM paths (claude-*, hf/*) call once per
            # day — they are slow + expensive, and the prompt would have to be
            # regenerated with growing `exclude_tickers` each pass.
            # Each filled trade excludes that ticker from subsequent calls today.
            traded_today: set[str] = set()
            day_filled = 0
            status, detail = "NO_DECISION", "decide returned None"
            decision = None
            max_intra = MAX_DECISIONS_PER_DAY if self.model_id == "ml_quant" else 1
            for _intra in range(max_intra):
                if self.model_id == "ml_quant":
                    decision = _ml_decide(sim_date, portfolio, signals, self.prices,
                                          run_id, rng, exclude_tickers=traded_today)
                else:
                    # LLM-based decision — build the same prompt the live trader
                    # would see and route through the unified adapter.
                    prompt = _build_prompt(run_id, seed, sim_date, portfolio,
                                           signals, self.prices)
                    raw = _llm_call(self.model_id, prompt)
                    decision = _parse_decision(raw)
                n_decisions += 1

                if not decision:
                    break
                if decision.get("action", "HOLD") == "HOLD":
                    status, detail = "HOLD", str(decision.get("reasoning", ""))[:200]
                    break

                status, detail = self._execute_decision(run_id, sim_date, decision, portfolio)
                ticker_acted = decision.get("ticker") or ""
                if status == "BUBBLE_GATE_SUPPRESSED":
                    n_bubble_suppressed += 1
                if status == "FILLED":
                    n_trades += 1
                    day_filled += 1
                    total = portfolio.total_value(self.prices, sim_date)
                    self.store.record_decision(run_id, sim_date.isoformat(), decision,
                                               status, detail, portfolio.cash, total,
                                               len(signals))
                if ticker_acted:
                    traded_today.add(ticker_acted)

            total = portfolio.total_value(self.prices, sim_date)
            # Record a terminal HOLD/NO_DECISION if nothing was filled or as end-of-day marker
            if day_filled == 0:
                self.store.record_decision(run_id, sim_date.isoformat(), decision,
                                           status, detail, portfolio.cash, total,
                                           len(signals))

            # Per-sample equity snapshot + every-5-samples DB persist so the
            # dashboard can render partial equity curves while the run executes.
            equity_curve.append({
                "date": sim_date.isoformat(),
                "value": round(total, 2),
                "cash": round(portfolio.cash, 2),
            })
            if idx % 5 == 0 or idx == len(sampled) - 1:
                try:
                    self.store.update_partial_progress(
                        run_id, total, n_trades, n_decisions, equity_curve,
                    )
                except Exception as pe:
                    print(f"[run {run_id}] partial persist failed: {pe}")

            if idx % 10 == 0 or idx == len(sampled) - 1:
                print(f"  [run {run_id} {idx+1}/{len(sampled)}] {sim_date} "
                      f"action={status} cash=${portfolio.cash:.2f} total=${total:.2f}")

        # final mark
        final_day = self.prices.trading_days[-1]
        # one more SL/TP sweep after last sample
        n_trades += _enforce_risk_exits(portfolio, self.prices, prev_sample,
                                        final_day, run_id, self.store)
        final_value = portfolio.total_value(self.prices, final_day)
        if not equity_curve or equity_curve[-1]["date"] != final_day.isoformat():
            equity_curve.append({
                "date": final_day.isoformat(),
                "value": round(final_value, 2),
                "cash": round(portfolio.cash, 2),
            })

        spy_return = self.prices.returns_pct("SPY", self.prices.trading_days[0], final_day)
        # Benchmark-honesty guard. When yfinance fails to return SPY for a
        # window the cache persists an EMPTY SPY series (observed live: 80 of
        # 485 complete runs / 16 windows, e.g. prices_2021-08-02_2025-08-01.json
        # had SPY_rows={} while 116 other tickers loaded). `_build_trading_days`
        # silently falls back to another ticker's calendar so the run still
        # completes, but `returns_pct("SPY", …)` then returns 0.0 — so
        # `vs_spy_pct` (the documented skill metric, and the live trader's
        # `_ml_is_qualified` median-alpha gate input) becomes a fabricated
        # `total_return - 0` with NO real benchmark. The NOT NULL DEFAULT 0
        # schema (invariant #13) makes a true NULL impossible without an
        # ALTER, so flag it honestly in the additive nullable `notes` column
        # instead: purely informational, zero behaviour change to returns /
        # winner selection / the gate, but a reading quant (and the dashboard)
        # can now SEE that this run's alpha is not benchmarked.
        benchmark_note = ""
        if not (self.prices.prices.get("SPY") or {}):
            benchmark_note = (
                "benchmark_unavailable: SPY price series empty for this window "
                "(yfinance fetch failed); spy_return_pct/vs_spy_pct are NOT a "
                "valid SPY benchmark — equals raw return, do not read as alpha"
            )
            print(f"[run {run_id}] WARNING: SPY series empty for "
                  f"{self.start}→{self.end} — vs_spy_pct is not a real benchmark")
        elif spy_return == 0.0 and (final_day - self.prices.trading_days[0]).days >= 30:
            # Second-class degenerate: SPY series is non-empty but
            # returns_pct came back at EXACTLY 0.0 for a ≥30-day window. SPY
            # has no realistic flat ≥30-day stretches — across 30+ years
            # since 1993 the empirical p10..p90 of 30d SPY return is roughly
            # -5%..+5% but is essentially never exactly 0. So either:
            #   * `price_on` walk-back collapsed both endpoints to the same
            #     historical close (a degenerate trading_days[0] / final_day
            #     pair we never want to compare),
            #   * yfinance returned SPY for the window but with both end
            #     dates missing (so `price_on` falls back to identical priors),
            # — either way the resulting `vs_spy_pct = total_return - 0` is
            # fabricated alpha, identical to the empty-SPY case. Flag it the
            # same way (additive note, no behaviour change). Observed live
            # in this checkout: 80/475 complete runs hit this branch and went
            # uncaught for months because the schema's NOT NULL DEFAULT 0
            # made the degenerate spy_return indistinguishable from a real 0%.
            benchmark_note = (
                "benchmark_unavailable: SPY returns_pct=0.0 for a "
                f"{(final_day - self.prices.trading_days[0]).days}-day window "
                "(degenerate price_on walk-back collapsed both endpoints to "
                "the same close); spy_return_pct/vs_spy_pct are NOT a valid "
                "SPY benchmark — equals raw return, do not read as alpha"
            )
            print(f"[run {run_id}] WARNING: SPY returns_pct=0 across "
                  f"{(final_day - self.prices.trading_days[0]).days}d window "
                  f"{self.start}→{self.end} — vs_spy_pct is not a real benchmark")
        self.store.finalize_run(run_id, final_value, spy_return, n_trades,
                                n_decisions, equity_curve, status="complete",
                                notes=benchmark_note)

        ret_pct = (final_value - INITIAL_CASH) / INITIAL_CASH * 100
        print(f"[run {run_id}] DONE  final=${final_value:.2f}  return={ret_pct:+.2f}%  "
              f"vs SPY {spy_return:+.2f}%  trades={n_trades}")

        return BacktestRun(
            run_id=run_id, seed=seed,
            start_date=self.start.isoformat(),
            end_date=self.end.isoformat(),
            final_value=round(final_value, 2),
            total_return_pct=round(ret_pct, 2),
            spy_return_pct=round(spy_return, 2),
            vs_spy_pct=round(ret_pct - spy_return, 2),
            n_trades=n_trades, n_decisions=n_decisions,
            bubble_gate_suppressions=n_bubble_suppressed,
            equity_curve=equity_curve, status="complete",
        )

    def _warm_gdelt_cache(self) -> None:
        """Pre-fetch all date×keyword combos into disk cache before parallel runs start.

        Uses GDELT_WARM_WORKERS parallel workers so we fill the cache fast without
        hammering GDELT (each worker obeys the rate-limit via the shared lock).
        When run_all() calls this first, every subsequent thread cache-lookup is a
        disk hit — zero outbound GDELT requests during the parallel phase.
        """
        days = self._sampled_days()
        combos = [(d, kw) for d in days for kw in KEYWORD_GROUPS]
        uncached = [(d, kw) for d, kw in combos
                    if not self.gdelt._cache_key(d, kw).exists()]
        if not uncached:
            print(f"[cache_warm] all {len(combos)} combos already cached — skipping")
            return

        # Cap warming to avoid multi-hour blocking: sample the most recent uncached dates
        # first (most likely to appear in tier-3 lookups) up to the budget limit.
        import random as _rnd
        _rnd.shuffle(uncached)  # randomize so different windows get variety
        budget = min(len(uncached), GDELT_MAX_WARM_REQUESTS)
        uncached = uncached[:budget]
        print(f"[cache_warm] warming {budget} (of {len(combos)} total) combos — "
              f"sequential, 1 req/{GDELT_RATE_LIMIT_S}s")

        done = 0
        for d, kw in uncached:
            try:
                self.gdelt.fetch(d, kw)
                done += 1
                if done % 20 == 0:
                    print(f"[cache_warm] {done}/{budget} warmed")
            except Exception:
                pass
        print(f"[cache_warm] done — {done}/{budget} new entries cached")

    def _fetch_yf_news(self, tickers: list[str], sim_date: date) -> list[dict]:
        """Supplement GDELT with yfinance ticker news. No rate limits, no API key.

        Returns headlines published on or before `sim_date` scored by the keyword
        heuristic. Filters by the article's publish timestamp so a
        recent-but-still-historical sim_date cannot pick up news published after
        it (avoiding forward leakage that would contaminate the DecisionScorer
        training set). Only fetched for dates within the last 30 days because
        yfinance keeps a short retention window per ticker.

        Schema-tolerant. yfinance changed its ``Ticker.news`` payload in late
        2024: the legacy flat shape (``{title, link, providerPublishTime}``)
        was replaced with a nested ``{id, content: {title, canonicalUrl: {url},
        pubDate, …}}`` shape. The prior implementation read only the legacy
        top-level keys, so every NVDA/SPY/etc. headline silently came back as
        ``title=""`` and was dropped at the ``if not title: continue`` guard
        — Tier 2 of ``_fetch_signals`` went dark. The helper below extracts
        title/url/pub_ts from BOTH shapes (new schema wins when present,
        legacy fields are a fallback), so a future schema rollback or any
        residual provider returning the legacy shape still works. Same
        defensive try/except contract — a parse failure on one item drops
        that item, never the whole ticker's news.
        """
        cutoff = date.today() - timedelta(days=30)
        if sim_date < cutoff:
            return []
        # sim_date is a calendar day — accept news whose unix timestamp falls on
        # or before that day's end. Use UTC end-of-day for the cutoff.
        sim_end_ts = int(datetime(sim_date.year, sim_date.month, sim_date.day,
                                  23, 59, 59, tzinfo=timezone.utc).timestamp())
        articles: list[dict] = []
        seen: set[str] = set()
        sample_tickers = tickers[:8]  # limit to avoid slow fetches
        for tk in sample_tickers:
            try:
                news = yf.Ticker(tk).news or []
                for n in news[:10]:
                    try:
                        title, url, pub_ts = _parse_yf_news_item(n)
                    except Exception:
                        continue
                    # Skip dedup for empty URLs — otherwise a single empty
                    # string in `seen` silently swallows EVERY subsequent
                    # URL-less yfinance article (only the first survives),
                    # even when their titles carry distinct signals. Mirrors
                    # the same idiom already used in `_fetch_signals`.
                    if not title:
                        continue
                    if url and url in seen:
                        continue
                    if (isinstance(pub_ts, (int, float))
                            and pub_ts > sim_end_ts):
                        continue  # future news — would leak forward into training
                    if url:
                        seen.add(url)
                    score, found_tickers = score_article({"title": title, "url": url})
                    articles.append({"title": title, "url": url,
                                     "score": score, "tickers": found_tickers})
            except Exception:
                pass
        return articles

    def run_all(self, n: int = 10, start_run_id: int = 1) -> list[BacktestRun]:
        import traceback
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Warm GDELT cache in background — doesn't block parallel runs.
        # Runs use local DB (tier 1) immediately; GDELT cache fills in over time.
        threading.Thread(target=self._warm_gdelt_cache, daemon=True,
                         name="gdelt-cache-warm").start()

        spy_return = self.prices.returns_pct("SPY", self.prices.trading_days[0],
                                             self.prices.trading_days[-1])
        print(f"[engine] SPY baseline {self.prices.trading_days[0]} → "
              f"{self.prices.trading_days[-1]}: {spy_return:+.2f}%")
        # Print persona map so the run log is self-describing.
        print(f"[engine] Launching {n} runs starting at run_id={start_run_id}")
        for i in range(start_run_id, start_run_id + n):
            p = persona_for(i)
            print(f"[engine]   run_id={i} persona={p['name']}")

        results: list[BacktestRun] = []
        completed = 0

        def _run(i: int):
            try:
                return self.run_one(i)
            except Exception as e:
                print(f"[engine] RUN {i} CRASHED: {e}")
                traceback.print_exc()
                self.store.upsert_run(i, 0, "failed",
                                      start=self.start, end=self.end,
                                      model_id=self.model_id)
                return None

        with ThreadPoolExecutor(max_workers=n) as pool:
            futures = {pool.submit(_run, i): i
                       for i in range(start_run_id, start_run_id + n)}
            for fut in as_completed(futures):
                run_id = futures[fut]
                result = fut.result()
                completed += 1
                if result is not None:
                    results.append(result)
                print(f"[engine] {completed}/{n} runs finished (run_id={run_id})")
                if completed % 2 == 0:
                    self._send_progress(completed, n, results, spy_return)

        self._send_final(results, spy_return)
        return results

    def _send_progress(self, done: int, total: int, results: list[BacktestRun],
                       spy: float) -> None:
        pass  # silent — check dashboard at :8090/backtests

    def _send_final(self, results: list[BacktestRun], spy: float) -> None:
        pass  # silent — check dashboard at :8090/backtests


if __name__ == "__main__":
    BacktestEngine().run_all(10)
