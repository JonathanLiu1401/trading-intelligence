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
# Global semaphore: caps concurrent claude CLI subprocesses to prevent OOM kills.
# 10 parallel runs × ~1.5 GB/process = OOM on 14 GB RAM. Cap at 3 concurrent.
_CLAUDE_SEM = threading.Semaphore(3)

WATCHLIST = [
    # Core US large-cap + semis (kept from v1 watchlist)
    "SPY", "QQQ", "NVDA", "AMD", "MU", "LITE", "AMAT", "LRCX",
    "SMH", "TSM", "INTC", "QCOM", "AAPL", "MSFT", "META", "GOOGL",
    "AMZN", "TSLA", "CRM", "SNOW", "BTC-USD", "GC=F",
    # Global / ADR
    "BABA", "ASML", "SAP", "NVO", "TM", "SONY", "HSBC", "BP", "RIO", "BHP",
    # US financials
    "GS", "JPM", "BAC", "BRK-B",
    # Energy / healthcare / payments
    "XOM", "CVX", "LLY", "UNH", "V", "MA",
    # Fintech / crypto-adjacent / speculative
    "SHOP", "SQ", "COIN", "MSTR", "PLTR", "RIVN", "NIO", "ARKK",
    # Macro / commodity ETFs
    "TLT", "GLD", "SLV", "USO", "UNG",
    # Leveraged ETFs — 3x Bull
    "TQQQ", "UPRO", "SPXL", "UDOW", "URTY",        # index 3x
    "SOXL", "TECL", "FNGU", "CURE", "LABU",         # sector 3x (semis/tech/health/bio)
    "NAIL", "WANT", "DFEN", "MIDU", "TNA",           # housing/China/defense/mid/small 3x
    "DPST", "FAS", "HIBL", "UTSL",                   # banks/financials/high-beta/utilities 3x
    # Leveraged ETFs — 2x Bull
    "QLD", "SSO", "MVV", "SAA", "UWM",               # index 2x
    "NVDU", "MSFU", "AMZU", "GOOGU", "METAU",        # single-stock 2x (NVDA/MSFT/AMZN/GOOG/META)
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
_LEVERAGED_ETFS = {
    "SOXL", "TQQQ", "UPRO", "SPXL", "UDOW", "URTY", "TECL", "FNGU", "CURE", "LABU",
    "NAIL", "DFEN", "DPST", "FAS", "HIBL",
    "QLD", "SSO", "MVV", "NVDU", "MSFU", "AMZU", "TSLT", "TSLL",
    "LNOK", "BOIL", "UCO", "AGQ", "BITU",
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

    def upsert_run(self, run_id: int, seed: int, status: str,
                   start: date, end: date) -> None:
        with self._lock:
            existing = self.conn.execute(
                "SELECT run_id FROM backtest_runs WHERE run_id=?", (run_id,)
            ).fetchone()
            now = datetime.now(timezone.utc).isoformat()
            if existing:
                self.conn.execute(
                    "UPDATE backtest_runs SET status=? WHERE run_id=?", (status, run_id)
                )
            else:
                self.conn.execute(
                    "INSERT INTO backtest_runs (run_id, seed, start_date, end_date, "
                    "start_value, status, started_at) VALUES (?,?,?,?,?,?,?)",
                    (run_id, seed, start.isoformat(), end.isoformat(),
                     INITIAL_CASH, status, now),
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
            rows = self.conn.execute(
                "SELECT * FROM backtest_runs ORDER BY run_id ASC"
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
        placeholders = ",".join("?" * len(run_ids))
        with self._lock:
            rows = self.conn.execute(
                f"SELECT run_id, start_date, start_value, equity_curve_json "
                f"FROM backtest_runs WHERE run_id IN ({placeholders})",
                run_ids,
            ).fetchall()
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
                    self.prices = {k: v for k, v in cached.items() if k != "_meta"}
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
                    series[iso] = float(close)
                self.prices[t] = series
                print(f"[price_cache]   {t}: {len(series)} rows")
            except Exception as e:
                print(f"[price_cache]   {t} failed: {e}")
                self.prices[t] = {}

        payload = {"_meta": {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "tickers": list(self.prices.keys()),
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }}
        payload.update(self.prices)
        out_path = self.cache_path
        out_path.write_text(json.dumps(payload))
        self._build_trading_days()
        print(f"[price_cache] saved → {out_path} "
              f"({len(self.trading_days)} trading days)")

    def _build_trading_days(self) -> None:
        spy = self.prices.get("SPY") or {}
        if not spy:
            # fallback: any ticker
            for t in self.prices:
                if self.prices[t]:
                    spy = self.prices[t]
                    break
        days = sorted(date.fromisoformat(d) for d in spy.keys()
                      if self.start <= date.fromisoformat(d) <= self.end)
        self.trading_days = days

    def price_on(self, ticker: str, d: date) -> float | None:
        """Close on `d` if available; else most recent prior close."""
        series = self.prices.get(ticker)
        if not series:
            return None
        iso = d.isoformat()
        if iso in series:
            return series[iso]
        # walk back up to 7 days
        for delta in range(1, 8):
            prior = (d - timedelta(days=delta)).isoformat()
            if prior in series:
                return series[prior]
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
# Track which (start, end) windows we've already loaded from disk into memory.
_VOLUME_CACHE_DISK_LOADED: set[tuple[str, str]] = set()


def _volume_cache_path(start: date, end: date) -> Path:
    return CACHE_DIR / f"volumes_{start.isoformat()}_{end.isoformat()}.json"


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
            return
        path = _volume_cache_path(start, end)
        loaded: dict[str, dict[str, float]] = {}
        if path.exists():
            try:
                loaded = json.loads(path.read_text())
            except Exception:
                loaded = {}
        for ticker, series in loaded.items():
            _VOLUME_CACHE[(ticker, key[0], key[1])] = series
        _VOLUME_CACHE_DISK_LOADED.add(key)


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
        path.write_text(json.dumps(flat))
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
        return 100.0
    rs = avg_g / avg_l
    return 100.0 - (100.0 / (1.0 + rs))


def _macd(closes: list[float]) -> tuple[str, float, float] | None:
    """Return (label, macd, signal). label is 'bullish'/'bearish'/'flat'."""
    if len(closes) < 35:
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
    label = "bullish" if m > s else "bearish" if m < s else "flat"
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
                return json.loads(path.read_text())
            except Exception:
                pass

        from gdeltdoc import Filters
        from gdeltdoc.errors import RateLimitError
        start_str = d.strftime("%Y-%m-%d")
        end_str = (d + timedelta(days=1)).strftime("%Y-%m-%d")

        articles: list[dict] = []
        success = False
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
            if success:
                break
            backoff = GDELT_RETRY_BACKOFF_S * (attempt + 1)
            print(f"[gdelt] {err} {d} {keywords[:30]!r} "
                  f"attempt {attempt+1}/3 — sleeping {backoff:.0f}s")
            time.sleep(backoff)

        if success:
            try:
                path.write_text(json.dumps(articles))
            except Exception:
                pass
        return articles


# ─────────────────────────── Alpha Vantage news fetcher ───────────────────────────

AV_CACHE_DIR = CACHE_DIR / "alphavantage"
AV_QUOTA_PATH = CACHE_DIR / "av_quota.json"
AV_MAX_DAILY = 22  # stay under 25/day limit with margin


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
            AV_QUOTA_PATH.write_text(json.dumps(q))

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
                    articles.extend(json.loads(path.read_text()))
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
                path.write_text(json.dumps(items))
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
    "dividend", "acqui", "bullish", "higher", "raise", "raised", "exceed",
}
_BEARISH_WORDS = {
    "miss", "misses", "plunge", "plunges", "downgrade", "downgraded", "cut",
    "cuts", "layoff", "layoffs", "loss", "losses", "warning", "shortfall",
    "selloff", "sell", "underperform", "weak", "decline", "declines", "crash",
    "lower", "reduce", "reduced", "concern", "concerns", "risk",
}
_WORD_TO_TICKER: dict[str, str] = {
    # Tech / semis — map bullish tech headlines straight to leveraged ETFs
    "nvidia": "NVDA", "amd": "AMD", "apple": "AAPL", "microsoft": "MSFT",
    "amazon": "AMZN", "google": "GOOGL", "alphabet": "GOOGL", "meta": "META",
    "tesla": "TSLA", "intel": "INTC", "micron": "MU", "broadcom": "AVGO",
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


def _article_sentiment(title: str) -> float:
    """Return -1..+1 based on bullish/bearish keyword count in title."""
    words = set(title.lower().split())
    bull = sum(1 for w in words if any(w.startswith(b) for b in _BULLISH_WORDS))
    bear = sum(1 for w in words if any(w.startswith(b) for b in _BEARISH_WORDS))
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
        for word, sym in _WORD_TO_TICKER.items():
            if word in title_lower and sym not in tickers:
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
        q_buy = quant.get(buy_ticker, {})
        buy_news_count = ticker_article_count.get(buy_ticker, 0)
        buy_news_urg = ticker_max_urgency.get(buy_ticker, 0.0)
        _scorer = _get_decision_scorer()
        scorer_pred = _scorer.predict(
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
        )
        _scorer_n = getattr(_scorer, "_n_train", 0)
        if _scorer.is_trained and _scorer_n >= 500:
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
        scorer_note = f" scorer={scorer_pred:+.1f}%" if _scorer.is_trained else ""
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
        if stop_loss:
            existing["stop_loss"] = stop_loss
        if take_profit:
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
    if not shutil.which("claude"):
        print("[backtest] claude CLI not found")
        return None
    with _CLAUDE_SEM:  # max 3 concurrent claude processes to avoid OOM
        for attempt in range(retries + 1):
            try:
                r = subprocess.run(
                    ["claude", "--model", MODEL, "--print",
                     "--permission-mode", "bypassPermissions"],
                    input=prompt, capture_output=True, text=True,
                    timeout=OPUS_TIMEOUT_S,
                )
                if r.returncode == 0 and r.stdout.strip():
                    return r.stdout.strip()
                print(f"[backtest] claude attempt {attempt+1} returncode={r.returncode} "
                      f"err={r.stderr.strip()[:200]!r}")
            except subprocess.TimeoutExpired:
                print(f"[backtest] claude timeout attempt {attempt+1}")
            except Exception as e:
                print(f"[backtest] claude exception attempt {attempt+1}: {e}")
            if attempt < retries:
                time.sleep(2)
    return None


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
    status: str = "pending"
    trades: list[dict] = field(default_factory=list)
    decisions: list[dict] = field(default_factory=list)
    equity_curve: list[dict] = field(default_factory=list)


LOCAL_ARTICLES_DB = Path(__file__).resolve().parent.parent.parent / "digital-intern" / "data" / "articles.db"


class BacktestEngine:
    def __init__(self, start: date | None = None, end: date | None = None):
        # Standalone runs (e.g. `python3 run_backtests.py`) get a sane default
        # equal to the pre-refactor hardcoded window so the one-shot launcher
        # keeps working without arg-plumbing changes. Continuous loop overrides.
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
            ).fetchall()
            # Hindsight label filter: an article whose `first_seen` lags
            # `published` by more than this many days was almost certainly
            # collected retroactively (e.g. by backfill_news.py / GDELT) and
            # then labeled by Claude with knowledge of what happened after
            # publication. Trusting that `ai_score` for a historical backtest
            # is silent lookahead — fall back to the keyword baseline instead.
            LABEL_STALENESS_DAYS = 60
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

            for (title, url, source, published, first_seen, ai_score,
                 kw_score, full_text, urgency) in rows:
                pub_d = _parse_day(published)
                if pub_d is None:
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

                if hindsight_contaminated:
                    score = float(kw_score or 0)
                else:
                    score = float(ai_score or kw_score or 0)

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
            print(f"[local_news] loaded {sum(len(v) for v in result.values())} articles "
                  f"across {len(result)} days from local DB")
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
            for e in entries or []:
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
                              start=self.start, end=self.end)
        print(f"\n══════ RUN {run_id}  seed={seed} window={self.start}→{self.end} ══════")

        portfolio = SimPortfolio()
        equity_curve: list[dict] = []
        n_trades = 0
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

            # Intraday loop: up to MAX_DECISIONS_PER_DAY ml_decide calls per day.
            # Each filled trade excludes that ticker from subsequent calls today.
            traded_today: set[str] = set()
            day_filled = 0
            status, detail = "NO_DECISION", "ml_decide returned None"
            decision = None
            for _intra in range(MAX_DECISIONS_PER_DAY):
                decision = _ml_decide(sim_date, portfolio, signals, self.prices,
                                      run_id, rng, exclude_tickers=traded_today)
                n_decisions += 1

                if not decision:
                    break
                if decision.get("action", "HOLD") == "HOLD":
                    status, detail = "HOLD", str(decision.get("reasoning", ""))[:200]
                    break

                status, detail = self._execute_decision(run_id, sim_date, decision, portfolio)
                ticker_acted = decision.get("ticker") or ""
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
        self.store.finalize_run(run_id, final_value, spy_return, n_trades,
                                n_decisions, equity_curve, status="complete")

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
        heuristic. Filters by `providerPublishTime` so a recent-but-still-historical
        sim_date cannot pick up news published after it (avoiding forward leakage
        that would contaminate the DecisionScorer training set).
        Only fetched for dates within the last 30 days because yfinance keeps a
        short retention window per ticker.
        """
        cutoff = date.today() - timedelta(days=30)
        if sim_date < cutoff:
            return []
        # sim_date is a calendar day — accept news whose unix timestamp falls on
        # or before that day's end. Use UTC end-of-day for the cutoff.
        sim_end_ts = int(datetime(sim_date.year, sim_date.month, sim_date.day,
                                  23, 59, 59, tzinfo=timezone.utc).timestamp())
        articles: list[dict] = []
        seen = set()
        sample_tickers = tickers[:8]  # limit to avoid slow fetches
        for tk in sample_tickers:
            try:
                news = yf.Ticker(tk).news or []
                for n in news[:10]:
                    title = n.get("title", "")
                    url = n.get("link", "")
                    pub_ts = n.get("providerPublishTime")
                    if not title or url in seen:
                        continue
                    if isinstance(pub_ts, (int, float)) and pub_ts > sim_end_ts:
                        continue  # future news — would leak forward into training
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
                                      start=self.start, end=self.end)
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
