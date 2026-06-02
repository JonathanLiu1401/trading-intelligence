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
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import yfinance as yf

from core.claude_cli import claude_call

from .strategy import (
    MODEL,
    CASH_RESERVE,
    MAX_POSITION_PCT,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
)

ROOT = Path(__file__).resolve().parent.parent
BACKTEST_DB = ROOT / "backtest.db"
CACHE_DIR = ROOT / "data" / "backtest_cache"
GDELT_CACHE = CACHE_DIR / "gdelt"
PRICE_CACHE_PATH = CACHE_DIR / "prices.json"

START_DATE = date(2025, 5, 13)
END_DATE = date(2026, 5, 13)
INITIAL_CASH = 1000.0
SAMPLE_EVERY_N_DAYS = 5
GDELT_RATE_LIMIT_S = 5.5
GDELT_MAX_RECORDS = 100
OPUS_TIMEOUT_S = 150

WATCHLIST = [
    "SPY", "QQQ", "NVDA", "AMD", "MU", "LITE", "AMAT", "LRCX",
    "SMH", "TSM", "INTC", "QCOM", "AAPL", "MSFT", "META", "GOOGL",
    "AMZN", "BTC-USD", "GC=F",
]
# LNOK is a thin OTC name and yfinance often returns nothing → omitted from default fetch.

KEYWORD_GROUPS = [
    "stock market earnings semiconductor",
    "NVDA AMD MU earnings revenue",
    "Federal Reserve interest rates inflation",
    "S&P 500 market rally selloff",
    "Micron DRAM memory chip supply",
    "Lumentum photonics optical",
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
    def __init__(self, path: Path = BACKTEST_DB):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), timeout=30, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def upsert_run(self, run_id: int, seed: int, status: str) -> None:
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
                (run_id, seed, START_DATE.isoformat(), END_DATE.isoformat(),
                 INITIAL_CASH, status, now),
            )
        self.conn.commit()

    def finalize_run(self, run_id: int, final_value: float, spy_return_pct: float,
                     n_trades: int, n_decisions: int, equity_curve: list,
                     status: str = "complete", notes: str = "") -> None:
        total_return_pct = (final_value - INITIAL_CASH) / INITIAL_CASH * 100
        vs_spy = total_return_pct - spy_return_pct
        self.conn.execute(
            "UPDATE backtest_runs SET final_value=?, total_return_pct=?, "
            "spy_return_pct=?, vs_spy_pct=?, n_trades=?, n_decisions=?, "
            "equity_curve_json=?, status=?, completed_at=?, notes=? WHERE run_id=?",
            (final_value, total_return_pct, spy_return_pct, vs_spy,
             n_trades, n_decisions, json.dumps(equity_curve), status,
             datetime.now(timezone.utc).isoformat(), notes, run_id),
        )
        self.conn.commit()

    def record_trade(self, run_id: int, sim_date: str, ticker: str, action: str,
                     qty: float, price: float, reason: str) -> None:
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
        self.conn.execute(
            "INSERT INTO backtest_decisions (run_id, sim_date, action, ticker, qty, "
            "confidence, reasoning, status, detail, cash, total_value, signal_count) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, sim_date, d.get("action"), d.get("ticker"), d.get("qty"),
             d.get("confidence"), d.get("reasoning"), status, detail, cash,
             total_value, signal_count),
        )
        self.conn.commit()

    def all_runs(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM backtest_runs ORDER BY run_id ASC"
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["equity_curve"] = json.loads(d.pop("equity_curve_json") or "[]")
            except Exception:
                d["equity_curve"] = []
            out.append(d)
        return out

    def run_detail(self, run_id: int) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM backtest_runs WHERE run_id=?", (run_id,)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        try:
            d["equity_curve"] = json.loads(d.pop("equity_curve_json") or "[]")
        except Exception:
            d["equity_curve"] = []
        trades = self.conn.execute(
            "SELECT * FROM backtest_trades WHERE run_id=? ORDER BY sim_date ASC, id ASC",
            (run_id,),
        ).fetchall()
        decisions = self.conn.execute(
            "SELECT * FROM backtest_decisions WHERE run_id=? ORDER BY sim_date ASC, id ASC",
            (run_id,),
        ).fetchall()
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

    def _load(self) -> None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if PRICE_CACHE_PATH.exists():
            try:
                cached = json.loads(PRICE_CACHE_PATH.read_text())
                meta = cached.get("_meta", {})
                if (meta.get("start") == self.start.isoformat()
                        and meta.get("end") == self.end.isoformat()
                        and set(meta.get("tickers", [])) >= set(self.tickers)):
                    self.prices = {k: v for k, v in cached.items() if k != "_meta"}
                    self._build_trading_days()
                    print(f"[price_cache] loaded {len(self.prices)} tickers from cache "
                          f"({len(self.trading_days)} trading days)")
                    return
            except Exception as e:
                print(f"[price_cache] cache read failed: {e}")

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
        PRICE_CACHE_PATH.write_text(json.dumps(payload))
        self._build_trading_days()
        print(f"[price_cache] saved → {PRICE_CACHE_PATH} "
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


# ─────────────────────────── GDELT fetcher ───────────────────────────

class GDELTFetcher:
    """Cached, rate-limited GDELT article fetcher."""

    def __init__(self):
        GDELT_CACHE.mkdir(parents=True, exist_ok=True)
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

        # rate limit
        elapsed = time.time() - self._last_request_ts
        if elapsed < GDELT_RATE_LIMIT_S:
            time.sleep(GDELT_RATE_LIMIT_S - elapsed)

        start_dt = f"{d.strftime('%Y%m%d')}000000"
        end_dt = f"{d.strftime('%Y%m%d')}235959"
        params = urllib.parse.urlencode({
            "query": keywords,
            "mode": "artlist",
            "maxrecords": GDELT_MAX_RECORDS,
            "startdatetime": start_dt,
            "enddatetime": end_dt,
            "format": "json",
            "sort": "hybridrel",
        })
        url = f"https://api.gdeltproject.org/api/v2/doc/doc?{params}"
        articles: list[dict] = []
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "paper-trader-backtest/1.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            self._last_request_ts = time.time()
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                # GDELT sometimes returns plain-text rate-limit messages
                print(f"[gdelt] non-JSON response for {d} {keywords[:30]!r}: {body[:120]!r}")
                data = {}
            articles = data.get("articles", []) or []
        except Exception as e:
            print(f"[gdelt] fetch failed {d} {keywords[:30]!r}: {e}")
            self._last_request_ts = time.time()

        try:
            path.write_text(json.dumps(articles))
        except Exception:
            pass
        return articles


# ─────────────────────────── heuristic scorer ───────────────────────────

_TICKER_RE = re.compile(r"\b([A-Z]{2,5})\b")
_NOT_TICKERS = {
    "AI", "AND", "FOR", "THE", "WITH", "FROM", "AFTER", "INTO", "HAVE", "WILL",
    "MAY", "JUNE", "JULY", "AUG", "SEPT", "OCT", "NOV", "DEC", "CEO", "ETF",
    "USA", "USD", "GDP", "CPI", "OPEC", "FED", "FOMC", "PMI", "ISM", "WHO",
    "NEW", "OLD", "ALL", "YES", "USA", "ITS", "OUR", "ONE", "TWO", "AND",
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
    """Walk daily closes between from_day (exclusive) and to_day (inclusive).
    Trip stop-loss / take-profit at first breach. Returns # of exits."""
    n = 0
    if not portfolio.positions:
        return 0
    cur = from_day + timedelta(days=1)
    while cur <= to_day:
        if not portfolio.positions:
            break
        # Skip non-trading days
        if cur not in prices.trading_days:
            cur += timedelta(days=1)
            continue
        for ticker in list(portfolio.positions.keys()):
            pos = portfolio.positions[ticker]
            px = prices.price_on(ticker, cur)
            if px is None:
                continue
            pl_pct = (px - pos["avg_cost"]) / pos["avg_cost"]
            # explicit stop / take from Opus
            sl = pos.get("stop_loss")
            tp = pos.get("take_profit")
            exited = False
            if sl and px <= sl:
                qty = pos["qty"]
                _sell(portfolio, ticker, qty, px)
                store.record_trade(run_id, cur.isoformat(), ticker, "SELL", qty, px,
                                   f"stop-loss @ {sl} (close {px:.2f})")
                n += 1
                exited = True
            elif tp and px >= tp:
                qty = pos["qty"]
                _sell(portfolio, ticker, qty, px)
                store.record_trade(run_id, cur.isoformat(), ticker, "SELL", qty, px,
                                   f"take-profit @ {tp} (close {px:.2f})")
                n += 1
                exited = True
            # default rules
            elif pl_pct <= STOP_LOSS_PCT:
                qty = pos["qty"]
                _sell(portfolio, ticker, qty, px)
                store.record_trade(run_id, cur.isoformat(), ticker, "SELL", qty, px,
                                   f"auto stop-loss {pl_pct*100:.1f}%")
                n += 1
                exited = True
            elif pl_pct >= TAKE_PROFIT_PCT and pos["qty"] > 0.0002:
                half = round(pos["qty"] / 2, 4)
                if half > 0:
                    _sell(portfolio, ticker, half, px)
                    store.record_trade(run_id, cur.isoformat(), ticker, "SELL", half, px,
                                       f"auto take-profit trim {pl_pct*100:.1f}%")
                    n += 1
            if exited:
                continue
        cur += timedelta(days=1)
    return n


# ─────────────────────────── Opus call ───────────────────────────

SYSTEM_PROMPT = """You are an aggressive but disciplined paper-trading desk running a $1000 simulated account.
This is a HISTORICAL backtest — you are deciding trades for a specific past date based on news available at that date.
Your goal: maximize total return over a 1-year horizon. Trade US stocks ONLY (no options or futures in this backtest).

Respond with a SINGLE JSON object — no prose, no markdown fences. Schema:

{
  "action": "BUY" | "SELL" | "HOLD",
  "ticker": "NVDA",
  "qty": 0.5,
  "confidence": 0.85,
  "reasoning": "1-3 sentences why",
  "stop_loss": 850.0,
  "take_profit": 950.0
}

Rules:
- HOLD when signal quality is low. Patience is alpha.
- Size by conviction. Confidence >= 0.8 → larger size. Below 0.5 → HOLD.
- Never bet the whole account on one trade. Position cap is 40% of equity.
- Never go below $100 cash.
- For SELL, ticker must match an open position.
- Set stop_loss / take_profit (absolute prices) when you have conviction on a target.
- Fractional shares are allowed (qty can be e.g. 0.5).

Return JSON ONLY.
"""


def _claude_call(prompt: str, retries: int = 1) -> str | None:
    for attempt in range(retries + 1):
        result = claude_call(prompt, model=MODEL, timeout=OPUS_TIMEOUT_S)
        if result:
            return result
        print(f"[backtest] llm attempt {attempt+1} returned no response")
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


def _build_prompt(run_id: int, seed: int, sim_date: date, portfolio: SimPortfolio,
                  top_articles: list[dict], prices: PriceCache) -> str:
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
        p = prices.price_on(t, sim_date)
        px_lines.append(f"  {t}: ${p:.2f}" if p else f"  {t}: N/A")

    total = portfolio.total_value(prices, sim_date)

    return f"""{SYSTEM_PROMPT}

---
SIMULATION CONTEXT:
You are simulation run #{run_id}/10 with seed {seed}. Explore your own trading style;
prior runs may have made different choices and that is intentional.

SIMULATED DATE: {sim_date.isoformat()}

PORTFOLIO:
  cash: ${portfolio.cash:.2f}
  total value: ${total:.2f}
  positions:
{chr(10).join(pos_lines) if pos_lines else '  (none)'}

WATCHLIST CLOSES on {sim_date.isoformat()}:
{chr(10).join(px_lines)}

TOP NEWS SIGNALS for {sim_date.isoformat()} (score 0..5 from keyword heuristic):
{chr(10).join(art_lines) if art_lines else '  (no signals)'}

Return JSON only.
"""


# ─────────────────────────── engine ───────────────────────────

@dataclass
class BacktestRun:
    run_id: int
    seed: int
    start_date: str = START_DATE.isoformat()
    end_date: str = END_DATE.isoformat()
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


class BacktestEngine:
    def __init__(self):
        self.store = BacktestStore()
        self.prices = PriceCache(WATCHLIST, START_DATE, END_DATE)
        self.gdelt = GDELTFetcher()
        if not self.prices.trading_days:
            raise RuntimeError("PriceCache has no trading days — yfinance fetch failed")

    def _sampled_days(self) -> list[date]:
        days = self.prices.trading_days
        return days[::SAMPLE_EVERY_N_DAYS]

    def _fetch_signals(self, d: date, seed: int, rng: random.Random) -> list[dict]:
        # rotate 2 keyword groups based on seed/day so different runs see different slices
        idxs = rng.sample(range(len(KEYWORD_GROUPS)), 2)
        articles: list[dict] = []
        seen_urls = set()
        for i in idxs:
            kw = KEYWORD_GROUPS[i]
            for a in self.gdelt.fetch(d, kw):
                url = a.get("url", "")
                if url in seen_urls or not a.get("title"):
                    continue
                seen_urls.add(url)
                score, tickers = score_article(a)
                articles.append({
                    "title": a["title"],
                    "url": url,
                    "score": score,
                    "tickers": tickers,
                })
        # sort by score, take top 10 then sample 5 with rng
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
            total = portfolio.total_value(self.prices, sim_date)
            if portfolio.cash - notional < CASH_RESERVE:
                return "BLOCKED", f"would breach cash floor (need ${notional:.2f})"
            if notional > total * MAX_POSITION_PCT:
                return "BLOCKED", f"position would exceed {int(MAX_POSITION_PCT*100)}% cap"
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
        self.store.upsert_run(run_id, seed, "running")
        print(f"\n══════ RUN {run_id}/10  seed={seed} ══════")

        portfolio = SimPortfolio()
        equity_curve: list[dict] = []
        n_trades = 0
        n_decisions = 0
        prev_sample = self.prices.trading_days[0] - timedelta(days=1)
        last_curve_day: date | None = None

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

            # fetch & score
            signals = self._fetch_signals(sim_date, seed, rng)

            # build prompt + call Opus
            prompt = _build_prompt(run_id, seed, sim_date, portfolio, signals, self.prices)
            raw = _claude_call(prompt)
            decision = _parse_decision(raw)
            n_decisions += 1

            if decision:
                status, detail = self._execute_decision(run_id, sim_date, decision, portfolio)
                if status == "FILLED":
                    n_trades += 1
            else:
                status, detail = "NO_DECISION", "claude returned no parseable JSON"

            total = portfolio.total_value(self.prices, sim_date)
            self.store.record_decision(run_id, sim_date.isoformat(), decision,
                                       status, detail, portfolio.cash, total,
                                       len(signals))

            # weekly equity snapshot
            if last_curve_day is None or (sim_date - last_curve_day).days >= 7:
                equity_curve.append({
                    "date": sim_date.isoformat(),
                    "value": round(total, 2),
                    "cash": round(portfolio.cash, 2),
                })
                last_curve_day = sim_date

            if idx % 10 == 0 or idx == len(sampled) - 1:
                print(f"  [run {run_id} {idx+1}/{len(sampled)}] {sim_date} "
                      f"action={status} cash=${portfolio.cash:.2f} total=${total:.2f}")

        # final mark
        final_day = self.prices.trading_days[-1]
        # one more SL/TP sweep after last sample
        _enforce_risk_exits(portfolio, self.prices, prev_sample, final_day,
                            run_id, self.store)
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
            final_value=round(final_value, 2),
            total_return_pct=round(ret_pct, 2),
            spy_return_pct=round(spy_return, 2),
            vs_spy_pct=round(ret_pct - spy_return, 2),
            n_trades=n_trades, n_decisions=n_decisions,
            equity_curve=equity_curve, status="complete",
        )

    def run_all(self, n: int = 10) -> list[BacktestRun]:
        results: list[BacktestRun] = []
        spy_return = self.prices.returns_pct("SPY", self.prices.trading_days[0],
                                             self.prices.trading_days[-1])
        print(f"[engine] SPY baseline {self.prices.trading_days[0]} → "
              f"{self.prices.trading_days[-1]}: {spy_return:+.2f}%")

        for i in range(1, n + 1):
            try:
                run = self.run_one(i)
                results.append(run)
            except Exception as e:
                import traceback
                print(f"[engine] RUN {i} CRASHED: {e}")
                traceback.print_exc()
                self.store.upsert_run(i, 0, "failed")
                continue

            if i % 2 == 0:
                self._send_progress(i, n, results, spy_return)

        self._send_final(results, spy_return)
        return results

    def _send_progress(self, done: int, total: int, results: list[BacktestRun],
                       spy: float) -> None:
        if not results:
            return
        last = results[-1]
        msg = (f"[Backtest] Run {done}/{total} complete. "
               f"Final: ${last.final_value:.2f} "
               f"({last.total_return_pct:+.2f}% vs SPY {spy:+.2f}%)")
        self._discord(msg)

    def _send_final(self, results: list[BacktestRun], spy: float) -> None:
        if not results:
            self._discord("[Backtest Complete] all runs failed")
            return
        avg_return = sum(r.total_return_pct for r in results) / len(results)
        avg_final = sum(r.final_value for r in results) / len(results)
        best = max(results, key=lambda r: r.final_value)
        worst = min(results, key=lambda r: r.final_value)
        msg = (f"[Backtest Complete] {len(results)}/10 runs done. "
               f"avg ${avg_final:.2f} ({avg_return:+.2f}%), "
               f"best ${best.final_value:.2f} ({best.total_return_pct:+.2f}%), "
               f"worst ${worst.final_value:.2f} ({worst.total_return_pct:+.2f}%). "
               f"SPY baseline {spy:+.2f}%. "
               f"Dashboard: http://localhost:8090/backtests")
        self._discord(msg)

    def _discord(self, message: str) -> bool:
        if not shutil.which("openclaw"):
            print(f"[discord] (no openclaw) {message}")
            return False
        try:
            r = subprocess.run(
                ["openclaw", "message", "send", "--channel", "discord",
                 "--target", "channel:1496099475838603324", "--message", message],
                capture_output=True, text=True, timeout=60,
            )
            if r.returncode != 0:
                print(f"[discord] failed: {r.stderr.strip()[:200]}")
                return False
            print(f"[discord] sent: {message[:100]}")
            return True
        except Exception as e:
            print(f"[discord] exception: {e}")
            return False


if __name__ == "__main__":
    BacktestEngine().run_all(10)
