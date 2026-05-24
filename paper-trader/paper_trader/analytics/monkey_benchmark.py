"""
10,000-monkey random trading benchmark — gold-standard edge validator.

Two simulation modes:

1. random_portfolio: pick K random tickers from the watchlist, equal-weight
   buy-and-hold for the entire window. Fast, pure stock-selection alpha test.
2. active_random: each trading day, each monkey has a p_buy chance to invest
   in a random ticker and a p_sell chance to liquidate, with a max_positions
   cap. Tests timing + selection alpha together.

Both modes load prices from the existing per-window PriceCache files in
``data/backtest_cache/prices_<start>_<end>.json`` — no new downloads. The
cache layout is ``{"_meta": {...}, "<ticker>": {"<iso_date>": <close>, ...}}``.

AI-vs-monkey rankings restrict comparison to backtest runs whose
``(start_date, end_date)`` exactly matches the monkey window — comparing AI
runs from 2004→2014 against monkeys simulated on 2020→2025 would be
meaningless.
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────
_BASE = Path(__file__).resolve().parent.parent.parent  # paper-trader root
_CACHE_DIR = _BASE / "data" / "backtest_cache"
_CACHE_OUT = _BASE / "data" / "monkey_benchmark.json"

# ── constants ──────────────────────────────────────────────────────────────
N_MONKEYS = 10_000
RANDOM_SEED = 42
INITIAL_CASH = 1_000.0

_WINDOW_FILE_RE = re.compile(r"^prices_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})\.json$")


# ── helpers ────────────────────────────────────────────────────────────────

def list_available_windows() -> list[tuple[str, str, int]]:
    """Return (start, end, file_size_bytes) for each per-window price cache.
    Sorted by size descending (largest = most data, preferred default)."""
    out: list[tuple[str, str, int]] = []
    if not _CACHE_DIR.exists():
        return out
    for f in _CACHE_DIR.glob("prices_*.json"):
        m = _WINDOW_FILE_RE.match(f.name)
        if not m:
            continue
        start, end = m.groups()
        out.append((start, end, f.stat().st_size))
    out.sort(key=lambda x: -x[2])
    return out


def default_window() -> tuple[str, str]:
    """Return the most data-rich cached window (start, end). Falls back to
    the BacktestEngine default if no caches exist."""
    windows = list_available_windows()
    if windows:
        s, e, _ = windows[0]
        return s, e
    return "2025-05-01", "2026-05-13"


def _load_prices(
    start_date: str,
    end_date: str,
    tickers: list[str],
) -> dict[str, list[float]]:
    """Load aligned daily closes from the per-window PriceCache JSON.

    Picks the cache file whose meta window exactly matches (start, end), or
    falls back to the largest file that fully contains the requested range.

    Returns ``{ticker: [close0, close1, ...]}`` aligned to the trading-day
    calendar derived from the most-common-date intersection. Only includes
    tickers with ≥10 prices in the window.
    """
    if not _CACHE_DIR.exists():
        raise FileNotFoundError(f"PriceCache dir not found: {_CACHE_DIR}")

    candidate: Path | None = None
    exact = _CACHE_DIR / f"prices_{start_date}_{end_date}.json"
    if exact.exists():
        candidate = exact
    else:
        # Fall back: largest cache whose window covers [start, end]
        best_size = -1
        for f in _CACHE_DIR.glob("prices_*.json"):
            m = _WINDOW_FILE_RE.match(f.name)
            if not m:
                continue
            ws, we = m.groups()
            if ws <= start_date and we >= end_date:
                sz = f.stat().st_size
                if sz > best_size:
                    candidate = f
                    best_size = sz

    if candidate is None:
        raise FileNotFoundError(
            f"No price cache covers {start_date} → {end_date}. "
            f"Available windows: {list_available_windows()}"
        )

    raw = json.loads(candidate.read_text())

    per_ticker: dict[str, dict[str, float]] = {}
    for ticker in tickers:
        bars = raw.get(ticker)
        if not isinstance(bars, dict):
            continue
        filt = {d: float(c) for d, c in bars.items()
                if start_date <= d <= end_date and c is not None}
        if len(filt) >= 10:
            per_ticker[ticker] = filt

    if not per_ticker:
        return {}

    # Pick the reference trading calendar. Crypto (BTC-USD) and some futures
    # trade 7 days/week so they have ~1827 dates over 5y while equities have
    # ~1256. Using the longest series as reference would reject every equity.
    # Prefer SPY's calendar (the canonical NYSE schedule). Fall back to the
    # ticker whose length matches the modal length among equities.
    if "SPY" in per_ticker:
        ref_dates = sorted(per_ticker["SPY"].keys())
    else:
        # Modal length across tickers — equities cluster on the trading-day count
        from collections import Counter
        lens = Counter(len(b) for b in per_ticker.values())
        modal_len, _ = lens.most_common(1)[0]
        ref_t = next(t for t, b in per_ticker.items() if len(b) == modal_len)
        ref_dates = sorted(per_ticker[ref_t].keys())

    ref_set = set(ref_dates)
    out: dict[str, list[float]] = {}
    for t, bars in per_ticker.items():
        # Need enough overlap with the reference calendar to be usable.
        common = ref_set & bars.keys()
        if len(common) < max(10, int(0.5 * len(ref_dates))):
            continue
        # Forward-fill missing dates from the previous available close so
        # series length always equals len(ref_dates).
        series: list[float] = []
        last: float | None = None
        for d in ref_dates:
            v = bars.get(d)
            if v is not None:
                last = v
            if last is not None:
                series.append(last)
        if len(series) >= 10:
            out[t] = series
    return out


# ── simulation ─────────────────────────────────────────────────────────────

def run_random_portfolio(
    prices: dict[str, list[float]],
    n_monkeys: int = N_MONKEYS,
    k_picks: int = 3,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    """Mode 1: each monkey picks ``k_picks`` random tickers and holds them
    equal-weight for the entire window. Returns final returns as fractions."""
    rng = np.random.default_rng(seed)
    tickers = list(prices.keys())
    if not tickers:
        return np.zeros(0)
    if len(tickers) < k_picks:
        k_picks = len(tickers)

    ticker_returns = np.array([
        prices[t][-1] / prices[t][0] - 1.0 for t in tickers
    ])

    results = np.zeros(n_monkeys)
    for i in range(n_monkeys):
        picks = rng.choice(len(tickers), size=k_picks, replace=False)
        results[i] = ticker_returns[picks].mean()
    return results


def run_active_random(
    prices: dict[str, list[float]],
    n_monkeys: int = N_MONKEYS,
    p_buy: float = 0.08,
    p_sell: float = 0.08,
    max_positions: int = 4,
    position_size_frac: float = 0.20,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    """Mode 2: active random trader — each day each monkey may buy a random
    ticker or sell a random held position. Tests timing alpha too.

    Conditional per-monkey logic precludes full vectorization, but at
    ~12M total step iterations (10k × 1250d) it runs in well under a
    minute on commodity hardware.
    """
    rng = np.random.default_rng(seed)
    tickers = list(prices.keys())
    n_tickers = len(tickers)
    if n_tickers == 0:
        return np.zeros(0)

    n_days = min(len(prices[t]) for t in tickers)
    price_matrix = np.array([prices[t][:n_days] for t in tickers])

    all_final_returns = np.zeros(n_monkeys)

    for m in range(n_monkeys):
        cash = INITIAL_CASH
        portfolio_value = INITIAL_CASH
        holdings: dict[int, float] = {}  # ticker_idx -> shares

        for d in range(n_days):
            held_value = sum(
                shares * price_matrix[t_idx, d]
                for t_idx, shares in holdings.items()
            )
            portfolio_value = cash + held_value

            # Random sell
            if holdings and rng.random() < p_sell:
                held_keys = list(holdings.keys())
                t_idx = held_keys[int(rng.integers(0, len(held_keys)))]
                cash += holdings[t_idx] * price_matrix[t_idx, d]
                del holdings[t_idx]

            # Random buy
            if (len(holdings) < max_positions
                    and cash > INITIAL_CASH * 0.05
                    and rng.random() < p_buy):
                t_idx = int(rng.integers(0, n_tickers))
                invest = min(portfolio_value * position_size_frac, cash)
                if invest > 0:
                    shares = invest / price_matrix[t_idx, d]
                    holdings[t_idx] = holdings.get(t_idx, 0.0) + shares
                    cash -= invest

        final_held = sum(
            shares * price_matrix[t_idx, n_days - 1]
            for t_idx, shares in holdings.items()
        )
        all_final_returns[m] = (cash + final_held) / INITIAL_CASH - 1.0

    return all_final_returns


# ── stats / ranking ────────────────────────────────────────────────────────

def compute_stats(returns: np.ndarray) -> dict:
    """Summary stats for a distribution of returns (as fractions)."""
    if len(returns) == 0:
        return {"n": 0}
    pcts = returns * 100
    return {
        "n": int(len(returns)),
        "mean_pct": round(float(np.mean(pcts)), 2),
        "median_pct": round(float(np.median(pcts)), 2),
        "p10_pct": round(float(np.percentile(pcts, 10)), 2),
        "p25_pct": round(float(np.percentile(pcts, 25)), 2),
        "p75_pct": round(float(np.percentile(pcts, 75)), 2),
        "p90_pct": round(float(np.percentile(pcts, 90)), 2),
        "p95_pct": round(float(np.percentile(pcts, 95)), 2),
        "p99_pct": round(float(np.percentile(pcts, 99)), 2),
        "min_pct": round(float(np.min(pcts)), 2),
        "max_pct": round(float(np.max(pcts)), 2),
        "std_pct": round(float(np.std(pcts)), 2),
    }


def rank_ai_vs_monkeys(ai_return_pct: float, monkey_returns: np.ndarray) -> dict:
    """How does one AI return compare to the monkey distribution?"""
    monkey_pcts = monkey_returns * 100
    beats = float(np.mean(monkey_pcts < ai_return_pct)) * 100
    return {
        "ai_return_pct": round(ai_return_pct, 2),
        "beats_pct_of_monkeys": round(beats, 1),
        "alpha_over_median_pct": round(
            ai_return_pct - float(np.median(monkey_pcts)), 2),
        "alpha_over_mean_pct": round(
            ai_return_pct - float(np.mean(monkey_pcts)), 2),
    }


# ── orchestration ──────────────────────────────────────────────────────────

def run_and_cache(
    tickers: list[str],
    start_date: str,
    end_date: str,
    ai_returns: list[float] | None = None,
) -> dict:
    """Run the full benchmark, compute stats, optionally rank AI runs, cache.

    ``ai_returns`` should be a list of AI backtest total_return_pct values
    (e.g. 45.2 for +45.2%) for runs that ran on the same window — the caller
    is responsible for filtering by (start, end).
    """
    t0 = time.time()

    print(f"[monkey] Loading prices for {len(tickers)} tickers "
          f"({start_date} → {end_date})...")
    prices = _load_prices(start_date, end_date, tickers)
    print(f"[monkey] {len(prices)} tickers with price data")

    if len(prices) < 5:
        raise ValueError(f"Too few tickers with price data: {len(prices)}")

    print(f"[monkey] Running {N_MONKEYS:,} random-portfolio simulations...")
    rp_returns = run_random_portfolio(prices, n_monkeys=N_MONKEYS)
    rp_stats = compute_stats(rp_returns)

    print(f"[monkey] Running {N_MONKEYS:,} active-random-trader simulations...")
    ar_returns = run_active_random(prices, n_monkeys=N_MONKEYS)
    ar_stats = compute_stats(ar_returns)

    result = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "window": {"start": start_date, "end": end_date},
        "n_tickers": len(prices),
        "tickers_used": sorted(prices.keys()),
        "n_monkeys": N_MONKEYS,
        "elapsed_s": round(time.time() - t0, 1),
        "random_portfolio": rp_stats,
        "active_random": ar_stats,
        "ai_rankings": [],
    }

    if ai_returns:
        for ret in ai_returns:
            result["ai_rankings"].append({
                "ai_return_pct": ret,
                "vs_random_portfolio": rank_ai_vs_monkeys(ret, rp_returns),
                "vs_active_random": rank_ai_vs_monkeys(ret, ar_returns),
            })

    _CACHE_OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = _CACHE_OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2))
    tmp.replace(_CACHE_OUT)
    print(f"[monkey] Done in {result['elapsed_s']}s → {_CACHE_OUT}")
    return result


def load_cached() -> dict | None:
    """Load cached benchmark results, or None if not available."""
    if not _CACHE_OUT.exists():
        return None
    try:
        return json.loads(_CACHE_OUT.read_text())
    except Exception:
        return None
