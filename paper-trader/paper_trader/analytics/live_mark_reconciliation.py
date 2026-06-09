"""Live-mark reconciliation against an independent quote read.

``mark_integrity`` catches the obvious failure: a quote is missing and a
position is explicitly marked at cost. The 2026-06-08 miss was subtler:
quotes were "available" through the bot's own path, but they were stale
regular-session marks, so the book displayed 0% P/L after after-hours fills.

This module compares held stock marks from the trader snapshot against a
small, direct yfinance read that intentionally does not call
``market.get_price``. It is advisory only: dashboard/reporter/CLI surface,
never a gate for the strategy.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Mapping, Sequence

DEFAULT_PRICE_ABS_TOLERANCE = 0.05       # dollars
DEFAULT_PRICE_PCT_TOLERANCE = 0.05       # percent
DEFAULT_BOOK_ABS_TOLERANCE = 1.00        # dollars
DEFAULT_MAX_TICKERS = 8


def _num(x: Any) -> float | None:
    if isinstance(x, bool) or x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return f


def held_stock_tickers(positions: Sequence[dict] | None) -> list[str]:
    """Unique held stock tickers, preserving position order."""
    out: list[str] = []
    seen: set[str] = set()
    for p in positions or []:
        if not isinstance(p, dict):
            continue
        if (p.get("type") or "stock") != "stock":
            continue
        tk = (p.get("ticker") or "").upper().strip()
        if not tk or tk in {"CASH", "NONE", "NO_DECISION", "BLOCKED"}:
            continue
        if tk in seen:
            continue
        seen.add(tk)
        out.append(tk)
    return out


def _fast_get(fast: Any, *keys: str) -> float | None:
    for key in keys:
        val = None
        try:
            if hasattr(fast, "get"):
                val = fast.get(key)
        except Exception:
            val = None
        if val is None:
            try:
                val = getattr(fast, key)
            except Exception:
                val = None
        f = _num(val)
        if f is not None and f > 0:
            return f
    return None


def independent_stock_quotes(
    tickers: Sequence[str],
    *,
    max_tickers: int = DEFAULT_MAX_TICKERS,
) -> dict[str, float | None]:
    """Direct yfinance quote read for a small held-position set.

    This deliberately avoids ``paper_trader.market.get_price`` so it can
    catch regressions in that adapter. CamelCase fast_info keys are tried
    first because yfinance exposes the live last price there on current
    builds; the history fallback includes pre/post-market rows.
    """
    import yfinance as yf

    out: dict[str, float | None] = {}
    for raw in list(tickers or [])[:max(0, int(max_tickers))]:
        tk = (raw or "").upper().strip()
        if not tk:
            continue
        quote: float | None = None
        try:
            ticker = yf.Ticker(tk)
            quote = _fast_get(
                ticker.fast_info,
                "lastPrice",
                "last_price",
                "regularMarketPrice",
                "regular_market_price",
            )
            if quote is None:
                hist = ticker.history(period="1d", interval="1m", prepost=True)
                if hist is not None and not hist.empty:
                    close = hist["Close"].dropna()
                    if not close.empty:
                        quote = _num(close.iloc[-1])
        except Exception:
            quote = None
        out[tk] = quote if quote is not None and quote > 0 else None
    return out


def build_live_mark_reconciliation(
    positions: Sequence[dict] | None,
    quotes: Mapping[str, Any] | None,
    *,
    price_abs_tolerance: float = DEFAULT_PRICE_ABS_TOLERANCE,
    price_pct_tolerance: float = DEFAULT_PRICE_PCT_TOLERANCE,
    book_abs_tolerance: float = DEFAULT_BOOK_ABS_TOLERANCE,
) -> dict:
    rows: list[dict] = []
    qmap = {str(k).upper(): _num(v) for k, v in (quotes or {}).items()}

    for p in positions or []:
        if not isinstance(p, dict):
            continue
        if (p.get("type") or "stock") != "stock":
            continue
        tk = (p.get("ticker") or "").upper().strip()
        if not tk:
            continue
        qty = _num(p.get("qty"))
        cur = _num(p.get("current_price"))
        avg = _num(p.get("avg_cost"))
        quote = qmap.get(tk)
        if qty is None or qty <= 0 or cur is None or cur <= 0:
            continue

        price_delta = None if quote is None else quote - cur
        price_delta_pct = None
        if quote is not None and cur > 0:
            price_delta_pct = price_delta / cur * 100.0
        book_delta = None if price_delta is None else price_delta * qty
        expected_unrealized = (
            None if quote is None or avg is None else (quote - avg) * qty
        )
        displayed_unrealized = _num(p.get("unrealized_pl"))
        divergent = False
        if price_delta is not None and price_delta_pct is not None:
            divergent = (
                abs(price_delta) > price_abs_tolerance
                and abs(price_delta_pct) > price_pct_tolerance
            )

        rows.append({
            "ticker": tk,
            "qty": qty,
            "stored_price": round(cur, 4),
            "independent_price": round(quote, 4) if quote is not None else None,
            "price_delta": round(price_delta, 4) if price_delta is not None else None,
            "price_delta_pct": (
                round(price_delta_pct, 4) if price_delta_pct is not None else None
            ),
            "book_delta_usd": round(book_delta, 2) if book_delta is not None else None,
            "displayed_unrealized_pl": (
                round(displayed_unrealized, 2)
                if displayed_unrealized is not None else None
            ),
            "expected_unrealized_pl": (
                round(expected_unrealized, 2)
                if expected_unrealized is not None else None
            ),
            "divergent": divergent,
        })

    n = len(rows)
    if n == 0:
        return {
            "verdict": "NO_DATA",
            "headline": "No held stock marks to reconcile.",
            "n_positions": 0,
            "n_missing_quotes": 0,
            "n_divergent": 0,
            "book_delta_usd": 0.0,
            "worst": None,
            "positions": [],
        }

    missing = [r for r in rows if r["independent_price"] is None]
    comparable = [r for r in rows if r["independent_price"] is not None]
    divergent_rows = [r for r in comparable if r["divergent"]]
    book_delta = sum(r["book_delta_usd"] or 0.0 for r in comparable)
    worst = (
        max(comparable, key=lambda r: abs(r["book_delta_usd"] or 0.0))
        if comparable else None
    )

    if not comparable:
        verdict = "NO_QUOTES"
        headline = (
            f"Independent quote read failed for all {n} held stock mark(s); "
            "cannot verify displayed P/L."
        )
    elif divergent_rows or abs(book_delta) >= book_abs_tolerance:
        verdict = "DIVERGED"
        w = worst or divergent_rows[0]
        headline = (
            "Stored marks disagree with independent live quotes: "
            f"book delta ${book_delta:+.2f} across {len(comparable)} held "
            f"stock mark(s); worst {w['ticker']} stored "
            f"${w['stored_price']:.2f} vs quote "
            f"${w['independent_price']:.2f} "
            f"({w['price_delta_pct']:+.2f}%). P/L display may be stale."
        )
    elif missing:
        verdict = "DEGRADED"
        headline = (
            f"{len(missing)}/{n} held stock mark(s) lacked independent quotes; "
            "verified marks agree within tolerance."
        )
    else:
        verdict = "CLEAN"
        headline = (
            f"All {n} held stock mark(s) agree with independent live quotes "
            "within tolerance."
        )

    return {
        "verdict": verdict,
        "headline": headline,
        "n_positions": n,
        "n_missing_quotes": len(missing),
        "n_divergent": len(divergent_rows),
        "book_delta_usd": round(book_delta, 2),
        "worst": worst,
        "positions": rows,
        "thresholds": {
            "price_abs_tolerance": price_abs_tolerance,
            "price_pct_tolerance": price_pct_tolerance,
            "book_abs_tolerance": book_abs_tolerance,
        },
    }


def run_live_mark_reconciliation() -> dict:
    from paper_trader.store import get_store
    from paper_trader.strategy import portfolio_snapshot_readonly

    store = get_store()
    snap = portfolio_snapshot_readonly(store)
    positions = snap.get("positions") or []
    tickers = held_stock_tickers(positions)
    quotes = independent_stock_quotes(tickers)
    return build_live_mark_reconciliation(positions, quotes)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    out = run_live_mark_reconciliation()
    if args.json:
        print(json.dumps(out, indent=2, sort_keys=True))
    else:
        print(f"[{out['verdict']}] {out['headline']}")
    return 2 if out.get("verdict") in {"DIVERGED", "NO_QUOTES"} else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
