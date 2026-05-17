"""Sector heatmap: per-bucket momentum + relative strength + news sentiment.

Buckets are tuned for the user's actual focus (DRAM, semis equipment, foundry,
leveraged plays). Each ticker returns:
  price, mom_5d %, mom_20d %, rsi, vs_sox_5d (relative strength vs SOXX over 5d),
  news_n (article count last 24h), news_avg_score, news_urgent.

The heatmap UI groups tickers by bucket and color-codes by mom_5d so a glance
tells you which corner of memory/semis is leading.
"""
from __future__ import annotations

import math
import time
from datetime import datetime, timezone

# Layout the user explicitly cares about (DRAM-focused).
HEATMAP_BUCKETS: dict[str, list[str]] = {
    "memory_core":      ["MU", "WDC", "STX"],
    "semis_equipment":  ["LRCX", "AMAT", "KLAC", "ASML"],
    "foundry":          ["TSM", "GFS", "UMC"],
    "design":           ["NVDA", "AMD", "MRVL", "AVGO"],
    "memory_leveraged": ["MUU", "SOXL", "NVDU", "SOXS"],
    "optical":          ["LITE", "LNOK", "CIEN"],
    "etf":              ["SMH", "SOXX"],
}
ALL_HEATMAP_TICKERS = sorted({t for tickers in HEATMAP_BUCKETS.values() for t in tickers})
RELATIVE_TO = "SOXX"  # semis benchmark for relative-strength calc


_CACHE: dict[str, tuple[float, dict]] = {}
_CACHE_TTL = 60.0  # seconds — yfinance bulk is slow, cache hard


def _cached(key: str):
    v = _CACHE.get(key)
    if v and time.time() - v[0] < _CACHE_TTL:
        return v[1]
    return None


def _store(key: str, val: dict):
    _CACHE[key] = (time.time(), val)


def _rsi(closes: list[float], period: int = 14) -> float | None:
    if len(closes) < period + 1:
        return None
    gains, losses = 0.0, 0.0
    for i in range(1, period + 1):
        diff = closes[i] - closes[i - 1]
        if diff >= 0:
            gains += diff
        else:
            losses -= diff
    avg_gain = gains / period
    avg_loss = losses / period
    for i in range(period + 1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gain = max(diff, 0)
        loss = -min(diff, 0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100.0 - 100.0 / (1.0 + rs), 2)


def _fetch_hist_bulk(tickers: list[str], period: str = "1mo") -> dict[str, list[float]]:
    """Bulk download of close prices via yfinance. Returns {ticker: [closes asc]}.

    Tolerant of missing tickers and yfinance's bulk-vs-single shape difference.
    """
    out: dict[str, list[float]] = {}
    if not tickers:
        return out
    try:
        import yfinance as yf
        data = yf.download(tickers, period=period, interval="1d",
                           group_by="ticker", progress=False,
                           threads=True, auto_adjust=False)
    except Exception as e:
        print(f"[heatmap] yf bulk failed: {e}")
        return out

    for t in tickers:
        try:
            if len(tickers) == 1:
                series = data["Close"].dropna()
            else:
                series = data[t]["Close"].dropna()
            closes = [float(x) for x in series.tolist() if x and not math.isnan(x)]
            if closes:
                out[t] = closes
        except Exception:
            continue
    return out


def _ticker_news_pulse_bulk(tickers: list[str], hours: int = 24) -> dict[str, dict]:
    """Aggregate article counts and avg scores per ticker from the digital-intern DB."""
    out: dict[str, dict] = {t.upper(): {
        "n": 0, "urgent": 0, "avg_score": 0.0, "max_score": 0.0,
    } for t in tickers}
    try:
        from ..signals import ticker_sentiments
    except Exception:
        return out
    try:
        rows = ticker_sentiments([t.upper() for t in tickers], hours=hours)
    except Exception:
        return out
    for r in rows:
        t = r.get("ticker", "").upper()
        if t in out:
            out[t] = {
                "n": int(r.get("n") or 0),
                "urgent": int(r.get("urgent") or 0),
                "avg_score": float(r.get("avg_score") or 0.0),
                "max_score": float(r.get("max_score") or 0.0),
            }
    return out


def compute_heatmap() -> dict:
    """One-shot heatmap snapshot. Caches for 60s."""
    cached = _cached("snapshot")
    if cached is not None:
        return cached

    tickers = ALL_HEATMAP_TICKERS + [RELATIVE_TO]
    hist = _fetch_hist_bulk(tickers, period="2mo")
    news = _ticker_news_pulse_bulk(ALL_HEATMAP_TICKERS, hours=24)

    # Reference: SOXX 5d return for relative-strength calc.
    ref_closes = hist.get(RELATIVE_TO, [])
    ref_5d = None
    if len(ref_closes) >= 6:
        ref_5d = (ref_closes[-1] - ref_closes[-6]) / ref_closes[-6] * 100.0

    buckets_out: list[dict] = []
    for bucket_name, ticker_list in HEATMAP_BUCKETS.items():
        rows = []
        for t in ticker_list:
            closes = hist.get(t, [])
            if len(closes) < 6:
                rows.append({
                    "ticker": t, "price": None, "mom_5d": None,
                    "mom_20d": None, "rsi": None, "vs_sox_5d": None,
                    **news.get(t, {"n": 0, "urgent": 0, "avg_score": 0.0, "max_score": 0.0}),
                })
                continue
            price = closes[-1]
            m5 = (price - closes[-6]) / closes[-6] * 100.0
            m20 = ((price - closes[-21]) / closes[-21] * 100.0) if len(closes) >= 21 else None
            rsi = _rsi(closes[-30:]) if len(closes) >= 16 else None
            vs_sox = (m5 - ref_5d) if (ref_5d is not None) else None
            rows.append({
                "ticker": t,
                "price": round(price, 2),
                "mom_5d": round(m5, 2),
                "mom_20d": round(m20, 2) if m20 is not None else None,
                "rsi": rsi,
                "vs_sox_5d": round(vs_sox, 2) if vs_sox is not None else None,
                **news.get(t, {"n": 0, "urgent": 0, "avg_score": 0.0, "max_score": 0.0}),
            })
        # Bucket-level avg momentum so the dashboard can label the strongest bucket.
        m5_vals = [r["mom_5d"] for r in rows if r["mom_5d"] is not None]
        bucket_m5 = round(sum(m5_vals) / len(m5_vals), 2) if m5_vals else None
        buckets_out.append({
            "name": bucket_name,
            "avg_mom_5d": bucket_m5,
            "tickers": rows,
        })

    snapshot = {
        "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reference": RELATIVE_TO,
        "reference_mom_5d": round(ref_5d, 2) if ref_5d is not None else None,
        "buckets": buckets_out,
    }
    _store("snapshot", snapshot)
    return snapshot


if __name__ == "__main__":
    import json
    print(json.dumps(compute_heatmap(), indent=2)[:2000])
