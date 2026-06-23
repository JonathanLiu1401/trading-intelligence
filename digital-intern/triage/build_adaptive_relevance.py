"""Build adaptive ArticleNet relevance weights from post-article returns.

This is intentionally an offline/bounded job. It reads recent ArticleNet rows,
resolves ticker mentions, downloads price bars, measures 1/3/5-trading-day
absolute and SPY-abnormal returns after each article, then writes
data/adaptive_relevance.json for triage.adaptive_relevance to consume.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import zlib
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yfinance as yf

from triage.adaptive_relevance import CACHE_PATH, _features

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "data" / "articles.db"
LIVE_ONLY = (
    "url NOT LIKE 'backtest://%' "
    "AND source NOT LIKE 'backtest_%' "
    "AND source NOT LIKE 'opus_annotation%'"
)
HORIZONS = (1, 3, 5)
KNOWN_TICKERS = {
    "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "META", "NVDA", "AMD", "AVGO",
    "ORCL", "MU", "TSM", "ASML", "AMAT", "LRCX", "KLAC", "MRVL", "INTC",
    "QCOM", "SMH", "SOXX", "SPY", "QQQ", "IWM", "DIA", "TLT", "GLD", "SLV",
    "TQQQ", "SQQQ", "SOXL", "SOXS", "TECL", "TECS", "FNGU", "FNGD", "UPRO",
    "SPXU", "SPXL", "SPXS", "QLD", "SSO", "NVDU", "MSFU", "AMZU", "TSLL",
    "LITE", "LNOK", "MUU", "DRAM", "SNDU", "SNK", "BIRD", "COIN", "TSLA",
    "CRM", "ADBE", "NOW", "PANW", "CRWD", "SMCI", "DELL", "HPE", "VRT",
    "CEG", "GEV", "ETN", "PWR", "ANET", "ARM",
}


def _decompress(blob) -> str:
    if blob is None:
        return ""
    if isinstance(blob, str):
        return blob
    try:
        return zlib.decompress(blob).decode("utf-8", errors="replace")
    except Exception:
        return ""


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _fetch_articles(days: int, limit: int) -> list[dict]:
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=20)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT title, source, first_seen, full_text "
            "FROM articles WHERE first_seen >= ? AND first_seen <= ? "
            f"AND {LIVE_ONLY} ORDER BY first_seen DESC LIMIT ?",
            (
                since,
                (datetime.now(timezone.utc) - timedelta(days=max(HORIZONS) + 1)).isoformat(),
                limit,
            ),
        ).fetchall()
    finally:
        conn.close()
    out = []
    for r in rows:
        out.append({
            "title": r["title"] or "",
            "summary": _decompress(r["full_text"])[:1200],
            "source": r["source"] or "",
            "first_seen": r["first_seen"],
        })
    return out


def _tickers_for(text: str) -> list[str]:
    import re
    cashtags = set(re.findall(r"\$([A-Z]{1,5})\b", text.upper()))
    uppercase = set(re.findall(r"\b([A-Z]{1,5})\b", text.upper()))
    return sorted((cashtags | uppercase) & KNOWN_TICKERS)[:5]


def _bars(tickers: set[str], start: datetime, end: datetime) -> dict[str, list[tuple[str, float]]]:
    if not tickers:
        return {}
    data = yf.download(
        sorted(tickers),
        start=start.date().isoformat(),
        end=(end + timedelta(days=1)).date().isoformat(),
        progress=False,
        auto_adjust=True,
        threads=False,
    )
    out: dict[str, list[tuple[str, float]]] = {}
    if data.empty:
        return out
    close = data["Close"] if "Close" in data else data
    if hasattr(close, "columns"):
        for tk in sorted(tickers):
            if tk not in close:
                continue
            series = close[tk].dropna()
            out[tk] = [(idx.date().isoformat(), float(v)) for idx, v in series.items()]
    else:
        tk = next(iter(tickers))
        out[tk] = [(idx.date().isoformat(), float(v)) for idx, v in close.dropna().items()]
    return out


def _forward_return(bars: list[tuple[str, float]], day: str, horizon: int) -> float | None:
    if not bars:
        return None
    dates = [d for d, _ in bars]
    closes = [c for _, c in bars]
    i = next((idx for idx, d in enumerate(dates) if d >= day), None)
    if i is None or i + horizon >= len(closes) or closes[i] <= 0:
        return None
    return (closes[i + horizon] / closes[i] - 1.0) * 100.0


def build(days: int = 45, limit: int = 5000) -> dict:
    articles = _fetch_articles(days, limit)
    ticker_map = {}
    all_tickers = {"SPY"}
    parsed = []
    for art in articles:
        ts = _parse_ts(art["first_seen"])
        if ts is None:
            continue
        tickers = _tickers_for(f"{art['title']} {art['summary']}")
        if not tickers:
            continue
        feats = _features(art["title"], art["summary"], art["source"])
        parsed.append((art, ts, tickers, feats))
        all_tickers.update(tickers)
    start = datetime.now(timezone.utc) - timedelta(days=days + 10)
    end = datetime.now(timezone.utc)
    price = _bars(all_tickers, start, end)
    acc = defaultdict(list)
    spy = price.get("SPY", [])
    for art, ts, tickers, feats in parsed:
        day = ts.date().isoformat()
        spy_rets = {
            h: _forward_return(spy, day, h)
            for h in HORIZONS
        }
        for tk in tickers:
            bars = price.get(tk)
            if not bars:
                continue
            vals = []
            abn_vals = []
            for h in HORIZONS:
                r = _forward_return(bars, day, h)
                if r is None:
                    continue
                vals.append(abs(r))
                if spy_rets.get(h) is not None:
                    abn_vals.append(r - spy_rets[h])
            if not vals:
                continue
            ret_abs = max(vals)
            ret_abn = max(abn_vals, key=abs) if abn_vals else 0.0
            for feat in [*feats, f"ticker:{tk}"]:
                acc[feat].append((ret_abs, ret_abn))

    weights = {}
    for feat, vals in acc.items():
        if len(vals) < 3:
            continue
        abs_vals = [v[0] for v in vals]
        abn_vals = [v[1] for v in vals]
        mean_abs = statistics.fmean(abs_vals)
        mean_abn = statistics.fmean(abn_vals)
        # 1.5% absolute move is ordinary noise; bigger realized impact earns
        # relevance. Abnormal return contributes but does not dominate because
        # relevance is about market-moving power, not only direction.
        boost = 1.0 + (mean_abs - 1.5) / 6.0 + min(0.4, abs(mean_abn) / 12.0)
        weights[feat] = {
            "n": len(vals),
            "boost": round(max(0.35, min(2.75, boost)), 4),
            "mean_abs_return_pct": round(mean_abs, 4),
            "mean_abnormal_return_pct": round(mean_abn, 4),
        }

    return {
        "version": 1,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "article_count": len(articles),
        "resolved_article_count": len(parsed),
        "weights": dict(sorted(weights.items())),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=45)
    p.add_argument("--limit", type=int, default=5000)
    args = p.parse_args()
    out = build(days=args.days, limit=args.limit)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    tmp.replace(CACHE_PATH)
    print(json.dumps({
        "cache": str(CACHE_PATH),
        "article_count": out["article_count"],
        "resolved_article_count": out["resolved_article_count"],
        "weights": len(out["weights"]),
    }))


if __name__ == "__main__":
    main()
