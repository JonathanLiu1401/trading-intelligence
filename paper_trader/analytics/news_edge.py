"""News Edge — does digital-intern's scored news actually predict moves?

The entire two-repo stack rests on one unproven assumption: that a high
``ai_score`` headline on a ticker is *informative* about that ticker's
near-term return. ``calibration.py`` measures the bot's *trades* (a handful of
FIFO round-trips). This module measures the **signal itself**, independent of
whether the bot ever acted on it: take every scored article that names a
watchlist ticker, find that ticker's close on the article day, and look at the
1-/3-/5-trading-day forward return.

Two honesty controls make the number trustworthy:

* **SPY-abnormal return.** A raw "+4% in 3d after a score-8 headline" is
  meaningless if SPY also rose 4%. Every forward return is reported both raw
  *and* abnormal (``ticker_return − spy_return`` over the identical span). The
  verdict is judged on abnormal return only.
* **ai_score banding.** If the pipeline has real edge, the mean abnormal return
  should rise monotonically with the score band. A flat or inverted curve means
  the score is noise — the verdict says so plainly instead of cherry-picking
  the one band that looks good.

``build_news_edge`` is pure and self-contained: it resolves tickers from
article text with the same ``(?:\\$|\\b)TICKER\\b`` regex ``position_thesis``
uses, so a test can feed synthetic articles + synthetic price series and assert
exact bucket numbers. The dashboard endpoint does the I/O (articles.db query
with the live-only filter, yfinance daily bars) and hands the rows in.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

# (label, lo_inclusive, hi_exclusive) over ai_score's 0..10 range.
DEFAULT_BANDS: list[tuple[str, float, float]] = [
    ("8.0+", 8.0, 1e9),
    ("6.0-8.0", 6.0, 8.0),
    ("4.0-6.0", 4.0, 6.0),
    ("2.0-4.0", 2.0, 4.0),
]
DEFAULT_HORIZONS = (1, 3, 5)
# A band needs at least this many resolved articles before its mean is allowed
# to drive the verdict — otherwise one lucky headline reads as "edge".
_MIN_BAND_N = 8


def _parse_date(ts: str | None) -> str | None:
    """ISO timestamp → ``YYYY-MM-DD`` (UTC calendar day of the article)."""
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%d")
    except Exception:
        return None


def _resolve_ticker(text: str, tickers: list[str],
                     _cache: dict[str, re.Pattern]) -> str | None:
    """First ticker in ``tickers`` whose ``$TK`` / word-boundary form appears.

    Same matching rule as ``position_thesis._ticker_news`` so the two panels
    agree on which articles belong to which name."""
    up = text.upper()
    for tk in tickers:
        pat = _cache.get(tk)
        if pat is None:
            pat = re.compile(rf"(?:\$|\b){re.escape(tk.upper())}\b")
            _cache[tk] = pat
        if pat.search(up):
            return tk
    return None


def _band_of(score: float, bands: list[tuple[str, float, float]]) -> str | None:
    for label, lo, hi in bands:
        if lo <= score < hi:
            return label
    return None


def _index_at_or_after(dates: list[str], day: str) -> int | None:
    """First bar index whose trading date is >= the article calendar day."""
    lo, hi = 0, len(dates)
    while lo < hi:
        mid = (lo + hi) // 2
        if dates[mid] < day:
            lo = mid + 1
        else:
            hi = mid
    return lo if lo < len(dates) else None


def build_news_edge(
    articles: list[dict],
    price_history: dict[str, list[tuple[str, float]]],
    spy_history: list[tuple[str, float]],
    tickers: list[str],
    now: datetime | None = None,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    bands: list[tuple[str, float, float]] | None = None,
) -> dict:
    """Forward-return edge of scored news, banded by ai_score.

    Args:
      articles: ``[{text, ai_score, urgency, published}]`` — caller must have
        already applied the live-only filter (no ``backtest://`` rows).
      price_history: ``{TICKER: [(YYYY-MM-DD, close), ...]}`` ascending daily
        bars (trading days only).
      spy_history: ``[(YYYY-MM-DD, close), ...]`` ascending — for abnormal
        return. Pass ``[]`` to get raw-only.
      tickers: resolution universe (the live WATCHLIST), tried in order.
    """
    now = now or datetime.now(timezone.utc)
    bands = bands or DEFAULT_BANDS
    horizons = tuple(sorted(set(int(h) for h in horizons if h > 0)))

    # Pre-split each ticker's bars into parallel date/close lists once.
    ph: dict[str, tuple[list[str], list[float]]] = {}
    for tk, bars in (price_history or {}).items():
        s = sorted(bars, key=lambda b: b[0])
        ph[tk.upper()] = ([d for d, _ in s], [float(c) for _, c in s])
    spy_by_date = {d: float(c) for d, c in (spy_history or [])}

    def _blank() -> dict:
        return {h: {"n": 0, "raw_sum": 0.0, "raw_up": 0,
                    "abn_n": 0, "abn_sum": 0.0, "abn_up": 0} for h in horizons}

    acc: dict[str, dict] = {lbl: _blank() for lbl, _, _ in bands}
    urgency_acc = {"urgent": _blank(), "normal": _blank()}

    rx_cache: dict[str, re.Pattern] = {}
    n_total = len(articles or [])
    n_resolved = 0
    n_scored_in_band = 0

    for art in articles or []:
        try:
            score = float(art.get("ai_score") or 0.0)
        except (TypeError, ValueError):
            continue
        band = _band_of(score, bands)
        if band is None:
            continue
        n_scored_in_band += 1
        text = str(art.get("text") or art.get("title") or "")
        tk = _resolve_ticker(text, tickers, rx_cache)
        if not tk or tk.upper() not in ph:
            continue
        day = _parse_date(art.get("published") or art.get("first_seen"))
        if day is None:
            continue
        dates, closes = ph[tk.upper()]
        i0 = _index_at_or_after(dates, day)
        if i0 is None:
            continue
        entry = closes[i0]
        if entry <= 0:
            continue
        entry_date = dates[i0]
        spy_entry = spy_by_date.get(entry_date)
        resolved_any = False
        urg = "urgent" if int(art.get("urgency") or 0) >= 1 else "normal"
        for h in horizons:
            j = i0 + h
            if j >= len(dates):
                continue
            fwd = closes[j]
            if fwd <= 0:
                continue
            raw = (fwd / entry - 1.0) * 100.0
            cell = acc[band][h]
            ucell = urgency_acc[urg][h]
            cell["n"] += 1
            cell["raw_sum"] += raw
            ucell["n"] += 1
            ucell["raw_sum"] += raw
            if raw > 0:
                cell["raw_up"] += 1
                ucell["raw_up"] += 1
            spy_fwd = spy_by_date.get(dates[j])
            if spy_entry and spy_fwd and spy_entry > 0:
                abn = raw - (spy_fwd / spy_entry - 1.0) * 100.0
                cell["abn_n"] += 1
                cell["abn_sum"] += abn
                ucell["abn_n"] += 1
                ucell["abn_sum"] += abn
                if abn > 0:
                    cell["abn_up"] += 1
                    ucell["abn_up"] += 1
            resolved_any = True
        if resolved_any:
            n_resolved += 1

    def _finalize(blank: dict) -> dict:
        rows = {}
        for h in horizons:
            c = blank[h]
            n = c["n"]
            an = c["abn_n"]
            rows[str(h)] = {
                "n": n,
                "mean_raw_pct": round(c["raw_sum"] / n, 3) if n else None,
                "raw_up_rate": round(c["raw_up"] / n * 100, 1) if n else None,
                "n_abnormal": an,
                "mean_abnormal_pct": round(c["abn_sum"] / an, 3) if an else None,
                "abnormal_hit_rate": round(c["abn_up"] / an * 100, 1) if an else None,
            }
        return rows

    band_rows = []
    for lbl, _, _ in bands:
        band_rows.append({"band": lbl, "horizons": _finalize(acc[lbl])})

    # Reference horizon for the verdict. Prefer the longest horizon whose top
    # band is adequately sampled (a 5d edge is the strongest claim); fall back
    # to the longest horizon with *any* top-band data, then to 3, then the
    # middle one. This makes the panel "graduate" automatically as
    # digital-intern's article history deepens — early on only 1d has a
    # forward window, so the panel reports the 1d edge instead of all dashes.
    top_lbl = bands[0][0] if bands else None
    ref = None
    if horizons and top_lbl is not None:
        well = [h for h in horizons
                if (acc[top_lbl][h]["abn_n"] or 0) >= _MIN_BAND_N]
        some = [h for h in horizons if (acc[top_lbl][h]["abn_n"] or 0) > 0]
        if well:
            ref = max(well)
        elif some:
            ref = max(some)
    if ref is None:
        ref = 3 if 3 in horizons else (
            horizons[len(horizons) // 2] if horizons else None)

    verdict, reason = _judge(band_rows, bands, ref)

    return {
        "as_of": now.isoformat(timespec="seconds"),
        "n_articles": n_total,
        "n_in_score_band": n_scored_in_band,
        "n_resolved": n_resolved,
        "horizons": list(horizons),
        "reference_horizon": ref,
        "spy_adjusted": bool(spy_by_date),
        "bands": band_rows,
        "by_urgency": {
            "urgent": _finalize(urgency_acc["urgent"]),
            "normal": _finalize(urgency_acc["normal"]),
        },
        "verdict": verdict,
        "verdict_reason": reason,
    }


def _judge(band_rows: list[dict], bands: list[tuple[str, float, float]],
           ref: int | None) -> tuple[str, str]:
    """Edge verdict from the abnormal-return curve at the reference horizon.

    Top band = highest ai_score (``bands`` is ordered high→low). The pipeline
    has edge only if the top band's abnormal return is both positive and beats
    the bottom band's, with enough samples to mean it."""
    if ref is None:
        return "NO_DATA", "no horizons configured"
    by_label = {r["band"]: r["horizons"].get(str(ref), {}) for r in band_rows}
    top_lbl = bands[0][0]
    bot_lbl = bands[-1][0]
    top = by_label.get(top_lbl, {})
    bot = by_label.get(bot_lbl, {})
    top_n = top.get("n_abnormal") or 0
    if top_n < _MIN_BAND_N:
        return ("INSUFFICIENT_DATA",
                f"only {top_n} SPY-adjusted samples at {ref}d in the "
                f"{top_lbl} band (need {_MIN_BAND_N}) — accumulate more "
                "news history")
    top_abn = top.get("mean_abnormal_pct")
    top_hit = top.get("abnormal_hit_rate")
    bot_abn = bot.get("mean_abnormal_pct")
    if top_abn is None:
        return "INSUFFICIENT_DATA", f"{top_lbl} band has no SPY-adjusted return"
    if top_abn <= 0:
        return ("NO_EDGE",
                f"top-band ({top_lbl}) {ref}d abnormal return is "
                f"{top_abn:+.2f}% — high scores do not beat the market")
    beats_bottom = bot_abn is None or top_abn > bot_abn
    if top_abn > 0 and (top_hit or 0) >= 50.0 and beats_bottom:
        return ("EDGE_CONFIRMED",
                f"{top_lbl} headlines lead +{top_abn:.2f}% abnormal at {ref}d "
                f"({top_hit:.0f}% hit rate) — the news pipeline has predictive "
                "power")
    return ("WEAK_EDGE",
            f"{top_lbl} band is +{top_abn:.2f}% abnormal at {ref}d but the "
            f"signal is weak (hit rate {top_hit}%, monotonic={beats_bottom})")


if __name__ == "__main__":  # smoke test against the live DB + yfinance
    import json
    import sqlite3
    import zlib
    from pathlib import Path

    import yfinance as yf

    from paper_trader.strategy import WATCHLIST

    di = Path("/media/zeph/projects/digital-intern/db/articles.db")
    if not di.exists():
        di = Path("/home/zeph/digital-intern/data/articles.db")
    conn = sqlite3.connect(f"file:{di}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT title, full_text, ai_score, urgency, first_seen FROM articles "
        "WHERE ai_score >= 2.0 AND first_seen >= datetime('now','-21 days') "
        "AND url NOT LIKE 'backtest://%' AND source NOT LIKE 'backtest_%' "
        "AND source NOT LIKE 'opus_annotation%' ORDER BY first_seen DESC LIMIT 1500"
    ).fetchall()
    conn.close()

    def _dz(b):
        try:
            return zlib.decompress(b).decode("utf-8", "replace") if b else ""
        except Exception:
            return ""

    arts = [{"text": f"{r['title']} {_dz(r['full_text'])}",
             "ai_score": r["ai_score"], "urgency": r["urgency"],
             "published": r["first_seen"]} for r in rows]
    universe = sorted({a_tk for a in arts for a_tk in WATCHLIST
                       if re.search(rf"(?:\$|\b){a_tk}\b", a["text"].upper())})
    ph = {}
    for tk in universe + ["SPY"]:
        h = yf.Ticker(tk).history(period="2mo", auto_adjust=False)
        ph[tk] = [(d.strftime("%Y-%m-%d"), float(c))
                  for d, c in zip(h.index, h["Close"]) if c == c]
    spy = ph.pop("SPY", [])
    print(json.dumps(build_news_edge(arts, ph, spy, WATCHLIST), indent=2,
                      default=str))
