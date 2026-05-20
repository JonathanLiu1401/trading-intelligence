"""Per-ticker theme aggregation over the live news feed.

The wire produces 100+ articles per hour across ~17 collectors. Existing
surfaces tell the operator a slice of what is in there:

* ``/api/news-deduped`` is the linear item list (one row per article, no
  ticker-level rollup).
* ``/api/news-velocity`` measures the per-held-ticker MENTION RATE vs
  baseline — a Poisson z-score, not a "which ticker is the wire spending
  its breath on RIGHT NOW" rollup.
* ``/api/sector-heatmap`` / ``/api/sector-signal-fit`` aggregate at the
  SECTOR level (and `sector-signal-fit` cross-weighs against held
  exposure) — a coarser bucket than the per-name view a discretionary PM
  watches.
* digital-intern's ``trend_velocity`` does market-wide mention-gainers,
  not score-weighted theme prominence.

``build_news_themes`` is the missing per-ticker, recency-decayed,
score-weighted view that answers the operator's actual question after a
60-second glance at the feed: *which tickers is the wire actually
talking about, weighted by both freshness and the ML's relevance score,
and which of those am I already holding vs ignoring?*

Per-ticker row:

* ``decayed_score`` — Σ ai_score × exp(-age_h / HALF_LIFE × ln 2).
  Multi-ticker articles split their score evenly across mentioned
  tickers (a 4-ticker article contributes 0.25× to each, not 1× to
  every theme — avoids a wide-net headline inflating four themes
  simultaneously).
* ``n_articles`` — raw count of articles mentioning this ticker.
* ``max_urgency`` — max urgency across mentioning articles.
* ``top_title`` / ``top_url`` — the single highest decayed-score article.
* ``held`` — whether the ticker is in the live book (case-insensitive).

Aggregate-level: ``total_decayed_score``, ``n_articles``,
``n_held_themes``, ``n_unheld_themes``, ``top_unheld_ticker`` (the
loudest theme not in the book — a missed-opportunity bookmark distinct
from `/api/watchlist-opportunities` which is heat-ranked across the
named watchlist; this is across the *entire* live feed regardless of
watchlist membership).

State ladder: NO_DATA (no articles in window) / OK. No sample-size
gate beyond "at least one article surviving the recency filter" — the
single ranked list is honest even with 1 input.

Pure builder: never touches DB or network, never raises on garbage
rows. Observational only — never gates Opus (AGENTS.md #2/#12). Defense-
in-depth: a leaked synthetic backtest row (``url`` LIKE
``backtest://%`` or ``source`` LIKE ``backtest_%``/``opus_annotation%``)
is dropped at the builder so user-facing JSON cannot carry it even if a
future caller forgets the SQL clause.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone


# 6h half-life: a typical news-cycle window. Older articles contribute
# half-weight per HALF_LIFE; an article at the window edge (window=12h,
# halflife=6h) is at ~25% weight. Tunable; the *value* is pinned by
# tests (TestSummary::test_summary_block_present uses it directly).
DECAY_HALF_LIFE_HOURS = 6.0


def _parse_ts(ts):
    if not ts:
        return None
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _is_synthetic(row):
    """Defense-in-depth backtest filter (the SSOT lives in SQL)."""
    url = str(row.get("url") or "")
    src = str(row.get("source") or "")
    if url.startswith("backtest://"):
        return True
    if src.startswith("backtest_"):
        return True
    if src.startswith("opus_annotation"):
        return True
    return False


def build_news_themes(
    articles,
    held_tickers=None,
    now=None,
    window_hours: float = 24.0,
    max_themes: int = 20,
):
    """Aggregate ``articles`` into per-ticker themes ranked by
    recency-decayed score sum.

    Inputs:
        articles: list of dicts with ``first_seen`` (ISO ts), ``tickers``
            (list), ``ai_score`` (float), ``urgency`` (int), ``title``,
            ``url``, ``source``. The shape of ``/api/news-deduped`` rows.
        held_tickers: iterable of held tickers (case-insensitive).
        now: datetime (default UTC now).
        window_hours: only articles newer than this contribute.
        max_themes: clip surfaced themes to top N by decayed_score.

    Returns dict (always a stable shape regardless of input):
        as_of, window_hours, decay_half_life_hours, state, themes,
        n_articles, n_articles_with_no_tickers, total_decayed_score,
        n_held_themes, n_unheld_themes, top_unheld_ticker, headline.
    """
    now = now or datetime.now(timezone.utc)
    held_norm = {str(t).upper() for t in (held_tickers or []) if t}

    base = {
        "as_of": now.isoformat(timespec="seconds"),
        "window_hours": window_hours,
        "decay_half_life_hours": DECAY_HALF_LIFE_HOURS,
        "max_themes": int(max_themes),
        "state": "NO_DATA",
        "themes": [],
        "n_articles": 0,
        "n_articles_with_no_tickers": 0,
        "total_decayed_score": 0.0,
        "n_held_themes": 0,
        "n_unheld_themes": 0,
        "top_unheld_ticker": None,
        "headline": "News themes: no articles in window.",
    }

    if not articles:
        return base

    window_cutoff = now.timestamp() - window_hours * 3600
    ln2 = math.log(2)

    per_ticker: dict[str, dict] = {}
    n_articles = 0
    n_no_tickers = 0
    total_decayed = 0.0

    for art in articles:
        if not isinstance(art, dict):
            continue
        if _is_synthetic(art):
            continue
        ts = _parse_ts(art.get("first_seen"))
        if ts is None:
            continue
        if ts.timestamp() < window_cutoff:
            continue
        try:
            ai = float(art.get("ai_score") or 0.0)
        except Exception:
            ai = 0.0
        if ai <= 0:
            # Zero-score articles still count toward n_articles but
            # contribute nothing to per-ticker decayed scores. Skip the
            # bookkeeping rather than emitting 0-weight rows.
            n_articles += 1
            continue
        try:
            urg = int(art.get("urgency") or 0)
        except Exception:
            urg = 0

        tickers = art.get("tickers") or []
        if not isinstance(tickers, list):
            tickers = []
        tickers_norm = [str(t).upper().strip() for t in tickers if t]
        tickers_norm = [t for t in tickers_norm if t]

        n_articles += 1
        if not tickers_norm:
            n_no_tickers += 1
            continue

        age_h = max(0.0, (now.timestamp() - ts.timestamp()) / 3600.0)
        decay = math.exp(-age_h / DECAY_HALF_LIFE_HOURS * ln2)
        weight = ai * decay
        split = weight / len(tickers_norm)
        total_decayed += weight

        title = str(art.get("title") or "")
        url = str(art.get("url") or "")

        for tk in tickers_norm:
            row = per_ticker.setdefault(tk, {
                "ticker": tk,
                "decayed_score": 0.0,
                "n_articles": 0,
                "max_urgency": 0,
                "top_title": None,
                "top_url": None,
                "_top_weight": -1.0,
            })
            row["decayed_score"] += split
            row["n_articles"] += 1
            if urg > row["max_urgency"]:
                row["max_urgency"] = urg
            # Track the single-article best decayed weight (full, not
            # split) — for the "top headline for this theme" UI surface.
            if weight > row["_top_weight"]:
                row["_top_weight"] = weight
                row["top_title"] = title or None
                row["top_url"] = url or None

    if not per_ticker:
        base["n_articles"] = n_articles
        base["n_articles_with_no_tickers"] = n_no_tickers
        return base

    themes = []
    for row in per_ticker.values():
        row["decayed_score"] = round(row["decayed_score"], 4)
        row["held"] = row["ticker"] in held_norm
        row.pop("_top_weight", None)
        themes.append(row)
    themes.sort(key=lambda r: -r["decayed_score"])
    clipped = themes[: max(1, int(max_themes))]

    n_held = sum(1 for t in clipped if t["held"])
    n_unheld = len(clipped) - n_held
    top_unheld = next((t["ticker"] for t in clipped if not t["held"]), None)

    if clipped:
        top = clipped[0]
        headline = (
            f"Top theme: {top['ticker']} — {top['n_articles']} article(s), "
            f"score {top['decayed_score']:.1f}"
            f"{' (held)' if top['held'] else ''}."
        )
    else:
        headline = base["headline"]

    return {
        "as_of": now.isoformat(timespec="seconds"),
        "window_hours": window_hours,
        "decay_half_life_hours": DECAY_HALF_LIFE_HOURS,
        "max_themes": int(max_themes),
        "state": "OK",
        "themes": clipped,
        "n_articles": n_articles,
        "n_articles_with_no_tickers": n_no_tickers,
        "total_decayed_score": round(total_decayed, 4),
        "n_held_themes": n_held,
        "n_unheld_themes": n_unheld,
        "top_unheld_ticker": top_unheld,
        "headline": headline,
    }
