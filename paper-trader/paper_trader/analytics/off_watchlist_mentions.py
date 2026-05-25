"""Tickers the wire is talking about that the bot is NOT watching.

The orthogonal complement to ``watchlist_opportunities``. That endpoint
ranks WATCHLIST names by news heat to surface "what should I rotate
INTO from my universe?". This module answers the prior question: "what
catalysts is the wire screaming about on names I have not even put on
my radar?" — the *universe-expansion* surface, not the rotation surface.

Distinct from every neighbour (mirrors the discipline of
``rising_unheld_themes`` invariant #10 — do not consolidate):

* ``/api/watchlist-opportunities`` — bounded to the named WATCHLIST.
  An off-watchlist ticker (e.g. KWEB / BIDU mentioned in the urgent
  feed during a China-AI rotation) is invisible there by design.
* ``/api/rising-unheld-themes`` — per-ticker decayed-score velocity but
  only across tickers ``_extract_tickers`` *already finds*; it does not
  partition the "found" set by watchlist membership, so an unheld
  WATCHLIST name (NVDA mid-print) and an off-watchlist ADR (KWEB) land
  in the same surface and the off-list mention drowns under the
  WATCHLIST noise.
* ``/api/news-themes`` — keyword/theme-level, not ticker-level.
* ``/api/news-deduped`` — raw article stream, no ticker aggregation.

The builder is **pure** over a single already-fetched
``signals.get_top_signals()`` list (the caller fetches once; this tallies
per ticker — no N-query fan-out, same shape as ``watchlist_opportunities``).
Reuses each article's ``tickers`` field (populated by ``signals.py``'s
SSOT ``_extract_tickers``) so this can never drift from the live trader's
own ticker extraction.

Pure, **never raises**. Advisory only — never gates Opus, adds no caps
(AGENTS.md #2/#12).
"""
from __future__ import annotations

import math

# Surface-specific defensive noise filter.
#
# The SSOT ``signals._extract_tickers`` is tuned for the live-decision prompt
# (where every WATCHLIST name is a real symbol and surrounding context filters
# itself). When the same extractor is pointed at the UNBOUNDED off-watchlist
# space, a different class of false-positive surfaces: news-source brand
# suffixes ("…- MSN"), exchange suffixes ("BB.TSX"), product-line names
# ("EUV", "EPYC"), and ALLCAPS financial-concept words ("CAPEX") all look
# like 2–5 char tickers to the regex and pass the SSOT's noise filter
# because they ARE in the "common all-caps token but ALSO a real symbol on
# OTHER exchanges" grey zone (e.g. EUV is a Canadian shell co — but on
# tradfi US news pages it ~always means extreme-UV lithography).
#
# These do NOT belong in ``_NOT_TICKERS`` (the SSOT) — they pollute only the
# off-watchlist exploration surface and would silently strip legitimate
# alias-paths on the held side (e.g. EPYC as a $cashtag for the AMD CPU
# brand may occasionally appear in penny-stock chatter the watchlist DOES
# want extracted, IF someone explicitly cashtags it). Keeping the filter
# local to this module avoids that cross-surface blast radius.
#
# Each token here verified live against 2026-05 article headlines as a pure
# false-positive on the off-watchlist path. If you add an entry, document
# the source phrase that drove it.
_OFF_WATCH_NOISE: frozenset[str] = frozenset({
    "MSN",      # "…- MSN" news-brand article-title suffix
    "TSX",      # "$BB.TSX" exchange suffix split
    "NA",       # "High-NA EUV", "N/A" → NA
    "EUV",      # extreme-UV lithography term
    "CAPEX",    # capital expenditure (financial concept word)
    "EPYC",     # AMD chip product line (not a tradable ticker on US tape)
    "AI",       # very high collision (Sportradar $AI cashtag is rare;
                # bare "AI" headline pollution is overwhelming)
})


def build_off_watchlist_mentions(watchlist, held, signals_list, *,
                                  min_articles: int = 1,
                                  min_avg_score: float = 4.0,
                                  limit: int = 12) -> dict:
    """Rank tickers mentioned in live news that are NOT on the watchlist.

    Args:
      ``watchlist``    — iterable of ticker symbols (the live universe)
      ``held``         — iterable of currently-held symbols (excluded; a
                         held off-watchlist name is already on the book,
                         not a discovery)
      ``signals_list`` — ``signals.get_top_signals()``-shaped rows; each
                         row's ``tickers`` field is the source of truth
                         (extracted by ``signals._extract_tickers``)
      ``min_articles`` — drop tickers with fewer matching live articles
      ``min_avg_score``— drop tickers whose mean ai_score is below this
      ``limit``        — cap the returned list

    ``heat`` mirrors ``watchlist_opportunities``:
      ``max_score × (1 + log1p(n)/3) × (1 + 0.25·urgent)``
    — the two surfaces are directly comparable so the operator can say
    "the loudest off-watchlist name (heat 12.4) outranks every name on
    my radar (top heat 9.1) — time to add it".

    Deterministic; ties break by (max_score, n, ticker) so output is stable.
    """
    watch_set = {str(t).upper().strip() for t in (watchlist or []) if t}
    held_set = {str(h).upper().strip() for h in (held or []) if h}
    # Off-watchlist *and* not currently held. A held ADR that isn't on the
    # written watchlist already has full position-level coverage elsewhere
    # (position_thesis, position_rationale, position_runrate); surfacing it
    # here would duplicate, not discover.
    exclude = watch_set | held_set | _OFF_WATCH_NOISE

    by_ticker: dict[str, dict] = {}
    for art in signals_list or []:
        if not isinstance(art, dict):
            continue
        tickers = art.get("tickers") or []
        try:
            ai = float(art.get("ai_score") or 0.0)
        except (TypeError, ValueError):
            ai = 0.0
        try:
            urg = int(art.get("urgency") or 0)
        except (TypeError, ValueError):
            urg = 0
        for raw in tickers:
            try:
                tk = str(raw).upper().strip()
            except Exception:
                continue
            if not tk or tk in exclude:
                continue
            bucket = by_ticker.setdefault(tk, {
                "ticker": tk,
                "n_articles": 0,
                "scores": [],
                "urgent": 0,
                "top_article": None,
                "top_score": -1.0,
            })
            bucket["n_articles"] += 1
            bucket["scores"].append(ai)
            if urg >= 1:
                bucket["urgent"] += 1
            if ai > bucket["top_score"]:
                bucket["top_score"] = ai
                bucket["top_article"] = art

    rows = []
    for bucket in by_ticker.values():
        n = bucket["n_articles"]
        if n < min_articles:
            continue
        scores = bucket["scores"]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        if avg_score < min_avg_score:
            continue
        max_score = max(scores) if scores else 0.0
        urgent = bucket["urgent"]
        heat = max_score * (1.0 + math.log1p(n) / 3.0) * (1.0 + 0.25 * urgent)
        top = bucket["top_article"] or {}
        rows.append({
            "ticker": bucket["ticker"],
            "n_articles": n,
            "avg_score": round(avg_score, 2),
            "max_score": round(max_score, 2),
            "urgent": urgent,
            "heat": round(heat, 3),
            "top_headline": top.get("title"),
            "top_source": top.get("source"),
            "top_url": top.get("url"),
        })

    rows.sort(key=lambda r: (r["heat"], r["max_score"], r["n_articles"], r["ticker"]),
              reverse=True)
    return {
        "discoveries": rows[: max(0, limit)],
        "n_scanned_articles": len(signals_list or []),
        "n_unique_off_watch": len(by_ticker),
        "n_surfaced": len(rows),
        "watchlist_size": len(watch_set),
        "held_size": len(held_set),
        "thresholds": {
            "min_articles": min_articles,
            "min_avg_score": min_avg_score,
        },
    }
