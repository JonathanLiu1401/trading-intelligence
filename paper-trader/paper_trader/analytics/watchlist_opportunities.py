"""Watchlist names lighting up in the news that you do NOT own yet.

Every existing analytics surface here is position-centric — drawdown,
track-record, thesis-drift, hold-discipline, game-plan all describe what the
book already holds. None answer the trader's other standing question: *"what
is the news flow screaming about that I have no exposure to?"* This is the
orthogonal panel — the missed-opportunity radar.

``build_watchlist_opportunities`` is a pure function over a single
already-fetched ``signals.get_top_signals()`` list (the caller fetches once;
this tallies per ticker — no N-query fan-out). Pure, **never raises**.
"""
from __future__ import annotations

from .ticker_dossier import articles_mentioning, _f


def build_watchlist_opportunities(watchlist, held, signals_list, *,
                                   min_articles: int = 1,
                                   min_avg_score: float = 4.0,
                                   limit: int = 12) -> dict:
    """Rank not-yet-held watchlist names by live news heat.

    Args:
      ``watchlist``    — iterable of ticker symbols (the live universe)
      ``held``         — iterable of currently-held symbols (excluded)
      ``signals_list`` — ``signals.get_top_signals()``-shaped rows
      ``min_articles`` — drop names with fewer matching live articles
      ``min_avg_score``— drop names whose mean ai_score is below this
      ``limit``        — cap the returned list

    ``heat`` = max_score × (1 + log1p(n)/3) × (1 + 0.25·urgent) — rewards a
    strong headline, more so when several corroborate and when any is urgent,
    without letting a flood of weak mentions outrank one decisive catalyst.
    Deterministic; ties break by (max_score, n, ticker) so output is stable.
    """
    import math

    held_set = {str(h).upper().strip() for h in (held or [])}
    universe = []
    for t in (watchlist or []):
        tk = str(t).upper().strip()
        if tk and tk not in held_set and tk not in universe:
            universe.append(tk)

    rows = []
    for tk in universe:
        arts = articles_mentioning(tk, signals_list)
        n = len(arts)
        if n < min_articles:
            continue
        scores = [s for s in (_f(a.get("ai_score")) for a in arts) if s is not None]
        if not scores:
            continue
        avg_score = sum(scores) / len(scores)
        if avg_score < min_avg_score:
            continue
        max_score = max(scores)
        urgent = sum(1 for a in arts if (_f(a.get("urgency")) or 0) >= 1)
        heat = max_score * (1.0 + math.log1p(n) / 3.0) * (1.0 + 0.25 * urgent)
        top = max(arts, key=lambda a: _f(a.get("ai_score")) or 0.0)
        rows.append({
            "ticker": tk,
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
        "opportunities": rows[: max(0, limit)],
        "n_scanned": len(universe),
        "n_surfaced": len(rows),
        "thresholds": {"min_articles": min_articles,
                       "min_avg_score": min_avg_score},
    }
