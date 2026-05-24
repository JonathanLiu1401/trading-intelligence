"""Persistent watchlist opportunity — the TIME dimension on missed setups.

``watchlist_opportunities`` already lists hot unheld watchlist names — but
it is a snapshot. ``idle_opportunity`` already lists the high-score watchlist
arrivals during a NO_DECISION drought — but only fires *while* the bot is
dark. Neither answers the longest-running operator question:

  **"How long has this name been screaming on the watchlist while the book
  had zero exposure?"**

The current pathology (live, 2026-05-24): the book has been 100% cash for
days, ``decision_paralysis`` reports ``PASSIVE_LOOP``, and NVDA has been
scoring ≥7 on the watchlist for hours — every individual cycle's HOLD CASH
looks defensible (Sunday, no high-score signals in last 2h) but the
*persistence* of the setup is invisible in the snapshot. A trader's "I keep
missing NVDA" is exactly this gap.

The TIME-aged dimension is the discriminator vs the four cousins:

* ``watchlist_opportunities`` — snapshot of the *current* news heat. No
  history; an NVDA spike that started 30 min ago looks identical to one
  that has been hot for 47h.
* ``idle_opportunity`` — only fires during a NO_DECISION drought (a
  specific failure mode). A PASSIVE_LOOP of pure HOLD decisions does not
  count as a drought, so the panel reports OK even when the book has been
  in cash for 5 days.
* ``opportunity_cost_skill`` — backward forward-return read on *past*
  cash-decisions. Hindsight, not standing setup.
* ``watchlist_news_silence`` — the *inverse* lens (which watchlist names
  have NO news). The complement is needed too: which have *persistent* news.

This builder takes a single already-fetched ``signals.get_top_signals()``
list, buckets articles per (ticker, hour-of-first_seen), and for each unheld
watchlist ticker computes the longest contiguous run of hours in which at
least one article scored ≥ ``min_score``, plus the *current* run anchored at
the latest bin. A ticker is "persistent" when the current run meets
``min_persistence_hours``.

Pure function — never raises on garbage rows. The endpoint owns the
``signals.get_top_signals`` read (the ``watchlist_opportunities`` precedent,
single fetch, no N-query fan-out).

**Observational only** — never gates Opus, never injected into the decision
prompt, no caps (AGENTS.md invariants #2/#12 — the ``watchlist_opportunities``
/ ``idle_opportunity`` / ``opportunity_cost_skill`` precedent).
"""
from __future__ import annotations

from datetime import datetime, timezone

from .ticker_dossier import _f, articles_mentioning

# Per-article ai_score threshold for a bin to count as "hot". Mirrors
# ``idle_opportunity.DEFAULT_MIN_AI_SCORE`` so a name flagged persistent
# here would also have shown up there under the right drought conditions —
# the two surfaces are TIME-axis complements over the same heat predicate.
DEFAULT_MIN_AI_SCORE = 6.0

# Minimum contiguous hot-hours anchored at the most recent bin before a
# ticker is surfaced as PERSISTENT. 6h is the operator-readable "more than
# a session-half"; below that the existing snapshot panel
# (``watchlist_opportunities``) is already sufficient. Module-owned so
# tests read it directly.
DEFAULT_MIN_PERSISTENCE_HOURS = 6.0

# Total lookback window. 48h is the operator's natural span for "did this
# story break a day or two ago and still hasn't been bought?" — wider than
# ``get_top_signals``'s default 2h, narrower than the 5d
# ``cash_redeployment_latency_skill`` window.
DEFAULT_WINDOW_HOURS = 48.0

# Cap surfaced rows — same shape as ``watchlist_opportunities.limit``.
DEFAULT_LIMIT = 12


def _parse_ts(ts):
    """Tolerant ISO parser — returns aware UTC datetime or None."""
    if not ts:
        return None
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def _hour_bins(arts, now, window_hours, min_score):
    """Return (bin_hits, max_score, n_articles_in_window, top_article).

    ``bin_hits`` is a list of bools, length ``window_hours`` (rounded
    down), oldest-first. ``bin_hits[i]`` is True iff at least one article
    in the [now - (window-i)h, now - (window-i-1)h) bin scored ≥ min_score.
    Anchoring at ``now`` means ``bin_hits[-1]`` is the most recent bin.
    """
    n_bins = max(1, int(window_hours))
    hits = [False] * n_bins
    n_in_window = 0
    top = None
    top_score = -1.0
    max_score = 0.0
    for a in arts:
        if not isinstance(a, dict):
            continue
        ts = _parse_ts(a.get("first_seen"))
        if ts is None:
            continue
        # Hours back from now (positive = older).
        delta_h = (now - ts).total_seconds() / 3600.0
        if delta_h < 0 or delta_h >= window_hours:
            continue
        n_in_window += 1
        score = _f(a.get("ai_score")) or 0.0
        if score > max_score:
            max_score = score
        if score > top_score:
            top_score = score
            top = a
        if score >= min_score:
            # Newest bin is index n_bins-1. An article 0..1h old →
            # delta_h in [0, 1) → bin index n_bins-1.
            idx = n_bins - 1 - int(delta_h)
            if 0 <= idx < n_bins:
                hits[idx] = True
    return hits, max_score, n_in_window, top


def _longest_run(hits):
    """Length of the longest contiguous True-run in ``hits``."""
    best = 0
    run = 0
    for h in hits:
        if h:
            run += 1
            if run > best:
                best = run
        else:
            run = 0
    return best


def _current_run(hits):
    """Length of the contiguous True-run ending at ``hits[-1]`` (0 if the
    most recent bin is cold)."""
    run = 0
    for h in reversed(hits):
        if h:
            run += 1
        else:
            break
    return run


def _headline(state: str, n_pers: int, top: dict | None,
              min_persistence_hours: float, min_score: float) -> str:
    """One-sentence operator headline matching the ``idle_opportunity`` /
    ``watchlist_opportunities`` voice."""
    if state == "NO_DATA":
        return ("Persistent watchlist opportunity: no signals visible in "
                "the lookback window.")
    if state == "NO_PERSISTENT":
        return ("Persistent watchlist opportunity: no unheld watchlist "
                f"name has held ai_score ≥{min_score:.1f} for "
                f"{min_persistence_hours:.1f}h+ — nothing standing.")
    # FLAG — at least one persistent miss
    if top is None:
        return ("Persistent watchlist opportunity: "
                f"{n_pers} unheld watchlist name(s) standing at "
                f"ai_score ≥{min_score:.1f} for "
                f"{min_persistence_hours:.1f}h+.")
    run_h = top.get("current_run_hours")
    run_s = f"{run_h:.1f}h" if isinstance(run_h, (int, float)) else "?h"
    max_s = top.get("max_score")
    score_s = f"{max_s:.1f}" if isinstance(max_s, (int, float)) else "?"
    others = max(0, n_pers - 1)
    plural = f" (+{others} more)" if others else ""
    return (
        f"Persistent watchlist opportunity: {top.get('ticker')} has held "
        f"ai_score ≥{min_score:.1f} for {run_s} (max {score_s}) with zero "
        f"book exposure{plural}."
    )


def build_persistent_watchlist_opportunity(
    watchlist,
    held,
    signals_list,
    *,
    now: datetime | None = None,
    min_score: float = DEFAULT_MIN_AI_SCORE,
    min_persistence_hours: float = DEFAULT_MIN_PERSISTENCE_HOURS,
    window_hours: float = DEFAULT_WINDOW_HOURS,
    limit: int = DEFAULT_LIMIT,
) -> dict:
    """Rank unheld watchlist names by *contiguous current run* of hours in
    which at least one article scored ≥ ``min_score``.

    Args:
      ``watchlist`` — iterable of ticker symbols (live universe;
                      typically ``strategy.WATCHLIST`` upper-cased)
      ``held``      — iterable of currently-held symbols (excluded)
      ``signals_list`` — ``signals.get_top_signals()``-shaped rows;
                         caller fetches once with a window ≥ window_hours
      ``now``       — injectable UTC datetime (default datetime.now(UTC))
      ``min_score`` — per-article ai_score threshold for a bin to be hot
      ``min_persistence_hours`` — minimum *current* run to surface a ticker
      ``window_hours`` — total lookback for bin counting
      ``limit``     — cap on surfaced rows

    Returns a JSON-ready dict. **Never raises** on garbage rows: each row
    that fails to parse degrades to a skip; an empty input degrades to
    ``state="NO_DATA"``.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    held_set: set[str] = set()
    for h in (held or []):
        try:
            tk = str(h).upper().strip()
            if tk:
                held_set.add(tk)
        except (TypeError, ValueError):
            continue

    universe: list[str] = []
    seen: set[str] = set()
    for t in (watchlist or []):
        try:
            tk = str(t).upper().strip()
        except (TypeError, ValueError):
            continue
        if tk and tk not in held_set and tk not in seen:
            seen.add(tk)
            universe.append(tk)

    # NO_DATA is reserved for "no signals at all to scan" OR "watchlist
    # not provided at all". The case where every watchlist name is HELD
    # (universe collapses to empty after the held filter) is a meaningful
    # NO_PERSISTENT — "nothing standing on a name you don't already own"
    # — and must flow through to the loop so the verdict matches the chat
    # contract. Tests pin this distinction.
    if not signals_list or not (watchlist or []):
        return {
            "as_of": now.isoformat(),
            "state": "NO_DATA",
            "headline": _headline("NO_DATA", 0, None,
                                  min_persistence_hours, min_score),
            "min_score": min_score,
            "min_persistence_hours": min_persistence_hours,
            "window_hours": window_hours,
            "n_scanned": len(universe),
            "n_persistent": 0,
            "opportunities": [],
        }

    rows: list[dict] = []
    for tk in universe:
        arts = articles_mentioning(tk, signals_list)
        if not arts:
            continue
        hits, max_score, n_in_window, top = _hour_bins(
            arts, now, window_hours, min_score)
        if n_in_window == 0:
            continue
        cur_run = _current_run(hits)
        longest_run = _longest_run(hits)
        n_hot = sum(1 for h in hits if h)
        rows.append({
            "ticker": tk,
            "current_run_hours": float(cur_run),
            "longest_run_hours": float(longest_run),
            "n_hot_bins": n_hot,
            "n_total_bins_in_window": len(hits),
            "n_articles_in_window": n_in_window,
            "max_score": round(max_score, 2),
            "top_headline": (top or {}).get("title"),
            "top_source": (top or {}).get("source"),
            "top_url": (top or {}).get("url"),
            "top_first_seen": (top or {}).get("first_seen"),
        })

    # Surface only persistent names. Sort by current run (descending), then
    # max score, then ticker for stable output (matches
    # ``watchlist_opportunities`` ordering discipline).
    persistent = [r for r in rows
                  if r["current_run_hours"] >= min_persistence_hours]
    persistent.sort(
        key=lambda r: (r["current_run_hours"], r["max_score"], r["ticker"]),
        reverse=True,
    )
    persistent = persistent[: max(0, limit)]

    if not persistent:
        state = "NO_PERSISTENT"
        headline = _headline(state, 0, None,
                             min_persistence_hours, min_score)
    else:
        state = "FLAG"
        headline = _headline(state, len(persistent), persistent[0],
                             min_persistence_hours, min_score)

    return {
        "as_of": now.isoformat(),
        "state": state,
        "headline": headline,
        "min_score": min_score,
        "min_persistence_hours": min_persistence_hours,
        "window_hours": window_hours,
        "n_scanned": len(universe),
        "n_persistent": len(persistent),
        "opportunities": persistent,
    }
