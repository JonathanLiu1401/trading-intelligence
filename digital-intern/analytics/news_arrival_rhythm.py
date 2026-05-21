"""News arrival rhythm — per-source hour-of-day distribution of urgent articles.

Operator question this answers: **"When does urgent news actually arrive,
and from which source?"** Existing surfaces describe related but distinct
slices:

* ``analytics/collector_uptime.py`` — silence GAPS per source in a 24h
  window (a backward-looking outage detector). Says nothing about
  *cadence patterns* — a source that fires 12 articles in two hours and
  is dark the other 22 looks fine to uptime.
* ``analytics/hourly_ingestion.py`` — aggregate ingestion rate; doesn't
  partition by urgency or source.
* ``analytics/source_lead_time.py`` — *relative* speed of one source
  against the median for the same cluster. Says nothing about *when*
  any source breaks news.

This builder is the *hour-of-day heatmap*: for the last ``hours`` window,
articles with ``urgency >= min_urgency`` are bucketed into (hour-of-day UTC,
source) cells. Per-source: 24 hourly counts, total, peak hour, quiet
hours (zero count). Across all sources: hour-of-day distribution + the
longest contiguous zero-count run (the "global quiet window").

Pure / no DB / no LLM (composes pre-fetched article rows; mirrors
``build_event_threads`` / ``build_portfolio_signals`` discipline). Never
raises on garbage inputs. Advisory only — read-only operator panel.

Hour-of-day is UTC (the daemon's clock and the articles.db ``first_seen``
column both default to UTC). The "global quiet window" wraps the 24h
cycle so 23h→0h→1h is reported as a 3-hour stretch.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

# Default ranking cap; the aggregate hour-of-day distribution always
# contains every source's contribution (the cap is a *display* limit,
# not a counting limit).
DEFAULT_TOP_SOURCES = 10

# Default urgency floor — `1` = "needs alert" or higher. Setting to `0`
# would include the noise floor of every scored article, which makes the
# heatmap useless for operator triage. `2` = "already alerted".
DEFAULT_MIN_URGENCY = 1

# Default window. Sized to cover one full circadian cycle on the news
# desk; clamps in the route layer keep this from running unbounded.
DEFAULT_WINDOW_HOURS = 24


def _parse_first_seen(ts) -> datetime | None:
    """Tolerate aware/naive ISO strings — the same convention as
    ``portfolio_signals._parse_ts`` / ``decision_drought._parse_ts``.
    A naive timestamp is treated as UTC (the daemon writes UTC)."""
    if not ts:
        return None
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(ts).strip().replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _longest_zero_run_circular(counts) -> tuple[int, int, int]:
    """Longest run of consecutive zeros in a 24-element circular array.

    Returns ``(length, start_hour, end_hour)``. ``end_hour`` is the LAST
    zero hour in the run (inclusive). For an all-zero input, returns
    ``(24, 0, 23)``. For an all-nonzero input, returns ``(0, -1, -1)``.

    The run wraps the 24h cycle: a quiet stretch from hour 23 through
    hour 1 reads as length 3, start 23, end 1.
    """
    if not counts:
        return (0, -1, -1)
    n = len(counts)
    # If everything is zero, the run is the whole cycle.
    if all(c == 0 for c in counts):
        return (n, 0, n - 1)
    # If nothing is zero, no run.
    if all(c > 0 for c in counts):
        return (0, -1, -1)
    # Walk twice around the cycle to capture the wrap-around case in
    # one pass; cap the running length at n so a full-zero cycle (already
    # handled above) can never inflate.
    best_len, best_start, best_end = 0, -1, -1
    cur_len, cur_start = 0, -1
    for i in range(2 * n):
        idx = i % n
        if counts[idx] == 0:
            if cur_len == 0:
                cur_start = idx
            cur_len += 1
            if cur_len > best_len and cur_len <= n:
                best_len = cur_len
                best_start = cur_start
                best_end = idx
        else:
            cur_len = 0
            cur_start = -1
    return (best_len, best_start, best_end)


def _empty_envelope(now: datetime, hours: int, min_urgency: int,
                    top_sources: int) -> dict:
    """Empty-state skeleton — same key set as a populated response so the
    UI binding never sees a missing field."""
    return {
        "as_of": now.isoformat(timespec="seconds"),
        "state": "NO_DATA",
        "headline": ("No urgent articles in the window — nothing to plot."),
        "window_hours": hours,
        "min_urgency": min_urgency,
        "n_articles_scanned": 0,
        "n_articles_kept": 0,
        "n_sources": 0,
        "hour_of_day_totals": [0] * 24,
        "peak_hour": None,
        "trough_hour": None,
        "quiet_window": {"length_hours": 0,
                         "start_hour": None, "end_hour": None},
        "sources": [],
        "top_sources_cap": top_sources,
    }


def build_news_arrival_rhythm(articles,
                              hours: int = DEFAULT_WINDOW_HOURS,
                              min_urgency: int = DEFAULT_MIN_URGENCY,
                              top_sources: int = DEFAULT_TOP_SOURCES,
                              now: datetime | None = None) -> dict:
    """Hour-of-day urgent-article distribution per source.

    ``articles``: iterable of dicts with keys ``source``, ``urgency``,
    ``first_seen``. Surplus keys (``title``, ``ai_score``, …) are ignored.

    ``hours``: lookback window. Articles with ``first_seen`` older than
    ``now - hours`` are dropped *before* hour-of-day bucketing.

    ``min_urgency``: floor — articles with ``urgency < min_urgency``
    contribute neither to the per-source nor to the aggregate.

    ``top_sources``: ranking cap on the ``sources`` list. The aggregate
    ``hour_of_day_totals`` always includes every kept article (the cap
    truncates the per-source breakdown, not the counts).

    Pure / never raises. Non-list input → empty envelope.
    """
    now = now or datetime.now(timezone.utc)

    if not isinstance(articles, (list, tuple)):
        return _empty_envelope(now, hours, min_urgency, top_sources)
    if hours <= 0:
        return _empty_envelope(now, hours, min_urgency, top_sources)

    window_start = now - timedelta(hours=hours)

    # Per-source 24-element hourly counts. dict-of-lists chosen so a
    # source that fires in a single hour produces a sparse cell rather
    # than a populated row across every bucket.
    per_source: dict[str, list[int]] = {}
    aggregate = [0] * 24
    n_scanned = 0
    n_kept = 0

    for art in articles:
        if not isinstance(art, dict):
            continue
        n_scanned += 1
        urg = art.get("urgency")
        try:
            urg = int(urg) if urg is not None else 0
        except (TypeError, ValueError):
            continue
        if urg < min_urgency:
            continue
        dt = _parse_first_seen(art.get("first_seen"))
        if dt is None:
            continue
        if dt < window_start or dt > now:
            continue
        source = art.get("source")
        if not source or not isinstance(source, str):
            source = "(unknown)"
        hour = dt.hour
        per_source.setdefault(source, [0] * 24)[hour] += 1
        aggregate[hour] += 1
        n_kept += 1

    if n_kept == 0:
        env = _empty_envelope(now, hours, min_urgency, top_sources)
        env["n_articles_scanned"] = n_scanned
        return env

    # Per-source rollup with derived fields. Quiet hours are the indices
    # with zero count — the operator wants to see them at a glance.
    source_rows: list[dict] = []
    for src, counts in per_source.items():
        total = sum(counts)
        peak_hour = max(range(24), key=lambda h: (counts[h], -h))
        quiet_hours = [h for h, c in enumerate(counts) if c == 0]
        # Defensive: an all-zero per_source row shouldn't be possible
        # (we only insert on an incremented bucket), but if a future
        # change ever broke that, the peak_hour fallback would lie.
        if total == 0:
            continue
        source_rows.append({
            "source": src,
            "total": total,
            "hourly_counts": counts,
            "peak_hour": peak_hour,
            "n_quiet_hours": len(quiet_hours),
        })

    # Most-active first; alphabetical tie-break so the card order is
    # stable across runs.
    source_rows.sort(key=lambda r: (-r["total"], r["source"]))
    n_sources = len(source_rows)
    if top_sources > 0:
        source_rows = source_rows[:top_sources]
    else:
        source_rows = []

    peak_hour = max(range(24), key=lambda h: (aggregate[h], -h))
    # trough_hour is the hour with the LOWEST nonzero count if any
    # nonzero hours exist; falls back to the earliest zero hour otherwise.
    nonzero_hours = [h for h, c in enumerate(aggregate) if c > 0]
    if len(nonzero_hours) == 24:
        trough_hour = min(range(24), key=lambda h: (aggregate[h], h))
    else:
        # Mixed zero / nonzero — report the earliest zero hour as the
        # trough for the operator (it's the most actionable "go look").
        zero_hours = [h for h in range(24) if aggregate[h] == 0]
        trough_hour = zero_hours[0]

    qw_len, qw_start, qw_end = _longest_zero_run_circular(aggregate)

    if n_kept >= 5:
        state = "STABLE"
    else:
        state = "SPARSE"

    top = source_rows[0] if source_rows else None
    if state == "SPARSE":
        headline = (
            f"Sparse — only {n_kept} urgent article(s) in the last "
            f"{hours}h (need ≥5 for a stable rhythm read)."
        )
    else:
        # The top-source clause tells the operator who's noisiest right
        # now; the peak-hour clause tells them when. Both are the two
        # things the heatmap renders large.
        top_clause = ""
        if top is not None:
            top_clause = (f" {top['source']} is loudest "
                          f"({top['total']} of {n_kept} urgent).")
        quiet_clause = ""
        if qw_len >= 3:
            quiet_clause = (
                f" Longest quiet window {qw_len}h "
                f"({qw_start:02d}:00–{qw_end:02d}:59 UTC).")
        headline = (
            f"Peak urgent-news hour {peak_hour:02d}:00 UTC "
            f"({aggregate[peak_hour]} article(s)).{top_clause}{quiet_clause}"
        )

    return {
        "as_of": now.isoformat(timespec="seconds"),
        "state": state,
        "headline": headline,
        "window_hours": hours,
        "min_urgency": min_urgency,
        "n_articles_scanned": n_scanned,
        "n_articles_kept": n_kept,
        "n_sources": n_sources,
        "hour_of_day_totals": aggregate,
        "peak_hour": peak_hour,
        "trough_hour": trough_hour,
        "quiet_window": {
            "length_hours": qw_len,
            "start_hour": qw_start if qw_len > 0 else None,
            "end_hour": qw_end if qw_len > 0 else None,
        },
        "sources": source_rows,
        "top_sources_cap": top_sources,
    }


if __name__ == "__main__":  # smoke test against the live DB
    import json
    import sqlite3
    import sys
    from pathlib import Path

    BASE = Path(__file__).resolve().parents[1]
    if str(BASE) not in sys.path:
        sys.path.insert(0, str(BASE))

    from storage.article_store import _LIVE_ONLY_CLAUSE, _get_db_path  # type: ignore

    db = _get_db_path()
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=15)
    rows = conn.execute(
        f"""SELECT source, urgency, first_seen
              FROM articles
             WHERE {_LIVE_ONLY_CLAUSE}
               AND urgency >= 1
               AND first_seen >= ?
             ORDER BY first_seen DESC
             LIMIT 5000""",
        ((datetime.now(timezone.utc) - timedelta(hours=24))
         .isoformat(timespec="seconds"),),
    ).fetchall()
    conn.close()
    arts = [{"source": r[0], "urgency": r[1], "first_seen": r[2]}
            for r in rows]
    rep = build_news_arrival_rhythm(arts)
    print(json.dumps(rep, indent=2, default=str))
