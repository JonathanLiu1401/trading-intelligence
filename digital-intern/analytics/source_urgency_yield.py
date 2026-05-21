"""Per-source urgent-yield audit — which collectors are signal vs noise.

Operator question this answers: **"Of all the collectors I run, which ones
are actually contributing urgent items the analyst sees, and which are
producing urgent flags that all get suppressed by the gates before
reaching Discord?"** Adjacent existing analytics describe related slices:

* ``analytics/news_arrival_rhythm.py`` — hour-of-day distribution of urgent
  articles per source. Says *when* news lands, not *quality*.
* ``analytics/publish_lag_audit.py`` — per-source publication latency.
  Says *how fresh* a source is, not its urgent-yield ratio.
* ``ArticleStore.source_throughput`` — rate change per source. Says
  *how fast* a source is moving, not what fraction crosses the urgent
  threshold or survives the alert-side gates.

This builder bridges that gap with three rates per source over the window:

* ``urgent_rate``         — ``urgent / total``: how often a source's
                            articles cross the urgency-1 threshold.
* ``alerted_rate``        — ``alerted / total``: how often a source's
                            articles get a real Discord push (urgency=2,
                            i.e. survived every alert-side gate —
                            quote-widget, recap, low-authority,
                            cross-cycle dedup, paraphrase).
* ``suppression_rate``    — ``(urgent − alerted) / urgent``: of a
                            source's *urgent-flagged* rows, how many got
                            gate-dropped before Discord. A high value
                            means the urgency head + downstream gates
                            are doing useful work on this source's
                            noise floor; a near-zero value means almost
                            every urgent-flagged item from this source
                            survives to push.

Three operator-actionable verdicts per source (assigned when the source
has enough samples to be statistically meaningful):

* ``NOISY``     — high urgent_rate AND high suppression_rate. The source
                  is signal-rich but most urgent flags get gate-dropped.
                  Useful candidate for ML-threshold tuning.
* ``CLEAN``     — urgent_rate >= floor AND suppression_rate < 0.20.
                  Most urgent flags survive every gate — this source's
                  urgent items consistently warrant a push.
* ``QUIET``     — no urgent flags in the window. Neutral observation;
                  some sources are sector-specific and idle most days.

Pure / no DB / no LLM (composes pre-fetched article rows; mirrors
``build_news_arrival_rhythm`` / ``build_briefing_coverage_audit``
discipline). Never raises on garbage inputs. Advisory only — read-only
operator panel. The route layer is the SQL adapter and applies the
canonical ``_LIVE_ONLY_CLAUSE`` so backtest rows can't leak into the
audit (invariant #5 preserved).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone


# Display cap on the per-source list. The aggregate totals always reflect
# every source; the cap truncates the ranked-cards output only.
DEFAULT_TOP_SOURCES = 15

# Default window. One full circadian cycle on the news desk. Route-layer
# clamps keep callers from running this unbounded.
DEFAULT_WINDOW_HOURS = 24

# Minimum total articles a source needs in the window before a verdict is
# assigned. Below this, a single oddly-scored row would dominate the rate
# computation. Sources below the floor are still returned (so the operator
# sees them) but with ``verdict="UNKNOWN"``.
DEFAULT_MIN_SAMPLES = 20

# Verdict thresholds — chosen so the typical analyst-facing source falls
# into CLEAN by default. Tuned against the 6h live-DB snapshot
# (2026-05-20): GN: Nvidia / GN: dividend buyback / stocktwits / YahooFinance
# all land below 25% suppression; Finnhub / GN: stock market land above.
_URGENT_RATE_FLOOR = 0.02       # 2% urgent floor for non-QUIET verdicts
_NOISY_SUPPRESSION = 0.30       # >=30% of urgent flags suppressed → NOISY
_CLEAN_SUPPRESSION = 0.20       # <20% suppressed AND urgent_rate ≥ floor → CLEAN


def _parse_first_seen(ts) -> datetime | None:
    """Tolerate aware/naive ISO strings — same convention as
    ``news_arrival_rhythm._parse_first_seen``."""
    if not ts:
        return None
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(ts).strip().replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _empty_envelope(now: datetime, hours: int,
                    min_samples: int, top_sources: int) -> dict:
    """Empty-state skeleton — same key set as a populated response so the
    UI binding never sees a missing field. Mirrors the empty-envelope
    discipline of ``news_arrival_rhythm`` / ``briefing_coverage_audit``."""
    return {
        "as_of": now.isoformat(timespec="seconds"),
        "state": "NO_DATA",
        "headline": "No articles in the window — nothing to score.",
        "window_hours": hours,
        "min_samples": min_samples,
        "top_sources_cap": top_sources,
        "n_articles_scanned": 0,
        "n_articles_kept": 0,
        "n_sources": 0,
        "n_noisy": 0,
        "n_clean": 0,
        "n_quiet": 0,
        "n_unknown": 0,
        "totals": {
            "articles": 0, "urgent": 0, "alerted": 0,
            "urgent_rate": None, "alerted_rate": None,
            "suppression_rate": None,
        },
        "sources": [],
    }


def _verdict(total: int, urgent: int, alerted: int,
             *, min_samples: int) -> str:
    """Pure verdict policy — pinned exactly by tests so a future threshold
    edit must update the pinned constants too."""
    if total < min_samples:
        return "UNKNOWN"
    if urgent == 0:
        return "QUIET"
    urgent_rate = urgent / total
    if urgent_rate < _URGENT_RATE_FLOOR:
        return "QUIET"
    suppression_rate = (urgent - alerted) / urgent
    if suppression_rate >= _NOISY_SUPPRESSION:
        return "NOISY"
    if suppression_rate < _CLEAN_SUPPRESSION:
        return "CLEAN"
    # Middle band — not noisy enough to flag, not clean enough to certify.
    return "MIXED"


def build_source_urgency_yield(
    articles,
    hours: int = DEFAULT_WINDOW_HOURS,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    top_sources: int = DEFAULT_TOP_SOURCES,
    now: datetime | None = None,
) -> dict:
    """Per-source urgent-yield audit over the last ``hours`` window.

    ``articles``: iterable of dicts with keys ``source``, ``urgency``,
    ``first_seen``. Surplus keys (``title``, ``ai_score``, …) ignored.

    ``hours``: lookback. Articles outside ``[now-hours, now]`` are dropped.

    ``min_samples``: minimum article count a source needs for a verdict.
    Below the floor, the row is returned but verdict is ``"UNKNOWN"``.

    ``top_sources``: display cap on ``sources``. The aggregate ``totals``
    always reflect every kept article.

    Pure / never raises. Non-list input → NO_DATA envelope.
    """
    now = now or datetime.now(timezone.utc)

    if not isinstance(articles, (list, tuple)) or hours <= 0:
        return _empty_envelope(now, hours, min_samples, top_sources)

    window_start = now - timedelta(hours=hours)

    # Per-source running totals.
    per_source: dict[str, dict[str, int]] = {}
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
        dt = _parse_first_seen(art.get("first_seen"))
        if dt is None:
            continue
        if dt < window_start or dt > now:
            continue
        source = art.get("source")
        if not source or not isinstance(source, str):
            source = "(unknown)"
        slot = per_source.setdefault(source, {
            "total": 0, "urgent": 0, "alerted": 0,
        })
        slot["total"] += 1
        if urg >= 1:
            slot["urgent"] += 1
        if urg >= 2:
            slot["alerted"] += 1
        n_kept += 1

    if n_kept == 0:
        env = _empty_envelope(now, hours, min_samples, top_sources)
        env["n_articles_scanned"] = n_scanned
        return env

    # Per-source rollup with verdict + rates.
    source_rows: list[dict] = []
    for src, counts in per_source.items():
        total = counts["total"]
        urgent = counts["urgent"]
        alerted = counts["alerted"]
        urgent_rate = round(urgent / total, 4) if total else 0.0
        alerted_rate = round(alerted / total, 4) if total else 0.0
        suppression_rate = (
            round((urgent - alerted) / urgent, 4) if urgent else None
        )
        verdict = _verdict(total, urgent, alerted, min_samples=min_samples)
        source_rows.append({
            "source": src,
            "total": total,
            "urgent": urgent,
            "alerted": alerted,
            "urgent_rate": urgent_rate,
            "alerted_rate": alerted_rate,
            "suppression_rate": suppression_rate,
            "verdict": verdict,
        })

    # Verdict tally — operator panel headline.
    n_noisy = sum(1 for r in source_rows if r["verdict"] == "NOISY")
    n_clean = sum(1 for r in source_rows if r["verdict"] == "CLEAN")
    n_quiet = sum(1 for r in source_rows if r["verdict"] == "QUIET")
    n_unknown = sum(1 for r in source_rows if r["verdict"] == "UNKNOWN")

    # Ranking: NOISY first (operator-actionable), then by urgent count desc
    # (high-urgency sources are interesting), alphabetical tie-break for a
    # byte-stable card order across runs.
    _verdict_rank = {"NOISY": 0, "MIXED": 1, "CLEAN": 2,
                     "QUIET": 3, "UNKNOWN": 4}
    source_rows.sort(
        key=lambda r: (_verdict_rank.get(r["verdict"], 5),
                       -r["urgent"], r["source"]),
    )
    n_sources = len(source_rows)
    if top_sources > 0:
        source_rows = source_rows[:top_sources]
    else:
        source_rows = []

    # Aggregate totals across every kept row.
    tot_total = sum(per_source[s]["total"] for s in per_source)
    tot_urgent = sum(per_source[s]["urgent"] for s in per_source)
    tot_alerted = sum(per_source[s]["alerted"] for s in per_source)
    tot_urgent_rate = round(tot_urgent / tot_total, 4) if tot_total else None
    tot_alerted_rate = round(tot_alerted / tot_total, 4) if tot_total else None
    tot_suppression_rate = (
        round((tot_urgent - tot_alerted) / tot_urgent, 4)
        if tot_urgent else None
    )

    state = "STABLE" if n_kept >= 20 else "SPARSE"
    if state == "SPARSE":
        headline = (
            f"Sparse — only {n_kept} article(s) in the last {hours}h "
            f"(need ≥20 for a stable read)."
        )
    elif n_noisy > 0:
        top_noisy = next(
            (r for r in source_rows if r["verdict"] == "NOISY"), None
        )
        if top_noisy is not None:
            headline = (
                f"{n_noisy} NOISY source(s) — top: {top_noisy['source']} "
                f"({int(round((top_noisy['suppression_rate'] or 0) * 100))}% "
                f"of urgent flags suppressed)."
            )
        else:
            headline = f"{n_noisy} NOISY source(s) flagged."
    else:
        headline = (
            f"{n_clean} CLEAN source(s), no noisy sources flagged "
            f"({tot_urgent}/{tot_total} urgent → "
            f"{tot_alerted}/{tot_urgent or 1} alerted)."
        )

    return {
        "as_of": now.isoformat(timespec="seconds"),
        "state": state,
        "headline": headline,
        "window_hours": hours,
        "min_samples": min_samples,
        "top_sources_cap": top_sources,
        "n_articles_scanned": n_scanned,
        "n_articles_kept": n_kept,
        "n_sources": n_sources,
        "n_noisy": n_noisy,
        "n_clean": n_clean,
        "n_quiet": n_quiet,
        "n_unknown": n_unknown,
        "totals": {
            "articles": tot_total,
            "urgent": tot_urgent,
            "alerted": tot_alerted,
            "urgent_rate": tot_urgent_rate,
            "alerted_rate": tot_alerted_rate,
            "suppression_rate": tot_suppression_rate,
        },
        "sources": source_rows,
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
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)
              ).isoformat(timespec="seconds")
    rows = conn.execute(
        f"""SELECT source, urgency, first_seen
              FROM articles
             WHERE {_LIVE_ONLY_CLAUSE}
               AND first_seen >= ?
             ORDER BY first_seen DESC
             LIMIT 20000""",
        (cutoff,),
    ).fetchall()
    conn.close()
    arts = [{"source": r[0], "urgency": r[1], "first_seen": r[2]}
            for r in rows]
    rep = build_source_urgency_yield(arts)
    print(json.dumps(rep, indent=2, default=str))
