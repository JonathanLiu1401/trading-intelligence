"""Quote-widget noise audit (read-only).

The live ticker-tape / quote-widget pseudo-article class — spaceless
price-tick titles ("NVDANVIDIA Corporation227.13-8.61(-3.65%)"),
parenthesised percent-change tapes ("NQ=FNasdaq 100 Jun
2629,215.25-472.50(-1.59%)"), Moomoo/Futu share-card listings ("$NVIDIA
(NVDA.US)$ - Moomoo"), and Yahoo screener-tape leads
("[YF/most_actives] MU ...") — is now gated on FOUR surfaces, mirroring
the recap-template gate audited by ``analytics.recap_template_audit``:

  1. ``collectors.web_scraper._looks_like_quote_widget`` — drops at
     ingestion for scraped Yahoo / Bloomberg pages.
  2. ``watchers.urgency_scorer.score_batch`` — pre-filters BEFORE the
     Sonnet call (saves quota + stops training-pool poisoning for the
     paths web_scraper does NOT cover: yahoo_ticker_rss / finnhub /
     google_news / market_movers).
  3. ``watchers.alert_agent.send_urgent_alert`` — suppresses standalone
     🚨 BREAKING pushes on caught rows.
  4. ``storage.article_store.score_pending`` — pre-floors before the ML
     urgency head writes ``urgency=1`` with ``score_source='ml'`` (the
     model-confident urgent path that bypassed surfaces 1-3 on rows the
     scrapers/yahoo_ticker_rss/finnhub didn't gate at ingestion). Live
     evidence (2026-05-24) showed a recap-template title reaching
     ``urgency=2`` with ``ml_score=10.0`` because surface 2 only ran on
     the LLM-bound branch; the symmetric ML-path pre-floor closes the
     gap. See ``tests/test_score_pending::
     test_score_pending_prefloor_recap_and_quote_widget``.

All four surfaces resolve fingerprints through ONE source of truth
(``alert_agent._looks_like_quote_widget`` and the
``_QUOTE_WIDGET_TITLE_PATTERNS`` tuple). This module is the
*calibration view* analysts and the dashboard need to ANSWER:

  * "Is the urgency-scorer pre-filter still working — did a new tape
    variant sneak through and start firing again?"
  * "Are quote-widget rows in the strong-label training pool growing
    again?" (the exact poisoning the urgency_scorer fix exists to prevent;
    live evidence 2026-05-21 30d audit: 111 such rows BEFORE the fix)
  * "Which sources generate the bulk of quote-widget noise so we can
    prune the worst feeders?"

The audit groups rows in the recent window by their CURRENT state
(score_source, ai_score band, urgency) so a regression manifests as
a nonzero ``score_source='llm' AND ai_score>=8`` row count — exactly
what the live evidence showed before the pre-filter landed.
Counterpart to ``analytics/recap_template_audit.py`` (sibling noise
class) and ``ml/label_audit.py`` (training-pool integrity).

Pure read-side (``COUNT(*)`` only). The quote-widget fingerprint set
lives in ``watchers.alert_agent`` so it can never silently drift from
the live gate. Backtest isolation is enforced via the same
``_LIVE_ONLY_CLAUSE`` fragment as the rest of the audit family — a
synthetic backtest row matching a quote-widget title cannot inflate the
calibration figure.

Run standalone::

    python3 -m analytics.quote_widget_audit             # aggregate JSON
    python3 -m analytics.quote_widget_audit --hours 6   # custom window
    python3 -m analytics.quote_widget_audit --by-source # per-source view
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from typing import Optional

from watchers.alert_agent import _QUOTE_WIDGET_TITLE_PATTERNS


# Mirrors storage.article_store._LIVE_ONLY_CLAUSE — duplicated as a string
# constant rather than imported because this module deliberately does NOT
# pull the ArticleStore writer graph (we operate on a read-only connection).
# Pinned by ``test_live_only_clause_in_sync_with_storage`` in the test file.
LIVE_ONLY_CLAUSE = (
    "url NOT LIKE 'backtest://%' "
    "AND source NOT LIKE 'backtest_%' "
    "AND source NOT LIKE 'opus_annotation%'"
)


def _count_widget_matches(
    conn,
    window_since: str,
    extra_where: str = "",
) -> dict[str, int]:
    """Per-fingerprint count of rows whose title matches a quote-widget
    pattern AND falls in the recent window AND satisfies ``extra_where``.
    Pure Python regex (patterns are compiled once at import time in
    ``watchers.alert_agent``) — no SQL ``REGEXP`` extension required.
    """
    where = f"first_seen >= ? AND {LIVE_ONLY_CLAUSE}"
    if extra_where:
        where += f" AND ({extra_where})"
    rows = conn.execute(
        f"SELECT title FROM articles WHERE {where}",
        (window_since,),
    ).fetchall()
    counts = {name: 0 for name, _pat in _QUOTE_WIDGET_TITLE_PATTERNS}
    for (title,) in rows:
        if not title:
            continue
        for name, pat in _QUOTE_WIDGET_TITLE_PATTERNS:
            if pat.search(title):
                counts[name] += 1
                break  # one fingerprint per row — first wins (alert_agent precedent)
    return counts


def audit(store, hours: int = 24, now: Optional[datetime] = None) -> dict:
    """Calibration report for the quote-widget gate over the last ``hours``.

    Returns::

        {
            "window_h": int,
            "by_fingerprint": {<name>: int, ...},  # all widget rows in window
            "total_widget_rows": int,
            "leaked_to_strong_pool": int,   # MUST be 0 post-fix; the regression
                                            # signal the audit exists to surface
            "leaked_urgent": int,           # urgency>=1 (some gate failed)
            "floored_to_noise": int,        # ai_score <= 0.5 (pre-filter goal)
            "leaked_by_fingerprint": {<name>: int, ...},
            "leak_fraction": float,         # leaked_to_strong_pool / total
            "ok": bool,                     # True iff zero strong-pool leaks
        }

    ``leaked_to_strong_pool`` is the load-bearing metric: a nonzero value
    means a quote-widget row has score_source='llm' AND ai_score>=8 — i.e.
    it landed in the training pool tagged urgent. That is the exact
    poisoning the urgency_scorer pre-filter exists to prevent (live
    evidence 2026-05-21: 111 such rows before the fix).
    """
    conn = store.conn
    now = now or datetime.now(timezone.utc)
    since = (now - timedelta(hours=hours)).isoformat()

    by_fp = _count_widget_matches(conn, since)
    total = sum(by_fp.values())

    leaked_by_fp = _count_widget_matches(
        conn, since, "score_source='llm' AND ai_score >= 8.0"
    )
    leaked_to_strong_pool = sum(leaked_by_fp.values())

    urgent_by_fp = _count_widget_matches(conn, since, "urgency >= 1")
    leaked_urgent = sum(urgent_by_fp.values())

    floored_by_fp = _count_widget_matches(
        conn, since, "ai_score > 0 AND ai_score <= 0.5"
    )
    floored = sum(floored_by_fp.values())

    leak_fraction = (
        round(leaked_to_strong_pool / total, 4) if total else 0.0
    )

    return {
        "window_h": int(hours),
        "by_fingerprint": by_fp,
        "total_widget_rows": total,
        "leaked_to_strong_pool": leaked_to_strong_pool,
        "leaked_urgent": leaked_urgent,
        "floored_to_noise": floored,
        "leaked_by_fingerprint": leaked_by_fp,
        "leak_fraction": leak_fraction,
        "ok": leaked_to_strong_pool == 0,
    }


def audit_by_source(
    store,
    hours: int = 24,
    top_n: int = 15,
    now: Optional[datetime] = None,
) -> dict:
    """Per-source quote-widget hit breakdown over the last ``hours``.

    Sibling to ``recap_template_audit.audit_by_source`` — same shape, same
    sort discipline, different noise class. The aggregate ``audit()``
    answers "is the widget gate still working?". A news analyst pruning
    low-signal sources needs the next question: WHICH SOURCES generate the
    bulk of quote-widget noise? The 30d live audit found ~95% of the 111
    leaked rows came from ``scraped/finance.yahoo.com`` — knowing this
    lets the analyst target the actual offenders instead of guessing.

    Returns::

        {
            "window_h": int,
            "by_source": [
                {
                    "source": str,
                    "widget_count": int,          # total widget hits from this source
                    "by_fingerprint": {name: count, ...},  # non-zero only
                    "top_fingerprint": str,
                    "leaked_urgent": int,         # urgency>=1 widget rows
                    "leaked_strong_pool": int,    # llm-tagged ai_score>=8
                },
                ...  # most-widget-first; capped at top_n
            ],
            "total_widget_rows": int,
            "total_sources": int,
            "ok": bool,  # True iff zero strong-pool leaks across all sources
        }
    """
    conn = store.conn
    now = now or datetime.now(timezone.utc)
    since = (now - timedelta(hours=hours)).isoformat()

    rows = conn.execute(
        "SELECT source, title, urgency, score_source, ai_score "
        "FROM articles "
        f"WHERE first_seen >= ? AND {LIVE_ONLY_CLAUSE}",
        (since,),
    ).fetchall()

    per_source: dict[str, dict] = {}
    total_widget = 0
    for source, title, urgency, score_source, ai_score in rows:
        if not title:
            continue
        hit_name: Optional[str] = None
        for name, pat in _QUOTE_WIDGET_TITLE_PATTERNS:
            if pat.search(title):
                hit_name = name
                break
        if hit_name is None:
            continue
        total_widget += 1
        bucket = per_source.setdefault(
            source or "",
            {"widget_count": 0, "by_fingerprint": {},
             "leaked_urgent": 0, "leaked_strong_pool": 0},
        )
        bucket["widget_count"] += 1
        bucket["by_fingerprint"][hit_name] = (
            bucket["by_fingerprint"].get(hit_name, 0) + 1
        )
        try:
            urg_int = int(urgency or 0)
        except (TypeError, ValueError):
            urg_int = 0
        if urg_int >= 1:
            bucket["leaked_urgent"] += 1
        try:
            ai_val = float(ai_score or 0.0)
        except (TypeError, ValueError):
            ai_val = 0.0
        # The exact "strong-pool poisoning" predicate from audit() —
        # score_source='llm' AND ai_score>=8. Mirrored verbatim so a
        # source-level metric drift is impossible.
        if (score_source == "llm") and ai_val >= 8.0:
            bucket["leaked_strong_pool"] += 1

    materialised: list[dict] = []
    for source, b in per_source.items():
        fps = b["by_fingerprint"]
        if fps:
            top_fp = sorted(
                fps.items(), key=lambda kv: (-kv[1], kv[0])
            )[0][0]
        else:
            top_fp = ""
        materialised.append({
            "source": source,
            "widget_count": b["widget_count"],
            "by_fingerprint": dict(fps),
            "top_fingerprint": top_fp,
            "leaked_urgent": b["leaked_urgent"],
            "leaked_strong_pool": b["leaked_strong_pool"],
        })

    # Sort by widget_count desc; ties broken alphabetically by source for
    # reproducibility (mirrors recap_template_audit.audit_by_source).
    materialised.sort(
        key=lambda r: (-r["widget_count"], r["source"])
    )

    total_strong_leaks = sum(
        r["leaked_strong_pool"] for r in materialised
    )

    return {
        "window_h": int(hours),
        "by_source": materialised[: max(int(top_n), 0)],
        "total_widget_rows": total_widget,
        "total_sources": len(materialised),
        "ok": total_strong_leaks == 0,
    }


def format_report(report: dict) -> str:
    return json.dumps(report, indent=2, sort_keys=True)


class _RoStore:
    """Read-only ``.conn``-bearing shim — mirrors
    ``analytics/recap_template_audit._RoStore`` and
    ``ml/label_audit._RoStore``. Never opens the writable ArticleStore
    (which would block on busy_timeout under daemon writer-contention
    and run the score_source migration)."""

    def __init__(self, db_path) -> None:
        import sqlite3
        self.conn = sqlite3.connect(
            f"file:{db_path}?mode=ro", uri=True, timeout=15,
        )

    def close(self) -> None:
        self.conn.close()


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--hours", type=int, default=24,
        help="Window size in hours (default: 24)",
    )
    parser.add_argument(
        "--by-source", action="store_true",
        help="Emit the per-source widget breakdown (which feeds dominate the "
             "noise) instead of the aggregate gate-calibration view.",
    )
    parser.add_argument(
        "--top-n", type=int, default=15,
        help="When --by-source is set, cap the per-source list at this many "
             "rows (default: 15).",
    )
    args = parser.parse_args(argv)

    from storage.article_store import _get_db_path

    store = _RoStore(_get_db_path())
    try:
        if args.by_source:
            report = audit_by_source(
                store, hours=args.hours, top_n=args.top_n,
            )
        else:
            report = audit(store, hours=args.hours)
    finally:
        store.close()
    print(format_report(report))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
