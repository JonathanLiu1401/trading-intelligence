"""Source-credibility coverage audit (read-only).

Surfaces source-tag spellings the daemon is collecting LIVE that still
resolve to ``DEFAULT_SOURCE_CRED`` — i.e. publishers the credibility
resolver doesn't know about, so:

  * feature[0] (source_credibility) of every row from those tags is flat at
    0.55 — wasted signal capacity for the ArticleNet relevance head;
  * the ``ALERT_MIN_LONE_SOURCE_CRED=0.45`` lone-alert authority gate sees
    them as "unknown" and gates nothing (default 0.55 > 0.45, by design —
    "unknown is never gated"). A defaulting tag from a *known-low-quality*
    publisher (a SEO mill, a forum, an entertainment site) is therefore
    NOT down-rated — exactly the failure class
    ``ml.features._LOW_AUTHORITY_DOMAINS`` exists to catch but cannot,
    because the *tag spelling* never reached its dotted-host or
    word-boundary form.

The audit is the standing leading indicator for this class of bug. The
fix that motivated it was concrete (2026-05-20): ~5,376 ``GN: <topic>``
rows / 24h (Google News topic feeds from ``config/sources.json``), 95
``YF/<bucket>`` rows (market_movers screener-tape), and several hundred
``YahooFinance/<symbol>`` rows (yahoo_ticker_rss) — three high-volume
aggregator-prefix conventions that ``_source_credibility`` defaulted
because the embedded publisher's spelling differed from any SOURCE_CRED
key (``GN:`` vs "googlenews"; ``YahooFinance/`` glued so
``\\byahoo\\b`` cannot match). The ``_PREFIX_ALIASES`` rescue closed
that exact gap.

This module reports, for the recent window:

  * the top-N defaulting source tags by row count, sorted desc;
  * a totals row separating differentiated rows from defaulting rows;
  * a ``defaulting_share`` ratio — feature[0] saturation expressed as the
    fraction of live rows whose credibility feature is the floor default.

Backtest isolation is enforced by the same ``_LIVE_ONLY_CLAUSE`` fragment
the rest of the audit family uses — a synthetic backtest/opus-annotation
row cannot inflate either side, even though those rows technically share
the table per CLAUDE.md §5.

Pure read-side: NO DB write, NO ai_score/ml_score/score_source/urgency
mutation, never mutates the store — all four load-bearing invariants
intact by construction.

Run standalone::

    python3 -m analytics.source_credibility_audit            # JSON, 24h window
    python3 -m analytics.source_credibility_audit --hours 6  # custom window
    python3 -m analytics.source_credibility_audit --top 25   # more rows
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from ml.features import DEFAULT_SOURCE_CRED, _source_credibility

OUT_PATH = Path("/home/zeph/logs/source_credibility_audit.json")


# Mirrors storage.article_store._LIVE_ONLY_CLAUSE — duplicated as a string
# constant rather than imported because this module operates on a read-only
# connection and must not pull the ArticleStore writer graph. Pinned by
# ``test_live_only_clause_in_sync_with_storage`` in the test file (same
# discipline as ``analytics/recap_template_audit.LIVE_ONLY_CLAUSE``).
LIVE_ONLY_CLAUSE = (
    "url NOT LIKE 'backtest://%' "
    "AND source NOT LIKE 'backtest_%' "
    "AND source NOT LIKE 'opus_annotation%'"
)


def _per_source_counts(conn, since: str) -> list[tuple[str, int]]:
    """``(source, count)`` for every live source observed since ``since``.

    Single ``GROUP BY`` over ``idx_first_seen`` so the scan is bounded by
    the recent window. Returns Python tuples to keep the rest of the
    audit DB-agnostic (the helper functions take a plain list).
    """
    rows = conn.execute(
        "SELECT source, COUNT(*) AS n FROM articles "
        f"WHERE first_seen >= ? AND {LIVE_ONLY_CLAUSE} "
        "GROUP BY source",
        (since,),
    ).fetchall()
    return [(src or "", int(n or 0)) for src, n in rows]


def _partition_defaulting(
    per_source: list[tuple[str, int]],
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Split ``(source, count)`` rows into ``(defaulting, differentiated)``.

    Defaulting = ``_source_credibility(source) == DEFAULT_SOURCE_CRED`` (the
    publisher is unknown to every resolution path — domain candidates,
    prefix aliases, verbatim word-boundary scan). Differentiated = any
    other resolved grade.

    Pure: no DB / IO. The split is the input the report builder turns
    into the analyst-facing leaderboard.
    """
    defaulting: list[tuple[str, int]] = []
    differentiated: list[tuple[str, int]] = []
    for source, n in per_source:
        if not source:
            # An empty source tag is its own kind of bug (lost upstream)
            # — counted in defaulting because feature[0] for those rows
            # falls to DEFAULT exactly like an unknown publisher.
            defaulting.append((source, n))
            continue
        cred = _source_credibility(source)
        if math.isclose(cred, DEFAULT_SOURCE_CRED, abs_tol=1e-9):
            defaulting.append((source, n))
        else:
            differentiated.append((source, n))
    return defaulting, differentiated


def audit(
    store,
    hours: int = 24,
    top: int = 15,
    now: Optional[datetime] = None,
) -> dict:
    """Coverage report over the last ``hours``.

    Returns::

        {
            "window_h": int,
            "top": int,
            "total_rows": int,                  # live rows in window
            "differentiated_rows": int,         # cred != DEFAULT
            "defaulting_rows": int,             # cred == DEFAULT
            "defaulting_sources": int,          # distinct tags hitting default
            "defaulting_share": float,          # defaulting / total, 0..1
            "top_defaulting": [                 # leaderboard, count desc
                {"source": "...", "count": N, "cred": DEFAULT_SOURCE_CRED},
                ...
            ],
            "ok": bool,                         # share < OK_THRESHOLD
        }

    ``defaulting_share`` is the load-bearing metric: a credibility feature
    that is mostly at the floor for the live corpus is a flattened feature
    — the maintenance team should review the top defaulting tags and either
    extend ``SOURCE_CRED``/``_DOMAIN_CRED``/``_PREFIX_ALIASES`` or
    ``_LOW_AUTHORITY_DOMAINS``. ``ok`` is True when the share is below the
    documented threshold (see ``OK_THRESHOLD``).
    """
    now = now or datetime.now(timezone.utc)
    since = (now - timedelta(hours=hours)).isoformat()

    per_source = _per_source_counts(store.conn, since)
    defaulting, differentiated = _partition_defaulting(per_source)

    total_rows = sum(n for _, n in per_source)
    diff_rows = sum(n for _, n in differentiated)
    def_rows = sum(n for _, n in defaulting)

    # Leaderboard: count desc, then source asc for stability when counts tie.
    defaulting_sorted = sorted(
        defaulting, key=lambda kv: (-kv[1], kv[0])
    )[: max(0, int(top))]
    top_defaulting = [
        {
            "source": src,
            "count": n,
            "cred": DEFAULT_SOURCE_CRED,
        }
        for src, n in defaulting_sorted
    ]

    share = round(def_rows / total_rows, 4) if total_rows else 0.0

    return {
        "window_h": int(hours),
        "top": int(top),
        "total_rows": total_rows,
        "differentiated_rows": diff_rows,
        "defaulting_rows": def_rows,
        "defaulting_sources": len(defaulting),
        "defaulting_share": share,
        "top_defaulting": top_defaulting,
        "ok": share < OK_THRESHOLD,
    }


# Share of live rows whose credibility feature is at the floor default that
# the maintenance team considers acceptable. Tuned conservatively against the
# 2026-05-20 24h snapshot: ~5,376 GN: rows / ~115k total live rows == ~5%
# defaulting share post-fix (the GN:/YF/YahooFinance prefix aliases moved
# them off DEFAULT). 25% is a deliberate generous bar — anything beyond it
# means a *new* large-volume aggregator prefix has started ingesting and
# nobody has triaged it. The maintainer can lower it as the resolver
# matures.
OK_THRESHOLD = 0.25


def format_report(report: dict) -> str:
    return json.dumps(report, indent=2, sort_keys=True)


class _RoStore:
    """Read-only ``.conn``-bearing shim — same shape as
    ``analytics.recap_template_audit._RoStore`` / ``ml.label_audit._RoStore``.
    Never opens the writable ArticleStore (which would block on busy_timeout
    under daemon writer-contention and run the score_source migration on
    the live DB)."""

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
        "--top", type=int, default=15,
        help="How many top defaulting tags to report (default: 15)",
    )
    parser.add_argument(
        "--no-write", action="store_true",
        help="skip writing OUT_PATH (stdout only)",
    )
    args = parser.parse_args(argv)

    from storage.article_store import _get_db_path

    store = _RoStore(_get_db_path())
    try:
        report = audit(store, hours=args.hours, top=args.top)
    finally:
        store.close()

    if not args.no_write:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = OUT_PATH.with_suffix(".json.tmp")
        tmp.write_text(format_report(report))
        tmp.replace(OUT_PATH)

    print(format_report(report))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
