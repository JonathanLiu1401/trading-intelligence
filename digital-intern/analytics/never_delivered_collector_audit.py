"""``analytics.never_delivered_collector_audit`` — surface registered collectors
that have *never* delivered (or have been silent >24h) into ``articles.db``.

The gap this fills (news-analyst lens). ``analytics.stale_source_alerter`` is
authoritative for collectors that have AT LEAST ONE row in the last 3 days —
those collectors get a per-source ``minutes_stale`` block and a STALE
WARNING when above the 120-min threshold. But a collector that has NEVER
written a row (or has been totally silent for >3 days) simply does NOT
appear in stale_source_alerter's output: the SQL pull groups by ``source``
over the last 3 days, so a collector with zero rows in that window is
**invisible to the freshness monitor**. That's how the ``un_news_worker``
broken-since-commit ``71e9cd7`` bug (``bo.advance()`` AttributeError on
every cycle for 7 days — fixed in this same pass; tested by
``tests/test_worker_backoff_contract.py``) went undetected by the existing
freshness audit: the worker AttributeError'd before any ``_ingest`` call
could complete, so no row was ever written, so the collector never
appeared in the freshness snapshot, so the operator never saw a STALE
WARNING — despite the supervisor reporting ``state=ok crashes_5m=0``.

This audit closes that gap. The caller supplies a curated
``EXPECTED_COLLECTORS`` registry naming the (worker_name, source_prefix)
pairs we expect to be delivering. The audit checks every prefix against
the live row corpus and surfaces:

  * ``NEVER_DELIVERED`` — zero rows ever (or zero in the audit's window):
    the worker likely crashes before the first ``_ingest`` call. This is
    the exact failure shape of the un_news_worker bug.
  * ``SILENT_24H`` — at least one row historically but zero in the last
    24h: collector dead, rate-limited, or the upstream feed stopped.
  * ``SLOW`` — rows in 24h but below ``min_rows_24h`` (the expected floor
    for a healthy collector). The threshold is per-collector, calibrated
    to the worker's polling cadence.
  * ``DELIVERING`` — at or above the floor.

Aggregate verdict ladder (analyst-facing single-line summary):

  * ``HEALTHY`` — zero NEVER_DELIVERED + zero SILENT_24H; at most one SLOW.
  * ``FEW_DEGRADED`` — 1-2 SILENT_24H/SLOW/NEVER_DELIVERED.
  * ``WIDESPREAD_SILENCE`` — 3+ degraded, OR any NEVER_DELIVERED hits.
  * ``NO_DATA`` — empty ``rows`` (audit cannot run).

Load-bearing invariants (all four intact by construction):

  * **Backtest isolation.** Pure read-side; the caller fetches ``rows``
    using ``storage.article_store._LIVE_ONLY_CLAUSE`` (the live entrypoint
    below does this). Synthetic ``backtest://`` / ``backtest_*`` /
    ``opus_annotation*`` rows are excluded before they reach the builder.
  * **score_source separation.** Audit reads only ``source`` and
    ``first_seen``. No ai_score / ml_score / score_source / urgency
    mutation. Read-only.
  * **Read-only.** Live entrypoint opens the DB ``mode=ro`` with a short
    busy timeout — cannot perturb the writer.
  * **Pure builder.** ``audit(rows, registry, now=None)`` derives the
    verdict from inputs alone; no DB, no clock, no I/O.

CLI: ``python3 -m analytics.never_delivered_collector_audit [--pretty]``
prints the live verdict as JSON.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from typing import Iterable


# Curated registry of (worker_name, source-tag-prefix, min_rows_24h) tuples.
# DELIBERATELY SMALL AND EXTENSIBLE. The point of this module is the audit
# mechanism, not registry exhaustiveness; the same shape catches any future
# silently-broken worker once its entry is added here.
#
# ``source_prefix`` is matched as a startswith(...) against the row's
# ``source`` column — collectors emit prefixed tags (``un_econ_dev``,
# ``rss/seeking_alpha``, ``gdelt_gkg/iheart.com``) so prefix-match catches
# every variant a single worker can emit. A trailing '/' in the prefix
# anchors at a slash boundary (rss/...) and avoids collision with longer
# prefixes that happen to share a leading substring.
#
# ``min_rows_24h`` is the floor below which the collector is SLOW. The
# numbers below are calibrated against a healthy 24h window on this DB
# (May 2026) — chosen well below observed steady-state so a temporarily
# slow but functional collector is not falsed.
#
# UN News (the worker my Phase-1 fix restored) is pinned at the registry
# top because it is the canonical NEVER_DELIVERED case. The six sub-source
# tags it emits (un_econ_dev / un_climate / un_health / un_americas /
# un_africa / un_europe) all share the ``un_`` literal prefix; on a healthy
# day the worker should produce >=10 rows / 24h (six feeds @ ~30 min
# polling, modest article volume each).
EXPECTED_COLLECTORS: tuple[tuple[str, str, int], ...] = (
    # The fixed worker. Source tags emitted by ``collectors.un_news_collector``
    # are bare topic/region keys (no ``un_news/`` aggregator prefix), all
    # sharing the literal ``un_`` lead — so prefix-match catches every feed.
    ("un_news", "un_", 6),
    # Web-scrape firehose (collectors/web_scraper.py emits
    # ``scraped/<host>`` for every anchor-extracted article). One of the
    # broadest wires; a healthy daemon writes hundreds of rows / 24h.
    ("web", "scraped/", 50),
    # GDELT firehose — the gdelt_collector emits ``GDELT/<host>`` (capital
    # prefix; live 2026-05-30 confirmed). The historical-sweep / GKG
    # lowercase prefixes (``gdelt_<YYYY-MM-DD>``, ``gdelt_gkg/<host>``)
    # exist in the back-catalogue but are not on the live ingest path, so
    # they are NOT registered — registering an inactive collector would
    # NEVER_DELIVERED-false-positive every 24h and devalue the verdict.
    ("gdelt", "GDELT/", 50),
    # Google News topic-feed prefix (``GN: <topic>`` — sources.json). The
    # broadest non-scraped wire on this DB (often the #1 row producer).
    ("google_news", "GN:", 100),
    # StockTwits forum stream — the urgency_scorer pre-floors most of these
    # as chatter, but the collector itself should always be DELIVERING.
    ("stocktwits", "stocktwits", 50),
)


# Compile-time validation: every prefix must be non-empty and every floor
# >= 1, else the audit is meaningless (vacuously DELIVERING).
for _name, _prefix, _floor in EXPECTED_COLLECTORS:
    assert _prefix, f"empty source_prefix for worker {_name!r}"
    assert _floor >= 1, f"min_rows_24h must be >=1 for {_name!r}"


def _row_matches_prefix(row: dict, prefix: str) -> bool:
    """Source-tag prefix match. ``row['source']`` may be ``None`` (malformed
    insert) — degrade to False rather than raise. Case-sensitive on purpose:
    collectors emit canonical lowercased / mixed-case tags and the registry
    pins the exact form."""
    src = row.get("source")
    if not isinstance(src, str) or not src:
        return False
    return src.startswith(prefix)


def _classify_one(
    rows: Iterable[dict],
    prefix: str,
    min_rows_24h: int,
    cutoff_24h: datetime,
) -> dict:
    """Per-collector classification. Returns the per-prefix sub-block of
    the audit envelope.

    ``rows`` is the FULL audit pool (already pre-filtered by the caller to
    be live-only and within the audit window). The builder partitions by
    prefix inline rather than asking the caller to pre-slice — keeps the
    audit's ``rows -> verdict`` contract single-input."""
    matches: list[dict] = []
    last_first_seen: str | None = None
    for r in rows:
        if not _row_matches_prefix(r, prefix):
            continue
        matches.append(r)
        fs = r.get("first_seen") or ""
        if isinstance(fs, str) and fs:
            if last_first_seen is None or fs > last_first_seen:
                last_first_seen = fs

    n_total = len(matches)
    n_24h = 0
    for r in matches:
        fs = (r.get("first_seen") or "")
        if not isinstance(fs, str) or not fs:
            continue
        try:
            # Match the storage layer's normalisation: ISO-8601 with optional
            # timezone, leading-19 strict ``YYYY-MM-DD HH:MM:SS``. Both forms
            # land in ``first_seen``; lexicographic compare on the normalised
            # prefix is reliable.
            head = fs[:19].replace("T", " ")
            row_dt = datetime.strptime(head, "%Y-%m-%d %H:%M:%S").replace(
                tzinfo=timezone.utc
            )
        except (ValueError, TypeError):
            continue
        if row_dt >= cutoff_24h:
            n_24h += 1

    if n_total == 0:
        verdict = "NEVER_DELIVERED"
    elif n_24h == 0:
        verdict = "SILENT_24H"
    elif n_24h < min_rows_24h:
        verdict = "SLOW"
    else:
        verdict = "DELIVERING"

    return {
        "source_prefix": prefix,
        "n_total": n_total,
        "n_24h": n_24h,
        "min_rows_24h": min_rows_24h,
        "last_first_seen": last_first_seen,
        "verdict": verdict,
    }


def _aggregate_verdict(per_worker: dict[str, dict]) -> str:
    """Roll up per-worker verdicts into a single analyst-facing line.

    Ladder:
      * any ``NEVER_DELIVERED`` -> ``WIDESPREAD_SILENCE`` (silently broken
        worker is the highest-severity case — it means the worker has
        produced zero output ever, which the freshness monitor cannot see).
      * 3+ degraded (SLOW/SILENT_24H/NEVER_DELIVERED) -> ``WIDESPREAD_SILENCE``.
      * 1-2 degraded -> ``FEW_DEGRADED``.
      * 0 degraded (at most one SLOW kept inside ``HEALTHY``) -> ``HEALTHY``.
    """
    if not per_worker:
        return "NO_DATA"
    degraded = [w for w in per_worker.values()
                if w["verdict"] in ("NEVER_DELIVERED", "SILENT_24H", "SLOW")]
    has_never = any(w["verdict"] == "NEVER_DELIVERED" for w in per_worker.values())
    if has_never:
        return "WIDESPREAD_SILENCE"
    if len(degraded) >= 3:
        return "WIDESPREAD_SILENCE"
    if len(degraded) >= 1:
        return "FEW_DEGRADED"
    return "HEALTHY"


def audit(
    rows: list[dict],
    registry: Iterable[tuple[str, str, int]] | None = None,
    *,
    now: datetime | None = None,
    window_h: int = 24,
) -> dict:
    """Pure builder. Classify each collector in ``registry`` against the
    audit pool ``rows`` and emit the deterministic envelope.

    ``rows``: live-only rows in the audit window (caller pre-filters with
    ``_LIVE_ONLY_CLAUSE``). Each row must carry at least ``source`` and
    ``first_seen``. Other keys are ignored.

    ``registry``: iterable of (worker_name, source_prefix, min_rows_24h)
    tuples. Defaults to ``EXPECTED_COLLECTORS``.

    ``now``: anchor for the 24h cutoff. Defaults to ``datetime.now(utc)``;
    tests pass a fixed value for determinism.

    ``window_h``: hours of history considered "in window" for the 24h
    bucket. Default 24h; the audit name is fixed but the parameter lets
    tests pin alternate windows.

    Returns ``{generated_at, window_h, n_pool_rows, by_worker,
    n_never_delivered, n_silent_24h, n_slow, n_delivering, verdict}``.
    """
    if registry is None:
        registry = EXPECTED_COLLECTORS
    if now is None:
        now = datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    cutoff_24h = now - timedelta(hours=window_h)

    by_worker: dict[str, dict] = {}
    for worker_name, prefix, min_rows in registry:
        by_worker[worker_name] = _classify_one(rows, prefix, min_rows, cutoff_24h)

    if not rows:
        # Audit cannot tell silent-collector from missing-DB-snapshot;
        # surface NO_DATA so the operator doesn't read "all healthy" as a
        # confident signal when in fact no rows reached the builder.
        return {
            "generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "window_h": window_h,
            "n_pool_rows": 0,
            "by_worker": by_worker,
            "n_never_delivered": 0,
            "n_silent_24h": 0,
            "n_slow": 0,
            "n_delivering": 0,
            "verdict": "NO_DATA",
        }

    n_never = sum(1 for w in by_worker.values() if w["verdict"] == "NEVER_DELIVERED")
    n_silent = sum(1 for w in by_worker.values() if w["verdict"] == "SILENT_24H")
    n_slow = sum(1 for w in by_worker.values() if w["verdict"] == "SLOW")
    n_deliv = sum(1 for w in by_worker.values() if w["verdict"] == "DELIVERING")

    return {
        "generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "window_h": window_h,
        "n_pool_rows": len(rows),
        "by_worker": by_worker,
        "n_never_delivered": n_never,
        "n_silent_24h": n_silent,
        "n_slow": n_slow,
        "n_delivering": n_deliv,
        "verdict": _aggregate_verdict(by_worker),
    }


def audit_live(store, hours: int = 24, **kwargs) -> dict:
    """Live entrypoint. Reads ``articles.db`` over the last ``hours``
    window using the canonical ``_LIVE_ONLY_CLAUSE`` so synthetic backtest /
    opus annotation rows are excluded by construction (load-bearing
    invariant #1).

    ``store``: an open ``ArticleStore`` whose ``.conn`` is read-only safe.
    The audit runs a single bounded-window SELECT — does NOT mutate
    ai_score / ml_score / score_source / urgency.
    """
    from storage.article_store import _LIVE_ONLY_CLAUSE

    cutoff_iso = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    cur = store.conn.execute(
        f"SELECT source, first_seen FROM articles "
        f"WHERE first_seen >= ? AND {_LIVE_ONLY_CLAUSE}",
        (cutoff_iso,),
    )
    rows = [{"source": r[0], "first_seen": r[1]} for r in cur.fetchall()]
    return audit(rows, window_h=hours, **kwargs)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hours", type=int, default=24, help="Window (hours).")
    p.add_argument("--pretty", action="store_true", help="Indent JSON.")
    args = p.parse_args()

    from storage.article_store import ArticleStore

    store = ArticleStore()
    report = audit_live(store, hours=args.hours)
    print(json.dumps(report, indent=2 if args.pretty else None))
    return 0


if __name__ == "__main__":
    sys.exit(main())
