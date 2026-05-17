"""Stale-source alerter + per-source freshness snapshot.

For every distinct live collector ``source`` in ``articles.db`` this computes
how many minutes have elapsed since that source's most recent article. Any
source with no new article in more than ``STALE_THRESHOLD_MIN`` minutes is
logged as a WARNING to the daemon log (same sinks as every other module via
``core.logger``). On every run a JSON snapshot is written to
``SNAPSHOT_PATH`` containing ``generated_at``, a per-source block
(``last_seen``, ``minutes_stale``, ``count_24h``, ``is_stale``) and the list
of stale sources.

Synthetic rows (``backtest://`` URLs, ``backtest_*`` / ``opus_annotation*``
sources) are excluded via the canonical ``_LIVE_ONLY_CLAUSE`` — those are
training-only injections, not live collectors, and would otherwise always
look "stale". This monitors live collectors only.

``first_seen`` is stored as ISO8601 with microseconds and a ``+00:00`` tz
suffix (e.g. ``2026-05-17T21:42:56.818669+00:00``); naive time-window SQL
against it is unreliable. We normalise it to a fixed-width
``YYYY-MM-DD HH:MM:SS`` (replace 'T'→' ', take the leading 19 chars) so both
the MAX and the 24h-count comparisons are correct lexicographic string
compares, then do the staleness arithmetic in Python with timezone-aware
UTC datetimes.

Standalone:   ``python3 -m analytics.stale_source_alerter``
Importable:   ``from analytics.stale_source_alerter import compute_freshness``
"""
import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from core.logger import get_logger
from storage.article_store import _LIVE_ONLY_CLAUSE, _get_db_path

log = get_logger("stale_source_alerter")

# A live collector with no new article in more than this many minutes is
# considered stale (its worker is silently dead, rate-limited, or the
# upstream feed stopped). 120 min comfortably exceeds the slowest collector
# cadence (alphavantage_worker @ 30 min, newsapi_worker @ 25 min) so a
# healthy source never trips it.
STALE_THRESHOLD_MIN = 120

# Written fresh on every run. This is a deliberate absolute path (not the
# repo ``logs/`` symlink) so the snapshot is co-located with the operator's
# other advisory artifacts under /home/zeph/logs/.
SNAPSHOT_PATH = Path("/home/zeph/logs/source_freshness.json")

_TS_FMT = "%Y-%m-%d %H:%M:%S"


def _parse_first_seen(normalised: str | None) -> datetime | None:
    """Parse a normalised ``YYYY-MM-DD HH:MM:SS`` string to aware UTC.

    Returns None if the value is missing or unparseable (caller treats that
    as stale — we cannot prove the source is healthy).
    """
    if not normalised:
        return None
    try:
        return datetime.strptime(normalised[:19], _TS_FMT).replace(
            tzinfo=timezone.utc
        )
    except (ValueError, TypeError):
        return None


def _collector_of(source: str) -> str:
    """The collector family a granular ``source`` belongs to.

    The ``source`` column is highly granular — GDELT/yfinance/GoogleNews/
    AlphaVantage tag every individual publisher (``gdelt_gkg/iheart.com``,
    ``yfinance/Bloomberg``), yielding tens of thousands of distinct values
    that are individually sparse. Alerting at that granularity would emit
    thousands of non-actionable WARNINGs. The actionable unit is the
    collector family — the prefix before the first '/' (or the whole string
    when there is no '/', e.g. ``rss``). Per-source detail is still retained
    in the snapshot; only the alerting/headline roll up to this level.
    """
    return source.split("/", 1)[0] if "/" in source else source


def compute_freshness(now: datetime | None = None) -> dict:
    """Return the full freshness report.

    Per-source detail (spec requirement) and a collector-family rollup
    (the actionable alerting unit) are both included. Shape::

        {
          "generated_at": "<iso utc>",
          "stale_threshold_min": 120,
          "stale_count": <int>,                   # stale sub-sources
          "tracked_count": <int>,                 # tracked sub-sources
          "stale_sources": ["src", ...],          # sorted
          "sources": {
             "src": {"last_seen": "...", "minutes_stale": 12.3,
                     "count_24h": 8, "is_stale": false},
             ...
          },
          "stale_collector_count": <int>,
          "collector_count": <int>,
          "stale_collectors": ["coll", ...],      # sorted, the alert set
          "collectors": {
             "coll": {"last_seen": "...", "minutes_stale": 4.0,
                      "count_24h": 41, "sub_source_count": 38,
                      "is_stale": false},
             ...
          }
        }
    """
    now = now or datetime.now(timezone.utc)
    cutoff_24h = (now - timedelta(hours=24)).strftime(_TS_FMT)

    db_path = _get_db_path()
    # Read-only URI connection: WAL allows concurrent readers without
    # blocking the live daemon writer, and mode=ro guarantees we never
    # take a write lock on the production DB.
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            f"""
            SELECT source,
                   MAX(substr(replace(first_seen, 'T', ' '), 1, 19))   AS last_seen,
                   SUM(CASE WHEN substr(replace(first_seen, 'T', ' '), 1, 19) >= ?
                            THEN 1 ELSE 0 END)                          AS count_24h,
                   COUNT(*)                                             AS total
            FROM articles
            WHERE {_LIVE_ONLY_CLAUSE}
              AND source IS NOT NULL
              AND source != ''
            GROUP BY source
            """,
            (cutoff_24h,),
        ).fetchall()
    finally:
        conn.close()

    sources: dict[str, dict] = {}
    stale_sources: list[str] = []
    for source, last_seen, count_24h, _total in rows:
        parsed = _parse_first_seen(last_seen)
        if parsed is None:
            minutes_stale = None
            is_stale = True
        else:
            minutes_stale = round((now - parsed).total_seconds() / 60.0, 1)
            is_stale = minutes_stale > STALE_THRESHOLD_MIN
        sources[source] = {
            "last_seen": last_seen,
            "minutes_stale": minutes_stale,
            "count_24h": int(count_24h or 0),
            "is_stale": is_stale,
        }
        if is_stale:
            stale_sources.append(source)

    stale_sources.sort()

    # ── Collector-family rollup (the actionable alert unit) ──────────────
    # freshest sub-source wins (last_seen is fixed-width normalised, so a
    # lexicographic max == chronological max); count_24h sums.
    coll_acc: dict[str, dict] = {}
    for source, info in sources.items():
        key = _collector_of(source)
        acc = coll_acc.setdefault(
            key, {"last_seen": None, "count_24h": 0, "sub_source_count": 0}
        )
        acc["count_24h"] += info["count_24h"]
        acc["sub_source_count"] += 1
        ls = info["last_seen"]
        if ls and (acc["last_seen"] is None or ls > acc["last_seen"]):
            acc["last_seen"] = ls

    collectors: dict[str, dict] = {}
    stale_collectors: list[str] = []
    for key, acc in coll_acc.items():
        parsed = _parse_first_seen(acc["last_seen"])
        if parsed is None:
            minutes_stale = None
            is_stale = True
        else:
            minutes_stale = round((now - parsed).total_seconds() / 60.0, 1)
            is_stale = minutes_stale > STALE_THRESHOLD_MIN
        collectors[key] = {
            "last_seen": acc["last_seen"],
            "minutes_stale": minutes_stale,
            "count_24h": acc["count_24h"],
            "sub_source_count": acc["sub_source_count"],
            "is_stale": is_stale,
        }
        if is_stale:
            stale_collectors.append(key)
    stale_collectors.sort()

    return {
        "generated_at": now.isoformat(),
        "stale_threshold_min": STALE_THRESHOLD_MIN,
        "stale_count": len(stale_sources),
        "tracked_count": len(sources),
        "stale_sources": stale_sources,
        "sources": sources,
        "stale_collector_count": len(stale_collectors),
        "collector_count": len(collectors),
        "stale_collectors": stale_collectors,
        "collectors": collectors,
    }


def write_snapshot(report: dict, path: Path = SNAPSHOT_PATH) -> Path:
    """Atomically write the report as pretty JSON to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
    return path


def alert_stale(report: dict) -> None:
    """Emit one WARNING per stale collector family (the alerting side-effect).

    Alerting is at collector granularity, not per sub-source: the ``source``
    column has tens of thousands of individually-sparse publisher tags;
    warning on each would flood the daemon log with non-actionable noise.
    """
    for coll in report["stale_collectors"]:
        info = report["collectors"][coll]
        ms = info["minutes_stale"]
        age = "unparseable last_seen" if ms is None else f"{ms:.0f} min ago"
        log.warning(
            "STALE COLLECTOR %s: newest article %s (>%d min), "
            "count_24h=%d, sub_sources=%d",
            coll, age, STALE_THRESHOLD_MIN,
            info["count_24h"], info["sub_source_count"],
        )


def main() -> int:
    """Compute → alert → snapshot. The FIRST stdout line is a single,
    information-dense summary (it is consumed verbatim by downstream
    reporting)."""
    report = compute_freshness()
    snap_path = write_snapshot(report)

    # Line 1 — the headline summary (kept short & dense on purpose; it is
    # consumed verbatim downstream). Reported at collector granularity —
    # the actionable unit.
    print(
        f"stale-source alerter: {report['stale_collector_count']} stale of "
        f"{report['collector_count']} collectors (>{STALE_THRESHOLD_MIN}m); "
        f"snapshot={snap_path}"
    )

    # Line 2 — freshest collector, for at-a-glance health.
    fresh = sorted(
        (
            (c, i["minutes_stale"], i["count_24h"])
            for c, i in report["collectors"].items()
            if i["minutes_stale"] is not None
        ),
        key=lambda t: t[1],
    )
    if fresh:
        c, m, n = fresh[0]
        print(f"freshest collector {c}: newest article {m:.1f} min ago "
              f"({n} in 24h)")
    # Lines 3..N — every stale collector (the alert set; ~tens, not 23k).
    for coll in report["stale_collectors"]:
        info = report["collectors"][coll]
        ms = info["minutes_stale"]
        age = "unparseable" if ms is None else f"{ms:.0f} min ago"
        print(f"STALE collector {coll}: newest article {age} "
              f"({info['count_24h']} in 24h, "
              f"{info['sub_source_count']} sub-sources)")

    log.info(
        "freshness snapshot written: %d stale of %d collectors "
        "(%d stale of %d sub-sources) -> %s",
        report["stale_collector_count"], report["collector_count"],
        report["stale_count"], report["tracked_count"], snap_path,
    )
    alert_stale(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
