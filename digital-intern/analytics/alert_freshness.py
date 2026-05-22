"""analytics/alert_freshness.py — read-only "is this alert still actionable?" monitor.

Why this exists (news-analyst lens): a 🚨 BREAKING alert fired on a 3-hour-old
article is almost worthless — the move is already in the market and the
analyst's only realistic action is to NOT chase it. The system already
measures collector slowness (``storage/ingestion_latency.py``) and per-source
calibration (``analytics/alert_source_breakdown.py``); what is **not**
measured is the bottom-line analyst metric: **of the rows that actually
fired urgent, how stale were they at the moment they were detected?**

A high p90 here means the alert pipeline is technically working but the
news is too old to be actionable — exactly the failure mode that reads
HEALTHY on every other monitor (alert_worker is firing, sources are
warm, calibration looks fine). The analyst will only know by manually
spot-checking ``age`` lines in their Discord channel.

This module is the dual of ``ingestion_latency``:

  * ``ingestion_latency`` — *all live rows*, per-source. "How fresh was
    the typical article on arrival?" Catches a slow collector.
  * ``alert_freshness`` — *urgency>=1 rows only*, aggregate + by
    ``score_source``. "How stale were the alerts the analyst was
    actually pushed?" Catches a quality problem the volume monitor
    misses.

Pure function ``compute_alert_freshness((published, first_seen,
score_source, urgency)…)`` is the unit-tested contract — clock parsing,
clamping, percentile maths, and the LLM-vs-ML calibration split all live
there. The DB shell is a thin wrapper.

Load-bearing invariants respected:

  * **Backtest isolation:** the SQL pull carries the canonical
    ``_LIVE_ONLY_CLAUSE`` verbatim (mirror of
    ``storage/article_store.py``; the test suite pins a drift check so a
    re-derivation that quietly diverges fails CI). Synthetic
    ``backtest://`` rows and ``backtest_*`` / ``opus_annotation*``
    sources can never colour the latency view.
  * **score_source separation:** ``ai_score`` / ``ml_score`` are never
    written here; the ``by_score_source`` breakdown is read-only and
    re-uses the SAME keys as ``ArticleStore.urgency_label_split`` so the
    two reads can never drift on what "vetted" means.
  * **Read-only:** the DB is opened ``mode=ro`` with a short busy
    timeout. Cannot add to writer contention.

CLI: ``python3 -m analytics.alert_freshness [--hours 24]`` prints a JSON report.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Iterable

# Canonical backtest-isolation clause. Duplicated verbatim from
# storage/article_store.py::_LIVE_ONLY_CLAUSE (same discipline as
# storage/ingestion_latency.py / storage/db_health.py / analytics/
# recap_template_audit.py) — the test suite pins a drift check so a
# re-derivation that quietly diverges fails CI loudly.
LIVE_ONLY_CLAUSE = (
    "url NOT LIKE 'backtest://%' "
    "AND source NOT LIKE 'backtest_%' "
    "AND source NOT LIKE 'opus_annotation%'"
)

_USB_PATH = Path(os.environ.get(
    "DIGITAL_INTERN_USB", "/media/zeph/projects/digital-intern/db"))
_LOCAL_PATH = Path(__file__).resolve().parent.parent / "data"

# An article whose computed staleness exceeds this is almost certainly an
# archive backfill (SEC EDGAR repost, Wikipedia revision sweep) rather than a
# fresh ingestion the analyst would interpret as breaking. Such rows are
# bucketed under ``skipped_implausible`` so one stale archive cannot dominate
# the p90/p99 percentiles — mirrors ``ingestion_latency``'s convention.
_MAX_PLAUSIBLE_SEC = 7 * 24 * 3600.0

# The score_source tag set urgency_label_split exposes — kept byte-identical
# so the two reads can never drift on what "vetted" means.
_VETTED_TAGS = ("llm", "briefing_boost")
_ALL_TAG_KEYS = ("llm", "ml", "briefing_boost", "null")


def resolve_db_path() -> Path:
    """Resolve live ``articles.db`` (USB-preferred), no side effects.

    Mirrors ``storage.ingestion_latency.resolve_db_path`` exactly so the
    monitor reads the same DB the daemon writes. Never calls ``mkdir`` —
    a read-only observer must not materialise an empty fallback directory.
    """
    usb_db = _USB_PATH / "articles.db"
    if _USB_PATH.exists() and (usb_db.exists() or _USB_PATH.is_mount()):
        return usb_db
    return _LOCAL_PATH / "articles.db"


def open_ro(path: Path) -> sqlite3.Connection:
    """Open ``path`` strictly read-only."""
    uri = f"file:{path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=5.0)
    conn.execute("PRAGMA busy_timeout=2000")
    return conn


def _parse_published(value: str | None) -> datetime | None:
    """Best-effort parse — RFC 2822 (RSS) and ISO-8601 variants.
    Returns ``None`` on anything unparseable; that row is surfaced under
    ``skipped_no_published`` so weak-metadata sources are visible."""
    if not value:
        return None
    s = value.strip()
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        try:
            dt = parsedate_to_datetime(s)
        except (TypeError, ValueError):
            return None
        if dt is None:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_first_seen(value: str | None) -> datetime | None:
    """Parse ``first_seen`` — written by ``article_store`` so always ISO-8601."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _percentile(sorted_xs: list[float], q: float) -> float:
    """Linear-interp percentile, no numpy. ``q`` in [0, 1]. ``sorted_xs`` non-empty."""
    if len(sorted_xs) == 1:
        return sorted_xs[0]
    idx = q * (len(sorted_xs) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_xs) - 1)
    frac = idx - lo
    return sorted_xs[lo] + frac * (sorted_xs[hi] - sorted_xs[lo])


def _tag_key(score_source: str | None) -> str:
    """Bucket key for the by_score_source breakdown. Mirrors
    ``urgency_label_split``'s null-collapse so the two reads stay byte-aligned."""
    if score_source in ("llm", "ml", "briefing_boost"):
        return score_source
    return "null"


def _summarise(samples: list[float]) -> dict:
    """Pure: percentile/threshold summary for a sample list.

    Returns a self-describing dict whose ``n==0`` shape carries every key
    with ``None`` numerics — callers can render an empty window without
    conditional branches (same shape discipline as
    ``ingestion_latency``'s per-source empty bucket).
    """
    n = len(samples)
    if n == 0:
        return {
            "n": 0,
            "p50_min": None, "p90_min": None, "p99_min": None,
            "max_min": None, "mean_min": None,
            "pct_under_5min": None, "pct_under_30min": None,
            "pct_over_1h": None, "pct_over_6h": None, "pct_over_24h": None,
        }
    sec = sorted(samples)
    pct = lambda thresh_sec: round(  # noqa: E731 — local helper, terser inline
        100.0 * sum(1 for s in sec if s > thresh_sec) / n, 2)
    pct_under = lambda thresh_sec: round(  # noqa: E731
        100.0 * sum(1 for s in sec if s <= thresh_sec) / n, 2)
    mean_sec = sum(sec) / n
    return {
        "n": n,
        "p50_min": round(_percentile(sec, 0.50) / 60.0, 2),
        "p90_min": round(_percentile(sec, 0.90) / 60.0, 2),
        "p99_min": round(_percentile(sec, 0.99) / 60.0, 2),
        "max_min": round(sec[-1] / 60.0, 2),
        "mean_min": round(mean_sec / 60.0, 2),
        "pct_under_5min": pct_under(300),
        "pct_under_30min": pct_under(1800),
        "pct_over_1h": pct(3600),
        "pct_over_6h": pct(6 * 3600),
        "pct_over_24h": pct(24 * 3600),
    }


def compute_alert_freshness(
    rows: Iterable[tuple],
) -> dict:
    """Pure: compute aggregate + by-score_source freshness for urgent rows.

    ``rows`` is an iterable of ``(published, first_seen, score_source,
    urgency)`` raw column values — same shape SQLite yields. Rows with
    ``urgency < 1`` are dropped (defense-in-depth: the SQL caller already
    filters this, but a future caller might pass a wider set).

    Returns::

        {
            "n_alerted_in_window": N,            # total urgency>=1 rows seen
            "n_with_published": M,               # rows that yielded a sample
            "skipped_no_published": N - M,       # operator visibility
            "skipped_implausible": K,            # >7d staleness, archives
            "aggregate": {<summary>},            # the bottom-line view
            "by_score_source": {                 # llm/ml/briefing_boost/null
                "llm": {<summary>},
                "ml":  {<summary>},
                "briefing_boost": {<summary>},
                "null": {<summary>},
            },
            "vetted_fraction": float,            # (llm+briefing_boost)/total,
                                                 #   parallels urgency_label_split
        }

    Negative latencies (``published`` ahead of ``first_seen`` — clock skew
    on the upstream feed) are clamped to 0 rather than dropped: they are
    real ingestions and silently dropping them would bias toward
    fresher-than-reality. Mirrors ``ingestion_latency``'s convention.
    """
    by_tag: dict[str, list[float]] = {k: [] for k in _ALL_TAG_KEYS}
    aggregate: list[float] = []
    by_tag_counts_total: dict[str, int] = {k: 0 for k in _ALL_TAG_KEYS}
    n_alerted = 0
    n_no_pub = 0
    n_implausible = 0

    for row in rows:
        try:
            published, first_seen, score_source, urgency = row
        except (TypeError, ValueError):
            # Non-conforming row shape — skip silently rather than crash an
            # audit run on malformed input. Same defensive convention as
            # ingestion_latency.compute_latency_stats.
            continue
        try:
            u = int(urgency or 0)
        except (TypeError, ValueError):
            u = 0
        if u < 1:
            continue
        n_alerted += 1
        tag = _tag_key(score_source)
        by_tag_counts_total[tag] += 1
        pub = _parse_published(published)
        seen = _parse_first_seen(first_seen)
        if pub is None or seen is None:
            n_no_pub += 1
            continue
        delta = (seen - pub).total_seconds()
        if delta < 0:
            delta = 0.0
        if delta > _MAX_PLAUSIBLE_SEC:
            n_implausible += 1
            continue
        aggregate.append(delta)
        by_tag[tag].append(delta)

    vetted = sum(by_tag_counts_total[k] for k in _VETTED_TAGS)
    vetted_fraction = (
        round(vetted / n_alerted, 4) if n_alerted else 0.0
    )

    return {
        "n_alerted_in_window": n_alerted,
        "n_with_published": len(aggregate),
        "skipped_no_published": n_no_pub,
        "skipped_implausible": n_implausible,
        "aggregate": _summarise(aggregate),
        "by_score_source": {
            k: _summarise(by_tag[k]) for k in _ALL_TAG_KEYS
        },
        "vetted_fraction": vetted_fraction,
    }


def load_alerted_rows(
    db_path: Path | None = None, hours: int = 24,
) -> list[tuple]:
    """Read urgency>=1 rows in the recent window. ``mode=ro`` so a writer
    storm cannot crash the audit; ``_LIVE_ONLY_CLAUSE`` so synthetic rows
    never colour the view. Minimal projection only — every other column
    is irrelevant."""
    path = db_path or resolve_db_path()
    cutoff = (
        datetime.now(timezone.utc) - timedelta(hours=hours)
    ).isoformat()
    con = open_ro(path)
    try:
        return con.execute(
            "SELECT published, first_seen, score_source, urgency "
            "FROM articles "
            f"WHERE urgency >= 1 AND first_seen >= ? AND {LIVE_ONLY_CLAUSE}",
            (cutoff,),
        ).fetchall()
    finally:
        con.close()


def run(db_path: Path | None = None, hours: int = 24) -> dict:
    """End-to-end: read → aggregate → report. Adds ``generated_at`` /
    ``window_hours`` / ``db_path`` to the pure summary so a JSON dump is
    self-describing."""
    rows = load_alerted_rows(db_path, hours=hours)
    report = compute_alert_freshness(rows)
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    report["window_hours"] = hours
    report["db_path"] = str(db_path or resolve_db_path())
    return report


_DEFAULT_OUT = Path("/home/zeph/logs/alert_freshness.json")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--hours", type=int, default=24)
    p.add_argument("--db", type=Path, default=None)
    p.add_argument("--out", type=Path, default=_DEFAULT_OUT,
                   help="Write JSON report to this file (default: %(default)s)")
    args = p.parse_args(argv)
    report = run(db_path=args.db, hours=args.hours)
    out_str = json.dumps(report, indent=2)
    print(out_str)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(out_str + "\n")
        print(f"[alert_freshness] report saved → {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
