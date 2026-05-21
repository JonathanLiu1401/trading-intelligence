"""analytics/alert_delivery_audit.py — were these "alerts" actually delivered?

Why this exists (news-analyst lens): the dashboard "urgent" tile counts every
``urgency>=1`` row. The alert worker fires Discord pushes for the rows that
clear every defense-in-depth gate (synthetic / quote-widget / recap-template
/ low-authority / stale-published / cross-cycle dup) — the rows that *don't*
clear those gates are marked ``urgency=2`` too (``mark_alerted_batch`` is
called by the gate so the row exits the queue), so the dashboard count
silently conflates "the analyst was pushed" with "the gate quietly absorbed
it". From the analyst's perspective those are very different events: a real
🚨 BREAKING is something they reacted to; a gate-mark is something they
never saw.

The ledger that distinguishes the two ALREADY EXISTS — ``watchers/alert_recency``
records the canonical signature of every alert that actually fired to Discord
(see ``record_alerted`` after ``discord_send`` succeeds in
``watchers/alert_agent.py``). This module is a read-only join of
``articles.db`` (urgency=2 rows) against ``alert_recency.db`` (delivered
signatures) over the same window, partitioning into:

  * ``delivered``  — urgency=2 row whose canonical signature is in
                     ``alerted_sig`` within the window. The analyst was
                     pushed (possibly as part of a syndicated fold).
  * ``suppressed`` — urgency=2 row whose canonical signature is NOT in
                     ``alerted_sig``. A defense-in-depth gate marked the row
                     alerted to exit the queue; the analyst never saw it.

Suppressed rows are then attributed to which fingerprint catches them
(``quote_widget`` / ``recap_template`` / ``low_authority`` / ``stale_published``
/ ``synthetic`` / ``unknown_gate``). The fingerprint set is composed VERBATIM
from ``watchers.alert_agent`` SSOT helpers (same lockstep discipline as
``analytics/recap_template_audit.py``) so a future change to the live gates
cannot silently diverge from the audit.

Two analyst-actionable numbers come out of this:

  1. ``delivery_rate`` = ``delivered / total``. The "noise pressure" index —
     what fraction of urgency-head fires actually reach Discord. A persistent
     drop means the model is producing more false positives that the gates
     are absorbing; an investigation hint without staring at logs.
  2. ``suppressed_by`` — per-gate counts. Which gate is doing the most work?
     A spike in ``recap_template`` means a new SEO variant is sneaking through
     the urgency head; a spike in ``low_authority`` means a social-tier feed
     is over-firing.

Pure function ``compute_delivery_audit(urgent_rows, alerted_sigs,
fingerprint_lookups)`` is the unit-tested contract; the DB shell is a thin
wrapper.

Load-bearing invariants respected:

  * **Backtest isolation:** the SQL pull carries the canonical
    ``_LIVE_ONLY_CLAUSE`` verbatim (mirror of
    ``storage/article_store.py``; the test suite pins a drift check so a
    re-derivation that quietly diverges fails CI). Synthetic ``backtest://``
    rows and ``backtest_*`` / ``opus_annotation*`` sources can never enter
    the audit set. A defense-in-depth synthetic check is also exposed as a
    fingerprint (``synthetic``) so a future caller bypassing the SQL filter
    is still attributed correctly rather than silently inflating
    ``delivered`` or ``unknown_gate``.
  * **score_source separation:** ``ai_score`` / ``ml_score`` / ``score_source``
    are READ only — never written. The audit derives no labels and writes
    no DB columns.
  * **Read-only:** both DBs are opened ``mode=ro`` with a short busy
    timeout. Cannot add to writer contention or perturb the alert path.

CLI: ``python3 -m analytics.alert_delivery_audit [--hours 6]`` prints a JSON
report. The default window matches ``alert_recency.ALERT_RECENCY_TTL_HOURS``
(6h) — values larger than the recency TTL would compare urgency=2 rows
against an already-pruned signature set and inflate ``suppressed`` falsely.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable

OUT_PATH = Path("/home/zeph/logs/alert_delivery_audit.json")

from watchers.alert_agent import (
    ALERT_MIN_LONE_SOURCE_CRED,
    _article_age_ok,
    _is_synthetic,
    _looks_like_quote_widget,
    _looks_like_recap_template,
)
from watchers.alert_dedup import _signature
from watchers.alert_recency import ALERT_RECENCY_TTL_HOURS
from ml.features import _source_credibility

# Canonical backtest-isolation clause. Duplicated verbatim from
# storage/article_store.py::_LIVE_ONLY_CLAUSE (same discipline as the rest of
# the analytics/ + storage/ family) — the test suite pins a drift check.
LIVE_ONLY_CLAUSE = (
    "url NOT LIKE 'backtest://%' "
    "AND source NOT LIKE 'backtest_%' "
    "AND source NOT LIKE 'opus_annotation%'"
)

_USB_PATH = Path(os.environ.get(
    "DIGITAL_INTERN_USB", "/media/zeph/projects/digital-intern/db"))
_LOCAL_PATH = Path(__file__).resolve().parent.parent / "data"

# Default window matches the recency-store TTL exactly. Asking for a wider
# window would compare urgency=2 rows against signatures that have been
# pruned out of ``alerted_sig`` and would over-attribute them to
# "suppressed" — that's the only knob that needs a comment.
DEFAULT_WINDOW_HOURS = ALERT_RECENCY_TTL_HOURS

# Capped per-bucket example list. The audit is a calibration view — the
# operator needs *which* titles, not all of them. 5 mirrors the convention
# in ``analytics/recap_template_audit.py``.
_EXAMPLE_CAP = 5

# Per-gate suppression fingerprints in the same order the live ``send_urgent_alert``
# applies them — so a row caught by multiple gates is attributed to the
# FIRST gate that would catch it, matching live precedence. Each entry is
# ``(name, predicate)`` where the predicate takes the article dict and
# returns True if the gate would suppress.
def _looks_low_authority(art: dict) -> bool:
    """Predicate matching ``alert_agent._filter_low_authority_lone`` for a
    lone urgent row. After-the-fact, every alerted row has dup_count=1 in the
    DB (syndication folds aren't persisted as a column), so a row that fired
    as part of a syndicated fold registers here as ``dup_count=1``. That is
    fine — the audit only fires this predicate on rows whose signature does
    NOT match an alerted_sig record, i.e. rows the live gates ACTUALLY
    suppressed. For those rows the live gate also saw dup_count=1 (otherwise
    the corroboration escape valve would have kept the row), so this matches
    the live behaviour."""
    cred = _source_credibility(art.get("source") or "")
    return cred < ALERT_MIN_LONE_SOURCE_CRED


def _looks_stale(art: dict) -> bool:
    """Predicate matching ``alert_agent._article_age_ok``'s drop branch."""
    return not _article_age_ok(art)


# Ordered the same way ``send_urgent_alert`` ordered the gates: synthetic
# first (the load-bearing invariant), then content-shape gates (quote_widget,
# recap_template), then quality gates (low_authority), then time gate
# (stale_published). A future gate addition only needs an entry here.
_GATE_PREDICATES: tuple[tuple[str, Callable[[dict], bool]], ...] = (
    ("synthetic", _is_synthetic),
    ("quote_widget", _looks_like_quote_widget),
    ("recap_template", lambda a: _looks_like_recap_template(a)[0]),
    ("low_authority", _looks_low_authority),
    ("stale_published", _looks_stale),
)


def resolve_db_paths() -> tuple[Path, Path]:
    """Resolve live ``articles.db`` (USB-preferred) and ``alert_recency.db``
    (always local, see ``alert_recency.DB_PATH``). No side effects."""
    usb_db = _USB_PATH / "articles.db"
    if _USB_PATH.exists() and (usb_db.exists() or _USB_PATH.is_mount()):
        articles_db = usb_db
    else:
        articles_db = _LOCAL_PATH / "articles.db"
    recency_db = _LOCAL_PATH / "alert_recency.db"
    return articles_db, recency_db


def _open_ro(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(
        f"file:{path}?mode=ro", uri=True, timeout=10,
    )
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def _fetch_urgent_rows(
    conn: sqlite3.Connection, hours: float,
) -> list[dict]:
    """Pull urgency=2 rows from articles.db inside the window.

    Returns the minimum surface the gate predicates need (``title`` / ``source``
    / ``link`` / ``published`` / ``first_seen``) plus an ``_id`` for traceability.
    ``ai_score`` / ``ml_score`` / ``score_source`` are READ only and surface in
    the example block so an operator can see the calibration story behind a
    suppression."""
    since = (
        datetime.now(timezone.utc) - timedelta(hours=hours)
    ).isoformat()
    rows = conn.execute(
        "SELECT id, url, title, source, published, first_seen, "
        "ai_score, ml_score, score_source "
        f"FROM articles WHERE urgency=2 AND first_seen >= ? AND {LIVE_ONLY_CLAUSE}",
        (since,),
    ).fetchall()
    out: list[dict] = []
    for r in rows:
        out.append({
            "_id": r[0], "link": r[1] or "", "title": r[2] or "",
            "source": r[3] or "", "published": r[4] or "",
            "first_seen": r[5] or "", "ai_score": r[6],
            "ml_score": r[7], "score_source": r[8],
        })
    return out


def _fetch_alerted_sigs(
    conn: sqlite3.Connection, hours: float,
) -> set[str]:
    """Pull every signature alerted in the last ``hours`` from alert_recency.db.

    Mirrors ``alert_recency.recent_signatures`` (read-only, exception-safe at
    the caller). The audit deliberately re-implements the SELECT rather than
    calling that helper so the audit can survive a recency module API change
    without breaking; the canonical signature function ``_signature`` is the
    cross-module contract and is imported."""
    cutoff = (
        datetime.now(timezone.utc) - timedelta(hours=hours)
    ).isoformat()
    rows = conn.execute(
        "SELECT sig FROM alerted_sig WHERE last_ts >= ?", (cutoff,),
    ).fetchall()
    return {r[0] for r in rows if r[0]}


def compute_delivery_audit(
    urgent_rows: Iterable[dict],
    alerted_sigs: set[str],
    *,
    gate_predicates: tuple[tuple[str, Callable[[dict], bool]], ...] = _GATE_PREDICATES,
    example_cap: int = _EXAMPLE_CAP,
) -> dict:
    """Pure function — no DB / IO. Partition urgent rows by delivery state
    and attribute suppressed rows to the gate that catches them.

    ``urgent_rows`` must each carry ``title`` / ``source`` / ``link`` (the
    minimum the live gates read). Missing keys are treated as empty strings;
    a row with no title yields an empty signature which can never appear in
    ``alerted_sigs`` (matching ``_signature``'s contract), so it is reported
    as suppressed and attributed to ``unknown_gate`` rather than crashing.
    """
    delivered: list[dict] = []
    suppressed: list[dict] = []
    suppressed_by: dict[str, int] = {name: 0 for name, _ in gate_predicates}
    suppressed_by["unknown_gate"] = 0
    examples: dict[str, list[dict]] = {
        k: [] for k in suppressed_by
    }

    for art in urgent_rows:
        sig = _signature(art.get("title"))
        if sig and sig in alerted_sigs:
            delivered.append(art)
            continue
        suppressed.append(art)
        # Attribute to the FIRST gate that catches the row — matches live
        # precedence in ``send_urgent_alert``. If nothing catches, the row
        # is a "phantom mark" the audit can't explain (an operator action,
        # a future gate's leftover state, or a bug); attribute to
        # ``unknown_gate`` so the count never silently drops to zero.
        attributed = "unknown_gate"
        for name, pred in gate_predicates:
            try:
                if pred(art):
                    attributed = name
                    break
            except Exception:
                # A predicate must never crash the audit. A buggy gate fires
                # zero suppressions and the row falls through to unknown.
                continue
        suppressed_by[attributed] += 1
        if len(examples[attributed]) < example_cap:
            examples[attributed].append({
                "_id": art.get("_id"),
                "title": (art.get("title") or "")[:140],
                "source": art.get("source") or "",
                "ai_score": art.get("ai_score"),
                "ml_score": art.get("ml_score"),
                "score_source": art.get("score_source"),
            })

    total = len(delivered) + len(suppressed)
    delivery_rate = round(len(delivered) / total, 4) if total else 0.0
    return {
        "total": total,
        "delivered": len(delivered),
        "suppressed": len(suppressed),
        "delivery_rate": delivery_rate,
        "suppressed_by": suppressed_by,
        "suppressed_examples": {
            k: v for k, v in examples.items() if v
        },
    }


def run_audit(hours: float = DEFAULT_WINDOW_HOURS) -> dict:
    """DB shell: open both stores read-only, pull data, compose the audit.

    Window is clamped to the recency TTL — see ``DEFAULT_WINDOW_HOURS``."""
    if hours > ALERT_RECENCY_TTL_HOURS + 1e-6:
        hours = ALERT_RECENCY_TTL_HOURS
    articles_db, recency_db = resolve_db_paths()
    art_conn = _open_ro(articles_db)
    try:
        urgent = _fetch_urgent_rows(art_conn, hours)
    finally:
        art_conn.close()
    try:
        rec_conn = _open_ro(recency_db)
    except sqlite3.OperationalError:
        # No recency DB yet (fresh install / unit test environment) — degrade
        # to "everything looks suppressed" rather than crashing. The audit is
        # still useful as a fingerprint attribution of urgency=2 rows.
        alerted = set()
    else:
        try:
            alerted = _fetch_alerted_sigs(rec_conn, hours)
        finally:
            rec_conn.close()
    out = compute_delivery_audit(urgent, alerted)
    out["window_h"] = round(float(hours), 3)
    out["generated_at"] = datetime.now(timezone.utc).isoformat()
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--hours", type=float, default=DEFAULT_WINDOW_HOURS,
        help=f"Window in hours (default and max: {DEFAULT_WINDOW_HOURS:.1f}, "
             f"the alert_recency TTL).",
    )
    p.add_argument(
        "--pretty", action="store_true",
        help="Pretty-print the JSON report.",
    )
    p.add_argument(
        "--no-write", action="store_true",
        help="Skip writing JSON to OUT_PATH (stdout only).",
    )
    args = p.parse_args()
    report = run_audit(hours=args.hours)
    if not args.no_write:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = OUT_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(report, indent=2))
        tmp.replace(OUT_PATH)
    print(json.dumps(report, indent=2 if args.pretty else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
