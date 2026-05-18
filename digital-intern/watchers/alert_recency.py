"""Cross-cycle (cross-time) syndication suppression for urgent alerts.

``watchers/alert_dedup.py`` collapses syndicated copies that are present in the
**same** ``get_unalerted_urgent()`` batch. It is a pure function over one list
and consults no persistent state. That leaves a real gap the analyst feels as
duplicate "🚨 BREAKING" pushes:

  A breaking story crosses the urgency threshold and is alerted at 10:00; those
  rows go ``urgency=2`` and are excluded from every future
  ``get_unalerted_urgent()``. A *slower* feed (GDELT 10-min sweep, the
  ``gdelt_gkg`` backfill, Google-News round-robin, Substack 10-min, Yahoo
  4-min) then re-collects the **same event** as a **new row** (new id,
  near-identical wire headline). The scorer marks it ``urgency=1``; the next
  20-second ``alert_worker`` cycle returns it; ``dedupe_urgent`` has nothing in
  *that* batch to collapse it against (the 10:00 copies are ``urgency=2``,
  filtered out) — so it fires a **second** standalone BREAKING alert for an
  event the analyst was already told about, possibly hours later.

Live evidence (2026-05): the "US clears/approves H200 chip sales to 10 China
firms" story fired two separate BREAKING pushes ~1.5 h apart
(``reddit/r/technology`` 07:42, ``reddit/r/wallstreetbets`` 09:11) — different
rows, same event. This is the consuming analyst's single most-cited complaint
(duplicate / repeated alerts).

This module records the canonical signature of every story that actually
fired and suppresses a later urgent row whose signature was alerted within
``ALERT_RECENCY_TTL_HOURS``. It reuses ``alert_dedup._signature`` verbatim as
the single source of truth for headline canonicalisation — re-deriving it here
would let the two dedup layers silently drift (the documented anti-drift
discipline; same rationale as ``alert_agent`` reusing
``ml.features._source_credibility``).

Design / safety:
  * A **separate** tiny SQLite file (``data/alert_recency.db``), hardened with
    the canonical ``timeout=30`` + ``WAL`` + ``busy_timeout=30000`` connection
    (mirrors ``article_store`` / ``source_health`` / the 11 ``seen_articles``
    writers). It NEVER touches ``articles.db`` — so the four load-bearing
    invariants (backtest isolation, ml_score≠ai_score, score_source, the
    ``urgency`` state machine) are untouched here *by construction*.
  * Every public entrypoint is best-effort and exception-guarded: a recency-DB
    failure degrades to the *old* behaviour (no suppression). A genuine
    breaking story must still reach the analyst even if this store is broken —
    a missed alert is far worse than a duplicate one.
  * ``partition_already_alerted`` is a pure function (no DB) so the suppression
    decision is unit-testable in isolation, mirroring the
    ``(kept, suppressed)`` shape of the other ``alert_agent`` gates. Untitled
    rows (empty signature) are never suppressed — same policy as
    ``dedupe_urgent``.
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from watchers.alert_dedup import _signature

try:
    from core.logger import get_logger
    _log = get_logger("alert_recency")
except Exception:
    _log = logging.getLogger("alert_recency")

# Tunable. 6 h spans the slowest live syndication delay observed (the GDELT
# 10-min sweep + gdelt_gkg backfill + Google-News round-robin can re-surface a
# wire headline hours after the fast feeds carried it) without being so long
# that a genuinely *new* development sharing an 8-token prefix is wrongly
# muted. This is the same coarse-signature tradeoff dedupe_urgent already makes
# within a batch — only the time axis is added.
ALERT_RECENCY_TTL_HOURS = 6.0

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "alert_recency.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS alerted_sig (
    sig      TEXT PRIMARY KEY,
    last_ts  TEXT NOT NULL,
    title    TEXT,
    hits     INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_alerted_sig_ts ON alerted_sig(last_ts);
"""


def _connect() -> sqlite3.Connection:
    """Hardened connection to the standalone recency DB. Canonical
    ``timeout=30`` + ``WAL`` + ``busy_timeout=30000`` (mirrors
    ``collectors`` ``seen_articles`` writers / ``article_store``)."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=30, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def _now(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    if now.tzinfo is None:
        return now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc)


def recent_signatures(
    ttl_hours: float = ALERT_RECENCY_TTL_HOURS,
    now: datetime | None = None,
    conn: sqlite3.Connection | None = None,
) -> set[str]:
    """Set of canonical signatures alerted within ``ttl_hours``.

    Best-effort: any failure (missing/locked DB, unparsable ts) yields an
    empty set, which makes ``partition_already_alerted`` a no-op — i.e. the
    pre-feature behaviour. Never raises into the alert path.
    """
    cutoff = (_now(now) - timedelta(hours=ttl_hours)).isoformat()
    own = conn is None
    try:
        conn = conn or _connect()
    except Exception as e:  # pragma: no cover - defensive
        _log.warning(f"[alert_recency] open failed (degrading to no-op): {e}")
        return set()
    try:
        rows = conn.execute(
            "SELECT sig FROM alerted_sig WHERE last_ts >= ?", (cutoff,)
        ).fetchall()
        return {r[0] for r in rows if r[0]}
    except Exception as e:
        _log.warning(f"[alert_recency] recent_signatures failed: {e}")
        return set()
    finally:
        if own:
            try:
                conn.close()
            except Exception:
                pass


def partition_already_alerted(
    articles: list[dict], recent_sigs: set[str]
) -> tuple[list[dict], list[dict]]:
    """Pure split of ``articles`` into ``(kept, suppressed)``.

    ``suppressed`` = a row whose canonical ``_signature`` was already alerted
    inside the TTL window (``sig in recent_sigs``). Untitled rows (empty
    signature) are NEVER suppressed — identical policy to ``dedupe_urgent``,
    which also refuses to merge titleless rows. No DB / IO so the decision is
    unit-testable on its own.
    """
    if not recent_sigs:
        return list(articles), []
    kept: list[dict] = []
    suppressed: list[dict] = []
    for a in articles:
        sig = _signature(a.get("title"))
        if sig and sig in recent_sigs:
            suppressed.append(a)
        else:
            kept.append(a)
    return kept, suppressed


def record_alerted(
    articles: list[dict],
    now: datetime | None = None,
    conn: sqlite3.Connection | None = None,
) -> int:
    """Persist the canonical signature of every story that actually fired.

    Upserts ``last_ts`` and bumps ``hits`` for an existing signature. Rows
    with no derivable signature (untitled) are skipped — they are never
    suppressed either, so recording them would be dead weight. Opportunistic
    prune of rows older than ``2 × TTL`` keeps the table tiny. Best-effort:
    a failure is logged and swallowed (the alert already fired; failing to
    record only means a future duplicate is not suppressed — never worse than
    the pre-feature behaviour). Returns the number of signatures recorded.
    """
    ts = _now(now).isoformat()
    seen: set[str] = set()
    payload: list[tuple[str, str, str]] = []
    for a in articles:
        sig = _signature(a.get("title"))
        if not sig or sig in seen:
            continue
        seen.add(sig)
        payload.append((sig, ts, (a.get("title") or "")[:200]))
    if not payload:
        return 0
    own = conn is None
    try:
        conn = conn or _connect()
    except Exception as e:  # pragma: no cover - defensive
        _log.warning(f"[alert_recency] open failed (record skipped): {e}")
        return 0
    try:
        conn.executemany(
            "INSERT INTO alerted_sig (sig, last_ts, title, hits) "
            "VALUES (?, ?, ?, 1) "
            "ON CONFLICT(sig) DO UPDATE SET "
            "  last_ts=excluded.last_ts, "
            "  title=excluded.title, "
            "  hits=alerted_sig.hits+1",
            payload,
        )
        prune_cutoff = (
            _now(now) - timedelta(hours=2 * ALERT_RECENCY_TTL_HOURS)
        ).isoformat()
        conn.execute("DELETE FROM alerted_sig WHERE last_ts < ?", (prune_cutoff,))
        conn.commit()
        return len(payload)
    except Exception as e:
        _log.warning(f"[alert_recency] record_alerted failed: {e}")
        return 0
    finally:
        if own:
            try:
                conn.close()
            except Exception:
                pass
