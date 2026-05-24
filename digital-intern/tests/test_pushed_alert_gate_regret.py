"""Tests for analytics.pushed_alert_gate_regret — the retrospective gate-
coverage audit on the canonical Discord-push ledger.

The pure builder ``build_regret_report`` is the unit-tested contract; the
DB-shell helper is integration-tested via a tmp_path sqlite file.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from analytics.pushed_alert_gate_regret import (
    build_regret_report,
    _load_pushed_titles,
)


# ── Pure builder: empty input ────────────────────────────────────────────────


def test_empty_input_returns_zero_shape():
    """No pushes in window → fully-shaped dict with zeros and an empty
    offenders list. Same zero-data discipline as urgency_label_split_trend
    (a quiet window still emits a stable shape for the dashboard / CLI to
    consume without conditional branches)."""
    out = build_regret_report([], window_h=24)
    assert out["window_h"] == 24
    assert out["total"] == 0
    assert out["would_suppress"] == 0
    assert out["would_keep"] == 0
    assert out["would_suppress_rate"] == 0.0
    assert out["top_offending_titles"] == []
    # All fingerprint buckets pre-seeded to zero — including quote_widget
    # plus every _RECAP_TEMPLATE_PATTERNS name. A consumer iterates a fixed-
    # length series without conditional branches.
    by = out["would_suppress_by"]
    assert by["quote_widget"] == 0
    # At least the canonical recap fingerprints should be seeded zero.
    for required in ("why_trading_today", "earnings_call_recap",
                     "heres_what_happened", "gurufocus_recap"):
        assert by.get(required) == 0, f"missing pre-seeded bucket: {required}"


def test_window_h_clamped_to_at_least_one():
    """A 0 or negative window in the input must be clamped to 1 so the
    snapshot can never report a 0h window (matches the int-clamp discipline
    of every other window-parameterised builder in this repo)."""
    out = build_regret_report([], window_h=0)
    assert out["window_h"] == 1
    out2 = build_regret_report([], window_h=-7)
    assert out2["window_h"] == 1


# ── Pure builder: gate attribution ──────────────────────────────────────────


def test_gurufocus_recap_pushes_counted():
    """A push whose title matches the gurufocus_recap fingerprint is
    counted in ``would_suppress_by['gurufocus_recap']`` and added to the
    offenders list with the right fingerprint label."""
    pushed = [
        {"title": "NVIDIA (NVDA) Reports Robust Earnings While Valuation "
                  "Appears At - GuruFocus", "age_hours": 0.5},
        {"title": "NVIDIA (NVDA) Stock Faces Setback Despite Strong "
                  "Earnings Report - GuruFocus", "age_hours": 1.2},
    ]
    out = build_regret_report(pushed, window_h=6)
    assert out["total"] == 2
    assert out["would_suppress"] == 2
    assert out["would_keep"] == 0
    assert out["would_suppress_rate"] == 1.0
    assert out["would_suppress_by"]["gurufocus_recap"] == 2
    # Every offender carries its fingerprint label so the operator can
    # immediately see WHICH gate would have caught each pushed row.
    fps = [o["fingerprint"] for o in out["top_offending_titles"]]
    assert fps == ["gurufocus_recap", "gurufocus_recap"]


def test_quote_widget_pushes_counted_and_attributed():
    """Quote-widget gate runs first in send_urgent_alert; a push whose
    title matches the listing-card fingerprint is attributed to
    ``quote_widget``, not to any recap bucket."""
    pushed = [
        # The classic $share-card listing-page title.
        {"title": "$NVIDIA (NVDA.US)$ - Moomoo", "age_hours": 0.3},
        # The screener-tape lead.
        {"title": "[YF/most_actives] MU (Micron Technology, Inc.) "
                  "+2.5% @ $698.74 | vol 6M", "age_hours": 1.0},
    ]
    out = build_regret_report(pushed)
    assert out["total"] == 2
    assert out["would_suppress"] == 2
    assert out["would_suppress_by"]["quote_widget"] == 2
    # No recap bucket should have inflated.
    for name, n in out["would_suppress_by"].items():
        if name == "quote_widget":
            continue
        assert n == 0, f"recap bucket {name!r} unexpectedly counted {n}"


def test_real_breaking_news_kept_not_suppressed():
    """The must-survive corpus: a real wire / breaking story should NOT be
    counted as ``would_suppress``. Mirrors the alert-recap-template
    test_*_does_not_overcatch discipline — verifies the builder doesn't
    secretly over-classify legit news."""
    pushed = [
        {"title": "Nvidia Q1 revenue rises 22% to $35.1 billion, beats "
                  "estimates", "age_hours": 0.1},
        {"title": "Fed cuts rates by 50bp, citing labor weakness",
         "age_hours": 0.5},
        {"title": "MU shares halted on pending news", "age_hours": 2.0},
        {"title": "Trump signs executive order on chip exports",
         "age_hours": 3.5},
    ]
    out = build_regret_report(pushed)
    assert out["total"] == 4
    assert out["would_suppress"] == 0
    assert out["would_keep"] == 4
    assert out["would_suppress_rate"] == 0.0
    assert out["top_offending_titles"] == []


def test_mixed_batch_correctly_partitioned():
    """A realistic mixed batch — some real breaking, some recap mill.
    Verifies the rate, the keep count, and the per-fingerprint
    attribution are all consistent."""
    pushed = [
        # Real news — survives.
        {"title": "Nvidia Q1 revenue rises 22% to $35.1 billion",
         "age_hours": 0.1},
        # GuruFocus mill — suppressed.
        {"title": "NVIDIA (NVDA) Reports Strong Earnings Amid AI Investment "
                  "Surge - GuruFocus", "age_hours": 0.4},
        # Quote-widget — suppressed.
        {"title": "$D-Wave Quantum (QBTS.US)$ - Moomoo", "age_hours": 0.6},
        # Real macro — survives.
        {"title": "Fed cuts rates by 50bp, citing labor weakness",
         "age_hours": 1.5},
        # GF Value mill — suppressed (existing fingerprint).
        {"title": "AXT Inc (AXTI) Shares Fall 14.3% -- GF Value Says "
                  "Still Overvalued - GuruFocus", "age_hours": 2.0},
    ]
    out = build_regret_report(pushed, window_h=6)
    assert out["total"] == 5
    assert out["would_suppress"] == 3
    assert out["would_keep"] == 2
    assert out["would_suppress_rate"] == 0.6
    assert out["would_suppress_by"]["gurufocus_recap"] == 1
    assert out["would_suppress_by"]["quote_widget"] == 1
    assert out["would_suppress_by"]["gf_value_says"] == 1


def test_offenders_sorted_newest_first():
    """``top_offending_titles`` sorted by age ascending — newest pushes
    first so the operator sees recent noise at the top of the list."""
    pushed = [
        {"title": "NVIDIA (NVDA) Reports Robust Earnings - GuruFocus",
         "age_hours": 5.0},
        {"title": "$NVIDIA (NVDA.US)$ - Moomoo", "age_hours": 0.1},
        {"title": "NVIDIA (NVDA) Stock Faces Setback Despite Beat - GuruFocus",
         "age_hours": 2.0},
    ]
    out = build_regret_report(pushed)
    ages = [o["age_hours"] for o in out["top_offending_titles"]]
    assert ages == [0.1, 2.0, 5.0]


def test_offenders_capped_at_twenty():
    """The offenders list is bounded — even with 100 suppressed pushes the
    output stays compact for chat / dashboard consumption."""
    pushed = [
        {"title": f"NVIDIA (NVDA) Reports Strong Earnings #{i} - GuruFocus",
         "age_hours": float(i)}
        for i in range(50)
    ]
    out = build_regret_report(pushed)
    assert out["total"] == 50
    assert out["would_suppress"] == 50
    assert len(out["top_offending_titles"]) == 20


def test_empty_or_whitespace_title_ignored():
    """A push row with no title (impossible in practice but defensive) must
    not crash the builder, must not be counted in ``total``."""
    pushed = [
        {"title": "", "age_hours": 0.1},
        {"title": "   ", "age_hours": 0.2},
        {"title": "Nvidia Q1 beats estimates", "age_hours": 0.3},
    ]
    out = build_regret_report(pushed)
    # Only the one real title counts.
    assert out["total"] == 1
    assert out["would_suppress"] == 0


# ── DB shell integration ────────────────────────────────────────────────────


def _make_alert_recency_db(path: Path, rows: list[tuple[str, str]]) -> None:
    """Mint a minimal alert_recency.db with the canonical alerted_sig schema
    and the supplied rows. Each row is (title, last_ts ISO string)."""
    conn = sqlite3.connect(str(path), timeout=5)
    conn.execute(
        "CREATE TABLE alerted_sig (sig TEXT PRIMARY KEY, last_ts TEXT NOT NULL, "
        "title TEXT, hits INTEGER NOT NULL DEFAULT 1)"
    )
    for i, (title, ts) in enumerate(rows):
        conn.execute(
            "INSERT INTO alerted_sig (sig, last_ts, title, hits) VALUES (?,?,?,?)",
            (f"sig{i}", ts, title, 1),
        )
    conn.commit()
    conn.close()


def test_load_pushed_titles_reads_in_window(tmp_path):
    """``_load_pushed_titles`` opens the DB read-only and pulls rows whose
    ``last_ts`` is within the window — the same shape ``build_regret_report``
    consumes. Cross-checks the SQL ``-N hours`` math doesn't drop a fresh
    push."""
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    db = tmp_path / "alert_recency.db"
    _make_alert_recency_db(db, [
        ("Fresh push within 1h", (now - timedelta(minutes=30)).isoformat()),
        ("Old push 48h ago",     (now - timedelta(hours=48)).isoformat()),
    ])
    rows = _load_pushed_titles(db, window_h=24)
    titles = [r["title"] for r in rows]
    assert "Fresh push within 1h" in titles
    assert "Old push 48h ago" not in titles


def test_load_pushed_titles_missing_db_returns_empty(tmp_path):
    """No DB file → empty list, no crash. The audit is best-effort
    (it is observability, not load-bearing) and must degrade silently
    on a fresh install before the alert worker has fired anything."""
    missing = tmp_path / "does_not_exist.db"
    assert _load_pushed_titles(missing, window_h=24) == []


def test_db_to_report_end_to_end(tmp_path):
    """Mint a DB with a known mix, load via the shell, build the report —
    the audit is the same shape as the pure-builder tests."""
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    db = tmp_path / "alert_recency.db"
    _make_alert_recency_db(db, [
        ("NVIDIA (NVDA) Reports Robust Earnings - GuruFocus",
         (now - timedelta(minutes=10)).isoformat()),
        ("Fed cuts rates by 50bp, citing labor weakness",
         (now - timedelta(minutes=20)).isoformat()),
        # Out-of-window — should be excluded by SQL pre-filter.
        ("NVIDIA (NVDA) Stock Faces Setback Despite - GuruFocus",
         (now - timedelta(hours=72)).isoformat()),
    ])
    rows = _load_pushed_titles(db, window_h=24)
    report = build_regret_report(rows, window_h=24)
    # 2 in window (1 GuruFocus suppressed, 1 Fed real news kept).
    assert report["total"] == 2
    assert report["would_suppress"] == 1
    assert report["would_keep"] == 1
    assert report["would_suppress_by"]["gurufocus_recap"] == 1
