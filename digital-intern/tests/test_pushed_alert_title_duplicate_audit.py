"""Specific-value tests for ``analytics.pushed_alert_title_duplicate_audit``.

Assertions pin exact computed values (push_count, duplication_rate_pct,
verdict, source rankings) so any future regex/grouping refactor that
silently shifts the verdict surface fails loudly.
"""
from __future__ import annotations

import re

import pytest

from analytics import pushed_alert_title_duplicate_audit as mod
from analytics.pushed_alert_title_duplicate_audit import (
    DUPLICATION_RATE_HEAVY_PCT,
    DUPLICATION_RATE_LIGHT_PCT,
    LIVE_ONLY_CLAUSE,
    MIN_PUSHES_FOR_VERDICT,
    _normalize_title,
    build_audit,
)


# ── Pure helper: _normalize_title ───────────────────────────────────────────

def test_normalize_lowercases_and_collapses_whitespace():
    assert _normalize_title("  NVDA   Earnings   Beat  ") == "nvda earnings beat"
    assert _normalize_title("Nvidia\n\tposts $81.6B") == "nvidia posts $81.6b"


def test_normalize_preserves_publisher_tag():
    # " - MSN" / " - Motley Fool" deliberately kept (the docstring contract).
    assert _normalize_title("Nvidia Q1 beats - MSN") == "nvidia q1 beats - msn"


def test_normalize_handles_empty_and_none():
    assert _normalize_title("") == ""
    assert _normalize_title(None) == ""  # type: ignore[arg-type]


# ── build_audit envelope ────────────────────────────────────────────────────

def _ts(h: int) -> str:
    """Fixed-anchor ISO timestamp at hour H — deterministic across test runs."""
    return f"2026-05-29T{h:02d}:00:00+00:00"


def test_no_data_below_min_pushes():
    """Sample too small to draw a rate verdict — collapses to NO_DATA."""
    rows = [("Nvidia Q1 beat", "MSN", _ts(1)) for _ in range(MIN_PUSHES_FOR_VERDICT - 1)]
    out = build_audit(rows, window_h=24)
    assert out["verdict"] == "NO_DATA"
    assert out["n_pushes"] == MIN_PUSHES_FOR_VERDICT - 1


def test_no_duplication_verdict_when_all_unique():
    rows = [(f"Distinct headline {i}", "MSN", _ts(1)) for i in range(10)]
    out = build_audit(rows, window_h=24)
    assert out["verdict"] == "NO_DUPLICATION"
    assert out["n_pushes"] == 10
    assert out["n_distinct_titles"] == 10
    assert out["n_duplicate_titles"] == 0
    assert out["n_redundant_pushes"] == 0
    assert out["duplication_rate_pct"] == 0.0
    assert out["duplicate_groups"] == []


def test_heavy_duplication_verdict_on_live_evidence_pattern():
    """Live evidence (2026-05-29 7d pull): NVDA-quarterly headline fired 10×.

    Construct 10 pushes of one title plus a few distinct headlines so the
    duplication rate clears the HEAVY threshold."""
    rows = []
    # The heavily-syndicated NVDA earnings recap (10 pushes)
    nvda = "NVIDIA projects $91B Q2 revenue while outlining $80B buyback - MSN"
    for i in range(10):
        rows.append((nvda, f"src_{i % 3}", _ts(i)))
    # A handful of unique fillers (still well-above the floor sample)
    for i in range(5):
        rows.append((f"Unique headline {i}", "src_z", _ts(i + 11)))

    out = build_audit(rows, window_h=168)
    assert out["n_pushes"] == 15
    assert out["n_distinct_titles"] == 6  # 1 dup + 5 unique
    assert out["n_duplicate_titles"] == 1
    assert out["n_redundant_pushes"] == 9  # 10 - 1 = 9 redundant copies
    # 9 / 15 = 60% — well above the HEAVY threshold (15%)
    assert out["duplication_rate_pct"] == pytest.approx(60.0, abs=0.01)
    assert out["verdict"] == "HEAVY_DUPLICATION"
    # Top group: 10 pushes, normalized title, source set capped + sorted.
    grp = out["duplicate_groups"][0]
    assert grp["push_count"] == 10
    assert grp["title"] == nvda.lower()
    assert grp["sources"] == ["src_0", "src_1", "src_2"]  # sorted, deduplicated
    assert grp["first_seen_oldest"] == _ts(0)
    assert grp["first_seen_newest"] == _ts(9)


def test_light_duplication_verdict_in_middle_band():
    """Rate above the LIGHT threshold (3%) but below HEAVY (15%) → LIGHT."""
    rows = []
    # One title duplicated once (1 redundant push)
    rows.append(("Dup A", "src_a", _ts(1)))
    rows.append(("Dup A", "src_b", _ts(2)))
    # 18 unique titles → total 20 pushes, 1 redundant → 5% rate
    for i in range(18):
        rows.append((f"Unique {i}", "src_z", _ts(i + 3)))

    out = build_audit(rows, window_h=24)
    assert out["n_pushes"] == 20
    assert out["n_redundant_pushes"] == 1
    assert out["duplication_rate_pct"] == pytest.approx(5.0, abs=0.01)
    assert DUPLICATION_RATE_LIGHT_PCT < 5.0 < DUPLICATION_RATE_HEAVY_PCT
    assert out["verdict"] == "LIGHT_DUPLICATION"


def test_borderline_at_light_threshold_collapses_to_no_duplication():
    """A rate EQUAL to the LIGHT threshold (3%) is NOT light — the verdict
    ladder uses strict ``>`` so the boundary belongs to the lower bucket."""
    # 100 pushes, 3 redundant → 3% exactly = threshold
    rows = [("Dup", "src", _ts(1))] * 4  # 3 redundant copies of "Dup"
    for i in range(96):
        rows.append((f"Unique {i}", "src", _ts(i + 2)))
    out = build_audit(rows, window_h=24)
    assert out["n_redundant_pushes"] == 3
    assert out["duplication_rate_pct"] == pytest.approx(3.0, abs=0.01)
    # 3.0 is NOT > 3.0 → NO_DUPLICATION
    assert out["verdict"] == "NO_DUPLICATION"


def test_normalization_collapses_whitespace_variant_to_same_group():
    """Same title with extra interior whitespace is folded into ONE group."""
    rows = []
    rows.append(("Nvidia  posts  $81B", "src_a", _ts(1)))    # double-space
    rows.append(("Nvidia posts $81B", "src_b", _ts(2)))        # single space
    rows.append(("Nvidia\tposts $81B", "src_c", _ts(3)))       # tab → space
    # Add enough unique rows so we get a verdict (NOT NO_DATA).
    for i in range(20):
        rows.append((f"Unique {i}", "src_z", _ts(i + 4)))
    out = build_audit(rows, window_h=24)
    # All three whitespace variants collapsed to ONE distinct title.
    assert out["n_duplicate_titles"] == 1
    grp = out["duplicate_groups"][0]
    assert grp["push_count"] == 3
    assert grp["title"] == "nvidia posts $81b"


def test_paraphrase_remains_distinct_titles():
    """The audit is exact-title (normalized) — paraphrases are NOT folded.

    A "$81.6B revenue" vs "$81.6B quarter" pair stays two distinct titles;
    paraphrase suppression is ``alert_recency.partition_paraphrase_alerted``'s
    job, not this audit's. Double-counting would obscure the exact-title
    duplicates the analyst most wants surfaced."""
    rows = [
        ("Nvidia posts record $81.6B revenue", "MSN", _ts(1)),
        ("Nvidia posts $81.6B quarter", "MSN", _ts(2)),
    ]
    # Pad to clear NO_DATA threshold.
    for i in range(20):
        rows.append((f"Unique {i}", "src_z", _ts(i + 3)))
    out = build_audit(rows, window_h=24)
    assert out["n_duplicate_titles"] == 0   # paraphrase → two distinct groups
    assert out["n_redundant_pushes"] == 0


def test_max_groups_caps_output():
    """``max_groups`` bounds the ``duplicate_groups`` array; total counts are unaffected."""
    rows = []
    # 5 distinct duplicate groups (each 2 pushes) → 5 dup titles total
    for i in range(5):
        rows.append((f"Dup title {i}", "src_a", _ts(i)))
        rows.append((f"Dup title {i}", "src_b", _ts(i + 1)))
    # Pad with uniques to clear the sample floor.
    for i in range(20):
        rows.append((f"Unique {i}", "src_z", _ts(i + 10)))

    out = build_audit(rows, window_h=24, max_groups=2)
    # Total count metrics see ALL duplicate titles, even ones capped from the
    # surfaced list.
    assert out["n_duplicate_titles"] == 5
    assert out["n_redundant_pushes"] == 5
    # ``duplicate_groups`` capped to 2.
    assert len(out["duplicate_groups"]) == 2


def test_max_source_examples_caps_per_group_sources():
    rows = [(f"Single title", f"src_{i}", _ts(i)) for i in range(8)]
    for i in range(20):
        rows.append((f"Unique {i}", "src_z", _ts(i + 9)))
    out = build_audit(rows, window_h=24, max_source_examples=3)
    grp = out["duplicate_groups"][0]
    assert grp["push_count"] == 8
    assert len(grp["sources"]) == 3  # cap respected
    # Determinism — sorted prefix of the unique source set.
    assert grp["sources"] == ["src_0", "src_1", "src_2"]


def test_empty_titles_silently_dropped():
    """Empty / whitespace-only titles never participate (no spurious group)."""
    rows = [("", "src_a", _ts(1)), (None, "src_b", _ts(2)),
            ("   ", "src_c", _ts(3))]
    # Add enough valid rows to reach the verdict floor.
    for i in range(20):
        rows.append((f"Title {i}", "src", _ts(i + 4)))
    out = build_audit(rows, window_h=24)
    assert out["n_distinct_titles"] == 20  # only the valid 20 titles counted
    assert out["n_duplicate_titles"] == 0


def test_ranking_is_deterministic_with_alphabetical_tie_break():
    """Two groups with identical push counts → alphabetical tie-break on title."""
    rows = []
    # Two equally-pushed groups (3× each)
    for _ in range(3):
        rows.append(("Zebra news", "src", _ts(1)))
        rows.append(("Apple news", "src", _ts(2)))
    # Pad
    for i in range(20):
        rows.append((f"Unique {i}", "src", _ts(i + 3)))
    out = build_audit(rows, window_h=24)
    groups = out["duplicate_groups"]
    assert groups[0]["title"] == "apple news"   # 'a' < 'z'
    assert groups[1]["title"] == "zebra news"
    assert groups[0]["push_count"] == groups[1]["push_count"] == 3


# ── Drift locks ─────────────────────────────────────────────────────────────

def test_live_only_clause_in_sync_with_storage():
    """``LIVE_ONLY_CLAUSE`` here mirrors the storage layer's. Hard-coded as a
    string constant to avoid pulling the ArticleStore writer graph, so a
    drift test pins them equal."""
    # Whitespace-tolerant equality — both sides normalised.
    from storage.article_store import _LIVE_ONLY_CLAUSE as storage_clause
    norm = lambda s: re.sub(r"\s+", " ", s).strip()
    assert norm(LIVE_ONLY_CLAUSE) == norm(storage_clause)


def test_verdict_thresholds_remain_in_sane_order():
    """LIGHT < HEAVY (the verdict ladder requires strict monotonicity).

    Pinned so a future tuning pass cannot accidentally invert them and
    create an unreachable verdict bucket."""
    assert DUPLICATION_RATE_LIGHT_PCT < DUPLICATION_RATE_HEAVY_PCT
    assert MIN_PUSHES_FOR_VERDICT >= 1


def test_envelope_keys_are_frozen():
    """Anti-drift: lock the set of keys this module returns so a future
    refactor must explicitly re-version the contract."""
    rows = [(f"Title {i}", "src", _ts(i)) for i in range(20)]
    out = build_audit(rows, window_h=24)
    assert set(out.keys()) == {
        "window_h", "n_pushes", "n_distinct_titles", "n_duplicate_titles",
        "n_redundant_pushes", "duplication_rate_pct", "duplicate_groups",
        "verdict",
    }


# ── Live-conn entrypoint ────────────────────────────────────────────────────

def test_audit_against_in_memory_store_filters_to_urgency_2_and_window():
    """End-to-end: insert a mix of urgency 0/1/2 rows and confirm ``audit``
    filters only urgency=2, applies LIVE_ONLY_CLAUSE, and excludes anything
    outside the window."""
    import sqlite3
    import sys, types
    from pathlib import Path
    # Inline a minimal store stub that exposes ``.conn`` like ArticleStore.
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE articles (id TEXT, url TEXT, title TEXT, source TEXT, "
        "first_seen TEXT, urgency INTEGER)"
    )

    from datetime import datetime, timezone, timedelta
    now = datetime.now(timezone.utc)
    in_window = (now - timedelta(hours=2)).isoformat()
    out_of_window = (now - timedelta(hours=48)).isoformat()

    rows = [
        # In-window urgency=2 — counted (3 copies of one title = duplicate)
        ("a1", "https://x.com/a", "Real headline", "rss", in_window, 2),
        ("a2", "https://x.com/b", "Real headline", "rss", in_window, 2),
        ("a3", "https://x.com/c", "Real headline", "rss", in_window, 2),
        # In-window urgency=1 — NOT counted (only urgency=2 is)
        ("b1", "https://x.com/d", "Queued urgent", "rss", in_window, 1),
        # In-window urgency=2 backtest — excluded by LIVE_ONLY_CLAUSE
        ("c1", "backtest://run/foo", "Synthetic", "rss", in_window, 2),
        # Out-of-window urgency=2 — excluded by time filter
        ("d1", "https://x.com/e", "Old news", "rss", out_of_window, 2),
    ]
    for r in rows:
        conn.execute(
            "INSERT INTO articles VALUES (?, ?, ?, ?, ?, ?)", r
        )
    conn.commit()

    # build a tiny store-stub
    store = types.SimpleNamespace(conn=conn)

    # min_pushes lowered so the small in-memory corpus still gets a verdict
    out = mod.audit(store, hours=24, min_pushes_for_verdict=2)
    assert out["n_pushes"] == 3  # only the three in-window live urgency=2 rows
    assert out["n_duplicate_titles"] == 1
    assert out["n_redundant_pushes"] == 2
    # 2/3 = 66.7% → HEAVY
    assert out["verdict"] == "HEAVY_DUPLICATION"
    assert out["duplicate_groups"][0]["title"] == "real headline"
    assert out["duplicate_groups"][0]["push_count"] == 3


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
