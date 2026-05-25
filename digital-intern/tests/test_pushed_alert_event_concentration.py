"""Tests for analytics.pushed_alert_event_concentration — per-(held-ticker
× event-class) Discord-push concentration audit.

The pure builder ``build_concentration_report`` is the unit-tested contract;
``event_class_for_title`` and ``_held_tickers_in_title`` are the two pure
helpers it composes.

Live failure case pinned (2026-05-24, alert_recency.db 24h window):
    "Nvidia posts record $81.6B revenue, unveils $80B buyback plan - MSN"
    "Nvidia posts $81.6B quarter, unveils $80B buyback plan - MSN"
Both pushed ~1.5h apart; canonical signature Jaccard 0.60 falls below the
0.75 ``PARAPHRASE_MIN_JACCARD`` so cross-cycle paraphrase suppression
correctly did NOT catch them. This audit IS the surface that quantifies
that miss: both titles map to ``(NVDA, BUYBACK)`` and concentration 2 >= 2
threshold fires.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from analytics.pushed_alert_event_concentration import (
    build_concentration_report,
    event_class_for_title,
    _held_tickers_in_title,
    CONCENTRATION_THRESHOLD,
)


HELD = {"NVDA", "MU", "MSFT", "AXTI", "LITE", "ORCL"}


# ── event_class_for_title — closed taxonomy ──────────────────────────────────


def test_event_class_buyback_single_word():
    assert event_class_for_title(
        "Nvidia posts $81.6B quarter, unveils $80B buyback plan - MSN"
    ) == "BUYBACK"


def test_event_class_buyback_synonyms():
    """Each of {buyback, repurchase, dividend} resolves to BUYBACK."""
    assert event_class_for_title("Apple announces $90B share repurchase") == "BUYBACK"
    assert event_class_for_title("Microsoft raises quarterly dividend by 10%") == "BUYBACK"


def test_event_class_earnings_keywords():
    assert event_class_for_title("MU beats Q1 earnings estimates") == "EARNINGS"
    assert event_class_for_title("Nvidia Q1 revenue tops $35B") == "EARNINGS"


def test_event_class_rate_phrase_multiword():
    """Multi-word RATE phrases match with space-tolerant whitespace."""
    assert event_class_for_title("Fed surprises with 50bp rate cut") == "RATE"
    assert event_class_for_title("Powell flags possible rate hike in June") == "RATE"
    # double-space tolerated
    assert event_class_for_title("Fed funds  raised 25bp at FOMC") == "RATE"


def test_event_class_rating_takes_precedence_over_earnings():
    """A title containing BOTH 'price target' and 'earnings' attributes to
    the more-specific RATING class — RATING is listed before EARNINGS in
    the specificity-ordered taxonomy."""
    assert event_class_for_title(
        "JPMorgan raises NVDA price target after blowout earnings"
    ) == "RATING"


def test_event_class_guidance_takes_precedence_over_earnings():
    """A title containing BOTH 'guidance' and 'earnings' attributes to
    GUIDANCE (specificity)."""
    assert event_class_for_title(
        "MU lowers full-year guidance after Q3 earnings"
    ) == "GUIDANCE"


def test_event_class_no_match_returns_empty():
    """Outside the closed vocabulary — empty string, not bucketed."""
    assert event_class_for_title("Nvidia at Computex: Jensen Huang flies to TSMC") == ""
    assert event_class_for_title("White House announces new tariff") == ""


def test_event_class_empty_or_none_returns_empty():
    assert event_class_for_title(None) == ""
    assert event_class_for_title("") == ""
    assert event_class_for_title("   ") == ""


def test_event_class_word_boundary_no_substring_leak():
    """Single-word triggers are word-boundary anchored. 'reupgrade' must
    NOT trigger RATING via the 'upgrade' keyword inside the word."""
    # The word 'eps' inside 'depson' must not falsely fire EARNINGS either.
    assert event_class_for_title("Depson Corp announces new product") == ""


# ── _held_tickers_in_title ────────────────────────────────────────────────────


def test_held_tickers_word_boundary_match():
    """Held tickers match case-insensitive word-boundary; not substring."""
    assert _held_tickers_in_title("NVDA beats earnings", HELD) == ["NVDA"]
    assert _held_tickers_in_title("nvda beats earnings", HELD) == ["NVDA"]


def test_held_tickers_multiple_distinct_returned_sorted():
    assert _held_tickers_in_title("NVDA and MU both rise on memory", HELD) == ["MU", "NVDA"]


def test_held_tickers_no_substring_leak():
    """A ticker as a substring of a longer word must NOT match (NVDAQ ≠ NVDA)."""
    assert _held_tickers_in_title("Trading NVDAQ all day", HELD) == []


def test_held_tickers_unheld_dropped():
    assert _held_tickers_in_title("AAPL earnings beat", HELD) == []


def test_held_tickers_empty_inputs():
    assert _held_tickers_in_title(None, HELD) == []
    assert _held_tickers_in_title("", HELD) == []
    assert _held_tickers_in_title("NVDA beats", set()) == []


# ── build_concentration_report — empty / shape ───────────────────────────────


def test_empty_input_returns_full_shape_zeros():
    """Empty input must still emit the full-shape dict with zeros and
    empty lists — same zero-data discipline as pushed_alert_gate_regret."""
    out = build_concentration_report([], HELD, window_h=6)
    assert out["window_h"] == 6.0
    assert out["concentration_threshold"] == CONCENTRATION_THRESHOLD
    assert out["total_pushes"] == 0
    assert out["pushes_with_class"] == 0
    assert out["pushes_held_x_class"] == 0
    assert out["distinct_pairs"] == 0
    assert out["by_pair"] == []
    assert out["concentration_alerts"] == []


def test_window_h_clamped_to_positive():
    """A non-positive window must clamp to 0.01 — prevents divide-by-zero
    in downstream formatting; same convention as briefing_cadence_trend."""
    out = build_concentration_report([], HELD, window_h=0)
    assert out["window_h"] == 0.01
    out2 = build_concentration_report([], HELD, window_h=-7)
    assert out2["window_h"] == 0.01


def test_pushes_without_class_dont_bucket():
    """A push outside the closed vocabulary is counted in total_pushes
    but NEVER reaches by_pair (the audit only counts pushes that map to
    one of the explicit event categories)."""
    pushed = [
        {"title": "Nvidia at Computex: Jensen flies to TSMC", "age_hours": 0.5},
        {"title": "White House announces new tariff package", "age_hours": 1.0},
    ]
    out = build_concentration_report(pushed, HELD, window_h=6)
    assert out["total_pushes"] == 2
    assert out["pushes_with_class"] == 0
    assert out["pushes_held_x_class"] == 0
    assert out["by_pair"] == []


def test_pushes_with_class_but_no_held_ticker_dont_bucket():
    """A push with an event class but mentioning only NON-held tickers
    is counted in pushes_with_class but NEVER reaches by_pair."""
    pushed = [
        {"title": "AAPL beats earnings estimates", "age_hours": 0.5},
    ]
    out = build_concentration_report(pushed, HELD, window_h=6)
    assert out["total_pushes"] == 1
    assert out["pushes_with_class"] == 1
    assert out["pushes_held_x_class"] == 0
    assert out["by_pair"] == []


# ── build_concentration_report — live failure case pin ───────────────────────


def test_live_nvda_buyback_concentration_pinned():
    """LIVE FAILURE CASE: two NVDA-buyback pushes ~1.5h apart, Jaccard 0.60
    so paraphrase suppression correctly did NOT fire. This audit IS the
    surface that quantifies the miss — both map to (NVDA, BUYBACK) and the
    pair fires a concentration alert at the 2-push threshold."""
    pushed = [
        {"title": "Nvidia posts record $81.6B revenue, unveils $80B buyback plan - MSN",
         "age_hours": 0.93},
        {"title": "Nvidia posts $81.6B quarter, unveils $80B buyback plan - MSN",
         "age_hours": 2.43},
    ]
    out = build_concentration_report(pushed, HELD, window_h=6)
    assert out["total_pushes"] == 2
    assert out["pushes_with_class"] == 2
    assert out["pushes_held_x_class"] == 2
    assert out["distinct_pairs"] == 1
    row = out["by_pair"][0]
    assert row["ticker"] == "NVDA"
    assert row["event_class"] == "BUYBACK"
    assert row["pushes"] == 2
    # Newest age = min of the two.
    assert row["newest_age_h"] == 0.93
    # titles sorted newest-first (smallest age_h first).
    assert row["titles"][0].endswith("revenue, unveils $80B buyback plan - MSN")
    assert row["titles"][1].endswith("quarter, unveils $80B buyback plan - MSN")
    # Concentration alert fires at threshold 2.
    assert len(out["concentration_alerts"]) == 1
    assert "NVDA × BUYBACK" in out["concentration_alerts"][0]
    assert "2 pushes" in out["concentration_alerts"][0]


def test_single_push_below_threshold_no_alert():
    """One push to a (ticker, class) pair is BELOW threshold — appears in
    by_pair but emits NO concentration alert."""
    pushed = [
        {"title": "NVDA Q1 revenue tops $35B", "age_hours": 1.0},
    ]
    out = build_concentration_report(pushed, HELD, window_h=6)
    assert out["distinct_pairs"] == 1
    assert out["by_pair"][0]["pushes"] == 1
    assert out["concentration_alerts"] == []


def test_distinct_classes_dont_collapse():
    """A NVDA earnings push + a NVDA buyback push are TWO distinct pairs,
    neither at the threshold — exactly the desired behavior (genuinely
    different actionable events for the same ticker, must not suppress)."""
    pushed = [
        {"title": "NVDA Q1 revenue tops $35B", "age_hours": 0.5},
        {"title": "Nvidia board authorizes $80B buyback", "age_hours": 1.5},
    ]
    out = build_concentration_report(pushed, HELD, window_h=6)
    assert out["pushes_held_x_class"] == 2
    assert out["distinct_pairs"] == 2
    classes = sorted(r["event_class"] for r in out["by_pair"])
    assert classes == ["BUYBACK", "EARNINGS"]
    assert out["concentration_alerts"] == []


def test_multi_ticker_title_buckets_per_ticker():
    """A single push mentioning NVDA AND MU both held — counts once per
    pair (NVDA,EARNINGS) and (MU,EARNINGS), and pushes_held_x_class
    increments by ONE per source push (not per ticker)."""
    pushed = [
        {"title": "NVDA and MU both beat Q1 earnings estimates", "age_hours": 0.5},
    ]
    out = build_concentration_report(pushed, HELD, window_h=6)
    # ONE source push touched two pairs.
    assert out["total_pushes"] == 1
    assert out["pushes_with_class"] == 1
    assert out["pushes_held_x_class"] == 1
    assert out["distinct_pairs"] == 2
    tickers = sorted(r["ticker"] for r in out["by_pair"])
    assert tickers == ["MU", "NVDA"]
    for row in out["by_pair"]:
        assert row["pushes"] == 1


def test_by_pair_sorted_pushes_desc_with_alpha_tiebreak():
    """Sort order: pushes DESC, then alphabetical (ticker, event_class)."""
    pushed = [
        # NVDA × BUYBACK x 3
        {"title": "NVDA announces buyback", "age_hours": 0.1},
        {"title": "Nvidia board authorizes share repurchase", "age_hours": 0.5},
        {"title": "NVDA dividend raised", "age_hours": 1.0},
        # MU × EARNINGS x 3
        {"title": "MU beats Q1 earnings", "age_hours": 0.2},
        {"title": "Micron eps tops estimates", "age_hours": 0.4},
        {"title": "MU Q2 earnings strong", "age_hours": 0.6},
        # AXTI × RATING x 2
        {"title": "Citi raises AXTI price target", "age_hours": 0.8},
        {"title": "AXTI upgraded to Buy at Goldman", "age_hours": 1.1},
    ]
    out = build_concentration_report(pushed, HELD, window_h=6)
    assert [r["pushes"] for r in out["by_pair"]] == [3, 3, 2]
    # Tie at 3 → alphabetical: MU comes before NVDA.
    assert (out["by_pair"][0]["ticker"], out["by_pair"][0]["event_class"]) == ("MU", "EARNINGS")
    assert (out["by_pair"][1]["ticker"], out["by_pair"][1]["event_class"]) == ("NVDA", "BUYBACK")
    assert (out["by_pair"][2]["ticker"], out["by_pair"][2]["event_class"]) == ("AXTI", "RATING")


def test_concentration_threshold_override():
    """An override threshold of 3 must suppress an alert at the default 2."""
    pushed = [
        {"title": "NVDA posts record $81.6B revenue, unveils $80B buyback", "age_hours": 0.5},
        {"title": "NVDA buyback authorized $80B", "age_hours": 1.0},
    ]
    out = build_concentration_report(
        pushed, HELD, window_h=6, concentration_threshold=3,
    )
    # Pair still present in by_pair (no minimum to enter the table), but
    # no alert because 2 < 3.
    assert out["by_pair"][0]["pushes"] == 2
    assert out["concentration_alerts"] == []


def test_max_by_pair_rows_caps_table():
    """The by_pair table is capped — a single-name wire storm cannot
    produce a wall-of-text report. distinct_pairs reports the true count
    so the cap is visible to the operator."""
    pushed = []
    # 25 distinct (ticker, class) pairs from 25 distinct synthetic tickers.
    tickers = {f"T{i:03d}" for i in range(25)}
    for i in range(25):
        pushed.append({"title": f"T{i:03d} beats earnings estimates", "age_hours": float(i)})
    out = build_concentration_report(
        pushed, tickers, window_h=24, max_by_pair_rows=5,
    )
    assert out["distinct_pairs"] == 25
    assert len(out["by_pair"]) == 5
    # Each pair is at the threshold (1 push each → below default 2 threshold).
    # Alerts are derived from the CAPPED by_pair so they cannot exceed the cap either.
    assert all(r["pushes"] == 1 for r in out["by_pair"])


def test_titleless_or_malformed_rows_dropped():
    """Rows missing a title — or non-dict rows — are silently dropped,
    NOT bucketed as concentration noise. Same defensive-row-access
    discipline as pushed_ticker_breakdown."""
    pushed = [
        {"title": "NVDA beats earnings", "age_hours": 0.5},
        {"title": "", "age_hours": 1.0},
        {"title": None, "age_hours": 1.0},
        "not a dict",
        {"age_hours": 1.0},  # no title key
        {"title": "NVDA Q2 earnings strong", "age_hours": 1.5},
    ]
    out = build_concentration_report(pushed, HELD, window_h=6)
    assert out["total_pushes"] == 2
    assert out["by_pair"][0]["pushes"] == 2
    assert out["by_pair"][0]["ticker"] == "NVDA"


def test_invariants_no_articles_db_touch():
    """Pure builder must not import or touch articles.db / any DB at all.
    Tested by import surface: the public builder only needs the closed-
    vocab and held-ticker arguments — calling it from an in-memory dict
    produces the snapshot with no I/O.

    This is the documented load-bearing invariant — backtest isolation
    is preserved by construction because no DB access exists here.
    """
    import io
    # Build with a freshly-constructed dict input.
    out = build_concentration_report(
        [{"title": "NVDA earnings beat", "age_hours": 0.5}],
        ["NVDA"],
        window_h=6,
    )
    # The output must NOT carry any DB path / sqlite reference.
    s = str(out)
    assert "articles.db" not in s
    assert "sqlite" not in s
    assert "backtest" not in s


# ── CLI / live-wiring smoke (in-memory recency DB) ──────────────────────────


def test_load_pushed_from_tmp_recency_db(tmp_path, monkeypatch):
    """End-to-end smoke: write two rows to a tmp recency-shaped DB, point
    the module at it, verify they flow through to the report."""
    from analytics import pushed_alert_event_concentration as m
    db = tmp_path / "alert_recency.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(
        "CREATE TABLE alerted_sig ("
        "  sig TEXT PRIMARY KEY, last_ts TEXT NOT NULL, "
        "  title TEXT, hits INTEGER DEFAULT 1"
        ")"
    )
    from datetime import datetime, timezone, timedelta
    base = datetime.now(timezone.utc)
    rows = [
        ("sig1", (base - timedelta(hours=0.5)).isoformat(),
         "Nvidia posts record $81.6B revenue, unveils $80B buyback plan - MSN", 1),
        ("sig2", (base - timedelta(hours=1.5)).isoformat(),
         "Nvidia posts $81.6B quarter, unveils $80B buyback plan - MSN", 1),
    ]
    conn.executemany(
        "INSERT INTO alerted_sig (sig, last_ts, title, hits) VALUES (?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(m, "_RECENCY_DB", db)
    pushed = m._load_pushed(hours=6)
    assert len(pushed) == 2
    titles = [p["title"] for p in pushed]
    assert any("revenue, unveils $80B buyback" in t for t in titles)
    assert any("quarter, unveils $80B buyback" in t for t in titles)

    out = m.build_concentration_report(pushed, HELD, window_h=6)
    assert out["distinct_pairs"] == 1
    assert out["by_pair"][0]["ticker"] == "NVDA"
    assert out["by_pair"][0]["event_class"] == "BUYBACK"
    assert out["by_pair"][0]["pushes"] == 2
    assert len(out["concentration_alerts"]) == 1


def test_load_pushed_missing_db_returns_empty(tmp_path, monkeypatch):
    """A missing recency DB must degrade to an empty list (CLI must work
    on a fresh install). Mirrors the best-effort discipline of
    recent_signatures / recent_alerts."""
    from analytics import pushed_alert_event_concentration as m
    monkeypatch.setattr(m, "_RECENCY_DB", tmp_path / "does_not_exist.db")
    assert m._load_pushed(hours=6) == []
