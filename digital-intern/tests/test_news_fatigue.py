"""Tests for analytics.news_fatigue.

The previous implementation read ``ai_score`` only and so was operating on
~7.6% of the live scored corpus (a 2026-05-24 live audit found 375 rows
with ``ai_score > 0`` of 4915 scored rows in 24h — the rest carry
``ai_score=0`` and ``ml_score`` set, the standard ML-only score). Every
ML-only article was a zero-valued sample, so the recent-vs-prior delta
was dominated by which window happened to catch the rare LLM-vetted
rows.

This suite pins:

  1. The pure ``compute_news_fatigue`` shape and contracts.
  2. The unified-score read: ML-only rows MUST contribute their
     ``ml_score`` to the mean (the bug this refactor fixes).
  3. Threshold gating (``min_total_24h``, ``min_recent_6h``,
     ``fatigue_drop``) — a real-volume genuinely-fatigued ticker is
     surfaced; a low-volume one is not; a slight drop is not.
  4. Time-window partitioning is anchored to the supplied ``now``,
     so tests can freeze the clock.
  5. Top-N cap is the ``score_drop``-desc head.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from analytics import news_fatigue


NOW = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)


def _ts(hours_ago: float) -> str:
    return (NOW - timedelta(hours=hours_ago)).isoformat()


# ── _unified_score: the bug this refactor fixes ─────────────────────────────


class TestUnifiedScore:
    def test_ai_score_wins_when_nonzero(self):
        assert news_fatigue._unified_score(7.5, 9.0) == 7.5

    def test_ml_score_used_when_ai_zero(self):
        """The original bug: an ML-only row (``ai_score=0, ml_score=9.7``)
        contributed 0 to the fatigue mean, dragging it down to noise."""
        assert news_fatigue._unified_score(0.0, 9.7) == 9.7

    def test_ml_score_used_when_ai_zero_and_ai_is_explicit_zero(self):
        """0.01 noise-floor is the urgency_scorer pre-floor: a row Sonnet
        labeled as noise (recap/widget). ai > 0 so it wins over ml."""
        assert news_fatigue._unified_score(0.01, 9.7) == 0.01

    def test_none_returned_when_no_label_at_all(self):
        """An unscored row (``ai_score=0, ml_score=NULL``) must NOT count —
        otherwise unscored rows would inflate both windows with zeros."""
        assert news_fatigue._unified_score(0.0, None) is None
        assert news_fatigue._unified_score(None, None) is None

    def test_garbage_strings_degrade_to_none(self):
        """A non-numeric score must not crash — degrade to no-contribution."""
        assert news_fatigue._unified_score("oops", None) is None
        assert news_fatigue._unified_score(None, "garbage") is None


# ── compute_news_fatigue: contract + threshold gating ───────────────────────


def _rows_fatigued_nvda():
    """NVDA story burning out: high score in the prior 18h, low in the
    recent 6h. Uses BOTH ai_score and ml_score paths to verify the unified
    read wires through correctly — ML-only rows (ai=0, ml=9) and
    LLM-vetted rows (ai=8, ml=0) both count."""
    rows = []
    # Prior window: 10 rows at unified ~9.0 (mix of ML-only and LLM)
    for i in range(5):
        rows.append((_ts(8 + i), f"NVDA strong story {i}", 9.0, None))   # LLM
        rows.append((_ts(10 + i), f"NVDA more strong news {i}", 0.0, 9.0))  # ML-only
    # Recent window: 5 rows at unified ~5.0 (story burning out)
    for i in range(5):
        rows.append((_ts(2 + i * 0.5), f"NVDA fading story {i}", 0.0, 5.0))
    return rows


def test_fatigued_ticker_surfaces_with_correct_means():
    out = news_fatigue.compute_news_fatigue(_rows_fatigued_nvda(), now=NOW)
    assert out["fatigued_count"] == 1, out
    assert len(out["tickers"]) == 1
    rec = out["tickers"][0]
    assert rec["ticker"] == "NVDA"
    # 10 prior + 5 recent = 15 total (= MIN_TOTAL_24H)
    assert rec["total_24h"] == 15
    assert rec["recent_6h_count"] == 5
    assert rec["prior_18h_count"] == 10
    # Recent mean ~5.0, prior mean ~9.0 — drop ~4.0
    assert rec["recent_mean_score"] == pytest.approx(5.0)
    assert rec["prior_mean_score"] == pytest.approx(9.0)
    assert rec["score_drop"] == pytest.approx(4.0)


def test_unified_read_changes_verdict_vs_ai_only():
    """Regression pin for the bug this refactor fixes: a ticker whose
    coverage is ML-only (zero LLM labels) would have read as a flat
    zero-mean in both windows under the prior ``ai_score``-only logic,
    so even a real fatigue pattern was invisible. With the unified read
    the same data correctly surfaces the drop."""
    rows = []
    # Prior: 12 ML-only rows averaging 9.0
    for i in range(12):
        rows.append((_ts(7 + i * 0.5), f"AMD strong setup {i}", 0.0, 9.0))
    # Recent: 4 ML-only rows averaging 6.0
    for i in range(4):
        rows.append((_ts(1 + i * 0.5), f"AMD fading print {i}", 0.0, 6.0))
    out = news_fatigue.compute_news_fatigue(rows, now=NOW)
    assert out["fatigued_count"] == 1
    rec = out["tickers"][0]
    assert rec["ticker"] == "AMD"
    assert rec["recent_mean_score"] == pytest.approx(6.0)
    assert rec["prior_mean_score"] == pytest.approx(9.0)
    assert rec["score_drop"] == pytest.approx(3.0)


def test_below_min_total_not_surfaced():
    """A ticker with only 10 articles in 24h (below MIN_TOTAL_24H=15)
    must NOT be flagged regardless of how big the score drop is — small
    samples are noise, not fatigue."""
    rows = []
    for i in range(5):
        rows.append((_ts(8 + i), f"TSLA hot story {i}", 9.0, None))
    for i in range(5):
        rows.append((_ts(2 + i * 0.5), f"TSLA cooling {i}", 3.0, None))
    out = news_fatigue.compute_news_fatigue(rows, now=NOW)
    assert out["fatigued_count"] == 0
    assert out["tickers"] == []


def test_below_min_recent_not_surfaced():
    """A ticker with plenty of volume but only 2 recent articles is dead,
    not fatigued — MIN_RECENT_6H=3 is the still-active gate."""
    rows = []
    for i in range(13):
        rows.append((_ts(8 + i * 0.5), f"MSFT strong {i}", 9.0, None))
    for i in range(2):  # 2 recent — below MIN_RECENT_6H=3
        rows.append((_ts(1 + i), f"MSFT update {i}", 4.0, None))
    out = news_fatigue.compute_news_fatigue(rows, now=NOW)
    assert out["fatigued_count"] == 0


def test_below_fatigue_drop_not_surfaced():
    """A ticker whose score is only slightly down (< FATIGUE_DROP=1.5)
    is normal-cooling, not fatigue."""
    rows = []
    for i in range(10):
        rows.append((_ts(8 + i * 0.5), f"INTC steady {i}", 6.0, None))
    for i in range(5):
        rows.append((_ts(1 + i * 0.5), f"INTC steady recent {i}", 5.5, None))
    out = news_fatigue.compute_news_fatigue(rows, now=NOW)
    # drop=0.5 < 1.5
    assert out["fatigued_count"] == 0


def test_unscored_rows_dropped_not_treated_as_zero():
    """An unscored row (ai=0, ml=None) must NOT participate. The prior
    bug treated ai_score=0 as a real zero-valued sample, dragging the
    mean down on every ML-only row even when they had a real ml_score
    set. The fix: None unified-score = drop from the pool entirely."""
    rows = []
    # 10 unscored rows that would have polluted the mean as zeros.
    for i in range(10):
        rows.append((_ts(7 + i * 0.5), f"AAPL unscored {i}", 0.0, None))
    # 10 prior + 5 recent strong scores — large drop.
    for i in range(10):
        rows.append((_ts(8 + i * 0.5), f"AAPL strong {i}", 9.0, None))
    for i in range(5):
        rows.append((_ts(1 + i * 0.5), f"AAPL fading {i}", 5.0, None))
    out = news_fatigue.compute_news_fatigue(rows, now=NOW)
    rec = next(r for r in out["tickers"] if r["ticker"] == "AAPL")
    # Strict assertion: means must reflect ONLY the scored rows, not
    # zero-padding from the 10 unscored ones.
    assert rec["prior_mean_score"] == pytest.approx(9.0)
    assert rec["recent_mean_score"] == pytest.approx(5.0)


def test_window_partition_anchored_to_now():
    """Articles older than TOTAL_WINDOW_HOURS=24 must be dropped; rows
    within the last RECENT_HOURS=6 land in the recent bucket; the rest
    in the prior."""
    rows = []
    # 5 rows at 30h old — outside the window, must be ignored.
    for i in range(5):
        rows.append((_ts(30 + i), f"GOOG stale story {i}", 9.0, None))
    # 12 prior-window rows.
    for i in range(12):
        rows.append((_ts(8 + i * 0.5), f"GOOG strong story {i}", 9.0, None))
    # 5 recent-window rows.
    for i in range(5):
        rows.append((_ts(1 + i * 0.5), f"GOOG fading story {i}", 4.0, None))
    out = news_fatigue.compute_news_fatigue(rows, now=NOW)
    rec = next(r for r in out["tickers"] if r["ticker"] == "GOOG")
    # Must reflect ONLY the 17 in-window rows, not the 5 stale.
    assert rec["total_24h"] == 17
    assert rec["prior_18h_count"] == 12
    assert rec["recent_6h_count"] == 5


def test_unparseable_timestamp_dropped():
    """A row with a garbage ``first_seen`` must be dropped, not crash."""
    rows = [
        ("not-a-timestamp", "NVDA junk row", 9.0, None),
        ("", "NVDA empty ts", 9.0, None),
    ]
    # Plus enough real data for one fatigue verdict so we know the call
    # returned a sensible payload rather than just an empty shape.
    rows += [(_ts(8 + i * 0.5), f"NVDA strong {i}", 9.0, None)
             for i in range(12)]
    rows += [(_ts(1 + i * 0.5), f"NVDA fading {i}", 5.0, None)
             for i in range(5)]
    out = news_fatigue.compute_news_fatigue(rows, now=NOW)
    rec = next(r for r in out["tickers"] if r["ticker"] == "NVDA")
    # The two junk rows must NOT have inflated total_24h.
    assert rec["total_24h"] == 17


def test_top_n_caps_output():
    """When ``top_n`` is small the output is the top-drop-first head."""
    rows = []
    # Three fatigue patterns of different sizes.
    for tk, prior, recent in [("AAA", 9.0, 5.0), ("BBB", 8.0, 6.0), ("CCC", 9.0, 3.0)]:
        for i in range(12):
            rows.append((_ts(8 + i * 0.5), f"{tk} strong {i}", prior, None))
        for i in range(5):
            rows.append((_ts(1 + i * 0.5), f"{tk} fading {i}", recent, None))
    out = news_fatigue.compute_news_fatigue(rows, now=NOW, top_n=2)
    assert out["fatigued_count"] == 3
    assert len(out["tickers"]) == 2  # capped
    # Highest drop first: CCC=6.0 > AAA=4.0 > BBB=2.0
    assert [r["ticker"] for r in out["tickers"]] == ["CCC", "AAA"]


def test_empty_rows_returns_clean_shape():
    out = news_fatigue.compute_news_fatigue([], now=NOW)
    assert out["fatigued_count"] == 0
    assert out["tickers"] == []
    assert out["generated_at"] == NOW.isoformat()


def test_naive_now_treated_as_utc():
    """A naive ``now`` is normalised to UTC — same convention as the rest
    of the codebase (urgency_scorer, alert_agent, features._parse_published).
    """
    naive = datetime(2026, 5, 24, 12, 0)  # no tzinfo
    rows = [(_ts(2), "NVDA new story", 9.0, None)]
    out = news_fatigue.compute_news_fatigue(rows, now=naive)
    # Smoke: did not raise and timestamps remained interpretable.
    assert out["generated_at"].endswith("+00:00") or "Z" in out["generated_at"]
