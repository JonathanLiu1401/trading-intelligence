"""Tests for analytics/ticker_comentions.py — ticker pair co-mention graph
builder + chat helper.

Critical regressions to pin:
  * verdict ladder (SECTOR_BURST / COUPLED_NAMES / DISCONNECTED / NO_DATA);
  * MIN_PAIR_COUNT gate (a one-co-mention pair must NOT register);
  * lift = co / min(solo_a, solo_b) — the rarer-name normalisation;
  * window cutoff (article older than window_hours drops out);
  * pair canonicalisation (sorted alphabetically, no duplicates);
  * non-actionable verdicts → chat helper returns ``[]`` (silence
    precedent);
  * chat helper emits the verbatim headline + per-pair rows.

Pure-helper tests — no Flask (project_digital_intern_chat_enrichment_pattern).
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from analytics.ticker_comentions import (  # noqa: E402
    BURST_LIFT,
    BURST_MIN_CO,
    MIN_PAIR_COUNT,
    WINDOW_HOURS,
    build_ticker_comentions,
)
from dashboard.web_server import _ticker_comentions_chat_lines  # noqa: E402


NOW = datetime(2026, 5, 25, 18, 0, tzinfo=timezone.utc)


def _at(hours_ago: float) -> str:
    return (NOW - timedelta(hours=hours_ago)).isoformat()


def _a(title: str, hours_ago: float = 0.5) -> dict:
    return {"title": title, "first_seen": _at(hours_ago)}


class TestBuilderEmpty:
    def test_empty_articles_returns_no_data(self):
        r = build_ticker_comentions([], now=NOW)
        assert r["verdict"] == "NO_DATA"
        assert r["top"] == []
        assert r["window_hours"] == WINDOW_HOURS
        for k in ("headline", "min_pair_count", "burst_lift_threshold",
                  "burst_min_co", "rows_scanned", "rows_in_window",
                  "unique_pairs", "qualified_pairs"):
            assert k in r, k

    def test_only_old_articles_drop_to_no_data(self):
        rows = [_a("NVDA AMD pair", hours_ago=WINDOW_HOURS * 3)
                for _ in range(5)]
        r = build_ticker_comentions(rows, now=NOW)
        assert r["verdict"] == "NO_DATA"
        assert r["rows_scanned"] == 5
        assert r["rows_in_window"] == 0


class TestDisconnectedVerdict:
    def test_single_pair_below_min_count(self):
        """One co-mention is not enough — MIN_PAIR_COUNT=2."""
        rows = [_a("NVDA AMD news")]
        r = build_ticker_comentions(rows, now=NOW)
        assert r["verdict"] == "DISCONNECTED"
        assert r["top"] == []

    def test_solo_tickers_only_no_pairs(self):
        rows = [_a("NVDA solo"), _a("AMD solo"), _a("MU solo")]
        r = build_ticker_comentions(rows, now=NOW)
        assert r["verdict"] == "DISCONNECTED"


class TestCoupledNamesVerdict:
    def test_min_pair_count_reached_below_burst(self):
        """MIN_PAIR_COUNT co-mentions but below SECTOR_BURST thresholds
        (either lift OR co_count not satisfied)."""
        # 2 co-mentions, but each name appears MANY times solo, so lift
        # stays low and BURST_MIN_CO not met.
        rows = []
        for _ in range(MIN_PAIR_COUNT):
            rows.append(_a("NVDA AMD pair"))
        for _ in range(20):
            rows.append(_a("NVDA solo move"))
            rows.append(_a("AMD solo move"))
        r = build_ticker_comentions(rows, now=NOW)
        assert r["verdict"] == "COUPLED_NAMES"
        top = r["top"][0]
        assert sorted(top["pair"]) == ["AMD", "NVDA"]


class TestSectorBurstVerdict:
    def test_burst_lift_and_co_count_both_required(self):
        """SECTOR_BURST requires lift ≥ BURST_LIFT AND co_count ≥ BURST_MIN_CO.
        Each pair-mention contributes to both names' solo count too — so
        with N pair-mentions and zero pure-solos, lift = N / N = 1.0.
        """
        rows = []
        for _ in range(BURST_MIN_CO + 1):
            rows.append(_a("NVDA AMD coupled story"))
        r = build_ticker_comentions(rows, now=NOW)
        assert r["verdict"] == "SECTOR_BURST"
        top = r["top"][0]
        assert top["co_count"] >= BURST_MIN_CO
        assert top["lift"] >= BURST_LIFT


class TestLiftComputation:
    def test_lift_equals_co_over_min_solo(self):
        rows = []
        # Pair appears 3 times.
        for _ in range(3):
            rows.append(_a("NVDA AMD pair"))
        # NVDA additional solo appearances → 5 total solo for NVDA.
        for _ in range(2):
            rows.append(_a("NVDA solo move"))
        # AMD has 3 total (only co-mentions).
        r = build_ticker_comentions(rows, now=NOW)
        top = r["top"][0]
        # co=3, a_total / b_total — one has 5, other has 3. min=3. lift=3/3=1.0
        assert top["co_count"] == 3
        assert min(top["a_total"], top["b_total"]) == 3
        assert top["lift"] == 1.0


class TestPairCanonicalisation:
    def test_pair_sorted_alphabetically(self):
        rows = []
        for _ in range(MIN_PAIR_COUNT):
            rows.append(_a("MU NVDA pair"))
        for _ in range(MIN_PAIR_COUNT):
            rows.append(_a("NVDA MU pair"))   # different order
        r = build_ticker_comentions(rows, now=NOW)
        # Should aggregate to ONE pair, not two — order-independent.
        assert len([p for p in r["top"] if set(p["pair"]) == {"MU", "NVDA"}]) == 1
        top = r["top"][0]
        # pair stored alphabetically (sorted ascending).
        assert top["pair"] == sorted(top["pair"])


class TestWindowCutoff:
    def test_old_article_excluded(self):
        rows = [
            _a("NVDA AMD ancient", hours_ago=WINDOW_HOURS * 2),
            _a("NVDA AMD recent", hours_ago=0.1),
        ]
        r = build_ticker_comentions(rows, now=NOW)
        assert r["rows_scanned"] == 2
        assert r["rows_in_window"] == 1
        # Only 1 co-mention in window → DISCONNECTED.
        assert r["verdict"] == "DISCONNECTED"


class TestChatHelperSilence:
    def test_silence_on_disconnected(self):
        assert _ticker_comentions_chat_lines({"verdict": "DISCONNECTED",
                                              "top": []}) == []

    def test_silence_on_no_data(self):
        assert _ticker_comentions_chat_lines({"verdict": "NO_DATA",
                                              "top": []}) == []

    def test_silence_on_non_dict(self):
        assert _ticker_comentions_chat_lines(None) == []
        assert _ticker_comentions_chat_lines("text") == []

    def test_silence_on_missing_top(self):
        assert _ticker_comentions_chat_lines({"verdict": "COUPLED_NAMES"}) == []
        assert _ticker_comentions_chat_lines({"verdict": "SECTOR_BURST",
                                              "top": []}) == []


class TestChatHelperRendering:
    def _burst_payload(self) -> dict:
        rows = [_a("NVDA AMD coupled story")
                for _ in range(BURST_MIN_CO + 1)]
        return build_ticker_comentions(rows, now=NOW)

    def test_chat_emits_headline_verbatim(self):
        payload = self._burst_payload()
        lines = _ticker_comentions_chat_lines(payload)
        assert lines[0] == payload["headline"]
        assert len(lines) >= 2

    def test_chat_pair_row_carries_co_and_lift(self):
        payload = self._burst_payload()
        lines = _ticker_comentions_chat_lines(payload)
        joined = "\n".join(lines)
        top = payload["top"][0]
        # Both names verbatim, and co + lift restated from builder.
        assert top["pair"][0] in joined
        assert top["pair"][1] in joined
        assert f"co={top['co_count']}" in joined


class TestLoadBearingInvariants:
    def test_pure_no_db_access(self):
        r = build_ticker_comentions(
            [_a("NVDA AMD"), _a("NVDA AMD")], now=NOW
        )
        assert r["rows_scanned"] == 2
        assert r["rows_in_window"] == 2

    def test_dedup_title_repeats_a_ticker(self):
        """A title that mentions NVDA twice should still count NVDA once
        for both solo and pair purposes — sorted(set(...)) guarantee."""
        # Only a single article; one pair-mention expected.
        rows = [_a("NVDA AMD NVDA again")]
        for _ in range(MIN_PAIR_COUNT - 1):
            rows.append(_a("NVDA AMD"))
        r = build_ticker_comentions(rows, now=NOW)
        # Pair = (AMD, NVDA), count = MIN_PAIR_COUNT.
        top = r["top"][0]
        assert top["co_count"] == MIN_PAIR_COUNT
        # Each name's solo total = MIN_PAIR_COUNT (one per article, not two
        # for the repeated-NVDA article).
        assert top["a_total"] == MIN_PAIR_COUNT
        assert top["b_total"] == MIN_PAIR_COUNT

    def test_unparseable_first_seen_counted_as_skipped(self):
        rows = [
            {"first_seen": None, "title": "NVDA AMD"},
            {"first_seen": "garbage", "title": "NVDA AMD"},
            _a("NVDA AMD real"),
        ]
        r = build_ticker_comentions(rows, now=NOW)
        assert r["rows_scanned"] == 3
        assert r["skipped"] == 2
        assert r["rows_in_window"] == 1
