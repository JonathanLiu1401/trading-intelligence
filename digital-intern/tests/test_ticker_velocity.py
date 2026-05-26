"""Tests for analytics/ticker_velocity_runner.py — top-ticker arrival-count
velocity (recent vs prior window) builder + chat helper.

Critical regressions to pin:
  * verdict ladder (BREAKING / WARMING / QUIET / NO_DATA);
  * per-ticker classification with both ratio AND min-recent gates;
  * window cutoff (article older than 2× window_min drops out);
  * top-N discovery picks the right tickers;
  * stopword filter (CEO / IPO / FED do not become tickers);
  * Laplace smoothing yields finite ratio when prior=0;
  * chat helper silence on QUIET/NO_DATA (silence-on-healthy precedent);
  * chat helper emits the verbatim headline + per-ticker rows.

Pure-helper tests — no Flask (project_digital_intern_chat_enrichment_pattern).
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from analytics.ticker_velocity_runner import (  # noqa: E402
    BREAKING_RATIO,
    BREAKING_MIN_RECENT,
    WARMING_RATIO,
    WARMING_MIN_RECENT,
    WINDOW_MIN,
    _extract_tickers,
    build_ticker_velocity,
)
from dashboard.web_server import _ticker_velocity_chat_lines  # noqa: E402


NOW = datetime(2026, 5, 25, 18, 0, tzinfo=timezone.utc)


def _at(minutes_ago: float) -> str:
    return (NOW - timedelta(minutes=minutes_ago)).isoformat()


def _a(title: str, minutes_ago: float) -> dict:
    return {"title": title, "first_seen": _at(minutes_ago)}


class TestExtractTickers:
    def test_basic_extraction(self):
        assert "NVDA" in _extract_tickers("NVDA beats earnings")
        assert "MU" in _extract_tickers("$MU surge after HBM news")

    def test_stopwords_excluded(self):
        assert "CEO" not in _extract_tickers("CEO of NVDA speaks")
        assert "IPO" not in _extract_tickers("NVDA IPO rumour false")
        assert "FED" not in _extract_tickers("FED meeting Tuesday")

    def test_too_short_excluded(self):
        assert _extract_tickers("A") == []
        assert _extract_tickers("") == []


class TestBuilderEmpty:
    def test_empty_articles_returns_no_data(self):
        r = build_ticker_velocity([], now=NOW)
        assert r["verdict"] == "NO_DATA"
        assert r["tickers"] == []
        assert r["rows_scanned"] == 0
        assert r["rows_in_window"] == 0
        assert r["window_min"] == WINDOW_MIN
        # Stable shape — every key the chat helper inspects must be present.
        for k in ("headline", "n_breaking", "n_warming",
                  "breaking_ratio_threshold", "breaking_min_recent",
                  "warming_ratio_threshold", "warming_min_recent"):
            assert k in r, k

    def test_only_old_articles_drop_to_no_data(self):
        """Articles older than 2× window_min must not enter the window."""
        rows = [_a("NVDA rallies", WINDOW_MIN * 3) for _ in range(5)]
        r = build_ticker_velocity(rows, now=NOW)
        assert r["verdict"] == "NO_DATA"
        assert r["rows_scanned"] == 5
        assert r["rows_in_window"] == 0


class TestBreakingVerdict:
    def test_breaking_fires_with_high_ratio_and_recent_count(self):
        # Prior window: empty (relies on Laplace smoothing on the
        # denominator so the ratio stays finite). Recent window:
        # BREAKING_MIN_RECENT mentions → ratio ≥ BREAKING_RATIO.
        rows = []
        for i in range(BREAKING_MIN_RECENT + 1):
            rows.append(_a(f"NVDA fresh story {i}", 5 + i))
        r = build_ticker_velocity(rows, now=NOW)
        assert r["verdict"] == "BREAKING"
        # NVDA ranked top.
        assert r["tickers"][0]["ticker"] == "NVDA"
        assert r["tickers"][0]["verdict"] == "BREAKING"
        assert r["n_breaking"] >= 1
        assert r["tickers"][0]["recent"] >= BREAKING_MIN_RECENT
        # newest_age_s recorded.
        assert r["tickers"][0]["newest_age_s"] is not None
        # Headline restates the top ticker.
        assert "NVDA" in r["headline"]
        assert "BREAKING" in r["headline"]


class TestWarmingVerdict:
    def test_warming_fires_below_breaking_above_warming(self):
        # Prior: 1 mention; Recent: WARMING_MIN_RECENT mentions; ratio ≥
        # WARMING_RATIO but recent < BREAKING_MIN_RECENT so it stays
        # WARMING. Use a title-set free of incidental all-caps tokens (no
        # HBM/AI/CEO) so the only ticker the regex catches is MU.
        rows = [_a("MU report due tomorrow", WINDOW_MIN + 30)]
        for i in range(WARMING_MIN_RECENT):
            rows.append(_a(f"MU rises today {i}", 5 + i))
        r = build_ticker_velocity(rows, now=NOW)
        assert r["verdict"] == "WARMING"
        top = r["tickers"][0]
        assert top["ticker"] == "MU"
        assert top["verdict"] == "WARMING"
        # No BREAKING ticker since recent < BREAKING_MIN_RECENT.
        assert r["n_breaking"] == 0


class TestQuietVerdict:
    def test_low_recent_count_stays_quiet(self):
        """A single new mention (recent=1) cannot reach WARMING."""
        rows = [_a("AMD note", WINDOW_MIN + 30),
                _a("AMD new", 5)]
        r = build_ticker_velocity(rows, now=NOW)
        assert r["verdict"] == "QUIET"
        assert r["n_breaking"] == 0
        assert r["n_warming"] == 0

    def test_no_acceleration_stays_quiet(self):
        """Steady arrival rate (ratio ≈ 1) does not fire."""
        rows = []
        for i in range(4):
            rows.append(_a(f"AMD steady {i}", WINDOW_MIN + 10 + i))
        for i in range(4):
            rows.append(_a(f"AMD steady recent {i}", 10 + i))
        r = build_ticker_velocity(rows, now=NOW)
        assert r["verdict"] == "QUIET"


class TestRanking:
    def test_ratio_descending_then_recent(self):
        rows = []
        # NVDA: recent=10, prior=1 → ratio ~5.5
        for i in range(10):
            rows.append(_a(f"NVDA rip {i}", 5 + i))
        rows.append(_a("NVDA quiet day", WINDOW_MIN + 10))
        # MU: recent=4, prior=1 → ratio ~2.5
        for i in range(4):
            rows.append(_a(f"MU steady {i}", 5 + i))
        rows.append(_a("MU prior", WINDOW_MIN + 10))
        r = build_ticker_velocity(rows, now=NOW)
        # Top must be NVDA (higher ratio).
        assert r["tickers"][0]["ticker"] == "NVDA"
        # MU should be second.
        symbols = [t["ticker"] for t in r["tickers"]]
        assert "MU" in symbols
        nvda_idx = symbols.index("NVDA")
        mu_idx = symbols.index("MU")
        assert nvda_idx < mu_idx


class TestLaplaceSmoothing:
    def test_prior_zero_yields_finite_ratio(self):
        """A brand-new ticker with prior=0 must NOT divide-by-zero — the
        Laplace +1 smoothing on numerator and denominator both keeps the
        ratio finite and ranks brand-new bursts highly without going to inf.
        """
        rows = []
        for i in range(BREAKING_MIN_RECENT + 1):
            rows.append(_a(f"WDC surprise {i}", 5 + i))
        r = build_ticker_velocity(rows, now=NOW)
        wdc = next(t for t in r["tickers"] if t["ticker"] == "WDC")
        assert wdc["prior"] == 0
        # ratio = (6+1)/(0+1) = 7.0 (>= BREAKING_RATIO=4.0).
        assert wdc["ratio"] >= BREAKING_RATIO
        assert wdc["ratio"] != float("inf")


class TestTopNLimit:
    def test_top_n_caps_returned_tickers(self):
        rows = []
        for sym in ("ABCD", "EFGH", "IJKL", "MNOP"):
            for _ in range(3):
                rows.append(_a(f"{sym} surge", 5))
        r = build_ticker_velocity(rows, top_n=2, now=NOW)
        assert len(r["tickers"]) == 2


class TestChatHelperSilence:
    def test_silence_on_quiet(self):
        assert _ticker_velocity_chat_lines({"verdict": "QUIET",
                                            "tickers": []}) == []

    def test_silence_on_no_data(self):
        assert _ticker_velocity_chat_lines({"verdict": "NO_DATA",
                                            "tickers": []}) == []

    def test_silence_on_non_dict(self):
        assert _ticker_velocity_chat_lines(None) == []
        assert _ticker_velocity_chat_lines("not a dict") == []
        assert _ticker_velocity_chat_lines(42) == []

    def test_silence_on_missing_tickers_list(self):
        assert _ticker_velocity_chat_lines({"verdict": "BREAKING"}) == []
        assert _ticker_velocity_chat_lines({"verdict": "WARMING",
                                            "tickers": []}) == []


class TestChatHelperRendering:
    def _breaking_payload(self) -> dict:
        # Prior empty + BREAKING_MIN_RECENT+1 recent → ratio >> BREAKING_RATIO.
        rows = []
        for i in range(BREAKING_MIN_RECENT + 1):
            rows.append(_a(f"NVDA fresh story {i}", 5 + i))
        return build_ticker_velocity(rows, now=NOW)

    def test_chat_emits_headline_verbatim(self):
        payload = self._breaking_payload()
        lines = _ticker_velocity_chat_lines(payload)
        # First line is the headline as-is from the builder.
        assert lines[0] == payload["headline"]
        # At least one per-ticker row follows.
        assert len(lines) >= 2

    def test_chat_per_ticker_row_carries_recent_prior_ratio(self):
        payload = self._breaking_payload()
        lines = _ticker_velocity_chat_lines(payload)
        nvda_line = next(ln for ln in lines if "NVDA" in ln and "ratio" in ln)
        # Verbatim numbers from the builder — recent and prior counts must
        # both appear in the rendered line so the analyst sees the raw
        # acceleration, not a re-derived number.
        nvda = next(t for t in payload["tickers"] if t["ticker"] == "NVDA")
        assert str(nvda["recent"]) in nvda_line
        assert str(nvda["prior"]) in nvda_line

    def test_quiet_tickers_excluded_from_detail_rows(self):
        """Only BREAKING/WARMING rows render; a coincident QUIET ticker
        in the tickers list must not appear in the detail block."""
        # Build a payload manually with mixed verdicts.
        payload = {
            "verdict": "BREAKING",
            "headline": "BREAKING: NVDA 1→6 (ratio 3.50)",
            "tickers": [
                {"ticker": "NVDA", "recent": 6, "prior": 1, "ratio": 3.5,
                 "newest_age_s": 30.0, "verdict": "BREAKING"},
                {"ticker": "AMD", "recent": 2, "prior": 1, "ratio": 1.5,
                 "newest_age_s": 60.0, "verdict": "QUIET"},
            ],
        }
        lines = _ticker_velocity_chat_lines(payload)
        joined = "\n".join(lines)
        assert "NVDA" in joined
        assert "AMD" not in joined


class TestLoadBearingInvariants:
    def test_pure_no_db_access(self):
        """Builder must NEVER touch the DB — pass an in-memory list and
        verify the result is computed from it alone."""
        r = build_ticker_velocity(
            [_a("NVDA rip", 5), _a("NVDA rip2", 6)],
            now=NOW,
        )
        # The result must derive entirely from the in-memory input.
        assert r["rows_scanned"] == 2
        assert r["rows_in_window"] == 2

    def test_unparseable_first_seen_counted_as_skipped(self):
        rows = [
            {"first_seen": None, "title": "NVDA rip"},
            {"first_seen": "not a timestamp", "title": "NVDA rip"},
            _a("NVDA real", 5),
        ]
        r = build_ticker_velocity(rows, now=NOW)
        assert r["rows_scanned"] == 3
        assert r["skipped"] == 2
        assert r["rows_in_window"] == 1
