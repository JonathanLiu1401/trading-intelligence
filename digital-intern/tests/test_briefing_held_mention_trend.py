"""Tests for analytics.briefing_held_mention_trend — per-held-ticker
briefing coverage trend builder.

Asserts specific values for each verdict branch (not "no crash"). Pinned
specifically:

* the verdict ladder (NO_DATA / CHRONIC_SILENCE / RECENT_GAP /
  SPORADIC_COVERAGE / ALL_COVERED) at exact threshold boundaries;
* per-ticker fields (appearance_pct, current_silence_streak,
  n_briefings_with, n_briefings, verdict);
* the static-vs-live distinction in CHRONIC_SILENCE (only static-book
  silence triggers it; live-only silence stays at SPORADIC_COVERAGE);
* word-boundary discipline so "MU" cannot fire inside "Museum";
* SSOT parity with ``analysis.claude_analyst._BOOK_TICKERS`` (drift-lock).
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from analytics import briefing_held_mention_trend as B


# ───────────────────────── helpers ───────────────────────────────────────────

def _now() -> datetime:
    # Fixed UTC anchor so test output is deterministic across runs.
    return datetime(2026, 5, 29, 12, 0, 0, tzinfo=timezone.utc)


def _briefing(text: str, hours_ago: float, *, article_count: int = 50) -> dict:
    ts = _now() - timedelta(hours=hours_ago)
    return {
        "id": int(hours_ago * 10),
        "ts": ts.isoformat(),
        "text": text,
        "article_count": article_count,
    }


# Default per_ticker card cap (12) truncates the universe; tests that look
# up specific tickers by name must request the full set.
_FULL = 999


# The full _BOOK_UNIVERSE includes live-only tickers from config/portfolio.json
# which would be SILENT in any synthetic test that only mentions static names —
# they'd flip the aggregate verdict to SPORADIC_COVERAGE. Tests use a single
# helper that mentions EVERY ticker in the universe so the baseline is
# guaranteed ALL_COVERED and the test isolates the one ticker being studied.
def _all_tickers_text() -> str:
    return " ".join(B._BOOK_UNIVERSE) + " market activity"


# ───────────────────────── NO_DATA branch ────────────────────────────────────

class TestNoData:
    def test_empty_input_yields_no_data(self):
        out = B.build_briefing_held_mention_trend([], now=_now())
        assert out["verdict"] == "NO_DATA"
        assert out["n_briefings"] == 0
        assert out["per_ticker"] == []
        # Envelope shape stable even on empty.
        assert "headline" in out and isinstance(out["headline"], str)
        assert out["card_cap"] == B._DEFAULT_CARD_CAP
        assert list(out["static_book_tickers"]) == list(B._BOOK_TICKERS)

    def test_below_min_briefings_yields_no_data(self):
        # 3 briefings — below the 4 floor (_MIN_BRIEFINGS=4 per source).
        out = B.build_briefing_held_mention_trend(
            [
                _briefing("MU MSFT NVDA tape", 1.0),
                _briefing("MU NVDA again", 5.0),
                _briefing("ORCL MSFT", 10.0),
            ],
            now=_now(),
        )
        assert out["verdict"] == "NO_DATA"
        assert out["n_briefings"] == 3
        assert "Only 3 usable briefing(s)" in out["headline"]

    def test_non_iterable_yields_no_data(self):
        out = B.build_briefing_held_mention_trend(42, now=_now())
        assert out["verdict"] == "NO_DATA"

    def test_garbage_rows_silently_skipped(self):
        # Non-dict + missing text + empty text all drop; only 1 valid row
        # remains -> NO_DATA (below MIN_BRIEFINGS=4).
        out = B.build_briefing_held_mention_trend(
            [None, "not a dict", {"text": None}, {"ts": "", "text": "  "},
             {"text": "MU NVDA"}],
            now=_now(),
        )
        assert out["verdict"] == "NO_DATA"
        assert out["n_briefings"] == 1

    def test_min_briefings_constant_at_four(self):
        # Lock the floor — a regression that drops it to 2 would let
        # single-briefing noise tip the trend.
        assert B._MIN_BRIEFINGS == 4


# ───────────────────────── ALL_COVERED branch ────────────────────────────────

class TestAllCovered:
    def test_every_static_ticker_in_every_briefing(self):
        # Build 5 briefings each mentioning every ticker in the FULL
        # universe — including config-derived live-only entries — so the
        # baseline is unambiguously ALL_COVERED regardless of the config
        # at test time.
        briefings = [_briefing(_all_tickers_text(), h) for h in (1, 6, 11, 16, 21)]
        out = B.build_briefing_held_mention_trend(
            briefings, card_cap=_FULL, now=_now())
        assert out["verdict"] == "ALL_COVERED"
        assert out["n_briefings"] == 5
        # Every static ticker should be COVERED with 100% appearance.
        for row in out["per_ticker"]:
            if row["is_static_book"]:
                assert row["verdict"] == "COVERED"
                assert row["appearance_pct"] == 1.0
                assert row["current_silence_streak"] == 0
                assert row["n_briefings_with"] == 5
        assert out["n_silent_book"] == 0
        assert out["n_recent_gap"] == 0
        assert "healthily" in out["headline"].lower()


# ───────────────────────── CHRONIC_SILENCE branch ────────────────────────────

class TestChronicSilence:
    def test_one_static_ticker_silent_all_n_briefings(self):
        # 4 briefings, all mention every OTHER ticker in the universe
        # but NEVER MU — isolates MU as the only silent static name.
        every_but_mu = " ".join(t for t in B._BOOK_UNIVERSE if t != "MU")
        briefings = [_briefing(every_but_mu, h) for h in (1, 6, 11, 16)]
        out = B.build_briefing_held_mention_trend(
            briefings, card_cap=_FULL, now=_now())
        assert out["verdict"] == "CHRONIC_SILENCE"
        mu = next(r for r in out["per_ticker"] if r["ticker"] == "MU")
        assert mu["verdict"] == "SILENT"
        assert mu["appearance_pct"] == 0.0
        assert mu["n_briefings_with"] == 0
        assert mu["current_silence_streak"] == 4  # full window
        assert mu["is_static_book"] is True
        # Severity sort: SILENT static comes first; MU is the only silent
        # static so it MUST be at position 0.
        assert out["per_ticker"][0]["ticker"] == "MU"

    def test_headline_names_silent_static(self):
        briefings = [_briefing("NVDA only", h) for h in (1, 6, 11, 16)]
        out = B.build_briefing_held_mention_trend(
            briefings, card_cap=_FULL, now=_now())
        assert out["verdict"] == "CHRONIC_SILENCE"
        # The headline must surface silent static tickers by name (one of
        # them — top of book order — must appear since headlines cap shown
        # names at 4). LITE is first in _BOOK_TICKERS so it's the lead.
        assert "missing from ALL" in out["headline"]
        # At least one of the static names should be in the headline.
        assert any(t in out["headline"] for t in B._BOOK_TICKERS)


# ───────────────────────── RECENT_GAP branch ─────────────────────────────────

class TestRecentGap:
    def test_three_consecutive_misses_at_top_trigger_recent_gap(self):
        # 6 briefings; MU mentioned in the OLDEST three only.
        # Newest-first order: [no, no, no, MU, MU, MU]. streak = 3.
        # RECENT_GAP_STREAK_FLOOR = 3, so this trips RECENT_GAP exactly.
        # Every OTHER universe ticker mentioned in every briefing so MU
        # is the only one with the gap pattern.
        bg = " ".join(t for t in B._BOOK_UNIVERSE if t != "MU")
        briefings = [
            _briefing(f"NVDA news. {bg}", 1.0),     # newest
            _briefing(f"NVDA news. {bg}", 6.0),
            _briefing(f"NVDA news. {bg}", 11.0),
            _briefing(f"MU earnings. {bg}", 16.0),
            _briefing(f"MU revenue. {bg}", 21.0),
            _briefing(f"MU outlook. {bg}", 26.0),   # oldest
        ]
        out = B.build_briefing_held_mention_trend(
            briefings, card_cap=_FULL, now=_now())
        assert out["verdict"] == "RECENT_GAP"
        mu = next(r for r in out["per_ticker"] if r["ticker"] == "MU")
        assert mu["verdict"] == "RECENT_GAP"
        assert mu["current_silence_streak"] == 3
        assert mu["appearance_pct"] == 0.5  # 3 of 6
        # Headline reports the streak in parens.
        assert "MU (last 3)" in out["headline"]

    def test_two_consecutive_misses_does_not_trigger_recent_gap(self):
        # Boundary test: streak=2 < floor=3 → ticker stays COVERED if pct
        # high enough, or SPORADIC otherwise.
        # MU mentioned in 4 of 5 briefings, misses only the latest 2.
        # Newest-first: [no, no, MU, MU, MU, MU].
        briefings = [
            _briefing("NVDA news", 1.0),
            _briefing("NVDA news", 6.0),
            _briefing("MU earnings", 11.0),
            _briefing("MU revenue", 16.0),
            _briefing("MU outlook", 21.0),
            _briefing("MU news", 26.0),
        ]
        out = B.build_briefing_held_mention_trend(
            briefings, card_cap=_FULL, now=_now())
        # MU has 4/6 ≈ 0.67 appearance + streak=2 — COVERED, not RECENT_GAP.
        mu = next(r for r in out["per_ticker"] if r["ticker"] == "MU")
        assert mu["current_silence_streak"] == 2
        assert mu["verdict"] == "COVERED"
        # No RECENT_GAP at the aggregate either — pick whatever covers the
        # rest of the tickers (they don't show up in this text → SILENT for
        # all of them → CHRONIC_SILENCE wins, but we're only locking the
        # boundary behaviour here, so don't assert aggregate).

    def test_streak_floor_constant(self):
        # Lock the threshold — a regression that drops it to 1 would flag
        # every single-cycle skip as RECENT_GAP.
        assert B.RECENT_GAP_STREAK_FLOOR == 3


# ───────────────────────── SPORADIC branch ───────────────────────────────────

class TestSporadic:
    def test_below_30pct_is_sporadic(self):
        # MU appears in 1 of 6 (16.7%), and only in the oldest position
        # so streak = 5 — but that would trip RECENT_GAP. To keep this
        # test about the SPORADIC threshold, scatter MU so streak < 3.
        # Order newest-first: [MU, no, MU? actually need just 1 MU and streak<3]
        # → [no, no, MU, no, no, no]: 1/6=0.167, streak=2.
        briefings = [
            _briefing("NVDA news", 1.0),    # newest
            _briefing("MSFT news", 6.0),
            _briefing("MU earnings", 11.0),  # only MU mention
            _briefing("ORCL news", 16.0),
            _briefing("NVDA news", 21.0),
            _briefing("NVDA news", 26.0),
        ]
        out = B.build_briefing_held_mention_trend(
            briefings, card_cap=_FULL, now=_now())
        mu = next(r for r in out["per_ticker"] if r["ticker"] == "MU")
        # 1/6 = 0.167 → below 0.30 floor.
        assert mu["appearance_pct"] == pytest.approx(0.167, abs=0.005)
        assert mu["current_silence_streak"] == 2
        assert mu["verdict"] == "SPORADIC"

    def test_at_30pct_is_covered_not_sporadic(self):
        # SPORADIC_FRACTION_FLOOR is strict < — exactly 0.30 should be
        # COVERED. Construct 3 of 10 = 0.30.
        # Newest-first: [MU, no, no, MU, no, no, MU, no, no, no]
        # streak leading misses = 0 (MU is newest).
        briefings = [
            _briefing("MU news", 1.0),       # newest, MU
            _briefing("NVDA", 6.0),
            _briefing("NVDA", 11.0),
            _briefing("MU news", 16.0),
            _briefing("NVDA", 21.0),
            _briefing("NVDA", 26.0),
            _briefing("MU news", 31.0),
            _briefing("NVDA", 36.0),
            _briefing("NVDA", 41.0),
            _briefing("NVDA", 46.0),         # oldest
        ]
        out = B.build_briefing_held_mention_trend(
            briefings, card_cap=_FULL, now=_now())
        mu = next(r for r in out["per_ticker"] if r["ticker"] == "MU")
        assert mu["appearance_pct"] == 0.30
        assert mu["current_silence_streak"] == 0
        # 0.30 is NOT < 0.30 → COVERED.
        assert mu["verdict"] == "COVERED"

    def test_sporadic_threshold_constant(self):
        assert B.SPORADIC_FRACTION_FLOOR == 0.30


# ───────────────────────── word-boundary discipline ──────────────────────────

class TestWordBoundary:
    def test_mu_does_not_fire_inside_museum(self):
        # 4 briefings mentioning "Museum" and "Munich" but never the
        # ticker MU as a standalone word.
        # NVDA mentioned in all so NVDA stays COVERED — that's what isolates
        # the MU-vs-Museum behaviour.
        briefings = [_briefing(
            "Visiting the Smithsonian Museum in Munich today, NVDA recap", h)
            for h in (1, 6, 11, 16)]
        out = B.build_briefing_held_mention_trend(
            briefings, card_cap=_FULL, now=_now())
        mu = next(r for r in out["per_ticker"] if r["ticker"] == "MU")
        assert mu["appearance_pct"] == 0.0
        # NVDA must still match.
        nvda = next(r for r in out["per_ticker"] if r["ticker"] == "NVDA")
        assert nvda["appearance_pct"] == 1.0

    def test_muu_does_not_collapse_to_mu(self):
        # Longest-first alternation — MUU should match standalone without
        # also marking MU as present.
        briefings = [_briefing("MUU was up", h) for h in (1, 6, 11, 16)]
        out = B.build_briefing_held_mention_trend(
            briefings, card_cap=_FULL, now=_now())
        muu = next(r for r in out["per_ticker"] if r["ticker"] == "MUU")
        mu = next(r for r in out["per_ticker"] if r["ticker"] == "MU")
        assert muu["appearance_pct"] == 1.0
        assert mu["appearance_pct"] == 0.0


# ───────────────────────── live-only silence ─────────────────────────────────

class TestLiveOnlySilenceDoesNotTriggerChronic:
    def test_live_only_silence_stays_at_sporadic_coverage(self):
        # Every STATIC book ticker mentioned in every briefing (so none
        # static-silent). A live-only ticker (e.g. one from the
        # sector_watchlist that's not in the static core) that is silent
        # must NOT promote the aggregate verdict to CHRONIC_SILENCE —
        # CHRONIC_SILENCE is reserved for the static core (deliberate
        # operator positioning).
        all_static = " ".join(B._BOOK_TICKERS)
        briefings = [_briefing(all_static, h) for h in (1, 6, 11, 16)]
        out = B.build_briefing_held_mention_trend(briefings, now=_now())
        # If there are no live-only entries this is ALL_COVERED — accept either
        # since the universe depends on the config file. The contract test is:
        # CHRONIC_SILENCE must NOT fire when every static name is covered.
        assert out["verdict"] != "CHRONIC_SILENCE"


# ───────────────────────── SSOT drift-lock ──────────────────────────────────

class TestSSOTDriftLock:
    def test_book_tickers_match_claude_analyst(self):
        # Mirror discipline: the static _BOOK_TICKERS literal MUST match
        # analysis.claude_analyst._BOOK_TICKERS byte-for-byte. The
        # briefing_coverage_audit module enforces the same pin; this test
        # adds a third anchor so any drift across the three literals fails
        # a focused test instead of going unnoticed.
        from analysis.claude_analyst import _BOOK_TICKERS as ANA
        assert B._BOOK_TICKERS == ANA

    def test_book_tickers_match_briefing_coverage_audit(self):
        from analytics.briefing_coverage_audit import _BOOK_TICKERS as BCA
        assert B._BOOK_TICKERS == BCA


# ───────────────────────── envelope completeness ─────────────────────────────

class TestEnvelopeShape:
    def test_response_carries_all_documented_keys(self):
        # Every populated response must carry the documented field set so
        # the dashboard / chat binding can render without conditional
        # branches. Mirrors briefing_coverage_audit's envelope discipline.
        briefings = [_briefing(" ".join(B._BOOK_TICKERS), h) for h in (1, 6, 11, 16)]
        out = B.build_briefing_held_mention_trend(briefings, now=_now())
        for key in (
            "as_of", "verdict", "headline", "n_briefings",
            "window_first_ts", "window_last_ts", "per_ticker",
            "n_silent_book", "n_recent_gap", "n_sporadic", "n_covered",
            "card_cap", "static_book_tickers",
        ):
            assert key in out, f"missing key {key!r}"
        # Per-ticker rows must carry their documented fields.
        for row in out["per_ticker"]:
            for k in (
                "ticker", "is_static_book", "appearance_pct",
                "n_briefings_with", "n_briefings",
                "current_silence_streak", "verdict",
            ):
                assert k in row, f"per_ticker missing {k!r}: {row}"
            # appearance_pct in [0, 1].
            assert 0.0 <= row["appearance_pct"] <= 1.0
            # streak non-negative, never > n_briefings.
            assert 0 <= row["current_silence_streak"] <= row["n_briefings"]

    def test_card_cap_truncates(self):
        # The universe is much larger than card_cap=2; the response must
        # return AT MOST card_cap rows but every counter still reflects
        # the full universe.
        briefings = [_briefing("NVDA", h) for h in (1, 6, 11, 16, 21)]
        out = B.build_briefing_held_mention_trend(briefings, card_cap=2, now=_now())
        assert len(out["per_ticker"]) == 2
        # Counters still cover the full universe.
        assert (out["n_silent_book"] + out["n_recent_gap"]
                + out["n_sporadic"] + out["n_covered"]) == len(B._BOOK_UNIVERSE)


# ───────────────────────── invariant: never raises ───────────────────────────

class TestNeverRaises:
    @pytest.mark.parametrize("garbage", [
        None,
        42,
        "string",
        [{"text": 42}, {"text": None}, {"text": ""}],   # all garbage rows
        [{"ts": "garbage", "text": "MU NVDA news"}] * 4,  # bad ts but good text
        # Iterable that yields then errors out: built as a list to keep test
        # deterministic — the function materializes via list() so a generator
        # exception isn't relevant. Exotic iterables are out of contract.
    ])
    def test_never_raises_on_garbage(self, garbage):
        out = B.build_briefing_held_mention_trend(garbage, now=_now())
        # Must return a dict-shaped envelope with a verdict key.
        assert isinstance(out, dict)
        assert "verdict" in out
        # Unparseable ts is recoverable: the row's text is still scoreable,
        # so a 4-row all-bad-ts list with valid text yields a real verdict
        # (not NO_DATA).
        if isinstance(garbage, list) and len(garbage) >= B._MIN_BRIEFINGS:
            if all(isinstance(r, dict) and isinstance(r.get("text"), str)
                   and r["text"].strip() for r in garbage):
                assert out["verdict"] != "NO_DATA"
