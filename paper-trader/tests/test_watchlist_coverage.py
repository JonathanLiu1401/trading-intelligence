"""Tests for analytics/watchlist_coverage.py — pure, deterministic.

The contract under test: scan the recent decision stream for
per-watchlist-ticker attention (last_seen_ts, mentions_24h /
mentions_7d, action counts) and emit a verdict ladder distinct from
``ticker_decision_mix`` (per-ticker counts but never names ignored
tickers).

Pins the verdict-ladder boundaries, the ``\\b``-anchored reasoning
regex (so "AMD" hits the AMD ticker but "formatted" does NOT hit "T"),
the longest-first alternation so AMD/AM never collide, and the
drift-lock against the dashboard's canonical ``_parse_action_ticker``.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from paper_trader.analytics.watchlist_coverage import (
    CONCENTRATED_TOP3_SHARE,
    DEEP_HOURS,
    RECENT_HOURS,
    STAGNANT_SHARE,
    STALE_HOURS,
    _action_ticker,
    _compile_ticker_pattern,
    _extract_reasoning_mentions,
    build_watchlist_coverage,
)


NOW = datetime(2026, 5, 21, 12, 0, 0, tzinfo=timezone.utc)


def _dec(ts: datetime, action: str = "NO_DECISION",
         reasoning: str = "") -> dict:
    return {"timestamp": ts.isoformat(),
            "action_taken": action,
            "reasoning": reasoning}


class TestActionTickerParity:
    def test_action_ticker_mirrors_dashboard(self):
        # Drift-lock: any future change in the dashboard's parser must
        # be mirrored here.
        from paper_trader.dashboard import _parse_action_ticker as canon
        for inp in (
            "BUY NVDA → FILLED",
            "SELL MU → BLOCKED",
            "HOLD NVDA → HOLD",
            "BUY_CALL TSLA → FILLED",
            "NO_DECISION",
            "BLOCKED",
            "",
            "REBALANCE → BLOCKED",
            "BUY CASH → BLOCKED",
            "BUY NONE → BLOCKED",
        ):
            assert _action_ticker(inp) == canon(inp), f"divergence on {inp!r}"


class TestTickerPattern:
    def test_whole_word_match_only(self):
        pat = _compile_ticker_pattern(["MU", "T"])
        # "MU" should match; "format" should NOT match "T"
        assert _extract_reasoning_mentions(
            "memory chip MU rallies", pat) == {"MU"}
        assert _extract_reasoning_mentions("format-only content", pat) == set()
        # Inside a longer word, no match.
        assert _extract_reasoning_mentions("MULTITUDE", pat) == set()

    def test_longest_match_wins(self):
        # AMD and AM both present in the watchlist: "AMD" must NOT be
        # split into "AM" + "D".
        pat = _compile_ticker_pattern(["AM", "AMD"])
        assert _extract_reasoning_mentions("AMD strong setup", pat) == {"AMD"}

    def test_case_insensitive_whole_word(self):
        pat = _compile_ticker_pattern(["MU"])
        # The regex itself is case-sensitive — production WATCHLIST is
        # uppercase and reasoning text typically capitalizes tickers.
        # We pin the documented contract here.
        assert _extract_reasoning_mentions("MU beat", pat) == {"MU"}
        assert _extract_reasoning_mentions("mu beat", pat) == set()

    def test_empty_or_garbage_pattern_inputs(self):
        assert _compile_ticker_pattern([]) is None
        assert _compile_ticker_pattern(None) is None
        # Numbers / non-alnum tickers are filtered out.
        assert _compile_ticker_pattern(["..."]) is None

    def test_reasoning_extraction_never_raises(self):
        pat = _compile_ticker_pattern(["MU"])
        assert _extract_reasoning_mentions(None, pat) == set()
        assert _extract_reasoning_mentions("", pat) == set()
        assert _extract_reasoning_mentions("text", None) == set()


class TestEmptyAndDegradedInputs:
    def test_empty_watchlist(self):
        r = build_watchlist_coverage([], [_dec(NOW)], now=NOW)
        assert r["verdict"] == "NO_DATA"
        assert r["by_ticker"] == []
        assert r["n_watchlist"] == 0

    def test_empty_decisions(self):
        r = build_watchlist_coverage(["NVDA", "MU"], [], now=NOW)
        assert r["verdict"] == "NO_DATA"

    def test_garbage_rows_dont_raise(self):
        # Rows missing timestamp / action_taken / reasoning.
        bad = [
            {"timestamp": None, "action_taken": None, "reasoning": None},
            {"timestamp": "garbage", "action_taken": "BUY NVDA → FILLED",
             "reasoning": None},
            _dec(NOW, "BUY MU → FILLED", "MU strong"),
        ]
        r = build_watchlist_coverage(["NVDA", "MU"], bad, now=NOW)
        # MU is the only row with a valid ts ⇒ only one mention; NVDA
        # has a parse-failed ts, so its hours_since_last_seen is None
        # but it's still "seen".
        by = {b["ticker"]: b for b in r["by_ticker"]}
        assert by["MU"]["mentions_24h"] == 1
        assert by["MU"]["never_seen"] is False
        # NVDA was seen via the malformed-ts row, so it isn't
        # never_seen, but the unparseable ts means hours_since is None
        # (we record None rather than raise).
        assert by["NVDA"]["never_seen"] is False
        assert by["NVDA"]["hours_since_last_seen"] is None


class TestActionTickerCapture:
    def test_action_taken_counts_as_seen(self):
        wl = ["NVDA", "MU"]
        dec = [_dec(NOW - timedelta(minutes=5), "BUY NVDA → FILLED",
                    reasoning="setup looks strong")]
        r = build_watchlist_coverage(wl, dec, now=NOW)
        by = {b["ticker"]: b for b in r["by_ticker"]}
        assert by["NVDA"]["never_seen"] is False
        assert by["NVDA"]["mentions_24h"] == 1
        assert by["NVDA"]["action_count_7d"] == 1
        assert by["MU"]["never_seen"] is True

    def test_no_decision_does_not_increment_action_count(self):
        wl = ["NVDA"]
        dec = [_dec(NOW - timedelta(minutes=5), "NO_DECISION",
                    reasoning="NVDA rally continues")]
        r = build_watchlist_coverage(wl, dec, now=NOW)
        by = {b["ticker"]: b for b in r["by_ticker"]}
        # Reasoning mention ⇒ seen, but no action ⇒ action_count_7d == 0
        assert by["NVDA"]["never_seen"] is False
        assert by["NVDA"]["action_count_7d"] == 0
        assert by["NVDA"]["mentions_24h"] == 1

    def test_reasoning_mention_only_still_counts_as_seen(self):
        wl = ["NVDA", "MU"]
        dec = [_dec(NOW - timedelta(minutes=5),
                    "BUY NVDA → FILLED",
                    reasoning="NVDA cuts, MU also strong")]
        r = build_watchlist_coverage(wl, dec, now=NOW)
        by = {b["ticker"]: b for b in r["by_ticker"]}
        assert by["MU"]["mentions_24h"] == 1
        assert by["MU"]["action_count_7d"] == 0  # action_taken was NVDA, not MU

    def test_off_watchlist_ticker_ignored(self):
        wl = ["NVDA"]
        dec = [_dec(NOW - timedelta(minutes=5),
                    "BUY GOOGL → FILLED",
                    reasoning="GOOGL earnings strong")]
        r = build_watchlist_coverage(wl, dec, now=NOW)
        # GOOGL is not on the watchlist ⇒ no row, NVDA stays unseen.
        assert {b["ticker"] for b in r["by_ticker"]} == {"NVDA"}
        assert r["by_ticker"][0]["never_seen"] is True


class TestRecencyWindows:
    def test_24h_vs_7d_split(self):
        wl = ["NVDA"]
        dec = [
            _dec(NOW - timedelta(hours=2),  "BUY NVDA → FILLED"),    # 24h
            _dec(NOW - timedelta(hours=72), "HOLD NVDA → HOLD"),     # 7d only
            _dec(NOW - timedelta(days=10),  "BUY NVDA → FILLED"),    # outside both
        ]
        r = build_watchlist_coverage(wl, dec, now=NOW)
        nvda = r["by_ticker"][0]
        assert nvda["mentions_24h"] == 1
        assert nvda["mentions_7d"] == 2

    def test_last_seen_is_the_newest_match(self):
        wl = ["NVDA"]
        dec = [
            _dec(NOW - timedelta(hours=1), "BUY NVDA → FILLED"),
            _dec(NOW - timedelta(hours=4), "HOLD NVDA → HOLD"),
        ]
        r = build_watchlist_coverage(wl, dec, now=NOW)
        nvda = r["by_ticker"][0]
        assert nvda["hours_since_last_seen"] == 1.0
        assert "BUY NVDA" in (nvda["last_seen_action"] or "")


class TestVerdictLadder:
    def test_stagnant_when_majority_untouched(self):
        # 10-ticker WL, only 2 ever seen, rest never ⇒ STAGNANT.
        wl = [f"T{i}" for i in range(10)]
        dec = [_dec(NOW - timedelta(hours=1),
                    f"BUY T0 → FILLED", reasoning="T1 mention")]
        r = build_watchlist_coverage(wl, dec, now=NOW)
        assert r["verdict"] == "STAGNANT"
        assert r["n_never_seen"] == 8

    def test_stagnant_when_all_stale_beyond_7d(self):
        wl = ["NVDA", "MU", "AMD"]
        dec = [_dec(NOW - timedelta(days=10),
                    "BUY NVDA → FILLED", reasoning="MU AMD")]
        r = build_watchlist_coverage(wl, dec, now=NOW)
        # All 3 seen, but all > 7d ⇒ STAGNANT (stale_share = 1.0 > 0.5)
        assert r["verdict"] == "STAGNANT"

    def test_concentrated_when_top3_dominates_with_many_mentions(self):
        # 10 BUYs of T0 each with reasoning citing T1+T2 → top three
        # tickers absorb 30 mentions. Add singleton attention for
        # T3..T5 so the stale-share dodges STAGNANT (4 never-seen of
        # 10 = 40% ≤ 50%). Total mentions = 30 + 3 = 33 ⇒
        # top3 share = 30/33 = 0.909 ≥ 0.80 ⇒ CONCENTRATED.
        wl = [f"T{i}" for i in range(10)]
        dec = []
        for _ in range(10):
            dec.append(_dec(NOW - timedelta(hours=1),
                            "BUY T0 → FILLED",
                            reasoning="T1 T2 noise"))
        for i in range(3, 6):
            dec.append(_dec(NOW - timedelta(hours=1),
                            f"HOLD T{i} → HOLD"))
        r = build_watchlist_coverage(wl, dec, now=NOW)
        assert r["verdict"] == "CONCENTRATED"
        assert 0.80 <= r["top_3_share_24h"] < 1.0

    def test_concentrated_silenced_when_few_mentions(self):
        # 3 mentions across 3 tickers — top-3 share = 100% but the
        # absolute volume is too low to call it concentration.
        wl = [f"T{i}" for i in range(4)]
        dec = [
            _dec(NOW - timedelta(hours=1), "BUY T0 → FILLED"),
            _dec(NOW - timedelta(hours=2), "BUY T1 → FILLED"),
            _dec(NOW - timedelta(hours=3), "BUY T2 → FILLED"),
        ]
        r = build_watchlist_coverage(wl, dec, now=NOW)
        # T3 is never-seen ⇒ 25% stale, below STAGNANT_SHARE ⇒ DIVERSIFIED.
        assert r["verdict"] == "DIVERSIFIED"

    def test_diversified_baseline(self):
        wl = [f"T{i}" for i in range(5)]
        dec = [_dec(NOW - timedelta(hours=i + 1),
                    f"BUY T{i} → FILLED") for i in range(5)]
        r = build_watchlist_coverage(wl, dec, now=NOW)
        assert r["verdict"] == "DIVERSIFIED"
        assert r["n_active_24h"] == 5

    def test_threshold_constants_drive_boundary(self):
        # STAGNANT_SHARE boundary: 5 of 10 stale = 50% (NOT > 50%) ⇒
        # NOT STAGNANT. 6 of 10 = 60% ⇒ STAGNANT.
        wl_10 = [f"T{i}" for i in range(10)]
        # 5 active, 5 never-seen.
        dec_50 = [_dec(NOW - timedelta(hours=1), f"BUY T{i} → FILLED")
                  for i in range(5)]
        r_50 = build_watchlist_coverage(wl_10, dec_50, now=NOW)
        assert r_50["verdict"] != "STAGNANT"
        # 4 active, 6 never-seen.
        dec_60 = [_dec(NOW - timedelta(hours=1), f"BUY T{i} → FILLED")
                  for i in range(4)]
        r_60 = build_watchlist_coverage(wl_10, dec_60, now=NOW)
        assert r_60["verdict"] == "STAGNANT"


class TestSortOrder:
    def test_never_seen_floats_to_top(self):
        wl = ["A", "B", "C"]
        dec = [
            _dec(NOW - timedelta(hours=1), "BUY A → FILLED"),
            _dec(NOW - timedelta(hours=5), "HOLD B → HOLD"),
        ]
        r = build_watchlist_coverage(wl, dec, now=NOW)
        # C is never-seen ⇒ sorted first; then B (older) before A
        # (newer) by hours-since DESC.
        order = [r["ticker"] for r in r["by_ticker"]]
        assert order[0] == "C"
        assert order.index("B") < order.index("A")


class TestThresholdContract:
    def test_thresholds_in_payload(self):
        r = build_watchlist_coverage(["NVDA"],
                                     [_dec(NOW, "BUY NVDA → FILLED")],
                                     now=NOW)
        t = r["thresholds"]
        assert t["stagnant_share"] == STAGNANT_SHARE
        assert t["concentrated_top3_share"] == CONCENTRATED_TOP3_SHARE
        assert t["recent_hours"] == RECENT_HOURS
        assert t["stale_hours"] == STALE_HOURS


class TestLiveWatchlistSmoke:
    def test_against_live_watchlist_and_db(self):
        from pathlib import Path
        import sqlite3
        db = Path(__file__).resolve().parent.parent / "data" / "paper_trader.db"
        if not db.exists():
            return
        from paper_trader.strategy import WATCHLIST
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        decisions = [dict(r) for r in conn.execute(
            "SELECT timestamp, action_taken, reasoning FROM decisions "
            "ORDER BY id DESC LIMIT 1000"
        ).fetchall()]
        conn.close()
        r = build_watchlist_coverage(WATCHLIST, decisions)
        # Contract assertions: never raises, n_watchlist matches input,
        # a verdict was produced.
        assert r["n_watchlist"] == len(WATCHLIST)
        assert r["verdict"] in (
            "STAGNANT", "CONCENTRATED", "DIVERSIFIED", "NO_DATA")
        assert len(r["by_ticker"]) == len(WATCHLIST)
