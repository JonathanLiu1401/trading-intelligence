"""Tests for analytics/track_record.py + its wiring into the decision prompt.

``build_track_record`` is the per-name closed-trade memory injected into the
live decision prompt (the ``self_review`` precedent — observational only,
never gates Opus). It is a *composition* layered on the single source of truth
(``build_round_trips`` via ``loser_autopsy``/``winner_autopsy`` — AGENTS.md
invariant #10): a re-derived P&L, a per-name net that disagrees with
``build_round_trips``, a display-cap that hides the true net, a names-filter
that drifts from the quant block's "names in play", an entry/exit reason that
is not surfaced verbatim, or a prescriptive (non-observational) prompt block
all fail an assertion here. Hand-computed arithmetic.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader import strategy
from paper_trader.analytics import track_record as tr_mod
from paper_trader.analytics.round_trips import build_round_trips
from paper_trader.analytics.track_record import (
    PER_NAME_CAP,
    REASON_CAP,
    build_track_record,
)

_BASE = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
NOW = datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc)


def _day(offset: int) -> str:
    return (_BASE + timedelta(days=offset)).isoformat()


def _rt(tid, ticker, buy_day, sell_day, qty, buy_px, sell_px,
        entry_reason="", exit_reason=""):
    return [
        {"id": tid, "timestamp": _day(buy_day), "ticker": ticker,
         "action": "BUY", "qty": qty, "price": buy_px,
         "value": qty * buy_px, "strike": None, "expiry": None,
         "option_type": None, "reason": entry_reason},
        {"id": tid + 1, "timestamp": _day(sell_day), "ticker": ticker,
         "action": "SELL", "qty": qty, "price": sell_px,
         "value": qty * sell_px, "strike": None, "expiry": None,
         "option_type": None, "reason": exit_reason},
    ]


def _ledger(specs):
    """specs: (ticker, buy_px, sell_px, hold_days, entry_reason, exit_reason).
    Each becomes its own disjoint closed round-trip (qty 10), sequential in
    time so a repeated ticker yields multiple independent round-trips.
    """
    trades, tid, day = [], 1, 0
    for ticker, bpx, spx, hold, er, xr in specs:
        trades += _rt(tid, ticker, day, day + hold, 10, bpx, spx, er, xr)
        tid += 2
        day += hold + 1
    return trades


# A controlled book:
#   NVDA  -$200 (-20%, 2d)   → LOSS  KNIFE_CATCH   net -200
#   MU    loss -$10 (-2%,2d) STOPPED_OUT  +  win +$80 (+20%,6d) HOME_RUN
#                                                          → net +70 (1W-1L)
#   AMD   3 losses (-10 STOPPED_OUT, -20 KNIFE_CATCH, -5 SLOW_BLEED)
#                                                          → net -35 (0W-3L)
#   TSLA  +$20 (+20%, 3d)    → WIN  HOME_RUN          net +20
# worst-net-first ⇒ NVDA(-200), AMD(-35), TSLA(+20), MU(+70)
_SPECS = [
    ("NVDA", 100.0, 80.0, 2, "AI capex supercycle thesis", "stopped out hard"),
    ("MU", 50.0, 49.0, 2, "DRAM bottoming", "small trim"),
    ("MU", 40.0, 48.0, 6, "HBM demand inflection", "let it run, target hit"),
    ("AMD", 10.0, 9.0, 1, "MI300 ramp", "noise"),
    ("AMD", 10.0, 8.0, 7, "datacenter share gains", "thesis broke"),
    ("AMD", 10.0, 9.5, 6, "rebound bet", "slow bleed exit"),
    ("TSLA", 10.0, 12.0, 3, "delivery beat", "took the win"),
]


def _build():
    return build_track_record(_ledger(_SPECS), now=NOW)


# ───────────────────── composition / per-name net ──────────────────────

class TestComposition:
    def test_state_and_round_trip_count(self):
        r = _build()
        assert r["state"] == "OK"
        assert r["n_round_trips"] == 7  # 1 NVDA + 2 MU + 3 AMD + 1 TSLA
        assert r["filtered"] is False
        assert r["as_of"] == NOW.isoformat(timespec="seconds")

    def test_per_name_net_is_single_source_of_truth(self):
        """net_usd per ticker == sum of build_round_trips pnl_usd for that
        ticker — never re-derived (invariant #10)."""
        ledger = _ledger(_SPECS)
        rts = build_round_trips(ledger)
        truth: dict[str, float] = {}
        for rt in rts:
            truth[rt["ticker"]] = round(
                truth.get(rt["ticker"], 0.0) + (rt.get("pnl_usd") or 0.0), 2)
        got = {e["ticker"]: e["net_usd"] for e in _build()["names"]}
        assert got == truth
        # explicit hand-computed values
        assert got["NVDA"] == -200.0
        assert got["MU"] == 70.0
        assert got["AMD"] == -35.0
        assert got["TSLA"] == 20.0

    def test_win_loss_counts_and_modes_exact(self):
        names = {e["ticker"]: e for e in _build()["names"]}
        mu = names["MU"]
        assert (mu["n_win"], mu["n_loss"], mu["n_closed"]) == (1, 1, 2)
        # newest closed first → the +$80 HOME_RUN win is recent[0]
        assert mu["recent"][0]["outcome"] == "WIN"
        assert mu["recent"][0]["mode"] == "HOME_RUN"
        assert mu["recent"][0]["pnl_usd"] == 80.0
        assert mu["recent"][1]["outcome"] == "LOSS"
        assert mu["recent"][1]["mode"] == "STOPPED_OUT"
        nvda = names["NVDA"]
        assert (nvda["n_win"], nvda["n_loss"]) == (0, 1)
        assert nvda["recent"][0]["mode"] == "KNIFE_CATCH"

    def test_verbatim_reasons_preserved_in_structured_output(self):
        mu = {e["ticker"]: e for e in _build()["names"]}["MU"]
        win = mu["recent"][0]
        # surfaced exactly as written — never NLP-parsed, never truncated in
        # the structured payload
        assert win["entry_reason"] == "HBM demand inflection"
        assert win["exit_reason"] == "let it run, target hit"


# ─────────────────────── ordering / display cap ────────────────────────

class TestOrderingAndCap:
    def test_worst_net_first(self):
        order = [e["ticker"] for e in _build()["names"]]
        assert order == ["NVDA", "AMD", "TSLA", "MU"]

    def test_cap_is_display_only_net_is_over_all(self):
        amd = {e["ticker"]: e for e in _build()["names"]}["AMD"]
        assert amd["n_closed"] == 3
        assert len(amd["recent"]) == PER_NAME_CAP == 2
        # net is over ALL 3 closed round-trips, not just the 2 narrated
        assert amd["net_usd"] == -35.0
        # newest two: SLOW_BLEED (-5, last window) then KNIFE_CATCH (-20)
        assert [t["mode"] for t in amd["recent"]] == ["SLOW_BLEED",
                                                      "KNIFE_CATCH"]

    def test_explicit_cap_argument_respected(self):
        r = build_track_record(_ledger(_SPECS), per_name_cap=1, now=NOW)
        amd = {e["ticker"]: e for e in r["names"]}["AMD"]
        assert len(amd["recent"]) == 1
        assert amd["recent"][0]["mode"] == "SLOW_BLEED"  # still the newest
        assert amd["net_usd"] == -35.0  # net unchanged by the cap


# ───────────────────────── names filter ────────────────────────────────

class TestNamesFilter:
    def test_filter_restricts_to_named_set(self):
        r = build_track_record(_ledger(_SPECS), names={"NVDA", "TSLA"},
                               now=NOW)
        assert r["filtered"] is True
        assert sorted(e["ticker"] for e in r["names"]) == ["NVDA", "TSLA"]
        assert "MU" not in r["prompt_block"]
        assert "NVDA" in r["prompt_block"]

    def test_filter_with_no_matches_yields_no_block_but_state_ok(self):
        r = build_track_record(_ledger(_SPECS), names={"AAPL"}, now=NOW)
        assert r["state"] == "OK"          # closed trips exist book-wide
        assert r["names"] == []
        assert r["prompt_block"] is None    # nothing relevant this cycle
        assert r["summary"] == "no-history"

    def test_open_position_without_close_is_absent(self):
        # NVDA opened but never closed → no round-trip → not in track record
        ledger = [
            {"id": 1, "timestamp": _day(0), "ticker": "NVDA", "action": "BUY",
             "qty": 10, "price": 100.0, "value": 1000.0, "strike": None,
             "expiry": None, "option_type": None, "reason": "still open"},
        ]
        r = build_track_record(ledger, now=NOW)
        assert r["state"] == "NO_DATA"
        assert r["names"] == []
        assert r["prompt_block"] is None

    def test_wash_round_trip_excluded(self):
        # buy == sell ⇒ pnl 0 ⇒ a wash, not a win and not a loss (the strict
        # round_trips/#10 convention loser_autopsy & winner_autopsy share)
        r = build_track_record(
            _ledger([("AAPL", 10.0, 10.0, 1, "flat", "flat")]), now=NOW)
        assert r["n_round_trips"] == 1
        assert r["names"] == []  # the wash produced no win/loss card


# ─────────────────────── NO_DATA / summary ─────────────────────────────

class TestNoData:
    def test_empty_ledger(self):
        r = build_track_record([], now=NOW)
        assert r["state"] == "NO_DATA"
        assert r["names"] == []
        assert r["prompt_block"] is None
        assert r["summary"] == "no-closed-round-trips"

    def test_summary_names_worst(self):
        assert _build()["summary"] == (
            "4 name(s); worst NVDA $-200.00 (0W-1L)")


# ─────────────────── prompt block: observational only ───────────────────

class TestPromptBlock:
    def test_block_is_observational_not_prescriptive(self):
        block = _build()["prompt_block"]
        assert block is not None
        low = block.lower()
        # the self_review observational contract: states facts + reaffirms
        # autonomy, issues no directives/limits
        assert "not directives or limits" in low
        assert "complete autonomy" in low
        for directive in ("you should", "you must", "reduce your",
                          "stop trading", "do not buy", "cut your"):
            assert directive not in low

    def test_block_carries_verbatim_reason_and_facts(self):
        block = _build()["prompt_block"]
        assert "AI capex supercycle thesis" in block   # verbatim entry reason
        assert "stopped out hard" in block              # verbatim exit reason
        assert "NVDA  0W-1L  net $-200.00  (1 closed)" in block
        assert "KNIFE_CATCH" in block
        # worst-net-first also governs the rendered block
        assert block.index("NVDA") < block.index("MU")

    def test_long_reason_truncated_in_block_but_verbatim_in_struct(self):
        long_reason = "X" * (REASON_CAP + 80)
        r = build_track_record(
            _ledger([("NVDA", 100.0, 80.0, 2, long_reason, "exit")]),
            now=NOW)
        # structured payload keeps it byte-for-byte verbatim
        assert r["names"][0]["recent"][0]["entry_reason"] == long_reason
        # the lean prompt block truncates with an ellipsis
        assert long_reason not in r["prompt_block"]
        assert "…" in r["prompt_block"]
        assert ("X" * (REASON_CAP - 1)) in r["prompt_block"]


# ───────────────────────── never raises ────────────────────────────────

class TestNeverRaises:
    def test_failing_loser_builder_degrades_not_raises(self, monkeypatch):
        def _boom(*a, **k):
            raise RuntimeError("simulated autopsy fault")
        monkeypatch.setattr(tr_mod, "build_loser_autopsy", _boom)
        r = build_track_record(_ledger(_SPECS), now=NOW)
        # winner side still produces names; no exception
        assert isinstance(r, dict)
        assert any(e["ticker"] == "MU" for e in r["names"])  # had a win
        assert all(e["n_loss"] == 0 for e in r["names"])     # losers lost

    def test_garbage_rows_do_not_raise(self):
        r = build_track_record(
            [{"nonsense": 1}, None, {"id": "x", "action": "BUY"}], now=NOW)
        assert isinstance(r, dict)
        assert r["state"] in ("NO_DATA", "OK")


# ───────────────── strategy wiring (single source of truth) ─────────────

class TestStrategyWiring:
    def test_names_in_play_is_held_mentioned_priority(self):
        positions = [{"ticker": "HELD1"}, {"ticker": "HELD2"}]
        signals = [{"tickers": ["SIGA", "SIGB"]}, {"tickers": ["SIGC"]}]
        watch = ["W1", "W2", "W3", "W4", "W5", "W6", "W7"]
        got = strategy._names_in_play(positions, signals, watch)
        assert got == {"HELD1", "HELD2", "SIGA", "SIGB", "SIGC",
                       "W1", "W2", "W3", "W4", "W5"}  # only top-5 watch

    def test_names_in_play_caps_signals_at_top_10(self):
        signals = [{"tickers": [f"S{i}"]} for i in range(12)]
        got = strategy._names_in_play([], signals, [])
        # only the first 10 signals contribute
        assert "S9" in got and "S10" not in got and "S11" not in got

    def test_block_injected_after_self_review_before_watchlist(self):
        snap = {"cash": 0.0, "open_value": 0.0, "total_value": 0.0,
                "positions": []}
        out = strategy._build_payload(
            snap, [], [], {}, {}, None, False,
            self_review_block="SR_SENTINEL",
            track_record_block="TR_SENTINEL")
        assert "TR_SENTINEL" in out
        assert out.index("SR_SENTINEL") < out.index("TR_SENTINEL")
        assert out.index("TR_SENTINEL") < out.index("WATCHLIST PRICES")

    def test_none_block_is_backward_compatible(self):
        snap = {"cash": 0.0, "open_value": 0.0, "total_value": 0.0,
                "positions": []}
        out = strategy._build_payload(snap, [], [], {}, {}, None, False)
        assert "TRACK RECORD" not in out
        assert "WATCHLIST PRICES" in out  # still a valid payload
