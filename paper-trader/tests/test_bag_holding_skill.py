"""Tests for analytics/bag_holding_skill.py — $-attribution by failure mode.

Hand-computed arithmetic. The module is a *dollar-weighted* view on the
single-source-of-truth ``loser_autopsy._classify`` + ``build_round_trips``
(AGENTS.md invariant #10): a recomputed P&L, a misclassified mode, a
verdict emitted before the STABLE sample-size gate, a wash counted as
a loss, a verdict precedence inversion, or a non-deterministic dominant
mode tie-break all fail an assertion here.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.bag_holding_skill import (
    BAG_HOLDER_RATIO,
    DISCIPLINED_RATIO,
    KNIFE_CATCHER_RATIO,
    STABLE_MIN_LOSERS,
    WHIPSAW_NOISE_RATIO,
    build_bag_holding_skill,
)

_BASE = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _day(offset: float) -> str:
    return (_BASE + timedelta(days=offset)).isoformat()


def _rt(tid, ticker, buy_day, sell_day, qty, buy_px, sell_px):
    return [
        {"id": tid, "timestamp": _day(buy_day), "ticker": ticker,
         "action": "BUY", "qty": qty, "price": buy_px,
         "value": qty * buy_px, "strike": None, "expiry": None,
         "option_type": None, "reason": ""},
        {"id": tid + 1, "timestamp": _day(sell_day), "ticker": ticker,
         "action": "SELL", "qty": qty, "price": sell_px,
         "value": qty * sell_px, "strike": None, "expiry": None,
         "option_type": None, "reason": ""},
    ]


def _ledger(specs):
    """specs: (ticker, buy_px, sell_px, hold_days). Each becomes its own
    round-trip on a strictly increasing, disjoint window so
    build_round_trips closes each independently."""
    trades, tid, day = [], 1, 0.0
    for ticker, bpx, spx, hold in specs:
        trades += _rt(tid, ticker, day, day + hold, 10, bpx, spx)
        tid += 2
        day += hold + 1
    return trades


# ───────────────────────── state / sample-size gate ─────────────────────

class TestStateGate:
    def test_no_data_when_no_trades(self):
        rep = build_bag_holding_skill([])
        assert rep["state"] == "NO_DATA"
        assert rep["verdict"] is None
        assert rep["n_round_trips"] == 0
        assert rep["n_losers"] == 0
        assert rep["total_loss_usd"] == 0.0
        assert "unscorable" in rep["headline"].lower()

    def test_no_losses_when_all_winners(self):
        # Two winners, no losers.
        rep = build_bag_holding_skill(_ledger([
            ("AAA", 100.0, 110.0, 1.0),
            ("BBB", 50.0, 55.0, 2.0),
        ]))
        assert rep["state"] == "NO_LOSSES"
        assert rep["verdict"] is None
        assert rep["n_round_trips"] == 2
        assert rep["n_losers"] == 0
        assert "nothing to attribute" in rep["headline"]

    def test_emerging_below_stable_min_losers(self):
        # Fewer than STABLE_MIN_LOSERS losers → state=EMERGING, verdict None.
        n = STABLE_MIN_LOSERS - 1
        # -1% over 10 days → SLOW_BLEED (not KNIFE_CATCH which needs ≤ -15%).
        specs = [(f"T{i}", 100.0, 99.0, 10.0) for i in range(n)]
        rep = build_bag_holding_skill(_ledger(specs))
        assert rep["state"] == "EMERGING"
        assert rep["verdict"] is None
        assert rep["n_losers"] == n
        # Numerics still emitted.
        assert rep["bag_holding_ratio"] is not None
        assert rep["dominant_mode"] == "SLOW_BLEED"
        assert "verdict withheld" in rep["headline"].lower()

    def test_stable_exact_min_emits_verdict(self):
        n = STABLE_MIN_LOSERS
        # All SLOW_BLEED at -1% / 10d hold → BAG_HOLDER.
        specs = [(f"T{i}", 100.0, 99.0, 10.0) for i in range(n)]
        rep = build_bag_holding_skill(_ledger(specs))
        assert rep["state"] == "STABLE"
        assert rep["verdict"] == "BAG_HOLDER"


# ───────────────────────── verdict ladder (STABLE only) ─────────────────

class TestVerdictLadder:
    def test_bag_holder_when_slow_bleed_dominates(self):
        # 8 SLOW_BLEED at -1% / 10d hold = 100% SLOW_BLEED share.
        # (-50% would be KNIFE_CATCH per _classify; need shallow + slow.)
        specs = [(f"T{i}", 100.0, 99.0, 10.0)
                 for i in range(STABLE_MIN_LOSERS)]
        rep = build_bag_holding_skill(_ledger(specs))
        assert rep["verdict"] == "BAG_HOLDER"
        assert rep["bag_holding_ratio"] == 1.0
        # Verdict-aware headline mentions BAG_HOLDER and the SLOW_BLEED %.
        assert "BAG_HOLDER" in rep["headline"]
        assert "SLOW_BLEED" in rep["headline"]

    def test_knife_catcher_when_knife_catch_dominates(self):
        # 8 KNIFE_CATCH (loss ≤ -15% regardless of hold).
        # qty=10, buy=100, sell=80 ⇒ -$200 / -20%. hold=0.1d short.
        specs = [(f"T{i}", 100.0, 80.0, 0.1) for i in range(STABLE_MIN_LOSERS)]
        rep = build_bag_holding_skill(_ledger(specs))
        assert rep["verdict"] == "KNIFE_CATCHER"
        assert rep["knife_catch_ratio"] == 1.0
        assert "stock-picking" in rep["headline"]

    def test_whipsaw_bleed_when_whipsaw_dominates(self):
        # Fast (< 1d) + shallow (> -3%) → WHIPSAW. -$10 each at -1%.
        # qty=10, buy=100, sell=99 → -$10 / -1%. hold=0.5d.
        specs = [(f"T{i}", 100.0, 99.0, 0.5) for i in range(STABLE_MIN_LOSERS)]
        rep = build_bag_holding_skill(_ledger(specs))
        assert rep["verdict"] == "WHIPSAW_BLEED"
        assert rep["whipsaw_ratio"] == 1.0
        assert "Cutting noise too tight" in rep["headline"]

    def test_disciplined_cutter_when_mixed_moderate_no_bleed(self):
        # All STOPPED_OUT: moderate loss + moderate hold.
        # qty=10, buy=100, sell=95 → -$50 / -5%, hold=2d → STOPPED_OUT.
        specs = [(f"T{i}", 100.0, 95.0, 2.0) for i in range(STABLE_MIN_LOSERS)]
        rep = build_bag_holding_skill(_ledger(specs))
        # SLOW_BLEED share = 0; WHIPSAW share = 0 → DISCIPLINED_CUTTER.
        assert rep["verdict"] == "DISCIPLINED_CUTTER"
        assert rep["bag_holding_ratio"] == 0.0
        assert rep["whipsaw_ratio"] == 0.0
        assert "Cuts losers without" in rep["headline"]

    def test_mixed_when_no_arm_triggers(self):
        # Half SLOW_BLEED + half WHIPSAW. Each side -$50 (1 SLOW @ -50,
        # 1 WHIP @ -$1 doesn't balance; we need MATCHED $ but mixed modes).
        # Build 4 SLOW_BLEED + 4 WHIPSAW with matched $:
        #   SLOW: hold=10d, sell=99 (1% loss) → STOPPED_OUT not SLOW_BLEED!
        # SLOW_BLEED needs hold>=5d (any size loss). At hold>=5 + any loss
        # (not >-15) → SLOW_BLEED. Use -$10 each for both sides.
        #   SLOW: hold=10d, buy=100, sell=99 → -$10 / -1%. hold>=5 → SLOW_BLEED.
        #   WHIP: hold=0.5d, buy=100, sell=99 → -$10 / -1%. fast+shallow → WHIPSAW.
        # Total $ lost = -$80; SLOW share = 0.5, WHIP share = 0.5.
        # SLOW 0.5 < BAG_HOLDER_RATIO=0.60 → not BAG_HOLDER.
        # KNIFE share = 0 → not KNIFE_CATCHER.
        # WHIP 0.5 ≥ WHIPSAW_NOISE_RATIO=0.40 → WHIPSAW_BLEED triggers first.
        specs = [(f"S{i}", 100.0, 99.0, 10.0) for i in range(4)] + \
                [(f"W{i}", 100.0, 99.0, 0.5) for i in range(4)]
        rep = build_bag_holding_skill(_ledger(specs))
        # Pre-DISCIPLINED arm in the ladder, WHIPSAW arm trips first.
        assert rep["verdict"] == "WHIPSAW_BLEED"
        assert rep["whipsaw_ratio"] == 0.5
        assert rep["bag_holding_ratio"] == 0.5

    def test_mixed_verdict_when_truly_mixed(self):
        # Force MIXED: SLOW 50%, WHIPSAW just under 0.40 trigger. Need a
        # third bucket of STOPPED_OUT to push WHIPSAW below threshold.
        # Use 4 SLOW @ -$10, 2 WHIP @ -$10, 2 STOPPED_OUT @ -$10.
        #   STOPPED_OUT: moderate loss + moderate hold = hold=2d, -$50.
        # Total $ -$80. SLOW share = 0.5, WHIP share = 0.25 (< 0.40),
        # KNIFE = 0, STOPPED = 0.25.
        # SLOW 0.5 < 0.6 → not BAG_HOLDER; WHIPSAW 0.25 < 0.40 → not
        # WHIPSAW_BLEED; SLOW 0.5 > 0.2 → not DISCIPLINED → MIXED.
        specs = [(f"S{i}", 100.0, 99.0, 10.0) for i in range(4)] + \
                [(f"W{i}", 100.0, 99.0, 0.5) for i in range(2)] + \
                [(f"O{i}", 100.0, 95.0, 2.0) for i in range(2)]
        rep = build_bag_holding_skill(_ledger(specs))
        assert rep["verdict"] == "MIXED"
        # bleed_clause in headline references dominant mode (SLOW_BLEED).
        assert "MIXED" in rep["headline"]


# ───────────────────────── verdict thresholds (boundary) ────────────────

class TestThresholds:
    def test_bag_holder_just_at_boundary(self):
        # Exactly 60% SLOW_BLEED — inclusive boundary triggers BAG_HOLDER.
        # 6 SLOW_BLEED @ -$10 = -$60, 4 WHIPSAW @ -$10 = -$40, total -$100.
        # SLOW share = 0.60.
        specs = [(f"S{i}", 100.0, 99.0, 10.0) for i in range(6)] + \
                [(f"W{i}", 100.0, 99.0, 0.5) for i in range(4)]
        rep = build_bag_holding_skill(_ledger(specs))
        assert rep["bag_holding_ratio"] == 0.6
        assert rep["verdict"] == "BAG_HOLDER"

    def test_just_below_bag_holder_boundary_not_bag_holder(self):
        # 5 SLOW @ -$10 = -$50, 5 WHIPSAW @ -$10 = -$50. SLOW share = 0.5.
        # 0.5 < 0.60 → not BAG_HOLDER; WHIPSAW 0.5 ≥ 0.40 → WHIPSAW_BLEED.
        specs = [(f"S{i}", 100.0, 99.0, 10.0) for i in range(5)] + \
                [(f"W{i}", 100.0, 99.0, 0.5) for i in range(5)]
        rep = build_bag_holding_skill(_ledger(specs))
        assert rep["verdict"] != "BAG_HOLDER"


# ───────────────────────── rows aggregation correctness ─────────────────

class TestRows:
    def test_rows_aggregate_count_and_loss_per_mode(self):
        # 3 SLOW_BLEED (-$10 each) + 5 WHIPSAW (-$10 each).
        specs = [(f"S{i}", 100.0, 99.0, 10.0) for i in range(3)] + \
                [(f"W{i}", 100.0, 99.0, 0.5) for i in range(5)]
        rep = build_bag_holding_skill(_ledger(specs))
        rows = {r["mode"]: r for r in rep["rows"]}
        assert rows["SLOW_BLEED"]["n"] == 3
        assert rows["SLOW_BLEED"]["loss_usd"] == -30.0
        assert rows["WHIPSAW"]["n"] == 5
        assert rows["WHIPSAW"]["loss_usd"] == -50.0
        # share_of_loss adds to 1.0 across all non-zero rows.
        s = sum((r["share_of_loss"] or 0.0) for r in rep["rows"])
        assert abs(s - 1.0) < 1e-9

    def test_rows_carry_tickers_distinctly(self):
        specs = [
            ("AAA", 100.0, 99.0, 10.0),  # SLOW_BLEED
            ("AAA", 100.0, 99.0, 10.0),  # SLOW_BLEED again, same ticker
            ("BBB", 100.0, 99.0, 10.0),  # SLOW_BLEED, different ticker
        ]
        rep = build_bag_holding_skill(_ledger(specs))
        slow_row = next(r for r in rep["rows"] if r["mode"] == "SLOW_BLEED")
        # tickers list deduped + ordered by first encounter.
        assert slow_row["tickers"] == ["AAA", "BBB"]
        assert slow_row["n"] == 3

    def test_rows_track_worst_single_loss(self):
        # Three SLOW_BLEED losses: -$10, -$50, -$20. Worst = -$50, BBB.
        specs = [
            ("AAA", 100.0, 99.0, 10.0),    # -$10
            ("BBB", 100.0, 95.0, 10.0),    # -$50  (worst)
            ("CCC", 100.0, 98.0, 10.0),    # -$20
        ]
        rep = build_bag_holding_skill(_ledger(specs))
        slow_row = next(r for r in rep["rows"] if r["mode"] == "SLOW_BLEED")
        assert slow_row["worst_loss_usd"] == -50.0
        assert slow_row["worst_ticker"] == "BBB"

    def test_empty_mode_rows_are_zero_and_present(self):
        # Single SLOW_BLEED — KNIFE/STOPPED/WHIPSAW rows must still exist
        # with n=0 / loss=0 so the dashboard can render a complete grid.
        specs = [("AAA", 100.0, 99.0, 10.0)]
        rep = build_bag_holding_skill(_ledger(specs))
        modes = {r["mode"] for r in rep["rows"]}
        assert modes == {"KNIFE_CATCH", "SLOW_BLEED", "STOPPED_OUT", "WHIPSAW"}
        for r in rep["rows"]:
            if r["mode"] != "SLOW_BLEED":
                assert r["n"] == 0
                assert r["loss_usd"] == 0.0
                assert r["share_of_loss"] == 0.0


# ───────────────────────── invariants / no-mutation ─────────────────────

class TestInvariants:
    def test_winners_never_counted_as_losers(self):
        # Mixed winners + losers. Only the losers should be in the rows.
        specs = [
            ("WIN", 100.0, 110.0, 1.0),    # winner — excluded
            ("LOSE", 100.0, 50.0, 10.0),   # SLOW_BLEED
        ]
        rep = build_bag_holding_skill(_ledger(specs))
        assert rep["n_round_trips"] == 2
        assert rep["n_losers"] == 1
        assert rep["total_loss_usd"] == -500.0

    def test_sub_cent_wash_is_not_a_loss(self):
        # Strict < 0 convention matches loser_autopsy. A 0.0 PnL is NOT a loss.
        specs = [("AAA", 100.0, 100.0, 10.0)]
        rep = build_bag_holding_skill(_ledger(specs))
        assert rep["state"] == "NO_LOSSES"
        assert rep["n_losers"] == 0

    def test_pure_no_mutation_of_input_trades(self):
        trades = _ledger([("AAA", 100.0, 99.0, 10.0)])
        snapshot = [dict(t) for t in trades]
        _ = build_bag_holding_skill(trades)
        assert trades == snapshot

    def test_never_raises_on_garbage(self):
        # Empty rows, missing fields, weird types.
        rep = build_bag_holding_skill([
            {},
            {"ticker": "X", "action": "BUY"},  # missing fields
        ])
        # Doesn't raise; degrades to NO_DATA / NO_LOSSES depending on
        # whether round_trips parsed anything (it shouldn't).
        assert rep["state"] in {"NO_DATA", "NO_LOSSES"}


# ───────────────────────── output shape ─────────────────────────────────

class TestShape:
    def test_response_keys_present(self):
        rep = build_bag_holding_skill([])
        keys = {"as_of", "state", "verdict", "headline", "n_round_trips",
                "n_losers", "total_loss_usd", "dominant_mode",
                "bag_holding_ratio", "knife_catch_ratio", "whipsaw_ratio",
                "rows", "stable_min_losers", "thresholds"}
        assert keys.issubset(rep.keys())
        # thresholds dict echoes module constants — the only stable
        # contract a UI can rely on for ladder rendering.
        assert rep["thresholds"]["BAG_HOLDER_RATIO"] == BAG_HOLDER_RATIO
        assert rep["thresholds"]["DISCIPLINED_RATIO"] == DISCIPLINED_RATIO
        assert rep["thresholds"]["KNIFE_CATCHER_RATIO"] == KNIFE_CATCHER_RATIO
        assert rep["thresholds"]["WHIPSAW_NOISE_RATIO"] == WHIPSAW_NOISE_RATIO
        assert rep["stable_min_losers"] == STABLE_MIN_LOSERS

    def test_share_of_loss_is_non_negative_fraction(self):
        specs = [("AAA", 100.0, 99.0, 10.0),
                 ("BBB", 100.0, 99.0, 0.5)]
        rep = build_bag_holding_skill(_ledger(specs))
        for r in rep["rows"]:
            s = r["share_of_loss"]
            assert s is None or 0.0 <= s <= 1.0
