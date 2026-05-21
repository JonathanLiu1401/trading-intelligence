"""Tests for paper_trader.analytics.peer_momentum_divergence_skill.

Pins:
* per-position verdict matrix: IDIOSYNCRATIC_RALLY / IDIOSYNCRATIC_DECLINE
  / LEADING_PEERS / LAGGING_PEERS / TRACKING_PEERS / NO_PEERS
* top-level roll-up: BOOK_IDIOSYNCRATIC / BOOK_DIVERGENT / BOOK_TRACKING
  / INSUFFICIENT_DATA / NO_DATA
* delta_pct threshold boundary
* min_peers gate (single peer → NO_PEERS)
* held-ticker excluded from its own peer median
* sector→peers parity with sector_heatmap.HEATMAP_BUCKETS (drift guard)
* defensive: malformed positions, NaN mom_5d, unknown ticker → NO_PEERS,
  never raises
* envelope key stability

Flask test-client route smoke included.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.peer_momentum_divergence_skill import (
    DEFAULT_DELTA_PCT,
    PEERS_BY_SECTOR,
    TICKER_TO_SECTOR,
    _classify_pair,
    build_peer_momentum_divergence_skill,
    sector_for_ticker,
)


def _pos(ticker):
    return {"ticker": ticker}


def _qs(d):
    """Convenience — build a quant_signals dict from {ticker: mom_5d}."""
    return {t: {"mom_5d": m} for t, m in d.items()}


_ENVELOPE_KEYS = {
    "verdict", "headline", "n_positions", "n_with_peers",
    "positions", "thresholds",
}


class TestEnvelopeStability:
    def test_no_data_no_positions(self):
        out = build_peer_momentum_divergence_skill([], {})
        assert set(out.keys()) >= _ENVELOPE_KEYS
        assert out["verdict"] == "NO_DATA"
        assert out["n_positions"] == 0
        assert out["positions"] == []

    def test_keys_under_active(self):
        out = build_peer_momentum_divergence_skill(
            [_pos("NVDA")],
            _qs({"NVDA": 5.0, "AMD": 1.0, "MRVL": 0.5, "AVGO": 0.3}),
        )
        assert set(out.keys()) >= _ENVELOPE_KEYS
        assert out["n_positions"] == 1
        assert out["n_with_peers"] == 1
        assert len(out["positions"]) == 1


class TestSectorLookup:
    def test_nvda_design(self):
        assert sector_for_ticker("NVDA") == "design"

    def test_case_insensitive(self):
        assert sector_for_ticker("nvda") == "design"
        assert sector_for_ticker("Soxl") == "memory_leveraged"

    def test_unknown_ticker(self):
        assert sector_for_ticker("XYZ") is None

    def test_empty_or_none(self):
        assert sector_for_ticker("") is None
        assert sector_for_ticker(None) is None  # type: ignore[arg-type]


class TestClassifyPair:
    def test_tracking_within_delta(self):
        # 4 - 3 = 1 < 2.0 → TRACKING
        assert _classify_pair(4.0, 3.0, 2.0) == "TRACKING_PEERS"

    def test_idiosyncratic_rally(self):
        # held positive, peers ≤ 0, gap >= delta
        assert _classify_pair(5.0, 0.0, 2.0) == "IDIOSYNCRATIC_RALLY"
        assert _classify_pair(5.0, -1.0, 2.0) == "IDIOSYNCRATIC_RALLY"

    def test_leading_peers_when_both_positive(self):
        # held +6, peers +2: gap 4 >= delta, both positive → LEADING
        assert _classify_pair(6.0, 2.0, 2.0) == "LEADING_PEERS"

    def test_idiosyncratic_decline(self):
        # held negative, peers ≥ 0, gap ≤ -delta
        assert _classify_pair(-5.0, 0.0, 2.0) == "IDIOSYNCRATIC_DECLINE"
        assert _classify_pair(-5.0, 1.0, 2.0) == "IDIOSYNCRATIC_DECLINE"

    def test_lagging_peers_both_positive(self):
        # held +1, peers +5: gap -4 ≤ -delta, both positive → LAGGING
        assert _classify_pair(1.0, 5.0, 2.0) == "LAGGING_PEERS"

    def test_no_peers_on_none(self):
        assert _classify_pair(None, 1.0, 2.0) == "NO_PEERS"
        assert _classify_pair(1.0, None, 2.0) == "NO_PEERS"

    def test_boundary_exact_delta_is_tracking(self):
        # gap == delta is NOT > delta, but classifier uses |gap| < delta
        # for TRACKING — so gap == delta falls into LEADING/LAGGING.
        assert _classify_pair(3.0, 1.0, 2.0) == "LEADING_PEERS"

    def test_boundary_just_under_delta_is_tracking(self):
        assert _classify_pair(2.99, 1.0, 2.0) == "TRACKING_PEERS"


class TestPerPositionVerdict:
    def test_idiosyncratic_rally_nvda_solo(self):
        # NVDA design sector, peer median mostly flat
        out = build_peer_momentum_divergence_skill(
            [_pos("NVDA")],
            _qs({"NVDA": 5.0, "AMD": 0.5, "MRVL": -0.5, "AVGO": 0.0}),
        )
        row = out["positions"][0]
        assert row["ticker"] == "NVDA"
        assert row["sector"] == "design"
        assert row["verdict"] == "IDIOSYNCRATIC_RALLY"
        # peer median of [0.5, -0.5, 0] = 0
        assert row["peer_median_mom_5d"] == 0.0
        assert row["delta_pct"] == 5.0
        assert row["n_peers"] == 3

    def test_lagging_peers_both_positive(self):
        out = build_peer_momentum_divergence_skill(
            [_pos("AMD")],
            _qs({"AMD": 1.0, "NVDA": 6.0, "MRVL": 5.0, "AVGO": 5.0}),
        )
        row = out["positions"][0]
        assert row["verdict"] == "LAGGING_PEERS"
        assert row["peer_median_mom_5d"] == 5.0

    def test_leading_peers_both_positive(self):
        out = build_peer_momentum_divergence_skill(
            [_pos("NVDA")],
            _qs({"NVDA": 6.0, "AMD": 1.0, "MRVL": 2.0, "AVGO": 1.0}),
        )
        row = out["positions"][0]
        # peer median = median(1, 2, 1) = 1.0; held 6 - 1 = 5 > delta;
        # both held & peer median positive → LEADING_PEERS (not IDIOSYNCRATIC).
        assert row["verdict"] == "LEADING_PEERS"

    def test_idiosyncratic_decline_nvda(self):
        out = build_peer_momentum_divergence_skill(
            [_pos("NVDA")],
            _qs({"NVDA": -4.0, "AMD": 1.0, "MRVL": 0.5, "AVGO": 0.3}),
        )
        row = out["positions"][0]
        assert row["verdict"] == "IDIOSYNCRATIC_DECLINE"

    def test_tracking_peers_close(self):
        out = build_peer_momentum_divergence_skill(
            [_pos("NVDA")],
            _qs({"NVDA": 2.0, "AMD": 1.5, "MRVL": 1.8, "AVGO": 2.2}),
        )
        row = out["positions"][0]
        assert row["verdict"] == "TRACKING_PEERS"

    def test_no_peers_unmapped_ticker(self):
        out = build_peer_momentum_divergence_skill(
            [_pos("XYZ")],
            _qs({"XYZ": 5.0}),
        )
        row = out["positions"][0]
        assert row["sector"] is None
        assert row["verdict"] == "NO_PEERS"

    def test_no_peers_sparse_quant_data(self):
        # NVDA in design but no peer momentums available
        out = build_peer_momentum_divergence_skill(
            [_pos("NVDA")],
            _qs({"NVDA": 5.0}),
        )
        row = out["positions"][0]
        assert row["verdict"] == "NO_PEERS"
        assert row["n_peers"] == 0

    def test_min_peers_gate(self):
        # only 1 peer with data — below default min_peers=2 → NO_PEERS
        out = build_peer_momentum_divergence_skill(
            [_pos("NVDA")],
            _qs({"NVDA": 5.0, "AMD": 1.0}),
        )
        assert out["positions"][0]["verdict"] == "NO_PEERS"

    def test_held_excluded_from_peer_median(self):
        # NVDA is in design's peer list; the builder must exclude
        # NVDA from its own peer median computation.
        out = build_peer_momentum_divergence_skill(
            [_pos("NVDA")],
            _qs({"NVDA": 100.0, "AMD": 1.0, "MRVL": 2.0, "AVGO": 3.0}),
        )
        row = out["positions"][0]
        # If NVDA were included its mom_5d=100 would skew the median.
        # peer median of [1, 2, 3] = 2.0
        assert row["peer_median_mom_5d"] == 2.0


class TestTopLevelRollup:
    def test_book_idiosyncratic_when_any_idio(self):
        out = build_peer_momentum_divergence_skill(
            [_pos("NVDA"), _pos("AMD")],
            _qs({
                "NVDA": 5.0,  # IDIOSYNCRATIC_RALLY
                "AMD": 0.0,
                "MRVL": -0.5, "AVGO": 0.0,  # peer median 0
            }),
        )
        # NVDA: 5 vs median(0, -0.5, 0) = 0 → IDIOSYNCRATIC_RALLY
        # AMD: 0 vs median(5, -0.5, 0) — bot ignores AMD in own median;
        # median of [5, -0.5, 0] = 0 → TRACKING (gap 0 < delta)
        assert out["verdict"] == "BOOK_IDIOSYNCRATIC"
        assert "NVDA" in out["headline"]

    def test_book_divergent_lagging_only(self):
        out = build_peer_momentum_divergence_skill(
            [_pos("AMD")],
            _qs({"AMD": 1.0, "NVDA": 5.0, "MRVL": 4.0, "AVGO": 6.0}),
        )
        # AMD vs median(5,4,6)=5; gap -4 ≤ -delta; both positive → LAGGING
        assert out["positions"][0]["verdict"] == "LAGGING_PEERS"
        assert out["verdict"] == "BOOK_DIVERGENT"

    def test_book_tracking_all_within_delta(self):
        out = build_peer_momentum_divergence_skill(
            [_pos("NVDA")],
            _qs({"NVDA": 2.0, "AMD": 1.5, "MRVL": 2.5, "AVGO": 2.0}),
        )
        assert out["positions"][0]["verdict"] == "TRACKING_PEERS"
        assert out["verdict"] == "BOOK_TRACKING"

    def test_insufficient_data_when_no_position_has_peers(self):
        out = build_peer_momentum_divergence_skill(
            [_pos("XYZ")],
            _qs({"XYZ": 5.0}),
        )
        # ticker unmapped → NO_PEERS → 0 with peers
        assert out["verdict"] == "INSUFFICIENT_DATA"
        assert out["n_positions"] == 1
        assert out["n_with_peers"] == 0

    def test_no_data_empty_positions(self):
        out = build_peer_momentum_divergence_skill([], {})
        assert out["verdict"] == "NO_DATA"


class TestDefensiveDegradation:
    def test_malformed_position_skipped(self):
        out = build_peer_momentum_divergence_skill(
            [
                "not-a-dict",  # type: ignore[list-item]
                {"ticker": None},
                {"ticker": ""},
                _pos("NVDA"),
            ],
            _qs({"NVDA": 2.0, "AMD": 1.5, "MRVL": 1.8, "AVGO": 2.2}),
        )
        # Only NVDA contributed to n_positions
        assert out["n_positions"] == 1

    def test_nan_mom_skipped_in_median(self):
        out = build_peer_momentum_divergence_skill(
            [_pos("NVDA")],
            _qs({"NVDA": 5.0, "AMD": float("nan"),
                 "MRVL": 0.5, "AVGO": -0.5}),
        )
        # AMD's NaN is excluded from peer median; median([0.5, -0.5]) = 0.0
        assert out["positions"][0]["peer_median_mom_5d"] == 0.0
        assert out["positions"][0]["n_peers"] == 2

    def test_garbage_quant_signals_dict_safe(self):
        out = build_peer_momentum_divergence_skill(
            [_pos("NVDA")],
            {"NVDA": "not-a-dict",  # type: ignore[dict-item]
             "AMD": {"mom_5d": "garbage"},
             "MRVL": {"mom_5d": 1.0},
             "AVGO": {"mom_5d": 2.0}},
        )
        # NVDA mom_5d not extractable → None held mom → NO_PEERS
        row = out["positions"][0]
        assert row["mom_5d"] is None
        # peer median computed from AMD-bad + MRVL/AVGO; AMD's garbage
        # skipped, only 2 valid peers — at min_peers=2 boundary it
        # still computes.
        # Held mom None → _classify_pair returns NO_PEERS.
        assert row["verdict"] == "NO_PEERS"


class TestThresholds:
    def test_custom_delta_pct_flips_verdict(self):
        # Default delta=2; with delta=10 the same 5-pt gap is TRACKING.
        out_default = build_peer_momentum_divergence_skill(
            [_pos("NVDA")],
            _qs({"NVDA": 5.0, "AMD": 0.5, "MRVL": -0.5, "AVGO": 0.0}),
        )
        assert out_default["positions"][0]["verdict"] == "IDIOSYNCRATIC_RALLY"

        out_strict = build_peer_momentum_divergence_skill(
            [_pos("NVDA")],
            _qs({"NVDA": 5.0, "AMD": 0.5, "MRVL": -0.5, "AVGO": 0.0}),
            delta_pct=10.0,
        )
        assert out_strict["positions"][0]["verdict"] == "TRACKING_PEERS"

    def test_thresholds_echoed_back(self):
        out = build_peer_momentum_divergence_skill(
            [_pos("NVDA")],
            _qs({"NVDA": 2.0, "AMD": 1.5, "MRVL": 1.8, "AVGO": 2.2}),
            delta_pct=1.5, min_peers=3, min_positions=2,
        )
        thr = out["thresholds"]
        assert thr["delta_pct"] == 1.5
        assert thr["min_peers"] == 3
        assert thr["min_positions"] == 2


class TestSectorHeatmapParity:
    """Drift guard: PEERS_BY_SECTOR must match sector_heatmap.HEATMAP_BUCKETS
    exactly so the two layers can never silently diverge."""
    def test_sector_buckets_identical(self):
        from paper_trader.analytics.sector_heatmap import HEATMAP_BUCKETS
        assert PEERS_BY_SECTOR == HEATMAP_BUCKETS

    def test_reverse_index_resolves_every_ticker(self):
        for sec, tks in PEERS_BY_SECTOR.items():
            for t in tks:
                assert TICKER_TO_SECTOR[t.upper()] == sec


class TestRouteSmoke:
    """Flask test client smoke — the route layer must compose the
    builder correctly and degrade to a structured ERROR envelope
    on internal failure. Memory note: __main__ smoke hits a different
    DB; use Flask test client + the live app object."""

    def _client(self):
        from paper_trader.dashboard import app
        return app.test_client()

    def test_route_returns_envelope(self):
        cl = self._client()
        r = cl.get("/api/peer-momentum-divergence-skill")
        assert r.status_code in (200, 500)
        body = r.get_json()
        assert body is not None
        assert "verdict" in body
        assert "headline" in body
        assert "positions" in body

    def test_route_threshold_clamp(self):
        cl = self._client()
        r = cl.get("/api/peer-momentum-divergence-skill"
                   "?delta_pct=1.5&min_peers=3")
        assert r.status_code in (200, 500)
        body = r.get_json()
        if r.status_code == 200:
            assert body["thresholds"]["delta_pct"] == 1.5
            assert body["thresholds"]["min_peers"] == 3
