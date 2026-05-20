"""Tests for analytics/correlation_cluster_warning.py — hidden-factor-bet
alarm built on top of analytics/correlation.

Single-linkage clustering, weight aggregation, and the cluster-share
verdict ladder are all locked to exact-value checks. The ``_components``
union-find is exercised against canonical graphs (disjoint singletons,
chain, fully connected).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.correlation_cluster_warning import (  # noqa: E402
    DOMINANT_WEIGHT,
    MIN_CLUSTER_SIZE,
    WATCHLIST_WEIGHT,
    _cluster_mean_corr,
    _components,
    build_correlation_cluster_warning,
)
from paper_trader.analytics.correlation import HIGH_CORR  # noqa: E402


# ───────────────────────────── _components ──────────────────────────────


class TestComponents:
    def test_disjoint_singletons(self):
        assert _components(["A", "B", "C"], []) == [["A"], ["B"], ["C"]]

    def test_one_edge_collapses_to_pair(self):
        comps = _components(["A", "B", "C"], [("A", "B")])
        # Pair sorts first (bigger), singleton second. Inside the pair the
        # tickers are alphabetically sorted.
        assert comps == [["A", "B"], ["C"]]

    def test_transitive_chain(self):
        # A-B-C-D as a chain ⇒ one 4-name cluster, not three pairs.
        comps = _components(["A", "B", "C", "D"],
                            [("A", "B"), ("B", "C"), ("C", "D")])
        assert comps == [["A", "B", "C", "D"]]

    def test_two_clusters_biggest_first(self):
        comps = _components(
            ["A", "B", "C", "D", "E"],
            [("A", "B"), ("B", "C"), ("D", "E")],
        )
        # 3-cluster sorts before 2-cluster.
        assert comps == [["A", "B", "C"], ["D", "E"]]

    def test_unknown_nodes_in_edges_are_skipped(self):
        # An edge mentioning a node we don't track must not crash or
        # invent a new component.
        comps = _components(["A", "B"], [("A", "B"), ("A", "ZZZ")])
        assert comps == [["A", "B"]]


# ──────────────────────── _cluster_mean_corr ────────────────────────────


class TestClusterMeanCorr:
    def test_average_of_in_cluster_pairs_only(self):
        pairs = [
            {"a": "A", "b": "B", "corr": 0.80},
            {"a": "A", "b": "C", "corr": 0.70},
            {"a": "B", "b": "C", "corr": 0.90},
            {"a": "A", "b": "D", "corr": 0.10},   # outside the cluster
        ]
        # mean of 0.80, 0.70, 0.90 = 0.80
        assert _cluster_mean_corr(["A", "B", "C"], pairs) == 0.80

    def test_ignores_none_corr(self):
        pairs = [
            {"a": "A", "b": "B", "corr": None},   # flat-series pair
            {"a": "A", "b": "C", "corr": 0.50},
        ]
        assert _cluster_mean_corr(["A", "B", "C"], pairs) == 0.50

    def test_no_scored_pair_returns_none(self):
        pairs = [{"a": "A", "b": "B", "corr": None}]
        assert _cluster_mean_corr(["A", "B"], pairs) is None


# ─────────────────────── upstream state propagation ─────────────────────


class TestUpstreamState:
    def test_no_data_propagates(self):
        upstream = {"state": "NO_DATA", "pairs": [], "weights": {},
                    "n_correlatable": 0}
        r = build_correlation_cluster_warning(upstream)
        assert r["state"] == "NO_DATA"
        assert r["verdict"] is None
        assert r["clusters"] == []
        assert r["biggest_cluster"] is None

    def test_insufficient_propagates(self):
        upstream = {"state": "INSUFFICIENT", "pairs": [], "weights": {},
                    "n_correlatable": 1}
        r = build_correlation_cluster_warning(upstream)
        assert r["state"] == "INSUFFICIENT"
        assert r["verdict"] is None

    def test_n_correlatable_too_small_is_insufficient(self):
        # Even if upstream said "OK", a singleton/empty correlatable set
        # can't form a multi-name cluster — degrade rather than emit a
        # bogus NO_CLUSTERS verdict.
        upstream = {"state": "OK", "pairs": [], "weights": {"A": 1.0},
                    "n_correlatable": 1}
        r = build_correlation_cluster_warning(upstream)
        assert r["state"] == "INSUFFICIENT"


# ───────────────────────── verdict ladder ───────────────────────────────


def _payload(pairs, weights, n_corr=None, state="OK"):
    if n_corr is None:
        # Default: derive n_correlatable from the pair set.
        nodes: set[str] = set()
        for p in pairs:
            nodes.add(p["a"])
            nodes.add(p["b"])
        n_corr = len(nodes)
    return {"state": state, "pairs": pairs, "weights": weights,
            "n_correlatable": n_corr}


class TestVerdict:
    def test_no_clusters_below_threshold(self):
        # All ρ < HIGH_CORR ⇒ no edges, no multi-name cluster, NO_CLUSTERS.
        pairs = [
            {"a": "A", "b": "B", "corr": 0.20},
            {"a": "A", "b": "C", "corr": 0.10},
            {"a": "B", "b": "C", "corr": 0.05},
        ]
        weights = {"A": 0.4, "B": 0.4, "C": 0.2}
        r = build_correlation_cluster_warning(_payload(pairs, weights))
        assert r["verdict"] == "NO_CLUSTERS"
        assert r["n_clusters"] == 0
        assert r["biggest_cluster"] is None

    def test_watchlist_cluster_at_low_weight(self):
        # Three high-ρ names but only 25% of book by weight ⇒
        # WATCHLIST_CLUSTER (below the 30% line).
        pairs = [
            {"a": "A", "b": "B", "corr": 0.80},
            {"a": "B", "b": "C", "corr": 0.75},
            {"a": "A", "b": "C", "corr": 0.85},
            {"a": "A", "b": "D", "corr": 0.10},
            {"a": "B", "b": "D", "corr": 0.05},
            {"a": "C", "b": "D", "corr": 0.00},
        ]
        weights = {"A": 0.10, "B": 0.10, "C": 0.05, "D": 0.75}
        r = build_correlation_cluster_warning(_payload(pairs, weights))
        assert r["verdict"] == "WATCHLIST_CLUSTER"
        assert sorted(r["biggest_cluster"]["tickers"]) == ["A", "B", "C"]
        # Total cluster weight: 0.25 (well under 0.30).
        assert r["biggest_cluster"]["total_weight"] == pytest.approx(0.25)
        assert r["biggest_cluster"]["total_weight_pct"] == 25.0

    def test_dominant_cluster_between_thresholds(self):
        pairs = [
            {"a": "A", "b": "B", "corr": 0.85},
            {"a": "A", "b": "C", "corr": 0.10},
            {"a": "B", "b": "C", "corr": 0.05},
        ]
        weights = {"A": 0.30, "B": 0.20, "C": 0.50}
        r = build_correlation_cluster_warning(_payload(pairs, weights))
        # AB cluster = 0.50 of book → between 0.30 and 0.60 → DOMINANT_CLUSTER.
        assert r["verdict"] == "DOMINANT_CLUSTER"
        assert r["biggest_cluster"]["tickers"] == ["A", "B"]
        assert r["biggest_cluster"]["total_weight"] == pytest.approx(0.50)

    def test_hidden_factor_bet_at_dominant_weight(self):
        pairs = [
            {"a": "A", "b": "B", "corr": 0.95},
            {"a": "A", "b": "C", "corr": 0.90},
            {"a": "B", "b": "C", "corr": 0.92},
            {"a": "A", "b": "D", "corr": 0.10},
            {"a": "B", "b": "D", "corr": 0.05},
            {"a": "C", "b": "D", "corr": 0.05},
        ]
        # A+B+C = 0.65 → ≥ DOMINANT_WEIGHT (0.60).
        weights = {"A": 0.25, "B": 0.20, "C": 0.20, "D": 0.35}
        r = build_correlation_cluster_warning(_payload(pairs, weights))
        assert r["verdict"] == "HIDDEN_FACTOR_BET"
        assert sorted(r["biggest_cluster"]["tickers"]) == ["A", "B", "C"]
        assert r["biggest_cluster"]["total_weight"] == pytest.approx(0.65)
        assert r["biggest_cluster"]["internal_mean_corr"] == pytest.approx(
            (0.95 + 0.90 + 0.92) / 3.0, rel=1e-3)

    def test_threshold_boundary_at_high_corr(self):
        # Edge condition: ρ exactly equal to HIGH_CORR is included
        # (`>=` not `>`).
        pairs = [{"a": "A", "b": "B", "corr": HIGH_CORR}]
        weights = {"A": 0.5, "B": 0.5}
        r = build_correlation_cluster_warning(_payload(pairs, weights))
        # AB cluster = 1.00 of book → HIDDEN_FACTOR_BET.
        assert r["verdict"] == "HIDDEN_FACTOR_BET"
        assert r["biggest_cluster"]["tickers"] == ["A", "B"]

    def test_just_below_high_corr_no_edge(self):
        pairs = [{"a": "A", "b": "B", "corr": HIGH_CORR - 0.001}]
        weights = {"A": 0.5, "B": 0.5}
        r = build_correlation_cluster_warning(_payload(pairs, weights))
        # No edge fires ⇒ no multi-name cluster.
        assert r["verdict"] == "NO_CLUSTERS"
        assert r["biggest_cluster"] is None


# ────────────────── biggest-cluster selection by weight ─────────────────


class TestBiggestSelection:
    def test_biggest_by_weight_not_size(self):
        # Two clusters: AB (2 names, 50% of book) and CDE (3 names, 30%).
        # AB wins on weight even though CDE is bigger by size.
        pairs = [
            {"a": "A", "b": "B", "corr": 0.85},
            {"a": "C", "b": "D", "corr": 0.80},
            {"a": "D", "b": "E", "corr": 0.80},
            {"a": "C", "b": "E", "corr": 0.80},
            # Cross-cluster pairs are uncorrelated (no edges).
            {"a": "A", "b": "C", "corr": 0.05},
            {"a": "A", "b": "D", "corr": 0.05},
            {"a": "A", "b": "E", "corr": 0.05},
            {"a": "B", "b": "C", "corr": 0.05},
            {"a": "B", "b": "D", "corr": 0.05},
            {"a": "B", "b": "E", "corr": 0.05},
        ]
        weights = {"A": 0.25, "B": 0.25, "C": 0.10, "D": 0.10, "E": 0.10,
                   "F": 0.20}  # F is unclustered ballast
        r = build_correlation_cluster_warning(_payload(pairs, weights))
        assert r["biggest_cluster"]["tickers"] == ["A", "B"]
        assert r["biggest_cluster"]["total_weight"] == pytest.approx(0.50)
        assert r["n_clusters"] == 2

    def test_excludes_singleton_components(self):
        # Single-name components are not "clusters" — must not appear in
        # r["clusters"].
        pairs = [
            {"a": "A", "b": "B", "corr": 0.80},
            {"a": "A", "b": "C", "corr": 0.10},
            {"a": "B", "b": "C", "corr": 0.10},
        ]
        weights = {"A": 0.4, "B": 0.4, "C": 0.2}
        r = build_correlation_cluster_warning(_payload(pairs, weights))
        # Only the AB pair counts.
        assert r["n_clusters"] == 1
        # MIN_CLUSTER_SIZE contract holds.
        assert all(c["size"] >= MIN_CLUSTER_SIZE for c in r["clusters"])


# ──────────────────────────── metadata ──────────────────────────────────


class TestMetadata:
    def test_threshold_corr_surfaced(self):
        pairs = [{"a": "A", "b": "B", "corr": 0.80}]
        r = build_correlation_cluster_warning(
            _payload(pairs, {"A": 0.5, "B": 0.5}))
        assert r["threshold_corr"] == HIGH_CORR

    def test_threshold_overridable(self):
        # A caller can pass a stricter threshold; with 0.95 the 0.80 edge
        # disappears and the verdict becomes NO_CLUSTERS.
        pairs = [{"a": "A", "b": "B", "corr": 0.80}]
        r = build_correlation_cluster_warning(
            _payload(pairs, {"A": 0.5, "B": 0.5}), threshold=0.95)
        assert r["threshold_corr"] == 0.95
        assert r["verdict"] == "NO_CLUSTERS"

    def test_constants_band_orderings(self):
        # The verdict-ladder bands must be strictly ordered or every test
        # above is over-determined. Lock the relationship explicitly so
        # nobody re-orders one without re-evaluating the others.
        assert 0.0 < WATCHLIST_WEIGHT < DOMINANT_WEIGHT <= 1.0
        assert MIN_CLUSTER_SIZE >= 2
