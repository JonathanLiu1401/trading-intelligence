"""Tests for the deployment-plan internal-conflict audit.

The builder is pure (no DB / no clock) so the tests are deterministic
fixtures over the same plan-row shape ``build_deployment_plan`` emits.

Coverage:

* Verdict ladder: NO_PLAN / CARRY_WASTE / OPPOSING_UNLEVERED /
  DIRECTIONAL_HEDGE / CLEAN with the exact severity boundaries pinned.
* Family conflict detection reuses ``inverse_pair_conflict._TICKER_INDEX``
  (drift between this audit and the held-book audit is impossible).
* Aggregate directional-hedge fires across families when no single
  family is itself conflicted.
* Cross-classification: a strict family conflict OUTRANKS the soft
  directional-hedge flag (CARRY_WASTE > DIRECTIONAL_HEDGE).
* Off-taxonomy tickers fall back to ``leverage_factor`` on the input row
  for aggregate totals.
* Garbage rows never raise; empty plan → NO_PLAN.
* Flask endpoint contract via the test_client.
"""
from __future__ import annotations

import pytest

from paper_trader.analytics.deployment_plan_conflicts import (
    build_deployment_plan_conflicts,
    DIRECTIONAL_MIN_PCT,
)
from paper_trader.analytics.inverse_pair_conflict import (
    SEVERITY_HIGH_PCT,
    SEVERITY_MEDIUM_PCT,
)


def _row(ticker: str, alloc_usd: float, leverage_factor: float = 1.0,
         is_leveraged: bool | None = None) -> dict:
    if is_leveraged is None:
        is_leveraged = abs(leverage_factor) > 1.0
    return {
        "ticker": ticker,
        "alloc_usd": alloc_usd,
        "leverage_factor": leverage_factor,
        "is_leveraged": is_leveraged,
        "pred_5d_return_pct": 5.0,
        "scorer_verdict": "STRONG_HOLD",
        "sector": "semis",
    }


class TestVerdictLadder:
    def test_no_plan_empty_list(self):
        r = build_deployment_plan_conflicts([])
        assert r["verdict"] == "NO_PLAN"
        assert r["n_plan_rows"] == 0
        assert r["family_conflicts"] == []
        assert r["directional_hedge"] is False

    def test_no_plan_none(self):
        r = build_deployment_plan_conflicts(None)  # type: ignore[arg-type]
        assert r["verdict"] == "NO_PLAN"

    def test_clean_single_long(self):
        # Single long leveraged ETF — no conflict.
        plan = [_row("TQQQ", 200.0, leverage_factor=3.0)]
        r = build_deployment_plan_conflicts(plan)
        assert r["verdict"] == "CLEAN"
        assert r["n_family_conflicts"] == 0
        assert r["directional_hedge"] is False

    def test_clean_off_taxonomy_only(self):
        # NVDA + AMAT — both off-taxonomy (single-name stocks). No
        # taxonomy family hit, no leverage-vs-inverse, no hedge.
        plan = [
            _row("NVDA", 200.0, leverage_factor=1.0),
            _row("AMAT", 200.0, leverage_factor=1.0),
        ]
        r = build_deployment_plan_conflicts(plan)
        assert r["verdict"] == "CLEAN"

    def test_carry_waste_same_family_both_leveraged(self):
        # TQQQ (+3x QQQ) + SQQQ (-3x QQQ) — both pay decay.
        plan = [
            _row("TQQQ", 200.0, leverage_factor=3.0),
            _row("SQQQ", 200.0, leverage_factor=-3.0),
        ]
        r = build_deployment_plan_conflicts(plan)
        assert r["verdict"] == "CARRY_WASTE"
        assert r["n_family_conflicts"] == 1
        c = r["family_conflicts"][0]
        assert c["family"] == "QQQ"
        assert c["classification"] == "CARRY_WASTE"
        # The severity metric is ``cancelled / gross`` where
        # gross = |long_delta| + |inverse_delta| and
        # cancelled = min(|long_delta|, |inverse_delta|). For a perfect
        # offset the ratio caps at 0.5 (50%), so MEDIUM is the maximum
        # severity reachable in practice — HIGH (≥80%) is unreachable by
        # construction. We pin MEDIUM to lock the metric semantics.
        assert c["severity"] == "MEDIUM"
        # And the per-family deltas are emitted with the exact dollar
        # amounts so the panel can show "long $600 vs inverse -$600".
        assert c["long_delta_usd"] == 600.0
        assert c["inverse_delta_usd"] == -600.0
        assert c["cancelled_delta_usd"] == 600.0
        assert c["net_delta_usd"] == 0.0

    def test_opposing_unlevered_core_plus_inverse(self):
        # QQQ (1x core) + SQQQ (-3x inverse) — no leveraged-long sleeve.
        plan = [
            _row("QQQ", 200.0, leverage_factor=1.0),
            _row("SQQQ", 200.0, leverage_factor=-3.0),
        ]
        r = build_deployment_plan_conflicts(plan)
        assert r["verdict"] == "OPPOSING_UNLEVERED"
        assert r["n_family_conflicts"] == 1
        c = r["family_conflicts"][0]
        assert c["classification"] == "OPPOSING_UNLEVERED"

    def test_carry_waste_outranks_opposing_unlevered(self):
        # USTECH family (TECL+TECS — CARRY_WASTE) AND QQQ family
        # (QQQ+SQQQ — OPPOSING_UNLEVERED). The overall verdict must be
        # CARRY_WASTE (the worst).
        plan = [
            _row("TECL", 200.0, leverage_factor=3.0),
            _row("TECS", 200.0, leverage_factor=-3.0),
            _row("QQQ", 200.0, leverage_factor=1.0),
            _row("SQQQ", 200.0, leverage_factor=-3.0),
        ]
        r = build_deployment_plan_conflicts(plan)
        assert r["verdict"] == "CARRY_WASTE"

    def test_directional_hedge_across_families(self):
        # TECS (-3x USTECH inverse) + TQQQ (+3x QQQ long) — different
        # families so no family-level conflict, but aggregate carries
        # both directions in leveraged form.
        # 200/400 = 50% inverse_lev, 200/400 = 50% long_lev — both above
        # DIRECTIONAL_MIN_PCT.
        plan = [
            _row("TECS", 200.0, leverage_factor=-3.0),
            _row("TQQQ", 200.0, leverage_factor=3.0),
        ]
        r = build_deployment_plan_conflicts(plan)
        # Different families so no family conflict, but cross-family
        # directional hedge fires.
        # Note: TQQQ is in QQQ family, TECS is in USTECH — they don't
        # cross. So family_conflicts == [] but directional_hedge == True.
        assert r["n_family_conflicts"] == 0
        assert r["directional_hedge"] is True
        assert r["verdict"] == "DIRECTIONAL_HEDGE"

    def test_directional_hedge_below_threshold(self):
        # Long-lev 96% vs inverse-lev 4% — below DIRECTIONAL_MIN_PCT (5%)
        # so no hedge flag.
        plan = [
            _row("TQQQ", 960.0, leverage_factor=3.0),
            _row("SQQQ", 40.0, leverage_factor=-3.0),
        ]
        # NOTE: This DOES still trip family_conflict (TQQQ + SQQQ same
        # family) — that's a CARRY_WASTE, the outer verdict.
        r = build_deployment_plan_conflicts(plan)
        assert r["verdict"] == "CARRY_WASTE"

    def test_directional_hedge_exact_threshold(self):
        # Each side exactly DIRECTIONAL_MIN_PCT (5%) of total. Different
        # families. Both sides hit the gate exactly.
        # long 50 / inverse 50 / unlev 900 — long_lev_pct = 5.0,
        # inverse_lev_pct = 5.0, both at the threshold → directional_hedge
        # True (>=).
        plan = [
            _row("TQQQ", 50.0, leverage_factor=3.0),
            _row("TECS", 50.0, leverage_factor=-3.0),
            _row("AAPL", 900.0, leverage_factor=1.0),
        ]
        r = build_deployment_plan_conflicts(plan)
        # No family-level conflict (TQQQ in QQQ, TECS in USTECH).
        assert r["directional_hedge"] is True
        assert r["verdict"] == "DIRECTIONAL_HEDGE"

    def test_directional_hedge_only_one_side(self):
        # 50% long-lev, 0% inverse-lev — no hedge, fully long-lev plan.
        plan = [
            _row("TQQQ", 500.0, leverage_factor=3.0),
            _row("AAPL", 500.0, leverage_factor=1.0),
        ]
        r = build_deployment_plan_conflicts(plan)
        assert r["directional_hedge"] is False
        assert r["verdict"] == "CLEAN"


class TestSeverityLadder:
    def test_severity_caps_at_medium_for_perfect_offset(self):
        # Perfect dollar-offset is the maximum-reachable cancellation
        # ratio: min(|L|, |I|) / (|L| + |I|) = 0.5. Below the 80% HIGH
        # cutoff, so MEDIUM is the highest practically-reachable
        # severity. The HIGH branch in ``_severity`` is dead code by
        # design — this test pins the contract so a future caller
        # doesn't assume HIGH is reachable.
        plan = [
            _row("TQQQ", 200.0, leverage_factor=3.0),
            _row("SQQQ", 200.0, leverage_factor=-3.0),
        ]
        r = build_deployment_plan_conflicts(plan)
        assert r["family_conflicts"][0]["severity"] == "MEDIUM"

    def test_severity_medium_partial_offset(self):
        # TQQQ $300 × 3 = +$900, SQQQ $200 × -3 = -$600 → cancelled $600,
        # gross $1500 → 40% → MEDIUM.
        plan = [
            _row("TQQQ", 300.0, leverage_factor=3.0),
            _row("SQQQ", 200.0, leverage_factor=-3.0),
        ]
        r = build_deployment_plan_conflicts(plan)
        c = r["family_conflicts"][0]
        assert c["severity"] == "MEDIUM"

    def test_severity_low_small_offset(self):
        # TQQQ $1000 × 3 = +$3000, SQQQ $100 × -3 = -$300 → cancelled
        # $300, gross $3300 → 9.1% → LOW.
        plan = [
            _row("TQQQ", 1000.0, leverage_factor=3.0),
            _row("SQQQ", 100.0, leverage_factor=-3.0),
        ]
        r = build_deployment_plan_conflicts(plan)
        assert r["family_conflicts"][0]["severity"] == "LOW"

    def test_severity_boundaries_pinned(self):
        # Document the exact thresholds so a drift in the source is
        # caught here.
        assert SEVERITY_HIGH_PCT == 80.0
        assert SEVERITY_MEDIUM_PCT == 40.0


class TestAggregateTotals:
    def test_aggregate_totals_off_taxonomy(self):
        # Off-taxonomy tickers contribute to aggregate totals based on
        # the input row's ``leverage_factor`` field.
        plan = [
            _row("NVDU", 100.0, leverage_factor=2.0),   # long-lev, off-taxonomy
            _row("NVDA", 200.0, leverage_factor=1.0),   # unlev, off-taxonomy
            _row("TECS", 100.0, leverage_factor=-3.0),  # inverse-lev, USTECH
        ]
        r = build_deployment_plan_conflicts(plan)
        assert r["total_plan_usd"] == 400.0
        assert r["totals"]["long_leveraged_usd"] == 100.0
        assert r["totals"]["inverse_leveraged_usd"] == 100.0
        assert r["totals"]["unleveraged_usd"] == 200.0
        # 25% / 25% → DIRECTIONAL_HEDGE (no family conflict between
        # NVDU and TECS — NVDU is off-taxonomy).
        assert r["directional_hedge"] is True

    def test_total_plan_usd_sum(self):
        plan = [
            _row("AAA", 100.0),
            _row("BBB", 250.0),
            _row("CCC", 50.0),
        ]
        r = build_deployment_plan_conflicts(plan)
        assert r["total_plan_usd"] == 400.0
        assert r["n_plan_rows"] == 3


class TestSemiFamilyConflict:
    def test_soxl_plus_soxs_is_carry_waste(self):
        plan = [
            _row("SOXL", 150.0, leverage_factor=3.0),
            _row("SOXS", 150.0, leverage_factor=-3.0),
        ]
        r = build_deployment_plan_conflicts(plan)
        assert r["verdict"] == "CARRY_WASTE"
        c = r["family_conflicts"][0]
        assert c["family"] == "SEMIS"

    def test_smh_plus_soxs_is_opposing_unlevered(self):
        plan = [
            _row("SMH", 150.0, leverage_factor=1.0),
            _row("SOXS", 150.0, leverage_factor=-3.0),
        ]
        r = build_deployment_plan_conflicts(plan)
        assert r["verdict"] == "OPPOSING_UNLEVERED"
        c = r["family_conflicts"][0]
        assert c["family"] == "SEMIS"


class TestRobustness:
    def test_garbage_rows_skipped(self):
        plan = [None, "garbage", 42, {}, {"ticker": ""},  # all skipped
                {"ticker": "TQQQ", "alloc_usd": 0},  # zero alloc skipped
                _row("TQQQ", 200.0, leverage_factor=3.0)]
        r = build_deployment_plan_conflicts(plan)  # type: ignore[arg-type]
        assert r["n_plan_rows"] == 1
        assert r["verdict"] == "CLEAN"

    def test_string_alloc_usd_coerced(self):
        # Real plans always emit numeric alloc_usd; defensive coercion
        # ensures a malformed upstream entry doesn't crash the audit.
        plan = [{"ticker": "TQQQ", "alloc_usd": "200.0",
                 "leverage_factor": 3.0}]
        r = build_deployment_plan_conflicts(plan)
        assert r["n_plan_rows"] == 1
        assert r["total_plan_usd"] == 200.0

    def test_negative_alloc_skipped(self):
        plan = [_row("TQQQ", -100.0, leverage_factor=3.0)]
        r = build_deployment_plan_conflicts(plan)
        # Negative alloc is treated as zero and skipped.
        assert r["n_plan_rows"] == 0
        assert r["verdict"] == "NO_PLAN"

    def test_lowercase_ticker_normalised(self):
        plan = [
            _row("tqqq", 200.0, leverage_factor=3.0),
            _row("sqqq", 200.0, leverage_factor=-3.0),
        ]
        r = build_deployment_plan_conflicts(plan)
        # Tickers upper-cased before taxonomy lookup.
        assert r["verdict"] == "CARRY_WASTE"


class TestHeadlines:
    def test_carry_waste_headline_names_tickers(self):
        plan = [
            _row("TQQQ", 200.0, leverage_factor=3.0),
            _row("SQQQ", 200.0, leverage_factor=-3.0),
        ]
        r = build_deployment_plan_conflicts(plan)
        assert "TQQQ" in r["headline"]
        assert "SQQQ" in r["headline"]
        assert "QQQ" in r["headline"]
        assert "decay" in r["headline"]

    def test_clean_headline(self):
        plan = [_row("AAPL", 200.0, leverage_factor=1.0)]
        r = build_deployment_plan_conflicts(plan)
        assert r["headline"] == "no inverse-pair conflicts in plan"

    def test_directional_hedge_headline(self):
        plan = [
            _row("TQQQ", 200.0, leverage_factor=3.0),
            _row("TECS", 200.0, leverage_factor=-3.0),
        ]
        r = build_deployment_plan_conflicts(plan)
        assert "long-leveraged" in r["headline"]
        assert "inverse-leveraged" in r["headline"]
        assert "double-decay" in r["headline"]


class TestConstants:
    def test_directional_min_pct_pinned(self):
        # If we drift this we should know via test; the trader UI assumes
        # 5% as the directional-hedge floor.
        assert DIRECTIONAL_MIN_PCT == 5.0


class TestFlaskEndpointContract:
    def test_endpoint_returns_json_shape(self):
        from paper_trader.dashboard import app
        client = app.test_client()
        resp = client.get("/api/deployment-plan-conflicts")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, dict)
        # Verdict is always one of the documented ladder values.
        assert data.get("verdict") in (
            "NO_PLAN", "CARRY_WASTE", "OPPOSING_UNLEVERED",
            "DIRECTIONAL_HEDGE", "CLEAN", "ERROR",
        )
        # Planner context surfaced.
        assert "planner_verdict" in data or "error" in data
        # The audit envelope's totals dict has the documented keys.
        if data["verdict"] != "ERROR":
            assert "totals" in data
            t = data["totals"]
            for k in ("long_leveraged_usd", "inverse_leveraged_usd",
                      "unleveraged_usd", "long_leveraged_pct",
                      "inverse_leveraged_pct"):
                assert k in t

    def test_endpoint_forwards_query_params(self):
        from paper_trader.dashboard import app
        client = app.test_client()
        # leveraged_cap_pct=0 forces the planner to skip every leveraged
        # name → no inverse-pair conflict possible.
        resp = client.get(
            "/api/deployment-plan-conflicts?leveraged_cap_pct=0"
        )
        assert resp.status_code == 200
        data = resp.get_json()
        if data.get("verdict") != "ERROR":
            # No leveraged names in the plan → no family conflict and no
            # directional hedge.
            assert data["verdict"] in ("NO_PLAN", "CLEAN")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
