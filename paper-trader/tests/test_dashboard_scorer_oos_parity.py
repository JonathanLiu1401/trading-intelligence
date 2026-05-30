"""Regression test: every dashboard endpoint that calls
``DecisionScorer.predict_with_meta`` MUST forward the 3 enhanced MACD
features (``ema200_above`` / ``hist_cross_up`` / ``macd_below_zero_cross``)
alongside the legacy 11 inputs.

Why this matters — the pass #36 OOS-parity fix wired these features through
every diagnostic predict call site (``gate_audit`` / ``gate_pnl`` /
``baseline_compare`` / ``feature_importance`` / ``response_audit`` / …).
But the live dashboard endpoints (``/api/scorer-predictions``,
``/api/scorer-attribution``, ``/api/scorer-opportunities``,
``/api/conviction-cards``, the held-positions helper) were called the same
way the diagnostics WERE — they would silently predict on a degraded
vector (defaults to 0.0 for the 3 new features) while the live
``_ml_decide`` gate already passes them. ``oos_parity_audit`` measures
that bias as ``BIAS_LARGE`` (delta_rank_ic≈+0.11) on the deployed pickle,
so the dashboard's score numbers diverge from the gate's by ~10 pp of
rank-IC on the same input row. A reading quant trusting the dashboard's
"pred_5d_return_pct" would be reading a different model than the gate
sees. Pin parity at the source so a future predict-call addition can't
silently regress.

The test parses ``paper_trader/dashboard.py`` with the ``ast`` module
(zero-runtime, no network, no Flask init) and walks every call expression
to ``scorer.predict_with_meta`` / ``scorer.predict`` / ``scorer.feature_contributions``,
asserting each carries the 3 required kwargs.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest


DASHBOARD_PATH = (
    Path(__file__).resolve().parent.parent / "paper_trader" / "dashboard.py"
)

# The 3 enhanced MACD features added to ``DecisionScorer.build_features``
# in the 2026-05 retrain cycle. The deployed pickle has non-zero learned
# weights for each (mean|w|≈0.42/0.30/0.27 — verified by
# ``deploy_audit.diagnose_dead_features``), so omitting them at predict
# time silently uses the 0.0 default and the gate sees a different
# prediction than every dashboard endpoint.
REQUIRED_KWARGS = frozenset({
    "ema200_above",
    "hist_cross_up",
    "macd_below_zero_cross",
})

# Methods whose calls must carry these kwargs.
# - ``predict_with_meta`` is the rich path (clamp / off_distribution /
#   percentile / calibrated meta).
# - ``predict`` is the scalar fast path.
# - ``feature_contributions`` is the per-feature attribution path; it
#   ALSO must take the new features so the attribution computes on the
#   same vector as ``predict_with_meta``.
# - ``feature_group_contributions`` for the same reason.
SCORER_METHODS = {
    "predict_with_meta",
    "predict",
    "feature_contributions",
    "feature_group_contributions",
    "predict_calibrated",
    "predict_percentile",
}


def _collect_scorer_calls(src: str) -> list[tuple[int, str, set[str]]]:
    """Return [(line_number, method_name, kwarg_names)] for every call
    expression of the form ``<x>.<scorer_method>(...)`` in ``src``.

    Captures both keyword args and the keys of any ``**dict`` literal
    spread (``predict_with_meta(**common)`` where ``common = dict(...)``)
    — the common-kw idiom is used in /api/scorer-attribution and must be
    walked back to the binding's kwargs.
    """
    tree = ast.parse(src)
    # Build a map of name -> set(kwarg names) for any local assignment of
    # the form ``common = dict(k=v, ...)`` so we can resolve **common into
    # its concrete kwarg set when surveying the call site.
    common_bindings: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) \
                and isinstance(node.value, ast.Call) \
                and isinstance(node.value.func, ast.Name) \
                and node.value.func.id == "dict":
            kws = {kw.arg for kw in node.value.keywords if kw.arg is not None}
            common_bindings[node.targets[0].id] = kws

    results: list[tuple[int, str, set[str]]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        method = node.func.attr
        if method not in SCORER_METHODS:
            continue
        # Only count attribute accesses on a name that looks like a scorer
        # (``scorer.predict_with_meta``). Otherwise broad string matches
        # (e.g. ``self.predict``) might capture unrelated methods.
        if not isinstance(node.func.value, ast.Name):
            continue
        if node.func.value.id not in ("scorer", "_scorer"):
            continue
        kws: set[str] = {kw.arg for kw in node.keywords if kw.arg is not None}
        # Resolve **common spread into the bound dict's kwargs.
        for kw in node.keywords:
            if kw.arg is None and isinstance(kw.value, ast.Name):
                kws.update(common_bindings.get(kw.value.id, set()))
        results.append((node.lineno, method, kws))
    return results


@pytest.fixture(scope="module")
def dashboard_scorer_calls() -> list[tuple[int, str, set[str]]]:
    src = DASHBOARD_PATH.read_text()
    return _collect_scorer_calls(src)


class TestDashboardScorerOosParity:
    """Every scorer-method call in dashboard.py must include the 3
    enhanced MACD features the deployed pickle was trained on, mirroring
    the pass #36 parity fix applied to every diagnostic module."""

    def test_at_least_one_scorer_call_is_audited(
        self, dashboard_scorer_calls
    ):
        """Sanity: the parser actually found dashboard scorer calls.
        Otherwise a future refactor that silently moved them to a helper
        the AST walk doesn't follow would degrade this test to a no-op
        instead of failing — making the regression invisible.
        """
        assert len(dashboard_scorer_calls) >= 5, (
            f"expected ≥5 dashboard scorer calls (predictions / "
            f"attribution / opportunities / conviction-cards / "
            f"held-helper); parser found {len(dashboard_scorer_calls)} — "
            f"the AST walk likely missed the call sites and this test "
            f"would falsely pass. Calls found: {dashboard_scorer_calls}"
        )

    def test_every_scorer_call_forwards_enhanced_macd_features(
        self, dashboard_scorer_calls
    ):
        """The actionable assertion: every collected call must include
        all 3 REQUIRED_KWARGS. The failure message names the exact line
        + missing kwargs so an operator can fix in one diff."""
        offenders = []
        for lineno, method, kws in dashboard_scorer_calls:
            missing = REQUIRED_KWARGS - kws
            if missing:
                offenders.append((lineno, method, sorted(missing)))
        assert not offenders, (
            "Dashboard scorer calls missing enhanced MACD kwargs:\n"
            + "\n".join(
                f"  paper_trader/dashboard.py:{ln} {meth}() missing "
                f"{miss} — predicts on a degraded vector vs the gate"
                for ln, meth, miss in offenders
            )
        )

    def test_kwargs_are_keyword_passed_not_positional(
        self, dashboard_scorer_calls
    ):
        """``predict_with_meta`` accepts the legacy 11 inputs both
        positionally and as kwargs; the 3 new features are kwarg-only
        (added at the tail). If a future refactor mixes positional
        + kwarg in a way that drops the new features, this test still
        catches it via REQUIRED_KWARGS — but pin the no-positional
        invariant explicitly so the failure mode stays one-line clear.
        Iterates the collected ast.keywords from the source — every
        scorer call in dashboard.py must use kwargs only (the codebase
        idiom already does), so any positional-arg form is a regression.
        """
        # Re-parse to access raw Call nodes for positional-arg detection.
        tree = ast.parse(DASHBOARD_PATH.read_text())
        offenders = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr not in SCORER_METHODS:
                continue
            if not isinstance(node.func.value, ast.Name):
                continue
            if node.func.value.id not in ("scorer", "_scorer"):
                continue
            if node.args:  # any positional → flag
                offenders.append((node.lineno, node.func.attr,
                                  len(node.args)))
        assert not offenders, (
            "Dashboard scorer calls use positional args (must be "
            "kwargs-only to preserve the predict_with_meta signature): "
            f"{offenders}"
        )
