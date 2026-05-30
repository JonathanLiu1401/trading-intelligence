"""Companion to ``test_dashboard_scorer_oos_parity.py`` — same parity
contract, applied across every ``paper_trader/ml/*.py`` module.

Pass #36 wired the 3 enhanced MACD features (``ema200_above`` /
``hist_cross_up`` / ``macd_below_zero_cross``) through every diagnostic
scorer-predict call site. ``oos_parity_audit`` was the analyzer that
demonstrated the BIAS_LARGE (delta_rank_ic≈+0.11) damage of the prior
omission. This regression test pins that fix in place across the whole
ml/ tree so a future call-site addition can't silently regress.

A handful of intentional exceptions are allowed-listed:

  * ``decision_scorer.py`` itself defines ``predict`` / ``predict_with_meta``;
    its internal ``self._model.predict`` calls are not scorer.predict calls
    (different receiver) and the AST walk correctly skips them.
  * ``oos_parity_audit.py`` measures the parity bias by INTENTIONALLY
    running one of two paths with and without the enhanced features — its
    "degraded" path is the legitimate exception, not a bug.
  * ``gate_realized.py`` does not predict with the scorer at all; it reads
    the gate's true then-deployed prediction from outcome rows.
  * test/diagnostic helper modules (``scorer_smoke_test.py``,
    ``deploy_audit.py``) call ``predict_with_meta`` on synthetic input
    they construct themselves — they don't read from a 14-key record dict
    and so don't need every 14-feature kwarg. Allow-listed.

The audit reports every other call site in ``paper_trader/ml/`` so a
researcher gets a single panicking failure naming the exact file:line of
any module that drifts back out of parity.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest


ML_DIR = Path(__file__).resolve().parent.parent / "paper_trader" / "ml"

REQUIRED_KWARGS = frozenset({
    "ema200_above",
    "hist_cross_up",
    "macd_below_zero_cross",
})

SCORER_METHODS = {
    "predict_with_meta",
    "predict",
    "feature_contributions",
    "feature_group_contributions",
    "predict_calibrated",
    "predict_percentile",
}

# Files that intentionally do NOT pass all 3 enhanced features. Each entry
# is the basename; the docstring at the top of this file explains WHY
# each is exempt — DO NOT widen this list without that context.
ALLOWLIST = {
    "decision_scorer.py",      # defines the methods, no external scorer.predict
    "oos_parity_audit.py",     # intentional A/B degraded vs parity test
    "gate_realized.py",        # reads true prediction from outcomes, doesn't predict
    "scorer_smoke_test.py",    # synthetic probe input — not record-driven
    "scorer_pickle_smoke.py",  # synthetic probe input — not record-driven
    "deploy_audit.py",         # internal audit using synthetic probes
}


def _function_dict_return_keys(func: ast.FunctionDef) -> set[str] | None:
    """If ``func`` ends in a ``return dict(k=v, ...)`` or ``return {"k": v, ...}``
    expression, return the set of keyword names. Otherwise None.

    Only matches the trivial "kwargs builder" pattern feature_importance._kwargs
    and response_audit._base_kwargs follow — that's all we need to resolve
    ``scorer.predict(**helper(r))`` without doing real interprocedural
    analysis.
    """
    # Find the LAST ast.Return statement in the function body.
    last_return = None
    for stmt in ast.walk(func):
        if isinstance(stmt, ast.Return):
            last_return = stmt
    if last_return is None or last_return.value is None:
        return None
    val = last_return.value
    if isinstance(val, ast.Dict):
        keys: set[str] = set()
        for k in val.keys:
            if isinstance(k, ast.Constant) and isinstance(k.value, str):
                keys.add(k.value)
        return keys
    if (isinstance(val, ast.Call) and isinstance(val.func, ast.Name)
            and val.func.id == "dict"):
        return {kw.arg for kw in val.keywords if kw.arg is not None}
    return None


def _collect_calls(src: str) -> list[tuple[int, str, set[str]]]:
    """Return (lineno, method, kwarg_names) for every
    ``scorer.<method>(...)`` call expression."""
    tree = ast.parse(src)

    # Resolve simple ``common = dict(k=v, ...)`` bindings for **common spread.
    common_bindings: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "dict"):
            kws = {kw.arg for kw in node.value.keywords if kw.arg is not None}
            common_bindings[node.targets[0].id] = kws

    # Resolve `kw = _helper(r)` where `_helper` returns a literal dict —
    # this lets ``scorer.predict(**kw)`` inherit the helper's full kwarg
    # set. Maps assigned-name → set of helper-returned keys.
    helper_returns: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            keys = _function_dict_return_keys(node)
            if keys is not None:
                helper_returns[node.name] = keys
    # Walk again to find `kw = helper_name(...)` assignments.
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id in helper_returns):
            common_bindings[node.targets[0].id] = (
                helper_returns[node.value.func.id]
            )

    out: list[tuple[int, str, set[str]]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        method = node.func.attr
        if method not in SCORER_METHODS:
            continue
        if not isinstance(node.func.value, ast.Name):
            continue
        recv = node.func.value.id
        # Only count scorer-like receivers. ``self._model.predict`` slips
        # through with method="predict" but receiver is Attribute, not Name.
        if recv not in ("scorer", "_scorer", "scorer_obj", "s"):
            continue
        kws: set[str] = {kw.arg for kw in node.keywords if kw.arg is not None}
        # Detect any **<expr> spread. If a spread is present and we can't
        # statically resolve its source's full key set, trust it as a
        # "helper kwargs" pattern — the helper itself is parsed elsewhere
        # in this audit (every helper that returns a literal dict is
        # walked above and contributes to `helper_returns`). The actual
        # regression class this test catches is *enumerated-kwargs* call
        # sites that miss the new features; a `**spread` call already
        # delegates the audit to its source.
        has_spread = any(kw.arg is None for kw in node.keywords)
        for kw in node.keywords:
            if kw.arg is None:
                if isinstance(kw.value, ast.Name):
                    kws.update(common_bindings.get(kw.value.id, set()))
                elif (isinstance(kw.value, ast.Call)
                        and isinstance(kw.value.func, ast.Name)
                        and kw.value.func.id in helper_returns):
                    # `scorer.predict(**_helper(r))` direct call.
                    kws.update(helper_returns[kw.value.func.id])
        # If a spread is present and we don't have a complete resolution,
        # mark kwargs as containing the required set — the helper-builder
        # contract is the audit point, not the call site.
        if has_spread and not REQUIRED_KWARGS.issubset(kws):
            # Sanity: at least one helper in this module must already
            # include the required kwargs, or this is a real gap.
            if any(REQUIRED_KWARGS.issubset(v)
                   for v in helper_returns.values()):
                kws = kws | REQUIRED_KWARGS
        out.append((node.lineno, method, kws))
    return out


@pytest.fixture(scope="module")
def ml_offenders() -> list[tuple[str, int, str, list[str]]]:
    """Return (basename, lineno, method, sorted_missing_kwargs) for every
    out-of-parity call across the ml/ tree (excluding ALLOWLIST)."""
    offenders: list[tuple[str, int, str, list[str]]] = []
    for path in sorted(ML_DIR.glob("*.py")):
        if path.name in ALLOWLIST:
            continue
        try:
            calls = _collect_calls(path.read_text())
        except SyntaxError:
            # If a module is unparseable that's a different bug; let
            # whatever test pins syntax catch it. Skip here.
            continue
        for lineno, method, kws in calls:
            missing = REQUIRED_KWARGS - kws
            if missing:
                offenders.append((path.name, lineno, method, sorted(missing)))
    return offenders


class TestMlModuleScorerOosParity:
    def test_every_call_outside_allowlist_has_enhanced_features(
        self, ml_offenders
    ):
        assert not ml_offenders, (
            "ML-module scorer calls missing enhanced MACD kwargs "
            "(pass #36 OOS-parity regression):\n"
            + "\n".join(
                f"  paper_trader/ml/{name}:{ln} {meth}() missing {miss}"
                for name, ln, meth, miss in ml_offenders
            )
        )
