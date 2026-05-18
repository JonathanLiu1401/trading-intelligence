"""Deployed-scorer config-vs-source audit — read-only.

The single most-repeated finding across ~16 ML/backtest review passes
(#15–#20) is **not a code bug** — it is a *deploy* gap. The running
`run_continuous_backtests.py` process imported `decision_scorer.py` at its
own start time and keeps the OLD `MLPRegressor` hyper-parameters resident in
memory, so every per-cycle retrain re-pickles a model built with the stale
config even though the on-disk source has long since been retuned. Observed
live and documented every pass: the deployed `data/ml/decision_scorer.pkl`
is `(64,32,16) / alpha=1e-4 / early_stopping=False` (the memorizing,
val≪oos-overfit net) while `decision_scorer.MLP_CONFIG` says
`(32,16) / alpha=1e-2 / early_stopping=True` (the regularized net commit
`5a0af2d` shipped to close that gap). The conviction gate (invariant #5)
acts on this pickle's predictions every cycle once `n_train ≥ 500`, so a
skeptical quant is sizing real conviction on a net the source no longer
endorses — with **no durable, trendable signal** that this is happening.

`/api/build-info` (test_build_info.py) answers the *adjacent* question —
"did this process boot on stale git bytecode" — at the git-SHA / process
level. It cannot answer the model-artifact question this module owns:
*"is the pickle the gate consumes the architecture the current source would
actually train?"* Those differ: a freshly restarted loop whose first
retrain has not completed still serves the old pickle; and the pickle can
lag the source for reasons (skipped retrains, a stale long-lived loop) that
a git SHA cannot see. This compares the **deployed fitted model's own
attributes** against `decision_scorer.MLP_CONFIG` (the single source of
truth `train_scorer` builds from), so the check is a true no-drift
comparison, never a hand-maintained mirror.

Operational discipline mirrors `overfit_gap` / `skill_trend` / `calibration`:
read-only, no train, no pickle *write*, no `build_features` / `N_FEATURES`
touch, no trade path — safe to run against the live unattended loop. Never
raises on a missing / unreadable / fallback pickle: it degrades to an
honest non-STALE verdict rather than a false alarm (the documented
"insufficient data is not a failure" discipline).

```bash
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.deploy_audit
```
"""
from __future__ import annotations

import math
import pickle
from pathlib import Path

_MISSING = object()


def _normalize(v):
    """Tuple/list → tuple for order-preserving structural compare; else as-is.

    sklearn stores `hidden_layer_sizes` exactly as passed at construction —
    the old code passed the tuple `(64, 32, 16)`, the new `(32, 16)`. A
    legacy pickle could in principle carry a list; normalising both sides to
    a tuple makes `(32,16)` and `[32,16]` compare equal so the verdict keys
    on the *architecture*, not the container type.
    """
    if isinstance(v, (list, tuple)):
        return tuple(v)
    return v


def _values_match(expected, actual) -> bool:
    """True iff `actual` equals `expected` for config-audit purposes.

    Floats (e.g. `alpha=1e-2`) are compared with `math.isclose` so a pickle
    round-trip's last-bit drift is not mis-reported as a stale config; bools
    (`early_stopping`) are excluded from the numeric branch because
    `bool` is an `int` subclass and `isclose(True, 1)` would be True.
    Everything else (strings, ints, normalised tuples) uses `==`.
    """
    if actual is _MISSING:
        return False
    e, a = _normalize(expected), _normalize(actual)
    if (isinstance(e, (int, float)) and not isinstance(e, bool)
            and isinstance(a, (int, float)) and not isinstance(a, bool)):
        return math.isclose(float(a), float(e), rel_tol=1e-9, abs_tol=1e-12)
    return a == e


def _is_lstsq_fallback(model) -> bool:
    """The numpy weighted-least-squares fallback used on sklearn-absent hosts.

    Identified by class name + defining module rather than an isinstance (no
    need to import sklearn or risk an import cycle): `_LstsqModel` lives in
    `paper_trader.ml.decision_scorer`. Its config (an MLP architecture) is
    not applicable, so the audit reports `LSTSQ_FALLBACK` honestly instead of
    fabricating a stale-config alarm against attributes it never had.
    """
    cls = type(model)
    return (cls.__name__ == "_LstsqModel"
            and "decision_scorer" in (getattr(cls, "__module__", "") or ""))


def audit_deployed_config(scorer_path: Path | str,
                          expected_config: dict) -> dict:
    """Compare a deployed scorer pickle's fitted model against the expected
    `MLP_CONFIG`. Pure, total, never raises.

    Returns ``{verdict, deployed, expected, mismatches, n_audited,
    n_mismatched, hint}``:

    - ``INSUFFICIENT_DATA``       — no pickle on disk yet (untrained loop /
      fresh checkout). Not a failure: there is simply nothing deployed to be
      stale *against*.
    - ``UNREADABLE_PICKLE``       — the file exists but could not be
      unpickled (torn write, or an MLPRegressor pickle on a host without
      sklearn). Cannot prove staleness ⇒ not a STALE alarm.
    - ``LSTSQ_FALLBACK``          — deployed model is the numpy lstsq
      fallback; the MLP config does not apply.
    - ``DEPLOYED_MATCHES_SOURCE`` — every audited key equals `MLP_CONFIG`.
      The gate is acting on the architecture the source endorses.
    - ``DEPLOYED_STALE_CONFIG``   — at least one audited key differs. The
      ``mismatches`` list names every drifted key with its deployed vs
      expected value (the operator-actionable payload — "restart the loop").

    ``deployed`` / ``expected`` echo the per-key values so the dashboard /
    a reading quant sees the exact `(64,32,16)` vs `(32,16)` drift, not just
    a boolean.
    """
    out: dict = {
        "verdict": "INSUFFICIENT_DATA",
        "deployed": None,
        "expected": dict(expected_config) if isinstance(expected_config, dict)
        else {},
        "mismatches": [],
        "n_audited": 0,
        "n_mismatched": 0,
        "hint": "",
    }
    try:
        p = Path(scorer_path)
        if not p.exists():
            out["hint"] = "no deployed pickle (untrained loop / fresh checkout)"
            return out
        try:
            with p.open("rb") as fh:
                state = pickle.load(fh)
            model = state["model"] if isinstance(state, dict) else state
        except Exception as exc:
            out["verdict"] = "UNREADABLE_PICKLE"
            out["hint"] = (f"pickle unreadable ({type(exc).__name__}) — "
                           "cannot prove staleness, not flagging STALE")
            return out

        if _is_lstsq_fallback(model):
            out["verdict"] = "LSTSQ_FALLBACK"
            out["hint"] = ("numpy lstsq fallback deployed (sklearn-absent "
                           "host) — MLP config not applicable")
            return out

        if not isinstance(expected_config, dict) or not expected_config:
            out["hint"] = "expected_config empty — nothing to audit"
            return out

        deployed: dict = {}
        mismatches: list[dict] = []
        for key, exp in expected_config.items():
            act = getattr(model, key, _MISSING)
            deployed[key] = None if act is _MISSING else act
            if not _values_match(exp, act):
                mismatches.append({
                    "key": key,
                    "deployed": None if act is _MISSING else act,
                    "expected": exp,
                })
        out["deployed"] = deployed
        out["n_audited"] = len(expected_config)
        out["n_mismatched"] = len(mismatches)
        out["mismatches"] = mismatches
        if mismatches:
            out["verdict"] = "DEPLOYED_STALE_CONFIG"
            drift = ", ".join(
                f"{m['key']}={m['deployed']!r}≠{m['expected']!r}"
                for m in mismatches
            )
            out["hint"] = (f"{len(mismatches)}/{len(expected_config)} "
                           f"hyper-params drifted ({drift}) — the running "
                           "loop predates the retune; restart "
                           "run_continuous_backtests.py to redeploy")
        else:
            out["verdict"] = "DEPLOYED_MATCHES_SOURCE"
            out["hint"] = (f"all {len(expected_config)} audited hyper-params "
                           "match source MLP_CONFIG")
        return out
    except Exception as exc:  # pragma: no cover - belt & braces
        return {
            "verdict": "INSUFFICIENT_DATA",
            "deployed": None,
            "expected": {},
            "mismatches": [],
            "n_audited": 0,
            "n_mismatched": 0,
            "hint": f"audit error ({type(exc).__name__})",
        }


def is_deploy_stale(scorer_path: Path | str | None = None,
                    expected_config: dict | None = None) -> bool | None:
    """Convenience boolean for the per-cycle scorer-skill ledger.

    Returns ``True`` only on a proven ``DEPLOYED_STALE_CONFIG``, ``False``
    on a proven match, and ``None`` for every can't-tell verdict
    (insufficient / unreadable / lstsq fallback) — so a ledger row records
    ``deploy_stale=None`` honestly rather than a misleading concrete
    ``False`` when staleness is simply unknowable. Never raises.
    """
    try:
        from .decision_scorer import SCORER_PATH, MLP_CONFIG
        rep = audit_deployed_config(
            scorer_path if scorer_path is not None else SCORER_PATH,
            expected_config if expected_config is not None else MLP_CONFIG,
        )
        v = rep.get("verdict")
        if v == "DEPLOYED_STALE_CONFIG":
            return True
        if v == "DEPLOYED_MATCHES_SOURCE":
            return False
        return None
    except Exception:
        return None


def analyze(scorer_path: Path | str | None = None) -> dict:
    """Full report for the live pickle vs the source `MLP_CONFIG`.

    Resolves both defaults from `decision_scorer` at call time (the
    AGENTS.md call-time-resolution rule) so a test can redirect
    `SCORER_PATH` / pass a synthetic config. Never raises.
    """
    try:
        from .decision_scorer import SCORER_PATH, MLP_CONFIG
    except Exception as exc:
        return {
            "verdict": "INSUFFICIENT_DATA", "deployed": None, "expected": {},
            "mismatches": [], "n_audited": 0, "n_mismatched": 0,
            "hint": f"decision_scorer import failed ({type(exc).__name__})",
        }
    return audit_deployed_config(
        scorer_path if scorer_path is not None else SCORER_PATH, MLP_CONFIG
    )


def _cli() -> int:
    """`python3 -m paper_trader.ml.deploy_audit` — read-only check that the
    deployed scorer pickle's architecture matches the source `MLP_CONFIG`.

    Exit code mirrors the sibling diagnostics so a cron / supervisor can
    branch on "the gate is acting on a stale net": 2 on
    DEPLOYED_STALE_CONFIG, 0 on everything else (match / insufficient /
    unreadable / lstsq fallback — none of which is a proven regression)."""
    rep = analyze()
    print(f"VERDICT: {rep['verdict']}")
    print(f"  {rep['hint']}")
    if rep["deployed"] is not None:
        print(f"  audited {rep['n_audited']} keys, "
              f"{rep['n_mismatched']} drifted")
        for m in rep["mismatches"]:
            print(f"    {m['key']}: deployed={m['deployed']!r}  "
                  f"expected={m['expected']!r}")
    return 2 if rep["verdict"] == "DEPLOYED_STALE_CONFIG" else 0


if __name__ == "__main__":
    raise SystemExit(_cli())
