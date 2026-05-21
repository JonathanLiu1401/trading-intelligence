"""Sample-weight policy A/B audit for the DecisionScorer training pipeline.

This module answers: *"Does the deployed sample-weight policy
(run-quality × LLM-annotation) ADD or SUBTRACT out-of-sample rank skill
versus simpler alternatives?"* It is a non-destructive A/B: trains
scorers in-memory under N candidate weighting policies, evaluates each
on the SAME temporal-OOS holdout, and reports the policy with the
highest OOS rank-IC.

Read-only operational discipline mirrors every existing
``paper_trader.ml.*`` diagnostic (``baseline_compare`` /
``calibration`` / ``per_ticker_skill`` / ``feature_value_skill``):
**never writes the deployed pickle**, **never touches the trade
path**, **never raises on bad input** — safe under the live
unattended continuous loop.

**Why this is not any existing tool.** Every "skill" module in
``paper_trader/ml/`` (action_skill, persona_skill, sector_skill,
calibration, baseline_compare, …) measures the *deployed* scorer
against realized outcomes — they take the model AS-IS. None of them
tests the prior question a quant asks when an OOS metric stalls:
*"would a DIFFERENTLY-TRAINED scorer (different weighting) have
better skill on the same data?"*

The current ``train_scorer`` weighting is:

    weight = max(0.5, min(2.0, 1.0 + return_pct/200)) * llm_mult

where ``llm_mult`` ∈ {3.0 endorsed, 1.0 neutral, 0.1 condemned}.
That stacks two policy choices (run-quality bias + LLM annotation)
into one weight. If either choice is HURTING OOS skill, only an A/B
sweep can detect it — until now this required manually editing
``train_scorer`` and running a full cycle. This module makes it a
one-command CLI.

**Policies tested**, each a pure function of the outcome record:

* ``current`` — the deployed
  ``max(0.5, min(2.0, 1.0 + return_pct/200)) * llm_mult``
* ``uniform`` — every record weighted 1.0 (the textbook null
  hypothesis: weighting adds no skill)
* ``run_only`` — keep the run-quality clamp but drop the LLM annotation
  multiplier (isolates LLM annotation's effect)
* ``llm_only`` — keep llm_mult, drop the run-quality clamp (isolates
  the return_pct policy's effect)
* ``abs_label`` — weight by ``|forward_return_5d|`` (large-move
  decisions counted as more important by magnitude)

All policies share the same dedup, feature-vector build, train/val
split, SELL sign-flip, label clamp, ``MLP_CONFIG``, and OOS evaluator
as the production path — so the *only* thing this audit varies is the
weight column.

Verdicts (crisp, threshold-driven so they're exactly testable):

| Verdict             | Meaning                                           |
|---------------------|---------------------------------------------------|
| ``CURRENT_OPTIMAL`` | ``current`` beats every alternative by ≥ ``IC_TOL`` |
| ``CURRENT_TIED``    | top alternative beats ``current`` by < ``IC_TOL``  |
| ``CURRENT_DOMINATED`` | top alternative beats ``current`` by ≥ ``IC_TOL``  |
| ``INSUFFICIENT_DATA`` | < ``MIN_OOS_PAIRS`` OOS pairs for any policy     |

CLI:

* ``python3 -m paper_trader.ml.sample_weight_audit``           — text table
* ``python3 -m paper_trader.ml.sample_weight_audit --json``    — JSON
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from paper_trader.ml.decision_scorer import (
    MLP_CONFIG,
    PRED_CLAMP_PCT,
    _to_float,
    build_features,
)

# Thresholds at module scope so tests assert exact verdicts and a tuning
# change is a single, reviewable edit (mirrors the codebase convention).
MIN_OOS_PAIRS = 30      # need ≥30 OOS records for a meaningful rank-IC
IC_TOL = 0.02            # spread under which two policies are "tied" in OOS IC

# Same dedup discipline as `train_scorer` — key on (ticker, sim_date, action)
# so a BUY and SELL of the same ticker on the same day are NOT collapsed (they
# carry OPPOSITE labels after the universal SELL sign-flip). Keep the highest-
# `return_pct` copy when a collision exists.
def _dedup(records: list[dict]) -> list[dict]:
    seen: dict[tuple, dict] = {}
    for r in records:
        key = (
            str(r.get("ticker") or ""),
            str(r.get("sim_date") or ""),
            str(r.get("action") or "BUY").upper(),
        )
        rp = _to_float(r.get("return_pct"), 0.0)
        if key not in seen or rp > _to_float(seen[key].get("return_pct"), 0.0):
            seen[key] = r
    return list(seen.values())


# Each policy is a pure function `record -> weight float`. NaN / non-finite
# inputs degrade to the policy's safe-fallback weight (see comments per
# policy) rather than raising — the discipline `train_scorer` already uses
# for forward_return validation.
def _weight_current(r: dict) -> float:
    """The deployed weighting in `train_scorer`. Run-quality clamp times
    LLM annotation multiplier."""
    rp = _to_float(r.get("return_pct"), 0.0)
    llm = int(r.get("llm_quality_label") or 0)
    llm_mult = {1: 3.0, -1: 0.1, 0: 1.0}.get(llm, 1.0)
    return max(0.5, min(2.0, 1.0 + rp / 200.0)) * llm_mult


def _weight_uniform(_r: dict) -> float:
    """Null hypothesis: every record counted equally."""
    return 1.0


def _weight_run_only(r: dict) -> float:
    """Run-quality clamp without LLM annotation — isolates the
    return_pct policy's effect."""
    rp = _to_float(r.get("return_pct"), 0.0)
    return max(0.5, min(2.0, 1.0 + rp / 200.0))


def _weight_llm_only(r: dict) -> float:
    """LLM annotation alone — isolates the annotation policy's effect."""
    llm = int(r.get("llm_quality_label") or 0)
    return {1: 3.0, -1: 0.1, 0: 1.0}.get(llm, 1.0)


def _weight_abs_label(r: dict) -> float:
    """Weight by the absolute size of the realized move. A bigger move
    is treated as more informative (the model should pay more attention
    to clearly-moving examples than to noise)."""
    fr = r.get("forward_return_5d")
    try:
        fr_f = float(fr)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(fr_f):
        return 1.0
    # Scale: |0%| → 0.5x, |5%| → 1.0x, |20%| → 2.5x, clamped to [0.5, 3.0]
    return max(0.5, min(3.0, 0.5 + abs(fr_f) / 10.0))


POLICIES: dict[str, "Callable[[dict], float]"] = {
    "current": _weight_current,
    "uniform": _weight_uniform,
    "run_only": _weight_run_only,
    "llm_only": _weight_llm_only,
    "abs_label": _weight_abs_label,
}


def _build_xy(records: list[dict]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Build (features, labels, dropped) — labels carry the SELL sign-flip
    and the ±PRED_CLAMP_PCT label clamp `train_scorer` already applies, so
    the train/eval target space is byte-identical to the deployed path.
    Records whose forward_return_5d is missing/non-finite are dropped
    (`train_scorer`'s n_label_dropped discipline)."""
    X_raw: list[list[float]] = []
    y: list[float] = []
    kept_records: list[dict] = []
    for r in records:
        fr_raw = r.get("forward_return_5d")
        if isinstance(fr_raw, bool) or fr_raw is None:
            continue
        try:
            fr = float(fr_raw)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(fr):
            continue
        fr = max(-PRED_CLAMP_PCT, min(PRED_CLAMP_PCT, fr))
        action = str(r.get("action") or "BUY").upper()
        y_val = -fr if action == "SELL" else fr  # universal SELL sign-flip
        X_raw.append(build_features(
            _to_float(r.get("ml_score"), 0.0),
            r.get("rsi"), r.get("macd"),
            r.get("mom5"), r.get("mom20"),
            _to_float(r.get("regime_mult"), 1.0),
            str(r.get("ticker") or ""),
            vol_ratio=r.get("vol_ratio"), bb_pos=r.get("bb_position"),
            news_urgency=r.get("news_urgency"),
            news_article_count=r.get("news_article_count"),
        ))
        y.append(y_val)
        kept_records.append(r)
    if not X_raw:
        return (np.zeros((0, 1), dtype=np.float32),
                np.zeros(0, dtype=np.float32), [])
    return (np.array(X_raw, dtype=np.float32),
            np.array(y, dtype=np.float32), kept_records)


def _train_with_weights(X_tr: np.ndarray, y_tr: np.ndarray,
                         w_tr: np.ndarray):
    """In-memory training with arbitrary sample weights. Returns
    (model, scaler) or (None, None) on any failure. NEVER writes to disk —
    that's the load-bearing operational invariant for this audit.

    Uses the same `MLP_CONFIG` / oversampling pattern as `train_scorer` so
    the only difference between A/B branches is the weight vector. The
    numpy-lstsq fallback path mirrors `train_scorer`'s ImportError branch.
    """
    try:
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_s = scaler.fit_transform(X_tr)
        # Oversample: round(w*2) replicas per row — same convention as
        # `train_scorer`. Below-rounding (rare for w<0.25) drops the row.
        rep = np.round(w_tr * 2).astype(int)
        keep = rep > 0
        if keep.any():
            X_w = np.repeat(X_s[keep], rep[keep], axis=0)
            y_w = np.repeat(y_tr[keep], rep[keep], axis=0)
        else:
            X_w = X_s
            y_w = y_tr
        model = MLPRegressor(**MLP_CONFIG)
        model.fit(X_w, y_w)
        return model, scaler
    except ImportError:
        # numpy weighted lstsq fallback (sklearn-absent hosts).
        scaler_mean = X_tr.mean(axis=0)
        scaler_std = X_tr.std(axis=0) + 1e-8
        X_s = (X_tr - scaler_mean) / scaler_std
        X_aug = np.hstack([X_s, np.ones((len(X_s), 1), dtype=np.float32)])
        sw = np.sqrt(w_tr).astype(np.float32).reshape(-1, 1)
        w, *_ = np.linalg.lstsq(X_aug * sw, y_tr * sw.ravel(), rcond=None)

        # Pickle-safe stand-ins so the (model, scaler) pair has the same
        # `.predict(X) -> array` contract whether sklearn was available.
        class _LstsqScaler:
            def __init__(self, mean, std):
                self.mean_ = mean.astype(np.float32)
                self.std_ = std.astype(np.float32)

            def transform(self, X):
                X = np.asarray(X, dtype=np.float32)
                return (X - self.mean_) / self.std_

        class _LstsqModel:
            def __init__(self, weights):
                self.w_ = weights.astype(np.float32)

            def predict(self, X):
                X = np.asarray(X, dtype=np.float32)
                Xa = np.hstack([X, np.ones((len(X), 1), dtype=np.float32)])
                return Xa @ self.w_

        return _LstsqModel(w), _LstsqScaler(scaler_mean, scaler_std)
    except Exception:
        return None, None


def _rank_ic_and_diracc(preds: np.ndarray, actuals: np.ndarray) -> tuple:
    """Returns (rank_ic, dir_acc, n). Reuses `calibration._spearman` so this
    metric and every other rank-IC consumer share one source of truth
    (tie-aware Spearman — load-bearing because clamped ±50 predictions tie
    at the empirical label support). Direction accuracy excludes zero
    pairs to mirror `_oos_rank_metrics` / `action_skill`'s convention."""
    try:
        from paper_trader.ml.calibration import _spearman

        if len(preds) < 2:
            return None, None, len(preds)
        ic = _spearman(np.asarray(preds, dtype=float),
                       np.asarray(actuals, dtype=float))
        ic_v = None if ic != ic else round(float(ic), 4)
        dir_pairs = [(p, a) for p, a in zip(preds, actuals)
                     if p != 0.0 and a != 0.0]
        if dir_pairs:
            hits = sum(1 for p, a in dir_pairs if (p > 0) == (a > 0))
            return ic_v, round(hits / len(dir_pairs), 4), len(preds)
        return ic_v, None, len(preds)
    except Exception:
        return None, None, len(preds)


def _predict_batch(model, scaler, X: np.ndarray) -> np.ndarray | None:
    try:
        Xs = scaler.transform(X)
        raw = np.asarray(model.predict(Xs), dtype=np.float64)
        # Mirror predict_with_meta's clamp so this and the deployed path
        # see the same target space — rank metrics are nearly insensitive
        # to clamping (only extreme ties shift) but the alignment makes
        # cross-policy comparison apples-to-apples.
        raw = np.clip(raw, -PRED_CLAMP_PCT, PRED_CLAMP_PCT)
        return raw
    except Exception:
        return None


def analyze(outcomes_path: "Path | str", oos_fraction: float = 0.2,
            policies: "dict[str, Callable[[dict], float]] | None" = None
            ) -> dict:
    """A/B sweep across sample-weight policies on the temporal-OOS split.

    Args:
        outcomes_path: Path to ``decision_outcomes.jsonl``.
        oos_fraction: Tail fraction reserved for OOS evaluation. Matches
            ``_train_decision_scorer``'s 0.2 default so this audit's
            holdout is the SAME slice the scorer-skill ledger reports
            against (apples-to-apples cross-check).
        policies: Optional override of the policy dict. Default: ``POLICIES``
            (the five module-level policies).

    Returns:
        JSON-safe dict with::

            {
              "status": "ok" | "insufficient_data" | "error",
              "verdict": "CURRENT_OPTIMAL" | "CURRENT_TIED" |
                         "CURRENT_DOMINATED" | "INSUFFICIENT_DATA",
              "n_outcomes": int,
              "n_train": int,
              "n_oos": int,
              "policies": [
                {"name": str, "oos_rank_ic": float|None,
                 "oos_dir_acc": float|None,
                 "ic_vs_current": float|None,
                 "is_current": bool}, ...
              ],
              "best_policy": str|None,
              "best_ic": float|None,
              "current_ic": float|None,
              "hint": str|None,
            }
    """
    out = {
        "status": "error", "verdict": "INSUFFICIENT_DATA",
        "n_outcomes": 0, "n_train": 0, "n_oos": 0,
        "policies": [], "best_policy": None, "best_ic": None,
        "current_ic": None, "hint": None,
    }
    pols = policies if policies is not None else POLICIES
    if "current" not in pols:
        out["hint"] = "policies dict must include 'current'"
        return out

    # Read + parse outcomes (defensive — a malformed line drops, doesn't
    # crash, mirroring `_inject_and_train`'s discipline).
    path = Path(outcomes_path)
    if not path.exists():
        out["hint"] = f"outcomes file not found: {path}"
        return out
    records: list[dict] = []
    try:
        for ln in path.read_text().splitlines():
            if not ln.strip():
                continue
            try:
                records.append(json.loads(ln))
            except Exception:
                continue
    except Exception as exc:
        out["hint"] = f"outcomes read failed: {exc}"
        return out

    out["n_outcomes"] = len(records)
    if not records:
        return out

    # Temporal split FIRST, dedup AFTER. This mirrors the deployed split
    # used by `validation.split_outcomes_temporal` (sorts by sim_date,
    # tails the last 20%); the audit's OOS is therefore the SAME tail the
    # scorer-skill ledger sees.
    records.sort(key=lambda r: str(r.get("sim_date") or ""))
    split_idx = max(1, int(len(records) * (1.0 - oos_fraction)))
    train_records = _dedup(records[:split_idx])
    oos_records = _dedup(records[split_idx:])

    # Build OOS features once (predictions vary per policy, OOS labels
    # don't). Drop records whose forward_return is unusable.
    X_oos, y_oos, oos_kept = _build_xy(oos_records)
    out["n_oos"] = len(oos_kept)
    if out["n_oos"] < MIN_OOS_PAIRS:
        out["status"] = "insufficient_data"
        out["hint"] = (f"only {out['n_oos']} OOS pairs after dedup + "
                       f"label validation (need ≥{MIN_OOS_PAIRS})")
        return out

    # Build train features once and reuse the same X across policies; only
    # the weight vector varies.
    X_tr, y_tr, tr_kept = _build_xy(train_records)
    out["n_train"] = len(tr_kept)
    if out["n_train"] < MIN_OOS_PAIRS:
        out["status"] = "insufficient_data"
        out["hint"] = (f"only {out['n_train']} valid train records (need "
                       f"≥{MIN_OOS_PAIRS}); add more cycles before A/B")
        return out

    policy_rows: list[dict] = []
    current_ic: float | None = None
    for name, fn in pols.items():
        # Compute weights for the train fold under THIS policy.
        weights = np.array(
            [max(0.0, float(fn(r))) for r in tr_kept],
            dtype=np.float32,
        )
        model, scaler = _train_with_weights(X_tr, y_tr, weights)
        if model is None or scaler is None:
            policy_rows.append({
                "name": name, "oos_rank_ic": None, "oos_dir_acc": None,
                "ic_vs_current": None, "is_current": (name == "current"),
                "n_weights_nonzero": int((weights > 0).sum()),
            })
            continue
        preds = _predict_batch(model, scaler, X_oos)
        if preds is None:
            policy_rows.append({
                "name": name, "oos_rank_ic": None, "oos_dir_acc": None,
                "ic_vs_current": None, "is_current": (name == "current"),
                "n_weights_nonzero": int((weights > 0).sum()),
            })
            continue
        rank_ic, dir_acc, n = _rank_ic_and_diracc(preds, y_oos)
        if name == "current":
            current_ic = rank_ic
        policy_rows.append({
            "name": name, "oos_rank_ic": rank_ic, "oos_dir_acc": dir_acc,
            "is_current": (name == "current"),
            "n_weights_nonzero": int((weights > 0).sum()),
        })

    # Fill in `ic_vs_current` once we have `current_ic`.
    for row in policy_rows:
        cic = current_ic
        ric = row["oos_rank_ic"]
        if cic is None or ric is None:
            row["ic_vs_current"] = None
        else:
            row["ic_vs_current"] = round(ric - cic, 4)

    # Verdict: how does current compare to the best alternative?
    out["policies"] = policy_rows
    out["current_ic"] = current_ic
    valid_ics = [(r["name"], r["oos_rank_ic"]) for r in policy_rows
                 if r["oos_rank_ic"] is not None]
    if not valid_ics:
        out["status"] = "error"
        out["hint"] = "every policy failed to produce a finite rank-IC"
        return out
    best_name, best_ic = max(valid_ics, key=lambda nv: nv[1])
    out["best_policy"] = best_name
    out["best_ic"] = best_ic
    out["status"] = "ok"
    if current_ic is None:
        out["verdict"] = "CURRENT_DOMINATED"
        out["hint"] = "current policy failed to produce a finite rank-IC"
    elif best_name == "current":
        # Current ties with itself by definition; check 2nd best.
        runner_up = sorted(
            [nv for nv in valid_ics if nv[0] != "current"],
            key=lambda nv: nv[1], reverse=True,
        )
        gap = (best_ic - runner_up[0][1]) if runner_up else float("inf")
        if gap >= IC_TOL:
            out["verdict"] = "CURRENT_OPTIMAL"
        else:
            out["verdict"] = "CURRENT_TIED"
    else:
        gap = best_ic - current_ic
        if gap >= IC_TOL:
            out["verdict"] = "CURRENT_DOMINATED"
            out["hint"] = (f"{best_name} beats current by {gap:+.3f} "
                           f"OOS rank-IC — consider switching")
        else:
            out["verdict"] = "CURRENT_TIED"
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser():
    import argparse
    p = argparse.ArgumentParser(
        prog="python3 -m paper_trader.ml.sample_weight_audit",
        description=("A/B sweep across DecisionScorer sample-weight policies "
                     "on the temporal-OOS holdout. Read-only — never writes "
                     "the deployed pickle."),
    )
    p.add_argument("--outcomes-path", default=None,
                   help="Path to decision_outcomes.jsonl (default: "
                        "data/decision_outcomes.jsonl relative to repo root)")
    p.add_argument("--oos-fraction", type=float, default=0.2,
                   help="OOS holdout tail fraction (default 0.2, matches "
                        "_train_decision_scorer)")
    p.add_argument("--json", action="store_true",
                   help="Machine-readable JSON output instead of a table")
    return p


def main(argv=None) -> int:
    import sys
    args = _build_arg_parser().parse_args(
        sys.argv[1:] if argv is None else argv)
    if args.outcomes_path is None:
        # Resolve relative to repo root (the same path `_train_decision_scorer`
        # uses in `run_continuous_backtests.main`).
        from paper_trader.backtest import ROOT
        path = ROOT / "data" / "decision_outcomes.jsonl"
    else:
        path = Path(args.outcomes_path)

    rep = analyze(path, oos_fraction=args.oos_fraction)

    if args.json:
        print(json.dumps(rep, indent=2, sort_keys=True))
        return 0 if rep.get("status") == "ok" else 1

    print(f"[sample_weight_audit] status={rep.get('status')}  "
          f"verdict={rep.get('verdict')}  "
          f"n_outcomes={rep.get('n_outcomes')}  "
          f"n_train={rep.get('n_train')}  n_oos={rep.get('n_oos')}")
    if rep.get("hint"):
        print(f"  hint: {rep['hint']}")
    rows = rep.get("policies") or []
    if rows:
        print(f"  {'policy':<14}{'oos_rank_ic':>14}{'oos_dir_acc':>14}"
              f"{'vs_current':>14}{'is_current':>12}")
        for r in rows:
            ric = r.get("oos_rank_ic")
            da = r.get("oos_dir_acc")
            vc = r.get("ic_vs_current")
            ric_s = f"{ric:+.4f}" if isinstance(ric, (int, float)) else "n/a"
            da_s = f"{da:.4f}" if isinstance(da, (int, float)) else "n/a"
            vc_s = f"{vc:+.4f}" if isinstance(vc, (int, float)) else "n/a"
            cur_s = "YES" if r.get("is_current") else ""
            print(f"  {r['name']:<14}{ric_s:>14}{da_s:>14}"
                  f"{vc_s:>14}{cur_s:>12}")
    if rep.get("best_policy"):
        bi = rep.get("best_ic")
        bi_s = f"{bi:+.4f}" if isinstance(bi, (int, float)) else "n/a"
        print(f"  best policy: {rep['best_policy']}  (OOS rank-IC {bi_s})")
    return 0 if rep.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
