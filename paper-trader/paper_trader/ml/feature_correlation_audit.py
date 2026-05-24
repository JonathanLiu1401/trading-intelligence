"""Feature correlation / multicollinearity audit — read-only.

The natural quant question after ``linear_probe`` reports
``LINEAR_HEAD_NO_BETTER_THAN_TRIVIAL`` and ``feature_importance`` reports
which numeric inputs the MLP leans on:

  *The scorer pulls 10 numeric features (ml_score, rsi, macd, mom5,
  mom20, regime_mult, vol_ratio, bb_pos, news_urgency,
  news_article_count). Are these 10 INDEPENDENT inputs, or are several
  effectively the same signal in different units — leaving the model
  with only a handful of true degrees of freedom?*

If e.g. mom5 and mom20 are highly correlated, the model's effective
input dimensionality is closer to 9 than 10 — and the
``MLP_NO_BETTER_THAN_TRIVIAL`` verdict the existing analyzers report has
a structural cause separate from architecture / training data: the
feature set itself carries redundant degrees of freedom. This module is
the only analyzer that answers that head-on.

| Verdict | Meaning |
|---------|---------|
| ``INSUFFICIENT_DATA`` | < ``MIN_OBS`` rows carry finite values across all 10 numeric features (the joint-complete count) — pairwise Spearman is unreliable below this floor. |
| ``LOW_COLLINEARITY`` | every pair has abs(spearman) < ``MODERATE_THR`` AND every VIF < 5. |
| ``MODERATE_COLLINEARITY`` | some pair has abs(spearman) in [``MODERATE_THR``, ``SEVERE_THR``) OR some VIF in [5, 10). |
| ``SEVERE_COLLINEARITY`` | some pair has abs(spearman) >= ``SEVERE_THR`` OR some VIF >= 10. |

**Why this is not any existing tool.** ``baseline_compare`` asks whether
one-feature rules beat the MLP. ``linear_probe`` asks whether a linear
head beats the MLP. ``feature_importance`` asks which feature the MLP
*uses*. ``corpus_diversity`` asks how many distinct training rows
survive dedup. **None of them measure cross-input redundancy in the
feature SPACE itself.** A `MULTICOLLINEAR` reading here is the
specific quant-actionable explanation for why a 10-feature MLP cannot
out-rank a 1-feature one-liner: there is not actually 10 features of
signal in the inputs.

This is a **read-only diagnostic**. It never trains, never touches
``decision_scorer.pkl``, ``decision_outcomes.jsonl``,
``build_features``, ``N_FEATURES``, or any trade path — the same
operational discipline as ``calibration.py`` / ``baseline_compare.py``
/ ``linear_probe.py``.

```bash
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.feature_correlation_audit
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.feature_correlation_audit --json
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.feature_correlation_audit --oos
```
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np


# The 10 numeric features the scorer consumes (mirrors the first 10 entries
# of ``decision_scorer.FEATURE_NAMES`` — the 7 sector one-hots are excluded
# because the correlation matrix of categorical dummies is degenerate by
# construction and would dominate every verdict).
NUMERIC_FEATURES = [
    "ml_score", "rsi", "macd", "mom5", "mom20", "regime_mult",
    "vol_ratio", "bb_position", "news_urgency", "news_article_count",
]

# Minimum row count before any verdict is produced. Spearman is unstable
# below ~30 paired observations; 50 is the same floor calibration.py uses.
MIN_OBS = 50

# Threshold for pairwise |spearman| above which a pair is "moderately
# correlated" (one or more pairs => MODERATE verdict). 0.5 is the
# textbook "strong" magnitude — anything beyond is redundancy worth
# flagging to a quant.
MODERATE_THR = 0.5

# Threshold above which a pair is "severely correlated". 0.8 is the
# textbook "very strong" magnitude — at this level two features are
# carrying ~64% shared variance and the MLP can't separate them in any
# meaningful way.
SEVERE_THR = 0.8

# VIF thresholds — textbook econometrics defaults. VIF >= 10 means the
# focal feature's variance is inflated 10× by collinearity with the
# others, which conventionally indicates "severe multicollinearity".
VIF_MODERATE = 5.0
VIF_SEVERE = 10.0


def _finite_float(v) -> float | None:
    """Like ``decision_scorer._to_float`` but returns None on missing/bad
    so callers can DROP rows rather than coerce to a fake default value
    (which would fabricate near-zero variance and inflate Spearman ties)."""
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _iter_rows(path: Path) -> Iterable[dict]:
    """Yield parsed dicts from a JSONL file; skip malformed lines silently."""
    try:
        with path.open() as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    yield json.loads(ln)
                except Exception:
                    continue
    except (OSError, ValueError):
        return


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average-rank rank transform (ties get the mean rank). Mirrors
    ``calibration._rankdata``; duplicated here to keep this module
    self-contained (the analyzer is invoked by hand and must not pull in
    sklearn or scipy that may not be available on every host)."""
    a = np.asarray(a, dtype=np.float64)
    order = a.argsort()
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1, dtype=np.float64)
    # Tie averaging — for any group of equal values, replace their ranks
    # with the mean of the group's ranks. Sort-then-scan is O(n log n).
    s = a[order]
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and s[j] == s[i]:
            j += 1
        if j - i > 1:
            mean_rank = (i + j + 1) / 2.0  # mean of (i+1 .. j) inclusive
            ranks[order[i:j]] = mean_rank
        i = j
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Tie-aware Spearman; returns 0.0 when either side has zero variance.
    Self-contained so this module never has to import ``calibration``."""
    if len(a) < 2:
        return 0.0
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return 0.0
    ar = _rankdata(a)
    br = _rankdata(b)
    if ar.std() == 0.0 or br.std() == 0.0:
        return 0.0
    return float(np.corrcoef(ar, br)[0, 1])


def _vif_via_ols(X: np.ndarray) -> list[float | None]:
    """Per-column Variance Inflation Factor via closed-form OLS.

    For each column j: VIF_j = 1 / (1 - R²_j) where R²_j is the
    coefficient of determination from regressing column j on all other
    columns. Standard textbook formula — no sklearn dep.

    Returns ``None`` for any column whose OLS R² hit numerical issues
    (degenerate covariance, all-equal column, NaN). VIF >= 10 is the
    conventional "severe multicollinearity" threshold.
    """
    n_cols = X.shape[1]
    vifs: list[float | None] = []
    for j in range(n_cols):
        y = X[:, j]
        if y.std() == 0.0:
            vifs.append(None)
            continue
        Xo = np.delete(X, j, axis=1)
        if Xo.shape[1] == 0:
            # Only one feature total — no others to regress on.
            vifs.append(None)
            continue
        # Add intercept column for proper centering.
        Xa = np.hstack([Xo, np.ones((Xo.shape[0], 1))])
        try:
            beta, *_ = np.linalg.lstsq(Xa, y, rcond=None)
            y_hat = Xa @ beta
            ss_res = float(((y - y_hat) ** 2).sum())
            ss_tot = float(((y - y.mean()) ** 2).sum())
            if ss_tot <= 0:
                vifs.append(None)
                continue
            r2 = 1.0 - ss_res / ss_tot
            # Numerical guard: R² can come back fractionally above 1 or
            # below 0 with poorly-conditioned designs.
            r2 = max(0.0, min(0.9999, r2))
            vifs.append(round(1.0 / (1.0 - r2), 3))
        except Exception:
            vifs.append(None)
    return vifs


def analyze(outcomes_path: "Path | str | None" = None,
            oos_only: bool = False) -> dict:
    """Compute pairwise Spearman + per-feature VIF over the 10 numeric
    scorer features.

    ``oos_only=True`` runs against the temporal holdout via
    ``validation.split_outcomes_temporal`` (the trustworthy generalization
    slice; same default as ``baseline_compare``). ``False`` uses the full
    accumulated corpus.

    Returns a JSON-safe dict:
    ``{status, verdict, n, slice, pairs:[{a,b,spearman}], vifs:[...],
       max_abs_corr, max_vif, hint}``.

    The ``pairs`` list is sorted by descending ``|spearman|`` so a quant
    reads the most-correlated pair first. ``vifs`` aligns with
    ``NUMERIC_FEATURES`` index by index. Never raises — every fault
    path degrades to an ``INSUFFICIENT_DATA`` row.
    """
    if outcomes_path is None:
        outcomes_path = (Path(__file__).resolve().parent.parent.parent
                         / "data" / "decision_outcomes.jsonl")
    path = Path(outcomes_path)

    rows = list(_iter_rows(path))
    slice_label = "full"
    if oos_only and rows:
        try:
            from paper_trader.validation import split_outcomes_temporal
            _, rows = split_outcomes_temporal(rows, oos_fraction=0.2)
            slice_label = "temporal_oos"
        except Exception as exc:
            return {
                "status": "error", "verdict": "INSUFFICIENT_DATA",
                "n": 0, "slice": "temporal_oos",
                "pairs": [], "vifs": [],
                "max_abs_corr": None, "max_vif": None,
                "hint": f"temporal split unavailable: {type(exc).__name__}",
            }

    # Build the joint-complete matrix: only rows where ALL 10 features
    # are finite. Pairwise Spearman on a column with mixed lengths
    # would inflate sample size for some pairs and not others, biasing
    # cross-pair comparison. The joint floor is the honest count.
    matrix: list[list[float]] = []
    for r in rows:
        vals: list[float | None] = []
        for fname in NUMERIC_FEATURES:
            # The outcome JSONL uses ``bb_position`` (the legacy spelling)
            # not ``bb_pos`` (the build_features kwarg). Honor both.
            key = fname
            v = _finite_float(r.get(key))
            vals.append(v)
        if all(v is not None for v in vals):
            matrix.append([float(v) for v in vals])  # type: ignore[arg-type]

    n = len(matrix)
    if n < MIN_OBS:
        return {
            "status": "insufficient_data",
            "verdict": "INSUFFICIENT_DATA",
            "n": n, "slice": slice_label,
            "pairs": [], "vifs": [],
            "max_abs_corr": None, "max_vif": None,
            "hint": (f"only {n} joint-complete rows; need ≥{MIN_OBS}. "
                     "Most BUYs lack one of news_urgency / vol_ratio / bb_position."),
        }

    X = np.asarray(matrix, dtype=np.float64)

    pairs: list[dict] = []
    max_abs = 0.0
    for i in range(len(NUMERIC_FEATURES)):
        for j in range(i + 1, len(NUMERIC_FEATURES)):
            rho = _spearman(X[:, i], X[:, j])
            if math.isfinite(rho):
                pairs.append({
                    "a": NUMERIC_FEATURES[i],
                    "b": NUMERIC_FEATURES[j],
                    "spearman": round(rho, 4),
                })
                if abs(rho) > max_abs:
                    max_abs = abs(rho)

    pairs.sort(key=lambda p: -abs(p["spearman"]))

    vifs_raw = _vif_via_ols(X)
    vifs = [
        {"feature": NUMERIC_FEATURES[i], "vif": vifs_raw[i]}
        for i in range(len(NUMERIC_FEATURES))
    ]
    finite_vifs = [v for v in vifs_raw if v is not None]
    max_vif = max(finite_vifs) if finite_vifs else None

    # Verdict — driven by the worst of the two views.
    severe_pair = max_abs >= SEVERE_THR
    moderate_pair = max_abs >= MODERATE_THR
    severe_vif = max_vif is not None and max_vif >= VIF_SEVERE
    moderate_vif = max_vif is not None and max_vif >= VIF_MODERATE

    if severe_pair or severe_vif:
        verdict = "SEVERE_COLLINEARITY"
        hint = (
            f"max |Spearman|={max_abs:.2f} (pair: {pairs[0]['a']}↔{pairs[0]['b']}), "
            f"max VIF={max_vif!s} — at least one feature pair is carrying "
            "≥64% shared variance OR a feature's variance is inflated 10×+ by "
            "redundancy. The 10-feature MLP has fewer than 10 effective inputs, "
            "which is a structural cause for the MLP_NO_BETTER_THAN_TRIVIAL "
            "verdict that is separate from the network architecture."
        )
    elif moderate_pair or moderate_vif:
        verdict = "MODERATE_COLLINEARITY"
        hint = (
            f"max |Spearman|={max_abs:.2f}, max VIF={max_vif!s} — some "
            "redundancy in the feature space. Worth noting but unlikely to "
            "dominate scorer skill alone."
        )
    else:
        verdict = "LOW_COLLINEARITY"
        hint = (
            f"max |Spearman|={max_abs:.2f}, max VIF={max_vif!s} — features "
            "carry independent signal. Whatever caps the scorer's OOS skill "
            "is NOT input redundancy."
        )

    return {
        "status": "ok",
        "verdict": verdict,
        "n": n,
        "slice": slice_label,
        "pairs": pairs,
        "vifs": vifs,
        "max_abs_corr": round(max_abs, 4),
        "max_vif": max_vif,
        "hint": hint,
    }


def _print_report(rep: dict) -> None:
    """Operator-readable table — top-3 most-correlated pairs + VIF list."""
    print(f"[feature_correlation_audit] status={rep.get('status')} "
          f"verdict={rep.get('verdict')} n={rep.get('n')} "
          f"slice={rep.get('slice')}")
    pairs = rep.get("pairs") or []
    if pairs:
        print(f"  max |Spearman| = {rep.get('max_abs_corr')}")
        print(f"  {'pair':<32} spearman")
        for p in pairs[:3]:
            print(f"    {p['a']:<14}↔ {p['b']:<14} "
                  f"{p['spearman']:+.4f}")
    vifs = rep.get("vifs") or []
    if vifs:
        print(f"  max VIF = {rep.get('max_vif')}")
        print(f"  {'feature':<22}{'VIF':>10}")
        for v in vifs:
            vstr = "n/a" if v["vif"] is None else f"{v['vif']:.2f}"
            print(f"  {v['feature']:<22}{vstr:>10}")
    if rep.get("hint"):
        print(f"  hint: {rep['hint']}")


def _build_arg_parser():
    import argparse

    p = argparse.ArgumentParser(
        prog="python3 -m paper_trader.ml.feature_correlation_audit",
        description="Pairwise Spearman + VIF for the scorer's 10 numeric "
                    "features. Read-only — never trains or writes.",
    )
    p.add_argument("--oos", action="store_true",
                   help="Run on the temporal holdout (last 20%) instead of "
                        "the full corpus.")
    p.add_argument("--path", default=None,
                   help="Override the decision_outcomes.jsonl path.")
    p.add_argument("--json", action="store_true",
                   help="Emit machine-readable JSON instead of a table.")
    return p


def main(argv: list[str] | None = None) -> int:
    import sys

    args = _build_arg_parser().parse_args(
        sys.argv[1:] if argv is None else argv)

    rep = analyze(outcomes_path=args.path, oos_only=args.oos)

    if args.json:
        print(json.dumps(rep, indent=2, sort_keys=True))
    else:
        _print_report(rep)

    # Exit code: 1 on SEVERE so a shell pipeline can gate on `$?` — same
    # convention as host_guard / decision_scorer CLI.
    return 1 if rep.get("verdict") == "SEVERE_COLLINEARITY" else 0


if __name__ == "__main__":
    raise SystemExit(main())
