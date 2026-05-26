"""Decile-granularity truth-aware realized return — gate edition.

This is a **read-only diagnostic**. It never trains, never touches
``decision_scorer.pkl``, ``decision_outcomes.jsonl``, ``build_features``,
``N_FEATURES``, or any trade path — same operational discipline as
``paper_trader/ml/gate_realized.py`` / ``gate_audit.py`` / ``gate_pnl.py`` /
``calibration.py``. Safe to run against the live unattended loop.

**Why this is not** ``gate_realized.py``\\ **.** The sibling tool buckets
realized returns by the gate's five *acted arms* (``strong_headwind``
``×0.60``, ``mild_headwind`` ``×0.85``, ``neutral`` ``×1.00``,
``mild_tailwind`` ``×1.15``, ``strong_tailwind`` ``×1.30``). Those
boundaries are the gate's *decision* surface, but they are ARBITRARY
percentage cuts (±10 / ±5 / 0) — not sample quantiles. Two pathologies
follow:

  1. ``strong_headwind`` is everything below −10%, but in the live corpus
     ~10% of captured rows are concentrated at the clamp boundary (−50%),
     dwarfing the −12..−10% intermediate signal inside that one bucket.
  2. The five-arm view smooths over the *shape* of the truth-aware
     realized curve. A scorer can have a single anti-predictive extreme
     decile (a few large losing bets the model is most confident in) while
     middle deciles carry healthy ordering — the five-arm spread averages
     that signal out.

**What this answers.** Bucket every captured BUY by its decision-time
``gate_scorer_pred`` percentile (sample-quantile, 10 equally-populated
deciles), report each decile's realized 5d forward return mean and
sample-mean CI, and verdict on the *shape* of the curve:

| Verdict | Meaning |
|---|---|
| ``GATE_CAPTURE_NOT_YET_POPULATED`` | 0 rows carry ``gate_scorer_pred`` |
| ``INSUFFICIENT_DATA`` | acted rows < ``MIN_TOTAL`` or per-decile < ``MIN_PER_DECILE`` |
| ``MONOTONE_REALIZED`` | adjacent-decile means non-decreasing in ≥ ``MONOTONE_GOOD`` of steps AND last_decile − first_decile > ``EDGE_TOL_PP`` (the gate's prediction ordering really does rank realized outcome) |
| ``MOSTLY_MONOTONE`` | ≥ ``MONOTONE_MIN`` adjacent steps non-decreasing but spread under ``EDGE_TOL_PP`` (rank-skilled but small effect) |
| ``EXTREME_INVERSION`` | first OR last decile mean is anti-predictive vs the adjacent next-inner decile by more than ``EDGE_TOL_PP`` (the gate's MOST confident calls are the WORST — the documented live state on the deployed scorer's OOS slice) |
| ``NO_SHAPE`` | otherwise — random scatter |

**Honesty discipline.** The off-distribution-abstain bucket is reported
separately and **excluded from the verdict** (the ``gate_realized``
precedent). Decile boundaries are computed from the acted rows ONLY
(NOT including the abstained set) so a heavy off-distribution mass at
±50 cannot fabricate boundary structure. SELL rows are dropped — the
gate is BUY-only by construction (the ``gate_scorer_pred`` capture
contract); a defensive ``-forward_return_5d`` flip is applied only for
the rare SELL row that smuggled through (matches ``train_scorer`` /
``gate_audit``).

CLI:

```bash
cd /home/zeph/trading-intelligence/paper-trader && \\
    python3 -m paper_trader.ml.gate_decile_realized
cd /home/zeph/trading-intelligence/paper-trader && \\
    python3 -m paper_trader.ml.gate_decile_realized --json
cd /home/zeph/trading-intelligence/paper-trader && \\
    python3 -m paper_trader.ml.gate_decile_realized --all  # full corpus, not just OOS
```
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Verdict thresholds — module-level so tests assert exact verdicts and a
# tuning change is a single reviewable edit (the gate_audit / gate_pnl
# / gate_realized convention).
N_DECILES = 10
MIN_TOTAL = 30          # need a real acted sample before any verdict (== gate_realized.MIN_TOTAL)
MIN_PER_DECILE = 5      # each decile needs at least this many rows for its mean to be stable
EDGE_TOL_PP = 1.0       # |first-last spread| band that reads as noise (== gate_audit.EDGE_TOL_PP)
MONOTONE_MIN = 0.60     # ≥60% of adjacent steps non-decreasing → mostly monotone
MONOTONE_GOOD = 0.80    # ≥80% → genuinely monotone (== ml.calibration.MONOTONE_GOOD)


def _f(v):
    """Finite float or None — same hardening class as gate_realized._f."""
    if v is None or isinstance(v, bool):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _ci(vals: list[float]) -> tuple[float | None, float | None]:
    """Sample-mean 95% CI (Gaussian, n>=2) — matches gate_realized._stats."""
    if len(vals) < 2:
        return None, None
    a = np.asarray(vals, dtype=np.float64)
    mu = float(a.mean())
    se = float(a.std(ddof=1) / np.sqrt(len(a)))
    return round(mu - 1.96 * se, 4), round(mu + 1.96 * se, 4)


def gate_decile_realized_report(rows) -> dict:
    """Decile-bucketed truth-aware realized return for captured gate decisions.

    ``rows`` — any iterable of ``decision_outcomes.jsonl``-shaped dicts.
    For each row:

    * ``gate_scorer_pred`` is the gate's true decision-time prediction;
      ``None`` ⇒ no gate decision on this row, dropped (matches
      ``gate_realized``).
    * ``gate_off_dist=True`` ⇒ the gate abstained (no multiplier
      applied); routed to the separate ``abstained`` bucket and
      excluded from decile-boundary computation AND the verdict.
    * ``forward_return_5d`` is the realized return; SELL rows are
      defensively flipped via ``-fwd`` (the gate is BUY-only so this is
      an honesty guard, not a common path).

    Decile boundaries are computed via ``numpy.quantile`` over the acted
    set's ``gate_scorer_pred`` values. Rows are then bucketed by
    boundary; ties at a boundary land in the lower decile (numpy's
    default ``np.searchsorted`` side='right' for ``digitize``).

    Returns a JSON-safe dict ``{"verdict", "n_captured", "n_acted",
    "n_abstained", "deciles": [{"i", "n", "mean_pred", "mean_realized",
    "ci_lo", "ci_hi", "boundary_lo", "boundary_hi"}, ...],
    "spread_pp", "monotone_fraction", "abstained": {...}}``.

    Never raises.
    """
    acted_preds: list[float] = []
    acted_real: list[float] = []
    abstained_real: list[float] = []
    n_captured = 0
    n_acted = 0
    n_abstained = 0
    n_skipped_no_5d = 0

    try:
        it = list(rows or [])
    except Exception:
        it = []

    for r in it:
        if not isinstance(r, dict):
            continue
        gp = _f(r.get("gate_scorer_pred"))
        if gp is None:
            continue
        n_captured += 1
        fv = _f(r.get("forward_return_5d"))
        if fv is None:
            n_skipped_no_5d += 1
            continue
        is_sell = str(r.get("action") or "BUY").upper() == "SELL"
        realized = -fv if is_sell else fv
        if bool(r.get("gate_off_dist")):
            n_abstained += 1
            abstained_real.append(realized)
            continue
        n_acted += 1
        acted_preds.append(gp)
        acted_real.append(realized)

    if n_captured == 0:
        return {
            "verdict": "GATE_CAPTURE_NOT_YET_POPULATED",
            "n_captured": 0, "n_acted": 0, "n_abstained": 0,
            "n_skipped_no_5d": 0,
            "deciles": [], "spread_pp": None, "monotone_fraction": None,
            "abstained": {"n": 0, "mean_realized": None,
                          "ci_lo": None, "ci_hi": None},
        }

    abstained_mean = (round(float(np.mean(abstained_real)), 4)
                      if abstained_real else None)
    abst_lo, abst_hi = _ci(abstained_real)
    abst_block = {
        "n": n_abstained,
        "mean_realized": abstained_mean,
        "ci_lo": abst_lo, "ci_hi": abst_hi,
    }

    if n_acted < MIN_TOTAL:
        return {
            "verdict": "INSUFFICIENT_DATA",
            "n_captured": n_captured, "n_acted": n_acted,
            "n_abstained": n_abstained,
            "n_skipped_no_5d": n_skipped_no_5d,
            "deciles": [], "spread_pp": None, "monotone_fraction": None,
            "abstained": abst_block,
        }

    preds_arr = np.asarray(acted_preds, dtype=np.float64)
    real_arr = np.asarray(acted_real, dtype=np.float64)
    # Decile boundaries from the acted set — 11 edges (0..100 percentile).
    # Equal-population deciles: each bucket holds ~n_acted/10 rows. Ties
    # at a boundary land in the LOWER decile (digitize with right=True).
    edges = np.quantile(preds_arr, np.linspace(0.0, 1.0, N_DECILES + 1))
    # np.digitize returns 1..N_DECILES+1; clamp the upper tail so the
    # maximum predicted value lands in decile N_DECILES, not N_DECILES+1.
    bucket = np.digitize(preds_arr, edges[1:-1], right=False)
    bucket = np.clip(bucket, 0, N_DECILES - 1)

    decile_rows: list[dict] = []
    decile_means: list[float | None] = []
    decile_ns: list[int] = []
    for i in range(N_DECILES):
        mask = bucket == i
        n_i = int(mask.sum())
        if n_i == 0:
            decile_rows.append({
                "i": i + 1, "n": 0,
                "mean_pred": None, "mean_realized": None,
                "ci_lo": None, "ci_hi": None,
                "boundary_lo": round(float(edges[i]), 4),
                "boundary_hi": round(float(edges[i + 1]), 4),
            })
            decile_means.append(None)
            decile_ns.append(0)
            continue
        mp = float(preds_arr[mask].mean())
        mr = float(real_arr[mask].mean())
        lo, hi = _ci(list(real_arr[mask]))
        decile_rows.append({
            "i": i + 1, "n": n_i,
            "mean_pred": round(mp, 4),
            "mean_realized": round(mr, 4),
            "ci_lo": lo, "ci_hi": hi,
            "boundary_lo": round(float(edges[i]), 4),
            "boundary_hi": round(float(edges[i + 1]), 4),
        })
        decile_means.append(mr)
        decile_ns.append(n_i)

    # Verdict: insufficient if any decile has < MIN_PER_DECILE.
    if any(n < MIN_PER_DECILE for n in decile_ns):
        verdict = "INSUFFICIENT_DATA"
        spread = None
        monotone_frac = None
    else:
        # Adjacent-step monotonicity — strictly the discipline used in
        # ml.calibration / scorer_learning_curve. Steps where consecutive
        # means are equal count as non-decreasing (≥, not >).
        nondec = 0
        steps = 0
        for i in range(1, N_DECILES):
            a, b = decile_means[i - 1], decile_means[i]
            if a is None or b is None:
                continue
            steps += 1
            if b >= a:
                nondec += 1
        monotone_frac = round(nondec / steps, 4) if steps else None
        spread = (decile_means[-1] - decile_means[0]
                  if decile_means[0] is not None and decile_means[-1] is not None
                  else None)
        spread_pp = round(spread, 4) if spread is not None else None
        # EXTREME_INVERSION — first or last bucket is anti-predictive
        # relative to its neighbor by more than EDGE_TOL_PP. Catches the
        # documented live state where the gate's MOST confident calls
        # (top/bottom decile) realize WORSE than the second-most.
        extreme_inv = False
        if (decile_means[0] is not None and decile_means[1] is not None
                and decile_means[0] > decile_means[1] + EDGE_TOL_PP):
            extreme_inv = True
        if (decile_means[-1] is not None and decile_means[-2] is not None
                and decile_means[-1] < decile_means[-2] - EDGE_TOL_PP):
            extreme_inv = True
        if extreme_inv:
            verdict = "EXTREME_INVERSION"
        elif (monotone_frac is not None
              and monotone_frac >= MONOTONE_GOOD
              and spread is not None and spread > EDGE_TOL_PP):
            verdict = "MONOTONE_REALIZED"
        elif (monotone_frac is not None
              and monotone_frac >= MONOTONE_MIN):
            verdict = "MOSTLY_MONOTONE"
        else:
            verdict = "NO_SHAPE"
        spread = spread_pp

    return {
        "verdict": verdict,
        "n_captured": n_captured, "n_acted": n_acted,
        "n_abstained": n_abstained,
        "n_skipped_no_5d": n_skipped_no_5d,
        "deciles": decile_rows,
        "spread_pp": spread,
        "monotone_fraction": monotone_frac,
        "abstained": abst_block,
    }


def _load_outcomes(use_all: bool) -> list[dict]:
    """Load outcomes from data/decision_outcomes.jsonl. With ``use_all=False``,
    return only the OOS slice (last 20%, matching the scorer ledger's
    temporal holdout). Never raises — returns ``[]`` on any fault."""
    p = Path(__file__).resolve().parents[2] / "data" / "decision_outcomes.jsonl"
    if not p.exists():
        return []
    rows: list[dict] = []
    try:
        with p.open() as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    rows.append(json.loads(ln))
                except Exception:
                    continue
    except Exception:
        return []
    if use_all:
        return rows
    try:
        from paper_trader.validation import split_outcomes_temporal
        _, oos = split_outcomes_temporal(rows, oos_fraction=0.2)
        return oos
    except Exception:
        # Defensive fallback to a manual 80/20 — never block the diagnostic.
        n = len(rows)
        return rows[int(n * 0.8):]


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys
    parser = argparse.ArgumentParser(
        prog="python3 -m paper_trader.ml.gate_decile_realized",
        description="Decile-granularity truth-aware realized-return view of "
                    "the conviction gate's captured then-deployed decisions.",
    )
    parser.add_argument("--json", action="store_true",
                        help="Emit machine-readable JSON instead of a table.")
    parser.add_argument("--all", action="store_true",
                        help="Use the FULL outcomes corpus (default: OOS holdout).")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    rows = _load_outcomes(use_all=args.all)
    rep = gate_decile_realized_report(rows)

    if args.json:
        print(json.dumps(rep, indent=2, sort_keys=True))
        return 0 if rep["verdict"] != "GATE_CAPTURE_NOT_YET_POPULATED" else 1

    slice_label = "FULL corpus" if args.all else "OOS holdout (last 20%)"
    print(f"[gate_decile_realized] {slice_label}")
    print(f"  verdict: {rep['verdict']}")
    print(f"  n_captured={rep['n_captured']} acted={rep['n_acted']} "
          f"abstained={rep['n_abstained']} skipped_no_5d={rep['n_skipped_no_5d']}")
    if rep["spread_pp"] is not None:
        print(f"  D10−D1 spread: {rep['spread_pp']:+.2f}pp  "
              f"(monotone fraction: {rep['monotone_fraction']})")
    if rep["deciles"]:
        print(f"  {'decile':<7}{'n':>6}{'boundary':>22}{'mean_pred':>12}"
              f"{'mean_real':>12}{'95% CI':>22}")
        for d in rep["deciles"]:
            b = f"[{d['boundary_lo']:+.2f}, {d['boundary_hi']:+.2f}]"
            if d["mean_pred"] is None:
                print(f"  D{d['i']:<6}{d['n']:>6}{b:>22}"
                      f"{'(empty)':>12}{'(empty)':>12}{'':>22}")
                continue
            ci = f"[{d['ci_lo']:+.2f}, {d['ci_hi']:+.2f}]" if d['ci_lo'] is not None else ""
            print(f"  D{d['i']:<6}{d['n']:>6}{b:>22}"
                  f"{d['mean_pred']:>+12.2f}{d['mean_realized']:>+12.2f}"
                  f"{ci:>22}")
    abst = rep["abstained"]
    if abst["n"]:
        print(f"  abstained (off-distribution): n={abst['n']}  "
              f"mean_realized={abst['mean_realized']:+.2f}%")
    return 0 if rep["verdict"] != "GATE_CAPTURE_NOT_YET_POPULATED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
