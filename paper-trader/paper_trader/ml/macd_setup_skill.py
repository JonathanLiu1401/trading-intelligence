"""MACD enhanced-feature skill diagnostic — do the three boolean
``build_features`` slots the live trader actually emits
(``ema200_above`` / ``hist_cross_up`` / ``macd_below_zero_cross``)
carry **univariate** rank skill on realized 5-day forward returns?

Read-only diagnostic. Mirrors ``wk52_skill.py`` discipline exactly:
never trains, never writes a pickle, never touches ``build_features``
/ ``N_FEATURES``, never modifies trade-path state — safe to run
against the unattended continuous loop. A fault degrades to
``status='error'`` + a verdict; this module never raises (the
AGENTS.md "ledger / diagnostic must not break the cycle" discipline).

**The question this answers, which no existing analyzer in
``paper_trader/ml/`` does:**

The 2026-05 enhancement to ``build_features`` added three boolean
slots tracking textbook MACD setups:

* ``ema200_above`` — close above the 200-day EMA (long-term-trend filter)
* ``hist_cross_up`` — MACD histogram crossed from negative to positive
  on the most recent bar (classic bullish momentum-change signal)
* ``macd_below_zero_cross`` — ``hist_cross_up`` AND the MACD line is
  still below zero (the textbook "early-reversal" setup: the bullish
  cross-up confirms before price has fully recovered)

The DecisionScorer learns from these as numeric inputs alongside RSI /
mom5 / mom20, BUT the standalone univariate questions — *does each of
these booleans, by ITSELF, predict positive forward return?* — have
never been measured. ``baseline_compare.py`` baselines 11 univariate
rules but NONE of the 3 enhanced MACD booleans. Worse, the deployed
scorer's first-layer weights for these slots are documented as
near-zero (the dead-feature-audit symptom): if the slots are dead it's
because the captured signal is genuinely useless, OR because the
trainer couldn't pick it up from the joint feature space.

This analyzer answers, on the SAME corpus the scorer trains on
(``data/decision_outcomes.jsonl``):

* For each of (ema200_above, hist_cross_up, macd_below_zero_cross,
  combined_setup), the realized 5d return MEAN when the flag is True
  vs when it's False — the most directly interpretable
  "this signal carries information / does not" answer.
* Rank-IC of the flag (encoded 1/0) vs realized return — same
  ``_spearman`` as every sibling diagnostic, so cross-tool
  comparability is preserved by construction.
* Counts on each side (``n_true`` / ``n_false``) so a reader can spot
  imbalanced corpora that make the gap unreliable.

**Verdict ladder** (per flag, threshold-driven, test-locked):

| Verdict | Trigger |
|---------|---------|
| ``INSUFFICIENT_DATA`` | Either bucket < ``MIN_BUCKET`` rows, OR total < ``MIN_PAIRS`` |
| ``SETUP_PREDICTS_UP`` | ``mean_true − mean_false ≥ MEAN_GAP_GOOD_PCT`` AND ``spearman ≥ SPEARMAN_GOOD`` — the flag carries a positive, economically meaningful univariate edge |
| ``SETUP_PREDICTS_DOWN`` | ``mean_true − mean_false ≤ -MEAN_GAP_GOOD_PCT`` AND ``spearman ≤ -SPEARMAN_GOOD`` — the flag is **anti-predictive** (a contrarian researcher would short on it) |
| ``SETUP_NO_SKILL`` | ``|mean_gap| < MEAN_GAP_TOL_PCT`` AND ``|spearman| < SPEARMAN_FLAT`` — the flag carries no rank or magnitude information |
| ``DIRECTIONAL_*`` | Weak directional verdicts when the spearman fires but the gap doesn't, or vice versa — see the implementation for the exact boundaries |

The "combined_setup" key reports the conjunction (all 3 booleans True
simultaneously — the textbook "bullish-reversal-from-undersold"
setup). Often the most economically interesting answer because the
individual flags carry weaker signals than their interaction.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .calibration import _spearman

# Module-level constants — tests pin exact verdicts, and a tuning change
# is one reviewable edit. Mirrors `wk52_skill` discipline.
MIN_PAIRS = 30          # total observations needed
MIN_BUCKET = 5          # min count per True/False side for a verdict
SPEARMAN_FLAT = 0.05    # below |this| → no rank skill
SPEARMAN_GOOD = 0.10    # strong rank skill bar (lower than wk52 because
                        # a boolean feature naturally caps |spearman|
                        # lower than a continuous one)
MEAN_GAP_TOL_PCT = 0.5  # |mean_true - mean_false| < this is FLAT
MEAN_GAP_GOOD_PCT = 1.5  # |gap| ≥ this is strong evidence

# The four named flags this analyzer reports. The combined key is the
# logical AND of the three primitive booleans — the textbook "bullish
# reversal confirmed below zero with an uptrend filter" setup.
FLAG_KEYS = (
    "ema200_above",
    "hist_cross_up",
    "macd_below_zero_cross",
    "combined_setup",
)


def _to_finite_float(v):
    """Coerce to a finite float or None. Mirrors `wk52_skill` exactly so
    a sentinel-handling change in one module ripples cleanly through
    all of `paper_trader/ml/`."""
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return f


def _bool_to_int(v) -> int | None:
    """Coerce a stored flag to ``1`` / ``0`` / ``None``.

    The outcome rows store booleans as JSON ``true`` / ``false`` (which
    round-trip to Python ``bool``) but **legacy** rows occasionally
    stored ``0`` / ``1`` (int) or — for tickers whose technical
    window had insufficient history — ``null``. Tolerate all three
    shapes; drop only the explicit None / unparseable.

    Returns:
    * ``1`` if the value is truthy (True, 1, 1.0, ...)
    * ``0`` if the value is exactly False / 0 / 0.0
    * ``None`` if the value is None / NaN / unparseable — drop the row
    """
    if v is None:
        return None
    if v is True:
        return 1
    if v is False:
        return 0
    # Numeric coercion. Reject NaN/Inf and non-numeric strings.
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return 1 if f > 0.0 else 0


def _flag_value(rec: dict, key: str) -> int | None:
    """Extract one flag value from an outcome record.

    For the three primitive keys the lookup is direct. For
    ``combined_setup`` the value is the logical AND of all three
    primitives — None if ANY primitive is None (so we don't fabricate
    a "False" verdict from a missing field).
    """
    if key != "combined_setup":
        return _bool_to_int(rec.get(key))
    a = _bool_to_int(rec.get("ema200_above"))
    b = _bool_to_int(rec.get("hist_cross_up"))
    c = _bool_to_int(rec.get("macd_below_zero_cross"))
    if a is None or b is None or c is None:
        return None
    return 1 if (a == 1 and b == 1 and c == 1) else 0


def _verdict_for(spearman: float, mean_gap: float, n_true: int,
                 n_false: int) -> tuple[str, str]:
    """Pick the verdict + hint string for one flag.

    Order matters: strong-evidence verdicts come first so the operator
    sees the decisive call when one exists; directional verdicts only
    fire when the strong bar is missed.
    """
    if n_true < MIN_BUCKET or n_false < MIN_BUCKET:
        return (
            "INSUFFICIENT_DATA",
            f"need ≥{MIN_BUCKET} rows on each side, "
            f"have n_true={n_true} n_false={n_false}",
        )
    if (mean_gap >= MEAN_GAP_GOOD_PCT and spearman >= SPEARMAN_GOOD):
        return (
            "SETUP_PREDICTS_UP",
            f"True flag realizes {mean_gap:+.2f}pp BETTER than False — "
            f"strong univariate edge (spearman {spearman:+.3f}).",
        )
    if (mean_gap <= -MEAN_GAP_GOOD_PCT and spearman <= -SPEARMAN_GOOD):
        return (
            "SETUP_PREDICTS_DOWN",
            f"True flag realizes {mean_gap:+.2f}pp WORSE than False — "
            f"anti-predictive (spearman {spearman:+.3f}).",
        )
    if (abs(mean_gap) < MEAN_GAP_TOL_PCT
            and abs(spearman) < SPEARMAN_FLAT):
        return (
            "SETUP_NO_SKILL",
            f"no univariate skill — mean gap {mean_gap:+.2f}pp, "
            f"spearman {spearman:+.3f} (both at noise).",
        )
    if spearman >= SPEARMAN_FLAT:
        return (
            "DIRECTIONAL_UP",
            f"weak positive correlation (spearman {spearman:+.3f}) — "
            f"some evidence the True flag carries upward edge, but below "
            f"strong bar (|spearman| ≥ {SPEARMAN_GOOD}).",
        )
    if spearman <= -SPEARMAN_FLAT:
        return (
            "DIRECTIONAL_DOWN",
            f"weak negative correlation (spearman {spearman:+.3f}) — "
            f"some evidence the True flag carries anti-predictive bias, "
            f"but below strong bar.",
        )
    # Spearman in [-FLAT, +FLAT] but gap > tolerance — inconsistent signals.
    return (
        "SETUP_NO_SKILL",
        f"mixed signal — spearman flat ({spearman:+.3f}) but mean gap "
        f"{mean_gap:+.2f}pp exceeds {MEAN_GAP_TOL_PCT}pp tolerance.",
    )


def build_macd_setup_skill(records) -> dict:
    """Compute per-flag univariate skill on captured outcome records.

    Selection criteria per flag (each flag tracks its own drop counts):

    * Record is a dict — drop otherwise.
    * Flag value parseable as 0/1 — drop None / NaN / non-bool / non-int.
    * ``forward_return_5d`` finite — drop non-finite outcomes.

    Returns a JSON-safe dict with one entry per ``FLAG_KEYS`` under
    ``flags``. ``status='ok'`` when at least one flag produced a non-
    INSUFFICIENT_DATA verdict; ``status='insufficient_data'`` when
    every flag dropped to that floor. The top-level ``verdict`` keys
    the most economically decisive call (the first verdict in this
    priority order across all flags):
    ``SETUP_PREDICTS_DOWN`` > ``SETUP_PREDICTS_UP`` >
    ``DIRECTIONAL_DOWN`` > ``DIRECTIONAL_UP`` > ``SETUP_NO_SKILL``
    > ``INSUFFICIENT_DATA``. (Anti-predictive wins because a quant
    cares more about an anti-signal than confirming one — it's a
    feature-removal cue.)
    """
    if not records:
        return _empty("no records supplied")

    flags_out: dict[str, dict] = {}
    for key in FLAG_KEYS:
        ys_true: list[float] = []
        ys_false: list[float] = []
        bools: list[int] = []
        ys_all: list[float] = []
        n_skip_flag = 0
        n_skip_return = 0
        for r in records:
            if not isinstance(r, dict):
                n_skip_flag += 1
                continue
            b = _flag_value(r, key)
            if b is None:
                n_skip_flag += 1
                continue
            y = _to_finite_float(r.get("forward_return_5d"))
            if y is None:
                n_skip_return += 1
                continue
            bools.append(b)
            ys_all.append(y)
            if b == 1:
                ys_true.append(y)
            else:
                ys_false.append(y)

        n = len(bools)
        n_true = len(ys_true)
        n_false = len(ys_false)
        if n < MIN_PAIRS or n_true < MIN_BUCKET or n_false < MIN_BUCKET:
            flags_out[key] = {
                "verdict": "INSUFFICIENT_DATA",
                "n": n,
                "n_true": n_true,
                "n_false": n_false,
                "n_dropped_flag": n_skip_flag,
                "n_dropped_return": n_skip_return,
                "mean_true": (round(float(np.mean(ys_true)), 4)
                              if ys_true else None),
                "mean_false": (round(float(np.mean(ys_false)), 4)
                               if ys_false else None),
                "mean_gap_pct": None,
                "spearman": None,
                "hint": (
                    f"need ≥{MIN_PAIRS} total AND ≥{MIN_BUCKET} on each "
                    f"side; have n={n} n_true={n_true} n_false={n_false}"
                ),
            }
            continue

        mean_true = float(np.mean(ys_true))
        mean_false = float(np.mean(ys_false))
        mean_gap = mean_true - mean_false
        sp = _spearman(
            np.asarray(bools, dtype=np.float64),
            np.asarray(ys_all, dtype=np.float64),
        )
        # `_spearman` returns NaN when one vector has zero variance —
        # impossible here since we have ≥MIN_BUCKET on each side, but
        # guard defensively to keep the JSON contract finite.
        if not np.isfinite(sp):
            sp = 0.0
        verdict, hint = _verdict_for(float(sp), mean_gap, n_true, n_false)
        flags_out[key] = {
            "verdict": verdict,
            "n": n,
            "n_true": n_true,
            "n_false": n_false,
            "n_dropped_flag": n_skip_flag,
            "n_dropped_return": n_skip_return,
            "mean_true": round(mean_true, 4),
            "mean_false": round(mean_false, 4),
            "mean_gap_pct": round(mean_gap, 4),
            "spearman": round(float(sp), 4),
            "hint": hint,
        }

    # Top-level rollup verdict — priority order documented in the
    # docstring above.
    priority = (
        "SETUP_PREDICTS_DOWN",
        "SETUP_PREDICTS_UP",
        "DIRECTIONAL_DOWN",
        "DIRECTIONAL_UP",
        "SETUP_NO_SKILL",
        "INSUFFICIENT_DATA",
    )
    top_verdict = "INSUFFICIENT_DATA"
    for cand in priority:
        if any(v["verdict"] == cand for v in flags_out.values()):
            top_verdict = cand
            break
    # status is 'ok' when at least one flag carries a non-INSUFFICIENT_DATA
    # verdict; otherwise it's 'insufficient_data' so the per-cycle ledger
    # row can be honest about a corpus that's too small to score any
    # flag yet.
    status = "ok" if top_verdict != "INSUFFICIENT_DATA" else "insufficient_data"

    return {
        "status": status,
        "verdict": top_verdict,
        "flags": flags_out,
        "n_records": len(records),
        "hint": (
            f"top-level verdict picked by priority — see flags[] for the "
            f"per-flag breakdown ({len(FLAG_KEYS)} flags audited)."
        ),
    }


def _empty(reason: str) -> dict:
    """Honest-empty payload — every flag entry exists with
    ``INSUFFICIENT_DATA`` so JSON consumers don't need to special-case
    a missing key. Mirrors ``wk52_skill._empty`` discipline."""
    return {
        "status": "insufficient_data",
        "verdict": "INSUFFICIENT_DATA",
        "flags": {key: {
            "verdict": "INSUFFICIENT_DATA",
            "n": 0,
            "n_true": 0,
            "n_false": 0,
            "n_dropped_flag": 0,
            "n_dropped_return": 0,
            "mean_true": None,
            "mean_false": None,
            "mean_gap_pct": None,
            "spearman": None,
            "hint": reason,
        } for key in FLAG_KEYS},
        "n_records": 0,
        "hint": reason,
    }


def load_outcomes(path: Path | str) -> list[dict]:
    """Stream-load outcome records. Returns ``[]`` on missing file or
    unparseable line — never raises (mirrors ``wk52_skill.load_outcomes``)."""
    p = Path(path)
    if not p.exists():
        return []
    out: list[dict] = []
    try:
        with p.open("r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        return []
    return out


def analyze(outcomes_path: Path | str) -> dict:
    """Convenience: load + build. The CLI uses this. Never raises — a
    fault in the loader degrades to ``insufficient_data``; a fault in
    the analyzer is caught and returned as ``status='error'`` so a shell
    caller can distinguish a real fault from INSUFFICIENT_DATA (the
    ``wk52_skill.analyze`` sibling discipline)."""
    try:
        recs = load_outcomes(outcomes_path)
        return build_macd_setup_skill(recs)
    except Exception as exc:
        out = _empty(f"analyze error: {type(exc).__name__}: {exc}")
        out["status"] = "error"
        return out


def _print_report(rep: dict) -> None:
    print(f"VERDICT: {rep['verdict']}  ({rep['hint']})")
    if rep.get("status") not in ("ok", "insufficient_data"):
        return
    print(f"  n_records={rep['n_records']}")
    print(f"  {'flag':<24}{'verdict':<22}{'n_true':>8}{'n_false':>8}"
          f"{'mean_true':>12}{'mean_false':>12}{'gap_pct':>10}"
          f"{'spearman':>10}")
    for key in FLAG_KEYS:
        f = rep["flags"][key]
        mt = f"{f['mean_true']:+.2f}%" if f.get("mean_true") is not None else "n/a"
        mf = f"{f['mean_false']:+.2f}%" if f.get("mean_false") is not None else "n/a"
        gp = (f"{f['mean_gap_pct']:+.2f}"
              if f.get("mean_gap_pct") is not None else "n/a")
        sp = (f"{f['spearman']:+.3f}"
              if f.get("spearman") is not None else "n/a")
        print(f"  {key:<24}{f['verdict']:<22}{f['n_true']:>8}{f['n_false']:>8}"
              f"{mt:>12}{mf:>12}{gp:>10}{sp:>10}")


def _cli(argv: list[str] | None = None) -> int:
    """``python3 -m paper_trader.ml.macd_setup_skill [--path P] [--json]`` —
    read-only verification of the enhanced MACD booleans' univariate
    skill. Exit code 0 on ``SETUP_PREDICTS_UP`` (at least one flag
    carries a positive economic edge); exit 2 on
    ``SETUP_PREDICTS_DOWN`` (at least one flag is anti-predictive —
    the quant-decisive "consider removing or inverting the feature"
    signal); exit 1 otherwise.
    """
    import argparse
    import sys

    p = argparse.ArgumentParser(
        prog="python3 -m paper_trader.ml.macd_setup_skill",
        description=(
            "MACD enhanced-feature univariate skill diagnostic. Reports "
            "the realized 5d return mean of trades with each of "
            "ema200_above / hist_cross_up / macd_below_zero_cross set "
            "True vs False, plus their conjunction. Read-only — never "
            "trains or writes."
        ),
    )
    p.add_argument("--path", default="data/decision_outcomes.jsonl",
                   help="Path to decision_outcomes.jsonl")
    p.add_argument("--json", action="store_true",
                   help="Emit machine-readable JSON instead of a table")
    args = p.parse_args(sys.argv[1:] if argv is None else argv)

    rep = analyze(args.path)
    if args.json:
        print(json.dumps(rep, indent=2, sort_keys=True))
    else:
        _print_report(rep)

    v = rep.get("verdict")
    if v == "SETUP_PREDICTS_UP":
        return 0
    if v == "SETUP_PREDICTS_DOWN":
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(_cli())
