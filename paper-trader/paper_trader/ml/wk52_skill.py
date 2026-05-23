"""52-week-position realized-return diagnostic — does proximity to the
52-week high actually predict POORER realized 5-day return, as the
``_ml_decide`` bubble-top gate's premise assumes?

Read-only diagnostic. Mirrors ``conviction_calibration.py`` discipline
exactly: never trains, never writes a pickle, never touches
``build_features`` / ``N_FEATURES``, never modifies trade-path state —
safe to run against the unattended continuous loop (AGENTS.md
"Operational quirks a quant should know"). A fault degrades to
``status='error'`` + a verdict; this module never raises (the AGENTS.md
"ledger / diagnostic must not break the cycle" discipline).

**The question this answers, which no existing analyzer in
``paper_trader/ml/`` does:**

``backtest._ml_decide`` carries a 52-week-high gate that **suppresses**
BUYs when ``wk52_pos > 0.80`` (the bubble-top gate, AGENTS.md: "prevents
buying into bubble peaks where news clusters at market tops causing
underperformance"). The premise is testable from captured outcomes — the
``wk52_pos`` field has been persisted alongside every BUY since the
2026-05-21 outcome-schema extension. But **nothing reads it back**. A
skeptical quant deciding whether to keep, tighten, or remove that gate
needs:

* Does realized 5d return DECREASE monotonically as ``wk52_pos`` rises,
  the way the gate assumes? — measured directly per quintile bucket.
* What is the rank correlation (spearman) between ``wk52_pos`` and
  realized return — strictly negative would justify the gate, near-zero
  or positive would not.
* What is the realized mean of trades in the GATE'S OWN SUPPRESSION
  ZONE (``wk52_pos > 0.80``) vs the rest of the corpus? — this is the
  exact counterfactual: "what return did the trades the gate would have
  blocked actually deliver?"

The aggregate ``calibration`` / ``conviction_calibration`` numbers cannot
answer either question — they don't slice by ``wk52_pos`` at all. This
analyzer fills the gap.

**Verdict ladder** (crisp, threshold-driven, test-locked):

| Verdict | Trigger |
|---------|---------|
| ``INSUFFICIENT_DATA`` | < ``MIN_PAIRS`` BUY rows with ``wk52_pos`` AND ``forward_return_5d`` |
| ``BUBBLE_GATE_JUSTIFIED`` | Top bucket realized < bottom bucket by ≥ ``BUCKET_GAP_GOOD_PCT`` AND spearman ≤ -``SPEARMAN_GOOD`` (high wk52 → lower realized — the gate's premise holds) |
| ``BUBBLE_GATE_HARMFUL`` | Top bucket realized > bottom bucket by ≥ ``BUCKET_GAP_GOOD_PCT`` AND spearman ≥ ``SPEARMAN_GOOD`` (high wk52 → HIGHER realized — the gate suppresses profitable trades) |
| ``BUBBLE_GATE_INEFFECTIVE`` | ``|spearman| < SPEARMAN_FLAT`` AND ``|top − bottom| < BUCKET_GAP_TOL_PCT`` (no meaningful effect either way) |
| ``DIRECTIONAL_AGAINST_GATE`` | ``spearman > SPEARMAN_FLAT`` (positive but below ``SPEARMAN_GOOD``) — weak evidence the gate is removing decent trades |
| ``DIRECTIONAL_FOR_GATE`` | ``spearman < -SPEARMAN_FLAT`` (negative but above ``-SPEARMAN_GOOD``) — weak evidence the gate's premise holds |
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .calibration import _spearman

# Module-level constants — tests pin exact verdicts, and a tuning change
# is one reviewable edit. Mirrors `conviction_calibration` discipline.
MIN_PAIRS = 30           # ≥6 per bucket at 5 buckets
N_BUCKETS = 5            # quintile cut
SPEARMAN_FLAT = 0.05     # below |this| → no rank skill
SPEARMAN_GOOD = 0.15     # strong rank skill bar
BUCKET_GAP_TOL_PCT = 1.0  # |top − bottom| < this is FLAT
BUCKET_GAP_GOOD_PCT = 2.0  # |top − bottom| ≥ this is strong evidence

# The gate's actual threshold from `_ml_decide`. The "in suppression
# zone" cut uses this exact value so a future tightening of the gate
# (e.g. lowering to 0.75) only requires one constant change here AND in
# `_ml_decide` — keeping them aligned is the responsibility of whoever
# tunes the gate.
BUBBLE_GATE_THRESHOLD = 0.80

# A wk52_pos must be a fraction in [0,1] to be a valid input. The
# upstream `_compute_technical_indicators` clamps via `(last-lo)/(hi-lo)`
# whose range is structurally [0,1]; out-of-range here means a corrupt
# row — drop it rather than pretend it's a real observation.
WK52_MIN, WK52_MAX = 0.0, 1.0


def _to_finite_float(v):
    """Coerce to a finite float or None. Mirrors
    decision_scorer._to_float / conviction_calibration._to_finite_float
    sentinel handling so an inf/nan row drops cleanly."""
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return f


def build_wk52_skill(records, n_buckets: int = N_BUCKETS) -> dict:
    """Bucket BUY decisions by ``wk52_pos`` quantile and report the
    realized-return spread + a bubble-gate verdict.

    Selection criteria:

    * ``action == 'BUY'`` — the bubble gate is BUY-only.
    * ``wk52_pos`` finite AND in ``[0,1]`` — the upstream clamp guarantees
      that range; an out-of-range value is corrupt.
    * ``forward_return_5d`` finite — a non-finite outcome is structurally
      uncomparable, drop it.

    Returns a JSON-safe dict with the documented schema (see
    ``conviction_calibration.build_conviction_calibration`` for the
    sibling shape this mirrors). Adds two zone-cut fields specific to
    this diagnostic:

    * ``in_zone_n`` / ``in_zone_mean_realized`` — BUYs with
      ``wk52_pos > BUBBLE_GATE_THRESHOLD`` (the gate's own
      suppression zone).
    * ``out_zone_n`` / ``out_zone_mean_realized`` — BUYs at or below the
      threshold (the gate's allowed zone).
    """
    if not records:
        return _empty("no records supplied")

    n_skip_action = 0
    n_skip_wk52 = 0
    n_skip_ret = 0
    pairs: list[tuple[float, float]] = []
    for r in records:
        if not isinstance(r, dict):
            n_skip_action += 1
            continue
        action = str(r.get("action") or "BUY").upper()
        if action != "BUY":
            n_skip_action += 1
            continue
        wk = _to_finite_float(r.get("wk52_pos"))
        if wk is None or not (WK52_MIN <= wk <= WK52_MAX):
            n_skip_wk52 += 1
            continue
        y = _to_finite_float(r.get("forward_return_5d"))
        if y is None:
            n_skip_ret += 1
            continue
        pairs.append((wk, y))

    n = len(pairs)
    if n < MIN_PAIRS:
        return {
            "status": "insufficient_data",
            "verdict": "INSUFFICIENT_DATA",
            "n": n,
            "n_dropped_action": n_skip_action,
            "n_dropped_wk52": n_skip_wk52,
            "n_dropped_return": n_skip_ret,
            "spearman": None,
            "mean_wk52": None,
            "mean_realized": None,
            "top_minus_bottom_realized_pct": None,
            "buckets": [],
            "monotone_fraction": None,
            "in_zone_n": 0,
            "in_zone_mean_realized": None,
            "out_zone_n": 0,
            "out_zone_mean_realized": None,
            "bubble_gate_threshold": BUBBLE_GATE_THRESHOLD,
            "hint": (f"need ≥{MIN_PAIRS} BUY rows with both wk52_pos "
                     f"and forward_return_5d, have {n}"),
        }

    # Stable sort by wk52_pos ascending so identical values carry
    # deterministic ordering through the quantile cut.
    pairs.sort(key=lambda t: t[0])
    W = np.array([p[0] for p in pairs], dtype=np.float64)
    Y = np.array([p[1] for p in pairs], dtype=np.float64)

    # Quantile cut — never more buckets than n/3 (each bucket needs
    # at least 3 samples for a meaningful mean).
    k = max(2, min(n_buckets, n // 3))
    buckets: list[dict] = []
    realized_means: list[float] = []
    for i in range(k):
        lo = i * n // k
        hi = (i + 1) * n // k
        if hi <= lo:
            continue
        seg_w = W[lo:hi]
        seg_y = Y[lo:hi]
        mw = float(seg_w.mean())
        my = float(seg_y.mean())
        sy = float(seg_y.std()) if len(seg_y) >= 2 else 0.0
        realized_means.append(my)
        buckets.append({
            "idx": i + 1,
            "n": int(hi - lo),
            "wk52_lo": round(float(seg_w.min()), 4),
            "wk52_hi": round(float(seg_w.max()), 4),
            "mean_wk52": round(mw, 4),
            "mean_realized": round(my, 4),
            "std_realized": round(sy, 4),
        })

    spearman = _spearman(W, Y)

    # Monotonicity — a perfectly bubble-gate-justifying world has
    # realized strictly *decreasing* with wk52 bucket. We report the
    # non-increasing fraction so the bullish-trend interpretation is
    # natural to read (high = gate-justified).
    steps = len(realized_means) - 1
    if steps <= 0:
        monotone_fraction = 1.0
    else:
        noninc = sum(1 for j in range(steps)
                     if realized_means[j + 1] <= realized_means[j])
        monotone_fraction = noninc / steps

    top_realized = realized_means[-1]
    bot_realized = realized_means[0]
    top_minus_bot = top_realized - bot_realized

    # Gate-zone cut — the exact counterfactual the gate's economic
    # value rides on. ``> threshold`` matches `_ml_decide`'s strict
    # inequality (``_w52 > 0.80``); rows AT exactly 0.80 are in the
    # out-of-zone bucket, consistent with the gate's emission boundary.
    in_zone_y = [p[1] for p in pairs if p[0] > BUBBLE_GATE_THRESHOLD]
    out_zone_y = [p[1] for p in pairs if p[0] <= BUBBLE_GATE_THRESHOLD]
    in_zone_n = len(in_zone_y)
    out_zone_n = len(out_zone_y)
    in_zone_mean = (round(float(np.mean(in_zone_y)), 4)
                    if in_zone_y else None)
    out_zone_mean = (round(float(np.mean(out_zone_y)), 4)
                     if out_zone_y else None)

    # Verdict ladder. Order is important: the strong-evidence verdicts
    # (JUSTIFIED / HARMFUL) take precedence over INEFFECTIVE so the
    # operator sees the decisive call when one exists. The directional
    # verdicts ride the same spearman sign but at the weaker threshold,
    # so they only fire when the strong bar is missed.
    if (spearman <= -SPEARMAN_GOOD
            and top_minus_bot <= -BUCKET_GAP_GOOD_PCT):
        verdict = "BUBBLE_GATE_JUSTIFIED"
        hint = (f"high-wk52 BUYs realize WORSE — top bucket "
                f"{top_realized:+.2f}% vs bottom {bot_realized:+.2f}% "
                f"(spread {top_minus_bot:+.2f}pp, spearman "
                f"{spearman:+.3f}). The bubble-top gate's premise holds.")
    elif (spearman >= SPEARMAN_GOOD
            and top_minus_bot >= BUCKET_GAP_GOOD_PCT):
        verdict = "BUBBLE_GATE_HARMFUL"
        hint = (f"high-wk52 BUYs realize BETTER, not worse — top bucket "
                f"{top_realized:+.2f}% vs bottom {bot_realized:+.2f}% "
                f"(spread {top_minus_bot:+.2f}pp, spearman "
                f"{spearman:+.3f}). The gate is suppressing profitable "
                f"trades.")
    elif (abs(spearman) < SPEARMAN_FLAT
            and abs(top_minus_bot) < BUCKET_GAP_TOL_PCT):
        verdict = "BUBBLE_GATE_INEFFECTIVE"
        hint = (f"wk52_pos does not rank realized returns (spearman "
                f"{spearman:+.3f}, top−bot {top_minus_bot:+.2f}pp). The "
                f"gate is neither protecting nor harming materially.")
    elif spearman <= -SPEARMAN_FLAT:
        verdict = "DIRECTIONAL_FOR_GATE"
        hint = (f"weak negative correlation (spearman {spearman:+.3f}) — "
                f"some evidence high-wk52 BUYs realize worse, but below "
                f"the strong bar (|spearman| ≥ {SPEARMAN_GOOD}).")
    elif spearman >= SPEARMAN_FLAT:
        verdict = "DIRECTIONAL_AGAINST_GATE"
        hint = (f"weak positive correlation (spearman {spearman:+.3f}) — "
                f"some evidence high-wk52 BUYs realize better, but below "
                f"the strong bar.")
    else:
        # Spearman in [-FLAT, +FLAT] but bucket gap > tolerance — the
        # bucket-cut and rank-correlation give inconsistent signals. Honest
        # report rather than forcing a verdict.
        verdict = "BUBBLE_GATE_INEFFECTIVE"
        hint = (f"mixed signal — rank correlation flat (spearman "
                f"{spearman:+.3f}) but bucket spread "
                f"{top_minus_bot:+.2f}pp exceeds {BUCKET_GAP_TOL_PCT}pp "
                f"tolerance.")

    return {
        "status": "ok",
        "verdict": verdict,
        "n": n,
        "n_dropped_action": n_skip_action,
        "n_dropped_wk52": n_skip_wk52,
        "n_dropped_return": n_skip_ret,
        "spearman": round(float(spearman), 4),
        "mean_wk52": round(float(W.mean()), 4),
        "mean_realized": round(float(Y.mean()), 4),
        "top_minus_bottom_realized_pct": round(top_minus_bot, 4),
        "buckets": buckets,
        "monotone_fraction": round(monotone_fraction, 4),
        "in_zone_n": in_zone_n,
        "in_zone_mean_realized": in_zone_mean,
        "out_zone_n": out_zone_n,
        "out_zone_mean_realized": out_zone_mean,
        "bubble_gate_threshold": BUBBLE_GATE_THRESHOLD,
        "hint": hint,
    }


def _empty(reason: str) -> dict:
    return {
        "status": "insufficient_data",
        "verdict": "INSUFFICIENT_DATA",
        "n": 0,
        "n_dropped_action": 0,
        "n_dropped_wk52": 0,
        "n_dropped_return": 0,
        "spearman": None,
        "mean_wk52": None,
        "mean_realized": None,
        "top_minus_bottom_realized_pct": None,
        "buckets": [],
        "monotone_fraction": None,
        "in_zone_n": 0,
        "in_zone_mean_realized": None,
        "out_zone_n": 0,
        "out_zone_mean_realized": None,
        "bubble_gate_threshold": BUBBLE_GATE_THRESHOLD,
        "hint": reason,
    }


def load_outcomes(path: Path | str) -> list[dict]:
    """Stream-load outcome records from a JSONL file. Returns ``[]`` on a
    missing file / unparseable line — never raises (mirrors the sibling
    ``conviction_calibration.load_outcomes`` / ``outcome_drift.load_outcomes``)."""
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


def analyze(outcomes_path: Path | str,
            n_buckets: int = N_BUCKETS) -> dict:
    """Convenience: load + build. The CLI uses this. Never raises — a
    fault in the loader degrades to ``insufficient_data``; a fault in
    the analyzer is caught and returned as ``status='error'`` so a shell
    caller can distinguish a real fault from INSUFFICIENT_DATA (the
    ``conviction_calibration.analyze`` sibling discipline)."""
    try:
        recs = load_outcomes(outcomes_path)
        return build_wk52_skill(recs, n_buckets=n_buckets)
    except Exception as exc:
        out = _empty(f"analyze error: {type(exc).__name__}: {exc}")
        out["status"] = "error"
        return out


def _print_report(rep: dict) -> None:
    print(f"VERDICT: {rep['verdict']}  ({rep['hint']})")
    if rep.get("status") != "ok":
        return
    print(f"  n={rep['n']}  spearman={rep['spearman']:+.3f}  "
          f"mean_wk52={rep['mean_wk52']:.3f}  "
          f"mean_realized={rep['mean_realized']:+.2f}%  "
          f"monotone_decreasing_frac={rep['monotone_fraction']:.2f}  "
          f"top−bot={rep['top_minus_bottom_realized_pct']:+.2f}pp")
    print(f"  dropped: action={rep['n_dropped_action']}  "
          f"wk52={rep['n_dropped_wk52']}  "
          f"return={rep['n_dropped_return']}")
    print(f"  gate-zone (wk52 > {rep['bubble_gate_threshold']:.2f}):  "
          f"in-zone n={rep['in_zone_n']} "
          f"mean={rep['in_zone_mean_realized']}%  "
          f"vs out-zone n={rep['out_zone_n']} "
          f"mean={rep['out_zone_mean_realized']}%")
    print(f"  {'idx':>3}  {'n':>5}  "
          f"{'wk52_range':>16}  {'mean_wk52':>10}  "
          f"{'mean_realized':>15}  {'std':>7}")
    for b in rep["buckets"]:
        rng = f"[{b['wk52_lo']:.2f},{b['wk52_hi']:.2f}]"
        print(f"  {b['idx']:>3}  {b['n']:>5}  {rng:>16}  "
              f"{b['mean_wk52']:>10.3f}  "
              f"{b['mean_realized']:>+14.2f}%  {b['std_realized']:>7.2f}")


def _cli(argv: list[str] | None = None) -> int:
    """``python3 -m paper_trader.ml.wk52_skill [--path P] [--json]`` —
    read-only verification of the bubble-top gate's premise against
    realized 5d returns. Exit code 0 only on ``BUBBLE_GATE_JUSTIFIED``
    (the gate is correctly removing weaker trades); exit 2 on
    ``BUBBLE_GATE_HARMFUL`` (the gate is removing profitable trades —
    the quant-decisive "tune or remove" signal); exit 1 otherwise
    (INEFFECTIVE / DIRECTIONAL_* / INSUFFICIENT_DATA / error)."""
    import argparse
    import sys

    p = argparse.ArgumentParser(
        prog="python3 -m paper_trader.ml.wk52_skill",
        description=(
            "52-week-position realized-return diagnostic. Buckets BUY "
            "decisions by `wk52_pos` quantile and reports whether the "
            "bubble-top gate's premise (high-wk52 BUYs underperform) "
            "holds on captured data. Read-only — never trains or writes."
        ),
    )
    p.add_argument("--path", default="data/decision_outcomes.jsonl",
                   help="Path to decision_outcomes.jsonl")
    p.add_argument("--buckets", type=int, default=N_BUCKETS,
                   help=f"Number of wk52 quantiles (default {N_BUCKETS})")
    p.add_argument("--json", action="store_true",
                   help="Emit machine-readable JSON instead of a table")
    args = p.parse_args(sys.argv[1:] if argv is None else argv)

    rep = analyze(args.path, n_buckets=args.buckets)
    if args.json:
        print(json.dumps(rep, indent=2, sort_keys=True))
    else:
        _print_report(rep)

    v = rep.get("verdict")
    if v == "BUBBLE_GATE_JUSTIFIED":
        return 0
    if v == "BUBBLE_GATE_HARMFUL":
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(_cli())
