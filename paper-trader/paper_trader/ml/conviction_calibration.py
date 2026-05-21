"""Conviction-sizing calibration diagnostic — does the gate's then-applied
position-sizing conviction actually predict realized return?

Read-only diagnostic. Mirrors ``calibration.py`` discipline exactly: never
trains, never writes a pickle, never touches ``build_features`` /
``N_FEATURES``, never modifies trade-path state — safe to run against the
unattended continuous loop (AGENTS.md "Operational quirks a quant should
know"). A fault degrades to ``status='error'`` + a verdict; this module
never raises (the AGENTS.md "ledger / diagnostic must not break the
cycle" discipline).

The question this answers, which no existing analyzer in
``paper_trader/ml/`` does:

* ``gate_audit`` / ``gate_pnl`` bucket by the *scorer's predicted-return
  rank* (the +5.2% → ×1.30 tailwind arm) and re-predict historical
  features with TODAY's pickle — they describe a counterfactual gate.
* ``calibration`` answers "does predicted 5d return rank realized 5d
  return" — i.e. the *scorer's* directional skill, not the *gate's*
  sizing rule.
* ``persona_skill`` / ``leveraged_skill`` cut by persona / instrument
  type, not by the size of the bet.

None of them measure the ACTUAL sizing rule the gate applied at decision
time:

    conviction = min(0.25, ml_score / 20)     # regular
              or min(0.40, ml_score / 15)     # leveraged + bull/sideways
              × gate_modulator (×0.6 .. ×1.3 once n_train ≥ 500)

That conviction IS now persisted as the additive ``conviction_pct`` key
(2026-05-21 feature, BUY-only fraction in [0,1]) but has no consumer.
This analyzer is that consumer. The bucketed verdict tells a skeptical
quant whether the bot's high-conviction calls (sized 25–40% of book)
actually realize higher returns than its low-conviction probes — the
calibration question existing diagnostics structurally cannot answer
because they only ever see rank skill, not economic weight.

Verdict ladder (crisp, threshold-driven, test-locked):

| Verdict | Trigger |
|---------|---------|
| ``INSUFFICIENT_DATA`` | < ``MIN_PAIRS`` BUY rows with ``conviction_pct`` AND ``forward_return_5d`` |
| ``INVERTED`` | rank skill ≤ -``SPEARMAN_FLAT`` OR top-conviction bucket realized < bottom-conviction bucket by more than ``BUCKET_GAP_TOL_PCT`` |
| ``MISCALIBRATED`` | ``|spearman| < SPEARMAN_FLAT`` (sizing carries no rank skill) |
| ``DIRECTIONAL`` | ``spearman > SPEARMAN_FLAT`` AND monotone bucket curve below the ``WELL_CALIBRATED`` bar |
| ``WELL_CALIBRATED`` | ``spearman ≥ SPEARMAN_GOOD`` AND monotone non-decreasing across all buckets AND top-vs-bottom realized spread ≥ ``BUCKET_GAP_GOOD_PCT`` |
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Single source of truth for the tie-aware Spearman that the existing
# scorer-calibration diagnostic uses. Re-using `calibration._spearman`
# means this module's rank skill metric and `calibration` / the scorer
# skill ledger's `oos_ic` can never drift.
from .calibration import _spearman

# Thresholds at module scope so tests pin exact verdicts AND a tuning
# change is a single, reviewable edit (mirrors `calibration.py`'s
# constants-at-module-scope convention).
MIN_PAIRS = 30           # ≥6 per bucket at 5 buckets, ≥10 per bucket at 3
N_BUCKETS = 5            # quintile cut; >3 keeps low/mid/high signal visible
SPEARMAN_FLAT = 0.05     # below |this| → no rank skill on sizing
SPEARMAN_GOOD = 0.15     # strong rank skill bar for WELL_CALIBRATED
BUCKET_GAP_TOL_PCT = 1.0  # top − bottom realized %. Within this is FLAT
BUCKET_GAP_GOOD_PCT = 3.0  # strong spread bar for WELL_CALIBRATED

# A conviction_pct must be a fraction in [0,1] to be a valid input. The
# upstream `_parse_conviction_pct` already clamps to that range, so an
# out-of-range value here means the row was hand-crafted or corrupted —
# drop it rather than pretend it is a real sizing observation.
CONVICTION_MIN, CONVICTION_MAX = 0.0, 1.0


def _to_finite_float(v):
    """Coerce to a finite float or None. Mirrors decision_scorer._to_float
    sentinel handling so an inf/nan row drops cleanly rather than poisoning
    the bucket means."""
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return f


def build_conviction_calibration(records,
                                 n_buckets: int = N_BUCKETS) -> dict:
    """Bucket BUY decisions by ``conviction_pct`` quantile and report the
    realized-return spread + a rank-skill verdict.

    ``records`` is any iterable of decision-outcome dicts (the
    ``decision_outcomes.jsonl`` row shape). Selection criteria:

    * ``action == 'BUY'`` — conviction is BUY-only (the ``_ml_decide`` token
      is emitted only on BUY reasoning); a SELL row's ``conviction_pct`` is
      always ``None``, mirroring the ``gate_scorer_pred`` SELL convention.
    * ``conviction_pct`` finite AND in ``[0,1]`` — the upstream parser
      already clamps to that range, so an out-of-range here is a corrupt
      / hand-crafted record.
    * ``forward_return_5d`` finite — a non-finite outcome is structurally
      uncomparable, drop it (same hardening as `calibration_report`).

    Returns a JSON-safe dict::

        {
            "status": "ok" | "insufficient_data" | "error",
            "verdict": "WELL_CALIBRATED" | "DIRECTIONAL" | "MISCALIBRATED"
                       | "INVERTED" | "INSUFFICIENT_DATA",
            "n": <int>,                 # BUY rows accepted
            "n_dropped_action": <int>,  # non-BUY rows skipped
            "n_dropped_conviction": <int>,  # missing/invalid conviction
            "n_dropped_return": <int>,  # missing/invalid 5d return
            "spearman": <float>,
            "mean_conviction": <float>,
            "mean_realized": <float>,
            "top_minus_bottom_realized_pct": <float>,
            "buckets": [
                {"idx": 1, "n": <int>, "conv_lo": <float>, "conv_hi": <float>,
                 "mean_conviction": <float>, "mean_realized": <float>,
                 "std_realized": <float>},
                ...
            ],
            "monotone_fraction": <float>,
            "hint": <str>,
        }
    """
    if not records:
        return _empty("no records supplied")

    n_skip_action = 0
    n_skip_conv = 0
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
        conv = _to_finite_float(r.get("conviction_pct"))
        if conv is None or not (CONVICTION_MIN <= conv <= CONVICTION_MAX):
            n_skip_conv += 1
            continue
        y = _to_finite_float(r.get("forward_return_5d"))
        if y is None:
            n_skip_ret += 1
            continue
        pairs.append((conv, y))

    n = len(pairs)
    if n < MIN_PAIRS:
        return {
            "status": "insufficient_data",
            "verdict": "INSUFFICIENT_DATA",
            "n": n,
            "n_dropped_action": n_skip_action,
            "n_dropped_conviction": n_skip_conv,
            "n_dropped_return": n_skip_ret,
            "spearman": None,
            "mean_conviction": None,
            "mean_realized": None,
            "top_minus_bottom_realized_pct": None,
            "buckets": [],
            "monotone_fraction": None,
            "hint": (f"need ≥{MIN_PAIRS} BUY rows with both conviction_pct "
                     f"and forward_return_5d, have {n}"),
        }

    # Stable sort by conviction ascending so identical-conviction rows
    # carry deterministic ordering through the quantile cut. Numpy default
    # is quicksort (unstable); mergesort is stable + still O(n log n).
    pairs.sort(key=lambda t: t[0])
    C = np.array([p[0] for p in pairs], dtype=np.float64)
    Y = np.array([p[1] for p in pairs], dtype=np.float64)

    # Quantile cut — never more buckets than n/3 (each bucket needs at
    # least 3 samples for a meaningful mean). Mirrors `calibration_report`.
    k = max(2, min(n_buckets, n // 3))
    buckets: list[dict] = []
    realized_means: list[float] = []
    for i in range(k):
        lo = i * n // k
        hi = (i + 1) * n // k
        if hi <= lo:
            continue
        seg_c = C[lo:hi]
        seg_y = Y[lo:hi]
        mc = float(seg_c.mean())
        my = float(seg_y.mean())
        sy = float(seg_y.std()) if len(seg_y) >= 2 else 0.0
        realized_means.append(my)
        buckets.append({
            "idx": i + 1,
            "n": int(hi - lo),
            "conv_lo": round(float(seg_c.min()), 4),
            "conv_hi": round(float(seg_c.max()), 4),
            "mean_conviction": round(mc, 4),
            "mean_realized": round(my, 4),
            "std_realized": round(sy, 4),
        })

    # Spearman on ALL (conviction, realized) pairs — bucket means hide
    # within-bucket noise (calibration.py docstring rationale).
    spearman = _spearman(C, Y)

    # Monotonicity of bucketed realized means. A perfectly calibrated
    # sizing rule has realized rising with the bucket index.
    steps = len(realized_means) - 1
    if steps <= 0:
        monotone_fraction = 1.0
    else:
        nondec = sum(1 for j in range(steps)
                     if realized_means[j + 1] >= realized_means[j])
        monotone_fraction = nondec / steps

    top_realized = realized_means[-1]
    bot_realized = realized_means[0]
    top_minus_bot = top_realized - bot_realized

    # Verdict ladder. Inversion takes precedence over MISCALIBRATED so the
    # operator sees "INVERTED" loud and clear when the sizing rule is
    # anti-predictive (the gate's tailwind arm realizes worse than its
    # headwind arm — the documented `GATE_HARMFUL` shape from `gate_audit`).
    if (spearman <= -SPEARMAN_FLAT
            or top_minus_bot < -BUCKET_GAP_TOL_PCT):
        verdict = "INVERTED"
        hint = (f"sizing is anti-predictive — top conviction bucket "
                f"realized {top_realized:+.2f}% but bottom realized "
                f"{bot_realized:+.2f}% (spread {top_minus_bot:+.2f}pp, "
                f"spearman {spearman:+.3f}). The gate is sizing UP the "
                f"worst calls and DOWN the best.")
    elif abs(spearman) < SPEARMAN_FLAT:
        verdict = "MISCALIBRATED"
        hint = (f"conviction does not rank realized returns (spearman "
                f"{spearman:+.3f}, |·| < {SPEARMAN_FLAT}). Sizing is "
                f"variance with no compensating realized edge — the "
                f"`GATE_INEFFECTIVE` shape from gate_audit.")
    elif (spearman >= SPEARMAN_GOOD
            and monotone_fraction >= 0.99  # strictly non-decreasing
            and top_minus_bot >= BUCKET_GAP_GOOD_PCT):
        verdict = "WELL_CALIBRATED"
        hint = (f"higher conviction reliably realizes higher return — "
                f"spearman {spearman:+.3f} ≥ {SPEARMAN_GOOD}, monotone, "
                f"top-vs-bottom spread {top_minus_bot:+.2f}pp ≥ "
                f"{BUCKET_GAP_GOOD_PCT}pp.")
    else:
        verdict = "DIRECTIONAL"
        hint = (f"some rank skill on sizing (spearman {spearman:+.3f}) "
                f"but below the strong bar — usable as a tie-breaker, "
                f"not a calibration certificate.")

    return {
        "status": "ok",
        "verdict": verdict,
        "n": n,
        "n_dropped_action": n_skip_action,
        "n_dropped_conviction": n_skip_conv,
        "n_dropped_return": n_skip_ret,
        "spearman": round(float(spearman), 4),
        "mean_conviction": round(float(C.mean()), 4),
        "mean_realized": round(float(Y.mean()), 4),
        "top_minus_bottom_realized_pct": round(top_minus_bot, 4),
        "buckets": buckets,
        "monotone_fraction": round(monotone_fraction, 4),
        "hint": hint,
    }


def _empty(reason: str) -> dict:
    return {
        "status": "insufficient_data",
        "verdict": "INSUFFICIENT_DATA",
        "n": 0,
        "n_dropped_action": 0,
        "n_dropped_conviction": 0,
        "n_dropped_return": 0,
        "spearman": None,
        "mean_conviction": None,
        "mean_realized": None,
        "top_minus_bottom_realized_pct": None,
        "buckets": [],
        "monotone_fraction": None,
        "hint": reason,
    }


def load_outcomes(path: Path | str) -> list[dict]:
    """Stream-load outcome records from a JSONL file. Returns ``[]`` on a
    missing file / unparseable line — never raises (mirrors the sibling
    ``outcome_drift.load_outcomes`` / ``feature_coverage.load_outcomes``)."""
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
    fault in the loader degrades to ``insufficient_data`` (the
    ``load_outcomes`` contract); a fault in the analyzer is caught and
    returned as a ``status='error'`` row so a shell caller can `if !` on
    a real fault distinctly from `INSUFFICIENT_DATA` (the calibration.py
    sibling discipline)."""
    try:
        recs = load_outcomes(outcomes_path)
        return build_conviction_calibration(recs, n_buckets=n_buckets)
    except Exception as exc:
        out = _empty(f"analyze error: {type(exc).__name__}: {exc}")
        out["status"] = "error"
        return out


def _print_report(rep: dict) -> None:
    print(f"VERDICT: {rep['verdict']}  ({rep['hint']})")
    if rep.get("status") != "ok":
        return
    print(f"  n={rep['n']}  spearman={rep['spearman']:+.3f}  "
          f"mean_conviction={rep['mean_conviction']:.3f}  "
          f"mean_realized={rep['mean_realized']:+.2f}%  "
          f"monotone={rep['monotone_fraction']:.2f}  "
          f"top−bot={rep['top_minus_bottom_realized_pct']:+.2f}pp")
    print(f"  dropped: action={rep['n_dropped_action']}  "
          f"conviction={rep['n_dropped_conviction']}  "
          f"return={rep['n_dropped_return']}")
    print(f"  {'idx':>3}  {'n':>5}  "
          f"{'conv_range':>16}  {'mean_conv':>10}  "
          f"{'mean_realized':>15}  {'std':>7}")
    for b in rep["buckets"]:
        rng = f"[{b['conv_lo']:.2f},{b['conv_hi']:.2f}]"
        print(f"  {b['idx']:>3}  {b['n']:>5}  {rng:>16}  "
              f"{b['mean_conviction']:>10.3f}  "
              f"{b['mean_realized']:>+14.2f}%  {b['std_realized']:>7.2f}")


def _cli(argv: list[str] | None = None) -> int:
    """``python3 -m paper_trader.ml.conviction_calibration [--path P] [--json]``
    — read-only calibration of the gate's then-applied sizing rule against
    realized 5d returns. Exit code 0 only on ``WELL_CALIBRATED`` so a shell
    caller can gate dashboards on a real-edge verdict (the
    ``calibration._cli`` discipline). Exit 2 on ``INVERTED`` (the
    quant-decisive "the gate is sizing UP the worst calls" state); exit
    1 otherwise (MISCALIBRATED / DIRECTIONAL / INSUFFICIENT_DATA / error)."""
    import argparse
    import sys

    p = argparse.ArgumentParser(
        prog="python3 -m paper_trader.ml.conviction_calibration",
        description=(
            "Conviction-sizing calibration diagnostic. Buckets BUY "
            "decisions by `conviction_pct` quantile and reports whether "
            "higher sizing actually predicts higher realized 5d return. "
            "Read-only — never trains or writes."
        ),
    )
    p.add_argument("--path", default="data/decision_outcomes.jsonl",
                   help="Path to decision_outcomes.jsonl")
    p.add_argument("--buckets", type=int, default=N_BUCKETS,
                   help=f"Number of conviction quantiles (default {N_BUCKETS})")
    p.add_argument("--json", action="store_true",
                   help="Emit machine-readable JSON instead of a table")
    args = p.parse_args(sys.argv[1:] if argv is None else argv)

    rep = analyze(args.path, n_buckets=args.buckets)
    if args.json:
        print(json.dumps(rep, indent=2, sort_keys=True))
    else:
        _print_report(rep)

    if rep.get("verdict") == "WELL_CALIBRATED":
        return 0
    if rep.get("verdict") == "INVERTED":
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(_cli())
