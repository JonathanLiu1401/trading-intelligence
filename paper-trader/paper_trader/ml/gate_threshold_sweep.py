"""Gate-threshold sweep diagnostic — would the conviction gate work better
with different ±strong-band boundaries?

The ``_ml_decide`` conviction gate uses hardcoded ±10% / ±5% boundaries
to classify scorer predictions into five arms (strong/mild headwind,
neutral, mild/strong tailwind). The ``conviction_calibration`` and
``gate_realized`` siblings answer "does the deployed gate add value with
those specific boundaries?" — verdict on the live OOS slice is
``MISCALIBRATED`` (sizing is variance with no edge). The natural next
question a quant asks is one the existing tools structurally cannot:

  *Are the ±10/±5 boundaries themselves suboptimal — would a different
   threshold pair (say ±15/±5 or percentile-based ±p90) improve the
   realized return spread between top-arm and bottom-arm trades?*

Sweeping candidate threshold pairs gives a direct empirical answer. If
the BEST candidate's top-minus-bottom spread is still indistinguishable
from zero, no threshold tuning can rescue the gate — the model carries
no usable rank signal at any boundary. If a different threshold pair
shows a robust positive spread, the gate's boundaries are themselves
underperforming the model's actual edge.

Read-only by design. Loads only ``data/decision_outcomes.jsonl`` (no
pickle load, no ``predict()`` calls — uses the captured
``gate_scorer_pred`` field directly, the same data the deployed gate
actually sized on at decision time). Mirrors ``gate_realized``'s
discipline exactly: never trains, never writes the pickle, never
touches ``build_features`` / ``N_FEATURES``, never modifies any trade
path. A fault degrades to ``status='error'`` + a verdict; this module
never raises (the AGENTS.md "ledger / diagnostic must not break the
cycle" discipline). Safe under the live unattended continuous loop.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np


# Min rows in EACH (top, bottom) arm to even attempt a spread. Below this
# the bootstrap CI is too wide to read — mirrors ``gate_audit.MIN_ARM_N``.
MIN_ARM_N = 5

# Minimum sample of usable BUY rows. Below this every candidate's CI
# overlaps zero by construction — no verdict is honest. Mirrors
# ``conviction_calibration.MIN_RECORDS`` / ``calibration.MIN_PAIRS`` spirit.
MIN_RECORDS = 60

# Bootstrap repetitions for the top-minus-bottom spread CI. 500 is the
# same default ``oos_bootstrap_ci`` / ``skill_uncertainty`` use.
N_BOOTSTRAP = 500

# Edge band — a top-minus-bottom spread inside ±this is reported as noise
# (``NO_THRESHOLD_HELPS``). Mirrors ``gate_audit.EDGE_TOL_PP``.
EDGE_TOL_PP = 1.0

# Candidate threshold pairs to sweep. Symmetric absolute boundaries on
# the captured ``gate_scorer_pred`` (%); rows with ``pred > +hi`` go to
# the top arm, ``pred < -lo`` to the bottom arm, everything else
# (including the deployed ``(0, 5]`` mild-tailwind / mild-headwind
# sub-bands) is treated as "middle / no strong call" and excluded from
# the spread. Keeping it symmetric reduces the cartesian explosion to
# something a CLI can show in one table — the asymmetric case
# (``conviction_calibration`` already buckets by deciles) is a separate
# question.
CANDIDATE_THRESHOLDS: tuple[float, ...] = (2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 20.0)

# Deployed gate boundary — exactly what the live ``_ml_decide`` uses to
# split the strong arms (±10%). Surfaced in the report so a reader can
# compare every candidate against the current gate without having to
# memorize the ``backtest.py::_ml_decide`` boundary value.
DEPLOYED_STRONG_BOUND = 10.0


def _to_float(v, default: float = float("nan")) -> float:
    """Cheap, total numeric coercion — `None` / strings / non-finite all
    return ``default`` (NaN by default so the per-row guard drops them)."""
    if v is None or isinstance(v, bool):
        return default
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(f):
        return default
    return f


def _load_rows(outcomes_path: Path | str) -> list[dict]:
    """Stream ``decision_outcomes.jsonl`` to a list of dicts; corrupt lines
    are dropped silently (mirrors every other ledger reader's discipline).
    A missing file returns an empty list (not an error) so a fresh
    deployment reports ``INSUFFICIENT_DATA`` rather than crashing.
    """
    p = Path(outcomes_path)
    if not p.exists():
        return []
    rows: list[dict] = []
    try:
        with p.open("r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if isinstance(rec, dict):
                    rows.append(rec)
    except Exception:
        return []
    return rows


def _usable_pairs(rows: list[dict]) -> list[tuple[float, float]]:
    """Extract the (gate_scorer_pred, realized_5d) pairs the sweep operates
    on. Only BUY rows that carry BOTH a captured gate prediction
    (``gate_scorer_pred is not None`` — emitted by the cycle's gate at
    decision time) AND a finite realized 5d return contribute. Off-
    distribution abstentions (``gate_off_dist=True``) are EXCLUDED — the
    live gate left conviction untouched there, so they don't describe the
    gate's threshold behaviour. This mirrors ``gate_realized``'s
    abstained-bucket exclusion exactly (the same honesty discipline a
    ``GATE_INEFFECTIVE`` reader needs).
    """
    out: list[tuple[float, float]] = []
    for r in rows:
        if str(r.get("action") or "").upper() != "BUY":
            continue
        if r.get("gate_off_dist"):
            continue
        gp_raw = r.get("gate_scorer_pred")
        if gp_raw is None:
            continue
        gp = _to_float(gp_raw)
        fr = _to_float(r.get("forward_return_5d"))
        if math.isnan(gp) or math.isnan(fr):
            continue
        out.append((gp, fr))
    return out


def _bootstrap_spread_ci(top: list[float], bot: list[float],
                         n_boot: int = N_BOOTSTRAP,
                         seed: int = 42) -> tuple[float, float, float]:
    """Return ``(mean_spread, lo95, hi95)`` for the bootstrap distribution
    of ``mean(top) - mean(bot)``. Resamples each arm INDEPENDENTLY with
    replacement at its own n — the natural null is "the two arms have the
    same mean", and resampling preserves their sample sizes so the CI
    width reflects the realized arm sizes honestly. Returns ``(nan, nan,
    nan)`` when either arm is empty.
    """
    if not top or not bot:
        return (float("nan"), float("nan"), float("nan"))
    rng = random.Random(seed)
    top_a = np.asarray(top, dtype=np.float64)
    bot_a = np.asarray(bot, dtype=np.float64)
    nt = len(top_a)
    nb = len(bot_a)
    diffs = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        # ``random.Random.choices`` is markedly faster than
        # ``np.random.default_rng().integers`` for small arms (<1000) and
        # keeps the seed plumbing single-source.
        ti = [rng.randrange(nt) for _ in range(nt)]
        bi = [rng.randrange(nb) for _ in range(nb)]
        diffs[i] = top_a[ti].mean() - bot_a[bi].mean()
    return (float(diffs.mean()),
            float(np.percentile(diffs, 2.5)),
            float(np.percentile(diffs, 97.5)))


def _evaluate_threshold(pairs: list[tuple[float, float]],
                        bound: float) -> dict:
    """Bucket the (pred, realized) pairs by a single symmetric ±bound
    threshold and report the realized return spread between the top arm
    (pred > +bound) and the bottom arm (pred < -bound). Middle rows
    (|pred| <= bound) are excluded — this measures the strong-band spread
    only, the most economically decisive bucket of the gate.

    Returns a per-threshold report dict; all fields are JSON-safe and
    ``None`` whenever the corresponding stat could not be computed.
    """
    top = [r for p, r in pairs if p > bound]
    bot = [r for p, r in pairs if p < -bound]
    n_top = len(top)
    n_bot = len(bot)
    out: dict = {
        "bound": round(bound, 4),
        "n_top": n_top,
        "n_bot": n_bot,
        "n_total": len(pairs),
        "n_middle": len(pairs) - n_top - n_bot,
        "mean_top": None, "mean_bot": None,
        "spread": None, "spread_ci_low": None, "spread_ci_high": None,
        "spread_significant": None,
    }
    if n_top:
        out["mean_top"] = round(float(np.mean(top)), 4)
    if n_bot:
        out["mean_bot"] = round(float(np.mean(bot)), 4)
    if n_top >= MIN_ARM_N and n_bot >= MIN_ARM_N:
        mean_s, lo95, hi95 = _bootstrap_spread_ci(top, bot)
        out["spread"] = round(mean_s, 4)
        out["spread_ci_low"] = round(lo95, 4)
        out["spread_ci_high"] = round(hi95, 4)
        # 95% CI strictly excludes zero ⇒ the spread is statistically
        # distinguishable from no edge. The standard interpretation
        # ``oos_bootstrap_ci`` / ``skill_uncertainty`` already use.
        out["spread_significant"] = bool(lo95 > 0.0 or hi95 < 0.0)
    return out


def analyze(outcomes_path: Path | str = None,
            candidates: tuple[float, ...] = CANDIDATE_THRESHOLDS) -> dict:
    """Sweep candidate symmetric ±bounds over the captured BUY outcomes
    and return a verdict.

    Verdict ladder (crisp, threshold-driven, test-locked):

    ===========================  ==============================================
    Verdict                      Trigger
    ===========================  ==============================================
    ``INSUFFICIENT_DATA``        ``n_pairs < MIN_RECORDS``
    ``NO_THRESHOLD_HELPS``       best |spread| ≤ ``EDGE_TOL_PP`` OR
                                 best spread CI overlaps zero
    ``DEPLOYED_IS_BEST``         deployed ±10 is the argmax spread
    ``ALTERNATIVE_THRESHOLD_BEATS_DEPLOYED``  some other bound has a
                                 strictly larger spread AND its CI excludes
                                 zero
    ===========================  ==============================================
    """
    out: dict = {
        "status": "ok",
        "verdict": "INSUFFICIENT_DATA",
        "n_pairs": 0,
        "candidates": [],
        "deployed_bound": DEPLOYED_STRONG_BOUND,
        "deployed_spread": None,
        "best_bound": None,
        "best_spread": None,
        "best_spread_ci_low": None,
        "best_spread_ci_high": None,
        "best_spread_significant": None,
    }
    try:
        if outcomes_path is None:
            outcomes_path = (Path(__file__).resolve().parent.parent.parent
                             / "data" / "decision_outcomes.jsonl")
        rows = _load_rows(outcomes_path)
        pairs = _usable_pairs(rows)
        out["n_pairs"] = len(pairs)
        if len(pairs) < MIN_RECORDS:
            out["hint"] = (
                f"n_pairs={len(pairs)} < MIN_RECORDS={MIN_RECORDS} — "
                "not enough captured BUY rows with both a then-deployed "
                "gate_scorer_pred and a finite forward_return_5d to "
                "evaluate threshold sweeps. Need more cycles with the "
                "gate active (n_train >= 500)."
            )
            return out

        per_bound = [_evaluate_threshold(pairs, b) for b in candidates]
        out["candidates"] = per_bound

        # Deployed bound report — surface its row separately so a reader
        # can compare every candidate against the current gate at a glance.
        deployed_row = next(
            (c for c in per_bound
             if c["bound"] == round(DEPLOYED_STRONG_BOUND, 4)),
            None,
        )
        if deployed_row is None:
            deployed_row = _evaluate_threshold(pairs, DEPLOYED_STRONG_BOUND)
        out["deployed_spread"] = deployed_row.get("spread")

        # Best candidate — only consider rows where a spread was computable
        # (both arms ≥ MIN_ARM_N). The argmax is over the SIGNED spread,
        # not |spread|: a negative spread (top arm WORSE than bottom) is
        # actively harmful for the gate's stated direction, not a virtue.
        eligible = [c for c in per_bound if c["spread"] is not None]
        if not eligible:
            out["verdict"] = "NO_THRESHOLD_HELPS"
            out["hint"] = (
                "no candidate threshold had both arms ≥ MIN_ARM_N "
                f"({MIN_ARM_N}); CI is undefined everywhere"
            )
            return out
        best = max(eligible, key=lambda c: c["spread"])
        out["best_bound"] = best["bound"]
        out["best_spread"] = best["spread"]
        out["best_spread_ci_low"] = best["spread_ci_low"]
        out["best_spread_ci_high"] = best["spread_ci_high"]
        out["best_spread_significant"] = best["spread_significant"]

        # Verdict.
        # Pre-conditions for ANY "the gate has a tunable edge" verdict:
        #   (1) best signed spread is strictly POSITIVE — a negative
        #       top-vs-bottom spread means the gate's directional intent
        #       is inverted, and a "DEPLOYED_IS_BEST" or "ALTERNATIVE_
        #       BEATS_DEPLOYED" verdict on a negative spread would lie
        #       to a reader (the gate is actively harmful, not best).
        #   (2) the magnitude exceeds the EDGE_TOL_PP noise band.
        #   (3) the CI excludes zero (a real signal, not luck on the
        #       argmax draw).
        if (best["spread"] is None
                or best["spread"] <= EDGE_TOL_PP
                or not best["spread_significant"]):
            out["verdict"] = "NO_THRESHOLD_HELPS"
        elif best["bound"] == round(DEPLOYED_STRONG_BOUND, 4):
            out["verdict"] = "DEPLOYED_IS_BEST"
        else:
            # Only flip the verdict if the new bound's spread is
            # MATERIALLY better than the deployed spread — otherwise the
            # argmax just rode a noisier arm's draw. Simple heuristic:
            # best_spread > deployed_spread + EDGE_TOL_PP.
            deployed_spread = deployed_row.get("spread")
            if (deployed_spread is not None
                    and best["spread"] > deployed_spread + EDGE_TOL_PP):
                out["verdict"] = "ALTERNATIVE_THRESHOLD_BEATS_DEPLOYED"
            else:
                out["verdict"] = "DEPLOYED_IS_BEST"
        return out
    except Exception as exc:
        # Last-ditch honesty: emit the same {status, verdict} shape every
        # other diagnostic uses on a fault so ledger writers can record a
        # gap rather than skip a row. NEVER raises.
        return {
            "status": "error",
            "verdict": "INSUFFICIENT_DATA",
            "hint": f"gate_threshold_sweep crashed: {type(exc).__name__}: {exc}",
            "n_pairs": 0,
            "candidates": [],
            "deployed_bound": DEPLOYED_STRONG_BOUND,
            "deployed_spread": None,
            "best_bound": None,
            "best_spread": None,
            "best_spread_ci_low": None,
            "best_spread_ci_high": None,
            "best_spread_significant": None,
        }


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python3 -m paper_trader.ml.gate_threshold_sweep",
        description=(
            "Sweep candidate ±strong-band gate thresholds over the captured "
            "BUY outcomes in data/decision_outcomes.jsonl; report whether a "
            "different boundary improves the realized return spread between "
            "the top and bottom arm. Read-only — never trains, never writes "
            "the pickle, never touches build_features / N_FEATURES."
        ),
    )
    p.add_argument("--outcomes", default=None,
                   help="Path to decision_outcomes.jsonl (default: live).")
    p.add_argument("--json", action="store_true",
                   help="Emit JSON instead of a human-readable table.")
    return p


def _cli(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(
        sys.argv[1:] if argv is None else argv)
    rep = analyze(outcomes_path=args.outcomes)

    if args.json:
        print(json.dumps(rep, indent=2, sort_keys=True))
        # Exit code: 0 on a real verdict, 1 on data-gap (INSUFFICIENT_DATA)
        # so shell callers can gate on `$?` like host_guard's CLI.
        return 0 if rep.get("verdict") not in (
            "INSUFFICIENT_DATA",) else 1

    print(f"[gate_threshold_sweep] n_pairs={rep['n_pairs']} "
          f"verdict={rep['verdict']}")
    if rep.get("hint"):
        print(f"  hint: {rep['hint']}")
    if rep.get("deployed_spread") is not None:
        ds = rep["deployed_spread"]
        print(f"  deployed ±{rep['deployed_bound']}% → "
              f"spread={ds:+.3f}pp")
    if rep.get("best_bound") is not None:
        bs = rep["best_spread"]
        bl = rep["best_spread_ci_low"]
        bh = rep["best_spread_ci_high"]
        sig = "*" if rep.get("best_spread_significant") else " "
        print(f"  best     ±{rep['best_bound']}% → "
              f"spread={bs:+.3f}pp  95%CI=[{bl:+.3f}, {bh:+.3f}]{sig}")
    if rep.get("candidates"):
        print(f"\n  {'bound':>7}{'n_top':>8}{'n_bot':>8}"
              f"{'mean_top':>12}{'mean_bot':>12}"
              f"{'spread':>10}{'95%CI':>22}")
        for c in rep["candidates"]:
            mt = c["mean_top"]
            mb = c["mean_bot"]
            sp = c["spread"]
            lo = c["spread_ci_low"]
            hi = c["spread_ci_high"]
            mt_s = f"{mt:+10.3f}" if mt is not None else "       N/A"
            mb_s = f"{mb:+10.3f}" if mb is not None else "       N/A"
            sp_s = f"{sp:+8.3f}" if sp is not None else "     N/A"
            ci_s = (f"[{lo:+6.3f}, {hi:+6.3f}]"
                    if lo is not None and hi is not None
                    else "             N/A")
            print(f"  ±{c['bound']:>5.1f}  {c['n_top']:>7d} {c['n_bot']:>7d}"
                  f" {mt_s} {mb_s} {sp_s}  {ci_s}")
    return 0 if rep.get("verdict") not in ("INSUFFICIENT_DATA",) else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
