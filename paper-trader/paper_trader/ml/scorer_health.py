"""Unified scorer health diagnostic.

Composes the three already-existing read-only diagnostics into ONE
operator-readable verdict:

    1. DecisionScorer pickle status (``is_trained``, ``n_train``).
       Below ``GATE_THRESHOLD`` the conviction gate in ``_ml_decide``
       (CLAUDE.md invariant #5) never engages, so EVERY downstream
       "gate is harmful / noise" signal is academic; surfacing that
       distinction in ONE verdict prevents a cron / dashboard from
       crying wolf on a dark scorer.
    2. Historical gate realized arm spread
       (``paper_trader.ml.gate_realized.analyze``) — captured-then-
       deployed prediction × realized 5d forward return, no
       re-prediction with today's pickle. Answers: "when the gate
       said strong tailwind, did the position actually outperform?"
    3. Historical gate calibration over (``gate_scorer_pred``,
       ``forward_return_5d``) pairs — bucket the captured predictions
       into deciles and ask whether the predicted % tracks the
       realized %.  Uses ``calibration.calibration_report`` so the
       in-sample / OOS calibration verdict can never drift from this
       one (single source of truth — the AGENTS.md spirit of #10).

Why a NEW module instead of extending one of those three? Because the
existing ones each answer a SLICE of the question
(`gate_realized` only describes arm allocation; `calibration` only
describes magnitude tracking; `decision_scorer` only reports the
pickle). An operator wanting to answer "is the deployed scorer
worth gating on RIGHT NOW" must currently run THREE commands and
mentally combine three verdicts — and the most common combinations
(``GATE_INACTIVE`` + ``MISCALIBRATED`` is benign; ``gate-active``
+ ``MISCALIBRATED`` is critical) are NOT visible from any one of
them in isolation.  This module is the missing reduction.

Module-level constants pin the verdict thresholds so tests assert
EXACT verdicts and a tuning change to those constants requires a
test update (the ``calibration`` / ``gate_realized`` discipline).
Read-only: NEVER loads a model in train mode, NEVER writes to disk,
NEVER mutates the scorer pickle or the outcomes file. Defensive
in every branch — the same "diagnostic must not break the loop"
discipline ``_append_scorer_skill_log`` and ``_parse_scorer_status``
already lock.

CLI exits 2 when the verdict requires operator action
(``GATE_HARMFUL`` or ``NOISE_GATE_ACTIVE``) so the same cron pattern
as ``gate_realized._cli`` / ``gate_pnl`` works unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

# Gate-active threshold — pinned to the same constant ``_ml_decide`` uses
# (CLAUDE.md invariant #5).  A scorer below this n_train does NOT modulate
# any trade, so a "calibration is broken" verdict on a sub-threshold scorer
# is academic.  Surfaced here so tests assert it and a future change to
# the gate threshold can be made in ONE place across the codebase (mirror
# of how MLP_CONFIG centralises the MLP hyper-params).
GATE_THRESHOLD = 500

# Minimum outcome rows needed to compute calibration / gate_realized.
# Mirrors ``calibration.MIN_PAIRS`` philosophy — below this the verdict is
# INSUFFICIENT_DATA, not a fabricated "looks healthy on 5 rows".
MIN_OUTCOMES_FOR_VERDICT = 30

# Path resolution — module-level so tests can monkeypatch ``OUTCOMES_PATH``
# / ``SCORER_PATH`` without poking into the live data file
# (the AGENTS.md "hardcoded paths must be module-level for testability"
# rule, same pattern as ``run_continuous_backtests.SCORER_SKILL_LOG``).
_ROOT = Path(__file__).resolve().parent.parent.parent
OUTCOMES_PATH = _ROOT / "data" / "decision_outcomes.jsonl"


def _safe_load_scorer() -> dict:
    """Read-only inspect of the deployed pickle: ``{trained, n_train,
    gate_active, error}``.  Never raises — a load failure degrades to
    ``trained=False`` with the error string captured so the operator
    can see it.
    """
    out = {"trained": False, "n_train": 0, "gate_active": False,
           "error": None}
    try:
        from paper_trader.ml.decision_scorer import DecisionScorer
        s = DecisionScorer()
        out["trained"] = bool(s.is_trained)
        out["n_train"] = int(s.n_train)
        out["gate_active"] = bool(s.is_trained and s.n_train >= GATE_THRESHOLD)
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def _safe_gate_realized(outcomes_path: Path) -> dict:
    """Wrap ``gate_realized.analyze`` in a degrade-never envelope.

    Returns ``{verdict, hint, status, n_acted, tail_minus_head, error}``.
    A diagnostic crash here NEVER blocks the health verdict — the
    AGENTS.md "scorer-train status must stay truthful" discipline.
    """
    out = {"verdict": "GATE_CAPTURE_NOT_YET_POPULATED", "hint": "",
           "status": "error", "n_acted": 0,
           "tail_minus_head": None, "error": None}
    try:
        from paper_trader.ml.gate_realized import analyze as _gr_analyze
        rep = _gr_analyze(outcomes_path, oos_only=True)
        out["verdict"] = str(rep.get("verdict") or out["verdict"])
        out["hint"] = str(rep.get("hint") or "")
        out["status"] = str(rep.get("status") or "error")
        out["n_acted"] = int(rep.get("n_acted") or 0)
        out["tail_minus_head"] = rep.get("strong_tailwind_minus_headwind_pp")
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def _safe_gate_calibration(outcomes_path: Path) -> dict:
    """Calibration of the gate's actually-captured historical prediction
    against the realized 5d forward return.

    Why this rather than ``calibration.scorer_calibration_oos``: that
    one re-predicts every OOS row with TODAY's pickle — a counterfactual
    ("what would the CURRENT model say"), provably NOT what the gate
    ever did at decision time.  ``gate_scorer_pred`` is what the gate
    ACTUALLY said with that cycle's then-deployed scorer.  Pairing it
    with the realized forward return measures the deployed scorer's
    *deployed* calibration over time — the question every other
    calibration module structurally cannot answer (the ``gate_pnl``
    reconstruction residual is documented).

    SELL outcomes carry no ``gate_scorer_pred`` (the gate is BUY-only),
    so they fall out naturally via the ``is None`` filter.
    """
    out: dict = {"verdict": "INSUFFICIENT_DATA", "hint": "",
                 "status": "error", "n": 0, "spearman": None,
                 "mean_abs_decile_error": None, "error": None}
    try:
        from paper_trader.ml.calibration import calibration_report
        p = Path(outcomes_path)
        if not p.exists():
            out["hint"] = f"no outcomes file at {p}"
            return out
        pairs: list[tuple[float, float]] = []
        for ln in p.read_text().splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            pred = obj.get("gate_scorer_pred")
            fr = obj.get("forward_return_5d")
            if pred is None or fr is None:
                continue
            # SELL outcomes flip the realized target so "good" has one
            # consistent meaning — mirrors ``_oos_rank_metrics`` and
            # ``evaluate_scorer_oos``. The captured ``gate_scorer_pred``
            # for SELL is already None (the gate is BUY-only), so this
            # branch reads pairs in BUY space only and the flip is a
            # no-op in practice — but the explicit guard keeps the
            # invariant verifiable.
            try:
                p_f = float(pred)
                a_f = float(fr)
            except (TypeError, ValueError):
                continue
            if str(obj.get("action") or "BUY").upper() == "SELL":
                a_f = -a_f
            pairs.append((p_f, a_f))
        rep = calibration_report(pairs)
        out["verdict"] = str(rep.get("verdict") or out["verdict"])
        out["hint"] = str(rep.get("hint") or "")
        out["status"] = str(rep.get("status") or "error")
        out["n"] = int(rep.get("n") or 0)
        out["spearman"] = rep.get("spearman")
        out["mean_abs_decile_error"] = rep.get("mean_abs_decile_error")
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def _derive_verdict(scorer: dict, gate_real: dict, gate_calib: dict) -> tuple[str, str]:
    """Combine the three sub-verdicts into a single overall verdict + hint.

    Precedence (a verdict higher in this list wins over a lower one):
      1. UNTRAINED              — pickle absent / load failed
      2. GATE_INACTIVE          — trained but n_train < GATE_THRESHOLD
      3. INSUFFICIENT_DATA      — too few outcome rows to verdict
      4. GATE_HARMFUL           — gate's realized arm allocation is inverted
      5. NOISE_GATE_ACTIVE      — gate-active but realized OR calibration is noise
      6. DIRECTIONAL_BUT_BIASED — gate-active and effective by rank, but
                                   magnitude is off (size on signal, NOT %)
      7. HEALTHY                — everything checks out

    Each step is a CONSERVATIVE veto — we never claim ``HEALTHY`` unless every
    sub-verdict is positively healthy.  A diagnostic error in EITHER child
    degrades to a less-trusting parent verdict, never silently passes.
    """
    if not scorer.get("trained"):
        err = scorer.get("error")
        hint = (f"decision_scorer pickle load failed: {err}" if err
                else "decision_scorer pickle absent or untrained — "
                     "gate is dark, no sizing modulation active")
        return "UNTRAINED", hint

    n_train = int(scorer.get("n_train") or 0)
    if not scorer.get("gate_active"):
        return ("GATE_INACTIVE",
                f"scorer trained (n_train={n_train}) but below the "
                f"GATE_THRESHOLD={GATE_THRESHOLD} — conviction gate "
                "(_ml_decide invariant #5) does NOT modulate trades")

    gr_v = gate_real.get("verdict")
    gc_v = gate_calib.get("verdict")
    n_acted = int(gate_real.get("n_acted") or 0)
    n_cal = int(gate_calib.get("n") or 0)
    if (n_acted < MIN_OUTCOMES_FOR_VERDICT
            and n_cal < MIN_OUTCOMES_FOR_VERDICT):
        return ("INSUFFICIENT_DATA",
                f"too few captured gate decisions (n_acted={n_acted}, "
                f"n_calibration={n_cal}, need ≥{MIN_OUTCOMES_FOR_VERDICT})")

    if gr_v == "GATE_HARMFUL":
        return "GATE_HARMFUL", (
            "gate's realized arm allocation is INVERTED — "
            "strong-tailwind underperformed strong-headwind by enough to "
            "be statistically distinguishable: "
            f"{gate_real.get('hint') or 'see gate_realized'}")

    is_gr_noise = gr_v in ("GATE_INEFFECTIVE", "GATE_CAPTURE_NOT_YET_POPULATED")
    is_gc_noise = gc_v in ("MISCALIBRATED", "WEAK_SIGNAL")
    if is_gr_noise or is_gc_noise:
        return ("NOISE_GATE_ACTIVE",
                f"gate active (n_train={n_train}≥{GATE_THRESHOLD}) but "
                f"realized={gr_v}, calibration={gc_v} — sizing variance "
                "with no edge")

    if gc_v == "DIRECTIONAL_BUT_BIASED":
        return ("DIRECTIONAL_BUT_BIASED",
                f"gate effective by rank (realized={gr_v}) but predicted "
                "% is biased — trust the sign / ordering, discount the "
                f"raw %: {gate_calib.get('hint') or 'see calibration'}")

    if gr_v == "GATE_EFFECTIVE" and gc_v == "WELL_CALIBRATED":
        return ("HEALTHY",
                f"realized={gr_v}, calibration={gc_v}, "
                f"n_train={n_train}≥{GATE_THRESHOLD} — the gate is "
                "carrying its weight")

    # Fall-through: at least one sub-verdict is one we don't recognise
    # (e.g. a future ``gate_realized`` verdict the literal-match above
    # didn't enumerate).  Don't claim HEALTHY — degrade to a noise label
    # so a future verdict addition doesn't silently inherit the green
    # bill of health.
    return ("NOISE_GATE_ACTIVE",
            f"unrecognised sub-verdict combination realized={gr_v}, "
            f"calibration={gc_v} — degrading to noise rather than "
            "claim health on an unknown combo")


def report(outcomes_path: Path | str | None = None) -> dict:
    """Top-level diagnostic — composes the three sub-checks into ONE
    structured report.  Always returns a JSON-safe dict; never raises.

    Returns ``{verdict, hint, scorer, gate_realized, gate_calibration,
    gate_threshold}`` where the three sub-dicts mirror the corresponding
    sub-diagnostic outputs.
    """
    path = Path(outcomes_path) if outcomes_path else OUTCOMES_PATH
    scorer = _safe_load_scorer()
    gate_real = _safe_gate_realized(path)
    gate_calib = _safe_gate_calibration(path)
    verdict, hint = _derive_verdict(scorer, gate_real, gate_calib)
    return {
        "verdict": verdict,
        "hint": hint,
        "gate_threshold": GATE_THRESHOLD,
        "scorer": scorer,
        "gate_realized": gate_real,
        "gate_calibration": gate_calib,
    }


# Exit codes mirror gate_realized: 2 on actionable, 0 on informational.
# A noise / harmful verdict is the actionable case an operator (or cron
# branch) should care about. UNTRAINED is also actionable — the gate is
# dark and the operator should know.
_EXIT_NONZERO = frozenset({"GATE_HARMFUL", "NOISE_GATE_ACTIVE", "UNTRAINED"})


def _cli(argv: list[str] | None = None) -> int:
    """``python3 -m paper_trader.ml.scorer_health [--json]`` — one-line
    operator-readable health verdict.  Read-only.

    Exits 2 on a verdict in ``_EXIT_NONZERO`` (mirror of
    ``gate_realized._cli``) so a shell branch ``if ! python3 -m
    paper_trader.ml.scorer_health; then …`` triggers exactly on
    actionable verdicts and stays silent on the informational ones
    (``GATE_INACTIVE`` / ``INSUFFICIENT_DATA``).
    """
    import sys

    argv = sys.argv[1:] if argv is None else argv
    rep = report()
    if "--json" in argv:
        print(json.dumps(rep, indent=2, sort_keys=True, default=str))
        return 2 if rep["verdict"] in _EXIT_NONZERO else 0

    s = rep["scorer"]
    gr = rep["gate_realized"]
    gc = rep["gate_calibration"]
    print(f"VERDICT: {rep['verdict']}  ({rep['hint']})")
    print(f"  scorer: trained={s.get('trained')} n_train={s.get('n_train')} "
          f"gate_active={s.get('gate_active')} (threshold≥{GATE_THRESHOLD})"
          + (f"  ERROR: {s['error']}" if s.get("error") else ""))
    tmh = gr.get("tail_minus_head")
    tmh_s = f"{tmh:+.2f}pp" if isinstance(tmh, (int, float)) else "n/a"
    print(f"  gate_realized: verdict={gr.get('verdict')} "
          f"n_acted={gr.get('n_acted')} tail-head={tmh_s}"
          + (f"  ERROR: {gr['error']}" if gr.get("error") else ""))
    sp = gc.get("spearman")
    sp_s = f"{sp:+.4f}" if isinstance(sp, (int, float)) else "n/a"
    de = gc.get("mean_abs_decile_error")
    de_s = f"{de:.2f}pp" if isinstance(de, (int, float)) else "n/a"
    print(f"  gate_calibration: verdict={gc.get('verdict')} n={gc.get('n')} "
          f"spearman={sp_s} mean_abs_decile_err={de_s}"
          + (f"  ERROR: {gc['error']}" if gc.get("error") else ""))
    return 2 if rep["verdict"] in _EXIT_NONZERO else 0


if __name__ == "__main__":
    raise SystemExit(_cli())
