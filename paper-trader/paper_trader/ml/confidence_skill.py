"""Confidence-conditional scorer skill — does the off_distribution flag
actually identify low-trust predictions?

The conviction gate (CLAUDE.md §6, invariant #5) treats any prediction
with ``off_distribution=True`` (raw |pred| > PRED_CLAMP_PCT OR the
predict path raised) as untrustworthy: it abstains from modulating
conviction on that decision. The PREMISE is that those predictions
carry less rank-skill than in-distribution ones — but no existing
diagnostic verifies that premise on real outcomes.

``gate_abstention.py`` reports the abstention RATE (how often the guard
fires) but not the rank-IC of trusted vs abstained slices. Without that
comparison, the guard could be:

  * **Working**: trusted-IC > abstained-IC (the documented intent — the
    guard correctly identifies extrapolated, low-trust predictions and
    keeps them out of the gate's sizing path).
  * **Neutral**: trusted-IC ≈ abstained-IC (the off-distribution flag
    does not separate skill levels — the guard isn't load-bearing
    either way, and the simpler "always act" version would behave the
    same).
  * **Harmful**: trusted-IC < abstained-IC (the guard is *removing*
    the most informative predictions from the gate's sizing — a
    counter-productive filter the documented MLP_WORSE_THAN_TRIVIAL
    state is consistent with).
  * **Inverted**: trusted-IC < 0 < abstained-IC (the model's trusted
    half is anti-predictive while its abstained half carries the
    actual signal — the worst possible operator-state).

This module answers the question on the SAME temporal OOS slice the
scorer skill ledger uses (``validation.split_outcomes_temporal``), with
the SAME ``_spearman`` from ``ml.calibration`` (single source of truth
— the AGENTS.md invariant-10 spirit), the SAME predict_with_meta-only
path the ``_oos_rank_metrics`` discipline uses (drop ``failed=True``
rows so fabricated-zero predictions cannot contaminate either bucket),
and the SAME universal SELL sign-flip every sibling rank metric uses.

Operational discipline is identical to its siblings: **read-only** — no
train, no ``decision_scorer.pkl`` / ``build_features`` / ``N_FEATURES``
/ trade path touch, never raises on bad input — so it is safe to run
against the live unattended continuous loop and cannot break pickle
compatibility. It does **not** change the gate; that is a separate,
explicit decision. This tool exists only to *inform* it.

Verdict ladder (crisp, threshold-driven so it is exactly testable):

| Verdict | Meaning |
|---------|---------|
| ``INSUFFICIENT_DATA`` | either bucket has < ``MIN_PER_BUCKET`` outcomes — no stable comparison |
| ``GUARD_INVERTED`` | ``trusted_ic < -IC_NOISE`` AND ``abstained_ic > IC_NOISE`` — the gate is filtering OUT the predictions that carry signal and keeping the anti-predictive ones |
| ``GUARD_HARMS`` | ``diff = trusted_ic − abstained_ic ≤ -DIFF_TOL`` — trusted is materially worse than abstained (less severe than INVERTED) |
| ``GUARD_HELPS`` | ``diff ≥ DIFF_TOL`` — trusted is materially better than abstained (the guard works as designed) |
| ``GUARD_NEUTRAL`` | ``|diff| < DIFF_TOL`` — the flag does not separate skill levels |

```bash
cd /home/zeph/trading-intelligence/paper-trader && \\
    python3 -m paper_trader.ml.confidence_skill
cd /home/zeph/trading-intelligence/paper-trader && \\
    python3 -m paper_trader.ml.confidence_skill --json
cd /home/zeph/trading-intelligence/paper-trader && \\
    python3 -m paper_trader.ml.confidence_skill --all  # full corpus
```
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from paper_trader.ml.calibration import _spearman
from paper_trader.ml.decision_scorer import _to_float

# Module-level so tests can assert exact thresholds and a tuning change
# is a single reviewable edit (mirrors action_skill / calibration / gate_audit).
MIN_RECORDS = 30           # overall floor — below this no comparison is stable
MIN_PER_BUCKET = 15        # need this many in each bucket independently
DIFF_TOL = 0.05            # |trusted_ic − abstained_ic| below this reads as no separation
IC_NOISE = 0.05            # |ic| below this is noise-level (mirrors action_skill.IC_MIN)


def _aligned_with_trust(scorer, record: dict) -> tuple[float, float, bool] | None:
    """Return ``(pred, action-aligned realized return, off_distribution)``
    for one outcome record, or ``None`` when the record is unusable.

    Mirrors ``action_skill._aligned_pred`` exactly EXCEPT it uses
    ``predict_with_meta`` (not the scalar ``predict``) so the
    ``off_distribution`` trust flag is captured per row. Rows where the
    predict path raised or produced a non-finite output
    (``failed=True``) are dropped — those are sentinels, not real
    predictions; including them would tie the rank-IC at zero in
    whichever bucket they landed in. The SELL sign-flip is applied only
    to the realized target (not the prediction): the model was trained
    on flipped SELL targets, so its raw output for a SELL feature
    vector already encodes action-aligned goodness — the universal
    convention across every sibling rank metric.
    """
    fr = record.get("forward_return_5d")
    if fr is None:
        return None
    t = _to_float(fr, float("nan"))
    if t != t:
        return None
    pwm = getattr(scorer, "predict_with_meta", None)
    if not callable(pwm):
        return None
    try:
        meta = pwm(
            ml_score=_to_float(record.get("ml_score"), 0.0),
            rsi=record.get("rsi"), macd=record.get("macd"),
            mom5=record.get("mom5"), mom20=record.get("mom20"),
            regime_mult=_to_float(record.get("regime_mult"), 1.0),
            ticker=str(record.get("ticker") or ""),
            vol_ratio=record.get("vol_ratio"),
            bb_pos=record.get("bb_position"),
            news_urgency=record.get("news_urgency"),
            news_article_count=record.get("news_article_count"),
            ema200_above=record.get("ema200_above"),
            hist_cross_up=record.get("hist_cross_up"),
            macd_below_zero_cross=record.get("macd_below_zero_cross"),
        )
    except Exception:
        return None
    if not isinstance(meta, dict):
        return None
    if meta.get("failed"):
        # Sentinel 0.0 — drop the row entirely so it cannot tie either
        # bucket at zero. Same discipline as `_oos_rank_metrics`.
        return None
    try:
        p = float(meta.get("pred", 0.0))
    except (TypeError, ValueError):
        return None
    if p != p:
        return None
    off_dist = bool(meta.get("off_distribution"))
    if str(record.get("action") or "BUY").upper() == "SELL":
        t = -t
    return p, t, off_dist


def _rank_ic(preds: list[float], acts: list[float]) -> float | None:
    """Tie-aware Spearman on n>=2; None otherwise. Pure, deterministic."""
    if len(preds) < 2:
        return None
    ic = _spearman(np.asarray(preds, dtype=np.float64),
                   np.asarray(acts, dtype=np.float64))
    return round(float(ic), 4) if ic == ic else None


def _verdict(trusted_ic: float | None, abstained_ic: float | None,
             n_trusted: int, n_abstained: int) -> tuple[str, str]:
    """Pure verdict + hint string from the two bucket ICs and sample counts.

    Order matters:
      1. INSUFFICIENT_DATA wins on either bucket below MIN_PER_BUCKET — no
         comparison can be honest.
      2. GUARD_INVERTED — the gate filters the WRONG side; this is the
         most operator-actionable red flag and is reported before HARMS so
         a single-glance read prioritizes it.
      3. GUARD_HELPS / GUARD_HARMS — material separation in either direction.
      4. GUARD_NEUTRAL — fall-through. The flag does not separate skill.
    """
    if (n_trusted < MIN_PER_BUCKET) or (n_abstained < MIN_PER_BUCKET):
        return (
            "INSUFFICIENT_DATA",
            (f"need ≥{MIN_PER_BUCKET} outcomes in each bucket "
             f"(trusted={n_trusted}, abstained={n_abstained})"),
        )
    if trusted_ic is None or abstained_ic is None:
        return (
            "INSUFFICIENT_DATA",
            "rank-IC could not be computed in one of the buckets "
            "(n<2 after dropping failed predictions)",
        )
    if trusted_ic < -IC_NOISE and abstained_ic > IC_NOISE:
        return (
            "GUARD_INVERTED",
            (f"trusted IC {trusted_ic:+.3f} is anti-predictive while "
             f"abstained IC {abstained_ic:+.3f} carries signal — the "
             f"off_distribution flag is filtering the WRONG slice"),
        )
    diff = trusted_ic - abstained_ic
    if diff <= -DIFF_TOL:
        return (
            "GUARD_HARMS",
            (f"trusted IC {trusted_ic:+.3f} is materially worse than "
             f"abstained IC {abstained_ic:+.3f} (diff {diff:+.3f}) — "
             f"the gate is removing the more informative predictions"),
        )
    if diff >= DIFF_TOL:
        return (
            "GUARD_HELPS",
            (f"trusted IC {trusted_ic:+.3f} is materially better than "
             f"abstained IC {abstained_ic:+.3f} (diff {diff:+.3f}) — "
             f"the off_distribution guard works as designed"),
        )
    return (
        "GUARD_NEUTRAL",
        (f"trusted IC {trusted_ic:+.3f} ≈ abstained IC {abstained_ic:+.3f} "
         f"(diff {diff:+.3f}) — the off_distribution flag does not "
         f"separate skill levels; the guard is not load-bearing"),
    )


def confidence_skill(scorer, records) -> dict:
    """Confidence-conditional OOS rank skill of a deployed scorer.

    Returns a JSON-safe dict with structure:
    ``{status, verdict, n_records, trusted:{n,rank_ic}, abstained:{n,rank_ic},
       diff, hint}``.

    ``status``:
      * ``ok`` — produced a comparison (may still be ``INSUFFICIENT_DATA``)
      * ``untrained`` — scorer has no model loaded
      * ``unsupported_scorer`` — fake/stub scorer without ``predict_with_meta``

    Never raises — every fault path returns the skeleton with a hint.
    """
    skel = {
        "status": "ok",
        "verdict": "INSUFFICIENT_DATA",
        "n_records": 0,
        "trusted": {"n": 0, "rank_ic": None},
        "abstained": {"n": 0, "rank_ic": None},
        "diff": None,
        "hint": "",
    }
    if not getattr(scorer, "is_trained", False):
        skel["status"] = "untrained"
        skel["hint"] = "scorer is not trained — no predictions to evaluate"
        return skel
    if not callable(getattr(scorer, "predict_with_meta", None)):
        skel["status"] = "unsupported_scorer"
        skel["hint"] = ("scorer lacks predict_with_meta — the off_distribution "
                        "trust flag cannot be captured")
        return skel

    trusted_preds: list[float] = []
    trusted_acts: list[float] = []
    abstained_preds: list[float] = []
    abstained_acts: list[float] = []
    n_aligned = 0
    for r in records:
        action = str(r.get("action") or "BUY").upper()
        if action not in ("BUY", "SELL"):
            continue
        triple = _aligned_with_trust(scorer, r)
        if triple is None:
            continue
        p, t, off = triple
        n_aligned += 1
        if off:
            abstained_preds.append(p)
            abstained_acts.append(t)
        else:
            trusted_preds.append(p)
            trusted_acts.append(t)

    if n_aligned < MIN_RECORDS:
        skel["n_records"] = n_aligned
        skel["hint"] = (f"need ≥{MIN_RECORDS} aligned outcomes overall, "
                        f"have {n_aligned}")
        return skel

    trusted_ic = _rank_ic(trusted_preds, trusted_acts)
    abstained_ic = _rank_ic(abstained_preds, abstained_acts)
    verdict, hint = _verdict(trusted_ic, abstained_ic,
                             len(trusted_preds), len(abstained_preds))
    diff: float | None = None
    if trusted_ic is not None and abstained_ic is not None:
        diff = round(trusted_ic - abstained_ic, 4)
    return {
        "status": "ok",
        "verdict": verdict,
        "n_records": n_aligned,
        "trusted": {"n": len(trusted_preds), "rank_ic": trusted_ic},
        "abstained": {"n": len(abstained_preds), "rank_ic": abstained_ic},
        "diff": diff,
        "hint": hint,
    }


def _load_outcomes(path: Path) -> list[dict]:
    """Same robust JSONL load as the sibling diagnostics. Never raises."""
    rows: list[dict] = []
    try:
        if not path.exists():
            return rows
        for ln in path.read_text().splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
                if isinstance(obj, dict):
                    rows.append(obj)
            except Exception:
                continue
    except Exception:
        return rows
    return rows


def analyze(outcomes_path: "Path | str | None" = None,
            oos_only: bool = True) -> dict:
    """End-to-end CLI/import entrypoint — same shape as ``action_skill.analyze``.

    Loads ``decision_outcomes.jsonl``, optionally restricts to the most
    recent 20% by ``sim_date`` (the same temporal holdout the per-cycle
    skill ledger uses), loads the deployed scorer, returns the report.
    """
    root = Path(__file__).resolve().parent.parent.parent
    if outcomes_path is None:
        outcomes_path = root / "data" / "decision_outcomes.jsonl"
    records = _load_outcomes(Path(outcomes_path))

    slice_label = "all"
    if oos_only and records:
        try:
            from paper_trader.validation import split_outcomes_temporal
            _, oos = split_outcomes_temporal(records, oos_fraction=0.2)
            if oos:
                records = oos
                slice_label = "oos"
        except Exception:
            pass

    try:
        from paper_trader.ml.decision_scorer import DecisionScorer
        scorer = DecisionScorer()
    except Exception as e:
        rep = {
            "status": "error",
            "verdict": "INSUFFICIENT_DATA",
            "n_records": 0,
            "trusted": {"n": 0, "rank_ic": None},
            "abstained": {"n": 0, "rank_ic": None},
            "diff": None,
            "hint": f"scorer load failed: {type(e).__name__}",
            "slice": slice_label,
        }
        return rep

    rep = confidence_skill(scorer, records)
    rep["slice"] = slice_label
    return rep


def _cli() -> int:
    """`python3 -m paper_trader.ml.confidence_skill` — confidence-conditional
    OOS rank skill over the live outcomes corpus. Read-only.

    Exit 0 healthy / insufficient / neutral, 2 if GUARD_HARMS or
    GUARD_INVERTED (so an operator/cron can branch on it). Mirrors
    action_skill._cli and calibration._cli."""
    import argparse, sys

    p = argparse.ArgumentParser(
        prog="python3 -m paper_trader.ml.confidence_skill",
        description="Confidence-conditional scorer skill — compare rank-IC "
                    "on predict_with_meta off_distribution=False vs True "
                    "OOS rows. Read-only; never writes anything.",
    )
    p.add_argument("--json", action="store_true",
                   help="emit machine-readable JSON instead of a table")
    p.add_argument("--all", action="store_true", dest="all_corpus",
                   help="use the full corpus instead of the temporal OOS slice")
    args = p.parse_args(sys.argv[1:])

    rep = analyze(oos_only=not args.all_corpus)
    if args.json:
        print(json.dumps(rep, indent=2, sort_keys=True))
        return 2 if rep["verdict"] in ("GUARD_HARMS", "GUARD_INVERTED") else 0

    print(f"slice={rep.get('slice', 'all')}  aligned_outcomes={rep['n_records']}")
    print(f"VERDICT: {rep['verdict']}  ({rep['hint']})")
    print(f"  {'bucket':<10} {'n':>6} {'rank_ic':>9}")
    tr = rep["trusted"]
    ab = rep["abstained"]
    tr_s = f"{tr['rank_ic']:>+9.3f}" if tr['rank_ic'] is not None else f"{'n/a':>9}"
    ab_s = f"{ab['rank_ic']:>+9.3f}" if ab['rank_ic'] is not None else f"{'n/a':>9}"
    print(f"  {'trusted':<10} {tr['n']:>6} {tr_s}")
    print(f"  {'abstained':<10} {ab['n']:>6} {ab_s}")
    if rep["diff"] is not None:
        print(f"  diff (trusted − abstained) = {rep['diff']:+.3f}")
    return 2 if rep["verdict"] in ("GUARD_HARMS", "GUARD_INVERTED") else 0


if __name__ == "__main__":
    raise SystemExit(_cli())
