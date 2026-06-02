"""Leveraged-vs-non-leveraged OOS skill breakdown for the DecisionScorer.

A skeptical-quant gap left open by every existing scorer-skill audit. The
production ``decision_outcomes.jsonl`` is empirically 35% leveraged ETFs
(SOXL alone = 12.6%, TQQQ = 9.6%), and the live BUY conviction-gate
acts on the SAME ``predict`` output for leveraged-and-non-leveraged BUYs
through one ladder (``_ml_decide`` ±10 / ±5 / 0). If the scorer's apparent
OOS rank-IC is carried mainly by leveraged-ETF beta amplification (a
strong-trending bull window predictably pushes TQQQ/SOXL up 5–15% in 5
trading days regardless of news / quant features), then gating on the
SAME predictions for a non-leveraged BUY is gating on noise.

The existing audits cannot answer this question:

  * ``sector_skill``     groups SOXL, TQQQ, NVDA, AAPL all under "tech" —
                         leveraged ETFs and their underliers collapse into
                         one bucket whose IC is dominated by the leveraged
                         tail.
  * ``persona_skill``    measures per-persona aggregate skill — persona is
                         not synonymous with leverage class.
  * ``baseline_compare`` measures overall scorer-vs-trivial — silent on
                         WHICH subset the edge lives in.

``leveraged_skill`` splits the OOS slice strictly on ``ticker IN
_LEVERAGED_ETFS`` (the SAME constant ``_ml_decide``'s conviction-cap arm
uses for the elevated 40% cap) and reports per-bucket rank-IC, dir_acc,
RMSE, mean_pred, mean_realized:

| Field | Meaning |
|---|---|
| ``n_train`` | training rows in this bucket — "is the scorer seeing this class at all?" |
| ``n_oos`` | OOS rows in this bucket |
| ``mean_pred`` / ``mean_realized`` | bucket-level mean predicted vs realized 5d return (%); a big positive gap on the leveraged bucket is a magnitude bias toward the leveraged tail |
| ``rmse`` | OOS RMSE for this bucket |
| ``dir_acc`` | OOS directional accuracy (0.5 = coin-flip) |
| ``rank_ic`` | tie-aware Spearman(pred, realized) — SAME ``_spearman`` as ``calibration`` / ``sector_skill`` / ``persona_skill`` (single source of truth, never drifts) |
| ``verdict`` | per-bucket threshold-driven label (see below) |

Per-bucket verdict (each is exactly testable on a threshold):

| Verdict | Trigger |
|---|---|
| ``SPARSE`` | ``n_oos < MIN_OUTCOMES_PER_BUCKET`` — Spearman is not stable |
| ``INVERTED_SIGNAL`` | ``rank_ic ≤ -IC_GOOD`` — actively anti-predictive in this bucket |
| ``SIGNAL_EDGE`` | ``rank_ic ≥ IC_GOOD`` — real rank skill in this bucket |
| ``WEAK_SIGNAL_EDGE`` | ``IC_MIN ≤ rank_ic < IC_GOOD`` |
| ``NO_SIGNAL_EDGE`` | otherwise |

Overall verdict ladder:

| Verdict | Trigger |
|---|---|
| ``INSUFFICIENT_DATA`` | fewer than ``MIN_RECORDS`` aligned OOS rows total |
| ``SCORER_UNTRAINED`` | ``is_trained=False`` |
| ``HAS_INVERTED_BUCKET`` | ≥1 bucket verdict is ``INVERTED_SIGNAL`` — the gate is harmful in that bucket |
| ``LEVERAGED_ONLY_EDGE`` | leveraged ≥ ``SIGNAL_EDGE`` AND non-leveraged < ``WEAK_SIGNAL_EDGE`` — scorer's edge is beta-amplification, not name selection |
| ``NONLEVERAGED_ONLY_EDGE`` | non-leveraged ≥ ``SIGNAL_EDGE`` AND leveraged < ``WEAK_SIGNAL_EDGE`` |
| ``LEVERAGED_DOMINATES`` | both have edge but leveraged ``rank_ic`` exceeds non-leveraged by ≥ ``IC_DOMINANCE_GAP`` (default 0.10) — same caveat as ``LEVERAGED_ONLY_EDGE``, milder |
| ``BALANCED_EDGE`` | both have edge and gap < ``IC_DOMINANCE_GAP`` — the gate generalises |
| ``NO_EDGE`` | neither bucket reaches ``WEAK_SIGNAL_EDGE`` |

Operational discipline matches the rest of ``paper_trader/ml`` (read-only,
never raises, never touches ``decision_scorer.pkl`` / ``build_features`` /
``N_FEATURES`` / trade path). Safe under the live unattended continuous
loop; cannot break pickle compatibility.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Single source of truth for "is this ticker leveraged" — IMPORTED from
# backtest.py (not redefined) so the leveraged-class membership here is
# in lockstep with what ``_ml_decide`` actually treats as a leveraged ETF
# for the elevated conviction cap. A future edit to that constant shifts
# every consumer in one place.
from paper_trader.backtest import _LEVERAGED_ETFS
from paper_trader.ml.decision_scorer import _to_float
# Tie-aware rank correlation reused from calibration so this OOS rank-IC
# and the in-sample calibration Spearman (and sector_skill / persona_skill /
# _oos_rank_metrics) can never drift. Tie-awareness is load-bearing —
# PRED_CLAMP_PCT ties off-distribution predictions at exactly ±50; a
# non-tie-aware Spearman fabricates rank skill there.
from paper_trader.ml.calibration import _spearman


# Module-level thresholds so tests assert exact verdicts and a tuning
# change is a single reviewable edit (the codebase's
# constants-at-module-scope convention; mirrors sector_skill / persona_skill).
MIN_RECORDS = 30                  # min aligned OOS rows overall for any verdict
MIN_OUTCOMES_PER_BUCKET = 20      # below this a bucket's Spearman is not stable
IC_MIN = 0.05                     # below |this| there is essentially no rank skill
IC_GOOD = 0.15                    # "real edge" bar (mirrors sector_skill.IC_GOOD)
IC_DOMINANCE_GAP = 0.10           # leveraged−nonleveraged IC gap that flips
                                  # BALANCED into LEVERAGED_DOMINATES

# Public verdict tuple — tests pin its exact contents so a future
# verdict rename is a single reviewable edit. Mirrors the
# `paper_trader.ml.sector_skill.VERDICTS` discipline.
VERDICTS: tuple[str, ...] = (
    "INSUFFICIENT_DATA",
    "SCORER_UNTRAINED",
    "HAS_INVERTED_BUCKET",
    "LEVERAGED_ONLY_EDGE",
    "NONLEVERAGED_ONLY_EDGE",
    "LEVERAGED_DOMINATES",
    "BALANCED_EDGE",
    "NO_EDGE",
)


def _bucket_of(ticker: str) -> str:
    """Return ``'leveraged'`` if ``ticker`` is in ``_LEVERAGED_ETFS`` else
    ``'nonleveraged'``. Ticker is upper-cased so a lower/mixed-case record
    (e.g. an external import that didn't normalise) maps to the same
    bucket ``_ml_decide`` would (which itself looks up uppercase tickers
    against the same set).
    """
    return "leveraged" if str(ticker or "").upper() in _LEVERAGED_ETFS \
        else "nonleveraged"


def _aligned_oos_pair(record: dict, scorer) -> tuple[float, float] | None:
    """Return ``(pred, realized)`` for one OOS record, action-aligned, or
    ``None`` to drop.

    Universal SELL sign-flip applied to ``realized`` (mirrors
    ``evaluate_scorer_oos`` / ``_oos_rank_metrics`` / ``sector_skill`` /
    ``train_scorer`` — one consistent meaning of "good"). A missing /
    non-finite ``forward_return_5d`` or a ``scorer.predict`` exception
    drops the row; a single poisoned outcomes line cannot corrupt the
    bucket report. The 11-kwarg ``predict`` signature is the EXACT one
    every other OOS diagnostic uses — keep it identical so a future
    ``predict`` signature refactor must update one call shape, not two.
    """
    fr = record.get("forward_return_5d")
    if fr is None:
        return None
    realized = _to_float(fr, float("nan"))
    if realized != realized:
        return None
    try:
        pred = scorer.predict(
            ml_score=_to_float(record.get("ml_score"), 0.0),
            rsi=record.get("rsi"),
            macd=record.get("macd"),
            mom5=record.get("mom5"),
            mom20=record.get("mom20"),
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
    pred = _to_float(pred, float("nan"))
    if pred != pred:
        return None
    if str(record.get("action") or "BUY").upper() == "SELL":
        realized = -realized
    return float(pred), float(realized)


def _verdict_for_bucket(n: int, ic: float) -> str:
    """Pin per-bucket verdict on (sample count, rank-IC) — mirrors
    ``sector_skill._verdict_for_sector`` exactly (constants and thresholds
    intentionally aligned so a quant reading both diagnostics sees one
    consistent ladder)."""
    if n < MIN_OUTCOMES_PER_BUCKET:
        return "SPARSE"
    if ic <= -IC_GOOD:
        return "INVERTED_SIGNAL"
    if ic >= IC_GOOD:
        return "SIGNAL_EDGE"
    if ic >= IC_MIN:
        return "WEAK_SIGNAL_EDGE"
    return "NO_SIGNAL_EDGE"


def _bucket_metrics(pred: np.ndarray, real: np.ndarray) -> dict:
    """RMSE / dir_acc / rank_ic / mean_pred / mean_realized for one bucket.

    ``_spearman`` returns 0.0 (never NaN) for a constant predictor — the
    documented "no rank skill" reading, not a tie-ordering artifact.
    ``dir_acc`` only counts pairs where BOTH sides carry a non-zero sign
    (a 0 has no direction); a bucket of all zeros yields ``dir_acc=None``
    rather than a misleading 0.5.
    """
    n = int(len(pred))
    if n == 0:
        return {"n_oos": 0, "mean_pred": None, "mean_realized": None,
                "magnitude_bias": None, "rmse": None,
                "dir_acc": None, "rank_ic": None}
    rmse = float(np.sqrt(np.mean((pred - real) ** 2)))
    dir_pairs = [(p, a) for p, a in zip(pred, real) if p != 0.0 and a != 0.0]
    if dir_pairs:
        hits = sum(1 for p, a in dir_pairs if (p > 0) == (a > 0))
        dir_acc = round(hits / len(dir_pairs), 4)
    else:
        dir_acc = None
    ic = round(float(_spearman(pred, real)), 4)
    mean_pred = round(float(pred.mean()), 4)
    mean_real = round(float(real.mean()), 4)
    return {
        "n_oos": n,
        "mean_pred": mean_pred,
        "mean_realized": mean_real,
        "magnitude_bias": round(mean_pred - mean_real, 4),
        "rmse": round(rmse, 4),
        "dir_acc": dir_acc,
        "rank_ic": ic,
    }


def leveraged_skill(
    scorer,
    train_records: list[dict],
    oos_records: list[dict],
) -> dict:
    """Aggregate leveraged-vs-non-leveraged OOS skill against ``scorer``.

    Inputs mirror ``sector_skill`` exactly:
      * ``scorer`` — anything implementing the 11-kwarg ``predict(...)``
        contract (real ``DecisionScorer`` or a test double).
      * ``train_records`` / ``oos_records`` — the temporal split returned
        by ``validation.split_outcomes_temporal``. The training side is
        only used to surface per-bucket training counts (``n_train``) so
        a quant can see how thinly each leverage class was trained.

    Returns a JSON-safe dict with the per-bucket breakdown plus an overall
    verdict from the ``VERDICTS`` tuple. Never raises; on any fault yields
    ``status='error'`` with a hint.
    """
    if not getattr(scorer, "is_trained", False):
        return {
            "status": "error",
            "verdict": "SCORER_UNTRAINED",
            "n_train": 0,
            "n_oos": 0,
            "buckets": [],
            "leveraged_share_oos": None,
            "ic_gap_leveraged_minus_nonleveraged": None,
            "hint": ("scorer.is_trained is False — leveraged skill is "
                     "meaningless on an untrained model; accumulate ≥30 "
                     "deduped outcomes then retrain"),
        }

    # Per-bucket training counts — surface how thin training was for
    # each leverage class. With ~65% non-leveraged in production, a
    # quant wants to know whether the scorer has actually seen enough
    # of each class to learn something class-specific.
    train_by_bucket: dict[str, int] = {"leveraged": 0, "nonleveraged": 0}
    for r in train_records or []:
        train_by_bucket[_bucket_of(r.get("ticker") or "")] += 1

    # Build per-bucket OOS (pred, realized) lists. Same alignment used by
    # sector_skill / persona_skill / evaluate_scorer_oos / _oos_rank_metrics.
    buckets: dict[str, dict] = {
        "leveraged": {"pred": [], "real": []},
        "nonleveraged": {"pred": [], "real": []},
    }
    n_aligned = 0
    for r in oos_records or []:
        pair = _aligned_oos_pair(r, scorer)
        if pair is None:
            continue
        pred, realized = pair
        b = _bucket_of(r.get("ticker") or "")
        buckets[b]["pred"].append(pred)
        buckets[b]["real"].append(realized)
        n_aligned += 1

    if n_aligned < MIN_RECORDS:
        return {
            "status": "insufficient_data",
            "verdict": "INSUFFICIENT_DATA",
            "n_train": sum(train_by_bucket.values()),
            "n_oos": n_aligned,
            "buckets": [],
            "leveraged_share_oos": None,
            "ic_gap_leveraged_minus_nonleveraged": None,
            "hint": (f"need ≥{MIN_RECORDS} aligned OOS outcomes with a "
                     f"finite forward_return_5d and a scorer prediction; "
                     f"have {n_aligned}"),
        }

    buckets_out: list[dict] = []
    bucket_verdicts: dict[str, str] = {}
    bucket_ics: dict[str, float | None] = {"leveraged": None,
                                            "nonleveraged": None}
    for name in ("leveraged", "nonleveraged"):
        b = buckets[name]
        pred = np.asarray(b["pred"], dtype=np.float64)
        real = np.asarray(b["real"], dtype=np.float64)
        m = _bucket_metrics(pred, real)
        verdict = _verdict_for_bucket(m["n_oos"], m["rank_ic"] or 0.0)
        bucket_verdicts[name] = verdict
        bucket_ics[name] = m["rank_ic"]
        buckets_out.append({
            "bucket": name,
            "n_train": int(train_by_bucket.get(name, 0)),
            **m,
            "verdict": verdict,
        })

    inverted = [b for b in ("leveraged", "nonleveraged")
                if bucket_verdicts[b] == "INVERTED_SIGNAL"]
    n_lev = next(b["n_oos"] for b in buckets_out if b["bucket"] == "leveraged")
    leveraged_share = round(n_lev / n_aligned, 4) if n_aligned > 0 else None

    ic_l = bucket_ics["leveraged"]
    ic_n = bucket_ics["nonleveraged"]
    # ic_gap only defined when both buckets produced a Spearman.
    if ic_l is not None and ic_n is not None:
        ic_gap = round(ic_l - ic_n, 4)
    else:
        ic_gap = None

    has_lev_edge = bucket_verdicts["leveraged"] == "SIGNAL_EDGE"
    has_non_edge = bucket_verdicts["nonleveraged"] == "SIGNAL_EDGE"
    weak_lev = bucket_verdicts["leveraged"] in (
        "SIGNAL_EDGE", "WEAK_SIGNAL_EDGE",
    )
    weak_non = bucket_verdicts["nonleveraged"] in (
        "SIGNAL_EDGE", "WEAK_SIGNAL_EDGE",
    )

    if inverted:
        verdict = "HAS_INVERTED_BUCKET"
        hint = (f"{len(inverted)} bucket(s) have anti-predictive rank skill "
                f"(rank_ic ≤ -{IC_GOOD}): {', '.join(sorted(inverted))}. "
                f"Gating BUYs in this bucket on the scorer's prediction "
                f"is actively harmful — every documented gate diagnostic "
                f"(gate_audit / gate_realized / gate_pnl) treats the gate "
                f"as on/off uniformly; this verdict says the scorer's "
                f"sign is wrong in one half of the universe.")
    elif has_lev_edge and not weak_non:
        verdict = "LEVERAGED_ONLY_EDGE"
        hint = (f"scorer rank-IC on leveraged ETFs = "
                f"{ic_l:+.3f} (≥ {IC_GOOD}) but on non-leveraged names = "
                f"{ic_n if ic_n is not None else 'n/a'}. The scorer's edge "
                f"is concentrated on the beta-amplified class — the gate's "
                f"value on a non-leveraged BUY is likely noise.")
    elif has_non_edge and not weak_lev:
        verdict = "NONLEVERAGED_ONLY_EDGE"
        hint = (f"scorer rank-IC on non-leveraged names = "
                f"{ic_n:+.3f} (≥ {IC_GOOD}) but on leveraged ETFs = "
                f"{ic_l if ic_l is not None else 'n/a'}. The scorer "
                f"generalises across single-name selection but not across "
                f"leveraged-ETF momentum — atypical for this corpus.")
    elif weak_lev and weak_non:
        # Both buckets have at least weak edge. Decide between
        # LEVERAGED_DOMINATES and BALANCED_EDGE on the IC gap.
        if (ic_l is not None and ic_n is not None
                and (ic_l - ic_n) >= IC_DOMINANCE_GAP):
            verdict = "LEVERAGED_DOMINATES"
            hint = (f"both buckets show rank skill but leveraged "
                    f"({ic_l:+.3f}) exceeds non-leveraged ({ic_n:+.3f}) "
                    f"by ≥ {IC_DOMINANCE_GAP}. The headline scorer "
                    f"oos_ic from the skill ledger is being lifted by "
                    f"the leveraged tail; per-bucket reads should be "
                    f"the primary signal, not the headline number.")
        else:
            verdict = "BALANCED_EDGE"
            hint = (f"both buckets show rank skill and the gap "
                    f"(leveraged − non-leveraged = "
                    f"{ic_gap if ic_gap is not None else 'n/a'}) is "
                    f"< {IC_DOMINANCE_GAP} — the scorer generalises "
                    f"across the leverage axis. The conviction gate's "
                    f"prediction carries comparable edge on both halves "
                    f"of the universe.")
    else:
        verdict = "NO_EDGE"
        hint = (f"neither bucket reaches the weak-edge bar "
                f"(rank_ic ≥ {IC_MIN}). leveraged={ic_l if ic_l is not None else 'n/a'}, "
                f"non-leveraged={ic_n if ic_n is not None else 'n/a'}. "
                f"The conviction gate's modulation is on a model with no "
                f"per-class rank skill on the temporal holdout.")

    # Sort: leveraged first when there's bucket-level edge concentration there
    # (matches the verdict's "leveraged tail dominates" reading); otherwise
    # by rank_ic descending. SPARSE sinks to the bottom regardless.
    def _sort_key(d: dict) -> tuple:
        is_sparse = d["verdict"] == "SPARSE"
        ic = d.get("rank_ic") or 0.0
        return (is_sparse, -ic)
    buckets_out.sort(key=_sort_key)

    return {
        "status": "ok",
        "verdict": verdict,
        "n_train": sum(train_by_bucket.values()),
        "n_oos": n_aligned,
        "leveraged_share_oos": leveraged_share,
        "ic_gap_leveraged_minus_nonleveraged": ic_gap,
        "buckets": buckets_out,
        "inverted_buckets": sorted(inverted),
        "hint": hint,
    }


def _load_outcomes(path: Path) -> list[dict]:
    """Robust JSONL load — skips unparseable lines, never raises. Mirrors
    ``sector_skill._load_outcomes`` so a CLI consumer reads the file the
    same way every diagnostic does.
    """
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


def analyze(
    outcomes_path: Path | str | None = None,
    oos_fraction: float = 0.2,
) -> dict:
    """End-to-end: load outcomes, temporal split, load deployed scorer,
    compute per-bucket OOS skill. The function the CLI / endpoint share.

    Never raises — on any fault returns ``{status='error', verdict=...,
    hint=...}``. ``outcomes_path`` defaults to the repo's canonical
    ``data/decision_outcomes.jsonl`` (override for tests).
    """
    if outcomes_path is None:
        outcomes_path = (Path(__file__).resolve().parent.parent.parent
                         / "data" / "decision_outcomes.jsonl")
    try:
        from paper_trader.validation import split_outcomes_temporal
    except Exception as exc:
        return {"status": "error", "verdict": "INSUFFICIENT_DATA",
                "n_train": 0, "n_oos": 0, "buckets": [],
                "leveraged_share_oos": None,
                "ic_gap_leveraged_minus_nonleveraged": None,
                "inverted_buckets": [],
                "hint": f"validation module unavailable: "
                        f"{type(exc).__name__}: {exc}"}
    try:
        from paper_trader.ml.decision_scorer import DecisionScorer
    except Exception as exc:
        return {"status": "error", "verdict": "INSUFFICIENT_DATA",
                "n_train": 0, "n_oos": 0, "buckets": [],
                "leveraged_share_oos": None,
                "ic_gap_leveraged_minus_nonleveraged": None,
                "inverted_buckets": [],
                "hint": f"decision_scorer unavailable: "
                        f"{type(exc).__name__}: {exc}"}

    recs = _load_outcomes(Path(outcomes_path))
    if not recs:
        return {"status": "insufficient_data", "verdict": "INSUFFICIENT_DATA",
                "n_train": 0, "n_oos": 0, "buckets": [],
                "leveraged_share_oos": None,
                "ic_gap_leveraged_minus_nonleveraged": None,
                "inverted_buckets": [],
                "hint": f"no records loaded from {outcomes_path}"}
    train, oos = split_outcomes_temporal(recs, oos_fraction=oos_fraction)
    scorer = DecisionScorer()
    return leveraged_skill(scorer, train, oos)


def _cli(argv: list[str] | None = None) -> int:
    """``python3 -m paper_trader.ml.leveraged_skill [--json]`` — per-bucket
    OOS skill over the live ``decision_outcomes.jsonl``. Read-only; never
    writes. Mirrors ``sector_skill._cli`` / ``calibration._cli``.

    Exit codes:
      * 0 — HEALTHY-ish ladder (BALANCED_EDGE / NONLEVERAGED_ONLY_EDGE /
            NO_EDGE / LEVERAGED_DOMINATES / LEVERAGED_ONLY_EDGE /
            INSUFFICIENT_DATA)
      * 1 — SCORER_UNTRAINED / other recoverable error
      * 2 — HAS_INVERTED_BUCKET (an operator/cron can branch on it)
    """
    import argparse
    import sys

    p = argparse.ArgumentParser(
        prog="python3 -m paper_trader.ml.leveraged_skill",
        description="Leveraged-vs-non-leveraged OOS rank skill breakdown "
                    "for the DecisionScorer (read-only).",
    )
    p.add_argument("--json", action="store_true",
                   help="Emit JSON instead of a human-readable table.")
    args = p.parse_args(sys.argv[1:] if argv is None else argv)

    rep = analyze()
    if args.json:
        print(json.dumps(rep, indent=2, sort_keys=True))
    else:
        print(f"oos_outcomes={rep.get('n_oos', 0)}  "
              f"train_outcomes={rep.get('n_train', 0)}  "
              f"leveraged_share_oos={rep.get('leveraged_share_oos')}  "
              f"ic_gap={rep.get('ic_gap_leveraged_minus_nonleveraged')}")
        print(f"VERDICT: {rep['verdict']}  ({rep.get('hint','')})")
        if rep.get("buckets"):
            print(f"  {'bucket':<14} {'n_tr':>6} {'n_oos':>6} {'rmse':>7} "
                  f"{'dir_acc':>8} {'rank_ic':>9} {'meanP':>7} {'meanR':>7}"
                  f"  verdict")
            for e in rep["buckets"]:
                da = e["dir_acc"]
                da_s = f"{da*100:>7.1f}%" if da is not None else "    n/a"
                ic = e.get("rank_ic")
                ic_s = f"{ic:>+9.3f}" if ic is not None else "      n/a"
                mp = e.get("mean_pred")
                mp_s = f"{mp:>+7.2f}" if mp is not None else "    n/a"
                mr = e.get("mean_realized")
                mr_s = f"{mr:>+7.2f}" if mr is not None else "    n/a"
                rmse = e.get("rmse")
                rmse_s = f"{rmse:>7.2f}" if rmse is not None else "    n/a"
                print(f"  {e['bucket']:<14} {e['n_train']:>6} {e['n_oos']:>6}"
                      f" {rmse_s} {da_s} {ic_s} {mp_s} {mr_s}"
                      f"  {e['verdict']}")
    if rep["verdict"] == "HAS_INVERTED_BUCKET":
        return 2
    if rep.get("status") == "error":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
