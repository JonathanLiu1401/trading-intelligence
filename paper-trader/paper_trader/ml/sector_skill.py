"""Per-sector OOS skill breakdown for the DecisionScorer.

A quant researcher's second question after "is the scorer calibrated?"
(``ml.calibration``) is "is the calibration uniform across the universe,
or is it carried by one fat sector?" The 17-feature ``build_features``
includes a 7-way sector one-hot, but ``SECTOR_MAP`` is dramatically
uneven — empirically the live ``decision_outcomes.jsonl`` tail is
~89% tech, ~9% financials, ~2% crypto, and the energy / healthcare /
commodities / other buckets see ≤1 sample per cycle. A model trained on
that mix is structurally near-mute on every non-tech sector — its
sector-one-hot weights for the rare classes are essentially random, and
its overall OOS rank-IC can look healthy purely on the tech mass.

``sector_skill`` answers per-sector, on the SAME temporal holdout the
scorer-skill ledger uses (``validation.split_outcomes_temporal``):

| Field | Meaning |
|---|---|
| ``n_train`` | training rows in this sector (the "is this sector even seen by the scorer?" count) |
| ``n_oos`` | OOS rows in this sector |
| ``mean_pred`` / ``mean_realized`` | sector-level mean predicted vs realized 5d return (%); a big gap is sector-level magnitude bias |
| ``rmse`` | OOS RMSE for this sector (compare to σ(target) for that sector — a model worse than a constant predictor adds noise) |
| ``dir_acc`` | OOS directional accuracy (a coin-flip = 0.5) |
| ``rank_ic`` | tie-aware Spearman(pred, realized) on OOS — same ``_spearman`` as ``calibration`` / ``persona_skill`` / ``_oos_rank_metrics`` (single source of truth, never drifts) |
| ``verdict`` | crisp threshold-driven per-sector classification (below) |

Per-sector verdict (each is exactly testable on a threshold):

| Verdict | Trigger |
|---|---|
| ``SPARSE`` | ``n_oos < MIN_OUTCOMES_PER_SECTOR`` — Spearman is not stable |
| ``INVERTED_SIGNAL`` | ``rank_ic ≤ -IC_GOOD`` — the more confident the scorer is, the WORSE the realized 5d outcome in this sector |
| ``SIGNAL_EDGE`` | ``rank_ic ≥ IC_GOOD`` — sector carries real rank skill |
| ``WEAK_SIGNAL_EDGE`` | ``IC_MIN ≤ rank_ic < IC_GOOD`` |
| ``NO_SIGNAL_EDGE`` | otherwise |

Overall verdict: ``INSUFFICIENT_DATA`` (<``MIN_RECORDS`` aligned OOS rows),
``HAS_INVERTED_SECTOR`` (≥1 ``INVERTED_SIGNAL`` — actionable red flag,
the scorer is actively HARMFUL in that sector), ``SECTOR_CONCENTRATED``
(any single sector holds ≥``SECTOR_CONCENTRATION_THRESHOLD`` of OOS rows
— the headline ``oos_ic`` from the scorer ledger is dominated by it),
``NO_SECTOR_EDGE`` (no sector reaches ``SIGNAL_EDGE``), or ``HEALTHY``.

Operational discipline matches the rest of ``paper_trader/ml`` (read-only,
never raises, never touches ``decision_scorer.pkl`` / ``build_features`` /
``N_FEATURES`` / trade path), so it is safe under the live unattended
continuous loop and cannot break pickle compatibility.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Single source of truth for ticker → sector. Importing (not
# reimplementing) the map means a future SECTOR_MAP edit can never
# silently shift every historical aggregate — the same discipline
# persona_skill uses for ``persona_for``.
from paper_trader.ml.decision_scorer import SECTOR_MAP, _to_float
# Reuse the tie-aware rank correlation from calibration — so this OOS
# sector-IC and the in-sample calibration spearman (and persona_skill's IC
# and _oos_rank_metrics's IC) can never drift. Tie-awareness is
# load-bearing because PRED_CLAMP_PCT ties off-distribution predictions at
# exactly ±50; a non-tie-aware Spearman fabricates rank skill there.
from paper_trader.ml.calibration import _spearman

# Module-level thresholds so tests assert exact verdicts and a tuning
# change is a single reviewable edit (the codebase's
# constants-at-module-scope convention).
MIN_RECORDS = 30                       # min aligned OOS rows overall for any verdict
MIN_OUTCOMES_PER_SECTOR = 20           # below this a sector's Spearman is not stable
IC_MIN = 0.05                          # below |this| there is essentially no rank skill
IC_GOOD = 0.15                         # "real edge" bar (mirrors persona_skill.IC_GOOD)
SECTOR_CONCENTRATION_THRESHOLD = 0.70  # ≥ this fraction of OOS in one sector ⇒ concentration


def _sector_of(ticker: str) -> str:
    """Return the SECTOR_MAP sector for ``ticker``; ``"other"`` on miss.

    Mirrors ``build_features``' default exactly — a single source of truth
    so a SECTOR_MAP edit (a ticker reassigned to a new sector) shifts every
    consumer in lockstep. ``ticker`` is upper-cased so a lower/mixed-case
    record (e.g. an external import that didn't normalise) still maps to
    the same bucket the trained model used.
    """
    return SECTOR_MAP.get(str(ticker or "").upper(), "other")


def _aligned_oos_pair(record: dict, scorer) -> tuple[float, float] | None:
    """Return ``(pred, realized)`` for one OOS record, action-aligned, or
    ``None`` to drop.

    Universal SELL sign-flip applied to ``realized`` (mirrors
    ``evaluate_scorer_oos`` / ``_oos_rank_metrics`` / ``train_scorer`` —
    one consistent meaning of "good") so this sector OOS metric describes
    the same prediction path the gate uses. A missing/non-finite
    ``forward_return_5d`` or a ``scorer.predict`` exception drops the row;
    a single poisoned outcomes line cannot corrupt the sector report.

    The 11-kwarg ``predict`` signature is the exact one every other OOS
    diagnostic uses — keep it identical so a future ``predict`` signature
    refactor must update one call shape, not two slightly different ones.
    """
    fr = record.get("forward_return_5d")
    if fr is None:
        return None
    realized = _to_float(fr, float("nan"))
    if realized != realized:  # NaN ⇒ drop
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
        )
    except Exception:
        return None
    pred = _to_float(pred, float("nan"))
    if pred != pred:  # NaN ⇒ drop
        return None
    if str(record.get("action") or "BUY").upper() == "SELL":
        realized = -realized
    return float(pred), float(realized)


def _verdict_for_sector(n: int, ic: float) -> str:
    if n < MIN_OUTCOMES_PER_SECTOR:
        return "SPARSE"
    if ic <= -IC_GOOD:
        return "INVERTED_SIGNAL"
    if ic >= IC_GOOD:
        return "SIGNAL_EDGE"
    if ic >= IC_MIN:
        return "WEAK_SIGNAL_EDGE"
    return "NO_SIGNAL_EDGE"


def sector_skill(
    scorer,
    train_records: list[dict],
    oos_records: list[dict],
) -> dict:
    """Aggregate per-sector OOS skill against the provided scorer.

    Inputs:
      * ``scorer`` — anything implementing the 11-kwarg
        ``predict(ml_score, rsi, macd, mom5, mom20, regime_mult, ticker,
        vol_ratio, bb_pos, news_urgency, news_article_count)`` contract
        (the real ``DecisionScorer`` or a test double).
      * ``train_records`` / ``oos_records`` — the temporal split returned
        by ``validation.split_outcomes_temporal``. The training side is
        used ONLY to surface the per-sector training-row count
        (``n_train``) so a quant can see how thinly the sector was trained.

    Returns a JSON-safe dict with the per-sector breakdown plus an overall
    verdict. ``sectors`` is sorted by ``rank_ic`` descending (``SPARSE``
    sectors sink to the bottom because their IC is not stable). Never
    raises; on a fault yields ``status='error'`` with a hint.
    """
    if not getattr(scorer, "is_trained", False):
        return {
            "status": "error",
            "verdict": "SCORER_UNTRAINED",
            "n_train": 0, "n_oos": 0, "sectors": [],
            "inverted_sectors": [],
            "hint": ("scorer.is_trained is False — sector skill is meaningless "
                     "on an untrained model; accumulate ≥30 deduped outcomes "
                     "then retrain"),
        }

    # Per-sector training counts — surface how thin training was for each
    # sector. A sector with n_train≈0 has random sector-one-hot weights;
    # any rank skill in OOS for it is statistical noise, not learned.
    train_by_sector: dict[str, int] = {}
    for r in train_records or []:
        train_by_sector[_sector_of(r.get("ticker") or "")] = (
            train_by_sector.get(_sector_of(r.get("ticker") or ""), 0) + 1
        )

    # Build per-sector OOS (pred, realized) lists.
    buckets: dict[str, dict] = {}
    n_aligned = 0
    for r in oos_records or []:
        pair = _aligned_oos_pair(r, scorer)
        if pair is None:
            continue
        pred, realized = pair
        sec = _sector_of(r.get("ticker") or "")
        b = buckets.setdefault(sec, {"pred": [], "real": []})
        b["pred"].append(pred)
        b["real"].append(realized)
        n_aligned += 1

    if n_aligned < MIN_RECORDS:
        return {
            "status": "insufficient_data",
            "verdict": "INSUFFICIENT_DATA",
            "n_train": sum(train_by_sector.values()),
            "n_oos": n_aligned,
            "sectors": [],
            "inverted_sectors": [],
            "hint": (f"need ≥{MIN_RECORDS} aligned OOS outcomes with a finite "
                     f"forward_return_5d and a scorer prediction; "
                     f"have {n_aligned}"),
        }

    sectors_out: list[dict] = []
    inverted: list[str] = []
    for sec, b in buckets.items():
        n = len(b["pred"])
        pred = np.asarray(b["pred"], dtype=np.float64)
        real = np.asarray(b["real"], dtype=np.float64)
        rmse = float(np.sqrt(np.mean((pred - real) ** 2)))
        # Directional accuracy: only count pairs where BOTH sides have a
        # non-zero sign (a 0 has no direction). A sector full of zeros
        # yields dir_acc=None — surfaced as None rather than a misleading
        # constant 0.5 (the universal SELL convention has already been
        # applied to ``real`` via ``_aligned_oos_pair``).
        dir_pairs = [(p, a) for p, a in zip(pred, real) if p != 0.0 and a != 0.0]
        if dir_pairs:
            hits = sum(1 for p, a in dir_pairs if (p > 0) == (a > 0))
            dir_acc = round(hits / len(dir_pairs), 4)
        else:
            dir_acc = None
        # _spearman returns 0.0 (never NaN) for a constant predictor — that
        # is the documented "no rank skill" reading, NOT a tie-ordering
        # artifact. Round-trip through round() so the JSON output is stable.
        ic = round(float(_spearman(pred, real)), 4)
        mean_pred = round(float(pred.mean()), 4)
        mean_real = round(float(real.mean()), 4)
        verdict = _verdict_for_sector(n, ic)
        if verdict == "INVERTED_SIGNAL":
            inverted.append(sec)
        sectors_out.append({
            "sector": sec,
            "n_train": int(train_by_sector.get(sec, 0)),
            "n_oos": n,
            "mean_pred": mean_pred,
            "mean_realized": mean_real,
            "magnitude_bias": round(mean_pred - mean_real, 4),
            "rmse": round(rmse, 4),
            "dir_acc": dir_acc,
            "rank_ic": ic,
            "verdict": verdict,
        })

    # Sort by rank_ic desc; SPARSE sinks last regardless of its
    # (unstable, small-n) IC.
    sectors_out.sort(
        key=lambda d: (d["verdict"] != "SPARSE", d["rank_ic"]),
        reverse=True,
    )

    # Sector concentration: any single sector ≥ threshold of the OOS rows
    # means the headline scorer rank-IC is essentially that sector's IC —
    # a quant should read the overall number with that caveat. Picks the
    # sector with the largest n_oos for the hint.
    max_n = max((s["n_oos"] for s in sectors_out), default=0)
    concentrated_sector: str | None = None
    if max_n / n_aligned >= SECTOR_CONCENTRATION_THRESHOLD:
        concentrated_sector = next(
            s["sector"] for s in sectors_out if s["n_oos"] == max_n
        )

    has_edge = any(s["verdict"] == "SIGNAL_EDGE" for s in sectors_out)
    # Verdict priority: inverted > concentrated > no-edge > healthy.
    # Inverted is the actionable red flag and outranks concentration —
    # an inverted sector inside a concentrated set is still inverted.
    if inverted:
        verdict = "HAS_INVERTED_SECTOR"
        hint = (f"{len(inverted)} sector(s) have anti-predictive rank skill "
                f"(rank_ic ≤ -{IC_GOOD}): {', '.join(sorted(inverted))}. The "
                f"scorer's prediction in this sector is WORSE than no "
                f"prediction — gating on it is actively harmful. This is the "
                f"data for a (separate, explicit) decision to exclude the "
                f"sector from the gate or retrain with rebalanced sectors; "
                f"do NOT change SECTOR_MAP from this read-only audit.")
    elif concentrated_sector is not None:
        verdict = "SECTOR_CONCENTRATED"
        max_pct = max_n / n_aligned * 100.0
        hint = (f"sector '{concentrated_sector}' carries {max_n}/{n_aligned} "
                f"({max_pct:.0f}%) of OOS rows; the scorer-ledger's headline "
                f"oos_ic is essentially this sector's IC. Read every "
                f"`oos_*` number in scorer_skill_log.jsonl with that caveat.")
    elif not has_edge:
        verdict = "NO_SECTOR_EDGE"
        hint = (f"no sector's predictions rank-predict realized 5d returns "
                f"(rank_ic ≥ {IC_GOOD}) on a stable sample — the scorer's "
                f"overall rank-IC, if positive, is coming from cross-sector "
                f"effects (e.g. an "
                f"unconditional regime bias), not per-name selection skill")
    else:
        verdict = "HEALTHY"
        hint = ("≥1 sector's predictions rank-predict realized 5d returns "
                "and none is anti-predictive")

    return {
        "status": "ok",
        "verdict": verdict,
        "n_train": sum(train_by_sector.values()),
        "n_oos": n_aligned,
        "concentrated_sector": concentrated_sector,
        "sectors": sectors_out,
        "inverted_sectors": sorted(inverted),
        "hint": hint,
    }


def _load_outcomes(path: Path) -> list[dict]:
    """Robust JSONL load — skips unparseable lines, never raises. Mirrors
    ``persona_skill._load_outcomes`` so a CLI consumer reads the file the
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
    compute per-sector OOS skill. The function the CLI / endpoint share.

    Never raises — on any fault returns ``{status='error', verdict=...,
    hint=...}``. ``outcomes_path`` defaults to the repo's canonical
    ``data/decision_outcomes.jsonl`` (test override).
    """
    if outcomes_path is None:
        outcomes_path = (Path(__file__).resolve().parent.parent.parent
                         / "data" / "decision_outcomes.jsonl")
    try:
        from paper_trader.validation import split_outcomes_temporal
    except Exception as exc:
        return {"status": "error", "verdict": "INSUFFICIENT_DATA",
                "n_train": 0, "n_oos": 0, "sectors": [],
                "inverted_sectors": [],
                "hint": f"validation module unavailable: {type(exc).__name__}: {exc}"}
    try:
        from paper_trader.ml.decision_scorer import DecisionScorer
    except Exception as exc:
        return {"status": "error", "verdict": "INSUFFICIENT_DATA",
                "n_train": 0, "n_oos": 0, "sectors": [],
                "inverted_sectors": [],
                "hint": f"decision_scorer unavailable: {type(exc).__name__}: {exc}"}

    recs = _load_outcomes(Path(outcomes_path))
    if not recs:
        return {"status": "insufficient_data", "verdict": "INSUFFICIENT_DATA",
                "n_train": 0, "n_oos": 0, "sectors": [],
                "inverted_sectors": [],
                "hint": f"no records loaded from {outcomes_path}"}
    train, oos = split_outcomes_temporal(recs, oos_fraction=oos_fraction)
    scorer = DecisionScorer()
    return sector_skill(scorer, train, oos)


def _cli() -> int:
    """``python3 -m paper_trader.ml.sector_skill`` — per-sector OOS skill
    over the live ``decision_outcomes.jsonl``. Read-only; never writes.

    Exit codes (mirror ``calibration._cli`` / ``persona_skill._cli``):
      * 0 — HEALTHY / NO_SECTOR_EDGE / SECTOR_CONCENTRATED / INSUFFICIENT_DATA
      * 1 — SCORER_UNTRAINED / other recoverable error
      * 2 — HAS_INVERTED_SECTOR (an operator/cron can branch on it)
    """
    rep = analyze()
    print(f"oos_outcomes={rep.get('n_oos', 0)}  "
          f"train_outcomes={rep.get('n_train', 0)}  "
          f"sectors={len(rep.get('sectors') or [])}")
    print(f"VERDICT: {rep['verdict']}  ({rep.get('hint','')})")
    if rep.get("sectors"):
        print(f"  {'sector':<14} {'n_tr':>6} {'n_oos':>6} {'rmse':>7} "
              f"{'dir_acc':>8} {'rank_ic':>9} {'meanP':>7} {'meanR':>7}  verdict")
        for e in rep["sectors"]:
            da = e["dir_acc"]
            da_s = f"{da*100:>7.1f}%" if da is not None else "    n/a"
            print(f"  {e['sector']:<14} {e['n_train']:>6} {e['n_oos']:>6} "
                  f"{e['rmse']:>7.2f} {da_s} {e['rank_ic']:>+9.3f} "
                  f"{e['mean_pred']:>+7.2f} {e['mean_realized']:>+7.2f}  "
                  f"{e['verdict']}")
    if rep["verdict"] == "HAS_INVERTED_SECTOR":
        return 2
    if rep.get("status") == "error":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
