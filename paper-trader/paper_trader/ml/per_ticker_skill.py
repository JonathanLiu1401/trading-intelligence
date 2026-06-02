"""Per-ticker OOS skill breakdown for the DecisionScorer.

``sector_skill`` answers "is the calibration uniform across the seven
sectors?" The next question a quant asks is one level finer: *"within
a single sector, which individual tickers does the scorer actually
predict well, and which is it actively wrong on?"* The 17-feature
``build_features`` includes a 7-way sector one-hot but NO per-ticker
identity — so two tickers in the same sector are forced to share
sector weights, even if their realized 5d behaviour diverges sharply
(e.g. NVDA vs INTC both map to ``sector_tech`` but trade very
differently). The live ``decision_outcomes.jsonl`` tail is also
ticker-concentrated — top 10 tickers carry ~70% of outcomes, with
SOXL / TQQQ / AMZN / MSTR / MSFT dominating — so the headline
``oos_ic`` from the scorer ledger is essentially a weighted average
of a handful of leveraged-ETF names. ``per_ticker_skill`` answers
per-ticker, on the SAME temporal holdout the scorer-skill ledger uses
(``validation.split_outcomes_temporal``):

| Field | Meaning |
|---|---|
| ``n_train`` | training rows for this ticker — proxies "did the scorer ever see this name?" |
| ``n_oos`` | OOS rows for this ticker |
| ``mean_pred`` / ``mean_realized`` | ticker-level mean predicted vs realized 5d return (%); a big gap is per-name magnitude bias |
| ``rmse`` | OOS RMSE for this ticker (compare to σ(target) for the ticker — a model worse than constant predictor adds noise) |
| ``dir_acc`` | OOS directional accuracy (a coin-flip = 0.5) |
| ``rank_ic`` | tie-aware Spearman(pred, realized) on OOS — same ``_spearman`` as ``calibration`` / ``sector_skill`` / ``persona_skill`` / ``_oos_rank_metrics`` (single source of truth, never drifts) |
| ``verdict`` | crisp threshold-driven per-ticker classification (below) |

Per-ticker verdict (each is exactly testable on a threshold):

| Verdict | Trigger |
|---|---|
| ``SPARSE`` | ``n_oos < MIN_OUTCOMES_PER_TICKER`` — Spearman is not stable |
| ``INVERTED_SIGNAL`` | ``rank_ic ≤ -IC_GOOD`` — the more confident the scorer is, the WORSE the realized 5d outcome on this name |
| ``SIGNAL_EDGE`` | ``rank_ic ≥ IC_GOOD`` — name carries real rank skill |
| ``WEAK_SIGNAL_EDGE`` | ``IC_MIN ≤ rank_ic < IC_GOOD`` |
| ``NO_SIGNAL_EDGE`` | otherwise |

Overall verdict: ``INSUFFICIENT_DATA`` (<``MIN_RECORDS`` aligned OOS rows),
``HAS_INVERTED_TICKER`` (≥1 ``INVERTED_SIGNAL`` — actionable red flag,
the scorer is actively HARMFUL on that name), ``NO_TICKER_EDGE`` (no
ticker reaches ``SIGNAL_EDGE``), or ``HEALTHY``.

Operational discipline matches the rest of ``paper_trader/ml`` (read-only,
never raises, never touches ``decision_scorer.pkl`` / ``build_features`` /
``N_FEATURES`` / trade path), so it is safe under the live unattended
continuous loop and cannot break pickle compatibility. Output is bounded
to ``MAX_TICKERS_IN_REPORT`` to keep the JSON small even when
``decision_outcomes.jsonl`` accumulates hundreds of unique names.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from paper_trader.ml.decision_scorer import _to_float
# Reuse the tie-aware rank correlation from calibration — so this OOS
# per-ticker IC and the in-sample calibration spearman (and sector_skill /
# persona_skill / _oos_rank_metrics) can never drift. Tie-awareness is
# load-bearing because PRED_CLAMP_PCT ties off-distribution predictions at
# exactly ±50; a non-tie-aware Spearman fabricates rank skill there.
from paper_trader.ml.calibration import _spearman

# Module-level thresholds so tests assert exact verdicts and a tuning
# change is a single reviewable edit (the codebase's
# constants-at-module-scope convention).
MIN_RECORDS = 30                       # min aligned OOS rows overall for any verdict
MIN_OUTCOMES_PER_TICKER = 20           # below this a ticker's Spearman is not stable
IC_MIN = 0.05                          # below |this| there is essentially no rank skill
IC_GOOD = 0.15                         # "real edge" bar (mirrors sector_skill.IC_GOOD)
MAX_TICKERS_IN_REPORT = 50             # cap report row count (live tail has ~80 unique)


def _aligned_oos_pair(record: dict, scorer) -> tuple[float, float] | None:
    """Return ``(pred, realized)`` for one OOS record, action-aligned, or
    ``None`` to drop.

    Universal SELL sign-flip applied to ``realized`` (mirrors
    ``evaluate_scorer_oos`` / ``_oos_rank_metrics`` / ``train_scorer`` /
    ``sector_skill._aligned_oos_pair`` — one consistent meaning of "good")
    so this per-ticker OOS metric describes the same prediction path the
    gate uses. A missing/non-finite ``forward_return_5d`` or a
    ``scorer.predict`` exception drops the row; a single poisoned outcomes
    line cannot corrupt the report.

    The 11-kwarg ``predict`` signature is the exact one every other OOS
    diagnostic uses — keep it identical so a future ``predict`` signature
    refactor must update one call shape, not several slightly different
    ones.
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
            ema200_above=record.get("ema200_above"),
            hist_cross_up=record.get("hist_cross_up"),
            macd_below_zero_cross=record.get("macd_below_zero_cross"),
        )
    except Exception:
        return None
    pred = _to_float(pred, float("nan"))
    if pred != pred:  # NaN ⇒ drop
        return None
    if str(record.get("action") or "BUY").upper() == "SELL":
        realized = -realized
    return float(pred), float(realized)


def _verdict_for_ticker(n: int, ic: float) -> str:
    if n < MIN_OUTCOMES_PER_TICKER:
        return "SPARSE"
    if ic <= -IC_GOOD:
        return "INVERTED_SIGNAL"
    if ic >= IC_GOOD:
        return "SIGNAL_EDGE"
    if ic >= IC_MIN:
        return "WEAK_SIGNAL_EDGE"
    return "NO_SIGNAL_EDGE"


def per_ticker_skill(
    scorer,
    train_records: list[dict],
    oos_records: list[dict],
) -> dict:
    """Aggregate per-ticker OOS skill against the provided scorer.

    Inputs mirror ``sector_skill.sector_skill`` exactly so the two read-only
    diagnostics can be invoked by the same caller with one argument shape:

      * ``scorer`` — anything implementing the 11-kwarg
        ``predict(ml_score, rsi, macd, mom5, mom20, regime_mult, ticker,
        vol_ratio, bb_pos, news_urgency, news_article_count)`` contract
        (the real ``DecisionScorer`` or a test double).
      * ``train_records`` / ``oos_records`` — the temporal split returned
        by ``validation.split_outcomes_temporal``. The training side is
        used ONLY to surface the per-ticker training-row count
        (``n_train``) so a quant can see how thinly the ticker was trained.

    Returns a JSON-safe dict. ``tickers`` is sorted by ``rank_ic``
    descending with ``SPARSE`` rows sinking to the bottom (their IC is not
    stable), capped at ``MAX_TICKERS_IN_REPORT``. Never raises; on a fault
    yields ``status='error'`` with a hint.
    """
    if not getattr(scorer, "is_trained", False):
        return {
            "status": "error",
            "verdict": "SCORER_UNTRAINED",
            "n_train": 0, "n_oos": 0, "tickers": [],
            "inverted_tickers": [],
            "hint": ("scorer.is_trained is False — per-ticker skill is "
                     "meaningless on an untrained model; accumulate ≥30 "
                     "deduped outcomes then retrain"),
        }

    # Per-ticker training counts — surface how thin training was for each
    # name. The scorer has no per-ticker identity feature (only a 7-way
    # sector one-hot), so a ticker with n_train=0 still gets predicted via
    # its sector + quant features — but a quant should see when a name in
    # the report has zero direct training exposure.
    train_by_ticker: dict[str, int] = {}
    for r in train_records or []:
        tk = str(r.get("ticker") or "").upper()
        if not tk:
            continue
        train_by_ticker[tk] = train_by_ticker.get(tk, 0) + 1

    # Build per-ticker OOS (pred, realized) lists.
    buckets: dict[str, dict] = {}
    n_aligned = 0
    for r in oos_records or []:
        tk = str(r.get("ticker") or "").upper()
        if not tk:
            continue
        pair = _aligned_oos_pair(r, scorer)
        if pair is None:
            continue
        pred, realized = pair
        b = buckets.setdefault(tk, {"pred": [], "real": []})
        b["pred"].append(pred)
        b["real"].append(realized)
        n_aligned += 1

    if n_aligned < MIN_RECORDS:
        return {
            "status": "insufficient_data",
            "verdict": "INSUFFICIENT_DATA",
            "n_train": sum(train_by_ticker.values()),
            "n_oos": n_aligned,
            "tickers": [],
            "inverted_tickers": [],
            "hint": (f"need ≥{MIN_RECORDS} aligned OOS outcomes with a finite "
                     f"forward_return_5d and a scorer prediction; "
                     f"have {n_aligned}"),
        }

    tickers_out: list[dict] = []
    inverted: list[str] = []
    for tk, b in buckets.items():
        n = len(b["pred"])
        pred = np.asarray(b["pred"], dtype=np.float64)
        real = np.asarray(b["real"], dtype=np.float64)
        rmse = float(np.sqrt(np.mean((pred - real) ** 2)))
        # Directional accuracy: only count pairs where BOTH sides have a
        # non-zero sign (a 0 has no direction). A ticker full of zeros
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
        verdict = _verdict_for_ticker(n, ic)
        if verdict == "INVERTED_SIGNAL":
            inverted.append(tk)
        tickers_out.append({
            "ticker": tk,
            "n_train": int(train_by_ticker.get(tk, 0)),
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
    # (unstable, small-n) IC. Tickers tied on rank_ic keep insertion order
    # (Python sort is stable), so the output is deterministic.
    tickers_out.sort(
        key=lambda d: (d["verdict"] != "SPARSE", d["rank_ic"]),
        reverse=True,
    )

    # Cap report size so a long-tail of one-off rare tickers can never
    # bloat the JSON payload. The top-MAX entries by the sort above are
    # the actionable ones; the tail is statistical noise. The cap retains
    # the full inverted_tickers list separately so an actionable red-flag
    # name far down the rank-IC sort is NEVER dropped from the report.
    tickers_capped = tickers_out[:MAX_TICKERS_IN_REPORT]

    has_edge = any(t["verdict"] == "SIGNAL_EDGE" for t in tickers_out)
    # Verdict priority: inverted > no-edge > healthy.
    # Inverted is the actionable red flag (the scorer is actively WORSE
    # than no scorer on that name), so it dominates the verdict.
    if inverted:
        verdict = "HAS_INVERTED_TICKER"
        hint = (f"{len(inverted)} ticker(s) have anti-predictive rank skill "
                f"(rank_ic ≤ -{IC_GOOD}): {', '.join(sorted(inverted)[:10])}"
                f"{'…' if len(inverted) > 10 else ''}. The scorer's prediction "
                f"on these names is WORSE than no prediction — gating on "
                f"them is actively harmful. This is the data for a (separate, "
                f"explicit) decision to exclude these tickers from the gate "
                f"or retrain with rebalanced ticker exposure; do NOT change "
                f"build_features or SECTOR_MAP from this read-only audit.")
    elif not has_edge:
        verdict = "NO_TICKER_EDGE"
        hint = (f"no ticker's predictions rank-predict realized 5d returns "
                f"(rank_ic ≥ {IC_GOOD}) on a stable sample — the scorer's "
                f"overall rank-IC, if positive, is coming from cross-name "
                f"effects (e.g. an unconditional regime / sector bias), not "
                f"per-name selection skill")
    else:
        verdict = "HEALTHY"
        hint = ("≥1 ticker's predictions rank-predict realized 5d returns "
                "and none is anti-predictive")

    return {
        "status": "ok",
        "verdict": verdict,
        "n_train": sum(train_by_ticker.values()),
        "n_oos": n_aligned,
        "n_unique_tickers_oos": len(buckets),
        "tickers": tickers_capped,
        "tickers_truncated": len(tickers_out) > MAX_TICKERS_IN_REPORT,
        "inverted_tickers": sorted(inverted),
        "hint": hint,
    }


def _load_outcomes(path: Path) -> list[dict]:
    """Robust JSONL load — skips unparseable lines, never raises. Mirrors
    ``sector_skill._load_outcomes`` / ``persona_skill._load_outcomes`` so a
    CLI consumer reads the file the same way every diagnostic does.
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
    compute per-ticker OOS skill. The function the CLI / endpoint share.

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
                "n_train": 0, "n_oos": 0, "tickers": [],
                "inverted_tickers": [],
                "hint": f"validation module unavailable: {type(exc).__name__}: {exc}"}
    try:
        from paper_trader.ml.decision_scorer import DecisionScorer
    except Exception as exc:
        return {"status": "error", "verdict": "INSUFFICIENT_DATA",
                "n_train": 0, "n_oos": 0, "tickers": [],
                "inverted_tickers": [],
                "hint": f"decision_scorer unavailable: {type(exc).__name__}: {exc}"}

    recs = _load_outcomes(Path(outcomes_path))
    if not recs:
        return {"status": "insufficient_data", "verdict": "INSUFFICIENT_DATA",
                "n_train": 0, "n_oos": 0, "tickers": [],
                "inverted_tickers": [],
                "hint": f"no records loaded from {outcomes_path}"}
    train, oos = split_outcomes_temporal(recs, oos_fraction=oos_fraction)
    scorer = DecisionScorer()
    return per_ticker_skill(scorer, train, oos)


def _cli() -> int:
    """``python3 -m paper_trader.ml.per_ticker_skill`` — per-ticker OOS
    skill over the live ``decision_outcomes.jsonl``. Read-only; never
    writes.

    Exit codes (mirror ``sector_skill._cli``):
      * 0 — HEALTHY / NO_TICKER_EDGE / INSUFFICIENT_DATA
      * 1 — SCORER_UNTRAINED / other recoverable error
      * 2 — HAS_INVERTED_TICKER (an operator/cron can branch on it)
    """
    rep = analyze()
    print(f"oos_outcomes={rep.get('n_oos', 0)}  "
          f"train_outcomes={rep.get('n_train', 0)}  "
          f"unique_tickers_oos={rep.get('n_unique_tickers_oos', 0)}  "
          f"rows_in_report={len(rep.get('tickers') or [])}"
          f"{' (truncated)' if rep.get('tickers_truncated') else ''}")
    print(f"VERDICT: {rep['verdict']}  ({rep.get('hint','')})")
    if rep.get("tickers"):
        print(f"  {'ticker':<10} {'n_tr':>6} {'n_oos':>6} {'rmse':>7} "
              f"{'dir_acc':>8} {'rank_ic':>9} {'meanP':>7} {'meanR':>7}  verdict")
        for e in rep["tickers"]:
            da = e["dir_acc"]
            da_s = f"{da*100:>7.1f}%" if da is not None else "    n/a"
            print(f"  {e['ticker']:<10} {e['n_train']:>6} {e['n_oos']:>6} "
                  f"{e['rmse']:>7.2f} {da_s} {e['rank_ic']:>+9.3f} "
                  f"{e['mean_pred']:>+7.2f} {e['mean_realized']:>+7.2f}  "
                  f"{e['verdict']}")
    if rep["verdict"] == "HAS_INVERTED_TICKER":
        return 2
    if rep.get("status") == "error":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
