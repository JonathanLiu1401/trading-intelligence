"""Outcome corpus health — pre-model data quality diagnostic.

Every existing ML/backtest diagnostic in this repo measures something the
model OUTPUTS (rank-IC, RMSE, dir-acc, gate-arm distribution, conviction
calibration, etc.). None directly answers the data-quality question that
PRECEDES every model verdict: *what is the training corpus actually
made of?*

A quant researcher reading `MLP_WORSE_THAN_TRIVIAL` or `gate_effectively_active=False`
needs to know whether the verdict reflects model weakness or data sparsity:

  * If `news_urgency` is present in only 1% of training rows (live state
    of the production corpus), then the model's `news_urgency` weight
    is structurally dead-trained — the feature isn't broken, the data
    pipeline is.
  * If `regime_label='bear'` accounts for 0.5% of rows, the
    bear-regime rank-IC reported in `scorer_skill_log` is sampling
    noise, not skill state. A reading quant looking at "bear IC = -0.20"
    needs to know it's n=36.
  * If 70%+ of the target distribution is BUY decisions, the SELL
    sub-skill is data-limited by construction, regardless of what
    the model achieves on the BUY side.

This analyzer surfaces those facts honestly per analysis. Verdicts:

  * `INSUFFICIENT_DATA`         — corpus < `MIN_RECORDS` rows
  * `NEWS_FEATURES_DARK`        — news_urgency/news_article_count
                                  populated in < `NEWS_DARK_THRESHOLD`
                                  of rows (the gate-feeding news
                                  features are effectively absent)
  * `ACTION_IMBALANCED`         — one of BUY/SELL is > `ACTION_IMBALANCE_THRESHOLD`
                                  of rows (the minority class is
                                  data-limited)
  * `REGIME_BUCKETS_SPARSE`     — fewer than `MIN_REGIME_ROWS` rows in
                                  one of bull/sideways/bear (regime-
                                  conditional metrics are sampling
                                  noise)
  * `HEALTHY`                   — none of the above

Pure read; never raises (a corpus-health analyzer must be safe to call
from any per-cycle ledger writer — the `_append_*_skill_log` discipline).
"""
from __future__ import annotations

import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

# Default outcomes path — module-level so tests can monkeypatch and the
# per-cycle ledger writer in `run_continuous_backtests.py` can override
# (same testability rule every sibling analyzer follows).
DEFAULT_OUTCOMES_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data" / "decision_outcomes.jsonl"
)

# Verdict thresholds. Values match the documented operating envelope:
#  * MIN_RECORDS=100        — below this the corpus is too small to draw
#                             any conclusion; INSUFFICIENT_DATA
#  * NEWS_DARK_THRESHOLD    — fraction of rows with non-null
#                             news_urgency AND news_article_count below
#                             which the news features are effectively
#                             dead-trained. 0.10 = 10% — well above the
#                             current live 1% so the verdict fires
#                             durably until the news pipeline really
#                             wakes up.
#  * ACTION_IMBALANCE_THRESHOLD — fraction of one action class above
#                             which the minority class is data-limited.
#                             0.85 = 85%; live state ≈ 72% BUY which
#                             reads HEALTHY at this threshold.
#  * MIN_REGIME_ROWS         — minimum rows in a regime bucket for the
#                             bucket's metrics to be statistically
#                             meaningful. 50 is the conventional
#                             rank-skill rule-of-thumb floor.
#  * TARGET_DEGENERATE_STD   — std(forward_return_5d) below this is
#                             pathological (the training set has no
#                             variance to learn from). 1.0pp is far
#                             below any realistic equity-tape std.
MIN_RECORDS = 100
NEWS_DARK_THRESHOLD = 0.10
ACTION_IMBALANCE_THRESHOLD = 0.85
MIN_REGIME_ROWS = 50
TARGET_DEGENERATE_STD = 1.0

# Features the analyzer reports per-key density for. Keys match the
# build_features inputs (numeric quant + news + the 3 enhanced MACD
# booleans) plus the captured outcome metadata (gate_scorer_pred,
# conviction_pct, etc.) a researcher reading the corpus actually cares
# about.
TRACKED_FEATURES: tuple[str, ...] = (
    "rsi", "macd", "mom5", "mom20", "regime_mult",
    "vol_ratio", "bb_position", "news_urgency", "news_article_count",
    "ema200_above", "hist_cross_up", "macd_below_zero_cross",
    "wk52_pos", "pct_from_52h",
    # Captured at decision time — useful for understanding gate coverage.
    "gate_scorer_pred", "conviction_pct",
    # Multi-horizon forward returns — the analyzer treats their absence
    # as "outcome computation skipped this horizon for this row"
    # (window ran past cached price history).
    "forward_return_10d", "forward_return_20d",
    "forward_intraperiod_min_5d", "forward_intraperiod_max_5d",
    # LLM annotation pipeline — currently dark per the
    # llm_annotation_skill ledger; surfacing per-row presence makes the
    # darkness visible in this view too.
    "llm_quality_label",
)

# Sectors recognized by the DecisionScorer. Kept local rather than
# imported so the analyzer is independent of any future scorer-side
# refactor (the SSOT-no-drift convention every sibling diagnostic
# follows).
_KNOWN_SECTORS = (
    "tech", "energy", "financials", "healthcare",
    "commodities", "crypto", "other",
)


def _load_outcomes(path: Path | str) -> list[dict]:
    """Read JSONL outcomes. Never raises — a missing file or any
    line-parse failure degrades gracefully, returning whatever rows
    were successfully parsed. Mirrors the
    `_append_*_skill_log` discipline."""
    p = Path(path)
    if not p.exists():
        return []
    out: list[dict] = []
    try:
        with p.open("r") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    out.append(json.loads(ln))
                except Exception:
                    continue
    except Exception:
        pass
    return out


def _density(rows: list[dict], key: str) -> tuple[int, float]:
    """Return (n_nonnull, fraction_nonnull) for one feature key.

    A bool feature is non-null when it is `True` or `False` (NOT None);
    a numeric feature is non-null when it is finite (rejects NaN/Inf
    silently — the same `_to_float` discipline `build_features` uses)."""
    n_nonnull = 0
    for r in rows:
        v = r.get(key)
        if v is None:
            continue
        # Bool is a subclass of int — treat True/False as non-null
        # regardless of finite-ness.
        if isinstance(v, bool):
            n_nonnull += 1
            continue
        if isinstance(v, (int, float)):
            try:
                if math.isfinite(float(v)):
                    n_nonnull += 1
            except (TypeError, ValueError):
                pass
            continue
        # Strings (e.g. unrelated keys) count as present too — captures
        # the "value exists but isn't numeric" case for a researcher's
        # visual inspection.
        n_nonnull += 1
    frac = n_nonnull / len(rows) if rows else 0.0
    return n_nonnull, round(frac, 4)


def _target_stats(rows: list[dict]) -> dict:
    """Compute distribution stats for `forward_return_5d` — the actual
    DecisionScorer training target. Drops non-finite values silently.

    Returns `{n, mean, std, min, max, p5, p95}` or all-None when the
    target sample is too small (n < 2 → stdev undefined).
    """
    vals: list[float] = []
    for r in rows:
        v = r.get("forward_return_5d")
        if isinstance(v, bool) or v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(fv):
            continue
        vals.append(fv)
    out: dict[str, Any] = {
        "n": len(vals), "mean": None, "std": None,
        "min": None, "max": None, "p5": None, "p95": None,
    }
    if not vals:
        return out
    vals.sort()
    out["min"] = round(vals[0], 4)
    out["max"] = round(vals[-1], 4)
    out["mean"] = round(statistics.mean(vals), 4)
    if len(vals) >= 2:
        out["std"] = round(statistics.stdev(vals), 4)
    # Manual percentile (avoid numpy dep — analyzer must be stdlib-only).
    n = len(vals)
    p5_idx = max(0, min(n - 1, int(round(0.05 * (n - 1)))))
    p95_idx = max(0, min(n - 1, int(round(0.95 * (n - 1)))))
    out["p5"] = round(vals[p5_idx], 4)
    out["p95"] = round(vals[p95_idx], 4)
    return out


def _sector_breakdown(rows: list[dict]) -> dict[str, int]:
    """Approximate per-sector breakdown using the SECTOR_MAP rules. This
    avoids importing DecisionScorer (independence rule). Tickers that
    don't match any known sector key land in `unknown`."""
    # Lazy import — keeps module import cost zero and lets a future
    # SECTOR_MAP refactor stay self-contained in decision_scorer.
    try:
        from .decision_scorer import SECTOR_MAP
    except Exception:
        SECTOR_MAP = {}
    counts: Counter = Counter()
    for r in rows:
        tk = str(r.get("ticker") or "").upper()
        sector = SECTOR_MAP.get(tk, "other") if tk else "unknown"
        counts[sector] += 1
    return dict(counts)


def analyze(outcomes_path: Path | str | None = None) -> dict:
    """Analyze the outcome corpus.

    Returns a JSON-safe dict ``{status, verdict, n_total, n_buys,
    n_sells, n_with_news, fraction_with_news, regime_counts,
    sector_counts, feature_density, target, date_range_start,
    date_range_end, persona_counts, hints}``.

    Verdicts (priority order — first match wins, so e.g. an empty
    corpus reads INSUFFICIENT_DATA rather than NEWS_FEATURES_DARK):

      INSUFFICIENT_DATA — corpus < MIN_RECORDS
      NEWS_FEATURES_DARK — news_urgency density < NEWS_DARK_THRESHOLD
      ACTION_IMBALANCED  — BUY or SELL > ACTION_IMBALANCE_THRESHOLD
      REGIME_BUCKETS_SPARSE — one of bull/sideways/bear has < MIN_REGIME_ROWS
      HEALTHY            — none of the above

    Never raises — a missing outcomes file degrades to
    `n_total=0 verdict=INSUFFICIENT_DATA`. The `hints` field carries
    a list of operator-readable strings explaining each negative
    verdict's specific trigger (e.g.
    `"news features in only 65/6581 (1.0%) — below 10% threshold"`).
    """
    if outcomes_path is None:
        outcomes_path = DEFAULT_OUTCOMES_PATH
    rows = _load_outcomes(outcomes_path)
    n_total = len(rows)

    if n_total < MIN_RECORDS:
        return {
            "status": "ok" if n_total > 0 else "empty",
            "verdict": "INSUFFICIENT_DATA",
            "n_total": n_total,
            "hints": [f"corpus has {n_total} rows, need ≥ {MIN_RECORDS} "
                      f"for any verdict beyond INSUFFICIENT_DATA"],
            "feature_density": {},
            "regime_counts": {},
            "sector_counts": {},
            "target": _target_stats(rows),
            "n_buys": 0,
            "n_sells": 0,
            "n_with_news": 0,
            "fraction_with_news": 0.0,
            "date_range_start": None,
            "date_range_end": None,
            "persona_counts": {},
        }

    # Action breakdown
    actions = Counter(str(r.get("action") or "").upper() for r in rows)
    n_buys = int(actions.get("BUY", 0))
    n_sells = int(actions.get("SELL", 0))
    n_other = sum(v for k, v in actions.items() if k not in ("BUY", "SELL"))

    # Regime breakdown — explicit regime_label (preferred); legacy
    # rows w/o the label decode regime_mult via the same map every
    # sibling analyzer uses (0.3=bear, 0.6=sideways, 1.0=bull).
    regime_counts: Counter = Counter()
    _REGIME_BY_MULT = {0.3: "bear", 0.6: "sideways", 1.0: "bull"}
    for r in rows:
        lbl = r.get("regime_label")
        if lbl in ("bull", "sideways", "bear", "unknown"):
            regime_counts[lbl] += 1
        else:
            try:
                rm = float(r.get("regime_mult"))
                regime_counts[_REGIME_BY_MULT.get(rm, "unknown")] += 1
            except (TypeError, ValueError):
                regime_counts["unknown"] += 1
    regime_counts_d = dict(regime_counts)

    # Sector breakdown (best-effort SECTOR_MAP lookup)
    sector_counts = _sector_breakdown(rows)

    # Persona breakdown — purely informational (no verdict gate on it).
    persona_counts = Counter(r.get("persona") or "unknown" for r in rows)
    persona_counts_d = {str(k): int(v) for k, v in persona_counts.items()}

    # Per-feature density
    feature_density: dict[str, dict] = {}
    for key in TRACKED_FEATURES:
        n_nn, frac = _density(rows, key)
        feature_density[key] = {"n_non_null": n_nn, "fraction": frac}

    # News-features density — joint AND (both keys must be present for
    # the row to carry news context the way build_features's neutral-
    # default convention expects).
    n_with_news = sum(
        1 for r in rows
        if r.get("news_urgency") is not None
        and r.get("news_article_count") is not None
    )
    fraction_with_news = (n_with_news / n_total) if n_total else 0.0

    # Date range — sim_date is an ISO YYYY-MM-DD string by contract.
    dates = sorted(
        d for d in (r.get("sim_date") for r in rows)
        if isinstance(d, str) and len(d) >= 10
    )
    date_range_start = dates[0] if dates else None
    date_range_end = dates[-1] if dates else None

    # Target distribution
    target = _target_stats(rows)

    # Verdict ladder
    hints: list[str] = []
    verdict = "HEALTHY"

    # First: news pipeline darkness
    if fraction_with_news < NEWS_DARK_THRESHOLD:
        if verdict == "HEALTHY":
            verdict = "NEWS_FEATURES_DARK"
        hints.append(
            f"news features in only {n_with_news}/{n_total} "
            f"({100.0 * fraction_with_news:.1f}%) — below "
            f"{100.0 * NEWS_DARK_THRESHOLD:.0f}% threshold"
        )

    # Second: action imbalance (independently surfaced as a hint even if
    # not the verdict — operators want every negative state in `hints`).
    if n_total > 0:
        max_action_frac = max(n_buys, n_sells) / n_total
        if max_action_frac > ACTION_IMBALANCE_THRESHOLD:
            if verdict == "HEALTHY":
                verdict = "ACTION_IMBALANCED"
            dominant = "BUY" if n_buys >= n_sells else "SELL"
            hints.append(
                f"{dominant} {max(n_buys, n_sells)}/{n_total} "
                f"({100.0 * max_action_frac:.1f}%) — exceeds "
                f"{100.0 * ACTION_IMBALANCE_THRESHOLD:.0f}% imbalance "
                f"threshold; minority-action skill is data-limited"
            )

    # Third: regime bucket sparsity (each bucket independently)
    sparse_regimes: list[str] = []
    for reg in ("bull", "sideways", "bear"):
        n_reg = int(regime_counts_d.get(reg, 0))
        if n_reg < MIN_REGIME_ROWS:
            sparse_regimes.append(f"{reg}={n_reg}")
    if sparse_regimes:
        if verdict == "HEALTHY":
            verdict = "REGIME_BUCKETS_SPARSE"
        hints.append(
            f"regime buckets below n={MIN_REGIME_ROWS}: "
            f"{', '.join(sparse_regimes)} — regime-conditional rank-IC "
            f"is sampling noise for these"
        )

    # Fourth: target std degenerate
    if target.get("std") is not None:
        try:
            if float(target["std"]) < TARGET_DEGENERATE_STD:
                if verdict == "HEALTHY":
                    verdict = "TARGET_DEGENERATE"
                hints.append(
                    f"target std={target['std']:.3f} below "
                    f"{TARGET_DEGENERATE_STD:.2f} — no variance to learn from"
                )
        except (TypeError, ValueError):
            pass

    if n_other > 0:
        hints.append(f"{n_other} rows with action ∉ {{BUY,SELL}} — "
                     f"likely a corrupted action field")

    return {
        "status": "ok",
        "verdict": verdict,
        "n_total": n_total,
        "n_buys": n_buys,
        "n_sells": n_sells,
        "n_other_action": n_other,
        "n_with_news": n_with_news,
        "fraction_with_news": round(fraction_with_news, 4),
        "regime_counts": regime_counts_d,
        "sector_counts": sector_counts,
        "persona_counts": persona_counts_d,
        "feature_density": feature_density,
        "target": target,
        "date_range_start": date_range_start,
        "date_range_end": date_range_end,
        "hints": hints,
    }


# ---------------------------------------------------------------------------
# CLI: `python3 -m paper_trader.ml.outcome_corpus_health [--json]`
# Read-only; loads the deployed outcomes JSONL and prints the analysis.
# ---------------------------------------------------------------------------

def _build_arg_parser():
    import argparse
    p = argparse.ArgumentParser(
        prog="python3 -m paper_trader.ml.outcome_corpus_health",
        description=(
            "Audit the training corpus (data/decision_outcomes.jsonl): "
            "per-feature non-null density, action/regime/sector breakdown, "
            "target distribution stats, and a verdict on whether the "
            "corpus is HEALTHY or carries one of the documented "
            "data-quality red flags (NEWS_FEATURES_DARK, ACTION_IMBALANCED, "
            "REGIME_BUCKETS_SPARSE, TARGET_DEGENERATE, INSUFFICIENT_DATA). "
            "Read-only — never writes."
        ),
    )
    p.add_argument("--path", default=None,
                   help="Outcomes JSONL path (default: data/decision_outcomes.jsonl).")
    p.add_argument("--json", action="store_true",
                   help="Emit machine-readable JSON instead of a table.")
    return p


def main(argv: list[str] | None = None) -> int:
    import sys
    args = _build_arg_parser().parse_args(
        sys.argv[1:] if argv is None else argv)
    rep = analyze(args.path)

    if args.json:
        print(json.dumps(rep, indent=2, sort_keys=True))
        return 0 if rep.get("verdict") == "HEALTHY" else 1

    print(f"[outcome_corpus_health] verdict={rep['verdict']} "
          f"n_total={rep['n_total']}")
    print(f"  date range: {rep.get('date_range_start')} → "
          f"{rep.get('date_range_end')}")
    print(f"  actions: BUY={rep['n_buys']} SELL={rep['n_sells']} "
          f"OTHER={rep.get('n_other_action', 0)}")
    print(f"  news context: {rep['n_with_news']}/{rep['n_total']} "
          f"({100.0 * rep['fraction_with_news']:.1f}%)")
    print(f"  regime: {rep.get('regime_counts')}")
    tgt = rep.get("target") or {}
    if tgt.get("std") is not None:
        print(f"  target (forward_return_5d): n={tgt['n']} "
              f"mean={tgt['mean']:+.2f}% std={tgt['std']:.2f} "
              f"min={tgt['min']:+.2f}% max={tgt['max']:+.2f}% "
              f"p5={tgt['p5']:+.2f}% p95={tgt['p95']:+.2f}%")
    sec = rep.get("sector_counts") or {}
    if sec:
        print("  sector counts:")
        for k in sorted(sec, key=lambda x: -sec[x]):
            print(f"    {k:<12}{sec[k]:>6}")
    fd = rep.get("feature_density") or {}
    if fd:
        print("  feature density (non-null / total):")
        for k in sorted(fd, key=lambda x: -fd[x]["fraction"]):
            v = fd[k]
            print(f"    {k:<32}{v['n_non_null']:>6} "
                  f"({100.0 * v['fraction']:.1f}%)")
    hints = rep.get("hints") or []
    if hints:
        print("  hints:")
        for h in hints:
            print(f"    - {h}")
    return 0 if rep["verdict"] == "HEALTHY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
