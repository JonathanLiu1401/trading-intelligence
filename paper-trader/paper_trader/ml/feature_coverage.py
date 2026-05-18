"""DecisionScorer feature-coverage diagnostic — read-only.

The natural quant question UPSTREAM of every existing skill arbiter
(`calibration` / `skill_trend` / `baseline_compare` / `feature_importance` /
`regime_audit`): those all measure whether the *model* extracts skill from its
inputs. None of them ask whether the inputs **carry any variation in the
training data at all**. A feature that is at its `build_features` default in
≥90% of rows is dead weight — the "17-feature MLP" is then effectively a much
smaller model, and `feature_importance` *structurally cannot* surface it: you
cannot permute a near-constant column into measurable importance, so a dead
feature reads as 0.0 importance exactly like a feature the model merely
ignores. Coverage disentangles "the model ignores a real signal" from "the
signal was never in the data" — a data-pipeline question, not a model one.

This is decisive on the live corpus. `_compute_decision_outcomes` only
populates `news_urgency` / `news_article_count` when a backtest day actually
had scored news; historical windows almost never do, and backtest articles
structurally carry `urgency=0` (CLAUDE.md invariant #2). So the deployed
scorer trains with two of its ten numeric features pinned at their
`build_features` neutral defaults for the overwhelming majority of rows —
wasted capacity / noise dimensions that corroborate the documented
`MLP_WORSE_THAN_TRIVIAL` verdict (the raw `ml_score` one-liner out-ranks the
net partly because the net spends parameters fitting dead inputs + a sparse
7-way sector one-hot).

Faithfulness: each record is pushed through the **exact** `build_features`
call shape `train_scorer` uses (same kwargs, same clamps), and the per-slot
"default" vector is derived by calling `build_features` with every numeric
source `None` — single source of truth, so a default change in
`decision_scorer.py` can never silently drift this diagnostic (the
`_oos_rank_metrics`-reuses-`_spearman` / `baseline_trend`-imports-
`baseline_compare` precedent).

Same operational discipline as `paper_trader/ml/calibration.py`: read-only,
no train, no pickle touch, no trade path — safe to run against the live
unattended loop. Never raises on bad input.

```bash
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.feature_coverage
```
"""
from __future__ import annotations

import json
from pathlib import Path

# Need this many records before a coverage verdict is meaningful (mirrors
# calibration.MIN_PAIRS intent — below this the default-fraction is noise).
MIN_ROWS = 30
# A numeric feature whose value equals its build_features default in ≥ this
# fraction of rows carries essentially no information — the model is fed a
# (near-)constant noise dimension. 0.90 ⇒ ≤10% of rows informative.
DEAD_FLOOR = 0.90
# A feature default-substituted in ≥ this (but < DEAD_FLOOR) fraction is
# degraded — present but sparse enough that the model learns it poorly.
DEGRADED_FLOOR = 0.50

# (display name, decision_outcomes.jsonl record key) for the 10 numeric
# DecisionScorer features, in build_features slot order 0..9. The 7-way
# sector one-hot (slots 10..16) is intentionally excluded: it is sparse *by
# construction* (exactly one hot per row) so a "default fraction" is
# meaningless there — feature_importance permutes it jointly for that reason.
_NUMERIC_FEATURES: list[tuple[str, str]] = [
    ("ml_score", "ml_score"),
    ("rsi", "rsi"),
    ("macd", "macd"),
    ("mom5", "mom5"),
    ("mom20", "mom20"),
    ("regime_mult", "regime_mult"),
    ("vol_ratio", "vol_ratio"),
    ("bb_pos", "bb_position"),
    ("news_urgency", "news_urgency"),
    ("news_article_count", "news_article_count"),
]


def _default_vector() -> list[float]:
    """The first 10 (numeric) build_features slots when every numeric source
    is None — the exact value the model sees for a missing/None field.

    Derived from build_features itself (single source of truth) so a default
    change in decision_scorer.py cannot silently desync this diagnostic."""
    from paper_trader.ml.decision_scorer import build_features
    return list(build_features(
        None, None, None, None, None, None, "__feature_coverage_none__"
    ))[:10]


def _feature_vector(rec: dict) -> list[float]:
    """Push one outcome record through the EXACT build_features call shape
    train_scorer uses (same kwargs, same clamps); return the 10 numeric slots."""
    from paper_trader.ml.decision_scorer import build_features
    return list(build_features(
        rec.get("ml_score"),
        rec.get("rsi"),
        rec.get("macd"),
        rec.get("mom5"),
        rec.get("mom20"),
        rec.get("regime_mult"),
        str(rec.get("ticker") or ""),
        vol_ratio=rec.get("vol_ratio"),
        bb_pos=rec.get("bb_position"),
        news_urgency=rec.get("news_urgency"),
        news_article_count=rec.get("news_article_count"),
    ))[:10]


def load_outcomes(path: Path | str) -> list[dict]:
    """Robust JSONL load of decision_outcomes.jsonl. Skips unparseable lines.

    Never raises — a missing/corrupt file yields ``[]`` so callers degrade to
    ``INSUFFICIENT_DATA`` rather than crashing (the producer is best-effort by
    construction; a reader of it must be too — the `skill_trend` precedent)."""
    p = Path(path)
    rows: list[dict] = []
    try:
        if not p.exists():
            return rows
        for ln in p.read_text().splitlines():
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


def feature_coverage_report(records: list[dict]) -> dict:
    """Per-feature default-substitution fractions + an exact verdict.

    Verdicts (exact-value test-locked in tests/test_feature_coverage.py):
      * ``INSUFFICIENT_DATA``     — < MIN_ROWS records
      * ``DEAD_FEATURES_PRESENT`` — ≥1 numeric feature default-substituted in
                                    ≥ DEAD_FLOOR of rows OR constant (< 2
                                    distinct values across all rows)
      * ``DEGRADED_COVERAGE``     — no dead feature but ≥1 with default-fraction
                                    ≥ DEGRADED_FLOOR
      * ``FULL_COVERAGE``         — every numeric feature varies and is
                                    default-substituted in < DEGRADED_FLOOR

    Per-feature stats: ``default_fraction`` (post-build_features, so clamps and
    None→default substitution are applied exactly as the model sees them),
    ``distinct`` (rounded to 6 dp), ``dead`` flag.
    """
    out: dict = {
        "verdict": "INSUFFICIENT_DATA",
        "n_rows": len(records),
        "n_numeric_features": len(_NUMERIC_FEATURES),
        "effective_feature_count": None,
        "dead_features": [],
        "degraded_features": [],
        "features": {},
        "hint": "",
    }
    n = len(records)
    if n < MIN_ROWS:
        out["hint"] = f"need ≥{MIN_ROWS} records; have {n}"
        return out

    try:
        defaults = _default_vector()
    except Exception as exc:  # pragma: no cover - defensive
        out["hint"] = f"build_features unavailable: {type(exc).__name__}"
        return out

    # Tally per-slot: count rows at the default value, and collect distinct
    # rounded values. Round to 6 dp so float noise doesn't fabricate distinct
    # values or miss an exact-default match.
    at_default = [0] * len(_NUMERIC_FEATURES)
    distinct: list[set] = [set() for _ in _NUMERIC_FEATURES]
    dflt_r = [round(float(d), 6) for d in defaults]
    counted = 0
    for rec in records:
        try:
            vec = _feature_vector(rec)
        except Exception:
            # A single pathological record must not skew or break the audit;
            # skip it (it would also have crashed train_scorer's build loop,
            # caught upstream — coverage is a best-effort read).
            continue
        counted += 1
        for i in range(len(_NUMERIC_FEATURES)):
            v = round(float(vec[i]), 6)
            if v == dflt_r[i]:
                at_default[i] += 1
            if len(distinct[i]) < 64:
                distinct[i].add(v)

    if counted < MIN_ROWS:
        out["hint"] = (f"only {counted} of {n} records were feature-buildable; "
                       f"need ≥{MIN_ROWS}")
        return out

    out["n_rows"] = counted
    dead: list[str] = []
    degraded: list[str] = []
    effective = 0
    for i, (name, key) in enumerate(_NUMERIC_FEATURES):
        frac = at_default[i] / counted
        ndist = len(distinct[i])
        is_dead = frac >= DEAD_FLOOR or ndist < 2
        is_degraded = (not is_dead) and frac >= DEGRADED_FLOOR
        out["features"][name] = {
            "record_key": key,
            "default_value": dflt_r[i],
            "default_fraction": round(frac, 4),
            "distinct": ndist,
            "dead": is_dead,
        }
        if is_dead:
            dead.append(name)
        elif is_degraded:
            degraded.append(name)
        else:
            effective += 1
    out["dead_features"] = dead
    out["degraded_features"] = degraded
    out["effective_feature_count"] = effective

    if dead:
        out["verdict"] = "DEAD_FEATURES_PRESENT"
        out["hint"] = (
            f"{len(dead)}/{len(_NUMERIC_FEATURES)} numeric features dead "
            f"({', '.join(dead)}) — the gate's MLP is fed (near-)constant "
            f"noise dimensions; effective numeric dim ≈ {effective}"
        )
    elif degraded:
        out["verdict"] = "DEGRADED_COVERAGE"
        out["hint"] = (
            f"no dead feature but {len(degraded)} degraded "
            f"({', '.join(degraded)}) — default-substituted in "
            f"≥{DEGRADED_FLOOR:.0%} of rows"
        )
    else:
        out["verdict"] = "FULL_COVERAGE"
        out["hint"] = (
            f"all {len(_NUMERIC_FEATURES)} numeric features vary and are "
            f"default-substituted in <{DEGRADED_FLOOR:.0%} of rows"
        )
    return out


def analyze(outcomes_path: Path | str) -> dict:
    """Load decision_outcomes.jsonl + return the full coverage report."""
    return feature_coverage_report(load_outcomes(outcomes_path))


def _cli() -> int:
    """`python3 -m paper_trader.ml.feature_coverage` — read-only feature
    coverage of the live outcomes corpus.

    Exit mirrors the sibling diagnostics so a cron can branch on "the gate's
    MLP is being fed dead inputs right now": 0 on FULL_COVERAGE /
    INSUFFICIENT_DATA, 2 on DEAD_FEATURES_PRESENT / DEGRADED_COVERAGE."""
    root = Path(__file__).resolve().parent.parent.parent
    rep = analyze(root / "data" / "decision_outcomes.jsonl")
    print(f"VERDICT: {rep['verdict']}  ({rep['hint']})")
    print(f"  rows={rep['n_rows']}  "
          f"effective_numeric_dim={rep['effective_feature_count']}"
          f"/{rep['n_numeric_features']}")
    for name, st in rep.get("features", {}).items():
        flag = " DEAD" if st["dead"] else ""
        print(f"  {name:20s} default={st['default_value']:<8} "
              f"default_frac={st['default_fraction']:.3f} "
              f"distinct={st['distinct']}{flag}")
    return 0 if rep["verdict"] in ("FULL_COVERAGE", "INSUFFICIENT_DATA") else 2


if __name__ == "__main__":
    raise SystemExit(_cli())
