"""Conviction-gate economic counterfactual — what does the multiplier
overlay actually add (or subtract) in realized-return terms?

This is a **read-only diagnostic**. It never trains, never touches
`decision_scorer.pkl`, `decision_outcomes.jsonl`, `build_features`,
`N_FEATURES`, or any trade path — so it cannot perturb the unattended
continuous loop or break pickle compatibility (same operational
discipline as `paper_trader/ml/gate_audit.py` / `calibration.py` /
`skill_trend.py`).

**Why this is not `gate_audit.py`.** `gate_audit` answers "do the two
*extreme* arms separate?" — it reports each arm's mean realized return
and a verdict driven *solely* by `strong_tailwind_mean − strong_headwind_mean`.
That deliberately ignores (a) the three middle arms (`mild_headwind` ×0.85,
`neutral` ×1.00, `mild_tailwind` ×1.15), and (b) how *often* each arm fires.
A gate can read `GATE_EFFECTIVE` on the two-extreme spread while the
portfolio-level effect is negative — e.g. if 90% of decisions land in
`mild_headwind` and those are the winners, shrinking them ×0.85 net
subtracts return even though the rare extreme arms separate cleanly.

This module answers the question a quant deciding *whether to keep the
gate* actually has: **aggregate across all five arms, weighted by how
often each fires, does the multiplier overlay's reallocation add or
subtract realized return versus not gating at all?** It is the single
economic number — "the gate added/subtracted X pp of realized return on
the trades it sized" — that `gate_audit`'s per-arm view never rolls up.

**Headline metric is assumption-free.** The gate does not change *which*
trades execute (it only resizes a ticker `_ml_decide` already picked; the
off-distribution path abstains entirely, and any trade the gate shrank
below the qty floor records as HOLD and so never reaches
`decision_outcomes.jsonl` — see "Scope" below). So with every base bet
held at equal size, the gate-on portfolio's realized mean is the
multiplier-weighted mean `Σ mᵢrᵢ / Σ mᵢ` and gate-off is the plain mean
`mean(rᵢ)`. Their difference needs **no** conviction reconstruction and
is what drives the verdict.

**Sized metric is secondary and clearly caveated.** The gate's effect
compounds with `_ml_decide`'s *variable* base conviction
`min(cap, ml_score/divisor)`. `_reconstruct_base_conviction` mirrors
`_ml_decide` exactly, but `ml_score` in the outcome row is the reasoning
string's 2-dp-rounded `best_score` and the leveraged-ETF cap/divisor turns
on a regime that `regime_mult==1.0` cannot distinguish (bull vs the rare
"unknown"). So the base-conviction-weighted contribution is reported as
an informational secondary number — **never folded into the verdict**
(the `gate_audit` arm-monotone honesty pattern).

**Scope (a documented limitation, like every sibling tool).** This
measures the gate's *resizing* effect on the trades that executed. It
cannot see a trade the ×0.6 arm shrank below `_ml_decide`'s
`qty < 0.01 → HOLD` floor — that records as HOLD and never enters
`decision_outcomes.jsonl` (BUY/SELL FILLED only). In the deployed
regime base conviction ≥ ~5% so the floor effectively never binds, but
the number is honestly "gate contribution *on the sized fills*", not
"on every decision the gate touched".

`gate_arm` is imported from `gate_audit` (single source of truth — the
five arms must never drift between the two gate diagnostics, the
codebase's invariant-#10 spirit) and the temporal-OOS slice from
`validation.split_outcomes_temporal` (the exact split `skill_trend` /
`gate_audit` / `_train_decision_scorer`'s `oos_rmse` use, so this
describes the same holdout).

```bash
cd /home/zeph/trading-intelligence/paper-trader && python3 -m paper_trader.ml.gate_pnl
cd /home/zeph/trading-intelligence/paper-trader && python3 -m paper_trader.ml.gate_pnl --all
```
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Single source of truth for the five gate arms / multipliers — importing
# (not re-declaring) guarantees this diagnostic and gate_audit can never
# disagree about what the live `_ml_decide` gate does.
from .gate_audit import gate_arm

# Verdict thresholds — module-level so tests assert exact verdicts and a
# tuning change is a single reviewable edit (mirrors gate_audit.py /
# calibration.py).
MIN_TOTAL = 30      # need a real sample before any verdict (== gate_audit.MIN_TOTAL)
EDGE_TOL_PP = 1.0   # |gate contribution| band that reads as noise (== gate_audit.EDGE_TOL_PP)


def _reconstruct_base_conviction(
    ml_score, regime_mult, ticker: str
) -> float | None:
    """Best-effort reconstruction of `_ml_decide`'s PRE-gate base conviction.

    `_ml_decide` computes, for the chosen buy ticker::

        if ticker in _LEVERAGED_ETFS and regime in ("bull", "sideways"):
            conviction = min(0.40, best_score / 15.0)
        else:
            conviction = min(0.25, best_score / 20.0)

    `best_score` here is the outcome row's `ml_score` (the reasoning
    string's 2-dp `score=`). The regime is mapped back from `regime_mult`
    (`_ml_decide`/`_compute_decision_outcomes` use bull/unknown→1.0,
    sideways→0.6, bear→0.3). The bull-vs-"unknown" ambiguity at
    `regime_mult==1.0` is irreducible from the outcome row — both take the
    leveraged branch only when regime ∈ {bull, sideways}, and "unknown"
    (a sub-200-SPY-day early-run state) is rare, so treating `1.0` as
    bull is the closest faithful approximation. The `min(…, 0.95)` gate
    cap is inert here (base ≤ 0.40, ×1.3 = 0.52 < 0.95) and so omitted.

    Returns the base conviction, or None if `ml_score` is unusable (so the
    sized metric simply skips that row rather than fabricating a size).
    """
    try:
        bs = float(ml_score)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(bs):
        return None
    try:
        rm = float(regime_mult)
    except (TypeError, ValueError):
        rm = 1.0
    # regime_mult → regime: 0.3→bear, 0.6→sideways, else (1.0)→bull/unknown.
    if abs(rm - 0.3) < 1e-6:
        regime = "bear"
    elif abs(rm - 0.6) < 1e-6:
        regime = "sideways"
    else:
        regime = "bull"
    try:
        from paper_trader.backtest import _LEVERAGED_ETFS
    except Exception:
        _LEVERAGED_ETFS = set()
    if str(ticker) in _LEVERAGED_ETFS and regime in ("bull", "sideways"):
        return min(0.40, bs / 15.0)
    return min(0.25, bs / 20.0)


def gate_pnl_report(triples) -> dict:
    """Aggregate the gate's economic effect over ``(pred, realized, base)``
    triples.

    ``triples`` — any iterable of ``(predicted, realized_action_aligned,
    base_conviction_or_None)``. ``realized`` MUST already carry the SELL
    sign-flip the caller applies (``scorer_gate_pnl`` does, exactly like
    ``train_scorer`` / ``gate_audit.scorer_gate_audit``). Non-finite
    ``predicted``/``realized`` rows are dropped (one nan must not poison the
    aggregate — the ``_to_float`` hardening class). ``base`` may be None per
    row; the sized metric is computed only over rows that supply a finite
    positive base.

    Two counterfactuals:

    - ``equal_weight_gate_contribution_pp`` (**verdict-driving, assumption
      -free**): ``Σ mᵢrᵢ / Σ mᵢ  −  mean(rᵢ)``. Every base bet held equal,
      so this isolates the multiplier overlay's pure reallocation effect.
    - ``sized_gate_contribution_pp`` (**informational only**, NaN-safe
      None when no usable base): ``Σ wᵢmᵢrᵢ / Σ wᵢmᵢ  −  Σ wᵢrᵢ / Σ wᵢ``,
      with ``wᵢ`` the reconstructed base conviction — the realistic
      compounding, but reconstruction-approximate, so NOT in the verdict.

    | Verdict | Meaning |
    |---------|---------|
    | ``INSUFFICIENT_DATA`` | < ``MIN_TOTAL`` usable pairs |
    | ``GATE_SUBTRACTS_RETURN`` | equal-weight contribution < −``EDGE_TOL_PP`` — the overlay net sizes toward the losers; not gating would have realized more |
    | ``GATE_RETURN_NEUTRAL`` | \\|contribution\\| ≤ ``EDGE_TOL_PP`` — the overlay reallocates with no net realized effect (pure added variance) |
    | ``GATE_ADDS_RETURN`` | contribution > +``EDGE_TOL_PP`` — the overlay net sizes toward the winners; the multipliers earn their reallocation |

    Returns a JSON-safe dict. Never raises.
    """
    clean: list[tuple[float, float, float, float | None]] = []
    for row in triples:
        try:
            p, y, b = row
        except Exception:
            continue
        try:
            pf = float(p)
            yf = float(y)
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(pf) and np.isfinite(yf)):
            continue
        _, mult = gate_arm(pf)
        bf: float | None = None
        if b is not None:
            try:
                bb = float(b)
                if np.isfinite(bb) and bb > 0:
                    bf = bb
            except (TypeError, ValueError):
                bf = None
        clean.append((pf, yf, mult, bf))

    n = len(clean)
    base = {
        "status": "ok",
        "verdict": "INSUFFICIENT_DATA",
        "n": n,
        "gate_off_mean_pct": None,
        "gate_on_mean_pct": None,
        "equal_weight_gate_contribution_pp": None,
        "sized_gate_contribution_pp": None,
        "sized_n": 0,
        "avg_gate_multiplier": None,
        "hint": "",
    }
    if n < MIN_TOTAL:
        base["hint"] = f"need ≥{MIN_TOTAL} usable pairs; have n={n}"
        return base

    r = np.array([c[1] for c in clean], dtype=np.float64)
    m = np.array([c[2] for c in clean], dtype=np.float64)

    gate_off_mean = float(r.mean())                       # equal base size
    gate_on_mean = float(np.sum(m * r) / np.sum(m))       # multiplier-weighted
    eq_contrib = gate_on_mean - gate_off_mean

    base["gate_off_mean_pct"] = round(gate_off_mean, 4)
    base["gate_on_mean_pct"] = round(gate_on_mean, 4)
    base["equal_weight_gate_contribution_pp"] = round(eq_contrib, 4)
    base["avg_gate_multiplier"] = round(float(m.mean()), 4)

    # Sized (informational): only over rows with a usable reconstructed base.
    sized = [(c[1], c[2], c[3]) for c in clean if c[3] is not None]
    base["sized_n"] = len(sized)
    if sized:
        sr = np.array([s[0] for s in sized], dtype=np.float64)
        sm = np.array([s[1] for s in sized], dtype=np.float64)
        sw = np.array([s[2] for s in sized], dtype=np.float64)
        off_sized = float(np.sum(sw * sr) / np.sum(sw))
        on_w = sw * sm
        on_sized = float(np.sum(on_w * sr) / np.sum(on_w))
        base["sized_gate_contribution_pp"] = round(on_sized - off_sized, 4)

    if eq_contrib < -EDGE_TOL_PP:
        base["verdict"] = "GATE_SUBTRACTS_RETURN"
        base["hint"] = (
            f"gate-on realized {gate_on_mean:+.2f}% < gate-off "
            f"{gate_off_mean:+.2f}% (contribution {eq_contrib:+.2f}pp) — the "
            f"multiplier overlay net sizes toward the losers; not gating "
            f"would have realized more on these fills"
        )
    elif abs(eq_contrib) <= EDGE_TOL_PP:
        base["verdict"] = "GATE_RETURN_NEUTRAL"
        base["hint"] = (
            f"gate-on {gate_on_mean:+.2f}% vs gate-off {gate_off_mean:+.2f}% "
            f"(contribution {eq_contrib:+.2f}pp, within ±{EDGE_TOL_PP:.1f}pp) "
            f"— the overlay reallocates capital with no net realized effect: "
            f"pure added sizing variance"
        )
    else:
        base["verdict"] = "GATE_ADDS_RETURN"
        base["hint"] = (
            f"gate-on realized {gate_on_mean:+.2f}% > gate-off "
            f"{gate_off_mean:+.2f}% (contribution {eq_contrib:+.2f}pp) — the "
            f"multiplier overlay net sizes toward the winners; the gate's "
            f"reallocation earns its keep on these fills"
        )
    return base


def scorer_gate_pnl(scorer, records, oos_only: bool = True) -> dict:
    """Run ``gate_pnl_report`` for a DecisionScorer over outcome records
    (the ``data/decision_outcomes.jsonl`` row shape).

    SELL realized goodness is ``-forward_return_5d`` exactly like
    ``train_scorer`` / ``gate_audit.scorer_gate_audit`` (a drop after a SELL
    was the *right* call). The 11-kwarg ``predict`` signature mirrors
    ``gate_audit`` / ``validation.evaluate_scorer_oos`` so this audit
    describes the SAME prediction path the live gate uses.

    ``oos_only`` (default True) restricts to the temporal-OOS slice via
    ``validation.split_outcomes_temporal`` — the trustworthy,
    generalization-relevant view (in-sample is optimistic; AGENTS.md:
    "always read it next to oos_rmse"). The chosen slice is reported in
    ``slice`` (``"oos"`` / ``"all"``).

    Never raises — any fault degrades to ``INSUFFICIENT_DATA``.
    """
    try:
        recs = list(records or [])
    except Exception:
        recs = []

    slice_name = "all"
    if oos_only:
        try:
            from paper_trader.validation import split_outcomes_temporal
            _, oos = split_outcomes_temporal(recs, oos_fraction=0.2)
            if oos:
                recs = oos
                slice_name = "oos"
        except Exception:
            slice_name = "all"

    triples: list[tuple[float, float, float | None]] = []
    for rrow in recs:
        try:
            pred = scorer.predict(
                ml_score=rrow.get("ml_score", 0.0),
                rsi=rrow.get("rsi"),
                macd=rrow.get("macd"),
                mom5=rrow.get("mom5"),
                mom20=rrow.get("mom20"),
                regime_mult=rrow.get("regime_mult", 1.0),
                ticker=rrow.get("ticker", ""),
                vol_ratio=rrow.get("vol_ratio"),
                bb_pos=rrow.get("bb_position"),
                news_urgency=rrow.get("news_urgency"),
                news_article_count=rrow.get("news_article_count"),
            )
        except Exception:
            continue
        y = rrow.get("forward_return_5d")
        if y is None:
            continue
        action = str(rrow.get("action") or "BUY").upper()
        realized = -y if action == "SELL" else y
        base_conv = _reconstruct_base_conviction(
            rrow.get("ml_score"), rrow.get("regime_mult", 1.0),
            str(rrow.get("ticker") or ""),
        )
        triples.append((pred, realized, base_conv))

    rep = gate_pnl_report(triples)
    rep["slice"] = slice_name
    rep["n_records_considered"] = len(recs)
    return rep


def analyze(outcomes_path: Path | str, oos_only: bool = True) -> dict:
    """Load the live pickled scorer + outcomes file and return the report.
    Read-only; never raises."""
    from .decision_scorer import DecisionScorer

    out: dict = {"status": "error", "verdict": "INSUFFICIENT_DATA",
                 "n": 0, "hint": ""}
    try:
        p = Path(outcomes_path)
        if not p.exists():
            out["hint"] = f"no outcomes file at {p}"
            return out
        records: list[dict] = []
        for ln in p.read_text().splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                records.append(json.loads(ln))
            except Exception:
                continue
        scorer = DecisionScorer()
        if not getattr(scorer, "is_trained", False):
            out["hint"] = "scorer not trained — nothing to audit"
            return out
        rep = scorer_gate_pnl(scorer, records, oos_only=oos_only)
        rep.setdefault("n_train", getattr(scorer, "n_train", None))
        return rep
    except Exception as e:
        out["hint"] = f"audit failed: {type(e).__name__}: {e}"
        return out


def _cli() -> int:
    """`python3 -m paper_trader.ml.gate_pnl [--all]` — the gate's economic
    counterfactual on the live pickled scorer vs the accumulated outcomes
    tail. Read-only. Exits 2 on ``GATE_SUBTRACTS_RETURN`` (cron-branchable —
    the actionable "turn it off would have helped" signal)."""
    import sys
    oos_only = "--all" not in sys.argv
    root = Path(__file__).resolve().parent.parent.parent
    rep = analyze(root / "data" / "decision_outcomes.jsonl", oos_only=oos_only)
    print(f"VERDICT: {rep['verdict']}  ({rep.get('hint', '')})")
    print(f"  slice={rep.get('slice')}  n={rep.get('n')}  "
          f"n_train={rep.get('n_train')}")
    print(f"  gate_off_mean={rep.get('gate_off_mean_pct')}%  "
          f"gate_on_mean={rep.get('gate_on_mean_pct')}%  "
          f"avg_mult={rep.get('avg_gate_multiplier')}")
    print(f"  equal_weight_contribution="
          f"{rep.get('equal_weight_gate_contribution_pp')}pp  "
          f"sized_contribution={rep.get('sized_gate_contribution_pp')}pp "
          f"(sized_n={rep.get('sized_n')}, informational)")
    return 2 if rep.get("verdict") == "GATE_SUBTRACTS_RETURN" else 0


if __name__ == "__main__":
    raise SystemExit(_cli())
