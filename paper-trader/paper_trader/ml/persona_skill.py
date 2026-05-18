"""Per-persona *decision-signal* skill diagnostic — does each trading
persona's own signal actually rank-predict realized outcomes, or is its
return pure leveraged-beta noise?

This is the **decision-level** sibling of
`paper_trader/ml/persona_leaderboard.py`. The leaderboard aggregates
*run-level* `vs_spy_pct` from `backtest.db` — but AGENTS.md is emphatic
that the per-run number is "the max of a high-variance leveraged-beta
draw … never read it as strategy skill" (a single persona routinely
posts +1000% on one window and −80% vs SPY on the *same* window). So a
positive leaderboard median can be pure leverage luck. The only honest
decision-level question is: **within a persona's own decisions, does a
stronger signal precede a better realized outcome?** That is a
rank-correlation the engine records every cycle (`decision_outcomes.jsonl`
carries `run_id`, `action`, `ml_score`, and the realized
`forward_return_5d`) but aggregates nowhere.

`score_ic` answers exactly that, per persona: tie-aware Spearman between
the **action-aligned** `ml_score` the `_ml_decide` engine acted on and the
**action-aligned** realized 5-trading-day forward return. "Action-aligned"
means the universal codebase SELL convention (`train_scorer`,
`ml.calibration`, `validation.evaluate_scorer_oos`, `_oos_rank_metrics`,
`ml.skill_trend` all do this): a SELL's realized goodness is
`-forward_return_5d` (a drop after a SELL was the *right* call), and —
applied symmetrically here — the SELL's `ml_score` sign is flipped too so
"higher signal ⇒ higher aligned outcome" has one consistent meaning across
BUY and SELL. `_spearman` is **imported from `ml.calibration`** (single
source of truth — the same tie-aware rank correlation, so this and the
in-sample calibration diagnostic can never drift; tie-awareness is
load-bearing because `ml_score` parsed from reasoning has heavy ties at the
per-persona buy threshold).

Operational discipline is identical to `ml/calibration.py`,
`ml/skill_trend.py`, and `ml/persona_leaderboard.py`: **read-only** — no
train, no `decision_scorer.pkl` / `build_features` / `N_FEATURES` / trade
path touch, never raises on bad input — so it is safe to run against the
live unattended continuous loop and cannot break pickle compatibility
(AGENTS.md "When to bump model versions" / "Common pitfalls"). It does
**not** prune `PERSONAS` or re-tune `_PERSONA_BOOSTS`; that is a
strategy-dynamics change requiring an explicit, separate decision. This
tool exists only to *inform* it.

Verdict per persona (crisp, threshold-driven so it is exactly testable):

| Verdict | Meaning |
|---------|---------|
| `INSUFFICIENT` | < ``MIN_OUTCOMES_PER_PERSONA`` aligned outcomes — no stable IC |
| `INVERTED_SIGNAL` | ``score_ic ≤ -IC_GOOD`` — signal is anti-predictive; the more confident this persona is, the *worse* it does (actively harmful) |
| `SIGNAL_EDGE` | ``score_ic ≥ IC_GOOD`` — signal genuinely rank-predicts realized goodness |
| `WEAK_SIGNAL_EDGE` | ``IC_MIN ≤ score_ic < IC_GOOD`` — usable as a tie-breaker, not a primary signal |
| `NO_SIGNAL_EDGE` | ``-IC_GOOD < score_ic < IC_MIN`` — no demonstrated rank skill; this persona's return is beta/luck, not signal skill |

Overall verdict: ``INSUFFICIENT_DATA`` (< ``MIN_RECORDS`` aligned rows),
``HAS_INVERTED_PERSONA`` (≥1 ``INVERTED_SIGNAL`` — actionable red flag),
``NO_PERSONA_EDGE`` (no persona reaches ``SIGNAL_EDGE``), or ``HEALTHY``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Single source of truth for run_id → persona. Importing (not
# reimplementing) the mapping means a future PERSONAS-dict reordering can
# never silently shift every historical aggregate — the same discipline
# persona_leaderboard.py uses.
from paper_trader.backtest import persona_for
# Single source of truth for the tie-aware rank correlation — reused so
# this decision-level IC and the in-sample calibration spearman can never
# drift (the AGENTS.md invariant-10 spirit, exactly as _oos_rank_metrics
# reuses it).
from paper_trader.ml.calibration import _spearman
from paper_trader.ml.decision_scorer import _to_float

# Thresholds are module-level so tests assert exact verdicts and a tuning
# change is a single reviewable edit (mirrors calibration.py /
# persona_leaderboard.py and the codebase's constants-at-module-scope rule).
MIN_RECORDS = 30              # minimum aligned outcomes overall for any verdict
MIN_OUTCOMES_PER_PERSONA = 20  # below this a persona's Spearman is not stable
IC_MIN = 0.05                 # below |this| there is essentially no rank skill
                              # (same convention as skill_trend.IC_MIN)
IC_GOOD = 0.15                # "real edge" bar for SIGNAL_EDGE / INVERTED_SIGNAL


def _aligned(record: dict):
    """Return ``(signal, target)`` both action-aligned, or ``None`` to drop.

    ``target``  = ``forward_return_5d``  (``-`` for a SELL).
    ``signal``  = ``ml_score``           (``-`` for a SELL).

    The SELL double-flip keeps "higher signal ⇒ higher realized goodness"
    monotone across BUY and SELL — the exact symmetric extension of the
    codebase's universal SELL target sign-flip. A missing/non-finite
    ``ml_score`` or ``forward_return_5d`` drops the row (``_to_float``
    rejects NaN/±inf on both the Python and numpy branches), so a single
    poisoned outcomes line cannot corrupt the IC — the same hardening
    class as ``calibration.calibration_report``'s finite filter.
    """
    fr = record.get("forward_return_5d")
    ms = record.get("ml_score")
    if fr is None or ms is None:
        return None
    # Sentinel defaults that are themselves finite, so a genuine 0.0 and a
    # dropped value are distinguishable: use NaN sentinels and re-check.
    t = _to_float(fr, float("nan"))
    s = _to_float(ms, float("nan"))
    if t != t or s != s:  # NaN ⇒ unparseable / non-finite ⇒ drop
        return None
    if str(record.get("action") or "BUY").upper() == "SELL":
        t = -t
        s = -s
    return s, t


def persona_skill(records) -> dict:
    """Aggregate ``decision_outcomes.jsonl`` rows by trading persona and
    report each persona's decision-signal rank skill (``score_ic``).

    ``records`` is any iterable of dicts with at least ``run_id``,
    ``action``, ``ml_score``, ``forward_return_5d`` (the
    ``decision_outcomes.jsonl`` row shape). Rows whose ``run_id`` is
    missing/unmappable, or whose signal/target is missing/non-finite, are
    dropped.

    Returns a JSON-safe dict:
    ``{status, verdict, n_records, n_personas, personas:[{persona, n,
       score_ic, mean_aligned_return, win_rate, mean_signal, std_return,
       verdict}], inverted_personas:[...], hint}``. ``personas`` is sorted
    by ``score_ic`` descending (``INSUFFICIENT`` personas last).
    """
    buckets: dict[str, dict] = {}
    n_aligned = 0

    for r in records:
        rid = r.get("run_id")
        try:
            persona = persona_for(int(rid))["name"]
        except Exception:
            continue
        pair = _aligned(r)
        if pair is None:
            continue
        s, t = pair
        n_aligned += 1
        b = buckets.setdefault(persona, {"sig": [], "tgt": []})
        b["sig"].append(s)
        b["tgt"].append(t)

    if n_aligned < MIN_RECORDS:
        return {
            "status": "insufficient_data",
            "verdict": "INSUFFICIENT_DATA",
            "n_records": n_aligned,
            "n_personas": len(buckets),
            "personas": [],
            "inverted_personas": [],
            "hint": (f"need ≥{MIN_RECORDS} aligned outcomes with a finite "
                     f"ml_score + forward_return_5d, have {n_aligned}"),
        }

    personas = []
    inverted = []
    for persona, b in buckets.items():
        n = len(b["sig"])
        sig = np.asarray(b["sig"], dtype=np.float64)
        tgt = np.asarray(b["tgt"], dtype=np.float64)
        # _spearman is the imported single-source tie-aware rank corr; it
        # already returns 0.0 (never NaN) for a constant signal/target.
        ic = round(float(_spearman(sig, tgt)), 4)
        mean_ret = round(float(tgt.mean()), 4)
        win_rate = round(float(np.mean(tgt > 0.0)), 4)
        mean_sig = round(float(sig.mean()), 4)
        std_ret = round(float(tgt.std(ddof=0)), 4)
        if n < MIN_OUTCOMES_PER_PERSONA:
            verdict = "INSUFFICIENT"
        elif ic <= -IC_GOOD:
            verdict = "INVERTED_SIGNAL"
            inverted.append(persona)
        elif ic >= IC_GOOD:
            verdict = "SIGNAL_EDGE"
        elif ic >= IC_MIN:
            verdict = "WEAK_SIGNAL_EDGE"
        else:
            verdict = "NO_SIGNAL_EDGE"
        personas.append({
            "persona": persona,
            "n": n,
            "score_ic": ic,
            "mean_aligned_return": mean_ret,
            "win_rate": win_rate,
            "mean_signal": mean_sig,
            "std_return": std_ret,
            "verdict": verdict,
        })

    # Sort by score_ic desc; INSUFFICIENT personas always sink last
    # regardless of their (unstable, small-n) IC.
    personas.sort(
        key=lambda d: (d["verdict"] != "INSUFFICIENT", d["score_ic"]),
        reverse=True,
    )

    has_edge = any(p["verdict"] == "SIGNAL_EDGE" for p in personas)
    if inverted:
        verdict = "HAS_INVERTED_PERSONA"
        hint = (f"{len(inverted)} persona(s) have an anti-predictive signal "
                f"(score_ic ≤ -{IC_GOOD}): {', '.join(sorted(inverted))}. The "
                f"stronger their signal, the WORSE the realized 5d outcome — "
                f"this is the data for a (separate, explicit) decision to "
                f"prune or invert their _PERSONA_BOOSTS row. Do NOT change "
                f"PERSONAS/_PERSONA_BOOSTS from this read-only audit.")
    elif not has_edge:
        verdict = "NO_PERSONA_EDGE"
        hint = (f"no persona's decision signal rank-predicts realized "
                f"outcomes (score_ic ≥ {IC_GOOD}) on a stable sample — "
                f"per-persona returns are leveraged-beta dispersion, not "
                f"demonstrated signal skill (exactly the AGENTS.md "
                f"'read vs_spy_pct skeptically' warning, at decision level)")
    else:
        verdict = "HEALTHY"
        hint = ("≥1 persona's decision signal rank-predicts realized "
                "outcomes and none is anti-predictive")

    return {
        "status": "ok",
        "verdict": verdict,
        "n_records": n_aligned,
        "n_personas": len(buckets),
        "personas": personas,
        "inverted_personas": sorted(inverted),
        "hint": hint,
    }


def _load_outcomes(path: Path) -> list[dict]:
    """Robust JSONL load of ``decision_outcomes.jsonl``. Skips unparseable
    lines; never raises (a missing/corrupt file yields ``[]`` so the CLI
    degrades to ``INSUFFICIENT_DATA`` rather than crashing — the same
    best-effort discipline as ``skill_trend.load_skill_ledger``)."""
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


def _cli() -> int:
    """`python3 -m paper_trader.ml.persona_skill` — per-persona
    decision-signal rank-skill over the live ``decision_outcomes.jsonl``.
    Read-only; never writes anything. Exit 0 healthy/insufficient, 2 if any
    persona is INVERTED (so an operator/cron can branch on it, exactly like
    calibration._cli / persona_leaderboard._cli)."""
    root = Path(__file__).resolve().parent.parent.parent
    out_path = root / "data" / "decision_outcomes.jsonl"
    recs = _load_outcomes(out_path)
    rep = persona_skill(recs)
    print(f"aligned_outcomes={rep['n_records']}  personas={rep['n_personas']}")
    print(f"VERDICT: {rep['verdict']}  ({rep['hint']})")
    if rep["personas"]:
        print(f"  {'persona':<34} {'n':>5} {'score_ic':>9} "
              f"{'mean_ret':>9} {'win%':>6} {'mean_sig':>9} "
              f"{'std_ret':>8}  verdict")
        for e in rep["personas"]:
            print(f"  {e['persona']:<34} {e['n']:>5} "
                  f"{e['score_ic']:>+9.3f} {e['mean_aligned_return']:>+9.2f} "
                  f"{e['win_rate'] * 100:>5.0f}% {e['mean_signal']:>+9.2f} "
                  f"{e['std_return']:>8.2f}  {e['verdict']}")
    return 0 if rep["verdict"] in ("HEALTHY", "INSUFFICIENT_DATA",
                                   "NO_PERSONA_EDGE") else 2


if __name__ == "__main__":
    raise SystemExit(_cli())
