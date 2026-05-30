"""Regime-conditional sizing-rule counterfactual.

Sibling of ``paper_trader.ml.sizing_rule_counterfactual`` — extends that
analyzer with a per-regime breakdown. Read-only research signal; never
trains, never writes the pickle / outcomes JSONL / any trade path; safe
to run against the live unattended continuous loop.

**The gap this closes.** ``sizing_rule_counterfactual`` reports the
global winner among 6 sizing rules. The current production verdict on
the 5512-row OOS corpus (AGENTS.md 2026-05-30 Agent 2 pass #3) is
``ALT_BEATS_ACTUAL`` — ``UNIFORM_25`` total +994pp vs ACTUAL +436pp,
a +128% relative improvement.

That global verdict hides a load-bearing question for a quant deciding
whether to actually deploy ``UNIFORM_25``:

  *Does the alternative win UNIFORMLY across regimes, or is the global
  winner hostage to one fat regime (typically bull, since the corpus
  is regime-skewed) that wins big enough to drown a bear-regime drag?*

The live corpus is ~99.8% non-bear (only 11/5512 bear rows per the
documented ``oos_bear_n=0`` finding), so a global ``UNIFORM_25`` win
could be entirely a beta-leverage artifact. A rule that loses badly in
bear cannot be safely deployed without further protection — even if it
wins on aggregate.

This module answers:
  * What is each rule's PnL contribution PER REGIME?
  * Which rule wins each regime?
  * Is the winner the SAME across regimes (deploy with confidence) or
    DIFFERENT (sizing should be regime-conditional, not global)?

The verdict ladder is verbatim ``sizing_rule_counterfactual`` per regime;
the aggregate verdict combines them into a single operator signal:

| Aggregate verdict | Trigger |
|---|---|
| ``INSUFFICIENT_DATA`` | < ``MIN_REGIMES`` (2) populated regimes |
| ``ALL_SAME_WINNER``   | Every populated regime's winner is the SAME alt (or ACTUAL); strong evidence for that rule |
| ``DIFFERENT_WINNERS`` | At least one regime's winner differs from another's; sizing should be regime-conditional |
| ``BULL_DOMINATED``    | Only bull has ≥ ``MIN_ROWS`` rows; the global verdict can't generalize |
| ``ALL_TIE``           | Every populated regime is TIE — no rule has a regime-level edge |

Read-only operational discipline mirrors every existing
``paper_trader.ml.*`` diagnostic. Errors degrade to ``status='error'``
envelopes — the analyzer never raises. CLI exits 0 on
ALL_SAME_WINNER / ALL_TIE / INSUFFICIENT_DATA / BULL_DOMINATED
(informational), 2 on DIFFERENT_WINNERS (quant-decisive — the deployed
global rule is masking per-regime structure).

CLI::

    cd /home/zeph/trading-intelligence/paper-trader

    # Default — analyze data/decision_outcomes.jsonl, table verdict
    python3 -m paper_trader.ml.sizing_rule_regime_breakdown

    # JSON output
    python3 -m paper_trader.ml.sizing_rule_regime_breakdown --json

    # Custom corpus / tolerance
    python3 -m paper_trader.ml.sizing_rule_regime_breakdown \\
        --outcomes path/to/alt.jsonl \\
        --min-rel-improvement 0.30
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .sizing_rule_counterfactual import (
    MIN_REL_IMPROVEMENT,
    MIN_ROWS,
    SIZING_RULES,
    build_sizing_counterfactual,
    load_outcomes,
    _to_finite_float,
)

# Minimum number of populated regimes (≥ ``MIN_ROWS`` rows each) before
# the aggregate verdict is meaningful. A single regime cannot tell us
# whether the global winner generalizes — but two is enough to detect
# DIFFERENT_WINNERS.
MIN_REGIMES = 2

# Regime label canonical order. Used for stable reporting and
# decode-fallback below. Mirrors ``_oos_rank_metrics`` /
# ``regime_audit`` so a downstream consumer can JOIN by regime label
# without a lookup table.
REGIMES: tuple[str, ...] = ("bull", "sideways", "bear")

# Decode ``regime_mult`` → regime label for legacy rows that pre-date
# the explicit ``regime_label`` field. Mirrors ``_oos_rank_metrics``'
# ``_REGIME_BY_MULT`` so the two analyzers agree on the bucket
# assignment for legacy corpora.
_REGIME_BY_MULT: dict[float, str] = {0.3: "bear", 0.6: "sideways", 1.0: "bull"}


def _row_regime(r: dict) -> str | None:
    """Decode a row's regime label.

    Prefer the explicit ``regime_label`` field (the 2026-05-19 outcome
    schema addition). For pre-feature legacy rows fall back to decoding
    ``regime_mult`` via ``_REGIME_BY_MULT``. An explicit ``"unknown"``
    label or a regime_mult that doesn't decode (the early days of a
    backtest window with <200 SPY closes) is intentionally dropped — the
    documented unknown-regime fall-through.

    Returns one of ``"bull"`` / ``"sideways"`` / ``"bear"`` or None
    when the row carries no usable regime signal.
    """
    label = r.get("regime_label")
    if isinstance(label, str) and label in REGIMES:
        return label
    if label is None:
        mult = _to_finite_float(r.get("regime_mult"))
        if mult is not None:
            return _REGIME_BY_MULT.get(mult)
    # Any other label (e.g. "unknown") is intentionally dropped.
    return None


def _empty(reason: str = "insufficient_data") -> dict:
    """Honest-empty envelope shared by every early-return path."""
    return {
        "status": reason if reason in ("error", "insufficient_data") else "ok",
        "verdict": "INSUFFICIENT_DATA",
        "n_total": 0,
        "n_regime_decoded": 0,
        "n_regime_missing": 0,
        "regimes": [],
        "hint": "",
    }


def _winner_name(report: dict) -> str | None:
    """Return the winning rule name for one per-regime report.

    For ALT_BEATS_ACTUAL the winner is ``best_alt_rule``. For ACTUAL_BEST
    the winner is ``"ACTUAL"``. For TIE there is no decisive winner —
    return None so the aggregate verdict treats this regime as
    inconclusive. For INSUFFICIENT_DATA / error return None.
    """
    v = report.get("verdict")
    if v == "ALT_BEATS_ACTUAL":
        return report.get("best_alt_rule")
    if v == "ACTUAL_BEST":
        return "ACTUAL"
    return None


def build_regime_breakdown(
    records: Iterable[dict],
    min_rel_improvement: float = MIN_REL_IMPROVEMENT,
) -> dict:
    """Run ``build_sizing_counterfactual`` per regime and aggregate.

    Returns a JSON-safe dict::

        {
            "status": "ok" | "error" | "insufficient_data",
            "verdict": "ALL_SAME_WINNER" | "DIFFERENT_WINNERS" | "ALL_TIE"
                       | "BULL_DOMINATED" | "INSUFFICIENT_DATA",
            "n_total": <int>,
            "n_regime_decoded": <int>,
            "n_regime_missing": <int>,
            "regimes": [
                {
                    "regime": "bull",
                    "verdict": "ALT_BEATS_ACTUAL" | ... ,
                    "n": <int>,
                    "winner": <str or None>,
                    "rules": [...],            # passthrough from per-regime
                    "actual_total_pp": <float>,
                    "best_alt_rule": <str or None>,
                    "best_alt_total_pp": <float>,
                    "rel_improvement": <float>,
                    ...
                },
                ...
            ],
            "winners": ["UNIFORM_25", "ACTUAL", ...],   # one per populated regime
            "hint": <str>,
        }

    Never raises — any unexpected failure degrades to an error envelope.
    """
    try:
        # Materialize records once so we can iterate per-regime without
        # consuming a generator twice.
        all_recs = [r for r in records if isinstance(r, dict)]
        n_total = len(all_recs)

        per_regime: dict[str, list[dict]] = {r: [] for r in REGIMES}
        n_decoded = 0
        for r in all_recs:
            reg = _row_regime(r)
            if reg is None:
                continue
            n_decoded += 1
            per_regime[reg].append(r)
        n_missing = n_total - n_decoded

        regime_reports: list[dict] = []
        for reg in REGIMES:
            rep = build_sizing_counterfactual(
                per_regime[reg], min_rel_improvement=min_rel_improvement)
            # Flatten the per-regime envelope into a regime row.
            regime_reports.append({
                "regime": reg,
                "verdict": rep.get("verdict"),
                "status": rep.get("status"),
                "n": rep.get("n", 0),
                "winner": _winner_name(rep),
                "rules": rep.get("rules", []),
                "actual_total_pp": rep.get("actual_total_pp"),
                "best_alt_rule": rep.get("best_alt_rule"),
                "best_alt_total_pp": rep.get("best_alt_total_pp"),
                "rel_improvement": rep.get("rel_improvement"),
                "hint": rep.get("hint", ""),
            })

        # Populated regime = enough rows that the per-regime analyzer
        # produced a non-INSUFFICIENT verdict.
        populated = [r for r in regime_reports
                     if r["verdict"] != "INSUFFICIENT_DATA"]
        winners = [r["winner"] for r in populated if r["winner"] is not None]

        # Aggregate verdict ladder.
        if not populated:
            verdict = "INSUFFICIENT_DATA"
            hint = (f"no regime has ≥{MIN_ROWS} BUY rows with finite "
                    f"conviction_pct AND forward_return_5d — analyzer "
                    f"degrades to INSUFFICIENT_DATA")
        elif len(populated) < MIN_REGIMES:
            # Exactly one populated regime — typically bull due to the
            # documented bull-skew in the live corpus.
            only_reg = populated[0]["regime"]
            verdict = "BULL_DOMINATED" if only_reg == "bull" else "INSUFFICIENT_DATA"
            hint = (f"only {only_reg} has ≥{MIN_ROWS} rows — global "
                    f"sizing-rule verdict can't generalize across regimes")
        elif not winners:
            # Every populated regime returned TIE.
            verdict = "ALL_TIE"
            hint = (f"every populated regime (n={len(populated)}) is TIE "
                    f"within ±{min_rel_improvement*100:.0f}% — no rule "
                    f"has a regime-level edge")
        elif len(set(winners)) == 1 and len(winners) == len(populated):
            # All populated regimes have the same decisive winner. Strong
            # evidence: the rule generalizes.
            verdict = "ALL_SAME_WINNER"
            hint = (f"{winners[0]} wins every populated regime "
                    f"(n={len(populated)}) — strong evidence the rule "
                    f"generalizes; deploy with confidence")
        elif len(set(winners)) > 1:
            verdict = "DIFFERENT_WINNERS"
            uniq = sorted(set(winners))
            hint = (f"different rules win across regimes "
                    f"({', '.join(uniq)}) — global verdict masks "
                    f"per-regime structure; sizing should be "
                    f"regime-conditional, not global")
        else:
            # winners list is a subset of populated (some regimes were
            # TIE), but the populated regimes that DID have a winner
            # all agreed. This is weaker evidence than ALL_SAME_WINNER
            # but still consistent with that rule.
            verdict = "ALL_SAME_WINNER"
            hint = (f"{winners[0]} wins {len(winners)} regime(s); the "
                    f"remaining {len(populated)-len(winners)} populated "
                    f"regime(s) were TIE — consistent with deploying "
                    f"{winners[0]} but with weaker per-regime power")

        return {
            "status": "ok",
            "verdict": verdict,
            "n_total": n_total,
            "n_regime_decoded": n_decoded,
            "n_regime_missing": n_missing,
            "regimes": regime_reports,
            "winners": winners,
            "hint": hint,
        }
    except Exception as exc:
        env = _empty("error")
        env["status"] = "error"
        env["hint"] = f"unexpected error: {type(exc).__name__}: {exc}"
        return env


def analyze(
    outcomes_path: Path | str,
    min_rel_improvement: float = MIN_REL_IMPROVEMENT,
) -> dict:
    """Convenience: load corpus → analyze → return dict.

    Never raises (``load_outcomes`` and ``build_regime_breakdown``
    both degrade).
    """
    return build_regime_breakdown(
        load_outcomes(outcomes_path),
        min_rel_improvement=min_rel_improvement,
    )


# ──────────────────────────── CLI ────────────────────────────


def _print_report(rep: dict) -> None:
    print(f"[sizing_rule_regime_breakdown] verdict={rep.get('verdict')}  "
          f"status={rep.get('status')}  n_total={rep.get('n_total')}  "
          f"n_decoded={rep.get('n_regime_decoded')}  "
          f"n_missing={rep.get('n_regime_missing')}")
    regimes = rep.get("regimes") or []
    for r in regimes:
        marker = "✓" if r["verdict"] != "INSUFFICIENT_DATA" else "·"
        print(f"  {marker} {r['regime']:<9}n={r['n']:<5}"
              f"verdict={r['verdict']:<20}"
              f"winner={(r['winner'] or '-'):<14}")
        if r.get("actual_total_pp") is not None and r.get("best_alt_total_pp") is not None:
            print(f"      ACTUAL={r['actual_total_pp']:+.2f}pp  "
                  f"best_alt={r['best_alt_rule']}="
                  f"{r['best_alt_total_pp']:+.2f}pp  "
                  f"rel_improvement={(r.get('rel_improvement') or 0)*100:+.1f}%")
    if rep.get("hint"):
        print(f"  hint: {rep['hint']}")


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="python3 -m paper_trader.ml.sizing_rule_regime_breakdown",
        description=(
            "Per-regime sizing-rule counterfactual. Sibling of "
            "sizing_rule_counterfactual. Read-only. "
            "Exit 2 on DIFFERENT_WINNERS."
        ),
    )
    p.add_argument(
        "--outcomes", default="data/decision_outcomes.jsonl",
        help="Path to decision_outcomes.jsonl",
    )
    p.add_argument(
        "--min-rel-improvement", type=float,
        default=MIN_REL_IMPROVEMENT,
        help=(f"Min relative improvement for an in-regime "
              f"ALT_BEATS_ACTUAL verdict (default: {MIN_REL_IMPROVEMENT})"),
    )
    p.add_argument("--json", action="store_true",
                   help="Emit JSON instead of a table")
    args = p.parse_args(argv)

    rep = analyze(args.outcomes,
                  min_rel_improvement=args.min_rel_improvement)

    if args.json:
        print(json.dumps(rep, indent=2, sort_keys=True))
    else:
        _print_report(rep)

    # Exit 2 on the quant-decisive state. DIFFERENT_WINNERS means the
    # deployed global sizing rule is masking per-regime structure — a
    # real, actionable finding.
    return 2 if rep.get("verdict") == "DIFFERENT_WINNERS" else 0


if __name__ == "__main__":
    raise SystemExit(_cli())
