"""Sizing-rule counterfactual — would a different sizing rule have made
more realized PnL on the same corpus?

The deployed sizing rule (``_ml_decide``'s ``min(0.25, ml_score/20.0)``
for regular tickers, ``min(0.40, ml_score/15.0)`` for leveraged ETFs in
bull/sideways) ties conviction to the ML+quant score. The documented
CONVICTION-SIZING IS INVERTED finding (AGENTS.md 2026-05-30 Agent 2
pass #2) shows mean realized 5d return DECREASES with conviction in
live data — high-score / high-conviction trades cratering. That makes
the sizing rule itself a candidate for review.

This diagnostic answers the natural quant question that no existing
``paper_trader/ml/*`` module covers:

  *On the same outcome corpus, would a different sizing rule have
  generated more or less realized PnL than the rule the gate actually
  applied?*

`conviction_calibration` already asks "does sizing RANK realized
return?" (per-bucket spread, rank-skill verdict). `sizing_pnl_skill`
already asks "which conviction bucket contributes how many dollars
under the deployed rule?". `gate_pnl` measures only the ×0.6..×1.3
multiplier overlay's effect, not the BASE rule. None of them compares
the deployed rule against alternative rules on the SAME rows. That is
the gap.

This module sums ``Σ rule(r) × forward_return_5d_r`` across BUY rows
under several alternative sizing rules and reports the winner. The
sum is the per-trade PnL contribution in percentage-points of book
under the assumption that every base bet is independently deployable
(the same assumption `sizing_pnl_skill` makes in its dollar attribution
— same caveat applies: real portfolio compounding / cash constraints /
correlated holdings are NOT modelled; this is a relative comparison
of sizing rules, not a real backtest).

Tested rules:

* ``ACTUAL``        — uses each row's ``conviction_pct`` (the gate's
                      actual then-applied sizing)
* ``UNIFORM_10``    — flat 10% sizing on every BUY
* ``UNIFORM_25``    — flat 25% sizing on every BUY (the regular cap)
* ``SCORE_BASED``   — reconstructed ``min(0.25, max(0.0, ml_score/20.0))``
                      (matches ACTUAL when the upstream `score=` parse
                      succeeded and no leveraged-ETF cap kicked in)
* ``INVERSE_SCORE`` — ``min(0.25, max(0.05, 0.25 - ml_score/100.0))``
                      — explicit operationalization of the documented
                      inversion: SHRINK conviction as ml_score rises
* ``NEWS_DRIVEN``   — when ``news_urgency`` is finite and positive,
                      size as ``min(0.25, news_urgency/100.0)``;
                      otherwise fall back to 0.10 (news-feature is the
                      strongest single-feature OOS baseline per
                      ``baseline_compare``'s persistent finding)

Read-only operational discipline mirrors every existing
``paper_trader.ml.*`` diagnostic: never trains, never writes the
deployed pickle / outcomes JSONL / any trade path; never raises in the
analyzer functions (errors degrade to ``status='error'`` envelopes —
the AGENTS.md "ledger / diagnostic must not break the cycle"
discipline). Safe to run against the live unattended continuous loop.

Verdict ladder (test-locked, exact-threshold):

| Verdict              | Trigger                                                  |
|----------------------|----------------------------------------------------------|
| ``INSUFFICIENT_DATA``| < ``MIN_ROWS`` valid BUY rows                             |
| ``ALT_BEATS_ACTUAL`` | best alternative beats ACTUAL by ≥ ``MIN_REL_IMPROVEMENT``|
| ``ACTUAL_BEST``      | ACTUAL beats every alternative by ≥ ``MIN_REL_IMPROVEMENT``|
| ``TIE``              | best alternative within ±``MIN_REL_IMPROVEMENT`` of ACTUAL|

The relative-improvement metric (rather than absolute pp) is the
operationally meaningful threshold — an ACTUAL of -2.0pp losing to an
alternative at +5.0pp is operationally distinct from a +0.5pp losing
to +1.0pp despite the same 0.5pp absolute gap. ``MIN_REL_IMPROVEMENT``
defaults to 0.20 (20% — a meaningful but not extreme gap).

CLI:

```bash
cd /home/zeph/trading-intelligence/paper-trader

# Default — analyze data/decision_outcomes.jsonl
python3 -m paper_trader.ml.sizing_rule_counterfactual

# JSON output
python3 -m paper_trader.ml.sizing_rule_counterfactual --json

# Custom corpus
python3 -m paper_trader.ml.sizing_rule_counterfactual \\
    --outcomes path/to/alt.jsonl
```

Exit code is 0 on every acceptable verdict (``ACTUAL_BEST`` / ``TIE`` /
``INSUFFICIENT_DATA`` — the gate's sizing is at least competitive), and
2 on ``ALT_BEATS_ACTUAL`` (the quant-decisive "an alternative rule
beats the deployed one" state) so a shell caller can gate on a real
edge — same discipline as ``sizing_pnl_skill`` /
``conviction_calibration`` / ``gate_abstention``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable, Iterable

# Minimum BUY rows with finite ``conviction_pct`` AND
# ``forward_return_5d`` before a verdict is attempted. 60 is the same
# floor `sizing_pnl_skill` uses; same minimum thinness guard.
MIN_ROWS = 60

# Default minimum relative improvement (best_alt - actual) / |actual|
# below which alternatives are deemed "TIE". 0.20 (20%) keeps the
# verdict from flipping on small noise gaps.
MIN_REL_IMPROVEMENT = 0.20

# Documented sizing-rule constants. Keep in lockstep with `_ml_decide`'s
# `min(0.25, best_score / 20.0)` (regular) / `min(0.40, best_score /
# 15.0)` (leveraged-bull). Mirrored here so a future tweak to the live
# rule shows up as a measurable delta in this diagnostic immediately
# (rather than after a manual sync).
_REGULAR_CAP = 0.25
_REGULAR_DIVISOR = 20.0
_LEVERAGED_CAP = 0.40
_LEVERAGED_DIVISOR = 15.0

# Inverse-score rule parameters. Range [0.05, 0.25] — never zero (a
# trade chosen as the BUY pick by `_ml_decide` always gets at least
# `min_floor` conviction; this matches the live rule's behaviour where
# a BUY is never sized to literally zero), capped at the regular cap.
# Subtracted from cap by `ml_score / 100` so a +25 score (very high)
# lands at 0.0 → floored to 0.05. A 0 score lands at 0.25.
_INVERSE_FLOOR = 0.05


# ──────────────────────────── helpers ────────────────────────────


def _to_finite_float(v) -> float | None:
    """Mirror the codebase-wide ``_to_finite_float`` convention: a
    bool (subclass of int) returns None — we want only honest numerics."""
    if isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


# ──────────────────────────── sizing rules ────────────────────────


def _rule_actual(r: dict) -> float | None:
    """Use the row's already-parsed `conviction_pct`. None when missing
    or out of range — the upstream parser already clamps to [0, 1], so
    out-of-range means a hand-crafted / corrupt record."""
    v = _to_finite_float(r.get("conviction_pct"))
    if v is None or not (0.0 <= v <= 1.0):
        return None
    return v


def _rule_uniform_10(_: dict) -> float:
    return 0.10


def _rule_uniform_25(_: dict) -> float:
    return 0.25


def _rule_score_based(r: dict) -> float:
    """Reconstructed `min(0.25, max(0, ml_score / 20))`. Does NOT
    reproduce the leveraged-ETF branch because the outcome row carries
    no `is_leveraged` field — this is a strict subset of the live rule
    that matches `_ml_decide` for the regular branch only."""
    s = _to_finite_float(r.get("ml_score"))
    if s is None:
        s = 0.0
    return min(_REGULAR_CAP, max(0.0, s / _REGULAR_DIVISOR))


def _rule_inverse_score(r: dict) -> float:
    """Documented inversion test: SHRINK conviction as ml_score rises.
    `min(0.25, max(0.05, 0.25 - ml_score / 100))`. A 0-score trade
    sizes at the cap (0.25); a +25-score trade falls to the floor
    (0.05)."""
    s = _to_finite_float(r.get("ml_score"))
    if s is None:
        s = 0.0
    return min(_REGULAR_CAP, max(_INVERSE_FLOOR,
                                 _REGULAR_CAP - s / 100.0))


def _rule_news_driven(r: dict) -> float:
    """When `news_urgency` is finite and positive, size by it; else
    fall back to a flat 10%. news_urgency in the corpus ranges roughly
    0..100 — divide by 100 to land in [0, 1], cap at the regular cap
    `0.25`. Fallback `0.10` matches the corpus mean conviction so
    no-news rows are sized "the way the gate ACTUALLY sized most
    trades" — preserves the comparison apples-to-apples for rows the
    rule has no signal on."""
    u = _to_finite_float(r.get("news_urgency"))
    if u is None or u <= 0:
        return 0.10
    return min(_REGULAR_CAP, u / 100.0)


# Registry of rules. Order matters for stable reporting (table /
# JSON serialization). The first entry is the reference ACTUAL — every
# verdict compares the rest against it.
SIZING_RULES: tuple[tuple[str, Callable[[dict], float | None]], ...] = (
    ("ACTUAL", _rule_actual),
    ("UNIFORM_10", _rule_uniform_10),
    ("UNIFORM_25", _rule_uniform_25),
    ("SCORE_BASED", _rule_score_based),
    ("INVERSE_SCORE", _rule_inverse_score),
    ("NEWS_DRIVEN", _rule_news_driven),
)


# ──────────────────────────── analyzer ────────────────────────────


def _empty(reason: str) -> dict:
    """Honest-empty envelope shared by every early-return path. Tests
    pin the exact shape so a future analyzer that consumes this output
    can never break on a missing key."""
    return {
        "status": reason if reason in ("error", "insufficient_data") else "ok",
        "verdict": "INSUFFICIENT_DATA",
        "n": 0,
        "n_dropped_action": 0,
        "n_dropped_conviction": 0,
        "n_dropped_return": 0,
        "rules": [],
        "best_alt_rule": None,
        "actual_total_pp": None,
        "best_alt_total_pp": None,
        "rel_improvement": None,
        "hint": reason if not isinstance(reason, str) else (
            "" if reason in ("error", "insufficient_data") else reason),
    }


def build_sizing_counterfactual(
    records: Iterable[dict],
    min_rel_improvement: float = MIN_REL_IMPROVEMENT,
) -> dict:
    """Compute Σ rule(r) × forward_return_5d_r for each rule and verdict.

    Selection criteria for a record:
      * ``action == 'BUY'`` — conviction sizing is BUY-only (SELLs have
        no `conviction_pct` token in their reasoning, mirroring the
        `gate_scorer_pred` SELL convention).
      * ``conviction_pct`` finite AND in [0, 1] — same as
        `conviction_calibration`. An out-of-range here is corrupt.
      * ``forward_return_5d`` finite — a non-finite outcome cannot be
        summed with any rule's contribution.

    Returns a JSON-safe dict::

        {
            "status": "ok" | "insufficient_data" | "error",
            "verdict": "ALT_BEATS_ACTUAL" | "ACTUAL_BEST" | "TIE"
                       | "INSUFFICIENT_DATA",
            "n": <int>,                 # BUY rows accepted
            "n_dropped_action": <int>,
            "n_dropped_conviction": <int>,
            "n_dropped_return": <int>,
            "rules": [
                {"name": "ACTUAL", "total_pp": <float>,
                 "per_trade_pp": <float>, "mean_conviction": <float>},
                ...
            ],
            "best_alt_rule": <str or None>,
            "actual_total_pp": <float>,
            "best_alt_total_pp": <float>,
            "rel_improvement": <float>,  # (best_alt - actual) / |actual|
            "hint": <str>,
        }

    Never raises — any unexpected failure degrades to an error
    envelope (the `decision_outcomes.jsonl` parser / scorer / gate
    discipline)."""
    try:
        recs: list[dict] = []
        n_skip_action = 0
        n_skip_conv = 0
        n_skip_ret = 0
        for r in records:
            if not isinstance(r, dict):
                n_skip_action += 1
                continue
            action = str(r.get("action") or "").upper()
            if action != "BUY":
                n_skip_action += 1
                continue
            # conviction_pct must be valid for the ACTUAL rule to compute,
            # AND for direct comparability of alternatives — only consider
            # rows where the ACTUAL rule could fire.
            conv = _rule_actual(r)
            if conv is None:
                n_skip_conv += 1
                continue
            y = _to_finite_float(r.get("forward_return_5d"))
            if y is None:
                n_skip_ret += 1
                continue
            recs.append(r)

        n = len(recs)
        if n < MIN_ROWS:
            env = _empty("insufficient_data")
            env["n"] = n
            env["n_dropped_action"] = n_skip_action
            env["n_dropped_conviction"] = n_skip_conv
            env["n_dropped_return"] = n_skip_ret
            env["hint"] = (
                f"need ≥{MIN_ROWS} BUY rows with finite conviction_pct "
                f"AND forward_return_5d, have {n}"
            )
            return env

        # Per-rule total contribution + mean conviction. Mean conviction
        # is reported so a reader can sanity-check the rule magnitude
        # (e.g. UNIFORM_25 should read mean_conviction = 0.25 exactly).
        rule_rows: list[dict] = []
        actual_total: float | None = None
        for name, fn in SIZING_RULES:
            total_pp = 0.0
            conv_sum = 0.0
            for r in recs:
                c = fn(r)
                if c is None:
                    # ACTUAL was already validated, so a None here means
                    # the alternative rule itself returned None — count
                    # as zero conviction (no contribution). This only
                    # happens for the optional alternatives.
                    c = 0.0
                y = _to_finite_float(r.get("forward_return_5d")) or 0.0
                total_pp += c * y
                conv_sum += c
            row = {
                "name": name,
                "total_pp": round(total_pp, 4),
                "per_trade_pp": round(total_pp / n, 6),
                "mean_conviction": round(conv_sum / n, 6),
            }
            rule_rows.append(row)
            if name == "ACTUAL":
                actual_total = total_pp

        # Find best alternative (excluding ACTUAL).
        alt_rows = [r for r in rule_rows if r["name"] != "ACTUAL"]
        best_alt = max(alt_rows, key=lambda x: x["total_pp"])
        best_alt_total = best_alt["total_pp"]

        # Relative improvement against ACTUAL. Denominator is |actual|
        # to make the metric symmetric (an alt that goes from -5pp to
        # +5pp is a 200% improvement, not infinite). Guard against
        # actual==0 by falling back to absolute difference normalized
        # to per-trade pp (a degenerate but well-defined fallback).
        if actual_total is None or abs(actual_total) < 1e-9:
            rel_improvement = (best_alt_total - (actual_total or 0.0)) / max(
                1e-9, abs(best_alt_total) + 1e-9)
        else:
            rel_improvement = (best_alt_total - actual_total) / abs(actual_total)

        if rel_improvement >= min_rel_improvement:
            verdict = "ALT_BEATS_ACTUAL"
            hint = (f"{best_alt['name']} total_pp={best_alt_total:+.2f} beats "
                    f"ACTUAL={actual_total:+.2f} by {rel_improvement*100:+.1f}%"
                    f" — sizing rule is a candidate for review")
        elif rel_improvement <= -min_rel_improvement:
            verdict = "ACTUAL_BEST"
            hint = (f"ACTUAL={actual_total:+.2f} beats best alt "
                    f"{best_alt['name']}={best_alt_total:+.2f} by "
                    f"{-rel_improvement*100:+.1f}%"
                    " — deployed rule is competitive on this corpus")
        else:
            verdict = "TIE"
            hint = (f"ACTUAL={actual_total:+.2f} and best alt "
                    f"{best_alt['name']}={best_alt_total:+.2f} are within "
                    f"±{min_rel_improvement*100:.0f}% — no decisive winner")

        return {
            "status": "ok",
            "verdict": verdict,
            "n": n,
            "n_dropped_action": n_skip_action,
            "n_dropped_conviction": n_skip_conv,
            "n_dropped_return": n_skip_ret,
            "rules": rule_rows,
            "best_alt_rule": best_alt["name"],
            "actual_total_pp": round(actual_total, 4),
            "best_alt_total_pp": round(best_alt_total, 4),
            "rel_improvement": round(rel_improvement, 6),
            "hint": hint,
        }
    except Exception as exc:
        env = _empty("error")
        env["status"] = "error"
        env["hint"] = f"unexpected error: {type(exc).__name__}: {exc}"
        return env


def load_outcomes(path: Path | str) -> list[dict]:
    """Read the corpus, one JSONL row per line, dropping unparseable.
    Mirrors `conviction_calibration.load_outcomes` exactly so the same
    loader convention covers every diagnostic — single source of truth
    for the JSONL read."""
    p = Path(path)
    if not p.exists():
        return []
    out: list[dict] = []
    try:
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                out.append(row)
    except Exception:
        return []
    return out


def analyze(
    outcomes_path: Path | str,
    min_rel_improvement: float = MIN_REL_IMPROVEMENT,
) -> dict:
    """Convenience: load corpus → analyze → return dict.
    Never raises (`load_outcomes` and `build_sizing_counterfactual`
    both degrade)."""
    return build_sizing_counterfactual(
        load_outcomes(outcomes_path),
        min_rel_improvement=min_rel_improvement,
    )


# ──────────────────────────── CLI ────────────────────────────


def _print_report(rep: dict) -> None:
    print(f"[sizing_rule_counterfactual] verdict={rep.get('verdict')}  "
          f"status={rep.get('status')}  n={rep.get('n')}")
    if rep.get("n_dropped_action") or rep.get("n_dropped_conviction") \
            or rep.get("n_dropped_return"):
        print(f"  dropped: action={rep.get('n_dropped_action')}  "
              f"conviction={rep.get('n_dropped_conviction')}  "
              f"return={rep.get('n_dropped_return')}")
    rows = rep.get("rules") or []
    if rows:
        print(f"  {'rule':<14}{'total_pp':>12}{'per_trade_pp':>15}"
              f"{'mean_conv':>12}")
        for r in rows:
            print(f"  {r['name']:<14}{r['total_pp']:>+12.2f}"
                  f"{r['per_trade_pp']:>+15.4f}{r['mean_conviction']:>12.4f}")
    if rep.get("hint"):
        print(f"  hint: {rep['hint']}")


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="python3 -m paper_trader.ml.sizing_rule_counterfactual",
        description=(
            "Compare actual conviction sizing against alternative "
            "rules on the same outcome corpus. Read-only. "
            "Exit 2 on ALT_BEATS_ACTUAL."
        ),
    )
    p.add_argument(
        "--outcomes", default="data/decision_outcomes.jsonl",
        help="Path to decision_outcomes.jsonl (default: data/decision_outcomes.jsonl)",
    )
    p.add_argument(
        "--min-rel-improvement", type=float,
        default=MIN_REL_IMPROVEMENT,
        help=(f"Min relative improvement to declare ALT_BEATS_ACTUAL "
              f"(default: {MIN_REL_IMPROVEMENT})"),
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

    # Exit 2 on the quant-decisive "deployed rule loses" state, 0 on
    # everything else — same gate-discipline as sizing_pnl_skill.
    return 2 if rep.get("verdict") == "ALT_BEATS_ACTUAL" else 0


if __name__ == "__main__":
    raise SystemExit(_cli())
