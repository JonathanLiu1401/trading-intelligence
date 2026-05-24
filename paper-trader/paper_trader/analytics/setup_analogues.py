"""Empirical conditional return distribution from historical analogues.

The DecisionScorer MLP (and its companion ``/api/scorer-confidence``,
``/api/calibration``) summarise the corpus by *prediction band* — "when the
model predicts +5%, what's the typical realized residual?" That answers
**model trust**, not **setup trust**: it cannot tell a trader "this exact
combination of RSI / 20-day momentum / regime has happened N times in the
8753-row history, and here's the empirical return distribution conditional
on that setup."

This module fills that gap. Given a current ticker's quant context and an
intended action, it finds rows in ``data/decision_outcomes.jsonl`` matching
that setup by coarse feature buckets and reports the empirical forward-5d
return distribution: p25 / median / p75 / mean / best / worst / win-rate.
It is the **non-parametric companion** to the MLP — no model is trained,
no pickle is touched. If the MLP's point estimate disagrees with the
empirical bucket median, that's a legitimate trust signal for the trader.

Buckets are intentionally coarse (4×4×3 = 48 cells) so even rare setups
get >= 20 matches in the 8753-row corpus, while still capturing the
regime / momentum / overbought-extreme distinctions that actually move
realized returns. Finer buckets fragment the sample; coarser ones collapse
distinct setups into one verdict — see ``_RSI_EDGES`` / ``_MOM20_EDGES`` /
``_REGIME_EDGES`` for the exact thresholds.

Action semantics: BUY analogues count win as ``forward_return_5d > 0``;
SELL analogues count win as ``forward_return_5d < 0`` (a successful sell
sits ahead of a down move). Percentile stats are always reported as
**raw market return** — the operator reads them in market space.

Read-only diagnostic. Never trains, never touches the pickle, never enters
a trade path.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Coarse buckets — 4 × 4 × 3 = 48 cells. Edges picked so common setups land
# in mid-buckets (not boundary-hugged) and the overbought / oversold tails
# stay distinct (their conditional distributions differ materially).
_RSI_EDGES = [(-float("inf"), 30.0, "oversold"),
              (30.0, 50.0, "mid_low"),
              (50.0, 70.0, "mid_high"),
              (70.0, float("inf"), "overbought")]
_MOM20_EDGES = [(-float("inf"), -10.0, "deep_neg"),
                (-10.0, 0.0, "neg"),
                (0.0, 10.0, "pos"),
                (10.0, float("inf"), "strong")]
# regime_mult is _market_regime's SPY-trend multiplier: ~0.6 bear, 1.0
# sideways, ~1.3 bull. Pick edges that catch the engine's actual values.
_REGIME_EDGES = [(-float("inf"), 0.85, "bear"),
                 (0.85, 1.05, "sideways"),
                 (1.05, float("inf"), "bull")]

_BUY_ACTIONS = {"BUY", "BUY_CALL", "BUY_PUT"}
_SELL_ACTIONS = {"SELL", "SELL_CALL", "SELL_PUT"}

# Minimum analogue count before emitting a verdict.  Below this, we still
# return the stats (caller may want them for a "thin sample" subhead) but
# the verdict collapses to ``INSUFFICIENT_DATA`` — the same honesty floor
# the calibration / scorer-confidence modules use.
DEFAULT_MIN_MATCHES = 20


def _bucket(value: Any, edges: list[tuple]) -> str | None:
    """Return the named bucket for ``value`` against ``(lo, hi, name)`` edges.

    None / non-numeric / NaN ⇒ None (the caller treats it as "no match").
    """
    if not isinstance(value, (int, float)) or value != value:  # NaN check
        return None
    v = float(value)
    for lo, hi, name in edges:
        if lo <= v < hi:
            return name
    # Right-most edge is half-open on the high side; catch the exact bound.
    if v == edges[-1][1]:
        return edges[-1][2]
    return None


def _bucketize(rsi: Any, mom20: Any, regime_mult: Any) -> dict[str, str | None]:
    """Map (rsi, mom20, regime_mult) → labelled bucket cell."""
    return {
        "rsi": _bucket(rsi, _RSI_EDGES),
        "mom20": _bucket(mom20, _MOM20_EDGES),
        "regime": _bucket(regime_mult, _REGIME_EDGES),
    }


def _is_win(action: str, forward_return_5d: float) -> bool:
    """Trader-perspective win: BUY wins when market up, SELL wins when down.

    Options match the underlying direction (BUY_CALL ↔ BUY, SELL_PUT ↔ SELL).
    """
    if action in _SELL_ACTIONS:
        return forward_return_5d < 0
    return forward_return_5d > 0


def _verdict(median: float, win_rate: float, n: int,
             min_matches: int) -> str:
    """Map (median, win_rate, n) → one of the closed verdict alphabet."""
    if n < min_matches:
        return "INSUFFICIENT_DATA"
    # Trader-perspective scoring: median is in raw market space, so for a
    # SELL the "good" direction is negative median.  Caller passes the
    # already-trader-signed median (positive ⇒ trade in this direction
    # historically pays).  See ``build_setup_analogues``.
    if median > 3.0 and win_rate > 0.60:
        return "STRONG_EDGE"
    if median > 1.0 and win_rate >= 0.55:
        return "EDGE"
    if median < -3.0 and win_rate < 0.40:
        return "STRONG_HEADWIND"
    if median < -1.0 and win_rate <= 0.45:
        return "HEADWIND"
    return "NEUTRAL"


def _percentiles(values: list[float]) -> dict[str, float]:
    """Pure-Python percentiles (no numpy dep — matches sibling pure builders)."""
    if not values:
        return {"p25": 0.0, "p50": 0.0, "p75": 0.0,
                "mean": 0.0, "best": 0.0, "worst": 0.0}
    s = sorted(values)
    n = len(s)

    def q(p: float) -> float:
        # Linear interpolation, matching numpy's default 'linear' method.
        idx = (n - 1) * p
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        return s[lo] * (1 - frac) + s[hi] * frac

    return {
        "p25": round(q(0.25), 2),
        "p50": round(q(0.50), 2),
        "p75": round(q(0.75), 2),
        "mean": round(sum(s) / n, 2),
        "best": round(s[-1], 2),
        "worst": round(s[0], 2),
    }


def build_setup_analogues(
    outcomes: list[dict],
    *,
    ticker: str | None = None,
    action: str = "BUY",
    rsi: Any = None,
    mom20: Any = None,
    regime_mult: Any = None,
    min_matches: int = DEFAULT_MIN_MATCHES,
    now: datetime | None = None,
) -> dict:
    """Empirical conditional return distribution for the (action, bucket) cell.

    Parameters
    ----------
    outcomes : list[dict]
        Rows from ``decision_outcomes.jsonl``.  Caller passes the full
        tail (or a deduped subset); we don't I/O.
    ticker : str | None
        Surface only — included in the response so the dashboard can label
        the card.  Does NOT filter the corpus (the whole point is that
        analogue power comes from cross-ticker pooling).
    action : str
        ``BUY`` / ``SELL`` / ``BUY_CALL`` etc.  Matched literally against
        ``outcomes[i]['action']``.  Used to sign the win predicate.
    rsi, mom20, regime_mult : float | None
        The CURRENT ticker's feature values.  If any is None, the
        corresponding bucket is None and the matcher falls back to
        "action-only" matches.
    min_matches : int
        Below this count, verdict collapses to ``INSUFFICIENT_DATA``.
    now : datetime | None
        Injected for tests; defaults to UTC now.

    Returns
    -------
    dict with keys: ``as_of``, ``ticker``, ``action``, ``current_features``,
    ``current_buckets``, ``n_outcomes``, ``n_action_only_matches``,
    ``n_matches``, ``stats``, ``trader_median_pct``, ``win_rate``,
    ``verdict``, ``headline``.
    """
    now = now or datetime.now(timezone.utc)
    action_u = (action or "").upper()
    cur_buckets = _bucketize(rsi, mom20, regime_mult)
    # If any bucket is missing (caller fed a None feature), the matcher
    # widens to "action-only matches" — but ``n_matches`` stays 0 and the
    # verdict will collapse honestly.
    have_all_buckets = all(v is not None for v in cur_buckets.values())

    # First pass: action-only matches (for the secondary headline).
    action_rows: list[dict] = []
    for row in outcomes:
        if not isinstance(row, dict):
            continue
        a = str(row.get("action") or "").upper()
        if a != action_u:
            continue
        fr = row.get("forward_return_5d")
        if not isinstance(fr, (int, float)) or fr != fr:
            continue
        action_rows.append(row)

    n_action_only = len(action_rows)

    # Second pass: full bucket match (only if all current buckets defined).
    matches: list[dict] = []
    if have_all_buckets:
        for row in action_rows:
            row_buckets = _bucketize(
                row.get("rsi"), row.get("mom20"), row.get("regime_mult"))
            if (row_buckets["rsi"] == cur_buckets["rsi"]
                    and row_buckets["mom20"] == cur_buckets["mom20"]
                    and row_buckets["regime"] == cur_buckets["regime"]):
                matches.append(row)

    returns = [float(r["forward_return_5d"]) for r in matches]
    stats = _percentiles(returns)

    n = len(matches)
    if n:
        wins = sum(1 for r in matches
                   if _is_win(action_u, float(r["forward_return_5d"])))
        win_rate = round(wins / n, 4)
    else:
        win_rate = 0.0

    # Trader-perspective median: positive ⇒ trade in this direction historically
    # pays.  For BUY: equals raw p50; for SELL: equals negated raw p50.
    trader_sign = -1.0 if action_u in _SELL_ACTIONS else 1.0
    trader_median = round(trader_sign * stats["p50"], 2)
    verdict = _verdict(trader_median, win_rate, n, min_matches)

    if n == 0:
        if not have_all_buckets:
            headline = (f"No bucket match — missing feature inputs "
                        f"(rsi/mom20/regime).  {n_action_only} action-only rows.")
        else:
            headline = (f"Zero analogues in this {action_u} cell — exotic "
                        f"setup (rsi/mom20/regime). {n_action_only} action-only "
                        f"rows.")
    elif verdict == "INSUFFICIENT_DATA":
        headline = (f"{n} analogues — too thin for a verdict (need >={min_matches}). "
                    f"Provisional median {stats['p50']:+.2f}% / win rate "
                    f"{win_rate*100:.0f}%.")
    else:
        verb = {
            "STRONG_EDGE": "STRONG EDGE",
            "EDGE": "edge",
            "NEUTRAL": "neutral",
            "HEADWIND": "headwind",
            "STRONG_HEADWIND": "STRONG HEADWIND",
        }[verdict]
        headline = (
            f"{n} analogues — median {stats['p50']:+.2f}% (trader {trader_median:+.2f}%), "
            f"win rate {win_rate*100:.0f}% — {verb} for {action_u}.")

    return {
        "as_of": now.isoformat(timespec="seconds"),
        "ticker": ticker,
        "action": action_u,
        "current_features": {
            "rsi": rsi if isinstance(rsi, (int, float)) else None,
            "mom20": mom20 if isinstance(mom20, (int, float)) else None,
            "regime_mult": regime_mult if isinstance(regime_mult, (int, float)) else None,
        },
        "current_buckets": cur_buckets,
        "min_matches": int(min_matches),
        "n_outcomes": len(outcomes),
        "n_action_only_matches": n_action_only,
        "n_matches": n,
        "stats": stats,
        "trader_median_pct": trader_median,
        "win_rate": win_rate,
        "verdict": verdict,
        "headline": headline,
    }


def _load_outcomes(path: Path | str, max_rows: int = 4000) -> list[dict]:
    """Read tail of decision_outcomes.jsonl.  Malformed lines silently skipped."""
    p = Path(path)
    if not p.exists():
        return []
    rows: list[dict] = []
    for ln in p.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            rows.append(json.loads(ln))
        except Exception:
            continue
    return rows[-max_rows:]


if __name__ == "__main__":  # smoke test against the live corpus
    here = Path(__file__).resolve().parent.parent.parent
    rows = _load_outcomes(here / "data" / "decision_outcomes.jsonl")
    # NVDA-like setup: bullish overbought
    print(json.dumps(build_setup_analogues(
        rows, ticker="NVDA", action="BUY",
        rsi=72.0, mom20=12.0, regime_mult=1.0), indent=2))
    # SOXL-like setup: leveraged drawdown
    print(json.dumps(build_setup_analogues(
        rows, ticker="SOXL", action="BUY",
        rsi=28.0, mom20=-15.0, regime_mult=0.9), indent=2))
