"""BUY-ticker quant-coverage audit — read-only.

The natural QUANT QUESTION this answers: *for how many BUY decisions in the
training corpus was the picked ticker OUTSIDE the cycle's pre-fetched
quant-signal set, and what fraction of those rows nevertheless carry real
training-time RSI/MACD?* That gap is the training/inference feature-space
mismatch a sentiment-only buy_ticker introduces:

``_ml_decide`` pre-fetches quant signals only for
``QUANT_SIGNAL_TICKERS ∪ portfolio.positions`` (~34 of 117 watchlist
tickers). A sentiment-only BUY of a ticker outside that set — historically
~21% of all BUYs in the live ``decision_outcomes.jsonl`` tail; typical
gap tickers: XLF/XLV/XLI sector ETFs, ARKK, BTC-USD, leveraged
single-stock 2x names (NVDU / AMZU / METAU / CONL) — used to feed the
DecisionScorer ``build_features`` neutral defaults (rsi=50, macd=0,
mom=0, bb=0) at inference while ``_compute_decision_outcomes`` calls
``_get_quant_signals(sim_date, [ticker], …)`` per outcome with no
WATCHLIST subset, so the **training** row carries the REAL indicators.
Net: the scorer trained on a feature manifold the gate never visited at
inference, and predicted at a manifold the model never saw in training.

The fix (commit ``9268ee0``) lazily computes ``_get_quant_signals`` for
the picked ticker on miss so the scorer feed matches the training feature
vector. NEW decisions made after the fix have parity; OLD rows in the
corpus carry the historical drift. This diagnostic answers — durably,
queryably — exactly how big that historical drift was and which tickers
dominated it.

Same operational discipline as ``paper_trader/ml/feature_coverage.py`` /
``calibration.py`` / ``skill_trend.py``: read-only, never trains, never
touches ``decision_scorer.pkl`` / ``decision_outcomes.jsonl`` (read-only)
/ ``build_features`` / ``N_FEATURES`` / any trade path. Safe under the
live unattended continuous loop. Never raises on bad input.

```bash
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.buy_ticker_quant_coverage
```
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

MIN_ROWS = 30
# Verdict thresholds — chosen against the live corpus (gap_fraction ≈ 0.21,
# gap_rsi_real_fraction ≈ 0.9 at the time of the fix). HEALTHY ⇒ negligible
# drift; GAP_PRESENT ⇒ material but bounded; GAP_DOMINANT ⇒ the scorer's
# OOS skill is being computed against a noticeably-different training
# manifold than the gate visits.
GAP_HEALTHY_FLOOR = 0.05
GAP_DOMINANT_CEIL = 0.30
TOP_TICKERS_IN_REPORT = 15


def _quant_signal_tickers() -> frozenset:
    """Frozen snapshot of the cycle's pre-fetched quant set. Imports late
    (mirrors feature_coverage's lazy build_features import) so a module
    refactor in backtest.py cannot silently break this diagnostic's import
    chain — the test discipline `decision_scorer ← feature_coverage` follows.
    """
    try:
        from paper_trader.backtest import QUANT_SIGNAL_TICKERS
        return frozenset(QUANT_SIGNAL_TICKERS)
    except Exception:
        return frozenset()


def load_outcomes(path: Path | str) -> list[dict]:
    """Robust JSONL load of decision_outcomes.jsonl. Skips unparseable lines.

    Never raises — a missing/corrupt file yields ``[]`` so callers degrade
    to ``INSUFFICIENT_DATA`` rather than crashing (the producer is best-effort
    by construction; a reader of it must be too — the ``skill_trend``
    precedent).
    """
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


def analyze(outcomes_path: Path | str = None, tail: int = 5000) -> dict:
    """Audit BUY rows in ``decision_outcomes.jsonl`` for the training/
    inference feature-parity gap.

    Args:
        outcomes_path: path to the JSONL (defaults to
            ``data/decision_outcomes.jsonl`` next to the repo).
        tail: only the last N rows are analyzed (the trainer caps at
            ``MAX_OUTCOMES_FOR_TRAINING=5000`` so this matches the trained
            scorer's actual exposure; a higher tail would mix in rows the
            model never saw).

    Returns a dict::

      {
        "status": "ok" | "error",
        "verdict": "HEALTHY"
                 | "GAP_PRESENT"
                 | "GAP_DOMINANT"
                 | "INSUFFICIENT_DATA",
        "n_total_buys": int,
        "n_quant_covered": int,         # ticker ∈ QUANT_SIGNAL_TICKERS
        "n_quant_gap": int,             # ticker ∉ QUANT_SIGNAL_TICKERS
        "gap_fraction": float,          # n_quant_gap / n_total_buys
        "gap_rsi_real_count": int,      # gap rows with rsi != None
        "gap_rsi_real_fraction": float, # fraction of gap rows w/ real RSI
                                        # (this is the "drift severity":
                                        # 1.0 ⇒ every gap row had real
                                        # training-time indicators but the
                                        # inference path saw defaults)
        "top_gap_tickers": [{"ticker": str, "count": int,
                             "rsi_real_count": int}],
        "quant_signal_tickers": int,    # size of pre-fetched set
      }
    """
    if outcomes_path is None:
        outcomes_path = Path(__file__).resolve().parent.parent.parent \
            / "data" / "decision_outcomes.jsonl"

    rows = load_outcomes(outcomes_path)
    if tail and len(rows) > tail:
        rows = rows[-tail:]

    qs = _quant_signal_tickers()
    if not qs:
        return {
            "status": "error",
            "verdict": "INSUFFICIENT_DATA",
            "n_total_buys": 0,
            "hint": "QUANT_SIGNAL_TICKERS unavailable from backtest module",
        }

    n_total = 0
    n_covered = 0
    n_gap = 0
    n_gap_rsi_real = 0
    gap_counter: Counter = Counter()
    gap_rsi_real_counter: Counter = Counter()

    for r in rows:
        action = str(r.get("action") or "").upper()
        if action != "BUY":
            continue
        ticker = str(r.get("ticker") or "").upper()
        if not ticker:
            continue
        n_total += 1
        rsi_real = r.get("rsi") is not None
        if ticker in qs:
            n_covered += 1
        else:
            n_gap += 1
            gap_counter[ticker] += 1
            if rsi_real:
                n_gap_rsi_real += 1
                gap_rsi_real_counter[ticker] += 1

    if n_total < MIN_ROWS:
        return {
            "status": "ok",
            "verdict": "INSUFFICIENT_DATA",
            "n_total_buys": n_total,
            "n_quant_covered": n_covered,
            "n_quant_gap": n_gap,
            "gap_fraction": None,
            "gap_rsi_real_count": n_gap_rsi_real,
            "gap_rsi_real_fraction": None,
            "top_gap_tickers": [],
            "quant_signal_tickers": len(qs),
        }

    gap_fraction = n_gap / n_total
    gap_rsi_real_fraction = (n_gap_rsi_real / n_gap) if n_gap else 0.0

    if gap_fraction < GAP_HEALTHY_FLOOR:
        verdict = "HEALTHY"
    elif gap_fraction >= GAP_DOMINANT_CEIL or (
        gap_fraction >= GAP_HEALTHY_FLOOR
        and gap_rsi_real_fraction >= 0.80
        and gap_fraction >= 0.15
    ):
        verdict = "GAP_DOMINANT"
    else:
        verdict = "GAP_PRESENT"

    top = []
    for ticker, count in gap_counter.most_common(TOP_TICKERS_IN_REPORT):
        top.append({
            "ticker": ticker,
            "count": count,
            "rsi_real_count": gap_rsi_real_counter.get(ticker, 0),
        })

    return {
        "status": "ok",
        "verdict": verdict,
        "n_total_buys": n_total,
        "n_quant_covered": n_covered,
        "n_quant_gap": n_gap,
        "gap_fraction": round(gap_fraction, 4),
        "gap_rsi_real_count": n_gap_rsi_real,
        "gap_rsi_real_fraction": round(gap_rsi_real_fraction, 4),
        "top_gap_tickers": top,
        "quant_signal_tickers": len(qs),
    }


def _print_report(rep: dict) -> None:
    print(f"[buy_ticker_quant_coverage]  verdict={rep.get('verdict')}  "
          f"status={rep.get('status')}")
    print(f"  total BUYs analyzed: {rep.get('n_total_buys', 0)}")
    print(f"  quant-covered:       {rep.get('n_quant_covered', 0)}")
    print(f"  quant-gap:           {rep.get('n_quant_gap', 0)}"
          f"  (fraction={rep.get('gap_fraction')})")
    print(f"  gap rows with real RSI: {rep.get('gap_rsi_real_count', 0)}"
          f"  (fraction={rep.get('gap_rsi_real_fraction')})")
    print(f"  pre-fetched quant tickers: "
          f"{rep.get('quant_signal_tickers', 0)}")
    if rep.get("hint"):
        print(f"  hint: {rep['hint']}")
    rows = rep.get("top_gap_tickers") or []
    if rows:
        print(f"  top {len(rows)} gap tickers:")
        print(f"    {'ticker':<10}{'count':>8}{'rsi_real':>12}")
        for r in rows:
            print(f"    {r['ticker']:<10}{r['count']:>8}"
                  f"{r['rsi_real_count']:>12}")


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="python3 -m paper_trader.ml.buy_ticker_quant_coverage",
        description="Audit training/inference feature-parity for BUY rows "
                    "in decision_outcomes.jsonl. Read-only; never trains.",
    )
    p.add_argument("--outcomes", default=None,
                   help="Path to decision_outcomes.jsonl (default: "
                        "data/decision_outcomes.jsonl)")
    p.add_argument("--tail", type=int, default=5000,
                   help="Only analyze the last N rows (default 5000, "
                        "matching the trainer's MAX_OUTCOMES_FOR_TRAINING).")
    p.add_argument("--json", action="store_true",
                   help="Emit machine-readable JSON instead of a table.")
    args = p.parse_args(argv)

    rep = analyze(args.outcomes, tail=args.tail)
    if args.json:
        print(json.dumps(rep, indent=2, sort_keys=True))
    else:
        _print_report(rep)
    return 0 if rep.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
