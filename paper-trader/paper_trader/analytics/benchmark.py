"""S&P 500 buy-and-hold benchmark — the single "is this bot worth running
versus just buying the index and doing nothing?" KPI.

Every trader's first question about an automated strategy is *not* "what is
my Sharpe" — it is "would I have more money if I'd just bought the index?".
This stack records ``sp500_price`` on **every** ``equity_curve`` write from
cycle one, so the answer is a pure arithmetic walk of one table — but until
now nothing computed it.

NOTE the figure is the ``^GSPC`` *index level* (~7400), **not** the SPY ETF
share price (~620). This module therefore says "S&P 500" / ``sp500`` /
``index`` everywhere and never "SPY" — a panel that mixes a 7400 mark with a
"$620 SPY" label loses the trader's trust. The buy-and-hold arithmetic
(notional capital × index ratio) is identical either way.

Why this is **not** a duplicate of an existing panel — do not "consolidate"
these; invariant #10 forbids re-derivation and they measure different things:

* ``/api/open-attribution`` (``open_attribution.py``) is per-**open**-
  position alpha *since each lot's entry* — blind to realised P&L and cash
  drag, and it resets its window every time a lot is re-opened (invariant
  #8). It answers "are my *current* picks beating the index since I bought
  them", not "is the *account* ahead".
* ``/api/analytics`` ``sp500_beta`` / ``sp500_correlation`` is a *statistical*
  regression needing many daily points (``null`` on the live book right
  now). It answers "how market-correlated am I", not "am I ahead in dollars".
* This is the **whole-account** number: cash + open positions + every
  realised round-trip + every unrealised mark, since the first equity write,
  versus the identical starting capital invested once in the S&P 500 at that
  same instant and held untouched. Defined from cycle 1 — no regression, no
  per-lot windowing.

Advisory only — it reports, never gates Opus, adds no caps (paper-trader
AGENTS.md invariants #2/#12; the ``self_review``/``desk_pulse``
observational precedent). Pure & network-free: a walk of the ``equity_curve``
rows the caller already read (the ``drawdown.py`` "network in the endpoint,
builder takes the dicts" split), so the core is offline & deterministically
testable. Never raises — a malformed row degrades; the contract is "no
benchmark this cycle", never an exception.
"""
from __future__ import annotations

from datetime import datetime, timezone

# Verdict is *withheld* until the book has a defensible amount of history —
# the news_edge / trade_asymmetry / decision_reliability sample-size-honesty
# precedent. A one-day-old paper book "beating the S&P by 4pp" is noise.
_MIN_SPAN_HOURS = 24.0      # < 1 calendar day of book history → INSUFFICIENT
_MIN_POINTS = 12            # …and at least this many benchmarkable points
_TRACK_BAND_PP = 0.5        # |alpha| ≤ this → TRACKING (neither beat nor lag)

_VERB = {"BEATING": "Beating", "LAGGING": "Lagging", "TRACKING": "Tracking"}


def _parse_ts(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def _usable(row: dict) -> bool:
    """Benchmarkable only if the row carries *both* a positive portfolio
    value and a positive S&P 500 mark. yfinance hiccups on a cold first
    cycle, so the inception anchor is the first such row, not blindly
    ``equity_curve[0]`` (the advisor's robustness note)."""
    try:
        tv = row.get("total_value")
        sp = row.get("sp500_price")
        return (tv is not None and float(tv) > 0
                and sp is not None and float(sp) > 0)
    except Exception:
        return False


def _external_cash_flow(prev: dict | None, cur: dict | None) -> float:
    if prev is None or cur is None:
        return 0.0
    try:
        explicit = cur.get("external_cash_flow")
        if explicit is not None:
            return float(explicit)
    except Exception:
        pass
    try:
        total_delta = (
            float(cur.get("total_value") or 0.0)
            - float(prev.get("total_value") or 0.0)
        )
        cash_delta = (
            float(cur.get("cash") or 0.0)
            - float(prev.get("cash") or 0.0)
        )
    except (TypeError, ValueError):
        return 0.0
    tolerance = max(1.0, abs(total_delta) * 0.005)
    if abs(total_delta) >= 100.0 and abs(total_delta - cash_delta) <= tolerance:
        return total_delta
    return 0.0


def build_benchmark(equity_curve: list[dict],
                    starting_equity: float = 1000.0) -> dict:
    """Whole-account return vs an equal-capital S&P 500 buy-and-hold since
    inception.

    Args:
        equity_curve: chronological (ascending) list of
            ``{timestamp, total_value, cash, sp500_price}`` — exactly
            ``store.equity_curve(...)``'s shape and order.
        starting_equity: notional invested in the index at inception. The
            endpoint passes ``store.INITIAL_CASH`` (never a literal 1000 —
            invariant #12). It is equal **by construction** to the first
            equity row's ``total_value`` (``store.py`` seeds the first write
            at ``INITIAL_CASH``); stated explicitly so a future change to
            that seeding is one visible drift point, not a silent skew.

    ``state`` ∈ ``NO_DATA`` (no row with both a value and an S&P mark) →
    ``INSUFFICIENT`` (history shorter than ``_MIN_SPAN_HOURS`` or fewer than
    ``_MIN_POINTS`` benchmarkable points — numerics still emitted, ``verdict``
    withheld) → ``OK`` (``verdict`` ∈ ``BEATING``/``LAGGING``/``TRACKING``).
    ``headline`` is the single source of truth the endpoint, CLI, Discord
    line and chat context all render, so they can never drift.
    """
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    base = {
        "as_of": now,
        "state": "NO_DATA",
        "verdict": None,
        "starting_equity": round(float(starting_equity), 2),
        "inception_ts": None,
        "current_ts": None,
        "current_value": None,
        "inception_sp500": None,
        "current_sp500": None,
        "port_return_pct": None,
        "sp500_return_pct": None,
        "alpha_pp": None,
        "sp500_equivalent_usd": None,
        "usd_vs_sp500": None,
        "capital_basis": round(float(starting_equity), 2),
        "net_external_cash_flow": 0.0,
        "pct_cycles_ahead": None,
        "best_alpha_pp": None,
        "best_alpha_ts": None,
        "worst_alpha_pp": None,
        "worst_alpha_ts": None,
        "span_hours": 0.0,
        "n_points": 0,
        "history": [],
        "headline": ("No benchmarkable equity history yet — need an equity "
                     "point carrying both a portfolio value and an S&P 500 "
                     "mark."),
    }

    try:
        rows = [r for r in (equity_curve or []) if _usable(r)]
    except Exception:
        rows = []
    if not rows:
        return base

    annotated: list[dict] = []
    basis = float(starting_equity)
    flows: list[tuple[float, float]] = []
    prev = None
    for r in rows:
        row = dict(r)
        flow = _external_cash_flow(prev, row) if prev is not None else 0.0
        basis += flow
        sp = float(row["sp500_price"])
        if flow:
            flows.append((flow, sp))
        row["_capital_basis"] = basis
        row["_external_cash_flow"] = flow
        annotated.append(row)
        prev = row

    anchor, latest = annotated[0], annotated[-1]
    sp0 = float(anchor["sp500_price"])
    sp1 = float(latest["sp500_price"])
    cur_val = float(latest["total_value"])
    init = float(starting_equity)
    current_basis = float(latest.get("_capital_basis") or init)
    net_flows = current_basis - init

    sp_equiv = init * (sp1 / sp0) if sp0 else init
    for flow, flow_sp in flows:
        if flow_sp:
            sp_equiv += flow * (sp1 / flow_sp)
        else:
            sp_equiv += flow
    port_ret = ((cur_val - current_basis) / current_basis * 100.0
                if current_basis else 0.0)
    sp_ret = ((sp_equiv - current_basis) / current_basis * 100.0
              if current_basis else 0.0)
    alpha_pp = port_ret - sp_ret
    usd_vs = cur_val - sp_equiv

    # Running alpha series (cheap, bounded): for each benchmarkable point,
    # account return − the index return the same capital would have had to
    # that instant. Surfaces "was it *ever* ahead, and the peak lead / max
    # lag" and feeds the UI a cumulative-alpha curve.
    n_ahead = 0
    best_a = best_ts = worst_a = worst_ts = None
    hist: list[dict] = []
    flow_lots: list[tuple[float, float]] = []
    for r in annotated:
        tv = float(r["total_value"])
        sp = float(r["sp500_price"])
        flow = float(r.get("_external_cash_flow") or 0.0)
        if flow:
            flow_lots.append((flow, sp))
        basis_i = float(r.get("_capital_basis") or init)
        sp_equiv_i = init * (sp / sp0) if sp0 else init
        for flow_amt, flow_sp in flow_lots:
            if flow_sp:
                sp_equiv_i += flow_amt * (sp / flow_sp)
            else:
                sp_equiv_i += flow_amt
        pr = (tv - basis_i) / basis_i * 100.0 if basis_i else 0.0
        sr = ((sp_equiv_i - basis_i) / basis_i * 100.0
              if basis_i else 0.0)
        a = pr - sr
        if a > 0:
            n_ahead += 1
        if best_a is None or a > best_a:
            best_a, best_ts = a, r.get("timestamp")
        if worst_a is None or a < worst_a:
            worst_a, worst_ts = a, r.get("timestamp")
        hist.append({"ts": r.get("timestamp"), "alpha_pp": round(a, 4)})

    n = len(rows)
    pct_ahead = n_ahead / n * 100.0 if n else 0.0

    t0 = _parse_ts(anchor.get("timestamp"))
    t1 = _parse_ts(latest.get("timestamp"))
    span_h = (t1 - t0).total_seconds() / 3600.0 if (t0 and t1) else 0.0

    # Down-sample history to ≤ 200 points (keep the response bounded; the
    # curve shape survives the stride). Ceil-stride to ≤199 strided points
    # then always pin the true latest — strictly ≤200 (the drawdown.py
    # precedent's `+ [hist[-1]]` can overshoot to 201; this cannot).
    if len(hist) > 200:
        step = -(-len(hist) // 199)  # ceil division
        sampled = hist[::step]
        if sampled[-1] is not hist[-1]:
            sampled.append(hist[-1])
        hist = sampled

    if span_h < _MIN_SPAN_HOURS or n < _MIN_POINTS:
        state, verdict = "INSUFFICIENT", None
    else:
        state = "OK"
        if alpha_pp > _TRACK_BAND_PP:
            verdict = "BEATING"
        elif alpha_pp < -_TRACK_BAND_PP:
            verdict = "LAGGING"
        else:
            verdict = "TRACKING"

    if state == "INSUFFICIENT":
        headline = (
            f"S&P 500 benchmark maturing — {span_h:.1f}h / {n} benchmarkable "
            f"points (verdict withheld until ≥{_MIN_SPAN_HOURS:.0f}h and "
            f"≥{_MIN_POINTS} points). Provisional: ${cur_val:.2f} vs "
            f"${sp_equiv:.2f} index-equivalent ({alpha_pp:+.2f}pp)."
        )
    else:
        rel = "by" if verdict != "TRACKING" else "within"
        basis_phrase = (
            f"the same ${current_basis:.2f} capital basis had been invested "
            "in the index as deposits landed"
            if abs(net_flows) >= 0.01
            else f"the same ${init:.2f} had bought the index at inception"
        )
        headline = (
            f"{_VERB[verdict]} buy-and-hold S&P 500 {rel} "
            f"{abs(alpha_pp):.2f}pp — ${cur_val:.2f} vs ${sp_equiv:.2f} if "
            f"{basis_phrase} "
            f"(${usd_vs:+.2f}); ahead in {pct_ahead:.1f}% of {n} cycles."
        )

    base.update({
        "state": state,
        "verdict": verdict,
        "inception_ts": anchor.get("timestamp"),
        "current_ts": latest.get("timestamp"),
        "current_value": round(cur_val, 2),
        "inception_sp500": round(sp0, 4),
        "current_sp500": round(sp1, 4),
        "port_return_pct": round(port_ret, 4),
        "sp500_return_pct": round(sp_ret, 4),
        "alpha_pp": round(alpha_pp, 4),
        "sp500_equivalent_usd": round(sp_equiv, 2),
        "usd_vs_sp500": round(usd_vs, 2),
        "capital_basis": round(current_basis, 2),
        "net_external_cash_flow": round(net_flows, 2),
        "pct_cycles_ahead": round(pct_ahead, 2),
        "best_alpha_pp": round(best_a, 4) if best_a is not None else None,
        "best_alpha_ts": best_ts,
        "worst_alpha_pp": round(worst_a, 4) if worst_a is not None else None,
        "worst_alpha_ts": worst_ts,
        "span_hours": round(span_h, 2),
        "n_points": n,
        "history": hist,
        "headline": headline,
    })
    return base


if __name__ == "__main__":  # one-screen answer, usable when :8090 is wedged
    import json
    import sqlite3
    import sys
    from pathlib import Path

    from paper_trader.store import INITIAL_CASH

    db = Path(__file__).resolve().parents[2] / "data" / "paper_trader.db"
    try:
        c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        rows = [
            {"timestamp": r[0], "total_value": r[1], "cash": r[2],
             "sp500_price": r[3]}
            for r in c.execute(
                "SELECT timestamp,total_value,cash,sp500_price FROM "
                "equity_curve ORDER BY timestamp ASC, id ASC").fetchall()
        ]
        c.close()
    except Exception as e:  # the desk_pulse / signals --check-freshness CLI precedent
        print(f"benchmark: cannot read {db}: {e}")
        sys.exit(2)

    rep = build_benchmark(rows, starting_equity=INITIAL_CASH)
    if "--json" in sys.argv:
        print(json.dumps(rep, indent=2, default=str))
    else:
        tag = rep["state"] + (f"/{rep['verdict']}" if rep["verdict"] else "")
        print(f"S&P 500 BENCHMARK  [{tag}]  {rep['headline']}")
        if rep["state"] != "NO_DATA":
            print(f"  account  ${rep['current_value']}  "
                  f"({rep['port_return_pct']:+}%)")
            print(f"  S&P 500  ${rep['sp500_equivalent_usd']}  "
                  f"({rep['sp500_return_pct']:+}%)  "
                  f"alpha {rep['alpha_pp']:+}pp  (${rep['usd_vs_sp500']:+})")
            print(f"  best lead {rep['best_alpha_pp']:+}pp @ "
                  f"{rep['best_alpha_ts']}")
            print(f"  worst lag {rep['worst_alpha_pp']:+}pp @ "
                  f"{rep['worst_alpha_ts']}")
