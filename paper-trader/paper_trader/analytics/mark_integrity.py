"""Mark integrity — how much of the displayed book value is *fictional*
right now.

When yfinance returns nothing for a held name, ``strategy._mark_to_market``
falls back to ``avg_cost`` and flags the row ``stale_mark=True`` (the live
2026-05-17 pathology: ``MU 0.5 @ 724.12``, ``current_price == avg_cost``,
``P/L $0.00`` — indistinguishable from a genuinely flat position). That
flag is already surfaced *per position* to Opus and Discord. What no panel
answers is the **aggregate**: *what share of total book value is marked at
cost, so every P/L number on every dashboard is partially false?* A book
that is 60% stale-marked makes ``/api/analytics`` Sharpe, ``/api/drawdown``,
the equity curve and the headline P&L all quietly fictional, with nothing
saying so in one number.

``build_mark_integrity`` is a **pure** roll-up over the already-marked
position rows (the ``thesis_drift``/``correlation`` "builder takes the
dicts" split — the read-only snapshot lives in the endpoint). It mints no
new opinion and gates nothing: advisory only, never injected into
``decide()``, no caps (paper-trader AGENTS.md invariants #2/#12). It never
raises — garbage rows degrade to zero value, never an exception (the
behavioural-builder ``_safe`` contract).

Verdict ladder (sample-size honest, like the rest of the desk):

* ``NO_DATA``      — no open positions, nothing to mark.
* ``CLEAN``        — every mark is live.
* ``DEGRADED``     — some marks are at cost but < ``UNTRUSTWORTHY_PCT`` of
  gross value (or gross is zero so the share is unquantifiable but stale
  marks exist).
* ``UNTRUSTWORTHY``— >= ``UNTRUSTWORTHY_PCT`` of gross book value is marked
  at cost: treat every displayed P/L as substantially fictional until the
  price feed recovers / the runner is restarted.
"""
from __future__ import annotations

#: Stale share of gross book value at or above which the displayed P/L is
#: considered substantially fictional. Inclusive boundary.
UNTRUSTWORTHY_PCT = 50.0


def _row_value(p: dict) -> float:
    """Best-effort absolute market value of a position row. Prefers the
    enriched ``market_value`` written by ``strategy._mark_to_market``;
    falls back to ``current_price * qty * mult``; never raises."""
    mv = p.get("market_value")
    if mv is not None:
        try:
            return abs(float(mv))
        except (TypeError, ValueError):
            pass
    try:
        cur = float(p.get("current_price") or 0.0)
        qty = float(p.get("qty") or 0.0)
        mult = 100 if p.get("type") in ("call", "put") else 1
        return abs(cur * qty * mult)
    except (TypeError, ValueError):
        return 0.0


def build_mark_integrity(positions: list[dict] | None) -> dict:
    """Aggregate stale-mark coverage over the (already marked) open
    positions. Pure, never raises (AGENTS.md #2/#12 — advisory only)."""
    rows = list(positions or [])
    n = len(rows)
    if n == 0:
        return {
            "verdict": "NO_DATA",
            "headline": "No open positions — nothing to mark.",
            "n_positions": 0,
            "n_stale": 0,
            "gross_value_usd": 0.0,
            "stale_value_usd": 0.0,
            "stale_value_pct": None,
            "stale_tickers": [],
            "positions": [],
        }

    gross = 0.0
    stale_value = 0.0
    stale_tickers: list[str] = []
    out_rows: list[dict] = []
    for p in rows:
        val = _row_value(p)
        is_stale = bool(p.get("stale_mark"))
        gross += val
        if is_stale:
            stale_value += val
            tk = p.get("ticker")
            if tk:
                stale_tickers.append(tk)
        out_rows.append({
            "ticker": p.get("ticker"),
            "type": p.get("type"),
            "qty": p.get("qty"),
            "avg_cost": p.get("avg_cost"),
            "current_price": p.get("current_price"),
            "market_value": round(val, 2),
            "stale_mark": is_stale,
        })

    n_stale = sum(1 for p in rows if bool(p.get("stale_mark")))
    pct = round(stale_value / gross * 100, 2) if gross > 0 else None

    if n_stale == 0:
        verdict = "CLEAN"
        headline = f"All {n} mark{'' if n == 1 else 's'} live."
    elif pct is not None and pct >= UNTRUSTWORTHY_PCT:
        verdict = "UNTRUSTWORTHY"
        headline = (
            f"{pct}% of ${round(gross, 2)} book value marked at cost "
            f"({n_stale}/{n}) — every displayed P/L is substantially "
            f"fictional; restart the runner / refresh the price feed."
        )
    else:
        verdict = "DEGRADED"
        share = (f"{pct}% of ${round(gross, 2)} book value"
                 if pct is not None
                 else "share unquantifiable (zero gross value)")
        headline = (
            f"{n_stale}/{n} position{'' if n_stale == 1 else 's'} marked at "
            f"cost — {share} is stale; P/L partially fictional."
        )

    return {
        "verdict": verdict,
        "headline": headline,
        "n_positions": n,
        "n_stale": n_stale,
        "gross_value_usd": round(gross, 2),
        "stale_value_usd": round(stale_value, 2),
        "stale_value_pct": pct,
        "stale_tickers": stale_tickers,
        "positions": out_rows,
    }
