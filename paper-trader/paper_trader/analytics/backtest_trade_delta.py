"""Trade-level diff between two backtest runs.

The existing ``/api/backtests/compare`` returns side-by-side **aggregate**
summaries (total return %, vs-SPY %, max DD, n_trades, n_decisions, win
rate, normalized equity curves) for 2–4 runs. It answers the obvious
question — "are these runs equivalent at the bottom line?" — but it
deliberately does NOT answer the trader's *follow-up*:

  "Run 105 is +5pp better than Run 100. **Which trades** drove the
  delta? Did 105 pick something 100 didn't, or did 100 trade
  something 105 avoided?"

A blind operator looking at `/api/backtests/compare` today has to fetch
``/api/backtests/<id>/trades`` for each run separately and do the diff
by hand — a structural read that every quant doing model ablation
performs on every cycle. This module makes it one endpoint.

``build_trade_delta(detail_a, detail_b)`` consumes the exact shape
``BacktestStore.run_detail`` already returns (the same shape
``/api/backtests/compare`` consumes), keys trades on
``(ticker, action, sim_date)`` (the natural identity of a backtest
trade — date is canonical from the persisted ``backtest_trades`` rows),
and emits:

  * ``only_in_a`` — trades A executed and B did not (at the same
    ticker/action/date)
  * ``only_in_b`` — symmetric counterpart
  * ``common`` — trades both runs executed; carries qty/price/value
    deltas so a near-miss on size/timing is visible
  * ``n_only_a`` / ``n_only_b`` / ``n_common`` / ``n_total_a`` /
    ``n_total_b`` — divergence counts
  * ``divergence_score`` — symmetric-difference / union (the canonical
    Jaccard distance over the trade sets; 0.0 = identical, 1.0 = fully
    disjoint, in [0.0, 1.0])
  * ``return_delta_pct`` — B's ``total_return_pct`` minus A's
  * ``attribution`` — per-ticker estimate of unique-to-B vs unique-to-A
    realized P/L using the same FIFO BUY→SELL pairing the existing
    ``backtest_compare`` win-rate computation uses (the SSOT — keep the
    two algorithms identical so callers reading both endpoints get
    consistent ticker-level dollar attributions)

Garbage-safe — non-dict inputs, missing trades, malformed numeric
fields all return a well-formed skeleton. Pure builder, no DB calls;
the endpoint wires it via ``BacktestStore.run_detail`` and the same
``try/except → ERROR envelope`` convention every other adjacent
analytics endpoint uses (see ``/api/backtests/compare`` itself).
"""

from __future__ import annotations

from typing import Any


def _norm(t: Any) -> tuple | None:
    """Return the canonical match-key for a trade row, or ``None`` if
    the row lacks a usable key. Match identity is (ticker upper,
    action upper, sim_date string) — the natural backtest identity.
    """
    if not isinstance(t, dict):
        return None
    tk = t.get("ticker")
    act = t.get("action")
    sd = t.get("sim_date")
    if not isinstance(tk, str) or not tk:
        return None
    if not isinstance(act, str) or not act:
        return None
    if not isinstance(sd, str) or not sd:
        return None
    return (tk.upper(), act.upper(), sd)


def _f(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    if v != v:  # NaN guard
        return 0.0
    return v


def _fifo_realized_pl(trades: list[dict]) -> dict[str, float]:
    """FIFO BUY → SELL pairing, returning a per-ticker realized P/L
    estimate in dollars. Mirrors the win-rate computation in
    ``dashboard.backtest_compare`` so the two endpoints can never
    silently disagree on ticker-level dollar attribution. Stock-only
    (BUY / SELL); option / leverage actions are skipped so the metric
    stays interpretable.
    """
    held: dict[str, list[tuple[float, float]]] = {}
    pl: dict[str, float] = {}
    for t in trades:
        if not isinstance(t, dict):
            continue
        act = (t.get("action") or "").upper()
        tk = t.get("ticker") or ""
        qty = _f(t.get("qty"))
        px = _f(t.get("price"))
        if not tk or qty <= 0 or px <= 0:
            continue
        if act == "BUY":
            held.setdefault(tk, []).append((qty, px))
        elif act == "SELL":
            lots = held.get(tk) or []
            remaining = qty
            ticker_pl = 0.0
            while remaining > 0 and lots:
                lot_qty, lot_px = lots[0]
                use = min(lot_qty, remaining)
                ticker_pl += (px - lot_px) * use
                if use >= lot_qty:
                    lots.pop(0)
                else:
                    lots[0] = (lot_qty - use, lot_px)
                remaining -= use
            held[tk] = lots
            pl[tk] = pl.get(tk, 0.0) + ticker_pl
    return pl


def _summary(detail: dict | None) -> dict:
    if not isinstance(detail, dict):
        return {"run_id": None, "error": "not found"}
    return {
        "run_id": detail.get("run_id"),
        "start_date": detail.get("start_date"),
        "end_date": detail.get("end_date"),
        "status": detail.get("status"),
        "total_return_pct": detail.get("total_return_pct"),
        "spy_return_pct": detail.get("spy_return_pct"),
        "vs_spy_pct": detail.get("vs_spy_pct"),
        "n_trades": detail.get("n_trades"),
        "n_decisions": detail.get("n_decisions"),
        "final_value": detail.get("final_value"),
    }


def _attribution_rows(only_a: list[dict],
                      only_b: list[dict]) -> list[dict]:
    """Per-ticker realized P/L attribution for trades unique to each
    side, sorted by absolute dollar magnitude descending. Top contributors
    surface the trades most responsible for the run-vs-run divergence;
    a ticker that BOTH sides traded but at different sizes will already
    have been flagged in the ``common`` diff and is excluded here.
    """
    pl_a = _fifo_realized_pl(only_a)
    pl_b = _fifo_realized_pl(only_b)
    rows: list[dict] = []
    tickers = sorted(set(pl_a) | set(pl_b))
    for tk in tickers:
        a = pl_a.get(tk, 0.0)
        b = pl_b.get(tk, 0.0)
        rows.append({
            "ticker": tk,
            "only_a_pl_usd": round(a, 2),
            "only_b_pl_usd": round(b, 2),
            "delta_pl_usd": round(b - a, 2),
        })
    rows.sort(key=lambda r: -abs(r["delta_pl_usd"]))
    return rows


def build_trade_delta(detail_a: Any, detail_b: Any) -> dict:
    """Pure: roll two ``BacktestStore.run_detail`` outputs into a
    trade-level diff. Garbage-safe — missing fields, bad types,
    ``None`` inputs all degrade to a well-formed envelope, never
    raise.
    """
    a_sum = _summary(detail_a)
    b_sum = _summary(detail_b)

    trades_a = (
        detail_a.get("trades") if isinstance(detail_a, dict) else None
    ) or []
    trades_b = (
        detail_b.get("trades") if isinstance(detail_b, dict) else None
    ) or []
    if not isinstance(trades_a, list):
        trades_a = []
    if not isinstance(trades_b, list):
        trades_b = []

    keyed_a: dict[tuple, list[dict]] = {}
    keyed_b: dict[tuple, list[dict]] = {}
    skipped_a = 0
    skipped_b = 0
    for t in trades_a:
        k = _norm(t)
        if k is None:
            skipped_a += 1
            continue
        keyed_a.setdefault(k, []).append(t)
    for t in trades_b:
        k = _norm(t)
        if k is None:
            skipped_b += 1
            continue
        keyed_b.setdefault(k, []).append(t)

    only_a: list[dict] = []
    only_b: list[dict] = []
    common: list[dict] = []

    all_keys = set(keyed_a) | set(keyed_b)
    for k in all_keys:
        rows_a = keyed_a.get(k, [])
        rows_b = keyed_b.get(k, [])
        # Pair off matched rows in order; surplus on either side goes to
        # the appropriate only_* bucket. A run that did "BUY NVDA on
        # 2026-04-01" twice when the other did it once leaves one row in
        # common and one in only_*.
        n_paired = min(len(rows_a), len(rows_b))
        for i in range(n_paired):
            ra, rb = rows_a[i], rows_b[i]
            common.append({
                "ticker": k[0],
                "action": k[1],
                "sim_date": k[2],
                "a_qty": _f(ra.get("qty")),
                "b_qty": _f(rb.get("qty")),
                "a_price": _f(ra.get("price")),
                "b_price": _f(rb.get("price")),
                "qty_delta": _f(rb.get("qty")) - _f(ra.get("qty")),
                "value_delta": _f(rb.get("value")) - _f(ra.get("value")),
            })
        for r in rows_a[n_paired:]:
            only_a.append({
                "ticker": k[0],
                "action": k[1],
                "sim_date": k[2],
                "qty": _f(r.get("qty")),
                "price": _f(r.get("price")),
                "value": _f(r.get("value")),
                "reason": r.get("reason"),
            })
        for r in rows_b[n_paired:]:
            only_b.append({
                "ticker": k[0],
                "action": k[1],
                "sim_date": k[2],
                "qty": _f(r.get("qty")),
                "price": _f(r.get("price")),
                "value": _f(r.get("value")),
                "reason": r.get("reason"),
            })

    only_a.sort(key=lambda r: (r["sim_date"], r["ticker"], r["action"]))
    only_b.sort(key=lambda r: (r["sim_date"], r["ticker"], r["action"]))
    common.sort(key=lambda r: (r["sim_date"], r["ticker"], r["action"]))

    n_only_a = len(only_a)
    n_only_b = len(only_b)
    n_common = len(common)
    union = n_only_a + n_only_b + n_common
    divergence_score = (
        (n_only_a + n_only_b) / union if union > 0 else 0.0
    )

    a_ret = a_sum.get("total_return_pct")
    b_ret = b_sum.get("total_return_pct")
    if isinstance(a_ret, (int, float)) and isinstance(b_ret, (int, float)):
        return_delta_pct = round(float(b_ret) - float(a_ret), 4)
    else:
        return_delta_pct = None

    attribution = _attribution_rows(only_a, only_b)

    # Headline: a one-line operator-facing summary that surfaces the
    # divergence + the top-attribution ticker, mirroring how
    # `deployment_plan_conflicts` / `slate_news_corroboration` build a
    # `headline` for the panel render.
    if union == 0:
        headline = (
            f"Run {a_sum.get('run_id')} vs {b_sum.get('run_id')}: "
            "no trades on either side."
        )
    elif n_only_a == 0 and n_only_b == 0:
        headline = (
            f"Run {a_sum.get('run_id')} vs {b_sum.get('run_id')}: "
            f"{n_common} identical trades, divergence_score=0.000."
        )
    else:
        top = attribution[0] if attribution else None
        if top is not None:
            headline = (
                f"Run {a_sum.get('run_id')} vs {b_sum.get('run_id')}: "
                f"{n_only_a} A-only + {n_only_b} B-only ({n_common} shared), "
                f"divergence_score={round(divergence_score, 3)}; "
                f"top ticker {top['ticker']} "
                f"delta_pl=${top['delta_pl_usd']}."
            )
        else:
            headline = (
                f"Run {a_sum.get('run_id')} vs {b_sum.get('run_id')}: "
                f"{n_only_a} A-only + {n_only_b} B-only ({n_common} shared), "
                f"divergence_score={round(divergence_score, 3)}."
            )

    return {
        "a": a_sum,
        "b": b_sum,
        "delta": {
            "only_in_a": only_a,
            "only_in_b": only_b,
            "common": common,
            "n_only_a": n_only_a,
            "n_only_b": n_only_b,
            "n_common": n_common,
            "n_total_a": n_only_a + n_common,
            "n_total_b": n_only_b + n_common,
            "n_skipped_a": skipped_a,
            "n_skipped_b": skipped_b,
            "divergence_score": round(divergence_score, 4),
            "return_delta_pct": return_delta_pct,
            "attribution": attribution,
        },
        "headline": headline,
    }
