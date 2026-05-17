"""Desk pulse — the single pure-DB "is the desk OK right now?" digest.

The operator's literal first question every session is four facts at once:

* **Money** — equity, today-anchored return, realised win-rate / profit
  factor, and how concentrated the open book is (the documented live
  pathology: ~60%+ in one name).
* **Liveness** — is ``paper_trader.runner`` still cycling, or wedged/dead?
* **Integrity** — is the live process running *stale code* (committed fixes
  inert until restart — the chronic pathology of this stack)?
* **The one thing wrong** — of the ~24 behavioural builders, which single
  highest-precedence flag should I look at first?

Today that needs four+ panels. ``/api/scorecard`` is behavioural-only (no
money KPIs). ``/api/state`` is the heavy everything-dump that is the slowest
endpoint on the box (SWR-wrapped; cold path seconds). ``/api/command-center``
lives on the *unified* dashboard (:8888) and gets its trader half by
**cross-fetching** :8090 — so it goes blank exactly when :8090 is the thing
that is slow or wedged (observed live, 2026-05-17). None of them is a single,
fast, dependency-free answer.

``build_desk_pulse`` **mints no new opinion** (paper-trader AGENTS.md
invariants #2/#12; the ``trader_scorecard`` "router, not a grader" precedent
it follows). It composes only the **network-free, pure, DB-read-only**
single-source-of-truth builders verbatim (invariant #10):

* ``round_trips.build_round_trips`` — the canonical realised-P&L /
  win-rate / profit-factor aggregation ``/api/analytics`` itself consumes
  (same strict ``> 0`` win split, so the two can never drift).
* ``runner_heartbeat.build_runner_heartbeat`` — loop-liveness verdict.
* ``trader_scorecard.build_trader_scorecard`` — its ``focus`` (the single
  highest-precedence behavioural flag) and ``state`` are forwarded verbatim.

It adds **no yfinance, no articles.db, no scorer** call. Concentration is
computed from the position rows' already-stored ``current_price`` /
``avg_cost`` × ``qty`` (the exact ``/api/correlation`` ``market_value``
recipe, minus the price-history fetch) so the whole digest is a handful of
SQLite reads — it stays sub-50ms and answers even when every yfinance-backed
panel on the dashboard is timing out. The ``__main__`` CLI opens
``paper_trader.db`` and prints the digest from a terminal, so the operator
still gets the answer when the :8090 process itself is wedged (the
``signals.py --check-freshness`` precedent).

The top-level ``state`` is a **router** over the constituents' own verdicts
(operational concerns above behavioural, same documented-precedence idea as
``trader_scorecard._FOCUS_ORDER``): a dead loop dominates a stale SHA
dominates a behavioural flag dominates a lagging loop. It forwards the chosen
axis's own headline verbatim — no synthesized grade, directive, or cap.
Advisory only; it has no path to ``_execute()`` and is **not** injected into
the live decision prompt (dashboard/chat/CLI only). Never raises — a failing
constituent degrades that block to ``None``/``ERROR``, never an exception
(the ``trader_scorecard`` ``_safe`` contract: "no pulse this cycle", never a
500 that takes the lifeline panel down).
"""
from __future__ import annotations

from datetime import datetime, timezone

from .round_trips import build_round_trips
from .runner_heartbeat import build_runner_heartbeat
from .trader_scorecard import build_trader_scorecard

# Default starting equity. The dashboard endpoint passes ``store.INITIAL_CASH``
# explicitly (AGENTS.md invariant #12 — one source of truth for the $1000
# baseline; a literal here would silently desync the moment INITIAL_CASH
# moves). This default exists only so the pure builder is callable in
# isolation; the live path never relies on it.
_DEFAULT_INITIAL_CASH = 1000.0


def _num(x) -> float | None:
    """Coerce to float or None — never raise on a NULL/garbage DB cell."""
    try:
        if x is None:
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def _safe(fn, *args, **kwargs) -> dict:
    """Run one constituent builder; on any failure return a typed marker
    instead of letting it sink the whole digest (the trader_scorecard
    precedent — the failure mode is 'that block missing', never an
    exception)."""
    try:
        out = fn(*args, **kwargs)
        return out if isinstance(out, dict) else {}
    except Exception as e:  # pragma: no cover - exercised via monkeypatch
        return {"state": "ERROR", "verdict": "ERROR",
                "error": f"{type(e).__name__}: {e}"}


def _money_block(portfolio: dict,
                  positions: list[dict],
                  trades: list[dict],
                  initial_cash: float) -> dict:
    """Equity, return %, and the canonical realised round-trip metrics.

    ``trades`` is store-native **newest-first** (``store.recent_trades()``);
    ``build_round_trips`` wants oldest→newest, so it is reversed here exactly
    as ``/api/analytics`` / ``/api/churn`` do. The win/loss split is the
    identical strict ``> 0`` / ``<= 0`` rule as ``analytics_api`` so this
    digest can never disagree with ``/api/analytics`` (single source of
    truth, invariant #10)."""
    equity = _num((portfolio or {}).get("total_value"))
    cash = _num((portfolio or {}).get("cash"))
    base = initial_cash if initial_cash else _DEFAULT_INITIAL_CASH
    total_return_pct = (round((equity - base) / base * 100, 2)
                        if equity is not None and base else None)

    rts = build_round_trips(list(reversed(trades or [])))
    pnls = [rt["pnl_usd"] for rt in rts]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    n = len(pnls)
    win_rate_pct = round(len(wins) / n * 100, 2) if n else None
    gross_loss = abs(sum(losses))
    profit_factor = (round(sum(wins) / gross_loss, 2)
                     if gross_loss > 1e-9 else None)
    realized_pl_usd = round(sum(pnls), 2) if n else 0.0

    # Open-book unrealised P&L — straight sum of the stored marks. A NULL
    # mark contributes 0 (not an error); if *every* position is unmarked the
    # value is a meaningful 0.0, not None.
    unreal = 0.0
    for p in positions or []:
        v = _num(p.get("unrealized_pl"))
        if v is not None:
            unreal += v

    return {
        "equity_usd": round(equity, 2) if equity is not None else None,
        "cash_usd": round(cash, 2) if cash is not None else None,
        "total_return_pct": total_return_pct,
        "realized_pl_usd": realized_pl_usd,
        "unrealized_pl_usd": round(unreal, 2),
        "n_round_trips": n,
        "win_rate_pct": win_rate_pct,
        "profit_factor": profit_factor,
    }


def _concentration_block(positions: list[dict]) -> dict:
    """Top-name weight + open-name count from stored marks only.

    The exact ``/api/correlation`` ``market_value`` recipe
    (``(current_price or avg_cost) * qty * (100 if option)``) minus the
    yfinance price-history fetch — so it is pure DB and never blocks. This
    is the concentration KPI ``/api/scorecard`` deliberately omits and the
    documented live pathology (60%+ one name). ``None`` when the book is
    empty / has no positive market value (undefined, not faked)."""
    weights: list[tuple[str, float]] = []
    total = 0.0
    for p in positions or []:
        ptype = p.get("type")
        is_opt = ptype in ("call", "put")
        mult = 100 if is_opt else 1
        price = _num(p.get("current_price")) or _num(p.get("avg_cost")) or 0.0
        qty = _num(p.get("qty")) or 0.0
        mv = price * qty * mult
        if mv > 0:
            weights.append((p.get("ticker") or "?", mv))
            total += mv

    if not weights or total <= 0:
        return {"n_open_positions": len(positions or []),
                "top_name": None, "top_weight_pct": None,
                "gross_exposure_usd": round(total, 2)}

    weights.sort(key=lambda w: -w[1])
    top_name, top_mv = weights[0]
    return {
        "n_open_positions": len(positions or []),
        "top_name": top_name,
        "top_weight_pct": round(top_mv / total * 100, 2),
        "gross_exposure_usd": round(total, 2),
    }


def build_desk_pulse(portfolio: dict,
                     positions: list[dict],
                     trades: list[dict],
                     decisions: list[dict],
                     equity_curve: list[dict],
                     build_info: dict | None = None,
                     market_open: bool = False,
                     initial_cash: float = _DEFAULT_INITIAL_CASH,
                     now: datetime | None = None) -> dict:
    """Compose the money / liveness / integrity / focus digest. Pure; never
    raises.

    Inputs are store-native (``get_portfolio`` / ``open_positions`` /
    ``recent_trades`` / ``recent_decisions`` / ``equity_curve``); the caller
    owns the I/O and supplies ``build_info`` (the ``/api/build-info``
    ``{stale, behind, ...}`` dict) and ``market_open`` (the
    ``runner_heartbeat`` cadence selector) — the ``thesis_drift`` "network in
    the endpoint, builder takes the dicts" split, so the builder stays a pure
    offline-testable leaf."""
    now = now or datetime.now(timezone.utc)
    bi = build_info if isinstance(build_info, dict) else {}

    money = _money_block(portfolio or {}, positions or [], trades or [],
                         initial_cash)
    concentration = _concentration_block(positions or [])

    decs = decisions or []
    last_decision_ts = decs[0].get("timestamp") if decs else None
    heartbeat = _safe(build_runner_heartbeat, last_decision_ts,
                      bool(market_open), now=now)

    scorecard = _safe(build_trader_scorecard, portfolio or {},
                       positions or [], trades or [], decisions or [],
                       equity_curve or [], now=now)
    focus = scorecard.get("focus")
    scorecard_state = scorecard.get("state")

    # Honesty: distinguish "told it's current" from "never checked". The CLI
    # passes build_info=None (no git-SHA context) — claiming "code current"
    # there would be optimistic-when-unknown, which this stack rejects
    # everywhere (the honest-None / sample-size-honesty discipline). UNKNOWN
    # also never triggers the CODE_STALE router branch (we can't assert a
    # problem we didn't check).
    integrity_known = build_info is not None and "stale" in bi
    code_stale = bool(bi.get("stale")) if integrity_known else False
    commits_behind = bi.get("behind")
    integrity = {
        "status": ("STALE" if code_stale
                   else "CURRENT" if integrity_known else "UNKNOWN"),
        "code_stale": code_stale,
        "commits_behind": commits_behind,
        "boot_sha": bi.get("boot_sha"),
        "head_sha": bi.get("head_sha"),
    }

    hb_verdict = heartbeat.get("verdict")
    hb_headline = heartbeat.get("headline")

    # Router precedence (operational before behavioural — a dead loop makes
    # every other verdict moot; running stale code makes committed fixes
    # inert; a behavioural flag is a real but slower-burning concern; a
    # lagging loop is the mildest). Forwards the chosen axis's own headline
    # verbatim — no minted grade. Same documented-precedence idea as
    # trader_scorecard._FOCUS_ORDER.
    has_history = bool(decs) or bool(trades)
    if hb_verdict == "STALLED":
        state, headline = "LOOP_STALLED", hb_headline
    elif code_stale:
        n = commits_behind
        state = "CODE_STALE"
        headline = (
            f"Live process is running stale code"
            + (f" ({n} commit{'s' if n != 1 else ''} behind HEAD)"
               if isinstance(n, int) and n > 0 else "")
            + " — restart paper-trader to apply committed fixes.")
    elif scorecard_state == "FLAGS_PRESENT":
        state = "BEHAVIOURAL_FLAGS"
        headline = (scorecard.get("headline")
                    or "Behavioural checks flagging — see focus.")
    elif hb_verdict == "LAGGING":
        state, headline = "LOOP_LAGGING", hb_headline
    elif not has_history:
        state = "NO_DATA"
        headline = "No trades or decisions recorded yet."
    else:
        state = "HEALTHY"
        headline = ("Loop alive, code current, no behavioural flags."
                    if integrity_known
                    else "Loop alive, no behavioural flags "
                         "(code integrity not checked).")

    return {
        "as_of": now.isoformat(timespec="seconds"),
        "state": state,
        "headline": headline,
        "money": money,
        "concentration": concentration,
        "liveness": {
            "verdict": hb_verdict,
            "headline": hb_headline,
            "restart_recommended": bool(heartbeat.get("restart_recommended")),
            "secs_since_last_decision": heartbeat.get(
                "secs_since_last_decision"),
        },
        "integrity": integrity,
        "focus": focus,
        "scorecard_state": scorecard_state,
    }


if __name__ == "__main__":  # one-screen status, usable when :8090 is wedged
    import json
    import sys

    from paper_trader.store import INITIAL_CASH, get_store

    try:
        from paper_trader import market as _mkt
        _mkt_open = _mkt.is_market_open(datetime.now(timezone.utc))
    except Exception:
        _mkt_open = False

    s = get_store()
    rep = build_desk_pulse(
        s.get_portfolio(),
        s.open_positions(),
        s.recent_trades(2000),
        s.recent_decisions(limit=3000),
        s.equity_curve(limit=5000),
        build_info=None,           # CLI has no git-SHA context; integrity omitted
        market_open=_mkt_open,
        initial_cash=INITIAL_CASH,
    )
    if "--json" in sys.argv:
        print(json.dumps(rep, indent=2, default=str))
    else:
        m, c = rep["money"], rep["concentration"]
        print(f"DESK PULSE  [{rep['state']}]  {rep['headline']}")
        print(f"  equity ${m['equity_usd']}  ret {m['total_return_pct']}%  "
              f"realised ${m['realized_pl_usd']}  unreal ${m['unrealized_pl_usd']}")
        print(f"  win-rate {m['win_rate_pct']}%  PF {m['profit_factor']}  "
              f"({m['n_round_trips']} round-trips)")
        print(f"  book: {c['n_open_positions']} names  top {c['top_name']} "
              f"{c['top_weight_pct']}%  gross ${c['gross_exposure_usd']}")
        print(f"  loop: {rep['liveness']['verdict']} — "
              f"{rep['liveness']['headline']}")
        if rep["focus"]:
            print(f"  look first: {rep['focus'].get('name')} — "
                  f"{rep['focus'].get('headline')}")
