"""Regime × leveraged-ETF-exposure fit — is the book's leverage
class appropriate to the current market regime?

The recurring 2026-05 backtest finding (paper-trader AGENTS.md):
*the alpha is largely a leveraged-bull-window artifact riding the
personas that boost 3x ETFs, NOT a portfolio-management edge.* The
top BUY tickers across the best-runs are 100% leveraged-bull or
single-stock-momentum vehicles (SOXL, TQQQ, UPRO, MSTR, SPXL, NVDA,
TECL, BTC-USD, FAS). A book whose realized return depends on the
bull tape getting *and staying* bull has a regime-flip failure mode
no behavioural-mirror endpoint currently surfaces.

The existing surfaces describe pieces:

* ``/api/risk`` reports ``leveraged_pct`` of the live book RIGHT NOW
  — a snapshot of exposure, no regime context.
* ``/api/sector-heatmap`` has a ``memory_leveraged`` bucket — price
  + RSI + news per bucket, but no book-level leverage verdict and
  no regime overlay.
* ``analytics.ml.leveraged_skill`` does OOS scorer-IC stratified by
  leveraged-vs-non-leveraged — measures whether the DecisionScorer's
  rank skill *generalises* across leverage classes; not a regime-fit
  call on the live book.

None answer the discretionary question: *given the current SPY-20d
regime, is my leveraged-ETF exposure (and recent trade flow into
those names) aligned with the tape, or am I levering into a
headwind?*

Verdict matrix (regime × lev_pct × lev_flow):

* ``BLIND_LEVERING`` — regime in (sideways, bear) AND recent
  buy-flow into leveraged ETFs ≥ ``HIGH_FLOW_PCT`` (active
  deterioration). Highest priority — fires even before
  ``DANGEROUS_HEADWIND`` because the *direction of change* matters
  more than the static exposure.
* ``DANGEROUS_HEADWIND`` — regime == bear AND
  lev_pct ≥ ``HIGH_LEV_FLOOR``. Static high exposure into a bear
  tape. Decay drag (-1% to -2% / month at 3x in flat-to-down)
  compounds against this.
* ``ALIGNED`` — regime == bull AND lev_pct ≥ ``ALIGNED_LEV_FLOOR``.
  Correctly tailwinded; the leverage IS doing work for you.
* ``MISSED_TAILWIND`` — regime == bull AND lev_pct ≤
  ``LOW_LEV_CEIL`` AND (no recent leveraged buys). Bull tape and
  you're not riding it. Affirmation of "find a tailwind name".
* ``DEFENSIVE`` — regime in (sideways, bear) AND lev_pct ≤
  ``LOW_LEV_CEIL`` AND no recent leveraged buy-flow. Correctly
  de-risked.
* ``NEUTRAL`` — none of the above. Mid-band exposure or
  ambiguous regime + ambiguous flow.
* ``NO_DATA`` — no SPY momentum AND empty positions AND empty
  trade flow. Always emits the envelope; never raises.

Pure builder. Snapshot in (positions, cash, spy_mom_20d, recent
trades, now), dict out, never raises. Observational only — never
gates Opus, no caps (AGENTS.md #2/#12).

The leveraged set is duplicated inline (precedent:
``strategy._LEVERAGED_ETFS_LIVE``, ``backtest._LEVERAGED_ETFS``,
``ml.leveraged_skill._LEVERAGED_ETFS``). The list of inverses is
included so SQQQ / SPXS / SOXS / TECS / FNGD count as leveraged
too — those are leveraged-SHORT names and an analytical regime-fit
endpoint must surface a flip into them in a bull regime as
``BLIND_LEVERING`` just as forcefully as it surfaces TQQQ in bear.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Sequence

# Verbatim mirror of ``strategy._LEVERAGED_ETFS_LIVE`` (precedent: the
# `_LEVERAGED_ETFS` duplication in ``ml.leveraged_skill``). Includes
# the leveraged-inverse set so a flip into SQQQ/SOXS/SPXS during a
# bull regime is also flagged as a regime mismatch.
_LEVERAGED_ETFS = frozenset({
    # Long leveraged (3x bull)
    "TQQQ", "UPRO", "SPXL", "UDOW", "URTY", "SOXL", "TECL", "FNGU",
    "CURE", "LABU", "NAIL", "DPST", "FAS", "DFEN", "TNA", "UTSL",
    # 2x and single-name leveraged
    "QLD", "SSO", "NVDU", "MSFU", "AMZU", "TSLL", "CONL", "BITU", "ETHU",
    # Inverse / leveraged-short
    "SQQQ", "SPXS", "SOXS", "TECS", "FNGD",
})

# SPY 20d momentum thresholds — mirror of ``strategy._ml_live_opinion``
# inline rule (mom20 > 3 = bull, < -3 = bear, else sideways).
DEFAULT_BULL_MOM_PCT = 3.0
DEFAULT_BEAR_MOM_PCT = -3.0

# Leverage exposure thresholds — % of total equity in leveraged
# names. 30% is the AGENTS.md-documented "elevated" cap arm.
DEFAULT_HIGH_LEV_FLOOR = 30.0
DEFAULT_ALIGNED_LEV_FLOOR = 20.0
DEFAULT_LOW_LEV_CEIL = 10.0

# Recent-trade-flow window and threshold (% of total equity bought
# into leveraged names within the window).
DEFAULT_FLOW_WINDOW_HOURS = 24.0
DEFAULT_HIGH_FLOW_PCT = 5.0

# Sample-size floor — below this we keep the envelope but withhold
# the regime-fit verdict.
MIN_TOTAL_VALUE_USD = 1.0


def is_leveraged(ticker: str) -> bool:
    """Pure membership check — uppercased lookup. NVDA → False,
    'soxl' → True, '' / None → False."""
    if not isinstance(ticker, str) or not ticker:
        return False
    return ticker.upper() in _LEVERAGED_ETFS


def classify_regime(spy_mom_20d: Any,
                    bull_mom_pct: float = DEFAULT_BULL_MOM_PCT,
                    bear_mom_pct: float = DEFAULT_BEAR_MOM_PCT) -> str:
    """Pure regime classifier — returns one of 'bull', 'bear',
    'sideways', 'unknown'. Mirrors ``strategy._ml_live_opinion``."""
    if not isinstance(spy_mom_20d, (int, float)):
        return "unknown"
    try:
        v = float(spy_mom_20d)
    except (TypeError, ValueError):
        return "unknown"
    if v != v:  # NaN
        return "unknown"
    if v > bull_mom_pct:
        return "bull"
    if v < bear_mom_pct:
        return "bear"
    return "sideways"


def _safe_pos_value(p: Any) -> tuple[str, float]:
    """Defensive — returns (ticker_upper, market_value) tuple; on
    malformed row returns ('', 0.0) so the row contributes nothing."""
    if not isinstance(p, dict):
        return "", 0.0
    t = p.get("ticker")
    if not isinstance(t, str) or not t:
        return "", 0.0
    mv = p.get("market_value")
    if mv is None:
        # Try to derive from qty * mark
        qty = p.get("qty")
        mark = p.get("mark") or p.get("current_price") or p.get("price")
        if isinstance(qty, (int, float)) and isinstance(mark, (int, float)):
            try:
                mv = float(qty) * float(mark)
            except (TypeError, ValueError):
                mv = 0.0
        else:
            mv = 0.0
    try:
        mv = float(mv)
    except (TypeError, ValueError):
        mv = 0.0
    if mv != mv:  # NaN
        mv = 0.0
    return t.upper(), mv


def _parse_iso(ts: Any) -> datetime | None:
    if not isinstance(ts, str) or not ts:
        return None
    try:
        s = ts.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (TypeError, ValueError):
        return None


def build_regime_leverage_fit_skill(
    positions: Sequence[Any] | None,
    cash_usd: Any,
    total_value_usd: Any,
    spy_mom_20d: Any,
    recent_trades: Sequence[Any] | None,
    *,
    now: datetime | None = None,
    bull_mom_pct: float = DEFAULT_BULL_MOM_PCT,
    bear_mom_pct: float = DEFAULT_BEAR_MOM_PCT,
    high_lev_floor: float = DEFAULT_HIGH_LEV_FLOOR,
    aligned_lev_floor: float = DEFAULT_ALIGNED_LEV_FLOOR,
    low_lev_ceil: float = DEFAULT_LOW_LEV_CEIL,
    flow_window_hours: float = DEFAULT_FLOW_WINDOW_HOURS,
    high_flow_pct: float = DEFAULT_HIGH_FLOW_PCT,
) -> dict[str, Any]:
    """Pure regime × leveraged-exposure verdict. Never raises.

    Inputs:
      ``positions`` — list of position dicts ``{ticker, market_value, ...}``.
        Closed positions filtered by caller.
      ``cash_usd`` — current cash position (optional, used for headline only).
      ``total_value_usd`` — total book equity (cash + open value).
      ``spy_mom_20d`` — 20-day SPY momentum % (regime input).
      ``recent_trades`` — list of trade dicts ``{action, ticker, ts, notional}``
        used to compute recent buy-flow into leveraged ETFs.
      ``now`` — defaults to ``datetime.now(utc)``.

    Threshold overrides exposed for tests + caller knobs.
    """
    now = now or datetime.now(timezone.utc)

    # ── 1. portfolio leveraged exposure ────────────────────────────
    pos_list = list(positions or [])
    total_open = 0.0
    leveraged_usd = 0.0
    leveraged_positions: list[dict[str, Any]] = []
    for p in pos_list:
        t, mv = _safe_pos_value(p)
        if not t:
            continue
        total_open += mv
        if is_leveraged(t):
            leveraged_usd += mv
            leveraged_positions.append({"ticker": t, "market_value": round(mv, 2)})

    try:
        cash_f = float(cash_usd) if isinstance(cash_usd, (int, float)) else 0.0
    except (TypeError, ValueError):
        cash_f = 0.0
    if cash_f != cash_f:
        cash_f = 0.0

    try:
        tv = float(total_value_usd) if isinstance(total_value_usd, (int, float)) else None
    except (TypeError, ValueError):
        tv = None
    if tv is not None and tv != tv:
        tv = None
    # Fallback: derive from cash + open_value
    if tv is None or tv <= 0.0:
        tv = cash_f + total_open

    leveraged_pct: float | None
    if tv > MIN_TOTAL_VALUE_USD:
        leveraged_pct = round((leveraged_usd / tv) * 100.0, 2)
    else:
        leveraged_pct = None

    # ── 2. recent trade flow into leveraged names ──────────────────
    flow_cutoff = None
    if flow_window_hours > 0:
        flow_cutoff_secs = flow_window_hours * 3600.0
    else:
        flow_cutoff_secs = 0.0
    leveraged_buy_usd = 0.0
    leveraged_sell_usd = 0.0
    n_leveraged_buys = 0
    n_leveraged_sells = 0
    for tr in (recent_trades or []):
        if not isinstance(tr, dict):
            continue
        action = tr.get("action")
        if not isinstance(action, str):
            continue
        action_u = action.upper()
        if action_u not in ("BUY", "SELL", "BUY_CALL", "BUY_PUT", "SELL_CALL", "SELL_PUT"):
            continue
        t = tr.get("ticker")
        if not isinstance(t, str) or not t:
            continue
        if not is_leveraged(t):
            continue
        ts_dt = _parse_iso(tr.get("ts") or tr.get("timestamp"))
        if ts_dt is None:
            continue
        age_s = (now - ts_dt).total_seconds()
        if age_s < 0 or age_s > flow_cutoff_secs:
            continue
        # Trades table uses `value` (qty*price). Accept `notional` as
        # well so a synthetic test can pass either name.
        notional = tr.get("notional")
        if notional is None:
            notional = tr.get("value")
        if notional is None:
            qty = tr.get("qty")
            price = tr.get("price") or tr.get("fill_price")
            if isinstance(qty, (int, float)) and isinstance(price, (int, float)):
                try:
                    notional = abs(float(qty) * float(price))
                except (TypeError, ValueError):
                    notional = 0.0
            else:
                notional = 0.0
        try:
            notional = float(notional)
        except (TypeError, ValueError):
            notional = 0.0
        if notional != notional or notional < 0:
            notional = 0.0
        if action_u in ("BUY", "BUY_CALL", "BUY_PUT"):
            leveraged_buy_usd += notional
            n_leveraged_buys += 1
        else:
            leveraged_sell_usd += notional
            n_leveraged_sells += 1

    buy_flow_pct: float | None
    sell_flow_pct: float | None
    if tv is not None and tv > MIN_TOTAL_VALUE_USD:
        buy_flow_pct = round((leveraged_buy_usd / tv) * 100.0, 2)
        sell_flow_pct = round((leveraged_sell_usd / tv) * 100.0, 2)
    else:
        buy_flow_pct = None
        sell_flow_pct = None

    # ── 3. regime classification ───────────────────────────────────
    regime = classify_regime(spy_mom_20d, bull_mom_pct, bear_mom_pct)
    try:
        spy_mom_f = float(spy_mom_20d) if isinstance(spy_mom_20d, (int, float)) else None
    except (TypeError, ValueError):
        spy_mom_f = None
    if spy_mom_f is not None and spy_mom_f != spy_mom_f:
        spy_mom_f = None

    # ── 4. verdict ladder ──────────────────────────────────────────
    has_book_data = total_open > 0 or cash_f > 0
    has_trade_flow = (n_leveraged_buys + n_leveraged_sells) > 0

    if regime == "unknown" and not has_book_data and not has_trade_flow:
        verdict = "NO_DATA"
        headline = "no data: missing spy momentum and empty portfolio + no recent trade flow"
    elif leveraged_pct is None and not has_trade_flow:
        # Book value too small to compute exposure share, and no flow either.
        verdict = "NO_DATA"
        headline = "no data: portfolio value below floor"
    elif regime == "unknown":
        verdict = "NEUTRAL"
        headline = f"regime unknown (spy_mom_20d missing); lev={leveraged_pct}%"
    else:
        # Priority ladder
        lev_high = leveraged_pct is not None and leveraged_pct >= high_lev_floor
        lev_aligned = leveraged_pct is not None and leveraged_pct >= aligned_lev_floor
        lev_low = leveraged_pct is None or leveraged_pct <= low_lev_ceil
        flow_high = buy_flow_pct is not None and buy_flow_pct >= high_flow_pct

        if regime in ("bear", "sideways") and flow_high:
            verdict = "BLIND_LEVERING"
            headline = (
                f"levering into {regime} (spy_mom_20d={spy_mom_f:.2f}%; "
                f"recent leveraged buy flow {buy_flow_pct}% in {flow_window_hours:g}h)"
            )
        elif regime == "bear" and lev_high:
            verdict = "DANGEROUS_HEADWIND"
            headline = (
                f"levered ({leveraged_pct}%) into bear "
                f"(spy_mom_20d={spy_mom_f:.2f}%)"
            )
        elif regime == "bull" and lev_aligned:
            verdict = "ALIGNED"
            headline = (
                f"levered ({leveraged_pct}%) with bull tailwind "
                f"(spy_mom_20d={spy_mom_f:.2f}%)"
            )
        elif regime == "bull" and lev_low and not flow_high:
            verdict = "MISSED_TAILWIND"
            headline = (
                f"bull tape (spy_mom_20d={spy_mom_f:.2f}%) but only "
                f"{leveraged_pct if leveraged_pct is not None else 0.0}% leveraged"
            )
        elif regime in ("bear", "sideways") and lev_low:
            verdict = "DEFENSIVE"
            headline = (
                f"de-risked ({leveraged_pct if leveraged_pct is not None else 0.0}% "
                f"lev) for {regime} (spy_mom_20d={spy_mom_f:.2f}%)"
            )
        else:
            verdict = "NEUTRAL"
            headline = (
                f"mid-band: regime={regime} "
                f"(spy_mom_20d={spy_mom_f:.2f}%); lev={leveraged_pct}%"
            )

    # ── 5. envelope ────────────────────────────────────────────────
    return {
        "verdict": verdict,
        "headline": headline,
        "as_of": now.isoformat(),
        "regime": regime,
        "spy_mom_20d": spy_mom_f,
        "portfolio": {
            "cash_usd": round(cash_f, 2),
            "total_value_usd": round(tv, 2) if tv is not None else None,
            "open_value_usd": round(total_open, 2),
            "leveraged_usd": round(leveraged_usd, 2),
            "leveraged_pct": leveraged_pct,
            "n_leveraged_positions": len(leveraged_positions),
            "leveraged_positions": leveraged_positions,
        },
        "recent_flow": {
            "window_hours": flow_window_hours,
            "leveraged_buy_usd": round(leveraged_buy_usd, 2),
            "leveraged_sell_usd": round(leveraged_sell_usd, 2),
            "buy_flow_pct": buy_flow_pct,
            "sell_flow_pct": sell_flow_pct,
            "n_leveraged_buys": n_leveraged_buys,
            "n_leveraged_sells": n_leveraged_sells,
        },
        "thresholds": {
            "bull_mom_pct": bull_mom_pct,
            "bear_mom_pct": bear_mom_pct,
            "high_lev_floor": high_lev_floor,
            "aligned_lev_floor": aligned_lev_floor,
            "low_lev_ceil": low_lev_ceil,
            "flow_window_hours": flow_window_hours,
            "high_flow_pct": high_flow_pct,
        },
    }
