"""Rotation skill — when the desk SELLs X and within hours BUYs a different
ticker Y, does Y outperform X over the next ``forward_days``?

Adjacent surfaces that all *almost* answer this, but miss the realized-edge
spread it captures:

* ``cash_redeployment_latency_skill`` — measures the *time* between SELL and
  next-BUY. Says nothing about whether the rotation was *skilled*. A
  fast-redeploy desk that rotates DRAM → MSTR while DRAM rips and MSTR sags
  reads FAST_REDEPLOY but is actively destroying alpha.
* ``round_trip_postmortem`` — measures post-exit drift on the SAME ticker.
  Per-position, not paired. A "PREMATURE" exit of X is judged in isolation
  from what we bought instead.
* ``rebuy_regret`` / ``reentry_velocity`` — SAME-name re-entry. Rotation
  skill targets the CROSS-ticker case (different ticker after the SELL).
* ``trade_attribution`` / ``pnl_attribution`` — book-level P&L breakdown,
  not paired-rotation alpha.

This module answers the missing question: **rotation alpha** —
``(Y_return_since_BUY) - (X_return_since_SELL)`` over ``forward_days``.
Positive alpha = skilled rotation (we picked a better idea). Negative alpha
= lazy/random rotation (we churned and made it worse).

Pair-detection rules (test-locked, mirror ``cash_redeployment_latency_skill``
discipline for cross-tool readability):

1. SELL action ∈ {SELL, SELL_CALL, SELL_PUT}; BUY action ∈ {BUY, BUY_CALL,
   BUY_PUT}. Walk SELLs in chronological order.
2. For each SELL find the EARLIEST subsequent BUY within ``pairing_max_h``
   (default 24h). Beyond that, the rotation thesis is decoupled — they're
   two independent decisions, not a paired rotation.
3. Exclude SAME-TICKER rebuys (those are ``rebuy_regret`` territory; a
   rotation by definition moves to a different name).
4. Optional cash-conservation gate: ``|buy_notional - sell_notional| /
   sell_notional`` within ``[cash_ratio_lo, cash_ratio_hi]``. Default
   bounds (0.3..3.0) are loose because the trader sometimes consolidates
   multiple SELLs into one larger BUY or vice versa — tight cash matching
   would erroneously reject legitimate consolidations.
5. Only score pairs where SELL is at least ``forward_days + 1`` old (so
   the forward window has matured). Younger pairs go to ``n_window_edge``.

For each scored pair we compute:

* ``sold_at_price`` — the SELL fill price (recorded; never re-derived).
* ``bought_at_price`` — the BUY fill price.
* ``sold_forward_price`` — ``price_at(sell_ticker, sell_ts + forward_days)``.
* ``bought_forward_price`` — ``price_at(buy_ticker, buy_ts + forward_days)``.
* ``sold_forward_pct`` — % move on the sold ticker had we held. Counterfactual.
* ``bought_forward_pct`` — % move on the bought ticker since BUY. Actual.
* ``rotation_alpha_pp`` — bought_forward_pct − sold_forward_pct. The metric.

Verdict ladder (most-severe-first; mirrors ``cash_redeployment_latency_skill``):

* ``INSUFFICIENT_DATA`` — fewer than ``MIN_PAIRS_FOR_VERDICT`` (3) scored
  pairs in window. Distinct from LAZY_ROTATION: an empty window is NOT a
  skill failure.
* ``LAZY_ROTATION`` — median alpha ≤ ``LAZY_MEDIAN_PP`` (-1.5pp) AND
  ``negative_alpha_pct`` ≥ ``LAZY_NEG_PCT`` (60%). The book is
  systematically churning into worse setups.
* ``NET_NEGATIVE`` — median alpha ≤ ``NEUTRAL_BAND_PP`` (-0.3pp). Some
  drag but not the systematic-failure mode.
* ``SKILLED_ROTATION`` — median alpha ≥ ``SKILLED_MEDIAN_PP`` (+1.5pp)
  AND ``positive_alpha_pct`` ≥ ``SKILLED_POS_PCT`` (60%).
* ``NET_POSITIVE`` — median alpha ≥ +0.3pp.
* ``NEUTRAL`` — |median alpha| < 0.3pp.

Top-level rollup priority (chat surfaces LAZY/SKILLED first; NEUTRAL/NET_*
are intermediate states that the chat block keeps silent on — same silence
discipline as ``_cash_redeployment_chat_lines``):

  LAZY_ROTATION > SKILLED_ROTATION > NET_NEGATIVE > NET_POSITIVE > NEUTRAL > INSUFFICIENT_DATA

Pure builder. Trades + price_at(ticker, ts) callable in, dict out, never
raises. Observational only — never gates Opus, no caps (AGENTS.md #2/#12 —
same precedent as ``cash_redeployment_latency_skill``).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Sequence

DEFAULT_WINDOW_DAYS = 60.0
DEFAULT_PAIRING_MAX_H = 24.0
DEFAULT_FORWARD_DAYS = 5.0
DEFAULT_CASH_RATIO_LO = 0.3
DEFAULT_CASH_RATIO_HI = 3.0

SKILLED_MEDIAN_PP = 1.5
SKILLED_POS_PCT = 60.0
LAZY_MEDIAN_PP = -1.5
LAZY_NEG_PCT = 60.0
NEUTRAL_BAND_PP = 0.3

MIN_PAIRS_FOR_VERDICT = 3

_BUY_ACTIONS = frozenset({"BUY", "BUY_CALL", "BUY_PUT"})
_SELL_ACTIONS = frozenset({"SELL", "SELL_CALL", "SELL_PUT"})


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


def _safe_notional(tr: dict) -> float:
    for key in ("value", "notional"):
        v = tr.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            try:
                f = float(v)
                if f == f and f >= 0:
                    return abs(f)
            except (TypeError, ValueError):
                pass
    qty = tr.get("qty")
    price = tr.get("price") or tr.get("fill_price")
    if (isinstance(qty, (int, float)) and not isinstance(qty, bool)
            and isinstance(price, (int, float)) and not isinstance(price, bool)):
        try:
            f = abs(float(qty) * float(price))
            if f == f:
                return f
        except (TypeError, ValueError):
            pass
    return 0.0


def _safe_price(tr: dict) -> float | None:
    for key in ("price", "fill_price"):
        v = tr.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            try:
                f = float(v)
                if f == f and f > 0:
                    return f
            except (TypeError, ValueError):
                pass
    return None


def _median(xs: list[float]) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    if n % 2:
        return float(s[mid])
    return float((s[mid - 1] + s[mid]) / 2.0)


def _percentile(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    if len(xs) == 1:
        return float(xs[0])
    s = sorted(xs)
    n = len(s)
    pos = (p / 100.0) * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return float(s[lo] + (s[hi] - s[lo]) * frac)


def build_rotation_skill(
    trades: Sequence[Any] | None,
    *,
    price_at: Callable[[str, datetime], float | None] | None = None,
    now: datetime | None = None,
    window_days: float = DEFAULT_WINDOW_DAYS,
    pairing_max_h: float = DEFAULT_PAIRING_MAX_H,
    forward_days: float = DEFAULT_FORWARD_DAYS,
    cash_ratio_lo: float = DEFAULT_CASH_RATIO_LO,
    cash_ratio_hi: float = DEFAULT_CASH_RATIO_HI,
) -> dict[str, Any]:
    """Pure SELL→cross-ticker-BUY rotation-alpha builder. Never raises.

    Inputs:
      trades — list of dicts ``{action, ticker, timestamp, price, value, ...}``.
      price_at — ``(ticker, ts) -> price_or_None``. If None, every forward
        lookup yields None and the verdict collapses to INSUFFICIENT_DATA.
      now — defaults to ``datetime.now(utc)``.
      window_days — scan SELLs whose timestamp is within this many days.
      pairing_max_h — SELL→BUY pairing horizon (beyond this they're independent).
      forward_days — forward-return measurement horizon.
      cash_ratio_lo/hi — cash-conservation gate (loose by default).
    """
    now = now or datetime.now(timezone.utc)
    window_cutoff = now - timedelta(days=max(0.0, window_days))
    pairing_cutoff = timedelta(hours=max(0.0, pairing_max_h))
    forward_delta = timedelta(days=max(0.0, forward_days))
    # A SELL must be older than `forward_days + 1d` for the forward window
    # to have fully matured. Use 1d of buffer for the price-lookup slippage.
    maturity_threshold = forward_delta + timedelta(days=1.0)

    parsed: list[tuple[datetime, str, dict]] = []
    for tr in (trades or []):
        if not isinstance(tr, dict):
            continue
        action = tr.get("action")
        if not isinstance(action, str):
            continue
        action_u = action.upper()
        if action_u not in (_BUY_ACTIONS | _SELL_ACTIONS):
            continue
        ts = tr.get("timestamp") or tr.get("ts")
        ts_dt = _parse_iso(ts)
        if ts_dt is None:
            continue
        parsed.append((ts_dt, action_u, tr))
    parsed.sort(key=lambda x: x[0])

    pairs: list[dict[str, Any]] = []
    n_sells_in_window = 0
    n_pairs_detected = 0
    n_pairs_scored = 0
    n_window_edge = 0
    n_unpriced = 0
    alphas_pp: list[float] = []

    for i, (sell_ts, sell_action, sell_tr) in enumerate(parsed):
        if sell_action not in _SELL_ACTIONS:
            continue
        if sell_ts < window_cutoff:
            continue
        n_sells_in_window += 1
        sell_ticker = sell_tr.get("ticker")
        if not isinstance(sell_ticker, str) or not sell_ticker:
            continue

        # Find earliest BUY within pairing_max_h that is a DIFFERENT ticker.
        next_buy_ts: datetime | None = None
        next_buy_tr: dict | None = None
        for j in range(i + 1, len(parsed)):
            cand_ts, cand_action, cand_tr = parsed[j]
            if cand_ts - sell_ts > pairing_cutoff:
                break
            if cand_action not in _BUY_ACTIONS:
                continue
            cand_ticker = cand_tr.get("ticker")
            if not isinstance(cand_ticker, str) or not cand_ticker:
                continue
            if cand_ticker == sell_ticker:
                # same-ticker re-entry — rebuy_regret territory, not rotation
                continue
            next_buy_ts = cand_ts
            next_buy_tr = cand_tr
            break

        if next_buy_ts is None or next_buy_tr is None:
            # No paired BUY — SELL doesn't constitute a rotation
            continue

        n_pairs_detected += 1
        buy_ticker = next_buy_tr.get("ticker")

        # Cash-conservation gate (loose). |ΔNotional|/sell_notional in band.
        sell_notional = _safe_notional(sell_tr)
        buy_notional = _safe_notional(next_buy_tr)
        cash_ratio = None
        if sell_notional > 0:
            cash_ratio = buy_notional / sell_notional
            if cash_ratio < cash_ratio_lo or cash_ratio > cash_ratio_hi:
                # Notional mismatch is too extreme to call a coherent rotation
                pairs.append({
                    "sell_ts": sell_ts.isoformat(),
                    "sell_ticker": sell_ticker,
                    "buy_ts": next_buy_ts.isoformat(),
                    "buy_ticker": buy_ticker,
                    "sell_notional_usd": round(sell_notional, 2),
                    "buy_notional_usd": round(buy_notional, 2),
                    "cash_ratio": round(cash_ratio, 3),
                    "status": "CASH_MISMATCH",
                })
                continue

        # Maturity gate — forward window must have elapsed.
        time_elapsed = now - sell_ts
        if time_elapsed < maturity_threshold:
            n_window_edge += 1
            pairs.append({
                "sell_ts": sell_ts.isoformat(),
                "sell_ticker": sell_ticker,
                "buy_ts": next_buy_ts.isoformat(),
                "buy_ticker": buy_ticker,
                "sell_notional_usd": round(sell_notional, 2),
                "buy_notional_usd": round(buy_notional, 2),
                "cash_ratio": (round(cash_ratio, 3) if cash_ratio is not None
                               else None),
                "status": "WINDOW_EDGE",
            })
            continue

        # Score the pair — fetch forward prices.
        sold_at = _safe_price(sell_tr)
        bought_at = _safe_price(next_buy_tr)
        if sold_at is None or bought_at is None:
            n_unpriced += 1
            pairs.append({
                "sell_ts": sell_ts.isoformat(),
                "sell_ticker": sell_ticker,
                "buy_ts": next_buy_ts.isoformat(),
                "buy_ticker": buy_ticker,
                "status": "MISSING_FILL_PRICE",
            })
            continue

        sold_fwd_price: float | None = None
        bought_fwd_price: float | None = None
        if price_at is not None:
            try:
                sold_fwd_price = price_at(sell_ticker, sell_ts + forward_delta)
            except Exception:
                sold_fwd_price = None
            try:
                bought_fwd_price = price_at(buy_ticker, next_buy_ts + forward_delta)
            except Exception:
                bought_fwd_price = None

        if (not isinstance(sold_fwd_price, (int, float)) or sold_fwd_price <= 0
                or not isinstance(bought_fwd_price, (int, float))
                or bought_fwd_price <= 0):
            n_unpriced += 1
            pairs.append({
                "sell_ts": sell_ts.isoformat(),
                "sell_ticker": sell_ticker,
                "buy_ts": next_buy_ts.isoformat(),
                "buy_ticker": buy_ticker,
                "sold_at_price": sold_at,
                "bought_at_price": bought_at,
                "status": "UNPRICED_FORWARD",
            })
            continue

        sold_fwd_pct = (sold_fwd_price / sold_at - 1.0) * 100.0
        bought_fwd_pct = (bought_fwd_price / bought_at - 1.0) * 100.0
        alpha_pp = bought_fwd_pct - sold_fwd_pct
        alphas_pp.append(alpha_pp)
        n_pairs_scored += 1

        pairs.append({
            "sell_ts": sell_ts.isoformat(),
            "sell_ticker": sell_ticker,
            "buy_ts": next_buy_ts.isoformat(),
            "buy_ticker": buy_ticker,
            "sell_notional_usd": round(sell_notional, 2),
            "buy_notional_usd": round(buy_notional, 2),
            "cash_ratio": (round(cash_ratio, 3) if cash_ratio is not None
                           else None),
            "sold_at_price": round(sold_at, 4),
            "bought_at_price": round(bought_at, 4),
            "sold_forward_price": round(float(sold_fwd_price), 4),
            "bought_forward_price": round(float(bought_fwd_price), 4),
            "sold_forward_pct": round(sold_fwd_pct, 3),
            "bought_forward_pct": round(bought_fwd_pct, 3),
            "rotation_alpha_pp": round(alpha_pp, 3),
            "status": "SCORED",
        })

    # Aggregates
    median_alpha = _median(alphas_pp)
    p25_alpha = _percentile(alphas_pp, 25.0)
    p75_alpha = _percentile(alphas_pp, 75.0)
    n_pos = sum(1 for x in alphas_pp if x > 0)
    n_neg = sum(1 for x in alphas_pp if x < 0)
    pos_pct = (round(100.0 * n_pos / len(alphas_pp), 2)
               if alphas_pp else None)
    neg_pct = (round(100.0 * n_neg / len(alphas_pp), 2)
               if alphas_pp else None)
    mean_alpha = (round(sum(alphas_pp) / len(alphas_pp), 3)
                  if alphas_pp else None)

    # Verdict
    if n_pairs_scored < MIN_PAIRS_FOR_VERDICT:
        verdict = "INSUFFICIENT_DATA"
        headline = (
            f"insufficient: {n_pairs_scored} scored rotations in last "
            f"{window_days:g}d (min {MIN_PAIRS_FOR_VERDICT})"
        )
    else:
        m = median_alpha if median_alpha is not None else 0.0
        p = pos_pct if pos_pct is not None else 0.0
        n = neg_pct if neg_pct is not None else 0.0
        if m <= LAZY_MEDIAN_PP and n >= LAZY_NEG_PCT:
            verdict = "LAZY_ROTATION"
            headline = (
                f"lazy: median alpha {m:+.2f}pp; {n_neg}/{n_pairs_scored} "
                f"rotations destroyed value ({n:.0f}%)"
            )
        elif m >= SKILLED_MEDIAN_PP and p >= SKILLED_POS_PCT:
            verdict = "SKILLED_ROTATION"
            headline = (
                f"skilled: median alpha {m:+.2f}pp; {n_pos}/{n_pairs_scored} "
                f"rotations added value ({p:.0f}%)"
            )
        elif m <= -NEUTRAL_BAND_PP:
            verdict = "NET_NEGATIVE"
            headline = (
                f"net-negative: median alpha {m:+.2f}pp across "
                f"{n_pairs_scored} rotations"
            )
        elif m >= NEUTRAL_BAND_PP:
            verdict = "NET_POSITIVE"
            headline = (
                f"net-positive: median alpha {m:+.2f}pp across "
                f"{n_pairs_scored} rotations"
            )
        else:
            verdict = "NEUTRAL"
            headline = (
                f"neutral: median alpha {m:+.2f}pp across {n_pairs_scored} "
                f"rotations (within ±{NEUTRAL_BAND_PP}pp band)"
            )

    pairs.sort(key=lambda p: p["sell_ts"], reverse=True)

    return {
        "verdict": verdict,
        "headline": headline,
        "as_of": now.isoformat(),
        "window_days": window_days,
        "forward_days": forward_days,
        "pairing_max_h": pairing_max_h,
        "stats": {
            "n_sells_in_window": n_sells_in_window,
            "n_pairs_detected": n_pairs_detected,
            "n_pairs_scored": n_pairs_scored,
            "n_window_edge": n_window_edge,
            "n_unpriced": n_unpriced,
            "median_alpha_pp": (round(median_alpha, 3)
                                if median_alpha is not None else None),
            "mean_alpha_pp": mean_alpha,
            "p25_alpha_pp": (round(p25_alpha, 3)
                             if p25_alpha is not None else None),
            "p75_alpha_pp": (round(p75_alpha, 3)
                             if p75_alpha is not None else None),
            "positive_alpha_pct": pos_pct,
            "negative_alpha_pct": neg_pct,
            "n_positive": n_pos,
            "n_negative": n_neg,
        },
        "thresholds": {
            "skilled_median_pp": SKILLED_MEDIAN_PP,
            "skilled_pos_pct": SKILLED_POS_PCT,
            "lazy_median_pp": LAZY_MEDIAN_PP,
            "lazy_neg_pct": LAZY_NEG_PCT,
            "neutral_band_pp": NEUTRAL_BAND_PP,
            "min_pairs_for_verdict": MIN_PAIRS_FOR_VERDICT,
            "cash_ratio_lo": cash_ratio_lo,
            "cash_ratio_hi": cash_ratio_hi,
        },
        "pairs": pairs,
    }
