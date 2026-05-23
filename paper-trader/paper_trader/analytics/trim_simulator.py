"""Per-position trim ladder with scorer-EV math.

What's missing today: ``/api/suggestion-impact`` projects ONE trim rung
(default 50%) and ONLY for tickers the suggestions engine already flagged as
TRIM/EXIT, and never references the DecisionScorer's 5-day forward-return
prediction. The current concentrated book (NVDA 65.7% with scorer EXIT at
~-22.6% 5d) leaves the operator without a "trim K shares to cap downside at
$X while forgoing $Y of upside" ladder for every held name.

``build_trim_simulator`` answers exactly that: for each held position, three
rungs (25/50/75% of qty by default) with shares_to_trim, cash_freed,
remaining_market_value, new_weight_pct, plus the **scorer-EV math** —
``ev_avoided_loss_usd`` = ``|pred%|×cash_freed`` when pred<0 (the loss you
avoid if the scorer is right) and ``ev_forgone_upside_usd`` =
``pred%×cash_freed`` when pred>0 (the upside you give up if the scorer is
right). A per-position verdict (RECOMMEND_EXIT / RECOMMEND_TRIM / NEUTRAL /
HOLD) synthesises scorer pred sign+magnitude × current weight.

Inputs are pre-fetched (positions + total_value + per-ticker scorer pred
dicts), keeping the builder pure / never-raises / trivially unit-testable —
the ``position_blowup`` precedent. The route is a thin SWR wrapper that fans
the scorer-predictions call out so the panel and the
``-m paper_trader.analytics.trim_simulator`` CLI can never disagree (SSOT,
AGENTS.md #10).

Verdict ladder (book-level):

* ``NO_DATA``           — no positions / zero total_value / no priced rows.
* ``ALL_HOLD``          — every position is HOLD or NEUTRAL (no trim
  pressure).
* ``TRIM_RECOMMENDED``  — at least one position carries ``RECOMMEND_TRIM``.
* ``EXIT_RECOMMENDED``  — at least one position carries ``RECOMMEND_EXIT``
  (scorer is bearish AND the position is concentrated).

Advisory only — never gates Opus, adds no caps (AGENTS.md #2/#12). The
scorer is observational here: the recommendation never auto-executes; the
operator (or Opus, on the next cycle) decides.
"""
from __future__ import annotations

from datetime import datetime, timezone

#: Default trim fractions of held qty (sorted ascending). 0.25/0.50/0.75 are
#: the three decision rungs the discretionary trader actually thinks in —
#: "shave a quarter", "halve", "almost all". A 1.0 rung is intentionally
#: omitted: that's the EXIT decision and lives in ``/api/liquidation-preview``
#: (full-book), not here (per-position ladder).
DEFAULT_RUNGS = (0.25, 0.50, 0.75)

#: Per-position verdict thresholds.
EXIT_PRED_PCT = -10.0           # scorer pred ≤ this → EXIT bias
TRIM_PRED_PCT = -5.0            # scorer pred ≤ this → TRIM bias
HOLD_PRED_PCT = 3.0             # scorer pred ≥ this → HOLD bias
CONCENTRATED_WEIGHT_PCT = 40.0  # weight ≥ this → "concentrated"
HEAVY_WEIGHT_PCT = 25.0         # weight ≥ this → "heavy"


def _z(v, ndigits: int = 2):
    """Round, folding ``-0.0 → 0.0`` so JSON never carries a signed zero
    (the ``position_blowup._z`` / ``stress_scenarios._z`` precedent)."""
    if v is None:
        return None
    try:
        r = round(float(v), ndigits)
    except (TypeError, ValueError):
        return None
    return 0.0 if r == 0 else r


def _position_value(p: dict) -> float:
    """Best-effort market value of a position row — prefers ``market_value``
    (written by ``strategy._mark_to_market``); falls back to
    ``current_price × qty × mult`` (options ×100). Never raises — a garbage
    row contributes 0.0, same contract as ``position_blowup._position_value``."""
    mv = p.get("market_value")
    if mv is not None:
        try:
            return float(mv)
        except (TypeError, ValueError):
            pass
    try:
        ptype = p.get("type") or "stock"
        mult = 100 if ptype in ("call", "put") else 1
        price = p.get("current_price") or p.get("avg_cost") or 0.0
        qty = float(p.get("qty") or 0)
        return float(price) * qty * mult
    except (TypeError, ValueError):
        return 0.0


def _position_qty(p: dict) -> float:
    try:
        return float(p.get("qty") or 0)
    except (TypeError, ValueError):
        return 0.0


def _classify_position(pred_pct: float | None, weight_pct: float) -> str:
    """Combine scorer pred + concentration weight into a per-position verdict.

    Two axes — magnitude of bearish prediction and current weight in the book.
    A bearish pred ON a heavy position is the EXIT bias (the scorer says this
    name will hurt, and you're already over-exposed to it). A bearish pred on
    a light position is just TRIM (smaller damage). A heavy weight with no
    scorer signal is still TRIM (concentration alone is decision-relevant).
    """
    # No scorer signal — fall back to weight-only triage.
    if pred_pct is None:
        if weight_pct >= CONCENTRATED_WEIGHT_PCT:
            return "RECOMMEND_TRIM"
        return "NEUTRAL"
    if pred_pct <= EXIT_PRED_PCT and weight_pct >= HEAVY_WEIGHT_PCT:
        return "RECOMMEND_EXIT"
    if pred_pct <= TRIM_PRED_PCT and weight_pct >= HEAVY_WEIGHT_PCT:
        return "RECOMMEND_TRIM"
    if pred_pct <= EXIT_PRED_PCT:
        # Bearish but not heavy — still flag as TRIM (the scorer is shouting).
        return "RECOMMEND_TRIM"
    if weight_pct >= CONCENTRATED_WEIGHT_PCT and pred_pct < HOLD_PRED_PCT:
        # Heavy + flat-to-mildly-bullish → still a trim case on concentration
        # alone (the position-blowup verdict).
        return "RECOMMEND_TRIM"
    if pred_pct >= HOLD_PRED_PCT:
        return "HOLD"
    return "NEUTRAL"


def _recommended_rung(verdict: str, rungs_out: list[dict]) -> dict | None:
    """Pick the rung the verdict points to.

    EXIT bias → deepest available rung; TRIM bias → middle rung; HOLD / NEUTRAL
    → no recommendation (None). The middle pick for TRIM mirrors the existing
    ``suggestion_impact``'s 50% default — same default, but here it's one rung
    in a ladder, not the only option.
    """
    if not rungs_out:
        return None
    if verdict == "RECOMMEND_EXIT":
        return max(rungs_out, key=lambda r: r.get("trim_fraction") or 0.0)
    if verdict == "RECOMMEND_TRIM":
        # Middle index — for a 3-rung default this is the 0.50 rung.
        return rungs_out[len(rungs_out) // 2]
    return None


def build_trim_simulator(
    positions: list[dict] | None,
    total_value: float | None,
    scorer_predictions: list[dict] | None = None,
    *,
    rungs: tuple[float, ...] = DEFAULT_RUNGS,
    now: datetime | None = None,
) -> dict:
    """Pure: per-position trim ladder with scorer-EV math. Never raises.

    ``positions``: list of dicts from ``store.open_positions()`` — keys read
       are ``ticker``, ``qty``, ``current_price``/``market_value``, ``type``,
       ``avg_cost``. Garbage rows contribute 0.0 and are silently skipped.
    ``total_value``: portfolio total_value in USD. Zero/negative → NO_DATA.
    ``scorer_predictions``: optional list of dicts
       ``{"ticker": "NVDA", "pred_5d_return_pct": -22.6, "verdict": "EXIT",
       "off_distribution": false}``. Missing entries degrade gracefully
       (per-position scorer fields become ``None``; verdict falls back to the
       weight-only triage).
    ``rungs``: trim fractions in (0.0, 1.0]; cleaned + deduped + sorted.
    """
    now = now or datetime.now(timezone.utc)
    try:
        tv = float(total_value or 0.0)
    except (TypeError, ValueError):
        tv = 0.0

    # Clean rungs: (0, 1] only, deduped, sorted ascending.
    cleaned_rungs = sorted({
        round(float(r), 4) for r in (rungs or ())
        if isinstance(r, (int, float)) and 0.0 < float(r) <= 1.0
    })
    if not cleaned_rungs:
        cleaned_rungs = list(DEFAULT_RUNGS)

    base = {
        "as_of": now.isoformat(timespec="seconds"),
        "n_positions": 0,
        "total_value_usd": _z(tv),
        "rungs_pct": [_z(r * 100.0) for r in cleaned_rungs],
        "positions": [],
        "n_trim_recommended": 0,
        "n_exit_recommended": 0,
    }

    rows_in = list(positions or [])
    if not rows_in or tv <= 0:
        base["state"] = "NO_DATA"
        base["headline"] = "Trim simulator: no priced book to simulate yet."
        return base

    # Index scorer predictions by ticker (uppercase) for O(1) lookup.
    preds_by_ticker: dict[str, dict] = {}
    for sp in (scorer_predictions or []):
        if not isinstance(sp, dict):
            continue
        tk = (sp.get("ticker") or "").upper()
        if not tk:
            continue
        raw_pred = sp.get("pred_5d_return_pct")
        try:
            pred = float(raw_pred) if raw_pred is not None else None
        except (TypeError, ValueError):
            pred = None
        preds_by_ticker[tk] = {
            "pred_5d_return_pct": pred,
            "verdict": sp.get("verdict"),
            "off_distribution": bool(sp.get("off_distribution")),
        }

    out_rows: list[dict] = []
    n_trim, n_exit = 0, 0
    for p in rows_in:
        val = _position_value(p)
        if val <= 0:
            continue
        qty = _position_qty(p)
        if qty <= 0:
            continue
        ticker = (p.get("ticker") or "").upper() or None
        ptype = (p.get("type") or "stock").lower()
        weight = val / tv * 100.0

        sp_meta = preds_by_ticker.get(ticker) if ticker else None
        pred = sp_meta.get("pred_5d_return_pct") if sp_meta else None
        scorer_verdict = sp_meta.get("verdict") if sp_meta else None
        off_dist = sp_meta.get("off_distribution", False) if sp_meta else False

        rungs_out: list[dict] = []
        for frac in cleaned_rungs:
            shares = qty * frac
            cash_freed = val * frac
            remaining_val = val - cash_freed
            new_weight = remaining_val / tv * 100.0
            # Scorer-EV math: if pred is -10% and we free $200, the kept
            # $200 was expected to lose $20 over 5d. Trimming captures
            # (avoids) that expected loss on the freed $; the kept $
            # still rides that pred. Mirror flip when pred>0.
            ev_kept = (pred / 100.0 * remaining_val) if pred is not None else None
            ev_freed = (pred / 100.0 * cash_freed) if pred is not None else None
            avoided_loss = None
            forgone_upside = None
            if pred is not None:
                if pred < 0:
                    # ev_freed is the negative $ the freed slice would have
                    # lost; the *avoided loss* is its positive magnitude.
                    avoided_loss = -ev_freed
                elif pred > 0:
                    forgone_upside = ev_freed
            rungs_out.append({
                "trim_fraction": _z(frac, 4),
                "trim_pct": _z(frac * 100.0),
                "shares_to_trim": _z(shares, 4),
                "cash_freed_usd": _z(cash_freed),
                "remaining_market_value_usd": _z(remaining_val),
                "new_weight_pct": _z(new_weight),
                "ev_kept_5d_usd": _z(ev_kept),
                "ev_freed_5d_usd": _z(ev_freed),
                "ev_avoided_loss_usd": _z(avoided_loss),
                "ev_forgone_upside_usd": _z(forgone_upside),
            })

        verdict = _classify_position(pred, weight)
        if verdict == "RECOMMEND_EXIT":
            n_exit += 1
        elif verdict == "RECOMMEND_TRIM":
            n_trim += 1
        rec_rung = _recommended_rung(verdict, rungs_out)

        out_rows.append({
            "ticker": ticker,
            "type": ptype,
            "qty": _z(qty, 4),
            "current_market_value_usd": _z(val),
            "current_weight_pct": _z(weight),
            "scorer_pred_5d_pct": _z(pred, 2),
            "scorer_verdict": scorer_verdict,
            "off_distribution": off_dist,
            "rungs": rungs_out,
            "verdict": verdict,
            "recommended_rung": rec_rung,
        })

    if not out_rows:
        base["state"] = "NO_DATA"
        base["headline"] = "Trim simulator: no priceable open positions."
        return base

    # Sort: EXIT bias first, then TRIM, then by descending weight.
    _URGENCY = {"RECOMMEND_EXIT": 0, "RECOMMEND_TRIM": 1, "NEUTRAL": 2, "HOLD": 3}
    out_rows.sort(key=lambda r: (
        _URGENCY.get(r["verdict"], 9),
        -(r["current_weight_pct"] or 0.0),
    ))

    base["positions"] = out_rows
    base["n_positions"] = len(out_rows)
    base["n_trim_recommended"] = n_trim
    base["n_exit_recommended"] = n_exit

    if n_exit:
        base["state"] = "EXIT_RECOMMENDED"
    elif n_trim:
        base["state"] = "TRIM_RECOMMENDED"
    else:
        base["state"] = "ALL_HOLD"

    # Headline: the top (most urgent / heaviest) name + its recommended rung.
    top = out_rows[0]
    rec = top.get("recommended_rung")
    if rec and top["verdict"] in ("RECOMMEND_EXIT", "RECOMMEND_TRIM"):
        avoided = rec.get("ev_avoided_loss_usd")
        if avoided is not None:
            # avoided-loss and forgone-upside are both displayed as positive
            # magnitudes (the operator-readable framing); their sign is
            # implicit in the verdict label.
            ev_phrase = f"avoids ~${avoided:.2f} expected loss"
        else:
            forgone = rec.get("ev_forgone_upside_usd")
            ev_phrase = (
                f"forgoes ~${forgone:.2f} expected upside"
                if forgone is not None else "scorer EV unknown"
            )
        pred_str = (
            f"{top['scorer_pred_5d_pct']:+.1f}% 5d"
            if top['scorer_pred_5d_pct'] is not None else "no scorer"
        )
        base["headline"] = (
            f"Trim simulator ({base['state']}): {top['ticker']} "
            f"({top['current_weight_pct']:.1f}% of book, scorer {pred_str}) — "
            f"trim {rec['shares_to_trim']:g} shares ({rec['trim_pct']:.0f}%) "
            f"frees ${rec['cash_freed_usd']:.2f}, weight "
            f"{top['current_weight_pct']:.1f}%→{rec['new_weight_pct']:.1f}%, "
            f"{ev_phrase}."
        )
    else:
        base["headline"] = (
            f"Trim simulator ({base['state']}): {base['n_positions']} "
            f"position{'' if base['n_positions'] == 1 else 's'}; "
            f"largest {top['ticker']} {top['current_weight_pct']:.1f}% of "
            f"book — no scorer-driven trim pressure."
        )
    return base


def _live_scorer_preds(tickers: list[str]) -> list[dict]:
    """CLI helper — fetch live scorer predictions for ``tickers``.

    Mirrors the feature shape ``/api/scorer-predictions`` uses (live quant +
    last-4h news sentiment + neutral regime). Best-effort: any failure yields
    an empty list so the CLI still prints the (scorer-less) ladder."""
    if not tickers:
        return []
    try:
        from ..ml.decision_scorer import DecisionScorer
        scorer = DecisionScorer()
        if not scorer.is_trained:
            return []
        from .. import signals as _sig
        from ..strategy import get_quant_signals_live
        quant = get_quant_signals_live(tickers) or {}
        sent = {s["ticker"]: s for s in (_sig.ticker_sentiments(tickers, hours=4) or [])}
        out: list[dict] = []
        for tk in tickers:
            q = quant.get(tk) or {}
            s = sent.get(tk) or {}
            try:
                meta = scorer.predict_with_meta(
                    ml_score=float(s.get("max_score") or 0.0),
                    rsi=q.get("rsi"),
                    macd=q.get("macd_signal"),
                    mom5=q.get("mom_5d"),
                    mom20=q.get("mom_20d"),
                    regime_mult=1.0,
                    ticker=tk,
                    vol_ratio=q.get("vol_ratio"),
                    bb_pos=q.get("bb_position"),
                )
                pred_val = float(meta["pred"])
                # Mirror dashboard._scorer_verdict's coarse bucketing so the
                # CLI's `scorer_verdict` matches what /api/trim-simulator
                # surfaces (route reuses scorer_predictions_api which sets it).
                if pred_val >= 3.0:
                    verdict = "STRONG_HOLD"
                elif pred_val >= 1.0:
                    verdict = "HOLD"
                elif pred_val >= -1.0:
                    verdict = "NEUTRAL"
                elif pred_val >= -3.0:
                    verdict = "TRIM"
                else:
                    verdict = "EXIT"
                out.append({
                    "ticker": tk,
                    "pred_5d_return_pct": pred_val,
                    "verdict": verdict,
                    "off_distribution": bool(meta.get("off_distribution")),
                })
            except Exception:
                continue
        return out
    except Exception:
        return []


def _cli_main() -> int:
    """Render the live book's trim simulator table — same SSOT as the route."""
    import json
    from ..store import get_store
    from ..strategy import portfolio_snapshot_readonly
    store = get_store()
    snap = portfolio_snapshot_readonly(store)
    positions = snap.get("positions") or []
    tickers = sorted({
        (p.get("ticker") or "").upper() for p in positions
        if (p.get("ticker") or "").strip()
    })
    preds = _live_scorer_preds(tickers)
    res = build_trim_simulator(positions, snap.get("total_value"), preds)
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli_main())
