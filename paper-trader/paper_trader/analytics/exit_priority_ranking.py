"""Composite 'sell-which-first' ranker over open stock positions.

The operator's blocking question during a heavy book: "if I had to free
cash NOW, which position do I trim FIRST?" Today this requires eyeballing
half a dozen narrow panels in parallel — concentration weight, days-held,
unrealized P/L, news silence, mark freshness — and mentally compositing
them. This builder does that composite explicitly: one 0-100 score per
position with the contributing factors itemized.

Distinct from every neighbour (do not consolidate, AGENTS.md invariant #10):

* ``/api/concentration-cap`` — fires only at a fixed threshold; no ranking.
* ``/api/profit-ladder`` — exit-price schedule per position; ranking by
  realized gain not exit priority.
* ``/api/thesis-drift`` — per-position thesis re-test; binary drift flag,
  no composite.
* ``/api/disagreement`` — scorer-vs-Opus mismatch; doesn't include
  age/silence/concentration.
* ``/api/risk`` — book-level concentration severity; not per-position.
* ``/api/trim-simulator`` — what-if simulator; needs operator to PICK
  the ticker first.

This builder is the missing PRIOR step: "for an honest exit pass, the
order should be X, Y, Z because ...". Pure: takes positions + news_counts
+ total_value, never raises on garbage inputs, never hits a DB or network.

Score composition (each factor 0..1, then weighted-summed and scaled 0..100):

  concentration_factor = clamp(weight_pct / high_conc_pct, 0, 1)
  pnl_factor           = 0.5 if breakeven; +0.5 if losing >= 5%; -0.5 if
                         winning >= 10% (cap 0..1)
  age_factor           = clamp(days_held / long_hold_d, 0, 1)
  silence_factor       = 1.0 if news_silence_h >= stale_news_h, else
                         clamp(news_silence_h / stale_news_h, 0, 1)
  stale_mark_penalty   = subtract 10 from final score if mark > 1d stale

Default weights (sum to 1.0):
  concentration=0.35  (the dominant exit lever in a heavy book)
  pnl          =0.25
  age          =0.20
  silence      =0.20

Empty-book → state=ALL_CASH; opaque headline mirroring the
``_all_cash_streak_chat_lines`` / ``_macro_calendar_chat_lines`` silence
precedent so a healthy all-cash book doesn't surface chat noise.
"""
from __future__ import annotations

from datetime import datetime, timezone


DEFAULT_HIGH_CONC_PCT = 25.0
DEFAULT_STALE_NEWS_HOURS = 168.0  # 1 week
DEFAULT_LONG_HOLD_DAYS = 30.0
DEFAULT_STALE_MARK_HOURS = 24.0

DEFAULT_WEIGHTS = {
    "concentration": 0.35,
    "pnl": 0.25,
    "age": 0.20,
    "silence": 0.20,
}

DEFAULT_STALE_MARK_PENALTY = 10.0


def _parse_ts(ts) -> datetime | None:
    if not ts:
        return None
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(ts).strip().replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _safe_float(x) -> float | None:
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return f


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _is_option(p: dict) -> bool:
    t = p.get("type")
    if t in ("call", "put", "CALL", "PUT"):
        return True
    return p.get("option_type") in ("call", "put", "CALL", "PUT")


def _concentration_factor(weight_pct: float, high_conc_pct: float) -> float:
    if high_conc_pct <= 0:
        return 0.0
    return _clamp(weight_pct / high_conc_pct)


def _pnl_factor(unrealized_pl_pct: float | None) -> float:
    """Heavier exit weight on losers (cut losses), lighter on big winners
    (let runners run). A small loss is roughly the same as a small gain.

    Mapping:
       loss >= 5%      → 1.0
       loss 0..5%      → 0.5..1.0 linear
       gain 0..10%     → 0.5..0.0 linear
       gain >= 10%     → 0.0
       breakeven       → 0.5
       missing         → 0.5 (neutral — no signal)
    """
    if unrealized_pl_pct is None:
        return 0.5
    p = unrealized_pl_pct
    if p <= -5.0:
        return 1.0
    if p < 0:
        # -5..0 maps to 1.0..0.5
        return 0.5 + (abs(p) / 5.0) * 0.5
    if p == 0:
        return 0.5
    if p >= 10.0:
        return 0.0
    # 0..10 maps to 0.5..0.0
    return 0.5 - (p / 10.0) * 0.5


def _age_factor(days_held: float | None, long_hold_d: float) -> float:
    if days_held is None or days_held < 0:
        return 0.0
    if long_hold_d <= 0:
        return 0.0
    return _clamp(days_held / long_hold_d)


def _silence_factor(news_silence_h: float | None, stale_news_h: float) -> float:
    if news_silence_h is None or news_silence_h < 0:
        return 0.5  # unknown → mid
    if stale_news_h <= 0:
        return 0.0
    return _clamp(news_silence_h / stale_news_h)


def _stale_mark(mark_age_h: float | None, stale_mark_h: float) -> bool:
    if mark_age_h is None:
        return False
    return mark_age_h > stale_mark_h


def _explain(factors: dict, weight_pct: float, pnl_pct: float | None,
             days_held: float | None, news_silence_h: float | None,
             stale_mark_penalized: bool,
             high_conc_pct: float, stale_news_h: float,
             long_hold_d: float) -> list[str]:
    out: list[str] = []
    if weight_pct >= high_conc_pct:
        out.append(f"concentration {weight_pct:.1f}% ≥ {high_conc_pct:.0f}% cap")
    elif weight_pct >= high_conc_pct * 0.6:
        out.append(f"concentration {weight_pct:.1f}% approaching {high_conc_pct:.0f}% cap")
    if pnl_pct is not None and pnl_pct <= -5.0:
        out.append(f"loss {pnl_pct:+.1f}% (cut-loss tier)")
    elif pnl_pct is not None and pnl_pct >= 10.0:
        out.append(f"winner {pnl_pct:+.1f}% (let-run tier)")
    elif pnl_pct is not None:
        out.append(f"pnl {pnl_pct:+.1f}%")
    if days_held is not None and days_held >= long_hold_d:
        out.append(f"held {days_held:.1f}d ≥ long-hold {long_hold_d:.0f}d")
    elif days_held is not None and days_held >= long_hold_d * 0.5:
        out.append(f"held {days_held:.1f}d (mid-tenure)")
    if news_silence_h is not None and news_silence_h >= stale_news_h:
        out.append(f"no live news ≥ {stale_news_h/24.0:.0f}d")
    elif news_silence_h is not None and news_silence_h >= stale_news_h * 0.5:
        out.append(f"news quiet {news_silence_h:.0f}h")
    if stale_mark_penalized:
        out.append("stale mark (penalty applied)")
    return out


def _headline(state: str, ranked: list[dict]) -> str:
    if state == "ALL_CASH":
        return "Exit priority: book is 100% cash — nothing to rank."
    if state == "NO_DATA":
        return "Exit priority: insufficient data to rank."
    if not ranked:
        return "Exit priority: no rankable open stock positions."
    top = ranked[0]
    extras: list[str] = []
    if len(ranked) > 1:
        rest = ", ".join(r["ticker"] for r in ranked[1:3])
        extras.append(f"then {rest}")
    pnl_pct = top.get("unrealized_pl_pct")
    pnl_str = f"{pnl_pct:+.1f}%" if isinstance(pnl_pct, (int, float)) else "?"
    return (f"Exit priority: trim {top['ticker']} first "
            f"(score {top['score']:.0f}, weight {top['weight_pct']:.1f}%, "
            f"pnl {pnl_str})"
            + (f" — {extras[0]}" if extras else "")
            + ".")


def build_exit_priority_ranking(
    open_positions: list[dict] | None,
    total_value: float | None,
    news_counts_by_ticker: dict[str, dict] | None = None,
    now: datetime | None = None,
    high_conc_pct: float = DEFAULT_HIGH_CONC_PCT,
    stale_news_h: float = DEFAULT_STALE_NEWS_HOURS,
    long_hold_d: float = DEFAULT_LONG_HOLD_DAYS,
    stale_mark_h: float = DEFAULT_STALE_MARK_HOURS,
    stale_mark_penalty: float = DEFAULT_STALE_MARK_PENALTY,
    weights: dict[str, float] | None = None,
) -> dict:
    """Rank open stock positions by composite 'exit-first' score.

    Args:
        open_positions: list of position rows from ``store.open_positions``.
            Required keys: ``ticker``, ``qty``, ``current_price`` (or 0),
            ``unrealized_pl`` (or 0), ``avg_cost``, ``opened_at``. Option
            rows are excluded (option P/L isn't comparable to %-of-NAV).
        total_value: portfolio total (cash + open value). Used as the
            denominator for the weight_pct factor. If <= 0 → NO_DATA.
        news_counts_by_ticker: optional ``{ticker: {n_articles, hours_since_last}}``.
            ``hours_since_last`` of None or > stale_news_h drives the
            silence_factor; missing ticker → factor falls back to mid (0.5).
        now: injectable wall clock for tests.
        high_conc_pct: weight (%) at which concentration_factor saturates
            to 1.0. Default 25% — the trader-CLAUDE.md heavy-book threshold.
        stale_news_h: news-silence hours at which silence_factor saturates.
        long_hold_d: days_held at which age_factor saturates.
        stale_mark_h: hours after which a current_price is stale.
        stale_mark_penalty: points deducted from final score when mark stale.
        weights: per-factor weights; falls back to DEFAULT_WEIGHTS.

    Returns a JSON-ready dict. Never raises.
    """
    now = now or datetime.now(timezone.utc)
    weights = weights or DEFAULT_WEIGHTS
    out: dict = {
        "as_of": now.isoformat(timespec="seconds"),
        "state": "NO_DATA",
        "n_positions": 0,
        "n_ranked": 0,
        "total_value": _safe_float(total_value),
        "high_conc_pct": float(high_conc_pct),
        "stale_news_hours": float(stale_news_h),
        "long_hold_days": float(long_hold_d),
        "stale_mark_hours": float(stale_mark_h),
        "weights": dict(weights),
        "rankings": [],
        "top_exit": None,
        "headline": "Exit priority: insufficient data to rank.",
    }

    tv = _safe_float(total_value)
    if tv is None or tv <= 0:
        out["headline"] = _headline("NO_DATA", [])
        return out
    out["total_value"] = tv

    if not open_positions:
        out["state"] = "ALL_CASH"
        out["headline"] = _headline("ALL_CASH", [])
        return out

    stock_positions: list[dict] = []
    for p in open_positions:
        if not isinstance(p, dict):
            continue
        if _is_option(p):
            continue
        tk = (p.get("ticker") or "").strip().upper()
        if not tk:
            continue
        qty = _safe_float(p.get("qty"))
        if qty is None or qty <= 0:
            continue
        stock_positions.append({**p, "_ticker": tk, "_qty": qty})

    out["n_positions"] = len(stock_positions)
    if not stock_positions:
        out["state"] = "ALL_CASH"
        out["headline"] = _headline("ALL_CASH", [])
        return out

    news_map = news_counts_by_ticker or {}

    rankings: list[dict] = []
    n_unmarked = 0
    for p in stock_positions:
        tk = p["_ticker"]
        qty = p["_qty"]
        cur_price_raw = _safe_float(p.get("current_price"))
        avg_cost = _safe_float(p.get("avg_cost")) or 0.0
        unrealized_pl = _safe_float(p.get("unrealized_pl"))
        # current_price is reset to 0 on every upsert and re-marked by
        # strategy._portfolio_snapshot before the next read. A leftover 0 is
        # NOT a -100% loss — it's an unmarked row. Skip the pnl_factor and
        # weight_pct contributions on those (silence_factor + age_factor
        # still rank fine on an unmarked open position).
        if cur_price_raw is None or cur_price_raw <= 0:
            cur_price = 0.0
            market_value = 0.0
            unrealized_pl_pct = None
            unmarked = True
            n_unmarked += 1
        else:
            cur_price = cur_price_raw
            market_value = qty * cur_price
            unrealized_pl_pct = (cur_price - avg_cost) / avg_cost * 100.0 \
                if avg_cost > 0 else None
            unmarked = False
        weight_pct = (market_value / tv * 100.0) if tv > 0 else 0.0

        opened_at = _parse_ts(p.get("opened_at"))
        days_held: float | None = None
        if opened_at is not None:
            secs = (now - opened_at).total_seconds()
            if secs >= 0:
                days_held = round(secs / 86400.0, 4)
            else:
                days_held = 0.0

        # Mark freshness — use ``last_updated`` from position row if present
        # (Store sets `current_price=0, last_updated=...` on upsert). Otherwise
        # the portfolio last_updated is the closest proxy but isn't per-row;
        # we leave mark_age_h as None when unavailable.
        mark_updated = p.get("mark_updated") or p.get("last_updated")
        mark_ts = _parse_ts(mark_updated)
        mark_age_h: float | None = None
        if mark_ts is not None:
            mark_age_h = (now - mark_ts).total_seconds() / 3600.0

        nc = news_map.get(tk) or news_map.get(tk.lower()) or {}
        news_silence_h = _safe_float(nc.get("hours_since_last"))
        n_articles_recent = nc.get("n_articles")
        try:
            n_articles_recent_i = int(n_articles_recent) if n_articles_recent is not None else 0
        except (TypeError, ValueError):
            n_articles_recent_i = 0

        c_f = _concentration_factor(weight_pct, high_conc_pct)
        p_f = _pnl_factor(unrealized_pl_pct)
        a_f = _age_factor(days_held, long_hold_d)
        s_f = _silence_factor(news_silence_h, stale_news_h)

        composite = (
            c_f * weights.get("concentration", DEFAULT_WEIGHTS["concentration"])
            + p_f * weights.get("pnl", DEFAULT_WEIGHTS["pnl"])
            + a_f * weights.get("age", DEFAULT_WEIGHTS["age"])
            + s_f * weights.get("silence", DEFAULT_WEIGHTS["silence"])
        ) * 100.0

        stale_mark_flag = _stale_mark(mark_age_h, stale_mark_h)
        if stale_mark_flag:
            composite -= stale_mark_penalty
        composite = max(0.0, min(100.0, composite))

        reasons = _explain(
            factors={
                "concentration": c_f, "pnl": p_f, "age": a_f, "silence": s_f,
            },
            weight_pct=weight_pct, pnl_pct=unrealized_pl_pct,
            days_held=days_held, news_silence_h=news_silence_h,
            stale_mark_penalized=stale_mark_flag,
            high_conc_pct=high_conc_pct, stale_news_h=stale_news_h,
            long_hold_d=long_hold_d,
        )

        if unmarked and "unmarked (current_price=0)" not in reasons:
            reasons.append("unmarked (current_price=0)")
        rankings.append({
            "ticker": tk,
            "score": round(composite, 2),
            "weight_pct": round(weight_pct, 3),
            "market_value": round(market_value, 4),
            "qty": qty,
            "avg_cost": avg_cost,
            "current_price": cur_price,
            "unmarked": unmarked,
            "unrealized_pl": unrealized_pl,
            "unrealized_pl_pct": round(unrealized_pl_pct, 3) if unrealized_pl_pct is not None else None,
            "days_held": days_held,
            "news_silence_hours": news_silence_h,
            "n_articles_recent": n_articles_recent_i,
            "mark_age_hours": round(mark_age_h, 2) if mark_age_h is not None else None,
            "stale_mark": stale_mark_flag,
            "factors": {
                "concentration": round(c_f, 4),
                "pnl": round(p_f, 4),
                "age": round(a_f, 4),
                "silence": round(s_f, 4),
            },
            "reasons": reasons,
        })

    rankings.sort(key=lambda r: (-r["score"], r["ticker"]))

    out["state"] = "OK"
    out["n_ranked"] = len(rankings)
    out["n_unmarked"] = n_unmarked
    out["rankings"] = rankings
    out["top_exit"] = rankings[0]["ticker"] if rankings else None
    out["headline"] = _headline("OK", rankings)
    return out
