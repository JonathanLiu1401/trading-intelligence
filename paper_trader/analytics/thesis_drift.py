"""Entry-thesis vs current-reality scorecard for the *open* book.

Every panel today shows the *current* state of a holding. None re-tests
the position against **the reason it was opened for**. ``/api/suggestions``
re-classifies from scratch; ``/api/position-thesis`` fuses current
scorer + technicals + last decision. The single discipline question a
trader actually asks — *"is the thing I bought this for still true?"* —
is unanswered, even though the answer is sitting verbatim in
``trades.reason`` of the opening fill.

``build_thesis_drift`` anchors each open position on its **own opening
BUY rationale** (the trader's literal words) and places it next to a
deterministic, objective read of the present: P/L since entry, days
held, and — when the caller supplies live quant/news — whether the
MACD/momentum/RSI/news conditions a thesis typically rests on have since
turned. It then assigns a single ``health`` ∈ ``INTACT`` / ``WEAKENING``
/ ``BROKEN``.

Distinct from its neighbours:

* ``/api/position-thesis`` — fuses *current* signals into a fresh view.
* ``/api/suggestions`` — re-derives an action from scratch per ticker.
* ``build_thesis_drift`` — **delta vs the position's stated entry
  rationale**: shows the original reason unmodified and grades how far
  reality has drifted from it.

The verdict is driven only by objective, deterministic inputs (P/L,
hold time, supplied quant/news). The opening reason text is surfaced
**verbatim** and never parsed for trading logic — the one heuristic that
reads it is an explicitly-labelled, optional "entry cited a news
catalyst, none live now" note.

Invariant #8: ``positions.opened_at`` is reset to the re-entry time when
a fully-closed lot is reopened, so the opening fill of *this* lot is the
BUY whose timestamp sits closest to ``opened_at`` (a prior closed lot's
BUY is far earlier). Pure, network-free, advisory only — never gates
Opus, adds no caps (AGENTS.md #2/#12).
"""
from __future__ import annotations

from datetime import datetime, timezone

# Objective health thresholds (exact-value test-locked):
PAIN_PCT = -8.0   # P/L since entry at/below this ⇒ thesis materially wrong
WEAK_PCT = -3.0   # at/below this ⇒ thesis under pressure
RSI_HOT = 78.0    # overextended; mean-reversion risk against a long thesis

_NEWS_KWS = ("news", "urgent", "earnings", "catalyst", "headline",
             "beat", "guidance", "upgrade", "downgrade")


def _parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def _norm_key(ticker, typ, strike, expiry) -> tuple:
    # Mirror store.upsert_position's NULL-normalised match
    # (IFNULL(strike,0), IFNULL(expiry,'')) so stock (NULL strike/expiry)
    # and options compare correctly.
    return (str(ticker).upper(), typ or "stock",
            float(strike) if strike not in (None, "") else 0.0,
            expiry or "")


def _opening_trade(pos: dict, trades: list[dict]) -> dict | None:
    """The BUY fill that opened *this* lot.

    Among BUY-action trades on the same normalised key, the opener of the
    current lot is the one whose timestamp is nearest ``opened_at``
    (invariant #8). Ties → earliest. Returns None if the ledger has no
    matching BUY (e.g. trade history truncated below the open).
    """
    pkey = _norm_key(pos.get("ticker"), pos.get("type"),
                     pos.get("strike"), pos.get("expiry"))
    opened = _parse_ts(pos.get("opened_at"))
    best = None
    best_delta = None
    for t in trades:
        if not (t.get("action") or "").upper().startswith("BUY"):
            continue
        tkey = _norm_key(t.get("ticker"), t.get("option_type"),
                         t.get("strike"), t.get("expiry"))
        if tkey != pkey:
            continue
        tdt = _parse_ts(t.get("timestamp"))
        if opened is None or tdt is None:
            # No usable timestamp on either side — fall back to the first
            # matching BUY encountered (deterministic given ledger order).
            if best is None:
                best, best_delta = t, None
            continue
        delta = abs((tdt - opened).total_seconds())
        if (best_delta is None
                or delta < best_delta
                or (delta == best_delta and tdt < _parse_ts(best.get("timestamp")))):
            best, best_delta = t, delta
    return best


def build_thesis_drift(positions: list[dict],
                        trades: list[dict],
                        signals: dict | None = None,
                        now: datetime | None = None) -> dict:
    """Re-test every open position against its opening rationale. Pure.

    * ``positions`` — ``Store.open_positions()``-shaped (carries
      ``avg_cost``, ``current_price``, ``opened_at``, key fields).
    * ``trades`` — ``Store.recent_trades()``-shaped ledger; order does
      not matter (the opener is selected by timestamp proximity).
    * ``signals`` — optional ``{TICKER: {rsi, macd, mom_5d, mom_20d,
      news_count, news_urgent}}``. ``macd`` is the suggestions-endpoint
      string ("bullish"/"bearish"); absent ⇒ price-only health (the
      MACD/momentum drift checks degrade off, never error).
    """
    now = now or datetime.now(timezone.utc)
    sig_map = {k.upper(): v for k, v in (signals or {}).items()}
    cards: list[dict] = []
    counts = {"INTACT": 0, "WEAKENING": 0, "BROKEN": 0}

    for pos in positions:
        ticker = pos.get("ticker")
        avg = float(pos.get("avg_cost") or 0.0)
        cur = float(pos.get("current_price") or 0.0)
        pl_pct = round((cur - avg) / avg * 100.0, 2) if avg > 0 and cur > 0 else None

        opened_dt = _parse_ts(pos.get("opened_at"))
        days_held = (round((now - opened_dt).total_seconds() / 86400.0, 2)
                     if opened_dt is not None else None)

        otrade = _opening_trade(pos, trades)
        entry_reason = (otrade or {}).get("reason") or None
        entry_price = (float(otrade["price"]) if otrade
                       and otrade.get("price") is not None else (avg or None))

        sig = sig_map.get(str(ticker).upper(), {}) if ticker else {}
        signals_present = bool(sig)
        macd = sig.get("macd")
        macd_bear = isinstance(macd, str) and macd.lower().startswith("bear")
        mom5 = sig.get("mom_5d")
        mom5_neg = isinstance(mom5, (int, float)) and mom5 < 0
        rsi = sig.get("rsi")
        rsi_hot = isinstance(rsi, (int, float)) and rsi >= RSI_HOT
        ncount = sig.get("news_count")
        news_cold = isinstance(ncount, (int, float)) and ncount == 0
        entry_cited_news = bool(
            entry_reason and any(k in entry_reason.lower() for k in _NEWS_KWS))

        reasons: list[str] = []
        if pl_pct is not None:
            reasons.append(f"P/L since entry {pl_pct:+.2f}%")
        if macd_bear:
            reasons.append("MACD turned bearish")
        if mom5_neg:
            reasons.append(f"5d momentum {mom5:+.2f}% (negative)")
        if rsi_hot:
            reasons.append(f"RSI {rsi:.0f} — overextended")
        if entry_cited_news and news_cold:
            reasons.append(
                "entry cited a news catalyst; no live coverage now "
                "(heuristic)")

        # Deterministic precedence: BROKEN > WEAKENING > INTACT.
        if ((pl_pct is not None and pl_pct <= PAIN_PCT)
                or (macd_bear and mom5_neg
                    and pl_pct is not None and pl_pct < 0)):
            health = "BROKEN"
        elif ((pl_pct is not None and pl_pct <= WEAK_PCT)
              or macd_bear or mom5_neg or rsi_hot
              or (entry_cited_news and news_cold)):
            health = "WEAKENING"
        else:
            health = "INTACT"
            if not reasons:
                reasons.append(
                    "P/L, momentum and news consistent with the entry "
                    "thesis" if signals_present
                    else "no adverse price move since entry"
                    if pl_pct is not None
                    else "no objective drift signal available")
        counts[health] += 1

        cards.append({
            "ticker": ticker,
            "type": pos.get("type") or "stock",
            "strike": pos.get("strike"),
            "expiry": pos.get("expiry"),
            "qty": pos.get("qty"),
            "entry_ts": pos.get("opened_at"),
            "entry_price": (round(entry_price, 4)
                            if entry_price is not None else None),
            "entry_reason": entry_reason,
            "days_held": days_held,
            "current_price": cur or None,
            "pl_pct": pl_pct,
            "health": health,
            "drift_reasons": reasons,
            "signals_present": signals_present,
        })

    # Worst first: BROKEN, then WEAKENING, then INTACT; within a band the
    # most-negative P/L leads (None P/L sorts last within its band).
    order = {"BROKEN": 0, "WEAKENING": 1, "INTACT": 2}
    cards.sort(key=lambda c: (order[c["health"]],
                              c["pl_pct"] if c["pl_pct"] is not None else 1e9))

    n = len(cards)
    state = "NO_DATA" if n == 0 else "OK"
    if state == "NO_DATA":
        headline = "No open positions — no entry theses to re-test."
    else:
        worst = cards[0]
        parts = []
        if counts["BROKEN"]:
            parts.append(f"{counts['BROKEN']} broken")
        if counts["WEAKENING"]:
            parts.append(f"{counts['WEAKENING']} weakening")
        if counts["INTACT"]:
            parts.append(f"{counts['INTACT']} intact")
        lead = (f"{worst['ticker']} thesis {worst['health'].lower()}"
                + (f" ({worst['pl_pct']:+.1f}% since entry)"
                   if worst["pl_pct"] is not None else "")
                if worst["health"] != "INTACT"
                else "all open theses still intact")
        headline = f"{n} open position(s): {', '.join(parts)} — {lead}."

    return {
        "as_of": now.isoformat(timespec="seconds"),
        "state": state,
        "headline": headline,
        "n_positions": n,
        "counts": counts,
        "positions": cards,
        "note": ("Advisory only — re-tests each holding against its own "
                 "opening rationale; never gates Opus, imposes no caps "
                 "(AGENTS.md #2/#12)."),
    }


if __name__ == "__main__":  # smoke test against the live DB
    import json

    from paper_trader.store import get_store
    s = get_store()
    rep = build_thesis_drift(s.open_positions(), s.recent_trades(2000))
    print(json.dumps(rep, indent=2, default=str))
