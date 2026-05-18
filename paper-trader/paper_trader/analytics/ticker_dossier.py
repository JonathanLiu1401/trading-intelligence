"""One-ticker dossier — the cross-system drill-down a trader actually wants.

Inspecting a single name today means stitching three surfaces together by
hand: the open position + marks (paper_trader.db), the closed round-trip
history (also paper_trader.db, via ``round_trips``), the Opus reasoning that
touched the name (``decisions``), and the live news flow for it
(digital-intern's articles.db, read through ``signals``). No endpoint fused
them, so ``/api/ticker/<sym>`` and the ``/ticker/<sym>`` page did not exist.

``build_ticker_dossier`` is the single source of truth behind both. It is a
**pure function over already-fetched data** — no DB handle, no network, no
yfinance — so the endpoint stays off the hot path (it composes only stored
marks + read-only sqlite the ``signals`` layer already self-degrades on) and
the arithmetic is unit-testable without Flask.

Contract: pure, **never raises**. Any malformed row degrades that row to a
skip, never an exception — a drill-down panel must fail soft like every other
analytics module here.
"""
from __future__ import annotations

from datetime import datetime, timezone

from .round_trips import build_round_trips


def _f(v) -> float | None:
    """Best-effort float; None on anything non-numeric (never raises)."""
    try:
        if v is None or isinstance(v, bool):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def articles_mentioning(symbol: str, signals_list) -> list[dict]:
    """Articles from a ``signals.get_top_signals()``-shaped list that mention
    ``symbol``. Matching is on the already-extracted ``tickers`` field (the
    same ``_extract_tickers`` SSOT the live trader uses) so this never
    re-implements ticker parsing. Pure; tolerates missing keys.
    """
    sym = (symbol or "").upper().strip()
    if not sym:
        return []
    out = []
    for a in signals_list or []:
        if not isinstance(a, dict):
            continue
        tickers = a.get("tickers") or []
        try:
            hit = sym in {str(t).upper() for t in tickers}
        except TypeError:
            hit = False
        if hit:
            out.append(a)
    return out


def build_ticker_dossier(symbol: str, *, positions, trades, decisions,
                          signals_list, sentiment, parse_action_ticker,
                          now: datetime | None = None) -> dict:
    """Fuse position + realized history + Opus decisions + news for one name.

    Args (all already fetched by the caller — keeps this pure):
      ``positions``   — ``store.open_positions()`` rows
      ``trades``      — ``store.recent_trades(N)`` rows (full list; round-trips
                        are computed over all of it then filtered to ``symbol``
                        so partial-lot accounting stays correct)
      ``decisions``   — ``store.recent_decisions(N)`` rows
      ``signals_list``— ``signals.get_top_signals()`` rows
      ``sentiment``   — ``signals.get_ticker_sentiment(symbol)`` dict
      ``parse_action_ticker`` — the canonical ``dashboard._parse_action_ticker``
                        (verb, ticker) extractor, injected so decision→ticker
                        mapping has exactly one implementation (CLAUDE.md
                        invariant #11). Tests pass a tiny stand-in.

    Returns a JSON-ready dict; ``held``/``position`` describe the live lot(s),
    ``realized`` the closed P&L for this name only, ``news`` the live flow.
    """
    sym = (symbol or "").upper().strip()
    now = now or datetime.now(timezone.utc)

    # ── live position(s): stock + any option legs on this ticker ──────────
    legs: list[dict] = []
    for p in positions or []:
        if not isinstance(p, dict):
            continue
        if str(p.get("ticker") or "").upper() != sym:
            continue
        qty = _f(p.get("qty")) or 0.0
        if qty <= 0:
            continue
        legs.append({
            "type": p.get("type") or "stock",
            "qty": qty,
            "avg_cost": _f(p.get("avg_cost")),
            "current_price": _f(p.get("current_price")),
            "unrealized_pl": _f(p.get("unrealized_pl")),
            "strike": _f(p.get("strike")),
            "expiry": p.get("expiry"),
            "opened_at": p.get("opened_at"),
        })
    held = bool(legs)
    unrealized_total = round(sum((l["unrealized_pl"] or 0.0) for l in legs), 2) if held else 0.0

    # ── closed round-trip history, this name only ────────────────────────
    try:
        all_rts = build_round_trips(trades or [])
    except Exception:
        all_rts = []
    rts = [rt for rt in all_rts if str(rt.get("ticker") or "").upper() == sym]
    wins = [rt for rt in rts if (_f(rt.get("pnl_usd")) or 0.0) > 0]
    losses = [rt for rt in rts if (_f(rt.get("pnl_usd")) or 0.0) < 0]
    total_pnl = round(sum((_f(rt.get("pnl_usd")) or 0.0) for rt in rts), 2)
    holds = [_f(rt.get("hold_days")) for rt in rts if _f(rt.get("hold_days")) is not None]
    realized = {
        "n_round_trips": len(rts),
        "n_wins": len(wins),
        "n_losses": len(losses),
        "win_rate_pct": round(100.0 * len(wins) / len(rts), 1) if rts else None,
        "total_pnl_usd": total_pnl,
        "avg_hold_days": round(sum(holds) / len(holds), 2) if holds else None,
    }

    # ── Opus decision trail for this name (canonical parser injected) ─────
    decision_trail: list[dict] = []
    for d in decisions or []:
        if not isinstance(d, dict):
            continue
        try:
            verb, tk = parse_action_ticker(d.get("action_taken") or "")
        except Exception:
            continue
        if not tk or tk.upper() != sym:
            continue
        decision_trail.append({
            "timestamp": d.get("timestamp"),
            "verb": verb,
            "action_taken": d.get("action_taken"),
            "reasoning": (d.get("reasoning") or "")[:600],
        })
    decision_trail = decision_trail[:15]

    # ── recent ledger rows for this name ─────────────────────────────────
    tk_trades = []
    for t in trades or []:
        if not isinstance(t, dict):
            continue
        if str(t.get("ticker") or "").upper() != sym:
            continue
        tk_trades.append({
            "timestamp": t.get("timestamp"),
            "action": t.get("action"),
            "qty": _f(t.get("qty")),
            "price": _f(t.get("price")),
            "value": _f(t.get("value")),
            "reason": (t.get("reason") or "")[:300],
        })
    tk_trades = tk_trades[:20]

    # ── live news flow ───────────────────────────────────────────────────
    arts = articles_mentioning(sym, signals_list)
    news_articles = [{
        "title": a.get("title"),
        "source": a.get("source"),
        "ai_score": _f(a.get("ai_score")),
        "urgency": a.get("urgency"),
        "first_seen": a.get("first_seen"),
        "url": a.get("url"),
        "summary": (a.get("summary") or "")[:280],
    } for a in arts[:15]]
    sent = sentiment if isinstance(sentiment, dict) else {}

    in_watch_universe = held or bool(rts) or bool(decision_trail) or bool(tk_trades) or bool(news_articles)

    return {
        "symbol": sym,
        "generated_at": now.isoformat(timespec="seconds"),
        "held": held,
        "position": {
            "legs": legs,
            "unrealized_pl_total": unrealized_total,
        } if held else None,
        "realized": realized,
        "round_trips": rts[:25],
        "decisions": decision_trail,
        "trades": tk_trades,
        "news": {
            "sentiment": {
                "avg_score": _f(sent.get("avg_score")) or 0.0,
                "max_score": _f(sent.get("max_score")) or 0.0,
                "n": sent.get("n") or 0,
                "urgent": sent.get("urgent") or 0,
            },
            "articles": news_articles,
        },
        # False ⇒ nothing on file anywhere: the page shows a clean "no
        # coverage" state instead of an error, so a typo'd ticker is obvious.
        "has_coverage": bool(in_watch_universe),
    }
