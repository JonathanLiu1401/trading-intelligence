"""News-to-trade attribution — *which articles plausibly preceded each fill?*

Every other analytics panel in this repo answers the symmetric "did past
signals predict past trades **across the book**?" question
(``news_edge``, ``source_edge``, ``signal_followthrough``,
``scorer_attribution``). What was missing — and is in the ``decisions``
table only as the opaque ``reasoning`` free-text blob — was the per-fill
audit: when the bot bought NVDA at 14:23, **which** of the articles in
the prior signal window plausibly drove the call?

This is a *post-hoc* / *implied* attribution: the chat-or-prompt context
the bot actually saw is not stored row-by-row, so we cannot literally
reconstruct it. What we *can* reconstruct, deterministically and from
``paper_trader.db`` + ``articles.db`` alone, is the **news landscape
immediately before the fill** — the highest-scored, live-only,
ticker-mentioning articles in a fixed pre-trade window. That is the
honest read every audit-trail panel in this codebase already mirrors
(``loser_autopsy`` / ``winner_autopsy`` / ``decision_context``).

Design parity with the codebase:

* **Pure builder; the I/O lives in the endpoint.** The same
  ``thesis_drift`` / ``correlation`` split — the route fetches trades and
  articles, the builder is offline and deterministically testable.
* **Live-only by construction.** The route applies the canonical
  ``_LIVE_ONLY_CLAUSE`` SQL fragment (invariant #1) when reading
  ``articles.db``; backtest:// rows never reach the attribution panel.
* **Stocks AND options.** The same ticker-mention match works on both —
  options carry an ``underlying`` ticker the news will name. Trades with
  pseudo-tickers (``CASH``, ``NONE``, ``NO_DECISION``, ``BLOCKED``) are
  silently dropped, mirroring ``dashboard._parse_action_ticker`` /
  invariant #11.
* **Sample-size honesty.** A trade with zero matching articles in the
  window emits ``attributed_articles: []`` and ``n_attributed: 0`` — never
  a fabricated "top article" filler. ``state`` flips to ``NO_DATA`` only
  when the entire trade list is empty.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

# Default lookback window for articles preceding a trade. Long enough to
# catch a morning headline preceding a midday fill, short enough that
# unrelated news between cycles doesn't dilute the top match.
DEFAULT_WINDOW_HOURS = 4.0
# Cap on attributed articles per trade. The block is "top movers" not "all
# context" — a long list buries the signal.
DEFAULT_MAX_PER_TRADE = 3
# Drop articles below this ai_score before attribution. Mirrors signals.py
# `min_score=4.0` threshold for what the live trader considers a real
# signal; using the same cutoff keeps the panel honest about WHAT WOULD
# HAVE BEEN IN THE PROMPT, not the dregs of the article stream.
DEFAULT_MIN_AI_SCORE = 2.0
# Pseudo-tickers that the live trader writes when nothing happened — same
# carve-out the dashboard's _parse_action_ticker uses (invariant #11). An
# attribution row keyed on "CASH" or "NO_DECISION" would be meaningless.
_PSEUDO_TICKERS = {"CASH", "NONE", "NO_DECISION", "BLOCKED", ""}


def _parse_iso(ts) -> datetime | None:
    """Parse the ISO timestamp shape that ``paper_trader.db`` and
    ``articles.db`` both write. Both writers normalize to UTC, but the
    historical mix in ``articles.first_seen`` contains a few RFC822
    fallbacks — return ``None`` on those rather than raising."""
    if not isinstance(ts, str) or not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _mentions_ticker(text: str, ticker: str) -> bool:
    """Case-insensitive ticker mention in the article title (or summary).

    A ticker like ``MU`` is short; bare substring match would alias
    ``MUTUAL FUND``. Word-boundary regex (``\\bMU\\b``) cleanly separates
    those — ``MUTUAL`` has no non-word char between ``MU`` and ``TUAL``,
    so the boundary doesn't fire. ``$MU`` mentions (cashtag) match too —
    the ``$`` is non-word, so the leading boundary is satisfied.
    """
    if not text or not ticker:
        return False
    return re.search(rf"\b{re.escape(ticker)}\b", text, re.IGNORECASE) is not None


def build_trade_attribution(
    trades: list[dict],
    articles: list[dict],
    *,
    window_hours: float = DEFAULT_WINDOW_HOURS,
    max_per_trade: int = DEFAULT_MAX_PER_TRADE,
    min_ai_score: float = DEFAULT_MIN_AI_SCORE,
    now: datetime | None = None,
) -> dict:
    """Per-trade article attribution. Pure, never raises.

    ``trades`` — recent FILLED trades, each
    ``{id, timestamp, ticker, action, qty, price, value?, reason?, type?}``.
    ``id`` and ``timestamp`` must be present for a trade to be considered;
    pseudo-tickers (``CASH`` / ``NONE`` / ``NO_DECISION`` / ``BLOCKED`` /
    blank) are silently dropped — they are not real fills.

    ``articles`` — already live-only-filtered (caller applied the
    invariant #1 SQL fragment); each
    ``{title, url?, source?, ai_score, urgency?, first_seen}``. Below
    ``min_ai_score`` are dropped before matching so the panel reflects
    what would have been in the prompt, not the long tail of the stream.

    Returns a dict with ``state`` (``OK`` / ``NO_DATA``), the parameters
    used, and per-trade rows. A trade with zero matching articles still
    appears (``n_attributed: 0``) so the analyst sees the negative space.
    """
    now = now or datetime.now(timezone.utc)
    window = timedelta(hours=max(0.0, float(window_hours)))
    max_per_trade = max(0, int(max_per_trade))

    # Pre-process articles once: parse first_seen, drop low-score and
    # unparseable timestamps. Stable input → deterministic order in the
    # per-trade match below.
    pruned: list[tuple[datetime, dict]] = []
    for a in articles or []:
        if not isinstance(a, dict):
            continue
        try:
            score = float(a.get("ai_score") or 0.0)
        except (TypeError, ValueError):
            continue
        if score < min_ai_score:
            continue
        dt = _parse_iso(a.get("first_seen"))
        if dt is None:
            continue
        pruned.append((dt, a))
    n_articles_examined = len(pruned)

    out_trades: list[dict] = []
    skipped_pseudo = 0
    for t in trades or []:
        if not isinstance(t, dict):
            continue
        ticker_raw = t.get("ticker")
        ticker = str(ticker_raw).upper().strip() if ticker_raw is not None else ""
        if ticker in _PSEUDO_TICKERS:
            skipped_pseudo += 1
            continue
        trade_ts = _parse_iso(t.get("timestamp"))
        if trade_ts is None:
            continue

        cutoff = trade_ts - window
        matches: list[dict] = []
        for art_ts, art in pruned:
            # Article must be in [trade - window, trade]. An article first_seen
            # AFTER the fill cannot have driven it; the strict <= guards
            # against the same-second tie where ingest and fill share a clock.
            if art_ts < cutoff or art_ts > trade_ts:
                continue
            title = str(art.get("title") or "")
            if not _mentions_ticker(title, ticker):
                continue
            minutes_before = round(
                (trade_ts - art_ts).total_seconds() / 60.0, 1)
            try:
                score = float(art.get("ai_score") or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            matches.append({
                "title": title,
                "url": art.get("url"),
                "source": art.get("source"),
                "ai_score": round(score, 3),
                "urgency": int(art.get("urgency") or 0),
                "first_seen": art.get("first_seen"),
                "minutes_before_trade": minutes_before,
            })

        # Highest ai_score first; ties broken by recency (closer to trade
        # is more plausibly causal). Deterministic.
        matches.sort(key=lambda m: (-m["ai_score"], m["minutes_before_trade"]))
        attributed = matches[:max_per_trade]

        # Cosmetic top-line for the per-trade row — the panel's "headline".
        if attributed:
            top = attributed[0]
            headline = (
                f"{t.get('action', '?')} {ticker}: "
                f"top article ai_score={top['ai_score']} "
                f"({top['minutes_before_trade']:.0f}min before fill) — "
                f"{top['title'][:120]}"
            )
        else:
            headline = (
                f"{t.get('action', '?')} {ticker}: no live-only article "
                f"mentioning {ticker} in the {window_hours:.1f}h before "
                f"the fill (above ai_score≥{min_ai_score})."
            )

        out_trades.append({
            "id": t.get("id"),
            "ticker": ticker,
            "action": t.get("action"),
            "qty": t.get("qty"),
            "price": t.get("price"),
            "timestamp": t.get("timestamp"),
            "n_attributed": len(attributed),
            "n_candidates": len(matches),
            "headline": headline,
            "attributed_articles": attributed,
        })

    # Newest fills first — what the operator scrolls to.
    out_trades.sort(
        key=lambda r: _parse_iso(r.get("timestamp")) or datetime.min.replace(
            tzinfo=timezone.utc),
        reverse=True,
    )

    state = "OK" if out_trades else "NO_DATA"
    return {
        "as_of": now.isoformat(timespec="seconds"),
        "state": state,
        "window_hours": float(window_hours),
        "min_ai_score": float(min_ai_score),
        "max_per_trade": int(max_per_trade),
        "n_trades": len(out_trades),
        "n_articles_examined": n_articles_examined,
        "n_skipped_pseudo_ticker": skipped_pseudo,
        "trades": out_trades,
    }


if __name__ == "__main__":  # smoke test against the live DBs
    import json
    import sqlite3
    import zlib
    from pathlib import Path

    from paper_trader.store import get_store

    # Recent trades from paper_trader.db
    store = get_store()
    with store._lock:
        cur = store._conn.execute(
            "SELECT id, timestamp, ticker, action, qty, price, value "
            "FROM trades WHERE timestamp >= datetime('now', '-12 hours') "
            "ORDER BY timestamp DESC LIMIT 50"
        )
        rows = cur.fetchall()
    trades = [{"id": r[0], "timestamp": r[1], "ticker": r[2],
               "action": r[3], "qty": r[4], "price": r[5], "value": r[6]}
              for r in rows]

    # Live-only articles in the relevant window
    di = Path("/media/zeph/projects/digital-intern/db/articles.db")
    if not di.exists():
        di = Path("/home/zeph/digital-intern/data/articles.db")
    arts: list[dict] = []
    if di.exists() and trades:
        conn = sqlite3.connect(f"file:{di}?mode=ro", uri=True)
        rows = conn.execute(
            "SELECT title, url, source, ai_score, urgency, first_seen "
            "FROM articles WHERE first_seen >= datetime('now','-16 hours') "
            "AND ai_score >= 2.0 "
            "AND url NOT LIKE 'backtest://%' "
            "AND source NOT LIKE 'backtest_%' "
            "AND source NOT LIKE 'opus_annotation%' "
            "ORDER BY ai_score DESC LIMIT 2000"
        ).fetchall()
        conn.close()
        arts = [{"title": r[0], "url": r[1], "source": r[2],
                 "ai_score": r[3], "urgency": r[4], "first_seen": r[5]}
                for r in rows]

    print(json.dumps(
        build_trade_attribution(trades, arts), indent=2, default=str))
