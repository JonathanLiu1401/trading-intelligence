"""Initiation drought — how long since the bot opened a position on a NEW ticker?

The existing drought / concentration / churn analytics each describe a different
shape of inactivity:

  * ``/api/decision-drought``           — NO_DECISION pace + the $ alpha it costs.
  * ``/api/concentration-trajectory``   — $-HHI of currently-OPEN positions.
  * ``/api/churn``                      — round-trip turnover + fast re-entry.
  * ``/api/concentration-cap``          — single-name cap pressure on OPEN sizing.

None of them answer the **watchlist-exploration** question the operator
actually has when they look at a trade ledger of "13 trades, all NVDA + TQQQ
out of a 50-ticker watchlist over six days":

  **"Is the bot still *initiating* new ideas, or has it stopped exploring
  the watchlist and become obsessed with two names?"**

A *BUY* is an **initiation** if the ticker has not been bought before in the
visible trade ledger; otherwise it's a **re-cycle**. The metric this surfaces
is the time since the last initiation, the count of re-cycles accumulated
since, and the share of the watchlist ever touched.

Pure builder, ``store.recent_trades`` shape: feed it the trades list
(newest-first, as ``store.recent_trades()`` returns) and optionally the live
``WATCHLIST`` to compute coverage. ``now`` is injectable for tests. Never
raises; garbage input → ``NO_DATA``. **Advisory only** — observational, no
caps, never gates Opus, has no path to ``_execute()`` (AGENTS.md invariants
#2/#12 — the ``no_decision_recovery`` precedent).

Run as a CLI for a one-shot ops view::

    python3 -m paper_trader.analytics.initiation_drought
"""
from __future__ import annotations

from datetime import datetime, timezone

# Stuck-on-names threshold: ≤ this many distinct tickers ever bought AND a
# decent number of total buys (so a fresh portfolio isn't flagged STUCK after
# its second trade).
_STUCK_MAX_DISTINCT = 2
_STUCK_MIN_BUYS = 10

# Re-cycling: many re-cycles have piled up since the last new-ticker entry
# AND that last entry is old enough to be a real cooling-off, not a one-day
# focused run.
_RECYCLING_MIN_RECYCLES = 5
_RECYCLING_MIN_HOURS = 48.0

# Exploring: very recent net-new initiation — the bot just opened a fresh
# name; whatever else looks bad in the history, it isn't stuck *right now*.
_EXPLORING_MAX_HOURS = 24.0

# Below this many BUYs we can't say anything meaningful (could legitimately
# be a 3-trade-old account that's still ramping up its book).
_INSUFFICIENT_HISTORY = 5


def _parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        s = ts.replace("Z", "+00:00") if isinstance(ts, str) else ""
        d = datetime.fromisoformat(s)
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except (ValueError, AttributeError, TypeError):
        return None


def _is_buy(action: str | None) -> bool:
    """A position-opening action. Accepts equity BUY plus options BUY_CALL /
    BUY_PUT — anything that *opens* exposure to a ticker counts toward
    initiation tracking. SELL / SELL_CALL / SELL_PUT close exposure and are
    excluded. Mirrors ``strategy._execute`` action vocabulary."""
    if not action:
        return False
    a = action.strip().upper()
    return a in ("BUY", "BUY_CALL", "BUY_PUT")


def build_initiation_drought(
    trades: list[dict] | None,
    watchlist: list[str] | None = None,
    now: datetime | None = None,
) -> dict:
    """Run-length encode the BUY ledger into initiations vs re-cycles.

    ``trades`` should be the ``store.recent_trades()`` shape (newest-first).
    ``watchlist`` is optional — when provided, watchlist coverage % is computed
    as ``distinct_tickers_ever / len(set(watchlist))``. ``now`` is injectable
    for tests; defaults to ``datetime.now(timezone.utc)``.

    Result shape (always present, never raises):

      * ``state``                       — ``"NO_DATA"`` if no trades; else ``"OK"``.
      * ``total_trades``                — count of trades considered.
      * ``total_buys``                  — buys (BUY/BUY_CALL/BUY_PUT) only.
      * ``total_initiations``           — first-ever buys per ticker.
      * ``total_recycles``              — buys whose ticker was bought before.
      * ``recycle_rate``                — recycles / total_buys (0.0..1.0).
      * ``distinct_tickers_ever``       — count of distinct buy-tickers.
      * ``distinct_tickers``            — sorted list of those tickers.
      * ``last_initiation_ticker``      — ticker of the most recent initiation.
      * ``last_initiation_ts``          — ISO timestamp of that initiation.
      * ``hours_since_last_initiation`` — hours since last initiation (None if no buys).
      * ``recycles_since_last_initiation`` — buys after the last initiation, all on prior tickers.
      * ``watchlist_size``              — len(set(watchlist)) or None.
      * ``watchlist_coverage_pct``      — % of unique watchlist tickers ever bought, or None.
      * ``watchlist_unseen``            — list of watchlist tickers never bought, capped at 20.
      * ``verdict``                     — see ladder below.
      * ``verdict_detail``              — one-line explanation suitable for chat.
      * ``headline``                    — dashboard / Discord one-liner.

    Verdict ladder:

      * ``NO_DATA``               — no trades / no buys at all.
      * ``INSUFFICIENT_HISTORY``  — < ``_INSUFFICIENT_HISTORY`` buys.
      * ``EXPLORING``             — last initiation < ``_EXPLORING_MAX_HOURS`` ago.
      * ``STUCK_ON_NAMES``        — ≤ ``_STUCK_MAX_DISTINCT`` distinct tickers AND
                                    ≥ ``_STUCK_MIN_BUYS`` total buys.
      * ``RECYCLING``             — ≥ ``_RECYCLING_MIN_RECYCLES`` re-cycles
                                    accumulated AND last initiation
                                    > ``_RECYCLING_MIN_HOURS`` hours ago.
      * ``STEADY``                — none of the above; the explore/recycle mix
                                    is within typical bounds.
    """
    now = now or datetime.now(timezone.utc)

    rows = list(trades or [])
    if not rows:
        return _no_data(watchlist)

    # Walk OLDEST → NEWEST so "initiation vs re-cycle" is a first-occurrence
    # decision, not a last-occurrence one. ``store.recent_trades`` returns
    # newest-first, so we reverse.
    chrono = list(reversed(rows))

    seen: set[str] = set()
    total_buys = 0
    total_inits = 0
    last_init_ticker: str | None = None
    last_init_ts_obj: datetime | None = None
    last_init_ts_str: str | None = None
    recycles_since_last_init = 0

    for t in chrono:
        if not _is_buy(t.get("action")):
            continue
        ticker = (t.get("ticker") or "").strip().upper()
        if not ticker:
            continue
        total_buys += 1
        if ticker not in seen:
            seen.add(ticker)
            total_inits += 1
            last_init_ticker = ticker
            ts = t.get("timestamp")
            last_init_ts_obj = _parse_ts(ts)
            last_init_ts_str = ts if isinstance(ts, str) else None
            recycles_since_last_init = 0
        else:
            recycles_since_last_init += 1

    total_recycles = total_buys - total_inits
    recycle_rate = (total_recycles / total_buys) if total_buys else 0.0
    distinct_list = sorted(seen)

    hours_since: float | None
    if last_init_ts_obj is not None:
        hours_since = max(0.0, (now - last_init_ts_obj).total_seconds() / 3600.0)
    else:
        hours_since = None

    # Watchlist coverage (optional).
    wl_size: int | None = None
    wl_cov_pct: float | None = None
    wl_unseen: list[str] = []
    if watchlist:
        wl_set = {(w or "").strip().upper() for w in watchlist if w}
        wl_set.discard("")
        wl_size = len(wl_set)
        if wl_size:
            wl_cov_pct = round(100.0 * len(seen & wl_set) / wl_size, 2)
            wl_unseen = sorted(wl_set - seen)[:20]

    # ── verdict ──────────────────────────────────────────────────
    if total_buys == 0:
        verdict = "NO_DATA"
        verdict_detail = f"{len(rows)} trades on record but none are buys — no initiation history"
    elif total_buys < _INSUFFICIENT_HISTORY:
        verdict = "INSUFFICIENT_HISTORY"
        verdict_detail = (
            f"only {total_buys} buy(s) on record (need ≥ {_INSUFFICIENT_HISTORY}) — "
            "too early to assess exploration"
        )
    elif hours_since is not None and hours_since < _EXPLORING_MAX_HOURS:
        verdict = "EXPLORING"
        verdict_detail = (
            f"just opened {last_init_ticker} {hours_since:.1f}h ago — "
            f"actively exploring ({total_inits} distinct names ever)"
        )
    elif total_inits <= _STUCK_MAX_DISTINCT and total_buys >= _STUCK_MIN_BUYS:
        names = ", ".join(distinct_list) if distinct_list else "—"
        verdict = "STUCK_ON_NAMES"
        verdict_detail = (
            f"{total_buys} buys but only {total_inits} distinct ticker(s) "
            f"({names}) — the bot has stopped exploring the watchlist"
        )
    elif (
        recycles_since_last_init >= _RECYCLING_MIN_RECYCLES
        and hours_since is not None
        and hours_since > _RECYCLING_MIN_HOURS
    ):
        verdict = "RECYCLING"
        verdict_detail = (
            f"{recycles_since_last_init} re-cycle buy(s) accumulated since the "
            f"last new name ({last_init_ticker}, {hours_since:.1f}h ago) — "
            "exploration has stalled"
        )
    else:
        verdict = "STEADY"
        if hours_since is None:
            verdict_detail = "no completed initiations yet — too early to grade"
        else:
            verdict_detail = (
                f"{total_inits} distinct names ever, last new entry "
                f"{last_init_ticker} {hours_since:.1f}h ago — exploration mix typical"
            )

    # ── headline ─────────────────────────────────────────────────
    cov_str = (
        f", watchlist coverage {wl_cov_pct:.0f}%" if wl_cov_pct is not None else ""
    )
    if hours_since is None:
        headline = (
            f"{total_buys} buys, {total_inits} distinct names "
            f"({recycle_rate*100:.0f}% re-cycle){cov_str} — {verdict}"
        )
    else:
        headline = (
            f"{total_buys} buys, {total_inits} distinct names "
            f"({recycle_rate*100:.0f}% re-cycle); last new entry "
            f"{last_init_ticker} {hours_since:.1f}h ago{cov_str} — {verdict}"
        )

    return {
        "state": "OK",
        "total_trades": len(rows),
        "total_buys": total_buys,
        "total_initiations": total_inits,
        "total_recycles": total_recycles,
        "recycle_rate": round(recycle_rate, 4),
        "distinct_tickers_ever": len(seen),
        "distinct_tickers": distinct_list,
        "last_initiation_ticker": last_init_ticker,
        "last_initiation_ts": last_init_ts_str,
        "hours_since_last_initiation": (
            round(hours_since, 2) if hours_since is not None else None
        ),
        "recycles_since_last_initiation": recycles_since_last_init,
        "watchlist_size": wl_size,
        "watchlist_coverage_pct": wl_cov_pct,
        "watchlist_unseen": wl_unseen,
        "verdict": verdict,
        "verdict_detail": verdict_detail,
        "headline": headline,
    }


def _no_data(watchlist: list[str] | None) -> dict:
    wl_size: int | None = None
    if watchlist:
        wl_set = {(w or "").strip().upper() for w in watchlist if w}
        wl_set.discard("")
        wl_size = len(wl_set) or None
    return {
        "state": "NO_DATA",
        "total_trades": 0,
        "total_buys": 0,
        "total_initiations": 0,
        "total_recycles": 0,
        "recycle_rate": 0.0,
        "distinct_tickers_ever": 0,
        "distinct_tickers": [],
        "last_initiation_ticker": None,
        "last_initiation_ts": None,
        "hours_since_last_initiation": None,
        "recycles_since_last_initiation": 0,
        "watchlist_size": wl_size,
        "watchlist_coverage_pct": None,
        "watchlist_unseen": [],
        "verdict": "NO_DATA",
        "verdict_detail": "no trades on record — no initiation history",
        "headline": "no trades on record — cannot assess watchlist exploration",
    }


def _main() -> int:
    """One-shot CLI."""
    import json
    from ..store import get_store
    try:
        from ..strategy import WATCHLIST
    except Exception:
        WATCHLIST = None
    out = build_initiation_drought(get_store().recent_trades(2000), watchlist=WATCHLIST)
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
