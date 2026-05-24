"""Exit-only-streak detector: how many consecutive exits since the last entry.

A specific structural failure mode no existing panel covers: the bot keeps
SELLing positions to free cash but never re-deploys it, drifting into a
defensive "liquidation only" mode while signals may still warrant entries.
``streak`` reads W/L outcomes on closed round-trips; ``churn`` measures
re-entry cadence (entries+exits per ticker); ``cash_drag`` measures the cost
of idle cash. None of them surface the *trade direction* sequence at the
book level — "the last 6 fills were all SELLs; we have not opened a new lot
in 14h."

Single source of truth: consumes ``store.recent_trades`` (which already
filters to logged fills). No P&L recompute. ``BUY`` / ``BUY_CALL`` /
``BUY_PUT`` are entries; ``SELL`` / ``SELL_CALL`` / ``SELL_PUT`` are exits.
Any other action (REBALANCE, the historical "NO_DECISION" etc.) is skipped
— this analyzer only reads logged FILLs, never decision rows.

Verdict ladder (sample-size-honest — the ``streak`` precedent):

================================  =======================================
``NO_DATA``                       no FILLED entry-or-exit trades yet
``MOST_RECENT_IS_ENTRY``          newest fill is an entry — no exit run
``DEFENSIVE_TRIM``                ``DEFENSIVE_TRIM_MIN`` ≤ run < ``DEFENSIVE_LIQUIDATION_MIN``
``DEFENSIVE_LIQUIDATION``         run ≥ ``DEFENSIVE_LIQUIDATION_MIN``
================================  =======================================

A run of 1-2 exits is statistically meaningless (book turnover; a clean
take-profit followed by re-entry next cycle) — the verdict stays
``MOST_RECENT_IS_ENTRY`` cosmetically below the trim floor too, so the
hourly summary stays quiet in that healthy regime.

Advisory only — never gates Opus, never caps positions, never blocks a
trade. The hourly summary surfaces the headline; Opus reads it as
context, exactly like ``streak`` (AGENTS.md invariants #2 / #12).
"""
from __future__ import annotations

from datetime import datetime, timezone

ENTRY_ACTIONS = ("BUY", "BUY_CALL", "BUY_PUT")
EXIT_ACTIONS = ("SELL", "SELL_CALL", "SELL_PUT")

# Verdict thresholds. ``3`` is the smallest run that beats "two-in-a-row by
# chance" given the BUY/SELL flip rate of a healthy live book (≈1 entry per
# 2-3 fills historically). ``6`` is the run length at which a trader watching
# the live tape would say "this engine is in liquidation mode, not running
# the strategy" — actionable in either direction (intervention, or a
# deliberate "I know, the market is bad" acknowledgement).
DEFENSIVE_TRIM_MIN = 3
DEFENSIVE_LIQUIDATION_MIN = 6

# Recent-run window. We expose the last N trade-direction symbols so a
# trader can eyeball the BUY/SELL alternation rhythm — same shape as
# ``streak.recent_sequence``.
RECENT_SEQUENCE_LEN = 12


def _direction(action: str | None) -> str | None:
    """Return ``"ENTRY"`` / ``"EXIT"`` for a known fill action, else None.

    Anything that is not a documented BUY/SELL family member is filtered
    out at the caller. We never coerce — an unknown action skips the slot
    entirely rather than mis-classify.
    """
    if not isinstance(action, str):
        return None
    a = action.upper().strip()
    if a in ENTRY_ACTIONS:
        return "ENTRY"
    if a in EXIT_ACTIONS:
        return "EXIT"
    return None


def _hours_since(ts: str | None, now: datetime) -> float | None:
    """Float hours from ``ts`` to ``now``; None if unparseable.

    A future ``ts`` (wall-clock stepped back — the runner has documented
    clock-skew hazards) clamps to 0.0 rather than rendering negative.
    """
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    secs = (now - dt).total_seconds()
    if secs < 0:
        secs = 0.0
    return round(secs / 3600.0, 2)


def build_exit_only_streak(trades: list[dict]) -> dict:
    """Compute the exit-only-streak snapshot for the book.

    ``trades`` must be oldest → newest (same convention as
    ``streak`` / ``winner_autopsy``). Unknown-action rows are silently
    skipped; only BUY*/SELL* fills count toward the run.

    Returned dict mirrors the existing ``streak`` shape — ``as_of``,
    ``state``, ``verdict`` (None below STABLE), ``headline``, and the
    raw counters callers may want to render. Pure read; never raises;
    every field is degrade-safe.
    """
    now = datetime.now(timezone.utc)

    # Normalize: drop non-BUY/SELL rows entirely so the run is computed
    # only over real entry/exit decisions. Same-microsecond ties are
    # already disambiguated by the store's ORDER BY id tie-break — we
    # respect whatever order the caller hands us.
    classified: list[tuple[str, dict]] = []
    for t in trades or []:
        d = _direction(t.get("action"))
        if d is None:
            continue
        classified.append((d, t))

    n_total = len(classified)
    n_entries = sum(1 for d, _ in classified if d == "ENTRY")
    n_exits = sum(1 for d, _ in classified if d == "EXIT")

    if n_total == 0:
        return {
            "as_of": now.isoformat(timespec="seconds"),
            "state": "NO_DATA",
            "verdict": None,
            "headline": "No BUY/SELL fills yet — no exit run to read.",
            "exit_run_length": 0,
            "exit_run_started_ts": None,
            "exit_run_tickers": [],
            "n_total_fills": 0,
            "n_entries": 0,
            "n_exits": 0,
            "last_entry_ts": None,
            "last_entry_action": None,
            "last_entry_ticker": None,
            "hours_since_last_entry": None,
            "most_recent_ts": None,
            "most_recent_action": None,
            "recent_sequence": [],
            "defensive_trim_min": DEFENSIVE_TRIM_MIN,
            "defensive_liquidation_min": DEFENSIVE_LIQUIDATION_MIN,
        }

    # Walk from newest backward, count the consecutive trailing exits.
    run_len = 0
    run_started_ts: str | None = None
    run_tickers: list[str] = []
    for d, t in reversed(classified):
        if d != "EXIT":
            break
        run_len += 1
        run_started_ts = t.get("timestamp") or run_started_ts
        tk = (t.get("ticker") or "").upper()
        if tk and tk not in run_tickers:
            run_tickers.append(tk)

    # Last ENTRY (most recent BUY*).
    last_entry: dict | None = None
    for d, t in reversed(classified):
        if d == "ENTRY":
            last_entry = t
            break

    last_entry_ts = last_entry.get("timestamp") if last_entry else None
    last_entry_action = last_entry.get("action") if last_entry else None
    last_entry_ticker = ((last_entry.get("ticker") or "").upper()
                         if last_entry else None)
    hours_since_last_entry = _hours_since(last_entry_ts, now)

    most_recent = classified[-1][1]
    most_recent_ts = most_recent.get("timestamp")
    most_recent_action = most_recent.get("action")

    # Verdict.
    if run_len == 0:
        # The newest fill was an entry — book is still opening lots.
        verdict = "MOST_RECENT_IS_ENTRY"
    elif run_len >= DEFENSIVE_LIQUIDATION_MIN:
        verdict = "DEFENSIVE_LIQUIDATION"
    elif run_len >= DEFENSIVE_TRIM_MIN:
        verdict = "DEFENSIVE_TRIM"
    else:
        # Below the trim floor (1-2 exits). Statistically a normal turnover
        # blip — collapse to the same cosmetic verdict as "newest is entry"
        # so the hourly summary stays quiet for benign book churn.
        verdict = "MOST_RECENT_IS_ENTRY"

    state = "STABLE" if n_total >= 1 else "NO_DATA"

    # Headline. Mirrors ``streak``'s prose-first headline style so the
    # hourly Discord line reads naturally.
    if verdict == "MOST_RECENT_IS_ENTRY" and run_len == 0:
        headline = (f"NEUTRAL — newest fill is an entry "
                    f"({last_entry_action} {last_entry_ticker}). "
                    f"{n_entries} entries / {n_exits} exits on record.")
    elif verdict == "MOST_RECENT_IS_ENTRY":
        # Below floor: 1-2 trailing exits.
        run_word = "exit" if run_len == 1 else "exits"
        headline = (f"NEUTRAL — {run_len} trailing {run_word} (below the "
                    f"{DEFENSIVE_TRIM_MIN}-fill trim floor).")
    elif verdict == "DEFENSIVE_TRIM":
        ago = (f"{hours_since_last_entry:.1f}h ago"
               if hours_since_last_entry is not None else "n/a")
        names = ", ".join(run_tickers[:3]) or "n/a"
        headline = (f"DEFENSIVE_TRIM — {run_len} consecutive exits "
                    f"({names}); last entry {ago}.")
    else:  # DEFENSIVE_LIQUIDATION
        ago = (f"{hours_since_last_entry:.1f}h ago"
               if hours_since_last_entry is not None else "n/a")
        names = ", ".join(run_tickers[:3]) or "n/a"
        headline = (f"DEFENSIVE_LIQUIDATION — {run_len} straight exits "
                    f"({names}), no new entry for {ago}. Engine is "
                    f"liquidating, not running the strategy.")

    recent_sequence = [
        "E" if d == "ENTRY" else "X"
        for d, _ in classified[-RECENT_SEQUENCE_LEN:]
    ]

    return {
        "as_of": now.isoformat(timespec="seconds"),
        "state": state,
        "verdict": verdict,
        "headline": headline,
        "exit_run_length": run_len,
        "exit_run_started_ts": run_started_ts,
        "exit_run_tickers": run_tickers,
        "n_total_fills": n_total,
        "n_entries": n_entries,
        "n_exits": n_exits,
        "last_entry_ts": last_entry_ts,
        "last_entry_action": last_entry_action,
        "last_entry_ticker": last_entry_ticker,
        "hours_since_last_entry": hours_since_last_entry,
        "most_recent_ts": most_recent_ts,
        "most_recent_action": most_recent_action,
        "recent_sequence": recent_sequence,
        "defensive_trim_min": DEFENSIVE_TRIM_MIN,
        "defensive_liquidation_min": DEFENSIVE_LIQUIDATION_MIN,
    }


if __name__ == "__main__":  # smoke test against the live DB
    import json

    from paper_trader.store import get_store
    s = get_store()
    rep = build_exit_only_streak(list(reversed(s.recent_trades(2000))))
    print(json.dumps(rep, indent=2, default=str))
