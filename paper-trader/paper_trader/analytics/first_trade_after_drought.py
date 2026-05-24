"""First-trade-after-drought characterisation — when the bot resumes, what does it do?

After a NO_DECISION storm clears, the next FILLED trade is structurally
distinct from a trade in the middle of normal cadence. The operator's
question is **"is the bot panic-trading because it just sat blank for an
hour, or is it making a real call?"** ``runner_heartbeat`` flags storms in
progress and ``no_decision_recovery`` grades how long they last; neither
looks at *what happens immediately after*.

A drought = a contiguous run of ≥ ``_MIN_DROUGHT_RUN`` NO_DECISION cycles.
A *post-drought first-trade* is the first FILLED decision whose timestamp
is after the last NO_DECISION cycle of that run. For each one we record:

  * was it on a **new** ticker (no prior buy in the visible ledger) — i.e.
    a fresh exploration after sitting blank?
  * or was it a **re-cycle** of an already-traded name (could be momentum-
    chasing the same favourite, or a routine add-on)?
  * was it a **flip** — a BUY on a ticker that was SELL'd within
    ``_FLIP_WINDOW_HOURS`` immediately before the drought (classic panic
    reversal pattern: sell, sit blank, buy back)?

The comparison is post-drought first-trades vs the **rest** of the trades.
``new_rate_delta`` = (post-drought new-trade rate) − (other-trade new-trade
rate). A meaningfully positive delta means droughts END in exploration
(arguably good — the bot is using the pause to step back); a meaningfully
negative delta means droughts END by re-grabbing familiar names (the
recycle-bias / panic-buy shape).

Pure builder; ``store.recent_decisions`` newest-first + ``store.recent_trades``
newest-first inputs. Never raises; garbage in → ``NO_DATA``. **Advisory only**
— observational, no caps, never gates Opus, has no path to ``_execute()``
(AGENTS.md invariants #2/#12 — the ``no_decision_recovery`` precedent).
"""
from __future__ import annotations

from datetime import datetime, timezone

# A NO_DECISION run shorter than this isn't a drought (matches the
# documented ``NO_DECISION_STORM_THRESHOLD = 5`` in runner_heartbeat —
# anything shorter is a transient hiccup, not a wedge to "recover" from).
_MIN_DROUGHT_RUN = 5

# A BUY within this many hours after a SELL of the same ticker is a "flip"
# in this builder's vocabulary. 24h is wide enough to cover the
# overnight-then-morning pattern the live trader's 60s cadence exposes,
# without dragging in re-entries that are actually new theses.
_FLIP_WINDOW_HOURS = 24.0

# Minimum number of post-drought trades before we'll grade the delta. With
# fewer than this the new-rate-delta is too noisy to act on (a single
# fresh-ticker fill swings the rate 100 pp).
_MIN_POST_DROUGHT = 3

# Effect-size threshold for surfacing a verdict that points to a pattern,
# in absolute percentage-points of new-rate delta. Below it the post-drought
# behaviour isn't meaningfully different from normal.
_SIGNIFICANT_DELTA_PCT = 20.0


def _parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        s = ts.replace("Z", "+00:00") if isinstance(ts, str) else ""
        d = datetime.fromisoformat(s)
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except (ValueError, AttributeError, TypeError):
        return None


def _is_no_decision(action_taken: str | None) -> bool:
    """Canonical predicate — verbatim mirror of
    ``no_decision_recovery._is_no_decision`` / ``decision_forensics`` /
    ``runner_heartbeat``. Drift-locked by unit test."""
    raw = (action_taken or "").strip()
    return not raw or raw == "NO_DECISION"


def _is_filled(action_taken: str | None) -> bool:
    """A decision row whose ``action_taken`` records a *filled* trade.
    Matches the ``decisions.action_taken`` free-text shape documented in
    CLAUDE.md §invariant #11: ``"BUY NVDA → FILLED"`` / ``"SELL TQQQ → FILLED"``.
    HOLD / BLOCKED / NO_DECISION / parse_failed are all excluded."""
    raw = (action_taken or "").strip().upper()
    return "FILLED" in raw and raw != "NO_DECISION"


def _trade_action_kind(action: str | None) -> str:
    """Map a trade row's ``action`` column to BUY / SELL / OTHER."""
    if not action:
        return "OTHER"
    a = action.strip().upper()
    if a in ("BUY", "BUY_CALL", "BUY_PUT"):
        return "BUY"
    if a in ("SELL", "SELL_CALL", "SELL_PUT"):
        return "SELL"
    return "OTHER"


def _droughts_from_decisions(
    decisions_newest_first: list[dict],
    min_run: int,
) -> list[dict]:
    """Run-length encode NO_DECISION runs and return a list of completed
    droughts whose length is ≥ ``min_run``.

    Each entry is::

        {"start_ts": <oldest NO_DECISION ts in run>,
         "end_ts":   <newest NO_DECISION ts in run>,
         "length":   <cycles>}

    Walks oldest → newest so timestamps come out monotonically. An
    *open* run (touches the newest row, no real decision after it) is
    excluded — it doesn't have a post-drought trade yet.
    """
    rows_oldest = list(reversed(decisions_newest_first or []))
    droughts: list[dict] = []
    cur_start: str | None = None
    cur_end: str | None = None
    cur_len = 0
    for d in rows_oldest:
        if _is_no_decision(d.get("action_taken")):
            if cur_len == 0:
                cur_start = d.get("timestamp")
            cur_end = d.get("timestamp")
            cur_len += 1
        else:
            if cur_len >= min_run and cur_start and cur_end:
                droughts.append(
                    {"start_ts": cur_start, "end_ts": cur_end, "length": cur_len}
                )
            cur_start = None
            cur_end = None
            cur_len = 0
    # An open run at the end is intentionally NOT included; no post-drought
    # trade exists yet.
    return droughts


def build_first_trade_after_drought(
    decisions: list[dict] | None,
    trades: list[dict] | None,
    min_drought_run: int = _MIN_DROUGHT_RUN,
) -> dict:
    """Characterise the first FILLED trade after each completed NO_DECISION
    drought, and contrast it with non-post-drought trades.

    Result shape (always present, never raises):

      * ``state``                 — ``"NO_DATA"`` if inputs empty; else ``"OK"``.
      * ``n_droughts``            — completed NO_DECISION runs of length
                                    ≥ ``min_drought_run``.
      * ``n_post_drought_trades`` — droughts that were followed by ≥1 trade.
      * ``post_drought_trades``   — list of per-event records (capped at 20),
                                    newest-first. Each row::
            {"drought_end_ts", "drought_length", "trade_ts", "ticker",
             "action", "is_new_ticker", "is_flip"}
      * ``post_drought_new_rate``   — % of post-drought trades on a brand-new
                                      ticker (None if n < ``_MIN_POST_DROUGHT``).
      * ``other_new_rate``          — % of *non*-post-drought trades on a brand-
                                      new ticker (None if no comparable rows).
      * ``new_rate_delta_pp``       — post-drought minus other, in pp (None if
                                      either rate is None).
      * ``post_drought_flip_rate``  — % of post-drought trades that are flips
                                      (BUY within ``_FLIP_WINDOW_HOURS`` of a
                                      prior SELL of the same ticker).
      * ``other_flip_rate``         — same for non-post-drought trades.
      * ``flip_rate_delta_pp``      — post-drought minus other, in pp.
      * ``verdict``                 — see ladder below.
      * ``verdict_detail``          — one-line chat suitable explanation.
      * ``headline``                — dashboard / Discord one-liner.

    Verdict ladder:

      * ``NO_DATA``               — no trades / no droughts / no post-drought
                                    fills.
      * ``INSUFFICIENT_HISTORY``  — fewer than ``_MIN_POST_DROUGHT`` post-
                                    drought fills.
      * ``PANIC_RECYCLE``         — new-rate delta ≤ −``_SIGNIFICANT_DELTA_PCT``;
                                    droughts end by grabbing familiar names.
      * ``PANIC_FLIP``            — flip-rate delta ≥ +``_SIGNIFICANT_DELTA_PCT``;
                                    droughts end with a same-ticker reversal.
      * ``DROUGHT_EXPLORATION``   — new-rate delta ≥ +``_SIGNIFICANT_DELTA_PCT``;
                                    droughts end with a fresh-ticker initiation.
      * ``STEADY_RECOVERY``       — none of the above; post-drought behaviour
                                    is statistically indistinguishable from
                                    normal.
    """
    rows_d = list(decisions or [])
    rows_t = list(trades or [])

    if not rows_d or not rows_t:
        return _no_data()

    droughts = _droughts_from_decisions(rows_d, max(1, int(min_drought_run)))
    if not droughts:
        return _no_data(
            n_droughts=0, headline=(
                "no completed NO_DECISION droughts of "
                f"≥ {min_drought_run} cycles — cannot assess recovery quality"
            ),
        )

    # Pair each drought with the first trade strictly after its end_ts.
    # Trades come newest-first; reverse so we can walk in chronological
    # order and binary-search-equivalent with a single pointer.
    trades_chrono = sorted(
        [t for t in rows_t if _parse_ts(t.get("timestamp")) is not None],
        key=lambda t: _parse_ts(t["timestamp"]),  # type: ignore[arg-type]
    )

    # Track ticker→prior-buy and ticker→prior-sell timestamps so we can
    # decide is_new_ticker / is_flip without re-walking for every drought.
    # We walk trades chrono once, mapping each trade to its index, then
    # for each drought we do a small linear scan from the boundary.
    # Pre-compute ticker→list-of-(ts_obj, kind, idx).
    ticker_history: dict[str, list[tuple[datetime, str, int]]] = {}
    for i, t in enumerate(trades_chrono):
        tk = (t.get("ticker") or "").strip().upper()
        if not tk:
            continue
        ts_obj = _parse_ts(t.get("timestamp"))
        if ts_obj is None:
            continue
        kind = _trade_action_kind(t.get("action"))
        ticker_history.setdefault(tk, []).append((ts_obj, kind, i))

    # For "new ticker" decisions across the whole ledger, walk chronologically
    # and for each trade decide whether it's the *first* buy on its ticker.
    seen_buys: set[str] = set()
    trade_is_new_buy: list[bool] = []
    trade_is_flip: list[bool] = []
    for t in trades_chrono:
        tk = (t.get("ticker") or "").strip().upper()
        kind = _trade_action_kind(t.get("action"))
        ts_obj = _parse_ts(t.get("timestamp"))
        is_new = False
        is_flip = False
        if kind == "BUY" and tk and tk not in seen_buys:
            is_new = True
        if kind == "BUY" and tk and ts_obj is not None and tk in ticker_history:
            for prev_ts, prev_kind, _ in ticker_history[tk]:
                if prev_ts >= ts_obj:
                    break
                if prev_kind == "SELL":
                    delta_h = (ts_obj - prev_ts).total_seconds() / 3600.0
                    if 0 <= delta_h <= _FLIP_WINDOW_HOURS:
                        is_flip = True
        if kind == "BUY" and tk and tk not in seen_buys:
            seen_buys.add(tk)
        trade_is_new_buy.append(is_new)
        trade_is_flip.append(is_flip)

    # Pair droughts with the first trade after each drought's end_ts.
    post_drought_idx: set[int] = set()
    pd_records: list[dict] = []
    for dr in droughts:
        end_ts = _parse_ts(dr["end_ts"])
        if end_ts is None:
            continue
        # First trade with ts > end_ts.
        found = None
        for i, t in enumerate(trades_chrono):
            ts_obj = _parse_ts(t.get("timestamp"))
            if ts_obj is None:
                continue
            if ts_obj > end_ts:
                found = (i, t, ts_obj)
                break
        if found is None:
            continue
        idx, trade, trade_ts = found
        if idx in post_drought_idx:
            # Multiple droughts share the same post-drought trade — only
            # count the first. Avoids double-counting in tightly-packed
            # storm windows.
            continue
        post_drought_idx.add(idx)
        pd_records.append({
            "drought_end_ts": dr["end_ts"],
            "drought_length": dr["length"],
            "trade_ts": trade.get("timestamp"),
            "ticker": (trade.get("ticker") or "").strip().upper() or None,
            "action": (trade.get("action") or "").strip().upper() or None,
            "is_new_ticker": bool(trade_is_new_buy[idx]),
            "is_flip": bool(trade_is_flip[idx]),
        })

    n_pd = len(pd_records)
    # Other = trades NOT in post_drought_idx.
    n_total = len(trades_chrono)
    n_other = n_total - n_pd

    if n_pd == 0:
        return _no_data(
            n_droughts=len(droughts),
            headline=(
                f"{len(droughts)} drought(s) of ≥ {min_drought_run} cycles "
                "but none followed by a trade in the visible ledger"
            ),
        )

    pd_new_count = sum(1 for r in pd_records if r["is_new_ticker"])
    pd_flip_count = sum(1 for r in pd_records if r["is_flip"])

    other_new_count = sum(
        1 for i in range(n_total)
        if i not in post_drought_idx and trade_is_new_buy[i]
    )
    other_flip_count = sum(
        1 for i in range(n_total)
        if i not in post_drought_idx and trade_is_flip[i]
    )

    pd_new_rate: float | None
    if n_pd >= _MIN_POST_DROUGHT:
        pd_new_rate = round(100.0 * pd_new_count / n_pd, 2)
    else:
        pd_new_rate = None
    other_new_rate: float | None = (
        round(100.0 * other_new_count / n_other, 2) if n_other else None
    )
    new_delta_pp: float | None = (
        round(pd_new_rate - other_new_rate, 2)
        if pd_new_rate is not None and other_new_rate is not None
        else None
    )

    pd_flip_rate = round(100.0 * pd_flip_count / n_pd, 2) if n_pd else 0.0
    other_flip_rate: float | None = (
        round(100.0 * other_flip_count / n_other, 2) if n_other else None
    )
    flip_delta_pp: float | None = (
        round(pd_flip_rate - other_flip_rate, 2)
        if other_flip_rate is not None
        else None
    )

    # ── verdict ──────────────────────────────────────────────────
    if n_pd < _MIN_POST_DROUGHT:
        verdict = "INSUFFICIENT_HISTORY"
        verdict_detail = (
            f"only {n_pd} post-drought trade(s) (need ≥ {_MIN_POST_DROUGHT}) — "
            "cannot grade recovery quality"
        )
    else:
        # Flip-bias is the strongest panic-pattern signal (sell, sit blank,
        # immediately buy back is unambiguously bad). Check it first.
        if (
            flip_delta_pp is not None
            and flip_delta_pp >= _SIGNIFICANT_DELTA_PCT
        ):
            verdict = "PANIC_FLIP"
            verdict_detail = (
                f"{pd_flip_count}/{n_pd} post-drought trades are flips of a "
                f"recent SELL ({pd_flip_rate:.0f}% vs {other_flip_rate:.0f}% "
                "normal) — panic-reversal pattern"
            )
        elif (
            new_delta_pp is not None
            and new_delta_pp <= -_SIGNIFICANT_DELTA_PCT
        ):
            verdict = "PANIC_RECYCLE"
            verdict_detail = (
                f"only {pd_new_count}/{n_pd} post-drought trades are net-new "
                f"names ({pd_new_rate:.0f}% vs {other_new_rate:.0f}% normal) — "
                "droughts end by re-grabbing familiar tickers"
            )
        elif (
            new_delta_pp is not None
            and new_delta_pp >= _SIGNIFICANT_DELTA_PCT
        ):
            verdict = "DROUGHT_EXPLORATION"
            verdict_detail = (
                f"{pd_new_count}/{n_pd} post-drought trades open fresh names "
                f"({pd_new_rate:.0f}% vs {other_new_rate:.0f}% normal) — "
                "droughts end with exploration; arguably healthy"
            )
        else:
            verdict = "STEADY_RECOVERY"
            verdict_detail = (
                f"post-drought behaviour matches the rest of the ledger "
                f"(new-rate {pd_new_rate:.0f}% vs "
                f"{other_new_rate if other_new_rate is not None else '—'}% normal)"
            )

    # ── headline ─────────────────────────────────────────────────
    headline = (
        f"{n_pd} post-drought trade(s) across {len(droughts)} drought(s); "
        f"new-rate {pd_new_rate if pd_new_rate is not None else '—'}% vs "
        f"{other_new_rate if other_new_rate is not None else '—'}% normal — {verdict}"
    )

    return {
        "state": "OK",
        "n_droughts": len(droughts),
        "n_post_drought_trades": n_pd,
        "post_drought_trades": list(reversed(pd_records))[:20],
        "post_drought_new_rate": pd_new_rate,
        "other_new_rate": other_new_rate,
        "new_rate_delta_pp": new_delta_pp,
        "post_drought_flip_rate": pd_flip_rate,
        "other_flip_rate": other_flip_rate,
        "flip_rate_delta_pp": flip_delta_pp,
        "min_drought_run": int(min_drought_run),
        "verdict": verdict,
        "verdict_detail": verdict_detail,
        "headline": headline,
    }


def _no_data(
    n_droughts: int = 0,
    headline: str = "no trades or decisions on record — cannot assess post-drought quality",
) -> dict:
    return {
        "state": "NO_DATA",
        "n_droughts": n_droughts,
        "n_post_drought_trades": 0,
        "post_drought_trades": [],
        "post_drought_new_rate": None,
        "other_new_rate": None,
        "new_rate_delta_pp": None,
        "post_drought_flip_rate": 0.0,
        "other_flip_rate": None,
        "flip_rate_delta_pp": None,
        "min_drought_run": _MIN_DROUGHT_RUN,
        "verdict": "NO_DATA",
        "verdict_detail": (
            "no completed droughts followed by visible trades — nothing to grade"
            if n_droughts
            else "no trades or decisions on record"
        ),
        "headline": headline,
    }


def _main() -> int:
    """One-shot CLI."""
    import json
    from ..store import get_store
    s = get_store()
    out = build_first_trade_after_drought(
        s.recent_decisions(500),
        s.recent_trades(500),
    )
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
