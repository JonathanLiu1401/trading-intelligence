r"""Hard-exit (mechanical SL/TP) visibility.

The hard SL/TP enforcement (commit 3176d2f, 2026-05-24) silently
auto-closes any stock position whose mark breaches the 2%/3% (standard)
or 4%/6% (leveraged) threshold the BUY path stamped at entry. The
runner posts a one-line ``**AUTO RISK EXIT** \`TICKER\``` alert per
event, but no surface aggregates the discipline:

  * Are mechanical exits dominating discretionary closes (mechanical
    discipline working), or barely firing (the bot is hand-managing
    its book)?
  * Is SL outpacing TP (entries are bad — bot is getting stopped out
    more than it's hitting targets), or vice versa (entries are
    landing in winners)?
  * What was the cumulative realized $ impact of each class?

The dashboard's ``/api/closed-positions`` shows per-lot realized P/L
but does not bucket by exit reason. ``trade_outcomes.py`` counts
``n_hard_sl`` / ``n_hard_tp`` but only as raw counts — no P/L. This
builder fills the operator-actionable gap.

Pure: feeds off ``store.recent_trades(N)`` (newest-first). No network,
no extra store reads, never raises (the dashboard endpoint contract:
any fault degrades to the ERROR envelope, not a 500). Companion to
``round_trips.py`` — both classify by trade ledger, but this one
keys on the exit *reason* string and answers the SL-vs-TP discipline
question those don't.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone


# Substring markers strategy._check_and_execute_hard_exits writes into
# the trade reason. Locked at module level so a test (or future operator
# tooling) reads the same constants the live emitter uses — the
# "tests read live module constants" digital-intern discipline.
HARD_SL_MARKER = "HARD_SL"
HARD_TP_MARKER = "HARD_TP"

# Verdict thresholds. A discipline ratio (TP / (SL+TP)) above this is
# clearly "winners dominate", below the inverse is "losers dominate";
# in between is mixed. 0.65 / 0.35 picks the same noise floor the
# session-delta and concordance verdicts use (a ≥2:1 imbalance) so the
# trader reads these surfaces in the same units.
_TP_HEAVY_THRESHOLD = 0.65
_SL_HEAVY_THRESHOLD = 0.35

# Minimum count before a directional verdict is published — same noise
# floor as recent_starvation_trend's _TREND_MIN_HALF in spirit (the
# "never cry wolf on a tiny sample" precedent). With 0–2 hard exits
# any imbalance is jitter; we wait for ≥3 before naming a verdict.
MIN_FOR_VERDICT = 3


def _classify_exit_reason(reason: str | None) -> str | None:
    """Return ``"HARD_SL"``, ``"HARD_TP"``, or ``None`` for a SELL trade's
    reason string. Pure, never raises.

    The strategy writes the marker as a prefix (``HARD_SL: price ...``)
    so a substring match is correct. Discretionary closes have arbitrary
    Opus-authored reasons and won't match either marker — they return
    ``None`` (the caller buckets them as "discretionary")."""
    if not reason:
        return None
    if HARD_SL_MARKER in reason:
        return "HARD_SL"
    if HARD_TP_MARKER in reason:
        return "HARD_TP"
    return None


def _verdict(n_sl: int, n_tp: int) -> tuple[str, str]:
    """Return ``(verdict, headline)`` for the SL/TP count pair.

    Verdict ladder:
      * ``NO_HARD_EXITS`` — neither has fired
      * ``INSUFFICIENT`` — combined count < MIN_FOR_VERDICT (still
        accumulating signal)
      * ``TP_HEAVY``     — TP share >= _TP_HEAVY_THRESHOLD
      * ``SL_HEAVY``     — TP share <= _SL_HEAVY_THRESHOLD
      * ``BALANCED``     — between the two thresholds
    """
    total = n_sl + n_tp
    if total == 0:
        return ("NO_HARD_EXITS",
                "No mechanical exits have fired yet — discipline guard "
                "armed but untested.")
    if total < MIN_FOR_VERDICT:
        return ("INSUFFICIENT",
                f"Only {total} hard exit(s) so far — need ≥{MIN_FOR_VERDICT} "
                "for a directional verdict.")
    tp_share = n_tp / total
    if tp_share >= _TP_HEAVY_THRESHOLD:
        return ("TP_HEAVY",
                f"Winners dominate: {n_tp}/{total} mechanical exits hit TP "
                f"({tp_share * 100:.0f}%). Entries are landing in trends.")
    if tp_share <= _SL_HEAVY_THRESHOLD:
        return ("SL_HEAVY",
                f"Losers dominate: {n_sl}/{total} mechanical exits hit SL "
                f"({(1 - tp_share) * 100:.0f}%). Entries are getting stopped "
                "out — review entry timing.")
    return ("BALANCED",
            f"Mixed: {n_tp} TP vs {n_sl} SL ({tp_share * 100:.0f}% TP). "
            "Mechanical discipline working both ways.")


def _trade_summary(t: dict) -> dict:
    """Compact representation of a trade for the last_hard_* fields.

    Picks only the operator-facing fields a trader scanning Discord /
    the dashboard cares about — full row is available on the trades
    endpoint."""
    return {
        "ticker": t.get("ticker"),
        "qty": t.get("qty"),
        "price": t.get("price"),
        "value": t.get("value"),
        "timestamp": t.get("timestamp"),
        "reason": (t.get("reason") or "")[:200],
    }


def build_hard_exit_summary(
    trades_newest_first: list[dict],
    *,
    now: datetime | None = None,
) -> dict:
    """Operator snapshot of hard SL/TP exit discipline over the entire
    trade ledger we were given.

    ``trades_newest_first`` is the store-native ordering (most recent
    first) — matches every other builder's input convention. Walks every
    SELL trade and buckets by reason marker. Discretionary SELLs (no
    HARD_SL/HARD_TP marker) are counted in ``n_discretionary_sells``
    so the operator can read the mechanical-vs-discretionary mix at a
    glance.

    P/L attribution: this builder counts **trade notional**
    (``value``), NOT round-trip P/L, on purpose. Round-trip pairing
    (round_trips.py) is already the canonical P/L surface; replicating
    that here would be redundant and risks the two surfaces drifting on
    edge cases (partial closes, ticker reuse). The ``realized_*_usd``
    field is therefore the *cash freed* from the auto-exit, which is the
    "how much did the discipline move" question this builder answers
    one dimension below "how much did I lock in".

    Always returns the full dict shape (no key-misses for consumers) —
    fields default to 0 / None / NO_HARD_EXITS on an empty/fresh book.

    Failure contract: never raises. Any internal fault degrades to a
    valid ERROR envelope (``state="ERROR"`` + a short error string) so
    the dashboard endpoint can render the cell without a 500 — the
    ``notify_health`` / ``feed_status`` precedent.

    ``now`` is injectable for deterministic tests."""
    now_ = (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")
    try:
        n_sl = 0
        n_tp = 0
        n_discretionary = 0
        notional_sl = 0.0
        notional_tp = 0.0
        last_sl: dict | None = None
        last_tp: dict | None = None
        # By-ticker break-down so the operator can see which name is
        # getting stopped out repeatedly (a strategy-level signal).
        per_ticker: dict[str, dict[str, int]] = defaultdict(
            lambda: {"sl": 0, "tp": 0}
        )

        for t in trades_newest_first:
            action = (t.get("action") or "").upper()
            if not action.startswith("SELL"):
                continue
            cls = _classify_exit_reason(t.get("reason"))
            ticker = (t.get("ticker") or "").upper()
            try:
                value = float(t.get("value") or 0.0)
            except (TypeError, ValueError):
                value = 0.0
            if cls == "HARD_SL":
                n_sl += 1
                notional_sl += value
                if ticker:
                    per_ticker[ticker]["sl"] += 1
                if last_sl is None:  # newest-first → first match is newest
                    last_sl = _trade_summary(t)
            elif cls == "HARD_TP":
                n_tp += 1
                notional_tp += value
                if ticker:
                    per_ticker[ticker]["tp"] += 1
                if last_tp is None:
                    last_tp = _trade_summary(t)
            else:
                n_discretionary += 1

        total_hard = n_sl + n_tp
        discipline_ratio = (
            n_tp / total_hard if total_hard > 0 else None
        )
        verdict, headline = _verdict(n_sl, n_tp)

        # Mechanical share — what fraction of ALL SELLs were mechanical.
        # A bot that hand-manages every exit (mechanical_share ≈ 0) is a
        # different beast from one where the SL/TP do most of the closing.
        # None when there are no SELLs at all (a hold-only fresh book).
        total_sells = total_hard + n_discretionary
        mechanical_share = (
            total_hard / total_sells if total_sells > 0 else None
        )

        # Per-ticker — sorted by combined count desc so the worst
        # repeat-offenders surface first; cap at 10 to keep the cell
        # lean. Each row: (ticker, sl, tp, total).
        top_tickers = sorted(
            (
                {"ticker": tk, "n_sl": d["sl"], "n_tp": d["tp"],
                 "n_total": d["sl"] + d["tp"]}
                for tk, d in per_ticker.items()
            ),
            key=lambda r: r["n_total"],
            reverse=True,
        )[:10]

        return {
            "as_of": now_,
            "state": "OK",
            "verdict": verdict,
            "headline": headline,
            "n_hard_sl": n_sl,
            "n_hard_tp": n_tp,
            "n_discretionary_sells": n_discretionary,
            "n_total_hard": total_hard,
            "realized_sl_usd": round(notional_sl, 2),
            "realized_tp_usd": round(notional_tp, 2),
            "net_hard_notional_usd": round(notional_tp - notional_sl, 2),
            "discipline_ratio": (
                round(discipline_ratio, 3)
                if discipline_ratio is not None else None
            ),
            "mechanical_share": (
                round(mechanical_share, 3)
                if mechanical_share is not None else None
            ),
            "last_hard_sl": last_sl,
            "last_hard_tp": last_tp,
            "top_tickers": top_tickers,
            # Echo the live thresholds so a dashboard consumer can render
            # them next to the verdict — same echo discipline feed_health
            # follows for live_min_score / blind_streak_min.
            "tp_heavy_threshold": _TP_HEAVY_THRESHOLD,
            "sl_heavy_threshold": _SL_HEAVY_THRESHOLD,
            "min_for_verdict": MIN_FOR_VERDICT,
        }
    except Exception as e:
        return {
            "as_of": now_,
            "state": "ERROR",
            "verdict": "ERROR",
            "headline": f"hard_exit_summary failed: {e}",
            "error": str(e),
            "n_hard_sl": 0,
            "n_hard_tp": 0,
            "n_discretionary_sells": 0,
            "n_total_hard": 0,
            "realized_sl_usd": 0.0,
            "realized_tp_usd": 0.0,
            "net_hard_notional_usd": 0.0,
            "discipline_ratio": None,
            "mechanical_share": None,
            "last_hard_sl": None,
            "last_hard_tp": None,
            "top_tickers": [],
            "tp_heavy_threshold": _TP_HEAVY_THRESHOLD,
            "sl_heavy_threshold": _SL_HEAVY_THRESHOLD,
            "min_for_verdict": MIN_FOR_VERDICT,
        }
