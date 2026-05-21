"""Per-ticker losing-streak detector.

The aggregate ``build_streak`` flags ``TILT_RISK`` on a ≥4-loss run but does
not name *which* tickers carried the losses. A trader on a 4-loss aggregate
streak whose losses are all on a single name (e.g. LITE × 4) has a very
different actionable response — "stop trading LITE" — than one whose losses
are spread across 4 different names ("general tilt → step back"). This
module closes that gap with a per-ticker consecutive-loser view.

Consumes ``round_trips.build_round_trips`` (AGENTS.md invariant #10) so the
P&L per closed trip matches ``trade_asymmetry`` / ``streak`` /
``winner_autopsy`` / ``loser_autopsy``. A "loser" is a round-trip with
``pnl_usd < 0`` — a flat close (``pnl_usd == 0``) is neither a win nor a
loss, matching ``streak.py``'s symmetric treatment; it breaks neither a
losing nor a winning per-ticker run (skipped, not boundary).

Also exposes ``prompt_block`` — a lean, advisory-only render of the
current offender set for injection into the live decision prompt (the
``track_record_block`` precedent — same observational contract,
AGENTS.md invariants #2/#12). NO_DATA / OK states collapse to silence:
a clean book produces no block, mirroring the chat-enrichment helpers'
silence convention.

Advisory only — never gates Opus, never caps a trade
(AGENTS.md invariants #2/#12 — the ``streak`` precedent).
"""
from __future__ import annotations

from datetime import datetime, timezone

from .round_trips import build_round_trips

# Surface threshold — 2 is the lowest non-trivial run a desk acts on
# ("I've lost on this name twice; don't size up again until I find the
# read"). The aggregate ``streak`` uses 4 because cross-ticker noise makes
# small streaks meaningless at the book level; on a SINGLE name even a
# 2-run is a concrete pattern (same instrument, same desk read, twice
# wrong) — the per-ticker equivalent of the aggregate's 4.
REPEAT_LOSER_MIN = 2

_PROMPT_PREAMBLE = (
    "YOUR REPEAT-LOSER WATCH (tickers where your most-recent closed "
    "round-trips are a contiguous losing run — these are observations "
    "and your own history, NOT directives or limits; you retain "
    "complete autonomy over the next decision):"
)


def _render_prompt_block(report: dict,
                          names: set[str] | None = None) -> str | None:
    """Turn a ``build_repeat_loser`` dict into a lean advisory block.

    Returns ``None`` when there is nothing actionable to surface
    (NO_DATA, OK, or — when ``names`` is given — every offender is
    out-of-scope). Same silence precedent as ``track_record_block``
    and the chat-enrichment helpers.

    ``names`` — optional set of in-play tickers to scope to. The block
    is kept lean (one line per offender) and ordered by the builder
    itself (most-streaks first, deepest-loss tie-break) so the prompt
    matches ``/api/repeat-loser``.
    """
    offs = (report or {}).get("offenders") or []
    if names is not None:
        wanted = {t.upper() for t in names}
        offs = [o for o in offs
                if (str(o.get("ticker") or "").upper() in wanted)]
    if not offs:
        return None
    lines = [_PROMPT_PREAMBLE]
    for o in offs:
        bits = [
            str(o.get("ticker") or "?"),
            f"{int(o.get('streak') or 0)} consecutive losses",
            f"${float(o.get('loss_usd') or 0.0):+.2f} cumulative",
            f"{int(o.get('n_round_trips') or 0)} closed trip"
            + ("s" if int(o.get('n_round_trips') or 0) != 1 else ""),
        ]
        last = o.get("last_exit_ts")
        if last:
            bits.append(f"last exit {last}")
        lines.append(f"  - {'  '.join(bits)}")
    return "\n".join(lines)


def _per_ticker_current_loss_streak(
    rts: list[dict],
) -> dict[str, dict]:
    """Walk closed round-trips oldest→newest; track the most-recent
    consecutive-losing run per ticker, plus the trip count.

    Flats (``pnl_usd == 0``) skip — they neither extend nor break a streak.
    A WIN resets the per-ticker streak counter to 0. Returns a dict keyed
    by ticker with: ``current_loss_streak`` (length of the active losing
    run ending at the last non-flat trip on this ticker; 0 if the last
    non-flat was a win), ``current_loss_usd`` (sum of pnl_usd over those
    trips; negative), ``last_loss_exit_ts`` (closing-SELL timestamp of the
    last trip in the current losing run; ``None`` when the streak is 0),
    ``n_round_trips`` (total closed trips on this ticker, flats included).
    """
    per: dict[str, dict] = {}
    for rt in rts:
        tk = (rt.get("ticker") or "").upper()
        if not tk:
            continue
        pnl = rt.get("pnl_usd")
        rec = per.setdefault(tk, {
            "current_loss_streak": 0,
            "current_loss_usd": 0.0,
            "last_loss_exit_ts": None,
            "n_round_trips": 0,
        })
        rec["n_round_trips"] += 1
        if pnl is None:
            continue
        try:
            pnl_f = float(pnl)
        except (TypeError, ValueError):
            continue
        if pnl_f == 0.0:
            # Flat — skip, mirrors streak.py
            continue
        if pnl_f < 0:
            rec["current_loss_streak"] += 1
            rec["current_loss_usd"] += pnl_f
            rec["last_loss_exit_ts"] = rt.get("exit_ts")
        else:
            # Win — reset the per-ticker streak
            rec["current_loss_streak"] = 0
            rec["current_loss_usd"] = 0.0
            rec["last_loss_exit_ts"] = None
    return per


def build_repeat_loser(trades: list[dict],
                        names: set[str] | list[str] | None = None) -> dict:
    """Detect tickers with ≥``REPEAT_LOSER_MIN`` consecutive closed losing
    round-trips ending in their most recent non-flat outcome.

    ``trades`` must be oldest → newest (same convention as
    ``build_streak`` / ``winner_autopsy`` / ``loser_autopsy``).
    ``names`` — when given, scopes the rendered ``prompt_block`` (and
    only the block) to in-play tickers; the full ``offenders`` /
    ``per_ticker`` payload remains unfiltered so ``/api/repeat-loser``,
    chat, and dashboard panels are unaffected. Mirrors
    ``track_record.build_track_record`` 's ``names`` precedent.

    Returns:

      * ``as_of`` — ISO-8601 second-precision timestamp
      * ``state`` — ``NO_DATA`` (no closed trips) | ``OK`` (no offenders) |
        ``REPEAT_LOSER`` (≥1 ticker over the threshold)
      * ``verdict`` — ``"REPEAT_LOSER"`` when there is an offender, else
        ``None`` (the streak.py None-on-non-actionable precedent)
      * ``headline`` — single-sentence summary (always present)
      * ``offenders`` — list of dicts sorted most-streaks-first then
        deepest-loss-first, each with: ``ticker``, ``streak``,
        ``loss_usd`` (sum, negative), ``last_exit_ts``, ``n_round_trips``
      * ``per_ticker`` — full per-ticker map (the offenders list is the
        filtered + sorted view)
      * ``n_offenders``, ``threshold``

    Suppression mirrors ``streak.py``'s OK/EMERGING/NO_DATA contract: a
    book with no qualifying offender is ``OK`` with ``verdict=None`` so the
    surfacing reporter line / endpoint can stay silent ("the summary must
    never become its own lying green light" — the ``streak`` precedent).
    """
    now = datetime.now(timezone.utc)
    rts = build_round_trips(trades)
    n_rts = len(rts)
    per = _per_ticker_current_loss_streak(rts)

    offenders = [
        {
            "ticker": tk,
            "streak": r["current_loss_streak"],
            "loss_usd": round(r["current_loss_usd"], 4),
            "last_exit_ts": r["last_loss_exit_ts"],
            "n_round_trips": r["n_round_trips"],
        }
        for tk, r in per.items()
        if r["current_loss_streak"] >= REPEAT_LOSER_MIN
    ]
    # Most-streaks first (worst tilt risk on the same name);
    # deepest-loss (most negative) breaks ties.
    offenders.sort(key=lambda o: (-o["streak"], o["loss_usd"]))

    if n_rts == 0:
        state = "NO_DATA"
        verdict = None
        headline = "No closed round-trips yet — no repeat-loser pattern to read."
    elif offenders:
        state = "REPEAT_LOSER"
        verdict = "REPEAT_LOSER"
        top = offenders[0]
        if len(offenders) == 1:
            headline = (
                f"REPEAT_LOSER — {top['ticker']} on a "
                f"{top['streak']}-loss run "
                f"(${top['loss_usd']:+.2f} across {top['streak']} closed "
                f"trip{'s' if top['streak'] != 1 else ''}). "
                f"Threshold {REPEAT_LOSER_MIN}."
            )
        else:
            others = ", ".join(
                f"{o['ticker']}×{o['streak']}" for o in offenders[1:4]
            )
            headline = (
                f"REPEAT_LOSER — {top['ticker']} on a "
                f"{top['streak']}-loss run "
                f"(${top['loss_usd']:+.2f}); {len(offenders) - 1} other "
                f"name{'s' if len(offenders) - 1 != 1 else ''} also "
                f"clustered: {others}. Threshold {REPEAT_LOSER_MIN}."
            )
    else:
        state = "OK"
        verdict = None
        headline = (
            f"OK — no ticker on a ≥{REPEAT_LOSER_MIN}-loss run "
            f"across {n_rts} closed trip{'s' if n_rts != 1 else ''}."
        )

    report = {
        "as_of": now.isoformat(timespec="seconds"),
        "state": state,
        "verdict": verdict,
        "headline": headline,
        "offenders": offenders,
        "per_ticker": {
            tk: {
                "current_loss_streak": r["current_loss_streak"],
                "current_loss_usd": round(r["current_loss_usd"], 4),
                "last_loss_exit_ts": r["last_loss_exit_ts"],
                "n_round_trips": r["n_round_trips"],
            }
            for tk, r in per.items()
        },
        "n_offenders": len(offenders),
        "n_round_trips": n_rts,
        "threshold": REPEAT_LOSER_MIN,
    }
    report["prompt_block"] = _render_prompt_block(
        report,
        names=set(names) if names is not None else None,
    )
    return report


if __name__ == "__main__":  # smoke test against the live DB
    import json

    from paper_trader.store import get_store
    s = get_store()
    rep = build_repeat_loser(list(reversed(s.recent_trades(2000))))
    print(json.dumps(rep, indent=2, default=str))
