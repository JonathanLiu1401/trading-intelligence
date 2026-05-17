"""Track record in play — per-name closed-trade memory for the decision prompt.

``self_review`` already feeds the live trader its *aggregate* behavioural
mirror (payoff ratio, the disposition gap, capital-paralysis state, open-book
alpha). What it does **not** give the trader is the concrete, name-specific
record: *"the last two times you closed NVDA you lost via KNIFE_CATCH — here
is the exact reason you wrote when you bought and when you sold."*

That per-name verbatim closed-trade history is precisely what maps onto the
documented live pathology (observed 2026-05-17: ``avg_holding_days`` ~0.27,
fast same-name re-entry churn, the NVDA→LITE→NVDA shape, ``KNIFE_CATCH``
losers repeating). The aggregate mirror tells the trader it has a disposition
problem *in general*; this tells it *its own outcome on the exact ticker it is
about to trade again*. ``/api/loser-autopsy`` & ``/api/winner-autopsy`` compute
this narrative but are dashboard/chat-only — never fed back into the decision
loop. This module closes that gap.

Single source of truth (paper-trader AGENTS.md invariant #10): it composes
``loser_autopsy.build_loser_autopsy`` + ``winner_autopsy.build_winner_autopsy``
**verbatim** and never re-derives P&L / hold-time / failure-mode. Both already
consume ``round_trips.build_round_trips`` (the one home for closed-round-trip
P&L), so ``/api/track-record``, the dashboard card and the in-prompt block can
never drift apart the way an inline copy would.

**Observational, never prescriptive** (the ``self_review`` precedent — AGENTS.md
invariants #2/#12). The block states facts and the autopsy builders' own
objective mode labels, issues no directives, imposes no caps, and its preamble
explicitly reaffirms full autonomy. It governs *informing* the decision, never
*gating* it — exactly like ``/api/self-review`` / ``/api/capital-paralysis``.
It is a memory, not a cage.

Pure; never raises. ``trades`` MUST be passed exactly as ``/api/loser-autopsy``
& ``/api/trade-asymmetry`` pass it — ``list(reversed(store.recent_trades(N)))``
(oldest→newest; ``build_round_trips`` reads in sequence and does not sort).
"""
from __future__ import annotations

from datetime import datetime, timezone

from .loser_autopsy import build_loser_autopsy
from .winner_autopsy import build_winner_autopsy

# Display caps. The block is injected every decision cycle, so it must stay
# lean: only the most recent ``PER_NAME_CAP`` closed round-trips per name are
# narrated (the net W-L / $ summary is always over *all* of that name's closed
# round-trips, so capping the narrative never hides the net result). A verbatim
# reason longer than ``REASON_CAP`` is truncated with an ellipsis — surfaced
# exactly as the trader wrote it, never NLP-parsed (the ``loser_autopsy`` /
# ``thesis_drift`` discipline).
PER_NAME_CAP = 2
REASON_CAP = 160
# Pass-through cap for the autopsy card lists — large enough to be "all of
# them" for any realistic paper book (their aggregates are always over all
# round-trips; only the *card list* is capped on their side).
_ALL = 10_000

_PREAMBLE = (
    "YOUR CLOSED-TRADE TRACK RECORD ON NAMES IN PLAY (your own past "
    "round-trips on the tickers in front of you this cycle — these are "
    "observations and your own history, NOT directives or limits; you retain "
    "complete autonomy over the next decision):"
)


def _safe(fn, *args, **kwargs) -> dict:
    """Run one autopsy builder; on any failure return a typed empty marker
    rather than letting it sink the block (and, downstream, a live cycle).
    The constituents already never raise — this is defense-in-depth, exercised
    via monkeypatch in the tests."""
    try:
        out = fn(*args, **kwargs)
        return out if isinstance(out, dict) else {}
    except Exception as e:  # pragma: no cover - defensive
        return {"state": "ERROR", "error": f"{type(e).__name__}: {e}"}


def _fmt_reason(r, cap: int) -> str:
    """Verbatim reason, blank→``''``, truncated with an ellipsis past ``cap``."""
    if r is None:
        return ""
    s = str(r).strip()
    if not s:
        return ""
    if len(s) > cap:
        s = s[: cap - 1].rstrip() + "…"
    return s


def _trip_line(t: dict, reason_cap: int) -> str:
    """One compact factual line for a single closed round-trip."""
    bits = [t["outcome"], t["mode"] or "?"]
    if t.get("pnl_pct") is not None:
        bits.append(f"{t['pnl_pct']:+.1f}%")
    bits.append(f"${t['pnl_usd']:+.2f}")
    if t.get("hold_days") is not None:
        bits.append(f"{t['hold_days']:.2f}d")
    line = "    - " + "  ".join(bits)
    er = _fmt_reason(t.get("entry_reason"), reason_cap)
    xr = _fmt_reason(t.get("exit_reason"), reason_cap)
    if er:
        line += f'  entry:"{er}"'
    if xr:
        line += f'  exit:"{xr}"'
    return line


def build_track_record(trades: list[dict],
                        names: set[str] | list[str] | None = None,
                        per_name_cap: int = PER_NAME_CAP,
                        reason_cap: int = REASON_CAP,
                        now: datetime | None = None) -> dict:
    """Per-name closed-trade record. Pure, never raises.

    ``names`` — when given, only tickers in this set are included (the live
    decision prompt passes the *exact* "names in play this cycle" set so the
    block can never disagree with the quant block on what is actionable). When
    ``None`` (the ``/api/track-record`` / dashboard / chat path) every traded
    name is included.

    Returns a dict with a prompt-ready ``prompt_block`` (``None`` when there is
    no relevant closed history — the caller treats ``None`` as "no block this
    cycle", exactly like ``self_review_block``) plus the structured ``names``
    list for the endpoint/UI.
    """
    now = now or datetime.now(timezone.utc)
    name_set = set(names) if names is not None else None

    la = _safe(build_loser_autopsy, trades, worst_n=_ALL, now=now)
    wa = _safe(build_winner_autopsy, trades, best_n=_ALL, now=now)

    # Closed-round-trip count is identical in both (both = len(build_round_trips
    # (trades))); take the max so an ERROR'd constituent can't zero it.
    n_rts = max(int(la.get("n_round_trips") or 0),
                int(wa.get("n_round_trips") or 0))

    # Normalise both card lists into one verbatim record shape. Nothing is
    # re-derived: pnl/pct/hold/mode/reasons are taken straight from the
    # autopsy cards (which took them straight from build_round_trips).
    records: list[dict] = []
    for c in (la.get("worst_losers") or []):
        records.append({
            "ticker": c.get("ticker"),
            "type": c.get("type"),
            "outcome": "LOSS",
            "mode": c.get("failure_mode"),
            "pnl_usd": c.get("pnl_usd"),
            "pnl_pct": c.get("pnl_pct"),
            "hold_days": c.get("hold_days"),
            "entry_ts": c.get("entry_ts"),
            "exit_ts": c.get("exit_ts"),
            "entry_reason": c.get("entry_reason"),
            "exit_reason": c.get("exit_reason"),
        })
    for c in (wa.get("best_winners") or []):
        records.append({
            "ticker": c.get("ticker"),
            "type": c.get("type"),
            "outcome": "WIN",
            "mode": c.get("success_mode"),
            "pnl_usd": c.get("pnl_usd"),
            "pnl_pct": c.get("pnl_pct"),
            "hold_days": c.get("hold_days"),
            "entry_ts": c.get("entry_ts"),
            "exit_ts": c.get("exit_ts"),
            "entry_reason": c.get("entry_reason"),
            "exit_reason": c.get("exit_reason"),
        })

    by_ticker: dict[str, list[dict]] = {}
    for r in records:
        tk = r["ticker"]
        if tk is None:
            continue
        if name_set is not None and tk not in name_set:
            continue
        by_ticker.setdefault(tk, []).append(r)

    name_entries: list[dict] = []
    for tk, trips in by_ticker.items():
        # Newest closed first; ISO timestamps are lexically comparable (the
        # signals.py `first_seen` pattern). Missing exit_ts → sorts oldest.
        trips_sorted = sorted(
            trips,
            key=lambda t: (t.get("exit_ts") or "", t.get("entry_ts") or ""),
            reverse=True,
        )
        n_win = sum(1 for t in trips if t["outcome"] == "WIN")
        n_loss = sum(1 for t in trips if t["outcome"] == "LOSS")
        # net is over ALL of this name's closed round-trips, not just the
        # narrated cap — single source of truth (each pnl_usd is verbatim from
        # build_round_trips via the autopsy card).
        net_usd = round(sum(float(t.get("pnl_usd") or 0.0) for t in trips), 2)
        name_entries.append({
            "ticker": tk,
            "n_closed": len(trips),
            "n_win": n_win,
            "n_loss": n_loss,
            "net_usd": net_usd,
            "recent": trips_sorted[: max(0, per_name_cap)],
        })

    # Worst net first (the names you are bleeding on lead — the desk-discipline
    # ordering loser_autopsy uses; ticker is a deterministic tie-break).
    name_entries.sort(key=lambda e: (e["net_usd"], e["ticker"]))

    state = "NO_DATA" if n_rts == 0 else "OK"

    # ---- prompt block --------------------------------------------------
    if not name_entries:
        prompt_block = None  # nothing relevant → no block this cycle
    else:
        lines = [_PREAMBLE]
        for e in name_entries:
            lines.append(
                f"  {e['ticker']}  {e['n_win']}W-{e['n_loss']}L  "
                f"net ${e['net_usd']:+.2f}  ({e['n_closed']} closed)"
            )
            for t in e["recent"]:
                lines.append(_trip_line(t, reason_cap))
        prompt_block = "\n".join(lines)

    # ---- compact one-liner (logs / chat single-source) -----------------
    if not name_entries:
        summary = "no-history" if name_set is not None else (
            "no-closed-round-trips" if n_rts == 0 else "no-history")
    else:
        worst = name_entries[0]
        summary = (
            f"{len(name_entries)} name(s); worst {worst['ticker']} "
            f"${worst['net_usd']:+.2f} ({worst['n_win']}W-{worst['n_loss']}L)"
        )

    return {
        "as_of": now.isoformat(timespec="seconds"),
        "state": state,
        "summary": summary,
        "prompt_block": prompt_block,
        "n_round_trips": n_rts,
        "filtered": name_set is not None,
        "names": name_entries,
    }


if __name__ == "__main__":  # smoke test against the live DB
    import json

    from paper_trader.store import get_store
    s = get_store()
    rep = build_track_record(list(reversed(s.recent_trades(2000))))
    print(rep.get("prompt_block") or "(no track-record block)")
    print("---")
    print(json.dumps({k: v for k, v in rep.items() if k != "prompt_block"},
                      indent=2, default=str))
