"""Portfolio risk mirror — concentration + churn, fed back into the prompt.

The behavioural self-review mirror (``self_review.py``) gave the live trader
its *payoff/disposition* feedback; the per-name track-record gave it its
*concrete outcome* on names in play. Neither surfaced the two pathologies the
live book actually exhibited (observed 2026-05-17: total value ~$973 with a
16.7% win rate, the book 60.9% in one name's sector, a 0.52-day median hold —
i.e. **concentrated** *and* **churning**). The dashboard already exposes both
(``/api/correlation`` / ``/api/churn``), but the decision engine itself never
saw them — exactly the gap the self-review module was built to close, one
dimension over.

Single source of truth (AGENTS.md invariant #10): this composes the two
existing pure builders **verbatim** and never re-derives a turnover or
concentration number —

* ``churn.build_churn``             — re-entry / cadence / sub-day churn
* ``correlation.build_correlation`` — pairwise ρ, effective bets, weight HHI

so ``/api/churn``, ``/api/correlation``, the dashboard and the in-prompt block
can never drift apart the way an inline copy would.

**Concentration without price history.** ``decide()`` does *not* fetch daily
close history on the hot path (a per-position yfinance call is a latency +
flake risk on a live trading cycle). Without it ``build_correlation`` reports
``state="INSUFFICIENT"`` and its *headline* collapses to the bare
"correlation verdict withheld" sentence — which would **bury** the
concentration signal in the prompt. But the weight-based fields
(``top_weight_pct`` / ``weight_hhi`` / ``effective_positions_naive``) are
computed unconditionally from ``market_value`` regardless of price history, so
the mirror surfaces *those* directly in that case and only uses the richer ρ
headline when real history is supplied (e.g. a future caller / the endpoint).

**Observational, never prescriptive.** Same contract as the self-review mirror
(AGENTS.md #2/#12): it states facts and the builders' own calibrated
verdicts/headlines, issues no directives, imposes no caps, and reaffirms full
autonomy in its preamble. It informs a decision; it does not gate one.

Pure and deterministic (``now`` injectable). ``trades`` MUST be store-native
**newest-first** (``store.recent_trades()``) — internally fed
``list(reversed(trades))`` to ``build_churn`` exactly as ``/api/churn`` /
``/api/analytics`` do, so the round-trip ordering never diverges. A single bad
builder degrades to "that line missing", never an exception — a diagnostics
fault must not sink a live trading cycle.
"""
from __future__ import annotations

from datetime import datetime, timezone

from .churn import build_churn
from .correlation import build_correlation


def _safe(fn, *args, **kwargs) -> dict:
    """Run one builder; on any failure return a typed empty marker rather than
    letting a single bad builder sink the whole mirror (and, downstream, a
    live trading cycle)."""
    try:
        out = fn(*args, **kwargs)
        return out if isinstance(out, dict) else {}
    except Exception as e:  # pragma: no cover - defensive; exercised via monkeypatch
        return {"state": "ERROR", "status": "ERROR",
                "error": f"{type(e).__name__}: {e}"}


def _churn_line(c: dict) -> str | None:
    """The build_churn headline verbatim, suppressed until there is genuine
    turnover history (a fresh book with zero closed round-trips has nothing
    behavioural to mirror — defer to the honest fallback line instead)."""
    if not c or c.get("state") in ("ERROR", "NO_DATA"):
        return None
    if int(c.get("n_round_trips") or 0) <= 0:
        return None
    hl = c.get("headline")
    if not hl:
        return None
    line = f"  Turnover: {hl}"
    vr = c.get("verdict_reason")
    # Only append the interpretive reason when it isn't already restated by
    # the headline (the self-review _capital_line discipline).
    if vr and vr not in hl:
        line += f"\n    ({vr})"
    return line


def _concentration_line(c: dict) -> str | None:
    """Concentration signal. With real price history (state OK) use the rich
    ρ headline verbatim. Without it (state INSUFFICIENT — the live
    ``decide()`` path) the headline is the buried "verdict withheld"
    sentence, so surface the weight-based concentration from the structured
    fields instead. No stock book ⇒ undefined ⇒ no line (don't fake it)."""
    if not c or c.get("state") in ("ERROR", "NO_DATA") or c.get("status") == "ERROR":
        return None

    if c.get("state") == "OK":
        hl = c.get("headline")
        return f"  Concentration: {hl}" if hl else None

    # state == "INSUFFICIENT": weight-based fallback (never the withheld line).
    top_pct = c.get("top_weight_pct")
    top_tk = c.get("top_weight_ticker")
    if top_pct is None or top_tk is None:
        return None
    n = int(c.get("n_stock_positions") or 0)
    eff = c.get("effective_positions_naive")
    hhi = c.get("weight_hhi")
    eff_clause = (f"{eff:.1f} effective name(s) by weight"
                  if eff is not None else f"{n} stock name(s)")
    hhi_clause = f", HHI={hhi:.2f}" if hhi is not None else ""
    return (f"  Concentration: {top_tk} is {top_pct:.0f}% of a {n}-name stock "
            f"book — {eff_clause}{hhi_clause}; pairwise correlation pending "
            f"more price history.")


_PREAMBLE = (
    "RISK MIRROR (your book's concentration and turnover, for your awareness "
    "only — observations and your own calibrated verdicts, NOT directives or "
    "limits; you retain complete autonomy over the next decision):"
)


def build_risk_mirror(trades: list[dict],
                       positions: list[dict],
                       price_history: dict | None = None,
                       now: datetime | None = None) -> dict:
    """Compose the churn + correlation diagnostics into one prompt-ready
    ``prompt_block`` string. Pure; never raises.

    ``trades`` — store-native **newest-first** ledger
    (``store.recent_trades()``); reversed internally for ``build_churn``.
    ``positions`` — open positions, each ``{ticker, market_value, type}``
    (the ``build_correlation`` input shape; ``snap['positions']`` already
    carries these). ``price_history`` — optional ``{ticker:[close,…]}``;
    ``None``/``{}`` ⇒ weight-only concentration (the live path).
    """
    now = now or datetime.now(timezone.utc)

    ch = _safe(build_churn, list(reversed(trades or [])), now=now)
    co = _safe(build_correlation, positions or [], price_history or {}, now=now)

    body = [ln for ln in (_churn_line(ch), _concentration_line(co)) if ln]
    if body:
        prompt_block = _PREAMBLE + "\n" + "\n".join(body)
    else:
        # NO_DATA / all-empty: an honest, short line beats an empty section
        # or a None the caller has to special-case (the self-review precedent).
        prompt_block = (
            _PREAMBLE
            + "\n  No closed round-trips and no concentratable stock book "
            "yet — nothing to mirror.")

    summary_bits = []
    cv = ch.get("verdict") or ch.get("state")
    if cv:
        summary_bits.append(f"churn={cv}")
    ov = co.get("verdict") or co.get("state")
    if ov:
        summary_bits.append(f"concentration={ov}")
    summary = " · ".join(summary_bits) if summary_bits else "no-data"

    return {
        "as_of": now.isoformat(timespec="seconds"),
        "summary": summary,
        "prompt_block": prompt_block,
        "churn": ch,
        "correlation": co,
    }


if __name__ == "__main__":  # smoke test against the live DB
    import json

    from paper_trader.store import get_store
    s = get_store()
    rep = build_risk_mirror(s.recent_trades(2000),
                            [{"ticker": p.get("ticker"),
                              "market_value": p.get("market_value"),
                              "type": p.get("type")}
                             for p in s.open_positions()])
    print(rep["prompt_block"])
    print("\n---\n")
    print(json.dumps({k: v for k, v in rep.items() if k != "prompt_block"},
                     indent=2, default=str))
