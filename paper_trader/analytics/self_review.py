"""Behavioural self-review — the trader's own mirror, fed back into its prompt.

The system prompt tells Opus to "THINK LIKE A HEDGE FUND MANAGER WHO WANTS
ASYMMETRIC RETURNS", yet ``strategy._build_payload`` historically gave it
*zero* feedback on its own track record. The dashboard / unified chat already
surface the pathology (observed live 2026-05-16: payoff ratio ~0.08, breakeven
win-rate 93%, cutting winners at ``+$0.57`` while riding losers to ``-$7.51``,
``PINNED`` with no dry powder, ``BLEEDING`` -2.2% alpha to parse-failure
droughts) — but the *decision engine itself* never saw it. A real desk reviews
its P&L attribution and behavioural biases before trading; this module gives
the live trader that same mirror.

Single source of truth (paper-trader AGENTS.md invariant #10): it composes the
three existing pure builders **verbatim** and never re-derives P&L —

* ``trade_asymmetry.build_trade_asymmetry``     — exit/sizing pathology
* ``capital_paralysis.build_capital_paralysis`` — trap + cost + unlock
* ``open_attribution.build_open_attribution``   — selection vs SPY on the book

So ``/api/self-review``, the dashboard and the in-prompt block can never drift
apart the way an inline copy would.

**Observational, never prescriptive.** The block states facts and the
builders' own calibrated verdicts/headlines — it issues no directives, imposes
no caps, and explicitly reaffirms full autonomy in its own preamble. This does
*not* violate the "no hard risk limits / Opus has full autonomy" invariant
(#2/#12): that invariant governs *gating decisions*, not *informing* them, just
as ``/api/liquidity`` / ``/api/capital-paralysis`` are advisory-only. It is a
mirror, not a cage.

Pure: feed it the ``store`` reads; ``now`` is injectable for deterministic
tests. ``trades`` MUST be store-native **newest-first**
(``store.recent_trades()``) — the asymmetry consumer is internally fed
``list(reversed(trades))`` exactly as ``/api/analytics`` /
``/api/trade-asymmetry`` does, while the liquidity/paralysis consumer wants the
newest-first order ``build_liquidity`` documents.
"""
from __future__ import annotations

from datetime import datetime, timezone

from .trade_asymmetry import build_trade_asymmetry
from .capital_paralysis import build_capital_paralysis
from .open_attribution import build_open_attribution


def _safe(fn, *args, **kwargs) -> dict:
    """Run one builder; on any failure return a typed empty marker rather than
    letting a single bad builder sink the whole mirror (and, downstream, a
    live trading cycle)."""
    try:
        out = fn(*args, **kwargs)
        return out if isinstance(out, dict) else {}
    except Exception as e:  # pragma: no cover - defensive; exercised via monkeypatch
        return {"state": "ERROR", "status": "ERROR", "error": f"{type(e).__name__}: {e}"}


def _asym_line(a: dict) -> str | None:
    hl = a.get("headline")
    if not hl or a.get("state") == "ERROR":
        return None
    line = f"  Behavioural edge: {hl}"
    vr = a.get("verdict_reason")
    if vr:
        line += f"\n    ({vr})"
    return line


def _capital_line(c: dict) -> str | None:
    hl = c.get("headline")
    if not hl or c.get("state") == "ERROR":
        return None
    line = f"  Capital state: {hl}"
    par = c.get("paralysis") or {}
    pvr = par.get("verdict_reason")
    # The PINNED headline already carries the bleed clause ("…has bled X%
    # alpha across N paralysis drought(s)"); appending the verdict reason
    # there just restates the same number back-to-back. Only add the
    # interpretive reason when the headline doesn't already state the bleed
    # (e.g. a now-FREE book that still has historical involuntary bleed).
    if (pvr and (par.get("involuntary_alpha_bleed_pct") or 0.0) < 0
            and "bled" not in hl):
        line += f"\n    ({pvr})"
    return line


def _attr_line(o: dict) -> str | None:
    hl = o.get("headline")
    if not hl or o.get("status") == "ERROR":
        return None
    return f"  Open-book vs SPY: {hl}"


_PREAMBLE = (
    "SELF-REVIEW (a mirror of your own past behaviour, for your awareness "
    "only — these are observations and your own track record, NOT directives "
    "or limits; you retain complete autonomy over the next decision):"
)


def build_self_review(portfolio: dict,
                       positions: list[dict],
                       trades: list[dict],
                       decisions: list[dict],
                       equity_curve: list[dict],
                       now: datetime | None = None) -> dict:
    """Compose the three pure diagnostics into one canonical behavioural
    mirror + a prompt-ready ``prompt_block`` string. Pure; never raises."""
    now = now or datetime.now(timezone.utc)

    # build_round_trips reads the ledger in sequence (oldest→newest); the
    # liquidity/paralysis path documents newest-first. Mirror the two existing
    # endpoints exactly so this never diverges from /api/trade-asymmetry or
    # /api/capital-paralysis.
    asym = _safe(build_trade_asymmetry, list(reversed(trades or [])), now=now)
    cap = _safe(build_capital_paralysis, portfolio or {}, positions or [],
                trades or [], decisions or [], equity_curve or [], now=now)
    oa = _safe(build_open_attribution, positions or [], equity_curve or [],
               now=now)

    body = [ln for ln in (_asym_line(asym), _capital_line(cap), _attr_line(oa))
            if ln]

    if body:
        prompt_block = _PREAMBLE + "\n" + "\n".join(body)
    else:
        # NO_DATA / all-empty: an honest, short line beats an empty section or
        # a None the caller has to special-case.
        prompt_block = (
            _PREAMBLE
            + "\n  No closed round-trips or behavioural history yet — "
            "the book is fresh; nothing to mirror.")

    # Compact one-liner for logs / Discord / a future chat single-source.
    summary_bits = []
    if asym.get("verdict") or asym.get("state") in ("EMERGING", "STABLE"):
        summary_bits.append(
            f"edge={asym.get('verdict') or asym.get('state')}")
    if cap.get("state"):
        summary_bits.append(f"capital={cap.get('state')}")
    if oa.get("status"):
        summary_bits.append(f"open-vs-spy={oa.get('status')}")
    summary = " · ".join(summary_bits) if summary_bits else "no-data"

    return {
        "as_of": now.isoformat(timespec="seconds"),
        "summary": summary,
        "prompt_block": prompt_block,
        "asymmetry": asym,
        "capital": cap,
        "open_attribution": oa,
    }


if __name__ == "__main__":  # smoke test against the live DB
    import json
    from paper_trader.store import get_store
    s = get_store()
    rep = build_self_review(
        s.get_portfolio(),
        s.open_positions(),
        s.recent_trades(2000),
        s.recent_decisions(limit=3000),
        s.equity_curve(limit=5000),
    )
    print(rep["prompt_block"])
    print("\n---\n")
    print(json.dumps({k: v for k, v in rep.items() if k != "prompt_block"},
                     indent=2, default=str))
