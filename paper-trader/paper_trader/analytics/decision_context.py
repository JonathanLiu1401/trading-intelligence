"""Decision context — *what is the live trader actually being shown right
now?*

The single biggest blind spot on this desk: the `decisions` table stores
only `action_taken` + `reasoning`, and the only raw capture is 1000 chars
of the *response* on a parse failure (`strategy.RAW_CAPTURE_CHARS`). When
the live trader spends cycle after cycle on `NO_DECISION (timeout/empty)`
or a flat `HOLD` (the dominant 2026-05-17 live pattern), an operator has
**no way to see the decision *input*** — the prompt Opus received, how many
signals were in it, how many watchlist prices yfinance actually resolved.
Every one of the ~45 endpoints diagnoses the *output*; none show the input.

This reconstructs the live decision context on demand:

* `build_decision_context` is **pure**. It renders the prompt through the
  *same* `strategy._build_payload` the live `decide()` uses and applies the
  identical `SYSTEM_PROMPT` / `ML ADVISOR` framing, so the string is
  byte-identical to the live prompt given identical inputs (single source
  of truth, AGENTS.md invariant #10 — no re-implemented prompt). It
  **never** invokes `_claude_call`.
* `assemble_inputs` is the orchestration half (the
  `thesis_drift`/`correlation` "network in the endpoint, builder takes the
  dicts" split): it mirrors `strategy.decide()`'s pre-`_claude_call`
  assembly using the **read-only** `strategy.portfolio_snapshot_readonly`
  (so inspecting from the dashboard thread can never mutate the live
  trader's persisted marks / equity_curve) and the same advisory builders,
  each wrapped non-fatally exactly as `decide()` wraps them. It is shared
  by the `/api/decision-context` endpoint and the `__main__` CLI so the
  two can't drift.

`feed_state` distils the upstream health the operator actually needs:
`BLIND` (the prompt carried **zero** signals — a HOLD here is forced, not
chosen), `DEGRADED` (half or more of the watchlist prices were missing —
yfinance starvation, the same root cause as the timeout storms), or `OK`.

Advisory only. It mints no opinion, gates nothing, adds no caps, and is
**not** injected into the live decision prompt — dashboard / chat / CLI
only (AGENTS.md invariants #2/#12; the `desk_pulse` router precedent).
"""
from __future__ import annotations

from datetime import datetime, timezone

from paper_trader import strategy

from .mark_integrity import build_mark_integrity

#: Default upper bound on the rendered-prompt string in the JSON response.
#: Matches the bounded-payload discipline of the other SWR endpoints
#: (`/api/state` is ~145KB; this caps the prompt mirror well under that).
MAX_PROMPT_CHARS = 40_000

#: Missing-watchlist-price share at or above which the feed is `DEGRADED`
#: (yfinance starvation — the documented NO_DECISION-storm root cause).
DEGRADED_MISSING_RATIO = 0.5


def _resolve(d: dict) -> dict:
    """{key: price|None} → resolution summary (insertion order preserved so
    the missing list is deterministic)."""
    total = len(d)
    missing = [k for k, v in d.items() if v is None]
    return {
        "n_total": total,
        "n_resolved": total - len(missing),
        "n_missing": len(missing),
        "missing": missing,
    }


def build_decision_context(
    snapshot: dict,
    merged_signals: list[dict],
    top_signals: list[dict],
    urgent: list[dict],
    sentiments: list[dict],
    watch_prices: dict[str, float | None],
    futures_prices: dict[str, float | None],
    sp500: float | None,
    market_open: bool,
    quant_signals: dict[str, dict] | None = None,
    self_review_block: str | None = None,
    track_record_block: str | None = None,
    risk_mirror_block: str | None = None,
    ml_opinion_block: str | None = None,
    max_prompt_chars: int = MAX_PROMPT_CHARS,
) -> dict:
    """Reconstruct the live decision prompt + an input summary. Pure; the
    model is never called (AGENTS.md #2/#12 — advisory only)."""
    # The exact same render the live decide() performs. _build_payload's 2nd
    # positional is the *merged* signal list (urgent-not-in-top + top), so we
    # pass merged_signals there — identical to strategy.decide().
    payload = strategy._build_payload(
        snapshot, merged_signals, sentiments, watch_prices, futures_prices,
        sp500, market_open, quant_signals=quant_signals,
        self_review_block=self_review_block,
        track_record_block=track_record_block,
        risk_mirror_block=risk_mirror_block,
    )
    prompt = f"{strategy.SYSTEM_PROMPT}\n\n---\nCONTEXT:\n{payload}"
    if ml_opinion_block:
        prompt += f"\n\n---\nML ADVISOR:\n{ml_opinion_block}"

    prompt_chars = len(prompt)
    truncated = prompt_chars > max_prompt_chars
    prompt_out = prompt[:max_prompt_chars] if truncated else prompt

    n_merged = len(merged_signals)
    wl = _resolve(watch_prices)
    feed_state, feed_headline = _feed_verdict(n_merged, wl)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "market_open": market_open,
        "claude_invoked": False,
        "note": ("Dry-run reconstruction of the live decision context. "
                 "_claude_call is NOT invoked; this is not a persisted past "
                 "cycle — it is what decide() would feed Opus right now."),
        "prompt": prompt_out,
        "prompt_chars": prompt_chars,
        "prompt_truncated": truncated,
        "input_summary": {
            "n_top_signals": len(top_signals),
            "n_urgent": len(urgent),
            "n_merged_signals": n_merged,
            # the exact value decide() records into decisions.signal_count
            "signal_count": n_merged,
            "watchlist": wl,
            "futures": _resolve(futures_prices),
            "sp500_resolved": sp500 is not None,
            "n_quant_tickers": len(quant_signals or {}),
            "n_sentiment_mentions": sum(
                1 for r in sentiments if (r.get("n") or 0) > 0),
        },
        "advisory_blocks": {
            "self_review": bool(self_review_block),
            "track_record": bool(track_record_block),
            "risk_mirror": bool(risk_mirror_block),
            "ml_opinion": bool(ml_opinion_block),
        },
        "mark_integrity": build_mark_integrity(snapshot.get("positions") or []),
        "feed_state": feed_state,
        "feed_headline": feed_headline,
    }


def _feed_verdict(n_merged: int, wl: dict) -> tuple[str, str]:
    if n_merged == 0:
        return ("BLIND", "Prompt carried 0 signals — the trader is flying "
                "blind this cycle (HOLD forced by an empty feed, not chosen).")
    total = wl["n_total"]
    if total and wl["n_missing"] / total >= DEGRADED_MISSING_RATIO:
        return ("DEGRADED",
                f"{wl['n_missing']}/{total} watchlist prices missing — "
                "yfinance starvation (the NO_DECISION-storm root cause).")
    return ("OK", f"{n_merged} signal(s) in prompt; "
            f"{wl['n_resolved']}/{total} watchlist prices resolved.")


def assemble_inputs(store) -> dict:
    """Mirror `strategy.decide()`'s pre-`_claude_call` assembly, read-only.

    Every fetch / advisory-builder call is the *same* one `decide()` makes
    (so the reconstructed prompt is faithful), but the snapshot is the
    write-free `portfolio_snapshot_readonly` and each advisory block is
    wrapped non-fatally exactly as `decide()` wraps it — a diagnostics fault
    degrades that one block to absent, never an exception. Returns the kwargs
    for `build_decision_context`."""
    from paper_trader import market, signals

    market_open = market.is_market_open()
    snap = strategy.portfolio_snapshot_readonly(store)

    top = signals.get_top_signals(20, hours=2, min_score=4.0)
    urgent = signals.get_urgent_articles(minutes=30)
    sents = signals.ticker_sentiments(strategy.WATCHLIST, hours=4)
    watch_px = market.get_prices(strategy.WATCHLIST)
    fut_px = {f: market.get_futures_price(f) for f in strategy.FUTURES}
    sp500 = market.benchmark_sp500()

    held = sorted({p["ticker"] for p in snap["positions"]})
    quant_tickers = sorted(set(strategy.QUANT_TICKERS_LIVE) | set(held))
    try:
        quant_sigs = strategy.get_quant_signals_live(quant_tickers)
    except Exception:
        quant_sigs = {}

    seen = {s["id"] for s in top}
    merged = [a for a in urgent if a["id"] not in seen] + top

    self_review_block = None
    try:
        from .self_review import build_self_review
        self_review_block = build_self_review(
            store.get_portfolio(), store.open_positions(),
            store.recent_trades(2000), store.recent_decisions(limit=3000),
            store.equity_curve(limit=5000),
        ).get("prompt_block")
    except Exception:
        pass

    track_record_block = None
    try:
        from .track_record import build_track_record
        track_record_block = build_track_record(
            list(reversed(store.recent_trades(2000))),
            names=strategy._names_in_play(
                snap.get("positions") or [], merged, strategy.WATCHLIST),
        ).get("prompt_block")
    except Exception:
        pass

    risk_mirror_block = None
    try:
        from .risk_mirror import build_risk_mirror
        risk_mirror_block = build_risk_mirror(
            store.recent_trades(2000), snap.get("positions") or [],
        ).get("prompt_block")
    except Exception:
        pass

    ml_opinion_block = None
    try:
        ml_qualified, ml_qual_reason = strategy._ml_is_qualified()
        if ml_qualified:
            ml_op = strategy._ml_live_opinion(merged, quant_sigs, snap, watch_px)
            if ml_op:
                ml_opinion_block = (
                    f"ML MODEL OPINION ({ml_qual_reason}):\n"
                    f"  Action: {ml_op['action']}"
                    + (f" {ml_op['ticker']}" if ml_op.get("ticker") else "")
                    + f"\n  Reasoning: {ml_op['reasoning']}\n"
                    "This is an advisory opinion only. You retain full "
                    "autonomy over the final decision."
                )
    except Exception:
        pass

    return dict(
        snapshot=snap, merged_signals=merged, top_signals=top, urgent=urgent,
        sentiments=sents, watch_prices=watch_px, futures_prices=fut_px,
        sp500=sp500, market_open=market_open, quant_signals=quant_sigs,
        self_review_block=self_review_block,
        track_record_block=track_record_block,
        risk_mirror_block=risk_mirror_block,
        ml_opinion_block=ml_opinion_block,
    )


if __name__ == "__main__":  # works even when :8090 is wedged (desk_pulse precedent)
    import json
    import sys

    from paper_trader.store import get_store

    ctx = build_decision_context(**assemble_inputs(get_store()))
    if "--json" in sys.argv:
        print(json.dumps(ctx, indent=2, default=str))
    else:
        s = ctx["input_summary"]
        wl, fut = s["watchlist"], s["futures"]
        print(f"DECISION CONTEXT  [{ctx['feed_state']}]  {ctx['feed_headline']}")
        print(f"  signals: top={s['n_top_signals']} urgent={s['n_urgent']} "
              f"merged/signal_count={s['signal_count']}  "
              f"sentiment mentions={s['n_sentiment_mentions']}")
        print(f"  prices: watchlist {wl['n_resolved']}/{wl['n_total']} "
              f"resolved (missing {wl['missing']}); "
              f"futures {fut['n_resolved']}/{fut['n_total']}; "
              f"S&P {'ok' if s['sp500_resolved'] else 'MISSING'}")
        a = ctx["advisory_blocks"]
        print(f"  blocks: self_review={a['self_review']} "
              f"track_record={a['track_record']} "
              f"risk_mirror={a['risk_mirror']} ml={a['ml_opinion']}")
        mi = ctx["mark_integrity"]
        print(f"  marks: [{mi['verdict']}] {mi['headline']}")
        print(f"  prompt: {ctx['prompt_chars']} chars"
              + ("  (truncated in JSON view)" if ctx["prompt_truncated"] else ""))
        if "--full" in sys.argv:
            print("\n" + "=" * 70 + "\n" + ctx["prompt"])
