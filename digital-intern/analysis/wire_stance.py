"""Wire-stance verdict on an arbitrary list of tickers.

``/api/held-wire-balance`` answers the per-name bull/bear question for
the *held book* — it hard-locks the ticker universe to
``ml.features.LIVE_PORTFOLIO_TICKERS`` because the chat-enrichment
contract is "is the wire bearish on a name I'm long?". That's the
right shape for that surface.

But the *operator question that comes next* sits in a different
universe: the paper-trader's scorer just emitted a deployment plan
saying "deploy $589 into MUU + KLAC". MUU and KLAC are NOT in the held
book (yet — that's the whole point of the plan). The desk needs to
answer:

  "Before I let the runner BUY MUU + KLAC, is the wire ACTUALLY
  saying anything on those names — bullish, bearish, silent? Is the
  scorer's pick corroborated by the news flow, or is the model
  picking a quant pattern the wire is fading?"

``/api/slate-news-corroboration`` on the paper-trader side answers
this with the SCORER's own ML news score (a per-row HOT_CONVERGENT /
QUANT_ONLY tag). That's the freshness verdict. What this endpoint
adds is the orthogonal *directional* verdict: even when the scorer
sees "10 articles, max ai_score 9" (HOT_CONVERGENT), is the wire's
bull/bear stance actually consistent with the scorer's BUY?

``build_wire_stance`` is a thin pure-function wrapper around
``build_held_wire_balance`` (single source of truth for the
classifier, the per-ticker verdict ladder, and the BULL/BEAR
thresholds) that:

  * REQUIRES an explicit ticker list — there is no
    ``LIVE_PORTFOLIO_TICKERS`` fallback, that's the whole point of
    splitting these endpoints.
  * Reframes the headline (``Wire stance on N name(s): ...`` instead
    of ``Held-wire balance: ...``) so the operator reading two
    panels on the same dashboard doesn't conflate them.
  * Adds a ``scope: "arbitrary"`` field so downstream consumers can
    tell the held-book and arbitrary-list reports apart.

The taxonomy (bull/bear words, MIN_CLASSIFIED_PER_TICKER, BOOK_LEAN_PCT)
is reused verbatim — if the held-book and arbitrary-list reports could
silently disagree on what "bullish" means, the operator's mental model
collapses. SSOT prevents that drift by construction.

Garbage-safe — non-list articles, missing tickers, malformed rows all
return a well-formed skeleton, never an exception. Same contract as
``build_held_wire_balance`` and ``build_sector_coherence``.

Advisory only — never gates Opus, never modifies anything. Operator
decides whether the wire's read overrules the scorer.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from analysis.held_wire_balance import (
    BOOK_LEAN_PCT,
    MIN_CLASSIFIED_PER_TICKER,
    build_held_wire_balance,
)


def _empty_skel(tickers_in: Any, now: datetime | None,
                window_hours: float | None) -> dict:
    """Well-formed empty result for invalid / empty input. Mirrors the
    shape ``build_held_wire_balance`` returns on its own skeleton path
    but with the arbitrary-list framing."""
    return {
        "generated_at": (now or datetime.now(timezone.utc))
            .isoformat(timespec="seconds"),
        "window_hours": window_hours,
        "min_classified_per_ticker": MIN_CLASSIFIED_PER_TICKER,
        "book_lean_pct": BOOK_LEAN_PCT,
        "scope": "arbitrary",
        "tickers_in": [],
        "n_scanned": 0,
        "n_held_mentions": 0,
        "n_classified": 0,
        "per_ticker": [],
        "book_verdict": "BOOK_INSUFFICIENT",
        "n_bull_lean": 0,
        "n_bear_lean": 0,
        "n_mixed": 0,
        "n_insufficient": 0,
        "headline": (
            "Wire stance: no tickers provided." if not tickers_in
            else "Wire stance: no valid tickers in input."
        ),
    }


def _normalize_tickers(tickers: Any) -> list[str]:
    """Coerce input to a list of upper-cased non-empty strings.
    Garbage / non-string / empty-string entries are dropped silently
    so a caller doing ``request.args.get('tickers').split(',')`` with
    a trailing-comma or whitespace-noise gets a clean list."""
    if not isinstance(tickers, (list, tuple, set)):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for t in tickers:
        if not isinstance(t, str):
            continue
        s = t.strip().upper()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _stance_headline(book: str, per_out: list[dict],
                      n_in: int, n_classified_names: int) -> str:
    """Reframed headline for arbitrary-ticker scope. Mirrors the
    silence-shape policy of held_wire_balance:

      * BOOK_BEAR — name the bears (operator's eye lands on misses)
      * BOOK_BULL — count only
      * BOOK_MIXED — split-vote summary
      * BOOK_INSUFFICIENT — coverage-deficit summary

    The strings deliberately differ from ``Held-wire balance:`` so the
    operator reading both panels on the same dashboard can tell them
    apart by skim. SSOT for the verdict; SSOT-but-distinct for the
    headline."""
    n_bull = sum(1 for t in per_out if t["verdict"] == "BULL_LEAN")
    n_bear = sum(1 for t in per_out if t["verdict"] == "BEAR_LEAN")
    n_mixed = sum(1 for t in per_out if t["verdict"] == "MIXED")
    n_insuf = sum(1 for t in per_out if t["verdict"] == "INSUFFICIENT")

    if book == "BOOK_BEAR":
        bear_names = [t["ticker"] for t in per_out
                      if t["verdict"] == "BEAR_LEAN"]
        return (
            f"Wire stance on {n_in} name(s): BOOK_BEAR — wire bearish "
            f"on {len(bear_names)} name(s)"
            + (f": {', '.join(bear_names[:5])}." if bear_names else ".")
        )
    if book == "BOOK_BULL":
        return (
            f"Wire stance on {n_in} name(s): BOOK_BULL — wire bullish "
            f"on {n_bull} name(s)."
        )
    if book == "BOOK_MIXED":
        opin = n_bull + n_bear + n_mixed
        return (
            f"Wire stance on {n_in} name(s): BOOK_MIXED — "
            f"{n_bull}↑/{n_bear}↓ across {opin} opinionated name(s)."
        )
    return (
        f"Wire stance on {n_in} name(s): BOOK_INSUFFICIENT — "
        f"{n_insuf} of {n_in} name(s) lack ≥{MIN_CLASSIFIED_PER_TICKER} "
        f"classified headlines."
    )


def build_wire_stance(
    articles: Any,
    tickers: Any,
    window_hours: float | None = None,
    now: datetime | None = None,
) -> dict:
    """Pure builder. Roll a list of article dicts + an arbitrary
    ticker list into the same per-name bull/bear coherence report
    ``build_held_wire_balance`` emits.

    Differences from ``build_held_wire_balance``:

      * No ``LIVE_PORTFOLIO_TICKERS`` fallback — ``tickers`` is
        REQUIRED. Caller-driven scope is the whole reason this
        endpoint exists separately.
      * Headline rephrased to "Wire stance on N name(s)" — distinct
        text from "Held-wire balance" so the two panels are
        unambiguously distinguishable on the same dashboard.
      * Adds ``scope: "arbitrary"`` + ``tickers_in`` to the output so
        downstream consumers can read which universe was scanned.

    Same SSOT for everything else: the bull/bear classifier
    (``sector_coherence._classify``), per-ticker verdict thresholds
    (``BULL_LEAN_PCT=70``, ``BEAR_LEAN_PCT=70``,
    ``MIN_CLASSIFIED_PER_TICKER=2``), book verdict thresholds
    (``BOOK_LEAN_PCT=66``, ``MIN_OPINIONATED_NAMES=2``).

    Garbage-safe; never raises."""

    tk_in = _normalize_tickers(tickers)
    if not tk_in:
        skel = _empty_skel(tickers, now, window_hours)
        # When input had non-list garbage (None, "ABC", 42), tickers_in
        # is correctly empty; when input was a list with all-non-string
        # entries, tickers_in is also empty. Both branches return the
        # same shape so the consumer never sees a partial structure.
        return skel

    # Delegate the heavy lifting to the SSOT builder. Pass the
    # normalized list as held_tickers — the builder uses it strictly
    # as the universe, not as a held-book sentinel.
    inner = build_held_wire_balance(
        articles=articles,
        held_tickers=tk_in,
        window_hours=window_hours,
        now=now,
    )

    # Reshape: rebrand the headline and add scope metadata. Other
    # fields pass through unchanged (per_ticker rows, book_verdict,
    # counts, classifier thresholds — all SSOT-preserved).
    per_out = list(inner.get("per_ticker") or [])
    book = inner.get("book_verdict") or "BOOK_INSUFFICIENT"
    headline = _stance_headline(book, per_out, len(tk_in),
                                 len(per_out))

    out = dict(inner)
    out["scope"] = "arbitrary"
    out["tickers_in"] = tk_in
    out["headline"] = headline
    return out
