"""Per-held-ticker wire-stance balance — is the wire bullish or bearish on
*your specific names* right now?

The existing chat surface answers two adjacent questions but never this one:

* ``/api/portfolio-signals`` ranks the *freshness* of articles per held
  ticker (which headline to read first) — no directional read.
* ``/api/held-news-silence`` reports *coverage* per held ticker (DARK vs
  ECHO vs COVERED) — also no directional read.
* ``/api/sector-coherence`` reports bull/bear coherence at the *sector*
  level, not per-name — a held single-name (LITE, AXTI, QBTS) does not
  necessarily ride its sector's coherence; a sector can be MACRO_BULL
  while the wire on the specific held name is bearish.

A trader sizing into a held position needs the structural read this
endpoint provides: "the wire on *this specific name* agrees with my long
bias" vs "the wire is quietly bearish on a name I'm long". Until now,
that question had to be answered by hand-reading ``/api/portfolio-signals``
output.

``build_held_wire_balance`` answers it with the *same* high-precision
word-bounded bull/bear classifier ``sector_coherence`` uses (SSOT — reuse,
don't duplicate). Per held ticker: ``BULL_LEAN`` / ``BEAR_LEAN`` /
``MIXED`` / ``INSUFFICIENT``. Roll-up to book level: ``BOOK_BULL`` /
``BOOK_BEAR`` / ``BOOK_MIXED`` / ``BOOK_INSUFFICIENT``.

Chat (silence-on-healthy, in the ``_sector_coherence_chat_lines`` mould):
only emit when at least one held name carries a ``BEAR_LEAN`` verdict —
the case where the wire opposes the desk's long bias on a held position.
``BULL_LEAN`` names are aligned-with-book good news; the operator does
not need a chat line saying "things are fine".

Observational only — never gates Opus and adds no caps (paper-trader
CLAUDE.md invariant spirit, same as ``sector_coherence``).
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

# Below this many *classified* (non-neutral) articles per ticker, the
# per-name verdict is withheld — at the per-name level, a 1-bull / 0-bear
# is meaningless (one syndicated headline). 2 is the smallest defensible
# floor; sector_coherence uses 3 across a whole sector — per-name coverage
# is strictly narrower so the floor is one lower.
MIN_CLASSIFIED_PER_TICKER = 2

# Per-name verdict thresholds: >= this share of classified ⇒ LEAN, else MIXED.
BULL_LEAN_PCT = 70.0
BEAR_LEAN_PCT = 70.0

# Book verdict: BOOK_BULL when >= this share of OPINIONATED held names lean
# bull (and BOOK_BEAR symmetric). Below MIN_OPINIONATED_NAMES the book
# verdict is BOOK_INSUFFICIENT.
BOOK_LEAN_PCT = 66.0
MIN_OPINIONATED_NAMES = 2


def _classify_title(title: str) -> str:
    """Delegate to ``sector_coherence._classify`` so the bull/bear taxonomy
    can never silently drift between the sector-level and per-name reports.
    """
    try:
        from analysis.sector_coherence import _classify
        return _classify(title)
    except Exception:  # noqa: BLE001
        return "neutral"


def _ticker_re(tickers):
    if not tickers:
        return None
    # Longest-first so the regex prefers \bMUU\b over \bMU\b — matches the
    # ``claude_analyst._BOOK_RE`` convention.
    return re.compile(
        r"\b(?:"
        + "|".join(re.escape(t) for t in
                   sorted(set(tickers), key=len, reverse=True))
        + r")\b"
    )


def _per_ticker_verdict(bull: int, bear: int) -> tuple[str, float, str]:
    classified = bull + bear
    if classified < MIN_CLASSIFIED_PER_TICKER:
        return "INSUFFICIENT", 0.0, "neutral"
    coh = max(bull, bear) / classified * 100.0
    lead = "bull" if bull > bear else ("bear" if bear > bull else "mixed")
    if coh >= BULL_LEAN_PCT and lead == "bull":
        return "BULL_LEAN", coh, "bull"
    if coh >= BEAR_LEAN_PCT and lead == "bear":
        return "BEAR_LEAN", coh, "bear"
    return "MIXED", coh, lead


def _book_verdict(per_ticker: list[dict]) -> str:
    opinionated = [
        t for t in per_ticker
        if t.get("verdict") in ("BULL_LEAN", "BEAR_LEAN", "MIXED")
    ]
    if len(opinionated) < MIN_OPINIONATED_NAMES:
        return "BOOK_INSUFFICIENT"
    n_bull = sum(1 for t in opinionated if t["verdict"] == "BULL_LEAN")
    n_bear = sum(1 for t in opinionated if t["verdict"] == "BEAR_LEAN")
    n = len(opinionated)
    if (n_bull / n * 100.0) >= BOOK_LEAN_PCT:
        return "BOOK_BULL"
    if (n_bear / n * 100.0) >= BOOK_LEAN_PCT:
        return "BOOK_BEAR"
    return "BOOK_MIXED"


def build_held_wire_balance(
    articles: Any,
    held_tickers: Any = None,
    window_hours: float | None = None,
    now: datetime | None = None,
) -> dict:
    """Pure: roll a list of live article dicts + held-ticker list into a
    per-held-ticker bull/bear coherence report.

    Garbage-safe — non-list articles, missing held_tickers, malformed rows
    all return a well-formed skeleton, never an exception. Matches the
    ``build_sector_coherence`` contract.
    """
    skel = {
        "generated_at": (now or datetime.now(timezone.utc))
            .isoformat(timespec="seconds"),
        "window_hours": window_hours,
        "min_classified_per_ticker": MIN_CLASSIFIED_PER_TICKER,
        "book_lean_pct": BOOK_LEAN_PCT,
        "n_scanned": 0,
        "n_held_mentions": 0,
        "n_classified": 0,
        "per_ticker": [],
        "book_verdict": "BOOK_INSUFFICIENT",
        "n_bull_lean": 0,
        "n_bear_lean": 0,
        "n_mixed": 0,
        "n_insufficient": 0,
        "headline": "Held-wire balance: no held tickers configured.",
    }
    if not isinstance(articles, list):
        return skel

    if held_tickers is None:
        try:
            from ml.features import LIVE_PORTFOLIO_TICKERS
            held_tickers = list(LIVE_PORTFOLIO_TICKERS)
        except Exception:  # noqa: BLE001
            held_tickers = []
    if not isinstance(held_tickers, (list, tuple, set)) or not held_tickers:
        return skel

    held = sorted({t for t in held_tickers if isinstance(t, str) and t})
    re_held = _ticker_re(held)
    if re_held is None:
        return skel

    per: dict[str, dict] = {
        t: {"bull": 0, "bear": 0, "neutral": 0,
            "lead_headline": ("", -1.0, "neutral")}
        for t in held
    }
    n_scanned = 0
    n_held_mentions = 0
    n_classified = 0

    for art in articles:
        if not isinstance(art, dict):
            continue
        n_scanned += 1
        title = art.get("title")
        if not isinstance(title, str) or not title.strip():
            continue
        matched = set(re_held.findall(title))
        if not matched:
            continue
        n_held_mentions += 1
        stance = _classify_title(title)
        if stance != "neutral":
            n_classified += 1
        try:
            ai = float(art.get("ai_score") or 0.0)
        except (TypeError, ValueError):
            ai = 0.0
        for tk in matched:
            d = per[tk]
            d[stance] = d.get(stance, 0) + 1
            if stance != "neutral" and ai > d["lead_headline"][1]:
                d["lead_headline"] = (title, ai, stance)

    per_out = []
    for tk in held:
        d = per[tk]
        verdict, coh, lead = _per_ticker_verdict(d["bull"], d["bear"])
        per_out.append({
            "ticker": tk,
            "n_bull": d["bull"],
            "n_bear": d["bear"],
            "n_neutral": d["neutral"],
            "n_classified": d["bull"] + d["bear"],
            "coherence_pct": round(coh, 1),
            "lead_direction": lead,
            "lead_headline": d["lead_headline"][0] or None,
            "verdict": verdict,
        })

    n_bull_lean = sum(1 for t in per_out if t["verdict"] == "BULL_LEAN")
    n_bear_lean = sum(1 for t in per_out if t["verdict"] == "BEAR_LEAN")
    n_mixed = sum(1 for t in per_out if t["verdict"] == "MIXED")
    n_insufficient = sum(1 for t in per_out if t["verdict"] == "INSUFFICIENT")
    book = _book_verdict(per_out)

    _RANK = {"BEAR_LEAN": 0, "MIXED": 1, "BULL_LEAN": 2, "INSUFFICIENT": 3}
    per_out.sort(
        key=lambda x: (_RANK.get(x["verdict"], 99),
                       -(x["n_bull"] + x["n_bear"]))
    )

    if book == "BOOK_BEAR":
        bear_names = [
            t["ticker"] for t in per_out if t["verdict"] == "BEAR_LEAN"
        ]
        headline = (
            f"Held-wire balance: BOOK_BEAR — wire bearish on "
            f"{len(bear_names)} held name(s)"
            + (f": {', '.join(bear_names[:5])}." if bear_names else ".")
        )
    elif book == "BOOK_BULL":
        headline = (
            f"Held-wire balance: BOOK_BULL — wire bullish on "
            f"{n_bull_lean} held name(s)."
        )
    elif book == "BOOK_MIXED":
        opin = n_bull_lean + n_bear_lean + n_mixed
        headline = (
            f"Held-wire balance: BOOK_MIXED — "
            f"{n_bull_lean}↑/{n_bear_lean}↓ across {opin} opinionated "
            f"held name(s)."
        )
    else:
        headline = (
            f"Held-wire balance: BOOK_INSUFFICIENT — "
            f"{n_insufficient} of {len(held)} held name(s) lack "
            f"≥{MIN_CLASSIFIED_PER_TICKER} classified headlines."
        )

    skel.update({
        "n_scanned": n_scanned,
        "n_held_mentions": n_held_mentions,
        "n_classified": n_classified,
        "per_ticker": per_out,
        "book_verdict": book,
        "n_bull_lean": n_bull_lean,
        "n_bear_lean": n_bear_lean,
        "n_mixed": n_mixed,
        "n_insufficient": n_insufficient,
        "headline": headline,
    })
    return skel
