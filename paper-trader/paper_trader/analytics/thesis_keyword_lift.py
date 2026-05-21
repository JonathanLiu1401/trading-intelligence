"""Thesis keyword lift — which words in entry reasoning correlate with wins vs losses.

The complement to ``catalyst_class_autopsy``: that builder labels each closed
round-trip with a *fixed taxonomy* of catalyst classes (ML_ADVISOR, EARNINGS_PLAY,
TECHNICALS, …) chosen by the analytics author. ``build_thesis_keyword_lift`` is
the *open-vocabulary* mirror: it learns the dominant keywords directly from the
trader's own ``entry_reason`` text. The pattern "trades whose entry reason
mentions 'guidance' win 80% of the time vs 50% baseline" surfaces here even if
no one wrote a CATALYST_GUIDANCE class for it.

Single source of truth: it consumes ``round_trips.build_round_trips`` and joins
the verbatim ``entry_reason`` back from the contributing trade rows by DB ``id``
(the loser/winner_autopsy + thesis_drift discipline — surface the reason
verbatim, never NLP-parse it for trading logic). Pure / no DB / no LLM — never
raises on garbage inputs. Advisory / diagnostic only: never gates Opus, never
injected into the decision prompt, no caps (AGENTS.md #2/#12).

Lift definition (``lift_pp``): for each keyword K that appears in at least
``min_kw_occurrences`` closed round-trips,

  win_rate(K)        = n_winners_with_K / n_trips_with_K
  baseline_win_rate  = n_winners_total / (n_winners_total + n_losers_total)
  lift_pp            = (win_rate(K) - baseline) * 100        # percentage points

Positive lift_pp = winning pattern; negative = losing pattern. lift_pp is
percentage-point delta (not multiplicative ratio) so a 0-winner keyword and an
all-winner keyword have well-defined finite lifts. Tied lifts break by sample
size (more occurrences first), then alphabetical so card order is stable.

Sample-size honesty mirrors ``winner_autopsy`` / ``loser_autopsy`` /
``trade_asymmetry``:

* ``NO_DATA``  — no closed round-trips
* ``NO_WINS``  — round-trips exist but every one lost
* ``NO_LOSSES`` — round-trips exist but every one won
* ``EMERGING`` — ``n_winners < STABLE_MIN_PER_SIDE`` OR ``n_losers <
  STABLE_MIN_PER_SIDE`` (numerics emitted, ``verdict=None``)
* ``STABLE``   — both sides at least ``STABLE_MIN_PER_SIDE`` (verdict surfaces)

The ``verdict`` field is the single most-positive-lift keyword once STABLE — a
one-word pattern label the operator can scan instantly.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

from .round_trips import build_round_trips

# Both sides need at least this many closed round-trips for the lift verdict
# to be reportable. Below this any "pattern" is fragile — a single new
# winner/loser flips the modal keyword.
STABLE_MIN_PER_SIDE = 4

# Default minimum count across (wins + losses) for a keyword to be eligible
# for the rankings. A 1-occurrence keyword has a degenerate lift (100pp or
# -baseline) that swamps the panel — mirrors min_articles=2 in
# news_corroboration and the catalyst_class STABLE_MIN_TRIPS_PER_CLASS=4 floor.
DEFAULT_MIN_KW_OCCURRENCES = 3

# Default cap on the two ranked output lists. The aggregate counts are
# returned in full; this only caps the displayed cards.
DEFAULT_TOP_N = 10

# Minimum token length. Shorter words ("a", "of", "in") leak through the
# stopword list otherwise — and a one-character token has no semantic content.
MIN_TOKEN_LEN = 3

# Stopwords that survive the length filter. Trading-text-specific (the
# generic English list under-covers "the bot", "this trade", etc. that
# clutter open_reason text). Authored conservatively — anything debatable
# is kept so the operator can see it surfaced. Frozen on import so a test
# that mutates the set doesn't bleed across runs.
_STOPWORDS = frozenset({
    # Articles / pronouns / common verbs
    "the", "and", "but", "for", "with", "this", "that", "from", "into",
    "are", "was", "were", "been", "being", "have", "has", "had", "will",
    "would", "could", "should", "may", "might", "can", "any", "all",
    "some", "more", "most", "less", "least", "very", "just", "than",
    "then", "now", "still", "also", "too", "yet", "only", "even",
    "such", "off", "out", "its", "their", "his", "her", "our",
    # Trading boilerplate phrases that show up in nearly every reason
    "buy", "sell", "long", "short", "trade", "position", "stock",
    "shares", "share", "open", "close", "entry", "exit", "hold",
    "holding", "held", "ticker", "price", "level", "market", "today",
    "tomorrow", "yesterday", "session", "morning", "afternoon",
    "decision", "decisions", "trader",
    # Generic descriptors (don't add information about WHY)
    "good", "bad", "high", "low", "big", "small", "new", "old",
    "fresh", "current", "recent", "previous", "next", "last", "first",
    "going", "looking", "look", "see", "saw", "seen", "want", "wanted",
    "need", "needs", "needed", "make", "made", "makes", "take", "took",
    "taken", "taking", "give", "given", "get", "got", "gets",
    "putting", "put", "pulls", "pull", "pulled",
    # Pure linkers
    "because", "since", "while", "where", "when", "what", "which",
    "who", "whom", "whose", "how", "why", "here", "there", "via",
    "between", "before", "after", "during", "above", "below", "under",
    "over", "again", "back", "down", "across", "around", "near", "far",
    "per", "vs",
})

# Matches a contiguous run of word characters (alphanumerics + underscore).
# Apostrophes are NOT included so "won't" → "won" (which is then filtered
# as a stopword if listed; otherwise it's a 3-char real token).
_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text) -> set[str]:
    """Lowercase, word-split, drop stopwords + numeric-only + short tokens.

    Returns a *set* — each keyword counts at most once per round-trip even if
    the entry reason mentions it five times (an analyst writing
    'earnings, earnings, earnings' shouldn't get triple credit). Tolerates
    non-string input (returns an empty set rather than raising).
    """
    if not isinstance(text, str) or not text:
        return set()
    out: set[str] = set()
    for tok in _WORD_RE.findall(text.lower()):
        if len(tok) < MIN_TOKEN_LEN:
            continue
        if tok in _STOPWORDS:
            continue
        # Pure-numeric tokens ("2026", "100", "75") have no thesis content.
        # Mixed alphanum ("q1", "rsi60") survive — they're often the
        # actual signal language ("rsi" alone passes too).
        if tok.isdigit():
            continue
        out.add(tok)
    return out


def _entry_reason_for(rt: dict, by_id: dict) -> str | None:
    """Verbatim entry reason. Same convention as winner/loser autopsy:
    the FIRST entry trade carries the opening thesis; later add-on buys
    refine it but the first is canonical."""
    eids = rt.get("entry_trade_ids") or []
    if not eids:
        return None
    row = by_id.get(eids[0])
    if not row:
        return None
    r = row.get("reason")
    return r if (r is not None and str(r).strip() != "") else None


def build_thesis_keyword_lift(trades,
                              top_n: int = DEFAULT_TOP_N,
                              min_kw_occurrences: int = DEFAULT_MIN_KW_OCCURRENCES,
                              now: datetime | None = None) -> dict:
    """Open-vocabulary keyword-lift across closed round-trips.

    ``trades`` must be ``Store.recent_trades()``-shaped ordered
    **oldest→newest** — exactly what ``/api/loser-autopsy`` and
    ``/api/catalyst-class-autopsy`` pass: ``list(reversed(
    store.recent_trades(2000)))``. Non-list / None inputs collapse to the
    empty skeleton (never raises).

    ``top_n`` caps each of the two ranked output lists (the aggregate counts
    are returned in full so the operator can drill).

    ``min_kw_occurrences`` is the floor on (n_winners_with_kw + n_losers_with_kw)
    for the keyword to be ranked. Defaults to 3.
    """
    now = now or datetime.now(timezone.utc)

    # Defensive: catalyst_class_autopsy + winner_autopsy + loser_autopsy all
    # delegate to build_round_trips, which sorts/scans the ledger. A
    # non-list input here would crash inside it; mirror their tolerance.
    if not isinstance(trades, list):
        trades = []

    rts = build_round_trips(trades)
    n_rts = len(rts)

    by_id: dict = {}
    for t in trades:
        if not isinstance(t, dict):
            continue
        tid = t.get("id")
        if tid is not None:
            by_id[tid] = t

    # Strict > 0 winner / < 0 loser — same convention as round_trips/#10 &
    # winner_autopsy / loser_autopsy / trade_asymmetry (sub-cent washes
    # read as non-win and non-loss; they are skipped entirely here, which
    # is the right call: a wash contributes no information about whether
    # any keyword is "winning" or "losing").
    winners: list[dict] = []
    losers: list[dict] = []
    for rt in rts:
        pnl = rt.get("pnl_usd") or 0.0
        if pnl > 0:
            winners.append(rt)
        elif pnl < 0:
            losers.append(rt)

    n_winners = len(winners)
    n_losers = len(losers)
    n_decisive = n_winners + n_losers  # excludes washes

    # Per-keyword (n_winners_with_kw, n_losers_with_kw) tally. A keyword
    # tracked here exists in at least ONE round-trip's tokenised entry
    # reason — the min_kw_occurrences filter is applied to the OUTPUT.
    win_counts: dict[str, int] = {}
    for rt in winners:
        for kw in _tokenize(_entry_reason_for(rt, by_id)):
            win_counts[kw] = win_counts.get(kw, 0) + 1
    loss_counts: dict[str, int] = {}
    for rt in losers:
        for kw in _tokenize(_entry_reason_for(rt, by_id)):
            loss_counts[kw] = loss_counts.get(kw, 0) + 1

    all_keywords = set(win_counts) | set(loss_counts)

    baseline_win_rate = (n_winners / n_decisive) if n_decisive else 0.0

    rows: list[dict] = []
    for kw in all_keywords:
        nw = win_counts.get(kw, 0)
        nl = loss_counts.get(kw, 0)
        n_total = nw + nl
        if n_total < min_kw_occurrences:
            continue
        win_rate_kw = nw / n_total
        # Percentage-point delta vs the global baseline. A keyword in a
        # 50/50 baseline that appears in 4 wins / 0 losses gets lift_pp =
        # +50.0; a keyword in 4 losses / 0 wins gets -50.0. Symmetric and
        # bounded.
        lift_pp = round((win_rate_kw - baseline_win_rate) * 100.0, 2)
        rows.append({
            "keyword": kw,
            "n_winners": nw,
            "n_losers": nl,
            "n_total": n_total,
            "win_rate_pct": round(win_rate_kw * 100.0, 2),
            "lift_pp": lift_pp,
        })

    # Winning-pattern ranking: highest lift first, ties broken by sample
    # size (more support is more convincing) then alphabetical (stable).
    top_winning = sorted(
        rows,
        key=lambda r: (-r["lift_pp"], -r["n_total"], r["keyword"]),
    )
    # Losing-pattern ranking: most negative lift first.
    top_losing = sorted(
        rows,
        key=lambda r: (r["lift_pp"], -r["n_total"], r["keyword"]),
    )

    # ----- state / verdict gate ----------------------------------------
    if n_rts == 0:
        state = "NO_DATA"
    elif n_winners == 0:
        state = "NO_WINS"
    elif n_losers == 0:
        state = "NO_LOSSES"
    elif n_winners >= STABLE_MIN_PER_SIDE and n_losers >= STABLE_MIN_PER_SIDE:
        state = "STABLE"
    else:
        state = "EMERGING"

    verdict = None
    if state == "STABLE" and top_winning:
        # Only emit the verdict if the top keyword actually has positive
        # lift — a degenerate "all keywords lose" pool returns None
        # (which the headline turns into "no winning pattern yet").
        if top_winning[0]["lift_pp"] > 0:
            verdict = top_winning[0]["keyword"]

    # ----- headline ----------------------------------------------------
    if state == "NO_DATA":
        headline = ("No closed round-trips yet — no thesis keywords to "
                    "rank.")
    elif state == "NO_WINS":
        headline = (f"No winning round-trips across {n_rts} closed — "
                    "every keyword is a losing pattern by definition.")
    elif state == "NO_LOSSES":
        headline = (f"No losing round-trips across {n_rts} closed — "
                    "every keyword is a winning pattern by definition.")
    else:
        base_pct = round(baseline_win_rate * 100.0, 1)
        if state == "EMERGING":
            headline = (
                f"Emerging — {n_winners}W/{n_losers}L (need "
                f"{STABLE_MIN_PER_SIDE} of each for a stable verdict). "
                f"Baseline win rate {base_pct}%."
            )
        elif verdict is None:
            headline = (
                f"STABLE pool ({n_winners}W/{n_losers}L) but no keyword "
                f"clears the {base_pct}% baseline by a positive margin."
            )
        else:
            top = top_winning[0]
            headline = (
                f"'{verdict}' is the dominant winning keyword — appears "
                f"in {top['n_winners']} of {top['n_total']} round-trips "
                f"({top['win_rate_pct']}% win rate vs {base_pct}% "
                f"baseline; +{top['lift_pp']}pp lift)."
            )

    return {
        "as_of": now.isoformat(timespec="seconds"),
        "state": state,
        "verdict": verdict,
        "headline": headline,
        "n_round_trips": n_rts,
        "n_winners": n_winners,
        "n_losers": n_losers,
        "n_decisive": n_decisive,
        "baseline_win_rate_pct": round(baseline_win_rate * 100.0, 2),
        "min_kw_occurrences": min_kw_occurrences,
        "stable_min_per_side": STABLE_MIN_PER_SIDE,
        "n_distinct_keywords": len(rows),
        "top_winning_keywords": top_winning[:max(0, top_n)],
        "top_losing_keywords": top_losing[:max(0, top_n)],
    }


if __name__ == "__main__":  # smoke test against the live DB
    import json

    from paper_trader.store import get_store
    s = get_store()
    rep = build_thesis_keyword_lift(list(reversed(s.recent_trades(2000))))
    print(json.dumps(rep, indent=2, default=str))
