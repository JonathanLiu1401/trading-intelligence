"""Per-held-ticker mention trend across recent briefings — the *which-of-my-
positions-keeps-getting-missed* trend sibling to ``briefing_coverage_audit``.

``briefing_coverage_audit`` is point-in-time: it audits ONE briefing against the
urgent flow that ran into its window and says "MU was urgent and missed by THIS
briefing". A miss in one briefing is sometimes fine — perhaps no real MU-moving
event happened that cycle. The harder failure mode is CHRONIC: a held name
absent from the LAST N briefings in a row, even though the analyst is carrying
a real position.

This is the trend axis exactly as:

* ``article_store.briefing_cadence_trend`` is the trend sibling to
  ``briefing_health``  (cadence — "is the path firing on schedule?");
* ``article_store.briefing_text_overlap_trend`` is the trend sibling to
  ``briefing_health``  (freshness — "is each briefing fresh content?");
* ``article_store.briefing_length_trend`` is the trend sibling
  (output density — "is Opus producing as much per briefing?");
* ``article_store.briefing_article_count_trend`` is the trend sibling
  (input pool — "is Opus seeing as many candidate articles?").

None of those measure *which held NAMES* Opus keeps surfacing. The
single-briefing ``briefing_coverage_audit`` measures one briefing, and the
post-briefing ``_format_portfolio_coverage`` Discord line names silent held
tickers AFTER one cycle. The TREND view — "MU has appeared in 0/10 of the
last 10 briefings" — has no surface.

Pure / no DB / no LLM. Composes a pre-fetched iterable of briefing rows
(newest-first, mirroring ``article_store.get_briefings_for_training`` and the
``ORDER BY id DESC`` convention every other briefing reader uses).
Mirrors the ``build_briefing_coverage_audit`` / ``build_news_arrival_rhythm``
discipline: dict-shaped envelope, never raises on garbage, surplus keys
ignored, advisory only, no row mutation. All four load-bearing invariants
intact by construction (no DB write, no ai_score / ml_score / score_source /
urgency touch, backtest isolation N/A — briefings table is Opus-write only
and never carries synthetic rows).
"""
from __future__ import annotations

import re
from datetime import datetime, timezone


# Mirrored verbatim from ``analysis.claude_analyst._BOOK_TICKERS`` (which is
# itself mirrored from ``daemon.PORTFOLIO_TICKERS``). The two are pinned by
# ``tests/test_briefing_book_tag.py`` and this module's own
# ``tests/test_briefing_held_mention_trend.py`` adds a drift-check so the
# three literals can't silently diverge. We duplicate rather than import
# claude_analyst because the analysis layer carries a heavy import graph
# (Anthropic SDK + subprocess wrappers) we don't want pulled in for a pure
# read-side analytics builder — same anti-import-cycle discipline as
# ``briefing_coverage_audit``.
_BOOK_TICKERS: tuple[str, ...] = (
    "LITE", "LNOK", "MUU", "DRAM", "SNDU",
    "MU", "MSFT", "AXTI", "ORCL", "TSEM", "QBTS", "NVDA",
)

# Live held/watched universe — union with config/portfolio.json's positions +
# option underlyings + sector_watchlist (same SSOT urgency_scorer /
# ml.features / analysis.claude_analyst use). Mirrors
# claude_analyst._BOOK_UNIVERSE byte-for-byte.
from ml.features import LIVE_PORTFOLIO_TICKERS as _LIVE_PORTFOLIO_TICKERS
_BOOK_UNIVERSE: tuple[str, ...] = _BOOK_TICKERS + tuple(
    sorted(set(_LIVE_PORTFOLIO_TICKERS) - set(_BOOK_TICKERS))
)
# Longest-first alternation so the regex prefers \bMUU\b over \bMU\b, and
# word-bounded so MU won't fire inside "Museum"/"Munich" — exact
# case-sensitive convention claude_analyst._BOOK_RE and
# briefing_coverage_audit._BOOK_RE already use.
_BOOK_RE = re.compile(
    r"\b(?:"
    + "|".join(re.escape(t) for t in sorted(set(_BOOK_UNIVERSE),
                                             key=len, reverse=True))
    + r")\b"
)


# Per-ticker verdict thresholds. The operator wants a quick triage signal.
#
# ``SILENT``       — appears in 0 of the last N briefings (chronic invisibility;
#                    the analyst is holding a real position Opus has never
#                    surfaced over the entire window — strongest signal).
# ``RECENT_GAP``   — appears at least once in the window, but the most-recent
#                    ``RECENT_GAP_STREAK_FLOOR`` briefings in a row all missed
#                    it. The held name HAS been surfaced historically, so the
#                    universe is correct; what changed in the last few cycles
#                    is what the operator wants to know about.
# ``SPORADIC``     — appears in < ``SPORADIC_FRACTION_FLOOR`` of the window;
#                    not silent, not in a recent gap, but uneven coverage —
#                    the operator should know which of their positions is
#                    being de-prioritized over time.
# ``COVERED``      — everything else.
RECENT_GAP_STREAK_FLOOR = 3
SPORADIC_FRACTION_FLOOR = 0.30

# Minimum window to compute a trend verdict. Fewer than 4 briefings (the
# ``briefing_length_trend`` floor) is too small a sample for "X of Y" to
# carry meaning — any single brief that names a ticker would tip a 3-row
# sample to 33%.
_MIN_BRIEFINGS = 4

# Per-card cap so the response stays cheap even when every book ticker has
# something to report. Each verdict bucket is capped independently so the
# operator-facing UI shows the worst N silent + worst N gapped without one
# tier crowding out the others.
_DEFAULT_CARD_CAP = 12


def _parse_briefing_ts(ts) -> datetime | None:
    """Best-effort timestamp parse — mirrors ``briefing_coverage_audit._parse_first_seen``
    so the two analytics builders agree on the briefing time format. ``None``
    on any unparseable value (defensive against a corrupt row that would
    otherwise crash the whole trend computation)."""
    if not ts:
        return None
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(ts).strip().replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _book_tickers_in_text(text) -> set:
    """Held/watchlist tickers (from ``_BOOK_UNIVERSE``) mentioned in ``text``.

    Same word-bounded, case-sensitive matching as
    ``briefing_coverage_audit._book_tickers_in_text`` so a regression here is
    visible to the same test patterns. Non-string / empty input returns an
    empty set rather than raising.
    """
    if not isinstance(text, str) or not text:
        return set()
    return set(_BOOK_RE.findall(text))


def _per_ticker_verdict(appearances: list[bool]) -> tuple[str, int, float]:
    """Compute (verdict, current_silence_streak, appearance_pct) for one
    ticker's mention sequence across the briefings.

    ``appearances`` is newest-first (matches the caller-supplied briefing
    order): ``[True, False, False]`` means "mentioned in the latest briefing,
    missed in the two before". A streak count is the LEADING run of False
    values, so on this input the streak is 0 (the latest carries it).
    """
    n = len(appearances)
    if n == 0:
        return "SILENT", 0, 0.0
    n_hits = sum(1 for a in appearances if a)
    appearance_pct = n_hits / n
    # Leading run of False — how many of the MOST RECENT briefings missed it.
    streak = 0
    for a in appearances:
        if a:
            break
        streak += 1
    if n_hits == 0:
        return "SILENT", streak, 0.0
    if streak >= RECENT_GAP_STREAK_FLOOR:
        return "RECENT_GAP", streak, appearance_pct
    if appearance_pct < SPORADIC_FRACTION_FLOOR:
        return "SPORADIC", streak, appearance_pct
    return "COVERED", streak, appearance_pct


def _aggregate_verdict(per_ticker_verdicts: list[str],
                       static_silent_count: int) -> str:
    """Aggregate per-ticker verdicts to one operator-facing summary.

    Ladder (most-severe-first, mirrors ``briefing_cadence_trend`` /
    ``briefing_length_trend`` discipline):

      * ``CHRONIC_SILENCE`` — at least one STATIC ``_BOOK_TICKERS`` member is
        SILENT (the analyst is carrying a hardcoded-core held position Opus
        has never mentioned in the window).
      * ``RECENT_GAP``      — at least one ticker has the RECENT_GAP verdict
        but no STATIC member is SILENT. The path historically covered the
        names; something changed in the last few cycles.
      * ``SPORADIC_COVERAGE`` — at least one ticker is SPORADIC, none
        SILENT or RECENT_GAP. Uneven but not failing.
      * ``ALL_COVERED``     — every ticker the window saw is COVERED.

    NOTE: a non-static (live-only) ticker that's SILENT downgrades to
    SPORADIC_COVERAGE rather than CHRONIC_SILENCE. The static core represents
    deliberate operator positioning; live-only entries can include
    short-lived sector watchlist names where silence is not necessarily
    actionable.
    """
    if static_silent_count > 0:
        return "CHRONIC_SILENCE"
    if "RECENT_GAP" in per_ticker_verdicts:
        return "RECENT_GAP"
    if "SPORADIC" in per_ticker_verdicts or "SILENT" in per_ticker_verdicts:
        return "SPORADIC_COVERAGE"
    return "ALL_COVERED"


def _empty_envelope(now: datetime,
                    *,
                    state: str,
                    headline: str,
                    n_briefings: int = 0,
                    card_cap: int = _DEFAULT_CARD_CAP) -> dict:
    """Skeleton envelope — same key set as a populated response so the
    UI / chat binding never sees a missing field. Mirrors
    ``briefing_coverage_audit._empty_envelope`` discipline."""
    return {
        "as_of": now.isoformat(timespec="seconds"),
        "verdict": state,
        "headline": headline,
        "n_briefings": n_briefings,
        "window_first_ts": None,
        "window_last_ts": None,
        "per_ticker": [],
        "n_silent_book": 0,
        "n_recent_gap": 0,
        "n_sporadic": 0,
        "n_covered": 0,
        "card_cap": card_cap,
        "static_book_tickers": list(_BOOK_TICKERS),
    }


def build_briefing_held_mention_trend(
    briefings,
    *,
    card_cap: int = _DEFAULT_CARD_CAP,
    now: datetime | None = None,
) -> dict:
    """Per-held-ticker mention coverage across recent briefings.

    ``briefings``: iterable of dict-shaped rows ``{ts, text, ...}``, **newest
    first** (matches ``article_store.get_briefings_for_training``'s
    ``ORDER BY id DESC`` and every other briefing reader's convention).
    Rows with no string ``text`` are silently dropped (a corrupt briefing
    row must not crash the trend computation — same garbage-safe discipline
    as ``briefing_coverage_audit``). Surplus keys ignored.

    Returns the envelope shape documented in ``_empty_envelope`` above,
    augmented with ``per_ticker`` rows for every member of
    ``_BOOK_UNIVERSE``, sorted by severity (SILENT first, then RECENT_GAP
    by streak desc, then SPORADIC by appearance_pct asc, then COVERED in
    canonical book order). Each per-ticker row carries:

        {
          "ticker":                  str,
          "is_static_book":          bool,      # in _BOOK_TICKERS core
          "appearance_pct":          float 0..1,
          "n_briefings_with":        int,
          "n_briefings":             int,
          "current_silence_streak":  int,       # leading-run of misses
          "verdict": "COVERED" | "SPORADIC" | "RECENT_GAP" | "SILENT",
        }

    Verdict ladder (aggregate ``verdict``, mirrors the conservative
    most-severe-first discipline of the trend siblings):
      * ``NO_DATA``           — fewer than ``_MIN_BRIEFINGS`` valid rows.
      * ``CHRONIC_SILENCE``   — at least one STATIC ``_BOOK_TICKERS`` member
                                 is SILENT.
      * ``RECENT_GAP``        — at least one ticker is RECENT_GAP, no static
                                 member is SILENT.
      * ``SPORADIC_COVERAGE`` — at least one ticker is SPORADIC or live-only
                                 SILENT, no RECENT_GAP / static SILENT.
      * ``ALL_COVERED``       — every ticker is COVERED.

    Pure: no DB, no LLM, never raises. Surplus keys ignored. All four
    load-bearing invariants intact by construction.
    """
    now = now or datetime.now(timezone.utc)
    card_cap = max(1, int(card_cap)) if isinstance(card_cap, int) else _DEFAULT_CARD_CAP

    # Materialise iterable + drop rows with no usable text. ``ts`` is
    # advisory — used for the window_first_ts / window_last_ts in the
    # envelope so the operator knows what window the trend covers.
    valid_texts: list[str] = []
    valid_ts: list[datetime | None] = []
    try:
        seq = list(briefings) if briefings is not None else []
    except TypeError:
        return _empty_envelope(
            now,
            state="NO_DATA",
            headline="briefings argument is not iterable — nothing to trend.",
            card_cap=card_cap,
        )
    for row in seq:
        if not isinstance(row, dict):
            continue
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        valid_texts.append(text)
        valid_ts.append(_parse_briefing_ts(row.get("ts")))

    n_briefings = len(valid_texts)
    if n_briefings < _MIN_BRIEFINGS:
        return _empty_envelope(
            now,
            state="NO_DATA",
            headline=(f"Only {n_briefings} usable briefing(s) — need at least "
                      f"{_MIN_BRIEFINGS} for a held-mention trend."),
            n_briefings=n_briefings,
            card_cap=card_cap,
        )

    # Build per-briefing token sets ONCE (each ticker scan is O(1) lookups
    # afterwards). ``texts`` are already newest-first.
    per_briefing_tickers: list[set] = [
        _book_tickers_in_text(t) for t in valid_texts
    ]

    # Per-ticker walk: build the appearance sequence + verdict.
    per_ticker: list[dict] = []
    static_set = set(_BOOK_TICKERS)
    for ticker in _BOOK_UNIVERSE:
        appearances = [ticker in tickers for tickers in per_briefing_tickers]
        verdict, streak, pct = _per_ticker_verdict(appearances)
        n_hits = sum(1 for a in appearances if a)
        per_ticker.append({
            "ticker": ticker,
            "is_static_book": ticker in static_set,
            "appearance_pct": round(pct, 3),
            "n_briefings_with": n_hits,
            "n_briefings": n_briefings,
            "current_silence_streak": streak,
            "verdict": verdict,
        })

    # Aggregate verdict.
    static_silent = sum(
        1 for r in per_ticker
        if r["verdict"] == "SILENT" and r["is_static_book"]
    )
    aggregate = _aggregate_verdict([r["verdict"] for r in per_ticker],
                                   static_silent_count=static_silent)

    # Counters for the envelope.
    n_silent = sum(1 for r in per_ticker if r["verdict"] == "SILENT")
    n_recent_gap = sum(1 for r in per_ticker if r["verdict"] == "RECENT_GAP")
    n_sporadic = sum(1 for r in per_ticker if r["verdict"] == "SPORADIC")
    n_covered = sum(1 for r in per_ticker if r["verdict"] == "COVERED")

    # Sort severity-first then by within-tier ranking — operator wants the
    # SILENT static names at the very top of the card.
    severity_rank = {"SILENT": 0, "RECENT_GAP": 1, "SPORADIC": 2, "COVERED": 3}
    book_order = {t: i for i, t in enumerate(_BOOK_UNIVERSE)}

    def _sort_key(r: dict) -> tuple:
        sev = severity_rank.get(r["verdict"], 4)
        # Within SILENT: static-book first (those are the actionable ones).
        # Within RECENT_GAP: longest streak first (most-stale held name).
        # Within SPORADIC: lowest appearance_pct first.
        # Within COVERED: canonical book order (stable).
        if r["verdict"] == "SILENT":
            return (sev, 0 if r["is_static_book"] else 1, book_order.get(r["ticker"], 99))
        if r["verdict"] == "RECENT_GAP":
            return (sev, -r["current_silence_streak"], book_order.get(r["ticker"], 99))
        if r["verdict"] == "SPORADIC":
            return (sev, r["appearance_pct"], book_order.get(r["ticker"], 99))
        return (sev, book_order.get(r["ticker"], 99))

    per_ticker_sorted = sorted(per_ticker, key=_sort_key)
    per_ticker_capped = per_ticker_sorted[:card_cap]

    # Headline tailored to the aggregate verdict — operator-readable single
    # sentence the chat / dashboard surface can render verbatim. Mirrors the
    # ``briefing_coverage_audit`` headline discipline (no f-string formatting
    # of None, no ambiguous units).
    window_first_ts = valid_ts[-1].isoformat(timespec="seconds") if valid_ts[-1] else None
    window_last_ts = valid_ts[0].isoformat(timespec="seconds") if valid_ts[0] else None

    if aggregate == "ALL_COVERED":
        headline = (f"All {len(_BOOK_UNIVERSE)} held/watched tickers appear "
                    f"healthily in the last {n_briefings} briefings.")
    elif aggregate == "CHRONIC_SILENCE":
        silent_static = [r["ticker"] for r in per_ticker_sorted
                         if r["verdict"] == "SILENT" and r["is_static_book"]]
        # Cap displayed names so the headline is one line.
        shown = silent_static[:4]
        headline = (f"{len(silent_static)} held position(s) "
                    f"({', '.join(shown)}{'…' if len(silent_static) > 4 else ''}) "
                    f"missing from ALL {n_briefings} recent briefings.")
    elif aggregate == "RECENT_GAP":
        gap_rows = [r for r in per_ticker_sorted if r["verdict"] == "RECENT_GAP"]
        shown = [f"{r['ticker']} (last {r['current_silence_streak']})"
                 for r in gap_rows[:3]]
        headline = (f"{n_recent_gap} held name(s) absent from the latest "
                    f"{RECENT_GAP_STREAK_FLOOR}+ briefings: "
                    f"{', '.join(shown)}{'…' if len(gap_rows) > 3 else ''}.")
    else:  # SPORADIC_COVERAGE
        sporadic_rows = [r for r in per_ticker_sorted
                         if r["verdict"] in ("SPORADIC", "SILENT")]
        shown = [f"{r['ticker']} ({int(r['appearance_pct']*100)}%)"
                 for r in sporadic_rows[:3]]
        headline = (f"{n_sporadic + n_silent} held name(s) covered sporadically: "
                    f"{', '.join(shown)}{'…' if len(sporadic_rows) > 3 else ''}.")

    return {
        "as_of": now.isoformat(timespec="seconds"),
        "verdict": aggregate,
        "headline": headline,
        "n_briefings": n_briefings,
        "window_first_ts": window_first_ts,
        "window_last_ts": window_last_ts,
        "per_ticker": per_ticker_capped,
        "n_silent_book": n_silent,
        "n_recent_gap": n_recent_gap,
        "n_sporadic": n_sporadic,
        "n_covered": n_covered,
        "card_cap": card_cap,
        "static_book_tickers": list(_BOOK_TICKERS),
    }
