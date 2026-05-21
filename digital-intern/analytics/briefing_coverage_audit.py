"""Briefing coverage audit — did the last 5h briefing actually mention the
tickers that fired urgent during its window?

This is the *retrospective* sibling of the analytics that enrich the briefing
prompt before it's generated:

* ``analysis.claude_analyst._coverage_gap_lines`` — *prospective* dark-intel
  channels to mention. Tells Opus what to talk about; doesn't verify
  Opus listened.
* ``analysis.claude_analyst._book_silence_lines`` — *prospective* held
  tickers with zero stories. Tells Opus to mark them N/A; doesn't verify
  he did.
* The post-briefing ``_format_portfolio_coverage`` Discord line — names
  silent held tickers AFTER the fact, but is appended outside Opus's text.

This builder closes the loop on the OTHER side: given the published briefing
text and the urgent flow that ran into its window, classify each book ticker
that *had urgent news* as COVERED (mentioned in the briefing) or MISSED
(absent from the briefing despite urgent stories). The operator question:
"the alerts fired all night — did the morning briefing surface those
tickers, or did Opus draft around them?"

Pure / no DB / no LLM — composes a pre-fetched ``briefing`` row + iterable
of urgent ``articles``. Mirrors the ``build_news_arrival_rhythm`` /
``build_event_threads`` discipline: dict-shaped envelope, never raises on
garbage, surplus keys ignored, advisory only.

Universe of tickers checked is ``_BOOK_TICKERS`` (held + sector watchlist),
the same canonical set the briefing's own ``_book_tickers`` /
``_book_heat_lines`` use. Tickers outside that set don't enter the audit —
a stray NEM or BABA appearing once in the urgent flow isn't the operator's
problem when the *book* tickers are what Opus is supposed to defend.

Window semantics: the urgent ``articles`` list is the caller's
responsibility — the typical caller pulls everything ``urgency >= 1`` with
``first_seen`` between the prior briefing's ``ts`` and the current
briefing's ``ts``. The builder is window-agnostic; it scores whatever it
receives. When the route layer can't find a prior briefing it falls back to
a 5h lookback from the current briefing (the heartbeat cadence).
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

# Mirrored verbatim from ``analysis.claude_analyst._BOOK_TICKERS`` (which is
# itself mirrored from ``daemon.PORTFOLIO_TICKERS``). The two are pinned by
# ``tests/test_briefing_book_tag.py``; this module duplicates the literal
# rather than importing claude_analyst because the analysis layer carries
# a heavy import graph (Anthropic SDK + subprocess wrappers) we don't want
# pulled in for a pure read-side analytics builder. A drift-check is added
# to tests/test_briefing_coverage_audit.py so the two literals can't
# silently diverge.
_BOOK_TICKERS: tuple[str, ...] = (
    "LITE", "LNOK", "MUU", "DRAM", "SNDU",
    "MU", "MSFT", "AXTI", "ORCL", "TSEM", "QBTS", "NVDA",
)
# Longest-first alternation so the regex prefers ``\bMUU\b`` over ``\bMU\b``,
# matching ``_BOOK_RE`` in claude_analyst exactly.
_BOOK_RE = re.compile(
    r"\b(?:"
    + "|".join(re.escape(t) for t in sorted(set(_BOOK_TICKERS),
                                             key=len, reverse=True))
    + r")\b"
)

# Per-card per-side cap so the response stays cheap even when a 5h window
# fires urgent flow on every book ticker. The two states each get this many
# rows; the totals still reflect the full set.
_DEFAULT_CARD_CAP = 12

# Coverage state thresholds. The operator wants a quick triage signal:
# COMPLETE ≥ 80% — Opus surfaced almost everything the alerts caught.
# PARTIAL  ≥ 50% — meaningful gap; worth checking which names were dropped.
# THIN     < 50% — the briefing materially diverged from the urgent flow.
_COMPLETE_FLOOR = 0.80
_PARTIAL_FLOOR = 0.50


def _book_tickers_in_text(text) -> set:
    """Tickers from the canonical book universe mentioned in ``text``.

    Case-sensitive (financial copy writes tickers uppercase) and
    word-bounded (``MU`` won't fire inside ``Museum``). Non-string input
    returns an empty set rather than raising.
    """
    if not isinstance(text, str) or not text:
        return set()
    return set(_BOOK_RE.findall(text))


def _parse_first_seen(ts) -> datetime | None:
    if not ts:
        return None
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(ts).strip().replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _hours_between(later: datetime, earlier: datetime) -> float:
    delta = later - earlier
    return round(delta.total_seconds() / 3600.0, 2)


def _empty_envelope(now: datetime,
                    card_cap: int,
                    *,
                    state: str,
                    headline: str,
                    briefing_ts: str | None = None,
                    briefing_age_hours: float | None = None,
                    window_start: str | None = None,
                    window_end: str | None = None,
                    window_hours: float | None = None,
                    n_urgent_articles: int = 0) -> dict:
    """Skeleton envelope — same key set as a populated response so the
    UI/chat binding never sees a missing field."""
    return {
        "as_of": now.isoformat(timespec="seconds"),
        "state": state,
        "headline": headline,
        "briefing_ts": briefing_ts,
        "briefing_age_hours": briefing_age_hours,
        "window_start": window_start,
        "window_end": window_end,
        "window_hours": window_hours,
        "n_urgent_articles": n_urgent_articles,
        "n_unique_tickers": 0,
        "n_covered": 0,
        "n_missed": 0,
        "coverage_ratio": None,
        "covered": [],
        "missed": [],
        "card_cap": card_cap,
    }


def build_briefing_coverage_audit(briefing,
                                  articles,
                                  *,
                                  window_start: datetime | None = None,
                                  window_end: datetime | None = None,
                                  card_cap: int = _DEFAULT_CARD_CAP,
                                  now: datetime | None = None) -> dict:
    """Audit briefing text vs the urgent articles that fired in its window.

    ``briefing``: dict-shaped row (``{ts, text, article_count}``) — the
    latest published briefing. ``None`` / non-dict / missing ``text`` →
    ``NO_BRIEFING``.

    ``articles``: iterable of dicts with at least ``title`` (and optionally
    ``summary``, ``urgency``, ``source``, ``first_seen``). Each article
    contributes its book-ticker mentions; rows with no book-ticker mention
    are silently dropped (they're outside the universe this audit defends).

    ``window_start`` / ``window_end``: advisory — used only for the
    envelope's ``window_*`` fields and the headline. Filtering is the
    caller's job (the route layer already SQL-windowed the rows).

    Pure, never raises. Surplus keys ignored.
    """
    now = now or datetime.now(timezone.utc)

    # --- NO_BRIEFING -----------------------------------------------------
    if not isinstance(briefing, dict):
        return _empty_envelope(
            now, card_cap,
            state="NO_BRIEFING",
            headline=("No briefing on record — nothing to audit yet."),
        )
    text = briefing.get("text") if isinstance(briefing.get("text"), str) else ""
    briefing_ts_raw = briefing.get("ts")
    briefing_ts_dt = _parse_first_seen(briefing_ts_raw)
    if not text.strip() or briefing_ts_dt is None:
        return _empty_envelope(
            now, card_cap,
            state="NO_BRIEFING",
            headline=("Latest briefing row is missing text/ts — "
                      "nothing to audit."),
            briefing_ts=str(briefing_ts_raw) if briefing_ts_raw else None,
        )

    briefing_age = _hours_between(now, briefing_ts_dt)

    window_start_iso = (window_start.isoformat(timespec="seconds")
                        if isinstance(window_start, datetime) else None)
    window_end_iso = (window_end.isoformat(timespec="seconds")
                      if isinstance(window_end, datetime) else None)
    window_hours = None
    if isinstance(window_start, datetime) and isinstance(window_end, datetime):
        window_hours = _hours_between(window_end, window_start)

    # --- Tally urgent ticker flow ---------------------------------------
    # ``per_ticker`` = ticker -> {n_articles, max_urgency, sample_title}.
    # The sample title is the highest-urgency one (tie-break: first seen),
    # so the missed-card narrative reads like a real headline rather than
    # a generic "MU mentioned in something".
    per_ticker: dict[str, dict] = {}
    n_urgent = 0
    if isinstance(articles, (list, tuple)):
        for art in articles:
            if not isinstance(art, dict):
                continue
            n_urgent += 1
            title = art.get("title") if isinstance(art.get("title"), str) else ""
            summary = (art.get("summary")
                       if isinstance(art.get("summary"), str) else "")
            blob = f"{title} {summary}".strip()
            if not blob:
                continue
            hits = _book_tickers_in_text(blob)
            if not hits:
                continue
            try:
                urg = int(art.get("urgency") or 0)
            except (TypeError, ValueError):
                urg = 0
            for tk in hits:
                slot = per_ticker.setdefault(tk, {
                    "ticker": tk,
                    "n_articles": 0,
                    "max_urgency": 0,
                    "sample_title": "",
                })
                slot["n_articles"] += 1
                # Keep the FIRST title that ties the running max-urgency, so
                # ranking is deterministic across re-runs of the same input.
                if urg > slot["max_urgency"] or not slot["sample_title"]:
                    slot["max_urgency"] = max(slot["max_urgency"], urg)
                    if title:
                        slot["sample_title"] = title

    # --- NO_URGENT --------------------------------------------------------
    if not per_ticker:
        return _empty_envelope(
            now, card_cap,
            state="NO_URGENT",
            headline=("No urgent book-ticker flow in the window — "
                      "briefing has nothing to cover."),
            briefing_ts=briefing_ts_dt.isoformat(timespec="seconds"),
            briefing_age_hours=briefing_age,
            window_start=window_start_iso,
            window_end=window_end_iso,
            window_hours=window_hours,
            n_urgent_articles=n_urgent,
        )

    # --- Classify covered vs missed --------------------------------------
    briefing_hits = _book_tickers_in_text(text)
    # Canonical rank for stable tie-breaks (same convention as
    # ``_book_heat_lines`` / ``_book_tickers``).
    rank = {t: i for i, t in enumerate(_BOOK_TICKERS)}

    covered_rows: list[dict] = []
    missed_rows: list[dict] = []
    for tk, slot in per_ticker.items():
        target = covered_rows if tk in briefing_hits else missed_rows
        target.append(slot)

    # Highest urgency × most articles first; ties broken by canonical rank.
    def _rank_key(row: dict) -> tuple:
        return (-row["max_urgency"], -row["n_articles"],
                rank.get(row["ticker"], len(rank)))

    covered_rows.sort(key=_rank_key)
    missed_rows.sort(key=_rank_key)

    n_unique = len(per_ticker)
    n_covered = len(covered_rows)
    n_missed = len(missed_rows)
    coverage_ratio = round(n_covered / n_unique, 3)

    if coverage_ratio >= _COMPLETE_FLOOR:
        state = "COMPLETE"
    elif coverage_ratio >= _PARTIAL_FLOOR:
        state = "PARTIAL"
    else:
        state = "THIN"

    # Headline frames the *miss list* first — the actionable side. A
    # COMPLETE state still gets a headline so the chat enrichment layer can
    # render a one-line "all clear" rather than fall back to a 0-key check.
    pct = int(round(coverage_ratio * 100))
    if state == "COMPLETE":
        headline = (
            f"Briefing covered {n_covered}/{n_unique} book tickers "
            f"({pct}%) with urgent flow — complete."
        )
    else:
        top_miss = missed_rows[0]["ticker"] if missed_rows else "—"
        headline = (
            f"Briefing covered {n_covered}/{n_unique} book tickers "
            f"({pct}%) — {state.lower()}. Top miss: {top_miss}."
        )

    if card_cap > 0:
        covered_rows = covered_rows[:card_cap]
        missed_rows = missed_rows[:card_cap]
    else:
        covered_rows = []
        missed_rows = []

    return {
        "as_of": now.isoformat(timespec="seconds"),
        "state": state,
        "headline": headline,
        "briefing_ts": briefing_ts_dt.isoformat(timespec="seconds"),
        "briefing_age_hours": briefing_age,
        "window_start": window_start_iso,
        "window_end": window_end_iso,
        "window_hours": window_hours,
        "n_urgent_articles": n_urgent,
        "n_unique_tickers": n_unique,
        "n_covered": n_covered,
        "n_missed": n_missed,
        "coverage_ratio": coverage_ratio,
        "covered": covered_rows,
        "missed": missed_rows,
        "card_cap": card_cap,
    }


if __name__ == "__main__":  # smoke against the live DB
    import json
    import sqlite3
    import sys
    from datetime import timedelta
    from pathlib import Path

    BASE = Path(__file__).resolve().parents[1]
    if str(BASE) not in sys.path:
        sys.path.insert(0, str(BASE))

    from storage.article_store import _LIVE_ONLY_CLAUSE, _get_db_path  # type: ignore

    db = _get_db_path()
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=15)
    cur = conn.execute(
        "SELECT ts, text, article_count FROM briefings "
        "ORDER BY ts DESC LIMIT 2"
    ).fetchall()
    if not cur:
        print(json.dumps(build_briefing_coverage_audit(None, []), indent=2))
        sys.exit(0)
    latest = {"ts": cur[0][0], "text": cur[0][1], "article_count": cur[0][2]}
    prior_ts = cur[1][0] if len(cur) > 1 else None

    latest_dt = datetime.fromisoformat(latest["ts"].replace("Z", "+00:00"))
    if prior_ts:
        start_dt = datetime.fromisoformat(prior_ts.replace("Z", "+00:00"))
    else:
        start_dt = latest_dt - timedelta(hours=5)

    rows = conn.execute(
        f"""SELECT title, urgency, source, first_seen
              FROM articles
             WHERE {_LIVE_ONLY_CLAUSE}
               AND urgency >= 1
               AND first_seen >= ?
               AND first_seen <= ?
             ORDER BY first_seen DESC
             LIMIT 5000""",
        (start_dt.isoformat(timespec="seconds"),
         latest_dt.isoformat(timespec="seconds")),
    ).fetchall()
    conn.close()
    arts = [{"title": r[0], "urgency": r[1],
             "source": r[2], "first_seen": r[3]} for r in rows]
    rep = build_briefing_coverage_audit(
        latest, arts,
        window_start=start_dt, window_end=latest_dt,
    )
    print(json.dumps(rep, indent=2, default=str))
