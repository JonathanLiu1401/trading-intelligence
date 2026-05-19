"""Per-held-ticker news-source-diversity verdict — is the catalyst REAL or
a SYNDICATED ECHO?

``analytics/news_velocity.py`` answers *rate*: is the article flow on this
held name building or fading versus a 6-day baseline. A z-score of +4.0
("SURGING") looks like a genuine catalyst — but the same z-score is what
you get when one outlet's wire is mirrored across N RSS feeds in the same
24h window. ``news_velocity`` cannot tell the two cases apart; it has only
one observable per article (the timestamp).

``build_news_source_mix`` adds the orthogonal *breadth* observable: how
many DISTINCT collector sources are talking about this name, and is any
single source dominating the article count. Combined with the velocity
read, the operator gets the right combined verdict:

  * **SURGING + STRONG breadth** → real catalyst worth re-evaluating.
  * **SURGING + ECHO breadth**   → syndication artifact; do NOT chase.
  * **STABLE + STRONG**          → steady-state coverage, no new news.
  * **FADING + QUIET**           → information vacuum, thesis is stale.

Pure (no DB, no network, never raises on garbage input); the endpoint
owns the I/O (the documented ``news_velocity``/``signal_followthrough``
builder/endpoint split). Observational only — never gates Opus, never
injected into the decision prompt, no caps (AGENTS.md #2 / #12 — the
``news_velocity`` / ``tail_risk`` / ``stress_scenarios`` precedent).
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

# Per-ticker classification thresholds. Pinned in tests.
MIN_FOR_VERDICT = 2          # ≤1 article in the window → QUIET (too sparse)
STRONG_MIN_SOURCES = 4       # ≥4 distinct collectors AND not echo → STRONG
STRONG_MIN_ARTICLES = 3      # AND ≥3 articles total
ECHO_MIN_ARTICLES = 3        # below this, single-source isn't an "echo"
ECHO_THRESHOLD_PCT = 70.0    # ≥70% of articles from one source → ECHO

# Top-N source breakdown carried in the per-ticker row (keeps payload small).
_BREAKDOWN_TOP_N = 5


def _parse_ts(ts):
    """Tolerate aware/naive ISO strings; mirrors news_velocity._parse_ts."""
    if not ts:
        return None
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _ticker_regex(ticker: str) -> re.Pattern:
    """``$TKR`` cashtag OR word-boundary ``TKR``. Same SSOT regex shape as
    ``news_velocity._ticker_regex`` / ``trade_attribution`` /
    ``signal_followthrough`` so a future tightening lands in one place."""
    return re.compile(rf"(?:\$|\b){re.escape(ticker.upper())}\b")


def _classify(n_articles: int, n_sources: int,
              top_share_pct: float) -> str:
    """State ladder. Pinned by tests; do not adjust thresholds without
    updating the tests AND the live source-aware verdict reasoning in
    the chat enrichment helper."""
    if n_articles < MIN_FOR_VERDICT:
        return "QUIET"
    if n_articles >= ECHO_MIN_ARTICLES and top_share_pct >= ECHO_THRESHOLD_PCT:
        return "ECHO"
    if (n_articles >= STRONG_MIN_ARTICLES
            and n_sources >= STRONG_MIN_SOURCES
            and top_share_pct < ECHO_THRESHOLD_PCT):
        return "STRONG"
    return "MODERATE"


def build_news_source_mix(
    articles: list[dict],
    held_tickers: list[str],
    *,
    now: datetime | None = None,
    window_hours: float = 24.0,
) -> dict:
    """Per-held-ticker source-diversity verdict.

    Inputs
    ------
    articles : list[dict]
        Pre-fetched, live-only article rows spanning at least the last
        ``window_hours``. Each dict needs ``title`` (str), ``source`` (str),
        ``first_seen`` (ISO str). ``body`` is optional and scanned too if
        present (the body of a wire repost still mentions the ticker).
    held_tickers : list[str]
        Tickers to bucket on. Case-insensitive dedup.
    now : datetime, optional
        Injectable for tests. Defaults to ``datetime.now(timezone.utc)``.
    window_hours : float
        Look-back window. Strict-inclusive on the cutoff (the standard
        ``signals.py`` precedent — ``first_seen >= now - window_hours``).

    Output
    ------
    JSON-ready dict. ``per_ticker`` is sorted ECHO-first then STRONG/
    MODERATE/QUIET by descending ``n_articles``, so the most-actionable
    false-signal warning surfaces first.
    """
    now = (now or datetime.now(timezone.utc))
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    base = {
        "as_of": now.isoformat(timespec="seconds"),
        "window_hours": float(window_hours),
        "echo_threshold_pct": ECHO_THRESHOLD_PCT,
        "strong_min_sources": STRONG_MIN_SOURCES,
        "strong_min_articles": STRONG_MIN_ARTICLES,
        "n_held": 0,
        "n_with_data": 0,
        "per_ticker": [],
        "any_echo": False,
        "state": "NO_DATA",
        "headline": "Source mix: no held positions.",
    }

    if not held_tickers or window_hours <= 0:
        return base

    # Dedup case-insensitively, preserve insert order.
    seen: set[str] = set()
    tickers: list[str] = []
    for t in held_tickers:
        if not t:
            continue
        u = str(t).upper().strip()
        if not u or u in seen:
            continue
        seen.add(u)
        tickers.append(u)
    base["n_held"] = len(tickers)
    if not tickers:
        return base

    patterns = {t: _ticker_regex(t) for t in tickers}
    window_cutoff = now.timestamp() - window_hours * 3600

    # Per-ticker accumulators: source-name → count.
    per_ticker_counts: dict[str, dict[str, int]] = {t: {} for t in tickers}

    for a in (articles or []):
        if not isinstance(a, dict):
            continue
        ts = _parse_ts(a.get("first_seen"))
        if ts is None:
            continue
        if ts.timestamp() < window_cutoff:
            continue
        src = str(a.get("source") or "").strip().lower() or "(unknown)"
        title = str(a.get("title") or "")
        body = title + " " + str(a.get("body") or "")
        body_up = body.upper()
        for t, pat in patterns.items():
            if pat.search(body_up):
                per_ticker_counts[t][src] = per_ticker_counts[t].get(src, 0) + 1

    rows: list[dict] = []
    n_with = 0
    any_echo = False
    for t in tickers:
        counts = per_ticker_counts[t]
        n_articles = sum(counts.values())
        n_sources = len(counts)
        breakdown_sorted = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        top_source = breakdown_sorted[0][0] if breakdown_sorted else None
        top_count = breakdown_sorted[0][1] if breakdown_sorted else 0
        top_share_pct = round(100.0 * top_count / n_articles, 2) if n_articles else 0.0
        state = _classify(n_articles, n_sources, top_share_pct)
        if n_articles > 0:
            n_with += 1
        if state == "ECHO":
            any_echo = True
        rows.append({
            "ticker": t,
            "state": state,
            "n_articles": n_articles,
            "n_unique_sources": n_sources,
            "top_source": top_source,
            "top_source_share_pct": top_share_pct,
            "sources_breakdown": [
                {"source": s, "n": n} for s, n in breakdown_sorted[:_BREAKDOWN_TOP_N]
            ],
        })

    # Sort: ECHO first (highest n_articles within), then STRONG, MODERATE,
    # QUIET. Within each state, ties break by n_articles DESC then ticker.
    state_rank = {"ECHO": 0, "STRONG": 1, "MODERATE": 2, "QUIET": 3}
    rows.sort(key=lambda r: (
        state_rank.get(r["state"], 4),
        -r["n_articles"],
        r["ticker"],
    ))

    base["per_ticker"] = rows
    base["n_with_data"] = n_with
    base["any_echo"] = any_echo

    if n_with == 0:
        base["state"] = "NO_DATA"
        base["headline"] = (
            f"Source mix: 0 articles matched any of {len(tickers)} held "
            f"name(s) in last {int(window_hours)}h."
        )
        return base

    base["state"] = "OK"
    # Headline picks the loudest signal: an ECHO warning beats a STRONG
    # confirmation beats a QUIET vacuum (most-actionable first).
    echo_rows = [r for r in rows if r["state"] == "ECHO"]
    if echo_rows:
        top = echo_rows[0]
        base["headline"] = (
            f"Source mix: {top['ticker']} ECHO "
            f"({top['n_articles']} articles, "
            f"{top['top_source_share_pct']:.0f}% from "
            f"{top['top_source']}) — surge may be syndication, not breadth."
        )
        return base
    strong_rows = [r for r in rows if r["state"] == "STRONG"]
    if strong_rows:
        top = strong_rows[0]
        base["headline"] = (
            f"Source mix: {top['ticker']} STRONG "
            f"({top['n_articles']} articles across "
            f"{top['n_unique_sources']} sources)."
        )
        return base
    quiet_rows = [r for r in rows if r["state"] == "QUIET" and r["n_articles"] == 0]
    if quiet_rows and len(quiet_rows) == len(rows):
        base["headline"] = (
            f"Source mix: all {len(rows)} held name(s) QUIET "
            f"(no live articles in last {int(window_hours)}h)."
        )
        return base
    # MODERATE-dominated: report the most-covered name.
    top = max(rows, key=lambda r: r["n_articles"])
    base["headline"] = (
        f"Source mix: {top['ticker']} {top['state']} "
        f"({top['n_articles']} articles, {top['n_unique_sources']} sources)."
    )
    return base
