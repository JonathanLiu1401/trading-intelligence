"""Signal-feed health — is the live trader actually *seeing* any news?

The mature analytics layer measures everything *about* the trader's behaviour
once a decision is made — ``decision_health`` (decision rate),
``decision_forensics`` (why a parse failed), ``decision_drought`` /
``capital_paralysis`` (the alpha cost of sitting pinned),
``signal_followthrough`` (did it act on the signals it saw),
``news_edge`` / ``source_edge`` (do the signals predict the move). Every one of
them assumes the trader *received* signals. None answer the prior question an
operator actually has when the book just HOLDs for hours:

    *Is the news feed even reaching the trader, or is it flying blind?*

``strategy.decide()`` builds Opus's prompt from
``signals.get_top_signals(hours=2, min_score=4.0)`` against the article DB that
``signals._db_path()`` resolves to. If that DB is stale the prompt's
``TOP SCORED SIGNALS`` block is empty, ``store.record_decision`` writes
``signal_count = 0``, and the trader HOLDs forever — silently, because a
0-signal HOLD looks identical to a deliberate one in every existing panel.
``/api/data-feed`` reports raw ``articles_1h`` / ``articles_24h`` counts but no
verdict, no path resolution, and no link to the decision log — a stale-feed
reading of ``articles_24h: 3801`` looks healthy.

This module's marginal contribution over ``/api/data-feed`` is exactly the
three dimensions that make the failure *visible and actionable*: the
**consecutive 0-signal decision streak** (the trader is provably blind, not
merely between news), the **resolved DB path + its newest-live-article age**
(where the trader is actually reading from, and how stale it is), and
**split-brain detection** — ``signals._db_path()`` prefers the USB mount
first while digital-intern's daemon and ``unified_dashboard._articles_db_path``
prefer the local copy first, so when the USB mirror goes stale the two halves
of the system resolve ``articles.db`` with *opposite precedence* and the trader
silently consumes a frozen mirror while every other surface reads the fresh
one.

Pure / deterministic: the builder takes the decision list and a ``feed`` dict
of already-resolved DB stats (the endpoint does all SQLite / filesystem IO,
mirroring the ``thesis_drift`` / ``self_review`` "network lives in the
endpoint, builder takes the dicts" shape). Advisory only — it never gates the
trader and adds no caps (AGENTS.md invariants #2 / #12); ``restart_recommended``
is an operator hint, not a control signal.
"""
from __future__ import annotations

from datetime import datetime, timezone

# A 0-signal HOLD can happen for a single legitimately-quiet cycle; a *run* of
# them means the trader is provably blind, not merely between headlines. Three
# consecutive cycles at OPEN_INTERVAL_S spans ~25–90 min of zero news — well
# past any normal lull given digital-intern's 24/7 collectors (~3.8k live
# articles/day). Tests read this constant so a retune can't false-fail them.
BLIND_STREAK_MIN = 3

# strategy.decide() feeds Opus get_top_signals(hours=2). A newest live article
# older than this is comfortably past that window — the feed is stale, not
# transiently between stories.
STALE_HOURS = 6.0

# How much fresher another candidate DB must be (vs the one the trader actually
# resolves) before we call it a split-brain rather than a system-wide quiet
# news period. 6h ≫ the 2h decision window and ≫ any real collector gap.
SPLIT_BRAIN_GAP_H = 6.0


def _parse_ts(ts: str | None) -> datetime | None:
    """ISO-8601 → tz-aware UTC; None on anything unparseable."""
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def _age_h(iso: str | None, now: datetime) -> float | None:
    """Hours between ``now`` and the ISO timestamp; None if unparseable."""
    dt = _parse_ts(iso)
    if dt is None:
        return None
    return round((now - dt).total_seconds() / 3600.0, 2)


def build_feed_health(decisions: list[dict], feed: dict,
                       now: datetime | None = None) -> dict:
    """Is the live trader receiving any news signals at all?

    ``decisions`` is newest-first (as ``store.recent_decisions`` returns);
    each row's ``signal_count`` is what ``strategy.decide()`` recorded for that
    cycle. ``feed`` is resolved by the endpoint (all IO there, builder stays
    pure):

      - ``resolved_path``     str|None — what ``signals._db_path()`` returns,
                                         i.e. the DB the trader actually reads
      - ``resolved_newest``   str|None — newest *live* article first_seen in it
      - ``resolved_live_2h``  int      — live-only rows in the last 2h (the
                                         exact ``get_top_signals`` window)
      - ``resolved_live_24h`` int
      - ``candidates``        list of ``{path, exists, newest}`` for every
                                         candidate DB so split-brain (a fresher
                                         alternative) can be detected

    ``now`` is injectable for deterministic tests. Never raises.
    """
    now = now or datetime.now(timezone.utc)
    feed = feed or {}
    decisions = decisions or []
    n = len(decisions)

    resolved_path = feed.get("resolved_path")
    resolved_newest = feed.get("resolved_newest")
    resolved_age = _age_h(resolved_newest, now)
    live_2h = int(feed.get("resolved_live_2h") or 0)
    live_24h = int(feed.get("resolved_live_24h") or 0)

    # Consecutive most-recent decisions whose signal_count is exactly 0. A
    # missing/None signal_count (partial dict, never a real row — schema is
    # NOT NULL) breaks the streak conservatively rather than over-reporting
    # blindness.
    blind_streak = 0
    for d in decisions:
        sc = d.get("signal_count")
        if sc is None:
            break
        try:
            if int(sc) != 0:
                break
        except (TypeError, ValueError):
            break
        blind_streak += 1

    # Echo every candidate with its computed age; find the freshest *other*
    # candidate (different real path, exists, parseable newest).
    cand_out: list[dict] = []
    fresher_path: str | None = None
    fresher_age: float | None = None
    for c in feed.get("candidates") or []:
        cpath = c.get("path")
        cage = _age_h(c.get("newest"), now) if c.get("exists") else None
        cand_out.append({
            "path": cpath,
            "exists": bool(c.get("exists")),
            "newest": c.get("newest"),
            "age_h": cage,
        })
        if cpath == resolved_path or cage is None:
            continue
        if fresher_age is None or cage < fresher_age:
            fresher_age, fresher_path = cage, cpath

    resolved_stale = resolved_age is None or resolved_age >= STALE_HOURS

    # Split-brain has two shapes, both meaning "a trader process is reading a
    # materially staler feed than a fresher copy that exists":
    #
    #  (a) legacy-resolution split — the canonical one since signals._db_path()
    #      became freshness-aware. The fixed resolver now picks the FRESH copy,
    #      but a runner/dashboard process that booted before that commit
    #      (/api/build-info `stale`) still runs the old existence-first
    #      resolver and reads `legacy_path`. The endpoint supplies
    #      signals._legacy_choice() as `legacy_path`; if it differs from the
    #      freshly-resolved DB and is ≥GAP staler, that stale process is blind
    #      and a RESTART (not a mount fix) is what reconverges it.
    #
    #  (b) resolved-stale split — the pre-fix shape: the resolved DB itself is
    #      stale while another candidate is materially fresher. Still asserted
    #      by the pure builder fixtures (which pass no `legacy_path`), so this
    #      term stays exactly as it was and is inert unless it fires on its own.
    legacy_path = feed.get("legacy_path")
    legacy_age = _age_h(feed.get("legacy_newest"), now)
    stale_process_split = bool(
        legacy_path
        and resolved_path
        and legacy_path != resolved_path
        and legacy_age is not None
        and resolved_age is not None
        and legacy_age - resolved_age >= SPLIT_BRAIN_GAP_H
    )
    resolved_stale_split = bool(
        resolved_path
        and resolved_stale
        and fresher_age is not None
        and (resolved_age is None or resolved_age - fresher_age >= SPLIT_BRAIN_GAP_H)
    )
    split_brain = stale_process_split or resolved_stale_split

    # Verdict precedence (locked by a dedicated test class). BLIND outranks
    # STALE_FEED because a proven streak of 0-signal *decisions* is the
    # actionable harm; a stale feed that hasn't yet manifested in the decision
    # log is the milder warning. <BLIND_STREAK_MIN decisions can never reach
    # BLIND — that is the built-in sample-size guard.
    if resolved_path is None or (n == 0 and resolved_newest is None):
        verdict = "NO_DATA"
    elif blind_streak >= BLIND_STREAK_MIN:
        verdict = "BLIND"
    elif resolved_stale:
        verdict = "STALE_FEED"
    else:
        verdict = "HEALTHY"

    restart_recommended = bool(split_brain)

    age_txt = (f"{resolved_age:.1f}h old" if resolved_age is not None
               else "no live article ever")
    split_clause = ""
    if resolved_stale_split:
        split_clause = (
            f" — split-brain: {fresher_path} is only {fresher_age:.1f}h old, "
            f"but the trader resolves {resolved_path} (signals._db_path() "
            f"prefers the USB mount; the daemon writes the local copy). "
            f"Restart the paper-trader runner / fix the mount so its signal "
            f"source reconverges.")
    elif stale_process_split:
        split_clause = (
            f" — split-brain: signals._db_path() now resolves the FRESH "
            f"{resolved_path} ({resolved_age:.1f}h old), but a trader process "
            f"that booted before the freshness-aware resolver still runs the "
            f"old existence-first code and reads the stale {legacy_path} "
            f"({legacy_age:.1f}h old). The on-disk fix does not rescue a "
            f"running process — RESTART the paper-trader so it reads the "
            f"fresh feed (check /api/build-info `stale`).")

    if verdict == "NO_DATA":
        headline = (
            "NO_DATA — no resolved article DB"
            + ("" if resolved_path else " (signals._db_path() found none)")
            + ("; no decisions recorded yet." if n == 0 else "."))
    elif verdict == "BLIND":
        headline = (
            f"BLIND — {blind_streak} consecutive decision(s) with 0 signals; "
            f"the trader is flying blind. Newest live article in "
            f"{resolved_path} is {age_txt}; {live_2h} live article(s) in the "
            f"last 2h (the get_top_signals window strategy.decide() feeds "
            f"Opus){split_clause}")
    elif verdict == "STALE_FEED":
        headline = (
            f"STALE_FEED — newest live article in {resolved_path} is "
            f"{age_txt} (>{STALE_HOURS:.0f}h); the trader's 2h signal window "
            f"holds {live_2h} article(s). {blind_streak} most-recent "
            f"decision(s) already saw 0 signals{split_clause}")
    else:
        headline = (
            f"HEALTHY — newest live article {age_txt}; {live_2h} live "
            f"article(s) in the last 2h, {live_24h} in 24h; the most-recent "
            f"decision received signals.")

    return {
        "as_of": now.isoformat(timespec="seconds"),
        "verdict": verdict,
        "headline": headline,
        "blind_streak": blind_streak,
        "n_decisions": n,
        "resolved_path": resolved_path,
        "resolved_newest": resolved_newest,
        "resolved_newest_age_h": resolved_age,
        "resolved_live_2h": live_2h,
        "resolved_live_24h": live_24h,
        "split_brain": split_brain,
        "legacy_path": legacy_path,
        "legacy_newest_age_h": legacy_age,
        "fresher_path": fresher_path,
        "fresher_age_h": fresher_age,
        "candidates": cand_out,
        "restart_recommended": restart_recommended,
        # Echoed so the UI / chat / tests read thresholds from one place and a
        # retune can't silently desync a hardcoded copy.
        "blind_streak_min": BLIND_STREAK_MIN,
        "stale_hours": STALE_HOURS,
        "split_brain_gap_h": SPLIT_BRAIN_GAP_H,
    }
