"""Bloomberg Terminal-style briefing — Claude Opus 4.7 via CLI."""
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse

from core.claude_cli import claude_call
# Reuse the *single* well-tested headline-canonicalisation primitive
# (tests/test_alert_dedup.py) instead of re-deriving a signature here — the
# documented anti-drift discipline (same reason watchers.alert_recency imports
# it, and alert_agent reuses ml.features._source_credibility). alert_dedup is
# pure stdlib+re (no DB / ml / aiohttp import graph), so this adds no cycle.
from watchers.alert_dedup import _signature
# Cross-cycle alert-recency store (pure stdlib+sqlite+re — NO ml/numpy/aiohttp
# graph, same import-safety profile as alert_dedup above). Used read-only to
# tag digest rows that already fired a standalone 🚨 BREAKING alert this
# window, so the briefing LEAD/TOP-SIGNALS don't re-surface a story the
# analyst was already pushed (their top "duplicate alerts" complaint).
from watchers import alert_recency
# Order-independent near-duplicate collapse. ml.dedup is pure stdlib+re —
# despite its package path it imports NO numpy/torch (ml/__init__.py is empty;
# verified import-clean), so this carries the SAME import-safety profile as
# alert_dedup/alert_recency above and adds no cycle. It is the purpose-built
# complement to _collapse_syndicated below (its own docstring names the
# "briefing pre-filter" as the intended integration); see _build_payload.
from ml.dedup import dedupe_articles as _dedupe_near_duplicates

MODEL = "claude-opus-4-7"


def _recent_alert_signatures() -> set:
    """Best-effort set of canonical headline signatures that fired a
    standalone 🚨 BREAKING alert within ``alert_recency.ALERT_RECENCY_TTL_HOURS``.

    Returns ``set()`` on ANY failure (missing/locked alert_recency.db, import
    error) — an alert↔briefing parity read must NEVER break or delay the 5h
    briefing it annotates (identical discipline to ``_collect_source_health``;
    ``alert_recency.recent_signatures`` is itself already best-effort and
    never raises, this wrapper is belt-and-braces + the documented shape)."""
    try:
        return alert_recency.recent_signatures()
    except Exception:
        return set()


def _collapse_syndicated(articles: list) -> list:
    """Collapse syndicated copies of one story in the briefing newswire.

    A breaking wire item is carried within minutes by GDELT, Reuters, Yahoo,
    RSS and half a dozen scrapers. Each lands as its own row and each can score
    high, so the top-50 digest Opus sees is dominated by 5-8 near-identical
    headlines — the consuming analyst's single biggest noise complaint, applied
    to the one path that never deduped it. ``watchers.alert_dedup`` collapses
    syndication on the *alert* path and ``article_store`` caps per-publisher in
    the briefing, but neither collapses the SAME wire headline arriving under
    DIFFERENT domain keys (``GDELT/reuters.com`` + ``scraped/finance.yahoo.com``
    + ``rss`` are three domains, all survive the per-domain cap).

    Pure, order-preserving, side-effect-free:

      * groups by ``alert_dedup._signature`` (the shared canonicalisation —
        wire markers / source attribution stripped, first-8-token key);
      * an empty signature (untitled / snapshot rows whose title is all
        stop-stripped) is NEVER merged — unique key per copy, identical policy
        to ``dedupe_urgent``, so the prepended PORTFOLIO/OPTIONS snapshot rows
        and titleless items always pass through untouched and keep their
        leading position;
      * the highest-score copy represents the cluster (score = ai_score, else
        _relevance_score; ties keep the earlier/ higher-ranked one — stable);
      * survivors keep their input order (the caller already score-ranked);
      * each survivor gains ``_corroboration`` = total copies it represents.
        N>1 is itself a real analyst signal (independent corroboration ⇒ the
        event is bigger), surfaced verbatim to Opus.

    Returns NEW dicts (shallow copies) so the caller's ``source_articles``
    list — which heartbeat_worker feeds to the briefing-label / training path —
    is never mutated. This keeps the load-bearing invariants (backtest
    isolation, ml_score≠ai_score, score_source, the urgency state machine)
    untouched here *by construction*: this function only ever reshapes the
    text Opus reads, never the DB or the label list.
    """
    def _score(a: dict) -> float:
        # Mirror the display logic exactly: ``ai_score or _relevance_score``.
        # A falsy ai_score (0 / 0.0 — neither LLM nor model has scored yet)
        # falls through to the kw _relevance_score, so the cluster
        # representative is chosen on the SAME number the row will render
        # with (no rank/display mismatch).
        for key in ("ai_score", "_relevance_score"):
            v = a.get(key)
            if isinstance(v, (int, float)) and v:
                return float(v)
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if fv:
                return fv
        return 0.0

    keep: dict[str, dict] = {}
    order: list[str] = []
    # Per-cluster set of distinct source keys (e.g. ``gdelt_gkg/iheart.com``
    # vs ``rss`` vs ``gdelt_gkg/joker.com``). Used to discriminate ``[echo]``
    # — a cluster whose ``_corroboration > 1`` but ``_distinct_sources == 1``
    # is the SAME source repeating itself (e.g. a mass-aggregator host
    # restating one wire under slightly-varied titles), NOT independent
    # cross-outlet corroboration. The ``[syndicated xN]`` tag oversells that
    # case; ``[echo]`` qualifies it so Opus down-weights it in TOP SIGNALS.
    # An empty / missing source is treated as the literal empty-string key so
    # two empty-source copies count as one distinct (still echo-eligible) —
    # the analyst-safe direction for noisy collectors that lose source tags.
    hosts: dict[str, set[str]] = {}
    for idx, art in enumerate(articles):
        sig = _signature(art.get("title"))
        if not sig:
            sig = f"__uniq__{idx}"
        src = str(art.get("source") or "")
        cur = keep.get(sig)
        if cur is None:
            rep = dict(art)
            rep["_corroboration"] = 1
            keep[sig] = rep
            order.append(sig)
            hosts[sig] = {src}
            continue
        cur["_corroboration"] += 1
        hosts[sig].add(src)
        # Strictly-greater so ties keep the earlier (already higher-ranked)
        # representative — deterministic and stable.
        if _score(art) > _score(cur):
            merged = dict(art)
            merged["_corroboration"] = cur["_corroboration"]
            keep[sig] = merged
    # Surface distinct-source count on the representative. The render layer
    # uses it for ``[echo]``; downstream consumers that already keyed on
    # ``_corroboration`` are unaffected (additive field).
    for sig in order:
        keep[sig]["_distinct_sources"] = len(hosts[sig]) or 1
    return [keep[s] for s in order]


# ── Echo detection (single-source self-syndication) ─────────────────────────
# A cluster of N>=ECHO_MIN_COPIES copies all carrying the SAME ``source`` key
# is one outlet repeating itself, not independent cross-outlet corroboration.
# GDELT GKG hosts (iheart.com, joker.com, wkrb13.com) are the recurring
# culprits — a single host can write 5-8 slight headline variants of one wire
# in an hour. The existing ``[syndicated xN]`` tag tells Opus "N wire copies
# of this story exist" — true, but reads as positive corroboration; this
# qualifies it. ``[echo]`` says: those N copies all came from ONE outlet —
# down-weight, not up-weight, when ranking TOP SIGNALS / choosing the LEAD.
#
# Threshold N>=3 (not 2) keeps a benign retitle by the same source quiet; a
# 2-copy single-source cluster might just be a corrected/updated headline.
# 3+ copies from one source is the firehose pattern the analyst persona
# complains about as "echo".
ECHO_MIN_COPIES = 3


def _is_echo_row(art: dict) -> bool:
    """Pure: True iff this collapsed-cluster representative is a single-source
    echo (>=ECHO_MIN_COPIES copies, all from one ``source`` key).

    Defaults assume the more conservative answer: a missing
    ``_distinct_sources`` (e.g. a row that bypassed ``_collapse_syndicated``,
    or a prepended snapshot row) defaults to the corroboration count, so a
    snapshot row that already has ``_corroboration==1`` never lights up.
    """
    if not isinstance(art, dict):
        return False
    try:
        corro = int(art.get("_corroboration", 1))
    except (TypeError, ValueError):
        return False
    if corro < ECHO_MIN_COPIES:
        return False
    distinct = art.get("_distinct_sources", corro)
    try:
        distinct = int(distinct)
    except (TypeError, ValueError):
        return False
    return distinct <= 1

# ── Coverage-gap intelligence ────────────────────────────────────────────────
# A news analyst's most dangerous failure is a *silent* one: a high-value intel
# channel goes dark and the briefing simply contains nothing from it, so the
# absence reads as "no news" rather than "blind here". Live inspection (2026-05)
# showed sec_edgar / sec_edgar_ft with 900+ consecutive empty polls and ZERO
# 8-K filings delivered — the analyst was completely blind to filings with no
# signal anywhere in the briefing. This surfaces that explicitly.
#
# Only curated, analyst-meaningful channels are listed (NOT per-query gdelt
# junk keys or unknown tags) so this stays signal, not noise — the analyst
# persona's top complaint. Mapping: source_health key → (label, priority);
# priority 0 = most market-critical (filings), higher = less.
_COVERAGE_LABELS: dict[str, tuple[str, int]] = {
    "sec_edgar":        ("SEC 8-K filings", 0),
    "sec_edgar_ft":     ("SEC full-text filings", 0),
    "finnhub":          ("Finnhub company news", 1),
    "polygon":          ("Polygon market news", 1),
    "gdelt":            ("GDELT global wire", 1),
    "rss":              ("RSS feed bundle", 1),
    "web":              ("Web-scrape wire", 1),
    "alphavantage":     ("AlphaVantage news-sentiment", 2),
    "newsapi":          ("NewsAPI keyword wire", 2),
    "google_news":      ("Google News round-robin", 2),
    "yahoo_ticker_rss": ("Yahoo per-ticker RSS", 2),
    "reddit":           ("Reddit retail sentiment", 2),
    "nitter":           ("Nitter/X feed", 3),
    "massive":          ("Massive aggregator", 3),
}
# Per-channel poll cadence (seconds), mirroring daemon.py's *_INTERVAL
# constants for the curated coverage set. Used to estimate how long a channel
# has been dark from ``consecutive_failures`` instead of from ``last_seen``.
#
# Why not ``last_seen``: ``source_health.record_result`` rewrites
# ``last_seen = now`` on EVERY poll, including the empty polls of a disabled
# channel (it is "last poll", not "last delivery" — and `get_stale_sources`
# legitimately relies on that for wedged-worker detection). So
# ``now - last_seen`` is ≈0 for ANY actively-polled disabled source: the live
# 5h briefing read "SEC 8-K filings — DARK 0.0h (932 empty polls, 0 delivered
# all session)", telling the analyst a channel blind the *entire* session was
# negligible — defeating the whole purpose of this section.
# ``consecutive_failures × cadence`` is the honest, data-available estimate.
# Keys MUST stay a superset of _COVERAGE_LABELS (a labelled channel without a
# cadence silently degrades to "DARK unknown"); pinned by the parity test.
_COVERAGE_POLL_SECS: dict[str, int] = {
    "sec_edgar": 300, "sec_edgar_ft": 900, "finnhub": 300, "polygon": 600,
    "gdelt": 600, "rss": 30, "web": 60, "alphavantage": 1800,
    "newsapi": 1500, "google_news": 120, "yahoo_ticker_rss": 240,
    "reddit": 45, "nitter": 180, "massive": 600,
}

# Never surface more than this many gap lines — a fully-degraded host should
# not produce a wall of text that itself becomes noise.
_MAX_COVERAGE_LINES = 8


def _collect_source_health() -> dict:
    """Best-effort read of the source-health report. Returns {} on any failure
    (missing source_health.db, import error, locked DB) — a coverage-gap read
    must NEVER break or delay the 5h briefing it annotates."""
    try:
        from collectors import source_health
        return source_health.get_health_report() or {}
    except Exception:
        return {}


def _coverage_gap_lines(report: dict, now: datetime | None = None) -> list[str]:
    """Pure: turn a source-health report into ranked analyst-facing gap lines.

    A channel is a gap when it is ``disabled`` (FAILURE_THRESHOLD consecutive
    empty polls) AND it is one of the curated high-value channels. Lines are
    sorted by criticality (filings first), then by the estimated dark-duration
    (``consecutive_failures × poll cadence``; see _COVERAGE_POLL_SECS for why
    not ``last_seen``). Returns [] when nothing curated is down.

    ``now`` is accepted for signature/back-compat stability (callers and tests
    pass it); it is unused since dark-duration no longer derives from a
    wall-clock delta.
    """
    if not isinstance(report, dict) or not report:
        return []
    rows: list[tuple[int, float, str]] = []
    for key, info in report.items():
        if key not in _COVERAGE_LABELS or not isinstance(info, dict):
            continue
        if not info.get("disabled"):
            continue
        label, priority = _COVERAGE_LABELS[key]
        fails = int(info.get("consecutive_failures") or 0)
        delivered = int(info.get("total_articles") or 0)
        # Estimate dark-duration from consecutive empty polls × the channel's
        # poll cadence — NOT from ``last_seen`` (see _COVERAGE_POLL_SECS: it is
        # last-*poll* time, always ≈now for an actively-polled disabled
        # channel, so it reported a misleading "DARK 0.0h" for a source blind
        # all session). ``~`` prefix flags it as an estimate. None when the
        # cadence is unknown or it has not failed yet → "DARK unknown".
        poll_secs = _COVERAGE_POLL_SECS.get(key)
        dark_h: float | None = None
        if poll_secs and fails > 0:
            dark_h = fails * poll_secs / 3600.0
        dark_str = f"~{dark_h:.1f}h" if dark_h is not None else "unknown"
        extra = ", 0 delivered all session" if delivered == 0 else ""
        line = f"{label} — DARK {dark_str} ({fails} empty polls{extra})"
        # Sort key: priority asc, then longest-dark first (unknown sorts last).
        rows.append((priority, -(dark_h if dark_h is not None else -1.0), line))
    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    return [line for _, _, line in rows[:_MAX_COVERAGE_LINES]]


# ── Throughput-degradation early warning ─────────────────────────────────────
# COVERAGE GAP only surfaces sources the FAILURE_THRESHOLD has already pushed
# to ``disabled`` — a binary, late signal. A live source can be silently
# collapsing in throughput (RSS feed that delivered 40/h yesterday, 3/h now)
# without ever hitting that bar; the analyst's "stale sources" complaint
# applies equally to a still-marginally-alive source as to a fully dark one.
#
# ``ArticleStore.source_throughput`` already computes recent-vs-prior counts
# per ``source`` column key (CLAUDE.md §6 / tests/test_source_throughput.py)
# but until now NO consumer used it. This is the missing read-side:
# pure-functional rendering of its rows into an analyst-facing block,
# surfaced to Opus exactly like COVERAGE GAP — same shape, same "reproduced"
# discipline, same anti-noise capping. Pure: no DB write, no
# ai_score/ml_score/score_source/urgency touch, never mutates source_articles
# — all four load-bearing invariants intact by construction.
#
# Thresholds tuned conservatively (the analyst persona's top complaint is
# noise, so a tiny baseline must never produce a wall-of-text alarm):
#
#   * ``prior >= 10``: a 5→0 drop is 100% deceleration but only a 5-row loss
#     of signal in an hour; ignored. ``prior >= 10`` ensures the source was
#     materially productive in the baseline window so a slowdown is real.
#   * ``decel_pct >= 60``: 40+% drop from prior → recent is "halved or
#     worse", a magnitude the analyst would notice and care about.
#
# Sort order: largest absolute loss first (``prior - recent``), tiebreak on
# higher ``prior`` — a 50→0 source matters more than a 20→0 source even
# though both are 100% deceleration. Capped at ``_MAX_DEGRADATION_LINES`` so
# this section can never itself become noise.
_THROUGHPUT_MIN_PRIOR = 10
_THROUGHPUT_MIN_DECEL_PCT = 60.0
_MAX_DEGRADATION_LINES = 6


def _collect_source_throughput(window_min: int = 60) -> list[dict]:
    """Best-effort read of per-source throughput. ``[]`` on ANY failure
    (missing/locked articles.db, import error) — a degradation-hint read
    must NEVER break or delay the 5h briefing it annotates (identical
    discipline to ``_collect_source_health`` / ``_recent_briefing_digest``).
    Opens a fresh short-lived ``mode=ro`` connection (never the daemon's
    shared ``self.conn`` — the documented cursor-collision hazard)."""
    try:
        import sqlite3
        from storage.article_store import _get_db_path, _LIVE_ONLY_CLAUSE
        from datetime import timedelta as _td
        now = datetime.now(timezone.utc)
        recent_cut = (now - _td(minutes=window_min)).isoformat()
        prior_cut = (now - _td(minutes=2 * window_min)).isoformat()
        conn = sqlite3.connect(
            f"file:{_get_db_path()}?mode=ro", uri=True, timeout=5,
        )
        try:
            conn.execute("PRAGMA busy_timeout=5000")
            rows = conn.execute(
                "SELECT source, "
                "SUM(CASE WHEN first_seen >= ? THEN 1 ELSE 0 END) AS recent, "
                "SUM(CASE WHEN first_seen >= ? AND first_seen < ? "
                "         THEN 1 ELSE 0 END) AS prior "
                f"FROM articles WHERE first_seen >= ? AND {_LIVE_ONLY_CLAUSE} "
                "GROUP BY source",
                (recent_cut, prior_cut, recent_cut, prior_cut),
            ).fetchall()
        finally:
            conn.close()
        out: list[dict] = []
        for source, recent, prior in rows:
            recent = int(recent or 0)
            prior = int(prior or 0)
            if recent == 0 and prior == 0:
                continue
            decel_pct = (
                round((prior - recent) / prior * 100.0, 1)
                if prior > 0 else None
            )
            out.append({"source": source or "", "recent": recent,
                        "prior": prior, "delta": recent - prior,
                        "decel_pct": decel_pct})
        return out
    except Exception:
        return []


def _throughput_degradation_lines(
    throughput: list[dict],
    min_prior: int = _THROUGHPUT_MIN_PRIOR,
    min_decel_pct: float = _THROUGHPUT_MIN_DECEL_PCT,
    max_lines: int = _MAX_DEGRADATION_LINES,
) -> list[str]:
    """Pure: render ``source_throughput`` rows into ranked degradation lines.

    A source qualifies when its baseline (``prior``) was materially
    productive AND it lost most of that flow in the recent window. The
    accepted-baseline / accepted-drop bars are deliberately conservative so
    the section is signal, not noise.

    ``decel_pct=None`` (no baseline) or non-decelerating sources are
    omitted — they are either brand-new (no signal yet) or accelerating
    (the opposite of what this section reports). Sorted by absolute loss
    desc, prior desc. Capped at ``max_lines``.
    """
    if not isinstance(throughput, list) or not throughput:
        return []
    # Source name as deterministic tiebreaker BEFORE the dict — without it,
    # two rows with the same (abs_loss, prior) would force Python to compare
    # the trailing dicts and raise ``TypeError: '<' not supported between
    # instances of 'dict' and 'dict'`` (e.g. multiple `prior=10, recent=0`
    # sources in the throughput snapshot — observed live: the briefing call
    # bubbled this from _throughput_degradation_lines so analyze() returned
    # the "[analyst] No response from Claude." sentinel for the whole 5h
    # cycle, blanking that window's heartbeat on the analyst's primary
    # consumed product).
    candidates: list[tuple[int, int, str, dict]] = []
    for r in throughput:
        if not isinstance(r, dict):
            continue
        prior = int(r.get("prior") or 0)
        recent = int(r.get("recent") or 0)
        decel_pct = r.get("decel_pct")
        if not isinstance(decel_pct, (int, float)):
            continue
        if prior < min_prior:
            continue
        if decel_pct < min_decel_pct:
            continue
        abs_loss = prior - recent
        src_key = str(r.get("source") or "")
        candidates.append((-abs_loss, -prior, src_key, r))
    if not candidates:
        return []
    candidates.sort()
    out: list[str] = []
    for _abs_loss_neg, _prior_neg, _src_key, r in candidates[:max_lines]:
        src = (r.get("source") or "").strip() or "unknown"
        decel = r.get("decel_pct")
        decel_str = f"{decel:.0f}%" if isinstance(decel, (int, float)) else "?"
        out.append(
            f"{src} — {r['recent']} in last 60min "
            f"(vs {r['prior']} prior; -{decel_str})"
        )
    return out


# ── Alert-velocity wire-temperature hint ─────────────────────────────────────
# The 🚨 BREAKING alert path is the analyst's most time-critical product. Its
# RAW FIRING RATE (urgency=2 rows landed per unit time) carries a magnitude
# signal no individual story's score can express: 24 alerts in 5h vs 8 in the
# prior 5h tells Opus the wire is materially HOT — a real macro event is
# under way (Fed surprise, geopolitical escalation, broad selloff), so the
# briefing should weight stories with appropriate cumulative weight; 2 vs 12
# tells Opus the wire is unusually QUIET, so a lone "8.5 score" headline this
# window is more noteworthy than the same score in a busy window. Today the
# briefing has neither read — it composes the LEAD/TOP SIGNALS without any
# awareness of how the standalone-push channel is firing.
#
# Same shape as COVERAGE GAP / THROUGHPUT DEGRADATION (the operational-status
# family): a separate input block, REPRODUCED in the output as a one-line
# section, omitted entirely when the change is below the magnitude bar (so it
# never becomes noise). Pure read-side: NO DB write, NO ai_score / ml_score /
# score_source / urgency mutation, never reads or mutates source_articles,
# backtest already excluded by _LIVE_ONLY_CLAUSE — all four load-bearing
# invariants intact by construction. Counts only urgency=2 (the alerted state
# — what actually fired); urgency=1 is the queued/phantom state (see the
# 2026-05-13 reaper evidence) and is correctly excluded.
#
# Thresholds (conservative — the analyst's #1 complaint is noise):
#   * recent + prior >= 5: a 1→3 swing on a sleepy wire is statistical
#     noise; we want a materially-different intensity, not micro-fluctuations.
#   * |delta_pct| >= 50: half-or-double from prior, both directions. A 25%
#     swing is normal news-rate variance, not analyst-actionable.
# Two special cases that bypass the percentage gate (the ratio is undefined
# or near-infinite, but the absolute change is itself the signal):
#   * recent >= 5 AND prior == 0: a previously-dark wire just lit up — a
#     real wire-event signal, regardless of % math (-/0 is undefined).
#   * recent == 0 AND prior >= 5: the wire just went silent — also notable.
_ALERT_VELOCITY_MIN_TOTAL = 5
_ALERT_VELOCITY_MIN_DELTA_PCT = 50.0


def _collect_alert_velocity(window_hours: int = 5) -> dict | None:
    """Best-effort: count BREAKING alerts ACTUALLY FIRED in the recent
    ``window_hours`` and the immediately-preceding window of the same length.
    Returns ``{"recent": int, "prior": int, "window_h": int}`` or ``None`` on
    ANY failure — an alert-temperature read must NEVER break or delay the 5h
    briefing it annotates (identical discipline to ``_collect_source_health``
    / ``_collect_source_throughput`` / ``_recent_briefing_digest``).

    Reads from ``watchers.alert_recency.DB_PATH`` (one row per fired-alert
    signature, ``last_ts`` = most recent successful Discord send). The earlier
    implementation counted ``urgency=2`` rows in ``articles.db``, but that
    state is ALSO set by the four pre-fire suppression gates in
    ``watchers.alert_agent.send_urgent_alert`` (quote-widget, stale-by-
    published, low-authority-lone, cross-cycle-syndication): each gate calls
    ``store.mark_alerted_batch`` to exit the suppressed row from the urgent
    queue WITHOUT firing a Discord push. So a story alerted once and then
    re-suppressed across five later re-syndicated copies counted as six
    "fires" and told Opus the wire was materially hotter than it was —
    inflating LEAD/macro framing toward noise on the analyst's primary
    consumed product. Live evidence (2026-05-19 10h window): 58 ``urgency=2``
    rows in ``articles.db`` vs 37 hits in ``alert_recency.db`` — the metric
    over-counted by ~57%. The recency store ONLY records on a successful
    Discord send (see ``alert_agent.send_urgent_alert`` and
    ``alert_recency.record_alerted``) so its count is the canonical
    "pushed to analyst" tally.

    Minor known under-count: ``record_alerted`` upserts on conflict (one row
    per signature, ``last_ts`` = LATEST fire), so a signature that fired
    twice within the queried 2× window is counted ONCE in whichever sub-
    window its latest fire falls in. Same 2026-05-19 snapshot: 36 distinct
    signatures vs 37 hits = a single re-fire over 10h, ~3% under-count.
    Trading a 57% over-count for a 3% under-count is unambiguous — and the
    error direction is analyst-safe (a false "wire quiet" just means a real
    story stands out more in TOP SIGNALS; a false "wire hot" inflated the
    LEAD toward macro framing on noise).

    Opens a fresh short-lived ``mode=ro`` connection (never the daemon's
    shared ``self.conn`` — the documented cursor-collision hazard); the
    ``alert_recency.db`` file is tiny (~36 rows in production) and indexed
    on ``last_ts``, so each count is a few µs. The four load-bearing
    invariants are unchanged: this read never touches ``articles.db``, so
    backtest isolation / ml_score≠ai_score / score_source / the urgency
    state machine are intact by construction."""
    try:
        import sqlite3
        from watchers.alert_recency import DB_PATH as _RECENCY_DB_PATH
        from datetime import timedelta as _td
        now = datetime.now(timezone.utc)
        recent_cut = (now - _td(hours=window_hours)).isoformat()
        prior_cut = (now - _td(hours=2 * window_hours)).isoformat()
        conn = sqlite3.connect(
            f"file:{_RECENCY_DB_PATH}?mode=ro", uri=True, timeout=5,
        )
        try:
            conn.execute("PRAGMA busy_timeout=5000")
            row = conn.execute(
                "SELECT "
                "SUM(CASE WHEN last_ts >= ? THEN 1 ELSE 0 END) AS recent, "
                "SUM(CASE WHEN last_ts >= ? AND last_ts < ? "
                "         THEN 1 ELSE 0 END) AS prior "
                "FROM alerted_sig WHERE last_ts >= ?",
                (recent_cut, prior_cut, recent_cut, prior_cut),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        recent = int(row[0] or 0)
        prior = int(row[1] or 0)
        return {"recent": recent, "prior": prior, "window_h": int(window_hours)}
    except Exception:
        return None


def _alert_velocity_lines(
    velocity: dict | None,
    min_total: int = _ALERT_VELOCITY_MIN_TOTAL,
    min_delta_pct: float = _ALERT_VELOCITY_MIN_DELTA_PCT,
) -> list[str]:
    """Pure: render an alert-velocity dict into 0 or 1 analyst-facing lines.

    Three emit paths (all anchored on absolute count so micro-fluctuations
    on a sleepy wire stay silent):

      * ``recent + prior >= min_total`` AND ``|delta_pct| >= min_delta_pct``
        — normal-range change, both directions ("+200%" / "-75%");
      * ``recent >= min_total`` AND ``prior == 0`` — previously-dark wire
        just lit up (percentage undefined; report "fired vs 0 prior");
      * ``recent == 0`` AND ``prior >= min_total`` — wire just went silent
        ("0 fired vs N prior — wire silent").

    Returns [] otherwise so the caller omits the whole section — same
    "omit when below threshold" discipline as ``_coverage_gap_lines`` /
    ``_throughput_degradation_lines``. The output is a SINGLE line (this
    section can never itself become a wall of text).
    """
    if not isinstance(velocity, dict):
        return []
    try:
        recent = int(velocity.get("recent") or 0)
        prior = int(velocity.get("prior") or 0)
        window_h = int(velocity.get("window_h") or 0)
    except (TypeError, ValueError):
        return []
    if recent < 0 or prior < 0 or window_h <= 0:
        return []
    # Newly-lit / newly-silent edges first — bypass the percentage gate
    # because the ratio is undefined / divergent.
    if recent >= min_total and prior == 0:
        return [
            f"BREAKING wire fired {recent} alert(s) in last {window_h}h "
            f"vs 0 in prior {window_h}h — wire newly active"
        ]
    if recent == 0 and prior >= min_total:
        return [
            f"BREAKING wire fired 0 alerts in last {window_h}h "
            f"vs {prior} in prior {window_h}h — wire silent"
        ]
    if recent + prior < min_total:
        return []
    # prior > 0 here (recent+prior >= 5 AND prior == 0 was handled above);
    # delta_pct is well-defined.
    delta_pct = (recent - prior) / prior * 100.0
    if abs(delta_pct) < min_delta_pct:
        return []
    sign = "+" if delta_pct >= 0 else ""
    label = "hot" if delta_pct > 0 else "cooling"
    return [
        f"BREAKING wire fired {recent} alerts in last {window_h}h "
        f"vs {prior} in prior {window_h}h ({sign}{delta_pct:.0f}%) — "
        f"wire materially {label}"
    ]


# ── Per-held-ticker alert velocity (book-level magnitude) ────────────────────
# ALERT VELOCITY tracks the OVERALL BREAKING wire firing-rate; BOOK HEAT tracks
# how many DISTINCT digest rows touch each held name. Neither answers the
# analyst-persona question the operator most cares about for *open positions*:
# is one of MY held names itself the centre of the breaking-wire activity this
# window? A held ticker carried by one alert is generic news (already surfaced
# by the per-row [BOOK:] tag); the SAME held ticker carried by 4 distinct
# breaking alerts in 5h vs 1 prior is a magnitude signal in its own right —
# concentration on the position the analyst has money in.
#
# Same pure read-side, BOOK-HEAT-shaped contract as the existing held-book
# blocks (separate input block, NEVER a per-row token, NEVER echoed): no DB
# write, no ai_score/ml_score/score_source/urgency touch, no row mutation,
# never reads or mutates source_articles, alert_recency.db is a separate file
# (NOT articles.db) so backtest isolation holds by construction — all four
# load-bearing invariants intact by construction. Data source mirrors
# ``_collect_alert_velocity`` (the canonical fires log — ``alert_recency.db``
# — NOT ``articles.db`` urgency=2 which also marks pre-fire-suppressed rows).
#
# Same minor under-count caveat as _collect_alert_velocity: ``record_alerted``
# upserts on signature conflict (one row per signature, ``last_ts`` = LATEST
# fire), so a signature firing twice across the 2× window is counted ONCE in
# whichever sub-window its latest fire falls in. Trading the over-count of
# articles.db-urgency=2 for this small ~3% under-count is unambiguous (the
# error direction is analyst-safe: a held ticker briefly under-counted just
# means it stays on the per-row [BOOK:] tag instead of getting the
# multiplicity callout — silent, not noisy).
#
# Thresholds (conservative — the analyst's #1 complaint is noise):
#   * recent >= 2: a single alert mentioning a held ticker is normal news,
#     already surfaced by the per-row [BOOK:] tag and the cross-cycle alert-
#     recency [ALERTED] briefing tag. This block's job is MULTIPLICITY —
#     two or more breaking alerts mentioning the same held name in the
#     window is the signal worth surfacing as a separate magnitude hint.
ALERT_BOOK_VELOCITY_MIN_RECENT = 2
_ALERT_BOOK_VELOCITY_MAX_LINES = 4


def _collect_alert_book_velocity(
    window_hours: float = 5.0,
    now: datetime | None = None,
) -> dict | None:
    """Best-effort: per-held-ticker BREAKING-alert counts recent vs prior window.

    Reads from ``watchers.alert_recency.recent_alerts`` (the canonical fires
    log — NOT ``articles.db`` ``urgency=2`` which is also set by the four
    pre-fire suppression gates in ``watchers.alert_agent.send_urgent_alert``
    without firing a Discord push; see ``_collect_alert_velocity`` for the
    full data-source rationale). Each fired-alert row's stored title is
    scanned for held tickers via ``_book_tickers`` (reuses the same primitive
    the briefing's BOOK HEAT / [BOOK:] tag use — single source of truth so
    the three surfaces can never silently drift apart about whether a wire
    touches the held book).

    Returns ``{"window_h": int, "tickers": {T: {"recent": N, "prior": N},
    ...}}`` or ``None`` on ANY failure (missing/locked DB, import error) —
    an alert-book-velocity read must NEVER break or delay the 5h briefing
    it annotates (identical discipline to ``_collect_source_health`` /
    ``_collect_alert_velocity`` / ``_recent_briefing_digest``).
    """
    try:
        from watchers.alert_recency import recent_alerts as _recent_alerts
        # Need 2× window of alerts to split recent vs prior. Defensive cap
        # at the alert_recency 2× TTL prune so we never ask for data the
        # store has pruned.
        from watchers.alert_recency import ALERT_RECENCY_TTL_HOURS as _TTL
        ttl = min(2.0 * float(window_hours), 2.0 * float(_TTL))
        alerts = _recent_alerts(ttl_hours=ttl, now=now) or []
        tickers: dict[str, dict[str, int]] = {}
        cutoff_recent = float(window_hours)
        cutoff_window = 2.0 * float(window_hours)
        for a in alerts:
            try:
                age_h = float(a.get("age_hours") or 0.0)
            except (TypeError, ValueError):
                continue
            if age_h < 0 or age_h > cutoff_window:
                continue
            book = _book_tickers(
                {"title": a.get("title") or "", "summary": ""}
            )
            if not book:
                continue
            bucket = "recent" if age_h <= cutoff_recent else "prior"
            for t in book:
                if t not in tickers:
                    tickers[t] = {"recent": 0, "prior": 0}
                tickers[t][bucket] += 1
        return {
            "window_h": int(round(float(window_hours))),
            "tickers": tickers,
        }
    except Exception:
        return None


def _alert_book_velocity_lines(
    velocity: dict | None,
    min_recent: int = ALERT_BOOK_VELOCITY_MIN_RECENT,
    max_lines: int = _ALERT_BOOK_VELOCITY_MAX_LINES,
) -> list[str]:
    """Pure: render per-held-ticker BREAKING-alert velocity into ranked lines.

    Skip a ticker with ``recent < min_recent``: single-alert noise is already
    surfaced by the per-row ``[BOOK:]`` tag and the briefing's ``[ALERTED]``
    parity tag. This block's job is MULTIPLICITY (≥2 alerts on the same held
    name in window). A ``prior == 0`` baseline with ``recent >= min_recent``
    is the STRONGEST per-position signal (newly-active wire on a held name)
    and is emitted, just like ``_alert_velocity_lines``' newly-lit edge.

    Sort: ``recent`` desc, then canonical ``_BOOK_TICKERS`` order — same
    stable cycle-to-cycle tiebreak as ``_book_heat_lines`` / ``_book_tickers``
    / ``_book_silence_lines`` so all four held-book surfaces describe the
    same window in the same order.

    Returns ``[]`` on non-dict input or unparseable ``window_h`` — same
    "degrade rather than crash the briefing" discipline as
    ``_alert_velocity_lines``.
    """
    if not isinstance(velocity, dict):
        return []
    tickers = velocity.get("tickers")
    if not isinstance(tickers, dict) or not tickers:
        return []
    try:
        window_h = int(velocity.get("window_h") or 0)
    except (TypeError, ValueError):
        return []
    if window_h <= 0:
        return []
    rank = {t: i for i, t in enumerate(_BOOK_TICKERS)}
    rows: list[tuple[int, int, str]] = []
    for t, counts in tickers.items():
        if not isinstance(counts, dict):
            continue
        try:
            r = int(counts.get("recent") or 0)
            p = int(counts.get("prior") or 0)
        except (TypeError, ValueError):
            continue
        if r < min_recent or r < 0 or p < 0:
            continue
        rows.append((
            -r,
            rank.get(t, len(rank)),
            f"{t} — {r} BREAKING alerts mention this name in last "
            f"{window_h}h (vs {p} in prior {window_h}h)",
        ))
    rows.sort()
    return [line for _, _, line in rows[:max_lines]]


# ── Forward macro calendar (the OUT-of-articles forward catalysts) ───────────
# EARNINGS CALENDAR is supplied by the caller (a separate yfinance scrape) so
# the briefing has explicit forward awareness of every held/watched ticker's
# next earnings print. The 2026-05-18 ``macro_calendar_collector`` ships the
# parallel FORWARD MACRO awareness — FOMC meetings + BLS CPI/Jobs/PPI releases
# — into articles.db with ``source='macro_calendar'`` and ``published`` set to
# the FUTURE event datetime. But those rows have no surfacing in the briefing
# beyond the generic NEWSWIRE rank order: a TODAY FOMC sitting at #34 in a
# busy-wire window is read by Opus as a generic newswire item, not as the
# market-wide rate decision it actually is.
#
# This is the read-side complement to that collector: a SEPARATE forward
# block, parallel in shape to EARNINGS CALENDAR (a REPRODUCED section,
# operational-status family — same surfacing discipline as COVERAGE GAP /
# THROUGHPUT DEGRADATION / ALERT VELOCITY, not an INPUT hint like BOOK HEAT).
# Pure read-side: a fresh ``mode=ro`` connection (never the daemon's shared
# self.conn — the documented cursor-collision hazard), best-effort → [] on
# any failure so the 5h briefing is NEVER broken or delayed. No DB write,
# no ai_score/ml_score/score_source/urgency touch, never reads or mutates
# source_articles, backtest already excluded by _LIVE_ONLY_CLAUSE — all four
# load-bearing invariants intact by construction.
#
# Why ``source='macro_calendar'`` only (not ``LIKE 'macro_calendar%'`` or a
# broader filter): the collector writes that exact tag, and we want this
# block to surface ONLY curated forward-event rows, NEVER a general news
# headline that happens to mention FOMC. The alert path already catches
# breaking-rate-decision news via the standard urgency pipeline; this is
# the SCHEDULED-event surface, kept clean and orthogonal.
#
# Why dedup by ``published`` (not by title): each event can be present in
# articles.db with multiple title prefixes ("UPCOMING (5d)" / "TOMORROW" /
# "TODAY") — the day-class transitions are a feature, not a bug. The most
# recent row (``MAX(first_seen)``) carries the sharpest prefix for the
# current day, so picking it per ``published`` instant naturally surfaces
# "TODAY: FOMC ..." when an FOMC is today and "UPCOMING (3d): FOMC ..."
# when it's three days out, without us re-parsing the title.
MACRO_CALENDAR_WINDOW_HOURS = 72
_MACRO_CALENDAR_MAX_LINES = 6


def _collect_macro_calendar_events(
    window_hours: int = MACRO_CALENDAR_WINDOW_HOURS,
    now: datetime | None = None,
) -> list[dict] | None:
    """Best-effort: forward macro events from articles.db (source=macro_calendar)
    within the next ``window_hours``. Returns a list of ``{"title": str,
    "published": str, "hours_until": float}`` sorted ascending by ``hours_until``
    (most-imminent first), or ``None`` on ANY failure — a forward-calendar
    read must NEVER break or delay the 5h briefing it annotates (identical
    discipline to ``_collect_source_health`` / ``_collect_alert_velocity``).

    Dedup by the ``published`` instant — each event lifetime can write up to
    4 rows (one per day_class transition; see ``macro_calendar_collector``);
    we want exactly ONE entry per scheduled event, carrying the freshest
    title prefix. ``MAX(first_seen)`` gives the latest day-class emission per
    event datetime, so "TODAY: FOMC Meeting — March 17, 2026" wins on the day
    of the meeting and the obsolete "UPCOMING (5d): ..." is correctly hidden.
    """
    try:
        import sqlite3
        from storage.article_store import _get_db_path
        from datetime import timedelta as _td
        cur_now = now or datetime.now(timezone.utc)
        horizon = (cur_now + _td(hours=window_hours)).isoformat()
        now_iso = cur_now.isoformat()
        conn = sqlite3.connect(
            f"file:{_get_db_path()}?mode=ro", uri=True, timeout=5,
        )
        try:
            conn.execute("PRAGMA busy_timeout=5000")
            # Dedup on ``published`` (the event instant) — the freshest
            # first_seen row per instant carries the sharpest day_class
            # prefix. Bounded LIMIT so a runaway insertion can never
            # produce a multi-page briefing block.
            rows = conn.execute(
                "SELECT title, published, MAX(first_seen) AS fs "
                "FROM articles "
                "WHERE source = 'macro_calendar' "
                "AND published >= ? AND published <= ? "
                "GROUP BY published "
                "ORDER BY published ASC "
                "LIMIT 50",
                (now_iso, horizon),
            ).fetchall()
        finally:
            conn.close()
        out: list[dict] = []
        for title, published, _fs in rows:
            if not title or not published:
                continue
            try:
                ev_dt = datetime.fromisoformat(str(published).replace("Z", "+00:00"))
                if ev_dt.tzinfo is None:
                    ev_dt = ev_dt.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                continue
            hours_until = (ev_dt - cur_now).total_seconds() / 3600.0
            if hours_until < 0:
                continue  # event already past — should not happen given SQL filter, defensive
            out.append({
                "title": str(title),
                "published": str(published),
                "hours_until": round(hours_until, 1),
            })
        return out
    except Exception:
        return None


def _macro_calendar_event_lines(
    events: list[dict] | None,
    max_lines: int = _MACRO_CALENDAR_MAX_LINES,
) -> list[str]:
    """Pure: render forward macro events as compact one-line entries.

    Each line carries the event title verbatim (which the collector already
    composed with the appropriate day-class prefix — TODAY / TOMORROW /
    UPCOMING (Nd) / IN Nd) plus a human-readable hours-until tag (``~Nh``
    for sub-day, ``~Nd`` for multi-day so the line communicates urgency at
    a glance independent of how the title is worded).

    ``[]`` when nothing usable — the same omit-when-empty discipline as
    ``_coverage_gap_lines`` / ``_alert_velocity_lines``. Non-list / empty
    / malformed entries are skipped, never raise (the analyst's #1
    complaint is noise, so a broken row degrades to silence)."""
    if not isinstance(events, list) or not events:
        return []
    out: list[str] = []
    for e in events:
        if not isinstance(e, dict):
            continue
        title = (e.get("title") or "").strip()
        if not title:
            continue
        h = e.get("hours_until")
        try:
            hours = float(h)
        except (TypeError, ValueError):
            out.append(title)
            continue
        if hours < 24.0:
            tag = f"~{int(round(hours))}h"
        else:
            tag = f"~{int(round(hours / 24.0))}d"
        out.append(f"{title} ({tag})")
        if len(out) >= max_lines:
            break
    return out


# ── Prior-digest continuity (anti-rehash) ────────────────────────────────────
# A news analyst reading consecutive 5h heartbeats complains most about
# repetition: the briefing re-LEADS with the SAME story it led with last time.
# Confirmed live 2026-05-18: briefing id26 (07:13Z) LEAD = "Global bond rout
# deepens … dragging Nasdaq -1.54% … two days before NVDA earnings"; id27
# (12:51Z, 5.6h later) LEAD = "Iran-war inflation scare drives a global bond
# rout … semis dump into NVDA earnings Wed" — the same story led twice, the
# analyst's documented #1 noise complaint, on the primary consumed product.
#
# The alert path already has alert↔briefing parity (the [ALERTED] tag reads
# alert_recency.db); the briefing path never saw its OWN previous output. A
# per-article-title match against the rendered prior briefing was measured at
# 0% recall (Opus paraphrases every headline, so a raw title prefix never
# appears verbatim in the prose) — so the robust mechanism is to parse the
# prior briefing's OWN deterministic SYSTEM_PROMPT format (the literal
# ``**LEAD:**`` line + ``**TOP SIGNALS**`` fenced block) and feed it back as a
# framing hint so OPUS does the semantic "is this the same story" comparison
# (its strength), exactly as it already does for BOOK HEAT / AGING TOP ROWS.
#
# Read-only and best-effort, mirroring _collect_source_health /
# _recent_alert_signatures EXACTLY: a lazy fresh ``mode=ro`` connection (never
# the daemon's shared self.conn — the documented cursor-collision hazard), one
# O(log N) indexed read of the tiny ``briefings`` table, ANY failure → None so
# the 5h briefing is never broken or delayed. The ``briefings`` table holds
# only Opus-rendered briefing rows (synthetic backtest rows live in
# ``articles``, NEVER here) so backtest isolation holds by construction; no
# articles.db write, no ai_score/ml_score/score_source/urgency touch, the
# ``source_articles`` newswire list is never read or mutated by this path —
# all four load-bearing invariants intact by construction.
_PRIOR_DIGEST_MAX_SIGNALS = 6
# Heartbeat retry can persist the "[analyst] No response from Claude." sentinel
# into ``briefings`` (live: 3 of 27 rows). The prior-digest read MUST skip
# those — a sentinel "prior briefing" carries no LEAD/signals and would
# silently disable the hint. Filtered in SQL so the newest *real* digest wins.
_PRIOR_DIGEST_SENTINEL_SQL = (
    "text NOT LIKE '%[analyst] No response%' "
    "AND text NOT LIKE '%No response from Claude%'"
)


def _parse_prior_digest(text: str) -> dict:
    """Pure: extract ``{"lead": str, "top_signals": [str, ...]}`` from a prior
    briefing's text by parsing OUR OWN deterministic SYSTEM_PROMPT output
    format (the ``**LEAD:**`` line and the ``**TOP SIGNALS**`` fenced block).

    Deliberately NOT a fuzzy headline match (that was measured at 0% recall —
    Opus paraphrases every title). Garbage / missing sections degrade to empty
    strings/lists, never raise."""
    out = {"lead": "", "top_signals": []}
    if not text or not isinstance(text, str):
        return out
    m = re.search(r"\*\*LEAD:\*\*\s*(.+)", text)
    if m:
        out["lead"] = m.group(1).strip()
    ts_idx = text.find("**TOP SIGNALS**")
    if ts_idx != -1:
        rest = text[ts_idx:]
        fences = [mm.start() for mm in re.finditer(r"```", rest)]
        if len(fences) >= 2:
            block = rest[fences[0] + 3:fences[1]]
            lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
            out["top_signals"] = lines[:_PRIOR_DIGEST_MAX_SIGNALS]
    return out


def _prior_digest_lines(prior: dict | None) -> list[str]:
    """Pure: render the PRIOR DIGEST input-block body lines from a parsed
    prior-digest dict. ``[]`` when nothing usable (so the caller omits the
    whole block — the deterministic, BOOK-HEAT/AGING-shaped contract)."""
    if not isinstance(prior, dict):
        return []
    lead = (prior.get("lead") or "").strip()
    sigs = [s for s in (prior.get("top_signals") or []) if str(s).strip()]
    if not lead and not sigs:
        return []
    out: list[str] = []
    if lead:
        out.append(f"LEAD (last briefing): {lead}")
    for s in sigs[:_PRIOR_DIGEST_MAX_SIGNALS]:
        out.append(f"TOP SIGNAL (last briefing): {str(s).strip()}")
    return out


def _recent_briefing_digest(now: datetime | None = None) -> dict | None:
    """Best-effort: the LEAD + TOP SIGNALS of the most recent NON-sentinel
    prior briefing, plus its age in hours. ``None`` on ANY failure (missing /
    locked DB, no real prior briefing, unparseable) — an anti-rehash read must
    NEVER break or delay the 5h briefing it annotates (identical discipline to
    ``_collect_source_health`` / ``_recent_alert_signatures``).

    Opens a fresh short-lived ``mode=ro`` connection (never the daemon's shared
    ``self.conn``); the ``briefings`` table is tiny and PK-ordered so
    ``ORDER BY id DESC LIMIT 1`` is an O(log N) rightmost-leaf walk."""
    try:
        import sqlite3
        from storage.article_store import _get_db_path
        conn = sqlite3.connect(f"file:{_get_db_path()}?mode=ro",
                               uri=True, timeout=5)
        try:
            conn.execute("PRAGMA busy_timeout=5000")
            row = conn.execute(
                "SELECT ts, text FROM briefings "
                f"WHERE {_PRIOR_DIGEST_SENTINEL_SQL} "
                "ORDER BY id DESC LIMIT 1"
            ).fetchone()
        finally:
            conn.close()
        if not row or not row[1]:
            return None
        ts, text = row
        parsed = _parse_prior_digest(text)
        if not parsed["lead"] and not parsed["top_signals"]:
            return None
        age_h: float | None = None
        try:
            dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            now = now or datetime.now(timezone.utc)
            a = (now - dt.astimezone(timezone.utc)).total_seconds() / 3600.0
            age_h = a if a > 0 else 0.0
        except Exception:
            age_h = None
        return {"age_h": age_h, "lead": parsed["lead"],
                "top_signals": parsed["top_signals"]}
    except Exception:
        return None


SYSTEM_PROMPT = """You are a financial intelligence briefing engine. Output is posted directly to Discord. Format must render cleanly there.

RULES:
- Every number exact. Every move has a cause. Zero hedging.
- Tickers in ALL CAPS. Prices to 2dp. Pct changes with sign (+/-).
- Each table in its own code block. Section headers as plain **bold** outside code blocks.
- Total output must fit in 1800 characters. Be ruthlessly concise. Cut low-signal rows.
- No nested backticks. No backtick dividers. Dividers are plain ━━━ lines outside code blocks.
- A newswire row tagged "[syndicated xN]" was independently carried by N sources — treat higher N as stronger corroboration/magnitude when choosing the LEAD and ordering TOP SIGNALS; a lone (untagged) item is single-sourced and less confirmed.
- A newswire row ALSO tagged "[echo]" alongside "[syndicated xN]" is a calibration warning: those N copies all came from ONE source (a single outlet self-syndicating slight title variants of the same wire) — NOT independent corroboration. Down-weight: do not lead a "[syndicated xN] [echo]" row over a comparable single-but-credible row, and never treat the N count as cross-outlet confirmation when choosing the LEAD or ranking TOP SIGNALS. An untagged "[syndicated xN]" (no [echo]) means the N copies WERE distinct outlets — full corroboration credit applies.
- A newswire row tagged "[model]" carries a score set by the local relevance model ONLY, with NO LLM verification; that model demonstrably over-scores forum/wiki/social rows. Treat an untagged (LLM-vetted) row as materially more trustworthy than a "[model]" row of equal or near-equal score: prefer untagged rows for the LEAD and rank them above "[model]" rows of similar score in TOP SIGNALS. NEVER make a lone "[model]" row the LEAD when an untagged row of comparable score exists.
- A newswire row tagged "[ALERTED]" ALREADY fired a standalone 🚨 BREAKING push to the analyst within the last few hours — it is a developing/continued story the analyst has ALREADY been told about, NOT new news. Do NOT make an "[ALERTED]" row the LEAD when any untagged story of comparable importance exists; rank a fresh untagged story above an "[ALERTED]" one of similar score in TOP SIGNALS; and frame any "[ALERTED]" item explicitly as continuation (e.g. "follows the earlier alert", "developing") — never as if it just broke. This is what separates new desk intel from a rehash of an alert already delivered.
- A newswire row tagged "[BOOK: TICKER,...]" names live portfolio/watchlist positions the analyst actually holds money in (LITE, LNOK, MUU, DRAM, MU, NVDA, MSFT, AXTI, ORCL, TSEM, QBTS). A "[BOOK:...]" row is directly actionable for the analyst's open risk: weight it ABOVE a same-score untagged general-market row when choosing the LEAD and ordering TOP SIGNALS, always reflect its named ticker(s) in the PORTFOLIO table with a concrete implication, and never bury a "[BOOK:...]" item below generic macro colour of similar magnitude. Absence of the tag means the row does not touch the held book.
- If a "BOOK HEAT" block is present, it ranks the analyst's held names by how many DISTINCT stories this window touched each one. Concentration on a single held name is itself a magnitude signal, independent of any one row's score: strongly prefer the most-concentrated held name for the LEAD when it has any material story, rank its stories up in TOP SIGNALS, and give that ticker a concrete forward-looking implication in the PORTFOLIO table. This is a ranking/weighting hint only — do NOT echo a literal "BOOK HEAT" section in the output (unlike COVERAGE GAP).
- If a "BOOK SILENCE" block is present, it lists held tickers with ZERO mentions this 5h window — the analyst has open risk on those names but no incoming catalyst this window. In the PORTFOLIO table, mark each silent ticker with a brief honest "N/A — no catalyst this window" (or similar terse note, never a fabricated implication or hedge filler like "continued caution"). Silent names should NOT lead and should NOT outrank a ticker with material news in TOP SIGNALS. This is an honesty/composition hint only — do NOT echo a literal "BOOK SILENCE" section in the output (same as BOOK HEAT / AGING TOP ROWS, unlike COVERAGE GAP).
- If an "AGING TOP ROWS" block is present, it names the highest-ranked digest rows whose deterministic wall-clock age (time since the story hit our wire) is several hours old — a ground-truth recency cross-check, independent of any row's score or decay rank. An aging top row is a developing/continued story, NOT one that just broke: do NOT make it the LEAD as if it were fresh when a comparably-important newer row exists, frame it explicitly as developing in the LEAD and TOP SIGNALS, and never imply a multi-hour-old item happened moments ago. This is a ranking/framing hint only — do NOT echo a literal "AGING TOP ROWS" section in the output (same as BOOK HEAT, unlike COVERAGE GAP).
- If a "COVERAGE GAP" block is present in the data input, reproduce it as a **COVERAGE GAP** section (one bullet per dark channel, verbatim). These are intel channels the system could NOT collect from this window — the analyst must know what they are blind to, not assume silence means calm. Omit the section entirely if no gap block is provided.
- If a "THROUGHPUT DEGRADATION" block is present in the data input, reproduce it as a **THROUGHPUT DEGRADATION** section directly under the COVERAGE GAP bullets (one bullet per source, verbatim). These sources are still alive but have lost most of their recent flow — partial blind spots an analyst must know about (the early-warning complement to COVERAGE GAP: a marginally-alive source has not crossed the disable threshold yet, but the briefing has materially less coverage from it than the prior window). Omit the section entirely if no degradation block is provided.
- If an "ALERT VELOCITY" block is present in the data input, reproduce it as an **ALERT VELOCITY** section under the THROUGHPUT DEGRADATION bullets (one bullet, verbatim). This is the BREAKING-alert wire's firing-rate vs the prior window of the same length: an objective magnitude signal independent of any individual story's score. Use it for cumulative framing — a "wire materially hot" window means cumulative event flow is itself the story (weight the LEAD accordingly: macro/risk regime, not a lone headline); a "wire silent" window means scrutinise any one BREAKING-tagged story more carefully. Omit the section entirely if no velocity block is provided.
- If an "ML SCORER STALE" block is present in the data input, reproduce it as an **ML SCORER STALE** section directly under the ALERT VELOCITY bullet (one bullet, verbatim). It means the local ArticleNet model that scores every collected article — and produces the "[model]" urgent calls — has not successfully retrained for several hours, so its relevance/urgency scores are running on stale weights. Treat "[model]"-tagged rows with extra caution this window and lean harder on LLM-vetted (untagged) rows when choosing the LEAD and ranking TOP SIGNALS. Omit the section entirely if no stale block is provided.
- If an "ALERT BOOK VELOCITY" block is present, it lists held names mentioned in MULTIPLE 🚨 BREAKING alerts in the recent window vs the prior window of the same length — per-position alert-rate magnitude. The per-row [BOOK:] tag flags WHICH rows touch the held book; this block flags that the held name is itself the WINDOW'S hot centre of breaking-wire activity (≥2 distinct alerts mention it). Weight these held names above other rows of comparable score when choosing the LEAD and ordering TOP SIGNALS, and give them a concrete forward-looking implication in the PORTFOLIO table. This is a ranking/weighting hint only — do NOT echo a literal "ALERT BOOK VELOCITY" section in the output (same as BOOK HEAT / BOOK SILENCE / AGING TOP ROWS, unlike COVERAGE GAP).
- If a "PRIOR DIGEST" block is present, it is the LEAD and TOP SIGNALS YOU YOURSELF published in your previous 5h briefing — the analyst just read that one. Re-leading the SAME story as if it were new is the single most-cited "repetitive digest" complaint. If the dominant story is materially unchanged, do NOT restate it as the LEAD: lead instead with what has CHANGED since (a new level/print/catalyst, a reversal, or a genuinely different top story that now outranks it), and frame any necessarily-carried theme explicitly as continuation/development ("rout now easing", "follows this morning's selloff"), never as a fresh break. This is a framing/selection hint only — do NOT echo a literal "PRIOR DIGEST" section in the output (same as BOOK HEAT / AGING TOP ROWS, unlike COVERAGE GAP).
- If a "MACRO CALENDAR" block is present in the data input, reproduce it as a **MACRO CALENDAR** section directly above the DESK NOTE (one bullet per scheduled event, verbatim). These are scheduled forward catalysts — FOMC rate decisions, BLS CPI/Jobs/PPI releases — within the next 72h that materially reshape risk for a leveraged-ETF-heavy book. An imminent TODAY/TOMORROW FOMC or CPI is the single biggest market-wide event; the LEAD must explicitly factor it (e.g. "ahead of FOMC tomorrow", "into Friday's jobs print") and the RISK / CATALYST section must NAME each upcoming event with its remaining timing. A 5h+ horizon shifts how every individual story should be weighted — a strong move into a Fed decision is positioning, not pure signal. Omit the section entirely if no macro block is provided.

OUTPUT FORMAT — use EXACTLY this, filled with real data:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
**DIGITAL INTERN** ◈ [DATE TIME UTC]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
**LEAD:** [single most market-moving event, one sentence]

**MACRO**
```
INDEX        LAST       CHG%
S&P 500    x,xxx.xx   +x.xx%
NASDAQ    xx,xxx.xx   +x.xx%
VIX           xx.xx   [+/-x.x]
10Y UST        x.xx%  [+/-xbp]
BTC        $xx,xxx    +x.xx%
Gold       $x,xxx     +x.xx%
Oil (WTI)    $xx.xx   +x.xx%
```

**PORTFOLIO** (SAO — LITE · LNOK · MUU · DRAM CALL C59)
```
TICKER       PRICE     CHG%   NOTE
LITE       $x,xxx.xx  +x.xx%  [implication]
LNOK          $xx.xx  +x.xx%  [implication]
MUU          $xxx.xx  +x.xx%  [implication]
MU (watch)   $xxx.xx  +x.xx%  [DRAM call driver]
```

**SEMIS PULSE**
```
NVDA  $xxx  +x.xx%  |  MU  $xxx  +x.xx%  |  TSM  $xxx  +x.xx%
AMD   $xxx  +x.xx%  |  AMAT $xxx +x.xx%  |  SMH  $xxx  +x.xx%
```

**TOP SIGNALS**
```
[HH:MM] [score] [TICKER] headline — one line each, max 5
```

**RISK / CATALYST**
- [risk 1 — specific, tied to ticker/level]
- [risk 2]
- [upcoming catalyst with date and ticker]

**COVERAGE GAP** (only if a gap block is provided — else omit this whole section)
- [dark channel verbatim from the COVERAGE GAP data block]

**THROUGHPUT DEGRADATION** (only if a degradation block is provided — else omit this whole section)
- [degrading source verbatim from the THROUGHPUT DEGRADATION data block]

**ALERT VELOCITY** (only if a velocity block is provided — else omit this whole section)
- [BREAKING-wire firing rate verbatim from the ALERT VELOCITY data block]

**ML SCORER STALE** (only if a stale block is provided — else omit this whole section)
- [scorer staleness verbatim from the ML SCORER STALE data block]

**MACRO CALENDAR** (only if a macro block is provided — else omit this whole section)
- [scheduled FOMC / CPI / Jobs / PPI event verbatim from the MACRO CALENDAR data block]

**DESK NOTE:** [1-2 sentences. One thesis. One level to watch.]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

If data unavailable write N/A. Omit empty sections entirely.
"""


def _now_utc_str():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _seen_utc_str(first_seen) -> str | None:
    """Compact ``HH:MM`` UTC clock time the article hit our wire, or ``None``.

    ``SYSTEM_PROMPT``'s TOP SIGNALS line asks Opus for ``[HH:MM] [score]
    [TICKER] headline`` per signal, but ``_build_payload`` historically fed
    zero per-article time data — so Opus had to fabricate or omit every
    timestamp on the analyst's primary digest. This surfaces the real one.

    ``first_seen`` (collection instant, ISO-8601 written by
    ``article_store.insert_batch``) is used rather than ``published``: it is
    what ``get_top_for_briefing`` already returns in the row dict (no
    storage-layer change), and "when this hit our desk" is the relevant clock
    for a newswire digest. ``get_top_for_briefing`` already clamps every real
    row to the last 24h via ``_published_older_than``, and the briefing header
    carries the date — so a bare ``HH:MM`` is unambiguous, no date needed.

    RFC822 + ISO (``Z``-suffix tolerated), naive→UTC — the exact convention
    ``alert_agent._article_age_hours`` / ``urgency_scorer`` use, so the time
    shown here is consistent with the rest of the pipeline. ``None`` (unparseable
    or absent) makes the caller omit the token silently — the synthetic
    PORTFOLIO/OPTIONS snapshot rows the daemon prepends carry no ``first_seen``
    and must pass through cleanly (never a fabricated ``00:00``).
    """
    if not first_seen:
        return None
    raw = str(first_seen).strip()
    if not raw:
        return None
    dt = None
    try:
        dt = parsedate_to_datetime(raw)
    except Exception:
        dt = None
    if dt is None:
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except Exception:
            return None
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%H:%M")


# ── Per-article recency decay (the ML time_sensitivity head, finally used) ───
# ArticleNet trains a dedicated time_sensitivity head (0..1: 1.0 = decays fast
# — earnings beats, price moves, "today"; 0.0 = timeless — macro thesis,
# secular trend) and the store persists it per row, but until now NO consumer
# applied it. ``article_store.get_top_for_briefing`` documents the exact decay
# curve ("ts=1.0 halves the score every 12h, ts=0.0 disables decay entirely")
# yet deliberately returns ai_score unchanged so a consumer can pick the
# policy — and no consumer ever did. The 5h Opus digest therefore ranked an
# 18h-old "STOCK SURGED TODAY" item identically to a fresh same-score one, the
# consuming analyst's exact complaint (a newswire must lead with what is
# moving NOW, not a half-day-old time-bound headline that already played out).
#
# This applies that documented curve, here, where it belongs — purely on the
# text Opus reads (read-side rerank of the already-fetched, already-live-only
# digest). No DB write, no ai_score/ml_score/score_source/urgency touch,
# backtest rows already excluded upstream by get_top_for_briefing's
# _LIVE_ONLY_CLAUSE — all four load-bearing invariants intact by construction.
BRIEFING_DECAY_HALFLIFE_H = 12.0  # ts=1.0 → score halves every 12h (per the
                                  # get_top_for_briefing docstring contract)
# Unscored rows (no time_sensitivity yet — rare; most are ML-scored within the
# 30s scorer cadence) get a mild middle decay, matching the
# ml.inference.ArticleScore default so behaviour is consistent system-wide.
BRIEFING_DEFAULT_TS = 0.5

# ── Second-stage near-duplicate collapse threshold ──────────────────────────
# Jaccard token-set similarity at/above which two digest headlines are the
# SAME story. _collapse_syndicated (above) only merges an exact first-8-token
# prefix signature, so a word-reordered or source-attribution-suffixed copy of
# one wire ("Apple beats Q2" vs "Q2 beaten by Apple"; "...inflation fears" vs
# "...inflation fears | Daily Star") survives it and reaches the analyst's
# primary Opus digest as a duplicate TOP SIGNAL — the consuming analyst's #1
# noise complaint, on the one consumed product that had no order-independent
# gate. 0.7 is deliberately conservative: it collapses genuine paraphrase /
# attribution variants while a single-token ANTONYM flip in a short (4-5
# token) headline — "Fed raises rates 25bp" vs "Fed cuts rates 25bp" (J=0.60),
# "NVDA earnings beat Q3 estimates" vs "...miss..." (J=0.667) — stays strictly
# below it, so opposite-direction stories are provably never merged. Evidence:
# the live 2026-05-18 07:13Z briefing window carried 5 such residual dups
# (bond-rout ×3, Trump-Intel-stake ×1, ...) at sim 0.60-0.73, ALL genuine
# same-story paraphrases (a full pairwise audit of that window found zero
# semantically-opposite pairs ≥0.60).
BRIEFING_NEAR_DUP_THRESHOLD = 0.7


def _seen_age_hours(first_seen, now: datetime | None = None) -> float:
    """Hours since the article hit our wire (``first_seen``), else ``0.0``.

    Returns ``0.0`` (→ no decay) when ``first_seen`` is absent, unparseable,
    or in the future (clock skew / bad row) — every uncertain path degrades
    to "do not decay" so the rerank can only ever *help*, never bury a row on
    a parse failure. Crucially the synthetic PORTFOLIO/OPTIONS snapshot rows
    the daemon prepends carry no ``first_seen``, so they get age 0 → factor 1
    → stay pinned at the top of the digest (see _rank_by_decayed_score).

    RFC822 + ISO (``Z``-suffix tolerated), naive→UTC — the exact convention
    ``_seen_utc_str`` / ``alert_agent._article_age_hours`` / ``urgency_scorer``
    use, kept a small local parser rather than cross-imported (same
    anti-import-cycle discipline as _collapse_syndicated reusing _signature)."""
    if not first_seen:
        return 0.0
    raw = str(first_seen).strip()
    if not raw:
        return 0.0
    dt = None
    try:
        dt = parsedate_to_datetime(raw)
    except Exception:
        dt = None
    if dt is None:
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except Exception:
            return 0.0
    if dt is None:
        return 0.0
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    now = now or datetime.now(timezone.utc)
    age_h = (now - dt.astimezone(timezone.utc)).total_seconds() / 3600.0
    return age_h if age_h > 0 else 0.0


def _effective_score(article: dict, now: datetime | None = None) -> float:
    """Recency-decayed ranking score for one digest row.

        effective = base * 0.5 ** (age_hours * time_sensitivity / 12h)

    ``base`` is ``ai_score`` else ``_relevance_score`` — the SAME fallback
    ``_collapse_syndicated._score`` and the render line use, so a row's
    ranking number stays consistent with what it displays. A non-positive /
    unparseable base returns 0.0 (sorts last). ``time_sensitivity`` None or
    junk → BRIEFING_DEFAULT_TS, clamped 0..1. age 0 (snapshots / unparseable
    first_seen) or ts 0 (timeless) → factor 1 → base returned unchanged."""
    base = 0.0
    for key in ("ai_score", "_relevance_score"):
        v = article.get(key)
        if isinstance(v, bool):
            continue  # never let a stray bool read as 1.0
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if fv:
            base = fv
            break
    if base <= 0:
        return 0.0
    ts = article.get("time_sensitivity")
    try:
        ts = float(ts)
    except (TypeError, ValueError):
        ts = BRIEFING_DEFAULT_TS
    if ts != ts:  # NaN guard (float('nan') is the one value != itself)
        ts = BRIEFING_DEFAULT_TS
    ts = min(1.0, max(0.0, ts))
    age_h = _seen_age_hours(article.get("first_seen"), now=now)
    if age_h <= 0.0 or ts <= 0.0:
        return base
    return base * (0.5 ** (age_h * ts / BRIEFING_DECAY_HALFLIFE_H))


def _rank_by_decayed_score(articles: list, now: datetime | None = None) -> list:
    """Stable rerank of a collapsed digest by recency-decayed effective score
    (desc). Pure, side-effect-free, returns the same dicts (no copy).

    Stability is load-bearing: the prepended synthetic PORTFOLIO/OPTIONS
    snapshot rows have no ``first_seen`` → age 0 → factor 1 → effective ==
    base == their ai_score (10, the digest max, which decay can only lower
    for everyone else), and ``_collapse_syndicated`` already put them first;
    a *stable* descending sort therefore keeps them pinned ahead of any
    real article that merely ties at 10. Same-effective real rows keep their
    incoming (score-then-collapse) order — the rerank only ever *promotes* a
    fresher item over an older equal-base one, never reshuffles ties."""
    return sorted(articles, key=lambda a: _effective_score(a, now=now),
                  reverse=True)


def _fmt_ticker(s):
    # Keep the price column at width=11 ("$" + 10-char number) and pct column at
    # width=8 (signed 7-char number + "%") so N/A rows don't break alignment.
    price = f"${s['price']:>10.2f}" if isinstance(s.get('price'), (int, float)) else f"{'N/A':>11}"
    pct   = f"{s['pct_change']:>+7.2f}%" if isinstance(s.get('pct_change'), (int, float)) else f"{'N/A':>8}"
    # `or '?'` / `or ''` guard a present-but-None value — dict.get() only
    # applies its default on a *missing* key, so a row carrying ticker=None
    # would format as f"{None:>12}" and raise TypeError mid-briefing.
    ticker = s.get('ticker') or '?'
    return f"{ticker:>12}  {price}  {pct}  {(s.get('name') or '')[:25]}"


# ── Quote-widget noise gate (defense-in-depth, briefing path) ────────────────
# Yahoo/Bloomberg/Seeking-Alpha list pages embed a live ticker-tape sidebar
# whose every entry is an <a href="/quote/NVDA"> wrapping the rendered quote
# string with NO inter-field spaces, e.g.
# "NVDANVIDIA Corporation227.13-8.61(-3.65%)". Because the price changes every
# poll, the title (hence the article id) is unique each cycle, so one widget
# manufactures an unbounded stream of fake "breaking news". Live evidence
# (2026-05-18): 3,476 of 5,847 sampled scraped/* rows were these and the ML
# relevance head scored them up to 9.99 — the consuming analyst's single
# biggest noise complaint.
#
# collectors.web_scraper rejects these at ingestion and
# watchers.alert_agent._filter_quote_widget_noise drops them on the *alert*
# path, but the 5h Opus heartbeat digest — the analyst's PRIMARY consumed
# product — had NO such gate: a widget pseudo-article entering via a non-
# web_scraper path (yahoo_ticker_rss, finnhub, a manual replay) and ML-scored
# high still landed in the top-60 newswire Opus reads, surfacing as a fake
# "[HH:MM] [score] TOP SIGNAL". This is the same formatter-side, layered-
# defense shape as the alert path: a read-side text drop at the single
# chokepoint the briefing funnels through, NOT an ML-threshold change and NOT
# a DB write. The two title fingerprints + Yahoo /quote/ landing-path regex
# are byte-identical to alert_agent / web_scraper so the three gates stay in
# lockstep. The helper is duplicated rather than cross-imported from
# alert_agent: that module pulls the ml.features (numpy) import graph and the
# analysis layer must not (same documented anti-import-cycle discipline as
# _collapse_syndicated reusing alert_dedup._signature, and
# article_store._briefing_domain_key duplicating ml.features).
_QW_PRICE_GLUE = re.compile(r"[A-Za-z]\$?\d{1,4}[.,]\d{2,3}")
_QW_PCT_PAREN = re.compile(r"\([+-]?\d{1,3}(?:\.\d+)?%\)")
_QW_QUOTE_PATH = re.compile(r"/quote/[^/]+/?$", re.I)
# Quote-aggregator share-card / listing-page pseudo-article — a DISTINCT
# surface the two title fingerprints above miss. Google News indexes the
# Moomoo/Futu/Webull "share this quote" landing pages whose title is the
# rendered card: "$NVIDIA (NVDA.US)$ - Moomoo" / "$Tencent (00700.HK)$ - Futu".
# These are a live quote page, not news — the same pseudo-article class as the
# ticker-tape widget, and (ML-relevance over-scored, e.g. live 9.77) they reach
# the top-60 newswire Opus reads and render as a fake "[HH:MM] [score] TOP
# SIGNAL", the analyst's recurring noise complaint. The alert path also gates
# this now; the briefing — the analyst's PRIMARY consumed product — must too
# (the pass-16 precedent: every consumed product gets the quote-widget gate).
# Fingerprint = leading "$" share-card lead glued to a "(SYMBOL.EXCH)$" close;
# bounded ({0,60}) so no catastrophic backtracking; validated zero false
# positives on the live $+paren headline corpus. Byte-identical to
# watchers.alert_agent / collectors.web_scraper (the documented lockstep).
_QW_LISTING = re.compile(
    r"^\s*\$[^$\n]{0,60}\([A-Za-z0-9.\-]{1,8}\.[A-Za-z]{1,4}\)\$"
)
# Yahoo Finance screener-tape pseudo-article — lockstep mirror of the fourth
# fingerprint added to ``watchers.alert_agent._QW_SCREENER_TAPE``. Live evidence
# (2026-05-19): 4 of 12 last-2h BREAKING alerts were YF screener entries
# (``[YF/most_actives] MU (Micron Technology, Inc.) +2.5% @ $698.74 | vol 6``)
# all ml_score 9.9. The briefing's quote-widget gate had no matching pattern,
# so these would also land in the 5h digest as TOP SIGNALS. The leading
# ``[YF/<bucket>]`` tag is unique to ``collectors/market_movers.py``; the real
# source-column tag convention is unbracketed (``GDELT/reuters.com``,
# ``scraped/finance.yahoo.com``, ``GN: Nvidia``). Same lockstep-duplication
# discipline (anti-import-cycle) as the other three quote-widget fingerprints.
_QW_SCREENER_TAPE = re.compile(
    r"^\s*\[YF/[a-z_]+\]\s+[A-Z]"
)
# StockTwits sentiment pseudo-article — lockstep mirror of the fifth fingerprint
# added to ``watchers.alert_agent._QW_STOCKTWITS_SENTIMENT``. Live evidence
# (2026-05-21, last 5h): 130 ``[StockTwits Sentiment]`` rows from
# ``collectors/stocktwits_sentiment.py``, 45 ML-scored >=5, several at the 10.0
# ceiling (the urgency head over-scores them because the title is structured
# data dense with held tickers and "Bullish:"/percent figures — pure model
# artefact). The briefing's per-domain cap admits up to 6 of them into the
# 50-row top pool every cycle, displacing real news in TOP SIGNALS. Same
# byte-identical lockstep-duplication discipline as the four quote-widget
# fingerprints above (anti-import-cycle: the analysis layer must not pull the
# watchers+ml_features graph). Validated zero false positives on the live
# headline corpus — no real news headline leads with this bracketed marker.
_QW_STOCKTWITS_SENTIMENT = re.compile(
    r"^\s*\[StockTwits\s+Sentiment\]\s+[A-Z]"
)
# Image-credit pseudo-article — lockstep mirror of the sixth fingerprint added
# to ``watchers.alert_agent._QW_IMAGE_CREDIT`` and
# ``collectors.web_scraper._QW_IMAGE_CREDIT``. Live evidence (2026-05-21
# 16:30:49Z, alert_recency.db): "Angela Weiss/AFP/Getty Images" fired a real
# 🚨 BREAKING push from ``scraped/www.bloomberg.com`` (cred=0.90 — above the
# 0.45 lone-source bar so the authority gate cannot catch it; content type IS
# the failure). The bug: news pages wrap the hero image inside the article's
# own <a> link, so the web scraper's anchor-text fallback picks up the photo
# credit line beneath the image as the article title. The ML urgency head
# scored it 10.0 (bloomberg.com URL + proper-noun tokens triggered the
# high-relevance pattern recognition). The briefing path — the analyst's
# PRIMARY consumed product — would have surfaced this credit string as a
# TOP SIGNAL had the urgency cascade not also pushed it. Same triple-gate
# discipline as the prior five quote-widget fingerprints: every consumed
# product gets the same defense-in-depth gate, byte-identical regex across
# the three modules (anti-import-cycle: the analysis layer must not pull
# the watchers+ml_features graph and the watchers layer must not pull the
# collectors/aiohttp graph).
_QW_IMAGE_CREDIT = re.compile(
    r"^\s*[A-Z][a-zA-Z]+(?:\s+(?:[A-Z]\.?|[A-Z][a-zA-Z]+))+"
    r"(?:/(?:AFP|Reuters|Getty\s+Images|AP|Bloomberg|EPA|TASS|"
    r"WireImage|Shutterstock|Polaris|Bloomberg\s+News))+"
    r"\s*$"
)


def _looks_like_quote_widget(article: dict) -> bool:
    """True for a live quote-tape / quote-listing / structured-data-summary /
    image-credit entry masquerading as a digest article.

    Six independent title fingerprints (a letter glued directly to a decimal
    price; a parenthesised signed % change; a "$NAME (SYMBOL.EXCH)$" share-card
    listing page; a ``[YF/<bucket>]`` screener-tape lead from
    ``market_movers``; a ``[StockTwits Sentiment]`` extreme-sentiment summary
    row from ``stocktwits_sentiment``; a ``Photographer Name/Agency/Getty
    Images`` photo credit the web scraper picked up as a title) plus a Yahoo
    /quote/ landing path. All anchored so real prose with $/%/comma numbers
    ("rises 22% to $35.1 billion", "5,123.41 record high"), real "$TICKER ..."
    headlines, real quote-scoped article URLs, and real headlines containing
    agency names ("Reuters/Yahoo Finance reports") are never caught.
    Byte-identical logic to watchers.alert_agent._looks_like_quote_widget.
    The synthetic PORTFOLIO/OPTIONS snapshot rows the daemon prepends
    ("PORTFOLIO P&L SNAPSHOT" / "OPTIONS SNAPSHOT", no url) never match any
    fingerprint, so they always pass through untouched."""
    title = article.get("title") or ""
    if (_QW_PRICE_GLUE.search(title) or _QW_PCT_PAREN.search(title)
            or _QW_LISTING.search(title) or _QW_SCREENER_TAPE.search(title)
            or _QW_STOCKTWITS_SENTIMENT.search(title)
            or _QW_IMAGE_CREDIT.search(title)):
        return True
    url = article.get("link") or article.get("url") or ""
    try:
        if _QW_QUOTE_PATH.search(urlparse(url).path):
            return True
    except Exception:
        pass
    return False


def _filter_quote_widget_noise(articles: list) -> tuple[list, list]:
    """Partition digest rows into ``(kept, suppressed)``; ``suppressed`` is the
    quote-tape pseudo-articles. Pure, order-preserving, side-effect-free —
    returns NEW lists and never mutates the caller's ``source_articles`` (which
    heartbeat_worker feeds to the briefing-label / training path), so all four
    load-bearing invariants (backtest isolation, ml_score≠ai_score,
    score_source, urgency state machine) are intact by construction: this only
    ever reshapes the text Opus reads."""
    kept: list = []
    suppressed: list = []
    for a in articles:
        (suppressed if _looks_like_quote_widget(a) else kept).append(a)
    return kept, suppressed


# ── Recap / SEO template gate (defense-in-depth, briefing path) ──────────────
# A *second* over-scored content class the urgency head over-weights: the
# *recap / preview / transcript-summary* template — content that is inherently
# retrospective ("trading up TODAY", "Q1 Earnings Call Highlights", a date-
# stamped "Stock Market Today, May 18:" wrap-up) or algorithmic press-mill
# output ("(LITE) Shares Fall 8.8% -- GF Value Says ..."). The alert path
# already gates these via watchers.alert_agent._filter_recap_template_noise
# (tests/test_alert_recap_template.py), but the briefing path — the analyst's
# PRIMARY consumed product — did not. Live evidence (2026-05-19 04:18Z
# heartbeat): "[00:50] 9.85 MU Motley: why MU dropped (cont., ~3.5h old)"
# made it into TOP SIGNALS — a model self-prediction at 9.85 on the exact
# "Why Did Micron Stock Drop Today ? | The Motley Fool" recap title. The
# 6-hour articles.db scan also surfaced six other ai_score>=7 recap rows in
# the same window (LITE GF Value, AXTI GF Value, QBTS Q1 Earnings Call
# Highlights ×2, "Stock Market Today, May 18: ...", "Why Nvidia Stock Is
# Trading Up Today"). These are exactly the rows
# watchers.alert_agent._filter_recap_template_noise drops on the standalone
# push.
#
# Same shape as ``_filter_quote_widget_noise``: a pure read-side text drop at
# the single chokepoint the briefing funnels through, NOT an ML-threshold
# change and NOT a DB write. Runs BEFORE dedup so a recap syndicated across
# multiple feeds (live: the "Stock Market Today, May 18: ..." wrap-up carried
# by Motley Fool + Nasdaq + YahooFinance) is caught on every copy. Returns
# NEW lists, never mutates the caller's source_articles (heartbeat_worker
# feeds those onward to the briefing-label / training path) — so all four
# load-bearing invariants (backtest isolation, ml_score≠ai_score,
# score_source, urgency state machine) are intact by construction.
#
# Patterns and lockstep discipline are duplicated from
# watchers.alert_agent._RECAP_TEMPLATE_PATTERNS rather than cross-imported:
# the analysis layer must not pull the watchers+ml_features import graph
# (same documented anti-import-cycle discipline as _collapse_syndicated
# reusing alert_dedup._signature and article_store._briefing_domain_key
# duplicating ml.features). Test parity (tests/test_briefing_recap_template
# vs tests/test_alert_recap_template) pins the two gates against drift.

# "Why <Co|TICKER> Stock Is Trading {Up|Down|Higher|Lower} Today" — Zacks /
# Yahoo / Finnhub recap. Anchored "^Why" so a headline that uses "why" mid-
# sentence is unaffected. "Stock Is Trading" + "Today" is the discriminator.
_BRIEFING_RT_WHY_TRADING = re.compile(
    r"^\s*why\s+.+?\s+stock\s+is\s+trading\s+(?:up|down|higher|lower)\s+today\b",
    re.IGNORECASE,
)
# "Why Did <X> Stock {Drop|Rise|Surge|Fall|Climb|Plunge|Soar|Jump|Tumble} Today"
# — Motley Fool / Zacks past-tense intraday-move recap.
_BRIEFING_RT_WHY_DID = re.compile(
    r"^\s*why\s+did\s+.+?\s+stock\s+"
    r"(?:drop|rise|surge|fall|climb|plunge|soar|jump|tumble)\b",
    re.IGNORECASE,
)
# "Stock Market Today, May 18: ..." — date-stamped daily market wrap-up.
_BRIEFING_RT_MARKET_TODAY = re.compile(
    r"^\s*stock\s+market\s+today\s*[,:]\s*"
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)\s+\d{1,2}\b",
    re.IGNORECASE,
)
# "Q1 2026 Earnings Call Highlights" / "Q1 Earnings Call Highlights" /
# "Q1 2027 Earnings Transcript" — GuruFocus / Seeking Alpha / Globe-and-Mail
# transcript-summary template (substring — appears mid-headline). Year and
# the "call " bridge are BOTH optional — lockstep with the relaxed alert-side
# ``_RT_EARNINGS_CALL`` (2026-05-20 NVDA earnings-day evidence: the briefing
# layer's stricter prior form missed "NVIDIA Q1 Earnings Call Highlights" (no
# year) and "Nvidia (NVDA) Q1 2027 Earnings Transcript" (no "Call"), and a
# 7-day live audit found ~700 such rows reaching ml_score 8+ that would have
# entered the top-50 digest pool. The discriminator stays the recap-noun list
# ``highlights|recap|takeaways|transcript|summary|review`` so forward-looking
# titles ("Q1 Earnings Preview", "earnings call begins at 5pm ET") never
# match. Byte-identical to ``watchers.alert_agent._RT_EARNINGS_CALL``.
_BRIEFING_RT_EARNINGS_CALL = re.compile(
    r"\bq[1-4](?:\s*(?:fy\s*)?20\d{2})?\s+earnings\s+(?:call\s+)?"
    r"(?:highlights|recap|takeaways|transcript|summary|review)\b",
    re.IGNORECASE,
)
# "Here['s|is] What the Street Thinks About <X>" — InsiderMonkey opinion-mill.
_BRIEFING_RT_STREET_THINKS = re.compile(
    r"^\s*here(?:'?s|\s+is)?\s+what\s+the\s+street\s+thinks\b",
    re.IGNORECASE,
)
# "(TICKER) Shares Fall 8.8% -- GF Value Says ..." — GuruFocus algorithmic
# press-mill. "GF Value Says" is unique enough to be a high-precision pattern.
_BRIEFING_RT_GF_VALUE = re.compile(
    r"\bgf\s+value\s+says\b",
    re.IGNORECASE,
)
# "<Company> Earnings: A Quick Glance at Key Metrics" — Zacks post-earnings
# recap-mill template (lockstep mirror of watchers.alert_agent._RT_QUICK_GLANCE).
# Live evidence (2026-05-21 NVDA earnings night): "NVIDIA Earnings: A Quick
# Glance at Key Metrics" reached urgency=2 with ml_score 9.9 — so it scores
# straight into the briefing's top-50 pool and would surface as a fresh TOP
# SIGNAL despite being a retrospective post-print summary. Same earnings-recap
# class as `_BRIEFING_RT_EARNINGS_CALL`; substring (not anchored) since the
# phrase follows the "<Company> Earnings:" lead.
_BRIEFING_RT_QUICK_GLANCE = re.compile(
    r"\ba\s+quick\s+glance\s+at\s+(?:key\s+)?(?:financial\s+)?metrics\b",
    re.IGNORECASE,
)
# "<headline>. Here's What Happened" — Motley Fool / MarketBeat retrospective
# tail; lockstep mirror of watchers.alert_agent._RT_HERES_WHAT_HAPPENED. Live
# evidence (2026-05-23): "Nvidia Just Crushed Earnings Estimates, but the
# Stock Fell. Here's What Happened" syndicated 6×, ml_score 9.22-9.41 — would
# enter the briefing's top-50 digest pool as a fresh TOP SIGNAL despite being
# a retrospective post-event explainer. Three apostrophe forms covered;
# past-tense "happened" required so present-continuous "Here's What's
# Happening" market wraps are NOT matched.
_BRIEFING_RT_HERES_WHAT_HAPPENED = re.compile(
    r"\bhere(?:[s'’]+|\s+is)?\s+what\s+happened\b",
    re.IGNORECASE,
)
# "[Wikipedia] <article title>" — lockstep mirror of
# ``watchers.alert_agent._RT_WIKIPEDIA_REF``. The ``collectors.wikipedia_collector``
# emits encyclopedic recent-changes rows with this exact prefix, and the ML
# urgency head over-scores them because the (often ticker-shaped) title plus
# semis-keyword summary tokens trip its high-relevance pattern recognition.
# Live evidence (2026-05-23, 7-day articles.db scan): ``[Wikipedia] DRAM
# (musician)`` at ml_score=10.0 (musician disambiguation page, not semis) and
# ``[Wikipedia] Nvidia RTX`` at ml_score=8.6 (long-standing reference page)
# both reached urgency=2, so both would score straight into the briefing's
# top-50 pool and could surface as fresh TOP SIGNALS despite being
# encyclopedic reference content, not news. The sibling
# ``collectors.wikipedia_pageviews`` collector — which IS a useful predictive
# signal — emits titles like ``"Wiki pageview SURGE NVDA (NVIDIA_Corporation):
# ..."`` without the leading bracketed-source tag, so its rows are NOT caught.
_BRIEFING_RT_WIKIPEDIA_REF = re.compile(
    r"^\s*\[Wikipedia\]\s+",
)

# Drift-closure patterns — lockstep mirrors of the seven recap fingerprints the
# alert path (``watchers.alert_agent``) added between 2026-05-19 and 2026-05-21
# that the briefing layer was never updated with. Each regex is byte-identical
# to its alert-side twin (same anti-import-cycle duplication discipline as the
# original briefing-side mirrors above). The drift was real and recently
# evidenced: a 7-day live audit found 26 ``why_just_moved``, 35 ``why_pct_after``,
# 74 ``todays_movers_list``, 91 ``is_buy_after``, 49 ``earnings_tomorrow_preview``,
# 105 ``why_is_pct_since``, and 4 ``why_stock_is_after`` rows that the urgency
# scorer pre-floored for the alert path (so their ai_score=0.01) but whose
# ml_score (often 9+) still made COALESCE(ai_score, ml_score) push them into
# the briefing's top-50 digest. The briefing — the analyst's PRIMARY consumed
# product — was missing every one of these gates.

# "Why <X> Stock {just|now|today|finally|...} {popped|surged|...}" — Motley
# Fool variant where the subject moves past-tense without "Did" between Why
# and the subject. Live failure: "Why Micron Stock Just Popped Again" was
# Sonnet-scored urgent and fired a real Discord push.
_BRIEFING_RT_WHY_JUST_MOVED = re.compile(
    r"^\s*why\s+.+?\s+stock\s+"
    r"(?:just|now|today|finally|suddenly|then|recently|already)\s+"
    r"(?:popped|surged|jumped|soared|crashed|tumbled|plunged|sank|fell|"
    r"dropped|climbed|spiked|slid|slipped|rallied|tanked|plummeted|"
    r"nosedived|hammered|skyrocketed|rocketed|rebounded)\b",
    re.IGNORECASE,
)
# "Why Is <X> {up|down|higher|lower} N.N% Since Last <event>" — Zacks /
# Seeking Alpha post-event price-attribution. Live failure: "Why Is AGNC
# Investment (AGNC) Down 7.2% Since Last Earnings Report?".
_BRIEFING_RT_WHY_IS_PCT_SINCE = re.compile(
    r"^\s*why\s+is\s+.+?\s+(?:up|down|higher|lower)\s+"
    r"\d+(?:\.\d+)?\s*%\s+since\b",
    re.IGNORECASE,
)
# "Why <X> Stock Is <state-verb> After <event>" — Barron's / MSN / Yahoo
# post-event explainer. Live failure: "Why Nvidia Stock Is Barely Moving
# After Earnings Crushed Expectations" reached urgency=2 with ml_score 9.97
# and would have surfaced in the briefing as a TOP SIGNAL despite being
# retrospective.
_BRIEFING_RT_WHY_STOCK_IS_AFTER = re.compile(
    r"^\s*why\s+.+?\s+stock\s+is\s+"
    r"(?:still\s+|barely\s+|now\s+|finally\s+|just\s+|currently\s+|"
    r"actually\s+|suddenly\s+|hardly\s+|so\s+|really\s+)?"
    r"(?:moving|trading|sliding|sinking|tumbling|crashing|plunging|"
    r"jumping|surging|soaring|rising|falling|climbing|dropping|"
    r"rallying|spiking|tanking|skyrocketing|nosediving|"
    r"up|down|higher|lower|flat|stuck|red|green|bid|offered)"
    r"\b.*?"
    r"\bafter\b.*?\b"
    r"(?:earnings|results|report|quarter|q[1-4]|beat|miss|guidance)\b",
    re.IGNORECASE,
)
# "Why <X> Is <up|down|...> N.N% After <event>" — Zacks / StockStory variant
# lacking the "Stock" token. Live failure: "Why AXT (AXTI) Is Down 14.2%
# After Betting Big On AI-Focused Indium Phosphide Expansion" syndicated
# heavily; "Why Tower Semiconductor (TSEM) Is Up 29.8% After ..." too.
_BRIEFING_RT_WHY_PCT_AFTER = re.compile(
    r"^\s*why\s+.+?\s+(?:is|are|was|were)\s+"
    r"(?:up|down|higher|lower)\s+"
    r"\d+(?:\.\d+)?\s*%\s+"
    r"after\b",
    re.IGNORECASE,
)
# "<X> Reports Earnings Tomorrow: What To Expect" — FinancialContent /
# StockStory / MSN / TradingView SEO-mill earnings-preview. By definition
# NOT breaking ("tomorrow"). Live failure: 49 such rows in 7 days, several
# with ml_score 8+.
_BRIEFING_RT_EARNINGS_TOMORROW = re.compile(
    r"\breports?\s+earnings\s+tomorrow\s*:\s*what\s+to\s+expect\b",
    re.IGNORECASE,
)
# "These Stocks Are Today's Movers: Nvidia, Micron, ..." — Barron's daily
# column heavily syndicated. Live failure: 74 such rows in 7 days, the
# Marketwatch + Finnhub/Yahoo copies ml_score 9.29 and 4.6.
_BRIEFING_RT_TODAYS_MOVERS = re.compile(
    r"^\s*these\s+stocks\s+are\s+today['’]?s\s+(?:top\s+|biggest\s+)?movers\s*:",
    re.IGNORECASE,
)
# "Is <X> a Buy After <Earnings|Q1|Results|...>" — Motley Fool / Yahoo /
# TipRanks post-event valuation question. Live failure: 91 such rows in
# 7 days; "Is Nvidia a Buy After Their Latest Earnings Report?" fired a
# real BREAKING push on the alert side.
_BRIEFING_RT_IS_BUY_AFTER = re.compile(
    r"^\s*(?:\S+\s+)?is\s+(?:\w+\s+){0,2}a\s+(?:buy|sell|hold)\b.*?"
    r"\bafter\b.*?\b(?:earnings|results|report|quarter|q[1-4])\b",
    re.IGNORECASE,
)

# Algorithmic-mill v2 fingerprints — lockstep mirror of watchers.alert_agent
# additions (2026-05-23 live audit). Briefing layer MUST carry the same set or
# the per-domain cap admits these into the TOP SIGNALS pool. Anti-drift test
# `test_alert_and_briefing_recap_tuples_have_same_length` enforces this
# structurally. Same byte-identical regexes as the alert side — see
# alert_agent._RT_HOLDINGS_BY_FUND / _RT_SHARES_BOUGHT_BY /
# _RT_FUTURES_WHY_TODAY / _RT_DAILY_PRICE_CITY for full live evidence.
_BRIEFING_RT_HOLDINGS_BY_FUND = re.compile(
    r"\bholdings\s+(?:raised|cut|lowered|increased|trimmed|boosted|reduced|"
    r"decreased|sold|acquired)\s+by\s+\S+(?:\s+\S+){0,5}\s+LLC\b",
    re.IGNORECASE,
)
_BRIEFING_RT_SHARES_BOUGHT_BY = re.compile(
    r"\bshares\s+(?:in|of)\s+\S+(?:\s+\S+){0,5}\s+"
    r"(?:bought|sold|acquired|disposed|purchased)\s+by\s+"
    r"\S+(?:\s+\S+){0,5}\s+LLC\b",
    re.IGNORECASE,
)
_BRIEFING_RT_FUTURES_WHY_TODAY = re.compile(
    r"^\s*why\s+are\s+stock\s+market\s+futures\s+"
    r"(?:up|down|higher|lower|mixed|moving|moved|sliding|rising|falling)\s+"
    r"today\b",
    re.IGNORECASE,
)
_BRIEFING_RT_DAILY_PRICE_CITY = re.compile(
    r"^\s*(?:gold|silver|petrol|diesel|crude\s+oil)\s+(?:rate|price)\s+"
    r"today\s+in\s+\S",
    re.IGNORECASE,
)

_BRIEFING_RECAP_TEMPLATE_PATTERNS = (
    ("why_trading_today", _BRIEFING_RT_WHY_TRADING),
    ("why_did_stock", _BRIEFING_RT_WHY_DID),
    ("why_just_moved", _BRIEFING_RT_WHY_JUST_MOVED),
    ("why_is_pct_since", _BRIEFING_RT_WHY_IS_PCT_SINCE),
    # ``why_stock_is_after`` is the strictly more-specific sibling of
    # ``why_pct_after`` (the title has a ``Stock`` token AND a state verb AND
    # an earnings-noun terminator) — must run first so a title like
    # "Why NVDA Stock Is Down 3% After Q1 Earnings" gets the more-precise
    # fingerprint name (mirrors the alert-side ordering).
    ("why_stock_is_after", _BRIEFING_RT_WHY_STOCK_IS_AFTER),
    ("why_pct_after", _BRIEFING_RT_WHY_PCT_AFTER),
    ("market_today_dated", _BRIEFING_RT_MARKET_TODAY),
    ("earnings_call_recap", _BRIEFING_RT_EARNINGS_CALL),
    ("quick_glance_metrics", _BRIEFING_RT_QUICK_GLANCE),
    ("heres_what_happened", _BRIEFING_RT_HERES_WHAT_HAPPENED),
    ("wikipedia_ref", _BRIEFING_RT_WIKIPEDIA_REF),
    ("earnings_tomorrow_preview", _BRIEFING_RT_EARNINGS_TOMORROW),
    ("todays_movers_list", _BRIEFING_RT_TODAYS_MOVERS),
    ("is_buy_after", _BRIEFING_RT_IS_BUY_AFTER),
    ("street_thinks", _BRIEFING_RT_STREET_THINKS),
    ("gf_value_says", _BRIEFING_RT_GF_VALUE),
    ("holdings_by_fund", _BRIEFING_RT_HOLDINGS_BY_FUND),
    ("shares_bought_by", _BRIEFING_RT_SHARES_BOUGHT_BY),
    ("futures_why_today", _BRIEFING_RT_FUTURES_WHY_TODAY),
    ("daily_price_city", _BRIEFING_RT_DAILY_PRICE_CITY),
)


def _looks_like_recap_template(article: dict) -> tuple[bool, str]:
    """``(True, fingerprint_name)`` for a recap/preview/transcript-summary or
    algorithmic-mill title — these are inherently retrospective and must not
    surface as a fresh TOP SIGNAL in the Opus heartbeat digest. ``(False, "")``
    for everything else.

    Pure, side-effect-free; reads only ``title``. Six independent fingerprints,
    all anchored so real breaking headlines are NEVER caught (the alert-path
    test_alert_recap_template.py must-survive corpus is verified verbatim
    against this gate too — "Nvidia Q3 revenue rises 22%...", "Fed cuts
    rates", "MU earnings blow past estimates", "Why investors are bullish on
    Nvidia", "MU shares halted", "Nvidia Q1 earnings preview")."""
    title = (article.get("title") or "").strip()
    if not title:
        return False, ""
    for name, pat in _BRIEFING_RECAP_TEMPLATE_PATTERNS:
        if pat.search(title):
            return True, name
    return False, ""


def _filter_recap_template_noise(articles: list) -> tuple[list, list]:
    """Partition digest rows into ``(kept, suppressed)``; ``suppressed`` is the
    recap/preview/transcript-summary/algorithmic-mill rows the urgency head
    over-scored. Pure, order-preserving, side-effect-free — returns NEW lists
    and never mutates the caller's ``source_articles`` (heartbeat_worker feeds
    those onward to the briefing-label / training path), so all four
    load-bearing invariants (backtest isolation, ml_score≠ai_score,
    score_source, urgency state machine) are intact by construction: this only
    ever reshapes the text Opus reads.

    Runs BEFORE ``_collapse_syndicated`` so a recap syndicated across multiple
    feeds (live: the "Stock Market Today, May 18: ..." wrap-up carried by
    Motley Fool + Nasdaq + YahooFinance) is suppressed on every copy and the
    dedup layer is never asked to discriminate against a real story with a
    similar prefix. Suppressed rows are tagged with ``_recap_fingerprint`` on a
    *defensive shallow copy* so the caller's row is never mutated (mirrors
    ``alert_agent._filter_recap_template_noise``'s mutation discipline)."""
    kept: list = []
    suppressed: list = []
    for a in articles:
        hit, name = _looks_like_recap_template(a)
        if hit:
            tagged = dict(a)
            tagged["_recap_fingerprint"] = name
            suppressed.append(tagged)
        else:
            kept.append(a)
    return kept, suppressed


# ── Held-book relevance tag (the analyst's open positions) ───────────────────
# The 5h Opus digest tells the analyst what is *important*, but never which
# rows touch money they actually have at risk RIGHT NOW. The Discord-only
# ``daemon._format_portfolio_coverage`` line is appended AFTER the briefing —
# Opus never sees it while composing the LEAD / TOP SIGNALS / PORTFOLIO table,
# so a held-position story scoring 8.0 is ranked identically to an 8.0 generic
# macro item. For an analyst whose persona is "I depend on this to react to
# events affecting MY positions", that is a real prioritisation miss. This
# surfaces the held tickers in the newswire text Opus reads, exactly like the
# established ``[syndicated xN]`` / ``[model]`` / ``[ALERTED]`` tags: a pure
# read-side annotation — no DB write, no ai_score/ml_score/score_source/urgency
# touch, no row mutation, backtest already excluded upstream by
# get_top_for_briefing's _LIVE_ONLY_CLAUSE — all four load-bearing invariants
# intact by construction.
#
# _BOOK_TICKERS is mirrored verbatim from daemon.PORTFOLIO_TICKERS (positions +
# sector watchlist). Kept a local literal rather than importing daemon — the
# analysis layer must not pull the daemon/collectors import graph (same
# documented anti-import-cycle discipline as _collapse_syndicated reusing
# alert_dedup._signature and article_store._briefing_domain_key duplicating
# ml.features). Parity with the daemon source is pinned by
# tests/test_briefing_book_tag.py so the two can never silently drift.
_BOOK_TICKERS: tuple[str, ...] = (
    "LITE", "LNOK", "MUU", "DRAM", "SNDU",
    "MU", "MSFT", "AXTI", "ORCL", "TSEM", "QBTS", "NVDA",
)
# Live held/watched universe — same single source of truth the urgency
# SCORE_PROMPT and ml.features portfolio-relevance features already use
# (config/portfolio.json's positions + option underlyings + sector_watchlist,
# unioned with the hardcoded fallback). The static ``_BOOK_TICKERS`` literal
# alone was silently drifting behind the trading UI: a 2026-05-23 live read
# showed GOOG / COHR / NVDL held in portfolio.json yet absent from the static
# tuple, so briefing rows mentioning those open positions never received the
# ``[BOOK:]`` tag (Opus had no signal these touched the analyst's book) and
# they could never light up ``_book_heat_lines``' concentration signal.
# _BOOK_UNIVERSE is the UNION — static first in their canonical order, then
# live-only additions in deterministic alphabetical order. Anti-drift parity
# of _BOOK_TICKERS with daemon.PORTFOLIO_TICKERS is unchanged (and still
# pinned by test_briefing_book_tag.py); the universe only EXTENDS the matching
# set, it never shrinks or reorders the static core.
from ml.features import LIVE_PORTFOLIO_TICKERS as _LIVE_PORTFOLIO_TICKERS
_BOOK_UNIVERSE: tuple[str, ...] = _BOOK_TICKERS + tuple(
    sorted(set(_LIVE_PORTFOLIO_TICKERS) - set(_BOOK_TICKERS))
)
# Longest-first alternation so the regex prefers \bMUU\b over \bMU\b and the
# word boundaries keep "MU" from matching inside "Micron"/"MUSEUM" — the exact
# case-sensitive convention daemon._format_portfolio_coverage and
# ml.features._LIVE_RE already use (financial copy writes tickers uppercase).
_BOOK_RE = re.compile(
    r"\b(?:"
    + "|".join(re.escape(t) for t in sorted(set(_BOOK_UNIVERSE),
                                             key=len, reverse=True))
    + r")\b"
)


def _book_tickers(article: dict) -> list[str]:
    """Held/watchlist tickers mentioned in a digest row's title+summary.

    Returns them in canonical ``_BOOK_UNIVERSE`` order (static
    ``_BOOK_TICKERS`` first in their existing order — stable cycle-to-cycle —
    then live-only additions from config/portfolio.json in deterministic
    alphabetical order). De-duplicated. Empty list when the row touches no
    held name. Pure and side-effect-free — reads only ``title``/``summary``
    via .get(), never mutates the article (heartbeat_worker feeds the same
    dicts onward to the briefing-label / training path, so read-only is
    load-bearing)."""
    blob = f"{article.get('title') or ''} {article.get('summary') or ''}"
    if not blob.strip():
        return []
    hits = set(_BOOK_RE.findall(blob))
    if not hits:
        return []
    return [t for t in _BOOK_UNIVERSE if t in hits]


# ── Held-book concentration ("BOOK HEAT") ────────────────────────────────────
# The per-row [BOOK: ...] tag tells Opus WHICH rows touch the analyst's open
# positions, but never that a single held name is the window's centre of
# gravity. A held ticker carried by ONE story scoring 7 may not lead; the SAME
# ticker spread across 6 *distinct* (post-syndication-collapse) stories in the
# 5h window is a magnitude signal in its own right — concentration the analyst
# persona ("I depend on this to react to events affecting MY positions") needs
# surfaced, yet Opus cannot infer it from per-row tags alone (it would have to
# mentally tally 60 rows). This emits it as ONE ranked input hint, the exact
# pure read-side shape of the [syndicated xN] / [BOOK:] tags: no DB write, no
# ai_score/ml_score/score_source/urgency touch, no row mutation, backtest
# excluded upstream by get_top_for_briefing's _LIVE_ONLY_CLAUSE, returns a NEW
# list and never mutates source_articles — all four load-bearing invariants
# intact by construction.
#
# Counted over the ALREADY-collapsed+capped digest the model actually reads, so
# "N distinct stories" is honest (6 syndicated copies of one wire = 1 story,
# not 6 — reuses _collapse_syndicated's output, never the raw pre-dedup list)
# and verifiable against the rendered newswire. The same real-url snapshot
# guard as [BOOK:] excludes the prepended PORTFOLIO/OPTIONS rows (whose P&L
# body legitimately lists held tickers). Conservative threshold (>=3) keeps
# this signal, not noise — the analyst's top complaint.
BOOK_HEAT_MIN_STORIES = 3
_BOOK_HEAT_MAX_LINES = 6


def _book_heat_lines(
    articles: list, min_stories: int = BOOK_HEAT_MIN_STORIES
) -> list[str]:
    """Pure: rank held names by how many DISTINCT digest rows mention them.

    ``articles`` must be the post-``_collapse_syndicated``, post-cap list the
    newswire actually renders (so syndicated copies of one event count once).
    A row with no real ``link``/``url`` (the prepended PORTFOLIO/OPTIONS
    snapshots) is skipped — identical guard to the ``[BOOK:]`` tag, so a
    snapshot P&L body listing held tickers can never manufacture phantom heat.
    Returns ``["MU — 6 distinct stories", ...]`` for tickers at/above
    ``min_stories``, ordered by count desc then canonical ``_BOOK_TICKERS``
    order (stable cycle-to-cycle, same tie-break discipline as
    ``_book_tickers``), capped at ``_BOOK_HEAT_MAX_LINES``. ``[]`` when nothing
    is concentrated. No DB / IO / mutation."""
    counts: dict[str, int] = {}
    for a in articles:
        if not (a.get("link") or a.get("url")):
            continue  # snapshot/synthetic row — same guard as the [BOOK:] tag
        for t in _book_tickers(a):
            counts[t] = counts.get(t, 0) + 1
    # Rank over the full universe (static core + live additions) so a heat-only
    # live-added position (GOOG / COHR / NVDL via config/portfolio.json) gets a
    # deterministic tie-break position rather than collapsing to the
    # ``len(rank)`` fallback below all static names regardless of mention order.
    rank = {t: i for i, t in enumerate(_BOOK_UNIVERSE)}
    hot = sorted(
        ((t, n) for t, n in counts.items() if n >= min_stories),
        key=lambda tn: (-tn[1], rank.get(tn[0], len(rank))),
    )
    return [f"{t} — {n} distinct stories"
            for t, n in hot[:_BOOK_HEAT_MAX_LINES]]


# ── Held-book SILENCE (the inverse of BOOK HEAT) ─────────────────────────────
# BOOK HEAT surfaces held names CONCENTRATED in the digest; this surfaces the
# opposite — held names with ZERO mentions this 5h window. The
# Discord-post-briefing ``daemon._format_portfolio_coverage`` line already
# names silent tickers, but it is appended AFTER Opus has written the
# briefing — Opus composes the LEAD / TOP SIGNALS / PORTFOLIO table BLIND to
# which held names had no story, and routinely fabricates a "neutral
# implication" for a dark ticker (live: a recent briefing's PORTFOLIO line
# wrote "AXTI: continued caution given thin coverage" — a fabrication on
# zero wires, the analyst persona's exact complaint about hedging filler).
#
# Surfacing the silent set as an INPUT block lets Opus mark these honestly
# (N/A — no catalyst this window) rather than guess. The silent list is
# itself a magnitude signal of its own kind: a held name with zero wires
# in 5h means the catalyst engine isn't running, so the position drifts on
# pure macro/peers. Different signal class from BOOK HEAT (concentration);
# complementary, not duplicative.
#
# Pure read-side, same shape as ``_book_heat_lines`` (input hint, NOT a
# reproduced section — the SYSTEM_PROMPT rule forbids echoing it): NO DB
# write, NO ai_score/ml_score/score_source/urgency touch, NO mutation of
# ``source_articles``, backtest already excluded upstream by
# ``get_top_for_briefing``'s ``_LIVE_ONLY_CLAUSE`` — all four load-bearing
# invariants intact by construction.
#
# Honest "absence" is computed against the *full* held set (`_BOOK_TICKERS`
# — 12 names), so the silent line is comparable cycle-to-cycle even when
# the digest is sparse. The 3-ticker floor mirrors `BOOK_HEAT_MIN_STORIES`
# (a 1-2 ticker silent list is the normal case for a busy macro window and
# would just be filler); we want a silent list big enough that PORTFOLIO
# composition is materially constrained.
BOOK_SILENCE_MIN_SILENT = 3


def _book_silence_lines(
    articles: list, min_silent: int = BOOK_SILENCE_MIN_SILENT
) -> list[str]:
    """Pure: list held tickers with ZERO mentions in the digest.

    ``articles`` must be the post-``_collapse_syndicated``, post-cap list the
    newswire actually renders — same input as ``_book_heat_lines`` so the two
    blocks describe the SAME window. The url guard (skip rows with no real
    ``link``/``url``) excludes the prepended PORTFOLIO/OPTIONS snapshot rows
    so a snapshot P&L body listing held tickers can never falsely "cover" a
    silent name (same guard as ``_book_heat_lines`` / the ``[BOOK:]`` tag).

    Returns ``[]`` (silence, no line) when:
      - the digest is empty (no held name can be honestly called silent);
      - fewer than ``min_silent`` held names are silent — a 1-2 ticker silent
        list is noise in a normal macro window (the analyst's #1 complaint
        is noise, so the bar is conservative).

    Otherwise: ONE compact line listing the silent tickers in canonical
    ``_BOOK_TICKERS`` order (stable cycle-to-cycle, same tie-break discipline
    as ``_book_tickers`` / ``_book_heat_lines``). Same shape as the
    BOOK HEAT block — a single input hint Opus uses for PORTFOLIO
    composition, NOT a section reproduced in the output. No DB / IO /
    mutation by construction."""
    if not articles:
        return []
    seen: set[str] = set()
    for a in articles:
        if not (a.get("link") or a.get("url")):
            continue  # snapshot/synthetic row — same guard as the [BOOK:] tag
        for t in _book_tickers(a):
            seen.add(t)
    silent = [t for t in _BOOK_TICKERS if t not in seen]
    if len(silent) < min_silent:
        return []
    return [" ".join(silent)]


# ── Aging-top-row recency cross-check ────────────────────────────────────────
# A news analyst whose persona is "react to BREAKING events fast" is misled
# worst by a stale story dressed as fresh. The model-estimated time_sensitivity
# decay rerank (_rank_by_decayed_score) already demotes stale time-bound rows —
# but only as far as the model's ts head scored them: a row the ts head
# under-scored stays time-bound yet barely decays, and in a sparse 5h window an
# already-decayed 5-6h-old item can still float to #1. Opus then has only the
# per-row [seen HH:MM UTC] absolute clock and the BRIEFING TIME header, and LLM
# clock subtraction across a bare-HH:MM 24h window is unreliable — so it can
# write a multi-hour-old developing story into the LEAD as if it just broke
# (the recurring duplicate/stale-framing complaint, on the analyst's primary
# product). This surfaces a DETERMINISTIC wall-clock age for the rows Opus
# actually leads with — an independent ground-truth cross-check on the
# model-estimated decay, NOT a re-expression of it. Same pure read-side,
# BOOK-HEAT-shaped contract (separate input block, never a per-row token, never
# echoed): no DB write, no ai_score/ml_score/score_source/urgency touch, no row
# mutation, backtest excluded upstream by get_top_for_briefing's
# _LIVE_ONLY_CLAUSE, returns a NEW list and never mutates source_articles —
# all four load-bearing invariants intact by construction. The wall-clock age
# reuses _seen_age_hours verbatim (the file's own wire-age primitive — the
# documented anti-drift discipline, same reason _collapse_syndicated reuses
# alert_dedup._signature).
#
# 3.0h mirrors the alert path's documented "materially old (≳3h)" RECENCY
# threshold (ALERT_PROMPT) so the two consumed products judge "stale" the same
# way. Only the first _AGING_TOP_SCAN rows are considered — Opus draws the
# LEAD / TOP SIGNALS from the very top, so flagging that row #40 is old is
# noise, not signal (the analyst's #1 complaint). Capped like _book_heat_lines.
BRIEFING_AGING_MIN_HOURS = 3.0
_AGING_TOP_SCAN = 10
_AGING_MAX_LINES = 6


def _aging_top_rows(
    articles: list, now: datetime | None = None,
    min_age_h: float = BRIEFING_AGING_MIN_HOURS,
) -> list[str]:
    """Pure: of the highest-ranked digest rows, which are several hours old.

    ``articles`` must be the post-collapse/dedup/decay/cap list Opus actually
    reads (``deduped[:60]``) so "rank #N" matches the rendered newswire. A row
    with no real ``link``/``url`` (the prepended PORTFOLIO/OPTIONS snapshots) is
    skipped — identical guard to the ``[BOOK:]`` tag / ``_book_heat_lines`` (a
    snapshot has no wire-arrival clock). Age is the deterministic
    ``_seen_age_hours`` (reused verbatim — its 0.0 sentinel for
    absent/unparseable/future ``first_seen`` is < ``min_age_h`` so an unknown
    age is correctly never flagged, and snapshots are url-guarded out anyway).
    Returns ``["#1 ~6.2h — <title>", ...]`` for the aged top rows in rank
    order, capped at ``_AGING_MAX_LINES``. ``[]`` when every top row is fresh.
    No DB / IO / mutation."""
    out: list[str] = []
    for rank, a in enumerate(articles[:_AGING_TOP_SCAN], 1):
        if not (a.get("link") or a.get("url")):
            continue  # snapshot/synthetic row — same guard as the [BOOK:] tag
        age_h = _seen_age_hours(a.get("first_seen"), now=now)
        if age_h < min_age_h:
            continue
        title = ((a.get("title") or "").strip()[:60]) or "(untitled)"
        out.append(f"#{rank} ~{age_h:.1f}h — {title}")
        if len(out) >= _AGING_MAX_LINES:
            break
    return out


# ── ML scorer freshness (operational-status family) ──────────────────────────
# ArticleNet scores EVERY collected article and produces the [model]-tagged
# urgent calls. When the ml_trainer worker fails persistently the model
# silently stops learning new labels — observed live 2026-05-22: train()
# returns {"status":"error","reason":"subprocess_timeout"} every cycle, and
# data/ml/training_metrics.jsonl had not been appended for ~80h. COVERAGE GAP
# tells the analyst which COLLECTORS went dark; nothing told them the SCORER
# itself went stale. This is the missing read — same surfacing discipline as
# COVERAGE GAP / THROUGHPUT DEGRADATION / ALERT VELOCITY: a separate input
# block, REPRODUCED as a one-line section, omitted entirely when the model is
# fresh so it never becomes noise.
#
# Source: data/ml/training_metrics.jsonl — ml.trainer._log_metrics appends one
# JSON line per SUCCESSFUL train/continuous cycle (the skipped/error paths
# return before logging), so the last line's ``ts`` is the last successful
# retrain. Resolved via the same DIGITAL_INTERN_ML_DIR env convention
# ml.trainer uses — NOT by importing ml.trainer, which would pull torch into
# the light analysis layer (same anti-import-weight discipline as
# article_store._briefing_domain_key duplicating ml.features). Pure stdlib.
#
# Pure read-side: no DB write, no ai_score/ml_score/score_source/urgency
# touch, never reads or mutates source_articles — all four load-bearing
# invariants intact by construction.

# Warn when the last successful retrain is older than this. ArticleNet
# retrains every ~30min (daemon.ML_TRAIN_INTERVAL) so a multi-hour gap means
# the trainer is genuinely stuck, not mid-cycle. 6h is conservative — it
# spans more than one full 5h briefing window, well clear of normal churn.
_ML_STALE_WARN_HOURS = 6.0
_ML_METRICS_FILENAME = "training_metrics.jsonl"


def _collect_ml_freshness() -> dict | None:
    """Best-effort: ISO ts of the last SUCCESSFUL ArticleNet retrain.

    Returns ``{"last_ts": str}`` from the final parseable line of
    ``data/ml/training_metrics.jsonl`` (ml.trainer logs one line per
    successful cycle), or ``None`` on ANY failure (missing/empty/corrupt
    file) — an ML-freshness read must NEVER break or delay the 5h briefing
    it annotates (identical discipline to ``_collect_source_health`` /
    ``_collect_alert_velocity``)."""
    try:
        import os
        import json as _json
        from pathlib import Path
        ml_dir = Path(os.environ.get(
            "DIGITAL_INTERN_ML_DIR",
            Path(__file__).resolve().parent.parent / "data" / "ml",
        ))
        path = ml_dir / _ML_METRICS_FILENAME
        last_ts = None
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = _json.loads(line)
                except Exception:
                    continue
                ts = rec.get("ts") if isinstance(rec, dict) else None
                if isinstance(ts, str) and ts:
                    last_ts = ts
        if not last_ts:
            return None
        return {"last_ts": last_ts}
    except Exception:
        return None


def _ml_freshness_lines(
    freshness: dict | None,
    now: datetime | None = None,
    warn_hours: float = _ML_STALE_WARN_HOURS,
) -> list[str]:
    """Pure: 0 or 1 analyst-facing line on ArticleNet scorer staleness.

    Emits a single line ONLY when the last successful retrain is older than
    ``warn_hours``. A fresh model, an unknown/unparseable ts, or a future ts
    (clock skew) returns ``[]`` so the caller omits the whole section — the
    same "omit when healthy" discipline as ``_coverage_gap_lines`` /
    ``_alert_velocity_lines``. Pure — no DB / IO / mutation."""
    if not isinstance(freshness, dict):
        return []
    last_ts = freshness.get("last_ts")
    if not isinstance(last_ts, str) or not last_ts:
        return []
    try:
        dt = datetime.fromisoformat(last_ts.strip().replace("Z", "+00:00"))
    except Exception:
        return []
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    now = now or datetime.now(timezone.utc)
    age_h = (now - dt).total_seconds() / 3600.0
    if age_h < warn_hours:
        return []
    return [
        f"ArticleNet scorer last retrained ~{age_h:.1f}h ago "
        f"(target every ~30min) — NN relevance/urgency scores and any "
        f"[model]-tagged urgent calls are running on stale weights"
    ]


def _build_payload(articles, stock_data, earnings, source_health_report=None,
                   prior_digest=None, source_throughput=None,
                   alert_velocity=None, alert_book_velocity=None,
                   macro_calendar_events=None, ml_freshness=None):
    parts = [f"BRIEFING TIME: {_now_utc_str()}\n"]

    macro_data   = stock_data.get("macro", [])   if isinstance(stock_data, dict) else []
    equity_data  = stock_data.get("equities", []) if isinstance(stock_data, dict) else []

    parts.append("=== LIVE MARKET DATA ===")
    for s in macro_data:
        parts.append(_fmt_ticker(s))

    parts.append("\n=== EQUITY DATA ===")
    for s in equity_data:
        parts.append(_fmt_ticker(s))

    parts.append("\n=== NEWSWIRE (scored, ranked) ===")
    # Quote-widget noise gate FIRST — drop live ticker-tape pseudo-articles
    # before dedup/decay/cap so the analyst's primary Opus digest never
    # surfaces "NVDANVIDIA Corporation227.13-8.61(-3.65%)" as a TOP SIGNAL
    # (the documented #1 noise complaint; the alert path and web_scraper
    # already gate it, the briefing path did not). Pure read-side reshape:
    # returns NEW lists, never mutates the caller's source_articles, so the
    # training-label / backtest-isolation invariants are untouched. An empty
    # input (or an all-widget cycle with no prepended snapshots) degrades to
    # the same "(no high-relevance ...)" line as before — behaviour-preserving
    # for the common path, strictly cleaner for the widget path.
    articles, _qw_suppressed = _filter_quote_widget_noise(articles or [])
    # Recap / SEO template gate — defense-in-depth, same shape as the quote-
    # widget gate above. Drops "Why X Stock Is Trading Up Today" / "Stock
    # Market Today, May 18:" / "Q1 2026 Earnings Call Highlights" / GuruFocus
    # "GF Value Says ..." rows the urgency head over-scored before dedup/decay/
    # cap so the analyst's primary Opus digest never surfaces them as TOP
    # SIGNALS (live evidence 2026-05-19 04:18Z briefing: "[00:50] 9.85 MU
    # Motley: why MU dropped" — a model-only 9.85 on the exact recap template
    # the alert path already filters). Pure read-side reshape: returns NEW
    # lists, never mutates caller's source_articles, so the training-label /
    # backtest-isolation invariants are untouched. Runs BEFORE dedup so a
    # recap syndicated across multiple feeds is caught on every copy.
    articles, _recap_suppressed = _filter_recap_template_noise(articles)
    if not articles:
        parts.append("(no high-relevance articles this cycle)")
    else:
        # Collapse cross-domain syndication FIRST, then cap at 60. Dedup only
        # frees slots for *distinct* stories, so the cap can only ever surface
        # MORE unique signal, never less. Cap is 60 (not 50) because the caller
        # prepends up to 2 synthetic snapshot rows (P&L, options) to a
        # 50-article top list; a [:50] cap silently truncates real articles.
        deduped = _collapse_syndicated(articles)
        # Second-stage ORDER-INDEPENDENT near-dup collapse. _collapse_syndicated
        # only merges an exact first-8-token prefix signature; a word-reordered
        # or source-attribution-suffixed copy of the SAME wire survives it and
        # reaches the analyst's primary Opus digest as a duplicate TOP SIGNAL
        # (their #1 noise complaint — the live 07:13Z window carried 5 such
        # residual dups). ml.dedup.dedupe_articles (pure stdlib, separately
        # unit-tested in tests/test_dedup.py; its own docstring names the
        # "briefing pre-filter" as the intended integration) is the
        # purpose-built complement. Threshold 0.7 is conservative enough that a
        # single-token antonym flip never merges opposite stories — see
        # BRIEFING_NEAR_DUP_THRESHOLD. score_key='ai_score' keeps the
        # highest-LLM-scored copy as the survivor, mirroring
        # _collapse_syndicated's _score tie-break intent. Pure read-side, the
        # SAME shape as _collapse_syndicated: returns the original dict objects,
        # never mutates the caller's source_articles (heartbeat_worker feeds
        # those onward to the briefing-label / training path), no DB write, no
        # ai_score/ml_score/score_source/urgency touch, backtest already
        # excluded upstream by get_top_for_briefing's _LIVE_ONLY_CLAUSE — all
        # four load-bearing invariants intact by construction. The rare
        # further-merged survivor keeps its OWN pre-merge [syndicated xN] count
        # (a conservative under-count, never over-stated): dedupe_articles is
        # reused verbatim, not forked, the documented anti-drift discipline
        # (same reason _collapse_syndicated reuses alert_dedup._signature).
        deduped = _dedupe_near_duplicates(
            deduped, threshold=BRIEFING_NEAR_DUP_THRESHOLD
        )
        # Apply the documented per-article recency decay (the ML
        # time_sensitivity head). A stable sort keeps the prepended
        # PORTFOLIO/OPTIONS snapshots pinned at the top (age 0 → no decay →
        # effective == 10, the max) and only promotes a fresher item above an
        # older equal-base one — exactly what a "what is moving NOW" newswire
        # wants. Done before the [:60] cap so decay decides what survives it.
        deduped = _rank_by_decayed_score(deduped)
        # Alert↔briefing parity. Canonical signatures that fired a standalone
        # 🚨 BREAKING push within alert_recency.ALERT_RECENCY_TTL_HOURS (6h ≈
        # the 5h briefing window). Fetched ONCE per briefing (single read of a
        # separate alert_recency.db — never articles.db), best-effort → set()
        # on failure so the digest is unaffected. A digest row whose headline
        # signature is in this set is a story the analyst was ALREADY pushed —
        # the LEAD must not re-surface it as fresh (their top duplicate-alert
        # complaint). Pure read-side: no DB write, no ai_score/ml_score/
        # score_source/urgency touch, backtest excluded upstream by
        # get_top_for_briefing's _LIVE_ONLY_CLAUSE — four invariants intact.
        alerted_sigs = _recent_alert_signatures()
        for i, a in enumerate(deduped[:60], 1):
            score = a.get("ai_score") or a.get("_relevance_score", "?")
            corro = a.get("_corroboration", 1)
            # Wide independent syndication is itself a magnitude signal —
            # surface it verbatim so Opus can weight a 6-wire story over a
            # lone mention in TOP SIGNALS / LEAD.
            tag = f" [syndicated x{corro}]" if corro > 1 else ""
            # Echo calibration: a >=3-copy cluster whose copies all came from
            # ONE source key is single-source self-syndication, not
            # independent corroboration. Qualifies the [syndicated xN] tag so
            # Opus doesn't over-weight a mass-aggregator host repeating
            # itself. Pinned by tests/test_briefing_echo_tag.py. Pure render-
            # side: no DB write, no ai_score / ml_score / score_source /
            # urgency touch — four invariants intact by construction.
            echo_tag = " [echo]" if _is_echo_row(a) else ""
            # Real wire-arrival clock so Opus fills the SYSTEM_PROMPT
            # TOP SIGNALS "[HH:MM]" slot from data, not invention. Omitted
            # for the synthetic PORTFOLIO/OPTIONS snapshot rows (no
            # first_seen) — see _seen_utc_str.
            seen = _seen_utc_str(a.get("first_seen"))
            seen_tag = f" [seen {seen} UTC]" if seen else ""
            # Verified-vs-model-only calibration tag. `_llm_vetted` is set by
            # article_store.get_top_for_briefing: True = a real Opus/Sonnet
            # ai_score, False = the displayed score came from ml_score only
            # (an UNVERIFIED local-model estimate; the relevance head
            # demonstrably over-scores forum/wiki/social rows). Only an
            # explicit False tags — the prepended PORTFOLIO/OPTIONS snapshot
            # rows carry no `_llm_vetted` key (.get → None, `is False` →
            # False) so they are never tagged, and an LLM-vetted row (True)
            # is not tagged either. Survives _collapse_syndicated's shallow
            # copy; reflects the cluster representative (the highest-scored
            # copy — i.e. the score actually shown — by design, NOT OR-ed
            # across siblings, so the tag always matches the rendered number).
            model_tag = " [model]" if a.get("_llm_vetted") is False else ""
            # Already-pushed parity tag. The row's canonical headline signature
            # (alert_dedup._signature — the SAME primitive the cross-cycle
            # alert-suppression uses, so this tag and that gate agree by
            # construction) is in the recent fired-alert set ⇒ the analyst was
            # already pushed this exact story as 🚨 BREAKING. Guarded on a real
            # url so the prepended PORTFOLIO/OPTIONS snapshot rows (no link/url
            # — same guard as _extract_briefing_labels) are NEVER tagged; an
            # empty/untitled signature is never in the set (recent_signatures
            # filters falsy sigs, mirroring partition_already_alerted's
            # "untitled rows never suppressed" policy). Survives
            # _collapse_syndicated's shallow copy; reflects the cluster
            # representative's title — the one actually rendered.
            pushed = ""
            if alerted_sigs and (a.get("link") or a.get("url")):
                if _signature(a.get("title")) in alerted_sigs:
                    pushed = " [ALERTED]"
            # Held-book relevance tag. Guarded on a real url for the SAME
            # reason [ALERTED] is: the prepended PORTFOLIO/OPTIONS snapshot
            # rows carry no link/url (and their P&L body legitimately lists
            # held tickers — e.g. "MU -6.6%"), so without this guard every
            # snapshot row would falsely render "[BOOK: MU,...]". Same
            # snapshot-exclusion discipline as _extract_briefing_labels.
            # Survives _collapse_syndicated's shallow copy and reflects the
            # cluster representative (the row actually rendered) by design.
            book_tag = ""
            if a.get("link") or a.get("url"):
                _bk = _book_tickers(a)
                if _bk:
                    book_tag = f" [BOOK: {','.join(_bk)}]"
            parts.append(
                f"{i:>2}. [score={score}]{model_tag}{seen_tag}{tag}{echo_tag}{pushed}{book_tag} [{a.get('source','?')}] {a.get('title','')}\n"
                f"    {(a.get('summary') or '')[:300]}"
            )
        # Held-book concentration hint. Computed over the SAME collapsed+capped
        # rows rendered above (deduped[:60]) so "distinct stories" is honest and
        # verifiable against the newswire. Pure read-side: NEW list, no mutation
        # of source_articles, no DB / ai_score / ml_score / score_source /
        # urgency touch, backtest already excluded upstream — four invariants
        # intact. An input ranking hint, NOT a reproduced section (unlike
        # COVERAGE GAP) — see the SYSTEM_PROMPT BOOK HEAT rule.
        heat_lines = _book_heat_lines(deduped[:60])
        if heat_lines:
            parts.append(
                "\n=== BOOK HEAT (analyst's held names by distinct-story "
                "concentration this window — a magnitude signal) ==="
            )
            for hl in heat_lines:
                parts.append(f"  - {hl}")

        # Held-book SILENCE hint — held names with ZERO mentions this window.
        # The complement to BOOK HEAT (concentration). Opus composes PORTFOLIO
        # BLIND to which held tickers are dark and historically fabricates a
        # "neutral implication" for them — surfacing the silent set as INPUT
        # lets PORTFOLIO honestly mark them N/A. Pure read-side, same shape as
        # BOOK HEAT (input hint, never echoed): NEW list, no mutation of
        # source_articles, no DB / ai_score / ml_score / score_source /
        # urgency touch, backtest already excluded upstream by
        # get_top_for_briefing's _LIVE_ONLY_CLAUSE — four invariants intact.
        silence_lines = _book_silence_lines(deduped[:60])
        if silence_lines:
            parts.append(
                "\n=== BOOK SILENCE (analyst's held names with ZERO stories "
                "this window — catalyst engine dark, drift on macro/peers) ==="
            )
            for sl in silence_lines:
                parts.append(f"  - {sl}")

        # Deterministic wall-clock recency cross-check on the rows Opus leads
        # with. Computed over the SAME collapsed+decayed+capped list rendered
        # above (deduped[:60]) so "#N" matches the newswire rank. Independent
        # of the model-estimated ts decay (a row the ts head under-scored can
        # still be stale; a sparse window floats an aged item to #1). Pure
        # read-side: NEW list, no mutation of source_articles, no DB /
        # ai_score / ml_score / score_source / urgency touch, backtest already
        # excluded upstream — four invariants intact. A framing hint, NOT a
        # reproduced section (unlike COVERAGE GAP) — see the SYSTEM_PROMPT
        # AGING TOP ROWS rule.
        aging_lines = _aging_top_rows(deduped[:60])
        if aging_lines:
            parts.append(
                "\n=== AGING TOP ROWS (high-ranked digest rows several hours "
                "old — developing, NOT fresh breaks) ==="
            )
            for al in aging_lines:
                parts.append(f"  - {al}")

    parts.append("\n=== EARNINGS CALENDAR (next 48h) ===")
    if not earnings:
        parts.append("None on calendar.")
    else:
        for e in earnings:
            # `or` (not the .get default) so a present-but-None value still
            # renders as the placeholder rather than the literal "None".
            parts.append(f"  {e.get('ticker') or '?'}  {e.get('earnings_date') or 'N/A'}")

    # Forward macro-calendar block — the FOMC / CPI / Jobs / PPI catalysts
    # the macro_calendar_collector writes to articles.db with future
    # ``published`` timestamps. Only emitted when an explicit event list is
    # supplied (analyze() fetches it live). When None, the section is omitted
    # entirely so the prompt's "omit if no macro block" rule fires and
    # callers/tests that build a payload without macro context stay
    # deterministic — exact same shape as source_health_report / source_throughput
    # / alert_velocity above (the documented anti-drift discipline). A
    # REPRODUCED section (operational-status family, like COVERAGE GAP /
    # THROUGHPUT DEGRADATION / ALERT VELOCITY), NOT an INPUT hint like
    # BOOK HEAT. Pure read-side: no DB write, no
    # ai_score/ml_score/score_source/urgency touch, never reads or mutates
    # source_articles, backtest already excluded by the _collector's
    # ``source='macro_calendar'`` filter — all four load-bearing invariants
    # intact by construction.
    if macro_calendar_events is not None:
        macro_lines = _macro_calendar_event_lines(macro_calendar_events)
        if macro_lines:
            parts.append(
                "\n=== MACRO CALENDAR (FOMC / CPI / Jobs / PPI within "
                f"{MACRO_CALENDAR_WINDOW_HOURS}h — scheduled forward "
                "catalysts) ==="
            )
            for ml in macro_lines:
                parts.append(f"  - {ml}")

    # Coverage-gap block — only emitted when an explicit report is supplied
    # (analyze() fetches it live). When None, the section is omitted entirely
    # so the prompt's "omit if no gap block" rule fires and callers/tests that
    # build a payload without health context stay deterministic.
    if source_health_report is not None:
        gap_lines = _coverage_gap_lines(source_health_report)
        if gap_lines:
            parts.append(
                "\n=== COVERAGE GAP (intel channels dark this window — "
                "absence is NOT 'no news') ==="
            )
            for gl in gap_lines:
                parts.append(f"  - {gl}")

    # Throughput-degradation block — partial blind spots, the early-warning
    # complement to COVERAGE GAP. Only emitted when an explicit throughput
    # report is supplied (analyze() fetches it live). When None, the section
    # is omitted entirely so the prompt's "omit if no degradation block"
    # rule fires and callers/tests that build a payload without throughput
    # context stay deterministic — exact same shape as the source_health_report
    # block above (the documented anti-drift discipline).
    if source_throughput is not None:
        deg_lines = _throughput_degradation_lines(source_throughput)
        if deg_lines:
            parts.append(
                "\n=== THROUGHPUT DEGRADATION (live sources still up but "
                "delivering far less this window — partial blind spots) ==="
            )
            for dl in deg_lines:
                parts.append(f"  - {dl}")

    # Alert-velocity block — BREAKING-wire firing-rate magnitude hint. Only
    # emitted when an explicit velocity dict is supplied (analyze() fetches it
    # live). When None, the section is omitted entirely so the prompt's "omit
    # if no velocity block" rule fires and callers/tests that build a payload
    # without alert-velocity context stay deterministic — exact same shape as
    # the source_throughput block above (the documented anti-drift discipline).
    # A REPRODUCED section (like COVERAGE GAP / THROUGHPUT DEGRADATION),
    # operational-status family. Pure read-side: no DB write, no
    # ai_score/ml_score/score_source/urgency touch, never reads or mutates
    # source_articles, backtest already excluded upstream — invariants intact.
    if alert_velocity is not None:
        av_lines = _alert_velocity_lines(alert_velocity)
        if av_lines:
            parts.append(
                "\n=== ALERT VELOCITY (BREAKING wire firing rate vs prior "
                "window — magnitude signal, not in any one row's score) ==="
            )
            for al in av_lines:
                parts.append(f"  - {al}")

    # ML-scorer-staleness block — operational-status family, REPRODUCED (like
    # COVERAGE GAP / THROUGHPUT DEGRADATION / ALERT VELOCITY). Only emitted
    # when an explicit freshness dict is supplied (analyze() reads it live)
    # AND the model is materially stale; a fresh model emits nothing so the
    # prompt's "omit if no stale block" rule fires and the common path stays
    # byte-deterministic — exact same shape as the alert_velocity block above
    # (the documented anti-drift discipline). Pure read-side: no DB write, no
    # ai_score/ml_score/score_source/urgency touch, never reads or mutates
    # source_articles — all four load-bearing invariants intact.
    if ml_freshness is not None:
        ml_lines = _ml_freshness_lines(ml_freshness)
        if ml_lines:
            parts.append(
                "\n=== ML SCORER STALE (local ArticleNet has not retrained "
                "for hours — scores running on stale weights) ==="
            )
            for ml in ml_lines:
                parts.append(f"  - {ml}")

    # Alert-book-velocity block — per-held-ticker BREAKING-alert magnitude.
    # ALERT VELOCITY measures the OVERALL wire firing rate; this is the
    # per-position complement, reading the SAME canonical fires log
    # (alert_recency.db, NOT articles.db urgency=2) but bucketing by which
    # held tickers each alert's title mentions. Same omit-when-None /
    # omit-when-below-threshold discipline as alert_velocity (so the
    # 7-arg path stays byte-identical for callers that don't pass it). An
    # INPUT hint, NOT a reproduced section (SYSTEM_PROMPT rule below) — same
    # BOOK-HEAT-shaped contract as the held-book block family. Pure read-side:
    # no DB write, no ai_score/ml_score/score_source/urgency touch, never
    # reads or mutates source_articles, alert_recency.db is a separate file —
    # all four load-bearing invariants intact by construction.
    if alert_book_velocity is not None:
        abv_lines = _alert_book_velocity_lines(alert_book_velocity)
        if abv_lines:
            parts.append(
                "\n=== ALERT BOOK VELOCITY (held names mentioned in multiple "
                "BREAKING alerts vs prior window — per-position magnitude) ==="
            )
            for al in abv_lines:
                parts.append(f"  - {al}")

    # Prior-digest continuity block — anti-rehash. Only emitted when an
    # explicit prior digest is supplied (analyze() reads it best-effort);
    # ``None`` ⇒ the section is omitted entirely and the 4-arg path stays
    # byte-deterministic (exact discipline as source_health_report above —
    # callers/tests that don't pass it are unaffected). A framing/selection
    # hint, NOT a reproduced section (the SYSTEM_PROMPT rule forbids echoing
    # it, like BOOK HEAT / AGING TOP ROWS). Pure read-side: reads only the
    # separately-fetched prior-briefing dict, never the newswire/source_articles
    # list, never the DB here — all four invariants intact by construction.
    if prior_digest is not None:
        pd_lines = _prior_digest_lines(prior_digest)
        if pd_lines:
            age = prior_digest.get("age_h") if isinstance(prior_digest, dict) \
                else None
            age_str = (f"~{age:.1f}h ago"
                       if isinstance(age, (int, float)) else "earlier")
            parts.append(
                f"\n=== PRIOR DIGEST (the LEAD/TOP SIGNALS you published "
                f"{age_str} — do NOT simply re-lead the same story; lead "
                f"with what materially CHANGED since) ==="
            )
            for pl in pd_lines:
                parts.append(f"  - {pl}")

    return "\n".join(parts)


def analyze(articles, stock_data, earnings):
    payload = _build_payload(
        articles, stock_data, earnings,
        source_health_report=_collect_source_health(),
        prior_digest=_recent_briefing_digest(),
        source_throughput=_collect_source_throughput(),
        alert_velocity=_collect_alert_velocity(),
        alert_book_velocity=_collect_alert_book_velocity(),
        macro_calendar_events=_collect_macro_calendar_events(),
        ml_freshness=_collect_ml_freshness(),
    )
    full_prompt = f"{SYSTEM_PROMPT}\n\n---\nDATA INPUT:\n{payload}"
    result = claude_call(full_prompt, model=MODEL, timeout=180)
    return result or "[analyst] No response from Claude."


if __name__ == "__main__":
    print(analyze([], {}, []))
