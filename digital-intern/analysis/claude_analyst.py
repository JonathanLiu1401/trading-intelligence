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
    for idx, art in enumerate(articles):
        sig = _signature(art.get("title"))
        if not sig:
            sig = f"__uniq__{idx}"
        cur = keep.get(sig)
        if cur is None:
            rep = dict(art)
            rep["_corroboration"] = 1
            keep[sig] = rep
            order.append(sig)
            continue
        cur["_corroboration"] += 1
        # Strictly-greater so ties keep the earlier (already higher-ranked)
        # representative — deterministic and stable.
        if _score(art) > _score(cur):
            merged = dict(art)
            merged["_corroboration"] = cur["_corroboration"]
            keep[sig] = merged
    return [keep[s] for s in order]

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

SYSTEM_PROMPT = """You are a financial intelligence briefing engine. Output is posted directly to Discord. Format must render cleanly there.

RULES:
- Every number exact. Every move has a cause. Zero hedging.
- Tickers in ALL CAPS. Prices to 2dp. Pct changes with sign (+/-).
- Each table in its own code block. Section headers as plain **bold** outside code blocks.
- Total output must fit in 1800 characters. Be ruthlessly concise. Cut low-signal rows.
- No nested backticks. No backtick dividers. Dividers are plain ━━━ lines outside code blocks.
- A newswire row tagged "[syndicated xN]" was independently carried by N sources — treat higher N as stronger corroboration/magnitude when choosing the LEAD and ordering TOP SIGNALS; a lone (untagged) item is single-sourced and less confirmed.
- A newswire row tagged "[model]" carries a score set by the local relevance model ONLY, with NO LLM verification; that model demonstrably over-scores forum/wiki/social rows. Treat an untagged (LLM-vetted) row as materially more trustworthy than a "[model]" row of equal or near-equal score: prefer untagged rows for the LEAD and rank them above "[model]" rows of similar score in TOP SIGNALS. NEVER make a lone "[model]" row the LEAD when an untagged row of comparable score exists.
- A newswire row tagged "[ALERTED]" ALREADY fired a standalone 🚨 BREAKING push to the analyst within the last few hours — it is a developing/continued story the analyst has ALREADY been told about, NOT new news. Do NOT make an "[ALERTED]" row the LEAD when any untagged story of comparable importance exists; rank a fresh untagged story above an "[ALERTED]" one of similar score in TOP SIGNALS; and frame any "[ALERTED]" item explicitly as continuation (e.g. "follows the earlier alert", "developing") — never as if it just broke. This is what separates new desk intel from a rehash of an alert already delivered.
- If a "COVERAGE GAP" block is present in the data input, reproduce it as a **COVERAGE GAP** section (one bullet per dark channel, verbatim). These are intel channels the system could NOT collect from this window — the analyst must know what they are blind to, not assume silence means calm. Omit the section entirely if no gap block is provided.

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


def _looks_like_quote_widget(article: dict) -> bool:
    """True for a live quote-tape entry masquerading as a digest article.

    Two independent title fingerprints (a letter glued directly to a decimal
    price; a parenthesised signed % change) plus a Yahoo /quote/ landing path.
    All anchored so real prose with $/%/comma numbers ("rises 22% to $35.1
    billion", "5,123.41 record high") and real quote-scoped article URLs
    ("/quote/NVDA/news/headline-123") are never caught. Byte-identical logic
    to watchers.alert_agent._looks_like_quote_widget and
    collectors.web_scraper._looks_like_quote_widget. The synthetic
    PORTFOLIO/OPTIONS snapshot rows the daemon prepends ("PORTFOLIO P&L
    SNAPSHOT" / "OPTIONS SNAPSHOT", no url) never match either fingerprint, so
    they always pass through untouched."""
    title = article.get("title") or ""
    if _QW_PRICE_GLUE.search(title) or _QW_PCT_PAREN.search(title):
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


def _build_payload(articles, stock_data, earnings, source_health_report=None):
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
    if not articles:
        parts.append("(no high-relevance articles this cycle)")
    else:
        # Collapse cross-domain syndication FIRST, then cap at 60. Dedup only
        # frees slots for *distinct* stories, so the cap can only ever surface
        # MORE unique signal, never less. Cap is 60 (not 50) because the caller
        # prepends up to 2 synthetic snapshot rows (P&L, options) to a
        # 50-article top list; a [:50] cap silently truncates real articles.
        deduped = _collapse_syndicated(articles)
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
            parts.append(
                f"{i:>2}. [score={score}]{model_tag}{seen_tag}{tag}{pushed} [{a.get('source','?')}] {a.get('title','')}\n"
                f"    {(a.get('summary') or '')[:300]}"
            )

    parts.append("\n=== EARNINGS CALENDAR (next 48h) ===")
    if not earnings:
        parts.append("None on calendar.")
    else:
        for e in earnings:
            # `or` (not the .get default) so a present-but-None value still
            # renders as the placeholder rather than the literal "None".
            parts.append(f"  {e.get('ticker') or '?'}  {e.get('earnings_date') or 'N/A'}")

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

    return "\n".join(parts)


def analyze(articles, stock_data, earnings):
    payload = _build_payload(
        articles, stock_data, earnings,
        source_health_report=_collect_source_health(),
    )
    full_prompt = f"{SYSTEM_PROMPT}\n\n---\nDATA INPUT:\n{payload}"
    result = claude_call(full_prompt, model=MODEL, timeout=180)
    return result or "[analyst] No response from Claude."


if __name__ == "__main__":
    print(analyze([], {}, []))
