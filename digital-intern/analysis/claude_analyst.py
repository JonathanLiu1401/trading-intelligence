"""Bloomberg Terminal-style briefing — Claude Opus 4.7 via CLI."""
from datetime import datetime, timezone

from core.claude_cli import claude_call
# Reuse the *single* well-tested headline-canonicalisation primitive
# (tests/test_alert_dedup.py) instead of re-deriving a signature here — the
# documented anti-drift discipline (same reason watchers.alert_recency imports
# it, and alert_agent reuses ml.features._source_credibility). alert_dedup is
# pure stdlib+re (no DB / ml / aiohttp import graph), so this adds no cycle.
from watchers.alert_dedup import _signature

MODEL = "claude-opus-4-7"


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
    sorted by criticality (filings first), then by how long it's been dark.
    Returns [] when nothing curated is down.
    """
    if not isinstance(report, dict) or not report:
        return []
    now = now or datetime.now(timezone.utc)
    rows: list[tuple[int, float, str]] = []
    for key, info in report.items():
        if key not in _COVERAGE_LABELS or not isinstance(info, dict):
            continue
        if not info.get("disabled"):
            continue
        label, priority = _COVERAGE_LABELS[key]
        last_seen = info.get("last_seen")
        dark_h: float | None = None
        if last_seen:
            try:
                dt = datetime.fromisoformat(str(last_seen))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                dark_h = max(0.0, (now - dt).total_seconds() / 3600.0)
            except (ValueError, TypeError):
                dark_h = None
        fails = int(info.get("consecutive_failures") or 0)
        delivered = int(info.get("total_articles") or 0)
        dark_str = f"{dark_h:.1f}h" if dark_h is not None else "unknown"
        extra = ", 0 delivered all session" if delivered == 0 else ""
        line = f"{label} — DARK {dark_str} ({fails} empty polls{extra})"
        # Sort key: priority asc, then longest-dark first (None sorts last).
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
    if not articles:
        parts.append("(no high-relevance articles this cycle)")
    else:
        # Collapse cross-domain syndication FIRST, then cap at 60. Dedup only
        # frees slots for *distinct* stories, so the cap can only ever surface
        # MORE unique signal, never less. Cap is 60 (not 50) because the caller
        # prepends up to 2 synthetic snapshot rows (P&L, options) to a
        # 50-article top list; a [:50] cap silently truncates real articles.
        deduped = _collapse_syndicated(articles)
        for i, a in enumerate(deduped[:60], 1):
            score = a.get("ai_score") or a.get("_relevance_score", "?")
            corro = a.get("_corroboration", 1)
            # Wide independent syndication is itself a magnitude signal —
            # surface it verbatim so Opus can weight a 6-wire story over a
            # lone mention in TOP SIGNALS / LEAD.
            tag = f" [syndicated x{corro}]" if corro > 1 else ""
            parts.append(
                f"{i:>2}. [score={score}]{tag} [{a.get('source','?')}] {a.get('title','')}\n"
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
