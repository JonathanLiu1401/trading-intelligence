"""
Urgent alert agent — Bloomberg BN newswire style, immediate Discord post.
"""
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse

from core.claude_cli import claude_call
# Reuse the *well-tested* word-boundary source-credibility lookup (pins the
# "ap matched inside snap" class of bug — tests/test_features.py) rather than
# duplicating the 40-entry SOURCE_CRED map here, which would silently drift
# (the recurring dashboard-parity / vendored-signals.py failure class).
from ml.features import _source_credibility, LIVE_PORTFOLIO_TICKERS, _LIVE_RE
from watchers import alert_recency
from watchers.alert_dedup import alerted_ids, dedupe_urgent

try:
    from core.logger import get_logger
    _log = get_logger("alert_agent")
except Exception:
    _log = logging.getLogger("alert_agent")

SONNET_MODEL = "claude-sonnet-4-6"
DISCORD_WEBHOOK = os.environ.get("DISCORD_WEBHOOK_URL", "")

ALERT_PROMPT = """You are a Bloomberg BN terminal newswire alert system. A high-urgency financial event has been detected.

Write a Discord alert in Bloomberg newswire style — dense, exact, no filler. Max 1800 chars.

Current UTC time (use this verbatim in the timestamp slot — do NOT guess): {now_utc}

FORMAT (use exactly):
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚨 BREAKING  ◈  [CATEGORY]  ◈  {now_utc} UTC
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[ONE LINE HEADLINE IN CAPS — what happened]

TICKERS:   [affected symbols]
IMPACT:    [BUY/SELL/WATCH] — [one sentence on direction]
CONTEXT:   [one sentence of background]
PORTFOLIO: [specific implication for LITE/MU/MSFT/AXTI/ORCL/TSEM/QBTS]
SOURCE:    [source name]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```
Then on a new line after the code block: [article url]

Categories: EARNINGS | RATING CHANGE | MACRO SHOCK | SUPPLY CHAIN | REGULATORY | FED | CRYPTO | M&A | GEOPOLITICAL

RECENCY: Each article below carries `age` = elapsed time since publication. Reflect it honestly — an item several hours old is a developing/continued story, NOT one that "just" broke; never imply a multi-hour-old item happened moments ago. {now_utc} is the alert send time, not the event time. If an item is materially old (≳3h), make that explicit in CONTEXT (e.g. "first reported ~Nh ago").

CONTINUITY: If an article carries a `related:` line, a standalone 🚨 BREAKING alert on a related developing story ALREADY fired to this analyst within the last few hours — they have already been told the headline event. Frame THIS alert explicitly as a continuation/update of it: lead the HEADLINE with a development verb (ESCALATES / EXTENDS / WIDENS / FOLLOWS), and in CONTEXT state it follows the earlier alert (e.g. "follows ~Nh-ago alert on <prior event>"). Do NOT present it as the first time this story broke. This is what stops the analyst seeing what reads as a duplicate BREAKING for an event they are already tracking.

BOOK: If an article carries a `book:` line, it names live portfolio/watchlist positions the analyst actually has money in (LITE/LNOK/MUU/DRAM/SNDU/MU/MSFT/AXTI/ORCL/TSEM/QBTS/NVDA). That event is directly actionable for the analyst's open risk: the PORTFOLIO line MUST name the listed held ticker(s) and give a concrete directional implication for each, and weight this article's IMPACT above generic macro colour of similar magnitude. Absence of a `book:` line means the event does not touch the held book — keep PORTFOLIO short (sector read-through only, no invented position).

Urgent articles detected:
{articles_text}

Output ONLY the alert message."""


ALERT_BATCH_SIZE = 5

# Minimum source credibility for a LONE (un-syndicated) article to fire a
# standalone urgent Bloomberg "🚨 BREAKING" alert. Below this is the
# social/forum tier the ML urgency head demonstrably over-scores:
#   reddit 0.40 · nitter 0.40 · twitter 0.35 · stocktwits 0.30
# (per ml.features.SOURCE_CRED). DEFAULT_SOURCE_CRED is 0.55, so an unknown /
# brand-new source is NEVER gated — only the explicitly-known low tier is.
# Everything legitimate clears it: rss 0.65, scraped 0.50, gdelt 0.58,
# wikipedia 0.60, google/yahoo 0.62-0.65, reuters/bloomberg 0.90, sec 0.95.
# Corroboration is the escape valve — a story syndicated across ≥2 sources
# (dedup ``dup_count`` > 1) bypasses this entirely. See
# ``_filter_low_authority_lone``. Observed live (24h): reddit/r/Daytrading and
# reddit/r/ValueInvesting each fired a BREAKING alert solo — exactly the noise
# this gate removes from the analyst's push channel.
ALERT_MIN_LONE_SOURCE_CRED = 0.45


def _filter_low_authority_lone(
    deduped: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Partition a *post-dedup* urgent list into ``(kept, suppressed)``.

    ``suppressed`` = a single, un-corroborated copy (``dup_count`` <= 1) from a
    known low-credibility social/forum source
    (``cred < ALERT_MIN_LONE_SOURCE_CRED``). Everything else is kept:

      - anything syndicated across ≥2 sources (``dup_count`` > 1) — independent
        corroboration is itself the signal the event is real;
      - anything from a credible *or unknown* source (≥ the threshold).

    Pure function — no DB / IO. Must run AFTER ``dedupe_urgent`` so
    ``dup_count`` reflects cross-source corroboration; running it before would
    suppress a genuinely breaking story that a low-cred feed merely happened to
    surface first."""
    kept: list[dict] = []
    suppressed: list[dict] = []
    for a in deduped:
        try:
            dup_count = int(a.get("dup_count") or 1)
        except (TypeError, ValueError):
            dup_count = 1
        if dup_count > 1:
            kept.append(a)
            continue
        cred = _source_credibility(a.get("source") or "")
        if cred < ALERT_MIN_LONE_SOURCE_CRED:
            suppressed.append(a)
        else:
            kept.append(a)
    return kept, suppressed


def _article_age_ok(art: dict) -> bool:
    """Return True if the article is less than 24 hours old."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    for field in ("published", "first_seen"):
        raw = (art.get(field) or "").strip()
        if not raw:
            continue
        try:
            # Try RFC 2822 (RSS/Atom)
            dt = parsedate_to_datetime(raw)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt >= cutoff
        except Exception:
            pass
        try:
            # Try ISO 8601
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt >= cutoff
        except Exception:
            pass
    # No parseable date in either field — block rather than risk stale alert.
    # Articles without any date were already pre-filtered by first_seen >= 24h
    # in get_unalerted_urgent, so reaching here means both fields are corrupt.
    _log.warning("[alert] article has no parseable date — dropping to be safe")
    return False


def _article_age_hours(art: dict) -> float | None:
    """Hours since the article was published — ``published`` preferred, else
    ``first_seen``. ``None`` when neither field parses (the caller then simply
    omits the age line; this NEVER blocks an alert — >24h staleness is already
    enforced by ``_article_age_ok``). First parseable field wins, RFC822 + ISO,
    naive→UTC: the exact convention ``_article_age_ok``/``urgency_scorer`` use,
    so the displayed age is consistent with the staleness gate."""
    now = datetime.now(timezone.utc)
    for field in ("published", "first_seen"):
        raw = (art.get(field) or "").strip()
        if not raw:
            continue
        dt = None
        try:
            dt = parsedate_to_datetime(raw)
        except Exception:
            dt = None
        if dt is None:
            try:
                dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except Exception:
                dt = None
        if dt is None:
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max(0.0, (now - dt).total_seconds() / 3600.0)
    return None


def _article_age_str(art: dict) -> str | None:
    """Compact, analyst-readable freshness label for an urgent row —
    ``4m`` / ``3.2h`` / ``16h``. ``None`` when the age is unknown so the
    caller omits the line entirely (silent, never a fabricated "0m")."""
    h = _article_age_hours(art)
    if h is None:
        return None
    if h < 1.0:
        m = int(round(h * 60))
        return f"{m}m" if m >= 1 else "<1m"
    if h < 10.0:
        return f"{h:.1f}h"
    return f"{int(round(h))}h"


# ── Quote-widget noise gate (defense-in-depth) ───────────────────────────────
# collectors/web_scraper now rejects Yahoo/Bloomberg live ticker-tape entries
# ("NVDANVIDIA Corporation227.13-8.61(-3.65%)") at ingestion, but web_scraper
# is not the only path a spaceless price-tick title can enter on
# (yahoo_ticker_rss, finnhub, a manual replay), and the urgency head
# demonstrably over-scores them (live: ml up to 9.99; one was Sonnet-scored
# 8.0 and fired a real 🚨 BREAKING push — the analyst's single biggest noise
# complaint). This is the same layered-defense shape as _is_synthetic /
# _article_age_ok / _filter_low_authority_lone: a formatter-side drop, NOT an
# ML-threshold change, at the single chokepoint every alert funnels through.
# The helper is duplicated from web_scraper rather than cross-imported — the
# watchers layer must not pull the collectors/aiohttp import graph (same
# rationale as article_store._briefing_domain_key duplicating ml.features).
_QW_PRICE_GLUE = re.compile(r"[A-Za-z]\$?\d{1,4}[.,]\d{2,3}")
_QW_PCT_PAREN = re.compile(r"\([+-]?\d{1,3}(?:\.\d+)?%\)")
_QW_QUOTE_PATH = re.compile(r"/quote/[^/]+/?$", re.I)
# Quote-aggregator share-card / listing-page pseudo-article — a DISTINCT
# surface the two title fingerprints above don't catch. Google News indexes
# the Moomoo/Futu/Webull "share this quote" landing pages whose title is the
# rendered card: "$NVIDIA (NVDA.US)$ - Moomoo" / "$Tencent (00700.HK)$ - Futu".
# Live evidence (2026-05-18, recurring across ≥6 prior passes): the exact row
# "$NVIDIA (NVDA.US)$ - Moomoo" from the `GN: Nvidia` collector was
# ML-relevance-scored 9.77 (ai_score 0) and fired a 🚨 BREAKING urgent alert
# (urgency=2) — the consuming analyst's recurring noise complaint, never
# fingerprint-gated (only the cred-bar approach was deferred as contested
# tuning; a fingerprint gate is the accepted quote-widget precedent). The
# fingerprint = leading "$" share-card lead glued to a "(SYMBOL.EXCH)$" close;
# bounded ({0,60}) so no catastrophic backtracking; validated zero false
# positives on the live $+paren headline corpus
# ("$NVDA breaks out (NYSE)", "Zscaler (NASDAQ:ZS) ... $223.00").
# Byte-identical to collectors.web_scraper / analysis.claude_analyst (lockstep).
_QW_LISTING = re.compile(
    r"^\s*\$[^$\n]{0,60}\([A-Za-z0-9.\-]{1,8}\.[A-Za-z]{1,4}\)\$"
)


def _looks_like_quote_widget(art: dict) -> bool:
    """True for a live quote-tape / quote-listing entry masquerading as an
    urgent article.

    Three independent title fingerprints (a letter glued to a decimal price; a
    parenthesised signed % change; a "$NAME (SYMBOL.EXCH)$" share-card listing
    page) plus a Yahoo /quote/ landing path. All are anchored so real headlines
    with $/%/comma numbers ("rises 22% to $35.1 billion", "5,123.41 record
    high"), real "$TICKER ..." prose ("$MU upgraded to Buy") and real
    quote-scoped article URLs are never caught. Mirrors
    collectors.web_scraper._looks_like_quote_widget."""
    title = art.get("title") or ""
    if (_QW_PRICE_GLUE.search(title) or _QW_PCT_PAREN.search(title)
            or _QW_LISTING.search(title)):
        return True
    url = art.get("link") or art.get("url") or ""
    try:
        if _QW_QUOTE_PATH.search(urlparse(url).path):
            return True
    except Exception:
        pass
    return False


def _filter_quote_widget_noise(
    arts: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Partition urgent rows into ``(kept, suppressed)``. ``suppressed`` is the
    quote-tape pseudo-articles; everything else is kept. Pure — no DB / IO.
    Runs BEFORE dedup so a price tick syndicated across two collectors
    (yahoo + finnhub surfacing the same NVDA tick) is still caught."""
    kept: list[dict] = []
    suppressed: list[dict] = []
    for a in arts:
        (suppressed if _looks_like_quote_widget(a) else kept).append(a)
    return kept, suppressed


# ── Recap / SEO template title gate (defense-in-depth) ──────────────────────
# A second, distinct surface the urgency head over-scores is the *recap /
# preview / transcript-summary* template — content that is inherently
# retrospective ("trading up TODAY", "Q1 Earnings Call Highlights", a date-
# stamped "Stock Market Today, May 18:" wrap-up) or algorithmic mill output
# ("(LITE) Shares Fall 8.8% -- GF Value Says ..."). These NEVER warrant a
# standalone 🚨 BREAKING push: by the time the recap was written the move was
# already in the market, the call was already over, the wire already printed.
#
# Live evidence (2026-05-18/19, recurring across multiple alert cycles
# inspected from the live articles.db urgency=2 set):
#   - "Why Nvidia (NVDA) Stock Is Trading Up Today" — fired 22:34 and 00:12
#     from Finnhub/Yahoo and YahooFinance/NVDA (two separate BREAKING pushes)
#   - "Why Did Micron Stock Drop Today ? | The Motley Fool" — fired 00:50
#   - "Stock Market Today, May 18: Micron Falls as Memory Concerns Test AI
#      Rally" — fired THREE times at 22:52 from YahooFinance/005930.KS,
#     Motley Fool, and Nasdaq Markets (three separate BREAKING pushes for
#     one wrap-up)
#   - "D-Wave Quantum Inc (QBTS) Q1 2026 Earnings Call Highlights: Surging
#      Bookings Amid Revenue Decline" — fired twice ~14 min apart
#   - "Lumentum Holdings Inc (LITE) Shares Fall 8.8% -- GF Value Says ..."
#     and "AXT Inc (AXTI) Shares Fall 14.3% -- GF Value Says Still
#     Overvalued" — GuruFocus algorithmic stock-mention press mill
#   - "Here What the Street Thinks About NVIDIA Corporation (NVDA)" —
#     opinion mill recap
#
# These all came from publishers ABOVE the ``ALERT_MIN_LONE_SOURCE_CRED``
# 0.45 bar (Finnhub 0.78, Motley Fool/yahoo ~0.65, GoogleNews 0.62) so the
# existing source-authority gate doesn't catch them — the failure is the
# *content type*, not the publisher's credibility tier.
#
# Same shape as ``_filter_quote_widget_noise``: a small, evidence-backed set
# of title fingerprints, anchored so real breaking headlines never match;
# runs BEFORE dedup so a recap syndicated across multiple feeds is
# suppressed once (not after dedup picks one survivor); suppressed rows are
# marked alerted UNCONDITIONALLY by ``send_urgent_alert`` so they exit the
# urgent queue. Pure read-side — no DB write, no ai_score/ml_score/
# score_source mutation, backtest already filtered upstream by
# ``_is_synthetic`` — all four load-bearing invariants intact.

# "Why <Co|TICKER> Stock Is Trading {Up|Down|Higher|Lower} Today" — the
# canonical Zacks/Yahoo/Finnhub recap template. Anchored ``^Why`` so a
# headline that USES "why" mid-sentence is unaffected. The "Stock Is
# Trading" + "Today" co-occurrence is the discriminator: a real headline
# like "Why investors are bullish on Nvidia" or "Why MU beat estimates"
# does NOT match.
_RT_WHY_TRADING = re.compile(
    r"^\s*why\s+.+?\s+stock\s+is\s+trading\s+(?:up|down|higher|lower)\s+today\b",
    re.IGNORECASE,
)
# "Why Did <X> Stock {Drop|Rise|Surge|Fall|Climb|Plunge|Soar|Jump} Today"
# (Motley Fool / Zacks variant — past-tense recap of an intraday move).
_RT_WHY_DID = re.compile(
    r"^\s*why\s+did\s+.+?\s+stock\s+"
    r"(?:drop|rise|surge|fall|climb|plunge|soar|jump|tumble)\b",
    re.IGNORECASE,
)
# "Stock Market Today, May 18: ..." — date-stamped daily market wrap-up
# (Motley Fool / Nasdaq / Yahoo). Anchored ^ + month-name + 1-2 digit day.
# Real headlines do not lead with this exact "Stock Market Today, <Month>
# <day>" pattern.
_RT_MARKET_TODAY = re.compile(
    r"^\s*stock\s+market\s+today\s*[,:]\s*"
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)\s+\d{1,2}\b",
    re.IGNORECASE,
)
# "Q1 2026 Earnings Call Highlights" — GuruFocus / Seeking Alpha transcript-
# summary template. The call already happened; this is recap, not breaking.
# Substring (not anchored) — the template appears mid-headline ("D-Wave
# Quantum Inc (QBTS) Q1 2026 Earnings Call Highlights: ..."). Quarter and
# year are bounded so a normal earnings preview doesn't match.
_RT_EARNINGS_CALL = re.compile(
    r"\bq[1-4]\s*20\d{2}\s+earnings\s+call\s+(?:highlights|recap|takeaways|"
    r"transcript|summary)\b",
    re.IGNORECASE,
)
# "Here['s|is] What the Street Thinks About <X>" — InsiderMonkey opinion-
# mill recap template. The "Street thinks" framing IS the recap signature.
_RT_STREET_THINKS = re.compile(
    r"^\s*here(?:'?s|\s+is)?\s+what\s+the\s+street\s+thinks\b",
    re.IGNORECASE,
)
# "(TICKER) Shares Fall 8.8% -- GF Value Says ..." — GuruFocus algorithmic
# press-mill template, posted on every fractional stock-price move. The
# "GF Value Says" tagline is unique to GuruFocus, so this is a high-
# precision pattern (no false positives on real headlines that mention
# value or analyst ratings).
_RT_GF_VALUE = re.compile(
    r"\bgf\s+value\s+says\b",
    re.IGNORECASE,
)

_RECAP_TEMPLATE_PATTERNS = (
    ("why_trading_today", _RT_WHY_TRADING),
    ("why_did_stock", _RT_WHY_DID),
    ("market_today_dated", _RT_MARKET_TODAY),
    ("earnings_call_recap", _RT_EARNINGS_CALL),
    ("street_thinks", _RT_STREET_THINKS),
    ("gf_value_says", _RT_GF_VALUE),
)


def _looks_like_recap_template(art: dict) -> tuple[bool, str]:
    """``(True, fingerprint_name)`` for a recap/preview/transcript-summary or
    algorithmic-mill title — these are inherently retrospective and must not
    fire a standalone 🚨 BREAKING push. ``(False, "")`` for everything else.

    Pure, side-effect-free; reads only ``title``. Six independent
    fingerprints, anchored so real breaking headlines are never caught
    (validated against the live noise set on 2026-05-18/19 and the
    must-survive corpus: "Nvidia Q3 revenue rises 22%...", "Fed cuts rates",
    "MU earnings blow past estimates", "Why investors are bullish on
    Nvidia", "MU shares halted")."""
    title = (art.get("title") or "").strip()
    if not title:
        return False, ""
    for name, pat in _RECAP_TEMPLATE_PATTERNS:
        if pat.search(title):
            return True, name
    return False, ""


def _filter_recap_template_noise(
    arts: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Partition urgent rows into ``(kept, suppressed)``. ``suppressed`` is
    the recap/preview/transcript-summary/algorithmic-mill rows the urgency
    head over-scored; everything else is kept. Pure — no DB / IO.

    Runs BEFORE ``dedupe_urgent`` so a recap syndicated across multiple
    feeds ("Stock Market Today, May 18: ..." carried by Motley Fool +
    Nasdaq + YahooFinance) is caught on every copy and the dedup layer
    isn't asked to discriminate against a real story with a similar prefix.

    The suppressed rows are tagged ``_recap_fingerprint`` for log clarity
    so an operator can see WHICH template fired — without dumping the full
    title, mirrors the syndication-dedup tagging discipline."""
    kept: list[dict] = []
    suppressed: list[dict] = []
    for a in arts:
        hit, name = _looks_like_recap_template(a)
        if hit:
            a = dict(a)  # shallow copy — never mutate caller's row
            a["_recap_fingerprint"] = name
            suppressed.append(a)
        else:
            kept.append(a)
    return kept, suppressed


# ── Held-book relevance (the analyst's open positions) ───────────────────────
# The 🚨 BREAKING alert is the analyst's most time-critical product, and the
# persona is explicitly "I depend on this to react to events affecting MY
# positions". Yet the prompt's mandatory PORTFOLIO line relied entirely on
# Sonnet *inferring* held-ticker relevance from the raw headline — a lone
# "Lumentum guides Q4 down" with no "LITE" in the text got a generic PORTFOLIO
# line, and a real held-name break read identically to generic macro colour.
# This surfaces the held tickers explicitly in the alert input, exactly like
# the briefing's well-tested ``[BOOK: ...]`` tag (tests/test_briefing_book_tag),
# so the two consumed products judge "touches the book" the same way (the
# documented cross-product anti-drift discipline). Reuses ml.features'
# LIVE_PORTFOLIO_TICKERS / _LIVE_RE verbatim — alert_agent already imports
# _source_credibility from that module, so this is a single source of truth
# with the model's own ticker features and can never drift from a local copy.
# Pure read-side: no DB write, no ai_score/ml_score/score_source/urgency touch,
# backtest rows already filtered by _is_synthetic / the store — only the prompt
# text Sonnet reads is enriched. All four load-bearing invariants intact.
def _book_tickers(art: dict) -> list[str]:
    """Held/watchlist tickers mentioned in an urgent row's title+summary.

    Sorted (deterministic, stable cycle-to-cycle), de-duplicated; ``[]`` when
    the row touches no held name. Matches on ``title + summary`` — the SAME
    surface as the briefing's ``_book_tickers`` and ml.features ticker density,
    so an alert and the 5h digest never disagree about whether a wire touches
    the book. Pure, side-effect-free; reads only ``title``/``summary`` via
    ``.get()`` and never mutates the article."""
    blob = f"{art.get('title') or ''} {art.get('summary') or ''}"
    if not blob.strip():
        return []
    hits = {m.upper() for m in _LIVE_RE.findall(blob)}
    if not hits:
        return []
    # hits ⊆ LIVE_PORTFOLIO_TICKERS by construction (the regex only matches
    # those literals); sorted() gives a stable, test-pinnable order.
    return sorted(hits)


def _is_synthetic(art: dict) -> bool:
    """True for backtest/opus-annotation rows that must never reach the live
    Bloomberg formatter. Mirrors storage.article_store._LIVE_ONLY_CLAUSE.

    The store's get_unalerted_urgent already excludes synthetic rows, but the
    invariant is load-bearing enough that we re-check at the formatter — a
    future caller that bypasses the store filter (e.g., a manual replay) must
    not leak training rows into Discord."""
    url = (art.get("link") or art.get("url") or "")
    source = (art.get("source") or "")
    if url.startswith("backtest://"):
        return True
    if source.startswith("backtest_") or source.startswith("opus_annotation"):
        return True
    return False


def send_urgent_alert(urgent_articles: list, store) -> bool:
    if not urgent_articles:
        return False
    if not DISCORD_WEBHOOK:
        _log.warning("[alert] No DISCORD_WEBHOOK_URL — skipping")
        return False

    # Defense-in-depth: synthetic backtest/opus-annotation rows must never
    # reach the live alert formatter. The store filter is the primary defense;
    # this is a second line.
    filtered = [a for a in urgent_articles if not _is_synthetic(a)]
    n_dropped = len(urgent_articles) - len(filtered)
    if n_dropped:
        _log.warning(
            f"[alert] dropped {n_dropped} synthetic rows leaked from upstream"
        )
    if not filtered:
        return False

    # Quote-widget noise gate (defense-in-depth, same shape as the synthetic
    # re-filter above). Runs before dedup. Suppressed rows are marked alerted
    # UNCONDITIONALLY here — regardless of whether Discord later fires — so a
    # spaceless price-tick that the urgency head over-scored exits the urgent
    # queue instead of being re-fetched and re-evaluated every 20s cycle. They
    # stay in articles.db untouched (training reads ai_score; ml_score /
    # score_source unchanged) — only the noisy push is dropped.
    filtered, qw_suppressed = _filter_quote_widget_noise(filtered)
    if qw_suppressed:
        try:
            store.mark_alerted_batch(alerted_ids(qw_suppressed))
        except Exception:
            _log.exception(
                "[alert] failed to mark quote-widget rows alerted"
            )
        srcs = ", ".join(
            f"{(a.get('source') or '?')}:{(a.get('title') or '')[:40]}"
            for a in qw_suppressed[:5]
        )
        _log.info(
            f"[alert] suppressed {len(qw_suppressed)} quote-widget pseudo-"
            f"article(s) (live ticker-tape, not breaking news) — {srcs}"
        )
    if not filtered:
        _log.info("[alert] all urgent rows were quote-widget noise — skipping")
        return False

    # Recap / SEO template gate (defense-in-depth, same shape as the quote-
    # widget gate above). "Why X Stock Is Trading Up Today" / "Stock Market
    # Today, May 18:" / "Q1 2026 Earnings Call Highlights" / GuruFocus's
    # "GF Value Says ..." are inherently retrospective or algorithmic mill
    # output the urgency head over-scores — the move was already in the
    # market by the time the recap was written. Suppressed rows are marked
    # alerted UNCONDITIONALLY so they exit the urgent queue instead of being
    # re-fetched every 20s cycle. They stay in articles.db untouched (ai_score
    # / ml_score / score_source unchanged) — only the noisy push is dropped.
    # Runs BEFORE dedup so a recap syndicated across multiple feeds ("Stock
    # Market Today, May 18: ..." live-evidenced three copies at the same
    # minute from Motley Fool + Nasdaq + YahooFinance) is caught on every
    # copy. Best-effort store mark — a store failure must never block a
    # genuine fresh alert in the same batch.
    filtered, recap_suppressed = _filter_recap_template_noise(filtered)
    if recap_suppressed:
        try:
            store.mark_alerted_batch(alerted_ids(recap_suppressed))
        except Exception:
            _log.exception(
                "[alert] failed to mark recap-template rows alerted"
            )
        fps = ", ".join(
            f"{a.get('_recap_fingerprint','?')}:{(a.get('title') or '')[:60]}"
            for a in recap_suppressed[:5]
        )
        _log.info(
            f"[alert] suppressed {len(recap_suppressed)} recap-template "
            f"row(s) (retrospective recap / SEO mill / transcript summary — "
            f"never breaking) — {fps}"
        )
    if not filtered:
        _log.info(
            "[alert] all urgent rows were recap-template noise — skipping"
        )
        return False

    # Drop articles older than 24 hours — stale news must not fire as breaking.
    # Suppressed (stale / un-datable) rows are marked alerted UNCONDITIONALLY
    # here — identical shape to the quote-widget gate above and the
    # low-authority / cross-cycle gates below — so a row get_unalerted_urgent
    # legitimately returns (recent first_seen) but whose `published` is >24h
    # old EXITS the urgent queue instead of being re-fetched and re-dropped
    # every 20s cycle until its first_seen ages out, then lingering forever as
    # a permanent urgency=1 residue (observed live 2026-05-18: 26 such rows
    # stuck 5 days, inflating the dashboard `urgent` tile and re-decompressed
    # every cycle). A stale-by-`published` row only ages further — it can
    # never become a valid fresh alert — so marking it loses no delivery; a
    # no-parseable-date row likewise can never pass _article_age_ok, so it too
    # must exit rather than churn forever. articles.db ai_score / ml_score /
    # score_source are untouched (mark_alerted_batch only sets urgency=2) and
    # synthetic rows were already filtered above — all four load-bearing
    # invariants intact. Best-effort mark (a store failure must never block a
    # genuine fresh alert in the same batch).
    fresh: list[dict] = []
    stale: list[dict] = []
    for a in filtered:
        (fresh if _article_age_ok(a) else stale).append(a)
    if stale:
        try:
            store.mark_alerted_batch(alerted_ids(stale))
        except Exception:
            _log.exception("[alert] failed to mark stale rows alerted")
        srcs = ", ".join(
            f"{(a.get('source') or '?')}:{(a.get('title') or '')[:40]}"
            for a in stale[:5]
        )
        _log.info(
            f"[alert] dropped {len(stale)} stale article(s) (>24h old) — {srcs}"
        )
    if not fresh:
        _log.info("[alert] all urgent articles are stale — skipping alert")
        return False
    filtered = fresh

    # Collapse syndicated duplicates first: one breaking story carried by GDELT
    # + Reuters + Yahoo + RSS would otherwise eat the whole 5-slot batch and
    # show the trader the same event five times. After dedup the batch holds
    # five DISTINCT stories; each survivor knows the ids of the copies it
    # absorbed (``_dup_ids``) so all of them can still be marked alerted.
    deduped = dedupe_urgent(filtered)

    # Source-authority gate (defense-in-depth, same shape as _is_synthetic /
    # _article_age_ok — a formatter-side drop, NOT an ML-threshold change). A
    # lone, un-corroborated social/forum post the urgency head over-scored must
    # not fire a standalone Bloomberg "🚨 BREAKING" alert: that is the single
    # biggest noise complaint from the analyst consuming this channel. A
    # syndicated story (dup_count>1) or any credible/unknown source still
    # fires. Suppressed rows are NOT lost — they stay in articles.db (training
    # reads ai_score, untouched; score_source/ml_score untouched) and Opus can
    # still surface them in the curated 5h briefing — only the noisy *push* is
    # dropped. They are marked alerted UNCONDITIONALLY (separate call, before
    # the Discord attempt, regardless of its outcome) so they exit the urgent
    # queue instead of being re-fetched and re-evaluated every 20s cycle.
    deduped, suppressed = _filter_low_authority_lone(deduped)
    if suppressed:
        try:
            store.mark_alerted_batch(alerted_ids(suppressed))
        except Exception:
            _log.exception(
                "[alert] failed to mark suppressed low-authority rows alerted"
            )
        srcs = ", ".join(
            f"{(a.get('source') or '?')}:{(a.get('title') or '')[:40]}"
            for a in suppressed[:5]
        )
        _log.info(
            f"[alert] suppressed {len(suppressed)} lone low-authority urgent "
            f"row(s) (cred<{ALERT_MIN_LONE_SOURCE_CRED}, no syndication) — "
            f"{srcs}"
        )
    if not deduped:
        _log.info(
            "[alert] all urgent rows were lone low-authority noise — skipping"
        )
        return False

    # Cross-cycle syndication suppression (defense-in-depth, same
    # formatter-side shape as _is_synthetic / _filter_quote_widget_noise /
    # _filter_low_authority_lone — NOT an ML-threshold change). dedupe_urgent
    # only collapses copies *inside this batch*; a slower feed re-collecting
    # an already-alerted event as a NEW row (urgency=1, the old copies are
    # urgency=2 and excluded from get_unalerted_urgent) would otherwise fire a
    # second standalone BREAKING push for an event the analyst was already
    # told about, possibly hours later (live: the H200/China story alerted
    # twice ~1.5h apart from different sources). Runs AFTER dedupe_urgent so a
    # signature is computed once per collapsed survivor, and after the
    # low-authority gate so a row suppressed there is never recorded. Best-
    # effort: a recency-store failure yields an empty set → no-op (the old
    # behaviour) — a genuine breaking story must still reach the analyst.
    # Suppressed rows are marked alerted UNCONDITIONALLY (separate call,
    # before any Discord attempt) so they leave the urgent queue instead of
    # being re-fetched every 20s. articles.db ai_score/ml_score/score_source
    # are untouched — only the noisy duplicate push is dropped.
    try:
        recent_sigs = alert_recency.recent_signatures()
    except Exception:
        recent_sigs = set()
    deduped, cross_suppressed = alert_recency.partition_already_alerted(
        deduped, recent_sigs
    )
    if cross_suppressed:
        try:
            store.mark_alerted_batch(alerted_ids(cross_suppressed))
        except Exception:
            _log.exception(
                "[alert] failed to mark cross-cycle duplicate rows alerted"
            )
        srcs = ", ".join(
            f"{(a.get('source') or '?')}:{(a.get('title') or '')[:40]}"
            for a in cross_suppressed[:5]
        )
        _log.info(
            f"[alert] suppressed {len(cross_suppressed)} cross-cycle "
            f"duplicate(s) (signature alerted within "
            f"{alert_recency.ALERT_RECENCY_TTL_HOURS:.0f}h) — {srcs}"
        )
    if not deduped:
        _log.info(
            "[alert] all urgent rows were cross-cycle duplicates — skipping"
        )
        return False

    # Only the first ALERT_BATCH_SIZE feed the prompt — and only those (plus the
    # duplicates they absorbed) get marked alerted. Marking the entire urgent
    # list would silently drop the tail (it'd never be picked up next cycle),
    # so we cap both ends.
    batch = deduped[:ALERT_BATCH_SIZE]

    # Continuation context (non-suppressing). Cross-cycle suppression above
    # already dropped EXACT-signature repeats; what survives can still be a
    # *different* headline about a story the analyst was already pushed (live:
    # the UAE-strike alert at 01:55 then a Brent/markets follow-up — distinct
    # signatures, correctly NOT collapsed). With no hint the LLM writes the
    # follow-up as if it just broke → the analyst's top "duplicate alerts"
    # complaint, on the one product (the standalone push) that never got the
    # mitigation the briefing's [ALERTED] tag gave. Annotate (never drop) each
    # survivor with the related prior alert so the prompt's CONTINUITY rule
    # frames it as a developing update. Best-effort: a recency-store failure
    # yields [] → no annotation → exact pre-feature behaviour (a genuine alert
    # must always still fire). Read-only: alert_recency.db only, NEVER
    # articles.db — no ai_score/ml_score/score_source/urgency touch, backtest
    # already filtered above. All four invariants intact by construction.
    try:
        _recent = alert_recency.recent_alerts()
    except Exception:
        _recent = []
    if _recent:
        for a in batch:
            try:
                rel = alert_recency.related_prior_alert(a.get("title"), _recent)
            except Exception:
                rel = None
            if rel:
                a["_related_prior"] = rel

    def _fmt(a: dict) -> str | None:
        # Defensive field access. The rest of this pipeline (_is_synthetic,
        # dedupe_urgent) reads every key through .get(); _fmt used to be the
        # one place with hard subscripts (a['link'], a['ai_score'], ...). A
        # single dict from a non-canonical caller (manual replay, or a row
        # carrying `url` instead of `link` — the exact alias _is_synthetic
        # already tolerates) raised KeyError, the broad except below swallowed
        # it, the WHOLE batch was dropped, nothing was marked alerted, and
        # urgent alerts silently failed every cycle. Skip one bad row instead
        # of unwinding the batch; only the headline is truly required.
        title = (a.get("title") or "").strip()
        if not title:
            _log.warning("[alert] skipping urgent row with no title (id=%s)",
                         a.get("_id"))
            return None
        try:
            score = float(a.get("ai_score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        link = a.get("link") or a.get("url") or ""
        source = (a.get("source") or "unknown").strip() or "unknown"
        block = (
            f"[score={score:.0f}] {title}\n"
            f"source: {source}\nurl: {link}"
        )
        # Freshness context: a news analyst reacting to "🚨 BREAKING" must be
        # able to tell a 4-minute-old 8-K (act now) from a 16-hour-old reused
        # headline (already priced in). The 0..24h band reaching here is wide
        # — the Sonnet prompt's RECENCY rule turns this into honest framing.
        age = _article_age_str(a)
        if age:
            block += f"\nage: {age} (time since publication)"
        dup_count = int(a.get("dup_count") or 1)
        if dup_count > 1:
            # Tell the alert LLM how broadly the story is being carried — wide
            # syndication is itself a signal of how big the event is.
            block += f"\nsyndication: reported by {dup_count} sources"
        # Held-book relevance — drives the prompt's BOOK rule so the mandatory
        # PORTFOLIO line names the analyst's actual open risk instead of Sonnet
        # guessing ticker relevance from the headline alone. Same shape as the
        # additive age/syndication/related lines; same concept as the briefing's
        # [BOOK:] tag (cross-product parity). Read-only — see _book_tickers.
        book = _book_tickers(a)
        if book:
            block += (
                f"\nbook: {','.join(book)} — analyst HOLDS/watches these; "
                f"the PORTFOLIO line MUST give a concrete directional "
                f"implication for them"
            )
        rel = a.get("_related_prior")
        if isinstance(rel, dict) and (rel.get("title") or "").strip():
            # Drives the prompt's CONTINUITY rule — a related 🚨 alert already
            # fired, so this is a developing update, not a fresh break.
            rh = float(rel.get("age_hours") or 0.0)
            if rh < 1.0:
                rage = f"{int(round(rh * 60))}m"
            elif rh < 10.0:
                rage = f"{rh:.1f}h"
            else:
                rage = f"{int(round(rh))}h"
            block += (
                f"\nrelated: a 🚨 BREAKING alert fired ~{rage} ago on a "
                f"related developing story — \"{rel['title'][:140]}\" "
                f"(frame THIS as a continuation/update, not a fresh break)"
            )
        summary = (a.get("summary") or "").strip()
        if summary:
            block += f"\nbody: {summary[:600]}"
        return block

    # Filter the batch to formattable rows BEFORE building the prompt AND
    # before alerted_ids(batch) — marking a skipped row alerted would silently
    # drop it forever; keeping it in batch would re-fire the whole cycle.
    formatted = [(a, _fmt(a)) for a in batch]
    batch = [a for a, t in formatted if t is not None]
    if not batch:
        _log.warning("[alert] no formattable urgent rows in batch — skipping")
        return False
    articles_text = "\n\n".join(t for _, t in formatted if t is not None)

    # Full date+time so Discord history is unambiguous across day boundaries.
    # Template already appends " UTC", so don't include it here.
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    prompt = ALERT_PROMPT.format(articles_text=articles_text, now_utc=now_utc)

    try:
        message = claude_call(prompt, model=SONNET_MODEL, timeout=60)
        if not message:
            _log.warning("[alert] No response from Claude — skipping")
            return False

        # post via discord_notifier which also fires TTS
        from notifier.discord_notifier import send as discord_send
        ok = discord_send(message, is_alert=True)

        if ok:
            # Bulk-mark in one transaction; previous code took the write lock
            # N times (5 round-trips for the default batch size). alerted_ids
            # includes the syndicated copies merged into the batch, so they
            # never re-fire — duplicates of still-queued stories stay urgent.
            mark_ids = alerted_ids(batch)
            store.mark_alerted_batch(mark_ids)
            # Record the canonical signature of every story that actually
            # fired so a later slower-feed copy of the same event is
            # cross-cycle-suppressed above. Best-effort — a failure here only
            # means a future duplicate is not muted (never worse than the
            # pre-feature behaviour); it must never undo a sent alert.
            try:
                alert_recency.record_alerted(batch)
            except Exception:
                _log.exception("[alert] alert_recency.record_alerted failed")
            collapsed = len(mark_ids) - len(batch)
            tail = len(deduped) - len(batch)
            notes = []
            if collapsed > 0:
                notes.append(f"{collapsed} syndicated dupes folded in")
            if tail > 0:
                notes.append(f"{tail} more queued")
            note = f" ({'; '.join(notes)})" if notes else ""
            _log.info(f"[alert] BN alert sent ({len(batch)} distinct stories){note}")
        else:
            _log.warning("[alert] Discord POST failed")
        return ok

    except Exception:
        _log.exception("[alert] Error sending urgent alert")
        return False
