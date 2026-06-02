"""
Urgency scorer — batches unscored articles to Claude Sonnet 4.6.
Marks articles as urgent (score >= 8) for immediate alerting.
"""
import json
import logging
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from core.claude_cli import DEFAULT_LLM_MODEL, claude_call
from core.json_extract import extract_json_array
# Live held/watched ticker universe — the SAME single source of truth the
# model's portfolio features and the alert path's ``book:`` tag already use
# (loaded from config/portfolio.json, unioned with a hardcoded fallback). The
# urgency SCORE_PROMPT historically hardcoded the held set, so a position
# added in the trading UI was invisible to Sonnet's urgency scoring — its
# earnings beat/miss was scored as generic sector news, never URGENT, and the
# analyst got no standalone push for their own open risk. ml.features is
# already in this module's import graph (pulled transitively via the
# watchers.alert_agent import above), so this adds no new dependency.
from ml.features import LIVE_PORTFOLIO_TICKERS
# Single source of truth for the recap-template AND quote-widget fingerprint
# sets — defined in alert_agent and re-used here so urgency_scorer, the alert
# formatter, and the briefing builder can never silently drift in WHICH titles
# count as a recap / quote-widget. Same intra-watchers import discipline as
# alert_agent->alert_recency. Pulls ml.features transitively (cheap — already
# in the daemon's process graph).
from watchers.alert_agent import (
    _looks_like_recap_template,
    _looks_like_quote_widget,
    _looks_like_stocktwits_chatter,
)

try:
    from core.logger import get_logger
    _log = get_logger("urgency_scorer")
except Exception:
    _log = logging.getLogger("urgency_scorer")

SONNET_MODEL = DEFAULT_LLM_MODEL
BATCH_SIZE = 100  # articles per Sonnet call
URGENT_THRESHOLD = 8.0
# Hard cap kicks in only for clearly stale articles (>48h).
# Timeless content (structural supply shifts, filings) can still be urgent within this window.
# The prompt instructs Claude to apply tighter judgment at >24h for time-sensitive language.
STALE_HOURS = 48.0
STALE_SCORE_CAP = 5.9  # max score for hard-capped stale articles (below urgent threshold of 8)
# When Sonnet hits its output-token limit the array comes back truncated and
# core.json_extract salvages only a leading prefix of indices. The contract
# there (and the existing empty-array guard below) is that the *caller*
# re-queues whatever is missing rather than burying it. Truncation has a
# distinct fingerprint: the returned indices are a clean prefix 0..k and the
# entire tail k+1..N-1 is absent. A model deliberately skipping an article it
# found ambiguous drops a handful at most; a long trailing block this size is
# truncation, so those articles are re-queued (left ai_score=0) instead of
# floored to noise.
MAX_TRAILING_OMISSIONS = 5

SCORE_PROMPT = """You are a real-time financial news urgency classifier. Score each article 0-10.

URGENT (8-10): Breaking news that will move markets NOW:
- Earnings beats/misses for the analyst's HELD POSITIONS: {portfolio_tickers}. Also sector peers (AMD, TSM, SK Hynix, Samsung, Micron)
- Fed surprise decisions, emergency rate changes
- Major analyst upgrades/downgrades (PT change >15%) on tracked stocks
- Memory/DRAM pricing shock (ASP move >5%)
- China/US escalation: new chip export bans, tariffs on semis
- Market circuit breakers, flash crashes
- Major geopolitical event (military action, sanctions) affecting tech supply chain
- HBM or DRAM supply disruption

RELEVANT (5-7): Important but not immediately actionable:
- General sector analysis, macro outlook
- Fed commentary, economic data (CPI, jobs, GDP)
- Crypto market moves, general tech news
- Analyst commentary without rating change

NOISE (0-4): Not relevant to this portfolio

STALENESS RULE: Each article includes age_hours. Use judgment:
- If the article uses time-sensitive language ("today", "breaking", "just announced", "this week", \
"yesterday") AND age_hours > 24 → the temporal claim is stale; cap your score at 5.
- If the content is fundamentally timeless (structural supply-chain shift, regulatory ruling, \
major SEC filing, long-term analyst thesis on tracked stocks) → score on merit regardless of age.
- If age_hours > 72 and urgency depends entirely on the event being recent → score ≤ 3.

Articles:
{articles_json}

Respond ONLY with a JSON array: [{{"index": 0, "score": 9, "reason": "MU earnings beat"}}, ...]"""


def _portfolio_ticker_line() -> str:
    """Comma-joined, sorted held/watched tickers for the SCORE_PROMPT.

    Reads ``ml.features.LIVE_PORTFOLIO_TICKERS`` (config/portfolio.json's
    positions + option underlyings + sector_watchlist, unioned with a
    hardcoded fallback) so Sonnet scores earnings urgency against the book the
    analyst *actually* holds, not a frozen literal. ``sorted`` gives a
    deterministic, test-pinnable order. Degrades to a minimal semiconductor
    default only if the live set is somehow empty — a urgency-scoring prompt
    must never go out with a blank held-positions slot."""
    tickers = sorted(t for t in LIVE_PORTFOLIO_TICKERS if t)
    return ", ".join(tickers) if tickers else "MU, NVDA, MSFT"


def _article_age_hours(article: dict) -> float:
    """Hours since the article was published — ``published`` preferred, else
    ``first_seen``. ``0.0`` when neither field parses (treat as fresh — safe
    default; staleness cap is a downward bound).

    Cascading fallback: a non-empty-but-unparseable ``published`` must NOT
    short-circuit the lookup at 0.0h, because that bypasses the
    ``STALE_HOURS`` cap on a row whose ``first_seen`` is genuinely old (live
    failure: an unparseable RFC822 date on a 40h-old article was treated as
    fresh, letting Sonnet's "urgent=9" through the staleness clamp). Mirrors
    ``alert_agent._article_age_hours``'s field-cascade convention so the two
    age helpers agree on which timestamp is authoritative."""
    now = datetime.now(timezone.utc)
    for field in ("published", "first_seen"):
        raw = (article.get(field) or "").strip()
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
    return 0.0


def score_batch(articles: list, store) -> int:
    """Score a batch of articles; update store. Returns count of urgent items found.

    Pre-filter pass (defense-in-depth): two classes of non-prose pseudo-articles
    are floored to noise (ai_score=0.01, urgency=0, score_source='llm') WITHOUT
    a Sonnet call.

      1. Live ticker-tape / quote-widget pseudo-articles
         ("NVDANVIDIA Corporation227.13-8.61(-3.65%)") — Sonnet and the urgency
         head both over-score these. Web_scraper drops them at ingestion but
         is not the only entry path (yahoo_ticker_rss / finnhub /
         google_news), and a 30-day live audit found 111 such rows tagged
         score_source='llm' with ai_score > 0 (one at 8.0). The alert path's
         own gate suppresses the Discord push but runs too late to prevent
         the training-pool contamination.

      2. Retrospective recap / preview / transcript-summary templates
         ("Why X Stock Is Trading Up Today", "Q1 2026 Earnings Call
         Highlights", "Stock Market Today, May 18:", "GF Value Says ...",
         "Here What the Street Thinks About ...") — Sonnet demonstrably
         mis-labels these as ai_score=8+ urgent (live evidence 2026-05-18/19:
         10 such rows landed in articles.db as score_source='llm', poisoning
         the training pool with retrospective fluff flagged as ground-truth
         urgent).

    The same fingerprint sets the alert path uses (``_looks_like_quote_widget``
    + ``_looks_like_recap_template`` — single source of truth) gate them here
    BEFORE the Claude call so we (1) save quota and (2) keep the LLM label
    distribution honest. The floor value 0.01 matches what
    ``urgency_scorer``'s anti-loop floor would write for a row Sonnet returned
    no score on, and is equivalent to what Sonnet would output if the prompt
    were extended with the same exclusion — so the trainer's strong-label pool
    inherits a noise-tagged 0.01 instead of an urgent-tagged 8.0.
    """
    if not articles:
        return 0

    # Quote-widget pre-filter runs FIRST: a spaceless price-tape title
    # ("NVDANVIDIA Corporation227.13-8.61(-3.65%)") that web_scraper missed
    # (or that entered via yahoo_ticker_rss / finnhub / google_news, none of
    # which apply the ingestion-time gate) would otherwise reach Sonnet and
    # be scored as if it were prose. Live evidence (2026-05-21, 30d audit):
    # 111 such rows landed with score_source='llm' and ai_score > 0 — one at
    # 8.0 (urgent territory) for a "NVDANVIDIA Corporation227.13-8.61(-3.65%)"
    # row. Those 111 rows then entered the trainer's strong-label pool tagged
    # as ground-truth LLM labels (STRONG_LABEL_WHERE includes score_source=
    # 'llm'), polluting the learning signal with patently-not-prose junk and
    # wasting Sonnet quota on every cycle. The alert path's defense-in-depth
    # quote-widget gate already drops these BEFORE Discord, but it runs too
    # late to prevent training-pool contamination — the row already carries
    # ai_score=8 by then. Same single-source-of-truth fingerprint set as the
    # alert formatter (``_looks_like_quote_widget`` imported above), so a
    # regex tightening on the alert side automatically engages here.
    quote_widget_articles: list = []
    recap_articles: list = []
    chatter_articles: list = []
    real_articles: list = []
    for a in articles:
        if _looks_like_quote_widget(a):
            quote_widget_articles.append(a)
            continue
        hit, _name = _looks_like_recap_template(a)
        if hit:
            recap_articles.append(a)
            continue
        # Raw stocktwits forum chatter pre-floor (defense-in-depth — the
        # storage-side prefloor_pseudo_articles helper does the same check
        # earlier in the ML path; this catches anything that slipped through
        # an alternate entry path — see _looks_like_stocktwits_chatter docstring
        # in alert_agent for the live evidence + discriminator).
        if _looks_like_stocktwits_chatter(a):
            chatter_articles.append(a)
            continue
        real_articles.append(a)

    if quote_widget_articles:
        qw_updates: list[tuple[str, float, int]] = [
            (a["_id"], 0.01, 0) for a in quote_widget_articles if a.get("_id")
        ]
        if qw_updates:
            store.update_ai_scores_batch(qw_updates)
        _log.info(
            f"[urgency] pre-floored {len(quote_widget_articles)} quote-widget "
            f"row(s) — skipped Sonnet call (urgency head over-scores live "
            f"ticker-tape; would have polluted training pool as score_source="
            f"'llm')"
        )

    if recap_articles:
        recap_updates: list[tuple[str, float, int]] = [
            (a["_id"], 0.01, 0) for a in recap_articles if a.get("_id")
        ]
        if recap_updates:
            store.update_ai_scores_batch(recap_updates)
        _log.info(
            f"[urgency] pre-floored {len(recap_articles)} recap-template "
            f"row(s) — skipped Sonnet call (would have over-scored to 8+)"
        )

    if chatter_articles:
        chatter_updates: list[tuple[str, float, int]] = [
            (a["_id"], 0.01, 0) for a in chatter_articles if a.get("_id")
        ]
        if chatter_updates:
            store.update_ai_scores_batch(chatter_updates)
        _log.info(
            f"[urgency] pre-floored {len(chatter_articles)} stocktwits-chatter "
            f"row(s) — skipped Sonnet call (forum chatter, urgency head "
            f"over-scores $TICKER + held-name density on short titles)"
        )

    if not real_articles:
        return 0

    # The remainder of this function (Sonnet prompt construction + per-index
    # mapping) operates on the live payload only. Rebinding here keeps the
    # downstream index math (truncation guard, anti-loop floor) intact without
    # an additional remap layer.
    articles = real_articles

    age_hours_map: dict[int, float] = {i: _article_age_hours(a) for i, a in enumerate(articles)}
    payload = [
        {
            "index": i,
            "age_hours": round(age_hours_map[i], 1),
            "title": (a.get("title") or "")[:200],
            "summary": (a.get("summary") or "")[:300],
        }
        for i, a in enumerate(articles)
    ]

    prompt = SCORE_PROMPT.format(
        articles_json=json.dumps(payload, ensure_ascii=False),
        portfolio_tickers=_portfolio_ticker_line(),
    )

    try:
        raw = claude_call(prompt, model=SONNET_MODEL, timeout=120)
        if raw is None:
            return 0

        scores = extract_json_array(raw)
        if scores is None:
            _log.warning(f"[urgency] Failed to parse JSON array from response: {raw[:200]!r}")
            return 0

        urgent_count = 0
        updates: list[tuple[str, float, int]] = []
        urgent_log: list[tuple[float, str, str]] = []
        scored_indices: set[int] = set()
        for item in scores:
            if not isinstance(item, dict):
                continue
            try:
                idx = int(item.get("index"))
                score = float(item.get("score", 0))
            except (TypeError, ValueError):
                continue
            if idx < 0 or idx >= len(articles):
                continue
            art = articles[idx]
            aid = art.get("_id")
            if not aid:
                continue
            # Hard staleness cap: articles older than STALE_HOURS cannot be urgent
            # regardless of time-sensitive language in their text ("today", "breaking").
            age_h = age_hours_map.get(idx, 0.0)
            if age_h > STALE_HOURS and score > STALE_SCORE_CAP:
                _log.debug(f"[urgency] Capping stale article (age={age_h:.1f}h) score {score:.1f}→{STALE_SCORE_CAP}")
                score = STALE_SCORE_CAP
            is_urgent = score >= URGENT_THRESHOLD
            # Floor non-urgent ai_score at 0.01 so Sonnet-scored noise (score=0)
            # isn't picked up by get_unscored on the next pass and re-sent to
            # the LLM forever. Clamp the upper bound to 10 defensively.
            score = max(0.01, min(float(score), 10.0))
            updates.append((aid, score, 1 if is_urgent else 0))
            scored_indices.add(idx)
            if is_urgent:
                urgent_log.append((score, (art.get("title") or "")[:80], item.get("reason", "")))
                urgent_count += 1
        # Anti-loop: articles Sonnet engaged with but omitted from its response
        # would otherwise remain ai_score=0 / ml_score=NULL and be re-routed to
        # the LLM every cycle forever. Floor them to 0.01 (noise) so they exit
        # the queue. Only apply when Sonnet returned at least one valid entry —
        # a completely empty array is ambiguous (refusal vs. true zero) and we
        # prefer a retry over mass-mislabeling 100 articles as noise.
        # Truncation guard: if the response is a clean prefix (indices
        # 0..max_idx with no internal gaps) and the missing tail is longer
        # than a model would plausibly omit on purpose, treat it as a
        # token-limit truncation and re-queue the tail (leave ai_score=0)
        # instead of flooring genuine articles to noise forever.
        truncated_tail = False
        if scored_indices:
            max_idx = max(scored_indices)
            is_clean_prefix = scored_indices == set(range(max_idx + 1))
            tail_len = len(articles) - (max_idx + 1)
            if is_clean_prefix and tail_len > MAX_TRAILING_OMISSIONS:
                truncated_tail = True
                _log.warning(
                    f"[urgency] Response looks truncated: got clean prefix "
                    f"0..{max_idx} of {len(articles)}, re-queuing {tail_len} "
                    f"tail articles instead of flooring them to noise"
                )
        if scored_indices and not truncated_tail:
            for i, art in enumerate(articles):
                if i in scored_indices:
                    continue
                aid = art.get("_id")
                if not aid:
                    continue
                updates.append((aid, 0.01, 0))
        if updates:
            # One bulk commit instead of N round-trips through the write lock.
            store.update_ai_scores_batch(updates)
        for score, title, reason in urgent_log:
            _log.info(f"[urgency] URGENT score={score:.0f} — {title} ({reason})")

        return urgent_count
    except Exception:
        _log.exception("[urgency] Scoring error")
        return 0


def run_scoring_pass(store, batch_size: int = BATCH_SIZE) -> int:
    """Score all unscored articles in the store. Returns total urgent found."""
    unscored = store.get_unscored(limit=batch_size * 5)
    if not unscored:
        return 0

    _log.info(f"[urgency] Scoring {len(unscored)} unscored articles in batches of {batch_size}...")
    total_urgent = 0
    for i in range(0, len(unscored), batch_size):
        batch = unscored[i:i + batch_size]
        total_urgent += score_batch(batch, store)

    return total_urgent
