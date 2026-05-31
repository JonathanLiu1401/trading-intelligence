"""YouTube-share-card SEO mill pre-floor — defense-in-depth gate for the noise pre-floor chain.

Google News indexes a tier of YouTube-video-style "stock analysis" pages
(GoogleNews/Mshale, GoogleNews/Fathom Journal, GoogleNews/fernandovasconcelos,
GoogleNews/fathomjournal.org) whose page titles carry a parenthesised opaque
YouTube video ID — 8-15 alphanumeric characters mixing lowercase + uppercase
+ digit. The ML urgency head over-scores them to ``ml_score >= 9`` because
the title prose is dense with held-ticker tokens ("$NVDA", "MSFT", "QBTS")
and clickbait verbs ("Surges", "CRASHES", "EXPOSED").

Live evidence (2026-05-30 articles.db pull, last 30 days, ``urgency >= 1``):
  * 293 total matches in the corpus
  * 6 reached ``urgency = 2`` (alerted state) — including
    "S&P 500 - Quadruple Top Airdrie Fc (8KsWofaZjy) - Mshale" (ml=9.7)
    "NVIDIA Stock Price Analysis ... Mortal Kombat (Tjm0z6tJEB) - Mshale" (ml=9.8)
    "Nvidia Stock (NVDA) Earnings Call ... Max Angioni (GtRnFn9fD5) - Mshale" (ml=9.2)
  * Source breakdown: 146 Mshale, 144 Fathom Journal, 3 other SEO mills.

The Mshale / Fathom Journal publishers sit above the 0.45
``ALERT_MIN_LONE_SOURCE_CRED`` bar (no entry in ``ml.features.SOURCE_CRED``
defaults them to 0.55) so the source-authority gate doesn't catch them;
content type IS the failure, same defense-in-depth class as
``watchers.alert_agent._looks_like_quote_widget`` /
``_looks_like_recap_template`` / ``_looks_like_stocktwits_chatter`` /
``watchers.non_english_filter.looks_non_english``.

Single source of truth — called from
``storage.article_store.prefloor_pseudo_articles`` (the daemon's ML-path
pre-floor) so YouTube-mill noise gets ``ml_score=0.01``, ``urgency=0``,
``score_source='ml'`` before it can reach urgency=1 — the same code path
the existing four siblings use.

Pure read-side: no DB write, no ai_score / ml_score / score_source /
urgency mutation in THIS module (the caller does the update via the
existing ``update_ml_scores_batch`` path). Backtest rows are already
filtered by the caller's upstream live-only clause. All four load-bearing
invariants intact by construction.
"""
from __future__ import annotations

import re


# YouTube-share-card fingerprint:
#   1. Parenthesised alphanumeric token of length 8-15 (lookahead anchors
#      the closing paren after exactly that many alnum chars — no separator
#      tolerance means real "(NASDAQ: NVDA)" / "(005930.KS)" / "(Q1 2026)"
#      with space/colon/dot are NEVER matched).
#   2. The token must contain AT LEAST ONE OF EACH: lowercase letter,
#      uppercase letter, and digit. Three additional lookaheads enforce
#      this. Real headline parentheticals (tickers "NVDA", exchanges
#      "NASDAQ", foreign symbols "005930KS", date qualifiers "2026Q1") all
#      lack at least one of these three character classes — verified zero
#      false positives across the 30-day live corpus AND a curated
#      must-survive headline set (see ``tests/test_youtube_mill_filter.py``).
#   3. The closing paren is followed by ``\s*-\s*\S`` (" - <Publisher>")
#      OR end-of-title. This anchors the fingerprint to the canonical
#      YouTube-share-card position — the parenthesised ID sits at the
#      END of the page-title, before the trailing publisher tag. A
#      mid-sentence parenthetical (a real headline that happens to embed
#      a similar-shape token like "(AlphaGo2)" mid-prose) does NOT match.
#
# Validated against the live ``articles.db`` corpus: 293 matches in 30
# days, all SEO-mill content; zero false positives on the curated
# must-survive corpus including legitimate parenthesised tickers/exchanges/
# quarters/products and accented headlines.
_YOUTUBE_MILL_TOKEN = re.compile(
    r"\((?=[A-Za-z0-9]{8,15}\))"  # opening paren + length-anchored alnum-only closer
    r"(?=[^)]*[a-z])"              # ...must contain lowercase before the closer
    r"(?=[^)]*[A-Z])"              # ...must contain uppercase before the closer
    r"(?=[^)]*\d)"                 # ...must contain a digit before the closer
    r"[A-Za-z0-9]+\)"
    r"(?:\s*-\s*\S|\s*$)"          # then " - <Publisher>" or end-of-title
)


def looks_like_youtube_mill(art: dict) -> bool:
    """True for a GoogleNews/Mshale-style YouTube-share-card SEO-mill row.

    The discriminator is a parenthesised 8-15 char alphanumeric token
    mixing lowercase + uppercase + digit, anchored at end-of-title (with
    an optional ``- Publisher`` tail). Pure, side-effect-free; reads only
    ``art['title']`` via ``.get()``."""
    title = (art.get("title") or "").strip()
    if not title:
        return False
    return bool(_YOUTUBE_MILL_TOKEN.search(title))


def filter_youtube_mill_noise(
    arts: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Partition rows into ``(kept, suppressed)`` where ``suppressed`` is
    every row whose title carries the YouTube-mill fingerprint. Mirrors
    ``watchers.non_english_filter.filter_non_english_noise`` /
    ``watchers.alert_agent._filter_quote_widget_noise`` so the pre-floor
    surfaces behave identically. Pure — no DB / IO."""
    kept: list[dict] = []
    suppressed: list[dict] = []
    for a in arts:
        (suppressed if looks_like_youtube_mill(a) else kept).append(a)
    return kept, suppressed
