"""Extra numeric features appended to the TF-IDF vector before the neural net.

Shape contract: ``extract_features_batch`` always returns ``(N, EXTRA_FEATURE_DIM)``
float32. The trainer concatenates this to the TF-IDF dense matrix. The model
automatically rebuilds when ``input_dim`` changes (see ml/model.py).

Features (15 dims):
   0   source_credibility       — 0..1, Reuters≈1.0, Reddit≈0.4, unknown=0.55
   1   ticker_mention_density   — portfolio tickers / total words, clipped 0..0.1
   2   hour_sin                 — temporal cyclic encoding
   3   hour_cos                 — temporal cyclic encoding
   4   dow_sin                  — day-of-week cyclic encoding
   5   dow_cos                  — day-of-week cyclic encoding
   6   days_since_published     — clipped 0..30, normalized to 0..1
   7   title_caps_ratio         — fraction of all-caps tokens in title (urgency proxy)
   8   has_question             — 0 or 1
   9   has_exclamation          — 0 or 1
  10   log_sentence_count       — log(1+count)/log(50), proxy for article depth
  11   log_named_entity_count   — log(1+count)/log(50), capitalized-word runs
  12   portfolio_flag           — 1 if any live-position ticker mentioned
  13   ticker_count             — log(1+count)/log(20), distinct portfolio tickers seen
  14   text_length              — log(1+chars)/log(5000), rough size signal
"""
from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Iterable

import numpy as np


EXTRA_FEATURE_DIM = 15

# Source credibility — broader coverage; mirrors urgency_scorer + briefing analyst weights.
SOURCE_CRED: dict[str, float] = {
    "reuters": 0.90, "bloomberg": 0.90, "wsj": 0.88, "financial times": 0.88,
    "ft.com": 0.88, "cnbc": 0.85, "associated press": 0.85, "ap": 0.85,
    "nikkei": 0.85, "koreaherald": 0.80, "korea herald": 0.80,
    "scmp": 0.78, "south china morning": 0.78,
    "marketwatch": 0.78, "barrons": 0.78, "seeking alpha": 0.72,
    "benzinga": 0.72, "thestreet": 0.68, "investors.com": 0.72,
    "zacks": 0.68, "finviz": 0.68, "marketbeat": 0.68,
    "theblock": 0.72, "coindesk": 0.68, "decrypt": 0.62,
    "sec edgar": 0.95, "sec_edgar": 0.95, "googlenews": 0.62,
    "google_news": 0.62, "yahoo": 0.65, "yfinance": 0.65,
    "gdelt": 0.58, "scraped": 0.50, "rss": 0.65,
    "reddit": 0.40, "twitter": 0.35, "stocktwits": 0.30,
    "nitter": 0.40, "substack": 0.65, "wikipedia": 0.60,
    "finnhub": 0.78, "polygon": 0.80, "newsapi": 0.65,
    "alphavantage": 0.72,
}
DEFAULT_SOURCE_CRED = 0.55

# Precompiled word-boundary patterns for source credibility lookup.
# Naive `if key in s` matched "ap " inside "snap " (e.g., a Snap-related source
# tagged with AP credibility). Word boundaries eliminate that whole class of bug.
_SOURCE_CRED_PATTERNS: list[tuple[re.Pattern, float]] = [
    (re.compile(r"\b" + re.escape(k.strip()) + r"\b", re.IGNORECASE), v)
    for k, v in SOURCE_CRED.items()
]

# Tickers that count as "live position" for the portfolio_relevance feature.
LIVE_PORTFOLIO_TICKERS = {
    "LITE", "LNOK", "MUU", "DRAM", "SNDU", "MU", "NVDA",
    "MSFT", "AXTI", "ORCL", "TSEM", "QBTS",
}
_LIVE_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(t) for t in LIVE_PORTFOLIO_TICKERS) + r")\b"
)
# Word tokens for ticker-density denominator.
_WORD_RE = re.compile(r"\b\w+\b")
# Run of two or more consecutive Capitalized words — a coarse named-entity proxy
# without spaCy. Avoids matching ALLCAPS tickers (require lowercase tail).
_NER_RE = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")
# Sentence boundaries — terminator followed by whitespace.
_SENT_RE = re.compile(r"[.!?]+\s+")


def _source_credibility(source: str) -> float:
    if not source:
        return DEFAULT_SOURCE_CRED
    for pat, score in _SOURCE_CRED_PATTERNS:
        if pat.search(source):
            return score
    return DEFAULT_SOURCE_CRED


def _parse_published(published: str) -> datetime | None:
    """Parse a published-date string and normalise it to UTC.

    Feeds emit their publish instant in their own timezone — Nikkei in JST
    (``+0900``), US wires in EST (``-0500``), most others in UTC. The hour /
    day-of-week cyclic features (indices 2..5) derive from ``dt.hour`` and
    ``dt.weekday()``; without normalisation the *same* publishing instant
    produces a different ``hour_sin``/``dow_sin`` per source (a -0500 feed
    can even land on the previous weekday). That is pure noise injected into
    4 of 15 extra features and a train/serve skew, since the trainer and
    live inference both feed raw ``published`` strings. Converting every
    parsed datetime to UTC here makes "hour of day" a stable signal the
    model can actually learn. A naive datetime (RFC string without an
    offset) is assumed UTC — same convention as the rest of the pipeline
    (``urgency_scorer``, ``alert_agent``, ``heuristic_scorer``)."""
    if not published:
        return None
    dt: datetime | None = None
    try:
        dt = parsedate_to_datetime(published)
    except Exception:
        dt = None
    if dt is None:
        try:
            dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
        except Exception:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _ticker_density(text: str) -> tuple[float, int, float]:
    """Return (density 0..0.1, distinct-ticker-count, portfolio_flag 0/1)."""
    if not text:
        return 0.0, 0, 0.0
    matches = _LIVE_RE.findall(text)
    n_words = len(_WORD_RE.findall(text)) or 1
    distinct = len({m.upper() for m in matches})
    density = min(len(matches) / n_words, 0.1)
    return density, distinct, (1.0 if matches else 0.0)


def _title_caps_ratio(title: str) -> float:
    """Fraction of tokens in title that are entirely uppercase (>= 2 chars)."""
    if not title:
        return 0.0
    tokens = _WORD_RE.findall(title)
    if not tokens:
        return 0.0
    caps = sum(1 for t in tokens if len(t) >= 2 and t.isupper())
    return caps / len(tokens)


def _sentence_count(text: str) -> int:
    if not text:
        return 0
    return len(_SENT_RE.split(text))


def _named_entity_count(text: str) -> int:
    if not text:
        return 0
    return len(_NER_RE.findall(text))


def extract_features(article: dict) -> np.ndarray:
    feats = np.zeros(EXTRA_FEATURE_DIM, dtype=np.float32)

    title   = article.get("title", "") or ""
    summary = article.get("summary", "") or ""
    text    = f"{title} {summary}"

    # 0 — source credibility
    feats[0] = _source_credibility(article.get("source", ""))

    # 1 — ticker mention density (also used below for flag/count features)
    density, distinct_tickers, portfolio = _ticker_density(text)
    feats[1] = density

    # 2..6 — temporal features
    dt = _parse_published(article.get("published", ""))
    if dt is None:
        dt = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    hour_angle = 2 * math.pi * (dt.hour / 24.0)
    feats[2] = math.sin(hour_angle)
    feats[3] = math.cos(hour_angle)
    dow_angle = 2 * math.pi * (dt.weekday() / 7.0)
    feats[4] = math.sin(dow_angle)
    feats[5] = math.cos(dow_angle)
    age_days = max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0)
    feats[6] = min(age_days, 30.0) / 30.0

    # 7 — title caps ratio (urgency correlate)
    feats[7] = _title_caps_ratio(title)

    # 8/9 — punctuation flags
    feats[8] = 1.0 if "?" in title else 0.0
    feats[9] = 1.0 if "!" in title else 0.0

    # 10 — sentence count (log-scaled, clipped)
    feats[10] = min(math.log1p(_sentence_count(summary)) / math.log(50.0), 1.0)

    # 11 — named entity count proxy
    feats[11] = min(math.log1p(_named_entity_count(text)) / math.log(50.0), 1.0)

    # 12 — portfolio_flag
    feats[12] = portfolio

    # 13 — distinct portfolio ticker count
    feats[13] = min(math.log1p(distinct_tickers) / math.log(20.0), 1.0)

    # 14 — overall text length
    feats[14] = min(math.log1p(len(text)) / math.log(5000.0), 1.0)

    # No global clip — sin/cos features (indices 2..5) live in [-1, 1].
    return feats.astype(np.float32)


def extract_features_batch(articles: Iterable[dict]) -> np.ndarray:
    rows = [extract_features(a) for a in articles]
    if not rows:
        return np.zeros((0, EXTRA_FEATURE_DIM), dtype=np.float32)
    return np.vstack(rows).astype(np.float32)
