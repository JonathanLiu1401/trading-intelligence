"""
Multi-dimensional heuristic scoring engine.

Final score = keyword_score * source_weight * event_bonus * recency_factor
Range: 0.0 – 10.0

Dimensions:
  1. Keyword relevance  — tiered term matching (portfolio > memory > semis > macro > general)
  2. Source authority   — Reuters/Bloomberg/WSJ/CNBC > blogs > Reddit
  3. Event multiplier   — pattern-matched high-value events (earnings, ratings, guidance, M&A, supply)
  4. Recency decay      — exponential decay; articles >48h old are downweighted
  5. Hard blacklist     — zero-score on irrelevant topics
  6. Portfolio boost    — direct ticker mentions get a flat boost
"""
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

# ── Portfolio tickers (direct mention = highest priority) ───────────────────
# Short tickers must match as whole words (regex \b); naive substring match
# was triggering false positives like "satellite"→"lite", "museum"→"mu".
PORTFOLIO_TICKERS = {"lite", "mu", "msft", "axti", "orcl", "tsem", "qbts"}
PORTFOLIO_NAMES = {"micron", "lumentum", "microsoft", "oracle", "tower semiconductor"}
_PORTFOLIO_TICKER_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(t) for t in PORTFOLIO_TICKERS) + r")\b",
    re.I,
)
# Back-compat alias for any external readers.
PORTFOLIO = PORTFOLIO_TICKERS | PORTFOLIO_NAMES

# ── Keyword tiers ────────────────────────────────────────────────────────────
# Phrase terms — safe substring matches (unique enough not to false-positive).
TIER_PORTFOLIO_PHRASES = {  # 4 pts each — direct portfolio/focus names
    "micron", "sk hynix", "kioxia", "western digital", "lumentum", "tower semi",
    "dram asp", "nand pricing", "hbm3", "hbm2e", "lpddr5",
    "memory pricing", "dram supply", "nand oversupply",
}
# Bare tickers — must match on word boundaries; substring matching produced
# false positives ("mu"→"museum") and false negatives ("MU." at end of string
# was missed by the old " mu " space-padded hack).
TIER_PORTFOLIO_TICKERS = {"mu", "wdc", "stx", "lrcx", "amat", "klac", "axti", "orcl", "qbts"}
_TIER_PORTFOLIO_TICKER_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(t) for t in TIER_PORTFOLIO_TICKERS) + r")\b",
    re.I,
)
# Back-compat: legacy callers may import TIER_PORTFOLIO as a flat set.
TIER_PORTFOLIO = TIER_PORTFOLIO_PHRASES | TIER_PORTFOLIO_TICKERS

TIER_MEMORY = {  # 3 pts each — kioxia intentionally NOT here (already in TIER_PORTFOLIO)
    "dram", "nand", "hbm", "lpddr", "memory chip", "flash storage",
    "wafer start", "bit growth", "capex intensity", "memory maker",
    "samsung memory", "samsung semiconductor",
}
TIER_SEMIS = {  # 2 pts each
    "semiconductor", "chip", "asml", "nvidia", "tsmc", "amd", "intel",
    "qualcomm", "marvell", "broadcom", "euv", "foundry", "fab",
    "chip equipment", "packaging", "advanced packaging",
    "ai chip", "gpu", "hpc", "data center chip",
}
TIER_MACRO = {  # 1.5 pts each
    "federal reserve", "fomc", "rate cut", "rate hike", "rate decision",
    "inflation", "cpi report", "pce inflation", "nonfarm payroll",
    "gdp growth", "recession risk", "yield curve", "treasury yield",
    "tariff", "export control", "china ban", "chip war", "sanctions",
    "bitcoin", "ethereum", "crypto market",
    "nikkei", "hang seng", "kospi", "dax", "ftse", "shanghai",
}
TIER_GENERAL = {  # 0.5 pts each
    "stock market", "equity", "earnings", "revenue", "guidance",
    "s&p 500", "nasdaq", "dow jones", "russell", "vix",
    "oil price", "gold price", "dollar index", "dxy",
    "interest rate", "central bank", "ecb", "bank of japan",
    "ipo", "merger", "acquisition", "buyback",
}

# ── Hard blacklist — zero score immediately ──────────────────────────────────
BLACKLIST = re.compile(
    r"\b(nfl|nba|mlb|nhl|premier league|formula[- ]?1|f1 race|ufc|"
    r"celebrity|kardashian|oscars|grammy|billboard|taylor swift|"
    r"recipe|lifestyle|wellness|skincare|fashion week|zodiac|horoscope|"
    r"travel deal|hotel review|restaurant|real estate listing|"
    r"lottery|gambling|casino|sports bet|"
    r"obituary|funeral|wedding|divorce settlement|"
    r"weather forecast|hurricane|earthquake damage)\b",
    re.I
)

# ── Event pattern detection — multiplier bonuses ────────────────────────────
EVENT_PATTERNS = [
    # Earnings
    (re.compile(r"\beps\b.{0,30}\bvs\b|\bearnings (beat|miss|surpass|disappoint)\b", re.I), 2.5, "earnings"),
    (re.compile(r"\b(q[1-4] \d{4}|quarterly) (result|earning|revenue|profit)", re.I), 1.8, "earnings"),
    (re.compile(r"\b(beat|miss|exceed|below).{0,20}(estimate|expect|consensus|forecast)\b", re.I), 2.0, "earnings"),
    # Guidance
    (re.compile(r"\b(raise|cut|lower|withdraw|reaffirm).{0,20}(guidance|outlook|forecast)\b", re.I), 2.3, "guidance"),
    (re.compile(r"\b(above|below).{0,20}(consensus|street|estimate)\b", re.I), 1.9, "guidance"),
    # Analyst ratings
    (re.compile(r"\b(upgrade|downgrade).{0,30}(buy|sell|hold|neutral|outperform|underperform)\b", re.I), 2.2, "rating"),
    (re.compile(r"\bprice target.{0,20}\$\d+|\braises? pt\b|\bcuts? pt\b", re.I), 1.8, "rating"),
    (re.compile(r"\b(initiat|resuming coverage|reiterat).{0,20}(buy|sell|hold)\b", re.I), 1.6, "rating"),
    # M&A
    (re.compile(r"\b(acqui|merger|buyout|takeover|bid for|deal with).{0,30}(billion|million|\$\d)\b", re.I), 2.4, "m&a"),
    # Supply chain / production
    (re.compile(r"\b(shortage|glut|oversupply|capacity cut|production halt|fab delay)\b", re.I), 2.0, "supply"),
    (re.compile(r"\b(wafer|bit|chip).{0,15}(shortage|surplus|cut|increase)\b", re.I), 2.2, "supply"),
    # Macro shocks
    (re.compile(r"\b(emergency|surprise|unexpected|shock).{0,20}(rate|cut|hike|decision)\b", re.I), 2.8, "macro_shock"),
    (re.compile(r"\b(circuit breaker|trading halt|market crash|flash crash)\b", re.I), 3.0, "crisis"),
    # Export controls / sanctions
    (re.compile(r"\b(export (ban|control|restrict)|entity list|blacklist).{0,30}(chip|semi|memory)\b", re.I), 2.5, "regulatory"),
    (re.compile(r"\b(sanction|restrict).{0,20}(china|huawei|smic|yangtze)\b", re.I), 2.3, "regulatory"),
    # ASP / pricing moves
    (re.compile(r"\b(asp|average selling price).{0,20}(\d+%|rise|fall|drop|jump)\b", re.I), 2.1, "pricing"),
    (re.compile(r"\b(dram|nand|hbm).{0,20}(price|pricing).{0,20}(\d+%|up|down|surge|plunge)\b", re.I), 2.4, "pricing"),
]

# ── Source authority weights ─────────────────────────────────────────────────
SOURCE_WEIGHTS = {
    # Tier A — wire services and major financial press
    "reuters": 1.4, "bloomberg": 1.4, "wsj": 1.35, "financial times": 1.35,
    "ft.com": 1.35, "cnbc": 1.3, "associated press": 1.3, "ap ": 1.3,
    "nikkei": 1.3, "koreaherald": 1.25, "korea herald": 1.25,
    "scmp": 1.2, "south china morning": 1.2,
    # Tier B — quality financial media
    "marketwatch": 1.2, "barrons": 1.2, "seeking alpha": 1.15,
    "benzinga": 1.15, "thestreet": 1.1, "investors.com": 1.15,
    "zacks": 1.1, "finviz": 1.1, "marketbeat": 1.1,
    "theblock": 1.15, "coindesk": 1.1,
    "gdelt": 1.0,  # GDELT is a news aggregator — neutral
    # Tier C — social / scraped
    "reddit": 0.75, "scraped": 0.8, "yfinance": 0.9,
    "twitter": 0.7, "stocktwits": 0.65,
}
DEFAULT_SOURCE_WEIGHT = 0.95


def _source_weight(source: str) -> float:
    s = source.lower()
    for key, w in SOURCE_WEIGHTS.items():
        if key in s:
            return w
    return DEFAULT_SOURCE_WEIGHT


def _recency_factor(published: str) -> float:
    """Exponential decay: 1.0 at t=0, ~0.5 at 12h, ~0.25 at 24h, ~0.1 at 48h."""
    if not published:
        return 0.85  # unknown age — slight penalty
    try:
        # try RFC 2822 (RSS dates)
        dt = parsedate_to_datetime(published)
        age_h = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
    except Exception:
        try:
            # try ISO format (GDELT seendate: 20260513T031500Z)
            s = published.replace("Z", "+00:00")
            dt = datetime.fromisoformat(s)
            age_h = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
        except Exception:
            return 0.85
    if age_h < 0:
        age_h = 0
    # decay: e^(-0.06 * hours)
    import math
    return max(0.1, math.exp(-0.06 * age_h))


def _portfolio_boost(text: str) -> float:
    """Extra additive boost if a portfolio ticker or company name is directly mentioned.

    Tickers are matched on word boundaries (e.g. \"mu\" matches \"MU beats\" but
    not \"museum\"); company names are matched as substrings (they are unique
    enough that substring matches are safe).
    """
    t = text.lower()
    if _PORTFOLIO_TICKER_RE.search(t):
        return 1.5
    for name in PORTFOLIO_NAMES:
        if name in t:
            return 1.5
    return 0.0


def score_article(title: str, summary: str, source: str = "", published: str = "") -> dict:
    """
    Score a single article. Returns dict with score and breakdown.
    """
    text = f"{title} {summary}".lower()

    # Hard blacklist
    if BLACKLIST.search(text):
        return {"score": 0.0, "reason": "blacklisted", "events": []}

    # Keyword score
    kw = 0.0
    for term in TIER_PORTFOLIO_PHRASES:
        if term in text:
            kw += 4.0
    # Bare tickers: count each unique ticker match once with word boundaries.
    for _ in _TIER_PORTFOLIO_TICKER_RE.findall(text):
        kw += 4.0
        break  # one boost per article, mirroring prior single-add semantics
    for term in TIER_MEMORY:
        if term in text:
            kw += 3.0
    for term in TIER_SEMIS:
        if term in text:
            kw += 2.0
    for term in TIER_MACRO:
        if term in text:
            kw += 1.5
    for term in TIER_GENERAL:
        if term in text:
            kw += 0.5

    if kw == 0.0:
        return {"score": 0.0, "reason": "no_keywords", "events": []}

    # Event detection
    event_bonus = 1.0
    events_found = []
    for pattern, multiplier, event_name in EVENT_PATTERNS:
        if pattern.search(text):
            event_bonus = max(event_bonus, multiplier)
            events_found.append(event_name)

    # Source authority
    src_w = _source_weight(source)

    # Recency
    rec = _recency_factor(published)

    # Portfolio boost (additive)
    port_boost = _portfolio_boost(text)

    # Composite
    raw = (kw * src_w * event_bonus * rec) + port_boost

    # Normalise to 0-10
    score = min(10.0, round(raw / 4.0 * 10.0, 2))  # 4.0 = rough "max normal" kw

    return {
        "score": score,
        "kw": round(kw, 1),
        "src_weight": src_w,
        "event_bonus": event_bonus,
        "recency": round(rec, 2),
        "port_boost": port_boost,
        "events": events_found,
        "reason": "scored",
    }


def score_and_rank(articles: list, min_score: float = 1.5, top_n: int = 200) -> list:
    """
    Score all articles, drop noise, return top_n sorted by score desc.
    Adds _relevance_score and _score_detail to each article.
    """
    scored = []
    dropped = 0
    for art in articles:
        result = score_article(
            art.get("title", ""),
            art.get("summary", ""),
            art.get("source", ""),
            art.get("published", ""),
        )
        if result["score"] < min_score:
            dropped += 1
            continue
        art["_relevance_score"] = result["score"]
        art["_score_detail"] = result
        scored.append(art)

    scored.sort(key=lambda a: a["_relevance_score"], reverse=True)
    print(f"[heuristic] {len(articles)} in → {len(scored)} pass filter "
          f"({dropped} dropped) → top {min(top_n, len(scored))} to Claude")
    return scored[:top_n]


if __name__ == "__main__":
    tests = [
        {"title": "Micron raises DRAM ASP guidance 20% on HBM3E demand surge", "summary": "Q2 beat, EPS $1.42 vs $1.18E", "source": "Reuters", "published": ""},
        {"title": "Federal Reserve surprises with emergency 50bp rate cut", "summary": "Unexpected FOMC action amid recession fears", "source": "Bloomberg", "published": ""},
        {"title": "SK Hynix upgrades to Buy, PT raised to $240", "summary": "Analyst cites HBM supply discipline", "source": "CNBC", "published": ""},
        {"title": "NAND flash oversupply glut worsens, Kioxia cuts wafer starts 30%", "summary": "Bit growth forecast slashed", "source": "Nikkei", "published": ""},
        {"title": "Celebrity breaks up with boyfriend", "summary": "Hollywood gossip", "source": "TMZ", "published": ""},
        {"title": "S&P 500 up 0.3% on mixed data", "summary": "Stocks drift higher", "source": "MarketWatch", "published": ""},
        {"title": "Bitcoin surges 8% past $100k as ETF inflows accelerate", "summary": "Crypto rally on risk-on sentiment", "source": "CoinDesk", "published": ""},
        {"title": "China imposes new export controls on advanced memory equipment", "summary": "DRAM manufacturers face supply chain disruption", "source": "Reuters", "published": ""},
    ]
    results = score_and_rank(tests, min_score=1.0, top_n=10)
    print()
    for a in results:
        d = a["_score_detail"]
        print(f"  {a['_relevance_score']:>5.1f}  [{','.join(d['events']) or 'kw'}]  {a['title'][:70]}")
