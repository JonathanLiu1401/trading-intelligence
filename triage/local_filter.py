"""Triage filter — heuristic pre-filter then Claude Sonnet 4.6 batch re-ranking."""
import json

from core.claude_cli import claude_call
from core.json_extract import extract_json_array
from triage.heuristic_scorer import score_article, score_and_rank

KEYWORD_CANDIDATES = 200   # articles passed to Claude after heuristic pre-filter
TOP_N = 50                 # final articles forwarded to Opus for the briefing
MIN_HEURISTIC_SCORE = 1.5  # minimum heuristic score to be a candidate

SONNET_MODEL = "claude-sonnet-4-6"


def _keyword_score(text: str) -> float:
    """Thin wrapper around heuristic scorer for backward compatibility with daemon._ingest()."""
    parts = text.split(" ", 1)
    title = parts[0] if parts else text
    summary = parts[1] if len(parts) > 1 else ""
    return score_article(title, summary)["score"]


BATCH_PROMPT = """You are a financial news triage filter. Score each article 0-10 for relevance to this investment focus:
- Memory/semiconductor stocks: DRAM, NAND, HBM, Micron (MU), SK Hynix, Samsung, WDC, LRCX, AMAT, ASML, NVDA, AMD, TSMC
- Portfolio positions: LITE, MU, MSFT, AXTI, ORCL, TSEM, QBTS
- Global macro: Fed/FOMC, CPI, GDP, tariffs, China, Japan, Korea, Europe
- Crypto: BTC, ETH, SOL market moves

Scoring: 0=irrelevant, 5=general market/macro, 7=relevant to semis/portfolio, 9-10=critical memory/semis news.

Articles (JSON array with index, title, summary):
{articles_json}

Respond with ONLY a JSON array of objects: [{{"index": 0, "score": 7}}, ...]. No explanation."""


def _claude_batch_score(candidates: list) -> list:
    """Send up to KEYWORD_CANDIDATES articles to Sonnet for batch scoring.
    Returns articles with updated _relevance_score, sorted desc."""
    payload = [
        {"index": i, "title": a.get("title", "")[:150], "summary": (a.get("summary") or "")[:200]}
        for i, a in enumerate(candidates)
    ]
    prompt = BATCH_PROMPT.format(articles_json=json.dumps(payload, ensure_ascii=False))

    try:
        raw = claude_call(prompt, model=SONNET_MODEL, timeout=90)
        if raw is None:
            print("[local_filter] Sonnet unavailable — using heuristic scores only")
            return candidates

        scores = extract_json_array(raw)
        if scores is None:
            print("[local_filter] Could not parse Sonnet response — using heuristic scores")
            return candidates

        score_map = {
            item["index"]: item["score"]
            for item in scores
            if isinstance(item, dict) and "index" in item and "score" in item
        }

        for i, art in enumerate(candidates):
            sonnet_score = score_map.get(i)
            if sonnet_score is not None:
                # blend: keyword 20%, Sonnet 80%
                kw = art.get("_relevance_score", 0)
                art["_relevance_score"] = round(kw * 0.2 + sonnet_score * 0.8, 1)

        print(f"[local_filter] Sonnet scored {len(score_map)}/{len(candidates)} articles")
    except Exception as e:
        print(f"[local_filter] Sonnet error: {e}")

    return candidates


def filter_articles(articles: list) -> list:
    """
    Step 1: multi-dimensional heuristic pre-filter → top KEYWORD_CANDIDATES candidates
    Step 2: Claude Sonnet 4.6 batch re-ranking
    Step 3: return top TOP_N by score
    """
    if not articles:
        return []

    print(f"[local_filter] Heuristic scoring {len(articles)} articles...")
    candidates = score_and_rank(articles, min_score=MIN_HEURISTIC_SCORE, top_n=KEYWORD_CANDIDATES)

    if not candidates:
        print("[local_filter] No candidates passed heuristic filter")
        return []

    print(f"[local_filter] {len(candidates)} candidates → Sonnet 4.6 batch re-ranking...")
    candidates = _claude_batch_score(candidates)

    result = sorted(candidates, key=lambda a: a["_relevance_score"], reverse=True)[:TOP_N]
    print(f"[local_filter] Forwarding {len(result)} articles to Opus 4.7")
    return result


if __name__ == "__main__":
    sample = [
        {"title": "Micron raises DRAM ASP guidance 20% on HBM demand", "summary": "Memory demand surging"},
        {"title": "Fed holds rates, signals two cuts in 2026", "summary": "FOMC meeting recap"},
        {"title": "Celebrity divorce news", "summary": "Hollywood gossip"},
        {"title": "Bitcoin surges past $100k on ETF inflows", "summary": "Crypto rally"},
        {"title": "S&P 500 rallies 1.2% on strong earnings beats", "summary": "Markets react"},
    ]
    out = filter_articles(sample)
    for a in out:
        print(f"score={a['_relevance_score']:>5}  {a['title']}")
