"""Adaptive relevance overlay driven by realized post-article returns.

The heuristic scorer is allowed to parse text, but it should not be the final
judge of relevance. This module turns historical article outcomes into learned
feature weights and applies them as a multiplier over the baseline score.

Cache format:
{
  "version": 1,
  "updated_at": "...",
  "weights": {
    "source:googlenews": {"n": 42, "boost": 1.18, "mean_abs_return_pct": 2.4},
    "ticker:NVDA": {"n": 31, "boost": 1.25, "mean_abs_return_pct": 3.1},
    "token:ai capex": {"n": 12, "boost": 1.35, "mean_abs_return_pct": 4.2}
  }
}
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
CACHE_PATH = Path(os.environ.get(
    "ARTICLE_RELEVANCE_EDGE_CACHE",
    str(BASE_DIR / "data" / "adaptive_relevance.json"),
))
CACHE_TTL_S = 300.0
MIN_WEIGHT_N = int(os.environ.get("ARTICLE_RELEVANCE_MIN_WEIGHT_N", "5"))

_cache_loaded_at = 0.0
_cache: dict | None = None

_TICKER_RE = re.compile(r"(?:\$|\b)([A-Z]{1,5})(?:\b)")
_NOT_TICKERS = {
    "AI", "THE", "AND", "FOR", "CEO", "CFO", "EPS", "GDP", "USA", "SEC",
    "FED", "FOMC", "CPI", "PCE", "PMI", "ISM", "IPO", "ETF", "ETFS",
}
_TOKEN_PATTERNS = {
    "ai capex": re.compile(r"\bAI\b.{0,40}\b(capex|capital expenditure|spending)\b", re.I),
    "data center": re.compile(r"\bdata centers?\b", re.I),
    "private credit": re.compile(r"\bprivate credit\b", re.I),
    "debt financing": re.compile(r"\b(debt|bond|loan|lease|financing|refinancing|SPV)\b", re.I),
    "guidance cut": re.compile(r"\b(cut|lower|miss|weak).{0,30}\b(guidance|outlook|forecast)\b", re.I),
    "selloff": re.compile(r"\b(selloff|sell-off|plunge|tumble|slump|correction)\b", re.I),
    "recession": re.compile(r"\b(recession|slowdown|unemployment|credit spread)\b", re.I),
    "earnings": re.compile(r"\b(earnings|revenue|EPS|margin)\b", re.I),
}


def _load_cache() -> dict:
    global _cache, _cache_loaded_at
    now = time.monotonic()
    if _cache is not None and now - _cache_loaded_at < CACHE_TTL_S:
        return _cache
    try:
        raw = json.loads(CACHE_PATH.read_text())
        if not isinstance(raw, dict):
            raw = {}
    except Exception:
        raw = {}
    _cache = raw
    _cache_loaded_at = now
    return raw


def _source_family(source: str | None) -> str:
    s = (source or "").strip().lower()
    if not s:
        return "unknown"
    return re.sub(r"_\d{4}-\d{2}(?:-\d{2})?$", "", s.split("/", 1)[0].strip())


def _features(title: str, summary: str, source: str) -> list[str]:
    text = f"{title or ''} {summary or ''}"
    feats = [f"source:{_source_family(source)}"]
    for ticker in sorted(set(_TICKER_RE.findall(text.upper()))):
        if ticker not in _NOT_TICKERS:
            feats.append(f"ticker:{ticker}")
    for name, pattern in _TOKEN_PATTERNS.items():
        if pattern.search(text):
            feats.append(f"token:{name}")
    return feats


def apply_adaptive_return_boost(
    base_score: float,
    title: str,
    summary: str,
    source: str = "",
) -> dict:
    """Return score adjusted by learned post-article return impact.

    The boost combines matching feature multipliers by sample-size-weighted
    log average so one noisy feature cannot dominate the whole score.
    """
    try:
        base = float(base_score)
    except (TypeError, ValueError):
        base = 0.0
    cache = _load_cache()
    weights = cache.get("weights") if isinstance(cache, dict) else None
    if not isinstance(weights, dict) or not weights:
        return {
            "score": round(max(0.0, min(10.0, base)), 2),
            "multiplier": 1.0,
            "features": [],
            "reason": "adaptive_cache_missing",
        }

    matched = []
    log_sum = 0.0
    n_sum = 0.0
    for feat in _features(title, summary, source):
        row = weights.get(feat)
        if not isinstance(row, dict):
            continue
        try:
            n = float(row.get("n") or 0.0)
            boost = float(row.get("boost") or 1.0)
        except (TypeError, ValueError):
            continue
        if n < MIN_WEIGHT_N or boost <= 0:
            continue
        boost = max(0.35, min(2.75, boost))
        w = min(25.0, n) ** 0.5
        log_sum += math.log(boost) * w
        n_sum += w
        matched.append({
            "feature": feat,
            "n": int(n),
            "boost": round(boost, 3),
            "mean_abs_return_pct": row.get("mean_abs_return_pct"),
            "mean_abnormal_return_pct": row.get("mean_abnormal_return_pct"),
        })

    if not matched or n_sum <= 0:
        return {
            "score": round(max(0.0, min(10.0, base)), 2),
            "multiplier": 1.0,
            "features": [],
            "reason": "no_adaptive_feature_match",
        }

    multiplier = max(0.35, min(2.75, math.exp(log_sum / n_sum)))
    adjusted = max(0.0, min(10.0, base * multiplier))
    return {
        "score": round(adjusted, 2),
        "multiplier": round(multiplier, 3),
        "features": matched[:8],
        "reason": "adaptive_return_boost",
    }
