"""TF-IDF vocabulary drift monitor.

Surfaces how stale the fitted embedder vocabulary is by measuring the share of
*terms* (unigrams + bigrams + trigrams — whatever the live vectorizer emits)
in recent articles that fall outside the current ``TfidfVectorizer.vocabulary_``.

High out-of-vocabulary (OOV) rates mean new tickers, slang, or topics have
entered the corpus that ``ArticleNet`` cannot represent — a *content-drift*
signal that complements ``ml.embedder.Embedder.should_refit`` (which only
triggers on corpus-*size* growth, never on content shift).

Read-only: no DB writes, no model writes, no refit. The live filter mirrors
``storage.article_store._LIVE_ONLY_CLAUSE`` so synthetic backtest / opus
annotation rows never poison the sample.

Honesty note: extraction is delegated to the vectorizer's own
``build_analyzer()`` so the measured terms are exactly the ones the model
would see — n-grams included. Unit tests pin the pure stat function with a
plain tokenizer; the production wrapper does the n-gram-aware extraction.

CLI::

    python3 -m ml.oov_monitor          # JSON report; exit 0 ok, 1 if drifted
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Callable, Iterable

# Average OOV ratio above this triggers a non-zero exit. A freshly-fit vocab
# on a representative corpus typically reports avg OOV in the 0.10-0.20 band
# (rare proper nouns, new ticker symbols); >0.35 indicates real drift worth
# a refit.
DRIFT_AVG_OOV = 0.35
SAMPLE_SIZE = 500
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "oov_monitor.json"

# Matches TfidfVectorizer's default token pattern (r"(?u)\b\w\w+\b"). Used
# only by tests and as a fallback when the embedder is unfitted; production
# extraction goes through ``vec.build_analyzer()``.
_TOKEN_RE = re.compile(r"\b\w\w+\b", flags=re.UNICODE)


def _default_analyzer(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def compute_oov_stats(
    sample_texts: Iterable[str],
    analyzer: Callable[[str], list[str]],
    vocab: set[str],
) -> dict:
    """Compute OOV statistics for ``sample_texts`` against ``vocab``.

    Pure function — no IO, no globals. ``analyzer`` must emit the same terms
    the live vectorizer would (unigrams + n-grams). Tests pass a simple
    tokenizer; production passes ``vec.build_analyzer()``.
    """
    ratios: list[float] = []
    total_terms = 0
    total_oov = 0
    for text in sample_texts:
        terms = analyzer(text)
        if not terms:
            continue
        oov = sum(1 for t in terms if t not in vocab)
        ratios.append(oov / len(terms))
        total_terms += len(terms)
        total_oov += oov

    if not ratios:
        return {
            "sample_size": 0,
            "vocab_size": len(vocab),
            "avg_oov_ratio": 0.0,
            "p95_oov_ratio": 0.0,
            "weighted_oov_ratio": 0.0,
            "total_terms": 0,
            "drifted": False,
        }

    ratios.sort()
    avg = sum(ratios) / len(ratios)
    # p95 with clamp so a single-sample audit returns the sample's ratio
    # rather than IndexError.
    p95_idx = min(len(ratios) - 1, int(len(ratios) * 0.95))
    p95 = ratios[p95_idx]
    weighted = total_oov / total_terms if total_terms else 0.0

    return {
        "sample_size": len(ratios),
        "vocab_size": len(vocab),
        "avg_oov_ratio": round(avg, 4),
        "p95_oov_ratio": round(p95, 4),
        "weighted_oov_ratio": round(weighted, 4),
        "total_terms": total_terms,
        "drifted": avg >= DRIFT_AVG_OOV,
    }


def sample_recent_titles(db_path: Path, n: int = SAMPLE_SIZE) -> list[str]:
    """Return up to ``n`` of the most-recent live-news titles."""
    conn = sqlite3.connect(
        f"file:{db_path}?mode=ro", uri=True, timeout=15
    )
    try:
        cur = conn.execute(
            "SELECT title FROM articles "
            "WHERE url NOT LIKE 'backtest://%' "
            "  AND source NOT LIKE 'backtest_%' "
            "  AND source NOT LIKE 'opus_annotation%' "
            "ORDER BY first_seen DESC LIMIT ?",
            (n,),
        )
        return [row[0] for row in cur.fetchall() if row[0]]
    finally:
        conn.close()


def _live_analyzer_and_vocab():
    """Return ``(analyzer, vocab)`` from the fitted embedder, or ``(None, set())``."""
    from ml.embedder import get_embedder

    emb = get_embedder()
    if not emb.fitted or emb._vec is None:
        return None, set()
    try:
        analyzer = emb._vec.build_analyzer()
        vocab = set(emb._vec.vocabulary_.keys())
    except Exception:
        return None, set()
    return analyzer, vocab


def audit() -> dict:
    """Run a full audit against the production embedder + DB. Read-only."""
    from storage.article_store import _get_db_path

    analyzer, vocab = _live_analyzer_and_vocab()
    if analyzer is None or not vocab:
        return {
            "sample_size": 0,
            "vocab_size": 0,
            "avg_oov_ratio": 0.0,
            "p95_oov_ratio": 0.0,
            "weighted_oov_ratio": 0.0,
            "total_terms": 0,
            "drifted": False,
            "note": "embedder not fitted",
        }
    sample = sample_recent_titles(_get_db_path())
    return compute_oov_stats(sample, analyzer, vocab)


def main(argv: list | None = None) -> int:
    report = audit()
    try:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True))
    except Exception:
        # Output dir is best-effort; never let it fail the CLI.
        pass
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if report.get("drifted") else 0


if __name__ == "__main__":
    sys.exit(main())
