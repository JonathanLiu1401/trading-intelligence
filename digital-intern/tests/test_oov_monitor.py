"""Pure tests for ml.oov_monitor — no DB, no embedder.

The pure ``compute_oov_stats`` function is the only piece worth unit-testing;
the CLI/IO glue follows the established ml/label_audit.py pattern (read-only
SQLite shim → JSON report → exit code) and is exercised by production runs.
"""
from __future__ import annotations

from ml import oov_monitor


# ── happy paths ─────────────────────────────────────────────────────────────
def test_all_terms_in_vocab_zero_oov():
    vocab = {"foo", "bar", "baz"}
    stats = oov_monitor.compute_oov_stats(
        ["foo bar", "baz baz foo"],
        oov_monitor._default_analyzer,
        vocab,
    )
    assert stats["sample_size"] == 2
    assert stats["avg_oov_ratio"] == 0.0
    assert stats["weighted_oov_ratio"] == 0.0
    assert stats["drifted"] is False


def test_full_oov_when_vocab_empty():
    stats = oov_monitor.compute_oov_stats(
        ["aapl mu micron"],
        oov_monitor._default_analyzer,
        set(),
    )
    assert stats["sample_size"] == 1
    assert stats["avg_oov_ratio"] == 1.0
    assert stats["weighted_oov_ratio"] == 1.0
    assert stats["drifted"] is True


# ── drift threshold ─────────────────────────────────────────────────────────
def test_drift_flag_when_avg_above_threshold():
    # "foo bar" → 1 in-vocab, 1 OOV → ratio 0.5, above DRIFT_AVG_OOV (0.35).
    stats = oov_monitor.compute_oov_stats(
        ["foo bar"],
        oov_monitor._default_analyzer,
        {"foo"},
    )
    assert stats["avg_oov_ratio"] == 0.5
    assert stats["drifted"] is True


def test_no_drift_when_avg_below_threshold():
    # 1 OOV out of 5 terms → 0.2, below threshold.
    stats = oov_monitor.compute_oov_stats(
        ["foo bar baz qux quux"],
        oov_monitor._default_analyzer,
        {"foo", "bar", "baz", "qux"},
    )
    assert stats["avg_oov_ratio"] == 0.2
    assert stats["drifted"] is False


# ── empty / degenerate inputs ───────────────────────────────────────────────
def test_empty_sample_returns_safe_zero():
    stats = oov_monitor.compute_oov_stats(
        [], oov_monitor._default_analyzer, {"a", "b"}
    )
    assert stats == {
        "sample_size": 0,
        "vocab_size": 2,
        "avg_oov_ratio": 0.0,
        "p95_oov_ratio": 0.0,
        "weighted_oov_ratio": 0.0,
        "total_terms": 0,
        "drifted": False,
    }


def test_texts_with_no_extractable_terms_are_skipped():
    # The default analyzer drops single-char and punctuation-only tokens.
    stats = oov_monitor.compute_oov_stats(
        ["", " ", "a !", "foo"],
        oov_monitor._default_analyzer,
        {"foo"},
    )
    # Only "foo" survives extraction.
    assert stats["sample_size"] == 1
    assert stats["avg_oov_ratio"] == 0.0


# ── analyzer pluggability (n-gram honesty) ──────────────────────────────────
def test_custom_analyzer_emits_ngrams():
    # Simulate a vectorizer that emits unigrams + bigrams the same way
    # sklearn does (space-joined). The monitor must see both shapes.
    def ngram_analyzer(text: str) -> list[str]:
        words = oov_monitor._default_analyzer(text)
        bigrams = [f"{a} {b}" for a, b in zip(words, words[1:])]
        return words + bigrams

    vocab = {"micron", "earnings", "micron earnings"}
    # "micron earnings beat" → tokens [micron, earnings, beat,
    #                                  micron earnings, earnings beat]
    # OOV: beat, earnings beat → 2 of 5 = 0.4.
    stats = oov_monitor.compute_oov_stats(
        ["micron earnings beat"], ngram_analyzer, vocab
    )
    assert stats["total_terms"] == 5
    assert stats["weighted_oov_ratio"] == 0.4


# ── tokenizer & normalization ───────────────────────────────────────────────
def test_default_tokenizer_matches_tfidf_pattern():
    # TfidfVectorizer's default token_pattern is r"(?u)\b\w\w+\b" — single
    # chars are stripped, words are lowercased upstream.
    terms = oov_monitor._default_analyzer("A foo, bar's BAZ.")
    assert "foo" in terms
    assert "bar" in terms
    assert "baz" in terms  # lowercased
    assert "a" not in terms  # single char dropped
    assert "s" not in terms  # 's contraction-tail dropped


def test_default_tokenizer_lowercases():
    stats = oov_monitor.compute_oov_stats(
        ["MICRON beats TSMC"],
        oov_monitor._default_analyzer,
        {"micron", "tsmc"},
    )
    # "beats" is the only OOV.
    expected = round(1 / 3, 4)
    assert stats["avg_oov_ratio"] == expected
    assert stats["weighted_oov_ratio"] == expected


# ── p95 numerics ────────────────────────────────────────────────────────────
def test_p95_clamps_to_max_index_on_tiny_sample():
    stats = oov_monitor.compute_oov_stats(
        ["alpha beta"],
        oov_monitor._default_analyzer,
        set(),  # everything OOV
    )
    assert stats["avg_oov_ratio"] == 1.0
    assert stats["p95_oov_ratio"] == 1.0


def test_weighted_vs_avg_diverge_under_imbalanced_lengths():
    # One long fully-in-vocab text + one short fully-OOV text.
    # Per-text ratios: 0.0, 1.0 → avg 0.5.
    # Weighted: 1 OOV / 6 total terms → 0.1667.
    stats = oov_monitor.compute_oov_stats(
        ["foo foo foo foo foo", "bar"],
        oov_monitor._default_analyzer,
        {"foo"},
    )
    assert stats["avg_oov_ratio"] == 0.5
    assert stats["weighted_oov_ratio"] == round(1 / 6, 4)
