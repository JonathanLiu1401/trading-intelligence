"""Tests for analytics.confluence_signals scoring logic (agent5 code review)."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analytics import confluence_signals as cs  # noqa: E402


@pytest.fixture
def patched_paths(tmp_path, monkeypatch):
    tv = tmp_path / "trend_velocity.json"
    sm = tmp_path / "ticker_sentiment_momentum.json"
    sc = tmp_path / "source_convergence.json"
    out = tmp_path / "confluence_signals.json"
    monkeypatch.setattr(cs, "TV_PATH", tv)
    monkeypatch.setattr(cs, "SM_PATH", sm)
    monkeypatch.setattr(cs, "SC_PATH", sc)
    monkeypatch.setattr(cs, "OUT_PATH", out)
    return tv, sm, sc, out


def test_all_inputs_missing_returns_empty(patched_paths):
    # No source files exist -> run() must not crash, returns []
    assert cs.run() == []


def test_triple_confirmed_ticker_scores_five(patched_paths):
    tv, sm, sc, out = patched_paths
    tv.write_text(json.dumps({"top": [{"ticker": "NVDA", "delta": 7, "ratio": 3.0}]}))
    sm.write_text(json.dumps({"tickers": [
        {"ticker": "NVDA", "direction": "bullish", "delta": 2.0}]}))
    sc.write_text(json.dumps({"events": [
        {"ticker": "NVDA", "distinct_sources": 4, "avg_ai_score": 8.0}]}))
    ranked = cs.run()
    assert len(ranked) == 1
    assert ranked[0]["ticker"] == "NVDA"
    # velocity 1+1, sentiment 1+1, convergence 1 == 5
    assert ranked[0]["score"] == 5


def test_bonus_thresholds_are_inclusive(patched_paths):
    tv, sm, sc, out = patched_paths
    # ratio exactly at bonus threshold, delta exactly at bonus threshold
    tv.write_text(json.dumps({"top": [{"ticker": "AMD", "delta": 3,
                                       "ratio": cs.VELOCITY_RATIO_BONUS}]}))
    sm.write_text(json.dumps({"tickers": [
        {"ticker": "AMD", "direction": "bullish",
         "delta": cs.SENTIMENT_DELTA_BONUS}]}))
    sc.write_text(json.dumps({"events": []}))
    ranked = cs.run()
    assert ranked[0]["score"] == 4  # both bonuses earned


def test_single_signal_filtered_out(patched_paths):
    tv, sm, sc, out = patched_paths
    # Only one velocity point (ratio below bonus) -> score 1 -> dropped (<2)
    tv.write_text(json.dumps({"top": [{"ticker": "F", "delta": 1, "ratio": 1.0}]}))
    sm.write_text(json.dumps({"tickers": []}))
    sc.write_text(json.dumps({"events": []}))
    assert cs.run() == []


def test_bearish_sentiment_ignored(patched_paths):
    tv, sm, sc, out = patched_paths
    tv.write_text(json.dumps({"top": []}))
    sm.write_text(json.dumps({"tickers": [
        {"ticker": "INTC", "direction": "bearish", "delta": 3.0}]}))
    sc.write_text(json.dumps({"events": []}))
    assert cs.run() == []  # bearish contributes no points


def test_output_file_written(patched_paths):
    tv, sm, sc, out = patched_paths
    tv.write_text(json.dumps({"top": [{"ticker": "T", "delta": 2, "ratio": 2.5}]}))
    sm.write_text(json.dumps({"tickers": [
        {"ticker": "T", "direction": "bullish", "delta": 0.5}]}))
    sc.write_text(json.dumps({"events": []}))
    cs.run()
    assert out.exists()
    payload = json.loads(out.read_text())
    assert payload["confluence_count"] == 1
    assert payload["events"][0]["ticker"] == "T"
