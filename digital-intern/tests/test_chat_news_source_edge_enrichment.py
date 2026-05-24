"""Pure-helper tests for the /api/chat news-source-edge enrichment.

`_news_source_edge_chat_lines` renders paper-trader's `/api/source-edge`
(the read-only per-collector predictive-edge diagnostic: which of digital-
intern's ~17 collectors' scored headlines actually precede the SPY-abnormal
move?) into compact chat-context lines, so the analyst can answer "should I
trust this MarketWatch headline?" or "which sources actually move the tape?"
without re-deriving from raw signals.

The surrounding chat handler is one large inline closure, so per the
established design (cf. `_baseline_compare_chat_lines` / `_tail_risk_chat_lines`)
the logic is a total/pure function unit-tested here — no Flask, no :8090.

Discriminating locks:
- **verbatim SSOT composition** (paper-trader invariant #10): the module's
  own `headline` string must pass through UNCHANGED — no re-derived verdict.
- **withheld ≠ verdict**: INSUFFICIENT_DATA / NO_DATA collapse to ONE honest
  line and must NOT leak headline (the never-raises trader endpoint may carry
  a deficiency note rather than an actionable verdict in adjacent fields).
- **pure/total**: non-dict / missing / unknown verdict / partial numerics
  never raise and degrade to silence or the safe subset.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dashboard.web_server import _news_source_edge_chat_lines


def _rep(verdict: str = "NO_EDGE", **over) -> dict:
    d = {
        "as_of": "2026-05-24T15:00:00+00:00",
        "n_articles": 5165,
        "n_scored": 5165,
        "n_resolved": 846,
        "min_score": 2.0,
        "horizons": [1, 3, 5],
        "reference_horizon": 5,
        "spy_adjusted": True,
        "lookback_days": 30,
        "best_source": "googlenews",
        "worst_source": "scraped",
        "verdict": verdict,
        "verdict_reason": ("even the best collector (googlenews) is only "
                           "+0.24pp abnormal at 5d"),
        "headline": ("NO_EDGE: best googlenews +0.24pp/5d (n=25); worst "
                     "scraped -4.56pp [6/57 graded]"),
        "sources": [],
    }
    d.update(over)
    return d


# ── pure/total contract ────────────────────────────────────────────────
@pytest.mark.parametrize("bad", [None, "x", 42, [], ["a"], (), object()])
def test_non_dict_is_silence(bad):
    assert _news_source_edge_chat_lines(bad) == []


def test_missing_or_unknown_verdict_is_silence():
    assert _news_source_edge_chat_lines({}) == []
    assert _news_source_edge_chat_lines({"verdict": None}) == []
    assert _news_source_edge_chat_lines({"verdict": "FObar"}) == []
    assert _news_source_edge_chat_lines(_rep(verdict="WAT")) == []


# ── INSUFFICIENT_DATA / NO_DATA — one honest withheld line ─────────────
@pytest.mark.parametrize("verdict", ["INSUFFICIENT_DATA", "NO_DATA"])
def test_insufficient_collapses_to_one_withheld_line(verdict):
    rep = _rep(verdict=verdict,
               headline=("INSUFFICIENT_DATA: 12 resolved, 17 sources, none "
                         "gradable at 5d yet"))
    out = _news_source_edge_chat_lines(rep)
    assert len(out) == 1
    line = out[0]
    assert "withheld" in line.lower()
    assert "source" in line.lower()
    # The deficiency-note headline must NOT propagate verbatim — the withheld
    # branch deliberately stays opaque to keep faults from leaking into the
    # analyst prompt (mirrors the baseline_compare INSUFFICIENT_DATA contract).
    assert "INSUFFICIENT_DATA" not in line
    assert "gradable" not in line


def test_insufficient_when_headline_missing():
    rep = _rep(verdict="INSUFFICIENT_DATA")
    rep.pop("headline", None)
    out = _news_source_edge_chat_lines(rep)
    assert out == ["News-source edge: insufficient resolved history — verdict "
                   "withheld."]


# ── real verdicts — verbatim headline + parenthetical tag ─────────────
@pytest.mark.parametrize("verdict", ["EDGE_FOUND", "NO_EDGE"])
def test_real_verdict_emits_verbatim_headline(verdict):
    rep = _rep(verdict=verdict)
    out = _news_source_edge_chat_lines(rep)
    blob = "\n".join(out)
    # Verdict on the first line.
    assert verdict in out[0]
    # SSOT headline on the second line — exact passthrough.
    assert rep["headline"] in blob
    # Parenthetical tag has lookback/horizon/n_resolved.
    assert "30d lookback" in out[0]
    assert "5d ref" in out[0]
    assert "n_resolved=846" in out[0]


def test_edge_found_phrasing():
    rep = _rep(verdict="EDGE_FOUND",
               headline=("EDGE_FOUND: best polygon +1.84pp/5d (n=60); worst "
                         "reddit -2.10pp [12/57 graded]"))
    out = _news_source_edge_chat_lines(rep)
    blob = "\n".join(out)
    assert "EDGE_FOUND" in blob
    assert "polygon" in blob          # verbatim from headline
    assert "+1.84pp/5d" in blob       # verbatim numeric


def test_partial_numerics_drop_silently_without_raising():
    """Missing lookback / ref / n_resolved drop only those parenthetical
    bits — verdict line + verbatim headline must survive. Never raises."""
    rep = _rep()
    rep.pop("lookback_days", None)
    rep.pop("reference_horizon", None)
    rep.pop("n_resolved", None)
    out = _news_source_edge_chat_lines(rep)
    assert out  # not silenced
    assert "NO_EDGE" in out[0]
    assert "lookback" not in out[0]
    assert "n_resolved" not in out[0]
    # Headline still surfaces verbatim.
    assert rep["headline"] in "\n".join(out)


def test_missing_headline_drops_second_line_keeps_verdict():
    rep = _rep()
    rep.pop("headline", None)
    out = _news_source_edge_chat_lines(rep)
    assert len(out) == 1
    assert "NO_EDGE" in out[0]


def test_empty_headline_drops_second_line():
    rep = _rep(headline="   ")
    out = _news_source_edge_chat_lines(rep)
    assert len(out) == 1


def test_non_string_headline_drops_second_line():
    rep = _rep(headline=42)
    out = _news_source_edge_chat_lines(rep)
    assert len(out) == 1
    assert "NO_EDGE" in out[0]


def test_non_int_numerics_dropped_safely():
    """A float / string lookback must not show up as 'X.0d lookback' or
    raise — we filter on isinstance(int)."""
    rep = _rep(lookback_days=30.5, reference_horizon="five", n_resolved=None)
    out = _news_source_edge_chat_lines(rep)
    assert "lookback" not in out[0]
    assert "ref" not in out[0]
    assert "n_resolved" not in out[0]
    # Headline survives.
    assert rep["headline"] in "\n".join(out)


# ── SSOT lock — the verbatim headline must equal the input headline ───
def test_headline_is_verbatim_passthrough_not_paraphrased():
    rep = _rep(headline=("NO_EDGE: best googlenews +0.24pp/5d (n=25); worst "
                         "scraped -4.56pp [6/57 graded]"))
    out = _news_source_edge_chat_lines(rep)
    headline_lines = [ln for ln in out if rep["headline"] in ln]
    assert len(headline_lines) == 1
    # Must be indented (two-space prefix is the established sibling
    # convention for SSOT continuation lines).
    assert headline_lines[0].startswith("  ")
