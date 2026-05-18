"""Pure-helper tests for the /api/chat factor-concentration enrichment.

``_correlation_chat_lines`` renders paper-trader's ``/api/correlation`` (the
diagnostic that exposes *factor* concentration: do the held names actually
move together, or is the book genuinely diversified across uncorrelated
risk?) into compact chat-context lines, so the analyst can answer "am I
really diversified?" honestly instead of only seeing /api/risk's
NAME-level concentration verdict.

The surrounding chat handler is one large inline closure, so per the
established design (cf. ``_baseline_compare_chat_lines`` /
``_macro_calendar_chat_lines``) the logic is a total/pure function
unit-tested here — no Flask, no :8090.

Discriminating locks:
- **verbatim SSOT composition** (paper-trader invariant #10): the builder's
  own ``headline`` string passes through UNCHANGED — no re-derived verdict.
- **NO_DATA is silence**: a book with no stock positions has no factor
  concentration to report; emitting filler would be noise.
- **INSUFFICIENT is one honest withheld line**: the builder's headline
  already explains what's missing; it passes through verbatim.
- **pure/total**: non-dict / missing state / unknown verdict on OK never
  raise and degrade to silence.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dashboard.web_server import _correlation_chat_lines


def _ok(verdict: str = "MODERATE", **over) -> dict:
    """Shape mirrors paper-trader analytics.correlation.build_correlation
    output for a 3-name book with one pair given. Overridable per test."""
    d = {
        "as_of": "2026-05-18T00:00:00+00:00",
        "state": "OK",
        "verdict": verdict,
        "headline": (f"{verdict} — partial diversification "
                     f"(mean ρ=+0.42); 1.87 effective independent bet(s) "
                     f"across 3 correlatable name(s). "
                     f"Most-coupled pair NVDA/SOXL ρ=+0.85."),
        "n_stock_positions": 3,
        "n_correlatable": 3,
        "mean_pairwise_corr": 0.42,
        "max_pair": {"tickers": ["NVDA", "SOXL"], "corr": 0.85},
        "pairs": [
            {"a": "NVDA", "b": "SOXL", "corr": 0.85},
            {"a": "NVDA", "b": "AAPL", "corr": 0.21},
            {"a": "SOXL", "b": "AAPL", "corr": 0.20},
        ],
        "weight_hhi": 0.36,
        "effective_positions_naive": 2.78,
        "effective_independent_bets": 1.87,
        "top_weight_pct": 45.0,
        "top_weight_ticker": "NVDA",
        "weights": {"NVDA": 0.45, "SOXL": 0.30, "AAPL": 0.25},
        "skipped_options": [],
        "short_series_tickers": [],
        "min_returns": 10,
    }
    d.update(over)
    return d


# ── pure/total contract ────────────────────────────────────────────────
@pytest.mark.parametrize("bad", [None, "x", 42, [], ["a"], (), object()])
def test_non_dict_is_silence(bad):
    assert _correlation_chat_lines(bad) == []


def test_missing_state_or_headline_is_silence():
    assert _correlation_chat_lines({}) == []
    assert _correlation_chat_lines({"state": "OK"}) == []
    assert _correlation_chat_lines({"state": "OK", "headline": ""}) == []
    assert _correlation_chat_lines({"state": "OK", "headline": "   "}) == []
    assert _correlation_chat_lines({"state": "OK", "headline": None}) == []


def test_unknown_state_is_silence():
    # The builder only ever emits NO_DATA / INSUFFICIENT / OK. A foreign
    # state on a payload that otherwise looks well-formed must NOT slip
    # through as a verdict line — the chat cannot validate it.
    assert _correlation_chat_lines(_ok(state="WAT")) == []


# ── NO_DATA — silence (no stock positions, concentration undefined) ────
def test_no_data_is_silence():
    """An all-options or empty book has no factor concentration to report;
    emitting a line would be noise, exactly as ``_behavioural_chat_lines``
    omits a NO_DATA scorecard/paralysis/churn block."""
    payload = {
        "state": "NO_DATA",
        "verdict": None,
        "headline": "No stock positions — concentration risk undefined.",
        "n_stock_positions": 0,
    }
    assert _correlation_chat_lines(payload) == []


# ── INSUFFICIENT — one honest withheld line (verbatim builder headline) ─
def test_insufficient_emits_one_verbatim_withheld_line():
    """The builder's own INSUFFICIENT headline already explains what's
    missing (count of correlatable names, threshold, withheld verdict). It
    passes through verbatim — the chat must NOT reword it (drift risk)."""
    hl = ("Only 1 correlatable stock name(s) "
          "(need ≥2 with ≥10 aligned daily returns) — "
          "correlation verdict withheld.")
    payload = {
        "state": "INSUFFICIENT",
        "verdict": None,
        "headline": hl,
        "n_stock_positions": 1,
        "n_correlatable": 1,
    }
    out = _correlation_chat_lines(payload)
    assert len(out) == 1
    assert hl in out[0]                    # SSOT — verbatim builder string
    assert "withheld" in out[0].lower()    # honest verdict-not-given language


def test_insufficient_no_overlapping_history_variant():
    """The other INSUFFICIENT branch the builder emits: ≥2 names but their
    daily-return series don't overlap enough yet. Same contract — verbatim
    passthrough, no chat-side re-derivation."""
    hl = ("Held names have no overlapping return history yet — "
          "correlation verdict withheld.")
    payload = {
        "state": "INSUFFICIENT", "verdict": None, "headline": hl,
        "n_stock_positions": 2, "n_correlatable": 2,
    }
    out = _correlation_chat_lines(payload)
    assert out == [f"Correlation: {hl}"]


# ── OK + real verdicts — verbatim SSOT headline ────────────────────────
@pytest.mark.parametrize("verdict", [
    "SINGLE_NAME_RISK", "CONCENTRATED", "MODERATE", "DIVERSIFIED",
])
def test_real_verdict_emits_verbatim_headline(verdict):
    payload = _ok(verdict=verdict)
    out = _correlation_chat_lines(payload)
    blob = "\n".join(out)
    # 1) the verdict label itself surfaces (operator-facing headline)
    assert verdict in blob
    # 2) SSOT: the module's own headline passes through UNCHANGED — a
    #    chat-side re-derivation that drifts from the trader fails here.
    assert payload["headline"] in blob
    # 3) the most-coupled pair clause that lives INSIDE headline is carried
    assert "NVDA/SOXL" in blob and "+0.85" in blob


def test_ok_with_unknown_verdict_is_silence():
    """A builder bug that emits state=OK with verdict='FOO' must NOT leak
    the unvalidatable label to the analyst — degrade silently rather than
    parrot a token the chat cannot trust."""
    out = _correlation_chat_lines(_ok(verdict="FOO"))
    assert out == []


def test_ok_with_none_verdict_is_silence():
    out = _correlation_chat_lines(_ok(verdict=None))
    assert out == []


def test_single_chat_line_only():
    """The block is intentionally compact — exactly one line. The builder's
    headline already carries verdict + mean ρ + eff-bets + max-pair; a
    second restatement line would double the prompt budget for the same
    information and risk drift."""
    out = _correlation_chat_lines(_ok())
    assert len(out) == 1
