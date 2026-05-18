"""Pure-helper tests for the /api/chat ML-gate-skill enrichment.

`_baseline_compare_chat_lines` renders paper-trader's `/api/baseline-compare`
(the read-only OOS-skill diagnostic: does the 17-feature DecisionScorer earn
its complexity out of sample, or does a one-line rule do as well?) into
compact chat-context lines, so the analyst can answer "is the bot's ML edge
real?" honestly instead of parroting the in-sample story.

The surrounding chat handler is one large inline closure, so per the
established design (cf. `_tail_risk_chat_lines` / `_behavioural_chat_lines`)
the logic is a total/pure function unit-tested here — no Flask, no :8090.

Discriminating locks:
- **verbatim SSOT composition** (paper-trader invariant #10): the module's
  own `hint` string must pass through UNCHANGED — no re-derived verdict.
- **withheld ≠ verdict**: INSUFFICIENT_DATA collapses to ONE honest line and
  must NOT leak `hint` (the never-raises endpoint puts an exception string
  there — that must never reach the analyst).
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

from dashboard.web_server import _baseline_compare_chat_lines


def _rep(verdict: str = "MLP_NO_BETTER_THAN_TRIVIAL", **over) -> dict:
    d = {
        "status": "ok",
        "verdict": verdict,
        "n": 1507,
        "mlp": {"rank_ic": 0.0597, "dir_acc": 0.5083, "n": 1507},
        "baselines": [
            {"name": "ml_score", "rank_ic": 0.0525, "dir_acc": 0.54,
             "degenerate": False, "n": 1507},
            {"name": "mom20", "rank_ic": 0.0823, "dir_acc": 0.519,
             "degenerate": False, "n": 1507},
        ],
        "best_baseline": "mom20",
        "best_baseline_ic": 0.0823,
        "ic_gap": -0.0226,
        "hint": ("MLP rank_ic +0.060 vs best one-liner 'mom20' +0.082 "
                 "(gap -0.023, within ±0.05 or below the 0.1 MLP skill "
                 "floor) — the neural net's complexity buys no OOS edge a "
                 "single feature doesn't already carry"),
        "slice": "oos",
        "n_records_considered": 1507,
        "n_train": 3997,
    }
    d.update(over)
    return d


# ── pure/total contract ────────────────────────────────────────────────
@pytest.mark.parametrize("bad", [None, "x", 42, [], ["a"], (), object()])
def test_non_dict_is_silence(bad):
    assert _baseline_compare_chat_lines(bad) == []


def test_missing_or_unknown_verdict_is_silence():
    assert _baseline_compare_chat_lines({}) == []
    assert _baseline_compare_chat_lines({"verdict": None}) == []
    assert _baseline_compare_chat_lines({"verdict": "FObar"}) == []
    assert _baseline_compare_chat_lines(_rep(verdict="WAT")) == []


def test_partial_numerics_never_raise_and_still_state_the_verdict():
    """mlp.rank_ic None / best ic missing → drop only the numeric race line,
    keep verdict + verbatim hint. Must not raise.

    Uses a hint WITHOUT the substring 'rank_ic' so the dropped-numeric-line
    discriminator is unambiguous (the real module hint legitimately contains
    'rank_ic' and passes through verbatim — asserted elsewhere)."""
    rep = _rep()
    rep["hint"] = "the net buys no OOS edge a single feature doesn't carry"
    rep["mlp"] = {"rank_ic": None, "dir_acc": None, "n": 0}
    rep["best_baseline_ic"] = None
    rep["ic_gap"] = None
    out = _baseline_compare_chat_lines(rep)
    assert out, "verdict line must survive missing numerics"
    assert any("MLP_NO_BETTER_THAN_TRIVIAL" in ln for ln in out)
    assert any(rep["hint"] in ln for ln in out)          # SSOT survives
    # the fabricated numeric race line ("  MLP rank_ic … vs best one-liner")
    # must be absent — no numbers → no number line, never a 'n/a' fabrication
    assert not any("vs best one-liner" in ln for ln in out)
    assert not any("rank_ic" in ln for ln in out)        # (hint has none here)


# ── INSUFFICIENT_DATA — one honest withheld line, no hint leak ─────────
def test_insufficient_is_one_withheld_line_without_hint_leak():
    rep = _rep(verdict="INSUFFICIENT_DATA",
               hint="endpoint fault: RuntimeError: db unreadable\n  File ...")
    out = _baseline_compare_chat_lines(rep)
    assert len(out) == 1
    line = out[0]
    assert "withheld" in line.lower()
    assert "OOS" in line or "out-of-sample" in line.lower()
    # the never-raises endpoint stuffs an exception/stack into `hint` on
    # fault — it must NEVER reach the analyst's prompt.
    assert "RuntimeError" not in line and "endpoint fault" not in line
    assert "File ..." not in line


# ── real verdicts — verbatim hint + numeric race ───────────────────────
@pytest.mark.parametrize("verdict", [
    "MLP_NO_BETTER_THAN_TRIVIAL", "MLP_WORSE_THAN_TRIVIAL", "MLP_ADDS_SKILL",
])
def test_real_verdict_emits_verbatim_hint_and_numeric_race(verdict):
    rep = _rep(verdict=verdict)
    out = _baseline_compare_chat_lines(rep)
    blob = "\n".join(out)
    # 1) the verdict token itself is stated (the operator-facing headline)
    assert verdict in blob
    # 2) SSOT: the module's own explanation passes through UNCHANGED — a
    #    chat-side re-derivation that drifts from the trader fails here.
    assert rep["hint"] in blob
    # 3) the scale-invariant numbers a skeptical quant checks
    assert "+0.060" in blob              # MLP rank_ic
    assert "mom20" in blob and "+0.082" in blob   # best one-liner + its ic
    assert "-0.023" in blob              # ic gap (MLP − best)
    assert "3997" in blob                # scorer n_train (is it even live?)


def test_minimal_real_verdict_no_hint_no_numbers_still_states_verdict():
    """A degraded payload that carries only `verdict` (hint='' , numerics
    None) must still surface the verdict — silence here would hide the very
    signal the panel exists for — and must not fabricate a numeric line."""
    out = _baseline_compare_chat_lines(
        {"verdict": "MLP_ADDS_SKILL", "hint": "", "mlp": {},
         "best_baseline_ic": None, "ic_gap": None})
    assert any("MLP_ADDS_SKILL" in ln for ln in out)
    assert not any("rank_ic" in ln for ln in out)
