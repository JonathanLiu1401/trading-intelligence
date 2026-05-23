"""Pure-helper tests for the /api/chat concurrent-opus-attribution enrichment.

``_concurrent_opus_attribution_chat_lines`` renders paper-trader's
``/api/concurrent-opus-attribution`` (per-parent-tree breakdown of
concurrent Opus subprocesses + targeted-kill recommendation) into compact
chat-context lines.

Discriminating locks:

- **verbatim SSOT** (paper-trader invariant #10): the builder's own
  ``headline`` and ``recommendation`` pass through UNCHANGED — no chat-
  side re-derived verdict.
- **healthy host = silence**: NO_OPUS / CLEAN / BENIGN all collapse to
  ``[]``, the ``_decision_paralysis_chat_lines`` silence precedent —
  never chat filler when host_guard's own threshold is not crossed.
- **ELEVATED / SATURATED are loud**: actionable verdicts emit the
  headline + the recommendation line verbatim. SATURATED is the
  operator-critical case the 2026-05-23 >55h paralysis exposed (17
  concurrent Opus all rooted in scripts/hourly_review.sh).
- **pure/total**: non-dict / missing keys / malformed values never raise.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dashboard.web_server import _concurrent_opus_attribution_chat_lines


def _rep(
    verdict="SATURATED",
    headline=(
        "17 concurrent Opus, 17 from non-legitimate parents — host is "
        "saturated by hourly self-review (scripts/hourly_review.sh) "
        "(17 Opus). Targeted kill: `pkill -f scripts/hourly_review.sh`."
    ),
    recommendation=(
        "17 Opus from hourly self-review (scripts/hourly_review.sh) — "
        "`pkill -f scripts/hourly_review.sh`."
    ),
):
    return {
        "verdict": verdict,
        "state": "STORM",
        "n_opus": 17,
        "headline": headline,
        "recommendation": recommendation,
        "dominant_culprit": {
            "parent_marker": "hourly_review",
            "parent_label": "hourly self-review (scripts/hourly_review.sh)",
            "n_opus": 17,
            "kill_command": "pkill -f scripts/hourly_review.sh",
        },
    }


# ─── silence on non-actionable verdicts ────────────────────────────────


def test_non_dict_input_returns_empty():
    assert _concurrent_opus_attribution_chat_lines(None) == []
    assert _concurrent_opus_attribution_chat_lines("string") == []
    assert _concurrent_opus_attribution_chat_lines(42) == []
    assert _concurrent_opus_attribution_chat_lines([]) == []


def test_no_opus_collapses_to_silence():
    assert _concurrent_opus_attribution_chat_lines(_rep(verdict="NO_OPUS")) == []


def test_clean_collapses_to_silence():
    # CLEAN = 1 Opus (the legitimate live runner) — never chat filler.
    assert _concurrent_opus_attribution_chat_lines(_rep(verdict="CLEAN")) == []


def test_benign_collapses_to_silence():
    # BENIGN = within host_guard.DEFAULT_MAX_OPUS=4 — never chat filler.
    assert _concurrent_opus_attribution_chat_lines(_rep(verdict="BENIGN")) == []


def test_unknown_verdict_collapses_to_silence():
    assert _concurrent_opus_attribution_chat_lines(_rep(verdict="GARBAGE")) == []


def test_missing_verdict_collapses_to_silence():
    rep = _rep()
    rep.pop("verdict")
    assert _concurrent_opus_attribution_chat_lines(rep) == []


# ─── ELEVATED / SATURATED render verbatim ──────────────────────────────


def test_saturated_emits_verbatim_headline_first():
    out = _concurrent_opus_attribution_chat_lines(_rep())
    assert len(out) >= 1
    # Verbatim SSOT — the builder's own headline must NOT be paraphrased.
    assert out[0].startswith("17 concurrent Opus")
    assert "scripts/hourly_review.sh" in out[0]
    assert "pkill -f scripts/hourly_review.sh" in out[0]


def test_saturated_emits_recommendation_verbatim():
    out = _concurrent_opus_attribution_chat_lines(_rep())
    assert len(out) >= 2
    # Recommendation line restates the builder's own string verbatim
    # (with leading indent, no paraphrase).
    rec_line = out[1].strip()
    assert rec_line.startswith("17 Opus from hourly")
    assert "pkill -f scripts/hourly_review.sh" in rec_line


def test_elevated_also_fires():
    # ELEVATED (5-8 Opus) is also operator-actionable — must not be
    # silenced just because it's not the 17-storm extreme.
    rep = _rep(
        verdict="ELEVATED",
        headline="6 concurrent Opus — saturated by hourly_review.sh (6 Opus).",
        recommendation="6 Opus from hourly self-review — `pkill -f scripts/hourly_review.sh`.",
    )
    out = _concurrent_opus_attribution_chat_lines(rep)
    assert len(out) == 2
    assert out[0].startswith("6 concurrent Opus")
    assert "pkill" in out[1]


# ─── pure / total contract ─────────────────────────────────────────────


def test_missing_headline_does_not_crash():
    rep = _rep()
    rep.pop("headline")
    out = _concurrent_opus_attribution_chat_lines(rep)
    # Recommendation still surfaces.
    assert len(out) == 1
    assert "pkill" in out[0]


def test_missing_recommendation_emits_only_headline():
    rep = _rep()
    rep.pop("recommendation")
    out = _concurrent_opus_attribution_chat_lines(rep)
    assert len(out) == 1
    assert out[0].startswith("17 concurrent Opus")


def test_empty_recommendation_string_does_not_emit_blank_line():
    rep = _rep(recommendation="")
    out = _concurrent_opus_attribution_chat_lines(rep)
    assert len(out) == 1  # headline only
    assert out[0].startswith("17 concurrent Opus")


def test_non_string_headline_is_dropped():
    rep = _rep(headline=None)
    out = _concurrent_opus_attribution_chat_lines(rep)
    # No headline → recommendation only.
    assert len(out) == 1
    assert "pkill" in out[0]


def test_non_string_recommendation_is_dropped():
    rep = _rep(recommendation=123)
    out = _concurrent_opus_attribution_chat_lines(rep)
    # No valid recommendation → headline only.
    assert len(out) == 1
    assert out[0].startswith("17 concurrent Opus")


def test_no_paraphrase_of_kill_command():
    # The exact `pkill -f scripts/hourly_review.sh` string must survive
    # round-trip verbatim — chat-side must not "improve" it to
    # `pkill -9 scripts/hourly_review.sh` or any variant.
    rep = _rep()
    out = _concurrent_opus_attribution_chat_lines(rep)
    joined = "\n".join(out)
    assert "pkill -f scripts/hourly_review.sh" in joined
    assert "pkill -9" not in joined


def test_live_17_opus_footprint_renders_actionably():
    # Reproduces the 2026-05-23 live SATURATED footprint end-to-end:
    # headline names the parent, recommendation gives the exact kill.
    out = _concurrent_opus_attribution_chat_lines(_rep())
    assert len(out) == 2
    assert "scripts/hourly_review.sh" in out[0]
    assert "scripts/hourly_review.sh" in out[1]
