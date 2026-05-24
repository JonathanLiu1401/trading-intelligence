"""Pure-helper tests for the /api/chat scorer-vs-book disagreement enrichment.

``_scorer_book_disagreement_chat_lines`` renders paper-trader's
``/api/disagreement`` (the scorer-vs-Opus per-position disagreement
panel — does the bot's OWN ML agree with what it's holding?) into
compact chat-context lines.

Discriminating locks:

- **off-distribution pre-filter** (the trader endpoint's own
  ``off_distribution`` flag marks clamped extrapolation, NOT a real
  scorer/Opus fight): such rows must never trigger a chat alert even
  when severity=HIGH. Mirrors the dashboard panel's own honesty flag.
- **scorer-trained qualification gate**: ``scorer_trained=False`` →
  silence, mirroring the trader endpoint's empty-rows behaviour when
  the scorer is unqualified — the chat must not carry an
  unqualified-scorer verdict.
- **MEDIUM-only silence**: MEDIUM is the trader's "mild discomfort"
  middle tier; only HIGH triggers chat. A book of only MEDIUM rows is
  silence — keeping the chat alert sharp.
- **worst-row selection determinism**: among HIGH rows the surfaced
  detail picks MIN ``scorer_pred_5d_pct`` (scorer wants OUT hardest),
  alphabetical tie-break by ticker.
- **headline composition restates only endpoint fields**: counts.HIGH
  + worst row's ticker / scorer_verdict / last_action — never a metric
  the endpoint did not emit (the ``_passive_signal_density_chat_lines``
  precedent for endpoints without a top-level headline string).
- **pure/total**: non-dict / missing keys / unparseable numerics never
  raise and degrade to silence or the safe subset.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dashboard.web_server import _scorer_book_disagreement_chat_lines


def _row(
    ticker="NVDA",
    severity="HIGH",
    *,
    scorer_verdict="EXIT",
    scorer_pred_5d_pct=-12.5,
    last_action="BUY",
    off_distribution=False,
    label=None,
    interval=None,
):
    return {
        "ticker": ticker,
        "scorer_verdict": scorer_verdict,
        "scorer_pred_5d_pct": scorer_pred_5d_pct,
        "last_action": last_action,
        "last_action_ts": "2026-05-24T10:00:00+00:00",
        "severity": severity,
        "label": label or f"{severity}: {scorer_verdict} vs {last_action}",
        "interval": interval,
        "off_distribution": off_distribution,
    }


def _rep(rows=None, *, scorer_trained=True, counts=None):
    if rows is None:
        rows = [_row()]
    if counts is None:
        c = {"HIGH": 0, "MEDIUM": 0, "ALIGNED": 0}
        for r in rows:
            sev = r.get("severity")
            if sev in c:
                c[sev] += 1
        counts = c
    return {
        "as_of": "2026-05-24T14:00:00+00:00",
        "scorer_trained": scorer_trained,
        "n_positions": len(rows),
        "counts": counts,
        "rows": rows,
    }


# ── pure / total contract ───────────────────────────────────────────────
@pytest.mark.parametrize("bad", [None, "x", 42, [], ["a"], (), object()])
def test_non_dict_is_silence(bad):
    assert _scorer_book_disagreement_chat_lines(bad) == []


def test_empty_dict_is_silence():
    assert _scorer_book_disagreement_chat_lines({}) == []


# ── scorer-trained qualification gate ───────────────────────────────────
def test_scorer_untrained_is_silence_even_with_high_rows():
    rep = _rep(rows=[_row(severity="HIGH")], scorer_trained=False)
    assert _scorer_book_disagreement_chat_lines(rep) == []


def test_missing_scorer_trained_treated_as_false():
    rep = _rep(rows=[_row(severity="HIGH")])
    rep.pop("scorer_trained")
    assert _scorer_book_disagreement_chat_lines(rep) == []


# ── healthy book = silence ──────────────────────────────────────────────
def test_no_rows_is_silence():
    rep = _rep(rows=[], counts={"HIGH": 0, "MEDIUM": 0, "ALIGNED": 0})
    assert _scorer_book_disagreement_chat_lines(rep) == []


def test_all_aligned_is_silence():
    rows = [
        _row(ticker="NVDA", severity="ALIGNED"),
        _row(ticker="AMD", severity="ALIGNED"),
    ]
    assert _scorer_book_disagreement_chat_lines(_rep(rows=rows)) == []


def test_medium_only_is_silence():
    rows = [
        _row(ticker="NVDA", severity="MEDIUM"),
        _row(ticker="AMD", severity="MEDIUM"),
        _row(ticker="MU", severity="ALIGNED"),
    ]
    assert _scorer_book_disagreement_chat_lines(_rep(rows=rows)) == []


# ── off-distribution pre-filter ─────────────────────────────────────────
def test_off_distribution_high_is_silence():
    rows = [
        _row(ticker="MU", severity="HIGH", off_distribution=True,
             scorer_pred_5d_pct=-20.0),
    ]
    assert _scorer_book_disagreement_chat_lines(_rep(rows=rows)) == []


def test_off_distribution_filtered_from_count_in_headline():
    # Mix one OOD HIGH with one in-distribution HIGH — chat surfaces
    # only the in-distribution one, count reads from len(filtered) when
    # counts.HIGH is missing.
    rows = [
        _row(ticker="OOD", severity="HIGH", off_distribution=True,
             scorer_pred_5d_pct=-30.0),
        _row(ticker="REAL", severity="HIGH", off_distribution=False,
             scorer_pred_5d_pct=-15.0, scorer_verdict="EXIT",
             last_action="BUY"),
    ]
    rep = _rep(rows=rows)
    rep.pop("counts")
    out = _scorer_book_disagreement_chat_lines(rep)
    body = "\n".join(out)
    # When counts is absent, headline derives from len(filtered)=1 not 2
    assert "1 HIGH-severity" in body
    assert "REAL" in body
    assert "OOD" not in body


# ── actionable surfacing ────────────────────────────────────────────────
def test_single_high_emits_headline_and_detail():
    out = _scorer_book_disagreement_chat_lines(_rep())
    assert len(out) == 2
    head = out[0]
    detail = out[1]
    assert "1 HIGH-severity scorer/Opus conflict" in head
    assert "NVDA" in head
    assert "EXIT" in head
    assert "BUY" in head
    assert "NVDA" in detail
    assert "-12.50%" in detail or "-12.5" in detail


def test_multiple_high_pluralizes_and_picks_worst():
    rows = [
        _row(ticker="MU", severity="HIGH", scorer_pred_5d_pct=-8.0),
        _row(ticker="NVDA", severity="HIGH", scorer_pred_5d_pct=-22.5,
             scorer_verdict="EXIT", last_action="BUY"),
        _row(ticker="AMD", severity="HIGH", scorer_pred_5d_pct=-15.0),
        _row(ticker="AVGO", severity="ALIGNED"),  # not counted
        _row(ticker="LRCX", severity="MEDIUM"),   # not counted
    ]
    out = _scorer_book_disagreement_chat_lines(_rep(rows=rows))
    head = out[0]
    detail = out[1]
    # Pluralized
    assert "3 HIGH-severity scorer/Opus conflicts" in head
    # Worst = NVDA (most-negative scorer pred)
    assert "NVDA" in head
    assert "NVDA" in detail
    assert "-22.50%" in detail or "-22.5" in detail
    assert "AMD" not in detail  # not the worst
    assert "MU" not in detail


def test_alphabetical_tiebreak_when_pred_equal():
    rows = [
        _row(ticker="ZZZ", severity="HIGH", scorer_pred_5d_pct=-10.0),
        _row(ticker="AAA", severity="HIGH", scorer_pred_5d_pct=-10.0),
        _row(ticker="MMM", severity="HIGH", scorer_pred_5d_pct=-10.0),
    ]
    out = _scorer_book_disagreement_chat_lines(_rep(rows=rows))
    head = out[0]
    detail = out[1]
    assert "AAA" in head
    assert "AAA" in detail
    assert "ZZZ" not in detail


def test_counts_high_from_endpoint_used_directly():
    # counts.HIGH may legitimately differ from len(filtered_rows) if
    # the trader endpoint reported a count > the surfaced row list (e.g.
    # truncation). The chat should honor counts.HIGH for the headline
    # number when present.
    rows = [
        _row(ticker="NVDA", severity="HIGH", scorer_pred_5d_pct=-12.0),
    ]
    rep = _rep(rows=rows, counts={"HIGH": 5, "MEDIUM": 2, "ALIGNED": 3})
    out = _scorer_book_disagreement_chat_lines(rep)
    assert "5 HIGH-severity scorer/Opus conflicts" in out[0]


def test_negative_counts_high_falls_back_to_filtered_len():
    rows = [_row(ticker="NVDA", severity="HIGH", scorer_pred_5d_pct=-12.0)]
    # Pathological counts.HIGH (negative or non-int) → fall back to len
    rep = _rep(rows=rows, counts={"HIGH": -3})
    out = _scorer_book_disagreement_chat_lines(rep)
    assert "1 HIGH-severity scorer/Opus conflict" in out[0]


# ── degraded inputs degrade silently, never raise ──────────────────────
def test_rows_not_a_list_is_silence():
    rep = _rep()
    rep["rows"] = "not a list"
    assert _scorer_book_disagreement_chat_lines(rep) == []


def test_garbage_rows_skipped():
    # The helper itself must filter junk rows when the trader endpoint
    # ever ships a malformed array. We construct the payload by hand so
    # the test fixture's own counts-derivation doesn't crash on the
    # junk before the helper sees it.
    rows = [
        None, "string", 42, ["nope"], object(),
        _row(ticker="NVDA", severity="HIGH", scorer_pred_5d_pct=-9.0),
    ]
    rep = {
        "as_of": "2026-05-24T14:00:00+00:00",
        "scorer_trained": True,
        "n_positions": len(rows),
        "counts": {"HIGH": 1, "MEDIUM": 0, "ALIGNED": 0},
        "rows": rows,
    }
    out = _scorer_book_disagreement_chat_lines(rep)
    assert any("NVDA" in line for line in out)


def test_missing_scorer_pred_degrades_to_zero_for_sort_but_drops_detail():
    # A HIGH row with no scorer_pred → headline still emits; detail line
    # only renders the pred fragment when present, but the row may still
    # be picked as "worst" via the 0.0 sort key.
    rows = [
        _row(ticker="UNPRED", severity="HIGH"),
    ]
    rows[0]["scorer_pred_5d_pct"] = None
    out = _scorer_book_disagreement_chat_lines(_rep(rows=rows))
    head = out[0]
    assert "UNPRED" in head
    # detail line absent because pred is None
    assert len(out) == 1


def test_garbage_pred_does_not_raise():
    rows = [
        _row(ticker="GOOD", severity="HIGH", scorer_pred_5d_pct=-5.0),
        _row(ticker="BAD", severity="HIGH"),
    ]
    rows[1]["scorer_pred_5d_pct"] = "not-a-number"
    out = _scorer_book_disagreement_chat_lines(_rep(rows=rows))
    body = "\n".join(out)
    # No raise; GOOD picked as worst (BAD's pred treated as 0 by sort)
    assert "GOOD" in body
    assert "-5.0" in body


def test_missing_ticker_renders_placeholder():
    rows = [_row(severity="HIGH", scorer_pred_5d_pct=-10.0)]
    rows[0].pop("ticker")
    out = _scorer_book_disagreement_chat_lines(_rep(rows=rows))
    head = out[0]
    # Placeholder "?" rather than raise
    assert "?" in head


def test_missing_scorer_verdict_renders_placeholder():
    rows = [_row(severity="HIGH", scorer_pred_5d_pct=-10.0)]
    rows[0].pop("scorer_verdict")
    out = _scorer_book_disagreement_chat_lines(_rep(rows=rows))
    assert "?" in out[0]
