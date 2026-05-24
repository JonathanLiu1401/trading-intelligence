"""Pure-helper tests for the /api/chat news-to-trade-lag enrichment.

`_news_to_trade_lag_chat_lines` renders paper-trader's
`/api/news-to-trade-lag` (distribution of the freshest plausibly-causal
article's minutes-before each FILLED trade) into compact chat-context
lines so the analyst can answer "is the bot reacting on time, or
consistently 2h+ behind the wire?" — the reactivity follow-up that
trade-attribution leaves implicit.

The surrounding chat handler is one large inline closure, so per the
established design (cf. `_cash_drag_chat_lines` /
`_no_decision_reasons_chat_lines` / `_round_trip_postmortem_chat_lines`)
the logic is a total/pure function unit-tested here — no Flask, no
:8090.

Discriminating locks:

- **verbatim SSOT composition** (paper-trader invariant #10): the
  builder's own ``headline`` string passes through UNCHANGED — no
  chat-side re-derived verdict.
- **healthy / unmeasurable = silence**: REACTIVE_FAST / REACTIVE /
  NO_ATTRIBUTION / NO_DATA / ERROR collapse to ``[]`` — only DELAYED is
  an alert. "Unmeasurable" (NO_ATTRIBUTION, NO_DATA) is not the same
  as "slow", so it must not become chat filler.
- **pure/total**: non-dict / missing keys / unparseable numbers never
  raise and degrade to silence or the safe subset.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dashboard.web_server import _news_to_trade_lag_chat_lines


def _rep(
    verdict="DELAYED",
    *,
    headline=None,
    median_lag_minutes=185.0,
    p75_lag_minutes=240.0,
    n_attributed=12,
):
    if headline is None:
        headline = (
            "DELAYED — desk is acting 185min after the freshest "
            "plausibly-causal article on the wire (p75 240min, n=12 "
            "attributed trades). Leverage decay is biting."
        )
    out = {
        "verdict": verdict,
        "state": verdict,
        "headline": headline,
        "median_lag_minutes": median_lag_minutes,
        "p75_lag_minutes": p75_lag_minutes,
        "n_attributed": n_attributed,
        "no_attribution_pct": 14.3,
        "bucket_fast": 0,
        "bucket_reactive": 2,
        "bucket_delayed": 10,
        "per_trade": [],
    }
    # Best-effort optional fields (only when the inputs are numeric so the
    # helper can take "unparseable" / None values without raising in the
    # fixture itself — these fields are not asserted on anywhere).
    if isinstance(median_lag_minutes, (int, float)) and not isinstance(
        median_lag_minutes, bool
    ):
        out["p25_lag_minutes"] = max(0.0, float(median_lag_minutes) - 60.0)
        out["min_lag_minutes"] = max(0.0, float(median_lag_minutes) - 120.0)
    if isinstance(p75_lag_minutes, (int, float)) and not isinstance(
        p75_lag_minutes, bool
    ):
        out["max_lag_minutes"] = float(p75_lag_minutes) + 60.0
    if isinstance(n_attributed, (int, float)) and not isinstance(
        n_attributed, bool
    ):
        out["n_trades"] = int(n_attributed) + 2
        out["n_no_attribution"] = 2
    return out


# ── pure/total contract ─────────────────────────────────────────────────
@pytest.mark.parametrize("bad", [None, "x", 42, 1.5, [], ["a"], (), object()])
def test_non_dict_is_silence(bad):
    assert _news_to_trade_lag_chat_lines(bad) == []


def test_missing_verdict_is_silence():
    assert _news_to_trade_lag_chat_lines({}) == []
    assert _news_to_trade_lag_chat_lines({"headline": "anything"}) == []


# ── healthy / unmeasurable = silence ────────────────────────────────────
@pytest.mark.parametrize(
    "verdict",
    [
        "REACTIVE_FAST",
        "REACTIVE",
        "NO_ATTRIBUTION",
        "NO_DATA",
        "ERROR",
        None,
        "",
        "UNKNOWN_VERDICT",
    ],
)
def test_non_actionable_verdicts_silence(verdict):
    rep = _rep(verdict=verdict)
    assert _news_to_trade_lag_chat_lines(rep) == []


# ── verbatim headline SSOT (invariant #10) ──────────────────────────────
def test_headline_passes_through_verbatim():
    custom = (
        "DELAYED — desk is acting 320min after the wire — by the time "
        "Opus fires, the news is priced in. n=20 attributed trades."
    )
    out = _news_to_trade_lag_chat_lines(_rep(headline=custom))
    assert out[0] == custom            # exact char-for-char passthrough


# ── detail line content ─────────────────────────────────────────────────
def test_detail_line_restates_percentiles():
    out = _news_to_trade_lag_chat_lines(
        _rep(median_lag_minutes=185.0, p75_lag_minutes=240.0, n_attributed=12)
    )
    body = "\n".join(out)
    assert "median lag 185min" in body
    assert "p75 240min" in body
    assert "n=12 attributed trades" in body


def test_detail_line_skips_missing_percentiles():
    rep = _rep(headline="DELAYED — chronic lag on every recent trade.")
    rep.pop("median_lag_minutes", None)
    rep.pop("p75_lag_minutes", None)
    out = _news_to_trade_lag_chat_lines(rep)
    body = "\n".join(out)
    assert "median lag" not in body
    assert "p75" not in body
    assert "n=12 attributed trades" in body


def test_detail_line_skips_unparseable_fields():
    rep = _rep(
        median_lag_minutes="x",
        p75_lag_minutes=True,           # bool must be rejected
        n_attributed=None,
    )
    out = _news_to_trade_lag_chat_lines(rep)
    # Headline only; detail line is suppressed because no parts survived.
    assert len(out) == 1


def test_no_numeric_fields_emits_headline_only():
    rep = {
        "verdict": "DELAYED",
        "headline": "DELAYED — chronic lag on every recent trade.",
    }
    out = _news_to_trade_lag_chat_lines(rep)
    assert out == ["DELAYED — chronic lag on every recent trade."]


def test_median_lag_rounds_to_minute():
    out = _news_to_trade_lag_chat_lines(
        _rep(median_lag_minutes=185.6, p75_lag_minutes=241.2, n_attributed=5)
    )
    body = "\n".join(out)
    # `:.0f` should round the percentiles to whole minutes.
    assert "median lag 186min" in body
    assert "p75 241min" in body


# ── headline-empty fallback ─────────────────────────────────────────────
def test_empty_headline_omits_first_line_but_detail_still_renders():
    rep = _rep(headline="")
    out = _news_to_trade_lag_chat_lines(rep)
    body = "\n".join(out)
    assert body.startswith("  ")        # only the indented detail line
    assert "median lag 185min" in body


def test_garbage_headline_omits_first_line():
    rep = _rep(headline=42)
    out = _news_to_trade_lag_chat_lines(rep)
    body = "\n".join(out)
    assert "median lag 185min" in body


# ── shape stability ─────────────────────────────────────────────────────
def test_returns_list_always():
    assert isinstance(_news_to_trade_lag_chat_lines({}), list)
    assert isinstance(_news_to_trade_lag_chat_lines(None), list)
    assert isinstance(_news_to_trade_lag_chat_lines(_rep()), list)


def test_no_attribution_with_single_trade_is_silence():
    """The live state at pass time is exactly this: 1 trade, no
    attributed article, NO_ATTRIBUTION verdict. The chat must NOT fire
    on this — a sample of 1 unmeasurable trade is not 'desk is slow'."""
    rep = _rep(verdict="NO_ATTRIBUTION")
    rep["n_attributed"] = 0
    rep["n_no_attribution"] = 1
    rep["n_trades"] = 1
    assert _news_to_trade_lag_chat_lines(rep) == []
