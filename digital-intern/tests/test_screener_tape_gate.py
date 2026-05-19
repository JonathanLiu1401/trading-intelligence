"""Screener-tape pseudo-article gate on alert AND briefing paths.

``collectors/market_movers.py`` emits Yahoo Finance screener entries as
pseudo-articles with a unique title shape:

    ``[YF/most_actives] MU (Micron Technology, Inc.) +2.5% @ $698.74 | vol 6``
    ``[YF/day_gainers] AXTI (AXT Inc) +6.6% @ $112.88 | vol 9.6M (0.8x avg)``

These describe CURRENT market state (this ticker is one of today's top
movers), NOT breaking news of an event. The urgency head demonstrably
over-scores them — they look "extreme" because they carry a signed % change,
a large vol number, and a dollar price — and the live evidence below shows
they fire as standalone 🚨 BREAKING alerts at ml_score 9.9 with
score_source='ml'.

Live evidence (2026-05-19, last 2h of articles.db urgency=2 set):

    20:23:03 YF/most_actives ml=9.9 MU (Micron Technology, Inc.) +2.5% @ $698.74
    20:23:03 YF/day_gainers  ml=9.9 AXTI (AXT Inc) +6.6% @ $112.88 | vol 9.6M
    20:07:38 YF/most_actives ml=9.9 MU (Micron Technology, Inc.) +2.5% @ $698.74
    20:07:38 YF/day_gainers  ml=9.9 AXTI (AXT Inc) +6.6% @ $112.88 | vol 9.5M
    19:57:55 YF/most_actives ml=9.9 MU (Micron Technology, Inc.) +1.7% @ $693.38

i.e. 4 of 12 last-2h BREAKING alerts (33%) were YF screener entries — the
analyst's single biggest CURRENT noise complaint. The fix is the same shape
as the three existing quote-widget fingerprints (price-glue / signed-%-paren /
$share-card listing): a defense-in-depth title regex at the alert + briefing
formatter chokepoint, anchored so real headlines never match.

This suite pins:

  1. The two formatter-side gates (alert / briefing) catch the LIVE noise
     verbatim — every observed title above.
  2. A must-survive corpus (real headlines, bracketed real text) is NOT
     caught — anchored discriminator, narrow by construction.
  3. **Lockstep parity**: alert_agent._QW_SCREENER_TAPE.pattern ==
     claude_analyst._QW_SCREENER_TAPE.pattern (single source of truth — a
     future fork of the regex fails this assertion, same drift-class
     precedent as the 3-way recap-template lockstep).
  4. Integration on ``send_urgent_alert``: a screener-only batch never
     reaches Claude/Discord, every dropped row is marked alerted (exits the
     urgent queue instead of churning every 20s).
  5. ``_filter_quote_widget_noise`` partitions correctly on both paths;
     order preserved.

Pure read-side gate; no DB write, no ai_score/ml_score/score_source/urgency
mutation on the suppression path (alert_agent calls store.mark_alerted_batch
on the recap-class precedent, which only sets urgency=2 — the labels-and-
data invariants are intact). Same load-bearing-invariants story as
test_briefing_quote_widget.py / test_alert_recap_template.py.
"""
from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from watchers import alert_agent
from analysis import claude_analyst


# ── 1. LIVE NOISE — alert gate catches every observed screener-tape title ────


@pytest.mark.parametrize("title", [
    # Each is a verbatim row from the 2026-05-19 live evidence set above.
    "[YF/most_actives] MU (Micron Technology, Inc.) +2.5% @ $698.74 | vol 6",
    "[YF/day_gainers] AXTI (AXT Inc) +6.6% @ $112.88 | vol 9.6M (0.8x avg)",
    "[YF/most_actives] MU (Micron Technology, Inc.) +1.7% @ $693.38 | vol 5",
    # Same template, day_losers bucket (the third Yahoo screener key)
    "[YF/day_losers] NVDA (NVIDIA Corp) -3.2% @ $215.40 | vol 12.3M",
])
def test_alert_gate_catches_screener_tape(title):
    assert alert_agent._looks_like_quote_widget({"title": title, "link": ""}) is True


@pytest.mark.parametrize("title", [
    "[YF/most_actives] MU (Micron Technology, Inc.) +2.5% @ $698.74 | vol 6",
    "[YF/day_gainers] AXTI (AXT Inc) +6.6% @ $112.88 | vol 9.6M (0.8x avg)",
    "[YF/most_actives] MU (Micron Technology, Inc.) +1.7% @ $693.38 | vol 5",
    "[YF/day_losers] NVDA (NVIDIA Corp) -3.2% @ $215.40 | vol 12.3M",
])
def test_briefing_gate_catches_screener_tape(title):
    assert claude_analyst._looks_like_quote_widget({"title": title, "link": ""}) is True


# ── 2. MUST-SURVIVE CORPUS — real headlines, bracketed real text ─────────────


@pytest.mark.parametrize("title", [
    # Real breaking headlines must NEVER be caught.
    "Nvidia Q3 revenue rises 22% to $35.1 billion in record quarter",
    "Fed cuts rates 25 bps as inflation cools to target",
    "Micron shares halted as Q3 guidance update pends",
    "Why investors are bullish on Nvidia heading into earnings",
    # Headlines with brackets but NOT the screener tag (different shape).
    "[BREAKING] Fed delivers emergency rate cut after market crash",
    "[UPDATE] MU CFO confirms supply tightness through Q2",
    # Different vendor bracketed (the regex is YF-only by design).
    "[Reuters] Senate passes chip subsidy package",
    "[GDELT/reuters.com] Senate passes chip subsidy package",
    # Cash-ticker headlines (real prose).
    "$NVDA breaks out ahead of earnings (NYSE)",
    "$MU upgraded to Buy (price target $150.00)",
    # The snapshot rows the daemon prepends — must pass through.
    "PORTFOLIO P&L SNAPSHOT",
    "OPTIONS SNAPSHOT",
])
def test_alert_gate_real_headlines_survive(title):
    assert alert_agent._looks_like_quote_widget({"title": title, "link": ""}) is False


@pytest.mark.parametrize("title", [
    "Nvidia Q3 revenue rises 22% to $35.1 billion in record quarter",
    "Fed cuts rates 25 bps as inflation cools to target",
    "Micron shares halted as Q3 guidance update pends",
    "Why investors are bullish on Nvidia heading into earnings",
    "[BREAKING] Fed delivers emergency rate cut after market crash",
    "[UPDATE] MU CFO confirms supply tightness through Q2",
    "[Reuters] Senate passes chip subsidy package",
    "[GDELT/reuters.com] Senate passes chip subsidy package",
    "$NVDA breaks out ahead of earnings (NYSE)",
    "$MU upgraded to Buy (price target $150.00)",
    "PORTFOLIO P&L SNAPSHOT",
    "OPTIONS SNAPSHOT",
])
def test_briefing_gate_real_headlines_survive(title):
    assert claude_analyst._looks_like_quote_widget({"title": title, "link": ""}) is False


# ── 3. LOCKSTEP PARITY — alert and briefing regexes must be byte-identical ──


def test_screener_tape_regex_lockstep_parity():
    """The two gates duplicate the regex by design (anti-import-cycle, same
    discipline as the four other quote-widget fingerprints + the recap-template
    family). A drift between them is exactly the live-evidence failure mode
    this whole layer exists to prevent — the briefing or alert layer would
    suppress something the other still leaks. A future edit to one regex
    without the other fails this assertion loudly."""
    assert (alert_agent._QW_SCREENER_TAPE.pattern
            == claude_analyst._QW_SCREENER_TAPE.pattern), (
        "alert vs briefing screener-tape regexes have drifted apart — see "
        "the lockstep discipline comment on _QW_SCREENER_TAPE in both files"
    )


# ── 4. Integration with send_urgent_alert: never reaches Claude/Discord ────


class _StoreSpy:
    def __init__(self):
        self.marked: list[str] = []

    def mark_alerted_batch(self, ids):
        self.marked.extend(ids)

    def mark_alerted(self, aid):
        self.marked.append(aid)


def _iso(hours_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()


def _row(_id, title, source="YF/most_actives", link=None, **kw):
    # NB: ``https://finance.yahoo.com/quote/<sym>`` is itself a quote-widget
    # URL (caught by ``_QW_QUOTE_PATH``), so callers that want to test a
    # NON-screener row must pass ``link=`` explicitly with a non-quote path.
    base = {
        "_id": _id,
        "link": link if link is not None else f"https://finance.yahoo.com/quote/{_id}",
        "title": title, "source": source, "ai_score": 9.0,
        "summary": "", "published": _iso(0.1), "first_seen": _iso(0.05),
    }
    base.update(kw)
    return base


def test_send_urgent_alert_screener_only_batch_short_circuits():
    """An all-screener batch must never reach Claude OR Discord and every row
    must exit the urgent queue (marked alerted). Mirrors the live failure mode
    where a YF/most_actives MU tick burned a real BREAKING push."""
    store = _StoreSpy()
    batch = [
        _row("mu1", "[YF/most_actives] MU (Micron Technology, Inc.) +2.5% @ $698.74 | vol 6"),
        _row("axti1", "[YF/day_gainers] AXTI (AXT Inc) +6.6% @ $112.88 | vol 9.6M",
             source="YF/day_gainers"),
    ]
    with patch.object(alert_agent, "claude_call") as cc, \
         patch("notifier.discord_notifier.send", return_value=True) as ds, \
         patch.object(alert_agent, "DISCORD_WEBHOOK", "https://discord/test"):
        result = alert_agent.send_urgent_alert(batch, store)
    assert result is False, "screener-only batch must return False"
    assert cc.call_count == 0, "Sonnet must NOT be called on screener-only batch"
    assert ds.call_count == 0, "Discord must NOT be posted on screener-only batch"
    assert set(store.marked) == {"mu1", "axti1"}, (
        f"all screener rows must be marked alerted to exit urgent queue; got "
        f"{store.marked}"
    )


def test_send_urgent_alert_mixed_batch_keeps_real_drops_screener():
    """A mixed batch must fire on the real story and suppress only the
    screener row. Both rows end up marked alerted (real via the normal
    success path; screener via the suppression mark) so neither re-fires."""
    store = _StoreSpy()
    batch = [
        _row("real1", "Fed delivers emergency rate cut after CPI surprise",
             source="reuters", link="https://reuters.com/markets/fed-cut-001"),
        _row("mu1", "[YF/most_actives] MU (Micron Technology, Inc.) +2.5% @ $698.74 | vol 6"),
    ]
    with patch.object(alert_agent, "claude_call", return_value="🚨 BREAKING — Fed cut") as cc, \
         patch("notifier.discord_notifier.send", return_value=True) as ds, \
         patch.object(alert_agent, "DISCORD_WEBHOOK", "https://discord/test"):
        result = alert_agent.send_urgent_alert(batch, store)
    assert result is True, "real story must still fire"
    assert cc.call_count == 1, "Sonnet should be called exactly once for the real row"
    assert ds.call_count == 1, "Discord should be posted exactly once"
    # Both ids should appear in marked: the screener via suppression, the real
    # via the post-success mark_alerted_batch call.
    assert "mu1" in store.marked
    assert "real1" in store.marked
    # The Claude prompt must NOT contain the screener title (it was filtered).
    prompt = cc.call_args[0][0]
    assert "YF/most_actives" not in prompt
    assert "Fed delivers" in prompt


# ── 5. _filter_quote_widget_noise partitions, both surfaces ─────────────────


def test_alert_filter_partition_screener():
    arts = [
        {"title": "Real headline: MU beats Q3 estimates", "_id": "a"},
        {"title": "[YF/most_actives] MU (Micron Technology, Inc.) +2.5% @ $698.74 | vol 6", "_id": "b"},
        {"title": "Another real headline: Fed cuts rates", "_id": "c"},
        {"title": "[YF/day_gainers] AXTI +6.6% @ $112.88 | vol 9.6M", "_id": "d"},
    ]
    snapshot = copy.deepcopy(arts)
    kept, suppressed = alert_agent._filter_quote_widget_noise(arts)
    assert [a["_id"] for a in kept] == ["a", "c"]
    assert [a["_id"] for a in suppressed] == ["b", "d"]
    assert arts == snapshot, "input must NOT be mutated"


def test_briefing_filter_partition_screener():
    arts = [
        {"title": "Real headline: MU beats Q3 estimates"},
        {"title": "[YF/most_actives] MU (Micron Technology, Inc.) +2.5% @ $698.74"},
        {"title": "Another real headline: Fed cuts rates"},
        {"title": "[YF/day_gainers] AXTI +6.6% @ $112.88 | vol 9.6M"},
    ]
    kept, suppressed = claude_analyst._filter_quote_widget_noise(arts)
    assert len(kept) == 2 and len(suppressed) == 2
    kept_titles = {a["title"] for a in kept}
    assert "Real headline: MU beats Q3 estimates" in kept_titles
    assert "Another real headline: Fed cuts rates" in kept_titles
