"""Pure-helper tests for /api/chat recency-decayed news ranking (web_server.py).

The chat news tiers were ordered by raw ``ai_score DESC``, so an 18h-old
"STOCK SURGED TODAY" 9.0 outranked a fresh 8.6 — the operator's primary
surface led with stale breaking news. ArticleNet already trains a
``time_sensitivity`` head and the 5h briefing already decays by it
(``analysis.claude_analyst._rank_by_decayed_score`` /  ``_effective_score``:
``effective = ai_score * 0.5 ** (age_h * ts / 12h)``). ``_rerank_chat_news``
applies that **same single source of truth** to the chat candidates so the
two surfaces rank consistently, instead of forking a second decay curve.

System-wide NULL policy (locked here so it can't silently drift): an unscored
row (``time_sensitivity`` NULL/missing/junk) gets ``BRIEFING_DEFAULT_TS`` (mild
decay), and an unparseable/absent ``first_seen`` → age 0 → factor 1 (a parse
failure can only ever *not decay*, never bury a row). On any failure the helper
degrades to the incoming order — decay can only help the chat, never sink it.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dashboard.web_server import _rerank_chat_news

NOW = datetime(2026, 5, 18, 12, 0, 0, tzinfo=timezone.utc)


def _art(title, ai, age_h, ts):
    return {
        "title": title,
        "source": "rss",
        "ai_score": ai,
        "summary": "",
        "time_sensitivity": ts,
        "first_seen": (NOW - timedelta(hours=age_h)).isoformat(),
    }


class TestRerankChatNews:
    def test_fresh_lower_score_beats_stale_higher_score(self):
        """The whole point: a fresh 8.6 must outrank an 18h-old, fast-decaying
        9.0 — the bug raw ai_score ordering caused on the operator's primary
        surface. 9.0 * 0.5**(18*1/12) ≈ 3.18 < 8.6 * 0.5**(0.5*1/12) ≈ 8.36."""
        stale_hot = _art("18h-old SURGED TODAY", 9.0, age_h=18.0, ts=1.0)
        fresh = _art("fresh breaking", 8.6, age_h=0.5, ts=1.0)
        out = _rerank_chat_news([stale_hot, fresh], limit=10, now=NOW)
        assert [a["title"] for a in out][0] == "fresh breaking"

    def test_timeless_news_does_not_decay(self):
        """ts=0 (secular/macro thesis) → factor 1 → pure ai_score order even
        when old. A 30h-old 9.0 macro piece must still beat a fresh 8.0."""
        old_macro = _art("secular thesis", 9.0, age_h=30.0, ts=0.0)
        fresh_minor = _art("minor blip", 8.0, age_h=0.1, ts=0.0)
        out = _rerank_chat_news([fresh_minor, old_macro], limit=10, now=NOW)
        assert [a["title"] for a in out][0] == "secular thesis"

    def test_null_time_sensitivity_still_decays_at_default(self):
        """An unscored row (ts=None) must NOT be treated as timeless — it
        gets the system default mild decay, so a very old NULL-ts 9.0 loses
        to a fresh NULL-ts 8.0 (locks the NULL policy against drift)."""
        old_unscored = _art("old unscored", 9.0, age_h=60.0, ts=None)
        fresh_unscored = _art("fresh unscored", 8.0, age_h=0.2, ts=None)
        out = _rerank_chat_news(
            [old_unscored, fresh_unscored], limit=10, now=NOW)
        assert [a["title"] for a in out][0] == "fresh unscored"

    def test_unparseable_first_seen_is_not_buried(self):
        """A row whose first_seen won't parse must get age 0 (no decay), not
        sink — a parse failure may only ever help, never bury a high-value
        row. Its 9.5 stays ahead of a fresh-but-decayed 7.0."""
        bad_ts = {"title": "bad date 9.5", "source": "rss", "ai_score": 9.5,
                  "summary": "", "time_sensitivity": 1.0,
                  "first_seen": "not-a-date"}
        fresh_low = _art("fresh 7.0", 7.0, age_h=0.1, ts=1.0)
        out = _rerank_chat_news([fresh_low, bad_ts], limit=10, now=NOW)
        assert [a["title"] for a in out][0] == "bad date 9.5"

    def test_truncates_to_limit(self):
        rows = [_art(f"a{i}", 9.0 - i * 0.1, age_h=1.0, ts=0.5)
                for i in range(20)]
        out = _rerank_chat_news(rows, limit=10, now=NOW)
        assert len(out) == 10
        # Highest-effective (all equal age/ts → ai_score order) come first.
        assert out[0]["title"] == "a0"

    def test_nonpositive_limit_returns_empty(self):
        rows = [_art("x", 9.0, age_h=1.0, ts=0.5)]
        assert _rerank_chat_news(rows, limit=0, now=NOW) == []

    def test_degrades_to_incoming_order_on_failure(self):
        """Pure/total: garbage rows must never raise into the chat handler.
        The fallback preserves the incoming (ai_score-DESC from SQL) order
        and the limit so the chat still gets *a* news block."""
        rows = ["not-a-dict", {"no_score": True}, 42]
        out = _rerank_chat_news(rows, limit=2, now=NOW)
        assert out == rows[:2]

    def test_empty_input(self):
        assert _rerank_chat_news([], limit=10, now=NOW) == []
