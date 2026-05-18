"""Per-article recency decay in the heartbeat briefing ranker.

Feature: ``analysis.claude_analyst`` finally applies the ML
``time_sensitivity`` head via the decay curve the
``article_store.get_top_for_briefing`` docstring has always specified but no
consumer ever applied:

    effective = base * 0.5 ** (age_hours * time_sensitivity / 12h)

These tests assert the exact arithmetic, the snapshot-pinning stability
property (prepended PORTFOLIO/OPTIONS rows must stay at the top), purity (no
input mutation — load-bearing invariants intact by construction), and the
analyst-facing behaviour (a fresh item out-ranks an older same-base one;
a timeless ts=0 item does not decay).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from analysis import claude_analyst as ca

NOW = datetime(2026, 5, 18, 12, 0, 0, tzinfo=timezone.utc)


def _iso(hours_ago: float) -> str:
    return (NOW - timedelta(hours=hours_ago)).isoformat()


# ── _seen_age_hours ──────────────────────────────────────────────────────────
class TestSeenAgeHours:
    def test_absent_or_empty_is_zero(self):
        assert ca._seen_age_hours(None, now=NOW) == 0.0
        assert ca._seen_age_hours("", now=NOW) == 0.0
        assert ca._seen_age_hours("   ", now=NOW) == 0.0

    def test_unparseable_is_zero(self):
        assert ca._seen_age_hours("not-a-date", now=NOW) == 0.0

    def test_future_is_zero_not_negative(self):
        # Clock skew / bad row must never produce a negative age (which would
        # *inflate* the score via a >1 decay factor).
        assert ca._seen_age_hours(_iso(-5.0), now=NOW) == 0.0

    def test_iso_24h_ago(self):
        assert ca._seen_age_hours(_iso(24.0), now=NOW) == pytest.approx(24.0)

    def test_rfc822_parsed(self):
        from email.utils import format_datetime
        raw = format_datetime(NOW - timedelta(hours=6))
        assert ca._seen_age_hours(raw, now=NOW) == pytest.approx(6.0, abs=0.05)

    def test_naive_assumed_utc(self):
        raw = (NOW - timedelta(hours=3)).replace(tzinfo=None).isoformat()
        assert ca._seen_age_hours(raw, now=NOW) == pytest.approx(3.0)


# ── _effective_score ─────────────────────────────────────────────────────────
class TestEffectiveScore:
    def test_no_decay_when_ts_zero(self):
        a = {"ai_score": 8.0, "time_sensitivity": 0.0, "first_seen": _iso(20)}
        assert ca._effective_score(a, now=NOW) == pytest.approx(8.0)

    def test_no_decay_when_age_zero(self):
        a = {"ai_score": 8.0, "time_sensitivity": 1.0, "first_seen": None}
        assert ca._effective_score(a, now=NOW) == pytest.approx(8.0)

    def test_halflife_exact_12h_ts1(self):
        a = {"ai_score": 9.0, "time_sensitivity": 1.0, "first_seen": _iso(12)}
        # 9 * 0.5 ** (12*1/12) == 9 * 0.5
        assert ca._effective_score(a, now=NOW) == pytest.approx(4.5)

    def test_two_halflives_24h_ts1(self):
        a = {"ai_score": 8.0, "time_sensitivity": 1.0, "first_seen": _iso(24)}
        assert ca._effective_score(a, now=NOW) == pytest.approx(2.0)

    def test_partial_decay_6h_ts_half(self):
        a = {"ai_score": 10.0, "time_sensitivity": 0.5, "first_seen": _iso(6)}
        # 10 * 0.5 ** (6*0.5/12) == 10 * 0.5**0.25
        assert ca._effective_score(a, now=NOW) == pytest.approx(10 * 0.5 ** 0.25)

    def test_relevance_score_fallback(self):
        a = {"_relevance_score": 4.0, "time_sensitivity": 0.0,
             "first_seen": _iso(1)}
        assert ca._effective_score(a, now=NOW) == pytest.approx(4.0)

    def test_zero_or_missing_base_sorts_last(self):
        assert ca._effective_score({}, now=NOW) == 0.0
        assert ca._effective_score(
            {"ai_score": 0, "_relevance_score": 0}, now=NOW) == 0.0

    def test_none_ts_uses_default(self):
        a = {"ai_score": 10.0, "time_sensitivity": None, "first_seen": _iso(12)}
        # default ts = 0.5 → 10 * 0.5 ** (12*0.5/12) = 10 * 0.5**0.5
        assert ca._effective_score(a, now=NOW) == pytest.approx(
            10 * 0.5 ** 0.5)
        assert ca.BRIEFING_DEFAULT_TS == 0.5

    def test_nan_ts_guarded(self):
        a = {"ai_score": 10.0, "time_sensitivity": float("nan"),
             "first_seen": _iso(12)}
        # NaN must fall back to the default, not propagate into the score.
        got = ca._effective_score(a, now=NOW)
        assert got == got  # not NaN
        assert got == pytest.approx(10 * 0.5 ** 0.5)

    def test_bool_base_not_read_as_one(self):
        # A stray bool must not be read as 1.0 (isinstance(True,int) is True).
        a = {"ai_score": True, "_relevance_score": 3.0,
             "time_sensitivity": 0.0, "first_seen": _iso(1)}
        assert ca._effective_score(a, now=NOW) == pytest.approx(3.0)


# ── _rank_by_decayed_score ───────────────────────────────────────────────────
class TestRankByDecayedScore:
    def test_fresh_beats_older_same_base(self):
        older = {"_id": "old", "ai_score": 9.0, "time_sensitivity": 1.0,
                 "first_seen": _iso(20)}
        fresh = {"_id": "new", "ai_score": 9.0, "time_sensitivity": 1.0,
                 "first_seen": _iso(1)}
        out = ca._rank_by_decayed_score([older, fresh], now=NOW)
        assert [a["_id"] for a in out] == ["new", "old"]

    def test_timeless_item_does_not_decay_below_decayed_peer(self):
        # Same base 8. The time-sensitive one is 20h old (heavy decay); the
        # timeless one (ts=0) keeps its full 8 and must rank first.
        timeless = {"_id": "thesis", "ai_score": 8.0, "time_sensitivity": 0.0,
                    "first_seen": _iso(20)}
        timely = {"_id": "surge", "ai_score": 8.0, "time_sensitivity": 1.0,
                  "first_seen": _iso(20)}
        out = ca._rank_by_decayed_score([timely, timeless], now=NOW)
        assert [a["_id"] for a in out] == ["thesis", "surge"]

    def test_snapshot_rows_stay_pinned_at_top(self):
        # The daemon prepends PORTFOLIO/OPTIONS snapshots with ai_score=10 and
        # NO first_seen. A real fresh article also at ai_score=10 ties on
        # effective score; the *stable* sort must keep the snapshots (which
        # _collapse_syndicated already placed first) ahead of it.
        snap1 = {"_id": "OPT", "title": "OPTIONS SNAPSHOT", "ai_score": 10}
        snap2 = {"_id": "PNL", "title": "PORTFOLIO P&L SNAPSHOT",
                 "ai_score": 10}
        real = {"_id": "real", "ai_score": 10.0, "time_sensitivity": 0.0,
                "first_seen": _iso(0.1)}
        out = ca._rank_by_decayed_score([snap1, snap2, real], now=NOW)
        assert [a["_id"] for a in out][:2] == ["OPT", "PNL"]

    def test_snapshot_outranks_any_decayed_real_article(self):
        snap = {"_id": "PNL", "title": "PORTFOLIO P&L SNAPSHOT", "ai_score": 10}
        # A higher *base* but heavily-decayed article must still sit below the
        # un-decayed snapshot (10).
        real = {"_id": "real", "ai_score": 10.0, "time_sensitivity": 1.0,
                "first_seen": _iso(24)}  # → effective 2.5
        out = ca._rank_by_decayed_score([real, snap], now=NOW)
        assert out[0]["_id"] == "PNL"

    def test_pure_no_input_mutation(self):
        """Load-bearing: the rerank only reshapes the text Opus reads. It must
        not mutate the rows (no ai_score/ml_score/score_source/urgency touch)
        and must return the SAME dict objects (shallow, not copies)."""
        a = {"_id": "a", "ai_score": 5.0, "time_sensitivity": 1.0,
             "first_seen": _iso(10)}
        b = {"_id": "b", "ai_score": 7.0, "time_sensitivity": 0.0,
             "first_seen": _iso(2)}
        snapshot_a = dict(a)
        src = [a, b]
        out = ca._rank_by_decayed_score(src, now=NOW)
        assert src == [a, b]            # input list order untouched
        assert a == snapshot_a          # row dict unmutated
        assert set(id(x) for x in out) == {id(a), id(b)}  # same objects
        assert "ml_score" not in a and "score_source" not in a


# ── integration: _build_payload reflects the decay ordering ─────────────────
class TestBuildPayloadIntegration:
    def test_newswire_section_reordered_by_decay(self, monkeypatch):
        monkeypatch.setattr(ca, "datetime", _FrozenDatetime)
        # Two real articles, same ai_score; the older time-sensitive one must
        # appear AFTER the fresh one in the rendered NEWSWIRE block.
        articles = [
            {"title": "OLD time-sensitive surge headline alpha",
             "source": "rss", "ai_score": 9.0, "time_sensitivity": 1.0,
             "first_seen": _iso(22), "summary": ""},
            {"title": "FRESH same-score headline bravo",
             "source": "rss", "ai_score": 9.0, "time_sensitivity": 1.0,
             "first_seen": _iso(1), "summary": ""},
        ]
        payload = ca._build_payload(articles, {}, [])
        i_fresh = payload.index("FRESH same-score headline bravo")
        i_old = payload.index("OLD time-sensitive surge headline alpha")
        assert i_fresh < i_old, "fresh item must render above the stale one"


class _FrozenDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW if tz else NOW.replace(tzinfo=None)
