"""Exact-value tests for analytics/feed_health.build_feed_health.

The feature's whole point: a 0-signal HOLD is invisible in every existing
panel, so the detector must (a) prove blindness from a *streak* of 0-signal
decisions, (b) age the resolved DB the trader actually reads, and (c) catch
the split-brain where signals._db_path() (USB-first) and the daemon /
unified_dashboard (local-first) resolve articles.db with opposite precedence.
These pin the exact blind-streak arithmetic, the freshness / split-brain
boundaries, the NO_DATA / BLIND / STALE_FEED / HEALTHY verdict precedence, and
the constant echo.
"""
from datetime import datetime, timedelta, timezone

from paper_trader.analytics.feed_health import (
    build_feed_health,
    BLIND_STREAK_MIN,
    STALE_HOURS,
    SPLIT_BRAIN_GAP_H,
)

NOW = datetime(2026, 5, 16, 14, 0, 0, tzinfo=timezone.utc)
USB = "/media/zeph/projects/digital-intern/db/articles.db"
LOCAL = "/home/zeph/digital-intern/data/articles.db"


def _dec(minutes_ago, signal_count):
    """One decisions-table row dict (store.recent_decisions shape)."""
    return {
        "timestamp": (NOW - timedelta(minutes=minutes_ago)).isoformat(),
        "market_open": 1,
        "signal_count": signal_count,
        "action_taken": "HOLD NVDA → HOLD" if signal_count else "HOLD NONE → HOLD",
        "reasoning": "{}",
        "portfolio_value": 972.0,
        "cash": 6.0,
    }


def _iso(hours_ago):
    return (NOW - timedelta(hours=hours_ago)).isoformat()


def _feed(resolved_path=USB, resolved_age_h=0.1, live_2h=12, live_24h=300,
          candidates=None):
    """A resolved-feed dict (what the endpoint hands the pure builder)."""
    newest = _iso(resolved_age_h) if resolved_age_h is not None else None
    if candidates is None:
        candidates = [{"path": resolved_path, "exists": True, "newest": newest}]
    return {
        "resolved_path": resolved_path,
        "resolved_newest": newest,
        "resolved_live_2h": live_2h,
        "resolved_live_24h": live_24h,
        "candidates": candidates,
    }


# ─── NO_DATA ────────────────────────────────────────────────────────────────

def test_empty_everything_is_no_data():
    out = build_feed_health([], {}, now=NOW)
    assert out["verdict"] == "NO_DATA"
    assert out["resolved_path"] is None
    assert out["blind_streak"] == 0
    assert out["n_decisions"] == 0
    assert out["split_brain"] is False
    assert out["restart_recommended"] is False
    assert "NO_DATA" in out["headline"]


def test_no_resolved_db_is_no_data_even_with_decisions():
    out = build_feed_health([_dec(1, 0), _dec(2, 0)],
                            {"resolved_path": None}, now=NOW)
    assert out["verdict"] == "NO_DATA"
    assert out["blind_streak"] == 2  # still counted, but path absence wins


# ─── HEALTHY ────────────────────────────────────────────────────────────────

def test_fresh_feed_with_signals_is_healthy():
    out = build_feed_health(
        [_dec(1, 5), _dec(60, 3), _dec(120, 0)],
        _feed(resolved_age_h=0.2, live_2h=8, live_24h=400),
        now=NOW,
    )
    assert out["verdict"] == "HEALTHY"
    assert out["blind_streak"] == 0
    assert out["resolved_newest_age_h"] == 0.2
    assert out["resolved_live_2h"] == 8
    assert out["split_brain"] is False
    assert out["restart_recommended"] is False


# ─── BLIND ──────────────────────────────────────────────────────────────────

def test_consecutive_zero_signal_streak_is_blind():
    decs = [_dec(i, 0) for i in range(14)]  # 14 in a row
    out = build_feed_health(decs, _feed(resolved_age_h=20.0, live_2h=0),
                            now=NOW)
    assert out["verdict"] == "BLIND"
    assert out["blind_streak"] == 14
    assert "14 consecutive" in out["headline"]


def test_blind_streak_stops_at_first_signal():
    # newest 4 are 0-signal, then a cycle that DID see signals → streak == 4
    decs = [_dec(1, 0), _dec(2, 0), _dec(3, 0), _dec(4, 0),
            _dec(5, 7), _dec(6, 0), _dec(7, 0)]
    out = build_feed_health(decs, _feed(resolved_age_h=0.1), now=NOW)
    assert out["blind_streak"] == 4
    assert out["verdict"] == "BLIND"  # 4 >= BLIND_STREAK_MIN, fresh-DB notwithstanding


def test_missing_signal_count_breaks_streak_conservatively():
    decs = [_dec(1, 0), _dec(2, 0), {"timestamp": NOW.isoformat()}, _dec(4, 0)]
    out = build_feed_health(decs, _feed(resolved_age_h=0.1), now=NOW)
    assert out["blind_streak"] == 2  # stops at the dict with no signal_count


# ─── STALE_FEED ─────────────────────────────────────────────────────────────

def test_stale_feed_without_blind_streak():
    # feed 7.5h stale (> STALE_HOURS) but only 2 recent 0-signal decisions
    out = build_feed_health(
        [_dec(1, 0), _dec(2, 0), _dec(3, 9)],
        _feed(resolved_age_h=7.5, live_2h=0),
        now=NOW,
    )
    assert out["verdict"] == "STALE_FEED"
    assert out["blind_streak"] == 2
    assert out["resolved_newest_age_h"] == 7.5


def test_no_live_article_ever_is_stale_not_healthy():
    out = build_feed_health(
        [_dec(1, 4)],
        _feed(resolved_age_h=None),  # resolved DB has no live article at all
        now=NOW,
    )
    assert out["verdict"] == "STALE_FEED"
    assert out["resolved_newest_age_h"] is None
    assert "no live article ever" in out["headline"]


# ─── verdict precedence (locked) ────────────────────────────────────────────

class TestVerdictPrecedence:
    def test_blind_outranks_stale_feed(self):
        # both true: 5 zero-signal decisions AND a 20h-stale feed → BLIND wins
        out = build_feed_health([_dec(i, 0) for i in range(5)],
                                _feed(resolved_age_h=20.0), now=NOW)
        assert out["verdict"] == "BLIND"

    def test_no_data_outranks_blind(self):
        out = build_feed_health([_dec(i, 0) for i in range(9)],
                                {"resolved_path": None}, now=NOW)
        assert out["verdict"] == "NO_DATA"

    def test_blind_streak_boundary(self):
        below = build_feed_health([_dec(i, 0) for i in range(BLIND_STREAK_MIN - 1)],
                                  _feed(resolved_age_h=0.1), now=NOW)
        assert below["blind_streak"] == BLIND_STREAK_MIN - 1
        assert below["verdict"] == "HEALTHY"  # fresh, streak below threshold
        at = build_feed_health([_dec(i, 0) for i in range(BLIND_STREAK_MIN)],
                               _feed(resolved_age_h=0.1), now=NOW)
        assert at["blind_streak"] == BLIND_STREAK_MIN
        assert at["verdict"] == "BLIND"

    def test_freshness_boundary(self):
        fresh = build_feed_health([_dec(1, 4)],
                                  _feed(resolved_age_h=STALE_HOURS - 0.1),
                                  now=NOW)
        assert fresh["verdict"] == "HEALTHY"
        stale = build_feed_health([_dec(1, 4)],
                                  _feed(resolved_age_h=STALE_HOURS + 0.1),
                                  now=NOW)
        assert stale["verdict"] == "STALE_FEED"


# ─── split-brain detection ──────────────────────────────────────────────────

class TestSplitBrain:
    def _split_feed(self, resolved_age_h):
        cands = [
            {"path": USB, "exists": True, "newest": _iso(resolved_age_h)},
            {"path": LOCAL, "exists": True, "newest": _iso(0.1)},  # fresh
        ]
        return _feed(resolved_path=USB, resolved_age_h=resolved_age_h,
                     live_2h=0, candidates=cands)

    def test_split_brain_flagged_and_recommends_restart(self):
        out = build_feed_health([_dec(i, 0) for i in range(14)],
                                self._split_feed(19.8), now=NOW)
        assert out["split_brain"] is True
        assert out["restart_recommended"] is True
        assert out["fresher_path"] == LOCAL
        assert out["fresher_age_h"] == 0.1
        assert out["verdict"] == "BLIND"
        assert "split-brain" in out["headline"]
        assert LOCAL in out["headline"] and USB in out["headline"]

    def test_no_split_brain_when_alt_not_materially_fresher(self):
        # both candidates ~equally stale → systemic quiet period, not split-brain
        cands = [
            {"path": USB, "exists": True, "newest": _iso(7.0)},
            {"path": LOCAL, "exists": True, "newest": _iso(7.2)},
        ]
        out = build_feed_health(
            [_dec(1, 0)],
            _feed(resolved_path=USB, resolved_age_h=7.0, candidates=cands),
            now=NOW,
        )
        assert out["split_brain"] is False
        assert out["restart_recommended"] is False
        assert out["verdict"] == "STALE_FEED"

    def test_split_brain_gap_boundary(self):
        # resolved 7h stale; alt exactly SPLIT_BRAIN_GAP_H fresher → split-brain
        cands = [
            {"path": USB, "exists": True, "newest": _iso(STALE_HOURS + 1.0)},
            {"path": LOCAL, "exists": True,
             "newest": _iso(STALE_HOURS + 1.0 - SPLIT_BRAIN_GAP_H)},
        ]
        out = build_feed_health(
            [_dec(1, 0)],
            _feed(resolved_path=USB, resolved_age_h=STALE_HOURS + 1.0,
                  candidates=cands),
            now=NOW,
        )
        assert out["split_brain"] is True

    def test_fresh_resolved_db_is_never_split_brain(self):
        # trader reads the fresh one; the stale USB mirror is irrelevant
        cands = [
            {"path": LOCAL, "exists": True, "newest": _iso(0.1)},
            {"path": USB, "exists": True, "newest": _iso(20.0)},
        ]
        out = build_feed_health(
            [_dec(1, 5)],
            _feed(resolved_path=LOCAL, resolved_age_h=0.1, candidates=cands),
            now=NOW,
        )
        assert out["split_brain"] is False
        assert out["verdict"] == "HEALTHY"


# ─── candidate echo + constants ─────────────────────────────────────────────

def test_candidates_echoed_with_computed_age():
    cands = [
        {"path": USB, "exists": True, "newest": _iso(19.8)},
        {"path": LOCAL, "exists": False, "newest": None},
    ]
    out = build_feed_health(
        [_dec(1, 0)],
        _feed(resolved_path=USB, resolved_age_h=19.8, candidates=cands),
        now=NOW,
    )
    by_path = {c["path"]: c for c in out["candidates"]}
    assert by_path[USB]["age_h"] == 19.8
    assert by_path[USB]["exists"] is True
    assert by_path[LOCAL]["exists"] is False
    assert by_path[LOCAL]["age_h"] is None


def test_constants_echoed():
    out = build_feed_health([], {}, now=NOW)
    assert out["blind_streak_min"] == BLIND_STREAK_MIN
    assert out["stale_hours"] == STALE_HOURS
    assert out["split_brain_gap_h"] == SPLIT_BRAIN_GAP_H
