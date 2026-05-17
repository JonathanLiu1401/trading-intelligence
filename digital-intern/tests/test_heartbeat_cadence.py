"""Heartbeat-briefing cadence resilience across daemon restarts.

``daemon._initial_heartbeat_last`` regression pin. heartbeat_worker used to
reset its 5h clock to ``time.time()`` on *every* daemon start. The daemon
restarts far more often than 5h under the documented OOM-restart churn
(hundreds of starts/day in the rotated logs; observed briefing gaps of 30-40h
in the `briefings` table vs the 5h target), so each restart silently pushed
the next briefing out a full interval — the analyst's scheduled digest was
being starved. The fix seeds the clock from the last *persisted* briefing.

These cases drive the real ``save_briefing → get_briefings_for_training``
path (not hand-built dicts) and read the live module constants so a retune
cannot false-fail them.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import daemon

HB = daemon.HEARTBEAT_INTERVAL
WARMUP = daemon.HEARTBEAT_RESTART_WARMUP_SECS


def _save_briefing_at(store, dt: datetime) -> None:
    store.save_briefing(dt.isoformat(), "briefing body text", 50)


def test_no_briefing_falls_back_to_now(store):
    """Fresh DB / first-ever launch: preserve the original 'wait a full
    interval' behaviour exactly (seeding to now)."""
    now = 1_000_000.0
    assert daemon._initial_heartbeat_last(store, now=now) == now


def test_recent_briefing_makes_worker_wait_the_remainder(store):
    """A briefing 1h ago must NOT trigger an immediate fire on restart — the
    worker should wait the remaining ~4h."""
    now = datetime.now(timezone.utc).timestamp()
    _save_briefing_at(store, datetime.now(timezone.utc) - timedelta(hours=1))
    last = daemon._initial_heartbeat_last(store, now=now)
    elapsed = now - last
    assert abs(elapsed - 3600) < 5            # clock ≈ 1h since last briefing
    assert elapsed < HB                       # so it does NOT fire immediately


def test_overdue_briefing_fires_after_warmup_not_instantly(store):
    """A briefing 40h ago (overdue under restart-churn) must fire shortly
    after startup — after the warm-up, not instantly and not a full
    interval later."""
    now = datetime.now(timezone.utc).timestamp()
    _save_briefing_at(store, datetime.now(timezone.utc) - timedelta(hours=40))
    last = daemon._initial_heartbeat_last(store, now=now)
    # Seeded exactly to earliest = now - HB + WARMUP → fires in WARMUP seconds.
    assert now - last == HB - WARMUP
    assert 0 < now - last < HB                # neither instant nor a full wait


def test_future_briefing_ts_clamps_to_now(store):
    """Clock skew / a bad future-dated row must not push the clock past now
    (which would suppress briefings); degrade to the original behaviour."""
    now = datetime.now(timezone.utc).timestamp()
    _save_briefing_at(store, datetime.now(timezone.utc) + timedelta(hours=2))
    assert daemon._initial_heartbeat_last(store, now=now) == now


def test_unparseable_briefing_ts_falls_back_to_now(store):
    """A non-ISO ts string must not raise — fall back to now."""
    store.save_briefing("not-a-real-timestamp", "body", 10)
    now = 2_000_000.0
    assert daemon._initial_heartbeat_last(store, now=now) == now


def test_store_error_falls_back_to_now():
    """If the briefings query itself raises, seed to now (never crash the
    heartbeat worker at startup)."""
    class _Boom:
        def get_briefings_for_training(self, limit=1):
            raise RuntimeError("db gone")

    now = 3_000_000.0
    assert daemon._initial_heartbeat_last(_Boom(), now=now) == now


def test_most_recent_briefing_wins(store):
    """get_briefings_for_training is id-DESC; the seed must use the newest row
    even when an older overdue row was inserted first."""
    now = datetime.now(timezone.utc).timestamp()
    _save_briefing_at(store, datetime.now(timezone.utc) - timedelta(hours=40))
    _save_briefing_at(store, datetime.now(timezone.utc) - timedelta(hours=1))
    last = daemon._initial_heartbeat_last(store, now=now)
    assert abs((now - last) - 3600) < 5       # uses the 1h-ago row, not 40h
