"""Pins `core/backoff.Backoff` — the exponential-backoff helper every collector
worker in `daemon.py` (~20 call sites) uses to throttle its retry loop.

The module had only an inline `__main__` smoke test; this is its first real
suite. It documents the *actual* contract (the code is the spec, per AGENTS.md):

- `peek()` is non-mutating and clamps the exponent at 32 so a permanently
  failing worker can't `OverflowError` on `2 ** failures` (the regression the
  clamp defends).
- jitter is applied *after* the cap, so the realized sleep is
  `cap_delay * (1 ± jitter)` — i.e. it can sit slightly above the nominal `cap`
  by design (de-correlates a thundering herd of workers all failing at once).
  This is intentional; the cap bounds the *base* delay, not the jittered one.
- the 0.5s floor on the realized delay.
- `sleep(should_continue)` polls the predicate so a shutdown doesn't have to
  wait the full window.
"""
from __future__ import annotations

import time

import pytest

from core.backoff import Backoff


class TestPeek:
    def test_peek_is_non_mutating(self):
        bo = Backoff(base=5.0, cap=300.0)
        before = bo.failures
        bo.peek()
        bo.peek()
        assert bo.failures == before == 0

    def test_growth_then_cap(self):
        bo = Backoff(base=5.0, cap=300.0)
        assert bo.peek() == 5.0          # 5 * 2**0
        bo.failures = 3
        assert bo.peek() == 40.0         # 5 * 2**3
        bo.failures = 6
        assert bo.peek() == 300.0        # 5 * 2**6 = 320 -> capped

    def test_exponent_clamped_no_overflow(self):
        """Without the exponent clamp, `2 ** failures` becomes an int too large
        to convert to float (~after 1024 failures) and crashes the worker."""
        bo = Backoff(base=5.0, cap=300.0)
        bo.failures = 100_000
        assert bo.peek() == 300.0        # does not raise OverflowError

    def test_clamp_boundary_at_32(self):
        bo = Backoff(base=1.0, cap=1e18, jitter=0.0)
        bo.failures = 32
        at_32 = bo.peek()
        bo.failures = 5000
        assert bo.peek() == at_32        # everything >=32 collapses to the same delay


class TestReset:
    def test_reset_zeroes_counter(self):
        bo = Backoff(base=5.0, cap=300.0, jitter=0.0)
        bo._next_delay()
        bo._next_delay()
        assert bo.failures == 2
        bo.reset()
        assert bo.failures == 0
        assert bo.peek() == 5.0


class TestNextDelay:
    def test_increments_failures(self):
        bo = Backoff(base=5.0, cap=300.0, jitter=0.0)
        bo._next_delay()
        assert bo.failures == 1
        bo._next_delay()
        assert bo.failures == 2

    def test_no_jitter_is_exact(self):
        bo = Backoff(base=5.0, cap=300.0, jitter=0.0)
        assert bo._next_delay() == 5.0

    def test_jitter_band_and_applied_after_cap(self, monkeypatch):
        """Jitter is intentionally applied *after* the cap. With the multiplier
        pinned to its max, the realized delay sits ABOVE the nominal cap — this
        is by design (anti-thundering-herd); the cap bounds the base delay only.
        """
        bo = Backoff(base=5.0, cap=300.0, jitter=0.2)
        bo.failures = 10                                  # base delay -> capped at 300
        monkeypatch.setattr("core.backoff.random.uniform", lambda a, b: b)  # +jitter
        d = bo._next_delay()
        assert d == pytest.approx(300.0 * 1.2)            # 360.0 > cap, intentionally

        bo.failures = 10
        monkeypatch.setattr("core.backoff.random.uniform", lambda a, b: a)  # -jitter
        d = bo._next_delay()
        assert d == pytest.approx(300.0 * 0.8)

    def test_floor_is_half_second(self, monkeypatch):
        bo = Backoff(base=0.01, cap=300.0, jitter=0.9)
        monkeypatch.setattr("core.backoff.random.uniform", lambda a, b: a)  # drive toward 0
        assert bo._next_delay() == 0.5


class TestSleep:
    def test_sleep_returns_waited_delay(self, monkeypatch):
        bo = Backoff(base=5.0, cap=300.0, jitter=0.0)
        slept = []
        monkeypatch.setattr("core.backoff.time.sleep", lambda s: slept.append(s))
        assert bo.sleep() == 5.0
        assert slept == [5.0]

    def test_should_continue_short_circuits(self, monkeypatch):
        """A shutdown signal must not have to wait the full backoff window."""
        bo = Backoff(base=100.0, cap=300.0, jitter=0.0)
        sleeps = []
        monkeypatch.setattr("core.backoff.time.sleep", lambda s: sleeps.append(s))
        calls = {"n": 0}

        def should_continue():
            calls["n"] += 1
            return calls["n"] < 3        # stop after 2 poll iterations

        waited = bo.sleep(should_continue)
        assert waited == 100.0           # reports the intended window
        # broke out via the predicate, not by sleeping the whole 100s
        assert sum(sleeps) < 100.0
        assert all(s <= 0.5 for s in sleeps)

    def test_should_continue_true_runs_to_deadline(self, monkeypatch):
        bo = Backoff(base=1.0, cap=300.0, jitter=0.0)
        clock = {"t": 1000.0}
        monkeypatch.setattr("core.backoff.time.time", lambda: clock["t"])

        def fake_sleep(s):
            clock["t"] += max(s, 0.0001)  # advance virtual clock; avoid 0-step stall

        monkeypatch.setattr("core.backoff.time.sleep", fake_sleep)
        waited = bo.sleep(lambda: True)
        assert waited == 1.0
        assert clock["t"] >= 1001.0       # ran to the full deadline
