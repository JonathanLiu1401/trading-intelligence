"""Tiny exponential-backoff helper for daemon workers.

Usage::

    bo = Backoff(name="rss", base=5.0, cap=300.0)
    while running:
        try:
            do_work()
            bo.reset()
        except Exception as e:
            log.warning(f"[rss] error: {e}; backing off {bo.peek():.0f}s")
            bo.sleep(running_check)
        time.sleep(normal_interval)

After every failure ``sleep()`` waits ``base * 2**failures``, capped at ``cap``,
and increments ``failures``. ``reset()`` zeroes the counter — the worker should
call it after a successful pass.
"""
from __future__ import annotations

import random
import time
from typing import Callable, Optional


class Backoff:
    __slots__ = ("name", "base", "cap", "failures", "jitter")

    def __init__(self, name: str = "worker", base: float = 5.0, cap: float = 300.0,
                 jitter: float = 0.2):
        self.name = name
        self.base = float(base)
        self.cap = float(cap)
        self.failures = 0
        self.jitter = float(jitter)

    def reset(self) -> None:
        self.failures = 0

    def peek(self) -> float:
        """Return the delay that the *next* failure will sleep for, without
        incrementing the counter.

        ``failures`` grows unbounded for a permanently-failing worker, so the
        exponent is clamped at 32: ``base * 2**32`` already dwarfs any sane
        ``cap``, and leaving it unclamped lets ``2 ** failures`` overflow into
        an int too large to convert to float (``OverflowError``) after ~1024
        failures, crashing the worker."""
        exp = self.failures if self.failures < 32 else 32
        delay = self.base * (2 ** exp)
        return min(self.cap, delay)

    def _next_delay(self) -> float:
        delay = self.peek()
        self.failures += 1
        if self.jitter > 0:
            delay *= 1.0 + random.uniform(-self.jitter, self.jitter)
        return max(0.5, delay)

    def sleep(self, should_continue: Optional[Callable[[], bool]] = None) -> float:
        """Sleep for the next backoff window. If ``should_continue`` is given,
        check it every 0.5s so a shutdown signal doesn't have to wait the full
        delay. Returns the delay actually waited."""
        delay = self._next_delay()
        if should_continue is None:
            time.sleep(delay)
            return delay
        deadline = time.time() + delay
        while time.time() < deadline:
            if not should_continue():
                break
            time.sleep(min(0.5, max(0.0, deadline - time.time())))
        return delay


if __name__ == "__main__":  # smoke test
    bo = Backoff(name="t", base=5.0, cap=300.0, jitter=0.0)
    assert bo.peek() == 5.0
    bo.failures = 6
    assert bo.peek() == 300.0  # 5 * 2**6 = 320, capped at 300
    bo.failures = 100_000  # would OverflowError without exponent clamp
    assert bo.peek() == 300.0
    print("OK")
