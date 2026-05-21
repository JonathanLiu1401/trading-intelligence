"""Centralised HTTP GET with rate-limit (429) and 5xx backoff.

Replaces the ad-hoc ``if r.status_code == 429: print(...)`` snippets scattered
across collectors. ``backoff_get`` honours the upstream ``Retry-After`` header
on 429 responses (capped at 60s so a hostile server can't pin a worker), and
falls back to exponential backoff ``base_delay * 2**attempt`` on transient
5xx errors. All sleeps carry +30% positive jitter to de-correlate a herd of
workers that all hit a rate limit at the same instant.

Non-retryable responses (2xx, 3xx, and 4xx other than 429) return immediately
— the caller decides what a 404 / 403 means in its domain. Network/connection
errors raised by ``requests`` propagate as ``None`` only after exhausting
retries; the first attempt's exception is logged at WARNING.

Per-host 429 counts are kept in a module-level dict guarded by a lock so
multiple collectors threading through the same module observe consistent
totals. ``get_rate_limit_stats()`` returns a snapshot copy.
"""
from __future__ import annotations

import random
import threading
import time
from typing import Optional
from urllib.parse import urlparse

import requests

from core.logger import get_logger

log = get_logger("rate_limit")

_MAX_RETRY_AFTER_SECONDS = 60.0
_JITTER_FRACTION = 0.30
_DEFAULT_TIMEOUT = 30.0

_stats_lock = threading.Lock()
_rate_limit_hits: dict[str, int] = {}


def _record_429(url: str) -> None:
    host = urlparse(url).hostname or "unknown"
    with _stats_lock:
        _rate_limit_hits[host] = _rate_limit_hits.get(host, 0) + 1


def _parse_retry_after(value: str | None) -> float | None:
    """Parse the Retry-After header. Returns seconds, or None if unparseable.

    RFC 7231 allows both a delta-seconds integer ("30") and an HTTP-date.
    Most APIs that bother sending the header use delta-seconds, so we only
    handle that form — an HTTP-date falls through to the default backoff."""
    if not value:
        return None
    try:
        return max(0.0, float(value.strip()))
    except (TypeError, ValueError):
        return None


def _jittered(delay: float) -> float:
    """Add +0..30% positive jitter. Never reduces the wait below the
    server-requested value — important for Retry-After compliance."""
    return delay * (1.0 + random.uniform(0.0, _JITTER_FRACTION))


def backoff_get(
    url: str,
    session: Optional[requests.Session] = None,
    collector_name: str = "unknown",
    max_retries: int = 3,
    base_delay: float = 2.0,
    **kwargs,
) -> Optional[requests.Response]:
    """HTTP GET with 429/5xx-aware retry. Returns the final Response, or None
    if every attempt raised a network error.

    ``**kwargs`` is forwarded to ``requests.get`` / ``Session.get`` so callers
    can pass ``params=``, ``headers=``, ``timeout=``, etc. A default 30s
    timeout is injected if the caller omits one — collectors that previously
    used module-level ``HTTP_TIMEOUT`` should pass it explicitly."""
    kwargs.setdefault("timeout", _DEFAULT_TIMEOUT)
    getter = session.get if session is not None else requests.get

    last_response: Optional[requests.Response] = None
    for attempt in range(max_retries + 1):
        try:
            response = getter(url, **kwargs)
        except requests.RequestException as e:
            if attempt >= max_retries:
                log.warning(
                    f"[{collector_name}] GET {url} failed after "
                    f"{attempt + 1} attempts: {e}"
                )
                return None
            delay = _jittered(base_delay * (2 ** attempt))
            log.warning(
                f"[{collector_name}] GET {url} network error "
                f"(attempt {attempt + 1}/{max_retries + 1}): {e}; "
                f"retrying in {delay:.1f}s"
            )
            time.sleep(delay)
            continue

        last_response = response
        status = response.status_code

        if status == 429:
            _record_429(url)
            if attempt >= max_retries:
                log.warning(
                    f"[{collector_name}] GET {url} still 429 after "
                    f"{attempt + 1} attempts; giving up"
                )
                return response
            retry_after = _parse_retry_after(response.headers.get("Retry-After"))
            base = retry_after if retry_after is not None else base_delay * (2 ** attempt)
            delay = _jittered(min(base, _MAX_RETRY_AFTER_SECONDS))
            log.warning(
                f"[{collector_name}] GET {url} rate-limited (429) "
                f"(attempt {attempt + 1}/{max_retries + 1}); "
                f"sleeping {delay:.1f}s "
                f"(Retry-After={retry_after})"
            )
            time.sleep(delay)
            continue

        if 500 <= status < 600:
            if attempt >= max_retries:
                log.warning(
                    f"[{collector_name}] GET {url} returned {status} after "
                    f"{attempt + 1} attempts; giving up"
                )
                return response
            delay = _jittered(base_delay * (2 ** attempt))
            log.warning(
                f"[{collector_name}] GET {url} returned {status} "
                f"(attempt {attempt + 1}/{max_retries + 1}); "
                f"retrying in {delay:.1f}s"
            )
            time.sleep(delay)
            continue

        return response

    return last_response


def get_rate_limit_stats() -> dict[str, int]:
    """Return a snapshot of 429 hit counts keyed by hostname."""
    with _stats_lock:
        return dict(_rate_limit_hits)
