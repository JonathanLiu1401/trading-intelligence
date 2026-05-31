"""Per-ticker quant cache health snapshot.

`paper_trader/strategy.py` keeps two caches around `get_quant_signals_live`:

  * ``_QUANT_CACHE`` (positive): the last successful (RSI/MACD/momentum/etc.)
    record per ticker, with the `_time.time()` stamp it was written.
  * ``_QUANT_NEG_CACHE`` (negative): tickers whose yfinance lookup returned
    empty / raised / had fewer than 60 closes — symbols the live trader is
    currently flying blind on this TTL window.

Today both caches are internal to ``strategy.py``. There is no operator
surface that answers the live trader's question: *which of my 50 watchlist
symbols have a stale or DARK quant signal feeding the decision prompt right
now?* A symbol that just expired out of ``_QUANT_NEG_CACHE`` and a symbol
whose positive entry is 4 min 30 s old behave very differently next cycle —
this builder makes the difference visible without parsing log lines.

Composes pure: reads only the two cache dicts (snapshot copies so a
concurrent ``get_quant_signals_live`` write can't mutate mid-scan), never
hits yfinance, never raises. Mirrors the discipline of
``market.dead_tickers()`` / ``alarm_latch_state()`` / ``notify_health()`` —
the same "expose internal state for the operator panel" pattern this
project has standardised on. Returning an envelope (``verdict`` + counts +
per-ticker rows) lets a dashboard panel render either branch off a single
code path.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone


# Operator-visible verdicts. Keep these stable — the dashboard panel keys
# CSS off them and a CI test pins the contract.
VERDICT_HEALTHY = "HEALTHY"        # every requested ticker has a fresh hit
VERDICT_DEGRADED = "DEGRADED"      # ≥1 ticker is dark (negative-cached) right now
VERDICT_EMPTY = "EMPTY"            # both caches are empty — engine just booted


def _snapshot_caches(strategy_mod):
    """Take a shallow snapshot of the live caches so the scan is atomic.

    The runner thread and the dashboard thread both reach ``strategy.py``
    concurrently. Iterating a dict while another thread mutates it can
    raise ``RuntimeError``; copying first avoids that without holding a
    lock the producer side doesn't take. ``list(...)`` is the same
    discipline ``market.dead_tickers`` uses for ``_DEAD_CACHE``.
    """
    pos_items = list(getattr(strategy_mod, "_QUANT_CACHE", {}).items())
    neg_items = list(getattr(strategy_mod, "_QUANT_NEG_CACHE", {}).items())
    return pos_items, neg_items


def _seconds_since(ts: float, now: float) -> int:
    """Wall-clock-step-back-safe age in seconds. Negative spans clamp to 0
    so a NTP correction never renders a negative age (the same hardening
    pattern ``alarm_latch_state`` and ``store._hold_duration`` use)."""
    try:
        return max(0, int(now - float(ts)))
    except (TypeError, ValueError):
        return 0


def _select_key_fields(rec: dict) -> dict:
    """The handful of quant fields a trader actually wants at a glance.

    The full quant record has ~15 fields; surfacing all of them would make
    the endpoint payload as heavy as ``/api/state``. We pick the four that
    the live decision prompt's TECHNICAL block already renders prominently
    (RSI, MACD label, mom_5d, mom_20d) plus the macd_below_zero_cross flag
    (the high-quality entry signal documented in SYSTEM_PROMPT). Missing
    keys degrade to None — never raise; a malformed cache row from a
    future schema change must not break this endpoint.
    """
    if not isinstance(rec, dict):
        return {}
    out: dict = {}
    for key in ("RSI", "MACD", "mom_5d", "mom_20d", "macd_below_zero_cross"):
        out[key] = rec.get(key)
    return out


def build_quant_cache_status(
    strategy_mod,
    *,
    requested: list[str] | None = None,
    now: float | None = None,
) -> dict:
    """Pure roll-up over ``strategy._QUANT_CACHE`` + ``_QUANT_NEG_CACHE``.

    ``requested`` is the optional set of tickers the operator cares about
    (typically WATCHLIST + held positions). When given, the response keys
    each requested ticker as one of:

      * ``"FRESH"`` — present in the positive cache, age < TTL
      * ``"STALE"`` — present in the positive cache, age ≥ positive TTL
        (would be re-fetched on the next ``get_quant_signals_live`` call)
      * ``"DARK"``  — present in the negative cache, age < neg TTL
        (yfinance is being suppressed this window)
      * ``"NEVER"`` — never fetched (or evicted from both caches)

    When ``requested`` is None, the response returns every ticker currently
    in either cache — the dashboard panel can poll without a filter to
    inspect the whole engine state.

    Returns a dict with the same envelope shape as ``/api/dead-tickers``:
    ``as_of``, ``service``, ``verdict``, counts, per-ticker rows. Never
    raises — every guard degrades to an ERROR envelope on the catch arm.
    """
    try:
        now_ts = float(now) if now is not None else time.time()
        pos_items, neg_items = _snapshot_caches(strategy_mod)

        # Lookup tables for fast O(1) status decoding inside the requested loop.
        pos_by_t = {t: (rec, ts) for t, (rec, ts) in pos_items}
        neg_by_t = {t: ts for t, ts in neg_items}

        # TTLs are module globals — read them dynamically so a future bump
        # to a different value is honored without an endpoint change.
        pos_ttl = float(getattr(strategy_mod, "_QUANT_TTL", 300.0))
        neg_ttl = float(getattr(strategy_mod, "_QUANT_NEG_TTL", 300.0))

        # Build the row list. Dedupe / normalize the requested tickers if a
        # caller passes duplicates or non-strings (defensive — the endpoint
        # accepts arbitrary `?tickers=A,B,A` input).
        if requested is None:
            requested_clean = sorted(set(pos_by_t.keys()) | set(neg_by_t.keys()))
        else:
            seen: set[str] = set()
            requested_clean = []
            for t in requested:
                if not isinstance(t, str):
                    continue
                u = t.strip().upper()
                if not u or u in seen:
                    continue
                seen.add(u)
                requested_clean.append(u)

        rows: list[dict] = []
        n_fresh = n_stale = n_dark = n_never = 0
        for tk in requested_clean:
            neg_ts = neg_by_t.get(tk)
            if neg_ts is not None and (now_ts - float(neg_ts)) < neg_ttl:
                # DARK wins over STALE: a symbol marked dark *after* a stale
                # positive entry is currently being suppressed; surfacing the
                # stale positive value would mislead the operator into
                # thinking the next cycle has data to fall back on.
                rows.append({
                    "ticker": tk,
                    "status": "DARK",
                    "neg_age_s": _seconds_since(neg_ts, now_ts),
                    "neg_ttl_remaining_s": max(
                        0, int(neg_ttl - (now_ts - float(neg_ts)))
                    ),
                })
                n_dark += 1
                continue
            pos_rec = pos_by_t.get(tk)
            if pos_rec is not None:
                rec, ts = pos_rec
                age = _seconds_since(ts, now_ts)
                is_fresh = age < int(pos_ttl)
                row = {
                    "ticker": tk,
                    "status": "FRESH" if is_fresh else "STALE",
                    "age_s": age,
                    "ttl_remaining_s": (
                        max(0, int(pos_ttl - age)) if is_fresh else 0
                    ),
                    "signals": _select_key_fields(rec),
                }
                rows.append(row)
                if is_fresh:
                    n_fresh += 1
                else:
                    n_stale += 1
                continue
            # Neither cache holds this ticker → never fetched (or evicted).
            rows.append({"ticker": tk, "status": "NEVER"})
            n_never += 1

        n_total = len(rows)
        if n_total == 0:
            verdict = VERDICT_EMPTY
        elif n_dark > 0:
            verdict = VERDICT_DEGRADED
        else:
            verdict = VERDICT_HEALTHY

        return {
            "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "verdict": verdict,
            "n_total": n_total,
            "n_fresh": n_fresh,
            "n_stale": n_stale,
            "n_dark": n_dark,
            "n_never": n_never,
            "pos_ttl_s": int(pos_ttl),
            "neg_ttl_s": int(neg_ttl),
            "rows": rows,
        }
    except Exception as e:  # never let a builder fault crash the endpoint
        return {
            "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "verdict": "ERROR",
            "headline": f"quant_cache_status builder error: {e}",
            "n_total": 0,
            "n_fresh": 0,
            "n_stale": 0,
            "n_dark": 0,
            "n_never": 0,
            "pos_ttl_s": 0,
            "neg_ttl_s": 0,
            "rows": [],
        }
