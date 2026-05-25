"""Main loop — drives the paper trader, runs the dashboard, dispatches Discord reports."""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import threading
import time
import traceback
from collections import namedtuple
from datetime import datetime, timezone

from zoneinfo import ZoneInfo

from . import market, reporter, strategy
from .analytics.dynamic_interval import compute_interval
from .store import DB_PATH, get_store

NY = ZoneInfo("America/New_York")
OPEN_INTERVAL_S = 1800      # decide every 30 min when market is open
CLOSED_INTERVAL_S = 3600    # every 1 hour when closed
# Grace after the NYSE session close before the daily-close report fires. The
# trigger is anchored to the *actual* session close (market.close_minute):
# 16:00 ET on a regular day → fires 16:05 ET (unchanged), but 13:00 ET on a
# NYSE early-close half-day (day-after-Thanksgiving / Christmas Eve) → fires
# 13:05 ET instead of sitting on a frozen, post-close book for three extra
# hours waiting for a hardcoded 16:05 that no longer matches the bell.
DAILY_CLOSE_GRACE_MIN = 5

# Git-watcher deadman: once a deferred restart is requested, the MAIN LOOP is
# the graceful actor (it exits at the next cycle boundary so a mid-Opus call
# is never killed). But under heavy host load (observed live: load avg ~23,
# a 3-day-uptime runner still on stale code with a committed fix never
# deployed) the loop can be wedged so long the boundary never arrives — the
# fix sits unapplied for days. A *healthy* cycle is hard-bounded by the
# strategy claude budgets (DECISION_TIMEOUT_S 180 + RETRY 45 + FALLBACK 60
# ≈ 5 min) plus the 180s poll cadence; if the deferred restart is still
# unhonored this long after it was requested the loop is genuinely wedged,
# not healthily mid-decision, so the watcher force-exits as a last resort
# (systemd Restart=always reboots onto fresh code). 600s leaves comfortable
# margin above the worst-case healthy cycle so a slow-but-live loop is never
# force-killed.
RESTART_GRACE_S = 600

# Auto-recovery circuit breaker: after this many consecutive cycles that
# produced no decision (Opus + Sonnet fallback both timed out / unparseable),
# kill any lingering claude subprocess so a wedged CLI can't keep starving the
# decision loop. strategy.decide() already returns status="NO_DECISION" for
# every failed cycle, so we key off summary["status"] (not decision["action"]).
CONSECUTIVE_NO_DECISION_LIMIT = 5


_daily_close_sent_for: str | None = None
_last_hourly: datetime | None = None
_consecutive_no_decisions = 0
# Dedupe latch for the "Claude quota exhausted" Discord alarm. Set True after
# the alarm is sent; cleared (re-armed) only when a real decision confirms the
# quota is back. Module global — deliberately NOT persisted to the runner_state
# sidecar: on a restart the next cycle's decide() re-detects an ongoing outage
# and re-alarms, which is the correct behaviour (a fresh process SHOULD tell
# the operator it is still frozen).
_quota_alert_active = False
# Companion latch for the consecutive-NO_DECISION circuit-breaker alarm.
# Same dedupe pattern as `_quota_alert_active`: set True the cycle the breaker
# fires (a Discord alert went out), cleared on the next real decision (engine
# is responding again — re-arm so a fresh wedge tomorrow re-alerts). Distinct
# latch from the quota one because the two failure modes are independent
# (a wedge is "Claude not responding"; quota is "Claude rejecting") and the
# operator action differs (a breaker fire wants escalation, a quota wants
# patience until the reset).
_breaker_alert_active = False
# Wall-clock timestamp (UTC) of the first NO_DECISION in the current run of
# consecutive NO_DECISION cycles. Used by the breaker Discord alarm to
# surface the REAL elapsed-time of the wedge instead of multiplying the count
# by a hardcoded 30-min cadence — under dynamic_interval the actual cycle gap
# is anywhere from 60s (earnings window) to 90 min (quiet-closed), so the old
# ``consecutive * 30`` math wildly mis-stated the freeze duration. Reset to
# None whenever the engine produces a real decision. Module global (in-memory
# only — a fresh process re-derives it on its next NO_DECISION).
_no_decision_first_ts: datetime | None = None
# Companion to ``_no_decision_first_ts`` for the quota-exhaustion path.
# The quota cycle deliberately resets ``_no_decision_first_ts`` to None on
# every quota tick (so a NON-quota wedge that follows starts its own clock
# fresh — see the quota arm of ``_cycle``), so the breaker's timestamp is
# unusable for the orthogonal quota bracket. This dedicated marker captures
# the wall-clock instant of the FIRST quota cycle in the current outage and
# is reset by the next real decision (HOLD / FILLED / BLOCKED) — exactly
# mirroring the breaker pattern one dimension over. Feeds the
# ``send_quota_recovered_alert(elapsed_s)`` token so the operator gets the
# same self-describing EXHAUSTED→RECOVERED bracket the breaker FIRED→CLEARED
# pair already provides.
_quota_first_ts: datetime | None = None

# Set by the git-watcher thread; checked by the main loop between cycles so
# a restart never kills a mid-Opus decision call.
_restart_requested = threading.Event()


# ── Restart-durable report markers ───────────────────────────────────────
# `_daily_close_sent_for` / `_last_hourly` are module globals — lost on every
# process restart, and the runner restarts often (a `/api/build-info` stale
# restart to apply a committed fix, systemd, the circuit breaker, an operator
# bounce). Two trader-visible failures result:
#
#   1. Hourly STARVATION. `main()` deliberately anchors `_last_hourly` to boot
#      so the first summary lands ~1h in (not alongside the online ping). But
#      a runner that bounces more often than hourly *never* sends an hourly
#      summary at all — every boot resets the 1h clock. ("I haven't gotten an
#      hourly in hours.")
#   2. Daily-close DUPLICATION. `_daily_close_sent_for` resets to None on
#      restart, so a bounce after 16:05 NY on a day the close already fired
#      re-posts a second "DAILY CLOSE" with the same numbers.
#
# A tiny JSON sidecar next to paper_trader.db makes both markers survive a
# restart. Deliberately NOT a new store.py table: SCHEMA is load-bearing
# (invariant #13) and this needs no WAL/locking — it is single-writer,
# best-effort, and a lost/corrupt file must degrade to "behave like today
# (in-memory only)", never crash the daemon loop.
_STATE_PATH = DB_PATH.parent / "runner_state.json"


# ── Single-instance guard ────────────────────────────────────────────────
# Two concurrent runners on the same $1000 paper book is a real, *observed*
# live pathology (2026-05-17: an orphaned manual launch under PID 1 AND a
# systemd-managed instance both cycling `runner.py`, double-trading the same
# `paper_trader.db`, doubling the concurrent `claude` RAM, and racing the
# decision/equity log so a trader sees 2–3 decisions clustered inside a
# minute then nothing for an hour). Nothing in `runner.py` prevented it —
# digital-intern's daemon has a singleton lock; this is the missing twin.
#
# An `fcntl.flock` advisory lock on a lockfile next to the DB is the robust
# primitive: the kernel releases it automatically when the holder *dies*
# (crash / SIGKILL / normal exit), so a restart never trips over a stale
# PID file — the exact failure a naive pid-file guard introduces. Held for
# the life of the process via a module-global handle (closing the fd frees
# the lock, so it MUST NOT be GC'd). Fail-OPEN by construction: if the lock
# infrastructure itself is unusable (non-POSIX, unwritable data dir, USB
# unmounted) we degrade to "run without the guard" and warn, never refuse
# to start the *only* trader — same philosophy as `_save_runner_state`'s
# best-effort sidecar. Fail-CLOSED only on the one signal we can trust:
# another live process is holding the lock → exit before booting anything.
_LOCK_PATH = DB_PATH.parent / "paper_trader.runner.lock"

# (handle, status, holder_pid). status ∈ {"acquired","busy","degraded"}.
SingletonLock = namedtuple("SingletonLock", ("handle", "status", "holder_pid"))

# Process-lifetime reference to the locked file object. Module-global so the
# fd stays open (and the flock held) for as long as the runner lives.
_SINGLETON_LOCK_FH = None

# Current single-instance lock state of THIS process.
#   "acquired" — we hold the flock (the only legitimate writer)
#   "degraded" — the lock plumbing was unusable at boot (fail-open, invariant
#                #19): a guard-less runner that keeps retrying each cycle and
#                exits the moment it confirms another live trader holds the
#                lock — the exact 2026-05-17 double-trade pathology, observed
#                live again (PID 1255030 degraded + PID 1465599 locked, both
#                cycling the same $1000 book for ~12h with clustered
#                NO_DECISION rows). Exposed via singleton_lock_state() so a
#                guard-less runner is no longer invisible from every operator
#                surface (/api/runner-heartbeat + the hourly Discord summary).
_lock_status: str = "degraded"
_lock_holder_pid: int | None = None
_degraded_recheck_warned = False


def singleton_lock_state() -> dict:
    """Best-effort snapshot of THIS process's single-instance lock status.

    Pure read of module globals — never raises, safe from any thread (the
    dashboard reads it from its request thread). ``degraded`` True means this
    runner booted without the guard and may be double-trading the shared
    paper book until it either upgrades or exits (see
    ``_recheck_singleton_lock``)."""
    return {
        "status": _lock_status,
        "holder_pid": _lock_holder_pid,
        "have_lock": _lock_status == "acquired",
        "degraded": _lock_status == "degraded",
    }


def alarm_latch_state() -> dict:
    """Best-effort snapshot of this process's silent-failure alarm latches.

    The runner dedupes two Discord alarm classes via in-memory latches:

      * ``_breaker_alert_active`` — the consecutive-NO_DECISION circuit
        breaker fired; further breaker alarms stay silent until a real
        decision (HOLD/FILLED/BLOCKED) lands.
      * ``_quota_alert_active`` — Claude rejected with a usage/quota limit;
        further quota alarms stay silent until a real decision lands.

    A trader watching Discord sees the FIRED/CLEARED bracket but cannot tell
    whether the latch is CURRENTLY held without scrolling. The dashboard's
    ``/api/runner-heartbeat`` already exposes the lock and Discord-delivery
    health; this accessor surfaces the orthogonal silent-failure latch state
    next to them so the operator gets the full liveness picture in one read.

    Returned fields:

      * ``breaker_active`` — bool, ``_breaker_alert_active``
      * ``quota_active`` — bool, ``_quota_alert_active``
      * ``consecutive_no_decisions`` — running count toward the threshold
      * ``breaker_threshold`` — ``CONSECUTIVE_NO_DECISION_LIMIT`` (= 5)
      * ``breaker_outage_s`` — int seconds since the FIRST NO_DECISION in
        the current wedge run (None when no wedge in progress)
      * ``quota_outage_s`` — int seconds since the FIRST quota cycle in the
        current quota outage (None when no quota outage in progress)
      * ``any_active`` — bool, True iff either latch is held (the lean
        single-bit flag a dashboard banner can key off)

    Pure read of module globals — never raises, safe from any thread.
    Mirrors ``singleton_lock_state``'s discipline. A wall-clock step-back
    (the documented clock-skew hazard the sidecar already hardens against)
    clamps the outage durations to 0 rather than rendering negative.
    """
    now = datetime.now(timezone.utc)

    def _age_s(ts):
        if ts is None:
            return None
        try:
            return max(0, int((now - ts).total_seconds()))
        except (TypeError, AttributeError):
            return None

    breaker = bool(_breaker_alert_active)
    quota = bool(_quota_alert_active)
    return {
        "breaker_active": breaker,
        "quota_active": quota,
        "any_active": breaker or quota,
        "consecutive_no_decisions": int(_consecutive_no_decisions),
        "breaker_threshold": CONSECUTIVE_NO_DECISION_LIMIT,
        "breaker_outage_s": _age_s(_no_decision_first_ts),
        "quota_outage_s": _age_s(_quota_first_ts),
    }


def _fmt_outage_seconds(secs) -> str:
    """Compact outage-duration label: ``42m`` / ``1h32m`` / ``2d4h``.

    Mirrors ``reporter._format_elapsed``'s contract but stays in runner.py
    so the runner-side ``alarm_latch_headline`` never imports the (huge,
    frequently-edited) reporter module. ``None`` / negative / non-numeric
    → ``""`` so callers suppress the duration token entirely (degrade-safe).
    Sub-minute clamps to ``0m`` so a 30s wedge reads cleanly, not as ``""``.
    Never raises."""
    try:
        s = float(secs) if secs is not None else None
    except (TypeError, ValueError):
        return ""
    if s is None or s < 0:
        return ""
    s_i = int(s)
    if s_i < 3600:
        return f"{s_i // 60}m"
    if s_i < 86400:
        h, m = divmod(s_i, 3600)
        return f"{h}h{m // 60}m"
    d, rem = divmod(s_i, 86400)
    return f"{d}d{rem // 3600}h"


def alarm_latch_headline(state: dict | None = None) -> str:
    """One-line operator headline derived from ``alarm_latch_state()``.

    The companion to ``alarm_latch_state``: that function exposes
    individual booleans + outage counters; this one formats them into
    the single human sentence an operator banner (or a Discord status
    embed) can render verbatim. The dashboard already builds an
    ad-hoc inline headline in ``/api/alarm-latches``, but that one
    omits the wall-clock outage durations — a trader pulled away from
    Discord during a multi-hour wedge cares first about *how long*
    each latch has been held.

    Contract:
      * No latch held → ``""`` (silence-when-nothing-actionable; the
        caller renders no headline rather than a misleading "OK" line).
      * One or both latches held → ``"⚠️ CLAUDE BREAKER held (47m) ·
        QUOTA latch held (3h12m) — operator alerted; waiting for the
        next real decision to clear."`` Each held latch carries its
        own outage-duration parenthetical when the duration field is
        present and non-negative; a missing/garbage duration drops the
        parenthetical but the latch-name token still ships.

    Pure formatting over the state dict — no I/O, no store reads. When
    ``state`` is omitted, defaults to a fresh ``alarm_latch_state()``
    read. Degrade-safe: a non-dict ``state`` returns ``""`` rather than
    raising. Never raises (mirrors the rest of runner's notification-
    helper discipline)."""
    try:
        if state is None:
            state = alarm_latch_state()
        if not isinstance(state, dict):
            return ""
        breaker = bool(state.get("breaker_active"))
        quota = bool(state.get("quota_active"))
        if not (breaker or quota):
            return ""
        bits: list[str] = []
        if breaker:
            dur = _fmt_outage_seconds(state.get("breaker_outage_s"))
            bits.append(
                f"CLAUDE BREAKER held ({dur})" if dur else "CLAUDE BREAKER held"
            )
        if quota:
            dur = _fmt_outage_seconds(state.get("quota_outage_s"))
            bits.append(
                f"QUOTA latch held ({dur})" if dur else "QUOTA latch held"
            )
        return (
            "⚠️ " + " · ".join(bits)
            + " — operator alerted; waiting for the next real decision to clear."
        )
    except Exception:
        return ""


def _acquire_singleton_lock(path=_LOCK_PATH) -> SingletonLock:
    """Try to take the exclusive runner lock.

    Returns a ``SingletonLock``:
      • ``status="acquired"`` — we hold it; ``handle`` is the open locked
        file (caller must keep it alive), ``holder_pid`` is our PID.
      • ``status="busy"``     — another live process holds it; ``handle`` is
        None, ``holder_pid`` is the PID read from the lockfile (or None if
        unreadable). The caller must NOT start a second trader.
      • ``status="degraded"`` — the lock primitive is unavailable (no fcntl,
        unwritable dir, …). ``handle`` is None. The caller continues WITHOUT
        the guard (fail-open: never take down the sole runner over lock
        plumbing). Never raises.
    """
    try:
        import fcntl  # POSIX only; ImportError → degrade (fail-open)
    except Exception as e:
        print(f"[runner] singleton lock unavailable (no fcntl: {e}); "
              f"running WITHOUT the single-instance guard")
        return SingletonLock(None, "degraded", None)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # r+ if it exists else create; never truncate (a truncate would wipe
        # the holder's PID out from under a running instance).
        fh = open(path, "a+", encoding="utf-8")
    except Exception as e:
        print(f"[runner] could not open lockfile {path} ({e}); "
              f"running WITHOUT the single-instance guard")
        return SingletonLock(None, "degraded", None)
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        # Held by another live process. Best-effort read of its PID for a
        # human-actionable log line; never let a read error mask "busy".
        holder = None
        try:
            fh.seek(0)
            txt = fh.read().strip()
            holder = int(txt) if txt.isdigit() else None
        except Exception:
            holder = None
        try:
            fh.close()
        except Exception:
            pass
        return SingletonLock(None, "busy", holder)
    except Exception as e:
        # flock raised something other than the contended OSError — treat as
        # infrastructure failure and fail-open rather than wedge the runner.
        print(f"[runner] flock failed unexpectedly ({e}); "
              f"running WITHOUT the single-instance guard")
        try:
            fh.close()
        except Exception:
            pass
        return SingletonLock(None, "degraded", None)
    # Acquired. Record our PID for operator visibility (best-effort — failing
    # to write it does not relinquish the kernel-held lock).
    try:
        fh.seek(0)
        fh.truncate()
        fh.write(f"{os.getpid()}\n")
        fh.flush()
    except Exception:
        pass
    return SingletonLock(fh, "acquired", os.getpid())


def _recheck_singleton_lock(path=_LOCK_PATH) -> None:
    """Re-attempt the single-instance lock when running degraded.

    A runner that booted while the lock plumbing was unusable (e.g. the
    USB-backed ``data/`` mount was transiently unavailable) runs fail-open
    *forever* (invariant #19 — never refuse the sole trader over plumbing).
    But once the plumbing recovers, a SECOND runner can cleanly acquire the
    flock — and now two runners double-trade the same $1000 book (observed
    live 2026-05-17/18: PID 1255030 degraded + PID 1465599 locked, ~12h of
    clustered NO_DECISION rows from each killing the other's claude). This
    closes that window by retrying each cycle:

      • still ``degraded`` — plumbing *still* unusable. Keep running
        (invariant #19 preserved: we only ever exit on a CONFIRMED other
        holder, never on plumbing failure). Warn once.
      • ``acquired``       — plumbing recovered, no other trader holds it.
        Upgrade in place: keep the handle so the flock is held for life.
        We are now the legitimate singleton.
      • ``busy``           — plumbing recovered and ANOTHER live trader
        already holds the lock. We are the redundant degraded runner
        double-trading the book → exit cleanly so the locked instance is
        the sole writer.

    No-op once we hold the lock: a second ``open()``+``flock`` on the same
    file from the same process gets a *distinct* open-file description and
    is denied by our OWN lock (flock fds are independent), which would
    mis-read as ``busy`` and make the real holder exit. Only ever
    re-attempt from the degraded state. Never raises (except the
    deliberate ``SystemExit`` on a confirmed duplicate)."""
    global _SINGLETON_LOCK_FH, _lock_status, _lock_holder_pid
    global _degraded_recheck_warned
    if _lock_status != "degraded":
        return
    lk = _acquire_singleton_lock(path)
    if lk.status == "acquired":
        _SINGLETON_LOCK_FH = lk.handle  # keep the flock held for our life
        _lock_status = "acquired"
        _lock_holder_pid = lk.holder_pid
        print(f"[runner] single-instance lock RECOVERED — upgraded from "
              f"degraded to locked (pid={lk.holder_pid}); this is now the "
              f"sole guarded trader")
        return
    if lk.status == "busy":
        who = f"pid={lk.holder_pid}" if lk.holder_pid else "pid unknown"
        print(f"[runner] another paper trader now holds the single-instance "
              f"lock ({who}); THIS instance booted WITHOUT the guard "
              f"(degraded) and has been double-trading the shared paper "
              f"book — exiting so the locked instance is the only writer.")
        sys.exit(1)
    # Still degraded — lock plumbing remains unusable. Keep running (#19);
    # warn once so the operator sees it without flooding the cycle log.
    if not _degraded_recheck_warned:
        _degraded_recheck_warned = True
        print("[runner] still running WITHOUT the single-instance guard "
              "(lock plumbing still unusable); retrying every cycle")


def _load_runner_state() -> dict:
    """Best-effort read of the persisted report markers. Returns {} on a
    missing/corrupt/unreadable file — never raises (the daemon must boot
    even if the sidecar is garbage)."""
    try:
        with open(_STATE_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_runner_state() -> None:
    """Atomically persist the current markers. Atomic (tmp + os.replace) so a
    kill mid-write — the circuit breaker / systemd / SIGKILL — can never leave
    a torn JSON that then reads back as {} and re-arms the duplicate-close
    bug. Best-effort: any IO error is swallowed (a read-only data dir must not
    take down the trade loop)."""
    payload = {
        "daily_close_sent_for": _daily_close_sent_for,
        "last_hourly_iso": (_last_hourly.isoformat()
                            if _last_hourly is not None else None),
    }
    try:
        _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = f"{_STATE_PATH}.{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        os.replace(tmp, _STATE_PATH)
    except Exception as e:
        print(f"[runner] could not persist runner state: {e}")


def _restore_runner_state() -> None:
    """Rehydrate `_daily_close_sent_for` / `_last_hourly` from the sidecar at
    boot. Called by `main()` *before* the loop. No persisted marker → leave the
    global as-is (fresh-boot behaviour: the daily flag stays None; `main()`
    still anchors `_last_hourly` to boot so a first-ever start doesn't fire an
    hourly immediately). A persisted `last_hourly` that is already >1h old
    correctly lets the first cycle send the overdue summary instead of
    swallowing another hour.

    FUTURE-marker hardening: a persisted marker can never *legitimately* be in
    the future — both are written as `datetime.now()` at save time. A
    future-dated marker means the wall clock stepped BACKWARD after the save
    (an NTP correction / VM time-sync — this box has documented clock+load
    stress). Restoring it verbatim is silently trader-visible:
      • a future `_last_hourly` makes `(now - _last_hourly) < 3600` true for up
        to (skew + 1h), so `_maybe_hourly` MUTES the hourly Discord summary —
        the operator's primary monitoring surface goes dark with no signal,
        the exact "Hourly STARVATION" class this sidecar exists to prevent;
      • a `daily_close_sent_for` strictly after today (NY) suppresses THAT
        day's close once the clock reaches it (the `== today` gate then
        matches a date for which nothing was ever sent).
    So clamp a future `_last_hourly` back to now (normal 1h cadence resumes,
    never muted longer than intended) and drop a future `daily_close_sent_for`
    (treat as "not sent" — fresh-boot behaviour, never suppress a real close).
    """
    global _daily_close_sent_for, _last_hourly
    st = _load_runner_state()
    now = datetime.now(timezone.utc)
    dcs = st.get("daily_close_sent_for")
    if isinstance(dcs, str) and dcs:
        # `daily_close_sent_for` is a NY-date isoformat (set from
        # `now_ny.date().isoformat()` in `_maybe_daily_close`); ISO dates
        # compare lexically. A value strictly after today (NY) is non-physical.
        today_ny = now.astimezone(NY).date().isoformat()
        if dcs <= today_ny:
            _daily_close_sent_for = dcs
        else:
            print(f"[runner] ignoring future daily_close_sent_for={dcs!r} "
                  f"(today NY={today_ny}); clock stepped back — treating as "
                  f"not-sent so today's close is not suppressed")
    lh = st.get("last_hourly_iso")
    if isinstance(lh, str) and lh:
        try:
            dt = datetime.fromisoformat(lh)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if dt > now:
                print(f"[runner] persisted last_hourly_iso={lh!r} is in the "
                      f"future (now={now.isoformat()}); clock stepped back — "
                      f"clamping to now so the hourly summary is not muted")
                dt = now
            _last_hourly = dt
        except ValueError:
            pass


def _maybe_hourly():
    global _last_hourly
    now = datetime.now(timezone.utc)
    if _last_hourly is not None and (now - _last_hourly).total_seconds() < 3600:
        return
    try:
        # Only advance _last_hourly on success — a transient openclaw failure
        # then retries next cycle instead of silently skipping the hour.
        if reporter.send_hourly_summary():
            _last_hourly = now
            _save_runner_state()  # survive a restart — don't starve the hour
        else:
            print("[runner] hourly send returned False, will retry next cycle")
    except Exception as e:
        print(f"[runner] hourly send failed: {e}")


def _maybe_daily_close():
    global _daily_close_sent_for
    now_ny = datetime.now(timezone.utc).astimezone(NY)
    today = now_ny.date().isoformat()
    if now_ny.weekday() >= 5:
        return  # no weekend close report
    if now_ny.date() in market.NYSE_HOLIDAYS_2026:
        # A full-holiday close is not a trading day — emitting a "DAILY CLOSE"
        # with stale marks and "Trades today 0" is misleading noise. Mirror the
        # weekend guard: bail before touching _daily_close_sent_for so the flag
        # stays where it was.
        return
    # Anchor the trigger to the *actual* NYSE session close, not a hardcoded
    # 16:00. close_minute() is 16:00 ET on a regular day (gate < 16:05 ET —
    # byte-identical to the old behaviour) and 13:00 ET on a known half-day
    # (gate < 13:05 ET) so the close report lands right after the early bell
    # instead of three hours late against a frozen post-close book.
    close_min = market.close_minute(now_ny.date())
    now_min = now_ny.hour * 60 + now_ny.minute
    if now_min < close_min + DAILY_CLOSE_GRACE_MIN:
        return
    if _daily_close_sent_for == today:
        return
    try:
        # Only mark as sent on actual success — _send returns False (no exception)
        # when openclaw is missing or fails; otherwise a transient failure would
        # permanently suppress today's close.
        if reporter.send_daily_close():
            _daily_close_sent_for = today
            _save_runner_state()  # survive a post-16:05 restart — no dup close
        else:
            print("[runner] daily close: send returned False, will retry next cycle")
    except Exception as e:
        print(f"[runner] daily close failed: {e}")


# ── Git-staleness self-restart ───────────────────────────────────────────
# The trading-intelligence repo auto-commits/pushes operator fixes; systemd
# runs this unit with Restart=always. Without a self-check, a running runner
# keeps executing the OLD code until something else bounces it — a committed
# fix can sit unapplied for hours. This watcher records git HEAD at boot and
# re-checks every 3 min; on a HEAD change it logs, pings Discord, and sets
# `_restart_requested` — the MAIN LOOP performs the graceful os._exit(0) at
# the next cycle boundary so a restart never kills a mid-Opus decision call.
# systemd (Restart=always) brings us back on the new code. REPO_ROOT is the
# paper-trader dir (a child of the trading-intelligence work tree);
# `git rev-parse` walks up to the repo, so this resolves HEAD correctly.
# Fail-OPEN by construction: ANY git/subprocess error just skips that
# iteration — the watcher must never crash the trade loop.
#
# DEADMAN SAFETY-NET: the watcher no longer `return`s after requesting the
# restart. It keeps polling and, if the graceful exit is still unhonored
# `RESTART_GRACE_S` after it was requested (the main loop is wedged — see the
# RESTART_GRACE_S note), it force-exits the process itself. This is the
# observed-live failure this closes: a deferred restart that the main loop
# never reaches leaves a committed fix undeployed indefinitely. The grace
# window preserves the "never kill a healthy mid-Opus call" intent — only a
# loop wedged well past any legitimate cycle is force-killed.
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _deferred_restart_overdue(requested_monotonic: float | None,
                              now_monotonic: float,
                              grace_s: float = RESTART_GRACE_S) -> bool:
    """True when a deferred restart was requested and the main loop has NOT
    honored it within ``grace_s`` seconds (so the watcher must force-exit).

    Pure predicate over monotonic clocks (immune to the wall-clock step-back
    the Phase-1 sidecar fix hardens against). ``requested_monotonic`` is None
    until a restart has been requested → never overdue."""
    if requested_monotonic is None:
        return False
    return (now_monotonic - requested_monotonic) >= grace_s


def _git_watcher():
    """Daemon-thread body: restart the process when new commits land."""
    try:
        old_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT
        ).decode().strip()
    except Exception as e:
        # Can't establish a baseline → nothing to compare against ever.
        # Degrade to a no-op watcher rather than crash the thread.
        print(f"[runner] git-watcher disabled (no baseline HEAD: {e})")
        return
    # Let the service fully start before the first check.
    time.sleep(120)
    while True:
        try:
            new_head = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT
            ).decode().strip()
        except Exception:
            # git transiently unavailable (mid-rebase, USB unmounted, …):
            # skip this iteration, never crash the watcher.
            time.sleep(180)
            continue
        if new_head != old_head:
            print(
                f"[runner] WARNING: git HEAD changed "
                f"(old={old_head[:7]}, new={new_head[:7]}) — new commits "
                f"detected; scheduling deferred restart after current cycle"
            )
            try:
                reporter._send(
                    f"paper-trader restart pending: new commits detected "
                    f"(old={old_head[:7]}, new={new_head[:7]}); "
                    f"will restart between cycles"
                )
            except Exception as e:
                print(f"[runner] git-watcher Discord notice failed: {e}")
            _restart_requested.set()
            # Deadman: the main loop is the graceful actor, but if it is
            # wedged it may never reach the boundary. Keep watching; if the
            # restart is still unhonored RESTART_GRACE_S later, the loop is
            # genuinely stuck — force the exit ourselves so the committed fix
            # actually deploys (systemd Restart=always reboots on new code).
            requested_at = time.monotonic()
            while True:
                time.sleep(180)
                if _deferred_restart_overdue(requested_at, time.monotonic()):
                    print(
                        f"[runner] WARNING: deferred restart still unhonored "
                        f">{int(RESTART_GRACE_S)}s after request — main loop "
                        f"appears wedged; force-exiting from the git-watcher "
                        f"so the committed fix deploys"
                    )
                    try:
                        reporter._send(
                            "paper-trader FORCE restart: deferred restart was "
                            f"not honored within {int(RESTART_GRACE_S)}s "
                            f"(main loop wedged under load); restarting now to "
                            f"apply new commits (old={old_head[:7]}, "
                            f"new={new_head[:7]})"
                        )
                    except Exception as e:
                        print(f"[runner] git-watcher force-exit notice "
                              f"failed: {e}")
                    os._exit(0)
                # else: main loop honored it (process already gone) or the
                # grace window has not elapsed yet — keep waiting.
        time.sleep(180)


def _start_git_watcher():
    try:
        threading.Thread(
            target=_git_watcher, daemon=True, name="git-watcher"
        ).start()
        print("[runner] git-watcher thread started (auto-restart on new commits)")
    except Exception as e:
        print(f"[runner] git-watcher disabled: {e}")


def _start_dashboard():
    try:
        from . import dashboard
        threading.Thread(target=dashboard.run, daemon=True, name="dashboard").start()
        print("[runner] dashboard thread started on :8090")
    except Exception as e:
        print(f"[runner] dashboard disabled: {e}")


def _kill_stale_claude():
    """Kill the runner's *own* lingering claude subprocess. strategy._claude_call()
    launches `claude --model <model> --print ...` as a direct child of this
    process; a wedged child that survives its Python-side timeout keeps holding
    resources and can re-starve the next cycle. Match the live (Opus) model
    first, then the Sonnet fallback model so a wedged fallback zombie is also
    reaped.

    Both patterns are anchored on `claude --model <family>` because the CLI
    is always invoked as `claude --model <model> --print …` — the `--model`
    arg sits between `claude` and `--print`, so a bare `claude --print`
    pattern is never a contiguous substring of the real command line and
    would silently match nothing (the exact bug this once had: a wedged
    Sonnet fallback survived the breaker, defeating auto-recovery in the
    very Opus-timeout→Sonnet-fallback path the breaker exists for).

    SCOPED TO OUR OWN CHILDREN (`pkill -P os.getpid()`). A bare host-wide
    `pkill -f "claude --model claude-opus"` is catastrophic collateral
    damage: this box also runs the hourly self-review agents
    (`scripts/hourly_review.sh` spawns 3× `claude --model claude-opus-4-7`),
    sibling automated-review agents, and possibly an operator's interactive
    `claude` session — all of which match the same pattern. A wedged trader
    recovering by SIGTERM-ing every Claude process on the machine (including
    the review agents that keep the system healthy, and the agent that may
    have just deployed a fix) is a cure far worse than the disease. The
    decision subprocess is always a *direct* child of the runner, so `-P`
    restricts the sweep to exactly the processes this breaker is meant to
    reap and nothing else."""
    own_pid = os.getpid()
    for pattern in ("claude --model claude-opus", "claude --model claude-sonnet"):
        try:
            killed = subprocess.run(
                ["pkill", "-P", str(own_pid), "-f", pattern],
                capture_output=True,
            )
            # pkill rc: 0 = killed something, 1 = nothing matched, >1 = error.
            print(
                f"[runner] circuit breaker: pkill -P {own_pid} -f "
                f"{pattern!r} returned {killed.returncode}"
            )
        except Exception as e:
            print(f"[runner] circuit breaker: pkill {pattern!r} failed: {e}")


def _open_position_tickers_for_interval(store) -> list[dict]:
    """``[{ticker: T}, ...]`` for ``compute_interval`` from the singleton store.

    Previously the main loop opened its OWN raw sqlite connection (no WAL pragma,
    no busy_timeout, missing the schema-true ``qty > 0`` filter that
    ``store.open_positions()`` always applies) to feed ``dynamic_interval``.
    Two real problems with that:

      * a duplicated reader connection competes with the singleton store for
        the WAL lock and is not subject to its 30s ``busy_timeout`` — under
        contention it raised ``database is locked`` and the ``except: pass``
        below silently degraded the dynamic-interval calc to "0 positions",
        making a real held earnings day read as MARKET_CLOSED.
      * a closed lot whose ``closed_at`` had not yet been written (a torn
        ``upsert_position`` mid-crash; rare but possible) would still satisfy
        ``WHERE closed_at IS NULL`` even though ``qty=0`` — silently inflating
        the held set with a row that has no actual exposure.

    Using ``store.open_positions()`` fixes both: same lock, same ``qty > 0``
    filter, same WAL/busy_timeout discipline. Any store fault degrades to
    ``[]`` (the dynamic-interval calc's documented "fall back to MARKET_OPEN
    or MARKET_CLOSED" path), never an exception — invariant: this helper
    never raises, mirroring the original inline ``try/except``.
    """
    try:
        return [
            {"ticker": p.get("ticker")}
            for p in store.open_positions()
            if p.get("ticker")
        ]
    except Exception as e:
        print(f"[runner] open-position read for interval failed: {e}")
        return []


def _no_decision_cause(summary: dict) -> str:
    """Best-effort extraction of the cause-code suffix written into the
    most recent NO_DECISION row by ``strategy.decide()`` for the breaker
    alert body. Pure dict lookup over the in-hand summary — no store read
    (the alert path must stay fast and degrade-safe; mirrors the rest of
    runner's "never raise from a notification helper" discipline).

    The decision schema buries the cause four ways:
      * ``summary["host_saturated"]`` True → "skipped — host saturated"
      * ``summary["quota_exhausted"]`` True → "quota/usage limit"
      * ``summary["raw"]`` is non-empty → parse failure, surface the first line
      * ``summary["last_claude_fail"]`` set (raw is None) → empty/timeout/CLI
        cause from strategy._claude_call's per-call tag (timeout / nonzero_rc /
        empty_stdout / cli_missing / exception) — distinct from a parse miss.

    Returns ``""`` only when NONE of the four channels carry a cause (a
    historical row from before this field landed, or a truly opaque miss)."""
    if summary.get("quota_exhausted"):
        return "quota/usage limit"
    if summary.get("host_saturated"):
        return "host saturated (skipped claude)"
    # Whitespace-only raw (rare but observed live when the CLI streams a
    # blank line before disconnecting) carries no diagnostic value — strip
    # first and only take the raw-response branch when something prose-y
    # remains. Otherwise this returned "raw response (first line): " with an
    # empty body, masking the per-call ``last_claude_fail`` tag the breaker
    # alert was specifically designed to surface (timeout / empty_stdout /
    # nonzero_rc / cli_missing / exception).
    raw = summary.get("raw")
    if isinstance(raw, str):
        stripped = raw.strip()
        if stripped:
            head = stripped.split("\n", 1)[0]
            return f"raw response (first line): {head[:120]}"
    fail = summary.get("last_claude_fail")
    if isinstance(fail, str) and fail:
        return f"claude no-response ({fail})"
    return ""


def _cycle():
    global _consecutive_no_decisions, _quota_alert_active
    global _breaker_alert_active, _no_decision_first_ts
    global _quota_first_ts
    summary = strategy.decide()

    status = summary.get("status", "NO_DECISION")
    quota = bool(summary.get("quota_exhausted"))

    if quota:
        # Quota / usage-limit exhaustion is a DISTINCT failure from a wedged
        # CLI: the claude process already exited (non-zero, fast), so the
        # circuit-breaker pkill is futile — running it just spams the log and
        # could reap an unrelated sibling. The actionable response is to
        # alarm the operator ONCE (the bot is silently frozen — no trades
        # will execute until the quota resets). Keep the breaker counter at 0
        # so a quota outage can never trip it.
        _consecutive_no_decisions = 0
        # Quota path never reaches the breaker, but the wedge-elapsed marker
        # must still re-arm so a NON-quota wedge that follows starts its own
        # clock from THIS cycle (not from a stale quota-era ts).
        _no_decision_first_ts = None
        # Anchor the quota outage clock at the FIRST quota cycle so the
        # recovery message can render the real elapsed duration. The breaker
        # path uses ``_no_decision_first_ts`` for this; quota has its own
        # marker because the quota arm just reset ``_no_decision_first_ts``
        # to None one line up.
        if _quota_first_ts is None:
            _quota_first_ts = datetime.now(timezone.utc)
        if not _quota_alert_active:
            try:
                detail = ""
                d = summary.get("decision")
                if not d:
                    detail = "Opus + Sonnet fallback both rejected with a usage/quota limit."
                # Pass the store so the alert body carries a one-line
                # snapshot of what the unattended book holds — the
                # operator paged on a quota outage needs to know what is
                # frozen (cash %, biggest position, unrealized P/L), not
                # just THAT it is frozen. Store fetch is best-effort:
                # any fault inside _book_exposure_line degrades to the
                # legacy no-context body (helper returns ``""``).
                if reporter.send_quota_alert(detail, store=get_store()):
                    _quota_alert_active = True  # dedupe until recovery
            except Exception as e:
                print(f"[runner] quota alert failed: {e}")
    else:
        # Auto-recovery circuit breaker. strategy.decide() returns a dict whose
        # top-level "status" is "NO_DECISION" whenever Claude failed (timeout /
        # empty / unparseable, after the Sonnet fallback and JSON-only retry).
        # A genuine HOLD comes back as status="HOLD", not "NO_DECISION", so a
        # quiet market does NOT trip the breaker — only repeated Claude failures.
        if status == "NO_DECISION":
            _consecutive_no_decisions += 1
            if _no_decision_first_ts is None:
                _no_decision_first_ts = datetime.now(timezone.utc)
            if _consecutive_no_decisions >= CONSECUTIVE_NO_DECISION_LIMIT:
                fired_at = _consecutive_no_decisions
                # Real elapsed-time since the first NO_DECISION in this run.
                # Falls back to None (the alert renders without the elapsed
                # token) when, somehow, the first-ts was not captured.
                elapsed_s: int | None
                if _no_decision_first_ts is not None:
                    elapsed_s = int(
                        (datetime.now(timezone.utc)
                         - _no_decision_first_ts).total_seconds()
                    )
                else:
                    elapsed_s = None
                print(
                    f"[runner] WARNING: {fired_at} consecutive "
                    f"NO_DECISION cycles — Claude appears wedged; killing stale "
                    f"claude processes to auto-recover"
                )
                _kill_stale_claude()
                _consecutive_no_decisions = 0  # reset after intervention
                # The breaker historically fired silently — only operator
                # signal was a stdout WARNING line nobody tails. Surface it
                # to Discord (the documented "primary surface"). Dedupe
                # latch matches the quota_alert pattern: one alert per
                # wedge, cleared on the next real decision. A latched
                # second breaker fire inside the same outage stays silent
                # so a long wedge doesn't flood the channel.
                if not _breaker_alert_active:
                    cause = _no_decision_cause(summary)
                    try:
                        # Pass the store so the alert body carries a
                        # one-line snapshot of what the unattended book
                        # holds (cash %, biggest position, unrealized
                        # P/L). The operator paged about a wedge needs
                        # to know what is frozen, not just THAT it is.
                        # Store fetch is best-effort: any fault inside
                        # _book_exposure_line degrades to the legacy
                        # no-context body (helper returns ``""``).
                        if reporter.send_breaker_fired_alert(
                            fired_at, cause, elapsed_s=elapsed_s,
                            store=get_store(),
                        ):
                            _breaker_alert_active = True
                    except Exception as e:
                        print(f"[runner] breaker alert failed: {e}")
        else:
            # Capture the wedge duration BEFORE clearing _no_decision_first_ts
            # — the breaker-cleared Discord notice needs to know how long the
            # bot was actually dark, and the ts is the only source of that
            # wall-clock figure (consecutive_no_decisions has already been
            # zeroed by the breaker-fire path under a sustained storm).
            recovery_elapsed_s: int | None = None
            if _breaker_alert_active and _no_decision_first_ts is not None:
                recovery_elapsed_s = int(
                    (datetime.now(timezone.utc)
                     - _no_decision_first_ts).total_seconds()
                )
            _consecutive_no_decisions = 0
            # Engine produced a real decision (HOLD, FILLED, BLOCKED) — clear
            # the breaker latch so a fresh wedge tomorrow re-alerts the
            # operator. Symmetric with the quota latch's recovery path
            # (re-arm only on confirmed responsiveness, never on an in-
            # flight retry). The ``_no_decision_first_ts`` anchor is held
            # armed until the send confirms — otherwise a transient
            # openclaw failure here loses the duration token forever (the
            # next cycle's retry would render the bare "responding again"
            # body with no "after ~Xh dark" close on the FIRED→CLEARED
            # bracket). When NO latch is held (sub-threshold wedge that
            # never alarmed), reset the anchor unconditionally so the next
            # outage starts its own clock fresh.
            if _breaker_alert_active:
                ok = False
                try:
                    ok = bool(reporter.send_breaker_cleared_alert(
                        elapsed_s=recovery_elapsed_s
                    ))
                except Exception as e:
                    print(f"[runner] breaker recovery notice failed: {e}")
                if ok:
                    _breaker_alert_active = False
                    _no_decision_first_ts = None  # re-arm only on confirmed send
            else:
                _no_decision_first_ts = None  # no latch — anchor not needed for retry
            # Defensive cleanup of ``_quota_first_ts``. The quota arm above
            # anchors ``_quota_first_ts`` on the FIRST quota cycle of an
            # outage, but ``_quota_alert_active`` only flips True when
            # ``send_quota_alert`` actually delivers (the
            # never-spam-without-delivery contract on line 879). A transient
            # openclaw failure on the very first quota cycle leaves the
            # latch False AND ``_quota_first_ts`` set — the recovery branch
            # below skips (its `_quota_alert_active` predicate is False), so
            # the anchor would stay stuck forever and a FUTURE quota
            # outage's recovery alert would render a wildly inflated
            # elapsed_s (the gap between BOTH outages, not the new one).
            # Symmetric to the breaker-anchor reset on line 975: when no
            # latch is held, the anchor carries no surviving payload — drop
            # it so the next outage starts a fresh clock.
            if not _quota_alert_active and _quota_first_ts is not None:
                _quota_first_ts = None
        # Quota recovered → tell the operator once, then re-arm the alarm so a
        # *future* outage alerts again. Only confirm on an actual claude
        # response (status != NO_DECISION); a non-quota timeout is not proof
        # the quota is back, so we hold the alarmed state until a real
        # decision lands rather than crying "recovered" prematurely.
        #
        # Mirror the alarm-path symmetry: only clear ``_quota_alert_active``
        # when the recovery notice was actually delivered. Without this, a
        # transient openclaw / Discord outage at recovery time silently drops
        # the operator-facing "we're back" message AND clears the latch — so
        # the operator (who only sees Discord) is stuck believing the trader
        # is still frozen until the NEXT quota outage re-alarms. Clearing the
        # latch only on successful send makes the next decide() cycle retry
        # the recovery notice; the latch itself dedupes so we never spam.
        if _quota_alert_active and status != "NO_DECISION":
            # Capture elapsed BEFORE clearing the ts marker — same discipline
            # as the breaker recovery path one block up. The send may fail
            # transiently (openclaw down) so the latch stays armed; the ts is
            # also kept armed in that case so a successful retry next cycle
            # still ships the real wedge duration. Only on confirmed send
            # success do we clear both the latch AND the ts (a fresh quota
            # outage that follows starts its own clock).
            quota_elapsed_s: int | None = None
            if _quota_first_ts is not None:
                quota_elapsed_s = int(
                    (datetime.now(timezone.utc)
                     - _quota_first_ts).total_seconds()
                )
            ok = False
            try:
                ok = bool(reporter.send_quota_recovered_alert(
                    elapsed_s=quota_elapsed_s,
                ))
            except Exception as e:
                print(f"[runner] quota recovery notice failed: {e}")
            if ok:
                _quota_alert_active = False
                _quota_first_ts = None  # re-arm: next outage starts fresh

    try:
        if summary.get("auto_exits") or summary.get("status") == "FILLED":
            # post the trade that was just executed
            store_ = get_store()
            trades = store_.recent_trades(1)
            if trades and summary.get("status") == "FILLED":
                # Pass the POST-trade snapshot (strategy.decide() re-marks at
                # the end of the cycle) + the store so the alert can append
                # a one-liner with the trade's immediate book impact (lot
                # weight, realized P/L, hold time, cash). Backwards-
                # compatible — send_trade_alert degrades to the old body
                # when either kwarg is missing.
                reporter.send_trade_alert(
                    trades[0],
                    snapshot=summary.get("snapshot"),
                    store=store_,
                )
            for ax in summary.get("auto_exits") or []:
                reporter._send(f"**AUTO RISK EXIT** `{ax}`")
        if summary.get("status") == "FILLED":
            reporter.send_decision_log(summary)
    except Exception as e:
        print(f"[runner] report failed: {e}")


def main():
    global _last_hourly, _SINGLETON_LOCK_FH, _lock_status, _lock_holder_pid
    print("[runner] starting paper trader")
    # Single-instance guard FIRST — before the store, the dashboard thread,
    # or the ONLINE ping. A second runner must not even mark-to-market the
    # shared book, let alone trade it. "busy" is the only fail-closed path;
    # "degraded" (lock plumbing unusable) continues so the sole runner is
    # never taken down by lock infrastructure.
    _lock = _acquire_singleton_lock()
    if _lock.status == "busy":
        who = f"pid={_lock.holder_pid}" if _lock.holder_pid else "pid unknown"
        print(f"[runner] another paper trader is already running ({who}); "
              f"refusing to start a second trader on the same paper book — "
              f"exiting. (kill the duplicate, or stop this launcher.)")
        sys.exit(1)
    _lock_status = _lock.status
    _lock_holder_pid = _lock.holder_pid
    if _lock.status == "acquired":
        _SINGLETON_LOCK_FH = _lock.handle  # keep the flock held for our life
        print(f"[runner] single-instance lock acquired (pid={_lock.holder_pid})")
    else:  # degraded — fail-open (#19); the loop re-attempts every cycle.
        print("[runner] WARNING: started WITHOUT the single-instance guard "
              "(lock plumbing unusable). Running fail-open (invariant #19); "
              "will retry the lock each cycle and exit if another live "
              "trader acquires it.")
    store = get_store()
    pf = store.get_portfolio()
    print(f"[runner] portfolio: cash=${pf['cash']:.2f} total=${pf['total_value']:.2f}")
    # Anchor the hourly clock to boot so a *first-ever* start's first summary
    # lands ~1h in, rather than firing immediately alongside the online ping.
    _last_hourly = datetime.now(timezone.utc)
    # Then rehydrate from the sidecar: on a *restart* this restores the real
    # last-hourly instant (an overdue summary fires this cycle instead of the
    # boot-anchor swallowing yet another hour) and the daily-close-sent date
    # (a post-16:05 bounce won't re-post the close). No sidecar → globals keep
    # their fresh-boot values, behaviour identical to before this change.
    _restore_runner_state()
    _start_git_watcher()
    _start_dashboard()
    try:
        reporter._send("**PAPER TRADER ONLINE** ◈ engine booted, decision loop starting")
    except Exception:
        pass

    while True:
        # If we booted degraded (no guard), keep trying to acquire the lock.
        # Exits the process if another live trader has since acquired it
        # (confirmed double-trade) — outside the try so the deliberate
        # SystemExit is never swallowed by `except Exception`.
        _recheck_singleton_lock()
        try:
            _cycle()
        except Exception:
            print("[runner] cycle exception:")
            traceback.print_exc()

        _maybe_hourly()
        _maybe_daily_close()

        # Deferred restart: exit between cycles so we never kill a mid-Opus call.
        if _restart_requested.is_set():
            print("[runner] deferred restart — exiting between cycles (new commits detected)")
            try:
                reporter._send("paper-trader restarting now: applying new commits")
            except Exception:
                pass
            os._exit(0)

        market_open = market.is_market_open()
        current_positions = _open_position_tickers_for_interval(store)
        sleep_s = compute_interval(current_positions)
        print(f"[runner] sleeping {sleep_s}s (market_open={market_open})")
        _restart_requested.wait(timeout=sleep_s)
        if _restart_requested.is_set():
            print("[runner] deferred restart triggered during sleep")
            try:
                reporter._send("paper-trader restarting now: applying new commits")
            except Exception:
                pass
            os._exit(0)


if __name__ == "__main__":
    main()
