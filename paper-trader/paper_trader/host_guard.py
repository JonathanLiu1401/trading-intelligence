"""Host-saturation primitive — the one diagnostic that still works when the
dashboard can't answer.

Root cause of the recurring live-trader `NO_DECISION` storms is *not* a prompt
or parser bug: it is host saturation. When `scripts/hourly_review.sh` fires its
parallel Opus review agents and the continuous backtest loop adds more, the box
(15 GB RAM) runs 10+ concurrent `claude --model claude-opus-4-7` subprocesses,
swap exhausts, load climbs past 15, and the live trader's own Opus call is
OOM-killed / starved → empty output → `"claude returned no response
(timeout/empty)"`. Observed 2026-05-18: 41/51 of the day's decisions, zero
trades.

`/api/empty-claude-rate` already correlates the empty rate with the
concurrent-Opus count — but that endpoint runs *inside the runner process*,
which is exactly the process being starved during a storm. When you most need
the answer ("is the box saturated right now?") the dashboard is the least able
to give it. Nothing in the repo answers that question from a plain shell with
zero third-party dependencies.

This module fills that gap with two distinct, deliberately small surfaces:

  1. `host_saturated()` — a pure, dependency-free, degrade-safe predicate
     `(bool, reason)`. This is the exact shape `strategy.py` would import for a
     pre-flight guard so it can *skip* a doomed 180 s Opus call during a storm
     instead of burning a decision cycle and adding 1.5 GB to the thrash.
     **It is intentionally NOT wired into `strategy.py` yet** — that file is a
     hot-contention path and the wire-in belongs to a calm session, not a
     storm. Shipped standalone + tested so the wire-in is a one-liner later.

  2. A `python3 -m paper_trader.host_guard` CLI that prints a saturation
     verdict and (read-only) the live trader's recent empty-rate, and exits
     non-zero when saturated — usable in shell guards and runnable when the
     Flask dashboard is unreachable.

Every public function is degrade-safe: it returns a conservative default and
never raises, mirroring the "non-fatal by construction" contract used across
the live decision path and the dashboard endpoints.
"""
from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

# Default decision DB — same location signals.py / store.py use.
_DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "paper_trader.db"

# Substring that identifies a live Opus decision/review subprocess. The live
# trader, the backtest committee, the hourly-review agents and the HYBRID
# agents all spawn `claude --model claude-opus-4-7 ...`.
_OPUS_MARKER = "claude --model claude-opus"

# ── thresholds ───────────────────────────────────────────────────────────────
# Why these numbers: backtest.py caps its own pool at 3 (`_CLAUDE_SEM`), the
# live trader needs exactly one free slot, so >4 concurrent Opus means an
# out-of-band review/HYBRID storm is contending for the live trader's call.
# Each claude proc is ~1.5 GB; <400 MB available or a near-full swap means the
# next spawn gets OOM-killed. load1/cpu > 4 is sustained over-subscription.
DEFAULT_MAX_OPUS = 4
DEFAULT_MIN_MEM_AVAIL_MB = 400
DEFAULT_MAX_SWAP_USED_PCT = 90.0
DEFAULT_LOAD_PER_CPU = 4.0

# `pulse()` state floor: fraction of recent decisions that must be a
# starvation NO_DECISION (timeout/empty *or* deliberately-skipped) for the
# verdict to read STARVED while the live /proc probe itself reads clear (an
# intermittent storm — the box has calmed but the decision log still carries
# the damage). Healthy is ~0.0; 25% is unambiguously abnormal. A module
# constant so tests read it instead of hardcoding the window (the
# digital-intern "tests read live module constants" discipline).
STARVATION_RATE_FLOOR = 0.25

# The two reasoning prefixes that mean "this cycle never produced an Opus
# decision because the host was overloaded". They are DISTINCT and counting
# only the first silently under-reports the live pathology:
#   * the OLD prefix is written by strategy.decide() when the Opus call ran
#     but returned empty / timed out (the box saturated mid-call);
#   * the NEW prefix is written when the pre-flight / mid-call host-saturation
#     guard *declined* the call (paper_trader/host_guard.host_saturated). Once
#     that guard is live a storm produces FEWER "no response" rows and MORE
#     "skipped" rows — so `recent_empty_rate` (old prefix only) collapses
#     toward zero precisely when saturation is worst. `pulse()` must key off
#     BOTH (see /api/host-guard's docstring on the same trap).
_STARVATION_PREFIXES = ("claude returned no response", "skipped claude call")


# Cause buckets for `recent_starvation_by_cause`. The cell the operator
# needs is finer than "starved vs not": each class needs a DIFFERENT action.
#
#   model_timeout  — Opus subprocess hit the 180s wall on the wire. May
#                    resolve when load drops; no operator action required
#                    unless persistent.
#   cli_nonzero_rc — `claude` CLI exited non-zero with NO quota marker (a
#                    transient API error or CLI crash). Distinct from a
#                    deliberate quota rejection; check the Anthropic
#                    status page if persistent.
#   model_empty    — CLI exited 0 but stdout was empty, OR Popen raised.
#                    Model-level miss; usually a one-cycle blip.
#   cli_missing    — `claude` not in PATH at call time. Real config bug;
#                    the live trader can never decide until fixed.
#   host_skip      — pre-flight or mid-call `host_saturated` guard
#                    DECLINED the call. The operator must reduce parallel
#                    Opus jobs (review/backtest agents) — trading cannot
#                    resolve this (see `_OPS_ACTION`).
#   unknown        — starved row whose reason text doesn't match any of
#                    the above (a legacy "timeout/empty" generic line
#                    written before strategy.py's per-cause sub-buckets,
#                    or future code paths). Surfaced so the breakdown
#                    sums back to `starved` exactly.
_CAUSE_LABELS = (
    "model_timeout", "cli_nonzero_rc", "model_empty",
    "cli_missing", "host_skip", "unknown",
)


def _classify_starvation_cause(reason: str) -> str:
    """Pick a single cause bucket for a starved NO_DECISION reason string.
    Pure / never raises. Returns one of ``_CAUSE_LABELS``.

    The host_skip prefix is checked FIRST because "skipped claude call —
    host saturated mid-call" may parenthesise the underlying reason in
    a future format (today it doesn't, but the dispatch order pins the
    intent so a future suffix can't be mis-classified as a model failure).
    """
    r = reason or ""
    if r.startswith("skipped claude call"):
        return "host_skip"
    if not r.startswith("claude returned no response"):
        return "unknown"
    # Parenthesised sub-bucket. strategy.py:1791 writes the cause from
    # `_last_claude_fail`: timeout / nonzero_rc / empty_stdout / cli_missing
    # / exception. The legacy generic "(timeout/empty)" rolls into
    # model_timeout (the dominant historical signature pre-sub-buckets).
    import re as _re
    m = _re.search(r"\(([^)]+)\)", r)
    suffix = (m.group(1).strip() if m else "").lower()
    if suffix in ("timeout", "timeout/empty"):
        return "model_timeout"
    if suffix == "nonzero_rc":
        return "cli_nonzero_rc"
    if suffix in ("empty_stdout", "exception"):
        return "model_empty"
    if suffix == "cli_missing":
        return "cli_missing"
    return "unknown"

# `pulse()` is inert under pytest unless a dedicated test opts in — the
# `dashboard._swr_active()` / runner `_STATE_PATH` offline-invariant precedent.
# The real path reads live /proc + the live WAL `paper_trader.db`; if it ran
# during the broad reporter integration tests it would (a) inject a
# host-state-dependent line into asserted Discord bodies and (b) break the
# "all tests offline & deterministic" invariant. Dedicated tests pass the
# `_snapshot`/`_starv` injectors (deterministic by construction — not gated)
# or flip `_PULSE_TEST_FORCE`; everything else gets a clean CLEAR.
_PULSE_TEST_FORCE = False  # dedicated pulse tests flip this; never in prod


def _read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except Exception:
        return ""


def concurrent_opus_count(marker: str = _OPUS_MARKER) -> int:
    """Number of live `claude --model claude-opus...` processes, excluding this
    interpreter. Walks /proc directly (no psutil) and degrades to 0."""
    me = os.getpid()
    n = 0
    try:
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            pid = int(entry)
            if pid == me:
                continue
            cmd = _read_text(f"/proc/{pid}/cmdline").replace("\x00", " ")
            if marker in cmd:
                n += 1
    except Exception:
        return 0
    return n


def _parse_meminfo(text: str) -> dict:
    """Pure parser for /proc/meminfo content → MB figures + swap-used %.

    Separated out so the threshold logic can be unit-tested deterministically
    without a real host."""
    vals: dict[str, float] = {}
    for line in text.splitlines():
        parts = line.split(":")
        if len(parts) != 2:
            continue
        key = parts[0].strip()
        num = parts[1].strip().split()
        if not num:
            continue
        try:
            vals[key] = float(num[0]) / 1024.0  # kB → MB
        except ValueError:
            continue
    swap_total = vals.get("SwapTotal", 0.0)
    swap_free = vals.get("SwapFree", 0.0)
    swap_used_pct = (
        round((swap_total - swap_free) / swap_total * 100.0, 1)
        if swap_total > 0
        else 0.0
    )
    return {
        "mem_available_mb": round(vals.get("MemAvailable", 0.0), 1),
        "swap_total_mb": round(swap_total, 1),
        "swap_free_mb": round(swap_free, 1),
        "swap_used_pct": swap_used_pct,
    }


def mem_pressure() -> dict:
    """Memory/swap snapshot from /proc/meminfo. Degrades to zeros (which read
    as 'no pressure' so a probe failure never spuriously trips the guard)."""
    info = _parse_meminfo(_read_text("/proc/meminfo"))
    if not info:
        return {
            "mem_available_mb": 0.0,
            "swap_total_mb": 0.0,
            "swap_free_mb": 0.0,
            "swap_used_pct": 0.0,
        }
    return info


def load_avg() -> tuple[float, float, float]:
    """(1, 5, 15) min load average; degrades to (0, 0, 0)."""
    try:
        a, b, c = os.getloadavg()
        return round(a, 2), round(b, 2), round(c, 2)
    except Exception:
        pass
    parts = _read_text("/proc/loadavg").split()
    try:
        return float(parts[0]), float(parts[1]), float(parts[2])
    except Exception:
        return 0.0, 0.0, 0.0


def cpu_count() -> int:
    return os.cpu_count() or 1


def probe() -> dict:
    """Collect the raw host signals the predicate needs, all degrade-safe."""
    mem = mem_pressure()
    la1, la5, la15 = load_avg()
    cpus = cpu_count()
    return {
        "opus_count": concurrent_opus_count(),
        "mem_available_mb": mem["mem_available_mb"],
        "swap_used_pct": mem["swap_used_pct"],
        "load1": la1,
        "load5": la5,
        "load15": la15,
        "cpus": cpus,
        "load_per_cpu": round(la1 / cpus, 2) if cpus else la1,
    }


def host_saturated(
    *,
    max_opus: int = DEFAULT_MAX_OPUS,
    min_mem_avail_mb: float = DEFAULT_MIN_MEM_AVAIL_MB,
    max_swap_used_pct: float = DEFAULT_MAX_SWAP_USED_PCT,
    load_per_cpu: float = DEFAULT_LOAD_PER_CPU,
    _probe: dict | None = None,
) -> tuple[bool, str]:
    """Pure saturation predicate → ``(saturated, human_reason)``.

    Saturated when ANY signal trips. ``mem_available_mb == 0`` is treated as
    "unknown" (probe failure) and ignored, so a /proc read error degrades to
    "not saturated" rather than wedging the live trader. ``_probe`` lets tests
    inject a deterministic reading instead of touching the real host.
    """
    p = _probe if _probe is not None else probe()
    opus = int(p.get("opus_count", 0) or 0)
    mem_av = float(p.get("mem_available_mb", 0.0) or 0.0)
    swap_pct = float(p.get("swap_used_pct", 0.0) or 0.0)
    lpc = float(p.get("load_per_cpu", 0.0) or 0.0)

    reasons: list[str] = []
    if opus > max_opus:
        reasons.append(f"{opus} concurrent Opus (>{max_opus})")
    if 0.0 < mem_av < min_mem_avail_mb:
        reasons.append(f"{mem_av:.0f}MB avail (<{min_mem_avail_mb:.0f})")
    if swap_pct > max_swap_used_pct:
        reasons.append(f"swap {swap_pct:.0f}% (>{max_swap_used_pct:.0f}%)")
    if lpc > load_per_cpu:
        reasons.append(f"load/cpu {lpc:.1f} (>{load_per_cpu:.1f})")

    if reasons:
        return True, "host saturated: " + ", ".join(reasons)
    return False, "host clear"


def recent_empty_rate(db_path: str | os.PathLike | None = None,
                       limit: int = 120) -> dict:
    """Read-only: fraction of the last ``limit`` live decisions that were the
    empty/timeout NO_DECISION signature. Correlates the live predicate above
    with the actual observed damage. Degrade-safe → ``ok=False`` on any error
    (missing DB, locked, schema drift) so the CLI still prints a verdict."""
    path = str(db_path) if db_path is not None else str(_DEFAULT_DB)
    out = {"n": 0, "empty": 0, "rate": 0.0, "ok": False}
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
        try:
            rows = conn.execute(
                "SELECT action_taken, reasoning FROM decisions "
                "ORDER BY id DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        finally:
            conn.close()
    except Exception:
        return out
    n = len(rows)
    empty = sum(
        1
        for a, r in rows
        if (a or "") == "NO_DECISION"
        and (r or "").startswith("claude returned no response")
    )
    out.update(n=n, empty=empty,
               rate=round(empty / n, 3) if n else 0.0, ok=True)
    return out


def recent_starvation_rate(db_path: str | os.PathLike | None = None,
                            limit: int = 120) -> dict:
    """Read-only: fraction of the last ``limit`` live decisions that were a
    host-starvation NO_DECISION — counting **both** the timeout/empty signature
    AND the new deliberately-skipped signature (see ``_STARVATION_PREFIXES``).

    This is the figure ``pulse()`` keys its STARVED verdict off. It deliberately
    supersedes ``recent_empty_rate`` (old prefix only) because once the
    pre-flight guard is live the empty-only rate falls toward zero exactly when
    the box is most saturated — the skip bucket merely absorbs it. Same
    degrade-safe contract: ``ok=False`` on any error so a caller still gets a
    verdict (the ``recent_empty_rate`` precedent)."""
    path = str(db_path) if db_path is not None else str(_DEFAULT_DB)
    out = {"n": 0, "starved": 0, "rate": 0.0, "ok": False}
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
        try:
            rows = conn.execute(
                "SELECT action_taken, reasoning FROM decisions "
                "ORDER BY id DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        finally:
            conn.close()
    except Exception:
        return out
    n = len(rows)
    starved = sum(
        1
        for a, r in rows
        if (a or "") == "NO_DECISION"
        and (r or "").startswith(_STARVATION_PREFIXES)
    )
    out.update(n=n, starved=starved,
               rate=round(starved / n, 3) if n else 0.0, ok=True)
    return out


def recent_starvation_by_cause(db_path: str | os.PathLike | None = None,
                                limit: int = 120) -> dict:
    """Read-only: bucket the last ``limit`` decisions' starvation cells by
    operator-actionable cause (``_CAUSE_LABELS``).

    The aggregate ``recent_starvation_rate`` answers "how many cycles never
    reached Opus?". This answers the follow-up the operator needs DURING a
    storm: "what proportion of those were model timeouts (Opus wedged) vs
    pre-flight skips (out-of-band Opus contention) vs CLI errors (Anthropic-
    side)?" — three classes that need three different actions (see the
    `_CAUSE_LABELS` docstring on each).

    Distinct from the dashboard's ``/api/no-decision-reasons`` (which buckets
    ALL no-decision reasons, including non-starvation parse failures, and
    runs inside the runner process): this stays inside the host_guard
    shell-runnable scope so the operator can answer the question even when
    the dashboard is unreachable.

    Returns:
        ``{"n": <total rows>, "starved": <starved count>,
           "by_cause": {label: count, ...}, "rate": <starved/n>,
           "ok": bool}``

        ``by_cause`` ALWAYS includes every label in ``_CAUSE_LABELS`` (with
        zero counts for missing classes) so consumers can render the cell
        without key-misses. Counts sum to ``starved`` exactly.

    Degrade-safe — ``ok=False`` on any error (missing DB, locked, schema
    drift); ``by_cause`` returns the all-zero shape (the
    ``recent_starvation_rate`` precedent)."""
    path = str(db_path) if db_path is not None else str(_DEFAULT_DB)
    zero_by_cause = {label: 0 for label in _CAUSE_LABELS}
    out = {"n": 0, "starved": 0, "by_cause": zero_by_cause,
           "rate": 0.0, "ok": False}
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
        try:
            rows = conn.execute(
                "SELECT action_taken, reasoning FROM decisions "
                "ORDER BY id DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        finally:
            conn.close()
    except Exception:
        return out
    n = len(rows)
    by_cause = {label: 0 for label in _CAUSE_LABELS}
    starved = 0
    for action, reasoning in rows:
        if (action or "") != "NO_DECISION":
            continue
        r = reasoning or ""
        if not r.startswith(_STARVATION_PREFIXES):
            continue
        starved += 1
        by_cause[_classify_starvation_cause(r)] += 1
    out.update(n=n, starved=starved, by_cause=by_cause,
               rate=round(starved / n, 3) if n else 0.0, ok=True)
    return out


def snapshot(db_path: str | os.PathLike | None = None) -> dict:
    """Full degrade-safe report: probe + verdict + recent empty-rate."""
    p = probe()
    sat, reason = host_saturated(_probe=p)
    return {
        "saturated": sat,
        "reason": reason,
        "probe": p,
        "recent_empty_rate": recent_empty_rate(db_path),
    }


# Action discriminator appended to every non-CLEAR pulse headline. It names
# the *opposite class* of remediation from _capital_pulse_line's
# ``unlock — sell {ticker}``: capital-paralysis the bot can fix by trading;
# host-saturation it provably cannot — the fix is reducing the out-of-band
# Opus load (review / backtest agents). Stacking the two Discord lines without
# this discriminator invites the operator to read a host freeze as "the bot
# needs to sell something", the exact misattribution this feature closes.
_OPS_ACTION = ("ops — reduce concurrent Opus (review/backtest agents); "
               "the bot cannot resolve this by trading")


def pulse(db_path: str | os.PathLike | None = None,
          *,
          _snapshot: dict | None = None,
          _starv: dict | None = None) -> dict:
    """Single source of truth for "is the desk frozen because the *box* is
    overloaded right now?" — the operator-facing fusion of the live /proc
    probe and the observed starvation rate.

    Distinct from ``snapshot()`` (raw signals) and from ``capital_paralysis``
    (the desk can't act because it's ~98% deployed — a *trading* fix). This
    answers the question those two can't: a 27 h NO_DECISION drought bleeding
    alpha can be host-saturation (Opus never ran — an OPS fix the bot cannot
    take) even while the book also reads capital-pinned. Surfacing only the
    capital story sends the operator to sell a position when the real fix is
    killing the parallel Opus jobs.

    State ladder (the discriminating logic, all test-locked):
      * ``SATURATED`` — the live /proc probe trips now (storm in flight);
        wins regardless of the decision-log rate.
      * ``STARVED``   — probe reads clear now BUT ≥ ``STARVATION_RATE_FLOOR``
        of recent decisions never reached Opus (intermittent storm; the
        damage is in the log even though the box just calmed).
      * ``CLEAR``     — neither (nothing actionable → the caller suppresses).

    ``reason`` is carried **verbatim** from ``snapshot()`` (single source of
    truth — the SATURATED headline embeds it, so this surface, the CLI and
    ``/api/host-guard`` can never drift). Probe-failure reads as "unknown →
    not saturated" and a DB-unreadable starvation probe (``ok=False``) never
    trips STARVED — the ``host_saturated`` mem==0 / ``recent_empty_rate``
    safe-default precedent (never cry wolf on a probe failure). Fully
    degrade-safe: any internal fault → a CLEAR/empty verdict, never raises
    (the module-wide contract). ``_snapshot``/``_starv`` inject deterministic
    readings for tests (the ``host_saturated(_probe=…)`` precedent)."""
    base = {
        "state": "CLEAR",
        "headline": "",
        "saturated": False,
        "reason": "",
        "starvation_rate_pct": 0.0,
        "starvation_n": 0,
        "starvation_ok": False,
        "opus_count": 0,
    }
    # Inert under pytest on the real path (no injectors, not force-flagged):
    # the broad reporter integration tests call _host_pulse_line() →
    # pulse() with no injectors and MUST see a deterministic, offline CLEAR
    # (the _swr_active precedent). Injector-driven and force-flagged calls
    # are deterministic by construction and proceed.
    if (_snapshot is None and _starv is None and not _PULSE_TEST_FORCE
            and os.environ.get("PYTEST_CURRENT_TEST")):
        return base

    try:
        snap = _snapshot if _snapshot is not None else snapshot(db_path)
        starv = _starv if _starv is not None else recent_starvation_rate(
            db_path)

        sat = bool(snap.get("saturated"))
        reason = str(snap.get("reason") or "")
        opus = int((snap.get("probe") or {}).get("opus_count", 0) or 0)
        s_ok = bool(starv.get("ok"))
        s_rate = float(starv.get("rate", 0.0) or 0.0)
        s_n = int(starv.get("n", 0) or 0)
        s_pct = round(s_rate * 100.0, 1)

        base.update(saturated=sat, reason=reason, opus_count=opus,
                    starvation_rate_pct=s_pct, starvation_n=s_n,
                    starvation_ok=s_ok)

        if sat:
            base["state"] = "SATURATED"
            tail = ""
            if s_ok and s_n:
                tail = (f"; {s_pct:.0f}% of the last {s_n} decisions never "
                        f"reached Opus")
            base["headline"] = (
                f"Opus is starved by the box — {reason}{tail}. The desk is "
                f"frozen by host load, not the market or capital — "
                f"{_OPS_ACTION}.")
        elif s_ok and s_rate >= STARVATION_RATE_FLOOR:
            base["state"] = "STARVED"
            base["headline"] = (
                f"{s_pct:.0f}% of the last {s_n} decisions never reached "
                f"Opus (timeout/skip) though the box reads clear now — "
                f"intermittent host starvation, {_OPS_ACTION}.")
        # else: CLEAR — base already correct, headline stays "".
        return base
    except Exception:
        return base


def main(argv: list[str] | None = None) -> int:
    """CLI: human verdict by default, ``--json`` for machines. Exit code 1 when
    saturated so it composes in shell guards
    (``python3 -m paper_trader.host_guard && start-trader``)."""
    argv = sys.argv[1:] if argv is None else argv
    snap = snapshot()
    cause = recent_starvation_by_cause()
    if "--json" in argv:
        import json

        # Surface the starvation figure (counts BOTH prefixes) alongside the
        # narrower empty rate so machine consumers see what the human line
        # below does — the two numbers diverge once the pre-flight guard is
        # live (skip rows aren't counted by recent_empty_rate; see the
        # recent_starvation_rate docstring on why the broader bucket exists).
        out = dict(snap)
        out["recent_starvation_rate"] = recent_starvation_rate()
        out["recent_starvation_by_cause"] = cause
        print(json.dumps(out, indent=2, sort_keys=True))
    else:
        p = snap["probe"]
        verdict = "SATURATED" if snap["saturated"] else "CLEAR"
        print(f"[host-guard] {verdict} — {snap['reason']}")
        print(
            f"  opus={p['opus_count']}  mem_avail={p['mem_available_mb']:.0f}MB"
            f"  swap_used={p['swap_used_pct']:.0f}%"
            f"  load1={p['load1']} ({p['load_per_cpu']}/cpu, {p['cpus']} cpu)"
        )
        # `recent_empty_rate` counts the OLD "claude returned no response"
        # prefix only — once the pre-flight host_saturated guard is live,
        # storms produce mostly "skipped claude call" rows that this bucket
        # silently misses, so the empty% alone underreports the live
        # pathology (observed live 2026-05-21: 14% empty vs 27% starved).
        # Show the broader starvation figure too so the operator can see
        # what the saturation guard *and* the model timeouts actually cost.
        er = snap["recent_empty_rate"]
        if cause["ok"]:
            sr_starved = cause["starved"]
            sr_n = cause["n"]
            sr_pct = round((sr_starved / sr_n) * 100.0, 0) if sr_n else 0.0
            skipped = sr_starved - (er["empty"] if er["ok"] else 0)
            print(
                f"  live trader: {sr_starved}/{sr_n} recent decisions"
                f" never reached Opus ({sr_pct:.0f}%) — "
                f"{er['empty'] if er['ok'] else 0} empty/timeout + "
                f"{max(skipped, 0)} skipped (host guard)"
            )
            # Per-cause breakdown: each class needs a different action, so
            # the aggregate alone is operator-incomplete. Only surface
            # non-zero buckets — a clean "0 model_timeout 0 host_skip ..."
            # line would add noise.
            non_zero = [
                (label, n) for label, n in cause["by_cause"].items() if n > 0
            ]
            if non_zero:
                by_cause_str = "  ".join(
                    f"{label}={n}" for label, n in non_zero
                )
                print(f"    by cause: {by_cause_str}")
        elif er["ok"]:
            # Starvation probe failed but empty probe ok — degrade to the
            # narrower line rather than printing nothing.
            print(
                f"  live trader: {er['empty']}/{er['n']} recent decisions were"
                f" empty/timeout NO_DECISION ({er['rate'] * 100:.0f}%)"
            )
        else:
            print("  live trader: recent starvation rate unavailable (DB unreadable)")
    return 1 if snap["saturated"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
