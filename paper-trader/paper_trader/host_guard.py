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


def main(argv: list[str] | None = None) -> int:
    """CLI: human verdict by default, ``--json`` for machines. Exit code 1 when
    saturated so it composes in shell guards
    (``python3 -m paper_trader.host_guard && start-trader``)."""
    argv = sys.argv[1:] if argv is None else argv
    snap = snapshot()
    if "--json" in argv:
        import json

        print(json.dumps(snap, indent=2, sort_keys=True))
    else:
        p = snap["probe"]
        verdict = "SATURATED" if snap["saturated"] else "CLEAR"
        print(f"[host-guard] {verdict} — {snap['reason']}")
        print(
            f"  opus={p['opus_count']}  mem_avail={p['mem_available_mb']:.0f}MB"
            f"  swap_used={p['swap_used_pct']:.0f}%"
            f"  load1={p['load1']} ({p['load_per_cpu']}/cpu, {p['cpus']} cpu)"
        )
        er = snap["recent_empty_rate"]
        if er["ok"]:
            print(
                f"  live trader: {er['empty']}/{er['n']} recent decisions were"
                f" empty/timeout NO_DECISION ({er['rate'] * 100:.0f}%)"
            )
        else:
            print("  live trader: recent empty-rate unavailable (DB unreadable)")
    return 1 if snap["saturated"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
