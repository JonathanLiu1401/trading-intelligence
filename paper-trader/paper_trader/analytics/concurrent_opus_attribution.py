"""Concurrent-Opus attribution — group live Opus subprocesses by spawning parent.

``host_guard.snapshot()`` and ``/api/host-guard`` already report the *count*
of live ``claude --model claude-opus`` subprocesses on the box and verdict
SATURATED above ``DEFAULT_MAX_OPUS=4``. What neither answers — and the
operator standing in front of a 17-Opus host saturation storm asks first —
is:

  **"WHICH parent trees own those 17 processes, and which can I safely
  terminate to restore the live runner's Opus call?"**

Without that attribution the operator either kills indiscriminately
(``pkill -9 claude`` also nukes the legitimate live runner's in-flight
decision and every feature-dev agent the operator is running) or waits
hours for the storm to clear naturally. The 2026-05-23 18:20Z live state
showed all 17 Opus traced to ``scripts/hourly_review.sh`` ancestors — a
single targeted kill would have unblocked the desk frozen since
2026-05-21 10:14Z (>55h).

This builder is the missing attribution lens. Pure on a synthetic process
table; the endpoint owns the ``/proc`` walk (the ``host_guard.probe`` /
endpoint split — pure builder, endpoint owns the I/O). Each Opus process
walks its parent chain (PPID → PPID → …) until either a known *root*
marker is hit (``hourly_review.sh`` / ``run_continuous_backtests`` /
``paper_trader.runner`` / ``daemon.py``) or the chain falls off (init).
Unrecognized roots bucket as ``unknown`` so the per-parent histogram sums
back to the total Opus count exactly.

Verdict ladder (Opus count crossed against ``host_guard`` thresholds):

|  Verdict       | Opus count                                           |
|----------------|------------------------------------------------------|
| ``NO_OPUS``    | 0 concurrent Opus — host clean                       |
| ``CLEAN``      | 1 Opus (the live runner itself or a single backtest) |
| ``BENIGN``     | 2-4 Opus (within ``host_guard.DEFAULT_MAX_OPUS``)    |
| ``ELEVATED``   | 5-8 Opus (above guard, not yet swap-thrashing)       |
| ``SATURATED``  | ≥9 Opus (multi-pool storm; live runner is starved)   |

Per-group report (one entry per parent-tree bucket):

* ``parent_marker``    — canonical bucket key (``hourly_review`` / ``backtest`` /
                         ``runner`` / ``daemon`` / ``unknown``)
* ``parent_label``     — human label for the dashboard / chat surface
* ``n_opus``           — count of Opus subprocesses attributed to this parent
* ``pids``             — list of Opus PIDs in this group (deterministic sort)
* ``is_legitimate``    — bool: True for the live ``runner`` group, False for
                         the rest (legitimate ≤ 1 — the runner spawns ONE
                         Opus per decision cycle). ``backtest`` is marked
                         legitimate iff its count ≤ ``backtest._CLAUDE_SEM=3``;
                         beyond that it's an out-of-band storm.
* ``kill_command``     — the *exact* shell command the operator would run to
                         terminate just this group. ``pkill -f <marker>`` for
                         script-rooted groups; ``kill <space-separated PIDs>``
                         for chain-fallthrough groups where no script-level
                         marker can be pkill'd.

Recommendation string mirrors the ``no_decision_reasons`` precedent: when
ANY non-legitimate group exists it prescribes the exact remediation; when
all Opus are legitimate it reads silent (verdict still CLEAN/BENIGN).

Pure & offline. The ``processes`` input is a list of plain dicts
``{pid: int, ppid: int, cmdline: str}``. Tests inject a synthetic table;
the production endpoint walks ``/proc`` directly (no psutil) and feeds the
same shape. Never raises — garbage rows degrade to ``NO_OPUS`` /
``unknown``.

Advisory only. It mints no directive, runs no kill, has no path to
``_execute()``. The kill commands are *prescribed* strings the operator
copies into their own shell — the same observational contract as every
other analytics surface (AGENTS.md #2/#12).
"""
from __future__ import annotations

from typing import Any

# Substring matching the live Opus model invocation. Mirrored from
# ``host_guard._OPUS_MARKER`` so a future model-name change updates both
# (cross-module SSOT — the live trader runs ONE Opus model at a time).
DEFAULT_OPUS_MARKER = "claude --model claude-opus"

# Maximum PPID hops when walking the parent chain. Real chains observed
# live are ≤4 deep (Opus → bash hourly_review.sh → bash hourly_review.sh →
# systemd/init). Cap defends against pathological cyclic /proc reads
# (extremely unlikely but possible during a /proc race) — chain hits the
# cap, group falls to ``unknown``, builder never spins.
DEFAULT_MAX_PPID_HOPS = 8

# Verdict-ladder cutoffs (Opus count). Aligned with
# ``host_guard.DEFAULT_MAX_OPUS=4`` so a SATURATED verdict here is a
# strict superset of host_guard's saturation predicate.
BENIGN_MAX = 4         # ≤4 is within host_guard's guard
ELEVATED_MAX = 8       # 5-8 is above guard, not yet swap-thrash
# 9+ is SATURATED — the live 17-Opus storm category

# Backtest committee's own ``_CLAUDE_SEM`` cap. A backtest group within
# this count is legitimate (within designed limits); above it the
# committee has slipped its cap (a real bug — flagged).
BACKTEST_LEGIT_MAX = 3

# Parent-marker dispatch table. Order matters: more specific markers must
# come first (``run_continuous_backtests`` BEFORE generic ``python`` /
# ``paper_trader``). Each entry's ``cmdline_substring`` is searched into
# the parent process's full cmdline (whitespace-joined argv) using a plain
# substring test — no regex, no normalization beyond what /proc gives us.
#
# ``kill_template`` accepts the placeholders ``{marker}`` (the substring
# itself, suitable for ``pkill -f``) and ``{pids}`` (space-separated PIDs).
_PARENT_MARKERS: tuple[dict[str, Any], ...] = (
    {
        "key": "hourly_review",
        "label": "hourly self-review (scripts/hourly_review.sh)",
        "cmdline_substring": "scripts/hourly_review.sh",
        "is_legitimate": False,
        "max_legit_count": 0,
        "kill_template": "pkill -f scripts/hourly_review.sh",
    },
    {
        "key": "backtest",
        "label": "continuous backtest committee (run_continuous_backtests.py)",
        "cmdline_substring": "run_continuous_backtests",
        "is_legitimate": True,
        "max_legit_count": BACKTEST_LEGIT_MAX,
        "kill_template": "pkill -f run_continuous_backtests",
    },
    {
        "key": "runner",
        "label": "live paper-trader runner (paper_trader.runner)",
        "cmdline_substring": "paper_trader.runner",
        "is_legitimate": True,
        "max_legit_count": 1,
        "kill_template": "systemctl --user restart paper-trader",
    },
    {
        "key": "daemon",
        "label": "digital-intern daemon (daemon.py)",
        "cmdline_substring": "daemon.py",
        "is_legitimate": False,
        "max_legit_count": 0,
        "kill_template": "systemctl --user restart digital-intern",
    },
)


def _safe_int(v: Any) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _safe_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    try:
        return str(v)
    except Exception:
        return ""


def _index(processes: list[dict] | None) -> dict[int, dict]:
    """Build a pid → {ppid, cmdline} lookup from the input rows.
    Garbage rows (missing pid, non-int pid) drop silently."""
    out: dict[int, dict] = {}
    if not processes:
        return out
    for row in processes:
        if not isinstance(row, dict):
            continue
        pid = _safe_int(row.get("pid"))
        if pid <= 0:
            continue
        out[pid] = {
            "ppid": _safe_int(row.get("ppid")),
            "cmdline": _safe_str(row.get("cmdline")),
        }
    return out


def _classify_parent(
    pid: int,
    table: dict[int, dict],
    *,
    max_hops: int,
) -> tuple[str, str]:
    """Walk the parent chain from ``pid`` until a known marker hits or the
    chain falls off. Returns ``(marker_key, parent_cmdline)``. ``unknown``
    is the sink for chains that don't match any marker.

    The walk INCLUDES ``pid`` itself only via its parent (we never classify
    the Opus process by its own cmdline — by construction every Opus row's
    own cmdline contains the Opus marker, which is not what we're
    attributing). The first hop is to ``table[pid].ppid``.
    """
    seen: set[int] = set()
    cur = table.get(pid)
    if not cur:
        return ("unknown", "")
    parent_pid = cur.get("ppid", 0)
    chain_top_cmdline = ""
    for _ in range(max_hops):
        if parent_pid <= 1 or parent_pid in seen:
            break
        seen.add(parent_pid)
        parent = table.get(parent_pid)
        if not parent:
            break
        pcmd = parent.get("cmdline", "")
        if pcmd:
            chain_top_cmdline = pcmd
        for entry in _PARENT_MARKERS:
            if entry["cmdline_substring"] in pcmd:
                return (entry["key"], pcmd)
        parent_pid = parent.get("ppid", 0)
    return ("unknown", chain_top_cmdline)


def _verdict_for_count(n: int) -> tuple[str, str]:
    """(verdict, state) for an Opus count. Mirrors host_guard's thresholds."""
    if n <= 0:
        return ("NO_OPUS", "READY")
    if n == 1:
        return ("CLEAN", "READY")
    if n <= BENIGN_MAX:
        return ("BENIGN", "READY")
    if n <= ELEVATED_MAX:
        return ("ELEVATED", "DEGRADED")
    return ("SATURATED", "STORM")


def _marker_meta(key: str) -> dict[str, Any]:
    for entry in _PARENT_MARKERS:
        if entry["key"] == key:
            return entry
    return {
        "key": "unknown",
        "label": "unknown parent (chain fell off or interactive shell)",
        "is_legitimate": False,
        "max_legit_count": 0,
        "kill_template": "",
    }


def _kill_command_for(meta: dict[str, Any], pids: list[int]) -> str:
    """Render the operator-facing kill command for a group.

    For known markers, the template is preferred (``pkill -f`` is safer than
    listing dozens of PIDs that may have churned by the time the operator
    copies). For ``unknown`` groups (chain fell off), the only honest answer
    is the explicit PID list — the operator must inspect each before kill.
    """
    if meta.get("kill_template"):
        return meta["kill_template"]
    if not pids:
        return ""
    return "kill " + " ".join(str(p) for p in sorted(pids))


def _build_recommendation(groups: list[dict]) -> str:
    """One-line operator action. ``""`` when every group is legitimate."""
    illegit = [g for g in groups if not g["is_legitimate"]]
    if not illegit:
        return ""
    # Order illegit groups by descending count — biggest culprit first.
    illegit_sorted = sorted(illegit, key=lambda g: -g["n_opus"])
    head = illegit_sorted[0]
    extras = illegit_sorted[1:]
    parts = [
        f"{head['n_opus']} Opus from {head['parent_label']} — "
        f"`{head['kill_command']}`"
    ]
    for g in extras:
        parts.append(
            f"plus {g['n_opus']} from {g['parent_label']} — "
            f"`{g['kill_command']}`"
        )
    return "; ".join(parts) + "."


def build_concurrent_opus_attribution(
    processes: list[dict] | None,
    *,
    marker: str = DEFAULT_OPUS_MARKER,
    max_ppid_hops: int = DEFAULT_MAX_PPID_HOPS,
    as_of: str | None = None,
) -> dict:
    """Pure builder. ``processes`` is a list of
    ``{pid: int, ppid: int, cmdline: str}`` rows (the same shape the
    production endpoint produces from a ``/proc`` walk; tests inject a
    deterministic synthetic table).

    Never raises. Garbage input → ``NO_OPUS`` / empty groups.
    """
    table = _index(processes)
    opus_pids: list[int] = []
    for pid, row in table.items():
        cmd = row.get("cmdline", "")
        if cmd and marker in cmd:
            opus_pids.append(pid)
    opus_pids.sort()
    n_opus = len(opus_pids)

    # Group every Opus by its classified parent.
    by_marker: dict[str, list[int]] = {}
    for opus_pid in opus_pids:
        key, _parent_cmd = _classify_parent(
            opus_pid, table, max_hops=max_ppid_hops
        )
        by_marker.setdefault(key, []).append(opus_pid)

    groups: list[dict] = []
    for key, pids in by_marker.items():
        meta = _marker_meta(key)
        n = len(pids)
        max_legit = int(meta.get("max_legit_count") or 0)
        is_legit = bool(meta.get("is_legitimate")) and n <= max_legit
        groups.append({
            "parent_marker": key,
            "parent_label": meta["label"],
            "n_opus": n,
            "pids": sorted(pids),
            "is_legitimate": is_legit,
            "kill_command": _kill_command_for(meta, pids),
        })
    # Deterministic order: by count desc, then by marker key asc.
    groups.sort(key=lambda g: (-g["n_opus"], g["parent_marker"]))

    verdict, state = _verdict_for_count(n_opus)
    recommendation = _build_recommendation(groups)

    # Dominant illegitimate group (the single biggest contributor the
    # operator should kill first), if any.
    illegit = [g for g in groups if not g["is_legitimate"]]
    dominant_culprit = illegit[0] if illegit else None

    if n_opus == 0:
        headline = "0 concurrent Opus — host clean."
    elif not illegit:
        headline = (
            f"{n_opus} concurrent Opus — all legitimate "
            f"(runner / backtest within caps)."
        )
    else:
        n_illegit = sum(g["n_opus"] for g in illegit)
        headline = (
            f"{n_opus} concurrent Opus, {n_illegit} from "
            f"non-legitimate parents — host is saturated by "
            f"{dominant_culprit['parent_label']} "
            f"({dominant_culprit['n_opus']} Opus). "
            f"Targeted kill: `{dominant_culprit['kill_command']}`."
        )

    return {
        "as_of": as_of,
        "verdict": verdict,
        "state": state,
        "n_opus": n_opus,
        "n_processes_scanned": len(table),
        "opus_marker": marker,
        "headline": headline,
        "groups": groups,
        "dominant_culprit": dominant_culprit,
        "recommendation": recommendation,
    }


def scan_proc_table() -> list[dict]:
    """Walk ``/proc`` and return one dict per running process.

    Endpoint-side helper. Kept here (not in ``host_guard``) because this
    builder is the only consumer of the (pid, ppid, cmdline) triple; the
    host_guard probe walks ``/proc`` for a count only. Degrade-safe: any
    /proc read error skips that row; an unreadable /proc as a whole
    returns ``[]``.

    Tests never call this — they inject a deterministic synthetic table
    via ``build_concurrent_opus_attribution(processes=...)``.
    """
    import os

    out: list[dict] = []
    try:
        entries = os.listdir("/proc")
    except Exception:
        return out
    for entry in entries:
        if not entry.isdigit():
            continue
        pid = int(entry)
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as fh:
                raw = fh.read()
        except Exception:
            continue
        cmdline = raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
        if not cmdline:
            # Kernel threads have empty cmdline — exclude from the attribution
            # surface (they cannot be claude/Opus and are noise in the parent
            # chain).
            continue
        ppid = 0
        try:
            with open(f"/proc/{pid}/status", "r", encoding="utf-8",
                      errors="replace") as fh:
                for line in fh:
                    if line.startswith("PPid:"):
                        try:
                            ppid = int(line.split(":", 1)[1].strip())
                        except ValueError:
                            ppid = 0
                        break
        except Exception:
            pass
        out.append({"pid": pid, "ppid": ppid, "cmdline": cmdline})
    return out
