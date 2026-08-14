"""Bound digital-intern workers so a wiped env cannot start ~147 threads.

DIGITAL_INTERN_WORKERS accepts either:
  - a positive integer cap (default 4, hard-max 8)
  - a comma-separated worker-name allowlist (legacy)

Empty / unset is treated as the default cap so max-throughput cannot return
on a 16GB box.
"""
from __future__ import annotations

DEFAULT_WORKER_CAP = 4
HARD_MAX_WORKER_CAP = 8

# First N of this list when the env value is numeric or empty.
# web_server stays first so :8080 survives a tight cap.
# continuous_trainer is intentionally absent.
BOUNDED_WORKER_PRIORITY = (
    "web_server",
    "rss",
    "scorer",
    "heartbeat",
    "web",
    "reddit",
    "alert",
    "purge",
)


def parse_worker_spec(raw: str | None) -> tuple[int | None, set[str]]:
    """Return (numeric_cap, name_allowlist). Cap wins when raw is empty or int."""
    text = (raw or "").strip()
    if not text:
        return DEFAULT_WORKER_CAP, set()
    if text.isdigit():
        n = int(text)
        if n <= 0:
            return DEFAULT_WORKER_CAP, set()
        return min(n, HARD_MAX_WORKER_CAP), set()
    names = {part.strip() for part in text.split(",") if part.strip()}
    return None, names


def resolve_worker_names(available: list[str] | tuple[str, ...], raw: str | None) -> list[str]:
    """Preserve *available* order. Numeric/empty uses BOUNDED_WORKER_PRIORITY."""
    cap, allow = parse_worker_spec(raw)
    known = list(available)
    known_set = set(known)
    if cap is not None:
        out: list[str] = []
        for name in BOUNDED_WORKER_PRIORITY:
            if name in known_set and name not in out:
                out.append(name)
                if len(out) >= cap:
                    break
        return out
    return [name for name in known if name in allow]


def configured_allowlist(raw: str | None, available: list[str] | tuple[str, ...] | None = None) -> set[str] | None:
    """Dashboard helper: set of enabled names. Never None for empty/numeric (that used to mean all 147)."""
    names = resolve_worker_names(available or BOUNDED_WORKER_PRIORITY, raw)
    return set(names) if names else set()
