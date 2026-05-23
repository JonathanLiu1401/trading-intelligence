"""Regression guard: no production code may open ``articles.db`` with
``immutable=1``.

Background — the ``immutable=1`` URI flag tells SQLite the file will *never*
change. On the live ``articles.db`` (~1.6 GB, ~30 worker threads writing into
it continuously) the flag causes intermittent "database disk image is
malformed" errors because the cached pages diverge from disk. Commit
``cdd8d4a`` removed it from ``score_drift_detector`` and ``source_score_drift``;
this guard pins the rule so the next analytics module added can't silently
reintroduce the same hazard.

What's still allowed
--------------------
``file:{path}?mode=ro`` (without ``immutable=1``) — read-only is fine on a
live DB and is the canonical pattern across ~50 analytics modules. The flag
this test forbids is specifically ``immutable=1``.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Subtrees the rule applies to. Test files in ``tests/`` may legitimately
# build a frozen fixture DB with ``immutable=1`` (the flag is correct *when*
# the file truly won't be written) so they are intentionally excluded.
_PRODUCTION_SUBTREES = (
    "analytics",
    "analysis",
    "collectors",
    "core",
    "dashboard",
    "ml",
    "scripts",
    "storage",
    "watchers",
)

_IMMUTABLE_RE = re.compile(r"immutable\s*=\s*1", re.IGNORECASE)


def _python_sources() -> list[Path]:
    files: list[Path] = []
    for sub in _PRODUCTION_SUBTREES:
        root = REPO_ROOT / sub
        if not root.is_dir():
            continue
        files.extend(root.rglob("*.py"))
    return files


def test_no_immutable_uri_in_production_code():
    """Every production .py file under analytics/analysis/collectors/core/
    dashboard/ml/scripts/storage/watchers must NOT contain ``immutable=1``.

    A test-side fixture builder may still use it (the flag is correct for a
    file that genuinely never changes during the test). Only production
    callers reading the live ``articles.db`` are guarded here."""
    offenders: list[str] = []
    for path in _python_sources():
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            # Allow it inside comments/docstrings that explain WHY the flag
            # is forbidden — those are the documentation, not the bug.
            if _IMMUTABLE_RE.search(line) and "sqlite3.connect" in text:
                stripped = line.lstrip()
                if stripped.startswith("#"):
                    continue
                # Require the offending line to participate in a connect()
                # URI: look for ``file:`` on the same line OR the previous
                # line's continuation. Cheap-and-correct: every real
                # occurrence in this repo's history was on the connect line.
                prev = text.splitlines()[lineno - 2] if lineno >= 2 else ""
                if "file:" in line or "file:" in prev:
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{lineno}: {line.strip()}")

    assert not offenders, (
        "production code must not open articles.db with immutable=1 — it "
        "causes 'database disk image is malformed' on the live, actively-"
        "written DB. Switch to plain ``mode=ro`` (without immutable). "
        "Offenders:\n  " + "\n  ".join(offenders)
    )


def test_guarded_files_use_mode_ro_instead():
    """The two files this commit fixed (junk_source_detector,
    source_lead_time) must use ``mode=ro`` and must NOT pass ``immutable=1``
    to ``sqlite3.connect``. ``immutable=1`` is allowed in explanatory
    comments/docstrings (they're documentation, not the bug)."""
    for rel in ("analytics/junk_source_detector.py",
                "analytics/source_lead_time.py"):
        p = REPO_ROOT / rel
        text = p.read_text(encoding="utf-8")
        assert "mode=ro" in text, f"{rel} should still open read-only"
        for lineno, line in enumerate(text.splitlines(), start=1):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            if "sqlite3.connect" in line and _IMMUTABLE_RE.search(line):
                raise AssertionError(
                    f"{rel}:{lineno} reintroduced immutable=1 in a connect "
                    f"call: {line.strip()}"
                )
