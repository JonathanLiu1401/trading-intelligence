"""Static-analysis regression guard: every ``bo.<method>(...)`` call inside a
daemon worker must resolve to an actual ``core.backoff.Backoff`` attribute.

Background. ``daemon.py::un_news_worker`` shipped in commit ``71e9cd7`` with
``bo.advance()`` in its except-branch — a method that has never existed on
``Backoff``. Every time ``collect_un_news()`` raised (frequent, because of DB
write-lock contention) the worker AttributeErrored, the supervisor counted a
crash, and the back-off window was never actually applied — so the failing
worker re-fired immediately at every cycle, burning supervisor crash budget
unnecessarily. Live evidence: ``logs/daemon.log.1`` line 270:

    [un_news] thread exited with AttributeError: 'Backoff' object has no
    attribute 'advance'

The fix replaces ``bo.advance()`` with the canonical ``bo.sleep(lambda:
_running)`` pattern used by every other worker. This test parses the AST of
``daemon.py`` and asserts:

  1. Every ``bo.<attr>(...)`` call inside a function whose body declares
     ``bo = Backoff(...)`` resolves to a real Backoff attribute.
  2. ``un_news_worker`` specifically uses ``while _running`` (not the
     ``while True`` of the buggy original) and contains ``bo.sleep(`` in its
     body — pinning the canonical worker shape so a future drift back to the
     broken pattern fails this test.

Pure read of daemon.py source — no daemon import, no side effects, no
SQLite, no network."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest


_DAEMON_PATH = Path(__file__).resolve().parent.parent / "daemon.py"


def _backoff_public_attrs() -> set[str]:
    """Return the set of real ``Backoff`` public attributes / methods.

    Includes dunders and slot fields so a future Backoff that *renames*
    ``sleep`` -> ``wait`` would be caught here (the worker call sites would
    need to update in lockstep). ``__slots__`` is respected — Backoff defines
    its full attribute surface there."""
    from core.backoff import Backoff
    attrs = set(dir(Backoff))
    # Add slot fields that dir() on the *class* may not enumerate the same
    # way as dir() on an instance.
    attrs.update(getattr(Backoff, "__slots__", ()))
    return attrs


def _worker_funcdefs() -> list[ast.FunctionDef]:
    """Top-level FunctionDefs in daemon.py whose body declares ``bo =
    Backoff(...)`` — the daemon's worker functions."""
    tree = ast.parse(_DAEMON_PATH.read_text())
    out: list[ast.FunctionDef] = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        for child in ast.walk(node):
            if (isinstance(child, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "bo"
                            for t in child.targets)
                    and isinstance(child.value, ast.Call)
                    and isinstance(child.value.func, ast.Name)
                    and child.value.func.id == "Backoff"):
                out.append(node)
                break
    return out


def _bo_attrs_called_in(fn: ast.FunctionDef) -> set[str]:
    """Names referenced as ``bo.<attr>`` inside ``fn``."""
    refs: set[str] = set()
    for child in ast.walk(fn):
        if (isinstance(child, ast.Attribute)
                and isinstance(child.value, ast.Name)
                and child.value.id == "bo"):
            refs.add(child.attr)
    return refs


class TestWorkerBackoffContract:
    def test_every_bo_method_resolves_to_backoff_attr(self):
        """For every worker that builds a ``bo = Backoff(...)``, each
        ``bo.<attr>(...)`` callsite must name a real attribute on the Backoff
        class. The exact bug class the regression fixes."""
        real = _backoff_public_attrs()
        workers = _worker_funcdefs()
        assert workers, "no Backoff-using worker functions found in daemon.py"
        offenders: list[tuple[str, str]] = []
        for fn in workers:
            for attr in _bo_attrs_called_in(fn):
                if attr not in real:
                    offenders.append((fn.name, attr))
        assert not offenders, (
            f"daemon.py workers reference non-existent Backoff attrs: "
            f"{offenders!r}. core.backoff.Backoff actually exposes: "
            f"{sorted(a for a in real if not a.startswith('_'))}. "
            f"Every other worker uses bo.sleep(lambda: _running) on the "
            f"except-branch; the regression introduces bo.advance()/.wait()/"
            f"some-other-typo and silently raises AttributeError on the "
            f"first failure, then never sleeps."
        )

    def test_un_news_worker_uses_canonical_shape(self):
        """Specific shape pin for ``un_news_worker``. The buggy original used
        ``while True:`` and ``bo.advance()``. Every other worker uses ``while
        _running:`` + ``bo.sleep(``. Pin BOTH so a regression to either
        broken form fails here."""
        tree = ast.parse(_DAEMON_PATH.read_text())
        un_news = next(
            (n for n in tree.body
             if isinstance(n, ast.FunctionDef) and n.name == "un_news_worker"),
            None,
        )
        assert un_news is not None, "un_news_worker function missing from daemon.py"

        # Body source — exact substring check is the easiest way to pin both
        # the `while _running:` header and the `bo.sleep(` callsite together.
        src = ast.get_source_segment(_DAEMON_PATH.read_text(), un_news) or ""
        assert "while _running:" in src, (
            "un_news_worker must loop on `while _running:` (matches every "
            "other worker; lets the daemon shutdown signal stop the loop "
            "promptly)"
        )
        assert "while True:" not in src, (
            "un_news_worker must NOT use `while True:` — the original buggy "
            "shape; replace with `while _running:`"
        )
        assert "bo.sleep(" in src, (
            "un_news_worker except-branch must call `bo.sleep(lambda: "
            "_running)` — the canonical exponential-backoff pattern. The "
            "original buggy version called the non-existent `bo.advance()`."
        )
        assert "bo.advance(" not in src, (
            "un_news_worker references `bo.advance()`, which does not exist "
            "on core.backoff.Backoff. The except-branch must use `bo.sleep("
            "lambda: _running)` instead."
        )


class TestRegressionEvidenceLocked:
    """One self-checking guard: the daemon-log evidence cited in this file's
    module docstring is meaningless if Backoff somehow grows an ``advance``
    method in the future and the worker silently keeps using it. Lock the
    expectation that ``advance`` is NOT on Backoff so any such future change
    must update this test consciously."""

    def test_backoff_does_not_define_advance(self):
        from core.backoff import Backoff
        assert not hasattr(Backoff, "advance"), (
            "core.backoff.Backoff grew an `advance` method. If that is "
            "intentional, update tests/test_worker_backoff_contract.py — "
            "the un_news_worker AttributeError this regression test pins "
            "(2026-05-23 .. 2026-05-29 daemon.log.1 evidence) needs a new "
            "lock once `advance` legitimately exists."
        )

    def test_backoff_exposes_canonical_methods(self):
        """The Backoff API every worker depends on: ``peek``, ``reset``,
        ``sleep``. If any of these are renamed, the worker call-sites need
        the same rename in lockstep."""
        from core.backoff import Backoff
        for name in ("peek", "reset", "sleep"):
            assert callable(getattr(Backoff, name, None)), (
                f"core.backoff.Backoff.{name} missing or non-callable — "
                f"every daemon worker depends on this method"
            )
