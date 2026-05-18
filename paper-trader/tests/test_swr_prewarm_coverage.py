"""Regression-lock: every @swr_cached endpoint must be pre-warmed at boot.

`dashboard._swr_prewarm` exists so the FIRST poll of each slow SWR endpoint
after a service restart serves real data instead of the cold-path stall +
``{"warming": true}`` placeholder. Its own docstring promises it pre-builds
*"every slow SWR cache once at boot"*.

The bug this locks: the 2026-05-18 commit that SWR-wrapped /api/risk,
/api/benchmark, /api/capital-paralysis and /api/decision-health (plus the
pre-existing runner-heartbeat / scorer-confidence) added the
``@swr_cached`` decorator but NOT the matching ``_swr_prewarm`` target.
Result, observed live under host load 13-21: a trader who opens the
dashboard right after a restart — exactly when triaging "why is the bot
frozen?" — gets "computing — retry shortly" on the five most
decision-critical panels (risk, capital-paralysis, decision-health,
runner-heartbeat, benchmark) because only those panels were never warmed.

This test fails the instant a new @swr_cached endpoint is added without a
prewarm target, so the prewarm contract can never silently rot again. It is
pure source/introspection — no DB, no network, no :8090 bind.
"""
import inspect
import re

import paper_trader.dashboard as d

# `@swr_cached("name", ttl)` — capture the cache name. Anchored on the
# decorator call form so prose mentioning the word in a comment / docstring
# (no `(` + quote) can never match. Excludes the `def swr_cached` definition.
_SWR_DECO_RE = re.compile(r"@swr_cached\(\s*[\"']([^\"']+)[\"']")
# The exact `("name", handler_symbol)` tuple form _swr_prewarm uses. Matching
# the whole tuple (quoted name followed by `, identifier)`) is robust to
# arbitrary quotes in surrounding comment prose — `{"warming": true}` in a
# comment is not followed by `, <identifier>)` so it can never masquerade as
# a target. Captures name and handler together so the two stay paired.
_TARGET_TUPLE_RE = re.compile(
    r"\(\s*[\"']([^\"']+)[\"']\s*,\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)"
)


def _swr_cached_names() -> set[str]:
    """Every cache name that carries an @swr_cached decorator in dashboard.py."""
    src = inspect.getsource(d)
    return set(_SWR_DECO_RE.findall(src))


def _prewarm_targets():
    """(names, handler_symbols) referenced by _swr_prewarm's targets list."""
    src = inspect.getsource(d._swr_prewarm)
    pairs = _TARGET_TUPLE_RE.findall(src)
    names = {n for n, _h in pairs}
    handlers = {h for _n, h in pairs}
    return names, handlers


def test_every_swr_cached_endpoint_is_prewarmed():
    cached = _swr_cached_names()
    warmed, _ = _prewarm_targets()
    # Sanity: the introspection actually found the known endpoints, so a
    # regex that silently matches nothing can't make this test vacuously pass.
    assert "state" in cached and "risk" in cached
    assert len(cached) >= 20, f"expected ~22 swr_cached endpoints, got {len(cached)}"

    missing = sorted(cached - warmed)
    assert not missing, (
        "These @swr_cached endpoints are NOT in _swr_prewarm.targets, so "
        "their panels cold-stall with {'warming': true} after every restart "
        f"(the freeze-triage blind spot): {missing}"
    )


def test_prewarm_handlers_resolve_to_callables():
    """Each prewarm target's handler symbol must resolve to a real module
    attribute, and its undecorated form (what _swr_refresh actually invokes
    via ``__wrapped__``) must be callable. A typo'd symbol would make
    _swr_prewarm silently skip that endpoint at runtime (its try/except just
    logs and continues), re-introducing the cold-stall it exists to prevent."""
    _, handler_syms = _prewarm_targets()
    assert handler_syms, "no handler symbols parsed from _swr_prewarm.targets"
    for sym in handler_syms:
        fn = getattr(d, sym, None)
        assert fn is not None, f"_swr_prewarm references unknown handler {sym!r}"
        raw = getattr(fn, "__wrapped__", fn)
        assert callable(raw), f"{sym}.__wrapped__ is not callable"


def test_freeze_triage_panels_specifically_prewarmed():
    """The five panels a trader hits FIRST when the bot looks frozen must be
    warm on first paint — this is the concrete operator-facing contract, not
    just set-coverage."""
    warmed, _ = _prewarm_targets()
    for critical in ("risk", "benchmark", "capital-paralysis",
                     "decision-health", "runner-heartbeat"):
        assert critical in warmed, (
            f"freeze-triage panel {critical!r} is not prewarmed — it will "
            f"show 'computing — retry shortly' right when the operator needs "
            f"it during a freeze"
        )
