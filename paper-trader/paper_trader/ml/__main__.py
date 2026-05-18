"""Unified launcher / index for the paper_trader.ml read-only diagnostics.

Every diagnostic in this package already ships its own
``python3 -m paper_trader.ml.<name>`` CLI (decision_scorer --explain,
feature_importance, gate_audit, calibration, deploy_audit, scorer_freshness,
overfit_gap, persona_skill, … — 22 of them). What was missing is a single
discoverable entry point. On the operator box the decision_scorer CLI comment
describes ("the 78%-NO_DECISION operational reality means an operator is
usually on a shell triaging, not in a browser") you had to already KNOW all
~22 module names — there was no `--help`-style index and no one place that
lists what each tool answers. This adds exactly that:

    python3 -m paper_trader.ml                  # table: every diagnostic + its 1-line purpose
    python3 -m paper_trader.ml --json           # same index, machine-readable
    python3 -m paper_trader.ml <name> [args…]   # run that diagnostic; argv passed through verbatim

It is pure dispatch + discovery: NO new analysis, NO existing module touched,
read-only, stdlib-only, lazy. The index is built by `ast`-parsing each
sibling's module docstring (no import → zero side effects, and a sibling with
a syntax error degrades to "(docstring unavailable)" instead of taking the
whole index down — relevant under the concurrent-agent staging race). A
selected diagnostic is executed exactly as ``python3 -m
paper_trader.ml.<name>`` would be (via ``runpy`` with ``run_name="__main__"``,
``sys.argv`` rebuilt), so its own argparse, ``--json`` and ``SystemExit``
exit-code contract are preserved verbatim — shell callers keep gating on
``$?`` just like host_guard / decision_scorer already document. Launcher
options (``--json`` / ``--help``) are only honoured as the FIRST token; once a
module name is given, every following token belongs to the child, so
``python3 -m paper_trader.ml decision_scorer --json --explain --ticker NVDA``
forwards ``--json --explain --ticker NVDA`` to decision_scorer untouched.

Unknown / missing name → print the index and exit 2.
"""
from __future__ import annotations

import ast
import json
import runpy
import sys
from pathlib import Path

_PKG = "paper_trader.ml"
_PKG_DIR = Path(__file__).resolve().parent
# Never dispatchable: package plumbing (and self — recursing into the launcher
# would be nonsense). Everything else in the dir that carries a
# `if __name__ == "__main__":` guard is a runnable diagnostic.
_NOT_A_TOOL = {"__init__", "__main__"}


def _module_purpose(path: Path) -> str:
    """First non-empty line of the module docstring, via ast (no import).

    A sibling mid-edit by a concurrent agent (syntax error) must not break the
    whole index — degrade that one row instead of raising."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        doc = ast.get_docstring(tree)
    except Exception:
        return "(docstring unavailable — module unparseable)"
    if not doc:
        return "(no module docstring)"
    for line in doc.splitlines():
        line = line.strip()
        if line:
            return line.rstrip(".")
    return "(no module docstring)"


def _discover() -> list[tuple[str, str]]:
    """Sorted [(module_name, one_line_purpose)] for every runnable diagnostic.

    "Runnable" == the file contains a top-level ``if __name__ == "__main__":``
    guard (the established convention across all 22 modules), so a pure library
    helper that ever lands here is correctly excluded with no allow-list to
    maintain."""
    tools: list[tuple[str, str]] = []
    for path in sorted(_PKG_DIR.glob("*.py")):
        name = path.stem
        if name in _NOT_A_TOOL:
            continue
        try:
            src = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if 'if __name__ == "__main__":' not in src:
            continue
        tools.append((name, _module_purpose(path)))
    return tools


def _print_index(tools: list[tuple[str, str]]) -> None:
    width = max((len(n) for n, _ in tools), default=4)
    print(f"paper_trader.ml — {len(tools)} read-only diagnostics\n")
    print(f"  {'tool':<{width}}  purpose")
    print(f"  {'-' * width}  {'-' * 7}")
    for name, purpose in tools:
        print(f"  {name:<{width}}  {purpose}")
    print(
        "\nRun one:   python3 -m paper_trader.ml <tool> [args…]"
        "\nTool help: python3 -m paper_trader.ml <tool> --help"
    )


def _cli(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    tools = _discover()
    names = {n for n, _ in tools}

    # Launcher options are only meaningful as the FIRST token. Anything after a
    # module name belongs to the child verbatim.
    if not args or args[0] in ("--list", "-l"):
        _print_index(tools)
        return 0
    if args[0] in ("-h", "--help"):
        print(__doc__)
        _print_index(tools)
        return 0
    if args[0] == "--json" and len(args) == 1:
        print(json.dumps(
            {"package": _PKG,
             "count": len(tools),
             "tools": [{"name": n, "purpose": p} for n, p in tools]},
            indent=2, sort_keys=True,
        ))
        return 0

    name = args[0]
    if name not in names:
        print(f"[paper_trader.ml] unknown diagnostic: {name!r}\n", file=sys.stderr)
        _print_index(tools)
        return 2

    # Execute exactly as `python3 -m paper_trader.ml.<name>`: rebuild argv so
    # the child's argparse (which hard-codes prog=) and exit code are
    # untouched, then hand off via runpy. The child ends in
    # `raise SystemExit(_cli())` — that SystemExit propagates straight through
    # this function and out of the process, preserving its $? for shell gating.
    sys.argv = [f"python3 -m {_PKG}.{name}", *args[1:]]
    runpy.run_module(f"{_PKG}.{name}", run_name="__main__", alter_sys=True)
    return 0  # only reached if the child neither raised SystemExit nor exited


if __name__ == "__main__":
    raise SystemExit(_cli())
