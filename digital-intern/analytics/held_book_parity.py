"""Held-book parity audit — three prompts, one held set.

Why this exists (anti-drift lens): three independent Claude prompts each
enumerate the analyst's held universe so the LLM can reason about which
articles touch the open book:

  * ``watchers.urgency_scorer.SCORE_PROMPT``  — Sonnet 4.6 urgency classifier
    (slot: ``{portfolio_tickers}``, helper: ``_portfolio_ticker_line()``)
  * ``watchers.alert_agent.ALERT_PROMPT``     — Sonnet 4.6 Bloomberg BREAKING
    formatter (slot: ``{held_book}``, helper: ``_held_book_phrase()``)
  * ``analysis.claude_analyst.SYSTEM_PROMPT`` — Opus 4.7 heartbeat briefing
    (slot: ``{held_book}``, helper: ``_held_book_phrase()``)

Each helper was independently introduced when the prompt-side held-book
literal silently drifted behind ``config/portfolio.json`` (the 2026-05-23
audit found GOOG / COHR / NVDL held but absent from every literal). All
three now interpolate the same SSOT (``ml.features.LIVE_PORTFOLIO_TICKERS``,
with claude_analyst additionally unioning the canonical core
``_BOOK_TICKERS``), so the three sets MUST agree on what counts as held —
otherwise Sonnet's urgency score, Sonnet's alert framing, and Opus's
briefing weighting would draw the held-book boundary differently and an
analyst's PORTFOLIO commentary could be inconsistent across the same wire.

This audit is the operator-facing cross-prompt parity check + drift
detector. It is also the structural counterpart to the per-prompt
regression guards (``test_urgency_portfolio_prompt`` /
``test_alert_held_book_prompt`` / ``test_briefing_held_book_prompt``): each
of those tests asserts the corresponding prompt sees the SSOT; this audit
asserts the three see the SAME thing.

Pure read-side, no DB, no LLM — composes the three prompt helpers verbatim.
Safe to call from a dashboard endpoint, a CI gate, or a one-shot CLI.

CLI::

    python3 -m analytics.held_book_parity
    python3 -m analytics.held_book_parity --json
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Iterable


def _split_slash(line: str) -> list[str]:
    return [t.strip() for t in line.split("/") if t.strip()]


def _split_comma(line: str) -> list[str]:
    return [t.strip() for t in line.split(",") if t.strip()]


def _collect() -> dict[str, set[str]]:
    """Return {prompt_name: held_set} for the three prompt enumeration helpers.

    Each helper produces a human-readable enumeration string (slash- or
    comma-separated) — we parse them back to sets here so set algebra works.
    Import the helpers lazily so a single broken module can still surface a
    partial parity report (e.g. ``urgency_scorer`` import failing should not
    blank out the alert+briefing parity).
    """
    out: dict[str, set[str]] = {}
    try:
        from watchers.urgency_scorer import _portfolio_ticker_line
        out["urgency_scorer"] = set(_split_comma(_portfolio_ticker_line()))
    except Exception as e:  # pragma: no cover - defensive
        out["urgency_scorer"] = set()
        out["_urgency_error"] = {str(e)[:200]}
    try:
        from watchers.alert_agent import _held_book_phrase as _alert_phrase
        out["alert_agent"] = set(_split_slash(_alert_phrase()))
    except Exception as e:  # pragma: no cover - defensive
        out["alert_agent"] = set()
        out["_alert_error"] = {str(e)[:200]}
    try:
        from analysis.claude_analyst import _held_book_phrase as _briefing_phrase
        out["claude_analyst"] = set(_split_comma(_briefing_phrase()))
    except Exception as e:  # pragma: no cover - defensive
        out["claude_analyst"] = set()
        out["_briefing_error"] = {str(e)[:200]}
    return out


def _ssot_set() -> set[str]:
    """The expected canonical held set.

    Equals ``ml.features.LIVE_PORTFOLIO_TICKERS`` for the alert + urgency
    prompts. The briefing prompt is a SUPERSET (it additionally enumerates the
    static ``_BOOK_TICKERS`` core that mirrors ``daemon.PORTFOLIO_TICKERS``);
    every member of LIVE_PORTFOLIO_TICKERS must still appear in it.
    """
    from ml.features import LIVE_PORTFOLIO_TICKERS
    return set(LIVE_PORTFOLIO_TICKERS)


def audit() -> dict:
    """Return a structured audit report.

    Schema::

        {
            "ssot_size": int,
            "prompts": {
                "<prompt_name>": {
                    "size": int,
                    "missing_from_prompt": [tickers in SSOT not in prompt],
                    "extra_in_prompt": [tickers in prompt not in SSOT],
                    "matches_ssot": bool,
                },
                ...
            },
            "pairwise_diffs": {
                "<a>_vs_<b>": {
                    "only_in_a": [...],
                    "only_in_b": [...],
                },
                ...
            },
            "verdict": "OK" | "DRIFT",
        }

    Pairwise diffs are between the prompts themselves — what one knows the
    other doesn't. Verdict is OK iff every prompt is a superset of the SSOT
    and pairwise diffs match the expected briefing-is-superset shape.
    """
    ssot = _ssot_set()
    collected = _collect()
    # Filter out error sentinel keys from set-comparison.
    prompts = {
        k: v for k, v in collected.items()
        if not k.startswith("_")
    }

    per_prompt: dict[str, dict] = {}
    for name, held in prompts.items():
        missing = sorted(ssot - held)
        extra = sorted(held - ssot)
        per_prompt[name] = {
            "size": len(held),
            "missing_from_prompt": missing,
            "extra_in_prompt": extra,
            "matches_ssot": (not missing and not extra),
        }

    pair_keys = sorted(prompts.keys())
    pairwise: dict[str, dict] = {}
    for i, a in enumerate(pair_keys):
        for b in pair_keys[i + 1:]:
            sa, sb = prompts[a], prompts[b]
            pairwise[f"{a}_vs_{b}"] = {
                "only_in_a": sorted(sa - sb),
                "only_in_b": sorted(sb - sa),
            }

    # OK iff every prompt is a superset of the SSOT (extras are only allowed
    # for briefing, which adds the static core). Any prompt MISSING a SSOT
    # member is drift — the bug class this audit exists to detect.
    any_missing = any(p["missing_from_prompt"] for p in per_prompt.values())
    # Non-briefing extras are also a drift signal — alert/urgency MUST equal
    # the SSOT exactly (no static core to merge in).
    non_briefing_extras = any(
        per_prompt[name]["extra_in_prompt"]
        for name in per_prompt
        if name != "claude_analyst"
    )
    verdict = "DRIFT" if (any_missing or non_briefing_extras) else "OK"

    return {
        "ssot_size": len(ssot),
        "prompts": per_prompt,
        "pairwise_diffs": pairwise,
        "verdict": verdict,
    }


def _format_report(report: dict) -> str:
    lines = [
        "Held-book parity audit",
        "=" * 50,
        f"SSOT size: {report['ssot_size']}",
        f"Verdict:   {report['verdict']}",
        "",
        "Per-prompt parity:",
    ]
    for name, info in report["prompts"].items():
        mark = "OK " if info["matches_ssot"] else "!! "
        lines.append(
            f"  {mark}{name:18s} size={info['size']} "
            f"missing={len(info['missing_from_prompt'])} "
            f"extra={len(info['extra_in_prompt'])}"
        )
        if info["missing_from_prompt"]:
            lines.append(f"      missing: {', '.join(info['missing_from_prompt'])}")
        if info["extra_in_prompt"]:
            lines.append(f"      extra:   {', '.join(info['extra_in_prompt'])}")
    lines.append("")
    lines.append("Pairwise diffs:")
    for pair, diff in report["pairwise_diffs"].items():
        a_only = diff["only_in_a"]
        b_only = diff["only_in_b"]
        if not a_only and not b_only:
            lines.append(f"  OK  {pair}: identical")
            continue
        lines.append(f"  ?? {pair}: a-{len(a_only)} b-{len(b_only)}")
        if a_only:
            lines.append(f"      only_in_a: {', '.join(a_only)}")
        if b_only:
            lines.append(f"      only_in_b: {', '.join(b_only)}")
    return "\n".join(lines)


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true",
                    help="emit the audit report as JSON instead of text")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 when the verdict is DRIFT (suitable for CI gates)")
    args = ap.parse_args(list(argv) if argv is not None else None)

    report = audit()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(_format_report(report))
    if args.strict and report["verdict"] != "OK":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
