#!/usr/bin/env python3
"""CLI entrypoint for the experimental Grok multi-agent ArticleNet discovery collector."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from collectors.grok_multiagent_discovery import run_once, summarize_trials  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Ignore host-load skip gate")
    parser.add_argument("--trials", type=int, default=1, help="Number of discovery cycles to run")
    parser.add_argument("--summarize", action="store_true", help="Print trial summary and exit")
    args = parser.parse_args()

    if args.summarize:
        print(json.dumps(summarize_trials(), indent=2))
        return 0

    results = []
    for i in range(max(1, args.trials)):
        print(f"[runner] trial {i+1}/{args.trials}")
        results.append(run_once(force=args.force))
    print(json.dumps({"results": results, "summary": summarize_trials()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
