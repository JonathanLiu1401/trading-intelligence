#!/bin/bash
set -euo pipefail

cd /Users/jonathan/trading-intelligence/digital-intern

export HOME=/Users/jonathan
export PYTHONPATH=.
export DASHBOARD_PORT="${DASHBOARD_PORT:-8765}"

printf '%s starting unified dashboard pid=%s port=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$" "$DASHBOARD_PORT" >&2

exec /Users/jonathan/trading-intelligence/.venv/bin/python \
  -m uvicorn dashboard.server:app \
  --host 127.0.0.1 \
  --port "$DASHBOARD_PORT"
