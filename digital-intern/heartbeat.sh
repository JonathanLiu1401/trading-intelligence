#!/bin/bash
# Liveness ping for the digital-intern daemon.
#
# Previous version invoked `python3 main.py`, which is the legacy one-shot
# pipeline (collect → filter → analyze → notify). Running that on a cron meant
# every "heartbeat" spawned a full collection + Opus briefing cycle in parallel
# with the daemon — a self-induced DoS that's documented in CLAUDE.md's failure
# modes as the cause of duplicate-daemon spawns and high CPU.
#
# What a heartbeat should actually do: probe the dashboard's /healthz endpoint
# and log the result. No collector calls, no LLM calls, no DB writes.

cd /home/zeph/digital-intern
HEALTH_URL="${HEALTH_URL:-http://127.0.0.1:8080/healthz}"
mkdir -p logs
ts=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
if out=$(curl -fsS --max-time 5 "$HEALTH_URL" 2>&1); then
    echo "$ts ok $out" >> logs/heartbeat.log
    exit 0
else
    echo "$ts FAIL ($?) $out" >> logs/heartbeat.log
    exit 1
fi
