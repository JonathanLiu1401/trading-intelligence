#!/bin/bash
# Digital Intern — hourly local health monitor
# Checks systemd service, error rates, scorer backlog, heartbeat watchdog.
# Sends Discord alerts only when something is wrong.

set -euo pipefail

WEBHOOK_URL=$(grep DISCORD_WEBHOOK_URL /home/zeph/trading-intelligence/digital-intern/.env 2>/dev/null | cut -d= -f2-)
PLAIN_LOG="/home/zeph/trading-intelligence/digital-intern/data/daemon.log"
STRUCT_LOG="/home/zeph/trading-intelligence/digital-intern/logs/structured.jsonl"
HC_LOG="/home/zeph/trading-intelligence/digital-intern/logs/healthcheck.log"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M')

mkdir -p /home/zeph/trading-intelligence/digital-intern/logs

send_discord() {
    [[ -z "${WEBHOOK_URL:-}" ]] && return 0
    local msg="${1//\"/\\\"}"
    curl -s -X POST "$WEBHOOK_URL" \
         -H "Content-Type: application/json" \
         -d "{\"content\": \"$msg\"}" > /dev/null 2>&1 || true
}

# ── 1. Service health ────────────────────────────────────────────────────────
# Brief grace re-check: avoids false-positive restarts during transient
# "activating"/"deactivating" states (e.g. systemd Restart= flapping after
# the singleton lock rejects a stray duplicate instance).
if ! systemctl is-active --quiet digital-intern 2>/dev/null; then
    sleep 8
fi
if ! systemctl is-active --quiet digital-intern 2>/dev/null; then
    STATUS=$(systemctl is-active digital-intern 2>/dev/null || echo "unknown")
    send_discord "🚨 **Digital Intern DOWN** (\`$STATUS\`) at $TIMESTAMP — attempting restart..."
    RESTART_ERR=$(sudo -n systemctl restart digital-intern 2>&1 >/dev/null || true)
    sleep 5
    if systemctl is-active --quiet digital-intern 2>/dev/null; then
        send_discord "✅ **Digital Intern** restarted successfully at $TIMESTAMP"
    else
        # Surface restart-command stderr so we can tell sudo/password failure
        # apart from a real service crash.
        ERR_SNIPPET=${RESTART_ERR:0:200}
        send_discord "❌ **Digital Intern** restart FAILED at $TIMESTAMP (\`${ERR_SNIPPET:-no stderr}\`) — manual intervention needed"
        echo "[$TIMESTAMP] CRITICAL: service restart failed err='${ERR_SNIPPET}'" >> "$HC_LOG"
        exit 1
    fi
fi

# ── 2. Error rate in last hour (from journalctl) ─────────────────────────────
_ERR_RAW=$(journalctl -u digital-intern --since "1 hour ago" --no-pager 2>/dev/null \
          | grep -cE "died|locked|Traceback|CRITICAL|ERROR" 2>/dev/null || true)
# grep -c prints "0" then exits 1 on no-match; `|| echo 0` was appending a 2nd "0\n".
ERRORS=$(printf '%s' "$_ERR_RAW" | head -1 | tr -dc '0-9')
ERRORS=${ERRORS:-0}

# ── 3. DB stats ──────────────────────────────────────────────────────────────
# Prefer structured JSON log (current), fall back to legacy plaintext daemon.log
# `-a` forces text mode: structured.jsonl can contain stray NUL bytes (from
# raw RSS/article payloads logged at DEBUG), which makes grep silently
# truncate to matches BEFORE the NUL — the symptom was healthcheck.log
# frozen at a stats snapshot from when the first NUL appeared.
# Also filter out worker_alive heartbeat lines ("[stats] alive ...") so we
# only land on a real stats line ("[stats] total=... urgent=... unscored=...").
STATS_LINE=$(grep -a '"\[stats\] total=' "$STRUCT_LOG" 2>/dev/null | tail -1 || echo "")
if [[ -z "$STATS_LINE" ]]; then
    STATS_LINE=$(grep -a '\[stats\] total=' "$PLAIN_LOG" 2>/dev/null | tail -1 || echo "")
fi
TOTAL=$(echo   "$STATS_LINE" | grep -oP 'total=\K[0-9]+'   2>/dev/null || echo "?")
UNSCORED=$(echo "$STATS_LINE" | grep -oP 'unscored=\K[0-9]+' 2>/dev/null || echo "?")
URGENT=$(echo  "$STATS_LINE" | grep -oP 'urgent=\K[0-9]+'   2>/dev/null || echo "?")

# ── 4. Heartbeat watchdog ────────────────────────────────────────────────────
# The "[heartbeat] sent" briefing event is infrequent (≈hourly), so a single
# size-rotation of structured.jsonl (RotatingFileHandler, 10MB) can leave the
# current file with no marker until the next briefing fires. Scanning only the
# current file then blanked HB_AGE_H to "?" AND silently disabled the >6h
# stale-scorer alert below (it was gated on a non-empty timestamp). Scan the
# current file plus the most recent rotated backup (.1) and take the latest ts.
HB_AGE_H=$(python3 - "$STRUCT_LOG" "${STRUCT_LOG}.1" <<'PY' 2>/dev/null || echo "?"
import sys, json
from datetime import datetime, timezone
latest = None
for path in sys.argv[1:]:
    try:
        with open(path, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if "[heartbeat] sent" not in line:
                    continue
                try:
                    ts = json.loads(line).get("ts", "")
                except Exception:
                    continue
                if ts and (latest is None or ts > latest):
                    latest = ts
    except FileNotFoundError:
        continue
if not latest:
    print("?")
else:
    dt = datetime.fromisoformat(latest.replace("Z", "+00:00"))
    print(f"{(datetime.now(timezone.utc) - dt).total_seconds() / 3600:.1f}")
PY
)
HB_AGE_H=${HB_AGE_H:-?}
if [[ "$HB_AGE_H" != "?" ]] && python3 -c "exit(0 if float('${HB_AGE_H}') > 6 else 1)" 2>/dev/null; then
    send_discord "⚠️ **Digital Intern**: no heartbeat briefing in ${HB_AGE_H}h — scorer may be stuck"
fi

# ── 5. Alert on high error count or scorer backlog ───────────────────────────
if [[ "$ERRORS" -gt 10 ]] 2>/dev/null; then
    RECENT=$(journalctl -u digital-intern --since "1 hour ago" --no-pager 2>/dev/null \
             | grep -E "died|ERROR|CRITICAL" | tail -3 | tr '\n' ' ')
    send_discord "⚠️ **Digital Intern** — $ERRORS errors in last hour at $TIMESTAMP\n\`\`\`${RECENT:0:400}\`\`\`"
fi

if [[ "$UNSCORED" =~ ^[0-9]+$ ]] && [[ "$UNSCORED" -gt 8000 ]]; then
    send_discord "⚠️ **Digital Intern** — scorer backlog: $UNSCORED unscored articles (total=$TOTAL)"
fi

# ── 6. Log summary ───────────────────────────────────────────────────────────
echo "[$TIMESTAMP] OK service=active total=$TOTAL unscored=$UNSCORED urgent=$URGENT errors=$ERRORS hb=${HB_AGE_H}h" >> "$HC_LOG"
echo "[$TIMESTAMP] Healthcheck complete — total=$TOTAL unscored=$UNSCORED errors=$ERRORS"

