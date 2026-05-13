#!/bin/bash
# Digital Intern — hourly local health monitor
# Checks systemd service, error rates, scorer backlog, heartbeat watchdog.
# Sends Discord alerts only when something is wrong.

set -euo pipefail

WEBHOOK_URL=$(grep DISCORD_WEBHOOK_URL /home/zeph/digital-intern/.env 2>/dev/null | cut -d= -f2-)
PLAIN_LOG="/home/zeph/digital-intern/data/daemon.log"
STRUCT_LOG="/home/zeph/digital-intern/logs/structured.jsonl"
HC_LOG="/home/zeph/digital-intern/logs/healthcheck.log"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M')

mkdir -p /home/zeph/digital-intern/logs

send_discord() {
    [[ -z "${WEBHOOK_URL:-}" ]] && return 0
    local msg="${1//\"/\\\"}"
    curl -s -X POST "$WEBHOOK_URL" \
         -H "Content-Type: application/json" \
         -d "{\"content\": \"$msg\"}" > /dev/null 2>&1 || true
}

# ── 1. Service health ────────────────────────────────────────────────────────
if ! systemctl is-active --quiet digital-intern 2>/dev/null; then
    STATUS=$(systemctl is-active digital-intern 2>/dev/null || echo "unknown")
    send_discord "🚨 **Digital Intern DOWN** (\`$STATUS\`) at $TIMESTAMP — attempting restart..."
    sudo systemctl restart digital-intern 2>/dev/null || true
    sleep 5
    if systemctl is-active --quiet digital-intern 2>/dev/null; then
        send_discord "✅ **Digital Intern** restarted successfully at $TIMESTAMP"
    else
        send_discord "❌ **Digital Intern** restart FAILED at $TIMESTAMP — manual intervention needed"
        echo "[$TIMESTAMP] CRITICAL: service restart failed" >> "$HC_LOG"
        exit 1
    fi
fi

# ── 2. Error rate in last hour (from journalctl) ─────────────────────────────
ERRORS=$(journalctl -u digital-intern --since "1 hour ago" --no-pager 2>/dev/null \
         | grep -cE "died|locked|Traceback|CRITICAL|ERROR" 2>/dev/null || echo 0)

# ── 3. DB stats ──────────────────────────────────────────────────────────────
STATS_LINE=$(grep "\[stats\]" "$PLAIN_LOG" 2>/dev/null | tail -1 || echo "")
TOTAL=$(echo   "$STATS_LINE" | grep -oP 'total=\K[0-9]+'   2>/dev/null || echo "?")
UNSCORED=$(echo "$STATS_LINE" | grep -oP 'unscored=\K[0-9]+' 2>/dev/null || echo "?")
URGENT=$(echo  "$STATS_LINE" | grep -oP 'urgent=\K[0-9]+'   2>/dev/null || echo "?")

# ── 4. Heartbeat watchdog ────────────────────────────────────────────────────
HB_AGE_H="?"
if [[ -f "$STRUCT_LOG" ]]; then
    LAST_HB_TS=$(grep '"heartbeat"' "$STRUCT_LOG" 2>/dev/null | grep '"sent"' | tail -1 \
                 | python3 -c "import sys,json; r=json.loads(sys.stdin.read().strip() or '{}'); print(r.get('ts',''))" 2>/dev/null || echo "")
    if [[ -n "$LAST_HB_TS" ]]; then
        HB_AGE_H=$(python3 -c "
from datetime import datetime, timezone
dt = datetime.fromisoformat('${LAST_HB_TS}'.replace('Z','+00:00'))
print(f'{(datetime.now(timezone.utc)-dt).total_seconds()/3600:.1f}')
" 2>/dev/null || echo "?")
        if python3 -c "exit(0 if float('${HB_AGE_H}') > 6 else 1)" 2>/dev/null; then
            send_discord "⚠️ **Digital Intern**: no heartbeat briefing in ${HB_AGE_H}h — scorer may be stuck"
        fi
    fi
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

