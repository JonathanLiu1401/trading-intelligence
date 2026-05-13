#!/bin/bash
# Hourly health check for digital-intern service
# Sends Discord alert only when issues are found

WEBHOOK_URL=$(grep DISCORD_WEBHOOK_URL /home/zeph/digital-intern/.env | cut -d= -f2-)
LOG="/home/zeph/digital-intern/data/daemon.log"

send_discord() {
    if [ -n "$WEBHOOK_URL" ]; then
        curl -s -X POST "$WEBHOOK_URL" \
            -H "Content-Type: application/json" \
            -d "{\"content\": \"$1\"}" > /dev/null
    fi
}

# Check service status
if ! systemctl is-active --quiet digital-intern; then
    send_discord "🚨 digital-intern is DOWN! Attempting restart..."
    systemctl restart digital-intern 2>/dev/null || \
        send_discord "❌ Restart failed — manual intervention needed."
    exit 1
fi

# Check for recent crashes/errors in last hour
ERRORS=$(journalctl -u digital-intern --since "1 hour ago" --no-pager 2>/dev/null \
    | grep -cE "died|locked|Traceback|CRITICAL" || echo 0)

# Get stats from last log line
STATS=$(grep "\[stats\]" "$LOG" 2>/dev/null | tail -1)
UNSCORED=$(echo "$STATS" | grep -oP 'unscored=\K[0-9]+' || echo "?")
TOTAL=$(echo "$STATS" | grep -oP 'total=\K[0-9]+' || echo "?")
URGENT=$(echo "$STATS" | grep -oP 'urgent=\K[0-9]+' || echo "?")

# Alert on errors or very high unscored backlog
if [ "$ERRORS" -gt 5 ] 2>/dev/null; then
    send_discord "⚠️ digital-intern: $ERRORS errors in last hour | total=$TOTAL unscored=$UNSCORED urgent=$URGENT"
elif [ "$UNSCORED" -gt 5000 ] 2>/dev/null; then
    send_discord "⚠️ digital-intern: scorer backlog high — $UNSCORED unscored articles"
fi

# Log health check result
echo "[$(date '+%Y-%m-%d %H:%M')] OK: total=$TOTAL unscored=$UNSCORED urgent=$URGENT errors=$ERRORS" \
    >> /home/zeph/digital-intern/logs/healthcheck.log

exit 0
