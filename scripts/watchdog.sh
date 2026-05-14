#!/bin/bash
# Trading Stack Watchdog — ensures all services run 24/7
set -euo pipefail

WEBHOOK=$(grep DISCORD_WEBHOOK_URL /home/zeph/digital-intern/.env 2>/dev/null | cut -d= -f2-)
TS=$(date '+%Y-%m-%d %H:%M')

alert() {
    [[ -z "${WEBHOOK:-}" ]] && return
    curl -s -X POST "$WEBHOOK" -H "Content-Type: application/json" \
         -d "{\"content\": \"$1\"}" >/dev/null 2>&1 || true
}

# 1. digital-intern (root systemd — use nopasswd restart)
if ! sudo -n systemctl is-active --quiet digital-intern 2>/dev/null; then
    alert "🔄 Watchdog: restarting digital-intern at $TS"
    sudo -n systemctl restart digital-intern 2>/dev/null || true
fi

# 2. paper-trader (user systemd)
if ! systemctl --user is-active --quiet paper-trader 2>/dev/null; then
    alert "🔄 Watchdog: restarting paper-trader at $TS"
    systemctl --user restart paper-trader 2>/dev/null || true
fi

# 3. Digital Intern FastAPI (port 8765) — managed by its own unit
if ! systemctl --user is-active --quiet openclaw-gateway 2>/dev/null; then
    systemctl --user restart openclaw-gateway 2>/dev/null || true
fi

# 4. Check port 8090 — paper trader dashboard
if ! ss -tlnp 2>/dev/null | grep -q ':8090 '; then
    alert "⚠️ Watchdog: port 8090 down, restarting paper-trader at $TS"
    systemctl --user restart paper-trader 2>/dev/null || true
fi

# 5. Check port 8080 — digital-intern Flask dashboard
# (Now runs inside the daemon, will restart with it)
if ! ss -tlnp 2>/dev/null | grep -q ':8080 '; then
    sudo -n systemctl restart digital-intern 2>/dev/null || true
fi

echo "[$TS] Watchdog OK"
