#!/usr/bin/env bash
# Bulk historical article collection — launches all three sweeps in parallel.
# Runs from the digital-intern root directory.
# Usage: bash scripts/bulk_collect.sh [start_year]
set -e
cd "$(dirname "$0")/.."

START_YEAR=${1:-2013}
DB_PATH="data/articles.db"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

echo "=== BULK COLLECTION START $(date) ==="
python3 -c "
import sqlite3
n = sqlite3.connect('$DB_PATH').execute('SELECT COUNT(*) FROM articles').fetchone()[0]
print(f'Starting count: {n:,} articles')
"

# Launch all three sweeps simultaneously
echo "[runner] Starting GDELT historical sweep (2013 → now, 40K tasks)..."
python3 scripts/gdelt_historical_sweep.py $START_YEAR 1 \
    > "$LOG_DIR/gdelt_sweep.log" 2>&1 &
GDELT_PID=$!

echo "[runner] Starting Finnhub historical sweep (2018 → now)..."
python3 scripts/finnhub_historical_news.py \
    > "$LOG_DIR/finnhub_sweep.log" 2>&1 &
FINNHUB_PID=$!

echo "[runner] Starting SEC EDGAR 8-K bulk import (1994 → now)..."
python3 scripts/sec_edgar_bulk.py 1994 \
    > "$LOG_DIR/edgar_sweep.log" 2>&1 &
EDGAR_PID=$!

echo "[runner] All 3 sweeps running:"
echo "  GDELT:    PID $GDELT_PID  (log: $LOG_DIR/gdelt_sweep.log)"
echo "  Finnhub:  PID $FINNHUB_PID  (log: $LOG_DIR/finnhub_sweep.log)"
echo "  EDGAR:    PID $EDGAR_PID  (log: $LOG_DIR/edgar_sweep.log)"
echo ""
echo "[runner] Progress monitor (Ctrl-C to stop monitoring, sweeps keep running):"

# Monitor loop
while true; do
    sleep 30
    COUNT=$(python3 -c "
import sqlite3
try:
    n = sqlite3.connect('$DB_PATH').execute('SELECT COUNT(*) FROM articles').fetchone()[0]
    print(f'{n:,}')
except: print('?')
" 2>/dev/null)

    GDELT_RUNNING=$(kill -0 $GDELT_PID 2>/dev/null && echo "running" || echo "done")
    FINNHUB_RUNNING=$(kill -0 $FINNHUB_PID 2>/dev/null && echo "running" || echo "done")
    EDGAR_RUNNING=$(kill -0 $EDGAR_PID 2>/dev/null && echo "running" || echo "done")

    echo "[$(date '+%H:%M:%S')] Articles: $COUNT | GDELT=$GDELT_RUNNING | Finnhub=$FINNHUB_RUNNING | EDGAR=$EDGAR_RUNNING"

    GDELT_LINE=$(tail -1 "$LOG_DIR/gdelt_sweep.log" 2>/dev/null || true)
    [ -n "$GDELT_LINE" ] && echo "  GDELT: $GDELT_LINE"

    # All done?
    if ! kill -0 $GDELT_PID 2>/dev/null && ! kill -0 $FINNHUB_PID 2>/dev/null && ! kill -0 $EDGAR_PID 2>/dev/null; then
        echo ""
        echo "=== ALL SWEEPS COMPLETE $(date) ==="
        python3 -c "
import sqlite3
n = sqlite3.connect('$DB_PATH').execute('SELECT COUNT(*) FROM articles').fetchone()[0]
print(f'Final count: {n:,} articles')
"
        break
    fi
done
