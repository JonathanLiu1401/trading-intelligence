#!/bin/bash
cd /home/zeph/digital-intern
source .env 2>/dev/null || true
python3 main.py >> logs/heartbeat.log 2>&1
