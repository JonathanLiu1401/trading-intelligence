# OpenClaw Local Trading Runbook

Last updated: 2026-06-08 local time.

This file is for future Codex/OpenClaw agents working on Jonathan's local
OpenClaw plus `trading-intelligence` setup on this Mac. It records the current
state, what was changed, what was verified, and the resource limits that matter.

## 2026-06-08 USB Runtime Recovery And Safe Startup

Recovered USB data is backed up in the private GitHub repo:

```text
https://github.com/JonathanLiu1401/openclaw-usb-recovery
local path: /Users/jonathan/openclaw-usb-recovery
```

The trading stack was populated from:

```text
/Users/jonathan/openclaw-usb-recovery/recovered
/Users/jonathan/openclaw-usb-recovery/recovered_sensitive
```

Do not copy recovered source code over the current `trading-intelligence`
working tree. Only runtime data/config/model artifacts were merged.

Current recovered runtime state:

```text
digital-intern/data/articles.db: 2,352,913 articles
digital-intern/data/seen_articles.db: recovered USB copy
digital-intern/data/source_health.db: recovered USB copy
digital-intern/data/paper_trader_signals.db: recovered USB copy
digital-intern/ml and digital-intern/db/ml*: recovered ArticleNet artifacts
paper-trader/backtest.db: 273 MB recovered USB copy
paper-trader/data: 733 MB recovered USB copy
paper-trader/.env: restored from recovered_sensitive and chmod 600
```

Runtime merge manifest:

```text
/Users/jonathan/trading-intelligence/run/usb-runtime-merge-20260608T074154Z.txt
```

The old tracked `paper-trader/logs` path was a broken Linux symlink to:

```text
/media/zeph/projects/paper-trader/logs
```

It was replaced locally with a real Mac directory so launchd services can
write logs. Keep `paper-trader/logs/` ignored.

Mac-safe launchd steady state after recovery:

```text
com.jonathan.trading-intelligence.unified-dashboard
  127.0.0.1:8765
  uvicorn dashboard.server:app

com.jonathan.trading-intelligence.digital-intern
  127.0.0.1:8080
  dashboard.web_server.run_server(None)
  standalone ArticleNet web/API only; does not start daemon collectors/trainers

com.jonathan.trading-intelligence.paper-trader
  127.0.0.1:8090
  paper_trader.dashboard.run()
  dashboard/API only; does not start runner.py live decision loop
  PAPER_TRADER_DASHBOARD_PREWARM=0
```

Important: do not revert these LaunchAgents to `digital-intern/daemon.py` or
`paper-trader/runner.py` on this Mac unless Jonathan explicitly asks for live
collector/trader loops. The safe default is dashboard/API plus recovered
ArticleNet data, not high-fanout live work.

Endpoint verification after recovery:

```text
http://127.0.0.1:8765/                         200
http://127.0.0.1:8765/api/health               200
http://127.0.0.1:8765/api/command-center       200
http://127.0.0.1:8765/intern/api/articles      200
http://127.0.0.1:8765/trader/api/healthz       200
http://127.0.0.1:8765/trader/api/portfolio     200
http://127.0.0.1:8080/api/stats                200
http://127.0.0.1:8080/api/articles             200
http://127.0.0.1:8090/api/healthz              200
http://127.0.0.1:8090/api/backtests            200
```

`digital-intern/dashboard/web_server.py` was changed so `/api/articles` uses
the existing `idx_first_seen` recency index. The recovered 2.35M-row database
made the old score-sort query scan the full table and block the unified
dashboard.

Light Playwright verification passed locally with cached Playwright and system
Chrome. Main routes rendered without application errors:

```text
/, /intern/, /intern/chat, /trader/, /trader/backtests,
/ops/, /system/, /strategy-lab, /journal, /personas, /tape, /pulse
```

Tailscale state on 2026-06-08:

```text
/Applications/Tailscale.app/Contents/MacOS/Tailscale status -> Logged out.
/Applications/Tailscale.app/Contents/MacOS/Tailscale ip -4 -> NeedsLogin
/Applications/Tailscale.app/Contents/MacOS/Tailscale serve status -> No serve config
```

`tailscale up --timeout=10s` produced a browser login URL and timed out waiting
for auth. Do not repeatedly force login/logout; that risks creating more
tailnet devices. Once Jonathan authenticates this Mac in the Tailscale app or
with the CLI URL, verify:

```bash
/Applications/Tailscale.app/Contents/MacOS/Tailscale status
/Applications/Tailscale.app/Contents/MacOS/Tailscale ip -4
curl -I http://your-macbook-3.tailaa3a85.ts.net:8765/
```

## Critical Safety Constraints

This Mac rebooted during the setup session while multiple expensive operations
had been running recently. Treat it as resource-constrained.

Do not run:

- `paper-trader/run_continuous_backtests.py`
- systemd-style continuous backtest services
- continuous digital-intern trainer on this Mac
- full repo test suites by default
- source builds such as `llvmlite`, LLVM, or large native dependency compiles
- broad `pip install --upgrade ...` operations
- blind `git checkout`, `git reset --hard`, or cleanup over `.openclaw`

Prefer:

- one command at a time for service startup
- short health probes
- focused tests only
- `DIGITAL_INTERN_WORKERS=web_server` for steady state after ArticleNet data
  is already populated
- `DIGITAL_INTERN_CONTINUOUS_TRAINER=0`

Before starting anything heavy:

```bash
uptime
vm_stat | sed -n '1,12p'
ps -ax -o pid,ppid,state,%cpu,%mem,etime,command | sort -k4 -nr | sed -n '1,25p'
```

If the load average is still elevated after reboot, wait.

## Local Repositories

OpenClaw runtime/config:

```text
/Users/jonathan/.openclaw
```

Trading intelligence:

```text
/Users/jonathan/trading-intelligence
```

Observed trading repo state:

```text
branch: master
remote: https://github.com/JonathanLiu1401/trading-intelligence.git
HEAD: a7870d6 fix: add BLOCKED to compact report priority to prevent it being hidden
```

Observed OpenClaw repo state:

```text
branch: main
main HEAD: ee00ae8ea sync latest changes
origin/zeph HEAD: 1eaea719 fix(openclaw): relink whatsapp and improve discord progress
```

The user originally linked `/tree/zephi`, but the local remote branch found by
`git fetch --all --prune` is `origin/zeph`. Do not assume `zephi` exists.

The `.openclaw` working tree is a live runtime directory. It contains logs,
sessions, credentials, generated node modules, SQLite files, media, and runtime
state. Do not blindly merge all of `origin/zeph` into it. Pull only relevant
configuration or code changes after inspecting diffs.

## OpenClaw Gateway

Gateway LaunchAgent label:

```text
ai.openclaw.gateway
```

Gateway listens on:

```text
127.0.0.1:18789
```

Check it:

```bash
lsof -nP -iTCP:18789 -sTCP:LISTEN
openclaw config validate --json
```

Restart it only if needed:

```bash
launchctl kickstart -k "gui/$(id -u)/ai.openclaw.gateway"
```

Gateway logs:

```text
/Users/jonathan/Library/Logs/openclaw/gateway.log
/tmp/openclaw/openclaw-2026-06-07.log
```

The gateway has shown:

```text
agent model: codex/gpt-5.5 (thinking=xhigh, fast=on)
```

## Model Configuration

Primary model:

```text
codex/gpt-5.5
```

Fallback chain:

```text
xai/grok-4.3
openai/gpt-5.5
codex/gpt-5.5-pro
openai/gpt-5.5-pro
anthropic/claude-sonnet-4-6
```

xAI/Grok is the backup model, not the Discord voice model override.

Do not dump `openclaw models status --json` into chat or docs without checking
for auth/profile leakage. It may include redacted credential labels, but treat
all auth output as sensitive.

## Discord Details

Guild:

```text
Name: Your father
ID: 1153698669040783404
```

Text channel used for proof/status:

```text
Name: chat
ID: 1153698670227750954
OpenClaw target form: channel:1153698670227750954
```

Voice channel:

```text
ID: 1153698670227750958
```

User/Lark ID for completion ping:

```text
454961974048980992
```

Bot:

```text
ID: 1487726935827021997
Username: OpenClawAgent
Server nickname: jarvis
```

Send a direct proof/status message without invoking an agent:

```bash
openclaw message send \
  --channel discord \
  --target channel:1153698670227750954 \
  --message 'Status text here'
```

Completion ping format:

```bash
openclaw message send \
  --channel discord \
  --target channel:1153698670227750954 \
  --message '<@454961974048980992> OpenClaw/trading setup status text.'
```

## Discord Voice Configuration

The working path is the built-in OpenClaw Discord voice bridge:

```text
channels.discord.voice
```

The old standalone plugin is intentionally disabled:

```text
plugins.entries.discord-voice.enabled = false
```

`openclaw config validate --json` may warn that the disabled plugin still has
config present. That warning is expected. Do not enable that plugin unless the
user explicitly asks to revive the old extension.

Working built-in voice settings:

```text
enabled: true
mode: stt-tts
daveEncryption: true
decryptionFailureTolerance: 50
model: codex/gpt-5.5
captureSilenceGraceMs: 500
autoJoin guildId: 1153698669040783404
autoJoin channelId: 1153698670227750958
agentSession.mode: target
agentSession.target: channel:1153698670227750954
tts.provider: openai
tts.providers.openai.model: tts-1
tts.providers.openai.responseFormat: pcm48s
tts.providers.openai.speakerVoice: am_michael
```

Important voice failure that already happened:

- The voice model was changed to `xai/grok-4.20-beta-latest-non-reasoning`.
- A later voice turn failed with:

```text
Model override "xai/grok-4.20-beta-latest-non-reasoning" is not allowed for agent "main".
```

Fix was to put:

```text
channels.discord.voice.model = codex/gpt-5.5
```

Do not repeat the xAI beta override. xAI/Grok belongs in the fallback chain.

Check voice join status:

```bash
openclaw message voice status \
  --channel discord \
  --guild-id 1153698669040783404 \
  --user-id 1487726935827021997 \
  --json
```

Expected bot status:

```text
channel_id: 1153698670227750958
self_mute: false
self_deaf: false
mute: false
deaf: false
```

Voice test already observed in logs:

- User spoke: `Hello, test test, can you hear me?`
- STT succeeded through xAI/Grok STT.
- Agent replied with Codex.
- TTS playback completed into Discord.

The user disliked the Microsoft/Brian voice. The local config currently uses
OpenAI TTS with `am_michael`.

If TTS leaves a stale `ffmpeg` child, remove only that stale child after
verifying playback has completed:

```bash
ps -ax -o pid,ppid,state,etime,command | rg 'ffmpeg|openclaw logs'
kill <stale-ffmpeg-pid>
```

Use `kill -KILL` only for a confirmed stuck child that ignores normal `kill`.

## Trading Python Environment

Use:

```text
/Users/jonathan/trading-intelligence/.venv/bin/python
```

Created with:

```text
/usr/local/bin/python3.12
```

Do not use system `python3`; it was too old for this repo on this Mac.

Installed dependencies include:

- `digital-intern/requirements.txt`
- `flask`
- `anthropic`
- `pytest`
- `pandas`
- `fastapi`
- `uvicorn`
- `websockets`
- `httpx`

Known dependency tradeoff:

- ArticleNet/Torch works with `numpy==1.26.4` and `torch==2.2.2`.
- `kokoro-onnx==0.5.0` declares `numpy>=2.0.2`.
- Upgrading NumPy to 2.x produced Torch compatibility warnings and is not
  appropriate for ArticleNet stability on this setup.
- Older Kokoro versions still required NumPy 2 or pulled `llvmlite` source
  builds, which failed without LLVM dev packages.
- Kokoro is optional here because OpenClaw is handling Discord voice/TTS.

Expected `pip check` caveat:

```text
kokoro-onnx 0.5.0 has requirement numpy>=2.0.2, but you have numpy 1.26.4.
```

Do not "fix" that by upgrading NumPy unless you also validate Torch/ArticleNet.

ArticleNet direct smoke that passed:

```bash
cd /Users/jonathan/trading-intelligence/digital-intern
PYTHONPATH=. ../.venv/bin/python - <<'PY'
import numpy as np, torch
from ml.model import ArticleNet
print("article torch ok", np.__version__, torch.__version__, ArticleNet.__name__)
PY
```

Observed output:

```text
article torch ok 1.26.4 2.2.2 ArticleNet
```

## Digital Intern: ArticleNet Only On This Mac

`digital-intern/daemon.py` now supports:

```text
DIGITAL_INTERN_CONTINUOUS_TRAINER=0
DIGITAL_INTERN_WORKERS=web_server
```

Meaning:

- `web_server` exposes the ArticleNet dashboard/API on port 8080.
- `continuous_trainer` is disabled.
- High-fanout collectors, scorer backlog processing, and trainers are not left
  running in the normal Mac steady state.

This is the preferred Mac startup mode because the user specifically said:

```text
do not start continuous training; just do the article net
```

On 2026-06-07 local time, a short bounded collector/scorer burst populated
ArticleNet to 2,112 articles, then the daemon was returned to
`DIGITAL_INTERN_WORKERS=web_server` after CPU stayed high. Current dashboard
stats showed 2,112 total articles, 0 urgent, 1,465 unscored, and 6.2 MB DB
size. The unscored backlog is intentional until Jonathan asks for another
bounded scoring pass.

If future agents need to refresh ArticleNet, do it briefly and visibly with a
small worker allowlist, then return to `web_server` only. Do not start the full
daemon with every collector, and do not run the continuous trainer/backtest
loops on this Mac.

## Paper Trader

Paper trader entrypoint:

```bash
cd /Users/jonathan/trading-intelligence/paper-trader
PYTHONPATH=. ../.venv/bin/python runner.py
```

Dashboard:

```text
http://127.0.0.1:8090
```

Health endpoint:

```text
http://127.0.0.1:8090/api/healthz
```

Do not start:

```bash
python run_continuous_backtests.py
```

The README discusses continuous backtests, but this Mac should not run them.

LaunchAgent:

```text
Label: com.jonathan.trading-intelligence.paper-trader
Plist: /Users/jonathan/Library/LaunchAgents/com.jonathan.trading-intelligence.paper-trader.plist
Logs:
  /Users/jonathan/trading-intelligence/logs/paper-trader.launchd.log
  /Users/jonathan/trading-intelligence/logs/paper-trader.launchd.err.log
```

Important macOS limit:

```text
SoftResourceLimits.NumberOfFiles = 4096
HardResourceLimits.NumberOfFiles = 8192
```

This was added after the threaded Flask dashboard exhausted launchd's default
`maxfiles` soft limit of `256` during a high-fanout browser test and began
resetting connections with `OSError: [Errno 24] Too many open files`. Keep
this limit in the plist. If Paper Trader starts returning `502` through the
unified dashboard, check:

```bash
launchctl print gui/$(id -u)/com.jonathan.trading-intelligence.paper-trader | sed -n '/pid =/p;/resource limits/,+5p'
curl -i http://127.0.0.1:8090/api/portfolio
tail -n 120 /Users/jonathan/trading-intelligence/logs/paper-trader.launchd.err.log
```

## Safe Startup Procedure

Use this sequence. Do not combine it with installs, tests, or git operations.
When launching from Codex or another non-interactive shell, prefer the macOS
LaunchAgents below. This command environment can reap background children even
when `nohup` is used.

1. Check load and ports:

```bash
uptime
lsof -nP -iTCP:8080 -sTCP:LISTEN
lsof -nP -iTCP:8090 -sTCP:LISTEN
pgrep -af 'daemon.py|runner.py|run_continuous_backtests.py'
```

2. Start digital-intern in ArticleNet-only mode with launchd:

```bash
mkdir -p /Users/jonathan/trading-intelligence/run /Users/jonathan/trading-intelligence/logs
launchctl bootout gui/$(id -u) /Users/jonathan/Library/LaunchAgents/com.jonathan.trading-intelligence.digital-intern.plist 2>/dev/null || true
launchctl bootstrap gui/$(id -u) /Users/jonathan/Library/LaunchAgents/com.jonathan.trading-intelligence.digital-intern.plist
launchctl kickstart gui/$(id -u)/com.jonathan.trading-intelligence.digital-intern
```

3. Wait 10 to 20 seconds, then probe:

```bash
curl --max-time 5 -fsS http://127.0.0.1:8080/healthz
tail -n 80 /Users/jonathan/trading-intelligence/logs/digital-intern.launchd.log
tail -n 80 /Users/jonathan/trading-intelligence/logs/digital-intern.launchd.err.log
```

Look for:

```text
continuous_trainer disabled
worker allowlist active
web_server
ml_trainer
scorer
```

4. Start paper-trader with launchd:

```bash
launchctl bootout gui/$(id -u) /Users/jonathan/Library/LaunchAgents/com.jonathan.trading-intelligence.paper-trader.plist 2>/dev/null || true
launchctl bootstrap gui/$(id -u) /Users/jonathan/Library/LaunchAgents/com.jonathan.trading-intelligence.paper-trader.plist
launchctl kickstart gui/$(id -u)/com.jonathan.trading-intelligence.paper-trader
```

5. Probe:

```bash
curl --max-time 5 -fsS http://127.0.0.1:8090/api/healthz
tail -n 80 /Users/jonathan/trading-intelligence/logs/paper-trader.launchd.log
tail -n 80 /Users/jonathan/trading-intelligence/logs/paper-trader.launchd.err.log
```

6. Confirm no continuous backtests:

```bash
pgrep -af 'run_continuous_backtests.py|continuous-backtests' || true
```

## Safe Stop Procedure

```bash
launchctl bootout gui/$(id -u) /Users/jonathan/Library/LaunchAgents/com.jonathan.trading-intelligence.paper-trader.plist 2>/dev/null || true
launchctl bootout gui/$(id -u) /Users/jonathan/Library/LaunchAgents/com.jonathan.trading-intelligence.digital-intern.plist 2>/dev/null || true
```

Then verify:

```bash
pgrep -af 'daemon.py|runner.py|run_continuous_backtests.py' || true
lsof -nP -iTCP:8080 -sTCP:LISTEN
lsof -nP -iTCP:8090 -sTCP:LISTEN
```

## Tests Already Run

Focused digital-intern tests passed:

```text
42 passed
```

Command:

```bash
cd /Users/jonathan/trading-intelligence/digital-intern
PYTHONPATH=. ../.venv/bin/python -m pytest -q \
  tests/test_model.py \
  tests/test_trainer.py \
  tests/test_features.py \
  tests/test_inference_grey_zone.py \
  tests/test_tts_shutil_import.py
```

Focused paper-trader tests passed:

```text
200 passed, 1 skipped
```

Command:

```bash
cd /Users/jonathan/trading-intelligence/paper-trader
PYTHONPATH=. ../.venv/bin/python -m pytest -q \
  tests/test_core_runner.py \
  tests/test_core_dashboard_helpers.py \
  tests/test_healthz_endpoint.py \
  tests/test_build_info.py \
  tests/test_decision_scorer.py
```

Do not rerun these casually while the Mac is hot. They are documented here so
future agents do not repeat work under load.

## Files Changed Locally

`digital-intern/daemon.py`:

- Added `_env_enabled`.
- Added `DIGITAL_INTERN_CONTINUOUS_TRAINER`.
- Added `DIGITAL_INTERN_WORKERS`.
- Removed `continuous_trainer` from unconditional worker startup.
- Appends `continuous_trainer` only when enabled.
- Applies worker allowlist when `DIGITAL_INTERN_WORKERS` is set.

Root docs added:

- `AGENTS.md`
- `OPENCLAW_LOCAL_RUNBOOK.md`

OpenClaw config changed during setup:

- primary model remains `codex/gpt-5.5`
- fallback chain starts with `xai/grok-4.3`
- xAI provider/plugin enabled locally
- built-in Discord voice enabled
- old standalone `discord-voice` plugin disabled
- Discord voice model restored to `codex/gpt-5.5`
- OpenAI TTS selected for Discord voice

## OpenClaw Agent Invocation Notes

The user asked to prompt OpenClaw to inspect/fix the trading repo. An OpenClaw
agent turn was started with `codex/gpt-5.5` and `xhigh`, but it ran silently for
several minutes and the Mac later rebooted. Do not repeat a long OpenClaw agent
turn while the Mac is under high load. Prefer short, scoped prompts.

Example of a safe short prompt only after services are up:

```bash
openclaw agent \
  --agent main \
  --session-key agent:main:trading-intelligence-status \
  --model codex/gpt-5.5 \
  --thinking xhigh \
  --timeout 180 \
  --message 'Inspect /Users/jonathan/trading-intelligence logs only. Do not run tests, installs, training, or continuous backtests. Report whether 8080 and 8090 are healthy.'
```

## Final Discord Ping

Only send the completion ping after:

- OpenClaw gateway is up
- Discord voice status says the bot is joined and not muted/deafened
- digital-intern ArticleNet-only mode is running or intentionally deferred due
  to high load
- paper-trader is running or intentionally deferred due to high load
- health probes/logs have been checked

Command:

```bash
openclaw message send \
  --channel discord \
  --target channel:1153698670227750954 \
  --message '<@454961974048980992> OpenClaw is running with Codex GPT-5.5 xhigh fast mode, Grok is first backup, Discord voice is joined, and trading services status: ...'
```

Keep the Discord message short and factual.

## Trading Intelligence Unified Dashboard Over Tailscale

The "unified dashboard" for this repo is the Trading Intelligence dashboard
folder, not the OpenClaw Control UI. In this checkout the original root
`dashboard/` folder was moved into:

```text
/Users/jonathan/trading-intelligence/digital-intern/dashboard
```

The unified dashboard entrypoint is:

```text
/Users/jonathan/trading-intelligence/digital-intern/dashboard/server.py
```

It is the FastAPI rich ops dashboard documented by
`digital-intern/dashboard.service`, and it runs on port `8765`.

Local URL:

```text
http://127.0.0.1:8765/
```

Tailnet URL:

```text
http://your-macbook-3.tailaa3a85.ts.net:8765/
http://100.125.75.25:8765/
```

Use the explicit `http://...:8765/` URL. As of 2026-06-07 local time,
Tailscale Serve is not a valid shortcut on this Mac: `tailscale serve` prints
`https://your-macbook-3.tailaa3a85.ts.net/`, but the network extension then
fails to persist the Serve config in Keychain (`tailscale-serve/... Operation
not permitted`), `tailscale serve status` returns `{}` or `No serve config`,
and port `443` stays closed. Do not tell the user to open the no-port HTTPS
URL unless that Keychain/Tailscale issue has been fixed and retested.

The service is intentionally split into a loopback FastAPI server plus a
tailnet-only TCP proxy. This keeps the dashboard available locally and over
Tailscale without binding the unauthenticated dashboard to the whole LAN.

```text
LaunchAgent: com.jonathan.trading-intelligence.unified-dashboard
Plist: /Users/jonathan/Library/LaunchAgents/com.jonathan.trading-intelligence.unified-dashboard.plist
Command: /Users/jonathan/trading-intelligence/.venv/bin/python -m uvicorn dashboard.server:app --host 127.0.0.1 --port 8765

LaunchAgent: com.jonathan.trading-intelligence.unified-dashboard-tailnet-proxy
Plist: /Users/jonathan/Library/LaunchAgents/com.jonathan.trading-intelligence.unified-dashboard-tailnet-proxy.plist
Script: /Users/jonathan/.openclaw/service-env/trading-unified-dashboard-tailnet-proxy.js
Proxy: 100.125.75.25:8765 -> 127.0.0.1:8765
Proxy: [fd7a:115c:a1e0::d93a:4b1a]:8765 -> 127.0.0.1:8765
Logs:
  /Users/jonathan/trading-intelligence/logs/unified-dashboard.launchd.log
  /Users/jonathan/trading-intelligence/logs/unified-dashboard.launchd.err.log
  /Users/jonathan/trading-intelligence/logs/unified-dashboard-tailnet-proxy.log
  /Users/jonathan/trading-intelligence/logs/unified-dashboard-tailnet-proxy.err.log
```

Validate the unified dashboard:

```bash
launchctl print gui/$(id -u)/com.jonathan.trading-intelligence.unified-dashboard
launchctl print gui/$(id -u)/com.jonathan.trading-intelligence.unified-dashboard-tailnet-proxy
lsof -nP -iTCP:8765 -sTCP:LISTEN
curl -i http://127.0.0.1:8765/ | head
curl -i http://your-macbook-3.tailaa3a85.ts.net:8765/ | head
curl -fsS http://your-macbook-3.tailaa3a85.ts.net:8765/api/command-center
```

Expected listeners include the FastAPI dashboard on `127.0.0.1:8765` and the
tailnet proxy on `100.125.75.25:8765`. The proxy also tries to bind the current
Tailscale IPv6 address. On 2026-06-07 local time the socket was visible on
`[fd7a:115c:a1e0::d93a:4b1a]:8765`, but local direct TCP to that IPv6 address
timed out. The tested, working user-facing path is still the MagicDNS/IPv4 URL:
`http://your-macbook-3.tailaa3a85.ts.net:8765/`.

Unified-dashboard route behavior:

```text
/                         Command Center landing page backed by /api/command-center
/api/command-center       Aggregates Digital Intern, Paper Trader desk pulse,
                          game plan, session delta, portfolio, and ops health
/api/action-queue         Operator action slice from the command-center payload
/ops/ and /system/        Same ops dashboard, with /ops/api/* aliases
/intern/                  Same-host proxy to ArticleNet/Digital Intern on 127.0.0.1:8080
/intern/api/*             Same-host proxy to 127.0.0.1:8080/api/*
/intern/chat              Same-host proxy to 127.0.0.1:8080/chat
/trader/                  Same-host proxy to Paper Trader on 127.0.0.1:8090
/trader/backtests         Same-host proxy to 127.0.0.1:8090/backtests
/trader/api/*             Same-host proxy to 127.0.0.1:8090/api/*
/backtests                Unified Compare section page backed by /api/sections/compare
/backtests/compare        Unified Compare section page backed by /api/sections/compare
/strategy-lab             Unified Strategy Lab section backed by /api/sections/strategy
/journal                  Unified Journal section backed by /api/sections/journal
/personas                 Unified Personas section backed by /api/sections/personas
/tape                     Unified Tape section backed by /api/sections/tape
/pulse                    Unified News Pulse section backed by /api/sections/pulse
/api/sections/{section}   Aggregates read-only cards from ArticleNet and Paper Trader
```

Do not replace these with redirects to ports `8080` or `8090`, and do not turn
the unified section pages back into redirects. Paper Trader uses same-origin
API calls and must stay under the unified `:8765` host when served through
Tailscale.

The ArticleNet-only local mode does not run the heavier intern portfolio worker,
so `/intern/api/portfolio` now falls back to the live Paper Trader portfolio
shape when `digital-intern/data/portfolio_pl.json` is absent. That keeps the
intern dashboard's P/L panel and edit controls usable without starting the
continuous worker stack.

Browser verification performed on 2026-06-07 local time with Playwright and
Google Chrome against:

```text
http://your-macbook-3.tailaa3a85.ts.net:8765/
```

Passed checks:

```text
PLAYWRIGHT_PAGE_STABILITY_OK
PLAYWRIGHT_FULL_CONTROLS_OK
PLAYWRIGHT_INTERN_OPS_BUTTONS_OK
PLAYWRIGHT_TRADER_BUTTONS_OK
PLAYWRIGHT_LAUNCHD_SMOKE_OK
PLAYWRIGHT_MAGICDNS_COMMAND_CENTER_OK
PLAYWRIGHT_NAV_STATUS_OK
PLAYWRIGHT_STABLE_PAGES_OK
PLAYWRIGHT_SECTION_PAGES_POPULATED_OK
PLAYWRIGHT_TOP_NAV_USEFUL_CLICK_OK
PLAYWRIGHT_SAFE_BUTTONS_OK
PLAYWRIGHT_POPULATED_DASHBOARD_OK
PLAYWRIGHT_PULSE_TABLE_OK
PLAYWRIGHT_UNIFIED_BUTTONS_POPULATED_OK
```

The checks clicked Command Center refresh/action buttons, top navigation,
intern portfolio edit/add/remove, collector refresh, intern chat open/close,
ops stat filters, Paper Trader mobile drawer, Trader/Backtests tabs, equity
mode/range buttons, and backtest section/filter buttons. They also verified the
relevant `/api/command-center`, `/api/action-queue`, `/intern/api/*`,
`/ops/api/*`, and `/trader/api/*` routes returned non-error HTTP statuses.
The latest stable-page pass opened every visible route over
`http://your-macbook-3.tailaa3a85.ts.net:8765/` with Google Chrome and found
no console or request failures while each page was held open.
The latest populated-button pass opened 13 direct routes, clicked 13 top-nav
routes, and clicked 4 generated service-card links through the same MagicDNS
URL. It required Command Center ArticleNet totals to render and each unified
section page to load real cards instead of placeholders. It also verified that
the Ops worker summary renders intentionally stopped workers as `off` while the
LaunchAgent is in `DIGITAL_INTERN_WORKERS=web_server` mode.

## Other Trading Dashboard Links Over Tailscale

Current Tailscale node observed on 2026-06-07:

```text
DNS: your-macbook-3.tailaa3a85.ts.net
IPv4: 100.125.75.25
```

These are separate trading service dashboards, not the unified dashboard.
They listen on `0.0.0.0`, so they are reachable directly over the tailnet:

```text
Digital Intern / ArticleNet dashboard:
http://your-macbook-3.tailaa3a85.ts.net:8080/
http://100.125.75.25:8080/

Paper Trader dashboard:
http://your-macbook-3.tailaa3a85.ts.net:8090/
http://100.125.75.25:8090/
```

Health probes that passed:

```bash
curl http://your-macbook-3.tailaa3a85.ts.net:8080/healthz
curl http://your-macbook-3.tailaa3a85.ts.net:8090/api/healthz
```

`tailscale serve --bg --yes 8080` and the later unified-dashboard Serve forms
printed URLs but did not retain a Serve config on this macOS client. The system
log showed the Tailscale network extension could not add `tailscale-serve/...`
to Keychain (`Operation not permitted`), `tailscale serve status` returned
`No serve config` or `{}`, and 443 refused connections. Direct tailnet
host:port access is the working path for now.

If the hostname changes again, get the current values with:

```bash
/Applications/Tailscale.app/Contents/MacOS/Tailscale status --json
/Applications/Tailscale.app/Contents/MacOS/Tailscale ip -4
```

Repeated `your-macbook-N` names usually mean Tailscale is re-registering this
Mac instead of reusing the prior node. Avoid `tailscale logout` for normal
disconnects; logout forces re-authentication. If the suffix keeps increasing
after ordinary restarts, inspect Tailscale state persistence in the macOS app
container/keychain and remove stale duplicate machines from the Tailscale admin
console.
