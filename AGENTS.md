# Codex/OpenClaw Notes For This Local Setup

Read `OPENCLAW_LOCAL_RUNBOOK.md` before touching this repo or the local
OpenClaw runtime. This Mac has already rebooted under load during setup work,
so future agents must prefer small, observable steps over broad installs,
source builds, full test suites, or high-fanout daemons.

Hard rules for this machine:

- Do not run `paper-trader/run_continuous_backtests.py` unless Jonathan
  explicitly reverses the instruction. This Mac is not meant to run the
  continuous backtest/training loop.
- Current Mac-safe steady state is dashboard/API only:
  `com.jonathan.trading-intelligence.digital-intern` runs
  `dashboard.web_server.run_server(None)` on `127.0.0.1:8080`, and
  `com.jonathan.trading-intelligence.paper-trader` runs
  `paper_trader.dashboard.run()` on `127.0.0.1:8090`.
- Do not start `digital-intern/daemon.py` or `paper-trader/runner.py` as the
  steady state on this Mac unless Jonathan explicitly asks for live collector
  or trader loops. If `digital-intern/daemon.py` is used for a short supervised
  refresh, set `DIGITAL_INTERN_CONTINUOUS_TRAINER=0` and return to the
  dashboard-only LaunchAgent afterward.
- Use `/Users/jonathan/trading-intelligence/.venv/bin/python`, created with
  Homebrew Python 3.12. Do not use the system `python3`.
- Keep `numpy<2` for ArticleNet/Torch stability. `kokoro-onnx` has a metadata
  conflict with this and is optional here because OpenClaw handles Discord
  voice/TTS.
- Do not enable the old `plugins.entries.discord-voice` plugin. The working
  Discord voice path is the built-in OpenClaw `channels.discord.voice` bridge.
- Do not set `channels.discord.voice.model` to an xAI beta override. Voice
  uses `codex/gpt-5.5`; Grok/xAI is configured as the backup model chain, not
  the voice model override.
- The unified Trading Intelligence dashboard is
  `digital-intern/dashboard/server.py` on port `8765`, exposed over Tailscale
  by `com.jonathan.trading-intelligence.unified-dashboard-tailnet-proxy`.
  The tested user-facing URL is
  `http://your-macbook-3.tailaa3a85.ts.net:8765/`, not the no-port HTTPS URL.
  OpenClaw Control `18789`, ArticleNet `8080`, and Paper Trader `8090` are not
  the unified trading dashboard.
- Do not rely on Tailscale Serve for this dashboard unless it has been fixed
  and retested. On 2026-06-07, `tailscale serve` printed the HTTPS URL but
  failed to persist config because the network extension could not add
  `tailscale-serve/...` to Keychain; `tailscale serve status` stayed empty and
  port `443` was closed.
- The unified dashboard must proxy `/intern/*` and `/trader/*` under the same
  `:8765` host. Do not "fix" it by redirecting the user to raw ports `8080`
  or `8090`; Paper Trader same-origin API calls will break.
- `/` is the unified Command Center and must render
  `digital-intern/dashboard/command_center.html` backed by
  `/api/command-center`. `/ops/` and `/system/` are the ops terminal view.
- `/backtests`, `/backtests/compare`, `/strategy-lab`, `/journal`,
  `/personas`, `/tape`, and `/pulse` are real unified section pages backed by
  `/api/sections/{section}`. Do not turn them back into redirects.
- Keep Paper Trader's LaunchAgent file-descriptor limits:
  `SoftResourceLimits.NumberOfFiles=4096` and
  `HardResourceLimits.NumberOfFiles=8192`. The default launchd soft limit of
  `256` caused `OSError: [Errno 24] Too many open files` and connection resets
  under dashboard fan-out.
- The light ArticleNet-only setup does not run the intern portfolio worker, so
  `/intern/api/portfolio` intentionally falls back to the live Paper Trader
  portfolio when `digital-intern/data/portfolio_pl.json` is absent.
- The recovered ArticleNet database has millions of rows. Keep
  `/api/articles` recency-indexed on `first_seen`; score-sorting the whole
  table blocked the unified dashboard after USB recovery.
- Browser checks against `http://your-macbook-3.tailaa3a85.ts.net:8765/`
  passed for Command Center, all visible nav routes, and stable page loads:
  `PLAYWRIGHT_MAGICDNS_COMMAND_CENTER_OK`, `PLAYWRIGHT_NAV_STATUS_OK`, and
  `PLAYWRIGHT_STABLE_PAGES_OK`. The populated unified dashboard/button pass
  also passed: `PLAYWRIGHT_UNIFIED_BUTTONS_POPULATED_OK` for 13 direct routes,
  13 nav clicks, and 4 service-card links. That check also verifies the
  Command Center worker summary reports intentionally stopped workers as `off`
  in `DIGITAL_INTERN_WORKERS=web_server` mode.
- Never print, paste, commit, or summarize tokens, API keys, OAuth payloads,
  credential files, or raw auth/profile JSON.

Before starting services, check the Mac is not still recovering from reboot:

```bash
uptime
ps -ax -o pid,ppid,state,%cpu,%mem,etime,command | sort -k4 -nr | sed -n '1,20p'
```

If load is still high, wait. Do not force service startup into a hot machine.
