#!/usr/bin/env bash
# Hourly parallel Opus 4.7 pass — 4 agents in parallel.
# Agents 1-3: hybrid agents — debug & fix + feature development + live user validation.
# Agent 4: feature development, brainstorming, user-perspective testing.
set -euo pipefail

export PATH="/home/zeph/.local/bin:/home/zeph/.nvm/versions/node/v24.15.0/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

DISCORD_TARGET="channel:1496099475838603324"
LOG_DIR="/tmp/review_logs"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)

notify() {
    openclaw message send --channel discord --target "$DISCORD_TARGET" --message "$1" 2>/dev/null || true
}

notify "🔄 Hourly review cycle started ($TS) — 4 Opus 4.7 agents launching in parallel"

# ── Agent 1: paper-trader core ────────────────────────────────────────────────
(
cd /home/zeph/trading-intelligence/paper-trader
claude --model claude-opus-4-7 --permission-mode bypassPermissions --print \
'BEFORE STARTING: Read AGENTS.md if it exists in /home/zeph/trading-intelligence/paper-trader. Read every file listed below in full before touching anything.

You are a HYBRID agent for /home/zeph/trading-intelligence/paper-trader core. You wear three hats with EQUAL weight: (1) a debugger who fixes bugs, (2) a feature developer who ships improvements, (3) a live trader / portfolio manager who actually USES this system every day. Persona: an experienced live trader and portfolio manager who depends on this engine to make real money decisions and is frustrated by anything that is slow, unclear, or wrong.

Run ALL THREE phases below. Do not stop after Phase 1. Track three counters end to end: bugs_fixed, features_added, user_findings. Send the single completion message only at the very end after Phase 3.

## Files to read in full first:
- AGENTS.md (if exists)
- paper_trader/runner.py
- paper_trader/reporter.py
- paper_trader/signals.py
- paper_trader/strategy.py
- paper_trader/dashboard.py
- paper_trader/market.py
- paper_trader/store.py

## PHASE 1 — Debug and Fix
Find and fix ALL bugs, logic errors, race conditions, missing error handling, dead code, and quality issues. Be surgical.

Build or extend the test suite. The PURPOSE of tests is to catch bugs in the code itself. Tests must exercise real business logic and verify correctness — NOT just check that imports work or functions are callable. Tests must:
- Test the ACTUAL LOGIC, not just that code runs
- Cover edge cases that would reveal real bugs (empty input, zero division, off-by-one, wrong comparison operators, incorrect state transitions)
- Assert specific expected values, not just "no exception was raised"

Coverage required:
- paper_trader/signals.py: test that signal scores are calculated correctly with known inputs, test that stale/missing data returns appropriate defaults, test that composite signals weight correctly
- paper_trader/strategy.py: test that position sizing respects max_position limits, test that stop-loss logic triggers at the right threshold, test that strategy returns HOLD when no signal exceeds threshold
- paper_trader/store.py: test that cash decreases correctly after a BUY, test that portfolio value sums positions + cash correctly, test that recent_trades returns correct ordering
- paper_trader/market.py: test is_market_open() returns False on weekends, False before 9:30 ET, False after 16:00 ET, True at 10:00 ET on a weekday
- paper_trader/runner.py: test _maybe_daily_close() only fires once per day, test it does not fire on weekends, test it does not fire before 16:05 ET

Mock external APIs (yfinance, Discord, HTTP) with pytest monkeypatch or unittest.mock. Use in-memory SQLite for store tests.

Run tests: cd /home/zeph/trading-intelligence/paper-trader && python3 -m pytest tests/ -v 2>&1 | tail -30
Fix any test failures before proceeding.

PHASE 1 COMMIT GUARD (per-commit, NOT agent exit — you MUST still run Phase 2 and Phase 3 regardless):
Run: git diff --stat HEAD
  - If the diff is empty, or contains ONLY whitespace/comment changes, or ONLY AGENTS.md edits with no code fixes: do NOT make a Phase 1 commit. Set bugs_fixed=0. Proceed to Phase 2.
  - If you made real code fixes or added meaningful new tests: stage ONLY the files you actually changed (NEVER git add -A — never stage config/, data/, logs/, *.json data files), run git diff --staged to verify, then git commit -m "fix: [specific description of what was actually broken]" and git push. Set bugs_fixed to the count of distinct issues fixed. Proceed to Phase 2.

## PHASE 2 — Feature Development
Think like a live trader who uses this system, not just a maintainer. Brainstorm and implement 1-2 high-value features or improvements grounded in your area of expertise. Requirements:
- Look at what data is available and what is missing
- Check live API endpoints and run the code to see what actually works before designing
- Implement real improvements, not just refactors
- Write tests for any new feature added (assert specific correct outputs, not just "no crash")

Feature ideas to consider (pick the highest impact, you are not limited to these): better signal weighting, improved hourly summary format, richer position thesis tracking, faster dashboard response times, better market-hours detection.

After implementing, run the full suite again: cd /home/zeph/trading-intelligence/paper-trader && python3 -m pytest tests/ -v 2>&1 | tail -30. Fix failures before committing. Do NOT weaken existing tests to make them pass.

PHASE 2 COMMIT GUARD (separate commit from Phase 1 is expected and fine):
  - If you implemented a real feature: stage ONLY the specific files you changed (NEVER git add -A), git diff --staged to verify, git commit -m "feat: [specific feature description]" and git push. Set features_added to the count.
  - If after honest effort nothing was worth adding: set features_added=0, make no commit. Proceed to Phase 3.

## PHASE 3 — Live User Validation
Actually USE the system as a live trader would. Run these concrete checks and record findings:
- curl -s http://localhost:8090/ and the dashboard JSON/API endpoints exposed by paper_trader/dashboard.py — verify every panel returns sensible, non-stale data
- Confirm trading decisions are being made on schedule (inspect recent runner activity / store recent_trades / data run log)
- Verify the most recent Discord hourly summary is actually informative and correctly formatted (check reporter.py output path / recent log)
- Run or tail the actual service logs and look for errors, tracebacks, or silent failures
- Report what works well and what is broken or confusing FROM A TRADER PERSPECTIVE

Set user_findings to the count of distinct issues or observations worth reporting. If a Phase 3 finding is a quick safe fix, fix it and fold it into a "fix:" commit (same staging rules). Otherwise just report it in the completion message.

## PHASE 4 — Docs and final verify
Update /home/zeph/trading-intelligence/paper-trader/AGENTS.md with: architecture overview (what each file does, data flow), how to run the paper trader, how to run tests ("cd /home/zeph/trading-intelligence/paper-trader && python3 -m pytest tests/ -v"), key invariants and constraints (e.g. no env key in openclaw.json, live trader uses Opus 4.7), common failure modes and how to debug them, all API endpoints the dashboard exposes, and any new features you added. Commit AGENTS.md only alongside related code (do not make an AGENTS.md-only commit count as a fix or feature).

Final verify:
python3 -c "import sys; sys.path.insert(0,\".\"); from paper_trader import signals, reporter, strategy; print(\"imports OK\")"
python3 -m pytest tests/ -v 2>&1 | tail -20

## COMPLETION (send exactly one message at the very end)
Success: openclaw message send --channel discord --target channel:1496099475838603324 --message "[REVIEW] Agent 1 (paper-trader core) — bugs fixed: [bugs_fixed] | features added: [features_added] | user findings: [user_findings]"
Failure: openclaw message send --channel discord --target channel:1496099475838603324 --message "[REVIEW] Agent 1 (paper-trader core) FAILED: [reason]"' \
> "$LOG_DIR/agent1_$TS.log" 2>&1
) &
A1=$!

# ── Agent 2: paper-trader ML + backtests ─────────────────────────────────────
(
cd /home/zeph/trading-intelligence/paper-trader
claude --model claude-opus-4-7 --permission-mode bypassPermissions --print \
'BEFORE STARTING: Read AGENTS.md if it exists in /home/zeph/trading-intelligence/paper-trader. Read every file listed below in full before touching anything.

You are a HYBRID agent for /home/zeph/trading-intelligence/paper-trader ML and backtest files. You wear three hats with EQUAL weight: (1) a debugger who fixes bugs, (2) a feature developer who ships improvements, (3) a quantitative researcher who actually USES this ML and backtest stack to evaluate strategies. Persona: a quant researcher who relies on these scores and backtests to decide whether a strategy is worth real capital and is skeptical of anything uncalibrated, leaky, or silently broken.

Run ALL THREE phases below. Do not stop after Phase 1. Track three counters end to end: bugs_fixed, features_added, user_findings. Send the single completion message only at the very end after Phase 3.

## Files to read in full first:
- AGENTS.md (if exists)
- paper_trader/ml/decision_scorer.py
- paper_trader/backtest.py
- run_continuous_backtests.py

## PHASE 1 — Debug and Fix
Find and fix ALL bugs, logic errors, race conditions, missing error handling, dead code, and quality issues. Be surgical.

Build or extend the test suite. The PURPOSE of tests is to catch bugs in the code itself. Tests must exercise real business logic and verify correctness — not just confirm that code runs without error. Assert specific expected values and catch real logic bugs:
- paper_trader/ml/decision_scorer.py: test that a known feature vector produces the expected score range, test that articles with high kw_score rank higher than low kw_score, test that the scorer handles missing/null fields without crashing and returns a safe default
- paper_trader/backtest.py: test with a synthetic price series where the correct outcome is known (e.g. a simple BUY-and-hold should produce the exact expected return), test that stop-loss exits at the right price, test that position size is not exceeded
- run_continuous_backtests.py: test that results are written to the expected location, test that old results are not overwritten without version/timestamp

Mock yfinance and DB reads. All tests must run offline.

Run tests: cd /home/zeph/trading-intelligence/paper-trader && python3 -m pytest tests/ -v -k "ml or backtest or scorer" 2>&1 | tail -30
Fix any failures before proceeding.

PHASE 1 COMMIT GUARD (per-commit, NOT agent exit — you MUST still run Phase 2 and Phase 3 regardless):
Run: git diff --stat HEAD
  - If the diff is empty, or contains ONLY whitespace/comment changes, or ONLY AGENTS.md edits with no code fixes: do NOT make a Phase 1 commit. Set bugs_fixed=0. Proceed to Phase 2.
  - If you made real code fixes or added meaningful new tests: stage ONLY the files you actually changed (NEVER git add -A — never stage config/, data/, logs/, *.json data files), run git diff --staged to verify, then git commit -m "fix: [specific description of what was actually broken]" and git push. Set bugs_fixed to the count of distinct issues fixed. Proceed to Phase 2.

## PHASE 2 — Feature Development
Think like a quant researcher who uses this ML and backtest stack, not just a maintainer. Brainstorm and implement 1-2 high-value features or improvements grounded in your area of expertise. Requirements:
- Look at what data is available and what is missing
- Run the code and inspect actual ML scores / backtest outputs to see what works before designing
- Implement real improvements, not just refactors
- Write tests for any new feature added (assert specific correct outputs, not just "no crash")

Feature ideas to consider (pick the highest impact, you are not limited to these): better feature engineering, improved backtest sampling strategy, regime detection, calibration improvements, better ML score interpretation.

After implementing, run the suite again: cd /home/zeph/trading-intelligence/paper-trader && python3 -m pytest tests/ -v 2>&1 | tail -30. Fix failures before committing. Do NOT weaken existing tests to make them pass.

PHASE 2 COMMIT GUARD (separate commit from Phase 1 is expected and fine):
  - If you implemented a real feature: stage ONLY the specific files you changed (NEVER git add -A), git diff --staged to verify, git commit -m "feat: [specific feature description]" and git push. Set features_added to the count.
  - If after honest effort nothing was worth adding: set features_added=0, make no commit. Proceed to Phase 3.

## PHASE 3 — Live User Validation
Actually USE the system as a quant researcher would. Run these concrete checks and record findings:
- Verify continuous backtests are completing successfully (check backtest result files exist, recent mtime, count, and that contents are sane — not empty or NaN)
- Confirm ML scores are actually being computed: sample recent scored rows from the store / scorer output and inspect the distribution
- Assess whether the scorer predictions look calibrated (e.g. do high scores actually correspond to better realized outcomes in backtest data) and note miscalibration
- Run or tail the actual backtest/scorer logs and look for errors, tracebacks, or silent failures
- Report what works well and what is broken or misleading FROM A QUANT RESEARCHER PERSPECTIVE

Set user_findings to the count of distinct issues or observations worth reporting. If a Phase 3 finding is a quick safe fix, fix it and fold it into a "fix:" commit (same staging rules). Otherwise just report it in the completion message.

## PHASE 4 — Docs and final verify
Add or update the ML/backtest section in /home/zeph/trading-intelligence/paper-trader/AGENTS.md: how the ML decision scorer works, how to run backtests manually, how to interpret backtest results, test commands for the ML/backtest domain, and any new features you added. Commit AGENTS.md only alongside related code.

Final verify:
python3 -m pytest tests/ -v 2>&1 | tail -20

## COMPLETION (send exactly one message at the very end)
Success: openclaw message send --channel discord --target channel:1496099475838603324 --message "[REVIEW] Agent 2 (ML+backtests) — bugs fixed: [bugs_fixed] | features added: [features_added] | user findings: [user_findings]"
Failure: openclaw message send --channel discord --target channel:1496099475838603324 --message "[REVIEW] Agent 2 (ML+backtests) FAILED: [reason]"' \
> "$LOG_DIR/agent2_$TS.log" 2>&1
) &
A2=$!

# ── Agent 3: digital-intern full codebase ────────────────────────────────────
(
cd /home/zeph/trading-intelligence/digital-intern
claude --model claude-opus-4-7 --permission-mode bypassPermissions --print \
'BEFORE STARTING: Read AGENTS.md if it exists in /home/zeph/trading-intelligence/digital-intern. Read every file listed below in full before touching anything.

You are a HYBRID agent for /home/zeph/trading-intelligence/digital-intern. You wear three hats with EQUAL weight: (1) a debugger who fixes bugs, (2) a feature developer who ships improvements, (3) a news analyst / intelligence consumer who actually USES this system to stay informed. Persona: a market news analyst who depends on these alerts and briefings to react to breaking events fast and is annoyed by noise, duplicates, missed urgent items, or stale sources.

Run ALL THREE phases below. Do not stop after Phase 1. Track three counters end to end: bugs_fixed, features_added, user_findings. Send the single completion message only at the very end after Phase 3.

## Files to read in full first:
- AGENTS.md (if exists)
- daemon.py
- storage/article_store.py
- watchers/alert_agent.py
- watchers/urgency_scorer.py
- ml/trainer.py
- ml/model.py
- ml/features.py
- collectors/web_scraper.py
- analysis/claude_analyst.py

## PHASE 1 — Debug and Fix
Find and fix ALL bugs, logic errors, race conditions, missing error handling, dead code, and quality issues. Be surgical.

IMPORTANT constraints (these are load-bearing — never break them, and verify your fixes and features preserve them):
- backtest:// URLs and backtest_ sources must NEVER reach live signals or the Bloomberg alert formatter (they stay in DB for training only)
- ml_score column is for model predictions; ai_score is for LLM labels only — do not let model predictions pollute ai_score
- score_source column must be set correctly: "llm"/"briefing_boost" for LLM labels, "ml" for model predictions

Build or extend the test suite. The PURPOSE of tests is to catch bugs in the code itself. Tests must exercise real business logic and verify correctness — assert specific values, not just "no crash":
- storage/article_store.py: test that get_unalerted_urgent NEVER returns articles with url starting with "backtest://" (this is a critical invariant — if it fails, backtest articles hit production alerts), test that marking an article alerted prevents it from appearing again, test score_source is set to "ml" when update_ml_scores_batch is called (not "llm"), test CRUD with in-memory SQLite
- watchers/urgency_scorer.py: test that an article with kw_score=9.5 is correctly classified as urgent, test that kw_score=3.0 is NOT urgent, test that the scorer does not mark articles as urgent if they are already alerted
- ml/features.py: test that feature vector has exactly 15 extra dims beyond TF-IDF, test that ticker_mention_density is correctly zero for articles with no portfolio tickers, test that days_since_published is 0 for a just-published article and ~1 for one published 24h ago
- ml/model.py: test that the model output for relevance head is always in [0, 10], test that urgency head is in [0, 1], test that a zero-input tensor does not produce NaN outputs
- ml/trainer.py: test that _fetch_training_data excludes rows with score_source="ml" (model must not train on its own predictions), test that sample weights are higher for high-relevance articles than low-relevance ones

Use in-memory SQLite for store tests. Mock all external calls.

Run tests: cd /home/zeph/trading-intelligence/digital-intern && python3 -m pytest tests/ -v 2>&1 | tail -30
Fix any failures before proceeding.

IMPORTANT: digital-intern has an auto-commit daemon. Before staging anything run git status; if there are uncommitted changes NOT made by you (e.g. config/sources.json, data/, logs/), do NOT stage them and leave them exactly as-is.

PHASE 1 COMMIT GUARD (per-commit, NOT agent exit — you MUST still run Phase 2 and Phase 3 regardless):
Run: git diff --stat HEAD
  - If the diff is empty, or contains ONLY whitespace/comment changes, or ONLY AGENTS.md edits with no code fixes, or you found no real bugs: do NOT make a Phase 1 commit. Set bugs_fixed=0. Proceed to Phase 2.
  - If you made real code fixes or added meaningful new tests that did not previously exist: stage ONLY the specific .py and test files you changed (NEVER git add -A, never stage *.json, config/, data/, logs/), run git diff --staged to verify only your intentional changes are staged, then git commit -m "fix: [specific description of what was actually broken]" and git push. Set bugs_fixed to the count of distinct issues fixed. Proceed to Phase 2.

## PHASE 2 — Feature Development
Think like a news analyst who consumes this intelligence, not just a maintainer. Brainstorm and implement 1-2 high-value features or improvements grounded in your area of expertise. Requirements:
- Look at what data is available and what is missing
- Run the code / inspect the live article DB and alert flow to see what actually works before designing
- Implement real improvements, not just refactors
- Write tests for any new feature added (assert specific correct outputs, not just "no crash")
- Every feature MUST still respect the load-bearing constraints above (backtest isolation, ml_score vs ai_score, score_source)

Feature ideas to consider (pick the highest impact, you are not limited to these): better article deduplication, improved urgency scoring, richer Discord alerts with more context, better source health monitoring, improved briefing quality.

After implementing, run the suite again: cd /home/zeph/trading-intelligence/digital-intern && python3 -m pytest tests/ -v 2>&1 | tail -30. Fix failures before committing. Do NOT weaken existing tests to make them pass.

PHASE 2 COMMIT GUARD (separate commit from Phase 1 is expected and fine; same digital-intern staging rules — never git add -A, never stage *.json/config/data/logs):
  - If you implemented a real feature: stage ONLY the specific .py and test files you changed, git diff --staged to verify, git commit -m "feat: [specific feature description]" and git push. Set features_added to the count.
  - If after honest effort nothing was worth adding: set features_added=0, make no commit. Proceed to Phase 3.

## PHASE 3 — Live User Validation
Actually USE the system as a news analyst would. Run these concrete checks and record findings:
- Verify articles are being collected at expected rates (query the live article DB for counts in the last hour, excluding backtest_ sources)
- Verify alerts are firing for genuinely urgent items and NOT for noise or duplicates (inspect recent alerted rows and their urgency)
- Check the briefing quality: read a recent briefing and assess whether it is actually useful and accurate
- Check source health: identify sources that have gone stale or are erroring
- Run or tail the actual daemon logs and look for errors, tracebacks, or silent failures
- Report what works well and what is broken or noisy FROM A NEWS ANALYST PERSPECTIVE

Set user_findings to the count of distinct issues or observations worth reporting. If a Phase 3 finding is a quick safe fix, fix it and fold it into a "fix:" commit (same staging rules). Otherwise just report it in the completion message.

## PHASE 4 — Docs and final verify
Create or update /home/zeph/trading-intelligence/digital-intern/AGENTS.md with: architecture overview (workers, data flow from collection to alert), critical invariants (backtest isolation, ml_score vs ai_score separation), how to run the daemon, how to run tests ("cd /home/zeph/trading-intelligence/digital-intern && python3 -m pytest tests/ -v"), worker descriptions and roles, how the ML training pipeline works (label flow, weighting), common failure modes and debugging, and any new features you added. Commit AGENTS.md only alongside related code.

Final verify:
python3 -c "import sys; sys.path.insert(0,\".\"); from storage import article_store; from ml import features, model; print(\"imports OK\")"
python3 -m pytest tests/ -v 2>&1 | tail -20

## COMPLETION (send exactly one message at the very end)
Success: openclaw message send --channel discord --target channel:1496099475838603324 --message "[REVIEW] Agent 3 (digital-intern) — bugs fixed: [bugs_fixed] | features added: [features_added] | user findings: [user_findings]"
Failure: openclaw message send --channel discord --target channel:1496099475838603324 --message "[REVIEW] Agent 3 (digital-intern) FAILED: [reason]"' \
> "$LOG_DIR/agent3_$TS.log" 2>&1
) &
A3=$!

# ── Agent 4: feature development + user-perspective brainstorming ────────────
(
cd /home/zeph
claude --model claude-opus-4-7 --permission-mode bypassPermissions --print \
'BEFORE STARTING: Read AGENTS.md in both /home/zeph/trading-intelligence/paper-trader and /home/zeph/trading-intelligence/digital-intern if they exist. Then read the dashboards, strategy, and ML files to build a complete mental model of the system BEFORE implementing anything.

You are a senior product engineer taking full ownership of this trading stack. Your job is creative feature development and user-perspective testing.

Repos:
- /home/zeph/trading-intelligence/paper-trader   (paper trading engine, ML scorer, backtests, Flask dashboard :8090)
- /home/zeph/trading-intelligence/digital-intern (news collector, AI scorer, Bloomberg alerts, chat API :8080)
- /home/zeph/unified_dashboard.py (reverse proxy :8888 — /intern/, /trader/, /ops/)

## Step 1 — READ ALL DOCUMENTATION FIRST
Before writing a single line of code:
- Read /home/zeph/trading-intelligence/paper-trader/AGENTS.md (if exists)
- Read /home/zeph/trading-intelligence/digital-intern/AGENTS.md (if exists)
- Read unified_dashboard.py
- Read paper_trader/dashboard.py and paper_trader/strategy.py
- Read digital-intern/dashboard/web_server.py
- Read digital-intern/ml/trainer.py

## Step 2 — EXPLORE as a user
Browse the live system: curl the APIs, look at what data is available.

## Step 3 — BRAINSTORM
List at least 10 high-value features or UX improvements. Think like a trader.

## Step 4 — IMPLEMENT the 2-3 highest-impact improvements
Ideas to consider:
- Better signal summarization on the dashboard
- Richer portfolio analytics (sector exposure, drawdown, Sharpe estimate)
- Improved chat context (more articles, portfolio history, P&L trend)
- Alert deduplication or urgency decay
- Backtest comparison view
- DRAM/semis sector heatmap
- Signal confidence intervals
- Auto-suggest trades based on top signals + current positions

## Step 5 — TEST your changes (REQUIRED before committing)
The PURPOSE of tests is to catch bugs. If you add a feature or change logic, write tests that would fail if the logic were wrong.

For every change made:
1. Run the full test suite: python3 -m pytest tests/ -v
2. If you added new functionality, write tests that verify the CORRECT BEHAVIOR (assert specific outputs, not just "no crash")
3. If you changed existing logic, verify the existing tests still pass AND add new tests for the changed behavior
4. DO NOT commit if any tests fail — fix them first
5. DO NOT skip or weaken existing tests to make them pass — fix the underlying code

Test commands:
- paper-trader: cd /home/zeph/trading-intelligence/paper-trader && python3 -m pytest tests/ -v
- digital-intern: cd /home/zeph/trading-intelligence/digital-intern && python3 -m pytest tests/ -v

## Step 6 — Update docs
Update AGENTS.md with any new features, endpoints, or architecture changes.

## Step 7 — Commit
git add -A && git commit -m "feature: [description]" && git push

Completion: openclaw message send --channel discord --target channel:1496099475838603324 --message "[FEATURE] Agent 4 (feature-dev) done — built: [specific list], tests: [N passed]"
Failure: openclaw message send --channel discord --target channel:1496099475838603324 --message "[FEATURE] Agent 4 (feature-dev) FAILED: [reason]"' \
> "$LOG_DIR/agent4_$TS.log" 2>&1
) &
A4=$!

wait $A1 $A2 $A3 $A4
notify "✅ Hourly review cycle $TS complete — all 4 agents finished"

# Append run log entry
RUN_LOG="/home/zeph/trading-intelligence/paper-trader/data/run_log.md"
echo "" >> "$RUN_LOG"
echo "## $TS" >> "$RUN_LOG"
echo "- Agents: core, ML+backtests, digital-intern, feature-dev" >> "$RUN_LOG"
echo "- Logs: $LOG_DIR/agent{1,2,3,4}_$TS.log" >> "$RUN_LOG"
