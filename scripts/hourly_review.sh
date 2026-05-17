#!/usr/bin/env bash
# Hourly parallel Opus 4.7 pass — 4 agents in parallel.
# Agents 1-3: systematic code review + bug fixes + test suite + docs.
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
cd /home/zeph/paper-trader
claude --model claude-opus-4-7 --permission-mode bypassPermissions --print \
'BEFORE STARTING: Read AGENTS.md if it exists in /home/zeph/paper-trader. Read every file listed below in full before touching anything.

You are doing a systematic code review, bug-fix, test suite, and documentation pass on /home/zeph/paper-trader core.

## Files to read in full first:
- AGENTS.md (if exists)
- paper_trader/runner.py
- paper_trader/reporter.py
- paper_trader/signals.py
- paper_trader/strategy.py
- paper_trader/dashboard.py
- paper_trader/market.py
- paper_trader/store.py

## Step 1 — Bug fix pass
Find and fix ALL bugs, logic errors, race conditions, missing error handling, dead code, and quality issues. Be surgical.

## Step 2 — Build comprehensive test suite
The PURPOSE of tests is to catch bugs in the code itself. Tests must exercise real business logic and verify correctness — NOT just check that imports work or functions are callable.

Create or update tests/ directory with pytest tests that:
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

Run tests: cd /home/zeph/paper-trader && python3 -m pytest tests/ -v 2>&1 | tail -30

Fix any test failures before proceeding.

## Step 3 — Write/update AGENTS.md
Create or update /home/zeph/paper-trader/AGENTS.md with:
- Architecture overview (what each file does, data flow)
- How to run the paper trader
- How to run tests: "cd /home/zeph/paper-trader && python3 -m pytest tests/ -v"
- Key invariants and constraints (e.g. no env key in openclaw.json, live trader uses Opus 4.7)
- Common failure modes and how to debug them
- All API endpoints the dashboard exposes

## Step 4 — Verify
python3 -c "import sys; sys.path.insert(0,\".\"); from paper_trader import signals, reporter, strategy; print(\"imports OK\")"
python3 -m pytest tests/ -v 2>&1 | tail -20

## Step 5 — Commit ONLY if you made real changes
Run: git diff --stat HEAD

If the diff is empty, or contains ONLY whitespace/comment changes, or ONLY AGENTS.md edits with no code fixes:
  - Do NOT commit anything
  - Send: openclaw message send --channel discord --target channel:1496099475838603324 --message "[REVIEW] Agent 1 (paper-trader core) NO-OP — everything already correct, [N] tests pass"
  - EXIT

If you made real code fixes or added meaningful new tests:
  - Stage only the files you actually changed (NOT git add -A — never stage config/, data/, logs/, *.json data files)
  - git diff --staged (verify the diff is what you intend)
  - git commit -m "fix: [specific description of what was actually broken]"
  - git push
  - Send: openclaw message send --channel discord --target channel:1496099475838603324 --message "[REVIEW] Agent 1 (paper-trader core) done — fixed: [specific issues], tests: [N passed]"

Failure: openclaw message send --channel discord --target channel:1496099475838603324 --message "[REVIEW] Agent 1 (paper-trader core) FAILED: [reason]"' \
> "$LOG_DIR/agent1_$TS.log" 2>&1
) &
A1=$!

# ── Agent 2: paper-trader ML + backtests ─────────────────────────────────────
(
cd /home/zeph/paper-trader
claude --model claude-opus-4-7 --permission-mode bypassPermissions --print \
'BEFORE STARTING: Read AGENTS.md if it exists in /home/zeph/paper-trader. Read every file listed below in full before touching anything.

You are doing a systematic code review, bug-fix, test suite, and documentation pass on /home/zeph/paper-trader ML and backtest files.

## Files to read in full first:
- AGENTS.md (if exists)
- paper_trader/ml/decision_scorer.py
- paper_trader/backtest.py
- run_continuous_backtests.py

## Step 1 — Bug fix pass
Find and fix ALL bugs, logic errors, race conditions, missing error handling, dead code, and quality issues. Be surgical.

## Step 2 — Build comprehensive test suite
The PURPOSE of tests is to catch bugs in the code itself. Tests must exercise real business logic and verify correctness — not just confirm that code runs without error.

Create or update tests/ with pytest tests that assert specific expected values and catch real logic bugs:
- paper_trader/ml/decision_scorer.py: test that a known feature vector produces the expected score range, test that articles with high kw_score rank higher than low kw_score, test that the scorer handles missing/null fields without crashing and returns a safe default
- paper_trader/backtest.py: test with a synthetic price series where the correct outcome is known (e.g. a simple BUY-and-hold should produce the exact expected return), test that stop-loss exits at the right price, test that position size is not exceeded
- run_continuous_backtests.py: test that results are written to the expected location, test that old results are not overwritten without version/timestamp

Mock yfinance and DB reads. All tests must run offline.

Run tests: cd /home/zeph/paper-trader && python3 -m pytest tests/ -v -k "ml or backtest or scorer" 2>&1 | tail -30

Fix any failures before proceeding.

## Step 3 — Update AGENTS.md
Add or update ML/backtest section in /home/zeph/paper-trader/AGENTS.md:
- How the ML decision scorer works
- How to run backtests manually
- How to interpret backtest results
- Test commands for ML/backtest domain

## Step 4 — Verify
python3 -m pytest tests/ -v 2>&1 | tail -20

## Step 5 — Commit ONLY if you made real changes
Run: git diff --stat HEAD

If the diff is empty, or contains ONLY whitespace/comment changes, or ONLY AGENTS.md edits with no code fixes:
  - Do NOT commit anything
  - Send: openclaw message send --channel discord --target channel:1496099475838603324 --message "[REVIEW] Agent 2 (ML+backtests) NO-OP — everything already correct, [N] tests pass"
  - EXIT

If you made real code fixes or added meaningful new tests:
  - Stage only the files you actually changed (NOT git add -A — never stage config/, data/, logs/, *.json data files)
  - git diff --staged (verify the diff is what you intend)
  - git commit -m "fix: [specific description of what was actually broken]"
  - git push
  - Send: openclaw message send --channel discord --target channel:1496099475838603324 --message "[REVIEW] Agent 2 (ML+backtests) done — fixed: [specific issues], tests: [N passed]"

Failure: openclaw message send --channel discord --target channel:1496099475838603324 --message "[REVIEW] Agent 2 (ML+backtests) FAILED: [reason]"' \
> "$LOG_DIR/agent2_$TS.log" 2>&1
) &
A2=$!

# ── Agent 3: digital-intern full codebase ────────────────────────────────────
(
cd /home/zeph/digital-intern
claude --model claude-opus-4-7 --permission-mode bypassPermissions --print \
'BEFORE STARTING: Read AGENTS.md if it exists in /home/zeph/digital-intern. Read every file listed below in full before touching anything.

You are doing a systematic code review, bug-fix, test suite, and documentation pass on /home/zeph/digital-intern.

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

## Step 1 — Bug fix pass
Find and fix ALL bugs, logic errors, race conditions, missing error handling, dead code, and quality issues. Be surgical.

IMPORTANT constraints:
- backtest:// URLs and backtest_ sources must NEVER reach live signals or Bloomberg alert formatter (they stay in DB for training only)
- ml_score column is for model predictions; ai_score is for LLM labels only — do not let model predictions pollute ai_score
- score_source column must be set correctly: "llm"/"briefing_boost" for LLM labels, "ml" for model predictions

## Step 2 — Build comprehensive test suite
The PURPOSE of tests is to catch bugs in the code itself. Tests must exercise real business logic and verify correctness — assert specific values, not just "no crash".

Create or update tests/ with pytest tests that would catch real bugs:
- storage/article_store.py: test that get_unalerted_urgent NEVER returns articles with url starting with "backtest://" (this is a critical invariant — if it fails, backtest articles hit production alerts), test that marking an article alerted prevents it from appearing again, test score_source is set to "ml" when update_ml_scores_batch is called (not "llm"), test CRUD with in-memory SQLite
- watchers/urgency_scorer.py: test that an article with kw_score=9.5 is correctly classified as urgent, test that kw_score=3.0 is NOT urgent, test that the scorer does not mark articles as urgent if they are already alerted
- ml/features.py: test that feature vector has exactly 15 extra dims beyond TF-IDF, test that ticker_mention_density is correctly zero for articles with no portfolio tickers, test that days_since_published is 0 for a just-published article and ~1 for one published 24h ago
- ml/model.py: test that the model output for relevance head is always in [0, 10], test that urgency head is in [0, 1], test that a zero-input tensor does not produce NaN outputs
- ml/trainer.py: test that _fetch_training_data excludes rows with score_source="ml" (model must not train on its own predictions), test that sample weights are higher for high-relevance articles than low-relevance ones

Use in-memory SQLite for store tests. Mock all external calls.

Run tests: cd /home/zeph/digital-intern && python3 -m pytest tests/ -v 2>&1 | tail -30

Fix any failures before proceeding.

## Step 3 — Write/update AGENTS.md
Create or update /home/zeph/digital-intern/AGENTS.md with:
- Architecture overview (workers, data flow from collection to alert)
- Critical invariants (backtest isolation, ml_score vs ai_score separation)
- How to run the daemon
- How to run tests: "cd /home/zeph/digital-intern && python3 -m pytest tests/ -v"
- Worker descriptions and their roles
- How the ML training pipeline works (label flow, weighting)
- Common failure modes and debugging

## Step 4 — Verify
python3 -c "import sys; sys.path.insert(0,\".\"); from storage import article_store; from ml import features, model; print(\"imports OK\")"
python3 -m pytest tests/ -v 2>&1 | tail -20

## Step 5 — Commit ONLY if you made real changes
IMPORTANT: digital-intern has an auto-commit daemon. Before touching anything:
  - Run: git status
  - If there are uncommitted changes NOT made by you (e.g. config/sources.json, data/, logs/), do NOT stage them. Leave them exactly as-is.

Run: git diff --stat HEAD

If the diff is empty, or contains ONLY whitespace/comment changes, or ONLY AGENTS.md edits with no code fixes, or you found no real bugs:
  - Do NOT commit anything
  - Send: openclaw message send --channel discord --target channel:1496099475838603324 --message "[REVIEW] Agent 3 (digital-intern) NO-OP — everything already correct, [N] tests pass"
  - EXIT

If you made real code fixes or added meaningful new tests that did not previously exist:
  - Stage ONLY the specific .py and test files you changed (never git add -A, never stage *.json, config/, data/, logs/)
  - Run: git diff --staged (verify only your intentional changes are staged)
  - git commit -m "fix: [specific description of what was actually broken]"
  - git push
  - Send: openclaw message send --channel discord --target channel:1496099475838603324 --message "[REVIEW] Agent 3 (digital-intern) done — fixed: [specific issues], tests: [N passed]"

Failure: openclaw message send --channel discord --target channel:1496099475838603324 --message "[REVIEW] Agent 3 (digital-intern) FAILED: [reason]"' \
> "$LOG_DIR/agent3_$TS.log" 2>&1
) &
A3=$!

# ── Agent 4: feature development + user-perspective brainstorming ────────────
(
cd /home/zeph
claude --model claude-opus-4-7 --permission-mode bypassPermissions --print \
'BEFORE STARTING: Read AGENTS.md in both /home/zeph/paper-trader and /home/zeph/digital-intern if they exist. Then read the dashboards, strategy, and ML files to build a complete mental model of the system BEFORE implementing anything.

You are a senior product engineer taking full ownership of this trading stack. Your job is creative feature development and user-perspective testing.

Repos:
- /home/zeph/paper-trader   (paper trading engine, ML scorer, backtests, Flask dashboard :8090)
- /home/zeph/digital-intern (news collector, AI scorer, Bloomberg alerts, chat API :8080)
- /home/zeph/unified_dashboard.py (reverse proxy :8888 — /intern/, /trader/, /ops/)

## Step 1 — READ ALL DOCUMENTATION FIRST
Before writing a single line of code:
- Read /home/zeph/paper-trader/AGENTS.md (if exists)
- Read /home/zeph/digital-intern/AGENTS.md (if exists)
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
- paper-trader: cd /home/zeph/paper-trader && python3 -m pytest tests/ -v
- digital-intern: cd /home/zeph/digital-intern && python3 -m pytest tests/ -v

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
RUN_LOG="/home/zeph/paper-trader/data/run_log.md"
echo "" >> "$RUN_LOG"
echo "## $TS" >> "$RUN_LOG"
echo "- Agents: core, ML+backtests, digital-intern, feature-dev" >> "$RUN_LOG"
echo "- Logs: $LOG_DIR/agent{1,2,3,4}_$TS.log" >> "$RUN_LOG"
