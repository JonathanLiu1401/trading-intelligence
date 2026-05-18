# AGENTS.md — paper-trader

Companion to `CLAUDE.md` aimed at coding agents that touch this repo
during automated review / fix cycles. Where `CLAUDE.md` documents the
*system*, this file documents the *workflows*.

## Repository layout (quick reference)

- `paper_trader/runner.py` — live trader main loop. **Single-instance guard** (`_acquire_singleton_lock`, `fcntl.flock` on `data/paper_trader.runner.lock`, invariant #19) refuses to boot a second trader on the same paper book — `main()` exits before the store/dashboard/ONLINE-ping when the lock is held by a live process; fails OPEN (degraded → continue) if the lock plumbing is unusable. Auto-recovery circuit breaker scoped to its own child claude subprocesses (`pkill -P os.getpid()`, invariant #18). Hourly / daily-close markers are restart-durable via the atomic `data/runner_state.json` sidecar (invariant #6)
- `paper_trader/strategy.py` — live Opus decision engine + watchlist (now injects the behavioural self-review mirror into the prompt). `_portfolio_snapshot` emits `stale_mark` per position and `_build_payload` annotates a `[STALE MARK …]` suffix so a missing-price mark (`current_price==avg_cost`, P/L $0.00) is not read by Opus as a genuine flat position (commit `f834c93`, review pass #4; advisory only, invariants #2/#12)
- `paper_trader/analytics/self_review.py` — canonical behavioural mirror; composes trade_asymmetry + capital_paralysis + open_attribution, fed into the live prompt **and** served at `/api/self-review`
- `paper_trader/analytics/risk_mirror.py` — third advisory mirror (after self_review + track_record): composes `build_churn` + `build_correlation` **verbatim** (single source of truth #10) into a compact `prompt_block` on the trader's *structural* risk — how concentrated the book is and how much it churns (the 2026-05-17 live pathology: ~$973 / 16.7% win-rate, 60%+ one-sector, 0.52-day median hold). No price history is fetched on the hot decision path (a per-position yfinance call is a live-cycle latency/flake risk); without it `build_correlation` is `INSUFFICIENT` and its headline is the bare "verdict withheld" sentence, so the mirror surfaces the weight-based concentration (`top_weight_pct`/`weight_hhi`/`effective_positions_naive`, computed from `market_value` unconditionally) and only uses the richer ρ headline when a caller supplies `price_history`. Observational only, never gates (invariants #2/#12 — the self_review precedent); `_safe`-wrapped so a builder fault is "no block this cycle", never "no decision". Wired into `decide()` + `_build_payload(... risk_mirror_block=)` (rendered after the track-record section); applies on next paper-trader restart. Locked by `tests/test_risk_mirror.py`
- `paper_trader/analytics/tail_risk.py` — left-tail / downside-shape diagnostic (the upside-heavy surface had none): historical 95/99% 1-day VaR (nearest-rank), positional expected-shortfall CVaR (float-robust — a value-threshold filter silently drops float-equal `-0.10` ties and halves the tail), population annualised vol & downside deviation, Fisher-Pearson population skew, worst/best day, max consecutive down-day streak, Ulcer index. Daily series resampled **byte-identically** to `dashboard.analytics_api`'s `by_day` loop (single-source-of-truth #10 spirit; vol `/n` matches its Sharpe, downside-dev `/n` matches its Sortino). Honesty-gated `NO_DATA`/`INSUFFICIENT`(<`MIN_RETURNS`=20)/`OK` (the `build_correlation` precedent — live book is 5 days so it correctly reads INSUFFICIENT until history matures). Served at `/api/tail-risk` **and** folded into `/api/analytics` as an additive `tail_risk` key so the digital-intern analyst chat inherits it. Observational only — never gates Opus, never injected into the decision prompt (invariants #2/#12). Locked by `tests/test_tail_risk.py` (hand-pinned discrete metrics + independent-impl cross-check for vol/skew) and `tests/test_core_analytics.py::TestTailRiskIntegration` (endpoint↔builder no-drift, additive-key contract)
- `paper_trader/analytics/event_calendar.py` — **forward** scheduled-event awareness fed into the live prompt (the mirrors above are all *backward*-looking; this is the one thing a discretionary desk tracks that the engine was fully blind to: upcoming **earnings**). `build_event_calendar(positions, names_in_play, calendar_path=None, now=None, horizon_days=14)` reads digital-intern's `data/earnings_calendar.json` snapshot **directly from disk** — explicitly **not** the `:8080 /api/earnings` endpoint (a network hop on the 60s decision cycle is the documented hang/latency hazard; the `signals.py` filesystem precedent). `_pick_freshest` selects the newest-`as_of` readable candidate across USB/repo/legacy paths (the `signals._db_path` freshness discipline, invariant #15). `days_away` is **recomputed** from `earnings_date` vs `now` (a stale snapshot still yields accurate timing — the digital-intern `api_earnings` rule mirrored verbatim, single source of truth #10), past events (`< -0.5d`) dropped, and each is tiered against the held book exactly as `/api/earnings-risk`: `HELD_IMMINENT` (held & ≤3d), `HELD_SOON` (held & within horizon), `WATCH` (in-play, not held; dropped beyond `horizon_days` as prompt noise — a *held* name's print is never hidden regardless of distance). Observational only, never gates (invariants #2/#12 — the self_review/risk_mirror precedent); `_safe`-style end-to-end so a missing/stale/corrupt/unparseable snapshot degrades to one honest line, **never** an exception that sinks a trading cycle. Served at `/api/event-calendar` (prompt↔endpoint parity — the existing network-sourced `/api/earnings-risk` left untouched, a different concern). Wired into `decide()` + `_build_payload(... event_calendar_block=)` (rendered after `risk_mirror`, before `WATCHLIST PRICES`); applies on next paper-trader restart. **Load-bearing scope:** `decide()` passes held ∪ the **full WATCHLIST** — deliberately **not** the lean `_names_in_play` set the quant / track-record blocks trim to. Those blocks are large per-ticker so they bound prompt length; an earnings event within the 14d horizon is rare (≈0–3 across all 50 names) so there is no bloat to bound, and `WATCHLIST[:5]` excludes most names (e.g. NVDA) — narrowing to `_names_in_play` would silently re-create the exact blind spot this closes (Opus buying a watchlist name the day before its print) **and** break the `/api/event-calendar` parity claim. Do not "consistency-fix" it to `_names_in_play`. Locked by `tests/test_event_calendar.py`
- `paper_trader/signals.py` — live news signal queries against digital-intern's articles.db
- `paper_trader/market.py` — yfinance wrapper + NYSE session calendar
- `paper_trader/store.py` — SQLite store (portfolio, trades, positions, decisions, equity_curve)
- `paper_trader/reporter.py` — Discord output via openclaw. `send_hourly_summary` / `send_daily_close` now append `_behavioural_block()` — the `build_trader_scorecard` verdict-alignment synthesis composed **verbatim** (single source of truth, invariant #10; same store reads as `/api/scorecard`) so the operator who lives in Discord sees the ~24 builders' synthesis without opening the (stale) dashboard. Observational only, no caps (invariants #2/#12 — the `self_review`/`scorecard` precedent). NO_DATA/ERROR suppressed; a builder/store fault degrades to *no block*, **never** *no summary* (the reporter failure contract). Applies on next paper-trader restart (the documented pattern for every recent feature). **Also appends `_session_block(store, window_h, label)`** (2026-05-17) — a compact "what the desk actually did this 1h / 24h" block: the decision-activity mix (`filled / hold / no-dec / blocked`, classified from the free-text `decisions.action_taken` via `_classify_decision_outcome` — bucket order is load-bearing so a `→ FILLED`/`→ BLOCKED` verb line is not misread as `hold`), the best/worst open mover by `unrealized_pl` (`_movers`; single position → one line via object identity), and the portfolio-vs-SPY window delta (`_window_delta`; `alpha_pct` only when both legs resolve, missing `sp500_price` degrades to port-only). All composed from existing store reads — no new state, observational only, same failure contract (store/compute fault → `""`, never an exception). The cutoff is a lexically-comparable UTC isoformat string (the `signals.py` `first_seen` pattern). Answers the trader's "did the bot do anything, and am I beating SPY this window?" from Discord without opening the (often slow/stale) dashboard. Locked by `tests/test_core_reporter.py` (`TestClassifyDecisionOutcome` / `TestActivityCounts` cutoff-inclusive boundary / `TestMovers` identity / `TestWindowDelta` exact port/spy/alpha + spy-missing degrade / `TestSessionBlock` end-to-end on a real temp Store + hourly-summary integration). **Also (2026-05-17): `send_daily_close` emits an *additive* true-realized-P/L line** — `Realized P/L (today, N round-trip(s) closed, WW/LL)  $±X` — driven by `_realized_pl_today()`, which consumes `build_round_trips` (invariant #10, no re-derived P&L) filtered to round-trips whose `exit_ts` is today (UTC). It answers "what did I actually lock in today?", distinct from the pre-existing **cash-flow-basis** line (a BUY-only day reads as a large negative there — correct-by-disclosure, so that line is left untouched, not reinterpreted). Same failure contract: any fault drops just this one line (`None`), never the report. A position merely opened today does not count; an old-open/today-close trip is attributed to today because `build_round_trips` pairs BUY→SELL in ledger order (deep `recent_trades(5000)` window passed so the open leg is in scope). Locked by `tests/test_core_reporter.py::TestSendDailyCloseRealizedRoundTrips` (exact `$+70.00` on a 2-closed/1-open NVDA+MU+AMD ledger with `1W/1L`; no-line-when-nothing-closed; singular-grammar). **Also (2026-05-17, review pass #4): `_portfolio_lines` appends `⚠ STALE` when a position carries `stale_mark=True`** — additive only, `open_positions()` table rows lack the key so the existing Discord path is byte-identical (a genuinely flat $0.00 is never falsely flagged). Locked by `tests/test_core_reporter.py::TestPortfolioLines`
- `paper_trader/dashboard.py` — Flask dashboard on :8090
- `paper_trader/backtest.py` — backtest engine, `_ml_decide`, indicators
- `paper_trader/ml/decision_scorer.py` — MLP that gates trade conviction
- `run_continuous_backtests.py` — long-running training loop
- `tests/` — pytest suite (all offline, all deterministic)

---

## Core (live trader) domain

### Architecture & data flow

One cycle of the live trader (`paper_trader/runner.py::_cycle`):

```
runner._cycle()
  └─▶ strategy.decide()
        ├─ market.is_market_open()                      (NYSE hours + 2026 holidays)
        ├─ _portfolio_snapshot(store)                   (mark-to-market every open position)
        ├─ signals.get_top_signals(20, hours=2, ≥4.0)   (live-only DB filter)
        ├─ signals.get_urgent_articles(minutes=30)
        ├─ signals.ticker_sentiments(WATCHLIST, hours=4)
        ├─ market.get_prices(WATCHLIST + futures + ^GSPC)
        ├─ get_quant_signals_live(...)                  (RSI / MACD / BB / momentum, 5-min cached)
        ├─ _build_payload(...) → SYSTEM_PROMPT          (single string)
        ├─ _claude_call(...) → JSON                     (subprocess: claude --print --permission-mode bypassPermissions)
        ├─ _parse_decision(...)                         (strip ```json fences, raw_decode first {…})
        ├─ _enforce_risk_pre_trade(...)                 (only blocks SELL beyond held qty)
        ├─ _execute(...)                                (BUY / SELL / BUY_CALL / BUY_PUT / SELL_CALL / SELL_PUT / HOLD / REBALANCE)
        ├─ store.record_decision(...) / store.record_equity_point(...)
        └─ return summary dict
  └─▶ if FILLED: reporter.send_trade_alert(...) + reporter.send_decision_log(...)
  └─▶ _maybe_hourly() + _maybe_daily_close()
  └─▶ sleep OPEN_INTERVAL_S (1800s) or CLOSED_INTERVAL_S (3600s)
```

`_portfolio_snapshot` is called twice in `decide()` — once before the trade
(input to the prompt) and once after (so the equity_point reflects post-trade
mark-to-market). The two calls keep the DB's `positions_json` and `total_value`
consistent through the cycle.

### How to run the paper trader

```bash
cd /home/zeph/paper-trader

# Foreground (logs to stdout)
python3 -m paper_trader.runner

# Under systemd
systemctl --user start paper-trader   # see paper-trader.service
journalctl --user -fu paper-trader

# Dashboard only (no decision loop)
python3 -c "from paper_trader.dashboard import run; run(host='0.0.0.0', port=8090)"
```

The runner starts a daemon thread for the Flask dashboard on `:8090` and
posts a `**PAPER TRADER ONLINE**` ping to Discord on first boot.

### How to run tests

```bash
cd /home/zeph/paper-trader && python3 -m pytest tests/ -v
```

All tests are offline — yfinance, Discord, and the digital-intern DB are
mocked. The `tests/conftest.py` autouse fixture redirects backtest paths to
a tmp directory; core tests use their own `fresh_store` fixture that points
`store.DB_PATH` at `tmp_path`.

Core tests live in `tests/test_core_*.py` — one file per module under
review:

| File | What it asserts |
|------|-----------------|
| `test_core_store.py` | cash bookkeeping, position upsert/blend/close, trade & equity ordering |
| `test_core_market.py` | weekend / pre-open / after-close / holiday gating, price-cache TTL, option chain lookup, futures 30s bucket lru_cache, **`get_prices` bulk-download seam** (previously **zero** coverage of the actual `yf.download` branch — only empty/full-cache short-circuits were tested): the load-bearing `len(missing)==1` switch between yfinance's flat-columns single-ticker frame (`data["Close"]`) and the multi-ticker per-ticker MultiIndex (`data[t]["Close"]`) with **real `pandas` frames** (a `MagicMock` would pass even if the branches were swapped), all-NaN-Close → per-ticker `get_price` fallback, missing-ticker-column KeyError → per-ticker fallback, whole-`download`-raises → per-ticker fallback, unresolvable → key present/`None` value, partial-cache fetches only the uncached symbol, full-cache never calls `download`; **`get_options_chain` nearest-DTE seam** (zero prior coverage) — picks the expiry with minimum `abs(date−(today+target_dte))` **even when it is not first listed**, `.head(30)` caps each side, no-expiries → `None`, yfinance-raises → `None` |
| `test_core_signals.py` | top-signal score threshold + sort order, backtest-row filter, urgent ai_score=NULL coercion, ticker regex word-boundary, **single-ticker `get_ticker_sentiment` seam** (a DISTINCT path from the covered bulk `ticker_sentiments` — its own compiled `(?:\$|\b)TKR\b` regex + avg/max/n/urgent aggregation, **zero** prior coverage): no-DB → zeroed dict (never raises), exact `avg_score`/`max_score`/`n`, `urgent` counts only `urgency≥1`, unmentioned ticker → zeroed dict, **"AMDOCS" must not match "AMD"** (the substring-leak regression the bulk path also locks via "MUSE"≠"MU"), `$AMD`-in-body matches, live-only clause excludes `backtest://`/`backtest_*` rows; **`get_ml_predictions` seam** (zero prior coverage; `ml.inference` faked via `sys.modules`, fully offline): import-fail → `[]`, explicit empty input short-circuits before scoring, `None` → `get_top_signals(30,h=6,min=0)` default (sentinel-identity asserted) → empty default → `[]`, `score_articles` raising → `[]`, the `zip(articles, scores)` body **truncates to the shorter** (2 articles + 1 score → 1 row), absent `tickers` key → `[]`, exact field mapping; **`get_historical_signals` gzip-fallback reader** (missing-file → `[]`; strict `< min_score` threshold incl. the `== min_score` KEPT boundary; `score`/`ai_score` `or`-fallback incl. `score:0` → ai_score; `limit` caps the moment `len(out) ≥ limit`; corrupt-JSON / non-numeric-score / blank lines skipped while reading **continues** — a `<`→`<=` or `continue`→`break` regression fails loudly); **freshness-aware `_db_path()` resolver (invariant #15)** — `TestChoosePure` (tie→LOCAL, fresher-local/fresher-usb wins, single-candidate, both-unreadable→LOCAL-first, neither→LOCAL — 6227cd5 LOCAL-first flip), `TestDbPathFreshness` (stale-USB loses to fresh-LOCAL **and** a newer `backtest://` row on USB is excluded from the freshness probe, both-fresh→USB, USB-only, LOCAL-only, candidate-keyed TTL cache), `TestAgeHours` (offset/`Z`/naive/garbage), `TestFeedStatusAndWarn` (split-brain restart signal, all-stale≠split-brain, one-shot WARN dedup), `TestCheckFreshnessCLI` (exit 3/2/0) |
| `test_core_strategy.py` | JSON parse w/ fences + trailing prose, RSI/EMA/MACD math, SELL-exceeds-held blocking, BUY insufficient cash blocking, **ambiguous option close blocking**, **expired-option settlement** (`_option_expired` boundary incl. expiry-day-still-live; `_expired_intrinsic` ITM/OTM/no-underlying; `_portfolio_snapshot` marks expired contracts to intrinsic/0 not premium; live-option transient-None still → avg_cost; `SELL_CALL` on a dead contract settles at intrinsic; **`_portfolio_snapshot` total_value = cash + Σ position market_value across a mixed stock+option book**), **`_stdev_live` population-stdev seam** (`n < 2` → exact `0.0` the `if sd20 > 0` caller-guard relies on; `n=2` computes ÷n not ÷(n-1); constant series → `0.0` via the full variance path; textbook set → exact `2.0` locking `/n` against a `/(n-1)` regression that would silently shift every `bb_position`), **`_format_quant_signals` prompt-block seam** (empty dict → the `(no quant signals available)` sentinel; `_pct` vs `_v` field coercion — momentum/52w use `_pct` "{x}%"/"?", rsi/macd/etc use `_v` no-%; rows `sorted` by ticker so a `.items()` regression can't reorder the prompt non-deterministically) |
| `test_core_runner.py` | `_maybe_daily_close` weekend/time gating + once-per-day flag + retry-on-failure, `_maybe_hourly` 3600s gating + retry-on-failure; **`TestKillStaleClaude`** circuit-breaker pkill is `-P os.getpid()`-scoped (host-wide-broadcast regression lock, invariant #18) **and** still model-anchors Opus+Sonnet; **`TestRunnerStatePersistence`** restart-durable markers (invariant #6) — sidecar IO contract + the no-double-close / no-starved-hourly exact-behaviour locks; **`TestSingletonLock`** single-instance guard (invariant #19) — real `fcntl.flock` acquire/busy/release-on-death/degraded-open + the `main()` busy⇒`SystemExit(1)`-before-`get_store` / degraded⇒continue wiring locks |
| `test_core_runner_cycle.py` | **`_cycle()` report-dispatch fan-out** — previously **zero** direct coverage despite real branching: FILLED gates BOTH trade-alert AND decision-log; HOLD/NO_DECISION/BLOCKED/missing-`status` stay silent **and never query the store** (outer-guard short-circuit asserted via a recording `_FakeStore`); `auto_exits` is an orthogonal `_send` channel independent of the FILLED gate (dead-today-on-purpose per invariant #12 — locked so re-enabling is deliberate, kept per the "do not delete as unreachable" note); the `if trades and status==FILLED` guard (empty `recent_trades(1)` → no alert but decision-log still fires); every reporter fault swallowed (daemon-loop survival, via `monkeypatch` so `boom` can't leak into other modules' reporter import) |
| `test_core_reporter.py` | openclaw missing → False, timeout/nonzero exit → False, trade alert + decision log + portfolio line formatting, **daily-close P/L baseline label tracks `_INITIAL_EQUITY` not a hardcoded `$1000`**, **`send_daily_close` `pnl_real` cash-flow sign (SELL\* credits / BUY\* debits) incl. the option ×100 multiplier via `store.record_trade`** (exact `$-400.00` on a mixed stock+option same-day ledger — a sign flip → `+400.00`, a dropped ×100 → `-449.50`), **`_behavioural_block` composes the scorecard state/headline/focus/concordance verbatim** (no re-derived verdict), suppresses NO_DATA, **returns `""` (never raises) when the builder faults — and `send_hourly_summary`/`send_daily_close` still send the summary regardless** (the "no block, never no summary" failure contract) |
| `test_round_trips.py` | `build_round_trips` arithmetic: simple/partial/re-entry round-trips, option ×100, distinct (ticker,type,strike,expiry) keys, open-lot exclusion, orphan SELL, zero-cost `pnl_pct=None`, negative/unparseable `hold_days`, sub-cent rounding |
| `test_core_analytics.py` | `/api/analytics` end-to-end via Flask test client: exact `win_rate_pct` / `profit_factor` / `avg_holding_days` / `realized_pl_usd` / `n_round_trips` for a fixed ledger; open positions excluded; empty ledger → null metrics |
| `test_core_dashboard_articles_db.py` | Regression lock for invariant #17: `dashboard._articles_db_path()` must resolve through the freshness-aware `signals._db_path()`, not its legacy USB-first existence probe — the discriminating stale-USB-loses-to-fresh-LOCAL assertion, fresher-USB-still-wins, `backtest://`-row excluded from the freshness probe, `None`-when-no-DB (caller contract), and `== signals._db_path()` no-drift |
| `test_core_dashboard_helpers.py` | Pure dashboard helpers with no prior coverage: `_scorer_verdict` 5-way boundary bucketing; `_position_ages_from_trades` open-lot state machine (partial-sell keeps entry, full-sell→re-buy resets, option trades ignored); `_next_market_open` open/close/weekend/holiday arithmetic; `_classify_action` co-pilot selection incl. the **EXIT-before-TRIM** ordering regression and "never BUY without a technical confirm"; **`TestTemplateIdsUnique` — no duplicate static `id="..."` in `dashboard.TEMPLATE`** (regression lock for the `dd-`/`drought-` card-id collision, invariant #14) |
| `test_decision_drought.py` | `build_decision_drought` segmentation: `_classify` fill/block/hold/no-decision; two-drought scenario with exact portfolio/SPY/alpha %; PARALYSIS vs DELIBERATE_HOLD split; ongoing drought detection; `involuntary_alpha_bleed_pct` counts PARALYSIS-only negative alpha; min-reportable-cycles filter; NEVER_TRADED / NO_DATA verdicts; alpha=None when SPY missing |
| `test_news_edge.py` | `build_news_edge`: `_index_at_or_after` exact/gap/overflow; EDGE_CONFIRMED with exact raw means; **SPY-abnormal subtraction is applied** (raw 2.0, spy +1.0 → abnormal 1.0); NO_EDGE on a falling top-band ticker; INSUFFICIENT_DATA under `_MIN_BAND_N`; `$TK`/word-boundary resolution incl. "AMDOCS" must not match AMD; **adaptive reference horizon degrades to 1d when only a 1d forward window exists** (the live-data early-history case) |
| `test_signal_followthrough.py` | `build_signal_followthrough`: exact-value EXPLOITING (acted NVDA+ beats ignored AMD-flat, `selection_edge`/follow-through/per-horizon means) / MISUSING (mirror image, negative edge) / IGNORING_FEED (0% follow-through, ignored-bucket numerics still emitted); **SPY-abnormal subtraction applied** (raw +10 → +8.75 abnormal at 5d under SPY +1/day); per-(decision,ticker) dedup (3 NVDA articles in one window → 1 signal); window boundary (future/stale news excluded); AMDOCS must not match AMD; sample-size honesty (`INSUFFICIENT` keeps numerics, empty → `NO_DATA`); `_fetch_live_articles` excludes planted `backtest://`/`backtest_*`/`opus_annotation*` rows |
| `test_churn.py` | `build_churn`: `NO_DATA`/`EMERGING`/`STABLE` sample-size gate; exact re-entry detection incl. the live NVDA close→re-buy shape (gap_days, `prior_pnl_usd` consumed from `build_round_trips` not recomputed); `REENTRY_WINDOW_DAYS` boundary inclusive **and** one-second-past exclusive; distinct-names→zero re-entries; `reentry_events` sorted fastest-first; both CHURNING paths (≥25% re-entry rate, and fast-cadence with zero re-entries); BUY_AND_HOLD; ACTIVE_TURNOVER between the lines; sub-day loss-concentration exact (= round-trips' own negative-`pnl_usd` sum, single source of truth #10); zero-span book → cadence `None` (no divide-by-zero); all-winners → concentration `None` |
| `test_thesis_drift.py` | `build_thesis_drift`: `NO_DATA` empty; INTACT when up & signals benign; BROKEN via −8% pain line regardless of signals **and** via MACD-flip+negative-mom+loss; WEAKENING via soft −3% loss (no signals), hot RSI while green, cold-catalyst heuristic; **opener selection nearest `opened_at` picks the re-entry lot's BUY not the prior closed lot's** (invariant #8); entry reason surfaced **verbatim** (long string equality); missing ledger → reason `None`, `entry_price` falls back to `avg_cost`, no error; cards sorted worst-first with exact counts |
| `test_loser_autopsy.py` | `build_loser_autopsy`: `_classify` failure-mode precedence (KNIFE_CATCH wins over the fast/shallow WHIPSAW arm, `< FAST_HOLD_DAYS` strict & `>= SLOW_HOLD_DAYS` inclusive boundaries, `None` hold/pnl_pct never raises and defaults); strict `pnl_usd < 0` loser convention (a `pnl==0` wash is **not** a loss — invariant #10); verbatim entry/exit reason joined by trade `id` (first BUY / last SELL; blank/whitespace → `None`, missing-id → `None`, never NLP-parsed); aggregates exact (total/avg, median odd **and** even count, ticker-bleed sorted most-negative-$ first, `repeat_offenders` n≥2, deterministic dominant-mode severity tie-break); P&L/cost/proceeds **consumed from `build_round_trips`** on a partial-then-full close (not recomputed); verdict withheld until `STABLE` (n_losers≥`STABLE_MIN_LOSERS`); NO_DATA/NO_LOSSES/EMERGING honesty; never raises on garbage rows |
| `test_hold_discipline.py` | `build_hold_discipline` — the open-book disposition trap (a loser held past the desk's *own* empirical losing-cut time, caught **while it is still happening**, not in a post-mortem). The discriminating lock is **no-drift**: the reference median is asserted **byte-identical** to `build_loser_autopsy(trades)["median_loser_hold_days"]` (composed verbatim, never re-derived — the `risk_mirror` embedded-headline discipline) **and** independently equal to `statistics.median` over `build_round_trips`' own `pnl_usd<0` holds, so a drift in *either* layer fails loudly; winners excluded from the reference. Strict boundary: `age == median` is **within** discipline, `age == median+ε` is overstayed, a *winner* past the median is **never** overstayed (the `is_losing` gate), an unparseable `opened_at` → `age None`/not flagged/no raise. State ladder `NO_DATA`(no open book)→`INSUFFICIENT`(< `MIN_REFERENCE_LOSERS`=3 closed losers — cards+ages still emitted but **nothing flagged & verdict withheld**, the `loser_autopsy` sample-size precedent)→`DISCIPLINED`→`DISPOSITION_DRAG`; exact `disposition_drag_usd` = Σ of the **overstayed** positions' `unrealized_pl` read **directly** (the option ×100 is already baked into that column — never re-derived from `avg_cost×qty`), `worst_overstayed` = most-negative, overstayed cards sort first deterministically, exact headline format. `_safe`: a monkeypatched `build_loser_autopsy` raising degrades to an honest `INSUFFICIENT`/`reference unavailable` (verdict withheld, `reference_state` `ERROR:…`), **never** an exception (the `event_calendar` contract — a diagnostics fault must not 500 the route or kill the close report); a garbage non-numeric `unrealized_pl` coerces to `0.0`, never raises. `TestEndpoint` drives the real `/api/hold-discipline` Flask view on a fresh temp `Store` (seeded controlled-timestamp losing round-trips + an overstayed open lot) → `DISPOSITION_DRAG` with exact `$-at-risk`. `TestReporterLine`: `_hold_discipline_line` returns `""` on NO_DATA/INSUFFICIENT/fault, emits the builder headline verbatim on `DISPOSITION_DRAG`, and `send_daily_close` still sends the whole report when the builder faults ("no block, never no summary") |
| `test_correlation.py` | `build_correlation`: `_returns` chain (a `0`/NaN/non-numeric bar **breaks then continues** — one bad yfinance bar must not zero the series; `pytest.approx` for the float-division results); `_pearson` exact `±1.0` under a positive/negative affine map, the hand-computed `0.6` fixture, flat-series → `None` (never a fabricated 0), length-mismatch/too-short → `None`; options flagged & skipped; single-name **and** sub-`MIN_RETURNS` series → `INSUFFICIENT` (verdict withheld, numerics where possible); `CONCENTRATED` (identical returns ρ=+1 → `effective_independent_bets`=1.0) / `DIVERSIFIED` (ρ=−1 → eff_bets `None` honest-undefined; constructed ρ=0 → eff_bets 2.0) / `SINGLE_NAME_RISK` overrides correlation when top weight ≥ `DOMINANT_WEIGHT` / `MODERATE` band; `weight_hhi` & `effective_positions_naive` exact (60/40 → HHI 0.52); unequal-length series aligned to the common tail; never raises on garbage |
| `test_risk_mirror.py` | `build_risk_mirror` — the third advisory mirror (concentration + churn) fed into the live prompt. Composes `build_churn`/`build_correlation` **verbatim** (single source of truth #10): the embedded churn headline is asserted **byte-identical** to `build_churn(reversed(trades)).headline` so an inline re-derivation that drifts from `/api/churn` fails loudly. The discriminating lock is **no "verdict withheld" leak**: with empty `price_history` (the live `decide()` path) `build_correlation`'s headline collapses to the bare "correlation verdict withheld" sentence, so the mirror MUST surface the weight-based concentration (`top_weight_pct`/`weight_hhi`/`effective_positions_naive`, all computed from `market_value` regardless of price history) instead — RED if the headline is pasted through. Also: the rich ρ headline **is** used verbatim when real price history makes `state==OK` (CONCENTRATED "moves as one", not the weight-pending fallback); options-only / cash book → concentration line omitted (undefined, not faked); empty book → honest one-line fallback (the self-review precedent), never an empty section; a monkeypatched builder fault degrades to "that line missing", never an exception (the `_safe` contract — a diagnostics fault must not sink a live trading cycle); `_build_payload` renders the block **after** the track-record section and **before** `WATCHLIST PRICES`, and `None` renders no stray text |
| `test_event_calendar.py` | `build_event_calendar` — the forward earnings-awareness block. The discriminating lock is **`days_away` recomputed from `earnings_date` vs injected `now`, not read from the file's stale field** (the file's `days_away` is set to garbage `999.0` in the fixture; a regression that trusts it tiers NVDA wrong → RED). Also: the `HELD_IMMINENT` `<= 3` day boundary is exact (`3.0`→IMMINENT, `3.01`→SOON, the api_earnings rule); an in-play-not-held name is `WATCH`, a neither-held-nor-in-play name is dropped (prompt stays lean); a **past** event (`-1d`) never leaks; a distant `WATCH` (>horizon) is dropped but a distant **held** name's print is always kept; sort is tier-rank then soonest-first; a missing **and** a corrupt file both degrade to an honest non-empty line with `source_ok=False` and **no raise** (the `_safe` contract — a diagnostics fault must not sink a live cycle); `_pick_freshest` picks the newer-`as_of` candidate order-independently and skips unreadable ones; the block carries the autonomy preamble and **no directive verb** (the observational invariant #2/#12 contract); valid-but-empty calendar → honest "no scheduled earnings" line, not a crash; `_build_payload` renders it **after** `risk_mirror` and **before** `WATCHLIST PRICES`, `None` renders no stray text; and `TestEventCalendarEndpoint` drives the real `/api/event-calendar` Flask view on a fresh temp `Store` (held NVDA via `upsert_position`, on-disk snapshot redirected) — route→builder→store wiring returns the imminent tier, not a 404/500 |
| `test_dashboard_threaded.py` | invariant #7 dashboard-concurrency lock. `test_run_passes_threaded` regression-locks the `dashboard.run` call site (monkeypatched `app.run`): `threaded=True` is passed **and** the existing `debug=False`/`use_reloader=False` hardening is preserved (RED before the 2026-05-17 fix — the kwarg was absent, so the in-process Werkzeug dev server served one request at a time and a single slow yfinance-backed endpoint head-of-line-blocked every concurrent panel / `/api/chat` fan-out / `:8080→:8090` cross-fetch). `test_threaded_server_parallelizes` is the behavioural lock: an independent ephemeral-port `make_server(..., threaded=True)` with a 0.4s route serves 4 concurrent requests in well under the serial 1.6s — so a future swap to a non-threaded WSGI entry point that silently drops the property is caught even though the monkeypatch lock still passes. Offline, deterministic, no real `:8090` bind. Found by user-perspective testing, not code review |
| `test_core_state_swr.py` | `/api/state` stale-while-revalidate + the main-page `refresh()` guard (2026-05-17). End-to-end through the real Flask view on a fresh temp `Store`: cold build returns the full shape + `cached:false`/`cache_age_s` honesty keys; a warm hit within the 15s TTL serves the **stale** payload and does **not** re-read the store even after the underlying portfolio/trades change (the latency win, asserted as behaviour); the documented "inert under pytest unless `_SWR_TEST_FORCE`" contract holds (no honesty keys, live reflection — keeps the other `/api/state`-shaped exact-value tests isolated). `TestRefreshGuard` is a static lock on `dashboard.TEMPLATE` (the `TestTemplateIdsUnique` discipline, comments stripped so it reasons about executable JS): the `/api/state` fetch is wrapped in try/catch and the `!r.portfolio`/`r.warming`/`r.error` early-return precedes the first `r.portfolio.total_value` deref — RED before the fix, when `refresh()` was the lone `refresh*` fn with no guard and any transient `/api/state` body (it has 500'd 28× in prod — the `store.get_portfolio` shared-connection note) froze the whole page |

### Key invariants and constraints

1. **Live trader uses Claude Opus 4.7** — `MODEL = "claude-opus-4-7"` in
   `strategy.py`. The whole prompt is tuned around Opus's reasoning. Do not
   downgrade to Sonnet without an explicit decision.

2. **No hard risk limits** — `_enforce_risk_pre_trade` only checks that a
   SELL doesn't exceed held quantity. There are no position-size, leverage,
   or daily-loss caps. The system prompt grants Opus full autonomy. If a
   reviewer "fixes" this by adding caps, it changes the system's identity —
   discuss before merging.

3. **Live-only DB filter** — every read in `signals.py` against digital-intern's
   `articles.db` includes:
   ```sql
   AND url NOT LIKE 'backtest://%'
   AND source NOT LIKE 'backtest_%'
   AND source NOT LIKE 'opus_annotation%'
   ```
   Mirror this in any new query. The dashboard's `_ticker_news_pulse` already
   does. Forgetting the filter contaminates live signals with the engine's
   own backtest annotations.

4. **Ambiguous option closes are rejected** — when `SELL_CALL` / `SELL_PUT`
   matches more than one open contract and `strike`/`expiry` are unspecified,
   `_execute` returns `BLOCKED` with the open legs in the detail string.
   Picking the "first match" silently could exit the wrong leg.

5. **openclaw env key invariant** — the Discord channel ID lives directly in
   `reporter.DISCORD_CHANNEL`. Do NOT add an env-key dependency or move the
   channel ID into `openclaw.json` — the current setup intentionally hard-codes
   the channel so a missing config doesn't silently route messages elsewhere.

6. **Hourly/daily close idempotence** — `_maybe_hourly` and `_maybe_daily_close`
   only advance their "last sent" markers on actual send success. A transient
   openclaw failure retries on the next cycle rather than silently skipping
   the hour or day. If a reviewer adds a "fire-and-forget" path, this property
   breaks. `_maybe_daily_close` also skips weekends **and** NYSE full-holiday
   closes (`market.NYSE_HOLIDAYS_2026`) — both guards `return` *before* touching
   `_daily_close_sent_for`, so the flag never advances on a non-trading day and
   the next real trading day still gets its close report. Locked by
   `tests/test_core_runner.py::TestMaybeDailyClose` (incl.
   `test_does_not_fire_on_nyse_holiday`).

   **Restart-durable (2026-05-17).** `_daily_close_sent_for` /
   `_last_hourly` were module globals lost on every restart, and the
   runner restarts often (a `/api/build-info` `stale` bounce, systemd,
   the circuit breaker). That broke this invariant *across* a restart in
   two trader-visible ways: a runner bouncing more often than hourly
   **never** sent an hourly summary (every boot re-anchored the 1h
   clock), and a post-16:05 NY restart **double-posted** the DAILY CLOSE.
   Both markers now persist to an **atomic** `data/runner_state.json`
   sidecar (tmp + `os.replace`, every IO swallowed) written on each
   successful send; `main()` rehydrates them *after* the boot-anchor
   default, so a fresh first-ever start is byte-for-byte unchanged while
   a restart restores the real last-hourly instant (an overdue summary
   fires this cycle; a recent one still waits) and the close-sent date
   (no dup). Deliberately a sidecar, **not** a `store.py` table — SCHEMA
   is load-bearing (#13) and this is single-writer best-effort that must
   degrade to today's in-memory-only behaviour, never crash the loop.
   Locked by `tests/test_core_runner.py::TestRunnerStatePersistence`
   (11 tests: IO contract missing/corrupt/non-dict→{}, atomic
   no-leftover-tmp, IO-error swallowed; rehydrate no-sidecar/both/
   corrupt-skip; and the two exact-behaviour bug locks — restart-after-
   close does not double-post, an overdue hourly fires post-restart, a
   <1h one does not). The autouse fixture redirects `_STATE_PATH` to tmp
   so no runner test writes the real sidecar (offline invariant).

7. **`paper_trader.db` uses WAL** — any external reader must use
   `PRAGMA journal_mode=WAL` or open the file as `file:...?mode=ro` to avoid
   lock contention with the live writer.
   *Dashboard concurrency (doc-truth correction, 2026-05-17 — the prior text
   here said dashboard reads were unlocked / "not strictly connection-safe" /
   "a proper fix would give the dashboard its own read-only connection"; the
   code has since superseded that):* the in-process Flask dashboard runs in a
   daemon thread (`runner._start_dashboard`) and shares the **same** `Store`
   singleton (`sqlite3.Connection`, `check_same_thread=False`) as the runner —
   but **every read now holds `Store._lock`**, not just writes. See the
   load-bearing NOTE at `store.py::Store.get_portfolio` ("every read below
   MUST hold self._lock … shared between the runner's writer thread and the
   Flask dashboard **thread(s)**" — plural). The shared connection is never
   used by two threads at once because `_lock` brackets every `.execute()`;
   the slow yfinance-backed endpoints use their own per-request
   `sqlite3.connect(file:…?mode=ro)`. The store is therefore already hardened
   for a multi-threaded dashboard. **`dashboard.run` now passes
   `threaded=True`** (it previously did not — `app.run` defaults
   `threaded=False`, so the dev server served one request at a time and a
   single slow endpoint head-of-line-blocked every concurrent panel fetch,
   the `/api/chat` ~15-way fan-out, and the `:8080→:8090` cross-fetch behind
   it). Locked by `tests/test_dashboard_threaded.py`. **Per-endpoint latency
   (largely treated):** `threaded=True` removed *cross-request* head-of-line
   blocking; the *per-endpoint* latency concern is now closed by
   `swr_cached` — every slow network endpoint (`/api/correlation`,
   `/api/news-edge`, `/api/source-edge`, `/api/feed-health`,
   `/api/sector-heatmap`, `/api/briefing`, `/api/suggestions`,
   `/api/thesis-drift`, `/api/scorer-predictions`, `/api/data-feed`) **and,
   2026-05-17, the heaviest pure-DB endpoint `/api/state`** (the trader-page
   lifeline — observed 8.7s under concurrent load, the last high-traffic
   gap) is now behind stale-while-revalidate with a bounded cold path. Each
   such cache is its own commit with its own evidence + tests (the
   `/api/state` one is `tests/test_core_state_swr.py`, which also locks the
   `refresh()` warming/error-body guard).

8. **Position uniqueness** — the `positions` table has a *table-wide* UNIQUE
   constraint on `(ticker, type, expiry, strike)` (it is **not** scoped to
   `closed_at IS NULL` — there is no partial index). A second BUY on an
   existing open lot blends the avg_cost; a SELL that zeros out qty marks the
   row closed. A re-BUY after a full close **reactivates the same row** (fresh
   qty/avg_cost/opened_at, marks reset, `closed_at` cleared) — it does *not*
   insert a new row. This is load-bearing: because SQLite treats NULLs as
   distinct in UNIQUE, the old "insert a new row" path only worked for stock
   (NULL strike/expiry); re-entering a previously-closed *option* raised an
   uncaught `IntegrityError` mid-`_execute`, leaving a recorded trade with no
   position and skipping the cash debit + decision/equity write. Locked by
   `tests/test_core_store.py::TestUpsertPosition::test_reopen_option_after_close_does_not_crash`.

9. **Deterministic ordering** — `store.recent_trades`, `recent_decisions`, and
   `equity_curve` order by `(timestamp DESC, id DESC)`. The `id` tiebreaker is
   load-bearing: two writes inside the same microsecond collide on `timestamp`
   alone, and `runner._cycle` reads `recent_trades(1)` immediately after
   `_execute` records a trade — without the tiebreaker `send_trade_alert` could
   post a stale same-microsecond row. `equity_curve` still returns ascending
   `{timestamp,total_value,cash,sp500_price}` (no `id` leaked to callers).
   Locked by `tests/test_core_invariants.py::TestSameTimestampOrdering`.

10. **Round-trip aggregation has one home** — `paper_trader/analytics/round_trips.py::build_round_trips`
   is the single source of truth for closed-round-trip P&L (a round-trip is the
   slice of same-`(ticker,type,strike,expiry)` trades from qty-leaves-zero to
   qty-returns-zero; a re-BUY after a full close starts a new one). `analytics_api`
   (`/api/analytics`) consumes it for `win_rate_pct` / `profit_factor` /
   `avg_holding_days`; do **not** reintroduce an inline copy here or in a future
   trade-attribution endpoint — they drift. `pnl_usd` is rounded to 4dp and the
   win/loss split is strict `> 0`, so a sub-cent artefact reads as a non-win
   (pinned by `tests/test_round_trips.py::TestEdgeCases::test_subcent_pnl_rounds_to_zero`).
   The `/api/backtests/compare` win-rate is a **different** metric (per-fill FIFO
   lot win/loss, stocks only) and intentionally does *not* use this helper.

11. **Scorer honesty is end-to-end** — every panel that surfaces a
   DecisionScorer prediction calls `predict_with_meta()` (never the bare
   scalar `predict()`) and propagates `off_distribution` +
   `raw_pred_5d_return_pct`: `/api/scorer-predictions`, `/api/position-thesis`
   (→ thesis card → unified conviction board), `/api/disagreement`,
   `/api/scorer-confidence`. A clamped ±50 floor must never reach a UI/board
   without its low-trust flag, or a phantom "confident EXIT" pins downstream
   conviction. Locked by `tests/test_scorer_honesty.py`. **The on-disk clamp
   is necessary but not sufficient: a long-running `:8090` process that
   booted before the clamp commit keeps extrapolating to ±700% in memory.**
   `/api/build-info` (`stale: true`) is the canonical signal that a restart
   is required to apply committed scorer/code fixes; locked by
   `tests/test_build_info.py`. **The `:8090` trader page now carries an
   always-on, page-wide red banner** (`#global-stale-banner`, polls
   `/api/build-info` every 60s; new 2026-05-16) that fires whenever
   `stale` **or** `behind > 0` — previously only the unified landing page
   and per-panel `fetchMaybeStale` degradation surfaced this, so a stale
   trader (e.g. the self-review mirror silently not injected, the exact
   live state on 2026-05-16) was invisible from the trader page itself.
   It is purely informational — it changes no behaviour and adds no caps.

12. **One source of truth for the $1000 baseline** — every starting-equity /
   P&L-% denominator must read `store.INITIAL_CASH`, never a hardcoded
   `1000.0`. `reporter._INITIAL_EQUITY`, `dashboard.portfolio_api`
   (`starting_value`), and `dashboard.analytics_api` (Calmar's
   `total_return_pct`) all reference the constant. A literal silently
   desyncs the moment `INITIAL_CASH` moves (fixed in `reporter.py`,
   commit `2a154df`; the analytics Calmar leak fixed in this pass). The
   backtest-side `1000.0` in `backtest_compare`'s empty-curve fallback is a
   *separate* baseline (`backtest.py`'s own `INITIAL_CASH`) and is out of
   scope of this rule. Locked by
   `tests/test_core_analytics.py::TestCalmarBaseline`.

13. **Expired options settle at intrinsic, never at premium** — yfinance has
   no option chain past expiry, so `market.get_option_price` returns `None`
   for a held-to-expiry contract. The old `cur = cur or p["avg_cost"]` in
   `strategy._portfolio_snapshot` then marked a (usually worthless) expired
   contract at its full purchase premium **forever**, never closing it —
   silently inflating `total_value` and every reported P/L. The system
   prompt explicitly tells Opus it "can hold options through expiry", so
   this is reachable *by design*, not an accident. Fixed at two sites:
   `_portfolio_snapshot` (the mark) and `_execute`'s `SELL_CALL`/`SELL_PUT`
   close path. Both now route an expired contract through
   `strategy._expired_intrinsic(ticker, otype, strike)` =
   `max(0, underlying−strike)` (call) / `max(0, strike−underlying)` (put),
   falling back to **0.0** (never avg_cost) when the underlying price is
   unavailable. The `or`→`is not None` change on the mark fallback is
   load-bearing: a legitimate `0.0` intrinsic must survive, and `0.0 or
   avg_cost` would clobber it straight back to premium. `_option_expired`
   uses `<` (an option is live *on* its expiry date).
   **This is a *valuation* fix, not a risk limit.** It does not violate the
   "no hard risk limits / Opus has full autonomy" invariant (#2) — that
   invariant governs *gating decisions*, not *valuing instruments*. Do not
   read this as an autonomy violation and revert it. Full auto-settlement
   (recording a synthetic SELL + closing the row at expiry) was
   *deliberately deferred*: it would make `_portfolio_snapshot` state-
   mutating for every caller (it is currently a pure mark), which is too
   invasive for a surgical pass and risks the parse-retry tests that
   monkeypatch it. The conservative fix removes the phantom-equity harm; an
   expired contract simply marks to its true value and stays an open row
   until Opus closes it (and closing it now also settles correctly). The
   live `paper_trader.db` has had **zero** option positions to date, so
   this is latent, not active — but the bug is real code-path and the test
   suite locks the desired behaviour. Locked by
   `tests/test_core_strategy.py::TestOptionExpired` /
   `::TestExpiredIntrinsic` / `::TestPortfolioSnapshotExpiredOptions` /
   `::TestExecuteCloseExpiredOption`.

14. **`dashboard.TEMPLATE` element IDs must be globally unique** — every
   panel is a separate card in one giant HTML document, and the JS drives
   them with bare `getElementById("…")`, which resolves to the *first*
   element in document order. Two cards sharing an id ⇒ one panel silently
   writes into the other's DOM. This actually happened: the **Decision
   drought drift** card (2026-05-16) reused the **Drawdown anatomy** card's
   (2026-05-15) `dd-` prefix, so `id="dd-card"`/`id="dd-current"` each
   appeared twice — `refreshDecisionDrought()` wrote its status into the
   drawdown card's "current equity" stat and the drought card's own status
   box stayed stuck on "loading…" forever. Fixed by renaming the *newer*
   (intruding) card to a `drought-*` namespace; the original `dd-*` owner
   is left untouched. When you add a card, pick a fresh id prefix — don't
   extend a neighbour's. Locked by
   `tests/test_core_dashboard_helpers.py::TestTemplateIdsUnique`
   (`test_no_duplicate_static_element_ids` would have failed pre-fix).

15. **`signals._db_path()` is freshness-aware, not existence-first.** It was
   `if USB_DB.exists(): return USB_DB` since the initial commit — but the
   digital-intern daemon falls back to writing the **LOCAL** copy when the USB
   mount is unavailable for writes, leaving a USB mirror that keeps
   `exists()`-ing while going day-stale. The live trader then read frozen news
   while every other surface (daemon, unified dashboard — both LOCAL-first)
   read the fresh DB. ~24 builders/endpoints *detected* this split-brain
   (`/api/feed-health`, chat fallbacks) but none root-fixed it. `_db_path()`
   now picks the candidate whose newest **live** article (`_LIVE_ONLY_SQL` —
   so a fresh batch of injected `backtest://` rows on a stale mirror can't win
   it) is most recent; LOCAL is preferred on a tie / when freshness is
   indeterminate (LOCAL is the live daemon's write path — 6227cd5 flipped
   this from the old USB-first default). TTL-cached (120s,
   keyed on the candidate tuple so a monkeypatching test always re-resolves);
   a one-shot stderr WARN fires when the chosen feed is ≥6h stale. **This is a
   data-sourcing fix, not a risk limit — invariants #2/#12 untouched (same
   reasoning as the #13 valuation fix).** **It does not rescue a running
   process:** a `:8090`/runner that booted pre-fix keeps the old resolver and
   reads USB until restart (`/api/build-info` `stale`). New operator CLI
   `python3 -m paper_trader.signals --check-freshness` (offline, no Flask —
   works even when the stale process makes every detector endpoint 404):
   prints each candidate's newest-live age + the freshest/legacy picks, exits
   `3` split-brain (a stale process is blind — RESTART) / `2` whole pipeline
   stale (restart won't help — fix the daemon) / `0` healthy. `feed_status()`
   is the reusable snapshot behind it. Resolver mirrored into digital-intern's
   **vendored** `paper_trader/signals.py` (port-only-the-change rule, Cross-
   system contract); parity locked by digital-intern's
   `tests/test_paper_trader_signals_isolation.py`. Locked by
   `tests/test_core_signals.py` (`TestChoosePure` tie/fresher/single/fallback
   matrix · `TestDbPathFreshness` end-to-end incl. backtest-row exclusion &
   candidate-keyed cache · `TestAgeHours` · `TestFeedStatusAndWarn` ·
   `TestCheckFreshnessCLI` exit codes). Consequence for `/api/feed-health`
   (next bullet).

16. **`/api/feed-health` split-brain is now legacy-vs-fresh.** Because #15
   made `_db_path()` resolve the *fresh* DB, the old "the **resolved** DB is
   stale while a fresher candidate exists" shape can no longer fire for a
   current-code process — that detector would have gone silently dead. The
   endpoint now also passes `signals._legacy_choice()` (the old existence-
   first pick — what a *stale running process* still reads) as `feed[
   "legacy_path"]`/`legacy_newest`; `build_feed_health` flags `split_brain`
   when that legacy pick differs from the fresh resolution and is ≥
   `SPLIT_BRAIN_GAP_H` staler (a pre-fix/stale process is blind → `restart_
   recommended`). The **pure** builder's original `resolved_stale_split` term
   is retained verbatim and is inert unless `legacy_path` is supplied, so the
   four `tests/test_feed_health.py::TestSplitBrain` exact-value fixtures stay
   green **untouched** (proof the locked invariant didn't actually conflict).
   New output keys `legacy_path` / `legacy_newest_age_h`. Only the
   *endpoint* test (`tests/test_feed_health_endpoint.py::test_endpoint_flags_
   blind_split_brain`) changed — its old assertions
   `resolved_path.endswith("usb_…")` literally codified the bug; corrected to
   the post-fix fresh `local_…` + the new `legacy_*` fields (a correction,
   not a weakened test).

17. **`dashboard._articles_db_path()` delegates to `signals._db_path()`.** It
   was the last un-fixed instance of the #15 split-brain: its own legacy
   USB-first existence probe (`if usb.exists(): return usb`) while its
   docstring *claimed* to "Match how paper_trader.signals discovers the
   digital-intern articles.db". So `/api/news-edge`, `/api/source-edge`,
   `/api/signal-followthrough`, `/api/sector-pulse` (via `_ticker_news_pulse`)
   read the **stale USB mirror** while the live trader read the fresh LOCAL
   one — the same documented split-brain, surviving in this one helper. It now
   calls `signals._db_path()` (the freshness-aware single source of truth) and
   returns `None` when the resolved DB does not exist, **preserving the caller
   contract** (`if path is None: <graceful>`) — `signals._db_path()` returns
   LOCAL_DB as its tie/fallback even when nothing exists, so the `.exists()`
   gate is load-bearing. Data-sourcing fix, not a risk limit (invariants
   #2/#12 untouched — same reasoning as #15). Like #15 it does **not** rescue
   a running process: a stale `:8090` keeps the old probe until restart
   (`/api/build-info` `stale`). Locked by
   `tests/test_core_dashboard_articles_db.py`
   (`TestArticlesDbPathIsFreshnessAware` — the discriminating
   stale-USB-loses-to-fresh-LOCAL assertion, fresher-USB-still-wins,
   backtest-row-excluded, `None`-when-missing, and the
   `== signals._db_path()` no-drift lock).

18. **The auto-recovery circuit breaker is scoped to the runner's own
   children.** `runner._kill_stale_claude()` (fired after
   `CONSECUTIVE_NO_DECISION_LIMIT`=5 NO_DECISION cycles) used to run a
   **host-wide** `pkill -f "claude --model claude-opus"` /
   `claude-sonnet`. On this multi-agent box that ERE also matches the
   hourly self-review agents (`scripts/hourly_review.sh` spawns 3×
   `claude --model claude-opus-4-7`), sibling automated-review agents,
   and any operator interactive `claude` session — so a wedged trader
   recovering would SIGTERM **every** Claude process on the machine,
   including the agents that keep the system healthy and one that may
   have just deployed a fix. It is now scoped with
   `pkill -P os.getpid()`: the decision subprocess is always a *direct*
   child of the runner, so `-P` restricts the sweep to exactly what the
   breaker is meant to reap. The model-anchored `claude --model <family>`
   pattern (Opus first, Sonnet fallback — never a bare `claude --print`
   that matches nothing) is **preserved unchanged**. This is a
   collateral-damage fix, not a risk limit (invariants #2/#12 untouched).
   Locked by `tests/test_core_runner.py::TestKillStaleClaude`
   (`test_kill_is_scoped_to_own_child_processes` — RED on a regression
   back to host-wide `["pkill","-f",pattern]`; the prior
   `assert argv[:2]==["pkill","-f"]` literally codified the broadcast
   bug, corrected not weakened, the invariant-#16 precedent;
   pattern-anchoring Opus+Sonnet assertions kept verbatim).

19. **One runner per paper book — the single-instance guard.** Two
   concurrent `runner.py` processes on the same `paper_trader.db` is a
   real, *observed* live pathology (2026-05-17: an orphaned manual launch
   under PID 1 **and** the systemd-managed instance both cycling, so a
   trader saw 2–3 decisions clustered inside a minute then an hour of
   nothing — double-trades, doubled concurrent-`claude` RAM, a raced
   decision/equity log). Nothing in `runner.py` prevented it
   (digital-intern's daemon has a singleton lock; this was the missing
   twin). `main()` now calls `_acquire_singleton_lock()` **first** —
   before `get_store()`, the dashboard thread, or the ONLINE ping — an
   `fcntl.flock(LOCK_EX|LOCK_NB)` advisory lock on
   `data/paper_trader.runner.lock`. `flock` is the robust primitive: the
   **kernel releases it when the holder dies** (crash / SIGKILL / normal
   exit), so a restart never trips over a stale PID file — the exact
   failure a naive pid-file guard introduces. The locked fd is retained
   in the module global `_SINGLETON_LOCK_FH` for process life (closing it
   frees the lock). Three outcomes: `acquired` (hold it, write our PID
   into the file for `cat`-ability), `busy` (another **live** process
   holds it → log the holder PID and `sys.exit(1)` — the *only*
   fail-closed path; a second trader must not even mark-to-market the
   shared book), `degraded` (no `fcntl` / unwritable dir / USB unmounted →
   **continue WITHOUT the guard** and warn — never take down the *sole*
   runner over lock plumbing, the `_save_runner_state` best-effort
   philosophy). **This is a safety guard, not a risk limit** — it gates
   *process startup*, not trading decisions; invariants #2/#12 untouched
   (same reasoning as #13/#15). Like every recent feature it **applies on
   the next paper-trader restart** — it does NOT kill an already-running
   duplicate (an operator must stop the orphan; the guard prevents
   *recurrence*). Locked by `tests/test_core_runner.py::TestSingletonLock`
   (real `fcntl.flock` on a tmp lockfile — a second `open()`+`flock` in
   the same process contends exactly as a second process would: first
   acquire writes the PID; second is `busy` with the holder PID;
   close→reacquire proves no stale-lock-blocks-restart; a file-as-parent
   path degrades open-not-closed; and the two `main()` wiring locks —
   `busy`⇒`SystemExit(1)` *before* `get_store`, `degraded`⇒continues).

   **Degraded self-recheck (2026-05-18, commit `7aa4d85`).** The boot-time
   `degraded` fail-open left a real hole: a runner that booted while the
   USB-backed `data/` dir was transiently unmounted ran guard-less
   *forever*, so a later runner cleanly took the flock and **both
   double-traded** (confirmed live: PID 1255030 no lock fd + PID 1465599
   holds `FLOCK …265831`; `/api/decision-reliability` 27.6% `TIMEOUT_EMPTY`,
   −2.21% involuntary alpha bleed). `_recheck_singleton_lock()` now runs at
   the top of every loop iteration and re-attempts the lock **only from the
   `degraded` state**: `acquired`→upgrade in place (keep the handle);
   `busy`→`sys.exit(1)` (another live trader **confirmed** holding it — the
   redundant degraded runner stands down); still `degraded`→keep running.
   **Invariant #19 is fully preserved: it exits ONLY on a confirmed other
   holder, NEVER on plumbing failure** (a USB flap during normal operation
   must not kill the sole trader). Hard **no-op once `acquired`** — a 2nd
   `open()`+`flock` of the same file in the same process is denied by our
   *own* lock and would mis-read as `busy`, exiting the real holder (the
   load-bearing guard). This is **cooperative self-introspection, not PID
   hunting / a host-wide scan** — no signal is sent to any other process;
   the runner inspects only *its own* lock and *itself* exits. So the guard
   now also self-heals an *already-running* degraded duplicate (within one
   cycle of the lock holder existing), narrowing — though not eliminating
   (a never-locked runner predating this code still needs an operator
   stop) — the "does NOT kill an already-running duplicate" caveat above.
   `runner.singleton_lock_state()` exposes `{status, holder_pid, have_lock,
   degraded}` for `/api/runner-heartbeat` (`singleton_lock` block) and the
   hourly/daily Discord summary (`⚠️ RUNNER DEGRADED`) so a guard-less
   runner self-reports. Locked by
   `tests/test_core_runner.py::TestRecheckSingletonLock`.

### Dashboard API endpoints (port 8090)

All endpoints serve `application/json`. CORS is wide open (`*`) so the
Digital Intern dashboard on `:8080` can cross-fetch.

| Endpoint | Purpose |
|----------|---------|
| `GET /` | HTML — live trader page (portfolio + trades + chart) |
| `GET /backtests` | HTML — backtest grid + equity overlay |
| `GET /api/state` | Portfolio + positions + last 40 trades + last 20 decisions + equity curve. **`swr_cached("state", 15.0)` (2026-05-17):** this is the trader page's lifeline (polled every 15s by `refresh()`, cross-fetched, observed bursting 2–5 req/s) and the heaviest pure-DB read — six lock-held `Store` reads + a ~145KB body (eq 5000 + 500 trades). It was measured at **8.7s under concurrent load** and was the *only* high-traffic core endpoint not behind `swr_cached` while every slow network endpoint already was (the invariant #7 gap). The portfolio only changes on a decision cycle (`OPEN_INTERVAL_S` ≥ 1800s) so a 15s stale-while-revalidate window is invisible to a trader, serves instantly from the last good payload, single-flight-refreshes in the background, and the runner already pushes every fill to Discord immediately. Injected `cached`/`cache_age_s` honesty keys. `refresh()` tolerates the SWR cold `{"warming":true}` placeholder (skips the tick, self-heals next poll). Locked by `tests/test_core_state_swr.py` |
| `GET /api/portfolio` | Compact portfolio read (consumed by Digital Intern at :8080) |
| `GET /api/data-feed` | Live news-collector pulse — proxies digital-intern's `articles.db` (live-only filter): articles in last 1h / 24h + top active sources. Returns zeros (with `error`) if the article DB is unreachable so the widget still renders |
| `GET /api/validation` | Signal-integrity validation history (permutation tests + label audits) read from `data/validation_results.json`, appended by the continuous loop's background validation runner (capped 50 on the writer side); UI renders the most recent entry |
| `GET /api/backtests` | Full backtest run list with SPY/QQQ baselines |
| `GET /api/backtests/<run_id>` | Single backtest detail (trades, decisions, equity) |
| `GET /api/backtests/compare?ids=1,2,3` | Normalized overlay of 2–4 runs |
| `GET /api/backtests/<run_id>/trades` | Trades for a single backtest run |
| `GET /api/backtests/<run_id>/decisions` | Decisions for a single backtest run |
| `GET /api/model-progress` | Per-cycle aggregated returns for the Model Progress chart |
| `GET /api/analytics` | Sector exposure, Sharpe, Sortino, Calmar, win rate, profit factor, beta, drawdown |
| `GET /api/sector-pulse` | Semis-focused card: price, RSI, vol_ratio, top headline per ticker |
| `GET /api/risk` | Concentration, leveraged exposure, position ages, SPY-shock estimate |
| `GET /api/briefing` | Pre-market / live briefing: futures, next-open countdown, urgent news |
| `GET /api/suggestions` | Trade-idea cards: BUY / ADD / TRIM / EXIT / WATCH per ticker |
| `GET /api/greeks` | Per-leg and portfolio-wide Black-Scholes Greeks |
| `GET /api/scorer-predictions` | DecisionScorer 5d-return predictions per held stock (clamped; `off_distribution` + `raw_pred_5d_return_pct` flag extrapolation) |
| `GET /api/sector-heatmap` | DRAM/semis sector heatmap with momentum + news pulse |
| `GET /api/news-deduped` | Top signals after dedup + exponential urgency decay |
| `GET /api/position-thesis` | Per-position cards combining scorer + technicals + news + last decision. Each card carries `off_distribution` + `raw_pred_5d_return_pct` so the unified conviction board can decay its ML axis off the explicit flag (not a re-derived magnitude heuristic) |
| `GET /api/calibration` | Confidence-bucket win rate + signal-source attribution |
| `GET /api/drawdown` | Drawdown anatomy: peak/trough, time-in-DD, per-position contribution |
| `GET /api/benchmark` | **"Is this bot worth running vs just buying the index?"** — the trader's *first* question, with no home until now. Whole-account return (cash + open + every realised round-trip + unrealised mark) since the first equity write vs the **identical starting capital invested once in the S&P 500 at that same instant and held untouched**. The figure is the `^GSPC` *index level* recorded on every `equity_curve` write from cycle one (~7400 — **not** the SPY ETF; the module says "S&P 500" everywhere, never "SPY", so a 7400 mark is never mislabelled $620). **Distinct from its neighbours — do not "consolidate" (invariant #10):** `/api/open-attribution` is per-**open**-lot alpha *since each lot's entry* (blind to realised P&L / cash drag, resets per re-opened lot, invariant #8); `/api/analytics` `sp500_beta` is a *statistical regression* needing many daily points (`null` on the live book). This is the full-account dollar answer, defined from cycle 1, no regression, no per-lot windowing. Outputs `port_return_pct`/`sp500_return_pct`/`alpha_pp`, `sp500_equivalent_usd`, `usd_vs_sp500`, `pct_cycles_ahead`, running best-lead/worst-lag extremes + a down-sampled (≤200, last point always pinned — strictly bounded, unlike `drawdown.py`'s `+[hist[-1]]` which can overshoot to 201) cumulative-alpha `history`. Sample-size honest like `news_edge`/`trade_asymmetry`: `NO_DATA` (no row with both a value and an S&P mark) → `INSUFFICIENT` (< `_MIN_SPAN_HOURS`=24h **or** < `_MIN_POINTS`=12 benchmarkable points — numerics emitted, **verdict withheld**) → `OK` with verdict `BEATING`/`LAGGING`/`TRACKING` (`\|alpha\|` ≤ `_TRACK_BAND_PP`=0.5pp → TRACKING). The inception anchor is the **first row carrying both a value and an S&P mark** (yfinance cold-start robustness), not blindly `equity_curve[0]`. `headline` is the single source of truth the endpoint, the **`python -m paper_trader.analytics.benchmark [--json]` CLI** (the `desk_pulse`/`signals --check-freshness` precedent — answers from a terminal when `:8090` is wedged/slow; verified live while `/api/state` was timing out) and the Discord line all render verbatim so they can never drift. Endpoint passes the module `INITIAL_CASH` (invariant #12, never a literal 1000). Advisory only — never gates Opus, adds no caps, **not** injected into the decision prompt (invariants #2/#12; the `desk_pulse`/`self_review` observational precedent). Pure core: `analytics/benchmark.py::build_benchmark` (never raises — a malformed row degrades, the contract is "no benchmark this cycle", never an exception). Locked by `tests/test_benchmark.py` (hand-computed BEATING/LAGGING/TRACKING + the **real 2026-05-17 live-book shape** `^GSPC 7444.88→7409.18`, $1000→$972.69 → `−2.25pp / −$22.52` arithmetic lock; NO_DATA/INSUFFICIENT honesty; first-usable-anchor robustness; invariant #12 init=2000 lock; never-raises-on-garbage; history strictly ≤200; reporter line composes the headline verbatim & a builder fault drops only its line while the hourly summary still sends; endpoint e2e via the Flask test client cross-checked equal to the builder on the same store). **Reporter:** `reporter._benchmark_line` appends a `**BENCHMARK** ◈ vs S&P 500 buy-and-hold` block to the hourly + daily-close summaries (composed verbatim, `NO_DATA` suppressed, same "no block, never no summary" failure contract as `_session_block`/`_behavioural_block`). Applies on next paper-trader restart (the documented pattern for every recent feature) |
| `GET /api/earnings-risk` | Upcoming earnings ⨯ held positions / watchlist, tiered (network-sourced from `:8080`; dashboard view, exposure-$ weighted) |
| `GET /api/event-calendar` | **The exact upcoming-earnings block the live trader now sees in its prompt** — the forward complement to the backward-looking behavioural mirrors. `analytics/event_calendar.py::build_event_calendar` over digital-intern's `earnings_calendar.json` snapshot read **directly from disk** (no `:8080` hop — the documented live-cycle hang hazard), `days_away` recomputed vs `now`, tiered `HELD_IMMINENT`/`HELD_SOON`/`WATCH` exactly as `/api/earnings-risk` (single source of truth #10). Distinct from `/api/earnings-risk` (that one is network-sourced + exposure-weighted for the dashboard; this one is the on-disk, prompt-parity, `_safe`-degrading view). Observational only — never gates Opus (invariants #2/#12). Locked by `tests/test_event_calendar.py::TestEventCalendarEndpoint`. Applies on next paper-trader restart |
| `GET /api/scorer-confidence` | Empirical residual bands + directional hit-rate for DecisionScorer |
| `GET /api/decision-health` | Action mix, NO_DECISION parse-failure rate, confidence trend |
| `GET /api/decision-forensics` | *Why* NO_DECISION: failure-mode taxonomy (timeout/truncated/no-json/fenced/prose/malformed/legacy), open-vs-closed split, hourly trend, retry-exhausted count, actionable hint + raw Opus excerpts |
| `GET /api/liquidity` | Capital deployment & liquidity: cash vs deployed %, position weights, unrealized P/L, days-since-last-entry, status (NO_DRY_POWDER/DRY_POWDER_LOW/BALANCED/CASH_HEAVY) + flags |
| `GET /api/build-info` | Code-freshness probe: `{boot_sha, head_sha, behind, stale}`. `stale: true` ⇒ this `:8090` process booted before the on-disk HEAD — committed fixes (e.g. the DecisionScorer ±50 clamp) are NOT applied until restart. The unified dashboard's landing banner reads this + its own to flag stale processes |
| `GET /api/decision-drought` | What the trader's *inaction* cost. Segments cycles into droughts between FILLED trades; per drought: duration, NO_DECISION/HOLD/BLOCKED mix, portfolio Δ% vs S&P Δ% over the idle window, alpha. Splits involuntary `PARALYSIS` (NO_DECISION-dominated) from `DELIBERATE_HOLD`; `involuntary_alpha_bleed_pct` sums the **negative alpha of PARALYSIS droughts only** (DELIBERATE_HOLD drift is a strategy choice, excluded). Complements decision-forensics (*why*) with the *cost*. DB-only, no network. Pure core: `analytics/decision_drought.py::build_decision_drought` |
| `GET /api/news-edge` | Does a high-`ai_score` headline actually predict the move? Per live (non-backtest) scored article naming a watchlist ticker, 1/3/5-trading-day forward return — raw **and SPY-abnormal** — banded by ai_score; verdict judged on abnormal return only. `?days=` (lookback, default 30) / `?min_score=` (default 2.0). Reference horizon is **adaptive**: the longest horizon whose top band is well-sampled, falling back to 1d early on — so the verdict *matures with article history* (digital-intern's `articles.db` only retains a few days of live news, so 3d/5d populate as history deepens; early state is honestly `INSUFFICIENT_DATA` with partial 1d data, never all-dashes). Live-only SQL filter inlined. Pure core: `analytics/news_edge.py::build_news_edge`; daily-bar yfinance history cached 30 min (`_NEWS_EDGE_PX_CACHE`) |
| `GET /api/capital-paralysis` | **Trap + cost + unlock in one view.** liquidity sees the trap (no dry powder), decision-drought sees the cost (alpha bled while pinned), suggestions lists ideas it can't fund — none connect them. Composes `build_liquidity` + `build_decision_drought` (single source of truth — no re-derived metrics) and adds the **unlock ladder**: open positions ranked in desk cut-priority (losers before winners, then largest value), each rung carrying the cash a sale frees, the deployed-% after, and `restores_action_alone` (does this single sale put cash back above `min_actionable_usd` = max($1, 1% of book)?). `recommended_unlock` = the first restoring sale; `state` ∈ `FREE`/`PINNED`/`EMPTY`/`NO_DATA`. **Advisory only — never gates Opus, adds no caps (invariant #2).** Pure core: `analytics/capital_paralysis.py::build_capital_paralysis`. Locked by `tests/test_capital_paralysis.py` |
| `GET /api/open-attribution` | Selection-vs-market on the **open** book — the live trader's *dominant* return source (it mostly HOLDs, so realized round-trips are tiny while open drift dominates; round_trips/`/api/analytics` only cover *closed* trades). Per open **stock** position: return since `opened_at`, SPY return over the same window (anchored to the equity curve's `sp500_price` **at-or-after** entry — `opened_at` is correct because invariant #8 resets it on a reopened lot), `alpha_pct`, and `excess_usd` (unrealized P&L − what the cost basis in SPY would have made). Book aggregate is computed over **anchored rows only** (an un-benchmarkable position would skew `book_open_alpha_pct`). Options are flagged & skipped (alpha-vs-SPY doesn't fit Greeks — `/api/backtests/compare` "stocks only" precedent, invariant #10). Pure core: `analytics/open_attribution.py::build_open_attribution`. Locked by `tests/test_open_attribution.py` |
| `GET /api/trade-asymmetry` | **Behavioural-edge / exit-&-sizing pathology** — the *why* behind the P&L, distinct from `/api/analytics` (raw aggregates) and `/api/calibration` (is the confidence axis accurate). Composes the single source of truth (`build_round_trips`, invariant #10 — no re-derived P&L) into payoff ratio, per-trade expectancy, the **breakeven win-rate the payoff ratio implies vs the actual win-rate** (the gap is the verdict), and the **disposition gap** = mean winner hold-days − mean loser hold-days (negative ⇒ cutting winners faster than losers — the disposition effect that produces a `win-small/lose-big` curve). Sample-size honesty mirrors `news_edge`: numeric metrics emit from the first closed round-trip but the **verdict label is withheld until `STABLE` (n≥20 round-trips)** — `NO_DATA`→`EMERGING`→`STABLE`; a five-trade verdict is noise. Verdicts (STABLE only, precedence in order): `PAYOFF_TRAP` (actual<breakeven ≡ expectancy<0), `DISPOSITION_BLEED` (net-positive but winners cut faster than losers — money left on the table), `EDGE_POSITIVE` (positive & well-managed), `FLAT`. **Intentional divergence from `/api/analytics`:** this module's win/loss split is strict `pnl_usd>0` / `<0` with washes (`==0`) excluded from *both* (matching round_trips' strict `>0` convention, invariant #10), so `avg_loser_usd` and the win-rate basis differ from `analytics_api` (which folds washes into its loser denominator). This is by design — do not "reconcile" them. Advisory only — never gates Opus, adds no caps (invariants #2/#12). Pure core: `analytics/trade_asymmetry.py::build_trade_asymmetry`. Locked by `tests/test_trade_asymmetry.py`. **UI:** Behavioural-edge panel + the previously-orphaned Capital-paralysis & Open-book-alpha panels are now wired into the `:8090` trader page; their JS degrades to an explicit "restart paper-trader to apply" message (not a silent error) when the running process predates the endpoint commit (the `/api/build-info` `stale` contract) |
| `GET /api/decision-reliability` | **The *true current-regime* NO_DECISION rate — not the inflated headline.** `decision-health`/`-forensics`/`-drought` give the rate/why/cost, but the headline % is dominated by *legacy* pre-diagnostics rows (`reasoning == "claude returned no parseable JSON"`) that **stop accruing the moment the runner restarts onto diagnostic code** — a fixed historical mass that never decays. This partitions the decision log at the **newest legacy-tagged failure timestamp** (boundary; `None` ⇒ no legacy ⇒ all rows current) and reports the *post-restart* failure rate + a current-only mode mix, reusing `decision_forensics.classify_failure` (taxonomy) and `build_decision_drought` (`involuntary_alpha_bleed_pct`) as the single source of truth — nothing re-derived (`capital_paralysis` precedent). Sample-size honesty mirrors `news_edge`/`trade_asymmetry`: `NO_DATA` → `STALE_LEGACY_DOMINATED` (legacy failures > current_total **and** current_total < `MIN_CURRENT`=12 → `restart_recommended=True`; the actionable state — restart so failures get diagnostic tags & the sample grows) → `INSUFFICIENT` (current_total < `MIN_CURRENT`, verdict withheld) → `HEALTHY`/`DEGRADED`/`CRITICAL` judged on the **current** rate (≥25 DEGRADED, ≥50 CRITICAL — thresholds identical to `decision_forensics` so they never disagree). `headline_failure_rate_pct` passes `build_decision_forensics` through verbatim for the contrast; `dead_cycles_per_day` = current_rate × decisions/day (cadence from the full timestamp span; `None` on a zero/1-point span — never divides by zero); unparseable-`timestamp` rows are counted in totals but excluded from the current partition when a boundary exists. The verdict *matures with history* (STALE→…→HEALTHY as post-restart cycles accumulate). Advisory only — never gates Opus, adds no caps (invariants #2/#12). Pure core: `analytics/decision_reliability.py::build_decision_reliability`. Locked by `tests/test_decision_reliability.py`. **UI:** `dr-card` panel on the `:8090` trader page; JS degrades via the `/api/build-info` `stale` contract |
| `GET /api/funded-suggestions` | **Pairs every unfundable BUY/ADD idea with the specific sale that funds it.** `liquidity`/`capital-paralysis`/`suggestions` each see part of the trap; none connect "idea I can't afford" to "position to sell to afford it". Composes the existing `/api/suggestions` list (the endpoint calls `suggestions_api()` verbatim — **no refactor**) with `build_capital_paralysis` (its `unlock_ladder` is already in desk cut-priority: biggest loser first). For each conviction-ranked BUY/ADD: `can_act` ⇒ `FUNDED`; PINNED ⇒ walk the ladder attaching the **minimum prefix** of sales whose `cumulative_freed_usd` ≥ an *advisory* suggested notional (`round(conviction × total_value, 2)`, explicitly labelled — sizes nothing) → `UNLOCKABLE` (`funded_by`, `frees_usd`, `enough=True`); whole-ladder-insufficient / empty-ladder / EMPTY / NO_DATA ⇒ `UNFUNDABLE` (full ladder, `enough=False`). Only BUY/ADD are funding-checked — HOLD/WATCH are no-ops and TRIM/EXIT *raise* cash. `top_actionable` = highest-conviction BUY/ADD (deterministic `(-conviction, ticker)` tie-break); `recommended_pairing` = `{sell: recommended_unlock.ticker, buy: top_actionable}` **only when PINNED**. Advisory only — never gates Opus, sizes nothing, adds no caps (invariants #2/#12). Pure core: `analytics/funded_suggestions.py::build_funded_suggestions`. Locked by `tests/test_funded_suggestions.py`. **UI:** `fund-card` panel; same `stale` degrade contract. **Chat:** `unified_dashboard.py::_build_chat_context_block` adds a `_fetch_decision_reliability` sub-fetch emitting one compact `TRADER STATE:` line (pinned + current-regime parse-fail + bleed) so `/api/chat` answers "why isn't it trading?" truthfully; degrades to the pinned/bleed half alone until the trader process restarts onto `/api/decision-reliability` |
| `GET /api/self-review` | **The behavioural mirror the live trader now sees in its own decision prompt** — and the first analytics ever fed back into the decision loop (every other endpoint is human/dashboard-facing only). Composes `build_trade_asymmetry` + `build_capital_paralysis` + `build_open_attribution` **verbatim** (single source of truth, invariant #10 — no re-derived P&L) into one report plus the exact `prompt_block` string `strategy._build_payload` injects right after the `PORTFOLIO` block every cycle. **Observational, never prescriptive:** it states facts and the builders' own calibrated verdicts/headlines, issues no directives, imposes no caps, and its preamble explicitly reaffirms full autonomy — it does **not** violate the "no hard risk limits / Opus has full autonomy" invariant (#2/#12); that invariant governs *gating* decisions, not *informing* them, exactly as `/api/capital-paralysis` & `/api/liquidity` are advisory-only. Do not read this as an autonomy violation and revert it — it is a mirror, not a cage; the system prompt already demands the trader "THINK LIKE A HEDGE FUND MANAGER" and a desk reviews its own P&L attribution before trading. Trades are passed store-native **newest-first**; `build_self_review` reverses internally only for the asymmetry consumer (mirrors `/api/analytics`/`/api/trade-asymmetry`; the liquidity/paralysis path wants newest-first). Pure core: `analytics/self_review.py::build_self_review`; **never raises** — a failing sub-builder degrades to "no mirror" and `strategy.decide()` swallows a self-review fault (failure mode is "no mirror this cycle", **never** "no decision this cycle"). Locked by `tests/test_self_review.py`. **Stale-process caveat (invariant #11):** a `:8090` / live-runner process that booted before this commit will neither serve `/api/self-review` nor inject the block — **restart paper-trader to apply** (check `/api/build-info` `stale`) |
| `GET /api/signal-followthrough` | **Is the trader actually *using* its own news edge?** — grades the *join* nothing else grades. `news-edge` grades the signal alone (*ignoring whether the bot acted*); `decision-drought` grades inaction cost *vs SPY* (*not vs the specific signals present*). This takes every high-`ai_score` **live** signal that named a watchlist ticker and was **visible at decision time** (an article whose `first_seen` fell in the `lookback_hours=2` window ending at a decision's `timestamp` — the exact `get_top_signals(hours=2, min_score=4.0)` window `strategy.decide()` feeds Opus), classifies it **ACTED** (the decision FILLED a transaction on that same ticker that cycle) vs **IGNORED** (HOLD/NO_DECISION/transacted a different name), and compares the 1/3/5-trading-day forward return — raw **and SPY-abnormal** — of the acted vs ignored sets. `selection_edge_pct` = acted − ignored mean abnormal at the **adaptive reference horizon** (longest horizon whose ACTED bucket is well-sampled, falling back to 1d early on — matures with history exactly like `news_edge`, because `articles.db` live news is only days-deep). Signals are deduped **one per (decision, ticker)** (max score/urgency) so a spammy ticker can't dominate. Sample-size honesty mirrors `news_edge`/`trade_asymmetry`/`decision_reliability`: `NO_DATA` (no visible signals) → `INSUFFICIENT` (`n_resolved < _MIN_RESOLVED`=12 — numerics still emitted, verdict withheld) → `IGNORING_FEED` (follow-through < `_IGNORE_THRESHOLD_PCT`=5% — the desk ignores its own newswire; the dominant honest verdict for a HOLD-dominated book) → `LOW_ACTIVITY` (acts, but `n_acted_resolved < _MIN_ACTED`=8 — too few to grade selection) → `MISUSING_SIGNALS` (`selection_edge < −0.25pp` — anti-selection: acts on the duds, sits on the winners) / `EXPLOITING_SIGNALS` (`> +0.25pp` & acted abnormal > 0) / `NEUTRAL_USE`. Ticker resolution, calendar-day mapping and the at-or-after bar lookup are **imported from `news_edge`** (`_resolve_ticker`/`_parse_date`/`_index_at_or_after`) so the two panels can never disagree on which article belongs to which name (single source of truth, invariant #10 spirit). The article fetch (`_fetch_live_articles`) inlines the canonical live-only clause verbatim (invariant #1 / the `signals.py` mirror) and is unit-tested against a planted `backtest://`/`backtest_*`/`opus_annotation*` row. `?days=` (lookback, default 30) / `?min_score=` (default 4.0, matches `strategy.decide`). Advisory only — never gates Opus, adds no caps (invariants #2/#12). Pure core: `analytics/signal_followthrough.py::build_signal_followthrough`. Locked by `tests/test_signal_followthrough.py` (exact-value fixtures: EXPLOITING/MISUSING/IGNORING_FEED, SPY-abnormal subtraction, per-cycle dedup, window boundary, AMDOCS≠AMD word-boundary, live-only SQL filter, `NO_DATA`/`INSUFFICIENT` honesty). **UI:** `sft-card` panel on the `:8090` trader page; **Chat:** `unified_dashboard.py::_build_chat_context_block` adds a `_fetch_signal_followthrough` sub-fetch emitting one compact `SIGNAL EDGE:` line so `/api/chat` can answer "is the bot using its news intelligence?". JS degrades via the `/api/build-info` `stale` contract — the running `:8090` process predates this commit, so it 404s there until **restart paper-trader to apply** |
| `GET /api/churn` | **Overtrading & same-name re-entry churn — the turnover question nothing else asks.** `/api/analytics` shows raw aggregates; `/api/trade-asymmetry` grades the *payoff* pathology (DISPOSITION_BLEED, breakeven-vs-actual win-rate). Neither measures **how often the book re-buys a name it just fully closed, and how fast** — the live NVDA→LITE→NVDA shape (2026-05-16: `avg_holding_days 0.26`, `profit_factor 0.04`). Composes the single source of truth (`build_round_trips`, invariant #10 — **no re-derived P&L/hold**) into: the count/rate of fast same-name re-entries (a same-`(ticker,type,strike,expiry)` re-BUY within `REENTRY_WINDOW_DAYS`=3 calendar days of that key's prior full close — calendar not trading days to stay consistent with `round_trips.hold_days`; 3d chosen because at `OPEN_INTERVAL_S=1800` cadence a genuine thesis *reversal* on the just-exited name rarely matures that fast — a re-buy that quick is turnover, not conviction), the per-active-day round-trip cadence (span-guarded — zero/one-instant span ⇒ `None`, never divides by zero, `decision_reliability` precedent), median hold, sub-day-trip %, and `churn_loss_concentration_pct` = **share of realised *loss* booked in <1-day round-trips** (honest framing — *not* a slippage model; the paper book has no spread). Sample-size honesty mirrors `trade_asymmetry`: numerics from the first round-trip but the **verdict withheld until `STABLE` (n≥`STABLE_MIN_RTS`=20**, identical threshold so the two panels never disagree on STABLE-ness) — `NO_DATA`→`EMERGING`→`STABLE`. Verdicts (STABLE only, precedence): `CHURNING` (≥`REENTRY_CHURN_PCT`=25% fast re-entries **or** ≥`CHURN_RT_PER_DAY`=1.0 round-trips/active-day with a sub-day median hold) / `BUY_AND_HOLD` (≥`HOLD_LONG_DAYS`=10d median hold, <`QUIET_RT_PER_DAY`=0.2 cadence, <25% re-entries) / `ACTIVE_TURNOVER` (between). **Intentional divergence:** the re-entry frequency & cadence are *this* module's headline contribution; median-hold/loss-concentration are derivative context — they are NOT the `trade_asymmetry` disposition gap (winner-vs-loser hold skew) re-derived. Advisory only — never gates Opus, adds no caps (invariants #2/#12). Pure core: `analytics/churn.py::build_churn`. Locked by `tests/test_churn.py` (exact-value fixtures incl. the live NVDA re-entry shape, window-boundary inclusive/exclusive, fastest-first sort, both CHURNING paths, BUY_AND_HOLD/ACTIVE_TURNOVER, sub-day loss-concentration consumed from `build_round_trips`, zero-span divide-by-zero guard, `NO_DATA`/`EMERGING` honesty). **UI:** `churn-card` panel on the `:8090` trader page; JS degrades via the `/api/build-info` `stale` contract |
| `GET /api/source-edge` | **Which of digital-intern's ~17 collectors is worth trusting?** — the operator question nothing else asks. `/api/news-edge` grades the *score* (does an 8.0 headline beat a 3.0?); `/api/signal-followthrough` grades whether the bot *acted*. Neither grades the **source**: of the collectors feeding the pipeline (`rss`, `gdelt`, `reddit`, `scraped`, `google_news`, `finnhub`, `sec_edgar`, …), whose scored headlines actually precede abnormal moves and which are noise to cut/down-weight? Bins every scored live article by **collector family** and reports the 1/3/5-trading-day forward return — raw **and SPY-abnormal** — **pooled across score bands** per family. Pooling (not per-band) is deliberate: digital-intern's live news is only days-deep (`articles.db` shallow-history), so a per-source × per-band × per-horizon split is starved on day 1; the pooled per-source view is both the actionable one (cut a collector) and the one that reaches a usable sample first. **The dirty `source` column is normalised once by `_source_family` — a load-bearing design choice (documented in the module):** substring before the first `/`, trailing `_YYYY-MM[-DD]` stripped, lower-cased — so the live `GDELT/finance.yahoo.com` and the schema-doc'd rolling `gdelt_2025-09` pool into one collector while distinct collectors stay distinct; without it the leaderboard fragments into dozens of n<3 NOISE buckets. Two honesty controls identical to `news_edge`: SPY-abnormal (verdict judged on abnormal only) and a per-source sample gate (`_MIN_SOURCE_N`=8 — mirrors `news_edge._MIN_BAND_N`); below it a source is reported but not graded and the overall verdict is the honest `INSUFFICIENT_DATA`, never a fabricated edge. Adaptive reference horizon + verdict *mature with history* exactly like `news_edge` (`NO_DATA` → `INSUFFICIENT_DATA` → `EDGE_FOUND`/`NO_EDGE`); per-source `verdict` ∈ `EXPLOITABLE`/`WEAK`/`NEGATIVE`/`INSUFFICIENT`; `headline` is the **single source of truth** the UI & chat both render so they can't drift. Ticker resolution / day-parse / at-or-after bar lookup are **imported from `news_edge`** (single source of truth, invariant #10 spirit) so the two panels can never disagree on which article belongs to which name; `_fetch_source_articles` inlines the canonical live-only clause verbatim (invariant #1) and is unit-tested against planted `backtest://`/`backtest_*`/`opus_annotation*` rows. `?days=` (lookback, default 30) / `?min_score=` (default 2.0). Advisory only — never gates Opus, adds no caps (invariants #2/#12). Pure core: `analytics/source_edge.py::build_source_edge`. Locked by `tests/test_source_edge.py` (exact-value fixtures: per-source forward returns, SPY-abnormal subtraction, `_source_family` normalisation incl. `gdelt_2025-09`≡`GDELT/…`, min_score floor, AMDOCS≠AMD word boundary, `NO_DATA`/`INSUFFICIENT_DATA` honesty, live-only SQL filter, **end-to-end via the Flask test client** — not module `__main__`). **UI:** `se-card` panel on the `:8090` trader page (JS degrades via the `/api/build-info` `stale` contract) **and** a cross-fetched mirror on the digital-intern `:8080` dashboard (where the operator who manages collectors sees it; 404→"restart paper-trader to apply"). **Chat:** `unified_dashboard.py::_build_chat_context_block` adds a `_fetch_source_edge` sub-fetch emitting one compact `NEWS SOURCE EDGE:` line so `/api/chat` can answer "which of my news collectors are actually worth trusting?"; silently absent until the trader restarts onto the endpoint |
| `GET /api/feed-health` | **Is the live trader even *seeing* news, or flying blind?** — the upstream question every other panel assumes away. `decision-health`/`-forensics`/`-drought`/`-reliability` measure the *rate/why/cost* of NO_DECISION; `signal-followthrough`/`news-edge`/`source-edge` grade *whether/which* signals predict — all of them presuppose signals *arrived*. None answer "the prompt's `TOP SCORED SIGNALS` block is empty so `signal_count=0` and a blind HOLD is indistinguishable from a deliberate one". `/api/data-feed` shows raw `articles_1h`/`24h` counts with no verdict, no resolved path, no link to the decision log — a stale `articles_24h:3801` reads as healthy. This adds the three dimensions that make the failure *visible & actionable*: the **consecutive 0-signal decision streak** (`blind_streak` — the trader is *provably* blind, not merely between headlines), the **resolved DB path + its newest-live-article age** (`signals._db_path()` — where the trader actually reads, how stale), and **split-brain detection** — historically `signals._db_path()` was existence-first (USB-if-exists) while the daemon/unified-dashboard are LOCAL-first, so a stale USB mirror silently blinded the trader (live state 2026-05-16: USB 24h stale, local 0h fresh). **Invariant #15 root-fixed `_db_path()` to be freshness-aware**, so split-brain is now **legacy-vs-fresh (invariant #16)**: the endpoint also passes `signals._legacy_choice()` (what a *stale running process* on the old resolver still reads); `split_brain` fires when that legacy pick is ≥`SPLIT_BRAIN_GAP_H` staler than the now-fresh resolution (a pre-fix process is blind → restart). New output keys `legacy_path`/`legacy_newest_age_h`. Verdict precedence (locked): `NO_DATA` (no resolved DB / no decisions) → `BLIND` (`blind_streak ≥ BLIND_STREAK_MIN`=3 — the actionable harm; <3 decisions can never reach it, the built-in sample-size guard) → `STALE_FEED` (`newest_live_article_age_h ≥ STALE_HOURS`=6, not yet a long streak) → `HEALTHY`. `split_brain` (legacy pick ≥`SPLIT_BRAIN_GAP_H`=6h staler than the fresh resolution — invariant #16; the pure builder's original `resolved_stale_split` term is retained verbatim & inert unless `legacy_path` is supplied, so the `TestSplitBrain` exact-value fixtures stay green untouched) drives `restart_recommended` — an operator hint, **never** a gate (invariants #2/#12; advisory only). The endpoint does all SQLite/filesystem IO via the testable module helper `dashboard._feed_db_probe` (live-only clause inlined verbatim, invariant #1/#3; cut-offs computed as ISO strings in Python mirroring `signals.get_top_signals` — **not** `datetime('now',…)`, whose space-vs-`T` lexical mis-compare subtly skews `data_feed_api`'s own count); the builder stays pure. Pure core: `analytics/feed_health.py::build_feed_health`. Locked by `tests/test_feed_health.py` (exact `blind_streak`/streak-break/missing-`signal_count`, freshness & split-brain-gap boundaries, NO_DATA/BLIND/STALE_FEED/HEALTHY precedence, constant echo) + `tests/test_feed_health_endpoint.py` (Flask test client end-to-end: a fresher planted `backtest://`/`backtest_*`/`opus_annotation*` row must never read as newest; `_feed_db_probe` live-only lock; the stale-USB/fresh-LOCAL split-brain). **UI:** `fh-card` panel on the `:8090` trader page (fresh id prefix per invariant #14; JS degrades via the `/api/build-info` `stale` contract). **Chat:** `unified_dashboard.py::_build_chat_context_block` adds a `_fetch_feed_health` sub-fetch emitting one compact `TRADER FEED:` line — and, **uniquely**, it does **not** go silent when `:8090` is stale: it degrades to a **direct articles.db read** (the trader-resolved path's newest-live age + split-brain vs the other candidate + the 0-signal streak from the still-served `/api/state`), stating *facts* not a re-derived verdict label so it can't drift from the builder — because feed blindness is precisely the failure that needs surfacing *while* the trader is stale (`/api/build-info` `stale`: the running `:8090` predates this commit so the panel/endpoint 404 there until **restart paper-trader to apply**; the chat fallback works regardless) |
| `GET /api/scorecard` | **Do the independent behavioural checks *agree* on a problem?** — the synthesis ~24 builders / ~30 endpoints never gave. Each existing panel answers one narrow question with its own verdict + chat line; an operator had to read a dozen to learn whether independent diagnostics *concur* (and concurrence is the real signal — `capital_paralysis` PINNED that `decision_drought` also bleeds alpha through, or `trade_asymmetry` PAYOFF_TRAP that `churn` also calls CHURNING, is far stronger than any one alone). **A *router*, not a *grader*** — it mints **no new opinion** (invariants #2/#12; the `self_review` "observational, never prescriptive" precedent it mirrors): composes the five pure, network-free, DB-read-only behavioural builders **verbatim** (`trade_asymmetry` + `churn` + `capital_paralysis` + `decision_reliability` + `open_attribution` — single source of truth, invariant #10, no re-derived P&L), classifies **each builder's own verdict** via a documented per-builder `FLAG`/`OK`/`IMMATURE` table (unknown label → `IMMATURE`, fail-safe: never invents a pathology from a verdict a builder added later; `_safe`'d ERROR marker is its own `ERROR` class, never a flag), counts where ≥2 builders flag the same coarse `theme` (`EXIT_DISCIPLINE`/`CAPITAL_TRAP`/`DECISION_INTEGRITY`/`SELECTION`) as factual `concordance` notes (count + the builders' **verbatim** labels), and forwards the single highest-precedence flag's **own headline verbatim** as `focus` (precedence is a documented factual ordering — same pattern as `trade_asymmetry`'s verdict precedence / `thesis_drift`'s worst-first sort: `DECISION_INTEGRITY > CAPITAL_TRAP > EXIT_DISCIPLINE(PAYOFF_TRAP>DISPOSITION_BLEED>CHURNING) > SELECTION` — it mints no number). `state` ∈ `NO_DATA` (every check immature/error) → `ALIGNED_HEALTHY` (≥1 mature OK, zero flags) → `FLAGS_PRESENT` (≥1 flag); `headline` is the descriptive count + verbatim labels (e.g. "4 of 5 behavioural checks flagging: PAYOFF_TRAP, CHURNING, PINNED, SELECTION_DRAG."). Same store reads as `/api/self-review` so the two can't drift; trades passed store-native newest-first, internally `reversed()` for the asymmetry/churn `build_round_trips` consumers exactly as `/api/analytics` does. **Unlike `/api/self-review` it is NOT injected into the live decision prompt** — it is dashboard/chat only (every endpoint except self-review), so the load-bearing `strategy.decide()` path is untouched. Pure core: `analytics/trader_scorecard.py::build_trader_scorecard` (never raises — a faulting constituent degrades to an `ERROR` check, the contract is "no scorecard this cycle", never an exception). Locked by `tests/test_trader_scorecard.py` (exact-value: NO_DATA/ALIGNED_HEALTHY/FLAGS_PRESENT, the 21-loss-ledger 4-flag concordance fixture, the full per-builder classification table incl. unknown-label→IMMATURE & ERROR class, single-source-of-truth verbatim-headline no-drift, a monkeypatched faulting builder is contained, **endpoint end-to-end via the Flask test client** — not `__main__` smoke). **UI:** `score-*`-prefixed panel on the `:8090` trader page (fresh id prefix per invariant #14; JS degrades via the `/api/build-info` `stale` contract). **Chat:** `unified_dashboard.py::_build_chat_context_block` adds a `_fetch_scorecard` sub-fetch emitting one compact `TRADER SCORECARD:` line (state + verbatim headline + focus + concordance) so `/api/chat` can answer "overall, is the desk behaving, and do the checks agree?"; silently absent (NO_DATA suppressed too) until the trader restarts onto the endpoint. `scorecard` is also registered in `_TRADER_API_PREFIXES` so the root-level `/api/` proxy routes it to the trader |
| `GET /api/desk-pulse` | **The single pure-DB "is the desk OK right now?" digest** — money + loop-liveness + code-staleness + the one behavioural flag to look at first, in one fast dependency-free call. `/api/scorecard` is behavioural-only (no money KPIs); `/api/state` is the heavy everything-dump and the slowest endpoint on the box (SWR cold path seconds); `:8888`'s `/api/command-center` gets its trader half by **cross-fetching** `:8090`, so it blanks exactly when `:8090` is the thing that is slow/wedged (observed live 2026-05-17 — the panel-storm HOL-block on a stale process without the committed `threaded=True`). A *router, not a grader* (the `trader_scorecard` precedent): mints **no new opinion**, composes only the **network-free, pure, DB-read-only** single-source-of-truth builders **verbatim** (invariant #10) — `build_round_trips` (the *identical* strict `>0` win-split as `/api/analytics`, asserted equal end-to-end so a re-derived copy fails loudly) + `build_runner_heartbeat` (loop liveness) + `build_trader_scorecard`'s `focus`+`state` — and adds the concentration KPI `/api/scorecard` omits (`top_weight_pct`/`top_name`/gross, the exact `/api/correlation` `market_value` recipe incl. option ×100 and `current_price`→`avg_cost` fallback, **minus** the yfinance fetch). **No yfinance, no articles.db, no scorer** — a handful of SQLite reads, sub-50ms, so it still answers when every network-backed panel is timing out. Top-level `state` is a documented-precedence rollup over the constituents' own verdicts (operational before behavioural: `LOOP_STALLED` > `CODE_STALE` > `BEHAVIOURAL_FLAGS` > `LOOP_LAGGING` > `HEALTHY`/`NO_DATA` — same idea as `trader_scorecard._FOCUS_ORDER`), forwarding the chosen axis's headline **verbatim** — no minted grade/directive/cap. Invariant #12: the endpoint passes `store.INITIAL_CASH` (never a literal 1000). Advisory only, **NOT** injected into the live decision prompt (dashboard/chat/CLI only) — `strategy.decide()` untouched. Also exposed as **`python -m paper_trader.analytics.desk_pulse [--json]`** — prints the same digest from a terminal, so the operator still gets the answer when the `:8090` process itself is wedged (the `signals.py --check-freshness` precedent). The CLI passes no git context, so `integrity.status` is honestly `UNKNOWN` there (never an optimistic "code current" — the honest-None discipline; `UNKNOWN` also never trips the `CODE_STALE` branch since we can't assert a problem we didn't check); the endpoint always supplies the SHA dict so it resolves `CURRENT`/`STALE`. Pure core: `analytics/desk_pulse.py::build_desk_pulse` (never raises — a faulting constituent degrades that block to an `ERROR`/`None` marker, the contract is "no pulse this cycle", never a 500 that takes the lifeline down). Locked by `tests/test_desk_pulse.py` (exact money metrics cross-checked equal to `/api/analytics` on the shared ledger; option ×100 + avg_cost-fallback concentration; empty book honest `None`; every router-precedence boundary incl. STALLED-beats-stale-beats-flags; invariant #12 −43.5% no-hardcode lock; monkeypatched constituent fault contained; **endpoint end-to-end via the Flask test client**, not `__main__` smoke). Applies on next paper-trader restart (the documented pattern for every recent feature) |
| `GET /api/thesis-drift` | **Is the reason each position was opened for still true?** — the one discipline question no panel answered. `/api/position-thesis` fuses *current* scorer+technicals+news; `/api/suggestions` re-derives an action from scratch. Neither re-tests a holding against **its own opening rationale**, which is sitting verbatim in the opening fill's `trades.reason`. Per open position: selects the opening BUY as the one whose timestamp is **nearest `opened_at`** (invariant #8 — `opened_at` is reset to the re-entry time on a reopened lot, so the nearest BUY is *this* lot's opener, not a prior closed lot's; ties→earliest), surfaces that reason **verbatim** (never NLP-parsed for trading logic — the lone heuristic that reads it is an explicitly-labelled "entry cited a news catalyst, none live now" note), and assigns `health` ∈ `INTACT`/`WEAKENING`/`BROKEN` from **objective deterministic inputs only**: P/L since entry vs `PAIN_PCT`=−8% / `WEAK_PCT`=−3%, plus (when the endpoint supplies live quant/news) MACD flip + negative 5d momentum + `RSI_HOT`=78 + news-gone-cold. Precedence BROKEN>WEAKENING>INTACT; cards sorted worst-first (BROKEN, then most-negative P/L). The endpoint feeds `signals` by reusing `strategy.get_quant_signals_live` + `_ticker_news_pulse` (the exact `/api/suggestions` sources — no re-derivation); a signals failure degrades to **price-only health, never an error**. `state` = `NO_DATA` (no open positions) / `OK`. Pure, network-free *builder* (the network lives in the endpoint, builder takes the dicts) — advisory only, never gates Opus, adds no caps (invariants #2/#12). Pure core: `analytics/thesis_drift.py::build_thesis_drift`. Locked by `tests/test_thesis_drift.py` (BROKEN via pain line / via MACD-flip+mom+loss, WEAKENING via soft loss / hot RSI / cold-catalyst, opener-nearest-`opened_at` on a re-entered lot, verbatim-reason preservation, missing-ledger degrade, worst-first sort). **UI:** `tdrift-card` panel on the `:8090` trader page; JS degrades via the `/api/build-info` `stale` contract |
| `GET /api/loser-autopsy` | **Per-closed-losing-round-trip post-mortem — *why each closed trade lost*.** The neighbours each see a different slice: `/api/thesis-drift` re-tests **open** positions against their opening rationale; `/api/trade-asymmetry` is **aggregate** payoff math (one number for the whole book); `/api/churn` counts re-entry **cadence**. None narrate the individual loss. Composes the single source of truth (`build_round_trips`, invariant #10 — **no re-derived P&L/hold**), joins the **verbatim** opening-fill thesis and closing-fill reason back from the contributing `trades.reason` rows by their DB `id` (the `thesis_drift` "surface verbatim, never NLP-parse for trading logic" discipline), and assigns an objective, documented failure mode per loser — `KNIFE_CATCH` (loss ≤ `BIG_LOSS_PCT`=−15%, precedence-first: the thesis was badly wrong) / `WHIPSAW` (closed < `FAST_HOLD_DAYS`=1d at a shallow > −3% loss) / `SLOW_BLEED` (held ≥ `SLOW_HOLD_DAYS`=5d and still red — the disposition behaviour `trade_asymmetry` aggregates, surfaced per-trade) / `STOPPED_OUT` (else). Rolls up *which name is the bleed* (`ticker_breakdown`, most-negative-$ first), *which mode dominates* (deterministic count then a fixed severity tie-break so the verdict never flips on dict order), and *which losing names recur* (`repeat_offenders`, n≥2 — distinct from `churn`'s re-entry-cadence framing). Strict `pnl_usd<0` loser convention (a sub-cent wash reads as a non-loss, matching `round_trips`/`trade_asymmetry`, #10). Sample-size honesty mirrors `trade_asymmetry`: per-loser cards + numerics emit from the first loss but the **pattern verdict is withheld until `STABLE`** (`n_losers ≥ STABLE_MIN_LOSERS`=8) — `NO_DATA`→`NO_LOSSES`→`EMERGING`→`STABLE`. Advisory only — never gates Opus, adds no caps (invariants #2/#12). Pure core: `analytics/loser_autopsy.py::build_loser_autopsy` (never raises — malformed rows degrade, never except). Locked by `tests/test_loser_autopsy.py`. **UI:** `lautopsy-card` panel on the `:8090` trader page (fresh id prefix per invariant #14; table built via DOM `textContent`, never `innerHTML`, so a verbatim reason can't inject markup); JS degrades via the `/api/build-info` `stale` contract |
| `GET /api/winner-autopsy` | **Per-closed-winning-round-trip post-mortem — *why each closed trade won*. The positive mirror of `/api/loser-autopsy`.** Every behavioural builder on the desk reflects a *pathology*: `/api/loser-autopsy` narrates losses, `/api/trade-asymmetry` flags `DISPOSITION_BLEED`, `/api/churn` counts overtrading, `/api/self-review` feeds **only the failures** back into the live decision prompt. None tell the desk *which winning behaviour to repeat*. This is the symmetric counterpart: composes the single source of truth (`build_round_trips`, invariant #10 — **no re-derived P&L/hold**), joins the **verbatim** opening-fill thesis and closing-fill reason back from the contributing `trades.reason` rows by their DB `id` (the `loser_autopsy`/`thesis_drift` "surface verbatim, never NLP-parse for trading logic" discipline), and assigns an objective, documented success mode per winner — the exact sign-flipped mirror of the loss taxonomy: `HOME_RUN` (gain ≥ `BIG_WIN_PCT`=+15%, precedence-first: the thesis was strongly right) / `SCALP` (closed < `FAST_HOLD_DAYS`=1d at a shallow < +3% gain — the disposition effect `trade_asymmetry` aggregates, surfaced per-trade on the *winning* side: a winner cut too fast) / `SLOW_GRIND` (held ≥ `SLOW_HOLD_DAYS`=5d and still green — *let a winner run*, the **good** disposition behaviour, the exact opposite of `loser_autopsy`'s `SLOW_BLEED`, the one to repeat) / `TARGET_HIT` (else). Rolls up *which name is the engine* (`ticker_breakdown`, most-positive-$ first), *which mode dominates* (deterministic count then a fixed significance tie-break `HOME_RUN>SLOW_GRIND>TARGET_HIT>SCALP` so the verdict never flips on dict order — the mirror of `loser_autopsy`'s `_SEVERITY` tie-break), and *which winning names recur* (`repeat_winners`, n≥2). Strict `pnl_usd>0` winner convention (a sub-cent wash reads as a non-win, matching `round_trips`/`trade_asymmetry`/`loser_autopsy`, #10). Sample-size honesty mirrors `loser_autopsy`: per-winner cards + numerics emit from the first win but the **pattern verdict is withheld until `STABLE`** (`n_winners ≥ STABLE_MIN_WINNERS`=8, identical threshold so the two panels never disagree on STABLE-ness) — `NO_DATA`→`NO_WINS`→`EMERGING`→`STABLE`. Advisory only — never gates Opus, **never injected into the decision prompt** (dashboard/chat-only, unlike `/api/self-review`), adds no caps (invariants #2/#12). Pure core: `analytics/winner_autopsy.py::build_winner_autopsy` (never raises — malformed rows degrade, never except). Locked by `tests/test_winner_autopsy.py` (22 cases, exact mirror of `test_loser_autopsy.py`: `_classify` boundary matrix incl. precedence & strict/inclusive edges, `NO_DATA`/`NO_WINS`/wash-not-a-win/`EMERGING`/`STABLE` gate, verbatim entry/exit reason join, best-first ordering + `best_n` cap, median even/odd, `ticker_breakdown`+`repeat_winners`, deterministic significance tie-break, P&L consumed from `build_round_trips` not recomputed, never-raises-on-garbage). **UI:** `wautopsy-card` panel on the `:8090` trader page directly below `lautopsy-card` (fresh id prefix per invariant #14; table built via DOM `textContent`, never `innerHTML`, so a verbatim reason can't inject markup); JS degrades via the `/api/build-info` `stale` contract |
| `GET /api/hold-discipline` | **The disposition trap, caught *while it is still happening* on the OPEN book.** The desk's documented pathology is the disposition effect (a 16.7%-win-rate book, ~0.52d median hold — cuts winners fast, rides losers down). Every neighbour sees it *after the fact* or from a *different* slice: `/api/loser-autopsy` & `/api/trade-asymmetry` post-mortem trades **already closed**; `/api/thesis-drift` re-tests an open position against its *thesis*; `/api/capital-paralysis` is about cash drag; `/api/position-thesis` shows days-held but has **no empirical reference**. None answer the forward discipline question a desk asks every day: *which open position am I, right now, holding at a loss past my own historical losing-cut time?* Anchors on the desk's **own** behaviour — the empirical median *losing* hold consumed **verbatim** from `build_loser_autopsy` → `build_round_trips` (single source of truth #10 — never a re-derived median/P&L) — and the per-position $ read **directly** from `positions.unrealized_pl` (the option ×100 is already baked into that column; re-deriving from `avg_cost×qty` would silently halve/×100 an option's risk). A position is *overstayed* iff `unrealized_pl < 0` **and** `age_days > median` (strict — `==` is within discipline, the `loser_autopsy` strict-boundary idiom; a winner past the median is never flagged). State `NO_DATA`(no open book)→`INSUFFICIENT`(< `MIN_REFERENCE_LOSERS`=3 closed losers — cards+ages still emitted, **nothing flagged, verdict withheld**, the `loser_autopsy` sample-size precedent)→`DISCIPLINED`→`DISPOSITION_DRAG`; `disposition_drag_usd` = Σ overstayed `unrealized_pl`, `worst_overstayed` most-negative, overstayed cards sort first. Advisory only — never gates Opus, **never injected into the decision prompt** (the `loser_autopsy`/`winner_autopsy` endpoint precedent; invariants #2/#12). `_safe`: a composed-builder fault degrades to an honest verdict-withheld state (`reference_state` `ERROR:…`), never an exception that 500s the route or kills the close report (the `event_calendar` contract). Pure core: `analytics/hold_discipline.py::build_hold_discipline` (never raises). Also surfaced in the **DAILY CLOSE** Discord report via `reporter._hold_discipline_line` (composed verbatim, NO_DATA/INSUFFICIENT suppressed, "no block, never no summary" failure contract — the operator lives in Discord, the dashboard is often stale). Locked by `tests/test_hold_discipline.py` (no-drift median lock, strict boundary, sample-size gate, `_safe` never-raises, endpoint parity on a temp Store, reporter suppress/emit/survive-fault). **No UI card** (invariant #14 `TestTemplateIdsUnique` footgun; endpoint + Discord consumers only). Applies on next paper-trader restart (the documented stale pattern — `/api/build-info` `stale`/`behind`) |
| `GET /api/game-plan` | **The single prioritised, trader-facing action plan for the next session — the synthesis the ~45 single-concern panels never did.** Every ingredient already exists separately: the co-pilot verb (`/api/suggestions` via `_classify_action`), the open-book disposition trap (`/api/hold-discipline`), name-level concentration (`/api/risk`), and forward earnings (`/api/event-calendar`). Before this a trader had to open four panels and fuse them by hand; *distinct from unified's `/api/action-queue`*, which is **operator** triage (stale process, decision-parse health, breaker state) — this is the **trade** plan (per held position: a verb + a priority + the fused reasons; plus portfolio directives and non-held opportunities). The route does the data-gathering and **reuses `_classify_action` verbatim** (no forked verb logic — the `funded_suggestions` "no refactor" precedent, single source of truth #10) and reuses `build_hold_discipline`/`build_event_calendar` + the `/api/risk` concentration math (`_classify`+`_concentration_severity`) so the panels can never disagree. Fusion is deterministic: an *overstayed losing* position (the `hold_discipline` flag the co-pilot alone can't see) escalates a co-pilot `HOLD`→`REVIEW EXIT`; the single largest position under **HIGH** concentration is pushed `HOLD`→`TRIM`; both only ever move **up** the sell ladder `TRIM<REVIEW EXIT<EXIT` — a stronger verb the co-pilot already produced is **never** weakened (a `EXIT` survives); imminent earnings on a *held* name is **awareness** — it raises the additive priority score and annotates, it never invents a sell verb (the observational invariants #2/#12 contract). Priority ties break deterministically (`-priority, unrealized_pl, ticker`); `opportunities` = non-held BUY/WATCH past a 0.30 conviction floor, conviction-desc. State `NO_DATA`(empty book & no setups)→`STEADY`(nothing actionable)→`ACTIONS_PRESENT`. Advisory only — it reorders/annotates existing signals; it never sizes a trade, never gates Opus, **never injected into the decision prompt**, adds no caps (the `hold_discipline`/`event_calendar` endpoint precedent). `_safe` end-to-end: every composed builder/network fetch is wrapped so a fault degrades that one input, never 500s the route. SWR-cached 45s (the multi-second `get_quant_signals_live`+`get_prices` fan-out — the `/api/suggestions` precedent). Pure core: `analytics/game_plan.py::build_game_plan` (no I/O, never raises — the network lives in the endpoint, the builder takes the dicts; the `thesis_drift` split). Locked by `tests/test_game_plan.py` (overstay→REVIEW EXIT escalation, EXIT-not-downgraded, HIGH-conc→TRIM + HIGH directive, held-earnings raises priority without a verb, opportunities exclude-held + conviction-sorted + floor, STEADY/NO_DATA states, deterministic priority order, never-raises-on-garbage, and a Flask-test-client endpoint test on a fresh temp Store that a deep single-name loss is not read as a calm HOLD). **No UI card** (invariant #14 `TestTemplateIdsUnique` footgun; endpoint consumers only — natural home is unified's command-center which already renders cards). Applies on next paper-trader restart (the documented stale pattern — `/api/build-info` `stale`/`behind`) |
| `GET /api/tail-risk` | **The left-tail view the upside-heavy surface was missing — "what is a realistic bad day?"** Every existing risk panel measures a *single worst path* (`/api/drawdown` max-DD) or *risk-adjusted upside* (`/api/analytics` Sharpe/Sortino/Calmar). None state the *frequency or shape* of daily losses. Returns historical 95/99% 1-day VaR (nearest-rank, sign kept honest — a positive quantile yields a negative "no loss" VaR, never a clamped 0), positional expected-shortfall CVaR (mean of the worst `ceil(q·n)` returns — **deliberately positional not value-threshold**: 99/110−1 and 89.1/99−1 are both "−0.10" but differ in the last float bit, so a `r<=threshold` filter silently drops one tie and halves the tail), population annualised vol & downside deviation (`/n` to match `analytics_api`'s Sharpe/Sortino exactly), Fisher-Pearson population skew (`None` when σ=0, never a fabricated 0), worst/best day, max consecutive down-day streak, Ulcer index. Daily series resampled **byte-identically** to `analytics_api`'s `by_day` last-write-wins loop (single-source-of-truth #10 spirit — a future refactor must change both or the dashboard's Sharpe and this panel silently disagree). Sample-size honesty mirrors `build_correlation`: `NO_DATA` (no equity) → `INSUFFICIENT` (<`MIN_RETURNS`=20 daily returns — numerics emitted, verdict withheld) → `OK`. Advisory only — never gates Opus, **never injected into the decision prompt** (invariants #2/#12; the tuned prompt + "no hard risk limits" identity). Also folded into `/api/analytics` as an additive top-level `tail_risk` key (keyed-assertion-safe) so the digital-intern analyst chat surfaces VaR/CVaR/skew with no extra fetch. Pure core: `analytics/tail_risk.py::build_tail_risk` (never raises). Locked by `tests/test_tail_risk.py` (hand-pinned discrete metrics, independent-impl cross-check for vol/skew, flat-book = the live 2026-05-14 shape, skew-sign, float-tie CVaR) + `tests/test_core_analytics.py::TestTailRiskIntegration` (endpoint↔builder no-drift). **No UI card** (invariant #14 `TestTemplateIdsUnique` footgun; endpoint + `/api/analytics` consumers only). Applies on next paper-trader restart (the documented stale pattern — `/api/build-info` `stale`/`behind`) |
| `GET /api/correlation` | **Concentration honesty — do the held names actually move *together*?** `/api/risk` reports **name-level** concentration (`concentration_top1_pct`/`top3_pct`) and a single 3% SPY-shock; it cannot see **factor** concentration — a "2-position 59/41" book reads as merely concentrated, but if both names co-move the operator is running a *single bet* and the SPY-shock understates the tail. Computes pairwise Pearson **return** correlation among the held **stock** positions (deterministic ticker-sorted pairs; a flat series → `None`, never a fabricated 0), the most-coupled pair, the weight-Herfindahl `effective_positions_naive` (1/HHI), and the **correlation-adjusted `effective_independent_bets`** = `n / (1 + (n−1)·mean_ρ)` clamped to [1, n] — which collapses toward 1 as the names co-move however many tickers are on the book (mean ρ=−1 with n=2 → denominator 0 → honest `None`, never a fabricated number). Options are flagged & skipped (correlating a Greeks payoff against a linear return is meaningless — the `open_attribution`/`/api/backtests/compare` "stocks only" carve-out, #10 spirit). **The builder is pure; the yfinance daily-bar fetch lives in the endpoint** via the shared `_daily_history_cached` (3mo, the existing 30-min `_NEWS_EDGE_PX_CACHE`) — exactly the `thesis_drift` "network in the endpoint, builder takes the dicts" split, so the core is offline & deterministically testable and a fetch failure degrades to `INSUFFICIENT`, never an error. Sample-size honesty mirrors `news_edge`/`trade_asymmetry`: `NO_DATA` (no stock positions) → `INSUFFICIENT` (<2 correlatable names, or series < `MIN_RETURNS`=10 aligned daily returns — numerics where computable, verdict withheld) → `OK` with verdict precedence `SINGLE_NAME_RISK` (top weight ≥ `DOMINANT_WEIGHT`=60% — single-name risk reads first, correlation is secondary) > `CONCENTRATED` (mean ρ ≥ `HIGH_CORR`=0.70 — the book moves as one) > `MODERATE` (≥ `MOD_CORR`=0.40) > `DIVERSIFIED`. Pairs are measured over a **common aligned tail** so every ρ uses the same window. Advisory only — never gates Opus, adds no caps (invariants #2/#12). Pure core: `analytics/correlation.py::build_correlation` (never raises). Locked by `tests/test_correlation.py`. **UI:** `pcorr-card` panel on the `:8090` trader page (fresh id prefix per invariant #14); JS degrades via the `/api/build-info` `stale` contract |
| `GET /api/decision-context` | **What is the live trader actually being *shown* right now?** — the decision *input* every one of the ~45 output-diagnostic endpoints presupposes. `decisions` stores only `action_taken`+`reasoning`; the only raw capture is `RAW_CAPTURE_CHARS`=1000 of the *response* on a parse failure. When the trader spends cycle after cycle on `NO_DECISION (timeout/empty)` / flat `HOLD` (the dominant 2026-05-17 live pattern — `$972.69`, `$18.49` cash, MU stale-marked) an operator has no way to see *what Opus was fed*. This reconstructs it on demand: the prompt rendered through the **same `strategy._build_payload`** the live `decide()` uses (+ the identical `SYSTEM_PROMPT`/`ML ADVISOR` framing) so it is **byte-identical to the live prompt given identical inputs** (single source of truth, invariant #10 — no re-implemented prompt), bounded to `MAX_PROMPT_CHARS`=40000 with `prompt_chars`/`prompt_truncated` honesty keys; an `input_summary` (top/urgent/merged counts — `signal_count` is the *exact* value `decide()` writes to `decisions.signal_count` — watchlist/futures resolved-vs-missing, quant tickers, sentiment mentions); `advisory_blocks` presence (self-review/track-record/risk-mirror/ml); the embedded `/api/mark-integrity`; and a `feed_state` ∈ `BLIND` (0 merged signals — a HOLD this cycle is *forced* by an empty feed, not chosen) / `DEGRADED` (≥`DEGRADED_MISSING_RATIO`=50% of watchlist prices missing — the yfinance starvation behind the timeout storms) / `OK`. **`_claude_call` is never invoked** (`claude_invoked:false`; locked by an endpoint test that monkeypatches it to raise and still expects 200). The snapshot is the new write-free `strategy.portfolio_snapshot_readonly`, which shares the extracted pure `strategy._mark_to_market` with the live `_portfolio_snapshot` so the inspector's marks (incl. expired-option intrinsic #13 + `stale_mark`) can never drift from the real ones (invariant #10) and the dashboard thread never mutates the live trader's persisted marks/equity. Orchestration (`assemble_inputs`, mirrors `decide()`'s pre-`_claude_call` assembly with each advisory builder wrapped non-fatally exactly as `decide()` wraps it) is shared by the endpoint **and** `python -m paper_trader.analytics.decision_context [--full|--json]` (works when `:8090` is wedged — the `desk_pulse`/`signals --check-freshness` precedent) so the two can't drift. SWR-cached 30s (the assemble fetch is multi-second; the `/api/state` precedent). Advisory only, **NOT** injected into the decision prompt — dashboard/chat/CLI only (invariants #2/#12; `strategy.decide()` untouched). Pure core: `analytics/decision_context.py::build_decision_context`. Locked by `tests/test_decision_context.py` (prompt section-header fidelity, exact input counts incl. `signal_count`, ML-advisor gating, feed_state boundaries, truncation honesty, embedded mark-integrity verbatim, and the `portfolio_snapshot_readonly` *marks-identically-but-never-writes* contract vs `_portfolio_snapshot`) + `tests/test_decision_context_endpoint.py` (Flask test client: never-calls-Opus 200, BLIND/DEGRADED, read-only, SWR honesty keys + warm-hit). Applies on next paper-trader restart (`/api/build-info` `stale`) |
| `GET /api/mark-integrity` | **How much of the displayed book value is *fictional* right now?** — the mark-trust meta-metric no panel surfaces. When yfinance returns nothing for a held name `strategy._mark_to_market` falls back to `avg_cost` and flags `stale_mark=True` (the live 2026-05-17 pathology: `MU 0.5 @ 724.12`, `current_price==avg_cost`, `P/L $0.00` — indistinguishable from a genuinely flat row). That flag is surfaced *per position* to Opus & Discord, but nothing answers the **aggregate**: what share of gross book value is marked at cost, so `/api/analytics` Sharpe, `/api/drawdown`, the equity curve and the headline P&L are all quietly partially false. Reports `n_stale`, `stale_value_usd`, `stale_value_pct` of gross, per-name rows, `stale_tickers`, and a verdict `NO_DATA`→`CLEAN`→`DEGRADED` (0<pct<`UNTRUSTWORTHY_PCT`=50, or gross 0 with stale rows so the share is unquantifiable) →`UNTRUSTWORTHY` (≥50% — treat every displayed P/L as substantially fictional until the feed recovers / runner restarts). Reads the write-free `strategy.portfolio_snapshot_readonly` (never mutates the live trader). Pure, never raises (garbage rows degrade to zero value — the behavioural-builder `_safe` contract). Advisory only — never gates Opus, adds no caps (invariants #2/#12). Also embedded inside `/api/decision-context`. **Folded as an additive `mark_trust` honesty key into the three equity-derived risk endpoints this docstring names as the silent victims — `/api/tail-risk`, `/api/drawdown`, `/api/analytics` (2026-05-18, Agent 4).** A stale cycle records a *cost-frozen flat* equity point; those flats deflate vol/drawdown, inflate Sharpe, and truncate the VaR tail, yet a grep showed `stale_mark` had only ever reached mark_integrity/strategy/dashboard/reporter — never these maths. `dashboard._mark_trust_block(store)` composes `build_mark_integrity` **verbatim** off the SAME write-free `portfolio_snapshot_readonly` snapshot (single source of truth #10 — no re-derived staleness), adds `{verdict,n_stale,n_positions,stale_value_pct,stale_tickers,headline,note}` (the `note` only when verdict ∉ CLEAN/NO_DATA), and is `_safe`: any fault → key **omitted** so the risk payload is byte-identical and the endpoint never 500s for this reason. Purely additive (keyed-assertion-safe, the existing `tail_risk`-in-`/api/analytics` precedent); observational only, no caps, not injected into the decision prompt, **no schema change** (invariants #2/#12/#13). `hold_discipline`/`thesis_drift` (which read open-position P/L and silently misread a stale `$0.00` as a genuine flat) are a known *deferred* contamination — their endpoints feed `store.open_positions()` which lacks `stale_mark`, so a fix needs an endpoint data-source change that risks their existing exact-value `TestEndpoint`s; see `docs/superpowers/specs/2026-05-18-mark-trust-risk-surface-design.md`. Pure core: `analytics/mark_integrity.py::build_mark_integrity`. Locked by `tests/test_mark_integrity.py` (the exact live MU-stale shape `stale_value_pct`=37.94 off the raw gross, `>=50`→UNTRUSTWORTHY inclusive boundary, zero-gross no-divide-by-zero, option ×100, never-raises-on-garbage) + `tests/test_decision_context_endpoint.py` (Flask test client read-only + UNTRUSTWORTHY-when-price-missing) + `tests/test_mark_trust.py` (Flask test client end-to-end on all three endpoints: stale book → `mark_trust` UNTRUSTWORTHY; clean → CLEAN/`note`=None; **additive no-risk-drift** vs a direct `build_tail_risk` call — only `mark_trust` added, every risk field byte-identical; the `_safe` snapshot-fault → 200 + key-omitted contract; single-source-of-truth no-drift vs `build_mark_integrity`). Applies on next paper-trader restart |
| `GET /api/model-reliability` | **Which model actually made each live decision — full Opus vs the degraded Sonnet fallback — and how often the cycle produced nothing.** The stack is tuned end-to-end around Opus's reasoning depth (invariant #3), but `strategy.decide()` has a degrade ladder Opus→(timeout)Sonnet-on-condensed-prompt→NO_DECISION and **no panel was blind-spot-free here**: `/api/decision-health` buckets by *outcome* (a Sonnet-on-a-stripped-prompt FILLED is counted identically to a full-Opus FILLED), `/api/decision-forensics` only dissects the *NO_DECISION* excerpts. This reads the authoritative `fallback_used` flag in each made-decision's `reasoning` JSON (rows predating that flag read back `None` — verified live, a large pre-instrumentation tail — and are bucketed `legacy_unknown` and **excluded from the ratio** so a stale history can't fake a healthy/unhealthy number) and the NO_DECISION reason-prefix (`timeout`/`parse_failed`/`retry_failed`, mirroring strategy.py's exact strings). Reports per 24h/7d/all: `opus`/`sonnet_fallback`/`legacy_unknown` counts, `opus_share_pct` (of *attributable*), `no_decision_pct`, and the money cut `filled_fallback`/`filled_total`/`filled_fallback_pct` (how many *executed trades* the degraded model placed); plus a recent-vs-older `trend` (improving/worsening/flat) and a verdict `NO_DATA`→`INSUFFICIENT` (<`_MIN_ATTRIBUTABLE`=10 attributable, verdict withheld — the sample-size-honesty precedent)→`OPUS_HEALTHY` (≥90% Opus) / `DEGRADED` (≥70%) / `FAILING`. Pure, never raises (non-str rows degrade, not raise). Observational only — never gates Opus, adds no caps (invariants #2/#12; the `decision_health`/`self_review` precedent). Also `python -m paper_trader.analytics.model_reliability [--json]` (works when `:8090` is wedged). Pure core: `analytics/model_reliability.py::build_model_reliability`. Locked by `tests/test_model_reliability.py` (legacy-`None`-not-counted-as-Opus, outcome-prefix parsing, verdict bands, FILLED-from-fallback only-counts-fills, 24h windowing, worsening-trend ordering, never-raises-on-garbage). Applies on next paper-trader restart |

### Common failure modes (live trader)

| Symptom | Likely cause | Where to look |
|---------|--------------|---------------|
| Loop posts `NO_DECISION` every cycle | Claude returned malformed JSON or timed out (`DECISION_TIMEOUT_S=120`) | `strategy.py::_parse_decision`; tail runner stdout for `[strategy] claude err:` |
| Live trader stuck on `BLOCKED` for a SELL | `_enforce_risk_pre_trade` rejected — qty > held, or option `strike+expiry` unspecified with multiple open legs | `strategy.py::_enforce_risk_pre_trade`, `_execute` (option ambiguity check) |
| Hourly summary never posts | `_maybe_hourly` only advances on send success; openclaw missing → permanent retry-loop with stdout log | Search runner stdout for `[reporter] openclaw not installed` |
| `signals.get_top_signals` returns `[]` | `articles.db` not at `USB_DB` (USB unmounted) or `LOCAL_DB`; live-only filter is correct so backtest contamination is *not* the cause | `signals._db_path()`; run `python3 -m paper_trader.signals` |
| `paper_trader.db is locked` | Another writer attached without `?mode=ro`; or a long-running query inside `_lock` | Check for ad-hoc scripts; only the runner should write |
| Dashboard `/api/scorer-predictions` shows `is_trained: false` | `data/decision_outcomes.jsonl` has < 500 rows — scorer hasn't trained enough yet | `wc -l data/decision_outcomes.jsonl` |
| Discord posts stop entirely (`[reporter] openclaw not installed; would send:` spam, every report dropped) | **`openclaw` is an npm-global under the nvm node bin; the systemd unit launches `runner.py` with a minimal PATH that excludes it, so `shutil.which('openclaw')` returned `None`** (live-finding 2026-05-17 — `openclaw` *was* installed at `~/.nvm/versions/node/<v>/bin/openclaw`, just unreachable). **Root-fixed (review pass #10):** `reporter._resolve_openclaw()` now tries `OPENCLAW_BIN` env override → `PATH` → well-known fallbacks (`~/.local/bin`, `/usr/local/bin`, `/usr/bin`, `~/.nvm/.../bin`). Applies on next runner restart. If it *still* fails: auth expired, or set `OPENCLAW_BIN=/abs/path/openclaw` in the unit | `which openclaw` (may be on *your* PATH but not the unit's — compare `tr '\0' '\n' </proc/<runner-pid>/environ \| grep ^PATH`); `python3 -c "from paper_trader.reporter import _resolve_openclaw; print(_resolve_openclaw())"` |
| Trader frozen — `NO_DECISION` every cycle for hours, equity flat, **no Discord alert** | **Claude CLI quota/usage-limit exhausted** — `claude` exits rc=1 with stdout `You've hit your org's monthly usage limit` (Opus *and* the Sonnet fallback). The circuit-breaker pkill is futile (the process already exited). **Surfaced (review pass #10):** `strategy._is_quota_exhausted` flags it → `summary["quota_exhausted"]` → `runner._cycle` fires ONE `reporter.send_quota_alert` (deduped; re-armed + a `RECOVERED` notice when a real decision lands) and skips the futile breaker. The alert only reaches Discord once the openclaw-resolution fix above is also live | `grep -a 'QUOTA EXHAUSTED' logs/runner.log`; `/api/decision-reliability` (`TIMEOUT_EMPTY` 100% of current failures with a fresh feed = quota, not a feed outage); resolve the Anthropic quota / upgrade the plan — a runner restart will NOT help |
| Live cross-dashboard (`:8080` → `:8090`) shows blanks | CORS or paper-trader process down | `curl http://localhost:8090/api/portfolio` |
| Strategy returns `HOLD` constantly even with strong signals | Opus is being conservative — by design, no threshold gating to override | Inspect the prompt context in `strategy.py::_build_payload`; if the watchlist has stale prices yfinance is rate-limited |
| Equity / P/L looks too high and won't come down; an option position never closes | Pre-fix `_portfolio_snapshot` marked an expired contract at avg_cost forever (no live chain past expiry). Fixed — see invariant #13. If you see this on an old `:8090` process, check `/api/build-info` `stale` and restart | `strategy._option_expired` / `_expired_intrinsic`; `SELECT * FROM positions WHERE type IN ('call','put') AND closed_at IS NULL AND expiry < date('now')` |
| `logs/runner.log` has no `[runner]`/`[strategy]` lines, only `"GET /api/… HTTP/1.1"` | **`logs/runner.log` captures only the Werkzeug HTTP access log, NOT the runner's own stdout** (live-finding 2026-05-17). Every "tail runner stdout / runner.log for `[strategy] claude err`" instruction above & in `CLAUDE.md` §11 is *blind* against that file — the NO_DECISION/timeout/circuit-breaker `print()`s go to the runner's real stdout (a terminal / launcher), not here. This is a launcher/logging-infra gap, not a code bug (deliberately not "fixed" in a surgical core pass — changing logging perturbs the live process) | Find the runner's true stdout: `tr '\0' ' ' </proc/$(pgrep -f 'paper_trader.runner\|paper-trader/runner.py' | head -1)/cmdline`; check the launcher's redirection / `journalctl`. Decision-level history is reliable via `/api/decision-forensics` + `recent_decisions` (DB), which *do* capture the failure taxonomy |
| Decisions ~hourly (not every `OPEN_INTERVAL_S`=1800s) while open; new endpoints 404; self-review maybe not injected | The running `:8090`/runner is **stale** — booted ≥1 commit ago (`/api/build-info` `stale:true`, `behind:N`). It runs pre-fix resolvers/cadence and 404s endpoints added since boot (live-finding 2026-05-17: `behind:33`, `/api/runner-heartbeat` 404). A long NO_DECISION run also inflates effective cadence (Opus 180s + Sonnet 60s + retry 45s per failed cycle). The on-disk fixes do **not** apply until restart | `curl -s localhost:8090/api/build-info`; restart `paper_trader.runner` to apply HEAD (also applies the invariant #6/#18 fixes). NO_DECISION cause → `/api/decision-forensics` |

| 2–3 decisions clustered inside a minute, then ~1h of nothing; equity/decision log looks raced; doubled `claude` RAM | **Two `runner.py` processes on the same paper book** — e.g. an orphaned manual launch (parent PID 1) *and* the systemd unit, each on its own cadence (live-finding 2026-05-17: PID 1255030 orphan + PID 1317545 systemd). The single-instance guard (invariant #19) prevents *recurrence* but only **on the next restart of each** — it does not kill an already-running duplicate | `ps -eo pid,ppid,etime,cmd | grep '[r]unner.py'` — if >1, stop the orphan (keep the systemd one); after each restarts, `cat data/paper_trader.runner.lock` shows the single live holder PID. A second start now self-exits with `[runner] another paper trader is already running (pid=…)` |
| Live trader makes decisions with `signal_count=0` for many cycles though `articles.db` is fresh | Upstream: digital-intern's scorer degraded — articles are *collected* (fresh `first_seen`) but `ai_score` stays `0.0`/`urgency 0`, so `get_top_signals(min_score=4.0)` + `get_urgent_articles` both return `[]`. **Not a paper-trader core bug** — `/api/feed-health` correctly reports `BLIND` with `resolved_live_2h>0`; the headline spells out the paradox. A paper-trader restart will NOT help (the fix is in digital-intern's scoring daemon) | `/api/feed-health` (`verdict:BLIND`, `resolved_live_2h`); `SELECT MAX(ai_score) FROM articles WHERE first_seen>=<2h-ago> AND <live-only>` → if `0.0`, the digital-intern scorer is down |

For ML / backtest-side failures, see the ML section below and `CLAUDE.md` §11.

## ML / backtest domain

### How the DecisionScorer works

`paper_trader/ml/decision_scorer.py` defines an MLP (`sklearn.MLPRegressor`,
with a numpy lstsq fallback) that predicts **5-trading-day forward return %**
from a 17-dim feature vector:

| Slot | Feature | Source | Default |
|------|---------|--------|---------|
| 0 | `ml_score` | parsed from `_ml_decide` reasoning | 0.0 |
| 1 | `rsi` (14-period) | `_compute_technical_indicators` | 50.0 |
| 2 | `macd_signal` (numeric) | same | 0.0 |
| 3 | `mom5` (5-day %) | same | 0.0 |
| 4 | `mom20` (20-day %) | same | 0.0 |
| 5 | `regime_mult` | `_market_regime` (bull=1.0, sideways=0.6, bear=0.3, unknown=1.0) | 1.0 |
| 6 | `vol_ratio` | clamped to [0, 5] | 1.0 |
| 7 | `bb_pos` | clamped to [-2, 2] | 0.0 |
| 8 | `news_urgency` | clamped to [0, 100] | 50.0 |
| 9 | `news_article_count` | clamped to [0, 20] | 1.0 |
| 10..16 | sector one-hot | `SECTOR_MAP` lookup | "other" |

Training happens in `run_continuous_backtests.py::_train_decision_scorer`
after each cycle. Until trained AND `_n_train >= 500`, `predict()` returns
`0.0` and `_ml_decide` ignores it entirely.

Once `_scorer.is_trained and _n_train >= 500`, the scorer **modulates BUY
conviction only — it never cancels a trade** (an earlier HOLD-block
version oscillated leveraged-ETF strategies; see the comment in
`_ml_decide`). Given the predicted 5-day return `p`:

| Condition | Effect on conviction |
|-----------|----------------------|
| `p < -10` | `× 0.6` (strong headwind, still buys) |
| `-10 ≤ p < 0` | `× 0.85` (mild headwind) |
| `0 ≤ p ≤ 5` | unchanged |
| `5 < p ≤ 10` | `× 1.15`, capped at 0.95 |
| `p > 10` | `× 1.3`, capped at 0.95 |

> Note: `CLAUDE.md` §6 still documents the older HOLD-blocking gate
> (`p < -5 → HOLD`, `p < 0 → ×0.7`). The code in `_ml_decide` above is
> authoritative; CLAUDE.md §6 is stale on this point.

**Prediction is clamped to the empirical label support.** `MLPRegressor`
has no output bound, so for off-distribution feature vectors it extrapolates
to nonsense (observed: −89% then +32% for the *same* LITE vector across two
retrain cycles — the unbounded head is volatile). `predict()` clamps its
output to `±PRED_CLAMP_PCT` (50%). The bound is load-bearing-safe: across the
9k+ rows in `decision_outcomes.jsonl` only ~0.4% of real 5d outcomes exceed
|50%| (p1=−25%, p99=+32%), and every gate boundary above (±10/±5/0) sits well
inside ±50, so a clamped −89→−50 stays in the same `p < -10 → ×0.6` bucket —
**gating behaviour is unchanged**. Clamping is output-only: it does not touch
`build_features`/`SECTORS`/`N_FEATURES`, so the pickle stays compatible, and
`train_scorer` never calls `predict()`, so there is no label-feedback loop.
The untrained short-circuit (`return 0.0`) still runs *before* the clamp.
`predict_with_meta()` is the sibling that exposes
`{pred, raw, clamped, off_distribution}` for panels that want to flag
extrapolation honestly (`/api/scorer-predictions` adds `off_distribution`
+ `raw_pred_5d_return_pct`; the unified dashboard's `_conviction_axes` decays
the ML axis toward a 0.3 trust floor once `|pred| > 20%` instead of letting a
clamped floor read as full ±1.0 conviction). Locked by
`tests/test_decision_scorer.py::TestPredictionClamp`.

**Honesty on a *failed* prediction (2026-05-17 fix).** When
`model.predict()` itself *raises* — the exact scenario the handler's
"silenced after first" log guards (a `build_features` feature added
without retraining the pickle ⇒ shape/dtype mismatch) —
`predict_with_meta()` now returns `clamped: True, off_distribution:
True` (was `False`/`False`). A scorer that *cannot score the input at
all* must not look identical to one confidently predicting a flat 0.0:
the honesty panels above read `off_distribution`, so the old value
rendered a broken scorer as gospel. This mirrors the non-finite branch
precedent and keeps the documented `off_distribution`-is-an-alias-of-
`clamped` invariant. `predict()`'s scalar contract is unchanged (still
the safe `0.0`) — only the meta trust flags move. Locked by
`tests/test_decision_scorer.py::TestPredictionClamp::test_predict_exception_is_flagged_low_trust`.

**Concurrency invariant (`backtest.py`):** the module-global
`_VOLUME_CACHE` is shared across the parallel run threads. Every read
*and* every iteration of it must hold `_VOLUME_CACHE_LOCK` — iterating it
unlocked while another run thread inserts raises
`RuntimeError: dictionary changed size during iteration`, which the
persist helper's `try/except` swallows (silently dropping the disk
cache so every run re-fetches volumes from yfinance). It is also
window-keyed and never evicted, so a long-lived continuous loop's RSS
grows slowly across cycles — restart the loop periodically; do not add
an ad-hoc eviction policy without measuring.

### How to run backtests manually

```bash
cd /home/zeph/paper-trader

# One-shot — 10 parallel year-long runs, default window 2025-05-01..2026-05-13
python3 run_backtests.py

# Continuous loop — 5 runs per cycle, retrains scorer between cycles
python3 run_continuous_backtests.py

# View results
sqlite3 backtest.db "SELECT run_id, total_return_pct, vs_spy_pct, status FROM backtest_runs ORDER BY run_id DESC LIMIT 20"

# Live dashboard
# http://localhost:8090/backtests
```

### How to interpret backtest results

- `total_return_pct` — full-window % change vs. $1000 starting capital.
  Positive means the persona made money; the "winner" of a cycle is the
  highest-positive run.
- `vs_spy_pct` — alpha vs. SPY buy-and-hold over the same window. The
  meaningful metric for skill evaluation.
- `status` — `running` / `complete` / `failed`. `failed` rows often mean
  yfinance returned nothing for the persona's preferred tickers; check
  `continuous.log` for the matching `[engine] RUN N CRASHED:` line.
- `equity_curve_json` — JSON list of `{date, value, cash}` snapshots; the
  dashboard renders these. Sparse during a run (every 5 samples) and full
  at finalize.

A healthy cycle log looks like:

```
[engine] SPY baseline 2025-05-01 → 2026-05-13: +X.X%
[engine] Launching 5 runs starting at run_id=N
[run K] DONE  final=$..  return=+Y.Y%  vs SPY +Z.Z%  trades=NN
[continuous] computed N decision outcomes from M runs
[continuous] scorer ok n=N rmse=...
[continuous] ml: injected I new | trainer n=N loss=...
```

If `scorer insufficient_after_dedup n=...` keeps appearing, the
`data/decision_outcomes.jsonl` tail is too small or too duplicated — more
cycles need to accumulate before the scorer can train.

> **Read `vs_spy_pct` skeptically on leveraged windows.** A single
> persona routinely posts `+1000%+ / vs_spy +1200%` over a 6–10yr window
> heavy in 3× ETFs (SOXL/TQQQ), while a *different* persona on the **same
> window** posts `+12% / vs_spy −80%`. That spread is leveraged-beta
> dispersion through a cherry-able bull window, **not** repeatable alpha.
> The "best run +N%" cycle line is the max of a high-variance leverage
> draw — never read it as strategy skill. The permutation/label-audit
> validation suite (`data/validation_results.json`) is the real
> skill-vs-luck arbiter; the per-run number is not.

### Scorer calibration diagnostic

`paper_trader/ml/calibration.py` is a **read-only** quant diagnostic
(no train, no pickle/`build_features`/`N_FEATURES` touch, no trade path —
safe to run against the live unattended loop). It answers *"does a high
predicted 5d return actually precede a high realized one?"* by separating
the two failure modes a single RMSE hides:

- **rank skill** — tie-aware Spearman over every `(pred, realized)` pair.
  Tie-awareness is load-bearing: the scorer clamps to ±`PRED_CLAMP_PCT`,
  so off-distribution predictions tie at exactly ±50 — plain
  `argsort(argsort)` fabricates rank skill there (a constant predictor
  would score 1.0).
- **magnitude bias** — per-decile `mean_pred` vs `mean_realized`.

```bash
# Calibration of the live pickle vs the accumulated outcomes tail
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.calibration
```

Verdicts: `INSUFFICIENT_DATA` (< `MIN_PAIRS`), `MISCALIBRATED`
(spearman < `SPEARMAN_MIN` or decile curve not mostly monotone),
`DIRECTIONAL_BUT_BIASED` (rank-skilled but mean decile error >
`BIAS_TOL_PCT` — trust the *sign/ordering*, discount the predicted %),
`WELL_CALIBRATED`, `WEAK_SIGNAL`. `scorer_calibration()` flips the SELL
target sign (`-forward_return_5d`) exactly like `train_scorer`, so a
rank-skilled SELL model is not a false `MISCALIBRATED`. Thresholds are
module constants; verdicts are exact-value test-locked in
`tests/test_calibration.py`.

> **Interpreting the verdict (2026-05-17 quant finding).** Pointed at the
> full `decision_outcomes.jsonl` the tool reports `WELL_CALIBRATED`
> (spearman ≈ 0.51, monotone deciles, ≈1.9pp decile error) — but that is
> **in-sample**: the scorer was trained on most of those rows. The
> trustworthy generalization metric is the temporal-holdout `oos_rmse`
> the continuous loop logs (`scorer ok … oos_rmse=…`). The correct
> comparator is the *trivial baseline on the same temporal-holdout
> slice*: the latest-20%-by-sim_date OOS slice has σ(aligned target)
> ≈ 11.7, so a model that just predicts the mean scores RMSE ≈ 11.7
> there. Observed `oos_rmse` runs **13–17** — i.e. *worse than
> predicting the mean*, so the scorer has **negative demonstrated
> out-of-sample skill** on the holdout even though it gates BUY
> conviction once `_n_train ≥ 500`. The in-sample `WELL_CALIBRATED` is optimistic;
> always read it next to `oos_rmse`. The decile tails over-predict
> (d1 pred −15.7 vs realized −10.7; d10 +15.4 vs +11.9) even in-sample —
> the same extrapolation the `predict_with_meta` `off_distribution` flag
> exists to surface. This is a reported observation, **not** a code
> change: altering the model/gate is a training-dynamics change out of
> scope for a surgical review (CLAUDE.md §6, AGENTS.md "When to bump
> model versions").
>
> **Update (2026-05-17 second pass).** The negative-skill picture is no
> longer uniform: the last 8 logged statuses show `oos_rmse` of
> 8.18 / 17.36 / 14.62 / 10.56 / 11.73 / 11.78 / 10.51 / 9.36 — i.e.
> recent cycles cluster *around* the σ≈11.7 mean-predictor baseline rather
> than uniformly above it, so OOS skill is now borderline/regime-dependent,
> not flatly negative. In-sample re-measured the same day: spearman 0.50,
> monotone deciles, 1.60 pp mean decile error, but the tails still
> over-predict (d10 pred +11.76 vs realized +6.64; d1 −8.05 vs −4.47) —
> exactly the extrapolation the new off-distribution gate-abstention guards.
> The grep-the-log method is fragile; the **wired
> `data/scorer_skill_log.jsonl` ledger is now the durable trend source** —
> use it to judge whether this borderline state is improving as
> `decision_outcomes.jsonl` accumulates.
>
> **Operational note (2026-05-17).** The *running* continuous-loop process
> predates all of the above commits, so it is still on stale code: no
> `oos_diracc`/`oos_ic`, no `scorer_skill_log.jsonl`, no
> `winner_training.jsonl` trim (file ~322 MB), no off-distribution gate,
> startup-only orphan reap. **Restart `run_continuous_backtests.py` to
> deploy these fixes** — they are inert until then. Separately,
> `_inject_and_train` has been logging `trainer timeout` on ~4 of every 5
> recent cycles (digital-intern's `ml.trainer.train(force=True)` exceeds the
> 120 s cap, likely GPU contention) — the winner→ArticleNet feedback loop
> (CLAUDE.md §5 step 5) is effectively non-functional; injection still
> succeeds, training does not. Reported, not fixed (root cause is
> GPU-side / out of this domain's surgical scope).

### Position sizing invariant (`_ml_decide`)

A backtest BUY's notional is `min(total_val * conviction, cash * 0.95)`.
`conviction` has a hard ceiling: `min(0.25, best_score/20)` for normal
tickers, `min(0.40, best_score/15)` for a `_LEVERAGED_ETFS` name in a
bull/sideways regime. The DecisionScorer (once `_n_train >= 500`) only
*modulates* this conviction — it never lifts the cap (the ×1.3/×1.15
tailwind arms are themselves capped at 0.95, and the notional is still
clipped by the two `min`s). Both arms are now test-locked:
`tests/test_backtest.py::TestMlDecide::test_oversize_buy_clipped_by_cash`
pins the cash arm; `::test_conviction_caps_position_size_when_cash_is_abundant`
pins the conviction arm with exact expected values (a regression that drops
`min(0.25, …)` doubles the notional and fails the assertion). If you change
the conviction formula, update both tests deliberately — they assert exact
numbers, not ranges, by design.

The five **scorer-gate arms** themselves are now exact-value locked in
`tests/test_ml_backtest_review.py::TestMlDecideScorerGate`: with the module
`_DECISION_SCORER` singleton swapped for a fake returning a fixed prediction,
each arm's effect on a base conviction of 0.25 is asserted as an exact share
qty (`p<-10 → 75.0`, `-10≤p<0 → 106.25`, `0≤p≤5 → 125.0`, `5<p≤10 → 143.75`,
`p>10 → 162.5`), plus the **n_train ≥ 500 gate** (a trained scorer with
`_n_train = 100` must NOT modulate, even on a -50 prediction — locks invariant
#5). Any change to the gate thresholds or multipliers must update these
assertions deliberately.

**Off-distribution gate abstention (2026-05-17).** `_ml_decide` now calls
`_scorer.predict_with_meta()` (falling back to the plain `predict()` scalar
for the `_Dummy` stub / predict-only test fakes, treated as in-distribution).
When the scorer flags `off_distribution=True` — the unbounded MLP head
extrapolated beyond `±PRED_CLAMP_PCT`, or `predict` raised / went non-finite
— the five conviction arms are **skipped entirely**: the quant-derived
conviction is left untouched rather than modulated on a clamped ±50 that
carries no information (AGENTS.md already documents the head emitting −89→+32
for the *same* LITE vector across retrains). In-distribution behaviour is
byte-identical to before (`predict()` delegates to
`predict_with_meta()["pred"]`), so every exact-value `TestMlDecideScorerGate`
assertion is unchanged. The abstention is surfaced in the decision
`reasoning` as `scorer=…%(off-dist,gate-skipped)`. Locked by
`tests/test_ml_backtest_review.py::TestMlDecideOffDistributionGate`
(catastrophic −50 off-dist → conviction unchanged; in-dist meta path still
modulates identically; reasoning surfaces the skip; independent of the
`n_train<500` guard).

### Continuous-loop durability & honesty (2026-05-17)

- **Scorer-skill ledger now wired in.** `_append_scorer_skill_log` /
  `_parse_scorer_status` existed but were **never called** — the durable
  per-cycle OOS-skill audit trail was dead code; the metrics only reached
  the ephemeral, rotated `continuous.log`. `main()` now appends exactly one
  structured row per cycle to `data/scorer_skill_log.jsonl`
  (`{cycle, timestamp, window_*, status, train_n, val_rmse, oos_n, oos_rmse,
  oos_dir_acc, oos_ic, gate_active}`). On a non-training cycle (no outcome
  records) it writes the `no outcome records` sentinel with a
  `_deployed_scorer_n_train()` hint so `gate_active` (⇔ deployed
  `n_train ≥ 500`, invariant #5) stays truthful. Bounded at
  `SCORER_SKILL_LOG_KEEP=2000` via the atomic tmp+`.replace` idiom.
  **This is the canonical instrument for the negative-OOS-skill question
  below — query it, not grep'd log lines.** Locked by
  `tests/test_continuous.py::TestParseScorerStatus` /
  `TestAppendScorerSkillLog` / `TestDeployedScorerNTrain` /
  `TestCycleWiringRegression`.

- **`winner_training.jsonl` is now bounded.** It had grown to ~322 MB /
  860 k lines, unbounded, while every sibling JSONL is trimmed — a latent
  disk-full risk (the OSError [Errno 28] class noted in
  `decision_scorer.py`). `_trim_winner_jsonl()` (called once per cycle from
  `main()`) keeps the last `WINNER_JSONL_KEEP=50000` records via the same
  atomic tmp+`.replace` idiom; far above the 10 k `_inject_and_train` tail so
  the consumer is never starved (older rows are already idempotently in
  `articles.db`). Locked by `TestTrimWinnerJsonl`.

- **Per-cycle orphan reap.** `_reap_orphaned_runs()` was startup-only, so a
  run thread hard-killed mid-cycle (OOM/SIGKILL — never reaches
  `finalize_run` or `run_all`'s caught-`failed` marker) stayed
  `status='running'` forever for a long-lived loop (observed live: 15 rows
  stuck 35 h while ~170 newer runs completed — the "dashboard shows running
  forever" symptom, CLAUDE.md §11). `main()` now also reaps once per cycle;
  the reaper is idempotent, best-effort and 6 h-age-guarded so it can never
  touch a live run. Locked by
  `TestCycleWiringRegression::test_main_reaps_orphans_per_cycle_not_only_at_startup`.

- **`vs_spy_pct` benchmark-honesty flag.** yfinance intermittently fails to
  return SPY for a window; `PriceCache` then persists an **empty SPY series**
  (verified: `prices_2021-08-02_2025-08-01.json` had `SPY_rows={}` while 116
  other tickers loaded). `_build_trading_days` falls back to another ticker's
  calendar so the run still completes, but `returns_pct("SPY",…)` returns
  `0.0` → `vs_spy_pct == total_return` with **no real benchmark** (80/485
  complete runs / 16 windows live). The NOT NULL DEFAULT 0 schema
  (invariant #13) blocks a true NULL, so `run_one` now writes a
  `benchmark_unavailable: …` string into the additive nullable `notes`
  column + a stderr WARNING. **Purely informational — zero change to
  returns, winner selection, or the live `_ml_is_qualified` gate.** Locked by
  `tests/test_integration_backtest.py::TestBenchmarkUnavailableNote`.
  > **Still open (reported, not fixed — out of surgical scope):** the
  > poisoned per-window price cache re-fabricates this every cycle the
  > window is drawn (the cache-validity check accepts an empty SPY series
  > because SPY is still listed in `_meta.tickers`), and the live trader's
  > `_ml_is_qualified` median-alpha gate (CLAUDE.md §15) counts these
  > `notes`-flagged runs because it filters on `vs_spy_pct IS NOT NULL`
  > (always true under NOT NULL). Treat any `vs_spy_pct` skeptically until a
  > cache-validity / gate-side fix lands. Read `notes` before trusting a
  > run's alpha.

### Out-of-sample calibration view + training-integrity filter (2026-05-18)

- **Only FILLED trades feed scorer/ArticleNet training (fix).**
  `_compute_decision_outcomes` and `_append_top_decisions` read
  `backtest_decisions` for BUY/SELL rows but did **not** filter on execution
  status. `run_one` records a terminal non-FILLED decision row for the last
  intraday decision when nothing filled that day; had that decision been a
  BUY/SELL `_execute_decision` rejected, its 5d forward return would have
  trained the DecisionScorer (and `winner_training.jsonl`→ArticleNet) as a
  *phantom outcome for a position that never moved capital* — and the
  blocking reason (out of cash / no price) is regime-correlated, so it is
  biased contamination, not noise. Both pipeline queries now require
  `status = 'FILLED'`. **Latent, not active**: an audit of the live
  `backtest.db.local_backup` showed BUY/SELL decisions are *100% FILLED*
  (5393 FILLED, 0 non-FILLED; HOLD is the only other status) — `_ml_decide`
  only ever emits executable decisions today. The filter makes the
  "trained only on real fills" invariant explicit and refactor-proof (one
  position-cap commit away from silently corrupting the scorer). Locked by
  `tests/test_continuous.py::TestFilledOnlyTrainingIntegrity` (FILLED
  survives, BLOCKED excluded, on *both* pipelines).

- **OOS calibration view (`calibration.py --oos`, feature).**
  `scorer_calibration` over the full `decision_outcomes.jsonl` is an
  *in-sample* read (the scorer trained on most of those rows), so its
  `WELL_CALIBRATED` verdict is optimistic — AGENTS.md already warned of
  this but there was no out-of-sample *decile* view (`skill_trend.py`
  trends the ledger's scalar `oos_rmse`/`oos_ic`; `gate_audit.py` buckets
  by the 5 economic gate arms — neither shows the magnitude-bias decile
  curve + crisp verdict on unseen data). `scorer_calibration_oos()` reuses
  `paper_trader.validation.split_outcomes_temporal` — the **exact** split
  `_train_decision_scorer` uses for `oos_rmse`/`oos_ic`, so this decile
  view and the ledger's scalar OOS metrics describe the *same* holdout —
  and runs the same `scorer_calibration` report on only the most-recent
  `oos_fraction` (default 0.2) by `sim_date`. Returns the report plus
  `{oos_n, train_n, oos_fraction}`. `python3 -m paper_trader.ml.calibration
  --oos` prints the in-sample report (byte-identical default), then the
  temporal-holdout report, then an explicit optimism-gap line when
  in-sample is `WELL_CALIBRATED` but OOS is not. Read-only, never raises
  (degrades to `INSUFFICIENT_DATA` — same operational discipline as the
  rest of the module). Exact-value locked by
  `tests/test_calibration.py::TestScorerCalibrationOOS` (split sizes &
  metadata, slice-equivalence to `scorer_calibration(recs[-oos:])`,
  history-`WELL_CALIBRATED` vs OOS-`MISCALIBRATED` overfit signature,
  `< 5`-row and empty degradation).

  > **Quant finding (2026-05-18, decisive).** Run live on the deployed
  > pickle (`n_train=3830`, 5000 outcomes): **in-sample `WELL_CALIBRATED`**
  > (spearman 0.51, monotone deciles, 2.08 pp decile error) vs **temporal
  > OOS `MISCALIBRATED`** on the 1000-row holdout (spearman **0.013**,
  > monotone 0.556, decile error 7.6 pp). The OOS decile-realized column is
  > flat noise across the whole prediction range — d1 (mean_pred −21.34)
  > realized −1.05 vs d10 (mean_pred +21.43) realized −0.27: the most
  > bearish and most bullish predicted buckets have *statistically
  > identical* realized outcomes. The scorer’s `WELL_CALIBRATED` is purely
  > a training artifact; out-of-sample it has **~zero rank skill**, yet it
  > gates BUY conviction every cycle (`gate_active=true`, `n_train ≥ 500`,
  > invariant #5). This corroborates `skill_trend`’s `NEGATIVE_OOS_SKILL`
  > verdict and the wired `scorer_skill_log.jsonl` (`oos_dir_acc` ≈ 0.47–
  > 0.55, `oos_ic` ≈ 0, `val_rmse` ≪ `oos_rmse` — textbook overfit). This
  > is a **reported observation, not a model change** — altering the MLP /
  > gate is a training-dynamics change out of surgical scope (CLAUDE.md §6).
  > A skeptical quant should treat the conviction gate as adding sizing
  > variance with no demonstrated compensating edge until OOS skill clears
  > the mean-predictor baseline (`skill_trend` / `--oos` are the arbiters).

### Multi-horizon outcome capture + horizon audit (2026-05-18)

- **`_compute_decision_outcomes` now additively records
  `forward_return_10d` / `forward_return_20d`** alongside the unchanged
  `forward_return_5d`. The DecisionScorer still trains **only** on the 5d
  label (`train_scorer` reads `forward_return_5d` exclusively) and the gate
  is untouched — the extra horizons are pure read-only research signal. The
  helper `_fwd_ret_h(ticker, sim_d, idx, h)` (defined beside `_td_index`)
  is best-effort: a horizon whose window runs past cached price history
  yields `None` and **never** skips or zeroes the 5d row training depends
  on (the 5d path is byte-identical — locked by
  `tests/test_horizon_audit.py::TestComputeDecisionOutcomesMultiHorizon`,
  exact `8.3333 / 16.6667 / 33.3333` on the synthetic curve + the
  5d-present/10d-20d-`None` tail case). Legacy rows in
  `decision_outcomes.jsonl` have no 10d/20d keys; they populate as the
  continuous loop runs the new code.

- **`paper_trader/ml/horizon_audit.py` (new read-only diagnostic).** Every
  pre-existing OOS arbiter (calibration / gate_audit / skill_trend /
  baseline_compare / regime_audit) can *only* measure skill against the 5d
  label — none can answer the decisive question that follows from their
  shared `oos_ic ≈ 0` finding: **is the scorer near-blind because the
  features carry no signal, or because the 5-trading-day target is just too
  noisy** (AGENTS.md already notes leveraged ETFs have "noisy 5d windows
  but strong 3-12 month returns"). On the temporal-OOS slice (the *same*
  `validation.split_outcomes_temporal` every other OOS tool uses) it
  rank-ICs the two signals that actually drive `_ml_decide` — `ml_score`
  (feature slot 0) and `mom20` — against each of 5d/10d/20d, reusing
  `calibration._spearman` and the codebase-universal SELL sign-flip
  (applied to probe *and* target, the `baseline_compare._aligned`
  precedent). Verdicts (exact-value test-locked in
  `tests/test_horizon_audit.py`, module constants `MIN_PAIRS=30`,
  `IC_MARGIN=0.05`, `EDGE_FLOOR=0.10`): `INSUFFICIENT_DATA` →
  `INSUFFICIENT_LONG_HORIZON` (5d sampled but 10d/20d not yet accumulated —
  the honest pre-population state) → `NO_HORIZON_HAS_EDGE` (best
  \|rank-IC\| < `EDGE_FLOOR` at *every* horizon — dead feature set, not a
  horizon problem) → `LONGER_HORIZON_MORE_PREDICTABLE` (a longer horizon
  beats 5d by > `IC_MARGIN` — the 5d target is the handicap) → `5D_ADEQUATE`.
  Read-only, never raises (same operational discipline as the rest of the
  module). CLI: `python3 -m paper_trader.ml.horizon_audit [--all]`.

  > **Quant finding (2026-05-18, this pass).** Live OOS arbiters on the
  > deployed pickle (`n_train=3485`, 1109-row temporal-OOS slice):
  > `skill_trend` = **`NEGATIVE_OOS_SKILL`** (recent median `oos_rmse`
  > 11.30 vs the fresh mean-predictor baseline **5.56**, `oos_ic` 0.02,
  > `oos_dir_acc` 0.505, **trend DEGRADING**, `gate_active=1.0`);
  > `calibration --oos` = **`MISCALIBRATED`** (spearman 0.039; the
  > OOS decile-realized column is flat noise — d1 realized +0.06 vs d10
  > +1.36); `regime_audit` = **`REGIME_UNIFORM_NULL`** (≈0 skill in every
  > measurable regime — not a regime-mix artifact). The decisive one:
  > `baseline_compare` = **`MLP_WORSE_THAN_TRIVIAL`** — the raw `ml_score`
  > one-liner scores OOS rank-IC **+0.204** while the 17-feature MLP scores
  > **+0.039** (gap −0.165): *the network destroys the signal it is fed*.
  > The new `horizon_audit` **independently reproduces `ml_score`'s 5d OOS
  > rank-IC at exactly +0.2038** (byte-identical to `baseline_compare`'s
  > number — a built-in cross-check confirming it is wired to the same
  > slice / sign-flip / Spearman), and currently returns
  > `INSUFFICIENT_LONG_HORIZON` (the outcomes file predates the 10d/20d
  > capture; the horizon question becomes answerable as the loop
  > accumulates rows). One nuanced counterpoint: `gate_audit` reads
  > **`GATE_EFFECTIVE`** on *this* OOS window (strong_tailwind +1.44% vs
  > strong_headwind −0.21%, spread +1.65 pp) — but `arm_monotone`=0.75
  > (the neutral arm +0.07% sits *below* mild_headwind +0.57%) and
  > `skill_trend` shows the edge is regime-contingent and degrading, so it
  > is a fragile, non-monotone, window-specific artifact, not a stable
  > edge. **All reported observations, not model changes** — altering the
  > MLP/gate is a training-dynamics change out of surgical scope
  > (CLAUDE.md §6). The actionable thread: the signal demonstrably *exists*
  > in raw `ml_score` (+0.20 OOS); the MLP is the lossy component.

### Tests (ML + backtest section)

```bash
# ML + backtest only — keep "calibration", "continuous" AND "horizon" in
# the filter: test_calibration.py / test_continuous.py / test_horizon_audit.py
# have none of "ml"/"backtest"/"scorer" in their node ids and are silently
# missed by the older filters (test_continuous.py holds the continuous-loop +
# scorer-skill-ledger + winner-trim + reaper-wiring locks;
# test_horizon_audit.py holds the multi-horizon-capture + horizon-audit locks).
cd /home/zeph/paper-trader && python3 -m pytest tests/ -v -k "ml or backtest or scorer or calibration or continuous or horizon"

# Core (live trader) only
cd /home/zeph/paper-trader && python3 -m pytest tests/test_core_*.py -v

# Full suite
cd /home/zeph/paper-trader && python3 -m pytest tests/ -v

# Scorer calibration diagnostic (exact-value verdict locks; incl. the
# TestScorerCalibrationOOS temporal-holdout view added 2026-05-18)
cd /home/zeph/paper-trader && python3 -m pytest tests/test_calibration.py -v

# In-sample vs temporal-OOS calibration of the LIVE pickle (read-only;
# surfaces the in-sample-optimism gap in one command)
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.calibration --oos

# Training-integrity (only FILLED trades train scorer/ArticleNet)
cd /home/zeph/paper-trader && python3 -m pytest tests/test_continuous.py::TestFilledOnlyTrainingIntegrity -v

# Per-persona strategy-quality leaderboard (exact-value verdict locks)
cd /home/zeph/paper-trader && python3 -m pytest tests/test_persona_leaderboard_20260517.py -v

# Per-persona decision-signal-skill diagnostic (exact-value verdict locks)
cd /home/zeph/paper-trader && python3 -m pytest tests/test_persona_skill.py -v

# Permutation feature-importance diagnostic (exact-value verdict locks)
cd /home/zeph/paper-trader && python3 -m pytest tests/test_feature_importance.py -v

# Regime-conditional scorer-skill audit (exact-value verdict locks)
cd /home/zeph/paper-trader && python3 -m pytest tests/test_regime_audit.py -v
# In-sample vs temporal-OOS skill bucketed by realized regime (read-only):
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.regime_audit          # OOS slice
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.regime_audit --all    # full in-sample

# Multi-horizon outcome capture + forward-return-horizon predictability
# audit (exact-value verdict + IC locks; the only file with "horizon" in
# its node ids — silently missed by the older "ml/backtest/scorer" filter)
cd /home/zeph/paper-trader && python3 -m pytest tests/test_horizon_audit.py -v
# Is the scorer's ~0 OOS skill a 5d-target-noise artifact? (read-only):
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.horizon_audit          # OOS slice
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.horizon_audit --all    # full in-sample

# Training-corpus & OOS-construction audit (exact-value verdict locks)
cd /home/zeph/paper-trader && python3 -m pytest tests/test_corpus_audit.py -v
# Is the loop's temporal-OOS holdout a real generalization test? (read-only;
# exits 2 on OOS_NOT_HELD_OUT — the corpus is one cycle's single window):
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.corpus_audit

# A single class
cd /home/zeph/paper-trader && python3 -m pytest tests/test_decision_scorer.py::TestTrainScorer -v

# This review pass's regression locks (risk-exit semantics, _ml_decide
# SELL/exclude, scorer-gate arms, outcome parsing, inject SQL + null hardening)
cd /home/zeph/paper-trader && python3 -m pytest tests/test_ml_backtest_review.py -v
```

ML/backtest test files (all offline, all deterministic):
`test_backtest.py` (PriceCache / SimPortfolio / risk-exits / indicators /
heuristic scorer / `_ml_decide` smoke + position-size caps / store
isolation), `test_decision_scorer.py` (`_to_float`, `build_features`,
`train_scorer`, prediction clamp / honesty incl. the failed-predict
`off_distribution` lock), `test_calibration.py` (the calibration
diagnostic — exact metrics + exact verdicts on deterministic synthetic
data: perfect / 0.2× biased / anti-correlated / weak-band / constant-
predictor / non-finite-drop / SELL-sign-flip / predict-exception-skip),
`test_horizon_audit.py` (2026-05-18 pass — the additive multi-horizon
capture in `_compute_decision_outcomes`: exact `8.3333/16.6667/33.3333`
5d/10d/20d returns on the synthetic curve + the 5d-present /
10d-20d-`None` past-history tail; and `horizon_audit` — exact verdict
locks via a symmetric-palindrome target that scores Spearman **exactly
0.0** against a monotone probe: `NO_HORIZON_HAS_EDGE` /
`LONGER_HORIZON_MORE_PREDICTABLE` (5d/10d noise, 20d IC 1.0) /
`5D_ADEQUATE` / `INSUFFICIENT_LONG_HORIZON` legacy-row shape /
`INSUFFICIENT_DATA` / SELL sign-flip makes a correct bearish call read
+1.0 not −1.0 / never-raises-on-garbage / `analyze` OOS-slice + missing
file / constant echo),
`test_continuous.py`
(`_pick_window`, `_trim_history`, `_append_top_decisions`,
`_compute_decision_outcomes`, `_query_news_context`, `_train_decision_scorer`),
`test_validation.py` (temporal split / OOS / permutation),
`test_ml_backtest_review.py` (a prior pass — see above),
`test_ml_backtest_coverage.py` (`_market_regime` bull/bear/sideways/unknown
classification — the `regime_mult` source for `_ml_decide` and
`_compute_decision_outcomes`; and `train_scorer`'s numpy weighted-lstsq
fallback — pickle round-trip, finite/clamped predictions, batch shape,
monotone ranking, non-finite-label guard — the entire scorer path on a
sklearn-less host, otherwise unexercised because every other
`TestTrainScorer` runs with sklearn present),
`test_execute_and_fetch_signals.py` (`_execute_decision` exact-cash BUY
boundary / one-cent-overspend block / SELL qty clamped to held position /
no-position SELL block, and `_fetch_signals` empty-URL-not-collapsed
invariant / repeated-URL dedup / top-10-by-score cut before the 5-sample —
two seams previously only reachable through the *mocked* integration test,
so their real ranking + dedup logic was unverified until this pass),
`test_ml_backtest_seams.py` (2026-05-16 pass — three seams with real logic
and *zero* prior direct coverage, found by grepping every symbol in
`tests/`: `_sector_rotation` exact trailing-return ranking incl. the
descending-sort verdict + the `start<=0` divide-by-zero guard + the
`<2 points` insufficient-history guard + future-dated-close exclusion;
`_get_decision_scorer`'s `_Dummy` except-path fallback honouring the
**exact 11-keyword `predict(**kw)` signature `_ml_decide` calls** plus
`is_trained is False` / `_n_train→0` / cached-singleton idempotence;
`_llm_annotate_outcomes`' `allowed_run_ids` restriction — the documented
contamination lock proving a winner/loser verdict does **not** leak onto an
identically-named trade in an unreviewed middle run, and an unparseable
LLM response leaves every label neutral),
`test_ml_backtest_store_views.py` (2026-05-16 pass — `BacktestStore`'s two
dashboard-facing read views had **zero** prior direct coverage yet feed
user-visible numbers: `all_runs`' `duration_days` exact calendar delta +
`annualized_return_pct` compounding formula (zero-growth → exactly `0.0`
locks the `-1.0` offset; a hand-computed `99.716` literal + an independent
`growth ** (365.25/duration)` form lock the `365.25` divisor and exponent
direction against a `365`-day or dropped-exponent regression) + `None`
before finalize + run_id-ASC ordering + `include_curves` JSON parse with
corrupt-JSON → `[]` degradation; `run_curves`' `value_pct`/`day_index`
exact normalization, unparseable point-date → `day_index None` but value
kept, corrupt `equity_curve_json` → `{rid: []}` not raise, the
`float(start_val or 1000.0)` zero-start-value divide-by-zero guard, and
empty `run_ids` → `{}`. Exact-value, not ranges — a normalization formula
change must update the literals deliberately),
`test_ml_macd_avquota_seams.py` (2026-05-16 pass — two load-bearing seams
with **zero** prior direct coverage, found by grepping every backtest
symbol against `tests/`: **`_macd`** — its numeric signal (`element [2]`,
`macd_signal`) is DecisionScorer feature slot 2 and drives `_ml_decide`'s
`adj += 0.5 if macd > 0 else -0.5`; the **input-agnostic alignment lock**
`round(m,9) == round(ema12[-1] − ema26[-1], 9)` plus a full independent
reconstruction of `signal_line` catches any shift of the
`offset = len(ema12) − len(ema26) = 14` EMA alignment a refactor could
silently introduce; label asserted only on *non-degenerate convex*
series (`m−s > 1.0`, real margin) — the linear-ramp label is a documented
float-noise sharp edge (m vs s differ at ~1e-15) whose **only** reader is
`_build_prompt`'s unused Opus path, so it is intentionally NOT locked —
plus the exact-zero `("flat", 0.0, 0.0)` tie on constant closes and the
`len < 35 → None` history guard; **`_ema`** seed-as-SMA + `v·k+prev·(1−k)`
recurrence pinned exactly (`[1..6]/p=3 → [2.0,3.0,4.0,5.0]`) — previously
only its `len<period → []` guard was touched; **`AlphaVantageNewsFetcher.
_quota`/`_inc_quota`** — CLAUDE.md §8 invariant #9 cross-restart daily
tracker: fresh/same-day-honored/corrupt-degrades, and the load-bearing
`q.get("date") == date.today()` rollover asserted end-to-end
(`yesterday calls=21 → _inc_quota → on-disk {today, 1}`, **not** 22 —
verified by reading the JSON file directly, never via `_quota()` whose
broad `except` would mask a bad write). Fully offline via the conftest
`AV_QUOTA_PATH`/`AV_CACHE_DIR` redirect; exact-value, not ranges),
`test_store_runid_partial_seams.py` (2026-05-16 pass — three
load-bearing seams with **zero** prior direct coverage, found by
grepping every backtest/continuous symbol against `tests/`:
**`_next_run_id`** the continuous-loop monotonic id allocator —
COALESCE guard on an empty table (→ 1, never `int(None)+1`) and
`MAX(run_id)+1` **not** `COUNT(*)+1` on a *non-contiguous* table (runs
3,9 → 10) so a post-`_trim_history` sparse table can't make the next
`upsert_run` overwrite a survivor; **`BacktestStore.upsert_run`
INSERT-vs-UPDATE branch** — a 2nd call for the same run_id with
deliberately different seed/window changes **only** `status` and
preserves the original `seed`/`start_date`/`end_date`/`start_value`/
`started_at` (still one row — UPDATE, not a 2nd INSERT): the
store-layer "completed historical run is not overwritten" guarantee,
asserted for the first time though `upsert_run` is a setup helper in 12
files; **`update_partial_progress` vs `finalize_run` arithmetic** —
both share `(value − 1000)/1000·100` (50.0 at $1500, −2.5 at $975,
exact) but the partial path must **not** write `spy_return_pct`/
`vs_spy_pct`/`status`/`completed_at`, and `vs_spy = total − spy` lives
**only** in `finalize_run` (pinned via a +50% run under SPY +80% →
`−30.0` to lock the subtraction *direction*). Exact-value, not ranges),
`test_ml_backtest_store_detail_sell.py` (2026-05-16 pass, 9th
consecutive no-new-bug review — two more zero-coverage seams found by
grepping every backtest symbol against `tests/`: **`BacktestStore.
run_detail`** — the read view behind `/api/backtests/<run_id>`; its
siblings `all_runs`/`run_curves` were locked the prior pass but
`run_detail` was not, despite real logic — missing-run → `None` (not
`{}`/raise, so the endpoint 404s not 500s), the `(sim_date ASC, id
ASC)` ordering on **both** child tables locked via an out-of-order
insert with a same-day pair (a `sim_date DESC` *or* `id DESC` tiebreak
regression scrambles the dashboard's trade/decision tables and fails
the exact-sequence assertion), corrupt-`equity_curve_json` → `[]`
degradation (a raise here 500s the endpoint), valid-curve round-trip;
**`backtest._sell`** the `SimPortfolio` mutator (distinct from
`strategy._sell`) — every backtest SELL / stop-loss / take-profit exit
routes through it yet it had **zero** direct unit coverage (only
transitive via `_enforce_risk_exits`/`_execute_decision`, which clamp
qty *before* calling it, so its own over-sell clamp and the
`pos["qty"] <= 1e-6` deletion boundary were never asserted in
isolation): no-position → `0.0` + no mutation, partial sell leaves
`avg_cost` untouched & credits cash == proceeds exactly (no rounding in
`_sell`), over-sell clamps to held qty & closes the row, the `1e-6`
epsilon boundary pinned both sides (residual 1e-7 → deleted, 1e-5 →
kept). The continuous-loop "old results are not overwritten without
version/timestamp" property is **not** re-tested here — it is already
locked by `test_store_runid_partial_seams.py`'s `upsert_run`
INSERT-vs-UPDATE seam above (a 2nd call for the same run_id changes
**only** `status`, preserving `seed`/`start_date`/`end_date`/
`start_value`/`started_at`)).

> A non-network collection error from an *untracked, out-of-scope* test
> file (e.g. one a parallel review agent left mid-flight that imports a
> not-yet-created module) will abort `pytest tests/` collection for the
> whole directory. It is **not** an ML/backtest regression — verify your
> own work with `--ignore=tests/<that_file>.py` and leave the file for its
> owner; never `git add -A` it into an unrelated review commit.

All tests are offline — `tests/conftest.py` redirects `SCORER_PATH`,
`PRICE_CACHE_PATH`, `BACKTEST_DB`, and the various cache paths to
`tmp_path` so a test run never clobbers real data. Synthetic deterministic
prices come from the `synthetic_prices` fixture. No test should reach the
network; if you add one that does, mock `yfinance.Ticker` (see
`test_variable_windows.py::_make_fake_hist`).

### Bug-fix workflow

For automated review agents that touch ML / backtest code:

1. **Read first**: `CLAUDE.md` §6 (the two-model section), this file's
   feature table, then the function you're about to edit. The invariants
   in `CLAUDE.md` §8 (especially #1 backtest live-only filter, #5 scorer
   gate threshold, #6 claude subprocess cap) are load-bearing.
2. **Be surgical**: prefer a 3-line edit over a refactor. The continuous
   loop runs unattended; cosmetic churn risks breaking pickle
   compatibility for `data/ml/decision_scorer.pkl` or schema
   compatibility for `data/decision_outcomes.jsonl`.
3. **Run tests before committing**:
   `python3 -m pytest tests/ -v 2>&1 | tail -20`. Failures block
   the commit.
4. **Append an entry to `data/run_log.md`** with the
   `## YYYY-MM-DDTHH:MM:SSZ` header described at the top of that file.

### Common pitfalls

- **Pickle compatibility** — adding a feature to `build_features`
  invalidates `data/ml/decision_scorer.pkl`. The `predict()` exception
  handler now logs once per instance (was silent — masked exactly this
  case during a feature rollout). After a feature change, force a retrain
  by deleting the pickle before the next continuous-loop cycle.
- **`_to_float` and numpy types** — `np.float32` is *not* a Python `float`
  subclass (`np.float64` is). `_to_float` falls back to an **`np.number`**
  check (NOT `np.generic`: `np.generic` also matches `np.str_`/`np.bool_`,
  and `np.isfinite(np.str_("x"))` raises an *unhandled* `TypeError` that
  would propagate out of `build_features` and crash `train_scorer`; numpy
  strings/bools must take the safe default like Python `str`/`bool` do).
  If you add new numpy inputs, verify they pass through. It rejects
  every non-finite value (NaN **and** ±inf) on both the Python and numpy
  branches via `math.isfinite` / `np.isfinite` — this is load-bearing: a
  single `decision_outcomes.jsonl` row with a non-finite `forward_return_5d`
  poisons `train_scorer`'s `y` vector, `MLPRegressor.fit` raises, and
  `_train_decision_scorer` swallows it — silently wedging scorer retraining
  for that cycle and every cycle after (the row persists in the 5000-record
  tail). Pinned by `tests/test_decision_scorer.py::TestToFloat` +
  `::TestTrainScorer::test_handles_non_finite_forward_return`.
- **`dict.get(k, default)` does NOT default a JSON `null`** — it only
  substitutes the default when the key is *absent*; an explicit `null`
  value still returns `None`. `_inject_and_train` reads
  `winner_training.jsonl` (which mixes top-decision, opus-lesson and
  opus-trade-label record shapes) and a single line with `"ai_score": null`
  or `"weight": null` reaching `float(None)` raises `TypeError` — caught by
  the function's broad outer `except`, which returns `"inject err: …"` and
  injects **zero** rows that cycle, so ArticleNet never retrains. The fix
  is the codebase's standard `float(rec.get("ai_score") or 0.0)` /
  `… or 1.0` idiom (same class as the `_ml_decide`
  `float(a.get("score") or 0.0)` hardening). Pinned by
  `tests/test_ml_backtest_review.py::TestInjectAndTrain::test_null_ai_score_and_weight_do_not_abort_batch`.
  That test also locks the **11-column INSERT alignment** (id…full_text),
  `ai_score == kw_score == min(10, ai·weight)`, hard-coded `urgency=0`, and
  `INSERT OR IGNORE` dedup by `_aid(url, title)`.
  The same null-default class lives in `_ml_decide`'s article loop: it
  hardens **both** `score` (`float(a.get("score") or 0.0)`) **and**
  `tickers` (`list(a.get("tickers") or [])`). A `"tickers": null` makes
  `list(None)` raise an uncaught `TypeError` — unlike `_inject_and_train`
  there is **no** broad `except` here, so it kills the whole run thread
  mid-cycle (run recorded `failed`, zero decisions), the same blast radius
  the adjacent `score` hardening comment describes. Pinned by
  `tests/test_ml_backtest_review.py::TestMlDecideMalformedArticles` (None
  `tickers` ⇒ same decision as the well-formed article; None `score` ⇒
  clean HOLD, never an exception).
- **Hardcoded cross-repo paths must be module-level for testability** —
  `_inject_and_train` writes into digital-intern's `articles.db`. Its path
  is now the module constant `run_continuous_backtests.DIGITAL_INTERN_ARTICLES_DB`
  (was a function-local string, untestable). Tests monkeypatch it +
  `WINNER_JSONL` + `subprocess.run` to exercise the injection offline. Keep
  any new cross-repo path at module scope for the same reason.
- **`_enforce_risk_exits` trading-day membership is O(1)** —
  `cur not in prices.trading_days` was a list scan inside a per-calendar-day
  loop; over a 1–10yr continuous-loop window that is tens of millions of
  comparisons per run. It now snapshots `set(prices.trading_days)` once at
  function entry (behavior-identical — no PriceCache change, so the
  `synthetic_prices` fixture that builds `PriceCache` via `__new__` is
  unaffected). The SL/TP exit semantics it guards (stop-loss priority via
  `if sl … elif tp …`, full-qty liquidation, no double-fire after close)
  are locked by
  `tests/test_ml_backtest_review.py::TestEnforceRiskExitsSemantics`.
- **Forward leakage** — anything that reads news must filter on
  `url NOT LIKE 'backtest://%'` and `source NOT LIKE 'backtest_%'` /
  `'opus_annotation%'`. The live `signals.py` and the backtest
  `_load_local_articles` / `_query_news_context` already do this; new
  readers must mirror it.
- **Single sqlite3 connection across threads** — `BacktestStore.conn` is
  shared across run threads and the background `_opus_annotate` thread.
  Every read / write must hold `store._lock`. If you add a new query path,
  copy the locking pattern from `_trim_history` / `_append_top_decisions`.
- **Resolve module-global paths at call time, not as default args** —
  `def __init__(self, path=BACKTEST_DB)` binds the global's *value* at
  import, so conftest's `monkeypatch.setattr(bt, "BACKTEST_DB", tmp)` is a
  no-op for that call and the no-arg `BacktestStore()` silently hits the
  real persistent DB (this caused an order-dependent flaky test). Use
  `path=None` then `path = path or BACKTEST_DB` inside the body. Same rule
  for any new constructor that touches `SCORER_PATH` / `CACHE_DIR` / a
  cache path the conftest redirects. Locked by
  `tests/test_backtest.py::TestBacktestStoreIsolation`.
- **`SAMPLE_EVERY_N_DAYS = 1`** — backtests sample every trading day.
  Don't change this casually; the continuous loop's timing budget
  assumes a year-long sim completes in ~minutes per run.
- **Scorer-train status must stay truthful** — in
  `run_continuous_backtests.py::_train_decision_scorer`, `train_scorer()`
  pickles the model to `SCORER_PATH` and returns `status="ok"` *before* the
  temporal-OOS diagnostic runs. The OOS block (`DecisionScorer()` reload +
  `evaluate_scorer_oos`) has its **own** `try/except` that degrades to
  `oos_rmse=n/a (...)`. Do not collapse it back into the outer
  train `try/except`: a post-train diagnostic crash would then surface as
  `scorer err` on the operator-facing log/Discord even though the scorer is
  trained and the next cycle's singleton reset deploys it — a false
  "scorer broken / gate never engages" signal. Locked by
  `tests/test_continuous.py::TestTrainDecisionScorer::test_oos_eval_failure_does_not_mask_successful_train`.
- **Run-return weight is applied twice into the ArticleNet feed (by
  design, not a bug)** — `_append_top_decisions` folds the per-run weight
  `w = 0.5 + 0.5·(ret−min)/span` into the JSONL `ai_score`
  (`w·5.0` for BUY, `w·0.5` for SELL) *and* stores the bare `w` as
  `weight`. `_inject_and_train` then writes `eff = min(10, ai_score·weight)`
  into digital-intern's `articles.db`, so a top-run BUY lands at `≈5·w²`
  (`w∈[0.5,1.0] → eff∈[1.25,5.0]`) — the run-quality term is **squared**,
  intentionally compressing lower-ranked runs harder than a linear weight
  would. This only affects ArticleNet's training emphasis (a *separate*
  model in digital-intern), never the DecisionScorer or any trade. Opus
  annotation rows side-step it (`weight=1.0`, so `eff=ai_score`). Do not
  "linearise" this in a surgical pass — it perturbs ArticleNet training
  dynamics and is out of scope for the ML/backtest review.
- **The `_get_decision_scorer` `_Dummy` fallback is a load-bearing
  contract, not a stub** — when the real `DecisionScorer` import or
  instantiation raises, the singleton degrades to an inline `_Dummy`.
  `_ml_decide` then calls `scorer.predict(**kwargs)` with a **fixed
  11-keyword signature** (`ml_score, rsi, macd, mom5, mom20, regime_mult,
  ticker, vol_ratio, bb_pos, news_urgency, news_article_count`) and reads
  `scorer.is_trained` + `getattr(scorer, "_n_train", 0)`. If a refactor
  "tidies" the Dummy into a positional or arg-less `predict`, **every**
  parallel backtest run thread throws `TypeError` mid-cycle the moment the
  real scorer can't load (recorded `failed`, zero decisions) — a silent,
  total backtest outage that only manifests on the import-failure path the
  happy-path tests never take. Keep `def predict(self, **kw): return 0.0`,
  `is_trained = False`. Locked by
  `tests/test_ml_backtest_seams.py::TestDecisionScorerDummyFallback`.

### 2026-05-17 review pass (GDELT coverage · run reaper · OOS dir-skill)

- **GDELT permanent-vs-transient errors (`backtest.py::GDELTFetcher.fetch`,
  committed `8899c16`).** GDELT DOC 2.0 only indexes ~2017-onward; a
  pre-coverage date raises a *deterministic* `ValueError` ("The query was
  not valid … Invalid query start date"). The fetcher previously treated
  this as transient — 3 retries with 20+40+60s backoff **and no cache
  write**, so the continuous loop (windows back to 1993) re-attempted it
  every cycle for hours (`continuous.log` was wall-to-wall these). Now a
  permanent message (`"not valid"` / `"invalid query"`, matched on text not
  type) breaks with **zero backoff** and **negative-caches `[]`** so the
  warm-cache `exists()`-filter and the tier-3 disk lookup skip it forever
  after. Transient errors (rate-limit / connection drop) keep the full
  retry+backoff and are **never** cached (caching a transient failure would
  poison a temporarily-failing date for the loop's life). A
  legitimately-empty result on a *covered* date is still cached (unchanged).
  Locked by `tests/test_gdelt_coverage_20260517.py`.

- **Orphaned-run reaper (`run_continuous_backtests.py::_reap_orphaned_runs`,
  called once at `main()` startup; committed `05b4df2`).** A run thread
  hard-killed by OOM/SIGKILL never reaches `finalize_run` *or* the
  `run_all` wrapper's `upsert_run("failed")` (that fallback only fires on a
  *caught* exception), so the `backtest_runs` row stays `status='running'`
  forever — the CLAUDE.md §11 "Backtest dashboard shows running forever"
  symptom (15 such stale rows were live in `backtest.db`). On a fresh loop
  start any pre-existing `running` row is orphaned (prior process is gone);
  the `max_age_hours=6.0` guard is defensive (no real run exceeds minutes).
  Runs single-threaded before any new run launches — no race. Best-effort:
  a DB hiccup never blocks loop start. Locked by
  `tests/test_continuous_review_20260517.py::TestReapOrphanedRuns`.

- **OOS directional skill in the continuous-loop status line
  (`_oos_rank_metrics`, appended to `_train_decision_scorer`'s string as
  `oos_diracc=` / `oos_ic=`; committed `05b4df2`).** `oos_rmse` answers
  *how big is the error* but the `_ml_decide` gate only acts on the
  prediction's **sign/bucket** (±10/±5/0), so a scorer with
  `oos_rmse ≳ σ(target)` (the documented current state) can still be
  gate-useful **iff it gets direction right**. `oos_diracc` = held-out
  sign-match fraction (zeros excluded — no directional truth); `oos_ic` =
  tie-aware Spearman(pred, realized) **reusing `ml.calibration._spearman`**
  (single source of truth — the tie-awareness is load-bearing because the
  scorer clamps to ±50 and a naïve argsort fabricates rank skill there; a
  constant predictor must read `oos_ic=+0.00`, not +1.00). Mirrors
  `validation.evaluate_scorer_oos`'s exact 11-kwarg predict signature +
  SELL sign-flip so it describes the **same** path the gate uses. Guarded
  *separately* from the `oos_rmse` block (own try/except → `n/a`) so a
  post-train diagnostic crash can't mask a successful train (the
  "scorer-train status must stay truthful" discipline). **Interpretation:**
  read `oos_diracc` next to `oos_rmse` — `oos_diracc ≤ 0.5` with
  `oos_rmse ≳ σ` means the BUY-conviction gate is riding noise;
  `oos_diracc` materially > 0.5 is the only evidence the gate's sign
  decision carries edge despite the poor RMSE. Diagnostic only — changes no
  model/gate (training-dynamics is out of scope; CLAUDE.md §6). Locked by
  `tests/test_continuous_review_20260517.py::TestOosRankMetrics`.

### 2026-05-17 review pass #2 (label-hygiene audit · live findings)

Hybrid quant pass (debug + feature + live validation). **Zero code bugs
found** — 10th consecutive no-new-bug review of the ML/backtest core.
One read-only diagnostic added; the rest is reported live findings, not
silent fixes (every actionable item is a training-dynamics change the
doc repeatedly scopes out).

- **Training-label hygiene audit (`paper_trader/ml/label_audit.py`,
  committed `9c844c9`).** The exact read-only sibling of
  `ml/calibration.py`: no train, no pickle, no `decision_outcomes.jsonl`
  rewrite, no `build_features`/`N_FEATURES` touch — safe against the
  unattended loop. `PriceCache` fetches `yf.history(auto_adjust=False)`,
  so a reverse split (DFEN's 2024-06 1:5) injects a step discontinuity
  recorded as a `forward_return_5d` of **+180.04%** (`mom5=-64.04` — a
  textbook split signature). The inference head is clamped to
  `PRED_CLAMP_PCT`, but **nothing measured how many *labels* sit past
  that bound**, and `train_scorer`'s run-quality oversampling up-weights
  them 2–4×. The audit reports the extreme-label rate (`|fwd| >
  EXTREME_RETURN_PCT`, imported `== PRED_CLAMP_PCT` — single source of
  truth, the `_oos_rank_metrics`-reuses-`_spearman` precedent) vs the
  documented ~0.5% real baseline, plus per-ticker worst offenders and an
  *informational-only* directional-anomaly subcount (it also fires on
  genuine 2020-03 COVID mean-reversions, so it never drives the verdict).
  Verdicts `CLEAN`/`ELEVATED`/`CONTAMINATED` are exact-testable module
  constants; the `CONTAMINATED` hint points at the **documented**
  remediation (delete the pkl, let the loop retrain) and explicitly says
  *do NOT winsorize `y` in `train_scorer`* — that is the out-of-scope
  training-dynamics change this tool exists to inform, not perform.

  ```bash
  # Label hygiene of the accumulated outcomes tail (read-only).
  # Exit 0 = CLEAN/INSUFFICIENT, 2 = ELEVATED/CONTAMINATED.
  cd /home/zeph/paper-trader && python3 -m paper_trader.ml.label_audit
  cd /home/zeph/paper-trader && python3 -m pytest tests/test_label_audit_20260517.py -v
  ```

  **Interpreting it (live finding).** On the current 5000-row corpus the
  aggregate is `CLEAN` (25/5000 = 0.500%, dead-on the documented
  baseline) — but the per-ticker view is the payoff: **DFEN 11/523 =
  2.10%** (4× the corpus rate), FAS 1.20%, MSTR 1.36%, all concentrated
  on reverse-split / COVID-crash dates. A corpus-wide `CLEAN` masks a
  per-name concentration the scorer's DFEN/FAS tail predictions ride on.
  Read this next to the `calibration` verdict, exactly like `oos_rmse`.
  Locked by `tests/test_label_audit_20260517.py` (exact verdict
  boundaries · strict `|fwd|>PRED_CLAMP_PCT` · single-source-of-truth ·
  non-finite drop · per-ticker sort · directional-anomaly-informational ·
  `_load_outcomes` corrupt-line skip).

- **Live finding — the running continuous loop is stale (NOT a code
  bug; operator action).** PID `1086675` started `02:21:35`; the GDELT
  permanent-error short-circuit (`8899c16`, `06:52`) and the
  orphaned-run reaper (`05b4df2`, `06:57`) were committed *after* it
  booted. Evidence it is running pre-fix code: `continuous.log` shows
  `Invalid query start date` errors still doing the full 3-retry
  20+40+60s backoff (the exact pathology `8899c16` removes), its
  `scorer ok …` lines lack the `oos_diracc=`/`oos_ic=` fields `05b4df2`
  adds, and `backtest.db` has **19 orphaned `running` rows** the
  startup reaper would have swept. Remediation is the documented
  `/api/build-info`-`stale` protocol: a clean SIGTERM between cycles +
  restart. Left for the operator — restarting a user-owned production
  loop is outward-facing and out of an automated pass's remit.

- **Live finding — `_llm_annotate_outcomes` has been structurally inert
  since deployment.** The `anthropic` SDK is importable but no
  `ANTHROPIC_API_KEY`/auth is configured (the whole system is `claude`
  CLI-subprocess-authed, not SDK), so every cycle logs
  `LLM annotation failed: Could not resolve authentication method` and
  `LLM labels: 0 endorsed, 0 condemned`. **All 5000 rows in
  `decision_outcomes.jsonl` carry `llm_quality_label: 0`** — the
  `{1: 3.0, -1: 0.1, 0: 1.0}` training-weight multiplier in
  `train_scorer` (a documented load-bearing training feature — see the
  "Run-return weight" pitfall) has *never once been live*. Not fixed
  here: routing it through the `claude` CLI like `_opus_annotate`
  activates a 3×/0.1× sample-weight on the unattended loop — a
  training-dynamics change requiring an explicit decision + pkl bump,
  not a surgical edit. Reported for that decision.

- **Live finding — in-sample calibration is optimistic vs OOS.**
  `python3 -m paper_trader.ml.calibration` on the live pkl
  (`n_train=3876`) reports `WELL_CALIBRATED` (spearman 0.51, monotone
  deciles, 1.85pp decile error) — but the continuous loop's
  trustworthy temporal-holdout `oos_rmse` is **14.62** on the latest
  matching cycle (range 8.18–17.36 across recent cycles), straddling /
  exceeding the documented σ(aligned target) ≈ 11.7. The scorer's
  out-of-sample RMSE is at or worse than predicting the mean even
  though it gates BUY conviction once `_n_train ≥ 500`. Always pair the
  in-sample `WELL_CALIBRATED` with `oos_rmse` (and now `label_audit`) —
  the in-sample verdict alone overstates the edge. Backtests themselves
  are healthy (486 complete, 0 null/NaN finals, fresh `completed_at`),
  but the per-cycle "best run +1294% / vs_spy +1202%" line sits next to
  a same-regime "+12% / vs_spy −80%" — the documented leveraged-beta
  dispersion, not repeatable alpha.

### 2026-05-17 review pass #3 (per-persona strategy-quality leaderboard)

Hybrid quant pass (debug + feature + live validation). **Zero code bugs
found in the existing ML/backtest core** — 11th consecutive no-new-bug
review. One read-only diagnostic added (the third in the
`calibration.py` / `label_audit.py` family); the rest is reported live
findings.

- **Per-persona strategy-quality leaderboard
  (`paper_trader/ml/persona_leaderboard.py`).** The exact read-only
  sibling of `ml/calibration.py` and `ml/label_audit.py`: no train, no
  pickle, no `decision_outcomes.jsonl`/`backtest.db` write (opens
  `backtest.db` strictly `mode=ro`), no `build_features`/`N_FEATURES`
  touch — safe against the unattended loop. Two prior diagnostics
  measure **scorer** quality; nothing measured **strategy/persona**
  quality, despite `backtest.db` holding ~490 `complete` runs each
  mapped to one of 10 personas. It **imports `backtest.persona_for`**
  (single source of truth — the `_oos_rank_metrics`-reuses-`_spearman`
  precedent), so a `PERSONAS` reorder can never silently desync the
  historical aggregates. Per persona it reports the **median** vs_spy
  (the honest central tendency — the per-cycle "best run +1294%" line is
  the max of a leveraged-beta draw; the *mean* is dominated by a few 3×
  bull-window rips), win-rate vs SPY, median total return, and
  risk-shape from the stored `equity_curve_json` (max drawdown,
  annualised Sharpe-equivalent, %-time-underwater — none surfaced
  anywhere before). Per-persona verdicts `EDGE` / `FLAT` / `DRAG` /
  `INSUFFICIENT` and overall `HEALTHY` / `HAS_DRAG_PERSONA` are
  exact-testable module constants; the `HAS_DRAG_PERSONA` hint points
  at the *separate, explicit* prune/re-tune decision and explicitly says
  **do NOT** change `PERSONAS`/`_PERSONA_BOOSTS` from the read-only
  audit (the out-of-scope strategy-dynamics change this tool exists to
  inform, not perform — same discipline as `label_audit`'s
  do-not-winsorize hint).

  A **numerical-robustness gap in the new module itself** was caught by
  its own exact-value test before commit: a flat/cash-parked or
  constant-return equity stretch has a returns std of pure
  float-representation noise (~1e-16, because `0.1` is not exactly
  representable), which sails past a naïve `sd > 0` and divides a real
  mean by ~1e-17 → a ~1e16 "Sharpe" that would dominate the per-persona
  median. Fixed with a `sd > 1e-9` floor (any genuine daily-variance
  curve is ≫1e-9; the floor cleanly separates degenerate from real).

  ```bash
  # Per-persona strategy-quality leaderboard over the live backtest.db.
  # Exit 0 = HEALTHY/INSUFFICIENT_DATA, 2 = HAS_DRAG_PERSONA.
  cd /home/zeph/paper-trader && python3 -m paper_trader.ml.persona_leaderboard
  cd /home/zeph/paper-trader && python3 -m pytest tests/test_persona_leaderboard_20260517.py -v
  ```

  **Interpreting it (live finding).** On the current ~490-run corpus the
  verdict is **`HAS_DRAG_PERSONA`**: 9 of 10 personas have a positive
  median alpha (Global Macro / Pure Speculator ≈ +96–101pp median, but
  with ~44–45% median max-drawdown and >90% time underwater — leveraged
  beta, not low-risk skill; Value Investor is the best risk-adjusted at
  Sharpe ≈ 0.79 / 27% maxDD / 92% win-rate), but **`Sector Rotator`
  (persona 7) is a `DRAG`: median vs_spy ≈ −1.9pp, mean ≈ −3.6pp,
  win-rate 45%, Sharpe ≈ 0.34 across 49 runs** — it does not beat SPY at
  the median and is contributing variance, not alpha. Its
  `_PERSONA_BOOSTS[7]` row (`FAS 2.5, DFEN 2.0, LABU 2.0, BOIL 1.5,
  XLE 2.0, XLF 2.0, XLI 1.5`) is the prime candidate for a future
  (separate, explicit, pkl/strategy-dynamics-aware) prune or re-tune —
  **reported, not actioned**, exactly like the pass #2 `_llm_annotate`
  finding. `ESG / Thematic` is the only other sub-EDGE persona (`FLAT`,
  median +16pp, below the +20pp strong bar). Read this next to the
  `calibration` and `label_audit` verdicts — it is the missing
  strategy-side measurement.

- **Live findings reconfirmed (operator action, NOT code bugs).** The
  pass #2 findings still hold on the running loop: `backtest.db` has
  **16 orphaned `running` rows** (the loop, PID 1086675 started 02:21,
  predates the `05b4df2` startup reaper — same documented `stale`
  protocol: clean SIGTERM + restart, left for the operator);
  `decision_outcomes.jsonl` is at the **5000-row cap** with **every row
  `llm_quality_label: 0`** (the `_llm_annotate_outcomes` SDK-auth
  finding from pass #2 — the 3×/0.1× training-weight multiplier has
  still never been live); calibration on the live pkl (`n_train=3876`)
  is `WELL_CALIBRATED` in-sample (spearman 0.51) while the decile tails
  over-predict (d1 −15.7 vs −10.7; d10 +15.4 vs +11.9) — pair with
  `oos_rmse` as documented; `label_audit` is `CLEAN` aggregate (0.500%)
  with the same DFEN 2.10% per-ticker split concentration. Backtests are
  healthy (485–490 complete, 0 null/NaN finals, `completed_at` fresh to
  the current hour).

### 2026-05-17 review pass #4 (stale-mark surfacing · core hybrid pass)

- **Feature shipped (commit `f834c93`): stale price marks are now
  surfaced, not silent.** `_portfolio_snapshot` (strategy.py) already
  fell back to `avg_cost` when a live price was unavailable, so
  `current_price == avg_cost` and `unrealized_pl == $0.00` — **visually
  identical to a genuinely flat position**. Seen live this pass: `MU`
  held at `avg == mark == 724.12`, P/L `$0.00`, which Opus and the
  operator both read as "flat" when the mark was actually *unknown*. The
  snapshot now emits a `stale_mark: bool` on every enriched position
  (`True` only when the live stock/option-chain lookup returned `None`
  and we fell back; **`False` for a deliberate expired-option intrinsic
  settlement** — that is a real mark, not a missing price). `_build_payload`
  appends `[STALE MARK: live price unavailable — shown at cost, P/L
  unreliable]` to the PORTFOLIO line Opus reads (advisory text only — no
  gating, invariants #2/#12), and `reporter._portfolio_lines` appends an
  additive `⚠ STALE` tag. The reporter change is **byte-identical for the
  existing Discord path**: `store.open_positions()` table rows carry no
  `stale_mark` key, so a genuinely-flat `$0.00` is never falsely flagged;
  only a missing-price mark is. `stale_mark` also rides into
  `portfolio.positions_json` (via `update_portfolio`) so any `/api/state`
  / `/api/portfolio` consumer gets it for free. Applies on next
  paper-trader restart (the documented pattern for every recent feature).
  Locked by `tests/test_core_strategy.py::TestStaleMarkFlag` (stock
  no-price → flagged + behaviour preserved; stock with price → not stale;
  live option `None` → flagged + still avg_cost; expired-option intrinsic
  → NOT stale; `_build_payload` annotates the stale name and not the
  fresh one) and `tests/test_core_reporter.py::TestPortfolioLines`
  (annotated when flagged; absent/`False` key → no annotation).

- **No core bug fixed (bugs_fixed = 0, no Phase-1 commit).** The 7
  in-scope core files (`runner`, `reporter`, `signals`, `strategy`,
  `dashboard`, `market`, `store`) were re-audited for logic / race /
  comparison / off-by-one / state-transition errors against fresh eyes;
  none found. The `core_*` suite is green (293 passed incl. the 9 new
  tests; +165 in snapshot/payload-adjacent modules). Per the Phase-1
  commit guard, no bug was fabricated.

- **Live findings (operator action, NOT code bugs).**
  (1) **Claude org monthly usage limit hit** — runner.log shows repeated
  `claude err (rc=1): "You've hit your org's monthly usage limit"`;
  `/api/decision-health` reads **NO_DECISION 59% all-time / ~27% last
  24h**, FILLED only 3.2%, `hours_since_fill ≈ 9`. The trader degrades
  gracefully (Opus→Sonnet fallback→retry→`NO_DECISION` recorded with the
  raw excerpt; circuit breaker can't help a quota wall) but is mostly
  *not trading*. Operator must address billing.
  (2) **Hourly Discord summaries failing to send** — `[runner] hourly
  send returned False` recurring since ~16:54 UTC; the summary composes
  correctly (format verified: Equity/Cash/P&L/S&P/Positions/Recent/SESSION/
  BEHAVIOURAL) but `openclaw` fails during the quota window. Auto-retries
  next cycle (correct behaviour, not a bug).
  (3) **Running process is stale/behind** — `/api/build-info`
  `stale:true`, `boot_sha 92fcd2f` vs `head f834c93` (`behind:3`); the
  on-disk fixes incl. this pass's feature do **not** apply until an
  operator restart of `paper_trader.runner` (by design, surfaced
  correctly).
  (4) **Extreme concentration** — `/api/risk` `HIGH`: LITE 60.9% / top3
  98.1% / cash 1.9% ($18.49). By design (no hard limits, invariant #12)
  but a live-desk red flag worth the operator's eye.
  (5) **LITE/MU marked ~10× plausible levels** (LITE ~$970–1006, MU
  ~$724–803 vs real-world ~$80–130). The system is internally consistent
  (buy & mark from the same yfinance source) and yfinance returns `None`
  for them right now so it is unverifiable from here — but **position
  sizing runs on these marks**; the operator should verify
  `yfinance.fast_info` is not returning a wrong-instrument price.
  (6) **Dashboard intermittent multi-second stalls / `CLOSE-WAIT`
  pileup under concurrent sibling-agent load** (recovered to 1–11 ms on
  isolated requests). Documented fragility (`dashboard.py:176-187` —
  `yfinance`/`requests` has no socket timeout, a hung call pins an SWR
  worker). No safe in-scope fix; reported.

### 2026-05-17 review pass #5 (ML+backtest hybrid · poison-cache fix · skill-trend reader)

- **Bug fixed (commit `6e3fa55`): poisoned per-window price caches.**
  `PriceCache._load` accepted ANY cached `prices_*.json` whose `_meta`
  matched and tickers were a superset — **including the 34 of 177 (19%)
  live per-window caches whose SPY series is `{}`** from a transient
  yfinance failure at build time. `_build_trading_days` fell back to
  another ticker's calendar so the run completed, but
  `returns_pct("SPY",…)` then returned 0.0 → `vs_spy_pct` was fabricated
  (`== total_return`) with no real benchmark, and that feeds the live
  trader's `_ml_is_qualified` median-alpha gate (CLAUDE.md §15) every
  cycle the window is redrawn. The "Continuous-loop durability & honesty"
  note above flagged this "Still open … out of surgical scope" by bundling
  it with a strategy-side gate fix; the **cache-side half is surgical and
  in-domain**, so it was taken (the gate-side half stays in
  core/strategy.py — reported there, not actioned here).
  A paired benchmark-integrity guard in `_load` now (1) rejects a cached
  payload whose SPY series is empty when SPY is requested (SPY has data to
  its 1993 inception ⇒ an empty series is ALWAYS a transient fetch
  failure, never a real gap) and re-downloads, and (2) skips persisting a
  fresh download whose SPY series is still empty so the next draw retries
  rather than re-poisoning. Guard inert when SPY ∉ watchlist. The run
  still completes off the fallback calendar; `run_one` keeps writing the
  honest `benchmark_unavailable` note. The 34 on-disk poisoned files
  **self-heal on next redraw once the loop restarts on this code** (inert
  until restart — the documented restart-required pattern). Locked by
  `tests/test_pricecache_benchmark_poison.py` (both guard halves +
  healthy-cache-accepted + SPY-not-requested no-op);
  `test_integration_backtest.py::TestBenchmarkUnavailableNote` still green.

- **Feature shipped (commit `6a9eb66`): scorer-skill trend reader.**
  AGENTS.md called `data/scorer_skill_log.jsonl` *"the canonical
  instrument for the negative-OOS-skill question"* but there was **no
  reader** — a quant had to `tail` JSONL and eyeball it.
  `paper_trader/ml/skill_trend.py` answers it with an exact verdict
  (`INSUFFICIENT_DATA` / `BEATS_MEAN_PREDICTOR` / `NEGATIVE_OOS_SKILL` /
  `DIRECTIONAL_BUT_HIGH_ERROR` / `BORDERLINE`) plus `trend`
  (IMPROVING/DEGRADING/STABLE) and `gate_active_fraction`. The comparator
  baseline is computed **fresh** from the current `decision_outcomes.jsonl`
  temporal-OOS slice (reusing `validation.split_outcomes_temporal` + the
  SELL sign-flip) — NOT the regime-stale σ≈11.7 literal. RMSE of a
  constant mean-predictor == population σ of the OOS targets, so it is the
  exact regime-current comparator for the ledger's `oos_rmse`. Same
  discipline as `ml/calibration.py`: read-only, no train/pickle/feature/
  trade touch, never raises — safe against the live loop.
  ```bash
  cd /home/zeph/paper-trader && python3 -m paper_trader.ml.skill_trend
  cd /home/zeph/paper-trader && python3 -m pytest tests/test_skill_trend.py -v
  ```
  17 exact-value verdict locks in `tests/test_skill_trend.py`.

- **Quant finding: the documented σ≈11.7 OOS baseline is regime-stale.**
  The fresh mean-predictor baseline on the *current* `decision_outcomes.jsonl`
  is **6.24** (temporal-OOS slice); the full 5000-row tail's realized 5d
  std is **7.49** — both far below the AGENTS.md σ≈11.7. The 3 ledger
  cycles since the loop restarted show `oos_rmse` ≈ 7.85/7.87/10.36 with
  `oos_ic` ≈ −0.0/0.10/−0.03. So the **relative** conclusion holds (oos
  error ≥ a mean predictor, ~zero rank-IC ⇒ no demonstrated OOS skill) but
  every **absolute** figure in the "negative-OOS-skill" note above is
  outdated — read `skill_trend` for the live numbers, not the literals.

- **Quant finding: scorer overfits (in-sample optimistic).** Live pkl
  `n_train=3283`, gate active. Calibration in-sample is `WELL_CALIBRATED`
  (spearman 0.48, monotone deciles, 1.59pp mean decile err) and sampled
  in-sample sign accuracy is **0.61**, but the ledger's OOS `dir_acc` ≈
  0.50 and `oos_ic` ≈ 0 — the in-sample/OOS gap is the overfitting
  signature. Decile tails over-predict by ~6pp (d10 pred +12.7 vs
  realized +6.9). The gate modulates real BUY conviction on a signal with
  near-zero demonstrated OOS edge; trust sign modestly, distrust the
  predicted magnitude. Reported, not actioned (model-dynamics change is
  out of surgical scope — CLAUDE.md §6).

- **Quant finding: pre-2020 windows trade a drastically narrowed
  universe.** Every 2x/3x single-stock leveraged ETF
  (NVDU/MSFU/AMZU/TSLT/CONL/TSLL/PLTU/BITU/BITX/ETHU/LNOK) and crypto-lev
  name returns `possibly delisted; no price data` for windows before its
  inception — handled gracefully (`prices[t]={}`), **not a code bug**, but
  a backtest-realism caveat: an old-window persona's return reflects a
  smaller, less-levered universe than the live watchlist, so its
  `_PERSONA_BOOSTS` leveraged-ETF tilts are partly inert there.

- **Live health.** 480 complete / 15 failed (all `[reaped: orphaned
  running row]` — the per-cycle reaper works) / 5 running (1.6h, under the
  6h guard). 0 NaN/null finals, 0 currently `benchmark_unavailable`-flagged
  (trimmed window). avg `vs_spy` +97.4% over 480 runs with same-window
  spread −39%→+40%+ (runs 6181–6185, 2009–2013) — leveraged-beta
  dispersion, not alpha, exactly as documented. Continuous loop is on
  stale code (predates this session's commits) — both shipped changes are
  inert until `run_continuous_backtests.py` restart.

### 2026-05-17 review pass #6 (ML+backtest hybrid · per-persona decision-signal skill · live findings)

- **Feature shipped: per-persona decision-signal-skill diagnostic.**
  `paper_trader/ml/persona_leaderboard.py` answers persona quality at the
  *run-return* level — but AGENTS.md is emphatic that the per-run number is
  leveraged-beta luck, not skill. There was **no decision-level**
  per-persona view: does a persona's own signal (`ml_score`) actually
  rank-predict the realized 5d outcome it acted on, or is its return pure
  beta noise? `paper_trader/ml/persona_skill.py` answers exactly that.
  For each persona (run_id→persona via the single-source-of-truth
  `backtest.persona_for`) it computes `score_ic` = tie-aware
  Spearman(action-aligned `ml_score`, action-aligned `forward_return_5d`),
  reusing `calibration._spearman` (single source of truth — cannot drift
  from the in-sample calibration metric; tie-awareness load-bearing because
  reasoning-parsed `ml_score` ties heavily at the persona buy threshold).
  The SELL convention is the codebase-universal target sign-flip applied
  **symmetrically** to the signal too, so "higher signal ⇒ higher realized
  goodness" is monotone across BUY/SELL. Verdicts per persona:
  `INSUFFICIENT` / `NO_SIGNAL_EDGE` / `WEAK_SIGNAL_EDGE` / `SIGNAL_EDGE` /
  `INVERTED_SIGNAL`; overall `INSUFFICIENT_DATA` / `NO_PERSONA_EDGE` /
  `HAS_INVERTED_PERSONA` / `HEALTHY`. Same discipline as
  `ml/calibration.py` / `ml/skill_trend.py` / `ml/persona_leaderboard.py`:
  read-only, no train / pickle / `build_features` / `N_FEATURES` / trade
  touch, never raises; it does **not** prune `PERSONAS` or re-tune
  `_PERSONA_BOOSTS` (a strategy-dynamics decision it only *informs*). CLI
  exits 2 if any persona is `INVERTED_SIGNAL` (operator/cron branchable,
  like `persona_leaderboard._cli`).
  ```bash
  cd /home/zeph/paper-trader && python3 -m paper_trader.ml.persona_skill
  cd /home/zeph/paper-trader && python3 -m pytest tests/test_persona_skill.py -v
  ```
  16 exact-value verdict/arithmetic locks in `tests/test_persona_skill.py`.

- **Quant finding (live, actionable): 4 of 8 personas have no decision-
  signal edge.** Run against the live `decision_outcomes.jsonl` (6782
  aligned outcomes): GARP `score_ic +0.24` and Value Investor `+0.15`
  carry real `SIGNAL_EDGE`; Sector Rotator (n=1470) `+0.11` and ESG
  (n=2158) `+0.08` are only `WEAK_SIGNAL_EDGE`; **Small/Mid Cap (n=1372)
  `-0.01`, Contrarian `-0.05`, Momentum `-0.06`, Global Macro `-0.09` are
  `NO_SIGNAL_EDGE`** — i.e. the two highest-*volume* personas have weak
  edge and four personas' returns are pure leveraged-beta dispersion, not
  signal skill (overall verdict `HEALTHY` only because ≥1 persona has edge
  and none is inverted). This is the decision-level confirmation of the
  repeatedly-documented "read `vs_spy_pct` skeptically" thesis. Reported,
  **not actioned** — pruning/re-tuning `_PERSONA_BOOSTS` is a
  strategy-dynamics decision out of surgical scope (CLAUDE.md §6).

- **Bug audit: bugs_fixed = 0, no Phase-1 commit.** `decision_scorer.py`,
  `backtest.py`, `run_continuous_backtests.py` re-audited (math:
  `_rsi`/`_macd`/`_ema` offsets, MACD signal alignment, BB/momentum
  windows; `train_scorer` dedup/sign-flip/oversampling; outcome
  parsing/regex; locking; atomic writes). No new safe surgical bug found
  after five prior passes — per the commit guard, none fabricated.

- **Quant finding (live, reported — not a surgical fix): `_llm_annotate_
  outcomes` has NEVER worked in production.** `continuous.log` shows
  `[continuous] LLM annotation failed: "Could not resolve authentication
  method…"` on **20/20** cycles, paired with `LLM labels: 0 endorsed, 0
  condemned`. Root cause: the function constructs `anthropic.Anthropic()`
  (needs `ANTHROPIC_API_KEY`, unset — the box authenticates the `claude`
  CLI via a user session, not an SDK key), while **every other LLM call in
  the codebase** (`_opus_annotate` 100 lines below, `backtest._claude_call`)
  uses `subprocess.run(["claude","--model",…,"--print","--permission-mode",
  "bypassPermissions"])` which works. Consequence: `llm_quality_label` is
  `0` on **all 6782** rows of `decision_outcomes.jsonl` — the documented
  3×-endorse / 0.1×-condemn `train_scorer` sample-weighting
  (AGENTS.md "Common pitfalls") has applied **zero** times in this
  dataset's history; the feature is dead. **Recommended fix (operator
  decision, deliberately NOT actioned here):** port `_llm_annotate_
  outcomes` to the proven `subprocess claude --print` transport like
  `_opus_annotate`. It is left as a finding because enabling a dormant
  3×/0.1× reweight on the live unattended scorer is a training-dynamics
  change — it would create a mixed-regime training set (6782 historic
  label-0 rows + newly-labeled rows) and warrants a deliberate decision +
  pickle reset, exactly the "report, don't action model dynamics in a
  surgical pass" discipline this file applies to the trainer-timeout and
  negative-OOS-skill findings.

- **Quant finding (reconfirmed live): scorer gates real conviction on a
  near-zero-edge signal.** `skill_trend` = `BORDERLINE` (recent median
  `oos_rmse` 10.96 vs fresh mean-predictor baseline 10.18; `oos_ic` ≈
  0.02, `oos_dir_acc` ≈ 0.51 — a coin flip), yet `gate_active=1.0` across
  all 7 ledger cycles (`n_train` 2972–3852 ≥ 500). In-sample calibration
  `DIRECTIONAL_BUT_BIASED` (spearman 0.38, monotone, decile error 3.0pp):
  the tails massively over-predict — d1 pred −18.3 vs realized −7.8, d10
  pred +14.9 vs +8.4 (~2× magnitude inflation, the documented
  extrapolation the `off_distribution` gate-abstention guards). Reported,
  not actioned (model-dynamics, out of surgical scope).

- **Live health.** backtest.db: 480 complete / 15 failed (all
  `[reaped: orphaned running row]` — per-cycle reaper works) / 10 running.
  0 NaN/null finals, 0 `benchmark_unavailable`-flagged. 7 scorer-skill
  ledger cycles all `status=ok`. Runs 6166–6170 stuck `running` ~4h
  (orphaned by a loop restart; **within** the 6h reap guard, will be
  reaped — not a new bug). External-only noise in `continuous.log`: GDELT
  `ConnectTimeout`/`ConnectionReset` (handled w/ backoff), SEC EDGAR HTTP
  500s, `GOOGU` yfinance 404 (`prices[t]={}`) — all graceful. Both shipped
  changes are inert until `run_continuous_backtests.py` restart (the
  documented restart-required pattern).

### 2026-05-18 review pass #7 (ML+backtest hybrid · conviction-gate effectiveness audit · live findings)

- **Phase 1 — no new bugs.** Full re-trace of `decision_scorer.py`,
  `backtest.py`, `run_continuous_backtests.py`, plus the coupled
  `validation.py` / `calibration.py` / `skill_trend.py`: regex `ml_score`
  parse (no `scorer=` false-match), `(ticker,sim_date,action)` dedup key,
  the universal SELL `-forward_return_5d` sign-flip, the 5-trading-day
  forward window guard, the off-distribution gate abstention, the WLS
  numpy-fallback math, and every module-global lock were all re-verified
  correct and exact-value test-locked. Consistent with the documented 9+
  prior no-new-bug passes. **bugs_fixed = 0; no Phase-1 commit** (commit
  guard honoured).

- **Feature shipped: conviction-gate effectiveness audit.**
  `paper_trader/ml/gate_audit.py`. The gap it fills: `calibration.py`
  answers a *statistical* question (is pred monotone with realized, bucketed
  by 10 quantile deciles) and `skill_trend.py` answers an *error-trend*
  question (oos_rmse vs a mean predictor) — **neither answers the economic
  one a quant asks before risking capital: do the five FIXED conviction
  multipliers `_ml_decide` applies (×0.60 / ×0.85 / ×1.00 / ×1.15 / ×1.30
  at FIXED prediction thresholds) actually buy realized edge?** A
  `WELL_CALIBRATED` decile curve can coexist with a gate whose ×1.30 arm
  realizes no more than its ×0.60 arm. `gate_audit` buckets every
  `decision_outcomes.jsonl` row by the exact `_ml_decide` gate arm the
  deployed scorer's prediction triggers (the if/elif chain reproduced
  byte-for-byte, boundary operators included — duplicated as `GATE_ARMS`
  module constants exactly as `calibration`/`skill_trend` avoid the
  `backtest.py` circular import), applies the codebase-universal SELL
  sign-flip, restricts to the **temporal-OOS slice** by default
  (`validation.split_outcomes_temporal` — the trustworthy view), and
  verdicts on the realized spread the 1.30/0.60 ratio is implicitly
  underwriting: `INSUFFICIENT_DATA` / `GATE_HARMFUL` (spread < −1pp — gate
  sizes UP the losers) / `GATE_INEFFECTIVE` (|spread| ≤ 1pp) /
  `GATE_EFFECTIVE` (spread > +1pp). Same discipline as `ml/calibration.py`:
  read-only, no train / pickle / `build_features` / `N_FEATURES` / trade
  touch, never raises — safe against the live unattended loop.
  ```bash
  cd /home/zeph/paper-trader && python3 -m paper_trader.ml.gate_audit
  cd /home/zeph/paper-trader && python3 -m pytest tests/test_gate_audit.py -v
  ```
  25 exact-value verdict/boundary locks in `tests/test_gate_audit.py`
  (gate-arm boundary operators mirror `_ml_decide`; SELL-flip regression
  lock; OOS-slice restriction; non-finite/missing-field hardening).

- **Quant finding: the live conviction gate is economically inert (and
  partially inverted).** `gate_audit` on the live pkl (`n_train=3446`,
  gate active) over the temporal-OOS slice (n=1000):

  | arm | mult | n | mean realized 5d |
  |-----|------|---|------------------|
  | strong_headwind | ×0.60 | 59 | **+2.09%** |
  | mild_headwind | ×0.85 | 483 | −0.12% |
  | neutral | ×1.00 | 304 | +0.79% |
  | mild_tailwind | ×1.15 | 109 | +1.88% |
  | strong_tailwind | ×1.30 | 45 | +3.07% |

  Verdict `GATE_INEFFECTIVE`: strong_tailwind − strong_headwind = **+0.98pp**
  (inside the ±1pp band) — a >2× capital swing buys ≈1pp of edge, noise
  against σ≈7–17 on 5d returns. Worse, the **tailwind half is monotone
  (0.79 → 1.88 → 3.07) but the headwind half is inverted**: the gate's
  *smallest* bet (strong_headwind ×0.60) realized the *second-highest*
  return (+2.09%), above neutral and mild_headwind. The ×0.60 down-sizing
  arm fires on the over-predicted d1 tail (calibration: d1 pred −9.63 vs
  realized −3.74 in-sample) and is mis-sizing names that don't deserve it.
  This is the missing economic complement to the existing split:
  `calibration` = `WELL_CALIBRATED` (in-sample, optimistic),
  `skill_trend` = `NEGATIVE_OOS_SKILL`, `gate_audit` = `GATE_INEFFECTIVE`.
  Reported, not actioned — re-sizing the multipliers or gate thresholds is
  a model-dynamics change out of surgical scope (CLAUDE.md §6; the gate is
  invariant #5).

- **Quant finding: the winner→ArticleNet feedback loop (CLAUDE.md §5
  step 5) is dead, now two ways.** Recent `continuous.log` `[continuous]
  ml:` lines are uniformly `trainer timeout (injected N)` **or**
  `inject err: database is locked`. AGENTS.md already documented the 120 s
  `ml.trainer.train(force=True)` timeout; the **`database is locked`** on
  the `_inject_and_train` write is a second, distinct failure (the live
  digital-intern daemon and the injector contend on `articles.db` — the
  injector opens a plain `sqlite3.connect(DB_PATH, timeout=15)` with no WAL
  pragma, unlike the read paths). Net: injection partially lands or is lost,
  ArticleNet never retrains from winners. Root cause is digital-intern-side
  (GPU contention + write contention) — reported, out of this domain's
  surgical scope, but the loop should not be read as "training on its
  winners" — it is not.

- **Quant finding: `[price_cache] XLI failed: 'Response' object has no
  attribute 'get'`** — an intermittent yfinance internal error during the
  per-window price-cache build leaves `prices["XLI"] = {}` (handled by the
  `except` — **not a code bug**). Consequence: `_sector_rotation` silently
  drops XLI from that cycle's rotation ranking and the XLI quant features
  no-op. Transient/network, self-heals on next redraw; noted as a
  data-realism caveat, not actioned.

- **Operational finding: hourly review agents are stacking.** Three
  identical-prompt ML+backtest review processes were observed running
  concurrently (started 06:00 / 06:30 + this pass), each taking >1 h, so
  `scripts/hourly_review.sh` overlaps itself. The working tree already
  carried another agent's uncommitted `run_curves` IN-clause chunking edit
  to `backtest.py`; to avoid sweeping it into this pass's commit, this pass
  touched **only new files** (`paper_trader/ml/gate_audit.py`,
  `tests/test_gate_audit.py`) plus this AGENTS.md section. Consider a
  lockfile / `flock` in `hourly_review.sh` so a still-running review skips
  rather than stacks.

### 2026-05-18 review pass #8 (ML+backtest hybrid · permutation feature-importance · live findings)

- **Phase 1 — no new bugs.** Full re-trace of `decision_scorer.py`,
  `backtest.py`, `run_continuous_backtests.py` plus the coupled
  `validation.py` / `calibration.py`: the `score=` vs `scorer=` regex
  disambiguation (first-match is `score=N`, `scorer=` has no `score=`
  substring — re-verified), the universal SELL `-forward_return_5d`
  sign-flip symmetry train↔inference, the 11-column `_inject_and_train`
  INSERT alignment, `_to_float`'s `np.number` (not `np.generic`) branch,
  the `score`/`tickers` null-default hardening class, the
  `_train_decision_scorer` separately-guarded OOS blocks, and
  `_parse_scorer_status`'s `(?:^|\s)key=` token regex were all re-verified
  correct and exact-value test-locked. Consistent with the documented 10+
  prior no-new-bug passes. **bugs_fixed = 0; no Phase-1 commit** (commit
  guard honoured — a clean 299/0 ML/backtest baseline, not a fabricated
  fix).

- **Feature shipped (commit `40715a7`): permutation feature-importance
  diagnostic.** `paper_trader/ml/feature_importance.py`. The gap it fills:
  `calibration` answers *is pred monotone with realized* (statistical),
  `skill_trend` answers *is oos_rmse better than a mean predictor*
  (error-trend), `gate_audit` answers *do the 5 fixed multipliers buy
  realized edge* (economic) — **none answers WHICH of the 17 features
  carries (or fails to carry) the prediction.** That is the natural quant
  question once the gate is known to be `GATE_INEFFECTIVE` /
  `NEGATIVE_OOS_SKILL`: is the model blind, sector-memorizing, or reading
  real signal that just doesn't generalize? `feature_importance` permutes
  each logical feature across the temporal-OOS slice (the 7-way sector
  one-hot permuted **jointly** via the `ticker` field so
  `build_features.SECTOR_MAP` stays the single source of truth — permuting
  one one-hot slot would fabricate importance), re-predicts through the
  **same** `scorer.predict` path `_ml_decide`'s gate uses, and reports
  `rmse_increase` / `rank_ic_drop` / `dir_acc_drop` per feature (the gate
  acts on sign *and* magnitude, so all three). Verdicts: `UNTRAINED` /
  `INSUFFICIENT_DATA` / `FLAT` / `SECTOR_DOMINATED` / `SECTOR_LEANING` /
  `SIGNAL_GROUNDED`. **Honesty guard:** a column with < 2 distinct non-null
  values on the slice is flagged `degenerate` (nothing to permute) and can
  never be `material`, so a *sparsity* 0.0 is never misread as "the model
  ignores this feature". Same discipline as `ml/calibration.py`: read-only,
  no train / pickle / `build_features` / `N_FEATURES` / trade touch, never
  raises, reuses `calibration._spearman` + `validation.split_outcomes_
  temporal`. CLI exits 2 on `SECTOR_DOMINATED` / `FLAT` (operator/cron
  branchable, like `label_audit` / `persona_skill`). It is a CLI / `ml/`
  reader, **not** wired into `main()` — zero deploy-stale impact, no loop
  restart needed (unlike a wiring change). 13 exact-value locks in
  `tests/test_feature_importance.py` (ignored-feature == exactly 0.0 on all
  three metrics; SECTOR_DOMINATED via a sector-only fake; FLAT via a
  constant predictor; SELL sign-flip regression; OOS-slice restriction;
  degenerate flag; never-raises on raising/NaN/untrained/too-few).
  ```bash
  cd /home/zeph/paper-trader && python3 -m paper_trader.ml.feature_importance
  cd /home/zeph/paper-trader && python3 -m pytest tests/test_feature_importance.py -v
  ```

- **Quant finding (NEW, the headline): the scorer's near-zero OOS skill is
  NOT sector-memorization or label contamination — it leans hardest on
  genuine quant mean-reversion features that just don't generalize.** Live
  pkl `n_train=3446`, gate active, OOS slice n=1000. Verdict
  `SIGNAL_GROUNDED`, ranked by `rmse_increase`: **rsi +3.80, bb_position
  +3.24, mom5 +2.08, mom20 +1.83**, ml_score +0.70, **sector only #6 at
  +0.58**, macd +0.49, vol_ratio +0.14, regime_mult +0.08. So the model is
  reading classic RSI/Bollinger/short-momentum mean-reversion *hard*, and
  the sector one-hot — the suspected DFEN/FAS extreme-label memorization
  vector — is near the bottom. Yet `gate_audit`=GATE_INEFFECTIVE,
  `skill_trend`=NEGATIVE_OOS_SKILL (oos_rmse 10.96 vs fresh mean-predictor
  baseline 6.90, oos_ic 0.02, dir_acc 0.51), `calibration`=WELL_CALIBRATED
  *in-sample only* (tails inflate ~2.6×: d1 pred −9.63 vs realized −3.74).
  **The diagnosis this narrows to: the scorer reads real quant signal that
  carries no out-of-sample edge in this leveraged-ETF-heavy universe —
  "signal with no OOS edge", not "blind / sector-memorizing".** A more
  actionable framing for a future model-dynamics decision than the prior
  passes could establish. Reported, not actioned (model-dynamics / CLAUDE.md
  §6).

- **Quant finding (NEW): 2 of the 17 scorer inputs are structurally dead in
  the gate-relevant slice.** `news_urgency` / `news_article_count` are null
  for **100% of the OOS slice** (122/5000 non-null in the full corpus;
  `news_urgency` has only **1 distinct value** corpus-wide). They feed
  `build_features` slots 8–9 as the constant defaults (50.0 / 1.0) on every
  live gate decision. Root cause is the already-documented dead
  `_llm_annotate_outcomes` + the sparsity of parsed `news_count`/`news_urg`
  reasoning tokens (`_compute_decision_outcomes` nulls them when
  `news_count<=0`, which is almost always). The new tool surfaces this
  honestly via `degenerate` rather than letting a reader conclude "the model
  ignores news". Quantified here for the first time; not actioned (removing
  features is an `N_FEATURES`/pickle-breaking model-dynamics change —
  CLAUDE.md §6 / "When to bump model versions").

- **Live health.** `backtest.db`: 480 complete / 15 failed (all
  `[reaped: orphaned running row]` — startup reap logged `reaped 15`) / 10
  running. 0 NaN/null finals, 0 `benchmark_unavailable`-flagged (trimmed
  window). vs_spy over 480 complete: median **+37.6%**, min −170%, max
  +2820% — textbook leveraged-beta dispersion, not alpha, exactly as
  documented. 9 scorer-skill ledger cycles all `status=ok`, `gate_active=1.0`.
  Loop is on recent code (per-cycle reaper + `oos_diracc`/`oos_ic` present,
  no stale-code pattern) — both this and prior shipped diagnostics are
  inert-by-design `ml/` readers, **no restart needed**. Runs 6166–6170
  `running` 6.6 h: just crossed the 6 h reap guard, the next per-cycle
  mid-loop reap sweeps them — expected self-healing, **not a bug**.

- **Operational finding (reconfirmed, out of scope): winner→ArticleNet
  feedback loop dead two ways every cycle.** `[continuous] ml:` lines are
  uniformly `trainer timeout (injected N)` or `inject err: database is
  locked`; a separate `engine init failed … sqlite3.OperationalError:
  locking protocol` traceback appears intermittently (gracefully handled —
  `main()` logs it, `sleep 30`, `continue`). All three share one root
  cause: `backtest.db` is a symlink onto a removable/networked volume
  (`/media/zeph/projects/...`) whose SQLite WAL locking is contended by the
  live digital-intern daemon + the injector. ArticleNet never retrains from
  winners. Digital-intern-side + infra; reported, not actioned (matches
  pass #6/#7 findings — the loop should not be read as "training on its
  winners"; it is not).

### When to bump model versions

The scorer model has no explicit version field. Treat a change to
`N_FEATURES`, `SECTORS`, or `build_features` parameter signature as a
breaking change: delete `data/ml/decision_scorer.pkl` and let the next
continuous cycle retrain from `data/decision_outcomes.jsonl`. The pickle
auto-recreates atomically (`.pkl.tmp` → `replace`) so a fresh-start
deletion is safe even if a backtest thread is mid-read.

### 2026-05-18 review pass #9 (paper-trader core hybrid · /api/drawdown invariant-#12 + drawdown CLI · live findings)

- **Phase 1 — 2 bugs fixed (commit `d5d00fe`).**
  1. **`drawdown_api()` did not thread `INITIAL_CASH`.** It called
     `compute_drawdown(eq, positions)` with no `starting_equity`, silently
     relying on the builder's hardcoded `1000.0` default — the exact
     invariant-#12 violation `benchmark_api`/`analytics_api`/
     `reporter._INITIAL_EQUITY` are explicitly written to avoid (the
     `analytics_api` "a literal here silently desyncs Calmar if
     INITIAL_CASH" comment). On a fresh/empty equity curve `/api/drawdown`
     reported peak/trough/current at a literal 1000 and always echoed a
     wrong `starting_equity` if `INITIAL_CASH` ever moved. Fixed:
     `compute_drawdown(eq, positions, starting_equity=INITIAL_CASH)`.
  2. **`compute_drawdown` empty-curve fallback omitted `starting_equity`
     + `trough_pct`** that the populated branch returns — an inconsistent
     response shape that hands the dashboard card / decision-context fold
     `undefined` on a day-one book. Surfaced *by writing the real-logic
     test*, not by inspection. Both keys added for shape parity.
  - New `tests/test_drawdown.py` (`drawdown.py` previously had **no**
    test file): hand-computed peak/trough/recovery math,
    trough-resets-on-new-peak, at-high-water 1bp boundary, contributor
    sort + zero-cost-basis guard, history tail-pin, and the endpoint
    regression lock (FAILS against pre-fix code). 12 tests; 327-test core
    suite green.

- **Phase 2 — feature shipped (commit `dd9af44`): `python -m
  paper_trader.analytics.drawdown [--json]`.** Drawdown — depth, time
  underwater, what's dragging, how much clawed back — is a top-of-mind
  live-trader risk question with no terminal access, while every peer risk
  module (`benchmark`, `desk_pulse`, `model_reliability`,
  `decision_context`, `signals --check-freshness`) ships a CLI for exactly
  the case that is **live right now**: `/api/build-info` `stale:true
  behind:11`, so `/api/drawdown` serves *pre-fix* code until a runner
  restart. Thin read-only `__main__` (the `benchmark.py` precedent
  verbatim — `?mode=ro`, `INITIAL_CASH` threaded so the CLI honours
  invariant #12 too, `--json` | one-screen human digest with badge +
  peak/trough/recovery + top draggers). Verified live: `IN DRAWDOWN
  −3.46% / −$34.90, 84.8h in DD, LITE drag`. 2 subprocess end-to-end
  locks (skipped where no live DB).
  - **Concurrency note:** `dd9af44` also contains 2 `digital-intern`
    files a *sibling agent staged into the shared git index* between this
    agent's `git add` and `git commit` — `git commit` commits the whole
    index, not just what you `git add`. The per-commit "stage only your
    files" guard is **insufficient under a concurrent writer**; use
    pathspec-limited `git commit -- <files>` (race-immune). No code lost;
    the sibling's work is valid and intact, just bundled under this
    message.

- **Phase 3 — live findings (trader perspective, 2026-05-18 ~02:40 UTC).**
  1. **Running :8090 is 11 commits stale** (`build-info` `behind:11
     stale:true`). Concrete cost: `/api/model-reliability` 404s and every
     fix/endpoint committed today (incl. the drawdown invariant-#12 fix)
     is inert until **the runner is restarted**. The dominant operator
     action item.
  2. **Capital paralysis live & severe:** cash $18.49 (1.9% of $972.69),
     ~25 cycles since last fill, deployed ~98%, top-1 concentration 60.8%
     (LITE) `severity HIGH`. ~17h of `HOLD LITE`; Opus cannot act on a new
     signal without first selling. (Already surfaced by
     `capital_paralysis`/`funded_suggestions`.)
  3. **Decision-reliability degraded:** `/api/decision-reliability`
     `current_failure_rate_pct 24.7%` (23 current `TIMEOUT_EMPTY`
     failures) — ~1 in 4 live cycles produces no decision (Opus wedged/
     slow). Matches the `NO_DECISION (timeout/empty)` rows in the
     decisions table.
  4. **Persisted-vs-live mark discrepancy:** the stored `positions` row
     for MU is `current_price==avg_cost==724.12, P/L $0.00` (a stale mark
     persisted from the last cycle MU was unfetchable), while a fresh
     read-only recompute (`/api/mark-integrity`) reports "All 2 marks
     live, n_stale 0". The `stale_mark` flag is **not persisted between
     cycles**, so the Discord hourly summary (reads persisted
     `open_positions()`) shows MU as a misleading flat $0.00 until the
     next decision cycle re-marks. Behavioural, not a core-code bug — left
     as a finding (persisting the flag would change the live mark path).
  5. **`logs/runner.log` is ~7h stale** (mtime 05-17 19:40) while the
     trader is demonstrably live (decision 1.5 min ago, equity point 1
     min ago). An operator tailing the documented health log sees a frozen
     file — log-based monitoring is blind to current activity/errors;
     there is no fresh tailable runner stdout at the documented path.
  - Decision loop itself **healthy and on-cadence** (fresh decision +
    equity point); dashboard endpoints return sensible non-stale JSON
    (`/api/risk` top1 60.8%, `/api/benchmark` `alpha_pp −2.25`,
    `/api/scorecard`/`/api/desk-pulse` 200).

- **Run the core suite:** `cd /home/zeph/trading-intelligence/paper-trader
  && python3 -m pytest tests/test_core_*.py tests/test_drawdown.py -q`
  (the 6 core files + the new drawdown lock = 303 fast offline tests; the
  full `tests/ -v` sweep is correct but slow under a concurrent pytest —
  the core subset is the meaningful core-domain proof).

### 2026-05-18 review pass #10 (paper-trader core hybrid · data-feed resolver fix · quota-exhaustion guard + robust openclaw · live findings)

- **Phase 1 — 1 bug fixed (commit `203bca4`).**
  **`/api/data-feed` bypassed the freshness-aware DB resolver and pinned a
  pre-migration path.** `data_feed_api()` resolved digital-intern's
  `articles.db` via its own hardcoded candidate list
  (`/home/zeph/digital-intern/data/articles.db` LOCAL first, USB fallback)
  instead of `_articles_db_path()` → `signals._db_path()`. Two real
  defects: (a) **invariant #17 violation** — every other news-analytics
  endpoint routes through the freshness-aware single source of truth so the
  dashboard and the live trader never disagree on which feed is canonical;
  this one didn't, so the live news-pulse panel could read a stale USB
  mirror while the trader read fresh LOCAL (the exact split-brain #17
  closed everywhere else); (b) the "LOCAL" literal is the **pre-migration**
  path — the repo now lives under `/home/zeph/trading-intelligence/`; it
  only resolves on the original box via a legacy migration symlink, so on a
  clean checkout the endpoint silently zeroes the panel with
  `error: articles.db not found`. Fix is surgical: `db_path =
  _articles_db_path()`; the None-graceful shape + live-only SQL filter are
  unchanged. New `tests/test_core_dashboard_data_feed.py` (5 tests) drives
  the real Flask view: the discriminating stale-USB-loses-to-fresh-LOCAL
  assertion (FAILS pre-fix — old code read the box's real 1.4 GB prod DB,
  not the test tmp DBs), fresher-USB-still-wins, backtest/opus row
  exclusion, graceful-zero-when-no-DB, independent 1h/24h window boundary.

- **Phase 2 — 2 features shipped (commit pending): quota-exhaustion guard
  + robust openclaw resolution.** Both motivated by Phase-3 live findings,
  not invented.
  1. **Quota-exhaustion alarm.** `strategy._is_quota_exhausted(text)` (tight
     marker set — `usage limit`/`quota exceeded`/`quota exhausted`/`out of
     credit`/`insufficient credit`, case-insensitive, no false alarms on a
     timeout/parse-miss) flags the observed live failure (`claude` rc=1,
     stdout `You've hit your org's monthly usage limit`). `_claude_call`
     sets a per-cycle module flag; `decide()` resets it each cycle and
     surfaces `summary["quota_exhausted"]` (+ a quota-specific
     `decisions.reasoning` instead of the generic `parse_failed`).
     `runner._cycle` fires **one** `reporter.send_quota_alert()` per outage
     (dedupe latch `_quota_alert_active`), **skips the futile
     circuit-breaker pkill** (the CLI already exited — nothing to kill — and
     holds the breaker counter at 0 so a quota outage can never trip it),
     and on recovery (a real non-NO_DECISION) sends a `RECOVERED` notice and
     re-arms. A non-quota timeout after an outage holds the alarmed state
     (not premature "recovered") and the ordinary breaker still counts it.
  2. **Robust openclaw resolution.** `reporter._resolve_openclaw()`:
     `OPENCLAW_BIN` env override → `PATH` (`shutil.which`) → well-known
     fallbacks (`~/.local/bin`, `/usr/local/bin`, `/usr/bin`, glob
     `~/.nvm/versions/node/*/bin/openclaw`, newest node first). Closes the
     live failure where the systemd unit's minimal PATH excluded the nvm
     bin so `shutil.which` returned `None` and **every** Discord message
     (incl. the new quota alert) was silently dropped. Verified live:
     resolves `/home/zeph/.nvm/versions/node/v24.15.0/bin/openclaw` with an
     empty PATH. New `tests/test_quota_guard.py` (29 tests) locks the full
     chain (marker precision, `_claude_call` flag wiring, `decide()`
     surface + per-cycle reset, resolver 4-way order, alert body,
     `_cycle` dedupe/recovery/re-arm/breaker-skip); one existing
     `test_core_reporter.py::test_returns_false_when_openclaw_missing`
     **adapted** (not weakened) to the new resolver seam — same assertions
     (False + logged would-send), now exercising all three resolver steps
     returning None.

- **Phase 3 — live findings (trader perspective, 2026-05-18 ~04:40 UTC).**
  1. **TWO live `runner.py` processes on the same $1000 book.** PID
     1255030 (started 11:30, running **pre-singleton-lock** in-memory code
     — no lock line in its boot log) and PID 1465599 (17:28, holds
     `data/paper_trader.runner.lock`). Both cycle `paper_trader.db` →
     double NO_DECISION rows 4 s apart (03:57:13 + 03:57:17, 02:39 + 02:42,
     01:39 + 01:41). The invariant #19 guard works for the code it's *in*;
     it cannot retroactively stop a process that never took the lock.
     **Operator action:** `kill 1255030` (keep the lock holder 1465599),
     then restart that one to pick up today's fixes. Not a code bug — a
     code "fix" that hunts sibling `runner.py` PIDs is exactly the
     host-wide-scan footgun the `_kill_stale_claude` comment forbids.
  2. **Claude quota exhausted** (`You've hit your org's monthly usage
     limit`) — the live trigger for the Phase-2 guard. Trader frozen on
     NO_DECISION/flat-HOLD for hours; `/api/decision-reliability`
     `current_failure_rate_pct 27.1%`, 100% `TIMEOUT_EMPTY`, with a
     **fresh feed** (`/api/feed-health` HEALTHY, newest live article 0.1h)
     — so it is the quota, not a feed outage. Operator action: resolve /
     upgrade the Anthropic quota; a restart will not help.
  3. **Every Discord report silently dropped** — `[reporter] openclaw not
     installed; would send:` on every hourly/daily/trade send (the live
     trigger for the Phase-2 openclaw fix). `openclaw` is installed at the
     nvm path but the runner's PATH excludes it.
  4. **Running :8090 is `behind:24 stale:true`** (`build-info`
     `boot_sha 310d16e`). All of today's fixes (data-feed, quota guard,
     openclaw) are inert until the (deduplicated) runner is restarted.
  - Decision loop **healthy & on-cadence** otherwise (heartbeat HEALTHY,
    last decision 46 min ago within the 60 min closed cadence);
    `/api/risk` correctly flags **HIGH** concentration (LITE 60.9% top-1,
    top-3 98.1%); `/api/portfolio` $972.69 / $18.49 cash;
    `/api/benchmark` `−2.25pp` vs SPY — all sensible, non-stale.

- **Run the core suite:** `cd /home/zeph/trading-intelligence/paper-trader
  && python3 -m pytest tests/test_core_*.py tests/test_quota_guard.py -q`
  (full `tests/` is 1361 tests, green, but slow under a concurrent pytest;
  the core subset + the new quota lock is the meaningful core-domain proof).

### 2026-05-18 review pass #11 (paper-trader core hybrid · degraded-runner self-recheck · degraded-runner self-reporting · live findings)

- **Phase 1 — 1 bug fixed (commit `7aa4d85`). The two-runner double-trade
  window is now closed *in code*, the right way.** Review pass #10 observed
  the live two-runner pathology (PID 1255030 degraded + PID 1465599 locked,
  both cycling `paper_trader.db`) and concluded "**Not a code bug** — a code
  fix that hunts sibling `runner.py` PIDs is exactly the host-wide-scan
  footgun the `_kill_stale_claude` comment forbids." That conclusion only
  ruled out *one* approach (PID hunting). The actual root cause is that
  `_acquire_singleton_lock` fails **open** at boot (invariant #19) when the
  USB-backed `data/` dir is transiently unmounted — and a degraded runner
  then runs guard-less *forever*, so a later runner cleanly takes the flock
  and both double-trade. Confirmed live again 2026-05-18: PID 1255030 has
  **no `runner.lock` fd at all** (`/proc/1255030/fd`), PID 1465599 holds
  `FLOCK …265831` (`/proc/locks`); `/api/decision-reliability`
  `current_failure_rate_pct 27.6%`, **100% `TIMEOUT_EMPTY`**,
  `involuntary_alpha_bleed_pct −2.21%` — the concrete trader cost of the two
  runners racing the API (each `_claude_call` / `_kill_stale_claude -P` reaps
  the *other's* in-flight claude). **Fix:** new
  `runner._recheck_singleton_lock()` called at the top of every loop
  iteration. It re-attempts the lock **only from the `degraded` state** and:
  upgrades in place (`acquired` — keeps the handle) if the lock is now free;
  `sys.exit(1)` if the result is `busy` (another live trader **confirmed**
  holding it — the redundant degraded runner stands down so the locked
  instance is sole writer); keeps running if still `degraded` (plumbing still
  unusable — **invariant #19 fully preserved: it exits ONLY on a confirmed
  other holder, NEVER on plumbing failure**). It is a hard **no-op once we
  hold the lock** — a 2nd `open()`+`flock` on the same file from the same
  process gets a distinct open-file description and is denied by our *own*
  lock, which would mis-read as `busy` and exit the real holder (the
  load-bearing guard; test `test_noop_when_already_acquired`). This is **not**
  PID hunting and **not** a host-wide scan: the runner cooperatively
  introspects *its own* lock and *itself* stands down — no signal is ever
  sent to another process. Do not revert this citing pass #10's "not a code
  bug" — that judgement predated the self-recheck design (advisor-validated).
  Locked by `tests/test_core_runner.py::TestRecheckSingletonLock` (noop-when-
  acquired · still-degraded-no-exit (#19) · upgrade-when-free · exit-on-
  confirmed-duplicate · `singleton_lock_state` accessor).

- **Phase 2 — 1 feature shipped (commit pending): the degraded runner is no
  longer invisible.** Motivated directly by the Phase-3/-pass-#10 finding
  that a guard-less runner was undetectable from every operator surface
  (`/api/runner-heartbeat` HEALTHY, dashboard fine, Discord fine — yet the
  book was being double-traded). `runner.singleton_lock_state()` is a pure
  module-global snapshot (`{status, holder_pid, have_lock, degraded}`),
  surfaced two ways: (1) **`/api/runner-heartbeat`** gains an additive
  `singleton_lock` block (the *process serving the dashboard* reports its
  own lock state — the dashboard runs in a runner thread; the pure
  `build_runner_heartbeat` is untouched, the process read is owned by the
  endpoint per the thesis_drift split; the existing liveness verdict is
  unchanged, a different test-locked concern); (2) **the hourly / daily-close
  Discord summary** gains a loud `⚠️ RUNNER DEGRADED` one-liner via
  `reporter._singleton_lock_line()` (the operator lives in Discord; the
  `runner` import is lazy — `runner` imports `reporter` at module load, so a
  top-level import would be circular). Same additive failure contract as
  every other reporter block: a fault drops just this line, never the
  summary; emits **nothing** when the lock is held (no noise). Observational
  only — never gates, no caps (invariants #2/#12). Locked by
  `tests/test_runner_heartbeat.py` (degraded + acquired endpoint shapes) and
  `tests/test_core_reporter.py::TestSingletonLockLine` (empty-when-acquired ·
  warns-when-degraded · fault-degrades-to-empty · hourly includes/excludes).

- **Phase 3 — live findings (trader perspective, 2026-05-18 ~05:30 UTC).**
  1. **Two-runner double-trade confirmed and root-caused** (see Phase 1):
     PID 1255030 degraded (no lock fd), PID 1465599 holds the flock, both
     live. **Now self-healing** once the deduplicated runner restarts onto
     this pass — the degraded one will exit on its next cycle. Operator
     action remains: restart the lock holder to also clear `build-info stale`.
  2. **Decision engine fails ~28% of *current-regime* cycles** (`/api/
     decision-reliability` 27.6% `TIMEOUT_EMPTY`, ~50 dead cycles/day, the
     58.8% all-time headline inflated by 410 legacy rows), costing **−2.21%
     alpha** of the −2.25pp SPY gap. This *is* the two-runner contention;
     the Phase-1 fix is the remedy (not quota — `/api/feed-health` HEALTHY,
     news 0.2h fresh; the book is correctly flat-HOLDing the weekend with
     $18.49 cash, the NO_DECISION rows interleaved are the contention).
  3. **`/api/risk` HIGH concentration is correct, not a bug** — LITE 60.9%
     top-1, top-3 98.1%, cash 1.9% ($18.49). Surfaced, never enforced
     (invariants #2/#12 working as intended).
  4. **Running :8090 is `behind:28 stale:true`** (`build-info`
     `boot_sha 310d16e`). This pass's fixes (and pass #10's) are inert until
     the runner is restarted; the new `singleton_lock` heartbeat block will
     not appear on the live endpoint until then (verified green via the
     Flask test client instead).
  - `openclaw` resolves via the nvm fallback (`/home/zeph/.nvm/versions/
    node/v24.15.0/bin/openclaw`) — pass #10's robust resolver works; Discord
    reporting is live. `/api/feed-health` HEALTHY, `/api/portfolio`
    $972.69 / $18.49, `/api/benchmark` −2.25pp — all sensible, non-stale.

- **Run the core suite:** `cd /home/zeph/trading-intelligence/paper-trader
  && python3 -m pytest tests/test_core_runner.py tests/test_core_reporter.py
  tests/test_runner_heartbeat.py -q` (the files this pass touched; full
  `tests/` is green but slow under the concurrent review pytest).

### 2026-05-18 review pass #11 (ML+backtest hybrid · regime-conditional scorer-skill audit · live findings)

- **Phase 1 — no new bugs (bugs_fixed = 0; no Phase-1 commit).** Full
  re-trace of `decision_scorer.py`, `backtest.py`,
  `run_continuous_backtests.py` plus coupled `validation.py` /
  `calibration.py`: `score=`/`scorer=` regex first-match disambiguation,
  the `(ticker,sim_date,action)` dedup key, the universal SELL
  `-forward_return_5d` sign-flip (train↔inference↔calibration↔gate), the
  5-trading-day forward-window guard, the off-distribution gate abstention,
  the 11-column `_inject_and_train` INSERT alignment, the separately-guarded
  `_train_decision_scorer` OOS blocks, every module-global lock — all
  re-verified correct and exact-value test-locked. Two candidates turned
  over and correctly judged not-worth-shipping: (a) temporal-boundary
  duplicate leakage in `split_outcomes_temporal` is bounded to ~one
  sim_date's rows (~2% of the OOS slice) and would only make the
  already-documented negative OOS skill look *slightly worse* while
  breaking `test_continuous.py` literals; (b) `scorer_calibration`'s `-y`
  on a non-numeric `forward_return_5d` is a hypothetical gap with no
  observed instance (the pipeline writes `round(float, 4)` only).
  Consistent with the 11+ prior no-new-bug ML/backtest passes — not a
  fabricated fix. ML/backtest subset 269/269 green before the feature.

- **Feature shipped (commit `816fd72`): regime-conditional scorer-skill
  audit.** `paper_trader/ml/regime_audit.py`. Gap filled: `calibration`
  (statistical deciles), `gate_audit` (economic gate arms), `skill_trend`
  (error-trend cycles), `feature_importance` (attribution) — **none
  conditions on market regime.** A scorer with ≈0 OOS rank skill *on
  average* could still be skilled in one regime and inverted in another, in
  which case the aggregate "no edge" verdict is a regime-mix artifact.
  `regime_audit` decodes regime from the `regime_mult` feature every
  `decision_outcomes.jsonl` row carries (`0.3→bear`, `0.6→sideways`,
  `1.0→bull_or_unknown` — the `1.0` label is deliberately honest:
  `_market_regime` collapses true-bull and "unknown" to the same `1.0`),
  restricts to the **temporal-OOS slice** by default
  (`validation.split_outcomes_temporal` — the EXACT split
  `_train_decision_scorer` uses, so this and the ledger's scalar OOS
  metrics describe the *same* holdout), and per regime reports `rank_ic`
  (via `calibration._spearman` — single source of truth, tie-aware vs the
  ±50 clamp), `dir_acc`, and the `gate_audit` extreme-arm spread
  *conditioned on regime*. Verdicts: `INSUFFICIENT_DATA` /
  `SINGLE_REGIME_ONLY` (OOS slice regime-degenerate — honest limitation) /
  `REGIME_UNIFORM_NULL` / `REGIME_DEPENDENT_EDGE` (actionable — aggregate
  hides regime structure) / `REGIME_UNIFORM_EDGE`. A regime needs
  `≥ MIN_REGIME_N = 20` pairs before its skill counts (thinner buckets
  reported but flagged `thin`, never misread as a discovered edge).
  Read-only, no train/pickle/`build_features`/`N_FEATURES`/trade touch,
  never raises, CLI exits 2 on `REGIME_DEPENDENT_EDGE`. NOT wired into
  `main()` — zero deploy-stale impact. 22 exact-value locks in
  `tests/test_regime_audit.py`.
  ```bash
  cd /home/zeph/paper-trader && python3 -m paper_trader.ml.regime_audit
  cd /home/zeph/paper-trader && python3 -m paper_trader.ml.regime_audit --all
  cd /home/zeph/paper-trader && python3 -m pytest tests/test_regime_audit.py -v
  ```

- **Quant finding (NEW, headline): the scorer's near-zero OOS skill is
  REGIME-UNIFORM — the conviction gate cannot be rescued by conditioning on
  regime.** Live pkl `n_train=3830`, gate active, OOS n=1000. Full
  **in-sample** `REGIME_UNIFORM_EDGE`: sideways `rank_ic +0.482`
  (`dir_acc 0.699`, gate tail−head **+23.90pp**, n=1455) and bull_or_unknown
  `rank_ic +0.531` (`dir_acc 0.692`, **+20.31pp**, n=3532) both look
  strongly skilled. Temporal **OOS** `REGIME_UNIFORM_NULL`: the SAME two
  regimes collapse to sideways `rank_ic +0.044` (**+1.33pp**) and
  bull_or_unknown `rank_ic −0.023` (**−1.64pp**). The in-sample→OOS
  collapse is essentially identical in *both* measurable regimes — a
  regime-invariant overfit signature. bear shows OOS `rank_ic +0.548` but
  n=8 (13/5000 corpus): correctly flagged `thin`/not-measurable so it never
  masquerades as edge. **Decisive addition to passes #7/#8: the negative
  OOS skill is NOT a regime-mix artifact; there is no measurable regime in
  which the gate carries edge — the "maybe it works in bull/sideways"
  escape hatch is closed by data.** Reported, not actioned (model-dynamics
  / CLAUDE.md §6; the gate is invariant #5).

- **Quant finding: the trustworthy OOS holdout is itself a single
  down-period.** All 10 OOS deciles realize *negative* (−0.08…−1.99%;
  `calibration --oos` re-confirmed); slice is ~half sideways (506/1000) /
  ~half bull_or_unknown (486/1000), only 8 bear. The
  `REGIME_UNIFORM_NULL` verdict is robust within what is measurable; bear
  is structurally untestable from this corpus — surfaced via the `thin`
  flag, not a fabricated 8-sample edge claim.

- **Live health.** `backtest.db`: 480 complete / 20 failed / 5 running; 0
  NaN finals; 0 `benchmark_unavailable` (current windows carry SPY).
  `total_return_pct` median **+62.7%** (min −54.6, max +2979);
  `vs_spy_pct` median **+38.4%** (min −170, max +2820) — leveraged-beta
  dispersion, not alpha. `scorer_skill_log.jsonl` last 8 cycles all
  `status=ok`, `gate_active=true`, `val_rmse` 6.0–12.7 ≪ `oos_rmse`
  10.2–17.7, `oos_dir_acc` 0.47–0.55, `oos_ic` −0.06…+0.12 — the overfit
  the new regime view now localizes. `continuous.log` fresh, mid-cycle, no
  crashes.

- **Operational (reconfirmed, out of scope):** winner→ArticleNet feedback
  loop still dead (`trainer timeout` / `inject err: database is locked` —
  digital-intern GPU + `articles.db` write contention on the `/media/...`
  symlinked volume). The loop should not be read as "training on its
  winners".

- **Run the ML/backtest suite:** `cd /home/zeph/trading-intelligence/paper-trader
  && python3 -m pytest tests/test_decision_scorer.py tests/test_backtest.py
  tests/test_calibration.py tests/test_validation.py tests/test_continuous.py
  tests/test_ml_backtest_review.py tests/test_gate_audit.py
  tests/test_feature_importance.py tests/test_skill_trend.py
  tests/test_regime_audit.py -q` (269 fast offline tests, green).

### 2026-05-18 review pass #12 (ML+backtest hybrid · trivial-baseline comparison · live findings)

- **Phase 1 — no new bugs (bugs_fixed = 0; no Phase-1 commit).** Full
  re-trace of `decision_scorer.py`, `backtest.py`,
  `run_continuous_backtests.py` plus coupled `validation.py` /
  `calibration.py`: the `predict_with_meta` off-distribution
  gate-abstention path in `_ml_decide` (the `_pwm` callable probe + the
  `not scorer_off_dist` guard on the n_train≥500 gate), the universal SELL
  `-forward_return_5d` sign-flip (train↔inference↔calibration↔gate↔
  `_oos_rank_metrics`), the `(ticker,sim_date,action)` dedup key, the
  5-trading-day forward-window guard, `split_outcomes_temporal`'s
  most-recent-by-sim_date holdout, the separately-guarded
  `_train_decision_scorer` train / oos-rmse / oos-rank blocks, and the
  `_parse_scorer_status` `(?:^|\s)key=` token regex were all re-verified
  correct and exact-value test-locked. Consistent with the documented 11+
  prior no-new-bug ML/backtest passes — not a fabricated fix. ML/backtest
  subset 269/269 green before the feature, 289/289 after.

- **Feature shipped (commit `7489716`): trivial-baseline comparison.**
  `paper_trader/ml/baseline_compare.py`. The gap it fills: `skill_trend`
  already compares the scorer's `oos_rmse` to the only trivial it knows —
  a **constant** mean-predictor (σ(target) floor). **Nothing compared the
  17-feature MLP to a non-constant one-line rule** (raw `ml_score`,
  momentum carry, RSI/Bollinger mean-reversion). That is the decisive
  quant question once `gate_audit=GATE_INEFFECTIVE` /
  `regime_audit=REGIME_UNIFORM_NULL` are on record: is the neural net
  extracting signal a single feature already carries, or is it genuinely
  additive OOS? It scores the deployed MLP and 6 trivial baselines on the
  **exact temporal-OOS slice** every sibling tool uses
  (`validation.split_outcomes_temporal`), on two **scale-invariant**
  primitives — `rank_ic` (reusing `calibration._spearman`, the tie-aware
  SSOT, mandatory vs the ±50 clamp) and `dir_acc` (RMSE is unusable: a
  `mom20` baseline predicts in a different unit, so an RMSE race is
  decided by scale not skill). The codebase-universal SELL sign-flip is
  applied to the realized target **and symmetrically to every baseline's
  prediction** (the training-aligned MLP pred is NOT flipped — exactly
  `calibration.scorer_calibration`); without the symmetric baseline flip a
  feature baseline fabricates a false `MLP_ADDS_SKILL` on the SELL subset.
  Verdicts: `INSUFFICIENT_DATA` / `MLP_WORSE_THAN_TRIVIAL` /
  `MLP_NO_BETTER_THAN_TRIVIAL` / `MLP_ADDS_SKILL`, with a `MLP_IC_MIN=0.10`
  skill floor so "beats every one-liner because *everything* is noise" is
  not misread as additive skill. A constant baseline is flagged
  `degenerate` and can never be selected as `best_baseline` (the
  `feature_importance` honesty pattern). Same discipline as
  `ml/calibration.py`: read-only, no train/pickle/`build_features`/
  `N_FEATURES`/trade touch, never raises, CLI exits 2 on
  `MLP_NO_BETTER`/`MLP_WORSE` (cron-branchable). NOT wired into `main()` —
  zero deploy-stale impact, no loop restart needed. 20 exact-value locks
  in `tests/test_baseline_compare.py` (full verdict matrix at ±1.0/0.0
  Spearman by construction; the skill floor isolated from the
  within-margin arm; an all-SELL slice locking BOTH flip arms;
  degenerate-never-best; OOS-slice restriction; never-raises).
  ```bash
  cd /home/zeph/trading-intelligence/paper-trader && python3 -m paper_trader.ml.baseline_compare
  cd /home/zeph/trading-intelligence/paper-trader && python3 -m paper_trader.ml.baseline_compare --all
  cd /home/zeph/trading-intelligence/paper-trader && python3 -m pytest tests/test_baseline_compare.py -v
  ```

- **Quant finding (NEW, headline): there is no simple OOS signal for the
  MLP to fail to generalize — every one-liner is dead OOS too.** Live pkl
  `n_train=3830`, gate active, OOS slice n=1000. Verdict
  `MLP_NO_BETTER_THAN_TRIVIAL`, but **not** because a one-liner wins: the
  MLP's OOS `rank_ic=+0.013` (`dir_acc 0.498`, a coin flip) is below the
  0.10 skill floor, and **every trivial baseline is also ≈0/negative OOS**
  (best `rsi_meanrev −0.003`; `ml_score −0.043`; `mom20 −0.046`;
  `mom5 −0.020`; `neg_bb −0.025`). On `--all` (in-sample) the SAME tool
  reports `MLP_ADDS_SKILL` (MLP `rank_ic +0.510` vs best baseline
  `ml_score +0.062`). The in-sample→OOS collapse of the MLP (0.510 → 0.013)
  while **no simple feature is even good in-sample** (best 0.062) is a
  crisp **pure-memorization fingerprint**: this refines pass #8's
  "signal-grounded — leans on rsi/bb/mom" finding — the MLP's leaned-on
  features carry ≈0 OOS rank skill *even as one-liners* in this
  leveraged-ETF universe, so the failure is **not** "the net can't
  generalize a good simple signal", it is "there is no simple signal here
  to generalize". The conviction gate (invariant #5, `gate_active` every
  cycle) is therefore underwriting sizing variance against a model whose
  apparent skill is entirely in-sample artifact, with no trivial
  alternative that would do better. Reported, **not actioned** —
  model-dynamics / CLAUDE.md §6.

- **Cross-check integrity confirmed (no tool drift).** `baseline_compare`'s
  OOS MLP `rank_ic = 0.0128` is byte-equal to `calibration --oos`
  `spearman = 0.0128` (both go through `calibration._spearman`, the single
  source of truth), and consistent with `skill_trend` median `oos_ic 0.015`
  / `NEGATIVE_OOS_SKILL`, `regime_audit REGIME_UNIFORM_NULL`,
  `gate_audit GATE_INEFFECTIVE` (`tail−head +0.58pp`, all five arms
  negative-realized on the OOS slice). The advisor's "if it reports
  `MLP_ADDS_SKILL` OOS, that is a sign-flip/split bug not a discovery"
  blocking concern is resolved: OOS verdict corroborates the documented
  negative-OOS-skill picture exactly.

- **Live health.** `backtest.db`: 480 complete / 20 failed / 5 running; 0
  NaN finals; 0 `benchmark_unavailable`. `total_return_pct` median
  **+62.7%** (min −54.6, max +2979); `vs_spy_pct` median **+38.4%**
  (min −170, max +2820) — leveraged-beta dispersion, not alpha, exactly as
  every prior pass documents. `scorer_skill_log.jsonl` cycles all
  `status=ok`, `gate_active=true`, `val_rmse` 6.0–12.7 ≪ `oos_rmse`
  10.2–14.6, `oos_ic` −0.06…+0.12 — the textbook overfit `baseline_compare`
  now localizes to "no simple OOS signal exists here". `continuous.log`
  fresh, mid-cycle; only external GDELT `ConnectionReset` noise (handled
  w/ backoff).

- **Operational (reconfirmed, out of scope):** winner→ArticleNet feedback
  loop still dead two ways — `[continuous] ml: trainer timeout (injected N)`
  and `[continuous] ml: inject err: database is locked` (digital-intern GPU
  + `articles.db` write contention on the `/media/...` symlinked volume).
  Matches passes #6/#7/#8/#11; the loop should not be read as "training on
  its winners". Reported, not actioned (digital-intern-side / infra).

- **Run the ML/backtest suite (now 289):** `cd
  /home/zeph/trading-intelligence/paper-trader && python3 -m pytest
  tests/test_decision_scorer.py tests/test_backtest.py
  tests/test_calibration.py tests/test_validation.py tests/test_continuous.py
  tests/test_ml_backtest_review.py tests/test_gate_audit.py
  tests/test_feature_importance.py tests/test_skill_trend.py
  tests/test_regime_audit.py tests/test_baseline_compare.py -q`
  (289 fast offline tests, green).

### 2026-05-18 review pass #13 (ML+backtest hybrid · training-corpus & OOS-construction audit · decisive live finding)

- **Phase 1 — no new bugs (bugs_fixed = 0; no Phase-1 commit).** Full
  re-trace of `decision_scorer.py`, `backtest.py`,
  `run_continuous_backtests.py` plus coupled `validation.py` /
  `calibration.py` / `gate_audit.py`: the BUY-path scorer-feature
  construction in `_ml_decide` vs the training-side reconstruction in
  `_compute_decision_outcomes` (`ml_score`=`best_score` regime-multiplied
  parsed from reasoning vs full-precision at inference — consistent to
  rounding; `regime_mult` recomputed identically off the same `engine.prices`;
  the news-default symmetry `buy_news_count==0 → None → build_features
  urg=50/cnt=1` on BOTH sides), the `score=` first-match regex vs `scorer=`
  (no `score=` substring inside `scorer=`), the universal SELL
  `-forward_return_5d` sign-flip (train↔inference↔calibration↔gate↔
  `_oos_rank_metrics`↔`evaluate_scorer_oos`), the off-distribution gate
  abstention, the 11-column `_inject_and_train` INSERT alignment, the
  separately-guarded `_train_decision_scorer` train/oos-rmse/oos-rank blocks,
  the numpy-lstsq fallback scaler, every module-global lock — all re-verified
  correct and exact-value test-locked. Consistent with the documented 12+
  prior no-new-bug ML/backtest passes — not a fabricated fix. ML/backtest
  subset 290/290 green before the feature, 309/309 after.

- **Feature shipped (commit `e109f88`): training-corpus & OOS-construction
  audit.** `paper_trader/ml/corpus_audit.py`. The gap it fills: every
  sibling diagnostic (`calibration` deciles, `gate_audit` arms, `skill_trend`
  ledger-trend, `feature_importance` attribution, `regime_audit` regime
  buckets, `baseline_compare` trivial one-liners) takes the corpus **as
  given** and measures the scorer's skill on the temporal-OOS slice
  `validation.split_outcomes_temporal` carves out. **None validate that the
  slice is a genuine held-out draw.** That matters because of how the corpus
  is produced: `MAX_OUTCOMES_FOR_TRAINING=5000` caps
  `decision_outcomes.jsonl`; each cycle runs `RUNS_PER_CYCLE=5` backtests
  over **one random multi-year window** emitting ≈1000 outcomes/run ≈ 5000
  rows — so the cap ≈ **one cycle's one window**; and each backtest run emits
  decisions across the whole window, so when the split sorts by `sim_date`
  and holds out the latest fraction, every run contributing to OOS (its late
  `sim_date` rows) **also contributed to train** (its early rows). The
  loop's "temporal OOS holdout" is therefore the late slice of the *same*
  runs over the *same* window — a within-window front/back split, **not** a
  generalization test against an unseen window/regime. The tool applies the
  EXACT `split_outcomes_temporal` (single source of truth — a split mismatch
  would describe a different slice than every other OOS tool) and verdicts on
  the train↔OOS run-set relationship: `INSUFFICIENT_DATA` /
  `OOS_NOT_HELD_OUT` (run-subset **and** ≤`NARROW_MAX_RUNS=10` distinct
  runs — the decisive alarm) / `OOS_OVERLAPS_TRAIN` (run-subset but
  many-window corpus — milder) / `OOS_HELD_OUT` (≥1 OOS run absent from
  train — genuine separation). `corpus_breadth`/`regime_mix` are
  informational, NOT folded into the verdict (the `gate_audit`
  arm-monotone honesty pattern), so the verdict stays crisply exact-value
  testable. Read-only, no train/pickle/`build_features`/`N_FEATURES`/trade
  touch, never raises, CLI exits 2 on `OOS_NOT_HELD_OUT`. NOT wired into
  `main()` — zero deploy-stale impact. 19 exact-value locks in
  `tests/test_corpus_audit.py`.
  ```bash
  cd /home/zeph/trading-intelligence/paper-trader && python3 -m paper_trader.ml.corpus_audit
  cd /home/zeph/trading-intelligence/paper-trader && python3 -m pytest tests/test_corpus_audit.py -v
  ```

- **Quant finding (NEW, decisive): the trustworthy OOS metric is not a
  generalization test.** Live `decision_outcomes.jsonl`: **5000 rows, 5
  distinct run_ids (6226–6230), one cycle, one window 2013-01-22 →
  2018-01-11** (`OOS_NOT_HELD_OUT`, breadth `SINGLE_DRAW`,
  `likely_single_cycle=True`, regime mix **80.9% bull_or_unknown**). The
  loop's `oos_rmse`/`oos_ic` and `calibration --oos` / `regime_audit` /
  `baseline_compare` OOS verdicts are computed on train sim_date ≤
  2017-04-07 vs OOS sim_date ≥ 2017-04-10 of the **same 5 backtest runs**
  (`oos_run_ids in_train=5, not_in_train=0, shares_all=True`). This refines
  every prior pass's "textbook overfit / negative OOS skill": that collapse
  is measured on the **most favorable possible holdout** — same runs, same
  window, one contiguous low-vol bull regime — and the scorer **still**
  collapses (`calibration --oos` MISCALIBRATED spearman 0.19, decile-realized
  flat d1 −0.39 vs d10 +2.27; `skill_trend` NEGATIVE_OOS_SKILL oos_rmse
  11.30 ≫ 5.67 mean-predictor baseline, **trend DEGRADING**, `gate_active=1.0`
  on all 11 ledger cycles; `baseline_compare` MLP OOS rank_ic 0.19 ≈ raw
  `ml_score` 0.20, `ic_gap −0.007` — the 17-dim net adds **nothing** over
  its own input feature OOS). A true held-out window would be *worse*, not
  better — so the no-edge conclusion is strengthened, and the conviction
  gate (invariant #5, active every cycle) is underwriting sizing variance
  against a model whose only measurable "OOS" number is itself a
  within-window artifact. Reported, **not actioned** — neither the
  `MAX_OUTCOMES_FOR_TRAINING` cap nor the gate is in surgical scope
  (model-/training-dynamics, CLAUDE.md §6).

- **Operational (durable, NEW — out of surgical ML scope, reported):**
  `backtest.db` (now **278 MB**, on the `/media/zeph/projects` symlinked
  volume, with a **stale 4.2 MB WAL not checkpointed since 2026-05-17
  01:58** though the loop is actively writing) cannot service a `mode=ro`
  `SELECT COUNT(*)` within 30 s even with `busy_timeout=8000` (`rc=124`,
  reproduced twice). The dashboard's `/api/backtests*` endpoints read this
  DB **per HTTP request**, and digital-intern's `:8080` dashboard
  cross-fetches them — so those panels are effectively unresponsive under
  this condition. Root cause is infra (volume latency / WAL-checkpoint
  starvation / 278 MB DB), not ML logic; surfaced here because a skeptical
  quant reading the backtest dashboard would see hangs, not data.

- **Operational (reconfirmed, out of scope):** winner→ArticleNet feedback
  loop still dead — `continuous.log`: `[continuous] ml: trainer rc=-15
  injected=10000` (SIGTERM on digital-intern's 120 s-capped
  `ml.trainer.train(force=True)`; injection succeeds, training does not).
  Matches passes #6/#7/#8/#11/#12 — the loop is not "training on its
  winners". digital-intern GPU + `articles.db` write contention; reported,
  not actioned.

- **Live health.** `backtest.db` (read via the static `.local_backup`
  snapshot, since the live symlink times out): 486 complete / 20 failed /
  4 running; 0 NaN finals; 1 `benchmark_unavailable`. `total_return_pct`
  median **+63.1%**; `vs_spy_pct` median **+40.0%** — leveraged-beta
  dispersion, not alpha, exactly as every prior pass documents. Scorer
  pickle `n_train=3234`, `gate_active=True`. `continuous.log` fresh,
  mid-cycle (run 6231); only handled external GDELT `ConnectionReset`/
  `RemoteDisconnected` noise (backoff 20/40/60 s) — no Python tracebacks,
  no `[engine] RUN N CRASHED`, no `scorer err` / `inject err`.

- **Run the ML/backtest suite (now 309):** `cd
  /home/zeph/trading-intelligence/paper-trader && python3 -m pytest
  tests/test_decision_scorer.py tests/test_backtest.py
  tests/test_calibration.py tests/test_validation.py tests/test_continuous.py
  tests/test_ml_backtest_review.py tests/test_gate_audit.py
  tests/test_feature_importance.py tests/test_skill_trend.py
  tests/test_regime_audit.py tests/test_baseline_compare.py
  tests/test_corpus_audit.py -q` (309 fast offline tests, green).

---

### 2026-05-18 review pass #13 (paper-trader core hybrid · news-DB lock no longer aborts the cycle · NYSE half-day enforcement · live findings)

- **Phase 1 — 1 bug fixed (commit `fe5881d`).** `signals.py`'s four
  decision-path readers (`get_top_signals`, `get_urgent_articles`,
  `get_ticker_sentiment`, `ticker_sentiments`) wrapped the query in
  `try: conn.execute(...) finally: conn.close()` with **no `except`**. A
  transient `sqlite3.OperationalError: database is locked` from the
  digital-intern `articles.db` (the daemon mid-WAL-checkpoint — observed live
  in `runner.log`, `get_top_signals` line 294) propagated out of
  `strategy.decide()`, which `runner._cycle` only catches generically — so
  the **entire decision cycle was lost**: no decision, no equity point, for a
  *news* DB hiccup. All four readers now `except sqlite3.Error`, log once, and
  degrade to the **same safe default the `if not conn` arm returns**
  (identical to a missing DB) so trading continues on quant + portfolio
  context. `sqlite3.Error` only — a non-sqlite bug still surfaces. Locked by
  `tests/test_signals_lock_degrade.py` (per-reader degraded value + the
  connection is still closed, no fd leak + the `decide()` merge survives) and
  an exact-value P&L regression guard `tests/test_round_trips_pnl.py` for
  `build_round_trips` (the realized-today single source of truth: scale-in /
  partial-close / fractional-residue / option ×100 — no bug found, pinned).

- **Phase 2 — 1 feature (commit see below).** `market.py` had **no NYSE
  early-close handling** ("Half-days not enforced — we'll trade through
  them"). On the day after Thanksgiving (2026-11-27) and Christmas Eve
  (2026-12-24) NYSE closes at **1:00 p.m. ET**; the engine believed the
  market was open 13:00–16:00 ET, ran the fast 30-min OPEN cadence and
  *executed trades against frozen post-close yfinance marks* for three hours
  of a CLOSED market, twice a year. Added `NYSE_HALF_DAYS_2026`,
  `is_half_day(d)`, `close_minute(d)` (13:00 on a known half-day, else the
  regular 16:00); `is_market_open` now gates on `close_minute(date)`. Fully
  backward-compatible — an unknown half-day still falls through to the 16:00
  close (same conservative default as the holiday calendar), and an
  exhaustive per-minute test proves every regular weekday is byte-identical
  to the old `9:30 ≤ m < 16:00` rule. Locked by
  `tests/test_market_half_day.py` (11 tests, + 36 existing `test_core_market`
  green). This corrects the runner sleep cadence, the prompt `MARKET_OPEN`
  flag, and every market-hours gate on those two days.

- **Phase 3 — live findings (reported, not all in-domain to fix):**
  1. **NO_DECISION rate 58.9% lifetime / 51.9% in 24h** (`/api/decision-health`)
     — the dominant failure mode; the live trader produces no decision more
     than half the time. Owned by the concurrent JSON-parse agent; the Phase 1
     fix at least stops a locked news DB *adding* to this count.
  2. **Strategy lagging buy-and-hold S&P by 2.25pp** ($972.69 vs $995.20),
     ahead in only **0.5% of 755 cycles** (`/api/benchmark`) — strategy
     underperformance, not a code defect.
  3. **Discord delivery DEGRADED** (`/api/runner-heartbeat` → `notify`):
     `verdict DEGRADED`, `last_ok_ts null`, `openclaw timeout (60s)`. The
     operator's only alarm channel is dark this process. Root cause is
     environmental — load avg **~23 on 16 cores** (the parallel review agents
     + continuous backtests + the test suite saturate the box; the
     `node`/PATH resolution itself is fixed and verified `rc 0`). The 60s
     `reporter._send` timeout is too tight under that load, but `reporter.py`
     was being concurrently edited by another agent so it was left untouched
     to avoid a collision.
  4. **Suspicious cost basis** (`/api/risk`): MU marked ≈ $724/sh, LITE ≈
     $970/sh — ~10× real prices; the open book appears to have been entered at
     corrupted yfinance prices at some past point. Equity accounting is
     internally consistent (cash + Σ market_value = total_value) but built on
     bad marks. Historical data corruption, not a live code path to patch
     surgically — flagged for an operator DB review.
  5. Dashboard endpoints all 200 and sub-10 ms (SWR cache healthy) even under
     load avg 23, though `/api/*` occasionally exceeds an 8 s client timeout
     at that saturation (environmental).

- **Run the core suite:** `cd /home/zeph/trading-intelligence/paper-trader &&
  python3 -m pytest tests/ -v` (full ~1491). Fast core subset for this pass:
  `python3 -m pytest tests/test_core_signals.py tests/test_core_strategy.py
  tests/test_core_store.py tests/test_core_market.py tests/test_core_runner.py
  tests/test_signals_lock_degrade.py tests/test_round_trips_pnl.py
  tests/test_market_half_day.py -q`.

---

### 2026-05-18 ops session (dashboard polish · backtest throttle · stale-code restarts · live findings)

Not a review pass — an operator-driven maintenance + deploy session. Every
commit hash below was verified on disk (`git show`), and the throttle / runner
changes were re-read in `run_continuous_backtests.py` and `paper_trader/runner.py`
at write time. A future agent picking this up should treat the "outstanding"
list as the live to-do.

**What was fixed / changed this session**

- **Dashboard cosmetics (commit `b49114c`, on disk).**
  - Removed the stray leading `→ ` from the **position thesis cards** — the
    JS template in `refreshThesis()` (`paper_trader/dashboard.py` ~L3708) was
    `<div …>→ ${c.thesis||"—"}</div>`; the arrow rendered as "random arrows on
    the left side" of every thesis card. Now `${c.thesis||"—"}` with no prefix.
  - **Last Validation** timestamp now human-readable: `refreshValidation()`
    (~L4376) sets `val-last-when` via
    `new Date(latest.timestamp).toLocaleString()` instead of dumping the raw
    ISO string. Pure front-end string formatting — no API/contract change.
  - This was a large diff (+162/−2) because the same commit also carried the
    `/api/hold-discipline` + `runner_heartbeat` work; the two one-line UI
    fixes are the lines quoted above.

- **Continuous-backtest throttle (commit `bf23133`, on disk).**
  `run_continuous_backtests.py`: `RUNS_PER_CYCLE` now **`1`** (the dispatching
  operator recalled it as `3→1`; the commit message and on-disk comment only
  assert "throttled to 1" — CLAUDE.md §7 documents the historical default as
  `5`, so the *current* value `1` is the load-bearing fact, not the "from"),
  `COOLDOWN_SECONDS` `300→600` (confirmed by the on-disk comment "throttled
  from 300s"), and `TOP_RUNS_TO_TRAIN` also dropped to `1` (only the single
  best run trains when throttled). Driven by a sustained load average of
  **37+**.
  **Treat these as a floor, not a default — do NOT raise them back without an
  explicit decision.** This mirrors and is consistent with the standing
  `continuous-backtests OOM` operating note (the box is RAM/load-constrained;
  `_CLAUDE_SEM=3`, `nice 10`, single run-cycle are all deliberate governors).
  Current on-disk values confirmed: `RUNS_PER_CYCLE = 1`,
  `TOP_RUNS_TO_TRAIN = 1`, `COOLDOWN_SECONDS = 600` (lines 48/49/52).

- **Git-watcher deferred restart (commit `cf516c0`, on disk, applied).**
  `paper_trader/runner.py::_git_watcher` records git HEAD at boot, sleeps 120s
  for startup, then re-polls every 180s. On a HEAD change it pings Discord,
  sets `_restart_requested`, and returns; the **main loop** performs the actual
  `os._exit(0)` at the next cycle boundary (or interrupts the inter-cycle
  sleep via `_restart_requested.wait()`), so a committed fix is auto-applied
  without ever killing a mid-Opus decision call. systemd `Restart=always`
  brings the process back on the new code. Fail-open: any git/subprocess error
  just skips that poll. Already committed and applied via the service restart
  below.

- **DB-count fast path (commit `5265d8e`, on disk, applied).**
  `/api/stats` is now O(log N): total via `MAX(rowid)` plus cached backlog
  counts instead of a full `COUNT(*)` over the large WAL DB. Already committed;
  applied via the service restart.

- **Stale-code service restarts (operator action, this session).**
  `paper-trader` and `unified-proxy` were running code older than HEAD
  (`/api/build-info` `stale:true`) so the committed fixes above were on disk
  but not live. Both were restarted via `systemctl --user restart` to pick up
  HEAD. After any commit to this repo, confirm the running process is current
  with `curl -s localhost:8090/api/build-info` (look for `stale:false`); the
  in-process git-watcher (`cf516c0`) now does this automatically within ~3 min,
  but a manual restart is the immediate remedy.

**Outstanding / known issues (live to-do for the next agent)**

- **NO_DECISION rate ~54%.** This session's working diagnosis was *Opus
  returning a valid decision wrapped in a markdown fence the parser missed*,
  and a fix agent was dispatched (fix pending, not yet landed at session end).
  **Important context for whoever picks this up:** the standing, repeatedly
  documented finding (CLAUDE.md §11; AGENTS.md ML/backtest passes #6–#12; the
  operator memory note "paper-trader NO_DECISION = quota, not JSON") is that a
  high NO_DECISION rate is most often **Claude org usage-limit/quota
  exhaustion plus concurrent-agent contention**, *not* a parser bug — the
  `_parse_decision` path already strips ```json fences and `raw_decode`s the
  first object, has a Sonnet fallback, and a JSON-only retry. Before "fixing
  the parser", verify with `/api/decision-forensics` and the runner stdout
  whether the failures are `quota_exhausted` (the runner sets that flag and
  fires `send_quota_alert`) vs genuinely-unparseable non-empty text. Do not
  re-fix an already-robust parser if the real cause is quota.
- **System load still elevated (~25).** Partly the parallel review/fix agents
  running the pytest suites; partly the box's baseline. The backtest throttle
  above is the main lever already pulled. Watch `uptime` / the
  `continuous-backtests` cooldown before adding any new concurrent workload.

**Service management (all user units — note `--user`)**

```bash
systemctl --user {start,stop,restart,status} paper-trader
systemctl --user {start,stop,restart,status} continuous-backtests
systemctl --user {start,stop,restart,status} unified-proxy
```

`paper-trader` is the live trader (`python3 -m paper_trader.runner`),
`continuous-backtests` is the training loop (`run_continuous_backtests.py`),
`unified-proxy` is the tailscale-funnel'd reverse proxy on `:8888`. These run
as **user** services — a `sudo systemctl` / system-unit invocation targets the
wrong unit (this has historically caused duplicate-runner double-trading; the
`runner.py` single-instance flock, invariant #19, is the guard).

**Key file locations**

| Path | Role |
|------|------|
| `paper_trader/dashboard.py` | Single-file Flask app on `:8090`, ~7–8k lines — HTML `TEMPLATE` + inline JS (`refreshThesis`, `refreshValidation`, …) + ~45 `/api/*` routes |
| `paper_trader/runner.py` | Live trading loop — cycle, single-instance flock, git-watcher, circuit breaker, restart-durable report markers |
| `run_continuous_backtests.py` | ML training loop (the `continuous-backtests` service) — throttle constants at lines 48–52 |
| `paper_trader/store.py` | SQLite store, `data/paper_trader.db` (WAL) — live portfolio/positions/trades/decisions/equity_curve |
| `backtest.db` | SQLite, `backtest_runs` / `_trades` / `_decisions` (run history, equity curves) |
| `data/decision_outcomes.jsonl` | DecisionScorer training data (forward 5d returns) |
| `data/ml/decision_scorer.pkl` | Trained MLP pickle |

**Architecture reminder (ports)**

- `paper-trader` dashboard → **`:8090`**
- `digital-intern` dashboard/API → **`:8080`** (paper-trader reads its
  `articles.db` read-only; digital-intern cross-fetches `:8090/api/portfolio`)
- `unified-proxy` (tailscale funnel front door) → **`:8888`** — single public
  ingress; both dashboards are reached through it

*Ops session appended 2026-05-18. Prior content above is unmodified.*

---

### 2026-05-18 review pass #14 (paper-trader core hybrid · clock-step-back marker hardening · git-watcher deadman · the `Restart=on-failure` self-restart defect)

- **Phase 1 — 1 bug fixed (commit `8ad0420`).**
  `runner._restore_runner_state` rehydrated the restart-durability sidecar
  (`runner_state.json`) **verbatim**, with no upper bound on the persisted
  markers. A wall-clock step BACKWARD *after* a `_save_runner_state` write
  (NTP correction / VM time-sync — this box has documented clock+load
  stress) leaves `last_hourly_iso` in the **future**. Restoring it makes
  `(now - _last_hourly) < 3600` true for up to (skew + 1h), so
  `_maybe_hourly` silently **MUTES** the hourly Discord summary — the
  operator's primary monitoring surface goes dark with zero signal, the
  exact "Hourly STARVATION" class the sidecar exists to prevent.
  Symmetrically a `daily_close_sent_for` strictly after today (NY)
  suppresses *that* day's real close once the clock reaches it (the
  `== today` gate then matches a date for which nothing was sent).
  Reproduced offline. Fix: clamp a future `_last_hourly` back to `now`
  (normal 1h cadence resumes, never muted longer than intended) and drop a
  future `daily_close_sent_for` (treat as not-sent — fresh-boot behaviour,
  never suppress a real close). Past/overdue markers restore verbatim (no
  dedup/overdue regression). Locked by 4 new tests in
  `TestRunnerStatePersistence`
  (`test_restore_clamps_future_last_hourly_so_hourly_is_not_muted`,
  `…drops_future_daily_close_sent_for`,
  `…keeps_today_and_past_daily_close`, `…past_last_hourly_unchanged`).

- **Phase 2 — 1 feature (commit `afaef6b`).** Git-watcher **deadman
  safety-net**. The watcher requested a deferred restart then `return`ed,
  trusting the main loop to `os._exit(0)` at the next cycle boundary. Under
  heavy host load (observed live: load avg ~23, a multi-day-uptime runner
  still on stale code, `/api/build-info behind:1` — a committed fix never
  deployed) the loop can be wedged so long the boundary never arrives and
  the fix sits unapplied indefinitely; with the watcher already returned
  there was no fallback. The watcher now **persists** as a deadman: after
  requesting the graceful restart it keeps polling and, if still unhonored
  `RESTART_GRACE_S=600s` later, force-exits itself (clean `os._exit(0)`;
  systemd reboots on fresh code — see the Phase-3 caveat). The grace window
  is provably above the worst-case *healthy* cycle (strategy claude budgets
  `DECISION_TIMEOUT_S 180 + RETRY 45 + FALLBACK 60` + 180s poll = 465s) so
  a slow-but-live loop is never force-killed — only a genuinely wedged one.
  Decision extracted to the pure `_deferred_restart_overdue()` predicate
  (monotonic clocks — immune to the very wall-clock step-back Phase-1
  hardens). Locked by 6 tests in `TestDeferredRestartOverdue` incl. the
  grace-vs-worst-healthy-cycle invariant.

- **Phase 3 — live findings (the first is fixed; commit `bb6a23f`).**
  1. **`paper-trader.service` had `Restart=on-failure`, silently breaking
     the ENTIRE self-restart mechanism — fixed → `Restart=always`.** Every
     `runner.py` self-restart exits **cleanly** via `os._exit(0)` (the
     git-watcher deferred restart, the new deadman, the deliberate
     duplicate-instance exit). Under `on-failure` systemd treats exit 0 as
     *success* and does **not** restart, so a committed fix never deploys
     and the trader stays down — the root cause of the observed
     `behind:1` / "stale for days" pathology. `runner.py` (L410/416/489)
     and `CLAUDE.md`/this file all explicitly assert "systemd
     `Restart=always` brings us back on the new code"; reality was
     `on-failure`. `Restart=always` makes the documented contract true and
     makes the Phase-2 deadman actually function. **Operator action
     required:** `systemctl --user daemon-reload && systemctl --user
     restart paper-trader` for the running unit to pick up the repo change
     (a repo edit alone does not reinstall the unit).
  2. **systemd restart-counter churn (≥13).** During heavy
     concurrent-commit deploy storms each restart briefly races two
     `runner.py` instances; the singleton flock correctly forces the loser
     to `sys.exit(1)` (logged "Failed with result exit-code"). This is the
     guard **working** — it self-heals to a single trader (heartbeat
     confirmed HEALTHY, one lock holder) — but it inflates the restart
     counter and is noisy. With default `StartLimitBurst=5 /
     StartLimitIntervalSec=10s` and `RestartSec=10` the burst limit is not
     tripped (≤1 restart per 10s), so it is noisy-but-safe; left as an
     observation, not patched.
  3. **NO_DECISION ~53% (24h), ~59% lifetime — confirmed = claude-CLI
     timeouts under host saturation**, NOT a parser bug and NOT (this
     sample) hard quota. The recorded reason string is uniformly
     `"claude returned no response (timeout/empty)"` (timeout path), with
     `quota_exhausted` *unset*. Consistent with the long-standing
     documented contention/quota diagnosis (CLAUDE.md §11; ML/backtest
     passes #6–#12) — load avg ~23 starves the 180s Opus budget. No code
     change: the parser is already robust; the lever is host load /
     concurrency, not `_parse_decision`.
  4. **Capital paralysis on corrupted marks (confirmed, documented #4).**
     Cash $18.49; ~97% of the $972.69 book is two fractional positions
     `MU 0.5 @ $724.12` and `LITE 0.61 @ $980.90` — yfinance returns these
     implausible prices *consistently* for both tickers (so an
     `avg_cost/current_price` divergence check would NOT catch it; ratio
     ≈ 1.0). Equity accounting is internally consistent but built on bad
     marks, and with ~$18 free the book cannot meaningfully trade. Historical
     data corruption + a price-feed anomaly for these symbols — an operator
     DB/feed review, not a surgical code path. (This is why the cost-basis
     *divergence* feature a prior advisor suggested was NOT built: the
     live data proves divergence is the wrong detector here.)
  5. **Positives verified:** Discord delivery is **HEALTHY** again
     (`/api/runner-heartbeat` → `notify.verdict HEALTHY`, recent
     `last_ok_ts`) — recovered since the session-#13 DEGRADED finding;
     dashboard `/` 200 in ~45 ms; singleton lock `acquired` (not degraded).

- **Run the core suite:** `cd /home/zeph/trading-intelligence/paper-trader
  && python3 -m pytest tests/ -v` (full). Fast core subset for this pass:
  `python3 -m pytest tests/test_core_runner.py tests/test_core_signals.py
  tests/test_core_strategy.py tests/test_core_store.py
  tests/test_core_market.py tests/test_runner_heartbeat.py
  tests/test_parse_retry.py -q` — `test_core_runner.py` now holds the
  future-marker-clamp + deadman-predicate locks (50 tests).

*Review pass #14 appended 2026-05-18. Prior content above is unmodified.*

---

### 2026-05-18 feature-dev pass — SWR cold-path failure observability + scorer-confidence bounded

**User-perspective testing surfaced a real production defect, not a missing
feature.** The `:8090` analytics surface is already very mature (~64 routes;
sector-heatmap / drawdown / calibration / suggestions / correlation / Calmar
all exist). The high-impact gap is **reliability/observability**, reproduced
live (read-only HTTP probes against the running service):

- **Observed (empirical, not inferred):** `/api/briefing` returned
  `{"warming":true}` on **8+ consecutive polls over 60s+**, never serving
  real data in-window; `news-edge` / `source-edge` / `decision-context`
  same. `/api/scorer-confidence` — the **one** expensive-replay endpoint
  **not** `@swr_cached` — hung the request thread `>30s` (curl code 000).
  `runner.log` carried **zero** SWR exception traces.
- **Root cause of the observability hole (proven):** `_swr_refresh._run`
  did `except Exception: return None`. A background rebuild that *raises*
  (vs. merely slow) populated the cache **never**, recorded the exception
  **nowhere** (no log, no counter, no placeholder field), and every poll
  re-served the same opaque `{"warming":true}` **forever**. 16 endpoints
  are exposed to this. An operator could not distinguish "slow, will
  self-heal" from "raising every cycle, will NEVER self-heal".
- **Deliberately NOT claimed:** whether briefing's specific never-warming
  is a raising handler vs. chronic `>TTL` slowness vs. pool starvation was
  *not* isolated (a fresh standalone import of `dashboard` blocks >70s, so
  the handler could not be cleanly bench-called out-of-process). The
  diagnostic surface is useful in all three cases; this commit is framed
  as *"add the diagnostic surface"*, not *"fixed briefing"*.

**Built (this commit — `paper_trader/dashboard.py` +
`tests/test_swr_failure_observability.py`):**

1. **SWR failure observability.** `_swr_entry` carries
   `fail_count / last_error / last_error_ts / last_ok_ts`. `_run` on
   exception increments the consecutive-failure count, records
   `Type: msg` (≤200 chars), and prints a **throttled** `[swr]` stderr
   line — the 1st failure (early warning) and every
   `_SWR_FAIL_LOG_EVERY=10`th (sustained), never once-per-poll. A
   successful build **resets** the streak (a transient blip is not
   reported forever). The cold placeholder now carries
   `attempts / last_error / stale_for_s` — `attempts==0 & last_error==None`
   ⇒ *slow but healthy* ("be patient"); `attempts>0` ⇒ *raising, will not
   self-heal* (actionable). Purely additive to the `{"warming":true}` body
   (verified: no exact-keyset consumer in tests/ or the template); the
   happy path is byte-identical.
2. **`/api/scorer-confidence` is now `@swr_cached("scorer-confidence",
   90.0)`** (TTL matches briefing / sector-heatmap / correlation — the
   other expensive ones). A cold scorer replay can no longer wedge a Flask
   request thread; it returns the bounded warming placeholder and
   self-heals. SWR is pytest-inert, so the existing exact-value
   `test_scorer_honesty.py` path is unchanged.

**Known upstream follow-up (NOT addressed here — different change,
different risk; flagged not silently fixed):** `_SWR_EXEC` has
`max_workers=6` but **16** `@swr_cached` endpoints, all cold-fetched on a
single dashboard load → guaranteed queue thrash under the documented load
avg ~23. If briefing's never-warming is mostly this, `attempts` will
correctly read `0` ("slow, not broken") indefinitely and the panel still
stays blank — the *observability is honest*, but the real lever is pool
sizing / cold-fetch fan-out, not this commit. `_SWR_COLD_BUDGET_S` and
`max_workers` were **deliberately left unchanged** (tuned for current
load; bumping them is a separate, riskier change that would muddy this
one).

**Operator action required:** live `:8090` runs stale code (chronic — see
CLAUDE.md §11 / project memory). The new diagnostic is inert until
`systemctl --user daemon-reload && systemctl --user restart paper-trader`.

**Tests:** `+5` in `tests/test_swr_failure_observability.py` (raising →
`attempts`/`last_error` surfaced & growing; slow → `attempts==0`/no error;
success resets then a fresh failure restarts at 1; consecutive failure
logs `[swr]` to stderr; scorer-confidence stays swr-wrapped — TDD,
RED→GREEN confirmed). Full suite **1613 passed** (+5 net), zero
regressions; SWR-adjacent set (`test_dashboard_swr`, `test_core_state_swr`,
`test_decision_context_endpoint`, `test_scorer_honesty`,
`test_core_dashboard_bounded_net`) 31 passed.

*Feature-dev pass appended 2026-05-18. Prior content above is unmodified.*

---

### 2026-05-18 review pass #15 (ML+backtest hybrid · gate economic counterfactual · decisive news-feature-deadness finding)

- **Phase 1 — no new bugs (bugs_fixed = 0; no Phase-1 commit).** Full
  re-trace of `decision_scorer.py`, `backtest.py`,
  `run_continuous_backtests.py` plus coupled `validation.py` /
  `calibration.py` / `gate_audit.py`: the `predict_with_meta`
  off-distribution gate-abstention path, the universal SELL
  `-forward_return_5d` sign-flip (train↔inference↔calibration↔gate↔
  `_oos_rank_metrics`↔`evaluate_scorer_oos`), the `(ticker,sim_date,
  action)` dedup key (correctly includes `action` so a BUY/SELL pair on
  one name/day with opposite labels both survive), the 5-trading-day
  forward-window guard, the `score=`/`scorer=` first-match
  disambiguation, the numpy-lstsq fallback weighted-LS scaler, the
  unlocked `_VOLUME_CACHE` membership read (safe: GIL-atomic `in`/`[]`,
  nothing ever deletes — the AGENTS.md concurrency invariant is about
  *iteration*), the `train_scorer` 80/20 split-before-scale, every
  module-global lock — all re-verified correct and exact-value
  test-locked. The temporal-boundary duplicate-straddle in
  `split_outcomes_temporal` is the **already-documented**
  `OOS_NOT_HELD_OUT` corpus-construction limitation (corpus_audit
  verdict), not a surgical code bug — and per CLAUDE.md §6 the split
  mechanism is training-dynamics, out of scope. Consistent with the
  documented 13+ prior no-new-bug ML/backtest passes — not a fabricated
  fix. ML/backtest regression 255/255 green before the feature, 280/280
  after.

- **Feature shipped (commit `35479f5`): gate economic counterfactual.**
  `paper_trader/ml/gate_pnl.py`. The gap it fills: `gate_audit` reports
  each arm's mean realized return and a verdict driven **solely** by
  `strong_tailwind_mean − strong_headwind_mean` — by construction it
  ignores the three middle arms (`mild_headwind` ×0.85, `neutral` ×1.00,
  `mild_tailwind` ×1.15) and how *often* each arm fires. A gate can read
  `GATE_EFFECTIVE`/`GATE_INEFFECTIVE` on the two-extreme spread while the
  **portfolio-level** effect is entirely different, because most of the
  reweighting happens in the populous middle arms. This computes the
  single economic number a quant deciding *whether to keep the gate*
  actually needs: the **assumption-free** equal-weight contribution
  `Σmᵢrᵢ/Σmᵢ − mean(rᵢ)` (gate-on minus gate-off realized mean, every
  base bet held equal — no conviction reconstruction needed, since the
  gate only *resizes* trades `_ml_decide` already picked) on the
  temporal-OOS slice. A base-conviction-weighted `sized_*` number
  (reconstructing `_ml_decide`'s `min(cap, ml_score/divisor)` incl. the
  leveraged-ETF/regime branch) is reported **informationally only —
  never folded into the verdict** (the `gate_audit` arm-monotone honesty
  pattern), because `ml_score` is the reasoning's 2-dp `score=` and the
  bull-vs-"unknown" regime at `regime_mult==1.0` is irreducible from the
  outcome row (cross-checked live: reconstructed base ≠ the reasoning's
  post-gate `conviction=` precisely *because* the latter already carries
  the multiplier — the formula structure is right, the residual is the
  gate itself + 2-dp rounding). Reuses `gate_audit.gate_arm` and
  `validation.split_outcomes_temporal` (single source of truth — the
  arms / OOS slice can never drift between the two gate diagnostics).
  Read-only, no train/pickle/`build_features`/`N_FEATURES`/trade touch,
  never raises, CLI exits 2 on `GATE_SUBTRACTS_RETURN`. **NOT wired into
  `main()` — zero deploy-stale impact, no loop restart needed.** 25
  exact-value locks in `tests/test_gate_pnl.py` (full verdict matrix at
  hand-computed `±3.6842`/`0.0` contributions; the SELL-sign-flip
  regression — without it GATE_ADDS reads GATE_RETURN_NEUTRAL; exact
  `1.9310` sized contribution; `_reconstruct_base_conviction` cap/divisor
  /leveraged/regime branches; OOS-slice restriction; `gate_arm is
  gate_audit.gate_arm` SSOT; never-raises).
  ```bash
  cd /home/zeph/trading-intelligence/paper-trader && python3 -m paper_trader.ml.gate_pnl
  cd /home/zeph/trading-intelligence/paper-trader && python3 -m paper_trader.ml.gate_pnl --all
  cd /home/zeph/trading-intelligence/paper-trader && python3 -m pytest tests/test_gate_pnl.py -v
  ```

- **Quant finding (NEW, headline — the gate's economic impact is
  ~0pp, not the +0.86pp the extreme-arm spread suggests).** Live pickle
  `n_train=3870`, gate active. **OOS slice (1418 fills):
  `GATE_RETURN_NEUTRAL`, equal-weight contribution +0.02pp** (gate-on
  +0.55% vs gate-off +0.53%, avg multiplier 0.96). The sibling
  `gate_audit` on the *same* slice reads `GATE_INEFFECTIVE` with a
  `strong_tailwind − strong_headwind` spread of **+0.86pp** — close
  enough to the ±1.0pp tolerance to look marginal — but rolled up across
  all five arms weighted by fire-frequency (`mild_headwind` n=570 @
  +0.68%, `neutral` n=505 @ +0.18%, `mild_tailwind` n=161 @ +1.27%,
  `strong_tailwind` n=115 @ +0.71%, `strong_headwind` n=67 @ −0.16%) the
  net portfolio contribution is **+0.02pp ≈ 0**. This is the decisive
  economic statement of the documented near-zero OOS skill: the gate
  (invariant #5, `gate_active` every cycle) underwrites **pure sizing
  variance with no compensating realized edge** — now quantified in
  realized-return pp, not rank-IC. In-sample `--all` reads +0.39pp
  (still NEUTRAL); the in-sample→OOS collapse mirrors the textbook
  overfit every prior pass documents. Cross-tool consistency confirms no
  drift: `calibration --oos` MISCALIBRATED (spearman 0.012 vs in-sample
  0.36), `gate_audit` GATE_INEFFECTIVE, `scorer_skill_log.jsonl`
  `oos_ic ≈ 0`. Reported, **not actioned** — turning the gate off is a
  training-dynamics change out of surgical scope (CLAUDE.md §6).

- **Quant finding (NEW, decisive — 2 of the 17 scorer features are
  constant noise in training).** `decision_outcomes.jsonl` (7093 rows):
  **98.1% have `news_article_count = NULL`** → `news_urgency` /
  `news_article_count` sit at their `build_features` defaults (50.0 /
  1.0) for 98% of training rows. The continuous loop draws deep
  historical windows (current corpus sim_dates **1996–2018**) where
  `digital-intern/articles.db` has effectively zero coverage, so almost
  every backtest decision is pure-quant. ~12% of the MLP's input
  dimensionality is therefore a near-constant the network can only
  memorize around — a concrete mechanism contributing to the
  `baseline_compare` "the net destroys the signal it is fed" finding.
  Reported, not actioned (feeding news into deep-history backtests, or
  pruning the dead features, is an architecture/training-dynamics change
  out of surgical scope, CLAUDE.md §6).

- **Quant findings (corroborating, not new).** Training tail = **5
  distinct run_ids (6227–6232)** spanning sim_date 1996–2018 — exactly
  `corpus_audit`'s `OOS_NOT_HELD_OUT`/`SINGLE_DRAW` (the temporal-OOS
  holdout is the late slice of the same ~5 runs, not an unseen draw).
  `forward_return_5d`: mean +1.26%, std 7.14, p1 −18.53, p99 +21.68,
  **only 0.08% exceed |50%|** — re-confirms `PRED_CLAMP_PCT=50` is amply
  load-bearing-safe (tighter than the AGENTS.md ~0.4% on the older 9k
  corpus). Action mix BUY 5526 / SELL 1567. `forward_return_10d` present
  on 0/7093 rows — the multi-horizon capture is still uncommitted
  in-flight work; legacy rows have no 10d/20d keys, as documented.

- **Operational (reconfirmed, out of scope):** winner→ArticleNet
  feedback loop still dead both ways — `continuous.log`: `[continuous]
  ml: trainer rc=-15 injected=10000` and `inject err: database locked
  after 4 attempts`. Matches passes #6–#13 (digital-intern GPU +
  `articles.db` write contention on the `/media/...` symlinked volume) —
  the loop is **not** "training on its winners". The scorer itself
  retrains cleanly every cycle (`scorer ok` every cycle, train_n growing
  3234→3485→3870, `val_rmse ≪ oos_rmse`). `backtest.db.local_backup` is
  a stale 2026-05-17 snapshot (max complete run_id=5) — the live symlink
  still times out per pass #13; `continuous.log` is fresh & mid-cycle.

- **Run the ML/backtest suite (now 280):** `cd
  /home/zeph/trading-intelligence/paper-trader && python3 -m pytest
  tests/test_decision_scorer.py tests/test_backtest.py
  tests/test_calibration.py tests/test_validation.py tests/test_continuous.py
  tests/test_gate_audit.py tests/test_gate_pnl.py tests/test_skill_trend.py
  tests/test_baseline_compare.py tests/test_ml_backtest_review.py -q`
  (280 fast offline tests, green). `test_gate_pnl.py` holds the gate
  economic-counterfactual locks; it has none of "ml"/"backtest"/"scorer"
  in its node ids, so add it explicitly like `test_calibration.py` /
  `test_gate_audit.py`.

*Review pass #15 appended 2026-05-18. Prior content above is unmodified.*

---

### 2026-05-18 review pass #15 (paper-trader core hybrid · NYSE half-day close fix · deployable-cash prompt block · live findings)

*(Numbered #15 alongside the ML/backtest #15 above — the established
two-entries-per-number convention, e.g. the dual #11/#14 passes.)*

- **Phase 1 — 1 bug fixed (commit `e556606`).**
  `dashboard._next_market_open()` computed the "next close" as a hardcoded
  `now_ny.replace(hour=16, …)`. But `market.is_market_open()` has enforced
  13:00 ET **early-close half-days** since `b6a1934` (`NYSE_HALF_DAYS_2026`
  = day-after-Thanksgiving 2026-11-27, Christmas Eve 2026-12-24), exposing
  `market.close_minute(d)` (780 half-day / 960 regular, minutes past ET
  midnight). So on those two sessions — while `is_market_open` correctly
  returned True 09:30–13:00 — the `/api/briefing` card ("Market OPEN —
  closes in 5h00m", *the first thing a trader sees on the pane each
  morning*) and `/api/game-plan`'s `next_open_seconds` reported the close
  **3h late**, exactly the figure a trader times exits on. Fix: derive the
  close from `market.close_minute(now_ny.date())` (`divmod` → hour/minute);
  regular sessions byte-identical. Locked by 2 new tests in
  `tests/test_core_dashboard_helpers.py::TestNextMarketOpen`
  (half-day → 13:00/2h; regular-day → 16:00/5h no-perturbation). RED
  before the fix.

- **Phase 2 — 1 feature (commit `b739a14`).** `analytics/buying_power.py`
  + `build_buying_power` — a **deployable-cash advisory block in the live
  Opus prompt**, the lean prompt-facing complement to the dashboard-only
  `capital_paralysis`. The mirrors (`self_review`/`track_record`/
  `risk_mirror`) + `event_calendar` all reach the prompt; the one
  *operational* fact still omitted is what a desk checks before every
  order — how much can I deploy, and if pinned what unlocks me? This is
  the **#2 documented live pathology** (pass #14 #4): a $972 book with
  ~$18 free across two underwater names, where Opus saw only a raw
  `cash: $18.49` line. `/api/capital-paralysis` synthesises it on the
  **dashboard**, but the decision engine never saw it — the
  `event_calendar` gap, one dimension over. Pure arithmetic over the
  **already-marked snapshot + already-fetched `watch_px`** `decide()`
  holds (NO extra store read, NO network — the `risk_mirror` hot-path
  discipline), scoped to the same `_names_in_play` set the quant /
  track-record blocks use. States: `DEPLOYABLE` (affordable whole-share
  counts, ≤6 names), `CASH_CONSTRAINED` (below every in-play price → only
  fractional / SELL / HOLD actionable + the most-underwater position whose
  exit frees the most cash, the `capital_paralysis` "biggest-loser-first"
  cut-priority), `NO_PRICED_NAMES`/`NO_DATA`/`ERROR` honest fallbacks.
  Observational only — autonomy preamble, **no directive verb**, no cap,
  never gates (invariants #2/#12, the `event_calendar` precedent);
  `_safe`-wrapped so a fault is "no block this cycle", **never** "no
  decision". Wired into `_build_payload(... buying_power_block=)`
  (rendered **last in the advisory stack — after `event_calendar`, before
  `WATCHLIST PRICES`**) + `decide()` (`_safe` try/except, after the
  `event_calendar` block); applies on next paper-trader restart. **No
  parity endpoint deliberately** — `/api/capital-paralysis` already serves
  this concern on the dashboard, so a `/api/buying-power` twin would
  duplicate it and add a concurrent-edit surface to the contested
  `dashboard.py` for no operator gain. Smoke-tested live on the real
  pinned book: `CASH_CONSTRAINED · $18.49 free (98.1% deployed) · cheapest
  in-play SOXL @ $28 · most-underwater LITE ($-6.21) frees ≈$592`. Locked
  by `tests/test_buying_power.py` (17 tests: live pinned-book shape;
  strict `int(cash//px)` floor + `cash==price` boundary;
  zero/negative/None price excluded; not-in-play excluded; unlock
  loser-vs-largest-mark pick; `_position_mark_value` consumes the enriched
  `market_value` and never re-derives the option ×100; observational
  voice; `_build_payload` last-in-stack placement + `None`-no-stray;
  never-raises-on-garbage).

- **Phase 3 — live findings (1–5; none a quick safe code fix).**
  1. **`/api/liquidity` field/headline semantic inconsistency.** For the
     live $18.49 (1.9%) book the endpoint returns `can_act_on_signal:
     true` *next to* a headline reading "Pinned … **no room to act**".
     Root cause: `can_act = cash>=1.0 and cash_pct>=1.0` but the
     `NO_DRY_POWDER`/"no room to act" headline triggers at `cash_pct<2.0`
     — the thresholds disagree in the `[1%,2%)` band, exactly where the
     live book sits. Each number is individually correct (a fractional
     order *is* possible at $18); only the prose overstates. **Reported,
     not fixed:** `liquidity.py` is a deliberately-designed,
     heavily-tested builder `capital_paralysis` composes verbatim
     (single-source-of-truth) and sibling agents are actively editing —
     churning its field semantics for a wording nit risks the composition
     + a merge collision (the "deliberately weird, leave it" category). A
     future pass that *does* touch it should align the two thresholds (or
     soften the headline to "minimal room") and re-pin the
     `capital_paralysis` composition.
  2. **NO_DECISION ~59% lifetime / 60% (24h)** — confirmed unchanged,
     uniformly `"claude returned no response (timeout/empty)"` under host
     load ~17 (`/api/decision-health`: `no_decision_rate_24h 60.3`,
     `last_fill_ts` 23.7h ago). NOT a parser bug, NOT quota
     (`quota_exhausted` unset) — the long-standing contention diagnosis
     (CLAUDE.md §11; passes #6–#14). The lever is host load, not code; the
     Phase-2 block does not fix timeouts but makes the cycles that *do*
     complete materially more decision-useful on the pinned book.
  3. **Capital paralysis active & bleeding alpha (the Phase-2
     justification).** `/api/capital-paralysis`: 98.1% deployed, "inaction
     has cost **-2.21% alpha** (6 paralysis drought(s))",
     `cycles_since_last_fill 55`, last fill ~24h ago, LITE 60.9% of book.
     Real, ongoing, measurably costly.
  4. **`/api/build-info` `behind:1 stale:true`** at session start, but the
     sole missing commit was docs-only (a pass-#14 `AGENTS.md` entry); the
     git-watcher auto-fast-forwards (observed in `runner.log`:
     "fast-forwarding your working tree from commit 3b09f87"). Self-healing
     — no action. The runner-restart churn ("another paper trader is
     already running … exiting") is the singleton flock **working** during
     the concurrent-deploy storm (pass #14 #3.2) — noisy, safe.
  5. **Positives verified:** `/` 200 in **38 ms**; `/api/state` SWR-served
     (`cached:true`, age 34s — by design) with the full correct shape;
     `runner-heartbeat` HEALTHY, singleton **acquired** (not degraded),
     Discord delivery **HEALTHY**; `/api/feed-health` HEALTHY (566 live
     articles/2h, not split-brain); decisions on cadence (last 79s ago).
     The system is operationally sound; its two real problems (host-load
     timeouts, data-corruption paralysis) are documented ops/data issues,
     not core code defects.

- **Concurrency note for the next agent.** This pass ran with ≥3 sibling
  agents committing in parallel (observed: `reporter.py`, the
  `analytics_api` `mark_trust` block in `dashboard.py`, `/api/supervision`,
  `feat(ml) gate economic counterfactual`, a parallel `AGENTS.md` #15
  append). `git add <file>` restages the **whole** working tree — it
  silently captures a sibling's in-progress hunk. The safe pattern used
  here: extract only your own hunk (`git diff` → filter → `git apply
  --cached --recount`) for any file a sibling also touched (`dashboard.py`,
  `AGENTS.md`), and only `git add` whole files exclusively yours (new
  modules, new test files, `strategy.py` here). Verify with `git diff
  --cached -- <file> | grep -c <sibling-token>` == 0 before every commit.

- **Run the suite:** `cd /home/zeph/trading-intelligence/paper-trader &&
  python3 -m pytest tests/ -v` (full). Fast subset for this pass:
  `python3 -m pytest tests/test_buying_power.py
  tests/test_core_dashboard_helpers.py tests/test_core_strategy.py
  tests/test_event_calendar.py -q`.

*Review pass #15 (paper-trader core hybrid) appended 2026-05-18. Prior content above is unmodified.*

---

### 2026-05-18 feature-dev pass (Agent 4) — live-book SECTOR concentration in the decision prompt

- **1 feature.** `paper_trader/analytics/sector_exposure.py` +
  `build_sector_exposure` — **the live book's sector concentration + the
  marginal in-play sector impact, fed into the live Opus decision prompt.**
  `risk_mirror` (pass 2026-05-17) closed *name*-level concentration (top
  weight / HHI by ticker). The book's documented **#3 pathology is exactly
  one dimension over** — *sector* clustering: `risk_mirror.py`'s own
  docstring names it ("the book 60.9% in one name's **sector** … the
  dashboard already exposes both … but the decision engine itself never saw
  them"). `/api/analytics` computes `sector_exposure_pct` and `/api/risk`
  per-position sector, but **the decision path had zero sector awareness**
  (`grep sector paper_trader/strategy.py` prompt path → 0 hits). The marginal
  question a desk checks before every order — *does this trade pile onto my
  single most concentrated sector?* — was invisible at decision time. This is
  the lean, prompt-facing complement to the dashboard-only sector breakdown,
  the same gap `risk_mirror`/`event_calendar`/`buying_power` each closed one
  dimension over. Smoke on the documented ~$973 book: `CONCENTRATED · top
  OPTICAL 60.7% · HHI 0.46 · 3 sector(s)`, and the marginal line correctly
  flags `LITE→OPTICAL (60.7% — your heaviest sector)` while tagging
  `TQQQ→BROAD_LEV (0.0% — diversifying)`.

- **Single source of truth.** The book-sector % mirrors `dashboard.py`'s
  `analytics_api` formula **verbatim** (`price = current_price or avg_cost;
  val = price*qty*(100 if option else 1); pct = val/total*100`, classified by
  `SECTOR_MAP`), so `/api/sector-exposure` is *numerically identical* to
  `/api/analytics` `sector_exposure_pct` for the same store. `SECTOR_MAP` /
  `classify` are a **test-pinned verbatim copy** of
  `dashboard.SECTOR_MAP`/`_classify` — duplicated **deliberately** (the
  `strategy._ml_live_opinion` precedent: importing the ~9k-line Flask
  `dashboard` onto the live decision hot path is a fragility a `_safe`
  wrapper should never have to catch, and a sibling edit that broke that
  import would silently re-blind the desk; the existing test suite already
  imports `dashboard` universally, so the drift test pays no *new* Flask
  cost). `tests/test_sector_exposure.py::TestDriftLocks` asserts byte-equality
  with `dashboard.SECTOR_MAP`, that `classify == dashboard._classify`, and
  that `SECTOR_HEAVY_PCT == game_plan._SECTOR_HEAVY_PCT == 60.0` — any drift
  fails CI. (Distinct from `buying_power`, which matches
  `/api/capital-paralysis` and prefers enriched `market_value`; this matches
  `/api/analytics`, a different SSoT — keeping the formula identical is what
  makes the parity test exact.)

- **Observational only, never gates** (invariants #2/#12 — the
  `risk_mirror`/`buying_power` contract). The preamble disclaims directive/
  limit and reaffirms full autonomy; the block states facts (per-sector %,
  sector-HHI + label, which in-play names sit in an already-heavy sector) and
  issues **no fabricated fill-size projection** (Opus chooses size — the
  honest deterministic fact is "MU is SEMIS, SEMIS is already 61% of your
  book", not an invented "would take 61%→73%"). States `NO_DATA`
  (no priced book — the `buying_power` fallback) → `DIVERSIFIED` →
  `CONCENTRATED` (top sector ≥ the drift-locked 60.0% heavy mark). Pure,
  deterministic, never raises (the `_safe` contract; the `decide()` caller
  also wraps it → a fault is "no sector block this cycle", never "no
  decision this cycle").

- **Wiring.** `decide()` (try/except, after `risk_mirror`, before
  `event_calendar`) + `_build_payload(... sector_exposure_block=)` rendered
  **immediately after `risk_section`, before `event_section`** (structural
  risk by name → by sector → then what is *coming*). Scoped to the same lean
  `_names_in_play(positions, merged, WATCHLIST)` set the quant /
  track-record / buying-power blocks use (the marginal view matches "what
  matters this cycle"). Served at **`/api/sector-exposure`** (prompt↔endpoint
  parity — `/api/analytics` and `/api/risk` left untouched, different
  concerns, already tested).

- **Tests — 24, all green; full suite 1640 passed, 0 failed, 0
  regressions.** `tests/test_sector_exposure.py` (22): SECTOR_MAP /
  threshold / classify drift-locks; **every WATCHLIST ticker is classified**
  (a future watchlist add missing a SECTOR_MAP entry fails here, not silently
  becomes "% other"); hand-computed exposure %, top-sector, and sector-HHI
  (0.4321 on a known book); option ×100; avg_cost fallback; CONCENTRATED
  flips exactly at 60.0 (>=, not 60.01); deterministic tie-break (top ==
  breakdown[0]); **parity** (builder `sector_pct` == an independent
  `analytics_api`-formula recompute); marginal heavy/diversifying flags +
  riskiest-first sort; NO_DATA / None-snapshot; never-raises-on-garbage;
  observational voice (no directive verb, autonomy preamble); `_build_payload`
  wiring + None-renders-nothing + after-risk-mirror placement.
  `tests/test_sector_exposure_endpoint.py` (2): the real `/api/sector-exposure`
  Flask view on a fresh temp `Store` returns the expected concentrated shape,
  **and `/api/sector-exposure` `sector_pct` == `/api/analytics`
  `sector_exposure_pct`** end-to-end (the SSoT promise proven through the app,
  not a `__main__` smoke — the paper-trader-analytics-verification note).

- **Deploy caveat (the chronic-stale pattern).** The live trader runs many
  commits behind until a manual restart (CLAUDE.md / passes #6–#15); this
  feature is **committed but inert until the next paper-trader restart** —
  `/api/build-info` will read `behind`/`stale` until then. Not restarted here
  (documented dual-systemd-scope footgun). "Shipped" ≠ "deployed".

- **Concurrency.** Ran with ≥3 sibling agents committing in parallel (HEAD
  moved `f29e134`→`5f40009` mid-pass; sibling-dirty `reporter.py` /
  `test_core_reporter.py` / `test_runner_heartbeat.py` and untracked
  `game_plan.py` / `gate_pnl.py` / … are **not mine — never staged**). New
  module + 2 new test files + the brainstorm doc are exclusively mine
  (`git add` whole). `strategy.py` / `dashboard.py` / `AGENTS.md` are
  contested → only my own hunks staged by path, `git diff --cached` verified
  to contain zero sibling tokens before commit. Brainstorm/decision recorded
  in `docs/feature-dev-sector-exposure-2026-05-18.md`.

*Feature-dev pass appended 2026-05-18 (Agent 4). Prior content above is unmodified.*
