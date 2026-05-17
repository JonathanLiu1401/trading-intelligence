# AGENTS.md — paper-trader

Companion to `CLAUDE.md` aimed at coding agents that touch this repo
during automated review / fix cycles. Where `CLAUDE.md` documents the
*system*, this file documents the *workflows*.

## Repository layout (quick reference)

- `paper_trader/runner.py` — live trader main loop
- `paper_trader/strategy.py` — live Opus decision engine + watchlist (now injects the behavioural self-review mirror into the prompt)
- `paper_trader/analytics/self_review.py` — canonical behavioural mirror; composes trade_asymmetry + capital_paralysis + open_attribution, fed into the live prompt **and** served at `/api/self-review`
- `paper_trader/signals.py` — live news signal queries against digital-intern's articles.db
- `paper_trader/market.py` — yfinance wrapper + NYSE session calendar
- `paper_trader/store.py` — SQLite store (portfolio, trades, positions, decisions, equity_curve)
- `paper_trader/reporter.py` — Discord output via openclaw. `send_hourly_summary` / `send_daily_close` now append `_behavioural_block()` — the `build_trader_scorecard` verdict-alignment synthesis composed **verbatim** (single source of truth, invariant #10; same store reads as `/api/scorecard`) so the operator who lives in Discord sees the ~24 builders' synthesis without opening the (stale) dashboard. Observational only, no caps (invariants #2/#12 — the `self_review`/`scorecard` precedent). NO_DATA/ERROR suppressed; a builder/store fault degrades to *no block*, **never** *no summary* (the reporter failure contract). Applies on next paper-trader restart (the documented pattern for every recent feature)
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
| `test_core_runner.py` | `_maybe_daily_close` weekend/time gating + once-per-day flag + retry-on-failure, `_maybe_hourly` 3600s gating + retry-on-failure |
| `test_core_runner_cycle.py` | **`_cycle()` report-dispatch fan-out** — previously **zero** direct coverage despite real branching: FILLED gates BOTH trade-alert AND decision-log; HOLD/NO_DECISION/BLOCKED/missing-`status` stay silent **and never query the store** (outer-guard short-circuit asserted via a recording `_FakeStore`); `auto_exits` is an orthogonal `_send` channel independent of the FILLED gate (dead-today-on-purpose per invariant #12 — locked so re-enabling is deliberate, kept per the "do not delete as unreachable" note); the `if trades and status==FILLED` guard (empty `recent_trades(1)` → no alert but decision-log still fires); every reporter fault swallowed (daemon-loop survival, via `monkeypatch` so `boom` can't leak into other modules' reporter import) |
| `test_core_reporter.py` | openclaw missing → False, timeout/nonzero exit → False, trade alert + decision log + portfolio line formatting, **daily-close P/L baseline label tracks `_INITIAL_EQUITY` not a hardcoded `$1000`**, **`send_daily_close` `pnl_real` cash-flow sign (SELL\* credits / BUY\* debits) incl. the option ×100 multiplier via `store.record_trade`** (exact `$-400.00` on a mixed stock+option same-day ledger — a sign flip → `+400.00`, a dropped ×100 → `-449.50`), **`_behavioural_block` composes the scorecard state/headline/focus/concordance verbatim** (no re-derived verdict), suppresses NO_DATA, **returns `""` (never raises) when the builder faults — and `send_hourly_summary`/`send_daily_close` still send the summary regardless** (the "no block, never no summary" failure contract) |
| `test_round_trips.py` | `build_round_trips` arithmetic: simple/partial/re-entry round-trips, option ×100, distinct (ticker,type,strike,expiry) keys, open-lot exclusion, orphan SELL, zero-cost `pnl_pct=None`, negative/unparseable `hold_days`, sub-cent rounding |
| `test_core_analytics.py` | `/api/analytics` end-to-end via Flask test client: exact `win_rate_pct` / `profit_factor` / `avg_holding_days` / `realized_pl_usd` / `n_round_trips` for a fixed ledger; open positions excluded; empty ledger → null metrics |
| `test_core_dashboard_helpers.py` | Pure dashboard helpers with no prior coverage: `_scorer_verdict` 5-way boundary bucketing; `_position_ages_from_trades` open-lot state machine (partial-sell keeps entry, full-sell→re-buy resets, option trades ignored); `_next_market_open` open/close/weekend/holiday arithmetic; `_classify_action` co-pilot selection incl. the **EXIT-before-TRIM** ordering regression and "never BUY without a technical confirm"; **`TestTemplateIdsUnique` — no duplicate static `id="..."` in `dashboard.TEMPLATE`** (regression lock for the `dd-`/`drought-` card-id collision, invariant #14) |
| `test_decision_drought.py` | `build_decision_drought` segmentation: `_classify` fill/block/hold/no-decision; two-drought scenario with exact portfolio/SPY/alpha %; PARALYSIS vs DELIBERATE_HOLD split; ongoing drought detection; `involuntary_alpha_bleed_pct` counts PARALYSIS-only negative alpha; min-reportable-cycles filter; NEVER_TRADED / NO_DATA verdicts; alpha=None when SPY missing |
| `test_news_edge.py` | `build_news_edge`: `_index_at_or_after` exact/gap/overflow; EDGE_CONFIRMED with exact raw means; **SPY-abnormal subtraction is applied** (raw 2.0, spy +1.0 → abnormal 1.0); NO_EDGE on a falling top-band ticker; INSUFFICIENT_DATA under `_MIN_BAND_N`; `$TK`/word-boundary resolution incl. "AMDOCS" must not match AMD; **adaptive reference horizon degrades to 1d when only a 1d forward window exists** (the live-data early-history case) |
| `test_signal_followthrough.py` | `build_signal_followthrough`: exact-value EXPLOITING (acted NVDA+ beats ignored AMD-flat, `selection_edge`/follow-through/per-horizon means) / MISUSING (mirror image, negative edge) / IGNORING_FEED (0% follow-through, ignored-bucket numerics still emitted); **SPY-abnormal subtraction applied** (raw +10 → +8.75 abnormal at 5d under SPY +1/day); per-(decision,ticker) dedup (3 NVDA articles in one window → 1 signal); window boundary (future/stale news excluded); AMDOCS must not match AMD; sample-size honesty (`INSUFFICIENT` keeps numerics, empty → `NO_DATA`); `_fetch_live_articles` excludes planted `backtest://`/`backtest_*`/`opus_annotation*` rows |
| `test_churn.py` | `build_churn`: `NO_DATA`/`EMERGING`/`STABLE` sample-size gate; exact re-entry detection incl. the live NVDA close→re-buy shape (gap_days, `prior_pnl_usd` consumed from `build_round_trips` not recomputed); `REENTRY_WINDOW_DAYS` boundary inclusive **and** one-second-past exclusive; distinct-names→zero re-entries; `reentry_events` sorted fastest-first; both CHURNING paths (≥25% re-entry rate, and fast-cadence with zero re-entries); BUY_AND_HOLD; ACTIVE_TURNOVER between the lines; sub-day loss-concentration exact (= round-trips' own negative-`pnl_usd` sum, single source of truth #10); zero-span book → cadence `None` (no divide-by-zero); all-winners → concentration `None` |
| `test_thesis_drift.py` | `build_thesis_drift`: `NO_DATA` empty; INTACT when up & signals benign; BROKEN via −8% pain line regardless of signals **and** via MACD-flip+negative-mom+loss; WEAKENING via soft −3% loss (no signals), hot RSI while green, cold-catalyst heuristic; **opener selection nearest `opened_at` picks the re-entry lot's BUY not the prior closed lot's** (invariant #8); entry reason surfaced **verbatim** (long string equality); missing ledger → reason `None`, `entry_price` falls back to `avg_cost`, no error; cards sorted worst-first with exact counts |
| `test_loser_autopsy.py` | `build_loser_autopsy`: `_classify` failure-mode precedence (KNIFE_CATCH wins over the fast/shallow WHIPSAW arm, `< FAST_HOLD_DAYS` strict & `>= SLOW_HOLD_DAYS` inclusive boundaries, `None` hold/pnl_pct never raises and defaults); strict `pnl_usd < 0` loser convention (a `pnl==0` wash is **not** a loss — invariant #10); verbatim entry/exit reason joined by trade `id` (first BUY / last SELL; blank/whitespace → `None`, missing-id → `None`, never NLP-parsed); aggregates exact (total/avg, median odd **and** even count, ticker-bleed sorted most-negative-$ first, `repeat_offenders` n≥2, deterministic dominant-mode severity tie-break); P&L/cost/proceeds **consumed from `build_round_trips`** on a partial-then-full close (not recomputed); verdict withheld until `STABLE` (n_losers≥`STABLE_MIN_LOSERS`); NO_DATA/NO_LOSSES/EMERGING honesty; never raises on garbage rows |
| `test_correlation.py` | `build_correlation`: `_returns` chain (a `0`/NaN/non-numeric bar **breaks then continues** — one bad yfinance bar must not zero the series; `pytest.approx` for the float-division results); `_pearson` exact `±1.0` under a positive/negative affine map, the hand-computed `0.6` fixture, flat-series → `None` (never a fabricated 0), length-mismatch/too-short → `None`; options flagged & skipped; single-name **and** sub-`MIN_RETURNS` series → `INSUFFICIENT` (verdict withheld, numerics where possible); `CONCENTRATED` (identical returns ρ=+1 → `effective_independent_bets`=1.0) / `DIVERSIFIED` (ρ=−1 → eff_bets `None` honest-undefined; constructed ρ=0 → eff_bets 2.0) / `SINGLE_NAME_RISK` overrides correlation when top weight ≥ `DOMINANT_WEIGHT` / `MODERATE` band; `weight_hhi` & `effective_positions_naive` exact (60/40 → HHI 0.52); unequal-length series aligned to the common tail; never raises on garbage |

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

7. **`paper_trader.db` uses WAL** — any external reader must use
   `PRAGMA journal_mode=WAL` or open the file as `file:...?mode=ro` to avoid
   lock contention with the live writer.
   *Known concern (not fixed here — too invasive for a surgical pass):* the
   in-process Flask dashboard runs in a daemon thread but shares the **same**
   `Store` singleton (and thus the same `sqlite3.Connection`,
   `check_same_thread=False`) as the runner. Writes are serialized by
   `Store._lock`; reads (`get_portfolio`, `recent_trades`, `open_positions`,
   `recent_decisions`, `equity_curve`) are **not**. Concurrent dashboard reads
   during a runner write are tolerated by WAL but are not strictly
   connection-safe. A proper fix would give the dashboard its own read-only
   connection rather than reworking locking on the live writer.

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

### Dashboard API endpoints (port 8090)

All endpoints serve `application/json`. CORS is wide open (`*`) so the
Digital Intern dashboard on `:8080` can cross-fetch.

| Endpoint | Purpose |
|----------|---------|
| `GET /` | HTML — live trader page (portfolio + trades + chart) |
| `GET /backtests` | HTML — backtest grid + equity overlay |
| `GET /api/state` | Portfolio + positions + last 40 trades + last 20 decisions + equity curve |
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
| `GET /api/earnings-risk` | Upcoming earnings ⨯ held positions / watchlist, tiered |
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
| `GET /api/thesis-drift` | **Is the reason each position was opened for still true?** — the one discipline question no panel answered. `/api/position-thesis` fuses *current* scorer+technicals+news; `/api/suggestions` re-derives an action from scratch. Neither re-tests a holding against **its own opening rationale**, which is sitting verbatim in the opening fill's `trades.reason`. Per open position: selects the opening BUY as the one whose timestamp is **nearest `opened_at`** (invariant #8 — `opened_at` is reset to the re-entry time on a reopened lot, so the nearest BUY is *this* lot's opener, not a prior closed lot's; ties→earliest), surfaces that reason **verbatim** (never NLP-parsed for trading logic — the lone heuristic that reads it is an explicitly-labelled "entry cited a news catalyst, none live now" note), and assigns `health` ∈ `INTACT`/`WEAKENING`/`BROKEN` from **objective deterministic inputs only**: P/L since entry vs `PAIN_PCT`=−8% / `WEAK_PCT`=−3%, plus (when the endpoint supplies live quant/news) MACD flip + negative 5d momentum + `RSI_HOT`=78 + news-gone-cold. Precedence BROKEN>WEAKENING>INTACT; cards sorted worst-first (BROKEN, then most-negative P/L). The endpoint feeds `signals` by reusing `strategy.get_quant_signals_live` + `_ticker_news_pulse` (the exact `/api/suggestions` sources — no re-derivation); a signals failure degrades to **price-only health, never an error**. `state` = `NO_DATA` (no open positions) / `OK`. Pure, network-free *builder* (the network lives in the endpoint, builder takes the dicts) — advisory only, never gates Opus, adds no caps (invariants #2/#12). Pure core: `analytics/thesis_drift.py::build_thesis_drift`. Locked by `tests/test_thesis_drift.py` (BROKEN via pain line / via MACD-flip+mom+loss, WEAKENING via soft loss / hot RSI / cold-catalyst, opener-nearest-`opened_at` on a re-entered lot, verbatim-reason preservation, missing-ledger degrade, worst-first sort). **UI:** `tdrift-card` panel on the `:8090` trader page; JS degrades via the `/api/build-info` `stale` contract |
| `GET /api/loser-autopsy` | **Per-closed-losing-round-trip post-mortem — *why each closed trade lost*.** The neighbours each see a different slice: `/api/thesis-drift` re-tests **open** positions against their opening rationale; `/api/trade-asymmetry` is **aggregate** payoff math (one number for the whole book); `/api/churn` counts re-entry **cadence**. None narrate the individual loss. Composes the single source of truth (`build_round_trips`, invariant #10 — **no re-derived P&L/hold**), joins the **verbatim** opening-fill thesis and closing-fill reason back from the contributing `trades.reason` rows by their DB `id` (the `thesis_drift` "surface verbatim, never NLP-parse for trading logic" discipline), and assigns an objective, documented failure mode per loser — `KNIFE_CATCH` (loss ≤ `BIG_LOSS_PCT`=−15%, precedence-first: the thesis was badly wrong) / `WHIPSAW` (closed < `FAST_HOLD_DAYS`=1d at a shallow > −3% loss) / `SLOW_BLEED` (held ≥ `SLOW_HOLD_DAYS`=5d and still red — the disposition behaviour `trade_asymmetry` aggregates, surfaced per-trade) / `STOPPED_OUT` (else). Rolls up *which name is the bleed* (`ticker_breakdown`, most-negative-$ first), *which mode dominates* (deterministic count then a fixed severity tie-break so the verdict never flips on dict order), and *which losing names recur* (`repeat_offenders`, n≥2 — distinct from `churn`'s re-entry-cadence framing). Strict `pnl_usd<0` loser convention (a sub-cent wash reads as a non-loss, matching `round_trips`/`trade_asymmetry`, #10). Sample-size honesty mirrors `trade_asymmetry`: per-loser cards + numerics emit from the first loss but the **pattern verdict is withheld until `STABLE`** (`n_losers ≥ STABLE_MIN_LOSERS`=8) — `NO_DATA`→`NO_LOSSES`→`EMERGING`→`STABLE`. Advisory only — never gates Opus, adds no caps (invariants #2/#12). Pure core: `analytics/loser_autopsy.py::build_loser_autopsy` (never raises — malformed rows degrade, never except). Locked by `tests/test_loser_autopsy.py`. **UI:** `lautopsy-card` panel on the `:8090` trader page (fresh id prefix per invariant #14; table built via DOM `textContent`, never `innerHTML`, so a verbatim reason can't inject markup); JS degrades via the `/api/build-info` `stale` contract |
| `GET /api/correlation` | **Concentration honesty — do the held names actually move *together*?** `/api/risk` reports **name-level** concentration (`concentration_top1_pct`/`top3_pct`) and a single 3% SPY-shock; it cannot see **factor** concentration — a "2-position 59/41" book reads as merely concentrated, but if both names co-move the operator is running a *single bet* and the SPY-shock understates the tail. Computes pairwise Pearson **return** correlation among the held **stock** positions (deterministic ticker-sorted pairs; a flat series → `None`, never a fabricated 0), the most-coupled pair, the weight-Herfindahl `effective_positions_naive` (1/HHI), and the **correlation-adjusted `effective_independent_bets`** = `n / (1 + (n−1)·mean_ρ)` clamped to [1, n] — which collapses toward 1 as the names co-move however many tickers are on the book (mean ρ=−1 with n=2 → denominator 0 → honest `None`, never a fabricated number). Options are flagged & skipped (correlating a Greeks payoff against a linear return is meaningless — the `open_attribution`/`/api/backtests/compare` "stocks only" carve-out, #10 spirit). **The builder is pure; the yfinance daily-bar fetch lives in the endpoint** via the shared `_daily_history_cached` (3mo, the existing 30-min `_NEWS_EDGE_PX_CACHE`) — exactly the `thesis_drift` "network in the endpoint, builder takes the dicts" split, so the core is offline & deterministically testable and a fetch failure degrades to `INSUFFICIENT`, never an error. Sample-size honesty mirrors `news_edge`/`trade_asymmetry`: `NO_DATA` (no stock positions) → `INSUFFICIENT` (<2 correlatable names, or series < `MIN_RETURNS`=10 aligned daily returns — numerics where computable, verdict withheld) → `OK` with verdict precedence `SINGLE_NAME_RISK` (top weight ≥ `DOMINANT_WEIGHT`=60% — single-name risk reads first, correlation is secondary) > `CONCENTRATED` (mean ρ ≥ `HIGH_CORR`=0.70 — the book moves as one) > `MODERATE` (≥ `MOD_CORR`=0.40) > `DIVERSIFIED`. Pairs are measured over a **common aligned tail** so every ρ uses the same window. Advisory only — never gates Opus, adds no caps (invariants #2/#12). Pure core: `analytics/correlation.py::build_correlation` (never raises). Locked by `tests/test_correlation.py`. **UI:** `pcorr-card` panel on the `:8090` trader page (fresh id prefix per invariant #14); JS degrades via the `/api/build-info` `stale` contract |

### Common failure modes (live trader)

| Symptom | Likely cause | Where to look |
|---------|--------------|---------------|
| Loop posts `NO_DECISION` every cycle | Claude returned malformed JSON or timed out (`DECISION_TIMEOUT_S=120`) | `strategy.py::_parse_decision`; tail runner stdout for `[strategy] claude err:` |
| Live trader stuck on `BLOCKED` for a SELL | `_enforce_risk_pre_trade` rejected — qty > held, or option `strike+expiry` unspecified with multiple open legs | `strategy.py::_enforce_risk_pre_trade`, `_execute` (option ambiguity check) |
| Hourly summary never posts | `_maybe_hourly` only advances on send success; openclaw missing → permanent retry-loop with stdout log | Search runner stdout for `[reporter] openclaw not installed` |
| `signals.get_top_signals` returns `[]` | `articles.db` not at `USB_DB` (USB unmounted) or `LOCAL_DB`; live-only filter is correct so backtest contamination is *not* the cause | `signals._db_path()`; run `python3 -m paper_trader.signals` |
| `paper_trader.db is locked` | Another writer attached without `?mode=ro`; or a long-running query inside `_lock` | Check for ad-hoc scripts; only the runner should write |
| Dashboard `/api/scorer-predictions` shows `is_trained: false` | `data/decision_outcomes.jsonl` has < 500 rows — scorer hasn't trained enough yet | `wc -l data/decision_outcomes.jsonl` |
| Discord posts stop entirely | `openclaw` binary missing / auth expired | `which openclaw`; `openclaw message send --channel discord ...` manually |
| Live cross-dashboard (`:8080` → `:8090`) shows blanks | CORS or paper-trader process down | `curl http://localhost:8090/api/portfolio` |
| Strategy returns `HOLD` constantly even with strong signals | Opus is being conservative — by design, no threshold gating to override | Inspect the prompt context in `strategy.py::_build_payload`; if the watchlist has stale prices yfinance is rate-limited |
| Equity / P/L looks too high and won't come down; an option position never closes | Pre-fix `_portfolio_snapshot` marked an expired contract at avg_cost forever (no live chain past expiry). Fixed — see invariant #13. If you see this on an old `:8090` process, check `/api/build-info` `stale` and restart | `strategy._option_expired` / `_expired_intrinsic`; `SELECT * FROM positions WHERE type IN ('call','put') AND closed_at IS NULL AND expiry < date('now')` |

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

### Tests (ML + backtest section)

```bash
# ML + backtest only
cd /home/zeph/paper-trader && python3 -m pytest tests/ -v -k "ml or backtest or scorer"

# Core (live trader) only
cd /home/zeph/paper-trader && python3 -m pytest tests/test_core_*.py -v

# Full suite
cd /home/zeph/paper-trader && python3 -m pytest tests/ -v

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
`train_scorer`, prediction clamp / honesty), `test_continuous.py`
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

### When to bump model versions

The scorer model has no explicit version field. Treat a change to
`N_FEATURES`, `SECTORS`, or `build_features` parameter signature as a
breaking change: delete `data/ml/decision_scorer.pkl` and let the next
continuous cycle retrain from `data/decision_outcomes.jsonl`. The pickle
auto-recreates atomically (`.pkl.tmp` → `replace`) so a fresh-start
deletion is safe even if a backtest thread is mid-read.
