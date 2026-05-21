# AGENTS.md — paper-trader

Companion to `CLAUDE.md` aimed at coding agents that touch this repo
during automated review / fix cycles. Where `CLAUDE.md` documents the
*system*, this file documents the *workflows*.

## Repository layout (quick reference)

- `paper_trader/runner.py` — live trader main loop. **Single-instance guard** (`_acquire_singleton_lock`, `fcntl.flock` on `data/paper_trader.runner.lock`, invariant #19) refuses to boot a second trader on the same paper book — `main()` exits before the store/dashboard/ONLINE-ping when the lock is held by a live process; fails OPEN (degraded → continue) if the lock plumbing is unusable. Auto-recovery circuit breaker scoped to its own child claude subprocesses (`pkill -P os.getpid()`, invariant #18). Hourly / daily-close markers are restart-durable via the atomic `data/runner_state.json` sidecar (invariant #6)
- `paper_trader/strategy.py` — live Opus decision engine + watchlist (now injects the behavioural self-review mirror into the prompt). `_portfolio_snapshot` emits `stale_mark` per position and `_build_payload` annotates a `[STALE MARK …]` suffix so a missing-price mark (`current_price==avg_cost`, P/L $0.00) is not read by Opus as a genuine flat position (commit `f834c93`, review pass #4; advisory only, invariants #2/#12)
- `paper_trader/analytics/self_review.py` — canonical behavioural mirror; composes trade_asymmetry + capital_paralysis + open_attribution, fed into the live prompt **and** served at `/api/self-review`
- `paper_trader/analytics/trade_attribution.py` — per-fill news-to-trade attribution: *which articles plausibly preceded each recent FILLED trade?* Every other news↔trade panel (`news_edge`, `source_edge`, `signal_followthrough`, `scorer_attribution`) answers the symmetric *across-the-book* question; this is the missing per-trade audit — when the bot bought NVDA at 14:23, which highest-scored live-only articles in the prior 4h mentioned NVDA? *Implied* attribution (the literal prompt context isn't stored row-by-row), but deterministic and SSOT-clean: matches by case-insensitive **word-boundary** regex (so `MU` does not alias `MUTUAL`; `$NVDA` cashtag still hits) inside `[trade − window, trade]`, sorts by `ai_score` DESC with **closer-to-fill tie-break** (more plausibly causal). `build_trade_attribution(trades, articles, window_hours=4, max_per_trade=3, min_ai_score=2)` is pure / never raises (the `correlation`/`thesis_drift` split — endpoint owns the I/O; builder is offline & testable). **Live-only by construction** (invariant #1): the endpoint applies the canonical `url NOT LIKE 'backtest://%' AND source NOT LIKE 'backtest_%' AND source NOT LIKE 'opus_annotation%'` SQL fragment when reading `articles.db` so synthetic rows never reach the panel. **Pseudo-tickers dropped** (`CASH`/`NONE`/`NO_DECISION`/`BLOCKED`/blank — the `_parse_action_ticker` carve-out, invariant #11). **No fabrication**: a trade with zero matching articles surfaces `n_attributed: 0` with an honest "no live-only article mentioning <ticker> in the Nh before the fill" headline — the `recovery`/`loser_autopsy` negative-space-is-data precedent. Served at `/api/trade-attribution` (query params: `hours_back` 1..168, `window_hours` 0.5..24, `max_per_trade` 1..10, `min_ai_score` 0..10); SWR-cached 60s like every news-IO endpoint. Observational only, never gates, no caps (invariants #2/#12). Locked by `tests/test_trade_attribution.py` (23 tests: NO_DATA/OK ladder, pre-trade-window strict inequality both sides, MU/MUTUAL substring discriminator, `$NVDA` cashtag, pseudo-ticker drop with param test across all 5 sentinel strings, score-desc + recency tie-break determinism, min_ai_score honesty, newest-first trade ordering, malformed-input degrade-never-raise on both trades and articles).
- `paper_trader/analytics/risk_mirror.py` — third advisory mirror (after self_review + track_record): composes `build_churn` + `build_correlation` **verbatim** (single source of truth #10) into a compact `prompt_block` on the trader's *structural* risk — how concentrated the book is and how much it churns (the 2026-05-17 live pathology: ~$973 / 16.7% win-rate, 60%+ one-sector, 0.52-day median hold). No price history is fetched on the hot decision path (a per-position yfinance call is a live-cycle latency/flake risk); without it `build_correlation` is `INSUFFICIENT` and its headline is the bare "verdict withheld" sentence, so the mirror surfaces the weight-based concentration (`top_weight_pct`/`weight_hhi`/`effective_positions_naive`, computed from `market_value` unconditionally) and only uses the richer ρ headline when a caller supplies `price_history`. Observational only, never gates (invariants #2/#12 — the self_review precedent); `_safe`-wrapped so a builder fault is "no block this cycle", never "no decision". Wired into `decide()` + `_build_payload(... risk_mirror_block=)` (rendered after the track-record section); applies on next paper-trader restart. Locked by `tests/test_risk_mirror.py`
- `paper_trader/analytics/tail_risk.py` — left-tail / downside-shape diagnostic (the upside-heavy surface had none): historical 95/99% 1-day VaR (nearest-rank), positional expected-shortfall CVaR (float-robust — a value-threshold filter silently drops float-equal `-0.10` ties and halves the tail), population annualised vol & downside deviation, Fisher-Pearson population skew, worst/best day, max consecutive down-day streak, Ulcer index. Daily series resampled **byte-identically** to `dashboard.analytics_api`'s `by_day` loop (single-source-of-truth #10 spirit; vol `/n` matches its Sharpe, downside-dev `/n` matches its Sortino). Honesty-gated `NO_DATA`/`INSUFFICIENT`(<`MIN_RETURNS`=20)/`OK` (the `build_correlation` precedent — live book is 5 days so it correctly reads INSUFFICIENT until history matures). Served at `/api/tail-risk` **and** folded into `/api/analytics` as an additive `tail_risk` key so the digital-intern analyst chat inherits it. Observational only — never gates Opus, never injected into the decision prompt (invariants #2/#12). Locked by `tests/test_tail_risk.py` (hand-pinned discrete metrics + independent-impl cross-check for vol/skew) and `tests/test_core_analytics.py::TestTailRiskIntegration` (endpoint↔builder no-drift, additive-key contract)
- `paper_trader/analytics/stress_scenarios.py` — **forward** beta/concentration shock estimate: the **day-one complement to `tail_risk`**, which correctly reads `INSUFFICIENT` until the book has ≥`MIN_RETURNS`=20 daily returns (live book is ~5 days). `build_stress_scenarios(positions, total_value, classify, beta_map, now=None)` is pure `Σ weight×β×shock` over the *current marked book* — **needs zero return history**, so it produces a real $ figure precisely when `tail_risk` is dark. Three families: **market** (SPY −1/−3/−5/−10 % + a +3 % honesty-symmetry line, β-amplified — the −3 % line is **byte-identical** to `/api/risk`'s `shock_usd = Σ −0.03·β·val`, single source of truth #10, locked by `test_minus3_market_equals_api_risk_shock_formula`), **single-name** (largest position alone gaps −10 %, **no β** — the idiosyncratic risk a 60-%-of-book name carries that a diversified book does not), **sector** (heaviest sector corrects −10 % thematically, **no β** — the most decision-relevant line for the ≈98 %-in-two-AI-names live pathology). Per-position β/value computed **identically** to `dashboard.risk_api` (option ×3 cap 4, put-negated). State ladder has **no sample-size gate** (that absence *is* the feature): `NO_DATA` only when the book is empty/unpriceable, else `OK`. Honesty: betas are the approximate `_LEVERAGE_BETA` sector constants and the headline says so (decision support, not VaR). Observational only, never gates, no caps (invariants #2/#12 — the `tail_risk`/`risk_mirror` precedent); pure, never raises (garbage row/None/zero book → honest degrade). Hot-path SSOT discipline: `_LEVERAGE_BETA` is a **test-pinned verbatim copy** of `dashboard._LEVERAGE_BETA` and the strategy/reporter callers pass `sector_exposure.classify` (its own pinned copy) so the live decision path never imports the ~9k-line Flask `dashboard` (the `sector_exposure.SECTOR_MAP` precedent); the `/api/stress-scenarios` endpoint passes the *real* dashboard objects so it is the true SSOT and the copies are CI-pinned to it. Served at `/api/stress-scenarios` **and** folded into `/api/analytics` as an additive `stress_scenarios` key (digital-intern analyst chat inherits it — the `tail_risk` precedent). Wired into `decide()` + `_build_payload(... stress_block=)` (rendered **after** `sector_exposure`, **before** `event_calendar`/`WATCHLIST PRICES`) **and** `reporter._stress_line` appended to `send_hourly_summary`/`send_daily_close` (the operator who lives in Discord sees the $-at-risk without opening the stale dashboard); applies on next paper-trader restart. Locked by `tests/test_stress_scenarios.py` (exact hand-computed $ per family, SSOT no-drift, monotone loss, option-β path, no-sample-gate OK, `_safe`/NO_DATA, prompt render order, `_stress_line` verbatim+fault-degrade, `TestStressScenariosEndpoint` endpoint↔analytics↔builder no-drift, `TestBetaMapIsPinnedToDashboard`)
- `paper_trader/analytics/recovery.py` — **forward** "path back to even": the rally **from here** required to return to the `INITIAL_CASH` start (the universal P/L baseline every Discord report is measured against) and the running high-water peak, per position and for the book — the one question a losing discretionary book asks first and the 50+-builder surface had **zero** of (`/api/drawdown` owns only the *backward* "% of trough clawed back"; this is its **forward complement**). `build_recovery(drawdown_result, tail_risk_result, initial_equity, now=None)` is pure, no I/O, never raises; it **composes verbatim** (single source of truth #10) `compute_drawdown`'s `current_value`/`peak_value` + per-lot `contributors` (the option ×100 is already baked into `unrealized_pl`, read **directly** — never re-derived from `avg_cost×qty`, the `hold_discipline` precedent) and `build_tail_risk`'s realized `annualized_vol_pct` (de-annualised `/√252`). Per-position breakeven is the **raw price ratio** `(avg_cost/current_price − 1)·100` — multiplier-invariant, so an option lot needs **no ×100** (a reviewer "consistency-fixing" it by multiplying through → a `10000%` figure, locked RED — the advisor discriminator); a winner needs `0.0` (never negative noise). The σ figure is **gated on `tail_risk.state=="OK"`**: a young book (the live case) reads `INSUFFICIENT` so the %/$ targets are still emitted but the dispersion figure is **withheld** with an honest sentence (the `tail_risk`/`correlation` sample-size-honesty precedent), and it is explicitly a **dispersion scale, NOT a time forecast** (a random walk's expected first-passage to even is undefined — the headline says so, the `stress_scenarios` "beta-approx, not VaR" honesty tone). State ladder `NO_DATA` (no priced book) → `ABOVE_WATER` (book ≥ start — nothing to recover, the Discord line self-suppresses; incl. the empty-book `compute_drawdown` starting-equity fallback — the `_drawdown_line` at-high-water precedent) → `UNDERWATER`. Observational only, never gates Opus, no caps (invariants #2/#12 — the `tail_risk`/`stress_scenarios` precedent). Served at `/api/recovery` **and** folded into `/api/analytics` as an additive `recovery` key (the digital-intern analyst chat that fetches `/api/analytics` inherits it for free — the `tail_risk`/`stress_scenarios` additive-key precedent; the endpoint and the fold consume the **same** per-request `compute_drawdown`+`build_tail_risk` objects so they cannot drift) **and** `reporter._recovery_line` appended **after `_stress_line`** in `send_hourly_summary`/`send_daily_close` (the operator who lives in Discord sees the rally-to-even without opening the stale dashboard; the load-bearing HOST→CAPITAL adjacency is preserved — recovery sits before HOST). Applies on next paper-trader restart. Locked by `tests/test_recovery.py`
- `paper_trader/analytics/event_calendar.py` — **forward** scheduled-event awareness fed into the live prompt (the mirrors above are all *backward*-looking; this is the one thing a discretionary desk tracks that the engine was fully blind to: upcoming **earnings**). `build_event_calendar(positions, names_in_play, calendar_path=None, now=None, horizon_days=14)` reads digital-intern's `data/earnings_calendar.json` snapshot **directly from disk** — explicitly **not** the `:8080 /api/earnings` endpoint (a network hop on the 60s decision cycle is the documented hang/latency hazard; the `signals.py` filesystem precedent). `_pick_freshest` selects the newest-`as_of` readable candidate across USB/repo/legacy paths (the `signals._db_path` freshness discipline, invariant #15). `days_away` is **recomputed** from `earnings_date` vs `now` (a stale snapshot still yields accurate timing — the digital-intern `api_earnings` rule mirrored verbatim, single source of truth #10), past events (`< -0.5d`) dropped, and each is tiered against the held book exactly as `/api/earnings-risk`: `HELD_IMMINENT` (held & ≤3d), `HELD_SOON` (held & within horizon), `WATCH` (in-play, not held; dropped beyond `horizon_days` as prompt noise — a *held* name's print is never hidden regardless of distance). Observational only, never gates (invariants #2/#12 — the self_review/risk_mirror precedent); `_safe`-style end-to-end so a missing/stale/corrupt/unparseable snapshot degrades to one honest line, **never** an exception that sinks a trading cycle. Served at `/api/event-calendar` (prompt↔endpoint parity — the existing network-sourced `/api/earnings-risk` left untouched, a different concern). Wired into `decide()` + `_build_payload(... event_calendar_block=)` (rendered after `risk_mirror`, before `WATCHLIST PRICES`); applies on next paper-trader restart. **Load-bearing scope:** `decide()` passes held ∪ the **full WATCHLIST** — deliberately **not** the lean `_names_in_play` set the quant / track-record blocks trim to. Those blocks are large per-ticker so they bound prompt length; an earnings event within the 14d horizon is rare (≈0–3 across all 50 names) so there is no bloat to bound, and `WATCHLIST[:5]` excludes most names (e.g. NVDA) — narrowing to `_names_in_play` would silently re-create the exact blind spot this closes (Opus buying a watchlist name the day before its print) **and** break the `/api/event-calendar` parity claim. Do not "consistency-fix" it to `_names_in_play`. Locked by `tests/test_event_calendar.py`
- `paper_trader/analytics/news_velocity.py` — **per-held-ticker** news-flow velocity (2026-05-18): is the catalyst BUILDING or FADING? Compares the article rate over the last `window_hours` (default 24h) to a **non-overlapping** `baseline_hours` (default 168h = prior 6 days) baseline and emits a Poisson z-score plus a state ladder (`SURGING` / `STABLE` / `FADING` / `INSUFFICIENT` / `NO_DATA`). Fills the gap left by adjacent surfaces: `analytics/position_thesis.py` reports the latest 24h headlines per held position (single-window snapshot, no baseline); digital-intern's `analytics/trend_velocity.py` is market-wide 2h-vs-prior-2h (not held-keyed); digital-intern's `analytics/breaking_news_detector.py` is 5-min burst detection (not thesis evolution). None answer the one question this does: *is the news flow on a position I actually own getting LOUDER or QUIETER?* `build_news_velocity(articles, held_tickers, now=None, window_hours=24.0, baseline_hours=168.0)` is **pure** (no DB / no network / never raises on garbage); the endpoint owns the I/O. Sample-size honesty mirrors `build_tail_risk`/`build_correlation`: per-ticker verdict is **withheld** (`INSUFFICIENT`) until `MIN_BASELINE_N=5` baseline articles exist for that ticker — the documented shallow articles.db history (~days deep, not 90) means INSUFFICIENT is the common branch on a new/quiet name and that is honest reporting, not a bug. `SURGING` requires **both** a Poisson z ≥ `Z_SURGE=2.0` AND `window_count ≥ MIN_WINDOW_FOR_SURGE=3` so a baseline of 1 vs a window of 2 (z=+0.6, count=2) does **not** false-positive. Ticker bucketing uses the canonical word-boundary regex `(?:\$|\b)TKR\b` (so `MU` does NOT alias `MUSE`/`MUTUAL`; `$NVDA` cashtag hits; `AMDOCS` ≠ `AMD` — the same locks as `trade_attribution` / `signal_followthrough`). Window-cutoff boundary is **inclusive on the window side** (`first_seen >= now - window_h`) — the `signals.py` precedent. `per_ticker` sorted SURGING-first then STABLE/FADING by z DESC then INSUFFICIENT last so the loudest catalyst surfaces at index 0. Served at `/api/news-velocity` (query params: `window_hours` 1..72, `baseline_hours` window+1..720, `tickers` comma-separated override of held). **Endpoint performance discipline**: a single DESC LIMIT query under-samples the baseline on a high-throughput day (live 2026-05-18 articles.db is 1.47M live rows in 7d, so any `LIMIT N DESC` pull returns only window-era rows — collapsing every baseline_count to 0 and forcing INSUFFICIENT everywhere). Fix: window + baseline are two queries, with the baseline using a **per-ticker `title LIKE '%TKR%'` union in SQL** so the scan drops from ~hundreds-of-thousands of rows to a few thousand before Python word-boundary refinement (1s vs 40s end-to-end). Title-only for baseline rows (skip ~hundreds of full_text decompressions — title regex captures the bucket reliably). Live-only clause applied (invariant #3 — `url NOT LIKE 'backtest://%' AND source NOT LIKE 'backtest_%' AND source NOT LIKE 'opus_annotation%'`). Reads the articles.db via the freshness-aware `_articles_db_path()` (invariant #15/#17). Observational only, never gates Opus, no caps (invariants #2/#12 — the `tail_risk`/`stress_scenarios` precedent). Locked by `tests/test_news_velocity.py` (25 tests: state-ladder honesty including the at-MIN_BASELINE_N FADING-not-INSUFFICIENT boundary, exact hand-computed Poisson z + ratio, surge-requires-both-z-AND-floor regression, window-cutoff strict inclusivity, MU≠MUSE/MUTUAL/AMDOCS≠AMD word-boundary discriminators, sort priority SURGING-first/INSUFFICIENT-last, case-insensitive held-ticker dedup, defensive `baseline ≤ window` → NO_DATA, garbage-row degrade-never-raise, plus `TestNewsVelocityEndpoint` Flask test_client coverage for no-DB degrade / ticker override / param garbage-clamp).
- `paper_trader/analytics/implied_move.py` — **forward, market-priced** complement to the backward-looking `earnings_shock` (historical-Gaussian σ) and `earnings_distribution` (empirical observed quartiles). Answers the discretionary trader's #1 pre-earnings question: *"what is the market currently pricing as the move on this print?"*. For each held imminent earnings event (held ∩ event_calendar within `horizon_days`=7), pulls the options chain at the expiry closest to `ceil(days_away)` so the chain **captures** the event, picks the ATM call (min `|strike − spot|`) and ATM put **independently** (so a one-sided chain gap doesn't silently mis-pair), computes the straddle mid as `(bid+ask)/2` when both > 0 else `lastPrice` else `None` (NaN-rejected — yfinance returns NaN for thin chains and a half-NaN sum would silently halve every implied), and reports `implied_move_pct = straddle / spot * 100` (desk-standard breakeven-move shorthand) plus the strict Black-Scholes 1σ ≈ 0.8 × straddle/spot and the per-strike `iv_atm` (decimal → percent). Dollarized: `position_value × implied_pct / 100 → $-at-risk`; `book_pct = $-at-risk / total_value * 100`. State ladder mirrors `earnings_shock`: `NO_DATA` (empty/unpriceable book) / `NO_EVENTS` (book fine, calendar quiet — distinct so the operator can tell "calendar quiet" from "book empty") / `OK`. Per-event honesty: `OK` / `NO_QUOTES` (chain returned but ATM bid/ask/last all zero — withhold rather than fabricate 0%) / `NO_CHAIN` (provider returned None or raised — `_safe` contract, never propagates). Composes `build_event_calendar`'s `events` list **verbatim** (single source of truth #10) so this endpoint, `/api/earnings-shock`, `/api/earnings-distribution` and `/api/event-calendar` can never disagree on what counts as held-imminent. Option positions degrade to `NO_CHAIN` (the current_price is the premium, not the underlying — an implied-move read on an option-on-an-earnings-name is its own feature, not this one). Pure builder, never raises on garbage rows (the `_safe` discipline); the endpoint owns the yfinance I/O (`market.get_options_chain` via a `_chain_provider` lambda). Served at `/api/implied-move`, SWR-cached 5 min matching `/api/earnings-shock` cadence, and registered in the prewarm set (the `test_swr_prewarm_coverage` invariant). Observational only, never gates Opus, never injected into the decision prompt, no caps (invariants #2/#12 — the `earnings_shock` / `stress_scenarios` / `recovery` precedent). Locked by `tests/test_implied_move.py` (34 tests: helper-fn discriminators (NaN-rejected `_safe_float`, mid-wins-over-last `_mid_or_last`, min-distance `_atm_row`); state ladder NO_DATA / NO_EVENTS-distance / NO_EVENTS-unheld / OK; exact hand-computed arithmetic on a pinned $1000-book NVDA scenario (`straddle=8.0 → implied=8% → $32 → 3.2% of book → MODERATE`); ELEVATED/MODERATE/LOW threshold boundaries; degrade paths NO_CHAIN-on-None / NO_CHAIN-on-raise / NO_QUOTES-on-thin / option-position-no-spot / garbage-row-never-raises / past-event-dropped; sort order by `days_to_earnings`; total_implied_book_pct sum-of-abs; `TestImpliedMoveEndpoint` Flask test_client (offline-deterministic via monkeypatched `market.get_options_chain` + on-disk event_calendar fixture) — endpoint↔builder no-drift on the load-bearing fields, empty-book→NO_DATA-not-500)
- `paper_trader/analytics/pnl_attribution.py` — **β-adjusted** decomposition of open unrealized P/L into `β·SPY-explained` vs **idiosyncratic** residual. The honest answer to *"is my NVDA gain just SPY going up?"* — `analytics/open_attribution.py` (`/api/open-attribution`) reports the unweighted `position_return − spy_return`, which **implicitly assumes β=1** and so systematically over-attributes "alpha" on the bot's documented leveraged-ETF/semis book (TQQQ β=3, NVDA β=1.5, FNGU β=3); the desk-correct read is `idiosyncratic = position_return − β × spy_return`. The discriminating disagreement: a +1% SPY day on a $200 TQQQ position contributes ≈$6 of β·SPY-explained P/L, not $2. `build_pnl_attribution(positions, equity_curve, classify, beta_map, now=None)` is pure (no I/O, never raises — the `_safe` contract). Anchors per-position to the equity curve's `sp500_price` at-or-after `opened_at` (same SSOT as `open_attribution`'s `_spy_at_or_after`, AGENTS.md #10 — a drift in either side fails the no-drift check). Uses the **same** `_classify`/`_LEVERAGE_BETA` SSOT as `stress_scenarios` / `/api/risk` so an unknown-sector ticker reads β=1.0 *exactly* the same as those panels do; an unmapped ticker → `classify→"other"` → β=1.0 (market-beta fallback, the `_position_betas` precedent). Options are flagged and skipped (β-attribution on a Greeks instrument is its own surface, see `/api/greeks` — the `open_attribution` precedent). State ladder `NO_DATA` (no stock positions / all skipped) / `NO_BENCHMARK` (positions exist but equity curve has no `sp500_price` history — cold start before the first equity tick) / `INSUFFICIENT` (positions exist + benchmark exists but no position's `opened_at` anchors against the available SPY history, e.g. all opened *after* the last equity tick — the row is still emitted with `anchored=False` for honest withholding) / `OK` (at least one anchored row). Per-row fields: `beta`, `position_return_pct`, `spy_return_pct`, `beta_explained_pct = β × spy_return`, `idiosyncratic_pct = position_return − beta_explained`, plus all four dollarized. Book totals satisfy `unrealized_usd = beta_explained_usd + idiosyncratic_usd` exactly (asserted in the suite). Rows sorted by `|idiosyncratic_usd|` DESC (the desk's "what is selection actually contributing" sort), unanchored rows last. Served at `/api/pnl-attribution` (the **true SSOT** — strategy-side pinned copies are CI-pinned to its `_classify`/`_LEVERAGE_BETA`) **and** folded into `/api/analytics` as an additive `pnl_attribution` key (digital-intern analyst chat that fetches `/api/analytics` inherits it for free — the `tail_risk`/`stress_scenarios`/`recovery` additive-key precedent). Observational only, never gates Opus, never injected into the decision prompt, no caps (invariants #2/#12 — the `open_attribution` precedent). Locked by `tests/test_pnl_attribution.py` (19 tests: state ladder NO_DATA-empty / NO_DATA-options-only / NO_BENCHMARK-no-SPY / INSUFFICIENT-opened-after-curve; **exact hand-computed β-decomposition** on a pinned NVDA β=1.5 scenario (`+10% pos, +1% SPY, +1.5% β-explained, +8.5% idio`); the **TQQQ leveraged discriminator** (`+6% pos, +2% SPY, β=3 → β-explained=6%, idio=0% — vs `open_attribution`'s would-be +4% naive alpha — that disagreement IS the value-add); negative-idiosyncratic-on-losing-leveraged-ETF (TQQQ flat while SPY +1%, β=3 → idio=-3%); totals math (book unrealized = β-explained + idiosyncratic, exact); unknown-ticker→β=1 fallback; option-skip; classify-raises→other-fallback (`_safe`); garbage-position-never-raises; corrupt-equity-curve-row-skipped; sort by |idio| DESC with unanchored last; `TestPnlAttributionEndpoint` Flask test_client (real Store + `update_position_marks` to seed `current_price`) — endpoint serves builder output with `β=1.5` on NVDA (the dashboard's true SSOT); `TestAnalyticsFold` no-drift between `/api/analytics["pnl_attribution"]` and `/api/pnl-attribution` on the load-bearing fields)
- `paper_trader/analytics/idle_opportunity.py` — **drought-regret** quantifier (2026-05-19): during the current PARALYSIS drought, which high-score live signals on the watchlist arrived that the bot never decided against? The recurring host-saturation NO_DECISION storms have three diagnostic surfaces — `/api/decision-drought` (WHEN + realized portfolio drift), `/api/host-guard`+`/api/decision-forensics` (WHY: host load + dominant parse-fail mode), `/api/shadow-vs-claude` (RIGHT-NOW rec, **snapshot-only by design** per its docstring, does not look across the drought window) — but none answer the operator's "while the bot was dark for 9.0h, did anything high-score actually arrive on a name I follow that I would have acted on?" question. `build_idle_opportunity(decision_drought_result, articles, watchlist, held_tickers=None, now=None, min_ai_score=6.0, max_opportunities=20)` is pure (no I/O, never raises on garbage — the `_safe` contract). **Composes `build_decision_drought.current_drought` verbatim** (SSOT, AGENTS.md #10 — so `/api/idle-opportunity` and `/api/decision-drought` can never disagree on what counts as an ongoing drought). Buckets live-only watchlist articles at-or-above the score floor inside `[drought_start, now]` per ticker, keeping the top article per ticker (score DESC, tie-break **newer first_seen** — more plausibly causal, the `trade_attribution` precedent). State ladder `NO_DATA` (no decisions yet) → `NO_DROUGHT` (bot filling normally — by definition nothing was missed, suppressed in reports) → `OK` (drought exists; `n_opportunities` carries the regret list, possibly empty — the silence-when-nothing-actionable case, the `_macro_calendar_chat_lines`/`_event_readiness_chat_lines` precedent). Word-boundary regex `(?:\$|\b)TKR\b` (so `MU` does NOT alias `MUTUAL`/`MUSE`; `$NVDA` cashtag hits; `AMDOCS` ≠ `AMD` — the same locks as `news_velocity` / `trade_attribution` / `signal_followthrough`). NaN/Inf `ai_score` explicitly rejected (digital-intern's column has been observed with stale NaNs from a half-trained model; Python NaN compared with min_ai_score is ALWAYS False which would silently drop the row otherwise). Held positions flagged via `held` so the headline can call out *"the bot was dark on MY OWN position's news"*. Served at `/api/idle-opportunity` (query params: `min_ai_score` 0..10, `max_opportunities` 1..50). **Endpoint short-circuits when no ongoing drought** — saves the articles.db read entirely (the operator-happy path; same cost discipline as `news_velocity`'s drought-bounded scan, typically narrows to hundreds of rows even on the 1.47M-rows/7d articles.db). Live-only clause applied at the SQL layer (invariant #3 — synthetic `backtest://` and `opus_annotation*` rows never reach the panel). `reporter._idle_opportunity_line` appended to `send_hourly_summary` and `send_daily_close` **after `_host_pulse_line`, before `_capital_pulse_line`** (load-bearing order: HOST names the CAUSE, IDLE names what was MISSED while the cause held, CAPITAL names the manual-fix lever; all three can be independently true and none suppresses the others — the same HOST/CAPITAL independence precedent). Composes the builder's `headline` verbatim so the Discord line and the endpoint can never drift. Observational only — never gates Opus, never injected into the decision prompt, no caps (AGENTS.md #2/#12 — the `shadow_vs_claude`/`stress_scenarios`/`recovery` precedent). Applies on next paper-trader restart. **Live integration (May 19, 9.0h ongoing drought, 34 NO_DECISION):** `state=OK, headline="loudest: MU @ ai_score 9.0"` — MU: BofA doubled price target to 950 (2 articles); NVDA (held): earnings tomorrow (8 articles); DRAM and AMD also flagged. Locked by `tests/test_idle_opportunity.py` (33 tests: state ladder NO_DATA/NO_DROUGHT/closed-drought→NO_DROUGHT/OK-quiet/OK-regret + headline-HELD-tag; drought-window strict-inclusive boundary; ai_score floor + clamp + NaN/Inf-reject + min_ai_score-carried-in-payload; word-boundary discriminators MU≠MUTUAL/AMD≠AMDOCS/$NVDA-cashtag + the compiled-regex helper directly; per-ticker bucketing including score-DESC then newest-first tie-break order-independent + max-urgency carried; sort top_score DESC; max_opportunities cap; held flag; garbage-degrade-never-raises for None-articles/non-dict-rows/missing-fields/unparseable-ts/empty-watchlist + `_safe_float` helper; defensive drought-with-no-start→NO_DATA; `TestIdleOpportunityEndpoint` Flask test_client with a real seeded articles.db proving the synthetic backtest row at score 9.5 is filtered by the SQL-side live-only clause + min_ai_score param clamp + NO_DROUGHT short-circuit no-DB-read sentinel) and `tests/test_core_reporter.py::TestIdleOpportunityLine` (7 tests: silence-when-no-drought / silence-when-quiet / regret-verbatim / builder-fault-degrade / HOST→IDLE→CAPITAL ordering lock in `send_hourly_summary` / hourly-silent-when-quiet / daily-close-includes-idle).
- `paper_trader/analytics/etf_lookthrough.py` — **single-name look-through** through leveraged-ETF positions (2026-05-19): pierce TQQQ/SOXL/FNGU/TECL/SPXL/etc. into the *effective* per-ticker exposure a book of NVDA + TQQQ truly carries. Every existing risk surface stops at the ticker boundary — `sector_exposure` classifies TQQQ as `broad_lev` and adds the full position to that sector; `risk_mirror` reports HHI on line-item tickers; `pnl_attribution` β-decomposes the move but does NOT pierce the ETF wrapper. The decision-relevant question this answers: *"I hold $445 NVDA cash AND $295 TQQQ. TQQQ is 3x QQQ, QQQ is ~9% NVDA. What is my TRUE effective NVDA exposure?"* Live answer on the actual book (2026-05-19): **52.4% effective NVDA vs 44.5% direct (8pp hidden via TQQQ amplification)**, plus **silent MSFT/AAPL/AMZN** lines via TQQQ that have **zero direct positions** but ride the next NVDA earnings move with the book regardless. `build_etf_lookthrough(snapshot, lookthrough_map=None, hidden_ratio=1.5, max_underlyings=12)` is pure (no I/O, never raises — the `_safe` contract). Position value mirrors `sector_exposure._position_value` **verbatim** (single source of truth #10 — same `(current_price or avg_cost) * qty * (100 if option else 1)` formula, so starting weights cannot drift). For each held position: if ticker is in `_ETF_LOOKTHROUGH`, decompose into virtual exposures `position_value × leverage × weight%`; otherwise contribute as direct. Inverse ETFs (SQQQ/SOXS/SPXS/FNGD/TECS) carry NEGATIVE leverage so they honestly SHORT their basket — a book of NVDA + SQQQ reads net NVDA = direct − indirect (the sign honesty IS the value-add — locked RED by the inverse-ETF test). Options skipped from look-through (delta-adjustment is its own surface — the `pnl_attribution` option-skip precedent) but still contribute as direct exposure. Per-underlying tier ladder: `HIDDEN_AMPLIFIED` (effective_pct ≥ `HIDDEN_RATIO`=1.5× direct_pct), `HIDDEN_ONLY` (no direct line, ETF-only exposure — the silent-MSFT case), `TRANSPARENT` (look-through within 1.5×), `TRIVIAL` (<0.5% of book). State ladder `NO_DATA` (no priced book) / `NO_ETF_HELD` (book exists but no mapped ETF held — look-through ≡ direct, suppressed in reports — the `decision_drought`/`no_decision_recovery` silence-when-nothing-actionable precedent) / `OK`. Sort by `|effective_usd|` DESC tie-break `direct_usd` DESC. Headline priority: HIDDEN_AMPLIFIED → HIDDEN_ONLY → largest-effective fallback. Served at `/api/etf-lookthrough` **and** folded into `/api/analytics` as an additive `etf_lookthrough` key (digital-intern analyst chat that fetches `/api/analytics` inherits it for free — the `tail_risk`/`stress_scenarios`/`recovery`/`pnl_attribution` additive-key precedent). Observational only, never gates Opus, no caps (invariants #2/#12). Static weights are approximate top-10 issuer-fact-sheet constituents (late-2025); the headline says so — decision support, not VaR. Locked by `tests/test_etf_lookthrough.py` (35 tests: state ladder NO_DATA/NO_ETF_HELD/OK; exact hand-computed arithmetic (`$100 × 3.0 × 10% = $30 indirect NVDA`); direct+indirect=effective totals; inverse-ETF NEGATIVE leverage sign honesty; multi-ETF compounding (TQQQ + SOXL both lift NVDA additively); tier classification HIDDEN_AMPLIFIED-at-1.5× boundary / HIDDEN_ONLY-when-no-direct / TRANSPARENT-below-ratio / TRIVIAL-below-0.5%; sort order |effective_usd| DESC; max_underlyings cap; options-not-looked-through (NVDA call doesn't decompose); live-map sanity (canonical TQQQ/SQQQ/SOXL/SOXS/FNGU/FNGD/TECL/SPXL/UPRO present + inverse-leverage-negative + long-leverage-positive + holdings-shape); garbage-degrade-never-raises (None snapshot / non-dict rows / missing keys / unparseable weights); `TestEtfLookthroughEndpoint` Flask test_client (real Store + `update_position_marks`) endpoint↔builder no-drift on state/headline/n_etfs_held + `/api/analytics` additive-fold no-drift).
- `paper_trader/analytics/earnings_war_room.py` — **pre-print game plan** composer (2026-05-19): one composite view answering *"if NVDA gaps by its implied move tomorrow, what does my book actually look like after?"*. Forward-earnings exposure has six siloed surfaces today (`/api/event-calendar`, `/api/implied-move`, `/api/earnings-shock`, `/api/earnings-distribution`, `/api/stress-scenarios`, `/api/sector-exposure`) and none composes the worst-case projection: post-shock book value vs the $1000 start, post-shock concentration, total $-at-risk across all held imminent prints. `build_earnings_war_room(positions, total_value, initial_equity, event_calendar_result, implied_move_result=None, earnings_shock_result=None, stress_scenarios_result=None, now=None, horizon_days=7.0)` is the composer — **no new measurement** beyond `position_value × shock_pct` arithmetic and the post-shock book-value projection; every input is read **verbatim** (single source of truth #10) from the sibling builders' results so this endpoint and the sibling endpoints can never disagree. Per-event row: `weight_pct`, `implied_move_pct`+`implied_dollar_at_risk`+`implied_book_pct` (from `implied_move`), `sigma_pct`+`sigma_dollar_move`+`sigma_book_pct` (from `earnings_shock`), `single_name_shock_dollar` (from `stress_scenarios.single_name` when ticker matches — verbatim SSOT — else a verbatim copy of `_SINGLE_NAME_GAP_PCT=-10` applied to position value; `single_name_shock_source` reads `"stress"`/`"default"` so drift is auditable), `post_shock_total_value` (book mark-to-market after this position drops by `|implied_move_pct|`, others flat — the worst-direction frame), `post_shock_vs_initial_pct` (vs the $1000 baseline), `post_shock_weight_pct` (concentration after the shock — provably reduces vs starting weight, locked). Per-event impact tier = `max(|implied_book_pct|, |sigma_book_pct|)`: HIGH ≥ 5 %, MEDIUM ≥ 2 %, LOW below. The MAX-over-both means a chain miss still tiers off historical σ AND an IPO-name with no prior prints still tiers off the implied straddle (the discriminating value-add: a row never goes UNKNOWN merely because one of two sources is missing). Top-level: `total_implied_dollars_at_risk` (sum of |implied|), `worst_case_event` (largest |implied|, fallback |σ|, fallback |single-name|), `verdict` (tier of the worst event), one-line Discord-ready `headline`. State ladder: `NO_DATA` (empty/unpriceable book) / `NO_EVENTS` (priced book, no held imminent event ≤ horizon — distinct so the operator can tell "calendar quiet" from "book empty") / `OK`. Per-event state ladder is independent: `OK` (at least one of implied/σ available) / `INSUFFICIENT` (both missing — row still emits ticker/days/weight so the operator never misses *"NVDA reports tomorrow"*, the `earnings_shock` honesty precedent). Served at `/api/earnings-war-room`, SWR-cached 5 min matching the sibling earnings endpoints, and registered in the prewarm set (the `test_swr_prewarm_coverage` invariant). Observational / advisory only — never gates Opus, never injected into the decision prompt, no caps (invariants #2/#12 — the `earnings_shock` / `stress_scenarios` / `recovery` precedent). Pure, no I/O, never raises. Locked by `tests/test_earnings_war_room.py` (40 tests: state ladder NO_DATA/NO_EVENTS-multiple-causes/OK; **exact hand-computed arithmetic** on a pinned $1000-book NVDA-2-at-$222.35 scenario (`weight 44.47%, implied 7% → $31.13 → 3.11% book → post-shock $968.87 → -3.11% vs start → weight reduces to 42.69%`); total `implied_dollars_at_risk = sum-of-abs`; tier ladder HIGH-at-5%-boundary/MEDIUM-at-2%-boundary/LOW-below + MAX-over-implied-and-σ + tier-when-only-one-source + UNKNOWN-when-both-missing; INSUFFICIENT honesty branch (row still emits ticker/weight/headline-without-numerics); single-name SSOT consumption from `stress_scenarios.single_name` when ticker matches + verbatim-default-gap fallback when stress names a different ticker; sort by `days_to_earnings`; worst-case picks largest `|implied_dollar|`; garbage-degrade-never-raises (None positions / non-dict rows / garbage total_value / "bad" initial_equity / None event_calendar / non-dict event_calendar / non-dict-event in events list / garbage implied_result / non-numeric days_away); options-position 100× multiplier (NVDA call @ $4 premium × 2 = $800, not $8 — verbatim from `_position_value`); `TestEarningsWarRoomEndpoint` Flask test_client (real seeded Store + monkeypatched `market.get_options_chain` + monkeypatched `_earnings_history_for` + monkeypatched `build_event_calendar` so the test is offline-deterministic) — endpoint state/n_events/weight_pct/tier/sigma/worst-case no-drift with the builder).
- `paper_trader/analytics/news_source_mix.py` — **per-held-ticker source-diversity verdict** (2026-05-19): is the catalyst REAL or a SYNDICATED ECHO? Orthogonal to `news_velocity` (which measures rate): a SURGING z-score of +4 looks identical whether five outlets are reporting genuine news OR one wire is being mirrored across five feeds. `build_news_source_mix` counts DISTINCT collector sources per held ticker over `window_hours` (default 24h) and tags `STRONG` (≥4 sources, ≥3 articles), `MODERATE` (≥2 sources, ≥3 articles), `ECHO` (≥`ECHO_MIN_ARTICLES`=3 articles AND single source ≥`ECHO_THRESHOLD_PCT`=70%), or `QUIET` (<2 articles). Combined with velocity: `SURGING + STRONG = real catalyst worth re-evaluating`, `SURGING + ECHO = syndication artifact — do NOT chase`. Pure builder; the endpoint owns the articles.db I/O via `_articles_db_path()` (invariant #15/#17) + live-only SQL clause + per-ticker `title LIKE` union (same cost discipline as `news_velocity`). `reporter._source_mix_line` appended to `send_hourly_summary` and `send_daily_close` **only when at least one held ticker reads ECHO** — every other state is silent (the silence-when-nothing-actionable precedent). Folded into `/api/analytics` as an additive `news_source_mix` key so the digital-intern analyst chat inherits the breadth verdict for free (the `tail_risk`/`stress_scenarios` additive-key precedent). Served at `/api/news-source-mix` (query params: `window_hours` 1..72, `tickers` comma override). Observational only — never gates Opus, never injected into the decision prompt, no caps (invariants #2/#12). Locked by `tests/test_news_source_mix.py` (40 tests: state ladder STRONG/MODERATE/ECHO/QUIET at the count + dominant-share boundaries; sort priority ECHO-first then STRONG then QUIET; sample-size honesty (`MIN_FOR_VERDICT`=2 → QUIET sparse); word-boundary regex (MU≠MUTUAL, AMD≠AMDOCS, $NVDA cashtag); `_BREAKDOWN_TOP_N`=5 sources cap; window-cutoff strict inclusivity; degrade-never-raise on None/non-dict/garbage rows; `TestNewsSourceMixEndpoint` Flask test_client offline-deterministic) and `tests/test_core_reporter.py::TestSourceMixLine` (9 tests: silence when no positions / no DB / state QUIET / state NO_DATA, verbatim ECHO headline on a real seeded articles.db, builder-fault-degrade, hourly+daily integration with the load-bearing IDLE→MIX ordering lock).
- `paper_trader/analytics/reentry_velocity.py` — **per-ticker close→re-buy cadence** distribution (2026-05-19): the missing fast-flip surface. `round_trips.build_round_trips` groups raw trades into closed round-trips; `track_record` composes the loser/winner-autopsy narrative; `churn` measures size-weighted intraday turnover. None answer *"after closing a position, how quickly did the desk re-enter the same ticker?"* — the documented live pathology (CLAUDE.md/AGENTS.md: `avg_holding_days` ~0.27 with the NVDA→LITE→NVDA shape and `KNIFE_CATCH` repeats). `build_reentry_velocity` is pure — composes `build_round_trips` (single source of truth #10) and walks each (ticker,type,strike,expiry) key's exits to the next same-key entry, computing the hour-gap distribution and bucket histogram `IMMEDIATE` (<1h) / `SAME_DAY` (1h..24h) / `QUICK` (1..3d) / `NORMAL` (3..14d) / `RARE` (>14d). Open positions whose key has a prior closed round-trip surface as `open_after_close=True` so the *live* fast-flip case is visible too (round_trips alone never sees a still-open re-entry). Verdict ladder `CHURN_RISK` (median gap <24h with ≥3 gaps) / `STABLE` / `SPARSE` (no gaps yet — the live cold-start state). Served at `/api/reentry-velocity` (query param: `recent_limit` 1..100, default 10). Observational only — never gates Opus, never injected into the decision prompt, no caps (invariants #2/#12 — the `track_record`/`churn` precedent). Locked by `tests/test_reentry_velocity.py` (16 tests: state ladder, bucket boundaries IMMEDIATE/SAME_DAY/QUICK/NORMAL/RARE, per-(ticker,type,strike,expiry) key isolation, open_after_close gap-from-last-close, verdict ladder CHURN_RISK at <24h median, bucket-sum=n_gaps, input-order-tolerance via build_round_trips, recent_gaps capped by limit).
- `paper_trader/analytics/blocked_repeats.py` — **repeated-BLOCKED action audit** (2026-05-19): orthogonal to every NO_DECISION surface (which says Claude didn't reply) — this catches the case where Claude DID reply but the engine rejected the trade, **and Opus keeps retrying the same action**. `decisions.action_taken` carries `"BUY NVDA → BLOCKED"` (invariant #11) with the BLOCKED reason in the JSON reasoning blob's `detail` field; nothing on the dashboard or in Discord named it. `build_blocked_repeats` is pure — parses (verb, ticker) from the action_taken prefix, classifies each detail to `CASH` / `DATA` / `SIZING` / `SPECIFICATION` / `OTHER` via a pinned phrase map (every documented `strategy._execute` BLOCKED message has a mapping), aggregates by (verb, ticker), filters to `count >= min_repeat` (default 2), and sorts by count DESC then latest_ts DESC. The cause buckets are operator-actionable: **CASH** = fund the trade, **DATA** = fix the feed, **SIZING** = re-prompt Opus with held qty, **SPECIFICATION** = clarify strike/expiry/qty. Verdict ladder `CLEAN` (no repeats — silent in any future reporter wiring) / `REPEATING` (≥1 repeat — headline names worst offender, count, dominant_cause). Served at `/api/blocked-repeats` (query params: `limit` 50..5000 default 500, `min_repeat` 2..20 default 2). Observational only — never gates Opus, never injected into the decision prompt, no caps (invariants #2/#12). Locked by `tests/test_blocked_repeats.py` (41 tests: every pinned cause phrase → bucket; sort tiebreaker (count DESC, then newest-latest_ts first); per-(verb, ticker) key isolation (BUY NVDA ≠ BUY_CALL NVDA); FILLED rows excluded; min_repeat override; malformed JSON reasoning + non-dict rows degrade-never-raise; the load-bearing **test-isolation rule below**; `TestEndpoint` offline-deterministic via pure list-backed stub Store).
- `paper_trader/analytics/holding_period_distribution.py` — **closed-trip P/L stratified by hold duration** (2026-05-20): the discretionary-PM "where in the holding-period axis does my P/L actually live?" question — orthogonal to `/api/track-record` (aggregate ledger), `/api/round-trip-postmortem` (per-trip exit drift), `/api/trade-asymmetry` (disposition gap), and the autopsy pair (per-trip narrative). Buckets every closed round-trip into SCALP (<1h) / INTRADAY (1-6h) / OVERNIGHT (6-24h) / SWING (1-3d) / TREND (3-7d) / POSITION (>7d), exclusive-upper inclusive-lower edges pinned by tests. Per-bucket: n_trips, n_winners/n_losers, total_pnl_usd, avg/median_pnl_usd, win_rate_pct (winners ÷ decided, zero-pnl excluded from denominator — `track_record` precedent), share_of_trips_pct, share_of_abs_pnl_pct (denominator is |Σ pnl|, never the signed sum). Cross-bucket verdicts: `alpha_engine` (highest-P/L bucket; must be net positive — a least-loss bucket isn't "alpha"); `dominant_bucket` (most trips; ties broken in canonical SCALP→POSITION order); `worst_bucket` (most negative). State ladder mirrors `round_trip_postmortem`: NO_DATA → INSUFFICIENT (<`STABLE_MIN_TRIPS`=5; bucket rows still emit, only the verdict withheld) → OK. Composes `analytics.round_trips.build_round_trips` (SSOT, invariant #10) — no re-implemented aggregation. Pure builder; endpoint owns I/O over `store.recent_trades`. Served at `/api/holding-period-distribution`. Observational only — never gates Opus, no caps (invariants #2/#12). Locked by `tests/test_holding_period_distribution.py` (20 tests: each bucket boundary inclusive-lower/exclusive-upper at 1h/6h/24h/72h/168h; canonical bucket order; NO_DATA/INSUFFICIENT/OK ladder; total_pnl/win_rate/median arithmetic; zero-pnl excluded from decided; share_of_abs_pnl denominator; alpha_engine must-be-net-positive; dominant tie-break canonical-order; degrade-never-raise on garbage rows / non-list input).
- `paper_trader/analytics/add_discipline.py` — **ADD-trade discipline audit: chasing vs averaging-down vs stacking** (2026-05-20). When the book BUYs into a name it already holds, the ADD carries a sign: paying above the running avg_cost is the textbook *chasing* pathology; paying below is *averaging down* (rational if the thesis holds, the `loser_autopsy` SLOW_BLEED setup if not). **No existing endpoint watched the ADD moment.** `trade_asymmetry` is disposition, `churn` is turnover, the autopsies narrate closed trips — none of them see the ADD's price vs basis. Walks the trade ledger per position-key, maintains a running avg_cost (same VWAP as `store.upsert_position`), classifies each non-opening BUY against the running basis BEFORE the trade is applied: `CHASING` (≥`CHASE_THRESHOLD_PCT`=1.5% above, inclusive at the band with a 1e-6 epsilon for IEEE-754 jitter at the boundary), `AVERAGING_DOWN` (≤-1.5% below), `STACKING` (inside the band). SELL-to-zero (full close) resets the per-position basis cycle so the next BUY is an "open" not an ADD (mirrors `build_round_trips` so the two views never disagree). Partial SELLs leave basis intact. Distinct (option_type, strike, expiry) position keys are isolated — an option BUY on a held stock is a new open, not an ADD on the stock basis. Reports per-add rows (ticker/ts/price/running_avg_cost_before/pct_above_cost/category), `counts`+`pct`, `by_ticker` (sorted by n_adds desc, ties by ticker for determinism), `closed_outcomes` (each closed round-trip tagged with its dominant ADD style via `entry_trade_ids` join; `CHASING > AVERAGING_DOWN > STACKING` precedence on ties so the riskiest behaviour can't be masked by a balancing avg-down), `outcomes_by_style` ({n, total_pnl_usd, mean_pnl_pct, median_pnl_pct} — answers the falsifiable question: did chasing-ADDs produce worse round-trip P/L than averaging-down ADDs?). State ladder: NO_DATA (no BUYs / only opens) → EMERGING (<`STABLE_MIN_ADDS`=3; counts emit but `dominant_style_overall` withheld) → STABLE. Composes `analytics.round_trips.build_round_trips` (SSOT, invariant #10). Pure builder; endpoint owns I/O over `store.recent_trades`. Served at `/api/add-discipline`. Observational only — never gates Opus, no caps (invariants #2/#12). Locked by `tests/test_add_discipline.py` (22 tests: opening-BUY ≠ ADD; classification at ±1.5% inclusive boundary (incl. IEEE-754 jitter at the band edge); VWAP-blended-basis update after each ADD (a stale basis would misclassify subsequent adds — the load-bearing math test); SELL-to-zero resets basis vs partial-SELL doesn't; distinct position keys (stock vs call) isolated; CHASING > AVG_DOWN > STACKING precedence on ties; round-trip outcome rollup signs (CHASING trip −P/L vs AVG_DOWN trip +P/L); NO_DATA/EMERGING/STABLE ladder; by_ticker sort + dominant per ticker; degrade-never-raise on garbage rows / non-list input / non-positive basis defensive STACKING fallback).
- `paper_trader/analytics/rebuy_regret.py` — **sell-then-rebuy $ regret** quantifier (2026-05-20): did the desk save or lose money on close→re-entry hops? `reentry_velocity` tracks the *time* gap; `churn` measures size-weighted turnover; `hold_discipline` reports hold duration. None answer *"when I sold a name and bought it back later, did I save money or lose money in the round-trip-to-re-entry hop?"* The discriminating disagreement vs `reentry_velocity`: a fast re-entry is not inherently bad — what matters is the *price delta* over the gap. Selling NVDA at $220 and re-buying 2h later at $218 *saved* money (timing edge); re-buying at $223 *cost* money (whipsaw). `build_rebuy_regret(trades, now=None, recent_limit=10)` is pure — composes `build_round_trips` (SSOT, invariant #10) for closed round-trips, then walks the trade stream for the next same-key BUY to measure `(rebuy_price − sell_price) × min(sell_qty, rebuy_qty)` against shared quantity, with the option ×100 multiplier honored (the `round_trips` precedent). Sign convention: **positive regret_usd = lost** (sold low, bought back higher); **negative = saved** (sold high, bought back lower). Per-event ladder: `REGRET_HIGH` (>$5) / `REGRET` ($0.50..$5) / `NEUTRAL` (±$0.50) / `SAVED` (−$0.50..−$5) / `SAVED_HIGH` (<−$5). Top-level verdict: `REGRETTING` / `SAVINGS` / `NET_NEUTRAL` / `NO_REBUYS` / `NO_DATA`. Per-ticker rollup sorted worst offender first. Headline names the worst single event by ticker, sold price, re-bought price. Pure / no I/O / never raises (the `_safe` contract — pre-filters garbage rows so `build_round_trips`'s raw `t["ticker"]` cannot KeyError). Served at `/api/rebuy-regret` (query param: `recent_limit` 1..100, default 10). Observational only — never gates Opus, no caps (invariants #2/#12 — the `reentry_velocity`/`churn` precedent). Locked by `tests/test_rebuy_regret.py` (23 tests: state ladder NO_DATA-empty / NO_REBUYS-no-re-entry / OK-with-events; **sign convention** sold-low-bought-higher=positive=REGRET vs sold-high-bought-lower=negative=SAVED; NEUTRAL floor boundary exact ($0.50 strict-> NEUTRAL); shared_qty = min(sell_qty, rebuy_qty) including the rebuy-larger and sell-larger cases; option ×100 multiplier exact ($100 regret on a 1-contract $1-delta hop); per-key isolation (NVDA stock ≠ NVDA call ≠ different strike call); per-ticker rollup sums correctly + worst offender first; mixed regret+savings nets correctly; recent_events newest-first + recent_limit cap; gap_hours computed from exit_ts→rebuy_ts; degrade-never-raise on None/non-dict/garbage timestamps/zero-price rebuy; input-order tolerance via internal sort).
- `paper_trader/analytics/news_to_trade_lag.py` — **news-to-trade lag distribution** (2026-05-20): is the desk reacting to fresh news or consistently 2 hours behind? `/api/trade-attribution` enumerates the highest-scored articles preceding each FILLED trade with a `minutes_before_trade` per attributed article; this builder compresses that detail to one distribution + verdict on the desk's *reactivity*. Composes `build_trade_attribution` (SSOT, invariant #10). For each attributed trade, takes the **minimum** `minutes_before_trade` across its attributed articles (the freshest plausibly-causal signal the trade could have reacted to). Trades with zero attributions are counted separately (`n_no_attribution`); honestly **excluded** from the distribution rather than assigned `window_hours` as a fake worst case (the `recovery`/`loser_autopsy` negative-space-is-data precedent — silence ≠ `= max`). Distribution surface: `min/p25/median/p75/max` of min-lag-minutes, nearest-rank quantile (math.ceil-based, no banker's-rounding shift) — the `tail_risk` precedent (no interpolation). Bucket counts: `bucket_fast` (<30min) / `bucket_reactive` (30..120) / `bucket_delayed` (≥120). Verdict ladder: `REACTIVE_FAST` (median <30) / `REACTIVE` (30..120) / `DELAYED` (≥120) / `NO_ATTRIBUTION` (>50% trades lack live news — **trumps** the numeric verdict so a one-attributed-trade median doesn't whipsaw the desk-wide verdict) / `NO_DATA`. Per-trade row: ticker/action/trade_ts/top_score (highest ai_score across attributed, orthogonal to freshest)/min_lag_minutes/classification/top_title. Pure / no I/O / never raises (the `_safe` contract — None/non-dict input → NO_DATA, non-list `trades` field → NO_DATA, garbage articles in attributed list skipped, negative/None `minutes_before_trade` rejected). Served at `/api/news-to-trade-lag` (query params forwarded to `trade_attribution`: `hours_back` 1..168 default 24, `window_hours` 0.5..24 default 4.0, `max_per_trade` 1..10 default 3, `min_ai_score` 0..10 default 2.0). Observational only — never gates Opus, never injected into the decision prompt, no caps (invariants #2/#12 — the `reentry_velocity`/`churn` precedent). Locked by `tests/test_news_to_trade_lag.py` (21 tests: state ladder NO_DATA-none-input / NO_DATA-empty-trades / NO_ATTRIBUTION-all-empty-attributed; **verdict ladder boundary exactness** REACTIVE_FAST at <30 strict (30min = REACTIVE, 120min = DELAYED); NO_ATTRIBUTION **trump** rule (>50% no-attribution overrides numeric verdict — and ≤50% keeps numeric); freshest = min across attributed list (NOT the top-scored article's lag); per-trade classification each independently bucketed; per_trade sorted newest trade_ts first; nearest-rank p25/median/p75 (10/20/30/40/50 → p25=20, median=30, p75=40); degrade-never-raise on None/non-dict/non-list-trades/garbage-articles-in-attributed/negative-lag-rejected/None-minutes-handled).
- `paper_trader/analytics/news_action_funnel.py` — **per-ticker news→action funnel** (2026-05-20): for each ticker in (held ∪ watchlist), count over the last `window_hours` the loud articles ≥ score floor, the decisions naming that ticker, the FILLED trades on that ticker, and attach the current unrealized P&L for held names. The operator's "MU has 30 hot articles in the last 24h — did the bot do *anything*?" question. Orthogonal to existing surfaces: `/api/signal-followthrough` grades **forward edge** of acted vs ignored (needs 1d/3d/5d forward bars; cannot answer current window); `/api/idle-opportunity` is **drought-gated by design** (silent when the bot is filling normally — but a per-ticker miss can still happen on a single hot name); `/api/trade-attribution` is the **reverse** direction (fills → articles); `/api/watchlist-opportunities` ranks unheld heat without reporting the desk's response. None answer the per-ticker funnel *right now*, always-on. Verdict ladder per row: `IGNORED` (≥`MIN_ARTICLES_FOR_VERDICT`=3 articles AND 0 decisions — the loud-news / no-action pathology, sorted FIRST so the operator sees missed names without scrolling); `DECIDED_NO_FILL` (≥3 articles, ≥1 decision, 0 fills — Opus saw it and chose to HOLD; distinct from IGNORED); `RESPONSIVE` (≥3 articles AND ≥1 fill — operator-happy); `ACTED_WITHOUT_NEWS` (<3 articles AND ≥1 fill — bot moved on a quiet name); `QUIET` (everything else; still surfaced so held positions with no news flow remain visible). The `MIN_ARTICLES_FOR_VERDICT=3` floor is sample-size honesty: a single stale wire mention is not enough to call the desk IGNORED (matches the `news_source_mix.ECHO_MIN_ARTICLES` precedent). Sort priority: IGNORED → DECIDED_NO_FILL → ACTED_WITHOUT_NEWS → RESPONSIVE → QUIET, then within each tier by article count DESC then decision count DESC then ticker ASC. `build_news_action_funnel(articles, decisions, trades, positions, tickers, held_tickers=None, now=None, window_hours=24.0, min_ai_score=6.0, max_tickers=25)` is pure (no I/O, never raises — the `_safe` contract). Word-boundary regex `(?:\$|\b)TKR\b` (so `MU` does NOT alias `MUTUAL`, `AMD` does NOT alias `AMDOCS`, `$NVDA` cashtag still hits — the same locks as `news_velocity` / `trade_attribution` / `idle_opportunity`). NaN/Inf `ai_score` explicitly rejected (digital-intern's column has been observed with stale NaNs). Decision ticker resolution **verbatim copy** of `dashboard._parse_action_ticker` (AGENTS.md #10 — SSOT; NO_DECISION / BLOCKED / CASH / NONE sentinels never bucket under a real ticker). **Live trades table has no `status` column** — every row in `paper_trader.db.trades` is by construction a fill, so the builder treats `status` missing as FILLED; when present (test fixtures or a future schema bump) it still skips explicit non-FILLED rows. Held positions flagged via `held` and attach `unrealized_pl` for at-a-glance "the bot is dark on MY OWN ticker's news and the position is bleeding $X". Headline names the loudest IGNORED row (with HELD tag if applicable), else the worst DECIDED_NO_FILL, else a RESPONSIVE count, else "quiet window". Served at `/api/news-action-funnel` (query params: `window_hours` 1..72 default 24, `min_ai_score` 0..10 default 6.0, `max_tickers` 1..100 default 25, `tickers` comma-separated override of the universe; default is held ∪ `strategy.WATCHLIST`). Endpoint reads the digital-intern articles.db read-only via the freshness-aware `_articles_db_path()` (invariant #15/#17). Window scan is bounded by `LIMIT 12000` with the live-only SQL clause applied (invariant #3 — synthetic `backtest://` and `opus_annotation*` rows never reach the panel). Observational only — never gates Opus, never injected into the decision prompt, no caps (invariants #2/#12 — the `idle_opportunity` / `news_velocity` / `signal_followthrough` precedent). Locked by `tests/test_news_action_funnel.py` (68 tests: helper-fn discriminators (`_safe_float` NaN/Inf reject, `_ticker_regex` MU≠MUTUAL/AMD≠AMDOCS/`$NVDA` cashtag, `_parse_action_ticker` BUY/NO_DECISION/BLOCKED/CASH carve-out, `_verdict` boundaries at MIN_ARTICLES_FOR_VERDICT); state ladder NO_DATA-empty-tickers / OK-with-all-QUIET / universe-dedup; verdict ladder IGNORED / DECIDED_NO_FILL / RESPONSIVE / ACTED_WITHOUT_NEWS / QUIET / below-floor-articles-don't-count / at-MIN-boundary IGNORED / one-below QUIET; word-boundary MU≠MUTUAL/AMD≠AMDOCS/`$NVDA` end-to-end; window cutoff strict inclusive at exactly cutoff / outside dropped; decisions+trades outside window excluded; sort priority IGNORED-first / within-tier-by-article-count-DESC / max_tickers cap; held flag + unrealized_pl attach for held / None for unheld / "(HELD)" tag in headline; NaN/Inf `ai_score` rejected; NO_DECISION / BLOCKED / CASH / different-ticker decisions don't bucket; status=REJECTED trade excluded; live-trades-no-status-counted (the live shape); garbage-degrade-never-raises for None-articles / non-dict / garbage-timestamps / missing-fields / garbage-decisions / garbage-trades / garbage-positions / empty-ticker-strings; tie-break top-article newest-on-equal-score; `TestNewsActionFunnelEndpoint` Flask test_client (real seeded articles.db proving the synthetic `backtest_run_1_winner` at score 10.0 is filtered by SQL-side live-only clause, 48h-old article dropped by window cutoff, the three live MU rows surface as IGNORED at top_score=9.0; parameter clamps `window_hours`/`min_ai_score`/`max_tickers`; real Store DECIDED_NO_FILL via INSERT INTO decisions and RESPONSIVE via INSERT INTO trades using the actual paper_trader.db schema).
- `paper_trader/signals.py` — live news signal queries against digital-intern's articles.db
- `paper_trader/market.py` — yfinance wrapper + NYSE session calendar
- `paper_trader/store.py` — SQLite store (portfolio, trades, positions, decisions, equity_curve)
- `paper_trader/reporter.py` — Discord output via openclaw. `send_hourly_summary` / `send_daily_close` now append `_behavioural_block()` — the `build_trader_scorecard` verdict-alignment synthesis composed **verbatim** (single source of truth, invariant #10; same store reads as `/api/scorecard`) so the operator who lives in Discord sees the ~24 builders' synthesis without opening the (stale) dashboard. Observational only, no caps (invariants #2/#12 — the `self_review`/`scorecard` precedent). NO_DATA/ERROR suppressed; a builder/store fault degrades to *no block*, **never** *no summary* (the reporter failure contract). Applies on next paper-trader restart (the documented pattern for every recent feature). **Also appends `_session_block(store, window_h, label)`** (2026-05-17) — a compact "what the desk actually did this 1h / 24h" block: the decision-activity mix (`filled / hold / no-dec / blocked`, classified from the free-text `decisions.action_taken` via `_classify_decision_outcome` — bucket order is load-bearing so a `→ FILLED`/`→ BLOCKED` verb line is not misread as `hold`), the best/worst open mover by `unrealized_pl` (`_movers`; single position → one line via object identity), and the portfolio-vs-SPY window delta (`_window_delta`; `alpha_pct` only when both legs resolve, missing `sp500_price` degrades to port-only). All composed from existing store reads — no new state, observational only, same failure contract (store/compute fault → `""`, never an exception). The cutoff is a lexically-comparable UTC isoformat string (the `signals.py` `first_seen` pattern). Answers the trader's "did the bot do anything, and am I beating SPY this window?" from Discord without opening the (often slow/stale) dashboard. Locked by `tests/test_core_reporter.py` (`TestClassifyDecisionOutcome` / `TestActivityCounts` cutoff-inclusive boundary / `TestMovers` identity / `TestWindowDelta` exact port/spy/alpha + spy-missing degrade / `TestSessionBlock` end-to-end on a real temp Store + hourly-summary integration). **Also (2026-05-19): `_session_block` appends a true-realized-P/L line for round-trips that closed in the window** — `Closed N trip(s) (WW/LL) realized $±X.XX` — driven by a new `_realized_pl_window(trades, since_iso)` helper that mirrors `_realized_pl_today` but uses a lexical ISO comparison instead of a date-only `startswith` so arbitrary windows (1h / 4h / 24h) compose. Same SSOT as the daily close (`build_round_trips`, invariant #10) — the hourly's realized figure and the daily close's realized line can never disagree on what counts as a closed trip. Suppressed when nothing closed in the window (the silence-when-nothing-actionable precedent) so a quiet hour adds no noise; same additive failure contract (a builder fault drops just this line, never the whole report). Live verified on the 2026-05-19 book — 24h preview reads `Closed 1 trip (0W/1L) realized $-0.45`. Locked by `tests/test_core_reporter.py::TestRealizedPlWindow` (6 helper tests: nothing-closed→None, exact PnL on winners/losers, window-boundary inclusion, breakeven-not-a-win, garbage-input degrade-never-raise) and `TestSessionBlockRealizedPl` (4 end-to-end on a real temp Store: line lands with correct plural grammar, suppressed when quiet, never raises on builder fault). **Also (2026-05-17): `send_daily_close` emits an *additive* true-realized-P/L line** — `Realized P/L (today, N round-trip(s) closed, WW/LL)  $±X` — driven by `_realized_pl_today()`, which consumes `build_round_trips` (invariant #10, no re-derived P&L) filtered to round-trips whose `exit_ts` is today (UTC). It answers "what did I actually lock in today?", distinct from the pre-existing **cash-flow-basis** line (a BUY-only day reads as a large negative there — correct-by-disclosure, so that line is left untouched, not reinterpreted). Same failure contract: any fault drops just this one line (`None`), never the report. A position merely opened today does not count; an old-open/today-close trip is attributed to today because `build_round_trips` pairs BUY→SELL in ledger order (deep `recent_trades(5000)` window passed so the open leg is in scope). Locked by `tests/test_core_reporter.py::TestSendDailyCloseRealizedRoundTrips` (exact `$+70.00` on a 2-closed/1-open NVDA+MU+AMD ledger with `1W/1L`; no-line-when-nothing-closed; singular-grammar). **Also (2026-05-17, review pass #4): `_portfolio_lines` appends `⚠ STALE` when a position carries `stale_mark=True`** — additive only, `open_positions()` table rows lack the key so the existing Discord path is byte-identical (a genuinely flat $0.00 is never falsely flagged). Locked by `tests/test_core_reporter.py::TestPortfolioLines`. **Also (2026-05-18): `_host_pulse_line` — the #1 live-pathology operator surface.** The recurring multi-hour `NO_DECISION` **PARALYSIS** droughts are host saturation: the live trader's Opus call OOM-starved by out-of-band parallel Opus (review / backtest agents). Observed live 2026-05-18: a **27.8 h** drought, **70/90** cycles `NO_DECISION`, **−5.87% alpha bleed** — and *invisible from Discord*. Worse, `_capital_pulse_line` (which does reach Discord) frames the same freeze as `CAPITAL ◈ PINNED`, steering the operator to *sell a position* when the real, provable fix is OPS (kill the parallel Opus jobs — selling frees cash but the next decision still won't happen). `_host_pulse_line` composes `host_guard.pulse()` **verbatim** (single source of truth #10 — state/headline never re-derived, so this line, `/api/host-guard` and the CLI can't drift) and is appended **before** `_capital_pulse_line` in `send_hourly_summary`/`send_daily_close` so a top-down read hits the non-trading-fixable cause first; the two can be independently true and **neither suppresses the other** — the `host_guard._OPS_ACTION` discriminator (`"the bot cannot resolve this by trading"`) in the headline is what stops the operator conflating them. `host_guard.pulse()` state ladder: `SATURATED` (the live /proc probe trips now — wins regardless of log) → `STARVED` (probe clear now but ≥ `host_guard.STARVATION_RATE_FLOOR`=0.25 of recent decisions never reached Opus — an intermittent storm whose damage is in the decision log) → `CLEAR` (suppressed, nothing actionable — the `_capital_pulse_line` precedent). **Load-bearing:** the rate is `host_guard.recent_starvation_rate()`, which counts **both** the old `claude returned no response` *and* the new `skipped claude call` reasoning prefixes — `recent_empty_rate` (old prefix only) collapses toward zero precisely when the pre-flight guard is most active (the documented `/api/host-guard` trap), so keying STARVED off the empty-only rate would go dark in the exact storm it must catch. `reason` is carried verbatim from `snapshot()`; a probe failure / DB-unreadable starvation probe (`ok=False`) never trips STARVED (never cry wolf — the `host_saturated` mem==0 precedent). `pulse()` is **inert under pytest** on the real path unless a test passes the `_snapshot`/`_starv` injectors or flips `_PULSE_TEST_FORCE` (the `dashboard._swr_active` offline-invariant precedent — keeps the broad reporter integration tests deterministic on a saturated CI box). Folded into `/api/host-guard` as an additive `pulse` key (dashboard/analyst inherit the same SSOT verdict — the `tail_risk`/`stress_scenarios` additive-key precedent). Observational only, never gates, no caps (invariants #2/#12); fault → `""`, never *no summary*; applies on next paper-trader restart (the documented pattern for every recent feature). Locked by `tests/test_host_guard.py` (`recent_starvation_rate` both-prefix divergence asserted *against* `recent_empty_rate` on the same DB; `TestPulse` — SATURATED-wins-regardless, floor-boundary inclusive, unreadable-log-never-cries-wolf, verbatim-`reason` no-drift, degrade-safe-never-raises, inert-under-pytest + force-flag override) and `tests/test_core_reporter.py::TestHostPulseLine` (CLEAR→`""`, SATURATED/STARVED headline verbatim incl. the OPS discriminator, fault→`""`, daily-close wiring, and the load-bearing **HOST-before-CAPITAL ordering** lock in `send_hourly_summary` with both lines independently emitted). **Also (2026-05-19): `_next_session_line` — off-hours orientation cue for the Discord operator.** A trader checking the hourly at 2 AM ET or on a weekend saw positions + stale marks with zero "when can I act next?" cue. Composes a new pure `market.next_session_open(now)` helper **verbatim** (single source of truth — the NYSE_HOLIDAYS_2026 calendar now has one consumer in market.py read by `is_market_open`/`close_minute`/this) into a single `**MARKET** ◈ closed — next session: Mon 05-18 09:30 ET (in 2d 16h)` line. Wired into `send_hourly_summary` ONLY (daily-close fires post-bell so the line would always read "tomorrow" — noise), suppressed when the market is currently open (in-session implies trading — the `_capital_pulse_line` FREE/not-bleeding suppression precedent). `next_session_open` is pure / zero-I/O / never raises; walks the weekday + holiday calendar with a 14-day forward bound, and **advances past today** when we are at-or-past 09:30 ET so a mid-session check returns tomorrow's open (never "today 09:30 in the past"); a half-day session is still an open day from this helper's perspective (only the close is early). Observational only, no caps (invariants #2/#12 — the `_host_pulse_line` precedent); `_countdown` clamps negative skew to "in 0m" so a tick of clock drift never renders "in -3m". Locked by `tests/test_core_market.py::TestNextSessionOpen` (9 tests: Fri-close→Mon, Sat/Sun→Mon, pre-open today→today, at-open→tomorrow, mid-session→tomorrow, Thanksgiving skip, Good-Friday weekend+holiday skip, UTC-aware tz tag) and `tests/test_core_reporter.py::TestNextSessionLine` + `TestCountdown` (13 tests: market-open suppression, Fri→Mon countdown, Sat→Mon, pre-market today→30m, holiday skip, helper-never-raises on builder fault, **wired into hourly when closed**, **NOT wired when open**, plus the H:M / D:H bucket shape and negative-clamp)
- `paper_trader/dashboard.py` — Flask dashboard on :8090
- `paper_trader/backtest.py` — backtest engine, `_ml_decide`, indicators
- `paper_trader/ml/decision_scorer.py` — MLP that gates trade conviction
- `paper_trader/should_restart.py` — **operator CLI** (2026-05-18): one-shot "should I restart this trader right now?" verdict. Composes `/api/supervision` + `/api/runner-heartbeat` + `host_guard.pulse()` into ONE answer printable from a plain shell (mirrors the `python3 -m paper_trader.host_guard` / `python3 -m paper_trader.signals --check-freshness` pattern). State ladder `OK`(exit 0) → `OPS_ONLY`(exit 2; host saturated — restart alone will NOT help, kill out-of-band Opus first) → `RESTART`(exit 1; stale code / wedged loop / Discord dark / lock degraded / dashboard unreachable) → `ERROR`(exit 3; no diagnostic source reachable). **Read-only — never restarts anything**, never writes. Closes the manual-fusion gap an operator hits weekly: three endpoints each diagnose a different facet of the same recurring live pathology and the operator has to consult all three by hand. The pure `build_should_restart` builder consumes already-built verdict dicts (no I/O, never raises — the `_safe` contract); the CLI's `gather()` does degrade-safe HTTP fetches with `host_guard.pulse()` fallback so the command works **when the dashboard is itself down** (which IS a signal — the runner is the most likely cause, so `RESTART` is the correct answer). `--json` for machine consumption (composes in shell guards: `python3 -m paper_trader.should_restart || systemctl --user restart paper-trader`). Locked by `tests/test_should_restart.py` (20 exact-value tests: OK/RESTART/OPS_ONLY/ERROR state ladder, exit-code contract, restart-wins-state-but-ops-action-first ordering, dashboard-unreachable→RESTART, degrade-safe fetch on connection error + garbage payload, --json output)
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

# Operator CLIs (read-only, plain shell, work when the dashboard is down)
python3 -m paper_trader.should_restart        # one-shot 'should I restart?' verdict (exit 0/1/2/3)
python3 -m paper_trader.host_guard            # 'is the box overloaded right now?' (exit 0/1)
python3 -m paper_trader.signals --check-freshness   # article-DB split-brain detector (exit 0/2/3)
```

The runner starts a daemon thread for the Flask dashboard on `:8090` and
posts a `**PAPER TRADER ONLINE**` ping to Discord on first boot.

The three CLIs above compose into the daily operator loop: `should_restart`
fuses the verdict from the other two with `/api/supervision` + the heartbeat
+ Discord delivery health into a single answer (`OK` / `OPS_ONLY` / `RESTART`
/ `ERROR`). When `RESTART` and `OPS_ONLY` both apply, the OPS action is
listed first — a freshly-booted runner re-starves immediately if you restart
without first reducing concurrent Opus load.

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
| `test_stress_scenarios.py` | `build_stress_scenarios` — the forward beta/concentration shock (day-one complement to the history-gated `tail_risk`). The discriminating locks: **SSOT no-drift** — the −3 % market scenario is asserted equal to an *independent* recompute of `/api/risk`'s `Σ −0.03·β·val` shock (a drift in either fails loudly); **exact hand-computed $** for every family on a pinned 2-name book (an off-by-sign / dropped-β is caught, not "no crash"); **strictly monotone** |loss| −1→−3→−5→−10 %; **option-β path** (×3 cap 4, **negated for puts** — a put book *gains* on a sell-off); **no sample-size gate** (the whole point vs `tail_risk`) verified by an `OK` verdict on a one-position book; `_safe`/`NO_DATA` (empty/None/zero-book/garbage-row/`classify`-raises → honest degrade, never an exception); `_build_payload` renders the block **after** `sector_exposure` and **before** `event_calendar`/`WATCHLIST PRICES`, `None` → no stray text; `TestReporterStressLine` — `_stress_line` is `""` on NO_DATA/fault, emits the builder headline **verbatim** otherwise, and `send_hourly_summary`/`send_daily_close` still send when the builder faults ("no block, never no summary"); `TestStressScenariosEndpoint` drives the real `/api/stress-scenarios` + `/api/analytics` Flask views on a fresh temp `Store` → both equal the builder recomputed with the dashboard's own `_classify`/`_LEVERAGE_BETA` (no hardcoded sector literals → robust to a `SECTOR_MAP` change), empty book → `NO_DATA` not 500; `TestBetaMapIsPinnedToDashboard` pins `_LEVERAGE_BETA == dashboard._LEVERAGE_BETA` and `sector_exposure.classify == dashboard._classify` (the hot-path-no-dashboard-import discipline) |
| `test_recovery.py` | `build_recovery` — the forward "path back to even" (forward complement to the backward `/api/drawdown`). **Exact hand-computed** per-position breakeven %/$ + book to_initial/to_peak %/$ on a pinned underwater book (a sign-flip / dropped figure is caught, not "no crash"); **option breakeven is the raw price ratio, NOT ×100** — an option lot at avg 2.00 / now 1.00 → `+100.0%` (a reviewer multiplying through → `10000.0`, RED — the advisor discriminator); a winner → `0.0` (never negative noise); **σ no-drift** — the σ-day figure recomputed from an independent `annualized_vol_pct/√252` equals the builder's; **dispersion honesty gate** — `tail_risk.state != OK` still emits %/$ but **withholds σ** with the honest sentence *even though the numeric exists* (the young-book precedent), `OK` emits it; state ladder `NO_DATA`/`ABOVE_WATER` (line-suppressed, incl. the empty-book `compute_drawdown` starting-equity fallback)/`UNDERWATER`; garbage rows / None / missing-peak never raise; `TestRecoveryEndpoint` drives the real `/api/recovery` + `/api/analytics` Flask client on a fresh temp `Store` → endpoint == `recovery` fold == builder over the **same** `compute_drawdown`+`build_tail_risk` (no drift), empty book → `ABOVE_WATER` not 500; `TestReporterRecoveryLine` — `""` on NO_DATA/ABOVE_WATER/fault, builder headline **verbatim** on `UNDERWATER`, and `send_hourly_summary` still sends when the builder faults ("no block, never no summary") |
| `test_dashboard_threaded.py` | invariant #7 dashboard-concurrency lock. `test_run_passes_threaded` regression-locks the `dashboard.run` call site (monkeypatched `app.run`): `threaded=True` is passed **and** the existing `debug=False`/`use_reloader=False` hardening is preserved (RED before the 2026-05-17 fix — the kwarg was absent, so the in-process Werkzeug dev server served one request at a time and a single slow yfinance-backed endpoint head-of-line-blocked every concurrent panel / `/api/chat` fan-out / `:8080→:8090` cross-fetch). `test_threaded_server_parallelizes` is the behavioural lock: an independent ephemeral-port `make_server(..., threaded=True)` with a 0.4s route serves 4 concurrent requests in well under the serial 1.6s — so a future swap to a non-threaded WSGI entry point that silently drops the property is caught even though the monkeypatch lock still passes. Offline, deterministic, no real `:8090` bind. Found by user-perspective testing, not code review |
| `test_should_restart.py` | `build_should_restart` operator-verdict CLI (2026-05-18) — `TestBuilderOK` (all-healthy → OK exit 0); `TestBuilderRestart` (STALE supervision, heartbeat `restart_recommended`, degraded singleton lock = double-trade risk, Discord notify dark, dashboard unreachable each independently → RESTART exit 1); `TestBuilderOps` (SATURATED/STARVED host alone → OPS_ONLY exit 2; ops-action-listed-first even when RESTART wins state — a freshly-booted runner re-starves immediately if you restart without clearing load); `TestBuilderError` (every input None → ERROR exit 3; one input present → still resolves); `TestHeadlineCount` ("(+ N more)" tail when multiple restart reasons); `TestRender` (OK omits sections, RESTART includes systemctl line, OPS_ONLY explicitly LACKS systemctl — the discriminator); `TestFetchDegradeSafe` (connection error / garbage non-JSON payload both return None, never raise); `TestMain` (CLI returns 0/1, `--json` round-trips through `json.loads` to the verdict shape) |
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

20. **NEVER call `store._connect()` from a test.** Even when a fixture
   monkeypatches `store.DB_PATH`, `_connect()` reads `DB_PATH` at call
   time from the module global — fine in principle — but if your test
   bypasses the established `fresh_store` fixture (see
   `tests/test_news_velocity.py:325` for the canonical pattern that
   first monkeypatches `store.DB_PATH = tmp_path / "paper_trader.db"`
   *then* instantiates `Store()`) and calls `_connect()` directly with
   no path redirection, it opens the **LIVE** `data/paper_trader.db`.
   Any DDL/DML the test runs (`DELETE FROM decisions`, schema add, etc.)
   then **silently wipes production data**. Caught live 2026-05-19 by
   the very review pass adding `tests/test_blocked_repeats.py` — a
   `_FreshStore` stub called `_connect()` it did not actually use and
   the subsequent `DELETE FROM decisions` truncated 96 rows of historical
   decisions before the bug was noticed (trades/positions/equity_curve
   untouched; the rogue DELETE was scoped to one table). Two compliant
   patterns:
     - **Reuse the canonical fixture** — depend on `fresh_store`
       (`tmp_path` + `monkeypatch(store.DB_PATH, …)` + `Store()`); see
       `test_news_velocity.py` / `test_news_source_mix.py`.
     - **Pure stub** — when the endpoint only reads (`recent_decisions`,
       `open_positions`, …), back the stub with a Python list and skip
       sqlite entirely. See
       `tests/test_blocked_repeats.py::TestEndpoint._setup_app`.
   If you must touch sqlite directly, monkeypatch `store.DB_PATH` first
   and the autouse `_isolate_data_dir` fixture in `conftest.py` should
   handle the rest. The autouse fixture deliberately does NOT redirect
   `store.DB_PATH` (that's owned by `fresh_store`), so a stray
   `_connect()` slips past it.

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
| `GET /ticker/<sym>` | HTML — **standalone per-ticker drill-down page** (2026-05-18). Deliberately NOT a new tab in the 9k-line SPA `TEMPLATE` (so it can't merge-conflict that file); a self-contained dark page that fetches `/api/ticker/<sym>` client-side. Has a ticker box to jump between names; links honour `X-Forwarded-Prefix` so it works behind the `:8888` proxy. |
| `GET /api/ticker/<sym>` | **Cross-system dossier for one name** — fuses the live lot + marks, the closed round-trip P&L *for that name only*, the Opus decision trail that touched it, and the live news flow + sentiment. Closes the "inspecting MU means crossing three surfaces by hand" gap. Pure SSOT `analytics/ticker_dossier.py::build_ticker_dossier` (no DB handle / no network / no yfinance — only stored marks + the read-only `signals` reads that already self-degrade on a locked feed). **Intentionally NOT `@swr_cached`:** that decorator keys on the query string only, so a `<sym>` *path* param would collide across tickers (serve MU's dossier for NVDA); the endpoint is lighter than the un-cached `/api/portfolio` peer so the hot path stays bounded without SWR. `has_coverage:false` ⇒ nothing on file anywhere (typo'd ticker shows a clean empty state, not an error). Locked by `tests/test_ticker_dossier.py` (exact aggregates) + `tests/test_ticker_endpoint.py` (Flask test client, incl. a two-ticker no-collision regression) |
| `GET /api/watchlist-opportunities` | **Missed-opportunity radar** (2026-05-18) — watchlist names with live news heat that the book has **no position in**. Orthogonal to every existing position-centric panel (drawdown / track-record / thesis-drift all describe what's held; this answers "what is the news screaming about that I have zero exposure to?"). One `get_top_signals` fetch; the pure SSOT `analytics/watchlist_opportunities.py::build_watchlist_opportunities` tallies per ticker (no N-query fan-out). `heat = max_score·(1+ln(1+n)/3)·(1+0.25·urgent)`. `@swr_cached("watchlist-opportunities", 60.0)` (no path param, safe to cache). Locked by `tests/test_watchlist_opportunities.py` (exact heat arithmetic + held-exclusion) |
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
| `GET /api/sector-signal-fit` | **Are you allocated where the wire is pointing?** — per-sector position weight vs. live-signal-density divergence. `/api/sector-exposure` shows position concentration; `/api/news-deduped`/`/api/signals` show what the wire is talking about. Neither answers the discretionary-PM question: am I overweight a sector the wire has moved on from, or underweight one where coverage is accelerating? Composes `build_sector_exposure(snap, names)` (single source of truth — no re-derived position weights) with a sector-weighted ai_score rollup of the last `?hours=` (default 6) of live signals (`signals.get_top_signals`, the same window the trader sees). A multi-sector signal **splits weight evenly across mentioned sectors** (so a 4-ticker SEMIS article contributes ONCE to SEMIS, not 4× — discriminator vs. a naive per-mention sum that would distort the share); a signal with zero extracted tickers contributes nothing (counted separately as `n_signals_with_no_tickers` for honesty, never absorbed as "other-sector noise"). Per-sector verdicts (precedence): `OVERWEIGHT` (`gap_pct > +gap_threshold_pct`, default ±15 — long but wire is quiet, de-risk candidate) / `UNDERWEIGHT` (`gap_pct < −threshold` — wire is loud but no position, lean-in candidate) / `ALIGNED`. Top-level `state` ∈ `ALIGNED`/`MISALIGNED`/`NO_DATA` (ALIGNED iff every sector within threshold). `signal_share_pct` ALWAYS sums to ~100.0 across surfaced sectors (the denominator is total weighted ai_score — never a fabricated fraction that doesn't add up). Sector classification reuses `analytics.sector_exposure.classify` (the verbatim mirror of `dashboard._classify`, drift-locked) so the position-side and signal-side columns share one SECTOR_MAP. Rows sorted descending by `\|gap_pct\|`, ties by sector name (worst-divergence first). Knobs echoed in the response (`window_hours`, `min_score`, `gap_threshold_pct`, `n_signals_input`) so the payload is self-describing. **Advisory only — never gates Opus, never fed into the decision prompt** (invariants #2/#12; observational sibling to `/api/sector-exposure` which IS in the prompt — this one stays dashboard/chat-side only). Pure core: `analytics/sector_signal_fit.py::build_sector_signal_fit` (never raises; bad inputs degrade to `NO_DATA` or empty rows). Locked by `tests/test_sector_signal_fit.py` (27 tests: signal-share-sums-to-100, multi-sector-single-contribution, cross-sector-split, no-tickers-counted-separately, zero/negative-score-dropped, unknown→other, ranking, threshold flip, NO_DATA/MISALIGNED honesty, garbage-rows-degrade, Flask end-to-end). Applies on next paper-trader restart (`/api/build-info` `stale`) |
| `GET /api/news-deduped` | Top signals after dedup + exponential urgency decay |
| `GET /api/position-thesis` | Per-position cards combining scorer + technicals + news + last decision. Each card carries `off_distribution` + `raw_pred_5d_return_pct` so the unified conviction board can decay its ML axis off the explicit flag (not a re-derived magnitude heuristic) |
| `GET /api/position-runrate` | **Per-position P/L pace + verdict — "is this position bleeding faster than I'd tolerate, or actually working?"** The complementary, *pace*-first view to `/api/position-thesis` (verdict-from-scorer) and `/api/hold-discipline` (aggregate-disposition). Returns per-position `runrate_per_day_usd` (= `unrealized_pl / hold_days`), `annualized_pct` (= `pl_pct × 365/hold_days`), `projected_pl_30d_usd`, plus an aggregate book-wide pace headline and a `worst_runrate` pointer. Verdict bands key off **annualized return %** so a $1000 book and a $100000 book read the same scale: BLEEDING (≤ −100%/yr), FLAT (in band), WORKING (≥ +25%/yr), FRESH (< 1h hold — slope dominated by intraday noise), UNKNOWN (stale mark — price unavailable, verdict withheld). Pure arithmetic over `store.open_positions()` + `portfolio.total_value` — NO network, NO extra store reads (the `risk_mirror` hot-path discipline). The endpoint joins `opened_at` from the open-positions table into the persisted `positions_json` snapshot since the snapshot strips that field on persist. Observational only — never gates Opus (invariants #2/#12), never injected into the decision prompt. Pure core: `analytics/position_runrate.py::build_position_runrate` (never raises). Locked by `tests/test_position_runrate.py` (16 tests: per-day-pace math, annualized verdict bands, stale-mark yields UNKNOWN not falsely FLAT, sub-1h fill yields FRESH not noise-driven BLEEDING, future opened_at clamps to zero, options ×100 multiplier in book-weight math, aggregate sums only valid rows, headline composition). Applies on next paper-trader restart (the documented stale pattern — `/api/build-info` `stale`/`behind`) |
| `GET /api/calibration` | Confidence-bucket win rate + signal-source attribution |
| `GET /api/drawdown` | Drawdown anatomy: peak/trough, time-in-DD, per-position contribution |
| `GET /api/benchmark` | **"Is this bot worth running vs just buying the index?"** — the trader's *first* question, with no home until now. Whole-account return (cash + open + every realised round-trip + unrealised mark) since the first equity write vs the **identical starting capital invested once in the S&P 500 at that same instant and held untouched**. The figure is the `^GSPC` *index level* recorded on every `equity_curve` write from cycle one (~7400 — **not** the SPY ETF; the module says "S&P 500" everywhere, never "SPY", so a 7400 mark is never mislabelled $620). **Distinct from its neighbours — do not "consolidate" (invariant #10):** `/api/open-attribution` is per-**open**-lot alpha *since each lot's entry* (blind to realised P&L / cash drag, resets per re-opened lot, invariant #8); `/api/analytics` `sp500_beta` is a *statistical regression* needing many daily points (`null` on the live book). This is the full-account dollar answer, defined from cycle 1, no regression, no per-lot windowing. Outputs `port_return_pct`/`sp500_return_pct`/`alpha_pp`, `sp500_equivalent_usd`, `usd_vs_sp500`, `pct_cycles_ahead`, running best-lead/worst-lag extremes + a down-sampled (≤200, last point always pinned — strictly bounded, unlike `drawdown.py`'s `+[hist[-1]]` which can overshoot to 201) cumulative-alpha `history`. Sample-size honest like `news_edge`/`trade_asymmetry`: `NO_DATA` (no row with both a value and an S&P mark) → `INSUFFICIENT` (< `_MIN_SPAN_HOURS`=24h **or** < `_MIN_POINTS`=12 benchmarkable points — numerics emitted, **verdict withheld**) → `OK` with verdict `BEATING`/`LAGGING`/`TRACKING` (`\|alpha\|` ≤ `_TRACK_BAND_PP`=0.5pp → TRACKING). The inception anchor is the **first row carrying both a value and an S&P mark** (yfinance cold-start robustness), not blindly `equity_curve[0]`. `headline` is the single source of truth the endpoint, the **`python -m paper_trader.analytics.benchmark [--json]` CLI** (the `desk_pulse`/`signals --check-freshness` precedent — answers from a terminal when `:8090` is wedged/slow; verified live while `/api/state` was timing out) and the Discord line all render verbatim so they can never drift. Endpoint passes the module `INITIAL_CASH` (invariant #12, never a literal 1000). Advisory only — never gates Opus, adds no caps, **not** injected into the decision prompt (invariants #2/#12; the `desk_pulse`/`self_review` observational precedent). Pure core: `analytics/benchmark.py::build_benchmark` (never raises — a malformed row degrades, the contract is "no benchmark this cycle", never an exception). Locked by `tests/test_benchmark.py` (hand-computed BEATING/LAGGING/TRACKING + the **real 2026-05-17 live-book shape** `^GSPC 7444.88→7409.18`, $1000→$972.69 → `−2.25pp / −$22.52` arithmetic lock; NO_DATA/INSUFFICIENT honesty; first-usable-anchor robustness; invariant #12 init=2000 lock; never-raises-on-garbage; history strictly ≤200; reporter line composes the headline verbatim & a builder fault drops only its line while the hourly summary still sends; endpoint e2e via the Flask test client cross-checked equal to the builder on the same store). **Reporter:** `reporter._benchmark_line` appends a `**BENCHMARK** ◈ vs S&P 500 buy-and-hold` block to the hourly + daily-close summaries (composed verbatim, `NO_DATA` suppressed, same "no block, never no summary" failure contract as `_session_block`/`_behavioural_block`). Applies on next paper-trader restart (the documented pattern for every recent feature) |
| `GET /api/earnings-risk` | Upcoming earnings ⨯ held positions / watchlist, tiered (network-sourced from `:8080`; dashboard view, exposure-$ weighted) |
| `GET /api/event-readiness` | **Will the live trader actually be able to react before the next earnings print?** `/api/earnings-risk` says NVDA reports in 16h, `/api/decision-drought` says the bot is in a 4.7h PARALYSIS streak, `/api/empty-claude-rate` says Claude returned empty on ~48% of recent cycles — each is half the picture and none of them answers the operator's actual pre-print question: given those three facts *together*, is the bot statistically going to land a usable decision in time? Distinct from each ingredient (invariant #10 — do not "consolidate"): `earnings-risk` is the calendar slice, `decision-drought` is the historical PARALYSIS ledger, `empty-claude-rate` is the rolling-rate diagnostic — none compose them into a per-held-event verdict. `analytics/event_readiness.py::build_event_readiness` is pure; the endpoint is the I/O seam (pulls earnings events from `:8080/api/earnings` exactly as `/api/earnings-risk` does, decisions from the live store). For each held ticker with earnings in next 72h, computes `expected_decisions_before_event = cycles_per_hour × hours_until_event × (1 − empty_rate)` over the last 6h, classifies `READY` / `DEGRADED` (≥5 but <30 expected) / `BLIND` (<5) / `IMMINENT_OVERDUE` (event already passed — flags a stale calendar). The expected-decisions math is a *historical* average; a `_apply_streak_downgrade` rule then penalizes an **active** NO_DECISION streak (≥10 → READY→DEGRADED, ≥20 → any→BLIND) — that's the live PARALYSIS regime `/api/decision-drought` catches, and a 6h average can honestly read OK while the most-recent 30 cycles are all empty. Per-event `recommended_action` is operator-actionable (`"ACTIVE PARALYSIS streak: 25 NO_DECISION cycles ({M}m since last real decision) — restart paper-trader before the print, or pre-trim exposure"`) — the chat enrichment carries this verbatim, never re-derives. Portfolio-level `worst_verdict` + `exposure_at_risk_usd` (sum of BLIND+DEGRADED rows only) + a builder-own `summary` (the SSOT chat headline). Observational only — never gates Opus (invariants #2/#12), never injected into the decision prompt. Pure core never raises on garbage rows (the `_safe` discipline); `source_ok` flags a missed `:8080/api/earnings` fetch (the endpoint still returns 200 with `worst_verdict=NO_EVENTS`). Locked by `tests/test_event_readiness.py` (19 tests: BLIND/DEGRADED/READY ladder with calibrated cycle fixtures; the headline streak-downgrade discriminators — `test_streak_downgrades_ready_to_blind` proves a base READY → BLIND when newest 30 cycles are NO_DECISION, `test_streak_downgrades_ready_to_degraded` for the 10–19 streak band; NO_EVENTS / NO_DECISIONS / IMMINENT_OVERDUE degrade paths; horizon + unheld-ticker filtering; option 100× multiplier; `_decision_velocity` window-only counting + NO_DECISION classification SSOT with `analytics/decision_drought.py::_classify`; malformed-row never-raises). **Chat:** digital-intern `dashboard/web_server.py::api_chat` adds a guarded 3s `:8090/api/event-readiness` sub-fetch composed verbatim by the pure unit-tested `_event_readiness_chat_lines` helper (SSOT — no re-derived verdict; READY/NO_EVENTS/NO_DECISIONS collapse to silence so a healthy pipeline is *silence*, never chat filler), under a `PAPER TRADER — EVENT READINESS` block so the analyst answers "is the bot going to be able to act before the print?" before the user has to ask. Applies on next paper-trader / digital-intern restart |
| `GET /api/event-calendar` | **The exact upcoming-earnings block the live trader now sees in its prompt** — the forward complement to the backward-looking behavioural mirrors. `analytics/event_calendar.py::build_event_calendar` over digital-intern's `earnings_calendar.json` snapshot read **directly from disk** (no `:8080` hop — the documented live-cycle hang hazard), `days_away` recomputed vs `now`, tiered `HELD_IMMINENT`/`HELD_SOON`/`WATCH` exactly as `/api/earnings-risk` (single source of truth #10). Distinct from `/api/earnings-risk` (that one is network-sourced + exposure-weighted for the dashboard; this one is the on-disk, prompt-parity, `_safe`-degrading view). Observational only — never gates Opus (invariants #2/#12). Locked by `tests/test_event_calendar.py::TestEventCalendarEndpoint`. Applies on next paper-trader restart |
| `GET /api/scorer-confidence` | Empirical residual bands + directional hit-rate for DecisionScorer |
| `GET /api/baseline-compare` | **Does the 17-feature DecisionScorer earn its complexity OUT OF SAMPLE, or would a one-line rule do as well?** The honesty signal `data/run_log.md` flagged (`MLP_NO_BETTER_THAN_TRIVIAL` — pure-memorization fingerprint: in-sample `MLP_ADDS_SKILL`, OOS rank_ic +0.06 below the 0.10 floor) but that had **no surface** — it lived only in `python3 -m paper_trader.ml.baseline_compare`, a CLI no operator runs, while every analytics endpoint the dashboard/chat already show report the *flattering in-sample* story. Thin SWR-cached (`baseline-compare`, 90s; mirrors `scorer_confidence_api` — loads the pickled scorer + `_load_decision_outcomes()`) wrapper over `ml/baseline_compare.scorer_baseline_compare(scorer, outcomes, oos_only=True)` — the **temporal-OOS slice** (the EXACT split `_train_decision_scorer` uses for `oos_rmse`, so the MLP's `rank_ic` here equals `calibration --oos`'s — a built-in cross-check), never the in-sample view. Scores the deployed MLP and trivial one-liners (`ml_score`, `mom20`, `mom5`, `rsi_meanrev`, `neg_bb`) on two scale-invariant primitives (tie-aware Spearman `rank_ic`, `dir_acc`) with the codebase-universal SELL sign-flip applied to *every* predictor. **Distinct from its neighbours (invariant #10 — do not "consolidate"):** `calibration` = is pred monotone with realized (deciles); `scorer-confidence` = empirical residual bands; `skill_trend` = MLP vs a *constant* predictor cycle-over-cycle; this = MLP vs a *non-constant one-line rule* — the decisive "is the net additive, or memorizing?" question none of them ask. Verdict ladder: `INSUFFICIENT_DATA` (untrained / < `MIN_PAIRS`=30 OOS pairs — withheld) → `MLP_WORSE_THAN_TRIVIAL` → `MLP_NO_BETTER_THAN_TRIVIAL` → `MLP_ADDS_SKILL` (clears best baseline + `IC_MARGIN`=0.05 AND its own `MLP_IC_MIN`=0.10 floor). Carries the module's verbatim `hint`, `mlp`/`baselines`/`best_baseline`/`ic_gap`, `slice`, `n`, `n_train`. **A read-only diagnostic, NOT a recommendation:** invariant #5 keeps the gate live at `n_train ≥ 500` regardless of this verdict — the value is *knowing* the gate is modulating real position sizing on noise, never an instruction to disable it (the card copy + chat framing both state this explicitly). Never raises into a panel — a load fault degrades to a verdict-keyed `INSUFFICIENT_DATA` body (HTTP 200, not a bare 500) so the card/chat never KeyError. Pure core: `paper_trader/ml/baseline_compare.py` (20 unit tests, `tests/test_baseline_compare.py`). Endpoint + prewarm wiring + card-wiring locked by `tests/test_baseline_compare_endpoint.py` (faithful-thin-wrapper equality, OOS-slice, CORS, never-raises, the `/` card+JS+interval regression-lock); `tests/test_swr_prewarm_coverage.py` auto-locks it into `_swr_prewarm` (same restart cold-stall contract — this pass also added the long-missing `stress_scenarios`/`watchlist-opportunities` prewarm targets it exposed). **UI:** `bc-card` panel on the `:8090` trader page (verdict-coloured badge, verbatim hint headline, MLP/best-one-liner/IC-gap/n_train stats; green only once MLP rank_ic clears the 0.10 real-skill floor); JS degrades via the `/api/build-info` `stale` contract. **Chat:** digital-intern `dashboard/web_server.py::api_chat` adds a guarded 3s `:8090/api/baseline-compare` sub-fetch composed verbatim by the pure unit-tested `_baseline_compare_chat_lines` helper (SSOT — no re-derived verdict; `INSUFFICIENT_DATA` collapses to one withheld line and never leaks the exception-string `hint`), under a `PAPER TRADER — ML GATE HONESTY` block so the analyst answers "is the bot's ML edge real?" with the generalization-relevant verdict instead of the in-sample one. Applies on next paper-trader / digital-intern restart (the documented stale-process pattern) |
| `GET /api/decision-health` | Action mix, NO_DECISION parse-failure rate, confidence trend |
| `GET /api/decision-forensics` | *Why* NO_DECISION: failure-mode taxonomy (timeout/truncated/no-json/fenced/prose/malformed/legacy), open-vs-closed split, hourly trend, retry-exhausted count, actionable hint + raw Opus excerpts |
| `GET /api/decision-paralysis` | **Consecutive HOLD streak detector — the HOLD_LOCK pathology.** `runner_heartbeat` only fires on `NO_DECISION` storms (IDLE_STORM ≥5); `decision_health` reports a 24h HOLD% aggregate that looks identical whether spread across the day or stacked into one immovable block. Neither detects the other "loop is alive but nothing moves" failure mode: a contiguous run of pure-HOLD decisions where Opus is deciding every cycle and never moving the book. Verdict ladder (most-specific first): `IDLE_STORM` (current NO_DECISION run ≥ `IDLE_STORM_THRESHOLD`=5, mirrors `runner_heartbeat.NO_DECISION_STORM_THRESHOLD`) → `HOLD_LOCK` (current HOLD run ≥ `HOLD_LOCK_THRESHOLD`=10, ~1h at OPEN_INTERVAL_S=1800s cadence — an order of magnitude beyond a normal cluster) → `PASSIVE_LOOP` (mixed HOLD ∪ NO_DECISION run ≥ `PASSIVE_LOOP_THRESHOLD`=15 but neither narrower band fires) → `ACTIVE` / `NO_DATA`. Carries `current_hold_streak` / `current_no_decision_streak` / `current_passive_streak`, `longest_*_24h` (contiguous 24h-bounded), `last_active_action` + `hours_since_last_active` for the most recent FILLED/BLOCKED row, plus the three thresholds so a UI can render the gap to the next verdict band without hardcoding module constants. Pure core: `analytics/decision_paralysis.py::build_decision_paralysis` (over `store.recent_decisions(500)`). `_classify` is a verbatim mirror of `decision_health._classify`'s outcome bucket (drift-locked by `tests/test_decision_paralysis.py::test_classify_mirrors_decision_health_classify`). Observational only — never gates Opus, no caps (invariants #2/#12). SWR-cached 30s. Locked by `tests/test_decision_paralysis.py` (17 tests: bucket predicate + decision_health drift-lock; leading/longest run helpers; empty/active/HOLD_LOCK/IDLE_STORM/PASSIVE_LOOP ladder + ladder precedence; 24h window contiguous bound; `last_active_ts` / `hours_since_last_active` carry; garbage-row degrade-never-raises; thresholds exposed in output; Flask test_client endpoint integration with SWR cache reset). **Chat:** digital-intern `dashboard/web_server.py::api_chat` adds a guarded 3s `:8090/api/decision-paralysis` sub-fetch composed verbatim by the pure unit-tested `_decision_paralysis_chat_lines` helper (SSOT — no re-derived verdict; ACTIVE/NO_DATA collapse to silence — never chat filler), under a `PAPER TRADER — DECISION PARALYSIS` block so the analyst answers "should I be doing something?" with the verdict the other panels' aggregates hide. Applies on next paper-trader / digital-intern restart |
| `GET /api/liquidity` | Capital deployment & liquidity: cash vs deployed %, position weights, unrealized P/L, days-since-last-entry, status (NO_DRY_POWDER/DRY_POWDER_LOW/BALANCED/CASH_HEAVY) + flags |
| `GET /api/build-info` | Code-freshness probe: `{boot_sha, head_sha, behind, stale}`. `stale: true` ⇒ this `:8090` process booted before the on-disk HEAD — committed fixes (e.g. the DecisionScorer ±50 clamp) are NOT applied until restart. The unified dashboard's landing banner reads this + its own to flag stale processes |
| `GET /api/decision-drought` | What the trader's *inaction* cost. Segments cycles into droughts between FILLED trades; per drought: duration, NO_DECISION/HOLD/BLOCKED mix, portfolio Δ% vs S&P Δ% over the idle window, alpha. Splits involuntary `PARALYSIS` (NO_DECISION-dominated) from `DELIBERATE_HOLD`; `involuntary_alpha_bleed_pct` sums the **negative alpha of PARALYSIS droughts only** (DELIBERATE_HOLD drift is a strategy choice, excluded). Complements decision-forensics (*why*) with the *cost*. DB-only, no network. Pure core: `analytics/decision_drought.py::build_decision_drought` |
| `GET /api/no-decision-reasons` | **Why isn't the bot deciding?** — buckets the WHY of recent NO_DECISION cycles into operator-distinct buckets (`quota_exhausted`, `host_saturated`, `model_empty`, `parse_failed`, `retry_failed`, `other`) using the canonical reasoning prefixes `strategy.decide()` writes per row. `state ∈ NO_DATA / DOMINANT / MIXED`: DOMINANT only when one bucket holds ≥50% of NO_DECISION rows in the window, and each dominant bucket carries a TARGETED recommendation (quota → "wait for reset, restart does NOTHING"; host_saturated → "kill parallel Opus jobs, restart does NOTHING"; model_empty → "wedged CLI, restart IS the fix"; parse/retry → "prompt regression"). MIXED suppresses the recommendation — no single fix is right. Closes the gap `runner-heartbeat` IDLE_STORM left: a generic "restart may help" line on storms that are actually quota/host-saturation outages restart cannot fix. Query: `?window=N` (1..500, default 50). Pure core: `analytics/no_decision_reasons.py::build_no_decision_reasons`. **Discord:** `reporter._no_decision_reasons_line` routes the DOMINANT verdict + recommendation into the hourly / daily-close summary (suppressed on NO_DATA / MIXED — the `_heartbeat_line` HEALTHY-suppression precedent so a balanced histogram never claims a wrong lever). Locked by `tests/test_no_decision_reasons.py` (26 tests covering bucket predicate, state ladder, dominant-recommendation correctness, MIXED suppression, window truncation, reporter integration). Pin: the `quota_exhausted` recommendation MUST NOT contain "restart the runner" (that misdirects on an org-level usage limit) — pinned by `TestDominant::test_quota_dominant_recommends_waiting` |
| `GET /api/news-edge` | Does a high-`ai_score` headline actually predict the move? Per live (non-backtest) scored article naming a watchlist ticker, 1/3/5-trading-day forward return — raw **and SPY-abnormal** — banded by ai_score; verdict judged on abnormal return only. `?days=` (lookback, default 30) / `?min_score=` (default 2.0). Reference horizon is **adaptive**: the longest horizon whose top band is well-sampled, falling back to 1d early on — so the verdict *matures with article history* (digital-intern's `articles.db` only retains a few days of live news, so 3d/5d populate as history deepens; early state is honestly `INSUFFICIENT_DATA` with partial 1d data, never all-dashes). Live-only SQL filter inlined. Pure core: `analytics/news_edge.py::build_news_edge`; daily-bar yfinance history cached 30 min (`_NEWS_EDGE_PX_CACHE`) |
| `GET /api/capital-paralysis` | **Trap + cost + unlock in one view.** liquidity sees the trap (no dry powder), decision-drought sees the cost (alpha bled while pinned), suggestions lists ideas it can't fund — none connect them. Composes `build_liquidity` + `build_decision_drought` (single source of truth — no re-derived metrics) and adds the **unlock ladder**: open positions ranked in desk cut-priority (losers before winners, then largest value), each rung carrying the cash a sale frees, the deployed-% after, and `restores_action_alone` (does this single sale put cash back above `min_actionable_usd` = max($1, 1% of book)?). `recommended_unlock` = the first restoring sale; `state` ∈ `FREE`/`PINNED`/`EMPTY`/`NO_DATA`. **Advisory only — never gates Opus, adds no caps (invariant #2).** Pure core: `analytics/capital_paralysis.py::build_capital_paralysis`. Locked by `tests/test_capital_paralysis.py` |
| `GET /api/open-attribution` | Selection-vs-market on the **open** book — the live trader's *dominant* return source (it mostly HOLDs, so realized round-trips are tiny while open drift dominates; round_trips/`/api/analytics` only cover *closed* trades). Per open **stock** position: return since `opened_at`, SPY return over the same window (anchored to the equity curve's `sp500_price` **at-or-after** entry — `opened_at` is correct because invariant #8 resets it on a reopened lot), `alpha_pct`, and `excess_usd` (unrealized P&L − what the cost basis in SPY would have made). Book aggregate is computed over **anchored rows only** (an un-benchmarkable position would skew `book_open_alpha_pct`). Options are flagged & skipped (alpha-vs-SPY doesn't fit Greeks — `/api/backtests/compare` "stocks only" precedent, invariant #10). Pure core: `analytics/open_attribution.py::build_open_attribution`. Locked by `tests/test_open_attribution.py` |
| `GET /api/trade-asymmetry` | **Behavioural-edge / exit-&-sizing pathology** — the *why* behind the P&L, distinct from `/api/analytics` (raw aggregates) and `/api/calibration` (is the confidence axis accurate). Composes the single source of truth (`build_round_trips`, invariant #10 — no re-derived P&L) into payoff ratio, per-trade expectancy, the **breakeven win-rate the payoff ratio implies vs the actual win-rate** (the gap is the verdict), and the **disposition gap** = mean winner hold-days − mean loser hold-days (negative ⇒ cutting winners faster than losers — the disposition effect that produces a `win-small/lose-big` curve). Sample-size honesty mirrors `news_edge`: numeric metrics emit from the first closed round-trip but the **verdict label is withheld until `STABLE` (n≥20 round-trips)** — `NO_DATA`→`EMERGING`→`STABLE`; a five-trade verdict is noise. Verdicts (STABLE only, precedence in order): `PAYOFF_TRAP` (actual<breakeven ≡ expectancy<0), `DISPOSITION_BLEED` (net-positive but winners cut faster than losers — money left on the table), `EDGE_POSITIVE` (positive & well-managed), `FLAT`. **Intentional divergence from `/api/analytics`:** this module's win/loss split is strict `pnl_usd>0` / `<0` with washes (`==0`) excluded from *both* (matching round_trips' strict `>0` convention, invariant #10), so `avg_loser_usd` and the win-rate basis differ from `analytics_api` (which folds washes into its loser denominator). This is by design — do not "reconcile" them. Advisory only — never gates Opus, adds no caps (invariants #2/#12). Pure core: `analytics/trade_asymmetry.py::build_trade_asymmetry`. Locked by `tests/test_trade_asymmetry.py`. **UI:** Behavioural-edge panel + the previously-orphaned Capital-paralysis & Open-book-alpha panels are now wired into the `:8090` trader page; their JS degrades to an explicit "restart paper-trader to apply" message (not a silent error) when the running process predates the endpoint commit (the `/api/build-info` `stale` contract) |
| `GET /api/decision-reliability` | **The *true current-regime* NO_DECISION rate — not the inflated headline.** `decision-health`/`-forensics`/`-drought` give the rate/why/cost, but the headline % is dominated by *legacy* pre-diagnostics rows (`reasoning == "claude returned no parseable JSON"`) that **stop accruing the moment the runner restarts onto diagnostic code** — a fixed historical mass that never decays. This partitions the decision log at the **newest legacy-tagged failure timestamp** (boundary; `None` ⇒ no legacy ⇒ all rows current) and reports the *post-restart* failure rate + a current-only mode mix, reusing `decision_forensics.classify_failure` (taxonomy) and `build_decision_drought` (`involuntary_alpha_bleed_pct`) as the single source of truth — nothing re-derived (`capital_paralysis` precedent). Sample-size honesty mirrors `news_edge`/`trade_asymmetry`: `NO_DATA` → `STALE_LEGACY_DOMINATED` (legacy failures > current_total **and** current_total < `MIN_CURRENT`=12 → `restart_recommended=True`; the actionable state — restart so failures get diagnostic tags & the sample grows) → `INSUFFICIENT` (current_total < `MIN_CURRENT`, verdict withheld) → `HEALTHY`/`DEGRADED`/`CRITICAL` judged on the **current** rate (≥25 DEGRADED, ≥50 CRITICAL — thresholds identical to `decision_forensics` so they never disagree). `headline_failure_rate_pct` passes `build_decision_forensics` through verbatim for the contrast; `dead_cycles_per_day` = current_rate × decisions/day (cadence from the full timestamp span; `None` on a zero/1-point span — never divides by zero); unparseable-`timestamp` rows are counted in totals but excluded from the current partition when a boundary exists. The verdict *matures with history* (STALE→…→HEALTHY as post-restart cycles accumulate). Advisory only — never gates Opus, adds no caps (invariants #2/#12). Pure core: `analytics/decision_reliability.py::build_decision_reliability`. Locked by `tests/test_decision_reliability.py`. **UI:** `dr-card` panel on the `:8090` trader page; JS degrades via the `/api/build-info` `stale` contract |
| `GET /api/funded-suggestions` | **Pairs every unfundable BUY/ADD idea with the specific sale that funds it.** `liquidity`/`capital-paralysis`/`suggestions` each see part of the trap; none connect "idea I can't afford" to "position to sell to afford it". Composes the existing `/api/suggestions` list (the endpoint calls `suggestions_api()` verbatim — **no refactor**) with `build_capital_paralysis` (its `unlock_ladder` is already in desk cut-priority: biggest loser first). For each conviction-ranked BUY/ADD: `can_act` **AND `cash ≥ notional`** ⇒ `FUNDED` (cash truly covers it — no sale); `can_act` **but `cash < notional`** ⇒ **`PARTIAL`** — `can_act_on_signal` only means cash cleared a *tiny* act-floor (≥ $1 and ≥ 1% of book), **not** that cash covers the advisory notional, so walk the **same** desk-cut ladder for the minimum sale prefix whose `cash + cumulative_freed_usd` ≥ notional (`funded_by`, `frees_usd`, `enough`; `enough=False` when even cash + the whole ladder still falls short — still `PARTIAL`, since cash funds *part*); PINNED ⇒ walk the ladder attaching the **minimum prefix** of sales whose `cumulative_freed_usd` ≥ an *advisory* suggested notional (`round(conviction × total_value, 2)`, explicitly labelled — sizes nothing) → `UNLOCKABLE` (`funded_by`, `frees_usd`, `enough=True`); whole-ladder-insufficient / empty-ladder / EMPTY / NO_DATA ⇒ `UNFUNDABLE` (full ladder, `enough=False`). Payload adds `n_partial`; the headline becomes `CASH-LIGHT — $X cash: …` instead of the old false `FREE — … fundable from cash now` whenever any idea is `PARTIAL`; the UI gains an amber `PARTIAL` chip + a `partial (cash + sale)` stat. (Before this fix `can_act ⇒ FUNDED` unconditionally painted an $856 advisory idea green "cash available now — no sale required" on $18.49 cash — the panel now states the cash shortfall and the exact sale prefix that closes it.) Only BUY/ADD are funding-checked — HOLD/WATCH are no-ops and TRIM/EXIT *raise* cash. `top_actionable` = highest-conviction BUY/ADD (deterministic `(-conviction, ticker)` tie-break); `recommended_pairing` = `{sell: recommended_unlock.ticker, buy: top_actionable}` **only when PINNED**. Advisory only — never gates Opus, sizes nothing, adds no caps (invariants #2/#12). Pure core: `analytics/funded_suggestions.py::build_funded_suggestions`. Locked by `tests/test_funded_suggestions.py`. **UI:** `fund-card` panel; same `stale` degrade contract. **Chat:** `unified_dashboard.py::_build_chat_context_block` adds a `_fetch_decision_reliability` sub-fetch emitting one compact `TRADER STATE:` line (pinned + current-regime parse-fail + bleed) so `/api/chat` answers "why isn't it trading?" truthfully; degrades to the pinned/bleed half alone until the trader process restarts onto `/api/decision-reliability` |
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
| `GET /api/correlation-cluster-warning` | **Hidden-factor-bet alarm — translates `/api/correlation`'s pairwise matrix into a single cluster-share verdict.** The parent reports *mean* ρ across all pairs + one global verdict (`DIVERSIFIED`/`MODERATE`/`CONCENTRATED`/`SINGLE_NAME_RISK`). A book like {NVDA, AMD, AVGO, KO, JNJ} reads as `MODERATE` on mean ρ — but the first three names form a single semis cluster running as one trade inside a wrapper of two uncorrelated consumer staples. This endpoint surfaces that cluster: single-linkages the names at ρ ≥ `HIGH_CORR` (the same constant `/api/correlation` already uses for `CONCENTRATED`, imported verbatim so the two endpoints can never disagree on the threshold), returns the largest multi-name cluster + its share of book by market value + its internal mean ρ, and emits an `NO_CLUSTERS / WATCHLIST_CLUSTER / DOMINANT_CLUSTER / HIDDEN_FACTOR_BET` verdict keyed off the cluster's *weight* (not the mean ρ — so it catches the "5 names, 3 are one bet" regime the mean-ρ verdict misses). Weight bands: `WATCHLIST_CLUSTER` < 30% < `DOMINANT_CLUSTER` < 60% < `HIDDEN_FACTOR_BET` — chosen to mirror the spirit of the parent's `DIVERSIFIED_MAX_TOP_WEIGHT`/`DOMINANT_WEIGHT` for *single* names, generalised to co-moving clusters. Pure builder over the existing `build_correlation` payload (no new yfinance — the endpoint pays for one shared fetch, hands the payload to `build_correlation_cluster_warning`); upstream `NO_DATA`/`INSUFFICIENT` propagate verbatim so degraded states are honest. Single-linkage via iterative union-find (deterministic, stable across runs); singleton components are excluded (single-name risk is the parent's job). Advisory only — never gates Opus, adds no caps (invariants #2/#12). Pure core: `analytics/correlation_cluster_warning.py::build_correlation_cluster_warning` (never raises). Locked by `tests/test_correlation_cluster_warning.py` (exact union-find / cluster-weight / verdict-band locks, threshold-boundary tests at ρ = `HIGH_CORR` and just below, biggest-by-weight-not-size selection lock). `@swr_cached("correlation-cluster-warning", 90.0)` — same TTL as the parent. Applies on next paper-trader restart |
| `GET /api/position-news-cooldown` | **Per-open-position news-flow cooldown — has the news desk gone quiet on this held ticker, or is the story still moving?** Distinct from `/api/position-attention` (times *Opus* looks) and `/api/thesis-drift` (re-tests entry rationale against current state). Answers a question NO other panel does: **for each held name, when was the last live article that actually scored above noise (`MIN_SCORE_THRESHOLD`=4.0)?** Catches *thesis decay through silence* — a position opened on a catalyst whose news flow has dried up while the operator's attention moved on. There is no error message, no NO_DECISION storm, no Discord ping — only an absence. Per-position verdict ladder mirrors `position_attention`'s shape so the operator's eye reads them the same way: `FRESH` ≤6h (one trading session) → `WARM` ≤24h ("yesterday's news still counts") → `COOL` ≤72h ("story is aging fast") → `DARK` (catalyst window effectively closed; also the bucket for "no entry in news map at all" — silent / never seen). Portfolio rollup: `INSUFFICIENT_DATA` (no open positions) → `DARK_BOOK` (any DARK) → `COOLING_BOOK` (any COOL but no DARK) → `OK`. Sorts worst-first (DARK on top, None-hours-since float to the very top inside DARK). Live-only filter applied upstream (mirrors `signals.get_top_signals` — invariant #1: backtest/opus_annotation synthetic rows never reach this read). Pure builder; endpoint pre-fetches per-ticker `{last_first_seen, top_score, top_title, n_24h, n_72h}` via `_ticker_news_cooldown` (newest-first ORDER BY first_seen DESC so the first regex hit per ticker IS the most recent — score-ranked top is informational only). Advisory only — never gates Opus, adds no caps (invariants #2/#12). Pure core: `analytics/position_news_cooldown.py::build_position_news_cooldown` (never raises; `now` injectable). Locked by `tests/test_position_news_cooldown.py` (exact verdict-boundary locks at each threshold, rollup-precedence locks, sort-order lock, "missing news entry ⇒ DARK" lock so the catch-all branch can't silently downgrade). `@swr_cached("position-news-cooldown", 60.0)`. Applies on next paper-trader restart |
| `GET /api/decision-context` | **What is the live trader actually being *shown* right now?** — the decision *input* every one of the ~45 output-diagnostic endpoints presupposes. `decisions` stores only `action_taken`+`reasoning`; the only raw capture is `RAW_CAPTURE_CHARS`=1000 of the *response* on a parse failure. When the trader spends cycle after cycle on `NO_DECISION (timeout/empty)` / flat `HOLD` (the dominant 2026-05-17 live pattern — `$972.69`, `$18.49` cash, MU stale-marked) an operator has no way to see *what Opus was fed*. This reconstructs it on demand: the prompt rendered through the **same `strategy._build_payload`** the live `decide()` uses (+ the identical `SYSTEM_PROMPT`/`ML ADVISOR` framing) so it is **byte-identical to the live prompt given identical inputs** (single source of truth, invariant #10 — no re-implemented prompt), bounded to `MAX_PROMPT_CHARS`=40000 with `prompt_chars`/`prompt_truncated` honesty keys; an `input_summary` (top/urgent/merged counts — `signal_count` is the *exact* value `decide()` writes to `decisions.signal_count` — watchlist/futures resolved-vs-missing, quant tickers, sentiment mentions); `advisory_blocks` presence (self-review/track-record/risk-mirror/ml); the embedded `/api/mark-integrity`; and a `feed_state` ∈ `BLIND` (0 merged signals — a HOLD this cycle is *forced* by an empty feed, not chosen) / `DEGRADED` (≥`DEGRADED_MISSING_RATIO`=50% of watchlist prices missing — the yfinance starvation behind the timeout storms) / `OK`. **`_claude_call` is never invoked** (`claude_invoked:false`; locked by an endpoint test that monkeypatches it to raise and still expects 200). The snapshot is the new write-free `strategy.portfolio_snapshot_readonly`, which shares the extracted pure `strategy._mark_to_market` with the live `_portfolio_snapshot` so the inspector's marks (incl. expired-option intrinsic #13 + `stale_mark`) can never drift from the real ones (invariant #10) and the dashboard thread never mutates the live trader's persisted marks/equity. Orchestration (`assemble_inputs`, mirrors `decide()`'s pre-`_claude_call` assembly with each advisory builder wrapped non-fatally exactly as `decide()` wraps it) is shared by the endpoint **and** `python -m paper_trader.analytics.decision_context [--full|--json]` (works when `:8090` is wedged — the `desk_pulse`/`signals --check-freshness` precedent) so the two can't drift. SWR-cached 30s (the assemble fetch is multi-second; the `/api/state` precedent). Advisory only, **NOT** injected into the decision prompt — dashboard/chat/CLI only (invariants #2/#12; `strategy.decide()` untouched). Pure core: `analytics/decision_context.py::build_decision_context`. Locked by `tests/test_decision_context.py` (prompt section-header fidelity, exact input counts incl. `signal_count`, ML-advisor gating, feed_state boundaries, truncation honesty, embedded mark-integrity verbatim, and the `portfolio_snapshot_readonly` *marks-identically-but-never-writes* contract vs `_portfolio_snapshot`) + `tests/test_decision_context_endpoint.py` (Flask test client: never-calls-Opus 200, BLIND/DEGRADED, read-only, SWR honesty keys + warm-hit). Applies on next paper-trader restart (`/api/build-info` `stale`) |
| `GET /api/mark-integrity` | **How much of the displayed book value is *fictional* right now?** — the mark-trust meta-metric no panel surfaces. When yfinance returns nothing for a held name `strategy._mark_to_market` falls back to `avg_cost` and flags `stale_mark=True` (the live 2026-05-17 pathology: `MU 0.5 @ 724.12`, `current_price==avg_cost`, `P/L $0.00` — indistinguishable from a genuinely flat row). That flag is surfaced *per position* to Opus & Discord, but nothing answers the **aggregate**: what share of gross book value is marked at cost, so `/api/analytics` Sharpe, `/api/drawdown`, the equity curve and the headline P&L are all quietly partially false. Reports `n_stale`, `stale_value_usd`, `stale_value_pct` of gross, per-name rows, `stale_tickers`, and a verdict `NO_DATA`→`CLEAN`→`DEGRADED` (0<pct<`UNTRUSTWORTHY_PCT`=50, or gross 0 with stale rows so the share is unquantifiable) →`UNTRUSTWORTHY` (≥50% — treat every displayed P/L as substantially fictional until the feed recovers / runner restarts). Reads the write-free `strategy.portfolio_snapshot_readonly` (never mutates the live trader). Pure, never raises (garbage rows degrade to zero value — the behavioural-builder `_safe` contract). Advisory only — never gates Opus, adds no caps (invariants #2/#12). Also embedded inside `/api/decision-context`. **Folded as an additive `mark_trust` honesty key into the three equity-derived risk endpoints this docstring names as the silent victims — `/api/tail-risk`, `/api/drawdown`, `/api/analytics` (2026-05-18, Agent 4).** A stale cycle records a *cost-frozen flat* equity point; those flats deflate vol/drawdown, inflate Sharpe, and truncate the VaR tail, yet a grep showed `stale_mark` had only ever reached mark_integrity/strategy/dashboard/reporter — never these maths. `dashboard._mark_trust_block(store)` composes `build_mark_integrity` **verbatim** off the SAME write-free `portfolio_snapshot_readonly` snapshot (single source of truth #10 — no re-derived staleness), adds `{verdict,n_stale,n_positions,stale_value_pct,stale_tickers,headline,note}` (the `note` only when verdict ∉ CLEAN/NO_DATA), and is `_safe`: any fault → key **omitted** so the risk payload is byte-identical and the endpoint never 500s for this reason. Purely additive (keyed-assertion-safe, the existing `tail_risk`-in-`/api/analytics` precedent); observational only, no caps, not injected into the decision prompt, **no schema change** (invariants #2/#12/#13). `hold_discipline`/`thesis_drift` (which read open-position P/L and silently misread a stale `$0.00` as a genuine flat) are a known *deferred* contamination — their endpoints feed `store.open_positions()` which lacks `stale_mark`, so a fix needs an endpoint data-source change that risks their existing exact-value `TestEndpoint`s; see `docs/superpowers/specs/2026-05-18-mark-trust-risk-surface-design.md`. Pure core: `analytics/mark_integrity.py::build_mark_integrity`. Locked by `tests/test_mark_integrity.py` (the exact live MU-stale shape `stale_value_pct`=37.94 off the raw gross, `>=50`→UNTRUSTWORTHY inclusive boundary, zero-gross no-divide-by-zero, option ×100, never-raises-on-garbage) + `tests/test_decision_context_endpoint.py` (Flask test client read-only + UNTRUSTWORTHY-when-price-missing) + `tests/test_mark_trust.py` (Flask test client end-to-end on all three endpoints: stale book → `mark_trust` UNTRUSTWORTHY; clean → CLEAN/`note`=None; **additive no-risk-drift** vs a direct `build_tail_risk` call — only `mark_trust` added, every risk field byte-identical; the `_safe` snapshot-fault → 200 + key-omitted contract; single-source-of-truth no-drift vs `build_mark_integrity`). Applies on next paper-trader restart |
| `GET /api/holding-period-distribution` | **Closed round-trip P/L stratified by hold duration — where in the holding-period axis does my P/L actually live?** (2026-05-20) Every other realised-P/L surface reduces the closed-trip set to a single aggregate (`/api/track-record`, `/api/trade-asymmetry`) or a per-trip story (`/api/round-trip-postmortem`, `/api/winner-autopsy`, `/api/loser-autopsy`) — none of them answer the discretionary-PM question "63% of trips are SCALP but 91% of P/L comes from SWING". This surfaces it. Buckets each closed round-trip into SCALP (<1h) / INTRADAY (1-6h) / OVERNIGHT (6-24h) / SWING (1-3d) / TREND (3-7d) / POSITION (>7d) and per bucket reports `n_trips`, `n_winners`/`n_losers`, `total_pnl_usd`, `avg_pnl_usd`, `median_pnl_usd`, `win_rate_pct` (winners ÷ decided — zero-pnl trips excluded from the denominator, same convention as track_record), `share_of_trips_pct`, `share_of_abs_pnl_pct` (denominator is the absolute-value sum so a signed sum near zero can't blow the share fractions to nonsense). Cross-bucket verdicts: `alpha_engine` (bucket with the highest total $ P/L — must be net positive to count as an "engine", never a least-loss bucket); `dominant_bucket` (most trips; ties broken in canonical SCALP→POSITION order for determinism); `worst_bucket` (most-negative total $ P/L). State ladder mirrors `round_trip_postmortem`: `NO_DATA` → `INSUFFICIENT` (<`STABLE_MIN_TRIPS`=5; bucket rows still emit so the distribution can be watched accumulating, only the verdict is withheld — sample-size-honesty precedent) → `OK`. Bucket edges are exclusive-upper inclusive-lower; the 1h/6h/24h/72h/168h boundaries are pinned by tests so a silent widen can't shift a whole bucket of trips. Composes `analytics.round_trips.build_round_trips` as the SSOT (invariant #10) — no re-implemented round-trip aggregation. Pure builder; the endpoint owns I/O over `store.recent_trades`. Observational only — never gates Opus, never injected into the decision prompt, no caps (invariants #2/#12). Pure core: `analytics/holding_period_distribution.py::build_holding_period_distribution`. Locked by `tests/test_holding_period_distribution.py` (20 tests: each bucket boundary inclusive at its lower edge / exclusive at its upper edge, state ladder, alpha_engine must-be-net-positive, dominant tie-break canonical order, zero-pnl-excluded-from-decided, share_of_abs_pnl denominator, degrade-never-raise on garbage). Applies on next paper-trader restart |
| `GET /api/add-discipline` | **ADD-trade discipline audit — is the book chasing (paying up vs running cost basis), averaging down (paying below), or stacking (neutral)?** (2026-05-20) When the book BUYs into a name it already holds, that ADD carries a sign: paying above the running avg_cost is the textbook *chasing* behaviour (anchoring on the original entry, bidding into strength after the easy entry is behind); paying below is *averaging down* (rational if the thesis is intact, the textbook setup for `loser_autopsy`'s SLOW_BLEED — doubling down on a broken thesis). **No existing endpoint watches the ADD moment itself.** `/api/trade-asymmetry` is the disposition gap; `/api/churn` counts overtrading; `/api/loser-autopsy`/`/api/winner-autopsy` narrate closed round-trips but neither sees the ADD pricing. Walks the trade ledger chronologically per position-key, maintains a running avg_cost (same VWAP arithmetic `store.upsert_position` uses), and classifies each non-opening BUY as `CHASING` (≥`CHASE_THRESHOLD_PCT`=1.5% above running cost, inclusive at the band edge with a 1e-6 epsilon for float-jitter robustness) / `AVERAGING_DOWN` (≤-1.5% below) / `STACKING` (inside the band). Reports `counts`, `pct`, per-add rows (ticker/ts/price/running_avg_cost_before/pct_above_cost/category), `by_ticker` (sorted by n_adds desc, ties by ticker for determinism), `closed_outcomes` (each closed round-trip tagged with its dominant ADD style via the `entry_trade_ids` join — `CHASING > AVERAGING_DOWN > STACKING` precedence on ties so the riskiest behaviour can't be masked by a balancing average-down), and `outcomes_by_style` ({style → n, total_pnl_usd, mean_pnl_pct, median_pnl_pct}). The closed-outcomes rollup answers the falsifiable question: did chasing-ADDs produce worse round-trip P/L than averaging-down ADDs? SELL-to-zero (full close) resets the per-position basis cycle so the next BUY is an "open", not an ADD — mirrors `build_round_trips` so the two views never disagree on what counts as open vs add. Partial SELLs leave basis intact. Distinct (option_type/strike/expiry) position keys are isolated — an option BUY on a held stock is a new open, not an ADD on the stock basis. State ladder: `NO_DATA` (no BUYs / only opens) → `EMERGING` (<`STABLE_MIN_ADDS`=3; counts emit but `dominant_style_overall` withheld) → `STABLE`. Composes `analytics.round_trips.build_round_trips` as the SSOT (invariant #10). Pure builder; the endpoint owns I/O over `store.recent_trades`. Observational only — never gates Opus, never injected into the decision prompt, no caps (invariants #2/#12). Pure core: `analytics/add_discipline.py::build_add_discipline`. Locked by `tests/test_add_discipline.py` (22 tests: opening-BUY suppression, CHASING/AVERAGING_DOWN/STACKING per ±1.5% boundary inclusive, VWAP basis update after each ADD (blended-basis lock so a stale basis can't misclassify subsequent adds), SELL-to-zero resets basis vs partial-sell does not, distinct position keys isolated, CHASING > AVG_DOWN > STACKING precedence on ties, round-trip outcome rollup verifies signs (CHASING trip −P/L vs AVG_DOWN trip +P/L), state ladder NO_DATA→EMERGING→STABLE, by_ticker sort + dominant, degrade-never-raise on garbage rows / non-list input / non-positive basis fallback to STACKING). Applies on next paper-trader restart |
| `GET /api/position-rationale` | **Per-open-position most-recent Opus rationale — *what did Opus actually say* about each held name?** (2026-05-20) `/api/position-attention` answers "did Opus look?" (freshness / neglect). `/api/thesis-drift` re-tests entry theses against current state. Neither surfaces the concrete current rationale — the trader's #1 review-the-book question: *"why am I still holding NVDA? what did Opus actually say last cycle?"* The data is already in `decisions.reasoning` (JSON envelope `{"decision": {"reasoning": "...", "confidence": x, "action": "..."}, ...}` written by `strategy.decide()`). Today the operator must scroll the decision feed by hand to find the most recent row per ticker; this puts the answer one HTTP read away. Walks the newest-first `recent_decisions` list and picks the FIRST match per held ticker via `_parse_action_ticker` (mirror of `position_attention`'s helper — kept local so the two builders never disagree on the (verb, ticker) split). The reasoning string is capped at `_MAX_REASON_CHARS`=600 so a long Opus block can't bloat the response. Per-position output: `ticker`, `type`, `qty`, `days_held`, `last_decision_ts`, `hours_since_last_decision`, `last_decision_verb` (HOLD / BUY / SELL / ...), `last_decision_confidence`, `last_decision_reasoning`. NO_DECISION rows are correctly skipped (the helper returns ticker=None for them). BLOCKED rows DO match their ticker (a real Opus decision the trader still wants to see) but carry no JSON envelope, so `last_decision_reasoning` reads None — `n_with_rationale` counts only rows with actual reasoning so the verdict reflects rationale presence, not match presence. Degrade-safe (the `_safe` contract): legacy pre-JSON reasoning (parse_failed/retry_failed/raw "no response" text), missing inner `decision` key, non-dict envelopes, non-string reasoning fields, unparseable timestamps all degrade gracefully — the row still ships with `last_decision_reasoning=None`, never raises. Sort: positions WITHOUT a rationale to the top (need attention first), then within each bucket oldest-since-last-decision first (most stale rationale most operator-actionable). Verdict ladder: `INSUFFICIENT_DATA` (no open positions) → `MISSING_RATIONALE` (any held position has no recent Opus rationale on file — including a position that's only been BLOCKED but never HOLD/BUY/SELL'd) → `OK`. Pure builder over `store.open_positions()` + `store.recent_decisions(limit)` — no DB writes, no network, no Opus invocation. Advisory only — never gates Opus, never injected into the decision prompt, adds no caps (invariants #2/#12 — the `position_attention` precedent). Pure core: `analytics/position_rationale.py::build_position_rationale`. Locked by `tests/test_position_rationale.py` (13 tests: most-recent-per-ticker, NO_DECISION/BLOCKED filtering, legacy unparseable reasoning, envelope-missing-decision / non-dict-decision / non-string-reasoning edges, 600-char cap, missing-rationale rows sort to top, OK-verdict-when-all-have-reasoning, empty input, garbage timestamp skipped not raised). Applies on next paper-trader restart |
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

**Persist-write serialization (`_VOLUME_PERSIST_LOCK`, 2026-05-20).** Two
backtest run threads that both fetched a fresh volume series in the
*same* window both reach `_persist_volume_cache_for_window` concurrently;
the shared `".json.tmp"` filename means they both `open(..., 'w')`
(O_TRUNC) the SAME file in parallel and their writes can interleave at
the OS level → torn JSON could then land under the canonical path via
`tmp.replace(path)`. The cache snapshot already runs under
`_VOLUME_CACHE_LOCK`; `_VOLUME_PERSIST_LOCK` (a *separate* lock so cache
readers/writers aren't blocked by disk I/O) wraps only the short
`tmp.write_text` + `tmp.replace` so one writer at a time touches the
file. Last writer wins by construction — both threads staged the same
snapshot under the cache lock above. Locked deterministically by
`tests/test_backtest.py::TestVolumeCachePersistAtomicity::test_concurrent_persists_serialize_tmp_write`
(8 concurrent writers must record zero "tmp write started without
persist lock held" violations + canonical file must remain valid JSON).

### Concept-drift report (`paper_trader/ml/outcome_drift.py`)

The scorer retrains every cycle on the last `MAX_OUTCOMES_FOR_TRAINING`
(5000) records of `data/decision_outcomes.jsonl`. `regime_audit` answers
"skill conditional on regime", but nothing in the existing surface
(`skill_trend`, `baseline_compare`, `calibration`, `feature_coverage`,
`feature_importance`) reports whether the trainer's input/output
distribution itself is drifting under the model. This module fills that
gap.

`build_outcome_drift(records, recent_fraction=0.25)` sorts records by
`sim_date`, splits the last 25% (recent) vs first 75% (older), and for
every numeric feature in `TRACKED_FEATURES` reports
`drift_score = (μ_recent − μ_older) / σ_older` — a z-style shift in
σ-of-older units. State ladder mirrors `tail_risk` / `build_correlation`:
`NO_DATA` / `INSUFFICIENT` (<`MIN_PER_BUCKET`=20 in either bucket) /
`STABLE` / `MILD_DRIFT` (≥0.5σ on any feature) / `SEVERE_DRIFT` (≥1.0σ
or constant-older-with-nonzero-shift → ±inf). Pure / no I/O / never
raises (the `_safe` contract — `None`/non-dict/garbage rows degrade
rather than crash; `_safe_float` rejects NaN/inf/bool exactly like
`decision_scorer._to_float`). Sort by `|drift_score|` DESC, INSUFFICIENT
rows last. Locked by `tests/test_outcome_drift.py` (39 tests: state
ladder boundaries, exact 1σ/0.5σ arithmetic, constant-older→±inf
convention, |drift| DESC sort, recent_fraction clamp + invalid-type
fallback, garbage-row degrade-never-raise, `load_outcomes` streaming
+ malformed-line skip, CLI JSON + table modes).

CLI: `python3 -m paper_trader.ml.outcome_drift [--path PATH] [--recent-fraction F] [--json]`.
Live read on the 7413-row outcomes file (2026-05-20) flags `MILD_DRIFT`:
`news_article_count +0.96σ` and `regime_mult -0.52σ` against the older
bucket — quant-actionable evidence the trainer's tail no longer
describes a stationary signal regime.

### Test commands for the ML / backtest domain

```bash
# Focused unit tests (offline, mocks yfinance):
python3 -m pytest tests/test_decision_scorer.py tests/test_backtest.py \
                 tests/test_outcome_drift.py tests/test_continuous.py \
                 tests/test_calibration.py -v

# Volume-cache + persist-race coverage:
python3 -m pytest tests/test_backtest.py -k VolumeCache -v

# Full ML/backtest slice (the spec the HYBRID reviewers run):
python3 -m pytest tests/ -v -k "ml or backtest or scorer" 2>&1 | tail -30
```

### Operational quirks a quant should know

- The deployed `data/ml/decision_scorer.pkl` and the per-cycle
  `data/scorer_skill_log.jsonl` are written by *different* code paths
  (the pickle by `train_scorer` after dedup; the ledger by
  `_train_decision_scorer` parsing the formatted status string). The
  pickle's `n_train` is the dedup'd training count for the train fold
  only; the ledger's `train_n` is the same number. If they disagree
  (observed 2026-05-20: pickle `n_train=400` vs ledger
  `train_n=3987`), the continuous loop has not run a fresh retrain
  cycle since the pickle was written — the gate state derived from the
  pickle (`gate_active = n_train ≥ 500`) is the authoritative live
  state, not what the ledger says.
- A stuck `status='running'` row in `backtest.db` older than 6h has
  already passed the `_reap_orphaned_runs` window — that means the
  continuous loop process itself is down (verified via
  `pgrep -af continuous_backtest`), not that the reaper is broken.
  Restart the loop; the next process's startup-reap (and the mid-loop
  reap each cycle) will sweep them to `failed`.

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

### Bootstrap CIs on OOS skill — `paper_trader/ml/skill_uncertainty.py`

Every existing OOS diagnostic (`_oos_rank_metrics`, `evaluate_scorer_oos`,
`calibration --oos`, `skill_trend`) reports the metric as a **single point
estimate** per cycle. With OOS sample sizes in the hundreds-to-low-thousands
and near-zero underlying skill, a single number can't tell a skeptical quant
whether a rank-IC of 0.04 is real-but-weak signal or noise around 0.

`skill_uncertainty.py` runs a percentile bootstrap (Efron) on the deployed
scorer's OOS pairs and reports **95% confidence intervals** on rank-IC /
RMSE / dir-acc, with a crisp verdict:

| Verdict | Trigger |
|---|---|
| `NOT_TRAINED` | `scorer.is_trained` is False |
| `INSUFFICIENT_DATA` | < `MIN_OOS=30` finite OOS pairs |
| `SKILL_DETECTED` | rank-IC CI strictly excludes 0 (statistically distinguishable) |
| `NO_SKILL_DETECTED` | rank-IC CI straddles or sits below 0 |

Same operational discipline as `calibration.py` / `gate_audit.py` /
`skill_trend.py`: read-only, no train, no pickle / `build_features` /
`N_FEATURES` touch, never raises (`bootstrap_skill_ci` degrades to
`status='error'` on any fault). Single source of truth — the OOS split is
`validation.split_outcomes_temporal` (the EXACT split
`_train_decision_scorer` uses for `oos_rmse`), the predict signature is
the same 11-kwarg path `_oos_rank_metrics` uses, the rank correlation is
`calibration._spearman`, and the universal SELL sign-flip is applied. So
this module and the ledger's scalar metrics can never drift.

```bash
# Default: 1000 resamples, 95% CI, deterministic seed
cd /home/zeph/trading-intelligence/paper-trader && python3 -m paper_trader.ml.skill_uncertainty

# Tighter CI with more resamples; JSON for downstream tooling
python3 -m paper_trader.ml.skill_uncertainty --bootstraps 2000 --json
```

Exit code 0 only when the verdict is `SKILL_DETECTED` — so a shell caller
can `if !; then` gate dashboards on real distinguishable skill, not noisy
point estimates. Locked by `tests/test_skill_uncertainty.py` (13 tests):
strong signal → SKILL_DETECTED with CI excluding 0; pure noise →
NO_SKILL_DETECTED with CI straddling 0; anti-correlated → CI fully below
0; insufficient n; untrained; SELL sign-flip; NaN-row dropping; seed
reproducibility (same seed ⇒ identical CIs, different seeds ⇒ different
CI bounds); crash-resilience.

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

### Persona name + raw regime label in outcomes (2026-05-19)

- **`_compute_decision_outcomes` now additively records `persona` (e.g.
  `"Momentum Trader"`) + `regime_label` (raw `bull` / `sideways` / `bear` /
  `unknown` string)** on every outcome row. Inert to scorer training
  (`train_scorer` / `build_features` ignore unknown dict keys — the
  `forward_return_10d/20d` / `wk52_pos` / `gate_scorer_pred` precedent), so
  no retrain is required and the scorer pickle / `N_FEATURES` / every
  existing OOS diagnostic stay byte-identical.
- **Why persona:** `persona_skill` / `persona_leaderboard` already derive
  the persona via `persona_for(run_id)` at analysis time. Capturing the
  NAME directly lets ad-hoc shell queries
  (`grep '"persona": "Momentum"' data/decision_outcomes.jsonl | …`) and
  future per-persona diagnostics filter without re-importing the live
  `PERSONAS` dict. If a future change renames or adds personas, old
  outcome rows still self-describe (they carry the persona name as it was
  when the decision happened, not as it is now).
- **Why regime_label:** `regime_mult` (0.3 / 0.6 / 1.0) is a stringly-typed
  encoding of the `bull` / `sideways` / `bear` / `unknown` label that
  `_ml_decide` and `_compute_decision_outcomes` both compute from the SPY
  50/200 MA via `_market_regime`. The multiplier alone cannot distinguish
  `bull` from `unknown` (both = 1.0), so a regime-conditional cut on
  `regime_mult == 1.0` silently lumps SPY-pre-200d-history `unknown`
  cycles with real bull cycles — biasing every per-regime analysis.
  Capturing the raw label resolves that ambiguity. `regime_audit` (which
  decodes the same label from `regime_mult`) keeps working unchanged.
- **Test-locked in `tests/test_continuous.py::TestWk52PosCapturedInOutcomes`:**
  `test_persona_and_regime_label_captured` (run_id=2 → `"Momentum Trader"`,
  `regime_label` is one of the documented strings),
  `test_persona_field_present_for_run_id_1` (run_id=1 → `"Value Investor"`,
  locks the `persona_for` `((run_id - 1) % 10) + 1` cycling formula),
  `test_capture_does_not_break_existing_keys` (full schema regression —
  every previously-documented outcome field must still appear).

### Per-window volume cache atomic write (2026-05-19)

- **`_persist_volume_cache_for_window` now uses tmp+`.replace`** instead of
  a bare `path.write_text(json.dumps(flat))`. The bare write is NOT atomic:
  a process kill (OOM / SIGKILL, the documented `CLAUDE.md §11` continuous-
  loop failure mode) mid-write leaves a torn / truncated JSON file. The
  next `_load_volume_cache_for_window` then fails `json.loads`, falls back
  to an empty dict (silent `except` swallows the error), and *still* marks
  the window "loaded" in `_VOLUME_CACHE_DISK_LOADED` — so every subsequent
  `vol_ratio` computation re-fetches the volume series from yfinance for
  the entire window. Worse, a CONCURRENT loader in another thread can read
  a partially-written file (same swallow path, same yfinance re-fetch
  cascade). The fix writes `<path>.json.tmp` then atomically renames it
  over the canonical `<path>.json`, mirroring the tmp+`.replace` idiom
  `train_scorer` (scorer.pkl.tmp), the outcomes-file trim, and the
  validation persister already use — all of which document the same class
  of "a process kill mid-write would corrupt the artifact" failure.
- **Test-locked in
  `tests/test_backtest.py::TestVolumeCachePersistAtomicity`:**
  `test_persist_writes_canonical_path_atomically` (canonical file exists +
  no tmp shadow remains after a successful persist),
  `test_torn_tmp_does_not_corrupt_canonical_path` (a half-written `.tmp`
  from a simulated prior crash is silently overwritten by the next
  successful persist — canonical content is the new payload, no torn data
  leaks), `test_load_ignores_tmp_shadow_filename` (loader opens the
  canonical `.json` path directly and never reads a corrupt `.json.tmp`
  shadow even when both exist on disk).

### Per-ticker scorer skill diagnostic (2026-05-19)

- **`paper_trader/ml/per_ticker_skill.py` (new read-only diagnostic).**
  `sector_skill` answers "is the calibration uniform across the seven
  sectors?" but the next quant question is one level finer: *within a
  sector, which individual tickers does the scorer actually predict well,
  and which is it actively wrong on?* `build_features` carries a 7-way
  sector one-hot but **no per-ticker identity**, so two tickers in the
  same sector are forced to share sector weights even if their realized 5d
  behaviour diverges sharply (e.g. NVDA vs INTC, both `sector_tech`).
  Empirically the live `decision_outcomes.jsonl` tail is also ticker-
  concentrated (top 10 tickers carry ~70% of outcomes — SOXL / TQQQ /
  AMZN / MSTR / MSFT dominate), so the headline `oos_ic` from the scorer
  ledger is essentially a weighted average of a handful of leveraged-ETF
  names. The diagnostic buckets the temporal-OOS rows by ticker (same
  `validation.split_outcomes_temporal` the scorer ledger uses) and
  produces per-ticker `n_train` / `n_oos` / `mean_pred` vs `mean_realized`
  (magnitude bias) / `rmse` / `dir_acc` / tie-aware
  `rank_ic` (via the shared `calibration._spearman` — never drifts from
  `sector_skill` / `persona_skill` / `_oos_rank_metrics`) plus a per-
  ticker verdict (`SPARSE`/`INVERTED_SIGNAL`/`SIGNAL_EDGE`/
  `WEAK_SIGNAL_EDGE`/`NO_SIGNAL_EDGE`) and an overall verdict
  (`HAS_INVERTED_TICKER` — actionable red flag, the gate is *actively
  harmful* on that name / `NO_TICKER_EDGE` / `HEALTHY` /
  `INSUFFICIENT_DATA`). Output is sorted by `rank_ic` desc with `SPARSE`
  rows sunk to the bottom and capped at `MAX_TICKERS_IN_REPORT=50`; the
  separate `inverted_tickers` list is **uncapped** so a red-flag name far
  down the rank-IC sort is never silently dropped. Read-only and never
  raises — same operational discipline as the rest of `paper_trader/ml`:
  no `decision_scorer.pkl` / `build_features` / `N_FEATURES` / trade-path
  touch, safe under the live unattended continuous loop. CLI:
  `python3 -m paper_trader.ml.per_ticker_skill` with exit codes mirroring
  `sector_skill._cli` — `0` on `HEALTHY`/`NO_TICKER_EDGE`/
  `INSUFFICIENT_DATA`, `1` on `SCORER_UNTRAINED`/other recoverable
  error, `2` on `HAS_INVERTED_TICKER` (a cron can branch on it).
- **Test-locked in `tests/test_per_ticker_skill.py`** (28 exact-value
  cases mirroring the `sector_skill` test shape): verdict thresholds at
  `IC_MIN`/`IC_GOOD`/`MIN_OUTCOMES_PER_TICKER`; universal SELL sign-flip
  on realized; non-finite / string / `None` `forward_return_5d` dropped;
  scorer-`predict` exception drops only that row; perfectly-correlated
  ticker → `rank_ic≈1.0` / `dir_acc=1.0` / `SIGNAL_EDGE`; anti-correlated
  ticker → `INVERTED_SIGNAL` and overall `HAS_INVERTED_TICKER`; inverted
  outranks no-edge in the overall verdict; ticker case normalised so
  mixed-case external rows bucket together; empty-ticker rows dropped;
  report capped at `MAX_TICKERS_IN_REPORT` with `tickers_truncated=True`
  but the `inverted_tickers` list stays complete; JSONL loader skips
  unparseable lines and non-dict tops; `analyze()` end-to-end against an
  empty file yields `INSUFFICIENT_DATA`; CLI exit codes 0/1/2 per verdict.

  > **Quant finding (2026-05-19, this pass).** Run against the live
  > ~7400-row `decision_outcomes.jsonl` corpus the diagnostic surfaces
  > **one INVERTED_SIGNAL ticker: `XLE`** (`n_oos=30`, `rank_ic=-0.277`,
  > `dir_acc=40%`, `mean_pred=+8.32%` vs `mean_realized=-1.58%` — the
  > scorer is loudly bullish on energy while realised 5d energy returns
  > were flat-to-negative). That is the operational definition of "the
  > gate is worse than no gate" on this name. Several `SIGNAL_EDGE`
  > names anchor the gate's overall positive contribution: `SPXL`
  > (`rank_ic +0.495 / dir_acc 76%`), `SOXL` (the dominant training name
  > at `n_train=558`, `rank_ic +0.294 / dir_acc 65%`), `CURE`
  > (`+0.278 / 57%`), `TQQQ` (`+0.271 / 62%`), `LLY` (`+0.264 / 62%`),
  > `NVO` (`+0.223 / 63%`), `UPRO` (`+0.206 / 57%`). The headline scorer
  > rank-IC the ledger reports is essentially the cap-weighted average of
  > these names; a targeted fix would be to **exclude XLE from the gate
  > or retrain with rebalanced ticker exposure** — but that is a separate
  > deliberate decision (CLAUDE.md §6 scope), this read-only diagnostic
  > only surfaces the data. The deployed pickle is currently `n_train=400`
  > on the local checkout (vs ~3987 in production on the USB-mounted
  > scorer.pkl) — production reads will surface the same INVERTED finding
  > against a larger trained corpus.

### `_VOLUME_CACHE` bounded LRU eviction (2026-05-19)

- **`paper_trader/backtest.py::_VOLUME_CACHE_DISK_LOADED` is now an
  `OrderedDict` (LRU bookkeeping), bounded to the most recent
  `_VOLUME_CACHE_MAX_WINDOWS = 16` (start, end) windows.** Before this
  pass, every continuous-loop cycle picked a fresh random window and the
  in-memory `_VOLUME_CACHE` / `_VOLUME_CACHE_DISK_LOADED` accumulated
  forever — roughly 30 quant-signal tickers × ~250 daily volumes × 8 bytes
  ≈ 60 KB / window; ~144 cycles / 24 h ≈ 8.6 MB / day, ≈ 60 MB / week of
  resident memory leaking with no clean release path on a 14 GB host.
- **LRU on access:** `_load_volume_cache_for_window` calls `move_to_end`
  when a previously-seen window is hit, so an actively-replayed window
  doesn't get evicted under the cap. New eviction helper
  `_evict_oldest_volume_windows_locked()` (called only under
  `_VOLUME_CACHE_LOCK`) drops both the bookkeeping entry AND every
  per-ticker series whose key matches the evicted (start, end) — memory
  is actually reclaimed, not just bookkeeping. On-disk caches at
  `data/backtest_cache/volumes_<start>_<end>.json` are untouched; a
  revisited window pays one disk read, not a yfinance fetch.
- **Test-locked in `tests/test_backtest.py::TestVolumeCacheBoundedLRU`:**
  `test_evicts_oldest_when_cap_exceeded` (eviction triggers at cap+1 and
  purges per-ticker series, not just the bookkeeping set),
  `test_access_refreshes_lru_order` (touching an old window moves it to
  most-recently-used so the next eviction drops a different victim).
  `conftest.py` and `tests/test_variable_windows.py` updated to seed an
  empty `OrderedDict()` instead of an empty `set()` for test isolation —
  same `key in …` / `len(…)` contract, plus the `move_to_end` /
  `popitem(last=False)` API the eviction helper needs.

### 52-week position outcome capture (2026-05-19)

- **`_compute_decision_outcomes` now additively records `wk52_pos`
  (0..1 trailing 52-week position) + `pct_from_52h`** alongside the
  existing quant features. Both are already computed by
  `_compute_technical_indicators` (`paper_trader/backtest.py`) and
  `wk52_pos` is the threshold driving `_ml_decide`'s **bubble-top BUY
  gate** (`wk52_pos > 0.80` → BUY suppressed; see invariant in
  `backtest._ml_decide`). Until this pass, neither value was persisted to
  `decision_outcomes.jsonl` — so the documented bubble-top gate
  explanation could never be empirically checked against realized
  forward returns by any downstream tool. The capture is purely additive:
  `train_scorer`/`build_features` consume explicit kwargs and ignore
  extra dict keys, so the scorer pickle, `N_FEATURES`, and every existing
  OOS diagnostic are byte-identical (the `forward_return_10d/20d`
  precedent). Legacy outcome rows have no `wk52_pos` key; new rows
  populate from the next continuous cycle.
- **Honest None semantics.** When `_compute_technical_indicators` returns
  `None` for a ticker (history < 60 closes at sim_date), `wk52_pos` /
  `pct_from_52h` are captured as **`None`** in the outcome row — *not* a
  0.0 sentinel, which would be indistinguishable from a real
  ticker-at-52w-low signal and silently poison the gate-explanation
  analysis. Test-locked in
  `tests/test_continuous.py::TestWk52PosCapturedInOutcomes`:
  `test_wk52_pos_field_present_when_history_sufficient` (monotonic
  80-day fixture → `wk52_pos ≈ 1.0`, `pct_from_52h ≈ 0%`),
  `test_wk52_pos_is_none_when_history_insufficient` (synthetic_prices
  fixture's 51-day series → keys present, values `None`),
  `test_capture_does_not_break_existing_keys` (full schema regression
  lock: every previously-documented outcome field must still appear).

### Baseline-trend reader (2026-05-18)

- **`paper_trader/ml/baseline_trend.py` (new read-only diagnostic).**
  `_append_baseline_skill_log` (committed `6ade72d`) writes one row per
  cycle to `data/baseline_skill_log.jsonl` carrying the decisive
  `ic_gap = MLP_rank_ic − best_one_liner_rank_ic` column — but **nothing
  read it**. `skill_trend.py` trends the *scorer-skill* ledger
  (`oos_rmse` vs a constant mean-predictor); the baseline ledger, which
  captures the single most economically-decisive recurring finding
  (`MLP_WORSE_THAN_TRIVIAL` — a one-liner out-ranks the 17-feature MLP
  the conviction gate sizes on), had no trender. This is the exact
  ledger-wired-but-unread gap the pass-#17 `skill_trend` addition closed
  for the sibling ledger; `baseline_trend` is its counterpart. It loads
  the baseline ledger, takes the **median** `ic_gap` over the recent
  window (window-specific `ic_gap` noise is large — median, not mean),
  and returns an exact verdict. `IC_MARGIN` / `MLP_IC_MIN` are
  **imported from `baseline_compare`** (single source of truth — this
  trends *that* tool's per-cycle verdict, so the margins must match by
  construction; the `_oos_rank_metrics`-reuses-`_spearman` precedent).
  Verdicts (exact-value test-locked, `MIN_CYCLES=5`,
  `RECENT_CYCLES=10`): `INSUFFICIENT_DATA` (< 5 usable rows — a row is
  usable iff `status=="ok"` AND `ic_gap` is finite, so a scorer-untrained
  `INSUFFICIENT_DATA` cycle with `ic_gap=None` is correctly excluded,
  mirroring `skill_trend`'s null-`oos_rmse` skip) → `MLP_WORSE_THAN_TRIVIAL`
  (recent median `ic_gap ≤ −IC_MARGIN`) → `MLP_ADDS_SKILL` (recent median
  `ic_gap ≥ +IC_MARGIN` AND recent median `mlp_rank_ic > MLP_IC_MIN` —
  the same dual gate `baseline_compare.MLP_ADDS_SKILL` uses) →
  `MLP_NO_BETTER_THAN_TRIVIAL` (otherwise). `trend`
  `IMPROVING/DEGRADING/STABLE/UNKNOWN` is recent-vs-older median `ic_gap`
  (higher = better). Also surfaces `most_common_best_baseline` (which
  one-liner keeps winning — on the live corpus this is `ml_score`, the
  decisive detail: the net destroys the signal it is fed),
  `gate_active_fraction`, and recent medians of `mlp_rank_ic` /
  `best_baseline_ic` / `n_train`. Read-only, never raises (same
  operational discipline as the rest of the module). CLI exit mirrors
  the sibling whose verdict it trends (`baseline_compare`): `0` on
  `MLP_ADDS_SKILL` / `INSUFFICIENT_DATA`, `2` on
  `MLP_WORSE_THAN_TRIVIAL` / `MLP_NO_BETTER_THAN_TRIVIAL`, so a cron can
  branch on "the net *persistently* fails to earn its complexity". CLI:
  `python3 -m paper_trader.ml.baseline_trend`. Locked by
  `tests/test_baseline_trend.py` (24 exact-value cases: single-source
  margin identity · inclusive/strict verdict boundaries at
  `±IC_MARGIN`/`MLP_IC_MIN` · null-`ic_gap` & non-`ok` usable-filter ·
  even-length median arithmetic · `most_common_best_baseline` ·
  trend axis independent of verdict axis · CLI exit codes).

  > **Quant finding (2026-05-18, this pass).** `baseline_trend` itself
  > reports `INSUFFICIENT_DATA` live — `data/baseline_skill_log.jsonl`
  > does **not exist on disk**: the running continuous loop (PID
  > `1734916`, booted `01:11 UTC`) predates `6ade72d` (the
  > `_append_baseline_skill_log` wiring, `10:11 UTC`), so it writes
  > `scorer_skill_log.jsonl` (14 rows, up to cycle 09:42 UTC) but **not**
  > the baseline ledger. This is the documented stale-loop operational
  > state, **not a code bug** — the trender is correct and will populate
  > a verdict once the operator restarts the loop. The point-in-time
  > picture (deployed pickle `n_train≈3860`, 1000-row temporal-OOS
  > slice) is unchanged and consistent across every arbiter:
  > `baseline_compare` = **`MLP_NO_BETTER_THAN_TRIVIAL`** (MLP rank_ic
  > +0.069 vs `ml_score` +0.111, gap −0.042); `skill_trend` =
  > **`NEGATIVE_OOS_SKILL`** (oos_rmse 11.30 ≥ fresh baseline 9.51,
  > median oos_ic 0.02, **trend DEGRADING**, `gate_active=1.0`);
  > `calibration --oos` = **`MISCALIBRATED`** (in-sample
  > `WELL_CALIBRATED` but OOS spearman 0.069, decile-realized column flat
  > noise — d1 mean_pred −34.49 realized −2.73 vs d10 mean_pred +22.49
  > realized +1.64; textbook overfit). One window-specific counterpoint:
  > `regime_audit` read **`REGIME_UNIFORM_EDGE`** on *this* draw
  > (sideways rank_ic +0.129, bull_or_unknown +0.063) — but the larger
  > bull bucket sits below `baseline_compare`'s `MLP_IC_MIN=0.10` floor,
  > so it is a fragile borderline artifact, not a stable edge,
  > consistent with `skill_trend`'s DEGRADING. **All reported
  > observations, not model changes** — altering the MLP/gate is a
  > training-dynamics change out of surgical scope (CLAUDE.md §6).

### Generalization-gap (val vs OOS) trender (2026-05-18)

- **`paper_trader/ml/overfit_gap.py` (new read-only diagnostic).** The
  scorer-skill ledger persists BOTH `val_rmse` (the random-split
  in-sample-ish error `train_scorer` reports) and `oos_rmse` (the
  temporal-holdout error). AGENTS.md cites `val_rmse ≪ oos_rmse` as
  *"textbook overfit"* repeatedly, and HEAD commit `5a0af2d`
  ("regularize DecisionScorer MLP — (32,16)+L2+early-stop kills the val≪oos
  overfit") exists solely to close that gap — yet **nothing trended the gap
  itself**: `skill_trend` verdicts `oos_rmse` vs a fresh mean-predictor
  baseline (only *reports* `recent_median_val_rmse` as a side metric, no gap
  verdict); `baseline_trend` verdicts `ic_gap` (a different axis). So a
  skeptical quant could not durably answer the one question `5a0af2d` is
  supposed to settle. This module does, with an exact verdict, off the same
  ledger (no `decision_outcomes.jsonl` read needed — both RMSEs are already
  persisted per cycle). The verdict is driven by the **ratio**
  `oos_rmse / val_rmse`, not the absolute `oos − val` pp: the loop draws
  random 1–10yr windows whose target σ varies several-fold, so an
  absolute-pp gap conflates regime σ with overfit; the ratio is scale-free.
  Aggregates the **median** ratio over the recent window (per-cycle ratio is
  noisy — one random window each). Reuses `skill_trend.load_skill_ledger` /
  `_median` / `MIN_CYCLES` / `RECENT_CYCLES` **verbatim** (single source of
  truth — the `baseline_trend`-imports-`baseline_compare` precedent; a
  ledger-schema change can never make this verdict and `skill_trend`'s
  disagree about which rows count). Verdicts (exact-value test-locked,
  module constants `SEVERE_RATIO=1.40`, `MILD_RATIO=1.15`,
  `RATIO_TOL=0.10`): `INSUFFICIENT_DATA` (< `MIN_CYCLES` usable rows — a row
  is usable iff `status=="ok"` AND `val_rmse` finite & > 0 AND `oos_rmse`
  finite, so the numpy-lstsq `val_rmse=NaN` fallback rows are correctly
  excluded) → `SEVERE_OVERFIT` (recent median ratio ≥ 1.40 — OOS error ≥40%
  above in-sample, the memorizing-net signature; the prior (64,32,16) net's
  oos≈16.7/val≈10.7≈1.56 sat here) → `MILD_OVERFIT` ([1.15, 1.40)) →
  `WELL_GENERALIZED` (< 1.15 — `5a0af2d`'s stated goal once it deploys).
  Boundaries inclusive at the lower edge (`>=`), matching `skill_trend`'s
  style. `trend` `IMPROVING/DEGRADING/STABLE/UNKNOWN` is recent-vs-older
  median ratio (lower = better). Surfaces `gate_active_fraction` (over ALL
  rows, like `skill_trend`): `SEVERE_OVERFIT` AND `gate_active=1.0` is the
  "underwriting sizing variance on a demonstrably memorized net right now"
  state. Read-only, never raises (same operational discipline as the rest
  of the module). CLI exit mirrors the sibling trenders so a cron can branch
  on "the net is *persistently* memorizing its training fold": `0` on
  `WELL_GENERALIZED` / `INSUFFICIENT_DATA`, `2` on `MILD_OVERFIT` /
  `SEVERE_OVERFIT`. CLI: `python3 -m paper_trader.ml.overfit_gap`. Locked by
  `tests/test_overfit_gap.py` (24 exact-value cases: single-source symbol
  identity · usable-row filter (non-ok/null/NaN/≤0-val excluded) ·
  inclusive/strict verdict boundaries at `±SEVERE_RATIO`/`MILD_RATIO` ·
  even-length median arithmetic · trend axis independent of verdict axis ·
  `gate_active_fraction` counts every row not just usable · never-raises on
  non-dict rows · `analyze` missing-file / JSONL load · CLI exit codes).

  > **Quant finding (2026-05-18, this pass).** Live on the wired
  > `data/scorer_skill_log.jsonl` (16 usable cycles, all on the *pre*-`5a0af2d`
  > unregularized net — the running loop PID 1734916 booted before the
  > commit): `overfit_gap` = **`MILD_OVERFIT`** (recent median oos/val ratio
  > **1.28**, older 1.18, overall 1.20, **trend STABLE**,
  > `gate_active=1.0`). Independently corroborated the same pass by
  > `calibration --oos` on the deployed pickle (in-sample `WELL_CALIBRATED`
  > but **temporal OOS `MISCALIBRATED`** — spearman 0.122, decile-realized
  > column flat: d1 mean_pred −20.45 realized −2.53 vs d10 mean_pred +17.66
  > realized +2.14) — the val/oos ratio reflects the *same* overfit the OOS
  > decile view shows, a built-in cross-check that the new instrument
  > measures something real. The decisive operational point: the ratio is
  > **STABLE at ~1.28, not improving** — because the regularization commit
  > `5a0af2d` has **not deployed** (stale loop). `overfit_gap` is now the
  > durable instrument to verify whether `5a0af2d` actually moves the ratio
  > toward `WELL_GENERALIZED` once the operator restarts the loop; until
  > then a skeptical quant should treat the gate as sizing on a moderately
  > memorized net (`gate_active=1.0` every cycle). **Reported observation,
  > not a model change** (CLAUDE.md §6 scope).

### Feature-coverage audit (2026-05-18)

- **`paper_trader/ml/feature_coverage.py` (new read-only diagnostic).** Every
  existing skill arbiter (`calibration` / `skill_trend` / `baseline_compare` /
  `feature_importance` / `regime_audit`) measures whether the *model* extracts
  skill from its inputs; **none ask whether the inputs carry any variation in
  the training data at all**. This is the upstream, decisive data-pipeline
  question — and `feature_importance` *structurally cannot* answer it: you
  cannot permute a near-constant column into measurable importance, so a dead
  feature reads as `0.0` importance identically to one the model merely
  ignores. `feature_coverage` disentangles "the model ignores a real signal"
  from "the signal was never in the data". Each `decision_outcomes.jsonl`
  record is pushed through the **exact** `build_features` call shape
  `train_scorer` uses (same kwargs, same clamps), and the per-slot default
  vector is derived by calling `build_features` with every numeric source
  `None` (single source of truth — a default change in `decision_scorer.py`
  cannot silently desync it; the `baseline_trend`-imports-`baseline_compare`
  precedent). Per numeric feature it reports `default_fraction` (rows whose
  post-`build_features` value equals the default), `distinct`, and a `dead`
  flag. Verdicts (exact-value test-locked, module constants `MIN_ROWS=30`,
  `DEAD_FLOOR=0.90`, `DEGRADED_FLOOR=0.50`): `INSUFFICIENT_DATA` →
  `DEAD_FEATURES_PRESENT` (≥1 feature default-substituted in ≥90% of rows OR
  constant <2 distinct) → `DEGRADED_COVERAGE` (no dead but ≥1 ≥50%) →
  `FULL_COVERAGE`. The 7-way sector one-hot is excluded (sparse *by
  construction* — exactly one hot per row — so a default fraction is
  meaningless there; `feature_importance` permutes it jointly for the same
  reason). Read-only, never raises (same operational discipline as the rest
  of the module). CLI exit mirrors the sibling diagnostics so a cron can
  branch on "the gate's MLP is being fed dead inputs right now": `0` on
  `FULL_COVERAGE` / `INSUFFICIENT_DATA`, `2` on `DEAD_FEATURES_PRESENT` /
  `DEGRADED_COVERAGE`. CLI: `python3 -m paper_trader.ml.feature_coverage`.
  Locked by `tests/test_feature_coverage.py` (17 exact-value cases:
  single-source default identity · missing/corrupt JSONL load · all four
  verdict boundaries · constant-non-default dead rule · dead-overrides-
  degraded · never-raises on garbage field types · CLI exit codes · the live
  news-feature shape).

  > **Quant finding (2026-05-18, decisive).** Live on
  > `data/decision_outcomes.jsonl` (7538 rows): verdict
  > **`DEAD_FEATURES_PRESENT`**, **effective numeric dim ≈ 7/10**.
  > `news_urgency` is default-substituted in **97.3%** of rows and its
  > remaining 2.7% are *all `0.0`* (backtest articles structurally carry
  > `urgency=0` — CLAUDE.md invariant #2), so it is a 2-value
  > near-constant; `news_article_count` is at its `1.0` default in
  > **99.3%** of rows. The price/quant features are healthy
  > (`ml_score` 0.0, `rsi`/`macd`/`mom5`/`mom20`/`vol_ratio`/`bb_pos` all
  > ≈4–5% default), and `regime_mult` is degraded-not-dead (78.9% at the
  > `1.0` bull/unknown default). **Why this matters:** the documented
  > "17-feature MLP" is really an ~8-numeric + 7-sparse-one-hot model with
  > **two pure noise dimensions** the net must spend capacity fitting —
  > a concrete, upstream mechanism behind the repeatedly-documented
  > `MLP_WORSE_THAN_TRIVIAL` (raw `ml_score` out-ranks the net OOS) and the
  > `feature_importance` ≈0 readings for the news slots (it could never have
  > distinguished "dead input" from "ignored input" — this can). The
  > actionable thread for a future *training-dynamics* change (out of
  > surgical scope here, CLAUDE.md §6): the news features cannot help until
  > the backtest news pipeline supplies non-constant urgency/count for
  > historical windows; until then they are removable at zero skill cost.
  > **Reported observation, not a model change.**

### Per-persona Sortino + Calmar (2026-05-18)

- **`persona_leaderboard._equity_risk` now also returns `sortino` and
  `calmar`** (commit `35735d9`). The module already computed a per-persona
  median **Sharpe**, max drawdown, and %-time-underwater off
  `equity_curve_json` — but Sharpe divides excess return by *total*
  volatility, so it penalises a persona for the large **upside** variance
  leveraged ETFs are bought for. That is precisely the wrong risk lens for
  this book: AGENTS.md repeatedly warns that "a single persona routinely
  posts +1000% … leveraged-beta dispersion through a cherry-able bull
  window, **not** repeatable alpha". Sharpe alone cannot tell a smooth
  leveraged compounder from a lucky volatile rip. The two metrics that can:
  - **Sortino** — `mean(daily_ret) / downside_dev × √252`, where
    `downside_dev` is the MAR=0 convention `sqrt(mean(min(r,0)²))` over
    **all** observations (not the std of just the negative subset). Only
    realised downside is penalised, so the leveraged right tail no longer
    counts against a persona.
  - **Calmar** — annualised return / worst peak→trough drawdown (the
    unrounded `max_dd` fraction). Return-per-unit-of-worst-pain — the
    metric a capital allocator actually cares about ("did +1000% earn its
    70% drawdown, or just ride a bull window?").
  Both share Sharpe's 252-day annualisation base and its
  degenerate-→-`None` discipline: no measurable downside ⇒ Sortino `None`
  (parallel to the zero-std Sharpe `None`); no drawdown / non-positive
  start / total wipe-out (`growth ≤ 0`) ⇒ Calmar `None` — an "infinite"
  ratio must never silently dominate the per-persona median.
  `persona_leaderboard` surfaces `median_sortino` / `median_calmar` per
  persona; the CLI gains both columns. **Read-only and additive** — no
  train, no `decision_scorer.pkl` / `decision_outcomes.jsonl` /
  `build_features` / trade-path touch; the DRAG/FLAT/EDGE verdict logic is
  **byte-identical** (Sortino/Calmar are informational like
  `sharpe`/`mdd`/`uw`, never verdict inputs — so every existing
  exact-value verdict lock is unchanged). Exact-value locked by
  `tests/test_persona_leaderboard_20260517.py::TestSortinoCalmar`
  (Sortino == 0.0 on a mean-zero curve; Sortino < 0 choppy-down;
  Sortino/Calmar `None` on no-downside / no-drawdown; **Calmar == 1.0
  exactly** on a constructed 20%-dd→recover 252-step curve where
  years = 252/252 = 1.0; Calmar sign tracks net P&L; Calmar `None` on a
  non-positive start; per-persona median aggregation incl. missing-curve
  robustness). The garbage-input exact-dict lock was strengthened to all
  five metrics.

  > **Quant finding (2026-05-18, this pass — decisive operational
  > state).** The running continuous loop is on **stale code**: (a)
  > `data/baseline_skill_log.jsonl` does **not exist** on disk although
  > `_append_baseline_skill_log` is wired unconditionally into `main()`
  > (run_continuous_backtests.py:1904) and writes correctly when invoked
  > manually; (b) **0 / 2000** recent `decision_outcomes.jsonl` rows carry
  > `forward_return_10d` or `gate_scorer_pred` despite commits `ccc4d31`
  > (multi-horizon capture) and `60b20d9` (gate-decision capture). So the
  > multi-horizon capture, the gate-decision capture, the trivial-baseline
  > ledger, **and** the regularised-MLP commit `5a0af2d` are all inert
  > until the operator restarts `run_continuous_backtests.py`. This is the
  > documented stale-process pattern, **not a code bug** — but a quant
  > relying on `baseline_trend` / `horizon_audit` / `gate_realized` should
  > know those instruments are not accumulating live data yet.
  >
  > The OOS picture is unchanged and consistent across every live arbiter
  > (deployed pickle `n_train=3997`, 1507-row temporal-OOS slice):
  > `baseline_compare` = **`MLP_NO_BETTER_THAN_TRIVIAL`** (MLP rank_ic
  > +0.060 vs one-line `mom20` +0.082, gap −0.023; raw `ml_score` alone
  > +0.0525 / dir_acc 0.54); `overfit_gap` = **`MILD_OVERFIT`** with
  > **trend=DEGRADING** (recent median oos/val ratio **1.38**, older 1.18,
  > overall 1.21, 18/18 cycles `gate_active=1.0`) — a *sharper* statement
  > than prior passes' STABLE ~1.28; `scorer_skill_log` `oos_ic` ≈
  > 0.02–0.08, `oos_dir_acc` ≈ 0.47–0.56 (coin-flip). The conviction gate
  > (invariant #5) sizes real position variance on this every cycle.
  > **Reported observation, not a model change** (CLAUDE.md §6 scope).
  >
  > **Operational note.** `python3 -m paper_trader.ml.persona_leaderboard`
  > (and any full `_load_runs` read of `backtest.db`) is effectively
  > unusable against the *live* writer: `backtest.db` is symlinked to a
  > slow external mount and `_load_runs` eagerly JSON-parses every
  > `complete` run's full multi-year `equity_curve_json` under the
  > continuous loop's WAL write contention — even a bounded
  > `ORDER BY run_id DESC LIMIT 12` read timed out at 60 s. This is a
  > **pre-existing** scalability characteristic, NOT introduced by the
  > Sortino/Calmar change (both are O(n) over the already-parsed `vals`).
  > Correctness is proven by the 26 exact-value unit tests; run the
  > leaderboard off a `backtest.db` snapshot, not against the live loop.

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

# Bootstrap CIs on OOS skill (paper_trader/ml/skill_uncertainty.py) —
# verdict locks (SKILL_DETECTED vs NO_SKILL_DETECTED), SELL sign-flip
# semantics, NaN-row drop, seed reproducibility, crash resilience.
cd /home/zeph/paper-trader && python3 -m pytest tests/test_skill_uncertainty.py -v

# Quick OOS-skill significance check vs the live pickle. Exit 0 only when
# the rank-IC CI strictly excludes 0 — so shell callers can gate on
# distinguishable skill instead of noisy point estimates.
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.skill_uncertainty

# Per-persona strategy-quality leaderboard (exact-value verdict locks;
# incl. TestSortinoCalmar — the 2026-05-18 Sortino/Calmar risk-metric add)
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

# Per-action scorer-skill diagnostic (the action-conditional sibling of
# persona_skill / regime_audit — answers "is the scorer's OOS edge on the
# BUY half (the gate-relevant slice, invariant #5), the SELL half, or
# both?"). The aggregate _oos_rank_metrics hides this asymmetry; an
# ASYMMETRIC_SELL_EDGE verdict means the aggregate OVERSTATES what the
# gate actually uses. Live read: ASYMMETRIC_BUY_EDGE (BUY rank_ic
# +0.214 / EDGE, SELL +0.088 / WEAK_EDGE on the 20% OOS holdout) — the
# gate IS on the skilled slice. Exit 2 only on HAS_INVERTED_ACTION (an
# anti-predictive slice). 26 exact-value verdict & threshold locks.
cd /home/zeph/paper-trader && python3 -m pytest tests/test_action_skill.py -v
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.action_skill          # OOS slice (read-only)

# Multi-horizon outcome capture + forward-return-horizon predictability
# audit (exact-value verdict + IC locks; the only file with "horizon" in
# its node ids — silently missed by the older "ml/backtest/scorer" filter)
cd /home/zeph/paper-trader && python3 -m pytest tests/test_horizon_audit.py -v
# Is the scorer's ~0 OOS skill a 5d-target-noise artifact? (read-only):
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.horizon_audit          # OOS slice
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.horizon_audit --all    # full in-sample

# Baseline-trend reader — trends the per-cycle baseline ledger's ic_gap
# (MLP − best one-liner OOS rank-IC). The counterpart to skill_trend for
# the baseline ledger; the canonical durable instrument for the
# MLP_WORSE_THAN_TRIVIAL question (24 exact-value verdict locks)
cd /home/zeph/paper-trader && python3 -m pytest tests/test_baseline_trend.py -v
# Is the MLP STILL net-negative complexity, and improving or worsening?
# (read-only; exit 2 on MLP_WORSE/NO_BETTER, 0 on ADDS_SKILL/INSUFFICIENT):
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.baseline_trend

# Generalization-gap (val vs OOS) trender — does the ledger's val_rmse≪oos_rmse
# "textbook overfit" persist, and is 5a0af2d closing it? (24 exact-value locks)
cd /home/zeph/paper-trader && python3 -m pytest tests/test_overfit_gap.py -v
# Is the scorer still memorizing its training fold? (read-only; exit 2 on
# MILD/SEVERE_OVERFIT, 0 on WELL_GENERALIZED/INSUFFICIENT):
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.overfit_gap

# Feature-coverage audit — how many of the 10 numeric MLP features actually
# carry variation in the training data (the upstream question feature_importance
# structurally cannot answer; 17 exact-value verdict locks)
cd /home/zeph/paper-trader && python3 -m pytest tests/test_feature_coverage.py -v
# Is the gate's MLP being fed dead/constant noise dimensions right now?
# (read-only; exit 2 on DEAD/DEGRADED, 0 on FULL_COVERAGE/INSUFFICIENT):
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.feature_coverage

# Off-distribution gate-abstention diagnostic — how often does the live
# gate's off_distribution guard (commit 84d8234) actually fire? The
# gate_off_dist field is captured per row but no diagnostic reports the
# rate; this answers "is the ±PRED_CLAMP_PCT protection live or dormant?"
# (25 exact-value verdict locks)
cd /home/zeph/paper-trader && python3 -m pytest tests/test_gate_abstention.py -v
# Read-only; exit 2 on GUARD_RAMPANT (gate mostly neutral), 0 elsewhere:
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.gate_abstention          # OOS slice
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.gate_abstention --all    # full corpus

# Training-corpus & OOS-construction audit (exact-value verdict locks)
cd /home/zeph/paper-trader && python3 -m pytest tests/test_corpus_audit.py -v
# Is the loop's temporal-OOS holdout a real generalization test? (read-only;
# exits 2 on OOS_NOT_HELD_OUT — the corpus is one cycle's single window):
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.corpus_audit

# Scorer response-shape / monotonicity audit (exact-value verdict locks;
# test_response_audit.py has none of "ml"/"backtest"/"scorer" in its node
# ids — add it explicitly like test_calibration.py / test_gate_pnl.py)
cd /home/zeph/paper-trader && python3 -m pytest tests/test_response_audit.py -v
# ICE-then-average: which way (and how hard) does the model bend each
# feature? Complements feature_importance (importance vs response-shape).
# exits 2 on FLAT_NO_RESPONSE / RESPONSIVE_JAGGED (read-only):
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.response_audit          # OOS slice
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.response_audit --all    # full in-sample

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

`test_baseline_trend.py` (2026-05-18 pass — the baseline-ledger trend
reader `paper_trader/ml/baseline_trend.py`: 24 exact-value cases —
`IC_MARGIN`/`MLP_IC_MIN` are the *same object* as `baseline_compare`'s
(single-source-of-truth identity assert, not just value equality);
inclusive `ic_gap ≤ −IC_MARGIN` WORSE boundary vs the −0.04 just-inside
case; the `MLP_ADDS_SKILL` dual gate (`ic_gap ≥ +IC_MARGIN` AND
`mlp_rank_ic > MLP_IC_MIN`) with the strict-floor `mlp_rank_ic == 0.10`
→ NO_BETTER case; the usable-row filter excluding both non-`ok` rows and
`status=="ok"` rows with `ic_gap=None` — the scorer-untrained
`INSUFFICIENT_DATA` cycle shape, mirroring `skill_trend`'s
null-`oos_rmse` skip; even-length `np.median` arithmetic pinned exactly
on a mixed `ic_gap`/`mlp_rank_ic`/`best_baseline_ic` set;
`most_common_best_baseline`; the trend axis proven independent of the
verdict axis (`MLP_WORSE_THAN_TRIVIAL` + `trend=IMPROVING`); CLI exit
codes 0/2 via monkeypatched `analyze`).

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

---

### 2026-05-18 review pass #16 (ML+backtest hybrid · scorer response-shape audit · decisive rsi-inversion finding)

- **Phase 1 — no new production bug (bugs_fixed = 0).** Full re-trace of
  `decision_scorer.py`, `backtest.py`, `run_continuous_backtests.py` plus
  coupled `validation.py` / `calibration.py`: the `predict_with_meta`
  off-distribution abstention, the universal SELL `-forward_return_5d`
  flip (train↔inference↔calibration↔gate↔`_oos_rank_metrics`↔
  `evaluate_scorer_oos`), the `(ticker,sim_date,action)` dedup key, the
  5-trading-day forward-window guard, the `score=`/`scorer=` first-match
  disambiguation, the numpy-lstsq weighted-LS fallback, the atomic
  tmp+`.replace` JSONL/pickle trims, the `_inject_and_train` 11-col
  INSERT tuple, every module-global lock — all re-verified correct and
  exact-value test-locked. Consistent with the documented 14+ prior
  no-new-bug ML/backtest passes. **Phase 1 deliverable (commit
  `b82f09e`, `test:`):** the existing `TestRiskExits` cases only asserted
  *an exit happened* (`n_exits==1`, `triggered_price >= 120.0`) — they did
  not pin the price/day the SL/TP daily scan fires at, so an off-by-one in
  the scan boundary (`cur = from_day + timedelta(days=1)`, `px <= sl` vs
  `px < sl`, partial- vs whole-position sell) slipped through. Added two
  exact-value regression locks against the deterministic synthetic series
  (`SPY[days[i]] == 100.0 + i`): TP fires at `days[20]`/**120.0** with cash
  exactly **600.0**; SL fires at `days[1]`/**101.0** (NOT `days[0]`/100.0 —
  locking the deliberate one-day scan offset) with cash exactly **901.0**.
  Additive; existing tests untouched. ML/backtest regression 280→321 green.

- **Feature shipped (`paper_trader/ml/response_audit.py`; landed in commit
  `b471188` via the shared-index race — see concurrency note).** Every
  documented inertness verdict (`skill_trend` ≈0 OOS, `gate_audit`
  GATE_INEFFECTIVE, `gate_pnl` ≈+0.1pp, `calibration --oos`
  MISCALIBRATED) is a **statistical summary**. `feature_importance`
  reports *how much* skill scrambling a feature costs but is **sign-blind**
  — it cannot say a feature the model relies on is bent the economically
  *backwards* way. `response_audit` is the missing **geometric**
  complement: **ICE-then-average** (per-record curves over the OOS slice's
  empirical p5..p95, all other features kept REAL then averaged — NOT
  PDP-at-median, which would fabricate off-distribution combos and measure
  the clamped-±50 head instead of learned structure). Primary verdict is
  **sign-agnostic** (`FLAT_NO_RESPONSE` / `RESPONSIVE_MONOTONE` /
  `RESPONSIVE_JAGGED` / `INSUFFICIENT_DATA`); the economic-sign tally is
  **informational only, never in the verdict** (the SELL flip makes the
  target a BUY/SELL blend, so a "wrong" sign is not provably a defect —
  the `gate_audit` arm-monotone honesty pattern). Reuses
  `calibration._spearman` (tie-aware — load-bearing at the ±`PRED_CLAMP_PCT`
  clamp) and `validation.split_outcomes_temporal` (SSOT — same OOS slice as
  every sibling tool). Read-only: no train/pickle/`build_features`/
  `N_FEATURES`/trade touch, never raises, CLI exit 2 on FLAT/JAGGED.
  **NOT wired into `main()` — zero deploy-stale impact, no loop restart
  needed.** 26 exact-value locks in `tests/test_response_audit.py`
  (ICE-vs-PDP mean≠median lock, monotone-but-wrong-sign still
  RESPONSIVE_MONOTONE, constant→FLAT, symmetric-U→JAGGED, FLAT_TOL
  boundary, degenerate-feature handling, SSOT identity, never-raises on
  raising/NaN/untrained/garbage, full CLI exit-code matrix).
  ```bash
  cd /home/zeph/trading-intelligence/paper-trader && python3 -m paper_trader.ml.response_audit
  cd /home/zeph/trading-intelligence/paper-trader && python3 -m paper_trader.ml.response_audit --all
  cd /home/zeph/trading-intelligence/paper-trader && python3 -m pytest tests/test_response_audit.py -v
  ```

- **Quant finding (NEW, decisive — the geometric mechanism for the
  documented near-zero OOS skill).** Live pickle `n_train=3894`, gate
  active. The scorer is **NOT flat-noise** — it moves materially (max
  response ~6.9pp) — but its single **largest** learned lever is
  **economically inverted**: `rsi` has the biggest response range of any
  feature (~6.9pp OOS / ~6.8pp in-sample) and the model bends it
  **spearman +0.82 OOS / +0.95 in-sample** — *higher* predicted 5d return
  for *higher/overbought* RSI (momentum-on-RSI, the **opposite** of the
  mean-reversion prior). Only `bb_position` is consistently coherent
  (spearman −1.0 OOS / −0.82 in-sample, sign ✓ both slices). Verdict
  degrades **RESPONSIVE_MONOTONE in-sample (4/7 sign-consistent) →
  RESPONSIVE_JAGGED OOS (3/7 monotone, 2/6 sign-consistent)**: `mom5`/
  `mom20` flip slope sign across the temporal split — the textbook overfit
  signature, now shown geometrically rather than as another scalar IC.
  No prior tool localized *which* learned relationship drives the ~0 OOS
  skill; this does: the model's dominant signal is backwards and its
  momentum response is sign-unstable. **Reported, not actioned** —
  retraining/feature surgery is training-dynamics, out of surgical scope
  (CLAUDE.md §6).

- **Quant findings (corroborating, fresh live numbers).**
  `decision_outcomes.jsonl` now **7858 rows** (BUY 5978 / SELL 1880),
  `forward_return_5d` mean +1.35 std 7.47 p1 −18.0 p99 +23.9, only
  **0.10% exceed |50%|** — re-confirms `PRED_CLAMP_PCT=50` is amply
  load-bearing-safe. **98.3% have `news_article_count = NULL`** →
  `response_audit` independently flags BOTH news features `degenerate` on
  every slice **and additionally `regime_mult` degenerate on the deep-
  history OOS slice** (the 1996–2018 corpus is uniformly "unknown→1.0"
  regime) — 2–3 of 17 inputs carry no training variance, sharpening pass
  #15's "2 dead features". `calibration --oos` MISCALIBRATED (spearman
  0.088, decile err 5.4pp) vs in-sample DIRECTIONAL_BUT_BIASED (spearman
  0.289) — the textbook in-sample→OOS collapse. `gate_pnl` OOS
  GATE_RETURN_NEUTRAL, equal-weight contribution **+0.11pp** (gate-on
  +0.38% vs gate-off +0.26%) — the gate underwrites pure sizing variance.
  `scorer_skill_log` last 6 cycles oos_ic ∈ {0.07,0.01,0.19,0.02,0.02,
  −0.01}, oos_dir_acc ≈ 0.50. Backtest dispersion stays leverage-driven
  (run 6230 +484.75%/vs_spy +396.7% beside run 6231 −49.44%).

- **Operational (reconfirmed, out of scope).** Continuous loop healthy:
  476 complete / 20 failed / 4 running, cycles 13–27 min, `scorer ok`
  every cycle (train_n 3234→3894). Winner→ArticleNet feedback loop still
  **dead both ways**: `continuous.log` shows `inject err: database locked
  after 4 attempts` and `trainer timeout (injected 2994)` — matches
  passes #6–#15 (digital-intern GPU + `articles.db` write contention on
  the symlinked volume). The scorer itself retrains cleanly; the loop is
  **not** training ArticleNet on its winners.

- **Concurrency note for the next agent.** This pass ran with ≥3 sibling
  agents committing in parallel. Observed concretely: a sibling's
  whole-index commit (`b471188 feat(strategy): … sector-exposure`) swept
  this pass's *already-staged* `response_audit.py` + `test_response_audit.py`
  into THAT commit before this agent's path-scoped `git commit -- <paths>`
  ran (which then found "no changes"). **The code/tests are durably on
  `origin/master` and 26/26 green** — only the commit-message attribution
  is the sibling's, not this agent's. Rewriting shared history with live
  concurrent agents is more dangerous than the misattribution, so it was
  left as-is. Lesson: in this monorepo a brand-new file is NOT safe from
  the shared index either — a sibling's `git add -A`/whole-tree commit
  captures anything staged. There is no fully race-free path short of a
  per-agent worktree; verify your deliverable is on `origin` by content
  (`git cat-file -e origin/master:<path>` + line-count), not by assuming
  it sits in your own commit.

- **Run the ML/backtest suite (now 321 in the listed subset):** `cd
  /home/zeph/trading-intelligence/paper-trader && python3 -m pytest
  tests/test_decision_scorer.py tests/test_backtest.py
  tests/test_calibration.py tests/test_validation.py tests/test_continuous.py
  tests/test_gate_audit.py tests/test_gate_pnl.py tests/test_skill_trend.py
  tests/test_baseline_compare.py tests/test_ml_backtest_review.py
  tests/test_feature_importance.py tests/test_response_audit.py -q`.
  `test_response_audit.py` has none of "ml"/"backtest"/"scorer" in its
  node ids — add it explicitly like `test_calibration.py` /
  `test_gate_pnl.py`.

*Review pass #16 (ML+backtest hybrid) appended 2026-05-18. Prior content above is unmodified.*

---

### 2026-05-18 review pass #16 (paper-trader core hybrid · decision-context advisory-block omission · capital-paralysis Discord pulse · live findings)

- **Phase 1 — 1 bug fixed (in HEAD via commit `5f40009`; see Concurrency
  note).** `analytics/decision_context.py` — `/api/decision-context`
  (the operator's **only** window into "what is the live trader actually
  shown right now?", whose docstring promises a string "byte-identical to
  the live prompt given identical inputs", single-source-of-truth
  invariant #10) reconstructed the prompt via `strategy._build_payload`
  but **never threaded `event_calendar_block` (forward earnings) or
  `buying_power_block` (deployable cash)** — both wired into the real
  `decide()` (`buying_power` since `b739a14`; `event_calendar` earlier).
  The inspector silently dropped **2 of 6 advisory blocks**: a trader
  curl-ing the endpoint to audit "did Opus get the buying-power /
  upcoming-earnings awareness this cycle?" got a false **NO**, and
  `advisory_blocks` lacked both keys entirely (a `KeyError` for any
  consumer iterating the documented set). Root cause: the inspector was
  built before those two blocks were added to `decide()` and never
  updated — exactly the class of regression that escapes a per-file
  review. Fix: add both kwargs to `build_decision_context`, pass them to
  `_build_payload`, report them in `advisory_blocks`, update the
  `__main__` CLI line, and build them in `assemble_inputs` **mirroring
  `decide()` byte-for-byte** (`event_calendar` scope = held ∪ the FULL
  `WATCHLIST` — *not* the lean `_names_in_play` set, which would
  re-blind the reconstruction the same way it would re-blind the live
  desk; `buying_power` scoped to `_names_in_play`). **Verified live:** the
  reconstructed prompt grew `8984→10002` chars (exactly the two omitted
  blocks) and the CLI now reports `event_calendar=True buying_power=True`.
  Locked by 5 RED-before regression tests
  (`test_decision_context.py::TestNewAdvisoryBlocksReachPrompt` — verbatim
  text + flags + `_build_payload` byte-faithful ordering;
  `TestInputSummary::test_advisory_block_flags` updated to the 6-key
  contract; `test_decision_context_endpoint.py::…test_buying_power_block_
  reaches_reconstructed_prompt` — the `assemble_inputs` wiring).

- **Phase 2 — 1 feature.** `reporter._capital_pulse_line` — a
  **capital-paralysis pulse in the hourly / daily-close Discord
  summary**. The #2 documented live pathology (pass #14 #4): a book
  pinned near 98% deployed with ~$18 free, unable to act for a day while
  involuntary NO_DECISION droughts bleed alpha. `capital_paralysis`
  serves this on the **dashboard** and `buying_power` now reaches the
  **Opus prompt** — but the operator, who lives in **Discord**, still got
  hourly/daily summaries that never said the desk was frozen and bleeding.
  This routes the existing builder's own verdict to the surface the
  operator actually reads (the exact dashboard→prompt→Discord trajectory
  `buying_power` followed, one surface over). Composes
  `build_capital_paralysis` **verbatim** (single source of truth,
  invariant #10 — headline / unlock / verdict are the builder's, never
  re-derived, so this line, `/api/capital-paralysis` and the prompt-side
  `buying_power` can never drift). **Pure store reads, NO network** (the
  Discord-path discipline — unlike `_benchmark_line` it adds zero
  latency). Observational only, no caps, never gates (invariants #2/#12;
  the `_hold_discipline_line` / `_benchmark_line` precedent). Suppression:
  `NO_DATA` and a healthy `FREE`-not-bleeding book are silent (nothing
  actionable); `PINNED`/`EMPTY` are **always** surfaced, and — the key
  subtlety — a `FREE` book whose involuntary-drought verdict is
  `BLEEDING` IS surfaced. That is the **live 2026-05-18 state**:
  `can_act_on_signal:true` (→ state `FREE`) masks that the desk has bled
  **-2.21% alpha across 6 involuntary droughts**; the verbatim live
  render is `**CAPITAL** ◈ FREE / > FREE — $18.49 cash (1.9%) available …
  / > 2.21% of alpha lost across 6 involuntary (parse-failure) droughts —
  the NO_DECISION problem is costing real performance`. Wired into both
  `send_hourly_summary` and `send_daily_close` (after the existing
  behavioural / hold-discipline blocks). Locked by 14 tests in
  `tests/test_capital_pulse.py` (suppression × NO_DATA/missing-state/
  FREE-healthy/non-dict/empty-headline; surfaced × PINNED-verbatim-with-
  unlock-and-reason / FREE-but-BLEEDING-live-state / minimal-no-unlock /
  garbage-frees_usd-no-crash; failure contract × builder-raises /
  store-raises → "" never raises; end-to-end × hourly + daily wiring +
  healthy-book-adds-no-noise regression).

- **Phase 3 — live findings (1–6; none a new quick safe code fix —
  finding 1 *became* the Phase-1 fix).**
  1. **`/api/decision-context` under-reported what Opus sees** — the
     Phase-1 bug, found by curl-ing the endpoint as a trader and diffing
     it against the `buying_power` smoke test (HAS BUYING POWER: False on
     the endpoint, fully rendered by the builder). Fixed this pass.
  2. **The live trader is UNSUPERVISED.** `/api/supervision` (the brand-
     new `dde6ee5`-era endpoint) correctly reports `orphan:true ppid:1
     supervised:false verdict:UNSUPERVISED`: the running pid (booted on
     `b82f09e`, current) is parented to init, **not** under
     `systemd --user` (`Failed to connect to bus: No medium found`). The
     git-watcher's deferred-restart + deadman force-exit logic
     **assumes** `systemd Restart=always` brings the process back on new
     code; an orphan with no supervisor that cleanly exits to apply a
     commit (or hits the deadman) just **dies**. Something external is
     currently relaunching it (boot_sha advanced `f29e134→b82f09e`
     mid-session) but there is no durable safety net. Real ops risk;
     **reported, not fixed** — touching the live process / systemd unit
     is out of scope and high-risk for a code pass. The new
     `/api/supervision` endpoint itself works correctly and is the right
     surface for this.
  3. **NO_DECISION ~59.5% lifetime / ~50% (24h)** — 458/770 lifetime,
     43/85 in 24h, uniformly `"claude returned no response
     (timeout/empty)"`; `quota_exhausted` unset. Unchanged, the
     long-standing host-load contention diagnosis (CLAUDE.md §11; passes
     #6–#15). Not a code defect; the lever is host load.
  4. **Capital paralysis active & bleeding (the Phase-2 justification).**
     `/api/capital-paralysis`: $18.49 cash (1.9%), 98.1% deployed, LITE
     61% of book, `cycles_since_last_fill 58`, last fill ~24h ago,
     `involuntary_alpha_bleed_pct -2.213` across 6 droughts, verdict
     `BLEEDING`. Real, ongoing, measurably costly — now visible in
     Discord (Phase 2).
  5. **`/api/capital-paralysis` self-contradiction in the [1%,2%) band,
     still present.** `can_act_on_signal:true` + `liquidity_status:
     NO_DRY_POWDER` + a `FREE — …the book can act…` headline while 98.1%
     deployed and `flags:["98.1% of book deployed", …]`. This is the
     **pass #15 finding #1** (the `liquidity.py` `can_act` vs
     `NO_DRY_POWDER` threshold disagreement in `cash_pct∈[1,2)`),
     **confirmed unchanged** at `cash_pct 1.9`. Still the
     "deliberately-weird, contested-builder, leave-it" category — a
     future pass that *does* touch `liquidity.py` should align the two
     thresholds and re-pin the `capital_paralysis` composition. NB: the
     Phase-2 pulse intentionally keys off the *paralysis* verdict
     (`BLEEDING`), **not** `can_act`, so it correctly surfaces this book
     despite the headline saying "FREE … can act".
  6. **MU position phantom-flat (stale mark, known/handled).** Live book:
     `MU stock qty=0.5 avg=724.12 now=724.12 P/L $+0.00` — the documented
     stale-mark case (yfinance returned no price → marked at cost,
     `stale_mark` flag set; surfaced as `[STALE MARK …]` in the prompt
     and `⚠ STALE` in `_portfolio_lines`). Working as designed; noted so
     a future agent does not misread the $0.00 P/L as a genuinely flat
     position.
  7. **Positives verified:** `/` 200 in 0.85s; `/api/runner-heartbeat`
     HEALTHY, singleton **acquired** (pid 1786434, not degraded),
     notify/Discord **HEALTHY**; `/api/build-info` not stale (boot==head);
     decisions on the 60m closed-market cadence (last 8m ago);
     `buying_power` smoke test renders the correct `CASH_CONSTRAINED`
     block verbatim on the live pinned book. The system is operationally
     sound; its real problems (host-load timeouts, data-corruption
     paralysis, no supervisor) are documented ops/data issues, not core
     code defects.

- **Concurrency note for the next agent — `git commit -- <pathspec>`
  defeats partial staging.** This pass ran with ≥3 sibling agents
  committing in parallel (observed: `feat(briefing)` `5f40009`,
  `notify_health`/`_record_send_outcome` in `reporter.py`, `dashboard.py`
  mark-trust/supervision, `sector_exposure.py`, parallel `AGENTS.md`
  appends). **Two attribution swaps happened, neither lost code:** (a)
  Phase-1's `decision_context.py` + 2 test files, after a clean
  `git add`, were swept into a sibling's `5f40009 feat(briefing)` commit
  before this agent's own commit landed; (b) Phase-2 carefully extracted
  *only* its `reporter.py` hunks into the index via `git diff | filter |
  git apply --cached --recount` (verified `git diff --cached |
  grep -c <sibling-token>` == 0) — but the subsequent
  `git commit -m … -- paper_trader/reporter.py …` **commits the
  working-tree file, not the index**, so the sibling's complete (and
  separately tested) `notify_health` work rode along under the Phase-2
  message. **Lesson:** the index-extraction discipline only holds if you
  commit the **index** (`git commit` with NO pathspec, after staging
  exactly your hunks and confirming `git diff --cached --name-only` +
  zero sibling tokens) — adding `-- <pathspec>` silently re-snapshots the
  whole working-tree file and re-imports sibling hunks. In this repo's
  trunk-based + auto-push-daemon model the *code* is never lost (every
  full-suite run stayed green: 1613→1640→1682) and history rewrite on
  `master` against the daemon is far more dangerous than muddied
  attribution — so accept it, verify HEAD parses + tests green, and move
  on (this pass did: `python3 -c "import ast; ast.parse(...)"` on
  `HEAD:reporter.py`, 94 targeted + 1682 full green).

- **Run the suite:** `cd /home/zeph/trading-intelligence/paper-trader &&
  python3 -m pytest tests/ -v` (full; ~50–200s). Fast subset for this
  pass: `python3 -m pytest tests/test_capital_pulse.py
  tests/test_decision_context.py tests/test_decision_context_endpoint.py
  tests/test_core_reporter.py -q`.

*Review pass #16 (paper-trader core hybrid) appended 2026-05-18. Prior content above is unmodified.*

---

### 2026-05-18 review pass #17 (ML+backtest hybrid · durable trivial-baseline ledger · live findings)

- **Phase 1 — no new bug (bugs_fixed = 0; no Phase-1 commit, commit guard
  explicitly permits).** Full re-trace of `decision_scorer.py`,
  `backtest.py`, `run_continuous_backtests.py` plus coupled
  `validation.py` / `calibration.py` / `baseline_compare.py` /
  `skill_trend.py`: the `predict_with_meta` off-distribution abstention,
  the universal SELL `-forward_return_5d` flip
  (train↔inference↔calibration↔gate↔`_oos_rank_metrics`↔
  `evaluate_scorer_oos`↔`baseline_compare`), the `(ticker,sim_date,
  action)` dedup key, the 5-trading-day forward-window guard, the
  `score=`/`scorer=`/`news_urg=`/`news_count=` reasoning regexes, the
  numpy-lstsq weighted-LS fallback (fits scaler on full X with
  `val_rmse=nan` — by-design fallback, locked by
  `test_ml_backtest_coverage.py`, not a leak bug), the atomic
  tmp+`.replace` JSONL/pickle trims, the `_inject_and_train` 11-col
  INSERT tuple + lock-retry, the singleton-reset under
  `_DECISION_SCORER_LOCK`, the `_LOAD_CACHE` (path,mtime_ns,size) key
  (atomic replace ⇒ key changes ⇒ per-cycle pickup), every module-global
  lock — all re-verified correct and exact-value test-locked. Consistent
  with the documented 15+ prior no-new-bug ML/backtest passes. No
  test-hardening commit either: the Phase-1 checklist items (known
  feature-vector score range / kw-rank ordering / null-safe defaults;
  synthetic BUY-and-hold + exact SL/TP price+cash — the latter freshly
  pinned by pass-#16 `b82f09e`; results-location + no-silent-overwrite)
  are already exact-value locked; a redundant test would be churn, not
  hardening. ML/backtest regression 374/374 green before the feature,
  380/380 after.

- **Feature shipped (commit `6ade72d`, `feat(continuous):`): durable
  per-cycle trivial-baseline ledger.** `run_continuous_backtests.py::
  _append_baseline_skill_log` + `BASELINE_SKILL_LOG`/`_KEEP` constants,
  wired into `main()` immediately after the scorer-skill-ledger block.
  **The gap it fills:** `baseline_compare`'s `MLP_WORSE_THAN_TRIVIAL`
  (raw `ml_score` carries higher OOS rank-IC than the 17-feature MLP the
  conviction gate relies on) is the single most economically decisive
  documented ML/backtest finding (~10 prior passes) — yet it was *only*
  observable by an operator manually running
  `python3 -m paper_trader.ml.baseline_compare`. There was **no durable,
  trendable signal** an unattended loop surfaced — the *exact*
  dead-audit-trail gap the pass-#15 `_append_scorer_skill_log` wiring fix
  closed for the scorer ledger (`scorer_skill_log.jsonl`/`skill_trend`),
  applied to the sibling decisive question. `scorer_skill_log` only ever
  trends `oos_rmse` vs a **constant** mean-predictor; *nothing* durably
  trended "does a one-liner beat the net this cycle". The new ledger row
  is `{cycle, timestamp, window_*, status, verdict, slice, n, n_train,
  mlp_rank_ic, mlp_dir_acc, best_baseline, best_baseline_ic, ic_gap,
  gate_active}`. **SSOT, never a re-derivation:** it calls
  `baseline_compare.analyze` *verbatim* — the same
  `validation.split_outcomes_temporal` slice + universal SELL sign-flip
  as `calibration --oos` and the scorer ledger's OOS metrics — so the
  persisted `mlp_rank_ic` equals the CLI's / `calibration --oos`'s **by
  construction** (a built-in no-drift cross-check, exact-value
  test-locked). `gate_active` mirrors invariant #5 (deployed
  `n_train ≥ 500`), so a `MLP_WORSE_THAN_TRIVIAL` + `gate_active=True`
  row is the quant-decisive *"the loop is sizing on a net the data says
  is worse than a free one-liner, right now"* state. Best-effort by
  construction (never raises — an untrained scorer / missing-or-short
  outcomes file / a raising `analyze` all degrade to an **honest**
  `status='error' verdict='INSUFFICIENT_DATA'` row so a trend gap is
  *visible*, never a silently-skipped cycle), atomic bounded trim at
  `BASELINE_SKILL_LOG_KEEP=2000` (the decision_outcomes idiom). **Applies
  on next `run_continuous_backtests.py` restart** (the running loop
  predates this commit — inert until restart, exactly the documented
  deploy-stale pattern). Ledger lives at `data/baseline_skill_log.jsonl`
  (gitignored-by-symlink like every sibling ledger — never staged). 6
  exact-value locks in `tests/test_continuous.py`
  (`TestAppendBaselineSkillLog`: SSOT cross-check vs
  `scorer_baseline_compare` — `mlp_rank_ic`/`best_baseline`/`ic_gap`/
  `verdict` must be byte-identical; honest untrained-`INSUFFICIENT_DATA`
  row; `analyze`-raises → honest error row + still returns True;
  past-2×-keep atomic trim with newest-survives; never-raises on
  unwritable path; + `TestCycleWiringRegression::
  test_main_invokes_baseline_skill_ledger` source-level wiring lock so a
  refactor can't silently re-orphan it the way the scorer ledger was
  until pass #15).
  ```bash
  cd /home/zeph/trading-intelligence/paper-trader && python3 -m pytest tests/test_continuous.py -v -k "Baseline or CycleWiring"
  # the CLI the ledger records, for ad-hoc reads:
  cd /home/zeph/trading-intelligence/paper-trader && python3 -m paper_trader.ml.baseline_compare
  ```

- **Quant finding (decisive — the gate underwrites pure sizing variance,
  now durably trended).** Live deployed pickle `n_train=3894`, gate
  active. The new ledger's first live row (smoke-tested on the real
  `decision_outcomes.jsonl`): **`MLP_WORSE_THAN_TRIVIAL`**, MLP OOS
  rank-IC **+0.0876** vs raw `ml_score` **+0.141** (`ic_gap −0.0534`,
  `n=1571`, `gate_active=true`) — byte-identical to the
  `baseline_compare` CLI (the SSOT cross-check holding live). Four
  independent OOS arbiters agree on the same slice: `gate_pnl`
  **`GATE_RETURN_NEUTRAL`** (gate-on +0.38% vs gate-off +0.26%,
  equal-weight contribution **+0.11pp** — "pure added sizing variance"),
  `calibration --oos` **MISCALIBRATED** (spearman 0.088, decile err
  5.4pp, OOS decile-realized flat noise), `skill_trend`
  **`NEGATIVE_OOS_SKILL`** (recent median oos_rmse 11.30 ≥ fresh
  mean-predictor baseline 8.20, **trend DEGRADING**, `gate_active=1.0`).
  Fresh `scorer_skill_log` cycles 1–3 today corroborate the textbook
  overfit: `train_n` 3485→3870→3894 growing, `val_rmse ≪ oos_rmse`
  (9.0 vs 12.76), `oos_ic` {0.02, 0.02, −0.01}, `oos_dir_acc ≈ 0.50`.
  **Reported, not actioned** — turning the gate off / retraining is a
  training-dynamics change out of surgical scope (CLAUDE.md §6). The
  feature's contribution is making this decisive finding *durable and
  per-cycle trendable* for the first time, not changing the model.

- **Quant findings (corroborating, fresh live numbers).**
  `decision_outcomes.jsonl` now **7858 rows** (BUY 5978 / SELL 1880),
  `forward_return_5d` mean +1.35 std 7.47, **0 non-finite rows** — the
  `_to_float` poison-row guard (load-bearing: one non-finite
  `forward_return_5d` wedges retraining indefinitely) is holding clean.
  **98.3% have `news_article_count = NULL`** → `news_urgency`/
  `news_article_count` pinned at their constant `build_features`
  defaults (the deep-history 1996–2018 windows pre-date
  `digital-intern/articles.db` coverage) — 2 of 17 inputs carry **zero
  training variance**, a concrete mechanism for the documented "the net
  destroys the signal it is fed". Backtest dispersion stays
  **leverage-beta, not skill**: same-recent-cycle run 6230
  +484.8%/vs_spy +396.7% beside run 6231 −49.4%/vs_spy −12.4% — the
  "best run" cycle line must never be read as strategy skill (the
  ledgers / permutation suite are the arbiters). 476 complete / 24
  failed runs; continuous loop process live, `continuous.log` fresh
  (mid-cycle), scorer retrains cleanly every cycle.

- **Operational (reconfirmed, out of scope).** Winner→ArticleNet
  feedback loop still **dead both ways**: `continuous.log` shows
  `[continuous] ml: inject err: database is locked after [4 attempts]`
  and `trainer timeout` / `trainer rc=-15` (digital-intern GPU +
  `articles.db` write-contention on the symlinked `/media` volume —
  matches passes #6–#16; the loop is **not** training ArticleNet on its
  winners; the scorer itself retrains cleanly). Exactly **one** Python
  traceback in the entire log — a transient
  `sqlite3.OperationalError: locking protocol` on `PRAGMA
  journal_mode=WAL` during a `BacktestStore()` init at *yesterday's*
  cycle-1 startup — and it is **correctly handled**: `main()`'s engine-
  init `try/except` caught it, logged `engine init failed … locking
  protocol`, slept 30s and continued; cycle 2 proceeded normally
  immediately after. Same symlinked-volume lock-contention class as the
  feedback-loop failure, not a code defect, not in this domain's
  surgical scope.

- **Concurrency note for the next agent.** This pass ran with ≥1 sibling
  agent active (a parallel `claude --model claude-opus-4-7` process
  observed in `ps`). The two changed files (`run_continuous_backtests.py`,
  `tests/test_continuous.py`) were path-scoped `git add`-ed (never
  `git add -A`), staged diff verified additions-only (`+280 / -0`, 2
  files), and the deliverable confirmed on `origin/master` **by content**
  (`git cat-file -e origin/master:<path>` + symbol grep), not by
  assuming it sits in this agent's commit message — the pass-#16
  shared-index-race lesson applied pre-emptively.

- **Run the ML/backtest suite (now 380 in the listed subset):** `cd
  /home/zeph/trading-intelligence/paper-trader && python3 -m pytest
  tests/test_decision_scorer.py tests/test_backtest.py
  tests/test_calibration.py tests/test_validation.py tests/test_continuous.py
  tests/test_gate_audit.py tests/test_gate_pnl.py tests/test_skill_trend.py
  tests/test_baseline_compare.py tests/test_ml_backtest_review.py
  tests/test_feature_importance.py tests/test_response_audit.py
  tests/test_horizon_audit.py tests/test_corpus_audit.py
  tests/test_regime_audit.py -q`. `test_continuous.py` holds the new
  `TestAppendBaselineSkillLog` + the extended `TestCycleWiringRegression`
  baseline-ledger wiring lock.

*Review pass #17 (ML+backtest hybrid) appended 2026-05-18. Prior content above is unmodified.*

---

## Review pass #17 — paper-trader CORE hybrid (2026-05-18)

**bugs_fixed = 1 · features_added = 1 · user_findings = 3.** Full suite is
not run end-to-end here (it times out >400s under concurrent sibling-agent
pytest load); a bounded representative sweep of every touched + adjacent
module is the evidence: **312 green** across `test_runner_heartbeat`,
`test_decision_context{,_endpoint}`, `test_sector_exposure{,_endpoint}`,
`test_core_runner`, `test_core_strategy`, `test_core_reporter`,
`test_decision_forensics`, `test_core_dashboard_helpers`.

- **Phase 1 — bug (`6cfcf46`, this agent's own clean commit).
  `analytics/decision_context.py` silently dropped the
  `sector_exposure_block`.** Commit `b471188` added `sector_exposure_block`
  to `strategy._build_payload` + wired it into the live `decide()` call but
  never updated `decision_context.py` — so `/api/decision-context` (and the
  CLI) reconstructed a prompt **missing the entire SECTOR EXPOSURE block**
  while its docstring still promised a string "byte-identical to the live
  prompt given identical inputs". A trader auditing *"did Opus see that
  this BUY piles onto an already-61%-semis book?"* got a false NO. **This
  is the exact `event_calendar`/`buying_power` regression class closed in
  pass #16, reintroduced one block later** — the recurring failure mode is
  "a new advisory block is threaded into `decide()`→`_build_payload` but
  the parallel `decision_context.assemble_inputs`/`build_decision_context`
  reconstruction is forgotten". `assemble_inputs` now builds
  `sector_exposure_block` exactly as `decide()` does (same read-only
  snapshot + lean `_names_in_play` set); `build_decision_context` threads
  it through `_build_payload` (which owns render order, so byte-fidelity is
  preserved) and reports it in `advisory_blocks` + the CLI summary. Locked
  by `tests/test_decision_context.py`
  (`TestNewAdvisoryBlocksReachPrompt::test_sector_exposure_block_reaches_prompt_verbatim_and_flagged`
  — verbatim text + flag + the exact `risk<sector<event<bp<WATCHLIST`
  ordering; `test_advisory_block_flags`/`…_omitted` updated to the
  now-7-key dict) and `tests/test_decision_context_endpoint.py`
  (`test_sector_exposure_block_reaches_reconstructed_prompt` — end-to-end
  via `assemble_inputs`). **Guidance for the next agent: any future
  `_build_payload` advisory block MUST be added to `decision_context.py`
  in the same commit — there is now a 3rd instance of this exact bug
  class; treat it as a standing checklist item.**

- **Phase 2 — feature. `runner_heartbeat` NO_DECISION-storm awareness so a
  brain-dead loop is no longer flat green** (this agent's code, swept into
  `4bd6610` by the auto-push daemon — muddied attribution, content verified
  on `origin/master` by symbol grep, the pass-#16 shared-index-race reality
  accepted not fought). `build_runner_heartbeat` measured only loop
  *cadence* (`now − last_decision_ts`); a loop cycling perfectly on
  schedule but emitting `NO_DECISION` every cycle — the documented live
  regime — reported `HEALTHY … restart_recommended:false`. The heartbeat
  is the surface a trader checks **first**, so it was actively reassuring
  them while the engine was wedged, suppressing the exact restart signal
  the runner's own auto-recovery breaker fires on. Additive overlay: the
  endpoint now passes `recent_actions` (`store.recent_decisions(20)`
  newest-first — the thesis_drift network/builder split); the builder
  computes a `decision_efficacy` sub-block (`PRODUCING`/`DEGRADED`/
  `IDLE_STORM`/`NO_DATA`) and on a genuine idle-storm
  (`>= NO_DECISION_STORM_THRESHOLD=5` consecutive — mirrors
  `runner.CONSECUTIVE_NO_DECISION_LIMIT`, drift-locked by test) folds it
  into the top-level `headline` + `restart_recommended`. **The liveness
  `verdict` enum is deliberately untouched** (the documented
  liveness/efficacy separation; every verdict-string lock stays green) and
  omitting `recent_actions` is **byte-identical** to before. The
  NO_DECISION predicate is a drift-locked verbatim mirror of the canonical
  `decision_forensics._is_no_decision` (invariant #10), inlined to keep
  this endpoint-path leaf import-cycle-free (the `OPEN_INTERVAL_S`
  precedent). 12 new tests in `tests/test_runner_heartbeat.py`.

- **Phase 3 — live findings (validated on the running `:8090` /
  pid-1819043 trader; both Phase-1 & Phase-2 changes confirmed *deployed*
  — git-watcher rebooted the runner onto the new SHA, `build-info`
  `stale:false`).**
  1. **Live IDLE_STORM, ongoing (HIGH).** `/api/runner-heartbeat`
     (now, via the Phase-2 feature) reports
     `decision_efficacy:IDLE_STORM consecutive_no_decision:17 (95% of
     last 20) restart_recommended:true` — and the last *FILLED* trade was
     `2026-05-17T09:38 BUY MU`, **>24h ago**, while the book is ~98%
     deployed in MU/LITE. Involuntary alpha bleed, real and current. Root
     cause is the documented host-load timeout storm (CLAUDE.md §11)
     **aggravated by git-watcher restart-thrash**: with N sibling agents
     committing every few minutes this session, the deferred-restart
     watcher bounces the runner repeatedly, and each restart abandons the
     in-flight Opus call mid-cycle. Not a code-fixable defect (host/ops
     lever) — but pass #17's Phase-2 is precisely what makes it *visible*
     instead of a lying green light.
  2. **Runner relaunch-and-refuse churn (MEDIUM, ops).** `logs/runner.log`
     shows ~49 `refusing to start a second trader` lines per 200 — the
     single-instance guard (invariant #19) is working **correctly** (exactly
     one trading pid 1819043 holds the flock; **no double-trade**), but a
     supervisor keeps attempting relaunches in the restart gaps. Confirms
     the pass-#16 "unsupervised / externally relaunched" ops state, still
     present. Reported, not fixed — touching the systemd unit / live
     process is out of scope and high-risk for a code pass.
  3. **Positive validation (LOW).** `/api/decision-context` now serves
     `advisory_blocks.sector_exposure:true` and `"SECTOR EXPOSURE"` is in
     the reconstructed prompt (7/7 blocks; the Phase-1 fix is live, proven
     via both the endpoint and the `__main__` CLI). All probed endpoints
     sub-100ms (`/` 0.035s, `/api/state` 0.004s, `/api/risk` 0.002s,
     `/api/sector-exposure` 0.002s, `/api/scorecard` 0.059s — SWR
     healthy); Discord delivery `notify:HEALTHY`. The data/dashboard layer
     is operationally sound; the system's real problem is finding #1.

- **Run the core suite (bounded, the full one times out under concurrent
  load):** `cd /home/zeph/trading-intelligence/paper-trader && python3 -m
  pytest tests/test_runner_heartbeat.py tests/test_decision_context.py
  tests/test_decision_context_endpoint.py tests/test_sector_exposure.py
  tests/test_sector_exposure_endpoint.py tests/test_core_runner.py
  tests/test_core_strategy.py tests/test_core_reporter.py
  tests/test_decision_forensics.py tests/test_core_dashboard_helpers.py -q`
  (312 green).

*Review pass #17 (paper-trader core hybrid) appended 2026-05-18. Prior content above is unmodified.*

---

## Feature: decision-loss clock (feature-dev agent, 2026-05-18)

**Gap.** Every pass since #6 documents the same #1 unsolved problem —
NO_DECISION ~60% driven by `TIMEOUT_EMPTY` (the `claude` CLI starving
under host load when the hourly self-review + continuous-backtest loops
contend for the 3-subprocess OOM cap). It is exhaustively *measured*
(`decision_health`/`_forensics`/`_drought`/`_reliability`/
`capital_paralysis`) and AGENTS.md repeats "the lever is host load" —
but nothing told the operator **which clock hours to deconflict**.
`decision_forensics.hourly` is a sparse 24h *calendar* timeline: it shows
*today* spiked at 14:00, never that 14:00 spikes *every* day.

**What shipped.** Purely-additive extension of
`analytics/decision_forensics.py` — folds the **current-regime** decision
history onto a 24h UTC clock:

- New keys on `build_decision_forensics` (and the existing
  `/api/decision-forensics` route, no new endpoint): `regime_boundary`,
  `hour_of_day` (sparse 0–23, `{hour,total,failures,fail_pct}`),
  `hour_of_day_window_n/_failures`, `hour_of_day_min_sample`,
  `worst_hours` (top-3, min-sample-gated), `clock_hint` (actionable, only
  when a worst hour beats the window rate by ≥`CLOCK_HINT_MARGIN_PP`).
- `_regime_boundary()` derives `max(ts where classify_failure().tag ==
  'legacy')` from *this module's* own `classify_failure`/`_parse_ts` —
  the **identical regime contract** `decision_reliability` partitions on
  (no circular import; no touch to that contested file). Verified on the
  live DB: forensics `regime_boundary` == the live
  `/api/decision-reliability` boundary (`2026-05-15T17:42:42…`) —
  the clock and the reliability headline can never tell different
  stories. The legacy-inflated `hourly`/`by_market`/verdict are
  byte-unchanged (additive only; observational, never gates — invariants
  #2/#12).
- Dashboard: a 24h clock mini-chart + hint line added to the existing
  Decision-failure-forensics card (`#df-clock*`).
- **Live-validated** (Flask test client, the live DB): 133 current-regime
  cycles, recurring morning-UTC storm — 06:00 87% (n=8), 10:00 100%
  (n=7) vs 42% window-wide; 12:00–14:00/17:00/23:00 clean. The 11:00
  100%/n=2 bucket is correctly min-sample-suppressed.

**Tests.** New `tests/test_decision_loss_clock.py` (16, behaviour-
asserting incl. the regime-window property test + same-story parity vs
`build_decision_reliability`). Full suite green: 1690→1717 (0 failed;
baseline taken at HEAD before the change for concurrent-agent noise
isolation).

*Decision-loss-clock feature appended 2026-05-18. Prior content above is unmodified.*

---

## Review pass #18 — paper-trader CORE hybrid (2026-05-18)

**bugs_fixed = 0 · features_added = 2 · user_findings = 4.** Bounded
representative sweep (the full suite times out >280s under concurrent
sibling-agent pytest load — the documented host state): **433 green**
across `test_core_reporter`, `test_runner_heartbeat{,_swr}`,
`test_core_state_swr`, `test_dashboard_threaded`, `test_core_runner{,_cycle}`,
`test_core_strategy`, `test_core_signals`, `test_core_store`,
`test_core_market`, `test_decision_forensics`,
`test_core_dashboard_helpers`. Two sibling ML+backtest agents and the
auto-push daemon were active; both feature commits were path-scoped
(`git add <file>`, never `-A`), staged-diff verified additions-only, and
confirmed on `origin/master` **by content** (`git show
origin/master:<path> | grep <symbol>`), not by commit attribution (the
pass #16/#17 shared-index-race lesson applied).

- **Phase 1 — no bug (honest `bugs_fixed = 0`).** A full read of
  `runner.py` / `reporter.py` / `signals.py` / `strategy.py` /
  `market.py` / `store.py` plus targeted `dashboard.py` probes found no
  genuine logic defect — consistent with 17 prior polished passes (pass
  #17 core found 1, a self-introduced regression; pass #17 ML found 0).
  Per the Phase-1 commit guard, no Phase-1 commit was made. Manufacturing
  a cosmetic "fix" to pad the counter was explicitly declined.

- **Phase 2 — feature 1 (`b7e0b5c`). `reporter._heartbeat_line` —
  runner-heartbeat verdict in the hourly/daily Discord summary.** Pass #17
  made a host-load NO_DECISION storm visible on `/api/runner-heartbeat`
  (the *dashboard*), but the operator **lives in Discord** and the
  hourly/daily summary still looked flat-green while the engine was wedged
  (the live 2026-05-18 state: 19/20 cycles NO_DECISION,
  `restart_recommended:true`). `send_quota_alert` covers only the
  *distinct* quota-exhaustion freeze (a specific `quota_exhausted` flag);
  a host-load idle storm had **no Discord surface at all**. The new line
  composes `build_runner_heartbeat` **verbatim** (single source of truth,
  invariant #10 — same `store.recent_decisions(20)` read +
  `market.is_market_open` split the endpoint uses, so the Discord line and
  `/api/runner-heartbeat` can never tell different stories), appended to
  `send_hourly_summary` + `send_daily_close` right after
  `_singleton_lock_line`. Surfaces **only when actionable**
  (`restart_recommended` / STALLED / LAGGING liveness / DEGRADED
  efficacy) so a healthy deciding loop adds no hourly noise (the summary
  must not become its own lying green light); IDLE_STORM detail is already
  folded into the builder's top-level headline so only DEGRADED gets an
  additive `efficacy —` sub-line. Observational only, never gates, no caps
  (invariants #2/#12 — the `_capital_pulse_line`/`_singleton_lock_line`
  precedent); a builder/store fault drops the line, **never** the summary
  (the reporter "no block, never no summary" failure contract). Locked by
  `tests/test_core_reporter.py::TestHeartbeatLine` (11 tests: verbatim-
  headline + restart-prefix on IDLE_STORM/STALLED, LAGGING-without-prefix,
  DEGRADED efficacy sub-line, HEALTHY+PRODUCING suppressed, builder-raises
  → `""`, a **no-drift real-builder** lock on a real Store seeded with an
  18-deep NO_DECISION storm, and the hourly+daily integration +
  fault-still-sends contract).

- **Phase 2 — feature 2 (`784201f`). SWR-cache `/api/runner-heartbeat`.**
  The surface a trader checks *first*, polled every 60s by the dashboard
  JS, was the **last high-traffic core endpoint not behind `swr_cached`**
  (the invariant #7 gap `/api/state` closed). Measured **9.45s under load
  avg 23** (the documented host-load storm) versus ~1ms warm — a pure DB +
  module-global read with no network, so the latency is pure CPU
  starvation, exactly what SWR absorbs. `@swr_cached("runner-heartbeat",
  20.0)`: runner cadence is ≥1800s/3600s with ≥1.25x/2x verdict
  multipliers and IDLE_STORM needs ≥5 cycles × ≥1800s, so a ≤20s stale
  window can **never** flip the verdict; the dashboard thread is
  independent of the runner thread, so a dead runner still gets a fresh
  background recompute of `secs_since_last_decision` from the frozen
  `last_decision_ts` and correctly goes STALLED — **SWR never masks the
  death it detects**. Inert under pytest unless `_SWR_TEST_FORCE`, so the
  existing `tests/test_runner_heartbeat.py` endpoint tests stay green on
  the live path. Locked by `tests/test_runner_heartbeat_swr.py` (mirrors
  `test_core_state_swr.py`: cold full-verdict + honesty keys; warm hit
  serves the **stale alarming** payload after the storm cleared — the
  discriminating "not silently recomputed every poll" lock;
  pytest-inert-by-default isolation). Like every recent feature, **applies
  on the next paper-trader restart** — a stale `:8090` keeps the old
  uncached path until then (`/api/build-info` `stale`).

- **Phase 3 — live findings (running `:8090`, build-info `stale`,
  `behind:2`, boot `18b40ec`).**
  1. **Live IDLE_STORM, ~25h since last trade (HIGH, host/ops, not
     code-fixable).** `/api/runner-heartbeat` reports `verdict=HEALTHY
     restart_recommended=true decision_efficacy=IDLE_STORM
     consecutive_no_decision=19`; the last *FILLED* trade was
     `2026-05-17T09:38 BUY MU`, **>25h ago**, the book ~98% deployed
     (cash $18.49 / total $972.69) in MU/LITE and frozen. Documented
     host-load timeout storm aggravated by git-watcher restart-thrash
     (the runner pid bounced 1822266→1827108 within this session). **This
     pass's feature 1 is precisely what now makes this visible in Discord
     instead of a flat-green hourly summary; feature 2 keeps the
     diagnostic surface instant under the same load.**
  2. **Running process is 2 commits stale (MEDIUM, documented).**
     `build-info` `boot 18b40ec / head 784201f / stale:true / behind:2`;
     the live heartbeat curl returns `cached:null` (old uncached path) —
     proof both new features are on disk but **not yet live**; they
     activate on the next restart (the invariant #7/#11 "does not rescue
     a running process" pattern, flagged so the operator restarts to
     arm the Discord heartbeat line + endpoint SWR).
  3. **`GOOGU` & `METAU` are permanently delisted but still in
     `strategy.WATCHLIST` *and* the SYSTEM_PROMPT "LEVERAGE INSTRUMENTS
     AVAILABLE" list (LOW, data hygiene).** Recurring yfinance 404s every
     `market._DEAD_TTL` window clutter `runner.log`; `_mark_dead`
     suppresses re-fetch so there is **no functional harm**, but Opus is
     still told two dead names are tradeable. **Reported, not fixed** —
     the trading universe + its mirrored prompt text is a judgment call
     outside a surgical pass (a WATCHLIST-only edit that left the prompt
     inconsistent would be worse); flagged per the task's "otherwise
     just report it".
  4. **Singleton-guard relaunch churn continues (MEDIUM, ops,
     pre-existing).** A supervisor keeps relaunching in restart gaps;
     the invariant-#19 guard correctly refuses every duplicate (**no
     double-trade**, `singleton_lock:acquired degraded:false`). Pass #17
     finding #2, still present; out of code scope.
  - **Positive validation.** Dashboard/data layer operationally sound:
    every probed panel (`/api/state` 0.004s, `/api/risk` 0.001s,
    `/api/decision-health` 0.012s, `/api/benchmark` 0.004s,
    `/api/capital-paralysis` 0.009s) HTTP 200 sub-13ms; Discord delivery
    `notify:HEALTHY` (0 consecutive failures, last OK recent) — the new
    heartbeat line WILL deliver on the next hourly. The system's only
    real problem is finding #1 (host load), not the code.

- **Run the core suite (bounded — the full one times out under concurrent
  load):** `cd /home/zeph/trading-intelligence/paper-trader && python3 -m
  pytest tests/test_core_reporter.py tests/test_runner_heartbeat_swr.py
  tests/test_runner_heartbeat.py tests/test_core_state_swr.py
  tests/test_dashboard_threaded.py tests/test_core_runner.py
  tests/test_core_strategy.py tests/test_core_signals.py
  tests/test_core_store.py tests/test_core_market.py -q`.
  `TestHeartbeatLine` (reporter) + `TestRunnerHeartbeatSwr` are the new
  pass-#18 locks.

*Review pass #18 (paper-trader core hybrid) appended 2026-05-18. Prior content above is unmodified.*

---

### 2026-05-18 review pass #18 (ML+backtest hybrid · gate-decision capture · live findings)

- **Phase 1 — no new bug (bugs_fixed = 0; no Phase-1 commit, commit guard
  explicitly permits).** Full re-trace of `decision_scorer.py`,
  `backtest.py`, `run_continuous_backtests.py` plus the newest, least-
  reviewed diagnostic modules (`baseline_trend.py`, `horizon_audit.py`,
  `gate_pnl.py`, `response_audit.py`, `corpus_audit.py`) and coupled
  `validation.split_outcomes_temporal` / `evaluate_scorer_oos`:
  the `_aligned` SELL `-forward_return` flip is applied to **both** probe
  and target uniformly in `horizon_audit._horizon_skill` (no spurious
  anti-correlation on the SELL subset); `baseline_trend` correctly
  excludes non-`ok`/null-`ic_gap` rows and never divides by zero on the
  small live slice (verified live: returns `INSUFFICIENT_DATA` on the
  *absent* `baseline_skill_log.jsonl`, no crash — the AGENTS.md claim
  holds); `_horizon_skill` drops non-finite/`nan` labels rather than
  zero-coercing them and guards `np.std==0` before `_spearman`;
  `_fwd_ret_h` gates on `price_on() is None` exactly like the byte-
  identical 5d path (no fabricated-zero asymmetry between the 5d and
  10d/20d windows); `split_outcomes_temporal` is deterministic and
  stable. Consistent with the documented 16+ prior no-new-bug
  ML/backtest passes. No test-hardening commit either — the Phase-1
  checklist items are already exact-value locked; a redundant test would
  be churn, not hardening (the pass-#17 precedent). ML/backtest
  regression 363/363 green before the feature, 372/372 after.

- **Feature shipped (commit `60b20d9`, `feat(continuous):`): the gate's
  ACTUAL then-deployed decision is now captured in
  `decision_outcomes.jsonl`.** `run_continuous_backtests.py::
  _parse_gate_decision` (pure/total/never-raises — the
  `_parse_scorer_status` discipline) + additive `gate_scorer_pred` /
  `gate_off_dist` keys in `_compute_decision_outcomes`. **The gap it
  fills:** `_compute_decision_outcomes` already parsed `score=` /
  `news_urg=` / `news_count=` out of each decision's reasoning but
  **discarded the gate's own `scorer=±X%` token** (and the
  `(off-dist,gate-skipped)` abstention marker `_ml_decide` emits). Every
  gate diagnostic (`gate_audit`, `gate_pnl`) therefore had to RE-predict
  with **today's** deployed pickle on the stored features — a
  counterfactual ("what would the current model say"), provably **not**
  what the gate did at decision time with that cycle's *own* model;
  `gate_pnl` itself documents the resulting reconstruction residual is
  explicitly *NOT in its verdict*. Capturing the true historical
  prediction + abstention makes the gate's realized effect *measurable*
  instead of reconstructed. **Zero training/gate/trade impact:**
  `train_scorer` reads ONLY `forward_return_5d`, so the new keys are
  inert — exactly the additive `forward_return_10d/20d` precedent (pass
  #18 multi-horizon). NOT a new diagnostic module (no treadmill); a
  data-fidelity fix to the existing outcomes pipeline. `None` on SELL
  rows (gate is BUY-only) and untrained/sub-gate cycles (no `scorer=`
  emitted). 9 exact-value locks + a 5d-byte-identity regression anchor
  in `tests/test_gate_decision_capture.py` (pure-helper matrix:
  in-dist / off-dist / `+0.0%` boundary / untrained / SELL /
  `score=`-vs-`scorer=` disambiguation / garbage-never-raises; +
  end-to-end through `_compute_decision_outcomes` proving the keys land
  and `forward_return_5d` is unperturbed). **Applies on next
  `run_continuous_backtests.py` restart** — the running loop predates it
  (the documented deploy-stale pattern; ledger lives at the gitignored
  `data/decision_outcomes.jsonl`).
  ```bash
  cd /home/zeph/trading-intelligence/paper-trader && python3 -m pytest tests/test_gate_decision_capture.py -v
  ```

- **Quant finding (decisive, reconfirmed fresh — the gate underwrites
  pure sizing variance).** Live deployed pickle `n_train=3860`, gate
  active. **Five independent OOS arbiters agree on the same temporal-OOS
  slice:** `baseline_compare` = **`MLP_NO_BETTER_THAN_TRIVIAL`** (MLP
  rank_ic **+0.069** vs raw `ml_score` one-liner **+0.111**, ic_gap
  −0.042 — the net carries *less* OOS rank skill than feature slot 0
  alone); `calibration --oos` = in-sample **`WELL_CALIBRATED`** (spearman
  0.458, decile err 2.30pp) collapsing to OOS **`MISCALIBRATED`**
  (spearman **0.069**, decile err **8.67pp** — the tool prints its own
  optimism-gap warning); `skill_trend` = **`NEGATIVE_OOS_SKILL`**
  (oos_rmse **11.30** ≥ fresh mean-predictor baseline **9.51**, median
  oos_ic 0.04, trend STABLE, `gate_active=1.0`); `gate_pnl` =
  **`GATE_RETURN_NEUTRAL`** (gate-on +0.76% vs gate-off +0.58%,
  equal-weight contribution **+0.18pp**, within ±1.0pp — "reallocates
  capital with no net realized effect: pure added sizing variance");
  `scorer_skill_log.jsonl` last cycles `oos_ic ∈ {0.01,0.19,0.02,0.02,
  −0.01,0.07}`, `oos_dir_acc ≈ 0.50`, `val_rmse ≈9 ≪ oos_rmse ≈10–17`
  (textbook overfit). The conviction gate (invariant #5, active every
  cycle) sizes real backtest positions on a 17-feature net with
  demonstrably zero/negative OOS skill. **Reported, not actioned** —
  turning the gate off / retraining is a training-dynamics change out of
  surgical scope (CLAUDE.md §6); the contribution of this pass is making
  the gate's *actual* historical decision durably measurable for the
  first time, not changing the model.

- **Quant findings (corroborating).** (1) **Running loop is stale code**
  — PID `1734916` predates the pass-#17 baseline-ledger commit
  (`6ade72d`), the pass-#18 multi-horizon capture, AND this pass's
  gate-decision capture: `data/baseline_skill_log.jsonl` does **not
  exist**, live `decision_outcomes.jsonl` rows carry only the 17 base
  keys (no `forward_return_10d/20d` / `gate_scorer_pred`). All three
  durability features are inert until the operator restarts
  `run_continuous_backtests.py` — the documented deploy-stale
  operational state, **not a code bug** (`baseline_trend` correctly
  returns `INSUFFICIENT_DATA`). (2) **Winner→ArticleNet feedback loop
  dead both ways** — 48 `inject err: database locked after 4 attempts` /
  `trainer timeout` / `trainer rc=` lines in the last 4000 log lines
  (digital-intern GPU + `articles.db` write-contention on the symlinked
  volume — matches passes #6–#17; the scorer itself retrains cleanly
  every cycle). (3) **Backtest dispersion is pure leverage-beta** — same
  recent batch: run 6230 +484.8%/vs_spy +396.7% beside run 6231
  −49.4%/vs_spy −12.4%; 476 complete / 24 failed, **no NaN**, max
  complete run_id 6234 (last completed 10:35 UTC — the loop is
  progressing). The "best run" cycle line must never be read as strategy
  skill. All reported observations, out of surgical scope.

- **Concurrency note.** This pass ran with **3 sibling agents executing
  the identical task prompt concurrently** (`ps`: PIDs 1752314 / 1824143
  / 1839361, same `claude --model claude-opus-4-7` HYBRID prompt) plus a
  swarm of dirty `../digital-intern/` working-tree files. **Never
  `git add -A`.** The two changed files (`run_continuous_backtests.py`
  with verified 0 sibling tokens, new exclusively-mine
  `tests/test_gate_decision_capture.py`) were path-scoped `git add`-ed,
  staged diff verified additions-only (`+224/-0`, 2 files, 0
  sibling-token grep hits), the index committed (NO pathspec — the
  pass-#16 `git commit -- <path>` re-snapshot lesson applied
  pre-emptively), and the deliverable confirmed on `origin/master` **by
  content** (`git cat-file -e origin/master:<test>` +
  `_parse_gate_decision` symbol grep), not by assuming it sits in this
  agent's commit message.

- **Run the ML/backtest suite:** `cd
  /home/zeph/trading-intelligence/paper-trader && python3 -m pytest
  tests/ -q -k "ml or backtest or scorer or calibration or continuous or
  horizon"`. `tests/test_gate_decision_capture.py` holds the new
  gate-decision-capture locks; it has none of "ml"/"backtest"/"scorer"
  in its node ids — add it explicitly like `test_calibration.py` /
  `test_horizon_audit.py` (it is picked up by the `continuous` keyword
  via `_compute_decision_outcomes`, but list it for clarity).

*Review pass #18 (ML+backtest hybrid) appended 2026-05-18. Prior content above is unmodified.*

---

## Feature: macro calendar — forward FOMC rate-decision awareness (feature-dev agent, 2026-05-18)

**Gap.** Across 47 analytics modules the live Opus decision prompt had
exactly **one** forward-looking block: `event_calendar` (single-name
**earnings**). **Zero** macro-event awareness existed anywhere — yet this
watchlist is leveraged-ETF heavy (SOXL/TQQQ/NVDL/SOXS), exactly the
instruments that gap hardest on a Fed surprise, and the system's own 5h
Opus briefings repeatedly *lead* with macro (bond rout, 10Y, FOMC). A
leveraged book entering the rate-decision instant blind is the macro
analog of the exact "added the day before an earnings print, blind"
mistake `event_calendar` was built to close — the same gap, one dimension
over.

**What shipped.** New `paper_trader/analytics/macro_calendar.py` —
`build_macro_calendar(now=None, horizon_days=14.0)`:

- **FOMC-only, by verifiability discipline.** The 2026 FOMC schedule is
  fully verified from federalreserve.gov (all 8 meetings, fetched
  2026-05-18). BLS CPI/Employment-Situation forward dates are *not*
  reliably verifiable (bls.gov hard-blocks every fetch HTTP 403;
  archive-URL dates conflict with summaries by ±2d; Jul–Dec unreleased).
  Encoding an unverified date on the live decision path would mislead Opus
  — declined. CPI/NFP are a documented future extension *pending a
  verifiable source*, NOT an oversight (a parallel table behind the same
  honesty bound).
- **Pure static table + date math** — NO file I/O, NO network, no import
  beyond stdlib (even safer than `event_calendar`'s disk read; the
  `risk_mirror` hot-path discipline). `now` injectable, deterministic,
  `_safe` (never raises — a top-level guard returns an honest dict).
- **Honesty bound (landmine guard).** `SCHEDULE_VALID_THROUGH` is exactly
  the last encoded instant; `now` past it degrades to one honest
  "schedule not loaded" line — never a fabricated event. Locked by a
  table↔bound **no-drift** test (extending one without the other fails
  RED) — the test written first.
- **Time-precision tiers** (the material differentiator vs
  `event_calendar`'s date-only granularity): `IMMINENT_HOURS` (<24h,
  rendered "in Xh") > `IMMINENT` (≤3d) > `UPCOMING` (≤horizon). Each FOMC
  statement instant is the 2nd-day 14:00 ET policy release, ET→UTC
  resolved across the 2026 DST boundary (Jan/Dec 19:00Z EST, Mar–Oct
  18:00Z EDT) so there is no tz-library dependency on the hot path.
- **Market-wide, not per-ticker** (the structural difference from
  `event_calendar`): no positions/names arg — an FOMC decision is
  relevant to a flat book too; always rendered.
- **Observational, never prescriptive** (invariants #2/#12 — the
  `event_calendar` contract): autonomy-preserving preamble, no directive
  verb, no cap, never gates Opus.

**Wiring (all one commit — the standing checklist item).**
`strategy._build_payload` gained a `macro_calendar_block` kwarg rendered
**between `event_calendar` and `buying_power`** (forward blocks stay
adjacent: earnings then macro) — the prompt order is now
`risk < sector < event < macro < bp < WATCHLIST PRICES`. `strategy.decide()`
builds it `_safe` after `event_calendar_block`. Per the AGENTS.md pass-#17
standing checklist item (3rd documented instance of "a new `_build_payload`
advisory block forgotten in `decision_context.py`"), **the same commit**
threads it through `decision_context.build_decision_context` +
`assemble_inputs` + the `advisory_blocks` dict + the `__main__` CLI line.
New endpoint **`GET /api/macro-calendar`** serves the same builder
(prompt↔endpoint parity — the `event_calendar`/`risk_mirror` discipline);
pure static-table, no store read.

**Tests.** New `tests/test_macro_calendar.py` (18, behaviour-asserting):
honesty-bound degrade (written first) + table↔bound no-drift; the 8
encoded instants == the federalreserve.gov-verified 2026 set; ET→UTC DST
correctness; IMMINENT_HOURS/IMMINENT/UPCOMING + exact 24h/3d/horizon
boundaries; past-event grace; soonest-first sort; observational/no-directive;
never-raises; `_build_payload` render position + None-renders-nothing;
`/api/macro-calendar` Flask parity. Plus `test_decision_context.py`
(updated exact `advisory_blocks` dict + new
`test_macro_calendar_block_reaches_prompt_verbatim_and_flagged` —
the full `risk<sector<event<macro<bp<WATCHLIST` ordering lock) and
`test_decision_context_endpoint.py` (assemble_inputs-builds-it discriminating
lock). Bounded sweep **330 green** across `test_macro_calendar`,
`test_decision_context{,_endpoint}`, `test_event_calendar`,
`test_core_strategy`, `test_buying_power`, `test_sector_exposure`,
`test_core_dashboard_helpers`, `test_core_runner`, `test_core_reporter`,
`test_core_state_swr`, `test_dashboard_threaded` (the full suite times out
>400s under concurrent sibling-agent load — the documented host state).

Like every recent feature this **applies on the next paper-trader restart**
— the live `:8090` is `stale` / `behind:2`, so it keeps the old prompt
(no macro block) until the runner reboots onto the new SHA.

*Macro-calendar feature appended 2026-05-18. Prior content above is unmodified.*

---

## Review pass #19 — paper-trader CORE hybrid (2026-05-18)

**bugs_fixed = 1 · features_added = 2 · user_findings = 4.** Bounded
representative sweep (the full suite times out under concurrent
sibling-agent + auto-commit-daemon load — the documented host state):
**503 green** across `test_core_{reporter,store,strategy,runner,runner_cycle,market,signals,dashboard_helpers}`,
`test_market_negcache`, `test_capital_paralysis{,_swr}`, `test_benchmark`,
`test_runner_heartbeat{,_swr}`, `test_core_state_swr`,
`test_dashboard_threaded`, `test_ml_live_opinion`, `test_parse_retry`,
`test_quota_guard`. Three+ sibling agents and the auto-push daemon were
active; commits were path-scoped (`git add <file>`, never `-A`),
staged-diff verified, and pushed fast-forward onto `origin/master`.

- **Phase 1 — bug (`b92b9c2`). `Store.get_portfolio()` only absorbed the
  `row is None` shared-connection corruption mode; the *equally documented*
  "row whose columns read back NULL" mode fell straight through.** The
  function's own docstring promised to absorb **both** modes (28x
  `/api/state` 500s over 2 days), but the code self-healed only `row is
  None`. A transient NULL `cash`/`total_value` was returned verbatim and
  then hit `strategy._portfolio_snapshot` as `None + open_value` — a
  `TypeError` that **aborts the entire `decide()` cycle** (no decision
  row, no equity point) until the next clean read, on top of 500ing
  `/api/state`. Fix: the self-heal guard now also fires on
  `row["cash"] is None or row["total_value"] is None`, routing through the
  same `_init_portfolio()`+re-read path — which recovers the real
  well-formed values on a transient corrupt read and never resets a live
  book (`_init_portfolio` only INSERTs a *missing* row). Existing
  resilience tests unchanged (no regression); 2 new
  `tests/test_core_store.py::TestGetPortfolioResilience` tests cover
  transient-recovery and persistent-degrade-arithmetic-safe. Distinct
  from the sibling's `05b406e` (cursor-collision `fetchone()==None`
  retry) — a different corruption mode on the same shared connection.

- **Phase 2 — feature 1 (`9bfbbf8`). SWR-cache `/api/risk`,
  `/api/benchmark`, `/api/capital-paralysis`, `/api/decision-health`.**
  Live probing (fresh runner = cold caches, host load avg ~16–23) found
  these four were the only remaining high-traffic **pure-store-read**
  core panels not behind `swr_cached`: each ran its heavy multi-read
  handler (`recent_trades(2000)` / `equity_curve(5000)` /
  `recent_decisions(3000)`) inline with no bounded cold path and **hung
  >15s (curl → 000)** under CPU starvation, while every already-cached
  endpoint served a fast `{"warming":true}` placeholder. Four core trader
  panels ("am I beating the index?", "why is my book stuck?") were
  effectively dead under load. All four verified pure store reads (no
  market/yfinance/network) — the latency is pure lock-contention, exactly
  what SWR absorbs (the pass-#18 runner-heartbeat precedent, invariant
  #7). `@swr_cached(.., 30.0)`: 30s ≪ the ≥1800s decision cadence so a
  ≤30s stale window can never flip a verdict. SWR is pytest-inert unless
  `_SWR_TEST_FORCE`, so the existing exact-value
  `test_capital_paralysis.py` / `test_benchmark.py` stay green. Locked by
  `tests/test_capital_paralysis_swr.py` (cold full-payload+honesty-keys,
  warm-served-stale, pytest-inert-default — mirrors
  `test_core_state_swr.py`). **Applies on the next paper-trader restart.**

- **Phase 2 — feature 2 (`2e8c60e`). `reporter._fmt_trade_stamp` — date +
  relative age on stale hourly recent-trades.** The hourly summary
  rendered every recent trade as a bare `HH:MM` (UTC) with no date; the
  desk's #1 documented pathology is a book frozen for many hours that
  still *looks* active — a 25h-old `BUY MU` shown as `[09:38]` reads as
  today's fill (the "unclear" the trader persona is frustrated by).
  Today's trades stay byte-identical `HH:MM`; older ones render
  `MM-DD HH:MM · Nd ago` so staleness is unmissable. Pure,
  now-injectable; future-skew guarded; a corrupt timestamp degrades to a
  clean `??:??` sentinel (the reporter additive-line contract — never
  raises). Locked by `tests/test_core_reporter.py::TestAgo` +
  `TestFmtTradeStamp` (bucket boundaries, today/stale/future/naive/corrupt,
  hourly-integration on a frozen clock).

- **Phase 3 finding-fix (`dbcc6e6`, folded into a `fix:` — NOT counted as
  a bug).** Resolved AGENTS.md review-pass-#18 core finding #3:
  permanently-delisted **GOOGU / METAU** removed from `strategy.WATCHLIST`
  *and* the `SYSTEM_PROMPT` "LEVERAGE INSTRUMENTS AVAILABLE" text (verified
  live: yfinance 404 / no quote, vs NVDU/MSFU/AMZU still trading). They
  could never fill (`market.get_price` → None → `_execute` BLOCKS) and
  re-404'd `runner.log` every `_DEAD_TTL` window. The code change merged
  via the repo's auto-commit daemon; `dbcc6e6` adds the missing regression
  lock (`test_core_strategy.py::TestWatchlistHygiene`) so neither the
  universe nor the mirrored prompt text can silently re-introduce a
  delisted name (the recurring "fix one place, not the other" concern).
  **Live side only** — `backtest.py`'s historical universe is the
  ML-domain owner's call, deliberately untouched.

- **Phase 3 — live findings (running `:8090`, build-info cycled
  `b92b9c2`→`548437a` during the session via relaunch churn).**
  1. **Dashboard `:8090` transiently unreachable during singleton-guard
     relaunch churn (MEDIUM, ops, pass-#18 finding #4 recurring).** A
     supervisor keeps relaunching `runner.py` in restart gaps; the
     invariant-#19 guard correctly refuses duplicates (no double-trade)
     but `:8090` is dead for the ~10–30s relaunch window (observed:
     curl exit 7 at ~04:13, 200 again seconds later on the fresh PID).
     Out of code scope.
  2. **A rogue/generic concurrent agent is committing to the shared repo
     (MEDIUM, ops).** PID 1845436 runs an outdated prompt referencing
     non-existent `paper_trader/scorer.py` / `trader.py` and targeting the
     `/home/zeph/paper-trader` symlink, with "Commit and push" — opaque
     pushes onto the same `master`. Combined with the auto-commit daemon
     that swept this pass's `strategy.py` change into a generic
     `feature(pt)` commit, scoped-commit attribution is unreliable; verify
     by **content** on `origin/master`, not by commit message.
  3. **Book still capital-paralysed (HIGH, host/ops, not code-fixable).**
     `/api/state`: cash $18.49 / total $972.69, ~98% deployed in MU/LITE,
     MU P/L $0.00 (stale-mark flat), no new fill since
     `2026-05-17T09:38 BUY MU`. The documented #1 pathology persists; this
     pass's feature 2 is precisely what now makes that staleness visible
     in the Discord hourly instead of a dateless `[09:38]`.
  4. **`/api/runner-heartbeat` returns all-None `{"warming":true}
     cached:false` on a cold call under load (LOW, expected SWR
     behaviour).** The check-first panel is uninformative for the first
     ~budget seconds after a relaunch — exactly when a trader most wants
     it. Inherent to the cold-start gap (no stale copy yet); the SWR-4
     feature does not change this (heartbeat was already cached). Noted,
     not fixed — a cold-start prewarm is a larger design question.
  - **Positive validation.** `/api/state` 0.004s, `/api/feed-health` /
    `/api/source-edge` 200 in ~2s (SWR cold budget honoured); after the
    SWR-4 fix the four previously-hung panels inherit the same bounded
    cold path. Phase-1 store fix + 503-green bounded sweep confirm the
    decision loop's portfolio read is no longer a `None`-propagation
    abort risk.

- **Run the core suite (bounded — the full one times out under concurrent
  load):** `python3 -m
  pytest tests/test_core_store.py tests/test_core_reporter.py
  tests/test_core_strategy.py tests/test_capital_paralysis_swr.py
  tests/test_core_state_swr.py tests/test_runner_heartbeat_swr.py -q`.
  `TestGetPortfolioResilience` (store), `TestWatchlistHygiene` (strategy),
  `TestAgo`/`TestFmtTradeStamp` (reporter) and
  `TestCapitalParalysisSwr` are the new pass-#19 locks.

*Review pass #19 (paper-trader core hybrid) appended 2026-05-18. Prior content above is unmodified.*

---

## Review pass #20 — paper-trader core hybrid (2026-05-18)

Repo HEAD cycled `b387414`→`542446d`→`d888261` during the session (this
pass's two commits, plus auto-commit-daemon / concurrent-agent churn — at
least three other `claude --model claude-opus-4-7` review agents were live
on the same tree; verify attribution by **content** on `origin/master`, not
by message — the recurring pass-#18/#19 concern).

- **Phase 1 — bug fixed (1): `_swr_prewarm` omitted 6 of 22 `@swr_cached`
  endpoints (`542446d`).** `dashboard._swr_prewarm`'s docstring promises it
  pre-builds *"every slow SWR cache once at boot"*, but its `targets` list
  carried only 16 of the 22 `@swr_cached` endpoints. The 6 omitted —
  `risk`, `benchmark`, `capital-paralysis`, `decision-health`,
  `runner-heartbeat`, `scorer-confidence` (the first four SWR-wrapped in the
  immediately-prior commit `9bfbbf8`) — alone paid the full cold path after
  **every** restart: a bounded stall then `{"warming": true}`, real data
  only on the frontend's next auto-refresh poll. Five of them are precisely
  the panels a trader opens FIRST to triage "why is the bot frozen?", so the
  freeze-triage surface was the last to populate — **this directly resolves
  pass-#19 Phase-3 finding #4**, which observed exactly this on
  `/api/runner-heartbeat` and explicitly deferred it ("Noted, not fixed — a
  cold-start prewarm is a larger design question"). Fix adds the 6 tuples +
  `tests/test_swr_prewarm_coverage.py`, a regression-lock asserting the
  prewarm list stays `== {@swr_cached names}` so a future SWR-wrapped
  endpoint can never silently re-rot the contract. Confirmed empirically
  before/after: `/api/risk` & `/api/runner-heartbeat` resolve on the 2nd
  poll; all 22 are now prewarmed.

- **Phase 2 — feature added (1): `/api/equity-integrity` — time-series P&L
  trust audit (`d888261`).** `mark_integrity` answers "is my book stale
  RIGHT NOW" (point-in-time). Nothing audited the recorded `equity_curve`
  *over time*, yet `/api/drawdown`, `/api/benchmark`, `/api/analytics`
  Sharpe and the hourly Discord P/L line are all derived from it — a silent
  corruption there poisons every P&L surface with nothing saying so. New
  pure builder `paper_trader/analytics/equity_integrity.py` +
  `equity_integrity_api` (EOF, lowest-collision insertion point) flags:
  **NEGATIVE_CASH** (the no-hard-cap book physically over-drawing —
  AGENTS.md #12), **NONPOSITIVE_EQUITY**, and **SUSPECT** no-trade jumps
  (|Δtotal|≥8% with no trade in the window — the mismark / stale-price
  unfreeze / option-settlement signature; a jump *with* a trade in-window is
  explained away). Pure store reads only (no network → no SWR wrap needed);
  advisory/read-only, gates nothing, adds no caps (the `mark_integrity`
  contract). 10 tests assert exact verdicts/values incl. the half-open
  `(lo, hi]` window boundary and the never-raises-on-garbage degrade.
  Verified live: **CLEAN across 787 points, min cash $2.61** — confirms the
  no-cap book has never over-drawn and the P&L history is trustworthy.

- **Phase 3 — live findings (running `:8090`, runner orphan PID 1866235
  on current code).**
  1. **Runner is an UNSUPERVISED ORPHAN (HIGH, ops, not code-fixable).**
     `journalctl --user -u paper-trader`: the systemd unit *failed* at
     01:46 (`status=1/FAILURE`, the invariant-#19 guard correctly refused
     to start while an older manual launch held the lock) and stayed
     **inactive/disabled**. The live trader is a bare `python3 runner.py`
     with PPID 1. `/api/supervision` correctly returns
     `verdict=UNSUPERVISED orphan=True`. The git-watcher's whole
     deferred-restart/deadman design relies on `systemd Restart=always`;
     with the unit down, the **next** git-watcher restart or deadman
     `os._exit(0)` (and commits keep landing, so it *will* fire) leaves the
     trader permanently DOWN. Operator action: `systemctl --user enable
     --now paper-trader` once the orphan is stopped.
  2. **Live trader frozen — IDLE_STORM (HIGH, host-saturation, not a
     code/prompt bug).** 60% lifetime `NO_DECISION` (473/785); the current
     ongoing `PARALYSIS` drought is 24.9h / 73 cycles / 53 `NO_DECISION`
     (`/api/decision-drought`). Last real fill `2026-05-17T09:38 BUY MU`
     (>26h ago). `/api/runner-heartbeat`: liveness `HEALTHY` but
     `decision_efficacy=IDLE_STORM`, `restart_recommended=true` (the
     instrumentation is correct). Root cause is the documented host
     starvation: load avg 13–21, ~356 MB free RAM, with 4+ concurrent
     `claude --model claude-opus-4-7` review agents **plus**
     `run_continuous_backtests.py` — every live Opus decision call times
     out (`claude returned no response (timeout/empty)`). The review
     harness induces the very freeze it observes.
  3. **Discord delivery seen failing `env: 'node': No such file`
     (MEDIUM).** `logs/runner.log` (stale, from the 01:46 failed systemd
     attempts) shows the documented openclaw shebang/PATH outage —
     operator's primary surface was dark for those attempts. `reporter._send`
     already has the `bin_dir`-onto-PATH fix; the current orphan is on
     patched code, but Discord health for the orphan could not be confirmed
     (it logs to a detached stdout; the unit journal is empty). Watch
     `notify_health.restart_recommended`.
  4. **Capital paralysis persists (HIGH, host/ops, correctly surfaced).**
     `/api/state` cash $18.49 / total $972.69 (~98% deployed); LITE held
     3.7d at **7.0× the empirical median losing hold**, −$6.21 disposition
     drag (`/api/hold-discipline`). Documented #1 pathology; instrumentation
     (`hold-discipline`, `capital-paralysis`, the Discord `_capital_pulse_
     line`) all fire correctly — nothing to fix in code.
  5. **`/api/position-thesis` not SWR-wrapped, intermittently slow (LOW).**
     One probe 200 in 1.56s, an earlier one >12s under the load spike. Not
     consistently broken; a candidate for the same `swr_cached` treatment
     the other slow endpoints got. Noted, not fixed (collision risk with
     concurrent agents; marginal under current load).

- **Run the touched/adjacent locks (bounded — the full suite is ~25 min and
  load-flakes `test_dashboard_threaded::test_threaded_server_parallelizes`,
  a pure timing assertion, under load avg >10):** `python3 -m pytest
  tests/test_swr_prewarm_coverage.py tests/test_equity_integrity.py
  tests/test_dashboard_swr.py tests/test_mark_integrity.py
  tests/test_core_dashboard_helpers.py -q`. The pass-#20 locks are
  `test_swr_prewarm_coverage.py` (prewarm == @swr_cached set, the
  regression-lock for the fix) and `test_equity_integrity.py` (exact-value
  verdicts incl. the half-open window boundary).

*Review pass #20 (paper-trader core hybrid) appended 2026-05-18. Prior content above is unmodified.*

---

### 2026-05-18 review pass #19 (ML+backtest hybrid · anti-overfit scorer config · live findings)

- **Phase 1 — no new bug (bugs_fixed = 0; no Phase-1 commit, the commit
  guard explicitly permits it).** Full re-trace of `decision_scorer.py`,
  `backtest.py`, `run_continuous_backtests.py` plus the newest
  least-reviewed `baseline_trend.py`: `_inject_and_train`'s 11-col INSERT
  matches its 11-tuple and the `for…else` lock-retry returns the honest
  error after exhausting `_LOCK_RETRY_SLEEPS`; `_ml_decide`'s
  `scorer_off_dist` gate-skip + `getattr(_scorer,"_n_train",0)` dummy
  fallback are sound; `_fwd_ret_h` gates on `price_on() is None` exactly
  like the byte-identical 5d path; `PriceCache._build_trading_days`
  empty-SPY fallback is paired with the honest `benchmark_unavailable`
  note. Consistent with the documented 17 prior no-new-bug ML/backtest
  passes (#5–#18) — the core trio is exhaustively exact-value locked. No
  redundant test-hardening commit (the pass-#17/#18 churn-avoidance
  precedent). ML/backtest regression **368 green before** the feature.

- **Feature shipped (Phase 2, `feat(ml):`): the DecisionScorer MLP is now
  regularized — `(32,16)` + L2 `alpha=1e-2` + `early_stopping` (was an
  unregularized `(64,32,16)`/600-iter net).** `paper_trader/ml/
  decision_scorer.py::train_scorer`, the exact location CLAUDE.md §12
  points to ("Train the DecisionScorer differently") and explicitly **in
  scope** (it is not the gate-threshold change CLAUDE.md §6 marks
  out-of-scope). **The gap it closes:** the per-cycle
  `scorer_skill_log.jsonl` records the overfit every cycle
  (`val_rmse ≈ 9–11 ≪ oos_rmse ≈ 12–17`) and this pass's fresh read
  reconfirmed it (cyc4 `val 10.74 / oos 16.68`, cyc5 `9.01 / 14.04`). A
  faithful A/B replaying `train_scorer`'s exact preprocessing on the
  **live** `decision_outcomes.jsonl` 5000-row tail under
  `validation.split_outcomes_temporal(0.2)` (the honest holdout
  `_train_decision_scorer` itself uses), across **4 MLP seeds**, showed
  the new config uniformly lowers temporal-OOS RMSE (**mean ≈14.97→≈12.58,
  up to 16.68→10.46** on the worst prior seed), closes the val/oos gap
  from ~6pp to **<1pp**, and leaves OOS rank-IC / dir-acc within ±0.04 /
  coin-flip noise. **Honest scope:** this removes the *magnitude* overfit;
  it does **not** create rank skill — the MLP stays
  `MLP_NO_BETTER_THAN_TRIVIAL` (a deeper data-signal limitation, unchanged
  by hyperparameters, still tracked by the baseline ledger / pass #18).
  The `_ml_decide` conviction gate acts on the prediction's MAGNITUDE
  bucket (±10/±5/0), so a uniformly lower-error, less-extrapolating head
  makes those bucket assignments materially less noisy. **Zero schema
  impact:** gate arms, the `±PRED_CLAMP_PCT` clamp, `build_features`,
  `SECTORS`, `N_FEATURES`, the `{model,scaler,n_train}` pickle, and the
  numpy-lstsq sklearn-absent fallback are all untouched — a drop-in the
  next retrain cycle picks up. Realigns the code with CLAUDE.md §3's
  long-documented "MLPRegressor 32→16" architecture (the code had
  silently drifted to `(64,32,16)`). 2 behaviour-asserting locks in
  `tests/test_decision_scorer.py::TestAntiOverfitConfig`: a config-lock
  (pickled `hidden_layer_sizes==(32,16)`, `alpha==1e-2`,
  `early_stopping`, `validation_fraction==0.15`, `n_iter_no_change==25`)
  and a **discriminating** noise-memorization test — on a pure-noise
  target the regularized net's `pred_std/target_std` ratio is **≈0.40**
  vs the old config's measured **≈1.00** (memorizes noise almost
  perfectly), asserted `< 0.65` with wide both-sided margin so it is
  non-flaky AND fails RED on a revert to the memorizing net. **370 green
  after** (368 + 2). **Applies on the next `run_continuous_backtests.py`
  restart** — the running loop (PID predates this) keeps the old config
  until the operator reboots it (the documented deploy-stale pattern;
  pickle at the gitignored `data/ml/decision_scorer.pkl`).
  ```bash
  cd /home/zeph/trading-intelligence/paper-trader && python3 -m pytest tests/test_decision_scorer.py::TestAntiOverfitConfig -v
  ```

- **Quant findings (Phase 3, live).** (1) **Overfit reconfirmed fresh and
  now actioned** — `scorer_skill_log.jsonl` last cycles all show
  `val_rmse ≪ oos_rmse`, `oos_ic ≈ 0`, `oos_dir_acc ≈ 0.49–0.56`; the
  shipped config is the first pass to *act* on it rather than only
  re-measure it. (2) **Zero OOS rank skill persists** —
  `oos_ic ∈ {0.19,0.02,0.02,−0.01,0.07,0.04}`; the MLP carries no durable
  rank edge over the raw `ml_score` one-liner (data-signal limitation, out
  of surgical scope, unchanged by this pass). (3) **Running loop is stale
  code** — the live `decision_outcomes.jsonl` tail rows carry only the 17
  base keys (no `forward_return_10d/20d`, no `gate_scorer_pred`), and this
  config change is likewise inert until restart; documented deploy-stale
  operational state, not a code bug. (4) **Backtest dispersion is pure
  leverage-beta** — run 6230 +484.8%/vs_spy +396.7% beside run 6231
  −49.4%/vs_spy −12.4% same recent batch; **477 complete / 24 failed, no
  NaN**, max completed run_id 6235 at 11:46 UTC (loop healthy and
  progressing). The "best run +N%" cycle line must never be read as
  strategy skill. The only `continuous.log` "errors" are external GDELT
  rate-limit/connection-reset noise (handled by the retry+backoff+
  permanent-cache path) and the documented winner→ArticleNet
  `database locked`/`trainer timeout` contention — no core-code traceback.
  Findings 2–4 reported, out of surgical scope.

- **Concurrency note.** Ran with **3+ sibling agents on the identical
  task** in the shared monorepo working tree (core-hybrid siblings
  committed their own `pass #19`/`#20` to `AGENTS.md` mid-pass) plus a
  swarm of dirty `../digital-intern/` + sibling-untracked
  `paper_trader/ml/gate_pnl.py` / `analytics/game_plan.py` files. **Never
  `git add -A`.** Exactly three path-scoped files staged
  (`paper_trader/ml/decision_scorer.py`, `tests/test_decision_scorer.py`,
  `AGENTS.md`), `git diff --staged` verified additions-only with zero
  sibling-token hits, committed via the index (no pathspec — the pass-#16
  re-snapshot lesson), deliverable confirmed on `origin/master` by content.

- **Run the ML/backtest suite:** `cd
  /home/zeph/trading-intelligence/paper-trader && python3 -m pytest
  tests/ -q -k "ml or backtest or scorer or calibration or continuous or
  horizon"` (370 green; `TestAntiOverfitConfig` is picked up by the
  `scorer` keyword).

*Review pass #19 (ML+backtest hybrid) appended 2026-05-18. Prior content above is unmodified.*

---

## Review pass #21 — paper-trader core hybrid (2026-05-18)

Repo HEAD at boot `5a0af2d`; this pass's single commit is `b0ac368`. Ran
alongside ≥1 live sibling `claude --model claude-opus-4-7` ML/backtest
agent (PID 1752314 on the identical task, ML files) — verify attribution
by **content** on `origin/master`, not message. **bugs_fixed = 0 ·
features_added = 1 · user_findings = 5.**

- **Phase 1 — no new bug (bugs_fixed = 0; no Phase-1 commit, the commit
  guard explicitly permits it).** Full re-trace of `runner.py`,
  `reporter.py`, `signals.py`, `strategy.py`, `market.py`, `store.py` plus
  the `dashboard.py` SWR/helper surface (`_swr_prewarm`,
  `_parse_action_ticker`, `_classify_disagreement`). Every candidate was
  considered and dismissed with a concrete reason: `is_market_open`
  boundaries (`570 ≤ m < close_minute`) are exact incl. the half-day
  exclusive 13:00 close; `signals._choose` LOCAL-first strict-`>`
  tie-break matches its docstring on every None/equal permutation;
  `strategy._execute` runs once per live cycle so the snapshot-cash read
  has no intra-cycle staleness; `_mark_to_market`'s `is not None`
  expired-option settlement preserves a legitimate 0.0; `store`'s
  closed-option-row reactivation + shared-connection NULL-row recovery are
  sound; `reporter._classify_decision_outcome` check order is
  arrow-safe. Consistent with the 20 prior core passes — the listed core
  files are exhaustively reviewed and exact-value locked. **338 core tests
  green in 4s before** the feature (`test_core_runner /_market /_store
  /_signals /_strategy /_reporter`); no churn-only test commit (the
  pass-#17/#19 churn-avoidance precedent).

- **Phase 2 — feature shipped (`feat(runner):`, commit `b0ac368`): the
  daily-close report now fires after the *actual* NYSE bell, not a
  hardcoded 16:05 ET.** `runner._maybe_daily_close` gated on
  `DAILY_CLOSE_HOUR_NY = 16` regardless of the session. On a NYSE
  early-close half-day (day-after-Thanksgiving `2026-11-27`, Christmas Eve
  `2026-12-24`) the bell is 13:00 ET, so the close report sat **three
  hours on a frozen post-close book** waiting for a 16:05 that no longer
  matched the session — twice a year the trader's end-of-day summary was
  emitted late against stale 13:00 marks with no signal that the session
  had already ended. Fix replaces the hardcoded hour gate with
  `market.close_minute(now_ny.date()) + DAILY_CLOSE_GRACE_MIN` (the
  existing half-day infra `is_market_open` already uses, single source of
  truth): **byte-identical 16:05 ET on every regular day** (`close_minute
  → 960`, gate `< 965`), **13:05 ET on a half-day** (`close_minute → 780`,
  gate `< 785`). Weekend + full-holiday guards untouched (half-days are
  weekdays not in `NYSE_HOLIDAYS_2026`, so only the new gate decides).
  Surgical: 2 hunks in `runner.py` (constant + gate), zero behaviour
  change off half-days. 8 new exact-minute locks in
  `tests/test_core_runner.py::TestMaybeDailyCloseHalfDay`: half-day
  no-fire at 12:59 / 13:04 → fire at 13:05 (both half-day dates),
  fires-once, **plus a regression lock that a regular weekday does NOT
  fire at 13:05** (the early-close shift must never leak into a normal
  day) and the 16:04→no / 16:05→yes backward-compat pin. **382 green
  after** (`test_core_runner /_runner_cycle /_market /_store /_signals
  /_strategy /_reporter /_runner_heartbeat`). **Applies on the next
  runner restart** — the running orphan (boot `5a0af2d`) keeps the old
  16:05 gate until rebooted (the documented deploy-stale pattern; see
  finding #1).

- **Phase 3 — live findings (running `:8090`, runner orphan PID 1868409,
  2026-05-18 ~12:07 UTC; the host is under the review-swarm load this
  pass's own siblings contribute to).**
  1. **Runner is UNSUPERVISED_STALE (HIGH, ops, not code-fixable —
     continuity of pass #20 #1).** `/api/supervision`:
     `verdict=UNSUPERVISED_STALE`, `orphan=true`, `ppid=1`, systemd bus
     unreadable (`Failed to connect to bus: No medium found`).
     `boot_sha=5a0af2d` vs `head_sha=b0ac368`, `behind:1` — the running
     trader is one commit stale and **that commit is this pass's own
     half-day fix**, which therefore will not deploy until the operator
     restarts. With no `Restart=always` net, the next git-watcher /
     deadman `os._exit(0)` leaves the trader DOWN. Operator action:
     `systemctl --user enable --now paper-trader`.
  2. **Live trader frozen — IDLE_STORM (HIGH, host-saturation, not a
     code/prompt bug — continuity of pass #20 #2 + recalled
     `pt-no-decision-host-saturation`).** `/api/runner-heartbeat`:
     liveness `HEALTHY` (last decision 8m ago) but
     `decision_efficacy=IDLE_STORM`, `consecutive_no_decision=20` (100% of
     last 20), `restart_recommended=true`. The last 6 `decisions` rows are
     **all** `NO_DECISION | claude returned no response (timeout/empty)`
     (a *timeout*, NOT quota — `_quota_exhausted` would tag it
     differently). Host **load avg 19.5**, ~1.7 GB free RAM, with the
     concurrent `claude` review agents + `run_continuous_backtests.py`
     starving every live Opus call past `DECISION_TIMEOUT_S` + Sonnet
     fallback. `/api/decision-drought`: ongoing PARALYSIS 25.3h / 76
     cycles / 56 NO_DECISION (73.7%). The review harness induces the very
     freeze it observes; instrumentation is correct.
  3. **Capital-paralysis BLEEDING (HIGH, downstream of #2, correctly
     surfaced — continuity of pass #20 #4).** `/api/capital-paralysis`:
     98.1% deployed, $18.49 cash, LITE 61% of book,
     `paralysis.verdict=BLEEDING`, `involuntary_alpha_bleed_pct=-2.21%`
     across 6 parse-failure droughts; last fill `2026-05-17T09:38 BUY MU`
     (>26h ago). `can_act_on_signal=true` (state FREE — `min_actionable
     $9.73 < $18.49`) yet BLEEDING because the NO_DECISION storm, not lack
     of dry powder, is the bind. Nothing to fix in code.
  4. **Data trust is intact — positive finding.** `/api/feed-health`
     HEALTHY (newest live article 0.1h old, 863 live/2h, 4939/24h, no
     split-brain) and `/api/equity-integrity` **CLEAN across 789 points,
     min cash $2.61, 0 negative-cash / 0 unexplained jumps**. This
     *isolates* the freeze: it is a host-timeout on the Opus call, NOT a
     blind feed or a corrupt book — the no-cap book has still never
     over-drawn and the P&L history feeding every Discord/`drawdown`/
     `benchmark` surface is trustworthy. The pass-#20 `equity-integrity`
     feature is paying off as a standing trust audit.
  5. **Discord delivery HEALTHY (continuity, fix holding).**
     `runner-heartbeat.notify`: `verdict=HEALTHY`, 0 consecutive failures,
     last OK 2026-05-18T11:55:39. The openclaw/PATH `bin_dir`-onto-PATH
     fix holds on the orphan (it is on patched `reporter` code even though
     `runner` is stale) — the operator's primary surface is live.

  None of 1–5 is a new quick safe code fix: 1–3 + 5 are
  ops/host/continuity, 4 is a positive confirmation. No Phase-3 fix
  folded in.

- **Concurrency / staging discipline.** Never `git add -A`. Exactly two
  path-scoped files staged for the feature commit
  (`paper_trader/runner.py`, `tests/test_core_runner.py`); AGENTS.md
  committed separately alongside this entry. `git diff --staged` verified
  additions-only with zero sibling-token hits before the commit; the dirty
  `../digital-intern/` tree, sibling-untracked `paper_trader/ml/
  gate_pnl.py` / `analytics/game_plan.py` / `preflight.py`, and the live
  ML sibling's files were never touched (the pass-#16/#19 re-snapshot
  lesson). Deliverable confirmed on `origin/master` as `b0ac368` by
  content.

- **Run the touched/adjacent locks (bounded — the full suite is ~25 min
  and load-flakes timing-assertion tests above load avg ~10):** `cd
  /home/zeph/trading-intelligence/paper-trader && python3 -m pytest
  tests/test_core_runner.py tests/test_market_half_day.py
  tests/test_core_market.py -q` (104 green). The pass-#21 lock is
  `tests/test_core_runner.py::TestMaybeDailyCloseHalfDay` (exact-minute
  half-day fire/no-fire boundaries + the regular-day non-leak
  regression).

*Review pass #21 (paper-trader core hybrid) appended 2026-05-18. Prior content above is unmodified.*

---

## Review pass #22 — paper-trader core hybrid (2026-05-18)

Repo HEAD at boot `188e819`; this pass's single feature commit is
`a1cc09c` (a sibling ML commit `377c6f7 feat(ml): gate_realized` landed
between staging and push — heavy concurrent activity; verify attribution by
**content** on `origin/master`, not message). **bugs_fixed = 0 ·
features_added = 1 · user_findings = 5.**

- **Phase 1 — no new bug (bugs_fixed = 0; no Phase-1 commit, the commit
  guard explicitly permits it).** Full re-trace of `runner.py`,
  `reporter.py`, `signals.py`, `strategy.py`, `market.py`, `store.py` plus
  the freshly-changed `analytics/funded_suggestions.py` PARTIAL ladder
  (commit `188e819`, the one file edited just before this pass — the only
  realistic place a fresh bug lives). Every candidate dismissed with a
  concrete reason: the `funded_suggestions` empty-ladder PARTIAL arm
  (`enough=False, by=[]` → "nothing to unlock") is intentional design
  (cash funds *some* ⇒ still PARTIAL, per the commit rationale);
  `is_market_open`'s half-day exclusive `< close_minute` boundary,
  `_choose` LOCAL-first strict-`>` tie-break, `_mark_to_market`'s
  `is not None` expired-option settlement, the closed-option-row
  reactivation, and `reporter._classify_decision_outcome`'s arrow-safe
  check order are all exact and consistent with the 21 prior core passes.
  **353 core tests green in 4s before** the feature.

- **Phase 2 — feature shipped (`feat(supervision):`, commit `a1cc09c`):
  the supervision verdict (orphan / stale-code / no-restart-net) now
  reaches Discord via a single-source-of-truth pure builder.** This is the
  **#1 recurring HIGH live finding** across passes #20/#21 (and earlier):
  the trader runs as an orphaned `runner.py` (PPID 1), systemd
  disabled/inactive, behind HEAD — the moment its git-watcher / deadman
  does `os._exit(0)` it stays DOWN permanently. The verdict was computed
  *inline* in `dashboard.supervision_api`, surfaced ONLY on
  `/api/supervision` (a dashboard the operator never opens), with **no
  pure builder and no test** — so the reporter could not reuse it without
  re-deriving it (an invariant-#10 violation in waiting). New
  `analytics/supervision.py::build_supervision` (pure, never-raises;
  verdict/recommendation strings extracted verbatim + an `actionable`
  flag so the suppression rule is single-sourced). `supervision_api`
  refactored to delegate (impure pid/ppid/systemctl/git probes stay in
  the caller — the `build_runner_heartbeat`/`_heartbeat_line` split);
  behaviour-preserving, only additive JSON key is `actionable`.
  `reporter._supervision_line` composes it verbatim, wired into hourly +
  daily-close (the `_singleton_lock_line`/`_heartbeat_line` precedent),
  additive-failure contract (fault → `""`, never an exception),
  observational only (invariants #2/#12). Suppression: HEALTHY silent;
  STALE/UNSUPERVISED/UNSUPERVISED_STALE **and UNKNOWN** surfaced (an
  unreadable user bus is closer to "no safety net" than "healthy" — the
  recommendation already names the verify commands). **21 new locks in
  `tests/test_supervision.py`**: the full verdict matrix with the exact
  recommendation strings pinned verbatim, orphan precedence over a
  stale-cached `systemctl active`, stale derivation on every None/equal
  SHA permutation, endpoint↔builder byte-parity via the Flask test
  client, and reporter suppression/surfacing. The `_safe`-contract test
  **caught and fixed a real gap in the builder's own `except` path** (it
  re-touched the passed `now`, so a malformed clock — itself a way into
  that branch — re-raised; fixed to derive the fallback `as_of` from a
  fresh clock). **273 touched/adjacent locks green after**
  (`test_supervision /_core_reporter /_core_runner /_runner_cycle
  /_build_info /_runner_heartbeat /_core_strategy`). **Applies on the
  next runner restart** — the running orphan keeps the old inline-only
  path until rebooted (the documented deploy-stale pattern; finding #1).

- **Phase 3 — live findings (running `:8090`, orphan PID 1884688,
  2026-05-18 ~12:23 UTC; host under the review-swarm load this pass's own
  siblings contribute to).**
  1. **Runner is UNSUPERVISED_STALE (HIGH, ops, not code-fixable —
     continuity of #20/#21 #1).** `/api/supervision`:
     `verdict=UNSUPERVISED_STALE`, `orphan=true`, `ppid=1`, systemd bus
     `No medium found`, `boot_sha=b4dfd48` vs `head_sha=9c14c96`,
     `behind:3`. No `Restart=always` net; the next git-watcher / deadman
     `os._exit(0)` leaves the trader DOWN. **This pass's own supervision
     feature will not deploy until the operator restarts** —
     `systemctl --user enable --now paper-trader`.
  2. **Live trader frozen — IDLE_STORM (HIGH, host-saturation, not a
     code/prompt bug — continuity + recalled
     `pt-no-decision-host-saturation`).** `/api/runner-heartbeat`:
     liveness HEALTHY but `decision_efficacy=IDLE_STORM`,
     `consecutive_no_decision=20` (100%), `restart_recommended=true`.
     The last 12 `decisions` rows are **all**
     `NO_DECISION | claude returned no response (timeout/empty)` — a host
     **timeout**, NOT quota (`_quota_exhausted` would tag it differently)
     and NOT the new host-guard skip (orphan predates it — finding #5).
     `/api/decision-drought`: ongoing PARALYSIS 25.5h / 77 cycles / 57
     NO_DECISION (74%). The review harness induces the freeze it
     observes; instrumentation is correct.
  3. **Capital-paralysis BLEEDING (HIGH, downstream of #2, correctly
     surfaced — continuity of #21 #3).** `/api/capital-paralysis`: 98.1%
     deployed, $18.49 cash, LITE 61% of book, `paralysis.verdict=BLEEDING`,
     `involuntary_alpha_bleed_pct=-2.21%` across 6 parse-failure droughts;
     last fill `2026-05-17T09:38 BUY MU` (>27h ago). `can_act=true` (FREE
     — `min_actionable $9.73 < $18.49`) yet BLEEDING because the
     NO_DECISION storm, not dry powder, is the bind. Nothing to fix in
     code.
  4. **Data trust intact — positive finding.** `/api/feed-health`
     HEALTHY (newest live article 0.0h old, 926 live/2h, 5041/24h, no
     split-brain). `equity_curve` tail flat at `tv=972.69 cash=18.49`
     across the last decision points — consistent with a frozen book, NOT
     a corrupt one. This *isolates* the freeze: a host-timeout on the
     Opus call, not a blind feed.
  5. **This pass's own + prior endpoints are deploy-stale on the orphan
     (continuity, expected).** `/api/host-guard` returned **empty**
     against the running orphan because that route was added in `188e819`
     *after* the orphan's boot `b4dfd48`; likewise the new
     `/api/supervision` builder-delegation and `_supervision_line` are
     inert until restart. The dashboard's `runner-heartbeat.notify` read
     `verdict=UNKNOWN` (that process never attempted a Discord send) —
     not a regression, the documented deploy-stale pattern. Restarting
     the trader (#1) resolves 1, 2, 5 and activates this pass's feature.

  None of 1–5 is a new quick safe code fix: 1–3 + 5 are
  ops/host/continuity, 4 is a positive confirmation. No Phase-3 fix
  folded in.

- **Concurrency / staging discipline.** Ran with **heavy concurrent
  siblings** mutating the SAME shared-tree files: `dashboard.py` (sibling
  `@swr_cached("stress_scenarios")` endpoint, uncommitted, not in HEAD)
  and `reporter.py` (sibling `stress_scenarios`/`sector_exposure` imports
  + `_stress_line`), plus dirty `../digital-intern/` and sibling-untracked
  `analytics/{stress_scenarios,game_plan}.py` /
  `ml/{gate_pnl,gate_realized}.py`. Never `git add -A`. My-only patches
  were extracted by hunk classification and staged into the index via
  `git apply --cached` (working tree — sibling work — untouched);
  `git diff --staged` verified **zero sibling tokens** and the staged
  blobs were `ast.parse`-clean and sibling-dependency-free before commit.
  The sibling-induced `test_swr_prewarm_coverage::…stress_scenarios`
  failure is NOT in the committed tree (the staged `dashboard.py` has 0
  `stress_scenarios` occurrences) and is the sibling's incomplete work,
  not a regression. Exactly 4 path-scoped files committed
  (`analytics/supervision.py`, `dashboard.py`, `reporter.py`,
  `tests/test_supervision.py`); AGENTS.md committed separately alongside
  this entry. Deliverable confirmed on `origin/master` as `a1cc09c` by
  content (4 files, 0 sibling tokens).

- **Run the touched/adjacent locks (bounded — the full suite is ~25 min
  and load-flakes timing tests above load avg ~10):** `cd
  /home/zeph/trading-intelligence/paper-trader && python3 -m pytest
  tests/test_supervision.py tests/test_core_reporter.py
  tests/test_build_info.py -q` (123 green). The pass-#22 lock is
  `tests/test_supervision.py` (full verdict matrix + endpoint↔builder
  byte-parity + reporter suppression/surfacing).

*Review pass #22 (paper-trader core hybrid) appended 2026-05-18. Prior content above is unmodified.*

---

### 2026-05-18 review pass #20 (ML+backtest hybrid · realized-gate measurement from captured decision · live findings)

- **Phase 1 — no new bug (bugs_fixed = 0; no Phase-1 commit, the commit
  guard explicitly permits it).** Re-traced the core trio
  (`decision_scorer.py`, `backtest.py`, `run_continuous_backtests.py`)
  plus the newest/least-reviewed diagnostics — `overfit_gap.py`,
  `baseline_trend.py`, `gate_pnl.py`, `horizon_audit.py`,
  `corpus_audit.py`, `response_audit.py` — and shared deps
  `validation.split_outcomes_temporal` / `evaluate_scorer_oos` /
  `calibration._spearman`. All defensive, all exact-value locked.
  Consistent with the 18 prior no-new-bug ML/backtest passes (#5–#19).
  **438 ML/backtest tests green before** the feature.

- **Feature shipped (Phase 2, `feat(ml):`): `paper_trader/ml/gate_realized.py`
  — the gate's REALIZED arm effect from its *captured then-deployed*
  decision, ZERO re-prediction.** `gate_audit`/`gate_pnl` call
  `scorer.predict()` with **today's** pickle — a counterfactual their own
  docstrings disclaim. Commit `60b20d9` added
  `gate_scorer_pred`/`gate_off_dist` to make the gate's true call
  measurable; nothing consumed it. This buckets realized 5d/10d/20d by
  the gate's *actual historical* arm with no predict/pickle, and routes
  `gate_off_dist=True` rows to a separate `abstained` bucket excluded
  from the verdict — the honesty re-prediction structurally cannot
  replicate. Reuses `gate_audit.gate_arm` (SSOT) +
  `validation.split_outcomes_temporal`; read-only, never raises; names
  the deploy-stale state `GATE_CAPTURE_NOT_YET_POPULATED`. CLI exits 2 on
  `GATE_HARMFUL`. **24 exact-value offline locks** in
  `tests/test_gate_realized.py`. Commit `377c6f7`.
  ```bash
  python3 -m pytest tests/test_gate_realized.py -q   # 24 green
  cd /home/zeph/trading-intelligence/paper-trader && python3 -m paper_trader.ml.gate_realized
  ```

- **Quant findings (Phase 3, live — 6 distinct, reported / out of
  surgical scope).** (1) **`gate_audit`'s live `GATE_EFFECTIVE`
  (+4.41pp) is a re-predicted counterfactual, NOT proof the deployed
  gate helped** — `gate_realized`=`GATE_CAPTURE_NOT_YET_POPULATED`
  (loop predates `60b20d9`, 0 captured rows), so the honest verdict is
  currently unmeasurable. (2) **Scorer has zero durable OOS skill while
  gating 100% of cycles** — `scorer_skill_log` oos_ic median ≈0.02,
  oos_dir_acc ≈0.5, `gate_active=true` every cycle; `skill_trend`
  oos_rmse recent **13.0 > mean-predictor 8.46**, `DEGRADING`;
  `baseline_compare --oos`=`MLP_NO_BETTER_THAN_TRIVIAL`. (3) **Pass-#19
  anti-overfit fix (`5a0af2d`) shipped but NOT deployed; gap widening**
  — `overfit_gap`=`MILD_OVERFIT`, oos/val ratio **1.38 DEGRADING**;
  running loop still gates on the memorizing `(64,32,16)` net. (4)
  **winner→ArticleNet loop broken WORSE than the documented "~4/5"** —
  ~7/8 recent cycles `inject err: database locked after 4 attempts` /
  `trainer rc=-15` / `trainer timeout` (CLAUDE.md §5 step 5
  non-functional; digital-intern-side lock). (5) **Backtest dispersion
  is pure leverage-beta** — run 6234 vs_spy +165% beside 6236 vs_spy
  −52%; 476 complete/24 failed, 0 NaN, orphan-reap + empty-SPY guards
  working, loop healthy. (6) **`calibration` in-sample `WELL_CALIBRATED`
  (spearman 0.355) optimistic, contradicted by OOS ledger.**
  `baseline_skill_log.jsonl` still absent (loop predates `6ade72d`,
  wiring correct/inert). **Decisive operator action: restart
  `run_continuous_backtests.py`** — deploys the regularized net,
  gate-decision capture (then `gate_realized` becomes measurable), the
  baseline ledger, and multi-horizon outcomes at once.

- **Concurrency note.** 3+ sibling agents on the shared monorepo tree;
  never `git add -A`; exactly two path-scoped files staged for the
  feature, AGENTS.md appended-only & committed separately.

*Review pass #20 (ML+backtest hybrid) appended 2026-05-18. Prior content above is unmodified.*

---

### 2026-05-18 review pass #23 (paper-trader core hybrid · per-position hold-age in the Opus prompt · live findings)

- **Phase 1 — no new bug (bugs_fixed = 0; no Phase-1 commit, the commit
  guard explicitly permits it).** Re-read the seven core files in full
  (`runner.py`, `reporter.py`, `signals.py`, `strategy.py`,
  `market.py`, `store.py`) plus a structured sweep of the 9.5k-line
  `dashboard.py` (70 `@app.route`s; read `_position_ages_from_trades`,
  `risk_api`, `supervision_api`, `equity_integrity_api` in full).
  Traced the full `decide()` claude/fallback/retry state machine, the
  host-saturation skip arms, the singleton-lock degrade/recheck path,
  the runner-state sidecar future-clamp, and the `_mark_to_market`
  expired-option intrinsic / `stale_mark` logic. All defensive, all
  exact-value locked, no genuine defect — consistent with the 22 prior
  mature core passes. Baseline green before the feature
  (`test_core_market /_store /_runner /_signals` 190, plus
  `test_core_runner_cycle /_parse_retry /_decision_context*` 230).

- **Feature shipped (Phase 2, `feat(strategy):`, commit `ab710a3`):
  per-position HOLD AGE in the Opus decision prompt.** The prompt's
  position lines rendered `qty/avg/mark/P/L` but never *how long* a lot
  had been held — leaving the decision engine structurally blind to the
  desk's **#1 documented pathology, the disposition effect** (riding
  losers / cutting winners). This was live and visible this pass:
  `/api/hold-discipline` `DISPOSITION_DRAG`, LITE held **3.8d** at a
  loss = **7.12× the empirical median losing hold**, yet the prompt gave
  Opus no age signal. New pure `strategy._hold_age_str(opened_at,
  now=None)` → compact `42m`/`5h`/`3d` (day-flooring **aligned with
  `dashboard._position_ages_from_trades` / `/api/risk`** so the two
  surfaces never disagree by a day), derived from the `opened_at`
  already carried on every `snap["positions"]` row (it is reset to the
  re-entry instant when a fully-closed lot reactivates — see
  `store.upsert_position` — so it is the correct *current* holding
  period, not the all-time first touch). `_build_payload` renders a
  ` held=<age>` token on every stock **and** option line, placed
  **before** the `[STALE MARK …]` suffix so the disposition signal
  never masks the unreliable-P/L warning. Observational only — surfaces
  the raw fact, never gates/caps (the `stale_mark` precedent;
  invariants #2/#12). **Degrade-safe:** missing/unparseable `opened_at`
  → no token, **byte-identical to pre-feature** for any snapshot lacking
  the field (incl. the handcrafted test snapshots — a regression guard
  asserts this on the existing stale-position test). Future `opened_at`
  (clock stepped back — the documented skew hazard) clamps to `0m`.
  Verified live offline against the real book:
  `LITE … P/L=$-6.21 (-1.0%) held=3d`, `MU … held=1d`. **14 exact-value
  locks** in `tests/test_core_strategy.py`
  (`TestHoldAgeStr` + `TestHoldAgeInPrompt`: bucket flooring incl. the
  1h/1d boundaries, sub-minute→`0m`, missing/empty/unparseable→`""`,
  future-clamp, naive-tz-as-UTC, stock+option render, no-opened_at→no
  token, token-precedes-STALE-MARK ordering). 87 `test_core_strategy`
  green after.
  ```bash
  cd /home/zeph/trading-intelligence/paper-trader && python3 -m pytest \
    tests/test_core_strategy.py -q   # 87 green (14 new)
  ```

- **Phase 3 — live findings (running `:8090`, orphan PID 1901379,
  2026-05-18 ~13:12 UTC; host under the review-swarm load this pass's
  own siblings contribute to). 5 distinct, none a new quick code fix.**
  1. **Live trader frozen — IDLE_STORM (HIGH, host-saturation, NOT a
     code/prompt bug — continuity + recalled
     `pt-no-decision-host-saturation`).** Last 8 `decisions` all
     `NO_DECISION`; `/api/runner-heartbeat`
     `decision_efficacy=IDLE_STORM`, `consecutive_no_decision=20`
     (100%), `restart_recommended=true`. The review harness induces the
     freeze it observes; instrumentation is correct.
  2. **POSITIVE — the pass-#22 host-guard skip (`9c14c96`) is now
     DEPLOYED and visibly correct in production.** Prior pass #22
     reported it deploy-stale/inert on the orphan; this pass the
     `decisions` log carries genuine `skipped claude call — host
     saturated: 5/6 concurrent Opus (>4)` rows **distinct from**
     `claude returned no response (timeout/empty)` model-timeout rows —
     the distinct-reason instrumentation validated live (it is dodging
     the +1.5GB doomed Opus subprocess during the storm exactly as
     designed).
  3. **Runner UNSUPERVISED_STALE (HIGH, ops, not code-fixable —
     continuity of the #1 recurring finding).** `/api/supervision`:
     `verdict=UNSUPERVISED_STALE`, `orphan=true`, `ppid=1`, systemd bus
     `No medium found`, `boot_sha=871795e` vs `head_sha=ab710a3`
     (`behind:1` — this pass's own commit). No `Restart=always` net.
     Decisive action: `systemctl --user enable --now paper-trader`.
  4. **POSITIVE — data + P&L trust intact.** `/api/feed-health`
     HEALTHY (newest live article 0.0h, 915 live/2h, 5184/24h, both
     candidate DBs in lockstep, **no split-brain**);
     `/api/equity-integrity` `CLEAN` (794 points, cash never negative,
     0 suspect jumps); `/api/runner-heartbeat.notify` `HEALTHY` (last
     Discord send OK, 0 consecutive failures — openclaw PATH/shebang
     resolution working). The freeze is **isolated to the Opus call**
     (host timeout), not a blind feed / corrupt book / dark channel.
     `continuous.log` 0 tracebacks (GDELT backoff only; backtest loop
     healthy at run 6237).
  5. **This pass's hold-age feature is deploy-stale on the running
     orphan (continuity, expected).** The runner booted on `871795e`
     before `ab710a3`; verified working offline against the live
     snapshot. Activates on the next runner restart (the documented
     deploy-stale pattern; restarting per #3 also resolves 1, 5 and
     activates this feature).

  1 + 3 are ops/host-saturation continuity; 2 + 4 are positive
  production confirmations; 5 is the expected deploy-stale pattern. No
  Phase-3 fix folded in.

- **Concurrency / staging discipline.** Ran with heavy concurrent
  siblings on the shared monorepo tree (sibling-modified
  `paper_trader/ml/decision_scorer.py`, untracked sibling
  `analytics/{game_plan,ticker_dossier}.py`,
  `ml/{deploy_audit,gate_pnl}.py`, `preflight.py`, dirty
  `../digital-intern/`). Never `git add -A`. Staged exactly the two
  path-scoped files I changed (`paper_trader/strategy.py`,
  `tests/test_core_strategy.py`); `git diff --staged` verified **zero
  sibling tokens** and both staged blobs were `ast.parse`-clean before
  commit. AGENTS.md committed separately alongside (not counted as the
  feature).

*Review pass #23 (paper-trader core hybrid) appended 2026-05-18. Prior content above is unmodified.*

### 2026-05-18 review pass #21 (ML+backtest hybrid · deployed-model staleness audit · live findings)

- **Phase 1 — no new bug (bugs_fixed = 0; no Phase-1 commit, the commit
  guard explicitly permits it).** Re-traced the core trio
  (`decision_scorer.py`, `backtest.py`, `run_continuous_backtests.py`) end
  to end: `_ml_decide` scorer-gate arms + off-dist abstention, the
  `score=`/`scorer=` first-match regex disambiguation,
  `_compute_decision_outcomes` 5d-window guards, `_enforce_risk_exits`
  SL-before-TP semantics, `train_scorer` dedup-keyed-by-action +
  split-before-scale, `_train_decision_scorer`'s three independently
  guarded OOS blocks. The advisor-flagged `sell_score = best_score` in the
  CONTRARIAN branch is cosmetic-only (the documented contrarian design —
  it sells *because* the name scored overbought-high; feeds only the
  reasoning string's `score=` token, which is the intended ml_score
  feature for that SELL). All defensive, all exact-value locked.
  Consistent with the 19 prior no-new-bug ML/backtest passes (#5–#20).
  **371 ML/backtest tests green before** the feature (394 after, incl. the
  new `deploy` filter).

- **Feature shipped (Phase 2, `feat(ml):`): `paper_trader/ml/deploy_audit.py`
  — a durable signal that the conviction gate is running on a stale net,
  the single most-repeated finding of passes #15–#20.** Every prior pass
  rediscovered "the running loop predates the retune; the deployed pickle
  is the memorizing `(64,32,16)` net, not the regularized `(32,16)` one;
  restart to redeploy" by ad-hoc `pickle.load` inspection — there was **no
  durable, trendable instrument** for it. This extracts the MLPRegressor
  kwargs to `decision_scorer.MLP_CONFIG` (single source of truth;
  `train_scorer` now builds `MLPRegressor(**MLP_CONFIG)`) and introspects a
  deployed pickle's *fitted-model attributes* against it. Verdicts
  `DEPLOYED_MATCHES_SOURCE` / `DEPLOYED_STALE_CONFIG` (names every drifted
  key with deployed≠expected values) / `LSTSQ_FALLBACK` / `UNREADABLE_PICKLE`
  / `INSUFFICIENT_DATA`; pure/total, never raises; CLI exits 2 on STALE.
  Distinct from `/api/build-info` (git-SHA / process level) — this measures
  the **model artifact the gate actually consumes**, which a git SHA cannot
  see (a pickle can lag the source even on a fresh process). `deploy_stale`
  (True/False/None) is wired into `_append_scorer_skill_log` so the state
  is trendable per-cycle, not only a CLI. **18 exact-value offline locks**
  in `tests/test_deploy_audit.py`. Commit `eb02c9f`.
  ```bash
  python3 -m pytest tests/test_deploy_audit.py -q          # 18 green
  cd /home/zeph/trading-intelligence/paper-trader && python3 -m paper_trader.ml.deploy_audit
  ```

- **Quant findings (Phase 3, live — 6 distinct, reported / out of surgical
  scope).** (1) **DECISIVE — the deployed scorer is provably stale:
  `deploy_audit` = `DEPLOYED_STALE_CONFIG`, 6/8 hyper-params drifted**
  (`hidden_layer_sizes=(64,32,16)≠(32,16)`, `max_iter=600≠1000`,
  `alpha=1e-4≠1e-2`, `early_stopping=False≠True`,
  `validation_fraction=0.1≠0.15`, `n_iter_no_change=10≠25`). The running
  loop keeps the old kwargs resident and re-pickles the memorizing net
  every cycle. (2) **Scorer has NEGATIVE OOS skill while gating 100% of
  cycles** — `skill_trend`=`NEGATIVE_OOS_SKILL`: oos_rmse recent 13.00 ≥
  mean-predictor baseline 8.46, median oos_ic 0.03, oos_dir_acc 0.505
  (coin-flip), `gate_active=1.0` every cycle, **trend DEGRADING**. (3)
  **Overfit gap real and widening** — `overfit_gap`=`MILD_OVERFIT`, recent
  median oos/val ratio **1.38** (older 1.17 → recent 1.38, DEGRADING); the
  regularized net that closes it is not deployed (same root cause as #1).
  (4) **Magnitude wildly miscalibrated at the top decile (OOS)** —
  `calibration --oos` d10 mean_pred **+17.92%** vs mean_realized **+1.07%**
  (16.8pp over-predict), d9 +4.93 vs +1.69; in-sample `WELL_CALIBRATED`
  but OOS `DIRECTIONAL_BUT_BIASED`. The gate's `p>10 → ×1.3` arm up-sizes
  conviction hardest exactly where prediction is least reliable. Nuance:
  `baseline_compare --oos` is **not** `MLP_WORSE_THAN_TRIVIAL` right now
  (MLP rank_ic 0.115 > best one-liner `mom20` 0.072, ic_gap +0.043 over
  n=1385) — but the gate acts on miscalibrated *magnitude buckets*, not
  rank, so the modest rank edge is unexploited. (5) **Gate-decision
  capture still 0/6927** — `gate_scorer_pred` populated in 0 of 6927
  `decision_outcomes.jsonl` rows ⇒ `gate_realized` =
  `GATE_CAPTURE_NOT_YET_POPULATED` (all buckets n=0); the honest realized-
  gate effect remains structurally unmeasurable until restart.
  `decision_outcomes.jsonl` itself is clean (6927 rows, **0 non-finite**,
  p1=−18.9 / med=0.4 / p99=27.0 / sd=8.07). (6) **winner→ArticleNet loop
  non-functional + dispersion is leverage-beta** — `_inject_and_train` =
  `inject err: database locked after 4 attempts` every recent cycle
  (digital-intern-side write-lock; CLAUDE.md §5 step 5 dead); 500 runs /
  476 complete / 24 failed / 0 running (orphan-reap + empty-SPY guards
  healthy, 0 NaN); run 6234 vs_spy **+165%** beside 6236 vs_spy **−52%** =
  pure 3×-ETF beta draw, not repeatable alpha. `baseline_skill_log.jsonl`
  still absent (loop predates `6ade72d`). **Decisive operator action
  (unchanged across 7 passes, now durably surfaced): restart
  `run_continuous_backtests.py`** — `deploy_audit` flips to
  `DEPLOYED_MATCHES_SOURCE`, the regularized net deploys, `gate_realized`
  becomes measurable, and the baseline ledger populates.

- **Concurrency note.** 3+ sibling agents on the shared monorepo tree;
  never `git add -A`; exactly four path-scoped files staged for the
  feature, AGENTS.md appended-only & committed separately (re-read after a
  sibling appended pass #22/#23 between my read and write).

*Review pass #21 (ML+backtest hybrid) appended 2026-05-18. Prior content above is unmodified.*

---

### 2026-05-18 review pass #24 (paper-trader core hybrid · EQUITY INTEGRITY Discord line · live findings)

- **Phase 1 — no new bug (bugs_fixed = 0; no Phase-1 commit, the commit
  guard explicitly permits it).** Read the seven core files in full
  (`runner.py`, `reporter.py`, `signals.py`, `strategy.py`, `market.py`,
  `store.py`) + a targeted sweep of `dashboard.py` (70 `@app.route`s; read
  `equity_integrity_api` in full). Traced `decide()`'s
  claude/fallback/retry + host-saturation pre-flight/mid-call-reprobe
  state machine, the singleton degrade/recheck path, `_mark_to_market`
  expired-option/`stale_mark`, `_maybe_daily_close` half-day anchoring,
  and `store.upsert_position` reactivate/blend. **Live forensics, not just
  a read:** confirmed the live portfolio↔equity_curve divergence is a
  *mid-cycle slow snapshot under load* (only `decide()` writes equity,
  paired with the decision; `buying_power.py:233`/`sector_exposure.py:296`
  `_portfolio_snapshot` are `__main__`-only smoke tests, NOT dashboard
  writers) and that real curve corruption is already covered by
  `equity_integrity`. Consistent with the 23 prior mature core passes.
  Focused-core baseline green before the feature (359:
  `test_core_{market,store,runner,strategy,signals,reporter}`).

- **Feature shipped (Phase 2, `feat(reporter):`, commit `1f72230`):
  EQUITY INTEGRITY Discord line.** `build_equity_integrity` was
  **dashboard-only** (`/api/equity-integrity`, pass #18) — yet its
  docstring states every headline P&L surface the operator reads
  (hourly Equity/P/L, `_benchmark_line`, `/api/drawdown`,
  `/api/analytics` Sharpe) derives from `equity_curve`, so a silent
  negative-cash over-draw (#12) / non-positive-equity / no-trade
  mismark-jump poisons all of them with nothing in Discord saying so.
  This is the exact dashboard→Discord gap `_heartbeat_line` /
  `_capital_pulse_line` / `_singleton_lock_line` each closed, one
  dimension over. New `reporter._equity_integrity_line(store)` composes
  `build_equity_integrity` **verbatim** (single source of truth,
  invariant #10) feeding it the **identical** store reads the endpoint
  uses (`equity_curve(limit=5000)` + `recent_trades(5000)`) so the two
  surfaces can never drift. Surfaces only `CORRUPT`/`SUSPECT`;
  `CLEAN`/`NO_DATA`/`ERROR`/non-dict suppressed (the `_heartbeat_line`
  lying-green-light precedent — a clean curve adds no hourly noise).
  Observational only, never gates, no caps (invariants #2/#12). Any
  builder/store fault drops the line, never the summary (the reporter
  failure contract). Wired into `send_hourly_summary` **and**
  `send_daily_close` immediately after `_heartbeat_line` (operational +
  data-integrity grouped, before the P&L/session blocks — "can you trust
  the number?" before the numbers). Applies on next paper-trader restart
  (the documented deploy-stale pattern). **11 new exact-assert tests** in
  `tests/test_core_reporter.py::TestEquityIntegrityLine` (CORRUPT/SUSPECT
  verbatim-headline surface, CLEAN/NO_DATA/ERROR/None suppression,
  builder-raises→`""`, **real-builder no-drift** on a temp Store for
  both a negative-cash CORRUPT and a no-trade-+20% SUSPECT, hourly+daily
  wiring, summary-still-sends-when-builder-faults). 93
  `test_core_reporter` green, 380 focused-core+`equity_integrity` green.
  ```bash
  cd /home/zeph/trading-intelligence/paper-trader && python3 -m pytest \
    tests/test_core_reporter.py -q   # 93 green (11 new)
  ```

- **Phase 3 — live findings (running `:8090`, runner pid 1921870 holds
  the singleton lock, 2026-05-18 ~13:5x UTC; host under the review-swarm
  load this pass's own siblings contribute to — load avg ~23, 9
  concurrent Opus, 210 MB free). 6 distinct, none a new quick code fix.**
  1. **Live trader frozen — IDLE_STORM (HIGH, host-saturation, NOT a
     code/prompt bug — continuity + recalled
     `pt-no-decision-host-saturation`).** 20/20 recent `decisions`
     `NO_DECISION`, `/api/runner-heartbeat`
     `decision_efficacy=IDLE_STORM`, `restart_recommended=true`. Of the
     last 22: 4 are the honest `skipped claude call — host saturated`
     (host-guard `9c14c96` deployed & correct), 18 are
     `claude returned no response (timeout/empty)` — the storm hits
     *during* the 180s call after pre-flight passed and the mid-call
     re-probe (`strategy.py:1494`) misses the window. Architectural
     reality of this box, self-clears when the review swarm ends.
  2. **Stale `equity_curve` poisons benchmark/alpha during the storm
     (MEDIUM, no integrity alarm covers it).** `/api/portfolio`
     **$928.92** but `/api/benchmark` computes off the frozen
     `equity_curve` (**$972.69**) → headline "lagging by 2.27pp"
     understates the true ~6.7pp gap (928.92 vs 995.43 buy-and-hold).
     During saturation yfinance is also starved so LITE/MU stale-mark
     and a run of equity points freezes at the stale total.
     `equity_integrity` reads `CLEAN` because the divergence is
     portfolio-vs-curve (and the recovery delta < the 8 % jump gate),
     **not within recorded points** — so the new EQUITY INTEGRITY line
     correctly stays silent here too. A "portfolio vs latest
     equity-point divergence" builder would close this; out of scope for
     a lean pass (a deliberate new-builder decision, not a surgical fix).
  3. **`/api/capital-paralysis` verdict `FREE` is misleading at $18.49
     (MEDIUM, trader-perspective).** "FREE — $18.49 cash (2.0 %)
     available; the book can act on a new signal without selling" — but
     $18.49 cannot fund the cheapest watchlist name (MU $728, NVDA $224,
     SPY $738); the book is *functionally* paralysed. `_capital_pulse_
     line` suppresses `FREE`-non-bleeding so the operator gets **no
     Discord signal** they are stuck. The affordability-aware
     `buying_power` builder is the right lens but is not the verdict
     driver. Changing the `FREE` threshold touches a documented builder
     with its own tests/precedent — a deliberate decision, not a quick
     fix.
  4. **`logs/runner.log` is stale/misleading (MEDIUM, operability).**
     Last `**HOURLY**` body is `2026-05-17 16:54 UTC`; the tail is a
     loop of `another paper trader is already running … exiting` failed
     launch attempts — **not** the live runner's (pid 1921870) output
     (its real stdout is the systemd journal). A trader tailing
     `runner.log` to answer "is my bot alive?" is badly misled (it looks
     dead). Recalled `pt-no-decision-host-saturation` ("runner.log often
     stale"); canonical live surface is `/api/runner-heartbeat`, not the
     file.
  5. **A launcher repeatedly attempts duplicate runners (LOW, info).**
     The single-instance `fcntl.flock` guard (invariant #19) is working
     **correctly** — it refuses every duplicate — but something (a
     systemd unit losing the lock race, or a wrapper) keeps trying. Not
     harmful (fail-closed by design); indicates launcher misconfig worth
     an operator note.
  6. **POSITIVE — data + channel trust intact.** `/api/mark-integrity`
     `CLEAN` ("All 2 marks live"), `/api/equity-integrity` `CLEAN` (797
     points, cash never negative, 0 suspect jumps),
     `/api/runner-heartbeat.notify` `HEALTHY` (last Discord send OK,
     0 failures — openclaw PATH/shebang fix holding; the `node`-not-found
     lines in `runner.log` are stale 2026-05-17). `continuous.log`
     healthy (GDELT backoff only, backtest loop progressing). The freeze
     is isolated to the Opus call, not a blind feed / corrupt book /
     dark channel.

  1+4+5 are host/ops continuity; 2+3 are real trader-perspective gaps
  (deliberate-decision, not surgical); 6 is positive confirmation. No
  Phase-3 fix folded in.

- **Concurrency / staging discipline.** Ran with heavy concurrent
  siblings on the shared monorepo tree (sibling ML/backtest HYBRID agent
  active, dirty `../digital-intern/`). Never `git add -A`. Staged exactly
  the two path-scoped files changed (`paper_trader/reporter.py`,
  `tests/test_core_reporter.py`); `git diff --staged` verified zero
  sibling tokens before commit. AGENTS.md committed separately alongside
  (not counted as the feature).

*Review pass #24 (paper-trader core hybrid) appended 2026-05-18. Prior content above is unmodified.*

### 2026-05-18 review pass #22 (ML+backtest hybrid · bootstrap gate-arm stability · live findings)

- **Phase 1 — no new bug (bugs_fixed = 0; no Phase-1 commit, the commit
  guard explicitly permits it).** Re-traced the core trio
  (`decision_scorer.py`, `backtest.py`, `run_continuous_backtests.py`) end
  to end: `train_scorer` dedup-key-includes-action + split-before-scale +
  weight-oversample; `_ml_decide` scorer-gate arms + off-dist abstention +
  the `score=`/`scorer=`/`news_urg=`/`news_count=` first-match regex
  disambiguation; `_compute_decision_outcomes` 5d-window guards + gate
  capture; `_enforce_risk_exits` SL-before-TP; `_train_decision_scorer`'s
  three independently-guarded OOS blocks; `predict_with_meta` clamp /
  off-distribution honesty path. The required Phase-1 test scenarios
  already have comprehensive exact-value coverage
  (`tests/test_backtest.py::TestBuyAndHold::test_buy_and_hold_exact_return`,
  `TestRiskExits` stop-loss/TP at exact close + cash, `TestSimPortfolio`
  position-cap; `tests/test_decision_scorer.py`
  `test_scorer_ranks_high_ml_score_above_low`, `test_handles_null_/
  non_finite_forward_return`, `test_predict_safe_with_garbage_features`).
  Consistent with the 20 prior no-new-bug ML/backtest passes (#5–#21).
  **Sole training-dynamics observation (reported, not fixed — the
  prior-pass precedent for in-scope-but-not-surgical):** `train_scorer`
  oversamples by weight (`np.repeat`) *before* `MLPRegressor`'s internal
  `early_stopping` split, so a duplicated row can land in BOTH the
  internal train and validation folds → optimistic `val_rmse` in the
  regularized (`early_stopping=True`) config. Inert today (deployed net
  has `early_stopping=False`); a real fix is a training-pipeline change
  out of surgical scope (CLAUDE.md §6).

- **Feature shipped (Phase 2, `feat(ml):`):
  `paper_trader/ml/gate_stability.py` — bootstrap-retrain stability of the
  conviction-gate ARM decision.** Closes the one question the saturated
  diagnostic suite structurally could not answer. Every sibling
  (`calibration`/`gate_audit`/`gate_pnl`/`gate_realized`/`skill_trend`/
  `baseline_compare`/`regime_audit`/`feature_importance`/`horizon_audit`)
  is a *point-estimate* tool on **one** model; `overfit_gap` trends the
  val/oos RMSE *ratio*; `deploy_audit` compares hyper-params. None can see
  the AGENTS.md-documented smoking gun — *"−89% then +32% for the same
  LITE vector across two retrain cycles"*, i.e. prediction instability
  *across* retrains. This bootstrap-resamples the temporal-train slice,
  fits K throwaway scorers with the **exact** `decision_scorer.MLP_CONFIG`
  + `build_features` pipeline `train_scorer` uses (SSOT imports; dedup +
  weight-oversample mirror `train_scorer` line-for-line the way
  `gate_pnl._reconstruct_base_conviction` mirrors `_ml_decide`), predicts
  the fixed OOS slice with each, and reports — at the five real gate arms
  (`gate_audit.gate_arm`, imported SSOT) — `gate_arm_flip_rate`,
  cross-bootstrap pred σ, modal-arm agreement. Verdicts
  `GATE_ARM_STABLE` / `GATE_ARM_BORDERLINE` / `GATE_ARM_UNSTABLE` /
  `INSUFFICIENT_DATA`; pure/total, never raises; seeded ⇒ reproducible;
  CLI exits 2 on UNSTABLE. **Safety-critical & test-locked: it never
  calls `train_scorer`, so it never writes the deployed
  `decision_scorer.pkl` the live gate consumes.** 13 exact-value offline
  locks in `tests/test_gate_stability.py` (strong-signal→STABLE,
  noisy-signal→UNSTABLE, consistent-offdist→STABLE [the non-obvious
  correctness lock: off-distribution *magnitude* alone is NOT
  instability], determinism, never-raises-on-garbage, SSOT identity, the
  no-pickle-write invariant, CLI exit-code/`--all` contract). Commit
  `69a6c94`.
  ```bash
  python3 -m pytest tests/test_gate_stability.py -q          # 13 green
  cd /home/zeph/trading-intelligence/paper-trader && python3 -m paper_trader.ml.gate_stability
  cd /home/zeph/trading-intelligence/paper-trader && python3 -m paper_trader.ml.gate_stability --all --bootstraps 16
  ```

- **Quant findings (Phase 3, live — 8 distinct, reported / out of
  surgical scope).** (1) **DECISIVE & NEW — the gate's economic spread is
  ~97% resample-luck.** `gate_stability` on the live
  `decision_outcomes.jsonl` (n_train=5460, n_eval=1507, K=10):
  `GATE_ARM_UNSTABLE`, **`gate_arm_flip_rate=0.9675`**, mean
  cross-bootstrap pred σ **5.5pp** (median 3.7), modal_agreement 0.586.
  This *reconciles the standing paradox* the AGENTS.md notes circle:
  `gate_audit --oos` reads **`GATE_EFFECTIVE`** on the deployed pickle
  (strong_tailwind +2.43% vs strong_headwind −0.85%, **spread +3.28pp**,
  arm_monotone 0.75) — looking economically justified — yet
  `skill_trend`/`baseline_compare` say ≈0 OOS skill. Resolution: that
  +3.28pp arm spread is **not a stable property of the features** —
  retrain on a different bootstrap of the *same* outcomes and 96.7% of
  decisions get a different conviction multiplier. The gate sizes capital
  on one pickle's training lottery; `gate_audit`'s single-model view
  structurally cannot see it. (2) **DECISIVE & NEW — `gate_pnl`'s
  documented "the floor effectively never binds" is empirically false: it
  binds on ~34% of BUY-intent.** Direct `backtest.db` count over recent
  runs 6208–6237: 20,613 `action='HOLD'` rows whose reasoning is
  `"ML score=… but notional too small"` vs 39,773 filled BUYs ⇒
  **floor-bind = 34.1% of BUY-intent** (samples are low-`ml_score`≈1.5
  signals silenced once the portfolio is fully invested / cash≈0). So
  *every* gate diagnostic (`gate_audit`/`gate_pnl`/`gate_realized`) and
  `decision_outcomes.jsonl` itself carries a **~34% selection bias** —
  they only ever see BUY-intent that survived the floor. `gate_pnl.py`'s
  Scope docstring ("base conviction ≥ ~5% so the floor effectively never
  binds") should be corrected to "binds on ≈⅓ of BUY-intent
  (cash-saturation, not the ×0.6 arm)". Reported not patched: a docstring
  edit to a sibling-owned module mid-flight on a 3+-agent shared tree is
  exactly the cross-agent contention the concurrency discipline warns
  against (prior passes likewise reported `gate_pnl`/`calibration`
  caveats rather than editing siblings); a DB-scanning tool to surface it
  durably is infeasible here (a 3-run scan of the USB-mounted
  `backtest.db` times out >45s). (3) **Deployed scorer still STALE** —
  `deploy_audit=DEPLOYED_STALE_CONFIG`, 6/8 drifted (`(64,32,16)≠(32,16)`,
  `alpha=1e-4≠1e-2`, `early_stopping=False≠True`, …); running loop
  predates the retune (continuity #15–#21). (4) **`baseline_compare`
  DEGRADED to `MLP_NO_BETTER_THAN_TRIVIAL`** — MLP OOS rank_ic **+0.060 <
  best one-liner `mom20` +0.082** (gap −0.023, dir_acc 0.508, n=1507).
  Pass #21 had MLP **+0.115 > mom20 +0.072** (explicitly NOT
  worse-than-trivial); the stale net's modest OOS rank edge has **eroded
  below a single momentum feature** as outcomes accumulated — a fresh
  degradation, same root cause as (1)/(3). (5) **Scorer OOS skill
  borderline-negative while gating 100% of cycles** —
  `scorer_skill_log.jsonl` (19 cycles): recent oos_rmse 10.2–16.7
  straddling the σ≈11.7 mean-predictor baseline, oos_dir_acc 0.47–0.56
  (coin-flip), oos_ic ≈ 0, `gate_active=true` every cycle. (6)
  **winner→ArticleNet loop still non-functional** — continuous.log:
  `inject err: database locked after 4 attempts` and `trainer timeout
  (injected 5276)`; CLAUDE.md §5 step 5 dead (continuity #21·6). (7)
  **Gate-decision capture still 0/7538 + 10d/20d absent** —
  `gate_scorer_pred` populated in 0 of 7538 rows, `forward_return_10d/20d`
  keys absent: running loop predates `60b20d9`/multi-horizon capture, so
  `gate_realized`/`horizon_audit` stay unmeasurable until restart.
  `decision_outcomes.jsonl` clean (7538 rows, **0 non-finite**, p1=−19.1 /
  med=0.5 / p99=26.7 / sd=7.95). (8) **Backtest health good; dispersion is
  leverage-beta** — 500 runs / 476 complete / 24 failed / **0 running**
  (orphan-reap + empty-SPY guards healthy); run 6234 vs_spy **+165%**
  beside 6236 vs_spy **−52%** on adjacent runs = 3×-ETF beta draw, not
  repeatable alpha. **Decisive operator action (unchanged across 8
  passes, now with a new instrument that *quantifies why it matters*):
  restart `run_continuous_backtests.py`** — the regularized net deploys,
  gate/horizon capture becomes measurable, and a post-restart
  `gate_stability` re-run measures whether the regularized (`alpha=1e-2`,
  `early_stopping`) net actually shrinks the 0.97 flip rate the
  memorizing net produces.

- **Concurrency note.** 3+ sibling agents on the shared monorepo tree
  (core-hybrid passes #22–#24 + AGENTS.md touches appeared between my
  read and write — re-read the bottom and used a race-immune append);
  never `git add -A`; exactly two path-scoped files staged for the
  feature (`paper_trader/ml/gate_stability.py`,
  `tests/test_gate_stability.py`); no domain file modified (Phase-1
  zero-diff); AGENTS.md append-only & committed separately.

*Review pass #22 (ML+backtest hybrid) appended 2026-05-18. Prior content above is unmodified.*

---

### 2026-05-18 review pass #25 (paper-trader core hybrid · EQUITY FRESHNESS builder+endpoint+Discord line · live findings)

- **Phase 1 — no new bug (bugs_fixed = 0; no Phase-1 commit, the commit
  guard explicitly permits it).** Read the seven core files in full
  (`runner.py`, `reporter.py`, `signals.py`, `strategy.py`, `market.py`,
  `store.py`) + targeted `dashboard.py` sweep. Re-traced `decide()`'s
  claude/fallback/retry + host-saturation pre-flight/mid-call-reprobe state
  machine, the singleton degrade/recheck path, `_mark_to_market`
  expired-option/`stale_mark`, `_maybe_daily_close` half-day anchoring,
  `store.upsert_position` reactivate/blend, `get_prices` bulk fallback.
  **Advisor-directed live forensic discriminator:**
  `grep -n "_portfolio_snapshot\|update_portfolio\|update_position_marks\|
  upsert_position\|record_equity_point" paper_trader/dashboard.py`
  returned **zero matches** — the dashboard never writes `portfolio` /
  `equity_curve`, so the live `/api/portfolio`($924.13) vs
  `/api/benchmark`($928.92) gap is the documented *mid-cycle slow snapshot*
  (only `decide()` writes equity, paired with the decision), **not** a
  hidden-writer bug. Memory `pt-portfolio-equity-divergence` still holds;
  Phase-1 `bugs_fixed=0` is the honest call, consistent with the 24 prior
  mature core passes. Focused-core baseline green before the feature (370:
  `test_core_{market,store,runner,strategy,signals,reporter}`).

- **Feature shipped (Phase 2, `feat(analytics):`): EQUITY FRESHNESS — the
  portfolio-vs-latest-equity-point divergence builder, the explicitly-
  deferred pass-#24 finding #2.** Pass #24 logged "A 'portfolio vs latest
  equity-point divergence' builder would close this; out of scope for a
  lean pass" — this ships it. New pure core
  `paper_trader/analytics/equity_freshness.py::build_equity_freshness`
  compares the live `portfolio` table total against the **latest recorded
  `equity_curve` point**. Under a host-saturation NO_DECISION storm the
  portfolio table re-marks every cycle while the curve lags a whole cycle
  behind, so `/api/benchmark` + the hourly `_benchmark_line`,
  `/api/drawdown`, `/api/analytics` Sharpe and the hourly P/L (all derived
  from `equity_curve`) silently misstate the true account by the
  divergence — `equity_integrity` reads CLEAN (the gap is
  portfolio-vs-curve, not *within* recorded points) so it does NOT cover
  this dimension. Verdict ladder `NO_DATA` / `FRESH` / `STALE_CURVE` /
  `DIVERGED`; **the actionable `DIVERGED` requires BOTH staleness AND
  >`divergence_pct` value gap** (advisor-directed — single-condition
  triggers spam on the by-construction ~1-cycle lag every healthy book
  carries). Cadence-aware stale threshold (2×`OPEN_INTERVAL_S` open /
  2×`CLOSED_INTERVAL_S` closed) selected from a `market_open` bool param
  (the `build_runner_heartbeat` precedent). Pure/total, never raises
  (non-iterable / garbage → degrades), future-`curve_ts` clamps age to 0
  (the `_hold_age_str` clock-skew precedent), anchors to the newest
  *positive* recorded point (skips corruption `equity_integrity` owns).
  Deliberately does **not** recompute a "corrected" benchmark (the
  `equity_integrity` reports-but-never-repairs precedent; AGENTS.md
  invariants #2/#12 — observational, never gates, no caps). `__main__` CLI
  (exit 2 on `DIVERGED`) for when :8090 is wedged (the `benchmark.py` CLI
  precedent). Wired `/api/equity-freshness` (mirrors `equity_integrity_api`
  exactly, EOF lowest-collision insertion) and
  `reporter._equity_freshness_line` into **both** `send_hourly_summary` and
  `send_daily_close` immediately after `_equity_integrity_line`
  (data-integrity grouped, before the P&L/session blocks); composes the
  builder **verbatim** (invariant #10) with the **identical** store reads
  the endpoint uses, surfaces only `DIVERGED`/`STALE_CURVE`
  (`FRESH`/`NO_DATA`/`ERROR` suppressed — the `_equity_integrity_line`
  lying-green-light precedent). **NOT** added to the Opus prompt
  (advisor-directed — operator-facing benchmark headlines, Opus never reads
  the hourly back). **32 new exact-assert tests** (`tests/
  test_equity_freshness.py` — builder verdicts/values, the strict-`>` band
  edge, cadence-aware open/closed threshold, corrupt-point skipping,
  clock-stepped-back clamp, never-raises, + Flask-client endpoint
  faithful-thin-wrapper/CORS/never-500s; `tests/test_core_reporter.py::
  TestEquityFreshnessLine` — verbatim-surface/suppression/degrade +
  real-builder no-drift on a backdated temp-Store row + hourly/daily
  wiring). 437 focused-core+related green.
  ```bash
  cd /home/zeph/trading-intelligence/paper-trader && python3 -m pytest \
    tests/test_equity_freshness.py tests/test_core_reporter.py -q   # green
  cd /home/zeph/trading-intelligence/paper-trader && \
    python3 -m paper_trader.analytics.equity_freshness            # CLI
  ```

- **Phase 3 — live findings (running `:8090`, runner pid 1946523, host
  under the review-swarm load this pass's own siblings contribute to —
  IDLE_STORM live). 3 distinct + positive confirmation.**
  1. **Live trader frozen — IDLE_STORM (HIGH, host-saturation, NOT a
     code/prompt bug — continuity, recalled
     `pt-no-decision-host-saturation`).** Last 6+ `decisions` all
     `NO_DECISION`; `/api/runner-heartbeat` `decision_efficacy=IDLE_STORM`,
     `restart_recommended=true`. Architectural reality of the 15GB box
     under the concurrent Opus review swarm + backtest committee;
     self-clears when the swarm ends. Not actionable in code.
  2. **NEW feature deployed & validated live.** The git-watcher
     auto-restarted the runner (pid 1946523, 68s uptime) onto the new
     on-disk code, so `/api/equity-freshness` serves **200** with the
     correct shape. Live state caught the gap precisely: `portfolio
     $926.95` vs frozen `curve $921.04` = **+0.64%** (over the 0.5% band)
     but correctly **`FRESH`** because the curve was only ~4m old (< the
     60m open stale threshold) — the "both stale AND diverged" gate
     suppressing normal mid-cycle drift exactly as the advisor prescribed;
     it flips to `DIVERGED` (and into Discord) only when a storm freezes
     the curve past threshold. The earlier-observed `$924.13` vs `$928.92`
     ($4.79 benchmark-headline overstatement) is the `DIVERGED` case this
     feature now closes — confirmed reachable.
  3. **Benchmark headline still computes off the curve (MEDIUM,
     by-design).** `/api/benchmark` "LAGGING 6.86pp / $928.92" is off the
     `equity_curve` latest point, not the live `portfolio` table — exactly
     the silent-misstatement this feature now makes *visible* in Discord
     (it does not, and by invariant #2/#12 must not, "fix" the benchmark
     math itself). Working as intended: surface, don't repair.
  4. **POSITIVE — data + book trust intact.** `/api/equity-integrity`
     `CLEAN` (800 points, cash never negative), `/api/mark-integrity`
     `CLEAN` (all marks live), `continuous.log` healthy (GDELT backoff
     only, backtest loop progressing). The freeze is isolated to the Opus
     call, not a blind feed / corrupt book / dark channel.

  1 is host/ops continuity; 2 is the shipped-and-validated feature; 3 is
  the real trader-perspective gap the feature now closes (surface, not
  repair — deliberate, per invariants #2/#12); 4 is positive confirmation.
  No Phase-3 fix folded in (the only code change in scope is the Phase-2
  feature).

- **Concurrency / staging discipline.** Heavy concurrent siblings on the
  shared monorepo tree (sibling ML/backtest HYBRID active, dirty
  `../digital-intern/`; AGENTS.md grew 5726→5934 lines mid-pass from
  sibling appends). Never `git add -A`. Staged exactly the five
  path-scoped files (`paper_trader/analytics/equity_freshness.py`,
  `paper_trader/dashboard.py`, `paper_trader/reporter.py`,
  `tests/test_equity_freshness.py`, `tests/test_core_reporter.py`);
  `git diff --staged` verified zero sibling tokens before commit. AGENTS.md
  appended-only & committed separately alongside (not counted as the
  feature).

*Review pass #25 (paper-trader core hybrid) appended 2026-05-18. Prior content above is unmodified.*

---

## Review pass #27 — paper-trader core hybrid (2026-05-18)

**bugs_fixed=0 · features_added=1 · user_findings=4**

- **Phase 1 — bugs_fixed=0 (mature).** Full read of the 7 core files
  (`runner.py`, `reporter.py`, `signals.py`, `strategy.py`, `dashboard.py`,
  `market.py`, `store.py`). No real bug, race, or dead-code defect found;
  389 focused-core tests green at baseline. The task's named coverage
  already exists with *specific-value* asserts (verified, not just
  imported): `test_core_store.py` (cash-after-BUY, recent_trades
  same-µs `id DESC` tie-break, equity ascending), `test_core_market.py` +
  `test_market_half_day.py` (weekend / pre-9:30 / 16:00-exclusive / 10:00
  / holiday / 13:00 half-day close), `test_core_runner.py`
  (`_maybe_daily_close` once-per-day, weekend, holiday, <16:05, half-day
  via `close_minute`). The template's "max_position / stop-loss" cases do
  NOT apply — invariant #12 (no hard limits; Opus full autonomy). No
  Phase-1 commit (guard-compliant).

- **Phase 2 — feature (committed `155f857`, pushed).**
  `reporter._pos_pct_weight()` + `_portfolio_lines(total_value=None)`:
  the Discord hourly/daily **Positions block now shows each position's own
  return % and its book weight %** (` (-11.0% · 59% bk)`), previously only
  raw `qty/avg/now/P/L$`. Closes a real operator gap — the live book sat
  **97.9% in 2 names (LITE 59% @ −11.0%, MU 39% @ −3.3%)** and the
  operator's primary surface rendered it identically to a balanced book
  (single-name concentration is the desk's #1 documented pathology). Pure
  position arithmetic (the same `pl_pct` `strategy._mark_to_market`
  already feeds Opus — NOT a re-derived builder verdict; invariant #10
  safe), additive/degrade-safe (stale/garbage/zero-cost/non-positive-total
  drops the offending token, never raises; the `stale_mark` precedent,
  invariants #2/#12). Default `total_value=None` keeps the existing
  unit-test callers byte-compatible. 11 new tests (canonical-format lock,
  live-LITE shape, stale suppression, option-100x weight,
  NaN/zero-cost/non-positive degrade, sub-1% precision); validated
  live-rendered against the real book.
  ```bash
  cd /home/zeph/trading-intelligence/paper-trader && python3 -m pytest \
    tests/test_core_reporter.py -q   # 121 green incl. TestPosPctWeight
  ```

- **Phase 3 — live findings (`:8090`; box under the review-swarm load
  this pass's own siblings contribute to). 4 distinct + positive.**
  1. **Live trader frozen — IDLE_STORM (HIGH, host-saturation, NOT a
     code bug — continuity, recalled `pt-no-decision-host-saturation`).**
     `/api/runner-heartbeat`: 20/20 cycles NO_DECISION,
     `decision_efficacy=IDLE_STORM`, `restart_recommended:true`.
     Architectural reality of the 15GB box under the concurrent Opus
     review swarm + backtest committee; self-clears when the swarm ends.
  2. **Dashboard `/api/state` and `/` hang >30s** under that same load
     while lightweight endpoints (`/api/portfolio`,
     `/api/runner-heartbeat`) stay instant — swap-thrash starving even
     the SWR cold path, not a code defect (SWR mitigation already in
     place; symptom of #1).
  3. **Capital paralysis live (the #2 documented pathology, surfaced):**
     cash $18.49 (2.1%), ~98% deployed in 2 names, total ~$900.84
     (≈−9.9%). The Phase-2 feature now makes the per-name % +
     concentration visible directly in the Discord positions block.
  4. **Concurrent same-role-agent collision (process risk).** A *second*
     "paper-trader core" HYBRID agent (pid 1958253) edited the same files
     simultaneously; its whole-file `git add` swept its
     `_host_pulse_line`/`host_guard.py` work into this pass's commit
     `155f857` *despite* a hunk-filtered `git apply --cached` that staged
     cleanly. **No work lost** (all committed + pushed) but attribution is
     imperfect — `155f857` bundles two features. Running multiple
     same-role agents on one shared monorepo tree is the root risk.
  - **POSITIVE.** Book/data trust intact: `/api/portfolio` reconciles,
    benchmark/freshness wiring intact, openclaw send healthy, continuous
    backtest loop progressing. The freeze is isolated to the Opus call
    (host load), not a blind feed / corrupt book / dark channel.

- **Concurrency / staging discipline.** Built a my-only patch by parsing
  `git diff` and dropping every hunk containing sibling tokens
  (`_host_pulse`/`TestHostPulseLine`/`**HOST**`), `git apply --cached`'d
  it (staged exactly 2 path-scoped files, `git diff --staged` verified
  zero sibling tokens). A concurrent same-role sibling's whole-file
  `git add` between stage and commit still bundled its work into
  `155f857`; chose NOT to `git reset` (rewriting in-flight shared history
  under a concurrent committer risks destroying the sibling's work — the
  destructive race is worse than an imperfect message). Surfaced
  transparently here and in the completion ping. AGENTS.md appended-only,
  single trailing hunk, committed separately (not counted as the feature).

*Review pass #27 (paper-trader core hybrid) appended 2026-05-18. Prior content above is unmodified.*

---

## Review pass #28 — paper-trader core hybrid (2026-05-18)

**bugs_fixed=0 · features_added=1 · user_findings=4**

- **Phase 1 — bugs_fixed=0 (mature).** Full read of the 7 core files
  (`runner.py`, `reporter.py`, `signals.py`, `strategy.py`, `market.py`,
  `store.py`) + `dashboard.py` surveyed by its 80+ `/api/*` route table
  (9.8k lines — endpoint-survey, not full read; stated honestly).
  Re-traced `decide()`'s claude / fallback / retry + pre-flight &
  mid-call host-saturation state machine, the singleton degrade/recheck
  path, `_mark_to_market` expired-option/`stale_mark`,
  `store.upsert_position` reactivate/blend, `_maybe_daily_close`
  half-day anchoring, the ~16 reporter conditional Discord lines. No
  real bug, race, or dead-code defect found — consistent with the 5+
  prior mature core passes (#23–#27 all bugs_fixed=0). Baseline green:
  423 focused-core+reporter+drawdown tests
  (`test_core_{market,store,signals,runner,strategy,reporter}` +
  `test_drawdown`) pass before the feature. The task template's
  "max_position / stop-loss" coverage does NOT apply — invariant #12
  (no hard limits; Opus full autonomy). No Phase-1 commit
  (guard-compliant — the guard explicitly permits this).

- **Phase 2 — feature (committed `aedda33`, pushed):
  `reporter._drawdown_line` — drawdown-from-peak in the hourly + daily.**
  The hourly/daily showed only `P/L $X (Y%)` **vs the $1000 start**,
  which silently conflates "never made money" with "made money then
  gave a chunk back". `compute_drawdown` existed (`/api/drawdown` + a
  `python -m paper_trader.analytics.drawdown` CLI) but had **no Discord
  surface** — the exact dashboard→Discord gap `_benchmark_line` /
  `_equity_integrity_line` / `_equity_freshness_line` each closed, one
  dimension over (vs-index → vs-own-peak, the two reference points a PM
  reads together). The new line consumes `compute_drawdown`'s OWN
  computed fields **verbatim** (the `_pos_pct_weight` precedent: pure
  formatting of a builder's already-computed numbers — invariant #10
  governs verdict/headline single-sourcing and `compute_drawdown` emits
  none, so suppression keys off the builder's OWN `at_high_water`
  boolean, never an invented threshold), feeds it the **byte-identical**
  store reads `drawdown_api` uses (`equity_curve(limit=2000)` +
  `open_positions()`, `starting_equity=_INITIAL_EQUITY`), is
  observational only / never gates / no caps (invariants #2/#12), pure
  store reads / NO network (the Discord-path discipline), and degrades
  to `""` on any fault (never an exception — the reporter additive
  contract). Renders depth (`-10.96% ($-110.39) from peak`),
  time-underwater via `_ago` (`4d in DD`), trough + the builder's own
  claw-back % (gated off when still at the lows), and the single worst
  open drag name. Suppressed at a fresh high (no hourly noise — the
  lying-green-light precedent). Wired immediately after `_benchmark_line`
  in **both** `send_hourly_summary` and `send_daily_close`. **13 new
  exact-assert tests** (`tests/test_core_reporter.py::TestDrawdownLine`
  — real-builder numbers `-5.00%/-10.00%/recovered 50%`, trough-gating,
  top-drag value, green-worst-name omits drag, `hours→_ago` backdated,
  non-dict/raises suppression, hourly+daily wiring, summary-still-sends
  on builder fault). 423 focused green, no regressions.
  ```bash
  python3 -m pytest \
    tests/test_core_reporter.py tests/test_drawdown.py -q   # green
  cd /home/zeph/trading-intelligence/paper-trader && \
    python3 -m paper_trader.analytics.drawdown               # CLI
  ```

- **Phase 3 — live findings (`:8090`, runner deciding but frozen; box
  under the review-swarm load this pass's own siblings contribute to).
  4 distinct + positive.**
  1. **IDLE_STORM / NO_DECISION host-saturation storm (HIGH,
     host-saturation, NOT a code bug — continuity, recalled
     `pt-no-decision-host-saturation`).** `/api/host-guard`: load1
     19.11 / 16 CPU, 7 concurrent Opus, swap 65%, "66% of the last 120
     decisions never reached Opus". `decisions` alternate
     `skipped claude call — host saturated: 6–11 concurrent Opus (>4)`
     and `claude returned no response (timeout/empty)`; last fill
     2026-05-17 09:38 (~30 h ago), 93 cycles since.
     `/api/runner-heartbeat` `IDLE_STORM`, `restart_recommended:true`.
     Architectural reality of the 15 GB box under the concurrent Opus
     review swarm + backtest committee; self-clears when the swarm ends.
     The pre-flight guard + mid-call re-probe are working correctly
     (distinct, honest reasons recorded). Not actionable in code.
  2. **Deep multi-day drawdown the operator was blind to in Discord —
     the gap THIS pass's feature closes, validated live.** Book peaked
     **$1007.59 (+0.76%)** 2026-05-14, now **-10.96% / -$110.39 from
     peak, 97.66 h (4 d) underwater, recovered only 3.2%, LITE -$64.90
     the anchor**. `/api/benchmark`: `best_alpha +0.31pp` at the 05-14
     peak, now **-9.71pp**. The hourly's "P/L -$103 vs $1000 start"
     hides that the strategy *made* money then bled and has been stuck
     4 days — a profoundly more actionable picture. The new
     `_drawdown_line` surfaces exactly this; confirmed reachable on the
     real book.
  3. **Capital paralysis (the #2 documented pathology, continuity).**
     `/api/capital-paralysis`: 97.95% deployed, $18.49 cash (2.0%), both
     open names underwater, LITE 59% of book, **-10.31% alpha cost over
     6 paralysis droughts**. Verdict reads `FREE` (technically can act
     with $18) but flags every structural problem. Continuity (pass #27
     #3); the Phase-2 DRAWDOWN line + pass-#27 per-name %/weight now make
     it visible in Discord.
  4. **Dashboard latency under host saturation (MEDIUM, symptom of #1,
     by-design).** `/api/host-guard` timed out at 6 s, returned correctly
     at 20 s; lightweight endpoints (`/api/portfolio`,
     `/api/benchmark`, `/api/mark-integrity`) stayed instant. Swap-thrash
     starving the SWR cold path, not a code defect (continuity, pass #27
     #2).
  - **POSITIVE — data/book/feed trust intact.**
    `/api/equity-integrity` `CLEAN` (806 points, cash never negative,
    min $2.61), `/api/mark-integrity` "All 2 marks live" (0 stale),
    `/api/feed-health` fresh (0.19 h, no split-brain, blind_streak 0),
    `continuous.log` only graceful external GDELT backoff
    (`attempt 1/3 — sleeping 20s`; no tracebacks / no engine CRASH),
    Discord send healthy. The freeze is isolated to the Opus call (host
    load), not a blind feed / corrupt book / dark channel.

  No Phase-3 fix folded in — every finding is host/ops continuity or
  the deliberate observe-don't-repair contract (invariants #2/#12); the
  only in-scope code change is the Phase-2 feature.

- **Concurrency / staging discipline.** Two concurrent same-role-ish
  HYBRID siblings live this pass (a second paper-trader-core agent pid
  1979382 + an ML/backtest agent pid 1979384; dirty `../digital-intern/`
  tree) — recalled `pt-concurrent-samerole-staging-race`. Never
  `git add -A`. Staged exactly the two path-scoped files
  (`paper_trader/reporter.py`, `tests/test_core_reporter.py`);
  `git diff --staged | grep` for sibling tokens (`digital-intern`,
  `decision_scorer`, `backtest.py`, `_host_pulse`, `persona_leaderboard`,
  `sortino`) returned **only the two `+++` headers of my own files** —
  zero sibling tokens — before commit `aedda33`. AGENTS.md re-read at the
  tail immediately before this append (still ended at pass #27, no
  sibling append yet), appended-only, committed separately (not counted
  as the feature).

*Review pass #28 (paper-trader core hybrid) appended 2026-05-18. Prior content above is unmodified.*

---

### 2026-05-18 review pass #29 (ML+backtest hybrid · linear-probe: feature-set vs model-class · live findings)

**bugs_fixed=0 · features_added=1 · user_findings=4**

- **Phase 1 — bugs_fixed=0 (mature).** Full read of the three named files
  (`paper_trader/ml/decision_scorer.py` 595 L, `paper_trader/backtest.py`
  2622 L, `run_continuous_backtests.py` 1976 L). Re-traced `train_scorer`
  dedup/split-before-scale/oversample, `_to_float` np.number/inf guards,
  the atomic pickle write, `_ml_decide`'s conviction gate +
  off-distribution abstention, `_compute_decision_outcomes`'s
  `score=`/`scorer=` first-match disambiguation, `_parse_gate_decision`,
  `_inject_and_train`'s 11-col INSERT + lock-retry, `_train_decision_scorer`
  separated OOS guards. No real bug, race, or dead-code defect — consistent
  with the 17+ prior mature ML/backtest passes (#5–#22) and the advisor's
  read; the heavy in-code comments record bugs already removed by those
  passes. **718 ML/backtest focused tests green at baseline**
  (`-k "ml or backtest or scorer or continuous or gate or baseline or
  calib or deploy or valid or horizon or regime or persona or skill or
  overfit or feature"`, 0 failed, 8m45s under review-swarm load). The
  task's "stop-loss exits / position-size not exceeded" cases are already
  locked in `test_backtest.py` / `test_integration_backtest.py`; the
  scorer null/known-vector cases in `test_decision_scorer.py`. No Phase-1
  commit (the commit guard explicitly permits this).

- **Feature shipped (Phase 2, `feat(ml):` — commit `9f8f54d`, pushed).**
  `paper_trader/ml/linear_probe.py` — the missing **discriminator** for the
  single most-repeated ML/backtest finding. `baseline_compare` =
  `MLP_NO_BETTER_THAN_TRIVIAL` has been on record ~10 passes but is
  *structurally ambiguous to a quant deciding what to do*: it cannot tell
  apart (1) **MLP-architecture failure** — the features carry a combinable
  linear signal the regularized/clamped net (or its sector-one-hot
  memorization `feature_importance` flags) wastes ⇒ *a linear head would
  beat the net*; from (2) **feature-set ceiling** — no model class helps ⇒
  *stop tuning the model*. `baseline_compare` (single-feature one-liners),
  `overfit_gap` (the MLP's own val/oos ratio), `feature_importance`
  (which input the MLP leans on) — none fit a *multi-feature model of a
  different class*. `linear_probe` does exactly that: a numpy closed-form
  L2 ridge on the **10 numeric features the MLP sees** (sector one-hot
  deliberately excluded — it is the documented memorization vector, not
  quant signal; sourced via `build_features()[:10]`, no re-listing → zero
  drift), **fit on the temporal-train slice** of
  `validation.split_outcomes_temporal` with standardization stats from
  **train only** (the "split before scale" no-leakage discipline), scored
  on the IDENTICAL temporal-OOS slice `baseline_compare`/`_train_decision_
  scorer` use. The deployed MLP is *data-advantaged* (trained in
  production on the full 5000-tail; the probe sees only this file's train
  slice) so **a probe win is conservative** — stated in the docstring so a
  skeptic reads it as a floor. Reuses `baseline_compare._skill` /
  `_aligned_pred_target` / `IC_MARGIN` / `MIN_PAIRS` / `MLP_IC_MIN` and
  `calibration._spearman` verbatim (single source of truth — the MLP
  rank-IC equals `baseline_compare`'s / `calibration --oos`'s by
  construction; a no-drift test locks it). Verdict ladder
  `INSUFFICIENT_DATA` / `LINEAR_PROBE_RECOVERS_SIGNAL` /
  `NO_COMBINABLE_SIGNAL` / `LINEAR_MATCHES_MLP`. Read-only — never trains
  the deployed model, never touches `decision_scorer.pkl` /
  `decision_outcomes.jsonl` / `_ml_decide` / `build_features` /
  `N_FEATURES` (same discipline as `baseline_compare`/`gate_audit`); CLI
  exit 2 on the two actionable verdicts. **20 known-answer tests**
  (`tests/test_linear_probe.py`): `_fit_ridge` recovers a noiseless
  `y=2x₀−3x₄` (rank-IC>0.99) where a single column cannot; the four-way
  ladder with Spearman fixed ±1/0 by construction; an end-to-end
  combinable `0.3·mom20+0.6·mom5` signal a noise-MLP can't model →
  RECOVERS; pure noise at the realistic n (n_oos≈320) →
  NO_COMBINABLE_SIGNAL (the `MLP_IC_MIN` floor is the small-sample
  buffer — locked); a **no-leakage lock** (signal only in OOS, train is
  noise ⇒ must NOT be RECOVERS); the SELL `−forward_return_5d` flip
  learned by the fitted probe; RIDGE_ALPHA robustness across 0.1–10;
  MLP-IC no-drift cross-check vs `baseline_compare`; never-raises.
  ```bash
  cd /home/zeph/trading-intelligence/paper-trader && python3 -m pytest \
    tests/test_linear_probe.py -q                       # 20 green
  cd /home/zeph/trading-intelligence/paper-trader && \
    python3 -m paper_trader.ml.linear_probe             # CLI, exit 2 actionable
  ```

- **Phase 3 — live findings (read-only diagnostics on the running loop;
  box under the concurrent review-swarm + backtest committee load this
  pass's own siblings contribute to). 4 distinct + positive.**
  1. **DECISIVE — the new feature resolves the 10-pass ambiguity live:
     `linear_probe` = `NO_COMBINABLE_SIGNAL` (HIGH, quant-grade).** On the
     live `decision_outcomes.jsonl` (n_oos=1981, n_train_fit=7928,
     deployed n=3799): an honest no-leakage ridge on ALL 10 numeric
     features reaches rank_ic **+0.071** — *not better than `ml_score`
     alone* (+0.072; gap −0.001). The MLP's near-zero OOS skill is a
     **feature-set ceiling, not a model-class bug**: the anti-overfit
     retune (#19) and the deploy-staleness redeploy (#21) cannot lift the
     conviction gate (invariant #5, active at deployed n≥500) because
     there is no combinable signal for *any* linear model to extract. The
     probe does beat the deployed MLP (+0.071 vs +0.051) — consistent
     with #21's stale `(64,32,16)` net — but by < `IC_MARGIN` and below
     the `MLP_IC_MIN` floor, so not independently actionable.
  2. **`calibration --oos` = MISCALIBRATED with a top-decile inversion
     (HIGH, corroborating).** spearman 0.051; decile-10 (highest
     predictions, mean_pred **+12.49**) realizes only **+0.99%** —
     *below* decile-9 (+1.81%) and decile-8 (+1.71%). The gate's ×1.3
     strong-tailwind arm upsizes exactly the bucket that realizes worst
     among the top half. Independent corroboration of finding 1 from the
     calibration axis.
  3. **The recent instrumentation chain is INERT in production
     (HIGH, continuity — recalled `pt-stale-manual-daemon` class).**
     `deploy_audit` = `DEPLOYED_STALE_CONFIG` (6/8 hyper-params drifted:
     `(64,32,16)≠(32,16)`, `alpha=1e-4≠1e-2`, `early_stopping=False≠True`,
     …); `gate_realized` = `GATE_CAPTURE_NOT_YET_POPULATED` (0 rows carry
     `gate_scorer_pred` — the loop predates commit `60b20d9`). The
     long-lived `run_continuous_backtests.py` retrains every cycle but
     imports the *pre-retune* module, so the gate sizes real backtest
     capital on the memorizing net AND none of #19–#22's instrumentation
     measures the live gate. Ops, not code — the source is correct;
     restart redeploys. Not a code bug.
  4. **`_llm_annotate_outcomes` fails EVERY cycle in the live loop
     (MEDIUM, ops/auth — dead training-weight feature).** `continuous.log`
     repeats `[continuous] LLM annotation failed: "Could not resolve
     authentication method…"` — the `anthropic` Python SDK path has no
     API key in the continuous process (unlike `_opus_annotate`, which
     uses the authenticated `claude` CLI). Effect: `llm_quality_label`
     never set, so `train_scorer`'s documented 3×-endorse/0.1×-condemn
     sample-weight multiplier silently never applies — every sample is
     1.0×. The code degrades correctly (caught, non-fatal, returns
     records unchanged) so this is an environment/auth gap, **not a code
     defect**; no fix folded in (the graceful-degrade is the right
     behavior; the fix is ops-side credential config).
  - **POSITIVE — data flow healthy.** `decision_outcomes.jsonl` ≈9.9k
    rows and growing, scorer pickle re-written across runs (n 3799→3997),
    continuous loop progressing (`gdelt_weekly 50/5055`), backtests
    recorded. The system is alive and accumulating — it is running stale
    code, not broken. (Aside: `backtest.db`/`data/` are symlinked to a
    `/media` external drive; under the review-swarm load read-only
    diagnostics against them queued for minutes — a blind-spot for
    go/no-go reads during exactly the high-load windows.)

  Findings 1+2 are the quant payoff (the shipped feature + an independent
  corroboration that the gate's ceiling is the inputs); 3 is the dominant
  continuity reality; 4 is a real silently-dead training feature. No
  Phase-3 fix folded in (the only in-scope code change is the Phase-2
  feature; 3 and 4 are ops/auth, not code).

- **Concurrency / staging discipline.** Heavy concurrent siblings on the
  shared monorepo tree (core pass #28 appended mid-pass: AGENTS.md grew
  6232→6364 L; `../digital-intern/` dirty). Never `git add -A`. Staged
  exactly the two path-scoped new files
  (`paper_trader/ml/linear_probe.py`, `tests/test_linear_probe.py`);
  `git diff --staged --name-only` filtered to those two returned CLEAN
  (zero sibling tokens) before commit `9f8f54d`. AGENTS.md re-read at the
  tail immediately before this append, appended-only, committed
  separately (not counted as the feature).

*Review pass #29 (ML+backtest hybrid) appended 2026-05-18. Prior content above is unmodified.*

---

### Review pass #28 — AMENDMENT (paper-trader core hybrid, 2026-05-18)

**Revised tally: bugs_fixed=1 · features_added=1 · user_findings=4**
(supersedes the `bugs_fixed=0` headline of pass #28 above — appended after a
sibling's pass #29; prior content is unmodified, append-only discipline
preserved.)

The full 2117-test suite (run to completion in 16m08s under the host-load
storm — **2116 passed, 1 failed**) surfaced one real, pre-existing,
in-scope defect the focused-core baseline does not cover:
`tests/test_swr_prewarm_coverage.py::test_every_swr_cached_endpoint_is_prewarmed`
**failed on master** — `scorer_attribution_api` carries
`@swr_cached("scorer-attribution", 60.0)` but the tuple
`("scorer-attribution", scorer_attribution_api)` was never added to
`dashboard._swr_prewarm.targets`. Consequence (the exact freeze-triage
blind spot that regression test exists to lock): the `/api/scorer-attribution`
panel cold-stalls with `{"warming": true}` on the first poll after **every**
restart, while every other `@swr_cached` panel is warmed at boot — so a
trader opening the scorer-attribution panel right after a restart (exactly
when triaging "why is the engine acting strange?") gets a dead placeholder.
Not introduced by this pass (the Phase-2 feature touched only
`reporter.py`); a prior commit added the decorator without the matching
prewarm target — the precise contract-rot the test guards against.

- **Fix (committed `99053ff`, pushed):** one missing target tuple +
  freeze-triage comment added to `_swr_prewarm.targets`, restoring the
  `prewarm == @swr_cached` set invariant. All 3
  `test_swr_prewarm_coverage.py` tests green
  (`test_prewarm_handlers_resolve_to_callables` independently confirms the
  `scorer_attribution_api` symbol resolves to a callable). 50
  SWR/dashboard-helper tests green, no regression. Surgical: 5 insertions,
  `dashboard.py` only — staged path-scoped, `git diff --staged` verified
  zero sibling tokens (only the `+++` header) before commit.
- **Process note:** the failure was only visible because the full suite was
  run to completion rather than substituting the focused-core gate — the
  honest-verification payoff. `pt-test-suite-timing` holds (16m under load;
  the focused gate is fast but does NOT cover `test_swr_prewarm_coverage`).
  The two prior pass-#28 commits (`aedda33` feature, `d83f756` AGENTS) and
  this fix (`99053ff`) are three separate path-scoped commits.

*Review pass #28 AMENDMENT (paper-trader core hybrid) appended 2026-05-18. Prior content above is unmodified.*

---

## Review pass #30 — paper-trader core hybrid (2026-05-18)

**bugs_fixed=1 · features_added=1 · user_findings=5**

- **Phase 1 — bugs_fixed=1 (commit `3e24437`, pushed).**
  `_swr_prewarm.targets` was missing **three** `@swr_cached` endpoints
  added in recent commits (`scorer-opportunities` from `20295e8`,
  `scorer-portfolio-attribution` from `6018347`, `trade-attribution`
  from `2a28eea`). Their panels cold-stalled with `{"warming": true}`
  on the first poll after every restart — the exact freeze-triage
  blind spot `tests/test_swr_prewarm_coverage.py` exists to lock
  against (the same class as commit `542446d` 6-endpoint batch fix
  and `99053ff` for `scorer-attribution`). Caught by running the
  prewarm coverage test, which failed listing all three. Surgical
  9-line addition with a 6-line freeze-triage comment naming the
  commits. Path-scoped: only `paper_trader/dashboard.py` staged
  (`git diff --staged --stat` verified before commit; sibling-token
  grep clean). After fix: 469 focused tests green
  (`test_core_{market,signals,store,strategy,runner,reporter,
  dashboard_helpers}` + `test_swr_prewarm_coverage` +
  `test_dashboard_swr`).
  ```bash
  python3 -m pytest tests/test_swr_prewarm_coverage.py -q   # 3 green
  ```

- **Phase 2 — feature (committed `4a576da`, pushed):
  `reporter._position_attention_line` — surface NEGLECTED/STALE held
  lots in Discord.**
  `/api/position-attention` (commit `f703cb2`) answers a question NO
  other hourly/daily block did: **which specific open lots has Opus
  stopped examining?** When a NO_DECISION storm drags on (the
  documented #1 pathology — `pt-no-decision-host-saturation`
  memory), the live trader silently defaults to holding every open
  lot while those lots are no longer being **evaluated**. Every
  other Discord block on the hourly is aggregate (decision-health
  rate, capital-paralysis, host-guard) or per-trade (asymmetry,
  autopsy). The operator who lives in Discord had no per-position
  attention surface — the exact dashboard→Discord gap
  `_host_pulse_line` / `_capital_pulse_line` /
  `_singleton_lock_line` each closed, one dimension over
  (aggregate-vs-host → per-held-position).

  Composes `build_position_attention` **verbatim** (single source of
  truth, AGENTS.md invariant #10 — the verdict / note are the
  builder's, never re-derived, so this Discord line and
  `/api/position-attention` can never tell different stories) and
  feeds it the EXACT same store reads the endpoint does
  (`open_positions()` + `recent_decisions(limit=3000)`). **Pure
  store reads only — NO network** (the Discord-path discipline; adds
  zero latency). Observational only, never gates, adds no caps
  (invariants #2/#12 — the `_host_pulse_line` precedent). Renders up
  to 3 worst-first per-position lines (`ticker`, `verdict`, `hours
  since last look`) so the operator sees the exact tickers to triage,
  not just an aggregate count. Wired immediately after
  `_capital_pulse_line` in both `send_hourly_summary` and
  `send_daily_close`.

  **Suppression** — surface ONLY when the model has stopped looking:
  `NEGLECTED_BOOK` (>=1 lot no Opus look in >24h) → ALWAYS surfaced;
  `STALE_BOOK` (>=1 lot last seen >6h ago) → surfaced; `OK` /
  `INSUFFICIENT_DATA` → silent (no noise; the
  `_hold_discipline_line` NO_DATA / `_heartbeat_line` HEALTHY
  suppression precedent — the summary must never become its own
  lying green light). Failure contract mirrors the rest of
  `reporter`: any builder/store fault degrades to `""` ("no
  attention line this report"), **never** an exception ("no Discord
  summary this report").

  **13 exact-assert tests** (`tests/test_core_reporter.py::
  TestPositionAttentionLine`): OK/INSUFFICIENT_DATA suppression;
  NEGLECTED/STALE rendering with ticker+verdict+hours;
  `hours_since=None` → `"no Opus look on record"`; 3-line per-position
  cap (no flooding); builder-fault / non-dict / empty-note
  degrade-to-empty; hourly + daily wiring (surfaces when neglected,
  silent when ok); summary-still-ships on builder fault (additive
  contract).
  ```bash
  python3 -m pytest tests/test_core_reporter.py -k "PositionAttention" -q
  # 13 passed in 0.91s
  python3 -m pytest tests/test_core_reporter.py tests/test_core_*  \
    tests/test_swr_prewarm_coverage.py tests/test_dashboard_swr.py -q
  # 469 passed in 6.48s
  ```

- **Phase 3 — live findings (`:8090`, runner alive but on stale code;
  box under the concurrent review-swarm load this pass's own siblings
  contribute to). 5 distinct + positive.**

  1. **UNSUPERVISED_STALE — trader running on stale code as an
     orphan (HIGH, ops/continuity, recalled
     `pt-stale-manual-daemon` + `pt-systemd-vs-manual-restart-spam`
     memories).** `/api/supervision` `UNSUPERVISED_STALE`, boot SHA
     `cef83f2` vs head `4a576da` (behind 2 — this pass's own
     `3e24437` + `4a576da`), systemd unit `disabled`/`inactive`,
     PPID=1269 (manual launch). The git-watcher's `os._exit(0)`
     would leave the trader DOWN permanently. `should_restart` CLI
     agrees: `RESTART RECOMMENDED`. Not actionable in code — the
     operator's existing observation is that this trader runs as a
     manual long-lived process, not under systemd. Remediation is
     `systemctl --user enable --now paper-trader` OR accept the
     manual-daemon arrangement and tolerate stale code between
     restarts.

  2. **Host saturation symptom — 100% of last decision didn't reach
     Opus (MEDIUM, host, continuity — recalled
     `pt-no-decision-host-saturation`).** `/api/host-guard`
     `STARVED`: load1=11.67/16 CPU, swap_used=64.9%, 4 concurrent
     Opus. `/api/empty-claude-rate` `INSUFFICIENT_DATA` (n=1).
     Architectural reality of the 16-CPU box under the concurrent
     review-swarm + backtest committee load this pass's own
     siblings contribute to; self-clears when the swarm ends. Not
     a code bug.

  3. **Empty book — $1000 cash, 0 positions, 0 decisions, 0 trades
     (LOW).** `/api/portfolio` `cash=$1000.0 total=$1000.0`,
     `/api/decision-health` `NO_DATA`,
     `/api/runner-heartbeat` `NO_DATA` (`secs_since_last_decision=
     None`). Live trader looks freshly reset (between this pass's
     reads); legitimate state, not a bug. My new
     `_position_attention_line` correctly returns `""` for this
     state (the `INSUFFICIENT_DATA` suppression branch — validated
     live against the empty book).

  4. **Runner on stale code still queries delisted GOOGU/METAU
     (LOW, symptom of #1).** `logs/runner.log` shows repeating
     yfinance HTTP 404 / `possibly delisted; no price data found`
     for `GOOGU` / `METAU`. The current `strategy.py` WATCHLIST on
     HEAD no longer includes them (removed `2026-05-18` per the
     code comment) but the running process still does — direct
     consequence of finding 1 (stale code). Self-resolves on
     restart.

  5. **POSITIVE — data/book/feed trust intact, dashboard healthy,
     ALL my new code paths validated live.**
     `/api/feed-health` HEALTHY, 0.1h newest, 8284 live articles
     24h, no split-brain; `/api/equity-freshness` curve current and
     agrees with live book; `/api/mark-integrity` `NO_DATA` (no
     open positions to mark — clean); `/api/position-attention`
     returns my new endpoint correctly with the empty-book
     `INSUFFICIENT_DATA` verdict; `/api/scorer-portfolio-
     attribution` returns sensibly (`is_trained:true`,
     `n_train:35`, `n_positions:0`); all queried endpoints respond
     within 5–8s under the saturated-host conditions; Discord send
     healthy (no consecutive failures). No tracebacks in runner.log
     besides the yfinance 404s of #4.

  No Phase-3 fix folded in — findings 1, 2, 4 are operational
  continuity (ops decision, not code), and 3 is legitimate fresh
  state. Finding 5 confirms the deployed product works.

- **Concurrency / staging discipline.** Multiple concurrent siblings
  active (pid 2291372 me, 2291374 ML/backtest, 2291377 feature-dev;
  the feature-dev sibling committed `/api/news-velocity` to
  `dashboard.py` mid-pass — uncommitted in my working tree when I
  read it, then auto-commit daemon picked it up between my reads).
  Recalled `pt-concurrent-samerole-staging-race`. Never `git add -A`.
  Staged exactly the path-scoped files (`paper_trader/dashboard.py`
  for the Phase-1 fix; `paper_trader/reporter.py` +
  `tests/test_core_reporter.py` for the Phase-2 feature);
  `git diff --staged` filtered for sibling tokens
  (`news-velocity`, `news_velocity_api`, `_stock_tickers_from_positions`,
  `digital-intern`, `decision_scorer`, `persona`) returned **zero
  hits** before each commit. AGENTS.md re-read at the tail
  immediately before this append (still ended at pass #28 amendment),
  appended-only, committed separately (not counted as the feature or
  fix). Three path-scoped commits: `3e24437` (fix), `4a576da`
  (feature), and the AGENTS.md commit to follow.

*Review pass #30 (paper-trader core hybrid) appended 2026-05-18. Prior content above is unmodified.*

### 2026-05-18 review pass #31 (ML+backtest hybrid · gate-abstention diagnostic · live findings)

- **Phase 1 — no new bugs (bugs_fixed = 0; no Phase-1 commit).** Full
  re-trace of `decision_scorer.py`, `backtest.py`,
  `run_continuous_backtests.py` plus coupled `validation.py` /
  `calibration.py` / `gate_realized.py`: the `score=` vs `scorer=` regex
  first-match disambiguation, the `(ticker, sim_date, action)` dedup key
  in `train_scorer`, the universal SELL `-forward_return_5d` sign-flip
  (train ↔ inference ↔ calibration ↔ gate ↔ `_oos_rank_metrics`), the
  5-trading-day forward-window guard, the `_parse_gate_decision` regex +
  `(off-dist` substring check, the off-distribution gate abstention path
  in `_ml_decide` (the `_pwm` callable probe + `not scorer_off_dist`
  guard), the `_Dummy` fallback's 11-keyword `predict(**kw)` contract,
  `_to_float`'s `np.number` (not `np.generic`) branch with inf/NaN/bool
  rejection, the `_inject_and_train` lock-retry `for…else` semantics,
  the separately-guarded OOS blocks in `_train_decision_scorer`, every
  module-global lock — all re-verified correct and exact-value
  test-locked. Consistent with the documented 12+ prior no-new-bug ML/
  backtest passes — not a fabricated fix. ML/backtest subset 387/387
  green before the feature.

- **Feature shipped: off-distribution gate-abstention diagnostic
  (`paper_trader/ml/gate_abstention.py`, `tests/test_gate_abstention.py`,
  25 exact-value verdict/boundary/threshold locks).** The gap it fills:
  commit `84d8234` added the off-distribution gate abstention to
  `_ml_decide` (when the scorer's clamped output exceeds `±PRED_CLAMP_PCT`
  the gate **leaves conviction untouched** — the multiplier arm is
  skipped entirely). Commit `60b20d9` then made that decision durable
  in `decision_outcomes.jsonl` via the `gate_off_dist` boolean.
  **Nothing reports how often the guard actually fires.**
  `gate_realized.py` reads the field (to route abstained rows to a
  separate bucket) but its verdict grades only ACTED rows; the *rate*
  of abstention itself, the per-arm distribution it abstained from, and
  whether it is rising or falling over time were unread. That is the
  ledger-wired-but-unread pattern of pass #15 / pass #20
  (`baseline_trend` for the baseline ledger), applied to a per-row
  field instead of a per-cycle ledger column.

  This matters operationally. Two failure modes look identical from a
  distance:

  * **GUARD_INACTIVE** — rate < 0.5%. Either the model is always
    in-distribution (good — but worth knowing) OR the abstention
    threshold is too lax to ever fire. The guard exists to catch the
    −89→+32 same-LITE-vector extrapolation `decision_scorer.py`
    documents; if it never fires, that protection is dead code.
  * **GUARD_RAMPANT** — rate ≥ 15%. The model is regularly emitting
    clamped ±50 outputs the gate then refuses to act on — so the gate
    is mostly *neutral* despite being "active". The 1.30/0.60
    multiplier spread `gate_audit` / `gate_realized` grade is being
    applied to a shrinking fraction of decisions.

  Imports `gate_arm` from `gate_audit` (single source of truth — the
  "would-have-been arm" distribution must match `_ml_decide` to the
  bit; the `gate_realized`-imports-`gate_arm` precedent) and
  `split_outcomes_temporal` from `validation` for the OOS slice (the
  same temporal split every sibling OOS tool uses). Same operational
  discipline as `paper_trader/ml/calibration.py`: read-only, no train,
  no pickle / `build_features` / `N_FEATURES` touch, no trade path —
  safe to run against the live unattended loop. Never raises. CLI
  exits 2 on `GUARD_RAMPANT` (operator-actionable: gate is mostly
  neutral), 0 on every other verdict — mirrors the sibling diagnostics'
  cron-branch convention.

  ```bash
  cd /home/zeph/paper-trader && python3 -m paper_trader.ml.gate_abstention
  cd /home/zeph/paper-trader && python3 -m paper_trader.ml.gate_abstention --all
  cd /home/zeph/paper-trader && python3 -m pytest tests/test_gate_abstention.py -v
  ```

  Trend axis (recent half vs older half of captured rows by `sim_date`):
  `IMPROVING` / `DEGRADING` / `STABLE` / `UNKNOWN`. Independent of the
  verdict axis (the `baseline_trend` precedent of orthogonal axes) so a
  ledger row reading `GUARD_INACTIVE` + `trend=DEGRADING` is the
  decisive "still mostly dormant but starting to fire more" signal —
  exactly what a quant would want to spot the moment the model begins
  drifting OOD.

  > **Quant finding (2026-05-18, live).** Run on the live
  > `decision_outcomes.jsonl` (5165 captured rows, 7413 total):
  >
  > | Slice | n_captured | n_abstained | rate | verdict |
  > |-------|-----------|-------------|------|---------|
  > | full corpus | 5165 | 4 | **0.08%** | `GUARD_INACTIVE` |
  > | OOS (last 20%) | 1089 | 4 | **0.37%** | `GUARD_INACTIVE` |
  >
  > **All 4 abstentions cluster at the extreme clamp arms** — 2 at
  > `strong_headwind` (would-have-been ×0.60), 2 at `strong_tailwind`
  > (×1.30) — which is the expected shape (the off-dist guard fires
  > exclusively when the unbounded MLP head emits a clamped ±50). All 4
  > are on **leveraged ETFs**: URTY (×2), SOXL, FAS — exactly the names
  > `decision_scorer.py`'s clamp comment cites as the documented
  > extrapolation case. The guard infrastructure is mostly dormant: the
  > ±PRED_CLAMP_PCT extrapolation protection it was added for is
  > engaging on **0.08% of decisions**. Trend `STABLE` (older 0.00% →
  > recent 0.15%, within ±2pp band). **Reported observation, not a code
  > change** — tuning the abstention threshold is a model-dynamics
  > change out of surgical scope (CLAUDE.md §6). The new diagnostic
  > makes this question durably answerable for the first time, and is
  > the operator's signal for if/when the model starts drifting OOD.

- **Phase 3 live findings (other arbiters, reconfirmed).** Reading every
  existing trender against the live ledgers:
  * `skill_trend` = **`DIRECTIONAL_BUT_HIGH_ERROR`** — recent median
    `oos_rmse=12.71` ≥ fresh mean-predictor baseline `9.65` (worse than
    a constant predictor), but median `oos_ic=+0.07 > IC_MIN=0.05` so
    the gate's sign decision carries marginal value. Trend
    **DEGRADING** (older 10.80 → recent 12.71), `gate_active=1.0` across
    all 22 ledger cycles.
  * `overfit_gap` = **`SEVERE_OVERFIT`** with **trend=DEGRADING**
    (recent median oos/val ratio **1.45 ≥ 1.40**, older 1.19, overall
    1.21). The new (32,16)+L2+early_stop config (`5a0af2d`) is deployed
    (`deploy_stale=False` on the last 4 cycles) but the gap has
    **widened**, not closed, on the most recent draws — a sharper
    statement than prior passes' `MILD_OVERFIT/STABLE`.
  * `gate_realized` = **`GATE_INEFFECTIVE`** — `strong_tailwind +3.50%`
    vs `strong_headwind +4.27%` on the OOS slice (spread **−0.77pp**,
    inside the ±1pp band — so the verdict is "ineffective" not
    "harmful", but the *direction is inverted*: the gate's ×0.60 arm
    (sizing DOWN bearish-predicted names) realized the BEST returns at
    exactly the moments the gate thought the smallest position was
    right). `arm_monotone_fraction=0.5` — only half the arm-mean
    sequence is non-decreasing.
  * `baseline_trend` = `INSUFFICIENT_DATA` (only 4 baseline-ledger
    cycles vs `MIN_CYCLES=5`) — populates as more cycles run; the most
    recent cycle 4 in the raw ledger reads `MLP_ADDS_SKILL`
    (mlp_rank_ic=+0.19, ic_gap=+0.11), in contrast to prior passes'
    documented `MLP_WORSE_THAN_TRIVIAL` verdict. **Sample is small**
    (n=4): a regime shift is plausible but unconfirmed.
  * `calibration --oos` = **`MISCALIBRATED`** on the live pickle on
    both full-corpus and OOS slice (spearman 0.05 / -0.05; OOS
    decile-realized column noise — d1 mean_pred −11.65 realized +3.17
    vs d10 mean_pred +5.99 realized +0.04: tails over-predict and
    direction is inverted on the bearish tail).

  Joint reading: the gate's economic skill is marginal-at-best on the
  same regime where the model's `gate_off_dist` guard is barely firing
  (0.08%). The gate is acting confidently on a signal that does not
  pay; the guard that would otherwise abstain on extrapolation is
  protecting fewer than 1-in-1000 decisions. **All reported
  observations, not model changes** — altering thresholds / abstention
  rules / the gate is a training-dynamics change out of surgical scope
  (CLAUDE.md §6, gate is invariant #5).

- **Live health.** `backtest.db` external/symlinked (`/media/...`)
  responding (read times >5s under the running-loop write contention,
  per the documented pre-existing scalability characteristic); ledgers
  fresh — `scorer_skill_log.jsonl` 22 rows on cycles 1-7 of the *new*
  loop process (after a redeploy: `deploy_stale=False` on the last 4
  cycles, the regularized config is live), `baseline_skill_log.jsonl` 4
  rows, `decision_outcomes.jsonl` 7413 rows / 3.2 MB. `continuous.log`
  fresh, mid-cycle, no Python tracebacks — only expected `[gdelt]
  permanent` short-circuits (commit `8899c16` working as designed) and
  yfinance `possibly delisted` lines for pre-IPO windows on names like
  LITE/SNOW/COIN/PLTR (handled gracefully via `prices[t]={}`).

- **Run the ML/backtest suite:** `cd /home/zeph/trading-intelligence/paper-trader
  && python3 -m pytest tests/ -k "ml or backtest or scorer or calibration
  or continuous or horizon or gate_abstention" -q` (412/412 green;
  +25 over the prior 387 baseline from the new `gate_abstention.py`
  tests).

*Review pass #31 (ML+backtest hybrid) appended 2026-05-18. Prior content above is unmodified.*

---

## 2026-05-19 feature-dev pass (Agent 4) — shadow-vs-claude + earnings-distribution

Two new read-only dashboard endpoints (observational, never gate Opus —
invariants #2/#12, sibling-precedent: `/api/earnings-shock`, `/api/disagreement`).

### `/api/shadow-vs-claude` — snapshot of deterministic shadow rec vs last Claude decision

The live 2026-05-19 shape exposed a gap on the operations surface:
`/api/empty-claude-rate` and `/api/host-guard` correctly surface
HOST_SATURATED (5 concurrent Opus, 73% NO_DECISION over last 11 cycles) but
say nothing about *what the bot would have done if Claude had come back*.
This endpoint joins `/api/suggestions` (the deterministic `_classify_action`
co-pilot rules engine) with the most recent row in the `decisions` table
and emits a verdict:

| verdict | meaning |
|---|---|
| `MISSED_OPPORTUNITY` | last Claude was NO_DECISION while shadow has a strong (conviction ≥ 0.70) directional BUY/ADD/TRIM/EXIT. The operationally meaningful case — operator may want to act manually. |
| `DROUGHT_OK` | Claude NO_DECISION but shadow is quiet (HOLD/WATCH only); nothing to act on. |
| `ALIGNED` | Claude and shadow agree on the same directional call on the same name (BUY≡ADD, SELL≡TRIM≡EXIT). |
| `DIVERGENT` | both produced directional calls — they disagree. |
| `CLAUDE_HOLDS` | Claude HOLD while shadow flags a directional rec. |
| `NO_CLAUDE_DATA` / `NO_SHADOW_DATA` | degraded inputs. |

**Snapshot-only by construction** (advisor framing): the two inputs are
produced from different points in time (last decision can be minutes-to-hours
old; the suggestion list is current), so the builder deliberately does NOT
compute "agreement %" over a historical window — that comparison would be
incoherent (signals at decision time ≠ signals now). For the aggregate-over-
time view of decisions see `/api/decision-health` (also SSOT, no re-derive).

Pure builder in `paper_trader/analytics/shadow_vs_claude.py`. Endpoint is a
thin wrapper that reuses `suggestions_api()` verbatim (the
`/api/funded-suggestions` precedent) — single source of truth for the
deterministic shadow engine; never re-derives.

### `/api/earnings-distribution` — empirical observed-quantile complement to `/api/earnings-shock`

`/api/earnings-shock` assumes a Gaussian shock model and reports a single 1σ
figure per held imminent print (e.g. for the live 2026-05-19 NVDA 44%
position: `σ ±1.8% → ±$8.17 (book ±0.82%); 3σ down stress $-24.51 (-2.45%)`).
Earnings reactions are fat-tailed; the Gaussian framing hides the historical
worst case. This endpoint surfaces the **empirical observed distribution**
per held imminent event:

* `observed_quartiles`: `{worst, q1, median, q3, best}` of historical
  1-day post-earnings reactions (in % terms, signed)
* `dollar_quartiles`: same shape × current position value
* `book_pct_quartiles`: same shape, normalised by total_value
* `downside_worst_dollar` / `downside_worst_book_pct`: loss-side worst case
  clipped at zero (a string of all-positive observations yields 0 here, not
  a manufactured negative — operator's clearest "what's the worst this
  name has cost on a print?" read)
* `row_verdict`: ELEVATED/MODERATE/LOW on |worst-observed book impact|,
  thresholds locked to `earnings_shock` (5% / 2%) by
  `test_thresholds_consistent_with_earnings_shock`

**Naming discipline** (per advisor): fields are `q1`/`median`/`q3` (observed
quartiles), **not** `p25`/`p50`/`p75` (which would imply distributional
percentile inference n=3–8 historical prints can't support). The
`test_quartile_keys_use_observational_naming` Flask test locks this — adding
a `p25` etc. field will fail the suite.

Composes `build_event_calendar` for the held set verbatim (SSOT, #10) so
this endpoint, `/api/earnings-shock` and `/api/event-calendar` can never
disagree on what counts as held-imminent. Reuses `_earnings_history_for`
as the I/O seam (same yfinance call shape, same 5-min SWR TTL on both).

Pure builder in `paper_trader/analytics/earnings_distribution.py`.
Mirrors the `earnings_shock` state ladder (`NO_DATA` / `NO_EVENTS` /
`OK`, per-row `INSUFFICIENT_HISTORY` at n<3) and `_z`/`_position_value`
shape so the two builders are byte-symmetric on the same shape of inputs.

### Tests

Pure-function tests:
- `tests/test_shadow_vs_claude.py` (29 tests) — verdict ladder, action
  classification, BUY≡ADD / SELL≡TRIM equivalence, naive-UTC timestamp
  handling, never-raises contract.
- `tests/test_earnings_distribution.py` (20 tests) — quartile interpolation
  (numpy-default linear, NIST type 7), INSUFFICIENT_HISTORY at n<3,
  downside-zero on all-positive history, threshold-consistency with
  `earnings_shock`, builder never raises on garbage inputs.

Flask test-client tests (no :8090 bind, no live DB):
- `tests/test_shadow_vs_claude_endpoint.py` (7 tests) — MISSED_OPPORTUNITY
  on the live 2026-05-19 shape, ALIGNED on BUY/ADD equivalence,
  suggestions_error pass-through, CORS for cross-fetch, degrade-to-error-
  body contract.
- `tests/test_earnings_distribution_endpoint.py` (7 tests) — payload
  shape, **quartile-key naming lock** (no `p25`/`p50`/`p75` regression),
  dollar = pos_value × pct math, NO_DATA / NO_EVENTS state ladder, CORS,
  degrade-to-error-body contract.

Full new suite: **66/66 green**. Adjacent endpoint suites
(`test_baseline_compare_endpoint`, `test_capital_paralysis_swr`) remain
green — no neighbor regressions.

*Feature-dev pass appended 2026-05-19. Prior content above is unmodified.*

## Review pass #31 — paper-trader core hybrid (2026-05-19)

Single HYBRID agent on the core surface (`runner.py`, `reporter.py`,
`signals.py`, `strategy.py`, `dashboard.py`, `market.py`, `store.py`).
Two real-bug fix commits + one trader-useful feature.

### Phase 1 fixes (commit 2db36ef)

Three regressions where a single malformed input crashed the live
trader's `decide()` cycle (no decision row, no equity point logged)
instead of recording a clean BLOCKED:

1. **`strategy._execute(BUY_CALL / BUY_PUT / SELL_CALL / SELL_PUT)`** —
   an unguarded `float(strike)` raised `ValueError` when Claude emitted
   `strike="ATM"` / `"ITM"` (a description rather than a number).
   Strike is now coerced under `try` / `except` once, reused (no second
   `float()` later that could re-raise); a non-numeric strike returns
   `BLOCKED` with the offending value in the detail.
2. **`strategy._enforce_risk_pre_trade`** — same unguarded `float()` on
   a non-numeric `qty` (`"all"` / `"half"`). Defensive: `_execute`
   already coerces before calling this helper, but a direct call (or
   test, or future refactor) would otherwise abort `decide()` with an
   uncaught `ValueError`. Returns `(False, "qty not numeric: …")` now.
3. **`signals._maybe_warn_stale`** — when one stale-feed sibling's
   `first_seen` was non-`None` but unparseable, `_age_hours` returned
   `None` and the `{age:.1f}h` format raised `TypeError` — the warn
   diagnostics line could take down `decide()`. The composer now
   silently skips a sibling with no usable age while still emitting
   the warning for the chosen feed.

4 new tests cover each path:
- `tests/test_core_strategy.py::TestExecuteBuyCall::test_buy_call_blocked_on_non_numeric_strike`
- `tests/test_core_strategy.py::TestExecuteBuyCall::test_sell_call_blocked_on_non_numeric_strike`
- `tests/test_core_strategy.py::TestEnforceRiskPreTrade::test_non_numeric_qty_blocks_cleanly_not_crashes`
- `tests/test_core_signals.py::TestFeedStatusAndWarn::test_unparseable_other_timestamp_does_not_crash`

### Phase 2 feature — trade-alert immediate book impact (commit c181e7d)

`reporter.send_trade_alert(trade, snapshot=None, store=None)` now
appends a `post: …` line summarising what the fill just did to the
book, the trader's #1 follow-up question after every alert:

```
**TRADE** `BUY` `NVDA`
qty `2.0` @ `$222.35` = `$444.70`
_Pre-earnings asymmetric setup; ~44% conviction sizing_
post: NVDA now 44.5% of book · cash $555.30
```

```
**TRADE** `SELL` `MU`
qty `5` @ `$120.00` = `$600.00`
_taking profits_
post: realized $+100.00 (+20.0%) · held 2.3d · cash $1100.00
```

- **BUY**: post-trade lot weight (per option leg, not aggregated by
  ticker) + remaining cash.
- **SELL (full close)**: realized $ P/L and pct from the single source
  of truth (`build_round_trips`), hold duration, cash. Cannot drift
  from the daily-close round-trip line.
- **SELL (partial close)**: never invents a P/L figure — falls back to
  `partial — NVDA still X% of book · cash $Y`.
- **Backwards-compatible**: callers passing only `trade` get the
  byte-compatible body. `runner._cycle()` now passes the post-trade
  snapshot from `strategy.decide()` + the store.

Failure contract mirrors the rest of `reporter`: any
snapshot/store/builder fault degrades to `""` ("no impact line on this
alert"), never an exception ("no trade alert this fill"). A missing
snapshot or non-positive `total_value` returns `""` so a flat / empty
book never emits a misleading `0.0% of book` token.

12 new tests cover both action paths, the option-lot label, partial
close fallback, snapshot=None backward-compat, and the
`_hold_str_from_days` bucket boundaries (`tests/test_core_reporter.py
::TestTradeAlertImpactLine`, `tests/test_core_reporter.py
::TestSendTradeAlert::test_no_impact_line_when_snapshot_missing`).

### Phase 3 live-validation findings

Probed the running system at `:8090` while the runner was on commit
`c181e7d` (git-watcher restarted onto the new code mid-session):

- **Heartbeat HEALTHY**, last decision 5m ago — within the 60m
  market-closed cadence; `restart_recommended=false`.
- **`decision_efficacy` DEGRADED** at 79% NO_DECISION across last
  14 cycles, traceable to the documented #1 pathology: host
  saturation (5 concurrent Opus, mem 6.3GB available, swap 70%). The
  pre-flight `host_saturated` guard correctly records distinct
  `"skipped claude call — host saturated"` reasons — these stay out
  of the "claude returned no response" empty-timeout bucket
  (`/api/empty-claude-rate` is honest).
- **Equity FRESH**: live $1000.00 (portfolio table) agrees with the
  recorded equity-curve point (5m old); benchmark/drawdown/Sharpe
  headlines are trustworthy.
- **NVDA position**: 2 sh @ $222.35 = $444.70 (44.5% of book),
  cash $555.30, total exactly $1000.00 (Opus's pre-earnings setup).
- **Position attention**: NVDA `FRESH`, 0.5h since last Opus look.
- **Reports flowing**: `runner_state.json` shows the hourly was sent
  within the last minute and yesterday's daily-close marker is set.
- **Singleton lock working**: runner.log shows several
  `another paper trader is already running (pid=…); refusing to
  start a second trader` rejects — the singleton fcntl guard is
  doing its job under systemd-Restart=always pressure (the known
  benign spam — `pt-systemd-vs-manual-restart-spam` memory note).
- **Live verification of the new alert path**: ran
  `send_trade_alert` against the real BUY NVDA trade with a
  hand-constructed post-trade snapshot — body included
  `post: NVDA now 44.5% of book · cash $555.30` exactly as designed.

#### Unrelated findings (not in this commit)

- **2 pre-existing test failures**:
  `tests/test_earnings_distribution.py::test_downside_worst_dollar_is_negative_when_worst_is_negative`
  and
  `tests/test_swr_prewarm_coverage.py::test_every_swr_cached_endpoint_is_prewarmed`.
  Both reference a sibling agent's incomplete uncommitted work
  (`paper_trader/analytics/earnings_distribution.py` is untracked
  on this tree, dashboard.py has an uncommitted
  `/api/earnings-distribution` route, prewarm list not yet updated).
  Not from this pass; sibling's commit will fix them.
- **GOOGU/METAU yfinance 404s in `logs/runner.log`** still appear
  occasionally — they were removed from `strategy.WATCHLIST` in a
  prior pass but still live in `backtest.py::WATCHLIST` and
  `dashboard.py::SECTOR_MAP`. The dead-cache (`_DEAD_CACHE` /
  `_DEAD_TTL=300s`) suppresses repeated requests for live trading,
  so it's noise-not-bug — but worth cleaning in a future pass.

### How to run

```bash
# Live trader (foreground)
cd /home/zeph/trading-intelligence/paper-trader
python3 -m paper_trader.runner

# Or via systemd
systemctl --user start paper-trader

# Core tests (this pass's surface — ~30s)
cd /home/zeph/trading-intelligence/paper-trader
python3 -m pytest tests/test_core_store.py tests/test_core_market.py \
                  tests/test_core_signals.py tests/test_core_strategy.py \
                  tests/test_core_runner.py tests/test_core_runner_cycle.py \
                  tests/test_core_reporter.py

# Full suite (~80s, may show 2 pre-existing sibling failures
# noted in "Unrelated findings" above)
python3 -m pytest tests/

# Final import check
python3 -c "from paper_trader import signals, reporter, strategy; print('OK')"
```

### Invariants reaffirmed by this pass

- **`decide()` must never crash on a malformed Claude decision** —
  every field-coercion is `try/except`-guarded; a non-numeric
  qty/strike returns `BLOCKED`, never raises. Tests lock the
  contract for both BUY and SELL legs of options.
- **Reporter additive contract** — any analytics-block fault drops
  the affected one-liner, never the whole Discord summary. The new
  `_trade_impact_line` follows the same rule (any fault → `""`).
- **Single source of truth for realized P/L** (invariant #10):
  the trade-alert impact line composes `build_round_trips`
  verbatim — the daily-close round-trip line and the alert can
  never disagree.
- **No new caps / gates** (invariants #2 / #12): the impact line
  is observational, never blocks an alert, never injects into
  the decision prompt.

## ML / backtest review pass (Agent 2, 2026-05-19)

Hybrid pass against `paper_trader/ml/decision_scorer.py`,
`paper_trader/backtest.py`, `run_continuous_backtests.py`. The codebase
already has 28+ documented review passes and a saturated diagnostic
suite (`calibration`, `gate_audit`, `gate_pnl`, `gate_realized`,
`gate_stability`, `gate_abstention`, `horizon_audit`, `feature_coverage`,
`feature_importance`, `baseline_compare`, `baseline_trend`,
`skill_trend`, `persona_skill`, `sector_skill`, `regime_audit`,
`label_audit`, `corpus_audit`, `overfit_gap`, `scorer_freshness`,
`deploy_audit`, `linear_probe`, `response_audit`, `action_skill`,
`persona_leaderboard`) so the bar for a "real bug" was high. None
surfaced; the additions below are coverage + a new diagnostic.

### New: `paper_trader.ml.attribution_audit`

Aggregates `DecisionScorer.feature_contributions` (the per-prediction
Shapley-style ablation `/api/scorer-attribution` already renders) across
the `decision_outcomes.jsonl` corpus. Complementary to existing tools:

- `feature_importance` — "what would the model LOSE if I removed feature X?" (permutation)
- `attribution_audit`  — "what does the model SAY it's actually using?" (attribution)
- `feature_coverage`   — "is feature X varying in the input data at all?"

The combination disambiguates "dead model dimension" (low coverage),
"model ignores a real signal" (high coverage, low attribution), and
"feature carries skill" (high coverage, high attribution, high
permutation drop) — three actionable states no single existing
diagnostic separates.

Verdicts (threshold-driven, exit-code mirrors siblings):

| Verdict | Meaning | rc |
|---------|---------|---:|
| `UNTRAINED` | scorer not trained — nothing to attribute | 2 |
| `INSUFFICIENT_DATA` | < `MIN_RECORDS=30` analyzable rows | 1 |
| `MODEL_INERT` | every feature `mean_abs < 0.10pp` — the gate has no leverage | 2 |
| `CONCENTRATED` | one feature > 50 % of total `|contribution|` — effectively a 1-feature rule | 0 |
| `DIVERSIFIED` | attribution spread across features | 0 |

Run:
```bash
cd /home/zeph/trading-intelligence/paper-trader
python3 -m paper_trader.ml.attribution_audit                # OOS slice (default)
python3 -m paper_trader.ml.attribution_audit --all          # full corpus
python3 -m paper_trader.ml.attribution_audit --json
```

### New test coverage

- `tests/test_decision_scorer_attribution.py` (9 tests) — locks
  `DecisionScorer.feature_contributions` schema, FEATURE_NAMES order
  invariant, the Shapley-style ablation identity
  (`pred_raw == pred_baseline + Σ contributions + interaction_residual`),
  off-distribution propagation, model.predict-raises degradation, and
  non-finite output safe-flagging. Previously zero direct coverage —
  the dashboard's `/api/scorer-attribution` consumer was at silent-
  regression risk.
- `tests/test_attribution_audit.py` (15 tests) — locks the new
  diagnostic's verdict thresholds, aggregate maths
  (top3_share sums to exactly 3.0 across all features),
  IO tolerance (corrupt JSONL lines skipped, missing file empty
  iterator), CLI exit-code convention.

### Phase 3 quant findings (worth reading)

1. **Local-vs-production scorer divergence**. `backtest.db` at this
   repo location is a **symlink** to `/media/zeph/projects/paper-trader/`
   but `data/` is NOT. Local
   `data/ml/decision_scorer.pkl` (`n_train=35`) is far below the
   `_ml_decide` gate threshold (`>= 500`, invariant #5) while the
   production deployment under `/media/zeph/projects/paper-trader/`
   has `n_train=3959`. A fresh continuous loop started from this
   repo would have an inert gate. If you ever run the continuous
   loop from THIS path, expect a 100-cycle warm-up before the gate
   engages.
2. **Continuous loop appears stopped** (at observation time).
   No `run_continuous_backtests` process in `ps -ef`; latest
   `scorer_skill_log.jsonl` cycle was at 18:06 (`>1.5h` stale).
   Two `backtest_runs` rows sit in `status='running'` indefinitely
   (`run 6238` ~5h stale, `run 6243` ~45min — within the 6h reaper
   grace window) so the dashboard would keep rendering dead runs
   as in-flight (CLAUDE.md §11 symptom) until the loop restarts.
3. **Calibration on production scorer + production outcomes**:
   - in-sample: `WEAK_SIGNAL` (spearman 0.286, monotone 1.0, bias 3.0pp)
   - temporal OOS: `DIRECTIONAL_BUT_BIASED` (spearman 0.192,
     monotone 0.78, bias 5.8pp)
   Predictions are directionally informative; magnitudes (the
   `±10/±5/0` gate thresholds) over-predict by ~6pp OOS — the gate
   reads tail predictions as more confident than the realized
   distribution supports.
4. **Attribution-audit surprise** (the new diagnostic, run on
   production scorer + production OOS slice n=1482): top drivers
   are `sector_tech` (8.81pp mean_abs), `regime_mult` (6.31pp),
   `sector_financials` (4.56pp), `mom20` (4.46pp), `rsi` (3.56pp).
   `ml_score` is buried (mean_abs=0.125pp) — the gate is sizing
   trades almost entirely on **sector + regime**, NOT on the
   news-derived ml_score that justified the BUY direction in the
   first place. Trade direction (news-based) and trade size
   (sector/regime-based) are now uncorrelated decisions. This is
   the kind of insight only `attribution_audit` could surface.
5. **`MLP_WORSE_THAN_TRIVIAL` may be outdated**. The
   `baseline_skill_log` shows recent cycles flipping to
   `MLP_ADDS_SKILL`: cycle 3 `ic_gap=+0.07`, cycle 4
   `ic_gap=+0.11`. The historical "the 17-feature MLP is worse
   than a one-line `ml_score` rule" finding may no longer hold
   post-anti-overfit retune. Worth confirming next pass.
6. **News features sparse**: only 4.4 % of training records have
   `news_article_count > 0` (already documented by
   `feature_coverage`; flagged again here because
   `attribution_audit` ranks `news_urgency` #3 despite this
   sparsity — the rare populated rows carry disproportionate
   attribution).

### Test commands for this domain

```bash
cd /home/zeph/trading-intelligence/paper-trader

# Just the ML / backtest / scorer scope (~2 min)
python3 -m pytest tests/ -v -k "ml or backtest or scorer or attribution \
                                or calibration or continuous or baseline"

# Specifically the new attribution coverage
python3 -m pytest tests/test_decision_scorer_attribution.py \
                  tests/test_attribution_audit.py -v
```

### Invariants reaffirmed by this pass

- **Read-only diagnostic discipline** — `attribution_audit` never
  trains, never touches the pickle, no `build_features` / `N_FEATURES`
  / trade-path mutation. Safe to run against the unattended loop
  (same operational rule as every sibling diagnostic).
- **Single-source-of-truth** — `attribution_audit` reuses
  `FEATURE_NAMES` (no shadow ordering),
  `decision_scorer.feature_contributions` (no re-derivation of the
  ablation), and `validation.split_outcomes_temporal` (the EXACT
  OOS slice `baseline_compare` and `calibration --oos` use, so
  cross-diagnostic verdicts compare apples to apples).
- **`feature_contributions` honesty contract** — every failure
  mode (untrained, model raise, non-finite, off-distribution)
  returns a safe dict with `off_distribution=True`, never an
  exception. The new tests lock this contract so a refactor
  cannot silently break `/api/scorer-attribution`.

---

### 2026-05-19 feat (Agent 4 product-engineer pass) — `/api/reasoning-coherence`

A live operator question no existing endpoint answers: when Opus repeats
HOLD for many cycles, is it reiterating *the same thesis* (conviction
signal) or citing *different content each pass* (confusion signal —
groping for a different justification)? Adjacent diagnostics are silent
on this:
  * `/api/decision-drought` counts consecutive NO_DECISION cycles —
    frequency of silence, not coherence of HOLDs that bracket it.
  * `/api/decision-forensics` diagnoses ONE latest decision — no
    across-time view of how reasoning evolves.
  * `/api/thesis-drift` re-tests an OPEN POSITION's entry rationale
    against current state — a position-level question, not a
    reasoning-evolution one. (**Naming-collision note**: my feature is
    reasoning-COHERENCE, the existing one is thesis-DRIFT; distinct
    domains, do not merge.)

**Route** `/api/reasoning-coherence?limit=100` (clamped 5..500). Pure
builder at `paper_trader/analytics/reasoning_coherence.py` reads the
last N rows of `store.recent_decisions`, filters to `action_taken`
starting with `HOLD`, parses each row's `reasoning` JSON envelope
(`{"decision": {"reasoning": "..."}}` — strips `parse_failed:` /
`retry_failed:` prefixes from `strategy._should_retry_parse` capture;
tolerates ```json``` fences), extracts the prose, then computes
**token-set Jaccard similarity between consecutive HOLD reasonings**
over content tokens (alphanumeric, length≥3, stopword-filtered).

**Regime ladder** (median Jaccard over emitted pairs):
  * `STABLE_THESIS` (median ≥ 0.60) — Opus reiterating same justification
  * `DRIFTING` (0.30 ≤ median < 0.60) — reasoning evolves between holds
  * `RAPID_DRIFT` (median < 0.30) — each HOLD cites different content

**State ladder** (operator clarity over verdict pressure):
  * `NO_DATA` — no HOLD rows with parseable reasoning in window
  * `INSUFFICIENT` — fewer than `MIN_PAIRS_FOR_VERDICT = 3` HOLD pairs;
    raw stats still emitted so the operator sees "only 1 HOLD pair this
    window" rather than mistaking silence for OK
  * `OK` — regime + headline emitted

Output carries per-pair `(a_ts, b_ts, similarity)` for drill-down, plus
`min` / `max` / `median` similarity and the threshold constants in-band
so a UI can render the band the regime fell in.

**Locks (`tests/test_reasoning_coherence.py`, 19 tests, 0.80s):**
  1. NO_DATA / INSUFFICIENT / OK ladder + min-pairs gating
  2. Regime thresholds (STABLE/DRIFTING/RAPID_DRIFT) all exercised with
     constructed pairs whose content tokens land deterministically in
     band
  3. JSON envelope extraction (canonical + top-level `reasoning`
     fallback)
  4. `parse_failed:` / `retry_failed:` prefix stripping
  5. HOLD filter excludes NO_DECISION / BLOCKED / FILLED / SKIPPED rows
  6. Garbage-input total-tolerance — no field type raises
  7. Caller's list not mutated
  8. **No-subprocess / no-claude_call / no-yfinance / no-sqlite3 purity**
  9. Route surface: JSON envelope, `limit` clamp (5..500), broken-store
     degrades to 500 + JSON error (not crash)
  10. **NOT behind `@swr_cached`** — cheap by construction
     (recent_decisions + pure builder, no yfinance / no LLM); locked so
     a future refactor doesn't bring in the prewarm-coverage obligation

**Observational only** (invariants #2/#12 — never gates Opus, never
injected into the decision prompt, no caps). Builder degrades to
NO_DATA on any input failure; route degrades to JSON `{"error": ...}` +
500 on store exceptions (mirrors `/api/shadow-vs-claude` +
`/api/thesis-drift` error-pass-through pattern). Builder lives at
`paper_trader/analytics/reasoning_coherence.py`, route appended in
`dashboard.py` IMMEDIATELY AFTER `/api/thesis-drift` — the two are
now the across-time thesis-stability pair the operator can read
together: position-level vs reasoning-level.

**Naming-collision lesson:** my first scope was `/api/thesis-drift` for
this feature; verified before writing that the route already existed
for a different domain (entry-thesis-vs-current scorecard). Renamed to
`/api/reasoning-coherence` BEFORE writing the builder. **Discipline
pin: always grep the dashboard for the proposed route name + builder
filename before scaffolding** — the analytics surface is mature and the
obvious names are usually taken.

**Verify:** `from paper_trader import dashboard;
dashboard.reasoning_coherence_api` imports OK;
`tests/test_reasoning_coherence.py` 19 / 19 pass; adjacent
`test_thesis_drift.py` (27 tests) + `test_shadow_vs_claude.py` (13
tests) both green — no neighboring regression. **Pre-existing failure**
`test_swr_prewarm_coverage::test_every_swr_cached_endpoint_is_prewarmed`
(missing `earnings-distribution` entry) confirmed on HEAD via
stash-swap — NOT from this pass; belongs to another agent's commit
window and is left untouched per concurrent-agent discipline.

## Feature pass — reasoning themes + decision confidence (2026-05-19)

Two new pure-builder endpoints layered on the existing decision-history
surface. Both descriptive complements to `/api/reasoning-coherence`
(which measures pair-wise Jaccard stability — a *vocabulary stability*
metric) that close gaps in the operator's view of Opus's mental state.

### `/api/reasoning-themes`

**Problem:** the decisions table accumulates Opus's reasoning prose
across thousands of cycles — recurring topics ("earnings premium",
"concentration drag", "memory super-cycle", "macro overhang") and
recurring failure prefixes ("claude returned no response (timeout/
empty)") — but the ~80-endpoint analytics surface has **no vocabulary
distribution** view. `/api/reasoning-coherence` measures pair stability,
`/api/decision-forensics` reads one row, and the dashboard reasoning
panels show the latest blob. None answer "what has Opus been *talking
about* this week?"

**Route** `/api/reasoning-themes?limit=100&top_k=10&include_bigrams=1`
(limit clamped 5..500, top_k clamped 3..50). Pure builder at
`paper_trader/analytics/reasoning_themes.py` reads `store.recent_decisions`,
extracts reasoning text from each row across the three real shapes
(JSON envelope `{"decision": {"reasoning": "..."}}`, top-level
`reasoning` key, bare prose for NO_DECISION timeout strings — the
`parse_failed:` / `retry_failed:` prefix + ```json``` fence stripping
mirrors `reasoning_coherence`), tokenises into content words (length≥3,
non-stopword — the same stopword set as `reasoning_coherence` extended
with prose-glue: `amid`/`against`/`during`/`their`/`them`/`such`/...),
walks the stream to emit 1-grams + (optionally) 2-grams of consecutive
content tokens, then ranks by **decisions_mentioning** (the
breadth-not-loudness contract — a phrase repeated 30× in one verbose
reasoning counts as ONE mentioning decision, but a phrase recurring
across 12 different decisions ranks high). Bigrams and unigrams compete
in the same leaderboard; on a tie the bigram wins (informativeness
tie-break — "super cycle" beats "super" + "cycle"). The example excerpt
is locally anchored (±80 chars around the first hit) so the operator
sees the surrounding context, not a beginning fragment.

**Output:** ranked `themes[]` with `{phrase, decisions_mentioning,
share_of_decisions, total_mentions, first_seen_ts, example, is_bigram}`
+ headline naming the top theme + state in `NO_DATA` (zero rows with
extractable reasoning text) / `OK` (leaderboard emitted) +
`n_unparseable` so the operator can spot a malformed-row spike.

**Locks (`tests/test_reasoning_themes.py`, 20 tests, ~0.8s):**
  1. `NO_DATA` / `OK` ladder + n_unparseable accounting
  2. **Breadth beats loudness** — a 30× repetition in one row does NOT
     outrank a phrase appearing 1× across 5 different rows; the
     discriminating test asserts ranking position AND that `total_mentions`
     is preserved as a *secondary* field (a loudness regression that
     swapped the primary sort key would surface high alpha first)
  3. **Bigram-wins-tie informativeness** — `"super cycle"` outranks
     `"super"` and `"cycle"` at identical (decisions_mentioning,
     total_mentions); a reviewer "consistency-fixing" the sort to prefer
     unigrams fails RED
  4. `include_bigrams=False` strips ALL multi-word phrases from leaderboard
  5. Stopword + len<3 filter applied to BOTH the unigram stream and the
     bigram pairing chain (so "the and for with" produces zero content)
  6. Three real reasoning shapes extract correctly: JSON envelope,
     top-level `reasoning` key, `parse_failed:` prefix
  7. NO_DECISION timeout strings (`"claude returned no response …"`) ARE
     themable as bare prose (the host-saturation pattern surface)
  8. `share_of_decisions` denominator is *parseable rows*, not total
     rows — an unparseable row in window does NOT dilute the share of
     a phrase present in every text-bearing row
  9. Example excerpt contains the phrase literally (case-insensitive)
  10. `top_k` clamps high (>50 → 50) and low (0/missing → 3, not 10 via
      Python `or` short-circuit — explicit `is not None` guard locked)
  11. Caller's list not mutated; garbage row keys never raise
  12. Route surface: limit + top_k clamp, garbage params fall back to
      defaults, `include_bigrams=0/false/no/off` all disable, fresh
      `Store` wiring through `monkeypatch.setattr(store.DB_PATH, …)`
  13. **NOT behind the SWR cache** — cheap pure builder; locked so a
      future refactor can't accidentally bring in the prewarm-coverage
      obligation. Discipline pin from #1 (the literal `@swr_cached`
      docstring lock): the route docstring must NOT contain the literal
      substring `@swr_cached` or the inspect-getsource lock self-trips

### `/api/decision-confidence`

**Problem:** every parseable decision blob carries a numeric
`confidence` 0..1 in the `{"decision": {"confidence": 0.7, ...}}`
envelope. Nothing across the ~80-endpoint surface aggregates it.
`/api/scorer-confidence` is the **DecisionScorer** (the tiny CPU MLP
on the backtest side), not Opus; `/api/decision-forensics` reads one
decision; `/api/reasoning-coherence` measures pair stability — none
answers "is Opus confidently sitting on its hands, or uncertainly doing
nothing?" An operator scanning a paralysed week cannot distinguish
high-conviction HOLDs around a binary event (correct) from low-
conviction churn (a flag to operators that the model is hedging).

**Route** `/api/decision-confidence?limit=100` (clamped 5..500). Pure
builder at `paper_trader/analytics/decision_confidence.py` reads
`store.recent_decisions`, extracts the numeric `confidence` from each
row's reasoning JSON (canonical envelope + top-level fallback +
`parse_failed:` prefix tolerance — same shape ladder as `reasoning_themes`),
computes median / mean / min / max / 4-bucket histogram + per-action
breakdown + recent-vs-older trend split + regime verdict.

**Numerical-robustness contract:**
  * Out-of-band values (`1.5`, `-0.3`) are **CLAMPED to [0, 1]**, not
    dropped. A model emitting 1.2 is bounded conviction noise, not
    invalid data — silently dropping would hide a real model bug from
    the operator. (Discriminator test: `[1.5, -0.5, 0.7, 0.6, 0.8]` →
    `n_with_confidence=5`, `max=1.0`, `min=0.0`.)
  * `NaN` IS dropped (treated as unparseable — the operator sees
    `n_unparseable` go up)
  * Non-numeric (`"high"`, `null`) dropped silently
  * Order of caller's list preserved — `recent_decisions` is newest-
    first, so the FIRST half of the value stream is the *recent* half
    in the trend split (TRENDING_UP = recent half median ≥ older half
    median by `TREND_DELTA = 0.10`)

**Regime ladder** (median confidence):
  * `CAUTIOUS` — median < `CAUTIOUS_THRESHOLD = 0.45` ("Opus is hedging")
  * `NEUTRAL` — 0.45 ≤ median < `CONVICTED_THRESHOLD = 0.70`
  * `CONVICTED` — median ≥ 0.70 ("Opus is decisive")
  Boundary inclusive on the *upper* side: `median == 0.45` → NEUTRAL;
  `median == 0.70` → CONVICTED.

**State ladder** (operator clarity over verdict pressure — the
`reasoning_coherence` / `tail_risk` precedent):
  * `NO_DATA` — zero parseable confidence values
  * `INSUFFICIENT` — fewer than `MIN_SAMPLES_FOR_VERDICT = 5` parsed;
    raw stats (median, mean, buckets) emitted, regime + trend withheld
  * `OK` — verdict emitted

**Output:** state + regime + median/mean/min/max + 4-bucket histogram
(`low <0.4`, `medium [0.4,0.6)`, `high [0.6,0.8)`, `very_high [0.8,1.0]`)
+ `by_action` (per-leading-verb median/mean/n, so the operator can spot
"HOLDs median 0.75, BUYs median 0.42 — Opus buys with less conviction
than it holds") + trend block (`{tag, recent_median, older_median, delta,
split_size}` or None when sample size won't support a split) +
threshold constants in-band for UI rendering. Headline composes the
regime line + a trend appendix when non-FLAT.

**Locks (`tests/test_decision_confidence.py`, 28 tests, ~0.4s):**
  1. `NO_DATA` / `INSUFFICIENT` / `OK` ladder + min-sample gating;
     raw stats emitted under INSUFFICIENT, regime + trend withheld
  2. Three real reasoning shapes extract correctly: envelope,
     top-level `confidence`, `parse_failed:` prefix
  3. **Out-of-band clamp, NOT drop** — the discriminating contract;
     `1.5` → `1.0` in `max`, not absent from `n_with_confidence`
  4. NaN dropped; non-numeric (`"high"`) dropped; `null` dropped
  5. CAUTIOUS / NEUTRAL / CONVICTED at exact threshold boundaries
     (inclusive-upper convention: 0.45 → NEUTRAL, 0.70 → CONVICTED)
  6. TRENDING_UP / TRENDING_DOWN / FLAT by `recent_median − older_median`
     vs `TREND_DELTA`; recent half is the FIRST half of caller-supplied
     order (the `recent_decisions` newest-first contract)
  7. Trend withheld when split half is < 2 samples
  8. Per-action breakdown groups by leading verb (`HOLD NVDA → HOLD` →
     `HOLD`; `SELL_CALL NVDA 200C → FILLED` → `SELL_CALL`); blank /
     None `action_taken` → `UNKNOWN` bucket
  9. Buckets partition [0,1] with no gaps; all 4 keys present (zeroed
     when empty); `very_high` includes exact `1.0`
  10. Garbage-row tolerance — `{}`, `{"reasoning": 12345}`, missing keys
      degrade to `NO_DATA`, never raise
  11. Route: fresh-Store wiring, limit clamp 5..500, garbage param
      `?limit=banana` falls back to default 100, **NOT behind the SWR
      cache** (same docstring-literal discipline as `reasoning_themes`)

**Observational only** (invariants #2/#12 — never gates Opus, never
injected into the decision prompt, no caps). Builder degrades to
NO_DATA on any input failure; route degrades to JSON `{"error": ...}` +
500 on store exceptions. Both endpoints sit immediately after
`/api/reasoning-coherence` in `dashboard.py` so the three across-time
decision-introspection diagnostics (coherence / themes / confidence)
are co-located.

**Applies on next paper-trader restart** (the documented pattern for
every recent feature — the running `:8090` continues until restart).

**Verify:** `from paper_trader import dashboard;
dashboard.reasoning_themes_api; dashboard.decision_confidence_api`
imports OK; `tests/test_reasoning_themes.py` 20/20 +
`tests/test_decision_confidence.py` 28/28 + adjacent
`test_reasoning_coherence.py` 19/19 + `test_core_dashboard_helpers.py` /
`test_core_dashboard_articles_db.py` all pass — no neighboring
regression. Full suite still collects (2657 tests).

## Review pass #32 — paper-trader core hybrid (2026-05-19)

**Agent persona:** debugger + feature dev + live trader; concurrent with
three sibling HYBRID agents (ML+backtest, core, feature-dev).

### Phase 1 fix (commit bdffbf8) — quota-recovery latch retries until delivered

**Bug**: `runner._cycle` cleared `_quota_alert_active = False` *unconditionally*
inside the recovery-notice path, even when `reporter._send` returned False or
raised. Result: a transient openclaw / Discord outage at the moment the quota
recovered silently dropped the "we're back" message AND cleared the latch —
the operator (who only sees Discord) was stuck believing the trader was still
frozen until the NEXT quota outage re-alarmed. The whole point of the latch is
operator-visible state; clearing it without delivery is the failure mode it
exists to prevent.

**Fix**: mirror the alarm-path symmetry — clear the latch ONLY on a confirmed
successful send. A failed send keeps the latch set so the next non-NO_DECISION
cycle retries the notice; the latch itself dedupes so we never spam. Exact
mirror of the original `if reporter.send_quota_alert(detail): _quota_alert_active = True`
asymmetry — both paths now key off the same `bool(_send(...))` contract.

```python
if _quota_alert_active and status != "NO_DECISION":
    ok = False
    try:
        ok = bool(reporter._send("✅ **CLAUDE QUOTA RECOVERED** ◈ ..."))
    except Exception as e:
        print(f"[runner] quota recovery notice failed: {e}")
    if ok:
        _quota_alert_active = False
```

**Test coverage** (4 new tests in `TestCycleQuotaRecoveryUndelivered`):
- `test_failed_recovery_send_keeps_latch_so_we_retry` — `_send` returns False:
  latch stays True, recovery msg was attempted (operator-visible log line),
  next cycle will retry.
- `test_recovery_send_exception_keeps_latch_so_we_retry` — `_send` raises:
  cycle does NOT propagate the exception (would crash the live loop), latch
  stays True.
- `test_retry_until_success_then_latch_clears` — flaky send: first attempt
  fails, second succeeds. After the second the latch clears and a fresh outage
  re-alarms (the rearm contract).
- `test_successful_recovery_send_clears_latch_unchanged` — regression guard:
  the happy path (single OK send) still clears the latch on the first try.

### Phase 2 feature (commit 9f12f65 + bundled into a028ad5)

**Enriched `/api/portfolio`** — backwards-compatible enrichment of the public
lean endpoint that Digital Intern's dashboard (port 8080) cross-fetches:

Legacy keys (UNCHANGED — cross-port consumers never break):
- `total_value`, `cash`, `starting_value`

New trader-actionable fields, composed *purely* from the already-cached
`portfolio.positions_json` row (no extra store reads, no network — the
endpoint stays the lowest-latency public surface):
- `n_positions` — open lots count
- `open_value` — Σ market_value across open lots (= total_value − cash)
- `unrealized_pl` / `unrealized_pl_pct` — book-wide drift since entry. The
  pct denominator is `total_value` (the equity base) to align with
  `/api/benchmark` / `/api/drawdown` % framing.
- `stale_marks` — count of positions flagged `stale_mark=True` (yfinance
  returned nothing; mark fell back to avg_cost). Nonzero means the
  `unrealized_pl` understates real exposure — explicit so the trader is never
  misled by a phantom "flat" book the `stale_mark` flag was added to expose.
- `last_updated` — ISO timestamp of the most recent mark-to-market write.
  A polling caller can detect "the trader has stopped writing" without
  re-reading equity_curve.
- `pnl_vs_start` / `pnl_vs_start_pct` — absolute and % delta from the $1000
  baseline (`INITIAL_CASH`, invariant #12; never a literal).

**Degrade-safe**: every numeric coercion is try/except so a malformed
`positions_json` row (defensive — `get_portfolio` already falls it back to
`[]`) or a non-numeric `unrealized_pl` degrades to zeros while the legacy
three keys are always present. `total_value=0` yields `unrealized_pl_pct=None`
(no ZeroDivisionError on the synthetic empty case).

**Test coverage** (`tests/test_core_dashboard_portfolio.py`, 13 tests):
- `TestPortfolioApiLegacyContract` — legacy three keys present, CORS header
  intact for cross-port fetch.
- `TestPortfolioApiEnrichedFields` — exact computed values: n_positions /
  open_value summed from cached marks, signed `unrealized_pl` (a +20 and a −5
  net to +15), the % denominator is `total_value` (a divide-by-open_value
  bug would inflate the % whenever cash is high), stale_marks counts only
  flagged rows, pnl_vs_start positive AND negative cases, last_updated parses
  as ISO.
- `TestPortfolioApiEmptyAndDegradeSafe` — empty book, non-list
  `positions_json`, non-numeric `unrealized_pl`, total_value=0.

**Note on commit attribution**: the dashboard.py edit was caught in a
sibling agent's commit `a028ad5` (concurrent-staging race — the
`pt-concurrent-samerole-staging-race.md` memory pattern). The test
file lands separately as `9f12f65`. Net: `/api/portfolio` enrichment
is live + locked by 13 regression tests.

### Phase 3 live-validation findings (live trader 2026-05-19 ~04:40 UTC)

1. **Host saturation pathology continues** (#1 known): /api/host-guard reports
   `state=STARVED — 78% of the last 27 decisions never reached Opus`. With
   four concurrent HYBRID Opus agents on the box (≥4 = `opus > MAX_OPUS`
   = saturation), the live trader's pre-flight guard skips most cycles. The
   bot cannot resolve this by trading — it is an OPS problem (`_host_pulse_line`
   in reporter.py routes it to Discord verbatim). NOT a code bug.

2. **NVDA earnings discipline holds**: trader sized 44.5% NVDA + 14.9% TQQQ
   into NVDA earnings (0.8d away). /api/event-calendar surfaces
   `NVDA — earnings in 0.8d [HELD_IMMINENT]`. /api/risk fires MEDIUM
   concentration warning at top1=44.47%. The forward stack works as designed.

3. **Build staleness churn**: /api/build-info reads `stale: true` on average
   within a few minutes of any push (multiple HYBRID agents committing
   concurrently). The git-watcher's deferred restart is honored, but each
   restart kills the in-flight cycle and contributes to the NO_DECISION rate
   (a self-amplifying loop: agents that touch the codebase trigger restarts
   that surface as NO_DECISION metrics). Documented; no code change here.

4. **Discord delivery healthy**: /api/runner-heartbeat `notify.verdict=HEALTHY`,
   `last_ok_ts` within seconds. The 2026-05-17 PATH/shebang outage that
   originally motivated the openclaw fallback resolver remains fully fixed.

5. **Enriched /api/portfolio live**: response shape post-fix:
   ```json
   { "n_positions": 2, "open_value": 593.26, "unrealized_pl": 0.0,
     "stale_marks": 0, "last_updated": "2026-05-19T04:40:01.797701+00:00",
     "pnl_vs_start": 0.0, "pnl_vs_start_pct": 0.0,
     "total_value": 1000.0, "cash": 406.74139404296875,
     "starting_value": 1000.0 }
   ```
   At-a-glance: the trader can confirm "2 positions, exactly at start, marks
   are fresh" in one call rather than fanning out across /api/state,
   /api/equity-freshness, and /api/risk.

6. **Equity integrity OK**: /api/equity-freshness `verdict=FRESH`, no
   portfolio/equity divergence — the live trader and the equity_curve agree
   on $1000.00 ± 0% (the trader memory `pt-portfolio-equity-divergence.md`
   shows this is a temporary mid-cycle artifact, not a bug — confirmed clean
   right now).

### How to run / test (unchanged but locked here)

```bash
# Live trader (foreground; manual launch — see pt-systemd-vs-manual-restart-spam.md)
cd /home/zeph/trading-intelligence/paper-trader
python3 -m paper_trader.runner

# Final import check
python3 -c "import sys; sys.path.insert(0,'.'); from paper_trader import signals, reporter, strategy; print('imports OK')"

# Phase-1 regression net (~5s)
python3 -m pytest tests/test_quota_guard.py -v

# Phase-2 regression net (~1s)
python3 -m pytest tests/test_core_dashboard_portfolio.py -v

# Full suite
python3 -m pytest tests/ -v
```

### Invariants reaffirmed by this pass
- **#10 single source of truth**: `_INITIAL_EQUITY = INITIAL_CASH` — the new
  /api/portfolio `pnl_vs_start` keys off the module constant, never a literal.
- **#12 cross-process value baseline**: `INITIAL_CASH` is the only baseline
  the new endpoint reports against. The legacy `starting_value` key is the
  same constant; consumers that ever switch on it never break.
- **Reporter dedupe symmetry**: the alarm-path "set on success" pattern is
  now mirrored on the recovery path "clear on success" — both keyed off the
  same `bool(_send(...))` contract, so a Discord outage at *either* edge of
  the quota cycle leaves the operator with the truthful state (alarmed if
  outage started, alarmed if recovery undelivered).
- **`pt-concurrent-samerole-staging-race.md` discipline observed**: this
  pass's dashboard.py edit was bundled into a sibling agent's commit. Test
  file committed standalone with explicit pathspec; AGENTS.md commit lands
  alongside the test. Never `git add -A` in this tree.

---

### 2026-05-19 review pass — ML+backtest hybrid (Agent 2): per-horizon scorer skill + evaluate_scorer_oos NaN-sentinel parity

**Phase 1 bug fix — `evaluate_scorer_oos` silently coerced null forward returns to 0.0**

`paper_trader/validation.py::evaluate_scorer_oos` used
`_to_float(r.get("forward_return_5d"), 0.0)` to read the realized
target on each OOS record. A row with a missing key, JSON null, or
non-finite (`inf`/`-inf`/`nan`) `forward_return_5d` then defaulted to
0.0 and contributed `(pred - 0.0)**2` to RMSE — fabricating a flat
outcome and biasing the reported skill metric. The sister function
`run_continuous_backtests._oos_rank_metrics` was already hardened with
a `float("nan")` sentinel + `a == a` drop (see
`TestOosRankMetrics::test_records_missing_forward_return_are_dropped_not_zeroed`);
`evaluate_scorer_oos` was the last function still trusting the silent
0.0 fallback. The bug is **latent** on current live data (7,413/7,413
outcomes carry a finite `forward_return_5d`) but the contract was
wrong, and a future writer emitting a null target would silently
inflate the OOS RMSE — pre-fix RMSE on a constructed (1 real + 4 null)
input is ~8.94 vs the correct 0.0.

**Test locks** (2 new tests in `tests/test_validation.py`):
* `test_missing_forward_return_dropped_not_zeroed` — five records,
  four with null/missing/inf/NaN, one well-formed; post-fix `n=1`,
  `rmse=0.0` exactly.
* `test_sell_sign_flip_still_applied_after_nan_filter` — the NaN
  filter must NOT bypass the SELL sign-flip on records it KEEPS.

**Phase 2 feature — per-horizon (10d, 20d) OOS rank-IC + dir-acc in the scorer-skill ledger**

The scorer trains on `forward_return_5d`, but each outcome row also
carries `forward_return_10d` / `forward_return_20d` (the 2026-05-18
multi-horizon instrumentation). Live: 6,300/7,413 outcomes carry 10d
and 6,265 carry 20d. The per-cycle scorer-skill ledger only reported
5d metrics — yet AGENTS.md notes leveraged-ETF strategies have noisy
5d windows but stronger multi-month returns, so a non-trivial signal
at 10d/20d when 5d sits at noise (current state: mean OOS IC ≈ 0.05,
dir_acc ≈ 0.52 across 22 cycles) is the exact research signal a
skeptical quant needs to evaluate the gate's edge.

**Adds**:
* `run_continuous_backtests._oos_multi_horizon_metrics(scorer,
  oos_records, horizons)` — mirrors `_oos_rank_metrics` (same predict
  signature, SELL sign-flip, tie-aware Spearman via
  `calibration._spearman`) but evaluates against `forward_return_{h}d`
  for each requested horizon. Each horizon reports its own `n`
  honestly — a row missing `forward_return_20d` drops from the 20d
  cell only, never poisons the 10d view. Never raises (returns empty
  per-horizon sentinels on any fault — the "scorer-train status must
  stay truthful" discipline, mirrored from `_oos_rank_metrics`).
* Six new tokens in `_train_decision_scorer`'s status string:
  `oos_n_10`, `oos_diracc_10`, `oos_ic_10`, and 20d siblings. Wired
  alongside the 5d tokens so the per-cycle scorer-skill ledger
  automatically carries them via the existing `**parsed` splat in
  `_append_scorer_skill_log`.
* `_parse_scorer_status` extracts the new tokens with a strict
  `(?:^|\s)key=` boundary so the legacy `oos_n=` lookup does **NOT**
  swallow `oos_n_10=`. Old (pre-feature) status strings parse cleanly
  with 10d/20d fields defaulting to `None` (read-side back-compat for
  every historical skill-log row).

**Test locks** (13 new tests in `tests/test_continuous.py`):
* `TestOosMultiHorizonMetrics` (6 tests) — untrained sentinel,
  perfect-rank-ordering at each horizon, per-horizon missing-target
  drop (10d-only row contributes to 10d only), SELL sign-flip parity,
  predict-exception isolation, and the wiring lock that asserts every
  new token appears in `_train_decision_scorer`'s status string.
* `TestParseScorerStatus` (3 new tests) — full multi-horizon parse,
  back-compat for legacy status, and the `oos_n` vs `oos_n_10`
  boundary lock that proves the strict regex prevents substring
  collision.

**Phase 3 quant findings (from running the new metric on live data)**

End-to-end smoke on the deployed pickle (`n_train=3959`,
`(32,16)/alpha=1e-2/early_stopping=True` — matches `MLP_CONFIG`, no
deploy_stale) against the most recent 5,000 outcomes with the
canonical temporal 80/20 split:

| horizon | n | rank_IC | dir_acc |
|---|---|---|---|
| 5d | 1000 | +0.1141 | 0.5516 |
| 10d | 990 | +0.0557 | 0.5592 |
| 20d | 957 | +0.0372 | 0.5350 |

**Finding 1 (decisive):** the scorer's edge is **concentrated at 5d
and decays at longer horizons** — rank_IC halves at 10d and thirds at
20d. The gate's captured then-deployed prediction on the full-history
slice (5,165 captured rows) shows the SAME monotone decay: 5d IC
+0.0471, 10d +0.0189, 20d +0.0136. This is the **opposite** of the
AGENTS.md qualitative hypothesis ("leveraged ETFs have noisy 5d but
stronger multi-month returns"). A future multi-horizon training
experiment cannot assume the longer-horizon target carries MORE
signal than 5d — at the current feature set the opposite holds.

**Finding 2:** the continuous loop appears to have been **DOWN** for
~10 hours at review time — last write to
`decision_outcomes.jsonl` / `scorer_skill_log.jsonl` was 11:06 vs
21:47 inspection. Two `status='running'` rows aged 9.9h and 14.4h
exceed `_reap_orphaned_runs`'s 6h `max_age_hours` guard; they will be
swept to `failed` on the next loop start. Action: confirm loop is
intended down.

**Finding 3:** `_ml_is_qualified()` currently **PASSES** — median
`vs_spy_pct` across the most recent 20 qualifying runs is **+143.33%**
(threshold 0.0%). Qualification is based on **persona-driven backtest
alpha**, NOT on scorer skill (the scorer itself has near-zero OOS
edge). An operator should NOT read advisor-presence as "the scorer
has skill."

**Finding 4:** `baseline_skill_log.jsonl` verdict oscillates by
window: cycles 1–2 → `MLP_NO_BETTER_THAN_TRIVIAL`, cycles 3–4 →
`MLP_ADDS_SKILL`. The 17-feature MLP's edge over a one-line
`ml_score` rule is **borderline and window-sensitive**. The new
per-horizon ledger fields will surface whether this holds at 10d/20d
once the loop runs the next ~10 cycles.

**Finding 5:** the news_urgency / news_article_count propagation is
sound — outcome rows that parse values use them, no-news rows fall
back to None which `build_features` neutralizes to 50/1 (the
documented inference-vs-train parity).

**Finding 6:** 24/501 (4.8%) backtest runs have `status='failed'`,
20 carrying the `[reaped: orphaned running row]` note. Reaper working
as designed.

**Test commands for this domain (Agent-2 scope):**
```
# Phase-1 + Phase-2 changes
python3 -m pytest tests/test_validation.py -k evaluate_scorer_oos -v
python3 -m pytest tests/test_continuous.py -k "MultiHorizon or ParseScorer" -v

# Full ML/backtest regression (the canonical filter)
python3 -m pytest tests/ -k "ml or backtest or scorer or validation or continuous"
```
398 tests pass (up from 389 — 2 new in Phase 1, 13 new in Phase 2).

**Invariants reaffirmed:**
* "Scorer-train status must stay truthful" — every new diagnostic
  block is guarded SEPARATELY so a downstream crash never masks a
  successful train or the previously-reported metrics. The per-horizon
  block degrades to all-n/a tokens on any fault.
* Read-side back-compat — the parser's strict boundary regex keeps
  every historical skill-log row parseable; dashboard panels reading
  the parsed dict get `None` for the new fields on old rows, never a
  KeyError.
* SELL sign-flip parity — `evaluate_scorer_oos`, `_oos_rank_metrics`,
  `_oos_multi_horizon_metrics`, and `train_scorer` all flip the
  realized target sign for SELL actions. Tested across all four
  surfaces now.

**Concurrent-staging hazard observed:** my Phase 2 `git add` was
overwritten by a sibling agent's `git add -A` between staging and
commit — the first commit `a028ad5` carries my message but a
sibling's `reasoning_coherence` files. Re-committed with explicit
pathspec (`git commit -- run_continuous_backtests.py
tests/test_continuous.py`) as `c18e679`, which contains the correct
files. The hazard matches [[pt-concurrent-samerole-staging-race]] in
the agent memory exactly.


## Review pass #33 — paper-trader core hybrid (2026-05-19)

### Phase 1 fixes (commit ef6e9d0) — 4 surgical bugs in the Discord & decision-clock surfaces

1. `reporter._trade_impact_line` SELL "closed" branch emitted
   `closed — cash $X` *and then* unconditionally appended a second
   `cash $X` token, so any full-close where the round-trip lookup
   could not match (caller did not pass `store`, or
   `build_round_trips` raised) produced a duplicated
   `cash $X · cash $X` tail in the trade alert. No test covered this
   path — added two: `store=None` and a `store` whose
   `recent_trades` raises both must produce **exactly one** cash
   token (`body.count("cash $X") == 1`).
2. `reporter._decision_clock_line` was defined but never wired into
   `send_hourly_summary` or `send_daily_close`, so the
   `HOURLY_CONCENTRATION` verdict (the canonical signal that a
   recurring NY hour is being starved by out-of-band Opus jobs) only
   reached `/api/decision-clock` and never Discord. The end-to-end
   test `test_hourly_summary_includes_line_when_concentrated` was
   failing for this reason — fixed by appending `_decision_clock_line`
   after `_position_attention_line` in both summary paths (preserves
   the load-bearing HOST→CAPITAL→ATTENTION→CLOCK ordering: structural
   risks first, drift signals after).
3. `analytics.decision_clock.build_decision_clock` used
   `int(days or 7)` which silently coerced an explicit `days=0` to
   the default 7 instead of clamping to 1. `test_days_clamped`
   `(0, 1)` was failing. Fixed by coercing `None`/non-numeric → 7
   explicitly, then clamping `[1..30]`.
4. `dashboard._swr_prewarm` was missing
   `("earnings-distribution", earnings_distribution_api)` — the
   `@swr_cached` `/api/earnings-distribution` endpoint cold-stalled
   with `{"warming": true}` on first poll after every restart.
   `test_swr_prewarm_coverage`'s "prewarm == @swr_cached" invariant
   locked the gap.

### Phase 2 feature (commit 838ec76) — per-position hold-age on Discord position lines

`_pos_hold_age_token(p, now=None)` reads `opened_at` (always carried
by `store.open_positions()`; absent on the unit-test position dicts
and the persisted `portfolio.positions_json` cache, so existing
assertions stay byte-compatible) and emits a compact `held 42m` /
`held 5h` / `held 3d` token appended in `_portfolio_lines` after the
existing `_pos_pct_weight` parenthetical. Mirrors the `held=Xd`
annotation the Opus decision prompt already shows per position
(`strategy._hold_age_str`) so the operator's Discord surface and the
decision engine see the same staleness signal at a glance.

A 4-day-stuck loser previously rendered identical to a fresh fill in
the hourly / daily Discord summary — the desk's #1 documented
pathology (disposition effect) was visible to the decision engine but
invisible to the operator reviewing the book. Sub-minute returns are
silent (no flicker on a just-opened lot); a future `opened_at` (NTP
step-back — documented clock-skew hazard) clamps to silent rather
than rendering a negative age (the `_fmt_trade_stamp` precedent).
Observational only, no caps (invariants #2/#12 — the `stale_mark` /
`pct-weight` precedent). 11 new tests under
`TestPosHoldAgeToken` cover bucket boundaries, missing / unparseable
/ naive / future fields, and the byte-compat guarantee for the
existing `_portfolio_lines` call sites.

### Phase 3 live-validation findings (live trader 2026-05-19 ~05:38 UTC)

1. **Service is in a systemd-vs-manual restart spam** (known —
   [[pt-systemd-vs-manual-restart-spam]]): the manual orphan instance
   (PID 2502051, PPID 1) holds the singleton flock; the systemd unit
   restarts every 15s, fails the lock acquisition, and exits — 276
   restart attempts in journal. Documented; do not "fix" by killing
   the manual instance, the operator chose this configuration.
2. **77% NO_DECISION rate over the last 35 cycles** due to host
   saturation (known — [[pt-no-decision-host-saturation]]): the
   `/api/host-guard` `pulse.state` reads `STARVED` (probe clear now,
   but intermittent out-of-band parallel Opus jobs OOM-starve the
   live trader). This very review pass is one of the parallel Opus
   sources. Not a code bug.
3. **`_option_expired` UTC-vs-NY same-day-after-close gap** (low
   impact, **not fixed this pass** — see Phase 1 advisor scope
   ruling). For an option whose expiry == today, the comparison
   `exp < datetime.now(timezone.utc).date()` only flips to True at
   UTC midnight, which is 3-4 hours after the actual 16:00 ET close.
   In that window an expired option is marked at avg_cost with
   `stale_mark=True` instead of at `_expired_intrinsic` (worth $0
   OTM or settlement). Market is closed so no trade executes against
   the wrong mark — the impact is one stale-display window per
   monthly expiry on the dashboard. Fix is straightforward (extend
   the signature to take an optional `now: datetime` and compare via
   NY date + same-day close-time check using `market.close_minute`)
   but adds test surface across 6 existing test cases that pin
   `today=date(...)`. Skipped this pass; documented for the next.
4. **`/api/feed-health` HEALTHY** — 419 live articles in 2h, newest
   0.1h old. The digital-intern pipeline is feeding the live trader
   correctly; the NO_DECISION storm is purely Opus-side.
5. **Dashboard responds fast** — 34 ms TTFB on `/` (280 kB HTML).
   `/api/build-info` reports `stale: true, behind: 1` (the live
   process is on `72285ac`, HEAD is `838ec76`) — the git-watcher will
   restart between cycles to apply the Phase 2 hold-age annotation.

### How to run / test

Live trader: managed by `paper-trader.service` (systemd --user) **or**
a manual orphan — current state has the manual instance holding the
flock. To verify after a config change:
```
python3 -m paper_trader.should_restart    # OK / RESTART / OPS_ONLY / ERROR
```

Tests (this pass's scope):
```
cd /home/zeph/trading-intelligence/paper-trader
python3 -m pytest tests/test_core_reporter.py -v          # 166 tests
python3 -m pytest tests/test_decision_clock_builder.py -v # 28 tests
python3 -m pytest tests/test_swr_prewarm_coverage.py -v   # 3 tests
python3 -m pytest tests/ -v 2>&1 | tail -20               # full sweep
```
2448 tests pass + 1 skipped after this pass (was 2438 before; the
+10 includes the 8 new hold-age tests, 2 new duplicate-cash tests,
and a small overlap with sibling Agent 4's `/api/sector-signal-fit`
additions).

### Invariants reaffirmed by this pass

* **No `git add -A`** — sibling Agent 2 and Agent 4 were running
  concurrently on the same tree; explicit pathspec on every commit
  ([[pt-concurrent-samerole-staging-race]] discipline). My Phase 2
  commit (`838ec76`) staged only `paper_trader/reporter.py` +
  `tests/test_core_reporter.py`; Agent 4's `/api/sector-signal-fit`
  + analytics file + tests were correctly left out.
* **Reporter additive contract** — every Phase 1 + Phase 2 change
  preserves the existing failure mode: a fault drops the offending
  line, never the whole summary. Locked by the existing
  `summary_still_sends_when_*_faults` tests across multiple
  reporter blocks.
* **Invariants #2/#12** — every new feature is observational only;
  none gate Opus, none add silent caps. The hold-age token is the
  same surfacing trajectory `stale_mark` / `_pos_pct_weight` /
  `_session_block` followed.


### 2026-05-19 review pass — ML+backtest hybrid (Agent 2): SECTOR_MAP coverage fix + live diagnostics

#### Phase 1 (commit: skipped) — no concrete code bugs to fix

After a focused audit of `paper_trader/ml/decision_scorer.py`,
`paper_trader/backtest.py`, and `run_continuous_backtests.py` against
the deep ML+backtest review history (passes #15–32), no concrete logic
bugs surfaced. The hardening loop (None-coercion guards, atomic
pickle writes, train_scorer dedup, off-distribution gate abstention,
benchmark-honesty notes, FILLED-only training filter, per-cycle
ledgers + trims) has covered every code path I'd otherwise flag.
Set `bugs_fixed=0` per the per-commit guard. Targeted suites stayed
green at the baseline (277 passed across
`test_decision_scorer.py`/`test_decision_scorer_attribution.py`/
`test_scorer_honesty.py`/`test_backtest.py`/`test_ml_backtest_seams.py`/
`test_ml_backtest_coverage.py`/`test_ml_backtest_review.py`/
`test_continuous.py`/`test_calibration.py`/`test_horizon_audit.py`/
`test_ml_live_opinion.py`/`test_attribution_audit.py`).

#### Phase 2 (commit `e691740`) — `SECTOR_MAP` coverage of all 41 unmapped watchlist tickers

The DecisionScorer's 7-way sector one-hot is one of its 17 features.
A coverage audit revealed **35% of `WATCHLIST` (41/118 tickers) had
no `SECTOR_MAP` entry** — all silently collapsing into `sector_other`
alongside Toyota and homebuilders. The scorer could not learn any
sector-conditional pattern for these names: LRCX (semi equipment) had
the same sector encoding as NAIL (homebuilders), and every broad-
index 2x/3x leveraged ETF (QLD/SSO/UDOW/URTY/TNA/...) was
indistinguishable from utility/defense ETFs in feature space.

Added 37 explicit mappings:

- **tech** — semi cap-equipment (`LITE`/`AMAT`/`LRCX`), int'l tech
  ADRs (`BABA`/`SAP`/`SONY`), EV/innovation (`RIVN`/`NIO`/`ARKK`),
  3x broad-index leveraged (`UDOW`/`URTY`/`MIDU`/`TNA`/`WANT`), 2x
  broad-index (`QLD`/`SSO`/`MVV`/`SAA`/`UWM`), single-stock 2x
  (`AAPLU`/`SMCI2X`/`PLTU`/`LNOK`), 2x tech rotation (`USD`/`ROM`),
  3x inverse broad-index — *same correlation magnitude*, opposite
  direction, mirroring the existing `SOXS`/`TECS`/`FNGD` pattern
  (`SQQQ`/`SPXS`/`SDOW`/`SRTY`/`TZA`/`HIBS`).
- **financials** — `BRK-B`/`HSBC`/`SQ`/`FAZ` (last is 3x inverse,
  mirrors `FAS`).

Added an explicit `INTENTIONALLY_OTHER` frozenset (6 tickers:
`TM`/`UXI`/`NAIL`/`DFEN`/`UTSL`/`XLI`) for names whose economic
sector has no enum in `SECTORS` — they correctly remain in
`sector_other` rather than being mis-coupled to tech/financials.

**Invariant locked** by 6 new tests in `TestSectorMapping`:

| Test | What it catches |
|------|-----------------|
| `test_watchlist_coverage` | a new `WATCHLIST` ticker added without an explicit `SECTOR_MAP` entry **and** not in `INTENTIONALLY_OTHER` fails loudly here, instead of silently degrading scorer feature quality |
| `test_sector_map_values_are_valid_sectors` | catches typos like `'techy'` that would yield an all-zero sector one-hot |
| `test_intentionally_other_does_not_overlap_sector_map` | a ticker can't appear in both halves (the intent would be ambiguous) |
| `test_specific_high_value_mappings` | pins `LRCX`/`BABA`/`RIVN`/`QLD`/`BRK-B`/`SPXS` so a "fix" that drops both halves the same wrong way is caught (the coverage test alone would pass) |
| `test_sector_encoding_changes_for_newly_mapped_tickers` | end-to-end proof that `build_features` now puts `LRCX` in the same sector block as `NVDA`, while `NAIL` correctly stays in `sector_other` |
| (pre-existing) `test_all_sectors_in_map` | every declared sector still appears somewhere |

**Pickle schema unchanged** — still 7-way sector one-hot at the same
7 feature slots (10..16). Old pickles still load. The next training
cycle picks up the expanded mapping; the transient cost is one cycle
of slightly off-distribution predictions for the 37 newly-mapped
tickers, and the existing `off_distribution` guard in `_ml_decide`
already abstains the conviction gate on those.

**421 tests pass** under the canonical filter
`-k "ml or backtest or scorer or calibration or continuous or horizon"`.

#### Phase 3 — live quant findings (skeptical-quant perspective)

1. **Continuous loop is NOT currently running.** `ps -ef | grep
   run_continuous` returns no Python process. `continuous.log` last
   touched 2026-05-18 12:03 (~10h stale); the structured per-cycle
   ledgers (`scorer_skill_log.jsonl`, `baseline_skill_log.jsonl`)
   stopped at 11:06 today with only **4 cycles** total ever logged
   on this deployment. A skeptical quant has effectively no live
   training trend to monitor — `skill_trend` / `baseline_trend`
   return `INSUFFICIENT_DATA` against the on-disk ledgers.
2. **Split-brain: `SCORER_PATH` local pkl (n_train=35) vs USB pkl
   (n_train=400).** The decision_scorer module's `SCORER_PATH`
   resolves to `paper-trader/data/ml/decision_scorer.pkl` (local
   filesystem) which carries `n_train=35` from a stub training
   probably done outside the continuous loop. The actually-updated
   USB copy at `/media/zeph/projects/paper-trader/data/ml/
   decision_scorer.pkl` (n_train=400, mtime 21:42 today) is
   **invisible** to every consumer of `DecisionScorer()` — the
   live trader's gate, the backtest engine's `_get_decision_scorer`,
   `/api/scorer-predictions`, all diagnostics. Either path is
   below the 500 gate threshold so the conviction gate (#5) is
   **currently dormant**, but the divergence is a latent footgun if
   either crosses 500 first. Pattern matches invariant #15 (the
   articles.db split-brain) — could merit the same freshness-aware
   resolver eventually.
3. **MLP_ADDS_SKILL verdict appearing in recent cycles** — cycle 3
   ic_gap=+0.069, cycle 4 ic_gap=+0.113 — contradicts the much-
   documented `MLP_WORSE_THAN_TRIVIAL` finding from older review
   passes. The anti-overfit `(32, 16) + alpha=1e-2 + early_stopping`
   config (pass #19) appears to have flipped the verdict for cycles
   that train on it; the per-cycle baseline ledger now alternates
   between `MLP_ADDS_SKILL` (recent) and `MLP_NO_BETTER` (early). A
   reading quant should now treat the loop's gate output as
   **marginally useful** rather than reflexively dismissing it —
   though the n=4-cycle sample is too small to call it a regime
   change. `_oos_rank_metrics` corroborates: `oos_dir_acc` 0.52–0.55,
   `oos_ic` 0.09–0.11 across the 4 cycles.
4. **Recent backtest returns are leverage-dispersed, not skill.**
   Median +143% vs SPY over the last 20 complete runs (max +610%,
   min −53%, mean +175%) — heavy 3x-ETF cherry-picked windows.
   AGENTS.md already warns to read `vs_spy_pct` skeptically on
   leveraged windows; live data confirms the warning was correctly
   loaded. The permutation/label-audit validation suite remains
   the only real skill-vs-luck arbiter.
5. **Run 6243 stuck in `'running'` for 4h+** — orphaned by the loop
   death. `_reap_orphaned_runs` would have cleaned it on the next
   loop start (6h age threshold), but the loop isn't running. Two
   total `'running'` rows on disk (the other looks similarly stale).
6. **USB pkl was retrained at 21:42** (≈1h before this audit)
   **after the ledger stopped at 11:06** — so the gate state shifted
   without leaving an audit trail in `scorer_skill_log.jsonl`. The
   most likely cause is a one-shot `run_backtests.py` or manual
   `train_scorer` invocation that bypassed `_append_scorer_skill_log`
   (which is only wired into `run_continuous_backtests.main()`).
7. **24 failed runs (~5%) historically.** Acceptable given GDELT
   rate-limit transients and yfinance flakes visible in the log
   tail, but worth noting for future failure-rate trending.

None of these are quick safe fixes (the loop heartbeat and split-
brain are systemic, not surgical); reported here rather than
patched.

#### Tests for this domain

```bash
cd /home/zeph/trading-intelligence/paper-trader

# Canonical filter (421 tests, ~30s)
python3 -m pytest tests/ -k "ml or backtest or scorer or calibration or continuous or horizon" -q

# Just this pass's surface
python3 -m pytest tests/test_decision_scorer.py::TestSectorMapping -v

# Inspect a single scorer prediction (the read-only CLI added pass #N)
python3 -m paper_trader.ml.decision_scorer --explain --ticker LRCX --ml-score 2.5

# Cycle-trend ledgers (require the continuous loop to be running)
python3 -m paper_trader.ml.skill_trend
python3 -m paper_trader.ml.baseline_compare
python3 -m paper_trader.ml.horizon_audit
python3 -m paper_trader.ml.calibration --oos
python3 -m paper_trader.ml.deploy_audit
python3 -m paper_trader.ml.scorer_freshness
```

#### Invariants reaffirmed by this pass

* **Pickle schema is N_FEATURES = 10 + 7 = 17** (10 numeric + 7
  sector one-hot). The Phase 2 commit changes which sector a ticker
  hashes to but does not change the schema, so old pickles still
  load.
* **`build_features` is the single source of truth for the feature
  vector** — `_ml_decide`, `_compute_decision_outcomes`,
  `feature_contributions`, and `predict_with_meta` all consume its
  output. Adding new sector entries propagates everywhere without
  touching any other call site.
* **`INTENTIONALLY_OTHER` is the documentation-as-code answer to
  "why does this ticker map to sector_other?"** — without it, a
  reviewer could easily mistake a real coverage gap (the 41 silently-
  mapped tickers this pass fixed) for an intentional choice.


## Review pass #34 — paper-trader core hybrid (2026-05-19, ~07:15 UTC)

### Phase 1 — no fixes

Read every load-bearing core file in full
(`runner.py`, `reporter.py`, `signals.py`, `strategy.py`, `dashboard.py`,
`market.py`, `store.py`) plus the recently-touched `host_guard.py` and
`analytics/correlation.py`. The 423 baseline core tests passed cleanly.
A careful trace through the quota-latch flow, singleton-lock degraded
→ recovered transitions, `_window_delta` / `_movers` / `_realized_pl_today`
edge cases, `_extract_tickers` cashtag-vs-bare regex asymmetry, the
`_execute` BUY/SELL/option paths, and `_mark_to_market` stale flagging
turned up no actionable bug. The reporter additive contract held
end-to-end; the host-guard pulse → discord wiring was correct under load.
Per the Phase 1 commit guard, **no Phase 1 commit** — the diff was empty
after the read pass.

### Phase 2 (commit `a5b1d0b`) — SINGLE_NAME_RISK concentration alert in Discord

`/api/correlation` exposes the `SINGLE_NAME_RISK` verdict (top stock-book
weight ≥ `DOMINANT_WEIGHT` = 60%) on the *dashboard* and the
`risk_mirror` block surfaces the same fields to Opus in the *prompt*
(strategy.py `build_risk_mirror` call site). But the operator who lives
in Discord never saw this verdict directly — per-position weight %s in
`_portfolio_lines` exposed the raw number, but nothing flagged the
categorical "this is single-name risk" alarm. The live 2026-05-19 book
sat at NVDA **75%** of stock book — deep in SINGLE_NAME_RISK territory —
with **nothing in any hourly/daily Discord report** saying so. This pass
closed the same dashboard→Discord gap `_capital_pulse_line` /
`_host_pulse_line` / `_position_attention_line` each closed, one
dimension over (capital → host → per-position → name-concentration).

`reporter._concentration_line(store)` composes
`build_correlation` **verbatim** (single source of truth, invariant #10;
same builder `/api/correlation` uses). **Pure store reads only — NO
network** (the Discord-path discipline; `price_history` is intentionally
passed as `{}` so a per-position yfinance hop is never required, the
`_stress_line` / `_recovery_line` / `risk_mirror` no-history precedent).
Computes per-position `market_value` inline (option ×100; stale-mark
falls back to `avg_cost` so a yfinance outage doesn't silently halve the
apparent concentration). Wired into both `send_hourly_summary` and
`send_daily_close` after `_capital_pulse_line` and before
`_position_attention_line` — load-bearing order
HOST→CAPITAL→CONCENTRATION→ATTENTION→CLOCK (structural risks first,
drift signals after, the established #33 ordering).

**Suppression: surface ONLY SINGLE_NAME_RISK.** DIVERSIFIED / MODERATE /
INSUFFICIENT-with-low-top-weight / NO_DATA / empty-headline all stay
silent so a balanced book adds no hourly noise (the `_capital_pulse_line`
FREE-and-not-bleeding precedent — the summary must never become its own
lying green light). The per-position weights in `_portfolio_lines`
continue to show raw numbers regardless, so a non-SINGLE_NAME_RISK book
remains fully diagnosable from the existing lines. The SINGLE_NAME_RISK
threshold uses the builder's own `DOMINANT_WEIGHT` constant (0.60) so
the no-history and OK-state paths land on the same gate.

Failure contract mirrors the rest of `reporter`: any builder/store fault
degrades to `""` ("no concentration line this report"), **never** an
exception ("no Discord summary this report"). Locked by 17 tests in
`TestConcentrationLine`:

* state-ladder suppression (`DIVERSIFIED` / `MODERATE` /
  `NO_DATA` / `INSUFFICIENT`-below-threshold / unparseable-top-weight /
  non-dict / builder-fault all silent);
* OK-state SINGLE_NAME_RISK headline surfaced verbatim;
* INSUFFICIENT-with-dominant-weight surfaces via the weight-based
  fallback synthesis (the buried "verdict withheld" sentence never leaks
  through);
* empty-headline OK-state synthesises a body rather than dropping the
  alarm (defensive against a future builder regression);
* `market_value` uses ×100 multiplier on options;
* `market_value` falls back to `avg_cost` when current_price is the
  default 0 (the stale-mark scenario);
* end-to-end wiring in both `send_hourly_summary` and
  `send_daily_close`;
* whole summary still ships when the correlation builder raises (the
  reporter additive failure contract);
* **end-to-end against the REAL `build_correlation`** with a seeded
  75% NVDA store — this last test is the regression-catch for the
  signature-mismatch bug fixed in `dc49740` (see Phase 3 below).

### Phase 3 — live validation against the running trader (07:08–07:15 UTC)

1. ✅ **`/api/portfolio`** healthy — $1000 equity, NVDA 2 sh @ $222.35
   + TQQQ 2 sh @ $74.28. `stale_marks: 0`. Zero P/L is coincidence
   (market closed, both buys filled at the after-hours close print).
2. ✅ **`/api/runner-heartbeat`** `verdict: HEALTHY`, last decision 1h
   ago, lock acquired by PID 2544540, Discord notify HEALTHY. Decision
   efficacy reads `DEGRADED — 70% of the last 20 cycles were
   NO_DECISION` — expected during this 4-agent concurrent review.
3. ✅ **`/api/correlation`** confirms `SINGLE_NAME_RISK — NVDA is 75%
   of the book` — exactly the case my new feature surfaces in Discord
   on the next hourly cycle.
4. ⚠️ **`/api/host-guard`** `state: SATURATED`, 7 concurrent Opus
   (`>4`), 75.6% starvation rate. The 4-agent review IS the saturation;
   known [[pt-no-decision-host-saturation]]. Not a code bug.
5. ⚠️ **`/api/decision-drought`** `current_drought: PARALYSIS` 3.99h
   long, `alpha_pct: 0.0` (market closed, no alpha to bleed).
6. ⚠️ **`/api/build-info`** `behind: 5` commits. The live runner is on
   boot SHA `f3e3020`; HEAD moved 5 commits forward during this review
   round (sibling agents' commits + my own). The git-watcher's deferred
   restart will trigger on the next cycle boundary.
7. ⚠️ **`/api/supervision`** `verdict: UNSUPERVISED_STALE`, `orphan:
   true` (PPID 1), `systemd: "Failed to connect to bus: No medium
   found"`. The trader runs as a manual orphan; if its git-watcher /
   deadman fires an `os._exit(0)`, there is **no systemd safety net to
   bring it back up**. Known [[pt-systemd-vs-manual-restart-spam]] —
   the operator chose this configuration.
8. 🐛 **My own bug, caught by live validation:**
   `reporter._concentration_line` invoked `build_correlation(sized)`
   with one positional arg, but the builder requires `(positions,
   price_history)`. Every Phase-2 test passed because they all
   monkeypatched `build_correlation` with a lambda accepting `*a, **k`
   — so the `TypeError` was silently swallowed by the outer
   `try/except` and the Discord block dropped every cycle. Fixed in
   commit `dc49740` by (a) passing `price_history={}` (the no-network
   discipline) and (b) keying SINGLE_NAME_RISK off `top_weight_pct ≥
   DOMINANT_WEIGHT` directly so the no-history INSUFFICIENT path still
   surfaces the alarm (the `risk_mirror` weight-based-fallback
   precedent). Added 4 regression tests, including one that exercises
   the **real `build_correlation`** end-to-end against a fresh_store
   seeded to the live 75% NVDA shape — a future signature drift fails
   loudly in CI rather than silently dropping the Discord block.
   Verified live with
   `python3 -c "from paper_trader import reporter; from
   paper_trader.store import get_store; print(reporter._concentration_line(get_store()))"`
   → produced
   `⚠️ **CONCENTRATION** ◈ SINGLE_NAME_RISK\n> SINGLE_NAME_RISK — NVDA
   is 75% of a 2-name stock book — 1.6 effective name(s) by weight.`

### How to run / test

```bash
cd /home/zeph/trading-intelligence/paper-trader

# Sanity import + full focused core suite (~3s, deterministic, offline)
python3 -c "import sys; sys.path.insert(0, '.'); from paper_trader import signals, reporter, strategy; print('imports OK')"
python3 -m pytest tests/test_core_strategy.py tests/test_core_runner.py tests/test_core_reporter.py tests/test_core_signals.py tests/test_core_market.py tests/test_core_runner_cycle.py tests/test_risk_mirror.py tests/test_correlation.py -v

# Just this pass's surface
python3 -m pytest tests/test_core_reporter.py::TestConcentrationLine -v   # 17 tests

# Inspect the new line against the live store
python3 -c "from paper_trader import reporter; from paper_trader.store import get_store; print(reporter._concentration_line(get_store()))"

# Operator-level diagnostics still working
python3 -m paper_trader.should_restart    # OK / RESTART / OPS_ONLY / ERROR
python3 -m paper_trader.host_guard        # CLEAR / SATURATED + opus_count + load + swap
```

### Invariants reaffirmed by this pass

* **No `git add -A`** — sibling agents were running concurrently on the
  same tree; explicit pathspec on every commit
  ([[pt-concurrent-samerole-staging-race]] discipline). Both my commits
  (`a5b1d0b`, `dc49740`) staged ONLY `paper_trader/reporter.py` +
  `tests/test_core_reporter.py`; sibling work was correctly left out.
* **Reporter additive contract** — fault → `""`, never *no summary*.
  The `_concentration_line` `try/except` swallows builder faults; the
  whole hourly / daily-close still ships.
* **Single source of truth (invariant #10)** — the line composes
  `build_correlation` verbatim, never re-derives a top-weight number.
  Even the no-history weight-based fallback reads the SAME
  `top_weight_pct` / `top_weight_ticker` / `effective_positions_naive`
  / `n_stock_positions` fields the OK-headline reads from.
* **Discord-path no-network discipline** — passing `price_history={}`
  preserves the zero-yfinance-call guarantee every recent reporter line
  reaffirms (`_stress_line` / `_recovery_line` / `_capital_pulse_line`).
* **Mock-vs-real test discipline** — every monkeypatch'd test was
  joined by ONE real-builder end-to-end test (the
  `test_calls_real_build_correlation_with_correct_signature` regression
  lock). Phase-3 caught that the original Phase-2 tests had ALL been
  monkeypatched, masking a real signature mismatch; the new
  real-builder test ensures any future signature drift fails in CI
  rather than silently in production.

## Review pass #35 — paper-trader core hybrid (2026-05-19, ~08:00 UTC)

### Phase 1 — no fixes

Re-read every load-bearing core file in full: `runner.py`, `reporter.py`,
`signals.py`, `strategy.py`, `market.py`, `store.py`, plus the relevant
slice of `dashboard.py`. 418-test core baseline (`test_core_*` + the
already-existing parse-retry + ML opinion suites) green on entry; 521
tests green after the Phase 2 additions. A careful trace through
`runner._cycle` (quota latch / circuit breaker / restart-requested
event), `strategy.decide` (host-guard pre-flight + mid-call re-probe +
Sonnet fallback + JSON-only retry), `_ml_live_opinion` (news-sentiment
+ quant adjustments + regime + watch_px universe gate), and
`_concentration_line` (the pass-#34 addition, intact and live) turned
up no actionable bug. Per the Phase-1 commit guard, **no Phase 1
commit** — diff was empty after the read pass. `bugs_fixed = 0`.

### Phase 2 (commit `5a3cb32`) — lock 3 untested paths in `_ml_live_opinion`

`tests/test_ml_live_opinion.py` covers the documented news-key + tokenization
regressions and quant-only path. Three branches in the function had no
direct lock yet — added as the new `TestRegimeAndUniverseGuards` class:

* **`test_bear_regime_suppresses_borderline_buy`** — locks that the
  `regime_mult` (1.0 bull / 0.6 sideways / 0.3 bear, SPY 20d-momentum
  derived) is actually applied to the score, not just rendered in the
  reasoning label. RSI-25-only NVDA (`adj=+1.5`) buys in bull (1.5 > 1.0
  threshold) but HOLDs in bear (1.5 × 0.3 = 0.45 < 1.0) — the exact
  regime-gating contract.
* **`test_keyword_mapping_picks_up_unticked_article`** — an article with
  empty `.tickers` but `nvidia` / `chip` in the title must still drive
  a BUY via `_WORD_TO_TICKER_LIVE` fallback. This is the canonical
  value-add of the keyword map (extractor misses; keyword recovers); a
  silent regression here would re-blind the advisor to any article
  whose ticker extractor failed.
* **`test_unpriced_ticker_cannot_be_chosen`** — a ticker with
  `watch_px[tk] = None` or `0.0` must NOT be picked as best even when
  its sentiment+quant score is the highest. Without the `px and px > 0`
  guard the engine would emit a BUY for a name the trader cannot
  actually transact (yfinance dead / delisted / off-hours hole).

Staged ONLY `tests/test_ml_live_opinion.py` per
[[pt-concurrent-samerole-staging-race]] — concurrent sibling agents
were running with modified `paper_trader/dashboard.py` and untracked
`paper_trader/analytics/event_readiness.py` in the working tree, all
correctly left out. `features_added = 1` (3 tests in one regression-
locking class).

### Phase 3 — live validation against the running trader (08:10 UTC)

1. ✅ **`/api/build-info`** `boot_sha == head_sha == 5a3cb32`,
   `stale: false`, `behind: 0`. The git-watcher autorestart fired
   between my push and the read — the running trader is on the
   newly-pushed commit.
2. ✅ **`/api/portfolio`** healthy — $1000 equity, cash $406.74,
   NVDA + TQQQ open value $593.26, `stale_marks: 0`.
3. ⚠️ **`/api/host-guard`** `state: SATURATED`, **14 concurrent Opus
   (>4)**, swap **95.3%** (>90% threshold), `mem_available_mb=1535`,
   `load1=24.93`. **77.6% of the last 49 decisions never reached Opus**.
   The 4-agent concurrent hybrid review (this run + 3 siblings, ~14
   Opus subprocesses) IS the saturation. Known
   [[pt-no-decision-host-saturation]]; not a code bug.
4. ⚠️ **`/api/runner-heartbeat`** stuck `warming` (SWR cold) for >5s —
   secondary symptom of dashboard contention under the same host load.
   `/api/portfolio` (smaller payload) replied fine.
5. ⚠️ **`/api/concentration` 404** — concentration is exposed via
   `/api/correlation`, not a dedicated endpoint. Operator surface only,
   no functional impact (the reporter Discord line landed in pass #34
   reaches the operator regardless).

`user_findings = 3`: (a) the saturation pattern remains the dominant
NO_DECISION cause and is operator-resolvable but not code-resolvable;
(b) dashboard SWR cold-path response time degrades sharply under host
load — a 5-second pause to warm `/api/runner-heartbeat` looks like a
dead endpoint to a trader checking by hand; (c) the running trader
auto-deployed my own commit within a minute of `git push` — the
git-watcher deferred-restart path is healthy end-to-end.

### Invariants reaffirmed by this pass

* **No `git add -A`** — staged only my own file; sibling agents'
  `dashboard.py` / `digital-intern/AGENTS.md` / `event_readiness.py`
  changes were correctly left out of the commit. [[pt-concurrent-samerole-staging-race]].
* **Mature-codebase Phase-1 honesty** — read the full surface, found no
  actionable bug, set `bugs_fixed = 0` per the explicit guard rather
  than manufacturing churn. The honest zero IS the answer when the
  codebase is mature.
* **Edge-case test value-add** — the three new tests don't repeat any
  branch already covered; each locks a distinct path
  (regime / keyword-map / universe-gate) whose silent break would
  degrade the advisor in a different mode.

## ML / backtest review pass (Agent 2, 2026-05-19, 2nd pass)

Hybrid pass against `paper_trader/ml/decision_scorer.py`,
`paper_trader/backtest.py`, `run_continuous_backtests.py`. After the
prior pass on 2026-05-19 (`attribution_audit` + 9+15 new tests) the
diagnostic suite is genuinely saturated — this pass found two real
issues missed by every existing check.

### Bug fix: CONDEMN train-fold weight floor

`train_scorer`'s replication code used
`rep = np.maximum(1, np.round(w_tr * 2).astype(int))`. The `max(1, …)`
floor silently promoted every CONDEMN-annotated row
(`llm_quality_label=-1`, multiplier 0.1×) to rep=1 — the docs claim a
0.1× weight, the measured CONDEMN/unlabeled ratio was ~0.5×. With this
fix rows whose rounded rep is 0 are DROPPED from the training fold
entirely (not floored to 1), so CONDEMN's near-zero weight is realized
in practice. Defensive empty-fold fallback added (impossible in any real
corpus — unlabeled records always weight ≥0.5 → rep≥1 — but keeps
`train_scorer` total-failure-free).

Empirically inert today: 7413/7413 production outcomes carry
`llm_quality_label=0` because `_llm_annotate_outcomes` has failed every
cycle since deployment (no `ANTHROPIC_API_KEY` set — see Phase 3
findings). The fix is correctness-on-future-use: when the auth issue
gets resolved and CONDEMN labels start appearing, they will be down-
weighted as the design intended rather than indistinguishable from a
0.5× weight on a losing run.

Locked by `tests/test_decision_scorer.py::TestLlmWeightReplication`
(3 tests): ENDORSE replicates 3× more than unlabeled (`max(rep)==6`,
`min(rep)==2`); CONDEMN rows are dropped (no `rep==1` floor leakage);
unlabeled rep=2 (×2 scaling pinned). Keeps the ×2 scaling rather than
re-scaling to ×10 — a wider scaling would weaken the unweighted L2
`alpha` term and break the
`test_regularization_suppresses_pure_noise_memorization` guarantee.

### Feature: PKL_REGRESSED verdict in `scorer_freshness.py`

The existing scorer-freshness ladder
(FRESH/INSUFFICIENT_DATA/STALE_PKL/LOOP_STALLED/LOOP_DEAD) reports
`pkl_n_train` and `last_train_n` as INPUTS but no verdict was driven by
their relationship. Observed live (2026-05-19) production state shows
the gap: deployed pkl `n_train=400` while the loop's most recent
skill-log row logged `train_n=3959` — a ~10% corpus, and the gate WAS
acting on it (`gate_active=true`). STALE_PKL does not fire because the
pkl mtime is *newer* than the heartbeat — just clobbered with a tiny-
corpus fit by a side process (manual retrain / out-of-tree script /
agent test that bypassed conftest's `SCORER_PATH` isolation).

`PKL_REGRESSED` fires when:
  - heartbeat is fresh (LOOP_DEAD / LOOP_STALLED take precedence)
  - pkl mtime ≥ heartbeat (STALE_PKL doesn't fire)
  - `last_train_n ≥ PKL_REGRESSION_MIN_TRAIN_N=500` (mutes early-cycle wander)
  - `pkl_n_train < last_train_n × PKL_REGRESSION_TOL=0.5`

Run:
```bash
cd /home/zeph/trading-intelligence/paper-trader
python3 -m paper_trader.ml.scorer_freshness
```
Exit code mirrors siblings: 0 on FRESH/INSUFFICIENT_DATA, 2 on every
actionable verdict including PKL_REGRESSED.

Locked by `tests/test_scorer_freshness.py` (15 tests — module had **zero
direct test coverage** before this pass): full verdict ladder; CLI
exit-code contract; verdict precedence (LOOP_DEAD wins over
PKL_REGRESSED); below-floor mute; robustness against unparseable JSONL
lines and corrupt pkls.

### Phase 3 quant findings (worth reading)

1. **Production pkl is in a clobbered state.** Deployed
   `decision_scorer.pkl` reports `n_train=400` (mtime 2026-05-18 21:42)
   while the last skill-log row (cycle 4, 2026-05-18 18:06) reports
   `train_n=3959`. Calibration on that pkl: spearman=0.0157
   (MISCALIBRATED) — the gate has near-zero predictive value right now.
   The new PKL_REGRESSED verdict will catch this on the next loop restart
   (currently masked by LOOP_STALLED since heartbeat is 14h+ old).
2. **Continuous loop is currently STOPPED.** No `run_continuous_backtests`
   process; last heartbeat 14h+ stale. Two `backtest_runs` rows remain
   `status='running'` indefinitely. The new freshness check correctly
   identifies this as LOOP_STALLED, escalating with "the conviction
   gate is ACTIVE, so trades are being modulated against this frozen
   model".
3. **LLM annotation has never worked.** Every cycle in `continuous.log`
   shows `[continuous] LLM annotation failed: "Could not resolve
   authentication method..."` because no `ANTHROPIC_API_KEY` is set.
   All 7413 production outcomes carry `llm_quality_label=0` — the
   ENDORSE/CONDEMN weighting scheme is inert. The Phase 1 fix is
   ready-for-when-it-starts-working; the auth gap is out of scope.
4. **Effective dedup is ~7%**, not the suspected 95%. Dedup by
   `(ticker, sim_date, action)` on 7413 outcomes ⇒ 6906 unique
   (ratio 0.932); tail-5000 ⇒ 4943 unique. The pkl's `n_train=400`
   anomaly is NOT explained by dedup — confirms the clobber hypothesis.
5. **Recent baseline-skill trend is FLIPPING POSITIVE.**
   `baseline_skill_log.jsonl` cycles 1-4 show:
   `MLP_NO_BETTER_THAN_TRIVIAL → MLP_NO_BETTER_THAN_TRIVIAL →
   MLP_ADDS_SKILL (ic_gap=+0.07) → MLP_ADDS_SKILL (ic_gap=+0.11)`.
   The historical "MLP is worse than a one-line `ml_score` rule"
   finding may no longer hold post-anti-overfit retune (echoing the
   prior pass's #5 observation). The clobbered pkl currently in
   production interferes with this trend; restart-and-retrain would
   reveal the true state.
6. **Backtest dispersion remains extreme.** 475 complete runs;
   `avg(total_return)=150%`, `avg(vs_spy)=118%`; last 50 runs:
   `avg(total_return)=271%`, `avg(vs_spy)=213%`. This is leveraged-ETF
   beta dispersion through cherry-picked windows, NOT repeatable
   alpha (as AGENTS.md has long emphasized).

### Test commands for this domain

```bash
cd /home/zeph/trading-intelligence/paper-trader

# Just the ML / backtest / scorer / freshness scope (~5 sec)
python3 -m pytest tests/test_scorer_freshness.py \
                  tests/test_decision_scorer.py \
                  tests/test_backtest.py \
                  tests/test_continuous.py \
                  tests/test_ml_backtest_review.py \
                  tests/test_calibration.py \
                  tests/test_baseline_compare.py -v

# Just the new coverage from this pass (~3 sec)
python3 -m pytest tests/test_scorer_freshness.py \
                  tests/test_decision_scorer.py::TestLlmWeightReplication -v
```

### Invariants reaffirmed by this pass

- **Read-only diagnostic discipline** — `scorer_freshness` never trains,
  never touches the pickle, no `build_features` / `N_FEATURES` /
  trade-path mutation. Safe to run against the unattended loop.
- **Conftest SCORER_PATH isolation discipline** — the bug class the new
  PKL_REGRESSED verdict catches is precisely "a side process clobbered
  the production pkl by bypassing conftest's monkeypatch". The
  diagnostic exists as a runtime alarm; the test discipline remains the
  primary defense.
- **`train_scorer` honesty contract** — every failure mode (empty
  records, insufficient_after_dedup, null/non-finite forward returns,
  sklearn absent) degrades gracefully. The new CONDEMN-drop codepath
  preserves this with a defensive empty-fold fallback.


## Review pass #36 — paper-trader ML / backtest hybrid (2026-05-19, ~17:00 UTC)

### Phase 1 — no fixes

Read every load-bearing ML/backtest file in full
(`paper_trader/ml/decision_scorer.py`, `paper_trader/backtest.py`,
`run_continuous_backtests.py`) plus the surrounding diagnostic modules
(`scorer_freshness.py`, `deploy_audit.py`, `calibration.py`,
`baseline_compare.py`, `corpus_audit.py`, `feature_importance.py`,
`gate_audit.py`, `gate_pnl.py`, `skill_trend.py`, `validation.py`,
`persona_leaderboard.py`). Ran the targeted scorer/backtest/continuous
test sweep (170 tests pass in 9.7s; the full `-k "ml or backtest or
scorer"` slice produced 357 pass with one flaky `test_corrupt_pkl_does_not_crash`
that pyc-cache-clears and isolated runs both pass — pytest assertion-
rewriting cache mismatch, not a code bug). The codebase is unusually
mature: every actionable bug class I could identify was already pinned
by a focused test (CONDEMN-drop, FILLED-only training-integrity, dedup
collision under SELL sign-flip, scorer freshness ladder, OOS rank-IC
SELL sign-flip, atomic pickle write, load-cache key by mtime/size).
Per the Phase 1 commit guard, **no Phase 1 commit** — the diff was
empty after the read pass.

### Phase 2 (commit `740c67c`) — scorer_smoke_test diagnostic + 12 tests

`scorer_freshness` answers *is the loop still re-pickling*; `deploy_audit`
answers *does the pickled config match source*. Neither asks the basic
sub-second question: *does `DecisionScorer().predict_with_meta(...)`
return finite, non-degenerate values for a sweep of realistic inputs?*
A model that loaded successfully and matches source config but predicts
the same constant for every input — a degenerate predictor — passes
both existing diagnostics and silently disables the conviction gate at
the predict level (the ±10/±5/0 buckets collapse to one bucket forever).

New module `paper_trader/ml/scorer_smoke_test.py` (~250 LOC) closes that
gap: 8 in-distribution probes spanning the 7-way sector axis + 2 edge
probes (extreme RSI/momentum), each routed through the public
`predict_with_meta` so it describes the EXACT path the live `_ml_decide`
gate uses. Verdicts mirror the sibling ladder:

  * `HEALTHY`              all probes finite, ≥2 distinct buckets       → 0
  * `UNTRAINED`            `is_trained=False` (no pkl / load failed)    → 0
  * `DEGENERATE_CONSTANT`  every probe collapses to one prediction      → 2
  * `BROKEN_PREDICT`       any probe raised or returned non-finite      → 2

CLI: `python3 -m paper_trader.ml.scorer_smoke_test [--json]`. Exit code
mirrors `scorer_freshness._cli` semantics so a cron can branch on `$?`.
Same operational discipline as every sibling diagnostic — read-only,
no train, no pickle/`build_features`/`N_FEATURES`/trade-path mutation;
safe to run against the live unattended loop. Locked by 12 focused
tests in `tests/test_scorer_smoke_test.py` (verdict ladder against
faked scorers, JSON-safety, schema cardinality, CLI exit code).
Verified end-to-end against the deployed pickle: `HEALTHY` with 8/8
distinct predictions and 0/8 off-distribution false positives.

### Phase 3 quant findings — backtest.db live state (2026-05-19)

Using `backtest.db` (475 complete runs, latest 14h ago):

1. **Deployed scorer is critically thin (n_train=35).** The pickle on
   disk reports n_train=35 — far below the 500-threshold the gate
   engages at (invariant #5). The conviction gate is currently a no-op
   regardless of `baseline_compare`/`scorer_smoke_test` verdict.
   Either someone retrained with a tiny test corpus, or the production
   continuous loop hasn't run here long enough to accumulate outcomes.
   A skeptical quant should NOT trust the gate's modulation right now.
2. **Backtest dispersion remains extreme.** Mean total_return=150%,
   median=63%, stdev=267%, range -54%..+2979%. 22% (103/475) of runs
   have negative alpha vs SPY. This is leveraged-ETF beta dispersion
   through cherry-picked random windows, NOT repeatable alpha — the
   long-documented finding.
3. **Two stuck `status=running` orphan rows** (6238, 6243). The 6h-age
   reaper hasn't touched them yet — they are below the threshold.
   `_reap_orphaned_runs` will sweep them on the next mid-loop reap.
4. **17% of complete runs (80/475) had `spy_return_pct = 0.0` without
   the `benchmark_unavailable` note.** The existing guard only fires
   when the SPY series is COMPLETELY empty; runs where SPY exists but
   `returns_pct` returned exactly 0 over a multi-week window (price_on
   walk-back collapsed both endpoints to the same prior close) silently
   produced `vs_spy_pct = total_return` — a fabricated benchmark. SPY
   essentially never has a flat ≥30-day stretch, so a 0 there is the
   same degenerate case as the empty-series branch. **Fixed in commit
   `739e8f4`** (additive note, no behaviour change to returns / winner
   selection / live gate; 30-day threshold prevents false positives on
   short legitimately-flat windows). Locked by
   `tests/test_integration_backtest.py::TestBenchmarkUnavailableNote`
   (degenerate ≥30d window flagged; <30d boundary NOT flagged).
5. **Local environment lacks live training ledgers.**
   `data/decision_outcomes.jsonl`, `data/scorer_skill_log.jsonl`,
   `data/baseline_skill_log.jsonl`, `data/backtest_cache/` do not
   exist in this checkout. The continuous loop is either running
   elsewhere or has not been run here. The deployed pickle (n_train=35)
   is suspicious — it doesn't reflect a real training run through the
   standard `run_continuous_backtests.py` path.

### Test commands for this domain

```bash
cd /home/zeph/trading-intelligence/paper-trader

# Just the ML / backtest / scorer / smoke-test scope (~7 sec)
python3 -m pytest tests/test_scorer_freshness.py \
                  tests/test_scorer_smoke_test.py \
                  tests/test_decision_scorer.py \
                  tests/test_decision_scorer_attribution.py \
                  tests/test_backtest.py \
                  tests/test_continuous.py \
                  tests/test_integration_backtest.py \
                  tests/test_ml_backtest_review.py \
                  tests/test_calibration.py \
                  tests/test_baseline_compare.py -v

# Just the new coverage from this pass (~1 sec)
python3 -m pytest tests/test_scorer_smoke_test.py \
                  tests/test_integration_backtest.py::TestBenchmarkUnavailableNote -v

# Run the new diagnostic against the deployed pickle (sub-second)
python3 -m paper_trader.ml.scorer_smoke_test            # human-readable
python3 -m paper_trader.ml.scorer_smoke_test --json     # machine output
```

### How to interpret the scorer / backtest ladder

The full diagnostic ladder, in roughly the order an operator should run
on a freshly redeployed pickle:

1. `python3 -m paper_trader.ml.scorer_smoke_test` — sub-second. Verifies
   the pickle even produces finite, non-degenerate predictions for a
   canonical input sweep. Catches DEGENERATE_CONSTANT / BROKEN_PREDICT.
2. `python3 -m paper_trader.ml.scorer_freshness` — verifies the loop is
   still re-pickling and the on-disk pkl matches the heartbeat ledger
   (catches LOOP_STALLED / STALE_PKL / PKL_REGRESSED).
3. `python3 -m paper_trader.ml.deploy_audit` — verifies the deployed
   pickle's fitted-model architecture matches `MLP_CONFIG` in source
   (catches a stale pre-retune net that the loop hasn't retrained
   under the current hyper-params).
4. `python3 -m paper_trader.ml.calibration --oos` — does the model's
   predictions monotonically rank realized outcomes on the temporal
   OOS slice?
5. `python3 -m paper_trader.ml.baseline_compare` — does the 17-feature
   MLP earn its complexity vs a one-line rule? Verdict ladder
   `MLP_WORSE_THAN_TRIVIAL` / `MLP_NO_BETTER_THAN_TRIVIAL` / `MLP_ADDS_SKILL`.
6. `python3 -m paper_trader.ml.gate_audit` — is the conviction gate's
   ±10/±5/0 bucketing actually buying realized edge?
7. `python3 -m paper_trader.ml.feature_importance` — permutation
   importance per feature. Catches the "model overfits to the sector
   one-hot, ignores quant signals" failure mode.

### Manual backtest run

```bash
cd /home/zeph/trading-intelligence/paper-trader

# One-shot 10-run sweep over the default window (legacy launcher)
python3 run_backtests.py

# Long-lived continuous loop (5 runs/cycle, retrain after every cycle)
python3 run_continuous_backtests.py
# Tails continuously; SIGTERM/SIGINT exits cleanly between cycles.
```

### Interpreting backtest results

Per CLAUDE.md §11 and AGENTS.md's longstanding caveats, **do not read
top-line `total_return_pct` as alpha**. The recurring extreme dispersion
(2979% top return, median 63%, stdev 267%) is leveraged-ETF beta through
cherry-picked random windows, not repeatable strategy edge. The
authoritative skill arbiters are:

- `vs_spy_pct` — alpha column, but ONLY meaningful when the
  `benchmark_unavailable` note is absent (after commit `739e8f4` this
  now also catches degenerate spy_return=0 cases).
- `baseline_compare` and `calibration --oos` — OOS rank skill of the
  scorer itself.
- `gate_audit` — realized PnL effect of the conviction gate's actual
  ±10/±5/0 buckets, not the (in-sample-flattering) calibration plot.

### Invariants reaffirmed by this pass

- **Read-only diagnostic discipline** — `scorer_smoke_test` never
  trains, never touches the pickle, no `build_features` / `N_FEATURES` /
  trade-path mutation. Safe to run against the unattended loop.
- **Verdict-ladder discipline** — every CLI in `paper_trader/ml/`
  emits a verdict from a fixed public `VERDICTS` tuple, exit 0 on
  benign/insufficient-data, exit 2 on actionable failure. The new
  `scorer_smoke_test` joins that ladder; tests pin the membership.
- **Benchmark honesty (additive notes)** — `run_one` now flags BOTH
  empty-SPY-series AND degenerate-walk-back spy_return=0 cases via
  the same `notes` column. No behaviour change to returns / winner
  selection / live gate; purely informational.

### 2026-05-19 feat (Agent 4 product-engineer pass) — `/api/suggestion-impact`

`/api/suggestions` ranks BUY / ADD / TRIM / EXIT / WATCH ideas but is silent
on the **operational consequence** — a BUY of MU at the default 5%-of-equity
sizing might tip `concentration_top1` past the 40% MEDIUM threshold (the same
threshold `/api/risk` displays) or burn the last cash on hand. The trader
then has to open `/api/risk` *and* `/api/portfolio` and project the impact
mentally before each act. This endpoint does that projection deterministically.

Pure builder `paper_trader.dashboard.build_suggestion_impact` augments each
suggestion with the per-trade projection (each act treated INDEPENDENTLY —
"if I take THIS idea ALONE", which IS the trader's actual decision unit):

- **BUY / ADD**: `projected_size_usd` (default 5% of `total_value`, capped
  at available cash for BUYs with `cash_constrained=true` when capped),
  `projected_qty` (`size / price`), `projected_cash_after`,
  `projected_position_pct_after`, `projected_top1_pct_after`,
  `projected_top3_pct_after`, `projected_severity_after`,
  `would_overconcentrate` (severity LOW → ≥MEDIUM after the act)
- **TRIM (default 50%) / EXIT (100%)**: `projected_proceeds_usd`,
  `projected_realized_pnl_usd` (`(current_price - avg_cost) × qty_sold ×
  multiplier`), `projected_cash_after`, `projected_position_pct_after`,
  `projected_top1_pct_after`, `projected_top3_pct_after`,
  `projected_severity_after`, `frees_concentration` (severity ≥MEDIUM → LOW
  after the act, or HIGH → MEDIUM)
- **HOLD / WATCH**: pass-through with `would_act: false` (no projection
  fields). The baseline still surfaces so the UI doesn't have to merge two
  endpoints.

**SSOT — single concentration taxonomy.** The projected severity comes from
the SAME `_concentration_severity` helper `/api/risk` uses; the per-position
rows come from the SAME `_classify` + `_LEVERAGE_BETA` constants. A trader
who reads "projected HIGH after BUY" here sees the same `concentration_warning`
on `/api/risk` after the act. No second threshold table to drift.

**Route** `/api/suggestion-impact?size_pct=5` (clamped 0..100, default 5.0).
SWR-cached 45s (matches the `/api/suggestions` cache lifecycle — the impact
view re-derives only when suggestions do). Re-uses `suggestions_api()`'s JSON
output as input so the contract follows automatically if the suggestions
shape evolves.

**Locks (`tests/test_suggestion_impact.py`, 17 tests, ~1.6s):**
  1. Empty / non-list inputs collapse to a well-formed envelope (never raise)
  2. `size_pct` parameter overrides the default 5% sizing
  3. BUY uses 5%-of-equity sizing when cash is sufficient
  4. BUY caps at available cash and flags `cash_constrained`
  5. BUY pushes severity through the SAME `_concentration_severity` buckets
     `/api/risk` displays (LOW → MEDIUM via the 40%-top1 threshold)
  6. BUY into an unheld ticker creates a new row in the projection
  7. EXIT projects full liquidation; realized P/L = `(current - avg_cost) × qty`
  8. TRIM defaults to 50% (matches the `_SUGGESTION_TRIM_FRACTION` constant)
  9. `frees_concentration` fires when EXIT drops severity HIGH → LOW
 10. Realized P/L is negative when current < avg_cost
 11. HOLD / WATCH pass through with `would_act: false` and no projection keys
 12. SSOT proof: projected severity through the builder agrees with calling
     `_concentration_severity` directly
 13. Flask route exists, clamps `size_pct` to 0..100, tolerates garbage

**Live evidence at rollout:** with NVDA at 44.47% of book (MEDIUM severity)
and $406.74 cash, a BUY card for MU at default 5% sizing ($50 → 0.07 shares
at $681) projects severity unchanged at MEDIUM (top1 stays NVDA, top3 barely
moves), so the trader can act without re-opening the risk panel.

Advisory only — never gates Opus, never sizes an actual trade, never enters
the decision prompt (invariants #2 / #12). **No UI card yet** (consumers
query the route; natural home is the live trader page's existing
suggestions panel as a per-card expand).


### 2026-05-19 review pass (paper-trader core hybrid · `closed_positions` realized-P/L correctness · post-trade summary stats · live findings)

`Store.closed_positions` (the data source for `/api/closed-positions` and
the `closed-positions` summary card) was systematically computing the
wrong realized P/L for **every** closed lot — but the bug was silent
because no trade had closed since the endpoint was added. Two distinct
failures stacked:

1. **The opening BUY was always excluded from the window.** The SQL
   filter was `timestamp >= opened_at`, but `record_trade` runs *before*
   `upsert_position` in `strategy._execute`, so the opening BUY's trade
   timestamp lands a few microseconds *before* the position row's
   `opened_at`. Observed in the live DB:
   - NVDA trade ts `02:56:34.147236`, position opened_at `.150387` → 3.15 ms gap
   - TQQQ trade ts `00:42:15.202798`, position opened_at `.203088` → 290 µs gap
   The window therefore skipped every opening BUY and `realized_pl`
   reported only the gross SELL proceeds with no cost deducted — wildly
   overstating every winner and silently zeroing every breakeven.

2. **The action filter missed option round-trips.** The exact-match list
   `("SELL","CLOSE","SELL_TO_CLOSE")` / `("BUY","OPEN","BUY_TO_OPEN")`
   never matched the live trader's `BUY_CALL` / `BUY_PUT` / `SELL_CALL`
   / `SELL_PUT` actions. Even when (1) was fixed, every option close
   would have read **$0 realized** regardless of strike or exit price.

The new `closed_positions` walks every trade for the lot's
`(ticker, type, expiry, strike)` key chronologically and identifies the
round-trip whose close lands at or before `closed_at` (held qty starts
at 0, BUYs add, SELLs subtract, every return-to-≈0 closes a round-trip
slice — the same `round_trips.py` walk pattern, single source of truth
#10 in spirit). The slice's BUY/SELL legs are summed with
`startswith("BUY")` / `startswith("SELL")` so every documented action
contributes, and the per-lot output now additionally carries `cost`,
`proceeds`, `realized_pl_pct`, `hold_seconds`, `hold_days`.

**Feature complement** (`/api/closed-positions` summary, the natural
post-mortem panel a trader reads after the bell):

- `total_cost` / `total_proceeds` — gross BUY / SELL dollar flow
- `avg_realized_pl_pct` — **cost-weighted** realized / cost across the
  slice. A simple mean of per-lot percentages would over-weight a $10
  lot up 100% against a $1000 lot down 5% (the test
  `test_cost_weighted_avg_differs_from_simple_mean` pins exactly this:
  a real-dollar mixed slice that the cost-weighted figure reports as
  NEGATIVE while a naive mean would report +47.5%)
- `avg_winner_pct` / `avg_loser_pct` — un-weighted mean inside each
  bucket so both edges of the payoff ratio remain visible (None when
  the bucket is empty)
- `median_hold_days` — median per-lot hold_days, None when no lot has
  a parseable opened_at/closed_at pair

Observational only — never gates Opus, never injected into the decision
prompt, no caps (invariants #2/#12 — the existing `/api/closed-positions`
contract). Pure builder; any unparseable timestamp degrades to None /
zero and never raises (the `_hold_duration` precedent).

**Locks**:
- `tests/test_core_store.py::TestClosedPositionsRealizedPL` (9 tests):
  stock winner, stock loser, option round-trip pinning ($0 regression),
  put round-trip losing, partial-then-full close, `hold_duration`
  helper round-trip + bad-input degrade-safe, summary newest-first
  ordering
- `tests/test_core_dashboard_portfolio.py::TestClosedPositionsApiSummary`
  (5 tests): empty-book null aggregates, one-winner-one-loser exact
  W/L/pct, per-lot `realized_pl_pct` present, cost-weighted vs simple-
  mean divergence, median_hold_days present + non-negative

**Live findings (this pass)**:
1. `closed_positions` realized P/L was wrong for every closed lot —
   bug surfaced via the lack of any test that exercised the documented
   action filter against the live trader's actual action strings
   (BUY_CALL/SELL_CALL never matched). The fix lands the new
   walk-trades-chronologically logic with regression tests.
2. The live runner is **UNSUPERVISED** at the moment of this pass —
   `/api/supervision` reports `verdict: UNSUPERVISED, orphan: true,
   ppid: 1`. Pre-existing operational state (not a code bug); a clean
   exit (git-watcher restart / deadman) would leave the trader DOWN.
   Recommendation lifted from `/api/supervision`: `systemctl --user
   enable --now paper-trader`.
3. The `git-watcher` self-restart path is **verified working live**
   during this pass — the runner booted on `e594450` and auto-restarted
   onto my `15bee44` commit (current `head_sha`), serving the new
   summary fields within ~60s of the push. The auto-deploy contract
   documented in `runner.py` is intact.

Applies to existing data immediately — no schema change. `/api/closed-
positions` was already gated behind `?limit=N`; new fields are purely
additive.


## ML / backtest review pass (Agent 2, 2026-05-19, 3rd pass) — gate-bucket diversity

Hybrid pass against `paper_trader/ml/decision_scorer.py`,
`paper_trader/backtest.py`, `run_continuous_backtests.py` after the prior
two 2026-05-19 passes already saturated the bug-fix surface. No new bug
fixes were warranted this cycle (Phase 1 commit guard cleanly skipped);
the value of this pass is in the smoke-test feature add plus 7 Phase 3
quant findings.

### Feature: `GATE_BUCKETS_DEGENERATE` verdict + per-bucket histogram in `scorer_smoke_test.py`

The existing `DEGENERATE_CONSTANT` verdict compares predictions at
`_CONSTANT_TOLERANCE_PCT = 1e-4` raw-magnitude tolerance — a scorer
producing 8 distinct predictions of 0.5%, 1.2%, 2.0%, 4.5% etc. (all
inside the conviction gate's neutral `[0, 5]` arm) cleanly clears that
check and verdicted HEALTHY despite **the conviction gate being
operationally dormant on those inputs**. Every BUY through that
deployed model ends up with the same `×1.0` multiplier regardless of
the model's "prediction" because none cross the gate's `±10 / ±5 / 0`
thresholds. This is the same failure pattern the prior `n_train=400`
clobber documented in review pass #2 produced live (calibration
spearman=0.0157 / MISCALIBRATED, gate near-zero predictive value).

`_gate_bucket(pred: float) -> str` is the lockstep mirror of
`_ml_decide`'s four-arm ladder (`paper_trader/backtest.py`):

| pred range            | label              | gate multiplier |
|-----------------------|--------------------|----------------:|
| `p < -10`             | `strong_headwind`  | `×0.6`          |
| `-10 ≤ p < 0`         | `mild_headwind`    | `×0.85`         |
| `0 ≤ p ≤ 5`           | `neutral`          | `×1.0` (no-op)  |
| `5 < p ≤ 10`          | `mild_tailwind`    | `×1.15`         |
| `p > 10`              | `strong_tailwind`  | `×1.3`          |

The mirror is pure / total / NaN-safe (a non-finite or non-numeric
prediction falls through to `neutral`, the no-op arm — a diagnostic
crash can never propagate from this helper to a verdict) and uses
strict comparators that exactly match `_ml_decide`'s `< -10` / `< 0` /
`> 10` / `> 5` chain at every boundary.

The smoke report gains:
- `gate_bucket_counts: {strong_headwind:int, mild_headwind:int,
  neutral:int, mild_tailwind:int, strong_tailwind:int}` — every
  documented arm pre-populated to `0` so dashboard / Discord templates
  never `KeyError` on an absent arm.
- `distinct_gate_buckets: int` — the count of arms with at least one
  probe.

Verdict precedence: `BROKEN_PREDICT` > `DEGENERATE_CONSTANT` >
`GATE_BUCKETS_DEGENERATE` > `HEALTHY`. The constant check (a strictly
stronger fail than gate-bucket collapse) takes precedence so an
operator reading "DEGENERATE_CONSTANT" knows it is also a gate-collapse
case, but a "GATE_BUCKETS_DEGENERATE" specifically tells them the model
varies but the gate is dormant — different actionable surfaces.

CLI:
```bash
cd /home/zeph/trading-intelligence/paper-trader
python3 -m paper_trader.ml.scorer_smoke_test
```
Now prints the bucket histogram inline (`strong_headwind=0
mild_headwind=2 neutral=5 mild_tailwind=1 strong_tailwind=0` for the
deployed `n_train=35` pkl at the time of this pass) and a per-probe
`gate=<arm>` label. Exit code 2 on `GATE_BUCKETS_DEGENERATE` mirrors
the other actionable verdicts (cron contract preserved).

Locked by `tests/test_scorer_smoke_test.py` (10 new tests, 24 total,
~1.4s):

`TestGateBucket` (6 tests) — pins every arm boundary:
- `strong_headwind` strictly below `-10` (boundary `-10.0` falls into
  `mild_headwind` per `_ml_decide`'s `< -10` strictness)
- `mild_headwind` strictly negative (boundary `0.0` is `neutral` per
  `_ml_decide`'s `< 0` strictness)
- `neutral` INCLUSIVE on both ends of `[0, 5]` (matches `_ml_decide`'s
  branchless fall-through chain)
- `mild_tailwind` strictly above `5` (boundary `10.0` stays in
  `mild_tailwind` per `_ml_decide`'s `> 10` strictness)
- `strong_tailwind` strictly above `10`
- NaN / `None` / non-numeric inputs fall through to `neutral`

`TestGateBucketsDegenerate` (6 tests):
- Neutral-only `_NeutralBucketScorer` (8 distinct preds, all in
  `[0.5, 4.5]`) verdicts `GATE_BUCKETS_DEGENERATE` — the exact failure
  pattern review pass #2 catches via `PKL_REGRESSED` AFTER the loop
  restarts; this verdict catches it BEFORE.
- Two-bucket boundary (exactly 2 arms populated) still HEALTHY (≥ 2,
  not > 2 — inclusive boundary).
- `gate_bucket_counts` includes every documented bucket key (zeros
  filled in for absent arms; JSON-schema lock).
- Sum across all bucket counts equals `n_probes` — no probe lost,
  none double-counted.
- DEGENERATE_CONSTANT takes precedence over GATE_BUCKETS_DEGENERATE
  (the stronger fail wins for diagnostic precision).
- Off-distribution scorer with predictions in ONE bucket still
  verdicts GATE_BUCKETS_DEGENERATE (off-dist alone doesn't save it).
- CLI exit code 2 for GATE_BUCKETS_DEGENERATE (cron contract).

`TestSchema::test_module_level_constants_are_stable` updated to lock
the 5-verdict cardinality (was 4) and the `_GATE_BUCKETS` tuple
ordering (headwind → neutral → tailwind, matching the `_ml_decide`
ladder for left-to-right histogram rendering).

### Phase 3 quant findings (worth reading)

1. **Continuous loop is STOPPED.** Last skill-log heartbeat
   `2026-05-18T18:06:04+00:00` (cycle 4), `continuous.log` mtime
   `2026-05-18 12:03 PDT`. The loop has been dead ~14h at the time of
   this pass. 2 `backtest_runs` rows remain `status='running'` (6238,
   6243) — the per-cycle reap-on-startup will clean them when the loop
   restarts (`_reap_orphaned_runs` with 6h age guard).
2. **The deployed pickle is in PKL_REGRESSED state.** Live read:
   `n_train=35` in `data/ml/decision_scorer.pkl`. The last skill-log
   row (cycle 4) recorded `train_n=3959`. Ratio is `0.009` — well
   below the `PKL_REGRESSION_TOL=0.5` threshold review pass #2 added
   detection for. The `_ml_decide` conviction gate is INACTIVE
   (`_scorer_n >= 500` is False) so live trading is unaffected; the
   smoke test's new verdict ladder would correctly verdict HEALTHY on
   the *predictions* (3/5 gate buckets populated, sane spread) but the
   `n_train=35` makes the gate operationally inert regardless.
3. **Production data directories are SPLIT across two on-disk
   locations.** The continuous loop writes JSONL ledgers to
   `/media/zeph/projects/paper-trader/data/` (USB mount). The current
   working directory `/home/zeph/trading-intelligence/paper-trader/`
   (the monorepo path) has `backtest.db` symlinked to USB but **no
   symlink for `data/`** — so `decision_outcomes.jsonl`,
   `winner_training.jsonl`, `scorer_skill_log.jsonl`, and
   `baseline_skill_log.jsonl` are NOT visible from this working tree.
   Any operator-run diagnostic from the monorepo path
   (`calibration` / `baseline_compare` / `scorer_freshness`) returns
   "no outcomes file" / `INSUFFICIENT_DATA` despite the production
   data existing on USB. This is an out-of-scope deployment issue;
   diagnostics themselves are correct, the path mismatch is the bug.
4. **Stale code on USB.** `/media/zeph/projects/paper-trader/`
   (where the production loop runs) has `paper_trader/ml/
   decision_scorer.py` mtime `2026-05-17 00:36`, while the monorepo
   path has it at `2026-05-19 00:48`. The production loop runs ~2-day-
   stale code. Out-of-scope deployment issue; flagged here so a future
   restart picks up the current head.
5. **Documented OOS skill plateau is stable.** Last four skill-log
   rows (cycle 1-4 of the most recent loop instance):
   `val_rmse ∈ {7.3, 12.9, 11.4, 10.9}`, `oos_rmse ∈ {11.0, 11.8,
   12.7, 14.1}`, `oos_dir_acc ∈ {0.52, 0.53, 0.52, 0.55}`, `oos_ic ∈
   {0.08, 0.09, 0.09, 0.11}`. The mild val<<oos gap is the documented
   modest-overfit; rank-IC of 0.08-0.11 / dir-acc 0.52-0.55 is barely
   above coin-flip, consistent with prior passes' "modest but real
   skill" reading. `gate_active=true` and `deploy_stale=false` on every
   row — when the loop was running, the gate was being trusted on a
   net that matches source config.
6. **Smoke test against the deployed (clobbered) pickle: HEALTHY,
   3/5 gate buckets populated.** Predictions span `-9.4%` (UNH
   healthcare_weak) to `+5.3%` (AAPL tech_overbought), all
   in-distribution (no probe clamped). No probe lands in
   `strong_headwind` (`< -10%`) or `strong_tailwind` (`> +10%`) — even
   the extreme-overbought edge probe predicts modestly. This is a
   genuinely useful operator signal added by this pass: the gate's
   tail arms are dormant on typical inputs to the deployed model.
7. **No bugs warranted Phase 1 fixes.** The prior two 2026-05-19
   review passes (CONDEMN floor + `PKL_REGRESSED`) saturated the
   bug-fix surface; the remaining failure-mode gap was the gate-bucket
   diversity check shipped above.

### Test commands for this domain

```bash
cd /home/zeph/trading-intelligence/paper-trader

# Just the ML / backtest / scorer / freshness scope (~5 sec)
python3 -m pytest tests/test_scorer_freshness.py \
                  tests/test_scorer_smoke_test.py \
                  tests/test_decision_scorer.py \
                  tests/test_backtest.py \
                  tests/test_continuous.py \
                  tests/test_ml_backtest_review.py \
                  tests/test_calibration.py \
                  tests/test_baseline_compare.py -v

# Just this pass's new coverage (~1.4s)
python3 -m pytest tests/test_scorer_smoke_test.py::TestGateBucket \
                  tests/test_scorer_smoke_test.py::TestGateBucketsDegenerate -v

# Inspect the deployed pickle interactively (read-only)
python3 -m paper_trader.ml.scorer_smoke_test          # table view + gate buckets
python3 -m paper_trader.ml.scorer_smoke_test --json   # machine-readable
```

### Invariants reaffirmed by this pass

- **`_gate_bucket` lockstep with `_ml_decide`.** A drift between the
  two would silently produce wrong `GATE_BUCKETS_DEGENERATE` verdicts;
  the boundary tests (every comparator pinned) catch it.
- **Read-only diagnostic discipline.** `scorer_smoke_test` still
  never trains, never touches the pickle, no `build_features` /
  `N_FEATURES` / trade-path mutation. The new bucket histogram is
  pure post-processing of predictions already produced by the
  existing probe sweep.
- **Schema additivity.** Existing `gate_bucket_counts` /
  `distinct_gate_buckets` consumers see no JSON-schema break; older
  callers that just read `verdict` / `probes` are byte-unaffected.


## ML / backtest review pass (Agent 2, 2026-05-19, 4th pass) — leveraged-vs-non-leveraged scorer-skill audit

Hybrid pass against `paper_trader/ml/decision_scorer.py`,
`paper_trader/backtest.py`, `run_continuous_backtests.py` after the
prior three 2026-05-19 passes saturated the bug-fix surface (CONDEMN
floor + `PKL_REGRESSED` + `GATE_BUCKETS_DEGENERATE` already shipped).
Phase 1 commit guard cleanly skipped this cycle (no concrete bugs);
value is in the new `leveraged_skill` diagnostic plus 7 Phase 3 quant
findings.

### Phase 2 (commit `7abd09a`) — `leveraged_skill.py` per-bucket OOS skill audit

The skeptical-quant gap left open by every existing OOS audit module.
Production `data/decision_outcomes.jsonl` is empirically **35.3%
leveraged ETFs** — SOXL alone = 12.6%, TQQQ = 9.6%. The live BUY
conviction-gate (`_ml_decide` line ~1655) acts on the SAME
`predict()` output for both classes through one ladder (±10/±5/0).
If the scorer's apparent `oos_ic` is carried by leveraged-ETF beta
amplification (a strong-trending bull window predictably pushes
TQQQ/SOXL up 5–15% in 5 trading days regardless of news/quant
features), then gating on the SAME predictions for a non-leveraged
BUY is gating on noise.

`sector_skill` cannot answer this question — it groups SOXL, TQQQ,
NVDA, AAPL all under "tech", collapsing the leveraged ETFs and their
underliers into one bucket whose IC is dominated by the leveraged
tail. `persona_skill` is orthogonal (persona ≠ leverage class).
`baseline_compare` is silent on subset breakdown.

`leveraged_skill` splits the OOS slice strictly on `ticker IN
_LEVERAGED_ETFS` (the SAME constant `_ml_decide` uses for the
elevated 40% conviction cap, imported — not redefined — so a future
edit to that set shifts every consumer in lockstep) and reports
per-bucket `n_train`, `n_oos`, `mean_pred`, `mean_realized`,
`magnitude_bias`, `rmse`, `dir_acc`, `rank_ic`. Verdict ladder:

| Verdict | Trigger |
|---|---|
| `INSUFFICIENT_DATA` | < `MIN_RECORDS=30` aligned OOS rows total |
| `SCORER_UNTRAINED` | `is_trained=False` |
| `HAS_INVERTED_BUCKET` | any bucket `rank_ic ≤ -IC_GOOD=0.15` |
| `LEVERAGED_ONLY_EDGE` | leveraged ≥ SIGNAL_EDGE AND non-leveraged < WEAK_SIGNAL_EDGE |
| `NONLEVERAGED_ONLY_EDGE` | non-leveraged ≥ SIGNAL_EDGE AND leveraged < WEAK_SIGNAL_EDGE |
| `LEVERAGED_DOMINATES` | both have edge, `(ic_lev − ic_non) ≥ IC_DOMINANCE_GAP=0.10` |
| `BALANCED_EDGE` | both have edge, gap < `IC_DOMINANCE_GAP` |
| `NO_EDGE` | neither bucket reaches `WEAK_SIGNAL_EDGE` |

Per-bucket ladder (`SPARSE` / `INVERTED_SIGNAL` / `SIGNAL_EDGE` /
`WEAK_SIGNAL_EDGE` / `NO_SIGNAL_EDGE`) mirrors `sector_skill`'s
exactly — thresholds (`IC_MIN=0.05`, `IC_GOOD=0.15`,
`MIN_OUTCOMES_PER_BUCKET=20`) are intentionally aligned so a quant
reading both diagnostics sees one consistent ladder.

CLI: `python3 -m paper_trader.ml.leveraged_skill [--json]`.
Exit-code contract:
- `0` — `BALANCED_EDGE` / `NONLEVERAGED_ONLY_EDGE` / `NO_EDGE` /
  `LEVERAGED_DOMINATES` / `LEVERAGED_ONLY_EDGE` / `INSUFFICIENT_DATA`
- `1` — `SCORER_UNTRAINED` / other recoverable error
- `2` — `HAS_INVERTED_BUCKET` (cron-actionable; the gate is harmful
  in that bucket)

Reuses `validation.split_outcomes_temporal` (same temporal holdout
the scorer-skill ledger uses) and `calibration._spearman` (tie-aware,
load-bearing because `PRED_CLAMP_PCT` ties off-distribution preds at
±50). Read-only, never raises, never touches `decision_scorer.pkl` /
`build_features` / `N_FEATURES` / trade path — safe under the live
unattended continuous loop.

Locked by `tests/test_leveraged_skill.py` (34 tests, ~0.8s):
- `TestBucketOf` (4 tests) — leveraged/non assignment, lowercase
  normalisation, None/empty
- `TestVerdictForBucket` (5 tests) — every per-bucket boundary pinned
- `TestAlignedOosPair` (4 tests) — missing fwd_5d / NaN / SELL
  sign-flip / scorer exception
- `TestBucketMetrics` (4 tests) — perfect/zero correlation, magnitude_bias
- `TestVerdictsTuple` + `TestSchema` (3 tests) — VERDICTS cardinality
  lock (8 verdicts), JSON-safety
- `TestOverallVerdicts` (7 tests) — every overall verdict reached
  exactly once on staged synthetic data
- `TestLeveragedDominates` (1 test) — IC gap exceeds threshold case
- `TestTrainCountSurfacing` (1 test) — per-bucket `n_train` exposed
- `TestAnalyzeCli` (5 tests) — analyze degrade-safe + CLI JSON/table
  + exit-code 2 on `HAS_INVERTED_BUCKET`

### Phase 3 quant findings (worth reading)

1. **Continuous loop is STOPPED.** Last `scorer_skill_log.jsonl`
   heartbeat `2026-05-18T18:06:04+00:00` (cycle 4), `continuous.log`
   mtime `2026-05-17`. Loop dead ≥14h at time of this pass. Same
   state the prior pass observed; no recovery between. 2 stuck
   `status='running'` rows on USB backtest.db (6238 from 14:17,
   6243 from 18:45); 6238 is >6h old so the per-cycle reaper will
   sweep it on restart.
2. **Deployed pickle remains in PKL_REGRESSED state.** Local read:
   `n_train=35` in `data/ml/decision_scorer.pkl`. Last skill-log
   row (cycle 4 of the most recent loop instance) recorded
   `train_n=3959`. Ratio `0.009` ≪ `PKL_REGRESSION_TOL=0.5` — the
   gate is OPERATIONALLY INERT (`_scorer_n >= 500` False); live
   trading unaffected. Same critical-thin state two prior passes
   documented.
3. **Outcomes corpus is 35.3% leveraged ETFs.** SOXL=12.6%,
   TQQQ=9.6%, UPRO/TECL/FNGU/CURE/LABU/NAIL/DPST/FAS the rest.
   This is exactly the asymmetry the new `leveraged_skill`
   diagnostic exists to surface: the scorer is trained heavily on
   leveraged-ETF forward returns whose 5d distribution has very
   different statistical properties than the single-name underliers.
4. **`leveraged_skill` on a 1500-row subsample of the production
   corpus surfaces HAS_INVERTED_BUCKET against the deployed (n=35
   clobbered) pickle.** `nonleveraged` rank_ic=+0.181 SIGNAL_EDGE,
   `leveraged` rank_ic=-0.205 INVERTED_SIGNAL — the deployed
   tiny-model's predictions are anti-predictive on leveraged ETFs
   AND weakly skilled on non-leveraged names. Of course `n_train=35`
   < 500 so the gate is inactive; this demonstrates the diagnostic
   reads a genuine asymmetry the headline `oos_ic` cannot show. The
   verdict on the FULL n=3959 production pickle will likely differ
   — re-run when the loop restarts and lifts the deployed model
   back above the gate threshold.
5. **Historical gate-decision capture: 5165/7413 rows (69.7%) carry
   a non-null `gate_scorer_pred` from the loop's then-deployed
   model.** Mean=+1.72%, stdev=7.31%. Bucket histogram:
   `strong_headwind=4.3% | mild_headwind=29.7% | neutral=43.3% |
   mild_tailwind=14.5% | strong_tailwind=8.3%`. The conviction
   modulation was historically weighted toward the headwind side
   (34.0% headwind vs 22.8% tailwind), so the gate cut conviction
   more often than it amplified. 4 off-distribution abstentions —
   the gate skipped modulation cleanly when the scorer flagged
   off-dist (the documented `(off-dist,gate-skipped)` marker).
6. **Baseline_compare verdict trajectory (cycles 1-4 of the most
   recent loop instance): cycle 2 `MLP_NO_BETTER_THAN_TRIVIAL` →
   cycles 3-4 `MLP_ADDS_SKILL`.** Latest cycle 4: `mlp_rank_ic=0.192`,
   best baseline (`rsi_meanrev`) IC=0.079, `ic_gap=+0.113`. The
   skill ledger headline `oos_ic=0.11` agrees with `mlp_rank_ic` by
   construction. The MLP is BEATING a one-line rule in the most
   recent cycle, contradicting the long-standing
   `MLP_NO_BETTER_THAN_TRIVIAL` finding — worth re-evaluating once
   the loop restarts and accumulates more cycles.
7. **Production data directories remain SPLIT** across the monorepo
   path (this checkout) and `/media/zeph/projects/paper-trader/`.
   `data/` is not symlinked here — `decision_outcomes.jsonl`,
   `scorer_skill_log.jsonl`, `baseline_skill_log.jsonl` are not
   visible from the monorepo working tree, so `analyze()` against
   the default path returns `INSUFFICIENT_DATA`. Out-of-scope
   deployment issue; flagged in the prior pass too. Operators must
   either symlink or pass an explicit `outcomes_path=` to invoke
   any diagnostic from this checkout.

### Test commands for this domain

```bash
cd /home/zeph/trading-intelligence/paper-trader

# Just the ML / backtest / scorer / smoke-test / leveraged scope (~6 sec)
python3 -m pytest tests/test_scorer_freshness.py \
                  tests/test_scorer_smoke_test.py \
                  tests/test_decision_scorer.py \
                  tests/test_backtest.py \
                  tests/test_continuous.py \
                  tests/test_ml_backtest_review.py \
                  tests/test_calibration.py \
                  tests/test_baseline_compare.py \
                  tests/test_leveraged_skill.py \
                  tests/test_sector_skill.py \
                  tests/test_persona_skill.py -v

# Just this pass's new coverage (~0.8s)
python3 -m pytest tests/test_leveraged_skill.py -v

# Inspect deployed pickle through the leveraged lens (read-only)
python3 -m paper_trader.ml.leveraged_skill            # table view
python3 -m paper_trader.ml.leveraged_skill --json     # machine-readable
# Against the production outcomes file on USB:
python3 -c "from paper_trader.ml.leveraged_skill import analyze; \
import json; print(json.dumps(analyze( \
  '/media/zeph/projects/paper-trader/data/decision_outcomes.jsonl'), indent=2))"
```

### How to interpret the leveraged-skill ladder

A `BALANCED_EDGE` verdict says the scorer's `rank_ic` generalises
across the leverage axis — the conviction gate's prediction carries
comparable edge on both halves of the watchlist. This is the
"gate is doing real work, not beta-amplification" reading.

A `LEVERAGED_DOMINATES` or `LEVERAGED_ONLY_EDGE` says the scorer's
apparent rank-IC from the headline `oos_ic` is essentially the
leveraged-ETF subset's IC — non-leveraged BUYs are gated on a
prediction the data says doesn't carry differential edge there.
Per-bucket reads should be the primary signal, not the headline.

A `HAS_INVERTED_BUCKET` is the actionable red flag — the scorer's
sign is systematically wrong on one half of the universe. Gating
on it there is actively harmful. This is the data for a (separate,
explicit) decision to exclude the bucket from gating or retrain
with rebalanced classes; do NOT change `_LEVERAGED_ETFS` or
`SECTOR_MAP` from this read-only audit — both are SSOTs.

### Invariants reaffirmed by this pass

- **Read-only diagnostic discipline.** `leveraged_skill` never
  trains, never touches `decision_scorer.pkl`, no
  `build_features` / `N_FEATURES` / trade-path mutation. Safe to
  run against the live unattended loop.
- **Verdict-ladder discipline.** Every CLI in `paper_trader/ml/`
  emits a verdict from a fixed public `VERDICTS` tuple, exit 0 on
  benign/insufficient-data, exit 2 on actionable failure. The new
  `leveraged_skill` joins that ladder; tests pin the membership.
- **Single source of truth for leveraged-ETF class.** Imports
  `_LEVERAGED_ETFS` from `paper_trader.backtest` rather than
  redefining — a future edit to that set shifts every consumer in
  one place (mirrors `sector_skill`'s `SECTOR_MAP` discipline).
- **Single source of truth for OOS split.** Reuses
  `validation.split_outcomes_temporal` (the EXACT split the
  scorer-skill ledger and `calibration --oos` /
  `baseline_compare` / `sector_skill` already use). The diagnostic
  describes the SAME holdout every other OOS tool reports on.

## Review pass — paper-trader core hybrid (2026-05-19, Agent 1, ~12:45 UTC)

### Phase 1 (commit `8f8d862`) — `_option_expired` flips at NYSE close

Strategy.py `_option_expired` used `exp < datetime.now(timezone.utc).date()`,
which kept an expired option marked at `avg_cost` (with `stale_mark=True`)
for ~3-4h after the actual 16:00 ET close — every monthly expiry. Same
window let `_execute` SELL_CALL/SELL_PUT settle a closed expired
contract at `avg_cost` (instead of intrinsic) when the chain returned
None. Documented as a *deferred* fix in review pass #33 ("Skipped this
pass; documented for the next") — now applied.

The fix preserves the existing 6 pin tests verbatim by keeping the
legacy `today=date` kwarg (date-only comparison, "expiry day itself is
not expired") and adds a new `now=datetime` kwarg that drives the NY-tz
+ `market.close_minute` close-gate path. Production callers
(`_mark_to_market`, `_execute`) pass neither kwarg, so they pick up the
new wall-clock NY-tz logic. 9 new tests in `TestOptionExpiredCloseGate`
lock the boundary at 16:00 ET, the half-day 13:00 ET early-close, the
UTC→NY normalization, the past-/future-date short circuits, and the
naive-datetime tolerance. `bugs_fixed = 1`.

### Phase 2 (commit `c83d31f`) — per-position earnings-imminent flag on Discord

The Opus decision prompt already sees the earnings calendar
(`event_calendar.build_event_calendar`), but the operator's Discord
hourly/daily summary's position lines showed weight + hold-age with
**no indication that a held name reports in <1d**. The live 2026-05-19
book sat at NVDA **75% of stock book** with NVDA reporting
**0.5d away** — the trader scanning the hourly saw nothing about the
imminent print. The same dashboard→Discord gap `_concentration_line`
closed for name concentration (pass #34), `_pos_hold_age_token` closed
for the disposition effect (pass #33), and `_capital_pulse_line` /
`_host_pulse_line` each closed one dimension over.

`_pos_earnings_token(p, events_by_ticker)` renders:

* `  ⚠ ER 0.7d` — tier `HELD_IMMINENT` (≤3d): must-see warning glyph;
* `  ER 5.0d`   — tier `HELD_SOON` within horizon: informational;
* `  ⚠ ER after close` / `  ER after close` — `days_away < 0`
  (same-day post-bell — print just happened); explicit wording beats a
  confusing "-0.1d".

`_portfolio_lines` now takes an optional `events_by_ticker` kwarg
(default `None` → byte-identical to prior behaviour; every existing
unit-test caller stays compatible). `send_hourly_summary` and
`send_daily_close` resolve the events dict via the new
`_earnings_events_by_ticker` helper, which composes
`build_event_calendar` **verbatim** (single source of truth, invariant
#10 — the prompt and the Discord surface can never tell different
earnings stories). Pure filesystem read inside the builder (the
`signals.py` / `event_calendar` precedent — **no network on the Discord
path**, the `_concentration_line` discipline).

Same observational-only contract as `_pos_pct_weight` /
`_pos_hold_age_token` (invariants #2/#12). Any builder fault degrades
to `None` so the calling line drops the earnings token but the whole
summary still ships (reporter additive failure contract).

17 new tests (`TestPosEarningsToken` 11 + `TestEarningsEventsByTicker`
6) lock the token shape, the same-day-post-bell wording, the
byte-compat path for the existing unit-test callers (no kwarg → no
token), and the degrade-to-None path on every documented builder
failure mode (raise / source_ok=False / non-dict / empty events /
malformed event rows). `features_added = 1`.

### Phase 3 — live validation against the running trader (12:44 UTC)

1. ✅ **`/api/event-calendar`** — `NVDA: HELD_IMMINENT, 0.47d`,
   `MRVL: WATCH, 7.47d`. My new Discord token resolves from this
   exact endpoint's underlying data.
2. ✅ **Dry-run hourly summary against live store** rendered NVDA's
   position line as `NVDA  qty 2.0  ... held 12h  ⚠ ER 0.5d` — the
   feature works end-to-end.
3. ✅ **`/api/portfolio`** healthy — $1000 equity, NVDA 2sh + TQQQ
   2sh, `stale_marks: 0`. (P/L $0 is the documented after-hours-
   close-print coincidence noted in pass #34, not a bug.)
4. ✅ **Notify (Discord) HEALTHY** — last successful send 12:20,
   `consecutive_failures: 0`.
5. ⚠️ **17 consecutive NO_DECISIONs (95% IDLE_STORM)** —
   `/api/no-decision-reasons` correctly buckets dominant cause as
   `host_saturated (50%)` with the right recommendation ("a runner
   restart does NOT help"). The 4-agent concurrent review (this run +
   3 siblings, ~14 Opus subprocesses) IS the saturation. Known
   [[pt-no-decision-host-saturation]]; not a code bug.
6. ⚠️ **`/api/supervision` `UNSUPERVISED` + `behind: 1` commit** —
   trader runs as orphan PID 2849533 (PPID 1), on boot SHA `5066b20`
   while HEAD moved to `c83d31f` after my Phase 2 push. The
   git-watcher's deferred restart will fire on the next cycle
   boundary to deploy the new feature. Known
   [[pt-systemd-vs-manual-restart-spam]]; operator chose this
   configuration.

`user_findings = 3`: (a) the saturation pattern remains the dominant
NO_DECISION cause, operator-resolvable, not code-resolvable; (b) the
runner-supervised-orphan / behind-HEAD pattern remains visible to
every operator surface (Discord summary correctly surfaces both
warnings, no silent failure); (c) the new earnings token surfaces
correctly inline next to NVDA on the live hourly body — a real
value-add for a trader checking Discord while holding into the print.

### How to run / test

```bash
cd /home/zeph/trading-intelligence/paper-trader

# This pass's surface (~1.5s)
python3 -m pytest tests/test_core_strategy.py::TestOptionExpiredCloseGate \
                  tests/test_core_reporter.py::TestPosEarningsToken \
                  tests/test_core_reporter.py::TestEarningsEventsByTicker -v

# Full focused core sweep (~15s)
python3 -m pytest tests/test_core_runner.py tests/test_core_signals.py \
                  tests/test_core_market.py tests/test_core_store.py \
                  tests/test_core_strategy.py tests/test_core_reporter.py \
                  tests/test_core_runner_cycle.py tests/test_core_invariants.py \
                  tests/test_parse_retry.py tests/test_event_calendar.py -q

# Inspect the new earnings token against the live store
python3 -c "
from paper_trader import reporter
ev = reporter._earnings_events_by_ticker()
from paper_trader.store import get_store
s = get_store(); pos = s.open_positions(); pf = s.get_portfolio()
for ln in reporter._portfolio_lines(pos, total_value=pf['total_value'], events_by_ticker=ev):
    print(ln)
"
```

### Invariants reaffirmed by this pass

* **No `git add -A`** — sibling agents were running concurrently;
  explicit pathspec on every commit ([[pt-concurrent-samerole-staging-race]]).
  Phase 1 staged only `paper_trader/strategy.py` +
  `tests/test_core_strategy.py`; Phase 2 staged only
  `paper_trader/reporter.py` + `tests/test_core_reporter.py`. The
  sibling untracked `paper_trader/analytics/restart_recommendation.py`
  / `position_action_brief.py` / `tests/test_restart_recommendation.py`
  + the `digital-intern/` working-tree edits were correctly left out.
* **Single source of truth (invariant #10)** — the Discord earnings
  token reads from `build_event_calendar` verbatim, the SAME builder
  the Opus prompt's `event_calendar_block` reads from. Never
  re-derives `days_away` / `tier`.
* **No network on the Discord path** — `_earnings_events_by_ticker`
  goes through `build_event_calendar` which reads the JSON snapshot
  *from disk* (digital-intern writes it via its earnings collector),
  never a `:8080` hop. Preserves the `_stress_line` /
  `_recovery_line` / `_concentration_line` no-yfinance discipline on
  the alert path.
* **Reporter additive failure contract** — every new block/function
  degrades to `""` or `None` on fault; the whole hourly / daily
  summary still ships. Locked by the `test_*_returns_none` and
  `summary_still_sends_when_*` test path across the suite.
* **Invariants #2/#12 (observational only)** — the earnings token
  states a fact (a held name reports in X days); never gates,
  imposes a cap, or sends a directive. The Opus decision prompt
  already had this data via `event_calendar_block`; this is the
  Discord-surface mirror.
* **NYSE close gate (invariant #14)** — `_option_expired` now uses
  `market.close_minute` (16:00 ET regular / 13:00 ET half-day) and
  the NYSE NY tz, the same source of truth `is_market_open` /
  `next_session_open` / `_maybe_daily_close` already use.


## Review pass — paper-trader product-engineer (2026-05-19, Agent 4, ~12:50 UTC) — false-HEALTHY closure

Two new composite analytics endpoints. Both pure builders, both backed by
SWR cache, both anchored on the live false-HEALTHY case observed at session
start: `desk_pulse.liveness.restart_recommended=false` while
`empty-claude-rate=81.4%` AND $445 NVDA exposure into an earnings print
11h away. The cadence-only liveness verdict is structurally blind to that
combination; neither it nor any single existing endpoint composes
parse-fail-rate × held-exposure × event-proximity into one operator-actionable
scalar.

### Feature: `/api/restart-recommendation` — single operator-actionable verdict

`paper_trader/analytics/restart_recommendation.py` + tiny route wrapper.
Pure builder takes already-computed scalars (empty-rate over 24h,
host-saturated flag, held-imminent exposure $, hours-to-nearest-event,
consecutive-no-decision count) and returns:

* `verdict` ∈ {OK, MONITOR, RESTART_RECOMMENDED, RESTART_URGENT}
* `restart_now: bool` — the ONE bit a cron / Discord poller acts on
* `urgency_score: 0..1`
* `reasons: [str]` — one-line human readables, headline-ready
* `next_check_seconds` — cadence shortens as urgency climbs (15m → 1m)
* `inputs` — transparency block for the operator
* `thresholds` — what the ladder cuts on

Precedence ladder, first match wins:

1. **RESTART_URGENT** — empty_rate ≥ 60% AND held-imminent exposure
   within 24h. The exact "BLIND into the print" wedge.
2. **RESTART_RECOMMENDED** — IDLE_STORM (≥5 consecutive NO_DECISION,
   matches `runner_heartbeat`'s gate), OR moderate empty_rate (≥50%)
   with any held-imminent exposure.
3. **MONITOR** — host saturated, mild empty_rate (≥30%) with held
   exposure, OR ≥3-cycle no-decision streak.
4. **OK** — none of the above.

The endpoint reads recent_decisions via the store, computes empty_rate +
consecutive-no-decision using the shared `_empty_rate_24h_pct` /
`_consecutive_no_decision` helpers so all three surfaces never disagree
by 1% on the same DB, and pulls held-imminent earnings exposure by
calling `build_event_readiness` against `:8080/api/earnings` exactly the
way `/api/event-readiness` and `/api/earnings-risk` already do (SSOT,
same intern hop). SWR cache 30s.

### Feature: `/api/position-action-brief` — per-held-position composite

`paper_trader/analytics/position_action_brief.py` + small route wrapper.
For every held position (stock + option lots folded into the
underlying), composes:

* exposure_usd / cost_basis_usd / unrealized_pl_usd / pct_portfolio
* hours_to_event + event_verdict + earnings_date (from `build_event_readiness`;
  reads the readiness verdict BLIND/DEGRADED/READY, falls back to the
  earnings-risk tier HELD_IMMINENT/HELD_SOON/WATCH when events come from
  that endpoint instead)
* news_state / window_count / z_score / top_window_title (from
  `build_news_velocity` — the same SSOT `/api/news-velocity` uses)
* last_decision_status ∈ {DECIDED, EMPTY, HOST_SKIP, PARSE_FAIL, NEVER}
  + last_decision_age_min + the action text
* recommended_action ∈ {OK, MONITOR, HOLD_THROUGH_EVENT,
  TRIM_BEFORE_EVENT, RESTART_RUNNER}
* urgency_score 0..1
* reasons

Per-position action ladder, TRIM_BEFORE_EVENT dominates: held-imminent
print × wedged bot ⇒ TRIM, otherwise HOLD_THROUGH_EVENT; wedged-bot with
SURGING news ⇒ RESTART_RUNNER; SURGING news alone or near-event without
wedge ⇒ MONITOR; clean state ⇒ OK. Briefs sort most-urgent-first;
overall headline surfaces the single most-actionable position.

The route owns the I/O — runs the same `articles.db` read shape as
`/api/news-velocity` (mode=ro + live-only clause + LIKE-prefilter for
baseline performance — the documented anti-INSUFFICIENT-everywhere
optimisation), pulls earnings events from `:8080/api/earnings`, and
calls `build_event_readiness` to get exposure-tagged held events. SWR
cache 90s — bounded by the cold news-velocity articles.db scan.

### Live validation

Both endpoints return on the live wedged book:

* `/api/restart-recommendation` reads `verdict=RESTART_RECOMMENDED`,
  `restart_now=true`, `urgency_score=0.85`, headline:
  *18 consecutive NO_DECISION cycles — engine cycling but not deciding,
  restart clears a wedged Claude CLI*. Closes the false-HEALTHY gap.
* `/api/position-action-brief` reads `overall_action=HOLD_THROUGH_EVENT`,
  `overall_urgency=0.6`, headline:
  *ACTION — NVDA HOLD_THROUGH_EVENT: earnings in 11.0h — hold through
  with a working bot*. NVDA brief shows exposure $444.70, hours_to_event
  11.04, news SURGING (231 articles in 24h, top "Nvidia Earnings And
  The Burden Of Perfection"), last_decision_status=DECIDED 5h ago.
  TQQQ brief reads OK clean.

The two endpoints complement rather than duplicate: restart-rec answers
*does the bot need to be kicked NOW (idle storm)?* while
position-action-brief answers *what is the situation on each held name
right now?*. The 18-cycle idle storm fires restart-rec but
position-action-brief still reads HOLD because 24h-empty-rate=42.6% is
below the 50% wedged threshold — the bot WAS working earlier and only
just now is wedged.

### Invariants reaffirmed by this pass

* **Pure builders** — both analytics modules never raise on garbage
  inputs and own no DB / network / module-global; the routes own the
  I/O (the documented `thesis_drift` split).
* **Advisory only** — neither endpoint gates Opus, neither injects into
  the decision prompt, neither adds caps (invariants #2 / #12).
* **SSOT** — news-velocity reuses `build_news_velocity`; event proximity
  reuses `build_event_readiness`; empty-rate / no-decision counters are
  shared module-level helpers so `/api/empty-claude-rate`,
  `/api/restart-recommendation`, and `/api/position-action-brief`
  always agree on what the same DB says.

### How to test

```sh
cd /home/zeph/trading-intelligence/paper-trader
python3 -m pytest tests/test_restart_recommendation.py \
                   tests/test_position_action_brief.py -v
# 55 tests, ~1s
```

Live verification, Flask test client (the documented `__main__` smoke
hits the wrong DB per the analytics-verification discipline):

```python
from paper_trader.dashboard import app
c = app.test_client()
print(c.get("/api/restart-recommendation").get_json())
print(c.get("/api/position-action-brief").get_json())
```


## Review pass — paper-trader core hybrid (2026-05-19, Agent 1, ~18:30 UTC) — close-side countdown helper

### Phase 1 — no bugs found, no commit

A read-pass through `runner.py`, `reporter.py`, `signals.py`, `strategy.py`,
`market.py`, `store.py`, and a partial sweep of `dashboard.py` found no
real bugs. The codebase has been through many prior review passes (see
this file's "Review pass" headers) and the obvious things are already
fixed. Per the Phase-1 commit guard, this pass set `bugs_fixed=0` and made
no Phase-1 commit. The 8 `TestSourceMixLine` failures present at session
start are sibling agent work-in-progress (broken
`store.update_portfolio(cash=…, total_value=…)` calls missing the
`positions` arg) and were intentionally left alone — that file belongs to
the sibling per the `pt-concurrent-samerole-staging-race` memory note.

### Phase 2 (commit `813c4c5`) — `next_session_close` / `seconds_until_close`

`paper_trader/market.py` already exposed `next_session_open()` and a
`close_minute(date)` helper, but had no *close-side* timestamp. Callers
who wanted "when is the next bell DOWN?" had to reach into
`close_minute` and roll their own date arithmetic — and several already
do (the `reporter._next_session_line` precedent, the prompt's
`MARKET_OPEN` flag). New pure helpers:

- `next_session_close(now=None) -> datetime | None` — the next NYSE
  session close (16:00 ET regular / 13:00 ET half-day) after `now`, as
  a UTC-aware datetime. Mirrors `next_session_open`'s semantics:
  mid-session returns today's close, pre-open today returns today's
  close, at-or-past-close advances to next trading day's close.
- `seconds_until_close(now=None) -> int | None` — integer-second
  countdown for prompt/Discord rendering. Always ≥ 0 (clock step-back
  clamps to 0).

Half-day aware (the day-after-Thanksgiving / Christmas Eve 13:00 ET
close lands correctly; 13:00 on a half-day advances past). Strict
`close_dt > now_ny` advance rule so a tick *at* the bell always
advances to the next session — same discipline as `next_session_open`
("at the open instant, today is no longer the next open").

22 new unit tests in
`tests/test_core_market.py::TestNextSessionClose` /
`TestSecondsUntilClose` covering: mid-session, pre-open, at-close
strict advance, post-close, Friday→Monday, weekend, holiday skip,
half-day morning, half-day at-1300 advance, Christmas Eve half-day,
Good Friday skip, UTC awareness, seconds arithmetic at boundaries,
clock-step-back clamp, int return type. The new helpers are not yet
wired into reporter/strategy/dashboard — Phase 2 ships the primitive;
wiring it into the hourly summary's "session ends in Xh Ym" line is
a follow-up (kept out of this pass so the staging is one feature, not
one feature + one Discord refactor).

### Phase 3 — live validation against the running trader (18:30 UTC)

Probed the live trader as an operator would. Findings:

1. **Dashboard endpoint timeouts during sibling Opus storms.** While
   ~4 sibling Opus subprocesses (this HYBRID pass + sibling hourly
   reviews) were active, `/api/empty-claude-rate`, `/api/host-guard`,
   and `/api/capital-paralysis` all timed out at 15–30s while
   `/api/healthz` and `/api/state` responded in <100ms. After the
   trader's own deferred restart cleared, the same endpoints responded
   in ~200ms. The dashboard's existing SWR machinery (see lines
   230–418) is *meant* to serve a stale payload during a cold rebuild,
   but the warming sometimes outlasts the client timeout — not a bug,
   but worth noting that operator polling needs to tolerate the SWR
   `{"warming": true}` shape for the first call after a busy interval.

2. **Self-healing stale-runner window.** Between this pass's commit
   `813c4c5` landing and the next decision-cycle boundary the runner's
   `/api/healthz.stale` reported `true` (boot_sha `83b4f2e`, head_sha
   `813c4c5`). The git-watcher does eventually fire the deferred
   restart, but the trader briefly executes on stale code. Per
   invariant #19 / `RESTART_GRACE_S` this is by design — never
   force-kill a healthy mid-Opus call — but is the documented "stale
   for up to ~1 cycle" UX. No action needed.

3. **HOST_SATURATED is the dominant NO_DECISION cause (60.9% over
   6h).** Verified via `/api/empty-claude-rate` —
   `concurrent_opus_processes: 4`, `verdict: "HOST_SATURATED — live
   trader starved by concurrent Opus"`. The trader's own mid-call
   re-probe (strategy.py:1538) is correctly bucketing these as
   "skipped claude call —" rather than the model-timeout signature.
   `/api/restart-recommendation` correctly says MONITOR (empty rate
   below 50% URGENT threshold) — restart would NOT help; the cause is
   external concurrent jobs, not a wedged CLI. Working as designed.

4. **systemd-vs-manual restart-counter pathology persists** (counter
   at 680). The systemd unit keeps failing to start while the manual
   instance holds the singleton lock. Per the
   `pt-systemd-vs-manual-restart-spam` memory note this is "not a bug,
   don't fix" — the singleton lock is correctly defending against
   double-trading; the systemd unit just needs to stay disabled
   (it is).

5. **Capital-paralysis recovered organically.** While the pass was
   running, the trader SOLD DRAM at 18:20:55, releasing $253 of pinned
   cash (from $7.86 → $260.91). This is healthy trading behavior —
   the trader correctly recycles capital when conviction shifts —
   confirming the no-hard-limits design (invariant #12) does what it
   says when Opus is willing to act.

### How to run / test

```sh
cd /home/zeph/trading-intelligence/paper-trader
python3 -m pytest tests/test_core_market.py -v
# 65 tests, ~1s (22 new ones added by this pass)
```

Inspect the new helpers against the live wall clock:

```python
from paper_trader import market
print(market.next_session_close())    # UTC-aware
print(market.seconds_until_close())   # int seconds
```

### Invariants reaffirmed by this pass

- **#1** (backtest articles never reach live signals) — untouched; no
  signals/SQL changes.
- **#2 / #12** (no hard limits, observational only) — new helpers are
  pure read of the calendar, never gate.
- **#10** (single source of truth) — both helpers share the
  `close_minute(date)` and `NYSE_HOLIDAYS_2026` / `NYSE_HALF_DAYS_2026`
  primitives `is_market_open` already uses; no parallel calendar.
- **#19** (singleton lock fail-open / never refuse the sole runner) —
  reaffirmed by Phase 3 finding #4: the manual instance holds the
  lock, systemd's competing instance correctly exits.


## ML / backtest review pass (Agent 2, 2026-05-19, ~11:30 UTC) — symmetric training-label clamp

### Scope

`paper_trader/ml/decision_scorer.py`, `run_continuous_backtests.py`
(scorer status string + parser), plus new tests. The deployed scorer
already clamps its OUTPUT to ±`PRED_CLAMP_PCT` (50%) but trained on
unclamped labels. Live audit of `data/decision_outcomes.jsonl` (7413
rows in the corpus, last 5000 used by the trainer) showed:

- 25 rows with `|fr_5d| > 50%` in the trainer tail
- 2 MSTR rows with `forward_return_5d > +100%` (one at +175%)
- 23 exact-zero rows (the documented walk-back-collision footprint
  pre-fix; current rate 14/5000 = 0.28% — below the audit's 0.5% gate)

The MSE gradient at fit time pulls weights toward magnitudes the gate
can never act on (`predict()` clamps regardless). Aligning the
training label space with the inference output space is a clean ML
hygiene fix with bounded impact on the heart of the distribution.

### Feature: clamp training labels to ±PRED_CLAMP_PCT

`train_scorer` now applies the same ±`PRED_CLAMP_PCT` clamp to every
training label BEFORE the SELL sign-flip, and reports the count of
rows it touched as `n_label_clamped` in the result dict. The
continuous loop forwards that count into the per-cycle scorer-skill
status string (and the structured ledger row) so a quant can trend
the outlier-rate of the training tail per cycle — a spike correlates
with leveraged-ETF crash/rip weeks (MSTR/SOXL/TQQQ) polluting the
corpus.

### How to interpret it

`n_label_clamped == 0` is the steady-state expectation. A sudden
single-cycle spike (>5% of `train_n`) means the new outcomes batch
carries an unusual cluster of >50% 5-day moves — typically a
short-window crypto/MSTR rip or a leveraged-ETF crash. Sustained
nonzero `n_label_clamped > 20` over many cycles means the corpus is
chronically populated by leveraged outliers; a quant should know
this before reading any of the OOS skill metrics, because the
non-clamped corpus would have produced a different model (and the
clamped version is the deployed one).

### How to run / test

```sh
cd /home/zeph/trading-intelligence/paper-trader

# Decision-scorer surface (57 tests, ~3s — 5 new for label clamp):
python3 -m pytest tests/test_decision_scorer.py -v

# Continuous-loop parser/status surface (2 new tests):
python3 -m pytest tests/test_continuous.py -v -k "ParseScorerStatus"

# Full ml/backtest/scorer focused sweep (~2 min):
python3 -m pytest tests/ -v -k "ml or backtest or scorer"
```

Inspect against the deployed pickle (read-only):

```python
from paper_trader.ml.decision_scorer import train_scorer
# Train returns the new key on every call:
#   {"status": "ok", "n": …, "val_rmse": …, "n_label_clamped": …}
```

### Phase 3 findings (live operator perspective, quant-relevant)

Concrete observations from probing the deployed system against
`/media/zeph/projects/paper-trader/data/` (the active continuous
loop's actual data directory):

1. **Continuous loop is dead.** `scorer_freshness` reports
   `LOOP_DEAD`, heartbeat 24.6h stale; last cycle was #4
   (2026-05-18T18:25:43+00:00). `run_continuous_backtests.py` is NOT
   running. While dead, the conviction gate operates on a frozen
   pickle that's increasingly stale relative to the accumulating
   `decision_outcomes.jsonl` tail.

2. **Deployed pickle is REGRESSED** (already detected by
   `tests/test_scorer_freshness.py::TestPklRegressed`). The deployed
   `data/ml/decision_scorer.pkl` has `n_train=400` but cycle 4
   logged `train_n=3959`. Something clobbered the pickle with a
   tiny-corpus fit. Since `n_train < 500`, the conviction gate is
   INACTIVE per invariant #5, even though `scorer_skill_log.jsonl`
   reports `gate_active=True` (based on the last LOGGED training
   event, not the deployed pickle).

3. **OOS rank-IC ≈ noise.** On the last 1483 outcomes (temporal
   20% holdout), Spearman rank-IC = +0.035, directional accuracy =
   0.492. The MLP has no useful OOS skill on this corpus —
   reaffirming the `MLP_WORSE_THAN_TRIVIAL` finding from
   `baseline_compare`.

4. **Mean prediction is biased −4.02% while mean realized is
   +0.85%.** A 4.87pp negative calibration bias — the model
   systematically over-pessimizes. The training-label clamp helps
   here: removing huge positive outliers (MSTR +175%, +128%) tames
   the gradient-magnitude story even though the mean target shifts
   by only ~0.1pp.

5. **Backtest runs themselves look healthy.** 475/501 complete, 24
   failed, 2 stuck running (orphans — the
   `_reap_orphaned_runs` should clear them at startup of the next
   loop revival). Returns of recent complete runs (+580%, +317%,
   +189%, +50%, +38%) are plausible for 3–9yr leveraged-watchlist
   windows.

### Invariants reaffirmed by this pass

- **#5** (DecisionScorer gates only when `_n_train ≥ 500`) —
  untouched: the clamp acts on TRAINING LABELS, not on
  `build_features` or the gate boundaries. `_ml_decide`'s arm
  thresholds (±10/±5/0) are byte-identical.
- **#10** (single source of truth) — the clamp reuses
  `PRED_CLAMP_PCT`, the same constant `predict()` /
  `predict_with_meta` already enforce; changing one moves both in
  lockstep.
- **#13** (pickle schema `{model, scaler, n_train}`) — untouched:
  the clamp is in the data-prep pass, not the persistence shape.


## 2026-05-20 feature-dev pass (Agent 4) — `/api/round-trip-postmortem` + `/api/news-themes` + `/api/held-theme-decay`

Three new dashboard surfaces that fill orthogonal operator gaps that
none of the existing ~120 endpoints close.

### `/api/round-trip-postmortem` — was the exit good?

`/api/round-trips` says WHAT closed (and realised P&L). Every
realised-P&L panel (track-record, churn, streak, winner/loser
autopsy, trade-asymmetry) reduces the round-trip list to summary
stats. None of them ask the operator's actual follow-up: *was the
exit good?* The DRAM 2026-05-19 round-trip (BUY 5 @50.70 17:13 UTC,
SELL 5 @50.61 18:20 UTC, -0.18% / -$0.45) is precisely the case the
existing surfaces flatten — the post-exit price action is the only
piece of data that makes a paper-cut sell falsifiable.

Pure SSOT `analytics/round_trip_postmortem.py::build_round_trip_postmortem`
(never raises, never network). Verdict ladder per closed RT:

- **CORRECT** — post-exit drift `≤ CORRECT_MAX_DRIFT_PCT` (-1%). Exit
  captured the local high.
- **PREMATURE** — drift between `PREMATURE_MIN_DRIFT_PCT` (1%) and
  `MISSED_RUNNER_MIN_DRIFT_PCT` (5%). Bot sold, the move continued
  against the exit direction.
- **MISSED_RUNNER** — drift ≥ 5%. Bot exited a big winner.
- **WHIPSAW** — short hold (≤ `WHIPSAW_MAX_HOLD_HOURS`=4h) + small loss
  (≥ -`WHIPSAW_MAX_LOSS_PCT`=1.5%) + post-exit recovery > half of
  `PREMATURE_MIN_DRIFT_PCT`. The specific DRAM-1h-paper-cut pathology;
  pnl signal + short-hold are the discriminator (a long-hold winner's
  rise-after is PREMATURE, not WHIPSAW).
- **NEUTRAL** — drift inside the band.
- **INSUFFICIENT** — exit `< MIN_HOURS_SINCE_EXIT` (2h) ago or no
  current price. Sample-size honest (same `build_tail_risk` /
  `build_correlation` / `build_news_velocity` precedent — numerics
  emitted whenever defined, verdict withheld until window matures).

Aggregate **`exit_quality_score`** is +1 CORRECT / -1 PREMATURE /
-2 WHIPSAW / -2 MISSED_RUNNER averaged over scored trips —
persistently negative ⇒ the bot is exiting too early. Single trip is
not load-bearing; the score matures with N≥3.

Distinct from neighbours (invariant #10 — do not consolidate):
`/api/thesis-drift` grades OPEN positions against entry rationale;
`/api/winner-autopsy`/`/api/loser-autopsy` reduce CLOSED P&L to
aggregate stats; neither incorporates post-exit price action. The
post-exit drift is the only new piece of data this endpoint adds —
and it makes the realised-P&L number falsifiable in hindsight.

Query params: `max_n` (default 10, clamp 1..50), `hours_back`
(default 168, clamp 1..720). SWR-cached 60s. Advisory only — never
gates Opus, never injected into the decision prompt (invariants
#2/#12). Pure builder `build_round_trip_postmortem(round_trips,
current_prices, now=None, max_n=10)` — never raises (garbage row /
None / negative price → INSUFFICIENT, never an exception). Yfinance
fetch lives in the endpoint (`market.get_prices`); a fetch failure
degrades the whole table to INSUFFICIENT, never raises.

Locked by `tests/test_round_trip_postmortem.py` (23 tests — verdict
ladder per case, WHIPSAW disambiguation from CORRECT vs PREMATURE,
per-share-avg arithmetic, `exit_quality_score` arithmetic,
NO_DATA/INSUFFICIENT/OK ladder, max_n clip, never-raises on
garbage) + `tests/test_round_trip_postmortem_endpoint.py` (6 tests —
Flask wiring with DRAM-replay fixture, `hours_back` filter, max_n
clamp, price-fetch-fails-degrades, NO_DATA on empty store).

### `/api/news-themes` — what is the wire ACTUALLY talking about?

The wire produces 100+ articles per hour across ~17 collectors. The
existing surfaces tell the operator a slice of what is in there but
not the per-name "loudest theme" view a discretionary PM watches:

- `/api/news-deduped` is the linear item list (one row per article,
  no ticker rollup).
- `/api/news-velocity` is the per-held-ticker MENTION RATE vs
  baseline (Poisson z-score) — not a score-weighted loudest-theme
  rollup.
- `/api/sector-heatmap` / `/api/sector-signal-fit` aggregate at the
  SECTOR level — coarser than the per-name view.
- `/api/watchlist-opportunities` ranks within the curated watchlist;
  this is across the *entire* live feed regardless of watchlist
  membership.
- digital-intern's `trend_velocity` does market-wide mention-gainers,
  not score-weighted theme prominence.

Pure SSOT `analytics/news_themes.py::build_news_themes`. Per-ticker
recency-decayed score: `Σ ai_score × exp(-age_h / 6h × ln 2)`.
Multi-ticker articles **split** their score evenly across mentioned
tickers (a 4-ticker headline contributes 0.25× to each — avoids one
wide-net article inflating four themes simultaneously; the same
discriminator as `sector_signal_fit`'s `signal_share_pct`). Per-row:
`decayed_score`, `n_articles`, `max_urgency`, `top_title` /
`top_url` (highest decayed-weight article for this theme),
`held` (case-insensitive against `store.open_positions()`).
Aggregate: `total_decayed_score`, `n_held_themes`,
`n_unheld_themes`, `top_unheld_ticker` (a missed-opportunity
bookmark distinct from `/api/watchlist-opportunities`).

Defense-in-depth backtest filter at the builder: any row whose `url`
LIKE `backtest://%` or `source` LIKE `backtest_%` /
`opus_annotation%` is dropped — so a leaked synthetic row cannot
reach user-facing JSON even if a future caller forgets the SQL
clause (the canonical SQL filter is still applied in the endpoint,
mirroring `signals.get_top_signals`).

State ladder: NO_DATA (no articles in window) / OK. No sample-size
gate beyond "at least one article surviving the recency filter" —
the single ranked list is honest even with one input.

Query params: `hours` (default 24, clamp 1..168), `max_themes`
(default 20, clamp 1..100), `min_score` (default 2.0, 0..10).
SWR-cached 60s. Ticker extraction reuses
`signals._extract_tickers` (the SSOT used by the live trader's
prompt-building path) so theme tickers never drift from the
universe Opus sees in `decide()`. Advisory only — never gates Opus,
never injected into the decision prompt (invariants #2/#12).

Locked by `tests/test_news_themes.py` (18 tests — decay arithmetic
on a halflife article, ranking by decayed score, count/max-urgency
aggregation, top_title is the highest-weight article, multi-ticker
SPLIT (not full-weight to each), held flag case-insensitive,
held/unheld counts, top_unheld bookmark, summary block arithmetic,
max_themes clipping, synthetic-row drop at the builder, never-raises
on `None`/`"not a dict"`/missing-ts/garbage-timestamp rows,
no-tickers articles counted separately) +
`tests/test_news_themes_endpoint.py` (4 tests — Flask wiring with a
real on-disk sqlite articles.db, synthetic-row filter via SQL,
NO_DATA when no DB, param clamps, store-failure-degrades-not-raises).

### `/api/held-theme-decay` — is the catalyst on my held positions still alive in the wire?

`/api/news-themes` is a single-window snapshot — a held theme that
went DARK looks identical to one that just lit up. `/api/news-velocity`
is the per-held-ticker Poisson MENTION-RATE z-score; a flood of
low-relevance mentions inflates the rate while one 9.5
Sonnet-labelled article moves it the same as a junk RSS row
(different signal, different question). `/api/position-thesis` is a
single-window 24h headlines+bull/bear split, no velocity dimension.
`/api/thesis-drift` grades against ENTRY rationale, not current
wire prominence. None of them answer the operator's actual
reassessment trigger: *for each ticker I currently own, is the
score-weighted news flow LOUDER NOW or QUIETER NOW than it was
earlier?* — i.e. is the catalyst that justified holding it still
alive in the wire, or has the wire moved on?

Pure SSOT `analytics/held_theme_decay.py::build_held_theme_decay`.
Per held ticker: a FRESH window (default 6h — matches
`news_themes.DECAY_HALF_LIFE_HOURS` so fresh_score lines up with
that endpoint's top-theme contribution) and an immediately
preceding PRIOR window of the same width. Each window carries a
decayed-score sum `Σ ai_score × exp(-age_h / 6h × ln 2)`. Multi-
ticker articles SPLIT their weight evenly across ALL mentioned
tickers (anti-inflation rule, same discriminator as `news_themes`
/ `sector_signal_fit`).

Verdict ladder per hold:

- **DARK** — no qualifying articles in either window (nobody is
  talking about it; the entry-time catalyst may have run out of
  fuel). Also the verdict when both windows are below
  `MIN_FRESH_SCORE` (1.0) — absolute prominence floor honesty,
  prevents a 0.1→0.5 noise jump from claiming BUILDING.
- **FADING** — `fresh < prior × FADE_RATIO` (0.7); the wire is
  moving away from this name → reassess thesis.
- **BUILDING** — `fresh > prior × BUILD_RATIO` (1.43) AND fresh
  meets `MIN_FRESH_SCORE`; catalyst strengthening, current entry
  well-timed.
- **STABLE** — ratio between 0.7 and 1.43; steady-state coverage.

Worst-verdict aggregator picks the highest-severity bucket present
(FADING > DARK > STABLE > BUILDING) — FADING anywhere is the
operator's first re-assessment signal. The headline leads with
FADING, then DARK, then a flat status line. `holds[]` is sorted by
the same severity so the most urgent row is at index 0.

Defense-in-depth backtest filter at the builder mirrors
`news_themes._is_synthetic`: any row whose `url` LIKE
`backtest://%` or `source` LIKE `backtest_%` / `opus_annotation%`
is dropped — so a leaked synthetic row cannot corrupt the held-
position view even if a future caller forgets the SQL clause (the
canonical SQL filter is still applied in the endpoint).

State ladder: `NO_HELD` (no held positions — collapse-to-silence,
the chat-enrichment SSOT precedent) / `OK`. Per-hold `ratio` is
`None` when `prior_score == 0` (never fabricates `+inf`).

Query params: `hours` (default 6, clamp 1..72), `min_score`
(default 2.0, 0..10). SWR-cached 60s. Ticker extraction reuses
`signals._extract_tickers` so the held-theme view never drifts
from the universe `decide()` builds prompts against. Advisory
only — never gates Opus, never injected into the decision prompt
(invariants #2/#12). On the live 2026-05-20 book (NVDA + TQQQ),
the endpoint reports NVDA FADING (fresh 2.57 / prior 4.00) and
TQQQ DARK — exactly the "your NVDA coverage is fading INTO the
earnings print and nobody is writing about TQQQ" the operator
needs surfaced.

Locked by `tests/test_held_theme_decay.py` (26 tests — SSOT decay
halflife matches `news_themes`, full state-ladder edge cases on
the FADE_RATIO/BUILD_RATIO/MIN_FRESH_SCORE cutoffs, exact
decayed-weight arithmetic per article age, multi-ticker SPLIT
arithmetic, articles-outside-both-windows ignored, unheld-ticker
articles dropped, defense-in-depth backtest filter, worst-verdict
+ headline aggregator, holds sorted by severity, held-tickers
case-insensitive dedupe, never-raises on garbage rows /
non-list-tickers / string-ai_score, stable output shape).

### How to run / test

```sh
cd /home/zeph/trading-intelligence/paper-trader

# All three feature surfaces (77 tests, <2s):
python3 -m pytest tests/test_round_trip_postmortem.py \
                   tests/test_round_trip_postmortem_endpoint.py \
                   tests/test_news_themes.py \
                   tests/test_news_themes_endpoint.py \
                   tests/test_held_theme_decay.py -v

# Live probe (after the next paper-trader restart picks up the new routes):
curl -s 'http://localhost:8090/api/round-trip-postmortem?max_n=5' | python3 -m json.tool
curl -s 'http://localhost:8090/api/news-themes?hours=12&max_themes=10' | python3 -m json.tool
curl -s 'http://localhost:8090/api/held-theme-decay?hours=6' | python3 -m json.tool
```

### Invariants reaffirmed by this pass

- **#1** (backtest articles must never reach live signals) — all three
  endpoints apply the canonical SQL filter; `build_news_themes` and
  `build_held_theme_decay` additionally drop `backtest://%` /
  `backtest_*` / `opus_annotation*` rows at the builder as defense-in-
  depth (both reference `_is_synthetic` shape for consistency).
- **#2 / #12** (live trader has no hard limits, advisory-only) — all
  three endpoints are observational, never gate Opus, never injected
  into the decision prompt; reporter Discord pulse not added.
- **#10** (single source of truth, do not consolidate) — verdict
  ladder lives in one builder per endpoint; the route computes nothing.
  News ticker extraction reuses `signals._extract_tickers` so the
  theme tickers cannot drift from the universe `decide()` builds
  prompts against. `held_theme_decay` imports `DECAY_HALF_LIFE_HOURS`
  from `news_themes` so the two endpoints share a decay shape; any
  future re-tune updates both simultaneously and is pinned by
  `tests/test_held_theme_decay.py::TestSSOTDecayHalfLife`.
- **#7** (SWR-cached slow read-only) — all three endpoints carry
  `@swr_cached("...", 60.0)` like every news-IO sibling, and
  `held-theme-decay` is added to the SWR prewarm list so the first
  poll right after a restart is exactly when the operator is
  checking "did any held thesis go dark while I was away".

Applies on next paper-trader restart (the documented pattern for
every recent feature).


## Review pass — paper-trader core hybrid (2026-05-20, Agent 1, ~02:30 UTC) — ML-advisor substring false-positives

### Scope

`paper_trader/strategy.py::_WORD_TO_TICKER_LIVE` keyword→ticker
fallback used `keyword in title` substring matching, which
silently false-positively triggered on short keys: `"ai"` (→TQQQ)
matched `"rain"` / `"pain"` / `"Spain"` / `"trail"`; `"gold"`
matched `"Goldman"` (very common in finance news); `"intel"`
(→INTC) matched `"intelligence"` (double-counted with the `"ai"`
map). Each silently inflated an unrelated watchlist ticker's
score on the ML advisor's BUY pick. CLAUDE.md §15 calls the
advisor advisory-only, so this never gated a trade — but it
polluted the only signal Opus reads next to its own decision.

### Phase 1 — Fix

`paper_trader/strategy.py` switches the per-article keyword
lookup to a pre-compiled `\bkeyword\b` regex
(`_WORD_TO_TICKER_LIVE_PATTERNS`). The canonical recovery case
locked by `test_keyword_mapping_picks_up_unticked_article`
(`"nvidia surges to record on chip demand"` → NVDA/SOXL) still
matches because the keyword appears as a standalone token; the
multi-word `"federal reserve"` / `"artificial intelligence"`
entries also match because `\b` sits between word/non-word
transitions (spaces included). Keys are lowercased and titles
are lowered before matching, so the pattern is built from the
lowercase keyword.

### Tests

`tests/test_ml_live_opinion.py::TestKeywordSubstringFalsePositives`
adds three new tests:

* `test_rain_in_title_does_not_alias_to_tqqq_via_ai` — the live
  false-positive. Pre-fix this returned BUY TQQQ (substring `"ai"`
  in `"rain"` bullishly routed the high-score article to TQQQ).
  Post-fix → HOLD.
* `test_pain_in_title_does_not_alias_to_tqqq_via_ai` —
  duplicate-stem regression lock (different word, same letter
  pattern) so a future "but `"rain"` is a special case" simplifier
  can't sneak past with `"ai"` substring intact.
* `test_standalone_ai_token_still_maps_to_tqqq` — locks the
  canonical recovery path (`"AI demand surges"` → TQQQ); both old
  and new code paths PASS this. The fix must not regress the
  whole reason the keyword map exists.

Verified via OLD-path simulation that the rain/pain tests would
emit `BUY TQQQ` under substring matching, `HOLD` under
word-boundary — these are real red→green tests, not pass-either-way
no-ops. Full `tests/test_ml_live_opinion.py` 16 tests pass, plus
`tests/test_core_strategy.py` 145 tests pass alongside.

Staged ONLY `paper_trader/strategy.py` + `tests/test_ml_live_opinion.py`
per [[pt-concurrent-samerole-staging-race]] — concurrent sibling
agents were running with modified `paper_trader/dashboard.py` and
untracked `paper_trader/analytics/decision_paralysis.py` /
`tests/test_decision_paralysis.py` in the working tree at the
time, all correctly left out of this commit.

### Phase 2 — no feature this pass

Per the established discipline ("inventing a 25th builder is low
value"), Phase 2 yielded `features_added=0`. The codebase has ~80
analytics modules and ~120 endpoints already covering every
operator surface I could identify a gap in (forward / backward
risk, paralysis, attribution, calendar). The only genuine gap I
saw — operator-visible SWR cold-stall under host saturation — is
a known live pathology (recorded in review pass #34 finding #4)
that requires a dashboard / thread-pool change I did not want to
ship into a tree the concurrent sibling agent was editing.

### Phase 3 — live validation (~02:35 UTC)

Probed against the live runner just after the git-watcher fired
on this commit:

1. ✅ **`/api/build-info`** `boot_sha == head_sha == e3b5af9`,
   `stale: false`. The runner is on the just-pushed commit
   carrying the substring fix.
2. ✅ **`/api/portfolio`** $993.93 equity, $260.91 cash, 2
   positions (NVDA + TQQQ), `stale_marks: 0`, -0.61% vs $1000
   start. Healthy.
3. ✅ **`/api/runner-heartbeat`** HEALTHY — last decision 3m ago,
   within the 60m market-closed cadence.
4. ⚠️ **`/api/correlation`** SINGLE_NAME_RISK — NVDA is **60% of
   the stock book**; 1.18 effective independent bets, mean
   pairwise corr 0.692. Decision-relevant: NVDA earnings tomorrow
   (`/api/event-calendar` shows 1 imminent event 0.3d away). The
   prompt block + Discord `_concentration_line` already reach the
   trader; not actionable here.
5. ⚠️ **`/api/supervision`** `actionable: true, orphan: true,
   ppid: 1` — the documented [[pt-systemd-vs-manual-restart-spam]]
   pattern. Manual instance holds the singleton lock; systemd unit
   retries are correctly rejected. The Discord `_supervision_line`
   already surfaces this to the operator. Not a fix.
6. ✅ **`/api/no-decision-reasons`** 1/28 cycles NO_DECISION
   (3.6%), dominant cause `host_saturated`. Within tolerance —
   not the multi-hour storm the [[pt-no-decision-host-saturation]]
   memory describes.
7. ✅ **`/api/host-guard`** `state: CLEAR`, opus_count=4 (≤
   threshold), load_per_cpu ≈ 0.86, swap 65.8%. The 4-agent
   concurrent hybrid review (this run + siblings) sits right at
   the saturation boundary but has NOT tripped the guard this
   cycle.
8. ⚠️ **SWR cold-stall** — `/api/risk` / `/api/runner-heartbeat`
   / `/api/restart-recommendation` returned `{"warming": true}`
   for ~30-60s after each runner restart (this pass observed two
   restarts: a sibling's push then mine, both inside ~10 min).
   The prewarm thread spaces 40 endpoints 0.5s apart and the SWR
   pool is 6 workers — under host load, the first user poll after
   a restart races the prewarm. Documented in pass #34 finding
   #4; pure UX, not a bug.

### Counters

`bugs_fixed=1, features_added=0, user_findings=3` (the three ⚠
lines above — none new, all documented; the value is that this
pass *confirmed* the live state matches the documented memory
notes rather than catching anything novel).

### Invariants reaffirmed by this pass

- **#10** (single source of truth) — the patterns dict is built
  *from* `_WORD_TO_TICKER_LIVE` at module load (no parallel keyword
  list to drift from).
- **#2 / #12** (no hard limits, advisory-only) — the ML advisor
  remains advisory; the fix changes which articles its score
  reads from, not the gate logic.

## ML / backtest review pass (Agent 2, 2026-05-19, ~22:30 UTC) — OOS label-clamp consistency + bootstrap CIs

### Phase 1 — fix: `oos_rmse` label-clamping consistency (1 bug)

`paper_trader/ml/decision_scorer.py::train_scorer` clamps training
labels to ±`PRED_CLAMP_PCT` (50%) before fit (the symmetric label-
clamp block landed 2026-05-18; see the in-code comment for the
OOS-RMSE evidence). The scorer's `predict()` then clamps outputs to
the same band. **Three OOS paths were reading `forward_return_5d/10d/20d`
RAW**, so the operator-facing `val_rmse` vs `oos_rmse` pair in the
per-cycle skill ledger was NOT apples-to-apples: one ±175% MSTR /
3x-leveraged crash-rip week contributes `(50 − 175)² = 15,625` to OOS
MSE but `(50 − 50)² = 0` to val MSE, inflating `oos_rmse` by ~0.3–0.5
RMSE points on a typical 1000-row OOS slice — making honest training
look identical to overfit.

Mirror the training-side symmetric clamp in all three OOS metric paths:

- `paper_trader/validation.py::evaluate_scorer_oos` — clamped `rmse`
  is the new headline; `rmse_unclamped` is surfaced as a sibling
  field so the raw real-world error stays visible (additive, never
  destructive — same precedent as the `predict_with_meta` `raw`
  alongside `pred`).
- `run_continuous_backtests.py::_oos_rank_metrics` — near-no-op for
  Spearman rank-IC (extreme rows now tie at ±50 instead of extending
  rank space the model can never reach) but locked for
  cross-diagnostic consistency.
- `run_continuous_backtests.py::_oos_multi_horizon_metrics` — same
  clamp for 10d/20d horizons so a single extreme-week row no longer
  extends rank space on one horizon and not another.

The status-string contract `_train_decision_scorer` returns is
unchanged (token shape is what `_parse_scorer_status` /
`test_continuous.py::TestParseScorerStatus` lock); only the numeric
values are now honest. No retrain required — the clamp is at the
OOS-evaluation seam, not the training seam.

### Phase 1 — tests (8 added)

- `tests/test_validation.py::TestEvaluateScorerOosLabelClamp` (6
  tests) — ±175% / ±100% / -80% SELL clamp paths + in-band no-op +
  empty + untrained → both `rmse` and the new `rmse_unclamped` are
  honestly None.
- `tests/test_continuous.py::TestOosRankMetrics
  ::test_extreme_label_clamped_keeps_dir_acc_truthful` — locks the
  cross-OOS-path clamp consistency.

All 478 ml/backtest/scorer/oos tests pass (`pytest -k "ml or backtest
or scorer or oos"`).

### Phase 2 — feat: `paper_trader/ml/oos_bootstrap_ci.py` (95% CIs)

**The decisive gap:** every other OOS diagnostic in this codebase
(`skill_trend`, `baseline_compare`, `calibration`, `persona_skill`,
`sector_skill`, `_oos_rank_metrics`, `_oos_multi_horizon_metrics`)
reports POINT ESTIMATES. None of them answers the operator-decisive
question:

> Is the +0.11 OOS rank-IC the skill ledger reports each cycle
> actually above zero, or a coin flip on a ~1000-row OOS slice?
> Is `oos_rmse=11.83` statistically distinguishable from the
> σ(target)≈11.7 mean-predictor baseline, or within sampling noise?

`paper_trader/ml/oos_bootstrap_ci.py::bootstrap_ci` answers both via
a **non-parametric percentile bootstrap** over the SAME temporal-OOS
slice `_train_decision_scorer` evaluates against
(`split_outcomes_temporal` at `oos_fraction=0.2`):

1. Pre-compute (pred, realized) once per record (the expensive
   `scorer.predict()` call only runs `n` times, not `n * n_bootstrap`).
2. Apply the universal SELL sign-flip + the ±`PRED_CLAMP_PCT` label
   clamp (SAME path as the Phase-1 fix across `evaluate_scorer_oos` /
   `_oos_rank_metrics` / `_oos_multi_horizon_metrics`, so CI bounds
   describe the same target space the model was trained against —
   single source of truth across the whole OOS suite).
3. Resample N indices with replacement, recompute
   `(rmse, dir_acc, rank_ic)`, repeat `n_bootstrap` times (default
   1000), report empirical 2.5%/97.5% percentiles as 95% CIs.

Output (CLI):

```
$ python3 -m paper_trader.ml.oos_bootstrap_ci
[oos_bootstrap_ci] slice=oos  n=1482  n_bootstrap=1000  n_train=400  (95% CI)
  rmse            = 14.9982  [14.06, 15.82]
  dir_acc         = 0.6111   [0.586, 0.636]
  rank_ic         = +0.2245  [+0.171, +0.287]
  → rank_ic CI EXCLUDES 0 — directional edge is real
```

The `rank_ic` CI is the decisive verdict: excluding 0 means the
ordering edge is statistically real (however small in magnitude);
straddling 0 means the recent +0.11 reads are within sampling noise
of zero — the conviction gate (invariant #5, active at `n_train ≥ 500`)
would be sizing on a signal the data can't distinguish from random.

**Read-only by construction:** never trains, never touches
`decision_scorer.pkl` / `build_features` / `N_FEATURES` / any trade
path. Safe to run against the unattended continuous loop — like the
other `ml/*` diagnostics, it loads the deployed pickle + outcomes
file fresh and exits. CLI pattern mirrors `decision_scorer.py::main`
(int return + `--json` + `SystemExit`) so an operator gets one
muscle memory.

**Tests (13 added):**

- `TestStatusSentinels` — empty / untrained / below `MIN_PAIRS_FOR_CI`
  → verdict-keyed insufficient-data dicts (the calibration /
  baseline_compare honest-empty precedent).
- `TestPointEstimatesAndCIs` — perfect predictor → CI tightly
  around 0 RMSE / 1.0 dir_acc / 1.0 rank_ic; anti-predictor → CI
  excludes 0 (anti-skill detected); constant predictor → rank_ic =
  0 via tie-aware Spearman (no fabricated +1).
- `TestCorrectnessInvariants` — SELL sign-flip honoured; ±175% row
  clamps to ±50 so RMSE CI doesn't spike; single-row predict
  exception drops just that row.
- `TestDeterminism` — same seed → identical CI bounds (so
  cycle-over-cycle CI drift reflects real data shifts, not RNG
  noise).
- `TestJsonSafety` — result is JSON-serializable, no numpy
  floats/NaN leak through.
- `TestPercentileBounds` — CI bounds bracket the point estimate
  for any well-behaved input.

### How to run the new diagnostic

```bash
# 95% CI over the temporal-OOS slice (default)
python3 -m paper_trader.ml.oos_bootstrap_ci

# JSON output for piping into a dashboard / downstream check
python3 -m paper_trader.ml.oos_bootstrap_ci --json

# Faster, fewer bootstraps (still stable to ±~2pp on bounds)
python3 -m paper_trader.ml.oos_bootstrap_ci --n-bootstrap 300

# Evaluate against ALL records (in-sample + OOS) for comparison only —
# the OOS slice remains the trustworthy generalization view.
python3 -m paper_trader.ml.oos_bootstrap_ci --all-records

# Custom outcomes path (e.g. when worktree pickle is stale, point
# the live production outcomes file)
python3 -m paper_trader.ml.oos_bootstrap_ci \
  --outcomes /media/zeph/projects/paper-trader/data/decision_outcomes.jsonl
```

### Test commands (ML / backtest domain)

```bash
# Focused suite — fastest signal on ML/scorer/backtest changes (~5s)
cd /home/zeph/trading-intelligence/paper-trader && \
  python3 -m pytest tests/test_decision_scorer.py tests/test_backtest.py \
  tests/test_continuous.py tests/test_validation.py \
  tests/test_oos_bootstrap_ci.py -v

# Broad ML/backtest suite (~5min)
python3 -m pytest tests/ -v -k "ml or backtest or scorer or oos"
```

### Phase 3 — quant-perspective findings (5)

1. **Continuous loop is STOPPED.** Last activity in
   `/media/zeph/projects/paper-trader/continuous.log` was a SIGTERM
   at 2026-05-17, with the latest `backtest_runs.completed_at` at
   2026-05-18T18:05Z. Two days of stale skill metrics — restart the
   loop to refresh `data/scorer_skill_log.jsonl` /
   `data/baseline_skill_log.jsonl`.

2. **Two orphaned `status='running'` rows** in `backtest.db`
   (run_id 6238 from 2026-05-18T14:17Z, run_id 6243 from
   2026-05-18T18:45Z). The startup-time `_reap_orphaned_runs()` will
   sweep them on next launch (both >6h old now, comfortably past
   the `max_age_hours=6.0` guard).

3. **3 cycles lost ArticleNet training to lock exhaustion** —
   `inject err: database is locked` after the `_LOCK_RETRY_SLEEPS =
   (3.0, 8.0, 15.0)` retries exhausted. ArticleNet was retrained on
   the previous cycle's data those cycles. Not a bug (the retry +
   backoff is correctly tuned) — a heads-up for the operator that
   under heavy host saturation the inject step IS the bottleneck.

4. **Recent (last 2000) OOS forward-return-5d distribution is calmer
   than the documented ~σ=11.7 baseline:** `μ=+0.84, σ=6.13,
   p1=-15.56, p99=+18.69, |fr|>50: 0, |fr|>30: 3`. The Phase-1
   label-clamp fix is essentially a no-op on this slice (only 3/2000
   rows >30%, none >50%), but it's still correct for older tail
   data (the docs cite 25/5000 above 50% in earlier cycles).

5. **`oos_bootstrap_ci` on the full accumulated outcomes file
   surfaces real edge:** `rank_ic = +0.2245 [+0.171, +0.287]` —
   CI cleanly excludes 0. This is materially higher than the
   per-cycle `oos_ic` of 0.02–0.11 the skill ledger reports
   because the per-cycle view runs on smaller temporal slices
   (~1000 rows from one window). The full-file accumulated view
   shows the scorer's directional edge IS real over the broader
   corpus — even though magnitude RMSE remains above σ. A
   skeptical quant can now read this as a single line with a
   statistically defensible verdict.

### Constraints honoured

- **Invariant #5** (gate at `n_train ≥ 500`) — unchanged. The CI
  diagnostic is read-only and never modulates the gate.
- **Invariant #10** (single source of truth) — `_spearman` is the
  same `paper_trader.ml.calibration._spearman` every other rank
  diagnostic uses; the label-clamp is `PRED_CLAMP_PCT` from
  `decision_scorer.py`; `_to_float` semantics from the same module.
- **CLAUDE.md §6** — gate arms (`±10/±5/0` buckets, ×0.6 / ×0.85 /
  ×1.15 / ×1.3) untouched; pickle schema untouched; predict() scalar
  contract untouched; `build_features` / `SECTORS` / `N_FEATURES`
  untouched.
- **Concurrent staging discipline** ([[pt-concurrent-samerole-staging-race]])
  — staged ONLY the 4 files I changed (validation.py,
  run_continuous_backtests.py, the two test files) for Phase 1, and
  the 2 new files for Phase 2; concurrent Agent 1 / Agent 4 changes
  to `paper_trader/dashboard.py` and untracked
  `paper_trader/analytics/position_runrate.py` were correctly left
  out of both commits.

## Review pass — paper-trader core hybrid (2026-05-20, Agent 1, ~06:35 UTC) — half-day cadence + signals alias resolution

### Phase 1 — fix: `compute_interval` honored simple weekday/hour rule, not NYSE half-days or full holidays (1 bug)

`paper_trader/analytics/dynamic_interval.py::_is_market_hours` used a flat
9:30–16:00 weekday rule and `_is_session_open_window` only excluded weekends.
On the two known half-days (2026-11-27 day-after-Thanksgiving, 2026-12-24
Christmas Eve, both 13:00 ET close) the runner stayed on the 30-min
MARKET_OPEN cadence for the three hours past the 13:00 early bell — and on
full holidays (MLK, Christmas, etc.) it cycled OPEN-tier (1800s) and could
even fire SESSION_OPEN-tier (300s) cycles on a closed market. Both wasted
Opus capacity against a frozen book (and `market.is_market_open()` correctly
reads False the whole time, so the cadence here disagreed with the gate the
trader's own decision path keys off).

Fix delegates `_is_market_hours` to `market.is_market_open(now_utc)` (the
existing SSOT — handles half-days, holidays, weekends together) and guards
`_is_session_open_window` against `NYSE_HOLIDAYS_2026`. Lazy import inside
both helpers so unrelated test suites that import `compute_interval` don't
pull yfinance through `market.py`; falls back to the pre-fix simple rule on
any market-module import failure (degrade-safe; the hot-path discipline).

### Tests added (4)

`tests/test_dynamic_interval.py`:
- `test_half_day_afternoon_after_early_close_is_closed_cadence` —
  2026-11-27 14:30 ET → 3600s (was 1800s pre-fix).
- `test_half_day_before_early_close_still_open_cadence` —
  2026-11-27 11:00 ET → 1800s (no over-correction).
- `test_full_holiday_uses_closed_cadence` — 2026-12-25 10:00 ET → 3600s.
- `test_holiday_does_not_trigger_session_open_window` — 2026-01-19 (MLK
  Monday) 9:45 ET → 3600s (NOT 300s SESSION_OPEN).

All 8 dynamic-interval tests + 335 nearby tests
(strategy/store/market/half_day/signals/runner_heartbeat/negcache) pass.
Staged only `paper_trader/analytics/dynamic_interval.py` +
`tests/test_dynamic_interval.py` per [[pt-concurrent-samerole-staging-race]]
— concurrent sibling agents had `paper_trader/backtest.py` open with their
own work in flight, correctly left untouched in this commit.

### Phase 2 — feat: company-name → ticker alias resolution in `signals.py`

A headline like `"Nvidia surges to record on chip demand"` historically
extracted **zero tickers** (no `$cashtag`, no ALLCAPS `NVDA` token), so the
article never contributed to `ticker_sentiments(["NVDA"])` counts or to the
per-article `tickers` field Opus reads in the live prompt. Every held name's
news-volume was silently undercounted by every headline that referenced the
company by name rather than by symbol — a real gap in the live prompt the
decision engine reads.

Added `_TICKER_ALIASES: dict[str, tuple[str, ...]]` mapping 19 well-known
companies (NVDA→nvidia, AAPL→apple, MSFT→microsoft, AMZN→amazon,
GOOGL→alphabet, META→facebook, TSLA→tesla, MU→micron, AVGO→broadcom,
QCOM→qualcomm, INTC→intel, MRVL→marvell, TSM→tsmc / taiwan semiconductor,
ASML, AMAT→applied materials, LRCX→lam research, KLAC→kla-tencor,
LITE→lumentum, COIN→coinbase). Each alias matched as a **case-insensitive
whole-word regex** (mirroring `strategy._WORD_TO_TICKER_LIVE_PATTERNS` after
its 2026-05-20 substring-FP fix), so `"applesauce"` / `"pineapple"` do NOT
match AAPL.

Wired through three paths via the new single-source-of-truth `_alias_match`
helper:
- `_extract_tickers` — additive on top of the existing $cashtag + ALLCAPS
  extraction. Every downstream consumer of `signal["tickers"]` benefits
  (the Opus prompt's `tickers=NVDA,...` line, the source-edge attribution,
  etc.).
- `ticker_sentiments` (bulk) — outer loop now `(symbol OR alias)`-matches
  per article; one article hit per ticker per row (no double-counting).
- `get_ticker_sentiment` (single) — same `(symbol OR alias)` semantics.

### Tests added (14)

`tests/test_core_signals.py`:
- `TestTickerAliasExtraction` (9 tests) — `"Nvidia surges"` → `NVDA`;
  `"Apple expands services"` → `AAPL`; `"Tesla cuts prices"` → `TSLA`;
  multi-word `"Taiwan Semiconductor"` → `TSM`; case-insensitive
  (`"nvidia"` / `"Nvidia"` both → NVDA); word-boundary discipline
  (`"applesauce"` / `"pineapple"` → NOT AAPL); no duplication when ticker
  AND alias both present in the same headline; alias pass keeps the
  existing ALLCAPS + `_NOT_TICKERS` filters intact; empty-text contract
  preserved.
- `TestTickerSentimentsAliasPath` (5 tests) — company-name headline
  counts toward ticker `n`; `get_ticker_sentiment` alias path locks
  same behaviour; no double-count when symbol + alias both present;
  unrelated alias does NOT pollute sibling ticker (cross-ticker
  isolation); substring of alias does NOT falsely match in body scan.

73 pre-existing signals tests still pass; 218 across the
signals-related suites (`test_core_signals` +
`test_execute_and_fetch_signals` + `test_signals_lock_degrade` +
`test_signal_followthrough`) clean.

Staged only `paper_trader/signals.py` + `tests/test_core_signals.py`
per [[pt-concurrent-samerole-staging-race]]; concurrent sibling agent had
`paper_trader/backtest.py` modified with their own atomic-write fix in
flight, correctly left untouched (their `a7d93a8` shipped between this
agent's Phase 1 and Phase 3).

### Phase 3 — live validation (~06:35 UTC)

Live runner restarted onto sibling commit `a7d93a8` (the backtest atomic-
write fix) between Phase 2 push and Phase 3 probe. Healthy state:

1. ✅ **`/api/build-info`** `boot_sha == head_sha == a7d93a8`, `stale: false`.
2. ✅ **`/api/portfolio`** $993.93 equity, $40.28 cash, 2 positions, -0.61% vs
   $1000 start, stale_marks=0.
3. ✅ **`/api/runner-heartbeat`** HEALTHY — last decision 3m ago in the
   60m closed-market cadence; `decision_efficacy=PRODUCING` (19/20).
4. ✅ **Discord delivery** — `notify_verdict=HEALTHY`, last_ok 4s before probe.
5. ✅ **Singleton lock** `acquired` (pid=3580806).
6. ✅ **Database** `paper_trader.db` last_updated current (within 3m of
   probe); decisions over last 24h: 43 cycles → 1 fill, 40 hold, 2
   no_decision (4.7% rate; matches `/api/host-guard` empty rate).
7. ⚠️ **Concentration HIGH** — NVDA 66.59% of book + TQQQ 29.35%
   (top3 95.94%). `/api/correlation` confirms SINGLE_NAME_RISK,
   `/api/risk` flags `concentration_severity: HIGH`. Documented in
   the hourly Discord block; not actionable from this pass.
8. ⚠️ **Capital pinned** — 4.05% cash with NVDA earnings ~7h away
   (`/api/briefing` → `Market CLOSED — opens in 7h03m`). The book is
   structurally inelastic to a post-print catalyst. Known
   `capital_paralysis` pathology; already surfaces in Discord.
9. ⚠️ **Host saturation boundary** — `/api/host-guard` `state=CLEAR`,
   `opus_count=4` (this run + 3 sibling concurrent hybrid agents),
   `load_per_cpu=0.72`, `swap_used_pct=77.8%`. The 4-agent concurrent
   review is right at the saturation knee; documented host pathology.

### Counters

`bugs_fixed=1, features_added=1, user_findings=3` (all 3 ⚠ items above
are documented patterns surfaced for awareness; no novel pathology
caught this pass — the value is confirming the live state matches the
documented invariants on a freshly-restarted runner).

### Invariants reaffirmed by this pass

- **#10** (single source of truth) — the alias map's matching is centralised
  in `_alias_match`; both extraction and the two sentiment-scan paths consume
  the same helper, so they can never disagree on whether a headline "counts".
- **#2 / #12** (no hard limits, advisory-only) — both Phase 1 (cadence) and
  Phase 2 (alias) are observational/data-sourcing changes; neither modulates
  a trade decision or adds a position cap.

---

## 2026-05-20 core hybrid pass (Phase 1 fix + Phase 2 feature)

### Phase 1 — fix: 4 missing `@swr_cached` endpoints added to `_swr_prewarm`

Commit `1bcb95f`. `decision-paralysis`, `position-news-cooldown`,
`correlation-cluster-warning`, and `launcher-restart-loop` were
`@swr_cached` but never added to `_swr_prewarm`'s targets list — the same
freeze-triage cold-stall blind spot the
`test_swr_prewarm_coverage.py::test_every_swr_cached_endpoint_is_prewarmed`
invariant exists to catch. A trader opening these panels right after a
restart got `{"warming": true}` instead of real data for one full TTL
cycle. The fix adds the four `(name, handler)` tuples to the prewarm
target list; the test now passes.

### Phase 2 — feature: `/api/buying-power` endpoint

Commit `7e30005`. Surfaces `build_buying_power` (the lean prompt-facing
complement to `capital_paralysis`, already in the Opus decision prompt)
via a new dashboard endpoint, closing the established
prompt→dashboard→Discord trajectory the `buying_power` block has been
following one surface at a time.

Returns: `cash`, `deployed_pct`, `affordable[ticker, price, whole_shares]`,
`cheapest_name` + `cheapest_price`, and the `unlock` candidate
(biggest-loser-first cut priority). Pure single-source-of-truth wrapper
over `build_buying_power` (invariant #10); observational only — never
gates Opus, never caps a trade (#2/#12). SWR-cached 60s with prewarm
registration (test_swr_prewarm_coverage invariant). Scoped to the FULL
WATCHLIST so an operator sees affordability across the universe, not
just the lean `_names_in_play` subset the prompt block trims to.

Tests in `tests/test_buying_power_endpoint.py` pin the
CASH_CONSTRAINED live pathology shape, the DEPLOYABLE whole-share counts
under known mocked prices, the strict `int(cash // px)` floor (999.99 /
500 ⇒ 1 share, never 2), warm-hit cache behaviour, and the prewarm
registration invariant.

### Phase 3 — live validation

Live runner restarted onto commit `7e30005` between Phase 2 push and
Phase 3 probe (git-watcher healthy; `boot_sha == head_sha`). Healthy
state confirmed:

1. ✅ `/api/build-info` `boot_sha == head_sha == 7e30005`, `stale: false`.
2. ✅ `/api/portfolio` $993.93 equity, $40.28 cash, 2 positions
   (NVDA + TQQQ), -0.61% vs $1000 start, `stale_marks=0`.
3. ✅ `/api/runner-heartbeat` HEALTHY — last decision 4m ago in the 60m
   closed-market cadence; `decision_efficacy=PRODUCING` (18/20).
4. ✅ Discord delivery — `notify_verdict=HEALTHY`, last_ok ~5m before probe.
5. ✅ Singleton lock `acquired` (pid=3700750).
6. ✅ `/api/buying-power` (new endpoint) returned a complete affordable
   list with whole-share counts at $40.28 cash against live yfinance
   prices for the full watchlist — feature working end-to-end on the
   live book.

### Counters

`bugs_fixed=1, features_added=1, user_findings=0` (no novel pathology
caught; the value is confirming the live state matches the documented
invariants on a freshly-restarted runner running the new code).

### Invariants reaffirmed by this pass

- **#10** (single source of truth) — `/api/buying-power` is a pure
  formatter over `build_buying_power`; cash/deployed_pct/affordable
  computation lives in one builder and feeds both the Opus prompt and
  the new endpoint, so they can never disagree.
- **#2 / #12** (no hard limits, advisory-only) — the new endpoint is
  observational; it surfaces what cash *can* fund, never caps what Opus
  *should* do. The advisory contract follows the `capital_paralysis`
  precedent.

## 2026-05-20 feature-dev pass (Agent 4) — `/api/rising-unheld-themes` + `/api/sector-velocity-delta`

Two new dashboard surfaces that close the **rotation-opportunity** gap
in the velocity-decomposition triplet. `/api/held-theme-decay` (shipped
earlier today) answers *"is the catalyst on a ticker I OWN still alive
in the wire?"*; these two answer the operator's complementary
question: *"what catalyst am I missing?"* at the per-ticker and
per-sector level respectively.

The live book context driving this pass: NVDA 67% / TQQQ 29% / cash 4%
(95.94% top-3 concentration), pre-NVDA-earnings, with the wire
actively running on MU (33 articles, decayed score 44.7) and AMD (10+
articles) — neither of which the book has any exposure to.
`/api/idle-opportunity` already surfaces these during DROUGHT periods,
but the operator wants a *always-on* rotation surface that doesn't
depend on a HOLD streak first.

### `/api/rising-unheld-themes` — what catalyst am I missing?

The mirror image of `/api/held-theme-decay`. Per-UNHELD-ticker fresh-
vs-prior decayed news score; same fresh/prior 6h-window decomposition,
same MIN_FRESH / FADE_RATIO / BUILD_RATIO floors imported verbatim
(rotation-pair invariant: re-tuning the decay shape in one place
updates both surfaces in lockstep — never drift). Held tickers are
EXCLUDED at the builder before any per-ticker computation, so no
overlap with `/api/held-theme-decay` (invariant #10).

Verdict ladder adds one rung over held-theme-decay's:
- **DARK** / **FADING** / **STABLE** / **BUILDING** — same definitions.
- **BREAKING** — prior == 0 AND fresh ≥ `BREAKING_FRESH_SCORE` (3.0).
  A brand-new catalyst with no prior coverage at all. Distinct from
  BUILDING (which lumps "accelerating existing story" + "fresh from no
  prior"); for an UNHELD name the zero-prior case is qualitatively
  different — it's a brand-new story, not an accelerating one. The
  operator triages BREAKING faster than BUILDING. (held-theme-decay
  doesn't split this case because for a held name the distinction is
  irrelevant — "my catalyst exists" is the actionable signal either
  way.)

The bar for BREAKING (3.0) is intentionally above MIN_FRESH (1.0): a
single 1.0-score article with no prior shouldn't claim "brand new
theme." Three middle-quality articles or one high-relevance article
at half-life age is the desk's mental model.

Sort order: BREAKING > BUILDING > STABLE > FADING > DARK, within each
bucket by descending fresh_score (loudest catalyst on top). Aggregate
counts (`n_building`, `n_breaking`, `n_fading`, `n_dark`, `n_stable`)
span the full unheld universe; `themes` is capped to `max_themes`
(default 20).

Multi-ticker article split denominator counts ALL tickers (held +
unheld) — same anti-inflation rule news_themes / held_theme_decay use,
so the unheld weights here are directly comparable to the held weights
in held-theme-decay. A 2-ticker article with one held + one unheld
name contributes 0.5× the article's full weight to each; the held name
is then dropped (not redistributed).

Distinct from neighbours (invariant #10 — do not consolidate):
`/api/news-themes` (single-window snapshot, no velocity);
`/api/held-theme-decay` (same math, HELD tickers — this is the
complement); `/api/watchlist-opportunities` (curated-watchlist scan,
no velocity); `/api/idle-opportunity` (drought-gated point-in-time
surface; this is always-on velocity); digital-intern `trend_velocity`
(market-wide mention-rate Poisson, not score-weighted decayed).

Query params: `hours` (default 6, clamp 1..72), `min_score` (default
2.0, 0..10), `max_themes` (default 20, clamp 1..100). SWR-cached 60s
with prewarm registration (test_swr_prewarm_coverage invariant).
Advisory only — never gates Opus, adds no caps (#2/#12).

Pure builder `paper_trader/analytics/rising_unheld_themes.py::
build_rising_unheld_themes(articles, held_tickers, now=None,
fresh_window_hours=6.0, max_themes=20)`. Never raises (garbage row /
None / "not-a-list" tickers / unparseable timestamp → skipped, never
exception). Defense-in-depth backtest filter at the builder so a
leaked synthetic row cannot reach user-facing JSON.

Locked by `tests/test_rising_unheld_themes.py` (37 tests — SSOT
constant sharing with held_theme_decay, held-exclusion case-
insensitivity, multi-ticker split denominator cross-surface
invariant against held_theme_decay, state ladder edge cases per
verdict, BREAKING floor honesty, sort order with bucket precedence,
max_themes cap on rows but not on counters, top_rising
BREAKING-over-BUILDING precedence, backtest defense-in-depth on URL /
source / opus_annotation paths, response shape stability across
NO_DATA / OK / held-only-wire / garbage paths, param clamps) +
`tests/test_rising_unheld_themes_endpoint.py` (4 tests — Flask wiring
with synthetic-filter live fixture, NO_DATA when no articles.db,
query-param clamps, store-failure degradation).

### `/api/sector-velocity-delta` — which sector is the wire rotating into?

The per-BUCKET aggregation of the same velocity shape. Uses
`sector_heatmap.HEATMAP_BUCKETS` as the bucket SSOT (so a
`memory_core ACCELERATING` verdict here lines up with the memory_core
row of `/api/sector-heatmap`). Decay / window / ratio constants
imported from `held_theme_decay` / `news_themes` (rotation-pair
invariant extended to the bucket level — three velocity surfaces, one
decay shape).

Verdict ladder at the bucket level introduces stricter thresholds
than the per-ticker view because at sum-over-multiple-tickers the
noise floor is correspondingly higher:

- **DARK** — no qualifying articles in either window.
- **DECELERATING** — `fresh < prior × DECEL_RATIO` (0.7) AND prior was
  bucket-level prominent (prior ≥ `bucket_floor` = `MIN_FRESH × n_tickers`).
  Rotation-OUT signal. A bucket whose news flow dropped from "loud
  whole-sector" to "quiet" — the formerly-loud bucket is the more
  material signal than a marginal cool-off.
- **FADING** — ratio drop on a marginal bucket (prior below bucket floor).
  Informational only — a single ticker in the bucket cooling is not a
  rotation-OUT signal.
- **STABLE** — between FADE_RATIO and BUILD_RATIO.
- **BUILDING** — `fresh > prior × BUILD_RATIO` (1.43) AND fresh ≥
  `MIN_FRESH_SCORE` (1.0). Individual ticker acceleration, below the
  sector-level rotation floor.
- **ACCELERATING** — `fresh > prior × ACCEL_RATIO` (1.6) AND fresh ≥
  `bucket_floor` (n_tickers × MIN_FRESH). Rotation-IN signal — the
  whole sector is meaningfully louder, not just one ticker spiking.
  Also covers the "no prior, loud bucket-level fresh" case (a sector
  lighting up from nothing IS the strongest rotation-in signal).

`ACCEL_RATIO` (1.6) is strictly above `BUILD_RATIO` (1.43) so the two
verdicts can't collapse to the same threshold. The bucket-prominence
floor scales linearly with bucket size (4-ticker bucket needs 4×
MIN_FRESH to claim sector-level rotation; 1-ticker bucket uses
MIN_FRESH directly — consistent with the per-ticker view for that
edge case).

Multi-ticker articles split decayed weight evenly across ALL mentioned
tickers; cross-bucket articles attribute weight to every bucket the
mentioned tickers belong to (a 2-ticker MU/NVDA article puts 0.5× the
full weight into both memory_core and design). The bucket-level
article COUNT (fresh_n / prior_n) increments once per bucket
regardless of multi-mention. Cross-surface invariant locked in tests:
the bucket's `fresh_score` equals the sum of per-held-ticker
`fresh_score` values from held_theme_decay over the same input.

`top_accelerating` is the loudest rotation-IN bucket (sorted by
fresh_score); `top_decelerating` is the formerly-loudest rotation-OUT
bucket (sorted by prior_score — the bigger the bucket was before, the
more material the drop). `rotating_in` / `rotating_out` lists carry
the bucket names for one-line surfacing.

Distinct from neighbours (invariant #10 — do not consolidate):
`/api/sector-heatmap` (PRICE + RSI + 24h news COUNT per bucket;
point-in-time, no velocity); `/api/sector-pulse` (per-ticker, not
bucketed); `/api/sector-signal-fit` (bucket news vs book weight; a
fit measure, not a wire-velocity); `/api/sector-exposure` ($-weight
per bucket; inverse direction book→wire); all three per-ticker
velocity endpoints (this is the per-bucket aggregation).

Query params: `hours` (default 6, clamp 1..72), `min_score` (default
2.0, 0..10). SWR-cached 60s with prewarm registration. Advisory only
(#2/#12).

Pure builder `paper_trader/analytics/sector_velocity_delta.py::
build_sector_velocity_delta(articles, now=None,
fresh_window_hours=6.0, buckets=None)`. The `buckets` param is a
test seam — production always uses the `HEATMAP_BUCKETS` SSOT.
Never raises on garbage rows.

Locked by `tests/test_sector_velocity_delta.py` (31 tests — SSOT
constant sharing across all three velocity surfaces (news_themes /
held_theme_decay / sector_heatmap), verdict ladder edge cases per
bucket size, bucket-prominence floor scaling, multi-ticker split
denominator invariant, cross-bucket article attribution to every
bucket the tickers belong to, cross-surface invariant against
held_theme_decay weights, top_accelerating / top_decelerating
surfacing, sort order, backtest defense-in-depth on URL / source /
opus_annotation paths, response shape stability, ratio-None on
prior==0, top_fresh_ticker drill-down identification) +
`tests/test_sector_velocity_delta_endpoint.py` (3 tests — Flask
wiring with memory_core ACCELERATING fixture, NO_DATA when no
articles.db, query-param clamps).

### Phase 3 — test validation (no live restart pre-earnings)

NVDA earnings ~7h out; the new routes are hot-pluggable on the next
natural service restart — this pass deliberately did NOT trigger a
restart. All 108 tests across the new modules + adjacent regression
(rising_unheld + endpoint, sector_velocity + endpoint, held_theme_decay,
news_themes + endpoint, swr_prewarm_coverage) pass:

```
python3 -m pytest tests/test_rising_unheld_themes.py \
  tests/test_rising_unheld_themes_endpoint.py \
  tests/test_sector_velocity_delta.py \
  tests/test_sector_velocity_delta_endpoint.py \
  tests/test_held_theme_decay.py \
  tests/test_news_themes_endpoint.py \
  tests/test_swr_prewarm_coverage.py
# 108 passed
```

Broader targeted regression `pytest -k "theme or sector"` clean at
145 passed / 0 failed.

### Counters

`bugs_fixed=0, features_added=2, user_findings=0` (clean rotation
surface gap fill; no novel pathology caught, but the live MU / AMD
unheld coverage will surface as BREAKING / BUILDING the moment the
service picks up the new code).

### Invariants reaffirmed by this pass

- **#10** (single source of truth) — three velocity surfaces
  (held_theme_decay, rising_unheld_themes, sector_velocity_delta)
  share decay half-life, fresh-window width, MIN_FRESH / FADE_RATIO /
  BUILD_RATIO via verbatim imports. Sector-bucket SSOT comes from
  `sector_heatmap.HEATMAP_BUCKETS` — the same map the heatmap UI
  uses. Held-set normalization (case-insensitive upper-strip) is the
  same shape both surfaces apply. The rotation-pair invariant is
  pinned by SSOT-import tests on both new modules.
- **#2 / #12** (no hard limits, advisory-only) — both endpoints are
  observational. Neither modulates a trade decision, neither adds a
  position cap, neither feeds the Opus prompt (no auto-injection;
  the operator reads the UI to triage rotation).
- **prewarm == @swr_cached** — both endpoints registered in the
  prewarm list; locked by test_swr_prewarm_coverage. First poll
  right after a restart returns real data, not `{"warming": true}`.
- **Defense-in-depth backtest filter** — both builders include the
  `_is_synthetic` SSOT shape (`backtest://` URL / `backtest_*` /
  `opus_annotation*` source) so a leaked synthetic row cannot reach
  user-facing JSON even if a future caller forgets the SQL clause.

## 2026-05-20 hybrid pass — ML/backtest (Agent 2)

### Phase 1 — fix: `_buy` truthiness on stop_loss/take_profit top-ups (1 bug)

`paper_trader/backtest.py::_buy` accumulate-into-existing branch used
`if stop_loss:` / `if take_profit:` to decide whether to overwrite the
existing position's stop_loss / take_profit. Truthiness silently dropped
an explicit `0.0` update while the **new-position branch** stores `0.0`
unconditionally via the dict literal — so accumulating into a position
with the same `0.0` diverged from a fresh open (the prior non-zero value
survived a deliberate wipe attempt). Switched to `is not None` so a
`None` caller still skips (legacy semantics preserved) but a real `0.0`
overwrites the prior value as the schema intends. `_enforce_risk_exits`'s
`if sl and px <= sl:` is intentionally left as truthiness — there `0.0`
correctly means "no real stop" (never fires when price > 0); the new
discipline applies only to the **assignment** path, where `0.0` is a
deliberate wipe-intent that must overwrite. Mirrors
`_execute_decision`'s `isinstance(..., (int, float))` value-check
discipline.

### Phase 1 — tests added (8)

Locked the new (None-skips, 0.0-applies, real-value-overwrites)
semantics with three regression tests in
`tests/test_backtest.py::TestBuyExistingPositionTruthiness` so a future
"looks-sensible" rewrite back to truthiness fails. Added five more
edge-case regression tests that lock previously-untested behaviour:
`TestComputeDecisionOutcomes` (forward_return_5d arithmetic on the
deterministic `synthetic_prices` fixture — pinned at +10.00% from
day-0 BUY at 100 → day-5 100+2·5=110, plus the
end-of-window window-overflow drop and FILLED-only outcome capture),
and `TestMlDecideNewsRanking` (high-score article wins over low-score
on the same ticker; empty-articles → HOLD-or-persona-buy).

### Phase 2 — feat: `paper_trader/ml/news_volume_skill.py`

**The gap none of the existing 30+ ML/backtest diagnostics filled.**
The aggregate `_oos_rank_metrics` (one number across all
news-availability slices) hides a load-bearing question every existing
diagnostic structurally misses: *is the scorer's near-zero OOS skill the
same in news-poor and news-rich conditions, or does adding news context
move it?* `feature_importance` answers the model's view ("if I scramble
news columns, how much skill is lost?"). `attribution_audit` answers the
model's per-record view ("how strongly does the model lean on news
features?"). **Neither slices the realized OOS outcomes by news
availability** and asks whether the model's predictions are more
accurate when more news exists. This module fills exactly that gap
(strictly orthogonal — no SSOT overlap with any sibling).

`news_volume_skill(scorer, records) → dict` buckets decisions by
`news_article_count` into four canonical bins (matching the
`_compute_decision_outcomes` and `build_features` clamp semantics):

| Bucket | Range | Meaning |
|--------|-------|---------|
| `no_news` | None or `0..1` | Decisions made without supporting news |
| `sparse` | `1..3` | 1-2 articles |
| `moderate` | `3..10` | 3-9 articles |
| `dense` | `10+` | 10+ articles (cnt_v clamps at 20) |

Per bucket it reports tie-aware Spearman rank-IC, directional accuracy,
mean action-aligned realized return, and an `EDGE` / `WEAK_EDGE` /
`NO_EDGE` / `INVERTED` / `INSUFFICIENT` verdict (mirrors the
`action_skill` per-action verdict ladder, same thresholds). Overall
verdict combines the trajectory across sufficient buckets:

| Overall verdict | Meaning |
|-----------------|---------|
| `INSUFFICIENT_DATA` | < `MIN_RECORDS`=30 aligned outcomes OR < 2 sufficient buckets |
| `NEWS_VALUE_MONOTONIC_POSITIVE` | rank-IC rises monotonically with news volume (spread ≥ `BUCKET_SPREAD_TOL`=0.05) — scorer extracts more signal from news-rich decisions |
| `NEWS_VALUE_MONOTONIC_NEGATIVE` | rank-IC FALLS monotonically — adding news context degrades predictions (audit upstream news scoring pipeline) |
| `NEWS_VALUE_INVARIANT` | spread < `BUCKET_SPREAD_TOL` — news doesn't differentiate skill (model leans on quant signals) |
| `MIXED` | non-monotone, no clean effect — per-bucket table is the honest read |
| `HAS_INVERTED_BUCKET` | ≥1 bucket has rank-IC ≤ `-IC_GOOD`=-0.10 (operator-actionable red flag, surfaced first) |

**Operational discipline mirrors the existing read-only ML diagnostics**
(`action_skill` / `calibration` / `skill_trend`): no train, never touches
`decision_scorer.pkl` / `decision_outcomes.jsonl` / `build_features` /
`N_FEATURES` / trade path, never raises on bad input — safe to run
against the live unattended continuous loop and cannot break pickle
compatibility. Reuses `ml.calibration._spearman` (SSOT — tie-awareness
is load-bearing because clamped ±50 predictions tie at the empirical
label support). Reuses `validation.split_outcomes_temporal` so the slice
matches what `_train_decision_scorer` reports `oos_rmse`/`oos_ic` on
(the SAME holdout the per-cycle skill ledger trends).

```bash
cd /home/zeph/trading-intelligence/paper-trader && python3 -m paper_trader.ml.news_volume_skill
cd /home/zeph/trading-intelligence/paper-trader && python3 -m pytest tests/test_news_volume_skill.py -v
```

CLI exits 0 on healthy/insufficient verdicts, exits 2 if any bucket is
INVERTED (so an operator/cron can branch on it, exactly like
`action_skill._cli` / `calibration._cli`).

Pinned by 30 unit tests in `tests/test_news_volume_skill.py`: bucket
boundaries (None/0 → no_news, 1-2 → sparse, 3-9 → moderate, 10+ →
dense; NaN/negative/unparseable rejected to None and dropped), per-bucket
verdict thresholds (full ladder at IC_GOOD / IC_MIN boundaries),
overall-verdict logic (monotonic positive/negative, INVARIANT, MIXED,
HAS_INVERTED_BUCKET, INSUFFICIENT_DATA), universal SELL sign-flip,
non-finite forward_return drop, single-sufficient-bucket →
INSUFFICIENT_DATA, missing/corrupt outcomes file degrades to
INSUFFICIENT_DATA not a crash.

### Phase 3 — quant-perspective findings (6)

Ran the diagnostic against the live `/media/zeph/projects/paper-trader/data/decision_outcomes.jsonl`
(7413 outcomes, sim-dates 1996-02-01 → 2021-06-01) and the per-cycle
ledgers:

1. **System is IDLE since 2026-05-18.** `continuous.log` mtime, latest
   `scorer_skill_log.jsonl` row (cycle 4, 2026-05-18T18:06:04), and the
   latest `backtest_runs` row all converge on this date. No cycles have
   run in ~2 days. (Not a regression — just operational state worth
   flagging to a reading quant.)
2. **The deployed `decision_scorer.pkl` has `n_train=400`**, which is
   **BELOW** the documented gate-active threshold of 500 (invariant #5)
   — yet the most recent skill-log row reports `train_n=3959` and
   `gate_active=True`. The deployed pickle has been overwritten with a
   smaller-corpus model since cycle 4 (both `data/ml/decision_scorer.pkl`
   and `/media/zeph/projects/paper-trader/data/ml/decision_scorer.pkl`
   read n=400). So the LIVE gate is effectively **INACTIVE** despite
   the cycle ledgers suggesting otherwise. Worth investigating the
   overwrite path before relying on the gate.
3. **`news_volume_skill` (the new diagnostic) reveals real
   news-driven skill on the actual outcomes corpus** that contradicts
   the recurring `MLP_NO_BETTER_THAN_TRIVIAL` view when looked at via
   this slice. OOS (1482 records, slice=oos):
   - `no_news`: ic=+0.19, dir_acc=0.60, n=1293 → EDGE
   - `sparse`: ic=+0.46, dir_acc=0.71, n=158 → EDGE (surprisingly strong)
   - `moderate`: ic=+0.28, dir_acc=0.74, n=31 → EDGE (small sample)
   - `dense`: insufficient data (no records)
   - Overall verdict: `MIXED` (non-monotone, spread 0.267)
   - **dir_acc rises monotonically with news count** (0.60 → 0.71 →
     0.74) — a real signal that news context measurably helps the
     scorer's directional skill.
4. **Mean action-aligned realized return DROPS as news volume rises**
   (+0.65 → +0.79 → -1.11). News-rich decisions appear to be on
   more-volatile / overcrowded / exhausted moves on average, even
   though the scorer's rank skill is BETTER in those buckets. The
   sample size in `moderate` (n=31) is small enough that this could be
   sampling noise — worth re-checking when `dense` accumulates.
5. **News features are heavily under-represented in training**: 95.6%
   of outcomes have `no_news` (None or 0 articles), 3.9% sparse, 0.5%
   moderate, 0% dense. The deployed model has scarce training examples
   for news-rich features (`news_urgency`, `news_article_count`) — yet
   finding #3 says the news-rich buckets carry the most rank skill.
   This is the strongest argument for a digital-intern news pipeline
   density improvement: more news-rich training examples could lift the
   scorer's overall OOS skill above the documented
   `MLP_NO_BETTER_THAN_TRIVIAL` plateau.
6. **ArticleNet feedback loop is broken on >75% of recent cycles**:
   in 35 `ml:` log lines, 13 fail with `inject err: database is locked`
   / `database locked after 4 attempts` (37%), and 14 succeed at inject
   but fail with `trainer timeout (injected N)` because the digital-intern
   trainer subprocess is killed by the 120s timeout (40%). Only ~22% of
   cycles complete the full inject + retrain pass. The retry budget
   (4 attempts × ~15s) and the trainer timeout are too tight for the
   digital-intern daemon's lock-holding patterns and dataset size.

### Counters

`bugs_fixed=1, features_added=1, user_findings=6` — one principled
correctness fix for `_buy`'s top-up truthiness; one genuinely orthogonal
diagnostic (`news_volume_skill`) that fills the only remaining unfilled
slice in the 30+ ML/backtest diagnostic suite; six independent live
observations including a real concern (deployed `n_train=400` < gate
threshold while ledgers say `gate_active=True`).

### Invariants reaffirmed by this pass

- **#5** (`DecisionScorer` only gates at `n_train >= 500`) — finding #2
  shows the LIVE pickle violates this threshold; the gate condition is
  load-bearing and the source-of-truth threshold remains correct.
- **#10** (SSOT) — `news_volume_skill` reuses `_spearman` from
  `calibration`, `_to_float` from `decision_scorer`, and
  `split_outcomes_temporal` from `validation` — never re-derives any
  rank-correlation, NaN handling, or holdout split. Cannot drift.
- **Read-only diagnostic discipline** — `news_volume_skill` follows the
  established `action_skill` / `calibration` / `skill_trend` pattern:
  no train, no pickle touch, never raises on bad input. Cannot perturb
  the unattended continuous loop or break pickle compatibility.
- **Universal SELL sign-flip** — `news_volume_skill._aligned_pred`
  mirrors `action_skill._aligned_pred` / `persona_skill._aligned`
  byte-for-byte (`-forward_return_5d` on SELL, prediction NOT flipped).
  Test `test_sell_action_flips_realized_sign` pins the convention.


## 2026-05-20 core hybrid pass — `_extract_tickers` ALLCAPS false positives + `/api/position-rationale` (Agent 1)

### Phase 1 fix — `signals._extract_tickers` strips shouted-company-name false positives

A headline shouted in all caps (`"APPLE BEATS EARNINGS today"`) extracted
BOTH `APPLE` (fake — Apple's real ticker is `AAPL`, never `APPLE`) and
`AAPL` (via the existing alias path). Opus then read
`tickers=APPLE,AAPL` in the live prompt's TOP SCORED SIGNALS block —
non-existent-ticker pollution that confuses the decision engine. Same
pattern affected the three other alias entries whose `len<=5` lowercase
alias upper-cases to something OTHER than the canonical ticker:
`TESLA→TSLA`, `INTEL→INTC`, `TSMC→TSM`. `ASML→ASML` (alias == ticker) is
a legitimate extraction and stays unfiltered.

Fix: precompute `_ALIAS_UPPER_FALSE_POSITIVES` from the existing
`_TICKER_ALIASES` table at import time:

```python
_ALIAS_UPPER_FALSE_POSITIVES = frozenset(
    a.upper()
    for tk, aliases in _TICKER_ALIASES.items()
    for a in aliases
    if 2 <= len(a) <= 5 and a.upper() != tk
)
```

…then add membership-check to the `_TICKER_RE` filter alongside
`_NOT_TICKERS`. The alias-derived ticker (`AAPL`, `TSLA`, `INTC`, `TSM`)
is unaffected — only the shouted-company-name form is stripped. New
aliases automatically inherit the protection at module load.

Locked by `tests/test_core_signals.py::TestExtractTickersAliasPath`:
`test_allcaps_company_name_does_not_pollute_with_fake_ticker` covers
the four documented collisions; `test_alias_filter_keeps_alias_that_
equals_real_ticker` pins the ASML carve-out;
`test_alias_false_positive_set_only_contains_distinct_aliases` locks
membership of the precomputed set itself so a future alias addition
with a same-as-ticker entry can't silently regress ASML.

### Phase 2 feature — `/api/position-rationale`

See the endpoint table entry. Sibling-orthogonal to
`/api/position-attention` (which answers *did Opus look?*) and
`/api/thesis-drift` (which re-tests the entry thesis): this surfaces the
**most-recent concrete Opus rationale** per held position by parsing the
JSON envelope `strategy.decide()` already writes into
`decisions.reasoning`. Today the operator must scroll the decision feed
by hand; this puts the answer one HTTP read away. 13 unit tests in
`tests/test_position_rationale.py`. Live smoke against the running book
confirms the desired shape (NVDA → HOLD @ 0.7 conviction with the
current reasoning; TQQQ → MISSING_RATIONALE — a genuinely actionable
finding flagged in Phase 3).

### Phase 3 live findings (user perspective)

1. **Runner is N commits behind HEAD between git-watcher polls** —
   normal cadence (3-min polls + 2-min boot grace). The new
   `/api/position-rationale` 404s until the runner restarts. Not a bug;
   expected behaviour of the deferred-restart pattern.
2. **TQQQ position is NEGLECTED** — `/api/position-attention` already
   flagged it (1.29d held, no real Opus look). Once the new endpoint
   goes live the same position will read MISSING_RATIONALE. The
   operator can now see *both* "Opus hasn't looked" AND "no rationale
   on file for the current state" at the per-position level — the
   exact double-blind the two endpoints together close.
3. **Host saturation at the edge** — 4 concurrent Opus subprocesses
   (== `DEFAULT_MAX_OPUS`), `load_per_cpu`=2.04 (below 4.0 threshold),
   6192 MB available. `recent_starvation_rate` 8.5% over last 59
   decisions (below `STARVATION_RATE_FLOOR`=25% so not flagged STARVED).
   Live HYBRID multi-agent runs nudge the box close to the gate; the
   pre-flight + mid-call guards are working as designed.
4. **Discord delivery briefly DEGRADED then recovered** — earlier
   heartbeat showed `openclaw timeout (60s)`; subsequent send
   succeeded. No persistent outage; the existing health tracker
   surfaces the flap as a single failed attempt.

### Counters

`bugs_fixed=1, features_added=1, user_findings=4`.

### Invariants reaffirmed by this pass

- **#1** (live-only article filter) — `_extract_tickers` operates on
  the title/summary text *after* the SQL live-only clause has run. The
  fix is post-extraction filtering of alias false-positives; it does
  not weaken any backtest-isolation gate.
- **#2 / #12** (advisory-only, no caps) — `/api/position-rationale` is
  a pure read of `open_positions` + `recent_decisions`. Never gates
  Opus, never injected into the decision prompt, no risk caps. The
  `position_attention` precedent.
- **#10** (SSOT) — `_parse_action_ticker` in `position_rationale` is a
  byte-equivalent local copy of `position_attention`'s helper (and
  matches `dashboard._parse_action_ticker` per AGENTS.md invariant #11
  on the free-text decision label). Kept local so the analytics module
  has no dashboard dependency — the `round_trips` / `decision_forensics`
  pattern. The JSON envelope shape is the same one `strategy.decide()`
  writes (read by `decision_health` and the dashboard already).


## 2026-05-21 feature-dev pass (Agent 4) — `/api/thesis-keyword-lift`

The open-vocabulary mirror of `/api/catalyst-class-autopsy`. That builder
labels each closed round-trip with a fixed taxonomy (ML_ADVISOR /
EARNINGS_PLAY / ANALYST_PT / TECHNICALS / MACRO / BREAKING_NEWS /
PUNDIT / SECTOR_SYMPATHY / CONCENTRATION / UNCLASSIFIED) chosen by the
analytics author. This learns the dominant keywords directly from the
trader's own `entry_reason` text — the pattern "trades whose reason
mentions 'guidance' win 80% of the time vs 50% baseline" surfaces here
even when no class was ever written for it.

**Lift definition** (`lift_pp`): percentage-point delta vs the pool
baseline, NOT a multiplicative ratio. For each keyword K that appears
in at least `min_kw_occurrences` (default 3) closed round-trips:

```
win_rate(K)        = n_winners_with_K / n_trips_with_K
baseline_win_rate  = n_winners_total / (n_winners_total + n_losers_total)
lift_pp            = (win_rate(K) - baseline) * 100        # bounded [-100, +100]
```

PP-delta (not ratio) was chosen because a 0-loser / all-winner keyword
has a well-defined finite lift (+baseline_pp) — a ratio formula
would be undefined exactly when the signal is strongest.

**Discipline**:
- Composes `round_trips.build_round_trips` (SSOT, AGENTS.md #10) — never
  recomputes P&L / hold-time. Pure / no LLM / no DB — never raises.
- Joins `entry_reason` verbatim from the contributing trade row by DB
  `id` (the loser/winner_autopsy / catalyst_class_autopsy "surface the
  reason verbatim, never NLP-parse it for trading logic" discipline).
- First-buy convention (`pick_last=False`) — add-on BUY rows after the
  opening thesis do NOT contribute keywords. Mirrors
  `winner_autopsy._reason_for`.
- Exit reason is **never** tokenised — this is a *thesis* keyword lift.
- `set(...)` per round-trip — 'earnings earnings earnings' counts once.
- Stopword list is *trading-text-specific* (the generic English list
  under-covers "this trade", "the position", "hold" — the open_reason
  boilerplate that would otherwise dominate every ranking). Frozen on
  import so a test that mutates the set can't bleed across runs.
- Pure-numeric tokens dropped ("2026", "80" — no thesis content; "rsi60"
  / "q1" mixed-alphanum survive). 3-char minimum length.
- Washes (sub-cent PnL) excluded from both sides — same convention as
  `winner_autopsy` / `loser_autopsy` / `trade_asymmetry`.

**STABLE gate**: `STABLE_MIN_PER_SIDE=4` on EACH side. Below this it is
EMERGING (numerics + ranked lists emitted, `verdict=None`). At STABLE,
the `verdict` field surfaces the single most-positive-lift keyword —
one word the operator can scan instantly. Verdict is withheld in the
degenerate "STABLE pool but no keyword cleared baseline" case (e.g.,
the only keyword appears equally in wins and losses → 0pp lift → no
verdict, headline says so).

**Ranking determinism**: lift_pp DESC for winners, lift_pp ASC for
losers, ties broken by n_total DESC then keyword ASC. A two-run
identical-data input must produce a byte-identical ranked list (the
disposition test pins this).

**Advisory only** — never gates Opus, never injected into the decision
prompt, no caps (AGENTS.md #2/#12). Same observational discipline as
loser_autopsy / winner_autopsy / catalyst_class_autopsy.

```sh
curl -s 'http://localhost:8090/api/thesis-keyword-lift' | python3 -m json.tool
```

Pinned by `tests/test_thesis_keyword_lift.py` (31 cases): tokenize
boundary (stopwords / numeric / short-tokens / non-string-defensive),
empty-pool and one-sided pools (NO_DATA / NO_WINS / NO_LOSSES), wash
exclusion, baseline arithmetic, pp-not-ratio formula, min_kw_occurrences
threshold, STABLE gate (3W/4L = EMERGING, 4W/4L = STABLE), degenerate
"no positive lift" verdict suppression, deterministic tie-break (sample
size then alphabetical), top_n cap (each side independently),
entry-reason-only discipline (exit reason cannot leak; add-on BUY cannot
leak), envelope shape stability.

---

## 2026-05-20 feature-dev pass (Agent 4) — `/api/catalyst-class-autopsy` + `/api/peer-earnings-shock`

Two orthogonal additions that close gaps the existing ~135-endpoint
surface still left open.

### `/api/catalyst-class-autopsy` — which ENTRY thesis classes win?

The desk question `/api/loser-autopsy` and `/api/winner-autopsy` don't
answer. Both of those classify the **exit** behaviour (KNIFE_CATCH /
WHIPSAW / SLOW_BLEED / STOPPED_OUT and HOME_RUN / SCALP / SLOW_GRIND /
TARGET_HIT respectively) — *how the trade was closed*. Neither
classifies the **entry thesis** — *which catalyst TYPE motivated the
open*. The verbatim entry rationale is the strongest learning signal
in the ledger (the live DRAM whipsaw's reason carried 6 classes:
`Citi bullish + HSBC/Melius PT + Cramer buy signal + ML advisor + NVDA
earnings + semis concentration` → ANALYST_PT, PUNDIT, ML_ADVISOR,
EARNINGS_PLAY, SECTOR_SYMPATHY, CONCENTRATION). Multi-labels each
closed round-trip by every matched class, then surfaces per-class
win-rate vs the pool baseline.

Taxonomy (regex, word-boundary, case-insensitive — substring traps
guarded against the same way ALLCAPS / news-themes pipelines guard):

- **ML_ADVISOR**       — "ML advisor", "DecisionScorer", "median +%
  alpha", scorer-flags/-gates/-nudges.
- **EARNINGS_PLAY**    — "earnings", "EPS", "Q1..Q4 report", "print",
  "guidance cut/raise".
- **ANALYST_PT**       — "price target", "$N PT", "PT raise/cut",
  "upgrade/downgrade/reiterate", explicit bank names (Citi / JPM /
  Goldman / HSBC / BofA / Morgan Stanley / Wells Fargo / Barclays /
  Melius / Wedbush / Piper / Bernstein / Jefferies / UBS / Deutsche
  Bank / Susquehanna / Mizuho / Raymond James / Stifel / Oppenheimer /
  Truist / Cantor / Loop Capital).
- **TECHNICALS**       — RSI / MACD / golden-cross / death-cross /
  moving-average / breakout / support / resistance / bollinger /
  stochastic / overbought / oversold / 52-week-high|low.
- **MACRO**            — FOMC / Fed / Powell / rate decision / CPI /
  PPI / PCE / NFP / DXY / GDP / recession.
- **BREAKING_NEWS**    — "breaking" / "just crossed" / "tape bomb" /
  "wire reports".
- **PUNDIT**           — Cramer / Buffett / Druckenmiller / Burry /
  Ackman / Tepper / Dalio / Munger / Cathie Wood / Loeb / Einhorn /
  Klarman / Marks / Soros / Icahn.
- **SECTOR_SYMPATHY**  — "sympathy" / "peer strength|weakness" /
  "sector rotation|momentum|move|leadership" / "cohort" / "leveraged
  cousin|peer".
- **CONCENTRATION**    — "concentration" / "over|under weight" /
  "trim" / "raise dry powder" / "cash headroom" / "rebalance".
- **UNCLASSIFIED**     — non-empty rationale matching nothing (kept
  bucketed so the pool WR doesn't silently inflate by dropping the
  worst un-rationalized trades).

Per-class row: `n_trips`, `n_wins`, `n_losses`, `win_rate_pct`,
`total_pnl_usd`, `avg_pnl_usd`, `avg_pnl_pct`, `median_hold_days`,
`verdict` (`UNSTABLE` / `BIASED_WINNER` / `BIASED_LOSER` /
`NEUTRAL`). Verdicts are sample-size-gated below
`STABLE_MIN_TRIPS_PER_CLASS=4` (the `loser_autopsy` /
`trade_asymmetry` STABLE idiom, dropped from 8 to 4 because per-class
N is necessarily smaller than the pooled losers count). The bias
band is `BIASED_WR_DELTA_PCT=15.0%` — a class WR has to beat/undershoot
the pool baseline by this margin to flip BIASED; inside the band is
NEUTRAL. The pool baseline anchors against **this trader's**
disposition (a 40% pool WR makes a 50% class WR a BIASED_WINNER, not
NEUTRAL).

Multi-class trips contribute to every bucket — a trip with classes
{ML_ADVISOR, ANALYST_PT} counts in BOTH (the operator wants "ML+PT
combined" win-rate, not "ML alone" win-rate). `n_scored` counts
trips, not bucket-fills.

Pure SSOT `analytics/catalyst_class_autopsy.py::
build_catalyst_class_autopsy`. Consumes `round_trips.build_round_trips`
verbatim (AGENTS.md #10); joins the entry reason back by DB id (the
`loser_autopsy._reason_for(pick_last=False)` pattern). Advisory only —
never gates Opus, never injected into the decision prompt, no caps
(#2/#12 — the `loser_autopsy` / `winner_autopsy` / `trade_asymmetry`
precedent). Never raises on garbage rows.

Locked by `tests/test_catalyst_class_autopsy.py` (31 tests — taxonomy
classifier per class, multi-label on the real DRAM trade rationale,
deterministic taxonomy order, case-insensitive, word-boundary
substring guard, EMERGING-vs-STABLE gate, pool baseline anchor with
4W/4L mixed-class fixture (ML wins 100% vs pool 50% → BIASED_WINNER,
ANALYST_PT loses 0% → BIASED_LOSER), multi-class bucket fill arithmetic,
UNCLASSIFIED bucket, zero-pnl-is-not-a-win parity with
`loser_autopsy`, never-raises on garbage rows, response shape
stability, headline correctness, band-edge NEUTRAL precision)
+ `tests/test_catalyst_class_autopsy_endpoint.py` (3 tests — Flask
wiring with DRAM-replay fixture, NO_DATA on empty store, store-
failure-degrades-not-raises).

### `/api/peer-earnings-shock` — indirect 1σ on held ETFs from peer prints

The fusion question `/api/earnings-shock` (direct held-name σ) and
`/api/etf-lookthrough` (hidden indirect $-exposure) don't answer.
The arithmetic is one multiplication — `indirect_usd × sigma_pct /
100` — but the desk has to manually fuse
`etf-lookthrough.etf_positions[i].breakdown[j].indirect_usd` with the
matching `earnings_shock.events[k].sigma_pct` row. Nothing on the
live dashboard does that fusion today. The live 2026-05-20 book runs
29% TQQQ; every mega-cap tech print pressures the basket
indirectly. NVDA earnings tonight + TQQQ 9% NVDA × 3x lev means
~$40 of indirect NVDA exposure on top of the $447 direct, and σ 7%
makes the indirect 1σ ±$2.80 — a real number no current surface
quantifies.

Per (ETF, underlying) row:
`underlying`, `weight_pct`, `days_away`, `indirect_usd`,
`sigma_pct`, `indirect_sigma_dollar`, `indirect_sigma_book_pct`,
`row_state` (`OK` / `INSUFFICIENT_SIGMA` — the `earnings_shock`
row-level σ-withheld discipline). Inverse ETFs (SQQQ / SOXS / SPXS /
FNGD / TECS) carry NEGATIVE `indirect_usd` (they short their
underlyings); the aggregate uses `|indirect_sigma_dollar|` (the
`earnings_shock` "worst-case all-surprise-same-way" convention — a
7% NVDA down move means SQQQ goes UP 21%, still real $-at-risk).

Per-ETF aggregate: `sum_indirect_sigma_dollar`,
`sum_indirect_sigma_book_pct`, `n_peer_events`. Book-wide aggregate:
`total_indirect_sigma_dollar`, `total_indirect_sigma_book_pct`,
verdict band `LOW / MODERATE / SEVERE` (`MODERATE_BOOK_PCT=2.0%`,
`SEVERE_BOOK_PCT=5.0%` — same calibration shape as `earnings_shock`
so the operator's mental model carries across the three pre-earnings
surfaces).

State ladder mirrors `earnings_shock` / `etf_lookthrough`:
`NO_DATA` / `NO_ETF_HELD` / `NO_PEER_EVENTS` / `OK`. Headline picks
the single (ETF, underlying) pair with the loudest
indirect_sigma_dollar — the most actionable line.

Pure SSOT `analytics/peer_earnings_shock.py::build_peer_earnings_shock`.
Composes `build_etf_lookthrough` and `build_event_calendar`
verbatim — never recomputes indirect_usd or which-tickers-have-
earnings. σ comes from the caller-supplied `sigma_provider`
callable (the builder/endpoint split `earnings_shock` /
`stress_scenarios` / `tail_risk` use). The endpoint wires it to
`_earnings_history_for` (yfinance) → `earnings_shock._pop_stdev` (the
σ-byte-equality discipline: a constituent's σ here byte-matches its
σ in `earnings_shock` for held names — SSOT). Provider returning
`None` reads `INSUFFICIENT_SIGMA`; provider raising is caught at the
builder, never propagates.

`DEFAULT_HORIZON_DAYS = 7.0` — wider than `earnings_shock`'s
HELD_IMMINENT (3d) because a basket is materially moved by the
first mega-cap print in a clustered earnings week. Endpoint reads
event_calendar at `horizon_days=14.0` so a 10-day peer event isn't
pre-filtered by the calendar's default. SWR-cached 300s (matches
`earnings-shock` / `earnings-distribution` / `implied-move` cadence
— yfinance earnings_dates + 3y history is the slowest per-name
shape). Prewarm-registered (locked by `test_swr_prewarm_coverage`).
Advisory only — never gates Opus, never injected into the decision
prompt, no caps (AGENTS.md #2/#12).

Locked by `tests/test_peer_earnings_shock.py` (20 tests —
state ladder per branch, indirect_usd arithmetic on the live
TQQQ-NVDA shape ($148 × 3 × 0.09 × 0.07 = $2.80), two-ETF two-
underlying aggregate, inverse-ETF sign honesty, INSUFFICIENT_SIGMA
exclusion from aggregate, horizon edge cases, custom-horizon
extension, past-event drop, verdict-band LOW/MODERATE/SEVERE
thresholds, never-raises on sigma_provider exception / garbage
event_calendar / None snapshot, response shape stability, SSOT
arithmetic byte-equality against build_etf_lookthrough) +
`tests/test_peer_earnings_shock_endpoint.py` (3 tests — Flask
wiring with NO_ETF_HELD branch, empty-history INSUFFICIENT_SIGMA
branch, store-failure-degrades-not-raises).

### How to run / test

```sh
cd /home/zeph/trading-intelligence/paper-trader

# All four new test modules (57 tests, <2s):
python3 -m pytest tests/test_catalyst_class_autopsy.py \
                   tests/test_catalyst_class_autopsy_endpoint.py \
                   tests/test_peer_earnings_shock.py \
                   tests/test_peer_earnings_shock_endpoint.py -v

# Live probe (after the next paper-trader restart picks up the routes):
curl -s 'http://localhost:8090/api/catalyst-class-autopsy' | python3 -m json.tool
curl -s 'http://localhost:8090/api/peer-earnings-shock' | python3 -m json.tool
```

### Invariants reaffirmed by this pass

- **#10** (single source of truth) — `catalyst_class_autopsy` consumes
  `round_trips.build_round_trips` verbatim; never recomputes P&L,
  hold-time, or which-trades-closed. `peer_earnings_shock` composes
  `build_etf_lookthrough` and `build_event_calendar` verbatim; never
  recomputes indirect_usd or earnings-imminent set. σ derivation
  reuses `earnings_shock._pop_stdev` so a constituent's σ here byte-
  matches its σ in `earnings_shock` for held names (test locked).
- **#2 / #12** (no hard limits, advisory-only) — both endpoints are
  observational. Neither modulates a trade decision, neither adds a
  position cap, neither feeds the Opus prompt (no auto-injection;
  the operator reads the UI to learn which catalyst classes earn
  and which baskets are exposed to peer prints).
- **prewarm == @swr_cached** — `peer-earnings-shock` registered in
  the prewarm list; locked by `test_swr_prewarm_coverage`. First
  poll right after a restart returns real data, not `{"warming":
  true}` — exactly the cold-stall blind spot during pre-mega-cap-
  print triage.
- **Defense-in-depth never-raises** — both builders catch any
  unexpected fault and degrade to one honest line (the `_safe`
  contract — `stress_scenarios` / `etf_lookthrough` /
  `earnings_shock` precedent); the endpoints also catch and 500
  with an error JSON, never propagate out to the WSGI layer.

Applies on next paper-trader restart (the documented pattern for
every recent feature).


## 2026-05-20 core hybrid pass — `_trade_impact_line` unit coverage + **TODAY** hourly block (Agent 1, ~13:50 UTC)

### Phase 1 fix — direct unit tests for `reporter._trade_impact_line`

The post-trade book-impact one-liner (`_trade_impact_line`, ~170 lines)
had *zero direct test coverage*; every existing assertion went through
`send_trade_alert`, so a regression that hit the helper's internals but
left the alert-side body green would ship silently. Added a
`TestTradeImpactLineUnit` class (15 tests, all PASS in 0.85s) that pins:

- `snapshot=None` / non-dict → `""` (the defensive guard)
- missing / zero / negative `total_value` → `""` (no
  div-by-zero, no misleading "0.0% of book")
- `HOLD` / `REBALANCE` actions → `""` (the non-BUY/SELL branch
  never invents an impact line)
- sub-0.1% lot weight suppression — emits cash-only token (rounding-noise
  fills never get a name token)
- BUY of a ticker whose lot has been closed post-trade
  (`same_lot_value=0`) → cash-only, no crash
- non-numeric cash field → coerces to 0.0, never raises (same contract
  as the existing market_value coercion)
- fractional option strike (600.5) → keeps decimal in label, *not*
  silently rounded to "600C" (would mislabel a different contract)
- non-numeric option strike "ATM" → drops label gracefully, no crash on
  the `int(sf) if sf == int(sf) else sf` formatter
- multi-strike same-ticker option book → weight attributed to the
  *matching contract leg*, not summed across the ticker stack
- SELL no-store / no-remaining → `"closed · cash $X"` (single cash
  token; locks the historic double-cash regression on the helper
  directly, not just via `send_trade_alert`)
- SELL partial close (still held qty > 0) → `"partial — NVDA still X%
  of book"`, never fabricates a realized P/L when no round-trip closed

### Phase 2 feature — **TODAY** hourly block (`_today_session_line`)

Closed a dashboard→Discord gap one dimension over from the existing 1h
`_session_block`: the hourly had no compact answer to the trader's
*first* morning question after the 09:30 ET bell — "what is **today's**
intraday motion so far?". For a book that gained $30 yesterday then
lost $20 today by 11 AM ET, TOTAL P/L reads +$10 (positive, looks fine)
and the 1h SESSION block only shows the latest hour. Neither answers
"today" until the 16:05 ET daily close (5+ hours later).

- `_today_session_anchor_iso(now)` — pure NYSE-calendar lookup. Returns
  today's 09:30 ET as a UTC-ISO string, or `None` when today is not a
  trading day (weekend / `market.NYSE_HOLIDAYS_2026`) or pre-open
  (`cur_min < 9*60+30`, strict `<` boundary). Half-days keep the
  regular 09:30 ET open (only the *close* shifts to 13:00 ET).
  Post-close still returns today's anchor until UTC date rolls over to
  the next trading day's pre-open.
- `_today_session_line(store, now)` — composes `_window_delta`
  **verbatim** (single source of truth, invariant #10 — same math the
  SESSION block uses with a different baseline, so this surface and the
  1h SESSION block can never disagree on direction) and emits:

  ```
  **TODAY** ◈ since 09:30 ET NYSE open
  > $+15.00 (+1.50%) · alpha `+0.50%`
  ```

  Pure store reads only — NO network (the Discord-path discipline; the
  `_drawdown_line` / `_benchmark_line` precedent). Observational only,
  never gates, adds no caps (invariants #2/#12). The alpha clause
  renders only when SPY is present on BOTH bookends; a partial-SPY
  curve degrades to port-only.

- Wired into `send_hourly_summary` immediately AFTER the 1h SESSION
  block (same "what has the desk done lately" theme, one window over)
  and BEFORE benchmark/drawdown. Suppressed silently when today is not
  a trading day, the session has not yet opened, or the equity_curve
  has no point at-or-after today's anchor.
- Failure contract mirrors the rest of `reporter`: any builder/store
  fault degrades to `""` ("no today-session line this report"), never
  an exception ("no Discord summary this report").

Locked by `tests/test_core_reporter.py::TestTodaySessionAnchorIso` (8)
+ `::TestTodaySessionLine` (9) — 17 tests covering weekend / holiday /
pre-open (incl. 09:29 ET strict-< boundary) / at-open / mid-session /
post-close / half-day; exact +$/+%/alpha on a pinned curve; negative
motion + missing SPY → port-only; store-fault → `""`; hourly wiring
includes block when in-session and OMITS it pre-open.

### Phase 3 live findings (trader perspective)

1. **Live trader on stale code (3 commits behind HEAD)** —
   `/api/build-info` `boot_sha=a673d6b head_sha=1682e2d behind=3`,
   `stale=true`. The new **TODAY** feature applies on the next
   git-watcher restart (~3-min poll cadence); no manual action required.
2. **Host saturated** — 5 concurrent Opus subprocesses (1 live trader +
   4 review agents), `load1=21.93` / `swap_used=70%`. `/api/host-guard`
   correctly flags `SATURATED` with 18% recent starvation rate; the
   pre-flight + mid-call guards dodge the doomed Opus calls so a storm
   only inflates the "skipped claude call —" bucket, not the
   model-empty bucket — the AGENTS.md `pulse()` design working.
3. **New TODAY feature validates value on live data** — current book
   reads `pnl_vs_start=+0.59%` (total P/L since $1000 start) and the
   new TODAY line resolves to `$-6.35 (-0.63%) · alpha -0.62%`. A
   trader checking the hourly without the TODAY block sees +$5.94 and
   would miss that today is actively bleeding -$6.35 against alpha of
   -0.62%. Exactly the gap this block exists to close.
4. **Feed & Discord both HEALTHY** — `/api/feed-health` HEALTHY (9390
   live articles in 24h, newest 0.1h old, no split-brain); Discord
   notify HEALTHY (last send 22m ago, no consecutive failures).

### Counters

`bugs_fixed=1, features_added=1, user_findings=4`.

### Invariants reaffirmed by this pass

- **#2 / #12** (no caps / observational) — `_today_session_line` is
  pure store reads + arithmetic over the existing `_window_delta`
  helper. Never gates Opus, never injected into the decision prompt,
  no risk caps. The `_drawdown_line` / `_benchmark_line` precedent.
- **#10** (SSOT) — `_window_delta` is reused verbatim; the same math
  the 1h SESSION block uses, so the TODAY line and the SESSION delta
  can never disagree on direction.
- **#13** (SCHEMA discipline) — no new tables. Reads `equity_curve`
  (already populated by the live trader) only.
- **NYSE-half-day**: `_today_session_anchor_iso` keeps the 09:30 ET
  open anchor unchanged on a half-day (only `market.close_minute`
  shifts to 13:00 ET — the open is regular). Locked by
  `test_half_day_open_unchanged`.

Applies on next paper-trader restart (the documented pattern for
every recent feature).

## 2026-05-20 ML/backtest hybrid pass (Agent 2) — _WORD_TO_TICKER substring false positives + feature_value_skill diagnostic

### Phase 1 — fix: backtest-side `_WORD_TO_TICKER` matched substrings (1 bug)

`paper_trader/backtest.py::_ml_decide` used naive `keyword in title_lower`
substring matching for every entry in `_WORD_TO_TICKER`. Exactly the
class of bug the LIVE side already fixed via
`strategy._WORD_TO_TICKER_LIVE_PATTERNS` (locked by
`tests/test_ml_live_opinion.TestKeywordSubstringFalsePositives`) — but
the backtest engine still ran the buggy form. Concrete false positives:

- `"ai"` → TQQQ matched `"rain"` / `"training"` / `"Spain"` / `"captain"` /
  `"blockchain"` (most common — every ML-training / crypto headline)
- `"gold"` → GLD matched `"Goldman"` (every Goldman Sachs headline)
- `"intel"` → INTC matched `"intelligence"` and `"artificial intelligence"`
  (double-counted with the broken `"ai"` map)
- `"oil"` → USO matched `"spoiled"` / `"coil"`

Each silently inflated unrelated tickers' `ticker_scores`, distorting
which ticker the backtest's `_ml_decide` picked AND **poisoning the
`decision_outcomes.jsonl` corpus that retrains the DecisionScorer** (the
gate the live trader eventually relies on). Fix mirrors strategy.py:
compile `\bkw\b` once at module import into `_WORD_TO_TICKER_PATTERNS`
and match via `Pattern.search`. Multi-word keys still match because `\b`
sits between word/non-word transitions (including spaces).

### Phase 1 — tests added (22)

`tests/test_ml_backtest_word_boundary.py` (new, 17 tests) —
`TestWordToTickerPatterns` pins every false-positive class at the
pattern level (rain/training/blockchain/captain/Goldman/intelligence/
spoiled don't match) AND that standalone tokens still match (ai/gold/
oil/multi-word `"natural gas"`). `TestMlDecideKeywordIntegration` pins
END-TO-END `_ml_decide`: a `"Heavy rain delays harvests"` headline must
NOT result in `BUY TQQQ`, etc., with a standalone-`"nvidia"`-token
regression guard.

`tests/test_decision_scorer.py` — 2 new tests:
- `test_feature_names_order_matches_build_features` pins the
  `FEATURE_NAMES` ↔ `build_features` output-order contract by name
  with a known-distinct input vector. A silent reorder in
  `build_features` without updating `FEATURE_NAMES` would mislabel
  every row in `feature_contributions` / `feature_importance` and the
  existing length assertion would NOT catch it.
- `test_dedup_survivor_is_highest_return_record` strengthens the
  existing dedup count test: rigs two same-key records with `fwd=+15`
  vs `fwd=-5`, asserts the **survivor's label sign** propagates
  through to the trained model's prediction.

`tests/test_backtest.py::TestMlDecideScorerGateBoundary` (3 new tests)
pins invariant #5 (gate engages at `_n_train >= 500`) with exact-value
conviction assertions at n=499 (gate inactive → conviction 0.25) vs
n=500 (strong-headwind arm fires → conviction 0.15 = 0.25 × 0.6). A
regression that drifts the threshold OR the strong-headwind multiplier
fails one of the three assertions.

Run: `cd /home/zeph/trading-intelligence/paper-trader && python3 -m
pytest tests/test_ml_backtest_word_boundary.py
tests/test_decision_scorer.py tests/test_backtest.py -v` — 178 pass.
Broader `-k "ml or backtest or scorer"`: 467 pass, no regressions.

### Phase 2 — feature: `feature_value_skill` per-feature-value rank skill (1 feature)

Adds `paper_trader/ml/feature_value_skill.py` — a read-only diagnostic
that buckets OOS decision_outcomes by **empirical quintiles** of a
configurable feature (`mom5` / `vol_ratio` / `bb_position` / `ml_score`)
and reports per-bucket `n / rank_ic / dir_acc / mean_realized`. Reuses
`calibration._spearman` (the single source of truth for tie-aware rank
correlation) and `decision_scorer._to_float` so the metric can never
drift from `_oos_rank_metrics` / the per-cycle skill ledger.

Genuinely orthogonal to the 35-module suite. Existing siblings bucket
by news volume, sector, regime, persona, leveraged-or-not, action, or
ticker — none of them slices by a value of a feature the scorer
**actually trains on**. Answers the quant-decisive question
`baseline_compare` cannot:

> Is the scorer's aggregate `+0.225` OOS rank-IC concentrated in one
> feature regime (LOCALIZED_EDGE — a single-feature rule could
> replicate it), or does it generalize across the feature's range
> (UNIFORM_EDGE — multi-feature combination is doing real work)?

Verdicts (threshold-driven, exactly testable):
- `HAS_INVERTED_BUCKET` (≥1 quintile anti-predictive — actionable, first)
- `LOCALIZED_EDGE` (only 1 quintile has EDGE — the model is a gate)
- `UNIFORM_EDGE` (3+ buckets EDGE, IC spread < `BUCKET_SPREAD_TOL=0.05`)
- `MIXED_EDGE` (2+ buckets EDGE with visible spread)
- `NO_EDGE_ANY` (no bucket clears `IC_MIN=0.05`)
- `INSUFFICIENT_DATA` (< 30 aligned outcomes or < 2 sufficient buckets)

Run against the **live OOS corpus** (n=1482 aligned outcomes,
deployed `decision_scorer.pkl` with `n_train=400`):

| Feature      | Verdict     | All 5 quintiles | IC spread |
|--------------|-------------|-----------------|-----------|
| `mom5`       | MIXED_EDGE  | all EDGE        | 0.104     |
| `vol_ratio`  | MIXED_EDGE  | all EDGE        | 0.162     |
| `ml_score`   | MIXED_EDGE  | all EDGE        | 0.088     |

⇒ The deployed scorer's `+0.225` aggregate rank-IC is NOT artifactual
from any single feature regime — it has measurable EDGE in EVERY
quintile of every feature we bucket on. The model genuinely combines
features rather than acting as a single-feature gate.

CLI: `python3 -m paper_trader.ml.feature_value_skill` (default mom5),
`--feature vol_ratio`, `--feature ml_score`, `--feature bb_position`,
`--json` for machine-readable output, `--no-oos-only` for the full
corpus. Exit code 2 on `HAS_INVERTED_BUCKET` so cron / operator
scripts can branch.

Tests: `tests/test_feature_value_skill.py` (32 tests, all
assertion-based). Pins `_safe_float` / `_bucket_for` /
`_quintile_breakpoints` math; `_verdict_for` per-bucket thresholds at
exact boundaries; END-TO-END verdict pins for UNIFORM_EDGE,
LOCALIZED_EDGE, HAS_INVERTED_BUCKET, NO_EDGE_ANY paths via rigged
scorer stubs; FEATURES dict completeness; analyze() entry on empty
file + CLI argparse rejection of unknown feature flag.

### Phase 3 live findings (quant researcher perspective)

1. **Continuous backtest loop is DOWN** (~2 days). Last
   `scorer_skill_log.jsonl` row is `cycle 4` at 2026-05-18T18:06 UTC;
   last completed `backtest_runs` row is `run_id 6242` at
   2026-05-18T18:05 UTC; `continuous.log` last modified 2026-05-18
   12:03 PDT. `ps aux | grep run_continuous_backtests` finds no
   process. The live trader (PID 3926526) IS running and making
   decisions (most recent 2026-05-20T13:57 UTC), but the scorer
   retraining loop that feeds the conviction gate is dead. Restart
   `python3 run_continuous_backtests.py` to resume per-cycle retrains.
2. **Deployed `decision_scorer.pkl` is stale relative to the
   accumulated corpus.** `paper_trader.ml.scorer_freshness` reports
   `n_train=400, age=24.77h`, but `data/decision_outcomes.jsonl` has
   7413 rows (most recent appended 2026-05-20 02:48). A fresh retrain
   on the 5000-record tail would land the scorer at `n_train≈3950`
   (consistent with the last logged cycles, `train_n=3987`/`3959`).
   The live gate is sizing on a model trained from a smaller, older
   corpus than the OOS evaluation suggests. The `MLP_ADDS_SKILL`
   verdict describes what an `n=400` model can rank OOS, NOT what a
   fresh retrain would produce.
3. **2 orphaned `running` backtest rows** (`run_id 6238` from
   2026-05-18T14:17, `run_id 6243` from 2026-05-18T18:45) — would be
   auto-reaped by the continuous loop's `_reap_orphaned_runs` 6h-age
   guard, but the loop is dead. Restarting the loop reaps them on
   the next cycle; no manual SQL needed.
4. **Scorer + diagnostics all healthy modulo the staleness above**:
   - `baseline_compare`: `MLP_ADDS_SKILL  rank_ic=+0.225 dir_acc=0.611
     best_baseline=rsi_meanrev +0.079 ic_gap=+0.145`.
   - `deploy_audit`: `DEPLOYED_MATCHES_SOURCE` (8/8 hyper-params
     match `MLP_CONFIG`).
   - `regime_audit`: `REGIME_UNIFORM_EDGE` (bull/sideways/bear all
     ≥+0.16 rank_ic).
   - `sector_skill`: `HAS_INVERTED_SECTOR` — energy `rank_ic=-0.277`
     on 30 OOS rows (documented finding, plausibly sampling noise).
   - `outcome_data_quality`: CLEAN (0 corrupt, 0 conflicts, 0
     nonfinite features).
   - `news_volume_skill`: `MIXED` (sparse-news bucket `+0.459`,
     no-news `+0.192`).

### How to run / interpret (ML & backtest domain)

**Live OOS skill snapshot:**
```bash
cd /home/zeph/trading-intelligence/paper-trader
python3 -m paper_trader.ml.baseline_compare      # MLP vs single-feature baselines
python3 -m paper_trader.ml.deploy_audit          # deployed pickle vs MLP_CONFIG drift
python3 -m paper_trader.ml.regime_audit          # bull/sideways/bear bucketed IC
python3 -m paper_trader.ml.sector_skill          # per-sector IC
python3 -m paper_trader.ml.feature_value_skill   # per-feature-quintile IC (NEW)
python3 -m paper_trader.ml.scorer_freshness      # pkl age + heartbeat
python3 -m paper_trader.ml.outcome_data_quality  # outcome-file integrity
```

**Manual backtest one-shot:**
```bash
python3 run_backtests.py    # 10 parallel year-long runs
```

**Continuous loop (currently down):**
```bash
python3 run_continuous_backtests.py    # 1 run/cycle, 600s cooldown, scorer retrain
```

**Interpreting backtest results** (`/api/backtests` on `:8090`):
- `total_return_pct` — raw return over the window
- `vs_spy_pct` — alpha (`total_return_pct - spy_return_pct`). Watch
  for `notes` containing `benchmark_unavailable` — empty/degenerate
  SPY series makes `vs_spy_pct` fabricated.
- `equity_curve_json` — per-sample equity points
- `status='running'` for >6h ⇒ orphan reap-candidate (only reaps
  when the continuous loop is alive)

**Tests for the ML/backtest domain:**
```bash
python3 -m pytest tests/ -v -k "ml or backtest or scorer"   # 467 tests
python3 -m pytest tests/test_feature_value_skill.py -v       # 32 NEW
python3 -m pytest tests/test_ml_backtest_word_boundary.py -v # 17 NEW
```

### Counters

`bugs_fixed=1, features_added=1, user_findings=4`.

### Invariants reaffirmed by this pass

- **#1** (live-only clause / training-corpus integrity) — the
  `_WORD_TO_TICKER` substring fix removes a vector that polluted
  `decision_outcomes.jsonl` with mislabeled-ticker decisions,
  protecting #1's "backtest articles must never reach live signals"
  spirit at the corpus level.
- **#5** (DecisionScorer gate at `n_train >= 500`) — locked in code
  by `TestMlDecideScorerGateBoundary`'s exact-value pins at n=499 /
  n=500.
- **#10** (SSOT) — `feature_value_skill` reuses
  `calibration._spearman` / `decision_scorer._to_float` /
  `validation.split_outcomes_temporal`; the same building blocks
  every sibling skill diagnostic uses.
- **scorer-train status truthfulness** — `feature_value_skill`'s
  bad-feature-name path returns `status='error'` with an honest
  hint, mirroring `news_volume_skill.analyze` /
  `sector_skill.analyze`.

Read-only diagnostic + bugfix in a path the live trader doesn't
import; safe to take effect immediately.

---

- **2026-05-20 feat (Agent 4 product-engineer pass) — `/api/today-action-tape`.**
  New deterministic, pure-DB chronological tape of every TRADE and every
  DECISION (HOLD / NO_DECISION / BLOCKED included) since today's UTC
  midnight. Solves the morning-glance gap: every other timeline panel is
  either ranked-by-materiality (`/api/session-delta` caps at 40 events,
  synthesizes EQUITY_MOVE/DRAWDOWN/INACTION rows), an aggregate
  (`/api/daily-recap` totals only), or behavioural-only
  (`/api/decision-forensics` et al — no trade rows interleaved). None of
  them is the literal "what did my bot do today, every move and every
  no-move, in order" feed.

  Pure builder `build_today_action_tape` at
  `paper_trader/analytics/today_action_tape.py`: interleaves every
  in-window `trades` and `decisions` row, sorts chronologically
  (oldest → newest), pins TRADE-before-DECISION on tie (matches
  `recent_trades`' deterministic `(timestamp, id) DESC` tie-break — the
  tape and the trades-table chronology can never disagree). Per-class
  fault drops that class only (the `session_delta` precedent — never
  sink the whole feed). Reuses `dashboard._parse_action_ticker` shape for
  free-text `action_taken` (verb, ticker) and collapses CASH/NONE
  pseudo-tickers — SSOT.

  Aggregate counters: `n_buys`, `n_sells`, `n_holds`, `n_no_decisions`,
  `n_blocked`, `n_other_decisions`, plus `notional_in_usd`,
  `notional_out_usd`, `net_cash_flow_usd = proceeds − spend` (NOT
  realized P&L — that's `/api/round-trips`' job; named accordingly so
  callers don't confuse the two).

  **Route** `/api/today-action-tape` — default since=today's UTC
  midnight; `?since=<ISO-8601>` overrides; `?minutes=<5..10080>` gives a
  rolling "last N minutes" recap. SWR-cached at 15s (decision cadence is
  60s, so 15s avoids stampedes without lagging the desk). Advisory only
  — never gates Opus, adds no caps (invariants #2/#12).

  **Locks (`tests/test_today_action_tape.py`, 15 tests, 0.66s):**
    1. Empty input → well-formed envelope, no exception
    2. Default since = today's UTC midnight
    3. Yesterday rows excluded; custom `since` widens window
    4. Tape chronological oldest → newest
    5. TRADE sorts before DECISION on tie (chronology pin)
    6. Decision-verb mix counts (HOLD / NO_DECISION / BLOCKED / executed)
    7. `BUY NVDA → FILLED`-style free-text → (verb, ticker) parse
    8. Pseudo-ticker (`HOLD CASH`) collapses to ticker=None
    9. Buy/sell tallies and `net_cash_flow_usd = out − in`
   10. Option trade notional uses the row's pre-computed `value` field
   11. Unparseable timestamp drops the row, not the report
   12. Future-dated rows excluded defensively
   13. Route returns JSON envelope via Flask test_client
   14. `?minutes=` overrides default and produces the expected window

  Builder appended to `analytics/`; route appended IMMEDIATELY AFTER
  `/api/session-delta` inside `dashboard.py` (sibling ordering). NEVER
  raises into the Flask handler — analytics import failure or store
  read failure returns a `{"error": ...}` 500 envelope, not a process
  crash.

## 2026-05-20 ML/backtest hybrid pass (Agent 2) — training-integrity hardening + per-action OOS skill

Three concrete training-integrity / diagnostic improvements informed by a
live quant-researcher pass over `data/decision_outcomes.jsonl` (7413 rows)
and the deployed scorer pickle (`data/ml/decision_scorer.pkl`, n_train=400
at start of pass — BELOW the 500 conviction-gate threshold).

### Phase 1 — surgical fixes

1. **`PriceCache._build_trading_days` densest fallback**
   (`paper_trader/backtest.py`). When SPY's series is empty for a window
   (transient yfinance failure), the trading-day calendar fell back to the
   FIRST non-empty ticker by dict-insertion order. That could land on a
   thin/foreign ETF and silently produce a sparse calendar that skipped
   real NYSE days for the entire backtest — every sampled decision day,
   the SL/TP scan, AND the equity curve all key on this calendar. Now
   picks the DENSEST non-empty series (`max len(series)` over
   `self.prices.items()`). SPY-like ETFs (QQQ, NVDA, etc.) all carry
   near-full NYSE coverage so the proxy is safe; thin names lose to them.
   The SPY-available path is unchanged (assertion lock:
   `test_build_trading_days_uses_spy_when_available`).

2. **`train_scorer` validates forward-return labels up front**
   (`paper_trader/ml/decision_scorer.py`). A record with
   `forward_return_5d=null/NaN/inf/bool/non-numeric` used to pass through
   `_to_float(..., 0.0)` and become a 0.0 label — a phantom flat-return
   training row that contaminated the 5000-record retrain tail (a single
   bad row distorts the entire fit because the SELL sign-flip means
   contradictory features → labels). Now drops invalid rows BEFORE
   `build_features`, counts them in `n_label_dropped`, and pickles
   `n_train` as the post-validation count so the gate-active>=500
   invariant (#5) reflects the model's true data exposure. When every row
   is invalid, returns `{"status": "no_valid_labels", "n": 0}` rather
   than fitting an all-zero target vector.

   The dirtiest-input forms now caught:
   `None`, `float("nan")`, `float("inf")`, `True` (bool slips through
   `isinstance(..., int)`), and any non-coercible string.

### Phase 2 — per-action (BUY vs SELL) OOS rank skill

`_oos_rank_metrics` in `run_continuous_backtests.py` now returns
**buy_rank_ic / buy_dir_acc / buy_n** and **sell_rank_ic / sell_dir_acc
/ sell_n** alongside the legacy aggregate. The conviction gate (#5) is
BUY-only, so a quant needs the GATE-RELEVANT skill separately from the
aggregate — an aggregate rank-IC ~0 (the documented current state) can
hide a positive BUY skill cancelled by a SELL anti-skill.

Wired through `_train_decision_scorer`'s status string and
`_parse_scorer_status` so the per-cycle scorer-skill ledger
(`data/scorer_skill_log.jsonl`) now persists the per-action breakdown
durably. Legacy status strings (pre-feature ledger rows) parse cleanly —
every new per-action key defaults to None, mirroring the
multi-horizon / label-clamp legacy-status precedents.

Aggregate metrics (rank_ic / dir_acc / n) byte-identical: same predict
pass, same SELL-flip, same `±PRED_CLAMP_PCT` clamp, single-record n=1
behaviour preserved (dir_acc computable on one non-zero pair, rank_ic
None below n=2).

### Phase 3 — live quant findings

Concrete observations from running the now-deployed scorer pickle against
the last 1000 outcomes (offline, after fixes shipped):

  * **Aggregate rank_ic = +0.061** (weak). **BUY-only rank_ic = +0.107
    (2x the aggregate)**. SELL-only rank_ic = +0.060. The per-action
    diagnostic immediately surfaced exactly the gate-relevant signal a
    quant cares about — without it, a reader would conclude "the model
    has no skill" when the BUY side is actually meaningfully positive.
  * **Continuous loop is DOWN**: no `run_continuous_backtests.py`
    process; `continuous.log` last activity ~2026-05-18, dominated by
    GDELT rate-limit / `ConnectionResetError` noise.
  * **Gate INACTIVE**: deployed `n_train=400 < 500` — the conviction
    gate is currently DISABLED for any live trader inference; the next
    full retrain cycle (5000-outcome tail → ~4000-5000 deduped) will
    push n_train past the threshold and re-arm the gate.
  * **2 orphaned 'running' rows** (run_id 6238, 6243) — would be reaped
    on next continuous-loop start by `_reap_orphaned_runs` (idempotent,
    6h-age-guarded).
  * **Returns distribution looks too good to be true**: last-50 complete
    runs show **mean total_return_pct = +300%, max = +1446%, mean α =
    +240%** on 1-year windows. For one-year backtests this is far
    outside any realistic public benchmark — suggests one of:
    (a) leveraged-ETF + persona-boost overfitting, (b) decision_outcome
    hindsight leakage (the `hindsight_contaminated` filter in
    `_load_local_articles` may not be catching everything), or
    (c) survivorship bias from `TOP_RUNS_TO_TRAIN=1` always keeping the
    best run. Worth a follow-up audit.
  * **1 run with fabricated alpha** (`benchmark_unavailable` note set) —
    the existing benchmark-honesty guard is catching the edge case as
    designed; vs_spy_pct values on that row should be ignored.

### Tests

`paper-trader/tests/`:
  - `test_backtest.py::TestPriceCache::test_build_trading_days_picks_densest_fallback`
    — synthesises 3 series (SPY empty, THIN 30-day, DENSE 250-day) and
    asserts the calendar resolves to DENSE's 250 days.
  - `test_backtest.py::TestPriceCache::test_build_trading_days_uses_spy_when_available`
    — adds a 5x-denser fake ticker, asserts SPY's calendar still wins.
  - `test_decision_scorer.py::TestTrainScorer::test_handles_null_forward_return`
    — all-null input → `status="no_valid_labels"`, `n=0`, `n_label_dropped=35`.
  - `test_decision_scorer.py::TestTrainScorer::test_handles_non_finite_forward_return`
    — 2 inf rows in a 35-record batch → trains on remaining 33, with
    `n_label_dropped=2`.
  - `test_decision_scorer.py::TestTrainScorer::test_drops_mixed_invalid_forward_returns`
    — None / NaN / inf / bool / string ALL drop, leaving 35 valid.
  - `test_continuous.py::TestOosRankMetrics::test_per_action_breakdown_isolates_buy_from_sell`
    — perfectly-concordant BUYs + perfectly-discordant SELLs yield
    `buy_rank_ic=+1.0` and `sell_rank_ic=-1.0` (aggregate would hide
    that signal).
  - `test_continuous.py::TestOosRankMetrics::test_per_action_buckets_handle_empty_one_side`
    — 100% BUYs → `sell_n=0`, `sell_rank_ic=None` honestly (not a
    fabricated value).
  - `test_continuous.py::TestParseScorerStatus::test_parses_per_action_breakdown_tokens`
    — status string with `oos_buy_n=600 oos_buy_ic=+0.11 oos_sell_n=108
    oos_sell_ic=-0.04` → parser extracts ints + floats correctly.
  - `test_continuous.py::TestParseScorerStatus::test_legacy_status_without_per_action_tokens_parses_cleanly`
    — pre-feature status strings parse with new keys defaulting to None.

### Test command (ML/backtest domain)

```bash
cd /home/zeph/trading-intelligence/paper-trader && \
  python3 -m pytest tests/test_decision_scorer.py tests/test_backtest.py \
                    tests/test_continuous.py -q
```

(217 tests, ~5s. The full suite is ~3900 tests + ~25min under load.)

### Manual backtest commands

```bash
# One-shot 10-run launcher (uses BacktestEngine().run_all(10) defaults)
python3 run_backtests.py

# Continuous loop (5 runs/cycle, picks a random 1–10yr window each cycle)
python3 run_continuous_backtests.py

# Inspect a single scorer prediction
python3 -m paper_trader.ml.decision_scorer \
  --explain --ticker NVDA --ml-score 4.0 --rsi 70 --macd 2.0 \
  --mom5 5.0 --mom20 12.0 --regime-mult 1.0
```

## 2026-05-20 core hybrid pass — `_streak_line` Discord surface (Agent 1, ~14:30 UTC)

### Phase 1 — debug

No real bugs surfaced after a careful review of `paper_trader/{runner,reporter,signals,strategy,market,store,dashboard,host_guard}.py`. Full test suite (4014 tests) was green at baseline; targeted code reads for race conditions, missing error handling, off-by-one boundaries, NULL-column reads, lock primitives, and JSON parsing edges all looked correct. The codebase has been heavily hardened over prior review passes — most of the previously suspect paths now carry their own surgical guards (e.g. `_acquire_singleton_lock` fail-open, `_recheck_singleton_lock` upgrade-or-exit, `_realized_pl_window` exit_ts lexical compare). Counters: `bugs_fixed=0`, no Phase 1 commit.

### Phase 2 feature — `_streak_line` (Discord)

`/api/streak` exposes the **closed-round-trip streak structure** (HOT_HAND on a ≥4-win cluster, TILT_RISK on a ≥4-loss cluster) on the dashboard, but the operator who lives in Discord never saw it. Two documented desk pathologies were invisible on the surface they actually read:

* **HOT_HAND** — overconfidence trap after a hot run; a PM that doesn't see it can over-size into the next setup.
* **TILT_RISK** — the loss-cluster bias every PM steps back from before adding more risk.

`reporter._streak_line(store)` composes `build_streak` **verbatim** (invariant #10 — the headline / verdict are the builder's, never re-derived, so this Discord line and `/api/streak` can never tell different stories) and is wired into both `send_hourly_summary` and `send_daily_close`. Pure store read (`recent_trades(2000)` reversed oldest-first — the build_streak contract), **no network** (the Discord-path discipline — the `_capital_pulse_line` / `_hold_discipline_line` precedent). Observational only, never gates, no caps (invariants #2/#12). Same additive failure contract as the rest of `reporter`: any builder/store fault degrades to `""` ("no streak line this report"), **never** an exception ("no Discord summary this report").

**Suppression** — silence-when-nothing-actionable (the `_decision_clock_line` EVEN_DISTRIBUTION / `_hold_discipline_line` NO_DATA precedent — the summary must never become its own lying green light):

* `HOT_HAND`   — ALWAYS surfaced.
* `TILT_RISK`  — ALWAYS surfaced.
* `NEUTRAL` / `EMERGING` (n_round_trips < `STABLE_MIN_ROUND_TRIPS=8`) / `NO_DATA` → silent.

Block placement: last among the behavioural lines in both summaries, after `_no_decision_reasons_line` — the existing structural+process+behavioural→run ordering (concentration → drift → host pulse → streak).

### Phase 2 — tests (14)

`paper-trader/tests/test_streak_reporter.py`:
  - `TestStreakLineSuppression::test_empty_book_returns_empty` — no trades → "".
  - `TestStreakLineSuppression::test_emerging_below_stable_returns_empty` — 3 round-trips < 8 → silent (the EMERGING precedent).
  - `TestStreakLineSuppression::test_neutral_streak_returns_empty` — 8 alternating W/L → NEUTRAL → silent.
  - `TestStreakLineSurfaces::test_hot_hand_surfaces_verbatim` — 4-win run at series end → HOT_HAND + builder's "4-win run" headline.
  - `TestStreakLineSurfaces::test_tilt_risk_surfaces_verbatim` — 4-loss run → TILT_RISK.
  - `TestStreakLineSurfaces::test_longer_loss_run_still_surfaces` — 6-loss run reads "6-loss run".
  - `TestStreakLineFailureContract::test_store_raise_returns_empty` — store.recent_trades raises → "".
  - `TestStreakLineFailureContract::test_builder_raise_returns_empty` — build_streak raises → "".
  - `TestStreakLineFailureContract::test_non_dict_result_returns_empty` — builder returns None → "".
  - `TestStreakLineFailureContract::test_missing_headline_returns_empty` — `headline=""` → "".
  - `TestStreakLineWiredIntoSummaries::test_hourly_summary_surfaces_tilt_risk` — end-to-end through `send_hourly_summary` body.
  - `TestStreakLineWiredIntoSummaries::test_hourly_summary_silent_when_neutral` — confirms no STREAK token in a balanced book's hourly.
  - `TestStreakLineWiredIntoSummaries::test_daily_close_surfaces_hot_hand` — end-to-end through `send_daily_close`.
  - `TestStreakLineWiredIntoSummaries::test_summary_still_sends_when_streak_builder_faults` — additive contract end-to-end.

### Phase 3 — live findings (trader perspective)

Run at 14:30 UTC against the live `:8090` dashboard + `data/paper_trader.db`:

  * **Host saturation is the dominant blocker right now.** 8/8 recent decisions in `data/paper_trader.db` were NO_DECISION: 6 with `skipped claude call — host saturated: 5–6 concurrent Opus (>4)` (the pre-flight / mid-call guard correctly declining), 2 with `claude returned no response (nonzero_rc)`. Concurrent Opus reads 3 at the moment of the probe but `recent_starvation_rate` reads 22.5 % — intermittent storm class. No actual TRADE has happened in ~9 h, even though the system is reporting "alive" through dashboard endpoints.
  * **Book is currently profitable.** 2 open positions (NVDA stock qty 3 @ avg $221.78, now $224.02, +$6.74; TQQQ qty 4 @ avg $73.48, now $74.93, +$5.78). `/api/portfolio` reads `total_value $1012.08, +1.21 % vs $1000 start`, `stale_marks 0` — marks are fresh and the headline P&L is honest.
  * **The runner is on stale code.** `/api/healthz` reads `boot_sha b5a765f, head_sha 8d25bb1, behind 2, stale true` (this Agent 1's `_streak_line` commit and a sibling Agent's edits are pushed but not yet running). The git-watcher should fire the deferred restart within its 3 min poll cycle.
  * **Singleton-lock is held.** `lock_status acquired, lock_holder_pid 3965827, lock_degraded false` — only one trader is writing the paper book (the documented 2026-05-17/18 double-trade pathology is not in force).
  * **`_streak_line` itself will be silent until the book has ≥8 closed round-trips.** Today's `/api/streak` reads `n_round_trips=1, state=EMERGING, verdict=null` — by design, the verdict is withheld below the stable threshold. The Discord surface will start firing HOT_HAND / TILT_RISK once enough round-trips accumulate; right now the suppression contract is the correct outcome.

### Counters

`bugs_fixed=0` (no real bugs found after a careful review) · `features_added=1` (`_streak_line` Discord surface) · `user_findings=5` (host-saturation freeze, no-trades-for-9h, profitable open book, runner on stale code awaiting restart, streak surface honestly silent below threshold).

### Invariants reaffirmed by this pass

* **#10 single source of truth** — `_streak_line` consumes `build_streak`'s own `headline` / `verdict` verbatim; the Discord line and `/api/streak` cannot disagree.
* **#2 / #12 advisory only** — observational, never gates Opus, adds no caps.
* **Discord-path discipline** — pure store reads, no network in the hot path (the `_capital_pulse_line` / `_hold_discipline_line` precedent).
* **Additive failure contract** — any builder/store fault degrades to `""`, never an exception that takes down the whole hourly / daily summary.
* **Silence-when-nothing-actionable** — NEUTRAL / EMERGING / NO_DATA are silent; the summary never becomes its own lying green light.

### 2026-05-20 review pass — ML+backtest hybrid (Agent 2): hindsight-urgency leak fix + gate_utilization distributional audit

- **Phase 1 — bug fixed (commit `a5d9678`): hindsight-contaminated `urgency` leak in `_load_local_articles`.** `paper_trader/backtest.py` already downgrades a stale-labeled article's `ai_score` to `kw_score` when `first_seen − published > LABEL_STALENESS_DAYS=60` (the documented case where a back-collected article was relabeled by ArticleNet *after* the events it describes already played out). But the same row's `urgency` column — also written by ArticleNet at first_seen time — was kept raw. `urgency` flows into the live decision pipeline via `_ml_decide`'s reasoning string (`news_urg={buy_news_urg:.1f}`), which `_compute_decision_outcomes` then re-extracts and stores as the `news_urgency` outcome field, which `train_scorer` consumes as a feature. End state: the DecisionScorer's `news_urgency` input could be set from a value computed with knowledge of future returns. Mirror the score filter — when `hindsight_contaminated=True`, set `urg_v = 0.0`. Locked by `tests/test_validation.py::TestHindsightLabelFilter::test_hindsight_contaminated_urgency_is_zeroed` and `::test_fresh_article_urgency_is_preserved` (the contaminated row vs the fresh row must behave oppositely; the latter test guards against a future "always zero out urgency" over-correction).

- **Feature shipped (commit `8de7d53`): `paper_trader/ml/gate_utilization.py` — distributional gate-arm reachability audit.** Complementary to `gate_audit.py` (economic spread). `gate_audit` answers "when each arm DOES fire, is the multiplier justified?" — but says nothing about how often each arm fires. With the gate's ±10/±5 thresholds set as constants relative to the `±PRED_CLAMP_PCT`-clamped scorer output, the OOS prediction distribution can render one or more arms structurally unreachable: (a) extremes empty so the ×0.60 / ×1.30 arms never fire, (b) neutral dominated so the gate is a no-op for most decisions, (c) wholly one-sided so the asymmetric multipliers never bite. None of those failures is visible in `gate_audit`'s verdict.

  `gate_utilization_report` reuses `gate_audit.gate_arm` (single source of truth — invariant #10 spirit) so arm-membership and ordering can never drift between the two audits. Reports per-arm `n`/`pct_of_total`/`multiplier`, the prediction-distribution `min/p10/p25/median/p75/p90/max`, `side_balance` (fraction of predictions strictly tailwind-side), and a verdict ladder ordered with the alarming verdict first: `EMPTY_EXTREME_ARM` > `NEUTRAL_DOMINATED` > `LOPSIDED` > `BALANCED` / `WEAK_BALANCE`. The empty-extreme verdict surfaces first because the operational consequence ("the gate cannot exercise its full multiplier range") is the strongest single signal, and it's not redundant with `gate_audit`'s `INSUFFICIENT_DATA` (which fires under `MIN_ARM_N=5` in either extreme — this fires only at exactly zero, a strictly stronger reading consistent with that gate_audit cell already being `INSUFFICIENT_DATA`).

  Same operational discipline as every other `paper_trader/ml/` diagnostic: read-only, no train / pickle / `build_features` / `N_FEATURES` / trade-path touch, never raises on bad input. Live read on the current OOS slice (n=1482, scorer with n_train=400): `verdict=BALANCED`, every arm 11.88-34.41%, distribution `min=-50 / median=-0.05 / max=+50`, `side_balance=0.497` — symmetric and well-utilized across the full multiplier range. CLI: `python3 -m paper_trader.ml.gate_utilization [--json] [--all]`. Locked by `tests/test_gate_utilization.py` (35 tests: SSOT pin to `gate_audit.gate_arm` + `_ARM_ORDER` + `_ARM_MULT`; `INSUFFICIENT_DATA` short-circuit at exact `MIN_TOTAL` boundary; `EMPTY_EXTREME_ARM` trump precedence over `NEUTRAL_DOMINATED`; `NEUTRAL_DOMINATED` vs `LOPSIDED` discriminator at the `LOPSIDED_PCT=0.60` boundary inclusive; `BALANCED` vs `WEAK_BALANCE` at the `MIN_ARM_PCT=0.05` boundary; `side_balance` arithmetic including `zero-counts-as-headwind` boundary; NaN routes to neutral via `gate_arm`'s non-finite contract; degrade-never-raises on non-iterable / generator / garbage input; `analyze()` honest-degrade on missing file and untrained scorer with redirected `SCORER_PATH`).

- **Live finding: continuous-loop ledger is stale (May 18 18:06 last cycle, ~36h gap).** `data/scorer_skill_log.jsonl` on USB has 4 cycle rows, all from 2026-05-18; the last is cycle 4 (`window_start=2016-04-29, window_end=2019-04-29, gate_active=true, train_n=3959, val_rmse=10.92, oos_rmse=14.09, oos_dir_acc=0.55, oos_ic=0.11`). The local repo's `data/ml/decision_scorer.pkl` is dated May 18-19 with `n_train=400` — BELOW the gate's `n_train >= 500` threshold (invariant #5), so the conviction gate is currently INACTIVE on the locally-deployed pickle. The repo has 7,413 outcome rows in `data/decision_outcomes.jsonl` that dedup to 6,906 distinct (ticker, sim_date, action) keys, so the moment `run_continuous_backtests.py` runs a single cycle, `_train_decision_scorer` will jump `n_train` from 400 to ~6906 and the gate will engage immediately. Operational, not a code bug.

- **Live finding: 2 orphaned `running` backtest rows in `backtest.db`.** `backtest_runs` carries `(status, count)` = `[('complete', 475), ('failed', 24), ('running', 2)]`; the running rows started 2026-05-18 and were never marked done because the parent process exited without writing `complete` or `failed`. `_reap_orphaned_runs(max_age_hours=6.0)` is built precisely for this case but only fires on the next `main()` start, so the dashboard would render them as in-flight indefinitely (CLAUDE.md §11 documented symptom). Operational; the reaper will clear them on next continuous-loop start.

- **Live finding: backtest engine is healthy.** 475 complete / 24 failed (~5% failure rate) since loop history; recent runs show very high `vs_spy_pct` (avg 67-176% alpha) but this is *backtest* return, dominated by leveraged ETFs sized aggressively over multi-year random-window samples — does NOT translate to live performance. Backtest health is the right read: the engine is producing valid outcome rows, which is what matters for scorer retraining.

- **Live finding: `gate_audit` on the actual OOS slice reads `GATE_EFFECTIVE`.** With `n=1482, arm_monotone=1.0` (all five arms have realized returns in correctly ascending order), spread `+6.07pp` between strong_tailwind (+3.69%) and strong_headwind (-2.38%). This is materially different from the pass-#7 finding of `GATE_INEFFECTIVE` (+0.98pp spread) at a different scorer state — the scorer's economic effectiveness IS realized once retrained. So the load-bearing finding from this pass: gate is BOTH structurally reachable (`gate_utilization=BALANCED`) AND economically effective (`gate_audit=GATE_EFFECTIVE`). The hindsight-urgency leak fix is a defensive improvement against a leak that was real but small in absolute impact on the existing trained model.

### Counters

`bugs_fixed=1` (hindsight-contaminated urgency leak) · `features_added=1` (`gate_utilization` distributional audit) · `user_findings=5` (continuous loop ledger stale ~36h, scorer pkl pre-gate at n_train=400 awaiting retrain, 2 orphaned `running` rows, backtest engine healthy at 95% completion, `gate_audit` reads `GATE_EFFECTIVE` on the live OOS slice).

### Test commands for the ML/backtest domain

```bash
# Focused ML/backtest tests (~10s):
cd /home/zeph/paper-trader && python3 -m pytest tests/test_decision_scorer.py tests/test_decision_scorer_attribution.py tests/test_backtest.py tests/test_continuous.py tests/test_calibration.py tests/test_validation.py tests/test_integration_backtest.py tests/test_ml_backtest_coverage.py tests/test_ml_backtest_review.py tests/test_ml_backtest_seams.py tests/test_ml_backtest_word_boundary.py tests/test_scorer_freshness.py tests/test_scorer_honesty.py tests/test_scorer_smoke_test.py tests/test_gate_audit.py tests/test_gate_utilization.py -q
# Run the new utilization audit on the live OOS slice:
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.gate_utilization
# Run gate_audit on the live OOS slice (economic complement):
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.gate_audit
```

### Invariants reaffirmed by this pass

* **#10 single source of truth** — `gate_utilization` imports `gate_arm`, `_ARM_ORDER`, `_ARM_MULT` from `gate_audit`; arm-membership semantics can never drift between the two audits.
* **#5 conviction gate** — gate fires only at `n_train >= 500`; new diagnostic is observational only, never gates or alters Opus.
* **#2 / #12 advisory only** — read-only, no pickle/train/feature/trade-path touch; same discipline as every other `paper_trader/ml/*` diagnostic.
* **Defence-in-depth on training labels** — hindsight-contaminated `ai_score` and `urgency` now both get filtered; the next contaminating column added to articles.db needs a matching filter at the same call-site.

### 2026-05-20 review pass — paper-trader-core HYBRID (Agent 1): wire missing `_decision_weekday_line` + per-position α-vs-SPY token

- **Phase 1 — bug fixed (commit `00f222c`): `_decision_weekday_line` was defined but never wired.** `paper_trader/reporter.py::_decision_weekday_line` had a complete builder/endpoint/Discord-helper triplet (`build_decision_weekday`, `/api/decision-weekday`, the helper) plus 5 sibling helpers (`_decision_clock_line`, `_repeat_loser_line`, etc.) all properly wired into both `send_hourly_summary` and `send_daily_close` — but `_decision_weekday_line` was missing from both send paths. So a recurring weekday NO_DECISION concentration (e.g. Friday-after-close quota slump) was *detectable* on the dashboard but silently *never reached* the operator's Discord surface. Wired into both send paths immediately after `_decision_clock_line` (its hour-of-day sibling). Also bundled the previously-uncommitted `_repeat_loser_line` analytics module so its `from .analytics.repeat_loser import build_repeat_loser` resolves at next process restart. 12 new tests (`tests/test_decision_weekday_reporter.py`) cover suppression / surface / failure contract / wiring; 14 new tests (`tests/test_repeat_loser*.py`) cover the per-ticker run logic + reporter wiring.

- **Feature shipped (commit `cf39657`): per-position α-vs-SPY token in Discord position lines.** Compact `  α +X.Y%` annotation added after the existing weight/age/earnings tokens on every Discord-summary position line in `_portfolio_lines`. The number is the position's % return MINUS the SPY's % return over the same hold window, computed entirely from the already-stored `equity_curve` (which carries `sp500_price` at each tick) + the live `sp500` the reporter already holds for the header — **zero network on the hot Discord path** (the `_pos_pct_weight` / `_pos_hold_age_token` precedent). Answers the one question per-position weight / age / earnings tokens can't: "is this trade contributing alpha, or just riding the index?" A position +1.2% in a +1.5%-since-entry tape is actually a 0.3pt drag on alpha; per-trade P/L % alone hides that. Suppression mirrors the existing per-position tokens (stale_mark / missing opened_at / no baseline equity point / non-positive avg or current → empty token, never raises). 20 new tests (`tests/test_pos_alpha_token.py`) cover the full suppression contract + known-input alpha arithmetic + the wiring into `_portfolio_lines` for both stocks and options.

- **Live finding: 99.36% NVDA single-name concentration.** `/api/risk` flags HIGH severity (`concentration_warning=true, concentration_severity=HIGH`), but Opus kept buying NVDA three more times in the last 18h (0.5 + 1.0 + 1.0 = 2.5 shares added on top of the existing 2.0 shares, now 4.5 shares at avg $222.33). Cash is at $6.49 (0.64% of book) — the desk is capital-paralysed and cannot act on any non-NVDA signal. Behavioural, not a code bug; the existing `_concentration_line` + `_capital_pulse_line` surfaces both already flag it.

- **Live finding: runner is UNSUPERVISED (orphan, ppid=1).** `/api/supervision` reads `verdict=UNSUPERVISED, orphan=true, actionable=true`; user `systemctl --user` is dead (`Failed to connect to bus: No medium found`) so the dbus probe can't see whether the unit is active+enabled. Recommendation per the builder: `systemctl --user enable --now paper-trader`. Trader is on current code (`boot_sha=cf39657, behind=0, stale=false`) — a clean exit (git-watcher restart, deadman) or crash would leave it DOWN until manually re-started. Matches the documented [PT systemd-vs-manual restart spam](../../../.claude/projects/-home-zeph-trading-intelligence/memory/pt-systemd-vs-manual-restart-spam.md) pathology.

- **Live finding: 3-minute dashboard outage during runner restart under load.** Probed `/api/runner-heartbeat` at 00:38:21 UTC, saw `uptime=79s`. By 00:38:25 the port was LISTENing with no users field (kernel socket alive, server thread gone). Recovered at 00:41:35 with new pid 149898 holding the lock. So the git-watcher's deferred-restart-with-deadman correctly fired (my new commit was deployed end-to-end), but the trader was blind/silent for the ~3-min transition window. Acceptable trade-off vs running stale code, but operators should know the window exists when they see `/api/healthz` time out post-push.

- **Live finding: no runner.log on disk anywhere.** The runner is launched directly by `runner.py` (or by systemd — currently dead-bus), and stdout/stderr have no documented redirect path. Operators tailing decisions or reaping error tracebacks must use `journalctl --user -u paper-trader` (which is dead in this session) or read `paper_trader.db` decision rows directly via sqlite. A documented log file would close the operator-facing "is the bot alive and what is it doing right now?" gap; the existing dashboard endpoints answer the first half but not the unstructured stderr / Python traceback view.

- **Live finding: `data/paper_trader.db?mode=ro` empty orphan file.** A literal filename of `paper_trader.db?mode=ro` (zero bytes) exists in `data/` — looks like someone invoked `sqlite3 "paper_trader.db?mode=ro"` without `uri=True`, and sqlite treated the whole string as a filename. Safe to delete; not mine to remove this pass.

- **Live finding: my new α-vs-SPY token works on the real live DB.** Running `_pos_alpha_token` against the current NVDA position (opened 2026-05-19, avg $222.33, current $223.43) with the live `equity_curve` (sp500 baseline $7432.77, current $7432.77 — markets closed weekend, SPY didn't move) yields `  α +0.1%` — the position is exactly +0.5 % over its 2-day hold while SPY is +0.0 %, so the trade has bought 0.5pt of alpha (rounded to +0.1%; tiny because the hold window is only 2 days of weekend-flat SPY). Sanity-checked end-to-end; will appear in the next hourly summary the live runner emits.

### Counters

`bugs_fixed=1` (`_decision_weekday_line` never-wired silent surface) · `features_added=1` (per-position α-vs-SPY Discord token) · `user_findings=6` (NVDA 99.36% concentration with Opus still buying, UNSUPERVISED orphan runner, 3-min dashboard outage during git-watcher restart, no runner.log on disk, empty `paper_trader.db?mode=ro` orphan file, new α token live-validated against the production NVDA position).

### Test commands for this pass

```bash
# Just the new tests added this pass (~3s under load):
cd /home/zeph/trading-intelligence/paper-trader && python3 -m pytest tests/test_decision_weekday_reporter.py tests/test_repeat_loser.py tests/test_repeat_loser_reporter.py tests/test_pos_alpha_token.py -q
# Re-verify no regressions in the core reporter suite (~20s):
cd /home/zeph/trading-intelligence/paper-trader && python3 -m pytest tests/test_core_reporter.py -q
```

### 2026-05-20 review pass — paper-trader-core HYBRID (Agent 1): lot cost-basis in trade-impact alert

- **Phase 1 — no bug-only commit.** A careful pass over `runner.py` / `reporter.py` / `signals.py` / `strategy.py` / `market.py` / `store.py` against the existing 731-test core suite (all passing) surfaced one piece of dead code in `reporter._trade_impact_line` — `same_ticker_value` was accumulated but never read — and no other concrete bugs. Per advisor guidance to avoid padding the bug counter, the dead-variable cleanup was folded into the Phase 2 commit instead of a standalone `fix:` commit. `bugs_fixed=0`.

- **Feature shipped (commit `2cfb6ac`): `@ avg $X.XX` cost-basis token in the trade-impact alert.** After a BUY the trader's first sizing question is "what's my NEW blended avg cost?" — vital for sizing the next add. After a partial SELL it's "what does the leftover cost me?" — the basis on which to judge whether to trim further or hold. The post-trade snapshot (`strategy._mark_to_market`) already carries `avg_cost` per position (just written by `upsert_position`'s blend), so `_trade_impact_line` now sums qty-weighted cost over the matched lot's rows and emits `@ avg $X.XX` inline with the existing leg-%-of-book token. No re-derivation, no extra read, no network — pure composition over the snapshot the caller already has. Suppression mirrors the existing `stale_mark` precedent: missing / non-numeric / zero `avg_cost` silently drops the token rather than emitting a misleading `"$0.00"`, so every existing test (whose hand-built snapshots omit `avg_cost`) stays byte-compatible. Locked by 5 new tests in `TestTradeImpactLineUnit` asserting *specific* numerics (e.g. `"NVDA now 100.0% of book @ avg $110.00 · cash $500.00"`) — not just "field present".

- **Live finding: host SATURATED during this very review.** `/api/host-guard` reads `state=SATURATED, opus_count=5, headline="Opus is starved by the box … host saturated: 5 concurrent Opus (>4); 24 % of the last 102 decisions never reached Opus. The desk is frozen by host load … ops — reduce concurrent Opus (review/backtest agents); the bot cannot resolve this by trading."` The 5 Opus processes are: the live trader (pid 167088) + this hourly review's 3 sibling HYBRID agents + 1 longer-running backtest committee process. **The hourly review is self-inducing the #1 documented pathology by design** — running 3 parallel Opus reviewers against a 16-cpu / 16-GB box that is already running the live trader and a backtest committee. Not actionable from inside the code, but a real architectural irony worth recording: the review mechanism is the saturation it would alarm on.

- **Live finding: heartbeat verdict and host-guard verdict disagree by design.** `/api/runner-heartbeat` reads `verdict=HEALTHY, headline="last decision 14m ago, within the 60m market-closed cadence", decision_efficacy=PRODUCING (pct=20)` while `/api/host-guard` reads `state=SATURATED, starvation_rate_pct=23.5`. Both are correct — heartbeat measures *cadence liveness* (cycles fire on schedule) and host-guard measures *throughput health* (cycles actually reach Opus). An operator reading one without the other gets a misleading mental model. The Discord `_host_pulse_line` does surface SATURATED when it's true, so the surfaces aren't dark — just orthogonal.

- **Live finding: 20% NO_DECISION rate in the last 100 cycles.** `/api/no-decision-reasons` shows the top buckets are `cli_nonzero_rc=12` and `host_saturated=9` over a window of 21 NO_DECISION cycles. Both are host-load symptoms (the nonzero_rc bucket is also likely an OOM-killed Opus subprocess). The `host_saturated` sub-bucket exists because of an earlier review-pass fix that distinguished a pre-flight/mid-call skip from a genuine model timeout — the bucket is correctly populated.

- **Live finding: portfolio is flat post-NVDA-earnings-trade.** The last 3 fills were: BUY 1.0 NVDA @ $223.43 (post-print continuation) → BUY 0.5 NVDA @ $223.43 (deploying remaining cash on the catalyst) → SELL 4.5 NVDA @ $223.43 (full close on weakening AH thesis + 99% concentration). Net realized was a small + (the trade ledger shows `total_value=$1011.95, +1.20 % vs $1000 start`). Discord trade alert path would have used the new `@ avg $X.XX` token on each leg if it had been deployed at the time. On the next live BUY after the restart, the token will appear in the operator's Discord channel for the first time.

- **Live finding: build SHA stale post-push as expected.** `/api/build-info` reads `boot_sha=4ab52d1, head_sha=2cfb6ac, behind=2, stale=true` immediately after my push. The `behind=2` confirms a sibling agent (Agent 2 ML or Agent 4 feature-dev) also pushed during this hour. The git-watcher will queue a deferred restart within its 3-min poll cycle so the new code deploys end-to-end without killing a mid-Opus call.

- **Live finding: historical singleton-guard spam in `logs/runner.log`.** Many `[runner] another paper trader is already running (pid=…); refusing to start a second trader on the same paper book — exiting` lines from an old systemd-Restart=always loop racing the manual singleton guard. Per the user's persisted memory `pt-systemd-vs-manual-restart-spam`, this is documented historical noise from a since-resolved configuration; the trader currently runs as a manual process with the systemd unit disabled. Skip.

### Counters

`bugs_fixed=0` (clean review — no real bugs surfaced beyond a single dead-variable cleanup folded into the feature commit) · `features_added=1` (cost-basis `@ avg $X.XX` token in `_trade_impact_line`) · `user_findings=7` (host SATURATED self-induced by this very review, heartbeat/host-guard verdict disagreement, 20 % NO_DECISION rate dominated by `cli_nonzero_rc` + `host_saturated`, portfolio flat post-NVDA-earnings round-trip, build SHA stale post-push, runner.log singleton-guard spam is historical).

### Test commands for this pass

```bash
# Just the new tests added this pass (~3s):
python3 -m pytest tests/test_core_reporter.py -k "avg_cost or partial_sell_emits_avg" -q
# Re-verify the full core reporter suite (~4s):
cd /home/zeph/trading-intelligence/paper-trader && python3 -m pytest tests/test_core_reporter.py -q
# Full suite (~3-5 min under load):
cd /home/zeph/trading-intelligence/paper-trader && python3 -m pytest tests/ -q
```

### Invariants reaffirmed by this pass

* **#10 single source of truth** — `_trade_impact_line` consumes the snapshot's own `avg_cost` (just written by `upsert_position`) verbatim; no re-derivation of the blend.
* **#2 / #12 advisory only** — observational, never gates Opus, adds no caps — additive to the existing alert body.
* **Discord-path discipline** — pure composition over the snapshot the caller already has; no network on the hot alert path (the `_pos_pct_weight` / `_pos_hold_age_token` precedent).
* **Additive failure contract** — any garbage avg_cost / qty / market_value silently suppresses the @avg token rather than crashing; existing snapshot shapes that omit `avg_cost` stay byte-compatible.
* **Silence-when-nothing-actionable** — zero / missing / non-positive avg_cost yields no token, never a misleading `"$0.00"`.

## 2026-05-20 ML/backtest hybrid pass (Agent 2) — benchmark_integrity diagnostic + skeptical-quant live audit

### Phase 1 — honest no-commit

Three advisor-suggested suspects checked carefully and ruled out:

* **`BacktestStore` concurrency** (background `_opus_annotate` thread + main thread + `_compute_decision_outcomes` + `_append_top_decisions` all share `engine.store.conn`): every public method holds `self._lock` for the full cursor lifetime. No bug.
* **`_train_decision_scorer` oversampling `_keep` guard**: `rep > 0` filter combined with the `_keep.any()` fallback correctly preserves training-fold integrity even when every row is a CONDEMN that rounds to rep=0. Behaviour matches the documented 0.1× weight semantic. No bug.
* **`_inject_and_train` for/else retry loop**: `for` exits via `break` on success (bypassing `else`) and via exhaustion on persistent lock (triggering `else`'s "locked after N attempts" return). Non-lock `OperationalError` returns early from inside the except. No bug.

Per the saturation-pass principle (~30 prior reviews, 466 ML/backtest tests green at baseline), `bugs_fixed=0`, no Phase 1 commit. Advisor approved.

### Phase 2 feature — `paper_trader/ml/benchmark_integrity.py` (commit `720d534`)

**Why.** `backtest.py::run_one` carries a benchmark-honesty guard (line ~2697) that writes `benchmark_unavailable` into a run's `notes` column when SPY returned `0.0` over a ≥30-day window — both branches (SPY series empty AND spy_return=0 on a non-trivial window) are degenerate-`returns_pct` outputs that make `vs_spy_pct = total_return - 0` a fabricated number, not real alpha. The guard catches every NEW run. But it landed AFTER ~80 historical runs had been finalized with `spy_return_pct=0.0` and empty `notes` — those runs still appear in `vs_spy_pct`-keyed analytics, the dashboard, AND in `strategy.py::_ml_is_qualified`'s 20-run median-α qualifier gate (the SQL filters `vs_spy_pct IS NOT NULL` but does NOT filter the degenerate case — those rows DO have non-NULL vs_spy_pct, just a fabricated one). A skeptical quant cannot judge the magnitude of this contamination by reading the dashboard.

**Behaviour.** Single SQL pass over `backtest_runs`. Degenerate criterion mirrors the in-engine guard exactly (`status='complete' AND spy_return_pct=0.0 AND (end_date − start_date) ≥ 30 days`). Flagged subset: degenerate AND `notes LIKE '%benchmark_unavailable%'`. Reports total / flagged / unflagged counts, per-500-run-id-bucket distribution (flat ⇒ leak frozen, growing ⇒ leak active), and the **economic impact** on the live qualifier window: median `vs_spy_pct` of the last `QUALIFIER_WINDOW=20` qualifying runs both as-is and with unflagged-degenerate rows excluded — the number a quant cares about ("does removing these flip the live ML-advisor gate state").

**Verdict ladder (most alarming first):**

| Verdict | Trigger |
|---|---|
| `INSUFFICIENT_DATA` | < `MIN_TOTAL=30` complete runs |
| `ACTIVE_LEAK` | ≥1 unflagged-degenerate run in the last `LEAK_DETECTION_WINDOW=50` runs — the engine's guard regressed or a code path bypasses it |
| `HISTORICAL_CONTAMINATION` | unflagged-degenerate runs exist but are all older than the leak window — pre-guard pollution that won't grow |
| `FLAGGED_ONLY` | every degenerate run carries the `benchmark_unavailable` flag; engine guard is working as intended |
| `CLEAN` | no degenerate runs at all |

**Same operational discipline as every other `paper_trader/ml/*` diagnostic:** read-only, never trains, never touches `decision_scorer.pkl` / `decision_outcomes.jsonl` / `build_features` / any trade path. Pure DB read. Module-level `DEFAULT_DB_PATH` resolved at call time (the test-isolation rule). Never raises — a malformed row drops itself only; missing DB degrades to `status='error', verdict='INSUFFICIENT_DATA'`. CLI pattern mirrors `gate_utilization` / `gate_audit` / `baseline_compare`.

```bash
python3 -m paper_trader.ml.benchmark_integrity
cd /home/zeph/trading-intelligence/paper-trader && python3 -m paper_trader.ml.benchmark_integrity --json
```

Exit codes for shell-script gating: `0` = CLEAN/FLAGGED_ONLY; `1` = HISTORICAL_CONTAMINATION; `2` = ACTIVE_LEAK; `3` = INSUFFICIENT_DATA/error.

**Live read on `backtest.db` (477 complete runs):**
```
verdict=HISTORICAL_CONTAMINATION
flagged degenerate   : 1
unflagged degenerate : 80 (16.8% of total)
max unflagged run_id : 6145
max flagged   run_id : 6225
unflagged by run_id bucket:
   5500- 5999: 55
   6000- 6499: 25
qualifier window (last 20):
   n_unflagged       : 0
   median α (asis)   : +143.33%
   median α (clean)  : +143.33%
   median α (delta)  : +0.00pp
```
The leak is **frozen** (max unflagged 6145 < max flagged 6225 — the in-engine elif guard landed at ~6146 and has caught every degenerate case since). The current live trader qualifier window has **zero** contaminated rows, so the ML advisor's gate state is unaffected by the historical pollution — but a future window that recedes back into the pre-guard run-id range could include them. (Cross-domain — see live findings.)

**Tests (`tests/test_benchmark_integrity.py`, 25 tests, ~1s offline):**
* `TestEmptyOrClean` (4 tests) — empty / non-complete-only / clean-corpus / below-MIN_TOTAL.
* `TestDegenerateDetection` (4 tests) — SPY=0 over 30d IS degenerate, SPY=0 under 30d is NOT, flagged vs unflagged separation, non-zero SPY never degenerate regardless of window.
* `TestVerdictLadder` (4 tests) — ACTIVE_LEAK / HISTORICAL_CONTAMINATION / FLAGGED_ONLY discrimination at the `LEAK_DETECTION_WINDOW=50` boundary; ACTIVE_LEAK dominates HISTORICAL (most-actionable-first precedent).
* `TestBucketing` (2 tests) — 500-id bucket arithmetic, ascending bucket order.
* `TestQualifierWindow` (4 tests) — run_id-DESC order, exact median-α arithmetic (clean=10 vs asis=105 ⇒ delta=95), `n_trades<5` filter mirrored from `_ml_is_qualified`, all-unflagged degenerate path returns `clean=None / delta=None` honestly.
* `TestDegradeNeverRaises` (3 tests) — malformed rows drop themselves only (good rows survive), `None` input → INSUFFICIENT_DATA, missing DB → status='error'.
* `TestAnalyzeAgainstRealDb` (1 test) — round-trip through real SQLite so a schema drift between this test's fixture and `BacktestStore` surfaces.
* `TestCli` (3 tests) — `--json` payload schema, exit code 0 (CLEAN) vs 2 (ACTIVE_LEAK) vs 3 (missing DB).

### Phase 3 — live quant findings (concrete, run against the actual deployed state)

1. **Continuous loop is DOWN locally.** `ps aux | grep continuous` returns nothing; `data/scorer_skill_log.jsonl` / `data/baseline_skill_log.jsonl` both absent. Last `decision_scorer.pkl` mtime is ~36.5h ago (`scorer_freshness` reads `pkl: age_h=36.49`). With the loop down, no skill ledger is being maintained, so `skill_trend` / `overfit_gap` / `baseline_trend` all read `INSUFFICIENT_DATA` ("0 usable cycles") and cannot answer "is OOS skill improving" durably.

2. **Deployed scorer pickle is BELOW the conviction-gate threshold.** `n_train=400 < 500` (invariant #5), so the conviction gate is **inactive** in production. `data/decision_outcomes.jsonl` has **7,413** rows ready to push `n_train` from 400 → ~6,906 (after dedup) on the next retrain cycle — so a single restart of `run_continuous_backtests.py` immediately re-arms the gate. Operational, not a code bug.

3. **OOS gate diagnostics on the live slice are HEALTHY** (the substantive finding for a skeptical quant). On the 1,482 OOS records:
   * `gate_audit`: **GATE_EFFECTIVE** — strong_tailwind realized `+3.69%`, strong_headwind `-2.38%`, spread `+6.07pp`, `arm_monotone=1.0` (every arm correctly ordered).
   * `gate_utilization`: **BALANCED** — every arm 11.88%-34.41%, `side_balance=0.497` (symmetric), distribution `min=-50/median=-0.05/max=+50`.
   * `baseline_compare` (the decisive question — does the 17-feature MLP carry more OOS rank skill than a one-line rule?): **MLP_ADDS_SKILL** — MLP `rank_ic=+0.2245 dir_acc=0.6111`, beats best baseline (`rsi_meanrev`, `rank_ic=+0.079`) by `+0.145`, well above the 0.05 skill floor. The MLP is genuinely additive OOS.
   * `deploy_audit`: **DEPLOYED_MATCHES_SOURCE** — all 8 hyper-params match `MLP_CONFIG`, no architectural drift.

   So once `n_train` crosses 500, the gate will engage on a model that is both economically effective AND structurally reachable AND genuinely skill-adding versus trivial baselines. The path is clear; only the operational gap (loop down) blocks the unlock.

4. **Cross-domain leak risk (report-only; strategy.py owns the fix).** `strategy.py::_ml_is_qualified` SQL filter is `status='complete' AND vs_spy_pct IS NOT NULL AND n_trades >= 5 ORDER BY run_id DESC LIMIT 20`. None of those clauses excludes the 80 historical unflagged-degenerate rows (they have non-NULL but fabricated `vs_spy_pct`). The current window happens to be clean by luck — `max_unflagged_run_id=6145` is well below the last 20 qualifying runs — but if the engine ever pauses long enough that the qualifier window receeds back into run_ids ≤6145, contaminated rows would re-enter the median and could falsely qualify the ML advisor. Defensive fix would be adding `AND NOT (spy_return_pct = 0.0 AND (julianday(end_date) - julianday(start_date)) >= 30)` to the qualifier SQL. Cross-domain (live trader strategy), not in this pass's scope.

5. **Calibration on the FULL corpus reads MISCALIBRATED** (`spearman=0.016`) but this is the well-documented in-sample noise artifact — the MLP memorizes the 80% training split, so a Spearman over all 7,413 rows is dominated by the noisy in-sample portion. The honest OOS-only view (baseline_compare above) shows the actual additive skill. This is exactly why the loop's skill ledger reports OOS metrics rather than the in-sample calibration verdict.

6. **`outcome_data_quality` reads CLEAN**: 7,413 rows parsed, 0 corrupt, 0 conflicts. Distribution `mean=+1.09% std=9.52 p1=-23.48% p99=+29.35%`. 25 rows have `|fr_5d|>50%` (the symmetric label-clamp at train time handles these correctly). Training-set integrity is intact.

7. **3 sibling Claude agents running concurrently** (this agent + 2 others all dispatched at ~18:17 UTC against the same paper-trader-core / digital-intern trees). Each is a `claude --print` subprocess holding ~230MB RSS. Past pass documented [PT concurrent same-role staging race](../../../.claude/projects/-home-zeph-trading-intelligence/memory/pt-concurrent-samerole-staging-race.md): a sibling agent's whole-file `git add` between this agent's stage and commit could bundle their work into ours. I staged only the two files I created (one new module + its test file), verified with `git diff --staged --stat` before committing — no contamination.

8. **Pre-existing test failure surfaces in the full suite (NOT this pass's fix).** `tests/test_swr_prewarm_coverage.py::test_every_swr_cached_endpoint_is_prewarmed` fails because the `today-action-tape` endpoint was added with `@swr_cached` but never registered in `_swr_prewarm.targets`. That endpoint is in the core/dashboard domain (Agent 1), not ML/backtest. Verified the failure exists on master before my changes by stashing+re-running. Reported here so the next core hybrid pass knows to look.

9. **`data/paper_trader.db?mode=ro` empty orphan file persists** from the 2026-05-20 core hybrid pass observation. Still safe to delete; not mine to remove this pass.

### Counters

`bugs_fixed=0` (honest no-commit per advisor guidance — the three reviewed suspects all checked out clean) · `features_added=1` (`benchmark_integrity` diagnostic + 25 tests + AGENTS.md docs) · `user_findings=9` (continuous loop DOWN locally, deployed pkl n_train=400 below gate threshold, OOS gate is HEALTHY/EFFECTIVE/BALANCED on 1482-record slice, MLP_ADDS_SKILL with +0.225 rank_ic, strategy.py `_ml_is_qualified` is structurally vulnerable to historical leak though current window is clean, calibration MISCALIBRATED on full corpus is in-sample-noise artifact not real degradation, outcome data quality CLEAN with intact training integrity, 3 concurrent same-role siblings without contamination, pre-existing core/dashboard test failure unrelated to ML/backtest).

### How the ML decision scorer works (consolidated reference)

The scorer is a small `MLPRegressor(32, 16)` (sklearn, with numpy `lstsq` fallback) that maps a 17-dim feature vector to a predicted 5-day forward return %.

* **Features** (`paper_trader/ml/decision_scorer.py::build_features`, 17 dims, single source of truth in `FEATURE_NAMES`): `ml_score`, `rsi`, `macd`, `mom5`, `mom20`, `regime_mult`, `vol_ratio`, `bb_pos`, `news_urgency`, `news_article_count`, plus 7-way sector one-hot (`tech / energy / financials / healthcare / commodities / crypto / other`).
* **Sector mapping** (`SECTOR_MAP`): explicit ticker → sector dict. `INTENTIONALLY_OTHER` frozenset pins tickers that fall through to `other` by design (TM/UXI/NAIL/DFEN/UTSL/XLI — no industrial/utility/real-estate/defense sector enum). A test (`test_decision_scorer.py`) asserts `WATCHLIST ⊆ (SECTOR_MAP ∪ INTENTIONALLY_OTHER)` so a NEW watchlist ticker without explicit classification fails loudly rather than silently degrading.
* **Hyper-params** live in module-level `MLP_CONFIG` (single source of truth shared with `deploy_audit`). Anti-overfit config: `hidden_layer_sizes=(32, 16)`, `alpha=1e-2`, `early_stopping=True`, `validation_fraction=0.15`, `n_iter_no_change=25`. Drift would be caught by `deploy_audit.is_deploy_stale`.
* **Prediction clamp** `PRED_CLAMP_PCT=50.0` — the MLP head is unbounded; any raw output exceeding ±50 is flagged off-distribution via `predict_with_meta` and the conviction-gate honours that abstention.
* **Training** (`train_scorer`): dedup by `(ticker, sim_date, action)` keeping the highest-`return_pct` copy; drop invalid `forward_return_5d` (None/NaN/inf/bool/string) BEFORE `build_features` (counted in `n_label_dropped`); symmetric label clamp at ±`PRED_CLAMP_PCT` (counted in `n_label_clamped`); sample-weight oversampling rounds `w*2` to ints, drops CONDEMN rows (rep=0) per the documented `llm_mult=0.1` semantic; atomic write via tmp+`.replace` so a torn pickle is impossible.
* **Gate** (`paper_trader/backtest.py::_ml_decide`): conviction multiplier only — NEVER cancels a trade (an earlier blocking version oscillated SOXL/TQQQ strategies, see the inline comment). Arms: `< -10 → ×0.6`, `< 0 → ×0.85`, `0..5 → unchanged`, `> 5 → ×1.15`, `> 10 → ×1.30`. Fires only when `is_trained AND _n_train >= 500 AND NOT scorer_off_dist`.
* **Outcome capture** (`run_continuous_backtests.py::_compute_decision_outcomes`): every FILLED BUY/SELL across ALL runs in a cycle (winners + losers — survivorship bias avoidance), forward-window from `PriceCache` so no network, walk-back collision check (`sim_res == end_res` ⇒ drop) so a fabricated 0% outcome can't enter training. Additive captures: `gate_scorer_pred / gate_off_dist` (gate's then-deployed prediction), `persona / regime_label / wk52_pos / pct_from_52h / forward_return_10d / forward_return_20d` — read-only research signal, training is unchanged.

### How to run backtests manually

```bash
# Live trader (live decisions, 60s cadence when market open)
python3 -m paper_trader.runner

# One-shot 10-run backtest launcher (BacktestEngine().run_all(10))
python3 run_backtests.py

# Continuous backtest loop (1 run/cycle currently — throttled for load,
# random 1-10yr window each cycle, ~10min cooldown)
python3 run_continuous_backtests.py

# Inspect a single scorer prediction with per-feature attribution
python3 -m paper_trader.ml.decision_scorer \
  --explain --ticker NVDA --ml-score 4.0 --rsi 70 --macd 2.0 \
  --mom5 5.0 --mom20 12.0 --regime-mult 1.0
```

### How to interpret backtest results (quant cheatsheet)

* **Returns alone are not skill.** `total_return_pct` can be inflated by leveraged-ETF over-sizing in a bull regime. Read `vs_spy_pct` (alpha) instead — BUT run `benchmark_integrity` first to see how much of historical alpha is fake-benchmark (the documented 16.8% historical-contamination figure).
* **`benchmark_unavailable` in the `notes` column** ⇒ that row's `spy_return_pct=0.0` is fabricated; treat `vs_spy_pct` as raw return only.
* **OOS rank-IC is the headline metric, not `val_rmse`.** The in-engine `val_rmse` is computed on a random 80/20 split (information leakage when records span time). `oos_rmse` / `oos_dir_acc` / `oos_ic` in the scorer-skill-log row come from a TEMPORAL holdout — that's the trustworthy view.
* **Three things must align to trust the gate:** (a) deployed `n_train >= 500` (`scorer_freshness`), (b) gate ordering economically correct (`gate_audit GATE_EFFECTIVE`), (c) prediction distribution reaches all arms (`gate_utilization BALANCED`). All three currently green on OOS; only (a) is below threshold on the deployed pickle.
* **`baseline_compare` is decisive.** If MLP rank-IC doesn't beat the best one-line baseline (`rsi_meanrev`, `mom20`, etc.) by ≥0.05, the 17-feature model is net-negative complexity and the conviction gate is sizing variance with no real edge. Currently: MLP +0.225 vs best baseline +0.079 ⇒ MLP_ADDS_SKILL.

### Test commands for the ML/backtest domain

```bash
# Focused ML/backtest tests (~10s; this is the dev-loop gate):
cd /home/zeph/trading-intelligence/paper-trader && \
  python3 -m pytest tests/test_decision_scorer.py \
                    tests/test_decision_scorer_attribution.py \
                    tests/test_backtest.py tests/test_continuous.py \
                    tests/test_calibration.py tests/test_validation.py \
                    tests/test_integration_backtest.py \
                    tests/test_ml_backtest_coverage.py \
                    tests/test_ml_backtest_review.py \
                    tests/test_ml_backtest_seams.py \
                    tests/test_ml_backtest_word_boundary.py \
                    tests/test_scorer_freshness.py tests/test_scorer_honesty.py \
                    tests/test_scorer_smoke_test.py tests/test_gate_audit.py \
                    tests/test_gate_utilization.py \
                    tests/test_benchmark_integrity.py -q

# Just the new benchmark_integrity tests (~1s):
python3 -m pytest tests/test_benchmark_integrity.py -q

# Full read-only diagnostic suite against live state:
python3 -m paper_trader.ml.benchmark_integrity    # SPY-benchmark integrity (NEW)
python3 -m paper_trader.ml.gate_audit             # Economic gate-arm spread
python3 -m paper_trader.ml.gate_utilization       # Distributional gate reachability
python3 -m paper_trader.ml.baseline_compare       # MLP vs trivial-baseline rank-IC
python3 -m paper_trader.ml.calibration            # Per-decile predicted-vs-realized
python3 -m paper_trader.ml.deploy_audit           # Deployed pickle matches MLP_CONFIG?
python3 -m paper_trader.ml.scorer_freshness       # Pickle age + skill-log heartbeat
python3 -m paper_trader.ml.outcome_data_quality   # decision_outcomes.jsonl integrity
python3 -m paper_trader.ml.overfit_gap            # val_rmse vs oos_rmse trend
```

### Invariants reaffirmed by this pass

* **#10 single source of truth** — `benchmark_integrity` mirrors the in-engine guard's degenerate criterion verbatim (`spy_return_pct=0.0 AND window>=30d`); a future tightening of the guard requires a matching update here.
* **#5 conviction gate** — gate fires only at `n_train >= 500`; new diagnostic is observational only, never gates or alters Opus.
* **#2 / #12 advisory only** — read-only, no pickle/train/feature/trade-path touch; same discipline as every other `paper_trader/ml/*` diagnostic.
* **Backtest articles never reach live signals (#1)** — diagnostic only reads `backtest_runs.*` columns, never reads `articles.db`; cannot accidentally cross the boundary.

## 2026-05-21 paper-trader-core HYBRID pass — cash-delta in trade-impact alert

A live trader's #1 follow-up after a fill is "how big is this name now and
how much cash do I have left?" — already answered by the post-trade impact
line (book-weight %, avg cost, current cash). Their **#2 follow-up** is
"how much of my cash did this just consume / free?" — the sizing input for
the NEXT decision. Absolute cash alone tells them what's *left*; the
signed delta tells them what just *moved*. Previously the alert showed
only the former.

### Phase 2 (feat) — `_cash_delta_token` in `reporter.py`

New helper `_cash_delta_token(cash, trade)` produces the cash token with a
signed delta:

* **BUY**:  `cash $565.08 (-$446.87)` — minus because cash decreased
* **SELL**: `cash $701.50 (+$301.36)` — plus because cash freed

Pure: derived from the in-hand `trade.value` (already `qty × price` for
stock, `qty × price × 100` for options — `store.record_trade` contract).
No network, no extra store reads, no re-derivation of cash arithmetic.
Composed into both branches of `_trade_impact_line` (the BUY-affordability
and SELL-realized contexts each grow the cash token to `cash $X (-$Y)` /
`cash $X (+$Y)`).

**Degrade-safe contract** mirrors the rest of `reporter`:

* missing / non-numeric / zero / negative `trade.value` → bare cash token,
  byte-identical to the pre-feature output (a hand-built test trade
  without a `value` field stays green);
* HOLD / REBALANCE / unknown action → no delta (the action discriminator
  prevents inventing a misleading "deployed" on a non-trading event);
* never raises — the additive reporter contract holds end-to-end.

### Tests

`tests/test_core_reporter.py` extended with:

* `TestCashDeltaToken` (10 unit tests) — pins SPECIFIC numerics for the
  sign convention (BUY=minus, SELL=plus), option scaling (the `× 100`
  multiplier is already in `trade.value`), and every degrade-safe branch
  (missing/garbage/zero/negative value, unknown action, negative cash).
* `TestTradeImpactLineCashDelta` (4 integration-shape tests) — pins the
  full `_trade_impact_line` composition including the delta in the
  canonical token order for BUY, partial SELL, full-close SELL, and the
  zero-value degrade-safe path.
* 6 pre-existing `TestTradeImpactLineUnit` exact-match assertions updated
  to expect the new delta — *spec updates*, not weakened tests (the
  exact-match form is preserved with the new expected output, so a
  future regression that drops the delta still fails loudly).

Verification:

```bash
cd /home/zeph/trading-intelligence/paper-trader
python3 -m pytest tests/test_core_reporter.py -k "Cash or TradeImpact or TradeAlert" -q  # 46 pass
python3 -m pytest tests/test_core_reporter.py tests/test_core_strategy.py \
                  tests/test_core_runner.py tests/test_core_signals.py \
                  tests/test_core_market.py tests/test_core_store.py -q              # 653 pass
```

### Phase 3 — live validation findings (trader perspective, observational)

Inspected `/api/healthz`, `/api/portfolio`, `/api/runner-heartbeat`,
`/api/host-guard`, `/api/empty-claude-rate`, `/api/decision-health`,
`/api/capital-paralysis`, `/api/benchmark`, `/api/drawdown`,
`/api/event-calendar`, and the last 8 rows of `decisions`:

* **Healthy and trading.** Heartbeat HEALTHY, lock acquired, Discord
  delivery HEALTHY, 1 open NVDA position, total $1011.95 (+1.19% from
  $1000 start), beating SPY buy-and-hold by **+0.79pp** ("ahead in 65.7%
  of 204 cycles"), 0.56% off peak ($1017.63), 11.6h in this DD.
* **Decision-efficacy DEGRADED right now.** 25% of the last 20 cycles
  were NO_DECISION; root cause visible in `/api/host-guard`:
  **`5 concurrent Opus (>4 threshold) — host saturated`**. **Working as
  designed**: this is the parallel-agent storm (the hybrid review agents
  themselves consume Opus slots — the trader correctly skips its doomed
  call rather than spawning a 1.5GB OOM-killer subprocess, and records
  a clean `skipped claude call —` instead of the empty/timeout signature
  that would inflate `model_reliability`). No trader-side bug.
* **MRVL earnings 5.89d away** on the watchlist — surfaces in the live
  Opus prompt's `event_calendar` block exactly as documented in #10.
* `unrealized_pl=0.0` on the NVDA lot (avg=mark=$223.43) reads as a
  flat position but is a market-closed coincidence (yfinance returns the
  same final price each call after the close); `stale_marks=0` confirms
  the mark is genuine, not a missing-price fallback. **Not a bug**.

### Invariants reaffirmed

* **Reporter additive contract (#2/#12)** — `_cash_delta_token` is pure
  observation, never gates, never invents data (no `value` → no delta).
* **Single source of truth (#10)** — the cash delta is `trade.value`
  verbatim (the same field `store.record_trade` writes), so the alert
  surface, the daily-close cash-flow P&L line and any future per-trade
  inspector can never disagree on what THIS trade moved.

### Counters

`bugs_fixed=0` (mature codebase — no concrete bug worth a fix this pass)
· `features_added=1` (cash-delta token + 14 new tests + 6 spec-updated)
· `user_findings=5` (HEALTHY heartbeat, BEATING SPY +0.79pp, host-saturation
storm from 5 parallel Opus is *expected* under multi-agent reviews and
correctly recorded as `skipped claude call`, 0.56% off peak, MRVL earnings
in 5.89d on watchlist).

## 2026-05-20 ML/backtest hybrid pass (Agent 2) — Broadcom-dead-mapping fix + dead_sector_audit diagnostic + stale-loop quant audit

### Phase 1 — fix (commit `15f6d92`)

Dead `_WORD_TO_TICKER` entry caught by a single-pass watchlist-coverage
scan: `"broadcom" → "AVGO"` had been silently dropped on every
Broadcom-headline call to `_ml_decide` because AVGO is NOT in
`WATCHLIST`, and the function's `if tk not in WATCHLIST: continue`
guard filters every extracted ticker not on the live list. Every other
of the 85 keyword mappings is now confirmed in-WATCHLIST by a generic
regression test that would fail RED on any future dead entry. Redirect
the keyword to **SOXL** (3x semis ETF, already in WATCHLIST) so
Broadcom news contributes to the same semi tracker as the existing
`"semiconductor" / "chip"` mappings.

Tests added (`tests/test_backtest.py::TestWordToTickerWatchlistCoverage`,
4 new):
* `test_no_dead_word_to_ticker_mappings` — generic regression guard.
* `test_broadcom_maps_to_soxl_not_dead_avgo` — pin the specific redirect.
* `test_pattern_compiled_for_every_mapping_key` — 1:1 contract between
  `_WORD_TO_TICKER` and `_WORD_TO_TICKER_PATTERNS`.
* `test_broadcom_headline_attributes_to_soxl` — end-to-end through
  `_ml_decide` with a synthetic price + portfolio.

### Phase 2 — feature (commit `8ee94b6`)

`paper_trader/ml/dead_sector_audit.py` — cross-references per-sector
record counts in `decision_outcomes.jsonl` against the deployed
scorer's per-sector first-layer weight share. The 40+ existing ML
diagnostics include `feature_importance` (per-feature importance, but
cannot distinguish "no data" from "model failure") and
`feature_coverage` (default-frequency, misleading for sectors — every
non-majority sector is at zero ~90% by construction). **Neither tool
answers the gate-relevant question for the operator: "do I have
sectors that carry real outcome mass yet contribute nothing to
predictions?"** This module fills the gap with one CLI call:

```bash
python3 -m paper_trader.ml.dead_sector_audit
```

Per-sector verdict (each threshold-driven):

| Verdict | Trigger |
|---|---|
| `DEAD_FEATURE` | `n_outcomes >= MIN_RECORDS=30` AND `importance_share < MIN_IMPORTANCE_SHARE=0.005` (THE ALERT) |
| `SPARSE_DATA`  | `n_outcomes < MIN_RECORDS` (expected near-zero importance) |
| `HEALTHY`      | otherwise |

Overall verdict: `INSUFFICIENT_DATA` / `HAS_DEAD_SECTORS` / `HEALTHY`.
The audit surfaces `corpus_growth_ratio = n_outcomes_total /
scorer.n_train` so the operator can distinguish "retrain will fix it"
(≥2×) from "data-pipeline issue" (≈1.0). Read-only — no train, no
pickle write, no trade-path touch. Exit codes match host_guard /
decision_scorer (0=HEALTHY, 1=INSUFFICIENT_DATA, 2=HAS_DEAD_SECTORS).

Live read on production `decision_scorer.pkl` (n_train=400) and
`decision_outcomes.jsonl` (7413 records):

```
verdict=HAS_DEAD_SECTORS  n_outcomes=7413  n_train_in_pickle=400  corpus_growth_ratio=18.53×
  sector          n_outcomes     share  verdict
  energy                 117     0.00%  DEAD_FEATURE
  crypto                 626     0.00%  DEAD_FEATURE
  healthcare            1043     5.93%  HEALTHY
  tech                  4793     5.48%  HEALTHY
  financials             613     4.09%  HEALTHY
  commodities              5     4.11%  SPARSE_DATA
  hint: 2 dead sector(s): energy,crypto. corpus_growth_ratio=18.53×
        — retrain will almost certainly fix it.
```

The deployed scorer is structurally mute on energy and crypto despite
626 crypto records (mostly MSTR) and 117 energy records existing in
the training pool. A retrain via the continuous loop would resolve it
— corpus_growth_ratio=18.5× shows the pickle predates the bulk of the
corpus.

Tests added (`tests/test_dead_sector_audit.py`, 27 new):
* `TestClassifySector` (3) — verdict boundaries.
* `TestPerSectorCounts` (5) — record→sector counting, None/missing
  ticker, unknown ticker → "other", empty-input zero counts.
* `TestReport` (12) — composition: insufficient-data, healthy,
  dead+sparse boundaries, growth ratio computation, retrain-hint
  threshold, sort order (dead first), malformed input safety.
* `TestAnalyze` (3) — missing file, empty file, corrupt JSON.
* `TestCli` (4) — exit codes 0/1/2, `--json` round-trip.

### Phase 3 — live validation findings (quant researcher perspective)

Inspected scorer pickle, outcomes file, backtest DB, continuous.log,
calibration and baseline_compare on production data.

1. **Continuous loop is DEAD** since 2026-05-18 12:03 (~2.5 days).
   No `continuous_backtests.py` process running. The whole ML
   pipeline (cycle-driven retraining, outcome accumulation,
   ArticleNet injection) is stalled. The last log entry shows run
   #6243 mid-cycle at step 1311/2516 of a 10-year window —
   probably killed by OOM or manual stop while battling GDELT
   rate-limits.
2. **2 orphaned `running` backtest rows** stuck since 2026-05-18:
   #6238 (5-yr window) and #6243 (10-yr window mid-step). The
   `_reap_orphaned_runs` helper is loop-coupled — it only fires from
   `main()`. With the loop down it never runs, so the dashboard
   shows these as in-flight indefinitely (CLAUDE.md §11 documents
   this exact symptom class).
3. **Dead sectors in deployed scorer** — confirmed by the new
   `dead_sector_audit`: sector_energy and sector_crypto have 0.0
   first-layer weight share despite carrying 117 and 626 outcomes
   respectively. The model is structurally mute on those sectors
   right now. Retrain (corpus_growth_ratio=18.5×) would resolve.
4. **`scorer_skill_log.jsonl` does not exist** on disk. The per-cycle
   ledger has never been populated, so `scorer_freshness` heartbeat
   is uncomputable. Either the loop was never restarted after the
   ledger was added, or the file was manually deleted. Either way,
   one of the documented liveness watchdogs has nothing to read.
5. **Production scorer n_train=400 < gate threshold 500** →
   conviction gate INACTIVE in production. `_ml_decide` runs but
   does NOT modulate conviction on any BUY. All sizing currently
   relies on quant signals + persona boosts alone. Per CLAUDE.md
   invariant #5 this is the correct behaviour below the threshold
   — but a fresh retrain on the 7413-row corpus would re-arm it.
6. **Calibration / OOS-skill divergence** —
   `python3 -m paper_trader.ml.calibration` reports `MISCALIBRATED`
   (spearman 0.0157, decile-mean realized flat at +0.1..+1.4% across
   all 10 deciles) on the full sample. Simultaneously
   `python3 -m paper_trader.ml.baseline_compare` reports
   `MLP_ADDS_SKILL` (OOS rank_ic +0.225, beats best baseline
   `rsi_meanrev` +0.079 by +0.145). The scorer has directional rank
   skill OOS but its prediction MAGNITUDES don't track realized
   outcomes — the decile bucket boundaries are saturated to
   ±PRED_CLAMP_PCT=50 while realized means cluster near 0. A quant
   should size by direction only, not magnitude, until the
   magnitude calibration improves (the planned retrain may help).
7. **GDELT instability is the root cause of the loop death.**
   continuous.log tail shows hundreds of `[gdelt] rate-limited` and
   `ConnectionError` retries, each backing off 20-60s. With 2516
   trading days × multiple keyword groups per 10-yr window, that's
   thousands of GDELT calls — the process spent most of its time in
   retry sleeps before being killed. The disk-cache + per-(date,
   keyword) negative-cache work, but cold-cache cycles are
   compute-starved. Operationally: starting the loop on cold cache
   requires patience and possibly capping window length.

### Counters

`bugs_fixed=1` (broadcom→AVGO dead mapping → SOXL redirect + 4 tests)
· `features_added=1` (dead_sector_audit + 27 tests)
· `user_findings=7` (loop dead, orphaned rows, dead sectors,
no skill ledger, gate inactive, calibration/OOS divergence, GDELT
instability)
