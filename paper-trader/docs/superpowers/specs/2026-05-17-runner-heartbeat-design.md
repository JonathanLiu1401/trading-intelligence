# Runner heartbeat — is the trading loop itself alive? (design)

*2026-05-17 — Agent 4 (feature-dev)*

## Problem

Every behavioural/diagnostic endpoint on the desk (`decision-health`,
`-forensics`, `-drought`, `-reliability`, `feed-health`, `build-info`, …)
reasons over **rows that exist** in `decisions` (or over code SHA / article
age). None answer the upstream question they all presuppose:

> *Has the trading loop produced **any** decision recently, versus its
> expected cadence?*

Verified directly against the code (not just AGENTS.md):

- `analytics/decision_drought.build_decision_drought` uses `now` **only** for
  the `as_of` display string. The "ongoing" drought's `duration_hours` is
  `last_row_ts − first_row_ts` — it **freezes** the instant the runner stops
  emitting rows. No verdict closes on `now − max(decisions.timestamp)`.
- `analytics/decision_reliability` uses `now` only for the decisions/day
  cadence denominator.
- `feed_health.blind_streak` counts consecutive 0-signal **decision rows**;
  it cannot grow when no rows are written. `STALE_HOURS=6` is *article* age.
- `build-info.stale` detects a stale **code** SHA, not a stalled loop.

So a dead or wedged `paper_trader.runner` is **invisible** from the trader's
own analytics surface: the panels show frozen-but-plausible state. This is
not hypothetical — at design time the live runner process was **down**
(only `run_continuous_backtests.py` running) and nothing said so.

## Approach

A new pure builder + read-only endpoint + dashboard card, following the
exact established convention (`feed_health` is the closest precedent:
pure builder, `now` injected, module-owned threshold constants, advisory
verdict precedence, JS degrades via the `/api/build-info` `stale` contract).

Considered and rejected:
- *Extend `decision_drought`* — it is row-driven by contract and reused
  verbatim by `trader_scorecard`/`capital_paralysis`; adding a wall-clock
  liveness verdict would change a single-source-of-truth builder consumed
  elsewhere. A separate concern deserves a separate module.
- *Process-up probe (`pgrep`)* — a process can be up while the cycle is
  wedged (stuck `claude` subprocess, deadlock). The honest signal is
  "no new decision row in N× the expected interval", not "PID exists".

## Component design

### `paper_trader/analytics/runner_heartbeat.py` (new, pure, no I/O)

```python
OPEN_INTERVAL_S   = 1800.0   # mirrors runner.OPEN_INTERVAL_S   (market open)
CLOSED_INTERVAL_S = 3600.0   # mirrors runner.CLOSED_INTERVAL_S (market closed)
LAGGING_MULT = 1.25
STALLED_MULT = 2.0

def build_runner_heartbeat(
    last_decision_ts: str | None,
    market_open: bool,
    now: datetime | None = None,
) -> dict
```

The module **owns** its cadence constants (the `feed_health.STALE_HOURS`
precedent — module is the spec, the test reads the module constant so a
retune can't false-fail). It does **not** import `runner` (zero circular
risk, fully offline-testable).

Verdict precedence (judged on `secs_since` vs `expected_interval`,
`expected = OPEN_INTERVAL_S if market_open else CLOSED_INTERVAL_S`):

| order | verdict | condition | `restart_recommended` |
|------|---------|-----------|-----------------------|
| 1 | `NO_DATA` | `last_decision_ts` absent / unparseable | `False` |
| 2 | `STALLED` | `secs_since > STALLED_MULT × expected` | **`True`** |
| 3 | `LAGGING` | `secs_since > LAGGING_MULT × expected` | `False` |
| 4 | `HEALTHY` | otherwise (incl. a future-skewed ts) | `False` |

Output keys: `as_of`, `market_open`, `expected_interval_s`,
`last_decision_ts`, `secs_since_last_decision`, `intervals_elapsed`
(`secs_since / expected`, clamped ≥ 0, `None` when NO_DATA), `verdict`,
`headline` (the **single source of truth** string the UI renders),
`restart_recommended`, plus the echoed constants
(`lagging_mult`, `stalled_mult`).

A future-dated timestamp (clock skew) → negative `secs_since` → naturally
`HEALTHY` (a just-written decision), `intervals_elapsed` clamped to 0.0.
Builder **never raises** — an unparseable ts degrades to `NO_DATA`.

**Advisory only.** It states a fact about loop liveness; it issues no
directive, imposes no cap, and has no path to `_execute()`. It does **not**
violate "no hard risk limits / Opus has full autonomy" (invariants #2/#12)
— that governs *gating decisions*, not *observing the loop*; same reasoning
as `feed_health`/`self_review`. A mirror, not a cage.

### `GET /api/runner-heartbeat` (new route in `dashboard.py`)

Network/IO in the endpoint (the `thesis_drift` "network in the endpoint,
builder takes the dicts" split): `store.recent_decisions(1)` for the newest
`timestamp`, `market.is_market_open(now_utc)` for the cadence selector,
`datetime.now(timezone.utc)` for `now`. Broad `except` → `{"error": …}, 500`
(the universal route contract). CORS already wide-open.

### UI — `rhb-` card on the `:8090` trader page

Fresh id prefix `rhb-` (invariant #14 — verified absent). Placed directly
after `score-card`. State badge colour map: `STALLED` red, `LAGGING` amber,
`HEALTHY` green, `NO_DATA` grey. `refreshRunnerHeartbeat()` mirrors
`refreshCorrelation()` verbatim incl. the `__unavailable` →
`markStale(...)` "restart paper-trader to apply" degrade (the running
`:8090` predates this commit, so it 404s there until restart — the
documented chronic-stale pattern). Polled every 30 s (a dead loop should
surface fast; the probe is one indexed `LIMIT 1` read).

## Testing (TDD — test written first, watched RED)

`tests/test_runner_heartbeat.py`, exact-value fixtures, deterministic
(injected `now`, no clock/network):

- each verdict at its **boundary**: just-inside `HEALTHY`, just-over
  `LAGGING` (`1.25×+ε`), just-over `STALLED` (`2.0×+ε`); market-open
  (1800 s) vs market-closed (3600 s) select different thresholds from the
  same elapsed gap (a 70-min gap is `HEALTHY` closed but `STALLED` open).
- `NO_DATA` on `None` and on an unparseable ts; `restart_recommended`
  only `True` for `STALLED`.
- future-skewed ts → `HEALTHY`, `intervals_elapsed == 0.0`, never raises.
- thresholds read from the live module constants (retune-proof).
- endpoint end-to-end via the Flask test client on a real temp `Store`:
  a stale-seeded decision row yields `STALLED`; empty `decisions` →
  `NO_DATA`; mirrors `test_feed_health_endpoint.py`.

## Scope

In: builder + endpoint + card + tests + AGENTS.md endpoint row.
Out (explicit, deferred — collision-avoidance with concurrent agents +
the no-remote `/home/zeph` repo): the `unified_dashboard.py` /
`web_server.py` `/api/chat` sub-fetch line. Documented as a follow-up in
the AGENTS.md row, exactly as other endpoints stage their chat wiring.

## Live-state expectation

The runner is **down at ship time**, so live `/api/runner-heartbeat` will
honestly report `STALLED, restart_recommended:true`. When the operator
restarts the runner it flips to `HEALTHY` on the next cycle — that
transition is the visible proof the feature works. The agent does **not**
restart the runner (outside the feature-dev brief; the down state may be
deliberate while the review agents run).
