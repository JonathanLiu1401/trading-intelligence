# HF Model Competition — Design Spec
**Date:** 2026-05-20  
**Status:** Approved

## Goal
Add HuggingFace-hosted LLMs (DeepSeek R1, Llama 70B, Qwen3 32B) as competing decision engines in the backtest framework, alongside the existing Claude Opus 4.7 and deterministic ML+quant paths. A new "Model Rankings" tab on `/backtests` shows average returns, win rate, and vs-SPY performance grouped by model.

Phase 2 (out of scope here): fine-tune a specialized smaller model on accumulated decision data.

---

## Architecture

### Files

| File | Change |
|------|--------|
| `paper_trader/llm_adapter.py` | NEW — unified `call_llm(model_id, prompt)` |
| `paper_trader/backtest.py` | MODIFIED — `BacktestRunner` takes `model_id`, delegates to adapter |
| `paper_trader/dashboard.py` | MODIFIED — `/api/model-rankings` + Model Rankings tab |
| `paper_trader/strategy.py` | UNTOUCHED — live trader stays on Claude Opus exclusively |

### Data flow
```
CLI: --model hf/deepseek-ai/DeepSeek-R1
  → BacktestRunner(model_id="hf/deepseek-ai/DeepSeek-R1")
  → per decision: llm_adapter.call_llm("hf/...", prompt)
  → POST https://router.huggingface.co/v1/chat/completions
  → _parse_decision() [unchanged parser]
  → stored in backtest_decisions; run tagged with model_id
```

---

## Component: `llm_adapter.py`

```python
HF_BASE = "https://router.huggingface.co/v1"
_HF_SEM = threading.Semaphore(3)    # HF: up to 3 concurrent calls
_CLAUDE_SEM = threading.Semaphore(2) # Claude: max 2 concurrent (OOM guard) — MOVED here from backtest.py

def call_llm(model_id: str, prompt: str, timeout: int = 90) -> str | None:
    """Route prompt to the right backend. Returns raw string or None."""
    if model_id.startswith("hf/"):
        return _hf_call(model_id[3:], prompt, timeout)
    elif model_id.startswith("claude-"):
        return _claude_call(model_id, prompt)   # existing logic, relocated here
    raise ValueError(f"Unknown model_id: {model_id!r}")
```

**Migration note:** `_CLAUDE_SEM` moves from `backtest.py` into `llm_adapter.py`. The existing `_claude_call` in `backtest.py` is replaced with a call to `llm_adapter.call_llm`. Remove the now-orphaned `_CLAUDE_SEM` definition from `backtest.py` to avoid a silent duplicate.

**`_hf_call` specifics:**
- Auth: `HUGGINGFACE_HUB_TOKEN` → `HF_TOKEN` → parse from `.env` file
- Endpoint: `POST {HF_BASE}/chat/completions`
- Payload: `{"model": hf_model, "messages": [{"role":"user","content":prompt}], "max_tokens": 512}`
- Timeout: 90s per call; 2 retries with 10s backoff on 5xx/timeout
- Returns `response["choices"][0]["message"]["content"]` or `None`

**Token resolution order:**
1. `os.environ["HUGGINGFACE_HUB_TOKEN"]`
2. `os.environ["HF_TOKEN"]`
3. Parse `HF_TOKEN=` or `HUGGINGFACE_HUB_TOKEN=` from `/home/zeph/trading-intelligence/digital-intern/.env`

**Supported model_id prefixes at launch:**
- `ml_quant` — deterministic engine (no LLM call; handled in `backtest.py`, not adapter)
- `claude-*` — Claude via subprocess
- `hf/*` — HuggingFace Inference router

---

## Component: `backtest.py` changes

### Schema migration (idempotent, runs on import)
```sql
ALTER TABLE backtest_runs ADD COLUMN model_id TEXT DEFAULT 'ml_quant';
ALTER TABLE backtest_runs ADD COLUMN hf_errors INT DEFAULT 0;
```
All 501 existing runs default to `ml_quant` correctly.

### `BacktestRunner` changes
- Constructor gains `model_id: str = "ml_quant"` parameter
- Validated on init: must start with `ml_quant`, `claude-`, or `hf/`
- Stored in `backtest_runs.model_id` on run creation
- Decision path: `ml_quant` → existing `_ml_decide`; anything else → `llm_adapter.call_llm`

### CLI flag
```
run_continuous_backtests.py --model hf/deepseek-ai/DeepSeek-R1
```
Default unchanged: `ml_quant`.

---

## Component: `dashboard.py` changes

### New API: `GET /api/model-rankings`
Query aggregates over `backtest_runs` grouped by `model_id`:

```json
{
  "models": [
    {
      "model_id": "claude-opus-4-7",
      "display_name": "Claude Opus 4.7",
      "runs": 12,
      "avg_return_pct": 47.3,
      "best_return_pct": 312.0,
      "median_return_pct": 38.1,
      "avg_vs_spy_pct": 28.4,
      "win_rate_pct": 75.0,
      "avg_trades": 89,
      "total_decisions": 4820
    }
  ],
  "as_of": "2026-05-20T08:00:00Z"
}
```

Only `complete` runs included. `display_name` mapped from a static dict: `{"claude-opus-4-7": "Claude Opus 4.7", "ml_quant": "ML+Quant (deterministic)", "hf/deepseek-ai/DeepSeek-R1": "DeepSeek R1", ...}`; unknown IDs show the raw `model_id`.

### New UI: "Model Rankings" tab
- Added as a tab on the existing `/backtests` page (no new route)
- Sortable table ranked by `avg_return_pct` descending by default
- Columns: Rank | Model | Runs | Avg Return | Best Return | Median | vs SPY | Win Rate | Avg Trades
- Each model name links to the existing backtest runs list filtered by that model
- "Run Backtest" button opens existing run modal; model selector dropdown pre-populated with supported models
- Visual badge: 🥇🥈🥉 for top 3 by avg return

---

## Error handling

| Scenario | Behaviour |
|----------|-----------|
| HF token missing | `_hf_call` returns `None`; decision recorded as `NO_DECISION`; run continues |
| HF 429 / rate limit | Retry ×2 with 10s backoff; then `None` + increment `hf_errors` |
| HF 5xx | Same as 429 |
| HF response unparseable JSON | `_parse_decision` returns `None` (existing behaviour) |
| Unknown `model_id` prefix | `ValueError` raised in `BacktestRunner.__init__`, before any DB writes |
| HF timeout (>90s) | `None` returned; NO_DECISION recorded |

---

## OpenClaw plugin setup

Enable HF plugin in `~/.openclaw/openclaw.json`:
```json
"huggingface": { "enabled": true, "config": {} }
```
HF token must be in env or `.env`; the adapter handles both — no OpenClaw-specific auth path needed for the backtest runner itself.

---

## Testing

1. **Unit** — `llm_adapter._hf_call` with `responses` mock: verifies JSON structure, 429 retry path, missing-token early return
2. **Unit** — `/api/model-rankings` SQL query against in-memory SQLite with 3 fake runs across 2 model IDs
3. **Integration smoke** — `BacktestRunner(model_id="hf/Qwen/Qwen3-8B")` with `n_runs=1`, 3 sim days, HF mocked — asserts `model_id` stored in DB, run status `complete`
4. **Regression** — existing default `ml_quant` path unchanged: run a 1-day simulation, assert `model_id = "ml_quant"` in DB

---

## Out of scope (Phase 2)

- Fine-tuning a specialized smaller model on accumulated (prompt, decision) pairs
- Storing full prompts sent to each model (to build training dataset)
- Nightly cron to auto-run N backtests per model
- Live paper trader model swapping (strategy.py stays Claude-only)
