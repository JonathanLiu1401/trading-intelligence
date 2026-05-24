# HF Model Competition — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add HuggingFace-hosted LLMs as competing decision engines in the backtest framework and display a Model Rankings tab on the `/backtests` dashboard page.

**Architecture:** New `paper_trader/llm_adapter.py` unifies Claude CLI and HF API calls behind one interface. `BacktestEngine` gains a `model_id` param; `BacktestStore` stores it per run via a schema migration. `dashboard.py` gets a `/api/model-rankings` endpoint and a new UI tab. `strategy.py` (live trader) is never touched.

**Tech Stack:** Python 3.12, SQLite (existing `backtest.db`), HuggingFace Inference API (OpenAI-compatible), Flask (existing dashboard), pytest

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `paper_trader/llm_adapter.py` | **Create** | `call_llm(model_id, prompt)` — routes to Claude CLI or HF API |
| `paper_trader/backtest.py` | **Modify** | Schema migration; `BacktestStore.upsert_run` gains `model_id`; `BacktestEngine.__init__` gains `model_id`; `_claude_call` delegates to adapter; remove orphaned `_CLAUDE_SEM` |
| `paper_trader/dashboard.py` | **Modify** | `/api/model-rankings` Flask route + Model Rankings tab HTML/JS |
| `tests/test_llm_adapter.py` | **Create** | Unit tests for adapter routing, HF call, missing-token path |
| `tests/test_model_rankings.py` | **Create** | Unit tests for `/api/model-rankings` SQL aggregation |
| `paper_trader/strategy.py` | **UNTOUCHED** | Live trader stays Claude Opus only |

---

## Task 1: Schema migration — add `model_id` to `backtest_runs`

**Files:**
- Modify: `paper_trader/backtest.py` (SCHEMA constant + BacktestStore._migrate)

The `SCHEMA` constant uses `CREATE TABLE IF NOT EXISTS`, so existing columns are preserved. New columns must be added with `ALTER TABLE` in a separate migration that is idempotent (wrapped in a try/except that swallows `OperationalError: duplicate column name`).

- [ ] **Step 1: Write failing test**

Create `tests/test_model_rankings.py`:

```python
import sqlite3
import pytest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def make_store(tmp_path):
    from paper_trader.backtest import BacktestStore
    return BacktestStore(path=tmp_path / "test.db")

def test_backtest_runs_has_model_id_column(tmp_path):
    store = make_store(tmp_path)
    cols = [row[1] for row in store.conn.execute("PRAGMA table_info(backtest_runs)").fetchall()]
    assert "model_id" in cols

def test_backtest_runs_model_id_defaults_to_ml_quant(tmp_path):
    store = make_store(tmp_path)
    store.conn.execute(
        "INSERT INTO backtest_runs (run_id, seed, start_date, end_date, start_value, status, started_at) "
        "VALUES (1, 42, '2025-01-01', '2026-01-01', 1000.0, 'running', '2026-01-01T00:00:00Z')"
    )
    store.conn.commit()
    row = store.conn.execute("SELECT model_id FROM backtest_runs WHERE run_id=1").fetchone()
    assert row[0] == "ml_quant"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/zeph/trading-intelligence/paper-trader
python3 -m pytest tests/test_model_rankings.py -v 2>&1 | head -30
```
Expected: FAIL — `model_id` column doesn't exist yet.

- [ ] **Step 3: Add `_migrate` to `BacktestStore` and call it in `__init__`**

In `paper_trader/backtest.py`, add a `_migrate` method to `BacktestStore` right after the `__init__` definition. In `__init__`, call it after `self.conn.executescript(SCHEMA)`:

```python
# In BacktestStore.__init__, after executescript(SCHEMA) line, add:
self._migrate()

# New method on BacktestStore (add right after __init__):
def _migrate(self) -> None:
    """Idempotent schema migrations for columns added after initial deploy."""
    migrations = [
        "ALTER TABLE backtest_runs ADD COLUMN model_id TEXT NOT NULL DEFAULT 'ml_quant'",
        "ALTER TABLE backtest_runs ADD COLUMN hf_errors INT NOT NULL DEFAULT 0",
    ]
    with self._lock:
        for sql in migrations:
            try:
                self.conn.execute(sql)
                self.conn.commit()
            except Exception:
                pass  # column already exists — idempotent
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /home/zeph/trading-intelligence/paper-trader
python3 -m pytest tests/test_model_rankings.py -v
```
Expected: 2 tests PASS.

- [ ] **Step 5: Verify migration runs against the real backtest.db**

```bash
cd /home/zeph/trading-intelligence/paper-trader
python3 -c "
from paper_trader.backtest import BacktestStore
s = BacktestStore()
cols = [r[1] for r in s.conn.execute('PRAGMA table_info(backtest_runs)').fetchall()]
print('Columns:', cols)
assert 'model_id' in cols
row = s.conn.execute('SELECT model_id, COUNT(*) FROM backtest_runs GROUP BY model_id').fetchall()
print('model_id distribution:', row)
"
```
Expected: `model_id` in cols; all 501 existing runs show `ml_quant`.

- [ ] **Step 6: Compile-check and commit**

```bash
cd /home/zeph/trading-intelligence/paper-trader
python3 -m py_compile paper_trader/backtest.py
git add paper_trader/backtest.py tests/test_model_rankings.py
git commit -m "feat(backtest): add model_id + hf_errors columns to backtest_runs"
```

---

## Task 2: Create `llm_adapter.py`

**Files:**
- Create: `paper_trader/llm_adapter.py`
- Create: `tests/test_llm_adapter.py`

This module owns both LLM call semaphores and both call implementations. `backtest.py` will import from here.

- [ ] **Step 1: Write failing tests**

Create `tests/test_llm_adapter.py`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from unittest.mock import patch, MagicMock


def test_call_llm_raises_on_unknown_model():
    from paper_trader.llm_adapter import call_llm
    with pytest.raises(ValueError, match="Unknown model_id"):
        call_llm("unknown/model", "prompt")


def test_call_llm_routes_claude_to_subprocess():
    from paper_trader import llm_adapter
    with patch("paper_trader.llm_adapter._claude_call") as mock_claude:
        mock_claude.return_value = '{"action":"HOLD"}'
        result = llm_adapter.call_llm("claude-opus-4-7", "test prompt")
        mock_claude.assert_called_once_with("claude-opus-4-7", "test prompt")
        assert result == '{"action":"HOLD"}'


def test_call_llm_routes_hf_prefix():
    from paper_trader import llm_adapter
    with patch("paper_trader.llm_adapter._hf_call") as mock_hf:
        mock_hf.return_value = '{"action":"BUY"}'
        result = llm_adapter.call_llm("hf/deepseek-ai/DeepSeek-R1", "test prompt")
        # strips "hf/" prefix before passing to _hf_call
        mock_hf.assert_called_once_with("deepseek-ai/DeepSeek-R1", "test prompt", 90)
        assert result == '{"action":"BUY"}'


def test_hf_call_returns_none_when_no_token(monkeypatch):
    from paper_trader import llm_adapter
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    # Patch _load_hf_token to return None (no .env file in test env)
    with patch("paper_trader.llm_adapter._load_hf_token", return_value=None):
        result = llm_adapter._hf_call("deepseek-ai/DeepSeek-R1", "prompt", 10)
    assert result is None


def test_hf_call_sends_correct_payload(monkeypatch):
    import json
    from paper_trader import llm_adapter
    monkeypatch.setenv("HUGGINGFACE_HUB_TOKEN", "test-token")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": '{"action":"HOLD"}'}}]
    }

    with patch("paper_trader.llm_adapter.requests.post", return_value=mock_resp) as mock_post:
        result = llm_adapter._hf_call("Qwen/Qwen3-32B", "buy or sell?", 30)

    assert result == '{"action":"HOLD"}'
    call_kwargs = mock_post.call_args
    payload = call_kwargs[1]["json"] if "json" in call_kwargs[1] else call_kwargs[0][1]
    assert payload["model"] == "Qwen/Qwen3-32B"
    assert payload["messages"][0]["content"] == "buy or sell?"


def test_hf_call_retries_on_500(monkeypatch):
    from paper_trader import llm_adapter
    monkeypatch.setenv("HUGGINGFACE_HUB_TOKEN", "test-token")

    fail_resp = MagicMock()
    fail_resp.status_code = 500

    ok_resp = MagicMock()
    ok_resp.status_code = 200
    ok_resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}

    with patch("paper_trader.llm_adapter.requests.post", side_effect=[fail_resp, ok_resp]), \
         patch("paper_trader.llm_adapter.time.sleep"):  # don't actually sleep in tests
        result = llm_adapter._hf_call("Qwen/Qwen3-8B", "prompt", 30)

    assert result == "ok"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/zeph/trading-intelligence/paper-trader
python3 -m pytest tests/test_llm_adapter.py -v 2>&1 | head -30
```
Expected: ImportError on `llm_adapter` — module doesn't exist yet.

- [ ] **Step 3: Create `paper_trader/llm_adapter.py`**

```python
"""Unified LLM call adapter for backtest and future callers.

Routes `call_llm(model_id, prompt)` to the appropriate backend:
  - "claude-*"  → Claude CLI subprocess (uses _CLAUDE_SEM, max 2 concurrent)
  - "hf/<org>/<model>" → HuggingFace Inference API (uses _HF_SEM, max 3 concurrent)

The live strategy.py is NOT a caller — it calls Claude directly.
This module owns both concurrency semaphores so they are shared across
all callers that import it.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

import requests

HF_BASE = "https://router.huggingface.co/v1"
HF_TIMEOUT_S = 90
HF_RETRIES = 2
HF_RETRY_BACKOFF_S = 10

_CLAUDE_SEM = threading.Semaphore(2)   # max 2 concurrent claude subprocesses (OOM guard)
_HF_SEM = threading.Semaphore(3)       # max 3 concurrent HF API calls


def call_llm(model_id: str, prompt: str, timeout: int = 90) -> str | None:
    """Route prompt to the right LLM backend. Returns raw response string or None."""
    if model_id.startswith("hf/"):
        return _hf_call(model_id[3:], prompt, timeout)
    elif model_id.startswith("claude-"):
        return _claude_call(model_id, prompt)
    raise ValueError(f"Unknown model_id: {model_id!r}. Must start with 'hf/' or 'claude-'.")


def _claude_call(model_id: str, prompt: str, retries: int = 1) -> str | None:
    if not shutil.which("claude"):
        print("[llm_adapter] claude CLI not found")
        return None
    with _CLAUDE_SEM:
        for attempt in range(retries + 1):
            try:
                r = subprocess.run(
                    ["claude", "--model", model_id, "--print",
                     "--permission-mode", "bypassPermissions"],
                    input=prompt, capture_output=True, text=True,
                    timeout=150,
                )
                if r.returncode == 0 and r.stdout.strip():
                    return r.stdout.strip()
                print(f"[llm_adapter] claude attempt {attempt+1} rc={r.returncode} "
                      f"err={r.stderr.strip()[:200]!r}")
            except subprocess.TimeoutExpired:
                print(f"[llm_adapter] claude timeout attempt {attempt+1}")
            except Exception as e:
                print(f"[llm_adapter] claude exception attempt {attempt+1}: {e}")
            if attempt < retries:
                time.sleep(2)
    return None


def _hf_call(hf_model: str, prompt: str, timeout: int) -> str | None:
    token = _load_hf_token()
    if not token:
        print("[llm_adapter] HF token not found — set HUGGINGFACE_HUB_TOKEN or HF_TOKEN")
        return None
    with _HF_SEM:
        for attempt in range(HF_RETRIES + 1):
            try:
                resp = requests.post(
                    f"{HF_BASE}/chat/completions",
                    headers={"Authorization": f"Bearer {token}"},
                    json={
                        "model": hf_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 512,
                    },
                    timeout=timeout,
                )
                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]["content"]
                print(f"[llm_adapter] HF attempt {attempt+1} status={resp.status_code}")
                if resp.status_code == 429 or resp.status_code >= 500:
                    if attempt < HF_RETRIES:
                        time.sleep(HF_RETRY_BACKOFF_S)
                    continue
                return None  # 4xx (not 429) — don't retry
            except requests.Timeout:
                print(f"[llm_adapter] HF timeout attempt {attempt+1}")
            except Exception as e:
                print(f"[llm_adapter] HF exception attempt {attempt+1}: {e}")
            if attempt < HF_RETRIES:
                time.sleep(HF_RETRY_BACKOFF_S)
    return None


def _load_hf_token() -> str | None:
    """Load HF token from env vars, then fall back to digital-intern .env file."""
    token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if token:
        return token
    env_path = Path("/home/zeph/trading-intelligence/digital-intern/.env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith(("HUGGINGFACE_HUB_TOKEN=", "HF_TOKEN=")):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/zeph/trading-intelligence/paper-trader
python3 -m pytest tests/test_llm_adapter.py -v
```
Expected: all 6 tests PASS.

- [ ] **Step 5: Compile check and commit**

```bash
python3 -m py_compile paper_trader/llm_adapter.py
git add paper_trader/llm_adapter.py tests/test_llm_adapter.py
git commit -m "feat: add llm_adapter — unified Claude/HF call interface"
```

---

## Task 3: Wire adapter into `backtest.py`

**Files:**
- Modify: `paper_trader/backtest.py`

Changes:
1. Import `call_llm` from `llm_adapter`
2. Delete the `_CLAUDE_SEM` module-level definition (now lives in `llm_adapter`)
3. Replace `_claude_call` body with a delegation to `llm_adapter.call_llm`
4. Add `model_id` param to `BacktestEngine.__init__`
5. Update `BacktestStore.upsert_run` to accept and persist `model_id`
6. Wire `model_id` through `BacktestEngine.run_one` → `store.upsert_run` + decision dispatch

- [ ] **Step 1: Write failing integration test** (append to `tests/test_model_rankings.py`)

```python
from unittest.mock import patch
from datetime import date

def test_backtest_engine_stores_model_id(tmp_path):
    """BacktestEngine with model_id='test-model' stores that value in backtest_runs."""
    import paper_trader.backtest as bt
    bt.BACKTEST_DB = tmp_path / "test.db"  # redirect to tmp

    engine = bt.BacktestEngine(
        start=date(2024, 1, 2),
        end=date(2024, 1, 5),
        model_id="claude-opus-4-7",
    )
    # Patch _claude_call to avoid real subprocess
    with patch("paper_trader.llm_adapter._claude_call", return_value=None):
        result = engine.run_one(run_id=1, seed=42)

    row = engine.store.conn.execute(
        "SELECT model_id FROM backtest_runs WHERE run_id=1"
    ).fetchone()
    assert row[0] == "claude-opus-4-7"
```

- [ ] **Step 2: Run to verify it fails**

```bash
cd /home/zeph/trading-intelligence/paper-trader
python3 -m pytest tests/test_model_rankings.py::test_backtest_engine_stores_model_id -v 2>&1 | head -20
```
Expected: FAIL — `BacktestEngine` doesn't accept `model_id` yet.

- [ ] **Step 3: Add import and remove orphaned `_CLAUDE_SEM`**

At the top of `paper_trader/backtest.py`, add after other imports:

```python
from paper_trader.llm_adapter import call_llm as _llm_call, _CLAUDE_SEM  # noqa: F401 – kept for any external callers
```

Then find and remove the line:
```python
_CLAUDE_SEM = threading.Semaphore(2)
```
(at line ~51 — the one that will now be a duplicate; keep only the import above).

- [ ] **Step 4: Replace `_claude_call` with a thin delegation**

Find `def _claude_call(prompt: str, retries: int = 1) -> str | None:` (~line 1918) and replace the entire function body:

```python
def _claude_call(prompt: str, retries: int = 1) -> str | None:
    from paper_trader.llm_adapter import call_llm
    return call_llm(MODEL, prompt)
```

- [ ] **Step 5: Add `model_id` to `BacktestEngine.__init__`**

Find `class BacktestEngine:` and its `__init__` (~line 2095). Change the signature:

```python
def __init__(self, start: date | None = None, end: date | None = None,
             model_id: str = "ml_quant") -> None:
```

Add after `self.store = BacktestStore()`:
```python
self.model_id = model_id
_VALID_PREFIXES = ("ml_quant", "claude-", "hf/")
if not any(model_id.startswith(p) for p in _VALID_PREFIXES):
    raise ValueError(f"Invalid model_id {model_id!r}. Must start with one of {_VALID_PREFIXES}")
```

- [ ] **Step 6: Update `BacktestStore.upsert_run` to accept `model_id`**

Change the signature of `upsert_run`:
```python
def upsert_run(self, run_id: int, seed: int, status: str,
               start: date, end: date, model_id: str = "ml_quant") -> None:
```

In the INSERT branch, add `model_id` to the column list and values:
```python
self.conn.execute(
    "INSERT INTO backtest_runs (run_id, seed, start_date, end_date, "
    "start_value, status, started_at, model_id) VALUES (?,?,?,?,?,?,?,?)",
    (run_id, seed, start.isoformat(), end.isoformat(),
     INITIAL_CASH, status, now, model_id),
)
```

- [ ] **Step 7: Pass `model_id` through `run_one`**

In `BacktestEngine.run_one`, change the `self.store.upsert_run` call:
```python
self.store.upsert_run(run_id, seed, "running",
                      start=self.start, end=self.end,
                      model_id=self.model_id)
```

In the decision dispatch loop (~line 2494), change `_ml_decide` to a branch:
```python
if self.model_id == "ml_quant":
    decision = _ml_decide(sim_date, portfolio, signals, self.prices,
                          run_id, rng, exclude_tickers=traded_today)
else:
    # LLM-based decision — build prompt and call via adapter
    prompt = _build_llm_prompt(sim_date, portfolio, signals, self.prices, rng)
    raw = _llm_call(self.model_id, prompt)
    decision = _parse_decision(raw)
```

> **Note on `_build_llm_prompt`**: The existing backtest already builds a prompt string and passes it to `_claude_call` in a different path (the `run_claude_backtest` function if it exists — check the file). If there's no standalone `_build_llm_prompt`, extract it from the Claude path or write a minimal one that assembles the same signals dict into the prompt string already used by `strategy.py`. See `strategy._build_prompt` or the nearest equivalent in backtest.py.

- [ ] **Step 8: Run tests**

```bash
cd /home/zeph/trading-intelligence/paper-trader
python3 -m pytest tests/test_model_rankings.py -v
python3 -m py_compile paper_trader/backtest.py
```
Expected: all tests PASS, no compile errors.

- [ ] **Step 9: Smoke test the ml_quant path (no regressions)**

```bash
cd /home/zeph/trading-intelligence/paper-trader
python3 -c "
from datetime import date
from paper_trader.backtest import BacktestEngine
e = BacktestEngine(start=date(2024, 1, 2), end=date(2024, 1, 5))
r = e.run_one(run_id=9999, seed=1)
print('status:', r.status, 'model_id in DB:', e.store.conn.execute('SELECT model_id FROM backtest_runs WHERE run_id=9999').fetchone())
e.store.conn.execute('DELETE FROM backtest_runs WHERE run_id=9999')
e.store.conn.commit()
"
```
Expected: `status: complete  model_id in DB: ('ml_quant',)`

- [ ] **Step 10: Commit**

```bash
git add paper_trader/backtest.py tests/test_model_rankings.py
git commit -m "feat(backtest): wire llm_adapter, add model_id to BacktestEngine + store"
```

---

## Task 4: `/api/model-rankings` dashboard endpoint

**Files:**
- Modify: `paper_trader/dashboard.py` (add route + append to test file)

- [ ] **Step 1: Write failing test** (append to `tests/test_model_rankings.py`)

```python
import json

def test_model_rankings_api(tmp_path):
    """GET /api/model-rankings returns correct aggregated stats per model."""
    import paper_trader.backtest as bt
    bt.BACKTEST_DB = tmp_path / "bt.db"
    store = bt.BacktestStore(path=tmp_path / "bt.db")

    # Insert two complete runs with different model_ids
    store.conn.execute(
        "INSERT INTO backtest_runs (run_id, seed, start_date, end_date, start_value, "
        "final_value, total_return_pct, spy_return_pct, vs_spy_pct, n_trades, n_decisions, "
        "status, started_at, model_id) VALUES (1, 1, '2025-01-01', '2026-01-01', 1000, "
        "1200, 20.0, 10.0, 10.0, 50, 300, 'complete', '2026-01-01T00:00:00Z', 'ml_quant')"
    )
    store.conn.execute(
        "INSERT INTO backtest_runs (run_id, seed, start_date, end_date, start_value, "
        "final_value, total_return_pct, spy_return_pct, vs_spy_pct, n_trades, n_decisions, "
        "status, started_at, model_id) VALUES (2, 2, '2025-01-01', '2026-01-01', 1000, "
        "1500, 50.0, 10.0, 40.0, 80, 250, 'complete', '2026-01-01T00:00:00Z', 'hf/deepseek-ai/DeepSeek-R1')"
    )
    store.conn.commit()

    import paper_trader.dashboard as dash
    dash.BACKTEST_DB = tmp_path / "bt.db"

    client = dash.app.test_client()
    resp = client.get("/api/model-rankings")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert "models" in data
    models = {m["model_id"]: m for m in data["models"]}
    assert "ml_quant" in models
    assert "hf/deepseek-ai/DeepSeek-R1" in models
    assert models["ml_quant"]["avg_return_pct"] == pytest.approx(20.0)
    assert models["hf/deepseek-ai/DeepSeek-R1"]["avg_return_pct"] == pytest.approx(50.0)
```

- [ ] **Step 2: Run to verify it fails**

```bash
cd /home/zeph/trading-intelligence/paper-trader
python3 -m pytest tests/test_model_rankings.py::test_model_rankings_api -v 2>&1 | head -20
```
Expected: FAIL — route doesn't exist yet.

- [ ] **Step 3: Add `BACKTEST_DB` module-level reference and `/api/model-rankings` route to `dashboard.py`**

Find the import section at the top of `paper_trader/dashboard.py` where `BACKTEST_DB` is already used. Make sure `BACKTEST_DB` is accessible as a module-level name (it likely already is — grep for it).

Add the route after `/api/backtests/stats` (~line 6320):

```python
_MODEL_DISPLAY_NAMES = {
    "ml_quant": "ML+Quant (deterministic)",
    "claude-opus-4-7": "Claude Opus 4.7",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "hf/deepseek-ai/DeepSeek-R1": "DeepSeek R1",
    "hf/deepseek-ai/DeepSeek-V3.2": "DeepSeek V3.2",
    "hf/meta-llama/Llama-3.3-70B-Instruct": "Llama 3.3 70B",
    "hf/Qwen/Qwen3-32B": "Qwen3 32B",
    "hf/Qwen/Qwen3-8B": "Qwen3 8B",
}


@app.route("/api/model-rankings")
def api_model_rankings():
    try:
        conn = sqlite3.connect(str(BACKTEST_DB), timeout=10)
        rows = conn.execute("""
            SELECT
                model_id,
                COUNT(*) AS runs,
                ROUND(AVG(total_return_pct), 2) AS avg_return_pct,
                ROUND(MAX(total_return_pct), 2) AS best_return_pct,
                ROUND(AVG(vs_spy_pct), 2) AS avg_vs_spy_pct,
                ROUND(AVG(n_trades), 1) AS avg_trades,
                ROUND(
                    100.0 * SUM(CASE WHEN total_return_pct > 0 THEN 1 ELSE 0 END)
                    / COUNT(*), 1
                ) AS win_rate_pct,
                SUM(n_decisions) AS total_decisions
            FROM backtest_runs
            WHERE status = 'complete'
            GROUP BY model_id
            ORDER BY avg_return_pct DESC
        """).fetchall()
        conn.close()
        models = []
        for r in rows:
            mid = r[0] or "ml_quant"
            models.append({
                "model_id": mid,
                "display_name": _MODEL_DISPLAY_NAMES.get(mid, mid),
                "runs": r[1],
                "avg_return_pct": r[2],
                "best_return_pct": r[3],
                "avg_vs_spy_pct": r[4],
                "avg_trades": r[5],
                "win_rate_pct": r[6],
                "total_decisions": r[7],
            })
        return jsonify({"models": models, "as_of": datetime.now(timezone.utc).isoformat()})
    except Exception as e:
        return jsonify({"error": str(e), "models": []}), 500
```

- [ ] **Step 4: Run tests**

```bash
cd /home/zeph/trading-intelligence/paper-trader
python3 -m pytest tests/test_model_rankings.py -v
python3 -m py_compile paper_trader/dashboard.py
```
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add paper_trader/dashboard.py tests/test_model_rankings.py
git commit -m "feat(dashboard): add /api/model-rankings endpoint"
```

---

## Task 5: Model Rankings UI tab

**Files:**
- Modify: `paper_trader/dashboard.py` (add HTML tab + JS fetch in the backtests page section)

The backtests page is a large inline HTML template inside `dashboard.py`. Find the tab navigation for `/backtests` (around the `@app.route("/backtests")` handler, ~line 5733) and add a new tab.

- [ ] **Step 1: Find the tab nav in the backtests page**

```bash
grep -n "tab\|Tab\|nav.*back\|backtest.*nav\|<button.*tab" /home/zeph/trading-intelligence/paper-trader/paper_trader/dashboard.py | grep -A2 -B2 "5[789][0-9][0-9]\|6[01][0-9][0-9]" | head -20
```

Identify the existing tab IDs (e.g., `runs-tab`, `leaderboard-tab`, etc.) and the `<div class="tab-content">` structure.

- [ ] **Step 2: Add the "Model Rankings" tab button to the tab nav**

In the tab nav HTML block, add:
```html
<button class="tab-btn" onclick="switchTab('model-rankings')" id="tab-btn-model-rankings">
  🏆 Model Rankings
</button>
```

- [ ] **Step 3: Add the tab content panel**

In the tab content area, add:
```html
<div class="tab-panel" id="tab-model-rankings" style="display:none">
  <div id="model-rankings-loading" style="padding:20px;color:#888">Loading rankings…</div>
  <table id="model-rankings-table" style="display:none;width:100%;border-collapse:collapse">
    <thead>
      <tr id="model-rankings-header"></tr>
    </thead>
    <tbody id="model-rankings-body"></tbody>
  </table>
  <div style="margin-top:16px">
    <label>Run backtest with model:
      <select id="run-model-select">
        <option value="ml_quant">ML+Quant (deterministic)</option>
        <option value="claude-opus-4-7">Claude Opus 4.7</option>
        <option value="hf/deepseek-ai/DeepSeek-R1">DeepSeek R1</option>
        <option value="hf/meta-llama/Llama-3.3-70B-Instruct">Llama 3.3 70B</option>
        <option value="hf/Qwen/Qwen3-32B">Qwen3 32B</option>
      </select>
    </label>
    <button onclick="triggerBacktestWithModel()">▶ Run Backtest</button>
    <span id="run-model-status" style="margin-left:12px;color:#888"></span>
  </div>
</div>
```

- [ ] **Step 4: Add the JavaScript fetch + render logic**

In the `<script>` section of the backtests page, add:

```javascript
async function loadModelRankings() {
  try {
    const data = await fetch(API_PREFIX + '/api/model-rankings').then(r => r.json());
    const medals = ['🥇', '🥈', '🥉'];
    const cols = [
      ['Rank', m => ''], // filled below
      ['Model', m => m.display_name],
      ['Runs', m => m.runs],
      ['Avg Return', m => (m.avg_return_pct > 0 ? '+' : '') + m.avg_return_pct + '%'],
      ['Best Return', m => (m.best_return_pct > 0 ? '+' : '') + m.best_return_pct + '%'],
      ['vs SPY', m => (m.avg_vs_spy_pct > 0 ? '+' : '') + m.avg_vs_spy_pct + '%'],
      ['Win Rate', m => m.win_rate_pct + '%'],
      ['Avg Trades', m => m.avg_trades],
    ];
    const header = document.getElementById('model-rankings-header');
    header.innerHTML = cols.map(([h]) => `<th style="text-align:left;padding:8px 12px;border-bottom:1px solid #333">${h}</th>`).join('');
    const body = document.getElementById('model-rankings-body');
    body.innerHTML = data.models.map((m, i) => {
      const medal = medals[i] || (i + 1) + '.';
      const cells = cols.map(([h, fn], ci) => {
        const val = ci === 0 ? medal : fn(m);
        const color = ci === 3 || ci === 4 || ci === 5
          ? (parseFloat(String(val)) > 0 ? '#4caf50' : parseFloat(String(val)) < 0 ? '#f44336' : '')
          : '';
        return `<td style="padding:8px 12px;border-bottom:1px solid #222;${color ? 'color:' + color : ''}">${val}</td>`;
      }).join('');
      return `<tr style="cursor:pointer" onclick="filterRunsByModel('${m.model_id}')">${cells}</tr>`;
    }).join('');
    document.getElementById('model-rankings-loading').style.display = 'none';
    document.getElementById('model-rankings-table').style.display = 'table';
  } catch(e) {
    document.getElementById('model-rankings-loading').textContent = 'Failed to load rankings: ' + e.message;
  }
}

function triggerBacktestWithModel() {
  const model = document.getElementById('run-model-select').value;
  document.getElementById('run-model-status').textContent = 'Use run_continuous_backtests.py --model ' + model + ' to start a run.';
}

// Load when tab is activated
const _origSwitchTab = typeof switchTab === 'function' ? switchTab : null;
function switchTab(name) {
  if (_origSwitchTab) _origSwitchTab(name);
  if (name === 'model-rankings') loadModelRankings();
}
```

> **Note:** The existing `switchTab` function structure varies. If `switchTab` is already defined differently in the page, adapt the hook to call `loadModelRankings()` when `name === 'model-rankings'` without breaking existing tabs. Check the existing `switchTab` implementation and add the conditional inside it instead of wrapping.

- [ ] **Step 5: Compile check**

```bash
python3 -m py_compile paper_trader/dashboard.py && echo OK
```

- [ ] **Step 6: Manual smoke test**

```bash
# Restart the paper-trader dashboard
systemctl --user restart paper-trader
sleep 5
curl -s http://127.0.0.1:8090/api/model-rankings | python3 -m json.tool | head -30
```
Expected: JSON with `"models"` array, `ml_quant` entry with 501 runs.

- [ ] **Step 7: Commit**

```bash
git add paper_trader/dashboard.py
git commit -m "feat(dashboard): add Model Rankings tab to /backtests page"
git push
```

---

## Task 6: Enable HF plugin + verify HF token

**Files:**
- Modify: `~/.openclaw/openclaw.json` (enable HF plugin)
- Verify: `.env` file has `HUGGINGFACE_HUB_TOKEN`

- [ ] **Step 1: Enable the HF plugin**

```bash
python3 -c "
import json
path = '/home/zeph/.openclaw/openclaw.json'
with open(path) as f:
    d = json.load(f)
d.setdefault('plugins', {}).setdefault('entries', {})['huggingface'] = {'enabled': True, 'config': {}}
with open(path, 'w') as f:
    json.dump(d, f, indent=2)
print('HF plugin enabled')
"
```

- [ ] **Step 2: Verify HF token is available**

```bash
python3 -c "
import sys; sys.path.insert(0, '/home/zeph/trading-intelligence/paper-trader')
from paper_trader.llm_adapter import _load_hf_token
t = _load_hf_token()
print('Token found:', bool(t), '| prefix:', (t or '')[:8] + '...' if t else 'MISSING')
"
```

If token is MISSING, add it to `/home/zeph/trading-intelligence/digital-intern/.env`:
```
HUGGINGFACE_HUB_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```
(Get from https://huggingface.co/settings/tokens — needs "Make calls to Inference Providers" permission)

- [ ] **Step 3: Smoke test a live HF API call**

```bash
python3 -c "
import sys; sys.path.insert(0, '/home/zeph/trading-intelligence/paper-trader')
from paper_trader.llm_adapter import _hf_call
result = _hf_call('Qwen/Qwen3-8B', 'Reply with only the word: READY', timeout=30)
print('HF response:', result)
"
```
Expected: something containing "READY" or a short response. If 401, check token permissions.

- [ ] **Step 4: Commit openclaw.json change**

```bash
git add ~/.openclaw/openclaw.json 2>/dev/null || true
# openclaw.json is not in the paper-trader repo — no commit needed for it
echo "HF plugin enabled in openclaw.json — no git commit needed (not in repo)"
```

---

## Task 7: Run first HF backtest and verify rankings page

- [ ] **Step 1: Run a short HF backtest (1 run, 5 days)**

```bash
cd /home/zeph/trading-intelligence/paper-trader
python3 -c "
from datetime import date
from paper_trader.backtest import BacktestEngine
engine = BacktestEngine(
    start=date(2025, 6, 1),
    end=date(2025, 6, 10),
    model_id='hf/Qwen/Qwen3-8B',
)
result = engine.run_one(run_id=10001, seed=42)
print(f'Run complete: {result.status} | return: {result.total_return_pct:.1f}%')
row = engine.store.conn.execute('SELECT model_id, total_return_pct FROM backtest_runs WHERE run_id=10001').fetchone()
print(f'DB record: model_id={row[0]} return={row[1]:.1f}%')
engine.store.conn.execute('DELETE FROM backtest_runs WHERE run_id=10001')
engine.store.conn.commit()
"
```
Expected: `Run complete: complete | return: X%` and `DB record: model_id=hf/Qwen/Qwen3-8B`.

- [ ] **Step 2: Check rankings endpoint shows both models**

```bash
curl -s http://127.0.0.1:8090/api/model-rankings | python3 -c "
import json, sys
d = json.load(sys.stdin)
for m in d['models']:
    print(f\"{m['model_id']}: {m['runs']} runs, avg {m['avg_return_pct']}%\")
"
```
Expected: `ml_quant: 501 runs, ...` plus the new HF run entry.

- [ ] **Step 3: Push final state**

```bash
cd /home/zeph/trading-intelligence/paper-trader
git push
```

---

## Self-Review Checklist

- [x] **Spec coverage:** All spec sections mapped to tasks
  - `llm_adapter.py` → Task 2
  - Schema migration → Task 1
  - `BacktestEngine` model_id → Task 3
  - `/api/model-rankings` → Task 4
  - Rankings tab UI → Task 5
  - HF plugin enable → Task 6
  - Error handling (token missing → None, 429 retry, unknown model → ValueError) → Task 2 Step 3
  - Tests → Tasks 1, 2, 4
- [x] **No placeholders:** All steps have concrete code
- [x] **Type consistency:** `call_llm(model_id: str, prompt: str)` used uniformly; `_hf_call(hf_model: str, ...)` strips prefix before call
- [x] **`_CLAUDE_SEM` migration:** Explicit note in Task 3 Step 3 to remove orphaned definition from backtest.py
- [x] **`_build_llm_prompt` gap:** Noted in Task 3 Step 7 — engineer must locate or extract the existing prompt builder
