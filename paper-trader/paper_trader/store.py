"""SQLite store for paper trading portfolio, trades, positions, and decisions."""
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "paper_trader.db"
INITIAL_CASH = 1000.0

SCHEMA = """
CREATE TABLE IF NOT EXISTS portfolio (
    id              INTEGER PRIMARY KEY CHECK (id = 1),
    cash            REAL NOT NULL,
    total_value     REAL NOT NULL,
    positions_json  TEXT NOT NULL DEFAULT '[]',
    last_updated    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS trades (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT NOT NULL,
    ticker      TEXT NOT NULL,
    action      TEXT NOT NULL,
    qty         REAL NOT NULL,
    price       REAL NOT NULL,
    value       REAL NOT NULL,
    reason      TEXT,
    expiry      TEXT,
    strike      REAL,
    option_type TEXT
);
CREATE INDEX IF NOT EXISTS idx_trades_ts ON trades(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_trades_ticker ON trades(ticker);

CREATE TABLE IF NOT EXISTS positions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker            TEXT NOT NULL,
    type              TEXT NOT NULL,
    qty               REAL NOT NULL,
    avg_cost          REAL NOT NULL,
    current_price     REAL DEFAULT 0,
    unrealized_pl     REAL DEFAULT 0,
    expiry            TEXT,
    strike            REAL,
    opened_at         TEXT NOT NULL,
    closed_at         TEXT,
    stop_loss_price   REAL,
    take_profit_price REAL,
    UNIQUE(ticker, type, expiry, strike)
);
CREATE INDEX IF NOT EXISTS idx_pos_open ON positions(closed_at) WHERE closed_at IS NULL;

CREATE TABLE IF NOT EXISTS decisions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,
    market_open     INTEGER NOT NULL,
    signal_count    INTEGER NOT NULL,
    action_taken    TEXT,
    reasoning       TEXT,
    portfolio_value REAL,
    cash            REAL
);
CREATE INDEX IF NOT EXISTS idx_dec_ts ON decisions(timestamp DESC);

CREATE TABLE IF NOT EXISTS equity_curve (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,
    total_value     REAL NOT NULL,
    cash            REAL NOT NULL,
    sp500_price     REAL
);
CREATE INDEX IF NOT EXISTS idx_eq_ts ON equity_curve(timestamp DESC);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hold_duration(opened_at: str | None, closed_at: str | None
                   ) -> tuple[int | None, float | None]:
    """Parse an opened_at / closed_at pair into ``(hold_seconds, hold_days)``.

    Returns ``(None, None)`` when either side is missing or unparseable; a
    negative span (closed_at strictly before opened_at — non-physical, would
    only happen on a wall-clock step-back) clamps to zero so the hold-time
    field never renders negative. ``hold_days`` is rounded to 4 places (the
    round_trips.py precedent) so callers can compute hourly rates without
    surprise float precision; ``hold_seconds`` stays an int for the test path
    that wants exact arithmetic on the per-second grain.
    """
    if not opened_at or not closed_at:
        return None, None
    try:
        op = datetime.fromisoformat(str(opened_at).replace("Z", "+00:00"))
        cl = datetime.fromisoformat(str(closed_at).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None, None
    secs = (cl - op).total_seconds()
    if secs < 0:
        secs = 0.0
    return int(secs), round(secs / 86400.0, 4)


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


class Store:
    def __init__(self):
        self.conn = _connect()
        self._lock = threading.Lock()
        self._init_portfolio()
        # Safe column migration — idempotent. SCHEMA above uses
        # CREATE TABLE IF NOT EXISTS, which does NOT add new columns to an
        # existing positions table. ALTER TABLE on every startup; swallow the
        # "duplicate column name" error once the column already exists.
        for col, sql_type in (
            ("stop_loss_price", "REAL"),
            ("take_profit_price", "REAL"),
        ):
            try:
                with self._lock:
                    self.conn.execute(
                        f"ALTER TABLE positions ADD COLUMN {col} {sql_type}"
                    )
                    self.conn.commit()
            except sqlite3.OperationalError:
                pass  # column already exists — fresh DB or prior boot ran the ALTER

    def _init_portfolio(self):
        with self._lock:
            row = self.conn.execute("SELECT id FROM portfolio WHERE id=1").fetchone()
            if not row:
                self.conn.execute(
                    "INSERT INTO portfolio (id, cash, total_value, positions_json, last_updated) "
                    "VALUES (1, ?, ?, '[]', ?)",
                    (INITIAL_CASH, INITIAL_CASH, _now()),
                )
                self.conn.commit()

    # ─── portfolio ─────────────────────────────────────────────
    # NOTE: every read below MUST hold self._lock. The connection is created
    # with check_same_thread=False and shared between the runner's writer
    # thread and the Flask dashboard thread(s). All writes already serialize
    # on self._lock; an unlocked read whose execute() interleaves with a
    # concurrent write on the same connection raises
    # `sqlite3.InterfaceError: bad parameter or other API misuse` or returns a
    # corrupted/None row (the dashboard /api/state 500s in runner.log).
    def get_portfolio(self) -> dict:
        with self._lock:
            row = self.conn.execute(
                "SELECT cash, total_value, positions_json, last_updated FROM portfolio WHERE id=1"
            ).fetchone()
        # The shared connection (see the note above) intermittently hands back
        # a None row OR a present row whose columns read back NULL even though
        # the stored values are well-formed. get_portfolio() backs /api/state,
        # polled every few seconds — a raw subscript / json.loads(None) here
        # 500s the whole dashboard (28x in runner.log over 2 days). Worse: a
        # NULL `cash`/`total_value` flows into strategy._portfolio_snapshot as
        # `None + open_value`, a TypeError that aborts the WHOLE decide() cycle
        # (no decision row, no equity point). Both modes must be absorbed; the
        # original code only handled `row is None` and silently dropped the
        # equally-documented NULL-column mode straight into the live loop.
        def _bad(r) -> bool:
            # `positions_json` is recovered separately below (NULL → "[]"); a
            # NULL `last_updated` is cosmetic. Only a missing row or a NULL
            # numeric cash/total_value is unrecoverable as-is.
            return (r is None
                    or r["cash"] is None
                    or r["total_value"] is None)

        if _bad(row):
            # Either a transient corrupt read (the common case — the re-read
            # below recovers the real well-formed values) or the id=1 row is
            # genuinely gone. _init_portfolio() re-INSERTs only when the row is
            # missing (a live book whose read merely came back NULL is never
            # reset) and takes self._lock itself, so it must be called WITHOUT
            # the lock held. One re-read.
            why = "row=None" if row is None else "NULL cash/total_value"
            print(f"[store] get_portfolio: bad row ({why}) — re-reading",
                  flush=True)
            self._init_portfolio()
            with self._lock:
                row = self.conn.execute(
                    "SELECT cash, total_value, positions_json, last_updated "
                    "FROM portfolio WHERE id=1"
                ).fetchone()
            if _bad(row):
                return {"cash": INITIAL_CASH, "total_value": INITIAL_CASH,
                        "positions": [], "last_updated": _now()}
        pj = row["positions_json"]
        if not pj:
            print("[store] get_portfolio: positions_json empty/NULL — defaulting to []",
                  flush=True)
            pj = "[]"
        return {
            "cash": row["cash"],
            "total_value": row["total_value"],
            "positions": json.loads(pj),
            "last_updated": row["last_updated"],
        }

    def update_portfolio(self, cash: float, total_value: float,
                         positions: list | None = None):
        """``positions=None`` writes ONLY cash + total_value and preserves the
        existing ``positions_json`` column. Used by ``strategy._execute`` mid-
        cycle, where the post-trade enriched position blend is computed by the
        end-of-cycle ``_portfolio_snapshot`` re-mark; writing the pre-trade
        snapshot list here would briefly desync ``portfolio.positions_json``
        from the underlying ``positions`` table (a dashboard /api/portfolio
        poll in that window saw the new lot's cash impact but the pre-trade
        position list — missing the just-bought lot, or still showing the
        just-sold one). Default ``None`` is back-compat for fresh callers."""
        with self._lock:
            if positions is None:
                self.conn.execute(
                    "UPDATE portfolio SET cash=?, total_value=?, last_updated=? WHERE id=1",
                    (cash, total_value, _now()),
                )
            else:
                self.conn.execute(
                    "UPDATE portfolio SET cash=?, total_value=?, positions_json=?, last_updated=? WHERE id=1",
                    (cash, total_value, json.dumps(positions), _now()),
                )
            self.conn.commit()

    # ─── trades ────────────────────────────────────────────────
    def record_trade(self, ticker: str, action: str, qty: float, price: float,
                     reason: str = "", expiry: str | None = None,
                     strike: float | None = None, option_type: str | None = None) -> int:
        value = qty * price * (100 if option_type in ("call", "put") else 1)
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO trades (timestamp, ticker, action, qty, price, value, reason, expiry, strike, option_type) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (_now(), ticker, action, qty, price, value, reason, expiry, strike, option_type),
            )
            self.conn.commit()
            return cur.lastrowid

    def recent_trades(self, limit: int = 50) -> list[dict]:
        # Tie-break on the autoincrement id so rows that share a timestamp
        # (two writes inside the same microsecond) still come back newest-first
        # deterministically. Without this, recent_trades(1) — used by
        # runner._cycle/send_trade_alert right after _execute records the
        # trade — could surface a stale same-microsecond row.
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM trades ORDER BY timestamp DESC, id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ─── positions ─────────────────────────────────────────────
    def upsert_position(self, ticker: str, type_: str, qty: float, avg_cost: float,
                        expiry: str | None = None, strike: float | None = None,
                        stop_loss_price: float | None = None,
                        take_profit_price: float | None = None) -> None:
        """Apply a position delta. ``qty=0`` is the metadata-only mode used by
        the BUY path to set ``stop_loss_price`` / ``take_profit_price`` on an
        existing open lot without changing its size — only fields explicitly
        passed (non-None) are written; everything else is preserved."""
        with self._lock:
            existing = self.conn.execute(
                "SELECT id, qty, avg_cost FROM positions "
                "WHERE ticker=? AND type=? AND IFNULL(expiry,'')=IFNULL(?,'') "
                "AND IFNULL(strike,0)=IFNULL(?,0) AND closed_at IS NULL",
                (ticker, type_, expiry, strike),
            ).fetchone()
            if existing:
                new_qty = existing["qty"] + qty
                if new_qty <= 0.0001:
                    self.conn.execute(
                        "UPDATE positions SET qty=0, closed_at=? WHERE id=?",
                        (_now(), existing["id"]),
                    )
                else:
                    blended = (existing["qty"] * existing["avg_cost"] + qty * avg_cost) / new_qty if qty > 0 else existing["avg_cost"]
                    # qty=0 is the metadata-only path — preserve qty/avg_cost.
                    if qty == 0:
                        # Only touch SL/TP when explicitly supplied; this is
                        # the post-BUY "stamp the hard exits on the just-
                        # opened lot" call from strategy._execute.
                        set_clauses = []
                        params: list = []
                        if stop_loss_price is not None:
                            set_clauses.append("stop_loss_price=?")
                            params.append(stop_loss_price)
                        if take_profit_price is not None:
                            set_clauses.append("take_profit_price=?")
                            params.append(take_profit_price)
                        if set_clauses:
                            params.append(existing["id"])
                            self.conn.execute(
                                f"UPDATE positions SET {', '.join(set_clauses)} WHERE id=?",
                                params,
                            )
                    else:
                        set_clauses = ["qty=?", "avg_cost=?"]
                        params = [new_qty, blended]
                        if stop_loss_price is not None:
                            set_clauses.append("stop_loss_price=?")
                            params.append(stop_loss_price)
                        if take_profit_price is not None:
                            set_clauses.append("take_profit_price=?")
                            params.append(take_profit_price)
                        params.append(existing["id"])
                        self.conn.execute(
                            f"UPDATE positions SET {', '.join(set_clauses)} WHERE id=?",
                            params,
                        )
            else:
                if qty > 0:
                    # No open lot for this key. A prior fully-closed lot with
                    # the SAME (ticker,type,expiry,strike) still occupies its
                    # row, and the table-wide UNIQUE(ticker,type,expiry,strike)
                    # constraint makes a plain INSERT raise IntegrityError when
                    # strike/expiry are non-NULL — i.e. every option re-entry
                    # after a full close crashed the cycle (stock NULLs are
                    # distinct so it only bit options). Reactivate the closed
                    # row instead so a re-entry after a full close just works.
                    closed = self.conn.execute(
                        "SELECT id FROM positions "
                        "WHERE ticker=? AND type=? AND IFNULL(expiry,'')=IFNULL(?,'') "
                        "AND IFNULL(strike,0)=IFNULL(?,0) AND closed_at IS NOT NULL "
                        "ORDER BY id DESC LIMIT 1",
                        (ticker, type_, expiry, strike),
                    ).fetchone()
                    if closed:
                        self.conn.execute(
                            "UPDATE positions SET qty=?, avg_cost=?, current_price=0, "
                            "unrealized_pl=0, opened_at=?, closed_at=NULL, "
                            "stop_loss_price=?, take_profit_price=? WHERE id=?",
                            (qty, avg_cost, _now(), stop_loss_price,
                             take_profit_price, closed["id"]),
                        )
                    else:
                        self.conn.execute(
                            "INSERT INTO positions (ticker, type, qty, avg_cost, expiry, strike, opened_at, stop_loss_price, take_profit_price) "
                            "VALUES (?,?,?,?,?,?,?,?,?)",
                            (ticker, type_, qty, avg_cost, expiry, strike,
                             _now(), stop_loss_price, take_profit_price),
                        )
            self.conn.commit()

    def positions_needing_hard_exit(self) -> list[dict]:
        """Open stock positions whose current_price breaches SL or TP.

        Read-only; pure SQL on the positions table. Skips options (only stock
        type is hard-exited), zero-qty lots, lots without SL/TP set, and lots
        with no fresh mark (current_price > 0 is the snapshot freshness
        proxy). Caller is responsible for executing the SELL."""
        with self._lock:
            cur = self.conn.execute(
                "SELECT * FROM positions WHERE closed_at IS NULL AND type='stock' "
                "AND qty > 0 AND stop_loss_price IS NOT NULL "
                "AND take_profit_price IS NOT NULL AND current_price > 0 "
                "AND (current_price <= stop_loss_price OR current_price >= take_profit_price)"
            )
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    def close_position(self, position_id: int):
        with self._lock:
            self.conn.execute(
                "UPDATE positions SET qty=0, closed_at=? WHERE id=?",
                (_now(), position_id),
            )
            self.conn.commit()

    def open_positions(self) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM positions WHERE closed_at IS NULL AND qty > 0 ORDER BY opened_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def closed_positions(self, limit: int = 100) -> list[dict]:
        """Closed position lots with realized P&L computed from the round-trip
        of trades that ran on the same (ticker,type,expiry,strike) key.

        Walks every trade on the key chronologically and picks out the
        round-trip whose final close lands at or before this lot's
        ``closed_at``: held qty starts at 0, BUYs add, SELLs subtract, and
        every time held returns to ≈0 a round-trip closes. We pick the
        round-trip whose closing trade matches the lot's ``closed_at`` (or
        the latest one ≤ ``closed_at`` when the timestamps don't byte-match).

        The previous implementation used a naive ``timestamp >= opened_at``
        SQL window. ``record_trade`` runs *before* ``upsert_position`` in the
        live trader (see strategy._execute), so the opening BUY's
        ``timestamp`` lands a few microseconds BEFORE the position row's
        ``opened_at`` — observed live: 290µs–3ms gap. The window therefore
        skipped every opening BUY and ``realized_pl`` reported only the
        SELL proceeds, hugely overstating every closed lot's P/L on
        /api/closed-positions. The same flaw made the BUY_CALL/SELL_CALL fix
        invisible (the option BUY was missed even when matched correctly).

        ``startswith`` matches every documented entry/exit action the live
        trader writes (BUY, BUY_CALL, BUY_PUT and SELL, SELL_CALL, SELL_PUT);
        any non-BUY/SELL action (REBALANCE, OPEN, etc.) is ignored.

        Returns newest-closed first.
        """
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM positions WHERE closed_at IS NOT NULL "
                "ORDER BY closed_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            out: list[dict] = []
            for r in rows:
                d = dict(r)
                ptype = (d.get("type") or "").lower()
                opt = ptype if ptype in ("call", "put") else None
                all_trades = self.conn.execute(
                    "SELECT timestamp, action, value, qty FROM trades "
                    "WHERE ticker=? AND IFNULL(option_type,'')=IFNULL(?,'') "
                    "AND IFNULL(expiry,'')=IFNULL(?,'') "
                    "AND IFNULL(strike,0)=IFNULL(?,0) "
                    "ORDER BY timestamp ASC, id ASC",
                    (d["ticker"], opt, d.get("expiry"), d.get("strike")),
                ).fetchall()
                closed_at = d["closed_at"]
                # Identify the round-trip that ends at this lot's closed_at.
                # Walk every trade for the key; whenever held returns to ≈0
                # we have a completed round-trip slice. Pick the slice whose
                # closing-trade timestamp is the latest one ≤ closed_at.
                held = 0.0
                start_idx = 0
                round_trips: list[tuple[int, int, str]] = []
                for i, t in enumerate(all_trades):
                    act = (t["action"] or "").upper()
                    try:
                        q = float(t["qty"] or 0.0)
                    except (TypeError, ValueError):
                        q = 0.0
                    if act.startswith("BUY"):
                        if abs(held) < 1e-6:
                            start_idx = i
                        held += q
                    elif act.startswith("SELL"):
                        held -= q
                        if abs(held) < 1e-6:
                            round_trips.append(
                                (start_idx, i, t["timestamp"] or "")
                            )
                # Pick the trip whose close ts == lot's closed_at when present,
                # otherwise the most recent trip that finished by closed_at.
                # ISO-8601 strings compare lexically.
                chosen: tuple[int, int, str] | None = None
                for tr in round_trips:
                    if tr[2] == closed_at:
                        chosen = tr
                        break
                if chosen is None:
                    for tr in round_trips:
                        if not closed_at or tr[2] <= closed_at:
                            chosen = tr  # last one wins (newest <= closed_at)
                realized = 0.0
                cost = 0.0
                proceeds = 0.0
                n_trades = 0
                if chosen is not None:
                    lo, hi, _ts = chosen
                    for t in all_trades[lo:hi + 1]:
                        act = (t["action"] or "").upper()
                        val = float(t["value"] or 0.0)
                        if act.startswith("SELL"):
                            proceeds += val
                            realized += val
                        elif act.startswith("BUY"):
                            cost += val
                            realized -= val
                        n_trades += 1
                d["realized_pl"] = round(realized, 2)
                d["cost"] = round(cost, 2)
                d["proceeds"] = round(proceeds, 2)
                d["realized_pl_pct"] = (round(realized / cost * 100.0, 2)
                                        if cost > 1e-9 else None)
                d["n_trades"] = n_trades
                # Hold duration. Both endpoints are ISO-8601 UTC; parse what
                # we can and surface a fractional-day figure for downstream %
                # framing (compounding) and a coarser hold_seconds for tests
                # that want exact integer arithmetic. ``None`` on either
                # unparseable side — the field is purely additive, callers
                # already tolerate missing keys (the realized_pl_pct
                # precedent).
                hold_seconds, hold_days = _hold_duration(
                    d.get("opened_at"), d.get("closed_at")
                )
                d["hold_seconds"] = hold_seconds
                d["hold_days"] = hold_days
                out.append(d)
            return out

    def update_position_marks(self, marks: dict):
        """marks: {position_id: (current_price, unrealized_pl)}"""
        with self._lock:
            for pid, (price, upl) in marks.items():
                self.conn.execute(
                    "UPDATE positions SET current_price=?, unrealized_pl=? WHERE id=?",
                    (price, upl, pid),
                )
            self.conn.commit()

    # ─── decisions ─────────────────────────────────────────────
    def record_decision(self, market_open: bool, signal_count: int, action_taken: str,
                        reasoning: str, portfolio_value: float, cash: float) -> int:
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO decisions (timestamp, market_open, signal_count, action_taken, reasoning, portfolio_value, cash) "
                "VALUES (?,?,?,?,?,?,?)",
                (_now(), 1 if market_open else 0, signal_count, action_taken, reasoning, portfolio_value, cash),
            )
            self.conn.commit()
            return cur.lastrowid

    def recent_decisions(self, limit: int = 20) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM decisions ORDER BY timestamp DESC, id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def last_real_decision(self) -> dict | None:
        """Most recent decision row where the engine actually decided something
        — the one row a trader's "when did this thing last *do* anything?"
        question wants answered.

        Filters out NO_DECISION cycles (those represent the engine cycling on
        cadence without producing a parseable Claude response: quota /
        host-saturated / timeout / parse miss — the documented IDLE_STORM
        regime). FILLED / HOLD / BLOCKED rows count as real decisions; a HOLD
        is a *deliberate* choice and a BLOCKED is a real decision the
        risk-check rejected, both meaningfully different from a NO_DECISION.
        Returns ``None`` when no real decision exists in history (a
        fresh-boot book whose first 24h was all host-saturated cycles — the
        existence of THIS pathology is what motivated the method: a trader
        watching the live ``/api/state`` heartbeat "HEALTHY — last decision
        6m ago" sees a green light, but ``recent_decisions(1)[0]`` is a
        ``NO_DECISION``; the engine has not actually decided anything for
        days). A targeted SQL — no Python-side filter, no scanning the whole
        table — so an old corpus stays cheap.

        Ordered by ``timestamp DESC, id DESC`` (same tie-break as
        ``recent_decisions`` / ``recent_trades`` — same-microsecond rows
        keep insertion order, the canonical ordering).

        Returns the full row shape ``recent_decisions`` returns (same
        columns) so callers can read ``timestamp`` / ``action_taken`` /
        ``reasoning`` / ``portfolio_value`` / ``cash`` directly. ``None``
        on no match keeps the additive contract: any builder that wants
        this signal can degrade-safely treat None as "engine has never
        decided" — the missing-data fallback every analytics builder
        already implements.
        """
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM decisions "
                "WHERE action_taken IS NOT NULL "
                "AND action_taken != '' "
                "AND action_taken != 'NO_DECISION' "
                "AND action_taken NOT LIKE 'NO_DECISION%' "
                "ORDER BY timestamp DESC, id DESC LIMIT 1"
            ).fetchone()
        return dict(row) if row else None

    # ─── equity curve ──────────────────────────────────────────
    def record_equity_point(self, total_value: float, cash: float, sp500: float | None):
        with self._lock:
            self.conn.execute(
                "INSERT INTO equity_curve (timestamp, total_value, cash, sp500_price) VALUES (?,?,?,?)",
                (_now(), total_value, cash, sp500),
            )
            self.conn.commit()

    def equity_curve(self, limit: int = 500) -> list[dict]:
        # Most recent `limit` points, returned in ascending order.
        with self._lock:
            rows = self.conn.execute(
                "SELECT id, timestamp, total_value, cash, sp500_price FROM equity_curve "
                "ORDER BY timestamp DESC, id DESC LIMIT ?", (limit,)
            ).fetchall()
        # reversed() → ascending by (timestamp, id): same-microsecond points
        # keep insertion order instead of an arbitrary sqlite ordering.
        return [{k: r[k] for k in ("timestamp", "total_value", "cash", "sp500_price")}
                for r in reversed(rows)]

    def close(self):
        self.conn.close()


_singleton: Store | None = None
_singleton_lock = threading.Lock()


def get_store() -> Store:
    # The runner's main thread and the Flask dashboard thread both call
    # get_store() at startup. Without serializing the create, an interleave on
    # the `is None` check spins up two Store instances — two separate sqlite
    # connections to the same WAL DB, one of which leaks. Double-checked lock:
    # the fast path stays lock-free once the singleton exists.
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = Store()
    return _singleton
