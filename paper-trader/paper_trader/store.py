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
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          TEXT NOT NULL,
    type            TEXT NOT NULL,
    qty             REAL NOT NULL,
    avg_cost        REAL NOT NULL,
    current_price   REAL DEFAULT 0,
    unrealized_pl   REAL DEFAULT 0,
    expiry          TEXT,
    strike          REAL,
    opened_at       TEXT NOT NULL,
    closed_at       TEXT,
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
        # a None row or a row whose columns read back NULL even though the
        # stored values are well-formed. get_portfolio() backs /api/state,
        # polled every few seconds — a raw subscript / json.loads(None) here
        # 500s the whole dashboard (28x in runner.log over 2 days). Absorb
        # both failure modes instead of crashing; never retry under the lock.
        if row is None:
            # Either a transient blip or the id=1 row is genuinely gone.
            # _init_portfolio() re-inserts only if missing and takes self._lock
            # itself, so it must be called WITHOUT the lock held. One re-read.
            print("[store] get_portfolio: row=None — re-initializing portfolio row",
                  flush=True)
            self._init_portfolio()
            with self._lock:
                row = self.conn.execute(
                    "SELECT cash, total_value, positions_json, last_updated "
                    "FROM portfolio WHERE id=1"
                ).fetchone()
            if row is None:
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

    def update_portfolio(self, cash: float, total_value: float, positions: list):
        with self._lock:
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
                        expiry: str | None = None, strike: float | None = None) -> None:
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
                    self.conn.execute(
                        "UPDATE positions SET qty=?, avg_cost=? WHERE id=?",
                        (new_qty, blended, existing["id"]),
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
                            "unrealized_pl=0, opened_at=?, closed_at=NULL WHERE id=?",
                            (qty, avg_cost, _now(), closed["id"]),
                        )
                    else:
                        self.conn.execute(
                            "INSERT INTO positions (ticker, type, qty, avg_cost, expiry, strike, opened_at) "
                            "VALUES (?,?,?,?,?,?,?)",
                            (ticker, type_, qty, avg_cost, expiry, strike, _now()),
                        )
            self.conn.commit()

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
