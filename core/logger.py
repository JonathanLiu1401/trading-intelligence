"""
Centralised structured logging for Digital Intern.

Every module should do:
    from core.logger import get_logger
    log = get_logger(__name__)

Outputs:
  - Console: coloured human-readable lines
  - logs/daemon.log: plain text (rotated daily, kept 14 days)
  - logs/structured.jsonl: one JSON object per line for dashboards/grep
"""
import json
import logging
import logging.handlers
import os
import sys
import time
import threading
from datetime import datetime, timezone
from pathlib import Path

_LOG_DIR = Path(os.environ.get("DIGITAL_INTERN_LOG_DIR",
                               Path(__file__).resolve().parent.parent / "logs"))
_LOG_DIR.mkdir(parents=True, exist_ok=True)

_PLAIN_LOG   = _LOG_DIR / "daemon.log"
_STRUCT_LOG  = _LOG_DIR / "structured.jsonl"
_METRICS_LOG = _LOG_DIR / "metrics.jsonl"

# ── ANSI colours for console ─────────────────────────────────────────────────
_COLOURS = {
    "DEBUG":    "\033[36m",    # cyan
    "INFO":     "\033[32m",    # green
    "WARNING":  "\033[33m",    # yellow
    "ERROR":    "\033[31m",    # red
    "CRITICAL": "\033[35m",    # magenta
}
_RESET = "\033[0m"


class _ColourFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        colour = _COLOURS.get(record.levelname, "")
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        name = record.name.replace("digital_intern.", "")
        return f"{colour}{ts} [{record.levelname[0]}] {name}: {record.getMessage()}{_RESET}"


class _JSONLHandler(logging.Handler):
    """Appends one JSON record per log line to structured.jsonl."""
    def __init__(self, path: Path):
        super().__init__()
        self._path = path
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord):
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        # copy any extra fields attached by the caller
        for k, v in record.__dict__.items():
            if k not in ("msg", "args", "levelname", "name", "pathname",
                          "filename", "module", "exc_info", "exc_text",
                          "stack_info", "lineno", "funcName", "created",
                          "msecs", "relativeCreated", "thread", "threadName",
                          "processName", "process", "message", "taskName"):
                try:
                    json.dumps(v)
                    entry[k] = v
                except (TypeError, ValueError):
                    pass
        line = json.dumps(entry, ensure_ascii=False)
        with self._lock:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(line + "\n")


def _build_root_logger():
    root = logging.getLogger()
    if root.handlers:
        return  # already initialised

    root.setLevel(logging.DEBUG)

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(_ColourFormatter())
    root.addHandler(ch)

    # Rotating plain-text file (10 MB × 7 files)
    fh = logging.handlers.RotatingFileHandler(
        _PLAIN_LOG, maxBytes=10 * 1024 * 1024, backupCount=7, encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    ))
    root.addHandler(fh)

    # Structured JSONL
    jh = _JSONLHandler(_STRUCT_LOG)
    jh.setLevel(logging.DEBUG)
    root.addHandler(jh)

    # Silence noisy third-party loggers that flood structured.jsonl with
    # per-connection DEBUG chatter (urllib3 alone produces dozens of entries
    # per Reddit poll, drowning out real signal during log audits).
    for noisy in (
        "urllib3", "urllib3.connectionpool", "urllib3.util.retry",
        "requests", "asyncio", "charset_normalizer", "chardet",
        "websockets", "websockets.client", "websockets.server",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)


_build_root_logger()


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


# ── Metrics helpers ──────────────────────────────────────────────────────────
_metrics_lock = threading.Lock()


def record_metric(name: str, value: float, tags: dict | None = None):
    """Append a time-series metric point to metrics.jsonl."""
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "metric": name,
        "value": value,
        **(tags or {}),
    }
    line = json.dumps(entry, ensure_ascii=False)
    with _metrics_lock:
        with open(_METRICS_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
