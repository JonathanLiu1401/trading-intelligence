"""Top-level entrypoint — `python3 runner.py` boots the paper trader."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_trader.runner import main

if __name__ == "__main__":
    main()
