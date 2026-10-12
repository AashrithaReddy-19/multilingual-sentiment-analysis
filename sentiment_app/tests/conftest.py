import os
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND))
os.environ.setdefault("PRELOAD_MODEL", "false")   # API tests never load the real 1.1 GB model
os.environ.setdefault("MAX_BATCH_ROWS", "5")
