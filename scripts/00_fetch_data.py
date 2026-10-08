"""Download every SkillCorner open-data match (metadata, dynamic events,
phases of play and the ~90-100 MB git-LFS tracking file) into data/.

Run from the repo root:  py scripts/00_fetch_data.py
Files already downloaded are skipped.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dsv import data as D  # noqa: E402

DATA_DIR = Path(os.environ.get("DSV_DATA_DIR", ROOT / "data"))

if __name__ == "__main__":
    for mid in D.list_open_matches(DATA_DIR):
        D.download_open_match(mid, DATA_DIR)
        print("ok", mid)
