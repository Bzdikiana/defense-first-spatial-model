"""
Threat Conceded on the open DFL/Sportec matches (independent data for the
pre-registered confirmation, reports/preregistration.md).

Everything is frozen from the SkillCorner open-data fits: pitch-control
parameters, vmax floors, expected threat, next-ball model and the
xG-from-here surface. Only each player's v95 is measured on the new data, as
the method prescribes. Optical tracking has no extrapolated positions, so
every player uses the detected-position spread.

For each match: load and synchronize (dsv.dfl), team possessions from the
tracking feed's ball-possession flag, shots placed on the tracking frame,
then Threat Conceded 5 times per second for every possession. Writes
data/derived/dfl/<match>_{frames.csv.gz,possessions.csv,shots.csv} and
data/derived/dfl/structure.csv (data checks only, no outcome analysis).

Needs kloppy. Takes about 30 minutes; matches already computed are skipped
unless REBUILD=1.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dsv import data as D, dfl, pitchcontrol as PC, threat as T  # noqa: E402

DATA_DIR = Path(os.environ.get("DSV_DATA_DIR", ROOT / "data"))
OUT = DATA_DIR / "derived" / "dfl"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = DATA_DIR / "dfl" / "cache"
STEP = 5   # frames at 25 fps: 5 per second, as on SkillCorner


def comparisons():
    spec = importlib.util.spec_from_file_location("s06", ROOT / "scripts" / "06_threat.py")
    s06 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s06)
    return s06.comparisons()


def match_tables(mid, comps):
    f_path, p_path, s_path = OUT / f"{mid}_frames.csv.gz", OUT / f"{mid}_possessions.csv", OUT / f"{mid}_shots.csv"
    if f_path.exists() and p_path.exists() and s_path.exists() and not os.environ.get("REBUILD"):
        return None
    t0 = time.perf_counter()
    meta, players, ball, poss, shots = dfl.prepare_match(mid, CACHE)
    v95 = D.compute_v95(players, meta)
    vt = PC.vmax_table(v95, floors=PC.SKILLCORNER_VMAX_FLOORS)
    frames = T.match_threat(players, ball, poss, meta, PC.vmax_lookup(vt, mid), step=STEP, comparisons=comps)
    frames.to_csv(f_path, index=False)
    poss.to_csv(p_path, index=False)
    shots.to_csv(s_path, index=False)
    alive_min = ball["alive"].sum() / meta.fps / 60
    row = dict(match_id=mid, home=meta.home_name, away=meta.away_name, alive_minutes=round(alive_min, 1),
               possessions=len(poss), frames=len(frames), shots=len(shots), shots_synced=int(shots["matched"].sum()),
               shots_in_possession=int(dfl.matched_shots(shots[shots["matched"]], poss).sum()),
               median_sync_m=round(float(shots.loc[shots["matched"], "sync_dist"].median()), 2),
               clock_offsets_s=str({k: round(v, 2) for k, v in meta.raw["sync_offsets"].items()}),
               minutes=round((time.perf_counter() - t0) / 60, 1))
    print(f"  {mid}: {row}", flush=True)
    return row


def main():
    comps = comparisons()
    rows = []
    for mid in dfl.OPEN_MATCHES:
        r = match_tables(mid, comps)
        if r is not None:
            rows.append(r)
    if rows:
        s_path = OUT / "structure.csv"
        old = pd.read_csv(s_path) if s_path.exists() else pd.DataFrame()
        new = pd.DataFrame(rows)
        if len(old):
            old = old[~old["match_id"].isin(new["match_id"])]
        pd.concat([old, new]).to_csv(s_path, index=False)
    print(pd.read_csv(OUT / "structure.csv").to_string(index=False))


if __name__ == "__main__":
    main()
