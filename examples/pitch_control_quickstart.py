"""
Pitch control on one SkillCorner open-data match with the parameters fitted
to the open data (reports/w2_parameter_fit.md).

Run from the repo root:  py examples/pitch_control_quickstart.py
Downloads the match (~95 MB) into data/ on first use.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dsv import data as D, metrics as M, pitchcontrol as PC  # noqa: E402

match_id = 1925299
match_dir = D.download_open_match(match_id, "data")

# 1. Load, put the home team attacking +x in both halves, add velocities.
meta = D.load_match_metadata(match_dir, match_id)
players, ball = D.normalize_direction(*D.load_tracking(match_dir, match_id, meta), meta)
players = D.add_velocity(players)

# 2. Per-player, per-half top speed (v95 on detected frames), floored per
#    position group with the floors pooled over all 20 open-data matches.
v95 = D.compute_v95(players, meta)
vmax = PC.vmax_lookup(PC.vmax_table(v95, floors=PC.SKILLCORNER_VMAX_FLOORS), match_id)
params = PC.SKILLCORNER_OPEN_DATA

# 3. A full-pitch control surface for one frame.
frame = int(ball.loc[ball["possession_group"].notna(), "frame"].iloc[2000])
state = PC.frame_state(players, ball, frame, meta, vmax, params)
grid = PC.make_grid(meta.pitch_length, meta.pitch_width)
surface = PC.pitch_control(state, grid, params)
print(f"frame {frame}: team in possession controls {surface.attack.mean():.0%} of the pitch")

# 4. Many (frame, point) queries at once: every pass in the match, scored by
#    the passing team's control where the ball ended up.
passes = D.passes_with_destination(D.load_events(match_dir, match_id), ball, meta)
batch = PC.point_batch(players, ball, passes["frame_end"], passes[["dest_x", "dest_y"]],
                       meta, vmax, params)
p = PC.team_control_at_points(batch, passes["passer_home"].to_numpy(), params)
s = M.summary(passes["success"], p)
print(f"{len(passes)} passes: AUC {s['auc']:.3f}, log loss {s['log_loss']:.3f}")
riskiest = passes.assign(p=p).sort_values("p").head(3)
print(riskiest[["minute_start", "player_name", "p", "success"]].to_string(index=False))
