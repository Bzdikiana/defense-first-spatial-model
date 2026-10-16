# Defensive Space Value (dsv)

A defense-first spatial model on SkillCorner open data (A-League 2024/25, 20 matches):
how much dangerous space a team concedes over time (Threat Conceded), which defenders
are responsible, and whether specific defensive actions were the right call.
The design is in [analytics-cup-brief.md](analytics-cup-brief.md).

## Pitch control fitted to SkillCorner open data

Most public pitch-control code ships with parameters from Spearman (2017), set on
different data. Here the model is **fitted to 15,078 passes from all 20 SkillCorner
open-data matches and validated on held-out matches** (match-grouped 5-fold
cross-validation). Full write-up: [reports/w2_parameter_fit.md](reports/w2_parameter_fit.md).

| model, scored on held-out matches | log loss | AUC |
|---|---|---|
| `SKILLCORNER_OPEN_DATA` (fitted, detection-aware) | **0.224** | **0.937** |
| `SKILLCORNER_OPEN_DATA_NO_FLAGS` (fitted, no detection flags needed) | 0.225 | 0.937 |
| logistic on nearest-teammate vs nearest-opponent distance + pass length | 0.243 | 0.907 |
| `SPEARMAN_SHAW` (literature defaults, 5 m/s for everyone) | 0.262 | 0.932 |
| Voronoi: is the nearest player to the destination a teammate? | 0.274 | 0.800 |

The score is the passing team's control where the ball ended up, used as the
probability that the pass is completed.

| parameter | literature | fitted |
|---|---|---|
| reaction time `t_react` | 0.7 s | 0.770 s |
| arrival-time spread `sigma` | 0.45 s | 0.335 s |
| spread for extrapolated positions `sigma_extrap` | – | 0.431 s |
| ball speed | 15 m/s | 14.7 m/s |
| control rate `lam` | 4.3 /s | 40 /s (bounded only from below) |
| top speed | 5 m/s | each player's v95 per half, floored at their position group's 25th percentile |

What the fit shows:

- **The literature defaults are miscalibrated on this data.** The fitted model has 14% lower
  held-out log loss than the defaults: about half from refitting the parameters, half from
  per-player top speed. It also beats the best simple baseline (95% match-bootstrap interval
  of the log-loss difference: −0.026 to −0.012).
- **Per-player top speed matters, more so once the model is fitted.** Against the same fitted
  model with 5 m/s for everyone, it lowers log loss by 0.020 (95% interval 0.015–0.025).
  Raw v95 alone gains 0.018 when fitted but only 0.011 under the literature parameters,
  where it barely shows in AUC.
  Overriding goalkeepers to 5 m/s makes things worse; their own v95 (about 3 m/s) predicts
  better.
- **Extrapolated positions are less certain, and the model can say so.** Players whose
  position is SkillCorner's fill-in rather than a detection get 29% more arrival-time
  spread. The overall gain is small: it helps most where both nearest players are
  extrapolated and slightly hurts where exactly one is.
- **The control rate is not identified.** Any `lam` from about 40/s upward fits equally
  well: the data favour the limit where the ball goes to whoever gets there first (partly by
  construction, since a pass's destination is where someone controlled it).
- **Control is not a calibrated pass probability.** Out of fold, the model is underconfident
  between 0.5 and 0.8 (passes it rates at 65% are completed about 80% of the time). If you
  need pass probabilities, add a recalibration step.

### Quickstart

```python
from dsv import data as D, pitchcontrol as PC

match_id = 1925299
match_dir = D.download_open_match(match_id, "data")            # ~95 MB
meta = D.load_match_metadata(match_dir, match_id)
players, ball = D.normalize_direction(*D.load_tracking(match_dir, match_id, meta), meta)
players = D.add_velocity(players)

v95 = D.compute_v95(players, meta)                               # per player and half
vmax = PC.vmax_lookup(PC.vmax_table(v95, floors=PC.SKILLCORNER_VMAX_FLOORS), match_id)
params = PC.SKILLCORNER_OPEN_DATA

# one frame, whole pitch
frame = int(ball.loc[ball["possession_group"].notna(), "frame"].iloc[2000])
state = PC.frame_state(players, ball, frame, meta, vmax, params)
surface = PC.pitch_control(state, PC.make_grid(meta.pitch_length, meta.pitch_width), params)
surface.attack                                                   # (68, 105) team in possession

# many (frame, point) queries at once, e.g. every pass in the match (~0.02 s)
passes = D.passes_with_destination(D.load_events(match_dir, match_id), ball, meta)
batch = PC.point_batch(players, ball, passes["frame_end"], passes[["dest_x", "dest_y"]],
                       meta, vmax, params)
p_complete = PC.team_control_at_points(batch, passes["passer_home"].to_numpy(), params)
```

A runnable version is [examples/pitch_control_quickstart.py](examples/pitch_control_quickstart.py).
With other tracking data, fill a `PC.PointBatch` (positions, velocities, top speeds,
team flags, ball and target, in metres) or a `PC.FrameState` directly, and use
`SKILLCORNER_OPEN_DATA_NO_FLAGS` unless your data marks extrapolated positions.
To refit on your own passes: `PC.fit_params(batch, passer_home, success)`.

## xG for SkillCorner open-data shots

The open data has shots but no xG. `dsv.xg` provides one, trained on 40,541 StatsBomb
open-data shots using only features SkillCorner tracking also has: shot geometry, headers,
blockers in the shooting triangle, keeper position and pressure, taken from the tracking
frame of the shot. Full write-up: [reports/w3_xg.md](reports/w3_xg.md).

| | log loss | AUC | total xG vs goals |
|---|---|---|---|
| StatsBomb, held-out leagues | 0.257 | 0.800 | 3,892 vs 3,884 |
| SkillCorner A-League shots (524, penalties excluded) | 0.315 | 0.755 | 56.7 vs 60 |
| same, geometry only (no tracking context) | 0.330 | 0.715 | 55.9 vs 60 |

```python
from dsv import xg
shots = xg.skillcorner_shots(events, players, ball, meta)   # one row per shot, with features
shots["xg"] = xg.xg_for_shots(shots)                         # coefficients ship in dsv.xg.OPEN_DATA_XG
```

Caveats: it ranks shots less well when opponents near the shooter are extrapolated, and in
the held-out lower league it undercounts goals by 14% (6% on the A-League shots).
StatsBomb data is used under its open-data terms, with attribution
([github.com/statsbomb/open-data](https://github.com/statsbomb/open-data)).

## Danger and where the ball goes next

`dsv.value` ships two surfaces fitted on the 20 matches ([reports/w3_value.md](reports/w3_value.md)):

- `OPEN_DATA_XT`: Markov expected threat (12 x 8 zones) with each zone's shot value from the
  xG model. It correlates 0.99 with Karun Singh's published grid. For ranking which
  possessions lead to shots, every danger surface tried (this xT, Singh's grid, xG from the
  location) lands within 0.006 AUC of plain distance to goal.
- `OPEN_DATA_TRANSITION`: where the ball goes next given where it is. A two-part mixture
  (76% short and centred on the ball, 24% long and about 18 m forward) beats the single
  Gaussian used in Spearman-style models on held-out log-likelihood in all 20 matches.

`offside_players` flags attackers in offside positions so they can be left out of pitch
control (`pitch_control(..., exclude=...)`).

## SkillCorner open data: things to know

Found while auditing all 20 matches ([reports/w1_data_audit.md](reports/w1_data_audit.md)).
`dsv.data` handles each of these.

- **Positions are often extrapolated, not observed.** The public tracking file carries all 22
  players in every frame, but only about 57% of positions are detected (49–66% per match).
  No frame has all 22 detected. Detection falls with distance from the ball, from about 90%
  within 5 m to almost nothing beyond 60 m. Use the per-position `is_detected` flag.
- **Defenders are well tracked where it matters.** Whole-match detection of center backs is
  about 46% and of goalkeepers 15–20%, but when their team is defending with the ball in its
  own third it is 90% and 63%. Condition on game situation before concluding a position is
  unusable.
- **Some frames have no detections at all** (replays, camera cuts) while the ball is still
  flagged as detected.
- **Goalkeepers have no position group of their own.** They are `position_group == "Other"`
  along with substitutes; use `player_role.acronym == "GK"` (`dsv.data.player_group`).
- **Pitch size varies by match**: 104, 105 or 106 m long (all 68 m wide). Read it from each
  match's `pitch_length` rather than assuming 105 m.
- **Teams switch ends at half-time.** `home_team_side` gives each period's direction;
  `normalize_direction` makes the home team attack +x throughout.
- **Event coordinates are rotated per event**: the event's own team always attacks +x,
  whatever the period. `orient_events` maps them into the tracking frame.
- **Events and tracking share one frame clock.** `frame_start` / `frame_end` index the
  tracking file directly with no offset; event locations match tracking positions to a
  median 0–0.26 m per match.
- **There is no shot xG, and goals are not always shots.** Shots are `end_type == "shot"` on
  `player_possession` events. The 20 final scores add up to 66 goals, but only 62 player
  possessions are followed by `game_interruption_after == "goal_for"`, and 61 of those end
  in a shot (the other ends in a pass: Perth's 68' winner in match 1925299). Reconcile goal
  labels with the final score.
- **The tracking file's possession flag** (`possession.group`) is empty in 27% of frames in
  the match we checked and can flip on deflections; prefer the event layer for who is in
  possession.
- **Files contain non-ASCII names**: open them as UTF-8 (on Windows the default codec fails).

## Setup and reproduction

```
py -m pip install -e .[dev]
py scripts/00_fetch_data.py         # ~1.8 GB into data/ (tracking files are git-LFS on GitHub)
py scripts/01_data_audit.py         # coverage audit, v95   -> reports/w1_data_audit.md
py scripts/02_pitch_control.py      # pitch-control sanity  -> reports/w2_pitch_control.md
py scripts/03_fit_pitch_control.py  # parameter fit, ~40 min -> reports/w2_parameter_fit.md
py scripts/04_xg.py                 # xG (downloads StatsBomb shots, ~3 min) -> reports/w3_xg.md
py scripts/05_value.py              # xT and ball transitions -> reports/w3_value.md
py -m pytest
```

## Layout

| path | what |
|---|---|
| `dsv/data.py` | fetch, parse tracking / events / phases, direction normalization, kinematics, v95, coverage audit, event alignment, passes with destinations |
| `dsv/pitchcontrol.py` | Spearman pitch control (grid and batched points), per-player vmax, detection-aware uncertainty, renormalized leave-one-out, parameter fitting |
| `dsv/xg.py` | xG model: shared shot features for StatsBomb and SkillCorner, fitted coefficients |
| `dsv/value.py` | expected threat, next-ball-location model, offside, fitted surfaces |
| `dsv/metrics.py` | log loss, Brier, AUC, calibration error, logistic baseline |
| `dsv/viz.py` | pitch, player and surface plots |
| `dsv/threat.py` | Week 4: Threat Conceded and aggregation |
| `dsv/attribution.py` | Week 6: per-defender suppression and beaten attribution |
| `dsv/counterfactual.py` | Week 7: alternative actions, DDV, pressing cost-benefit |
| `scripts/` | runnable pipeline steps |
| `reports/`, `figs/` | generated write-ups and figures |
