# Defensive Space Value: research notes

The full documentation behind the [README](../README.md): every model, how it was fitted,
how it was tested, and what the SkillCorner open data taught us. Everything here uses only
the 20 A-League 2024/25 matches of the SkillCorner open data. Each section links to its
generated report in [`reports/`](../reports).

## Pitch control fitted to SkillCorner open data

Most public pitch-control code ships with parameters from Spearman (2017), set on
different data. Here the model is **fitted to 15,078 passes from all 20 matches and
validated on held-out matches** (match-grouped 5-fold cross-validation;
[w2_parameter_fit.md](../reports/w2_parameter_fit.md)).

| model, scored on held-out matches | log loss | AUC |
|---|---|---|
| `SKILLCORNER_OPEN_DATA` (fitted, detection-aware) | **0.224** | **0.937** |
| `SKILLCORNER_OPEN_DATA_NO_FLAGS` (fitted, no detection flags needed) | 0.225 | 0.937 |
| logistic on nearest-teammate vs nearest-opponent distance + pass length | 0.243 | 0.907 |
| `SPEARMAN_SHAW` (literature defaults, 5 m/s for everyone) | 0.262 | 0.932 |
| Voronoi: is the nearest player to the destination a teammate? | 0.274 | 0.800 |

The score is the passing team's control where the ball ended up, used as the probability
that the pass is completed.

| parameter | literature | fitted |
|---|---|---|
| reaction time `t_react` | 0.7 s | 0.770 s |
| arrival-time spread `sigma` | 0.45 s | 0.335 s |
| spread for extrapolated positions `sigma_extrap` | – | 0.431 s |
| ball speed | 15 m/s | 14.7 m/s |
| control rate `lam` | 4.3 /s | 40 /s (bounded only from below) |
| top speed | 5 m/s | each player's v95 per half, floored at their position group's 25th percentile |

- **The literature defaults are miscalibrated on this data.** The fitted model has 14% lower
  held-out log loss: about half from refitting the parameters, half from per-player top
  speed. It also beats the best simple baseline (95% match-bootstrap interval of the
  log-loss difference: −0.026 to −0.012).
- **Per-player top speed matters.** Against the same fitted model with 5 m/s for everyone,
  it lowers log loss by 0.020 (0.015–0.025). Goalkeepers predict better with their own v95
  (about 3 m/s) than with 5 m/s.
- **Extrapolated positions are less certain, and the model can say so.** Players whose
  position is SkillCorner's fill-in get 29% more arrival-time spread.
- **The control rate is not identified.** Any `lam` from about 40/s upward fits equally well.
- **Control is not a calibrated pass probability.** Out of fold, the model is underconfident
  between 0.5 and 0.8. Add a recalibration step if you need pass probabilities.

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

A runnable version is [examples/pitch_control_quickstart.py](../examples/pitch_control_quickstart.py).
With other tracking data, fill a `PC.PointBatch` or a `PC.FrameState` directly, and use
`SKILLCORNER_OPEN_DATA_NO_FLAGS` unless your data marks extrapolated positions. To refit on
your own passes: `PC.fit_params(batch, passer_home, success)`.

## xG from the open-data shots

The open data has shots but no xG. `dsv.xg` fits one on the competition's own 524
non-penalty shots (60 goals), with features from the tracking at the shot frame: geometry,
headers, blockers in the shooting triangle, keeper position, and the nearest opponents
([w3_xg.md](../reports/w3_xg.md)). The feature set and ridge penalty were chosen by
leave-one-match-out log loss.

| held out by match | log loss | AUC | total xG vs goals |
|---|---|---|---|
| full features, ridge 0.01 (kept: `OPEN_DATA_XG`) | 0.328 | 0.723 | 60.0 vs 60 |
| geometry only | 0.334 | 0.692 | 59.9 vs 60 |
| constant goal rate | 0.359 | – | 60.0 vs 60 |

```python
from dsv import xg
shots = xg.skillcorner_shots(events, players, ball, meta)   # one row per shot, with features
shots["xg"] = xg.xg_for_shots(shots)
danger = xg.xg_from_here(cells_attacking_frame, length, width)  # shooting-danger map
```

Sixty goals make this a small model. It is used mainly for its shape (how danger rises
toward goal), as a danger map and as the shot value inside expected threat.

## Danger and where the ball goes next

`dsv.value` ships two surfaces fitted on the 20 matches ([w3_value.md](../reports/w3_value.md)):

- `OPEN_DATA_XT`: Markov expected threat (Singh's method, 12 x 8 zones), each zone's shot
  value from the xG model. For ranking which possessions lead to shots, xT, xG from the
  location, and plain distance to goal are within 0.003 AUC of each other.
- `OPEN_DATA_TRANSITION`: where the ball goes next given where it is. A two-part mixture
  (76% short and centred on the ball, 24% long and about 18 m forward) beats the single
  Gaussian used in Spearman-style models on held-out log-likelihood in all 20 matches.

## Does Threat Conceded predict shots? The pre-declared test

Threat Conceded (TC) is the danger a defense allows at a moment: the attack's pitch control,
weighted by where the ball can go next and how dangerous each spot is (`dsv.threat`). The
test was declared before it was run: on 2,931 possessions, does peak TC (at least 2 s
before any shot) predict a shot or goal better than every baseline? Logistic regression,
same controls for every measure, held out by match ([w5_validation.md](../reports/w5_validation.md)).

| measure (controls + ...) | held-out log loss | AUC |
|---|---|---|
| ball distance to goal (closest point) | **0.283** | 0.885 |
| TC, xT as the danger map | 0.289 | 0.882 |
| ball-location danger only (no defense) | 0.290 | 0.879 |
| TC with nearest-player-wins instead of pitch control | 0.299 | 0.872 |
| control × xT without reachability (Fernández-style) | 0.298 | 0.863 |
| raw space conceded (m²) | 0.309 | 0.850 |

**The test failed.** TC beats every other space measure but not ball distance. With "xG
from here" as the danger map the defense does add beyond its own ball-location version
(−0.0037 [−0.0065, −0.0007] in log loss), but TC still only ties ball distance. The next
section explains why.

## The two-part model: the ball's situation and the defense's cover

**Why the product failed.** TC = RD × (1 − cover), so log TC = log RD + log(1 − cover):
the ball's situation (reachable danger RD) and the share the defense leaves uncovered are
forced to carry the same weight. Fitted freely in a model of a shot within 5 s, the
situation weighs about 2.3 times as much (coefficients 2.43 vs 1.04 on the log scale, xT
map; [w6_two_part.md](../reports/w6_two_part.md)). So the defense's share, **cover**, enters
with its own coefficient (`dsv.defense`):

    logit P(shot within 5 s) = controls + f(ball situation) + β · logit(cover)

f(ball situation) is flexible (distance and its square, goal-mouth angle, |y|, x, log
reachable danger under both maps). All results below are held out by match; they were
found after the failed test, so treat them as a strong lead rather than a confirmed result.

| adding cover to the ball's situation | held-out log loss difference [95%] |
|---|---|
| moment level: shot within 5 s (308,004 frames) | −0.0026 [−0.0035, −0.0017] |
| moment level: ball into the box within 5 s | −0.0037 [−0.0051, −0.0023] |
| possession level: cover at the moment of peak danger | −0.0054 [−0.0096, −0.0013] |
| possession level: cover averaged over the possession | −0.0003 [−0.0021, +0.0016] |

- **Cover predicts shots at a given ball situation**, moment by moment and per possession.
- **It counts at the critical moment.** Averaged over a possession it adds nothing.
- **It matters most where the defense is not set:** set plays, direct play, the finishing
  phase and unorganised defending; hardly at all in patient build-up against an organised
  block.
- `DF.OPEN_DATA_TWO_PART` and `DF.OPEN_DATA_TYPICAL_COVER` ship the fitted model and what a
  typical A-League defense covers in a given situation (the cover coefficient is −1.04:
  each unit of logit cover multiplies the odds of a shot by 0.35).

## Who holds the cover, and who loses it

Pitch control sums over players, so cover splits exactly across defenders
(`dsv.attribution`): each defender's share is the part of the reachable shooting danger
they would reach first. Checked against SkillCorner's flags for pressing defenders who were
beaten ([w6_attribution.md](../reports/w6_attribution.md)), blame by lost share does not single
out the beaten presser: their own loss separates beaten presses with AUC 0.54, and after a
beaten press they hold only 16% of the team's drop in cover. The rest belongs to the
teammates behind. Shares are a description of who holds the danger, not a verdict on a
press.

## What did the press do?

Each of the 14,872 on-ball engagements SkillCorner records during an opponent possession is
measured 3 s after the decision to press, against typical play from the same situation, in
the two-part model's chance of a shot ([w7_pressing.md](../reports/w7_pressing.md),
`dsv.counterfactual`): **on the ball** (less dangerous than usual, won, or out of play?) and
**in the shape** (better cover than a typical defense with the ball where it now is?).

| SkillCorner outcome | presses | on the ball | in the shape | total, shots prevented per 100 presses [95%] |
|---|---|---|---|---|
| beaten | 311 | −10.8 | −2.4 | −13.1 [−16.7, −9.4] |
| reduced danger | 424 | +6.5 | −1.3 | +5.1 [+2.6, +7.6] |
| stopped danger | 195 | +4.6 | −1.1 | +3.6 [+0.9, +6.3] |
| forced backward | 871 | +1.4 | +0.2 | +1.6 [+1.4, +1.9] |
| no flagged outcome | 13,071 | −1.2 | −0.1 | −1.2 [−1.6, −0.9] |

- **The grades agree with SkillCorner's independent outcome flags.** A beaten press also
  costs the shape: space opens behind it.
- **Most presses work, a few fail badly.** 75% leave the attack worse off than typical; the
  average is slightly negative, pulled down by the beaten ones. The losses sit in the own
  third (−3.5 per 100); elsewhere the average is close to zero.
- **Players: role, not yet individual skill.** Raw values sort players by role (centre-backs
  lowest). Compared with presses of the same type in the same zone, a player's two halves of
  presses agree 0.49 across players but only 0.10 within a position group: with 20 matches
  individual pressing skill cannot yet be separated from noise.
- **Why not "what if only the presser had held?"** That first-order counterfactual keeps the
  teammates who shifted to cover in place, so a presser who "holds" only duplicates their
  cover. It is computed and reported, not used as the grade.

## Interactive explorer and animations

`scripts/13_explorer.py` builds an interactive replay of five possessions
(`explorer/build/possession_explorer.html`): the danger left open as a continuous field or
summed into the 18 classic zones (same total), pitch control, the cover against a typical
defense, the chance of a shot and each defender's share of the cover, five times a second.
`scripts/14_animate.py` renders the same replays as GIFs (`figs/w8_*.gif`).

**Continuous or discrete?** Time is continuous (every number recomputed five times a
second; possession summaries take the moment of peak danger). Space is a continuous field
(pitch control and xG from here are smooth functions of position, evaluated on 2 m cells);
zones only add the same cells up. Expected threat is estimated on 12 x 8 zones and
interpolated between their centres.

## SkillCorner open data: things to know

Found while auditing all 20 matches ([w1_data_audit.md](../reports/w1_data_audit.md)).
`dsv.data` handles each of these.

- **Positions are often extrapolated, not observed.** Every frame carries all 22 players,
  but only about 57% of positions are detected (49–66% per match), falling from about 90%
  within 5 m of the ball to almost nothing beyond 60 m. Use the `is_detected` flag.
- **Defenders are well tracked where it matters.** With the ball in their own third,
  centre-backs are detected 90% of the time and goalkeepers 63%.
- **You cannot judge offside from it.** For the 51 passes called offside, the tracking puts
  the receiver offside at the pass frame only 12% of the time, so Threat Conceded has no
  offside rule ([w3_value.md](../reports/w3_value.md)).
- **Some frames have no detections at all** (replays, camera cuts).
- **Goalkeepers have no position group of their own**; use `player_role.acronym == "GK"`.
- **Pitch size varies by match**: 104, 105 or 106 m long.
- **Teams switch ends at half-time**; `normalize_direction` makes the home team attack +x.
- **Event coordinates are rotated per event**; `orient_events` maps them into the tracking frame.
- **Events and tracking share one frame clock**, with no offset.
- **Goals are not always shots.** The final scores add up to 66 goals; 61 are on
  shot-tagged possessions.
- **The tracking file's possession flag** is empty in about a quarter of frames; prefer the
  event layer for who is in possession.
- **Files contain non-ASCII names**: open them as UTF-8.

## Full reproduction

```
py -m pip install -e .[dev]
py scripts/00_fetch_data.py         # ~1.8 GB into data/ from github.com/SkillCorner/opendata
py scripts/01_data_audit.py         # coverage audit, v95   -> reports/w1_data_audit.md
py scripts/02_pitch_control.py      # pitch-control sanity  -> reports/w2_pitch_control.md
py scripts/03_fit_pitch_control.py  # parameter fit, ~40 min -> reports/w2_parameter_fit.md
py scripts/04_xg.py                 # xG on the open-data shots -> reports/w3_xg.md
py scripts/05_value.py              # xT and ball transitions -> reports/w3_value.md
py scripts/06_threat.py             # Threat Conceded, all matches, ~40 min -> reports/w4_threat.md
py scripts/07_validation.py         # the pre-declared test -> reports/w5_validation.md
py scripts/08_two_part.py           # ball situation + cover, ~25 min -> reports/w6_two_part.md
py scripts/11_attribution.py        # who loses the cover -> reports/w6_attribution.md
py scripts/12_pressing.py           # what did each press do? ~15 min -> reports/w7_pressing.md
py scripts/13_explorer.py           # interactive replay -> explorer/build/possession_explorer.html
py scripts/14_animate.py            # GIF replays -> figs/w8_*.gif
py -m pytest
```

## Layout

| path | what |
|---|---|
| `dsv/data.py` | fetch, parse tracking / events / phases, direction normalization, kinematics, v95, coverage audit |
| `dsv/pitchcontrol.py` | Spearman pitch control (grid and batched points), per-player vmax, detection-aware uncertainty, parameter fitting |
| `dsv/xg.py` | xG from tracking at the shot frame, fitted on the open-data shots; xG-from-here danger map |
| `dsv/value.py` | expected threat, next-ball-location model, fitted surfaces |
| `dsv/threat.py` | Threat Conceded per frame (reachable danger, cover), team possessions, summaries |
| `dsv/defense.py` | the two-part model (ball situation + cover), typical cover, shots prevented, moments of peak danger |
| `dsv/attribution.py` | each defender's share of the cover, blame, leave-one-out |
| `dsv/counterfactual.py` | what a press did vs typical play (on the ball, in the shape); held-position comparison |
| `dsv/metrics.py`, `dsv/viz.py` | scores and plots |
| `scripts/` | the pipeline, `explorer/` the interactive page, `reports/` and `figs/` the outputs |
