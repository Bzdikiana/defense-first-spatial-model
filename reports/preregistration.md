# Pre-registration: does the defense's cover predict shots on independent data?

Written 2026-11-02 and committed before the analysis below is run. The analysis code,
`scripts/10_confirmation.py`, is committed in the same commit, unrun. Anything not listed here
is exploratory and reported as such.

## Why

The pre-declared Week 5 test failed ([w5_validation.md](w5_validation.md)). Threat Conceded
(TC = reachable danger × (1 − cover)) did not beat ball distance to goal at predicting shots.
Diagnostics run afterwards on the same 20 SkillCorner matches suggested a reason. The product
forces the ball's situation and the defense's share to count equally. When the defense's
share, cover, gets its own coefficient on top of a flexible model of where the ball is, it
predicts shots both moment by moment and, measured at the moment of greatest danger, per
possession. Those diagnostics used every match we have, so they cannot confirm themselves.
This test uses data the project has not analysed.

## Data

The open DFL/Sportec data (Bassek et al. 2025, *Scientific Data* 12:195, CC BY 4.0) has
seven matches from the 2022/23 Bundesliga and 2. Bundesliga: J03WMX, J03WN1, J03WPY, J03WOH,
J03WQQ, J03WOY and J03WR9. It has optical tracking at 25 fps and official event data.

Before this document was written, we looked at the data only to build the loader
(`dsv/dfl.py`):

- On J03WMX: event-type counts (21 shots), the location and result of the first five shots,
  how far the tracked ball was from each shot's recorded location, the possession count, and
  a smoke test of Threat Conceded on six possessions (value ranges, and ball position at the
  synchronized shots).
- The script that computes the frame tables for all seven matches (`scripts/09_dfl_threat.py`)
  was started before this commit. It fits nothing and relates no predictor to any outcome.

## Frozen ingredients (no refitting on DFL)

- **Pitch control:** `SKILLCORNER_OPEN_DATA` with a 0.08 s step on a 2 m grid. Optical
  positions are all observed, so the detected-position spread (0.335 s) applies everywhere.
- **vmax:** each player's v95 per half, measured on DFL (the method's rule), floored at
  `SKILLCORNER_VMAX_FLOORS`. Goalkeepers use their own v95.
- **Danger maps and next-ball model:** `OPEN_DATA_XT` (expected threat), xG from here
  (`scripts/06_threat.py: comparisons()`) and `OPEN_DATA_TRANSITION` (next-ball model).
- **Tracking processing:** 5 frames per second (every 5th tracking frame); velocities from a
  17-frame Savitzky-Golay window.

## Units and labels

- **Team possessions:** runs of live-ball frames with one value of the tracking feed's
  ball-possession flag. Spells of the other team shorter than 1 s are absorbed unless that
  team shoots during them. A dead ball ends a possession (`dsv.dfl.dfl_possessions`).
- **Shots:** official shot events, shifted by a per-half clock offset, then placed on the
  frame within ±4 s that minimizes ball-to-shot-location plus ball-to-shooter distance. A shot
  counts only if the shooter is within 2.5 m of the ball at that frame and it falls in a
  possession of the shooting team (up to 2 s after the possession ends) (`dsv.dfl.dfl_shots`).
  Unmatched shots are reported, not used.
- **Frame label:** the possession's first shot comes within the next 5 s. Frames after it are
  dropped.
- **Box-entry label:** the ball enters the penalty area (attacking-frame x > 36 m, |y| <
  20.16 m) within the next 5 s of the same possession. Only frames with the ball outside the
  area and no shot within 5 s are used.
- **Possession label:** a shot or goal. Only possessions with at least one frame 2 s or more
  before the first shot (or the end) are used.

## Models (`dsv.defense`)

All models are logistic regression (`dsv.metrics.fit_logistic`, ridge 1e-4).

- **Frame level:**
  - **M0** uses `design(frames, FRAME_CONTROLS, cover_col=None)`. The controls are time in
    possession and defenders goal-side of the ball. The ball features are distance to goal
    and its square, goal-mouth angle, |y|, x, and log reachable danger under xT and under xG
    from here.
  - **M1** = M0 + logit(cover).
- **Possession level:**
  - The **moment of peak danger** is the frame with the highest reachable xG-from-here danger
    at least 2 s before the shot or end (`possession_moments`).
  - **M0** uses `possession_design(p, cover_col=None)`. The controls are log duration, mean
    defenders goal-side and starting x. The ball features are closest distance to goal and its
    square, log peak reachable danger under both maps, and distance and |y| at the moment of
    peak danger.
  - **M1** = M0 + logit(cover at the moment of peak danger).
- There are no phase-of-play controls, because DFL has no phases.

## Evaluation

- **Fitting and scoring:** leave-one-match-out over the seven matches. The score is held-out
  log loss of M1 minus M0, averaged over units.
- **Interval:** a 95% match-bootstrap interval (2,000 resamples of matches, per-match sums,
  seed 0; `scripts/07_validation.py: boot_diff`).
- **Coefficient:** the sign of the cover coefficient comes from M1 fitted on all seven matches.

**Decision rule, per hypothesis:**

- **Confirmed:** the interval's upper end is below zero *and* the cover coefficient is
  negative (more cover, fewer shots).
- **Consistent, not confirmed:** the point estimate is below zero but the interval reaches
  zero.
- **Not confirmed:** otherwise.

## Hypotheses

| | test | cover | unit, label |
|---|---|---|---|
| **H1 (primary)** | M1 vs M0 | cover (xT) | frame, shot within 5 s |
| H2 | M1 vs M0 | cover (xT) at the moment of peak danger | possession, shot |
| H3a | M1 vs M0 | cover (xG from here) | frame, shot within 5 s |
| H3b | M1 vs M0 | cover (xG from here) at peak danger | possession, shot |
| H4 | M1 vs M0 | cover (xT) | frame, box entry within 5 s |
| H5 | M1 vs M0, both fitted on all 20 SkillCorner matches, scored on DFL without refitting | cover (xT) | frame, shot within 5 s |

H1 is the single primary test. The rest are secondary, all reported, with no multiplicity
correction; read them as support or contradiction, not as separate claims.

## Reported, but not part of the confirmation

- The same models on the 20 SkillCorner matches. Those data were already seen, so this is a
  reference only.
- The product form (adding log TC instead of logit cover).
- Cover averaged over the possession.
- Shot rate by danger quartile and cover tercile.
- Data checks: shots synchronized and matched, possession counts, clock offsets.

## Caveats known in advance

- Seven matches make the match bootstrap coarse, and the possession-level tests have few
  shots. A "consistent, not confirmed" result is a real possibility even if the effect exists.
- The Bundesliga differs from the A-League in level and style, and its tracking differs too
  (optical vs broadcast). A failure could reflect those differences, not only the absence of
  an effect.
- The shot labels depend on the event synchronization described above.
