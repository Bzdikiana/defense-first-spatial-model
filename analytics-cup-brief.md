# Analytics Cup 2.0: Technical Design (Defensive Space Value)

## Summary
A defense-first spatial model that (1) measures the dangerous space a team concedes over time (metric: Threat Conceded, TC), (2) attributes it to individual defenders, and (3) grades real defensive decisions (press, step, hold, drop, cover) against physically plausible alternatives, validated against shots and xG conceded. Shipped as a pip-installable module on SkillCorner open data. Research track, defensive-positioning theme.

## How this differs from Amar's winning project
The only shared piece is the raw surface. Amar optimized an xT-weighted pitch-control surface; this project builds a danger-weighted pitch-control surface too. Everything past that diverges:
- Question. Amar: given one frozen frame, where should the defenders stand to optimize the surface (prescriptive). This project: across a possession, how much danger did the defense actually concede, who is responsible, and was this specific real action the right call (descriptive plus evaluative).
- Unit of analysis. Amar: a single static frame. This project: time windows and real event moments.
- Method. Amar: simulated annealing to move all defenders to an optimum. This project: a descriptive metric, a per-defender attribution, and short-horizon counterfactuals for the one acting defender. It deliberately does NOT re-optimize everyone's position, which is Amar's move.
- Output. Amar: an idealized optimal re-positioning. This project: a validated conceded-threat number, a per-90 attribution, and a graded real decision.
- Data. This project leans on SkillCorner's event layer (presses, off-ball runs, phases); Amar's does not.
Amar's project became a positioning optimizer inside databallpy. This would be a complementary defensive-valuation and decision-grading module.

## Research question and hypotheses
Can tracking plus SkillCorner's contextual event layer explain how teams and players concede dangerous space, and let us grade whether specific defensive actions were the right call?
- H1: danger-weighted, defense-side pitch control (TC) predicts shots and xG conceded within a possession, better than naive baselines.
- H2: per-defender attribution of TC identifies who repeatedly concedes high-value space.
- H3: many pressing actions raise conceded threat (space opened behind) rather than reduce it, and a counterfactual model separates good presses from bad ones.

## Method

### Layer 0: pitch control surface
Spearman (2017) time-to-intercept model, evaluated on a 105 x 68 grid (1 m cells) per frame. Reuse Laurie Shaw's LaurieOnTracking or nvsclub/KNNPitchControl as the base.
For grid cell x and player j:
- Arrival time: tau_j(x) = t_react + |x - (p_j + v_j * t_react)| / vmax_j. Player continues current velocity for the reaction time, then runs straight at top speed.
- Interception CDF: f_j(x, T) = 1 / (1 + exp(-(T - tau_j(x)) / (sqrt(3) * sigma / pi))).
- Control accrual (Spearman ODE): dPC_j/dT = (1 - sum_k PC_k) * f_j(x, T) * lambda_j, integrated in T from the ball arrival time to +10 s, PC_j initialized at 0.
- PC_att(x, t) = sum of PC_j over attacking players, with attack plus defense control summing to about 1.
Defaults to start: t_react 0.7 s, amax 7.0 m/s^2, lambda 4.3 per s (goalkeeper higher), sigma 0.45 s, ball speed 15 m/s, vmax 5.0 m/s replaced by each player's v95 measured from the data (per half, as in the stamina/top-speed paper).

### Layer 1 (MVP): Threat Conceded
- Danger surface V(x): an xT grid (Karun Singh Markov xT) fit from A-League events, or Singh's published grid for the MVP. Normalize to [0, 1].
- Accessibility weight A(x, t): weight by whether the attack can actually use x (a pass or ball reachability surface, the Dangerous Accessible Space idea). Without this, raw pitch control counts space behind the ball as controlled, which is not dangerous, and TC will not beat baselines.
- Metric: TC(t) = sum over cells x of PC_att(x, t) * V(x) * A(x, t). Units are xT-equivalent controlled by the attack.
- Aggregations: TC time series per possession; peak TC; time-integral (area under TC); TC at the frame before the key event; means per phase (Build Up, Progression, Finish), team, match, and game-state.

### Layer 2: per-defender attribution
- Marginal suppression (leave-one-out): delta_i(t) = TC_minus_i(t) - TC(t) >= 0, the danger defender i is currently suppressing (their defensive value). Removing i and renormalizing control lets attacker control rise where i was covering.
- Fair credit (Shapley over defenders) when several defenders jointly cover a region, to avoid double counting. Full Shapley is 2^11, so sample it; leave-one-out is the MVP with the caveat that it overcounts in overlap.
- Accountability (who got beaten, not just who suppresses): at a TC spike, find the cell cluster that gained the most attacker control and attribute it to the responsible defender (nearest by expected coverage). A defender can suppress a lot on average and still be the one beaten on a goal, so track both.
- Aggregate to per-90 "danger suppressed" and "danger conceded when beaten," plus per-defender conceding heatmaps.

### Layer 3 (headline): defensive decision value (DDV)
- Trigger frames from events: a defender's on_ball_engagement (press, pressing, recovery_press), the start of an opponent's off_ball_run (a run to track), or a pass where a defender must step or cover.
- For acting defender j at trigger t0, build a small alternative action set: hold (stay on current trajectory), drop toward own goal, step to press, shift laterally to cover the channel. Realize each as a modified velocity profile over a 1 to 2 s horizon, constrained by vmax_j and amax. Keep about four.
- Hold the other players on their actual future trajectories (first-order counterfactual). Optionally let the nearest opponents react via a light heuristic, but that is where it gets data-hungry.
- For each alternative a, re-simulate j and recompute TC over the window; summarize by peak TC (or end-of-window attack xT). DDV(j, t0) = TC_actual_summary - min over a of TC_alt_summary. Positive means the real choice conceded more than the best plausible alternative.
- Pressing cost-benefit: compare the press branch to the hold branch. "Press worth it" when TC_press < TC_hold over the window, which automatically charges the press for space opened behind because stepping out raises attacker control in the vacated zone.
- Design choice that separates this from Amar: this is a decision grader, not a positioning optimizer. It compares one real action to a short, coach-realistic menu for the acting player, not a global optimum over all defenders.

## Validation
- Predictive gate (week 5, go/no-go). Unit: possession. Label: shot occurred and xG sum in the possession. Model: logistic regression, shot ~ peak_TC + controls (ball zone, phase, defenders behind ball). Report AUC, Brier, calibration. Baselines to beat: ball distance to goal, raw unweighted space conceded, a Fernandez-style space value. TC must beat these. Use leave-one-match-out cross-validation given the few matches. Do not build Layer 3 until this passes.
- Decision-grading validity (no ground-truth best decision). Three checks: (a) do graded-bad (high-DDV) real decisions lead to worse short-term outcomes than low-DDV ones on average, (b) find later situations where the alternative actually happened and compare, (c) animate a few cases and eye-check them.
- Attribution sanity: do per-90 conceders match the eye test, do conceding maps localize sensibly.

## Data engineering
- Load via kloppy skillcorner loader. Align tracking to events and phases, handling the known start-frame mismatch between event and tracking data (sync is nontrivial).
- Smooth positions (Savitzky-Golay), derive velocity and acceleration, compute each player's v95 per match and per half.
- Coverage audit (week 1). Fraction of frames with all 22 players; broadcast tracking drops off-screen players. Decide: restrict TC to full-coverage frames, restrict to the attacking half (better covered near the ball), or impute missing defenders (role-anchored baseline or Graph Imputer). Report TC sensitivity to the choice.
- Direction normalization: flip so the attacking team always attacks the same way, handle periods.

## Hard problems (and how to handle them)
1. Removing a defender and normalization. Pitch control is roughly zero-sum, so removing i inflates nearby attacker control. Renormalize PC over present players and compute delta_i consistently. Shapley fixes double-counting but is expensive, so sample it.
2. Counterfactual realism. Holding opponents fixed while moving one defender is first-order; attackers would react. Keep the horizon short (1 to 2 s), alternatives few and physically bounded, and frame as what-if under a motion model, not truth.
3. Ball accessibility. Weight TC by reachable danger (the A(x,t) term) or it will reward space behind the ball and fail validation.
4. Suppress vs beaten. delta_i measures suppression; add the separate beaten attribution at danger spikes for the accountability story coaches want.
5. Small data (about 10 matches). The physics and optimization core needs no training; only the danger surface and the validation logistic are fit, both low-parameter. Use leave-one-match-out CV, do not over-claim.
6. Goalkeeper and offside. Higher GK control rate; fold offside into the accessibility weight so dangerous space in behind only counts when usable.

## Outputs and deliverables
- TC time-series plots per possession with event markers; conceded-threat heatmaps.
- Per-defender per-90 leaderboards (danger suppressed, danger conceded when beaten) and conceding maps.
- Pressing cost-benefit report: TC_press vs TC_hold, "worth it" rate by player, team, and zone.
- Two or three animated case studies: the TC spike, the beaten defender, and the graded decision with its best alternative overlaid.
- The dsv package (pip-installable, kloppy/databallpy-compatible): pitch control, danger surface, TC, attribution, counterfactual DDV, viz. End-to-end submission notebook.

## Repo structure
```
defensive-space-value/
  pyproject.toml
  README.md
  dsv/
    __init__.py
    data.py           # load SkillCorner via kloppy, align events/phases, smooth, velocities, v95
    pitchcontrol.py   # Spearman / KNN control + per-player v95 + renormalized leave-one-out
    value.py          # xT / danger surface + accessibility weight
    threat.py         # TC computation and aggregation
    attribution.py    # leave-one-out / sampled Shapley + beaten attribution
    counterfactual.py # alternative-move simulation + DDV + pressing cost-benefit
    viz.py            # pitch plots, animations (mplsoccer)
  notebooks/
    01_data_audit.ipynb
    02_pitch_control.ipynb
    03_threat_conceded.ipynb
    04_validation.ipynb
    05_attribution.ipynb
    06_counterfactual_cases.ipynb
  submission.ipynb
  tests/
```

## Schedule (now to mid-Dec, submit by about Dec 18)
- W1 Oct 6-12: data audit, coverage decision, data.py.
- W2 Oct 13-19: pitch control base, v95, velocities, renormalized leave-one-out. pitchcontrol.py, sanity figures.
- W3 Oct 20-26: danger surface plus accessibility weight. value.py, xT heatmap.
- W4 Oct 27-Nov 2: TC metric and aggregation. threat.py, TC tables and series.
- W5 Nov 3-9: validation gate. AUC vs baselines, go/no-go.
- W6 Nov 10-16: attribution (suppression plus beaten). attribution.py, per-90 leaderboard, maps.
- W7 Nov 17-23: counterfactual engine. counterfactual.py, DDV on sample events.
- W8 Nov 24-30: scale DDV, pressing cost-benefit across matches.
- W9 Dec 1-7: package, tests, README, animations.
- W10 Dec 8-14: submission notebook, write-up, demo.

## Immediate next steps
1. Pull the data (git lfs) and run the week-1 coverage audit.
2. Scaffold the repo skeleton above.
3. Lock the danger surface choice (borrow an xT grid vs fit one from A-League) and the accessibility weighting.
