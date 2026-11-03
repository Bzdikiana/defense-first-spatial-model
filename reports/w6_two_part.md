# The two-part model: the ball's situation and the defense's cover

**Status: exploratory.** These analyses were run on the 20 SkillCorner matches after the pre-declared Week 5 test failed ([w5_validation.md](w5_validation.md)), so treat them as a well-supported lead rather than a confirmed result. Script: `scripts/08_two_part.py`; model: `dsv/defense.py`.

## 1. Why the product failed

Threat Conceded multiplies reachable danger (where the ball is and can go) by the share the defense leaves uncovered, so log TC = log RD + log(1 − cover). In a model of a shot within 5 s, that forces both to carry the same coefficient. Fitted freely (with time in possession, defenders goal-side and phase controls; 95% match-bootstrap intervals):

| danger map | coefficient on log RD (situation) | coefficient on log(1 − cover) (defense) | ratio defense / situation |
|---|---|---|---|
| xT | 2.43 [2.28, 2.60] | 1.04 [0.76, 1.40] | 0.32 to 0.56 |
| xG from here | 1.72 [1.63, 1.86] | 1.18 [0.91, 1.49] | 0.53 to 0.86 |

The product is right only if the ratio is 1. The defense's share matters, but with a weight of its own, so it has to enter the model separately.

## 2. Cover with its own coefficient

Held out by match. The baseline describes where the ball is flexibly (distance and its square, goal-mouth angle, |y|, x, log reachable danger under both maps) with time in possession, defenders goal-side and phase of play. Negative differences mean cover helps.

| outcome | cover | frames (rate) | log loss without → with | difference [95%] | AUC without → with |
|---|---|---|---|---|---|
| shot within 5 s | xT | 308,004 (3.3%) | 0.0961 → 0.0939 | -0.0022 [-0.0031, -0.0014] | 0.922 → 0.928 |
| shot within 5 s | xG from here | 308,004 (3.3%) | 0.0961 → 0.0935 | -0.0026 [-0.0035, -0.0017] | 0.922 → 0.929 |
| ball into the box within 5 s | xT | 288,896 (5.0%) | 0.1380 → 0.1342 | -0.0038 [-0.0053, -0.0025] | 0.898 → 0.907 |
| ball into the box within 5 s | xG from here | 288,896 (5.0%) | 0.1380 → 0.1343 | -0.0037 [-0.0051, -0.0023] | 0.898 → 0.906 |

**Per possession** (2,931 possessions, 416 with a shot or goal; baseline log loss 0.2538, AUC 0.899):

| added to the baseline | log loss | AUC | difference [95%] |
|---|---|---|---|
| cover (xT) at the moment of peak danger | 0.2484 | 0.906 | -0.0054 [-0.0097, -0.0012] |
| cover (xG from here) at the moment of peak danger | 0.2484 | 0.907 | -0.0054 [-0.0096, -0.0013] |
| cover (xG from here) averaged over the possession | 0.2535 | 0.900 | -0.0003 [-0.0021, +0.0016] |
| Threat Conceded (xT) peak, product form | 0.2525 | 0.902 | -0.0013 [-0.0031, +0.0005] |
| Threat Conceded (xG from here) peak, product form | 0.2514 | 0.903 | -0.0024 [-0.0048, -0.0000] |

The defense's cover counts at the moment of greatest danger; averaged over the whole possession it carries nothing. Defending is decided at the critical moments.

Shot or goal rate (%) by how dangerous the possession got and the cover at that moment:

| danger | low cover | mid | high cover |
|---|---|---|---|
| Q1 (least) | 0.0 (n=431) | 0.0 (n=198) | 0.0 (n=104) |
| Q2 | 0.0 (n=188) | 0.0 (n=251) | 0.0 (n=294) |
| Q3 | 18.1 (n=182) | 19.3 (n=264) | 10.8 (n=286) |
| Q4 (most dangerous) | 44.3 (n=176) | 48.1 (n=264) | 32.8 (n=293) |

**Where cover matters** (moment level, xG-from-here cover; difference in held-out log loss within each group, negative = cover helps):

| split | group | frames | shot rate | difference [95%] |
|---|---|---|---|---|
| phase of play | create | 109,072 | 0.6% | -0.0001 [-0.0006, +0.0005] |
| phase of play | direct | 3,972 | 2.0% | -0.0071 [-0.0132, -0.0023] |
| phase of play | finish | 59,128 | 9.5% | -0.0031 [-0.0057, -0.0006] |
| phase of play | none | 53,463 | 0.9% | -0.0081 [-0.0110, -0.0051] |
| phase of play | quick_break | 3,898 | 24.6% | +0.0039 [-0.0145, +0.0244] |
| phase of play | set_play | 6,187 | 18.7% | -0.0132 [-0.0253, -0.0026] |
| phase of play | transition | 9,851 | 10.9% | -0.0064 [-0.0164, +0.0033] |
| defense | not labelled | 126,852 | 4.2% | -0.0052 [-0.0071, -0.0033] |
| defense | not organised | 64,080 | 6.1% | -0.0018 [-0.0034, -0.0003] |
| defense | organised | 117,072 | 0.7% | -0.0002 [-0.0007, +0.0004] |

## 3. Benchmark: SkillCorner's own EPV

SkillCorner's dynamic events carry an expected possession value built from full tracking context (chance of scoring in the next 30 s), recorded at the start of each player possession. Same possession test (2,890 possessions, 416 with a shot), peak at least 2 s before any shot, with the Week 5 controls:

| measure | log loss | AUC |
|---|---|---|
| SkillCorner EPV (peak) | 0.3030 | 0.865 |
| ball distance to goal (closest) | 0.2867 | 0.884 |
| Threat Conceded, xG from here (peak) | 0.2840 | 0.884 |
| Threat Conceded, xT (peak) | 0.2927 | 0.880 |

EPV vs ball distance: difference +0.0163 [+0.0085, +0.0243].

EPV + Threat Conceded (xG) vs EPV alone: difference -0.0198 [-0.0271, -0.0122].

EPV is only recorded when a player receives the ball, while Threat Conceded is measured five times a second, so the peak comparison favours ours; EPV also targets goals, not shots.

## 4. What a press does

For each of SkillCorner's on-ball engagements (a defender pressing the ball carrier), the change from the start of the engagement to 1 s after it, within the same possession, of the ball's reachable shooting danger and of the share the defense leaves uncovered (both log scale, xG-from-here map; 95% match-bootstrap intervals). Outcome flags are SkillCorner's.

| outcome | engagements | ball danger | uncovered share (xG map) | uncovered share (xT map) |
|---|---|---|---|---|
| beaten | 183 | +1.16 [+1.06, +1.24] | +0.10 [+0.06, +0.14] | +0.01 [-0.01, +0.04] |
| stopped danger | 80 | +0.61 [+0.43, +0.83] | +0.09 [+0.01, +0.18] | +0.03 [-0.06, +0.12] |
| reduced danger | 241 | +0.12 [+0.04, +0.22] | +0.11 [+0.06, +0.15] | +0.09 [+0.06, +0.13] |
| forced backward | 933 | -0.43 [-0.45, -0.40] | +0.20 [+0.18, +0.21] | +0.19 [+0.17, +0.21] |
| no flagged outcome | 10,489 | +0.22 [+0.19, +0.24] | -0.01 [-0.02, -0.00] | -0.03 [-0.04, -0.02] |

The largest rise in the uncovered share follows presses flagged "forced backward" (+0.20); after a beaten press it is +0.10, against -0.01 when no outcome is flagged. Every flagged outcome opens more space than an unflagged engagement: a press that forces the ball backward cuts ball danger sharply but leaves the defense covering a smaller share of what remains. That trade-off is what the pressing analysis weighs ([w7_pressing.md](w7_pressing.md)). "Stopped danger" engagements usually end the possession, so the few that remain are not typical.

![engagements](../figs/w6_engagements.png)

## 5. The fitted model

Fitted on all frames with the xG-from-here cover, the cover coefficient is -1.04: each unit of logit cover multiplies the odds of a shot in the next 5 s by 0.35.

![cover effect](../figs/w6_cover_effect.png)

**Shots prevented** compares each defense with a typical A-League defense in the same situations: the model's shot chance with the cover a typical defense holds there, minus the chance with the actual cover, summed over the match (fitted without the match being scored). Positive: fewer shots than a typical defense would have allowed. Per team and match (`data/derived/shots_prevented_team_matches.csv`):

| defending team | opponent | shots prevented | shots conceded |
|---|---|---|---|
| Central Coast Mariners Football Club | Melbourne City FC | +1.09 | 10 |
| Melbourne City FC | Central Coast Mariners Football Club | +0.86 | 5 |
| Auckland FC | Melbourne Victory Football Club | +0.84 | 15 |
| Auckland FC | Central Coast Mariners Football Club | +0.77 | 11 |
| Melbourne Victory Football Club | Central Coast Mariners Football Club | +0.31 | 1 |
| Wellington Phoenix FC | Auckland FC | -1.87 | 19 |
| Melbourne Victory Football Club | Wellington Phoenix FC | -2.14 | 10 |
| Western Sydney Wanderers FC | Auckland FC | -2.23 | 20 |
| Adelaide United Football Club | Sydney Football Club | -2.40 | 16 |
| Newcastle United Jets FC | Auckland FC | -2.74 | 18 |

Across the 40 team-matches, Spearman correlation between shots prevented and shots conceded: -0.47. Expected to be negative but not strongly so: shots conceded also depend on how often the opponent reached dangerous positions, which cover does not measure.

