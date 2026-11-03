# What did the press do?

Every on-ball engagement SkillCorner records during an opponent possession (a defender pressing the ball carrier) is measured 3 s after the decision to press, against what typically happens from the same situation. Unit: the two-part model's chance of a shot in the next 5 s, with a won ball or a ball out of play counting as no danger and a shot as certain danger (`dsv/counterfactual.py`, `scripts/12_pressing.py`).

- **On the ball**: how much less dangerous the ball's position is than is typical from that situation, scored with a typical defense around it. Forcing the carrier back, winning the ball or kicking it out show up here.
- **In the shape**: how much better the defense covers the danger at that moment than a typical defense would with the ball where it now is, relative to what is typical after such situations. Space opened behind a press shows up here.
- **Total**: the two together; positive means the attack ended up less likely to shoot than is typical.

"Typical" comes from all moments in the other 19 matches, pressed or not, with the same ball situation, cover, time in possession, defenders goal-side and phase of play. Units: shots prevented per 100 presses. The decision point is 1 s before SkillCorner's engagement starts (the presser is then a median 9 m from the ball). Intervals: 95% match bootstrap.

**14,872 presses.** Mean on the ball -0.91, in the shape -0.18, total -1.08 per 100; 75% left the attack worse off than typical; 17% ended with the ball won or out of play within 3 s, 3.2% with a shot.

## Against SkillCorner's outcome flags

SkillCorner flags the outcome of each engagement from its own tracking rules, independently of this model.

| SkillCorner outcome | presses | on the ball | in the shape | total | attack worse off than typical | ball won or out by 3 s |
|---|---|---|---|---|---|---|
| beaten | 311 | -10.47 [-13.88, -7.13] | -2.73 [-3.48, -2.03] | -13.21 [-16.73, -9.55] | 31% | 10% |
| stopped danger | 195 | +4.77 [+2.62, +6.89] | -1.42 [-2.47, -0.42] | +3.34 [+0.65, +6.08] | 58% | 36% |
| reduced danger | 424 | +6.54 [+4.33, +8.93] | -1.36 [-2.06, -0.82] | +5.18 [+2.68, +7.68] | 64% | 28% |
| forced backward | 871 | +1.35 [+1.08, +1.64] | +0.30 [+0.24, +0.36] | +1.65 [+1.37, +1.95] | 93% | 0% |
| no flagged outcome | 13,071 | -1.16 [-1.52, -0.78] | -0.09 [-0.17, -0.02] | -1.25 [-1.61, -0.88] | 76% | 17% |

![press value](../figs/w7_press_value.png)

## By type and by zone

| SkillCorner subtype | presses | on the ball | in the shape | total | attack worse off than typical | ball won or out by 3 s |
|---|---|---|---|---|---|---|
| counter_press | 1,369 | +0.15 [-0.05, +0.34] | +0.00 [-0.12, +0.11] | +0.15 [-0.09, +0.39] | 86% | 7% |
| other | 2,063 | -2.72 [-3.70, -1.83] | -0.36 [-0.53, -0.20] | -3.08 [-4.20, -2.06] | 66% | 20% |
| pressing | 3,647 | +0.30 [+0.25, +0.34] | +0.07 [+0.04, +0.10] | +0.37 [+0.30, +0.43] | 91% | 11% |
| pressure | 4,059 | -0.10 [-0.62, +0.45] | -0.04 [-0.19, +0.08] | -0.14 [-0.67, +0.41] | 76% | 20% |
| recovery_press | 3,734 | -2.34 [-2.93, -1.72] | -0.53 [-0.70, -0.38] | -2.88 [-3.51, -2.17] | 59% | 20% |

| ball at the decision | presses | on the ball | in the shape | total | attack worse off than typical | ball won or out by 3 s |
|---|---|---|---|---|---|---|
| high (opponent's third) | 3,832 | +0.01 [+0.00, +0.02] | +0.01 [+0.01, +0.02] | +0.02 [+0.01, +0.03] | 91% | 20% |
| middle third | 6,810 | -0.12 [-0.27, +0.02] | -0.14 [-0.26, -0.05] | -0.26 [-0.48, -0.05] | 74% | 15% |
| own third | 4,230 | -3.00 [-4.06, -2.03] | -0.40 [-0.62, -0.21] | -3.41 [-4.49, -2.38] | 62% | 17% |

The own third dominates: a press there is a last line of defense, and its outcomes swing between a cleared ball and a shot. Far from goal the immediate danger is small either way; the main point of a high press, winning the ball for your own attack, is not counted by a measure of danger conceded.

## By team

| pressing team | matches | presses | on the ball | in the shape | total | total vs same type and zone |
|---|---|---|---|---|---|---|
| Newcastle United Jets FC | 1 | 441 | +1.38 | -0.38 | +0.99 | +2.14 |
| Melbourne City FC | 3 | 943 | +0.10 [-0.26, +0.46] | +0.14 [+0.09, +0.20] | +0.25 [-0.12, +0.66] | +0.87 |
| Perth Glory Football Club | 2 | 848 | +0.23 | -0.07 | +0.15 | +1.47 |
| Macarthur FC | 2 | 775 | -0.22 | -0.38 | -0.60 | +0.38 |
| Adelaide United Football Club | 3 | 1,177 | -0.21 [-0.79, +0.23] | -0.58 [-1.10, -0.26] | -0.80 [-1.89, -0.03] | +0.56 |
| Melbourne Victory Football Club | 4 | 1,223 | -0.88 [-2.61, +0.77] | -0.27 [-0.60, +0.07] | -1.15 [-3.12, +0.84] | -0.01 |
| Central Coast Mariners Football Club | 4 | 1,539 | -1.14 [-2.25, -0.08] | -0.11 [-0.37, +0.10] | -1.24 [-2.35, -0.08] | -0.11 |
| Auckland FC | 7 | 2,566 | -1.29 [-2.02, -0.70] | +0.03 [-0.12, +0.15] | -1.26 [-2.02, -0.76] | -0.33 |
| Sydney Football Club | 2 | 756 | -1.15 | -0.20 | -1.36 | -0.37 |
| Western Sydney Wanderers FC | 2 | 705 | -1.13 | -0.23 | -1.36 | -0.25 |
| Brisbane Roar FC | 3 | 1,064 | -1.35 [-1.87, -0.67] | -0.14 [-0.20, -0.03] | -1.50 [-2.04, -0.87] | -0.39 |
| Western United | 4 | 1,665 | -1.38 [-1.86, -0.79] | -0.27 [-0.53, -0.03] | -1.65 [-1.90, -1.37] | -0.67 |
| Wellington Phoenix FC | 3 | 1,170 | -2.05 [-3.04, -0.55] | -0.20 [-0.27, -0.14] | -2.25 [-3.30, -0.69] | -0.80 |

Intervals only for teams in at least 3 of the 20 matches.

## Players: role, not yet individual skill

Raw values mostly sort players by where they press: per position group (all presses),

| position | presses | share in own third | total | vs same type and zone |
|---|---|---|---|---|
| Central Defender | 1,229 | 48% | -9.46 | -7.39 |
| Midfield | 4,971 | 31% | -1.14 | +0.03 |
| Full Back | 2,204 | 38% | -0.94 | +0.48 |
| Other | 402 | 25% | -0.90 | +0.08 |
| Wide Attacker | 2,914 | 26% | +0.59 | +1.57 |
| Center Forward | 3,152 | 13% | +0.61 | +1.03 |

Central defenders stand out even against presses of the same type in the same zone. When they engage they are usually the last defender, so a lost duel is a chance at goal; comparing with typical play from the same ball situation does not see that role.

So players are compared on each press **against presses of the same SkillCorner type in the same zone** ("relative"). Players with at least 30 presses (164); 95% bootstrap intervals over their presses. **Is it a stable trait?** Splitting each player's presses into alternate halves, the correlation of the two halves' means across players is 0.49 for the relative value (Spearman-Brown reliability of a player's full mean 0.66); 0.10 (0.18) after also removing position-group means. For the raw total it is 0.55, much of it position.

**Within a position group the two halves barely agree**, so with 20 matches a player's pressing value cannot yet be told apart from noise once their role is accounted for. The table is shown for transparency, not as a ranking; a season of data per player would be needed.

| player | team | position | presses | in own third | relative [95%] | total | on the ball | in the shape |
|---|---|---|---|---|---|---|---|---|
| Dean Bosnjak | Macarthur FC | Wide Attacker | 40 | 42% | +5.78 [+3.30, +8.54] | +4.16 | +3.94 | +0.22 |
| Fabian Talladira | Adelaide United Football Club | Full Back | 38 | 63% | +5.64 [+2.26, +9.23] | +2.86 | +3.94 | -1.08 |
| Callum Timmins | Newcastle United Jets FC | Midfield | 41 | 44% | +4.94 [-0.80, +10.76] | +3.52 | +3.82 | -0.30 |
| Dylan Pierias | Adelaide United Football Club | Full Back | 60 | 53% | +4.47 [+0.58, +8.96] | +2.39 | +3.39 | -1.00 |
| Joel  Anasmo | Perth Glory Football Club | Wide Attacker | 32 | 53% | +4.26 [+2.21, +6.39] | +2.58 | +1.79 | +0.79 |
| Harry Van Der Saag | Brisbane Roar FC | Full Back | 38 | 45% | +4.09 [+0.66, +8.81] | +2.37 | +1.86 | +0.51 |
| Sam Sutton | Wellington Phoenix FC | Full Back | 73 | 37% | +3.98 [+0.34, +7.25] | +2.59 | +2.63 | -0.04 |
| Eli Adams | Newcastle United Jets FC | Wide Attacker | 63 | 37% | +3.92 [+1.78, +6.25] | +2.96 | +3.07 | -0.11 |
| Daniel Hall | Auckland FC | Central Defender | 60 | 37% | -8.13 [-14.62, -2.41] | -9.97 | -8.67 | -1.30 |
| Isaac Hughes | Wellington Phoenix FC | Central Defender | 44 | 59% | -9.96 [-18.75, -1.40] | -12.18 | -13.01 | +0.83 |
| Brendan Hamill | Melbourne Victory Football Club | Central Defender | 30 | 47% | -10.48 [-19.06, -2.56] | -13.00 | -10.55 | -2.46 |
| Panagiotis Kikianis | Adelaide United Football Club | Central Defender | 51 | 53% | -11.58 [-19.77, -3.83] | -14.23 | -11.65 | -2.58 |
| Dylan Leonard | Western United | Central Defender | 71 | 48% | -12.15 [-19.45, -5.52] | -14.45 | -13.67 | -0.78 |
| Nando Pijnaker | Auckland FC | Central Defender | 54 | 44% | -12.59 [-19.75, -6.28] | -14.51 | -14.21 | -0.31 |
| Scott Wootton | Wellington Phoenix FC | Central Defender | 47 | 74% | -14.85 [-25.36, -5.22] | -17.88 | -16.08 | -1.79 |
| James Donachie | Western United | Central Defender | 34 | 38% | -15.55 [-26.21, -5.64] | -17.68 | -16.23 | -1.45 |

## Two presses

![cases](../figs/w7_press_cases.png)

## Why not "what if they had held their position?"

The brief's first design moved only the presser: same moment, presser carried back into the shape (their position at the decision, moved along with their teammates), everyone else and the ball where they really were. Over 11,947 presses still live 3 s later, that comparison says pressing gave better cover than holding 51% of the time (mean -0.06 shots per 100 against pressing; holding from 2 s before instead: -0.07, correlation 0.96; dropping toward goal instead: +0.11). The reason is built in: the teammates who shifted to cover for the presser stay where they are, so a presser who "holds" only duplicates their cover and nobody is near the ball. A fair counterfactual would have to move the whole defense, which needs a model of how teams reorganize. Measuring against typical play from the same situation sidesteps that, at the price below.

## Limits

- The grade compares what happened with what is typical, so a single press's value includes the luck of how it went; averages over many presses (player, team, zone) are the measurement, single presses are illustrations.
- Defenders choose when to press, so "typical play from the same situation" is not a randomized comparison; the situation features (ball, cover, time, defenders goal-side, phase) remove only what they describe.
- SkillCorner's engagements include light pressure as well as full presses ("no flagged outcome" is most of them).
