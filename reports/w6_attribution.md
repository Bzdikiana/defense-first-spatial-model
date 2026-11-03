# Does blame point at the beaten defender?

Each defender's share of the cover is the part of the reachable shooting danger (xG-from-here map) they control; the shares add up to the team's cover (`dsv/attribution.py`). Blame splits a loss of cover across the defenders whose share fell. SkillCorner's dynamic events independently flag on-ball engagements in which the pressing defender was beaten, and name the defender. Script: `scripts/11_attribution.py`.

15,672 engagements during an opponent possession, from the start of the engagement to 1 s after it (or the end of the possession). Means with 95% match-bootstrap intervals:

| SkillCorner outcome | engagements | pressing defender's lost share (pp) | blamed most | their part of the team's drop | team's uncovered share (log change) |
|---|---|---|---|---|---|
| beaten | 327 | +2.4 [+1.6, +3.2] | 19% [14%, 24%] | 16% | +0.08 [+0.05, +0.12] |
| stopped danger | 204 | +2.8 [+2.4, +3.2] | 21% [18%, 24%] | 19% | +0.09 [+0.04, +0.13] |
| reduced danger | 458 | +1.8 [+1.3, +2.2] | 17% [15%, 20%] | 17% | +0.09 [+0.06, +0.12] |
| forced backward | 927 | +1.4 [+1.0, +1.7] | 10% [8%, 13%] | 12% | +0.21 [+0.19, +0.22] |
| no flagged outcome | 13,756 | +1.6 [+1.5, +1.6] | 22% [21%, 22%] | 13% | -0.00 [-0.01, +0.01] |

![attribution](../figs/w6_attribution.png)

**Separating beaten from other engagements** (AUC, 0.5 = no information):

- pressing defender's own loss of cover: 0.536
- team's loss of cover: 0.576
- ball danger gained: 0.833

Logistic model of "beaten", held out by match: the ball's gain in danger, the team's change in uncovered share and the defender's starting share give log loss 0.0793; adding the pressing defender's own loss of cover gives 0.0780 (difference -0.0013 [-0.0021, -0.0005]; AUC +0.013 [+0.008, +0.018]).

## Reading

- **The team side.** After a beaten press the share of the shooting danger the defense leaves uncovered changes by +0.08 (log scale), against -0.00 when no outcome is flagged; the largest change follows "forced backward" (+0.21). The team's loss of cover separates beaten presses from the rest with AUC 0.58.
- **The individual side.** The pressing defender's own lost share separates "beaten" with AUC 0.54, and they are the defender blamed most in 19% of beaten presses against 22% of presses with no flagged outcome. Blaming the individual presser by lost cover does not single out the beaten ones.
- **Where the lost cover goes.** After a beaten press, the presser holds 16% of the team's drop in cover on average; the rest belongs to teammates, since the space that opens is behind the presser. Judging the press itself needs the comparison with typical play in [w7_pressing.md](w7_pressing.md), not attribution.

SkillCorner's flags are themselves computed from tracking geometry, so agreement shows the two constructions see the same events, not that either is ground truth.

