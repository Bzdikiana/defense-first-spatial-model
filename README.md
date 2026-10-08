# Defensive Space Value (dsv)

A defense-first spatial model on SkillCorner open data (A-League 2024/25, 20 matches):
how much dangerous space a team concedes over time (Threat Conceded), which defenders
are responsible, and whether specific defensive actions were the right call.
The design is in [analytics-cup-brief.md](analytics-cup-brief.md).

## Setup

```
py -m pip install -e .[dev]
py scripts/00_fetch_data.py      # ~1.8 GB into data/ (tracking files are git-LFS on GitHub)
py scripts/01_data_audit.py      # Week 1 audit -> reports/w1_data_audit.md, figs/
py scripts/02_pitch_control.py   # Week 2 pitch control checks -> reports/w2_pitch_control.md
py -m pytest
```

## Layout

| path | what |
|---|---|
| `dsv/data.py` | fetch, parse tracking / events / phases, direction normalization, kinematics, v95, coverage audit, event alignment |
| `dsv/pitchcontrol.py` | Spearman pitch control, per-player vmax, renormalized leave-one-out |
| `dsv/value.py` | Week 3: danger surface and accessibility weight |
| `dsv/threat.py` | Week 4: Threat Conceded and aggregation |
| `dsv/attribution.py` | Week 6: per-defender suppression and beaten attribution |
| `dsv/counterfactual.py` | Week 7: alternative actions, DDV, pressing cost-benefit |
| `dsv/viz.py` | pitch, player and surface plots (animations in Week 9) |
| `scripts/` | runnable pipeline steps |
| `reports/`, `figs/` | generated write-ups and figures |

## Data conventions

- Tracking is SkillCorner's *extrapolated* file: all 22 players in every frame, with an
  `is_detected` flag per position (about 57% detected overall).
- After `normalize_direction`, the home team attacks +x in both periods.
- Event coordinates have the event's team attacking +x; `orient_events` maps them to the
  tracking frame. Event and tracking frame numbers share one clock (no offset).
- There is no shot xG in the open data.
