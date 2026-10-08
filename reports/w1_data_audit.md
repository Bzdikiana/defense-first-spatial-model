# Week 1 data audit

Focal match: Brisbane Roar FC vs Perth Glory Football Club (match 1925299). The cross-match sections cover all 20 SkillCorner open-data matches.

## Focal match: format
- Date: 2024-12-21T06:00:00Z
- Final score: Brisbane Roar FC 0 - 1 Perth Glory Football Club
- Pitch: 105 x 68 m; frame rate 10 fps (verified from timestamps)
- Periods: P1 50.4 min (frames 10-30240); P2 51.8 min (frames 30250-61300)
- In-play frames parsed: 47,942; players in tracking: 32 (roster incl. subs: 36)
- Home attacking direction by period: ['right_to_left', 'left_to_right'] (normalized so home attacks +x in both periods)

## Focal match: detection coverage
- Overall position detection rate: **57.9%**; the rest is SkillCorner extrapolation. Ball detection: 80.6%
- Frames with all 22 detected: 0.0%. Players detected per frame: mean 12.7, median 14, range 0-21
- By period: P1 55.1%, P2 60.7%
- By position group, whole match (GKs split out of SkillCorner's 'Other'):
  - Goalkeeper: 18.8% (n=95,884)
  - Central Defender: 46.2% (n=191,768)
  - Full Back: 56.3% (n=183,637)
  - Center Forward: 64.0% (n=143,826)
  - Wide Attacker: 67.2% (n=199,899)
  - Midfield: 72.6% (n=239,710)
- By distance to ball: 2 m 91%, 8 m 88%, 12 m 84%, 18 m 76%, 22 m 63%, 28 m 51%, 32 m 40%, 38 m 31%, 42 m 21%, 48 m 12%, 52 m 6%, 58 m 2%, 62 m 1%, 68 m 0%, 72 m 0%
- Lowest-detection players: Macklin Freke 14%, Cameron Cook 24%, Lucas Herrington 44%, Hosine Bility 46%, Adriano Lebib 47%

## Back-line detection when it matters (all matches pooled)
Whole-match rates are dominated by frames where the ball is at the other end. Restricted to frames where the player's team is **out of possession**, split by where the ball is relative to that player's own goal:

| group | ball in own third | middle third | opp third |
|---|---|---|---|
| Goalkeeper | 63.1% | 4.8% | 0.0% |
| Central Defender | 90.0% | 59.5% | 12.5% |
| Full Back | 85.7% | 64.7% | 30.8% |
| Midfield | 84.8% | 87.9% | 67.1% |
| Wide Attacker | 69.4% | 83.9% | 82.1% |
| Center Forward | 47.2% | 79.5% | 88.3% |

## Event <-> tracking alignment (focal match)
- 5,217 of 5,222 events join to a tracking row on (frame_start, player_id).
- Event coordinates are rotated so the event's own team attacks +x. After `orient_events`, median distance between the event start location and the player's tracking position at frame_start: off_ball_run 0.20 m, on_ball_engagement 0.19 m, passing_option 0.22 m, player_possession 0.19 m
- Player-to-ball distance for possession events is smallest at frame_start +2 frames (1.05 m vs 1.23 m at +0): the ball arriving at the receiver, not a sync offset. **No frame offset to correct** in the native SkillCorner events.
- The open data has **no shot xG**. Shots are `end_type == 'shot'` on player_possession events; phases carry `team_possession_lead_to_shot`.

## Kinematics (focal match)
- Speed from detected runs only (`speed_obs`): median 1.92 m/s, max 12.0 m/s. 57.9% of player-frames have one (detected runs at least one smoothing window long).
- Mean v95 by group and half (m/s):
  - Center Forward: P1 5.15, P2 4.85
  - Central Defender: P1 4.91, P2 4.24
  - Full Back: P1 5.21, P2 4.94
  - Goalkeeper: P1 3.21, P2 2.90
  - Midfield: P1 5.03, P2 4.80
  - Wide Attacker: P1 5.45, P2 5.15
- Fastest player-halves: Joel  Anasmo P2 (5.7), Abdelelah Faisal P1 (5.6), Jack Hingert P2 (5.6)

## Cross-match summary
| match | date | detection | all 22 | CB whole | CB own-3rd def | FB own-3rd def | GK whole | GK own-3rd def | event err (m) | shots | phases→shot |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Brisbane FC v Adelaide United (1874553) | 2024-11-23 | 57.9% | 0.0% | 50.2% | 90.6% | 85.2% | 14.9% | 56.9% | 0.00 | 26 | 46 |
| Auckland FC v Newcastle (1886347) | 2024-11-30 | 59.3% | 0.0% | 49.8% | 93.8% | 91.6% | 14.8% | 59.9% | 0.18 | 27 | 45 |
| Auckland FC v Wellington P FC (1899585) | 2024-12-07 | 55.2% | 0.0% | 48.7% | 91.4% | 85.8% | 15.4% | 57.0% | 0.14 | 27 | 33 |
| Brisbane FC v Perth Glory (1925299) | 2024-12-21 | 57.9% | 0.0% | 46.2% | 93.0% | 89.9% | 18.8% | 71.0% | 0.19 | 30 | 52 |
| Western Sydney v Wellington P FC (1927964) | 2024-12-22 | 61.5% | 0.0% | 53.3% | 90.2% | 86.4% | 16.6% | 64.8% | 0.00 | 30 | 58 |
| CC Mariners v Melbourne City (1953632) | 2024-12-31 | 62.1% | 0.0% | 58.4% | 86.5% | 84.0% | 18.2% | 60.1% | 0.16 | 15 | 17 |
| Melbourne V FC v Western United (1959846) | 2025-01-10 | 57.0% | 0.0% | 51.9% | 89.8% | 83.4% | 21.2% | 66.7% | 0.00 | 34 | 65 |
| Western Sydney v Auckland FC (1986691) | 2025-01-26 | 58.8% | 0.0% | 49.3% | 89.7% | 89.0% | 20.3% | 66.5% | 0.00 | 32 | 49 |
| Sydney FC v Adelaide United (1996435) | 2025-02-01 | 65.8% | 0.0% | 58.0% | 95.3% | 92.3% | 22.9% | 71.7% | 0.17 | 30 | 64 |
| Auckland FC v Macarthur FC (1996436) | 2025-02-01 | 55.4% | 0.0% | 48.7% | 93.7% | 92.6% | 16.1% | 64.5% | 0.00 | 33 | 48 |
| Melbourne City v Macarthur FC (2006229) | 2025-03-07 | 62.9% | 0.0% | 58.0% | 92.1% | 91.5% | 20.7% | 76.5% | 0.16 | 17 | 35 |
| Melbourne V FC v CC Mariners (2006363) | 2025-03-08 | 53.0% | 0.0% | 44.6% | 84.8% | 77.1% | 16.7% | 55.6% | 0.00 | 24 | 40 |
| Perth Glory v Brisbane FC (2007448) | 2025-03-15 | 54.0% | 0.0% | 46.6% | 92.9% | 89.1% | 17.6% | 61.4% | 0.00 | 26 | 38 |
| Auckland FC v CC Mariners (2007721) | 2025-03-16 | 59.6% | 0.0% | 53.2% | 93.8% | 93.4% | 13.7% | 58.1% | 0.00 | 23 | 43 |
| Melbourne City v CC Mariners (2010085) | 2025-04-05 | 55.4% | 0.0% | 48.0% | 86.0% | 76.8% | 18.1% | 60.5% | 0.00 | 15 | 28 |
| Wellington P FC v Melbourne V FC (2011166) | 2025-04-12 | 52.4% | 0.0% | 42.5% | 88.4% | 86.8% | 16.5% | 63.3% | 0.14 | 32 | 55 |
| Western United v Sydney FC (2013725) | 2025-04-27 | 54.0% | 0.0% | 46.7% | 86.6% | 78.9% | 20.6% | 63.4% | 0.24 | 25 | 39 |
| Western United v Auckland FC (2015213) | 2025-05-03 | 52.5% | 0.0% | 43.6% | 88.8% | 83.3% | 17.2% | 65.9% | 0.19 | 22 | 35 |
| Western United v Adelaide United (2016236) | 2025-05-09 | 49.1% | 0.0% | 40.7% | 85.9% | 78.4% | 15.2% | 58.1% | 0.00 | 26 | 31 |
| Melbourne V FC v Auckland FC (2017461) | 2025-05-17 | 51.2% | 0.0% | 43.0% | 84.4% | 76.6% | 15.7% | 60.2% | 0.26 | 32 | 43 |

Detection across matches: mean 56.7%, range 49.1%-65.8%. Shots: 526 in total. Phases leading to a shot: 864 of 8834.

## Figures
- [figs/detected_per_frame.png](../figs/detected_per_frame.png)
- [figs/detection_vs_distance.png](../figs/detection_vs_distance.png)
- [figs/detection_grid.png](../figs/detection_grid.png)
- [figs/detection_over_time.png](../figs/detection_over_time.png)
- [figs/speed_v95.png](../figs/speed_v95.png)
- [figs/detection_by_ball_zone.png](../figs/detection_by_ball_zone.png)

## Derived files
- `data/derived/w1_match_summary.csv`: the cross-match table
- `data/derived/v95_by_player_half.csv`: per-player, per-half v95 for every match (Week 2 vmax input)