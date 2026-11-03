# Confirmation on independent data: the defense's cover

Pre-registered in [preregistration.md](preregistration.md) (committed before this was run). Data: the seven open DFL/Sportec matches (Bassek et al. 2025, CC BY 4.0), optical tracking. Every ingredient is frozen from the SkillCorner fits. Script: `scripts/10_confirmation.py`.

Each test adds the defense's cover (logit) to a model of where the ball is and compares held-out log loss, leave-one-match-out; negative differences favour cover. Intervals are 95% match bootstraps (7 matches for DFL). The coefficient is from the model fitted on all matches; negative means more cover, fewer shots.

## Pre-registered tests (DFL)

| test | units (events) | log loss without → with cover | difference [95%] | AUC | cover coefficient | verdict |
|---|---|---|---|---|---|---|
| **H1 (primary)** frame: shot within 5 s, + cover (xT) | 109,711 (2,821) | 0.0832 → 0.0821 | -0.0011 [-0.0019, -0.0000] | 0.918 → 0.923 | -0.75 | CONFIRMED |
| H2 possession: shot, + cover (xT) at peak danger | 2,031 (117) | 0.1612 → 0.1579 | -0.0033 [-0.0066, +0.0005] | 0.879 → 0.889 | -0.87 | consistent, not confirmed |
| H3a frame: shot within 5 s, + cover (xG from here) | 109,711 (2,821) | 0.0832 → 0.0824 | -0.0008 [-0.0014, -0.0001] | 0.918 → 0.922 | -0.57 | CONFIRMED |
| H3b possession: shot, + cover (xG from here) at peak danger | 2,031 (117) | 0.1612 → 0.1585 | -0.0027 [-0.0064, +0.0014] | 0.879 → 0.889 | -0.67 | consistent, not confirmed |
| H4 frame: box entry within 5 s, + cover (xT) | 103,796 (7,136) | 0.1665 → 0.1636 | -0.0029 [-0.0043, -0.0014] | 0.903 → 0.908 | -0.87 | CONFIRMED |
| H5 fitted on SkillCorner, scored on DFL: + cover (xT) | 109,711 (2,821) | 0.0830 → 0.0822 | -0.0009 [-0.0016, +0.0002] | 0.922 → 0.927 | -0.94 | consistent, not confirmed |

## Same models on the 20 SkillCorner matches (reference; these data were already seen)

| test | units (events) | log loss without → with cover | difference [95%] | AUC | cover coefficient | fit |
|---|---|---|---|---|---|---|
| **H1 (primary)** frame: shot within 5 s, + cover (xT) | 308,004 (10,028) | 0.0965 → 0.0938 | -0.0027 [-0.0036, -0.0017] | 0.922 → 0.928 | -0.94 | - |
| H2 possession: shot, + cover (xT) at peak danger | 2,931 (416) | 0.2530 → 0.2470 | -0.0060 [-0.0105, -0.0015] | 0.899 → 0.908 | -0.85 | - |
| H3a frame: shot within 5 s, + cover (xG from here) | 308,004 (10,028) | 0.0965 → 0.0935 | -0.0029 [-0.0038, -0.0020] | 0.922 → 0.929 | -0.95 | - |
| H3b possession: shot, + cover (xG from here) at peak danger | 2,931 (416) | 0.2530 → 0.2470 | -0.0060 [-0.0102, -0.0019] | 0.899 → 0.908 | -0.75 | - |
| H4 frame: box entry within 5 s, + cover (xT) | 288,896 (14,360) | 0.1385 → 0.1344 | -0.0041 [-0.0057, -0.0026] | 0.898 → 0.906 | -1.05 | - |

## Exploratory (DFL, not part of the confirmation)

| comparison | difference in log loss [95%] |
|---|---|
| frame: + log TC (product form) vs M0 | -0.0011 [-0.0017, -0.0002] |
| possession: + log peak TC (product form) vs M0 | -0.0026 [-0.0052, +0.0013] |
| possession: + mean cover (xG) over the possession vs M0 | +0.0008 [-0.0030, +0.0061] |

Shot or goal rate (%) by danger quartile and cover at the moment of peak danger (possessions):

| danger | low cover | mid | high cover |
|---|---|---|---|
| Q1 (least) | 0.0 (n=274) | 0.0 (n=161) | 0.0 (n=73) |
| Q2 | 0.6 (n=157) | 0.0 (n=172) | 0.6 (n=179) |
| Q3 | 6.4 (n=125) | 2.4 (n=165) | 3.7 (n=217) |
| Q4 (most dangerous) | 24.8 (n=121) | 18.4 (n=179) | 15.4 (n=208) |

## Data checks

- 7 matches, 2,715 possessions (157 with a shot), 109,711 frames tested, 2,031 possessions with a frame at least 2 s before the shot or end.
- 159 of 171 shots synchronized to the tracking (shooter within 2.5 m of the ball); the rest are not used. Per-match details: `data/derived/dfl/structure.csv`.
