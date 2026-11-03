# Independent confirmation on DFL/Sportec data (research branch only)

This branch keeps the pre-registered confirmation of the two-part model on independent data.
It is **not part of the Analytics Cup submission**: the competition rules allow only the
SkillCorner data provided for the event, so `master` contains none of this.

- **What was tested:** whether the defense's cover, entered with its own coefficient, predicts
  shots beyond a flexible model of where the ball is, on the seven open DFL/Sportec matches
  (Bundesliga and 2. Bundesliga 2022/23, optical tracking; Bassek, Rein, Weber & Memmert
  2025, *Scientific Data* 12:195, CC BY 4.0).
- **The plan:** [reports/preregistration.md](../reports/preregistration.md), committed and
  pushed (commit 3874717) before the analysis was run.
- **The result:** [reports/w6_confirmation.md](../reports/w6_confirmation.md). Primary test
  (shot within 5 s, xT cover) confirmed narrowly, −0.0011 [−0.0019, −0.00002] in held-out
  log loss; the xG-cover and box-entry tests confirmed; the possession-level and
  fitted-on-SkillCorner tests pointed the same way without reaching the bar.
- **Which models:** the confirmation froze the ingredients as they were on 2 November 2026,
  including the earlier xG model trained on StatsBomb open data (and the xT and
  xG-from-here map built from it). The competition version on `master` refits xG on the
  SkillCorner shots only; the confirmation has not been re-run with those models.
- **Code:** `dsv/dfl.py` (loader, event/tracking clock sync, possessions, shots; needs
  `pip install kloppy`), `scripts/09_dfl_threat.py`, `scripts/10_confirmation.py`,
  `tests/test_dfl.py`.
