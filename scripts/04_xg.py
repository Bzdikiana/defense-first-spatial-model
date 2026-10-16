"""
An xG model for SkillCorner open-data shots, trained on StatsBomb open data.

1. StatsBomb shots (5 seasons, penalties excluded) with features computed from
   freeze frames; SkillCorner shots with the same features from tracking.
2. Leave-one-league-out validation on StatsBomb: a geometry-only model
   (distance, angle, header) vs the full model (plus blockers, keeper
   position, pressure), with StatsBomb's own xG as a reference.
3. Final fit on all StatsBomb shots, applied to the 526 SkillCorner shots:
   total xG vs goals, log loss and AUC vs geometry-only, calibration, and a
   split by how many nearby opponents are extrapolated.
4. Saves per-shot xG for SkillCorner (data/derived/skillcorner_shots_xg.csv),
   the coefficients (data/derived/xg_model.json), a report and figures.

Run from the repo root:  py scripts/04_xg.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dsv import data as D, metrics as M, xg  # noqa: E402

DATA_DIR = Path(os.environ.get("DSV_DATA_DIR", ROOT / "data"))
FIGS, REPORTS, DERIVED = ROOT / "figs", ROOT / "reports", DATA_DIR / "derived"
GEOMETRY = ["dist", "log_dist", "angle", "header"]
PENALTY_XG = 0.76
LEAGUES = {(2, 27): "Premier League 15/16", (11, 27): "La Liga 15/16", (12, 27): "Serie A 15/16",
           (7, 27): "Ligue 1 15/16", (1238, 108): "Indian Super League 21/22"}


def skillcorner_shots() -> pd.DataFrame:
    path = DERIVED / "skillcorner_shots.csv"
    if path.exists() and not os.environ.get("REBUILD"):
        return pd.read_csv(path)
    parts = []
    for mid in D.list_open_matches(DATA_DIR):
        md = DATA_DIR / "matches" / str(mid)
        meta = D.load_match_metadata(md, mid)
        pl, ball = D.normalize_direction(*D.load_tracking(md, mid, meta), meta)
        parts.append(xg.skillcorner_shots(D.load_events(md, mid), pl, ball, meta))
    sc = pd.concat(parts, ignore_index=True)
    sc.to_csv(path, index=False)
    return sc


def poisson_binomial_interval(p, q=(0.025, 0.975), n=20000, seed=0):
    """Interval for the number of goals if the xG values were exactly right."""
    rng = np.random.default_rng(seed)
    sims = (rng.random((n, len(p))) < np.asarray(p)[None]).sum(1)
    return np.quantile(sims, q)


def calibration_panel(ax, y, preds: dict, title):
    for (name, p), c in zip(preds.items(), ["#8a8f98", "#2b6cb0", "#dd8a2c"]):
        bins = np.quantile(p, np.linspace(0, 1, 9))
        b = np.clip(np.searchsorted(bins, p, side="right") - 1, 0, 7)
        df = pd.DataFrame({"b": b, "p": p, "y": y}).groupby("b").agg(p=("p", "mean"), y=("y", "mean"))
        ax.plot(df["p"], df["y"], "o-", ms=4, color=c, label=name)
    lim = max(0.6, float(np.max([np.max(p) for p in preds.values()])))
    ax.plot([0, lim], [0, lim], "--", color="#bbbbbb", lw=1)
    ax.set_xlabel("predicted xG (octiles)"); ax.set_ylabel("observed goal rate")
    ax.set_title(title, fontsize=10); ax.legend(fontsize=7.5, frameon=False); ax.grid(alpha=.3)


def main():
    sb = xg.fetch_statsbomb_shots(DERIVED / "statsbomb_shots.csv")
    sb["league"] = [LEAGUES[(c, s)] for c, s in zip(sb["competition_id"], sb["season_id"])]
    sc = skillcorner_shots()
    print(f"StatsBomb {len(sb):,} shots / {int(sb.goal.sum()):,} goals; "
          f"SkillCorner {len(sc)} shots / {int(sc.goal.sum())} goals", flush=True)

    # ----- leave-one-league-out on StatsBomb -----------------------------------
    rows, oof = [], {"geometry only": np.zeros(len(sb)), "full model": np.zeros(len(sb))}
    for lg in LEAGUES.values():
        tr, te = sb["league"] != lg, sb["league"] == lg
        for name, feats in (("geometry only", GEOMETRY), ("full model", xg.FEATURES)):
            mdl = xg.fit_xg(sb[tr], feats)
            oof[name][te.to_numpy()] = xg.predict_xg(mdl, sb[te])
    for name, p in {**oof, "StatsBomb xG (reference)": sb["statsbomb_xg"].to_numpy()}.items():
        for lg in list(LEAGUES.values()) + ["all"]:
            m = np.ones(len(sb), bool) if lg == "all" else (sb["league"] == lg).to_numpy()
            rows.append(dict(model=name, league=lg, **M.summary(sb["goal"][m], p[m]),
                             xg_sum=float(p[m].sum()), goals=int(sb["goal"][m].sum())))
    sb_table = pd.DataFrame(rows)

    # ----- final fit, applied to SkillCorner -----------------------------------
    full = xg.fit_xg(sb, xg.FEATURES)
    geo = xg.fit_xg(sb, GEOMETRY)
    ev = sc[~sc["likely_penalty"]].copy()
    ev["xg"] = xg.predict_xg(full, ev)
    ev["xg_geometry"] = xg.predict_xg(geo, ev)
    base = np.full(len(ev), ev["goal"].mean())
    sc_rows = [dict(model=n, **M.summary(ev["goal"], p), xg_sum=float(p.sum()))
               for n, p in (("full model", ev["xg"]), ("geometry only", ev["xg_geometry"]),
                            ("constant (in-sample goal rate)", base))]
    sc_table = pd.DataFrame(sc_rows)
    lo, hi = poisson_binomial_interval(ev["xg"].to_numpy())
    ext = ev["extrap_share"] > 0
    strata = pd.DataFrame([dict(stratum=s, **M.summary(ev["goal"][m], ev["xg"][m]),
                                xg_sum=float(ev["xg"][m].sum()), goals=int(ev["goal"][m].sum()))
                           for s, m in (("no nearby opponent extrapolated", ~ext),
                                        ("some nearby opponents extrapolated", ext))])

    out = sc.copy()
    out["xg"] = xg.predict_xg(full, out)
    out.loc[out["likely_penalty"], "xg"] = PENALTY_XG
    out.to_csv(DERIVED / "skillcorner_shots_xg.csv", index=False)
    (DERIVED / "xg_model.json").write_text(json.dumps(
        dict(features=full["features"], extra_terms=["angle^2", "dist*header"],
             coef=[float(c) for c in full["coef"]], n_shots=int(len(sb)),
             seasons=[f"{c}/{s}" for c, s in LEAGUES]), indent=2))

    # ----- figures --------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    calibration_panel(axes[0], sb["goal"].to_numpy(),
                      {"StatsBomb xG": sb["statsbomb_xg"].to_numpy(), "full model": oof["full model"],
                       "geometry only": oof["geometry only"]},
                      "StatsBomb, leave-one-league-out")
    calibration_panel(axes[1], ev["goal"].to_numpy(),
                      {"geometry only": ev["xg_geometry"].to_numpy(), "full model": ev["xg"].to_numpy()},
                      f"SkillCorner A-League shots (n={len(ev)})")
    fig.tight_layout(); fig.savefig(FIGS / "w3_xg_calibration.png", dpi=130); plt.close(fig)

    write_report(sb, sc, ev, sb_table, sc_table, strata, full, (lo, hi))
    print(sb_table[sb_table["league"] == "all"].round(4).to_string(index=False))
    print(sc_table.round(4).to_string(index=False))
    print(f"SkillCorner: xG {ev['xg'].sum():.1f} vs {int(ev['goal'].sum())} goals "
          f"(95% if xG were exact: {lo:.0f}-{hi:.0f})")


def write_report(sb, sc, ev, sb_table, sc_table, strata, full, interval):
    L = []
    w = L.append
    w("# xG for SkillCorner open-data shots\n")
    w("The SkillCorner open data has shots but no xG. This model is trained on **StatsBomb open data** "
      f"({len(sb):,} non-penalty shots with freeze frames from five seasons: "
      + ", ".join(LEAGUES.values()) + ") using only features that SkillCorner tracking also provides, "
      f"then applied to the {len(sc)} SkillCorner shots. Code: `dsv/xg.py`; script: `scripts/04_xg.py`. "
      "StatsBomb data is used under its open-data terms, with attribution.\n")
    w("## Features\n")
    w("Shot distance and log-distance to the goal centre, visible goal angle (and its square), header "
      "(and distance x header), outfield opponents inside the shooter-posts triangle, the keeper's "
      "distance from the shot line and from the goal centre, distance to the nearest opponent, and "
      "opponents within 3 m. Logistic regression with a small ridge penalty.\n")
    w("## Validation on StatsBomb (leave one league out)\n")
    w("| model | log loss | Brier | AUC | ECE | xG / goals |")
    w("|---|---|---|---|---|---|")
    for _, r in sb_table[sb_table["league"] == "all"].iterrows():
        w(f"| {r['model']} | {r['log_loss']:.4f} | {r['brier']:.4f} | {r['auc']:.3f} | {r['ece']:.3f} "
          f"| {r['xg_sum']:.0f} / {r['goals']} |")
    w("\nPer held-out league (full model):\n")
    w("| league | shots | log loss | AUC | xG / goals |")
    w("|---|---|---|---|---|")
    for _, r in sb_table[(sb_table["model"] == "full model") & (sb_table["league"] != "all")].iterrows():
        w(f"| {r['league']} | {r['n']:,} | {r['log_loss']:.4f} | {r['auc']:.3f} | {r['xg_sum']:.0f} / {r['goals']} |")
    w("\nStatsBomb's own xG uses more information (shot technique, assist type, body position) and is "
      "shown as a reference, not as a baseline to beat.\n")
    lo, hi = interval
    w("## Applied to the SkillCorner shots\n")
    w(f"{len(ev)} non-penalty shots, {int(ev['goal'].sum())} goals (2 likely penalties are set to "
      f"{PENALTY_XG} xG and left out of these numbers). Total xG **{ev['xg'].sum():.1f}**; if the xG values "
      f"were exactly right, 95% of seasons would produce {lo:.0f}–{hi:.0f} goals.\n")
    w("| model | log loss | Brier | AUC | ECE | total xG |")
    w("|---|---|---|---|---|---|")
    for _, r in sc_table.iterrows():
        w(f"| {r['model']} | {r['log_loss']:.4f} | {r['brier']:.4f} | {r['auc']:.3f} | {r['ece']:.3f} | {r['xg_sum']:.1f} |")
    w("\nThe constant model uses the A-League goal rate itself, so it is an in-sample reference. "
      f"With {int(ev['goal'].sum())} goals these numbers are noisy: treat differences in the second decimal "
      "of AUC as noise.\n")
    w("**Extrapolated positions.** Shots split by whether any opponent within 20 m of the shooter was "
      "extrapolated at the shot frame:\n")
    w("| shots | n | goals | xG | log loss | AUC |")
    w("|---|---|---|---|---|---|")
    for _, r in strata.iterrows():
        w(f"| {r['stratum']} | {r['n']} | {r['goals']} | {r['xg_sum']:.1f} | {r['log_loss']:.4f} | {r['auc']:.3f} |")
    w("\n![xG calibration](../figs/w3_xg_calibration.png)\n")
    w("## Caveats\n")
    isl = sb_table[(sb_table["model"] == "full model") & (sb_table["league"] == "Indian Super League 21/22")].iloc[0]
    w("- Trained on European top flights (2015/16) plus one Indian Super League season; the A-League "
      "check above is the only evidence it transfers. In the held-out lower league (ISL) goals exceed "
      f"xG by {isl['goals'] / isl['xg_sum'] - 1:.0%}, and on the A-League shots by "
      f"{ev['goal'].sum() / ev['xg'].sum() - 1:.0%}: the model may slightly undervalue chances outside "
      "the European top flights.")
    w("- SkillCorner positions at the shot frame can be extrapolated. When any opponent near the shooter "
      "is, the model ranks shots less well and undercounts goals (split above).")
    w("- Penalties are excluded from the model. SkillCorner does not label them; two shots from the spot "
      "with a short possession are treated as penalties.")
    w("- Goals are possessions followed by `goal_for`. Five of the 66 goals in the final scores are not "
      "on a shot-tagged possession and so are missing here (see the README).\n")
    w("Outputs: `data/derived/skillcorner_shots_xg.csv` (one row per SkillCorner shot), "
      "`data/derived/xg_model.json` (coefficients).")
    (REPORTS / "w3_xg.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
