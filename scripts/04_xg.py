"""
An xG model fitted on the SkillCorner open-data shots alone.

The Analytics Cup allows only the SkillCorner data provided for the event, so
the model is fitted on the competition's own shots (penalties excluded), with
features from the tracking at the shot frame.

1. SkillCorner shots with features from tracking (distance, angle, header,
   blockers in the shooting triangle, keeper position, nearest opponent).
2. Leave-one-match-out comparison of a geometry-only model and the full
   model at three ridge strengths; the best on held-out log loss is kept.
3. Final fit on all shots; per-shot xG (in-sample and held-out by match),
   calibration and a split by how many nearby opponents are extrapolated.

Saves data/derived/skillcorner_shots_xg.csv, data/derived/xg_model.json, a
report (reports/w3_xg.md) and a figure. Run from the repo root:
py scripts/04_xg.py
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
PENALTY_XG = 0.76        # the usual penalty conversion rate; penalties are not modelled
L2S = (1e-4, 1e-2, 1e-1)


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


def lomo_xg(ev, feats, l2):
    oof = np.zeros(len(ev))
    m = ev["match_id"].to_numpy()
    for mid in np.unique(m):
        tr, te = m != mid, m == mid
        oof[te] = xg.predict_xg(xg.fit_xg(ev[tr], feats, l2=l2), ev[te])
    return oof


def poisson_binomial_interval(p, q=(0.025, 0.975), n=20000, seed=0):
    """Interval for the number of goals if the xG values were exactly right."""
    rng = np.random.default_rng(seed)
    sims = (rng.random((n, len(p))) < np.asarray(p)[None]).sum(1)
    return np.quantile(sims, q)


def main():
    sc = skillcorner_shots()
    ev = sc[~sc["likely_penalty"].astype(bool)].reset_index(drop=True)
    y = ev["goal"].astype(int).to_numpy()
    print(f"SkillCorner {len(ev)} non-penalty shots / {int(y.sum())} goals, {ev['match_id'].nunique()} matches", flush=True)

    rows, oofs = [], {}
    for name, feats in (("geometry only", GEOMETRY), ("full model", xg.FEATURES)):
        for l2 in L2S:
            p = lomo_xg(ev, feats, l2)
            oofs[(name, l2)] = p
            rows.append(dict(model=name, l2=l2, **M.summary(y, p), xg_sum=float(p.sum())))
    base = np.zeros(len(ev))
    for mid in ev["match_id"].unique():
        te = (ev["match_id"] == mid).to_numpy()
        base[te] = y[~te].mean()
    rows.append(dict(model="constant (goal rate of the other matches)", l2=np.nan, **M.summary(y, base),
                     xg_sum=float(base.sum())))
    table = pd.DataFrame(rows)
    best = table.dropna(subset=["l2"]).sort_values("log_loss").iloc[0]
    feats = GEOMETRY if best["model"] == "geometry only" else xg.FEATURES
    final = xg.fit_xg(ev, feats, l2=float(best["l2"]))
    oof = oofs[(best["model"], best["l2"])]
    lo, hi = poisson_binomial_interval(oof)
    ext = (ev["extrap_share"] > 0).to_numpy()
    strata = pd.DataFrame([dict(stratum=s, **M.summary(y[m], oof[m]), xg_sum=float(oof[m].sum()), goals=int(y[m].sum()))
                           for s, m in (("no nearby opponent extrapolated", ~ext),
                                        ("some nearby opponents extrapolated", ext))])

    out = sc.copy()
    out["xg"] = xg.predict_xg(final, out)
    out["xg_heldout"] = np.nan
    out.loc[~out["likely_penalty"].astype(bool), "xg_heldout"] = oof
    out.loc[out["likely_penalty"].astype(bool), ["xg", "xg_heldout"]] = PENALTY_XG
    out.to_csv(DERIVED / "skillcorner_shots_xg.csv", index=False)
    (DERIVED / "xg_model.json").write_text(json.dumps(
        dict(features=final["features"], extra_terms=["angle^2", "dist*header"], l2=float(best["l2"]),
             coef=[float(c) for c in final["coef"]], n_shots=int(len(ev)), n_goals=int(y.sum()),
             data="SkillCorner open data, A-League 2024/25, 20 matches"), indent=2))

    fig, ax = plt.subplots(figsize=(5.4, 4.3))
    for (name, p), c in zip({"geometry only": oofs[("geometry only", best["l2"])],
                             f"{best['model']} (kept)": oof}.items(), ["#8a8f98", "#2b6cb0"]):
        bins = np.quantile(p, np.linspace(0, 1, 7))
        b = np.clip(np.searchsorted(bins, p, side="right") - 1, 0, 5)
        d = pd.DataFrame({"b": b, "p": p, "y": y}).groupby("b").agg(p=("p", "mean"), y=("y", "mean"))
        ax.plot(d["p"], d["y"], "o-", ms=4, color=c, label=name)
    ax.plot([0, 0.6], [0, 0.6], "--", color="#bbbbbb", lw=1)
    ax.set_xlabel("held-out xG (sextiles)"); ax.set_ylabel("observed goal rate")
    ax.set_title(f"SkillCorner shots, leave-one-match-out (n={len(ev)})", fontsize=10)
    ax.legend(fontsize=8, frameon=False); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIGS / "w3_xg_calibration.png", dpi=130); plt.close(fig)

    write_report(ev, table, best, final, strata, (lo, hi), oof)
    print(table.round(4).to_string(index=False))
    print("kept:", best["model"], best["l2"])
    print("OPEN_DATA_XG coef:", [float(c) for c in final["coef"]])


def write_report(ev, table, best, final, strata, interval, oof):
    L = []
    w = L.append
    y = ev["goal"].astype(int)
    w("# xG for the SkillCorner open-data shots\n")
    w("The SkillCorner open data has shots but no xG, and the Analytics Cup allows only the SkillCorner data "
      f"provided for the event, so the model is fitted on the competition's own shots: {len(ev)} non-penalty "
      f"shots and {int(y.sum())} goals in {ev['match_id'].nunique()} A-League matches, with features from the "
      "tracking at the shot frame. Code: `dsv/xg.py`; script: `scripts/04_xg.py`.\n")
    w("## Features\n")
    w("Shot distance and log-distance to the goal centre, visible goal angle (and its square), header (and "
      "distance x header), outfield opponents inside the shooter-posts triangle, the keeper's distance from "
      "the shot line and from the goal centre, distance to the nearest opponent, and opponents within 3 m. "
      "The geometry-only model keeps the first four. Logistic regression with a ridge penalty.\n")
    w("## Which model (held out by match)\n")
    w("| model | ridge | log loss | Brier | AUC | ECE | total xG |")
    w("|---|---|---|---|---|---|---|")
    for _, r in table.iterrows():
        l2 = "" if pd.isna(r["l2"]) else f"{r['l2']:g}"
        w(f"| {r['model']} | {l2} | {r['log_loss']:.4f} | {r['brier']:.4f} | {r['auc']:.3f} | {r['ece']:.3f} | "
          f"{r['xg_sum']:.1f} |")
    lo, hi = interval
    w(f"\nKept: **{best['model']}**, ridge {best['l2']:g} (lowest held-out log loss). Held-out xG totals "
      f"{oof.sum():.1f} for {int(y.sum())} goals; if the xG values were exactly right, 95% of samples like this "
      f"would produce {lo:.0f}–{hi:.0f} goals. With {int(y.sum())} goals, differences in the second decimal of "
      "AUC are noise.\n")
    w("**Extrapolated positions.** Shots split by whether any opponent within 20 m of the shooter was "
      "extrapolated at the shot frame (held-out xG):\n")
    w("| shots | n | goals | xG | log loss | AUC |")
    w("|---|---|---|---|---|---|")
    for _, r in strata.iterrows():
        w(f"| {r['stratum']} | {r['n']} | {r['goals']} | {r['xg_sum']:.1f} | {r['log_loss']:.4f} | {r['auc']:.3f} |")
    w("\n![xG calibration](../figs/w3_xg_calibration.png)\n")
    w("## Caveats\n")
    w(f"- {int(y.sum())} goals is a small sample for an xG model; the ridge penalty and the held-out choice keep "
      "it simple. The model is used for its shape (how danger rises toward goal), as the danger map and as the "
      "value of a shot in expected threat, more than for single-shot estimates.")
    w("- SkillCorner positions at the shot frame can be extrapolated (split above).")
    w(f"- Penalties are excluded from the model and set to {PENALTY_XG} xG. SkillCorner does not label them; "
      "shots from the spot at the start of a short possession are treated as penalties.")
    w("- Goals are possessions followed by `goal_for`. A few goals in the final scores are not on a shot-tagged "
      "possession and so are missing here (see the README).\n")
    w("Outputs: `data/derived/skillcorner_shots_xg.csv` (one row per shot, in-sample and held-out xG), "
      "`data/derived/xg_model.json` (coefficients).")
    (REPORTS / "w3_xg.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
