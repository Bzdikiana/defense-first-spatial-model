"""
Pre-registered confirmation on the open DFL/Sportec data
(reports/preregistration.md): does the defense's cover, with its own
coefficient, predict shots beyond a flexible model of where the ball is?

Reads the frame tables from scripts/09_dfl_threat.py. The same models run on
the 20 SkillCorner matches as a reference (those data were already seen).
Writes reports/w6_confirmation.md and data/derived/dfl/confirmation.json.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dsv import defense as DF, metrics as M  # noqa: E402

DATA_DIR = Path(os.environ.get("DSV_DATA_DIR", ROOT / "data"))
DERIVED, REPORTS = DATA_DIR / "derived", ROOT / "reports"
DFL_DIR = DERIVED / "dfl"
HORIZON_S, LOOKBACK_S = 5.0, 2.0
BOX_X, BOX_Y = 52.5 - 16.5, 20.16

_spec = importlib.util.spec_from_file_location("s07", ROOT / "scripts" / "07_validation.py")
s07 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s07)
lomo, boot_diff = s07.lomo, s07.boot_diff


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def load_dfl():
    F = [pd.read_csv(p) for p in sorted(DFL_DIR.glob("*_frames.csv.gz"))]
    P = [pd.read_csv(p) for p in sorted(DFL_DIR.glob("*_possessions.csv"))]
    S = [pd.read_csv(p).assign(match_id=p.name.split("_")[0]) for p in sorted(DFL_DIR.glob("*_shots.csv"))]
    return pd.concat(F, ignore_index=True), pd.concat(P, ignore_index=True), pd.concat(S, ignore_index=True)


def load_skillcorner():
    frames = s07.load_frames()
    frames["rd_xg_here"] = s07.reachable_xg_here(frames)
    return frames, pd.read_csv(DERIVED / "threat_possessions.csv")


def frame_table(frames, poss, fps):
    f = frames.merge(poss[["match_id", "possession_id", "first_shot_frame"]], on=["match_id", "possession_id"])
    lead = (f["first_shot_frame"] - f["frame"]) / fps
    f = f[~(lead < 0)].copy()
    f["y"] = lead.loc[f.index].between(0, HORIZON_S, inclusive="right").astype(int)
    f = f.dropna(subset=["tc", "rd", "cover", "tc_xg_here", "rd_xg_here", "ball_x", "ball_y"])
    f = DF.add_cover_columns(f).sort_values(["match_id", "possession_id", "frame"]).reset_index(drop=True)
    f["in_box"] = (f["ball_x"] > BOX_X) & (f["ball_y"].abs() < BOX_Y)
    box_next = np.zeros(len(f), int)
    fr, ib = f["frame"].to_numpy(), f["in_box"].to_numpy().astype(int)
    for idx in f.groupby(["match_id", "possession_id"]).indices.values():
        c = np.concatenate([[0], np.cumsum(ib[idx])])
        hi = np.searchsorted(fr[idx], fr[idx] + HORIZON_S * fps, side="right")
        box_next[idx] = (c[hi] - c[np.arange(len(idx)) + 1]) > 0
    f["box_next"] = box_next
    return f


def possession_table(frames_with_cover, poss, fps):
    p = DF.possession_moments(frames_with_cover, poss, fps, lookback_s=LOOKBACK_S)
    return p.dropna(subset=["cover_at_peak", "cover_xg_here_at_peak"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def verdict(b, coef):
    if b["dll_hi"] < 0 and coef < 0:
        return "CONFIRMED"
    return "consistent, not confirmed" if b["dll"] < 0 else "not confirmed"


def frame_test(f, y_col, cover_col, with_auc=False):
    y = f[y_col].to_numpy()
    X0 = DF.design(f, DF.FRAME_CONTROLS, None)
    X1 = DF.design(f, DF.FRAME_CONTROLS, cover_col)
    p0, p1 = lomo(f, X0, y), lomo(f, X1, y)
    b = boot_diff(f, y, p1, p0, with_auc=with_auc)
    coef = float(M.fit_logistic(X1, y)[-1])
    return dict(n=len(f), events=int(y.sum()), ll0=M.log_loss(y, p0), ll1=M.log_loss(y, p1),
                auc0=M.auc(y, p0), auc1=M.auc(y, p1), coef=coef, **b, verdict=verdict(b, coef))


def possession_test(p, cover_col):
    y = p["y"].to_numpy()
    X0, X1 = DF.possession_design(p, None), DF.possession_design(p, cover_col)
    p0, p1 = lomo(p, X0, y), lomo(p, X1, y)
    b = boot_diff(p, y, p1, p0)
    coef = float(M.fit_logistic(X1, y)[-1])
    return dict(n=len(p), events=int(y.sum()), ll0=M.log_loss(y, p0), ll1=M.log_loss(y, p1),
                auc0=M.auc(y, p0), auc1=M.auc(y, p1), coef=coef, **b, verdict=verdict(b, coef))


def transport_test(train, test):
    """Fit M0 and M1 on SkillCorner, score DFL without refitting."""
    y_tr, y_te = train["y"].to_numpy(), test["y"].to_numpy()
    c0 = M.fit_logistic(DF.design(train, DF.FRAME_CONTROLS, None), y_tr)
    c1 = M.fit_logistic(DF.design(train, DF.FRAME_CONTROLS, "cover"), y_tr)
    p0 = M.predict_logistic(c0, DF.design(test, DF.FRAME_CONTROLS, None))
    p1 = M.predict_logistic(c1, DF.design(test, DF.FRAME_CONTROLS, "cover"))
    b = boot_diff(test, y_te, p1, p0, with_auc=False)
    return dict(n=len(test), events=int(y_te.sum()), ll0=M.log_loss(y_te, p0), ll1=M.log_loss(y_te, p1),
                auc0=M.auc(y_te, p0), auc1=M.auc(y_te, p1), coef=float(c1[-1]), **b, verdict=verdict(b, c1[-1]))


def exploratory(f, p):
    """Not part of the confirmation: product form, averaged cover, and the
    descriptive table."""
    out = {}
    y = f["y"].to_numpy()
    X0 = DF.design(f, DF.FRAME_CONTROLS, None)
    p0 = lomo(f, X0, y)
    pt = lomo(f, np.column_stack([X0, np.log(np.maximum(f["tc"].to_numpy(), DF.EPS))]), y)
    out["frame: + log TC (product form) vs M0"] = boot_diff(f, y, pt, p0, with_auc=False)
    yp = p["y"].to_numpy()
    Q0 = DF.possession_design(p, None)
    q0 = lomo(p, Q0, yp)
    qt = lomo(p, np.column_stack([Q0, np.log(np.maximum(p["peak_tc_early"].to_numpy(), DF.EPS))]), yp)
    qm = lomo(p, np.column_stack([Q0, DF.logit(p["mean_cover_xg_here_early"])]), yp)
    out["possession: + log peak TC (product form) vs M0"] = boot_diff(p, yp, qt, q0, with_auc=False)
    out["possession: + mean cover (xG) over the possession vs M0"] = boot_diff(p, yp, qm, q0, with_auc=False)
    p = p.copy()
    p["danger"] = pd.qcut(p["peak_rd_xg_here_early"], 4, labels=["Q1 (least)", "Q2", "Q3", "Q4 (most dangerous)"])
    p["cover_t"] = pd.qcut(p["cover_at_peak"], 3, labels=["low cover", "mid", "high cover"])
    rate = p.pivot_table(index="danger", columns="cover_t", values="y", aggfunc="mean", observed=False)
    n = p.pivot_table(index="danger", columns="cover_t", values="y", aggfunc="size", observed=False)
    return out, rate, n


def run_all(f, p):
    box = f[~f["in_box"] & (f["y"] == 0)].reset_index(drop=True)
    return {
        "H1": frame_test(f, "y", "cover"),
        "H2": possession_test(p, "cover_at_peak"),
        "H3a": frame_test(f, "y", "cover_xg_here"),
        "H3b": possession_test(p, "cover_xg_here_at_peak"),
        "H4": frame_test(box, "box_next", "cover"),
    }


# --------------------------------------------------------------------------- #
def main():
    frames, poss, shots = load_dfl()
    f = frame_table(frames, poss, fps=25.0)
    p = possession_table(f, poss, fps=25.0)
    checks = dict(matches=int(f["match_id"].nunique()), shots=len(shots), shots_synced=int(shots["matched"].sum()),
                  possessions=len(poss), possessions_with_shot=int(poss["shot"].sum()),
                  frames=len(f), possessions_tested=len(p))
    print("DFL", checks, flush=True)
    res = run_all(f, p)
    print("DFL done", {k: v["verdict"] for k, v in res.items()}, flush=True)

    sc_frames, sc_poss = load_skillcorner()
    sf = frame_table(sc_frames, sc_poss, fps=10.0)
    sp = possession_table(sf, sc_poss, fps=10.0)
    res["H5"] = transport_test(sf, f)
    ref = run_all(sf, sp)
    print("SkillCorner reference done", flush=True)
    expl, rate, n = exploratory(f, p)

    js = {"checks": checks, "dfl": res, "skillcorner_reference": ref,
          "exploratory": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in expl.items()}}
    (DFL_DIR / "confirmation.json").write_text(json.dumps(js, indent=2, default=float), encoding="utf-8")
    write_report(checks, res, ref, expl, rate, n)
    for k, v in res.items():
        print(k, v["verdict"], round(v["dll"], 5), [round(v["dll_lo"], 5), round(v["dll_hi"], 5)], flush=True)


LABELS = {
    "H1": "**H1 (primary)** frame: shot within 5 s, + cover (xT)",
    "H2": "H2 possession: shot, + cover (xT) at peak danger",
    "H3a": "H3a frame: shot within 5 s, + cover (xG from here)",
    "H3b": "H3b possession: shot, + cover (xG from here) at peak danger",
    "H4": "H4 frame: box entry within 5 s, + cover (xT)",
    "H5": "H5 fitted on SkillCorner, scored on DFL: + cover (xT)",
}


def _row(k, v):
    auc = f"{v['auc0']:.3f} → {v['auc1']:.3f}"
    return (f"| {LABELS[k]} | {v['n']:,} ({v['events']:,}) | {v['ll0']:.4f} → {v['ll1']:.4f} | "
            f"{v['dll']:+.4f} [{v['dll_lo']:+.4f}, {v['dll_hi']:+.4f}] | {auc} | {v['coef']:+.2f} | {v['verdict']} |")


def write_report(checks, res, ref, expl, rate, n):
    L = []
    w = L.append
    w("# Confirmation on independent data: the defense's cover\n")
    w("Pre-registered in [preregistration.md](preregistration.md) (committed before this was run). "
      "Data: the seven open DFL/Sportec matches (Bassek et al. 2025, CC BY 4.0), optical tracking. Every "
      "ingredient is frozen from the SkillCorner fits. Script: `scripts/10_confirmation.py`.\n")
    w("Each test adds the defense's cover (logit) to a model of where the ball is and compares held-out "
      "log loss, leave-one-match-out; negative differences favour cover. Intervals are 95% match bootstraps "
      "(7 matches for DFL). The coefficient is from the model fitted on all matches; negative means more "
      "cover, fewer shots.\n")
    w("## Pre-registered tests (DFL)\n")
    w("| test | units (events) | log loss without → with cover | difference [95%] | AUC | cover coefficient | verdict |")
    w("|---|---|---|---|---|---|---|")
    for k in ["H1", "H2", "H3a", "H3b", "H4", "H5"]:
        w(_row(k, res[k]))
    w("\n## Same models on the 20 SkillCorner matches (reference; these data were already seen)\n")
    w("| test | units (events) | log loss without → with cover | difference [95%] | AUC | cover coefficient | fit |")
    w("|---|---|---|---|---|---|---|")
    for k in ["H1", "H2", "H3a", "H3b", "H4"]:
        w(_row(k, ref[k]).rsplit("|", 2)[0] + "| - |")
    w("\n## Exploratory (DFL, not part of the confirmation)\n")
    w("| comparison | difference in log loss [95%] |")
    w("|---|---|")
    for k, v in expl.items():
        w(f"| {k} | {v['dll']:+.4f} [{v['dll_lo']:+.4f}, {v['dll_hi']:+.4f}] |")
    w("\nShot or goal rate (%) by danger quartile and cover at the moment of peak danger (possessions):\n")
    t = (rate * 100).round(1).astype(str) + " (n=" + n.astype(int).astype(str) + ")"
    w("| danger | " + " | ".join(map(str, t.columns)) + " |")
    w("|---|" + "---|" * len(t.columns))
    for idx, row in t.iterrows():
        w(f"| {idx} | " + " | ".join(row.values) + " |")
    w("\n## Data checks\n")
    w(f"- {checks['matches']} matches, {checks['possessions']:,} possessions "
      f"({checks['possessions_with_shot']} with a shot), {checks['frames']:,} frames tested, "
      f"{checks['possessions_tested']:,} possessions with a frame at least 2 s before the shot or end.")
    w(f"- {checks['shots_synced']} of {checks['shots']} shots synchronized to the tracking (shooter within "
      "2.5 m of the ball); the rest are not used. Per-match details: `data/derived/dfl/structure.csv`.")
    (REPORTS / "w6_confirmation.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
