"""
Week 5: does Threat Conceded predict shots? The go/no-go test.

Possession level (main test). Unit: a team possession with frames at least
2 s before any shot. Label: the possession produced a shot or a goal. Each
model is a logistic regression with the same controls (log possession
length, first phase of play, mean defenders goal-side, where the possession
started) plus one danger measure taken as its peak at least 2 s before any
shot. Scored out of sample with leave-one-match-out cross-validation.

Pre-declared pass rule: Threat Conceded beats every baseline on held-out log
loss, with the 95% match-bootstrap interval of the difference below zero.

Also: whether TC adds to reachable danger (ball location) in the same model;
frame level (shot within 5 s); sensitivity to the danger surface, the
next-ball model and tracking quality; and possession xG.

Run from the repo root after scripts/06_threat.py:  py scripts/07_validation.py
"""
from __future__ import annotations

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
from dsv import metrics as M  # noqa: E402

DATA_DIR = Path(os.environ.get("DSV_DATA_DIR", ROOT / "data"))
FIGS, REPORTS, DERIVED = ROOT / "figs", ROOT / "reports", DATA_DIR / "derived"
N_BOOT = 2000
LOOKBACK_S = 2.0
HORIZON_S = 5.0
EPS = 1e-6

# danger measures: column at possession level -> label
POSS_MEASURES = {
    "neg_min_dist": "ball distance to goal (closest point)",
    "peak_rd_early": "reachable danger (ball location only)",
    "peak_space_m2_early": "raw space conceded (m², unweighted)",
    "peak_tc_no_access_early": "control × xT, no reachability (Fernández-style)",
    "peak_tc_voronoi_early": "Threat Conceded with nearest-player-wins",
    "peak_tc_early": "Threat Conceded",
}
SENSITIVITY = {
    "peak_tc_early": "Threat Conceded (A-League xT, fitted next-ball model)",
    "peak_tc_singh_xt_early": "with Singh's published xT",
    "peak_tc_xg_here_early": "with xG-from-here as danger",
    "peak_tc_iso_access_early": "with a single-Gaussian next-ball model",
}
FRAME_MEASURES = {
    "neg_dist": "ball distance to goal",
    "rd": "reachable danger (ball location only)",
    "space_m2": "raw space conceded",
    "tc_no_access": "control × xT, no reachability",
    "tc_voronoi": "Threat Conceded with nearest-player-wins",
    "tc": "Threat Conceded",
}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_frames():
    parts = [pd.read_csv(f) for f in sorted((DERIVED / "threat").glob("*_frames.csv.gz"))]
    return pd.concat(parts, ignore_index=True)


def reachable_xg_here(frames):
    """Reachable danger with xG-from-here as the danger map, from the ball
    position alone (no pitch control needed): sum_x A(x | ball) * xG_here(x)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("s06", ROOT / "scripts" / "06_threat.py")
    s06 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s06)
    xg_here = s06.comparisons().values["xg_here"]
    from dsv import data as D, pitchcontrol as PC, value as V
    out = np.zeros(len(frames))
    for mid, idx in frames.groupby("match_id").groups.items():
        meta = D.load_match_metadata(DATA_DIR / "matches" / str(mid), mid)
        L, W = meta.pitch_length, meta.pitch_width
        cells = PC.make_grid(L, W, 2.0).cells
        v = xg_here(cells, L, W)
        idx = np.asarray(idx)
        for s in range(0, len(idx), 2000):
            ii = idx[s:s + 2000]
            bx = frames.loc[ii, "ball_x"].to_numpy()[:, None]
            by = frames.loc[ii, "ball_y"].to_numpy()[:, None]
            lp = V.OPEN_DATA_TRANSITION.logpdf(bx, by, cells[None, :, 0], cells[None, :, 1], L / 2, W / 2)
            a = np.exp(lp - lp.max(axis=1, keepdims=True))
            out[ii] = (a * v).sum(1) / a.sum(1)
    return out


def possession_table(frames, poss):
    """Possessions with controls and early-window baselines computed from
    the frames."""
    f = frames.merge(poss[["match_id", "possession_id", "frame_end", "first_shot_frame"]],
                     on=["match_id", "possession_id"])
    f = f[f["frame"] <= f["first_shot_frame"].fillna(f["frame_end"]) - LOOKBACK_S * 10].copy()
    f["dist_goal"] = np.hypot(52.5 - f["ball_x"], f["ball_y"])
    f["rd_xg_here"] = reachable_xg_here(f.reset_index(drop=True)).tolist() if len(f) else []
    g = f.sort_values("frame").groupby(["match_id", "possession_id"])
    extra = pd.DataFrame({
        "peak_rd_xg_here_early": g["rd_xg_here"].max(),
        "neg_min_dist": -g["dist_goal"].min(),
        "start_x": g["ball_x"].first(),
        "mean_def_goal_side_early": g["def_goal_side"].mean(),
        "mean_def_extrap_early": g["def_extrap_share"].mean(),
    }).reset_index()
    p = poss.merge(extra, on=["match_id", "possession_id"], how="inner").dropna(subset=["peak_tc_early"])
    p["y"] = (p["shot"].astype(bool) | p["goal"].astype(bool)).astype(int)
    p["log_duration"] = np.log(p["duration_s"].clip(lower=0.5))
    p["first_phase"] = p["first_phase"].fillna("none")
    return p


def controls(df, phases):
    X = [df["log_duration"], df["mean_def_goal_side_early"], df["start_x"]]
    X += [(df["first_phase"] == ph).astype(float) for ph in phases[1:]]
    return np.column_stack(X)


def feature(df, col):
    """Log-transform positive, skewed measures; distances stay linear."""
    v = df[col].to_numpy(dtype=float)
    return v if col.startswith("neg_") else np.log(np.maximum(v, EPS))


# --------------------------------------------------------------------------- #
# Leave-one-match-out logistic regression and bootstrap
# --------------------------------------------------------------------------- #
def lomo(df, X, y):
    oof = np.zeros(len(df))
    m = df["match_id"].to_numpy()
    for mid in np.unique(m):
        tr, te = m != mid, m == mid
        oof[te] = M.predict_logistic(M.fit_logistic(X[tr], y[tr]), X[te])
    return oof


def boot_diff(df, y, pa, pb, n=N_BOOT, seed=0, with_auc=True):
    """Held-out log loss (a) - (b) and AUC (a) - (b), 95% match-bootstrap
    intervals. The log-loss bootstrap uses per-match sums; the AUC one
    re-ranks each resample, so it can be skipped for very large tables."""
    p_a, p_b = np.clip(pa, EPS, 1 - EPS), np.clip(pb, EPS, 1 - EPS)
    ll = lambda p: -(y * np.log(p) + (1 - y) * np.log(1 - p))  # noqa: E731
    d = pd.DataFrame({"m": df["match_id"].to_numpy(), "d": ll(p_a) - ll(p_b), "y": y, "a": pa, "b": pb})
    per = d.groupby("m")["d"].agg(["sum", "size"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per), size=(n, len(per)))
    dll = per["sum"].to_numpy()[idx].sum(1) / per["size"].to_numpy()[idx].sum(1)
    out = dict(dll=d["d"].mean(), dll_lo=np.percentile(dll, 2.5), dll_hi=np.percentile(dll, 97.5),
               dauc=M.auc(y, pa) - M.auc(y, pb), dauc_lo=np.nan, dauc_hi=np.nan)
    if with_auc:
        groups = {k: g for k, g in d.groupby("m")}
        ids = np.array(list(groups))
        dauc = []
        for row in idx:
            s = pd.concat([groups[ids[i]] for i in row])
            dauc.append(M.auc(s["y"], s["a"]) - M.auc(s["y"], s["b"]))
        out.update(dauc_lo=np.percentile(dauc, 2.5), dauc_hi=np.percentile(dauc, 97.5))
    return out


def nested_choice(df, candidates, phases):
    """For each held-out match, pick the candidate measure with the lowest
    log loss on the training matches (same model size for all), then predict
    the held-out match with it. Returns out-of-fold predictions and the
    picks per match."""
    y = df["y"].to_numpy()
    C = controls(df, phases)
    m = df["match_id"].to_numpy()
    oof, picks = np.zeros(len(df)), {}
    for mid in np.unique(m):
        tr, te = m != mid, m == mid
        best = None
        for col in candidates:
            X = np.column_stack([C, feature(df, col)])
            coef = M.fit_logistic(X[tr], y[tr])
            ll = M.log_loss(y[tr], M.predict_logistic(coef, X[tr]))
            if best is None or ll < best[0]:
                best = (ll, col, coef, X)
        oof[te] = M.predict_logistic(best[2], best[3][te])
        picks[mid] = best[1]
    return oof, picks


def evaluate(df, measures, phases, base_cols=()):
    """Out-of-fold predictions and scores for controls + each measure."""
    y = df["y"].to_numpy()
    C = controls(df, phases)
    extra = [feature(df, c) for c in base_cols]
    oof = {"controls only": lomo(df, np.column_stack([C] + extra) if extra else C, y)}
    for col in measures:
        oof[col] = lomo(df, np.column_stack([C] + extra + [feature(df, col)]), y)
    scores = pd.DataFrame([dict(model=k, **M.summary(y, p)) for k, p in oof.items()])
    return oof, scores


# --------------------------------------------------------------------------- #
def main():
    frames = load_frames()
    poss = pd.read_csv(DERIVED / "threat_possessions.csv")
    p = possession_table(frames, poss)
    phases = sorted(p["first_phase"].unique())
    y = p["y"].to_numpy()
    print(f"{len(p):,} possessions, {y.sum()} with a shot or goal, {p['match_id'].nunique()} matches", flush=True)

    # ----- main test ----------------------------------------------------------
    oof, scores = evaluate(p, list(POSS_MEASURES), phases)
    tc = oof["peak_tc_early"]
    vs = {lab: boot_diff(p, y, tc, oof[c]) for c, lab in POSS_MEASURES.items() if c != "peak_tc_early"}
    vs["controls only"] = boot_diff(p, y, tc, oof["controls only"])
    passed = all(v["dll_hi"] < 0 for k, v in vs.items() if k != "controls only")

    # does TC add to ball-location danger in the same model?
    oof_rd, _ = evaluate(p, ["peak_tc_early"], phases, base_cols=["peak_rd_early"])
    add_rd = boot_diff(p, y, oof_rd["peak_tc_early"], oof["peak_rd_early"])

    # ----- post hoc (after seeing the main result) ------------------------------
    # A. on top of the best baseline (ball distance), does TC add more than
    #    ball-location danger does? The difference is the defense's contribution.
    oof_d, scores_d = evaluate(p, ["peak_rd_early", "peak_tc_early"], phases, base_cols=["neg_min_dist"])
    over_dist = {
        "+ reachable danger vs ball distance alone": boot_diff(p, y, oof_d["peak_rd_early"], oof_d["controls only"]),
        "+ Threat Conceded vs ball distance alone": boot_diff(p, y, oof_d["peak_tc_early"], oof_d["controls only"]),
        "+ Threat Conceded vs + reachable danger": boot_diff(p, y, oof_d["peak_tc_early"], oof_d["peak_rd_early"]),
    }
    # B. danger surface chosen inside each training fold, scored on the held-out match
    nested, picks = nested_choice(p, ["peak_tc_early", "peak_tc_singh_xt_early", "peak_tc_xg_here_early"], phases)
    nested_vs = boot_diff(p, y, nested, oof["neg_min_dist"])
    # C. with xG-from-here as the danger map: is the gain the defense, or a
    #    better ball-location map? Compare with its own reachable danger.
    oof_x, scores_x = evaluate(p, ["peak_rd_xg_here_early", "peak_tc_xg_here_early"], phases)
    oof_xb, _ = evaluate(p, ["peak_tc_xg_here_early"], phases, base_cols=["peak_rd_xg_here_early"])
    xg_map = {
        "Threat Conceded vs its reachable danger (separately)": boot_diff(
            p, y, oof_x["peak_tc_xg_here_early"], oof_x["peak_rd_xg_here_early"]),
        "reachable danger + Threat Conceded vs reachable danger alone": boot_diff(
            p, y, oof_xb["peak_tc_xg_here_early"], oof_x["peak_rd_xg_here_early"]),
        "reachable danger alone vs ball distance": boot_diff(
            p, y, oof_x["peak_rd_xg_here_early"], oof["neg_min_dist"]),
    }

    # ----- sensitivity ----------------------------------------------------------
    oof_s, scores_s = evaluate(p, list(SENSITIVITY), phases)
    det_med = p["mean_def_extrap_early"].median()
    hi = p[p["mean_def_extrap_early"] <= det_med].reset_index(drop=True)
    _, scores_hi = evaluate(hi, ["peak_rd_early", "peak_tc_early"], phases)

    # ----- xG ---------------------------------------------------------------
    shot = p[p["y"] == 1]
    xg_rho = {lab: float(shot[[c, "xg"]].corr(method="spearman").iloc[0, 1])
              for c, lab in (("peak_tc_early", "Threat Conceded"), ("peak_rd_early", "reachable danger"))}
    p["tc_decile"] = pd.qcut(p["peak_tc_early"], 10, labels=False, duplicates="drop")
    deciles = p.groupby("tc_decile").agg(possessions=("y", "size"), shot_rate=("y", "mean"), xg=("xg", "sum"))
    deciles["xg_share"] = deciles["xg"] / deciles["xg"].sum()

    # ----- frame level ----------------------------------------------------------
    fr = frame_level(frames, poss, phases)

    calibration_figure(p, oof)
    comparison_figure(scores, vs)
    write_report(p, scores, vs, passed, add_rd, scores_s, scores_hi, det_med, xg_rho, deciles, fr,
                 scores_d, over_dist, nested, picks, nested_vs, scores_x, xg_map)
    print(scores.round(4).to_string(index=False))
    print("PASSED" if passed else "NOT PASSED", flush=True)


def frame_level(frames, poss, phases):
    f = frames.merge(poss[["match_id", "possession_id", "first_shot_frame", "duration_s", "first_phase"]],
                     on=["match_id", "possession_id"])
    lead = (f["first_shot_frame"] - f["frame"]) / 10.0
    f = f[~(lead < 0)].copy()
    f["y"] = lead.loc[f.index].between(0, HORIZON_S, inclusive="right").astype(int)
    f["neg_dist"] = -np.hypot(52.5 - f["ball_x"], f["ball_y"])
    f = f.dropna(subset=["tc", "rd"]).reset_index(drop=True)
    y = f["y"].to_numpy()
    C = np.column_stack([f["t_in_possession"], f["def_goal_side"], f["ball_x"]] +
                        [(f["first_phase"].fillna("none") == ph).astype(float) for ph in phases[1:]])
    oof = {"controls only": lomo(f, C, y)}
    for col in FRAME_MEASURES:
        oof[col] = lomo(f, np.column_stack([C, feature(f, col)]), y)
    scores = pd.DataFrame([dict(model=k, **M.summary(y, q)) for k, q in oof.items()])
    vs = {lab: boot_diff(f, y, oof["tc"], oof[c], with_auc=False) for c, lab in FRAME_MEASURES.items() if c != "tc"}
    return dict(n=len(f), rate=y.mean(), scores=scores, vs=vs)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def calibration_figure(p, oof):
    y = p["y"].to_numpy()
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    for col, lab, c in (("controls only", "controls only", "#8a8f98"),
                        ("peak_rd_early", "+ reachable danger", "#dd8a2c"),
                        ("peak_tc_early", "+ Threat Conceded", "#c53030")):
        q = oof[col]
        b = pd.qcut(q, 10, labels=False, duplicates="drop")
        r = pd.DataFrame({"b": b, "p": q, "y": y}).groupby("b").mean()
        ax.plot(r["p"], r["y"], "o-", ms=4, color=c, label=lab)
    lim = max(0.5, float(max(np.max(v) for v in oof.values())))
    ax.plot([0, lim], [0, lim], "--", color="#bbbbbb", lw=1)
    ax.set_xlabel("predicted chance of a shot or goal (held-out match)")
    ax.set_ylabel("observed rate"); ax.grid(alpha=.3)
    ax.set_title("Possession-level calibration (deciles)", fontsize=10); ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(FIGS / "w5_calibration.png", dpi=130); plt.close(fig)


def comparison_figure(scores, vs):
    labels = [POSS_MEASURES[c] for c in POSS_MEASURES if c != "peak_tc_early"] + ["controls only"]
    d = [vs[k] for k in labels]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ys = np.arange(len(labels))[::-1]
    pts = [-v["dll"] for v in d]
    err = np.array([[-v["dll"] - -v["dll_hi"], -v["dll_lo"] - -v["dll"]] for v in d]).T
    ax.errorbar(pts, ys, xerr=err, fmt="o", color="#c53030", capsize=3)
    ax.axvline(0, color="#111", lw=1)
    ax.set_yticks(ys, labels, fontsize=8)
    ax.set_xlabel("held-out log loss improvement of Threat Conceded over each alternative\n"
                  "(right of zero = Threat Conceded better; bars = 95% match bootstrap)")
    ax.grid(axis="x", alpha=.3)
    fig.tight_layout(); fig.savefig(FIGS / "w5_comparison.png", dpi=130); plt.close(fig)


# --------------------------------------------------------------------------- #
def write_report(p, scores, vs, passed, add_rd, scores_s, scores_hi, det_med, xg_rho, deciles, fr,
                 scores_d, over_dist, nested, picks, nested_vs, scores_x, xg_map):
    L = []
    w = L.append
    y = p["y"]
    w("# Does Threat Conceded predict shots?\n")
    w(f"**Result: {'PASSED' if passed else 'NOT PASSED'}** the pre-declared test: Threat Conceded must beat "
      "every baseline on held-out log loss, with the 95% match-bootstrap interval of the difference below "
      "zero.\n")
    w(f"Unit: {len(p):,} team possessions from {p['match_id'].nunique()} matches that have frames at least "
      f"{LOOKBACK_S:.0f} s before any shot; {int(y.sum())} ({y.mean():.1%}) produced a shot or goal. Each model "
      "is a logistic regression with the same controls (log possession length, first phase of play, mean "
      "defenders goal-side, where the possession started) plus one danger measure, taken as its peak at least "
      "2 s before any shot so the shot itself cannot drive it. Every prediction is for a match the model did "
      "not see (leave-one-match-out). Script: `scripts/07_validation.py`.\n")

    w("## Main comparison\n")
    w("| model (controls + ...) | log loss | Brier | AUC | ECE |")
    w("|---|---|---|---|---|")
    for _, r in scores.iterrows():
        lab = POSS_MEASURES.get(r["model"], r["model"])
        lab = f"**{lab}**" if r["model"] == "peak_tc_early" else lab
        w(f"| {lab} | {r['log_loss']:.4f} | {r['brier']:.4f} | {r['auc']:.3f} | {r['ece']:.3f} |")
    w("\nThreat Conceded minus each alternative (negative log loss and positive AUC favour Threat Conceded):\n")
    w("| vs | Δ log loss [95% interval] | Δ AUC [95% interval] |")
    w("|---|---|---|")
    for k, v in vs.items():
        w(f"| {k} | {v['dll']:+.4f} [{v['dll_lo']:+.4f}, {v['dll_hi']:+.4f}] | "
          f"{v['dauc']:+.3f} [{v['dauc_lo']:+.3f}, {v['dauc_hi']:+.3f}] |")
    w("\n![comparison](../figs/w5_comparison.png)\n")

    w("## Does the defense's shape add anything beyond where the ball is?\n")
    w("Reachable danger depends only on the ball's location. Adding Threat Conceded on top of it (same "
      f"controls) changes held-out log loss by {add_rd['dll']:+.4f} [{add_rd['dll_lo']:+.4f}, "
      f"{add_rd['dll_hi']:+.4f}] and AUC by {add_rd['dauc']:+.3f} [{add_rd['dauc_lo']:+.3f}, "
      f"{add_rd['dauc_hi']:+.3f}]. An interval below zero for log loss means the defense's control of "
      "the space adds information that ball location alone does not have.\n")
    w("![calibration](../figs/w5_calibration.png)\n")

    w("## Post hoc: after seeing the main result\n")
    w("These two analyses were added after the pre-declared test failed. They do not change its verdict.\n")
    w("**A. On top of the best baseline.** Controls plus ball distance to goal, then adding ball-location "
      "danger or Threat Conceded. If Threat Conceded adds more than ball-location danger, the extra comes "
      "from the defense's control of the space.\n")
    w("| model (controls + ball distance + ...) | log loss | AUC |")
    w("|---|---|---|")
    for _, r in scores_d.iterrows():
        lab = {"controls only": "nothing more", "peak_rd_early": "reachable danger",
               "peak_tc_early": "Threat Conceded"}[r["model"]]
        w(f"| {lab} | {r['log_loss']:.4f} | {r['auc']:.3f} |")
    w("\n| comparison | Δ log loss [95% interval] | Δ AUC [95% interval] |")
    w("|---|---|---|")
    for k, v in over_dist.items():
        w(f"| {k} | {v['dll']:+.4f} [{v['dll_lo']:+.4f}, {v['dll_hi']:+.4f}] | "
          f"{v['dauc']:+.3f} [{v['dauc_lo']:+.3f}, {v['dauc_hi']:+.3f}] |")
    counts = pd.Series(picks).value_counts()
    names = {"peak_tc_early": "A-League xT", "peak_tc_singh_xt_early": "Singh's xT",
             "peak_tc_xg_here_early": "xG from here"}
    y = p["y"].to_numpy()
    w(f"\n**B. Danger surface chosen fairly.** In each fold the danger surface (A-League xT, Singh's xT or "
      "xG from here) is chosen on the training matches only, then the held-out match is predicted with it. "
      "Chosen: " + ", ".join(f"{names[k]} in {v} of {len(picks)} folds" for k, v in counts.items())
      + f". Held-out log loss {M.log_loss(y, nested):.4f}, AUC {M.auc(y, nested):.3f}. Against ball "
      f"distance to goal: Δ log loss {nested_vs['dll']:+.4f} [{nested_vs['dll_lo']:+.4f}, "
      f"{nested_vs['dll_hi']:+.4f}], Δ AUC {nested_vs['dauc']:+.3f} [{nested_vs['dauc_lo']:+.3f}, "
      f"{nested_vs['dauc_hi']:+.3f}].\n")

    w("**C. Is the xG-from-here gain the defense, or a better ball-location map?** Reachable danger "
      "built with the same map (ball position only, no pitch control) against Threat Conceded built "
      "with it:\n")
    w("| model (controls + ...) | log loss | AUC |")
    w("|---|---|---|")
    for _, r in scores_x.iterrows():
        lab = {"controls only": "nothing more", "peak_rd_xg_here_early": "reachable danger (xG from here)",
               "peak_tc_xg_here_early": "Threat Conceded (xG from here)"}[r["model"]]
        w(f"| {lab} | {r['log_loss']:.4f} | {r['auc']:.3f} |")
    w("\n| comparison | Δ log loss [95% interval] | Δ AUC [95% interval] |")
    w("|---|---|---|")
    for k, v in xg_map.items():
        w(f"| {k} | {v['dll']:+.4f} [{v['dll_lo']:+.4f}, {v['dll_hi']:+.4f}] | "
          f"{v['dauc']:+.3f} [{v['dauc_lo']:+.3f}, {v['dauc_hi']:+.3f}] |")
    w("")

    w("## Sensitivity\n")
    w("Same test, rebuilding Threat Conceded with other ingredients:\n")
    w("| version | log loss | AUC |")
    w("|---|---|---|")
    for _, r in scores_s.iterrows():
        if r["model"] == "controls only":
            continue
        w(f"| {SENSITIVITY[r['model']]} | {r['log_loss']:.4f} | {r['auc']:.3f} |")
    w(f"\n**Tracking quality.** Restricted to possessions where extrapolated players hold at most the median "
      f"share ({det_med:.0%}) of the defense's weighted control:\n")
    w("| model | log loss | AUC |")
    w("|---|---|---|")
    for _, r in scores_hi.iterrows():
        lab = {"peak_rd_early": "+ reachable danger", "peak_tc_early": "+ Threat Conceded"}.get(r["model"], r["model"])
        w(f"| {lab} | {r['log_loss']:.4f} | {r['auc']:.3f} |")
    w("")

    w("## xG\n")
    w("Among possessions with a shot, Spearman correlation with the possession's xG: "
      + "; ".join(f"{k} {v:.2f}" for k, v in xg_rho.items()) + ".\n")
    w("| peak Threat Conceded decile | possessions | shot or goal | share of all xG |")
    w("|---|---|---|---|")
    for d, r in deciles.iterrows():
        w(f"| {int(d) + 1} | {int(r['possessions'])} | {r['shot_rate']:.1%} | {r['xg_share']:.1%} |")
    w("")

    w("## Frame level\n")
    w(f"{fr['n']:,} frames; {fr['rate']:.1%} are followed by a shot within {HORIZON_S:.0f} s. Same approach "
      "(controls: time into the possession, defenders goal-side, ball position, phase), leave-one-match-out:\n")
    w("| model (controls + ...) | log loss | AUC |")
    w("|---|---|---|")
    for _, r in fr["scores"].iterrows():
        lab = FRAME_MEASURES.get(r["model"], r["model"])
        w(f"| {lab} | {r['log_loss']:.4f} | {r['auc']:.3f} |")
    w("\nThreat Conceded minus each alternative (at frame level the AUC difference has no interval):\n")
    w("| vs | Δ log loss [95% interval] | Δ AUC |")
    w("|---|---|---|")
    for k, v in fr["vs"].items():
        w(f"| {k} | {v['dll']:+.4f} [{v['dll_lo']:+.4f}, {v['dll_hi']:+.4f}] | {v['dauc']:+.3f} |")
    w("\nFrames within a possession are strongly correlated; the match bootstrap accounts for this at the "
      "match level.\n")

    w("## Caveats\n")
    w("- The baselines share the same pitch-control and next-ball computations where relevant, so they "
      "isolate one ingredient at a time.")
    w("- Very short possessions (no frame 2 s before a shot or the end) are excluded.")
    w("- Velocities use ±0.3 s of smoothing; the 2 s lookback keeps this well clear of the shot.")
    w("- 20 matches: differences are judged by match-bootstrap intervals, not by point estimates.")
    (REPORTS / "w5_validation.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
