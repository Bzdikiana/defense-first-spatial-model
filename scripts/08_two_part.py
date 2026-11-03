"""
Week 6: the two-part model on the 20 SkillCorner matches (exploratory: these
analyses came after the Week 5 test failed; the confirmation on independent
data is scripts/10_confirmation.py).

1. Why the product failed: log TC = log RD + log(1 - cover) forces the ball's
   situation and the defense's share to carry the same coefficient. Fitted
   freely, they do not.
2. Cover with its own coefficient, on top of a flexible ball-location model:
   moment level (shot within 5 s; ball into the box within 5 s) and
   possession level (cover at the moment of peak danger), with phase
   controls. Where it matters most: by phase of play and organised defense.
3. Benchmark: SkillCorner's own EPV (in the dynamic events).
4. Defensive engagements (SkillCorner's on-ball engagement events): how ball
   danger and the defense's cover change after a press, by outcome.
5. The fitted two-part model: how much cover changes the chance of a shot,
   and shots prevented by each defense relative to a typical one
   (leave-one-match-out).

Writes reports/w6_two_part.md, figs/w6_*.png and
data/derived/shots_prevented_team_matches.csv.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker  # noqa: E402,F401

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dsv import data as D, defense as DF, metrics as M, pitchcontrol as PC, value as V  # noqa: E402

DATA_DIR = Path(os.environ.get("DSV_DATA_DIR", ROOT / "data"))
DERIVED, REPORTS, FIGS = DATA_DIR / "derived", ROOT / "reports", ROOT / "figs"
FPS, HORIZON_S = 10.0, 5.0
EPS = 1e-6


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s07 = _load("s07", ROOT / "scripts" / "07_validation.py")
rng = np.random.default_rng(0)


def load_skillcorner():
    """Frames of all 20 matches (scripts/06_threat.py) and their possessions."""
    return s07.load_frames(), pd.read_csv(DERIVED / "threat_possessions.csv")


def all_events():
    return pd.concat([D.load_events(DATA_DIR / "matches" / str(m), m) for m in D.list_open_matches(DATA_DIR)],
                     ignore_index=True)


def phase_dummies(df, phases):
    return [(df["first_phase"].fillna("none") == ph).astype(float).to_numpy() for ph in phases[1:]]


def boot_mean_by_match(df, col, n=2000):
    per = df.groupby("match_id")[col].agg(["sum", "size"])
    idx = rng.integers(0, len(per), (n, len(per)))
    v = per["sum"].to_numpy()[idx].sum(1) / per["size"].to_numpy()[idx].sum(1)
    return float(df[col].mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


# --------------------------------------------------------------------------- #
# 1. Product vs free coefficients
# --------------------------------------------------------------------------- #
def decomposition(f, phases, n_boot=100):
    """logit P(shot in 5 s) = controls + b_rd log RD + b_def log(1 - cover).
    Threat Conceded assumes b_rd = b_def."""
    y = f["y"].to_numpy()
    out = {}
    for name, rd, cov in (("xT", "rd", "cover"), ("xG from here", "rd_xg_here", "cover_xg_here")):
        X = np.column_stack([f["t_in_possession"], f["def_goal_side"], *phase_dummies(f, phases),
                             np.log(np.maximum(f[rd], EPS)), np.log(np.maximum(1 - f[cov], 1e-4))])
        c = M.fit_logistic(X, y)
        mids = f["match_id"].unique()
        groups = f.groupby("match_id").indices
        bs = []
        for _ in range(n_boot):
            ii = np.concatenate([groups[m] for m in rng.choice(mids, len(mids))])
            bs.append(M.fit_logistic(X[ii], y[ii])[-2:])
        bs = np.array(bs)
        out[name] = dict(b_rd=c[-2], b_def=c[-1], b_rd_lo=np.percentile(bs[:, 0], 2.5),
                         b_rd_hi=np.percentile(bs[:, 0], 97.5), b_def_lo=np.percentile(bs[:, 1], 2.5),
                         b_def_hi=np.percentile(bs[:, 1], 97.5),
                         ratio_lo=np.percentile(bs[:, 1] / bs[:, 0], 2.5),
                         ratio_hi=np.percentile(bs[:, 1] / bs[:, 0], 97.5))
    return out


# --------------------------------------------------------------------------- #
# 2. Cover on top of a flexible ball model (with phase controls)
# --------------------------------------------------------------------------- #
def frame_cover_tests(f, phases):
    def X(d, cover):
        base = [DF.design(d, DF.FRAME_CONTROLS, None)] + [np.column_stack(phase_dummies(d, phases))]
        if cover is not None:
            base.append(DF.logit(d[cover])[:, None])
        return np.column_stack(base)

    out, oof_main = {}, None
    box = f[~f["in_box"] & (f["y"] == 0)].reset_index(drop=True)
    for label, d, ycol in (("shot within 5 s", f, "y"), ("ball into the box within 5 s", box, "box_next")):
        y = d[ycol].to_numpy()
        p0 = s07.lomo(d, X(d, None), y)
        for cov, cl in (("cover", "xT"), ("cover_xg_here", "xG from here")):
            p1 = s07.lomo(d, X(d, cov), y)
            b = s07.boot_diff(d, y, p1, p0, with_auc=False)
            out[(label, cl)] = dict(n=len(d), rate=y.mean(), ll0=M.log_loss(y, p0), ll1=M.log_loss(y, p1),
                                    auc0=M.auc(y, p0), auc1=M.auc(y, p1), **b)
            if ycol == "y" and cov == "cover_xg_here":
                oof_main = (p0, p1)
    return out, oof_main


def heterogeneity(f, oof):
    y = f["y"].to_numpy()
    ll = lambda p: -(y * np.log(np.clip(p, EPS, 1 - EPS)) + (1 - y) * np.log(np.clip(1 - p, EPS, 1 - EPS)))  # noqa
    d = f[["match_id", "phase", "organised_defense", "y"]].copy()
    d["gain"] = ll(oof[1]) - ll(oof[0])
    rows = []
    for key, grp in (("phase of play", d["phase"].fillna("none")),
                     ("defense", d["organised_defense"].map({True: "organised", False: "not organised"})
                      .fillna("not labelled"))):
        for val, sub in d.groupby(grp):
            if sub["y"].sum() < 30:
                continue
            m, lo, hi = boot_mean_by_match(sub, "gain")
            rows.append(dict(split=key, value=val, frames=len(sub), shot_rate=sub["y"].mean(), dll=m, lo=lo, hi=hi))
    return pd.DataFrame(rows)


def possession_cover_tests(p, phases):
    y = p["y"].to_numpy()
    ph = np.column_stack(phase_dummies(p, phases))
    X0 = np.column_stack([DF.possession_design(p, None), ph])
    p0 = s07.lomo(p, X0, y)
    out = {}
    for label, extra in (("cover (xT) at the moment of peak danger", DF.logit(p["cover_at_peak"])),
                         ("cover (xG from here) at the moment of peak danger", DF.logit(p["cover_xg_here_at_peak"])),
                         ("cover (xG from here) averaged over the possession", DF.logit(p["mean_cover_xg_here_early"])),
                         ("Threat Conceded (xT) peak, product form", np.log(np.maximum(p["peak_tc_early"], EPS))),
                         ("Threat Conceded (xG from here) peak, product form",
                          np.log(np.maximum(p["peak_tc_xg_here_early"], EPS)))):
        p1 = s07.lomo(p, np.column_stack([X0, extra]), y)
        out[label] = dict(ll1=M.log_loss(y, p1), auc1=M.auc(y, p1), **s07.boot_diff(p, y, p1, p0))
    base = dict(n=len(p), shots=int(y.sum()), ll0=M.log_loss(y, p0), auc0=M.auc(y, p0))
    p = p.copy()
    p["danger"] = pd.qcut(p["peak_rd_xg_here_early"], 4, labels=["Q1 (least)", "Q2", "Q3", "Q4 (most dangerous)"])
    p["cover_t"] = pd.qcut(p["cover_at_peak"], 3, labels=["low cover", "mid", "high cover"])
    rate = p.pivot_table(index="danger", columns="cover_t", values="y", aggfunc="mean", observed=False)
    n = p.pivot_table(index="danger", columns="cover_t", values="y", aggfunc="size", observed=False)
    return base, out, rate, n


# --------------------------------------------------------------------------- #
# 3. SkillCorner EPV
# --------------------------------------------------------------------------- #
def epv_benchmark(p, ev, phases):
    pp = ev[ev["event_type"] == "player_possession"].dropna(subset=["possession_epv_for_start"])
    pe = p[["match_id", "possession_id", "team_id", "frame_start", "frame_end", "first_shot_frame"]].merge(
        pp.rename(columns={"frame_start": "pf", "team_id": "pt"})[["match_id", "pf", "pt", "possession_epv_for_start"]],
        on="match_id")
    lim = pe["first_shot_frame"].fillna(pe["frame_end"]) - 2.0 * FPS
    pe = pe[(pe["pt"] == pe["team_id"]) & (pe["pf"] >= pe["frame_start"]) & (pe["pf"] <= lim)]
    epv = pe.groupby(["match_id", "possession_id"])["possession_epv_for_start"].max().rename("peak_epv_early")
    q = p.merge(epv.reset_index(), on=["match_id", "possession_id"]).reset_index(drop=True)
    y = q["y"].to_numpy()
    C = np.column_stack([q["log_duration"], q["mean_def_goal_side_early"], q["start_x"], *phase_dummies(q, phases)])
    feats = {"SkillCorner EPV (peak)": np.log(q["peak_epv_early"]),
             "ball distance to goal (closest)": q["neg_min_dist"].to_numpy(),
             "Threat Conceded, xG from here (peak)": np.log(np.maximum(q["peak_tc_xg_here_early"], EPS)),
             "Threat Conceded, xT (peak)": np.log(np.maximum(q["peak_tc_early"], EPS))}
    oof = {k: s07.lomo(q, np.column_stack([C, v]), y) for k, v in feats.items()}
    e = feats["SkillCorner EPV (peak)"]
    oof_both = s07.lomo(q, np.column_stack([C, e, feats["Threat Conceded, xG from here (peak)"]]), y)
    scores = pd.DataFrame([dict(measure=k, log_loss=M.log_loss(y, v), auc=M.auc(y, v)) for k, v in oof.items()])
    diffs = {"EPV vs ball distance": s07.boot_diff(q, y, oof["SkillCorner EPV (peak)"],
                                                   oof["ball distance to goal (closest)"]),
             "EPV + Threat Conceded (xG) vs EPV alone": s07.boot_diff(q, y, oof_both, oof["SkillCorner EPV (peak)"])}
    return len(q), int(y.sum()), scores, diffs


# --------------------------------------------------------------------------- #
# 4. Engagements
# --------------------------------------------------------------------------- #
def engagements(frames, ev):
    ob = ev[ev["event_type"] == "on_ball_engagement"][[
        "match_id", "event_id", "frame_start", "frame_end", "team_id", "event_subtype", "beaten_by_possession",
        "beaten_by_movement", "stop_possession_danger", "reduce_possession_danger", "force_backward"]].copy()
    flag = lambda c: ob[c].astype(bool).to_numpy()  # noqa: E731  (object dtype in the csv)
    ob["outcome"] = np.select(
        [flag("beaten_by_possession") | flag("beaten_by_movement"), flag("stop_possession_danger"),
         flag("reduce_possession_danger"), flag("force_backward")],
        ["beaten", "stopped danger", "reduced danger", "forced backward"], "no flagged outcome")
    cols = ["match_id", "possession_id", "frame", "attacking_team_id", "tc", "rd", "tc_xg_here", "rd_xg_here"]
    fr = frames[cols].sort_values("frame")
    a = pd.merge_asof(ob.sort_values("frame_start"), fr.rename(columns={"frame": "f0"}), left_on="frame_start",
                      right_on="f0", by="match_id", direction="nearest", tolerance=2)
    ob["t1"] = ob["frame_end"] + int(FPS)
    b = pd.merge_asof(ob.sort_values("t1"), fr.rename(columns={"frame": "f1"}), left_on="t1", right_on="f1",
                      by="match_id", direction="nearest", tolerance=2)
    key = ["match_id", "event_id"]
    m = a.merge(b[key + ["possession_id", "tc", "rd", "tc_xg_here", "rd_xg_here"]], on=key, suffixes=("_0", "_1"))
    m = m[(m["possession_id_0"] == m["possession_id_1"]) & (m["attacking_team_id"] != m["team_id"])].copy()
    lg = lambda c: np.log(np.maximum(m[f"{c}_1"], EPS)) - np.log(np.maximum(m[f"{c}_0"], EPS))  # noqa
    m["d_ball_xg"], m["d_ball_xt"] = lg("rd_xg_here"), lg("rd")
    m["d_def_xg"] = lg("tc_xg_here") - m["d_ball_xg"]      # change in log(1 - cover)
    m["d_def_xt"] = lg("tc") - m["d_ball_xt"]
    rows = []
    for g in ["beaten", "stopped danger", "reduced danger", "forced backward", "no flagged outcome"]:
        s = m[m["outcome"] == g]
        r = dict(outcome=g, n=len(s))
        for c in ("d_ball_xg", "d_def_xg", "d_def_xt"):
            r[c], r[c + "_lo"], r[c + "_hi"] = boot_mean_by_match(s, c)
        rows.append(r)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 5. Fitted model, cover effect, shots prevented
# --------------------------------------------------------------------------- #
def shots_prevented_lomo(f, metas):
    rows = []
    for mid in f["match_id"].unique():
        tr, te = f[f["match_id"] != mid], f[f["match_id"] == mid]
        model = DF.fit_two_part(tr, tr["y"], cover_col="cover_xg_here")
        typ = DF.fit_typical_cover(tr, cover_col="cover_xg_here")
        te = te.assign(prevented=DF.shots_prevented(model, typ, te, horizon_s=HORIZON_S, frame_dt=0.2))
        meta = metas[mid]
        for att_home, g in te.groupby("attacking_home"):
            rows.append(dict(match_id=mid, team=meta.away_name if att_home else meta.home_name,
                             opponent=meta.home_name if att_home else meta.away_name,
                             shots_prevented=g["prevented"].sum()))
    out = pd.DataFrame(rows)
    poss = pd.read_csv(DERIVED / "threat_possessions.csv")
    conceded = poss.assign(team_home=~poss["attacking_home"]).groupby(["match_id", "team_home"])["n_shots"].sum()
    out["shots_conceded"] = [conceded.get((m, metas[m].home_name == t), np.nan) for m, t in zip(out["match_id"], out["team"])]
    return out


def cover_effect_figure(model, typical, f, values):
    """P(shot within 5 s) against the defense's cover, three ball positions."""
    L, W = 105.0, 68.0
    cells = PC.make_grid(L, W, 2.0).cells
    xt = V.zone_value_at(V.OPEN_DATA_XT, cells[:, 0] / L + 0.5, cells[:, 1] / W + 0.5)
    xgh = values(cells, L, W)
    spots = [("edge of the box, central (x = 34 m)", 34.0, 0.0),
             ("wide in the final third (x = 30 m, y = 25 m)", 30.0, 25.0),
             ("halfway line, central (x = 0 m)", 0.0, 0.0)]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    cov = np.linspace(0.30, 0.97, 60)
    for (lab, x, y), col in zip(spots, ["#c0392b", "#e67e22", "#2c7fb8"]):
        d = pd.DataFrame({"ball_x": x, "ball_y": y, "t_in_possession": f["t_in_possession"].median(),
                          "def_goal_side": f["def_goal_side"].median(), "cover_xg_here": cov})
        d["rd"] = DF.reachable_danger(d["ball_x"], d["ball_y"], cells, xt, L, W)
        d["rd_xg_here"] = DF.reachable_danger(d["ball_x"], d["ball_y"], cells, xgh, L, W)
        ax.plot(cov, 100 * model.predict(d), color=col, lw=2, label=lab)
        t = typical.predict(d.iloc[:1])[0]
        ax.plot([t], [100 * model.predict(d.iloc[:1], cover=np.array([t]))[0]], "o", color=col)
    ax.set_xlabel("defense's cover of the reachable shooting danger (share)")
    ax.set_ylabel("chance of a shot within 5 s (%)")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_title("Same ball position, different defensive cover\n(dots: a typical A-League defense there)", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIGS / "w6_cover_effect.png", dpi=150)
    plt.close(fig)


def engagement_figure(eng):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    e = eng.set_index("outcome")
    yy = np.arange(len(e))
    ax.barh(yy + 0.2, e["d_ball_xg"], 0.38, color="#95a5a6", label="ball danger (where the ball can go)")
    ax.barh(yy - 0.2, e["d_def_xg"], 0.38, color="#c0392b", label="uncovered share (1 − cover)")
    for c, off in (("d_ball_xg", 0.2), ("d_def_xg", -0.2)):
        ax.errorbar(e[c], yy + off, xerr=[e[c] - e[c + "_lo"], e[c + "_hi"] - e[c]], fmt="none", color="k", lw=0.8)
    ax.set_yticks(yy, [f"{o} (n={n})" for o, n in zip(e.index, e["n"])])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("change from the start of the press to 1 s after it (log scale)")
    ax.set_title("What a press does: pushes the ball back, opens space behind", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGS / "w6_engagements.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    frames, poss = load_skillcorner()
    ev = all_events()
    pp = ev[ev["event_type"] == "player_possession"][["match_id", "frame_start", "frame_end", "team_id",
                                                      "organised_defense"]]
    fr = pd.merge_asof(frames.sort_values("frame"), pp.sort_values("frame_start").rename(columns={
        "frame_start": "pp_start", "frame_end": "pp_end", "team_id": "pp_team"}), left_on="frame",
        right_on="pp_start", by="match_id", direction="backward")
    ok = (fr["frame"] <= fr["pp_end"] + FPS) & (fr["pp_team"] == fr["attacking_team_id"])
    fr.loc[~ok, "organised_defense"] = np.nan
    frames = fr.drop(columns=["pp_start", "pp_end", "pp_team"])
    f = DF.shot_frames(frames, poss, FPS).merge(poss[["match_id", "possession_id", "first_phase"]],
                                                 on=["match_id", "possession_id"], how="left")
    phases = sorted(f["first_phase"].fillna("none").unique())
    print(f"{len(f):,} frames", flush=True)

    dec = decomposition(f, phases)
    print("decomposition", dec, flush=True)
    ft, oof = frame_cover_tests(f, phases)
    print("frame tests done", flush=True)
    het = heterogeneity(f, oof)
    p = DF.possession_moments(f, poss, FPS).dropna(subset=["cover_at_peak", "cover_xg_here_at_peak"]
                                                   ).reset_index(drop=True)
    p["first_phase"] = p["first_phase"].fillna("none")
    pbase, pt, rate, n = possession_cover_tests(p, sorted(p["first_phase"].unique()))
    print("possession tests done", flush=True)
    epv = epv_benchmark(p, ev, sorted(p["first_phase"].unique()))
    eng = engagements(frames, ev)
    print("epv/engagements done", flush=True)

    metas = {m: D.load_match_metadata(DATA_DIR / "matches" / str(m), m) for m in f["match_id"].unique()}
    model = DF.fit_two_part(f, f["y"], cover_col="cover_xg_here")
    typical = DF.fit_typical_cover(f, cover_col="cover_xg_here")
    sp = shots_prevented_lomo(f, metas)
    sp.to_csv(DERIVED / "shots_prevented_team_matches.csv", index=False)
    values = _load("s06", ROOT / "scripts" / "06_threat.py").comparisons().values["xg_here"]
    cover_effect_figure(model, typical, f, values)
    engagement_figure(eng)
    write_report(f, dec, ft, het, pbase, pt, rate, n, epv, eng, model, sp)
    print("done", flush=True)


def write_report(f, dec, ft, het, pbase, pt, rate, n, epv, eng, model, sp):
    L = []
    w = L.append
    w("# The two-part model: the ball's situation and the defense's cover\n")
    w("**Status: exploratory.** These analyses were run on the 20 SkillCorner matches after the pre-declared "
      "Week 5 test failed ([w5_validation.md](w5_validation.md)), so treat them as a well-supported lead rather "
      "than a confirmed result. Script: `scripts/08_two_part.py`; model: `dsv/defense.py`.\n")
    w("## 1. Why the product failed\n")
    w("Threat Conceded multiplies reachable danger (where the ball is and can go) by the share the defense "
      "leaves uncovered, so log TC = log RD + log(1 − cover). In a model of a shot within 5 s, that forces "
      "both to carry the same coefficient. Fitted freely (with time in possession, defenders goal-side "
      "and phase controls; 95% match-bootstrap intervals):\n")
    w("| danger map | coefficient on log RD (situation) | coefficient on log(1 − cover) (defense) | ratio defense / situation |")
    w("|---|---|---|---|")
    for k, d in dec.items():
        w(f"| {k} | {d['b_rd']:.2f} [{d['b_rd_lo']:.2f}, {d['b_rd_hi']:.2f}] | {d['b_def']:.2f} "
          f"[{d['b_def_lo']:.2f}, {d['b_def_hi']:.2f}] | {d['ratio_lo']:.2f} to {d['ratio_hi']:.2f} |")
    w("\nThe product is right only if the ratio is 1. The defense's share matters, but with a weight of its own, "
      "so it has to enter the model separately.\n")
    w("## 2. Cover with its own coefficient\n")
    w("Held out by match. The baseline describes where the ball is flexibly (distance and its square, "
      "goal-mouth angle, |y|, x, log reachable danger under both maps) with time in possession, defenders "
      "goal-side and phase of play. Negative differences mean cover helps.\n")
    w("| outcome | cover | frames (rate) | log loss without → with | difference [95%] | AUC without → with |")
    w("|---|---|---|---|---|---|")
    for (lab, cl), d in ft.items():
        w(f"| {lab} | {cl} | {d['n']:,} ({d['rate']:.1%}) | {d['ll0']:.4f} → {d['ll1']:.4f} | "
          f"{d['dll']:+.4f} [{d['dll_lo']:+.4f}, {d['dll_hi']:+.4f}] | {d['auc0']:.3f} → {d['auc1']:.3f} |")
    w(f"\n**Per possession** ({pbase['n']:,} possessions, {pbase['shots']} with a shot or goal; baseline log loss "
      f"{pbase['ll0']:.4f}, AUC {pbase['auc0']:.3f}):\n")
    w("| added to the baseline | log loss | AUC | difference [95%] |")
    w("|---|---|---|---|")
    for k, d in pt.items():
        w(f"| {k} | {d['ll1']:.4f} | {d['auc1']:.3f} | {d['dll']:+.4f} [{d['dll_lo']:+.4f}, {d['dll_hi']:+.4f}] |")
    w("\nThe defense's cover counts at the moment of greatest danger; averaged over the whole possession it "
      "carries nothing. Defending is decided at the critical moments.\n")
    w("Shot or goal rate (%) by how dangerous the possession got and the cover at that moment:\n")
    t = (rate * 100).round(1).astype(str) + " (n=" + n.astype(int).astype(str) + ")"
    w("| danger | " + " | ".join(map(str, t.columns)) + " |")
    w("|---|" + "---|" * len(t.columns))
    for idx, row in t.iterrows():
        w(f"| {idx} | " + " | ".join(row.values) + " |")
    w("\n**Where cover matters** (moment level, xG-from-here cover; difference in held-out log loss within each "
      "group, negative = cover helps):\n")
    w("| split | group | frames | shot rate | difference [95%] |")
    w("|---|---|---|---|---|")
    for _, r in het.iterrows():
        w(f"| {r['split']} | {r['value']} | {r['frames']:,} | {r['shot_rate']:.1%} | {r['dll']:+.4f} "
          f"[{r['lo']:+.4f}, {r['hi']:+.4f}] |")
    nq, nshots, sc, dif = epv
    w(f"\n## 3. Benchmark: SkillCorner's own EPV\n")
    w(f"SkillCorner's dynamic events carry an expected possession value built from full tracking context "
      f"(chance of scoring in the next 30 s), recorded at the start of each player possession. Same "
      f"possession test ({nq:,} possessions, {nshots} with a shot), peak at least 2 s before any shot, with "
      "the Week 5 controls:\n")
    w("| measure | log loss | AUC |")
    w("|---|---|---|")
    for _, r in sc.iterrows():
        w(f"| {r['measure']} | {r['log_loss']:.4f} | {r['auc']:.3f} |")
    for k, d in dif.items():
        w(f"\n{k}: difference {d['dll']:+.4f} [{d['dll_lo']:+.4f}, {d['dll_hi']:+.4f}].")
    w("\nEPV is only recorded when a player receives the ball, while Threat Conceded is measured five times a "
      "second, so the peak comparison favours ours; EPV also targets goals, not shots.\n")
    w("## 4. What a press does\n")
    w("For each of SkillCorner's on-ball engagements (a defender pressing the ball carrier), the change from the "
      "start of the engagement to 1 s after it, within the same possession, of the ball's reachable shooting "
      "danger and of the share the defense leaves uncovered (both log scale, xG-from-here map; 95% match-"
      "bootstrap intervals). Outcome flags are SkillCorner's.\n")
    w("| outcome | engagements | ball danger | uncovered share (xG map) | uncovered share (xT map) |")
    w("|---|---|---|---|---|")
    for _, r in eng.iterrows():
        w(f"| {r['outcome']} | {r['n']:,} | {r['d_ball_xg']:+.2f} [{r['d_ball_xg_lo']:+.2f}, {r['d_ball_xg_hi']:+.2f}] | "
          f"{r['d_def_xg']:+.2f} [{r['d_def_xg_lo']:+.2f}, {r['d_def_xg_hi']:+.2f}] | "
          f"{r['d_def_xt']:+.2f} [{r['d_def_xt_lo']:+.2f}, {r['d_def_xt_hi']:+.2f}] |")
    e = eng.set_index("outcome")
    flagged = e.drop(index="no flagged outcome")
    top = flagged["d_def_xg"].idxmax()
    opens = (flagged["d_def_xg_lo"] > e.loc["no flagged outcome", "d_def_xg_hi"]).all()
    w(f"\nThe largest rise in the uncovered share follows presses flagged \"{top}\" "
      f"({e.loc[top, 'd_def_xg']:+.2f}); after a beaten press it is {e.loc['beaten', 'd_def_xg']:+.2f}, against "
      f"{e.loc['no flagged outcome', 'd_def_xg']:+.2f} when no outcome is flagged. "
      + ("Every flagged outcome opens more space than an unflagged engagement: " if opens else
         "Most flagged outcomes open some space: ")
      + "a press that forces the ball backward cuts ball danger sharply but leaves the defense covering a smaller "
      "share of what remains. That trade-off is what the pressing analysis weighs ([w7_pressing.md](w7_pressing.md)). "
      "\"Stopped danger\" engagements usually end the possession, so the few that remain are not typical.\n")
    w("![engagements](../figs/w6_engagements.png)\n")
    w("## 5. The fitted model\n")
    w(f"Fitted on all frames with the xG-from-here cover, the cover coefficient is {model.cover_effect:.2f}: "
      f"each unit of logit cover multiplies the odds of a shot in the next 5 s by "
      f"{np.exp(model.cover_effect):.2f}.\n")
    w("![cover effect](../figs/w6_cover_effect.png)\n")
    w("**Shots prevented** compares each defense with a typical A-League defense in the same situations: the "
      "model's shot chance with the cover a typical defense holds there, minus the chance with the actual "
      "cover, summed over the match (fitted without the match being scored). Positive: fewer shots than a "
      "typical defense would have allowed. Per team and match "
      "(`data/derived/shots_prevented_team_matches.csv`):\n")
    s = sp.sort_values("shots_prevented", ascending=False)
    w("| defending team | opponent | shots prevented | shots conceded |")
    w("|---|---|---|---|")
    for _, r in pd.concat([s.head(5), s.tail(5)]).iterrows():
        w(f"| {r['team']} | {r['opponent']} | {r['shots_prevented']:+.2f} | {r['shots_conceded']:.0f} |")
    rho = sp[["shots_prevented", "shots_conceded"]].corr(method="spearman").iloc[0, 1]
    w(f"\nAcross the {len(sp)} team-matches, Spearman correlation between shots prevented and shots conceded: "
      f"{rho:.2f}. Expected to be negative but not strongly so: shots conceded also depend on how often the "
      "opponent reached dangerous positions, which cover does not measure.\n")
    (REPORTS / "w6_two_part.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
