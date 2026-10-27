"""
Week 4: Threat Conceded over whole matches, frame by frame and possession by
possession.

1. Accuracy of the fast computation (2 m grid, 0.08 s step, skip-ahead
   integration) against the full one (1 m, 0.04 s) on sampled frames.
2. Frame-level Threat Conceded, reachable danger and cover for every team
   possession of all 20 matches, 5 frames per second (cached per match).
3. Frame level: how danger relates to a shot by the attacking team within the
   next 5 s, overall, by phase of play, and with ball location held fixed
   (does the defense's cover matter beyond where the ball is?).
4. Possession level: peak / exposure / mean per possession against shots,
   goals and xG, with the peak taken at least 2 s before any shot; example
   danger curves.
5. Team level: danger conceded per match against xG conceded.

This is descriptive. The formal test (Week 5) adds controls and
cross-validation; numbers here are a first look.

Run from the repo root:  py scripts/06_threat.py   (about an hour the first time)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dsv import data as D, metrics as M, pitchcontrol as PC, threat as T, value as V, xg  # noqa: E402

DATA_DIR = Path(os.environ.get("DSV_DATA_DIR", ROOT / "data"))
FIGS, REPORTS, DERIVED = ROOT / "figs", ROOT / "reports", DATA_DIR / "derived"
CACHE = DERIVED / "threat"
CACHE.mkdir(parents=True, exist_ok=True)
STEP = 2              # frames: 5 per second
HORIZON_S = 5.0       # frame-level label: shot within the next 5 s
LOOKBACK_S = 2.0      # possession-level: peak at least 2 s before the shot
FOCAL = 1925299
N_BOOT = 2000
# per-frame comparison measures kept for the shot-prediction test
EXTRA = ["space_m2", "tc_no_access", "tc_voronoi", "tc_singh_xt", "tc_xg_here", "tc_iso_access",
         "def_goal_side"]


def load(mid):
    md = DATA_DIR / "matches" / str(mid)
    meta = D.load_match_metadata(md, mid)
    players, ball = D.normalize_direction(*D.load_tracking(md, mid, meta), meta)
    return meta, D.add_velocity(players), ball, D.load_events(md, mid), D.load_phases(md, mid)


def comparisons():
    """Alternative ingredients for baselines and sensitivity checks: Karun
    Singh's published xT and "xG of a typical shot from here" as danger
    surfaces, and the single isotropic Gaussian as the next-ball model."""
    singh = np.array(json.loads((DERIVED / "singh_xt_12x8.json").read_text()))
    sc = pd.read_csv(DERIVED / "skillcorner_shots.csv")
    ctx = sc[["n_block", "gk_off_line", "gk_dist_goal", "press_dist", "n_close"]].median().to_dict()

    def singh_xt(c, L, W):
        return V.zone_value_at(singh, c[:, 0] / L + 0.5, c[:, 1] / W + 0.5)

    def xg_here(c, L, W):
        gx, gy = c[:, 0] - L / 2, c[:, 1]
        dist = np.hypot(gx, gy)
        angle = np.abs(np.arctan2(3.66 - gy, -gx) - np.arctan2(-3.66 - gy, -gx))
        df = pd.DataFrame({"dist": dist, "log_dist": np.log(np.maximum(dist, 0.5)), "angle": angle,
                           "header": 0.0, **ctx})
        return xg.xg_for_shots(df)

    trs = pd.read_pickle(DERIVED / "value_tables.pkl")[2]
    iso = V.fit_transition(trs, k=1, isotropic=True)
    return T.Comparisons(values={"singh_xt": singh_xt, "xg_here": xg_here},
                         transitions={"iso_access": iso})


def match_tables(mid, vt, xgs, comps):
    """Frame and possession tables for one match, cached."""
    f_path, p_path = CACHE / f"{mid}_frames.csv.gz", CACHE / f"{mid}_possessions.csv"
    meta = D.load_match_metadata(DATA_DIR / "matches" / str(mid), mid)
    if f_path.exists() and p_path.exists() and not os.environ.get("REBUILD"):
        return meta, pd.read_csv(f_path), pd.read_csv(p_path)
    meta, players, ball, events, phases = load(mid)
    poss = T.team_possessions(events, meta, xgs)
    t0 = time.perf_counter()
    frames = T.match_threat(players, ball, poss, meta, PC.vmax_lookup(vt, mid), step=STEP, phases=phases,
                            comparisons=comps)
    frames.to_csv(f_path, index=False)
    poss.to_csv(p_path, index=False)
    print(f"  {mid}: {len(poss)} possessions, {len(frames):,} frames, {time.perf_counter() - t0:.0f} s", flush=True)
    return meta, frames, poss


def accuracy_check(vt, n=100, seed=0):
    """Fast computation vs the full 1 m / 0.04 s one on sampled frames of the
    focal match."""
    meta, players, ball, events, _ = load(FOCAL)
    poss = T.team_possessions(events, meta)
    rng = np.random.default_rng(seed)
    pick = poss.sample(n, replace=True, random_state=seed)
    vm = PC.vmax_lookup(vt, FOCAL)
    fast_model, full_params = T.ThreatModel(), PC.SKILLCORNER_OPEN_DATA
    fx = T._Frames(players, ball, meta, vm, full_params)
    grid_fast = PC.make_grid(meta.pitch_length, meta.pitch_width, fast_model.cell)
    grid_full = PC.make_grid(meta.pitch_length, meta.pitch_width, 1.0)
    rows = []
    for _, p in pick.iterrows():
        f = int(rng.integers(p["frame_start"], p["frame_end"] + 1))
        if not fx.has(f) or not np.isfinite(fx.ball.loc[f]).all():
            continue
        s, ah = fx.state(f), bool(p["attacking_home"])
        t0 = time.perf_counter(); fast = T.frame_threat(s, ah, grid_fast, meta, fast_model)["tc"]; t_fast = time.perf_counter() - t0
        sg = 1.0 if ah else -1.0
        t0 = time.perf_counter()
        r = PC.pitch_control(s, grid_full, full_params)
        w = T.danger_weights(sg * s.ball, sg * grid_full.cells, meta.pitch_length, meta.pitch_width)
        full = float((r.team(ah).ravel() * w).sum()); t_full = time.perf_counter() - t0
        rows.append(dict(fast=fast, full=full, t_fast=t_fast, t_full=t_full))
    df = pd.DataFrame(rows)
    df["rel_err"] = (df["fast"] - df["full"]).abs() / df["full"]
    return df


def bootstrap_auc_diff(df, a, b, label, n=N_BOOT, seed=0):
    """AUC(a) - AUC(b) for `label`, with a 95% interval from resampling matches."""
    rng = np.random.default_rng(seed)
    ids = df["match_id"].unique()
    by = {m: g for m, g in df.groupby("match_id")}
    point = M.auc(df[label], df[a]) - M.auc(df[label], df[b])
    diffs = []
    for _ in range(n):
        s = pd.concat([by[m] for m in rng.choice(ids, len(ids))])
        diffs.append(M.auc(s[label], s[a]) - M.auc(s[label], s[b]))
    return point, np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)


def main():
    t_start = time.perf_counter()
    vt = PC.vmax_table(pd.read_csv(DERIVED / "v95_by_player_half.csv"))
    xgs = pd.read_csv(DERIVED / "skillcorner_shots_xg.csv")
    acc = accuracy_check(vt)
    print(f"accuracy: median rel error {acc['rel_err'].median():.4f}, max {acc['rel_err'].max():.4f}; "
          f"{acc['t_fast'].median() * 1000:.0f} ms vs {acc['t_full'].median() * 1000:.0f} ms per frame", flush=True)

    comps = comparisons()
    metas, F, P = {}, [], []
    for mid in D.list_open_matches(DATA_DIR):
        meta, frames, poss = match_tables(mid, vt, xgs, comps)
        metas[mid] = meta
        F.append(frames); P.append(poss)
    frames, poss = pd.concat(F, ignore_index=True), pd.concat(P, ignore_index=True)

    # ----- frame level --------------------------------------------------------
    frames = frames.merge(poss[["match_id", "possession_id", "first_shot_frame"]], on=["match_id", "possession_id"])
    lead = (frames["first_shot_frame"] - frames["frame"]) / 10.0
    frames = frames[~(lead < 0)].copy()                      # drop frames after the shot
    frames["shot_next5"] = lead.loc[frames.index].between(0, HORIZON_S, inclusive="right")
    frames["dist_goal"] = np.hypot(frames["ball_x"].sub(52.5).abs(), frames["ball_y"])
    frames["neg_dist"] = -frames["dist_goal"]
    fl = pd.DataFrame([dict(measure=lab, auc=M.auc(frames["shot_next5"], frames[c]))
                       for c, lab in (("tc", "Threat Conceded"), ("rd", "reachable danger (ball location only)"),
                                      ("neg_dist", "ball distance to goal (closer = higher)"))])
    fl_boot = bootstrap_auc_diff(frames, "tc", "rd", "shot_next5", n=500)
    frames["rd_q"] = pd.qcut(frames["rd"], 5, labels=[f"Q{i}" for i in range(1, 6)])
    frames["cover_t"] = frames.groupby("rd_q", observed=True)["cover"].transform(
        lambda s: pd.qcut(s, 3, labels=["low cover", "mid cover", "high cover"]))
    within = (frames.groupby(["rd_q", "cover_t"], observed=True)["shot_next5"].mean().unstack())
    by_phase = (frames.dropna(subset=["phase"]).groupby("phase")
                .agg(frames=("tc", "size"), mean_tc=("tc", "mean"), mean_rd=("rd", "mean"),
                     mean_cover=("cover", "mean"), shot_next5=("shot_next5", "mean"))
                .sort_values("mean_tc", ascending=False))

    # ----- possession level ---------------------------------------------------
    ps = T.possession_summary(frames.drop(columns=["first_shot_frame"]), poss, lookback_s=LOOKBACK_S,
                              extra_columns=EXTRA)
    ps.to_csv(DERIVED / "threat_possessions.csv", index=False)
    pe = ps.dropna(subset=["peak_tc_early"]).copy()
    pe["shot"] = pe["shot"].astype(bool)
    pl_auc = pd.DataFrame([dict(measure=lab, auc=M.auc(pe["shot"], pe[c]))
                           for c, lab in (("peak_tc_early", "peak Threat Conceded"),
                                          ("peak_rd_early", "peak reachable danger (ball location only)"))])
    pl_boot = bootstrap_auc_diff(pe, "peak_tc_early", "peak_rd_early", "shot")
    shot_p = pe[pe["shot"]]
    xg_corr = float(shot_p[["peak_tc_early", "xg"]].corr(method="spearman").iloc[0, 1])
    pe["rd_t"] = pd.qcut(pe["peak_rd_early"], 3, labels=["low", "mid", "high"])
    pe["tc_ratio"] = pe["peak_tc_early"] / pe["peak_rd_early"]
    pe["share_t"] = pe.groupby("rd_t", observed=True)["tc_ratio"].transform(
        lambda s: pd.qcut(s, 3, labels=["defense covered most", "middle", "defense covered least"]))
    within_p = pe.groupby(["rd_t", "share_t"], observed=True)["shot"].agg(["mean", "size"]).unstack()

    # ----- team level ---------------------------------------------------------
    tm = T.team_match_summary(ps, metas)
    tm.to_csv(DERIVED / "threat_team_matches.csv", index=False)
    team_corr = {c: float(tm[["tc_exposure", c]].corr(method="spearman").iloc[0, 1])
                 for c in ("xg_conceded", "shots_conceded")}

    # ----- figures ------------------------------------------------------------
    curves_figure(frames, ps, metas)
    frame_figure(frames, within)
    possession_figure(pe)
    team_figure(tm, team_corr)
    maps_figure(frames)

    write_report(acc, frames, ps, pe, fl, fl_boot, within, by_phase, pl_auc, pl_boot, xg_corr,
                 within_p, tm, team_corr)
    print(fl.round(3).to_string(index=False)); print(pl_auc.round(3).to_string(index=False))
    print(f"done in {(time.perf_counter() - t_start) / 60:.1f} min", flush=True)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def curves_figure(frames, ps, metas):
    """Danger curves for four possessions of the focal match."""
    f = frames[frames["match_id"] == FOCAL]
    p = ps[(ps["match_id"] == FOCAL) & ps["peak_tc"].notna()]
    picks = []
    goal = p[p["goal"].astype(bool)]
    if len(goal):
        picks.append((goal.iloc[0], "goal"))
    rest = p.drop(index=[g[0].name for g in picks])
    picks.append((rest[rest["shot"].astype(bool)].sort_values("peak_tc").iloc[-1], "highest danger, shot"))
    picks.append((rest[~rest["shot"].astype(bool)].sort_values("peak_tc").iloc[-1], "highest danger, no shot"))
    mid_dur = rest[(rest["duration_s"].between(15, 40)) & ~rest["shot"].astype(bool)]
    picks.append((mid_dur.iloc[len(mid_dur) // 2], "typical attack, no shot"))
    meta = metas[FOCAL]
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.6), sharey=False)
    for ax, (row, label) in zip(axes.ravel(), picks):
        c = f[f["possession_id"] == row["possession_id"]].sort_values("frame")
        # break the lines where tracking is missing (stoppages inside a possession)
        gap = c["frame"].diff() > 5 * STEP
        if gap.any():
            filler = c[gap].assign(tc=np.nan, rd=np.nan, t_in_possession=c.loc[gap, "t_in_possession"] - 0.01)
            c = pd.concat([c, filler]).sort_values("t_in_possession")
        ax.fill_between(c["t_in_possession"], 0, c["rd"], color="#8a8f98", alpha=0.18, lw=0,
                        label="reachable danger (if the attack won everything)")
        ax.plot(c["t_in_possession"], c["tc"], color="#c53030", lw=1.6, label="Threat Conceded")
        if pd.notna(row["first_shot_frame"]):
            ax.axvline((row["first_shot_frame"] - row["frame_start"]) / 10, color="#111", lw=1, ls="--")
            ax.text((row["first_shot_frame"] - row["frame_start"]) / 10, ax.get_ylim()[1] * 0.92, " shot",
                    fontsize=8)
        team = meta.team_name(row["team_id"])
        ax.set_title(f"{label}: {team} attacking, {row['duration_s']:.0f} s, xG {row['xg']:.2f}", fontsize=9)
        ax.set_xlabel("seconds into the possession"); ax.set_ylabel("goals (xT)"); ax.grid(alpha=.3)
    axes[0, 0].legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.suptitle(f"Danger curves, {meta.home_name} vs {meta.away_name}", fontsize=10)
    fig.tight_layout(); fig.savefig(FIGS / "w4_possession_curves.png", dpi=125); plt.close(fig)


def frame_figure(frames, within):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    ax = axes[0]
    for col, lab, c in (("tc", "Threat Conceded", "#c53030"), ("rd", "reachable danger", "#8a8f98")):
        q = pd.qcut(frames[col], 10, labels=False, duplicates="drop")
        r = frames.groupby(q).agg(x=(col, "mean"), y=("shot_next5", "mean"))
        ax.plot(r["x"], r["y"], "o-", ms=4, color=c, label=lab)
    ax.set_xscale("log"); ax.set_xlabel("value at the frame (goals, log scale; deciles)")
    ax.set_ylabel("share of frames followed by a shot within 5 s")
    ax.set_title("Frame level: danger now vs shot soon", fontsize=10); ax.legend(fontsize=8, frameon=False); ax.grid(alpha=.3)
    ax = axes[1]
    xs = np.arange(len(within.index)); wd = 0.26
    for i, (col, c) in enumerate(zip(within.columns, ["#c53030", "#dd8a2c", "#2b6cb0"])):
        ax.bar(xs + (i - 1) * wd, within[col], wd, color=c, label=col)
    ax.set_xticks(xs, [f"{q}" for q in within.index])
    ax.set_xlabel("reachable danger quintile (where the ball is: Q1 safest, Q5 most dangerous)")
    ax.set_ylabel("share followed by a shot within 5 s")
    ax.set_title("Same ball danger, different defensive cover", fontsize=10)
    ax.legend(fontsize=8, frameon=False); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(FIGS / "w4_frame_level.png", dpi=125); plt.close(fig)


def possession_figure(pe):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    ax = axes[0]
    bins = np.logspace(np.log10(pe["peak_tc_early"].min()), np.log10(pe["peak_tc_early"].max()), 30)
    for shot, lab, c in ((False, "no shot", "#8a8f98"), (True, "shot", "#c53030")):
        v = pe.loc[pe["shot"] == shot, "peak_tc_early"]
        ax.hist(v, bins=bins, density=True, alpha=0.55, color=c, label=f"{lab} (n={len(v):,})")
    ax.set_xscale("log"); ax.set_xlabel("peak Threat Conceded, at least 2 s before any shot (goals)")
    ax.set_ylabel("density"); ax.set_title("Possession level: peak danger by outcome", fontsize=10)
    ax.legend(fontsize=8, frameon=False); ax.grid(alpha=.3)
    ax = axes[1]
    for col, lab, c in (("peak_tc_early", "peak Threat Conceded", "#c53030"),
                        ("peak_rd_early", "peak reachable danger", "#8a8f98")):
        q = pd.qcut(pe[col], 10, labels=False, duplicates="drop")
        r = pe.groupby(q).agg(x=(col, "mean"), y=("shot", "mean"))
        ax.plot(r["x"], r["y"], "o-", ms=4, color=c, label=lab)
    ax.set_xscale("log"); ax.set_xlabel("possession peak (goals, log scale; deciles)")
    ax.set_ylabel("share of possessions with a shot")
    ax.set_title("Shot rate by possession peak", fontsize=10); ax.legend(fontsize=8, frameon=False); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIGS / "w4_possession_level.png", dpi=125); plt.close(fig)


def team_figure(tm, corr):
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.scatter(tm["tc_exposure"], tm["xg_conceded"], s=26, color="#2b6cb0")
    for _, r in tm.nlargest(3, "tc_exposure").iterrows():
        ax.annotate(r["team"].split(" ")[0], (r["tc_exposure"], r["xg_conceded"]), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Threat Conceded exposure in the match (goal-seconds)")
    ax.set_ylabel("xG conceded in the match")
    ax.set_title(f"Team-matches (n={len(tm)}): Spearman r = {corr['xg_conceded']:.2f}", fontsize=10)
    ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIGS / "w4_team_matches.png", dpi=125); plt.close(fig)


def maps_figure(frames):
    """Mean Threat Conceded and mean cover by ball location (attacking right)."""
    from dsv import viz as VZ
    grid = PC.make_grid(105, 68, 5.0)
    ix = np.clip(((frames["ball_x"] + 52.5) / 105 * len(grid.xs)).astype(int), 0, len(grid.xs) - 1)
    iy = np.clip(((frames["ball_y"] + 34) / 68 * len(grid.ys)).astype(int), 0, len(grid.ys) - 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.0))
    for ax, col, title, cmap in ((axes[0], "tc", "mean Threat Conceded by ball location", "Reds"),
                                 (axes[1], "cover", "mean cover (share of reachable danger the defense holds)", "Blues")):
        g = frames.assign(ix=ix, iy=iy).groupby(["iy", "ix"])[col].agg(["mean", "size"])
        z = np.full(grid.shape, np.nan)
        for (a, b), r in g.iterrows():
            if r["size"] >= 30:
                z[a, b] = r["mean"]
        VZ.draw_pitch(ax)
        im = VZ.plot_surface(ax, grid, z, cmap=cmap, vmin=np.nanmin(z), vmax=np.nanmax(z), alpha=1)
        ax.set_title(title + " (attacking →)", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.75)
    fig.tight_layout(); fig.savefig(FIGS / "w4_maps.png", dpi=125); plt.close(fig)


# --------------------------------------------------------------------------- #
def write_report(acc, frames, ps, pe, fl, fl_boot, within, by_phase, pl_auc, pl_boot, xg_corr,
                 within_p, tm, team_corr):
    L = []
    w = L.append
    n_m = frames["match_id"].nunique()
    w("# Threat Conceded over whole matches\n")
    w(f"Threat Conceded computed for every team possession of all {n_m} SkillCorner open-data matches, "
      f"5 times per second: **{len(frames):,} frames** in **{len(ps):,} possessions**. Module: "
      "`dsv/threat.py`; script: `scripts/06_threat.py`. This is a descriptive first look; the formal "
      "shot-prediction test adds controls and cross-validation.\n")

    w("## What is measured\n")
    w("| level | measure | definition | football meaning |")
    w("|---|---|---|---|")
    w("| frame | **Threat Conceded (TC)** | Σ A(x \\| ball) · PC_att(x) · xT(x) | danger the defense is allowing right now, in goals |")
    w("| frame | **reachable danger (RD)** | Σ A(x \\| ball) · xT(x) | danger if the attack won every spot: depends only on where the ball is |")
    w("| frame | **cover** | 1 − TC / RD | share of the reachable danger the defense controls |")
    w("| possession | **peak** | highest TC in the possession, at least 2 s before any shot | the worst moment the defense allowed, before the shot itself |")
    w("| possession | **exposure** | Σ TC × time between frames | how much danger, for how long (goal-seconds) |")
    w("| team-match | **TC exposure** | sum of exposure over the opponent's possessions | total danger allowed in the match |")
    w("")

    w("## Computing it fast enough\n")
    w("Full resolution (1 m grid, 0.04 s steps) takes about a quarter of a second per frame, two hours per "
      "match. The match-long computation uses a 2 m grid, 0.08 s steps, and starts each spot's race just "
      "before the first player could plausibly arrive, dropping spots once decided. Checked against the "
      f"full computation on {len(acc)} sampled frames: median relative error **{acc['rel_err'].median():.2%}**, "
      f"maximum {acc['rel_err'].max():.2%}; {acc['t_fast'].median() * 1000:.0f} ms vs "
      f"{acc['t_full'].median() * 1000:.0f} ms per frame.\n")

    w("## Frame by frame\n")
    w(f"Typical values: TC median {frames['tc'].median():.4f} (90th percentile {frames['tc'].quantile(.9):.4f}); "
      f"reachable danger median {frames['rd'].median():.4f}; the defense covers a median "
      f"{frames['cover'].median():.0%} of the danger within reach.\n")
    w(f"How well does the value at a moment rank whether the attacking team shoots within the next "
      f"{HORIZON_S:.0f} s? ({frames['shot_next5'].mean():.1%} of frames are followed by a shot.)\n")
    w("| measure | AUC |")
    w("|---|---|")
    for _, r in fl.iterrows():
        w(f"| {r['measure']} | {r['auc']:.3f} |")
    pt, lo, hi = fl_boot
    w(f"\nTC minus reachable danger: {pt:+.3f} AUC (95% match-bootstrap interval {lo:+.3f} to {hi:+.3f}). "
      "Frames within a possession are strongly correlated, so these are descriptive.\n")
    w("**Holding ball location fixed.** Frames are split into five bands of reachable danger (where the "
      "ball is), then within each band into thirds by cover. Share of frames followed by a shot within 5 s:\n")
    w("| ball danger band | " + " | ".join(within.columns) + " |")
    w("|---|" + "---|" * len(within.columns))
    for q, r in within.iterrows():
        w(f"| {q} | " + " | ".join(f"{v:.1%}" for v in r) + " |")
    w("\n**By phase of play** (the attacking team's phase):\n")
    w("| phase | frames | mean TC | mean reachable danger | mean cover | shot within 5 s |")
    w("|---|---|---|---|---|---|")
    for ph, r in by_phase.iterrows():
        w(f"| {ph} | {int(r['frames']):,} | {r['mean_tc']:.4f} | {r['mean_rd']:.4f} | {r['mean_cover']:.0%} | {r['shot_next5']:.1%} |")
    w("\n![frame level](../figs/w4_frame_level.png)\n")
    w("![maps](../figs/w4_maps.png)\n")

    w("## Possession by possession\n")
    w(f"{len(ps):,} possessions; {int(ps['shot'].sum())} with a shot, {int(ps['goal'].sum())} with a goal. "
      f"{len(pe):,} have frames at least 2 s before any shot (the rest are very short) and are used below.\n")
    w("| measure (≥ 2 s before any shot) | AUC for a shot in the possession |")
    w("|---|---|")
    for _, r in pl_auc.iterrows():
        w(f"| {r['measure']} | {r['auc']:.3f} |")
    pt, lo, hi = pl_boot
    w(f"\nPeak TC minus peak reachable danger: {pt:+.3f} AUC (95% match-bootstrap interval {lo:+.3f} to {hi:+.3f}).\n")
    w(f"Among possessions with a shot, peak TC and the possession's xG have a Spearman correlation of "
      f"{xg_corr:.2f}.\n")
    w("**Holding ball location fixed.** Possessions split into thirds by peak reachable danger, then by how "
      "much of it the attack controlled at its peak (peak TC / peak reachable danger). Share with a shot:\n")
    w("| ball danger third | " + " | ".join(within_p["mean"].columns) + " |")
    w("|---|" + "---|" * len(within_p["mean"].columns))
    for q in within_p.index:
        w(f"| {q} | " + " | ".join(f"{within_p['mean'].loc[q, c]:.1%} (n={int(within_p['size'].loc[q, c])})"
                                     for c in within_p["mean"].columns) + " |")
    w("\n![possession curves](../figs/w4_possession_curves.png)\n")
    w("![possession level](../figs/w4_possession_level.png)\n")

    w("## Team by match\n")
    w(f"{len(tm)} team-matches. Spearman correlation of a team's Threat Conceded exposure in a match with "
      f"its xG conceded: **{team_corr['xg_conceded']:.2f}**; with shots conceded: "
      f"{team_corr['shots_conceded']:.2f}. Exposure grows with how long the opponent has the ball, which "
      "also drives shots, so this is not yet a measure of defensive quality per possession.\n")
    w("![team matches](../figs/w4_team_matches.png)\n")

    w("## Caveats\n")
    w("- Descriptive only: no controls, no cross-validation, frames within a possession are not independent.")
    w("- Velocities are smoothed over ±0.3 s, so a frame's state uses 0.3 s of future positions.")
    w("- Possessions and outcomes come from the SkillCorner event layer; goals not on a shot-tagged "
      "possession are missed (see the README).")
    w("- About two fifths of the defense's weighted control comes from players whose positions are "
      f"extrapolated (frame median {frames['def_extrap_share'].median():.0%}).\n")
    w("Outputs: `data/derived/threat/<match>_frames.csv.gz` (frame level), "
      "`data/derived/threat_possessions.csv`, `data/derived/threat_team_matches.csv`.")
    (REPORTS / "w4_threat.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
