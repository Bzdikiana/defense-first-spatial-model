"""
Week 3: the danger surface V(x) and the accessibility weight A(x | ball).

1. Expected threat (xT, 12 x 8 zones) fitted on the 20 matches' player
   possessions, with each zone's shot value from the xG model (a typical
   A-League shot from the zone centre). Compared with Karun Singh's published
   grid, and validated by how well the xT of where a possession starts ranks
   whether that team possession produces a shot (match-grouped 5-fold CV;
   baselines: Singh's grid, distance to goal).
2. Ball-transition model (where the ball goes next): isotropic Gaussian
   (Spearman-style), forward-shifted Gaussian, and 2- and 3-component mixtures,
   compared by held-out log-likelihood (match-grouped 5-fold CV).
3. One frame decomposed: attacking control (fitted parameters, offside
   attackers excluded) x danger x accessibility = Threat Conceded density.

Run from the repo root:  py scripts/05_value.py   (after 03 and 04)
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
from dsv import data as D, metrics as M, pitchcontrol as PC, value as V, viz as VZ, xg  # noqa: E402

DATA_DIR = Path(os.environ.get("DSV_DATA_DIR", ROOT / "data"))
FIGS, REPORTS, DERIVED = ROOT / "figs", ROOT / "reports", DATA_DIR / "derived"
NX, NY = 12, 8
N_FOLDS = 5
FOCAL = 1925299
TRANSITION_SPECS = {
    "isotropic Gaussian (Spearman-style)": dict(k=1, isotropic=True),
    "forward-shifted Gaussian": dict(k=1, forward=True),
    "2-component mixture": dict(k=2, forward=True),
    "3-component mixture": dict(k=3, forward=True),
}
TCOLS = ("bx", "by", "tx", "ty", "half_l", "half_w")


def build_tables():
    """xT actions, possession starts (for validation) and ball transitions."""
    cache = DERIVED / "value_tables.pkl"
    if cache.exists() and not os.environ.get("REBUILD"):
        return pd.read_pickle(cache)
    acts, poss, trs = [], [], []
    for mid in D.list_open_matches(DATA_DIR):
        md = DATA_DIR / "matches" / str(mid)
        meta = D.load_match_metadata(md, mid)
        ev = D.load_events(md, mid)
        acts.append(V.xt_actions(ev, meta))
        pp = ev[(ev["event_type"] == "player_possession")].dropna(subset=["x_start", "y_start"])
        poss.append(pd.DataFrame(dict(
            match_id=mid, x01=np.clip(pp["x_start"] / meta.pitch_length + 0.5, 0, 1 - 1e-9),
            y01=np.clip(pp["y_start"] / meta.pitch_width + 0.5, 0, 1 - 1e-9),
            dist_goal=np.hypot(meta.pitch_length / 2 - pp["x_start"], pp["y_start"]),
            lead_to_shot=pp["lead_to_shot"].astype(bool))))
        pl, ball = D.normalize_direction(*D.load_tracking(md, mid, meta), meta)
        trs.append(V.ball_transitions(ev, ball, meta))
        print(f"  {mid}", flush=True)
    out = (pd.concat(acts, ignore_index=True), pd.concat(poss, ignore_index=True),
           pd.concat(trs, ignore_index=True))
    pd.to_pickle(out, cache)
    return out


def shot_value_fn():
    """xG of a typical A-League shot (median context, not a header) from a
    given distance and angle."""
    sc = pd.read_csv(DERIVED / "skillcorner_shots.csv")
    ctx = sc[["n_block", "gk_off_line", "gk_dist_goal", "press_dist", "n_close"]].median()

    def xg_at(dist, angle):
        df = pd.DataFrame({"dist": dist.ravel(), "log_dist": np.log(np.maximum(dist.ravel(), 0.5)),
                           "angle": angle.ravel(), "header": 0.0, **ctx.to_dict()})
        return xg.xg_for_shots(df).reshape(dist.shape)
    return xg_at, ctx


def match_folds(ids):
    order = sorted(ids)
    return {m: i % N_FOLDS for i, m in enumerate(order)}


def main():
    acts, poss, trs = build_tables()
    xg_at, ctx = shot_value_fn()
    g = V.zone_shot_value(NX, NY, xg_at)
    singh = np.array(json.loads((DERIVED / "singh_xt_12x8.json").read_text()))
    fold = match_folds(acts["match_id"].unique())

    # ----- xT: fit, compare, validate -----------------------------------------
    xt = V.fit_xt(acts, g)
    oof_zone, oof_interp = np.zeros(len(poss)), np.zeros(len(poss))
    for k in range(N_FOLDS):
        tr_m = [m for m, f in fold.items() if f != k]
        te = poss["match_id"].map(fold).to_numpy() == k
        xt_k = V.fit_xt(acts[acts["match_id"].isin(tr_m)], g)
        oof_zone[te] = V.zone_lookup(xt_k, poss.loc[te, "x01"], poss.loc[te, "y01"])
        oof_interp[te] = V.zone_value_at(xt_k, poss.loc[te, "x01"], poss.loc[te, "y01"])
    y = poss["lead_to_shot"].to_numpy()
    xm, ym = (poss["x01"].to_numpy() - 1) * 105.0, (poss["y01"].to_numpy() - 0.5) * 68.0
    ang = np.abs(np.arctan2(3.66 - ym, -xm) - np.arctan2(-3.66 - ym, -xm))
    xt_val = pd.DataFrame([
        dict(model="xT fitted on A-League, interpolated (used for TC; out of fold)", auc=M.auc(y, oof_interp)),
        dict(model="xT fitted on A-League, zone value (out of fold)", auc=M.auc(y, oof_zone)),
        dict(model="Singh's published xT, interpolated", auc=M.auc(y, V.zone_value_at(singh, poss["x01"], poss["y01"]))),
        dict(model="xG of a typical shot from here", auc=M.auc(y, xg_at(np.hypot(xm, ym), ang))),
        dict(model="distance to goal (closer = higher)", auc=M.auc(y, -poss["dist_goal"].to_numpy())),
    ])
    corr = float(np.corrcoef(xt.ravel(), singh.ravel())[0, 1])

    # ----- transition model ----------------------------------------------------
    tfold = trs["match_id"].map(fold).to_numpy()
    rows = []
    for name, kw in TRANSITION_SPECS.items():
        ll = np.zeros(len(trs))
        for k in range(N_FOLDS):
            m = V.fit_transition(trs[tfold != k], **kw)
            te = tfold == k
            ll[te] = m.logpdf(*[trs.loc[te, c] for c in TCOLS])
        rows.append(dict(model=name, heldout_ll=float(ll.mean()),
                         per_match=pd.Series(ll).groupby(trs["match_id"].to_numpy()).mean()))
        print(f"  transition {name}: held-out log-lik {ll.mean():.4f}", flush=True)
    tr_table = pd.DataFrame(rows).drop(columns="per_match")
    best_name = tr_table.loc[tr_table["heldout_ll"].idxmax(), "model"]
    best = V.fit_transition(trs, **TRANSITION_SPECS[best_name])
    iso = V.fit_transition(trs, **TRANSITION_SPECS["isotropic Gaussian (Spearman-style)"])
    per = {r["model"]: r["per_match"] for r in rows}
    wins = int((per[best_name] > per["isotropic Gaussian (Spearman-style)"]).sum())

    # ----- save ---------------------------------------------------------------
    (DERIVED / "value_fit.json").write_text(json.dumps(dict(
        xt_12x8=np.round(xt, 6).tolist(), shot_value_context=ctx.round(3).to_dict(),
        transition=dict(name=best_name, weights=best.weights.tolist(), mu_x=best.mu_x.tolist(),
                        sx=best.sx.tolist(), sy=best.sy.tolist())), indent=2))

    # ----- figures ------------------------------------------------------------
    xt_figure(xt, singh)
    transition_figure(best, iso)
    frame_info = decomposition_figure(xt, best)
    write_report(acts, poss, trs, xt, singh, corr, xt_val, tr_table, best_name, best, wins, frame_info)
    print(xt_val.round(4).to_string(index=False))
    print(tr_table.round(4).to_string(index=False))


def xt_figure(xt, singh):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    grid = PC.make_grid(105, 68)
    vmax = max(xt.max(), singh.max())
    for ax, z, title in ((axes[0], xt, "xT fitted on A-League (shot value from our xG)"),
                         (axes[1], singh, "Singh's published xT")):
        VZ.draw_pitch(ax)
        im = VZ.plot_surface(ax, grid, V.zone_surface(z, grid, 105, 68), cmap="magma", vmin=0, vmax=vmax, alpha=1)
        ax.set_title(title + " (attacking →)", fontsize=9)
    fig.colorbar(im, ax=axes, shrink=0.8, label="xT")
    fig.savefig(FIGS / "w3_xt.png", dpi=130, bbox_inches="tight"); plt.close(fig)


def transition_figure(best, iso):
    grid = PC.make_grid(105, 68)
    balls = [(-30.0, 0.0), (10.0, -20.0), (35.0, 10.0)]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.2))
    for row, (m, name) in zip(axes, ((iso, "isotropic Gaussian"), (best, "fitted mixture"))):
        for ax, b in zip(row, balls):
            VZ.draw_pitch(ax)
            s = m.surface(np.array(b), grid, 105, 68)
            VZ.plot_surface(ax, grid, s / s.max(), cmap="Greens", vmin=0, vmax=1, alpha=1)
            ax.scatter(*b, s=18, c="k", zorder=5)
            ax.set_title(f"{name}, ball at ({b[0]:.0f}, {b[1]:.0f})", fontsize=9)
    fig.suptitle("Where the ball goes next (attacking →, density scaled to its maximum)", fontsize=10)
    fig.tight_layout(); fig.savefig(FIGS / "w3_transition.png", dpi=120); plt.close(fig)


def decomposition_figure(xt, trans):
    """The W2 high-detection defending-third frame of the focal match:
    control x danger x accessibility."""
    md = DATA_DIR / "matches" / str(FOCAL)
    meta = D.load_match_metadata(md, FOCAL)
    players, ball = D.normalize_direction(*D.load_tracking(md, FOCAL, meta), meta)
    players = D.add_velocity(players)
    vmax = PC.vmax_lookup(PC.vmax_table(D.compute_v95(players, meta), floors=PC.SKILLCORNER_VMAX_FLOORS), FOCAL)
    params = PC.SKILLCORNER_OPEN_DATA
    grid = PC.make_grid(meta.pitch_length, meta.pitch_width)
    det = players.groupby("frame")["is_detected"].sum()
    b = ball.set_index("frame")
    third = meta.pitch_length / 3
    home_def = (b["possession_group"] == "away team") & (b["bx"] < -meta.pitch_length / 2 + third)
    away_def = (b["possession_group"] == "home team") & (b["bx"] > meta.pitch_length / 2 - third)
    cand = det[(home_def | away_def).reindex(det.index, fill_value=False) & b["ball_detected"].reindex(det.index)]
    frame = int(cand.idxmax())
    s = PC.frame_state(players, ball, frame, meta, vmax, params)
    att_home = bool(s.attacking_home)
    off = V.offside_players(s, meta, att_home)
    pc = PC.pitch_control(s, grid, params, exclude=off)
    sgn = 1.0 if att_home else -1.0
    flip = (lambda a: a) if att_home else (lambda a: a[::-1, ::-1])  # noqa: E731
    vsurf = flip(V.zone_surface(xt, grid, meta.pitch_length, meta.pitch_width))
    asurf = flip(trans.surface(sgn * s.ball, grid, meta.pitch_length, meta.pitch_width))
    dens = pc.attack * vsurf * asurf
    tc = float(dens.sum())

    fig, axes = plt.subplots(1, 4, figsize=(17, 3.6))
    panels = [(pc.attack, "attacking control", "RdBu_r", 1.0),
              (vsurf, "danger V (xT)", "magma", vsurf.max()),
              (asurf, "accessibility A (next ball location)", "Greens", asurf.max()),
              (dens, f"control x V x A  (TC = {tc:.4f} xT)", "Reds", dens.max())]
    for ax, (z, title, cmap, vm) in zip(axes, panels):
        VZ.draw_pitch(ax)
        VZ.plot_surface(ax, grid, z, cmap=cmap, vmin=0, vmax=vm, alpha=0.9)
        VZ.plot_players(ax, s, arrows=False)
        ax.set_title(title, fontsize=9)
    for j in np.flatnonzero(off):
        axes[0].annotate("offside", s.pos[j], xytext=(4, -9), textcoords="offset points", fontsize=7)
    fig.suptitle(f"{meta.home_name} vs {meta.away_name}, frame {frame}: "
                 f"{'home' if att_home else 'away'} attacking; {int(det[frame])}/22 detected", fontsize=10)
    fig.tight_layout(); fig.savefig(FIGS / "w3_tc_decomposition.png", dpi=120); plt.close(fig)
    return dict(frame=frame, tc=tc, n_offside=int(off.sum()), detected=int(det[frame]))


def write_report(acts, poss, trs, xt, singh, corr, xt_val, tr_table, best_name, best, wins, info):
    L = []
    w = L.append
    w("# Week 3: danger and accessibility\n")
    w("Threat Conceded weights the attacking team's pitch control by **danger** V(x) (how much a "
      "location is worth) and **accessibility** A(x | ball) (how likely the ball is to go there next). "
      "Both are fitted on the 20 SkillCorner open-data matches. Code: `dsv/value.py`; script: "
      "`scripts/05_value.py`.\n")

    w("## Danger: expected threat\n")
    w(f"Markov xT (Singh 2018) on 12 x 8 zones from {len(acts):,} actions in player possessions "
      f"({(acts['kind'] == 'shot').sum()} shots; {acts.loc[acts['kind'] == 'move', 'success'].mean():.0%} "
      "of moves kept the ball). Each possession contributes a carry (start to end, if at least 3 m) and a "
      "release (a shot, or a move to where the team's next possession starts; failed if the other team "
      "gets the ball). A zone's shot value is the xG model's value for a typical A-League shot from the "
      "zone centre (median blockers, keeper position and pressure; not a header), which avoids sparse "
      "zone-level goal counts.\n")
    w(f"The fitted grid correlates **{corr:.2f}** with Karun Singh's published grid (fitted on "
      f"European event data). Its maximum is higher ({xt.max():.3f} vs {singh.max():.3f}), in the zones "
      "just in front of goal, where its shot value comes from the xG model rather than counts.\n")
    w("Does a possession's starting xT rank whether that team possession produces a shot? "
      f"({len(poss):,} player possessions, {poss['lead_to_shot'].mean():.0%} in team possessions "
      "that led to a shot; the A-League fit is out of fold by match.)\n")
    w("| model | AUC |")
    w("|---|---|")
    for _, r in xt_val.iterrows():
        w(f"| {r['model']} | {r['auc']:.3f} |")
    w("\nAll danger surfaces rank possessions within about 0.01 AUC of each other, and none clearly beats "
      "plain distance to goal: for this question, closeness to goal explains nearly everything, and finer xT "
      "grids (16 x 12, 24 x 16) do not change that. xT is kept as V because it values every location "
      "(including progression through midfield) in goal units. Whether the choice of V matters for Threat "
      "Conceded is tested where it counts, in the Week 5 shot-prediction gate, by rebuilding TC with each "
      "surface.")
    w("\n![xT](../figs/w3_xt.png)\n")

    w("## Accessibility: where the ball goes next\n")
    w(f"{len(trs):,} transitions: the ball at a possession's release to the ball where the next "
      "possession (either team) starts, in the releasing team's attacking frame. Shots and gaps over "
      "10 s are excluded. Each model is truncated to the pitch (closed form for axis-aligned Gaussians) "
      "and scored by held-out log-likelihood per transition (match-grouped 5-fold).\n")
    w("| model | held-out log-likelihood |")
    w("|---|---|")
    for _, r in tr_table.iterrows():
        tag = " (selected)" if r["model"] == best_name else ""
        w(f"| {r['model']}{tag} | {r['heldout_ll']:.4f} |")
    w(f"\nThe selected model beats the isotropic Gaussian in {wins} of 20 matches. Fitted on all "
      "transitions:\n")
    w("| component | weight | forward shift (m) | spread along x (m) | spread across (m) |")
    w("|---|---|---|---|---|")
    for i in np.argsort(-best.weights):
        w(f"| {i + 1} | {best.weights[i]:.2f} | {best.mu_x[i]:.1f} | {best.sx[i]:.1f} | {best.sy[i]:.1f} |")
    w("\nMost next touches are short and roughly centred on the ball; a minority are long and played "
      "forward. A single Gaussian has to compromise between the two.\n")
    w("![transition](../figs/w3_transition.png)\n")

    w("## Putting it together on one frame\n")
    w("Attacking control uses `SKILLCORNER_OPEN_DATA` with attackers in offside positions (in the "
      "opponents' half, ahead of the ball and of the second-last defender) left out. Threat Conceded for "
      "the frame is the sum over cells of control x V x A. A sums to 1, so TC is the expected xT of where "
      "the ball goes next, counted only where the attack would control it.\n")
    w(f"Frame {info['frame']} ({info['detected']}/22 players detected, {info['n_offside']} attacker(s) "
      f"offside): TC = {info['tc']:.4f} xT.\n")
    w("![decomposition](../figs/w3_tc_decomposition.png)\n")

    w("## Caveats\n")
    w("- xT is fitted on 20 matches; zones far from goal rest on many actions, zones in the box on few. "
      "The xG-based shot value keeps those zones stable.")
    w("- The transition model ignores who has the ball and the game state; Week 4 can test whether "
      "conditioning on phase of play helps.")
    w("- Offside uses the frame's positions as they are, including extrapolated ones, so a mistracked "
      "last defender can flip an attacker's status.")
    w("- V is the value of the ball being at x for the team in possession; TC does not yet account for "
      "what the attack does after reaching x beyond that.\n")
    w("Outputs: `data/derived/value_fit.json` (xT grid, transition parameters).")
    (REPORTS / "w3_value.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
