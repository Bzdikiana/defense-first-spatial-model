"""
Week 2: pitch-control base, sanity figures and checks.

1. Sample frames: attacking control + the share of defending control held by
   extrapolated players, for a high-detection and a low-detection defending-
   third frame and the match-winning shot.
2. Leave-one-out on the high-detection frame: per-defender suppression.
3. Pass calibration over all matches: control of the passing team at the ball's
   destination vs pass success, for three vmax choices.
4. How much defending control comes from extrapolated positions, near the ball
   vs the whole pitch.
5. Timing.

Run from the repo root:  py scripts/02_pitch_control.py
Needs scripts/01_data_audit.py to have produced data/derived/v95_by_player_half.csv.
"""
from __future__ import annotations

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
from dsv import data as D, pitchcontrol as PC, viz as V  # noqa: E402

MATCH_ID = int(os.environ.get("MATCH_ID", "1925299"))
DATA_DIR = Path(os.environ.get("DSV_DATA_DIR", ROOT / "data"))
FIGS, REPORTS, DERIVED = ROOT / "figs", ROOT / "reports", DATA_DIR / "derived"
FIGS.mkdir(exist_ok=True); REPORTS.mkdir(exist_ok=True)
GRID = PC.make_grid()
THIRD = 105.0 / 3


def load_match(mid):
    md = DATA_DIR / "matches" / str(mid)
    meta = D.load_match_metadata(md, mid)
    players, ball = D.normalize_direction(*D.load_tracking(md, mid, meta), meta)
    return meta, D.add_velocity(players), ball, D.load_events(md, mid)


def frame_context(players, ball):
    """Per-frame detected count and whether the ball is in the defending
    team's own third (home defends -x)."""
    ctx = ball.set_index("frame")[["bx", "by", "ball_detected", "possession_group"]].copy()
    ctx["n_detected"] = players.groupby("frame")["is_detected"].sum()
    home_def = ctx["possession_group"] == "away team"
    away_def = ctx["possession_group"] == "home team"
    ctx["def_third"] = (home_def & (ctx["bx"] < -105 / 2 + THIRD)) | (away_def & (ctx["bx"] > 105 / 2 - THIRD))
    return ctx


def pick_frames(ctx, events, meta):
    """High- and low-detection defending-third frames, plus the end of the
    possession before the first goal (or the first shot if there is none).
    'Low' is the 10th percentile of detected count: the minimum is 0/22,
    which is a replay or camera cut rather than a defending situation."""
    cand = ctx[ctx["def_third"] & ctx["ball_detected"]]
    hi = int(cand["n_detected"].idxmax())
    q10 = cand["n_detected"].quantile(0.10)
    lo = int((cand["n_detected"] - q10).abs().idxmin())
    pp = events[events["event_type"] == "player_possession"]
    goal = pp[pp["game_interruption_after"] == "goal_for"]
    key = goal if len(goal) else pp[pp["end_type"] == "shot"]
    ev = key.iloc[0]
    what = "Goal" if len(goal) else "Shot"
    # (frame, title, attacking team override). For the goal the attacking team
    # comes from the event: tracking's possession flag can flip on a deflection.
    return [
        (hi, f"High detection ({int(ctx.loc[hi, 'n_detected'])}/22), defending third", None),
        (lo, f"Low detection ({int(ctx.loc[lo, 'n_detected'])}/22), defending third", None),
        (int(ev["frame_end"]),
         f"{what}: {ev['player_name']} ({meta.team_name(ev['team_id'])}) {int(ev['minute_start'])}'",
         bool(ev["team_id"] == meta.home_team_id)),
    ]


def sample_frame_figure(players, ball, meta, vm, frames):
    fig, axes = plt.subplots(len(frames), 2, figsize=(12, 3.7 * len(frames)), layout="constrained")
    for row, (fr, title, att_home) in zip(axes, frames):
        s = PC.frame_state(players, ball, fr, meta, vm)
        if att_home is not None:
            s.possession = "home team" if att_home else "away team"
        r = PC.pitch_control(s, GRID)
        dh = not s.attacking_home
        V.draw_pitch(row[0]); V.draw_pitch(row[1])
        im0 = V.plot_surface(row[0], GRID, r.attack)
        V.plot_players(row[0], s)
        row[0].set_title(f"{title}\nattacking ({'home' if s.attacking_home else 'away'}) control",
                         fontsize=9)
        im1 = V.plot_surface(row[1], GRID, r.extrapolated_share(home=dh), cmap="Purples")
        V.plot_players(row[1], s, arrows=False)
        row[1].set_title("share of defending control held by extrapolated players", fontsize=9)
    fig.colorbar(im0, ax=axes[:, 0], shrink=0.5, label="attacking control")
    fig.colorbar(im1, ax=axes[:, 1], shrink=0.5, label="extrapolated share")
    fig.suptitle("red = home, blue = away; filled = detected, hollow = extrapolated; "
             "arrows = 1 s of velocity. Home attacks +x.", fontsize=9)
    fig.savefig(FIGS / "w2_sample_frames.png", dpi=120); plt.close(fig)


def loo_figure(players, ball, meta, vm, frame):
    s = PC.frame_state(players, ball, frame, meta, vm)
    t0 = time.perf_counter()
    loo = PC.leave_one_out(s, GRID)
    t_loo = time.perf_counter() - t0
    sup = loo.suppression().sort_values(ascending=False)
    roles = {pid: f"{meta.players[pid].role or '?'} #{meta.players[pid].number}"
             for pid in loo.defender_ids}
    det = dict(zip(s.player_id, s.detected))
    top = sup.index[0]
    i_top = int(np.flatnonzero(loo.defender_ids == top)[0])

    fig = plt.figure(figsize=(13, 3.9))
    ax0 = fig.add_subplot(1, 3, 1); ax1 = fig.add_subplot(1, 3, 2); ax2 = fig.add_subplot(1, 3, 3)
    V.draw_pitch(ax0); V.plot_surface(ax0, GRID, loo.base_attack); V.plot_players(ax0, s)
    ax0.set_title("attacking control, all players", fontsize=9)
    V.draw_pitch(ax1)
    im = V.plot_surface(ax1, GRID, loo.delta[i_top], cmap="Reds", vmin=0, vmax=float(loo.delta[i_top].max()))
    V.plot_players(ax1, s, arrows=False, label_ids={top: roles[top]})
    ax1.set_title(f"attacking control gained if {roles[top]} is removed", fontsize=9)
    fig.colorbar(im, ax=ax1, shrink=0.7)
    labels = [f"{roles[p]}{'' if det[p] else ' (extrap.)'}" for p in sup.index]
    ypos = np.arange(len(sup))[::-1]
    ax2.barh(ypos, sup.to_numpy(), color=V.HOME_C if loo.defending_home else V.AWAY_C)
    ax2.set_yticks(ypos, labels)
    ax2.set_xlabel("attacking control suppressed (m², unweighted)")
    ax2.set_title("leave-one-out suppression per defender", fontsize=9)
    fig.tight_layout(); fig.savefig(FIGS / "w2_leave_one_out.png", dpi=120); plt.close(fig)
    return sup.rename(index=roles), t_loo


def pass_destinations(events, ball, meta):
    """Completed and failed passes with the ball's destination: the ball
    position at the start of the next player possession (either team), if it
    starts within 5 s of the pass in the same period."""
    pp = events[events["event_type"] == "player_possession"].sort_values("frame_start")
    nxt = pp[["frame_start", "period"]].shift(-1)
    pp = pp.assign(next_frame=nxt["frame_start"], next_period=nxt["period"])
    ps = pp[(pp["end_type"] == "pass") & pp["pass_outcome"].isin(["successful", "unsuccessful"])
            & (pp["next_period"] == pp["period"])
            & (pp["next_frame"] - pp["frame_end"]).between(1, 50)].copy()
    b = ball.set_index("frame")[["bx", "by"]]
    ps[["dest_x", "dest_y"]] = b.reindex(ps["next_frame"].astype(int)).to_numpy()
    ps["passer_home"] = ps["team_id"] == meta.home_team_id
    ps["success"] = ps["pass_outcome"] == "successful"
    return ps.dropna(subset=["dest_x"])


def control_at(state, x, y):
    """Passing-team-agnostic per-team control at a single point."""
    g = PC.Grid(np.array([x]), np.array([y]))
    return PC.pitch_control(state, g)


def auc(y, p):
    y, p = np.asarray(y, bool), np.asarray(p, float)
    ranks = pd.Series(p).rank().to_numpy()
    n1, n0 = y.sum(), (~y).sum()
    return (ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def pass_calibration(ids, vmax_variants):
    rows = []
    for mid in ids:
        meta, players, ball, events = load_match(mid)
        ps = pass_destinations(events, ball, meta)
        ps = ps[ps["frame_end"].isin(set(players["frame"]))]  # drop dead-ball pass frames
        frames = set(ps["frame_end"].astype(int))
        pf = players[players["frame"].isin(frames)]
        bf = ball[ball["frame"].isin(frames)]
        for _, p in ps.iterrows():
            for name, vt in vmax_variants.items():
                vm = PC.vmax_lookup(vt, mid) if vt is not None else {}
                s = PC.frame_state(pf, bf, int(p["frame_end"]), meta, vm)
                r = control_at(s, p["dest_x"], p["dest_y"])
                rows.append(dict(match_id=mid, variant=name, success=p["success"],
                                 pc=float(r.team(bool(p["passer_home"]))[0, 0])))
        print(f"  passes {mid}: {len(ps)}")
    return pd.DataFrame(rows)


def calibration_figure(cal):
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    bins = np.linspace(0, 1, 11)
    for (name, g), c in zip(cal.groupby("variant", sort=False), ["#8a8f98", "#dd8a2c", "#2b6cb0"]):
        b = pd.cut(g["pc"], bins, include_lowest=True)
        agg = g.groupby(b, observed=True).agg(pc=("pc", "mean"), rate=("success", "mean"), n=("pc", "size"))
        agg = agg[agg["n"] >= 20]
        ax.plot(agg["pc"], agg["rate"], "o-", color=c, label=name, ms=4)
    ax.plot([0, 1], [0, 1], "--", color="#bbbbbb", lw=1)
    ax.set_xlabel("passing team's control at ball destination")
    ax.set_ylabel("observed pass completion")
    ax.set_title("Pass calibration, all matches", fontsize=10)
    ax.legend(fontsize=8, frameon=False); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIGS / "w2_pass_calibration.png", dpi=130); plt.close(fig)


def extrapolated_control_stats(players, ball, meta, vm, ctx, n=200, seed=0):
    """Share of the defending team's control held by extrapolated players,
    within 20 m of the ball, in the defending third, and over the pitch."""
    frames = ctx[ctx["possession_group"].notna() & ctx["ball_detected"]].sample(n, random_state=seed).index
    cells = GRID.cells
    out, times = [], []
    for fr in frames:
        s = PC.frame_state(players, ball, int(fr), meta, vm)
        t0 = time.perf_counter(); r = PC.pitch_control(s, GRID); times.append(time.perf_counter() - t0)
        dh = not s.attacking_home
        d = s.is_home == dh
        dc = r.pc[:, d]
        ext = dc[:, ~s.detected[d]].sum(axis=1)
        tot = dc.sum(axis=1)
        near = np.hypot(cells[:, 0] - s.ball[0], cells[:, 1] - s.ball[1]) < 20
        own3 = (cells[:, 0] < -105 / 2 + THIRD) if dh else (cells[:, 0] > 105 / 2 - THIRD)
        out.append(dict(
            frame=fr, def_third=bool(ctx.loc[fr, "def_third"]),
            near_ball=ext[near].sum() / max(tot[near].sum(), 1e-9),
            own_third=ext[own3].sum() / max(tot[own3].sum(), 1e-9),
            pitch=ext.sum() / max(tot.sum(), 1e-9),
        ))
    return pd.DataFrame(out), np.array(times)


def main():
    v95 = pd.read_csv(DERIVED / "v95_by_player_half.csv")
    vt = PC.vmax_table(v95)
    meta, players, ball, events = load_match(MATCH_ID)
    vm = PC.vmax_lookup(vt, MATCH_ID)
    ctx = frame_context(players, ball)

    frames = pick_frames(ctx, events, meta)
    sample_frame_figure(players, ball, meta, vm, frames)
    sup, t_loo = loo_figure(players, ball, meta, vm, frames[0][0])
    ext, times = extrapolated_control_stats(players, ball, meta, vm, ctx)
    print("sample frames, LOO, extrapolation stats done")

    raw = v95.assign(vmax=v95["v95"])
    variants = {"uniform 5 m/s": None, "raw v95": raw, "v95, group floor, GK 5 m/s": vt}
    ids = [i for i in D.list_open_matches(DATA_DIR)
           if (DATA_DIR / "matches" / str(i) / f"{i}_tracking_extrapolated.jsonl").exists()]
    cal_csv = DERIVED / "w2_pass_calibration.csv"
    if cal_csv.exists() and not os.environ.get("RECALIBRATE"):
        cal = pd.read_csv(cal_csv)   # cached; set RECALIBRATE=1 to recompute
    else:
        cal = pass_calibration(ids, variants)
        cal.to_csv(cal_csv, index=False)
    calibration_figure(cal)

    # ----- report ------------------------------------------------------------
    eps = 1e-6
    summ = []
    for name, g in cal.groupby("variant", sort=False):
        p = g["pc"].clip(eps, 1 - eps); y = g["success"]
        summ.append(dict(variant=name, n=len(g), auc=auc(y, p),
                         brier=float(((p - y) ** 2).mean()),
                         logloss=float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
                         pc_success=g.loc[y, "pc"].mean(), pc_fail=g.loc[~y, "pc"].mean()))
    summ = pd.DataFrame(summ)

    L = []
    w = L.append
    w("# Week 2: pitch-control base\n")
    w(f"Focal match {meta.home_name} vs {meta.away_name} ({MATCH_ID}); pass calibration over "
      f"all {len(ids)} open-data matches. Module: `dsv/pitchcontrol.py`.\n")
    w("## Model as implemented")
    p = PC.PCParams()
    w(f"- Spearman time-to-intercept: t_react {p.t_react} s, sigma {p.sigma} s, lambda {p.lam}/s "
      f"(goalkeepers x{p.lam_gk_mult}), ball speed {p.ball_speed} m/s, integrated for "
      f"{p.t_max} s after the ball arrives, dT {p.dt} s, stop when all cells reach "
      f"{1 - p.tol:.0%} control. 1 m grid (105 x 68).")
    w("- Each dT step is integrated exactly (exponential decay of the uncontrolled share), "
      "so total control never exceeds 1. Plain Euler overshoots on cells several players "
      "reach before the ball.")
    w("- vmax per player and half: v95 from detected runs, floored at the 25th percentile of "
      "the player's position group across all 20 matches. Goalkeepers use 5 m/s: their v95 "
      "(about 3 m/s) reflects how little they move, while their 99.9th-percentile speed is about 6 m/s.")
    g = vt.groupby("group")[["v95", "vmax_floor", "vmax"]].mean().round(2)
    w("\n| group | mean v95 | floor | mean vmax |\n|---|---|---|---|")
    for grp, r in g.iterrows():
        w(f"| {grp} | {r['v95']:.2f} | {r['vmax_floor']:.2f} | {r['vmax']:.2f} |")
    w("- `amax` is not used: the brief's arrival-time formula has no acceleration term. "
      "It returns in the Week 7 counterfactual motion model.")
    w("- Leave-one-out recomputes the model without defender i, which is the renormalization: "
      "the accrual ODE gives i's share to whoever else reaches each cell. All 11 removals plus "
      "the base run in one batched integration.\n")

    w("## Speed")
    w(f"- One frame, full grid: median {np.median(times):.2f} s (p90 {np.percentile(times, 90):.2f} s) "
      f"over {len(times)} frames.")
    w(f"- Leave-one-out (base + 11 removals): {t_loo:.1f} s per frame.")
    w(f"- Whole match at 1 m and 10 fps would be about {np.median(times) * 48000 / 3600:.1f} h; "
      "Threat Conceded series will run at 2 to 5 fps or on a 2 m grid (Week 4 decision).\n")

    w("## Pass calibration (does the surface mean anything?)")
    w("For every completed or failed pass: control of the passing team at the ball's destination "
      "(ball position when the next possession starts, within 5 s), evaluated at the pass frame. "
      "A failed pass's destination is where an opponent took the ball, which favours the model "
      "slightly. Spearman (2017) validated the same way.\n")
    w("| vmax | passes | AUC | Brier | log loss | mean control, completed | mean control, failed |")
    w("|---|---|---|---|---|---|---|")
    for _, r in summ.iterrows():
        w(f"| {r['variant']} | {r['n']:,} | {r['auc']:.3f} | {r['brier']:.3f} | {r['logloss']:.3f} "
          f"| {r['pc_success']:.2f} | {r['pc_fail']:.2f} |")
    ours = cal[cal["variant"] == "v95, group floor, GK 5 m/s"]
    mid = ours[ours["pc"].between(0.4, 0.6)]
    w(f"\n- The surface separates completed from failed passes well, but it is underconfident "
      f"in the middle: passes to a destination with 40-60% control complete "
      f"{mid['success'].mean():.0%} of the time (n={len(mid):,}). Some of this is selection "
      f"(passers pick targets they expect to reach), but it also suggests the default "
      f"parameters give defenders too much reach. Next: fit t_react, sigma and lambda on these "
      f"passes with leave-one-match-out CV.")
    w("- Per-player vmax barely changes pass calibration. It is kept because it decides *who* "
      "covers a region, which matters for attribution rather than pass outcomes; the Week 5 "
      "sensitivity check should re-test it.")
    w("\nSee [figs/w2_pass_calibration.png](../figs/w2_pass_calibration.png).\n")

    w("## How much defending control is extrapolated?")
    w(f"Over {len(ext)} random in-possession frames of the focal match, the share of the "
      "defending team's control held by players whose position is extrapolated:\n")
    w("| region | all frames | ball in defending third |\n|---|---|---|")
    dt3 = ext[ext["def_third"]]
    for col, lab in [("near_ball", "within 20 m of the ball"), ("own_third", "defending third"),
                     ("pitch", "whole pitch")]:
        w(f"| {lab} | {ext[col].mean():.1%} | {dt3[col].mean():.1%} (n={len(dt3)}) |")
    w("")

    w("## Sample frames")
    for fr, title, _ in frames:
        w(f"- frame {fr}: {title}")
    w("\n![sample frames](../figs/w2_sample_frames.png)\n")
    w("## Leave-one-out on the high-detection frame")
    w("Unweighted suppression (m² of attacking control each defender removes); Week 3 weights "
      "this by danger and accessibility.\n")
    for role, v in sup.items():
        w(f"- {role}: {v:.0f} m²")
    w("\n![leave-one-out](../figs/w2_leave_one_out.png)\n")
    w("Unweighted, forwards rank high because they 'suppress' attacking control of empty "
      "space in the attacking team's own half. The goalkeeper ranks first because removing "
      "them hands over the goalmouth; once Week 3's danger surface weights this, keepers "
      "will dominate attribution unless Week 6 handles them separately.\n")
    w("## What raw control does not do yet")
    w("- The attacking team 'controls' its own empty half (the focal frames show 60-80% "
      "attacking control with the ball deep in the defending third). That is the space behind "
      "the ball the brief warns about; Week 3's danger surface and accessibility weight fix it.")
    w("- No offside: attackers beyond the last defender control space they cannot legally use. "
      "Handled in the accessibility weight.")

    (REPORTS / "w2_pitch_control.md").write_text("\n".join(L), encoding="utf-8")
    print(summ.round(3).to_string())
    print(f"report -> {REPORTS / 'w2_pitch_control.md'}")


if __name__ == "__main__":
    main()
