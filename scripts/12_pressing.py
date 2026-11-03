"""
Week 7: what did the press do?

Every SkillCorner on-ball engagement (a defender pressing the ball carrier)
during an opponent possession is measured 3 s after the decision to press,
against what typically happens from the same situation, in the two-part
model's chance of a shot (dsv.counterfactual): on the ball (is the ball less
dangerous, or won?) and in the shape (does the defense cover the danger
better than typical?). The decision point is 1 s before SkillCorner's
engagement starts (the presser is then a median 9 m from the ball).

Also computed: the first-order "what if they had held their position"
comparison from the brief, and why it is not the main grade.

Checks: agreement with SkillCorner's outcome flags; whether a player's
pressing value is stable across their presses (split-half).

Writes data/derived/pressing/<match>.csv, data/derived/press_values.csv,
reports/w7_pressing.md and figs/w7_*.png. About 15 minutes.
"""
from __future__ import annotations

import importlib.util
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
from dsv import (counterfactual as CF, data as D, defense as DF, metrics as M,  # noqa: E402
                 pitchcontrol as PC, threat as T, viz as VZ)

DATA_DIR = Path(os.environ.get("DSV_DATA_DIR", ROOT / "data"))
DERIVED, REPORTS, FIGS = DATA_DIR / "derived", ROOT / "reports", ROOT / "figs"
CACHE = DERIVED / "pressing"
CACHE.mkdir(parents=True, exist_ok=True)
FPS = 10.0
LEAD_S, LEAD_ALT_S, HORIZON_S = 1.0, 2.0, 3.0
MIN_PRESSES = 30         # players shown in the leaderboard
EPS = 1e-6
OUTCOMES = ["beaten", "stopped danger", "reduced danger", "forced backward", "no flagged outcome"]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s06 = _load("s06", ROOT / "scripts" / "06_threat.py")
s07 = _load("s07", ROOT / "scripts" / "07_validation.py")
s10 = _load("s10", ROOT / "scripts" / "10_confirmation.py")
rng = np.random.default_rng(0)


def even(f):
    return int(f) - int(f) % 2


def outcome(ob):
    flag = lambda c: ob[c].astype(bool).to_numpy()  # noqa: E731
    return np.select([flag("beaten_by_possession") | flag("beaten_by_movement"), flag("stop_possession_danger"),
                      flag("reduce_possession_danger"), flag("force_backward")], OUTCOMES[:4], OUTCOMES[4])


# --------------------------------------------------------------------------- #
# Branches per press
# --------------------------------------------------------------------------- #
def match_presses(mid, vt, values, model):
    path = CACHE / f"{mid}.csv"
    if path.exists() and not os.environ.get("REBUILD"):
        return pd.read_csv(path)
    meta, players, ball, events, _ = s06.load(mid)
    fx = T._Frames(players, ball, meta, PC.vmax_lookup(vt, mid), model.params)
    grid = PC.make_grid(meta.pitch_length, meta.pitch_width, model.cell)
    poss = T.team_possessions(events, meta)
    ob = events[events["event_type"] == "on_ball_engagement"].copy()
    ob["outcome"] = outcome(ob)
    lead, lead_alt, hor = int(LEAD_S * FPS), int(LEAD_ALT_S * FPS), int(HORIZON_S * FPS)
    rows = []
    for _, e in ob.iterrows():
        fs = int(e["frame_start"])
        m = (poss["team_id"] != e["team_id"]) & (poss["frame_start"] <= fs) & (poss["frame_end"] >= fs)
        if not m.any():
            continue
        p = poss[m].iloc[0]
        p0 = int(np.ceil(p["frame_start"] / 2) * 2)
        t_dec = max(even(fs - lead), p0)
        t_alt = max(even(fs - lead_alt), p0)
        t1 = t_dec + hor
        shot = bool(pd.notna(p["first_shot_frame"]) and p["first_shot_frame"] <= t1)
        alive = bool(t1 <= p["frame_end"])
        if pd.notna(p["first_shot_frame"]) and p["first_shot_frame"] <= t_dec:
            continue
        ah = bool(p["attacking_home"])
        row = dict(match_id=mid, event_id=e["event_id"], player_id=int(e["player_id"]), team_id=int(e["team_id"]),
                   subtype=e["event_subtype"], outcome=e["outcome"], possession_id=int(p["possession_id"]),
                   frame_start=fs, frame_end=int(e["frame_end"]), t_dec=t_dec, t1=t1, lead_s=(fs - t_dec) / FPS,
                   shot_by_t1=shot, alive_t1=alive, t_in_possession_t1=(t1 - p["frame_start"]) / FPS,
                   possession_frame_end=int(p["frame_end"]),
                   first_shot_frame=p["first_shot_frame"])
        if alive and not shot:
            ok = all(fx.has(f) and np.isfinite(fx.ball.loc[f]).all() for f in (t_dec, t_alt, t1))
            if not ok:
                continue
            s_dec, s_alt, s1 = fx.state(t_dec), fx.state(t_alt), fx.state(t1)
            pid = row["player_id"]
            if not all(pid in s.player_id for s in (s_dec, s_alt, s1)):
                continue
            dh = not ah
            branches = {
                "act": s1,
                "hold": CF.move_player(s1, pid, *CF.hold_position(s_dec, s1, pid, dh)),
                "hold2": CF.move_player(s1, pid, *CF.hold_position(s_alt, s1, pid, dh)),
                "drop": CF.move_player(s1, pid, *CF.drop_position(s_dec, s1, pid, dh, meta.pitch_length, FPS)),
            }
            for b, st in branches.items():
                for k, v in CF.shape_value(st, ah, grid, meta, model, values).items():
                    row[f"{b}_{k}"] = v
            i1 = np.flatnonzero(s1.player_id == pid)[0]
            hp = CF.hold_position(s_dec, s1, pid, dh)[0]
            row["displacement_m"] = float(np.hypot(*(s1.pos[i1] - hp)))
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(path, index=False)
    return out


# --------------------------------------------------------------------------- #
# Values
# --------------------------------------------------------------------------- #
def branch_frame(pr, b):
    cols = ["ball_x", "ball_y", "rd", "rd_xg_here", "cover_xg_here", "cover", "def_goal_side"]
    d = pd.DataFrame({c: pr[f"{b}_{c}"].to_numpy() for c in cols})
    d["t_in_possession"] = pr["t_in_possession_t1"].to_numpy()
    return d


def phase_cols(df, phases):
    return np.column_stack([(df["phase"].fillna("none") == ph).astype(float) for ph in phases[1:]])


def values_for_presses(pr, f, poss):
    """Two-part model and typical cover; expected values at the horizon from
    the situation at the decision (leave-one-match-out); the press's value
    split into the ball part and the shape part; the first-order "held
    position" cost as a secondary comparison."""
    model = DF.fit_two_part(f, f["y"], cover_col="cover_xg_here")
    typical = DF.fit_typical_cover(f, cover_col="cover_xg_here")
    p_full = model.predict(f)
    p_ball = model.predict(f, cover=typical.predict(f))
    tgt = {k: CF.frame_horizon_targets(f, poss, q, HORIZON_S, FPS).to_numpy()
           for k, q in (("full", p_full), ("ball", p_ball))}
    phases = sorted(f["phase"].fillna("none").unique())
    X = np.column_stack([DF.design(f, DF.FRAME_CONTROLS, "cover_xg_here"), phase_cols(f, phases)])
    dec = pr.merge(f.reset_index()[["match_id", "possession_id", "frame", "index"]].rename(
        columns={"frame": "t_dec", "index": "row"}), on=["match_id", "possession_id", "t_dec"], how="left")
    has = dec["row"].notna().to_numpy()
    pr = pr.assign(expected_full=np.nan, expected_ball=np.nan)
    for mid in pr["match_id"].unique():
        tr = (f["match_id"] != mid).to_numpy()
        sel = has & (pr["match_id"] == mid).to_numpy()
        Xs = X[dec.loc[sel, "row"].astype(int).to_numpy()]
        for k in ("full", "ball"):
            coef = CF.fit_expected_ball(X[tr], tgt[k][tr])
            pr.loc[sel, f"expected_{k}"] = M.predict_logistic(coef, Xs)
    shot, alive = pr["shot_by_t1"].to_numpy(bool), pr["alive_t1"].to_numpy(bool)
    live = alive & ~shot
    p = {}
    for b in ("act", "hold", "hold2", "drop"):
        q = np.full(len(pr), np.nan)
        q[live] = model.predict(branch_frame(pr, b)[live])
        p[b] = q
    d = branch_frame(pr, "act")
    qb = np.full(len(pr), np.nan)
    qb[live] = model.predict(d[live], cover=typical.predict(d[live]))
    V = {b: CF.horizon_value(np.nan_to_num(q), shot, alive) for b, q in p.items()}
    pr["v_full"], pr["v_ball"] = V["act"], CF.horizon_value(np.nan_to_num(qb), shot, alive)
    pr["value"] = pr["expected_full"] - pr["v_full"]          # > 0: attack worse off than typical
    pr["ball_part"] = pr["expected_ball"] - pr["v_ball"]
    pr["shape_part"] = pr["value"] - pr["ball_part"]
    for b, lab in (("hold", "cost_vs_hold"), ("hold2", "cost_vs_hold2s"), ("drop", "cost_vs_drop")):
        pr[lab] = np.where(live, V["act"] - V[b], 0.0)
    pr["worth_it"] = pr["value"] > 0
    pr["ball_won_or_out"] = ~alive & ~shot
    ok = pr["expected_full"].notna() & (~live | pd.Series(np.isfinite(p["act"]) & np.isfinite(p["hold"])))
    return pr[ok].reset_index(drop=True), model, typical, f


def zone(x_att):
    return np.select([x_att < -17.5, x_att <= 17.5], ["high (opponent's third)", "middle third"], "own third")


# --------------------------------------------------------------------------- #
# Analyses
# --------------------------------------------------------------------------- #
PARTS = ("ball_part", "shape_part", "value")


def boot_mean(sub, col, n=2000):
    per = sub.groupby("match_id")[col].agg(["sum", "size"])
    idx = rng.integers(0, len(per), (n, len(per)))
    v = per["sum"].to_numpy()[idx].sum(1) / per["size"].to_numpy()[idx].sum(1)
    return sub[col].mean(), np.percentile(v, 2.5), np.percentile(v, 97.5)


def summary_by(pr, col, order=None):
    rows = []
    for g in (order or sorted(pr[col].dropna().unique())):
        s = pr[pr[col] == g]
        if len(s) < 20:
            continue
        r = dict(group=g, n=len(s), worth_it=s["worth_it"].mean(), won_or_out=s["ball_won_or_out"].mean())
        for c in PARTS + ("cost_vs_hold",):
            r[c], r[c + "_lo"], r[c + "_hi"] = boot_mean(s, c)
        rows.append(r)
    return pd.DataFrame(rows)


def add_relative(pr):
    """Each press against presses of the same SkillCorner type in the same
    zone: where and how a player presses decides much of the raw value
    (centre-backs engage in their own box, forwards far from goal)."""
    pr["relative"] = pr["value"] - pr.groupby(["zone", "subtype"])["value"].transform("mean")
    return pr


def players(pr, metas):
    names = {}
    for meta in metas.values():
        for pid, pl in meta.players.items():
            names[pid] = (pl.name, meta.team_name(pl.team_id), pl.position_group)
    g = pr.groupby("player_id")
    t = pd.DataFrame({"n": g.size(), "relative": g["relative"].mean(), **{c: g[c].mean() for c in PARTS},
                      "own_third": g["zone"].agg(lambda z: (z == "own third").mean()),
                      "worth_it": g["worth_it"].mean()})
    t = t[t["n"] >= MIN_PRESSES].copy()
    lo, hi = [], []
    for pid in t.index:
        v = pr.loc[pr["player_id"] == pid, "relative"].to_numpy()
        bs = rng.choice(v, (2000, len(v))).mean(1)
        lo.append(np.percentile(bs, 2.5)); hi.append(np.percentile(bs, 97.5))
    t["lo"], t["hi"] = lo, hi
    for k, i in (("name", 0), ("team", 1), ("position", 2)):
        t[k] = [names.get(p, (str(p), "", ""))[i] for p in t.index]
    # split-half reliability: alternate presses in time order; for the
    # relative value also within position groups, so position alone cannot
    # make it look stable
    s = pr.sort_values(["player_id", "match_id", "t_dec"]).copy()
    s["half"] = s.groupby("player_id").cumcount() % 2
    h = s[s["player_id"].isin(t.index)].groupby(["player_id", "half"])
    rel = {}
    for c in ("relative",) + PARTS:
        w = h[c].mean().unstack()
        r = float(w[0].corr(w[1]))
        rel[c] = (r, 2 * r / (1 + r) if r > -1 else np.nan)
    w = h["relative"].mean().unstack()
    w = w - w.groupby(t["position"]).transform("mean")
    r = float(w[0].corr(w[1]))
    rel["relative within position"] = (r, 2 * r / (1 + r) if r > -1 else np.nan)
    pos = pr[pr["player_id"].isin(names)].assign(position=lambda d: d["player_id"].map(lambda i: names[i][2]))
    by_pos = pos.groupby("position").agg(presses=("value", "size"), total=("value", "mean"),
                                         relative=("relative", "mean"),
                                         own_third=("zone", lambda z: (z == "own third").mean())).reset_index()
    return t.sort_values("relative", ascending=False), rel, by_pos


def held_position_check(pr):
    """The first-order counterfactual: same moment, presser moved back into
    the shape, everyone else where they really were."""
    live = pr[pr["alive_t1"] & ~pr["shot_by_t1"]]
    return dict(n=len(live), mean=live["cost_vs_hold"].mean(),
                share_pressing_better=(live["cost_vs_hold"] < 0).mean(),
                mean_2s=live["cost_vs_hold2s"].mean(), mean_drop=live["cost_vs_drop"].mean(),
                corr_2s=live[["cost_vs_hold", "cost_vs_hold2s"]].corr().iloc[0, 1])


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def value_figure(by_out, by_zone):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for ax, t, title in ((axes[0], by_out, "by SkillCorner outcome"), (axes[1], by_zone, "by where the ball was")):
        yy = np.arange(len(t))
        for c, off, col, lab in (("ball_part", 0.27, "#2c7fb8", "on the ball"),
                                 ("shape_part", 0.0, "#c0392b", "in the shape"),
                                 ("value", -0.27, "#333333", "total")):
            ax.barh(yy + off, 100 * t[c], 0.25, color=col, label=lab)
            ax.errorbar(100 * t[c], yy + off, xerr=[100 * (t[c] - t[c + "_lo"]), 100 * (t[c + "_hi"] - t[c])],
                        fmt="none", color="k", lw=0.7)
        ax.set_yticks(yy, [f"{g} (n={n:,})" for g, n in zip(t["group"], t["n"])])
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("shots prevented per 100 presses, vs typical play")
    axes[0].legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle("What a press did, 3 s after the decision, compared with typical play from the same situation",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGS / "w7_press_value.png", dpi=150)
    plt.close(fig)


def case_figure(pr, vt, model):
    """Two presses from one match, at the decision and 3 s later."""
    live = pr[pr["alive_t1"] & ~pr["shot_by_t1"]]
    mid = live.groupby("match_id")["value"].agg(lambda s: s.max() - s.min()).idxmax()
    m = live[live["match_id"] == mid]
    good, bad = m.sort_values("value").iloc[-1], m.sort_values("value").iloc[0]
    meta, players, ball, _, _ = s06.load(mid)
    fx = T._Frames(players, ball, meta, PC.vmax_lookup(vt, mid), model.params)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.6))
    for col, (r, title) in enumerate(((good, "a press that paid off"), (bad, "a press that did not"))):
        pid = int(r["player_id"])
        for row, (f, when) in enumerate(((int(r["t_dec"]), "decision"), (int(r["t1"]), "3 s later"))):
            ax = axes[row, col]
            st = fx.state(f)
            VZ.draw_pitch(ax, meta.pitch_length, meta.pitch_width)
            VZ.plot_players(ax, st, arrows=True)
            i = np.flatnonzero(st.player_id == pid)[0]
            ax.scatter(*st.pos[i], s=170, facecolors="none", edgecolors="#f1c40f", lw=2.2, zorder=7)
            if row == 0:
                ax.set_title(f"{title}: {r['subtype']} (SkillCorner: {r['outcome']})\n{when}", fontsize=9)
            else:
                ax.set_title(f"{when}: on the ball {100 * r['ball_part']:+.1f}, in the shape "
                             f"{100 * r['shape_part']:+.1f}, total {100 * r['value']:+.1f} shots prevented per 100",
                             fontsize=9)
    fig.suptitle(f"{meta.home_name} (red) v {meta.away_name} (blue), red attacks left to right; yellow ring: "
                 "the presser; hollow: position extrapolated by SkillCorner", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "w7_press_cases.png", dpi=150)
    plt.close(fig)
    return good, bad


# --------------------------------------------------------------------------- #
def main():
    t0 = time.perf_counter()
    model_t = T.ThreatModel()
    values = {"xg_here": s06.comparisons().values["xg_here"]}
    vt = PC.vmax_table(pd.read_csv(DERIVED / "v95_by_player_half.csv"))
    parts = []
    for mid in D.list_open_matches(DATA_DIR):
        parts.append(match_presses(mid, vt, values, model_t))
        print(f"  {mid}: {len(parts[-1])} presses, {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    pr = pd.concat(parts, ignore_index=True)
    frames, poss = s10.load_skillcorner()
    f = s10.frame_table(frames, poss, FPS)
    pr, model, typical, f = values_for_presses(pr, f, poss)
    dec_x = pr.merge(f[["match_id", "possession_id", "frame", "ball_x"]].rename(columns={"frame": "t_dec"}),
                     on=["match_id", "possession_id", "t_dec"], how="left")["ball_x"].to_numpy()
    pr["zone"] = zone(dec_x)
    metas = {m: D.load_match_metadata(DATA_DIR / "matches" / str(m), m) for m in pr["match_id"].unique()}
    pr["team"] = [metas[m].team_name(t) for m, t in zip(pr["match_id"], pr["team_id"])]
    pr.to_csv(DERIVED / "press_values.csv", index=False)
    print(f"{len(pr):,} presses graded", flush=True)

    by_out = summary_by(pr, "outcome", OUTCOMES)
    by_sub = summary_by(pr, "subtype")
    by_zone = summary_by(pr, "zone", ["high (opponent's third)", "middle third", "own third"])
    pr = add_relative(pr)
    pr.to_csv(DERIVED / "press_values.csv", index=False)
    by_team = summary_by(pr, "team").sort_values("value", ascending=False)
    by_team["matches"] = by_team["group"].map(pr.groupby("team")["match_id"].nunique())
    by_team["relative"] = by_team["group"].map(pr.groupby("team")["relative"].mean())
    pl, rel, by_pos = players(pr, metas)
    held = held_position_check(pr)
    value_figure(by_out, by_zone)
    case_figure(pr, vt, model_t)
    write_report(pr, by_out, by_sub, by_zone, by_team, pl, rel, by_pos, held)
    print(f"done in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)


def _table(w, t, label):
    w(f"| {label} | presses | on the ball | in the shape | total | attack worse off than typical | "
      "ball won or out by 3 s |")
    w("|---|---|---|---|---|---|---|")
    for _, r in t.iterrows():
        cells = " | ".join(f"{100 * r[c]:+.2f} [{100 * r[c + '_lo']:+.2f}, {100 * r[c + '_hi']:+.2f}]" for c in PARTS)
        w(f"| {r['group']} | {r['n']:,} | {cells} | {r['worth_it']:.0%} | {r['won_or_out']:.0%} |")


def write_report(pr, by_out, by_sub, by_zone, by_team, pl, rel, by_pos, held):
    L = []
    w = L.append
    w("# What did the press do?\n")
    w("Every on-ball engagement SkillCorner records during an opponent possession (a defender pressing the "
      "ball carrier) is measured 3 s after the decision to press, against what typically happens from the same "
      "situation. Unit: the two-part model's chance of a shot in the next 5 s, with a won ball or a ball out of "
      "play counting as no danger and a shot as certain danger (`dsv/counterfactual.py`, "
      "`scripts/12_pressing.py`).\n")
    w("- **On the ball**: how much less dangerous the ball's position is than is typical from that situation, "
      "scored with a typical defense around it. Forcing the carrier back, winning the ball or kicking it out "
      "show up here.")
    w("- **In the shape**: how much better the defense covers the danger at that moment than a typical defense "
      "would with the ball where it now is, relative to what is typical after such situations. Space opened "
      "behind a press shows up here.")
    w("- **Total**: the two together; positive means the attack ended up less likely to shoot than is typical.\n")
    w("\"Typical\" comes from all moments in the other 19 matches, pressed or not, with the same ball "
      "situation, cover, time in possession, defenders goal-side and phase of play. Units: shots prevented per "
      "100 presses. The decision point is 1 s before SkillCorner's engagement starts (the presser is then a "
      "median 9 m from the ball). Intervals: 95% match bootstrap.\n")
    a = pr
    w(f"**{len(a):,} presses.** Mean on the ball {100 * a['ball_part'].mean():+.2f}, in the shape "
      f"{100 * a['shape_part'].mean():+.2f}, total {100 * a['value'].mean():+.2f} per 100; "
      f"{a['worth_it'].mean():.0%} left the attack worse off than typical; {a['ball_won_or_out'].mean():.0%} "
      f"ended with the ball won or out of play within 3 s, {a['shot_by_t1'].mean():.1%} with a shot.\n")
    w("## Against SkillCorner's outcome flags\n")
    w("SkillCorner flags the outcome of each engagement from its own tracking rules, independently of this "
      "model.\n")
    _table(w, by_out, "SkillCorner outcome")
    w("\n![press value](../figs/w7_press_value.png)\n")
    w("## By type and by zone\n")
    _table(w, by_sub, "SkillCorner subtype")
    w("")
    _table(w, by_zone, "ball at the decision")
    w("\nThe own third dominates: a press there is a last line of defense, and its outcomes swing between a "
      "cleared ball and a shot. Far from goal the immediate danger is small either way; the main point of a "
      "high press, winning the ball for your own attack, is not counted by a measure of danger conceded.\n")
    w("## By team\n")
    w("| pressing team | matches | presses | on the ball | in the shape | total | total vs same type and zone |")
    w("|---|---|---|---|---|---|---|")
    for _, r in by_team.iterrows():
        ci = lambda c: (f"{100 * r[c]:+.2f} [{100 * r[c + '_lo']:+.2f}, {100 * r[c + '_hi']:+.2f}]"  # noqa: E731
                        if r["matches"] >= 3 else f"{100 * r[c]:+.2f}")
        w(f"| {r['group']} | {r['matches']} | {r['n']:,} | {ci('ball_part')} | {ci('shape_part')} | {ci('value')} | "
          f"{100 * r['relative']:+.2f} |")
    w("\nIntervals only for teams in at least 3 of the 20 matches.\n")
    w("## Players: role, not yet individual skill\n")
    w("Raw values mostly sort players by where they press: per position group (all presses),\n")
    w("| position | presses | share in own third | total | vs same type and zone |")
    w("|---|---|---|---|---|")
    for _, r in by_pos.sort_values("total").iterrows():
        w(f"| {r['position']} | {r['presses']:,} | {r['own_third']:.0%} | {100 * r['total']:+.2f} | "
          f"{100 * r['relative']:+.2f} |")
    w("\nCentral defenders stand out even against presses of the same type in the same zone. When they engage "
      "they are usually the last defender, so a lost duel is a chance at goal; comparing with typical play from "
      "the same ball situation does not see that role.")
    r_v, sb_v = rel["relative"]
    r_p, sb_p = rel["relative within position"]
    w(f"\nSo players are compared on each press **against presses of the same SkillCorner type in the same zone** "
      f"(\"relative\"). Players with at least {MIN_PRESSES} presses ({len(pl)}); 95% bootstrap intervals over their "
      "presses. **Is it a stable trait?** Splitting each player's presses into alternate halves, the correlation "
      f"of the two halves' means across players is {r_v:.2f} for the relative value (Spearman-Brown reliability "
      f"of a player's full mean {sb_v:.2f}); {r_p:.2f} ({sb_p:.2f}) after also removing position-group means. "
      f"For the raw total it is {rel['value'][0]:.2f}, much of it position.\n")
    if r_p < 0.3:
        w("**Within a position group the two halves barely agree**, so with 20 matches a player's pressing value "
          "cannot yet be told apart from noise once their role is accounted for. The table is shown for "
          "transparency, not as a ranking; a season of data per player would be needed.\n")
    w("| player | team | position | presses | in own third | relative [95%] | total | on the ball | in the shape |")
    w("|---|---|---|---|---|---|---|---|---|")
    for _, r in pd.concat([pl.head(8), pl.tail(8)]).iterrows():
        w(f"| {r['name']} | {r['team']} | {r['position']} | {r['n']} | {r['own_third']:.0%} | "
          f"{100 * r['relative']:+.2f} [{100 * r['lo']:+.2f}, {100 * r['hi']:+.2f}] | {100 * r['value']:+.2f} | "
          f"{100 * r['ball_part']:+.2f} | {100 * r['shape_part']:+.2f} |")
    w("\n## Two presses\n")
    w("![cases](../figs/w7_press_cases.png)\n")
    w("## Why not \"what if they had held their position?\"\n")
    w("The brief's first design moved only the presser: same moment, presser carried back into the shape "
      "(their position at the decision, moved along with their teammates), everyone else and the ball where "
      f"they really were. Over {held['n']:,} presses still live 3 s later, that comparison says pressing gave "
      f"better cover than holding {held['share_pressing_better']:.0%} of the time (mean "
      f"{100 * held['mean']:+.2f} shots per 100 against pressing; holding from 2 s before instead: "
      f"{100 * held['mean_2s']:+.2f}, correlation {held['corr_2s']:.2f}; dropping toward goal instead: "
      f"{100 * held['mean_drop']:+.2f}). The reason is built in: the teammates who shifted to cover for the "
      "presser stay where they are, so a presser who \"holds\" only duplicates their cover and nobody is near "
      "the ball. A fair counterfactual would have to move the whole defense, which needs a model of how teams "
      "reorganize. Measuring against typical play from the same situation sidesteps that, at the price "
      "below.\n")
    w("## Limits\n")
    w("- The grade compares what happened with what is typical, so a single press's value includes the luck "
      "of how it went; averages over many presses (player, team, zone) are the measurement, single presses are "
      "illustrations.")
    w("- Defenders choose when to press, so \"typical play from the same situation\" is not a randomized "
      "comparison; the situation features (ball, cover, time, defenders goal-side, phase) remove only what "
      "they describe.")
    w("- SkillCorner's engagements include light pressure as well as full presses (\"no flagged outcome\" is "
      "most of them).")
    (REPORTS / "w7_pressing.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
