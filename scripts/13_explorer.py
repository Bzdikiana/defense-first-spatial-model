"""
The possession explorer: an interactive replay of a few possessions with the
defense's open danger drawn as a continuous field or summed into the 18
classic zones, the cover against a typical defense, the chance of a shot in
the next 5 s (two-part model) and each defender's share of the cover.

Picks five possessions (a goal, a beaten press that leads to a shot, a
defense that holds under high danger, a press that forces the ball back, a
counter-attack), computes everything 5 times a second, and writes
explorer/build/possession_explorer.html (the page in explorer/template.html
with the data inlined) plus data/derived/explorer.json.

Needs the outputs of scripts 06, 07 and 12. About 5 minutes.
"""
from __future__ import annotations

import base64
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dsv import data as D, defense as DF, pitchcontrol as PC, threat as T, value as V  # noqa: E402

DATA_DIR = Path(os.environ.get("DSV_DATA_DIR", ROOT / "data"))
DERIVED = DATA_DIR / "derived"
OUT_DIR = ROOT / "explorer" / "build"
FPS, STEP = 10.0, 2
WINDOW_S = 20.0           # at most this much of a possession, ending at the shot or the end
FOCAL = 1925299


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s06 = _load("s06", ROOT / "scripts" / "06_threat.py")
s07 = _load("s07", ROOT / "scripts" / "07_validation.py")


# --------------------------------------------------------------------------- #
# Which possessions
# --------------------------------------------------------------------------- #
def possession_peaks(f):
    g = f.sort_values("frame").groupby(["match_id", "possession_id"])
    i = g["rd_xg_here"].idxmax()
    return f.loc[i.values, ["match_id", "possession_id", "rd_xg_here", "cover_xg_here"]].rename(
        columns={"rd_xg_here": "peak_rd_xg", "cover_xg_here": "cover_at_peak"})


def pick(poss, pr, f):
    poss = poss.copy()
    poss["dur"] = (poss["frame_end"] - poss["frame_start"]) / FPS
    picks, used = [], set()

    def add(row, key, title, blurb, **kw):
        k = (int(row["match_id"]), int(row["possession_id"]))
        if k in used:
            return False
        used.add(k)
        picks.append(dict(match_id=k[0], possession_id=k[1], key=key, title=title, blurb=blurb, **kw))
        return True

    g = poss[(poss["match_id"] == FOCAL) & poss["goal"].astype(bool) & (poss["dur"] >= 8)]
    g = g if len(g) else poss[poss["goal"].astype(bool) & (poss["dur"] >= 8)]
    add(g.sort_values("xg", ascending=False).iloc[0], "goal", "A goal",
        "The attack works the ball into the box and scores. Watch the open danger build in the last seconds.")

    b = pr[(pr["outcome"] == "beaten") & pr["first_shot_frame"].notna()].sort_values("value")
    for _, r in b.iterrows():
        if add(r, "beaten", "A press beaten",
               "A defender steps out to press and is beaten; the space behind opens and the attack shoots.",
               focus_frame=int(r["frame_start"]), presser=int(r["player_id"])):
            break

    pk = possession_peaks(f).merge(poss, on=["match_id", "possession_id"])
    hold = pk[~pk["shot"].astype(bool) & (pk["peak_rd_xg"] >= pk["peak_rd_xg"].quantile(0.9)) & (pk["dur"] >= 8)]
    for _, r in hold.sort_values("cover_at_peak", ascending=False).iterrows():
        if add(r, "held", "The defense holds",
               "The ball reaches a dangerous area, but the defense keeps the space covered and no shot comes."):
            break

    fb = pr[pr["outcome"].isin(["forced backward", "reduced danger"])].merge(
        poss[["match_id", "possession_id", "dur"]], on=["match_id", "possession_id"])
    for _, r in fb[fb["dur"] >= 6].sort_values("value", ascending=False).iterrows():
        if add(r, "press_works", "A press that works",
               "A defender presses the ball carrier and forces the play back; the chance of a shot drops.",
               focus_frame=int(r["frame_start"]), presser=int(r["player_id"])):
            break

    c = poss[poss["first_phase"].isin(["quick_break", "transition"]) & poss["shot"].astype(bool) & (poss["dur"] >= 6)]
    for _, r in c.sort_values("xg", ascending=False).iterrows():
        if add(r, "counter", "A counter-attack",
               "The ball is won and played forward fast, before the defense can settle."):
            break
    return picks


# --------------------------------------------------------------------------- #
# Per-frame computation
# --------------------------------------------------------------------------- #
def b64(a):
    return base64.b64encode(np.asarray(a, np.uint8).tobytes()).decode("ascii")


def to_display(v, grid, s):
    """Per-cell vector (pitch frame) -> (17, 26) image in the attacking frame,
    each display cell the sum of 2 x 2 model cells."""
    img = v.reshape(grid.shape)
    if s < 0:
        img = img[::-1, ::-1]
    ny, nx = img.shape
    img = img[: ny - ny % 2, : nx - nx % 2]
    return img.reshape(ny // 2, 2, nx // 2, 2).sum(axis=(1, 3))


def zones18(v, cells_att, L, W):
    ix = np.clip(((cells_att[:, 0] + L / 2) / (L / 6)).astype(int), 0, 5)
    iy = np.clip(((cells_att[:, 1] + W / 2) / (W / 3)).astype(int), 0, 2)
    z = np.zeros(18)
    np.add.at(z, iy * 6 + ix, v)
    return z


def possession_payload(pk, vt, xg_fn, model, typical, model_t):
    mid, pid = pk["match_id"], pk["possession_id"]
    meta, players, ball, events, phases = s06.load(mid)
    fx = T._Frames(players, ball, meta, PC.vmax_lookup(vt, mid), model_t.params)
    grid = PC.make_grid(meta.pitch_length, meta.pitch_width, model_t.cell)
    L, W = meta.pitch_length, meta.pitch_width
    poss = T.team_possessions(events, meta)
    p = poss[poss["possession_id"] == pid].iloc[0]
    ah = bool(p["attacking_home"])
    s = 1.0 if ah else -1.0
    end = int(p["first_shot_frame"] + FPS) if pd.notna(p["first_shot_frame"]) else int(p["frame_end"])
    end = min(end, int(p["frame_end"]) + int(FPS))
    start = max(int(p["frame_start"]), end - int(WINDOW_S * FPS))
    if pk.get("focus_frame") is not None:
        start = max(int(p["frame_start"]), min(start, int(pk["focus_frame"]) - int(6 * FPS)))
        end = min(end, start + int(WINDOW_S * FPS))
    cells = grid.cells
    cells_att = s * cells
    xt_v = V.zone_value_at(model_t.xt, cells_att[:, 0] / L + 0.5, cells_att[:, 1] / W + 0.5)
    xg_v = np.asarray(xg_fn(cells_att, L, W), float)
    num = {i: pl.number for i, pl in meta.players.items()}
    name = {i: pl.name for i, pl in meta.players.items()}

    rows, fields = [], {"open_xg": [], "open_xt": [], "pc": []}
    for f in range(start + start % 2, end + 1, STEP):
        if not fx.has(f) or not np.isfinite(fx.ball.loc[f]).all():
            continue
        st = fx.state(f)
        a = T._transition_probs(model_t.transition, s * st.ball, cells_att, L, W)
        pc = T.player_control(st, cells, model_t.params)
        att = st.is_home == ah
        pc_att = pc[:, att].sum(1)
        w_xg, w_xt = a * xg_v, a * xt_v
        rd_xg, rd_xt = float(w_xg.sum()), float(w_xt.sum())
        o_xg, o_xt = w_xg * pc_att, w_xt * pc_att
        tc_xg, tc_xt = float(o_xg.sum()), float(o_xt.sum())
        dmask = ~att
        shares = (w_xg[:, None] * pc[:, dmask]).sum(0) / rd_xg
        bx, by = s * st.ball
        row = dict(frame=f, t=(f - start) / FPS, ball_x=bx, ball_y=by, rd=rd_xt, rd_xg_here=rd_xg,
                   cover=1 - tc_xt / rd_xt, cover_xg_here=1 - tc_xg / rd_xg, tc=tc_xt, tc_xg=tc_xg,
                   t_in_possession=(f - p["frame_start"]) / FPS,
                   def_goal_side=int((dmask & (s * st.pos[:, 0] > bx)).sum()),
                   zones_xg=zones18(o_xg, cells_att, L, W), zones_xt=zones18(o_xt, cells_att, L, W),
                   att=[[round(float(s * x), 1), round(float(s * y), 1), num.get(int(i)), int(d)]
                        for (x, y), i, d in zip(st.pos[att], st.player_id[att], st.detected[att])],
                   dfn=[[round(float(s * x), 1), round(float(s * y), 1), num.get(int(i)), int(d), int(i)]
                        for (x, y), i, d in zip(st.pos[dmask], st.player_id[dmask], st.detected[dmask])],
                   shares={int(i): float(v) for i, v in zip(st.player_id[dmask], shares)})
        rows.append(row)
        fields["open_xg"].append(to_display(o_xg, grid, s))
        fields["open_xt"].append(to_display(o_xt, grid, s))
        fields["pc"].append(to_display(pc_att, grid, s) / 4.0)
    df = pd.DataFrame(rows)
    df["p_shot"] = model.predict(df)
    df["typical_cover"] = typical.predict(df)

    scale = {k: float(np.max(v)) for k, v in fields.items() if k != "pc"}
    q = {k: [b64(np.round(255 * np.clip(img / scale[k], 0, 1))) for img in v] for k, v in fields.items() if k != "pc"}
    q["pc"] = [b64(np.round(255 * np.clip(img, 0, 1))) for img in fields["pc"]]

    ev = []
    for _, e in events.iterrows():
        fs = int(e["frame_start"])
        if not (start <= fs <= end):
            continue
        t = round((fs - start) / FPS, 2)
        who = f"#{num.get(int(e['player_id']))} {name.get(int(e['player_id']), '')}".strip() \
            if pd.notna(e.get("player_id")) else ""
        if e["event_type"] == "on_ball_engagement" and e["team_id"] != p["team_id"]:
            flags = [k for k, c in (("beaten", "beaten_by_possession"), ("beaten", "beaten_by_movement"),
                                    ("stopped the danger", "stop_possession_danger"),
                                    ("reduced the danger", "reduce_possession_danger"),
                                    ("forced it backward", "force_backward")) if bool(e.get(c))]
            ev.append(dict(t=t, t_end=round((int(e["frame_end"]) - start) / FPS, 2), kind="press",
                           label=f"{who} presses ({e['event_subtype'].replace('_', ' ')})"
                                 + (f": {flags[0]}" if flags else ""), outcome=flags[0] if flags else ""))
        elif e["event_type"] == "player_possession" and e["team_id"] == p["team_id"]:
            kind = "shot" if e.get("end_type") == "shot" else "ball"
            label = f"{who} shoots" if kind == "shot" else f"{who} on the ball"
            if kind == "shot" and e.get("game_interruption_after") == "goal_for":
                label += ": goal"
            ev.append(dict(t=t if kind == "ball" else round((int(e["frame_end"]) - start) / FPS, 2),
                           t_end=round((int(e["frame_end"]) - start) / FPS, 2), kind=kind, label=label))

    shots = [e for e in ev if e["kind"] == "shot"]
    if bool(p["goal"]) and shots and not shots[-1]["label"].endswith("goal"):
        shots[-1]["label"] += ": goal"
    phase_names = ">".join(dict.fromkeys(str(x).replace("_", " ") for x in str(p.get("phases", "")).split(">") if x))
    return dict(
        key=pk["key"], title=pk["title"], blurb=pk["blurb"], match_id=int(mid), possession_id=int(pid),
        home=meta.home_name, away=meta.away_name,
        attack=meta.home_name if ah else meta.away_name, defend=meta.away_name if ah else meta.home_name,
        period=int(p["period"]), phases=phase_names, shot=bool(p["shot"]), goal=bool(p["goal"]),
        L=L, W=W, nx=fields["open_xg"][0].shape[1], ny=fields["open_xg"][0].shape[0], fps=FPS / STEP,
        presser=pk.get("presser"),
        names={int(i): f"#{num.get(i)} {name.get(i)}" for i in meta.players},
        scale=scale, events=ev,
        frames=dict(
            t=[round(x, 2) for x in df["t"]], ball=[[round(x, 1), round(y, 1)] for x, y in zip(df["ball_x"], df["ball_y"])],
            att=list(df["att"]), dfn=list(df["dfn"]),
            rd_xg=[round(x, 5) for x in df["rd_xg_here"]], rd_xt=[round(x, 5) for x in df["rd"]],
            cover_xg=[round(x, 4) for x in df["cover_xg_here"]], cover_xt=[round(x, 4) for x in df["cover"]],
            tc_xg=[round(x, 6) for x in df["tc_xg"]], tc_xt=[round(x, 6) for x in df["tc"]],
            p_shot=[round(x, 4) for x in df["p_shot"]], typical_cover=[round(x, 4) for x in df["typical_cover"]],
            zones_xg=[[round(v, 6) for v in z] for z in df["zones_xg"]],
            zones_xt=[[round(v, 6) for v in z] for z in df["zones_xt"]],
            shares=[{str(k): round(v, 4) for k, v in d.items()} for d in df["shares"]],
        ),
        fields=q,
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_t = T.ThreatModel()
    frames, poss_all = s07.load_frames(), pd.read_csv(DERIVED / "threat_possessions.csv")
    f = DF.shot_frames(frames, poss_all, FPS)
    model = DF.fit_two_part(f, f["y"], cover_col="cover_xg_here")
    typical = DF.fit_typical_cover(f, cover_col="cover_xg_here")
    pr = pd.read_csv(DERIVED / "press_values.csv")
    picks = pick(poss_all, pr, f)
    print([(p["key"], p["match_id"], p["possession_id"]) for p in picks], flush=True)
    vt = PC.vmax_table(pd.read_csv(DERIVED / "v95_by_player_half.csv"))
    xg_fn = s06.comparisons().values["xg_here"]
    data = dict(model=dict(cover_coef=model.cover_effect, horizon_s=5),
                possessions=[possession_payload(pk, vt, xg_fn, model, typical, model_t) for pk in picks])
    js = json.dumps(data, separators=(",", ":"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
    (DERIVED / "explorer.json").write_text(js, encoding="utf-8")
    tpl = (ROOT / "explorer" / "template.html").read_text(encoding="utf-8")
    page = tpl.replace("/*__DATA__*/null", js.replace("</", "<\\/"))
    (OUT_DIR / "possession_explorer.html").write_text(page, encoding="utf-8")
    print(f"{len(js) / 1e6:.1f} MB of data; wrote {OUT_DIR / 'possession_explorer.html'}", flush=True)


if __name__ == "__main__":
    main()
