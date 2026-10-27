"""
dsv.threat
==========
Threat Conceded (TC): the danger a defending team allows at each moment, and
its aggregation by possession, phase, team and match.

At frame t, with the ball at b_t and the attack's pitch control PC_att:

    TC(t)    = sum_x A(x | b_t) * PC_att(x, t) * xT(x)    danger conceded
    RD(t)    = sum_x A(x | b_t) * xT(x)                   reachable danger
    cover(t) = 1 - TC(t) / RD(t)                          share the defense holds

A is the next-ball-location model (sums to 1 over the grid), xT the expected
threat surface. RD depends only on where the ball is: it is TC if the attack
won every spot. cover separates the defense's shape from the ball's location.
Units of TC and RD are xT, i.e. goals.

Frames are direction-normalized tracking (home attacks +x); A and xT are
evaluated in the attacking team's frame.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np
import pandas as pd

from . import pitchcontrol as PC
from . import value as V
from .data import MatchMeta, player_group


@dataclass(frozen=True)
class ThreatModel:
    """Settings for match-long Threat Conceded. The defaults are the fitted
    open-data components on a 2 m grid with a 0.08 s integration step,
    which stays within 1% of the 1 m / 0.04 s computation
    (reports/w4_threat.md) at about a tenth of the cost."""
    params: PC.PCParams = replace(PC.SKILLCORNER_OPEN_DATA, dt=0.08)
    xt: np.ndarray = field(default_factory=lambda: V.OPEN_DATA_XT)
    transition: V.TransitionModel = field(default_factory=lambda: V.OPEN_DATA_TRANSITION)
    cell: float = 2.0


# --------------------------------------------------------------------------- #
# One frame
# --------------------------------------------------------------------------- #
def danger_weights(ball_att: np.ndarray, cells_att: np.ndarray, length: float, width: float,
                   model: ThreatModel = ThreatModel()) -> np.ndarray:
    """A(x | ball) * xT(x) for each cell, both in the attacking frame. A is
    normalized to sum to 1 over the cells."""
    a = _transition_probs(model.transition, ball_att, cells_att, length, width)
    return a * V.zone_value_at(model.xt, cells_att[:, 0] / length + 0.5, cells_att[:, 1] / width + 0.5)


@dataclass
class Comparisons:
    """Alternative ingredients evaluated on the same pitch control, for
    baselines and sensitivity checks (each adds one weighted sum per frame).

    values: name -> function(cells_att (k, 2), length, width) -> (k,) danger
        surface used in place of xT (e.g. another xT grid).
    transitions: name -> TransitionModel used in place of the fitted one.
    """
    values: dict = field(default_factory=dict)
    transitions: dict = field(default_factory=dict)
    _cache: dict = field(default_factory=dict, repr=False)

    def value(self, name, cells_att, length, width):
        key = (name, cells_att.shape[0], float(cells_att[0, 0]), length, width)
        if key not in self._cache:
            self._cache[key] = np.asarray(self.values[name](cells_att, length, width), dtype=float)
        return self._cache[key]


def _transition_probs(trans: V.TransitionModel, ball_att, cells_att, length, width):
    a = np.exp(trans.logpdf(ball_att[0], ball_att[1], cells_att[:, 0], cells_att[:, 1], length / 2, width / 2))
    return a / a.sum()


def frame_threat(state: PC.FrameState, attacking_home: bool, grid: PC.Grid, meta: MatchMeta,
                 model: ThreatModel = ThreatModel(), comparisons: Optional[Comparisons] = None) -> dict:
    """TC, reachable danger, cover, and the share of the defense's weighted
    control held by players whose positions are extrapolated.

    With `comparisons`, also (all from the same pitch control):
    space_m2: area the attack controls, unweighted;
    tc_no_access: control x xT without the next-ball weight, as a share of
        the pitch's total xT (a Fernandez-style space value);
    tc_voronoi: TC with "nearest player wins" in place of pitch control;
    tc_<name>: TC with each alternative danger surface or transition model;
    def_goal_side: defenders between the ball and their own goal line.
    """
    prm = model.params
    s = 1.0 if attacking_home else -1.0
    cells = grid.cells
    L, W = meta.pitch_length, meta.pitch_width
    cells_att = s * cells
    a = _transition_probs(model.transition, s * state.ball, cells_att, L, W)
    xt = V.zone_value_at(model.xt, cells_att[:, 0] / L + 0.5, cells_att[:, 1] / W + 0.5)
    w = a * xt
    n = len(state.player_id)
    tau = PC._arrival_times(state.pos, state.vel, state.vmax, cells[:, None, :], prm.t_react)
    t_ball = np.linalg.norm(cells - state.ball, axis=-1) / prm.ball_speed
    lam = np.ascontiguousarray(np.broadcast_to(PC._rates(state.is_gk, prm), (len(cells), n)))
    alpha = np.ascontiguousarray(np.broadcast_to(
        np.broadcast_to(PC._alpha(state.detected, prm), (n,)), (len(cells), n)))
    pc = PC._integrate_rows(tau, t_ball, lam, alpha, prm)
    att = state.is_home == attacking_home
    pc_att = pc[:, att].sum(axis=1)
    pc_def = pc[:, ~att].sum(axis=1)
    pc_def_ext = pc[:, (~att) & (~state.detected)].sum(axis=1)
    tc, rd = float((w * pc_att).sum()), float(w.sum())
    wd = float((w * pc_def).sum())
    out = dict(tc=tc, rd=rd, cover=1.0 - tc / rd if rd > 0 else np.nan,
               def_extrap_share=float((w * pc_def_ext).sum()) / wd if wd > 0 else np.nan)
    if comparisons is not None:
        d = np.hypot(cells[:, None, 0] - state.pos[None, :, 0], cells[:, None, 1] - state.pos[None, :, 1])
        vor_att = att[np.argmin(d, axis=1)].astype(float)
        x_att = s * state.pos[:, 0]
        out.update(
            space_m2=float(pc_att.sum() * grid.cell_area),
            tc_no_access=float((xt * pc_att).sum() / xt.sum()),
            tc_voronoi=float((w * vor_att).sum()),
            def_goal_side=int(((~att) & (x_att > s * state.ball[0])).sum()),
        )
        for name in comparisons.values:
            v = comparisons.value(name, cells_att, L, W)
            out[f"tc_{name}"] = float((a * v * pc_att).sum())
        for name, trans in comparisons.transitions.items():
            out[f"tc_{name}"] = float((_transition_probs(trans, s * state.ball, cells_att, L, W) * xt * pc_att).sum())
    return out


# --------------------------------------------------------------------------- #
# Possessions
# --------------------------------------------------------------------------- #
def team_possessions(events: pd.DataFrame, meta: MatchMeta,
                     shot_xg: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """SkillCorner team possessions: runs of player possessions by one team,
    delimited by the event layer's `first_player_possession_in_team_possession`.

    One row per possession: frames from the first possession's start to the
    last one's end, number of player possessions, phases of play seen,
    whether it produced a shot or a goal (a possession followed by
    `goal_for`), and the xG of its shots if `shot_xg` (columns match_id,
    event_id, xg; e.g. data/derived/skillcorner_shots_xg.csv) is given.
    Event ids repeat across matches, so they are matched within this match."""
    pp = events[events["event_type"] == "player_possession"].sort_values("frame_start").copy()
    pp["possession_id"] = pp["first_player_possession_in_team_possession"].astype(bool).cumsum()
    pp["is_shot"] = pp["end_type"] == "shot"
    pp["is_goal"] = pp["game_interruption_after"] == "goal_for"
    if shot_xg is not None:
        m = shot_xg[shot_xg["match_id"] == meta.match_id].set_index("event_id")["xg"]
        pp["xg"] = np.where(pp["is_shot"], pp["event_id"].map(m).fillna(0.0), 0.0)
    else:
        pp["xg"] = np.nan
    g = pp.groupby("possession_id")
    out = pd.DataFrame({
        "match_id": meta.match_id,
        "team_id": g["team_id"].first(),
        "period": g["period"].first(),
        "frame_start": g["frame_start"].min(),
        "frame_end": g["frame_end"].max(),
        "n_player_possessions": g.size(),
        "phases": g["team_in_possession_phase_type"].agg(lambda s: ">".join(pd.unique(s.dropna()))),
        "first_phase": g["team_in_possession_phase_type"].first(),
        "shot": g["is_shot"].any(),
        "goal": g["is_goal"].any(),
        "n_shots": g["is_shot"].sum(),
        "xg": g["xg"].sum(min_count=1),
        "first_shot_frame": pp[pp["is_shot"]].groupby("possession_id")["frame_end"].min(),
    }).reset_index()
    out["attacking_home"] = out["team_id"] == meta.home_team_id
    out["duration_s"] = (out["frame_end"] - out["frame_start"]) / meta.fps
    return out


# --------------------------------------------------------------------------- #
# A whole match
# --------------------------------------------------------------------------- #
class _Frames:
    """Fast per-frame access to direction-normalized tracking."""

    def __init__(self, players: pd.DataFrame, ball: pd.DataFrame, meta: MatchMeta,
                 vmax: dict, params: PC.PCParams):
        p = players.sort_values(["frame", "is_home", "player_id"], ascending=[True, False, True])
        self.frame = p["frame"].to_numpy()
        self.pid = p["player_id"].to_numpy()
        self.home = p["is_home"].to_numpy(dtype=bool)
        self.det = p["is_detected"].to_numpy(dtype=bool)
        self.pos = p[["x", "y"]].to_numpy(dtype=float)
        self.vel = np.nan_to_num(p[["vx", "vy"]].to_numpy(dtype=float))
        gk = {i for i in np.unique(self.pid) if player_group(meta, int(i)) == "Goalkeeper"}
        self.gk = np.isin(self.pid, list(gk))
        b = ball.set_index("frame")
        self.period = b["period"]
        self.ball = b[["bx", "by"]]
        self.ball_det = b["ball_detected"]
        per = self.period.reindex(self.frame).to_numpy()
        self.vmax = np.array([vmax.get((int(i), int(q)), params.vmax_default)
                              for i, q in zip(self.pid, per)])
        self.uframes, self.start = np.unique(self.frame, return_index=True)
        self.end = np.append(self.start[1:], len(self.frame))

    def has(self, f: int) -> bool:
        i = np.searchsorted(self.uframes, f)
        return i < len(self.uframes) and self.uframes[i] == f

    def state(self, f: int) -> PC.FrameState:
        i = np.searchsorted(self.uframes, f)
        a, b = self.start[i], self.end[i]
        return PC.FrameState(
            frame=int(f), period=int(self.period.loc[f]), player_id=self.pid[a:b],
            is_home=self.home[a:b], is_gk=self.gk[a:b], detected=self.det[a:b],
            pos=self.pos[a:b], vel=self.vel[a:b], vmax=self.vmax[a:b],
            ball=self.ball.loc[f].to_numpy(dtype=float), possession=None)


def match_threat(players: pd.DataFrame, ball: pd.DataFrame, possessions: pd.DataFrame,
                 meta: MatchMeta, vmax: dict, model: ThreatModel = ThreatModel(),
                 step: int = 2, phases: Optional[pd.DataFrame] = None,
                 comparisons: Optional[Comparisons] = None) -> pd.DataFrame:
    """Frame-level Threat Conceded for every team possession, every `step`
    frames (step=2 is 5 per second at 10 fps). `players` needs the
    `add_velocity` columns. Frames without tracking or ball position are
    skipped. With `phases`, each frame gets the attacking team's phase of
    play."""
    fx = _Frames(players, ball, meta, vmax, model.params)
    grid = PC.make_grid(meta.pitch_length, meta.pitch_width, model.cell)
    if phases is not None:
        ph = phases.sort_values("frame_start")
        ph_start, ph_end = ph["frame_start"].to_numpy(), ph["frame_end"].to_numpy()
        ph_type = ph["team_in_possession_phase_type"].to_numpy()
    rows = []
    for _, pos in possessions.iterrows():
        f0 = int(np.ceil(pos["frame_start"] / step) * step)
        for f in range(f0, int(pos["frame_end"]) + 1, step):
            if not fx.has(f) or not np.isfinite(fx.ball.loc[f]).all():
                continue
            st = fx.state(f)
            r = frame_threat(st, bool(pos["attacking_home"]), grid, meta, model, comparisons)
            sgn = 1.0 if pos["attacking_home"] else -1.0
            row = dict(match_id=meta.match_id, possession_id=int(pos["possession_id"]), frame=f,
                       period=st.period, t_in_possession=(f - pos["frame_start"]) / meta.fps,
                       attacking_team_id=int(pos["team_id"]), attacking_home=bool(pos["attacking_home"]),
                       ball_x=sgn * st.ball[0], ball_y=sgn * st.ball[1],
                       ball_detected=bool(fx.ball_det.loc[f]), n_detected=int(st.detected.sum()), **r)
            if phases is not None:
                i = np.searchsorted(ph_start, f, side="right") - 1
                row["phase"] = ph_type[i] if i >= 0 and f <= ph_end[i] else None
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def possession_summary(frames: pd.DataFrame, possessions: pd.DataFrame, fps: float = 10.0,
                       lookback_s: float = 2.0, extra_columns=()) -> pd.DataFrame:
    """One row per possession with its Threat Conceded curve summarized.

    peak_tc / peak_rd / min_cover: over the whole possession.
    peak_tc_early / peak_rd_early: over frames at least `lookback_s` before
    the possession's first shot (or its end, if no shot), so the moment of
    the shot itself cannot drive the summary.
    exposure: sum of TC x frame spacing (goal-seconds).
    mean_tc, mean_cover, mean_def_extrap_share: averages over frames.
    For each name in `extra_columns`: peak_<name>_early and mean_<name>.
    """
    df = frames.merge(possessions[["match_id", "possession_id", "frame_end", "first_shot_frame"]],
                      on=["match_id", "possession_id"])
    cut = df["first_shot_frame"].fillna(df["frame_end"]) - lookback_s * fps
    df["early"] = df["frame"] <= cut
    spacing = df.groupby(["match_id", "possession_id"])["frame"].diff().median() / fps
    g = df.groupby(["match_id", "possession_id"])
    e = df[df["early"]].groupby(["match_id", "possession_id"])
    out = pd.DataFrame({
        "n_frames": g.size(),
        "peak_tc": g["tc"].max(),
        "mean_tc": g["tc"].mean(),
        "exposure": g["tc"].sum() * (spacing if np.isfinite(spacing) else 0.2),
        "peak_rd": g["rd"].max(),
        "min_cover": g["cover"].min(),
        "mean_cover": g["cover"].mean(),
        "mean_def_extrap_share": g["def_extrap_share"].mean(),
        "peak_tc_early": e["tc"].max(),
        "peak_rd_early": e["rd"].max(),
        "t_peak": g.apply(lambda d: d.loc[d["tc"].idxmax(), "t_in_possession"], include_groups=False),
        **{f"peak_{c}_early": e[c].max() for c in extra_columns},
        **{f"mean_{c}": g[c].mean() for c in extra_columns},
    }).reset_index()
    return possessions.merge(out, on=["match_id", "possession_id"], how="left")


def team_match_summary(possessions: pd.DataFrame, meta_by_match: dict) -> pd.DataFrame:
    """Per match and defending team: possessions faced, Threat Conceded
    exposure, mean peak, shots, goals and xG conceded. `possessions` is the
    output of `possession_summary` for one or more matches."""
    p = possessions.dropna(subset=["peak_tc"]).copy()
    p["defending_team_id"] = [
        meta_by_match[m].away_team_id if h else meta_by_match[m].home_team_id
        for m, h in zip(p["match_id"], p["attacking_home"])]
    g = p.groupby(["match_id", "defending_team_id"])
    out = pd.DataFrame({
        "possessions_faced": g.size(),
        "tc_exposure": g["exposure"].sum(),
        "mean_peak_tc": g["peak_tc"].mean(),
        "shots_conceded": g["n_shots"].sum(),
        "goals_conceded": g["goal"].sum(),
        "xg_conceded": g["xg"].sum(),
    }).reset_index()
    out["team"] = [meta_by_match[m].team_name(t) for m, t in zip(out["match_id"], out["defending_team_id"])]
    return out
