"""
dsv.counterfactual
==================
What did a press do? Measuring a defensive action by what it did to the ball
and to the shape, a few seconds later, against typical play.

A press has a purpose and a price. The purpose is the ball: hurry the
carrier, force them backward, win it. The price is space: the presser leaves
their place, and the cover can thin out. Both are measured in the same unit,
the two-part model's chance of a shot (`dsv.defense`), at a fixed horizon
after the decision:

    value V(t) = 1                              if the attack has shot by t
               = 0                              if the possession has ended without a shot
               = P(shot within 5 s | ball(t), cover(t))   otherwise

V_ball(t) is the same with a typical defense's cover for the ball's
position, so it scores the ball alone. For a press decided at t_dec and
measured at t1 = t_dec + horizon, against expectations from all moments in
the same situation, pressed or not (`fit_expected_ball`):

    total        = E[V(t1) | situation at t_dec] - V(t1)        (> 0: attack worse off than typical)
    on the ball  = E[V_ball(t1) | situation] - V_ball(t1)
    in the shape = total - on the ball

These compare what happened with what usually happens, so a single press
carries luck as well as decision quality; averages over many presses (by
player, team, zone) are the intended use.

`hold_position`, `drop_position` and `move_player` give the brief's
first-order counterfactual (same moment, only the presser moved, everyone
else on their real paths). It is kept as a comparison, not as the grade:
teammates who shifted to cover for the presser stay in place, so a presser
who "holds" only duplicates their cover (reports/w7_pressing.md).
"""
from __future__ import annotations

from dataclasses import replace
from typing import Optional

import numpy as np
import pandas as pd

from . import pitchcontrol as PC
from . import value as V
from .data import MatchMeta
from .threat import ThreatModel, _transition_probs, player_control


# --------------------------------------------------------------------------- #
# Alternative positions
# --------------------------------------------------------------------------- #
def _teammates(state: PC.FrameState, pid: int, defending_home: bool) -> np.ndarray:
    return (state.is_home == defending_home) & (state.player_id != pid) & ~state.is_gk


def hold_position(state_dec: PC.FrameState, state_t: PC.FrameState, pid: int,
                  defending_home: bool) -> tuple[np.ndarray, np.ndarray]:
    """Where defender `pid` would be at state_t's moment had they held their
    place in the shape from state_dec's moment: their position then, moved
    by the shift of their outfield teammates' centroid since. Velocity: the
    teammates' mean velocity now."""
    i0 = np.flatnonzero(state_dec.player_id == pid)[0]
    m0, m1 = _teammates(state_dec, pid, defending_home), _teammates(state_t, pid, defending_home)
    shift = state_t.pos[m1].mean(0) - state_dec.pos[m0].mean(0)
    return state_dec.pos[i0] + shift, state_t.vel[m1].mean(0)


def drop_position(state_dec: PC.FrameState, state_t: PC.FrameState, pid: int, defending_home: bool,
                  length: float, fps: float = 10.0, speed: float = 3.0,
                  max_m: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    """Like `hold_position`, but also retreating toward their own goal at
    `speed` m/s for up to `max_m` metres (delay rather than engage). In the
    direction-normalized frame the home team defends the goal at -length/2."""
    pos, vel = hold_position(state_dec, state_t, pid, defending_home)
    i0 = np.flatnonzero(state_dec.player_id == pid)[0]
    goal = np.array([-length / 2 if defending_home else length / 2, 0.0])
    u = goal - state_dec.pos[i0]
    u = u / max(np.linalg.norm(u), 1e-9)
    dt = (state_t.frame - state_dec.frame) / fps
    d = min(speed * max(dt, 0.0), max_m)
    return pos + d * u, vel + (speed * u if speed * max(dt, 0.0) < max_m else 0.0)


def move_player(state: PC.FrameState, pid: int, pos, vel) -> PC.FrameState:
    """The same frame with one player's position and velocity replaced."""
    i = np.flatnonzero(state.player_id == pid)[0]
    p, v = state.pos.copy(), state.vel.copy()
    p[i], v[i] = pos, vel
    return replace(state, pos=p, vel=v)


# --------------------------------------------------------------------------- #
# The defensive picture at one moment
# --------------------------------------------------------------------------- #
def shape_value(state: PC.FrameState, attacking_home: bool, grid: PC.Grid, meta: MatchMeta,
                model: ThreatModel = ThreatModel(), values: Optional[dict] = None) -> dict:
    """What the two-part model needs at one frame: the ball in the attacking
    frame, reachable danger and cover for xT and each extra danger map
    (rd, cover, rd_<name>, cover_<name>), and defenders goal-side."""
    s = 1.0 if attacking_home else -1.0
    cells = grid.cells
    L, W = meta.pitch_length, meta.pitch_width
    cells_att = s * cells
    a = _transition_probs(model.transition, s * state.ball, cells_att, L, W)
    maps = {"": V.zone_value_at(model.xt, cells_att[:, 0] / L + 0.5, cells_att[:, 1] / W + 0.5)}
    for name, fn in (values or {}).items():
        maps[f"_{name}"] = np.asarray(fn(cells_att, L, W), float)
    att = state.is_home == attacking_home
    pc_att = player_control(state, cells, model.params)[:, att].sum(1)
    out = dict(ball_x=s * state.ball[0], ball_y=s * state.ball[1],
               def_goal_side=int(((~att) & (s * state.pos[:, 0] > s * state.ball[0])).sum()))
    for suffix, v in maps.items():
        w = a * v
        rd = float(w.sum())
        out[f"rd{suffix}"] = rd
        out[f"cover{suffix}"] = 1 - float((w * pc_att).sum()) / rd if rd > 0 else np.nan
    return out


# --------------------------------------------------------------------------- #
# Values at the horizon
# --------------------------------------------------------------------------- #
def horizon_value(p_alive: np.ndarray, shot_by_t1: np.ndarray, alive_t1: np.ndarray) -> np.ndarray:
    """V(t1): 1 if the attack has already shot, 0 if the possession ended
    without a shot, otherwise the model's chance of a shot (`p_alive`)."""
    return np.where(shot_by_t1, 1.0, np.where(alive_t1, p_alive, 0.0))


def frame_horizon_targets(frames: pd.DataFrame, possessions: pd.DataFrame, p_ball: np.ndarray,
                          horizon_s: float, fps: float) -> pd.Series:
    """For every frame t of a frame table (one row per frame and possession,
    frames after the possession's first shot already dropped), V_ball at
    t + horizon: 1 if the possession's first shot comes in (t, t + horizon],
    0 if the possession ends before t + horizon without a shot, else the
    ball-only chance `p_ball` at the frame t + horizon (NaN if that frame is
    missing, e.g. a gap in tracking)."""
    f = frames[["match_id", "possession_id", "frame"]].copy()
    f["p_ball"] = p_ball
    f = f.merge(possessions[["match_id", "possession_id", "frame_end", "first_shot_frame"]],
                on=["match_id", "possession_id"], how="left")
    t1 = f["frame"] + int(round(horizon_s * fps))
    look = f.set_index(["match_id", "possession_id", "frame"])["p_ball"]
    later = look.reindex(pd.MultiIndex.from_arrays([f["match_id"], f["possession_id"], t1])).to_numpy()
    shot = f["first_shot_frame"].notna() & (f["first_shot_frame"] <= t1)
    alive = t1 <= f["frame_end"]
    out = np.where(shot, 1.0, np.where(alive, later, 0.0))
    return pd.Series(out, index=frames.index)


def fit_expected_ball(X: np.ndarray, target: np.ndarray, l2: float = 1e-4) -> np.ndarray:
    """Logistic regression with a fractional target (quasi-likelihood): the
    expected value at the horizon (V or V_ball) from the situation now."""
    from .metrics import fit_logistic
    ok = np.isfinite(target) & np.isfinite(X).all(1)
    return fit_logistic(X[ok], target[ok], l2=l2)
