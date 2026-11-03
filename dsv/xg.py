"""
dsv.xg
======
An expected-goals model for SkillCorner tracking, fitted on the SkillCorner
open-data shots alone (the Analytics Cup allows only the competition's data).

Features come from the tracking at the shot frame: shot geometry, headers,
and the positions of the opponents and the keeper. `skillcorner_shots`
builds the shot table; `fit_xg` / `predict_xg` fit and apply the logistic
model; `OPEN_DATA_XG` holds the fitted coefficients (scripts/04_xg.py,
reports/w3_xg.md).

Frame: metres, shooting team attacks +x, goal centre at (GOAL_X, 0), posts at
y = +-3.66.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from . import metrics as M

GOAL_X = 0.0          # features are computed relative to the goal line
POST_Y = 7.32 / 2

FEATURES = ["dist", "log_dist", "angle", "header", "n_block", "gk_off_line",
            "gk_dist_goal", "press_dist", "n_close"]


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def _in_triangle(p, a, b, c):
    """Boolean mask: points p (k, 2) inside triangle abc."""
    def side(p1, p2, p3):
        return (p1[..., 0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[..., 1] - p3[1])
    d1, d2, d3 = side(p, a, b), side(p, b, c), side(p, c, a)
    neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    return ~(neg & pos)


def shot_features(shot_xy, header: bool, opp_xy, gk_xy: Optional[np.ndarray]) -> dict:
    """Features for one shot. Coordinates in metres relative to the goal
    centre, attacking towards +x with the goal line at x = 0 (so shots have
    x < 0). opp_xy: (k, 2) outfield opponents; gk_xy: the opposing keeper or None.
    """
    s = np.asarray(shot_xy, float)
    opp = np.asarray(opp_xy, float).reshape(-1, 2)
    post_l, post_r = np.array([0.0, POST_Y]), np.array([0.0, -POST_Y])
    dist = float(np.hypot(*s))
    a1 = np.arctan2(POST_Y - s[1], -s[0])
    a2 = np.arctan2(-POST_Y - s[1], -s[0])
    angle = float(abs(a1 - a2))
    defenders = opp if gk_xy is None else np.vstack([opp, np.asarray(gk_xy, float)[None]])
    n_block = int(_in_triangle(opp, s, post_l, post_r).sum()) if len(opp) else 0
    if gk_xy is None:
        gk_off_line, gk_dist_goal = 3.0, 1.0       # typical values when unknown
    else:
        g = np.asarray(gk_xy, float)
        u = -s / max(dist, 1e-6)                    # unit vector shot -> goal centre
        rel = g - s
        gk_off_line = float(abs(rel[0] * u[1] - rel[1] * u[0]))   # distance from the shot line
        gk_dist_goal = float(np.hypot(*g))
    d_all = np.hypot(*(defenders - s).T) if len(defenders) else np.array([30.0])
    return dict(dist=dist, log_dist=float(np.log(max(dist, 0.5))), angle=angle,
                header=float(header), n_block=float(n_block), gk_off_line=min(gk_off_line, 20.0),
                gk_dist_goal=min(gk_dist_goal, 30.0), press_dist=float(min(d_all.min(), 10.0)),
                n_close=float((d_all < 3.0).sum()))


# --------------------------------------------------------------------------- #
# SkillCorner
# --------------------------------------------------------------------------- #
def skillcorner_shots(events: pd.DataFrame, players_df: pd.DataFrame, ball_df: pd.DataFrame,
                      meta) -> pd.DataFrame:
    """Shots from SkillCorner dynamic events with features from tracking at
    the shot frame (frame_end of the shooting possession). Expects
    direction-normalized tracking (home attacks +x). Adds `goal` (the
    possession is followed by goal_for) and `extrap_share` (share of the
    opponents used in the features whose positions are extrapolated).
    Positions of shots likely to be penalties (start of possession on the spot)
    are flagged `likely_penalty`."""
    from .data import player_group
    pp = events[(events["event_type"] == "player_possession") & (events["end_type"] == "shot")]
    tracked = set(ball_df["frame"])
    half = meta.pitch_length / 2
    rows = []
    for _, e in pp.iterrows():
        fr = int(e["frame_end"])
        if fr not in tracked:
            continue
        p = players_df[players_df["frame"] == fr]
        shooter_home = e["team_id"] == meta.home_team_id
        sgn = 1.0 if shooter_home else -1.0          # rotate so the shooter attacks +x
        b = ball_df.loc[ball_df["frame"] == fr, ["bx", "by"]].to_numpy(dtype=float)[0]
        shooter = p[p["player_id"] == e["player_id"]][["x", "y"]].to_numpy(dtype=float)
        s_xy = shooter[0] if len(shooter) else b
        to_goal = lambda xy: np.column_stack([sgn * xy[:, 0] - half, sgn * xy[:, 1]])  # noqa: E731
        opp = p[p["is_home"] != shooter_home]
        is_gk = opp["player_id"].map(lambda i: player_group(meta, i) == "Goalkeeper").to_numpy()
        opp_xy = to_goal(opp[["x", "y"]].to_numpy(dtype=float))
        f = shot_features(to_goal(s_xy[None])[0], bool(e["is_header"]), opp_xy[~is_gk],
                          opp_xy[is_gk][0] if is_gk.any() else None)
        near = np.hypot(*(opp_xy - to_goal(s_xy[None])[0]).T) < 20
        rows.append(dict(
            match_id=meta.match_id, event_id=e["event_id"], frame=fr, period=e["period"],
            minute=e["minute_start"], team_id=e["team_id"], player_id=e["player_id"],
            player_name=e["player_name"], goal=e.get("game_interruption_after") == "goal_for",
            likely_penalty=bool(abs(f["dist"] - 11.0) < 0.8 and f["angle"] > 0.6
                                and e.get("start_type") in ("unknown", "keep_possession")
                                and e.get("duration", 99) < 2),
            extrap_share=float((~opp["is_detected"].to_numpy()[near]).mean()) if near.any() else 0.0,
            **f))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def design(df: pd.DataFrame, features=FEATURES) -> np.ndarray:
    X = df[list(features)].to_numpy(dtype=float)
    # a little curvature where it matters: angle^2 and distance x header
    extra = np.column_stack([df["angle"] ** 2, df["dist"] * df["header"]])
    return np.column_stack([X, extra])


def fit_xg(train: pd.DataFrame, features=FEATURES, l2: float = 1e-4) -> dict:
    return dict(features=list(features), coef=M.fit_logistic(design(train, features), train["goal"], l2=l2))


def predict_xg(model: dict, df: pd.DataFrame) -> np.ndarray:
    return M.predict_logistic(model["coef"], design(df, model["features"]))


# Fitted on the 524 non-penalty SkillCorner open-data shots (60 goals), full
# features, ridge 0.01 chosen by leave-one-match-out log loss (scripts/04_xg.py,
# reports/w3_xg.md): held-out log loss 0.328 (constant rate 0.359), AUC 0.72,
# 60.0 xG for 60 goals. Coefficients are [intercept, *FEATURES, angle^2,
# dist*header] on the raw features.
OPEN_DATA_XG = dict(features=list(FEATURES), coef=np.array([
    -0.36038873983259845, -0.05722641204383733, -0.6544079187725333, 0.9527467243662217,
    -0.5009370079992108, -0.10410573556283778, 0.05732102366686082, 0.09563185224855306,
    0.1156509279638444, -0.183533726740007, -0.02196428776013739, 0.037582051718371305,
]))


def xg_for_shots(shots: pd.DataFrame, model: dict = OPEN_DATA_XG) -> np.ndarray:
    """xG for a table of shots with FEATURES columns, e.g. the output of
    `skillcorner_shots`. Penalties are not modelled: set them yourself
    (about 0.76)."""
    return predict_xg(model, shots)


# Median context of the SkillCorner open-data shots: no blocker in the shooting
# triangle, keeper 0.44 m off the shot line and 3.2 m from the goal centre,
# nearest opponent 2.2 m away, one opponent within 3 m.
TYPICAL_SHOT_CONTEXT = dict(n_block=0.0, gk_off_line=0.43943801674550526, gk_dist_goal=3.243262235344732,
                            press_dist=2.188132193991036, n_close=1.0)


def xg_from_here(cells_att: np.ndarray, length: float, width: float, model: dict = OPEN_DATA_XG,
                 context: dict = TYPICAL_SHOT_CONTEXT) -> np.ndarray:
    """xG of a typical shot (not a header, median context) from each point:
    the shooting-danger map used beside expected threat. `cells_att` is
    (k, 2) in metres in the attacking frame, goal attacked at x = length / 2;
    `width` is unused but kept for the danger-map signature."""
    gx, gy = cells_att[:, 0] - length / 2, cells_att[:, 1]
    dist = np.hypot(gx, gy)
    angle = np.abs(np.arctan2(POST_Y - gy, -gx) - np.arctan2(-POST_Y - gy, -gx))
    df = pd.DataFrame({"dist": dist, "log_dist": np.log(np.maximum(dist, 0.5)), "angle": angle,
                       "header": 0.0, **context})
    return predict_xg(model, df)
