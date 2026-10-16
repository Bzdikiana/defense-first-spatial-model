"""
dsv.xg
======
An expected-goals model that transfers to SkillCorner tracking.

The SkillCorner open data has shots but no xG. This module trains a small,
transparent xG model on StatsBomb open data and applies it to SkillCorner
shots. It uses only features both sources provide: shot geometry, headers, and
the positions of every player at the moment of the shot (StatsBomb freeze
frames; SkillCorner tracking).

Frame: metres, shooting team attacks +x, goal centre at (GOAL_X, 0), posts at
y = +-3.66. StatsBomb coordinates are 120 x 80 "yards" with true-to-scale
penalty areas and goals, so positions are taken relative to the goal and
converted at 0.9144 m per yard.

StatsBomb open data: https://github.com/statsbomb/open-data (free to use with
attribution to StatsBomb).
"""
from __future__ import annotations

import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from . import metrics as M

SB_RAW = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
YD = 0.9144
GOAL_X = 0.0          # features are computed relative to the goal line
POST_Y = 7.32 / 2

# 2015/16 Premier League, La Liga, Serie A and Ligue 1 (all matches) and the
# 2021/22 Indian Super League: 1,632 matches.
TRAIN_SEASONS = [(2, 27), (11, 27), (12, 27), (7, 27), (1238, 108)]
FEATURES = ["dist", "log_dist", "angle", "header", "n_block", "gk_off_line",
            "gk_dist_goal", "press_dist", "n_close"]


# --------------------------------------------------------------------------- #
# Features (shared by both data sources)
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
# StatsBomb
# --------------------------------------------------------------------------- #
def _sb_to_goal_frame(loc):
    """StatsBomb [x, y] (yards, attacking +x to x=120, goal centre y=40) to
    metres relative to the goal centre."""
    return np.array([(loc[0] - 120.0) * YD, (40.0 - loc[1]) * YD])


def _get_json(url: str):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def statsbomb_shots_from_events(events: list, match_id: int) -> list[dict]:
    rows = []
    for e in events:
        if e.get("type", {}).get("name") != "Shot":
            continue
        shot = e["shot"]
        stype = shot.get("type", {}).get("name")
        ff = shot.get("freeze_frame")
        if stype == "Penalty" or not ff:
            continue
        s = _sb_to_goal_frame(e["location"])
        opp = [_sb_to_goal_frame(p["location"]) for p in ff
               if not p["teammate"] and p.get("position", {}).get("name") != "Goalkeeper"]
        gk = [_sb_to_goal_frame(p["location"]) for p in ff
              if not p["teammate"] and p.get("position", {}).get("name") == "Goalkeeper"]
        header = shot.get("body_part", {}).get("name") == "Head"
        f = shot_features(s, header, np.array(opp).reshape(-1, 2), gk[0] if gk else None)
        rows.append(dict(match_id=match_id, minute=e.get("minute"), shot_type=stype,
                         play_pattern=e.get("play_pattern", {}).get("name"),
                         goal=shot.get("outcome", {}).get("name") == "Goal",
                         statsbomb_xg=shot.get("statsbomb_xg"), **f))
    return rows


def fetch_statsbomb_shots(out_csv: str | Path, seasons: Iterable = TRAIN_SEASONS,
                          workers: int = 8) -> pd.DataFrame:
    """Download the seasons' event files and keep only shot rows with features
    (penalties and shots without freeze frames dropped). Cached at out_csv."""
    out_csv = Path(out_csv)
    if out_csv.exists():
        return pd.read_csv(out_csv)
    jobs = []
    for comp, season in seasons:
        for m in _get_json(f"{SB_RAW}/matches/{comp}/{season}.json"):
            jobs.append((comp, season, m["match_id"]))

    def one(job):
        comp, season, mid = job
        rows = statsbomb_shots_from_events(_get_json(f"{SB_RAW}/events/{mid}.json"), mid)
        for r in rows:
            r.update(competition_id=comp, season_id=season)
        return rows

    with ThreadPoolExecutor(workers) as ex:
        rows = [r for part in ex.map(one, jobs) for r in part]
    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


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


# Fitted on 40,541 StatsBomb open-data shots (scripts/04_xg.py,
# reports/w3_xg.md). Leave-one-league-out AUC 0.800; on the 524 non-penalty
# SkillCorner open-data shots AUC 0.755 and 56.7 xG for 60 goals. Coefficients
# are [intercept, *FEATURES, angle^2, dist*header] on the raw features.
OPEN_DATA_XG = dict(features=list(FEATURES), coef=np.array([
    -4.9881467802812605, -0.14006425138607714, 0.9531096018597013, 3.595684559713745,
    -0.5526998803123565, -0.3622347747365759, 0.020532306647584684, 0.15127542373294017,
    0.2416655123118095, -0.051978810552150194, -0.5265501036040059, -0.03683984197569169,
]))


def xg_for_shots(shots: pd.DataFrame, model: dict = OPEN_DATA_XG) -> np.ndarray:
    """xG for a table of shots with FEATURES columns, e.g. the output of
    `skillcorner_shots`. Penalties are not modelled: set them yourself
    (about 0.76)."""
    return predict_xg(model, shots)
