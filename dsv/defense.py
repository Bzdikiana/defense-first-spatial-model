"""
dsv.defense
===========
The two-part model: the ball's situation and the defense's cover, combined
in a calibrated model of "a shot soon".

Threat Conceded multiplies the two parts, TC = RD x (1 - cover), so in logs
log TC = log RD + log(1 - cover): the ball's situation and the defense's
share are forced to count equally. On the SkillCorner open data that product
added little beyond a flexible model of where the ball is, while cover with
its own coefficient added clearly, at the moment level and, measured at the
moment of greatest danger, at the possession level
(reports/w6_two_part.md). So the defensive number is cover, read against the
situation, and a logistic model says how much it matters:

    logit P(shot within 5 s) = b0 + controls + f(ball) + beta * logit(cover)

f(ball) is a flexible function of the ball's position and its reachable
danger (`ball_features`). `TypicalCover` predicts the cover an average
defense holds in the same situation, which turns the model into a
counterfactual: how many shots the actual defense prevented, or conceded,
compared with a typical one (`shots_prevented`).

Frames need the columns of `dsv.threat.match_threat` with comparisons:
ball_x, ball_y (attacking frame, goal at +L/2), rd, cover, rd_xg_here,
tc_xg_here, t_in_possession, def_goal_side.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from . import metrics as M
from . import value as V

EPS = 1e-6
FRAME_CONTROLS = ("t_in_possession", "def_goal_side")


def logit(p, clip: float = 1e-4) -> np.ndarray:
    p = np.clip(np.asarray(p, float), clip, 1 - clip)
    return np.log(p / (1 - p))


def add_cover_columns(frames: pd.DataFrame) -> pd.DataFrame:
    """cover_xg_here = 1 - tc_xg_here / rd_xg_here: the defense's share of
    the reachable shooting danger (cover itself is the xT version)."""
    f = frames.copy()
    f["cover_xg_here"] = 1 - f["tc_xg_here"] / f["rd_xg_here"]
    return f


def ball_features(df: pd.DataFrame, length: float = 105.0) -> np.ndarray:
    """Flexible description of where the ball is: distance to goal and its
    square, the angle the goal mouth subtends, |y|, x, and the log reachable
    danger under both danger maps (xT and xG from here)."""
    gx = length / 2
    dx = gx - df["ball_x"].to_numpy(float)
    ay = np.abs(df["ball_y"].to_numpy(float))
    dist = np.hypot(dx, ay)
    angle = np.arctan2(7.32 * dx, dx ** 2 + ay ** 2 - (7.32 / 2) ** 2)
    return np.column_stack([dist, dist ** 2, angle, ay, df["ball_x"].to_numpy(float),
                            np.log(np.maximum(df["rd"].to_numpy(float), EPS)),
                            np.log(np.maximum(df["rd_xg_here"].to_numpy(float), EPS))])


def design(df: pd.DataFrame, controls: Sequence[str] = FRAME_CONTROLS, cover_col: Optional[str] = "cover",
           length: float = 105.0) -> np.ndarray:
    """Controls, ball features and (unless `cover_col` is None) logit cover."""
    parts = [df[c].to_numpy(float)[:, None] for c in controls] + [ball_features(df, length)]
    if cover_col is not None:
        parts.append(logit(df[cover_col])[:, None])
    return np.column_stack(parts)


@dataclass
class TwoPartModel:
    """Logistic model of a shot soon from controls, the ball's situation and
    the defense's cover. `coef` is [intercept, controls..., ball..., cover]."""
    coef: np.ndarray
    controls: tuple
    cover_col: str
    length: float = 105.0

    @property
    def cover_effect(self) -> float:
        """Change in log-odds of a shot per unit of logit(cover); negative
        when more cover means fewer shots."""
        return float(self.coef[-1])

    def predict(self, df: pd.DataFrame, cover: Optional[np.ndarray] = None) -> np.ndarray:
        """P(shot soon). With `cover`, use those values instead of the
        frames' own (e.g. a typical defense's)."""
        d = df if cover is None else df.assign(**{self.cover_col: cover})
        return M.predict_logistic(self.coef, design(d, self.controls, self.cover_col, self.length))


def fit_two_part(df: pd.DataFrame, y, cover_col: str = "cover", controls: Sequence[str] = FRAME_CONTROLS,
                 length: float = 105.0, l2: float = 1e-4) -> TwoPartModel:
    X = design(df, controls, cover_col, length)
    return TwoPartModel(M.fit_logistic(X, np.asarray(y, float), l2=l2), tuple(controls), cover_col, length)


@dataclass
class TypicalCover:
    """Cover an average defense holds in a given situation: least squares of
    logit(cover) on the ball features (plus defenders goal-side)."""
    coef: np.ndarray
    cover_col: str
    length: float = 105.0

    def _X(self, df):
        return np.column_stack([np.ones(len(df)), ball_features(df, self.length), df["def_goal_side"].to_numpy(float)])

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        z = self._X(df) @ self.coef
        return 1 / (1 + np.exp(-z))


def fit_typical_cover(df: pd.DataFrame, cover_col: str = "cover", length: float = 105.0,
                      ridge: float = 1e-6) -> TypicalCover:
    tc = TypicalCover(np.zeros(0), cover_col, length)
    X = tc._X(df)
    mu, sd = X[:, 1:].mean(0), X[:, 1:].std(0) + 1e-12
    Z = np.column_stack([np.ones(len(X)), (X[:, 1:] - mu) / sd])
    w = np.linalg.solve(Z.T @ Z + ridge * np.eye(Z.shape[1]), Z.T @ logit(df[cover_col]))
    b = w[1:] / sd
    tc.coef = np.concatenate([[w[0] - np.sum(b * mu)], b])
    return tc


def shots_prevented(model: TwoPartModel, typical: TypicalCover, df: pd.DataFrame,
                    horizon_s: float = 5.0, frame_dt: float = 0.2) -> np.ndarray:
    """Per frame, shots prevented by the actual cover compared with a
    typical defense in the same situation (negative: conceded). The model
    gives P(shot within `horizon_s`); dividing by the horizon turns it into a
    rate per second, and each frame stands for `frame_dt` seconds, so sums
    over frames approximate shots. The approximation holds while the
    probabilities are small, as they are for nearly all frames."""
    p_typ = model.predict(df, cover=typical.predict(df))
    p_act = model.predict(df)
    return (p_typ - p_act) * frame_dt / horizon_s


def shot_frames(frames: pd.DataFrame, possessions: pd.DataFrame, fps: float, horizon_s: float = 5.0,
                box_x: float = 52.5 - 16.5, box_y: float = 20.16) -> pd.DataFrame:
    """Frame table for the two-part model: frames up to each possession's
    first shot, with y (the shot comes within `horizon_s`), cover_xg_here,
    in_box (the ball is in the penalty area) and box_next (it enters the
    area within `horizon_s`, same possession). `frames` is
    `dsv.threat.match_threat` output with the xg_here comparison."""
    f = frames.drop(columns=[c for c in ("first_shot_frame",) if c in frames.columns]).merge(
        possessions[["match_id", "possession_id", "first_shot_frame"]], on=["match_id", "possession_id"])
    lead = (f["first_shot_frame"] - f["frame"]) / fps
    f = f[~(lead < 0)].copy()
    f["y"] = lead.loc[f.index].between(0, horizon_s, inclusive="right").astype(int)
    f = f.dropna(subset=["tc", "rd", "cover", "tc_xg_here", "rd_xg_here", "ball_x", "ball_y"])
    f = add_cover_columns(f).sort_values(["match_id", "possession_id", "frame"]).reset_index(drop=True)
    f["in_box"] = (f["ball_x"] > box_x) & (f["ball_y"].abs() < box_y)
    box_next = np.zeros(len(f), int)
    fr, ib = f["frame"].to_numpy(), f["in_box"].to_numpy().astype(int)
    for idx in f.groupby(["match_id", "possession_id"]).indices.values():
        c = np.concatenate([[0], np.cumsum(ib[idx])])
        hi = np.searchsorted(fr[idx], fr[idx] + horizon_s * fps, side="right")
        box_next[idx] = (c[hi] - c[np.arange(len(idx)) + 1]) > 0
    f["box_next"] = box_next
    return f


# --------------------------------------------------------------------------- #
# Possession level
# --------------------------------------------------------------------------- #
def possession_moments(frames: pd.DataFrame, possessions: pd.DataFrame, fps: float,
                       lookback_s: float = 2.0, danger_col: str = "rd_xg_here",
                       length: float = 105.0) -> pd.DataFrame:
    """One row per possession with the moment of greatest danger: among
    frames at least `lookback_s` before the first shot (or the end), the one
    with the highest `danger_col`. Returns the possession columns plus
    y (shot or goal), log_duration, start_x, mean_def_goal_side_early,
    neg_min_dist, peak_rd_early, peak_rd_xg_here_early, dist_at_peak,
    abs_y_at_peak, cover_at_peak, cover_xg_here_at_peak,
    mean_cover_xg_here_early, peak_tc_early, peak_tc_xg_here_early."""
    f = frames.drop(columns=[c for c in ("frame_end", "first_shot_frame") if c in frames.columns]).merge(
        possessions[["match_id", "possession_id", "frame_end", "first_shot_frame"]], on=["match_id", "possession_id"])
    f = f[f["frame"] <= f["first_shot_frame"].fillna(f["frame_end"]) - lookback_s * fps].copy()
    f["dist"] = np.hypot(length / 2 - f["ball_x"], f["ball_y"])
    f = f.sort_values(["match_id", "possession_id", "frame"])
    g = f.groupby(["match_id", "possession_id"])
    at = f.loc[g[danger_col].idxmax().values,
               ["match_id", "possession_id", "dist", "ball_y", "cover", "cover_xg_here"]].rename(columns={
        "dist": "dist_at_peak", "ball_y": "y_at_peak", "cover": "cover_at_peak",
        "cover_xg_here": "cover_xg_here_at_peak"})
    at["abs_y_at_peak"] = at.pop("y_at_peak").abs()
    agg = pd.DataFrame({
        "neg_min_dist": -g["dist"].min(), "start_x": g["ball_x"].first(),
        "mean_def_goal_side_early": g["def_goal_side"].mean(),
        "peak_rd_early": g["rd"].max(), "peak_rd_xg_here_early": g["rd_xg_here"].max(),
        "peak_tc_early": g["tc"].max(), "peak_tc_xg_here_early": g["tc_xg_here"].max(),
        "mean_cover_xg_here_early": g["cover_xg_here"].mean(),
    }).reset_index()
    new = [c for c in list(agg.columns) + list(at.columns) if c not in ("match_id", "possession_id")]
    base = possessions.drop(columns=[c for c in new if c in possessions.columns])
    p = base.merge(agg, on=["match_id", "possession_id"]).merge(at, on=["match_id", "possession_id"])
    p["y"] = (p["shot"].astype(bool) | p["goal"].astype(bool)).astype(int)
    p["log_duration"] = np.log(p["duration_s"].clip(lower=0.5))
    return p.reset_index(drop=True)


POSSESSION_CONTROLS = ("log_duration", "mean_def_goal_side_early", "start_x")


def possession_design(p: pd.DataFrame, cover_col: Optional[str] = "cover_at_peak",
                      controls: Sequence[str] = POSSESSION_CONTROLS) -> np.ndarray:
    """Controls, the possession's ball features (closest distance and its
    square, log peak reachable danger under both maps, distance and |y| at
    the moment of peak danger) and logit cover at that moment."""
    nd = p["neg_min_dist"].to_numpy(float)
    parts = [p[c].to_numpy(float)[:, None] for c in controls] + [np.column_stack([
        nd, nd ** 2, np.log(np.maximum(p["peak_rd_early"].to_numpy(float), EPS)),
        np.log(np.maximum(p["peak_rd_xg_here_early"].to_numpy(float), EPS)),
        p["dist_at_peak"].to_numpy(float), p["abs_y_at_peak"].to_numpy(float)])]
    if cover_col is not None:
        parts.append(logit(p[cover_col])[:, None])
    return np.column_stack(parts)


# --------------------------------------------------------------------------- #
# Reachable danger from the ball position alone
# --------------------------------------------------------------------------- #
def reachable_danger(ball_x, ball_y, cells_att: np.ndarray, values: np.ndarray, length: float, width: float,
                     transition: V.TransitionModel = V.OPEN_DATA_TRANSITION, chunk: int = 2000) -> np.ndarray:
    """sum_x A(x | ball) * value(x) for many ball positions (attacking frame),
    the same quantity `frame_threat` reports as rd / rd_<name>; useful for
    frames computed before a danger map was added."""
    bx, by = np.asarray(ball_x, float), np.asarray(ball_y, float)
    out = np.empty(len(bx))
    for s in range(0, len(bx), chunk):
        lp = transition.logpdf(bx[s:s + chunk, None], by[s:s + chunk, None], cells_att[None, :, 0],
                               cells_att[None, :, 1], length / 2, width / 2)
        a = np.exp(lp - lp.max(axis=1, keepdims=True))
        out[s:s + chunk] = (a * values).sum(1) / a.sum(1)
    return out


__all__ = ["add_cover_columns", "shot_frames", "ball_features", "design", "TwoPartModel", "fit_two_part", "TypicalCover",
           "fit_typical_cover", "shots_prevented", "possession_moments", "possession_design",
           "reachable_danger", "logit", "OPEN_DATA_TWO_PART", "OPEN_DATA_TYPICAL_COVER"]


# --------------------------------------------------------------------------- #
# Fitted on the 20 SkillCorner open-data matches
# --------------------------------------------------------------------------- #
# Shot within 5 s from 308,004 frames (5 per second, 10,028 followed by a
# shot), xG-from-here cover (scripts/08_two_part.py, reports/w6_two_part.md).
# Coefficients: intercept, time in possession, defenders goal-side, distance,
# distance^2, goal-mouth angle, |y|, x, log reachable danger (xT), log
# reachable danger (xG from here), logit cover.
OPEN_DATA_TWO_PART = TwoPartModel(coef=np.array([
    2.147685529615747, -0.003954352481676918, -0.09841600357806148, -0.06282893481184518,
    -0.001264098020386546, 0.26484764013013057, -0.0074422873987725, -0.043275529037481594,
    -1.0845818343339175, 0.571472981489067, -1.0422175913438154,
]), controls=FRAME_CONTROLS, cover_col="cover_xg_here")
# Typical cover in a situation: intercept, distance, distance^2, angle, |y|, x,
# log reachable danger (xT), log reachable danger (xG from here), defenders
# goal-side; logit scale.
OPEN_DATA_TYPICAL_COVER = TypicalCover(coef=np.array([
    4.942854756316365, 0.03555940617867066, -0.00011806739385454092, 0.08968538268084161,
    0.009157879228563568, -0.04735357673806021, -0.34480319645599833, 1.317441408345691,
    -0.11416817528326369,
]), cover_col="cover_xg_here")
