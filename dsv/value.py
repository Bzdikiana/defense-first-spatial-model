"""
dsv.value
=========
The two weights Threat Conceded puts on the attacking team's pitch control:

- V(x), danger: Markov expected threat (xT, Singh 2018) fitted on SkillCorner
  player possessions, with each zone's shot value taken from the xG model.
- A(x | ball), accessibility: where the ball goes next given where it is now
  (Spearman 2017's transition model), fitted on SkillCorner possession changes.

plus offside positions, for checking only (see `offside_players`).

Frames: zone grids are in the attacking team's frame normalized to [0, 1]
(x = 1 is the goal being attacked). Pitch surfaces use dsv.pitchcontrol.Grid
cells in metres, centre origin, attacking +x; flip them for a team attacking -x.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import ndtr

from .data import MatchMeta, player_group


# --------------------------------------------------------------------------- #
# Expected threat
# --------------------------------------------------------------------------- #
def xt_actions(events: pd.DataFrame, meta: MatchMeta, max_gap_frames: int = 100) -> pd.DataFrame:
    """Moves and shots from one match's player possessions, in the acting
    team's frame normalized to [0, 1] (x = 1 at the goal attacked).

    Each possession gives up to two actions:
    - a carry from where it starts to where it ends, if they are >= 3 m apart
      (successful: the same player still has the ball);
    - a release where it ends: a shot, or a move that succeeds if the same
      team's next possession starts within `max_gap_frames` (it ends there)
      and fails otherwise. Fouls suffered and unknown endings are skipped.
    """
    pp = events[events["event_type"] == "player_possession"].sort_values("frame_start").reset_index(drop=True)
    nxt = pp[["team_id", "period", "frame_start", "x_start", "y_start"]].shift(-1)
    L, W = meta.pitch_length, meta.pitch_width
    nx = lambda x: np.clip(np.asarray(x, float) / L + 0.5, 0, 1 - 1e-9)  # noqa: E731
    ny = lambda y: np.clip(np.asarray(y, float) / W + 0.5, 0, 1 - 1e-9)  # noqa: E731
    rows = []
    carry = np.hypot(pp["x_end"] - pp["x_start"], pp["y_end"] - pp["y_start"]) >= 3.0
    c = pp[carry]
    rows.append(pd.DataFrame(dict(kind="move", success=True,
                                  x0=nx(c["x_start"]), y0=ny(c["y_start"]),
                                  x1=nx(c["x_end"]), y1=ny(c["y_end"]))))
    shot = pp["end_type"] == "shot"
    rows.append(pd.DataFrame(dict(kind="shot", success=False, x0=nx(pp.loc[shot, "x_end"]),
                                  y0=ny(pp.loc[shot, "y_end"]), x1=np.nan, y1=np.nan)))
    mv = pp["end_type"].isin(["pass", "possession_loss", "clearance"])
    keep = (mv & (nxt["team_id"] == pp["team_id"]) & (nxt["period"] == pp["period"])
            & ((nxt["frame_start"] - pp["frame_end"]) <= max_gap_frames))
    rows.append(pd.DataFrame(dict(
        kind="move", success=keep[mv].to_numpy(),
        x0=nx(pp.loc[mv, "x_end"]), y0=ny(pp.loc[mv, "y_end"]),
        x1=np.where(keep[mv], nx(nxt.loc[mv, "x_start"].fillna(0)), np.nan),
        y1=np.where(keep[mv], ny(nxt.loc[mv, "y_start"].fillna(0)), np.nan))))
    out = pd.concat(rows, ignore_index=True)
    out = out[np.isfinite(out["x0"]) & np.isfinite(out["y0"])].reset_index(drop=True)
    # a "successful" move whose end location is missing cannot be placed
    out.loc[out["success"] & ~(np.isfinite(out["x1"]) & np.isfinite(out["y1"])), "success"] = False
    out["match_id"] = meta.match_id
    return out


def zone_shot_value(nx: int, ny: int, xg_at: Callable[[np.ndarray, np.ndarray], np.ndarray],
                    length: float = 105.0, width: float = 68.0) -> np.ndarray:
    """(ny, nx) goal probability of a shot from each zone centre. `xg_at`
    maps arrays of (distance to goal centre, visible goal angle) to xG."""
    xc = (np.arange(nx) + 0.5) / nx * length - length          # metres from the goal line (< 0)
    yc = ((np.arange(ny) + 0.5) / ny - 0.5) * width
    gx, gy = np.meshgrid(xc, yc)
    dist = np.hypot(gx, gy)
    angle = np.abs(np.arctan2(3.66 - gy, -gx) - np.arctan2(-3.66 - gy, -gx))
    return xg_at(dist, angle)


def fit_xt(actions: pd.DataFrame, shot_value: np.ndarray, n_iter: int = 200,
           tol: float = 1e-8) -> np.ndarray:
    """Markov expected threat on the zone grid of `shot_value` (ny, nx).

    xT(z) = P(shot | z) * shot_value(z) + P(move | z) * sum_z' T(z -> z') xT(z'),
    where T counts only successful moves (a failed move is worth 0).
    """
    ny, nx = shot_value.shape
    zi = lambda x, y: (np.minimum((y * ny).astype(int), ny - 1) * nx  # noqa: E731
                       + np.minimum((x * nx).astype(int), nx - 1))
    z0 = zi(actions["x0"].to_numpy(), actions["y0"].to_numpy())
    n = np.bincount(z0, minlength=nx * ny).astype(float)
    is_shot = (actions["kind"] == "shot").to_numpy()
    s = np.bincount(z0[is_shot], minlength=nx * ny) / np.maximum(n, 1)
    m = np.bincount(z0[~is_shot], minlength=nx * ny) / np.maximum(n, 1)
    ok = (~is_shot) & actions["success"].to_numpy(dtype=bool)
    z1 = zi(actions.loc[ok, "x1"].to_numpy(), actions.loc[ok, "y1"].to_numpy())
    T = np.zeros((nx * ny, nx * ny))
    np.add.at(T, (z0[ok], z1), 1.0)
    T /= np.maximum(np.bincount(z0[~is_shot], minlength=nx * ny), 1)[:, None]
    g = shot_value.ravel()
    xt = np.zeros(nx * ny)
    for _ in range(n_iter):
        new = s * g + m * (T @ xt)
        if np.abs(new - xt).max() < tol:
            xt = new
            break
        xt = new
    return xt.reshape(ny, nx)


def zone_lookup(xt: np.ndarray, x01, y01) -> np.ndarray:
    """xT of the zones containing normalized points (x01, y01)."""
    ny, nx = xt.shape
    ix = np.minimum((np.asarray(x01) * nx).astype(int), nx - 1)
    iy = np.minimum((np.asarray(y01) * ny).astype(int), ny - 1)
    return xt[iy, ix]


def zone_value_at(zones: np.ndarray, x01, y01) -> np.ndarray:
    """Bilinear interpolation between zone centres at normalized points
    (x01, y01) in the attacking frame; constant beyond the outer centres."""
    ny, nx = zones.shape
    fx = np.clip(np.asarray(x01, float) * nx - 0.5, 0, nx - 1)
    fy = np.clip(np.asarray(y01, float) * ny - 0.5, 0, ny - 1)
    x0, y0 = np.floor(fx).astype(int), np.floor(fy).astype(int)
    x1, y1 = np.minimum(x0 + 1, nx - 1), np.minimum(y0 + 1, ny - 1)
    tx, ty = fx - x0, fy - y0
    return (zones[y0, x0] * (1 - tx) * (1 - ty) + zones[y0, x1] * tx * (1 - ty)
            + zones[y1, x0] * (1 - tx) * ty + zones[y1, x1] * tx * ty)


def zone_surface(zones: np.ndarray, grid, length: float, width: float) -> np.ndarray:
    """A zone grid (attacking frame, x = 1 at the goal attacked) on pitch
    grid cells (metres, attacking +x), bilinearly interpolated."""
    c = grid.cells
    return zone_value_at(zones, c[:, 0] / length + 0.5, c[:, 1] / width + 0.5).reshape(grid.shape)


# --------------------------------------------------------------------------- #
# Accessibility: where the ball goes next
# --------------------------------------------------------------------------- #
def ball_transitions(events: pd.DataFrame, ball_df: pd.DataFrame, meta: MatchMeta,
                     max_gap_frames: int = 100) -> pd.DataFrame:
    """Each possession's release point and where the next possession (either
    team) starts, from tracked ball positions, in the releasing team's
    attacking frame (metres, centre origin, attacking +x). Shots and
    possessions with no tracked follow-up within `max_gap_frames` are dropped."""
    pp = events[events["event_type"] == "player_possession"].sort_values("frame_start")
    nxt = pp[["frame_start", "period"]].shift(-1)
    ok = ((pp["end_type"] != "shot") & (nxt["period"] == pp["period"])
          & (nxt["frame_start"] - pp["frame_end"]).between(1, max_gap_frames))
    pp, nf = pp[ok], nxt.loc[ok, "frame_start"].astype(int)
    b = ball_df.set_index("frame")[["bx", "by"]]
    p0 = b.reindex(pp["frame_end"]).to_numpy()
    p1 = b.reindex(nf).to_numpy()
    s = np.where(pp["team_id"].to_numpy() == meta.home_team_id, 1.0, -1.0)[:, None]
    out = pd.DataFrame(dict(match_id=meta.match_id, bx=s[:, 0] * p0[:, 0], by=s[:, 0] * p0[:, 1],
                            tx=s[:, 0] * p1[:, 0], ty=s[:, 0] * p1[:, 1],
                            half_l=meta.pitch_length / 2, half_w=meta.pitch_width / 2))
    return out.dropna().reset_index(drop=True)


@dataclass
class TransitionModel:
    """Mixture of axis-aligned Gaussians for the next ball location, centred
    on the ball plus a forward shift, truncated to the pitch.

    weights (k,), mu_x (k,) forward shift in m, sx (k,), sy (k,) spreads in m.
    """
    weights: np.ndarray
    mu_x: np.ndarray
    sx: np.ndarray
    sy: np.ndarray

    def _parts(self, bx, by, tx, ty, half_l, half_w):
        b = [np.asarray(a, float)[..., None] for a in (bx, by, tx, ty, half_l, half_w)]
        bx, by, tx, ty, hl, hw = b
        cx = bx + self.mu_x
        dens = (np.exp(-0.5 * ((tx - cx) / self.sx) ** 2 - 0.5 * ((ty - by) / self.sy) ** 2)
                / (2 * np.pi * self.sx * self.sy))
        mass = ((ndtr((hl - cx) / self.sx) - ndtr((-hl - cx) / self.sx))
                * (ndtr((hw - by) / self.sy) - ndtr((-hw - by) / self.sy)))
        return dens, mass

    def logpdf(self, bx, by, tx, ty, half_l, half_w) -> np.ndarray:
        dens, mass = self._parts(bx, by, tx, ty, half_l, half_w)
        return np.log(np.maximum((dens * self.weights).sum(-1), 1e-300)) - np.log((mass * self.weights).sum(-1))

    def surface(self, ball_xy, grid, length: float, width: float) -> np.ndarray:
        """(ny, nx) probability that the ball goes next to each grid cell
        (sums to 1), for a ball at ball_xy in the attacking frame."""
        c = grid.cells
        lp = self.logpdf(ball_xy[0], ball_xy[1], c[:, 0], c[:, 1], length / 2, width / 2)
        p = np.exp(lp - lp.max())
        return (p / p.sum()).reshape(grid.shape)


def fit_transition(tr: pd.DataFrame, k: int = 1, forward: bool = False, isotropic: bool = False,
                   init_sigma: float = 15.0) -> TransitionModel:
    """Maximum-likelihood fit of a k-component transition model.
    `forward` lets each component's centre shift along x; `isotropic` ties
    sx = sy."""
    cols = [tr[c].to_numpy(dtype=float) for c in ("bx", "by", "tx", "ty", "half_l", "half_w")]

    def build(z):
        w = np.exp(np.append(z[:k - 1], 0.0)); w /= w.sum()
        i = k - 1
        mu = z[i:i + k] if forward else np.zeros(k)
        i += k if forward else 0
        sx = np.exp(z[i:i + k])
        sy = sx if isotropic else np.exp(z[i + k:i + 2 * k])
        return TransitionModel(w, mu, sx, sy)

    spreads = np.log(init_sigma * (2.0 ** np.arange(k)) / 2 ** ((k - 1) / 2))
    z0 = np.concatenate([np.zeros(k - 1), np.full(k, 2.0) if forward else [],
                         spreads, [] if isotropic else spreads])
    nll = lambda z: -build(z).logpdf(*cols).mean()  # noqa: E731
    res = minimize(nll, z0, method="Nelder-Mead", options=dict(maxiter=4000, xatol=1e-4, fatol=1e-7))
    return build(res.x)


# --------------------------------------------------------------------------- #
# Fitted on the 20 SkillCorner open-data matches (scripts/05_value.py,
# reports/w3_value.md)
# --------------------------------------------------------------------------- #
# xT on 12 x 8 zones, rows from y = -width/2 to +width/2, columns from the own
# goal to the goal attacked. Read it with `zone_value_at` / `zone_surface`.
OPEN_DATA_XT = np.array([
    [0.00193, 0.00232, 0.00339, 0.00430, 0.00625, 0.00784, 0.01022, 0.01325, 0.01862, 0.02405, 0.02960, 0.03741],
    [0.00268, 0.00295, 0.00438, 0.00540, 0.00735, 0.00932, 0.01171, 0.01584, 0.02109, 0.03256, 0.04134, 0.06371],
    [0.00255, 0.00332, 0.00470, 0.00608, 0.00764, 0.00948, 0.01319, 0.01796, 0.02302, 0.02856, 0.06019, 0.06229],
    [0.00264, 0.00376, 0.00520, 0.00647, 0.00809, 0.01023, 0.01254, 0.01735, 0.02435, 0.03628, 0.10951, 0.37709],
    [0.00286, 0.00394, 0.00533, 0.00682, 0.00842, 0.01057, 0.01246, 0.01780, 0.02137, 0.03542, 0.09483, 0.36641],
    [0.00257, 0.00377, 0.00519, 0.00639, 0.00835, 0.01009, 0.01319, 0.01781, 0.02001, 0.02995, 0.04831, 0.05991],
    [0.00240, 0.00338, 0.00403, 0.00561, 0.00751, 0.00972, 0.01308, 0.01644, 0.02065, 0.02940, 0.04422, 0.04811],
    [0.00189, 0.00280, 0.00388, 0.00442, 0.00644, 0.00835, 0.01119, 0.01389, 0.01908, 0.02394, 0.02689, 0.03323],
])
# Next-ball-location model: 76% short (centred 1.1 m ahead of the ball, spread
# 8.9 m along / 12.0 m across), 24% long (17.6 m ahead, spread 21.1 / 20.1 m).
OPEN_DATA_TRANSITION = TransitionModel(
    weights=np.array([0.7593, 0.2407]), mu_x=np.array([1.1363, 17.6017]),
    sx=np.array([8.9247, 21.1406]), sy=np.array([11.9821, 20.0650]),
)


# --------------------------------------------------------------------------- #
# Offside
# --------------------------------------------------------------------------- #
def offside_players(state, meta: MatchMeta, attacking_home: bool) -> np.ndarray:
    """(n,) True for attackers in an offside position: in the opponents'
    half, ahead of the ball and ahead of the second-last defender (keeper
    included). Uses the frame's positions as they are, extrapolated or not.

    Not used in Threat Conceded: on SkillCorner broadcast tracking it rarely
    agrees with the referee (reports/w3_value.md), so excluding these players
    would mostly drop the wrong ones."""
    s = 1.0 if attacking_home else -1.0
    x = s * state.pos[:, 0]
    att = state.is_home == attacking_home
    dx = np.sort(x[~att])[::-1]
    second_last = dx[1] if len(dx) > 1 else meta.pitch_length / 2
    return att & (x > 0) & (x > s * state.ball[0]) & (x > second_last)
