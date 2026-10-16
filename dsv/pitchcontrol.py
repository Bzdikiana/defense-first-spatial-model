"""
dsv.pitchcontrol
================
Spearman (2017) time-to-intercept pitch control on a 105 x 68 grid, with
per-player, per-half vmax from v95 and a renormalized leave-one-out.

Model, for grid cell x and player j (all times in seconds):
- arrival time  tau_j(x) = t_react + |x - (p_j + v_j * t_react)| / vmax_j
  (keep current velocity for the reaction time, then run straight at vmax)
- intercept probability by time T: f_j(x, T) = logistic((T - tau_j(x)) * pi / (sqrt(3) * sigma_j)),
  where sigma_j is `sigma`, or `sigma_extrap` when player j's position in this
  frame is SkillCorner extrapolation rather than a detection
- control accrual  dPC_j/dT = (1 - sum_k PC_k) * f_j(x, T) * lambda_j, integrated
  from the ball's arrival time at x for `t_max` seconds.

Integration: f is held constant within each dT step and the step is solved
exactly (the uncontrolled share decays by exp(-sum_j f_j lambda_j dT) and the
decrease is split in proportion to f_j lambda_j). A plain Euler step overshoots
total control > 1 whenever several players are already at a cell before the
ball, which is common on cells far from the ball.

Coordinates are the direction-normalized tracking frame (home attacks +x),
centre origin, metres. Offside is not modelled here; it belongs to the
accessibility weight (dsv.value).
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Optional

import numpy as np
import pandas as pd

from .data import MatchMeta, player_group


@dataclass(frozen=True)
class PCParams:
    t_react: float = 0.7        # s, reaction time
    sigma: float = 0.45         # s, spread of the intercept-time logistic
    sigma_extrap: Optional[float] = None  # s, spread for extrapolated positions
                                          # (is_detected False); None = sigma
    lam: float = 4.3            # 1/s, control rate
    lam_gk_mult: float = 3.0    # goalkeepers control the ball faster (hands)
    ball_speed: float = 15.0    # m/s, straight-line ball travel
    vmax_default: float = 5.0   # m/s, when a player has no v95
    dt: float = 0.04            # s, integration step
    t_max: float = 10.0         # s, integration window after ball arrival
    tol: float = 0.01           # stop once every cell has >= 1 - tol control


# Fitted to 15,078 passes in the 20 SkillCorner open-data matches (A-League
# 2024/25, broadcast tracking): match-grouped 5-fold CV, see
# reports/w2_parameter_fit.md and scripts/03_fit_pitch_control.py. Use with
# vmax from `vmax_table(v95)` (25th-percentile group floor, keepers' own v95).
# Out-of-fold log loss 0.224 vs 0.262 for SPEARMAN_SHAW and 0.243 for the best
# simple baseline. lam is only bounded from below by the data (any value
# >= 40/s fits about equally well). sigma_extrap applies to extrapolated
# positions (is_detected False).
SKILLCORNER_OPEN_DATA = PCParams(
    t_react=0.770, sigma=0.335, sigma_extrap=0.431, lam=40.0, ball_speed=14.70,
)
# The same fit without the detection-aware term, for tracking data that has no
# is_detected flag (out-of-fold log loss 0.225, same vmax rule).
SKILLCORNER_OPEN_DATA_NO_FLAGS = PCParams(
    t_react=0.777, sigma=0.347, lam=40.0, ball_speed=14.20,
)
# The literature defaults (Spearman 2017 as implemented by Laurie Shaw).
SPEARMAN_SHAW = PCParams()


# --------------------------------------------------------------------------- #
# Grid
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Grid:
    xs: np.ndarray  # cell-centre x coordinates, (nx,)
    ys: np.ndarray  # cell-centre y coordinates, (ny,)

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.ys), len(self.xs)

    @property
    def cells(self) -> np.ndarray:
        """(ny * nx, 2) cell centres, row-major so `.reshape(self.shape)` maps
        a per-cell vector back to an image with row 0 at y = -width/2."""
        gx, gy = np.meshgrid(self.xs, self.ys)
        return np.column_stack([gx.ravel(), gy.ravel()])

    @property
    def cell_area(self) -> float:
        return float((self.xs[1] - self.xs[0]) * (self.ys[1] - self.ys[0]))

    def index(self, x: float, y: float) -> int:
        """Flat index of the cell containing (x, y), clipped to the pitch."""
        ix = int(np.argmin(np.abs(self.xs - x)))
        iy = int(np.argmin(np.abs(self.ys - y)))
        return iy * len(self.xs) + ix


def make_grid(length: float = 105.0, width: float = 68.0, cell: float = 1.0) -> Grid:
    nx, ny = int(round(length / cell)), int(round(width / cell))
    xs = -length / 2 + (np.arange(nx) + 0.5) * length / nx
    ys = -width / 2 + (np.arange(ny) + 0.5) * width / ny
    return Grid(xs, ys)


# --------------------------------------------------------------------------- #
# Per-player vmax
# --------------------------------------------------------------------------- #
# 25th percentile of v95 by position group over all 1,049 player-halves of the
# 20 SkillCorner open-data matches (A-League 2024/25), detected runs only. Use
# with `vmax_table(v95, floors=SKILLCORNER_VMAX_FLOORS)` when you have fewer
# matches than that to pool over.
SKILLCORNER_VMAX_FLOORS = {
    "Center Forward": 5.17, "Central Defender": 4.72, "Full Back": 5.16,
    "Goalkeeper": 2.72, "Midfield": 4.99, "Wide Attacker": 5.31,
}


def vmax_table(
    v95: pd.DataFrame, floor_quantile: Optional[float] = 0.25, gk_vmax: Optional[float] = None,
    floors: Optional[dict] = None,
) -> pd.DataFrame:
    """Per-player, per-half vmax from a v95 table (`compute_v95` output with a
    `group` column, pooled over matches, e.g. data/derived/v95_by_player_half.csv).

    vmax = max(v95, floor of the player's position group), where the floor is
    the `floor_quantile` of that group's v95 over all player-halves (None: no
    floor). `floors` ({group: m/s}) overrides the computed floors, e.g.
    SKILLCORNER_VMAX_FLOORS. Goalkeepers get `gk_vmax` if given, otherwise
    their own v95.

    The defaults are the rule that predicted passes best on the open data
    (reports/w2_parameter_fit.md): a 25th-percentile group floor helps a
    little; overriding goalkeepers to 5 m/s clearly hurts.
    """
    vt = v95.copy()
    if floors is None and floor_quantile is not None:
        floors = vt.groupby("group")["v95"].quantile(floor_quantile).to_dict()
    if floors is None:
        vt["vmax_floor"] = np.nan
        vt["vmax"] = vt["v95"]
    else:
        vt["vmax_floor"] = vt["group"].map(floors)
        vt["vmax"] = np.fmax(vt["v95"], vt["vmax_floor"])
    if gk_vmax is not None:
        vt.loc[vt["group"] == "Goalkeeper", "vmax"] = gk_vmax
    return vt


def vmax_lookup(vt: pd.DataFrame, match_id: int) -> dict:
    """{(player_id, period): vmax} for one match."""
    m = vt[vt["match_id"] == match_id]
    return {(int(p), int(per)): float(v) for p, per, v in
            zip(m["player_id"], m["period"], m["vmax"])}


# --------------------------------------------------------------------------- #
# Frame state
# --------------------------------------------------------------------------- #
@dataclass
class FrameState:
    frame: int
    period: int
    player_id: np.ndarray   # (n,)
    is_home: np.ndarray     # (n,) bool
    is_gk: np.ndarray       # (n,) bool
    detected: np.ndarray    # (n,) bool
    pos: np.ndarray         # (n, 2)
    vel: np.ndarray         # (n, 2)
    vmax: np.ndarray        # (n,)
    ball: np.ndarray        # (2,)
    possession: Optional[str]  # 'home team' / 'away team' / None

    @property
    def attacking_home(self) -> Optional[bool]:
        return {"home team": True, "away team": False}.get(self.possession)


def frame_state(
    players: pd.DataFrame, ball: pd.DataFrame, frame: int, meta: MatchMeta,
    vmax: Optional[dict] = None, params: PCParams = PCParams(),
) -> FrameState:
    """Collect one frame's positions, velocities and vmax.

    `players` needs the `add_velocity` columns. `vmax` is a
    {(player_id, period): vmax} dict (see `vmax_lookup`); players missing from
    it get `params.vmax_default`.
    """
    p = players[players["frame"] == frame]
    b = ball[ball["frame"] == frame].iloc[0]
    per = int(b["period"])
    pid = p["player_id"].to_numpy()
    vmax = vmax or {}
    return FrameState(
        frame=int(frame),
        period=per,
        player_id=pid,
        is_home=p["is_home"].to_numpy(dtype=bool),
        is_gk=np.array([player_group(meta, i) == "Goalkeeper" for i in pid]),
        detected=p["is_detected"].to_numpy(dtype=bool),
        pos=p[["x", "y"]].to_numpy(dtype=float),
        vel=np.nan_to_num(p[["vx", "vy"]].to_numpy(dtype=float)),
        vmax=np.array([vmax.get((int(i), per), params.vmax_default) for i in pid]),
        ball=b[["bx", "by"]].to_numpy(dtype=float),
        possession=b["possession_group"] if isinstance(b["possession_group"], str) else None,
    )


# --------------------------------------------------------------------------- #
# Core model
# --------------------------------------------------------------------------- #
def _arrival_times(pos, vel, vmax, points, t_react) -> np.ndarray:
    """tau (..., n): time for each player to reach each point.

    pos, vel: (..., n, 2); vmax: broadcastable to (..., n); points (..., 2).
    Grid use: pos (n, 2), points (cells, 1, 2) -> (cells, n).
    Batch use: pos (P, n, 2), points (P, 1, 2) -> (P, n)."""
    reach = pos + vel * t_react
    dist = np.linalg.norm(points - reach, axis=-1)
    return (t_react + dist / vmax).astype(np.float32)


def _rates(is_gk, params: PCParams) -> np.ndarray:
    return np.where(is_gk, params.lam * params.lam_gk_mult, params.lam).astype(np.float32)


def _alpha(detected, params: PCParams) -> np.ndarray:
    """Logistic slope per player: pi / (sqrt(3) * sigma_j)."""
    k = np.float32(np.pi / np.sqrt(3.0))
    if params.sigma_extrap is None or detected is None:
        return np.float32(k / params.sigma)
    return np.where(detected, k / params.sigma, k / params.sigma_extrap).astype(np.float32)


def _integrate(tau, t_ball, lam, params: PCParams, early_stop: bool = True,
               alpha=None) -> np.ndarray:
    """Spearman control accrual for any batch shape.

    tau (..., n) arrival times; t_ball (...) ball arrival time at each cell or
    point; lam (..., n) control rates with 0 for absent players; alpha
    logistic slopes broadcastable to (..., n) (default from params.sigma).
    Shapes broadcast, so (1, cells, n) times against (k, 1, n) rates gives k
    variants of one frame. Returns per-player control, (..., n) float32.
    """
    if alpha is None:
        alpha = _alpha(None, params)
    shape = np.broadcast_shapes(tau.shape, lam.shape)
    pc = np.zeros(shape, dtype=np.float32)
    remaining = np.ones(shape[:-1] + (1,), dtype=np.float32)
    t_ball = np.asarray(t_ball, dtype=np.float32)[..., None]
    dt = np.float32(params.dt)
    for step in range(int(round(params.t_max / params.dt))):
        f = 1.0 / (1.0 + np.exp(np.clip(-alpha * (t_ball + step * dt - tau), -50, 50)))
        w = f * lam
        rate = w.sum(axis=-1, keepdims=True)
        gain = remaining * (1.0 - np.exp(-rate * dt))
        pc += gain * w / np.maximum(rate, 1e-12)
        remaining -= gain
        if early_stop and step % 5 == 0 and remaining.max() < params.tol:
            break
    return pc


def _control(state: FrameState, grid: Grid, params: PCParams, masks: np.ndarray) -> np.ndarray:
    """Per-player control for k variants of the frame.

    masks: (k, n) bool, which players take part in each variant.
    Returns (k, cells, n) float32, control of each player at each cell.
    """
    cells = grid.cells
    tau = _arrival_times(state.pos, state.vel, state.vmax, cells[:, None, :], params.t_react)
    t_ball = np.linalg.norm(cells - state.ball, axis=-1) / params.ball_speed
    lam = _rates(state.is_gk, params)[None, None, :] * masks[:, None, :]
    return _integrate(tau[None], t_ball[None], lam.astype(np.float32), params,
                      alpha=_alpha(state.detected, params))


@dataclass
class PCResult:
    grid: Grid
    state: FrameState
    pc: np.ndarray  # (cells, n) per-player control

    def team(self, home: bool) -> np.ndarray:
        """(ny, nx) control of the home (True) or away (False) team."""
        cols = self.state.is_home == home
        return self.pc[:, cols].sum(axis=1).reshape(self.grid.shape)

    @property
    def attack(self) -> np.ndarray:
        """(ny, nx) control of the team in possession."""
        if self.state.attacking_home is None:
            raise ValueError(f"frame {self.state.frame}: no team in possession")
        return self.team(self.state.attacking_home)

    @property
    def total(self) -> np.ndarray:
        return self.pc.sum(axis=1).reshape(self.grid.shape)

    def extrapolated_share(self, home: Optional[bool] = None) -> np.ndarray:
        """(ny, nx) share of control held by players whose position in this
        frame is SkillCorner extrapolation (is_detected False). Restricted to
        one team if `home` is given. A per-cell data-quality layer."""
        cols = ~self.state.detected
        tot = self.pc.sum(axis=1)
        if home is not None:
            cols &= self.state.is_home == home
            tot = self.pc[:, self.state.is_home == home].sum(axis=1)
        out = self.pc[:, cols].sum(axis=1) / np.maximum(tot, 1e-9)
        return out.reshape(self.grid.shape)


def pitch_control(state: FrameState, grid: Grid, params: PCParams = PCParams(),
                  exclude: Optional[np.ndarray] = None) -> PCResult:
    """Control surface for one frame. `exclude` (n,) bool leaves players out,
    e.g. attackers in offside positions (dsv.value.offside_players)."""
    masks = np.ones((1, len(state.player_id)), dtype=bool)
    if exclude is not None:
        masks[0] &= ~np.asarray(exclude, dtype=bool)
    return PCResult(grid, state, _control(state, grid, params, masks)[0])


# --------------------------------------------------------------------------- #
# Leave-one-out
# --------------------------------------------------------------------------- #
@dataclass
class LOOResult:
    grid: Grid
    state: FrameState
    defending_home: bool
    base_attack: np.ndarray        # (ny, nx) attacking control with everyone present
    defender_ids: np.ndarray       # (d,)
    attack_without: np.ndarray     # (d, ny, nx) attacking control with defender i removed

    @property
    def delta(self) -> np.ndarray:
        """(d, ny, nx) attacking control each defender is suppressing:
        delta_i(x) = PC_att without i - PC_att >= 0."""
        return self.attack_without - self.base_attack[None]

    def suppression(self, weight: Optional[np.ndarray] = None) -> pd.Series:
        """Per-defender sum of delta_i over the pitch, times `weight` (ny, nx)
        if given (the danger x accessibility surface, from Week 3). Unweighted,
        it is the area (m^2) of attacking control the defender removes."""
        w = 1.0 if weight is None else weight[None]
        vals = (self.delta * w).sum(axis=(1, 2)) * self.grid.cell_area
        return pd.Series(vals, index=self.defender_ids, name="suppression")


def leave_one_out(
    state: FrameState, grid: Grid, params: PCParams = PCParams(),
    defending_home: Optional[bool] = None,
) -> LOOResult:
    """Remove each defender in turn and recompute control.

    Recomputing the Spearman model without player i is the renormalization:
    the accrual ODE hands i's share to whoever else reaches each cell, so the
    surfaces still sum to ~1 and the attacking gain is concentrated where i was
    the one covering. All variants run in one batched integration.
    """
    if defending_home is None:
        if state.attacking_home is None:
            raise ValueError("no team in possession; pass defending_home")
        defending_home = not state.attacking_home
    n = len(state.player_id)
    def_idx = np.flatnonzero(state.is_home == defending_home)
    masks = np.ones((1 + len(def_idx), n), dtype=bool)
    for r, j in enumerate(def_idx, start=1):
        masks[r, j] = False
    pc = _control(state, grid, params, masks)                      # (1+d, cells, n)
    att = pc[:, :, state.is_home != defending_home].sum(axis=2)    # (1+d, cells)
    att = att.reshape((len(masks),) + grid.shape)
    return LOOResult(grid, state, defending_home, att[0], state.player_id[def_idx], att[1:])


# --------------------------------------------------------------------------- #
# Batched point queries (passes, shots, any (frame, location) pairs)
# --------------------------------------------------------------------------- #
@dataclass
class PointBatch:
    """P independent queries, each one frame's players and ball plus one
    target point. Players are padded to a common n per row, with `present`
    marking real ones. Build it from SkillCorner tracking with `point_batch`,
    or fill the arrays from any other source (metres, any consistent frame).
    """
    pos: np.ndarray                          # (P, n, 2)
    vel: np.ndarray                          # (P, n, 2)
    vmax: np.ndarray                         # (P, n)
    is_gk: np.ndarray                        # (P, n) bool
    is_home: np.ndarray                      # (P, n) bool
    present: np.ndarray                      # (P, n) bool
    ball: np.ndarray                         # (P, 2)
    target: np.ndarray                       # (P, 2)
    detected: Optional[np.ndarray] = None    # (P, n) bool
    player_id: Optional[np.ndarray] = None   # (P, n), 0 where padded

    def __len__(self) -> int:
        return len(self.ball)

    def subset(self, idx) -> "PointBatch":
        return PointBatch(**{f.name: None if getattr(self, f.name) is None
                             else getattr(self, f.name)[idx] for f in fields(self)})


def point_batch(
    players: pd.DataFrame, ball: pd.DataFrame, frames, targets, meta: MatchMeta,
    vmax: Optional[dict] = None, params: PCParams = PCParams(), n_max: int = 22,
) -> PointBatch:
    """Batch rows for (frame, target) pairs from direction-normalized tracking
    with `add_velocity` columns. `vmax` is a {(player_id, period): vmax} dict;
    missing players get `params.vmax_default`. Frames may repeat."""
    frames = np.asarray(frames, dtype=int)
    targets = np.asarray(targets, dtype=float)
    uf, inv = np.unique(frames, return_inverse=True)
    p = players[players["frame"].isin(uf)].sort_values(
        ["frame", "is_home", "player_id"], ascending=[True, False, True])
    row = np.searchsorted(uf, p["frame"].to_numpy())
    slot = p.groupby("frame").cumcount().to_numpy()
    if len(slot) and slot.max() >= n_max:
        raise ValueError(f"more than {n_max} players in a frame")
    u = len(uf)

    def grid(values, fill, dtype):
        out = np.full((u, n_max) + values.shape[1:], fill, dtype=dtype)
        out[row, slot] = values
        return out

    period = ball.set_index("frame")["period"].reindex(uf).to_numpy()
    pid = p["player_id"].to_numpy()
    vm = vmax or {}
    gk_ids = {i for i in np.unique(pid) if player_group(meta, int(i)) == "Goalkeeper"}
    per_row = period[row]
    b = ball.set_index("frame")[["bx", "by"]].reindex(uf).to_numpy(dtype=float)
    batch = PointBatch(
        pos=grid(p[["x", "y"]].to_numpy(dtype=float), 0.0, float),
        vel=grid(np.nan_to_num(p[["vx", "vy"]].to_numpy(dtype=float)), 0.0, float),
        vmax=grid(np.array([vm.get((int(i), int(pr)), params.vmax_default)
                            for i, pr in zip(pid, per_row)]), params.vmax_default, float),
        is_gk=grid(np.isin(pid, list(gk_ids)), False, bool),
        is_home=grid(p["is_home"].to_numpy(dtype=bool), False, bool),
        present=grid(np.ones(len(p), bool), False, bool),
        ball=b,
        target=np.zeros((u, 2)),
        detected=grid(p["is_detected"].to_numpy(dtype=bool), False, bool),
        player_id=grid(pid, 0, np.int64),
    ).subset(inv)
    batch.target = targets
    return batch


def control_at_points(batch: PointBatch, params: PCParams = PCParams(),
                      early_stop: bool = True) -> np.ndarray:
    """(P, n) control of each player at each row's target point."""
    vmax = np.where(batch.present, batch.vmax, 1.0)
    tau = _arrival_times(batch.pos, batch.vel, vmax, batch.target[:, None, :], params.t_react)
    t_ball = np.linalg.norm(batch.target - batch.ball, axis=-1) / params.ball_speed
    lam = _rates(batch.is_gk, params) * batch.present
    return _integrate(tau, t_ball, lam, params, early_stop, alpha=_alpha(batch.detected, params))


def team_control_at_points(batch: PointBatch, home, params: PCParams = PCParams(),
                           early_stop: bool = True) -> np.ndarray:
    """(P,) control of the home team (home[i] True) or the away team at each
    row's target. For a pass, `home` is the passing team's side and the result
    is the model's pass-completion probability."""
    pc = control_at_points(batch, params, early_stop)
    side = batch.is_home == np.asarray(home, dtype=bool)[:, None]
    return (pc * side).sum(axis=1)


# --------------------------------------------------------------------------- #
# Fitting on pass outcomes
# --------------------------------------------------------------------------- #
FIT_BOUNDS = {
    "t_react": (0.0, 1.5), "sigma": (0.05, 1.5), "lam": (0.5, 40.0),
    "ball_speed": (5.0, 35.0), "lam_gk_mult": (1.0, 10.0), "sigma_extrap": (0.05, 3.0),
}


def fit_params(
    batch: PointBatch, home, success, init: PCParams = PCParams(),
    fit=("t_react", "sigma", "lam", "ball_speed"), bounds: Optional[dict] = None,
    maxiter: int = 300, tol: float = 1e-4,
):
    """Maximum-likelihood fit of model parameters on pass outcomes.

    The model's completion probability for a pass is the passing team's
    control at the ball's destination (Spearman 2017). Minimizes log loss with
    Nelder-Mead over the parameters named in `fit`, each kept inside its
    bounds by a logistic transform. `tol` tightens the integration's
    early-stop so the objective is smooth. Returns (fitted PCParams, scipy
    OptimizeResult).
    """
    from scipy.optimize import minimize
    from .metrics import log_loss

    bnd = {**FIT_BOUNDS, **(bounds or {})}
    lo = np.array([bnd[k][0] for k in fit], dtype=float)
    hi = np.array([bnd[k][1] for k in fit], dtype=float)
    y = np.asarray(success, dtype=float)
    base = replace(init, tol=tol)

    def to_params(z):
        x = lo + (hi - lo) / (1.0 + np.exp(-z))
        return replace(base, **{k: float(v) for k, v in zip(fit, x)})

    def objective(z):
        return log_loss(y, team_control_at_points(batch, home, to_params(z)))

    start = [getattr(init, k) if getattr(init, k) is not None else init.sigma for k in fit]
    x0 = np.clip(start, lo + 1e-3 * (hi - lo), hi - 1e-3 * (hi - lo))
    z0 = np.log((x0 - lo) / (hi - x0))
    simplex = np.vstack([z0] + [z0 + 0.5 * np.eye(len(fit))[i] for i in range(len(fit))])
    res = minimize(objective, z0, method="Nelder-Mead",
                   options=dict(maxiter=maxiter, initial_simplex=simplex, xatol=1e-3, fatol=1e-6))
    return replace(to_params(res.x), tol=init.tol), res
