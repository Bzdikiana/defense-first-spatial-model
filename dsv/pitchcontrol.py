"""
dsv.pitchcontrol
================
Spearman (2017) time-to-intercept pitch control on a 105 x 68 grid, with
per-player, per-half vmax from v95 and a renormalized leave-one-out.

Model, for grid cell x and player j (all times in seconds):
- arrival time  tau_j(x) = t_react + |x - (p_j + v_j * t_react)| / vmax_j
  (keep current velocity for the reaction time, then run straight at vmax)
- intercept probability by time T: f_j(x, T) = logistic((T - tau_j(x)) * pi / (sqrt(3) * sigma))
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

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .data import MatchMeta, player_group


@dataclass(frozen=True)
class PCParams:
    t_react: float = 0.7        # s, reaction time
    sigma: float = 0.45         # s, spread of the intercept-time logistic
    lam: float = 4.3            # 1/s, control rate
    lam_gk_mult: float = 3.0    # goalkeepers control the ball faster (hands)
    ball_speed: float = 15.0    # m/s, straight-line ball travel
    vmax_default: float = 5.0   # m/s, when a player has no v95
    dt: float = 0.04            # s, integration step
    t_max: float = 10.0         # s, integration window after ball arrival
    tol: float = 0.01           # stop once every cell has >= 1 - tol control


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
def vmax_table(
    v95: pd.DataFrame, floor_quantile: float = 0.25, gk_vmax: float = 5.0
) -> pd.DataFrame:
    """Per-player, per-half vmax from a v95 table (`compute_v95` output with a
    `group` column, pooled over matches, e.g. data/derived/v95_by_player_half.csv).

    vmax = max(v95, floor of the player's position group), where the floor is
    the `floor_quantile` of that group's v95 over all player-halves. The floor
    guards against v95 underestimating players whose sprints were mostly
    off-camera. Goalkeepers get `gk_vmax`: their v95 (~3 m/s) measures how
    little they usually move, not how fast they come for a through ball (their
    99.9th-percentile speed is ~6 m/s).
    """
    vt = v95.copy()
    floors = vt.groupby("group")["v95"].quantile(floor_quantile)
    vt["vmax_floor"] = vt["group"].map(floors)
    vt["vmax"] = np.fmax(vt["v95"], vt["vmax_floor"])
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
def _control(state: FrameState, grid: Grid, params: PCParams, masks: np.ndarray) -> np.ndarray:
    """Per-player control for k variants of the frame.

    masks: (k, n) bool, which players take part in each variant.
    Returns (k, cells, n) float32, control of each player at each cell.
    """
    cells = grid.cells
    reach = state.pos + state.vel * params.t_react                      # (n, 2)
    dist = np.hypot(cells[:, None, 0] - reach[None, :, 0],
                    cells[:, None, 1] - reach[None, :, 1])              # (cells, n)
    tau = (params.t_react + dist / state.vmax[None, :]).astype(np.float32)
    t_ball = (np.hypot(cells[:, 0] - state.ball[0], cells[:, 1] - state.ball[1])
              / params.ball_speed).astype(np.float32)
    lam = np.where(state.is_gk, params.lam * params.lam_gk_mult, params.lam).astype(np.float32)
    lam_m = lam[None, None, :] * masks[:, None, :].astype(np.float32)   # (k, 1, n)
    alpha = np.float32(np.pi / (np.sqrt(3.0) * params.sigma))

    k, n_cells, n = masks.shape[0], len(cells), len(state.player_id)
    pc = np.zeros((k, n_cells, n), dtype=np.float32)
    remaining = np.ones((k, n_cells, 1), dtype=np.float32)
    dt = np.float32(params.dt)
    for step in range(int(round(params.t_max / params.dt))):
        T = t_ball[:, None] + step * dt
        f = 1.0 / (1.0 + np.exp(np.clip(-alpha * (T - tau), -50, 50)))  # (cells, n)
        w = f[None] * lam_m                                             # (k, cells, n)
        rate = w.sum(axis=2, keepdims=True)
        gain = remaining * (1.0 - np.exp(-rate * dt))
        pc += gain * w / np.maximum(rate, 1e-12)
        remaining -= gain
        if step % 5 == 0 and remaining.max() < params.tol:
            break
    return pc


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


def pitch_control(state: FrameState, grid: Grid, params: PCParams = PCParams()) -> PCResult:
    masks = np.ones((1, len(state.player_id)), dtype=bool)
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
