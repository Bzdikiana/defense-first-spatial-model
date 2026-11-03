"""
dsv.attribution
===============
Who holds the defense's cover, and who loses it.

Cover splits exactly across defenders. With danger weights
w(x) = A(x | ball) * V(x) (V: expected threat, or xG from here) and reachable
danger RD = sum_x w(x),

    share_i = sum_x w(x) * PC_i(x) / RD          defender i's part of the cover
    sum_i share_i = cover                        (up to the integration tolerance)

because the defense's pitch control is the sum of its players'. Three
readings:

- holding: share_i at a moment, the part of the reachable danger defender i
  controls. Cheap: the pitch control is computed once per frame.
- blame: between two moments when cover falls, loss_i = max(0, share_i(t0) -
  share_i(t1)), scaled so the losses add up to the team's drop in cover. The
  defenders whose control of the danger disappeared. Blame is the reading
  to trust most: errors are easier to measure than contributions
  (Bischofberger, Ma et al. 2026, arXiv 2606.19931), and on SkillCorner data
  cover counts at critical moments, not on average (reports/w6_two_part.md).
- leave-one-out: cover minus cover with defender i removed (pitch control
  recomputed), what the defense would lose without i. It credits position
  as well as control, but costs one pitch-control computation per defender.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Callable, Optional

import numpy as np
import pandas as pd

from . import pitchcontrol as PC
from . import value as V
from .data import MatchMeta
from .threat import ThreatModel, _transition_probs, player_control

ValueFn = Callable[[np.ndarray, float, float], np.ndarray]


def _weights(state: PC.FrameState, attacking_home: bool, cells: np.ndarray, meta: MatchMeta,
             model: ThreatModel, values: Optional[dict]) -> dict:
    """Danger weights A * V for xT and each extra map, in the pitch frame."""
    s = 1.0 if attacking_home else -1.0
    L, W = meta.pitch_length, meta.pitch_width
    cells_att = s * cells
    a = _transition_probs(model.transition, s * state.ball, cells_att, L, W)
    out = {"xt": a * V.zone_value_at(model.xt, cells_att[:, 0] / L + 0.5, cells_att[:, 1] / W + 0.5)}
    for name, fn in (values or {}).items():
        out[name] = a * np.asarray(fn(cells_att, L, W), float)
    return out


def defender_shares(state: PC.FrameState, attacking_home: bool, grid: PC.Grid, meta: MatchMeta,
                    model: ThreatModel = ThreatModel(), values: Optional[dict] = None) -> pd.DataFrame:
    """Each defender's share of the cover at one frame: one row per defender
    with player_id and share_xt plus share_<name> for each extra danger map
    in `values` (name -> function(cells_att, length, width), e.g. the
    xg_here surface of scripts/06_threat.py). The frame's reachable danger
    under each map is repeated in rd_xt / rd_<name>."""
    cells = grid.cells
    pc = player_control(state, cells, model.params)
    d = state.is_home != attacking_home
    out = pd.DataFrame({"player_id": state.player_id[d]})
    for name, w in _weights(state, attacking_home, cells, meta, model, values).items():
        out[f"share_{name}"] = (w[:, None] * pc[:, d]).sum(0) / w.sum()
        out[f"rd_{name}"] = w.sum()
    return out


def blame(shares_t0: pd.DataFrame, shares_t1: pd.DataFrame, col: str = "share_xt") -> pd.Series:
    """Per defender, the share of cover lost between two moments, scaled so
    the losses add up to the team's drop in cover (zero for everyone if
    cover did not fall). Defenders missing at one moment count as zero."""
    s0 = shares_t0.set_index("player_id")[col]
    s1 = shares_t1.set_index("player_id")[col]
    ids = s0.index.union(s1.index)
    s0, s1 = s0.reindex(ids, fill_value=0.0), s1.reindex(ids, fill_value=0.0)
    drop = s0.sum() - s1.sum()
    loss = (s0 - s1).clip(lower=0)
    if drop <= 0 or loss.sum() <= 0:
        return loss * 0.0
    return loss / loss.sum() * drop


def _without(state: PC.FrameState, keep: np.ndarray) -> PC.FrameState:
    return replace(state, player_id=state.player_id[keep], is_home=state.is_home[keep], is_gk=state.is_gk[keep],
                   detected=state.detected[keep], pos=state.pos[keep], vel=state.vel[keep], vmax=state.vmax[keep])


def leave_one_out(state: PC.FrameState, attacking_home: bool, grid: PC.Grid, meta: MatchMeta,
                  model: ThreatModel = ThreatModel(), values: Optional[dict] = None) -> pd.DataFrame:
    """Cover with each defender removed: one row per defender with
    loo_<map> = cover - cover without that defender (what the defense would
    lose), for xT and each extra map."""
    cells = grid.cells
    ws = _weights(state, attacking_home, cells, meta, model, values)
    att = state.is_home == attacking_home

    def cover(st, att_mask):
        pc_att = player_control(st, cells, model.params)[:, att_mask].sum(1)
        return {k: 1 - (w * pc_att).sum() / w.sum() for k, w in ws.items()}

    full = cover(state, att)
    rows = []
    for i in np.flatnonzero(~att):
        keep = np.ones(len(state.player_id), bool)
        keep[i] = False
        c = cover(_without(state, keep), att[keep])
        rows.append({"player_id": state.player_id[i], **{f"loo_{k}": full[k] - c[k] for k in ws}})
    return pd.DataFrame(rows)
