"""
dsv.viz
=======
Pitch plots in the direction-normalized frame (centre origin, metres, home
attacks +x). Plain matplotlib for now; animations come in Week 9.
"""
from __future__ import annotations

import numpy as np
from matplotlib.patches import Arc, Circle, Rectangle

HOME_C, AWAY_C, BALL_C = "#c53030", "#2b6cb0", "#111111"


def draw_pitch(ax, length: float = 105.0, width: float = 68.0, color: str = "#666666", lw: float = 0.8):
    hx, hy = length / 2, width / 2
    kw = dict(fill=False, ec=color, lw=lw, zorder=1)
    ax.add_patch(Rectangle((-hx, -hy), length, width, **kw))
    ax.plot([0, 0], [-hy, hy], color=color, lw=lw, zorder=1)
    ax.add_patch(Circle((0, 0), 9.15, **kw))
    for s in (-1, 1):
        ax.add_patch(Rectangle((s * hx - (16.5 if s > 0 else 0), -20.16), 16.5, 40.32, **kw))
        ax.add_patch(Rectangle((s * hx - (5.5 if s > 0 else 0), -9.16), 5.5, 18.32, **kw))
        ax.add_patch(Arc((s * (hx - 11), 0), 18.3, 18.3, theta1=127 if s > 0 else -53,
                         theta2=233 if s > 0 else 53, ec=color, lw=lw))
    ax.set_xlim(-hx - 2, hx + 2)
    ax.set_ylim(-hy - 2, hy + 2)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    return ax


def plot_players(ax, state, arrows: bool = True, label_ids=None):
    """Players as dots: filled = detected, hollow = extrapolated. Arrows show
    velocity (1 s of travel). The ball is a small black dot."""
    for home, c in ((True, HOME_C), (False, AWAY_C)):
        m = state.is_home == home
        det, ext = m & state.detected, m & ~state.detected
        ax.scatter(*state.pos[det].T, s=36, c=c, ec="white", lw=0.6, zorder=4)
        ax.scatter(*state.pos[ext].T, s=36, facecolors="white", ec=c, lw=1.4, zorder=4)
        if arrows:
            ax.quiver(*state.pos[m].T, *state.vel[m].T, color=c, angles="xy",
                      scale_units="xy", scale=1, width=0.003, zorder=3)
    if label_ids is not None:
        for pid, lab in label_ids.items():
            j = np.flatnonzero(state.player_id == pid)
            if len(j):
                ax.annotate(lab, state.pos[j[0]], xytext=(4, 4), textcoords="offset points",
                            fontsize=7, zorder=5)
    ax.scatter(*state.ball, s=14, c=BALL_C, zorder=5)


def plot_surface(ax, grid, surface, cmap="RdBu_r", vmin=0.0, vmax=1.0, alpha=0.85):
    """Show a (ny, nx) grid surface on the pitch; returns the image."""
    dx = (grid.xs[1] - grid.xs[0]) / 2
    dy = (grid.ys[1] - grid.ys[0]) / 2
    extent = [grid.xs[0] - dx, grid.xs[-1] + dx, grid.ys[0] - dy, grid.ys[-1] + dy]
    return ax.imshow(surface, origin="lower", extent=extent, cmap=cmap,
                     vmin=vmin, vmax=vmax, alpha=alpha, zorder=0)
