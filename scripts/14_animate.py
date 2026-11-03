"""
Animated replays (GIF) of the explorer's possessions, for the README and the
pitch video: the danger the defense leaves open as a field, the players, and
the chance of a shot in the next 5 s with the defense's cover underneath.

Reads data/derived/explorer.json (scripts/13_explorer.py). Writes
figs/w8_<possession>.gif. Usage: py scripts/14_animate.py [key ...]
(default: beaten goal).
"""
from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dsv import viz as VZ  # noqa: E402

DATA_DIR = Path(os.environ.get("DSV_DATA_DIR", ROOT / "data"))
FIGS = ROOT / "figs"
SUB = 2                 # drawn frames per data frame (10 per second)
ATT, DEF, HEAT = "#d4511c", "#1f5f9e", "#e8890c"
OPEN = LinearSegmentedColormap.from_list("open", [(1, 1, 1, 0), (0.91, 0.54, 0.05, 0.75), (0.78, 0.2, 0.17, 0.95)])


def decode(b):
    return np.frombuffer(base64.b64decode(b), np.uint8).astype(float) / 255


def animate(p, out):
    F, L, W = p["frames"], p["L"], p["W"]
    n = len(F["t"])
    fig = plt.figure(figsize=(9, 7.4), dpi=90)
    ax = fig.add_axes([0.02, 0.30, 0.96, 0.585])
    tl = fig.add_axes([0.08, 0.06, 0.86, 0.17])
    VZ.draw_pitch(ax, L, W, color="#9aa59f")
    img = ax.imshow(np.zeros((p["ny"], p["nx"])), extent=(-L / 2, L / 2, -W / 2, W / 2), origin="lower",
                    cmap=OPEN, vmin=0, vmax=1, interpolation="bilinear", zorder=0)
    att = ax.scatter([], [], s=150, c=ATT, edgecolors="white", linewidths=0.8, zorder=4)
    dfn = ax.scatter([], [], s=150, c=DEF, edgecolors="white", linewidths=0.8, zorder=4)
    halo = ax.scatter([], [], s=[], c=DEF, alpha=0.18, linewidths=0, zorder=3)
    ball = ax.scatter([], [], s=45, c="white", edgecolors="black", linewidths=1.2, zorder=6)
    labels = []
    ax.text(L / 2 - 1, -W / 2 - 1.2, "attack →", ha="right", va="top", fontsize=9, color="#5a6b61")
    title = fig.text(0.02, 0.978, "", fontsize=12.5, weight="bold", va="top")
    sub = fig.text(0.02, 0.944, "", fontsize=10, va="top", color="#33433a")
    ev_text = fig.text(0.02, 0.913, "", fontsize=10, va="top", color="#c8342c")

    t = np.array(F["t"])
    ps = np.array(F["p_shot"])
    tl.fill_between(t, 0, ps, color=HEAT, alpha=0.3, lw=0)
    tl.plot(t, ps, color=HEAT, lw=2, label="chance of a shot in 5 s")
    tl.plot(t, F["cover_xg"], color=DEF, lw=1.6, label="defense's cover")
    tl.plot(t, F["typical_cover"], color="#7d8a83", lw=1.1, ls="--", label="typical cover here")
    for e in p["events"]:
        if e["kind"] == "press":
            tl.plot([e["t"]], [1.04], marker="v", ms=7, clip_on=False,
                    color="#c8342c" if e["outcome"] == "beaten" else "#2b8a5e" if e["outcome"] else "#7d8a83")
        elif e["kind"] == "shot":
            tl.plot([e["t"]], [1.04], marker="D", ms=7, color="#c8342c", clip_on=False)
    tl.set_ylim(0, 1); tl.set_xlim(t[0], t[-1])
    tl.set_yticks([0, 0.5, 1], ["0", "50%", "100%"]); tl.set_xlabel("seconds", fontsize=9)
    tl.tick_params(labelsize=8); tl.spines[["top", "right"]].set_visible(False)
    tl.legend(fontsize=8, frameon=False, loc="upper left", ncol=3, bbox_to_anchor=(0, 1.32))
    cursor = tl.axvline(t[0], color="#17221c", lw=1.4)
    title.set_text(f"{p['title']}: {p['attack']} attacking, {p['defend']} defending")

    # square root: the same perceptual scale as the web explorer, so low danger stays visible
    fields = [np.sqrt(decode(b).reshape(p["ny"], p["nx"])) for b in p["fields"]["open_xg"]]

    def frame(k):
        i, frac = min(k // SUB, n - 1), (k % SUB) / SUB
        j = min(i + 1, n - 1)
        mix = lambda A, B, key: np.array([[a[0] + (b[0] - a[0]) * frac, a[1] + (b[1] - a[1]) * frac]  # noqa: E731
                                          if (b := {q[key]: q for q in B}.get(a[key])) else a[:2] for a in A])
        A, Dn = mix(F["att"][i], F["att"][j], 2), mix(F["dfn"][i], F["dfn"][j], 4)
        img.set_data(fields[i] if frac < 0.5 else fields[j])
        att.set_offsets(A); dfn.set_offsets(Dn)
        sh = F["shares"][i]
        halo.set_offsets(Dn); halo.set_sizes([150 + 9000 * sh.get(str(q[4]), 0) for q in F["dfn"][i]])
        b0, b1 = F["ball"][i], F["ball"][j]
        ball.set_offsets([[b0[0] + (b1[0] - b0[0]) * frac, b0[1] + (b1[1] - b0[1]) * frac]])
        for lab in labels:
            lab.remove()
        labels.clear()
        for (x, y), q in zip(np.vstack([A, Dn]), F["att"][i] + F["dfn"][i]):
            if q[2] is not None:
                labels.append(ax.text(x, y, str(q[2]), ha="center", va="center", fontsize=6.5, color="white",
                                      weight="bold", zorder=5))
        tt = t[i] + (t[j] - t[i]) * frac
        cursor.set_xdata([tt, tt])
        sub.set_text(f"{tt:4.1f} s   chance of a shot in 5 s {100 * ps[i]:4.1f}%   cover {100 * F['cover_xg'][i]:3.0f}% "
                     f"(typical here {100 * F['typical_cover'][i]:3.0f}%)   shaded: danger left open, rings: share of cover")
        ev = [e for e in p["events"] if e["t"] <= tt + 0.05 and e["kind"] != "ball"]
        ev_text.set_text(ev[-1]["label"] if ev else "")
        return []

    anim = FuncAnimation(fig, frame, frames=n * SUB, interval=1000 / (5 * SUB), blit=False)
    anim.save(out, writer=PillowWriter(fps=5 * SUB))
    plt.close(fig)


def main():
    data = json.loads((DATA_DIR / "derived" / "explorer.json").read_text(encoding="utf-8"))
    keys = sys.argv[1:] or ["beaten", "goal"]
    for p in data["possessions"]:
        if p["key"] in keys:
            out = FIGS / f"w8_{p['key']}.gif"
            animate(p, out)
            print(f"{out}: {out.stat().st_size / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
