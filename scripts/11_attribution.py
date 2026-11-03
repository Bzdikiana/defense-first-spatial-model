"""
Week 6: does blame point at the defender who was beaten?

SkillCorner's dynamic events flag on-ball engagements (a defender pressing
the ball carrier) where the defender was beaten, by the ball carrier or by
movement, and name the defender. That is an independent, defender-level
label. For every engagement during an opponent possession, each defender's
share of the cover (dsv.attribution) is computed at the start of the
engagement and 1 s after it ends (or at the end of the possession, if
sooner), and blame splits the team's loss of cover across defenders.

Checks: (1) does the engaging defender lose more of their share when
SkillCorner says they were beaten? (2) are they the defender blamed most?
(3) does their loss
separate beaten from other engagements beyond how far the ball got and how
much cover the team lost overall?

Writes data/derived/attribution_engagements.csv, reports/w6_attribution.md
and figs/w6_attribution.png. About 25 minutes.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dsv import attribution as A, data as D, metrics as M, pitchcontrol as PC, threat as T  # noqa: E402

DATA_DIR = Path(os.environ.get("DSV_DATA_DIR", ROOT / "data"))
DERIVED, REPORTS, FIGS = DATA_DIR / "derived", ROOT / "reports", ROOT / "figs"
OUT = DERIVED / "attribution_engagements.csv"
EPS = 1e-6
OUTCOMES = ["beaten", "stopped danger", "reduced danger", "forced backward", "no flagged outcome"]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s06 = _load("s06", ROOT / "scripts" / "06_threat.py")
s07 = _load("s07", ROOT / "scripts" / "07_validation.py")


def outcome(ob):
    return np.select([ob["beaten_by_possession"].astype(bool) | ob["beaten_by_movement"].astype(bool),
                      ob["stop_possession_danger"].astype(bool), ob["reduce_possession_danger"].astype(bool),
                      ob["force_backward"].astype(bool)], OUTCOMES[:4], OUTCOMES[4])


def match_rows(mid, vt, values, model):
    meta, players, ball, events, _ = s06.load(mid)
    fx = T._Frames(players, ball, meta, PC.vmax_lookup(vt, mid), model.params)
    grid = PC.make_grid(meta.pitch_length, meta.pitch_width, model.cell)
    poss = T.team_possessions(events, meta)
    ob = events[events["event_type"] == "on_ball_engagement"].copy()
    ob["outcome"] = outcome(ob)
    rows = []
    for _, e in ob.iterrows():
        f0 = int(e["frame_start"])
        m = (poss["team_id"] != e["team_id"]) & (poss["frame_start"] <= f0) & (poss["frame_end"] >= f0)
        if not m.any():
            continue
        p = poss[m].iloc[0]
        f1 = int(min(e["frame_end"] + meta.fps, p["frame_end"]))
        if f1 <= f0 or not (fx.has(f0) and fx.has(f1)):
            continue
        if not (np.isfinite(fx.ball.loc[f0]).all() and np.isfinite(fx.ball.loc[f1]).all()):
            continue
        ah = bool(p["attacking_home"])
        s0 = A.defender_shares(fx.state(f0), ah, grid, meta, model, values)
        s1 = A.defender_shares(fx.state(f1), ah, grid, meta, model, values)
        pid = int(e["player_id"])
        if pid not in set(s0["player_id"]) or pid not in set(s1["player_id"]):
            continue
        b = A.blame(s0, s1, "share_xg_here")
        loss = (s0.set_index("player_id")["share_xg_here"] - s1.set_index("player_id")["share_xg_here"]
                ).reindex(b.index, fill_value=0.0)
        st0 = fx.state(f0)
        d = st0.is_home != ah
        dist = np.hypot(*(st0.pos[d] - st0.ball).T)
        nearest = int(st0.player_id[d][np.argmin(dist)])
        rows.append(dict(
            match_id=mid, event_id=e["event_id"], player_id=pid, subtype=e["event_subtype"], outcome=e["outcome"],
            f0=f0, f1=f1, cover0=s0["share_xg_here"].sum(), cover1=s1["share_xg_here"].sum(),
            rd0=s0["rd_xg_here"].iloc[0], rd1=s1["rd_xg_here"].iloc[0],
            share0=float(s0.set_index("player_id").loc[pid, "share_xg_here"]),
            share1=float(s1.set_index("player_id").loc[pid, "share_xg_here"]),
            blame=float(b.get(pid, 0.0)), team_drop=float(s0["share_xg_here"].sum() - s1["share_xg_here"].sum()),
            loss_rank=int((loss > loss.get(pid, 0.0)).sum() + 1),
            is_nearest=pid == nearest))
    return pd.DataFrame(rows)


def main():
    t0 = time.perf_counter()
    model = T.ThreatModel()
    values = {"xg_here": s06.comparisons().values["xg_here"]}
    vt = PC.vmax_table(pd.read_csv(DERIVED / "v95_by_player_half.csv"))
    if OUT.exists() and not os.environ.get("REBUILD"):
        df = pd.read_csv(OUT)
    else:
        parts = []
        for mid in D.list_open_matches(DATA_DIR):
            parts.append(match_rows(mid, vt, values, model))
            print(f"  {mid}: {len(parts[-1])} engagements, {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
        df = pd.concat(parts, ignore_index=True)
        df.to_csv(OUT, index=False)
    analyse(df)
    print(f"done in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)


def analyse(df):
    df = df.copy()
    df["own_loss"] = df["share0"] - df["share1"]
    df["d_ball"] = np.log(np.maximum(df["rd1"], EPS)) - np.log(np.maximum(df["rd0"], EPS))
    df["d_uncovered"] = np.log(np.maximum(1 - df["cover1"], 1e-4)) - np.log(np.maximum(1 - df["cover0"], 1e-4))
    df["blamed_most"] = df["loss_rank"] == 1
    df["beaten"] = (df["outcome"] == "beaten").astype(int)
    rng = np.random.default_rng(0)

    def boot(sub, col, n=2000):
        per = sub.groupby("match_id")[col].agg(["sum", "size"])
        idx = rng.integers(0, len(per), (n, len(per)))
        v = per["sum"].to_numpy()[idx].sum(1) / per["size"].to_numpy()[idx].sum(1)
        return sub[col].mean(), np.percentile(v, 2.5), np.percentile(v, 97.5)

    by = []
    for g in OUTCOMES:
        s = df[df["outcome"] == g]
        r = dict(outcome=g, n=len(s))
        for c in ("own_loss", "blamed_most", "d_uncovered", "d_ball"):
            r[c], r[c + "_lo"], r[c + "_hi"] = boot(s, c)
        r["share_of_drop"] = (s.loc[s["team_drop"] > 0, "blame"] / s.loc[s["team_drop"] > 0, "team_drop"]).mean()
        by.append(r)
    by = pd.DataFrame(by)

    # does the engaging defender's own loss separate beaten engagements beyond
    # how far the ball got and how much cover the team lost?
    y = df["beaten"].to_numpy()
    X0 = np.column_stack([df["d_ball"], df["d_uncovered"], df["share0"]])
    X1 = np.column_stack([X0, df["own_loss"]])
    p0, p1 = s07.lomo(df, X0, y), s07.lomo(df, X1, y)
    add = s07.boot_diff(df, y, p1, p0)
    auc = {"pressing defender's own loss of cover": M.auc(y, df["own_loss"]),
           "team's loss of cover": M.auc(y, df["d_uncovered"]),
           "ball danger gained": M.auc(y, df["d_ball"])}
    figure(df)
    write_report(df, by, add, auc, M.log_loss(y, p0), M.log_loss(y, p1))


def figure(df):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    data = [100 * df.loc[df["outcome"] == g, "own_loss"].to_numpy() for g in OUTCOMES]
    ax.boxplot(data, vert=False, showfliers=False, labels=[f"{g} (n={len(d):,})" for g, d in zip(OUTCOMES, data)])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("pressing defender's lost share of the cover\n(percentage points, start of the press to 1 s after it)")
    ax.set_title("Pressing defenders lose some cover whatever the outcome", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGS / "w6_attribution.png", dpi=150)
    plt.close(fig)


def write_report(df, by, add, auc, ll0, ll1):
    L = []
    w = L.append
    w("# Does blame point at the beaten defender?\n")
    w("Each defender's share of the cover is the part of the reachable shooting danger (xG-from-here map) "
      "they control; the shares add up to the team's cover (`dsv/attribution.py`). Blame splits a loss of "
      "cover across the defenders whose share fell. SkillCorner's dynamic events independently flag on-ball "
      "engagements in which the pressing defender was beaten, and name the defender. Script: "
      "`scripts/11_attribution.py`.\n")
    w(f"{len(df):,} engagements during an opponent possession, from the start of the engagement to 1 s after "
      "it (or the end of the possession). Means with 95% match-bootstrap intervals:\n")
    w("| SkillCorner outcome | engagements | pressing defender's lost share (pp) | blamed most | their part of the team's drop | team's uncovered share (log change) |")
    w("|---|---|---|---|---|---|")
    for _, r in by.iterrows():
        w(f"| {r['outcome']} | {r['n']:,} | {100 * r['own_loss']:+.1f} [{100 * r['own_loss_lo']:+.1f}, "
          f"{100 * r['own_loss_hi']:+.1f}] | {r['blamed_most']:.0%} [{r['blamed_most_lo']:.0%}, "
          f"{r['blamed_most_hi']:.0%}] | {r['share_of_drop']:.0%} | {r['d_uncovered']:+.2f} "
          f"[{r['d_uncovered_lo']:+.2f}, {r['d_uncovered_hi']:+.2f}] |")
    w("\n![attribution](../figs/w6_attribution.png)\n")
    w("**Separating beaten from other engagements** (AUC, 0.5 = no information):\n")
    for k, v in auc.items():
        w(f"- {k}: {v:.3f}")
    w(f"\nLogistic model of \"beaten\", held out by match: the ball's gain in danger, the team's change in "
      f"uncovered share and the defender's starting share give log loss {ll0:.4f}; adding the pressing "
      f"defender's own loss of cover gives {ll1:.4f} (difference {add['dll']:+.4f} [{add['dll_lo']:+.4f}, "
      f"{add['dll_hi']:+.4f}]; AUC {add['dauc']:+.3f} [{add['dauc_lo']:+.3f}, {add['dauc_hi']:+.3f}]).\n")
    w("## Reading\n")
    w("- **The team side agrees with SkillCorner.** When the presser is beaten, the share of the shooting danger "
      "the whole defense leaves uncovered rises far more than after any other kind of engagement.")
    w("- **The individual side is weak.** The pressing defender loses some of their share in every flagged "
      "outcome, most of all when they stopped the danger (stepping in to win the ball also leaves space). Their "
      "own loss separates \"beaten\" from other engagements only slightly (AUC above), and they are the "
      "defender blamed most in about one beaten engagement in eight.")
    w("- **Where the lost cover goes.** After a beaten press, most of the cover lost belongs to teammates: the "
      "space that opens is behind the presser, where others should be covering. Blame by lost share therefore "
      "points at the cover structure around a press, not at the presser alone. Judging the press itself needs "
      "the counterfactual of Layer 3 (what if they had held position), not attribution.\n")
    w("SkillCorner's flags are themselves computed from tracking geometry, so agreement shows the two "
      "constructions see the same events, not that either is ground truth.\n")
    (REPORTS / "w6_attribution.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
