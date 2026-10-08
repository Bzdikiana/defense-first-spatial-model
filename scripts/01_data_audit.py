"""
Week 1 coverage audit for the Defensive Space Value project.

For one focal match: kinematics, detection coverage (observed vs extrapolated),
figures. For every downloaded open-data match: the same headline numbers, the
conditional back-line detection check, event <-> tracking alignment, and
per-half v95 (saved for the Week 2 vmax input).

Run (from the repo root):
    py scripts/01_data_audit.py                 # focal 1925299 + all cached matches
    MATCH_ID=1925299 py scripts/01_data_audit.py
    DSV_DATA_DIR=/path/to/data py scripts/01_data_audit.py
Download everything first with `py scripts/00_fetch_data.py`.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dsv import data as D  # noqa: E402

MATCH_ID = os.environ.get("MATCH_ID", "1925299")
DATA_DIR = Path(os.environ.get("DSV_DATA_DIR", ROOT / "data"))
FIGS = ROOT / "figs"
REPORTS = ROOT / "reports"
DERIVED = DATA_DIR / "derived"
for d in (FIGS, REPORTS, DERIVED):
    d.mkdir(exist_ok=True, parents=True)

GROUPS = ["Goalkeeper", "Central Defender", "Full Back", "Midfield",
          "Wide Attacker", "Center Forward"]
ZONES = ["own third", "middle third", "opp third"]


def has_tracking(match_id) -> bool:
    f = DATA_DIR / "matches" / str(match_id) / f"{match_id}_tracking_extrapolated.jsonl"
    return f.exists() and f.stat().st_size > 1_000_000


def load_match(match_id):
    match_dir = DATA_DIR / "matches" / str(match_id)
    meta = D.load_match_metadata(match_dir, match_id)
    players, ball = D.load_tracking(match_dir, match_id, meta)
    players, ball = D.normalize_direction(players, ball, meta)
    players = D.add_velocity(players)
    events = D.load_events(match_dir, match_id)
    phases = D.load_phases(match_dir, match_id)
    return meta, players, ball, events, phases


def match_summary(meta, players, ball, events, phases):
    """Headline numbers for the cross-match table, plus the tidy pieces that
    get pooled across matches (zone detection, v95)."""
    dpf = D.detected_per_frame(players)
    by_grp = D.detection_by(players, meta, "group").set_index("grp")["detection_rate"]
    zone = D.detection_by_ball_zone(players, ball, meta)
    zone["match_id"] = meta.match_id
    own = zone[zone["ball_zone"] == "own third"].set_index("grp")["detection_rate"]
    align = D.event_alignment(events, players, ball, meta)
    v95 = D.compute_v95(players, meta, detected_only=True, by_period=True)
    v95["group"] = v95["player_id"].map(lambda i: D.player_group(meta, i))
    v95["match_id"] = meta.match_id
    row = {
        "match_id": meta.match_id,
        "match": f"{meta.raw['home_team']['short_name']} v {meta.raw['away_team']['short_name']}",
        "date": str(meta.raw.get("date_time", ""))[:10],
        "detection": D.detection_overall(players),
        "ball_detection": float(ball["ball_detected"].mean()),
        "frames_all22": float((dpf == 22).mean()),
        "detected_per_frame": float(dpf.mean()),
        "cb_overall": by_grp.get("Central Defender", np.nan),
        "gk_overall": by_grp.get("Goalkeeper", np.nan),
        "cb_own_third_def": own.get("Central Defender", np.nan),
        "fb_own_third_def": own.get("Full Back", np.nan),
        "gk_own_third_def": own.get("Goalkeeper", np.nan),
        "event_coord_err_m": float(np.median(list(align["coord_err_median_m"].values()))),
        "events_matched": align["n_matched"] / max(align["n_events"], 1),
        "reception_lag_frames": align["best_lag_frames"],
        "shots": int((events["end_type"] == "shot").sum()),
        "phases": len(phases),
        "phases_to_shot": int(phases["team_possession_lead_to_shot"].sum()),
    }
    return row, zone, v95, align


def focal_figures(meta, players, ball, v95):
    dpf = D.detected_per_frame(players)
    vs_dist = D.detection_vs_distance_to_ball(players, ball)
    grid = D.detection_grid(players, meta)
    over_time = D.detection_over_time(players)

    # 1. detected-per-frame histogram
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.hist(dpf.to_numpy(), bins=np.arange(-0.5, 23.5, 1), color="#2b6cb0")
    ax.set_xlabel("players detected per frame (of 22)")
    ax.set_ylabel("frames")
    ax.set_title(f"Detected players per frame — {meta.home_name} vs {meta.away_name}", fontsize=9)
    fig.tight_layout(); fig.savefig(FIGS / "detected_per_frame.png", dpi=130); plt.close(fig)

    # 2. detection vs distance to ball
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(vs_dist["dist_mid"], vs_dist["detection_rate"], "o-", color="#c53030")
    ax.set_xlabel("distance to ball (m)"); ax.set_ylabel("detection rate"); ax.set_ylim(0, 1)
    ax.set_title("Detection rate falls off with distance to the ball"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIGS / "detection_vs_distance.png", dpi=130); plt.close(fig)

    # 3. detection grid heatmap; after normalize_direction home attacks +x
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(grid, origin="lower", extent=[-meta.pitch_length/2, meta.pitch_length/2,
                   -meta.pitch_width/2, meta.pitch_width/2], aspect="equal",
                   cmap="viridis", vmin=0, vmax=1)
    ax.set_title("Detection rate by pitch location (home attacks +x)")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    fig.colorbar(im, ax=ax, label="detection rate", shrink=.8)
    fig.tight_layout(); fig.savefig(FIGS / "detection_grid.png", dpi=130); plt.close(fig)

    # 4. detection over time
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for per, g in over_time.groupby("period"):
        ax.plot(g["bin"], g["detection_rate"], label=f"period {per}")
    ax.set_xlabel("match minute (within period)")
    ax.set_ylabel("detection rate"); ax.set_ylim(0, 1); ax.legend(); ax.grid(alpha=.3)
    ax.set_title("Detection rate over time")
    fig.tight_layout(); fig.savefig(FIGS / "detection_over_time.png", dpi=130); plt.close(fig)

    # 5. speed distribution (detected runs only) + v95 by group
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].hist(players["speed_obs"].dropna().to_numpy(), bins=60, color="#2f855a", alpha=.8)
    axes[0].set_xlabel("speed (m/s), detected runs only"); axes[0].set_ylabel("count")
    axes[0].set_title("Speed distribution")
    pg = v95.dropna(subset=["group"]).groupby("group")["v95"].mean().sort_values()
    axes[1].barh(pg.index, pg.to_numpy(), color="#6b46c1")
    axes[1].set_xlabel("mean v95 (m/s)"); axes[1].set_title("Top speed (v95) by position group")
    fig.tight_layout(); fig.savefig(FIGS / "speed_v95.png", dpi=130); plt.close(fig)


def zone_figure(zone_all: pd.DataFrame) -> pd.DataFrame:
    """6. Detection while defending, by ball zone, pooled over matches."""
    k = (zone_all["detection_rate"] * zone_all["n"]).groupby(
        [zone_all["grp"], zone_all["ball_zone"]], observed=True).sum()
    n = zone_all.groupby(["grp", "ball_zone"], observed=True)["n"].sum()
    pooled = (k / n).unstack()[ZONES]
    groups = [g for g in GROUPS if g in pooled.index]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    xs = np.arange(len(groups)); wd = 0.26
    for i, (z, c) in enumerate(zip(ZONES, ["#c53030", "#dd8a2c", "#8a8f98"])):
        ax.bar(xs + (i - 1) * wd, pooled.loc[groups, z], wd, label=f"ball in {z}", color=c)
    ax.set_xticks(xs, [g.replace(" ", "\n") for g in groups], fontsize=8)
    ax.set_ylim(0, 1); ax.set_ylabel("detection rate"); ax.grid(axis="y", alpha=.3)
    ax.set_title("Detection while defending, by ball zone (relative to own goal)", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(FIGS / "detection_by_ball_zone.png", dpi=130); plt.close(fig)
    return pooled


def pct(v) -> str:
    return "n/a" if pd.isna(v) else f"{v:.1%}"


def write_report(fb: dict, table: pd.DataFrame, pooled: pd.DataFrame) -> str:
    meta, players, ball, al = fb["meta"], fb["players"], fb["ball"], fb["align"]
    lines = []
    w = lines.append
    w("# Week 1 data audit\n")
    w(f"Focal match: {meta.home_name} vs {meta.away_name} (match {meta.match_id}). "
      f"The cross-match sections cover all {len(table)} SkillCorner open-data matches.\n")

    w("## Focal match: format")
    w(f"- Date: {meta.raw.get('date_time')}")
    w(f"- Final score: {meta.home_name} {meta.raw.get('home_team_score')} - "
      f"{meta.raw.get('away_team_score')} {meta.away_name}")
    w(f"- Pitch: {meta.pitch_length:.0f} x {meta.pitch_width:.0f} m; "
      f"frame rate {D.infer_fps(ball):.0f} fps (verified from timestamps)")
    w("- Periods: " + "; ".join(
        f"P{p} {info['duration_minutes']:.1f} min (frames {info['start_frame']}-{info['end_frame']})"
        for p, info in meta.periods.items()))
    w(f"- In-play frames parsed: {players['frame'].nunique():,}; players in tracking: "
      f"{players['player_id'].nunique()} (roster incl. subs: {len(meta.players)})")
    w(f"- Home attacking direction by period: {meta.home_side} "
      f"(normalized so home attacks +x in both periods)\n")

    dpf = fb["dpf"]
    w("## Focal match: detection coverage")
    w(f"- Overall position detection rate: **{D.detection_overall(players):.1%}**; "
      f"the rest is SkillCorner extrapolation. Ball detection: {ball['ball_detected'].mean():.1%}")
    w(f"- Frames with all 22 detected: {(dpf == 22).mean():.1%}. Players detected per frame: "
      f"mean {dpf.mean():.1f}, median {int(dpf.median())}, range {int(dpf.min())}-{int(dpf.max())}")
    w("- By period: " + ", ".join(f"P{int(r['grp'])} {r['detection_rate']:.1%}"
                                  for _, r in fb["by_period"].iterrows()))
    w("- By position group, whole match (GKs split out of SkillCorner's 'Other'):")
    for _, r in fb["by_grp"].iterrows():
        w(f"  - {r['grp']}: {r['detection_rate']:.1%} (n={int(r['n']):,})")
    w("- By distance to ball: " + ", ".join(
        f"{r['dist_mid']:.0f} m {r['detection_rate']:.0%}" for _, r in fb["vs_dist"].iterrows()))
    w("- Lowest-detection players: " + ", ".join(
        f"{r['grp']} {r['detection_rate']:.0%}" for _, r in fb["by_player"].head(5).iterrows()))
    w("")

    w("## Back-line detection when it matters (all matches pooled)")
    w("Whole-match rates are dominated by frames where the ball is at the other end. "
      "Restricted to frames where the player's team is **out of possession**, split by "
      "where the ball is relative to that player's own goal:\n")
    w("| group | ball in own third | middle third | opp third |")
    w("|---|---|---|---|")
    for g in GROUPS:
        if g in pooled.index:
            w(f"| {g} | {pct(pooled.loc[g, 'own third'])} | {pct(pooled.loc[g, 'middle third'])} "
              f"| {pct(pooled.loc[g, 'opp third'])} |")
    w("")

    w("## Event <-> tracking alignment (focal match)")
    w(f"- {al['n_matched']:,} of {al['n_events']:,} events join to a tracking row on "
      f"(frame_start, player_id).")
    w("- Event coordinates are rotated so the event's own team attacks +x. After "
      "`orient_events`, median distance between the event start location and the "
      "player's tracking position at frame_start: " + ", ".join(
          f"{k} {v:.2f} m" for k, v in al["coord_err_median_m"].items()))
    lag = al["best_lag_frames"]
    d = al["possession_ball_dist_by_lag"]
    w(f"- Player-to-ball distance for possession events is smallest at frame_start "
      f"{lag:+d} frames ({d[lag]:.2f} m vs {d[0]:.2f} m at +0): the ball arriving at "
      f"the receiver, not a sync offset. **No frame offset to correct** in the native "
      f"SkillCorner events.")
    w("- The open data has **no shot xG**. Shots are `end_type == 'shot'` on "
      "player_possession events; phases carry `team_possession_lead_to_shot`.\n")

    w("## Kinematics (focal match)")
    sp = players["speed_obs"].dropna()
    w(f"- Speed from detected runs only (`speed_obs`): median {sp.median():.2f} m/s, "
      f"max {sp.max():.1f} m/s. {players['speed_obs'].notna().mean():.1%} of player-frames "
      f"have one (detected runs at least one smoothing window long).")
    v = fb["v95"].dropna(subset=["group"])
    w("- Mean v95 by group and half (m/s):")
    for g, gg in v.groupby("group"):
        w(f"  - {g}: " + ", ".join(
            f"P{int(p)} {x:.2f}" for p, x in gg.groupby("period")["v95"].mean().items()))
    fastest = fb["v95"].sort_values("v95", ascending=False).head(3)
    w("- Fastest player-halves: " + ", ".join(
        f"{r['name']} P{int(r['period'])} ({r['v95']:.1f})" for _, r in fastest.iterrows()))
    w("")

    w("## Cross-match summary")
    w("| match | date | detection | all 22 | CB whole | CB own-3rd def | FB own-3rd def | "
      "GK whole | GK own-3rd def | event err (m) | shots | phases→shot |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in table.iterrows():
        w(f"| {r['match']} ({r['match_id']}) | {r['date']} | {pct(r['detection'])} | "
          f"{pct(r['frames_all22'])} | {pct(r['cb_overall'])} | {pct(r['cb_own_third_def'])} | "
          f"{pct(r['fb_own_third_def'])} | {pct(r['gk_overall'])} | {pct(r['gk_own_third_def'])} | "
          f"{r['event_coord_err_m']:.2f} | {r['shots']} | {r['phases_to_shot']} |")
    w(f"\nDetection across matches: mean {table['detection'].mean():.1%}, range "
      f"{table['detection'].min():.1%}-{table['detection'].max():.1%}. "
      f"Shots: {table['shots'].sum()} in total. Phases leading to a shot: "
      f"{table['phases_to_shot'].sum()} of {table['phases'].sum()}.\n")

    w("## Figures")
    for fn in ["detected_per_frame", "detection_vs_distance", "detection_grid",
               "detection_over_time", "speed_v95", "detection_by_ball_zone"]:
        w(f"- [figs/{fn}.png](../figs/{fn}.png)")
    w("")
    w("## Derived files")
    w("- `data/derived/w1_match_summary.csv`: the cross-match table")
    w("- `data/derived/v95_by_player_half.csv`: per-player, per-half v95 for every match "
      "(Week 2 vmax input)")
    return "\n".join(lines)


def main() -> None:
    focal = int(MATCH_ID)
    if not has_tracking(focal):
        print("Downloading match", focal, "...")
        D.download_open_match(focal, DATA_DIR)

    ids = [i for i in D.list_open_matches(DATA_DIR) if has_tracking(i)]
    if focal not in ids:
        ids.append(focal)
    rows, zones, v95s, fb = [], [], [], None
    for mid in ids:
        meta, players, ball, events, phases = load_match(mid)
        row, zone, v95, align = match_summary(meta, players, ball, events, phases)
        rows.append(row); zones.append(zone); v95s.append(v95)
        print(f"{mid}: detection {row['detection']:.1%}, "
              f"CB own-third (defending) {pct(row['cb_own_third_def'])}")
        if mid == focal:
            focal_figures(meta, players, ball, v95)
            fb = dict(
                meta=meta, players=players, ball=ball, align=align, v95=v95,
                by_period=D.detection_by(players, meta, "period"),
                by_grp=D.detection_by(players, meta, "group"),
                by_player=D.detection_by(players, meta, "player"),
                vs_dist=D.detection_vs_distance_to_ball(players, ball),
                dpf=D.detected_per_frame(players),
            )
        del players, ball, events

    table = pd.DataFrame(rows).sort_values("date")
    pooled = zone_figure(pd.concat(zones, ignore_index=True))
    table.to_csv(DERIVED / "w1_match_summary.csv", index=False)
    pd.concat(v95s, ignore_index=True).to_csv(DERIVED / "v95_by_player_half.csv", index=False)

    out = REPORTS / "w1_data_audit.md"
    out.write_text(write_report(fb, table, pooled), encoding="utf-8")
    print(f"\nreport -> {out}; figures -> {FIGS}")


if __name__ == "__main__":
    main()
