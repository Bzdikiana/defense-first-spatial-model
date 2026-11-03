"""
dsv.dfl
=======
Load the open DFL/Sportec data (seven 2022/23 Bundesliga and 2. Bundesliga
matches, optical tracking at 25 fps plus official events; Bassek et al. 2025,
Scientific Data 12:195, CC BY 4.0) into the same tables the SkillCorner
pipeline uses, so `dsv.threat` runs on it unchanged.

Needs kloppy (`pip install kloppy`), which downloads and caches the files.

Differences from SkillCorner open data, and how they are handled:
- Optical tracking: every position is observed, so `is_detected` is True
  everywhere and the detection-aware pitch-control spread never applies.
- 25 fps instead of 10: `meta.fps` is 25; use step=5 for 5 frames a second.
- No SkillCorner event layer: team possessions are runs of live-ball frames
  with the same value of the tracking feed's ball-possession flag, and shots
  come from the official event data (`dfl_shots`). There are no phases of
  play.
- Ids are strings in the source; players and teams get small integer ids
  (home team 1, away team 2) and the originals are kept in `meta.raw`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .data import MatchMeta, Player

FPS = 25.0
OPEN_MATCHES = ["J03WMX", "J03WN1", "J03WPY", "J03WOH", "J03WQQ", "J03WOY", "J03WR9"]

# kloppy position names -> the SkillCorner position groups used for vmax floors
_GROUPS = {
    "Goalkeeper": "Goalkeeper",
    "Center Back": "Central Defender", "Left Center Back": "Central Defender",
    "Right Center Back": "Central Defender",
    "Left Back": "Full Back", "Right Back": "Full Back",
    "Left Wing Back": "Full Back", "Right Wing Back": "Full Back",
    "Defensive Midfield": "Midfield", "Left Defensive Midfield": "Midfield",
    "Right Defensive Midfield": "Midfield", "Central Midfield": "Midfield",
    "Left Central Midfield": "Midfield", "Right Central Midfield": "Midfield",
    "Center Attacking Midfield": "Midfield", "Left Midfield": "Midfield", "Right Midfield": "Midfield",
    "Left Wing": "Wide Attacker", "Right Wing": "Wide Attacker",
    "Left Forward": "Wide Attacker", "Right Forward": "Wide Attacker",
    "Striker": "Center Forward", "Center Forward": "Center Forward",
    "Left Striker": "Center Forward", "Right Striker": "Center Forward",
}


def _kloppy(cache_dir: Optional[str | Path]):
    try:
        from kloppy import sportec
        from kloppy.config import set_config
    except ImportError as e:  # pragma: no cover
        raise ImportError("dsv.dfl needs kloppy: pip install kloppy") from e
    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        set_config("cache", str(cache_dir))
    return sportec


def load_dfl_match(match_id: str, cache_dir: Optional[str | Path] = None):
    """Tracking and events for one open DFL match.

    Returns (meta, players_df, ball_df, events_df) in the formats of
    `dsv.data.load_match_metadata` / `load_tracking`, with raw (not yet
    direction-normalized) coordinates in metres, origin at the centre spot.
    ball_df has an extra `alive` column. events_df: one row per event with
    period, time_s (event clock, see `sync_offsets`), team_id (1 home,
    2 away), player_id, event_type, result, x, y.
    """
    sportec = _kloppy(cache_dir)
    tr = sportec.load_open_tracking_data(match_id, coordinates="secondspectrum")
    ev = sportec.load_open_event_data(match_id, coordinates="secondspectrum")
    md = tr.metadata
    home, away = [t for t in md.teams if t.ground.name == "HOME"][0], [t for t in md.teams if t.ground.name == "AWAY"][0]
    team_int = {home.team_id: 1, away.team_id: 2}
    pid_int, players = {}, {}
    for team in (home, away):
        for p in team.players:
            i = len(pid_int) + 1
            pid_int[p.player_id] = i
            pos = str(p.starting_position) if p.starting_position is not None else None
            grp = _GROUPS.get(pos)
            players[i] = Player(player_id=i, name=p.full_name or p.player_id, team_id=team_int[team.team_id],
                                is_home=team is home, number=p.jersey_no,
                                role="GK" if grp == "Goalkeeper" else None, position_group=grp,
                                start_s=0.0, end_s=float("inf"))

    df = tr.to_df(engine="pandas")
    df["time_s"] = df["timestamp"].dt.total_seconds()
    ball = pd.DataFrame({
        "frame": df["frame_id"].astype(int), "period": df["period_id"].astype(int), "time_s": df["time_s"],
        "bx": df["ball_x"], "by": df["ball_y"], "bz": df["ball_z"], "ball_detected": True,
        "possession_group": df["ball_owning_team_id"].map({home.team_id: "home team", away.team_id: "away team"}),
        "alive": df["ball_state"].eq("alive"),
    })
    long = []
    for raw_id, i in pid_int.items():
        if f"{raw_id}_x" not in df.columns:
            continue
        part = pd.DataFrame({"frame": ball["frame"], "period": ball["period"], "time_s": ball["time_s"],
                             "player_id": i, "x": df[f"{raw_id}_x"], "y": df[f"{raw_id}_y"]})
        long.append(part.dropna(subset=["x", "y"]))
    pl = pd.concat(long, ignore_index=True)
    pl["team_id"] = pl["player_id"].map(lambda i: players[i].team_id)
    pl["is_home"] = pl["team_id"] == 1
    pl["is_detected"] = True
    pl = pl[["frame", "period", "time_s", "player_id", "team_id", "is_home", "x", "y", "is_detected"]]

    periods = {int(p.id): {"start_frame": int(ball.loc[ball["period"] == p.id, "frame"].min()),
                           "end_frame": int(ball.loc[ball["period"] == p.id, "frame"].max()),
                           "duration_minutes": None} for p in md.periods}
    # home attacking direction per period, from where the home goalkeeper stands
    gk_home = [i for i, p in players.items() if p.is_home and p.role == "GK"]
    side = []
    for per in sorted(periods):
        gx = pl.loc[(pl["period"] == per) & pl["player_id"].isin(gk_home), "x"].mean()
        side.append("right_to_left" if gx > 0 else "left_to_right")
    meta = MatchMeta(match_id=match_id, home_team_id=1, away_team_id=2, home_name=home.name, away_name=away.name,
                     pitch_length=float(md.pitch_dimensions.pitch_length),
                     pitch_width=float(md.pitch_dimensions.pitch_width), fps=FPS, periods=periods,
                     home_side=side, players=players,
                     raw={"team_ids": {v: k for k, v in team_int.items()}, "player_ids": {v: k for k, v in pid_int.items()}})

    rows = []
    for e in ev.records:
        c = e.coordinates
        rows.append(dict(period=int(e.period.id), time_s=e.timestamp.total_seconds(),
                         team_id=team_int.get(e.team.team_id) if e.team is not None else None,
                         player_id=pid_int.get(e.player.player_id) if e.player is not None else None,
                         event_type=e.event_type.name, result=getattr(getattr(e, "result", None), "name", None),
                         x=getattr(c, "x", np.nan), y=getattr(c, "y", np.nan)))
    return meta, pl, ball, pd.DataFrame(rows)


def sync_offsets(events: pd.DataFrame, ball: pd.DataFrame, max_s: float = 20.0, step_s: float = 0.04) -> dict:
    """Per-period constant offset (s) to add to event times so the tracked
    ball is nearest the events' recorded locations: the offset minimizing the
    median ball-to-event distance over all located passes, shots, recoveries
    and fouls. The event clock in the open DFL files runs up to a few seconds
    off the tracking clock, and individual events carry about a second of
    manual timing jitter on top (see `dfl_shots`)."""
    e = events[events["event_type"].isin(["PASS", "SHOT", "RECOVERY", "FOUL_COMMITTED"]) & events["x"].notna()]
    out = {}
    for per, ee in e.groupby("period"):
        b = ball[ball["period"] == per].sort_values("time_s")
        bt, bx, by = b["time_s"].to_numpy(), b["bx"].to_numpy(), b["by"].to_numpy()
        t, x, y = ee["time_s"].to_numpy(), ee["x"].to_numpy(), ee["y"].to_numpy()
        best = (0.0, np.inf)
        for d in np.arange(-max_s, max_s + step_s / 2, step_s):
            j = np.clip(np.searchsorted(bt, t + d), 0, len(bt) - 1)
            md = float(np.median(np.hypot(bx[j] - x, by[j] - y)))
            if md < best[1]:
                best = (float(d), md)
        out[int(per)] = best[0]
    return out


def dfl_shots(events: pd.DataFrame, ball: pd.DataFrame, players: pd.DataFrame,
              window_s: float = 4.0, touch_m: float = 2.5, offsets: Optional[dict] = None) -> pd.DataFrame:
    """Shots placed on the tracking frame where they happened.

    After the per-period clock offset (`sync_offsets`), each shot is moved
    to the frame within +-`window_s` that minimizes ball-to-shot-location
    plus ball-to-shooter distance. `matched` is True when the shooter is
    within `touch_m` of the ball at that frame; `sync_dist` is the ball's
    distance to the recorded shot location there."""
    off = sync_offsets(events, ball) if offsets is None else offsets
    s = events[events["event_type"] == "SHOT"]
    pos = players.set_index(["frame", "player_id"])[["x", "y"]]
    out = []
    for _, r in s.iterrows():
        b = ball[ball["period"] == r["period"]]
        t = r["time_s"] + off.get(int(r["period"]), 0.0)
        w = b[(b["time_s"] >= t - window_s) & (b["time_s"] <= t + window_s)]
        if w.empty:
            continue
        d_loc = np.hypot(w["bx"] - r["x"], w["by"] - r["y"]).to_numpy()
        try:
            sp = pos.xs(int(r["player_id"]), level="player_id").reindex(w["frame"])
            d_pl = np.hypot(w["bx"].to_numpy() - sp["x"].to_numpy(), w["by"].to_numpy() - sp["y"].to_numpy())
        except (KeyError, TypeError, ValueError):
            d_pl = np.full(len(w), np.nan)
        score = d_loc + np.nan_to_num(d_pl, nan=50.0)
        j = int(np.argmin(score))
        out.append(dict(period=int(r["period"]), time_s=r["time_s"], team_id=r["team_id"], player_id=r["player_id"],
                        frame=int(w["frame"].iloc[j]), goal=r["result"] == "GOAL", x=r["x"], y=r["y"],
                        sync_dist=float(d_loc[j]), shooter_dist=float(d_pl[j]),
                        matched=bool(np.isfinite(d_pl[j]) and d_pl[j] <= touch_m)))
    return pd.DataFrame(out)


def dfl_possessions(ball: pd.DataFrame, meta: MatchMeta, shots: Optional[pd.DataFrame] = None,
                    min_flicker_s: float = 1.0, shot_tolerance_s: float = 2.0) -> pd.DataFrame:
    """Team possessions from the tracking feed's ball-possession flag.

    A possession is a run of live-ball frames held by one team. Spells of
    the other team shorter than `min_flicker_s` inside a run (duels, touches
    that do not settle) are absorbed unless that team shoots during the
    spell; a dead ball always ends a possession. Pass only shots that
    `dfl_shots` matched.
    A shot belongs to the shooting team's possession that contains its
    frame or ended at most `shot_tolerance_s` before it. Same columns as
    `dsv.threat.team_possessions` (phases are empty, xg is NaN)."""
    b = ball.sort_values("frame").reset_index(drop=True)
    team = b["possession_group"].map({"home team": 1, "away team": 2}).where(b["alive"])
    # runs of identical (team, alive) values
    key = team.fillna(0).astype(int).to_numpy()
    frame = b["frame"].to_numpy()
    brk = np.ones(len(b), bool)
    brk[1:] = (key[1:] != key[:-1]) | (np.diff(frame) != 1)
    rid = np.cumsum(brk) - 1
    runs = pd.DataFrame({"key": key, "frame": frame, "period": b["period"].to_numpy(), "rid": rid}).groupby("rid").agg(
        key=("key", "first"), start=("frame", "first"), end=("frame", "last"), period=("period", "first"))
    runs["dur"] = (runs["end"] - runs["start"] + 1) / meta.fps
    # absorb short spells of the other team between two runs of the same team
    rs = runs.to_dict("records")
    tol = int(round(shot_tolerance_s * meta.fps))
    shot_at = [] if shots is None else list(zip(shots["team_id"], shots["frame"]))

    def has_shot(r):
        return any(t == r["key"] and r["start"] <= f <= r["end"] + tol for t, f in shot_at)

    merged = []
    for r in rs:
        if r["key"] == 0:
            merged.append(dict(r)); continue
        if (len(merged) >= 2 and merged[-1]["key"] not in (0, r["key"]) and merged[-1]["dur"] < min_flicker_s
                and not has_shot(merged[-1]) and merged[-2]["key"] == r["key"] and merged[-2]["period"] == r["period"]
                and merged[-1]["start"] == merged[-2]["end"] + 1 and r["start"] == merged[-1]["end"] + 1):
            merged.pop()
            merged[-1]["end"] = r["end"]
            merged[-1]["dur"] = (merged[-1]["end"] - merged[-1]["start"] + 1) / meta.fps
        elif merged and merged[-1]["key"] == r["key"] and r["start"] == merged[-1]["end"] + 1:
            merged[-1]["end"] = r["end"]
            merged[-1]["dur"] = (merged[-1]["end"] - merged[-1]["start"] + 1) / meta.fps
        else:
            merged.append(dict(r))
    p = pd.DataFrame([r for r in merged if r["key"] != 0]).reset_index(drop=True)
    out = pd.DataFrame({
        "possession_id": np.arange(1, len(p) + 1), "match_id": meta.match_id, "team_id": p["key"].astype(int),
        "period": p["period"].astype(int), "frame_start": p["start"].astype(int), "frame_end": p["end"].astype(int),
        "n_player_possessions": np.nan, "phases": "", "first_phase": None,
        "shot": False, "goal": False, "n_shots": 0, "xg": np.nan, "first_shot_frame": np.nan,
    })
    out["unmatched_shot"] = False
    if shots is not None and len(shots):
        for _, s in shots.iterrows():
            m = (out["team_id"] == s["team_id"]) & (out["frame_start"] <= s["frame"]) & (out["frame_end"] + tol >= s["frame"])
            if not m.any():
                continue
            i = out.index[m][-1]
            out.loc[i, "shot"] = True
            out.loc[i, "goal"] = bool(out.loc[i, "goal"] or s["goal"])
            out.loc[i, "n_shots"] += 1
            f = min(int(s["frame"]), int(out.loc[i, "frame_end"]))
            out.loc[i, "first_shot_frame"] = np.nanmin([out.loc[i, "first_shot_frame"], f])
    out["attacking_home"] = out["team_id"] == 1
    out["duration_s"] = (out["frame_end"] - out["frame_start"]) / meta.fps
    return out.drop(columns=["unmatched_shot"])


def matched_shots(shots: pd.DataFrame, poss: pd.DataFrame, fps: float = FPS, shot_tolerance_s: float = 2.0) -> pd.Series:
    """True for each shot that `dfl_possessions` assigned to a possession."""
    tol = int(round(shot_tolerance_s * fps))
    return shots.apply(lambda s: bool(((poss["team_id"] == s["team_id"]) & (poss["frame_start"] <= s["frame"])
                                       & (poss["frame_end"] + tol >= s["frame"])).any()), axis=1)


def prepare_match(match_id: str, cache_dir: Optional[str | Path] = None):
    """Everything `dsv.threat.match_threat` needs for one open DFL match:
    (meta, players with velocities, ball, possessions, shots), with the home
    team attacking +x in both halves. Velocities use a 17-frame (0.68 s)
    Savitzky-Golay window, about the +-0.3 s used on the 10 fps SkillCorner
    data."""
    from .data import add_velocity, normalize_direction
    meta, pl, ball, ev = load_dfl_match(match_id, cache_dir)
    meta.raw["sync_offsets"] = sync_offsets(ev, ball)
    shots = dfl_shots(ev, ball, pl, offsets=meta.raw["sync_offsets"])
    poss = dfl_possessions(ball, meta, shots[shots["matched"]])
    pl, ball = normalize_direction(pl, ball, meta)
    pl = add_velocity(pl, fps=meta.fps, window=17)
    return meta, pl, ball, poss, shots
