"""
dsv.data
========
Data layer for the Defensive Space Value project (SkillCorner open data).

Responsibilities
----------------
1. Fetch a SkillCorner open-data match (match.json + the git-LFS tracking file).
2. Parse metadata (teams, players, roles, pitch, periods, play direction).
3. Parse the extrapolated tracking JSONL into tidy long DataFrames.
4. Derive kinematics: smoothed positions, velocity, acceleration, speed, v95.
5. Normalize play direction.
6. Coverage-audit helpers built around the per-position `is_detected` flag.

Key fact about SkillCorner open data
------------------------------------
The public tracking file is `<match>_tracking_extrapolated.jsonl`. Every
in-play frame carries all 22 on-pitch players, but each player position has an
`is_detected` flag. When False, the position is SkillCorner's extrapolation,
not an observation. So the data-quality question is not "are players missing"
(they are not) but "what fraction of positions are observed vs extrapolated,
and where". That is what the coverage audit measures.
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

RAW_BASE = "https://raw.githubusercontent.com/SkillCorner/opendata/master/data/matches"
# git-LFS objects are served from the media endpoint, not raw.githubusercontent.com
MEDIA_BASE = "https://media.githubusercontent.com/media/SkillCorner/opendata/master/data/matches"
FPS = 10.0  # SkillCorner open data is 10 Hz (verified from timestamps)


# --------------------------------------------------------------------------- #
# Fetching
# --------------------------------------------------------------------------- #
def list_open_matches(data_dir: str | Path) -> list[int]:
    """Match ids in the SkillCorner open-data index (data/matches.json).

    Downloads the index on first use. Returns ids oldest first.
    """
    idx = Path(data_dir) / "matches.json"
    if not idx.exists():
        idx.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["curl", "-sS", "-f", RAW_BASE.rsplit("/", 1)[0] + "/matches.json", "-o", str(idx)],
            check=True,
        )
    with open(idx, encoding="utf-8") as fh:
        ms = json.load(fh)
    return [m["id"] for m in sorted(ms, key=lambda m: m["date_time"])]


def download_open_match(match_id: int | str, data_dir: str | Path) -> Path:
    """Download match.json, dynamic events, phases of play and the (LFS)
    extrapolated tracking file for a match.

    Uses curl, which is the most reliable option behind an outbound proxy and is
    usually available locally too. The tracking file is git-LFS backed, so it is
    pulled from the media endpoint. Returns the match directory.
    Alternative for local use: `git clone` the repo then `git lfs pull`.
    """
    match_id = str(match_id)
    out = Path(data_dir) / "matches" / match_id
    out.mkdir(parents=True, exist_ok=True)
    targets = {
        f"{match_id}_match.json": f"{RAW_BASE}/{match_id}/{match_id}_match.json",
        f"{match_id}_dynamic_events.csv": f"{RAW_BASE}/{match_id}/{match_id}_dynamic_events.csv",
        f"{match_id}_phases_of_play.csv": f"{RAW_BASE}/{match_id}/{match_id}_phases_of_play.csv",
        f"{match_id}_tracking_extrapolated.jsonl": f"{MEDIA_BASE}/{match_id}/{match_id}_tracking_extrapolated.jsonl",
    }
    for fname, url in targets.items():
        dest = out / fname
        if dest.exists() and dest.stat().st_size > 1000:
            continue
        subprocess.run(["curl", "-sS", "-f", url, "-o", str(dest)], check=True)
    return out


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #
@dataclass
class Player:
    player_id: int
    name: str
    team_id: int
    is_home: bool
    number: Optional[int]
    role: Optional[str]            # e.g. "LCB"
    position_group: Optional[str]  # e.g. "Central Defender"
    start_s: float                 # seconds into the match the player came on
    end_s: float                   # seconds the player left (inf if to the end)


@dataclass
class MatchMeta:
    match_id: int
    home_team_id: int
    away_team_id: int
    home_name: str
    away_name: str
    pitch_length: float
    pitch_width: float
    fps: float
    periods: dict               # period -> {start_frame, end_frame, duration_minutes}
    home_side: list             # per-period attacking direction for the home team
    players: dict               # player_id -> Player
    raw: dict = field(repr=False, default_factory=dict)

    def team_name(self, team_id: int) -> str:
        return self.home_name if team_id == self.home_team_id else self.away_name


def _clock_to_seconds(s: Optional[str]) -> float:
    if not s:
        return 0.0
    h, m, sec = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(sec)


def load_match_metadata(match_dir: str | Path, match_id: int | str) -> MatchMeta:
    with open(Path(match_dir) / f"{match_id}_match.json", encoding="utf-8") as fh:
        md = json.load(fh)
    home_id = md["home_team"]["id"]
    away_id = md["away_team"]["id"]
    players: dict[int, Player] = {}
    for p in md["players"]:
        role = (p.get("player_role") or {}) or {}
        end = p.get("end_time")
        players[p["id"]] = Player(
            player_id=p["id"],
            name=f'{p.get("first_name","")} {p.get("last_name","")}'.strip(),
            team_id=p["team_id"],
            is_home=(p["team_id"] == home_id),
            number=p.get("number"),
            role=role.get("acronym"),
            position_group=role.get("position_group"),
            start_s=_clock_to_seconds(p.get("start_time")),
            end_s=_clock_to_seconds(end) if end else float("inf"),
        )
    periods = {
        mp["period"]: {
            "start_frame": mp["start_frame"],
            "end_frame": mp["end_frame"],
            "duration_minutes": mp.get("duration_minutes"),
        }
        for mp in md["match_periods"]
    }
    return MatchMeta(
        match_id=int(md["id"]),
        home_team_id=home_id,
        away_team_id=away_id,
        home_name=md["home_team"]["name"],
        away_name=md["away_team"]["name"],
        pitch_length=float(md["pitch_length"]),
        pitch_width=float(md["pitch_width"]),
        fps=FPS,
        periods=periods,
        home_side=md.get("home_team_side", []),
        players=players,
        raw=md,
    )


# --------------------------------------------------------------------------- #
# Tracking
# --------------------------------------------------------------------------- #
def load_tracking(match_dir: str | Path, match_id: int | str, meta: MatchMeta):
    """Parse the extrapolated tracking JSONL into tidy DataFrames.

    Returns (players_df, ball_df).
    players_df columns: frame, period, time_s, player_id, team_id, is_home,
                        x, y, is_detected
    ball_df columns:    frame, period, time_s, bx, by, bz, ball_detected,
                        possession_group  ("home team" / "away team" / None)
    `time_s` is seconds from the start of that period.
    """
    path = Path(match_dir) / f"{match_id}_tracking_extrapolated.jsonl"
    start_frame = {per: info["start_frame"] for per, info in meta.periods.items()}

    prows, brows = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            fr = json.loads(line)
            per = fr.get("period")
            pdl = fr.get("player_data")
            if per is None or not pdl:
                continue  # pre-match, half-time, dead pre-kick frames
            frame = fr["frame"]
            time_s = (frame - start_frame.get(per, 0)) / meta.fps
            b = fr.get("ball_data") or {}
            poss = fr.get("possession") or {}
            brows.append(
                (frame, per, time_s, b.get("x"), b.get("y"), b.get("z"),
                 bool(b.get("is_detected")), poss.get("group"))
            )
            for e in pdl:
                pid = e["player_id"]
                pl = meta.players.get(pid)
                prows.append(
                    (frame, per, time_s, pid,
                     pl.team_id if pl else None,
                     pl.is_home if pl else None,
                     e.get("x"), e.get("y"), bool(e.get("is_detected")))
                )

    players_df = pd.DataFrame(
        prows,
        columns=["frame", "period", "time_s", "player_id", "team_id",
                 "is_home", "x", "y", "is_detected"],
    )
    ball_df = pd.DataFrame(
        brows,
        columns=["frame", "period", "time_s", "bx", "by", "bz",
                 "ball_detected", "possession_group"],
    )
    return players_df, ball_df


# --------------------------------------------------------------------------- #
# Kinematics
# --------------------------------------------------------------------------- #
def _smooth_diff(x, y, dt, window, poly, max_speed, smooth=True):
    """Savitzky-Golay smooth one contiguous track and differentiate it.

    Returns (vx, vy, speed, ax, ay); speed is clipped at `max_speed`.
    """
    if smooth and len(x) >= window:
        x = savgol_filter(x, window, poly)
        y = savgol_filter(y, window, poly)
    vx = np.gradient(x, dt)
    vy = np.gradient(y, dt)
    sp = np.clip(np.hypot(vx, vy), 0, max_speed)
    return vx, vy, sp, np.gradient(vx, dt), np.gradient(vy, dt)


def _run_bounds(breaks: np.ndarray, n: int):
    """(start, end) index pairs of the runs delimited by boolean `breaks`
    (True where a new run starts)."""
    starts = np.flatnonzero(breaks)
    ends = np.append(starts[1:], n)
    return zip(starts, ends)


def add_velocity(
    players_df: pd.DataFrame,
    fps: float = FPS,
    smooth: bool = True,
    window: int = 7,
    poly: int = 2,
    max_speed: float = 12.0,
) -> pd.DataFrame:
    """Add smoothed vx, vy, speed, ax, ay per player within each period, plus
    `speed_obs`, a speed computed from detected positions only.

    Savitzky-Golay smooths each contiguous on-pitch segment; velocity and
    acceleration are finite differences of the smoothed path. Speeds above
    `max_speed` (m/s) are treated as tracking artifacts and clipped.

    `vx, vy, speed, ax, ay` use every frame (pitch control needs a velocity
    for all 22 players at all times). `speed_obs` is recomputed on runs of
    consecutive *detected* frames only, so the smoothing window never reaches
    into SkillCorner's extrapolated positions; it is NaN on extrapolated frames
    and on detected runs shorter than the smoothing window. Use it for v95.
    """
    df = players_df.sort_values(["player_id", "period", "frame"]).reset_index(drop=True)
    w = window if window % 2 == 1 else window + 1
    dt = 1.0 / fps
    n = len(df)
    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    det = df["is_detected"].to_numpy(dtype=bool)
    pid = df["player_id"].to_numpy()
    per = df["period"].to_numpy()
    frame = df["frame"].to_numpy()

    # new segment whenever player or period changes or a frame is skipped
    seg_break = np.ones(n, dtype=bool)
    seg_break[1:] = (pid[1:] != pid[:-1]) | (per[1:] != per[:-1]) | (np.diff(frame) != 1)
    out = np.full((n, 5), np.nan)
    for s0, s1 in _run_bounds(seg_break, n):
        if s1 - s0 < 3:
            continue
        out[s0:s1] = np.column_stack(
            _smooth_diff(x[s0:s1], y[s0:s1], dt, w, poly, max_speed, smooth)
        )

    # detected-only runs: additionally break wherever detection toggles
    det_break = seg_break.copy()
    det_break[1:] |= det[1:] != det[:-1]
    speed_obs = np.full(n, np.nan)
    for s0, s1 in _run_bounds(det_break, n):
        if not det[s0] or s1 - s0 < w:
            continue
        speed_obs[s0:s1] = _smooth_diff(x[s0:s1], y[s0:s1], dt, w, poly, max_speed, smooth)[2]

    df[["vx", "vy", "speed", "ax", "ay"]] = out
    df["speed_obs"] = speed_obs
    return df


def compute_v95(
    players_df: pd.DataFrame,
    meta: MatchMeta,
    detected_only: bool = True,
    by_period: bool = True,
) -> pd.DataFrame:
    """95th-percentile running speed per player (the robust 'top speed', v95).

    By default uses `speed_obs` (detected runs only, see `add_velocity`) so
    neither extrapolated positions nor smoothing across them inflate or deflate
    the estimate. Note this still likely underestimates true top speed, since
    sprints happen off the ball where detection is low. Returns a tidy table
    with name, role, position group and the number of frames used.
    """
    col = "speed_obs" if detected_only else "speed"
    df = players_df.dropna(subset=[col])
    keys = ["player_id"] + (["period"] if by_period else [])
    v95 = (df.groupby(keys)[col]
             .agg(v95=lambda s: s.quantile(0.95), n_frames="size")
             .reset_index())
    v95["name"] = v95["player_id"].map(lambda i: meta.players[i].name if i in meta.players else str(i))
    v95["role"] = v95["player_id"].map(lambda i: meta.players[i].role if i in meta.players else None)
    v95["position_group"] = v95["player_id"].map(
        lambda i: meta.players[i].position_group if i in meta.players else None
    )
    v95["group"] = v95["player_id"].map(lambda i: player_group(meta, i))
    v95["match_id"] = meta.match_id
    return v95


# --------------------------------------------------------------------------- #
# Direction normalization
# --------------------------------------------------------------------------- #
def normalize_direction(
    players_df: pd.DataFrame, ball_df: pd.DataFrame, meta: MatchMeta
):
    """Flip coordinates so the HOME team always attacks +x, in both periods.

    `home_side[period-1]` gives the home attacking direction for that period
    ('left_to_right' or 'right_to_left'). Where it is right_to_left we rotate
    180 degrees (x,y -> -x,-y) so home consistently attacks +x. Per-possession
    orientation (so the attacking team attacks +x) is applied later when the
    threat metric is computed; this just removes the half-time switch.
    """
    pdf, bdf = players_df.copy(), ball_df.copy()
    for per in meta.periods:
        side = meta.home_side[per - 1] if per - 1 < len(meta.home_side) else "left_to_right"
        if side == "right_to_left":
            for d, cols in ((pdf, ("x", "y")), (bdf, ("bx", "by"))):
                m = d["period"] == per
                d.loc[m, cols[0]] = -d.loc[m, cols[0]]
                d.loc[m, cols[1]] = -d.loc[m, cols[1]]
    return pdf, bdf


# --------------------------------------------------------------------------- #
# Coverage audit
# --------------------------------------------------------------------------- #
def infer_fps(ball_df: pd.DataFrame) -> float:
    t = np.sort(ball_df["time_s"].to_numpy())
    dt = np.diff(t)
    dt = dt[(dt > 0) & (dt < 1)]
    return float(round(1.0 / np.median(dt), 1)) if len(dt) else FPS


def detection_overall(players_df: pd.DataFrame) -> float:
    return float(players_df["is_detected"].mean())


def detection_by(players_df: pd.DataFrame, meta: MatchMeta, level: str) -> pd.DataFrame:
    """Detection rate grouped by 'period', 'group' (position group with GKs
    split out, see `player_group`), 'position_group', 'role', or 'player'."""
    df = players_df.copy()
    if level == "group":
        df["grp"] = df["player_id"].map(lambda i: player_group(meta, i))
    elif level in ("position_group", "role", "player"):
        attr = {"position_group": "position_group", "role": "role", "player": "name"}[level]
        df["grp"] = df["player_id"].map(
            lambda i: getattr(meta.players[i], attr if level != "player" else "name", None)
            if i in meta.players else None
        )
    elif level == "period":
        df["grp"] = df["period"]
    else:
        raise ValueError(level)
    out = (df.groupby("grp")["is_detected"]
             .agg(detection_rate="mean", n="size")
             .reset_index()
             .sort_values("detection_rate"))
    return out


def detection_vs_distance_to_ball(
    players_df: pd.DataFrame, ball_df: pd.DataFrame, bins=None
) -> pd.DataFrame:
    """Detection rate as a function of a player's distance to the ball.

    This is the number that matters most for pitch control: positions far from
    the ball are where detection falls off and extrapolation takes over.
    """
    if bins is None:
        bins = np.arange(0, 80, 5.0)
    b = ball_df.dropna(subset=["bx", "by"])[["frame", "bx", "by"]]
    df = players_df.dropna(subset=["x", "y"]).merge(b, on="frame", how="inner")
    df["dist"] = np.hypot(df["x"] - df["bx"], df["y"] - df["by"])
    df["bin"] = pd.cut(df["dist"], bins)
    out = (df.groupby("bin", observed=True)["is_detected"]
             .agg(detection_rate="mean", n="size").reset_index())
    out["dist_mid"] = out["bin"].apply(lambda iv: iv.mid)
    return out


def detection_grid(players_df: pd.DataFrame, meta: MatchMeta, nx=12, ny=8) -> np.ndarray:
    """Detection rate over a pitch grid in raw camera coordinates (origin at
    centre). Reveals static camera/pitch framing bias. Returns an (ny, nx) array
    oriented so row 0 is the bottom of the pitch."""
    df = players_df.dropna(subset=["x", "y"])
    hx, hy = meta.pitch_length / 2, meta.pitch_width / 2
    xi = np.clip(((df["x"] + hx) / meta.pitch_length * nx).astype(int), 0, nx - 1)
    yi = np.clip(((df["y"] + hy) / meta.pitch_width * ny).astype(int), 0, ny - 1)
    det = df["is_detected"].to_numpy()
    grid = np.full((ny, nx), np.nan)
    cnt = np.zeros((ny, nx))
    tot = np.zeros((ny, nx))
    for gx, gy, d in zip(xi.to_numpy(), yi.to_numpy(), det):
        tot[gy, gx] += 1
        cnt[gy, gx] += d
    mask = tot > 0
    grid[mask] = cnt[mask] / tot[mask]
    return grid


def detected_per_frame(players_df: pd.DataFrame) -> pd.Series:
    """How many of the 22 players are detected in each frame."""
    return players_df.groupby("frame")["is_detected"].sum()


def detection_over_time(players_df: pd.DataFrame, bin_minutes: float = 2.0) -> pd.DataFrame:
    df = players_df.copy()
    df["t_min"] = df["time_s"] / 60.0
    df["bin"] = (df["t_min"] // bin_minutes) * bin_minutes
    return (df.groupby(["period", "bin"])["is_detected"]
              .mean().rename("detection_rate").reset_index())


def player_group(meta: MatchMeta, player_id: int) -> Optional[str]:
    """Position group with goalkeepers split out. SkillCorner files GKs under
    position_group 'Other', alongside substitutes (role 'SUB')."""
    p = meta.players.get(player_id)
    if p is None:
        return None
    return "Goalkeeper" if p.role == "GK" else p.position_group


def detection_by_ball_zone(
    players_df: pd.DataFrame, ball_df: pd.DataFrame, meta: MatchMeta,
    out_of_possession: bool = True,
) -> pd.DataFrame:
    """Detection rate of each position group by where the ball is relative to
    that player's OWN goal (own / middle / opposition third).

    Expects direction-normalized data (home attacks +x, see
    `normalize_direction`). With `out_of_possession`, keeps only frames where
    the player's team is defending (the frames Threat Conceded is built on).
    The whole-match per-group rate is dominated by frames where the ball is far
    away; this is the rate that matters for defensive attribution.
    """
    hx = meta.pitch_length / 2
    b = ball_df.dropna(subset=["bx"])[["frame", "bx", "possession_group"]]
    df = players_df.merge(b, on="frame", how="inner")
    # distance from the ball to the player's own goal line along x
    own_goal_dist = np.where(df["is_home"], df["bx"] + hx, hx - df["bx"])
    third = meta.pitch_length / 3
    df["ball_zone"] = pd.cut(
        own_goal_dist, [-np.inf, third, 2 * third, np.inf],
        labels=["own third", "middle third", "opp third"],
    )
    if out_of_possession:
        defending = np.where(df["is_home"], "away team", "home team")
        df = df[df["possession_group"] == defending]
    df["grp"] = df["player_id"].map(lambda i: player_group(meta, i))
    return (df.groupby(["grp", "ball_zone"], observed=True)["is_detected"]
              .agg(detection_rate="mean", n="size").reset_index())


# --------------------------------------------------------------------------- #
# Dynamic events and phases of play
# --------------------------------------------------------------------------- #
def load_events(match_dir: str | Path, match_id: int | str) -> pd.DataFrame:
    """SkillCorner dynamic events (player_possession, passing_option,
    off_ball_run, on_ball_engagement), one row per event.

    `frame_start` / `frame_end` index the same frame numbering as the tracking
    file (verified by `event_alignment`: no offset). Event coordinates are
    rotated so that the event's own team attacks +x (`attacking_side` is that
    team's direction); use `orient_events` to put them in the tracking frame.
    Note the open data carries no shot xG: shots are `end_type == "shot"` on
    player_possession events.
    """
    return pd.read_csv(
        Path(match_dir) / f"{match_id}_dynamic_events.csv", low_memory=False
    )


def load_phases(match_dir: str | Path, match_id: int | str) -> pd.DataFrame:
    """SkillCorner phases of play: one row per in-possession phase (build_up,
    create, finish, direct, quick_break, transition, chaotic, set_play) with
    frame_start / frame_end, team in possession, and whether the team
    possession led to a shot or goal. Events link to it via `phase_index`."""
    ph = pd.read_csv(Path(match_dir) / f"{match_id}_phases_of_play.csv")
    return ph.rename(columns={"index": "phase_index"})


def orient_events(events: pd.DataFrame, meta: MatchMeta) -> pd.DataFrame:
    """Add x/y start/end columns in the direction-normalized tracking frame
    (home attacks +x, the output of `normalize_direction`): `xh_start`,
    `yh_start`, `xh_end`, `yh_end`.

    Event coords have the event team attacking +x, so home-team events are
    already in that frame and away-team events are rotated 180 degrees.
    """
    ev = events.copy()
    s = np.where(ev["team_id"] == meta.home_team_id, 1.0, -1.0)
    for a in ("start", "end"):
        ev[f"xh_{a}"] = s * ev[f"x_{a}"]
        ev[f"yh_{a}"] = s * ev[f"y_{a}"]
    return ev


def event_alignment(
    events: pd.DataFrame, players_df: pd.DataFrame, ball_df: pd.DataFrame,
    meta: MatchMeta, lags=range(-10, 11),
) -> dict:
    """Check event <-> tracking sync on direction-normalized tracking.

    1. Coordinates: median distance between each event's (oriented) start
       location and the tracking position of the event player at
       `frame_start`, per event type.
    2. Timing: for player_possession events, median player-to-ball distance at
       `frame_start + lag`; the minimizing lag is the frame offset.
    """
    ev = orient_events(events, meta)
    pos = players_df[["frame", "player_id", "x", "y"]]
    m = ev.merge(pos, left_on=["frame_start", "player_id"],
                 right_on=["frame", "player_id"], how="inner")
    m["err"] = np.hypot(m["xh_start"] - m["x"], m["yh_start"] - m["y"])
    coord_err = m.groupby("event_type")["err"].median().to_dict()

    pp = ev[ev["event_type"] == "player_possession"][["frame_start", "player_id"]]
    pidx = pos.set_index(["frame", "player_id"])[["x", "y"]]
    bidx = ball_df.set_index("frame")[["bx", "by"]]
    lag_dist = {}
    for k in lags:
        f = pp["frame_start"].to_numpy() + k
        pxy = pidx.reindex(list(zip(f, pp["player_id"]))).to_numpy()
        bxy = bidx.reindex(f).to_numpy()
        lag_dist[k] = float(np.nanmedian(np.hypot(*(pxy - bxy).T)))
    return {
        "n_events": len(events),
        "n_matched": len(m),
        "coord_err_median_m": coord_err,
        "possession_ball_dist_by_lag": lag_dist,
        "best_lag_frames": min(lag_dist, key=lag_dist.get),
    }


def passes_with_destination(
    events: pd.DataFrame, ball_df: pd.DataFrame, meta: MatchMeta, max_gap_frames: int = 50,
) -> pd.DataFrame:
    """Completed and failed passes with where the ball actually went.

    A pass is a player_possession ending in a pass with outcome successful or
    unsuccessful. Its destination is the tracked ball position at the start
    of the next player possession (either team), kept only if that starts in
    the same period within `max_gap_frames` (default 5 s) and both frames have
    tracking. Passes that go out of play, or whose next possession starts
    later, are dropped. Coordinates are the direction-normalized frame of
    `ball_df`. Adds: dest_x, dest_y, next_frame, passer_home, success,
    pass_length (ball at release to destination, m).
    """
    pp = events[events["event_type"] == "player_possession"].sort_values("frame_start")
    nxt = pp[["frame_start", "period"]].shift(-1)
    pp = pp.assign(next_frame=nxt["frame_start"], next_period=nxt["period"])
    ps = pp[(pp["end_type"] == "pass")
            & pp["pass_outcome"].isin(["successful", "unsuccessful"])
            & (pp["next_period"] == pp["period"])
            & (pp["next_frame"] - pp["frame_end"]).between(1, max_gap_frames)].copy()
    b = ball_df.set_index("frame")[["bx", "by"]]
    tracked = set(b.index)
    ps = ps[ps["frame_end"].isin(tracked) & ps["next_frame"].isin(tracked)].copy()
    ps["next_frame"] = ps["next_frame"].astype(int)
    ps[["dest_x", "dest_y"]] = b.reindex(ps["next_frame"]).to_numpy()
    rel = b.reindex(ps["frame_end"]).to_numpy()
    ps["pass_length"] = np.hypot(ps["dest_x"] - rel[:, 0], ps["dest_y"] - rel[:, 1])
    ps["passer_home"] = ps["team_id"] == meta.home_team_id
    ps["success"] = ps["pass_outcome"] == "successful"
    return ps.dropna(subset=["dest_x", "dest_y"])
