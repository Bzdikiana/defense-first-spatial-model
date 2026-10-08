import numpy as np
import pandas as pd
import pytest

from dsv import data as D


def make_meta(home_side=("right_to_left", "left_to_right")):
    players = {
        1: D.Player(1, "Home CB", 10, True, 4, "LCB", "Central Defender", 0.0, float("inf")),
        2: D.Player(2, "Away GK", 20, False, 1, "GK", "Other", 0.0, float("inf")),
    }
    return D.MatchMeta(
        match_id=1, home_team_id=10, away_team_id=20, home_name="H", away_name="A",
        pitch_length=105.0, pitch_width=68.0, fps=10.0,
        periods={1: {"start_frame": 0, "end_frame": 99, "duration_minutes": 0.2},
                 2: {"start_frame": 100, "end_frame": 199, "duration_minutes": 0.2}},
        home_side=list(home_side), players=players,
    )


def straight_run(n=40, v=(3.0, 4.0), detected=None, period=1, pid=1):
    t = np.arange(n) / 10.0
    return pd.DataFrame({
        "frame": np.arange(n) + (0 if period == 1 else 100),
        "period": period, "time_s": t, "player_id": pid, "team_id": 10, "is_home": True,
        "x": v[0] * t, "y": v[1] * t,
        "is_detected": np.ones(n, bool) if detected is None else detected,
    })


def test_velocity_recovers_constant_speed():
    out = D.add_velocity(straight_run())
    assert np.allclose(out["speed"], 5.0)
    assert np.allclose(out["vx"], 3.0) and np.allclose(out["vy"], 4.0)
    assert np.allclose(out["speed_obs"], 5.0)


def test_speed_obs_only_on_long_detected_runs():
    det = np.ones(40, bool)
    det[10:20] = False   # extrapolated stretch
    det[30:34] = False   # leaves a 4-frame detected run at 34..39 (< window 7) ...
    out = D.add_velocity(straight_run(detected=det)).sort_values("frame")
    so = out["speed_obs"].to_numpy()
    assert np.isnan(so[10:20]).all() and np.isnan(so[30:34]).all()
    assert np.isnan(so[34:]).all()          # ... too short to smooth: NaN
    assert np.allclose(so[:10], 5.0) and np.allclose(so[20:30], 5.0)
    assert out["speed"].notna().all()       # pitch-control speed exists everywhere


def test_v95_uses_detected_runs_only():
    det = np.ones(40, bool)
    det[15:30] = False
    df = straight_run(detected=det)
    df.loc[15:29, "x"] += np.linspace(0, 30, 15)   # a wild extrapolated jump
    v95 = D.compute_v95(D.add_velocity(df), make_meta(), by_period=False)
    assert v95.loc[0, "v95"] == pytest.approx(5.0, abs=1e-6)


def test_normalize_direction_flips_only_right_to_left_periods():
    meta = make_meta()
    pl = pd.concat([straight_run(period=1), straight_run(period=2)], ignore_index=True)
    ball = pl[["frame", "period", "time_s"]].assign(bx=pl["x"], by=pl["y"])
    p2, b2 = D.normalize_direction(pl, ball, meta)
    p1m, p2m = pl["period"] == 1, pl["period"] == 2
    assert np.allclose(p2.loc[p1m, "x"], -pl.loc[p1m, "x"])
    assert np.allclose(p2.loc[p2m, "x"], pl.loc[p2m, "x"])
    assert np.allclose(b2.loc[p1m, "by"], -ball.loc[p1m, "by"])


def test_orient_events_rotates_away_team_only():
    ev = pd.DataFrame({"team_id": [10, 20], "x_start": [30.0, 30.0], "y_start": [5.0, 5.0],
                       "x_end": [40.0, 40.0], "y_end": [-2.0, -2.0]})
    out = D.orient_events(ev, make_meta())
    assert out["xh_start"].tolist() == [30.0, -30.0]
    assert out["yh_end"].tolist() == [-2.0, 2.0]


def test_player_group_splits_out_goalkeepers():
    meta = make_meta()
    assert D.player_group(meta, 2) == "Goalkeeper"
    assert D.player_group(meta, 1) == "Central Defender"


def test_ball_zone_is_relative_to_own_goal():
    meta = make_meta()
    # home CB (defends -x) and away GK (defends +x), ball deep at x = +45
    pl = pd.DataFrame({"frame": [0, 0], "period": 1, "player_id": [1, 2],
                       "is_home": [True, False], "is_detected": [False, True]})
    ball_home_attacking = pd.DataFrame({"frame": [0], "bx": [45.0], "possession_group": ["home team"]})
    z = D.detection_by_ball_zone(pl, ball_home_attacking, meta).set_index("grp")
    # only the away team is defending; for them the ball is in their own third
    assert list(z.index) == ["Goalkeeper"]
    assert z.loc["Goalkeeper", "ball_zone"] == "own third"
    assert z.loc["Goalkeeper", "detection_rate"] == 1.0
