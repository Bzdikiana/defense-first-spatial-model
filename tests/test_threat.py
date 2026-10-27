from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from dsv import pitchcontrol as PC, threat as T


class Meta:
    match_id, home_team_id, away_team_id = 7, 10, 20
    pitch_length, pitch_width, fps = 105.0, 68.0, 10.0


def state(pos, home, ball=(0.0, 0.0), vel=None, detected=None):
    pos = np.asarray(pos, float)
    n = len(pos)
    return PC.FrameState(0, 1, np.arange(n) + 1, np.asarray(home, bool), np.zeros(n, bool),
                         np.ones(n, bool) if detected is None else np.asarray(detected, bool),
                         pos, np.zeros((n, 2)) if vel is None else np.asarray(vel, float),
                         np.full(n, 5.0), np.asarray(ball, float), None)


GRID = PC.make_grid(105, 68, 3.0)


def test_danger_weights_are_transition_times_xt():
    cells = GRID.cells
    w = T.danger_weights(np.array([30.0, 0.0]), cells, 105, 68)
    assert (w >= 0).all()
    assert w.sum() <= T.ThreatModel().xt.max() + 1e-9   # a weighted average of xT values


def test_threat_bounds_and_extremes():
    rng = np.random.default_rng(0)
    s = state(rng.uniform([-50, -32], [50, 32], (22, 2)), [True] * 11 + [False] * 11, ball=(20, 5))
    r = T.frame_threat(s, True, GRID, Meta())
    assert 0 <= r["tc"] <= r["rd"]
    assert 0 <= r["cover"] <= 1
    # only attackers on the pitch: the attack wins everything, TC = RD
    only_att = state([[20, 5], [40, -10]], [True, True], ball=(20, 5))
    r = T.frame_threat(only_att, True, GRID, Meta())
    assert r["tc"] == pytest.approx(r["rd"], rel=0.01)    # each spot stops at < 1% undecided
    # only defenders: the attack wins nothing
    only_def = state([[20, 5], [40, -10]], [False, False], ball=(20, 5))
    assert T.frame_threat(only_def, True, GRID, Meta())["tc"] == pytest.approx(0, abs=1e-6)


def test_defender_in_the_way_reduces_threat():
    att = [[30, 0]]
    open_play = state(att + [[-40, 30]], [True, False], ball=(30, 0))
    covered = state(att + [[40, 0]], [True, False], ball=(30, 0))
    assert (T.frame_threat(covered, True, GRID, Meta())["tc"]
            < T.frame_threat(open_play, True, GRID, Meta())["tc"])


def test_fast_integration_matches_reference():
    rng = np.random.default_rng(1)
    prm = PC.SKILLCORNER_OPEN_DATA
    s = state(rng.uniform([-50, -32], [50, 32], (22, 2)), [True] * 11 + [False] * 11,
              vel=rng.normal(0, 2, (22, 2)), detected=rng.random(22) > 0.4)
    cells = GRID.cells
    tau = PC._arrival_times(s.pos, s.vel, s.vmax, cells[:, None, :], prm.t_react)
    tb = np.linalg.norm(cells - s.ball, axis=-1) / prm.ball_speed
    lam = np.ascontiguousarray(np.broadcast_to(PC._rates(s.is_gk, prm), tau.shape))
    alpha = np.ascontiguousarray(np.broadcast_to(PC._alpha(s.detected, prm), tau.shape))
    ref = PC._integrate(tau, tb, lam, replace(prm, tol=1e-6), alpha=alpha)
    fast = PC._integrate_rows(tau, tb, lam, alpha, prm)
    team_ref, team_fast = ref[:, :11].sum(1), fast[:, :11].sum(1)
    assert np.abs(team_ref - team_fast).max() < 0.02


def events_table():
    rows = [  # (frame_start, frame_end, team, first, end_type, interruption_after, event_id)
        (0, 10, 10, True, "pass", None, "a"),
        (12, 30, 10, False, "shot", None, "b"),
        (40, 50, 20, True, "pass", None, "c"),
        (52, 60, 20, False, "shot", "goal_for", "d"),
    ]
    return pd.DataFrame([dict(event_type="player_possession", frame_start=a, frame_end=b, team_id=t, period=1,
                              first_player_possession_in_team_possession=f, end_type=e,
                              game_interruption_after=g, event_id=i, team_in_possession_phase_type="create")
                         for a, b, t, f, e, g, i in rows])


def test_team_possessions_and_xg_by_match():
    xg = pd.DataFrame({"match_id": [7, 7, 99], "event_id": ["b", "d", "b"], "xg": [0.1, 0.4, 0.9]})
    p = T.team_possessions(events_table(), Meta(), xg)
    assert p["team_id"].tolist() == [10, 20]
    assert p["frame_start"].tolist() == [0, 40] and p["frame_end"].tolist() == [30, 60]
    assert p["shot"].tolist() == [True, True] and p["goal"].tolist() == [False, True]
    assert p["xg"].tolist() == pytest.approx([0.1, 0.4])          # match 99's "b" ignored
    assert p["first_shot_frame"].tolist() == [30, 60]


def test_possession_summary_early_peak_excludes_the_shot_moment():
    p = T.team_possessions(events_table(), Meta())
    frames = pd.DataFrame({
        "match_id": 7, "possession_id": 1, "frame": [0, 2, 4, 6, 8, 10, 12, 20, 28, 30],
        "t_in_possession": np.arange(10) * 0.2, "tc": [.01, .01, .02, .01, .01, .01, .01, .01, .05, .09],
        "rd": .1, "cover": .5, "def_extrap_share": .2})
    s = T.possession_summary(frames, p).set_index("possession_id")
    assert s.loc[1, "peak_tc"] == pytest.approx(0.09)
    assert s.loc[1, "peak_tc_early"] == pytest.approx(0.02)   # frames <= 30 - 20 only


def test_comparison_measures():
    comps = T.Comparisons(values={"flat": lambda c, L, W: np.full(len(c), 0.05)},
                          transitions={"same": T.ThreatModel().transition})
    # home attacks +x; two away defenders goal-side of the ball, one behind it
    s = state([[20, 0], [35, 0], [45, 5], [0, 0]], [True, False, False, False], ball=(20, 0))
    r = T.frame_threat(s, True, GRID, Meta(), comparisons=comps)
    assert r["def_goal_side"] == 2
    assert 0 <= r["tc_no_access"] <= 1
    assert r["tc_same"] == pytest.approx(r["tc"], rel=1e-9)       # same transition -> same TC
    assert r["tc_flat"] <= 0.05 + 1e-9
    assert 0 < r["space_m2"] < 105 * 68
    only_att = state([[20, 5], [40, -10]], [True, True], ball=(20, 5))
    r = T.frame_threat(only_att, True, GRID, Meta(), comparisons=comps)
    assert r["tc_voronoi"] == pytest.approx(r["rd"])                 # attack nearest everywhere
