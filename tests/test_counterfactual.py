import numpy as np
import pandas as pd
import pytest

from dsv import counterfactual as CF, pitchcontrol as PC


class Meta:
    match_id, home_team_id, away_team_id = 7, 10, 20
    pitch_length, pitch_width, fps = 105.0, 68.0, 10.0


def state(pos, home, ball=(0.0, 0.0), frame=0, vel=None):
    pos = np.asarray(pos, float)
    n = len(pos)
    return PC.FrameState(frame, 1, np.arange(n) + 1, np.asarray(home, bool), np.zeros(n, bool), np.ones(n, bool),
                         pos, np.zeros((n, 2)) if vel is None else np.asarray(vel, float), np.full(n, 5.0),
                         np.asarray(ball, float), None)


def test_hold_moves_with_the_teammates_and_drop_retreats_toward_own_goal():
    # away defenders (ids 2-4) defend the +x goal; presser is id 2
    home = [True, False, False, False]
    s0 = state([[0, 0], [10, 0], [20, 5], [20, -5]], home, frame=0)
    s1 = state([[5, 0], [3, 2], [24, 5], [24, -5]], home, frame=30,
               vel=[[0, 0], [0, 0], [1.0, 0], [1.0, 0]])
    pos, vel = CF.hold_position(s0, s1, 2, defending_home=False)
    assert pos == pytest.approx([14.0, 0.0])          # 10 + the teammates' 4 m shift
    assert vel == pytest.approx([1.0, 0.0])
    dpos, _ = CF.drop_position(s0, s1, 2, False, 105.0, fps=10.0, speed=3.0, max_m=5.0)
    assert dpos[0] > pos[0]                            # toward the +x goal
    assert np.hypot(*(dpos - pos)) == pytest.approx(5.0)
    moved = CF.move_player(s1, 2, pos, vel)
    assert moved.pos[1] == pytest.approx(pos) and s1.pos[1] == pytest.approx([3, 2])


def test_shape_value_cover_rises_when_a_defender_stands_goal_side():
    home = [True, False]
    grid = PC.make_grid(105, 68, 3.0)
    open_ = CF.shape_value(state([[30, 0], [-40, 30]], home, ball=(30, 0)), True, grid, Meta())
    covered = CF.shape_value(state([[30, 0], [42, 0]], home, ball=(30, 0)), True, grid, Meta())
    assert covered["cover"] > open_["cover"]
    assert covered["def_goal_side"] == 1 and open_["def_goal_side"] == 0
    assert covered["rd"] == pytest.approx(open_["rd"])  # same ball, same reachable danger


def test_horizon_value_and_frame_targets():
    v = CF.horizon_value(np.array([0.2, 0.2, 0.2]), np.array([True, False, False]), np.array([True, False, True]))
    assert v.tolist() == [1.0, 0.0, 0.2]
    # frames after a possession's first shot never appear in the frame tables
    frames = pd.DataFrame({"match_id": 1, "possession_id": [1, 1, 1, 2, 2],
                           "frame": [0, 30, 60, 90, 100]})
    poss = pd.DataFrame({"match_id": [1, 1], "possession_id": [1, 2], "frame_end": [60, 140],
                         "first_shot_frame": [np.nan, 125]})
    t = CF.frame_horizon_targets(frames, poss, np.array([.1, .2, .3, .4, .5]), horizon_s=3.0, fps=10.0)
    # frame 0 -> value at frame 30; frame 30 -> frame 60; frame 60 -> possession over by 90: 0;
    # frame 90 -> frame 120 is in play but missing from the table: NaN; frame 100 -> shot at 125: 1
    assert t.iloc[:3].tolist() == pytest.approx([0.2, 0.3, 0.0])
    assert np.isnan(t.iloc[3]) and t.iloc[4] == 1.0
