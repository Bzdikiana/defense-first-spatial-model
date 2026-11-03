import numpy as np
import pandas as pd
import pytest

from dsv import dfl
from dsv.data import MatchMeta


# --------------------------------------------------------------------------- #
# DFL loader pieces that need no download
# --------------------------------------------------------------------------- #
def dfl_meta():
    return MatchMeta(match_id="X", home_team_id=1, away_team_id=2, home_name="H", away_name="A",
                     pitch_length=105, pitch_width=68, fps=25.0, periods={1: {}}, home_side=["left_to_right"],
                     players={})


def ball_from(owners, alive=None):
    n = len(owners)
    return pd.DataFrame({"frame": np.arange(n), "period": 1, "time_s": np.arange(n) / 25.0,
                         "bx": 0.0, "by": 0.0, "possession_group": owners,
                         "alive": np.ones(n, bool) if alive is None else alive})


def test_dfl_possessions_absorb_short_flickers_but_not_shots():
    owners = ["home team"] * 50 + ["away team"] * 10 + ["home team"] * 50 + ["away team"] * 60
    b = ball_from(owners)
    p = dfl.dfl_possessions(b, dfl_meta())
    assert p["team_id"].tolist() == [1, 2]                    # the 0.4 s away spell is absorbed
    assert p.loc[0, "frame_end"] == 109
    shot = pd.DataFrame({"team_id": [2], "frame": [55], "goal": [False]})
    p = dfl.dfl_possessions(b, dfl_meta(), shot)
    assert p["team_id"].tolist() == [1, 2, 1, 2]              # a spell with a shot is kept
    assert p.loc[1, "shot"] and p.loc[1, "first_shot_frame"] == 55


def test_dfl_possessions_dead_ball_ends_possession():
    owners = ["home team"] * 100
    alive = np.r_[np.ones(40, bool), np.zeros(20, bool), np.ones(40, bool)]
    p = dfl.dfl_possessions(ball_from(owners, alive), dfl_meta())
    assert len(p) == 2 and p["frame_start"].tolist() == [0, 60]


def test_sync_offset_recovers_a_clock_shift():
    t = np.arange(0, 60, 0.04)
    ball = pd.DataFrame({"frame": np.arange(len(t)), "period": 1, "time_s": t,
                         "bx": 30 * np.sin(t / 3), "by": 20 * np.cos(t / 5)})
    ev_t = np.arange(5, 55, 1.0)
    true_shift = 1.4
    events = pd.DataFrame({"period": 1, "time_s": ev_t - true_shift, "event_type": "PASS",
                           "x": 30 * np.sin(ev_t / 3), "y": 20 * np.cos(ev_t / 5)})
    assert dfl.sync_offsets(events, ball)[1] == pytest.approx(true_shift, abs=0.05)
