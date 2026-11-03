import numpy as np
import pandas as pd
import pytest

from dsv import attribution as A, pitchcontrol as PC, threat as T


class Meta:
    match_id, home_team_id, away_team_id = 7, 10, 20
    pitch_length, pitch_width, fps = 105.0, 68.0, 10.0


def state(pos, home, ball=(0.0, 0.0)):
    pos = np.asarray(pos, float)
    n = len(pos)
    return PC.FrameState(0, 1, np.arange(n) + 1, np.asarray(home, bool), np.zeros(n, bool), np.ones(n, bool),
                         pos, np.zeros((n, 2)), np.full(n, 5.0), np.asarray(ball, float), None)


GRID = PC.make_grid(105, 68, 3.0)


def random_state(seed=0, ball=(25, 5)):
    rng = np.random.default_rng(seed)
    return state(rng.uniform([-50, -32], [50, 32], (22, 2)), [True] * 11 + [False] * 11, ball=ball)


def test_shares_add_up_to_cover():
    s = random_state()
    flat = {"flat": lambda c, L, W: np.full(len(c), 0.05)}
    sh = A.defender_shares(s, True, GRID, Meta(), values=flat)
    r = T.frame_threat(s, True, GRID, Meta())
    assert len(sh) == 11 and (sh["share_xt"] >= 0).all()
    assert sh["share_xt"].sum() == pytest.approx(r["cover"], abs=0.01)
    assert 0 < sh["share_flat"].sum() <= 1


def test_blame_sums_to_the_drop_and_is_nonnegative():
    s0 = pd.DataFrame({"player_id": [1, 2, 3], "share_xt": [0.3, 0.2, 0.1]})
    s1 = pd.DataFrame({"player_id": [1, 2, 3], "share_xt": [0.05, 0.25, 0.1]})
    b = A.blame(s0, s1)
    assert b.sum() == pytest.approx(0.2)            # cover fell from 0.6 to 0.4
    assert (b >= 0).all() and b.idxmax() == 1
    assert A.blame(s1, s0).sum() == 0               # cover rose: no blame


def test_leave_one_out_credits_the_defender_in_the_way():
    # attacker on the ball at x=30 attacking +x; one defender goal-side, one far away
    s = state([[30, 0], [42, 0], [-45, 30]], [True, False, False], ball=(30, 0))
    loo = A.leave_one_out(s, True, GRID, Meta()).set_index("player_id")
    assert loo.loc[2, "loo_xt"] > 0.05
    assert loo.loc[2, "loo_xt"] > 10 * abs(loo.loc[3, "loo_xt"])
