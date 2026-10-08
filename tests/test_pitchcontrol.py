import numpy as np
import pandas as pd
import pytest

from dsv import pitchcontrol as PC


def state(pos, home, vel=None, gk=None, ball=(0.0, 0.0), possession="home team", vmax=5.0):
    pos = np.asarray(pos, float)
    n = len(pos)
    return PC.FrameState(
        frame=0, period=1, player_id=np.arange(n) + 1,
        is_home=np.asarray(home, bool),
        is_gk=np.zeros(n, bool) if gk is None else np.asarray(gk, bool),
        detected=np.ones(n, bool), pos=pos,
        vel=np.zeros((n, 2)) if vel is None else np.asarray(vel, float),
        vmax=np.full(n, vmax), ball=np.asarray(ball, float), possession=possession,
    )


GRID = PC.make_grid(cell=3.0)


def test_grid_covers_pitch():
    g = PC.make_grid()
    assert g.shape == (68, 105)
    assert g.cells[0].tolist() == [-52.0, -33.5]
    assert g.cell_area == pytest.approx(1.0)


def test_total_control_is_one_and_never_exceeds_it():
    rng = np.random.default_rng(0)
    s = state(rng.uniform([-50, -32], [50, 32], (22, 2)), [True] * 11 + [False] * 11,
              vel=rng.normal(0, 2, (22, 2)))
    r = PC.pitch_control(s, GRID)
    assert r.total.max() <= 1.0 + 1e-5
    assert r.total.min() >= 0.98


def test_mirror_symmetry():
    s = state([[-10, 5], [10, -5]], [True, False])
    r = PC.pitch_control(s, GRID)
    home, away = r.team(True), r.team(False)
    # rotating the pitch 180 degrees swaps the two players
    assert np.allclose(home, away[::-1, ::-1], atol=1e-4)


def test_closer_player_controls_cell():
    s = state([[-20, 0], [30, 0]], [True, False], ball=(-20, 0))
    r = PC.pitch_control(s, GRID)
    j = GRID.index(-20, 0)
    assert r.pc[j, 0] > 0.95


def test_running_towards_space_increases_control():
    still = state([[0, 0], [10, 0]], [True, False])
    running = state([[0, 0], [10, 0]], [True, False], vel=[[5, 0], [0, 0]])
    j = GRID.index(15, 0)
    assert PC.pitch_control(running, GRID).pc[j, 0] > PC.pitch_control(still, GRID).pc[j, 0]


def test_leave_one_out_deltas_are_nonnegative_and_local():
    pos = [[-30, 0], [-30, 20], [0, 0], [10, 10]]
    s = state(pos, [False, False, True, True], possession="home team")
    loo = PC.leave_one_out(s, GRID)
    assert loo.defending_home is False
    assert list(loo.defender_ids) == [1, 2]
    assert loo.delta.min() >= -1e-5
    # each defender's gain sits on their own side: the one at y=0 covers lower y
    # than the one at y=20
    ys = GRID.cells[:, 1]
    cy = [(d.ravel() * ys).sum() / d.sum() for d in loo.delta]
    assert cy[0] < cy[1]
    sup = loo.suppression()
    assert (sup > 0).all()


def test_vmax_table_floor_and_goalkeeper():
    v95 = pd.DataFrame({
        "match_id": 1, "player_id": [1, 2, 3, 4], "period": 1,
        "v95": [4.0, 5.0, 6.0, 3.0],
        "group": ["Midfield", "Midfield", "Midfield", "Goalkeeper"],
    })
    vt = PC.vmax_table(v95, floor_quantile=0.5, gk_vmax=5.0)
    assert vt["vmax"].tolist() == [5.0, 5.0, 6.0, 5.0]
    assert PC.vmax_lookup(vt, 1)[(1, 1)] == 5.0
