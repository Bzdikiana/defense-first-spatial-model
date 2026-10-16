import numpy as np
import pandas as pd
import pytest

from dsv import pitchcontrol as PC, value as V


def test_fit_xt_on_a_toy_chain():
    # 2 x 1 zones: in zone 1 (near goal) every action is a shot worth 0.5;
    # in zone 0 every action is a successful move into zone 1
    acts = pd.DataFrame({
        "kind": ["shot"] * 4 + ["move"] * 4,
        "success": [False] * 4 + [True] * 4,
        "x0": [0.75] * 4 + [0.25] * 4, "y0": 0.5,
        "x1": [np.nan] * 4 + [0.75] * 4, "y1": [np.nan] * 4 + [0.5] * 4,
    })
    xt = V.fit_xt(acts, np.array([[0.0, 0.5]]))
    assert xt[0].tolist() == pytest.approx([0.5, 0.5])
    # half the moves fail: zone 0 is worth half as much
    acts.loc[[4, 5], "success"] = False
    acts.loc[[4, 5], ["x1", "y1"]] = np.nan
    assert V.fit_xt(acts, np.array([[0.0, 0.5]]))[0, 0] == pytest.approx(0.25)


def test_zone_value_at_interpolates_between_centres():
    z = np.array([[0.0, 1.0]])
    assert V.zone_value_at(z, [0.25, 0.5, 0.75, 0.99], [0.5] * 4).tolist() == pytest.approx([0, 0.5, 1, 1])
    grid = PC.make_grid(105, 68, cell=1.0)
    s = V.zone_surface(V.OPEN_DATA_XT, grid, 105, 68)
    assert s.shape == (68, 105)
    assert s[:, -1].max() > s[:, 0].max()          # higher near the goal attacked (+x)


def test_transition_density_is_normalized_on_the_pitch():
    grid = PC.make_grid(105, 68, cell=0.5)
    c = grid.cells
    for ball in [(0.0, 0.0), (45.0, 25.0), (-50.0, -30.0)]:
        lp = V.OPEN_DATA_TRANSITION.logpdf(ball[0], ball[1], c[:, 0], c[:, 1], 52.5, 34.0)
        assert np.exp(lp).sum() * grid.cell_area == pytest.approx(1.0, abs=0.01)


def test_offside_players():
    meta = type("M", (), {"pitch_length": 105.0})()
    # home attacks +x; away defenders at x = 40 (second last) and 50 (keeper)
    s = PC.FrameState(0, 1, np.arange(5) + 1, np.array([True, True, True, False, False]),
                      np.array([False, False, False, False, True]), np.ones(5, bool),
                      np.array([[45.0, 0.0], [35.0, 0.0], [-10.0, 0.0], [40.0, 5.0], [50.0, 0.0]]),
                      np.zeros((5, 2)), np.full(5, 5.0), np.array([20.0, 0.0]), "home team")
    assert V.offside_players(s, meta, True).tolist() == [True, False, False, False, False]
    s.ball = np.array([46.0, 0.0])                   # ball ahead of the attacker: onside
    assert not V.offside_players(s, meta, True).any()
