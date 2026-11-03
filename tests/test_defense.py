import numpy as np
import pandas as pd
import pytest

from dsv import defense as DF, pitchcontrol as PC, value as V


def frames(n=4000, seed=0, effect=-1.0):
    """Synthetic frames where shots depend on ball distance and on cover."""
    rng = np.random.default_rng(seed)
    bx, by = rng.uniform(-50, 50, n), rng.uniform(-30, 30, n)
    dist = np.hypot(52.5 - bx, by)
    rd = np.exp(-dist / 15) * 0.05 + 1e-4
    cover = 1 / (1 + np.exp(-rng.normal(1.0, 0.8, n)))
    z = -1.5 - dist / 8 + effect * DF.logit(cover)
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return pd.DataFrame({
        "match_id": rng.integers(0, 8, n), "possession_id": np.arange(n), "frame": np.arange(n),
        "ball_x": bx, "ball_y": by, "rd": rd, "rd_xg_here": rd / 3, "cover": cover,
        "tc": rd * (1 - cover), "tc_xg_here": rd / 3 * (1 - cover),
        "t_in_possession": rng.uniform(0, 20, n), "def_goal_side": rng.integers(3, 11, n), "y": y})


def test_cover_coefficient_recovered_and_counterfactual_sign():
    f = frames()
    m = DF.fit_two_part(f, f["y"])
    assert m.cover_effect == pytest.approx(-1.0, abs=0.35)
    typ = DF.fit_typical_cover(f)
    prevented = DF.shots_prevented(m, typ, f)
    # frames whose cover beats the typical cover prevent shots, and vice versa
    better = f["cover"].to_numpy() > typ.predict(f)
    assert prevented[better].mean() > 0 > prevented[~better].mean()
    # with typical cover substituted for the actual cover, nothing is prevented
    same = DF.shots_prevented(m, typ, f.assign(cover=typ.predict(f)))
    assert np.abs(same).max() < 1e-9


def test_design_shapes_and_cover_columns():
    f = DF.add_cover_columns(frames(200))
    assert np.allclose(f["cover_xg_here"], f["cover"])          # same construction in the synthetic data
    assert DF.design(f, cover_col=None).shape == (200, 2 + 7)
    assert DF.design(f).shape == (200, 2 + 7 + 1)


def test_possession_moments_uses_peak_danger_before_lookback():
    f = pd.DataFrame({
        "match_id": 1, "possession_id": 1, "frame": [0, 10, 20, 30, 40, 50],
        "ball_x": [0, 10, 30, 20, 45, 50.0], "ball_y": 0.0,
        "rd": [.01, .02, .05, .03, .09, .1], "rd_xg_here": [.001, .002, .02, .01, .05, .08],
        "tc": .01, "tc_xg_here": .001, "cover": [.5, .6, .7, .8, .9, .95],
        "cover_xg_here": [.5, .6, .7, .8, .9, .95], "def_goal_side": 8})
    poss = pd.DataFrame({"match_id": [1], "possession_id": [1], "frame_end": [50], "first_shot_frame": [50],
                         "shot": [True], "goal": [False], "duration_s": [5.0], "peak_tc_early": [99.0]})
    p = DF.possession_moments(f, poss, fps=10, lookback_s=2.0)
    assert len(p) == 1
    row = p.iloc[0]
    assert row["cover_at_peak"] == pytest.approx(0.7)       # frame 20: the most dangerous before frame 30
    assert row["peak_tc_early"] == pytest.approx(0.01)      # recomputed, not the stale column
    assert row["y"] == 1


def test_reachable_danger_matches_frame_weights():
    grid = PC.make_grid(105, 68, 2.0)
    vals = np.linspace(0, 1, len(grid.cells))
    out = DF.reachable_danger([20.0], [5.0], grid.cells, vals, 105, 68)
    a = np.exp(V.OPEN_DATA_TRANSITION.logpdf(20.0, 5.0, grid.cells[:, 0], grid.cells[:, 1], 52.5, 34))
    assert out[0] == pytest.approx((a * vals).sum() / a.sum())
