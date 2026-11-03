import numpy as np
import pandas as pd
import pytest

from dsv import defense as DF, dfl, pitchcontrol as PC, value as V
from dsv.data import MatchMeta


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
