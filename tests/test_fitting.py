from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from dsv import data as D, metrics as M, pitchcontrol as PC


def random_batch(P=1200, n=22, seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.uniform([-50, -32], [50, 32], (P, n, 2))
    is_home = np.zeros((P, n), bool)
    is_home[:, : n // 2] = True
    # targets near a random player so outcomes are not all trivial
    j = rng.integers(0, n, P)
    target = np.clip(pos[np.arange(P), j] + rng.normal(0, 6, (P, 2)), [-52, -33], [52, 33])
    return PC.PointBatch(
        pos=pos, vel=rng.normal(0, 2, (P, n, 2)), vmax=np.full((P, n), 5.0),
        is_gk=np.zeros((P, n), bool), is_home=is_home, present=np.ones((P, n), bool),
        ball=rng.uniform([-50, -32], [50, 32], (P, 2)), target=target,
    )


def test_points_match_grid():
    b = random_batch(P=3)
    prm = PC.PCParams(tol=1e-6)
    grid = PC.make_grid(cell=5.0)
    for i in range(3):
        s = PC.FrameState(0, 1, np.arange(22) + 1, b.is_home[i], b.is_gk[i], b.present[i],
                          b.pos[i], b.vel[i], b.vmax[i], b.ball[i], "home team")
        cell = grid.cells[grid.index(*b.target[i])]
        one = b.subset([i])
        one.target = cell[None]
        g = PC.pitch_control(s, grid, prm).team(True).ravel()[grid.index(*cell)]
        p = PC.team_control_at_points(one, [True], prm)[0]
        assert p == pytest.approx(g, abs=1e-4)


def test_absent_players_are_ignored():
    b = random_batch(P=50)
    full = PC.team_control_at_points(b, np.ones(50, bool))
    b.present[:, 21] = False
    b.pos[:, 21] = 1e6  # garbage in a padded slot must not matter
    out = PC.team_control_at_points(b, np.ones(50, bool))
    assert np.all(out >= full - 1e-5)      # removing an away player never hurts home
    assert np.isfinite(out).all()


def test_fit_recovers_known_parameters():
    b = random_batch(P=2500, seed=1)
    home = np.random.default_rng(2).random(len(b)) < 0.5
    truth = replace(PC.PCParams(), t_react=0.5, sigma=0.3, lam=40.0, tol=1e-4)
    p = PC.team_control_at_points(b, home, truth)
    y = np.random.default_rng(3).random(len(b)) < p
    init = replace(PC.PCParams(), lam=40.0)          # starts at t_react 0.7, sigma 0.45
    fitted, res = PC.fit_params(b, home, y, init=init, fit=("t_react", "sigma"))
    assert res.success or res.nit > 10
    assert fitted.t_react == pytest.approx(0.5, abs=0.12)
    assert fitted.sigma == pytest.approx(0.3, abs=0.1)
    assert M.log_loss(y, PC.team_control_at_points(b, home, fitted)) <= M.log_loss(
        y, PC.team_control_at_points(b, home, init)) + 1e-9


def test_metrics():
    y = np.array([0, 0, 1, 1])
    assert M.auc(y, [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert M.auc(y, [0.5, 0.5, 0.5, 0.5]) == 0.5
    assert M.log_loss(y, [0.5] * 4) == pytest.approx(np.log(2))
    assert M.brier(y, y) == 0.0
    assert M.ece([0, 1] * 50, [0.5] * 100) == pytest.approx(0.0)


def test_logistic_recovers_coefficients():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (20000, 2)) * [1.0, 10.0]
    y = rng.random(20000) < 1 / (1 + np.exp(-(0.5 + 1.5 * X[:, 0] - 0.1 * X[:, 1])))
    c = M.fit_logistic(X, y, l2=0)
    assert c == pytest.approx([0.5, 1.5, -0.1], abs=0.08)


def test_passes_with_destination():
    meta = type("M", (), {"home_team_id": 10})()
    ev = pd.DataFrame({
        "event_type": ["player_possession"] * 4,
        "frame_start": [0, 20, 40, 300], "frame_end": [10, 30, 50, 310], "period": 1,
        "end_type": ["pass", "pass", "pass", "shot"],
        "pass_outcome": ["successful", "unsuccessful", "successful", None],
        "team_id": [10, 10, 20, 20],
    })
    ball = pd.DataFrame({"frame": np.arange(0, 400), "bx": np.arange(400.0) / 10, "by": 0.0})
    ps = D.passes_with_destination(ev, ball, meta)
    # third pass: next possession starts 250 frames later -> dropped
    assert ps["frame_end"].tolist() == [10, 30]
    assert ps["dest_x"].tolist() == [2.0, 4.0]
    assert ps["success"].tolist() == [True, False]
    assert ps["passer_home"].tolist() == [True, True]
    assert ps["pass_length"].tolist() == pytest.approx([1.0, 1.0])


def test_sigma_extrap_softens_only_extrapolated_players():
    b = random_batch(P=300, seed=4)
    home = np.ones(len(b), bool)
    base = PC.PCParams(lam=40.0, tol=1e-6)
    b.detected = np.ones_like(b.present)
    p_det = PC.team_control_at_points(b, home, replace(base, sigma_extrap=1.5))
    assert p_det == pytest.approx(PC.team_control_at_points(b, home, base), abs=1e-6)
    b.detected[:] = False
    p_soft = PC.team_control_at_points(b, home, replace(base, sigma_extrap=1.5))
    p_sharp = PC.team_control_at_points(b, home, base)
    # more timing uncertainty pulls confident predictions toward the middle
    sure = np.abs(p_sharp - 0.5) > 0.45
    assert np.mean(np.abs(p_soft[sure] - 0.5)) < np.mean(np.abs(p_sharp[sure] - 0.5))


def test_xg_features_and_shipped_model():
    from dsv import xg
    # central shot 11 m out, nobody around: angle ~0.64 rad, sizeable xG
    f = xg.shot_features([-11.0, 0.0], False, np.empty((0, 2)), np.array([-0.5, 0.0]))
    assert f["dist"] == pytest.approx(11.0)
    assert f["angle"] == pytest.approx(2 * np.arctan(3.66 / 11), abs=1e-3)
    assert f["n_block"] == 0
    # a defender in the shooter-posts triangle counts as a blocker
    g = xg.shot_features([-11.0, 0.0], False, np.array([[-5.0, 0.0]]), np.array([-0.5, 0.0]))
    assert g["n_block"] == 1
    df = pd.DataFrame([f, g, xg.shot_features([-30.0, 15.0], True, np.empty((0, 2)), None)])
    p = xg.xg_for_shots(df)
    assert p[0] > p[1] > p[2] > 0          # blocked is worse; a far header is worst


def test_auc_large_inputs_do_not_overflow():
    rng = np.random.default_rng(0)
    y = rng.random(400_000) < 0.05
    p = y + rng.normal(0, 1, len(y))
    a = M.auc(y, p)
    assert 0.5 < a < 1.0
