"""
dsv.metrics
===========
Scores for probabilistic binary predictions (pass completion now, shots in
Week 5) and a dependency-free logistic regression for simple baselines.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata

EPS = 1e-4


def log_loss(y, p, eps: float = EPS) -> float:
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def auc(y, p) -> float:
    """Area under the ROC curve (Mann-Whitney, ties averaged)."""
    y = np.asarray(y, bool)
    r = rankdata(np.asarray(p, float))
    # floats: on Windows numpy's default int is 32-bit and n1 * n0 overflows
    # for a few hundred thousand rows
    n1, n0 = float(y.sum()), float((~y).sum())
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def reliability(y, p, bins: int = 10) -> pd.DataFrame:
    """Mean prediction and observed rate per equal-width probability bin."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    b = np.clip((p * bins).astype(int), 0, bins - 1)
    df = pd.DataFrame({"bin": b, "p": p, "y": y})
    return df.groupby("bin").agg(p_mean=("p", "mean"), rate=("y", "mean"), n=("y", "size")).reset_index()


def ece(y, p, bins: int = 10) -> float:
    """Expected calibration error: |predicted - observed|, averaged over
    bins weighted by their share of predictions."""
    r = reliability(y, p, bins)
    return float((r["n"] * (r["p_mean"] - r["rate"]).abs()).sum() / r["n"].sum())


def summary(y, p) -> dict:
    return dict(n=len(y), log_loss=log_loss(y, p), brier=brier(y, p), auc=auc(y, p), ece=ece(y, p))


def fit_logistic(X, y, l2: float = 1e-4) -> np.ndarray:
    """Logistic regression with intercept and a small ridge penalty.
    X (n, k) is standardized internally; returns coefficients for raw X
    as [intercept, b_1..b_k]."""
    X = np.atleast_2d(np.asarray(X, float))
    if X.shape[0] != len(y):
        X = X.T
    y = np.asarray(y, float)
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Z = np.column_stack([np.ones(len(X)), (X - mu) / sd])

    def nll(w):
        s = Z @ w
        return np.mean(np.logaddexp(0, s) - y * s) + l2 * np.sum(w[1:] ** 2)

    def grad(w):
        p = 1 / (1 + np.exp(-(Z @ w)))
        g = Z.T @ (p - y) / len(y)
        g[1:] += 2 * l2 * w[1:]
        return g

    w = minimize(nll, np.zeros(Z.shape[1]), jac=grad, method="BFGS").x
    b = w[1:] / sd
    return np.concatenate([[w[0] - np.sum(b * mu)], b])


def predict_logistic(coef: np.ndarray, X) -> np.ndarray:
    X = np.atleast_2d(np.asarray(X, float))
    if X.shape[1] != len(coef) - 1:
        X = X.T
    return 1 / (1 + np.exp(-(coef[0] + X @ coef[1:])))
