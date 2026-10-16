"""Model B: HAR volatility forecasts with and without the Hawkes filing intensity."""
import numpy as np


def variance(o, h, l, c):
    """Parkinson range variance plus the squared overnight gap, floored at 1e-6. NaN where prices are missing."""
    prev = np.vstack([np.full((1, c.shape[1]), np.nan), c[:-1]])
    v = np.log(h / l) ** 2 / (4 * np.log(2)) + np.log(o / prev) ** 2
    return np.where(np.isfinite(v), np.maximum(v, 1e-6), np.nan)


def trailing_mean(v, n):
    """Mean of v over sessions t-n+1..t; NaN unless every value is present."""
    T = v.shape[0]
    ok = np.isfinite(v)
    C = np.vstack([np.zeros((1, v.shape[1])), np.cumsum(np.where(ok, v, 0), 0)])
    K = np.vstack([np.zeros((1, v.shape[1])), np.cumsum(ok, 0)])
    hi = np.arange(1, T + 1)
    lo = np.maximum(hi - n, 0)
    m = (C[hi] - C[lo]) / n
    return np.where((K[hi] - K[lo] == n) & (hi >= n)[:, None], m, np.nan)


def har_panel(v, member):
    """HAR regressors at t and the target log mean(v[t+1..t+5])."""
    vm = np.nanmean(np.where(member, v, np.nan), 1, keepdims=True) * np.ones_like(v)
    feats = [np.log(trailing_mean(x, n)) for x in (v, vm) for n in (1, 5, 22)]
    fwd = np.vstack([trailing_mean(v, 5)[5:], np.full((5, v.shape[1]), np.nan)])
    return feats, np.log(fwd)


def ols(Xtr, ytr):
    return np.linalg.lstsq(np.column_stack([np.ones(len(Xtr)), Xtr]), ytr, rcond=None)[0]


def predict(b, X):
    return np.column_stack([np.ones(len(X)), X]) @ b


def nw_t(d, lags=5):
    """Newey-West t statistic for the mean of d."""
    d = np.asarray(d, float)
    n, m = len(d), d.mean()
    e = d - m
    lrv = e @ e / n
    for k in range(1, lags + 1):
        lrv += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    return m / np.sqrt(lrv / n)
