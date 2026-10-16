"""Discrete-time Hawkes model with a piecewise-constant kernel and four kinds of excitation.

lam[i,t] = mu[i] + sum_b a_own[b] O[t,i,b] + a_ind[b] I[t,i,b] + a_sec[b] S[t,i,b] + a_mkt[b] M[t,i,b] + a_seas Q[t,i]

Each feature counts events in a lag window strictly before t, so lam[i,t] uses only sessions < t.
"""
import numpy as np
from scipy.optimize import minimize

BUCKETS = [(1, 1), (2, 5), (6, 20), (21, 60)]          # lags in sessions, inclusive
WIDTH = np.array([b - a + 1 for a, b in BUCKETS], float)
SEAS = (242, 262)
KINDS = ["own", "ind", "sec", "mkt"]
NAMES = [f"{k}{a}-{b}" for k in KINDS for a, b in BUCKETS] + ["seas"]


def window(E, a, b):
    """W[t] = sum of E over sessions t-b .. t-a (partial at the start)."""
    C = np.vstack([np.zeros((1, E.shape[1])), np.cumsum(E, 0)])
    T = E.shape[0]
    hi = np.clip(np.arange(T) - a + 1, 0, T)
    lo = np.clip(np.arange(T) - b, 0, T)
    return C[hi] - C[lo]


def onehot(ids):
    """N x G indicator; id -1 means no group."""
    G = ids.max() + 1
    H = np.zeros((len(ids), G))
    H[ids >= 0, ids[ids >= 0]] = 1
    return H


def features(E, member, ind, sec):
    """Feature tensor X[t, i, k] (float32) in NAMES order.

    E: T x N event counts. Own windows use all of a firm's events; peer windows use events filed while a member.
    ind, sec: integer group ids per firm, -1 if unclassified.
    """
    Em = E * member
    Hi, Hs = onehot(ind), onehot(sec)
    T, N = E.shape
    X = np.zeros((T, N, len(NAMES)), np.float32)
    for j, (a, b) in enumerate(BUCKETS):
        Wo, W = window(E, a, b), window(Em, a, b)
        it = (W @ Hi)[:, np.maximum(ind, 0)] * (ind >= 0)      # industry total incl. own
        st = (W @ Hs)[:, np.maximum(sec, 0)] * (sec >= 0)      # sector total incl. own
        tot = W.sum(1, keepdims=True)
        own_m = W * (ind >= 0)
        X[:, :, j] = Wo
        X[:, :, 4 + j] = it - own_m
        X[:, :, 8 + j] = st - it
        X[:, :, 12 + j] = tot - np.where(sec >= 0, st, W)
    W = window(Em, *SEAS)
    X[:, :, 16] = ((W @ Hi)[:, np.maximum(ind, 0)] - W) * (ind >= 0)
    return X


def fit(y, X, firm, nfirm, cols=None):
    """Poisson MLE with mu >= 0 per firm and a >= 0. y: (n,), X: (n, K), firm: (n,) index in [0, nfirm).

    cols: indices of X columns to use (others fixed at 0). Returns dict(mu, a, nll).
    """
    K = X.shape[1]
    cols = np.arange(K) if cols is None else np.asarray(cols)
    Z = X[:, cols].astype(np.float64)
    scale = Z.mean(0) + 1e-12
    Z = Z / scale
    cnt = np.bincount(firm, minlength=nfirm)
    ybar = np.bincount(firm, weights=y, minlength=nfirm) / np.maximum(cnt, 1)
    th0 = np.concatenate([np.maximum(ybar * 0.8, 1e-6), np.full(len(cols), 0.01 * y.mean())])

    def f(th):
        mu, a = th[:nfirm], th[nfirm:]
        lam = mu[firm] + Z @ a
        r = y / lam - 1.0
        nll = lam.sum() - (y * np.log(lam)).sum()
        g = -np.concatenate([np.bincount(firm, weights=r, minlength=nfirm), Z.T @ r])
        return nll, g

    bounds = [(1e-9, None)] * nfirm + [(0, None)] * len(cols)
    res = minimize(f, th0, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 5000, "ftol": 1e-12})
    a = np.zeros(K)
    a[cols] = res.x[nfirm:] / scale
    return {"mu": res.x[:nfirm], "a": a, "nll": res.fun, "ok": bool(res.success), "msg": str(res.message)}


def excitation(a):
    """K per kind: expected extra filings at a firm caused by one event at a firm of that kind, over 60 sessions."""
    return {k: float(a[4 * j:4 * j + 4] @ WIDTH) for j, k in enumerate(KINDS)}


def loglik(y, X, firm, mu, a):
    """Per-observation Poisson log-likelihood (without the log y! term, which cancels in comparisons)."""
    lam = mu[firm] + X.astype(np.float64) @ a
    return y * np.log(lam) - lam


def shuffle_labels(ind, sec, rng):
    """Reassign classified firms to the existing (industry, sector) slots at random; every group keeps its size."""
    ind, sec = ind.copy(), sec.copy()
    k = np.flatnonzero(ind >= 0)
    p = rng.permutation(k)
    ind[k], sec[k] = ind[p], sec[p]
    return ind, sec


def ratio(E, member, ind, sec, rows, cols_t, firm, nfirm):
    """Fit on the given (t, i) rows and return (K_ind / K_mkt, K dict, fit)."""
    X = features(E, member, ind, sec)[rows, cols_t]
    r = fit(E[rows, cols_t], X, firm, nfirm)
    k = excitation(r["a"])
    return k["ind"] / max(k["mkt"], 1e-12), k, r
