"""Simulate panels from the model with known parameters and check the features and the fitter."""
import numpy as np

from contagion import hawkes as H


def simulate(a, mu=0.01, T=4000, n_ind=8, per_ind=10, ind_per_sec=4, seed=0):
    rng = np.random.default_rng(seed)
    N = n_ind * per_ind
    ind = np.repeat(np.arange(n_ind), per_ind)
    sec = ind // ind_per_sec
    member = np.ones((T, N), bool)
    E = np.zeros((T, N))
    used = np.zeros((T, N, len(H.NAMES)))
    for t in range(T):
        x = np.zeros((N, len(H.NAMES)))
        for j, (lo, hi) in enumerate(H.BUCKETS):
            w = E[max(t - hi, 0):max(t - lo + 1, 0)].sum(0)
            it = np.bincount(ind, weights=w, minlength=n_ind)[ind]
            st = np.bincount(sec, weights=w, minlength=sec.max() + 1)[sec]
            x[:, j], x[:, 4 + j], x[:, 8 + j], x[:, 12 + j] = w, it - w, st - it, w.sum() - st
        used[t] = x
        E[t] = rng.poisson(mu + x @ a)
    return E, member, ind, sec, used


def params(own, ind, sec, mkt):
    return np.concatenate([own, ind, sec, mkt, [0.0]])


OWN = [0.05, 0.01, 0.002, 0.0005]
MKT = [0.0005, 0.0002, 0.00005, 0.00001]


def rows(E, t0=300):
    T, N = E.shape
    tt, ii = np.meshgrid(np.arange(t0, T), np.arange(N), indexing="ij")
    return tt.ravel(), ii.ravel()


def fit_sim(a, seed):
    E, member, ind, sec, used = simulate(a, seed=seed)
    X = H.features(E, member, ind, sec)
    # The features the fitter sees are exactly the ones the simulation used: nothing after t leaks in.
    assert np.allclose(X[300:, :, :16], used[300:, :, :16])
    tt, ii = rows(E)
    r = H.fit(E[tt, ii], X[tt, ii], ii, E.shape[1])
    assert r["ok"], r["msg"]
    return H.excitation(r["a"]), H.excitation(a)


def perm_p(a, seed, n=30):
    """The pre-registered decision rule: share of label shuffles with a ratio at least the observed one."""
    E, member, ind, sec, _ = simulate(a, seed=seed)
    tt, ii = rows(E)
    obs = H.ratio(E, member, ind, sec, tt, ii, ii, E.shape[1])[0]
    rng = np.random.default_rng(seed)
    null = [H.ratio(E, member, *H.shuffle_labels(ind, sec, rng), tt, ii, ii, E.shape[1])[0] for _ in range(n)]
    return (1 + sum(r >= obs for r in null)) / (n + 1)


CONTAGION = params(OWN, [0.01, 0.003, 0.0005, 0.0001], [0.002, 0.0006, 0.0001, 0.00002], MKT)
NULL = params(OWN, MKT, MKT, MKT)          # a peer's event excites exactly as much as anyone's


def test_recovers_contagion():
    k, true = fit_sim(CONTAGION, seed=1)
    for kind in ("own", "ind"):
        assert abs(k[kind] / true[kind] - 1) < 0.35, (kind, k, true)


def test_permutation_detects_contagion():
    assert perm_p(CONTAGION, seed=3) < 0.05


def test_permutation_no_false_positive():
    # The raw ratio is biased upward under the null (a >= 0 censors noise); the permutation test must not be.
    assert perm_p(NULL, seed=2) > 0.05
