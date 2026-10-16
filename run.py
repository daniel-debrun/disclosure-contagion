"""Run the pre-registered study end to end and write results/results.json and the figures.

    python run.py [--perms 100]
"""
import argparse
import json

import numpy as np

from contagion import data, hawkes as H, vol as V

TRAIN = ("2017-01-01", "2021-12-31")
TEST = ("2022-01-01", "2026-06-30")
OUT = "results"


def split(sessions, lo, hi):
    return (sessions >= np.datetime64(lo)) & (sessions <= np.datetime64(hi))


def group_ids(labels):
    names = sorted(set(labels) - {"?"})
    idx = {n: k for k, n in enumerate(names)}
    return np.array([idx.get(x, -1) for x in labels])


def contagion(E, member, ind, sec, tr, te, perms, seed=0):
    T, N = E.shape
    ok = member & (ind >= 0)
    rtr, ctr = np.nonzero(ok & tr[:, None])
    X = H.features(E, member, ind, sec)
    full = H.fit(E[rtr, ctr], X[rtr, ctr], ctr, N)
    no_peer = [j for j, n in enumerate(H.NAMES) if not n.startswith(("ind", "sec"))]
    restricted = H.fit(E[rtr, ctr], X[rtr, ctr], ctr, N, cols=no_peer)

    # Test: firms with a train mu only; per-session sums of the log-likelihood gain.
    seen = np.bincount(ctr, minlength=N) > 0
    rte, cte = np.nonzero(ok & te[:, None] & seen[None, :])
    gain = (H.loglik(E[rte, cte], X[rte, cte], cte, full["mu"], full["a"])
            - H.loglik(E[rte, cte], X[rte, cte], cte, restricted["mu"], restricted["a"]))
    per_session = np.bincount(rte, weights=gain, minlength=T)[np.unique(rte)]

    k = H.excitation(full["a"])
    obs = k["ind"] / k["mkt"]
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(perms):
        null.append(H.ratio(E, member, *H.shuffle_labels(ind, sec, rng), rtr, ctr, ctr, N)[0])
    p = (1 + sum(r >= obs for r in null)) / (perms + 1) if perms else None
    t_gain = float(V.nw_t(per_session))
    return {
        "K": k, "ratio_ind_mkt": obs, "ratio_sec_mkt": k["sec"] / k["mkt"],
        "a": dict(zip(H.NAMES, map(float, full["a"]))),
        "fit_ok": full["ok"] and restricted["ok"],
        "train_obs": int(len(rtr)), "train_events": int(E[rtr, ctr].sum()),
        "test_obs": int(len(rte)), "test_events": int(E[rte, cte].sum()),
        "test_loglik_gain": float(gain.sum()), "test_gain_nw_t": t_gain,
        "perm_p": p, "perm_null": [float(r) for r in null],
        "perm_null_median": float(np.median(null)) if null else None,
        "verdict": verdict_a(p, gain.sum(), t_gain),
    }, full, X, seen


def verdict_a(p, gain, t):
    if gain <= 0:
        return "refuted"
    return "supported" if (p is not None and p < 0.05 and t > 2) else "inconclusive"


def volatility(d, member, ind, X, full, seen, tr, te):
    v = V.variance(d["open"], d["high"], d["low"], d["close"])
    har, target = V.har_panel(v, member)
    T, N = v.shape
    mu = full["mu"][None, :]
    a = full["a"]
    # Hawkes parts of the intensity for session t+1, which use filings in sessions <= t only.
    own = np.vstack([X[1:, :, 0:4].astype(float) @ a[0:4], np.full((1, N), np.nan)])
    peer = np.vstack([X[1:, :, 4:12].astype(float) @ a[4:12], np.full((1, N), np.nan)])
    f_own, f_peer = np.log1p(own / mu), np.log1p(peer / mu)

    cols = har + [f_own, f_peer]
    allf = np.all([np.isfinite(c) for c in cols + [target]], 0)
    base = member & (ind >= 0)[None, :] & seen[None, :] & allf
    idx = np.arange(T)
    tr5 = tr & np.isin(idx + 5, np.flatnonzero(tr))          # train targets end inside train
    rows = {s: np.nonzero(base & m[:, None]) for s, m in (("train", tr5), ("test", te))}
    M = {s: (np.column_stack([c[r] for c in cols]), target[r], r[0]) for s, r in rows.items()}

    specs = {"B0": slice(0, 6), "B1": slice(0, 7), "B2": slice(0, 8)}
    out, err = {}, {}
    for name, sl in specs.items():
        b = V.ols(M["train"][0][:, sl], M["train"][1])
        e = (V.predict(b, M["test"][0][:, sl]) - M["test"][1]) ** 2
        err[name] = e
        out[name] = {"test_mse": float(e.mean()), "coef": [float(x) for x in b]}
    sess = M["test"][2]
    _, inv = np.unique(sess, return_inverse=True)

    def compare(a_, b_):
        d = np.bincount(inv, weights=err[a_] - err[b_]) / np.bincount(inv)
        return {"mse_change_pct": float(100 * (err[b_].mean() / err[a_].mean() - 1)), "nw_t": float(V.nw_t(d))}

    b2_vs_b1 = compare("B1", "B2")
    passed = b2_vs_b1["mse_change_pct"] < 0 and b2_vs_b1["nw_t"] > 2
    return {**out, "B2_vs_B1": b2_vs_b1, "B1_vs_B0": compare("B0", "B1"),
            "peer_coef_B2": out["B2"]["coef"][-1], "train_obs": int(len(M["train"][1])),
            "test_obs": int(len(M["test"][1])), "verdict": "supported" if passed else
            ("refuted" if b2_vs_b1["mse_change_pct"] >= 0 else "inconclusive")}


def figures(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a = res["A"]["a"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    x = np.arange(len(H.BUCKETS))
    for j, (kind, label) in enumerate([("ind", "same industry"), ("sec", "same sector"), ("mkt", "other sectors")]):
        vals = [a[f"{kind}{lo}-{hi}"] for lo, hi in H.BUCKETS]
        ax[0].bar(x + (j - 1) * 0.27, vals, 0.27, label=label)
    ax[0].set_xticks(x, [f"{lo}-{hi}" if lo != hi else str(lo) for lo, hi in H.BUCKETS])
    ax[0].set_xlabel("sessions since the other firm's filing")
    ax[0].set_ylabel("extra filings per day, per event")
    ax[0].set_yscale("log")
    ax[0].legend(frameon=False)
    ax[0].set_title("Excitation by lag (train fit)")
    null = res["A"]["perm_null"]
    ax[1].hist(null, bins=20, color="0.7")
    ax[1].axvline(res["A"]["ratio_ind_mkt"], color="#E61919", lw=2, label="observed")
    ax[1].set_xlabel("K_ind / K_mkt")
    ax[1].set_title(f"Shuffled industries ({len(null)}), p = {res['A']['perm_p']:.3f}")
    ax[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/contagion.png", dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=100)
    args = ap.parse_args()
    d = data.load()
    s = d["sessions"]
    member = d["member"]
    ind, sec = group_ids(d["industry"]), group_ids(d["sector"])
    tr, te = split(s, *TRAIN), split(s, *TEST)

    A, full, X, seen = contagion(d["y"].astype(float), member, ind, sec, tr, te, args.perms)
    A_narrow, *_ = contagion(d["y_narrow"].astype(float), member, ind, sec, tr, te, perms=0)
    B = volatility(d, member, ind, X, full, seen, tr, te)
    res = {"train": TRAIN, "test": TEST, "firms": int(len(d["ciks"])), "classified": int((ind >= 0).sum()),
           "industries": int(ind.max() + 1), "sectors": int(sec.max() + 1), "A": A, "A_narrow": A_narrow, "B": B}
    with open(f"{OUT}/results.json", "w") as f:
        json.dump(res, f, indent=1)
    figures(res)
    show = {k: v for k, v in A.items() if k != "perm_null"}
    print(json.dumps({"A": show, "A_narrow_K": A_narrow["K"], "A_narrow_gain": A_narrow["test_loglik_gain"],
                      "A_narrow_t": A_narrow["test_gain_nw_t"], "B": {k: v for k, v in B.items()
                                                                       if not isinstance(v, dict) or "coef" not in v}},
                     indent=1))


if __name__ == "__main__":
    main()
