"""Datasets with an MNAR training log and an unbiased (MAR / fully observed) test.

Each loader returns a dict with dense float tensors
    O      (m, n)  1 if the pair was exposed in the training log
    Y      (m, n)  binary outcome on exposed pairs (0 elsewhere)
    P_given        optional (m, n) propensities shipped with the dataset
    test   list of (user, item_array, relevance_array) over unbiased candidates
"""

import os

import numpy as np
import pandas as pd
import torch

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def load_coat(root=ROOT, threshold=4):
    d = os.path.join(root, "coat")
    tr = np.loadtxt(os.path.join(d, "train.ascii"))
    te = np.loadtxt(os.path.join(d, "test.ascii"))
    pr = np.loadtxt(os.path.join(d, "propensities.ascii"))
    O = (tr > 0).astype(np.float64)
    Y = (tr >= threshold).astype(np.float64)
    test = []
    for u in range(te.shape[0]):
        # 7.9% of the random test ratings hit items the user also rated in the
        # MNAR log; as for KuaiRec, only never-logged items are ranked.
        items = np.nonzero((te[u] > 0) & (tr[u] == 0))[0]
        test.append((u, items, (te[u, items] >= threshold).astype(np.float64)))
    return {
        "name": "coat",
        "O": torch.tensor(O),
        "Y": torch.tensor(Y),
        "P_given": torch.tensor(pr),
        "test": test,
    }


def load_kuairec(root=ROOT, threshold=2.0):
    d = os.path.join(root, "KuaiRec 2.0", "data")
    if not os.path.isdir(d):
        d = os.path.join(root, "KuaiRec", "data")
    cols = ["user_id", "video_id", "watch_ratio"]
    big = pd.read_csv(os.path.join(d, "big_matrix.csv"), usecols=cols)
    small = pd.read_csv(os.path.join(d, "small_matrix.csv"), usecols=cols)
    users = np.union1d(big.user_id.unique(), small.user_id.unique())
    items = np.union1d(big.video_id.unique(), small.video_id.unique())
    uidx = {u: k for k, u in enumerate(users)}
    iidx = {i: k for k, i in enumerate(items)}
    m, n = len(users), len(items)
    bu = big.user_id.map(uidx).values
    bi = big.video_id.map(iidx).values
    # A pair can be logged several times; keep the mean watch ratio.
    agg = pd.DataFrame({"u": bu, "i": bi, "w": big.watch_ratio.values}).groupby(["u", "i"]).w.mean()
    O = torch.zeros(m, n, dtype=torch.float64)
    Y = torch.zeros(m, n, dtype=torch.float64)
    uu = torch.as_tensor(agg.index.get_level_values(0).values)
    ii = torch.as_tensor(agg.index.get_level_values(1).values)
    O[uu, ii] = 1.0
    Y[uu, ii] = torch.as_tensor((agg.values >= threshold).astype(np.float64))
    su = small.user_id.map(uidx).values
    si = small.video_id.map(iidx).values
    sg = pd.DataFrame({"u": su, "i": si, "w": small.watch_ratio.values}).groupby(["u", "i"]).w.mean().reset_index()
    Onp = O.numpy()
    test = []
    for u, grp in sg.groupby("u"):
        it = grp.i.values
        keep = Onp[u, it] == 0          # never rank what was already logged
        it = it[keep]
        rel = (grp.w.values[keep] >= threshold).astype(np.float64)
        test.append((int(u), it, rel))
    return {"name": "kuairec", "O": O, "Y": Y, "P_given": None, "test": test}


def load_yahoo(root=ROOT, threshold=4, mar_holdout=0.05, seed=2024):
    """Yahoo! R3: 311,704 self-selected song ratings of 15,400 users (MNAR
    log) and 54,000 ratings of 10 uniformly random songs for 5,400 of them
    (MAR test). Files as distributed with the AutoDebias code
    (datasets/yahooR3/{user,random}.txt, 0-indexed "user,item,rating")."""
    d = os.path.join(root, "yahooR3")
    tr = pd.read_csv(os.path.join(d, "user.txt"), header=None, names=["u", "i", "r"])
    te = pd.read_csv(os.path.join(d, "random.txt"), header=None, names=["u", "i", "r"])
    m = int(max(tr.u.max(), te.u.max())) + 1
    n = int(max(tr.i.max(), te.i.max())) + 1
    O = torch.zeros(m, n, dtype=torch.float64)
    Y = torch.zeros(m, n, dtype=torch.float64)
    O[tr.u.values, tr.i.values] = 1.0
    Y[tr.u.values, tr.i.values] = torch.as_tensor((tr.r.values >= threshold).astype(np.float64))
    # A small random share of the MAR users is held out entirely: it is used
    # only to estimate P(Y=1 | MAR) for the Naive-Bayes propensity and never
    # appears in validation or test.
    rng = np.random.default_rng(seed)
    te_users = np.sort(te.u.unique())
    hold = set(rng.choice(te_users, int(round(mar_holdout * len(te_users))), replace=False).tolist())
    cal = te[te.u.isin(hold)]
    mar_rate = float((cal.r.values >= threshold).mean())
    Onp = O.numpy()
    test = []
    for u, grp in te[~te.u.isin(hold)].groupby("u"):
        it = grp.i.values
        keep = Onp[u, it] == 0
        test.append((int(u), it[keep], (grp.r.values[keep] >= threshold).astype(np.float64)))
    return {"name": "yahoo", "O": O, "Y": Y, "P_given": None, "test": test,
            "mar_rate": mar_rate, "mar_holdout_users": sorted(hold)}


def split_test(test, frac_val, seed, by="entry"):
    """Split the unbiased data into validation (tuning) and test parts.

    by="entry": per user, a random fraction of candidates goes to validation
                (Coat, where every user has only 16 MAR ratings);
    by="user" : a random fraction of users goes to validation (KuaiRec).
    """
    rng = np.random.default_rng(seed)
    val, tst = [], []
    if by == "user":
        perm = rng.permutation(len(test))
        nv = int(round(frac_val * len(test)))
        vs = set(perm[:nv].tolist())
        for k, t in enumerate(test):
            (val if k in vs else tst).append(t)
        return val, tst
    for u, it, rel in test:
        mask = rng.random(len(it)) < frac_val
        val.append((u, it[mask], rel[mask]))
        tst.append((u, it[~mask], rel[~mask]))
    return val, tst


def load(name):
    if name == "coat":
        return load_coat()
    if name == "kuairec":
        return load_kuairec()
    if name == "yahoo":
        return load_yahoo()
    raise ValueError(name)
