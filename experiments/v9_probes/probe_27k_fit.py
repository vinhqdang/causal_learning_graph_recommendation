"""Probe L27-fit: can the LEAF-tied channel weights be learned from the training log alone (no validation
data)?  Temporal protocol, all Pure users, lifelong tables from data/raw/lifelong_27k.npz.

Features of a (user, pool video) pair: the row-standardised EASE(click + 0.5 shown) score (lam 3000), the
popularity offset, and the four lifelong channels of probe_27k.py (tag lift, author familiarity, author
lift, music familiarity), standardised exactly as there. A logistic regression of click on these
features is fitted on the in-pool training impressions (clicked vs shown-not-clicked, before the
cut-off: the exposure-conditional target that a uniform insertion measures). Test scores use the fitted
coefficients; no hyperparameter is selected on validation. Reported on the 5 test splits of probe_27k.py.
python3 experiments/v9_probes/probe_27k_fit.py OUT.txt
"""
import json
import sys

import numpy as np
import pandas as pd
import torch
from scipy import stats

from common import REPO, BucketSelector, ease, rowstd, std
from drup.data import split_test
from drup.data2 import load_kuairand_rich

torch.set_num_threads(3)
out = sys.argv[1]
CUT = 1650556800000
d = load_kuairand_rich(version="pure", n_users=None, train_until=CUT, dense_dtype=torch.float32, with_signals=False)
O, Y = d["O"], d["Y"]
m, n = O.shape
users_raw, test, itf, lg = d["users_raw"], d["test"], d["item_feat"], d["log"]
X = O * Y
Z = X + 0.5 * (O - X)
B = ease(Z, 3000.0)
popv = std(torch.log1p(X.sum(0)))
tr = lg.groupby(["u", "i"]).is_click.max().reset_index()          # training pairs (impressed in-pool)
print(f"training pairs {len(tr)}, click rate {tr.is_click.mean():.3f}", flush=True)

# test candidate layout as in probe_27k.py
U = len(test)
Lmax = max(len(it) for _, it, _ in test)
cand = torch.zeros(U, Lmax, dtype=torch.long)
for k, (_, it, _) in enumerate(test):
    cand[k, :len(it)] = torch.as_tensor(it)
ctest = [(k, np.arange(len(it)), rel) for k, (_, it, rel) in enumerate(test)]
D = dict(ds="kuairand", ks=(5, 10, 20), key="ndcg@10", by="user", S=5, rows=torch.arange(U),
         row_of={k: k for k in range(U)})
D["splits"] = [split_test(ctest, 0.3, s, by="user") for s in range(5)]
test_rows = torch.tensor([u for u, _, _ in test])

# CF feature at training pairs and test candidates (row-standardised over all items, chunked)
cf_test = torch.zeros(U, Lmax)
cf_tr = np.zeros(len(tr), np.float32)
tr_u = tr.u.values
order = np.argsort(tr_u, kind="stable")
for a0 in range(0, m, 2048):
    S = rowstd(Z[a0:a0 + 2048] @ B)
    sel = (tr_u >= a0) & (tr_u < a0 + 2048)
    cf_tr[sel] = S[torch.as_tensor(tr_u[sel] - a0), torch.as_tensor(tr.i.values[sel])].numpy()
    tm = (test_rows >= a0) & (test_rows < a0 + 2048)
    idx = torch.nonzero(tm)[:, 0]
    if len(idx):
        cf_test[idx] = torch.gather(S[test_rows[idx] - a0], 1, cand[idx])
del Z, B, O, Y, X

L = np.load(f"{REPO}/data/raw/lifelong_27k.npz", allow_pickle=True)
tot = pd.DataFrame(L["tot_pre"], index=L["tot_pre_user"], columns=["n", "c", "l"])
base = (tot.c + 1.0) / (tot.n + 2.0)


def keystr(col, k):
    if k == "tag":
        return col.astype(str).str.split(",").str[0]
    return pd.to_numeric(col, errors="coerce").astype("Int64").astype(str)


tu = np.concatenate([np.full(len(it), u) for u, it, _ in test])
ti = np.concatenate([it for _, it, _ in test])
t_pos = np.concatenate([np.arange(len(it)) for _, it, _ in test])
t_row = np.concatenate([np.full(len(it), k) for k, (_, it, _) in enumerate(test)])
feats_test, feats_tr, names = [], [], []
for k, what in [("tag", "lift"), ("author_id", "cnt"), ("author_id", "lift"), ("music_id", "cnt")]:
    vals = pd.Index(L[f"{k}_values"])
    code_i = vals.get_indexer(keystr(itf[k], k))
    df = pd.DataFrame({"u_raw": L[f"{k}_pre_u"], "k": L[f"{k}_pre_k"]})
    df[["n", "c", "l"]] = L[f"{k}_pre_ncl"]
    b = df.u_raw.map(base).fillna(base.mean()).values
    df["v"] = (df.c + 5 * b) / (df.n + 5) - b if what == "lift" else np.log1p(df.n)
    qt = pd.DataFrame({"u_raw": users_raw[tu], "k": code_i[ti]}).merge(df[["u_raw", "k", "v"]], on=["u_raw", "k"], how="left")
    vt = qt.v.fillna(0.0).values.astype(np.float32)
    sc = vt[vt != 0].std() if (vt != 0).any() else 1.0              # same scaling as probe_27k.py
    St = torch.zeros(U, Lmax)
    St[torch.as_tensor(t_row), torch.as_tensor(t_pos)] = torch.as_tensor(vt / sc)
    qr = pd.DataFrame({"u_raw": users_raw[tr.u.values], "k": code_i[tr.i.values]}).merge(
        df[["u_raw", "k", "v"]], on=["u_raw", "k"], how="left")
    feats_test.append(St)
    feats_tr.append(qr.v.fillna(0.0).values.astype(np.float32) / sc)
    names.append(f"{k}_{what}")
pop_test = popv[cand]
pop_tr = popv[torch.as_tensor(tr.i.values)].numpy()

Ftr = torch.tensor(np.stack([cf_tr, pop_tr] + feats_tr, 1))
ytr = torch.tensor(tr.is_click.values.astype(np.float32))
res_lines = []
for label, cols in [("policy only (cf, pop)", [0, 1]), ("LEAF-fit (cf, pop, lifelong)", list(range(Ftr.shape[1])))]:
    F_ = Ftr[:, cols]
    mu, sd = F_.mean(0), F_.std(0).clamp_min(1e-6)
    w = torch.zeros(len(cols), requires_grad=True)
    b0 = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b0], max_iter=200)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(((F_ - mu) / sd) @ w + b0, ytr) + 1e-4 * w.pow(2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    coef = (w.detach() / sd)                                   # coefficients on the raw features
    allf = [cf_test, pop_test] + feats_test
    S = sum(coef[j] * allf[c] for j, c in enumerate(cols))
    sel = BucketSelector(D, label)
    sel.add(S, {"coef": [round(float(c), 4) for c in coef]})
    r = sel.result()
    names_ = ["cf", "pop"] + names
    line = (f"{label}: test {r['test_mean']:.4f} +- {r['test_sd']:.4f}; coefficients "
            f"{dict(zip([names_[c] for c in cols], [round(float(c), 4) for c in coef]))}")
    print(line, flush=True)
    res_lines.append(line)
    json.dump(r, open(out.replace(".txt", f"_{label.split()[0]}.json"), "w"))
with open(out, "a") as f:
    f.write("\n".join(res_lines) + "\n")
