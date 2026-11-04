"""Probe C: impression-level response model with side information and scenario (tab) adjustment,
predicting the response to a uniformly inserted video in the test scenario.

score(u, i) = logit P(click | impressed, u, i, tab = 1)
            = c + b_u + b_i + b_author(i) + b_tag(i) + b_dur(i) + t_tab
              + <p_u + W x_u, q_i + q_author(i) + q_tag(i)>
trained with binary cross-entropy on every standard impression (shown-not-clicked = 0), optionally with
multi-behaviour heads (long_view, like) sharing all embeddings, optionally on the logs of all users.
Ranking uses tab = 1 (99.3% of random insertions). Selection over (cfg, epoch) on validation per split.
python3 experiments/v9_probes/probe_ctr.py OUT.txt [--aux] [--quick]
"""
import argparse
import itertools
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from common import Selector, load, report, std

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--aux", action="store_true", help="also train on the standard logs of the other 17K users")
ap.add_argument("--epochs", type=int, default=6)
ap.add_argument("--grid", default="main")
ap.add_argument("--threads", type=int, default=2)
a = ap.parse_args()
torch.set_num_threads(a.threads)
D = load(extra_users=a.aux)
d, O, Y, rows = D["d"], D["O"], D["Y"], D["rows"]
m, n = O.shape
del D["O"], D["Y"]
pop = std(torch.log1p((O * Y).sum(0)))[None, :]
from common import ease, rowstd  # noqa: E402
_X = O * Y
SX = rowstd(_X[rows] @ ease(_X, 3000.0))           # click channel (EASE)
SO = rowstd(O[rows] @ ease(O, 3000.0))             # impression channel (EASE)
del O, Y, _X
itf = d["item_feat"]
auth = torch.as_tensor(itf.author_id.astype("category").cat.codes.values.astype(np.int64) + 1)
tag = torch.as_tensor(itf.tag.astype("category").cat.codes.values.astype(np.int64) + 1)
dur = torch.as_tensor(pd.qcut(itf.video_duration.fillna(itf.video_duration.median()), 10, labels=False,
                              duplicates="drop").values.astype(np.int64))
lg = d["log"]
uu, ii, tt = [torch.as_tensor(lg[c].values.astype(np.int64)) for c in ("u", "i", "tab")]
ys = {k: torch.as_tensor(lg[k].values.astype(np.float32)) for k in ("is_click", "long_view", "is_like")}
m_all = m
if a.aux:
    ax = d["aux"]
    uu = torch.cat([uu, torch.as_tensor(ax.u_aux.values.astype(np.int64)) + m])
    ii = torch.cat([ii, torch.as_tensor(ax.i.values.astype(np.int64))])
    tt = torch.cat([tt, torch.as_tensor(ax.tab.values.astype(np.int64))])
    for k in ys:
        ys[k] = torch.cat([ys[k], torch.as_tensor(ax[k].values.astype(np.float32))])
    m_all = m + int(ax.u_aux.max()) + 1
Xu = torch.as_tensor(d["user_feat"])                      # sample users only; aux users get zeros
Xu = torch.cat([Xu, torch.zeros(m_all - m, Xu.shape[1])]) if m_all > m else Xu
N = len(uu)
print(f"impressions {N}, users {m_all}, items {n}", flush=True)


class RM(torch.nn.Module):
    def __init__(self, dim, side, ufeat, heads):
        super().__init__()
        self.side, self.ufeat = side, ufeat
        E = lambda k, dd: torch.nn.Embedding(k, dd)
        self.bu, self.bi, self.ba, self.bt, self.bd = E(m_all, 1), E(n, 1), E(int(auth.max()) + 1, 1), \
            E(int(tag.max()) + 1, 1), E(int(dur.max()) + 1, 1)
        self.tab = E(16, heads)
        self.p, self.q = E(m_all, dim), E(n, dim)
        self.qa, self.qt = E(int(auth.max()) + 1, dim), E(int(tag.max()) + 1, dim)
        self.W = torch.nn.Linear(Xu.shape[1], dim, bias=False)
        self.head = torch.nn.Parameter(torch.zeros(heads))     # per-head intercept
        self.hs = torch.nn.Parameter(torch.ones(heads))        # per-head scale on the shared logit
        for e in (self.p, self.q, self.qa, self.qt):
            torch.nn.init.normal_(e.weight, std=0.01)
        for e in (self.bu, self.bi, self.ba, self.bt, self.bd, self.tab):
            torch.nn.init.zeros_(e.weight)
        torch.nn.init.normal_(self.W.weight, std=0.01)

    def item_parts(self, i):
        b = self.bi(i)[:, 0]
        q = self.q(i)
        if self.side:
            b = b + self.ba(auth[i])[:, 0] + self.bt(tag[i])[:, 0] + self.bd(dur[i])[:, 0]
            q = q + self.qa(auth[i]) + self.qt(tag[i])
        return b, q

    def user_parts(self, u):
        p = self.p(u)
        if self.ufeat:
            p = p + self.W(Xu[u])
        return self.bu(u)[:, 0], p

    def forward(self, u, i, t):
        bu, p = self.user_parts(u)
        bi, q = self.item_parts(i)
        z = bu + bi + (p * q).sum(1)
        return self.head[None, :] + self.hs[None, :] * z[:, None] + self.tab(t)

    @torch.no_grad()
    def scores(self, rows_, tab1=True):
        bu, p = self.user_parts(rows_)
        it = torch.arange(n)
        bi, q = self.item_parts(it)
        return bi[None, :] + p @ q.T     # tab and user terms are constant within a user's ranking


COMB = {}


def run(cfg, sel):
    torch.manual_seed(0)
    heads = 3 if cfg["mt"] else 1
    model = RM(cfg["dim"], cfg["side"], cfg["uf"], heads)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    Yt = torch.stack([ys["is_click"], ys["long_view"], ys["is_like"]], 1)[:, :heads]
    wts = torch.tensor([1.0, 0.5, 0.5])[:heads]
    bs = 8192
    for ep in range(a.epochs):
        perm = torch.randperm(N)
        for k in range(0, N, bs):
            b = perm[k:k + bs]
            t = tt[b] if cfg["tab"] else torch.ones_like(tt[b])
            z = model(uu[b], ii[b], t)
            loss = (F.binary_cross_entropy_with_logits(z, Yt[b], reduction="none") * wts).sum(1).mean()
            reg = cfg["wd"] * (model.p(uu[b]).pow(2).sum() + model.q(ii[b]).pow(2).sum()) / len(b)
            opt.zero_grad()
            (loss + reg).backward()
            opt.step()
        S = std(model.scores(rows))
        for beta in cfg["betas"]:
            sel.add(S + beta * pop, dict(cfg, betas=None, beta=beta, epoch=ep))
        cs = COMB.setdefault(sel.name, Selector(D, sel.name + "+EASE(click)+EASE(imp)"))
        for h, g, beta in itertools.product((0.25, 0.5, 1.0), (1.0, 2.0), (0.0, 0.25)):
            cs.add(h * S + SX + g * SO + beta * pop, dict(cfg, betas=None, h=h, g=g, beta=beta, epoch=ep))
        print(cfg, ep, "val", round(float(np.mean([b[0] for b in sel.best])), 4), flush=True)


GRIDS = {
    "main": [dict(dim=dd, lr=3e-3, wd=wd, side=sd, uf=uf, tab=tb, mt=mt, betas=(0.0, 0.5))
             for dd, wd, sd, uf, tb, mt in [(32, 1e-4, True, False, False, False),    # + item side info
                                            (32, 1e-4, True, True, False, False),     # + user features
                                            (32, 1e-4, True, True, True, False),      # + scenario (tab) adjustment
                                            (32, 1e-4, True, True, True, True)]],      # + multi-behaviour heads
}
tag_ = "aux" if a.aux else "10k"
variants = {}
for cfg in GRIDS[a.grid]:
    name = ("RM" + ("+side" if cfg["side"] else "") + ("+uf" if cfg["uf"] else "") +
            ("+tab" if cfg["tab"] else "") + ("+mt" if cfg["mt"] else "") + f"[{tag_}]")
    variants.setdefault(name, []).append(cfg)
allsel = Selector(D, f"RM-all[{tag_}]")
for name, cfgs in variants.items():
    sel = Selector(D, name)
    for cfg in cfgs:
        run(cfg, sel)
    report(D, name, sel.result(), a.out)
    report(D, COMB[name].name, COMB[name].result(), a.out)
    for s, b in enumerate(sel.best):
        if b[0] > allsel.best[s][0]:
            allsel.best[s] = b
    allsel.n += sel.n
report(D, f"RM-best-of-variants[{tag_}]", allsel.result(), a.out)
