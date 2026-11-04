"""Probe S: which KuaiRand signals add to a tuned closed-form backbone (EASE + popularity offset)?
All variants selected per split on validation; paired against stored LightGCN / r-AdjNorm / QP rows.
python3 experiments/v9_probes/probe_signals.py OUT.txt [variant ...]"""
import itertools
import sys

import torch

from common import Selector, ease, ease_gram, load, report, rowstd, std

torch.set_num_threads(4)
out = sys.argv[1]
which = sys.argv[2:] or ["base", "mb", "imp", "aux", "author"]
UNTIL = 1650556800000 if "until" in which else None     # 2022-04-22 00:00 UTC+8: temporal protocol
D = load(extra_users=("aux" in which), train_until=UNTIL)
if UNTIL is not None:
    out = out.replace(".txt", "_until0422.txt")
d, O, Y, rows = D["d"], D["O"], D["Y"], D["rows"]
sig = d["signals"]
for k in ("n_imp", "play_ratio", "is_hate"):
    sig.pop(k, None)
d["O"] = d["Y"] = None
X = (O * Y)
pop = std(torch.log1p(X.sum(0)))[None, :]
LAMS = (1000.0, 3000.0, 10000.0)
BETAS = (0.0, 0.25, 0.5, 1.0)


def sweep(name, make, grid):
    sel = Selector(D, name)
    for g in grid:
        P = make(**g)
        for b in BETAS:
            sel.add(P + b * pop, dict(g, beta=b))
    report(D, name, sel.result(), out)


cache = {}


def ease_rows(Xm, lam, key):
    k = (key, lam)
    if key not in ("click", "imp"):          # keep memory bounded: cache only the reused channels
        return rowstd((Xm[rows] @ ease(Xm, lam)))
    if k not in cache:
        cache[k] = rowstd((Xm[rows] @ ease(Xm, lam)))
    return cache[k]


if "base" in which:
    sweep("EASE+pop", lambda lam: ease_rows(X, lam, "click"), [dict(lam=l) for l in LAMS])
if "mb" in which:
    eng = torch.clamp(sig["is_like"] + sig["is_follow"] + sig["is_comment"] + sig["is_forward"] +
                      sig["is_profile_enter"], max=1.0)
    def mk(lam, a, b):
        Xm = X + a * sig["long_view"] + b * eng
        return ease_rows(Xm, lam, f"mb{a}_{b}")
    sweep("EASE-multibehaviour+pop", mk, [dict(lam=l, a=a, b=b) for l, a, b in itertools.product(LAMS, (0.0, 1.0), (0.0, 1.0, 3.0))])
if "imp" in which:
    def mk(lam, g):
        return ease_rows(X, lam, "click") + g * ease_rows(O, lam, "imp")
    sweep("EASE+impressions+pop", mk, [dict(lam=l, g=g) for l, g in itertools.product(LAMS, (0.0, 0.5, 1.0))])
if "aux" in which:
    # item-item model fitted on the standard log of all 27K users (aux users' random logs never loaded)
    ax = d["aux"]
    ax = ax[ax.is_click > 0].drop_duplicates(["u_aux", "i"])
    Xa = torch.zeros(int(ax.u_aux.max()) + 1, O.shape[1])
    Xa[torch.as_tensor(ax.u_aux.values), torch.as_tensor(ax.i.values)] = 1.0
    G = X.T @ X + Xa.T @ Xa
    popa = std(torch.log1p(X.sum(0) + Xa.sum(0)))[None, :]
    sel = Selector(D, "EASE-allusers+popall")
    for lam in LAMS:
        S = rowstd(X[rows] @ ease_gram(G, lam))
        for b in BETAS:
            sel.add(S + b * popa, dict(lam=lam, beta=b))
    report(D, "EASE-allusers+popall", sel.result(), out)
if "author" in which:
    itf = d["item_feat"]
    a_codes = torch.as_tensor(itf.author_id.astype("category").cat.codes.values.astype("int64"))
    A = torch.zeros(O.shape[1], int(a_codes.max()) + 1)
    A[torch.arange(O.shape[1]), a_codes] = 1.0
    UA = X @ A                                   # clicks per (user, author)
    Sa = rowstd((UA @ A.T)[rows])               # user's affinity to the item's author
    def mk(lam, c):
        return ease_rows(X, lam, "click") + c * Sa
    sweep("EASE+author+pop", mk, [dict(lam=l, c=c) for l, c in itertools.product(LAMS, (0.0, 0.25, 0.5))])
if "imp2" in which:
    # follow-up: is the gain from impressions as such, from the shown-not-clicked part, or from the weight?
    NC = O - X                                   # impressed, not clicked
    def mk(lam, g):
        return ease_rows(X, lam, "click") + g * ease_rows(O, lam, "imp")
    sweep("EASE+impressions(g>=1)+pop", mk, [dict(lam=l, g=g) for l, g in itertools.product((3000.0, 10000.0), (1.0, 2.0, 4.0))])
    sweep("EASE(impressions only)+pop", lambda lam: ease_rows(O, lam, "imp"), [dict(lam=l) for l in (1000.0, 3000.0, 10000.0)])
    def mk2(lam, g):
        return ease_rows(X, lam, "click") + g * rowstd(X[rows] @ ease(NC, lam) * 0 + NC[rows] @ ease(NC, lam))
    sweep("EASE+notclicked-channel+pop", mk2, [dict(lam=l, g=g) for l, g in itertools.product((3000.0,), (0.5, 1.0, 2.0))])
    def mk3(lam, w):
        Xm = X + w * NC
        return rowstd(Xm[rows] @ ease(Xm, lam))
    sweep("EASE(clicks + w*notclicked)+pop", mk3, [dict(lam=l, w=w) for l, w in itertools.product((3000.0,), (0.25, 0.5, 1.0))])
