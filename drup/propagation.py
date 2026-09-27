"""Walk-corrected, doubly robust graph propagation (DRUP).

Setting
-------
Users u = 1..m, items i = 1..n. Every pair has a potential outcome
Y_ui in {0, 1} (would the user like the item if it were shown) and an
exposure indicator O_ui ~ Bernoulli(p_ui), independent across pairs given the
propensities (unconfoundedness). Only O and O * Y are observed.

The quantity a graph recommender wants to propagate over is the *full*
preference graph Y, not the exposure-filtered graph O * Y. For a fixed edge
weighting C (degree normalisation), the target 3-hop user->item signal is

    T*_ui = sum_{j, v} C_uj Y_uj  C_vj Y_vj  C_vi Y_vi .

Edge-level estimators
---------------------
    IPS : W_e = O_e Y_e / p_e
    DR  : W_e = Yhat_e + O_e (Y_e - Yhat_e) / p_e

Both satisfy E[W_e] = Y_e, so a product of W over *distinct* edges is unbiased
for the product of Y (independence across edges). A walk that traverses the
same edge k > 1 times, however, contributes W_e^k whose expectation is not
Y_e^k = Y_e (Y is binary): for IPS E[W_e^2] = Y_e / p_e. Plugging W into a
standard propagation operator (NAVIP-style inverse-propensity adjacency, or a
DR-imputed adjacency) is therefore biased for every walk that backtracks, and
the bias grows like 1/p_e -- it is largest exactly on the rarely exposed
edges that debiasing is supposed to recover.

Because Y_e^k = Y_e, the unbiased replacement for W_e^k is simply W_e
(times the deterministic weight C_e^k). This module implements that
"idempotent walk correction" exactly for the 3-hop operator and for the
item-item Gram operator, using only row/column sums, so the cost equals the
cost of the uncorrected operator.
"""

from __future__ import annotations

import numpy as np
import torch


def edge_estimate(O, Y, P, Yhat=None):
    """Edge-level unbiased estimate W of the potential-outcome matrix Y.

    O, Y, P, Yhat are dense (m, n) tensors. Y may hold anything where O == 0.
    """
    Yobs = O * Y
    if Yhat is None:
        return Yobs / P
    return Yhat + O * (Yobs - Yhat) / P


def degree_weights(W, alpha=0.5, floor=1.0, D=None):
    """Symmetric degree normalisation C_ui = d_u^-alpha d_i^-(1-alpha).

    Degrees are row/column sums of D (default: W, which is unbiased for the
    degrees of Y). Passing D = Yhat makes C independent of the exposures, in
    which case the walk-corrected estimator is exactly unbiased; with D = W it
    is exact conditionally on the degrees and consistent as the graph grows.
    Degrees are floored at ``floor`` because DR edge estimates can be negative.
    """
    D = W if D is None else D
    du = D.sum(1).clamp_min(floor)
    di = D.sum(0).clamp_min(floor)
    return du.pow(-alpha)[:, None] * di.pow(-(1.0 - alpha))[None, :]


def three_hop(W, C, rows=None, correct=True):
    """3-hop user->item propagation over the weighted graph Wt = C * W.

    If ``correct`` is True, every walk u-j-v-i that repeats an edge uses the
    idempotent replacement, giving an estimator that is exactly unbiased for
    T* conditionally on C. Otherwise it returns the plug-in (Wt Wt^T Wt)_ui.

    rows: optional index tensor of the users to score (saves memory).
    """
    Wt = C * W
    Wr = Wt if rows is None else Wt[rows]
    naive = (Wr @ Wt.T) @ Wt
    if not correct:
        return naive
    # Excess of a doubly-traversed edge over its unbiased replacement.
    # E[C^2 W^2] != C^2 Y, while E[C^2 W] = C^2 Y.
    delta = Wt * Wt - C * C * W                       # (m, n)
    row_delta = delta.sum(1, keepdim=True)            # sum_j delta_uj
    col_delta = delta.sum(0, keepdim=True)            # sum_v delta_vi
    dr = delta if rows is None else delta[rows]
    Wr_ = Wr
    Cr = C if rows is None else C[rows]
    Wraw = W if rows is None else W[rows]
    rd = row_delta if rows is None else row_delta[rows]
    # (b) v = u, j != i :  Wt_uj^2 Wt_ui  ->  C_uj^2 W_uj Wt_ui
    corr_b = Wr_ * (rd - dr)
    # (c) v != u, j = i :  Wt_ui Wt_vi^2  ->  Wt_ui C_vi^2 W_vi
    corr_c = Wr_ * (col_delta - dr)
    # (d) v = u, j = i :  Wt_ui^3  ->  C_ui^3 W_ui
    corr_d = Wr_ ** 3 - Cr ** 3 * Wraw
    return naive - corr_b - corr_c - corr_d


def item_gram(W, C, correct=True):
    """Item-item co-preference operator G = Wt^T Wt (2-hop, item side).

    Off-diagonal entries are products of distinct edges and already
    unbiased; the diagonal sum_u Wt_ui^2 is replaced by sum_u C_ui^2 W_ui.
    """
    Wt = C * W
    G = Wt.T @ Wt
    if correct:
        diag = (C * C * W).sum(0)
        G = G - torch.diag(torch.diagonal(G)) + torch.diag(diag)
    return G


def drup_scores(O, Y, P, Yhat=None, alpha=0.5, beta=1.0, rows=None,
                correct=True):
    """DRUP score = 1-hop (direct DR/IPS estimate) + beta * corrected 3-hop.

    The 1-hop term is the edge estimate itself (the user's own debiased
    signal on the item); the 3-hop term transfers preference along
    user-item-user-item walks of the *counterfactual full-exposure* graph.
    Each term is scaled to unit mean absolute value per user so that beta is
    a scale-free mixing weight.
    """
    W = edge_estimate(O, Y, P, Yhat)
    C = degree_weights(W, alpha)
    s1 = (C * W) if rows is None else (C * W)[rows]
    s3 = three_hop(W, C, rows=rows, correct=correct)
    s1 = s1 / s1.abs().mean(1, keepdim=True).clamp_min(1e-12)
    s3 = s3 / s3.abs().mean(1, keepdim=True).clamp_min(1e-12)
    return s1 + beta * s3


def to_tensor(x, dtype=torch.float64):
    if isinstance(x, torch.Tensor):
        return x.to(dtype)
    return torch.as_tensor(np.asarray(x), dtype=dtype)


def local_three_hop(w_u, c_u, G_corr, rows_u_W=None):
    """Corrected 3-hop score of one or more users from the *public* item
    operator G_corr = item_gram(W, C, correct=True) and the users' own rows.

    s_u = wt_u G_corr - wt_u * (sum_j delta_uj - 2 delta_u) - (wt_u^3 - c_u^3 w_u)
    with wt_u = c_u * w_u and delta_u = wt_u^2 - c_u^2 w_u. Everything except
    G_corr is local to the user, so G_corr is the only object that has to be
    shared (and privatised, see drup.fat.dp_item_operator).
    """
    wt = c_u * w_u
    delta = wt * wt - c_u * c_u * w_u
    return (wt @ G_corr - wt * (delta.sum(-1, keepdim=True) - 2 * delta)
            - (wt ** 3 - c_u ** 3 * w_u))
