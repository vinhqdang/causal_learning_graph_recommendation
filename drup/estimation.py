"""Nuisance models: exposure propensities and outcome imputation."""

import torch


def _logit(x, eps=1e-6):
    x = x.clamp(eps, 1 - eps)
    return torch.log(x) - torch.log1p(-x)


def popularity_propensity(O, iters=50, ridge=1e-6):
    """Logistic exposure model p_ui = sigmoid(a + b*logit(r_u) + c*logit(r_i)).

    r_u / r_i are the user / item exposure rates. Fitted by Newton's method on
    all m*n pairs using row/column reductions (no (m, n, k) design tensor).
    A feature with no variation (e.g. every Coat user rated 24 items) is
    dropped, since it is collinear with the intercept.
    """
    dt = O.dtype
    feats = []  # (vector, axis) with axis 0 = user feature, 1 = item feature
    for vec, ax in ((_logit(O.mean(1)), 0), (_logit(O.mean(0)), 1)):
        if vec.std() > 1e-9:
            feats.append(((vec - vec.mean()) / vec.std(), ax))
    k = 1 + len(feats)
    theta = torch.zeros(k, dtype=torch.float64)
    theta[0] = _logit(O.mean()).item()

    def linpred(th):
        z = torch.full(O.shape, th[0].item(), dtype=dt)
        for c, (v, ax) in enumerate(feats, start=1):
            z = z + th[c].to(dt) * (v[:, None] if ax == 0 else v[None, :])
        return z

    for _ in range(iters):
        p = torch.sigmoid(linpred(theta))
        r = O - p
        w = p * (1 - p)
        red = {0: (r.sum(1), w.sum(1)), 1: (r.sum(0), w.sum(0))}
        g = torch.zeros(k, dtype=torch.float64)
        H = torch.zeros(k, k, dtype=torch.float64)
        g[0], H[0, 0] = r.sum(), w.sum()
        for a_, (va, axa) in enumerate(feats, start=1):
            g[a_] = (red[axa][0] * va).sum()
            H[0, a_] = H[a_, 0] = (red[axa][1] * va).sum()
            for b_, (vb, axb) in enumerate(feats, start=1):
                if axa == axb:
                    H[a_, b_] = (red[axa][1] * va * vb).sum()
                else:
                    vu = va if axa == 0 else vb
                    vi = vb if axa == 0 else va
                    H[a_, b_] = (vu[:, None] * w * vi[None, :]).sum()
        step = torch.linalg.solve(H + ridge * torch.eye(k, dtype=torch.float64), g)
        theta = theta + step
        if step.abs().max() < 1e-8:
            break
    return torch.sigmoid(linpred(theta)), theta


def baseline_imputation(O, Y, P, lam=5.0, iters=20, ips=True):
    """Additive imputation Yhat_ui = mu + b_u + b_i fitted on exposed pairs.

    Optionally inverse-propensity weighted so that the fit targets the
    full-population outcome surface. Closed-form alternating ridge updates.
    """
    Wt = O / P if ips else O.clone()
    Yo = O * Y
    mu = (Wt * Yo).sum() / Wt.sum()
    bu = torch.zeros(O.shape[0], dtype=O.dtype)
    bi = torch.zeros(O.shape[1], dtype=O.dtype)
    for _ in range(iters):
        res = Yo - mu - bi[None, :]
        bu = (Wt * res).sum(1) / (Wt.sum(1) + lam)
        res = Yo - mu - bu[:, None]
        bi = (Wt * res).sum(0) / (Wt.sum(0) + lam)
    return (mu + bu[:, None] + bi[None, :]).clamp(0.0, 1.0)


def clip_propensity(P, floor):
    return P.clamp(min=floor, max=1.0)
