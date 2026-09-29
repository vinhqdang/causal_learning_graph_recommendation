"""End-to-end construction of a propagation recommender from one config.

method in {"Obs", "IPS", "IPS+WC", "DR", "DRUP"}; the configuration keys are
those selected by experiments/run_filters.py (alpha, floor, lam, imp, cv, beta).

With ``Nuisance(..., K>1)`` the propensity and the imputation at a pair in
fold k are fitted on the other K-1 folds (cross-fitting over pairs), so they
do not use that pair's own exposure. This is weaker than Assumption 2 of the
paper: the nuisances still depend on the exposures of the other edges of a
walk, and the degree weights C, which sum the imputation (deg = 'Yhat') or the
edge estimates (deg = 'Wx') over a user's or an item's pairs, depend on the
log as well. experiments/mc_protocol.py measures the resulting bias.
"""

import torch

from .estimation import (baseline_imputation, clip_propensity, fold_ids, get_propensity,
                         lowrank_imputation, popularity_propensity, predict_propensity)
from .propagation import degree_weights, edge_estimate, three_hop


def _impute(O, Y, P, cfg, mask=None):
    Om = O if mask is None else O * mask
    if cfg.get("imp", "add") == "lr":
        return lowrank_imputation(Om, Y, P, rank=cfg.get("rank", 32), lam=cfg.get("lr_lam", 10.0),
                                  ridge=cfg["lam"])
    return baseline_imputation(Om, Y, P, lam=cfg["lam"])


class Nuisance:
    """Propensities (raw, before clipping) and cross-fitted imputations.

    K <= 1: no cross-fitting (nuisances fitted on the whole log).
    K > 1 : pairs are split into K random folds; for a pair in fold k the
            propensity model and the imputation are both fitted on the pairs
            outside fold k, so the nuisance at a pair does not use that
            pair's exposure and outcome (it still uses the other pairs).
    """

    def __init__(self, d, O, Y, prop, K=0, seed=0):
        self.O, self.Y, self.prop, self.K = O, Y, prop, K
        self.given = prop == "given" and d.get("P_given") is not None
        if K > 1:
            self.folds = fold_ids(O.shape, K, seed)
            self.masks = lambda k: (self.folds != k).to(O.dtype)
            if self.given:
                self.P_folds = None
                self.P = d["P_given"].to(O.dtype)
            else:
                # store the fitted parameters of each fold's exposure model
                # (not the m x n matrices) and rebuild them on demand
                self.P_folds = []
                self.P = torch.empty_like(O)
                for k in range(K):
                    Pk, params = self._prop(d, self.masks(k))
                    self.P_folds.append(params)
                    ins = self.folds == k
                    self.P[ins] = Pk[ins]
                    del Pk
        else:
            self.folds = None
            self.P = get_propensity(d, O, Y, prop)
        self._cache = {}
        self.deg_folds = self.folds if self.folds is not None else fold_ids(O.shape, 10, seed)

    def _prop(self, d, mask):
        if self.prop == "pop":
            P, _, params = popularity_propensity(self.O, mask=mask, return_params=True)
            return P, params
        raise ValueError("cross-fitting is implemented for prop in {given, pop}")

    def fold_propensity(self, k):
        if self.P_folds is None:
            return self.P
        return predict_propensity(self.P_folds[k], self.O.shape, self.O.dtype)

    def imputation(self, floor, cfg):
        key = (floor, cfg["lam"], cfg.get("imp", "add"))
        if key in self._cache:
            return self._cache[key]
        self._cache.clear()
        O, Y = self.O, self.Y
        if self.folds is None:
            Yh = _impute(O, Y, clip_propensity(self.P, floor), cfg)
        else:
            Yh = torch.empty_like(O)
            for k in range(self.K):
                Pk = self.fold_propensity(k)
                est = _impute(O, Y, clip_propensity(Pk, floor), cfg, mask=self.masks(k))
                del Pk
                ins = self.folds == k
                Yh[ins] = est[ins]
                del est
        self._cache[key] = Yh
        return Yh


def as_nuisance(d, O, Y, P_or_nuis):
    if isinstance(P_or_nuis, Nuisance):
        return P_or_nuis
    nz = Nuisance.__new__(Nuisance)
    nz.O, nz.Y, nz.K, nz.folds, nz.P_folds, nz.P, nz._cache = O, Y, 0, None, None, P_or_nuis, {}
    nz.deg_folds = fold_ids(O.shape, 10, 0)
    return nz


def crossfit_degree_weights(W, alpha, folds, floor=1.0):
    """Degree weights from cross-fitted edge estimates: at a pair of fold k,
    d_u and d_i are the row and column sums of W over the other folds
    (rescaled by K / (K - 1)). C at a pair does not use that pair's exposure
    directly, but W over the other folds contains cross-fitted imputations that do."""
    K = int(folds.max()) + 1
    C = torch.empty_like(W)
    for k in range(K):
        out = folds != k
        Wm = torch.where(out, W, torch.zeros((), dtype=W.dtype))
        du = (Wm.sum(1) * K / (K - 1)).clamp_min(floor)
        di = (Wm.sum(0) * K / (K - 1)).clamp_min(floor)
        ins = ~out
        C[ins] = (du.pow(-alpha)[:, None] * di.pow(-(1.0 - alpha))[None, :])[ins]
        del Wm
    return C


def edge_weights(nz, W, Ydeg, alpha, deg):
    """deg = 'Yhat': degrees from the imputation; 'Wx': cross-fitted degrees
    from the edge estimates (see crossfit_degree_weights)."""
    if deg == "Wx":
        return crossfit_degree_weights(W, alpha, nz.deg_folds)
    return degree_weights(W, alpha, D=Ydeg)


def impute(O, Y, P, cfg):
    """Outcome imputation on the whole log: additive ('add') or low-rank ('lr')."""
    return _impute(O, Y, P, cfg)


def build(O, Y, P_raw, method, cfg, Yhat=None):
    """P_raw is a propensity tensor (no cross-fitting) or a Nuisance.
    For IPS / IPS+WC the imputation is used only for the degree weights."""
    if method == "Obs":
        P = torch.ones_like(O)
        W = O * Y
        C = degree_weights(W, cfg["alpha"])
        return {"method": method, "cfg": cfg, "P": P, "Yhat": None, "Ydeg": None, "W": W, "C": C,
                "correct": False}
    nz = as_nuisance(None, O, Y, P_raw)
    P = clip_propensity(nz.P, cfg["floor"])
    Ydeg = Yhat if Yhat is not None else nz.imputation(cfg["floor"], dict(cfg, imp=cfg.get("imp", "add")))
    Yh = Ydeg if method.startswith("DR") else None
    W = edge_estimate(O, Y, P, Yh, cfg.get("cv", 1.0) if Yh is not None else 1.0)
    C = edge_weights(nz, W, Ydeg, cfg["alpha"], cfg.get("deg", "Yhat"))
    return {"method": method, "cfg": cfg, "P": P, "Yhat": Yh, "Ydeg": Ydeg, "W": W, "C": C,
            "correct": method in ("IPS+WC", "DRUP")}


def raw_parts(M, rows):
    s1 = (M["C"] * M["W"])[rows]
    s3 = three_hop(M["W"], M["C"], rows=rows, correct=M["correct"])
    return s1, s3


def scale_constants(M, rows):
    """Global (not per-user) scale constants so that the final score is a
    fixed linear combination a * s1 + b * s3 of the 1-hop and 3-hop terms.
    Fixed constants keep every score affine in the user's own row (needed for
    exact explanations) and make the score a fixed function of the edge
    estimates; beta then sets b / a in units of the terms' mean magnitudes."""
    s1, s3 = raw_parts(M, rows)
    return float(s1.abs().mean().clamp_min(1e-12)), float(s3.abs().mean().clamp_min(1e-12))


def combine(s1, s3, beta, consts):
    if beta >= 1e3:
        return s3 / consts[1]
    return s1 / consts[0] + beta * s3 / consts[1]


def scores(M, rows, consts=None):
    s1, s3 = raw_parts(M, rows)
    if consts is None:
        consts = (float(s1.abs().mean().clamp_min(1e-12)), float(s3.abs().mean().clamp_min(1e-12)))
    return combine(s1, s3, M["cfg"].get("beta", 1.0), consts)


def scores_with_consts(M, rows):
    """scores(M, rows, scale_constants(M, rows)) with a single propagation."""
    s1, s3 = raw_parts(M, rows)
    c1 = float(s1.abs().mean().clamp_min(1e-12))
    c3 = float(s3.abs().mean().clamp_min(1e-12))
    return combine(s1, s3, M["cfg"].get("beta", 1.0), (c1, c3)), (c1, c3)
