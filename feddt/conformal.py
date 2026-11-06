"""Drift-weighted federated conformal calibration, DW-FCC (paper Algorithm 2).

Split conformal prediction assumes calibration and test data are exchangeable. When a
vehicle drifts (hotter working face, new duty cycle) this fails. Weighted conformal
prediction (Tibshirani et al. 2019) restores coverage when calibration scores are weighted
by the likelihood ratio w(x) = dQ/dP between test and calibration covariates.

DW-FCC estimates w with an exponential tilt of random-Fourier-feature embeddings,
    w_i = exp(eta * <psi_i, Delta>),   Delta = mu_Q - mu_P,
choosing eta by bisection so that the weighted calibration mean embedding moves onto the
target mean embedding along Delta, capped so the effective sample size stays >= ess_min * m.
Every quantity the server needs is a secure sum of per-vehicle scalars or histograms.
"""
from __future__ import annotations

import math

import numpy as np


def tilt_eta(proj, target, ess_min=0.2, eta_max=50.0, iters=60):
    """Solve sum w_i proj_i / sum w_i = target for eta >= 0 with w_i = exp(eta * proj_i).
    proj: <psi_i, Delta> for all calibration points; target: <mu_Q, Delta>.
    Returns eta, capped so that ESS = (sum w)^2 / sum w^2 >= ess_min * m."""
    proj = np.asarray(proj, float)
    m = len(proj)

    def stats(eta):
        a = eta * (proj - proj.max())
        w = np.exp(a)
        return float(np.sum(w * proj) / np.sum(w)), float(np.sum(w) ** 2 / np.sum(w ** 2))

    if stats(0.0)[0] >= target:
        return 0.0
    lo, hi = 0.0, eta_max
    if stats(hi)[0] < target:
        lo = hi
    else:
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if stats(mid)[0] < target:
                lo = mid
            else:
                hi = mid
    eta = lo if stats(lo)[0] >= target else hi
    # effective-sample-size cap
    if stats(eta)[1] < ess_min * m:
        a, b = 0.0, eta
        for _ in range(iters):
            mid = 0.5 * (a + b)
            if stats(mid)[1] >= ess_min * m:
                a = mid
            else:
                b = mid
        eta = a
    return float(eta)


def weighted_conformal_quantile(scores, weights, w_test, a=0.05):
    """Quantile of sum_i p_i delta_{s_i} + p_test delta_{+inf} at level 1 - a (Tibshirani et al. 2019)."""
    s = np.asarray(scores, float)
    w = np.asarray(weights, float)
    order = np.argsort(s)
    s, w = s[order], w[order]
    tot = w.sum() + w_test
    cdf = np.cumsum(w) / tot
    j = int(np.searchsorted(cdf, 1 - a))
    return float(s[j]) if j < len(s) else float("inf")


def federated_weighted_quantile(client_hists, edges, w_test_mass, a=0.05):
    """Server side: client_hists are weighted score histograms (same edges), secure-summed."""
    h = np.sum(client_hists, axis=0)
    tot = h.sum() + w_test_mass
    cdf = np.cumsum(h) / tot
    j = int(np.searchsorted(cdf, 1 - a))
    return float(edges[j + 1]) if j < len(h) else float("inf")


def rff(X, D=128, bandwidth=1.0, seed=0):
    g = np.random.default_rng(seed)
    W = g.normal(0, 1.0 / bandwidth, (X.shape[1], D // 2))
    Z = X @ W
    return np.concatenate([np.cos(Z), np.sin(Z)], 1) / math.sqrt(D // 2)
