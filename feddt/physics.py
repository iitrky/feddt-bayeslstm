"""Federated physics-guided prior (paper Sect. 4.2, Fig. 2).

Semi-empirical capacity-fade law (Arrhenius in temperature, power law in throughput) of
Wang et al. (2011):
    L(Ah) = B * exp(-(Ea - Ea_c * c) / (R * T)) * Ah^z,       SoH = SoH_0 * (1 - L)
Each vehicle fits (log k, z) on its own history, where k = B * exp(-(Ea - Ea_c c)/(R T)).
The server pools these per chemistry. If a chemistry cluster spans different temperatures,
Ea and log B are re-estimated from the clients' (1/T, log k) pairs; otherwise the literature
value is kept. Only a few scalars per vehicle leave the vehicle.
The LSTM then learns the residual between observed SoH and this prior.
"""
from __future__ import annotations

import numpy as np

R_GAS = 8.314
DEFAULT_EA = 31700.0      # J/mol, Wang et al. (2011)
DEFAULT_EA_C = 370.3      # J/mol per C-rate, Wang et al. (2011)
DEFAULT_Z = 0.55
T_INDEX, C_INDEX = 7, 11  # mean temperature and mean C-rate health indicators


def cell_series(rec, ambient_c=25.0):
    """Per-cycle temperature [K], C-rate and cumulative discharge throughput [Ah]."""
    his = rec.his
    T = his[:, T_INDEX] if np.isfinite(his[:, T_INDEX]).mean() > 0.5 else np.full(len(rec.soh), np.nan)
    T = np.where(np.isfinite(T), T, rec.meta.get("ambient_c", ambient_c)) + 273.15
    c = his[:, C_INDEX] if np.isfinite(his[:, C_INDEX]).mean() > 0.5 else np.full(len(rec.soh), 1.0)
    c = np.where(np.isfinite(c), c, 1.0)
    block = rec.meta.get("block", 1)
    dah = rec.soh * rec.rated_ah * block
    ah = np.cumsum(dah)
    return T, c, ah, dah


def fit_local(rec, ea=DEFAULT_EA, ea_c=DEFAULT_EA_C, upto=None):
    """Fit SoH = s0 * (1 - k * Ah^z) on the vehicle's own history (first `upto` cycles).
    Grid search over z with an exact linear least-squares fit of (s0, s0*k) inside, so the
    initial SoH is estimated jointly rather than taken from noisy early cycles."""
    T, c, ah, _ = cell_series(rec)
    n = len(rec.soh) if upto is None else min(max(int(upto), 3), len(rec.soh))
    y, x = rec.soh[:n], ah[:n]
    base = {"T": float(np.mean(T[:n])), "c": float(np.mean(c[:n])), "n": int(n)}
    if n < 5:
        return {**base, "logk": np.log(1e-4), "z": DEFAULT_Z, "soh0": float(np.median(y))}
    best = None
    for z in np.linspace(0.3, 2.5, 221):
        A = np.vstack([np.ones(n), -np.power(x, z)]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        sse = float(np.sum((A @ coef - y) ** 2))
        if coef[0] > 0 and coef[1] > 0 and (best is None or sse < best[0]):
            best = (sse, z, coef)
    if best is None:
        return {**base, "logk": np.log(1e-4), "z": DEFAULT_Z, "soh0": float(np.median(y[:5]))}
    _, z, (s0, s0k) = best
    return {**base, "logk": float(np.log(s0k / s0)), "z": float(z), "soh0": float(s0)}


def aggregate(local_fits, ea=DEFAULT_EA, ea_c=DEFAULT_EA_C, min_dT=10.0):
    """Server: pool local fits of one chemistry into (logB, Ea, z)."""
    w = np.array([max(f["n"], 1) for f in local_fits], float)
    z = float(np.sum(w * [f["z"] for f in local_fits]) / w.sum())
    Ts = np.array([f["T"] for f in local_fits])
    cs = np.array([f["c"] for f in local_fits])
    logk = np.array([f["logk"] for f in local_fits])
    if Ts.max() - Ts.min() >= min_dT and len(local_fits) >= 3:
        x = -1.0 / (R_GAS * Ts)                    # log k = log B + Ea * x + ea_c * c/(R T)
        yv = logk - ea_c * cs / (R_GAS * Ts)
        A = np.vstack([np.ones_like(x), x]).T * np.sqrt(w)[:, None]
        coef, *_ = np.linalg.lstsq(A, yv * np.sqrt(w), rcond=None)
        ea_hat = float(np.clip(coef[1], 5e3, 1.2e5))
        logB = float(coef[0])
        return {"logB": logB, "Ea": ea_hat, "Ea_c": ea_c, "z": z, "Ea_estimated": True}
    logB = float(np.sum(w * (logk + (ea - ea_c * cs) / (R_GAS * Ts))) / w.sum())
    return {"logB": logB, "Ea": ea, "Ea_c": ea_c, "z": z, "Ea_estimated": False}


def k_rate(params, T_K, c):
    return np.exp(params["logB"] - (params["Ea"] - params["Ea_c"] * c) / (R_GAS * T_K))


def prior_next(soh_n, ah_n, dah, T_K, c, params, soh0):
    """Physics prior for SoH_{n+1} given SoH_n (increment form, anchored to the observation)."""
    z = params["z"]
    k = k_rate(params, T_K, c)
    inc = k * (np.power(np.maximum(ah_n + dah, 1e-9), z) - np.power(np.maximum(ah_n, 1e-9), z))
    return soh_n - soh0 * inc


def prior_series(rec, params, L):
    """Prior for each window target y_n = SoH_{n+1}, aligned with data.make_windows."""
    T, c, ah, dah = cell_series(rec)
    soh0 = float(np.median(rec.soh[:5]))
    out = []
    for n in range(L - 1, len(rec.soh) - 1):
        out.append(prior_next(rec.soh[n], ah[n], dah[n], T[n], c[n], params, soh0))
    return np.asarray(out, np.float32)


def fit_logB_fixed_z(rec, z, ea=DEFAULT_EA, ea_c=DEFAULT_EA_C, upto=None):
    """Second federated pass: refit a vehicle's fade rate with the pooled exponent z and convert it to the
    temperature-free log B (one scalar per vehicle)."""
    T, c, ah, _ = cell_series(rec)
    n = len(rec.soh) if upto is None else min(max(int(upto), 3), len(rec.soh))
    A = np.vstack([np.ones(n), -np.power(np.maximum(ah[:n], 1e-9), z)]).T
    (s0, s0k), *_ = np.linalg.lstsq(A, rec.soh[:n], rcond=None)
    if s0 <= 0 or s0k <= 0:
        return None
    logk = np.log(s0k / s0)
    return float(logk + (ea - ea_c * np.mean(c[:n])) / (R_GAS * np.mean(T[:n])))


def fade_path_samples(soh_hist, ah_hist, params, rated_ah, T_hist, T_future, c_hist, c_future, M, level_mu, level_sd,
                      h_max, eol, block=1, rng=None):
    """Physics-guided RUL (paper Sect. 4.6).
    Fade rate: fleet prior on log B (mean logB_mu, sd logB_sd from the federation) combined by precision with
    the vehicle's own least-squares estimate on its history (delta-method variance), i.e. an empirical-Bayes
    posterior; early in life the fleet dominates, later the vehicle's own data. The path starts at the
    Bayesian LSTM posterior level, is scaled to the planned temperature and C-rate by the Arrhenius factor,
    and is propagated to the end-of-life threshold; measurement noise is not compounded."""
    rng = rng or np.random.default_rng(0)
    z = params["z"]
    soh0 = float(np.median(soh_hist[:5]))
    to_logk = lambda logB, T, c: logB - (params["Ea"] - params["Ea_c"] * c) / (R_GAS * T)
    prior_mu = to_logk(params.get("logB_mu", params["logB"]), T_hist, c_hist)
    prior_sd = max(params.get("logB_sd", 1.0), 0.05)
    A = np.vstack([np.ones_like(ah_hist), -np.power(np.maximum(ah_hist, 1e-9), z)]).T
    coef, *_ = np.linalg.lstsq(A, soh_hist, rcond=None)
    resid = soh_hist - A @ coef
    s2 = float(resid @ resid) / max(len(soh_hist) - 2, 1)
    # AR(1) correction: autocorrelated residuals (curve misfit, regeneration) carry less information
    rho = float(np.corrcoef(resid[:-1], resid[1:])[0, 1]) if len(resid) > 3 and np.std(resid) > 0 else 0.0
    rho = min(max(rho, 0.0), 0.99)
    inflate = (1 + rho) / (1 - rho)
    cov = s2 * inflate * np.linalg.pinv(A.T @ A)
    if coef[0] > 0 and coef[1] > 0:
        loc = float(np.log(coef[1] / coef[0]))
        var_loc = cov[1, 1] / coef[1] ** 2 + cov[0, 0] / coef[0] ** 2 - 2 * cov[0, 1] / (coef[0] * coef[1])
        var_loc = max(float(var_loc), 1e-6)
        prec = 1 / prior_sd ** 2 + 1 / var_loc
        post_mu = (prior_mu / prior_sd ** 2 + loc / var_loc) / prec
        post_sd = np.sqrt(1 / prec)
    else:
        post_mu, post_sd = prior_mu, prior_sd
    logk = rng.normal(post_mu, post_sd, size=M)
    k = np.exp(logk) * np.exp((params["Ea"] / R_GAS) * (1.0 / T_hist - 1.0 / T_future)
                              + params["Ea_c"] * (c_future / T_future - c_hist / T_hist) / R_GAS)
    ah0 = float(ah_hist[-1])
    level = rng.normal(level_mu, max(level_sd, 1e-6), size=M)
    rul = np.full(M, h_max * block, float)
    ah = np.full(M, ah0)
    s = level.copy()
    alive = np.ones(M, bool)
    for h in range(1, h_max + 1):
        ah = ah + np.maximum(s, 0.05) * rated_ah * block
        s = level - soh0 * k * (ah ** z - ah0 ** z)
        hit = alive & (s <= eol)
        rul[hit] = h * block
        alive &= ~hit
        if not alive.any():
            break
    return rul
