"""NumPy federation that mirrors feddt/fed.py: FedDT-BayesLSTM and the FedAvg / FedProx / SCAFFOLD
baselines, physics prior, drift-aware weights, chemistry heads, top-k + 8-bit compression with
error feedback, split conformal and DW-FCC (Algorithm 2), physics-guided RUL rollout."""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np

from feddt import physics
from feddt.conformal import tilt_eta
from feddt.data import make_windows
from feddt.features import DEGRADATION_HI, USAGE_HI
from feddt.config import FedConfig, uplink_bytes
from .model_np import (BACKBONE, HEAD, ORDER, Adam, embed, flatten, init_params, loss_and_grad, n_params, predict,
                       sample_masks, unflatten)


def stochastic_quant(v, bits, rng):
    if bits >= 32:
        return v
    levels = 2 ** (bits - 1) - 1
    s = max(np.abs(v).max(), 1e-12)
    x = v / s * levels
    lo = np.floor(x)
    q = lo + (rng.random(x.shape) < (x - lo))
    out = q / levels * s
    out[v == 0] = 0.0
    return out


def compress(delta, ef, rho, bits, rng):
    v = delta + ef
    if rho < 1.0:
        k = max(1, int(math.ceil(rho * v.size)))
        idx = np.argpartition(np.abs(v), -k)[-k:]
        sp = np.zeros_like(v); sp[idx] = v[idx]
    else:
        sp = v
    sent = stochastic_quant(sp, bits, rng)
    return sent, v - sent


class RFF:
    def __init__(self, d_in, D, bandwidth, seed):
        g = np.random.default_rng(seed)
        self.W = g.normal(0, 1.0 / bandwidth, (d_in, D // 2))

    def features(self, H):
        Z = H @ self.W
        return np.concatenate([np.cos(Z), np.sin(Z)], 1) / math.sqrt(self.W.shape[1])


@dataclass
class ClientNP:
    rec: object
    chem: str
    Xtr: np.ndarray
    ytr: np.ndarray
    Xcal: np.ndarray
    ycal: np.ndarray
    Xval: np.ndarray
    yval: np.ndarray
    mask: np.ndarray
    ptr: np.ndarray = None
    pcal: np.ndarray = None
    pval: np.ndarray = None
    n_train_cycles: int = 0
    join_round: int = 0
    ef: np.ndarray = None
    c_local: np.ndarray = None
    dc: np.ndarray = None
    personal: dict = None
    drift_hist: list = field(default_factory=list)

    @staticmethod
    def from_record(rec, L, chem):
        X, mask, y = make_windows(rec, L)
        n = len(y)
        a, b = int(0.8 * n), int(0.9 * n)
        c = ClientNP(rec, chem, X[:a], y[:a], X[a:b], y[a:b], X[b:], y[b:], mask)
        c.ptr, c.pcal, c.pval = np.zeros_like(c.ytr), np.zeros_like(c.ycal), np.zeros_like(c.yval)
        c.n_train_cycles = a + L
        return c

    def set_prior(self, prior):
        a, b = len(self.ytr), len(self.ytr) + len(self.ycal)
        self.ptr, self.pcal, self.pval = prior[:a], prior[a:b], prior[b:]


def federated_norm(clients):
    stats = {}
    for c in clients:
        X = c.Xtr.reshape(-1, c.Xtr.shape[-1]).astype(np.float64)
        s = stats.setdefault(c.chem, [0, 0.0, 0.0])
        s[0] += X.shape[0]; s[1] = s[1] + X.sum(0); s[2] = s[2] + (X ** 2).sum(0)
    out = {}
    for chem, (n, s1, s2) in stats.items():
        mu = s1 / n
        sd = np.sqrt(np.maximum(s2 / n - mu ** 2, 1e-12))
        mu[0], sd[0] = 0.0, 1.0
        sd = np.where(sd < 1e-6, 1.0, sd)
        out[chem] = (mu, sd)
    return out


def train_epochs(p, X, y, cfg, epochs, rng, prox=None, scaffold=None, head_only=False, mask=None):
    keys = list(HEAD) if head_only else list(ORDER)
    opt = Adam(p, lr=cfg.lr, keys=keys)
    n = len(y)
    steps = 0
    H = p["l0_Wh"].shape[0]
    for _ in range(epochs):
        perm = rng.permutation(n)
        for i in range(0, n, cfg.batch):
            b = perm[i:i + cfg.batch]
            if len(b) < 2:
                continue
            masks = sample_masks(len(b), X.shape[2], H, cfg.dropout, rng)
            xb = X[b] * mask if mask is not None else X[b]
            _, g = loss_and_grad(p, xb, y[b], masks, wd=cfg.weight_decay,
                                 prox=prox if (prox is not None and not head_only) else None, prox_mu=cfg.mu_prox)
            extra = None
            if scaffold is not None and not head_only:
                c_glob, c_loc = scaffold
                extra = unflatten(c_glob - c_loc, p)
            opt.step(p, g, extra)
            steps += 1
    return steps


class FederationNP:
    def __init__(self, clients, n_in, cfg: FedConfig):
        self.cfg, self.clients = cfg, clients
        self.rng = np.random.default_rng(cfg.seed)
        self.norm = federated_norm(clients)
        self.phys = {}
        if cfg.use_physics:
            fits = {}
            for c in clients:
                fits.setdefault(c.chem, []).append(physics.fit_local(c.rec, upto=c.n_train_cycles))
            self.phys = {ch: physics.aggregate(f) for ch, f in fits.items()}
            # second pass: fleet prior on log B at the pooled exponent (scalars only)
            by = {}
            for c in clients:
                lb = physics.fit_logB_fixed_z(c.rec, self.phys[c.chem]["z"], upto=c.n_train_cycles)
                if lb is not None:
                    by.setdefault(c.chem, []).append(lb)
            for ch, v in by.items():
                self.phys[ch]["logB_mu"] = float(np.mean(v))
                self.phys[ch]["logB_sd"] = float(np.std(v)) if len(v) > 1 else 1.0
            for c in clients:
                c.set_prior(physics.prior_series(c.rec, self.phys[c.chem], cfg.window))
        else:  # data-driven: residual to persistence (last observed SoH), a strong reference
            for c in clients:
                c.set_prior(np.concatenate([c.Xtr, c.Xcal, c.Xval])[:, -1, 0])
        self.p = init_params(n_in, cfg.hidden, seed=cfg.seed)
        head0 = {k: self.p[k].copy() for k in HEAD}
        chems = sorted({c.chem for c in clients}) if cfg.chem_heads else ["shared"]
        self.heads = {ch: copy.deepcopy(head0) for ch in chems}
        self.P = n_params(self.p)
        for c in clients:
            c.ef = np.zeros(self.P); c.c_local = np.zeros(self.P)
        self.c_global = np.zeros(self.P)
        self.rff = RFF(cfg.hidden, cfg.rff_dim, bandwidth=math.sqrt(cfg.hidden) * 0.5, seed=cfg.seed + 7)
        self.ref_mean = {}
        self.log = {"uplink_bytes": [], "downlink_bytes": [], "alpha": [], "drift": []}

    def head_key(self, chem):
        return chem if self.cfg.chem_heads else "shared"

    def params_for(self, chem, personal=None):
        p = {k: self.p[k].copy() for k in BACKBONE}
        src = personal if personal is not None else self.heads.get(self.head_key(chem), next(iter(self.heads.values())))
        for k in HEAD:
            p[k] = src[k].copy()
        return p

    def xn(self, chem, X):
        mu, sd = self.norm.get(chem, next(iter(self.norm.values())))
        return (X - mu) / sd

    def run(self, eval_fn=None):
        cfg = self.cfg
        for t in range(cfg.rounds):
            avail = [k for k, c in enumerate(self.clients) if c.join_round <= t]
            S = self.rng.choice(avail, size=max(1, int(math.ceil(cfg.participation * len(avail)))), replace=False)
            ups = [self.client_update(self.clients[k]) for k in S]
            self.aggregate(ups)
            self.log["uplink_bytes"].append(sum(u[2] for u in ups))
            self.log["downlink_bytes"].append(self.P * 4 * len(S))
            if eval_fn is not None and (t % cfg.eval_every == 0 or t == cfg.rounds - 1):
                eval_fn(self, t)
        return self

    def client_update(self, c):
        cfg = self.cfg
        p = self.params_for(c.chem)
        theta0 = flatten(p)
        X = self.xn(c.chem, c.Xtr) * c.mask
        y = c.ytr - c.ptr
        prox = {k: v.copy() for k, v in p.items()} if cfg.mu_prox > 0 else None
        sc = (self.c_global, c.c_local) if cfg.method == "scaffold" else None
        steps = train_epochs(p, X, y, cfg, cfg.local_epochs, self.rng, prox=prox, scaffold=sc)
        theta1 = flatten(p)
        delta = theta1 - theta0
        if cfg.method == "scaffold":
            c_new = c.c_local - self.c_global + (theta0 - theta1) / max(steps * cfg.lr, 1e-12)
            c.dc = c_new - c.c_local
            c.c_local = c_new
        r = 1.0
        if cfg.use_reliability and len(c.yval):
            mu, ve, va = predict(p, self.xn(c.chem, c.Xval) * c.mask, T=min(cfg.mc_T, 20), p_drop=cfg.dropout, rng=self.rng)
            var = np.maximum(ve + va, 1e-10)
            nll = float(np.mean(0.5 * (((c.yval - c.pval) - mu) ** 2 / var + np.log(var) + math.log(2 * math.pi))))
            r = math.exp(-nll / cfg.tau_r) if nll > -50 else 1.0
        d, phi = 0.0, None
        if cfg.use_drift:
            Xr = np.concatenate([c.Xcal, c.Xval]) if len(c.ycal) + len(c.yval) else c.Xtr[-20:]
            phi = self.rff.features(embed(p, self.xn(c.chem, Xr) * c.mask)).mean(0)
            ref = self.ref_mean.get(c.chem)
            d = float(np.sum((phi - ref) ** 2)) if ref is not None else 0.0
            c.drift_hist.append(d)
        sent, c.ef = compress(delta, c.ef, cfg.topk, cfg.qbits, self.rng)
        nbytes = uplink_bytes(self.P, cfg.topk, cfg.qbits, len(ORDER), cfg.rff_dim, cfg.use_drift)
        return sent, {"n": len(c.ytr), "r": r, "d": d, "phi": phi, "chem": c.chem, "client": c, "p": p}, nbytes

    def aggregate(self, ups):
        cfg = self.cfg
        info = [u[1] for u in ups]
        n = np.array([i["n"] for i in info], float)
        r = np.array([i["r"] for i in info], float)
        d = np.array([i["d"] for i in info], float)
        w = n ** cfg.beta
        if cfg.use_reliability:
            w = w * np.maximum(r, 1e-12)
        if cfg.use_drift and np.any(d > 0):
            med = np.median(d[d > 0])
            w = w * np.exp(-cfg.gamma * d / max(med, 1e-12))
        a = w / w.sum()
        if cfg.alpha_floor > 0:
            a = np.maximum(a, cfg.alpha_floor / len(a)); a = a / a.sum()
        self.log["alpha"].append(a.tolist()); self.log["drift"].append(d.tolist())
        nb = sum(self.p[k].size for k in BACKBONE)
        upd = sum(float(a[i]) * ups[i][0] for i in range(len(ups)))
        bb = unflatten(upd[:nb], self.p, BACKBONE)
        for k in BACKBONE:
            self.p[k] = self.p[k] + bb[k]
        for key in self.heads:
            mem = [i for i in range(len(ups)) if self.head_key(info[i]["chem"]) == key]
            if not mem:
                continue
            am = np.array([a[i] for i in mem]); am = am / am.sum()
            hu = sum(float(am[j]) * ups[i][0][nb:] for j, i in enumerate(mem))
            hd = unflatten(hu, self.p, HEAD)
            for k in HEAD:
                self.heads[key][k] = self.heads[key][k] + hd[k]
        if cfg.method == "scaffold":
            self.c_global = self.c_global + sum(i["client"].dc for i in info) / len(self.clients)
        if cfg.use_drift:
            for key in {i["chem"] for i in info}:
                mem = [i for i in info if i["chem"] == key]
                wn = np.array([i["n"] for i in mem], float)
                self.ref_mean[key] = sum(float(wn[j] / wn.sum()) * mem[j]["phi"] for j in range(len(mem)))
            med = np.median(d) if len(d) else 0.0
            for i in info:
                if med > 0 and i["d"] > 1.5 * med and cfg.personal_epochs > 0:
                    c, p = i["client"], i["p"]
                    train_epochs(p, self.xn(c.chem, c.Xtr) * c.mask, c.ytr - c.ptr, cfg, cfg.personal_epochs, self.rng,
                                 head_only=True)
                    c.personal = {k: p[k].copy() for k in HEAD}

    # ------------------------------------------------------------- conformal (split and DW-FCC)
    def conformal_quantile(self, a=None, target_phi=None, bins=4000, smax=40.0, ess_min=0.2):
        a = self.cfg.conformal_a if a is None else a
        scores, feats = [], []
        for c in self.clients:
            if not len(c.ycal):
                continue
            p = self.params_for(c.chem)
            Xc = self.xn(c.chem, c.Xcal) * c.mask
            mu, ve, va = predict(p, Xc, T=self.cfg.mc_T, p_drop=self.cfg.dropout, rng=self.rng)
            scores.append(np.abs((c.ycal - c.pcal) - mu) / np.sqrt(ve + va + 1e-12))
            if target_phi is not None and self.cfg.dwfcc:
                feats.append(self.rff.features(embed(p, Xc)))
        s_all = np.concatenate(scores)
        m_tot = len(s_all)
        if target_phi is None or not self.cfg.dwfcc:
            level = min(math.ceil((m_tot + 1) * (1 - a)) / m_tot, 1.0)
            return float(np.quantile(s_all, level, method="higher")), {"eta": 0.0, "ess_frac": 1.0}
        F = np.concatenate(feats)
        delta = target_phi - F.mean(0)
        proj = F @ delta
        target = float(target_phi @ delta)
        eta = tilt_eta(proj, target, ess_min=ess_min)
        w = np.exp(eta * (proj - proj.max()))
        w_test = float(np.exp(eta * (target - proj.max())))
        order = np.argsort(s_all)
        cdf = np.cumsum(w[order]) / (w.sum() + w_test)
        j = int(np.searchsorted(cdf, 1 - a))
        q = float(s_all[order][j]) if j < m_tot else float(s_all.max())
        return q, {"eta": eta, "ess_frac": float(w.sum() ** 2 / (w ** 2).sum() / m_tot)}

    def target_embedding(self, chem, X_norm):
        return self.rff.features(embed(self.params_for(chem), X_norm)).mean(0)


def rollout_rul_np(p, rec, start, L, norm_fn, mask, eol, M=200, h_max=3000, block=1, seed=0, phys=None, p_drop=0.2):
    """RUL samples from cycle `start`.
    phys given (FedDT): physics-guided path, physics.fade_path_samples, anchored at the Bayesian LSTM
    posterior level (prior + residual mean, epistemic sd from MC passes).
    phys None (data-driven baselines): autoregressive LSTM rollout with one dropout mask per trajectory
    (epistemic) and the latent mean fed back, so measurement noise does not compound."""
    rng = np.random.default_rng(seed)
    soh_hist, his = rec.soh[:start + 1], rec.his[:start + 1].copy()
    if phys is not None:
        T_s, c_s, ah_s, _ = physics.cell_series(rec)
        X, _, _ = make_windows(rec, L)
        xw = norm_fn(X[start - L + 1:start - L + 2]) * mask if start - L + 1 >= 0 else None
        if xw is not None and len(xw):
            mu, ve, va = predict(p, xw, T=30, p_drop=p_drop, rng=rng)
            prior = physics.prior_series(rec, phys, L)[start - L + 1]
            level_mu, level_sd = float(prior + mu[0]), float(np.sqrt(ve[0]))
        else:
            level_mu, level_sd = float(soh_hist[-1]), 0.01
        Tw, cw = float(np.mean(T_s[max(0, start - 19):start + 1])), float(np.mean(c_s[max(0, start - 19):start + 1]))
        return physics.fade_path_samples(soh_hist, ah_s[:start + 1], phys, rec.rated_ah, Tw, Tw, cw, cw, M,
                                         level_mu, level_sd, h_max, eol, block, rng)
    col_med = np.nanmedian(np.where(np.isfinite(his), his, np.nan), axis=0) if np.isfinite(his).any() else np.zeros(his.shape[1])
    col_med = np.where(np.isfinite(col_med), col_med, 0.0)
    his = np.where(np.isfinite(his), his, col_med)
    coefs = {}
    for j in DEGRADATION_HI:
        A = np.vstack([soh_hist, np.ones_like(soh_hist)]).T
        coefs[j] = np.linalg.lstsq(A, his[:, j], rcond=None)[0]
    usage = his[-20:, USAGE_HI].mean(0)
    feat = np.concatenate([soh_hist[:, None], his], 1)[-L:]
    W = np.repeat(feat[None], M, 0)
    H = p["l0_Wh"].shape[0]
    masks = sample_masks(M, W.shape[-1], H, p_drop, rng)
    from .model_np import forward
    rul = np.full(M, float(h_max * block)); alive = np.ones(M, bool)
    for h in range(1, h_max + 1):
        mu, logv, _ = forward(p, norm_fn(W) * mask, masks)
        s = W[:, -1, 0] + mu          # persistence + learned residual, latent mean fed back
        hit = alive & (s <= eol)
        rul[hit] = h * block
        alive &= ~hit
        if not alive.any():
            break
        new = np.zeros((M, W.shape[-1]))
        new[:, 0] = s
        for j in DEGRADATION_HI:
            new[:, 1 + j] = coefs[j][0] * s + coefs[j][1]
        new[:, [1 + j for j in USAGE_HI]] = usage
        W = np.concatenate([W[:, 1:], new[:, None]], 1)
    return rul
