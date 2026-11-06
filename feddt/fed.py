"""Federated training: clients, drift-aware server, compression, conformal (paper Sec. 4.3-4.6)."""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np
import torch

from . import physics
from .conformal import tilt_eta
from .data import CellRecord, make_windows
from .model import BayesLSTM, gaussian_nll, is_backbone, is_head


from .config import FedConfig, uplink_bytes  # noqa: E402,F401


# ------------------------------------------------------------------ helpers
def flat(model, names):
    return torch.cat([dict(model.named_parameters())[n].detach().reshape(-1) for n in names])


def unflat_add(model, names, vec, scale=1.0):
    params = dict(model.named_parameters())
    i = 0
    with torch.no_grad():
        for n in names:
            p = params[n]
            k = p.numel()
            p.add_(scale * vec[i:i + k].view_as(p))
            i += k


def compress(delta: torch.Tensor, ef: torch.Tensor, rho: float, bits: int, gen: torch.Generator):
    """Top-k + stochastic b-bit quantisation with error feedback. Returns (sent, new_ef)."""
    v = delta + ef
    if rho < 1.0:
        k = max(1, int(math.ceil(rho * v.numel())))
        idx = torch.topk(v.abs(), k, sorted=False).indices
        sparse = torch.zeros_like(v)
        sparse[idx] = v[idx]
    else:
        sparse = v
    if bits < 32:
        levels = 2 ** (bits - 1) - 1
        s = sparse.abs().max().clamp_min(1e-12)
        x = sparse / s * levels
        lo = torch.floor(x)
        q = lo + (torch.rand(x.shape, generator=gen) < (x - lo)).float()
        sent = q / levels * s
        sent[sparse == 0] = 0.0
    else:
        sent = sparse
    return sent, v - sent


class RFF:
    """Random Fourier features for a Gaussian kernel; mean embedding difference ~ MMD^2."""

    def __init__(self, d_in, D, bandwidth, seed):
        g = np.random.default_rng(seed)
        self.W = torch.tensor(g.normal(0, 1.0 / bandwidth, (d_in, D // 2)), dtype=torch.float32)

    def features(self, H):
        Z = H @ self.W
        return torch.cat([torch.cos(Z), torch.sin(Z)], 1) / math.sqrt(self.W.shape[1])

    def mean_embed(self, H):
        Z = H @ self.W
        phi = torch.cat([torch.cos(Z), torch.sin(Z)], 1) / math.sqrt(self.W.shape[1])
        return phi.mean(0)


# ------------------------------------------------------------------ normalisation (federated sufficient stats)
def federated_norm(clients):
    stats = {}
    for c in clients:
        X = c.Xtr.reshape(-1, c.Xtr.shape[-1]).astype(np.float64)
        s = stats.setdefault(c.chem, [0, 0.0, 0.0])
        s[0] += X.shape[0]
        s[1] = s[1] + X.sum(0)
        s[2] = s[2] + (X ** 2).sum(0)
    out = {}
    for chem, (n, s1, s2) in stats.items():
        mu = s1 / n
        sd = np.sqrt(np.maximum(s2 / n - mu ** 2, 1e-12))
        mu[0], sd[0] = 0.0, 1.0  # keep SoH channel in physical units
        out[chem] = (mu.astype(np.float32), sd.astype(np.float32))
    return out


# ------------------------------------------------------------------ client
@dataclass
class Client:
    rec: CellRecord
    chem: str
    Xtr: np.ndarray = None
    ytr: np.ndarray = None
    Xcal: np.ndarray = None
    ycal: np.ndarray = None
    Xval: np.ndarray = None
    yval: np.ndarray = None
    mask: np.ndarray = None
    ef: torch.Tensor = None
    c_local: torch.Tensor = None
    personal: dict = None
    drift_hist: list = field(default_factory=list)
    join_round: int = 0            # scenario S3: late-joining chemistry
    ptr: np.ndarray = None         # physics prior for each target (train / cal / val)
    pcal: np.ndarray = None
    pval: np.ndarray = None
    n_train_cycles: int = 0

    @staticmethod
    def from_record(rec: CellRecord, L: int, chem_key: str):
        X, mask, y = make_windows(rec, L)
        n = len(y)
        a, b = int(0.8 * n), int(0.9 * n)
        c = Client(rec, chem_key, X[:a], y[:a], X[a:b], y[a:b], X[b:], y[b:], mask)
        c.ptr, c.pcal, c.pval = np.zeros_like(c.ytr), np.zeros_like(c.ycal), np.zeros_like(c.yval)
        c.n_train_cycles = a + L
        return c

    def set_prior(self, prior):
        a, b = len(self.ytr), len(self.ytr) + len(self.ycal)
        self.ptr, self.pcal, self.pval = prior[:a], prior[a:b], prior[b:]

    def tensors(self, X, norm):
        mu, sd = norm[self.chem]
        return torch.tensor((X - mu) / sd, dtype=torch.float32)


def train_local(model, X, y, cfg, epochs, gen, global_bb=None, scaffold=None, head_only=False):
    names = [n for n, _ in model.named_parameters()]
    params = [p for n, p in model.named_parameters() if (is_head(n) or not head_only)]
    opt = torch.optim.Adam(params, lr=cfg.lr)
    model.train()
    n = len(y)
    steps = 0
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, cfg.batch):
            b = perm[i:i + cfg.batch]
            mu, logv, _ = model(X[b])
            loss = gaussian_nll(y[b], mu, logv)
            loss = loss + cfg.weight_decay * sum((p ** 2).sum() for p in params)
            if global_bb is not None and cfg.mu_prox > 0 and not head_only:
                cur = torch.cat([p.reshape(-1) for nm, p in model.named_parameters() if is_backbone(nm)])
                loss = loss + 0.5 * cfg.mu_prox * ((cur - global_bb) ** 2).sum()
            opt.zero_grad()
            loss.backward()
            if scaffold is not None and not head_only:
                c_glob, c_loc = scaffold
                off = 0
                for nm, p in model.named_parameters():
                    k = p.numel()
                    if p.grad is not None:
                        p.grad.add_((c_glob[off:off + k] - c_loc[off:off + k]).view_as(p))
                    off += k
            opt.step()
            steps += 1
    _ = names
    return steps


# ------------------------------------------------------------------ server / federation
class Federation:
    def __init__(self, clients: list[Client], n_in: int, cfg: FedConfig):
        self.cfg, self.clients = cfg, clients
        torch.manual_seed(cfg.seed)
        self.gen = torch.Generator().manual_seed(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)
        self.norm = federated_norm(clients)
        # federated physics prior: local fits on training cycles -> per-chemistry pooling
        self.phys = {}
        if cfg.use_physics:
            fits = {}
            for c in clients:
                fits.setdefault(c.chem, []).append(physics.fit_local(c.rec, upto=c.n_train_cycles))
            self.phys = {ch: physics.aggregate(f) for ch, f in fits.items()}
            for c in clients:
                c.set_prior(physics.prior_series(c.rec, self.phys[c.chem], cfg.window))
        self.model = BayesLSTM(n_in, cfg.hidden, 2, cfg.dropout)
        self.names = [n for n, _ in self.model.named_parameters()]
        self.bb_names = [n for n in self.names if is_backbone(n)]
        self.hd_names = [n for n in self.names if is_head(n)]
        head0 = {n: p.detach().clone() for n, p in self.model.named_parameters() if is_head(n)}
        chems = sorted({c.chem for c in clients}) if cfg.chem_heads else ["shared"]
        self.heads = {ch: copy.deepcopy(head0) for ch in chems}
        self.P = sum(p.numel() for p in self.model.parameters())
        self.n_tensor = len(self.names)
        for c in clients:
            c.ef = torch.zeros(self.P)
            c.c_local = torch.zeros(self.P)
        self.c_global = torch.zeros(self.P)
        # drift machinery: bandwidth by median heuristic on a first local estimate (scalars only)
        self.rff = RFF(cfg.hidden, cfg.rff_dim, bandwidth=math.sqrt(cfg.hidden) * 0.5, seed=cfg.seed + 7)
        self.ref_mean = {}
        self.log = {"round": [], "uplink_bytes": [], "downlink_bytes": [], "alpha": [], "drift": []}

    # -- model for a chemistry
    def head_key(self, chem):
        return chem if self.cfg.chem_heads else "shared"

    def model_for(self, chem, personal=None):
        m = copy.deepcopy(self.model)
        with torch.no_grad():
            params = dict(m.named_parameters())
            src = personal if personal is not None else self.heads.get(self.head_key(chem))
            if src is None:  # unseen chemistry: caller should pass cold_start_key(); fallback = first head
                src = next(iter(self.heads.values()))
            for n, v in src.items():
                params[n].copy_(v)
        return m

    def set_mask(self, m, client):
        m.in_mask.copy_(torch.tensor(client.mask, dtype=torch.float32))

    def run(self, eval_fn=None):
        cfg = self.cfg
        for t in range(cfg.rounds):
            avail = [k for k, c in enumerate(self.clients) if c.join_round <= t]
            S = self.rng.choice(avail, size=max(1, int(math.ceil(cfg.participation * len(avail)))), replace=False)
            ups, info = [], []
            for k in S:
                ups.append(self.client_update(self.clients[k]))
                info.append(ups[-1][1])
            self.aggregate(S, ups)
            P_dl = self.P * 4
            self.log["round"].append(t)
            self.log["uplink_bytes"].append(sum(u[2] for u in ups))
            self.log["downlink_bytes"].append(P_dl * len(S))
            if eval_fn is not None and (t % cfg.eval_every == 0 or t == cfg.rounds - 1):
                eval_fn(self, t)
        return self

    def client_update(self, c: Client):
        cfg = self.cfg
        m = self.model_for(c.chem)
        self.set_mask(m, c)
        theta0 = flat(m, self.names)
        X, y = c.tensors(c.Xtr, self.norm), torch.tensor(c.ytr - c.ptr)
        bb0 = flat(m, self.bb_names) if cfg.mu_prox > 0 else None
        sc = (self.c_global, c.c_local) if cfg.method == "scaffold" else None
        steps = train_local(m, X, y, cfg, cfg.local_epochs, self.gen, global_bb=bb0, scaffold=sc)
        theta1 = flat(m, self.names)
        delta = theta1 - theta0
        if cfg.method == "scaffold":
            c_new = c.c_local - self.c_global + (theta0 - theta1) / max(steps * cfg.lr, 1e-12)
            c.dc = c_new - c.c_local
            c.c_local = c_new
        # reliability: NLL on the most recent held-out cycles
        r = 1.0
        if cfg.use_reliability and len(c.yval):
            m.eval()
            mu, ve, va = m.predict(c.tensors(c.Xval, self.norm), T=min(cfg.mc_T, 20))
            var = (ve + va).clamp_min(1e-10)
            nll = float((0.5 * ((torch.tensor(c.yval - c.pval) - mu) ** 2 / var + var.log() + math.log(2 * math.pi))).mean())
            r = math.exp(-nll / cfg.tau_r) if nll > -50 else 1.0
        # drift: RFF mean embedding of recent windows vs cluster reference
        d, phi = 0.0, None
        if cfg.use_drift:
            m.eval()
            Xr = np.concatenate([c.Xcal, c.Xval]) if len(c.ycal) + len(c.yval) else c.Xtr[-20:]
            with torch.no_grad():
                H = m.embed(c.tensors(Xr, self.norm))
            phi = self.rff.mean_embed(H)
            ref = self.ref_mean.get(c.chem)
            d = float(((phi - ref) ** 2).sum()) if ref is not None else 0.0
            c.drift_hist.append(d)
        sent, c.ef = compress(delta, c.ef, cfg.topk, cfg.qbits, self.gen)
        nbytes = uplink_bytes(self.P, cfg.topk, cfg.qbits, self.n_tensor, cfg.rff_dim, cfg.use_drift)
        return sent, {"n": len(c.ytr), "r": r, "d": d, "phi": phi, "chem": c.chem, "client": c, "model": m}, nbytes

    def aggregate(self, S, ups):
        cfg = self.cfg
        info = [u[1] for u in ups]
        n = np.array([i["n"] for i in info], float)
        r = np.array([i["r"] for i in info], float)
        d = np.array([i["d"] for i in info], float)
        w = n ** cfg.beta
        if cfg.use_reliability:
            w = w * np.maximum(r, 1e-12)
        if cfg.use_drift and np.any(d > 0):
            med = np.median(d[d > 0]) if np.any(d > 0) else 1.0
            w = w * np.exp(-cfg.gamma * d / max(med, 1e-12))
        a = w / w.sum()
        if cfg.alpha_floor > 0:
            a = np.maximum(a, cfg.alpha_floor / len(a))
            a = a / a.sum()
        self.log["alpha"].append(a.tolist())
        self.log["drift"].append(d.tolist())
        # backbone over all clients
        bb_idx = self._index(self.bb_names)
        upd = sum(float(a[i]) * ups[i][0] for i in range(len(ups)))
        unflat_add(self.model, self.bb_names, upd[bb_idx])
        # heads within chemistry cluster
        hd_idx = self._index(self.hd_names)
        for key in self.heads:
            members = [i for i in range(len(ups)) if self.head_key(info[i]["chem"]) == key]
            if not members:
                continue
            am = np.array([a[i] for i in members])
            am = am / am.sum()
            hu = sum(float(am[j]) * ups[i][0][hd_idx] for j, i in enumerate(members))
            off = 0
            for nm in self.hd_names:
                p = self.heads[key][nm]
                k = p.numel()
                self.heads[key][nm] = p + hu[off:off + k].view_as(p)
                off += k
        # scaffold global control variate
        if cfg.method == "scaffold":
            self.c_global = self.c_global + sum(i["client"].dc for i in info) / len(self.clients)
        # drift references (secure-aggregated means) and personalisation budget
        if cfg.use_drift:
            for key in {i["chem"] for i in info}:
                mem = [i for i in info if i["chem"] == key]
                wn = np.array([i["n"] for i in mem], float)
                self.ref_mean[key] = sum(float(wn[j] / wn.sum()) * mem[j]["phi"] for j in range(len(mem)))
            med = np.median(d) if len(d) else 0.0
            for i in info:
                if med > 0 and i["d"] > 1.5 * med and cfg.personal_epochs > 0:
                    c, m = i["client"], i["model"]
                    train_local(m, c.tensors(c.Xtr, self.norm), torch.tensor(c.ytr - c.ptr), cfg, cfg.personal_epochs,
                                self.gen, head_only=True)
                    c.personal = {nm: p.detach().clone() for nm, p in m.named_parameters() if is_head(nm)}

    def cold_start_key(self, X_norm: torch.Tensor):
        """Nearest chemistry cluster by RFF mean-embedding distance (paper Sec. 4.4)."""
        if not self.ref_mean:
            return next(iter(self.heads))
        self.model.eval()
        with torch.no_grad():
            phi = self.rff.mean_embed(self.model.embed(X_norm))
        return min(self.ref_mean, key=lambda k: float(((phi - self.ref_mean[k]) ** 2).sum()))

    def _index(self, subset):
        idx, off = [], 0
        sub = set(subset)
        for nm, p in self.model.named_parameters():
            k = p.numel()
            if nm in sub:
                idx.append(torch.arange(off, off + k))
            off += k
        return torch.cat(idx)

    # ------------------------------------------------------------- conformal
    def conformal_quantile(self, a=None, target_phi=None, bins=4000, smax=40.0, ess_min=0.2):
        """Federated split conformal factor q_hat (Eq. 10). With DW-FCC (Algorithm 2) and a target
        vehicle embedding `target_phi`, calibration scores are reweighted by an exponential tilt
        of their RFF embeddings towards the target, so coverage tracks the target's conditions."""
        a = self.cfg.conformal_a if a is None else a
        scores, feats = [], []
        for c in self.clients:                      # each vehicle computes on its own calibration set
            if not len(c.ycal):
                continue
            m = self.model_for(c.chem)
            self.set_mask(m, c)
            m.eval()
            Xc = c.tensors(c.Xcal, self.norm)
            mu, ve, va = m.predict(Xc, T=self.cfg.mc_T)
            scores.append(np.abs((c.ycal - c.pcal) - mu.numpy()) / np.sqrt((ve + va).numpy() + 1e-12))
            if target_phi is not None and self.cfg.dwfcc:
                with torch.no_grad():
                    feats.append(self.rff.features(m.embed(Xc)).numpy())
        if not scores:
            return 1.96
        s_all = np.concatenate(scores)
        m_tot = len(s_all)
        if target_phi is None or not self.cfg.dwfcc:
            hist, _ = np.histogram(np.clip(s_all, 0, smax), bins=bins, range=(0, smax))
            level = min(math.ceil((m_tot + 1) * (1 - a)) / m_tot, 1.0)
            j = int(np.searchsorted(np.cumsum(hist) / m_tot, level))
            return (j + 1) * smax / bins
        # Algorithm 2: secure sums of mean embedding, projections and weighted histograms
        F = np.concatenate(feats)
        mu_p = F.mean(0)
        delta = np.asarray(target_phi, float) - mu_p
        proj = F @ delta
        target = float(np.asarray(target_phi, float) @ delta)
        eta = tilt_eta(proj, target, ess_min=ess_min)
        w = np.exp(eta * (proj - proj.max()))
        w_test = float(np.exp(eta * (target - proj.max())))
        hist, edges = np.histogram(np.clip(s_all, 0, smax), bins=bins, range=(0, smax), weights=w)
        cdf = np.cumsum(hist) / (hist.sum() + w_test)
        j = int(np.searchsorted(cdf, 1 - a))
        self.log.setdefault("dwfcc", []).append({"eta": eta, "ess_frac": float(w.sum() ** 2 / (w ** 2).sum() / m_tot)})
        return float(edges[j + 1]) if j < bins else smax

    def target_embedding(self, chem, X_norm, mask=None):
        """Mean RFF embedding of a target vehicle's observed windows (sent under secure aggregation)."""
        m = self.model_for(chem)
        if mask is not None:
            m.in_mask.copy_(torch.tensor(mask, dtype=torch.float32))
        m.eval()
        with torch.no_grad():
            return self.rff.features(m.embed(X_norm)).mean(0).numpy()
