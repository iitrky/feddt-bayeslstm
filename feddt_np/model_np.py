"""NumPy implementation of the physics-guided Bayesian LSTM (same architecture as feddt/model.py).

Used to run the experiments where PyTorch is not available. Two-layer LSTM with variational
(time-shared) dropout masks on inputs, recurrent states and output; residual chemistry adapter
(16-unit bottleneck, GELU); heteroscedastic head (mean, log-variance). Gradients by explicit
back-propagation through time, checked against finite differences in tests/test_np_gradients.py.
"""
from __future__ import annotations

import numpy as np

BACKBONE = ("l0_Wx", "l0_Wh", "l0_b", "l1_Wx", "l1_Wh", "l1_b")
HEAD = ("a_W1", "a_b1", "a_W2", "a_b2", "h_W", "h_b")
ORDER = BACKBONE + HEAD
LOGV_MIN, LOGV_MAX = -14.0, 4.0


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def gelu(x):
    c = np.sqrt(2.0 / np.pi)
    u = c * (x + 0.044715 * x ** 3)
    t = np.tanh(u)
    return 0.5 * x * (1.0 + t), 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t ** 2) * c * (1.0 + 3 * 0.044715 * x ** 2)


def init_params(n_in, hidden=64, adapter=16, seed=0):
    g = np.random.default_rng(seed)
    H = hidden
    k = 1.0 / np.sqrt(H)
    p = {}
    for l, d in ((0, n_in), (1, H)):
        p[f"l{l}_Wx"] = g.uniform(-k, k, (d, 4 * H))
        p[f"l{l}_Wh"] = g.uniform(-k, k, (H, 4 * H))
        b = g.uniform(-k, k, 4 * H)
        b[H:2 * H] += 1.0  # forget-gate bias
        p[f"l{l}_b"] = b
    ka, kb = 1.0 / np.sqrt(H), 1.0 / np.sqrt(adapter)
    p["a_W1"] = g.uniform(-ka, ka, (H, adapter)); p["a_b1"] = np.zeros(adapter)
    p["a_W2"] = g.uniform(-kb, kb, (adapter, H)) * 0.1; p["a_b2"] = np.zeros(H)
    p["h_W"] = g.uniform(-ka, ka, (H, 2)) * 0.1; p["h_b"] = np.array([0.0, -6.0])
    return p


def n_params(p):
    return sum(v.size for v in p.values())


def flatten(p, keys=ORDER):
    return np.concatenate([p[k].ravel() for k in keys])


def unflatten(vec, like, keys=ORDER):
    out, i = {}, 0
    for k in keys:
        n = like[k].size
        out[k] = vec[i:i + n].reshape(like[k].shape)
        i += n
    return out


def sample_masks(B, n_in, H, p_drop, rng):
    if p_drop <= 0:
        return None
    keep = 1.0 - p_drop
    m = lambda d: (rng.random((B, d)) < keep) / keep
    return {"in0": m(n_in), "h0": m(H), "in1": m(H), "h1": m(H), "out": m(H)}


def forward(p, X, masks=None, cache=False):
    """X: (B, L, F). Returns mu, logv, h_embed (and cache for backward)."""
    B, L, F = X.shape
    H = p["l0_Wh"].shape[0]
    ones = lambda d: np.ones((B, d))
    mk = masks or {"in0": ones(F), "h0": ones(H), "in1": ones(H), "h1": ones(H), "out": ones(H)}
    inp = X
    caches = []
    for l in (0, 1):
        Wx, Wh, b = p[f"l{l}_Wx"], p[f"l{l}_Wh"], p[f"l{l}_b"]
        m_in, m_h = mk[f"in{l}"], mk[f"h{l}"]
        h = np.zeros((B, H)); c = np.zeros((B, H))
        hs, steps = [], []
        for t in range(L):
            xin = inp[:, t] * m_in
            hin = h * m_h
            z = xin @ Wx + hin @ Wh + b
            i_, f_, g_, o_ = sigmoid(z[:, :H]), sigmoid(z[:, H:2 * H]), np.tanh(z[:, 2 * H:3 * H]), sigmoid(z[:, 3 * H:])
            c_prev = c
            c = f_ * c + i_ * g_
            tc = np.tanh(c)
            h = o_ * tc
            hs.append(h)
            if cache:
                steps.append((xin, hin, i_, f_, g_, o_, c_prev, c, tc))
        caches.append(steps)
        inp = np.stack(hs, 1)
    h_last = inp[:, -1] * mk["out"]
    a1 = h_last @ p["a_W1"] + p["a_b1"]
    act, dact = gelu(a1)
    h2 = h_last + act @ p["a_W2"] + p["a_b2"]
    out = h2 @ p["h_W"] + p["h_b"]
    mu = out[:, 0]
    logv_raw = out[:, 1]
    logv = np.clip(logv_raw, LOGV_MIN, LOGV_MAX)
    if not cache:
        return mu, logv, h_last
    return mu, logv, h_last, {"layers": caches, "mk": mk, "h_last": h_last, "a1": a1, "act": act, "dact": dact,
                              "h2": h2, "logv_raw": logv_raw}


def loss_and_grad(p, X, y, masks=None, wd=1e-4, prox=None, prox_mu=0.0):
    """Gaussian NLL of the residual target + L2 + proximal term (Eq. 4). Returns loss, grads dict."""
    B, L, F = X.shape
    H = p["l0_Wh"].shape[0]
    mu, logv, _, cc = forward(p, X, masks, cache=True)
    iv = np.exp(-logv)
    err = y - mu
    loss = float(np.mean(0.5 * (err ** 2 * iv + logv)))
    g = {k: np.zeros_like(v) for k, v in p.items()}
    dmu = -err * iv / B
    dlogv = 0.5 * (1.0 - err ** 2 * iv) / B
    dlogv = dlogv * ((cc["logv_raw"] > LOGV_MIN) & (cc["logv_raw"] < LOGV_MAX))
    dout = np.stack([dmu, dlogv], 1)
    g["h_W"] = cc["h2"].T @ dout; g["h_b"] = dout.sum(0)
    dh2 = dout @ p["h_W"].T
    g["a_W2"] = cc["act"].T @ dh2; g["a_b2"] = dh2.sum(0)
    da1 = (dh2 @ p["a_W2"].T) * cc["dact"]
    g["a_W1"] = cc["h_last"].T @ da1; g["a_b1"] = da1.sum(0)
    dh_last = dh2 + da1 @ p["a_W1"].T
    mk = cc["mk"]
    dh_above = np.zeros((B, L, H))
    dh_above[:, -1] = dh_last * mk["out"]
    for l in (1, 0):
        Wx, Wh = p[f"l{l}_Wx"], p[f"l{l}_Wh"]
        steps = cc["layers"][l]
        dh_next = np.zeros((B, H)); dc_next = np.zeros((B, H))
        dx_below = np.zeros((B, L, Wx.shape[0]))
        for t in range(L - 1, -1, -1):
            xin, hin, i_, f_, g_, o_, c_prev, c, tc = steps[t]
            dh = dh_above[:, t] + dh_next
            do = dh * tc
            dc = dh * o_ * (1 - tc ** 2) + dc_next
            di, dg, df = dc * g_, dc * i_, dc * c_prev
            dc_next = dc * f_
            dz = np.concatenate([di * i_ * (1 - i_), df * f_ * (1 - f_), dg * (1 - g_ ** 2), do * o_ * (1 - o_)], 1)
            g[f"l{l}_Wx"] += xin.T @ dz
            g[f"l{l}_Wh"] += hin.T @ dz
            g[f"l{l}_b"] += dz.sum(0)
            dx_below[:, t] = (dz @ Wx.T) * mk[f"in{l}"]
            dh_next = (dz @ Wh.T) * mk[f"h{l}"]
        dh_above = dx_below
    for k in p:
        loss += wd * float(np.sum(p[k] ** 2))
        g[k] += 2 * wd * p[k]
    if prox is not None and prox_mu > 0:
        for k in BACKBONE:
            d = p[k] - prox[k]
            loss += 0.5 * prox_mu * float(np.sum(d ** 2))
            g[k] += prox_mu * d
    return loss, g


class Adam:
    def __init__(self, p, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, keys=None):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.keys = keys or list(p.keys())
        self.m = {k: np.zeros_like(p[k]) for k in self.keys}
        self.v = {k: np.zeros_like(p[k]) for k in self.keys}
        self.t = 0

    def step(self, p, g, extra=None):
        self.t += 1
        for k in self.keys:
            gk = g[k] if extra is None else g[k] + extra[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * gk
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * gk ** 2
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            p[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


def predict(p, X, T=50, p_drop=0.2, rng=None, batch=4096):
    """MC-dropout predictive moments: mean, epistemic var, aleatoric var (Eq. 5)."""
    rng = rng or np.random.default_rng(0)
    H = p["l0_Wh"].shape[0]
    mus, vs = [], []
    for _ in range(max(T, 1)):
        mu_t, v_t = [], []
        for s in range(0, len(X), batch):
            xb = X[s:s + batch]
            masks = sample_masks(len(xb), X.shape[2], H, p_drop, rng) if (T > 1 and p_drop > 0) else None
            mu, logv, _ = forward(p, xb, masks)
            mu_t.append(mu); v_t.append(np.exp(logv))
        mus.append(np.concatenate(mu_t)); vs.append(np.concatenate(v_t))
    mus, vs = np.stack(mus), np.stack(vs)
    return mus.mean(0), mus.var(0), vs.mean(0)


def embed(p, X):
    return forward(p, X, None)[2]
