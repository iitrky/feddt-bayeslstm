"""Variational Bayesian LSTM with chemistry adapter and heteroscedastic head (paper Sec. 4.2)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

BACKBONE_PREFIX = ("lstm.",)
HEAD_PREFIX = ("adapter.", "head.")


class VariationalLSTM(nn.Module):
    """Stacked LSTM with Gal-Ghahramani variational dropout: one mask per sequence,
    shared over time, on inputs, recurrent state and outputs."""

    def __init__(self, n_in: int, hidden: int = 64, layers: int = 2, p: float = 0.2):
        super().__init__()
        self.hidden, self.p = hidden, p
        self.cells = nn.ModuleList([nn.LSTMCell(n_in if l == 0 else hidden, hidden) for l in range(layers)])

    def _mask(self, shape, ref, active):
        if not active or self.p == 0:
            return torch.ones(shape, device=ref.device)
        return torch.bernoulli(torch.full(shape, 1 - self.p, device=ref.device)) / (1 - self.p)

    def sample_masks(self, B, n_in, device):
        """Masks fixed across calls: used for RUL trajectories (one network sample per trajectory)."""
        ms = []
        for l in range(len(self.cells)):
            d_in = n_in if l == 0 else self.hidden
            ms.append((self._mask((B, d_in), torch.empty(0, device=device), True),
                       self._mask((B, self.hidden), torch.empty(0, device=device), True)))
        ms.append(self._mask((B, self.hidden), torch.empty(0, device=device), True))
        return ms

    def forward(self, x, mc: bool = False, masks=None):
        B, L, _ = x.shape
        active = mc or self.training
        inp = x
        for li, cell in enumerate(self.cells):
            if masks is not None:
                m_in, m_h = masks[li]
            else:
                m_in = self._mask((B, inp.shape[-1]), x, active)
                m_h = self._mask((B, self.hidden), x, active)
            h = torch.zeros(B, self.hidden, device=x.device)
            c = torch.zeros_like(h)
            outs = []
            for t in range(L):
                h, c = cell(inp[:, t] * m_in, (h * m_h, c))
                outs.append(h)
            inp = torch.stack(outs, 1)
        m_out = masks[-1] if masks is not None else self._mask((B, self.hidden), x, active)
        return inp[:, -1] * m_out


class BayesLSTM(nn.Module):
    def __init__(self, n_in: int, hidden: int = 64, layers: int = 2, p: float = 0.2, adapter: int = 16):
        super().__init__()
        self.lstm = VariationalLSTM(n_in, hidden, layers, p)
        self.adapter = nn.Sequential(nn.Linear(hidden, adapter), nn.GELU(), nn.Linear(adapter, hidden))
        self.head = nn.Linear(hidden, 2)  # mu, log sigma^2
        self.register_buffer("in_mask", torch.ones(n_in))

    def embed(self, x, mc=False, masks=None):
        return self.lstm(x * self.in_mask, mc, masks)

    def forward(self, x, mc=False, masks=None):
        h = self.embed(x, mc, masks)
        h = h + self.adapter(h)
        out = self.head(h)
        return out[:, 0], out[:, 1].clamp(-14, 4), h

    @torch.no_grad()
    def predict(self, x, T: int = 50):
        """MC-dropout predictive moments: mean, epistemic var, aleatoric var."""
        mus, vs = [], []
        for _ in range(max(T, 1)):
            mu, logv, _ = self.forward(x, mc=T > 1)
            mus.append(mu)
            vs.append(logv.exp())
        mus, vs = torch.stack(mus), torch.stack(vs)
        return mus.mean(0), mus.var(0, unbiased=False), vs.mean(0)


def gaussian_nll(y, mu, logv):
    return 0.5 * ((y - mu) ** 2 / logv.exp() + logv + math.log(2 * math.pi)).mean()


def is_backbone(name: str) -> bool:
    return name.startswith(BACKBONE_PREFIX)


def is_head(name: str) -> bool:
    return name.startswith(HEAD_PREFIX)


def count_params(model: nn.Module):
    bb = sum(p.numel() for n, p in model.named_parameters() if is_backbone(n))
    hd = sum(p.numel() for n, p in model.named_parameters() if is_head(n))
    n_tensors = sum(1 for _ in model.parameters())
    return bb, hd, n_tensors
