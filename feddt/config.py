"""Shared configuration and communication accounting (no PyTorch dependency)."""
from __future__ import annotations

import math
from dataclasses import dataclass


# ------------------------------------------------------------------ config
@dataclass
class FedConfig:
    method: str = "feddt"          # feddt | fedavg | fedprox | scaffold
    rounds: int = 100
    local_epochs: int = 2
    participation: float = 0.5
    lr: float = 1e-3
    batch: int = 32
    weight_decay: float = 1e-4
    mu_prox: float = 0.01
    dropout: float = 0.2
    hidden: int = 64
    window: int = 10
    mc_T: int = 50
    beta: float = 0.5
    gamma: float = 1.0
    tau_r: float = 1.0
    alpha_floor: float = 0.2       # floor = alpha_floor / |S|
    rff_dim: int = 128
    topk: float = 0.1              # 1.0 = no sparsification
    qbits: int = 8                 # 32 = no quantisation
    chem_heads: bool = True
    use_drift: bool = True
    use_reliability: bool = True
    personal_epochs: int = 2
    conformal_a: float = 0.05
    seed: int = 0
    eval_every: int = 5
    use_physics: bool = True       # physics-guided prior, residual learning (Sect. 4.2)
    dwfcc: bool = True             # drift-weighted federated conformal calibration (Algorithm 2)

    @staticmethod
    def preset(method: str, **kw):
        c = FedConfig(method=method)
        if method in ("fedavg", "fedprox", "scaffold"):
            c.beta, c.use_drift, c.use_reliability, c.chem_heads = 1.0, False, False, False
            c.topk, c.qbits, c.alpha_floor = 1.0, 32, 0.0
            c.use_physics, c.dwfcc = False, False
            c.mu_prox = 0.01 if method == "fedprox" else 0.0
        for k, v in kw.items():
            setattr(c, k, v)
        return c



def uplink_bytes(P, rho, bits, n_tensor, rff_dim, drift=True):
    if rho >= 1.0 and bits >= 32:
        payload = P * 4
    else:
        k = int(math.ceil(rho * P))
        idx_bits = 0 if rho >= 1.0 else int(math.ceil(math.log2(P)))
        payload = math.ceil(k * (bits + idx_bits) / 8) + 4 * n_tensor
    return payload + (4 * (rff_dim + 1) if drift else 0)
