"""Finite-difference check of the NumPy Bayesian LSTM gradients (run: python tests/test_np_gradients.py)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feddt_np.model_np import init_params, loss_and_grad, sample_masks

rng = np.random.default_rng(0)
p = init_params(5, hidden=6, adapter=3, seed=1)
p["h_W"] *= 10; p["h_b"][:] = [0.1, -0.5]; p["a_W2"] *= 10
X = rng.normal(size=(4, 3, 5)); y = rng.normal(size=4) * 0.1
masks = sample_masks(4, 5, 6, 0.3, rng)
prox = {k: v + 0.01 for k, v in p.items()}
loss, g = loss_and_grad(p, X, y, masks, wd=1e-3, prox=prox, prox_mu=0.1)
worst = 0.0
for k in p:
    for idx in [tuple(rng.integers(0, s) for s in p[k].shape) for _ in range(6)]:
        old = p[k][idx]; e = 1e-6
        p[k][idx] = old + e; lp, _ = loss_and_grad(p, X, y, masks, 1e-3, prox, 0.1)
        p[k][idx] = old - e; lm, _ = loss_and_grad(p, X, y, masks, 1e-3, prox, 0.1)
        p[k][idx] = old
        num = (lp - lm) / (2 * e); ana = g[k][idx]
        rel = abs(num - ana) / max(1e-8, abs(num) + abs(ana))
        worst = max(worst, rel)
print("max relative gradient error:", worst)
assert worst < 1e-5, worst
print("NUMPY GRADIENT CHECK PASSED")
