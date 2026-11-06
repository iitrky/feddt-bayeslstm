"""Synthetic covariate-shift check for DW-FCC, Algorithm 2 (paper Table 10).

Calibration covariates x ~ N(0, 1); test covariates x ~ N(shift, 1) (a vehicle moving to a
hotter face). Noise sd grows with |x|, while the predictor uses a constant sigma, so the
normalised scores of the test vehicle are larger than in calibration. Compares the coverage
and width of 95% intervals from standard split conformal and from DW-FCC.
Run: python tests/synthetic_shift_check.py
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feddt.conformal import rff, tilt_eta, weighted_conformal_quantile


def run(shift, reps=200, m=1000, n_test=500, a=0.05, seed=0):
    rng = np.random.default_rng(seed)
    cov_u, cov_w, wid_u, wid_w, ess = [], [], [], [], []
    for r in range(reps):
        xc = rng.normal(0, 1, (m, 1)); xt = rng.normal(shift, 1, (n_test, 1))
        sd = lambda x: 0.1 + 0.15 * np.abs(x[:, 0])
        yc = xc[:, 0] + rng.normal(0, 1, m) * sd(xc); yt = xt[:, 0] + rng.normal(0, 1, n_test) * sd(xt)
        sig_hat = 0.2
        sc = np.abs(yc - xc[:, 0]) / sig_hat
        q_u = np.quantile(sc, min(np.ceil((m + 1) * (1 - a)) / m, 1))
        Pc = rff(xc, 128, 1.0, seed=7); Pt = rff(xt, 128, 1.0, seed=7)
        delta = Pt.mean(0) - Pc.mean(0)
        proj = Pc @ delta; target = Pt.mean(0) @ delta
        eta = tilt_eta(proj, target)
        w = np.exp(eta * (proj - proj.max()))
        w_test = float(np.exp(eta * (target - proj.max())))
        q_w = weighted_conformal_quantile(sc, w, w_test, a)
        cov_u.append(np.mean(np.abs(yt - xt[:, 0]) <= q_u * sig_hat))
        cov_w.append(np.mean(np.abs(yt - xt[:, 0]) <= q_w * sig_hat))
        wid_u.append(2 * q_u * sig_hat); wid_w.append(2 * q_w * sig_hat)
        ess.append(w.sum() ** 2 / (w ** 2).sum() / m)
    f = lambda v: f"{np.mean(v):.3f} ± {np.std(v):.3f}"
    return {"shift": shift, "PICP_split": f(cov_u), "PICP_DWFCC": f(cov_w), "width_split": f(wid_u),
            "width_DWFCC": f(wid_w), "ESS_fraction": f(ess)}


if __name__ == "__main__":
    print("| Shift | PICP split conformal | PICP DW-FCC | Width split | Width DW-FCC | ESS / m |")
    print("|---|---|---|---|---|---|")
    for s in (0.0, 0.5, 1.0, 1.5):
        r = run(s)
        print(f"| {s} | {r['PICP_split']} | {r['PICP_DWFCC']} | {r['width_split']} | {r['width_DWFCC']} | {r['ESS_fraction']} |")
