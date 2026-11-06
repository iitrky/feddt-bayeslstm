"""Experiment runner for the NumPy backend (used when PyTorch is unavailable).

Same protocol as run_experiment.py: leave-one-cell-out folds (one test cell per dataset),
scenarios S1/S2/S3, FedDT-BayesLSTM, FedAvg/FedProx/SCAFFOLD (deterministic and MC dropout),
local-only, centralised Bayesian LSTM, centralised GPR, ablations. Writes CSVs per job.
Example:
  python run_np.py --cells cells.pkl --scenario S1 --methods feddt fedavg --seeds 0 1 2 --out results_np
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import pickle
import time

import numpy as np
import pandas as pd

from feddt import physics
from feddt.config import FedConfig
from feddt.data import make_windows
from feddt.features import N_HI
from feddt.twin import eol_index, rul_metrics, soh_metrics
from feddt_np.fed_np import ClientNP, FederationNP, rollout_rul_np, train_epochs
from feddt_np.model_np import init_params, predict

N_IN = 1 + N_HI
START_FRACS = (0.3, 0.5, 0.7)
HORIZON_CYCLES = 100
ABLATIONS = {
    "no_physics": {"use_physics": False},
    "no_drift": {"use_drift": False},
    "no_reliability": {"use_reliability": False},
    "shared_head": {"chem_heads": False},
    "no_dwfcc": {"dwfcc": False},
    "no_conformal": {"no_conformal": True},
    "no_compression": {"topk": 1.0, "qbits": 32},
}


def folds(recs):
    by = {}
    for i, r in enumerate(recs):
        by.setdefault(r.dataset, []).append(i)
    n = max(len(v) for v in by.values())
    for f in range(n):
        yield f, sorted(v[f % len(v)] for v in by.values())


def inject_sensor_drift(rec, frac=0.5, dv=0.010, dT=2.0):
    r = copy.deepcopy(rec)
    k = int(frac * len(r.soh))
    r.his[k:, 4] += dv
    r.his[k:, 7] += dT
    r.his[k:, 8] += dT
    return r


def eval_cell(p, rec, norm_fn, L, T, qs, p_drop, rng, phys=None, do_rul=False, rul_q=None, M=200):
    X, mask, y = make_windows(rec, L)
    cut = int(0.3 * len(y))
    Xn = norm_fn(X[cut:]) * mask
    mu, ve, va = predict(p, Xn, T=T, p_drop=p_drop, rng=rng)
    if phys is not None:
        mu = mu + physics.prior_series(rec, phys, L)[cut:]
    else:
        mu = mu + X[cut:, -1, 0]      # persistence reference for data-driven models
    sig = np.sqrt(ve + va)
    out = {}
    for name, q in qs.items():
        out[name] = {**soh_metrics(y[cut:], mu, sig, q=q), "epi_share": float(np.mean(ve / (ve + va + 1e-12))), "q": q}
    rows = []
    eidx = eol_index(rec.soh, rec.eol)
    block = rec.meta.get("block", 1)
    if do_rul and eidx is not None:
        for fr in START_FRACS:
            s = int(fr * eidx)
            if s < L:
                continue
            smp = rollout_rul_np(p, rec, s, L, norm_fn, mask, rec.eol, M=M, h_max=math.ceil(1500 / block), block=block,
                                 seed=int(rng.integers(1e9)), phys=phys, p_drop=p_drop)
            pred = float(np.median(smp))
            if rul_q is not None:
                sd = float(np.std(smp)) + 1e-9
                lo, hi = max(0.0, pred - rul_q * sd), pred + rul_q * sd
            else:
                lo, hi = np.percentile(smp, [2.5, 97.5])
            rows.append({"start_frac": fr, "true": (eidx - s) * block, "pred": pred, "lo": float(lo), "hi": float(hi),
                         "p_le_h": float(np.mean(smp <= HORIZON_CYCLES)), "horizon_cycles": HORIZON_CYCLES})
    return out, rows


def rul_conformal_q(fed, L, a=0.05, M=100):
    scores = []
    for c in fed.clients:
        rec = c.rec
        eidx = eol_index(rec.soh, rec.eol)
        if eidx is None:
            continue
        p = fed.params_for(c.chem)
        nf = lambda W, ch=c.chem: fed.xn(ch, W)
        for fr in START_FRACS:
            s = int(fr * eidx)
            if s < L:
                continue
            smp = rollout_rul_np(p, rec, s, L, nf, c.mask, rec.eol, M=M, h_max=1500, phys=fed.phys.get(c.chem),
                                 p_drop=fed.cfg.dropout, seed=s)
            scores.append(abs((eidx - s) - np.median(smp)) / (np.std(smp) + 1e-9))
    if not scores:
        return None
    n = len(scores)
    return float(np.quantile(scores, min(math.ceil((n + 1) * (1 - a)) / n, 1.0)))


def run_fed(method, train, test, args, seed, scenario, abl=None):
    kw = dict(rounds=args.rounds, seed=seed, mc_T=args.T)
    abl = abl or {}
    kw.update({k: v for k, v in abl.items() if k != "no_conformal"})
    base = method.replace("_mc", "")
    cfg = FedConfig.preset(base, **kw)
    if method in ("fedavg", "fedprox", "scaffold"):
        cfg.dropout, cfg.mc_T = 0.0, 1
    clients = [ClientNP.from_record(r, cfg.window, r.chemistry) for r in train]
    if scenario == "S3":
        for c in clients:
            if c.chem == args.late_chem:
                c.join_round = cfg.rounds // 2
    curve = []
    rng = np.random.default_rng(seed + 99)

    def eval_fn(fed, t):
        v = []
        for r in test:
            p = fed.params_for(r.chemistry)
            o, _ = eval_cell(p, r, lambda W, ch=r.chemistry: fed.xn(ch, W), cfg.window, min(cfg.mc_T, 5), {"g": 1.96},
                             cfg.dropout, rng, phys=fed.phys.get(r.chemistry) if cfg.use_physics else None)
            v.append(o["g"]["RMSE_%"])
        curve.append({"round": t, "rmse_%": float(np.mean(v)), "cum_uplink_MB": sum(fed.log["uplink_bytes"]) / 1e6})

    t0 = time.time()
    fed = FederationNP(clients, N_IN, cfg).run(eval_fn)
    train_s = time.time() - t0
    use_conf = method == "feddt" and not abl.get("no_conformal")
    q_split, _ = fed.conformal_quantile() if use_conf else (1.96, None)
    do_rul = method in ("feddt", "fedavg_mc") and args.rul
    rq = rul_conformal_q(fed, cfg.window) if (use_conf and do_rul) else None
    soh_rows, rul_rows = [], []
    for r in test:
        p = fed.params_for(r.chemistry)
        nf = lambda W, ch=r.chemistry: fed.xn(ch, W)
        qs = {"gaussian": 1.96}
        info = {}
        if use_conf:
            qs["split"] = q_split
            if cfg.dwfcc:
                X, mask, y = make_windows(r, cfg.window)
                phi = fed.target_embedding(r.chemistry, nf(X[:int(0.3 * len(y))]) * mask)
                qs["dwfcc"], info = fed.conformal_quantile(target_phi=phi)
        o, rr = eval_cell(p, r, nf, cfg.window, cfg.mc_T, qs, cfg.dropout, rng,
                          phys=fed.phys.get(r.chemistry) if cfg.use_physics else None, do_rul=do_rul, rul_q=rq)
        for name, m in o.items():
            soh_rows.append({"cell": r.cell_id, "dataset": r.dataset, "interval": name, **m, **{f"dw_{k}": v for k, v in info.items()}})
        rul_rows += [{"cell": r.cell_id, "dataset": r.dataset, **x} for x in rr]
    comm = {"uplink_kB_per_client_round": float(np.mean(fed.log["uplink_bytes"])) / 1e3 /
            max(1, math.ceil(cfg.participation * len(clients))), "train_s": train_s,
            "phys": json.dumps({k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in fed.phys.items()})}
    return soh_rows, rul_rows, comm, curve


def run_local(test, args, seed, epochs=60):
    rows = []
    for r in test:
        rng = np.random.default_rng(seed)
        cfg = FedConfig(seed=seed, mc_T=args.T)
        X, mask, y = make_windows(r, cfg.window)
        cut = int(0.3 * len(y))
        flat = X[:cut].reshape(-1, X.shape[-1])
        mu, sd = flat.mean(0), flat.std(0)
        mu[0], sd[0] = 0, 1
        sd = np.where(sd < 1e-6, 1, sd)
        nf = lambda W, mu=mu, sd=sd: (W - mu) / sd
        p = init_params(N_IN, cfg.hidden, seed=seed)
        train_epochs(p, nf(X[:cut]) * mask, y[:cut] - X[:cut, -1, 0], cfg, epochs, rng)
        o, _ = eval_cell(p, r, nf, cfg.window, cfg.mc_T, {"gaussian": 1.96}, cfg.dropout, rng)
        rows.append({"cell": r.cell_id, "dataset": r.dataset, "interval": "gaussian", **o["gaussian"]})
    return rows


def run_central(train, test, args, seed, epochs=None):
    rng = np.random.default_rng(seed)
    cfg = FedConfig(seed=seed, mc_T=args.T)
    Xs, ys = [], []
    for r in train:
        X, mask, y = make_windows(r, cfg.window)
        Xs.append(X * mask); ys.append(y)
    X, y = np.concatenate(Xs), np.concatenate(ys)
    flat = X.reshape(-1, X.shape[-1]); mu, sd = flat.mean(0), flat.std(0)
    mu[0], sd[0] = 0, 1
    sd = np.where(sd < 1e-6, 1, sd)
    nf = lambda W: (W - mu) / sd
    p = init_params(N_IN, cfg.hidden, seed=seed)
    ep = epochs or max(2, int(args.rounds * cfg.local_epochs * cfg.participation))
    train_epochs(p, nf(X), y - X[:, -1, 0], cfg, ep, rng)
    rows = []
    for r in test:
        o, _ = eval_cell(p, r, nf, cfg.window, cfg.mc_T, {"gaussian": 1.96}, cfg.dropout, rng)
        rows.append({"cell": r.cell_id, "dataset": r.dataset, "interval": "gaussian", **o["gaussian"]})
    return rows


def run_gpr(train, test, seed, max_n=2000):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    Xs, ys = [], []
    for r in train:
        X, _, y = make_windows(r, 10)
        Xs.append(X[:, -1]); ys.append(y)
    X, y = np.concatenate(Xs), np.concatenate(ys)
    X = np.nan_to_num(X)
    idx = np.random.default_rng(seed).choice(len(y), size=min(max_n, len(y)), replace=False)
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd < 1e-6, 1, sd)
    gp = GaussianProcessRegressor(RBF(np.ones(X.shape[1])) + WhiteKernel(1e-4), normalize_y=True, random_state=seed)
    gp.fit((X[idx] - mu) / sd, y[idx] - X[idx, 0])
    rows = []
    for r in test:
        Xt, _, yt = make_windows(r, 10)
        cut = int(0.3 * len(yt))
        m, s = gp.predict((np.nan_to_num(Xt[cut:, -1]) - mu) / sd, return_std=True)
        m = m + Xt[cut:, -1, 0]
        rows.append({"cell": r.cell_id, "dataset": r.dataset, "interval": "gaussian", **soh_metrics(yt[cut:], m, s)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True)
    ap.add_argument("--methods", nargs="+", default=["feddt"])
    ap.add_argument("--ablations", nargs="*", default=[])
    ap.add_argument("--scenario", default="S1", choices=["S1", "S2", "S3"])
    ap.add_argument("--late_chem", default="NASA-18650")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--folds", nargs="*", type=int, default=None)
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--T", type=int, default=50)
    ap.add_argument("--rul", action="store_true")
    ap.add_argument("--out", default="results_np")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    recs = pickle.load(open(args.cells, "rb"))
    jobs = [(m, None) for m in args.methods] + [("feddt", a) for a in args.ablations]
    for f, test_idx in folds(recs):
        if args.folds is not None and f not in args.folds:
            continue
        train = [recs[i] for i in range(len(recs)) if i not in test_idx]
        test = [recs[i] for i in test_idx]
        if args.scenario == "S2":
            g = np.random.default_rng(123 + f)
            dr = set(g.choice(len(train), size=max(1, len(train) // 4), replace=False).tolist())
            train = [inject_sensor_drift(r) if i in dr else r for i, r in enumerate(train)]
        for seed in args.seeds:
            for method, abl in jobs:
                tag = method if abl is None else f"feddt-{abl}"
                fn = f"{args.out}/{args.scenario}_{tag}_f{f}_s{seed}.json"
                if os.path.exists(fn):
                    continue
                t0 = time.time()
                if method == "local":
                    s, r, c, cv = run_local(test, args, seed), [], {}, []
                elif method == "central":
                    s, r, c, cv = run_central(train, test, args, seed), [], {}, []
                elif method == "gpr":
                    s, r, c, cv = run_gpr(train, test, seed), [], {}, []
                else:
                    s, r, c, cv = run_fed(method, train, test, args, seed, args.scenario, ABLATIONS.get(abl))
                base = {"method": tag, "fold": f, "seed": seed, "scenario": args.scenario}
                json.dump({"soh": [{**base, **x} for x in s], "rul": [{**base, **x} for x in r],
                           "comm": [{**base, **c}] if c else [], "curve": [{**base, **x} for x in cv],
                           "seconds": time.time() - t0}, open(fn, "w"))
                print(f"{tag} fold {f} seed {seed}: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
