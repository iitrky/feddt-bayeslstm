"""Run the experiments of Sections 5-6 and write CSVs that fill Tables 4-8.

Example:
    python run_experiment.py --config data_paths.json --methods feddt fedavg fedprox scaffold local central gpr \
        --scenario S1 --seeds 0 1 2 3 4 --out results/
    python make_tables.py --results results/
data_paths.json (all keys optional):
{
  "nasa":   ["data/nasa/B0005.mat", "data/nasa/B0006.mat", "data/nasa/B0007.mat", "data/nasa/B0018.mat"],
  "calce":  ["data/calce/CS2_35", "data/calce/CS2_36", "data/calce/CS2_37", "data/calce/CS2_38"],
  "oxford": "data/oxford/Oxford_Battery_Degradation_Dataset_1.mat",
  "inhouse": {"csv": "data/lab/cycles.csv", "rated_ah": 2.5, "chemistry": "NMC"}
}
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch

from feddt.data import load_calce_cs2, load_inhouse_csv, load_nasa, load_oxford, make_windows
from feddt.fed import Client, FedConfig, Federation, train_local
from feddt.features import N_HI
from feddt.model import BayesLSTM, count_params
from feddt.twin import eol_index, rollout_rul, rul_metrics, soh_metrics

START_FRACS = (0.3, 0.5, 0.7)
HORIZON_CYCLES = 100  # decision horizon for Algorithm 2 evaluation (e.g. 3 shifts x ~33 cycles)
N_IN = 1 + N_HI


def load_all(cfg):
    recs = []
    for p in cfg.get("nasa", []):
        recs.append(load_nasa(p))
    for p in cfg.get("calce", []):
        recs.append(load_calce_cs2(p))
    if cfg.get("oxford"):
        recs.extend(load_oxford(cfg["oxford"]))
    if cfg.get("inhouse"):
        ih = cfg["inhouse"]
        recs.extend(load_inhouse_csv(ih["csv"], ih["rated_ah"], ih["chemistry"], ih.get("eol", 0.8)))
    return recs


def inject_sensor_drift(rec, frac=0.5, dv=0.010, dT=2.0):
    """Scenario S2: +10 mV voltage offset and +2 degC temperature bias from `frac` of life."""
    r = copy.deepcopy(rec)
    k = int(frac * len(r.soh))
    r.his[k:, 4] += dv               # IC peak voltage
    r.his[k:, 5] += 0.0              # voltage drop is offset-invariant
    r.his[k:, 7] += dT
    r.his[k:, 8] += dT
    return r


def folds(recs):
    by_ds = {}
    for i, r in enumerate(recs):
        by_ds.setdefault(r.dataset, []).append(i)
    n = max(len(v) for v in by_ds.values())
    for f in range(n):
        yield f, sorted(v[f % len(v)] for v in by_ds.values())


def norm_fn_for(fed, chem):
    mu, sd = fed.norm.get(chem, next(iter(fed.norm.values())))
    mu_t, sd_t = torch.tensor(mu), torch.tensor(sd)
    return lambda W: (W - mu_t) / sd_t


def evaluate_test_cell(model, rec, mask, norm, L, T, q, rul_q=None, M=200, seed=0, do_rul=True):
    X, _, y = make_windows(rec, L)
    cut = int(0.3 * len(y))
    model.in_mask.copy_(torch.tensor(mask, dtype=torch.float32))
    model.eval()
    mu, ve, va = model.predict(norm(torch.tensor(X[cut:])), T=T)
    sig = (ve + va).sqrt().numpy()
    soh = soh_metrics(y[cut:], mu.numpy(), sig, q=q)
    soh["epi_share"] = float((ve / (ve + va + 1e-12)).mean())
    rows = []
    eidx = eol_index(rec.soh, rec.eol)
    block = rec.meta.get("block", 1)
    if do_rul and eidx is not None:
        for fr in START_FRACS:
            s = int(fr * eidx)
            if s < L:
                continue
            samples = rollout_rul(model, rec, s, L, norm, rec.eol, M=M, h_max=math.ceil(3000 / block), block=block, seed=seed)
            pred = float(np.median(samples))
            if rul_q is not None:
                sd = float(np.std(samples)) + 1e-9
                lo, hi = pred - rul_q * sd, pred + rul_q * sd
            else:
                lo, hi = np.percentile(samples, [2.5, 97.5])
            h_cyc = HORIZON_CYCLES
            rows.append({"start_frac": fr, "true": (eidx - s) * block, "pred": pred, "lo": float(lo), "hi": float(hi),
                         "p_le_h": float(np.mean(samples <= h_cyc)), "horizon_cycles": h_cyc})
    return soh, rows


def rul_conformal_q(fed, L, a=0.05, M=100):
    """Federated split-conformal factor for RUL using training cells that reached EoL."""
    scores = []
    for c in fed.clients:
        rec = c.rec
        eidx = eol_index(rec.soh, rec.eol)
        if eidx is None:
            continue
        m = fed.model_for(c.chem)
        m.in_mask.copy_(torch.tensor(c.mask, dtype=torch.float32))
        nf = norm_fn_for(fed, c.chem)
        block = rec.meta.get("block", 1)
        for fr in START_FRACS:
            s = int(fr * eidx)
            if s < L:
                continue
            smp = rollout_rul(m, rec, s, L, nf, rec.eol, M=M, h_max=math.ceil(3000 / block), block=block)
            scores.append(abs((eidx - s) * block - np.median(smp)) / (np.std(smp) + 1e-9))
    if not scores:
        return None
    n = len(scores)
    return float(np.quantile(scores, min(math.ceil((n + 1) * (1 - a)) / n, 1.0)))


def run_fed(method, train_recs, test_recs, args, seed, scenario, ablation=None):
    kw = dict(rounds=args.rounds, seed=seed, mc_T=args.T)
    if ablation:
        kw.update(ablation)
    cfg = FedConfig.preset(method if method != "feddt" else "feddt", **kw)
    if method == "fedavg_mc" or method == "fedprox_mc" or method == "scaffold_mc":
        cfg = FedConfig.preset(method.replace("_mc", ""), **kw)
    if method in ("fedavg", "fedprox", "scaffold"):
        cfg.dropout = 0.0
        cfg.mc_T = 1
    clients = [Client.from_record(r, cfg.window, r.chemistry) for r in train_recs]
    if scenario == "S3" and args.late_chem:
        for c in clients:
            if c.chem == args.late_chem:
                c.join_round = cfg.rounds // 2
    curve = []

    def eval_fn(fed, t):
        vals = []
        for r in test_recs:
            key = r.chemistry if r.chemistry in fed.heads else (fed.cold_start_key(
                norm_fn_for(fed, r.chemistry)(torch.tensor(make_windows(r, cfg.window)[0][:50]))) if cfg.chem_heads else "shared")
            m = fed.model_for(key)
            s, _ = evaluate_test_cell(m, r, make_windows(r, cfg.window)[1], norm_fn_for(fed, r.chemistry), cfg.window,
                                      T=min(cfg.mc_T, 10), q=1.96, do_rul=False)
            vals.append(s["RMSE_%"])
        curve.append({"round": t, "rmse_%": float(np.mean(vals)),
                      "cum_uplink_MB": sum(fed.log["uplink_bytes"]) / 1e6,
                      "cum_downlink_MB": sum(fed.log["downlink_bytes"]) / 1e6})

    t0 = time.time()
    fed = Federation(clients, N_IN, cfg).run(eval_fn)
    train_s = time.time() - t0
    q = fed.conformal_quantile() if method == "feddt" and not (ablation or {}).get("no_conformal") else 1.96
    rq = rul_conformal_q(fed, cfg.window) if method == "feddt" and not (ablation or {}).get("no_conformal") else None
    soh_rows, rul_rows = [], []
    for r in test_recs:
        key = r.chemistry if (not cfg.chem_heads or r.chemistry in fed.heads) else "shared"
        m = fed.model_for(key)
        s, rr = evaluate_test_cell(m, r, make_windows(r, cfg.window)[1], norm_fn_for(fed, r.chemistry), cfg.window,
                                   T=cfg.mc_T, q=q, rul_q=rq, seed=seed)
        soh_rows.append({"cell": r.cell_id, "dataset": r.dataset, **s})
        rul_rows += [{"cell": r.cell_id, "dataset": r.dataset, **x} for x in rr]
    bb, hd, nt = count_params(fed.model)
    comm = {"P": bb + hd, "P_backbone": bb, "P_head": hd, "n_tensor": nt,
            "uplink_MB_per_round": float(np.mean(fed.log["uplink_bytes"])) / 1e6,
            "uplink_kB_per_client_round": float(np.mean(fed.log["uplink_bytes"])) / 1e3 / max(1, math.ceil(cfg.participation * len(clients))),
            "total_uplink_MB": sum(fed.log["uplink_bytes"]) / 1e6, "total_downlink_MB": sum(fed.log["downlink_bytes"]) / 1e6,
            "train_s": train_s, "conformal_q": q}
    return soh_rows, rul_rows, comm, curve, fed


def run_central(train_recs, test_recs, args, seed, epochs=None):
    torch.manual_seed(seed)
    cfg = FedConfig(seed=seed, mc_T=args.T)
    clients = [Client.from_record(r, cfg.window, "pooled") for r in train_recs]
    X = np.concatenate([np.concatenate([c.Xtr, c.Xcal, c.Xval]) for c in clients])
    y = np.concatenate([np.concatenate([c.ytr, c.ycal, c.yval]) for c in clients])
    mu, sd = X.reshape(-1, X.shape[-1]).mean(0), X.reshape(-1, X.shape[-1]).std(0) + 1e-6
    mu[0], sd[0] = 0, 1
    norm = lambda W: (W - torch.tensor(mu, dtype=torch.float32)) / torch.tensor(sd, dtype=torch.float32)
    m = BayesLSTM(N_IN, cfg.hidden, 2, cfg.dropout)
    gen = torch.Generator().manual_seed(seed)
    train_local(m, norm(torch.tensor(X)), torch.tensor(y), cfg, epochs or args.rounds, gen)
    soh_rows, rul_rows = [], []
    for r in test_recs:
        s, rr = evaluate_test_cell(m, r, make_windows(r, cfg.window)[1], norm, cfg.window, T=cfg.mc_T, q=1.96, seed=seed)
        soh_rows.append({"cell": r.cell_id, "dataset": r.dataset, **s})
        rul_rows += [{"cell": r.cell_id, "dataset": r.dataset, **x} for x in rr]
    return soh_rows, rul_rows


def run_local(test_recs, args, seed, epochs=100):
    """Local-only: each unseen vehicle trains on its own first 30% of life."""
    soh_rows, rul_rows = [], []
    for r in test_recs:
        torch.manual_seed(seed)
        cfg = FedConfig(seed=seed, mc_T=args.T)
        X, mask, y = make_windows(r, cfg.window)
        cut = int(0.3 * len(y))
        mu, sd = X[:cut].reshape(-1, X.shape[-1]).mean(0), X[:cut].reshape(-1, X.shape[-1]).std(0) + 1e-6
        mu[0], sd[0] = 0, 1
        norm = lambda W, mu=mu, sd=sd: (W - torch.tensor(mu, dtype=torch.float32)) / torch.tensor(sd, dtype=torch.float32)
        m = BayesLSTM(N_IN, cfg.hidden, 2, cfg.dropout)
        m.in_mask.copy_(torch.tensor(mask, dtype=torch.float32))
        train_local(m, norm(torch.tensor(X[:cut])), torch.tensor(y[:cut]), cfg, epochs, torch.Generator().manual_seed(seed))
        s, rr = evaluate_test_cell(m, r, mask, norm, cfg.window, T=cfg.mc_T, q=1.96, seed=seed)
        soh_rows.append({"cell": r.cell_id, "dataset": r.dataset, **s})
        rul_rows += [{"cell": r.cell_id, "dataset": r.dataset, **x} for x in rr]
    return soh_rows, rul_rows


def run_gpr(train_recs, test_recs, seed, max_n=2000):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    Xs, ys = [], []
    for r in train_recs:
        X, _, y = make_windows(r, 10)
        Xs.append(X[:, -1]); ys.append(y)
    X, y = np.concatenate(Xs), np.concatenate(ys)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), size=min(max_n, len(y)), replace=False)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    gp = GaussianProcessRegressor(RBF(np.ones(X.shape[1])) + WhiteKernel(1e-4), normalize_y=True, random_state=seed)
    gp.fit((X[idx] - mu) / sd, y[idx])
    rows = []
    for r in test_recs:
        Xt, _, yt = make_windows(r, 10)
        cut = int(0.3 * len(yt))
        m, s = gp.predict((Xt[cut:, -1] - mu) / sd, return_std=True)
        rows.append({"cell": r.cell_id, "dataset": r.dataset, **soh_metrics(yt[cut:], m, s)})
    return rows


ABLATIONS = {
    "no_drift": {"use_drift": False},
    "no_reliability": {"use_reliability": False},
    "fedavg_weighting": {"use_drift": False, "use_reliability": False, "beta": 1.0, "alpha_floor": 0.0},
    "shared_head": {"chem_heads": False},
    "no_conformal": {"no_conformal": True},
    "no_compression": {"topk": 1.0, "qbits": 32},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--methods", nargs="+", default=["feddt", "fedavg", "fedavg_mc", "fedprox", "fedprox_mc",
                                                     "scaffold", "scaffold_mc", "local", "central", "gpr"])
    ap.add_argument("--ablations", nargs="*", default=[])
    ap.add_argument("--scenario", default="S1", choices=["S1", "S2", "S3"])
    ap.add_argument("--late_chem", default=None, help="chemistry that joins at R/2 in S3")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--T", type=int, default=50)
    ap.add_argument("--max_folds", type=int, default=None)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    recs = load_all(json.load(open(args.config)))
    print(f"loaded {len(recs)} cells:", [(r.cell_id, r.dataset, len(r.soh)) for r in recs])
    SOH, RUL, COMM, CURVE = [], [], [], []
    rng = np.random.default_rng(123)
    for f, test_idx in folds(recs):
        if args.max_folds is not None and f >= args.max_folds:
            break
        train = [recs[i] for i in range(len(recs)) if i not in test_idx]
        test = [recs[i] for i in test_idx]
        if args.scenario == "S2":
            drifted = set(rng.choice(len(train), size=max(1, len(train) // 4), replace=False).tolist())
            train = [inject_sensor_drift(r) if i in drifted else r for i, r in enumerate(train)]
        jobs = [(m, None) for m in args.methods] + [("feddt", a) for a in args.ablations]
        for seed in args.seeds:
            for method, abl in jobs:
                tag = method if abl is None else f"feddt-{abl}"
                print(f"fold {f} seed {seed} {tag}", flush=True)
                if method in ("local",):
                    s, r = run_local(test, args, seed)
                    c, cv = {}, []
                elif method == "central":
                    s, r = run_central(train, test, args, seed)
                    c, cv = {}, []
                elif method == "gpr":
                    s, r, c, cv = run_gpr(train, test, seed), [], {}, []
                else:
                    s, r, c, cv, _ = run_fed(method, train, test, args, seed, args.scenario, ABLATIONS.get(abl))
                base = {"method": tag, "fold": f, "seed": seed, "scenario": args.scenario}
                SOH += [{**base, **x} for x in s]
                RUL += [{**base, **x} for x in r]
                if c:
                    COMM.append({**base, **c})
                CURVE += [{**base, **x} for x in cv]
                pd.DataFrame(SOH).to_csv(f"{args.out}/soh_{args.scenario}.csv", index=False)
                pd.DataFrame(RUL).to_csv(f"{args.out}/rul_{args.scenario}.csv", index=False)
                pd.DataFrame(COMM).to_csv(f"{args.out}/comm_{args.scenario}.csv", index=False)
                pd.DataFrame(CURVE).to_csv(f"{args.out}/curve_{args.scenario}.csv", index=False)


if __name__ == "__main__":
    main()
