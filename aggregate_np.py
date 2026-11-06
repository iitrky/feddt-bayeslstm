"""Aggregate run_np.py JSON results into the values of Tables 4-9 (mean ± sd over folds and seeds).
Run: python aggregate_np.py results_np/  > tables.md"""
import glob
import json
import sys

import numpy as np
import pandas as pd

from feddt.twin import decision_quality

d = sys.argv[1] if len(sys.argv) > 1 else "results_np"
soh, rul, curve, secs = [], [], [], []
for f in glob.glob(f"{d}/*.json"):
    j = json.load(open(f))
    soh += j["soh"]; rul += j["rul"]; curve += j["curve"]
    secs.append(j["seconds"])
S, R, C = pd.DataFrame(soh), pd.DataFrame(rul), pd.DataFrame(curve)
pm = lambda x, nd=2: f"{np.mean(x):.{nd}f} ± {np.std(x):.{nd}f}" if len(x) else "–"
out = {}


def cellmean(df, col):
    """average over test cells within each (fold, seed) run first, then report mean ± sd over runs"""
    return df.groupby(["fold", "seed"])[col].mean().to_numpy()


s1 = S[S.scenario == "S1"]
names = ["local", "central", "gpr", "fedavg", "fedprox", "scaffold", "fedavg_mc", "scaffold_mc", "feddt"]
print("## Table 4 (S1, RMSE / MAE in %, per dataset)")
for m in names:
    row = [m]
    for ds in ["NASA", "CALCE"]:
        g = s1[(s1.method == m) & (s1.dataset == ds) & (s1.interval == "gaussian")]
        row.append(f"{pm(g['RMSE_%'])} / {pm(g['MAE_%'])} (n={len(g)})")
    print(" | ".join(row))

print("\n## Table 5 (S1, both datasets, mean over cells)")
for m, iv in [("gpr", "gaussian"), ("fedavg_mc", "gaussian"), ("scaffold_mc", "gaussian"), ("feddt", "gaussian"),
              ("feddt", "split"), ("feddt", "dwfcc")]:
    g = s1[(s1.method == m) & (s1.interval == iv)]
    print(m, iv, "| PICP", pm(g["PICP"], 3), "| MPIW", pm(g["MPIW_%"]), "| NLL", pm(g["NLL"]), "| CRPS", pm(g["CRPS_%"], 3),
          "| epi share", pm(g["epi_share"], 3) if "epi_share" in g else "", f"(n={len(g)})")
g = s1[(s1.method == "feddt") & (s1.interval == "dwfcc")]
if len(g) and "dw_eta" in g:
    print("DW-FCC eta", pm(g["dw_eta"]), "ESS frac", pm(g["dw_ess_frac"], 3), "q split vs dwfcc",
          pm(s1[(s1.method == "feddt") & (s1.interval == "split")]["q"], 3), pm(g["q"], 3))

print("\n## Table 6 (S1 RUL, pooled over cells, by start)")
r1 = R[R.scenario == "S1"] if len(R) else R
for m in ["fedavg_mc", "feddt"]:
    for fr in [0.3, 0.5, 0.7]:
        g = r1[(r1.method == m) & (r1.start_frac == fr)]
        if not len(g):
            continue
        err = g["true"] - g["pred"]
        cov = ((g["true"] >= g["lo"]) & (g["true"] <= g["hi"])).mean()
        al = (np.abs(err) <= 0.2 * g["true"].clip(lower=1)).mean()
        print(m, fr, f"RMSE {np.sqrt(np.mean(err ** 2)):.1f} | MAE {np.mean(np.abs(err)):.1f} | PICP {cov:.3f} | alpha-lambda {al:.3f} (n={len(g)})")
    for ds in ["NASA", "CALCE"]:
        g = r1[(r1.method == m) & (r1.dataset == ds)]
        if len(g):
            err = g["true"] - g["pred"]
            print("   ", m, ds, f"RMSE {np.sqrt(np.mean(err ** 2)):.1f} MAE {np.mean(np.abs(err)):.1f} rel {np.mean(np.abs(err) / g['true']):.3f}")

print("\n## Table 7 (Algorithm 3 decision quality, horizon 100 cycles, p_crit 0.10)")
for m in ["fedavg_mc", "feddt"]:
    g = r1[r1.method == m]
    if len(g):
        q = decision_quality(g.to_dict("records"), 100, 0.10)
        print(m, {k: round(v, 3) for k, v in q.items()}, "positives", int((g["true"] <= 100).sum()), "of", len(g))

print("\n## Table 8 (rounds and cumulative uplink to target RMSE)")
c1 = C[C.scenario == "S1"] if len(C) else C
fin = c1[c1["round"] == c1["round"].max()].groupby("method")["rmse_%"].mean()
print("final-round mean test RMSE per method:", fin.round(3).to_dict())
for target in (1.5, 2.0):
    print("target", target)
    for m in ["fedavg", "fedprox", "scaffold", "fedavg_mc", "feddt"]:
        rr, mb = [], []
        for (f, s), g in c1[c1.method == m].groupby(["fold", "seed"]):
            g = g.sort_values("round")
            hit = g[g["rmse_%"] <= target]
            rr.append(hit["round"].iloc[0] + 1 if len(hit) else np.nan)
            mb.append(hit["cum_uplink_MB"].iloc[0] if len(hit) else np.nan)
        rr, mb = np.array(rr, float), np.array(mb, float)
        print("  ", m, "rounds", f"{np.nanmean(rr):.1f}" if np.isfinite(rr).any() else "not reached",
              "| MB", f"{np.nanmean(mb):.3f}" if np.isfinite(mb).any() else "–", f"| reached in {np.isfinite(rr).sum()}/{len(rr)} runs")

print("\n## Table 9 (RMSE %, PICP) by scenario")
rows = [("FedAvg + MC dropout", "fedavg_mc", "gaussian"), ("FedDT full (DW-FCC)", "feddt", "dwfcc"),
        ("without physics prior", "feddt-no_physics", "dwfcc"), ("without drift term", "feddt-no_drift", "dwfcc"),
        ("without reliability", "feddt-no_reliability", "dwfcc"), ("shared head", "feddt-shared_head", "dwfcc"),
        ("split conformal instead of DW-FCC", "feddt", "split"), ("without conformal step", "feddt", "gaussian")]
for lab, m, iv in rows:
    cells = [lab]
    for sc in ["S1", "S2", "S3"]:
        g = S[(S.scenario == sc) & (S.method == m) & (S.interval == iv)]
        cells.append(f"{np.mean(g['RMSE_%']):.2f}, {np.mean(g['PICP']):.3f} (n={len(g)})" if len(g) else "–")
    print(" | ".join(cells))
print("\nphysics parameters (first FedDT run per fold):")
for f in sorted(glob.glob(f"{d}/S1_feddt_f*_s0.json")):
    j = json.load(open(f))
    print(f.split("/")[-1], j["comm"][0]["phys"] if j["comm"] else "")
print("\ntotal compute (s):", round(sum(secs)), "runs:", len(secs))
