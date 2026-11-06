"""Aggregate result CSVs into the Markdown tables of Section 6 (mean +/- std over seeds and folds)."""
import argparse
import glob

import numpy as np
import pandas as pd


def pm(g, col, nd=2):
    return f"{g[col].mean():.{nd}f} ± {g[col].std(ddof=0):.{nd}f}"


def table(df, cols, by=("method", "dataset"), nd=2):
    rows = []
    for key, g in df.groupby(list(by)):
        rows.append(dict(zip(by, key if isinstance(key, tuple) else (key,)), **{c: pm(g, c, nd) for c in cols}))
    return pd.DataFrame(rows).to_markdown(index=False)


def rounds_to_target(curve, target):
    out = []
    for (m, f, s), g in curve.groupby(["method", "fold", "seed"]):
        hit = g[g["rmse_%"] <= target]
        out.append({"method": m, "rounds": hit["round"].min() if len(hit) else np.nan,
                    "MB_to_target": hit["cum_uplink_MB"].min() if len(hit) else np.nan})
    return pd.DataFrame(out).groupby("method").agg(["mean", "std"]).to_markdown()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--target_rmse", type=float, default=2.0, help="SoH RMSE target in % for Table 6")
    a = ap.parse_args()
    for f in sorted(glob.glob(f"{a.results}/soh_*.csv")):
        sc = f.split("_")[-1][:-4]
        df = pd.read_csv(f)
        print(f"\n### Scenario {sc}: SoH accuracy (Table 4)\n")
        print(table(df, ["RMSE_%", "MAE_%"]))
        print(f"\n### Scenario {sc}: SoH uncertainty (Table 5)\n")
        print(table(df, ["PICP", "MPIW_%", "NLL", "CRPS_%"], nd=3))
        print(f"\n### Scenario {sc}: overall by method\n")
        print(table(df, ["RMSE_%", "MAE_%", "PICP", "MPIW_%"], by=("method",), nd=3))
    for f in sorted(glob.glob(f"{a.results}/rul_*.csv")):
        df = pd.read_csv(f)
        if df.empty:
            continue
        df["abs_err"] = (df["true"] - df["pred"]).abs()
        df["cover"] = ((df["true"] >= df["lo"]) & (df["true"] <= df["hi"])).astype(float)
        df["width"] = df["hi"] - df["lo"]
        df["al"] = (df["abs_err"] <= 0.2 * df["true"].clip(lower=1)).astype(float)
        df["sq"] = (df["true"] - df["pred"]) ** 2
        g = df.groupby(["method", "start_frac"]).agg(RUL_RMSE=("sq", lambda s: np.sqrt(s.mean())), RUL_MAE=("abs_err", "mean"),
                                                     PICP=("cover", "mean"), MPIW=("width", "mean"), alpha_lambda=("al", "mean"))
        print(f"\n### {f}: RUL (Table 6)\n")
        print(g.round(3).to_markdown())
        if "p_le_h" in df:
            from feddt.twin import decision_quality
            out = []
            for m, gm in df.groupby("method"):
                hc = float(gm["horizon_cycles"].iloc[0])
                out.append({"method": m, **decision_quality(gm.to_dict("records"), hc)})
            print(f"\n### {f}: Algorithm 3 decision quality (Table 7)\n")
            print(pd.DataFrame(out).round(3).to_markdown(index=False))
    for f in sorted(glob.glob(f"{a.results}/comm_*.csv")):
        df = pd.read_csv(f)
        if df.empty:
            continue
        print(f"\n### {f}: communication (Table 7)\n")
        print(df.groupby("method")[["P", "uplink_kB_per_client_round", "uplink_MB_per_round", "total_uplink_MB", "train_s"]]
              .mean().round(3).to_markdown())
    for f in sorted(glob.glob(f"{a.results}/curve_*.csv")):
        df = pd.read_csv(f)
        if df.empty:
            continue
        print(f"\n### {f}: rounds and uplink MB to reach SoH RMSE <= {a.target_rmse}%\n")
        print(rounds_to_target(df, a.target_rmse))


if __name__ == "__main__":
    main()
