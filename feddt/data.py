"""Dataset loaders -> per-cell `CellRecord` (one federated client per cell).

Supported sources (download them yourself; paths are passed on the command line):
  * NASA PCoE battery set (B0005.mat, B0006.mat, B0007.mat, B0018.mat)
  * CALCE CS2 cells (one folder per cell with the original .xlsx files)
  * Oxford Battery Degradation Dataset 1 (Oxford_Battery_Degradation_Dataset_1.mat)
  * In-house cycler export as CSV (see `load_inhouse_csv` for the column schema)
Every loader returns cycle-level health indicators (12 HIs, NaN where unavailable),
capacity per cycle, SoH = Q/Q_rated and metadata (chemistry, EoL threshold).
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.io import loadmat

from .features import N_HI, capacity_from_discharge, extract_cycle_features


@dataclass
class CellRecord:
    cell_id: str
    dataset: str
    chemistry: str
    rated_ah: float
    eol: float
    cycles: np.ndarray            # cycle index (equivalent cycles)
    soh: np.ndarray               # SoH per cycle in [0, 1]
    his: np.ndarray               # (n_cycles, N_HI), NaN = unavailable
    meta: dict = field(default_factory=dict)

    def clean(self, min_soh=0.3, max_soh=1.2):
        ok = np.isfinite(self.soh) & (self.soh > min_soh) & (self.soh < max_soh)
        self.cycles, self.soh, self.his = self.cycles[ok], self.soh[ok], self.his[ok]
        return self


# ---------------------------------------------------------------- NASA
def load_nasa(path_mat: str, rated_ah: float = 2.0, chemistry: str = "NASA-18650", eol: float = 0.70) -> CellRecord:
    name = os.path.splitext(os.path.basename(path_mat))[0]
    raw = loadmat(path_mat, simplify_cells=True)[name]["cycle"]
    caps, his, last_charge = [], [], None
    for c in raw:
        d = c["data"]
        if c["type"] == "charge":
            last_charge = {"t": np.asarray(d["Time"], float), "v": np.asarray(d["Voltage_measured"], float),
                           "i": np.asarray(d["Current_measured"], float), "T": np.asarray(d["Temperature_measured"], float)}
        elif c["type"] == "discharge":
            dis = {"t": np.asarray(d["Time"], float), "v": np.asarray(d["Voltage_measured"], float),
                   "i": np.asarray(d["Current_measured"], float), "T": np.asarray(d["Temperature_measured"], float)}
            cap = float(np.atleast_1d(d.get("Capacity", np.nan))[0])
            if not np.isfinite(cap):
                cap = capacity_from_discharge(dis)
            caps.append(cap)
            his.append(extract_cycle_features(last_charge, dis, rated_ah))
    caps = np.asarray(caps)
    return CellRecord(name, "NASA", chemistry, rated_ah, eol, np.arange(len(caps)), caps / rated_ah, np.vstack(his),
                      meta={"ambient_c": 24.0}).clean()


# ---------------------------------------------------------------- CALCE CS2
def load_calce_cs2(cell_dir: str, rated_ah: float = 1.1, chemistry: str = "LCO", eol: float = 0.80) -> CellRecord:
    files = sorted(glob.glob(os.path.join(cell_dir, "*.xls*")))
    if not files:
        raise FileNotFoundError(f"no xlsx files in {cell_dir}")
    frames, offset = [], 0
    for fp in files:
        book = pd.read_excel(fp, sheet_name=None)
        for sh, df in book.items():
            if not sh.lower().startswith("channel"):
                continue
            df = df.rename(columns=lambda s: str(s).strip())
            df["cyc"] = df["Cycle_Index"].astype(int) + offset
            frames.append(df)
        if frames:
            offset = int(frames[-1]["cyc"].max()) + 1
    df = pd.concat(frames, ignore_index=True)
    caps, his = [], []
    for _, g in df.groupby("cyc"):
        t = g["Test_Time(s)"].to_numpy(float)
        i = g["Current(A)"].to_numpy(float)
        v = g["Voltage(V)"].to_numpy(float)
        ch, dc = i > 1e-3, i < -1e-3
        if dc.sum() < 10 or ch.sum() < 10:
            continue
        charge = {"t": t[ch], "v": v[ch], "i": i[ch]}
        dis = {"t": t[dc], "v": v[dc], "i": i[dc]}
        cap = float(g["Discharge_Capacity(Ah)"].max())
        if "Internal_Resistance(Ohm)" in g:
            r = g["Internal_Resistance(Ohm)"].replace(0, np.nan).median()
        else:
            r = np.nan
        f = extract_cycle_features(charge, dis, rated_ah)
        if np.isfinite(r):
            f[6] = r
        caps.append(cap)
        his.append(f)
    caps = np.asarray(caps)
    cid = os.path.basename(os.path.normpath(cell_dir))
    return CellRecord(cid, "CALCE", chemistry, rated_ah, eol, np.arange(len(caps)), caps / rated_ah, np.vstack(his),
                      meta={"ambient_c": 25.0}).clean()


# ---------------------------------------------------------------- Oxford
def load_oxford(path_mat: str, rated_ah: float = 0.74, chemistry: str = "LCO-NCO", eol: float = 0.80) -> list[CellRecord]:
    """Oxford Dataset 1: characterisation every 100 cycles (C1ch / C1dc at 1C, 40 degC).
    If scipy cannot read the file (MATLAB v7.3), re-save it in MATLAB/Octave with -v7."""
    raw = loadmat(path_mat, simplify_cells=True)
    out = []
    for cell in sorted(k for k in raw if k.startswith("Cell")):
        caps, his, cyc = [], [], []
        for key in sorted(k for k in raw[cell] if k.startswith("cyc")):
            r = raw[cell][key]
            if "C1dc" not in r or "C1ch" not in r:
                continue
            ch, dc = r["C1ch"], r["C1dc"]
            rated_ma = rated_ah * 1000.0
            charge = {"t": np.asarray(ch["t"], float) * (86400.0 if np.nanmax(ch["t"]) < 5 else 1.0),
                      "v": np.asarray(ch["v"], float), "i": np.full(len(ch["v"]), rated_ah), "T": np.asarray(ch["T"], float)}
            dis = {"t": np.asarray(dc["t"], float) * (86400.0 if np.nanmax(dc["t"]) < 5 else 1.0),
                   "v": np.asarray(dc["v"], float), "i": np.full(len(dc["v"]), -rated_ah), "T": np.asarray(dc["T"], float)}
            cap = float(np.nanmax(np.abs(np.asarray(dc["q"], float)))) / 1000.0
            f = extract_cycle_features(charge, dis, rated_ah)
            f[2] = np.nan  # no CV phase in Oxford characterisation
            caps.append(cap)
            his.append(f)
            cyc.append(int(key.replace("cyc", "")))
            _ = rated_ma
        caps = np.asarray(caps)
        out.append(CellRecord(cell, "Oxford", chemistry, rated_ah, eol, np.asarray(cyc), caps / rated_ah,
                              np.vstack(his), meta={"block": 100, "ambient_c": 40.0}).clean())
    return out


# ---------------------------------------------------------------- in-house CSV
def load_inhouse_csv(path_csv: str, rated_ah: float, chemistry: str, eol: float = 0.80,
                     ambient_c: float = 25.0) -> list[CellRecord]:
    """Columns: cell_id, cycle, step_type (charge|discharge|rest), time_s, voltage_V, current_A, temp_C.
    Optional column capacity_Ah (per-row, max over the discharge step is used)."""
    df = pd.read_csv(path_csv)
    out = []
    for cid, gc in df.groupby("cell_id"):
        caps, his, cyc = [], [], []
        for n, g in gc.groupby("cycle"):
            seg = {}
            for st in ("charge", "discharge"):
                s = g[g["step_type"] == st]
                seg[st] = None if s.empty else {"t": s["time_s"].to_numpy(float), "v": s["voltage_V"].to_numpy(float),
                                                "i": s["current_A"].to_numpy(float),
                                                "T": s["temp_C"].to_numpy(float) if "temp_C" in s else None}
            if seg["discharge"] is None:
                continue
            if "capacity_Ah" in g and g.loc[g["step_type"] == "discharge", "capacity_Ah"].notna().any():
                cap = float(g.loc[g["step_type"] == "discharge", "capacity_Ah"].max())
            else:
                cap = capacity_from_discharge(seg["discharge"])
            caps.append(cap)
            his.append(extract_cycle_features(seg["charge"], seg["discharge"], rated_ah))
            cyc.append(int(n))
        out.append(CellRecord(str(cid), "InHouse", chemistry, rated_ah, eol, np.asarray(cyc),
                              np.asarray(caps) / rated_ah, np.vstack(his), meta={"ambient_c": ambient_c}).clean())
    return out


# ---------------------------------------------------------------- windows
def make_windows(rec: CellRecord, L: int = 10):
    """Inputs x_n = [SoH history, 12 HIs] over cycles n-L+1..n ; target y = SoH_{n+1}.
    Returns X (N, L, 1+N_HI), mask (1+N_HI,), y (N,)."""
    his = rec.his.copy()
    avail = np.isfinite(his).mean(axis=0) > 0.5
    col_med = np.nanmedian(np.where(np.isfinite(his), his, np.nan), axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0)
    his = np.where(np.isfinite(his), his, col_med)
    his[:, ~avail] = 0.0
    feat = np.concatenate([rec.soh[:, None], his], axis=1)
    X, y = [], []
    for n in range(L - 1, len(rec.soh) - 1):
        X.append(feat[n - L + 1:n + 1])
        y.append(rec.soh[n + 1])
    mask = np.concatenate([[True], avail])
    if not X:
        return np.zeros((0, L, 1 + N_HI)), mask, np.zeros(0)
    return np.asarray(X, np.float32), mask, np.asarray(y, np.float32)
