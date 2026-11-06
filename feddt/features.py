"""Edge feature extraction (paper Sec. 4.1).

Each cycle is reduced to 12 health indicators (HIs). A cycle is described by a
charge segment and a discharge segment, each a dict of 1-D numpy arrays:
    t [s], v [V], i [A] (charge > 0, discharge < 0 or abs values), T [degC] (optional)
Missing indicators are returned as NaN and later masked.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

HI_NAMES = [
    "cc_charge_ah",      # 1 charge throughput in CC phase
    "cc_duration_s",     # 2 CC-phase duration
    "cv_duration_s",     # 3 CV-phase duration
    "ic_peak_height",    # 4 main dQ/dV peak height (Ah/V)
    "ic_peak_voltage",   # 5 voltage at main dQ/dV peak
    "dv_10s",            # 6 voltage drop over first 10 s of discharge
    "r0_ohm",            # 7 ohmic resistance from current step
    "t_mean",            # 8 mean temperature
    "t_max",             # 9 max temperature
    "t_rise",            # 10 temperature rise over cycle
    "dis_energy_wh",     # 11 discharge energy
    "c_rate_mean",       # 12 mean |C-rate|
]
# indicators whose value is driven by degradation (regressed on SoH during RUL rollout)
DEGRADATION_HI = [0, 1, 2, 3, 4, 5, 6, 10]
# indicators describing usage / environment (held at recent mean during rollout)
USAGE_HI = [7, 8, 9, 11]
N_HI = len(HI_NAMES)


def _trapz(y, x):
    if len(x) < 2:
        return 0.0
    return float(np.trapezoid(y, x)) if hasattr(np, "trapezoid") else float(np.trapz(y, x))


def _cc_cv_split(seg, v_max_tol=0.01):
    """Index where CV starts: first sample within tol of the max charge voltage."""
    v = seg["v"]
    if len(v) == 0:
        return 0
    vmax = np.nanmax(v)
    idx = np.where(v >= vmax - v_max_tol)[0]
    return int(idx[0]) if len(idx) else len(v)


def incremental_capacity(v, q, sigma_mv=5.0, dv=0.002):
    """dQ/dV on a uniform voltage grid with Gaussian smoothing (sigma in mV)."""
    order = np.argsort(v)
    v, q = v[order], q[order]
    v, uniq = np.unique(v, return_index=True)
    q = q[uniq]
    if len(v) < 10:
        return np.array([]), np.array([])
    grid = np.arange(v[0], v[-1], dv)
    if len(grid) < 5:
        return np.array([]), np.array([])
    qg = np.interp(grid, v, q)
    qg = gaussian_filter1d(qg, sigma=max(sigma_mv / (dv * 1000.0), 1.0))
    ic = np.gradient(qg, grid)
    return grid, ic


def extract_cycle_features(charge: dict | None, discharge: dict | None, rated_ah: float) -> np.ndarray:
    f = np.full(N_HI, np.nan)
    if charge is not None and len(charge.get("t", [])) > 5:
        t, v, i = charge["t"], charge["v"], np.abs(charge["i"])
        k = _cc_cv_split(charge)
        f[0] = _trapz(i[:k], t[:k]) / 3600.0
        f[1] = float(t[k - 1] - t[0]) if k > 1 else np.nan
        f[2] = float(t[-1] - t[k]) if k < len(t) - 1 else 0.0
        q = np.concatenate([[0.0], np.cumsum(0.5 * (i[1:k] + i[:k - 1]) * np.diff(t[:k]))]) / 3600.0 if k > 1 else np.array([])
        if len(q) > 10:
            grid, ic = incremental_capacity(v[:k], q)
            if len(ic):
                j = int(np.nanargmax(ic))
                f[3], f[4] = float(ic[j]), float(grid[j])
        if "T" in charge and charge["T"] is not None and len(charge["T"]):
            Tc = charge["T"]
        else:
            Tc = None
    else:
        Tc = None
    if discharge is not None and len(discharge.get("t", [])) > 5:
        t, v, i = discharge["t"], discharge["v"], np.abs(discharge["i"])
        t0 = t[0]
        m10 = t <= t0 + 10.0
        if m10.sum() > 1:
            f[5] = float(v[0] - v[m10][-1])
        di = np.diff(i)
        step = np.argmax(np.abs(di)) if len(di) else 0
        if len(di) and abs(di[step]) > 0.05 * rated_ah:
            f[6] = float(abs((v[step + 1] - v[step]) / di[step]))
        f[10] = _trapz(v * i, t) / 3600.0
        f[11] = float(np.mean(i) / rated_ah)
        Td = discharge.get("T")
        Ts = [x for x in (Tc, Td) if x is not None and len(x)]
        if Ts:
            T = np.concatenate(Ts)
            f[7], f[8] = float(np.nanmean(T)), float(np.nanmax(T))
            f[9] = float(np.nanmax(T) - T[0])
    return f


def capacity_from_discharge(discharge: dict) -> float:
    t, i = discharge["t"], np.abs(discharge["i"])
    return _trapz(i, t) / 3600.0
