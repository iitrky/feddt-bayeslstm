"""Fleet digital twin: RUL rollout, metrics and decision rules (paper Sec. 4.6, 5.5)."""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy.stats import norm as gauss

from .features import DEGRADATION_HI, USAGE_HI


# ------------------------------------------------------------------ metrics
def soh_metrics(y, mu, sigma, q=1.96):
    """SoH in fraction; RMSE/MAE/MPIW reported in percentage points."""
    y, mu, sigma = map(np.asarray, (y, mu, sigma))
    err = y - mu
    lo, hi = mu - q * sigma, mu + q * sigma
    z = err / np.maximum(sigma, 1e-9)
    crps = sigma * (z * (2 * gauss.cdf(z) - 1) + 2 * gauss.pdf(z) - 1 / math.sqrt(math.pi))
    nll = 0.5 * (z ** 2 + np.log(2 * math.pi * np.maximum(sigma, 1e-9) ** 2))
    return {
        "RMSE_%": 100 * float(np.sqrt(np.mean(err ** 2))),
        "MAE_%": 100 * float(np.mean(np.abs(err))),
        "PICP": float(np.mean((y >= lo) & (y <= hi))),
        "MPIW_%": 100 * float(np.mean(hi - lo)),
        "NLL": float(np.mean(nll)),
        "CRPS_%": 100 * float(np.mean(crps)),
    }


def rul_metrics(rows, alpha=0.2):
    """rows: list of dicts with true, pred, lo, hi (cycles)."""
    if not rows:
        return {}
    t = np.array([r["true"] for r in rows], float)
    p = np.array([r["pred"] for r in rows], float)
    lo = np.array([r["lo"] for r in rows], float)
    hi = np.array([r["hi"] for r in rows], float)
    return {
        "RUL_RMSE_cyc": float(np.sqrt(np.mean((t - p) ** 2))),
        "RUL_MAE_cyc": float(np.mean(np.abs(t - p))),
        "RUL_PICP": float(np.mean((t >= lo) & (t <= hi))),
        "RUL_MPIW_cyc": float(np.mean(hi - lo)),
        "alpha_lambda_acc": float(np.mean(np.abs(t - p) <= alpha * np.maximum(t, 1))),
    }


# ------------------------------------------------------------------ RUL rollout
def eol_index(soh, eol):
    idx = np.where(soh <= eol)[0]
    return int(idx[0]) if len(idx) else None


def rollout_rul(model, rec, start, L, norm_fn, eol, M=200, h_max=3000, block=1, seed=0, phys=None):
    """Monte Carlo RUL distribution from cycle index `start` (history up to start, inclusive).

    Degradation-linked HIs are predicted from SoH by per-HI linear fits on the cell's own
    history (local, private); usage HIs are held at their mean over the last 20 cycles.
    One dropout mask per trajectory (fixed network sample) + aleatoric noise per step.
    With `phys` (federated physics prior), each step is prior(SoH, throughput, T, C-rate)
    plus the network's residual, which keeps long rollouts on a physically plausible path."""
    from . import physics
    torch.manual_seed(seed)
    soh_hist, his = rec.soh[:start + 1], rec.his[:start + 1].copy()
    col_med = np.nanmedian(np.where(np.isfinite(his), his, np.nan), axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0)
    his = np.where(np.isfinite(his), his, col_med)
    coefs = {}
    for j in DEGRADATION_HI:
        A = np.vstack([soh_hist, np.ones_like(soh_hist)]).T
        coefs[j] = np.linalg.lstsq(A, his[:, j], rcond=None)[0]
    usage = his[-20:, USAGE_HI].mean(0)
    feat = np.concatenate([soh_hist[:, None], his], 1)[-L:]
    W = torch.tensor(np.repeat(feat[None], M, 0), dtype=torch.float32)
    if phys is not None:
        T_s, c_s, ah_s, _ = physics.cell_series(rec)
        T_use, c_use = float(np.mean(T_s[max(0, start - 19):start + 1])), float(np.mean(c_s[max(0, start - 19):start + 1]))
        ah = np.full(M, ah_s[start])
        soh0 = float(np.median(rec.soh[:5]))
    model.eval()
    masks = model.lstm.sample_masks(M, W.shape[-1], W.device) if model.lstm.p > 0 else None
    rul = np.full(M, np.nan)
    alive = np.ones(M, bool)
    s_prev = np.full(M, soh_hist[-1])
    with torch.no_grad():
        for h in range(1, h_max + 1):
            mu, logv, _ = model(norm_fn(W), mc=False, masks=masks)
            s = (mu + torch.randn(M) * (0.5 * logv).exp()).numpy()
            if phys is not None:
                dah = s_prev * rec.rated_ah * block
                s = physics.prior_next(s_prev, ah, dah, T_use, c_use, phys, soh0) + s
                ah = ah + dah
            hit = alive & (s <= eol)
            rul[hit] = h * block
            alive &= ~hit
            if not alive.any():
                break
            new = np.zeros((M, W.shape[-1]), np.float32)
            new[:, 0] = s
            for j in DEGRADATION_HI:
                new[:, 1 + j] = coefs[j][0] * s + coefs[j][1]
            new[:, [1 + j for j in USAGE_HI]] = usage
            W = torch.cat([W[:, 1:], torch.tensor(new)[:, None]], 1)
            s_prev = s
    rul[np.isnan(rul)] = h_max * block
    return rul


# ------------------------------------------------------------------ Algorithm 3: calibrated risk shift planner
def shift_planner(vehicles, horizon_shifts=3, p_crit=0.10, bay_slots=2, derate_soh=0.82,
                  second_life_soh=0.70, epi_quantile=0.95):
    """Calibrated Risk Shift Planner (paper Algorithm 3).

    vehicles: list of dicts with keys
        id, chemistry, rul_samples (cycles, conformally calibrated), soh_mu, soh_sigma (calibrated),
        epi_var, eol (threshold), cycles_per_shift, drift_flag (bool)
    Returns (plan, schedule): per-vehicle actions and the inspection queue limited to `bay_slots`.
    """
    by_chem = {}
    for v in vehicles:
        by_chem.setdefault(v["chemistry"], []).append(v["epi_var"])
    epi_thr = {c: float(np.quantile(x, epi_quantile)) for c, x in by_chem.items()}
    plan, queue = [], []
    for v in vehicles:
        rul_shifts = np.asarray(v["rul_samples"], float) / max(v["cycles_per_shift"], 1e-9)
        p_fail = float(np.mean(rul_shifts <= horizon_shifts))
        soh_p5 = v["soh_mu"] - 1.645 * v["soh_sigma"]
        actions = []
        if v.get("drift_flag"):
            actions.append("monitor_drift_reset_calibration")
        if v["soh_mu"] <= v["eol"]:
            route = ("second_life" if soh_p5 >= second_life_soh and v["epi_var"] <= epi_thr[v["chemistry"]]
                     else f"recycle_{v['chemistry']}")
            actions.append(f"retire_pack_route_{route}")
        elif p_fail > p_crit:
            queue.append((p_fail, v["id"]))
            actions.append("request_inspection")
        if soh_p5 < derate_soh or v["epi_var"] > epi_thr[v["chemistry"]]:
            actions.append("swap_or_slow_charge_cap_c_rate_soc_20_80")
        else:
            actions.append("fast_charge_allowed")
        plan.append({"id": v["id"], "p_fail_horizon": p_fail, "soh_p5": float(soh_p5), "actions": actions})
    queue.sort(reverse=True)
    schedule = [vid for _, vid in queue[:bay_slots]]
    deferred = [vid for _, vid in queue[bay_slots:]]
    for p in plan:
        if p["id"] in deferred:
            p["actions"] = [a if a != "request_inspection" else "inspection_deferred_derate" for a in p["actions"]]
    return plan, schedule


def decision_quality(rows, horizon_cycles, p_crit=0.10):
    """Missed-failure and false-withdrawal rates of the inspection rule against ground truth.
    rows: dicts with true (RUL cycles) and p_le_h (predicted Pr(RUL <= horizon_cycles))."""
    if not rows:
        return {}
    truth = np.array([r["true"] <= horizon_cycles for r in rows])
    flag = np.array([r["p_le_h"] > p_crit for r in rows])
    return {"missed_failure_rate": float(np.mean(~flag[truth])) if truth.any() else float("nan"),
            "false_withdrawal_rate": float(np.mean(flag[~truth])) if (~truth).any() else float("nan")}


# ------------------------------------------------------------------ simple rule (kept for compatibility)
def decide(rul_samples, soh_mu, soh_sigma, epi_var, h_service=200, p_crit=0.1,
           derate_soh=0.82, epi_thresh=None):
    """Risk-aware maintenance and charging rules of the fleet twin."""
    p_fail = float(np.mean(rul_samples <= h_service))
    p5 = soh_mu - 1.645 * soh_sigma
    actions = []
    if p_fail > p_crit:
        actions.append("schedule_inspection")
    if p5 < derate_soh or (epi_thresh is not None and epi_var > epi_thresh):
        actions.append("derate_charge_rate_and_narrow_soc_window_20_80")
    return {"p_rul_le_service": p_fail, "soh_p5": float(p5), "actions": actions or ["none"]}
