import sys, types, math
import numpy as np
import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# stub torch so twin.py imports (only numpy parts tested)
_t = types.ModuleType("torch"); _t.Tensor = type("Tensor", (), {}); sys.modules.setdefault("torch", _t)
from feddt.features import extract_cycle_features, capacity_from_discharge, HI_NAMES, N_HI
from feddt.data import CellRecord, make_windows
from feddt.twin import soh_metrics, rul_metrics, eol_index

rng = np.random.default_rng(0)
def synth_cycle(cap, T0=25.0):
    # CC-CV charge
    t1 = np.arange(0, 3600*cap/1.0, 5.0)
    v1 = 3.6 + 0.6*(t1/t1[-1])**0.6 + 0.02*np.sin(8*t1/t1[-1])
    i1 = np.full_like(t1, 1.0)
    tcv = np.arange(t1[-1]+5, t1[-1]+1200, 5.0)
    vcv = np.full_like(tcv, 4.2); icv = np.exp(-(tcv-tcv[0])/300)
    ch = {"t": np.r_[t1, tcv], "v": np.r_[v1, vcv], "i": np.r_[i1, icv], "T": T0+np.r_[t1, tcv]/3600}
    td = np.arange(0, 3600*cap/2.0, 2.0)
    idc = np.full_like(td, -2.0); idc[:3] = 0.0
    R = 0.05 + 0.1*(2.0-cap)
    vd = 4.1 - 1.3*(td/td[-1])**2 - R*np.abs(idc) - 0.02*(1-np.exp(-td/5))*(np.abs(idc)>0)
    dc = {"t": td, "v": vd, "i": idc, "T": T0+3+5*td/td[-1]}
    return ch, dc

caps = 2.0*(1 - 0.0018*np.arange(200)) + rng.normal(0, 0.004, 200)
his = []
for c in caps:
    ch, dc = synth_cycle(c)
    his.append(extract_cycle_features(ch, dc, 2.0))
his = np.vstack(his)
print("HI finite fraction per feature:")
for n, f in zip(HI_NAMES, np.isfinite(his).mean(0)): print(f"  {n:16s} {f:.2f}")
print("first-cycle HIs:", np.round(his[0], 4))
print("capacity from discharge integration:", round(capacity_from_discharge(synth_cycle(2.0)[1]), 4), "(expect ~2.0)")
print("R0 first/last:", round(his[0,6],4), round(his[-1,6],4), " dv_10s first/last:", round(his[0,5],4), round(his[-1,5],4))
rec = CellRecord("syn", "SYN", "NMC", 2.0, 0.8, np.arange(200), caps/2.0, his).clean()
X, mask, y = make_windows(rec, 10)
print("windows", X.shape, "mask", mask.astype(int), "y range", y.min().round(3), y.max().round(3))
assert X.shape == (190, 10, 1+N_HI) and np.isfinite(X).all()
assert np.allclose(X[0, -1, 0], rec.soh[9]) and np.isclose(y[0], rec.soh[10])
m = soh_metrics(y, y+0.01, np.full_like(y, 0.01/1.96*1.5))
print("metrics (bias 1%):", {k: round(v,3) for k,v in m.items()})
assert abs(m["RMSE_%"]-1.0) < 1e-4 and m["PICP"] == 1.0
print("eol idx", eol_index(rec.soh, 0.8))
print(rul_metrics([{"true":100,"pred":90,"lo":70,"hi":120},{"true":50,"pred":70,"lo":60,"hi":80}]))
# uplink accounting vs paper
def uplink_bytes(P, rho, bits, n_tensor, rff_dim, drift=True):
    if rho >= 1.0 and bits >= 32: payload = P*4
    else:
        k = int(math.ceil(rho*P)); idx_bits = 0 if rho >= 1.0 else int(math.ceil(math.log2(P)))
        payload = math.ceil(k*(bits+idx_bits)/8) + 4*n_tensor
    return payload + (4*(rff_dim+1) if drift else 0)
assert uplink_bytes(55762,0.1,8,14,128) == 17303 and uplink_bytes(55762,1,32,14,128,False) == 223048
print("ALL NON-TORCH CHECKS PASSED")

# ---- Algorithm 3: calibrated risk shift planner
from feddt.twin import shift_planner, decision_quality
rng2 = np.random.default_rng(1)
veh = [
    {"id": "LHD-1", "chemistry": "LFP", "rul_samples": rng2.normal(400, 40, 200), "soh_mu": 0.90, "soh_sigma": 0.01,
     "epi_var": 1e-5, "eol": 0.8, "cycles_per_shift": 2, "drift_flag": False},
    {"id": "LHD-2", "chemistry": "LFP", "rul_samples": rng2.normal(5, 2, 200), "soh_mu": 0.82, "soh_sigma": 0.01,
     "epi_var": 2e-5, "eol": 0.8, "cycles_per_shift": 2, "drift_flag": True},
    {"id": "TRK-1", "chemistry": "NMC", "rul_samples": rng2.normal(0, 1, 200), "soh_mu": 0.78, "soh_sigma": 0.01,
     "epi_var": 1e-5, "eol": 0.8, "cycles_per_shift": 2, "drift_flag": False},
]
plan, sched = shift_planner(veh, horizon_shifts=3, bay_slots=1)
acts = {p["id"]: p["actions"] for p in plan}
print("shift plan:", acts, "schedule:", sched)
assert sched == ["LHD-2"] and "request_inspection" in acts["LHD-2"] and "monitor_drift_reset_calibration" in acts["LHD-2"]
assert any(a.startswith("retire_pack_route_second_life") for a in acts["TRK-1"])
assert "fast_charge_allowed" in acts["LHD-1"]
dq = decision_quality([{"true": 50, "p_le_h": 0.9}, {"true": 500, "p_le_h": 0.0}, {"true": 80, "p_le_h": 0.0}], 100)
assert abs(dq["missed_failure_rate"] - 0.5) < 1e-9 and dq["false_withdrawal_rate"] == 0.0
print("ALGORITHM 3 (SHIFT PLANNER) CHECKS PASSED")

# ---- physics-guided prior: recover Ea from cells at different temperatures
from feddt import physics
from feddt.features import N_HI as _NHI
def synth_phys_cell(T_c, B=np.exp(7.5), Ea=31700.0, z=0.55, n=300, c=1.0, seed=0):
    r_ = np.random.default_rng(seed)
    soh = np.empty(n); soh[0] = 1.0; ah = 0.0; Tk = T_c + 273.15
    k = B * np.exp(-(Ea - 370.3 * c) / (8.314 * Tk))
    for i in range(1, n):
        ah += soh[i - 1] * 2.0
        soh[i] = 1.0 * (1 - k * ah ** z)
    his = np.full((n, _NHI), np.nan); his[:, 7] = T_c; his[:, 11] = c
    return CellRecord(f"T{T_c}", "SYN", "NMC", 2.0, 0.8, np.arange(n), soh + r_.normal(0, 2e-4, n), his)
cells = [synth_phys_cell(T, seed=i) for i, T in enumerate([25, 25, 40, 45, 45])]
fits = [physics.fit_local(cc) for cc in cells]
agg = physics.aggregate(fits)
print("physics prior: Ea_hat =", round(agg["Ea"]), "J/mol (true 31700), z =", round(agg["z"], 3), "(true 0.55)")
assert agg["Ea_estimated"] and abs(agg["Ea"] - 31700) / 31700 < 0.15 and abs(agg["z"] - 0.55) < 0.08
pr = physics.prior_series(cells[3], agg, 10)
err = np.abs(pr - cells[3].soh[10:])
print("one-step prior MAE on hot cell:", float(err.mean()))
assert err.mean() < 2e-3
print("PHYSICS PRIOR CHECKS PASSED")

# ---- DW-FCC: weighted conformal improves coverage under shift
from feddt.conformal import rff, tilt_eta, weighted_conformal_quantile
rng3 = np.random.default_rng(3)
xc = rng3.normal(0, 1, (2000, 1)); xt = rng3.normal(1.5, 1, (2000, 1))
sd = lambda x: 0.1 + 0.15 * np.abs(x[:, 0])
yc = xc[:, 0] + rng3.normal(0, 1, 2000) * sd(xc); yt = xt[:, 0] + rng3.normal(0, 1, 2000) * sd(xt)
sc = np.abs(yc - xc[:, 0]) / 0.2
Pc, Pt = rff(xc, 128, 1.0, 7), rff(xt, 128, 1.0, 7)
d = Pt.mean(0) - Pc.mean(0); proj = Pc @ d; tgt = Pt.mean(0) @ d
eta = tilt_eta(proj, tgt); w = np.exp(eta * (proj - proj.max()))
qw = weighted_conformal_quantile(sc, w, float(np.exp(eta * (tgt - proj.max()))))
qu = np.quantile(sc, 0.95)
cu = np.mean(np.abs(yt - xt[:, 0]) <= qu * 0.2); cw = np.mean(np.abs(yt - xt[:, 0]) <= qw * 0.2)
print(f"DW-FCC coverage under shift: split {cu:.3f}, weighted {cw:.3f}")
assert cw > cu + 0.04
print("DW-FCC CHECKS PASSED")
