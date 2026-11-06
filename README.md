# FedDT-BayesLSTM

Reference implementation of **"Federated Bayesian Digital Twin for Battery Health Prognosis in
Electric Mining Fleets"** (S. Marisargunam and G. Arul Dalton, submitted to *Mining, Metallurgy &
Exploration*).

FedDT-BayesLSTM estimates battery state of health (SoH) and predicts remaining useful life (RUL)
with calibrated confidence bounds across a fleet of battery electric mining vehicles, without
moving raw voltage, current or temperature data off the vehicles.

| Component | Paper | Code |
|---|---|---|
| Edge feature extraction (12 health indicators) | Sect. 4.1 | `feddt/features.py` |
| Physics-guided prior: federated Arrhenius throughput fade law, residual learning | Sect. 4.2, Eq. (3), Fig. 2 | `feddt/physics.py` |
| Variational Bayesian LSTM, heteroscedastic head | Sect. 4.2, Eqs. (4)–(5) | `feddt/model.py` |
| Drift-aware weights from volume, NLL reliability, RFF-MMD drift | Sect. 4.3, Eqs. (6)–(7) | `feddt/fed.py` (`Federation.client_update`, `aggregate`) |
| Chemistry clusters, cold start, top-k + 8-bit compression | Sect. 4.4, Eq. (8) | `Federation.heads`, `cold_start_key`, `compress`, `uplink_bytes` |
| Algorithm 1: federated training | Sect. 4.5 | `Federation.run` |
| Algorithm 2: drift-weighted federated conformal calibration (DW-FCC) | Sect. 4.6, Eqs. (9)–(10), Fig. 3 | `feddt/conformal.py`, `Federation.conformal_quantile` |
| Physics-guided MC RUL rollout | Sect. 4.6 | `feddt/twin.py:rollout_rul` |
| Algorithm 3: calibrated risk shift planner | Sect. 4.7, Fig. 4 | `feddt/twin.py:shift_planner`, `decision_quality` |
| Synthetic identifiability and coverage checks (Table 10) | Sect. 6.4 | `tests/test_nontorch.py`, `tests/synthetic_shift_check.py` |
| Figs. 1–4 block diagrams | Sects. 3.2, 4.2, 4.6, 4.7 | `figures/make_figures.py` |

## Install

```bash
git clone https://github.com/iitrky/feddt-bayeslstm.git
cd feddt-bayeslstm
pip install -r requirements.txt
```

Python 3.10 or later; PyTorch 2.x (CPU is enough).

## Data

Download the public datasets from their providers and list the paths in `data_paths.json`
(copy `data_paths.example.json`):

| Dataset | Files expected |
|---|---|
| NASA PCoE battery set | `B0005.mat`, `B0006.mat`, `B0007.mat`, `B0018.mat` |
| CALCE CS2 | one folder per cell (`CS2_35/`, …) with the original `.xlsx` files |
| Oxford Battery Degradation Dataset 1 | `Oxford_Battery_Degradation_Dataset_1.mat` (re-save with `-v7` if SciPy cannot read it) |
| In-house 18650 cycling | one CSV: `cell_id, cycle, step_type (charge/discharge/rest), time_s, voltage_V, current_A, temp_C[, capacity_Ah]` |

## Reproduce the paper

```bash
# quick check (minutes)
python run_experiment.py --config data_paths.json --seeds 0 --rounds 10 --T 10 --max_folds 1 --out results/quick

# Tables 4-8, scenario S1 (5 seeds, all leave-one-cell-out folds)
python run_experiment.py --config data_paths.json --scenario S1 --out results/
# Table 9: drift scenarios and ablation
python run_experiment.py --config data_paths.json --scenario S2 --methods feddt fedavg_mc \
    --ablations no_physics no_drift no_reliability shared_head no_dwfcc no_conformal --out results/
python run_experiment.py --config data_paths.json --scenario S3 --late_chem <CHEMISTRY> \
    --methods feddt fedavg_mc --ablations no_drift shared_head --out results/
# Table 10: synthetic covariate-shift check of DW-FCC (seconds, no data needed)
python tests/synthetic_shift_check.py
# print every table, including Algorithm 3 decision quality
python make_tables.py --results results/ --target_rmse 2.0
# figures
python figures/make_figures.py
```

## Tests

```bash
python tests/test_nontorch.py    # features, windows, metrics, uplink bytes, physics prior, DW-FCC, shift planner (no PyTorch needed)
```

## Repository layout

```
feddt/               package: features, data loaders, physics prior, model, federation, conformal calibration, digital twin
run_experiment.py    experiments of Sect. 5 -> CSV files
make_tables.py       CSV files -> tables of Sect. 6
figures/             Figs. 1-4 (script, EPS, TIFF, PNG)
tests/               unit tests that run without PyTorch
```

## Citation

See `CITATION.cff`. Please cite the paper if you use this code.

## Licence

MIT, see `LICENSE`.
