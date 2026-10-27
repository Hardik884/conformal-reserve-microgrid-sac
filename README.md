# Conformal reserve control for residential batteries

Code, data and result files for the paper **"Conformal Quantile Regression for Uncertainty-Aware Soft Actor-Critic Control in Residential Microgrids"**
(Hardik Lohan, Ekaansh Jain, Badrinath N., Vellore Institute of Technology).

**What is here:** a conformal reserve layer that keeps enough energy in a home battery to avoid capacity-limit violations, a quantile CNN-LSTM forecaster calibrated with CQR and adaptive conformal inference (ACI), and Soft Actor-Critic agents, tested on three homes from the CityLearn 2022 dataset.

Folders: `code/` (all scripts), `data/` (CityLearn 2022 files, MIT licence), `out/` (every result file behind the paper's tables and figures), `figures/` (paper figures).
Trained SAC model files (`*_model.zip`) are not included; rerunning `worker.py` recreates them.

**How to run:** go into `code/` and follow the steps below. Every command is run from inside `code/`.

If you use this code, please cite the paper (citation to be added after publication). Licence: MIT.

---

Everything in the paper's Section 6 is produced by the scripts below, in this order.
CPU only; about 6 CPU-hours in total (most of it SAC training).

```
pip install torch stable-baselines3 gymnasium scipy scikit-learn pandas
# data: CityLearn Challenge 2022 files (MIT licence) copied into ../data
#   git clone https://github.com/intelligent-environments-lab/CityLearn
#   cp CityLearn/data/datasets/citylearn_challenge_2022_phase_all/* ../data/

python common.py                     # data-quality filter + donor/balanced/spiky selection -> out/selected.json
python forecast.py Building_8        # cross-fitted quantile CNN-LSTM, CQR, ACI -> out/fc_<home>.npz
python forecast.py Building_2
python forecast.py Building_10
python fc_eval.py Building_8 Building_2 Building_10   # coverage, width, Winkler
python run_baselines.py              # NoBattery, RBC-SC, MPC-P/CQR/ACI, Oracle, +reserve layer -> out/baselines.json
python dial.py                       # risk dial over alpha -> out/dial.json
python worker.py                     # 5 SAC variants x 3 homes x 5 seeds -> out/sac/*.json (resume-safe; run 2 in parallel)
python analyze.py                    # summaries + paired Wilcoxon tests (Holm within metric and family) -> out/stats.json, out/sac_all.csv
python sensitivity.py                # re-pricing under other penalties / export credits -> out/sensitivity.json
python fc_tables.py                  # home statistics + forecast tables -> out/table_homes_fc.json
python make_figures.py               # publication figures -> ../figures/*.png, *.pdf
```

Files
- `common.py` constants fixed before testing (tariff, feed-in 0.05 $/kWh, penalty 1 $/kWh, P_cap = 90th pct of training net load, splits Aug-Mar / Apr / May-Jul).
- `forecast.py` model, pinball loss, split CQR, ACI with delayed feedback, 4-fold blocked cross-fitting.
- `env.py` energy-conserving battery model, two-level tracker, reserve-target action, conformal reserve layer, Gymnasium env.
- `baselines.py` rule-based, MPC (SciPy HiGHS LP), perfect-foresight oracle.
- `run_sac.py` SAC training (Stable-Baselines3) and test evaluation, incl. evaluation-time swaps (ACI inputs, layer on/off).
- `pilot.py`, `tune.py`, `diag.py` design pilots on April validation data (reported as design history in Section 5.4).
  The action design is selected with the env variable ACTION (default `reserve`, the final design).
  The threshold-action pilots and the `tune.py` screen were run with the earlier two-threshold action (ACTION=threshold);
  their JSON files in out/pilot and out/tune do not record this field.

Preliminary study note: the original notebooks used Pecan Street data (not redistributable). To rerun on those homes,
replace `load_building()` in `common.py` with a loader returning the same columns (load, pv, price, weather, calendar).
