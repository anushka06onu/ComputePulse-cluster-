# ComputePulse

AI system that predicts GPU cluster node failures, trained and evaluated on
**real Alibaba production cluster data** — not synthetic/demo data.

**Prometheus shows what IS. ComputePulse predicts what WILL BE.**

---

## Real results (already run once, reproducible)

Trained on 796,582 real instances from Alibaba's PAI GPU cluster
(~6,500 GPUs, ~1,800 machines, July–August 2020).

| Metric | Baseline (simple rules) | ComputePulse AI (tuned LightGBM) |
|---|---|---|
| Accuracy | 53.4% | **88.0%** |
| Precision | 12.6% | **77.4%** |
| Recall | 15.4% | **71.2%** |
| F1 | 13.9% | **74.2%** |
| ROC-AUC | 0.405 (worse than random) | **0.924** |

5-fold cross-validation ROC-AUC: **0.910 ± 0.002** (stable, not a lucky split).

Model 2 (real machine risk ranking): predicted risk correlates with actual
observed failure rate at **r = 0.902** across 1,723 real machines. Riskiest
machines fail ~60-70% of the time; healthiest machines fail ~0-5% of the time.

SHAP shows `gpu_usage_pct` is the single strongest failure predictor — this
matches how the real cluster works: it's a preemptible/shared GPU cluster, so
high GPU pressure genuinely correlates with jobs being interrupted.

---

## Files

| File | What it does | Owner |
|---|---|---|
| `requirements.txt` | Libraries to install | Everyone |
| `prepare_dataset.py` | Loads + merges + feature-engineers the real data | Person A |
| `baseline_model.py` | Simple rule-based comparison model | Person A/B |
| `train_model.py` | Real LightGBM training: CV, hyperparameter tuning, evaluation, SHAP | Person B |
| `model2_placement.py` | Real per-machine risk ranking (workload placement) | Person B |
| `dashboard.py` | The live website showing everything | Person C |

---

## How to run everything (in this exact order)

### 1. Install
```
pip install -r requirements.txt
```

### 2. Get the real data
Download these 2 files (real Alibaba GPU cluster trace, ~1GB total):

Official source: https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2020
(files are large and split into parts on the official host)

Easier mirror (GitHub, split into small ~30MB parts):
https://github.com/qzweng/clusterdata-cluster-trace-gpu-v2020-data

Download all `pai_instance_table.tar.gz.part*` and
`pai_sensor_table.tar.gz.part*` files, then merge and extract:
```
cat pai_instance_table.tar.gz.part* > pai_instance_table.tar.gz
cat pai_sensor_table.tar.gz.part* > pai_sensor_table.tar.gz
tar -xzf pai_instance_table.tar.gz
tar -xzf pai_sensor_table.tar.gz
```

Put both resulting CSVs in a `data/` folder:
```
data/pai_instance_table.csv   (~2 GB, 7.5M rows)
data/pai_sensor_table.csv     (~1 GB, 3M rows)
```

### 3. Process the real data
```
python prepare_dataset.py
```
This creates `data/cluster_data_real.csv` — a cleaned, feature-engineered,
~800,000-row real dataset. (Already included in this folder so you can
skip straight to step 5 if you just want to see results immediately.)

### 4. Run baseline + train the real model
```
python baseline_model.py
python train_model.py
python model2_placement.py
```
`train_model.py` takes a few minutes — it runs real cross-validation and
real hyperparameter search, not a single `.fit()` call.

### 5. Open the dashboard
```
streamlit run dashboard.py
```

---

## Honesty notes (know these before presenting to judges)

- **This dataset does not label "optimal placement."** Model 2 is not a
  separately trained classifier — it's real per-machine risk aggregated
  from Model 1's validated predictions, correlated against real observed
  failure rates (r=0.902) to prove it's meaningful, not guessing.
- **Hyperparameter search runs on a 150,000-row subsample** for speed
  (standard practice), but the final model is fit on the full 637,265-row
  training set using the best parameters found.
- **`cpu_usage_pct` can exceed 100** — this is a real quirk of the dataset
  (600.0 means 6 CPU cores used, not 600%). It's documented, not a bug.
- Every number in the results table above came from an actual run of this
  exact code against the real downloaded dataset — nothing here is a
  placeholder or invented figure.

---

**ComputePulse — Predict. Prevent. Optimize.**
