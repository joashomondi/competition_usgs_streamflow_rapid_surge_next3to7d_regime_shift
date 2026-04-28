## USGS Streamflow Rapid Surge Risk (Next 3–7 Days)

Portfolio-ready, Kaggle-style prediction task built from **USGS NWIS Daily Values** discharge time series.

### What you’re predicting
- **Task**: binary classification
- **Predict**: `pred_rapid_surge_next` (probability in \([0,1]\))
- **Target**: `target_rapid_surge_next` — a **large forward surge** occurs within the next **3 or 7 days**
- **Panelization**: each physical day becomes two examples via `horizon_d ∈ {3,7}`

### Data & evaluation highlights
- **Rows**: train **152,115**, test **62,062** (see `build_meta.json`)
- **Positive rate (train)**: ~**0.117**
- **Split**: time-based regime shift (train ≤ 2020, test ≥ 2023, deterministic bridge years). Details in `dataset_card.md`.
- **Metric**: composite scorer (LogLoss overall + LogLoss on slice + (1 − AUPRC)). Exact formula in `instruction.md`.
- **Slice**: `slice_lowflow` emphasizes “low baseline” periods where surprise surges are harder.

### Repository contents
- **Data**: `train.csv`, `test.csv`, `solution.csv`
- **Submission format**: `sample_submission.csv`, `perfect_submission.csv`
- **Reproducibility**: `build_dataset.py` (rebuilds from cached `.rdb` upstream), `build_meta.json`
- **Scoring**: `score_submission.py`
- **Docs**: `instruction.md`, `golden_workflow.md`, `dataset_card.md`

### Quickstart
Rebuild the dataset:

```bash
python build_dataset.py
```

Score a submission locally:

```bash
python score_submission.py --submission-path sample_submission.csv --solution-path solution.csv
```

Train a baseline model:
- Start with LightGBM/CatBoost
- Treat `station_token` and `horizon_d` as categorical
- Use `*_bin` features as ordered integers
- Calibrate probabilities (LogLoss-heavy metric)

