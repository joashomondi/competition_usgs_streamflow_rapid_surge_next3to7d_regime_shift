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

### Why this is interesting (and non-trivial)
- **Exogenous drivers**: surges are often caused by localized precipitation you don’t observe directly—so you’re learning *precursors* (flashiness, rising volatility), not the storm itself.
- **Station heterogeneity**: “big river” vs “small creek” is not the point; thresholds are station-relative, so the model must generalize across scales.
- **Regime shift**: later years are held out; seasonality shortcuts and station memorization get punished.
- **Multi-horizon calibration**: 3-day vs 7-day risk has different base rates and noise—calibration by `horizon_d` often matters.

### Target intuition (plain English)
The label answers:
> “Starting tomorrow, will this gauge experience a surprisingly large jump in flow at least once within the next \(h\) days?”

The “surge magnitude” is measured on a log scale (\(\log(q+1)\)) and compared to a **train-derived 90th percentile** threshold (station-specific when enough history exists; global fallback otherwise).

### Common pitfalls
- **Random CV leakage**: random splits leak station-season patterns; use blocked time validation.
- **Station overfit**: `station_token` can dominate; regularize and sanity-check by ablation.
- **Miscalibration**: ranking can look fine, but LogLoss-heavy scoring punishes overconfident predictions.

### Source
USGS NWIS Water Services:
- `https://waterservices.usgs.gov/`
- Daily Values endpoint: `https://waterservices.usgs.gov/nwis/dv/`

