## USGS Streamflow Rapid Surge Risk (Next 3–7 Days)

Kaggle-style competition package built from **USGS NWIS Daily Values** discharge time series.

- **Task**: binary classification — predict `pred_rapid_surge_next`
- **Target**: whether discharge will surge sharply within the next **3 or 7 days** (panelized by `horizon_d`)
- **Split**: time-based regime shift (see `dataset_card.md`)
- **Metric**: composite LogLoss + slice LogLoss + (1 - AUPRC) (see `instruction.md`)

### What’s in this repo
- `train.csv`, `test.csv`
- `sample_submission.csv`, `perfect_submission.csv`
- `solution.csv` (for local validation)
- `build_dataset.py` (rebuilds dataset from cached `.rdb` upstream)
- `score_submission.py` (deterministic scorer)
- `instruction.md`, `golden_workflow.md`, `dataset_card.md`

### Quickstart
Rebuild:

```bash
python build_dataset.py
```

Score a submission:

```bash
python score_submission.py --submission-path sample_submission.csv --solution-path solution.csv
```

