## Goal
Predict whether a river gauge will experience a **rapid surge** soon.

Each row is an anonymized `(station_token, date, horizon)` snapshot derived from USGS NWIS daily discharge (“DV”) time series. Your model must output a probability in \([0,1]\) that, over the next **3 or 7 days**, discharge will **surge** by an unusually large amount relative to the recent baseline.

This is a **binary classification** task on a **panel** (station × day × horizon).

## Files
- `train.csv`: features + `target_rapid_surge_next`
- `test.csv`: features only (no target)
- `sample_submission.csv`: submission format
- `solution.csv`: ground truth for scoring (included for local testing)
- `perfect_submission.csv`: sanity-check perfect predictions
- `score_submission.py`: deterministic scoring script (matches Kaggle evaluation)

## Submission format
Your submission must have exactly:
- `row_id`
- `pred_rapid_surge_next` (probability in \([0,1]\))

## Target definition
Let \(q_t\) be daily discharge (cfs) and \(z_t = \log(q_t + 1)\).

Features are computed using **past-only** information up to day \(t-1\) (e.g., `q_1d` / `logq_1d`).
For each horizon \(h \in \{3,7\}\), define the forward surge magnitude:
\[
\text{surge}_h(t) = \max_{k=0..h-1} z_{t+k} - z_{t-1}
\]

The label is:
- `target_rapid_surge_next = 1` if \(\text{surge}_h(t)\) is at or above a **train-derived 90th percentile** threshold (station-specific when enough history exists; otherwise global fallback)
- else 0

## Slice
The scorer includes a deterministic slice:
- `slice_lowflow = 1` when the station’s **30-day mean discharge** is in a **low-baseline** regime (station-specific, calibrated on training-era history)

## Metric
Lower is better. The final score is:
\[
0.55\cdot \text{LogLoss}_{all} + 0.30\cdot \text{LogLoss}_{slice\_lowflow} + 0.15\cdot (1-\text{AUPRC}_{all})
\]

If the slice is empty, the scorer deterministically uses \(\text{LogLoss}_{slice} \leftarrow \text{LogLoss}_{all}\).

## Split & leakage notes
- Split is **time-based** with a regime shift (later years held out) and deterministic bridge assignment.
- Quantile bin edges for `*_bin` features are computed using training rows only.
- Station-relative thresholds for the target and the slice are computed using training-era history only.

