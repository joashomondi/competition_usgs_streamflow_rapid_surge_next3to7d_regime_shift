## 1) Restate the problem in “alerting” terms
You are building a daily early-warning model for **sharp streamflow surges** at many gauges:
- Input at day \(t\): past-only hydrologic state up to \(t-1\) (levels, volatility, flashiness, calendar)
- Output: \(P(\)a large surge happens within the next \(h\in\{3,7\}\) days\()\)

This is intentionally hard because surges are often driven by localized, exogenous precipitation you don’t observe directly.

## 2) What makes this task non-trivial
- **Station heterogeneity**: baselines differ by orders of magnitude; station-relative thresholds reduce trivial “big rivers are always positive” shortcuts.
- **Regime shift**: the test set is later years; calibration and generalization matter.
- **Horizon mismatch**: 3d vs 7d positives differ; one model might be miscalibrated across horizons.
- **Slice pressure**: `slice_lowflow` forces performance in “calm/low baseline” periods where sudden surges are the most surprising.

## 3) Validation that matches the data generation
- Use **blocked time CV** within train years (e.g., validate on 2019, then 2020).
- Always report:
  - overall LogLoss and AUPRC
  - slice LogLoss on `slice_lowflow`
  - per-horizon breakdown (`horizon_d=3` vs `7`)

Avoid random CV; it leaks station-season patterns across time.

## 4) Baseline model that should be hard to beat
Start with a strong tabular classifier:
- **Model**: LightGBM / CatBoost
- **Categoricals**: `station_token`, `horizon_d`, `event_era` (and optionally `is_weekend`)
- **Numerics**: treat `*_bin` columns as ordered integers; include `missing_count`
- **Regularization**: prevent station memorization
  - increase `min_data_in_leaf`, add L1/L2, lower `num_leaves`

## 5) Interpreting the provided features (useful signal)
Surge risk often rises when:
- baseline is low but volatility is creeping up (`slice_lowflow` + higher `abs_dlogq_mean7_bin`)
- recent flashiness is elevated (`flashiness_7d_bin`)
- levels are high relative to recent mean (captured indirectly by level + range bins)

## 6) Calibration (high leverage for this metric)
The score is LogLoss-heavy; calibration matters.
- Fit a calibration layer on a **time-based validation** set.
- Consider calibrating **per horizon** if base rates differ meaningfully.

## 7) Common failure modes
- **Overfit to station_token**: offline looks great; test-year performance collapses.
- **Seasonality shortcut**: the model predicts based on month bins only; good AUPRC, poor LogLoss.
- **Horizon leakage via thresholding**: if you rebuild, never compute bin edges/thresholds using test years.

## 8) Final checklist
- Submission has exactly `row_id`, `pred_rapid_surge_next`
- Predictions clipped to \([0,1]\), no NaNs
- Validate locally with `score_submission.py` against `solution.csv`

