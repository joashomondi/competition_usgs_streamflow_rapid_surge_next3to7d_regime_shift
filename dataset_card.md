## Overview
This dataset is built from USGS NWIS daily discharge time series and targets **rapid surge risk**: whether discharge will jump sharply within the next **3 or 7 days** relative to the last observed day. The evaluation uses a **chronological regime-shift** split to mimic deployment on future years.

## Source
USGS National Water Information System (NWIS), Daily Values (DV) service.

Primary service documentation and endpoint:
- Docs: `https://waterservices.usgs.gov/`
- DV endpoint: `https://waterservices.usgs.gov/nwis/dv/`

The build in this workspace reads locally cached NWIS DV `.rdb` responses from:
- `competition_usgs_streamflow_floodwatch_3day_regime_shift/_cache/*.rdb`

## License
Public-Domain-US-Gov

## Features
Each row corresponds to a (station, day, horizon) panel point. Columns include:
- **Identifiers**
  - `row_id`: unique row key
  - `station_token`: anonymized gauge identifier (hash of USGS site number)
- **Panel context**
  - `horizon_d`: prediction horizon in days (3 or 7)
  - `event_era`: coarse time regime bucket (0..3)
- **Binned numeric features** (`*_bin`, 12 quantile bins learned on training-only)
  - level & baseline: `q_1d_bin`, `q_mean_7d_bin`, `q_mean_30d_bin`
  - variability: `q_std_7d_bin`, `q_std_30d_bin`, `flashiness_7d_bin`, `abs_dlogq_mean7_bin`
  - range: `q_min_30d_bin`, `q_max_30d_bin`
  - log dynamics: `logq_1d_bin`, `dlogq_1d_bin`
  - missingness: `q_missing_30d_bin`
  - calendar: `month_bin`, `dayofyear_bin`
  - Missing values are encoded as `-1`
- **Other**
  - `is_weekend`: weekend flag (0/1)
  - `missing_count`: number of missing raw numeric features before binning
- **Slice flag**
  - `slice_lowflow`: low-baseline regime indicator (station-relative, train-era calibrated)

## Splitting & Leakage
- **Split type**: time-based regime shift
  - Train: years ≤ 2020
  - Bridge: years 2021–2022, deterministic hashed assignment (with “flashier” gauges pushed to test more often)
  - Test: years ≥ 2023
- **Leakage controls**
  - Features are strictly past-only (shifted and rolling windows on prior days).
  - Target thresholds are computed using training rows only (station-specific when enough data exists).
  - Quantile bin edges are computed using training rows only.

