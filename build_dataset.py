from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


COMP_DIR = Path(__file__).resolve().parent
ROOT = COMP_DIR.parent

UPSTREAM_CACHE_DIR = ROOT / "competition_usgs_streamflow_floodwatch_3day_regime_shift" / "_cache"

ID_COLUMN = "row_id"
STATION_COLUMN = "station_token"
HORIZON_COLUMN = "horizon_d"  # 3 or 7 days
ERA_COLUMN = "event_era"

TARGET_COLUMN = "target_rapid_surge_next"
PRED_COLUMN = "pred_rapid_surge_next"
SLICE_COLUMN = "slice_lowflow"

N_BINS = 12
MIN_TRAIN_ROWS = 12_000
EPS = 1e-9


@dataclass(frozen=True)
class Config:
    parameter_cd: str = "00060"  # discharge

    horizons_days: Tuple[int, ...] = (3, 7)

    # Past-only feature windows.
    lb_short: int = 7
    lb_long: int = 30

    # Deterministic split.
    train_max_year: int = 2020
    test_min_year: int = 2023
    bridge_years: Tuple[int, ...] = (2021, 2022)
    bridge_test_rate_flashy: int = 75
    bridge_test_rate_stable: int = 30

    # Sampling (stable) to control file size.
    keep_percent_train: int = 75
    keep_percent_test: int = 95

    max_sites: int = 80

    # Target: "surge magnitude" threshold (train-derived).
    surge_quantile: float = 0.90


def _hash_percent(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16) % 100


def _station_token(site_no: str) -> str:
    return hashlib.sha256(site_no.encode("utf-8")).hexdigest()[:12]


def _row_id(station_token: str, date: pd.Timestamp, h: int) -> int:
    s = f"surge_{h}d_{station_token}_{date:%Y-%m-%d}"
    hsh = hashlib.md5(s.encode("utf-8")).hexdigest()[:15]
    return int(hsh, 16)


def _read_dv_rdb(path: Path, parameter_cd: str) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    if len(lines) < 3:
        return pd.DataFrame(columns=["date", "q_cfs", "site_no"])

    header = lines[0].split("\t")
    rows = [ln.split("\t") for ln in lines[2:]]  # skip types row
    df = pd.DataFrame(rows, columns=header)
    if "datetime" not in df.columns or "site_no" not in df.columns:
        return pd.DataFrame(columns=["date", "q_cfs", "site_no"])

    matches = [c for c in df.columns if f"_{parameter_cd}_" in c]
    if not matches:
        return pd.DataFrame(columns=["date", "q_cfs", "site_no"])
    value_col = matches[0]

    out = pd.DataFrame(
        {
            "site_no": df["site_no"].astype(str),
            "date": pd.to_datetime(df["datetime"], errors="coerce"),
            "q_cfs": pd.to_numeric(df[value_col], errors="coerce"),
        }
    )
    out = out[out["date"].notna()].copy()
    out = out.drop_duplicates(subset=["site_no", "date"]).sort_values("date").reset_index(drop=True)
    out["date"] = out["date"].dt.floor("D")
    return out


def _quantile_edges(train_values: np.ndarray, n_bins: int) -> np.ndarray:
    x = train_values.astype(float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.array([], dtype=float)
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    edges = np.quantile(x, qs).astype(float)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-12
    return edges


def _bin_with_edges(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    out = np.full(values.shape[0], -1, dtype=int)
    mask = np.isfinite(values)
    if edges.size == 0:
        out[mask] = 0
        return out
    out[mask] = np.digitize(values[mask], edges, right=False).astype(int)
    return out


def _make_bins(df_all: pd.DataFrame, df_train: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, List[float]]]:
    edges_map: Dict[str, List[float]] = {}
    for c in cols:
        edges = _quantile_edges(df_train[c].to_numpy(dtype=float), N_BINS)
        edges_map[c] = edges.tolist()
        df_all[f"{c}_bin"] = _bin_with_edges(df_all[c].to_numpy(dtype=float), edges)
    return df_all, edges_map


def _event_era(year: int) -> int:
    # Coarse eras for hydrologic regimes (drought/wet years are region-dependent).
    if year <= 2016:
        return 0
    if year <= 2019:
        return 1
    if year <= 2022:
        return 2
    return 3


def _engineer_site(site_no: str, dv: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if dv.empty:
        return pd.DataFrame()
    sub = dv[dv["site_no"].astype(str) == str(site_no)].copy()
    if sub.empty:
        return pd.DataFrame()

    sub = sub.sort_values("date").reset_index(drop=True)

    # Reindex to daily calendar to make gaps explicit.
    start = sub["date"].min()
    end = sub["date"].max()
    if pd.isna(start) or pd.isna(end) or start >= end:
        return pd.DataFrame()
    full = pd.date_range(start=start, end=end, freq="D")
    df = sub.set_index("date").reindex(full)
    df.index.name = "date"
    df = df.reset_index()

    df["q_cfs"] = pd.to_numeric(df["q_cfs"], errors="coerce")
    df["year"] = df["date"].dt.year.astype(int)
    df["month"] = df["date"].dt.month.astype(int)
    df["dayofyear"] = df["date"].dt.dayofyear.astype(int)
    df["dow"] = df["date"].dt.dayofweek.astype(int)
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df[ERA_COLUMN] = df["year"].map(_event_era).astype(int)

    # Past-only rolling features.
    s = df["q_cfs"].shift(1)
    df["q_1d"] = s
    df["q_mean_7d"] = s.rolling(cfg.lb_short).mean()
    df["q_std_7d"] = s.rolling(cfg.lb_short).std()
    df["q_mean_30d"] = s.rolling(cfg.lb_long).mean()
    df["q_std_30d"] = s.rolling(cfg.lb_long).std()
    df["q_min_30d"] = s.rolling(cfg.lb_long).min()
    df["q_max_30d"] = s.rolling(cfg.lb_long).max()
    df["q_missing_30d"] = s.rolling(cfg.lb_long).apply(lambda x: float(np.isnan(x).sum()), raw=True)

    # Log-scale dynamics.
    df["logq_1d"] = np.log(np.maximum(df["q_1d"].to_numpy(dtype=float), 0.0) + 1.0)
    logq = np.log(np.maximum(df["q_cfs"].to_numpy(dtype=float), 0.0) + 1.0)
    df["dlogq_1d"] = pd.Series(logq, index=df.index).diff(1).shift(1)
    df["abs_dlogq_mean7"] = df["dlogq_1d"].abs().rolling(cfg.lb_short).mean()

    # Flashiness proxy (past-only): mean absolute fractional change (7d).
    dq = df["q_cfs"].diff(1).shift(1)
    denom = df["q_cfs"].shift(1).abs() + 1.0
    df["flashiness_7d"] = (dq.abs() / denom).rolling(cfg.lb_short).mean()

    token = _station_token(str(site_no))
    df[STATION_COLUMN] = token

    # Drop rows with too many missing observations in past window.
    df = df[df["q_missing_30d"].fillna(9999) <= 5].copy()
    if df.empty:
        return pd.DataFrame()

    # Require baseline and current q.
    df = df.dropna(subset=["q_1d", "q_mean_30d"]).copy()
    if df.empty:
        return pd.DataFrame()

    # Slice: low-flow baseline (station-relative, train-era calibrated).
    train_mask = df["year"].astype(int) <= cfg.train_max_year
    base_vals = df.loc[train_mask, "q_mean_30d"].to_numpy(dtype=float)
    base_vals = base_vals[np.isfinite(base_vals)]
    if base_vals.size < 365:
        return pd.DataFrame()
    low_thr = float(np.quantile(base_vals, 0.25))
    df[SLICE_COLUMN] = (df["q_mean_30d"].to_numpy(dtype=float) <= low_thr).astype(int)

    return df


def _assign_split(data: pd.DataFrame, cfg: Config) -> pd.Series:
    years = data["date"].dt.year.astype(int)

    flashy_cut = data.groupby(STATION_COLUMN, sort=False)["flashiness_7d"].transform(
        lambda s: float(np.nanquantile(s.to_numpy(dtype=float)[np.isfinite(s.to_numpy(dtype=float))], 0.75))
        if np.isfinite(s.to_numpy(dtype=float)).sum() >= 20
        else float("nan")
    )
    is_flashy = data["flashiness_7d"] >= flashy_cut

    key = data[STATION_COLUMN].astype(str) + "_" + data["dayofyear"].astype(int).astype(str)
    key_pct = key.map(_hash_percent).astype(int)

    is_test = years >= cfg.test_min_year
    is_test |= years.isin(cfg.bridge_years) & is_flashy & (key_pct < cfg.bridge_test_rate_flashy)
    is_test |= years.isin(cfg.bridge_years) & (~is_flashy) & (key_pct < cfg.bridge_test_rate_stable)
    is_test &= years > cfg.train_max_year
    return is_test


def main() -> None:
    cfg = Config()
    if not UPSTREAM_CACHE_DIR.exists():
        raise FileNotFoundError(f"Missing upstream cache dir: {UPSTREAM_CACHE_DIR}")

    rdb_paths = sorted(UPSTREAM_CACHE_DIR.glob("*.rdb"))
    if not rdb_paths:
        raise RuntimeError(f"No .rdb files found in {UPSTREAM_CACHE_DIR}")

    # Pick up to max_sites deterministic by site_no from filenames.
    site_nos: List[str] = []
    for p in rdb_paths:
        # pattern dv_{site}_{param}_{start}_{end}.rdb
        parts = p.stem.split("_")
        if len(parts) >= 2 and parts[0] == "dv":
            site_nos.append(parts[1])
    site_nos = sorted({s for s in site_nos if s.isdigit()})[: cfg.max_sites]
    if not site_nos:
        raise RuntimeError("No valid site numbers inferred from cache filenames.")

    # Read DV per file and engineer site frames.
    frames: List[pd.DataFrame] = []
    for site_no in site_nos:
        # Find matching rdb file for this site (first match).
        match = next((p for p in rdb_paths if f"dv_{site_no}_" in p.name), None)
        if match is None:
            continue
        dv = _read_dv_rdb(match, parameter_cd=cfg.parameter_cd)
        eng = _engineer_site(site_no, dv, cfg)
        if not eng.empty:
            frames.append(eng)

    if not frames:
        raise RuntimeError("No engineered site frames produced.")

    base = pd.concat(frames, axis=0, ignore_index=True)
    base["is_test"] = _assign_split(base, cfg).astype(int)

    # Panelize horizons for target construction.
    panel_rows: List[pd.DataFrame] = []
    taus: Dict[str, float] = {}

    for h in cfg.horizons_days:
        sub = base.copy()
        sub[HORIZON_COLUMN] = int(h)

        # Compute future max surge: max_{k=0..h-1} (log(q_{t+k}+1)) - log(q_{t-1}+1)
        # Features are built using past-only information up to t-1 (e.g., `q_1d` / `logq_1d`),
        # so the label is allowed to use q_t..q_{t+h-1}.
        sub = sub.dropna(subset=["logq_1d"]).copy()
        g = sub.groupby(STATION_COLUMN, sort=False)
        sub["_logq"] = np.log(np.maximum(sub["q_cfs"].to_numpy(dtype=float), 0.0) + 1.0)
        fut_max = (
            g["_logq"]
            .apply(lambda s: s.rolling(int(h), min_periods=int(h)).max().shift(-(int(h) - 1)))
            .reset_index(level=0, drop=True)
        )
        sub["surge_mag_h"] = fut_max.to_numpy(dtype=float) - sub["logq_1d"].to_numpy(dtype=float)

        # Keep rows with complete future window.
        sub = sub[np.isfinite(sub["surge_mag_h"].to_numpy(dtype=float))].copy()
        if sub.empty:
            continue

        train_sub = sub[sub["is_test"] == 0].copy()
        if train_sub.empty:
            continue

        # Threshold: station-relative when enough history, else global.
        global_tau = float(np.nanquantile(train_sub["surge_mag_h"].to_numpy(dtype=float), cfg.surge_quantile))
        thr_by_station: Dict[str, float] = {}
        for tok, gg in train_sub.groupby(STATION_COLUMN, sort=True):
            v = gg["surge_mag_h"].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            if v.size >= 250:
                thr_by_station[str(tok)] = float(np.quantile(v, cfg.surge_quantile))
        sub["tau_station"] = sub[STATION_COLUMN].map(lambda t: float(thr_by_station.get(str(t), global_tau))).astype(float)
        sub[TARGET_COLUMN] = (sub["surge_mag_h"].to_numpy(dtype=float) >= sub["tau_station"].to_numpy(dtype=float)).astype(int)

        taus[f"h{h}"] = float(global_tau)
        panel_rows.append(sub)

    if not panel_rows:
        raise RuntimeError("No panel rows produced; cannot build dataset.")

    panel = pd.concat(panel_rows, axis=0, ignore_index=True)

    feature_cols = [
        "q_1d",
        "q_mean_7d",
        "q_std_7d",
        "q_mean_30d",
        "q_std_30d",
        "q_min_30d",
        "q_max_30d",
        "logq_1d",
        "dlogq_1d",
        "abs_dlogq_mean7",
        "flashiness_7d",
        "q_missing_30d",
        "month",
        "dayofyear",
    ]
    for c in feature_cols:
        panel[c] = pd.to_numeric(panel[c], errors="coerce")
    panel["missing_count"] = panel[feature_cols].isna().sum(axis=1).astype(int)

    # Deterministic sampling post-split.
    key = panel[STATION_COLUMN].astype(str) + "_" + panel["dayofyear"].astype(int).astype(str) + "_" + panel[HORIZON_COLUMN].astype(str)
    pct = key.map(_hash_percent).astype(int)

    is_test = panel["is_test"].astype(int) > 0
    keep_train = pct < int(cfg.keep_percent_train)
    keep_test = pct < int(cfg.keep_percent_test)
    panel = panel[(~is_test & keep_train) | (is_test & keep_test)].copy()

    train_rows = panel[panel["is_test"] == 0].copy()
    if train_rows.empty:
        raise RuntimeError("No training rows after split; cannot build dataset.")

    panel_all, edges_map = _make_bins(panel, train_rows, feature_cols)

    keep = [STATION_COLUMN, HORIZON_COLUMN, ERA_COLUMN, "is_weekend", "missing_count"] + [f"{c}_bin" for c in feature_cols] + [
        SLICE_COLUMN
    ]
    out = panel_all[keep + [TARGET_COLUMN, "is_test", "date"]].copy()
    out = out.sort_values([STATION_COLUMN, "date", HORIZON_COLUMN]).reset_index(drop=True)
    out[ID_COLUMN] = [
        _row_id(str(st), pd.Timestamp(dt), int(h))
        for st, dt, h in zip(
            out[STATION_COLUMN].astype(str).to_numpy(),
            pd.to_datetime(out["date"]).to_numpy(),
            out[HORIZON_COLUMN].astype(int).to_numpy(),
        )
    ]
    out = out.drop(columns=["date"])

    train = out[out["is_test"] == 0].drop(columns=["is_test"]).copy()
    test = out[out["is_test"] == 1].drop(columns=["is_test", TARGET_COLUMN]).copy()
    solution = out[out["is_test"] == 1][[ID_COLUMN, TARGET_COLUMN, SLICE_COLUMN]].copy()

    train = train.sort_values(ID_COLUMN).reset_index(drop=True)
    test = test.sort_values(ID_COLUMN).reset_index(drop=True)
    solution = solution.sort_values(ID_COLUMN).reset_index(drop=True)

    if len(train) < MIN_TRAIN_ROWS:
        raise RuntimeError(f"Train set too small ({len(train)} rows). Need >= {MIN_TRAIN_ROWS} to avoid rejection risk.")

    sample = test[[ID_COLUMN]].copy()
    sample[PRED_COLUMN] = 0.5

    perfect = solution[[ID_COLUMN, TARGET_COLUMN]].copy()
    perfect[PRED_COLUMN] = perfect[TARGET_COLUMN].astype(float)
    perfect = perfect[[ID_COLUMN, PRED_COLUMN]]

    train.to_csv(COMP_DIR / "train.csv", index=False)
    test.to_csv(COMP_DIR / "test.csv", index=False)
    solution.to_csv(COMP_DIR / "solution.csv", index=False)
    sample.to_csv(COMP_DIR / "sample_submission.csv", index=False)
    perfect.to_csv(COMP_DIR / "perfect_submission.csv", index=False)

    meta = {
        "upstream_cache_dir": str(UPSTREAM_CACHE_DIR),
        "parameter_cd": cfg.parameter_cd,
        "horizons_days": list(cfg.horizons_days),
        "n_bins": int(N_BINS),
        "surge_quantile": float(cfg.surge_quantile),
        "global_tau_by_horizon": taus,
        "bin_edges": edges_map,
        "slice_definition": {"slice_column": SLICE_COLUMN, "rule": "slice_lowflow = 1 if q_mean_30d <= station train-era 25th percentile."},
        "row_counts": {"train": int(len(train)), "test": int(len(test))},
        "positive_rate": {
            "train": float(train[TARGET_COLUMN].mean()) if len(train) else None,
            "test": float(solution[TARGET_COLUMN].mean()) if len(solution) else None,
        },
        "sites_used": int(base[STATION_COLUMN].nunique()),
    }
    (COMP_DIR / "build_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    print("Wrote competition files to", COMP_DIR)
    print("train rows:", len(train), "test rows:", len(test))
    print("train positive rate:", float(train[TARGET_COLUMN].mean()) if len(train) else float("nan"))
    print("test positive rate:", float(solution[TARGET_COLUMN].mean()) if len(solution) else float("nan"))


if __name__ == "__main__":
    main()

