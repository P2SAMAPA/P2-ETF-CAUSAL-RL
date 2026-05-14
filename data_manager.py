"""data_manager.py — Data loading and feature engineering for Causal RL engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

import config

ALL_TICKERS = sorted(set(
    config.EQUITY_SECTORS_TICKERS + config.FI_COMMODITIES_TICKERS
))


def load_data(token: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download master_data.parquet → (log_returns, macro_df)."""
    file_path = hf_hub_download(
        repo_id=config.HF_DATA_REPO,
        filename=config.HF_DATA_FILE,
        repo_type="dataset",
        token=token,
        cache_dir="./hf_cache",
    )
    df = pd.read_parquet(file_path)
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={"index": "Date"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True).set_index("Date")

    available   = [t for t in ALL_TICKERS if t in df.columns]
    prices      = df[available].ffill()
    log_returns = np.log(prices / prices.shift(1)).dropna()

    macro_cols = [c for c in config.MACRO_COLS if c in df.columns]
    macro_df   = df[macro_cols].reindex(log_returns.index).ffill().fillna(0.0)

    print(
        f"Loaded {len(log_returns)} rows × {len(log_returns.columns)} ETFs"
        f" | Macro: {macro_cols}"
    )
    return log_returns, macro_df


def build_joint_array(
    log_returns: pd.DataFrame,
    macro_df: pd.DataFrame,
    tickers: list[str],
) -> tuple[np.ndarray, list[str], pd.DatetimeIndex]:
    """Stack ETF returns and macro into a single (T, n_vars) array.

    Used as input for causal graph discovery.

    Returns
    -------
    data   : (T, n_etf + n_macro) float64 — z-scored
    names  : list of variable names aligned to columns
    dates  : DatetimeIndex
    """
    avail  = [t for t in tickers if t in log_returns.columns]
    mac_c  = [c for c in config.MACRO_COLS if c in macro_df.columns]

    ret_df = log_returns[avail].copy()
    mac_df = macro_df[mac_c].copy()

    # Align indices
    idx      = ret_df.index.intersection(mac_df.index)
    ret_arr  = ret_df.loc[idx].values
    mac_arr  = mac_df.loc[idx].values

    # Z-score each variable for graph discovery
    def _zs(arr):
        mu  = arr.mean(axis=0, keepdims=True)
        std = arr.std(axis=0, keepdims=True) + 1e-8
        return (arr - mu) / std

    data  = np.concatenate([_zs(ret_arr), _zs(mac_arr)], axis=1)
    names = avail + mac_c

    return data.astype(np.float64), names, pd.DatetimeIndex(idx)
