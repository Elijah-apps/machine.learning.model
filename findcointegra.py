#!/usr/bin/env python3
"""
find_cointegrated_pairs.py

Usage:
    python find_cointegrated_pairs.py --query path/to/query.csv --dir path/to/csv_dir --out results_folder

If --dir is omitted, uses the directory containing the query file.
Saves:
 - plots to <out>/plots/
 - summary CSV to <out>/summary.csv
 - prints ranked table to stdout
"""

import argparse
import os
import glob
import math
from typing import Tuple, Dict, Any, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint, adfuller
import statsmodels.api as sm
from scipy import stats

# ---------------------------------------------------------
# Helper functions for reading MT5-style CSVs and aligning
# ---------------------------------------------------------
POSSIBLE_TIME_COLS = ["time", "timestamp", "Date", "datetime", "Time", "TimeStamp"]

def read_mt5_csv(path: str) -> pd.DataFrame:
    """
    Read CSV exported from MT5 or similar. Attempts to parse a datetime/time column and a price column.
    Returns DataFrame indexed by datetime and containing 'close' column.
    """
    df = pd.read_csv(path)
    # find time column
    time_col = None
    for c in POSSIBLE_TIME_COLS:
        if c in df.columns:
            time_col = c
            break
    if time_col is None:
        # fallback: first column
        time_col = df.columns[0]
    # parse datetime
    try:
        df[time_col] = pd.to_datetime(df[time_col])
    except Exception:
        # try common formats
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    df = df.dropna(subset=[time_col])
    df = df.set_index(time_col).sort_index()
    # find close-like column
    close_col = None
    for c in ["close", "Close", "ClosePrice", "Adj Close", "PRICE", "price"]:
        if c in df.columns:
            close_col = c
            break
    if close_col is None:
        # fallback heuristic: numeric column other than time with the least missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise ValueError(f"No numeric columns found in {path}")
        close_col = numeric_cols[0]
    res = df[[close_col]].rename(columns={close_col: "close"}).copy()
    return res

def align_on_index(a: pd.Series, b: pd.Series) -> pd.DataFrame:
    """
    Align two series on intersection of timestamps, drop NaNs.
    Returns DataFrame with columns ['x','y'] where x is a, y is b.
    """
    df = pd.concat([a.rename("x"), b.rename("y")], axis=1).dropna()
    return df

# ---------------------------------------------------------
# Statistics & tests
# ---------------------------------------------------------
def ols_hedge_ratio(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Regress y ~ x (y = beta * x + eps). Returns beta (hedge ratio) and intercept.
    Uses OLS (no constant optionally, but we include constant).
    """
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    intercept = model.params[0]
    beta = model.params[1]
    return beta, intercept

def half_life(spread: np.ndarray) -> float:
    """
    Estimate half-life of mean reversion for spread using method:
      dS_t = a + b * S_{t-1} + err -> half-life = -ln(2)/b
    If b ~ 0 or positive, return large number (np.inf)
    """
    s = spread
    s_lag = s[:-1]
    s_diff = np.diff(s)
    if len(s_lag) < 5:
        return float('inf')
    X = sm.add_constant(s_lag)
    res = sm.OLS(s_diff, X).fit()
    b = res.params[1]
    if b >= 0:
        return float('inf')
    try:
        halflife = -math.log(2) / b
    except Exception:
        halflife = float('inf')
    return float(halflife)

def adf_test(residuals: np.ndarray):
    """
    Augmented Dickey-Fuller test on residuals. Returns (adf_stat, pvalue, usedlag)
    """
    try:
        adf_res = adfuller(residuals, autolag='AIC')
        return float(adf_res[0]), float(adf_res[1]), int(adf_res[2])
    except Exception:
        return np.nan, np.nan, np.nan

# ---------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------
def save_pair_plots(times: pd.DatetimeIndex, x: np.ndarray, y: np.ndarray, spread: np.ndarray,
                    zscore: np.ndarray, beta: float, intercept: float, outpath_prefix: str):
    """
    Create and save multiple plots showing relationship:
     - scatter x vs y with OLS line
     - spread over time with rolling mean/std and zscore plot
     - histogram of spread
    Saves files like <outpath_prefix>_scatter.png, _spread.png, _hist.png
    """
    try:
        # Scatter with OLS line
        plt.figure(figsize=(7,5))
        plt.scatter(x, y, s=4, alpha=0.6)
        xs = np.array([np.min(x), np.max(x)])
        ys = intercept + beta * xs
        plt.plot(xs, ys, linewidth=1.5)
        plt.xlabel("Query price")
        plt.ylabel("Other price")
        plt.title("Scatter & OLS")
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(f"{outpath_prefix}_scatter.png", dpi=150)
        plt.close()

        # Spread over time + zscore (two-subplots vertical)
        fig, axs = plt.subplots(2,1, figsize=(10,6), sharex=True, gridspec_kw={'height_ratios':[2,1]})
        axs[0].plot(times, spread, linewidth=0.7, label='spread')
        roll_mean = pd.Series(spread, index=times).rolling(window=max(5, int(len(spread)*0.05))).mean()
        roll_std = pd.Series(spread, index=times).rolling(window=max(5, int(len(spread)*0.05))).std()
        axs[0].plot(times, roll_mean, linewidth=0.9, label='rolling mean')
        axs[0].fill_between(times, roll_mean - 2*roll_std, roll_mean + 2*roll_std, alpha=0.1)
        axs[0].legend()
        axs[0].set_title("Spread (y - beta*x - intercept)")

        axs[1].plot(times, zscore, linewidth=0.8)
        axs[1].axhline(0, linestyle='--')
        axs[1].axhline(2, linestyle='--', alpha=0.6)
        axs[1].axhline(-2, linestyle='--', alpha=0.6)
        axs[1].set_title("Spread z-score")
        plt.tight_layout()
        plt.savefig(f"{outpath_prefix}_spread.png", dpi=150)
        plt.close()

        # Histogram of spread
        plt.figure(figsize=(6,4))
        plt.hist(spread, bins=60)
        plt.title("Spread histogram")
        plt.tight_layout()
        plt.savefig(f"{outpath_prefix}_hist.png", dpi=150)
        plt.close()
    except Exception as e:
        print("Warning: failed to save plots:", e)

# ---------------------------------------------------------
# Main testing per pair
# ---------------------------------------------------------
def analyze_pair(query_ser: pd.Series, other_ser: pd.Series, other_name: str) -> Dict[str, Any]:
    df = align_on_index(query_ser, other_ser)
    result = {"other": other_name, "n": len(df)}
    if len(df) < 30:
        # Not enough samples: return NaNs but include n
        result.update({
            "beta": np.nan, "intercept": np.nan, "corr": np.nan,
            "coint_stat": np.nan, "coint_p": np.nan,
            "adf_stat": np.nan, "adf_p": np.nan, "adf_lag": np.nan,
            "halflife": np.nan, "spread_mean": np.nan, "spread_std": np.nan,
            "zscore_latest": np.nan
        })
        return result

    x = df['x'].values
    y = df['y'].values

    # Hedge ratio via OLS
    beta, intercept = ols_hedge_ratio(x, y)

    # Spread
    spread = y - (beta * x + intercept)
    # z-score latest
    spread_mean = float(np.mean(spread))
    spread_std = float(np.std(spread, ddof=1)) if len(spread) > 1 else np.nan
    zscore = (spread - spread_mean) / (spread_std if spread_std != 0 else 1)
    zscore_latest = float(zscore[-1]) if len(zscore)>0 else np.nan

    # Correlation
    corr = float(np.corrcoef(x, y)[0,1])

    # Engle-Granger test (coint from statsmodels) -> returns (tstat, pvalue, crit)
    try:
        coint_res = coint(df['y'], df['x'])
        coint_stat = float(coint_res[0])
        coint_p = float(coint_res[1])
    except Exception:
        coint_stat, coint_p = np.nan, np.nan

    # ADF on residuals
    adf_stat, adf_p, adf_lag = adf_test(spread)

    # Half-life
    hl = half_life(spread)

    result.update({
        "beta": float(beta), "intercept": float(intercept),
        "corr": corr,
        "coint_stat": coint_stat, "coint_p": coint_p,
        "adf_stat": adf_stat, "adf_p": adf_p, "adf_lag": adf_lag,
        "halflife": hl,
        "spread_mean": spread_mean, "spread_std": spread_std,
        "zscore_latest": zscore_latest,
        "n_common": len(df)
    })
    # also keep data needed for plotting
    result["_times"] = df.index
    result["_x"] = x
    result["_y"] = y
    result["_spread"] = spread
    result["_zscore"] = zscore
    return result

def score_result(r: Dict[str, Any]) -> float:
    """
    Create a score that ranks candidate files relative to the query.
    Lower is better. Components:
     - Engle-Granger p-value (lower better)
     - ADF p-value on residuals (lower better)
     - abs(correlation) (higher better)
     - half-life (lower better)
     - number of matching rows (higher better)
    This is heuristic; users can modify weights as desired.
    """
    # handle missing
    p_e = r.get("coint_p", np.nan)
    p_adf = r.get("adf_p", np.nan)
    corr = r.get("corr", 0.0)
    hl = r.get("halflife", np.inf)
    n = r.get("n_common", 0)

    # Normalize terms and combine
    # clamp p-values
    if np.isnan(p_e):
        p_e = 1.0
    if np.isnan(p_adf):
        p_adf = 1.0
    # small epsilon to avoid division by zero
    eps = 1e-8
    # weights
    w_p_e = 0.5
    w_p_adf = 0.25
    w_corr = -0.15   # negative because higher corr improves score (we subtract)
    w_hl = 0.05
    w_n = -0.05

    score = (w_p_e * p_e) + (w_p_adf * p_adf) + (w_corr * abs(corr)) + (w_hl * (min(hl, 1e6)/100.0)) + (w_n * (n/1000.0))
    # ensure numeric
    return float(score)

# ---------------------------------------------------------
# CLI / main
# ---------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Find cointegrated pairs relative to a query CSV (MT5 style).")
    p.add_argument("--query", "-q", required=True, help="Path to query CSV")
    p.add_argument("--dir", "-d", default=None, help="Directory to search for other CSVs (defaults to query file directory)")
    p.add_argument("--out", "-o", default="cointegration_results", help="Output folder to save plots and summary")
    p.add_argument("--min_rows", type=int, default=30, help="Minimum overlapping rows to attempt analysis")
    p.add_argument("--ext", default="csv", help="CSV file extension to search for")
    args = p.parse_args()

    query_path = os.path.abspath(args.query)
    if not os.path.exists(query_path):
        raise FileNotFoundError(query_path)
    search_dir = args.dir if args.dir is not None else os.path.dirname(query_path)
    search_dir = os.path.abspath(search_dir)

    out_dir = os.path.abspath(args.out)
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # read query
    try:
        query_df = read_mt5_csv(query_path)
        query_ser = query_df['close']
    except Exception as e:
        raise RuntimeError(f"Failed to read query CSV: {e}")

    # find other CSVs
    pattern = os.path.join(search_dir, f"*.{args.ext}")
    files = sorted(glob.glob(pattern))
    # exclude query itself
    files = [f for f in files if os.path.abspath(f) != query_path]

    if not files:
        print("No other CSV files found in directory.")
        return

    results = []
    for f in files:
        name = os.path.basename(f)
        try:
            other_df = read_mt5_csv(f)
            other_ser = other_df['close']
        except Exception as e:
            print(f"Skipping {name}: failed to read ({e})")
            continue
        res = analyze_pair(query_ser, other_ser, name)
        if res.get("n_common", 0) < args.min_rows:
            print(f"Skipping {name}: not enough overlapping rows ({res.get('n_common',0)})")
            continue

        # save plots
        prefix = os.path.join(plots_dir, os.path.splitext(name)[0] + "_vs_" + os.path.splitext(os.path.basename(query_path))[0])
        save_pair_plots(res["_times"], res["_x"], res["_y"], res["_spread"], res["_zscore"], res["beta"], res["intercept"], prefix)

        results.append(res)

    if not results:
        print("No candidate pairs analyzed (maybe none had sufficient overlapping rows).")
        return

    # compute scores
    for r in results:
        r['score'] = score_result(r)

    # create DataFrame summary and sort
    summary_cols = ["other", "n_common", "beta", "intercept", "corr", "coint_stat", "coint_p",
                    "adf_stat", "adf_p", "halflife", "spread_mean", "spread_std", "zscore_latest", "score"]
    summary = pd.DataFrame([{k: r.get(k, np.nan) for k in summary_cols} for r in results])
    summary = summary.sort_values(by="score", ascending=True).reset_index(drop=True)

    # Save CSV summary
    summary_csv = os.path.join(out_dir, "summary.csv")
    os.makedirs(out_dir, exist_ok=True)
    summary.to_csv(summary_csv, index=False)

    # Print a ranked text table
    pd.set_option('display.max_rows', None)
    print("\nRanked candidates (lower score = better match):\n")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print(f"\nPlots saved to: {plots_dir}")
    print(f"Summary CSV saved to: {summary_csv}")

if __name__ == "__main__":
    main()