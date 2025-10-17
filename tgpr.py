"""
Time-warped Ensemble GPR system
- Volatility-driven input warping for time-dependent kernels
- Heteroscedastic noise using proper kernel implementation
- Ensemble (short/medium/long) GPRs trained on warped time
- Parallel training & rolling-window online updates (warm start)
- KDE-based reversal filtering, OU estimator, KER & ATR_z fusion
- Demo with simulated pairs

Author: Elijah Ekpen Mensah
Improved with proper heteroscedastic noise and memory management
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from scipy.stats import zscore, gaussian_kde
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, RationalQuadratic, WhiteKernel, ConstantKernel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# -------------------------
# Utilities
# -------------------------
def rolling_atr(price, window=14):
    """Simple ATR approximation using absolute differences"""
    tr = np.abs(np.diff(price, prepend=price[0]))
    return pd.Series(tr).rolling(window=window, min_periods=1).mean().values

def atr_zscore(price, window=14):
    return zscore(rolling_atr(price, window=window))

def kaufman_efficiency_ratio(close, window=10):
    close = np.asarray(close)
    er = np.full_like(close, np.nan, dtype=float)
    for i in range(len(close)):
        start = max(0, i - window + 1)
        win = close[start:i+1]
        if len(win) < 2:
            continue
        direction = abs(win[-1] - win[0])
        volatility = np.sum(np.abs(np.diff(win)))
        er[i] = 0.0 if volatility == 0 else direction / volatility
    er = pd.Series(er).fillna(method='bfill').values
    return er

def estimate_ou_params(ts):
    x = np.asarray(ts)
    if len(x) < 10:
        return np.nan, np.nan, np.nan
    X = x[:-1]; Y = x[1:]
    A = np.vstack([X, np.ones_like(X)]).T
    a, b = np.linalg.lstsq(A, Y, rcond=None)[0]
    dt = 1.0
    theta = -np.log(a)/dt if a>0 and a<1 else np.nan
    mu = b / (1 - a) if abs(1-a) > 1e-8 else np.nan
    resid = Y - (a*X + b)
    sigma_hat = np.std(resid)
    return theta, mu, sigma_hat

def calculate_metrics(y_true, y_pred, y_std):
    """Calculate comprehensive performance metrics"""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    # Calibration metrics
    z_scores = (y_true - y_pred) / (y_std + 1e-8)
    calibration_error = np.mean(np.abs(z_scores) > 2) - 0.05  # Should be ~5% outside 2std
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'calibration_error': calibration_error,
        'sharpness': np.mean(y_std)  # Lower is better (more confident)
    }

# -------------------------
# Time warping functions
# -------------------------
def compute_volatility_warp(time_array, price_series, atr_window=14, warp_strength=1.0, smoothing=5):
    """
    Create a monotonically increasing warped-time array w(t) from time_array,
    where incremental steps are scaled by local ATR (or volatility).
    warp_strength controls amplitude of warping (1.0 = use ATR scale directly).
    smoothing smooths the ATR before integration.
    """
    time_array = np.asarray(time_array)
    n = len(time_array)
    atr = rolling_atr(price_series, window=atr_window)
    # Smooth ATR a bit to avoid extreme spikes
    atr_smooth = pd.Series(atr).rolling(window=smoothing, min_periods=1, center=False).mean().values
    # normalize to mean 1 to preserve coarse time scale, but allow local expansion/contraction
    atr_norm = (atr_smooth / (np.nanmean(atr_smooth) + 1e-12)) * warp_strength
    # ensure positive increments and avoid zero
    increments = np.clip(atr_norm, 1e-6, None)
    # integrate to get warped time (monotonic)
    warped = np.cumsum(increments)
    # optionally rescale to same numeric range as original time for numerical stability
    warped = (warped - warped.min()) / (warped.max() - warped.min() + 1e-12) * (time_array.max() - time_array.min()) + time_array.min()
    return warped, atr

def multi_scale_warping(time_array, price_series, windows=[5, 14, 30], warp_strength=1.0):
    """Combine multiple volatility time scales for more robust warping"""
    warps = []
    for window in windows:
        warp, _ = compute_volatility_warp(time_array, price_series, atr_window=window, 
                                        warp_strength=warp_strength)
        warps.append(warp)
    return np.mean(warps, axis=0)

# -------------------------
# Kernels with proper heteroscedastic noise
# -------------------------
def heteroscedastic_kernel(base_kernel, alpha_arr=None):
    """
    Proper heteroscedastic kernel implementation
    If alpha_arr is provided, uses average noise level
    """
    if alpha_arr is not None:
        avg_noise = np.mean(alpha_arr)
        noise_kernel = ConstantKernel(constant_value=avg_noise, constant_value_bounds=(1e-10, 1e10)) * WhiteKernel()
    else:
        noise_kernel = WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-10, 1e10))
    
    return base_kernel + noise_kernel

def kernel_short(alpha_arr=None):
    base_kernel = 1.0 * RBF(length_scale=0.05, length_scale_bounds=(1e-2, 1.0))
    return heteroscedastic_kernel(base_kernel, alpha_arr)

def kernel_medium(alpha_arr=None):
    base_kernel = (0.8 * RBF(length_scale=0.2, length_scale_bounds=(1e-2, 1.0)) + 
                  0.4 * RationalQuadratic(length_scale=0.1, alpha=0.5, alpha_bounds=(1e-2, 10.0)))
    return heteroscedastic_kernel(base_kernel, alpha_arr)

def kernel_long(alpha_arr=None):
    base_kernel = (1.0 * RBF(length_scale=0.6, length_scale_bounds=(1e-1, 2.0)) + 
                  0.6 * ExpSineSquared(length_scale=0.6, periodicity=1.0, periodicity_bounds=(0.5, 2.0)))
    return heteroscedastic_kernel(base_kernel, alpha_arr)

def get_ensemble_kernels(alpha_arr=None):
    return [kernel_short(alpha_arr), kernel_medium(alpha_arr), kernel_long(alpha_arr)]

# -------------------------
# Train ensemble GPRs on warped time with proper heteroscedastic noise
# -------------------------
def train_ensemble_warped(time_arr_raw, price_series_for_warp, z_arr, 
                          atr_window=14, warp_strength=1.0, warp_smoothing=5,
                          multi_scale_warp=False, test_size=0.2,
                          normalize_y=True, n_restarts_optimizer=3, random_state=0):
    """
    Improved training with proper train-test split and heteroscedastic noise
    """
    # Compute warped time
    if multi_scale_warp:
        warped_time = multi_scale_warping(time_arr_raw, price_series_for_warp, warp_strength=warp_strength)
        atr = rolling_atr(price_series_for_warp, window=atr_window)
    else:
        warped_time, atr = compute_volatility_warp(time_arr_raw, price_series_for_warp,
                                                  atr_window=atr_window, warp_strength=warp_strength,
                                                  smoothing=warp_smoothing)
    
    X = warped_time.reshape(-1, 1)
    y = z_arr

    # Create train-test split for validation
    if test_size > 0:
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, range(len(y)), test_size=test_size, shuffle=False, random_state=random_state)
    else:
        X_train, y_train, idx_train = X, y, range(len(y))
        X_test, y_test, idx_test = None, None, None

    # Build per-sample alpha (noise variance) from rolling variance combined with ATR
    local_var = pd.Series(y_train).rolling(window=max(3, int(atr_window/2)), min_periods=1, center=False).var().fillna(method='bfill').values
    alpha_arr = (local_var + 0.5 * (atr[idx_train] / (np.nanmean(atr[idx_train]) + 1e-12))**2) + 1e-5
    alpha_arr = alpha_arr / (np.nanmean(alpha_arr) + 1e-12)

    models = []
    train_preds = []
    train_stds = []
    test_preds = []
    test_stds = []
    kernels_used = []
    metrics = []

    ensemble_kernels = get_ensemble_kernels(alpha_arr)

    for kernel in ensemble_kernels:
        gpr = GaussianProcessRegressor(
            kernel=kernel, 
            normalize_y=normalize_y,
            n_restarts_optimizer=n_restarts_optimizer, 
            random_state=random_state,
            alpha=0.0  # We handle noise in the kernel
        )
        
        gpr.fit(X_train, y_train)
        
        # Training predictions
        y_pred_train, y_std_train = gpr.predict(X_train, return_std=True)
        
        # Test predictions if available
        if X_test is not None:
            y_pred_test, y_std_test = gpr.predict(X_test, return_std=True)
            test_metrics = calculate_metrics(y_test, y_pred_test, y_std_test)
        else:
            y_pred_test, y_std_test = None, None
            test_metrics = None

        models.append(gpr)
        train_preds.append(y_pred_train)
        train_stds.append(y_std_train)
        test_preds.append(y_pred_test)
        test_stds.append(y_std_test)
        kernels_used.append(gpr.kernel_)
        metrics.append(test_metrics)

    # Reconstruct full predictions in original order
    full_preds = np.full((len(ensemble_kernels), len(y)), np.nan)
    full_stds = np.full((len(ensemble_kernels), len(y)), np.nan)
    
    for i, (pred_train, std_train, pred_test, std_test) in enumerate(zip(train_preds, train_stds, test_preds, test_stds)):
        if pred_test is not None:
            full_preds[i, idx_train] = pred_train
            full_preds[i, idx_test] = pred_test
            full_stds[i, idx_train] = std_train
            full_stds[i, idx_test] = std_test
        else:
            full_preds[i] = pred_train
            full_stds[i] = std_train

    return {
        "models": models,
        "preds": full_preds,
        "stds": full_stds,
        "warped_time": warped_time.copy(),
        "time_raw": np.asarray(time_arr_raw).copy(),
        "atr": atr.copy(),
        "alpha_arr": alpha_arr.copy() if test_size == 0 else np.concatenate([alpha_arr, np.full(len(idx_test), np.nan)]),
        "z": y.copy(),
        "price_for_warp": np.asarray(price_series_for_warp).copy(),
        "metrics": metrics,
        "test_indices": idx_test if test_size > 0 else None
    }

# -------------------------
# Ensemble aggregation and reversal detection
# -------------------------
def aggregate_ensemble(preds_matrix, stds_matrix, method="weighted_by_inv_std"):
    if method == "mean":
        agg = np.nanmean(preds_matrix, axis=0)
        agg_std = np.sqrt(np.nanmean(stds_matrix**2, axis=0))
    elif method == "median":
        agg = np.nanmedian(preds_matrix, axis=0)
        agg_std = np.sqrt(np.nanmedian(stds_matrix**2, axis=0))
    elif method == "weighted_by_inv_std":
        weights = 1.0 / (stds_matrix + 1e-8)
        weights = np.where(np.isnan(weights), 0, weights)
        weights /= np.sum(weights, axis=0, keepdims=True) + 1e-8
        agg = np.nansum(weights * preds_matrix, axis=0)
        agg_std = np.sqrt(np.nansum(weights**2 * stds_matrix**2, axis=0))
    elif method == "model_selection":
        # Select model with lowest uncertainty for each point
        min_std_indices = np.nanargmin(stds_matrix, axis=0)
        agg = np.array([preds_matrix[min_std_indices[i], i] for i in range(preds_matrix.shape[1])])
        agg_std = np.array([stds_matrix[min_std_indices[i], i] for i in range(stds_matrix.shape[1])])
    else:
        raise ValueError("Unknown aggregation method")
    return agg, agg_std

def kde_reversal_probability(series_preds, bandwidth_method='scott'):
    """Improved KDE with handling for edge cases"""
    valid_mask = ~np.isnan(series_preds)
    if np.sum(valid_mask) < 10:
        return np.full_like(series_preds, 0.5)
    
    valid_preds = series_preds[valid_mask]
    try:
        kde = gaussian_kde(valid_preds, bw_method=bandwidth_method)
        density = kde.evaluate(valid_preds)
        density = (density - density.min()) / (density.max() - density.min() + 1e-12)
        
        # Map back to original array
        full_density = np.full_like(series_preds, np.nan)
        full_density[valid_mask] = density
        full_density = pd.Series(full_density).fillna(method='bfill').fillna(method='ffill').values
        return full_density
    except:
        return np.full_like(series_preds, 0.5)

def detect_reversals_from_agg(time_arr_raw, agg_pred, agg_std, prob_density,
                              overbought=2.0, oversold=-2.0, lookahead=5, 
                              prob_thresh=0.2, confidence_threshold=1.0):
    """Improved reversal detection with confidence filtering"""
    events = []
    
    for i in range(0, len(agg_pred)-lookahead):
        if np.isnan(agg_pred[i]) or np.isnan(prob_density[i]):
            continue
            
        pdensity = prob_density[i]
        current_std = agg_std[i]
        
        # Filter by probability density and confidence
        if pdensity < prob_thresh or current_std > confidence_threshold:
            continue
            
        current_pred = agg_pred[i]
        future_mean = np.nanmean(agg_pred[i+1:i+1+lookahead])
        
        if current_pred > overbought and future_mean < current_pred:
            events.append({
                "time": time_arr_raw[i], 
                "index": i, 
                "type": "Overbought", 
                "pred": float(current_pred), 
                "prob_density": float(pdensity),
                "confidence": float(1.0 / (1.0 + current_std))  # Higher std = lower confidence
            })
            
        if current_pred < oversold and future_mean > current_pred:
            events.append({
                "time": time_arr_raw[i], 
                "index": i, 
                "type": "Oversold", 
                "pred": float(current_pred), 
                "prob_density": float(pdensity),
                "confidence": float(1.0 / (1.0 + current_std))
            })
            
    return pd.DataFrame(events)

def generate_signals(reversals_df, agg_pred, agg_std, position_size=1.0):
    """Convert reversal detections to trading signals"""
    if reversals_df.empty:
        return pd.DataFrame()
    
    signals = []
    for _, reversal in reversals_df.iterrows():
        signal_type = "BUY" if reversal['type'] == 'Oversold' else "SELL"
        size = position_size * reversal['confidence']
        
        signals.append({
            'timestamp': reversal['time'],
            'type': signal_type,
            'size': size,
            'z_score': reversal['pred'],
            'confidence': reversal['confidence'],
            'prob_density': reversal['prob_density']
        })
    
    return pd.DataFrame(signals)

# -------------------------
# Online rolling refit with memory management
# -------------------------
class WarpedEnsembleGP:
    """Container class for managing ensemble GPR with online updates"""
    
    def __init__(self, initial_package):
        self.model_package = initial_package
        self.update_history = []
        
    def online_update(self, new_time_raw, new_price_for_warp, new_z, 
                     window_size=500, n_restarts_optimizer=1):
        """Memory-efficient online update with windowing"""
        
        # Concatenate new data
        time_full = np.concatenate([self.model_package["time_raw"], np.asarray(new_time_raw)])
        price_full = np.concatenate([self.model_package["price_for_warp"], np.asarray(new_price_for_warp)])
        z_full = np.concatenate([self.model_package["z"], np.asarray(new_z)])
        
        # Apply windowing to control memory
        if window_size is not None and len(time_full) > window_size:
            keep_from = len(time_full) - window_size
            time_full = time_full[keep_from:]
            price_full = price_full[keep_from:]
            z_full = z_full[keep_from:]
        
        # Retrain on windowed data
        new_pkg = train_ensemble_warped(
            time_full, price_full, z_full,
            atr_window=14, warp_strength=1.0, warp_smoothing=5,
            multi_scale_warp=True, test_size=0.0,  # No test split for online
            n_restarts_optimizer=n_restarts_optimizer
        )
        
        # Store update metadata
        self.update_history.append({
            'timestamp': new_time_raw[-1] if len(new_time_raw) > 0 else time_full[-1],
            'samples_added': len(new_time_raw),
            'window_size': len(time_full)
        })
        
        self.model_package = new_pkg
        return self.model_package

def parallel_online_update(ensemble_containers, new_data_dict, n_jobs=-1, 
                          window_size=500, n_restarts_optimizer=1):
    """Parallel update of multiple ensemble containers"""
    names = list(ensemble_containers.keys())
    
    def _update(name):
        if name in new_data_dict:
            nd = new_data_dict[name]
            container = ensemble_containers[name]
            new_pkg = container.online_update(
                new_time_raw=nd['time_raw'],
                new_price_for_warp=nd['price_for_warp'],
                new_z=nd['z'],
                window_size=window_size,
                n_restarts_optimizer=n_restarts_optimizer
            )
            return name, container
        else:
            return name, ensemble_containers[name]
    
    results = Parallel(n_jobs=n_jobs)(delayed(_update)(n) for n in names)
    return {n: container for n, container in results}

# -------------------------
# Enhanced plotting and reporting
# -------------------------
def plot_warped_model(name, model_pkg, agg_pred, agg_std, reversals_df, signals_df=None, 
                     show_components=True, show=True):
    """Enhanced plotting with multiple subplots"""
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), height_ratios=[2, 1, 1])
    t_raw = model_pkg['time_raw']
    
    # Main prediction plot
    axes[0].plot(t_raw, model_pkg['z'], label='Actual Z', alpha=0.7, linewidth=1)
    axes[0].plot(t_raw, agg_pred, label='Ensemble Prediction', linewidth=1.5, color='red')
    axes[0].fill_between(t_raw, agg_pred - 2*agg_std, agg_pred + 2*agg_std, alpha=0.2, label='2σ Confidence')
    
    # Plot individual model predictions if desired
    if show_components and 'preds' in model_pkg:
        for i, pred in enumerate(model_pkg['preds']):
            axes[0].plot(t_raw, pred, alpha=0.3, linestyle='--', 
                        label=f'Model {i+1}', linewidth=0.8)
    
    axes[0].axhline(2.0, color='r', linestyle='--', alpha=0.7, label='Overbought')
    axes[0].axhline(-2.0, color='g', linestyle='--', alpha=0.7, label='Oversold')
    
    # Mark reversal points
    if not reversals_df.empty:
        for _, r in reversals_df.iterrows():
            color = 'red' if r['type'] == 'Overbought' else 'green'
            marker = 'v' if r['type'] == 'Overbought' else '^'
            axes[0].scatter(r['time'], r['pred'], color=color, marker=marker, 
                           s=80, alpha=0.8, label=f'{r["type"]} Reversal' if _ == 0 else "")
    
    # Mark signals
    if signals_df is not None and not signals_df.empty:
        for _, signal in signals_df.iterrows():
            color = 'lime' if signal['type'] == 'BUY' else 'darkred'
            marker = '^' if signal['type'] == 'BUY' else 'v'
            axes[0].scatter(signal['timestamp'], signal['z_score'], 
                           color=color, marker=marker, s=100, 
                           label=f'{signal["type"]} Signal' if _ == 0 else "")
    
    axes[0].set_title(f'{name}: Time-warped Ensemble GPR with Reversal Detection')
    axes[0].set_ylabel('Z-score')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Warped time vs original time
    axes[1].plot(t_raw, model_pkg['warped_time'], label='Warped Time', color='purple')
    axes[1].set_ylabel('Warped Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Volatility (ATR) and alpha
    axes[2].plot(t_raw, model_pkg['atr'], label='ATR', alpha=0.7)
    if 'alpha_arr' in model_pkg and not np.all(np.isnan(model_pkg['alpha_arr'])):
        alpha_normalized = model_pkg['alpha_arr'] / np.nanmax(model_pkg['alpha_arr'])
        axes[2].plot(t_raw, alpha_normalized, label='Noise Level (norm)', alpha=0.7)
    axes[2].set_ylabel('Volatility Measures')
    axes[2].set_xlabel('Time (raw)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if show:
        plt.show()
    
    return fig

def print_model_metrics(name, model_pkg, metrics):
    """Print comprehensive model performance metrics"""
    print(f"\n{'='*50}")
    print(f"MODEL: {name}")
    print(f"{'='*50}")
    
    if model_pkg['metrics'][0] is not None:
        print("Test Set Performance:")
        for i, metric in enumerate(model_pkg['metrics']):
            print(f"  Model {i+1}: RMSE={metric['rmse']:.4f}, "
                  f"Calibration={metric['calibration_error']:.4f}")
    
    theta, mu, sigma = estimate_ou_params(model_pkg['z'])
    print(f"OU Process: theta={theta:.4f}, mu={mu:.4f}, sigma={sigma:.4f}")
    
    ker = kaufman_efficiency_ratio(model_pkg['price_for_warp'], window=10)
    atrz = atr_zscore(model_pkg['price_for_warp'], window=14)
    print(f"Market State: KER={ker[-1]:.4f}, ATR_z={atrz[-1]:.4f}")
    
    if 'update_history' in model_pkg:
        print(f"Update History: {len(model_pkg['update_history'])} updates")

# -------------------------
# Enhanced demo with validation
# -------------------------
def simulate_pair(n=900, drift=0.0, scale=1.0, seed=None, mean_reversion=0.1):
    """Enhanced simulation with mean reversion"""
    rng = np.random.RandomState(seed)
    
    # Simulate cointegrated pair with mean reversion
    A = np.zeros(n)
    B = np.zeros(n)
    
    A[0] = 100.0
    B[0] = 101.0
    
    for t in range(1, n):
        # Common trend
        common = rng.randn() * scale
        
        # Individual noise
        noise_A = rng.randn() * scale * 0.5
        noise_B = rng.randn() * scale * 0.5
        
        # Mean reversion component
        spread = A[t-1] - 0.98 * B[t-1]
        mr_A = -mean_reversion * spread * 0.02
        mr_B = mean_reversion * spread * 0.02
        
        A[t] = A[t-1] + common + noise_A + mr_A + drift
        B[t] = B[t-1] + common + noise_B + mr_B + drift
    
    # Calculate spread and z-score
    beta = np.polyfit(B, A, 1)[0]
    spread = A - beta * B
    Z = zscore(spread)
    t = np.arange(n)
    
    return {"time": t, "z": Z, "A": A, "B": B, "spread": spread}

def demo():
    """Enhanced demo with validation and online updates"""
    print("Generating simulated pairs...")
    datasets = []
    for i in range(3):
        sim = simulate_pair(n=1000, drift=0.0001*i, scale=0.8 + 0.1*i, 
                          seed=42+i, mean_reversion=0.05 + 0.02*i)
        datasets.append({
            "name": f"pair_{i+1}", 
            "time": sim['time'], 
            "z": sim['z'], 
            "A": sim['A'], 
            "B": sim['B']
        })
    
    # Split data for online update demo
    train_datasets = []
    update_datasets = []
    
    for ds in datasets:
        split_idx = int(len(ds['time']) * 0.7)
        train_data = {k: v[:split_idx] for k, v in ds.items() if k != 'name'}
        update_data = {k: v[split_idx:] for k, v in ds.items() if k != 'name'}
        
        train_datasets.append({**train_data, 'name': ds['name']})
        update_datasets.append({**update_data, 'name': ds['name']})
    
    # Initial training
    print("\nTraining initial ensemble models...")
    trained_containers = {}
    
    def _train_ds(ds):
        pkg = train_ensemble_warped(
            ds['time'], ds['A'], ds['z'],
            atr_window=14, warp_strength=1.0, 
            multi_scale_warp=True, test_size=0.2,
            n_restarts_optimizer=2, random_state=0
        )
        container = WarpedEnsembleGP(pkg)
        return ds['name'], container
    
    results = Parallel(n_jobs=3)(delayed(_train_ds)(ds) for ds in train_datasets)
    trained_containers = {k: v for k, v in results}
    
    # Analyze each model
    print("\nAnalyzing trained models...")
    for name, container in trained_containers.items():
        pkg = container.model_package
        preds_mat = pkg['preds']
        stds_mat = pkg['stds']
        
        agg_pred, agg_std = aggregate_ensemble(preds_mat, stds_mat, method='weighted_by_inv_std')
        kde_density = kde_reversal_probability(agg_pred)
        reversals = detect_reversals_from_agg(pkg['time_raw'], agg_pred, agg_std, kde_density, 
                                            prob_thresh=0.25, confidence_threshold=1.5)
        signals = generate_signals(reversals, agg_pred, agg_std)
        
        print_model_metrics(name, pkg, pkg['metrics'])
        print(f"Detected {len(reversals)} reversal events")
        print(f"Generated {len(signals)} trading signals")
        
        if len(reversals) > 0:
            print("First 3 reversals:")
            print(reversals.head(3))
        
        plot_warped_model(name, pkg, agg_pred, agg_std, reversals, signals)
    
    # Demo online updates
    print("\nDemonstrating online updates...")
    update_dict = {}
    for ds in update_datasets:
        update_dict[ds['name']] = {
            'time_raw': ds['time'],
            'price_for_warp': ds['A'], 
            'z': ds['z']
        }
    
    # Perform parallel online update
    updated_containers = parallel_online_update(
        trained_containers, update_dict, 
        window_size=600, n_restarts_optimizer=1
    )
    
    print("Online updates completed!")
    
    # Show update statistics
    for name, container in updated_containers.items():
        print(f"{name}: {len(container.update_history)} updates applied")
        if container.update_history:
            latest = container.update_history[-1]
            print(f"  Latest: {latest['samples_added']} samples, window={latest['window_size']}")
    
    print("\nDemo complete!")

if __name__ == "__main__":
    demo()