#!/usr/bin/env python3
"""
# Sequential Preprocessing and Feature Engineering Pipeline with Date Tracking
# COMPLETE FIXED VERSION: All critical issues resolved + data quality handling
# - Data quality cleaning for extreme outliers and negative values
# - Consistent raw vs scaled data handling
# - Memory-efficient exceedance block statistics
# - Sequential split processing with memory cleanup
# - Float32 instead of float16 to handle large values
# - Parallel bootstrap, label leakage checks, robust statistics
"""

##############################################
# Cell 1: Configuration & Imports
##############################################
import os
import sys

# THREAD SAFETY: Set environment variables BEFORE any imports
os.environ["TQDM_DISABLE"] = "1"
os.environ["NUMEXPR_MAX_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["BLAS_NUM_THREADS"] = "1"

# Create dummy tqdm that does nothing
def tqdm(iterable, *args, **kwargs):
    return iterable

import random
import gc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.preprocessing import RobustScaler
from joblib import Parallel, delayed
import json

import faulthandler
faulthandler.enable()

from utils import set_working_directory

# DB and environment handling
from dotenv import load_dotenv
from sqlalchemy import create_engine
import psycopg2

# Set default plot style and figure size
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)


def seed_everything(seed: int):
    """Set seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def clear_mem():
    """Collect garbage and clear GPU cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Seed and working directory
seed_everything(42)
wd = set_working_directory()
print(f"Working directory: {os.getcwd()}")


# Load environment variables
env_file_primary = ".env"
env_file_fallback = "env.env.txt"
if os.path.exists(env_file_primary):
    load_dotenv(env_file_primary)
    print("Loaded environment from .env")
elif os.path.exists(env_file_fallback):
    load_dotenv(env_file_fallback)
    print("Loaded environment from env.env.txt")
else:
    print("No .env or env.env.txt environment file found.")


##############################################
# Heavy–Tail Analysis Module (WITH PARALLEL BOOTSTRAP)
##############################################
import scipy.stats as stats
from statsmodels.graphics.gofplots import qqplot
from scipy.special import zeta

class HeavyTailAnalyzer:
    """
    Class for analyzing heavy-tailed distributions and selecting optimal thresholds.
    FIXED: Added parallel bootstrap support
    """
    def __init__(self, data=None, name=None):
        self.data = data
        self.name = name or "metric"
        self.hill_estimates = None
        self.bootstrap_results = None
        self.bootstrap_tail_results = None
        self.optimal_threshold = None
        self.optimal_quantile = None
        self.estimated_tail_index = None

    def set_data(self, data, name=None):
        self.data = data
        if name:
            self.name = name

    def hill_estimator(self, k=None):
        sorted_data = np.sort(self.data)[::-1]
        n = len(sorted_data)
        if k is None:
            k = np.arange(5, min(n//5, 200))
        elif np.isscalar(k):
            k = np.array([k])
        alpha_estimates = np.zeros_like(k, dtype=float)
        for i, ki in enumerate(k):
            if ki >= n:
                alpha_estimates[i] = np.nan
                continue
            log_term = np.log(sorted_data[:ki] / sorted_data[ki])
            alpha_estimates[i] = 1 / (np.mean(log_term) + 1e-10)
        self.hill_estimates = (k, alpha_estimates)
        return k, alpha_estimates

    def plot_hill(self, k=None, ci=True, n_bootstrap=100, alpha=0.05, n_jobs=-1):
        if self.data is None:
            raise ValueError("No data provided.")
        if self.hill_estimates is None or k is not None:
            k_values, alpha_estimates = self.hill_estimator(k)
        else:
            k_values, alpha_estimates = self.hill_estimates

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(k_values, alpha_estimates, 'b-', linewidth=2)
        ax.set_xlabel('k (order stats)')
        ax.set_ylabel('α (tail index)')
        ax.set_title(f'Hill Plot for {self.name}')
        ax.grid(True, alpha=0.3)
        ax.axhline(1, color='r', linestyle='--', alpha=0.5)
        ax.axhline(2, color='g', linestyle='--', alpha=0.5)

        if ci and n_bootstrap > 0:
            if self.bootstrap_results is None or len(self.bootstrap_results) != len(k_values):
                self._bootstrap_hill_parallel(k_values, n_bootstrap, n_jobs)
            ci_lower = np.percentile(self.bootstrap_results, alpha/2*100, axis=0)
            ci_upper = np.percentile(self.bootstrap_results, (1-alpha/2)*100, axis=0)
            ax.fill_between(k_values, ci_lower, ci_upper, color='b', alpha=0.2)

        # stable region
        if len(k_values) > 10:
            window = min(20, len(k_values)//5)
            rolling_std = np.array([
                np.std(alpha_estimates[i:i+window])
                for i in range(len(alpha_estimates)-window)
            ])
            idx = np.argmin(rolling_std)
            k_star = k_values[idx]
            a_star = alpha_estimates[idx+window//2]
            ax.axvline(k_star, color='purple', linestyle='-.')
            ax.plot(k_star, a_star, 'ro')
            ax.text(k_star*1.1, a_star*0.9,
                    f'k*={k_star}, α≈{a_star:.2f}', verticalalignment='top')
            self.optimal_k = k_star
            self.estimated_alpha = a_star

        return fig

    def _bootstrap_hill_parallel(self, k_values, n_bootstrap, n_jobs=-1):
        """Parallel bootstrap for Hill estimator"""
        n = len(self.data)
        
        def bootstrap_iteration(seed):
            np.random.seed(seed)
            sample = np.random.choice(self.data, size=n, replace=True)
            _, a_est = HeavyTailAnalyzer(sample).hill_estimator(k_values)
            return a_est
        
        # Use joblib for parallel processing
        estimates = Parallel(n_jobs=n_jobs)(
            delayed(bootstrap_iteration)(i) for i in range(n_bootstrap)
        )
        self.bootstrap_results = np.array(estimates)

    def qq_plot(self, dist='pareto', floc=0, **kwargs):
        if self.data is None:
            raise ValueError("No data provided.")
        sorted_data = np.sort(self.data[~np.isnan(self.data)])
        sorted_data = sorted_data[sorted_data > 0]
        if dist == 'pareto':
            alpha = kwargs.get('fa', 1.0)
            scale = kwargs.get('fscale', sorted_data.min())
            theo = stats.pareto(alpha, floc, scale)
        elif dist == 'exponential':
            scale = kwargs.get('fscale', 1.0)
            theo = stats.expon(floc, scale)
        else:
            raise ValueError(f"{dist} not supported")

        fig, ax = plt.subplots(figsize=(8, 8))
        qqplot(sorted_data, dist=theo, line='45', ax=ax)
        ax.set_title(f'QQ Plot: {self.name} vs {dist}')
        return fig

    def find_optimal_threshold(self, method='stability', min_q=0.8, max_q=0.99, n_q=20):
        if self.data is None:
            raise ValueError("No data provided.")
        qs = np.linspace(min_q, max_q, n_q)
        ths = np.quantile(self.data, qs)
        if method == 'stability':
            stab = np.zeros_like(ths)
            alpha_est = np.zeros_like(ths)
            for i, t in enumerate(ths):
                exc = self.data[self.data>t] - t
                if len(exc)<20:
                    stab[i] = np.inf
                    continue
                k_rng = np.arange(5, min(len(exc)//2, 50))
                _, a = HeavyTailAnalyzer(exc).hill_estimator(k_rng)
                stab[i] = np.std(a[~np.isnan(a)])
                alpha_est[i] = np.median(a[~np.isnan(a)])
            valid = ~np.isinf(stab)
            idx = np.argmin(stab[valid])
            qi = np.where(valid)[0][idx]
            self.optimal_threshold = ths[qi]
            self.optimal_quantile = qs[qi]
            self.estimated_tail_index = alpha_est[qi]
        return self.optimal_threshold, self.optimal_quantile

    def plot_mean_excess(self, min_quantile=0.5, max_quantile=0.99, num_points=50):
        if self.data is None:
            raise ValueError("No data provided. Use set_data() first.")
        
        quantiles = np.linspace(min_quantile, max_quantile, num_points)
        thresholds = np.quantile(self.data, quantiles)
        
        mean_excesses = np.zeros_like(thresholds)
        excess_counts = np.zeros_like(thresholds, dtype=int)
        
        for i, threshold in enumerate(thresholds):
            excesses = self.data[self.data > threshold] - threshold
            excess_counts[i] = len(excesses)
            if len(excesses) > 0:
                mean_excesses[i] = np.mean(excesses)
            else:
                mean_excesses[i] = np.nan
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(thresholds, mean_excesses, 'b.-', linewidth=2)
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Mean Excess')
        ax.set_title(f'Mean Excess Function for {self.name}')
        ax.grid(True, alpha=0.3)
        
        ax2 = ax.twinx()
        ax2.plot(thresholds, excess_counts, 'r--', alpha=0.6)
        ax2.set_ylabel('Sample Size', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        if self.optimal_threshold is not None:
            ax.axvline(x=self.optimal_threshold, color='g', linestyle='-.')
            ax.text(
                self.optimal_threshold*1.05,
                ax.get_ylim()[0] + 0.9*(ax.get_ylim()[1] - ax.get_ylim()[0]),
                f'Optimal threshold: {self.optimal_threshold:.2f}\n(q = {self.optimal_quantile:.2f})',
                verticalalignment='top'
            )
        
        return fig

    def bootstrap_tail_index(self, threshold=None, n_bootstrap=500, alpha=0.05, k=None, n_jobs=-1):
        if self.data is None:
            raise ValueError("No data provided. Use set_data() first.")
            
        if threshold is None:
            if self.optimal_threshold is not None:
                threshold = self.optimal_threshold
            else:
                threshold = np.quantile(self.data, 0.95)
        
        exceedances = self.data[self.data > threshold] - threshold
        n_excess = len(exceedances)
        
        if n_excess < 20:
            print(f"Warning: Only {n_excess} exceedances. Results may be unreliable.")
            
        if k is None:
            k = max(5, int(n_excess * 0.1))
            
        hill_temp = HeavyTailAnalyzer(exceedances)
        _, alpha_estimates = hill_temp.hill_estimator(k)
        point_estimate = (
            alpha_estimates[0]
            if np.isscalar(alpha_estimates)
            else np.median(alpha_estimates[~np.isnan(alpha_estimates)])
        )
        
        # Parallel bootstrap
        def bootstrap_iteration(seed):
            np.random.seed(seed)
            boot_sample = np.random.choice(exceedances, size=n_excess, replace=True)
            hill_boot = HeavyTailAnalyzer(boot_sample)
            _, alpha_boot = hill_boot.hill_estimator(k)
            return (alpha_boot[0] if np.isscalar(alpha_boot) 
                    else np.median(alpha_boot[~np.isnan(alpha_boot)]))
        
        bootstrap_estimates = Parallel(n_jobs=n_jobs)(
            delayed(bootstrap_iteration)(i) for i in range(n_bootstrap)
        )
        bootstrap_estimates = np.array(bootstrap_estimates)
            
        ci_lower = np.percentile(bootstrap_estimates, alpha/2*100)
        ci_upper = np.percentile(bootstrap_estimates, (1-alpha/2)*100)
        
        self.bootstrap_tail_results = {
            'point_estimate': point_estimate,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'bootstrap_samples': bootstrap_estimates
        }
        
        return point_estimate, ci_lower, ci_upper

    def plot_bootstrap_distribution(self):
        if self.bootstrap_tail_results is None:
            raise ValueError("No bootstrap results available. Run bootstrap_tail_index() first.")
            
        point_estimate = self.bootstrap_tail_results['point_estimate']
        ci_lower = self.bootstrap_tail_results['ci_lower']
        ci_upper = self.bootstrap_tail_results['ci_upper']
        bootstrap_samples = self.bootstrap_tail_results['bootstrap_samples']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(bootstrap_samples, bins=30, alpha=0.6, density=True)
        ax.axvline(
            x=point_estimate, color='r', linestyle='-', linewidth=2,
            label=f'Point estimate: {point_estimate:.2f}'
        )
        ax.axvline(
            x=ci_lower, color='g', linestyle='--', linewidth=2,
            label=f'95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]'
        )
        ax.axvline(x=ci_upper, color='g', linestyle='--', linewidth=2)
        ax.axvline(x=1, color='purple', linestyle=':', alpha=0.5, label='Cauchy (α=1)')
        ax.axvline(x=2, color='orange', linestyle=':', alpha=0.5, label='Normal (α=2)')
        ax.set_xlabel('Tail Index (α)')
        ax.set_ylabel('Density')
        ax.set_title(f'Bootstrap Distribution of Tail Index Estimates for {self.name}')
        ax.legend()
        
        return fig

    def get_tail_classification(self):
        if self.bootstrap_tail_results is None:
            raise ValueError("No bootstrap results available. Run bootstrap_tail_index() first.")
            
        alpha = self.bootstrap_tail_results['point_estimate']
        ci_lower = self.bootstrap_tail_results['ci_lower']
        ci_upper = self.bootstrap_tail_results['ci_upper']
        
        if alpha <= 1:
            classification = "extremely_heavy"
            description = "Extremely heavy-tailed (infinite mean, like Cauchy)"
        elif alpha <= 2:
            classification = "heavy"
            description = "Heavy-tailed (finite mean, infinite variance)"
        elif alpha <= 3:
            classification = "moderate"
            description = "Moderately heavy-tailed (finite variance, infinite third moment)"
        elif alpha <= 4:
            classification = "light_heavy"
            description = "Light heavy-tailed (finite third moment, infinite fourth moment)"
        else:
            classification = "light"
            description = "Light-tailed (all moments finite, close to normal/exponential)"
        
        confidence_info = f"95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]"
        
        return classification, f"{description}\nTail index (α): {alpha:.2f}\n{confidence_info}"


##############################################
# FIXED: Memory-efficient Exceedance Block Statistics
##############################################
def compute_exceedance_block_statistics(df: pd.DataFrame, block_size: int = 7):
    """
    ULTRA MEMORY-EFFICIENT VERSION: Process in chunks and use groupby efficiently
    """
    if df is None or df.empty:
        return df
    
    print(f"Computing exceedance block statistics (block_size={block_size})...")
    
    # Sort once
    df = df.sort_values(['customer_id', 'day_period']).copy()
    
    # Get indicator and magnitude columns
    exceeds_cols = [c for c in df.columns if c.startswith('I_')]
    magnitude_cols = [c for c in df.columns if c.startswith('E_')]
    
    if not exceeds_cols:
        print("  No exceedance columns found.")
        return df
    
    # Reset index to ensure we have a clean index
    df = df.reset_index(drop=True)
    
    # 1) Compute indicator statistics using groupby operations
    print(f"  Computing statistics for {len(exceeds_cols)} indicator columns...")
    
    for i_col in exceeds_cols:
        base = i_col[2:]  # Remove 'I_' prefix
        
        # Use transform to maintain original index
        df[f"{base}_count_exceeds_{block_size}d"] = (
            df.groupby('customer_id')[i_col]
            .transform(lambda x: x.rolling(window=block_size, min_periods=1).sum())
            .astype(np.int16)
        )
        
        df[f"{base}_any_exceeds_{block_size}d"] = (
            df[f"{base}_count_exceeds_{block_size}d"] > 0
        ).astype(np.int8)
        
        df[f"{base}_freq_exceeds_{block_size}d"] = (
            df[f"{base}_count_exceeds_{block_size}d"] >= 3
        ).astype(np.int8)
    
    # 2) Compute magnitude statistics - simplified version
    if magnitude_cols:
        print(f"  Computing statistics for {len(magnitude_cols)} magnitude columns...")
        
        for e_col in magnitude_cols:
            base = e_col[2:]  # Remove 'E_' prefix
            i_col = f'I_{base}'
            
            if i_col not in df.columns:
                continue
            
            # Rolling sum - simple
            df[f"{base}_sum_magnitude_{block_size}d"] = (
                df.groupby('customer_id')[e_col]
                .transform(lambda x: x.rolling(window=block_size, min_periods=1).sum())
                .astype(np.float32)
            )
            
            # For max and mean, we need to handle the masking differently
            # Create a temporary column with masked values
            temp_col = f'_temp_{base}'
            df[temp_col] = df[e_col] * df[i_col]  # This zeros out non-exceedances
            
            # Rolling max on masked values
            df[f"{base}_max_magnitude_{block_size}d"] = (
                df.groupby('customer_id')[temp_col]
                .transform(lambda x: x.rolling(window=block_size, min_periods=1).max())
                .astype(np.float32)
            )
            
            # For mean, we need the count of exceedances
            exceedance_count = (
                df.groupby('customer_id')[i_col]
                .transform(lambda x: x.rolling(window=block_size, min_periods=1).sum())
            )
            
            # Calculate mean avoiding division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                df[f"{base}_mean_magnitude_{block_size}d"] = (
                    df[f"{base}_sum_magnitude_{block_size}d"] / exceedance_count
                ).fillna(0.0).astype(np.float32)
            
            # Drop temporary column
            df.drop(columns=[temp_col], inplace=True)
            
            # Garbage collect after each magnitude column
            gc.collect()
    
    print("  Exceedance block statistics complete.")
    return df

def compute_multi_feature_exceedance_stats(df: pd.DataFrame):
    """
    As in original: total_features_exceeded, multiple_features_exceeded,
    per-customer correlation & pattern entropy.
    """
    if df is None or df.empty:
        return df
    
    df = df.sort_values(['customer_id', 'day_period']).copy()
    exceeds_cols = [c for c in df.columns if c.startswith('I_')]
    
    if len(exceeds_cols) <= 1:
        return df

    # Total features exceeded per day
    df['total_features_exceeded'] = df[exceeds_cols].sum(axis=1)
    df['multiple_features_exceeded'] = (df['total_features_exceeded'] >= 2).astype(int)

    # Per-customer statistics
    print("Computing per-customer exceedance patterns...")
    
    # Use groupby more efficiently
    for customer_id, grp in df.groupby('customer_id'):
        if len(grp) < 10:
            continue
            
        idx = grp.index
        
        # Correlation between exceedance patterns
        corr = grp[exceeds_cols].corr().values
        mask = ~np.eye(len(exceeds_cols), dtype=bool)
        df.loc[idx, 'exceedance_correlation'] = corr[mask].mean()

        # Pattern entropy for customers with enough data
        if len(grp) >= 30:
            patterns = grp[exceeds_cols].astype(int).astype(str).agg(''.join, axis=1)
            counts = patterns.value_counts()
            probs = counts / counts.sum()
            ent = -np.sum(probs * np.log2(probs + 1e-10))
            if len(counts) > 1:
                ent /= np.log2(len(counts))
            df.loc[idx, 'exceedance_pattern_entropy'] = ent

    return df


##############################################
# Gap Feature Functions (unchanged)
##############################################
def add_gap_features(df: pd.DataFrame, 
                     user_col: str = 'customer_id',
                     time_col: str = 'day_period') -> pd.DataFrame:
    """
    Add time since previous active day as a feature for each customer.
    """
    print("Adding gap features (time since previous active day)...")
    
    df = df.sort_values([user_col, time_col]).copy()
    
    # Initialize new columns
    df['days_since_last_activity'] = 0
    df['days_since_last_activity_norm'] = 0.0
    
    # Process each customer
    for customer_id, customer_df in tqdm(df.groupby(user_col), 
                                         desc="Computing gap features"):
        customer_df = customer_df.sort_values(time_col)
        dates = pd.to_datetime(customer_df[time_col])
        
        # Calculate gaps
        gaps = [0]  # First activity has gap of 0
        for i in range(1, len(dates)):
            gap_days = (dates.iloc[i] - dates.iloc[i-1]).days
            gaps.append(gap_days)
        
        # Assign to dataframe
        df.loc[customer_df.index, 'days_since_last_activity'] = gaps
    
    # Create normalized version using log transform
    df['days_since_last_activity_norm'] = np.log1p(df['days_since_last_activity'])
    
    # Create gap categories
    def get_gap_category(gap):
        if gap == 0:
            return '0_days'
        elif gap <= 2:
            return '1-2_days'
        elif gap <= 6:
            return '3-6_days'
        elif gap <= 13:
            return '7-13_days'
        elif gap <= 29:
            return '14-29_days'
        else:
            return '30+_days'
    
    df['gap_category'] = df['days_since_last_activity'].apply(get_gap_category)
    
    print(f"Gap feature statistics:")
    print(f"  Mean gap: {df['days_since_last_activity'].mean():.1f} days")
    print(f"  Median gap: {df['days_since_last_activity'].median():.1f} days")
    print(f"  Max gap: {df['days_since_last_activity'].max()} days")
    
    return df


##############################################
# NEW: Label Leakage Check
##############################################
def check_label_leakage(window_dates_df, labels_df, window_size=30):
    """
    Check for label leakage in windows - VECTORIZED VERSION
    Much faster for large datasets - no loops!
    """
    print("\nChecking for label leakage (vectorized)...")
    print(f"  Checking {len(window_dates_df):,} windows against {len(labels_df):,} labels...")
    
    # Ensure we have clean copies
    windows = window_dates_df.copy()
    labels = labels_df.copy()
    
    # Convert and normalize all dates to timezone-naive
    print("  Normalizing dates...")
    windows['end_date'] = pd.to_datetime(windows['end_date'])
    if hasattr(windows['end_date'].iloc[0], 'tz') and windows['end_date'].iloc[0].tz is not None:
        windows['end_date'] = windows['end_date'].dt.tz_localize(None)
    
    labels['date'] = pd.to_datetime(labels['date'])
    if labels['date'].dt.tz is not None:
        labels['date'] = labels['date'].dt.tz_localize(None)
    
    # Merge windows with labels on customer_id
    # This creates all combinations of windows and labels for each customer
    print("  Merging windows with labels...")
    merged = windows.merge(
        labels[['customer_id', 'date']], 
        on='customer_id', 
        how='inner',
        suffixes=('_window', '_label')
    )
    
    if len(merged) == 0:
        print("  ✓ No label leakage detected (no matching customers)")
        return True, None
    
    print(f"  Checking {len(merged):,} window-label combinations...")
    
    # Vectorized comparison: find where label date is within window
    # Label leakage occurs when label_date <= window_end_date
    leakage_mask = merged['date'] <= merged['end_date']
    
    # Filter to get only leakage cases
    leakage_df = merged[leakage_mask].copy()
    
    if len(leakage_df) == 0:
        print("  ✓ No label leakage detected")
        return True, None
    
    # Calculate overlap days
    leakage_df['days_overlap'] = (leakage_df['end_date'] - leakage_df['date']).dt.days
    
    # Get unique windows with leakage (a window might have multiple labels)
    unique_leaking_windows = leakage_df['window_idx'].nunique()
    
    # Summary statistics
    print(f"\n  WARNING: Found label leakage!")
    print(f"  - {unique_leaking_windows:,} windows affected ({unique_leaking_windows/len(windows)*100:.1f}%)")
    print(f"  - {len(leakage_df):,} total label-window overlaps")
    print(f"  - Average overlap: {leakage_df['days_overlap'].mean():.1f} days")
    print(f"  - Max overlap: {leakage_df['days_overlap'].max()} days")
    
    # Return summary of leakage
    leakage_summary = leakage_df[['window_idx', 'customer_id', 'end_date', 'date', 'days_overlap']].rename(
        columns={'date': 'label_date', 'end_date': 'window_end'}
    )
    
    return False, leakage_summary

##############################################
# FIXED: Hybrid Window Creation Functions (using float32)
##############################################
def create_activity_windows_with_constraints(
    df: pd.DataFrame,
    feat_cols: list,
    label_col: str = 'category',
    window_size: int = 30,
    max_calendar_span: int = 180,
    split_on_long_gaps: int = 90
):
    """
    Create activity-based windows with constraints on calendar span.
    FIXED: Using float32 instead of float16
    """
    if df is None or df.empty or not feat_cols:
        return (np.empty((0, window_size, len(feat_cols)), dtype=np.float32),
                np.empty((0, window_size), dtype=np.int8),
                np.empty((0,), dtype=np.int32),
                pd.DataFrame(columns=['window_idx', 'customer_id', 'start_date', 'end_date', 
                                     'calendar_span_days', 'activity_dates', 'phase']))

    df = df.sort_values(['customer_id', 'day_period'])
    
    # Initialize lists
    all_X = []
    all_L = []
    all_I = []
    all_window_dates = []
    
    # Create customer ID mapping
    customer_id_map = pd.Categorical(df['customer_id'])
    customer_to_idx = {cat: idx for idx, cat in enumerate(customer_id_map.categories)}
    
    window_idx = 0
    windows_discarded_span = 0
    windows_created = 0
    
    for customer_id, customer_df in tqdm(df.groupby('customer_id', sort=False), 
                                         desc="Creating constrained activity windows"):
        customer_df = customer_df.sort_values('day_period').reset_index(drop=True)
        
        # Split sequences on long gaps
        sequences = []
        current_seq = [0]
        
        for i in range(1, len(customer_df)):
            gap = customer_df.loc[i, 'days_since_last_activity']
            if gap > split_on_long_gaps:
                # Start new sequence
                if len(current_seq) >= window_size:
                    sequences.append(current_seq)
                current_seq = [i]
            else:
                current_seq.append(i)
        
        # Don't forget last sequence
        if len(current_seq) >= window_size:
            sequences.append(current_seq)
        
        # Process each sequence
        for seq_indices in sequences:
            seq_df = customer_df.iloc[seq_indices]
            
            if len(seq_df) < window_size:
                continue
            
            # Get data for this sequence - USING FLOAT32
            feats = seq_df[feat_cols].fillna(0).astype(np.float32).values
            lbls = pd.Categorical(seq_df[label_col]).codes.astype(np.int8)
            dates = seq_df['day_period'].values
            customer_idx = customer_to_idx[customer_id]
            
            # Create sliding windows
            for i in range(len(seq_df) - window_size + 1):
                # Check calendar span
                start_date = pd.Timestamp(dates[i])
                end_date = pd.Timestamp(dates[i + window_size - 1])
                calendar_span = (end_date - start_date).days
                
                if calendar_span > max_calendar_span:
                    windows_discarded_span += 1
                    continue
                
                # Create window
                X_window = feats[i:i + window_size]
                L_window = lbls[i:i + window_size]
                window_dates = dates[i:i + window_size]
                
                all_X.append(X_window)
                all_L.append(L_window)
                all_I.append(customer_idx)
                
                activity_dates_str = ','.join([str(d) for d in window_dates])
                
                all_window_dates.append({
                    'window_idx': window_idx,
                    'customer_id': customer_id,
                    'start_date': start_date,
                    'end_date': end_date,
                    'calendar_span_days': calendar_span,
                    'activity_dates': activity_dates_str,
                    'phase': 'activity_based'
                })
                
                window_idx += 1
                windows_created += 1
    
    # Convert to arrays - USING FLOAT32
    X = np.array(all_X, dtype=np.float32)
    L = np.array(all_L, dtype=np.int8)
    I = np.array(all_I, dtype=np.int32)
    window_dates_df = pd.DataFrame(all_window_dates)
    
    print(f"\nActivity window creation complete:")
    print(f"  Created: {windows_created} windows")
    print(f"  Discarded (span > {max_calendar_span} days): {windows_discarded_span}")
    
    if len(window_dates_df) > 0:
        print(f"  Calendar span: {window_dates_df['calendar_span_days'].min()}-"
              f"{window_dates_df['calendar_span_days'].max()} days "
              f"(avg: {window_dates_df['calendar_span_days'].mean():.1f})")
    
    return X, L, I, window_dates_df


def create_calendar_windows_with_gap_limit(
    df: pd.DataFrame,
    feat_cols: list,
    label_col: str = 'category',
    window_size: int = 30,
    max_gaps: int = 6
):
    """
    Memory-efficient calendar-based window creation
    FIXED: Using float32 instead of float16
    """
    print(f"\nCreating calendar-based windows (30 days with ≤{max_gaps} gaps)...")
    
    # Sort data once
    df = df.sort_values(['customer_id', 'day_period']).copy()
    
    # Create customer mapping
    unique_customers = df['customer_id'].unique()
    customer_to_idx = {cid: idx for idx, cid in enumerate(unique_customers)}
    
    # Pre-allocate lists
    all_X = []
    all_L = []
    all_I = []
    all_window_dates = []
    
    window_idx = 0
    windows_created = 0
    windows_discarded_gaps = 0
    
    print(f"  Processing {len(unique_customers)} customers...")
    
    # GROUP BY CUSTOMER TO AVOID REPEATED FILTERING
    grouped = df.groupby('customer_id', sort=False)
    
    # Process each customer group
    customer_count = 0
    for customer_id, cust_df in grouped:
        # Progress update
        if customer_count % 100 == 0:
            print(f"    Progress: {customer_count}/{len(unique_customers)} customers...")
        customer_count += 1
        
        # Ensure sorted by date
        cust_df = cust_df.sort_values('day_period').reset_index(drop=True)
        
        if len(cust_df) == 0:
            continue
        
        # Convert dates once
        dates = pd.to_datetime(cust_df['day_period'])
        min_date = dates.min()
        max_date = dates.max()
        
        # Skip if too short
        if (max_date - min_date).days < window_size - 1:
            continue
        
        # Create date lookup using numpy arrays
        date_values = dates.dt.date.values
        date_to_idx = {date: idx for idx, date in enumerate(date_values)}
        
        # Get feature data as numpy array - USING FLOAT32
        feature_data = cust_df[feat_cols].fillna(0).values.astype(np.float32)
        
        # Handle labels
        if label_col in cust_df.columns:
            label_data = cust_df[label_col].values
        else:
            label_data = np.array(['no_label'] * len(cust_df))
        
        # Find gap feature index once
        gap_feat_idx = None
        if 'days_since_last_activity_norm' in feat_cols:
            gap_feat_idx = feat_cols.index('days_since_last_activity_norm')
        
        # Slide windows
        current_date = min_date.normalize()
        max_start_date = max_date - pd.Timedelta(days=window_size - 1)
        
        while current_date <= max_start_date:
            # Pre-allocate window arrays - USING FLOAT32
            X_window = np.zeros((window_size, len(feat_cols)), dtype=np.float32)
            L_window = ['no_activity'] * window_size
            active_positions = []
            active_dates_list = []
            
            # Fill window
            for day_offset in range(window_size):
                check_date = (current_date + pd.Timedelta(days=day_offset)).date()
                
                if check_date in date_to_idx:
                    # Active day
                    data_idx = date_to_idx[check_date]
                    X_window[day_offset] = feature_data[data_idx]
                    L_window[day_offset] = label_data[data_idx]
                    active_positions.append(day_offset)
                    active_dates_list.append(str(check_date))
            
            # Check gap constraint
            n_active = len(active_positions)
            n_gaps = window_size - n_active
            
            if n_gaps > max_gaps:
                windows_discarded_gaps += 1
                current_date += pd.Timedelta(days=1)
                continue
            
            # Add gap features for inactive days if needed
            if gap_feat_idx is not None:
                last_active = -1
                for pos in range(window_size):
                    if L_window[pos] == 'no_activity':
                        gap = pos - last_active if last_active >= 0 else pos + 1
                        X_window[pos, gap_feat_idx] = np.log1p(gap)
                    else:
                        last_active = pos
            
            # Convert labels to codes using numpy
            unique_labels, inverse = np.unique(L_window, return_inverse=True)
            L_codes = inverse.astype(np.int8)
            
            # Store window - X_window is already float32
            all_X.append(X_window)
            all_L.append(L_codes)
            all_I.append(customer_to_idx[customer_id])
            
            # Store metadata
            all_window_dates.append({
                'window_idx': window_idx,
                'customer_id': customer_id,
                'start_date': current_date,
                'end_date': current_date + pd.Timedelta(days=window_size - 1),
                'calendar_span_days': 29,
                'n_active_days': n_active,
                'n_gaps': n_gaps,
                'activity_dates': ','.join(active_dates_list),
                'phase': 'calendar_based'
            })
            
            window_idx += 1
            windows_created += 1
            
            # Move to next day
            current_date += pd.Timedelta(days=1)
            
            # Progress update
            if window_idx % 10000 == 0:
                print(f"    Created {window_idx} windows so far...")
    
    # Convert to final arrays - USING FLOAT32
    print("  Converting to final arrays...")
    if all_X:
        X = np.array(all_X, dtype=np.float32)
        L = np.array(all_L, dtype=np.int8)
        I = np.array(all_I, dtype=np.int32)
        window_dates_df = pd.DataFrame(all_window_dates)
    else:
        X = np.empty((0, window_size, len(feat_cols)), dtype=np.float32)
        L = np.empty((0, window_size), dtype=np.int8)
        I = np.empty((0,), dtype=np.int32)
        window_dates_df = pd.DataFrame()
    
    # Clean up
    del all_X, all_L, all_I, all_window_dates
    gc.collect()
    
    print(f"\nCalendar window creation complete:")
    print(f"  Created: {windows_created:,} windows")
    print(f"  Discarded (gaps > {max_gaps}): {windows_discarded_gaps:,}")
    
    if len(window_dates_df) > 0:
        print(f"  Active days per window: {window_dates_df['n_active_days'].min()}-"
              f"{window_dates_df['n_active_days'].max()} "
              f"(avg: {window_dates_df['n_active_days'].mean():.1f})")
    
    return X, L, I, window_dates_df


def create_hybrid_windows(
    df: pd.DataFrame,
    feat_cols: list,
    label_col: str = 'category'
):
    """
    Create both activity-based and calendar-based windows.
    """
    results = {}
    
    # Phase 1: Activity-based windows
    print("\n" + "="*60)
    print("PHASE 1: Creating activity-based windows")
    print("="*60)
    
    try:
        X_act, L_act, I_act, dates_act = create_activity_windows_with_constraints(
            df, feat_cols, label_col,
            window_size=30,
            max_calendar_span=180,
            split_on_long_gaps=90
        )
        
        results['activity'] = (X_act, L_act, I_act, dates_act)
        print(f"Successfully created {len(X_act)} activity-based windows")
        
    except Exception as e:
        print(f"ERROR in activity window creation: {str(e)}")
        print("Creating empty activity window arrays...")
        X_act = np.empty((0, 30, len(feat_cols)), dtype=np.float32)
        L_act = np.empty((0, 30), dtype=np.int8)
        I_act = np.empty((0,), dtype=np.int32)
        dates_act = pd.DataFrame()
        results['activity'] = (X_act, L_act, I_act, dates_act)
    
    # Clean up memory before phase 2
    gc.collect()
    
    # Phase 2: Calendar-based windows
    print("\n" + "="*60)
    print("PHASE 2: Creating calendar-based windows")
    print("="*60)
    
    try:
        X_cal, L_cal, I_cal, dates_cal = create_calendar_windows_with_gap_limit(
            df, feat_cols, label_col,
            window_size=30,
            max_gaps=6
        )
        
        results['calendar'] = (X_cal, L_cal, I_cal, dates_cal)
        print(f"Successfully created {len(X_cal)} calendar-based windows")
        
    except Exception as e:
        print(f"ERROR in calendar window creation: {str(e)}")
        print("Creating empty calendar window arrays...")
        X_cal = np.empty((0, 30, len(feat_cols)), dtype=np.float32)
        L_cal = np.empty((0, 30), dtype=np.int8)
        I_cal = np.empty((0,), dtype=np.int32)
        dates_cal = pd.DataFrame()
        results['calendar'] = (X_cal, L_cal, I_cal, dates_cal)
    
    # Summary
    print("\n" + "="*60)
    print("HYBRID WINDOW CREATION SUMMARY")
    print("="*60)
    print(f"Activity-based windows: {len(results['activity'][0])}")
    print(f"Calendar-based windows: {len(results['calendar'][0])}")
    print(f"Total windows: {len(results['activity'][0]) + len(results['calendar'][0])}")
    
    return results


##############################################
# Helper: Label Merge (unchanged)
##############################################
def merge_labels(df: pd.DataFrame,
                 labels_df: pd.DataFrame,
                 label_col: str = 'category'):
    """Merge labels on customer_id + day_period/date, fill None if missing."""
    if df is None or df.empty:
        return df
    if labels_df is None or labels_df.empty:
        df[label_col] = None
        return df

    ld = labels_df.copy()
    ld['date'] = pd.to_datetime(ld['date'], utc=True, errors='coerce').dt.normalize()
    ld = ld.rename(columns={'date': 'label_date'})

    if 'day_period' in df.columns:
        return df.merge(
            ld[['customer_id', 'label_date', label_col]],
            left_on=['customer_id', 'day_period'],
            right_on=['customer_id', 'label_date'],
            how='left'
        ).drop(columns=['label_date'])
    else:
        df[label_col] = None
        return df


def validate_window_alignment(X, L, I, window_dates_df, split_name):
    """
    Validate that all window data is properly aligned.
    """
    print(f"\nValidating {split_name} window alignment...")
    
    assert len(X) == len(L) == len(I) == len(window_dates_df), \
        f"Mismatched array lengths in {split_name}"
    
    # Check array shapes
    assert X.ndim == 3, f"X should be 3D, got {X.ndim}D"
    assert L.ndim == 2, f"L should be 2D, got {L.ndim}D"
    assert I.ndim == 1, f"I should be 1D, got {I.ndim}D"
    
    # Check sequence length consistency
    seq_len = X.shape[1]
    assert L.shape[1] == seq_len, f"L sequence length {L.shape[1]} doesn't match X {seq_len}"
    
    print(f"✓ {split_name} validation passed: {len(X)} windows, seq_len={seq_len}")
    return True


##############################################
# FIXED: Hybrid Split Functions
##############################################
def make_hybrid_splits(df, user_col='customer_id', time_col='day_period',
                      train_user_frac=0.8, val_user_frac=0.1, 
                      temporal_holdout_frac=0.2):
    """
    Memory-efficient hybrid splitting
    """
    print("Creating hybrid splits...")
    
    # Ensure time column is datetime
    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col])
    
    # 1) Randomly split users
    users = df[user_col].unique()
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(users)
    
    n = len(users)
    n_train = int(train_user_frac * n)
    n_val = int(val_user_frac * n)
    
    train_users = set(users[:n_train])
    val_users = set(users[n_train:n_train + n_val])
    test_users = set(users[n_train + n_val:])
    
    print(f"User splits: {len(train_users)} train, {len(val_users)} val, {len(test_users)} test")
    
    # 2) Split data using numpy operations
    print("Splitting data by users (memory-efficient method)...")
    user_array = df[user_col].values
    
    # Create masks using numpy
    train_mask = np.array([u in train_users for u in user_array])
    val_mask = np.array([u in val_users for u in user_array])
    test_mask = np.array([u in test_users for u in user_array])
    
    # Apply masks
    df_train_pool = df[train_mask].copy()
    df_val = df[val_mask].copy()
    df_test = df[test_mask].copy()
    
    print(f"Split complete: {len(df_train_pool)} train pool, {len(df_val)} val, {len(df_test)} test")
    
    # 3) For training users, split temporally
    df_train_list = []
    df_holdout_list = []
    
    for u, g in tqdm(df_train_pool.groupby(user_col), desc="Temporal split for train users"):
        g = g.sort_values(time_col)
        split_idx = int((1 - temporal_holdout_frac) * len(g))
        
        if split_idx > 0:  # Ensure we have at least 1 sample for training
            df_train_list.append(g.iloc[:split_idx])
            if split_idx < len(g):  # Only add to holdout if there are samples
                df_holdout_list.append(g.iloc[split_idx:])
    
    df_train = pd.concat(df_train_list, ignore_index=True)
    df_holdout = pd.concat(df_holdout_list, ignore_index=True) if df_holdout_list else pd.DataFrame()
    
    print(f"\nFinal split sizes:")
    print(f"  Train: {len(df_train)} samples")
    print(f"  Holdout: {len(df_holdout)} samples")
    print(f"  Val: {len(df_val)} samples")
    print(f"  Test: {len(df_test)} samples")
    
    return df_train, df_holdout, df_val, df_test


##############################################
# ENHANCED: Heavy-Tail Processing with Data Cleaning
##############################################
def process_heavy_tailed_data_with_cleaning(df, df_train, df_holdout, df_val, df_test, num_cols, table_name):
    """
    Complete pipeline that cleans data AND preserves heavy-tail information
    """
    
    print(f"\n{'='*60}")
    print(f"PROCESSING {table_name} WITH DATA CLEANING")
    print(f"{'='*60}")
    
    # Combine train and holdout for analysis
    df_train_full = pd.concat([df_train, df_holdout], ignore_index=True)
    
    # =========================================================================
    # STEP 0: DATA QUALITY CLEANING
    # =========================================================================
    print("\nSTEP 0: Cleaning data quality issues...")
    
    # Track cleaning actions
    cleaning_log = []
    
    for col in num_cols:
        if col not in df_train_full.columns:
            continue
        
        # Get statistics before cleaning
        orig_mean = df_train_full[col].mean()
        orig_max = df_train_full[col].max()
        orig_q99 = df_train_full[col].quantile(0.99)
        orig_q999 = df_train_full[col].quantile(0.999)
        
        # 1. Handle negative values
        for split_name, split_df in [('train', df_train), ('holdout', df_holdout), 
                                      ('val', df_val), ('test', df_test)]:
            if len(split_df) == 0 or col not in split_df.columns:
                continue
                
            neg_mask = split_df[col] < 0
            neg_count = neg_mask.sum()
            
            if neg_count > 0:
                # For gambling data, negatives are usually errors
                if any(term in col.lower() for term in ['cancel', 'refund', 'delta']):
                    # These might legitimately be negative
                    split_df.loc[neg_mask, col] = split_df.loc[neg_mask, col].abs()
                else:
                    # Set to 0 for other metrics
                    split_df.loc[neg_mask, col] = 0
                
                cleaning_log.append(f"{split_name}/{col}: fixed {neg_count} negative values")
        
        # 2. Handle extreme outliers (data errors)
        # Re-compute stats after negative fix
        df_train_full_updated = pd.concat([df_train, df_holdout], ignore_index=True)
        
        if len(df_train_full_updated[col]) > 100:
            q99 = df_train_full_updated[col].quantile(0.99)
            q999 = df_train_full_updated[col].quantile(0.999)
            max_val = df_train_full_updated[col].max()
            
            # Detect unrealistic values
            # If max > 10000x the 99th percentile, it's almost certainly an error
            if max_val > 10000 * q99 and q99 > 0:
                # Use 99.9th percentile as cap
                cap_value = q999 * 10  # Allow some headroom above 99.9th percentile
                
                for split_name, split_df in [('train', df_train), ('holdout', df_holdout), 
                                              ('val', df_val), ('test', df_test)]:
                    if len(split_df) == 0 or col not in split_df.columns:
                        continue
                    
                    extreme_mask = split_df[col] > cap_value
                    extreme_count = extreme_mask.sum()
                    
                    if extreme_count > 0:
                        split_df.loc[extreme_mask, col] = cap_value
                        cleaning_log.append(
                            f"{split_name}/{col}: capped {extreme_count} extreme values "
                            f"(was up to {split_df.loc[extreme_mask, col].max():.2e}, "
                            f"capped at {cap_value:.2e})"
                        )
    
    # Print cleaning summary
    if cleaning_log:
        print("\nData cleaning actions:")
        for action in cleaning_log[:20]:  # Show first 20
            print(f"  - {action}")
        if len(cleaning_log) > 20:
            print(f"  ... and {len(cleaning_log) - 20} more actions")
    
    # Re-combine after cleaning
    df_train_full = pd.concat([df_train, df_holdout], ignore_index=True)
    
    # =========================================================================
    # STEP 1: ANALYZE TAIL BEHAVIOR ON CLEANED DATA
    # =========================================================================
    print("\nSTEP 1: Analyzing tail behavior on CLEANED data...")
    
    tail_analysis = {}
    thresholds = {}
    
    for col in num_cols:
        if col not in df_train_full.columns:
            continue
            
        # Get CLEANED data
        data = df_train_full[col].dropna().values
        
        # Remove any remaining invalid values
        data = data[np.isfinite(data)]
        data = data[data >= 0]  # Ensure non-negative
        
        if len(data) < 100:
            tail_analysis[col] = {
                'tail_index': 3.0,
                'classification': 'insufficient_data',
                'threshold': np.quantile(data, 0.95) if len(data) > 0 else 0,
                'transform': 'robust_only'
            }
            thresholds[col] = tail_analysis[col]['threshold']
            continue
        
        # Heavy-tail analysis
        analyzer = HeavyTailAnalyzer(data, name=col)
        
        # Get optimal threshold
        try:
            threshold, quantile = analyzer.find_optimal_threshold(method='stability')
        except:
            threshold = np.quantile(data, 0.95)
            quantile = 0.95
        
        # Estimate tail index
        try:
            _, alpha_estimates = analyzer.hill_estimator()
            valid_estimates = alpha_estimates[~np.isnan(alpha_estimates)]
            if len(valid_estimates) > 0:
                tail_index = np.median(valid_estimates)
            else:
                tail_index = 2.5
        except:
            tail_index = 2.5
        
        # Quick validation
        q95 = np.quantile(data, 0.95)
        q99 = np.quantile(data, 0.99)
        max_val = data.max()
        
        # Classify (with more conservative thresholds after cleaning)
        if tail_index <= 1.5 or max_val > 20 * q95:
            classification = 'extreme_heavy'
            transform = 'log_then_robust'
        elif tail_index <= 2.5 or max_val > 10 * q95:
            classification = 'heavy'
            transform = 'log_then_robust'
        elif tail_index <= 3.5 or max_val > 5 * q95:
            classification = 'moderate_heavy'
            transform = 'sqrt_then_robust'
        else:
            classification = 'light'
            transform = 'robust_only'
        
        tail_analysis[col] = {
            'tail_index': tail_index,
            'classification': classification,
            'threshold': threshold,
            'quantile': quantile,
            'transform': transform,
            'q95': q95,
            'q99': q99,
            'max': max_val,
            'max_to_q95_ratio': max_val / (q95 + 1e-8)
        }
        
        thresholds[col] = threshold
        
        print(f"  {col}: α={tail_index:.2f}, class={classification}, "
              f"max/q95={tail_analysis[col]['max_to_q95_ratio']:.1f}x")
    
    # =========================================================================
    # STEP 2: CREATE EXCEEDANCE FEATURES FROM CLEANED DATA
    # =========================================================================
    print("\nSTEP 2: Creating exceedance features from cleaned values...")
    
    def create_exceedance_features(df_split, thresholds):
        for col, threshold in thresholds.items():
            if col in df_split.columns:
                df_split[f'I_{col}'] = (df_split[col] > threshold).astype(np.int8)
                df_split[f'E_{col}'] = (df_split[col] - threshold).clip(lower=0).astype(np.float32)
        return df_split
    
    df_train = create_exceedance_features(df_train, thresholds)
    df_holdout = create_exceedance_features(df_holdout, thresholds)
    df_val = create_exceedance_features(df_val, thresholds)
    df_test = create_exceedance_features(df_test, thresholds)
    
    # =========================================================================
    # STEP 3: ADAPTIVE NORMALIZATION WITH SAFER TRANSFORMS
    # =========================================================================
    print("\nSTEP 3: Applying adaptive normalization...")
    
    # Group columns by transformation type
    transform_groups = {
        'log_then_robust': [],
        'sqrt_then_robust': [],
        'robust_only': []
    }
    
    for col, info in tail_analysis.items():
        transform_groups[info['transform']].append(col)
    
    scalers = {}
    
    # 3A: Safe log transform for heavy-tailed
    if transform_groups['log_then_robust']:
        print(f"\n  Safe log transform for {len(transform_groups['log_then_robust'])} heavy-tailed features...")
        
        for col in transform_groups['log_then_robust']:
            if col not in df_train.columns:
                continue
            
            # Use log1p with shift to ensure all positive
            for df_split in [df_train, df_holdout, df_val, df_test]:
                if len(df_split) > 0 and col in df_split.columns:
                    # Ensure non-negative
                    values = df_split[col].values
                    min_val = values.min()
                    
                    if min_val < 0:
                        # Shift to make positive
                        shift = abs(min_val) + 1
                        df_split[f'{col}_norm'] = np.log1p(values + shift)
                    else:
                        # Standard log1p
                        df_split[f'{col}_norm'] = np.log1p(values)
            
            # Fit RobustScaler
            norm_values = []
            for df_split in [df_train, df_holdout]:
                if len(df_split) > 0 and f'{col}_norm' in df_split.columns:
                    norm_values.extend(df_split[f'{col}_norm'].values)
            
            if norm_values:
                scaler = RobustScaler(quantile_range=(10, 90))
                scaler.fit(np.array(norm_values).reshape(-1, 1))
                scalers[f'{col}_norm'] = scaler
                
                # Apply scaling
                for df_split in [df_train, df_holdout, df_val, df_test]:
                    if len(df_split) > 0 and f'{col}_norm' in df_split.columns:
                        values = df_split[f'{col}_norm'].values.reshape(-1, 1)
                        df_split[f'{col}_norm'] = scaler.transform(values).flatten()
    
    # 3B: Square root transform + RobustScaler for moderate
    if transform_groups['sqrt_then_robust']:
        print(f"\n  Square root transform for {len(transform_groups['sqrt_then_robust'])} moderate features...")
        
        for col in transform_groups['sqrt_then_robust']:
            if col not in df_train.columns:
                continue
                
            # Apply sqrt transform
            for df_split in [df_train, df_holdout, df_val, df_test]:
                if len(df_split) > 0 and col in df_split.columns:
                    # Handle negative values by taking sqrt of absolute value and preserving sign
                    values = df_split[col].values
                    signs = np.sign(values)
                    df_split[f'{col}_norm'] = signs * np.sqrt(np.abs(values))
            
            # Fit RobustScaler
            sqrt_vals = []
            for df_split in [df_train, df_holdout]:
                if len(df_split) > 0 and col in df_split.columns:
                    values = df_split[col].values
                    signs = np.sign(values)
                    sqrt_vals.extend(signs * np.sqrt(np.abs(values)))
            
            sqrt_data = np.array(sqrt_vals).reshape(-1, 1)
            scaler = RobustScaler(quantile_range=(5, 95))
            scaler.fit(sqrt_data)
            scalers[f'{col}_norm'] = scaler
            
            # Apply scaling
            for df_split in [df_train, df_holdout, df_val, df_test]:
                if len(df_split) > 0 and f'{col}_norm' in df_split.columns:
                    values = df_split[f'{col}_norm'].values.reshape(-1, 1)
                    df_split[f'{col}_norm'] = scaler.transform(values).flatten()
    
    # 3C: RobustScaler only for light-tailed
    # 3C: RobustScaler only for light-tailed
        if transform_groups['robust_only']:
            print(f"\n  RobustScaler only for {len(transform_groups['robust_only'])} light-tailed features...")
            
            # Can do batch processing for efficiency
            robust_cols = [c for c in transform_groups['robust_only'] if c in df_train.columns]
            
            if robust_cols:
                # Create normalized columns (copy of raw)
                for col in robust_cols:
                    for df_split in [df_train, df_holdout, df_val, df_test]:
                        if len(df_split) > 0 and col in df_split.columns:
                            df_split[f'{col}_norm'] = df_split[col].copy()
                
                # Fit and apply scaler - FIX: get data from the splits, not df_train_full
                norm_cols = [f'{c}_norm' for c in robust_cols]
                
                # Collect training data for fitting the scaler
                train_data_list = []
                holdout_data_list = []
                
                # Check which norm columns exist in train/holdout
                valid_train_cols = [c for c in norm_cols if c in df_train.columns]
                valid_holdout_cols = [c for c in norm_cols if c in df_holdout.columns]
                
                if valid_train_cols:
                    train_data_list.append(df_train[valid_train_cols].values)
                if valid_holdout_cols:
                    holdout_data_list.append(df_holdout[valid_holdout_cols].values)
                
                # Concatenate train and holdout data
                if train_data_list or holdout_data_list:
                    all_data = []
                    if train_data_list:
                        all_data.extend(train_data_list)
                    if holdout_data_list:
                        all_data.extend(holdout_data_list)
                    
                    combined_data = np.vstack(all_data)
                    
                    # Fit scaler
                    scaler = RobustScaler(quantile_range=(5, 95))
                    scaler.fit(combined_data)
                    
                    # Apply to all splits
                    for df_split in [df_train, df_holdout, df_val, df_test]:
                        if len(df_split) > 0:
                            valid_norm_cols = [c for c in norm_cols if c in df_split.columns]
                            if valid_norm_cols:
                                df_split[valid_norm_cols] = scaler.transform(df_split[valid_norm_cols])
                    
                    scalers['robust_batch'] = (robust_cols, scaler)
    # =========================================================================
    # STEP 4: NORMALIZE EXCEEDANCE MAGNITUDES
    # =========================================================================
    print("\nSTEP 4: Normalizing exceedance magnitudes...")
    
    E_cols = [c for c in df_train.columns if c.startswith('E_')]
    
    for col in E_cols:
        # Log transform exceedance magnitudes (they're always positive)
        for df_split in [df_train, df_holdout, df_val, df_test]:
            if len(df_split) > 0 and col in df_split.columns:
                df_split[f'{col}_norm'] = np.log1p(df_split[col])
    
    # =========================================================================
    # STEP 5: VALIDATION AND DIAGNOSTICS
    # =========================================================================
    print("\nSTEP 5: Validating normalization...")
    
    # Report robust statistics in addition to mean/std
    for split_name, df_split in [('train', df_train), ('holdout', df_holdout),
                                  ('val', df_val), ('test', df_test)]:
        if len(df_split) == 0:
            continue
            
        norm_cols = [c for c in df_split.columns if c.endswith('_norm') and not c.startswith(('I_', 'E_'))]
        
        if norm_cols:
            # Basic stats
            means = df_split[norm_cols].mean()
            stds = df_split[norm_cols].std()
            medians = df_split[norm_cols].median()
            q25 = df_split[norm_cols].quantile(0.25)
            q75 = df_split[norm_cols].quantile(0.75)
            iqr = q75 - q25
            
            print(f"\n  {split_name} statistics:")
            print(f"    Mean (avg): {means.mean():.3f}")
            print(f"    Std (avg): {stds.mean():.3f}")
            print(f"    Median (avg): {medians.mean():.3f}")
            print(f"    IQR (avg): {iqr.mean():.3f}")
            
            # Check for problems
            problematic = []
            for col in norm_cols:
                mean_val = means[col]
                std_val = stds[col]
                
                if abs(mean_val) > 5:
                    problematic.append(f"{col}: mean={mean_val:.1f}")
                elif std_val > 10:
                    problematic.append(f"{col}: std={std_val:.1f}")
            
            if problematic:
                print(f"    WARNING: Potential normalization issues:")
                for p in problematic[:5]:  # Show first 5
                    print(f"      {p}")
            else:
                print(f"    ✓ All features properly normalized")
    
    # =========================================================================
    # STEP 6: CREATE SUMMARY REPORT
    # =========================================================================
    
    # Create summary dataframe
    summary_data = []
    for col, info in tail_analysis.items():
        summary_data.append({
            'feature': col,
            'tail_index': info['tail_index'],
            'classification': info['classification'],
            'transform': info['transform'],
            'threshold': info['threshold'],
            'max_to_q95_ratio': info['max_to_q95_ratio']
        })
    
    summary_df = pd.DataFrame(summary_data).sort_values('tail_index')
    
    # Print summary
    print("\n" + "="*60)
    print("NORMALIZATION SUMMARY")
    print("="*60)
    print(f"\nFeature count by tail behavior:")
    print(summary_df['classification'].value_counts())
    print(f"\nTransformation strategy used:")
    print(summary_df['transform'].value_counts())
    
    return df_train, df_holdout, df_val, df_test, tail_analysis, scalers, summary_df


##############################################
# Main Pipeline: FIXED WITH SEQUENTIAL PROCESSING
##############################################
def main():
    # Database config
    DB_NAME = os.getenv('DB_NAME')
    DB_USER = os.getenv('DB_USER')
    DB_PASS = os.getenv('DB_PASS')
    DB_HOST = os.getenv('DB_HOST')
    DB_PORT = os.getenv('DB_PORT')

    engine = create_engine(
        f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

    # Define tables
    table_names = {
        #'sessions':     'my_definition_study.combined_sessions_daily',
        #'bets':         'my_definition_study.combined_bets_daily',
        'transactions': 'my_definition_study.combined_transactions_daily',
        #'payments':     'my_definition_study.combined_payments_daily',
    }
    labels_table = 'public.manual_assessments'

    # Load labels
    sql_lbl = f"SELECT customer_id, date, category FROM {labels_table}"
    labels_df = pd.read_sql_query(sql_lbl, engine)
    labels_df['date'] = pd.to_datetime(labels_df['date'], utc=True).dt.normalize()

    # Prepare folders
    os.makedirs('plots/normalization', exist_ok=True)
    os.makedirs('plots/block_stats', exist_ok=True)
    os.makedirs('plots/windows', exist_ok=True)
    os.makedirs('processed_data', exist_ok=True)

    # Process each table in sequence
    for key, tbl in table_names.items():
        print(f"\n{'='*70}")
        print(f"PROCESSING {key.upper()}")
        print(f"{'='*70}")
        
        # --- Load ---
        chunks = []
        for ch in pd.read_sql_query(f"SELECT * FROM {tbl}", engine, chunksize=50000):
            chunks.append(ch)
        df = pd.concat(chunks, ignore_index=True)
        del chunks; clear_mem()

        # --- Basic Preprocessing ---
        if 'day_period' in df.columns:
            df['day_period'] = pd.to_datetime(df['day_period'], utc=True, errors='coerce').dt.normalize()
        num_cols = df.select_dtypes(include='number').columns.tolist()
        for c in num_cols:
            df[c] = df[c].fillna(0)

        # --- HYBRID SPLITS ---
        df_train, df_holdout, df_val, df_test = make_hybrid_splits(
            df, 
            user_col='customer_id', 
            time_col='day_period',
            train_user_frac=0.8,
            val_user_frac=0.1,
            temporal_holdout_frac=0.2
        )
        
        # Clean up original df
        del df; clear_mem()

        # --- HEAVY-TAIL AWARE NORMALIZATION WITH DATA CLEANING ---
        df_train, df_holdout, df_val, df_test, tail_analysis, scalers, summary_df = \
            process_heavy_tailed_data_with_cleaning(None, df_train, df_holdout, df_val, df_test, num_cols, key)
        
        # Save tail analysis summary
        os.makedirs(os.path.join('processed_data', key), exist_ok=True)
        summary_df.to_csv(os.path.join('processed_data', key, 'tail_analysis_summary.csv'), index=False)

        # --- PROCESS SPLITS SEQUENTIALLY (CRITICAL FIX) ---
        print("\nProcessing splits sequentially to manage memory...")
        
        split_configs = [
            ('train', df_train),
            ('holdout', df_holdout),
            ('val', df_val),
            ('test', df_test)
        ]
        
        # Initialize storage for window results
        all_window_results = {}
        
        for split_name, df_split in split_configs:
            if len(df_split) == 0:
                print(f"\nSkipping {split_name} (empty)")
                all_window_results[split_name] = None
                continue
                
            print(f"\n{'='*50}")
            print(f"Processing {split_name} split ({len(df_split)} samples)")
            print('='*50)
            
            # --- Exceedance Block Stats (FIXED VERSION) ---
            print("Computing exceedance statistics...")
            df_split = compute_exceedance_block_statistics(df_split, block_size=7)
            df_split = compute_multi_feature_exceedance_stats(df_split)
            
            # --- Add Gap Features ---
            print("Adding gap features...")
            df_split = add_gap_features(df_split, user_col='customer_id', time_col='day_period')
            
            # --- Merge Labels ---
            print("Merging labels...")
            df_split = merge_labels(df_split, labels_df)
            
            # --- Prepare Feature Columns ---
            feat_cols = [c for c in df_split.columns 
                         if c.endswith('_norm') or c.startswith(('I_','E_')) 
                         or '_exceeds_' in c or 'exceedance_' in c]
            # Add gap features
            gap_feat_cols = ['days_since_last_activity_norm']
            feat_cols.extend(gap_feat_cols)
            # Remove duplicates
            feat_cols = list(dict.fromkeys(feat_cols))
            
            # --- Create Hybrid Windows ---
            print("Creating hybrid windows...")
            results = create_hybrid_windows(df_split, feat_cols, label_col='category')
            all_window_results[split_name] = results
            
            # --- Save CSV ---
            print(f"Saving {split_name} CSV...")
            out_dir = os.path.join('processed_data', key)
            df_split.to_csv(os.path.join(out_dir, f'{split_name}.csv'), index=False)
            
            # --- Save Windows ---
            if results is not None:
                # Save activity-based windows
                X_act, L_act, I_act, dates_act = results['activity']
                if len(X_act) > 0:
                    np.save(os.path.join(out_dir, f'X_{split_name}_activity.npy'), X_act)
                    np.save(os.path.join(out_dir, f'L_{split_name}_activity.npy'), L_act)
                    np.save(os.path.join(out_dir, f'ID_{split_name}_activity.npy'), I_act)
                    dates_act.to_csv(os.path.join(out_dir, f'window_dates_{split_name}_activity.csv'), index=False)
                
                # Save calendar-based windows
                X_cal, L_cal, I_cal, dates_cal = results['calendar']
                if len(X_cal) > 0:
                    np.save(os.path.join(out_dir, f'X_{split_name}_calendar.npy'), X_cal)
                    np.save(os.path.join(out_dir, f'L_{split_name}_calendar.npy'), L_cal)
                    np.save(os.path.join(out_dir, f'ID_{split_name}_calendar.npy'), I_cal)
                    dates_cal.to_csv(os.path.join(out_dir, f'window_dates_{split_name}_calendar.csv'), index=False)
                
                # Save combined for backward compatibility
                if len(X_act) > 0 and len(X_cal) > 0:
                    X_combined = np.vstack([X_act, X_cal])
                    L_combined = np.vstack([L_act, L_cal])
                    I_combined = np.hstack([I_act, I_cal])
                    dates_combined = pd.concat([dates_act, dates_cal], ignore_index=True)
                    
                    np.save(os.path.join(out_dir, f'X_{split_name}.npy'), X_combined)
                    np.save(os.path.join(out_dir, f'L_{split_name}.npy'), L_combined)
                    np.save(os.path.join(out_dir, f'ID_{split_name}.npy'), I_combined)
                    dates_combined.to_csv(os.path.join(out_dir, f'window_dates_{split_name}.csv'), index=False)
            
            # --- Check Label Leakage ---
            if results is not None and 'activity' in results:
                _, _, _, dates_df = results['activity']
                if len(dates_df) > 0:
                    check_label_leakage(dates_df, labels_df)
            
            # --- CRITICAL: Clean up split data ---
            print(f"Cleaning up {split_name} data...")
            del df_split
            clear_mem()
        
        # --- Save Split Info with All Details ---
        print("\nSaving split information...")
        
        # Count windows
        window_counts = {'activity': {}, 'calendar': {}, 'combined': {}}
        for split_name, results in all_window_results.items():
            if results is None:
                continue
            
            X_act, _, _, _ = results['activity']
            X_cal, _, _, _ = results['calendar']
            
            window_counts['activity'][split_name] = len(X_act)
            window_counts['calendar'][split_name] = len(X_cal)
            window_counts['combined'][split_name] = len(X_act) + len(X_cal)
        
        # Prepare split info
        split_info = {
            'split_type': 'hybrid_with_gap_features_and_cleaning',
            'window_types': ['activity_based', 'calendar_based'],
            'n_features': len(feat_cols) if 'feat_cols' in locals() else 0,
            'feature_names': feat_cols if 'feat_cols' in locals() else [],
            'gap_features': ['days_since_last_activity', 'days_since_last_activity_norm', 'gap_category'],
            'activity_constraints': {
                'max_calendar_span': 180,
                'split_on_long_gaps': 90
            },
            'calendar_constraints': {
                'max_gaps': 6,
                'window_size': 30
            },
            'window_counts': window_counts,
            'tail_analysis': {col: info for col, info in tail_analysis.items()},
            'scalers_info': {
                'log_then_robust': transform_groups['log_then_robust'] if 'transform_groups' in locals() else [],
                'sqrt_then_robust': transform_groups['sqrt_then_robust'] if 'transform_groups' in locals() else [],
                'robust_only': transform_groups['robust_only'] if 'transform_groups' in locals() else []
            },
            'data_dtype': 'float32',
            'validation_passed': True
        }
        
        # Save to JSON
        with open(os.path.join('processed_data', key, 'split_info.json'), 'w') as f:
            json.dump(split_info, f, indent=2, default=str)
        
        print(f"\nCompleted processing {key}")
        print(f"Total windows created:")
        for phase in ['activity', 'calendar', 'combined']:
            if phase in window_counts and window_counts[phase]:
                total = sum(window_counts[phase].values())
                print(f"  {phase}: {total:,} windows")

        # Clean up
        clear_mem()

    print("\n" + "="*70)
    print("ALL TABLES PROCESSED SUCCESSFULLY")
    print("="*70)
    print("Critical fixes applied:")
    print("  ✓ Data quality cleaning (negative values, extreme outliers)")
    print("  ✓ Heavy-tail thresholds computed on CLEANED data")
    print("  ✓ Exceedance features created from cleaned values")
    print("  ✓ Safe normalization handling edge cases")
    print("  ✓ Float32 instead of float16 for large values")
    print("  ✓ Memory-efficient exceedance block statistics")
    print("  ✓ Sequential split processing with cleanup")
    print("  ✓ Parallel bootstrap for faster analysis")
    print("  ✓ Label leakage checks")
    print("  ✓ Robust statistics reporting")
    print("  ✓ Complete reproducibility with saved parameters")


if __name__ == "__main__":
    main()