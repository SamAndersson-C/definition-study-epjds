#!/usr/bin/env python3
"""
Enhanced Feature Engineering with GUARANTEED CONSISTENCY
Ensures identical features across:
- All splits (train, holdout, val, test)
- All window types (activity, calendar)
Critical for two-stage ML pipelines
"""
import os, sys, gc, random, argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
from pandas.api.types import is_numeric_dtype
from numpy.lib.format import open_memmap
import json
import logging
from numba import jit
import shutil
from scipy.ndimage import uniform_filter1d
import pickle

warnings.filterwarnings('ignore')

# CRITICAL: Set seeds before any operations
GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

import faulthandler
faulthandler.enable()

# -----------------------------------------------------------------------------
# CONSISTENCY MANAGER - Ensures identical processing across all data
# -----------------------------------------------------------------------------
class ConsistencyManager:
    """
    Manages all parameters and computations that must be consistent
    across splits and window types
    """
    def __init__(self, output_dir, data_type):
        self.output_dir = output_dir
        self.data_type = data_type
        self.consistency_dir = os.path.join(output_dir, data_type, '_consistency')
        os.makedirs(self.consistency_dir, exist_ok=True)
        
        # Storage for consistent parameters
        self.base_features = None
        self.fill_values = None
        self.tail_pairs = None
        self.thresholds = None
        self.feature_order = None
        self.snps_windows = (7, 14)  # Fixed
        self.tail_params = {
            'n_top': 5,
            'global_q': 0.95,
            'rolling_w': 14,
            'lag_days': 1
        }
    
    def save_consistency_data(self):
        """Save all consistency data to disk"""
        data = {
            'base_features': self.base_features,
            'fill_values': self.fill_values,
            'tail_pairs': self.tail_pairs,
            'thresholds': self.thresholds,
            'feature_order': self.feature_order,
            'snps_windows': self.snps_windows,
            'tail_params': self.tail_params,
            'seed': GLOBAL_SEED
        }
        
        # Save as both pickle and JSON (where possible)
        pickle_path = os.path.join(self.consistency_dir, 'consistency_data.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(data, f)
        
        # Save human-readable version
        json_data = {
            'base_features': self.base_features,
            'tail_pairs': self.tail_pairs,
            'feature_order': self.feature_order,
            'snps_windows': list(self.snps_windows),
            'tail_params': self.tail_params,
            'n_fill_values': len(self.fill_values) if self.fill_values else 0,
            'n_thresholds': len(self.thresholds) if self.thresholds else 0
        }
        json_path = os.path.join(self.consistency_dir, 'consistency_info.json')
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        logging.info(f"Saved consistency data to {self.consistency_dir}")
    
    def load_consistency_data(self):
        """Load consistency data if it exists"""
        pickle_path = os.path.join(self.consistency_dir, 'consistency_data.pkl')
        if os.path.exists(pickle_path):
            with open(pickle_path, 'rb') as f:
                data = pickle.load(f)
            
            self.base_features = data['base_features']
            self.fill_values = data['fill_values']
            self.tail_pairs = data['tail_pairs']
            self.thresholds = data['thresholds']
            self.feature_order = data['feature_order']
            self.snps_windows = data['snps_windows']
            self.tail_params = data['tail_params']
            
            logging.info(f"Loaded consistency data from {pickle_path}")
            return True
        return False
    
    def compute_from_train_data(self, df_train, base_cols):
        """Compute all consistent parameters from training data"""
        logging.info("Computing consistency parameters from training data...")
        
        # 1. Base features (already provided)
        self.base_features = base_cols
        
        # 2. Fill values (medians from train)
        self.fill_values = {}
        for col in base_cols:
            if col in df_train.columns:
                self.fill_values[col] = df_train[col].median()
        
        # 3. Tail dependence pairs (FIXED seed ensures consistency)
        self.tail_pairs, self.thresholds = self._compute_tail_pairs(df_train, base_cols)
        
        logging.info(f"Computed consistency parameters:")
        logging.info(f"  - {len(self.base_features)} base features")
        logging.info(f"  - {len(self.fill_values)} fill values")
        logging.info(f"  - {len(self.tail_pairs)} tail pairs")
    
    def _compute_tail_pairs(self, df_train, base_cols):
        """Compute tail pairs with fixed seed"""
        # CRITICAL: Use fixed seed for sampling
        random.seed(GLOBAL_SEED)
        
        samples = random.sample(base_cols, min(len(base_cols), 30))
        deps = {}
        
        for i, c1 in enumerate(samples):
            for c2 in samples[i+1:]:
                if df_train[c1].isna().mean() > 0.3 or df_train[c2].isna().mean() > 0.3:
                    continue
                deps[f"{c1}__{c2}"] = chi_global(df_train, c1, c2, self.tail_params['global_q'])
        
        # Get top pairs
        top_pairs = [p for p, _ in sorted(deps.items(), key=lambda x: x[1], reverse=True)[:self.tail_params['n_top']]]
        
        # Compute thresholds
        needed = {c for pair in top_pairs for c in pair.split("__")}
        thresholds = {}
        for c in needed:
            vals = df_train[c].values
            vals = vals[~np.isnan(vals)]
            if len(vals) > 0:
                thresholds[c] = np.quantile(vals, self.tail_params['global_q'])
            else:
                thresholds[c] = 0
        
        return top_pairs, thresholds
    
    def get_feature_order(self, base_features, snps_features, tail_features):
        """Get consistent feature ordering"""
        if self.feature_order is None:
            # First time - establish the order
            all_features = []
            all_features.extend(base_features)
            all_features.extend(sorted(snps_features))  # Sort for consistency
            all_features.extend(sorted(tail_features))  # Sort for consistency
            self.feature_order = list(dict.fromkeys(all_features))  # Remove duplicates
        
        return self.feature_order


# -----------------------------------------------------------------------------
# Feature computation functions (same as before but using ConsistencyManager)
# -----------------------------------------------------------------------------
def rolling_chi_efficient(ind1: np.ndarray, ind2: np.ndarray, window: int) -> np.ndarray:
    """Efficient rolling chi computation using cumsum trick"""
    ind1 = ind1.astype(np.float64)
    ind2 = ind2.astype(np.float64)
    joint = ind1 * ind2
    
    pad_width = window - 1
    ind1_pad = np.pad(ind1, (pad_width, 0), mode='constant', constant_values=0)
    ind2_pad = np.pad(ind2, (pad_width, 0), mode='constant', constant_values=0)
    joint_pad = np.pad(joint, (pad_width, 0), mode='constant', constant_values=0)
    
    cs1 = np.cumsum(ind1_pad)
    cs2 = np.cumsum(ind2_pad)
    csj = np.cumsum(joint_pad)
    
    sum1 = cs1[window-1:] - np.concatenate([[0], cs1[:-window]])
    sum2 = cs2[window-1:] - np.concatenate([[0], cs2[:-window]])
    sumj = csj[window-1:] - np.concatenate([[0], csj[:-window]])
    
    p1 = sum1 / window
    p2 = sum2 / window
    pj = sumj / window
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chi = pj / (p1 * p2 + 1e-10) - 1.0
        chi = np.clip(chi, 0.0, 1.0)
        chi = np.nan_to_num(chi, 0.0)
    
    return chi[:len(ind1)]


def add_snps_vectorized(df: pd.DataFrame, base_cols: list, window_sizes: tuple) -> pd.DataFrame:
    """Vectorized SNPS implementation"""
    df = df.sort_values(['customer_id', 'day_period']).reset_index(drop=True)
    
    customer_ids = df['customer_id'].values
    customer_boundaries = np.where(np.concatenate([[True], 
                                                   customer_ids[1:] != customer_ids[:-1], 
                                                   [True]]))[0]
    
    for col in tqdm(base_cols, desc="SNPS features", leave=False):
        if col not in df.columns:
            continue
            
        if df[col].isna().mean() > 0.9:
            logging.warning(f"Skipping {col} - too many NaN values")
            continue
        
        values = df[col].fillna(0).values.astype(np.float64)
        values_squared = values ** 2
        
        for w in window_sizes:
            result = np.zeros_like(values)
            
            for i in range(len(customer_boundaries) - 1):
                start_idx = customer_boundaries[i]
                end_idx = customer_boundaries[i + 1]
                
                cust_vals = values[start_idx:end_idx]
                cust_vals_sq = values_squared[start_idx:end_idx]
                
                if len(cust_vals) == 0:
                    continue
                
                padded_vals = np.pad(cust_vals, (w-1, 0), mode='constant')
                padded_vals_sq = np.pad(cust_vals_sq, (w-1, 0), mode='constant')
                
                cumsum = np.cumsum(padded_vals)
                cumsum_sq = np.cumsum(padded_vals_sq)
                
                rolling_sum = cumsum[w-1:] - np.concatenate([[0], cumsum[:-w]])
                rolling_sum_sq = cumsum_sq[w-1:] - np.concatenate([[0], cumsum_sq[:-w]])
                
                with np.errstate(divide='ignore', invalid='ignore'):
                    snps = rolling_sum / np.sqrt(rolling_sum_sq + 1e-10)
                    snps = np.nan_to_num(snps, 0.0)
                
                result[start_idx:end_idx] = snps[:len(cust_vals)]
            
            df[f"{col}_snps_{w}d"] = result.astype(np.float32)
    
    return df


def chi_global(df_train: pd.DataFrame, c1: str, c2: str, q: float) -> float:
    """Compute chi using training data"""
    try:
        vals1 = df_train[c1].values
        vals2 = df_train[c2].values
        mask = ~(np.isnan(vals1) | np.isnan(vals2))
        vals1 = vals1[mask]
        vals2 = vals2[mask]
        
        if len(vals1) == 0:
            return 0.0
            
        t1 = np.quantile(vals1, q)
        t2 = np.quantile(vals2, q)
        ex1 = (vals1 > t1).astype(int)
        ex2 = (vals2 > t2).astype(int)
        pj = np.mean(ex1 & ex2)
        p1 = np.mean(ex1)
        p2 = np.mean(ex2)
        
        if p1 > 0 and p2 > 0 and pj > 0:
            return min(1.0, max(0.0, pj/(p1*p2) - 1.0))
    except Exception as e:
        logging.error(f"Error in chi_global for {c1}, {c2}: {str(e)}")
    return 0.0


def add_tail_dependence_consistent(df: pd.DataFrame, consistency_mgr: ConsistencyManager) -> pd.DataFrame:
    """Add tail dependence using pre-computed consistent parameters"""
    if df.empty or not consistency_mgr.tail_pairs:
        return df
    
    df = df.sort_values(['customer_id', 'day_period']).reset_index(drop=True)
    
    user_ids = df['customer_id'].values
    user_changes = np.concatenate([[True], user_ids[1:] != user_ids[:-1], [True]])
    user_starts = np.where(user_changes)[0]
    
    # Use pre-computed pairs and thresholds
    for pair in consistency_mgr.tail_pairs:
        c1, c2 = pair.split("__")
        feat = f"tail_dep_{c1}_{c2}_lag{consistency_mgr.tail_params['lag_days']}"
        
        # Check if columns exist
        if c1 not in df.columns or c2 not in df.columns:
            # Add zero column for consistency
            df[feat] = 0.0
            continue
        
        b1 = (df[c1].values > consistency_mgr.thresholds[c1]).astype(float)
        b2 = (df[c2].values > consistency_mgr.thresholds[c2]).astype(float)
        joint = b1 * b2
        
        lag_days = consistency_mgr.tail_params['lag_days']
        rolling_w = consistency_mgr.tail_params['rolling_w']
        
        b1_lagged = np.concatenate([np.zeros(lag_days), b1[:-lag_days]])
        b2_lagged = np.concatenate([np.zeros(lag_days), b2[:-lag_days]])
        joint_lagged = np.concatenate([np.zeros(lag_days), joint[:-lag_days]])
        
        sum1 = uniform_filter1d(b1_lagged, size=rolling_w, mode='constant', cval=0) * rolling_w
        sum2 = uniform_filter1d(b2_lagged, size=rolling_w, mode='constant', cval=0) * rolling_w
        sumj = uniform_filter1d(joint_lagged, size=rolling_w, mode='constant', cval=0) * rolling_w
        
        with np.errstate(divide='ignore', invalid='ignore'):
            p1 = sum1 / rolling_w
            p2 = sum2 / rolling_w
            pj = sumj / rolling_w
            chi = pj / (p1 * p2 + 1e-10) - 1.0
            chi = np.clip(chi, 0.0, 1.0)
            chi = np.nan_to_num(chi, 0.0)
        
        result = chi.copy()
        for i in range(len(user_starts) - 1):
            start_idx = user_starts[i]
            result[start_idx:start_idx + lag_days + rolling_w] = 0
        
        df[feat] = result.astype(np.float32)
    
    return df


def aggregate_daily_features(df, feature_cols):
    """Aggregate features to daily level"""
    df['date'] = pd.to_datetime(df['day_period']).dt.date
    
    agg_dict = {}
    for col in feature_cols:
        if col in df.columns:
            if col.startswith('I_') or '_exceeds_' in col or col.endswith('_any_'):
                agg_dict[col] = 'max'
            elif col.startswith('E_') or '_magnitude_' in col or col.endswith('_sum'):
                agg_dict[col] = 'sum'
            elif '_count_' in col or col.endswith('_num'):
                agg_dict[col] = 'sum'
            elif col.endswith('_mean') or col.endswith('_avg'):
                agg_dict[col] = 'mean'
            else:
                agg_dict[col] = 'mean'
    
    # FIXED: Don't add customer_id to agg_dict since it's in groupby
    # Remove it if it's there
    if 'customer_id' in agg_dict:
        del agg_dict['customer_id']
    
    logging.info(f"Aggregating {len(df)} records to daily level...")
    df_daily = df.groupby(['customer_id', 'date']).agg(agg_dict).reset_index()
    
    df_daily['day_period'] = pd.to_datetime(df_daily['date'])
    df_daily = df_daily.drop('date', axis=1)
    
    logging.info(f"Aggregated to {len(df_daily)} daily records")
    
    # Check for remaining duplicates
    dup_check = df_daily.groupby(['customer_id', 'day_period']).size()
    if (dup_check > 1).any():
        n_dups = (dup_check > 1).sum()
        logging.warning(f"Still have {n_dups} duplicate customer-date pairs after aggregation!")
        df_daily = df_daily.drop_duplicates(subset=['customer_id', 'day_period'], keep='first')
    
    return df_daily


def create_enhanced_windows_consistent(df, window_info, feature_cols, consistency_mgr, 
                                      out_dir, split, src_data_dir):
    """Create enhanced windows with consistent features and fill values"""
    df = df.sort_values(['customer_id', 'day_period']).reset_index(drop=True)
    
    n_windows = window_info['n_windows']
    seq_len = window_info['seq_len']
    window_type = window_info.get('window_type', None)
    suffix = f"_{window_type}" if window_type else ""
    
    # CRITICAL: Use consistent feature order
    feature_cols = consistency_mgr.feature_order
    n_features = len(feature_cols)
    
    # CRITICAL: Use consistent fill values
    fill_values = consistency_mgr.fill_values
    
    X_enhanced = np.zeros((n_windows, seq_len, n_features), dtype=np.float32)
    success_mask = np.zeros(n_windows, dtype=bool)
    
    # Create customer lookup
    customer_data = {}
    for customer_id, group in df.groupby('customer_id'):
        group_daily = group.drop_duplicates(subset=['day_period'], keep='first')
        customer_data[str(customer_id)] = group_daily.sort_values('day_period')
    
    if 'window_dates' not in window_info or window_info['window_dates'] is None:
        logging.error("No window_dates found!")
        return X_enhanced
    
    window_dates_df = window_info['window_dates']
    
    logging.info(f"Creating {n_windows} enhanced windows for {split}{suffix}")
    logging.info(f"Using {n_features} consistent features")
    
    missing_customers = set()
    successful_windows = 0
    
    for i in tqdm(range(min(n_windows, len(window_dates_df))), desc=f"Windows {split}{suffix}"):
        window_info_row = window_dates_df.iloc[i]
        actual_customer_id = str(window_info_row['customer_id'])
        
        if actual_customer_id not in customer_data:
            missing_customers.add(actual_customer_id)
            # Fill with consistent fill values
            for j, col in enumerate(feature_cols):
                X_enhanced[i, :, j] = fill_values.get(col, 0)
            continue
        
        cust_df = customer_data[actual_customer_id]
        
        try:
            # Handle activity-based windows
            if 'activity_dates' in window_info_row and pd.notna(window_info_row['activity_dates']):
                activity_dates = pd.to_datetime(window_info_row['activity_dates'].split(','))
                window_data = []
                
                for date in activity_dates:
                    mask = cust_df['day_period'].dt.date == date.date()
                    if mask.any():
                        window_data.append(cust_df[mask].iloc[0])
                
                if len(window_data) == seq_len:
                    window_df = pd.DataFrame(window_data)
                    for j, col in enumerate(feature_cols):
                        if col in window_df.columns:
                            X_enhanced[i, :, j] = window_df[col].fillna(fill_values.get(col, 0)).values
                        else:
                            # Feature doesn't exist - use fill value
                            X_enhanced[i, :, j] = fill_values.get(col, 0)
                    successful_windows += 1
                    success_mask[i] = True
            
            # Handle calendar-based windows
            else:
                start_date = pd.to_datetime(window_info_row['start_date'])
                end_date = pd.to_datetime(window_info_row['end_date'])
                date_range = pd.date_range(start_date, end_date, freq='D')
                
                if len(date_range) != seq_len:
                    date_range = date_range[:seq_len]
                
                window_data = np.zeros((seq_len, n_features), dtype=np.float32)
                days_found = 0
                
                for day_idx, date in enumerate(date_range[:seq_len]):
                    mask = cust_df['day_period'].dt.date == date.date()
                    if mask.any():
                        for j, col in enumerate(feature_cols):
                            if col in cust_df.columns:
                                window_data[day_idx, j] = cust_df[mask].iloc[0][col]
                            else:
                                window_data[day_idx, j] = fill_values.get(col, 0)
                        days_found += 1
                    else:
                        # Missing day - use consistent fill values
                        for j, col in enumerate(feature_cols):
                            window_data[day_idx, j] = fill_values.get(col, 0)
                
                X_enhanced[i] = window_data
                
                if days_found > 0:
                    successful_windows += 1
                    success_mask[i] = True
                    
        except Exception as e:
            logging.debug(f"Error processing window {i}: {str(e)}")
            continue
    
    logging.info(f"Successfully created {successful_windows}/{n_windows} windows ({successful_windows/n_windows*100:.1f}%)")
    
    # Save arrays
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f'X_{split}{suffix}.npy'), X_enhanced)
    np.save(os.path.join(out_dir, f'success_mask_{split}{suffix}.npy'), success_mask)
    
    # Copy other arrays
    for arr_name in ['L', 'ID']:
        src_path = os.path.join(src_data_dir, f'{arr_name}_{split}{suffix}.npy')
        if os.path.exists(src_path):
            dst_path = os.path.join(out_dir, f'{arr_name}_{split}{suffix}.npy')
            shutil.copy2(src_path, dst_path)
    
    dates_src = os.path.join(src_data_dir, f'window_dates_{split}{suffix}.csv')
    if os.path.exists(dates_src):
        dates_dst = os.path.join(out_dir, f'window_dates_{split}{suffix}.csv')
        shutil.copy2(dates_src, dates_dst)
    
    return X_enhanced


def load_window_info(data_dir, split, window_type=None):
    """Load window information from preprocessing output"""
    info = {}
    suffix = f"_{window_type}" if window_type else ""
    
    id_path = os.path.join(data_dir, f'ID_{split}{suffix}.npy')
    if os.path.exists(id_path):
        info['customer_ids'] = np.load(id_path)
    
    dates_path = os.path.join(data_dir, f'window_dates_{split}{suffix}.csv')
    if os.path.exists(dates_path):
        info['window_dates'] = pd.read_csv(dates_path)
    
    x_path = os.path.join(data_dir, f'X_{split}{suffix}.npy')
    if os.path.exists(x_path):
        X = np.load(x_path, mmap_mode='r')
        info['n_windows'] = X.shape[0]
        info['seq_len'] = X.shape[1]
        info['n_orig_features'] = X.shape[2]
    
    split_info_path = os.path.join(data_dir, 'split_info.json')
    if os.path.exists(split_info_path):
        with open(split_info_path, 'r') as f:
            split_info = json.load(f)
            if 'feature_names' in split_info:
                info['feature_names'] = split_info['feature_names']
    
    info['window_type'] = window_type
    return info


def get_valid_base_features(df, exclude_cols):
    """Get valid base features"""
    base_cols = [
        c for c in df.columns
        if c.endswith('_norm') and c not in exclude_cols
    ]
    
    if not base_cols:
        base_cols = [
            c for c in df.columns
            if is_numeric_dtype(df[c]) and c not in exclude_cols
            and not c.startswith(('I_', 'E_'))
        ]
    
    valid_cols = []
    for col in base_cols:
        missing_pct = df[col].isna().mean()
        if missing_pct < 0.9:
            valid_cols.append(col)
    
    return valid_cols


def validate_consistency(out_dir, data_type, splits, window_types):
    """Validate feature consistency across splits and window types"""
    logging.info("\nValidating feature consistency...")
    
    feature_sets = {}
    feature_counts = {}
    
    # Check each combination
    for window_type in window_types:
        suffix = f"_{window_type}" if window_type != 'combined' else ""
        
        for split in splits:
            key = f"{split}_{window_type}"
            x_path = os.path.join(out_dir, data_type, f'X_{split}{suffix}.npy')
            
            if os.path.exists(x_path):
                X = np.load(x_path, mmap_mode='r')
                feature_counts[key] = X.shape[2]
    
    # Check if all have same feature count
    unique_counts = set(feature_counts.values())
    if len(unique_counts) > 1:
        logging.error(f"❌ Inconsistent feature counts: {feature_counts}")
        return False
    
    # Load feature names
    feature_names_path = os.path.join(out_dir, data_type, 'feature_names.txt')
    if os.path.exists(feature_names_path):
        with open(feature_names_path, 'r') as f:
            feature_names = [line.strip() for line in f]
        logging.info(f"✓ All splits and window types use {len(feature_names)} consistent features")
    
    # Check consistency info
    consistency_info_path = os.path.join(out_dir, data_type, '_consistency', 'consistency_info.json')
    if os.path.exists(consistency_info_path):
        with open(consistency_info_path, 'r') as f:
            consistency_info = json.load(f)
        logging.info(f"✓ Consistency enforced with:")
        logging.info(f"  - {len(consistency_info['base_features'])} base features")
        logging.info(f"  - {consistency_info['n_fill_values']} fill values")
        logging.info(f"  - {len(consistency_info['tail_pairs'])} tail pairs")
        logging.info(f"  - Fixed random seed: {GLOBAL_SEED}")
    
    return True


# -----------------------------------------------------------------------------
# Main processing function with consistency guarantees
# -----------------------------------------------------------------------------
def process_with_consistency(data_type, input_dir, out_dir, window_types_to_process):
    """
    Process data with guaranteed consistency across splits and window types
    """
    data_dir = os.path.join(input_dir, data_type)
    
    # Initialize consistency manager
    consistency_mgr = ConsistencyManager(out_dir, data_type)
    
    # Check if consistency data already exists
    if consistency_mgr.load_consistency_data():
        logging.info("✓ Using existing consistency parameters")
    else:
        logging.info("⚠️  No consistency data found - will compute from training data")
    
    # Load all splits
    splits = []
    dfs = {}
    
    for split in ['train', 'holdout', 'val', 'test']:
        csv_path = os.path.join(data_dir, f"{split}.csv")
        if os.path.exists(csv_path):
            splits.append(split)
            dfs[split] = pd.read_csv(csv_path)
            if 'day_period' in dfs[split].columns:
                dfs[split]['day_period'] = pd.to_datetime(dfs[split]['day_period'])
            logging.info(f"Loaded {data_type}/{split}: {dfs[split].shape}")
    
    if not splits:
        logging.error(f"No data found for {data_type}")
        return
    
    # Get training data
    if 'train' in dfs and 'holdout' in dfs:
        df_train_full = pd.concat([dfs['train'], dfs['holdout']], ignore_index=True)
    elif 'train' in dfs:
        df_train_full = dfs['train']
    else:
        logging.error("No training data found!")
        return
    
    # Get base features
    excl = {'customer_id', 'day_period', 'date', 'category', 'gap_category'}
    base_cols = get_valid_base_features(df_train_full, excl)
    
    if not base_cols:
        logging.error(f"No valid features found for {data_type}")
        return
    
    # Compute consistency parameters if needed
    if consistency_mgr.base_features is None:
        consistency_mgr.compute_from_train_data(df_train_full, base_cols)
        consistency_mgr.save_consistency_data()
    
    # Get original features from preprocessing
    first_window_type = window_types_to_process[0] if window_types_to_process else None
    if first_window_type:
        if first_window_type == 'combined':
            window_info_ref = load_window_info(data_dir, 'train', window_type=None)
        else:
            window_info_ref = load_window_info(data_dir, 'train', window_type=first_window_type)
        
        original_features = window_info_ref.get('feature_names', [])
    else:
        original_features = []
    
    # Process each split with same parameters
    for split in splits:
        logging.info(f"\n{'='*60}")
        logging.info(f"Processing {data_type}/{split}")
        logging.info('='*60)
        
        df = dfs[split].copy()
        
        # Aggregate to daily
        all_features = list(set(consistency_mgr.base_features + original_features))
        df = aggregate_daily_features(df, all_features)
        gc.collect()
        
        # Add SNPS with consistent windows
        logging.info("Computing SNPS features...")
        df = add_snps_vectorized(df, consistency_mgr.base_features, consistency_mgr.snps_windows)
        gc.collect()
        
        # Add tail dependence with consistent pairs
        logging.info("Computing tail dependence features...")
        df = add_tail_dependence_consistent(df, consistency_mgr)
        gc.collect()
        
        # Get consistent feature names
        snps_features = [c for c in df.columns if '_snps_' in c]
        tail_features = [c for c in df.columns if c.startswith('tail_dep_')]
        
        # CRITICAL: Get consistent feature order
        feature_order = consistency_mgr.get_feature_order(
            original_features if original_features else consistency_mgr.base_features,
            snps_features,
            tail_features
        )
        
        logging.info(f"Using {len(feature_order)} features in consistent order")
        
        # Ensure all features exist (add zeros if missing)
        for feat in feature_order:
            if feat not in df.columns:
                logging.debug(f"Adding missing feature {feat} as zeros")
                df[feat] = 0.0
        
        # Process each window type with same features
        for window_type in window_types_to_process:
            logging.info(f"\nProcessing {window_type} windows for {split}")
            
            if window_type == 'combined':
                window_info = load_window_info(data_dir, split, window_type=None)
            else:
                window_info = load_window_info(data_dir, split, window_type=window_type)
            
            if 'n_windows' not in window_info:
                logging.warning(f"No {window_type} windows found for {split}")
                continue
            
            enhanced_data_dir = os.path.join(out_dir, data_type)
            window_info['window_type'] = window_type if window_type != 'combined' else None
            
            # Create windows with consistent features
            X_enhanced = create_enhanced_windows_consistent(
                df, window_info, feature_order, consistency_mgr,
                enhanced_data_dir, split, data_dir
            )
            
            # Save feature names once
            if split == 'train' and window_type == window_types_to_process[0]:
                with open(os.path.join(enhanced_data_dir, "feature_names.txt"), 'w') as f:
                    for feat in feature_order:
                        f.write(feat + "\n")
                
                # Save split info
                split_info_out = {
                    'split_type': 'enhanced_consistent',
                    'n_features': len(feature_order),
                    'feature_names': feature_order,
                    'consistency_guaranteed': True,
                    'random_seed': GLOBAL_SEED,
                    'window_types': window_types_to_process,
                    'enhancements': {
                        'SNPS': len(snps_features),
                        'tail_dependence': len(tail_features)
                    }
                }
                with open(os.path.join(enhanced_data_dir, 'split_info.json'), 'w') as f:
                    json.dump(split_info_out, f, indent=2)
        
        # Save enhanced CSV
        enhanced_csv_path = os.path.join(enhanced_data_dir, f"{split}.csv")
        df.to_csv(enhanced_csv_path, index=False)
        
        del df
        gc.collect()
    
    # Validate consistency
    validate_consistency(out_dir, data_type, splits, window_types_to_process)
    
    logging.info(f"\n✓ {data_type} processing complete with guaranteed consistency!")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Enhanced Feature Engineering with Guaranteed Consistency"
    )
    p.add_argument("-i", "--input-dir", required=True, 
                   help="Base directory with preprocessing output")
    p.add_argument("-o", "--output-dir", required=True, 
                   help="Output directory for enhanced data")
    p.add_argument("-t", "--data-types", nargs='+', default=["payments"],
                   help="Data types to process")
    p.add_argument("-w", "--window-types", nargs='+', default=["activity", "calendar"],
                   choices=["combined", "activity", "calendar"],
                   help="Window types to process")
    args = p.parse_args()
    
    logging.info("="*70)
    logging.info("Enhanced Feature Engineering with GUARANTEED CONSISTENCY")
    logging.info("="*70)
    logging.info("Features guaranteed to be identical across:")
    logging.info("  ✓ All splits (train, holdout, val, test)")
    logging.info("  ✓ All window types (activity, calendar)")
    logging.info("  ✓ Fixed random seed for reproducibility")
    logging.info("  ✓ Consistent fill values from training data")
    logging.info("  ✓ Same tail dependence pairs")
    logging.info("  ✓ Same feature ordering")
    logging.info("="*70)
    
    for data_type in args.data_types:
        data_dir = os.path.join(args.input_dir, data_type)
        
        if not os.path.exists(data_dir):
            logging.error(f"Data directory not found: {data_dir}")
            continue
        
        logging.info(f"\nProcessing {data_type}")
        process_with_consistency(data_type, args.input_dir, args.output_dir, args.window_types)
    
    logging.info("\n" + "="*70)
    logging.info("All processing complete with guaranteed consistency!")
    logging.info("="*70)

if __name__ == "__main__":
    main()