#!/usr/bin/env python3
"""
Feature Selection for Enhanced Gambling Data - With Supervised Enhancement
Handles both activity and calendar windows with unified selection option
Integrates unsupervised AND supervised feature selection methods
OPTIMIZED VERSION with stratified sampling, vectorization, and batch processing
FIXED VERSION addressing dimension mismatch issues
FIXED: Proper organization by data type (sessions/bets/payments/transactions)
MODIFIED: Uses RG predictions instead of manual labels for supervised selection
MODIFIED: Always includes gap feature (days_since_last_activity_norm)
FIXED: Excludes raw features, only selects normalized and derived features
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

import numpy as np
import pandas as pd
import os
import seaborn as sns
from tqdm import tqdm
import torch
import torch.nn as nn

from scipy import fftpack
import tempfile
import shutil
import gc
import psutil
import json
import time
from datetime import datetime
from contextlib import contextmanager
from collections import OrderedDict
import warnings
import argparse

# Import supervised feature selection methods
from feature_selection_enhanced_supervised import (
    load_label_data,
    compute_mutual_information_features,
    compute_rg_correlation_features,
    compute_gradient_boosting_importance,
    compute_risk_category_separation,
    combine_supervised_unsupervised_scores
)

# Additional imports for optimization
from sklearn.model_selection import train_test_split
from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import IncrementalPCA

# Suppress warnings
warnings.filterwarnings('ignore')

import faulthandler
faulthandler.enable()

# FIXED: Feature groups now only match normalized and derived features
FEATURE_GROUPS = OrderedDict([
    ('short', [
        # Pattern-based matching for normalized features
        '_norm',  # Catches all normalized features
        'I_',     # Binary indicators (these are fine)
        '_count_exceeds_',  # Block statistics
        '_any_exceeds_',
        '_freq_exceeds_',
        '_snps_7d'  # Derived features
    ]),
    ('mid', [
        '_sum_magnitude_',  # Block magnitude stats
        '_max_magnitude_',
        '_mean_magnitude_',
        '_snps_14d'  # Derived features
    ]),
    ('long', [
        'tail_dep_',  # Tail dependence features
        'exceedance_correlation',  # Meta features
        'exceedance_pattern_entropy',
        'multiple_features_exceeded',  # Additional meta features
        'total_features_exceeded'
    ]),
])

# CRITICAL: Gap feature that must always be included
MANDATORY_FEATURES = ['days_since_last_activity_norm']

# Valid data types
VALID_DATA_TYPES = ['sessions', 'bets', 'payments', 'transactions']

# Configuration - ENHANCED with supervised settings and optimization
CONFIG = {
    'seed': 42,  # FIXED: Added explicit seed to config
    'data_type': 'sessions',  # NEW: Specify which data type we're processing
    'base_input_dir': 'enhanced_data',  # NEW: Base directory for all data types
    'base_output_dir': 'processed_data_selected',  # NEW: Base output directory
    'base_results_dir': 'feature_selection_results',  # NEW: Base results directory
    'window_types': ['activity', 'calendar'],  # Two window types
    'process_window_types': 'both',  # 'both', 'activity', 'calendar', 'combined', or 'unified'
    'unified_selection': {
        'enabled': True,  # Set to True for 2-stage autoencoder
        'source_window_type': 'activity',  # Which window type to run selection on
        'apply_to_window_type': 'calendar'  # Apply the same features to this type
    },
    'feature_selection': {
        'methods': [
            'temporal_variance', 
            'temporal_correlation', 
            'fourier_analysis', 
            'temporal_importance'
        ],
        'supervised_methods': [  # NEW: Supervised methods
            'rg_score_correlation',  # MODIFIED: Use RG score correlation
            'rg_category_importance',  # MODIFIED: Use RG category importance
            'gradient_boosting_rg',  # MODIFIED: GB using RG labels
            'rg_consistency'  # MODIFIED: RG consistency across time
        ],
        'enable_supervised': True,  # NEW: Toggle supervised methods
        'supervised_weight': 0.5,   # NEW: Weight for supervised vs unsupervised (0.3 = 30% supervised, 70% unsupervised)
        'selection_ratio': 0.5,
        'min_features': 10,
        'use_train_holdout': True,
        'use_all_data': False,  # NEW: Set to True to use ALL data (no sampling)
        # OPTIMIZATION: Better sampling parameters
        'sampling': {
            'sample_percentage': .10,  # 5% of data
            'min_samples': 10000,       # At least 10k samples
            'max_samples': 70000000,      # Cap at 100k
            'stratified': True,         # Use stratified sampling
            'cache_samples': True       # Reuse sample indices
        }
    },
    'memory': {
        'chunk_size': 500,
        'precision': 'float32',
        'max_in_memory_samples': 5000,
        'skip_cleanup': True,
        'fft_batch_size': 10000  # NEW: Batch size for FFT processing
    },
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seq_len': 30
}

# FIXED: Add feature validation function
def is_valid_feature(feature_name):
    """
    Check if a feature should be included in selection.
    Returns True only for normalized, derived, or meta features.
    """
    # Always include mandatory features
    if feature_name in MANDATORY_FEATURES:
        return True
    
    # EXCLUDE HIGH MISSINGNESS FEATURES
    if feature_name == 'exceedance_correlation':
        return False
    
    # Binary indicators are fine
    if feature_name.startswith('I_'):
        return True
    
    # Normalized features are good
    if feature_name.endswith('_norm'):
        return True
    
    # Derived features are good
    derived_patterns = [
        '_snps_', 'tail_dep_', '_exceeds_', '_magnitude_',
        'exceedance_correlation', 'exceedance_pattern_entropy',
        'gap_category', 'days_since_last_activity',
        'multiple_features_exceeded', 'total_features_exceeded'
    ]
    if any(pattern in feature_name for pattern in derived_patterns):
        return True
    
    # Exceedance magnitudes must be normalized
    if feature_name.startswith('E_') and not feature_name.endswith('_norm'):
        return False
    
    # Raw numeric features should be excluded
    raw_features = {
        'total_sessions', 'total_duration', 'total_login_num',
        'total_num', 'total_sum', 'avg_session_duration',
        'avg_login_num', 'avg_num', 'avg_sum', 'session_frequency',
        'delta_session_frequency'
    }
    
    # Check if it's a raw feature (exact match)
    if feature_name in raw_features:
        return False
    
    # Check for raw patterns in feature name
    # This catches variations like total_sum_sessions, etc.
    for raw_feat in raw_features:
        if feature_name == raw_feat or (raw_feat in feature_name and not feature_name.endswith('_norm')):
            return False
    
    # Default: include if it doesn't match raw patterns
    return True

def debug_feature_distribution(data):
    """Debug function to analyze feature types in the data"""
    feature_names = data.get('feature_names', [])
    
    categories = {
        'normalized': 0,
        'indicators': 0,
        'raw_exceedances': 0,
        'normalized_exceedances': 0,
        'derived': 0,
        'raw_base': 0,
        'other': 0
    }
    
    examples = {cat: [] for cat in categories}
    
    for feat in feature_names:
        if feat.endswith('_norm') and not feat.startswith('E_'):
            categories['normalized'] += 1
            if len(examples['normalized']) < 5:
                examples['normalized'].append(feat)
        elif feat.startswith('I_'):
            categories['indicators'] += 1
            if len(examples['indicators']) < 5:
                examples['indicators'].append(feat)
        elif feat.startswith('E_') and not feat.endswith('_norm'):
            categories['raw_exceedances'] += 1
            if len(examples['raw_exceedances']) < 5:
                examples['raw_exceedances'].append(feat)
        elif feat.startswith('E_') and feat.endswith('_norm'):
            categories['normalized_exceedances'] += 1
            if len(examples['normalized_exceedances']) < 5:
                examples['normalized_exceedances'].append(feat)
        elif any(p in feat for p in ['_snps_', 'tail_dep_', '_exceeds_', '_magnitude_']):
            categories['derived'] += 1
            if len(examples['derived']) < 5:
                examples['derived'].append(feat)
        elif feat in ['total_sessions', 'total_duration', 'total_login_num', 
                      'total_num', 'total_sum', 'avg_session_duration',
                      'avg_login_num', 'avg_num', 'avg_sum']:
            categories['raw_base'] += 1
            if len(examples['raw_base']) < 5:
                examples['raw_base'].append(feat)
        else:
            categories['other'] += 1
            if len(examples['other']) < 5:
                examples['other'].append(feat)
    
    print("\n" + "="*60)
    print("FEATURE DISTRIBUTION ANALYSIS")
    print("="*60)
    
    for cat, count in categories.items():
        print(f"\n{cat.replace('_', ' ').title()}: {count}")
        if examples[cat]:
            print(f"  Examples: {', '.join(examples[cat])}")
    
    print("\n" + "="*60)
    
    return categories

def parse_arguments():
    """Parse command line arguments to determine data type"""
    parser = argparse.ArgumentParser(description='Feature selection for gambling data')
    parser.add_argument('--data-type', type=str, choices=VALID_DATA_TYPES, 
                       default='sessions', help='Type of data to process')
    parser.add_argument('--enable-supervised', action='store_true',
                       help='Enable supervised feature selection methods')
    parser.add_argument('--supervised-weight', type=float, default=0.3,
                       help='Weight for supervised methods (0-1)')
    parser.add_argument('--unified', action='store_true', default=True,
                       help='Use unified selection (apply activity features to calendar)')
    parser.add_argument('--window-types', type=str, default='both',
                       choices=['both', 'activity', 'calendar', 'combined', 'unified'],
                       help='Which window types to process')
    args = parser.parse_args()
    return args

# Parse arguments and update config
args = parse_arguments()
CONFIG['data_type'] = args.data_type
CONFIG['feature_selection']['enable_supervised'] = args.enable_supervised
CONFIG['feature_selection']['supervised_weight'] = args.supervised_weight
CONFIG['unified_selection']['enabled'] = args.unified
CONFIG['process_window_types'] = args.window_types

# Update paths based on data type
CONFIG['input_dir'] = os.path.join(CONFIG['base_input_dir'], CONFIG['data_type'])
CONFIG['output_dir'] = os.path.join(CONFIG['base_output_dir'], CONFIG['data_type'])
CONFIG['results_dir'] = os.path.join(CONFIG['base_results_dir'], CONFIG['data_type'])

# Set random seeds with config value
np.random.seed(CONFIG['seed'])
torch.manual_seed(CONFIG['seed'])

# Configure plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_palette('viridis')

def print_memory_usage():
    """Print current memory usage"""
    try:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        print(f"Memory usage: {mem_info.rss / (1024 ** 3):.2f} GB")
    except:
        print("Could not get memory usage")

# Create directories with proper data type hierarchy
print(f"\nCreating directories for data type: {CONFIG['data_type']}")
os.makedirs(CONFIG['output_dir'], exist_ok=True)
os.makedirs(CONFIG['results_dir'], exist_ok=True)

# Create subdirectories for each window type under the data type
for window_type in CONFIG['window_types']:
    os.makedirs(os.path.join(CONFIG['output_dir'], window_type), exist_ok=True)
    os.makedirs(os.path.join(CONFIG['results_dir'], window_type), exist_ok=True)

# Also create combined directory if needed
if CONFIG['process_window_types'] == 'combined':
    os.makedirs(os.path.join(CONFIG['output_dir'], 'combined'), exist_ok=True)
    os.makedirs(os.path.join(CONFIG['results_dir'], 'combined'), exist_ok=True)

# Use system temp directory
TEMP_DIR = tempfile.gettempdir()
print(f"Using temporary directory: {TEMP_DIR}")
print(f"CLEANUP DISABLED to prevent BSOD")
print_memory_usage()

def get_stratified_sample_indices(n_total, labels=None, config=None):
    """Get stratified sample indices based on configuration"""
    if config is None:
        config = CONFIG['feature_selection']['sampling']
    
    # Calculate sample size
    n_samples = min(
        max(int(n_total * config['sample_percentage']), 
            config['min_samples']),
        config['max_samples'],
        n_total  # Don't sample more than we have
    )
    
    print(f"Sampling {n_samples:,} from {n_total:,} total samples ({n_samples/n_total*100:.1f}%)")
    
    if labels is not None and config['stratified']:
        try:
            # FIX: Handle sequence labels properly
            if len(labels.shape) > 1:
                # For sequence data, use the last label or most common label
                # Option 1: Use last timestep label
                stratify_labels = labels[:, -1]
                # Option 2: Use most frequent label in sequence
                # stratify_labels = np.mode(labels, axis=1)[0].flatten()
            else:
                stratify_labels = labels
            
            # Check if stratification is possible
            unique, counts = np.unique(stratify_labels, return_counts=True)
            min_class_count = counts.min()
            
            if min_class_count < 2:
                print(f"Cannot stratify: minimum class has only {min_class_count} samples")
                return np.random.choice(n_total, n_samples, replace=False)
            
            # Additional check: if we're sampling more than 50%, just use random
            ratio = n_samples / n_total
            if ratio >= 0.5:
                print(f"Sample ratio too high ({ratio:.2f}), using random sampling")
                return np.random.choice(n_total, n_samples, replace=False)
            
            indices = np.arange(n_total)
            _, sample_idx = train_test_split(
                indices, 
                test_size=ratio,
                stratify=stratify_labels,
                random_state=CONFIG['seed']
            )
            print(f"Using stratified sampling based on labels (using last timestep)")
            return sample_idx
            
        except Exception as e:
            print(f"Stratified sampling failed: {e}, falling back to random")
    
    # Random sampling
    return np.random.choice(n_total, n_samples, replace=False)

def safe_load_memmap(filepath):
    """Safely load a memory-mapped file"""
    try:
        gc.collect()
        time.sleep(0.1)
        arr = np.load(filepath, mmap_mode='r')
        return arr
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def load_rg_prediction_data(data_path, window_type, split='train'):
    """
    MODIFIED: Load RG prediction data for supervised feature selection
    This loads the seq2seq RG predictions that we'll use instead of manual labels
    """
    rg_data = {}
    
    # Load RG scores sequence
    rg_scores_path = os.path.join(data_path, f"rg_scores_seq_{split}_{window_type}.npy")
    if os.path.exists(rg_scores_path):
        rg_data['rg_scores_seq'] = np.load(rg_scores_path, mmap_mode='r')
        print(f"  Loaded RG scores sequence: {rg_data['rg_scores_seq'].shape}")
    
    # Load RG category distributions
    rg_cats_path = os.path.join(data_path, f"rg_category_dist_seq_{split}_{window_type}.npy")
    if os.path.exists(rg_cats_path):
        rg_data['rg_category_dist_seq'] = np.load(rg_cats_path, mmap_mode='r')
        print(f"  Loaded RG category distributions: {rg_data['rg_category_dist_seq'].shape}")
    
    # Load RG data availability mask
    rg_has_data_path = os.path.join(data_path, f"rg_has_data_seq_{split}_{window_type}.npy")
    if os.path.exists(rg_has_data_path):
        rg_data['rg_has_data_seq'] = np.load(rg_has_data_path, mmap_mode='r')
        print(f"  Loaded RG data mask: {rg_data['rg_has_data_seq'].shape}")
    
    # Load window-level aggregates
    window_mean_path = os.path.join(data_path, f"window_mean_score_{split}_{window_type}.npy")
    if os.path.exists(window_mean_path):
        rg_data['window_mean_score'] = np.load(window_mean_path, mmap_mode='r')
        print(f"  Loaded window mean scores: {rg_data['window_mean_score'].shape}")
    
    window_coverage_path = os.path.join(data_path, f"window_coverage_{split}_{window_type}.npy")
    if os.path.exists(window_coverage_path):
        rg_data['window_coverage'] = np.load(window_coverage_path, mmap_mode='r')
        print(f"  Loaded window coverage: {rg_data['window_coverage'].shape}")
    
    return rg_data

def load_data_for_window_type(window_type, data_type=None):
    """Load data for a specific window type (activity or calendar)"""
    if data_type is None:
        data_type = CONFIG['data_type']
    
    data_path = CONFIG['input_dir']
    
    if not os.path.exists(data_path):
        print(f"Directory not found: {data_path}")
        return None
    
    data = {
        'window_type': window_type,
        'data_type': data_type  # NEW: Track data type
    }
    
    print(f"\nLoading {data_type} data for {window_type} windows...")
    
    # Load arrays for each split
    for split in ['train', 'holdout', 'val', 'test']:
        # Features
        X_path = os.path.join(data_path, f"X_{split}_{window_type}.npy")
        if os.path.exists(X_path):
            arr = safe_load_memmap(X_path)
            if arr is not None:
                data[f'X_{split}'] = arr
                print(f"Loaded {data_type}/{window_type} X_{split}: {arr.shape}")
            else:
                data[f'X_{split}'] = None
        else:
            print(f"Warning: Missing file {X_path}")
            data[f'X_{split}'] = None
        
        # Labels
        L_path = os.path.join(data_path, f"L_{split}_{window_type}.npy")
        if os.path.exists(L_path):
            arr = safe_load_memmap(L_path)
            if arr is not None:
                data[f'L_{split}'] = arr
                print(f"Loaded {data_type}/{window_type} L_{split}: {arr.shape}")
        
        # Customer IDs
        ID_path = os.path.join(data_path, f"ID_{split}_{window_type}.npy")
        if os.path.exists(ID_path):
            arr = safe_load_memmap(ID_path)
            if arr is not None:
                data[f'ID_{split}'] = arr
                print(f"Loaded {data_type}/{window_type} ID_{split}: {arr.shape}")
        
        # Window dates (CSV)
        dates_path = os.path.join(data_path, f"window_dates_{split}_{window_type}.csv")
        if os.path.exists(dates_path):
            data[f'window_dates_{split}'] = pd.read_csv(dates_path)
            print(f"Loaded {data_type}/{window_type} window_dates_{split}: {len(data[f'window_dates_{split}'])} rows")
    
    # Load feature names
    feat_path = os.path.join(data_path, "feature_names.txt")
    if os.path.exists(feat_path):
        with open(feat_path, 'r') as f:
            data['feature_names'] = [line.strip() for line in f]
        print(f"Loaded feature names: {len(data['feature_names'])} features")
    else:
        data['feature_names'] = []
    
    # Load split info if available
    split_info_path = os.path.join(data_path, "split_info.json")
    if os.path.exists(split_info_path):
        with open(split_info_path, 'r') as f:
            data['split_info'] = json.load(f)
    
    # MODIFIED: Load RG prediction data if supervised methods are enabled
    if CONFIG['feature_selection']['enable_supervised']:
        print("\nLoading RG prediction data for supervised feature selection...")
        rg_data = load_rg_prediction_data(CONFIG['input_dir'], window_type, split='train')
        data['rg_data'] = rg_data
    
    # NEW: Pre-compute sample indices for caching
    if CONFIG['feature_selection']['sampling']['cache_samples'] and data.get('X_train') is not None:
        # For RG-based stratification, use RG categories if available
        if 'rg_data' in data and 'rg_category_dist_seq' in data['rg_data']:
            # Use the mode of RG categories for stratification
            rg_cats = data['rg_data']['rg_category_dist_seq']
            # Get the most likely category for each window
            rg_mode = rg_cats.mean(axis=1).argmax(axis=1)  # Average over time, then get mode
            data['cached_sample_indices'] = get_stratified_sample_indices(
                data['X_train'].shape[0], 
                labels=rg_mode
            )
        else:
            # Fallback to no stratification
            data['cached_sample_indices'] = get_stratified_sample_indices(
                data['X_train'].shape[0], 
                labels=None
            )
    
    print_memory_usage()
    return data

def load_combined_data():
    """Load and combine data from both window types"""
    print(f"Loading combined {CONFIG['data_type']} data from both window types...")
    
    # Load both types
    activity_data = load_data_for_window_type('activity')
    calendar_data = load_data_for_window_type('calendar')
    
    if activity_data is None or calendar_data is None:
        print("Error: Could not load both window types")
        return None
    
    # Create combined data structure
    combined_data = {
        'window_type': 'combined',
        'data_type': CONFIG['data_type'],  # NEW: Track data type
        'feature_names': activity_data['feature_names']  # Should be the same
    }
    
    # Combine arrays for each split
    for split in ['train', 'holdout', 'val', 'test']:
        X_act = activity_data.get(f'X_{split}')
        X_cal = calendar_data.get(f'X_{split}')
        
        if X_act is not None and X_cal is not None:
            # Create a view that concatenates along the first axis
            combined_data[f'X_{split}'] = np.concatenate([X_act, X_cal], axis=0)
            print(f"Combined X_{split}: {combined_data[f'X_{split}'].shape}")
            
            # Also combine labels and IDs
            L_act = activity_data.get(f'L_{split}')
            L_cal = calendar_data.get(f'L_{split}')
            if L_act is not None and L_cal is not None:
                combined_data[f'L_{split}'] = np.concatenate([L_act, L_cal], axis=0)
            
            ID_act = activity_data.get(f'ID_{split}')
            ID_cal = calendar_data.get(f'ID_{split}')
            if ID_act is not None and ID_cal is not None:
                combined_data[f'ID_{split}'] = np.concatenate([ID_act, ID_cal], axis=0)
    
    # MODIFIED: Combine RG data if available
    if CONFIG['feature_selection']['enable_supervised']:
        if 'rg_data' in activity_data and 'rg_data' in calendar_data:
            combined_rg_data = {}
            for key in activity_data['rg_data']:
                if key in calendar_data['rg_data']:
                    act_data = activity_data['rg_data'][key]
                    cal_data = calendar_data['rg_data'][key]
                    combined_rg_data[key] = np.concatenate([act_data, cal_data], axis=0)
            combined_data['rg_data'] = combined_rg_data
    
    return combined_data

# LSTM Model
class LSTMFeatureImportance(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.linear(lstm_out[:, -1, :])

def compute_temporal_variance_safe(data_type, data, n_features, feature_indices=None):
    """OPTIMIZED: Compute temporal variance using vectorized operations"""
    print(f"\n*** Computing temporal variance for {data_type} ***\n")
    start_time = datetime.now()
    
    feature_names = data['feature_names']
    
    if feature_indices is not None:
        local_indices = feature_indices
    else:
        local_indices = list(range(len(feature_names)))
    n_local = len(local_indices)
    
    variance_sum = np.zeros(n_local, dtype=np.float64)
    sample_counts = np.zeros(n_local, dtype=np.int64)
    
    datasets = ['X_train']
    if CONFIG['feature_selection']['use_train_holdout'] and 'X_holdout' in data and data['X_holdout'] is not None:
        datasets.append('X_holdout')
        print("Will process train and holdout separately")
    
    chunk_size = CONFIG['memory']['chunk_size']
    
    for dataset_name in datasets:
        X_data = data[dataset_name]
        if X_data is None:
            continue
            
        print(f"Processing {dataset_name}...")
        n_samples = X_data.shape[0]
        
        # OPTIMIZED: Vectorized variance computation
        for start in tqdm(range(0, n_samples, chunk_size), desc=f"{dataset_name} chunks"):
            end = min(start + chunk_size, n_samples)
            
            if start % (chunk_size * 10) == 0:
                gc.collect()
                time.sleep(0.01)
            
            # Get chunk for all selected features at once
            chunk = X_data[start:end, :, local_indices].astype(CONFIG['memory']['precision'], copy=False)
            
            # Compute variance along time axis for all features
            chunk_var = np.var(chunk, axis=1)  # Shape: (batch, n_local)
            
            # Sum up variances
            variance_sum += np.sum(chunk_var, axis=0)
            sample_counts += chunk_var.shape[0]
    
    # FIXED: Compute average variance with single operation
    avg_variance = variance_sum / np.maximum(sample_counts, 1)
    
    # Select top features
    local_pick = np.argsort(avg_variance)[::-1][:n_features]
    selected_indices = [local_indices[i] for i in local_pick]
    selected_features = [feature_names[i] for i in selected_indices]
    
    # Save results and create visualization
    window_type = data.get('window_type', 'unknown')
    out_dir = os.path.join(CONFIG['results_dir'], window_type)
    os.makedirs(out_dir, exist_ok=True)
    
    plt.figure(figsize=(14, 6))
    top_names = [feature_names[local_indices[i]] for i in local_pick[:20]]
    plt.bar(range(len(top_names)), avg_variance[local_pick[:20]])
    plt.xticks(range(len(top_names)), top_names, rotation=90)
    plt.title(f"Top 20 Features by Temporal Variance for {data_type}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'temporal_variance_plot_{data_type}.png'))
    plt.close()
    
    results = {
        'method': 'temporal_variance',
        'data_type': data_type,
        'data_source': data.get('data_type', CONFIG['data_type']),  # NEW
        'window_type': window_type,
        'n_features': n_features,
        'selected_indices': selected_indices,
        'selected_features': selected_features,
        'runtime': str(datetime.now() - start_time)
    }
    
    results_path = os.path.join(out_dir, f'temporal_variance_results_{data_type}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTemporal variance completed in {datetime.now() - start_time}")
    print_memory_usage()
    
    return results

def compute_temporal_correlation_safe(data_type, data, n_features, feature_indices=None):
    """FIXED: Compute temporal redundancy using approximate methods"""
    print(f"\n*** Computing temporal correlation for {data_type} ***\n")
    start_time = datetime.now()
    
    feature_names = data['feature_names']
    
    if feature_indices is not None:
        local_indices = feature_indices
    else:
        local_indices = list(range(len(feature_names)))
    n_local = len(local_indices)
    
    # Get data
    X_data = data.get('X_train')
    if X_data is None:
        print("No training data found for correlation analysis")
        return {'selected_indices': local_indices[:n_features], 
                'selected_features': [feature_names[i] for i in local_indices[:n_features]],
                'data_type': data_type,
                'data_source': data.get('data_type', CONFIG['data_type']),
                'window_type': data.get('window_type', 'unknown')}
    
    # Get sample indices (use cached if available)
    if 'cached_sample_indices' in data:
        sample_idx = data['cached_sample_indices']
        print(f"Using cached sample indices: {len(sample_idx)} samples")
    else:
        labels = data.get('L_train')
        sample_idx = get_stratified_sample_indices(X_data.shape[0], labels=labels)
    
    # FIXED: Use approximate redundancy scores instead of correlation
    print(f"Computing approximate redundancy scores...")
    
    # Reshape data
    sampled_data = X_data[sample_idx, :, :][:, :, local_indices]  # (S, T, F')
    n_samples, seq_len, n_feat = sampled_data.shape
    X_flat = sampled_data.reshape(n_samples * seq_len, n_feat)
    
    # Use random projection if many features
    if n_feat > 100 and n_feat > n_features * 2:
        print(f"Using random projection to reduce from {n_feat} to {min(100, n_features * 2)} dimensions")
        n_components = min(100, n_features * 2)
        rp = GaussianRandomProjection(n_components=n_components, random_state=CONFIG['seed'])
        X_projected = rp.fit_transform(X_flat)
        
        # FIXED: Compute redundancy in projected space properly
        # Normalize projected features
        proj_std = X_projected.std(axis=0, keepdims=True)
        proj_std[proj_std == 0] = 1  # Avoid division by zero
        proj_norm = X_projected / proj_std
        
        # Compute redundancy matrix (average absolute inner product)
        redundancy = np.abs(proj_norm.T @ proj_norm) / proj_norm.shape[0]
        np.fill_diagonal(redundancy, 0)
        
        # Map back to feature space (approximate)
        # This gives us a redundancy score for each original feature
        feature_redundancy = np.abs(rp.components_).T @ redundancy.mean(axis=1)
        avg_redundancy = feature_redundancy
    else:
        # Direct correlation computation
        corr_matrix = np.abs(np.corrcoef(X_flat.T))
        np.fill_diagonal(corr_matrix, 0)
        corr_matrix = np.nan_to_num(corr_matrix, 0)
        avg_redundancy = np.mean(corr_matrix, axis=1)
    
    # Select features with low redundancy
    local_pick = np.argsort(avg_redundancy)[:n_features]
    selected_indices = [local_indices[i] for i in local_pick]
    selected_features = [feature_names[i] for i in selected_indices]
    
    # Visualization
    window_type = data.get('window_type', 'unknown')
    out_dir = os.path.join(CONFIG['results_dir'], window_type)
    
    plt.figure(figsize=(14, 6))
    top_names = [feature_names[local_indices[i]] for i in local_pick[:20]]
    plt.bar(range(len(top_names)), avg_redundancy[local_pick[:20]])
    plt.xticks(range(len(top_names)), top_names, rotation=90)
    plt.title(f"Top 20 Features by Low Redundancy for {data_type}")
    plt.ylabel("Redundancy Score")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'temporal_correlation_plot_{data_type}.png'))
    plt.close()
    
    results = {
        'method': 'temporal_correlation',
        'data_type': data_type,
        'data_source': data.get('data_type', CONFIG['data_type']),  # NEW
        'window_type': window_type,
        'n_features': n_features,
        'selected_indices': selected_indices,
        'selected_features': selected_features,
        'runtime': str(datetime.now() - start_time)
    }
    
    results_path = os.path.join(out_dir, f'temporal_correlation_results_{data_type}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTemporal correlation completed in {datetime.now() - start_time}")
    print_memory_usage()
    
    return results

def compute_fourier_analysis_safe(data_type, data, n_features, feature_indices=None):
    """FIXED: Compute Fourier analysis with proper normalization"""
    print(f"\n*** Computing Fourier analysis for {data_type} ***\n")
    start_time = datetime.now()
    
    feature_names = data['feature_names']
    
    if feature_indices is not None:
        local_indices = feature_indices
    else:
        local_indices = list(range(len(feature_names)))
    n_local = len(local_indices)
    
    X_data = data.get('X_train')
    if X_data is None:
        print("No training data found for Fourier analysis")
        return {'selected_indices': local_indices[:n_features], 
                'selected_features': [feature_names[i] for i in local_indices[:n_features]],
                'data_type': data_type,
                'data_source': data.get('data_type', CONFIG['data_type']),
                'window_type': data.get('window_type', 'unknown')}
    
    # Get sample indices
    if 'cached_sample_indices' in data:
        sample_idx = data['cached_sample_indices']
        print(f"Using cached sample indices: {len(sample_idx)} samples")
    else:
        labels = data.get('L_train')
        sample_idx = get_stratified_sample_indices(X_data.shape[0], labels=labels)
    
    # OPTIMIZED: Batch FFT processing
    batch_size = CONFIG['memory']['fft_batch_size']
    n_samples = len(sample_idx)
    seq_len = X_data.shape[1]
    
    power_spectrum = np.zeros(n_local, dtype=np.float64)
    total_samples_processed = 0  # FIXED: Track actual samples processed
    
    print(f"Processing FFT in batches of {batch_size}...")
    
    for i in tqdm(range(0, n_samples, batch_size), desc="FFT batches"):
        batch_end = min(i + batch_size, n_samples)
        batch_idx = sample_idx[i:batch_end]
        actual_batch_size = len(batch_idx)  # FIXED: Track actual batch size
        
        # Get batch data
        batch_data = X_data[batch_idx][:, :, local_indices]  # (B, T, F')
        
        # Remove DC component
        batch_data = batch_data - np.mean(batch_data, axis=1, keepdims=True)
        
        # Compute FFT for entire batch at once
        fft_batch = np.fft.rfft(batch_data, axis=1)
        power_batch = np.abs(fft_batch[:, 1:])**2  # Remove DC, shape: (B, T/2, F')
        
        # Sum power across frequencies and average across batch
        # FIXED: Weight by actual batch size
        power_spectrum += np.sum(np.mean(power_batch, axis=0), axis=0) * actual_batch_size
        total_samples_processed += actual_batch_size
        
        if i % (batch_size * 5) == 0:
            gc.collect()
    
    # FIXED: Average power using correct total count
    power_spectrum /= total_samples_processed
    
    # Select top features
    local_pick = np.argsort(power_spectrum)[::-1][:n_features]
    selected_indices = [local_indices[i] for i in local_pick]
    selected_features = [feature_names[i] for i in selected_indices]
    
    # Visualization
    window_type = data.get('window_type', 'unknown')
    out_dir = os.path.join(CONFIG['results_dir'], window_type)
    
    plt.figure(figsize=(14, 6))
    top_names = [feature_names[local_indices[i]] for i in local_pick[:20]]
    plt.bar(range(len(top_names)), power_spectrum[local_pick[:20]])
    plt.xticks(range(len(top_names)), top_names, rotation=90)
    plt.title(f"Top 20 Features by FFT Power for {data_type}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'fourier_analysis_plot_{data_type}.png'))
    plt.close()
    
    results = {
        'method': 'fourier_analysis',
        'data_type': data_type,
        'data_source': data.get('data_type', CONFIG['data_type']),  # NEW
        'window_type': window_type,
        'n_features': n_features,
        'selected_indices': selected_indices,
        'selected_features': selected_features,
        'runtime': str(datetime.now() - start_time)
    }
    
    results_path = os.path.join(out_dir, f'fourier_analysis_results_{data_type}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nFourier analysis completed in {datetime.now() - start_time}")
    print_memory_usage()
    
    return results

def compute_temporal_importance_safe(data_type, data, n_features, feature_indices=None):
    """Compute temporal importance using LSTM with optimized sampling and training"""
    print(f"\n*** Computing temporal importance for {data_type} ***\n")
    start_time = datetime.now()
    
    feature_names = data['feature_names']
    
    if feature_indices is not None:
        local_indices = feature_indices
    else:
        local_indices = list(range(len(feature_names)))
    n_local = len(local_indices)
    
    # Add early check for too few features
    if n_local < 2:
        print(f"Too few features ({n_local}) for LSTM importance. Returning all features.")
        return {
            'selected_indices': local_indices[:n_features], 
            'selected_features': [feature_names[i] for i in local_indices[:n_features]],
            'data_type': data_type,
            'data_source': data.get('data_type', CONFIG['data_type']),
            'window_type': data.get('window_type', 'unknown')
        }
    
    X_data = data.get('X_train')
    if X_data is None:
        print("No training data found for temporal importance")
        return {'selected_indices': local_indices[:n_features], 
                'selected_features': [feature_names[i] for i in local_indices[:n_features]],
                'data_type': data_type,
                'data_source': data.get('data_type', CONFIG['data_type']),
                'window_type': data.get('window_type', 'unknown')}
    
    # OPTIMIZED: Use smaller sample for better learning
    lstm_config = {
        'max_samples': 150000,  # Reduced from 426k
        'epochs': 50,          # Increased from 5
        'batch_size': 64,      # Slightly larger batches for stability
        'learning_rate': 0.001,
        'early_stopping_patience': 3,
        'min_delta': 0.001    # Stop if improvement less than 0.1%
    }
    
    # Get appropriate sample
    if 'cached_sample_indices' in data:
        sample_idx = data['cached_sample_indices']
        if len(sample_idx) > lstm_config['max_samples']:
            # Subsample from the cached indices
            lstm_subsample = np.random.choice(len(sample_idx), lstm_config['max_samples'], replace=False)
            sample_idx = sample_idx[lstm_subsample]
            print(f"Using {len(sample_idx):,} samples for LSTM (subsampled for better learning)")
        else:
            print(f"Using cached sample indices: {len(sample_idx):,} samples")
    else:
        labels = data.get('L_train')
        n_available = X_data.shape[0]
        n_samples = min(n_available, lstm_config['max_samples'])
        sample_idx = np.random.choice(n_available, n_samples, replace=False)
        print(f"Using {n_samples:,} random samples for LSTM")
    
    # Prepare data - NO NORMALIZATION since it's already normalized
    sampled_data = []
    for idx_pos, feat_idx in enumerate(local_indices):
        feat_data = X_data[sample_idx, :, feat_idx:feat_idx+1]
        sampled_data.append(feat_data)
    
    X_sampled = np.concatenate(sampled_data, axis=2)
    
    # ADD: Check for NaN/Inf in sampled data
    if np.any(np.isnan(X_sampled)) or np.any(np.isinf(X_sampled)):
        print("Warning: NaN or Inf values detected in sampled data. Cleaning...")
        X_sampled = np.nan_to_num(X_sampled, nan=0.0, posinf=0.0, neginf=0.0)
    
    # ADD: Check variance to avoid constant features
    feature_vars = np.var(X_sampled.reshape(-1, n_local), axis=0)
    if np.all(feature_vars < 1e-10):
        print("Warning: All features have near-zero variance. Using random selection.")
        selected_indices = local_indices[:n_features]
        selected_features = [feature_names[i] for i in selected_indices]
        return {
            'method': 'temporal_importance',
            'data_type': data_type,
            'data_source': data.get('data_type', CONFIG['data_type']),
            'window_type': data.get('window_type', 'unknown'),
            'n_features': n_features,
            'selected_indices': selected_indices,
            'selected_features': selected_features,
            'runtime': str(datetime.now() - start_time)
        }
    
    # Convert to tensors
    device = torch.device(CONFIG['device'] if torch.cuda.is_available() else 'cpu')
    
    X_tensor = torch.FloatTensor(X_sampled).to(device)
    
    # Create temporal targets (predict future mean)
    targets = torch.mean(X_tensor[:, 15:, :], dim=1).to(device)
    
    # ADD: Check for NaN in targets
    if torch.isnan(targets).any():
        print("Warning: NaN in targets. Using alternative target computation.")
        targets = torch.mean(X_tensor[:, -5:, :], dim=1).to(device)
        if torch.isnan(targets).any():
            # Last resort: use last timestep
            targets = X_tensor[:, -1, :].to(device)
    
    # Build model with appropriate size
    hidden_dim = min(64, n_local // 2) if n_local > 64 else 32
    model = LSTMFeatureImportance(n_local, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lstm_config['learning_rate'])
    criterion = nn.MSELoss()
    
    # Split data for validation
    n_train = int(0.8 * len(X_tensor))
    train_indices = torch.randperm(len(X_tensor))[:n_train]
    val_indices = torch.randperm(len(X_tensor))[n_train:]
    
    X_train = X_tensor[train_indices]
    y_train = targets[train_indices]
    X_val = X_tensor[val_indices]
    y_val = targets[val_indices]
    
    # Training with early stopping
    print(f"Training LSTM model ({lstm_config['epochs']} epochs with early stopping)...")
    model.train()
    batch_size = lstm_config['batch_size']
    best_val_loss = float('inf')
    patience_counter = 0
    
    # CRITICAL FIX: Initialize best_model_state
    best_model_state = model.state_dict()  # Save initial state
    
    train_losses = []
    val_losses = []
    
    for epoch in range(lstm_config['epochs']):
        # Training
        perm = torch.randperm(len(X_train))
        epoch_loss = 0
        n_batches = 0
        
        for i in range(0, len(X_train), batch_size):
            batch_idx = perm[i:i+batch_size]
            batch_x = X_train[batch_idx]
            batch_y = y_train[batch_idx]
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # ADD: Check for NaN loss
            if torch.isnan(loss):
                print(f"Warning: NaN loss at epoch {epoch+1}, batch {i//batch_size}")
                continue
            
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        if n_batches > 0:
            avg_train_loss = epoch_loss / n_batches
        else:
            avg_train_loss = float('nan')
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val).item()
        val_losses.append(val_loss)
        model.train()
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        # Early stopping - FIXED: Handle NaN losses
        if not np.isnan(val_loss) and val_loss < best_val_loss - lstm_config['min_delta']:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model state
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= lstm_config['early_stopping_patience']:
                print(f"Early stopping at epoch {epoch+1}")
                model.load_state_dict(best_model_state)
                break
    
    # If all losses were NaN, use random importance
    if all(np.isnan(loss) for loss in train_losses):
        print("Warning: All training losses were NaN. Using random feature importance.")
        feature_losses = np.random.rand(n_local)
    else:
        # Compute feature importance with best model
        model.eval()
        test_n = min(1000, len(X_tensor) // 2)  # Use more test samples
        
        with torch.no_grad():
            base_out = model(X_tensor[:test_n])
            base_loss = criterion(base_out, targets[:test_n]).item()
        
        feature_losses = np.zeros(n_local)
        
        # More robust importance calculation
        n_permutations = 5  # Multiple permutations for stability
        
        for idx_pos in tqdm(range(n_local), desc="Feature importance"):
            importance_scores = []
            
            for _ in range(n_permutations):
                with torch.no_grad():
                    pert = X_tensor[:test_n].clone()
                    perm = torch.randperm(pert.shape[1])
                    pert[:, :, idx_pos] = pert[:, :, idx_pos][:, perm]
                    
                    out2 = model(pert)
                    diff = criterion(out2, targets[:test_n]).item() - base_loss
                    if not np.isnan(diff):
                        importance_scores.append(diff)
                    
                    del pert
            
            if importance_scores:
                feature_losses[idx_pos] = np.mean(importance_scores)
            else:
                feature_losses[idx_pos] = 0.0
            
            if idx_pos % 20 == 0 and device.type == 'cuda':
                torch.cuda.empty_cache()
    
    # Rest remains the same...
    local_pick = np.argsort(feature_losses)[::-1][:n_features]
    selected_indices = [local_indices[i] for i in local_pick]
    selected_features = [feature_names[i] for i in selected_indices]
    
    # Visualization
    plt.figure(figsize=(14, 6))
    names20 = [feature_names[local_indices[i]] for i in local_pick[:20]]
    plt.bar(range(len(names20)), feature_losses[local_pick[:20]])
    plt.xticks(range(len(names20)), names20, rotation=90)
    plt.title(f"Top 20 Features by LSTM Importance for {data_type}")
    plt.tight_layout()
    
    window_type = data.get('window_type', 'unknown')
    out_dir = os.path.join(CONFIG['results_dir'], window_type)
    plt.savefig(os.path.join(out_dir, f'temporal_importance_plot_{data_type}.png'))
    plt.close()
    
    results = {
        'method': 'temporal_importance',
        'data_type': data_type,
        'data_source': data.get('data_type', CONFIG['data_type']),  # NEW
        'window_type': window_type,
        'n_features': n_features,
        'selected_indices': selected_indices,
        'selected_features': selected_features,
        'lstm_config': lstm_config,
        'final_train_loss': train_losses[-1] if train_losses else None,
        'final_val_loss': val_losses[-1] if val_losses else None,
        'epochs_trained': len(train_losses),
        'runtime': str(datetime.now() - start_time)
    }
    
    results_path = os.path.join(out_dir, f'temporal_importance_results_{data_type}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nLSTM importance completed in {datetime.now() - start_time}")
    if train_losses and not all(np.isnan(loss) for loss in train_losses):
        print(f"Final losses - Train: {train_losses[-1]:.4f}, Val: {val_losses[-1]:.4f}")
    else:
        print("Warning: Used fallback importance due to NaN losses")
    print_memory_usage()
    
    del model, X_tensor, targets
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
    
    return results

def compute_incremental_pca_features(data_type, data, n_features, feature_indices=None):
    """NEW: Compute feature importance using Incremental PCA"""
    print(f"\n*** Computing Incremental PCA for {data_type} ***\n")
    start_time = datetime.now()
    
    feature_names = data['feature_names']
    
    if feature_indices is not None:
        local_indices = feature_indices
    else:
        local_indices = list(range(len(feature_names)))
    n_local = len(local_indices)
    
    X_data = data.get('X_train')
    if X_data is None:
        print("No training data found for PCA analysis")
        return {'selected_indices': local_indices[:n_features], 
                'selected_features': [feature_names[i] for i in local_indices[:n_features]],
                'data_type': data_type,
                'data_source': data.get('data_type', CONFIG['data_type']),
                'window_type': data.get('window_type', 'unknown')}
    
    # Prepare for incremental PCA
    n_samples, seq_len, _ = X_data.shape
    n_components = min(50, n_local // 2)  # Use 50 components or half the features
    batch_size = 5000
    
    print(f"Running Incremental PCA with {n_components} components...")
    
    # Initialize PCA
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    
    # Fit PCA in batches
    # Note: keeping original behavior of using (window, time-step) as samples
    # as changing this would alter functionality
    for start in tqdm(range(0, n_samples, batch_size // seq_len), desc="PCA batches"):
        end = min(start + batch_size // seq_len, n_samples)
        
        # Get batch and reshape
        batch = X_data[start:end, :, local_indices]  # (B, T, F')
        batch_flat = batch.reshape(-1, n_local)  # (B*T, F')
        
        # Partial fit
        ipca.partial_fit(batch_flat)
        
        if start % (batch_size * 5) == 0:
            gc.collect()
    
    # Compute feature importance from PCA components
    feature_importance = np.sum(np.abs(ipca.components_), axis=0)
    
    # Select top features
    local_pick = np.argsort(feature_importance)[::-1][:n_features]
    selected_indices = [local_indices[i] for i in local_pick]
    selected_features = [feature_names[i] for i in selected_indices]
    
    # Visualization
    window_type = data.get('window_type', 'unknown')
    out_dir = os.path.join(CONFIG['results_dir'], window_type)
    
    plt.figure(figsize=(14, 6))
    names20 = [feature_names[local_indices[i]] for i in local_pick[:20]]
    plt.bar(range(len(names20)), feature_importance[local_pick[:20]])
    plt.xticks(range(len(names20)), names20, rotation=90)
    plt.title(f"Top 20 Features by PCA Importance for {data_type}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'pca_importance_plot_{data_type}.png'))
    plt.close()
    
    # Also plot explained variance
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(ipca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title(f'PCA Explained Variance for {data_type}')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'pca_variance_plot_{data_type}.png'))
    plt.close()
    
    results = {
        'method': 'incremental_pca',
        'data_type': data_type,
        'data_source': data.get('data_type', CONFIG['data_type']),  # NEW
        'window_type': window_type,
        'n_features': n_features,
        'selected_indices': selected_indices,
        'selected_features': selected_features,
        'explained_variance_ratio': ipca.explained_variance_ratio_.tolist(),
        'runtime': str(datetime.now() - start_time)
    }
    
    results_path = os.path.join(out_dir, f'pca_results_{data_type}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nIncremental PCA completed in {datetime.now() - start_time}")
    print(f"Explained variance with {n_components} components: {np.sum(ipca.explained_variance_ratio_):.2%}")
    print_memory_usage()
    
    return results

def prepare_data_for_supervised(data, feature_indices, sample_indices=None):
    """
    FIXED: Prepare data properly for supervised methods
    Ensures correct shape and handling of sequence data
    """
    X_data = data.get('X_train')
    if X_data is None:
        return None, None
    
    # Use provided sample indices or get all
    if sample_indices is None:
        if 'cached_sample_indices' in data:
            sample_indices = data['cached_sample_indices']
        else:
            sample_indices = np.arange(X_data.shape[0])
    
    # Get the subset of data
    # X_data shape: (n_samples, seq_len, n_features)
    # We need to extract specific features
    X_subset = X_data[sample_indices, :, :][:, :, feature_indices]  # (n_samples, seq_len, n_selected_features)
    
    # Get labels if available
    labels = None
    if 'label_data' in data and data['label_data']:
        # Assuming label_data has appropriate keys
        if 'labels' in data['label_data']:
            labels = data['label_data']['labels'][sample_indices]
        elif 'risk_labels' in data['label_data']:
            labels = data['label_data']['risk_labels'][sample_indices]
    
    return X_subset, labels

# MODIFIED SUPERVISED METHODS TO USE RG PREDICTIONS

def compute_rg_score_correlation_features(data_type, data, n_features, feature_indices=None):
    """
    MODIFIED: Compute correlation between features and RG scores
    Uses seq2seq RG scores instead of manual labels
    """
    print(f"\n*** Computing RG score correlation for {data_type} ***\n")
    from datetime import datetime
    start_time = datetime.now()
    
    feature_names = data['feature_names']
    
    if feature_indices is not None:
        local_indices = feature_indices
    else:
        local_indices = list(range(len(feature_names)))
    n_local = len(local_indices)
    
    # Get features and RG data
    X_data = data.get('X_train')
    rg_data = data.get('rg_data', {})
    
    if X_data is None or not rg_data:
        print("Warning: No training data or RG data available")
        return None
    
    rg_scores_seq = rg_data.get('rg_scores_seq')
    rg_has_data = rg_data.get('rg_has_data_seq')
    
    if rg_scores_seq is None or rg_has_data is None:
        print("Warning: No RG score data found")
        return None
    
    # Sample data if too large
    if 'cached_sample_indices' in data:
        sample_idx = data['cached_sample_indices']
    else:
        sample_idx = np.random.choice(X_data.shape[0], 
                                    min(10000, X_data.shape[0]), 
                                    replace=False)
    
    # Compute correlation scores
    corr_scores = np.zeros(n_local)
    
    for idx_pos, feat_idx in enumerate(local_indices):
        try:
            # Get feature data
            feat_data = X_data[sample_idx, :, feat_idx]
            
            # Get corresponding RG scores
            rg_scores = rg_scores_seq[sample_idx]
            rg_mask = rg_has_data[sample_idx]
            
            # Compute correlation at each timestep
            correlations = []
            for t in range(feat_data.shape[1]):
                # Only use timesteps with RG data
                mask_t = rg_mask[:, t] & (rg_scores[:, t] > 0)
                if mask_t.sum() > 10:
                    # Compute Spearman correlation
                    from scipy.stats import spearmanr
                    corr, _ = spearmanr(feat_data[mask_t, t], rg_scores[mask_t, t])
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
            
            if correlations:
                corr_scores[idx_pos] = np.mean(correlations)
                
        except Exception as e:
            print(f"Error computing RG correlation for feature {idx_pos}: {e}")
            continue
    
    # Select top features
    local_pick = np.argsort(corr_scores)[::-1][:n_features]
    selected_indices = [local_indices[i] for i in local_pick]
    selected_features = [feature_names[i] for i in selected_indices]
    
    # Visualization
    window_type = data.get('window_type', 'unknown')
    out_dir = os.path.join(CONFIG['results_dir'], window_type)
    
    plt.figure(figsize=(14, 6))
    top_names = [feature_names[local_indices[i]] for i in local_pick[:20]]
    plt.bar(range(len(top_names)), corr_scores[local_pick[:20]])
    plt.xticks(range(len(top_names)), top_names, rotation=90)
    plt.title(f"Top 20 Features by RG Score Correlation for {data_type}")
    plt.ylabel("Correlation Score")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'rg_score_correlation_plot_{data_type}.png'))
    plt.close()
    
    results = {
        'method': 'rg_score_correlation',
        'data_type': data_type,
        'window_type': window_type,
        'n_features': n_features,
        'selected_indices': selected_indices,
        'selected_features': selected_features,
        'scores': corr_scores[local_pick].tolist(),
        'runtime': str(datetime.now() - start_time)
    }
    
    results_path = os.path.join(out_dir, f'rg_score_correlation_results_{data_type}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def compute_rg_category_importance_features(data_type, data, n_features, feature_indices=None):
    """
    MODIFIED: Fast correlation between feature dynamics and RG distribution changes
    Uses distribution divergence with STRATIFIED SAMPLING
    """
    print(f"\n*** Computing RG category importance for {data_type} ***\n")
    from datetime import datetime
    from scipy.spatial.distance import jensenshannon
    from scipy.stats import spearmanr, entropy
    from sklearn.model_selection import train_test_split
    
    start_time = datetime.now()
    
    feature_names = data['feature_names']
    
    if feature_indices is not None:
        local_indices = feature_indices
    else:
        local_indices = list(range(len(feature_names)))
    n_local = len(local_indices)
    
    print(f"  Processing {n_local} features with class-balanced approach...")
    
    # Get features and RG data
    X_data = data.get('X_train')
    rg_data = data.get('rg_data', {})
    
    if X_data is None or not rg_data:
        print("Warning: No data available for RG category importance")
        return None
    
    rg_cats = rg_data.get('rg_category_dist_seq')
    rg_has_data = rg_data.get('rg_has_data_seq')
    
    if rg_cats is None:
        print("Warning: No RG category data found")
        return None
    
    # Get valid windows with decent coverage
    window_coverage = rg_has_data.mean(axis=1)
    valid_windows = window_coverage > 0.5
    valid_indices = np.where(valid_windows)[0]
    
    # CRITICAL: Handle class imbalance properly
    rg_dominant = rg_cats[valid_windows].mean(axis=1).argmax(axis=1)
    
    # Count samples per category
    unique_cats, cat_counts = np.unique(rg_dominant, return_counts=True)
    print(f"  RG category distribution in valid windows:")
    for cat, count in zip(unique_cats, cat_counts):
        print(f"    Category {cat}: {count} ({count/len(rg_dominant)*100:.1f}%)")
    
    # BALANCED SAMPLING STRATEGY
    min_samples_per_class = 500  # Ensure at least 500 samples per class if available
    max_total_samples = 50000
    
    sample_idx = []
    
    # First, get indices for each category
    category_indices = {}
    for cat in range(5):  # 0-4 categories
        cat_mask = rg_dominant == cat
        if cat_mask.sum() > 0:
            category_indices[cat] = valid_indices[cat_mask]
    
    # Sample from each category
    samples_per_category = {}
    for cat, indices in category_indices.items():
        n_available = len(indices)
        
        if n_available <= min_samples_per_class * 2:
            # Use all samples from minority classes
            samples_per_category[cat] = indices
            print(f"    Using all {n_available} samples from category {cat}")
        else:
            # Sample from majority classes
            n_to_sample = min(min_samples_per_class * 5, n_available)  # 5x for majority
            samples_per_category[cat] = np.random.choice(indices, n_to_sample, replace=False)
            print(f"    Sampling {n_to_sample} from {n_available} samples in category {cat}")
    
    # Combine samples
    for cat in sorted(samples_per_category.keys()):
        sample_idx.extend(samples_per_category[cat])
    
    sample_idx = np.array(sample_idx)
    np.random.shuffle(sample_idx)  # Shuffle to mix categories
    
    # Limit total samples if needed
    if len(sample_idx) > max_total_samples:
        sample_idx = sample_idx[:max_total_samples]
    
    print(f"  Final sample size: {len(sample_idx)} (balanced across categories)")
    
    # Verify balance in final sample
    final_cats = rg_cats[sample_idx].mean(axis=1).argmax(axis=1)
    final_unique, final_counts = np.unique(final_cats, return_counts=True)
    print(f"  Final sample distribution:")
    for cat, count in zip(final_unique, final_counts):
        print(f"    Category {cat}: {count} ({count/len(final_cats)*100:.1f}%)")
    
    # OPTIMIZED: Focus on transitions and changes rather than static states
    importance_scores = np.zeros(n_local)
    
    # Pre-compute category transitions for all samples
    print("  Finding category transitions...")
    transition_info = []
    for idx in sample_idx:
        cat_sequence = rg_cats[idx].argmax(axis=1)
        transitions = []
        for t in range(1, 30):
            if rg_has_data[idx, t] and rg_has_data[idx, t-1]:
                if cat_sequence[t] != cat_sequence[t-1]:
                    transitions.append((t, cat_sequence[t-1], cat_sequence[t]))
        if transitions:
            transition_info.append((idx, transitions))
    
    print(f"  Found {len(transition_info)} windows with category transitions")
    
    # Process features
    batch_size = 10
    
    for batch_start in range(0, n_local, batch_size):
        batch_end = min(batch_start + batch_size, n_local)
        batch_indices = local_indices[batch_start:batch_end]
        
        print(f"\r  Processing features {batch_start+1}-{batch_end}/{n_local}", end='', flush=True)
        
        for batch_idx, feat_idx in enumerate(batch_indices):
            scores = []
            
            # Score 1: Feature change during transitions
            for idx, transitions in transition_info:
                feat_data = X_data[idx, :, feat_idx]
                for t, cat_from, cat_to in transitions:
                    if t >= 2 and t < 28:
                        # Feature change around transition
                        before = np.mean(feat_data[t-2:t])
                        after = np.mean(feat_data[t:t+2])
                        change = abs(after - before)
                        # Weight by transition importance (rare transitions = higher weight)
                        weight = 1.0 + (4 - min(cat_from, cat_to))  # Higher categories get more weight
                        scores.append(change * weight)
            
            # Score 2: Feature correlation with RG uncertainty
            uncertainty_scores = []
            for i, idx in enumerate(sample_idx[:1000]):  # Subsample for speed
                if rg_has_data[idx].sum() > 15:
                    # RG uncertainty (entropy)
                    rg_uncertainty = []
                    feat_vals = []
                    for t in range(30):
                        if rg_has_data[idx, t]:
                            dist = rg_cats[idx, t] / (rg_cats[idx, t].sum() + 1e-10)
                            ent = -np.sum(dist * np.log(dist + 1e-10))
                            rg_uncertainty.append(ent)
                            feat_vals.append(X_data[idx, t, feat_idx])
                    
                    if len(rg_uncertainty) > 10:
                        corr, _ = spearmanr(feat_vals, rg_uncertainty)
                        if not np.isnan(corr):
                            uncertainty_scores.append(abs(corr))
            
            # Combine scores
            transition_score = np.mean(scores) if scores else 0
            uncertainty_score = np.mean(uncertainty_scores) if uncertainty_scores else 0
            importance_scores[batch_start + batch_idx] = 0.6 * transition_score + 0.4 * uncertainty_score
    
    print()  # New line after progress
    
    # Select top features
    local_pick = np.argsort(importance_scores)[::-1][:n_features]
    selected_indices = [local_indices[i] for i in local_pick]
    selected_features = [feature_names[i] for i in selected_indices]
    
    # Visualization
    window_type = data.get('window_type', 'unknown')
    out_dir = os.path.join(CONFIG['results_dir'], window_type)
    
    plt.figure(figsize=(14, 6))
    top_names = [feature_names[local_indices[i]] for i in local_pick[:20]]
    plt.bar(range(len(top_names)), importance_scores[local_pick[:20]])
    plt.xticks(range(len(top_names)), top_names, rotation=90)
    plt.title(f"Top 20 Features by RG Distribution Correlation for {data_type}")
    plt.ylabel("Importance Score")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'rg_category_importance_plot_{data_type}.png'))
    plt.close()
    
    results = {
        'method': 'rg_category_importance',
        'data_type': data_type,
        'window_type': window_type,
        'n_features': n_features,
        'selected_indices': selected_indices,
        'selected_features': selected_features,
        'scores': importance_scores[local_pick].tolist(),
        'runtime': str(datetime.now() - start_time),
        'n_samples_used': len(sample_idx),
        'n_transitions_found': len(transition_info)
    }
    
    results_path = os.path.join(out_dir, f'rg_category_importance_results_{data_type}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nRG category importance completed in {datetime.now() - start_time}")
    print(f"  Used {len(sample_idx)} balanced samples")
    print(f"  Found {len(transition_info)} windows with transitions")
    print(f"  Top feature: {selected_features[0] if selected_features else 'None'}")
    
    return results

# FIXED: Main feature selection function with feature validation
def run_feature_selection_methods_safe(data_type, data):
    """Run feature selection with safe methods - ENHANCED with feature validation"""
    print(f"\n{'-'*80}\nProcessing {CONFIG['data_type']}/{data_type} data\n{'-'*80}")
    feature_names = data['feature_names']
    
    # CRITICAL: Debug feature distribution before processing
    debug_feature_distribution(data)
    
    # CRITICAL: Validate features before building groups
    valid_feature_names = [f for f in feature_names if is_valid_feature(f)]
    invalid_features = [f for f in feature_names if not is_valid_feature(f)]
    
    print(f"\nFeature validation:")
    print(f"  Total features: {len(feature_names)}")
    print(f"  Valid features: {len(valid_feature_names)}")
    print(f"  Excluded raw features: {len(invalid_features)}")
    
    if invalid_features:
        print(f"\nExcluded features (first 20):")
        for feat in invalid_features[:20]:
            print(f"    - {feat}")
        if len(invalid_features) > 20:
            print(f"    ... and {len(invalid_features) - 20} more")
    
    # Build index lists for each group using ONLY valid features
    group_indices = {}
    for grp, patterns in FEATURE_GROUPS.items():
        idxs = []
        for i, name in enumerate(feature_names):
            if not is_valid_feature(name):
                continue  # Skip invalid features
            
            # Check if feature matches any pattern in the group
            for pat in patterns:
                matched = False
                
                if pat.startswith('_') and pat.endswith('_'):
                    # Pattern for "contains"
                    if pat[1:-1] in name:
                        matched = True
                elif pat.endswith('_'):
                    # Pattern for "starts with"
                    if name.startswith(pat):
                        matched = True
                elif pat.startswith('_'):
                    # Pattern for "ends with"
                    if name.endswith(pat[1:]):
                        matched = True
                else:
                    # Exact match or contains
                    if pat == name or pat in name:
                        matched = True
                
                if matched:
                    idxs.append(i)
                    break
        
        # Remove duplicates while preserving order
        idxs = list(dict.fromkeys(idxs))
        print(f"Group '{grp}': {len(idxs)} valid features")
        group_indices[grp] = idxs
    
    all_group_selected = {}
    supervised_results_by_group = {}  # NEW: Store supervised results
    
    for grp, idxs in group_indices.items():
        if len(idxs) == 0:
            print(f"Skipping group '{grp}' - no features found")
            continue
            
        print(f"\n>>> Processing group '{grp}'")
        n_feats = max(
            CONFIG['feature_selection']['min_features'],
            int(len(idxs) * CONFIG['feature_selection']['selection_ratio'])
        )
        
        gc.collect()
        time.sleep(0.5)
        
        # Run unsupervised methods (UNCHANGED)
        res_var = compute_temporal_variance_safe(
            data_type + f"_{grp}", data, n_feats, feature_indices=idxs)
        gc.collect()
        time.sleep(0.5)
        
        res_corr = compute_temporal_correlation_safe(
            data_type + f"_{grp}", data, n_feats, feature_indices=idxs)
        gc.collect()
        time.sleep(0.5)
        
        res_fft = compute_fourier_analysis_safe(
            data_type + f"_{grp}", data, n_feats, feature_indices=idxs)
        gc.collect()
        time.sleep(0.5)
        
        res_imp = compute_temporal_importance_safe(
            data_type + f"_{grp}", data, n_feats, feature_indices=idxs)
        gc.collect()
        time.sleep(0.5)
        
        # NEW: Add PCA analysis
        res_pca = compute_incremental_pca_features(
            data_type + f"_{grp}", data, n_feats, feature_indices=idxs)
        gc.collect()
        time.sleep(0.5)
        
        # Store unsupervised results
        all_group_selected[grp] = {
            'variance': res_var['selected_indices'],
            'correlation': res_corr['selected_indices'],
            'fourier': res_fft['selected_indices'],
            'importance': res_imp['selected_indices'],
            'pca': res_pca['selected_indices'],  # NEW
        }
        
        # MODIFIED: Run RG-based supervised methods if enabled
        if CONFIG['feature_selection']['enable_supervised'] and 'rg_data' in data and data['rg_data']:
            print(f"\n>>> Running RG-based supervised methods for group '{grp}'")
            supervised_results = []
            
            # RG Score Correlation
            if 'rg_score_correlation' in CONFIG['feature_selection']['supervised_methods']:
                try:
                    res_rg_score = compute_rg_score_correlation_features(
                        data_type + f"_{grp}", data, n_feats, feature_indices=idxs
                    )
                    if res_rg_score and 'selected_indices' in res_rg_score:
                        supervised_results.append(res_rg_score)
                except Exception as e:
                    print(f"Error in RG score correlation: {e}")
                gc.collect()
                time.sleep(0.5)
            
            # RG Category Importance
            if 'rg_category_importance' in CONFIG['feature_selection']['supervised_methods']:
                try:
                    res_rg_cat = compute_rg_category_importance_features(
                        data_type + f"_{grp}", data, n_feats, feature_indices=idxs
                    )
                    if res_rg_cat and 'selected_indices' in res_rg_cat:
                        supervised_results.append(res_rg_cat)
                except Exception as e:
                    print(f"Error in RG category importance: {e}")
                gc.collect()
                time.sleep(0.5)
            
            supervised_results_by_group[grp] = supervised_results
    
    # NEW: Combine supervised and unsupervised results
    combined_indices = []
    combined_names = []
    combined_scores = {}
    
    if CONFIG['feature_selection']['enable_supervised'] and supervised_results_by_group:
        print(f"\n>>> Combining supervised and unsupervised results")
        alpha = 1 - CONFIG['feature_selection']['supervised_weight']  # Weight for unsupervised
        
        for grp in FEATURE_GROUPS:
            if grp in all_group_selected:
                # Get unsupervised results for this group
                unsupervised_results = all_group_selected[grp]
                
                # Get supervised results for this group
                supervised_results = supervised_results_by_group.get(grp, [])
                
                if supervised_results:
                    # Combine using the weighted approach
                    group_indices, group_scores = combine_supervised_unsupervised_scores(
                        unsupervised_results, supervised_results, alpha=alpha
                    )
                    
                    # Take top features based on combined scores
                    n_feats = max(
                        CONFIG['feature_selection']['min_features'],
                        int(len(group_indices) * CONFIG['feature_selection']['selection_ratio'])
                    )
                    
                    for idx in group_indices[:n_feats]:
                        if idx not in combined_indices:
                            combined_indices.append(idx)
                            combined_names.append(feature_names[idx])
                            combined_scores[idx] = group_scores[idx]
                else:
                    # No supervised results, use unsupervised only
                    for idx in all_group_selected[grp]['variance']:
                        if idx not in combined_indices:
                            combined_indices.append(idx)
                            combined_names.append(feature_names[idx])
    else:
        # Fallback to original hierarchical combination
        for grp in FEATURE_GROUPS:
            if grp in all_group_selected:
                for idx in all_group_selected[grp]['variance']:
                    if idx not in combined_indices:
                        combined_indices.append(idx)
                        combined_names.append(feature_names[idx])
    
    # CRITICAL: Final validation of selected features
    print(f"\n>>> Validating selected features...")
    valid_combined_indices = []
    valid_combined_names = []
    excluded_in_final = []
    
    for idx, name in zip(combined_indices, combined_names):
        if is_valid_feature(name):
            valid_combined_indices.append(idx)
            valid_combined_names.append(name)
        else:
            excluded_in_final.append(name)
    
    if excluded_in_final:
        print(f"WARNING: Excluded {len(excluded_in_final)} invalid features from final selection:")
        for feat in excluded_in_final[:10]:
            print(f"  - {feat}")
    
    combined_indices = valid_combined_indices
    combined_names = valid_combined_names
    
    # CRITICAL: Always include mandatory features (gap feature)
    for feat_name in MANDATORY_FEATURES:
        if feat_name in feature_names:
            feat_idx = feature_names.index(feat_name)
            if feat_idx not in combined_indices:
                print(f"\n>>> Adding mandatory feature: {feat_name}")
                combined_indices.insert(0, feat_idx)  # Add at the beginning
                combined_names.insert(0, feat_name)
    
    # Save enhanced results
    final_results = {
        'selected_indices': combined_indices,
        'selected_features': combined_names,
        'window_type': data.get('window_type', 'unknown'),
        'data_type': data.get('data_type', CONFIG['data_type']),
        'all_features': feature_names,
        'method': 'hierarchical_combination_enhanced' if CONFIG['feature_selection']['enable_supervised'] else 'hierarchical_combination',
        'used_supervised': CONFIG['feature_selection']['enable_supervised'],
        'supervised_weight': CONFIG['feature_selection']['supervised_weight'] if CONFIG['feature_selection']['enable_supervised'] else 0,
        'mandatory_features': MANDATORY_FEATURES,
        'scores': combined_scores if CONFIG['feature_selection']['enable_supervised'] else {},
        'sampling_strategy': CONFIG['feature_selection']['sampling']
    }
    
    # Visualization and saving (same as before)
    window_type = data.get('window_type', 'unknown')
    out_dir = os.path.join(CONFIG['results_dir'], window_type)
    
    if combined_scores and CONFIG['feature_selection']['enable_supervised']:
        plt.figure(figsize=(14, 6))
        top_indices = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)[:20]
        top_names = [feature_names[idx] for idx in top_indices]
        top_scores = [combined_scores[idx] for idx in top_indices]
        
        plt.bar(range(len(top_names)), top_scores)
        plt.xticks(range(len(top_names)), top_names, rotation=90)
        plt.title(f"Top 20 Features by Combined Score (α={alpha:.2f}) for {data_type}")
        plt.ylabel("Combined Score")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'combined_scores_plot_{data_type}.png'))
        plt.close()
    
    save_selected_features_safe(
        data_type, final_results['method'], data, final_results)
    
    # Save hierarchical combination results to JSON for unified selection
    with open(os.path.join(out_dir, f'{final_results["method"]}_results_{data_type}.json'), 'w') as f:
        json.dump({
            'method': final_results['method'],
            'data_type': data_type,
            'data_source': CONFIG['data_type'],
            'window_type': window_type,
            'selected_indices': combined_indices,
            'selected_features': combined_names,
            'all_features': feature_names,
            'n_features': len(combined_indices),
            'used_supervised': final_results.get('used_supervised', False),
            'supervised_weight': final_results.get('supervised_weight', 0),
            'mandatory_features': MANDATORY_FEATURES,
            'sampling_strategy': CONFIG['feature_selection']['sampling'],
            'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'seed': CONFIG['seed']
        }, f, indent=2)
    
    return all_group_selected

def save_selected_features_safe(data_type, method, data, results):
    """FIXED: Save selected features with proper memory-mapped array handling"""
    print(f"\n*** Saving selected features for {CONFIG['data_type']}/{data_type} using {method} method ***\n")
    start_time = datetime.now()
    
    # CRITICAL: Disable garbage collection during memory-mapped operations
    gc_was_enabled = gc.isenabled()
    gc.disable()
    
    try:
        # CRITICAL: Final validation before saving
        selected_indices = results['selected_indices']
        selected_features = results['selected_features']
        
        # Final validation
        valid_indices = []
        valid_features = []
        excluded_features = []
        
        for idx, feat in zip(selected_indices, selected_features):
            if is_valid_feature(feat):
                valid_indices.append(idx)
                valid_features.append(feat)
            else:
                excluded_features.append(feat)
        
        if excluded_features:
            print(f"\nWARNING: Excluding {len(excluded_features)} invalid features from final save:")
            for feat in excluded_features[:10]:
                print(f"  - {feat}")
            if len(excluded_features) > 10:
                print(f"  ... and {len(excluded_features) - 10} more")
        
        # Update with validated features
        selected_indices = np.array(valid_indices)
        feature_names_selected = valid_features
        
        print(f"\nFinal selection: {len(selected_indices)} valid features")
        
        window_type = data.get('window_type', 'unknown')
        
        output_dir = os.path.join(CONFIG['output_dir'], window_type)
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each split
        for split in ['train', 'holdout', 'val', 'test']:
            X_data = data.get(f'X_{split}')
            if X_data is None:
                print(f"Skipping {split} - not found")
                continue
                
            print(f"Processing {split} set...")
            shape = (X_data.shape[0], X_data.shape[1], len(selected_indices))
            
            # Determine output filename based on window type
            if window_type == 'combined':
                out_file = os.path.join(output_dir, f'X_{split}.npy')
            else:
                out_file = os.path.join(output_dir, f'X_{split}_{window_type}.npy')
            
            # CRITICAL FIX: Use regular numpy save instead of memory-mapped files
            # This avoids all the memory-mapping issues
            print(f"Creating output array for {split}...")
            X_selected = np.zeros(shape, dtype=CONFIG['memory']['precision'])
            
            chunk_size = CONFIG['memory']['chunk_size']
            for start in tqdm(range(0, X_data.shape[0], chunk_size), desc=f"{split} chunks"):
                end = min(start + chunk_size, X_data.shape[0])
                
                # Just copy the data
                try:
                    X_selected[start:end] = X_data[start:end][:, :, selected_indices]
                except Exception as e:
                    print(f"Error in vectorized copy, falling back to loop: {e}")
                    # Fallback to loop if vectorized fails
                    for j, feat_idx in enumerate(selected_indices):
                        try:
                            X_selected[start:end, :, j] = X_data[start:end, :, feat_idx]
                        except Exception as e2:
                            print(f"Error copying feature {feat_idx}: {e2}")
                            X_selected[start:end, :, j] = 0
            
            # Save as regular numpy file
            print(f"Saving {split} to disk...")
            np.save(out_file, X_selected)
            
            # Clear the array from memory
            del X_selected
            
            # Small delay between splits
            time.sleep(0.5)
        
        # Copy labels, IDs, and window dates
        for split in ['train', 'holdout', 'val', 'test']:
            # Determine source filenames based on window type
            if window_type in ['activity', 'calendar']:
                suffix = f"_{window_type}"
            else:
                suffix = ""
            
            # Labels
            src = os.path.join(CONFIG['input_dir'], f"L_{split}{suffix}.npy")
            dst = os.path.join(output_dir, f"L_{split}{suffix}.npy")
            if os.path.exists(src) and not os.path.exists(dst):
                try:
                    shutil.copy2(src, dst)
                    print(f"Copied L_{split}{suffix}.npy")
                except Exception as e:
                    print(f"Error copying L_{split}{suffix}.npy: {e}")
            
            # IDs
            src = os.path.join(CONFIG['input_dir'], f"ID_{split}{suffix}.npy")
            dst = os.path.join(output_dir, f"ID_{split}{suffix}.npy")
            if os.path.exists(src) and not os.path.exists(dst):
                try:
                    shutil.copy2(src, dst)
                    print(f"Copied ID_{split}{suffix}.npy")
                except Exception as e:
                    print(f"Error copying ID_{split}{suffix}.npy: {e}")
            
            # Window dates CSV
            src = os.path.join(CONFIG['input_dir'], f"window_dates_{split}{suffix}.csv")
            dst = os.path.join(output_dir, f"window_dates_{split}{suffix}.csv")
            if os.path.exists(src) and not os.path.exists(dst):
                try:
                    shutil.copy2(src, dst)
                    print(f"Copied window_dates_{split}{suffix}.csv")
                except Exception as e:
                    print(f"Error copying window_dates_{split}{suffix}.csv: {e}")
            
            # MODIFIED: Also copy RG prediction files if they exist
            rg_files = [
                f"rg_scores_seq_{split}{suffix}.npy",
                f"rg_category_dist_seq_{split}{suffix}.npy",
                f"rg_has_data_seq_{split}{suffix}.npy",
                f"window_mean_score_{split}{suffix}.npy",
                f"window_coverage_{split}{suffix}.npy",
                f"manual_labels_{split}{suffix}.npy",
                f"manual_dates_{split}{suffix}.npy",
                f"window_score_std_{split}{suffix}.npy"
            ]
            
            for rg_file in rg_files:
                src = os.path.join(CONFIG['input_dir'], rg_file)
                dst = os.path.join(output_dir, rg_file)
                if os.path.exists(src) and not os.path.exists(dst):
                    try:
                        shutil.copy2(src, dst)
                        print(f"Copied {rg_file}")
                    except Exception as e:
                        print(f"Error copying {rg_file}: {e}")
        
        # Write feature names
        with open(os.path.join(output_dir, 'feature_names.txt'), 'w') as f:
            for feat in feature_names_selected:
                f.write(f"{feat}\n")
        
        # Save selection info
        info = {
            'method': method,
            'data_type': CONFIG['data_type'],  # NEW: Which data type this is
            'window_type': window_type,
            'selected_indices': selected_indices.tolist(),
            'n_features': len(selected_indices),
            'used_supervised': results.get('used_supervised', False),
            'supervised_weight': results.get('supervised_weight', 0),
            'use_all_data': CONFIG['feature_selection']['use_all_data'],
            'sampling_strategy': CONFIG['feature_selection']['sampling'],
            'mandatory_features': MANDATORY_FEATURES,
            'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'seed': CONFIG['seed']  # Include seed
        }
        with open(os.path.join(output_dir, 'selection_info.json'), 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"\nSaved {CONFIG['data_type']}/{data_type} data with selected features")
        print(f"  Method: {method}")
        print(f"  Data type: {CONFIG['data_type']}")
        print(f"  Window type: {window_type}")
        print(f"  Features: {len(selected_indices)}")
        print(f"  Mandatory features included: {MANDATORY_FEATURES}")
        print(f"  Used supervised: {results.get('used_supervised', False)}")
        if results.get('used_supervised'):
            print(f"  Supervised weight: {results.get('supervised_weight', 0):.2f}")
        print(f"  Sampling: {CONFIG['feature_selection']['sampling']['sample_percentage']*100:.1f}% " + 
              f"({CONFIG['feature_selection']['sampling']['min_samples']:,} - " +
              f"{CONFIG['feature_selection']['sampling']['max_samples']:,} samples)")
        print(f"  Completed in: {datetime.now() - start_time}")
        print_memory_usage()
        
    finally:
        # Re-enable garbage collection if it was enabled before
        if gc_was_enabled:
            gc.enable()
    
    return True

def apply_feature_selection_from_results(source_window_type, target_window_type, selection_results):
    """Apply feature selection results from one window type to another"""
    print(f"\n{'='*80}")
    print(f"Applying feature selection from {CONFIG['data_type']}/{source_window_type} to {CONFIG['data_type']}/{target_window_type}")
    print('='*80)
    
    # Load the selection results
    source_results_dir = os.path.join(CONFIG['results_dir'], source_window_type)
    
    # Try to find the hierarchical combination results first (enhanced or regular)
    results_file = None
    for method in ['hierarchical_combination_enhanced', 'hierarchical_combination', 'temporal_variance', 'temporal_correlation']:
        potential_file = os.path.join(source_results_dir, f'{method}_results_{source_window_type}.json')
        if os.path.exists(potential_file):
            results_file = potential_file
            break
    
    if results_file is None:
        print(f"Error: No feature selection results found for {CONFIG['data_type']}/{source_window_type}")
        return None
    
    print(f"Loading selection results from: {results_file}")
    with open(results_file, 'r') as f:
        source_results = json.load(f)
    
    selected_indices = source_results['selected_indices']
    selected_features = source_results['selected_features']
    
    print(f"Found {len(selected_indices)} selected features from {source_window_type}")
    print(f"Method used: {source_results.get('method', 'unknown')}")
    if source_results.get('used_supervised'):
        print(f"Used supervised methods with weight: {source_results.get('supervised_weight', 0):.2f}")
    print(f"First 10 features: {selected_features[:10]}")
    
    # Check if mandatory features are included
    for feat in MANDATORY_FEATURES:
        if feat in selected_features:
            print(f"✓ Mandatory feature '{feat}' is included")
        else:
            print(f"⚠️  WARNING: Mandatory feature '{feat}' is missing!")
    
    # Load target window type data
    target_data = load_data_for_window_type(target_window_type)
    if target_data is None:
        print(f"Error: Could not load {CONFIG['data_type']}/{target_window_type} data")
        return None
    
    # Verify features match
    if target_data['feature_names'] != source_results.get('all_features', target_data['feature_names']):
        print("Warning: Feature names might not match perfectly between window types")
    
    # Apply the selection
    results = {
        'selected_indices': selected_indices,
        'selected_features': selected_features,
        'window_type': target_window_type,
        'data_type': CONFIG['data_type'],  # NEW
        'source_window_type': source_window_type,
        'method': source_results.get('method', 'unknown'),
        'used_supervised': source_results.get('used_supervised', False),
        'supervised_weight': source_results.get('supervised_weight', 0)
    }
    
    # Save the selected features for target window type
    save_selected_features_safe(
        target_window_type, 
        f'unified_from_{source_window_type}', 
        target_data, 
        results
    )
    
    # Also save the selection info in results directory
    target_results_dir = os.path.join(CONFIG['results_dir'], target_window_type)
    os.makedirs(target_results_dir, exist_ok=True)
    
    unified_results = {
        'method': f'unified_from_{source_window_type}',
        'data_type': CONFIG['data_type'],  # NEW
        'window_type': target_window_type,
        'source_window_type': source_window_type,
        'source_method': source_results.get('method', 'unknown'),
        'selected_indices': selected_indices,
        'selected_features': selected_features,
        'n_features': len(selected_indices),
        'used_supervised': source_results.get('used_supervised', False),
        'supervised_weight': source_results.get('supervised_weight', 0),
        'mandatory_features': MANDATORY_FEATURES,
        'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'seed': CONFIG['seed']
    }
    
    with open(os.path.join(target_results_dir, f'unified_selection_results.json'), 'w') as f:
        json.dump(unified_results, f, indent=2)
    
    print(f"\nSuccessfully applied {len(selected_indices)} features to {CONFIG['data_type']}/{target_window_type}")
    return unified_results

def run_feature_selection():
    """Run the full feature selection pipeline"""
    print("\n" + "="*80)
    print("FEATURE SELECTION FOR ENHANCED GAMBLING DATA")
    print(f"DATA TYPE: {CONFIG['data_type'].upper()}")
    print("WITH RG-BASED SUPERVISED ENHANCEMENT" if CONFIG['feature_selection']['enable_supervised'] else "")
    print("FIXED: EXCLUDING RAW FEATURES")
    print("="*80)
    
    if CONFIG['unified_selection']['enabled']:
        print(f"\nUNIFIED SELECTION MODE ENABLED")
        print(f"Will run selection on: {CONFIG['data_type']}/{CONFIG['unified_selection']['source_window_type']}")
        print(f"Will apply results to: {CONFIG['data_type']}/{CONFIG['unified_selection']['apply_to_window_type']}")
    else:
        print(f"\nProcessing window types: {CONFIG['process_window_types']}")
    
    if CONFIG['feature_selection']['enable_supervised']:
        print(f"\nRG-BASED SUPERVISED METHODS ENABLED")
        print(f"Supervised weight: {CONFIG['feature_selection']['supervised_weight']:.2f}")
        print(f"Enabled methods: {', '.join(CONFIG['feature_selection']['supervised_methods'])}")
    
    print(f"\nMANDATORY FEATURES:")
    for feat in MANDATORY_FEATURES:
        print(f"  - {feat}")
    
    print(f"\nSAMPLING STRATEGY:")
    print(f"  Percentage: {CONFIG['feature_selection']['sampling']['sample_percentage']*100:.1f}%")
    print(f"  Min samples: {CONFIG['feature_selection']['sampling']['min_samples']:,}")
    print(f"  Max samples: {CONFIG['feature_selection']['sampling']['max_samples']:,}")
    print(f"  Stratified: {CONFIG['feature_selection']['sampling']['stratified']}")
    print(f"  Random seed: {CONFIG['seed']}")
    
    print("\nIMPORTANT: Automatic cleanup is DISABLED to prevent BSOD")
    print("Temporary files will remain in:", TEMP_DIR)
    print("You can manually delete them after rebooting if needed\n")
    
    results_by_type = {}
    
    if CONFIG['unified_selection']['enabled']:
        # Unified selection mode for 2-stage autoencoder
        source_type = CONFIG['unified_selection']['source_window_type']
        target_type = CONFIG['unified_selection']['apply_to_window_type']
        
        # Step 1: Run feature selection on source window type
        print(f"\n{'='*60}")
        print(f"Step 1: Running feature selection on {CONFIG['data_type']}/{source_type} windows")
        print('='*60)
        
        source_data = load_data_for_window_type(source_type)
        if source_data is not None:
            results_by_type[source_type] = run_feature_selection_methods_safe(
                source_type, source_data
            )
            
            # Step 2: Apply the same features to target window type
            print(f"\n{'='*60}")
            print(f"Step 2: Applying selected features to {CONFIG['data_type']}/{target_type} windows")
            print('='*60)
            
            apply_feature_selection_from_results(source_type, target_type, results_by_type[source_type])
            
            print(f"\n{'='*60}")
            print("UNIFIED FEATURE SELECTION COMPLETE")
            print(f"Both {CONFIG['data_type']}/{source_type} and {CONFIG['data_type']}/{target_type} now have the same features")
            print('='*60)
    
    elif CONFIG['process_window_types'] == 'both':
        # Process both window types separately
        for window_type in CONFIG['window_types']:
            print(f"\n{'='*80}")
            print(f"Processing {CONFIG['data_type']}/{window_type} windows")
            print('='*80)
            
            data = load_data_for_window_type(window_type)
            if data is not None:
                results_by_type[window_type] = run_feature_selection_methods_safe(
                    window_type, data
                )
            else:
                print(f"Failed to load {CONFIG['data_type']}/{window_type} data")
    
    elif CONFIG['process_window_types'] == 'combined':
        # Process combined data
        print(f"\n{'='*80}")
        print(f"Processing combined {CONFIG['data_type']} data from both window types")
        print('='*80)
        
        data = load_combined_data()
        if data is not None:
            results_by_type['combined'] = run_feature_selection_methods_safe(
                'combined', data
            )
        else:
            print(f"Failed to load combined {CONFIG['data_type']} data")
    
    elif CONFIG['process_window_types'] in CONFIG['window_types']:
        # Process single window type
        window_type = CONFIG['process_window_types']
        print(f"\n{'='*80}")
        print(f"Processing {CONFIG['data_type']}/{window_type} windows only")
        print('='*80)
        
        data = load_data_for_window_type(window_type)
        if data is not None:
            results_by_type[window_type] = run_feature_selection_methods_safe(
                window_type, data
            )
        else:
            print(f"Failed to load {CONFIG['data_type']}/{window_type} data")
    
    print("\n" + "="*80)
    print("FEATURE SELECTION COMPLETE")
    print(f"DATA TYPE: {CONFIG['data_type'].upper()}")
    print("OPTIMIZATIONS APPLIED:")
    print("  ✓ Proper data type organization (sessions/bets/payments/transactions)")
    print("  ✓ EXCLUDED RAW FEATURES - only normalized/derived selected")
    print("  ✓ Validated features at multiple stages")
    print("  ✓ RG-based supervised selection (if enabled)")
    print("  ✓ Mandatory features always included")
    print("  ✓ Stratified sampling with edge-case handling")
    print("  ✓ Vectorized temporal variance computation")
    print("  ✓ Redundancy scores instead of correlation (proper scaling)")
    print("  ✓ Fixed FFT normalization")
    print("  ✓ Vectorized feature copying during save")
    print("  ✓ Incremental PCA for dimensionality analysis")
    if CONFIG['feature_selection']['enable_supervised']:
        print("  ✓ Combined RG-based supervised and unsupervised methods")
    print("="*80)
    
    # Summary
    for wtype, results in results_by_type.items():
        print(f"\n{CONFIG['data_type']}/{wtype}:")
        for grp in FEATURE_GROUPS:
            if grp in results:
                n_selected = len(results[grp].get('variance', []))
                print(f"  Group '{grp}': {n_selected} features selected")
    
    print(f"\nResults saved to: {CONFIG['results_dir']}")
    print(f"Selected features saved to: {CONFIG['output_dir']}")
    print("\nDone!")

if __name__ == "__main__":
    run_feature_selection()