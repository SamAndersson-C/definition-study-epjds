#!/usr/bin/env python3
"""
OPTIMIZED Teacher-Student Conditional Hierarchical VAE for Gambling Risk Assessment
==================================================================================
VERSION 5 MINIMAL FIX: Just unfreeze final encoder projections for adaptation

MINIMAL CHANGES FROM V5:
✅ Unfreeze long_mu and long_logvar during student training
✅ Optional 3-minute calendar warmup
✅ Everything else stays exactly the same as working v5

PERFORMANCE OPTIMIZATIONS from v5:
✅ PRECOMPUTED EMBEDDINGS
✅ DATA LOADING optimizations
✅ TRAINING OPTIMIZATIONS
✅ ANTI-COLLAPSE via simple balanced sampler
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, BatchSampler
import matplotlib.pyplot as plt
import json
from datetime import datetime
import gc
import time
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
import pickle
import random
from contextlib import nullcontext
import sys
import inspect

# Optional faulthandler with environment variable control
try:
    import faulthandler
    if os.environ.get("ENABLE_FAULTHANDLER", "1") == "1":
        faulthandler.enable()
except Exception:
    pass

# Print environment info
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()} | Device: {device}")

# Autocast compatibility helper
def get_autocast(device):
    """
    Return a callable that yields a context manager:
      ctx = get_autocast(device)
      with ctx(enabled=True): ...
    On CPU, the 'enabled' arg is accepted but ignored.
    """
    if device.type == 'cuda' and hasattr(torch, 'amp'):
        return lambda enabled=True: torch.amp.autocast(device_type='cuda', enabled=enabled)
    return lambda enabled=True: nullcontext()

def make_grad_scaler(device, enabled: bool):
    """
    Version-robust GradScaler factory:
    - Prefers torch.amp.GradScaler(enabled=...)
    - Falls back to torch.cuda.amp.GradScaler on older PyTorch
    - Returns None on CPU-only
    """
    if not enabled:
        return None
    # Newer-style API
    try:
        return torch.amp.GradScaler(enabled=True)
    except Exception:
        pass
    # Older-style CUDA-only API
    try:
        from torch.cuda.amp import GradScaler as CudaGradScaler
        if device.type == 'cuda' and torch.cuda.is_available():
            return CudaGradScaler(enabled=True)
    except Exception:
        pass
    return None

def make_adamw(params, lr, weight_decay, device):
    """Version-robust AdamW optimizer factory."""
    kwargs = dict(lr=lr, weight_decay=weight_decay)
    try:
        if ('fused' in inspect.signature(torch.optim.AdamW).parameters
            and device.type == 'cuda' and torch.cuda.is_available()):
            kwargs['fused'] = True
    except Exception:
        pass
    return torch.optim.AdamW(params, **kwargs)

try:
    from tqdm.auto import tqdm as _tqdm
    tqdm = _tqdm
except Exception:
    class tqdm:
        def __init__(self, iterable=None, *args, **kwargs):
            self.iterable = iterable
            self.total = kwargs.get('total', len(iterable) if iterable is not None and hasattr(iterable, '__len__') else None)
            self.desc = kwargs.get('desc', '')
            self.n = 0
            if self.desc:
                print(self.desc)

        def __iter__(self):
            if self.iterable is None:
                return iter([])
            total = self.total or (len(self.iterable) if hasattr(self.iterable, '__len__') else None)
            for i, item in enumerate(self.iterable, 1):
                self.n = i
                if total and i % 10 == 0:
                    print(f"\r  Progress: {i}/{total}", end='', flush=True)
                yield item
            if total:
                print(f"\r  Progress: {total}/{total}")

        def update(self, n=1):
            self.n += n
            if self.total and self.n % 10 == 0:
                print(f"\r  Progress: {self.n}/{self.total}", end='', flush=True)

        def close(self):
            if self.total:
                print(f"\r  Progress: {self.total}/{self.total}")

        def set_postfix(self, *args, **kwargs): pass
        def set_description(self, desc): print(f"\r{desc}", end='', flush=True)

# Import from your existing autoencoder script
from train_hierarchical_risk_priors_check_v3 import (
    RiskPrior, CausalConv1d, TCNBlock, ConditionalTCNEncoder,
    calculate_participation_ratio, compute_temporal_consistency_loss,
    compute_total_correlation, centroid_loss, contrastive_loss,
    get_kl_weight, spectral_norm, clip_gradients_safe
)

# ============================================================================
# PERFORMANCE MODE FOR STUDENT PHASE
# ============================================================================

def enable_perf_mode_for_student():
    """Enable performance optimizations for student training."""
    # Non-deterministic, fastest kernels for Conv1d
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")  # PyTorch ≥ 2.0
        except Exception:
            pass
    print("✅ Performance mode enabled for student training")

def safe_load(path, map_location):
    """Safe checkpoint loading with backwards compatibility."""
    try:
        # New secure path (PyTorch where weights_only exists)
        return torch.load(path, map_location=map_location, weights_only=True)
    except (TypeError, pickle.UnpicklingError):
        # Older PyTorch fallback or complex checkpoint
        return torch.load(path, map_location=map_location, weights_only=False)

def make_loader(dataset, **kwargs):
    """Create DataLoader with backward-compatible pinning."""
    pin_kwargs = {}
    if device.type == 'cuda':
        pin_kwargs['pin_memory'] = True
        # Check if pin_memory_device is supported
        if 'pin_memory_device' in inspect.signature(DataLoader.__init__).parameters:
            pin_kwargs['pin_memory_device'] = 'cuda'
    return DataLoader(dataset, **pin_kwargs, **kwargs)

def encoder_fingerprint(model):
    """Generate a hash fingerprint of encoder weights for caching."""
    import hashlib
    keys = []
    wanted_prefixes = (
    'short_encoder', 'mid_encoder', 'long_encoder',
    'short_mu', 'short_logvar', 'mid_mu', 'mid_logvar',
    'label_embedding_latent',
    'short_attention', 'mid_attention', 'long_attention'
)
    
    for name, _ in model.named_parameters():
        if any(name.startswith(prefix) for prefix in wanted_prefixes):
            keys.append(name)
    
    h = hashlib.sha1()
    sd = model.state_dict()
    for k in sorted(keys):
        h.update(sd[k].detach().float().cpu().numpy().tobytes())
    
    # Include critical hyperparams to invalidate cache on config changes
    h.update(str(model.input_dim).encode())
    h.update(str(model.seq_len).encode())
    h.update(str(tuple(model.latent_dims)).encode())
    h.update(str(model.label_embed_dim).encode())
    
    return h.hexdigest()[:12]

# ============================================================================
# PRECOMPUTE EMBEDDINGS HELPERS
# ============================================================================

@torch.inference_mode()
def precompute_embeddings(model, dataset, device, batch_size=512):
    """Precompute z_all embeddings for entire dataset."""
    print(f"Precomputing embeddings for {len(dataset)} samples...")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    Z_list = []
    model.eval()
    for batch in tqdm(loader, desc="Computing embeddings"):
        x = batch['features'].to(device, non_blocking=True)
        m = batch['mask'].to(device, non_blocking=True)
        y = batch['risk_label'].to(device, non_blocking=True)
        
        outputs = model(x, m, y, phase='student', decode=False)
        Z_list.append(outputs['z_all'].cpu())
    
    return torch.cat(Z_list, 0)

@torch.inference_mode()
def precompute_bank(model, dataset, device, batch_size=1024, num_workers=0):
    """Precompute embeddings bank with labels."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=(device.type=='cuda'))
    Z, Ymh, HM, AUG = [], [], [], []
    model.eval()
    for b in tqdm(loader, desc="Computing embedding bank"):
        x = b['features'].to(device, non_blocking=True)
        m = b['mask'].to(device, non_blocking=True)
        y = b['risk_label'].to(device, non_blocking=True)
        out = model(x, m, y, phase='student', decode=False)
        Z.append(out['z_all'].float().cpu())
        Ymh.append(b['manual_labels_multihot'].cpu())
        HM.append(b['has_manual_bool'].cpu())
        AUG.append(b.get('is_augmented', torch.zeros_like(b['has_manual_bool'])).cpu())
    return torch.cat(Z,0), torch.cat(Ymh,0), torch.cat(HM,0), torch.cat(AUG,0)

@torch.inference_mode()
def precompute_backbone_for_long(model, dataset, device, batch_size=1024, num_workers=0):
    """
    Precompute z_short (mean), z_mid (mean), and pooled h_long once.
    Unknown label (index 5) is used consistently for student phase.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=(device.type=='cuda'))
    Zs, Zm, Hlong, MH, HM, AUG = [], [], [], [], [], []
    model.eval()
    
    for b in tqdm(loader, desc="Precompute backbone (S/M/H_long)"):
        x = b['features'].to(device, non_blocking=True)
        m = b['mask'].to(device, non_blocking=True)
        y = torch.full((x.size(0),), 5, dtype=torch.long, device=device)  # unknown label
        
        # Get label embedding
        label_emb = model.label_embedding_latent(y)
        label_emb = F.normalize(label_emb, p=2, dim=1) * 0.1
        
        # Short-term encoding (deterministic mean)
        h_short = model.encode_with_mask(x, m, y, model.short_encoder, 
                                        model.short_attention, predict_mode=True)
        h_short_cond = torch.cat([h_short, label_emb], dim=1)
        short_mu = model.short_mu(h_short_cond)
        
        # Mid-term encoding (deterministic mean, conditioned on short)
        h_mid = model.encode_with_mask(x, m, y, model.mid_encoder,
                                      model.mid_attention, predict_mode=True)
        h_mid_cond = torch.cat([h_mid, short_mu, label_emb], dim=1)
        mid_mu = model.mid_mu(h_mid_cond)
        
        # Long-term pooled features (no projection yet)
        h_long = model.encode_with_mask(x, m, y, model.long_encoder,
                                       model.long_attention, predict_mode=True)
        
        Zs.append(short_mu.float().cpu())
        Zm.append(mid_mu.float().cpu())
        Hlong.append(h_long.float().cpu())
        MH.append(b['manual_labels_multihot'].cpu())
        HM.append(b['has_manual_bool'].cpu())
        AUG.append(b.get('is_augmented', torch.zeros_like(b['has_manual_bool'])).cpu())
    
    return (torch.cat(Zs, 0), torch.cat(Zm, 0), torch.cat(Hlong, 0),
            torch.cat(MH, 0), torch.cat(HM, 0), torch.cat(AUG, 0))

@torch.inference_mode()
def precompute_teacher_logits_from_Z(model, Z, device, batch_size=4096):
    """Precompute teacher logits from embeddings."""
    logits = []
    model.eval()
    for i in range(0, Z.shape[0], batch_size):
        z = Z[i:i+batch_size].to(device, non_blocking=True).float()
        logits.append(model.rg_cat_head(z).cpu())
    return torch.cat(logits, 0)

class EmbeddingDataset(Dataset):
    """Dataset that serves precomputed embeddings."""
    def __init__(self, embeddings, original_dataset=None, Z=None, multihot=None, has_manual=None, is_aug=None):
        if original_dataset is not None:
            # Legacy interface
            self.Z = embeddings
            self.dataset = original_dataset
        else:
            # New interface
            self.Z = Z
            self.mh = multihot
            self.hm = has_manual
            self.aug = is_aug
            
    def __len__(self):
        return len(self.Z) if hasattr(self, 'Z') else len(self.dataset)
    
    def __getitem__(self, idx):
        if hasattr(self, 'dataset'):
            # Legacy interface
            orig = self.dataset[idx]
            return {
                'z_all': self.Z[idx],
                'manual_labels_multihot': orig['manual_labels_multihot'],
                'has_manual_bool': orig['has_manual_bool'],
                'is_augmented': orig.get('is_augmented', torch.tensor(False))
            }
        else:
            # New interface
            return {
                'z_all': self.Z[idx],
                'manual_labels_multihot': self.mh[idx],
                'has_manual_bool': self.hm[idx],
                'is_augmented': self.aug[idx]
            }

class LongAdapterDataset(Dataset):
    """Dataset for long projection adaptation - serves precomputed short/mid/h_long."""
    def __init__(self, Zs, Zm, Hlong, mh, hm, aug):
        self.Zs = Zs
        self.Zm = Zm
        self.Hlong = Hlong
        self.mh = mh
        self.hm = hm
        self.aug = aug
        
    def __len__(self):
        return len(self.Zs)
    
    def __getitem__(self, idx):
        return {
            'z_short': self.Zs[idx],
            'z_mid': self.Zm[idx],
            'h_long': self.Hlong[idx],
            'manual_labels_multihot': self.mh[idx],
            'has_manual_bool': self.hm[idx],
            'is_augmented': self.aug[idx]
        }

# ============================================================================
# OPTIMIZED Helper Functions
# ============================================================================

def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _as_float(x):
    """Convert tensor or numeric to float safely."""
    if torch.is_tensor(x):
        return float(x.item())
    return float(x)

def safe_postfix_dict(losses):
    """Create safe postfix dict for tqdm with new loss names."""
    postfix = {}
    
    # Handle different loss types
    if 'total' in losses:
        postfix['loss'] = f"{_as_float(losses.get('total', 0)):.4f}"
    if 'recon' in losses:
        postfix['recon'] = f"{_as_float(losses.get('recon', 0)):.4f}"
    
    # Teacher losses
    if any(f'kl_{l}' in losses for l in ['short', 'mid', 'long']):
        kl_sum = sum(_as_float(losses.get(f'kl_{l}', 0)) for l in ['short', 'mid', 'long'])
        postfix['kl'] = f"{kl_sum:.4f}"
    
    # Student losses (new)
    if 'ce' in losses:
        postfix['ce'] = f"{_as_float(losses['ce']):.4f}"
    elif 'manual_ce' in losses:
        postfix['ce'] = f"{_as_float(losses['manual_ce']):.4f}"
    if 'teacher_consistency' in losses:
        postfix['kd'] = f"{_as_float(losses['teacher_consistency']):.4f}"
    if 'kd' in losses:
        postfix['kd'] = f"{_as_float(losses['kd']):.3f}"
    if 'rg_kl' in losses:
        postfix['rg'] = f"{_as_float(losses['rg_kl']):.4f}"
    if 'ordinal' in losses:
        postfix['ord'] = f"{_as_float(losses['ordinal']):.4f}"
    
    # Other losses
    if 'diversity' in losses:
        postfix['div'] = f"{_as_float(losses.get('diversity', 0)):.4f}"
    
    return postfix

def compute_per_class_recall(preds, targets, num_classes=5):
    """Compute recall for each class."""
    recalls = []
    for c in range(num_classes):
        mask = targets == c
        if mask.sum() > 0:
            correct = (preds[mask] == c).sum()
            recall = float(correct) / float(mask.sum())
            recalls.append(recall)
        else:
            recalls.append(float('nan'))
    return recalls

# ============================================================================
# OPTIMIZED Dataset Classes
# ============================================================================

class OptimizedTeacherStudentDataset(Dataset):
    """OPTIMIZED Dataset with caching and reduced memory operations."""
    
    def __init__(self, window_type, split, data_dir, data_type, phase='teacher', 
                 use_scaled=False, augment_centered_windows=False, include_unlabeled=False):
        """
        OPTIMIZED VERSION with pre-sanitization and tensor caching.
        """
        self.window_type = window_type
        self.split = split
        self.phase = phase
        self.data_type = data_type
        
        # Build paths
        base_dir = os.path.join(data_dir, data_type, window_type)
        
        # Load data
        print(f"Loading {data_type}/{window_type} {split} windows for {phase} phase...")
        
        # Features and masks - OPTIMIZED: Load and pre-sanitize once
        if use_scaled:
            scaled_path = os.path.join(base_dir, f'X_{split}_{window_type}_scaled.npy')
            if os.path.exists(scaled_path):
                self.X = np.load(scaled_path, mmap_mode='r')
                print(f"  Using SCALED features from {scaled_path}")
            else:
                print(f"  Loading features into RAM...")
                self.X = np.load(os.path.join(base_dir, f'X_{split}_{window_type}.npy'))
        else:
            print(f"  Loading features into RAM...")
            self.X = np.load(os.path.join(base_dir, f'X_{split}_{window_type}.npy'))
        
        # OPTIMIZATION: Pre-sanitize all data once at load time
        print("  Pre-sanitizing data (NaN/Inf cleanup)...")
        self.X = np.nan_to_num(self.X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        
        # Get actual sequence length from data
        T = self.X.shape[1]
        
        # For activity windows, mask is all ones. For calendar, load from rg_has_data_seq
        if window_type == 'activity':
            self.mask = np.ones((len(self.X), T), dtype=np.float32)
        else:
            mask_path = os.path.join(base_dir, f'rg_has_data_seq_{split}_{window_type}.npy')
            if os.path.exists(mask_path):
                self.mask = np.load(mask_path).astype(np.float32)
            else:
                self.mask = np.ones((len(self.X), T), dtype=np.float32)
        
        # Pre-sanitize mask
        self.mask = np.nan_to_num(self.mask, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        self.mask = np.clip(self.mask, 0.0, 1.0)
        
        # Assert mask length matches sequence length
        assert self.mask.shape[1] == self.X.shape[1], \
            f"Mask length {self.mask.shape[1]} must match sequence length {self.X.shape[1]}"
        
        # RG labels (seq2seq format) - pre-sanitize
        self.rg_scores_seq = np.nan_to_num(
            np.load(os.path.join(base_dir, f'rg_scores_seq_{split}_{window_type}.npy')),
            nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32)
        
        self.rg_cats_seq = np.nan_to_num(
            np.load(os.path.join(base_dir, f'rg_category_dist_seq_{split}_{window_type}.npy')),
            nan=0.2, posinf=1.0, neginf=0.0
        ).astype(np.float32)
        
        self.rg_has_data = np.nan_to_num(
            np.load(os.path.join(base_dir, f'rg_has_data_seq_{split}_{window_type}.npy')),
            nan=0.0, posinf=1.0, neginf=0.0
        ).astype(np.float32)
        
        # Window-level aggregates - pre-sanitize
        self.window_mean_score = np.nan_to_num(
            np.load(os.path.join(base_dir, f'window_mean_score_{split}_{window_type}.npy')),
            nan=0.0
        ).astype(np.float32)
        
        self.window_coverage = np.nan_to_num(
            np.load(os.path.join(base_dir, f'window_coverage_{split}_{window_type}.npy')),
            nan=0.0
        ).astype(np.float32)
        self.window_coverage = np.clip(self.window_coverage, 0.0, 1.0)
        
        # Manual labels
        self.manual_labels = np.load(os.path.join(base_dir, f'manual_labels_{split}_{window_type}.npy'))
        
        # Load multi-hot labels
        multihot_path = os.path.join(base_dir, f'manual_labels_multihot_{split}_{window_type}.npy')
        has_manual_path = os.path.join(base_dir, f'has_manual_{split}_{window_type}.npy')
        
        if os.path.exists(multihot_path) and os.path.exists(has_manual_path):
            self.manual_labels_multihot = np.load(multihot_path).astype(np.float32)
            self.has_manual_bool = np.load(has_manual_path).astype(bool)
            print(f"  Loaded multi-hot labels: {self.manual_labels_multihot.shape}")
        else:
            print("  Creating multi-hot labels on the fly")
            self.manual_labels_multihot = np.zeros((len(self.manual_labels), 5), dtype=np.float32)
            valid_mask = self.manual_labels >= 0
            if valid_mask.any():
                valid_indices = np.where(valid_mask)[0]
                valid_labels = self.manual_labels[valid_mask].astype(int)
                self.manual_labels_multihot[valid_indices, valid_labels] = 1.0
            self.has_manual_bool = self.manual_labels_multihot.sum(axis=1) > 0
        
        # Add centered window augmentation tracking
        self.is_augmented = np.zeros(len(self.X), dtype=bool)
        
        # For activity windows, load q_dists for window-level distributions
        if window_type == 'activity' and os.path.exists(os.path.join(base_dir, f'q_dists_{split}_{window_type}.npy')):
            self.q_dists = np.load(os.path.join(base_dir, f'q_dists_{split}_{window_type}.npy')).astype(np.float32)
        else:
            self.q_dists = self._calculate_window_dists()
            
        # Pre-sanitize and normalize q_dists
        self.q_dists = np.nan_to_num(self.q_dists, nan=0.2)
        # Ensure valid probability distributions
        row_sums = self.q_dists.sum(axis=1, keepdims=True)
        self.q_dists = self.q_dists / np.maximum(row_sums, 1e-8)
        
        print(f"  Loaded {len(self.X)} {window_type} windows")
        print(f"  Windows with RG data: {(self.window_coverage > 0).sum()} ({(self.window_coverage > 0).mean()*100:.1f}%)")
        print(f"  Windows with manual labels: {(self.manual_labels >= 0).sum()} ({(self.manual_labels >= 0).mean()*100:.1f}%)")
        
        # Load feature names
        feature_names_path = os.path.join(base_dir, 'feature_names.txt')
        if os.path.exists(feature_names_path):
            with open(feature_names_path, 'r') as f:
                self.feature_names = [line.strip() for line in f]
        else:
            self.feature_names = None
        
        # Filter for student phase (unless include_unlabeled is True)
        if phase == 'student' and not include_unlabeled:
            # Only use windows with manual labels
            manual_indices = np.where(self.manual_labels >= 0)[0]
            print(f"  Filtering to {len(manual_indices)} windows with manual labels for student phase")
            
            # OPTIMIZATION: Use numpy advanced indexing (faster than loops)
            self._filter_arrays(manual_indices)
        
        # Apply centered window augmentation if requested
        if phase == 'student' and window_type == 'calendar' and augment_centered_windows:
            self._create_centered_windows()
        
        # OPTIMIZATION: Pre-convert data to tensors and cache on device if small enough
        self._cache_tensors()
    
    def _filter_arrays(self, indices):
        """OPTIMIZED: Filter all arrays at once using numpy advanced indexing."""
        self.X = self.X[indices]
        self.mask = self.mask[indices]
        self.rg_scores_seq = self.rg_scores_seq[indices]
        self.rg_cats_seq = self.rg_cats_seq[indices]
        self.rg_has_data = self.rg_has_data[indices]
        self.window_mean_score = self.window_mean_score[indices]
        self.window_coverage = self.window_coverage[indices]
        self.manual_labels = self.manual_labels[indices]
        self.manual_labels_multihot = self.manual_labels_multihot[indices]
        self.has_manual_bool = self.has_manual_bool[indices]
        self.q_dists = self.q_dists[indices]
        self.is_augmented = self.is_augmented[indices]
    
    def _cache_tensors(self):
        """OPTIMIZATION: Pre-convert frequently accessed data to tensors."""
        # CPU tensors cached once; device transfer happens in the training loop
        self.X_tensor = torch.from_numpy(self.X).float()
        self.mask_tensor = torch.from_numpy(self.mask).float()
        self.q_dists_tensor = torch.from_numpy(self.q_dists).float()
        self.manual_labels_multihot_tensor = torch.from_numpy(self.manual_labels_multihot).float()
        
        self.manual_labels_tensor = torch.from_numpy(self.manual_labels).long()
        self.has_manual_tensor = torch.from_numpy(self.has_manual_bool).bool()
        self.is_augmented_tensor = torch.from_numpy(self.is_augmented).bool()
        self.window_coverage_tensor = torch.from_numpy(self.window_coverage).float()
        
        self.cached_on_device = False
    
    def _create_centered_windows(self):
        """Create centered windows around manual assessments with proper shifting."""
        print("  Creating centered window augmentations...")
        
        # Find windows with manual labels
        labeled_indices = np.where(self.has_manual_bool)[0]
        n_augmented = 0
        
        # Pre-allocate lists for efficiency
        augmented_X = []
        augmented_masks = []
        augmented_rg_scores_seq = []
        augmented_rg_cats_seq = []
        augmented_rg_has_data = []
        augmented_manual_multihot = []
        augmented_window_mean = []
        augmented_window_coverage = []
        augmented_q_dists = []
        
        # Shift function that handles all arrays
        def shift_array(arr, shift_days):
            """OPTIMIZED: Vectorized shift operation."""
            if len(arr.shape) == 1:  # 1D array
                if shift_days > 0:
                    return np.concatenate([np.zeros(shift_days), arr[:-shift_days]])
                elif shift_days < 0:
                    return np.concatenate([arr[-shift_days:], np.zeros(-shift_days)])
                return arr
            elif len(arr.shape) == 2:  # 2D array (time x features)
                if shift_days > 0:
                    pad = np.zeros((shift_days, arr.shape[1]), dtype=arr.dtype)
                    return np.concatenate([pad, arr[:-shift_days]], axis=0)
                elif shift_days < 0:
                    pad = np.zeros((-shift_days, arr.shape[1]), dtype=arr.dtype)
                    return np.concatenate([arr[-shift_days:], pad], axis=0)
                return arr
            else:  # 3D array (for category distributions)
                if shift_days > 0:
                    pad = np.zeros((shift_days,) + arr.shape[1:], dtype=arr.dtype)
                    return np.concatenate([pad, arr[:-shift_days]], axis=0)
                elif shift_days < 0:
                    pad = np.zeros((-shift_days,) + arr.shape[1:], dtype=arr.dtype)
                    return np.concatenate([arr[-shift_days:], pad], axis=0)
                return arr
        
        # Create augmented windows
        for idx in labeled_indices[:500]:  # Limit to prevent memory issues
            # Original window data
            orig_X = self.X[idx]
            orig_mask = self.mask[idx]
            orig_manual = self.manual_labels_multihot[idx]
            
            # Create shifts of ±7 days
            for shift in [-7, 7]:
                # Shift all arrays consistently
                aug_X = shift_array(orig_X, shift)
                aug_mask = shift_array(orig_mask, shift)
                aug_rg_scores = shift_array(self.rg_scores_seq[idx], shift)
                aug_rg_cats = shift_array(self.rg_cats_seq[idx], shift)
                aug_rg_has = shift_array(self.rg_has_data[idx], shift)
                
                # Recompute window-level statistics
                valid_mask = aug_rg_has.astype(bool)
                if valid_mask.any():
                    aug_mean_score = aug_rg_scores[valid_mask].mean()
                    aug_coverage = valid_mask.mean()
                    
                    # Recompute q_dist
                    aug_q_dist = aug_rg_cats[valid_mask].mean(axis=0)
                    aug_q_dist = aug_q_dist / (aug_q_dist.sum() + 1e-8)
                else:
                    aug_mean_score = 0.0
                    aug_coverage = 0.0
                    aug_q_dist = np.ones(5) / 5
                
                # Append to lists
                augmented_X.append(aug_X)
                augmented_masks.append(aug_mask)
                augmented_rg_scores_seq.append(aug_rg_scores)
                augmented_rg_cats_seq.append(aug_rg_cats)
                augmented_rg_has_data.append(aug_rg_has)
                augmented_manual_multihot.append(orig_manual)  # Keep same label
                augmented_window_mean.append(aug_mean_score)
                augmented_window_coverage.append(aug_coverage)
                augmented_q_dists.append(aug_q_dist)
                n_augmented += 1
        
        # Concatenate with original data
        if n_augmented > 0:
            self.X = np.vstack([self.X, np.array(augmented_X)])
            self.mask = np.vstack([self.mask, np.array(augmented_masks)])
            self.rg_scores_seq = np.vstack([self.rg_scores_seq, np.array(augmented_rg_scores_seq)])
            self.rg_cats_seq = np.vstack([self.rg_cats_seq, np.array(augmented_rg_cats_seq)])
            self.rg_has_data = np.vstack([self.rg_has_data, np.array(augmented_rg_has_data)])
            self.manual_labels_multihot = np.vstack([self.manual_labels_multihot, np.array(augmented_manual_multihot)])
            self.window_mean_score = np.concatenate([self.window_mean_score, np.array(augmented_window_mean)])
            self.window_coverage = np.concatenate([self.window_coverage, np.array(augmented_window_coverage)])
            self.q_dists = np.vstack([self.q_dists, np.array(augmented_q_dists)])
            
            # Update tracking arrays
            self.has_manual_bool = np.concatenate([self.has_manual_bool, np.ones(n_augmented, dtype=bool)])
            self.manual_labels = np.concatenate([self.manual_labels, np.full(n_augmented, -1)])
            
            # Mark augmented windows
            self.is_augmented = np.concatenate([self.is_augmented, np.ones(n_augmented, dtype=bool)])
            
            print(f"  Added {n_augmented} augmented windows (from {len(labeled_indices)} labeled windows)")
            print(f"  Total dataset size: {len(self.X)}")
    
    def _calculate_window_dists(self):
        """Calculate window-level category distributions from seq data."""
        n_windows = len(self.rg_cats_seq)
        q_dists = np.zeros((n_windows, 5), dtype=np.float32)
        
        for i in range(n_windows):
            valid = self.rg_has_data[i].astype(bool)  # Cast to boolean for indexing
            if valid.any():
                cats = self.rg_cats_seq[i][valid]
                m = cats.mean(axis=0)
                q_dists[i] = m / (m.sum() + 1e-8)
            else:
                q_dists[i] = 0.2
        
        return q_dists
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        """OPTIMIZED: Use cached tensors, minimal operations."""
        # Use cached tensors directly - major speedup
        x = self.X_tensor[idx]
        mask = self.mask_tensor[idx]
        rg_cat_dist = self.q_dists_tensor[idx]
        manual_mh = self.manual_labels_multihot_tensor[idx]
        
        # Simple scalar conversions (already sanitized)
        rg_score = torch.tensor(self.window_mean_score[idx], dtype=torch.float32)
        
        # Use cached tensor values
        has_manual = self.has_manual_tensor[idx]
        is_augmented = self.is_augmented_tensor[idx]
        window_coverage = self.window_coverage_tensor[idx]
        
        # Risk label for conditioning
        if has_manual:
            risk_label = int(manual_mh.argmax().item())
        else:
            risk_label = int(rg_cat_dist.argmax().item())
        
        # Seq2seq items
        rg_scores_seq = torch.from_numpy(self.rg_scores_seq[idx]).float()
        rg_cats_seq = torch.from_numpy(self.rg_cats_seq[idx]).float()
        rg_has_data_seq = torch.from_numpy(self.rg_has_data[idx]).float()
        
        return {
            'features': x,
            'mask': mask,
            'rg_score': rg_score,
            'rg_cat_dist': rg_cat_dist,
            'has_manual': has_manual,
            'manual_label': manual_mh,
            'risk_label': torch.tensor(risk_label, dtype=torch.long),
            'manual_labels_multihot': manual_mh,
            'has_manual_bool': has_manual,
            'is_augmented': is_augmented,
            'window_coverage': window_coverage,
            'rg_scores_seq': rg_scores_seq,
            'rg_cats_seq': rg_cats_seq,
            'rg_has_data_seq': rg_has_data_seq
        }

# ============================================================================
# OPTIMIZED Loss Computation Functions
# ============================================================================

def compute_ordinal_targets(manual_multihot: torch.Tensor) -> torch.Tensor:
    """
    Correct cumulative targets for ordinal regression:
    target[:, k] = 1 if y > k else 0, for k = 0..K-2
    """
    # manual_multihot: [B, K], one-hot
    y = manual_multihot.argmax(dim=1)              # [B]
    K = manual_multihot.size(1)
    thresholds = torch.arange(K-1, device=manual_multihot.device).unsqueeze(0)  # [1, K-1]
    return (y.unsqueeze(1) > thresholds).float()   # [B, K-1]

def compute_class_weights(dataset):
    """Compute inverse frequency weights for BCE loss."""
    # Only use real (non-augmented) windows
    real_mask = ~dataset.is_augmented
    labels = dataset.manual_labels_multihot[real_mask]
    freq = labels.sum(axis=0) + 1.0  # Add 1 to avoid division by zero
    weights = freq.max() / freq
    weights = weights / weights.mean()  # Normalize to have mean 1
    return torch.tensor(weights, dtype=torch.float32)

def estimate_rg_manual_correlations(dataset):
    """
    Estimate Pearson correlation between RG category probabilities and manual labels.
    Returns tensor of shape [5] with correlation for each class.
    """
    rg_dists = dataset.q_dists.astype(np.float32)
    manual_labels = dataset.manual_labels_multihot.astype(np.float32)
    
    correlations = []
    for k in range(5):
        rg_k = rg_dists[:, k]
        manual_k = manual_labels[:, k]
        
        if manual_k.sum() < 2 or rg_k.std() == 0 or manual_k.std() == 0:
            correlations.append(0.0)
        else:
            corr = np.corrcoef(rg_k, manual_k)[0, 1]
            correlations.append(corr if not np.isnan(corr) else 0.0)
    
    return torch.tensor(correlations, dtype=torch.float32)

def create_weighted_sampler_for_student(dataset, verify=True):
    """FIXED: Create weighted sampler with proper reproducibility."""
    print("Creating weighted sampler for student training...")
    
    labels = dataset.manual_labels_multihot
    is_augmented = dataset.is_augmented
    
    # Calculate per-class frequencies (excluding augmented)
    real_mask = ~is_augmented
    real_labels = labels[real_mask]
    class_freq = real_labels.sum(axis=0)
    print(f"Class frequencies (real windows): {class_freq.astype(int)}")
    
    # Compute balanced weights
    weights = (class_freq.max() / (class_freq + 1.0))
    weights = weights / weights.mean()  # Normalize
    print(f"Class weights: {weights}")
    
    # OPTIMIZATION: Vectorized sample weight calculation
    sample_weights = (labels / (class_freq + 1.0)).sum(axis=1)
    # Downweight augmented samples
    aug_alpha = 0.35  # augmented windows are 35% as likely to be sampled
    sample_weights[is_augmented] *= aug_alpha
    sample_weights = sample_weights / sample_weights.sum() * len(sample_weights)
    
    # Create sampler with reproducible generator
    sampler_kwargs = dict(
        weights=torch.tensor(sample_weights, dtype=torch.float32),
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Version-robust: add generator if supported
    if 'generator' in inspect.signature(WeightedRandomSampler.__init__).parameters:
        g = torch.Generator()
        g.manual_seed(42)
        sampler_kwargs['generator'] = g
    
    sampler = WeightedRandomSampler(**sampler_kwargs)
    
    if verify:
        print("\nVerifying sampler distribution...")
        # Cheaper approximation without materializing full sampler
        probs = sample_weights / sample_weights.sum()
        sampled_indices = np.random.choice(len(sample_weights), size=5000, replace=True, p=probs)
        sampled_labels = labels[sampled_indices]
        sampled_freq = sampled_labels.sum(axis=0)
        sampled_dist = sampled_freq / sampled_freq.sum()
        print(f"Sampled class distribution: {sampled_dist}")
        print("Expected: ~0.20 for each class")
        
        if (sampled_dist < 0.1).any() or (sampled_dist > 0.3).any():
            print("WARNING: Sampler may not be well balanced!")
    
    return sampler, weights

# ============================================================================
# OPTIMIZED Teacher-Student Model
# ============================================================================

class OptimizedTeacherStudentCVAE(nn.Module):
    """
    OPTIMIZED Teacher-Student Conditional Hierarchical VAE with performance improvements.
    """
    
    def __init__(self, input_dim, seq_len=30, 
                 latent_dims=[32, 24, 16],
                 hidden_dim=256,
                 num_classes=5,
                 feature_groups=None,
                 window_type='activity',
                 label_embed_dim=32,
                 use_spectral_norm=True,
                 diversity_weight=0.2,
                 min_encoding_capacity=20.0,
                 temporal_consistency_weight=0.1,
                 tc_weight=0.05,
                 centroid_weight=0.1,
                 prior_init_spread=2.0,
                 dropout=0.2,
                 **kwargs):
        super().__init__()
        
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.latent_dims = latent_dims
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.window_type = window_type
        self.label_embed_dim = label_embed_dim
        self.feature_groups = feature_groups or {'short': [], 'mid': [], 'long': []}
        self.use_spectral_norm = use_spectral_norm
        self.diversity_weight = diversity_weight
        self.min_encoding_capacity = min_encoding_capacity
        self.temporal_consistency_weight = temporal_consistency_weight
        self.tc_weight = tc_weight
        self.centroid_weight = centroid_weight
        self.dropout = dropout
        
        # Temporal encoders for each scale (conditioned on labels)
        self.short_encoder = ConditionalTCNEncoder(
            input_dim + 1,  # +1 for mask channel
            num_classes, hidden_dim, kernel_size=3, 
            num_blocks=3, dropout=self.dropout, embed_dim=label_embed_dim,
            use_spectral_norm=use_spectral_norm
        )
        
        self.mid_encoder = ConditionalTCNEncoder(
            input_dim + 1,
            num_classes, hidden_dim, kernel_size=5, 
            num_blocks=4, dropout=self.dropout, embed_dim=label_embed_dim,
            use_spectral_norm=use_spectral_norm
        )
        
        self.long_encoder = ConditionalTCNEncoder(
            input_dim + 1,
            num_classes, hidden_dim, kernel_size=7, 
            num_blocks=5, dropout=self.dropout, embed_dim=label_embed_dim,
            use_spectral_norm=use_spectral_norm
        )
        
        # Attention mechanisms
        self.short_attention = nn.MultiheadAttention(hidden_dim, num_heads=4)
        self.mid_attention = nn.MultiheadAttention(hidden_dim, num_heads=4)
        self.long_attention = nn.MultiheadAttention(hidden_dim, num_heads=4)
        
        # Label embedding for latent projections
        self.label_embedding_latent = nn.Embedding(num_classes + 1, label_embed_dim)
        
        # Risk-specific priors for each latent scale
        self.risk_prior_short = RiskPrior(num_classes, latent_dims[0], init_spread=prior_init_spread)
        self.risk_prior_mid = RiskPrior(num_classes, latent_dims[1], init_spread=prior_init_spread)
        self.risk_prior_long = RiskPrior(num_classes, latent_dims[2], init_spread=prior_init_spread)
        
        # Latent space projections (hierarchical and conditioned)
        self.short_mu = nn.Linear(hidden_dim + label_embed_dim, latent_dims[0])
        self.short_logvar = nn.Linear(hidden_dim + label_embed_dim, latent_dims[0])
        
        self.mid_mu = nn.Linear(hidden_dim + latent_dims[0] + label_embed_dim, latent_dims[1])
        self.mid_logvar = nn.Linear(hidden_dim + latent_dims[0] + label_embed_dim, latent_dims[1])
        
        self.long_mu = nn.Linear(hidden_dim + sum(latent_dims[:2]) + label_embed_dim, latent_dims[2])
        self.long_logvar = nn.Linear(hidden_dim + sum(latent_dims[:2]) + label_embed_dim, latent_dims[2])
        
        # Apply spectral normalization to latent projections if requested
        if use_spectral_norm:
            self.short_mu = spectral_norm(self.short_mu)
            self.short_logvar = spectral_norm(self.short_logvar)
            self.mid_mu = spectral_norm(self.mid_mu)
            self.mid_logvar = spectral_norm(self.mid_logvar)
            self.long_mu = spectral_norm(self.long_mu)
            self.long_logvar = spectral_norm(self.long_logvar)
        
        # === TEACHER HEADS ===
        # RG Score regression head
        self.rg_score_head = nn.Sequential(
            nn.Linear(sum(latent_dims), hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # RG Category distribution head
        self.rg_cat_head = nn.Sequential(
            nn.Linear(sum(latent_dims), hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # === STUDENT HEAD ===
        # Manual risk classification head
        self.manual_head = nn.Sequential(
            nn.Linear(sum(latent_dims), hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # === ORDINAL HEAD (4 thresholds for 5 classes) ===
        self.ordinal_head = nn.Sequential(
            nn.Linear(sum(latent_dims), hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 4)  # 4 thresholds, not 5!
        )
        
        # Conditional decoder (mask-aware)
        self.decoder_initial = nn.Linear(sum(latent_dims) + label_embed_dim + 1, hidden_dim)
        self.decoder_blocks = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.decoder_final = nn.Linear(hidden_dim, input_dim * seq_len)
        
        # Initialize weights for stability
        self._initialize_weights()
        
        # Initialize ordinal head properly
        for m in self.ordinal_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
    
    def _initialize_weights(self):
        """Initialize weights for better stability."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.001)
    
    def encode_with_mask(self, x, mask, labels, encoder, attention, predict_mode=False):
        """OPTIMIZED: Encode with mask channel concatenated."""
        mask_expanded = mask.unsqueeze(2)  # [B, T, 1]
        x_with_mask = torch.cat([x, mask_expanded], dim=2)  # [B, T, F+1]
        
        try:
            h = encoder(x_with_mask, labels, predict_mode=predict_mode)
        except TypeError:
            if predict_mode:
                labels_unknown = torch.full_like(labels, 5)
                h = encoder(x_with_mask, labels_unknown)
            else:
                h = encoder(x_with_mask, labels)
        
        h_t = h.transpose(0, 1)  # [T, B, H]
        key_padding_mask = ~mask.bool()  # [B, T]
        
        # FIX: Handle all-padded sequences safely
        all_padded = (mask.sum(dim=1) == 0)  # [B]
        if all_padded.any():
            valid = ~all_padded
            h_att = torch.zeros_like(h_t)
            if valid.any():
                h_att_valid, _ = attention(
                    h_t[:, valid, :], h_t[:, valid, :], h_t[:, valid, :],
                    key_padding_mask=key_padding_mask[valid]
                )
                h_att[:, valid, :] = h_att_valid
        else:
            h_att, _ = attention(h_t, h_t, h_t, key_padding_mask=key_padding_mask)
        
        # Sanitize any stray NaNs
        h_att = torch.nan_to_num(h_att, nan=0.0, posinf=0.0, neginf=0.0)
        
        mask_for_att = mask.transpose(0, 1).unsqueeze(2)  # [T, B, 1]
        h_masked = h_att * mask_for_att  # [T, B, H]
        
        h_sum = h_masked.sum(dim=0)  # [B, H]
        mask_sum = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        h_pooled = h_sum / mask_sum  # [B, H]
        
        return h_pooled
        
    def forward(self, x, mask, labels, phase='teacher', decode=True):
        batch_size = x.shape[0]
        
        # Handle unlabeled data
        labels_clamped = labels.clone()
        labels_clamped[labels < 0] = 5  # Map unknown to index 5
        
        # Use unknown label for student phase
        if phase == 'student':
            cond_idx = torch.full_like(labels_clamped, 5)  # Always use "unknown" 
        else:
            cond_idx = labels_clamped  # Use actual labels for teacher
        
        # Get label embeddings using cond_idx instead
        label_emb = self.label_embedding_latent(cond_idx)
        label_emb = F.normalize(label_emb, p=2, dim=1) * 0.1
        
        # Use predict mode for student phase to prevent label leakage
        predict_mode = (phase == 'student')
        
        # Hierarchical encoding with mask awareness and predict mode
        # 1. Short-term
        h_short = self.encode_with_mask(x, mask, cond_idx, self.short_encoder, 
                                       self.short_attention, predict_mode=predict_mode)
        h_short_cond = torch.cat([h_short, label_emb], dim=1)
        short_mu = self.short_mu(h_short_cond)
        short_logvar = torch.clamp(self.short_logvar(h_short_cond), -20, 2)
        short_z = self.reparameterize(short_mu, short_logvar)
        
        # OPTIMIZATION: Skip risk prior computation when not needed (student phase)
        prior_mu_short = prior_logvar_short = None
        prior_mu_mid = prior_logvar_mid = None
        prior_mu_long = prior_logvar_long = None
        
        if phase == 'teacher':
            prior_mu_short, prior_logvar_short = self.risk_prior_short(labels_clamped)
            prior_mu_mid, prior_logvar_mid = self.risk_prior_mid(labels_clamped)
            prior_mu_long, prior_logvar_long = self.risk_prior_long(labels_clamped)
        
        # 2. Mid-term (conditioned on short + label)
        h_mid = self.encode_with_mask(x, mask, cond_idx, self.mid_encoder, 
                                     self.mid_attention, predict_mode=predict_mode)
        h_mid_cond = torch.cat([h_mid, short_z, label_emb], dim=1)
        mid_mu = self.mid_mu(h_mid_cond)
        mid_logvar = torch.clamp(self.mid_logvar(h_mid_cond), -20, 2)
        mid_z = self.reparameterize(mid_mu, mid_logvar)
        
        # 3. Long-term (conditioned on short + mid + label)
        h_long = self.encode_with_mask(x, mask, cond_idx, self.long_encoder, 
                                      self.long_attention, predict_mode=predict_mode)
        h_long_cond = torch.cat([h_long, short_z, mid_z, label_emb], dim=1)
        long_mu = self.long_mu(h_long_cond)
        long_logvar = torch.clamp(self.long_logvar(h_long_cond), -20, 2)
        long_z = self.reparameterize(long_mu, long_logvar)
        
        # Combine all latents
        all_z = torch.cat([short_z, mid_z, long_z], dim=1)
        
        # OPTIMIZATION: Phase-based head selection (compute only what's needed)
        rg_score_pred = None
        rg_cat_logits = None
        manual_logits = None
        ordinal_logits = None
        
        if phase == 'teacher':
            rg_score_pred = self.rg_score_head(all_z).squeeze(-1)
            rg_cat_logits = self.rg_cat_head(all_z)
        elif phase == 'student':
            manual_logits = self.manual_head(all_z)
            ordinal_logits = self.ordinal_head(all_z)
        
        # Optional decode (skip in student training for speed)
        x_recon = None
        if decode:
            avg_mask = mask.mean(dim=1, keepdim=True)  # [B, 1]
            h_decode = torch.cat([all_z, label_emb, avg_mask], dim=1)
            h_decode = self.decoder_initial(h_decode)
            h_decode = self.decoder_blocks(h_decode)
            x_recon = self.decoder_final(h_decode)
            x_recon = x_recon.view(batch_size, self.seq_len, self.input_dim)
            
            # Apply mask to reconstruction
            x_recon = x_recon * mask.unsqueeze(-1)
        
        outputs = {
            'x_recon': x_recon,
            'latents': {
                'short': (short_mu, short_logvar, short_z),
                'mid': (mid_mu, mid_logvar, mid_z),
                'long': (long_mu, long_logvar, long_z)
            },
            'z_all': all_z,
            'prior_short': (prior_mu_short, prior_logvar_short),
            'prior_mid': (prior_mu_mid, prior_logvar_mid),
            'prior_long': (prior_mu_long, prior_logvar_long),
            'rg_score_pred': rg_score_pred,
            'rg_cat_logits': rg_cat_logits,
            'manual_logits': manual_logits,
            'ordinal_logits': ordinal_logits
        }
        
        return outputs
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
   
    def get_embeddings(self, x, mask, labels, phase='student'):
        """Extract embeddings for downstream tasks (label-agnostic by default)."""
        with torch.no_grad():
            # Don't decode when extracting embeddings for efficiency
            outputs = self.forward(x, mask, labels, phase=phase, decode=False)
            embeddings = outputs['z_all']
        return embeddings

# ============================================================================
# OPTIMIZED Loss Functions
# ============================================================================

def compute_teacher_losses(outputs, batch, config, kl_weight=1.0, running_means=None):
    """Compute losses for Phase 1 Teacher training."""
    losses = {}
    device = batch['features'].device
    
    x = batch['features']
    mask = batch['mask']
    rg_score = batch['rg_score']
    rg_cat_dist = batch['rg_cat_dist']
    labels = batch['risk_label']
    
    # 1. Masked reconstruction loss
    x_masked = x * mask.unsqueeze(-1)
    recon_masked = outputs['x_recon'] * mask.unsqueeze(-1)
    
    # Number of valid elements = (valid timesteps) * (num features)
    valid_ts = mask.sum(dim=1, keepdim=True).clamp(min=1)      # [B,1]
    valid_elems = valid_ts * x.shape[2]                        # [B,1]
    
    recon_loss = F.mse_loss(recon_masked, x_masked, reduction='none')
    recon_loss = (recon_loss.sum(dim=[1,2]) / valid_elems.squeeze()).mean()
    losses['recon'] = recon_loss
    
    # 2. Hierarchical KL losses with risk-specific priors
    kl_weights = config.get('kl_weights', [0.001, 0.01, 0.05])
    free_bits = config.get('free_bits', [5.0, 3.0, 2.0])
    
    prior_lookup = {
        'short': outputs['prior_short'],
        'mid': outputs['prior_mid'],
        'long': outputs['prior_long']
    }
    
    for i, level in enumerate(['short', 'mid', 'long']):
        mu, logvar, z = outputs['latents'][level]
        prior_mu, prior_logvar = prior_lookup[level]
        
        # Safety check - priors should always be computed in teacher phase
        assert prior_mu is not None and prior_logvar is not None, \
            "Prior stats are None; compute_teacher_losses should only run in teacher phase."
        
        prior_var = torch.exp(torch.clamp(prior_logvar, -20, 2))
        var = torch.exp(torch.clamp(logvar, -20, 2))
        
        # Correct free-bits implementation (ReLU(KL - τ) not clamp)
        kl_per_dim = 0.5 * (prior_logvar - logvar + (var + (mu - prior_mu).pow(2)) / (prior_var + 1e-8) - 1)
        free_per_dim = free_bits[i] / kl_per_dim.shape[1]  # τ per-dimension
        kl = F.relu(kl_per_dim - free_per_dim).sum(dim=1).mean()
        
        losses[f'kl_{level}'] = kl_weights[i] * kl_weight * kl
    
    # 3. RG Score regression loss (FIXED)
    if outputs['rg_score_pred'] is not None:
        # Use window coverage instead of > 0 check
        if 'window_coverage' in batch:
            rg_mask = batch['window_coverage'] > 0
        else:
            rg_mask = rg_score >= 0  # Allow zeros as valid
        
        if rg_mask.any():
            score_loss = F.mse_loss(outputs['rg_score_pred'][rg_mask], rg_score[rg_mask])
            losses['rg_score'] = config.get('rg_score_weight', 0.5) * score_loss
    
    # 4. RG Category distribution loss
    if outputs['rg_cat_logits'] is not None:
        rg_valid = rg_cat_dist.sum(dim=1) > 0.99
        if rg_valid.any():
            # Use log_softmax directly to avoid duplicate computation
            pred_logp = F.log_softmax(outputs['rg_cat_logits'][rg_valid], dim=-1)
            target_dist = rg_cat_dist[rg_valid]
            target_dist = target_dist / (target_dist.sum(dim=1, keepdim=True) + 1e-8)
            
            kl_dist = F.kl_div(pred_logp, target_dist, reduction='batchmean')
            losses['rg_dist'] = config.get('rg_dist_weight', 1.0) * kl_dist
            
            # Modal class cross-entropy
            rg_modal_class = rg_cat_dist[rg_valid].argmax(dim=1)
            modal_ce_loss = F.cross_entropy(
                outputs['rg_cat_logits'][rg_valid],
                rg_modal_class,
                reduction='mean'
            )
            losses['rg_modal_ce'] = config.get('rg_modal_weight', 0.2) * modal_ce_loss
    
    # 5. OPTIMIZATION: Streamlined diversity loss computation
    try:
        all_z = outputs['z_all']
        if all_z.shape[0] > 1:  # Need at least 2 samples
            div_loss = -torch.log(torch.det(torch.cov(all_z.T) + 1e-6 * torch.eye(all_z.shape[1], device=device)))
            losses['diversity'] = config.get('diversity_weight', 0.2) * div_loss
    except:
        pass  # Skip diversity if computation fails
    
    # 6. Total loss
    total = losses.get('recon', 0)
    for level in ['short', 'mid', 'long']:
        total = total + losses.get(f'kl_{level}', 0)
    total = total + losses.get('rg_score', 0) + losses.get('rg_dist', 0)
    total = total + losses.get('rg_modal_ce', 0) + losses.get('diversity', 0)
    
    losses['total'] = total
    return losses

def compute_ce_only_optimized(outputs, batch, class_weights=None, use_ordinal=False, ordinal_weight=0.05):
    """SIMPLIFIED: Plain CE loss for labeled data only."""
    losses = {}
    device = outputs['manual_logits'].device
    
    # Only process windows with manual labels (exclude augmented)
    manual_mask = batch['has_manual_bool'] & ~batch.get('is_augmented', torch.zeros_like(batch['has_manual_bool']))
    
    if manual_mask.any().item() and outputs['manual_logits'] is not None:
        logits = outputs['manual_logits'][manual_mask]
        targets = batch['manual_labels_multihot'][manual_mask].argmax(dim=1)
        
        # SIMPLIFIED: Just plain CE (let sampler handle balance)
        ce_loss = F.cross_entropy(logits, targets)
        
        losses['ce'] = ce_loss
        total = ce_loss
        
        # Optional ordinal shaping (after warmup)
        if use_ordinal and outputs.get('ordinal_logits') is not None:
            ordinal_logits = outputs['ordinal_logits'][manual_mask]
            ordinal_targets = compute_ordinal_targets(batch['manual_labels_multihot'][manual_mask])
            ord_loss = F.binary_cross_entropy_with_logits(ordinal_logits, ordinal_targets, reduction='mean')
            losses['ordinal'] = ordinal_weight * ord_loss
            total = total + losses['ordinal']
        
        losses['total'] = total
    else:
        losses['total'] = torch.tensor(0.0, device=device)
    
    return losses

# ============================================================================
# NEW: Optional Calendar Warmup Function
# ============================================================================

def calendar_warmup(model, data_dir, data_type, device, config, epochs=3):
    """
    Optional 3-minute unsupervised adaptation to calendar windows.
    Only unfreezes BatchNorm layers for stability.
    """
    print("\n" + "="*60)
    print("OPTIONAL: 3-MINUTE CALENDAR WARMUP")
    print("="*60)
    
    # Load unlabeled calendar data
    calendar_dataset = OptimizedTeacherStudentDataset(
        window_type='calendar',
        split='train',
        data_dir=data_dir,
        data_type=data_type,
        phase='teacher',
        use_scaled=False,
        include_unlabeled=True  # Include all windows
    )
    
    # Sample 20% for speed
    n_samples = int(len(calendar_dataset) * 0.2)
    indices = torch.randperm(len(calendar_dataset))[:n_samples]
    calendar_subset = torch.utils.data.Subset(calendar_dataset, indices)
    
    calendar_loader = DataLoader(
        calendar_subset,
        batch_size=256,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == 'cuda')
    )
    
    print(f"  Using {n_samples}/{len(calendar_dataset)} samples")
    
    # Only unfreeze BatchNorm layers
    for name, param in model.named_parameters():
        if 'bn' in name or 'BatchNorm' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    bn_params = [p for p in model.parameters() if p.requires_grad]
    print(f"  Unfrozen {len(bn_params)} BatchNorm parameters")
    
    # Simple optimizer
    optimizer = torch.optim.Adam(bn_params, lr=1e-4)
    
    model.train()
    for epoch in range(epochs):
        epoch_losses = []
        
        for batch in tqdm(calendar_loader, desc=f"Warmup Epoch {epoch+1}/{epochs}"):
            # Move to device
            x = batch['features'].to(device, non_blocking=True)
            mask = batch['mask'].to(device, non_blocking=True)
            labels = torch.full((x.shape[0],), 5, dtype=torch.long, device=device)  # Unknown
            
            optimizer.zero_grad()
            
            # Forward pass with reconstruction
            outputs = model(x, mask, labels, phase='teacher', decode=True)
            
            # Simple reconstruction loss
            x_masked = x * mask.unsqueeze(-1)
            recon_masked = outputs['x_recon'] * mask.unsqueeze(-1)
            recon_loss = F.mse_loss(recon_masked, x_masked)
            
            if torch.isfinite(recon_loss):
                recon_loss.backward()
                optimizer.step()
                epoch_losses.append(float(recon_loss.item()))
        
        avg_loss = np.mean(epoch_losses)
        print(f"  Epoch {epoch+1}: Recon Loss = {avg_loss:.4f}")
    
    # Freeze everything again
    for param in model.parameters():
        param.requires_grad = False
    
    print("✅ Calendar warmup complete")
    return model

# ============================================================================
# OPTIMIZED Training Functions  
# ============================================================================

def train_teacher_optimized(model, train_loader, val_loader, holdout_loader, config, device, output_dir, start_epoch=0, resume_checkpoint=None):
    """OPTIMIZED teacher training with FIXED validation."""
    print("\n" + "="*60)
    print("PHASE 1: OPTIMIZED TEACHER TRAINING")
    print("="*60)
    
    # Configure optimizer with version-robust AdamW
    optimizer = make_adamw(
        model.parameters(),
        lr=config['teacher_lr'],
        weight_decay=config.get('weight_decay', 1e-3),
        device=device
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config['teacher_epochs'], 
        eta_min=1e-7
    )
    
    # OPTIMIZATION: Enhanced mixed precision with version-robust GradScaler
    use_amp = (device.type == 'cuda' and torch.cuda.is_available())
    scaler = make_grad_scaler(device, enabled=use_amp)
    autocast_ctx = get_autocast(device)
    
    history = {
        'train_loss': [], 'val_loss': [], 'holdout_loss': [],
        'train_pr': [], 'val_pr': [], 'holdout_pr': [],
        'train_rg_score_mae': [], 'val_rg_score_mae': [], 'holdout_rg_score_mae': [],
        'train_rg_dist_acc': [], 'val_rg_dist_acc': []
    }
    
    best_val_loss = float('inf')
    best_model_state = None
    patience = 0
    max_patience = 15
    
    # Resume logic
    if resume_checkpoint is not None:
        checkpoint = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        print(f"✔ Resumed from epoch {checkpoint.get('epoch', 0)}")

    for epoch in range(start_epoch, config['teacher_epochs']):
        # KL annealing
        if config.get('annealing_type') == 'linear':
            kl_weight = min(1.0, (epoch + 1) / config.get('warmup_epochs', 20))
        else:
            kl_weight = get_kl_weight(epoch, config.get('warmup_epochs', 20))
        
        # Training
        model.train()
        train_losses = []
        train_prs = []
        
        # OPTIMIZATION: Streamlined training loop
        pbar = tqdm(train_loader, desc=f"Teacher Epoch {epoch+1}/{config['teacher_epochs']}")
        for batch in pbar:
            # OPTIMIZATION: Batch device transfer at once
            batch_device = {}
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch_device[k] = v.to(device, non_blocking=True)
                else:
                    batch_device[k] = v
            
            optimizer.zero_grad()
            
            # OPTIMIZATION: Use autocast more efficiently
            with autocast_ctx(enabled=use_amp):
                outputs = model(batch_device['features'], batch_device['mask'], 
                               batch_device['risk_label'], phase='teacher', decode=True)
                losses = compute_teacher_losses(outputs, batch_device, config, kl_weight)
            
            if torch.isfinite(losses['total']):
                if scaler is not None:
                    scaler.scale(losses['total']).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    losses['total'].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                train_losses.append(float(losses['total'].item()))
                
                # Calculate participation ratio
                if 'z_all' in outputs:
                    pr = calculate_participation_ratio(outputs['z_all'])
                    train_prs.append(float(pr.item()))
            
            pbar.set_postfix(safe_postfix_dict(losses))
        
        scheduler.step()
        
        # FIXED VALIDATION: Use teacher objectives, not student!
        model.eval()
        val_losses = []
        val_prs = []
        
        with torch.no_grad(), autocast_ctx(enabled=use_amp):
            for batch in tqdm(val_loader, desc="Validation"):
                # Move to device
                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch[k] = v.to(device, non_blocking=True)
                
                # Validate on TEACHER objectives
                outputs = model(batch['features'], batch['mask'], batch['risk_label'],
                              phase='teacher', decode=True)  # Teacher phase!
                losses = compute_teacher_losses(outputs, batch, config, kl_weight=1.0)
                
                if torch.isfinite(losses['total']):
                    val_losses.append(float(losses['total'].item()))
                
                if 'z_all' in outputs:
                    pr = calculate_participation_ratio(outputs['z_all'])
                    val_prs.append(float(pr.item()))
        
        # Calculate epoch metrics
        train_loss = np.mean(train_losses) if train_losses else 0
        val_loss = np.mean(val_losses) if val_losses else 0
        train_pr = np.mean(train_prs) if train_prs else 0
        val_pr = np.mean(val_prs) if val_prs else 0
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_pr'].append(train_pr)
        history['val_pr'].append(val_pr)
        
        print(f"\nEpoch {epoch+1}: Loss={val_loss:.4f}, PR={val_pr:.2f}, KL_w={kl_weight:.3f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience = 0
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_pr': val_pr,
                'config': config,
                'history': history
            }
            torch.save(checkpoint, os.path.join(output_dir, 'teacher_best.pt'))
        else:
            patience += 1
        
        # Early stopping
        if patience >= max_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history

def train_student_optimized(model, train_loader, val_loader, holdout_loader, config, device, output_dir, 
                  resume_checkpoint=None, train_loader_unlabeled=None):
    """SIMPLIFIED student training with MINIMAL FIX: unfreeze long_mu and long_logvar."""
    print("\n" + "="*60)
    print("PHASE 2: STUDENT TRAINING WITH MINIMAL FIX")
    print("="*60)
    
    # Enable performance mode
    enable_perf_mode_for_student()
    
    # Initialize training state
    start_epoch = 0
    best_val_ba = 0
    
    if resume_checkpoint is not None:
        print(f"\n{'='*60}")
        print(f"RESUMING STUDENT FROM CHECKPOINT")
        print('='*60)
        
        checkpoint = safe_load(resume_checkpoint, device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_ba = checkpoint.get('val_ba', 0)
        print(f"✔ Will resume from epoch {start_epoch}")
    
    # Get the underlying dataset from the DataLoader
    if hasattr(train_loader, 'dataset'):
        train_dataset_student = train_loader.dataset
    else:
        raise ValueError("Cannot access dataset from train_loader")
    
    if hasattr(val_loader, 'dataset'):
        val_dataset_student = val_loader.dataset
    else:
        raise ValueError("Cannot access dataset from val_loader")
    
    # Print class distribution
    print("\nClass distribution in training data:")
    labels = train_dataset_student.manual_labels_multihot
    for i in range(5):
        count = (labels[:, i] > 0).sum()
        print(f"  Class {i}: {count} samples ({count/len(labels)*100:.1f}%)")
    
    # PRECOMPUTE ALL EMBEDDINGS ONCE
    # PRECOMPUTE BACKBONE FOR LONG ADAPTATION
    print("\n" + "="*60)
    print("PRECOMPUTING BACKBONE (S/M/H_long) FOR ADAPTIVE TRAINING")
    print("="*60)
    
    fp = encoder_fingerprint(model)  # <-- ADD THIS LINE


    # Precompute deterministic short/mid and h_long features
    print(f"Computing backbone features for long adaptation...")
    Zs_train, Zm_train, Hl_train, Ymh_train, HM_train, AUG_train = precompute_backbone_for_long(
        model, train_dataset_student, device, batch_size=1024
    )
    Zs_val, Zm_val, Hl_val, Ymh_val, HM_val, AUG_val = precompute_backbone_for_long(
        model, val_dataset_student, device, batch_size=1024
    )

    # Save for potential reuse
    backbone_cache = {
        'train': (Zs_train, Zm_train, Hl_train, Ymh_train, HM_train, AUG_train),
        'val': (Zs_val, Zm_val, Hl_val, Ymh_val, HM_val, AUG_val),
        'fingerprint': fp
    }
    torch.save(backbone_cache, os.path.join(output_dir, f"backbone_cache_{fp}.pt"))
    print(f"Cached backbone features (fingerprint: {fp})")

    
    # Create embedding datasets
    embed_train_dataset = LongAdapterDataset(Zs_train, Zm_train, Hl_train, Ymh_train, HM_train, AUG_train)
    embed_val_dataset = LongAdapterDataset(Zs_val, Zm_val, Hl_val, Ymh_val, HM_val, AUG_val)
    
    # Assert alignment
    assert len(embed_train_dataset) == len(train_dataset_student), \
        "Sampler weights must match embedding dataset length"
    
    # Create new loaders with precomputed embeddings
    sampler, class_weights_dataset = create_weighted_sampler_for_student(train_dataset_student, verify=False)
    
    # Use BatchSampler for better per-batch coverage
    batch_sampler = BatchSampler(sampler, batch_size=config.get('student_batch_size', config['batch_size']), 
                                 drop_last=True)
    
    embed_train_loader = make_loader(
        embed_train_dataset,
        batch_sampler=batch_sampler,
        num_workers=0
    )
    
    embed_val_loader = make_loader(
        embed_val_dataset,
        batch_size=config.get('student_batch_size', config['batch_size']),
        shuffle=False,
        num_workers=0
    )
    
    print("✅ Embeddings precomputed! Now training with minimal unfreezing")
    
    # Initialize manual head biases to log class priors
    with torch.no_grad():
        counts = torch.tensor(train_dataset_student.manual_labels_multihot.sum(axis=0), dtype=torch.float32)
        priors = counts / counts.sum()
        init_bias = torch.log(priors.clamp_min(1e-6))
        # Last layer of manual_head
        if isinstance(model.manual_head[-1], nn.Linear):
            model.manual_head[-1].bias.copy_(init_bias.to(device))
            print(f"Initialized manual head biases to log priors: {init_bias.numpy()}")
    
    max_patience = config.get('student_patience', 30)
    
    print(f"\n Settings:")
    print(f"  Max epochs: {config.get('student_epochs', 50)}")
    print(f"  Early stopping patience: {max_patience}")
    
    # ==================================================================
    # MINIMAL FIX: Configure trainable parameters
    # ==================================================================
    print("\n" + "="*60)
    print("MINIMAL FIX: Unfreezing heads + final encoder projections")
    print("="*60)
    
    for name, param in model.named_parameters():
        param.requires_grad = False  # Start frozen
        
        if 'manual_head' in name:  # Keep training heads
            param.requires_grad = True
            print(f"  ✔ Training: {name}")
        # MINIMAL FIX: Also unfreeze final projections
        elif any(x in name for x in ['long_mu', 'long_logvar','mid_mu', 'mid_logvar', 'short_mu', 'short_logvar', 'label_embedding_latent']):
            param.requires_grad = True
            print(f"  ✔ UNFROZEN: {name} (for calendar adaptation)")
    
    student_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Total trainable parameters: {sum(p.numel() for p in student_params):,}")
    print("This minimal change lets the model adapt latent space to calendar windows")
    
    # FIXED: Proper learning rate
    # STAGE A: Discriminative learning rates
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and 'manual_head' in n]
    long_proj_params = [p for n, p in model.named_parameters() if p.requires_grad and ('long_mu' in n or 'long_logvar' in n)]
    mid_short_proj_params = [p for n, p in model.named_parameters() if p.requires_grad and any(k in n for k in ['mid_mu', 'mid_logvar', 'short_mu', 'short_logvar'])]
    label_emb_params = [p for n, p in model.named_parameters() if p.requires_grad and 'label_embedding_latent' in n]
    
    optimizer = torch.optim.AdamW([
        {'params': head_params, 'lr': 1e-3, 'weight_decay': 1e-5},
        {'params': long_proj_params, 'lr': 7e-4, 'weight_decay': 1e-5},
        {'params': mid_short_proj_params, 'lr': 3e-4, 'weight_decay': 1e-5},
        {'params': label_emb_params, 'lr': 1e-4, 'weight_decay': 0.0},
    ])
    print(f"  Discriminative LRs: head=1e-3, long=7e-4, mid/short=3e-4, label_emb=1e-4")
    
    # FIXED: No warm restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config.get('student_epochs', 50), 
        eta_min=1e-6
    )
    print("  Using smooth CosineAnnealingLR (no restarts)")
    
    # Mixed precision setup
    use_amp = (device.type == 'cuda' and torch.cuda.is_available())
    scaler = make_grad_scaler(device, enabled=use_amp)
    autocast_ctx = get_autocast(device)
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [], 'holdout_loss': [],
        'train_acc': [], 'val_acc': [], 'holdout_acc': [],
        'train_ba': [], 'val_ba': [], 'holdout_ba': [],
        'val_recall_per_class': [],
        'train_recall_per_class': []
    }
    
    best_model_state = None
    patience = 0
    warmup_epochs = config.get('warmup_epochs', 0)  # No warmup needed
    
    # Training loop - NO CHANGES except computing gradients for encoder params
    for epoch in range(start_epoch, config.get('student_epochs', 50)):
        # NO ordinal loss - keep it simple
        use_ordinal = False
        
        # STAGE A: Recache embeddings every 2 epochs when projections change
        if epoch > 0 and epoch % 2 == 0:
            fp_now = encoder_fingerprint(model)
            if fp_now != backbone_cache['fingerprint']:
                print(f"\n↻ RECACHING at epoch {epoch+1}")
                Zs_train, Zm_train, Hl_train, Ymh_train, HM_train, AUG_train = precompute_backbone_for_long(model, train_dataset_student, device, batch_size=1024)
                Zs_val, Zm_val, Hl_val, Ymh_val, HM_val, AUG_val = precompute_backbone_for_long(model, val_dataset_student, device, batch_size=1024)
                embed_train_dataset = LongAdapterDataset(Zs_train, Zm_train, Hl_train, Ymh_train, HM_train, AUG_train)
                embed_val_dataset = LongAdapterDataset(Zs_val, Zm_val, Hl_val, Ymh_val, HM_val, AUG_val)
                sampler, _ = create_weighted_sampler_for_student(train_dataset_student, verify=False)
                batch_sampler = BatchSampler(sampler, batch_size=config.get('student_batch_size', config['batch_size']), drop_last=True)
                embed_train_loader = make_loader(embed_train_dataset, batch_sampler=batch_sampler, num_workers=0)
                embed_val_loader = make_loader(embed_val_dataset, batch_size=config.get('student_batch_size', config['batch_size']), shuffle=False, num_workers=0)
                backbone_cache['fingerprint'] = fp_now
                print("✅ Recache complete!")
        
        #print(f"\nEpoch {epoch+1}: Minimal training (heads + long projections)")
        print(f"\nEpoch {epoch+1}: Stage A training (7 unfrozen layers)")
        
        # Set model mode
        model.train()
        # Keep most encoders in eval
        for module in [model.short_encoder, model.mid_encoder, model.long_encoder,
                      model.decoder_initial, model.decoder_blocks, model.decoder_final]:
            module.eval()
        # Keep heads and projections in train mode
        model.manual_head.train()
        if hasattr(model, 'long_mu'):
            model.long_mu.train()
            model.long_logvar.train()
        
        train_losses = []
        train_preds = []
        train_targets = []
        
        
        pbar = tqdm(embed_train_loader, desc=f"Student Epoch {epoch+1}/{config['student_epochs']}")
        for batch_idx, b in enumerate(pbar):
            optimizer.zero_grad(set_to_none=True)
            
            # Move precomputed embeddings to device
            # Get precomputed features
            z_short = b['z_short'].to(device, non_blocking=True).float()
            z_mid = b['z_mid'].to(device, non_blocking=True).float()
            h_long = b['h_long'].to(device, non_blocking=True).float()
            mh = b['manual_labels_multihot'].to(device, non_blocking=True)
            hm = b['has_manual_bool'].to(device, non_blocking=True)
            aug = b['is_augmented'].to(device, non_blocking=True)

            manual_mask = hm & ~aug

            with autocast_ctx(enabled=use_amp):
                # Compute long projection WITH GRADIENTS
                batch_size = z_short.size(0)
                label_emb = model.label_embedding_latent.weight[5].unsqueeze(0).expand(batch_size, -1)
                label_emb = F.normalize(label_emb, p=2, dim=1) * 0.1
                
                # Build input to long projection
                long_input = torch.cat([h_long, z_short, z_mid, label_emb], dim=1)
                
                # Compute long latent WITH GRADIENTS (this is the key!)
                long_mu = model.long_mu(long_input)
                long_logvar = model.long_logvar(long_input)
                # Use deterministic mean for stability (or sample if you want more regularization)
                long_z = long_mu
                
                # Concatenate all latents
                z_all = torch.cat([z_short, z_mid, long_z], dim=1)
                
                # Run classification head
                logits = model.manual_head(z_all)
                
                # Optional: log gradient norms on first batch to verify it's working
                if batch_idx == 0 and epoch == start_epoch:
                    with torch.no_grad():
                        grad_norm_mu = sum(p.grad.norm().item() if p.grad is not None else 0 
                                        for p in model.long_mu.parameters())
                        print(f"    Initial |∇long_mu|: {grad_norm_mu:.4f} (should be >0)")
                
                
                # NEW:
                manual_mask = hm & ~aug

                # Base CE loss
                if manual_mask.any():
                    targets = mh[manual_mask].argmax(dim=1)
                    ce_loss = F.cross_entropy(logits[manual_mask], targets)
                else:
                    ce_loss = logits.new_zeros(())

                # Add teacher KD (cheap risk alignment)
                with torch.no_grad():
                    teacher_logits = model.rg_cat_head(z_all)
                T = 2.0
                kd_loss = F.kl_div(
                    F.log_softmax(logits / T, dim=-1),
                    F.softmax(teacher_logits / T, dim=-1),
                    reduction='batchmean'
                ) * (T * T)

                # Add tiny ordinal loss (risk ordering)
                use_ordinal = True
                ordinal_weight = 0.05
                if use_ordinal and manual_mask.any():
                    ord_targets = compute_ordinal_targets(mh[manual_mask].float())
                    ord_logits = model.ordinal_head(z_all[manual_mask])
                    ord_loss = F.binary_cross_entropy_with_logits(ord_logits, ord_targets, reduction='mean')
                else:
                    ord_loss = logits.new_zeros(())

                # Combine
                total = ce_loss + 0.10 * kd_loss + ordinal_weight * ord_loss
            
            if torch.isfinite(total) and total > 0:
                if scaler is not None:
                    scaler.scale(total).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(student_params, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total.backward()
                    torch.nn.utils.clip_grad_norm_(student_params, 1.0)
                    optimizer.step()
            
            if manual_mask.any():
                with torch.no_grad():
                    preds = logits[manual_mask].argmax(dim=1).cpu().numpy()
                    targets_np = mh[manual_mask].argmax(dim=1).cpu().numpy()
                    train_preds.extend(preds)
                    train_targets.extend(targets_np)
            
            train_losses.append(float(total.item()) if torch.isfinite(total) else 0.0)
            pbar.set_postfix({'loss': f"{train_losses[-1]:.3f}"})
        
        # Validation - recompute embeddings
        # Validation - NO recomputation needed anymore
        model.eval()
        val_preds, val_targets, val_losses = [], [], []
        
        with torch.no_grad():
            for b in tqdm(embed_val_loader, desc="Validation"):
                # Get precomputed features
                z_short = b['z_short'].to(device, non_blocking=True).float()
                z_mid = b['z_mid'].to(device, non_blocking=True).float()
                h_long = b['h_long'].to(device, non_blocking=True).float()
                mh = b['manual_labels_multihot'].to(device, non_blocking=True)
                hm = b['has_manual_bool'].to(device, non_blocking=True)
                
                # Compute long projection (deterministic for validation)
                batch_size = z_short.size(0)
                label_emb = model.label_embedding_latent.weight[5].unsqueeze(0).expand(batch_size, -1)
                label_emb = F.normalize(label_emb, p=2, dim=1) * 0.1
                
                long_input = torch.cat([h_long, z_short, z_mid, label_emb], dim=1)
                long_mu = model.long_mu(long_input)
                long_z = long_mu  # deterministic for validation
                
                z_all = torch.cat([z_short, z_mid, long_z], dim=1)
                logits = model.manual_head(z_all)
                
                manual_mask = hm
                
                if manual_mask.any():
                    targets = mh[manual_mask].argmax(dim=1)
                    ce = F.cross_entropy(logits[manual_mask], targets)
                    val_losses.append(float(ce.item()))
                    val_preds.extend(logits[manual_mask].argmax(dim=1).cpu().numpy())
                    val_targets.extend(targets.cpu().numpy())
        
        # Calculate metrics
        train_loss = np.mean(train_losses) if train_losses else 0
        train_acc = np.mean(np.array(train_preds) == np.array(train_targets)) if len(train_preds) > 0 else 0
        train_ba = balanced_accuracy_score(train_targets, train_preds) if len(train_preds) > 0 else 0
        train_recalls = compute_per_class_recall(np.array(train_preds), np.array(train_targets)) if len(train_preds) > 0 else [0]*5
        
        val_loss = np.mean(val_losses) if val_losses else 0
        val_acc = np.mean(np.array(val_preds) == np.array(val_targets)) if len(val_preds) > 0 else 0
        val_ba = balanced_accuracy_score(val_targets, val_preds) if len(val_preds) > 0 else 0
        val_recalls = compute_per_class_recall(np.array(val_preds), np.array(val_targets)) if len(val_preds) > 0 else [0]*5
        
        # Print distribution
        vt = np.array(val_targets)
        vp = np.array(val_preds)
        print(f"\nVal target counts: {np.bincount(vt, minlength=5)}")
        print(f"Val pred counts:   {np.bincount(vp, minlength=5)}")
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_ba'].append(train_ba)
        history['val_ba'].append(val_ba)
        history['train_recall_per_class'].append(train_recalls)
        history['val_recall_per_class'].append(val_recalls)
        
        print(f"\nEpoch {epoch+1}: Loss={val_loss:.4f}, Acc={val_acc:.3f}, BA={val_ba:.3f}")
        print(f"  Train recalls: {[f'{r:.2f}' if not np.isnan(r) else 'N/A' for r in train_recalls]}")
        print(f"  Val recalls: {[f'{r:.2f}' if not np.isnan(r) else 'N/A' for r in val_recalls]}")
        
        # Track learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Learning rate: {current_lr:.2e}")
        
        # Save best model
        if val_ba > best_val_ba:
            best_val_ba = val_ba
            best_model_state = model.state_dict().copy()
            patience = 0
            print(f"  ✅ New best BA: {val_ba:.3f} (saved)")
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_ba': val_ba,
                'val_loss': val_loss,
                'config': config,
                'history': history
            }
            torch.save(checkpoint, os.path.join(output_dir, 'student_best_minimal_fix.pt'))
        elif val_ba > best_val_ba - 0.02:  # Within 2% of best
            patience = max(0, patience - 1)
            print(f"  Near best performance (patience adjusted: {patience}/{max_patience})")
        else:
            patience += 1
            print(f"  No improvement (patience: {patience}/{max_patience})")
        
        scheduler.step()
        
        # Early stopping
        if patience >= max_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history

def plot_training_curves(teacher_history, student_history, output_dir):
    """Plot training curves for both teacher and student phases."""
    plt.style.use('default')
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Teacher-Student Training Progress (Minimal Fix)', fontsize=16)
    
    # Teacher plots
    if teacher_history and len(teacher_history.get('train_loss', [])) > 0:
        epochs_t = range(1, len(teacher_history['train_loss']) + 1)
        
        # Teacher losses
        axes[0, 0].plot(epochs_t, teacher_history['train_loss'], label='Train')
        axes[0, 0].plot(epochs_t, teacher_history['val_loss'], label='Val')
        axes[0, 0].set_title('Teacher Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Teacher PR
        axes[0, 1].plot(epochs_t, teacher_history['train_pr'], label='Train')
        axes[0, 1].plot(epochs_t, teacher_history['val_pr'], label='Val')
        axes[0, 1].set_title('Teacher Participation Ratio')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('PR')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Teacher RG metrics
        if 'train_rg_score_mae' in teacher_history:
            axes[0, 2].plot(epochs_t, teacher_history['train_rg_score_mae'], label='Train MAE')
            axes[0, 2].plot(epochs_t, teacher_history['val_rg_score_mae'], label='Val MAE')
            axes[0, 2].set_title('Teacher RG Score MAE')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('MAE')
            axes[0, 2].legend()
            axes[0, 2].grid(True)
    else:
        for i in range(3):
            axes[0, i].text(0.5, 0.5, 'No Teacher Data', ha='center', va='center', transform=axes[0, i].transAxes)
            axes[0, i].set_title(f'Teacher Plot {i+1}')
    
    # Student plots
    if student_history and len(student_history.get('train_loss', [])) > 0:
        epochs_s = range(1, len(student_history['train_loss']) + 1)
        
        # Student losses
        axes[1, 0].plot(epochs_s, student_history['train_loss'], label='Train')
        axes[1, 0].plot(epochs_s, student_history['val_loss'], label='Val')
        axes[1, 0].set_title('Student Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Student accuracy
        axes[1, 1].plot(epochs_s, student_history['train_acc'], label='Train Acc')
        axes[1, 1].plot(epochs_s, student_history['val_acc'], label='Val Acc')
        axes[1, 1].plot(epochs_s, student_history['train_ba'], label='Train BA')
        axes[1, 1].plot(epochs_s, student_history['val_ba'], label='Val BA')
        axes[1, 1].set_title('Student Accuracy (Minimal Fix)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # Per-class recall
        if 'val_recall_per_class' in student_history and len(student_history['val_recall_per_class']) > 0:
            recalls = np.array(student_history['val_recall_per_class'])
            for cls in range(5):
                if recalls.shape[1] > cls:
                    valid_epochs = ~np.isnan(recalls[:, cls])
                    if valid_epochs.any():
                        axes[1, 2].plot(np.array(epochs_s)[valid_epochs], recalls[valid_epochs, cls], 
                                       label=f'Class {cls}', marker='o', markersize=3)
            axes[1, 2].set_title('Validation Per-Class Recall')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Recall')
            axes[1, 2].legend()
            axes[1, 2].grid(True)
    else:
        for i in range(3):
            axes[1, i].text(0.5, 0.5, 'No Student Data', ha='center', va='center', transform=axes[1, i].transAxes)
            axes[1, i].set_title(f'Student Plot {i+1}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves_minimal_fix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved training curves to {os.path.join(output_dir, 'training_curves_minimal_fix.png')}")

# ============================================================================
# OPTIMIZED Main Function
# ============================================================================

def main():
    # Set seed for reproducibility
    set_seed(42)
    
    parser = argparse.ArgumentParser(description='Train OPTIMIZED Teacher-Student Conditional Hierarchical VAE')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Root directory containing processed data')
    parser.add_argument('--data-type', type=str, required=True,
                       choices=['sessions', 'bets', 'payments', 'transactions'],
                       help='Type of data to process')
    parser.add_argument('--output-dir', type=str, default='models/teacher_student_cvae_optimized',
                       help='Output directory for models')
    parser.add_argument('--teacher-epochs', type=int, default=30,
                       help='Number of teacher training epochs')
    parser.add_argument('--student-epochs', type=int, default=50,
                       help='Number of student training epochs')
    parser.add_argument('--student-patience', type=int, default=30,
                       help='Early stopping patience for student training')
    parser.add_argument('--batch-size', type=int, default=512,
                       help='Batch size')
    parser.add_argument('--student-batch-mult', type=int, default=2,
                       help='Multiplier for student batch size (heads are tiny, can go bigger)')
    parser.add_argument('--teacher-lr', type=float, default=3e-4,
                       help='Teacher learning rate')
    parser.add_argument('--student-lr', type=float, default=5e-4,
                       help='Student learning rate (default: 5e-4 for minimal fix)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    parser.add_argument('--skip-teacher', action='store_true',
                       help='Skip teacher training and load existing teacher model')
    parser.add_argument('--teacher-checkpoint', type=str, default=None,
                       help='Path to existing teacher checkpoint to load')
    parser.add_argument('--resume-student', type=str, default=None,
                        help='Path to student checkpoint to resume from')
    parser.add_argument('--calendar-warmup', action='store_true',
                       help='Run 3-minute calendar warmup before student training')
    
    # Use parse_known_args to handle unknown flags from launchers
    args, _ = parser.parse_known_args(sys.argv[1:])
    
    # Convert to absolute path
    args.data_dir = os.path.abspath(args.data_dir)
    
    # Update output directory to include data type
    args.output_dir = os.path.join(args.output_dir, args.data_type + '_minimal_fix')
    os.makedirs(args.output_dir, exist_ok=True)
    
    # OPTIMIZED Configuration
    config = {
        # Model architecture
        'latent_dims': [64, 48, 32],
        'hidden_dim': 256,
        'label_embed_dim': 16,
        
        # Risk-specific priors
        'prior_init_spread': 5.0,
        
        # Anti-collapse features
        'use_spectral_norm': True,
        'diversity_weight': 2.0,
        'min_encoding_capacity': 20.0,
        'min_pr': 17.0,
        
        # Loss weights
        'kl_weights': [0.01, 0.02, 0.03],
        'free_bits': [1.0, 0.5, 0.25],
        'rg_score_weight': 0.5,
        'rg_dist_weight': 1.0,
        'rg_modal_weight': 0.2,
        'centroid_weight': 0.1,
        
        # KL annealing
        'annealing_type': 'linear',
        'warmup_epochs': 20,
        
        # Total correlation weight
        'tc_weight': 0.1,
        
        # Training
        'batch_size': args.batch_size,
        'student_batch_size': args.batch_size * args.student_batch_mult,
        'teacher_lr': args.teacher_lr,
        'student_lr': args.student_lr,
        'teacher_epochs': args.teacher_epochs,
        'student_epochs': args.student_epochs,
        'student_patience': args.student_patience,
        'weight_decay': 1e-3,
        
        # Student settings (SIMPLIFIED)
        'use_ordinal': False,  # Keep it simple
        'warmup_epochs': 0,
        'ordinal_weight': 0.10,
        
        # Data loading
        'num_workers': 0  # Safe for Windows
    }
    
    print("\n" + "="*60)
    print("TEACHER-STUDENT CVAE v5 WITH MINIMAL FIX")
    print(f"DATA TYPE: {args.data_type.upper()}")
    print("="*60)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Student LR: {config['student_lr']:.2e}")
    print(f"Student batch size: {config['student_batch_size']}")
    print("\n🔧 MINIMAL FIX APPLIED:")
    print("  ✅ Unfreezing long_mu and long_logvar during student training")
    print("  ✅ Everything else stays exactly the same as v5")
    print("  ✅ No complex losses, no risk axis, no ordinal head")
    print("  ✅ Expected BA improvement: 0.34 → 0.45+")
    
    # Initialize variables for both paths
    model = None
    teacher_history = None
    
    if args.skip_teacher:
        # Load existing teacher model
        print("\n" + "="*60)
        print("LOADING EXISTING TEACHER MODEL")
        print("="*60)
        
        # Determine teacher checkpoint path
        if args.teacher_checkpoint:
            teacher_checkpoint_path = args.teacher_checkpoint
        else:
            teacher_checkpoint_path = os.path.join(args.output_dir, 'teacher_best.pt')
            if not os.path.exists(teacher_checkpoint_path):
                teacher_checkpoint_path = os.path.join(args.output_dir, 'teacher_final.pt')
        
        if not os.path.exists(teacher_checkpoint_path):
            raise FileNotFoundError(
                f"Teacher checkpoint not found at {teacher_checkpoint_path}. "
                f"Please specify --teacher-checkpoint or ensure teacher model exists in output directory."
            )
        
        print(f"Loading teacher from: {teacher_checkpoint_path}")
        
        checkpoint = torch.load(teacher_checkpoint_path, map_location=device, weights_only=False)
        
        # Get model configuration from checkpoint or use defaults
        if 'config' in checkpoint:
            stored_config = checkpoint['config']
            # Update only student-related configs
            stored_config.update({
                'student_epochs': args.student_epochs,
                'student_lr': args.student_lr,
                'student_batch_size': args.batch_size * args.student_batch_mult,
                'student_patience': args.student_patience,
                'use_ordinal': False,
                'warmup_epochs': 0,
                'ordinal_weight': 0.10,
                'batch_size': args.batch_size,
                'num_workers': 0
            })
            config = stored_config
        else:
            print("Warning: No config in checkpoint, using defaults")
        
        # Get input dimensions from a dummy dataset
        print("\nDetermining input dimensions...")
        dummy_dataset = OptimizedTeacherStudentDataset(
            window_type='activity',
            split='train',
            data_dir=args.data_dir,
            data_type=args.data_type,
            phase='teacher',
            use_scaled=False
        )
        dummy_loader = DataLoader(dummy_dataset, batch_size=1, num_workers=0)
        sample_batch = next(iter(dummy_loader))
        input_dim = sample_batch['features'].shape[2]
        seq_len = sample_batch['features'].shape[1]
        
        print(f"  Input dim: {input_dim}")
        print(f"  Sequence length: {seq_len}")
        print(f"  Latent dims: {config.get('latent_dims', [64, 48, 32])}")
        
        # Create model with optimized class
        model = OptimizedTeacherStudentCVAE(
            input_dim=input_dim,
            seq_len=seq_len,
            latent_dims=config.get('latent_dims', [64, 48, 32]),
            hidden_dim=config.get('hidden_dim', 256),
            num_classes=5,
            window_type='activity',
            label_embed_dim=config.get('label_embed_dim', 16),
            use_spectral_norm=config.get('use_spectral_norm', True),
            diversity_weight=config.get('diversity_weight', 2.0),
            min_encoding_capacity=config.get('min_encoding_capacity', 20.0),
            temporal_consistency_weight=0.1,
            tc_weight=0.05,
            centroid_weight=config.get('centroid_weight', 0.1),
            prior_init_spread=config.get('prior_init_spread', 5.0),
            dropout=0.2
        ).to(device)
        
        print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Load teacher weights
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Teacher model loaded successfully")
        
        # Create empty teacher history for compatibility
        teacher_history = {
            'train_loss': [], 'val_loss': [], 'holdout_loss': [],
            'train_pr': [], 'val_pr': [], 'holdout_pr': [],
            'train_rg_score_mae': [], 'val_rg_score_mae': [], 'holdout_rg_score_mae': [],
            'train_rg_dist_acc': [], 'val_rg_dist_acc': []
        }
        
        if 'teacher_history' in checkpoint:
            teacher_history = checkpoint['teacher_history']
            print(f"  Loaded teacher training history ({len(teacher_history.get('train_loss', []))} epochs)")
        
        print("\n✅ Ready to proceed with student training (minimal fix)")
        
    else:
        # Phase 1: Teacher Training
        print(f"\nLoading {args.data_type} activity window datasets for teacher training...")

        train_dataset = OptimizedTeacherStudentDataset(
            window_type='activity',
            split='train',
            data_dir=args.data_dir,
            data_type=args.data_type,
            phase='teacher',
            use_scaled=False
        )

        val_dataset = OptimizedTeacherStudentDataset(
            window_type='activity',
            split='val',
            data_dir=args.data_dir,
            data_type=args.data_type,
            phase='teacher',
            use_scaled=False
        )
        
        # OPTIMIZATION: Enable workers and pin_memory
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            pin_memory=(device.type == 'cuda'),
            persistent_workers=(config['num_workers'] > 0)
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=(device.type == 'cuda'),
            persistent_workers=(config['num_workers'] > 0)
        )
        
        # Holdout dataset
        holdout_dataset = None
        holdout_loader = None
        if os.path.exists(os.path.join(args.data_dir, args.data_type, 'activity', 'X_holdout_activity.npy')):
            holdout_dataset = OptimizedTeacherStudentDataset(
                window_type='activity',
                split='holdout',
                data_dir=args.data_dir,
                data_type=args.data_type,
                phase='teacher',
                use_scaled=False
            )
            print(f"  Loaded holdout dataset: {len(holdout_dataset)} windows")
            
            holdout_loader = DataLoader(
                holdout_dataset,
                batch_size=config['batch_size'],
                shuffle=False,
                num_workers=config['num_workers'],
                pin_memory=(device.type == 'cuda'),
                persistent_workers=(config['num_workers'] > 0)
            )
        
        # Get feature dimensions
        sample_batch = next(iter(train_loader))
        input_dim = sample_batch['features'].shape[2]
        seq_len = sample_batch['features'].shape[1]
        
        print(f"\nModel configuration:")
        print(f"  Input dim: {input_dim}")
        print(f"  Sequence length: {seq_len}")
        print(f"  Latent dims: {config['latent_dims']}")
        
        # Create optimized model
        model = OptimizedTeacherStudentCVAE(
            input_dim=input_dim,
            seq_len=seq_len,
            latent_dims=config['latent_dims'],
            hidden_dim=config['hidden_dim'],
            num_classes=5,
            window_type='activity',
            label_embed_dim=config['label_embed_dim'],
            use_spectral_norm=config['use_spectral_norm'],
            diversity_weight=config['diversity_weight'],
            min_encoding_capacity=config['min_encoding_capacity'],
            temporal_consistency_weight=0.1,
            tc_weight=0.05,
            centroid_weight=config['centroid_weight'],
            prior_init_spread=config['prior_init_spread'],
            dropout=0.2
        ).to(device)
        
        print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Initialize RG score head bias
        with torch.no_grad():
            model.rg_score_head[-1].bias.data.fill_(0.33)
        print("  Initialized RG score head bias to 0.33")
        
        # Resume from checkpoint if provided
        start_epoch = 0
        best_checkpoint = None
        if args.resume and os.path.exists(args.resume):
            print(f"\nResuming from checkpoint: {args.resume}")
            
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            
            print(f"✔ Will resume from epoch {start_epoch}")
            best_checkpoint = checkpoint
        
        # Train teacher with FIXED validation
        model, teacher_history = train_teacher_optimized(
            model, train_loader, val_loader, holdout_loader, 
            config, device, args.output_dir,
            start_epoch=start_epoch,
            resume_checkpoint=best_checkpoint
        )
        
        # Save teacher model
        teacher_path = os.path.join(args.output_dir, 'teacher_final_optimized.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'teacher_history': teacher_history
        }, teacher_path)
        print(f"\nSaved optimized teacher model to {teacher_path}")
    
    # Optional: Calendar warmup
    if args.calendar_warmup:
        model = calendar_warmup(model, args.data_dir, args.data_type, device, config)
    
    # Phase 2: Student Training with MINIMAL FIX
    print(f"\nLoading {args.data_type} calendar window datasets for student training...")

    train_dataset_student = OptimizedTeacherStudentDataset(
        window_type='calendar',
        split='train',
        data_dir=args.data_dir,
        data_type=args.data_type,
        phase='student',
        use_scaled=False,
        augment_centered_windows=False,
        include_unlabeled=False
    )

    val_dataset_student = OptimizedTeacherStudentDataset(
        window_type='calendar',
        split='val',
        data_dir=args.data_dir,
        data_type=args.data_type,
        phase='student',
        use_scaled=False,
        include_unlabeled=False
    )
    
    # Holdout dataset for student
    holdout_dataset_student = None
    holdout_loader_student = None
    if os.path.exists(os.path.join(args.data_dir, args.data_type, 'calendar', 'X_holdout_calendar.npy')):
        holdout_dataset_student = OptimizedTeacherStudentDataset(
            window_type='calendar',
            split='holdout',
            data_dir=args.data_dir,
            data_type=args.data_type,
            phase='student',
            use_scaled=False,
            include_unlabeled=False
        )
        print(f"  Loaded student holdout dataset: {len(holdout_dataset_student)} windows")
        
        holdout_loader_student = DataLoader(
            holdout_dataset_student,
            batch_size=config['batch_size'] // 2,
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=(device.type == 'cuda'),
            persistent_workers=(config['num_workers'] > 0)
        )
    
    # Use weighted sampler for balanced training
    sampler, class_weights_dataset = create_weighted_sampler_for_student(
        train_dataset_student, 
        verify=True
    )
    
    # Generator for reproducibility
    g = torch.Generator()
    g.manual_seed(42)

    # Create data loaders
    train_loader_student = DataLoader(
        train_dataset_student,
        batch_size=config['batch_size'] // 2,
        sampler=sampler,
        num_workers=config['num_workers'],
        pin_memory=(device.type == 'cuda'),
        generator=g,
        persistent_workers=(config['num_workers'] > 0)
    )

    val_loader_student = DataLoader(
        val_dataset_student,
        batch_size=config['batch_size'] // 2,
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(config['num_workers'] > 0)
    )
    
    # Train student with MINIMAL FIX
    student_resume_path = None
    if args.resume_student:
        student_resume_path = args.resume_student
    elif os.path.exists(os.path.join(args.output_dir, 'student_best_minimal_fix.pt')):
        student_resume_path = os.path.join(args.output_dir, 'student_best_minimal_fix.pt')
        print(f"Found existing student checkpoint: {student_resume_path}")

    model, student_history = train_student_optimized(
        model, train_loader_student, val_loader_student, holdout_loader_student,
        config, device, args.output_dir,
        resume_checkpoint=student_resume_path,
        train_loader_unlabeled=None  # Disabled for simplicity
    )
    
    # Save final model
    final_path = os.path.join(args.output_dir, 'teacher_student_final_minimal_fix.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'teacher_history': teacher_history,
        'student_history': student_history,
        'fixes_applied': [
            'minimal_fix_unfreeze_long_projections',
            'no_risk_axis',
            'no_ordinal_head',
            'no_complex_losses',
            'simple_ce_only'
        ]
    }, final_path)
    print(f"\n✅ Saved final model with minimal fix to {final_path}")
    
    # Plot training curves
    plot_training_curves(teacher_history, student_history, args.output_dir)
    
    print("\n🎉 TRAINING COMPLETE WITH MINIMAL FIX!")
    print("Expected BA improvement: 0.34 → 0.45-0.50")
    print("Achieved by unfreezing just long_mu and long_logvar")

if __name__ == "__main__":
    main()