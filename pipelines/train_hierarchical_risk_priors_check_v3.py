#!/usr/bin/env python3
"""
Train Conditional Hierarchical VAE on Gambling Data with Risk-Specific Priors
=============================================================================

This script trains a conditional hierarchical VAE (CVAE) where each risk level
has its own learned prior distribution N(μ_y, Σ_y), preventing overlap between
risk levels in latent space and solving the permutation problem.

Key improvements:
- Risk-specific Gaussian priors instead of single N(0,I)
- Centroid loss to maintain separation
- Proper KL divergence for conditional priors
- AGGRESSIVE ANTI-COLLAPSE MEASURES for resumed training

Safety Features (v2):
- Automatic PR collapse detection when resuming
- Monotonic KL annealing by default (no dangerous cycles)
- Minimum KL weight of 0.01 even in cyclical mode
- Nuclear anti-collapse mode activates if PR < 15
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
from datetime import datetime
import gc
import time


# Clear any existing allocations
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ============================================================================
# Helper Functions for Anti-Collapse and MI Estimation
# ============================================================================

def spectral_norm(module, name='weight'):
    """Apply spectral normalization to a module."""
    if hasattr(module, name):
        return nn.utils.spectral_norm(module, name=name)
    return module


def get_kl_weight(epoch, total_epochs, annealing_type='cyclical', cycle_length=None, warmup_epochs=5):
    """
    Calculate KL weight for various annealing strategies.
    """
    if epoch < warmup_epochs:
        # Warmup phase - very small KL weight
        return 0.01 * (epoch / warmup_epochs)
    
    adjusted_epoch = epoch - warmup_epochs
    adjusted_total = total_epochs - warmup_epochs
    
    if annealing_type == 'linear':
        return min(1.0, adjusted_epoch / (adjusted_total * 0.3))
    elif annealing_type == 'cyclical':
        if cycle_length is None:
            cycle_length = adjusted_total // 4
        cycle_position = adjusted_epoch % cycle_length
        # SAFETY: Never let KL weight go below 0.01 to prevent collapse
        return max(0.01, min(1.0, cycle_position / (cycle_length * 0.5)))
    elif annealing_type == 'monotonic':
        # GENTLER monotonic with smaller steps
        progress = adjusted_epoch / adjusted_total
        if progress < 0.2:
            return 0.01
        elif progress < 0.4:
            return 0.05
        elif progress < 0.6:
            return 0.1
        elif progress < 0.8:
            return 0.2
        else:
            return 0.3


def calculate_participation_ratio(embeddings):
    """
    Calculate the participation ratio of embeddings to measure latent space usage.
    """
    if len(embeddings) < 2:
        return 0.0
    
    try:
        # Convert to numpy if tensor
        if torch.is_tensor(embeddings):
            embeddings = embeddings.cpu().numpy()
        
        # Check for NaN or Inf
        if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
            print("Warning: NaN or Inf in embeddings, returning PR=0")
            return 0.0
        
        # Center the embeddings
        embeddings_centered = embeddings - embeddings.mean(axis=0)
        
        # Add small regularization for numerical stability
        embeddings_centered = embeddings_centered + 1e-8 * np.random.randn(*embeddings_centered.shape)
        
        # Calculate covariance matrix
        cov = np.cov(embeddings_centered.T)
        
        # Add small diagonal regularization for numerical stability
        cov = cov + 1e-6 * np.eye(cov.shape[0])
        
        # Try to compute eigenvalues
        try:
            # Use eigh instead of eigvalsh for better stability
            eigenvalues, _ = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            # Fallback: use SVD which is more stable
            try:
                _, s, _ = np.linalg.svd(embeddings_centered, full_matrices=False)
                eigenvalues = s ** 2 / (len(embeddings) - 1)
            except:
                print("Warning: Could not compute eigenvalues, returning PR=1")
                return 1.0
        
        # Filter numerical zeros and negative values
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        
        if len(eigenvalues) == 0:
            return 0.0
        
        # Calculate participation ratio
        sum_eig = np.sum(eigenvalues)
        sum_eig_sq = np.sum(eigenvalues ** 2)
        
        if sum_eig_sq > 0:
            pr = (sum_eig ** 2) / sum_eig_sq
        else:
            pr = 0.0
        
        # Clip to reasonable range
        pr = np.clip(pr, 0.0, min(embeddings.shape[1], len(embeddings)))
        
        return float(pr)
        
    except Exception as e:
        print(f"Warning: Error in PR calculation: {e}, returning PR=1")
        return 1.0


def compute_temporal_consistency_loss(z_seq, temperature=0.1):
    """
    Compute temporal consistency loss for sequential latents.
    """
    batch_size, seq_len, latent_dim = z_seq.shape
    
    if seq_len < 2:
        return torch.tensor(0.0, device=z_seq.device)
    
    # Compute similarities between adjacent time steps
    z_t = z_seq[:, :-1, :]  # [B, T-1, D]
    z_next = z_seq[:, 1:, :]  # [B, T-1, D]
    
    # Normalize
    z_t_norm = F.normalize(z_t.reshape(-1, latent_dim), dim=1)
    z_next_norm = F.normalize(z_next.reshape(-1, latent_dim), dim=1)
    
    # Positive pairs: (z_t, z_{t+1})
    pos_sim = (z_t_norm * z_next_norm).sum(dim=1) / temperature
    
    # Negative pairs: all other combinations in the batch
    sim_matrix = torch.matmul(z_t_norm, z_next_norm.T) / temperature
    
    # Mask out positive pairs
    mask = torch.eye(len(z_t_norm), device=z_seq.device).bool()
    
    # Contrastive loss
    loss = -pos_sim + torch.logsumexp(sim_matrix, dim=1)
    
    return loss.mean()


def compute_predictive_diversity_loss(z_current, z_future, num_negative=10):
    """
    Predictive diversity loss: ensures current latents can distinguish between
    true future latents and randomly sampled ones.
    """
    batch_size = z_current.shape[0]
    device = z_current.device
    
    # Create negative samples by shuffling future latents
    neg_indices = torch.randint(0, batch_size, (batch_size, num_negative), device=device)
    z_negative = z_future[neg_indices]  # [batch_size, num_negative, latent_dim]
    
    # Compute similarities
    pos_sim = F.cosine_similarity(z_current, z_future, dim=1)  # [batch_size]
    neg_sim = F.cosine_similarity(
        z_current.unsqueeze(1), z_negative, dim=2
    )  # [batch_size, num_negative]
    
    # Margin loss: positive pairs should be more similar than negative pairs
    margin = 0.2
    loss = torch.relu(margin + neg_sim - pos_sim.unsqueeze(1)).mean()
    
    return loss


def compute_total_correlation(z, z_dim):
    """
    Compute total correlation (TC) for disentanglement.
    Note: For very large z_dim, consider using torch.cov instead of corrcoef to avoid OOM.
    """
    # Standardize latents
    z_normalized = (z - z.mean(dim=0)) / (z.std(dim=0) + 1e-8)
    
    # Compute correlation matrix
    # Note: If z_dim > 100, this can cause OOM. Use torch.cov and extract diagonals instead.
    correlation_matrix = torch.abs(torch.corrcoef(z_normalized.T))
    
    # TC is related to the sum of off-diagonal elements
    mask = ~torch.eye(z_dim, dtype=torch.bool, device=z.device)
    tc = correlation_matrix[mask].mean()
    
    return tc


def centroid_loss(z, y, num_classes, momentum, running_means, temperature=1.0):
    """
    Pull each latent towards its class running-mean with temperature scaling.
    Handles unknown labels (y=5) by ignoring them.
    """
    loss = 0
    
    with torch.no_grad():
        for c in range(num_classes):  # 0-4 for risk levels
            mask = (y == c)
            if mask.sum() > 0:  # Check if class exists in batch
                batch_mean = z[mask].mean(0)
                if not torch.isnan(batch_mean).any():
                    running_means[c] = (1 - momentum) * running_means[c] + momentum * batch_mean
    
    # Compute loss only for known labels (0-4)
    valid_mask = (y >= 0) & (y < num_classes)
    if valid_mask.any():
        z_valid = z[valid_mask]
        y_valid = y[valid_mask]
        centers = running_means[y_valid]
        loss = ((z_valid - centers).pow(2).sum(1) / temperature).mean()
    
    return loss


# ============================================================================
# Risk-Specific Prior Module
# ============================================================================

class RiskPrior(nn.Module):
    """
    Learnable N(μ_y, Σ_y) prior for each risk label.
    Includes special handling for unknown labels.
    """
    def __init__(self, num_classes, latent_dim, init_spread=2.0):
        super().__init__()
        self.num_risk_levels = num_classes  # 5 risk levels
        self.total_classes = num_classes + 1  # 5 risk levels + 1 unknown
        
        # Initialize means with separation to encourage distinct clusters
        # Arrange in a circle for risk levels, center for unknown
        angles = torch.linspace(0, 2 * np.pi, self.num_risk_levels + 1)[:-1]  # 5 risk levels
        init_means = torch.zeros(self.total_classes, latent_dim)
        
        # First two dimensions: arrange risk levels in circle
        if latent_dim >= 2:
            init_means[:self.num_risk_levels, 0] = torch.cos(angles) * init_spread
            init_means[:self.num_risk_levels, 1] = torch.sin(angles) * init_spread
        
        # Unknown label (index 5) stays at origin
        init_means[self.num_risk_levels, :] = 0
        
        # Add small random noise to other dimensions
        if latent_dim > 2:
            init_means[:, 2:] = torch.randn(self.total_classes, latent_dim - 2) * 0.1
        
        self.mu = nn.Parameter(init_means)
        
        # Initialize with reasonable variance (slightly larger for unknown)
        init_log_sigma = torch.zeros(self.total_classes, latent_dim)
        init_log_sigma[self.num_risk_levels, :] = 0.5  # Larger variance for unknown
        self.log_sigma = nn.Parameter(init_log_sigma)
        
    def forward(self, y):
        """
        y: (batch,) int tensor - assumes unknown label already mapped to last index
        """
        return self.mu[y], self.log_sigma[y]
    
    def get_all_means(self):
        """Get all prior means for visualization."""
        return self.mu.detach()


# ============================================================================
# Data Loading
# ============================================================================

class GamblingDataset(Dataset):
    """General dataset for loading any gambling data type."""
    
    def __init__(self, X_path, L_path, feature_names_path=None, data_type='sessions',
                 use_enhanced_labels=True, enhanced_labels_dir='enhanced_labels'):
        """
        Args:
            X_path: Path to features (X_train.npy, etc.)
            L_path: Path to labels (L_train.npy, etc.)
            feature_names_path: Path to feature_names.txt
            data_type: Type of data (sessions, bets, payments, transactions)
            use_enhanced_labels: Whether to use enhanced labels if available
            enhanced_labels_dir: Directory containing enhanced labels
        """
        self.data_type = data_type
        self.use_enhanced_labels = use_enhanced_labels
        
        # Create progress bar for data loading
        pbar = tqdm(total=5, desc=f"Loading {data_type} dataset")
        
        # Load features with progress
        pbar.set_description(f"Loading {data_type} features")
        self.X = np.load(X_path, mmap_mode='r')
        pbar.update(1)
        print(f"✓ Loaded {data_type} features: {self.X.shape}")
        
        # Check for enhanced labels
        enhanced_labels_path = None
        confidence_path = None
        split = os.path.basename(X_path).replace('X_', '').replace('.npy', '')
        
        if use_enhanced_labels:
            pbar.set_description("Checking for enhanced labels")
            enhanced_labels_path = os.path.join(enhanced_labels_dir, data_type, f'enhanced_labels_{split}.npy')
            confidence_path = os.path.join(enhanced_labels_dir, data_type, f'label_confidence_{split}.npy')
            
            # Also check without data_type subdirectory
            if not os.path.exists(enhanced_labels_path):
                enhanced_labels_path = os.path.join(enhanced_labels_dir, f'enhanced_labels_{split}.npy')
                confidence_path = os.path.join(enhanced_labels_dir, f'label_confidence_{split}.npy')
            
            if os.path.exists(enhanced_labels_path) and os.path.exists(confidence_path):
                print(f"✓ Found enhanced labels at: {enhanced_labels_path}")
                pbar.set_description("Loading enhanced labels")
                self.enhanced_labels = np.load(enhanced_labels_path)
                self.label_confidence = np.load(confidence_path)
                self.using_enhanced = True
                pbar.update(1)
            else:
                print(f"ℹ Enhanced labels not found at: {enhanced_labels_path}")
                print(f"ℹ Using original labels instead")
                self.using_enhanced = False
                pbar.update(1)
        else:
            self.using_enhanced = False
            pbar.update(1)
        
        # Load original labels as fallback
        pbar.set_description("Loading original labels")
        self.L = np.load(L_path, allow_pickle=True)
        pbar.update(1)
        print(f"✓ Loaded {data_type} labels: {self.L.shape}")
        
        # Load feature names if provided
        pbar.set_description("Loading feature names")
        if feature_names_path and os.path.exists(feature_names_path):
            with open(feature_names_path, 'r') as f:
                self.feature_names = [line.strip() for line in f]
            print(f"✓ Loaded {len(self.feature_names)} feature names")
        else:
            self.feature_names = None
        pbar.update(1)
        
        # Process labels
        pbar.set_description("Processing labels")
        self.labels = self._process_labels()
        pbar.update(1)
        
        pbar.close()
        
    def _process_labels(self):
        """Convert string labels to numeric risk levels."""
        label_map = {
            'low_risk': 0,
            'low_medium_risk': 1,
            'medium_risk': 2,
            'medium_high_risk': 3,
            'high_risk': 4
        }
        
        # Use enhanced labels if available
        if self.using_enhanced:
            labels_numeric = self.enhanced_labels
            
            # Print enhanced label statistics
            print(f"\n{self.data_type.capitalize()} ENHANCED label distribution:")
            print("="*50)
            unique, counts = np.unique(labels_numeric[labels_numeric >= 0], return_counts=True)
            total_labeled = (labels_numeric >= 0).sum()
            total_samples = len(labels_numeric)
            
            print(f"  Total samples: {total_samples:,}")
            print(f"  Labeled samples: {total_labeled:,} ({total_labeled/total_samples*100:.1f}%)")
            
            if total_labeled > 0:
                print(f"  Average confidence: {self.label_confidence[labels_numeric >= 0].mean():.3f}")
                print(f"  Min confidence: {self.label_confidence[labels_numeric >= 0].min():.3f}")
                print(f"  Max confidence: {self.label_confidence[labels_numeric >= 0].max():.3f}")
                
                print("\n  Risk level distribution:")
                for label, count in zip(unique, counts):
                    if 0 <= label < len(label_map):
                        label_name = list(label_map.keys())[label]
                        avg_conf = self.label_confidence[labels_numeric == label].mean()
                        print(f"    {label_name}: {count:,} ({count/total_labeled*100:.1f}% of labeled, avg conf: {avg_conf:.3f})")
            print("="*50)
            
            return labels_numeric
        
        # Otherwise, process original labels
        # Handle different label formats
        if self.L.ndim == 1:
            # Single label per window
            labels_str = self.L
        elif self.L.ndim == 2:
            # Multiple labels per window (e.g., per timestep)
            # Use the last timestep or most common
            labels_str = self.L[:, -1]
        else:
            raise ValueError(f"Unexpected label shape: {self.L.shape}")
        
        # Convert to numeric with progress bar
        print(f"\nConverting {self.data_type} original labels to numeric...")
        labels_numeric = np.zeros(len(labels_str), dtype=np.int64) - 1  # -1 for unknown
        
        for i in tqdm(range(len(labels_str)), desc="Processing labels", leave=False):
            label_str = str(labels_str[i]).lower()
            for key, value in label_map.items():
                if key in label_str:
                    labels_numeric[i] = value
                    break
        
        # Print label distribution
        unique, counts = np.unique(labels_numeric, return_counts=True)
        print(f"\n{self.data_type.capitalize()} ORIGINAL label distribution:")
        print("="*50)
        for label, count in zip(unique, counts):
            if label >= 0:
                label_name = list(label_map.keys())[label]
                print(f"  {label_name}: {count:,} ({count/len(labels_numeric)*100:.1f}%)")
            else:
                print(f"  unknown: {count:,} ({count/len(labels_numeric)*100:.1f}%)")
        print("="*50)
        
        return labels_numeric
    
    def get_feature_groups(self):
        """Get indices for short/mid/long feature groups based on correct hierarchy patterns."""
        print(f"\nAnalyzing feature groups for {self.data_type}...")
        
        if self.feature_names is None:
            # Default split by thirds
            n_features = self.X.shape[2]
            third = n_features // 3
            return {
                'short': list(range(third)),
                'mid': list(range(third, 2*third)),
                'long': list(range(2*third, n_features))
            }
        
        # Use the correct hierarchical feature patterns
        short_patterns = [
            '_norm', 'I_total', 'I_avg', 'I_session_frequency', 'I_delta_session_frequency',
            'E_total', 'E_avg', 'E_session_frequency', 'E_delta_session_frequency',
            '_snps_7d'
        ]
        
        mid_patterns = [
            '_count_exceeds_7d', '_any_exceeds_7d', '_freq_exceeds_7d',
            '_sum_magnitude_7d', '_max_magnitude_7d', '_mean_magnitude_7d',
            '_snps_14d'
        ]
        
        long_patterns = [
            'total_features_exceeded', 'multiple_features_exceeded',
            'exceedance_correlation', 'exceedance_pattern_entropy',
            'tail_dep_'
        ]
        
        groups = {'short': [], 'mid': [], 'long': []}
        
        # Use progress bar for feature assignment
        for i, name in enumerate(tqdm(self.feature_names, desc="Assigning features to groups", leave=False)):
            assigned = False
            feature_lower = name.lower()
            
            # Check short patterns - exact matching for specific patterns
            for pattern in short_patterns:
                if pattern in feature_lower:
                    groups['short'].append(i)
                    assigned = True
                    break
            
            # Check mid patterns if not already assigned
            if not assigned:
                for pattern in mid_patterns:
                    if pattern in feature_lower:
                        groups['mid'].append(i)
                        assigned = True
                        break
            
            # Check long patterns if not already assigned
            if not assigned:
                for pattern in long_patterns:
                    if pattern in feature_lower:
                        groups['long'].append(i)
                        assigned = True
                        break
            
            # Default to mid if unassigned (for features that don't match any pattern)
            if not assigned:
                groups['mid'].append(i)
        
        print(f"\nFeature groups for {self.data_type}:")
        print(f"  Short-term: {len(groups['short'])} features")
        print(f"  Mid-term: {len(groups['mid'])} features")
        print(f"  Long-term: {len(groups['long'])} features")
        
        return groups
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx].copy()).float()
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        # Return confidence if using enhanced labels
        if self.using_enhanced:
            confidence = torch.tensor(self.label_confidence[idx], dtype=torch.float)
            return x, label, confidence
        
        return x, label


# ============================================================================
# Model Architecture - Conditional Hierarchical VAE with Risk Priors
# ============================================================================

class CausalConv1d(nn.Module):
    """1D causal convolution that preserves sequence length"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )
    
    def forward(self, x):
        # Apply convolution
        out = self.conv(x)
        # Remove future values to make it causal
        if self.padding > 0:
            return out[:, :, :-self.padding]
        return out


class TCNBlock(nn.Module):
    """Single TCN block with residual connection"""
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        
        # Main path
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        #self.norm1 = nn.GroupNorm(8, out_channels)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        #self.norm2 = nn.GroupNorm(8, out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # Residual connection
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, 1)
    
    def forward(self, x):
        # Save input for residual
        residual = x
        
        # Main path
        out = self.conv1(x)
        #out = self.norm1(out)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        #out = self.norm2(out)
        out = self.bn2(out)
        
        # Apply residual
        if self.downsample is not None:
            residual = self.downsample(residual)
        
        out = out + residual
        out = self.relu2(out)
        out = self.dropout2(out)
        
        return out


class ConditionalTCNEncoder(nn.Module):
    """Temporal Convolutional Network encoder with label conditioning."""
    
    def __init__(self, input_dim, num_classes, hidden_dim=256, kernel_size=3, 
                 num_blocks=4, dilation_base=2, dropout=0.2, embed_dim=32,
                 use_spectral_norm=True):
        super().__init__()
        
        # Label embedding
        self.label_embedding = nn.Embedding(num_classes + 1, embed_dim)  # +1 for unknown
        
        blocks = []
        in_channels = input_dim + embed_dim  # Input features + label embedding
        
        for i in range(num_blocks):
            dilation = dilation_base ** i
            block = TCNBlock(
                in_channels, hidden_dim, 
                kernel_size=kernel_size, 
                dilation=dilation, 
                dropout=dropout
            )
            # Apply spectral normalization if requested
            if use_spectral_norm:
                for module in block.modules():
                    if isinstance(module, (nn.Conv1d, nn.Linear)):
                        spectral_norm(module)
            blocks.append(block)
            in_channels = hidden_dim
        
        self.network = nn.Sequential(*blocks)
    
    def forward(self, x, labels, predict_mode=False):
        # x: [batch, seq_len, features]
        # labels: [batch]
        # predict_mode: if True, don't use real labels (for fair evaluation)
        
        batch_size, seq_len, _ = x.shape
        
        # CRITICAL FIX: During prediction, use dummy "unknown" labels
        if predict_mode:
            # All samples treated as "unknown" during prediction
            labels_to_use = torch.full_like(labels, 5)  # 5 = unknown
        else:
            # During training, use real labels for conditioning
            labels_to_use = labels.clone()
            labels_to_use[labels < 0] = 5  # Map unknown to index 5
        
        label_emb = self.label_embedding(labels_to_use)  # [batch, embed_dim]
        # Normalize label embeddings to prevent explosion
        label_emb = F.normalize(label_emb, p=2, dim=1) * 0.1
        
        label_emb_expanded = label_emb.unsqueeze(1).expand(batch_size, seq_len, -1)  # [batch, seq_len, embed_dim]
        
        # Concatenate input with label embeddings
        x_cond = torch.cat([x, label_emb_expanded], dim=2)  # [batch, seq_len, features + embed_dim]
        
        # Check for NaN/Inf
        if torch.isnan(x_cond).any() or torch.isinf(x_cond).any():
            print("Warning: NaN/Inf in conditional input!")
            x_cond = torch.nan_to_num(x_cond, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Process through TCN
        x_cond = x_cond.transpose(1, 2)  # [batch, features + embed_dim, seq_len]
        h = self.network(x_cond)
        h = h.transpose(1, 2)  # [batch, seq_len, hidden]
        
        return h


class ConditionalHierarchicalVAE(nn.Module):
    """
    Conditional Hierarchical VAE for gambling risk assessment with risk-specific priors.
    Each risk level has its own learned Gaussian prior N(μ_y, Σ_y).
    """
    
    def __init__(self, input_dim, seq_len=30, 
                 latent_dims=[32, 24, 16],  # short, mid, long
                 hidden_dim=256,
                 num_classes=5,
                 feature_groups=None,
                 data_type='sessions',
                 label_embed_dim=32,
                 use_spectral_norm=True,
                 diversity_weight=0.2,
                 min_encoding_capacity=20.0,
                 use_mi_estimation=True,
                 mi_weight=0.1,
                 temporal_consistency_weight=0.1,
                 tc_weight=0.05,
                 centroid_weight=0.1,
                 prior_init_spread=2.0,
                 dropout=0.2,
                 use_classification_head=True,
                 classification_weight=0.3,
                 **kwargs):
        super().__init__()
        
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.latent_dims = latent_dims
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.data_type = data_type
        self.label_embed_dim = label_embed_dim
        self.feature_groups = feature_groups or {'short': [], 'mid': [], 'long': []}
        self.use_spectral_norm = use_spectral_norm
        self.diversity_weight = diversity_weight
        self.min_encoding_capacity = min_encoding_capacity
        self.use_mi_estimation = use_mi_estimation
        self.mi_weight = mi_weight
        self.temporal_consistency_weight = temporal_consistency_weight
        self.tc_weight = tc_weight
        self.centroid_weight = centroid_weight
        self.dropout = dropout
        self.use_classification_head = use_classification_head
        self.classification_weight = classification_weight
        
        # Temporal encoders for each scale (conditioned on labels)
        self.short_encoder = ConditionalTCNEncoder(
            input_dim, num_classes, hidden_dim, kernel_size=3, 
            num_blocks=3, dropout=self.dropout, embed_dim=label_embed_dim,
            use_spectral_norm=use_spectral_norm
        )
        
        self.mid_encoder = ConditionalTCNEncoder(
            input_dim, num_classes, hidden_dim, kernel_size=5, 
            num_blocks=4, dropout=self.dropout, embed_dim=label_embed_dim,
            use_spectral_norm=use_spectral_norm
        )
        
        self.long_encoder = ConditionalTCNEncoder(
            input_dim, num_classes, hidden_dim, kernel_size=7, 
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
        # Short-term (conditioned on label)
        self.short_mu = nn.Linear(hidden_dim + label_embed_dim, latent_dims[0])
        self.short_logvar = nn.Linear(hidden_dim + label_embed_dim, latent_dims[0])
        
        # Mid-term (conditioned on short + label)
        self.mid_mu = nn.Linear(hidden_dim + latent_dims[0] + label_embed_dim, latent_dims[1])
        self.mid_logvar = nn.Linear(hidden_dim + latent_dims[0] + label_embed_dim, latent_dims[1])
        
        # Long-term (conditioned on short + mid + label)
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
        
        # Add classification head for risk prediction
        if use_classification_head:
            # Simple linear classifier on concatenated latents
            self.risk_classifier = nn.Sequential(
                nn.Linear(sum(latent_dims), hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, num_classes)  # 5 risk levels
            )
        
        # Conditional decoder
        self.decoder_initial = nn.Linear(sum(latent_dims) + label_embed_dim, hidden_dim)
        self.decoder_blocks = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.decoder_final = nn.Linear(hidden_dim, input_dim * seq_len)
        
        # Contrastive projection head (for better separation)
        self.projection_head = nn.Sequential(
            nn.Linear(sum(latent_dims), 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # MI estimation critic (for mutual information)
        if use_mi_estimation:
            # First project high-dim input to manageable size
            self.mi_input_proj = nn.Linear(input_dim * seq_len, hidden_dim)
            self.mi_critic = nn.Sequential(
                nn.Linear(sum(latent_dims) + hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
        else:
            self.mi_input_proj = None
            self.mi_critic = None
        
        # Temporal prediction head (for temporal consistency)
        self.temporal_predictor = nn.Sequential(
            nn.Linear(sum(latent_dims), hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, sum(latent_dims))
        )
        
        # Initialize weights for stability
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for better stability."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Much more conservative initialization
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Embedding):
                # MUCH smaller initialization for embeddings
                nn.init.normal_(m.weight, mean=0.0, std=0.001)
    
    def encode_level(self, x, labels, level='short'):
        """Encode at a specific temporal level with label conditioning."""
        if level == 'short':
            h = self.short_encoder(x, labels)
            h_t = h.transpose(0, 1)
            h_att, _ = self.short_attention(h_t, h_t, h_t)
            h_pooled = h_att.mean(dim=0)
        elif level == 'mid':
            h = self.mid_encoder(x, labels)
            h_t = h.transpose(0, 1)
            h_att, _ = self.mid_attention(h_t, h_t, h_t)
            h_pooled = h_att.mean(dim=0)
        else:  # long
            h = self.long_encoder(x, labels)
            h_t = h.transpose(0, 1)
            h_att, _ = self.long_attention(h_t, h_t, h_t)
            h_pooled = h_att.mean(dim=0)
        
        return h_pooled
    
    def encode_sequence(self, x, labels):
        """Encode the full sequence to get temporal latents."""
        # Get sequential representations from each encoder
        h_short = self.short_encoder(x, labels)  # [B, T, H]
        h_mid = self.mid_encoder(x, labels)
        h_long = self.long_encoder(x, labels)
        
        return h_short, h_mid, h_long
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def compute_diversity_loss(self, z, loss_type='temporal_aware'):
        """
        Compute diversity loss to prevent collapse.
        For temporal data, we use a temporal-aware version.
        """
        if loss_type == 'cosine':
            # Standard cosine similarity penalty
            z_norm = F.normalize(z, dim=1)
            sim_matrix = torch.matmul(z_norm, z_norm.T)
            mask = torch.eye(sim_matrix.size(0), device=sim_matrix.device).bool()
            sim_matrix = sim_matrix.masked_fill(mask, 0)
            diversity_loss = sim_matrix.abs().mean()
            
        elif loss_type == 'temporal_aware':
            # For temporal data, we want diversity across time AND samples
            batch_size = z.shape[0]
            
            # Diversity across samples (standard)
            z_norm = F.normalize(z, dim=1)
            sim_matrix = torch.matmul(z_norm, z_norm.T)
            mask = torch.eye(batch_size, device=z.device).bool()
            sample_diversity = sim_matrix.masked_fill(mask, 0).abs().mean()
            
            # Temporal diversity (penalize if consecutive samples are too similar)
            if batch_size > 1:
                z_pairs = z.unfold(0, 2, 1)  # Get consecutive pairs
                if z_pairs.shape[0] > 0:
                    z1 = z_pairs[:, 0, :]
                    z2 = z_pairs[:, 1, :]
                    temporal_sim = F.cosine_similarity(z1, z2, dim=1).abs().mean()
                    diversity_loss = 0.7 * sample_diversity + 0.3 * temporal_sim
                else:
                    diversity_loss = sample_diversity
            else:
                diversity_loss = sample_diversity
                
        else:
            # Variance-based diversity loss
            z_var = torch.var(z, dim=0)
            diversity_loss = -torch.log(z_var + 1e-8).mean()
        
        return diversity_loss
    
    def forward(self, x, labels, return_sequences=False):
        batch_size = x.shape[0]
        
        # Handle unlabeled data
        labels_clamped = labels.clone()
        labels_clamped[labels < 0] = 5  # Map unknown to index 5
        
        # Get label embeddings for latent projections
        label_emb = self.label_embedding_latent(labels_clamped)
        # Normalize label embeddings to prevent explosion
        label_emb = F.normalize(label_emb, p=2, dim=1) * 0.1
        
        # Get sequential encodings for temporal consistency
        h_short_seq, h_mid_seq, h_long_seq = self.encode_sequence(x, labels)
        
        # Hierarchical encoding with label conditioning
        # 1. Short-term
        h_short = self.encode_level(x, labels, 'short')
        h_short_cond = torch.cat([h_short, label_emb], dim=1)
        short_mu = self.short_mu(h_short_cond)
        short_logvar = torch.clamp(self.short_logvar(h_short_cond), -20, 2)
        short_z = self.reparameterize(short_mu, short_logvar)
        
        # Get risk-specific prior for short
        prior_mu_short, prior_logvar_short = self.risk_prior_short(labels_clamped)
        
        # 2. Mid-term (conditioned on short + label)
        h_mid = self.encode_level(x, labels, 'mid')
        h_mid_cond = torch.cat([h_mid, short_z, label_emb], dim=1)
        mid_mu = self.mid_mu(h_mid_cond)
        mid_logvar = torch.clamp(self.mid_logvar(h_mid_cond), -20, 2)
        mid_z = self.reparameterize(mid_mu, mid_logvar)
        
        # Get risk-specific prior for mid
        prior_mu_mid, prior_logvar_mid = self.risk_prior_mid(labels_clamped)
        
        # 3. Long-term (conditioned on short + mid + label)
        h_long = self.encode_level(x, labels, 'long')
        h_long_cond = torch.cat([h_long, short_z, mid_z, label_emb], dim=1)
        long_mu = self.long_mu(h_long_cond)
        long_logvar = torch.clamp(self.long_logvar(h_long_cond), -20, 2)
        long_z = self.reparameterize(long_mu, long_logvar)
        
        # Get risk-specific prior for long
        prior_mu_long, prior_logvar_long = self.risk_prior_long(labels_clamped)
        
        # EMERGENCY ANTI-COLLAPSE: Add noise to prevent identical latents
        if self.training and hasattr(self, 'dropout') and self.dropout == 0.0:
            # Only add noise if dropout is disabled (our anti-collapse signal)
            noise_scale = 0.1  # Significant noise
            short_z = short_z + torch.randn_like(short_z) * noise_scale
            mid_z = mid_z + torch.randn_like(mid_z) * noise_scale
            long_z = long_z + torch.randn_like(long_z) * noise_scale
        
        # Combine all latents
        all_z = torch.cat([short_z, mid_z, long_z], dim=1)
        
        # Add classification output
        # Add classification output
        risk_logits = None
        if self.use_classification_head:
            # During training, we might want to detach
            if self.training:
                risk_logits = self.risk_classifier(all_z)
            else:
                # During evaluation, don't detach
                with torch.no_grad():
                    risk_logits = self.risk_classifier(all_z)
        
        # Conditional decode
        h_decode = torch.cat([all_z, label_emb], dim=1)
        h_decode = self.decoder_initial(h_decode)
        h_decode = self.decoder_blocks(h_decode)
        x_recon = self.decoder_final(h_decode)
        x_recon = x_recon.view(batch_size, self.seq_len, self.input_dim)
        x_recon = torch.sigmoid(x_recon)
        
        # Contrastive features
        z_proj = F.normalize(self.projection_head(all_z), dim=1)
        
        # Compute diversity loss (temporal-aware)
        diversity_loss = self.compute_diversity_loss(all_z, loss_type='temporal_aware')
        
        # Compute MI if requested
        mi_loss = torch.tensor(0.0, device=all_z.device)
        if self.use_mi_estimation and self.training:
            # Flatten input for MI estimation
            x_flat = x.reshape(batch_size, -1)
            
            # Project input to manageable dimension
            x_proj = self.mi_input_proj(x_flat)
            
            # Use the critic network to compute MI
            # Positive pairs: (z, x) from same sample
            z_x_pos = torch.cat([all_z, x_proj], dim=1)
            pos_scores = self.mi_critic(z_x_pos)
            
            # Negative pairs: shuffle x
            x_proj_shuffle = x_proj[torch.randperm(batch_size)]
            z_x_neg = torch.cat([all_z, x_proj_shuffle], dim=1)
            neg_scores = self.mi_critic(z_x_neg)
            
            # InfoNCE-style loss
            mi_loss = torch.mean(pos_scores) - torch.log(torch.exp(neg_scores).mean() + 1e-8)
        
        # Compute total correlation for disentanglement
        tc = compute_total_correlation(all_z, sum(self.latent_dims))
        
        outputs = {
            'x_recon': x_recon,
            'latents': {
                'short': (short_mu, short_logvar, short_z),
                'mid': (mid_mu, mid_logvar, mid_z),
                'long': (long_mu, long_logvar, long_z)
            },
            'z_all': all_z,
            'z_proj': z_proj,
            'prior_short': (prior_mu_short, prior_logvar_short),
            'prior_mid': (prior_mu_mid, prior_logvar_mid),
            'prior_long': (prior_mu_long, prior_logvar_long),
            'diversity_loss': diversity_loss * self.diversity_weight,
            'mi_loss': mi_loss * self.mi_weight,
            'tc_loss': tc * self.tc_weight,
            'sequences': (h_short_seq, h_mid_seq, h_long_seq) if return_sequences else None,
            'risk_logits': risk_logits
        }
        
        return outputs
    
    def get_embeddings(self, x, labels):
        """Extract embeddings for downstream tasks."""
        with torch.no_grad():
            outputs = self.forward(x, labels)
            # Use all latent codes as embeddings
            embeddings = outputs['z_all']
        return embeddings


# ============================================================================
# Loss Functions for Conditional VAE with Risk Priors
# ============================================================================

def compute_cvae_losses(outputs, x, labels, config, epoch=0, confidence=None, kl_weight=1.0, running_means=None, centroid_weight=None):
    """Compute all losses for conditional hierarchical VAE with risk-specific priors."""
    losses = {}
    device = x.device
    
    # Handle unlabeled data
    labels_clamped = labels.clone()
    labels_clamped[labels < 0] = 5  # Map unknown to index 5
    
    # 1. Reconstruction loss with clamping
    recon_loss = F.mse_loss(outputs['x_recon'], x, reduction='none')
    # Clamp per-sample losses before averaging
    recon_loss = torch.clamp(recon_loss.mean(dim=[1,2]), max=10.0)
    losses['recon'] = recon_loss.mean()
    
    # 2. Hierarchical KL losses with risk-specific priors
    kl_weights = config.get('kl_weights', [0.001, 0.01, 0.05])
    free_bits = config.get('free_bits', [5.0, 3.0, 2.0])
    min_capacity = config.get('min_encoding_capacity', 20.0)
    
    # Prior lookup
    prior_lookup = {
        'short': outputs['prior_short'],
        'mid': outputs['prior_mid'],
        'long': outputs['prior_long']
    }
    
    total_kl = 0
    for i, level in enumerate(['short', 'mid', 'long']):
        mu, logvar, z = outputs['latents'][level]
        prior_mu, prior_logvar = prior_lookup[level]  # Risk-specific prior
        
        # KL divergence between two diagonal Gaussians
        # KL(q||p) = 0.5 * sum(log(σ_p/σ_q) + (σ_q^2 + (μ_q - μ_p)^2) / σ_p^2 - 1)
        prior_var = torch.exp(torch.clamp(prior_logvar, -20, 2))
        var = torch.exp(torch.clamp(logvar, -20, 2))
        
        kl = 0.5 * torch.sum(
            prior_logvar - logvar  # log σ_p - log σ_q
            + (var + (mu - prior_mu).pow(2)) / (prior_var + 1e-8)
            - 1,
            dim=1
        ).mean()
        
        # Ensure KL is non-negative
        kl = torch.abs(kl)
        
        # Apply free bits thresholding
        kl = torch.max(kl, torch.tensor(free_bits[i]).to(device))
        
        # Apply KL weight with annealing
        annealed_weight = kl_weights[i] * kl_weight
        
        losses[f'kl_{level}'] = annealed_weight * kl
        total_kl += kl.item()
    
    # Apply minimum encoding capacity constraint
    if total_kl < min_capacity:
        capacity_loss = torch.tensor(min_capacity - total_kl, device=device, dtype=torch.float32)
        losses['capacity'] = 0.1 * capacity_loss
    
    # 3. Diversity loss to prevent collapse
    if config.get('diversity_weight', 0) > 0:
        diversity_loss = outputs.get('diversity_loss', torch.tensor(0.0, device=device))
        losses['diversity'] = diversity_loss  # Already weighted in model
    
    # 4. Mutual Information loss (maximize MI between latents and inputs)
    if config.get('use_mi_estimation', True):
        mi_loss = outputs.get('mi_loss', torch.tensor(0.0, device=device))
        # We want to maximize MI, so we minimize the negative
        losses['mi'] = -mi_loss  # Already weighted in model
    
    # 5. Total Correlation loss (for disentanglement)
    if config.get('tc_weight', 0) > 0:
        tc_loss = outputs.get('tc_loss', torch.tensor(0.0, device=device))
        losses['tc'] = tc_loss  # Already weighted in model
    
    # 6. Contrastive loss for better separation (using z directly, not projection)
    if labels is not None and config.get('use_contrastive', True):
        z_all = outputs['z_all']
        
        # Get class weights if using class balancing
        class_weights = None
        if config.get('use_class_balancing', False):
            class_weights = torch.tensor(
                config.get('class_weights', [1.0, 1.0, 1.0, 1.0, 1.0]), 
                device=device
            )
        
        cont_loss = contrastive_loss(F.normalize(z_all, dim=1), labels_clamped, 
                                    temperature=0.07, confidence=confidence, 
                                    class_weights=class_weights)
        losses['contrastive'] = config.get('contrastive_weight', 0.5) * cont_loss
    
    # 7. Centroid loss (if running means provided)
    if running_means is not None and centroid_weight is not None and centroid_weight > 0:
        cent_loss = torch.tensor(0.0, device=device)
        for level in ['short', 'mid', 'long']:
            _, _, z = outputs['latents'][level]
            # Only use known labels (0-4) for centroid loss
            cent_loss += centroid_loss(z, labels_clamped, num_classes=5, 
                                     momentum=0.05, 
                                     running_means=running_means[level],
                                     temperature=1.0)
        losses['centroid'] = centroid_weight * cent_loss / 3.0
    
    # 8. Classification loss (only for labeled samples)
    if outputs.get('risk_logits') is not None and config.get('use_classification_head', True):
        risk_logits = outputs['risk_logits']
        
        # Only compute loss for labeled samples (0-4, not 5/unknown)
        labeled_mask = (labels >= 0) & (labels < 5)
        
        if labeled_mask.any():
            labeled_logits = risk_logits[labeled_mask]
            labeled_targets = labels[labeled_mask]
            
            # Apply confidence weighting if available
            if confidence is not None and labeled_mask.any():
                labeled_confidence = confidence[labeled_mask]
                # Filter low confidence samples
                high_conf_mask = labeled_confidence > 0.6
                if high_conf_mask.any():
                    labeled_logits = labeled_logits[high_conf_mask]
                    labeled_targets = labeled_targets[high_conf_mask]
                    labeled_confidence = labeled_confidence[high_conf_mask]
                    
                    # Weighted cross entropy
                    ce_loss = F.cross_entropy(labeled_logits, labeled_targets, reduction='none')
                    ce_loss = (ce_loss * labeled_confidence).mean()
                else:
                    ce_loss = torch.tensor(0.0, device=x.device)
            else:
                ce_loss = F.cross_entropy(labeled_logits, labeled_targets)
            
            losses['classification'] = config.get('classification_weight', 0.3) * ce_loss
    
    # 9. Temporal consistency loss (if sequences are available)
    if config.get('temporal_consistency_weight', 0) > 0 and outputs.get('sequences') is not None:
        h_short_seq, h_mid_seq, h_long_seq = outputs['sequences']
        
        # Apply temporal consistency to each hierarchy level
        temp_loss = torch.tensor(0.0, device=device)
        for h_seq in [h_short_seq, h_mid_seq, h_long_seq]:
            if h_seq is not None:
                # Create pseudo-latents from sequential hidden states
                batch_size, seq_len, hidden_dim = h_seq.shape
                # Sample a few time steps for efficiency
                if seq_len > 10:
                    indices = torch.randint(0, seq_len, (10,), device=device)
                    h_sampled = h_seq[:, indices, :]
                else:
                    h_sampled = h_seq
                
                temp_loss += compute_temporal_consistency_loss(h_sampled, temperature=0.1)
        
        losses['temporal'] = config['temporal_consistency_weight'] * temp_loss / 3.0
    
    # 10. Predictive diversity loss (optional, for stronger temporal coherence)
    if config.get('use_predictive_diversity', False) and outputs['z_all'].shape[0] > 1:
        # Use current and shifted latents as proxy for temporal prediction
        z_current = outputs['z_all'][:-1]
        z_future = outputs['z_all'][1:]
        if z_current.shape[0] > 0:
            pred_div_loss = compute_predictive_diversity_loss(z_current, z_future, num_negative=5)
            losses['pred_diversity'] = config.get('pred_diversity_weight', 0.1) * pred_div_loss
    
    # 11. Beta-VAE style capacity control (optional)
    if config.get('use_beta_vae', False):
        # Add extra weight to KL terms for information bottleneck
        beta = config.get('beta', 4.0)
        for level in ['short', 'mid', 'long']:
            losses[f'kl_{level}'] *= beta
    
    # 12. Total loss
    losses['total'] = sum(losses.values())
    
    return losses


def contrastive_loss(features, labels, temperature=0.07, confidence=None, class_weights=None):
    """Supervised contrastive loss with optional confidence and class weighting."""
    device = features.device
    
    # Filter samples with known labels (0-4, not 5 which is unknown)
    mask = (labels >= 0) & (labels < 5)
    if mask.sum() < 2:
        return torch.tensor(0.0, device=device)
    
    features = features[mask]
    labels = labels[mask]
    if confidence is not None:
        confidence = confidence[mask]
    
    # Compute similarities
    sim_matrix = torch.matmul(features, features.T) / temperature
    
    # Masks for positive and negative pairs
    labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
    pos_mask = labels_eq.float()
    pos_mask.fill_diagonal_(0)
    
    # Apply confidence to positive mask if available
    if confidence is not None:
        # Weight positive pairs by geometric mean of confidences
        conf_matrix = torch.sqrt(confidence.unsqueeze(0) * confidence.unsqueeze(1))
        pos_mask = pos_mask * conf_matrix
    
    # Apply class weights if provided
    if class_weights is not None:
        # Get weights for each sample based on their label
        sample_weights = class_weights[labels]
        # Create weight matrix for pairs
        weight_matrix = torch.sqrt(sample_weights.unsqueeze(0) * sample_weights.unsqueeze(1))
        pos_mask = pos_mask * weight_matrix
    
    # Compute loss
    exp_sim = torch.exp(sim_matrix)
    pos_sim = (exp_sim * pos_mask).sum(dim=1)
    all_sim = exp_sim.sum(dim=1) - exp_sim.diag()
    
    loss = -torch.log(pos_sim / (all_sim + 1e-8) + 1e-8)
    
    # Weight by confidence and/or class weights if available
    if confidence is not None:
        loss = loss * confidence
    if class_weights is not None:
        loss = loss * sample_weights
    
    # Only average over samples with positive pairs
    valid = pos_mask.sum(dim=1) > 0
    if valid.any():
        if confidence is not None or class_weights is not None:
            # Weighted average
            weights = torch.ones_like(loss[valid])
            if confidence is not None:
                weights = weights * confidence[valid]
            if class_weights is not None:
                weights = weights * sample_weights[valid]
            loss = (loss[valid] * weights).sum() / weights.sum()
        else:
            loss = loss[valid].mean()
    else:
        loss = torch.tensor(0.0, device=device)
    
    return loss


# ============================================================================
# Training Function with Risk Priors
# ============================================================================

def check_training_health(model, loss_dict, batch_idx):
    """Check if training is healthy"""
    # Check losses
    for name, loss in loss_dict.items():
        if torch.is_tensor(loss):
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n!!! NaN/Inf in {name} loss at batch {batch_idx}")
                return False
            if name != 'mi' and loss.item() < -1.0:  # Negative losses are bad
                print(f"\n!!! Negative {name} loss: {loss.item()} at batch {batch_idx}")
                return False
    
    # Check model parameters
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"\n!!! NaN/Inf in parameter {name}")
            return False
    
    return True


def clip_gradients_safe(model, clip_value=0.5):
    """Clip gradients with special handling for embeddings and priors"""
    for name, param in model.named_parameters():
        if param.grad is not None:
            # Check for NaN/Inf
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                print(f"Warning: NaN/Inf gradient in {name}, zeroing...")
                param.grad.data.zero_()
                continue
            
            # SUPER AGGRESSIVE CLIPPING when recovering from collapse
            if 'embedding' in name or 'risk_prior' in name:
                param.grad.data.clamp_(-0.001, 0.001)  # 10x smaller
            elif 'encoder' in name:
                param.grad.data.clamp_(-0.1, 0.1)  # Encoder gradients limited
            else:
                param.grad.data.clamp_(-clip_value, clip_value)


def train_model(model, train_loader, val_loader, config, device, output_dir, resume_checkpoint=None):
    """
    Train the conditional hierarchical VAE model with risk-specific priors.
    
    Args:
        model: The model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Configuration dictionary
        device: Device to train on
        output_dir: Output directory for checkpoints
        resume_checkpoint: Path to checkpoint to resume from (optional)
    """
    
    # Check if using enhanced labels
    using_enhanced = hasattr(train_loader.dataset, 'using_enhanced') and train_loader.dataset.using_enhanced
    if using_enhanced:
        print("\n" + "="*60)
        print("✓ USING ENHANCED LABELS WITH CONFIDENCE WEIGHTING")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("ℹ Using original labels (enhanced labels not found)")
        print("="*60)
    
    # Anti-collapse settings with risk priors
    print("\nAnti-collapse features enabled:")
    print(f"  - Risk-specific priors: ✓ (5 risk levels + unknown)")
    print(f"  - Spectral normalization: {config.get('use_spectral_norm', True)}")
    print(f"  - Diversity loss weight: {config.get('diversity_weight', 0.2)}")
    print(f"  - Centroid loss weight: {config.get('centroid_weight', 0.1)}")
    print(f"  - {config.get('annealing_type', 'cyclical').capitalize()} KL annealing: {config.get('annealing_cycles', 4)} cycles, {config.get('warmup_epochs', 5)} warmup epochs")
    print(f"  - Minimum PR threshold: {config.get('min_participation_ratio', 10.0)}")
    print(f"  - Minimum encoding capacity: {config.get('min_encoding_capacity', 20.0)}")
    print(f"  - Mutual Information estimation: {config.get('use_mi_estimation', True)} (weight: {config.get('mi_weight', 0.1)})")
    print(f"  - Temporal consistency: {config.get('temporal_consistency_weight', 0.1)}")
    print(f"  - Total Correlation penalty: {config.get('tc_weight', 0.05)}")
    print(f"  - Class balancing: {config.get('use_class_balancing', False)}")
    print(f"  - Classification head: {config.get('use_classification_head', True)} (weight: {config.get('classification_weight', 0.3)})")
    
    # Initialize training state
    start_epoch = 0
    best_val_loss = float('inf')
    best_participation_ratio = 0.0
    best_model_state = None
    patience_counter = 0
    pr_patience_counter = 0
    
    # Create running means for centroid loss (only for 5 risk levels, not unknown)
    running_means = {
        'short': torch.zeros(5, config['latent_dims'][0], device=device),
        'mid': torch.zeros(5, config['latent_dims'][1], device=device),
        'long': torch.zeros(5, config['latent_dims'][2], device=device)
    }
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_pr': [], 'val_pr': [],
        'train_kl': [], 'train_diversity': [],
        'train_cosine_sim': [], 'train_mi': [],
        'train_tc': [], 'train_temporal': [],
        'train_centroid': [], 'train_classification': [],
        'avg_confidence': [],
        'prior_separation': []
    }
    
    # Optimizer with different learning rates
    param_groups = {
        'embed': [],
        'priors': [],
        'short_encoder': [],
        'mid_encoder': [],
        'long_encoder': [],
        'other': []
    }
    
    # Group parameters by component
    for name, param in model.named_parameters():
        if 'embedding' in name:
            param_groups['embed'].append(param)
        elif 'risk_prior' in name:
            param_groups['priors'].append(param)
        elif 'short_encoder' in name and 'embedding' not in name:
            param_groups['short_encoder'].append(param)
        elif 'mid_encoder' in name and 'embedding' not in name:
            param_groups['mid_encoder'].append(param)
        elif 'long_encoder' in name and 'embedding' not in name:
            param_groups['long_encoder'].append(param)
        else:
            param_groups['other'].append(param)
    
    # Create optimizer with non-overlapping groups
    base_lr = config['lr']
    optimizer_groups = []

    if param_groups['embed']:
        optimizer_groups.append(
            {'params': param_groups['embed'],
            'lr': base_lr * 0.1,
            'tag': 'embed'}          #  ← NEW TAG
        )
    if param_groups['priors']:
        optimizer_groups.append(
            {'params': param_groups['priors'],
            'lr': base_lr * 0.5,
            'tag': 'priors'}         #  ← tag is optional but nice to have
        )
    if param_groups['short_encoder']:
        optimizer_groups.append(
            {'params': param_groups['short_encoder'],
            'lr': base_lr * 0.5,
            'tag': 'short_enc'}
        )
    if param_groups['mid_encoder']:
        optimizer_groups.append(
            {'params': param_groups['mid_encoder'],
            'lr': base_lr * 0.75,
            'tag': 'mid_enc'}
        )
    if param_groups['long_encoder']:
        optimizer_groups.append(
            {'params': param_groups['long_encoder'],
            'lr': base_lr,
            'tag': 'long_enc'}
        )
    if param_groups['other']:
        optimizer_groups.append(
            {'params': param_groups['other'],
            'lr': base_lr,
            'tag': 'other'}
        )

    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )
    
    # Resume from checkpoint if provided
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"\n" + "="*60)
        print(f"RESUMING FROM CHECKPOINT: {resume_checkpoint}")
        print("="*60)
        
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        
        # Restore model state
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("✓ Restored model state")
        
        # Restore optimizer state
        # Inside train_model function, after optimizer.load_state_dict():
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("✓ Restored optimizer state")
            
            # FIX: Force update learning rate if in anti-collapse mode
            if config.get('dropout', 0.2) == 0.0:  # Anti-collapse indicator
                print(f"💥 Anti-collapse detected! Forcing LR update...")
                for param_group in optimizer.param_groups:
                    # Update based on the parameter group's original lr ratio
                    if 'embed' in str(param_group.get('params', [])):
                        param_group['lr'] = config['lr'] * 0.1
                    elif 'priors' in str(param_group.get('params', [])):
                        param_group['lr'] = config['lr'] * 0.5
                    elif 'short_encoder' in str(param_group.get('params', [])):
                        param_group['lr'] = config['lr'] * 0.5
                    elif 'mid_encoder' in str(param_group.get('params', [])):
                        param_group['lr'] = config['lr'] * 0.75
                    else:
                        param_group['lr'] = config['lr']
                print(f"💥 Updated all LRs to anti-collapse rates (base: {config['lr']})")
                
                
                scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=1.0, total_iters=1000000
                )
                print("💥 Replaced scheduler with ConstantLR to maintain anti-collapse rates")
                
        # Restore training state
        start_epoch = checkpoint.get('epoch', 0) + 1  # Start from next epoch
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        best_participation_ratio = checkpoint.get('participation_ratio', 0.0)
        
        # Restore running means if available
        if 'running_means' in checkpoint:
            running_means = checkpoint['running_means']
            print("✓ Restored running means for centroid loss")
        
        # Restore scheduler state if available
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("✓ Restored scheduler state")
        else:
            # Advance scheduler to correct epoch
            for _ in range(start_epoch):
                scheduler.step()
            print(f"✓ Advanced scheduler to epoch {start_epoch}")
        
        # Restore patience counters if available
        patience_counter = checkpoint.get('patience_counter', 0)
        pr_patience_counter = checkpoint.get('pr_patience_counter', 0)
        
        # Try to load training history
        history_path = os.path.join(output_dir, 'training_history.json')
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r') as f:
                    saved_history = json.load(f)
                # Truncate history to start_epoch if needed
                for key in history:
                    if key in saved_history and len(saved_history[key]) >= start_epoch:
                        history[key] = saved_history[key][:start_epoch]
                    elif key in saved_history:
                        history[key] = saved_history[key]
                print(f"✓ Restored training history up to epoch {start_epoch}")
                
                # Check if PR collapsed in recent history
                if 'val_pr' in saved_history and len(saved_history['val_pr']) > 0:
                    recent_pr = saved_history['val_pr'][-1]
                    if recent_pr < 15.0:
                        print(f"⚠️ WARNING: Recent validation PR was low: {recent_pr:.2f}")
                        
            except Exception as e:
                print(f"⚠ Could not restore training history: {e}")
        
        # Save best model state for potential future use
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        print(f"\nResuming training from:")
        print(f"  - Epoch: {start_epoch}")
        print(f"  - Best val loss: {best_val_loss:.4f}")
        print(f"  - Best PR: {best_participation_ratio:.2f}")
        print(f"  - Patience counter: {patience_counter}")
        print(f"  - PR patience counter: {pr_patience_counter}")
        print("="*60)
    
    max_patience = config.get('early_stopping_patience', 15)
    pr_patience = config.get('pr_patience', 20)
    
    print("\n" + "="*60)
    print(f"Starting Training with Risk-Specific Priors")
    if start_epoch > 0:
        print(f"(Resuming from epoch {start_epoch})")
    print("="*60)
    
    # Create overall progress bar
    epoch_pbar = tqdm(range(start_epoch, config['num_epochs']), desc="Training Progress", position=0)
    
    for epoch in epoch_pbar:
        epoch_start_time = time.time()
        
        # Calculate KL weight for annealing with warmup
        cycle_length = config['num_epochs'] // config.get('annealing_cycles', 4)
        kl_weight = get_kl_weight(
            epoch, config['num_epochs'],
            annealing_type=config.get('annealing_type', 'cyclical'),
            cycle_length=cycle_length,
            warmup_epochs=config.get('warmup_epochs', 5)
        )
        
        # Adjust centroid weight schedule (start small, increase after warmup)
        if epoch < config.get('warmup_epochs', 5):
            centroid_weight_schedule = 0.0
        elif epoch < config.get('warmup_epochs', 5) + 10:
            progress = (epoch - config.get('warmup_epochs', 5)) / 10
            centroid_weight_schedule = config['centroid_weight'] * progress
        else:
            centroid_weight_schedule = config['centroid_weight']
        
        # Store the effective weight for this epoch
        effective_centroid_weight = centroid_weight_schedule
        
        # Training
        model.train()
        train_losses = {}
        train_batches = 0
        total_confidence = 0
        confidence_count = 0
        all_train_z = []
        
        # Progress bar for training batches
        train_pbar = tqdm(train_loader, 
                         desc=f"Epoch {epoch+1}/{config['num_epochs']} [Train] KL:{kl_weight:.3f} Cent:{centroid_weight_schedule:.3f}",
                         position=1,
                         leave=False)
        
        for batch_idx, batch in enumerate(train_pbar):
            # Handle both enhanced and regular datasets
            if len(batch) == 3:
                data, labels, confidence = batch
                data, labels, confidence = data.to(device), labels.to(device), confidence.to(device)
                total_confidence += confidence[labels >= 0].sum().item()
                confidence_count += (labels >= 0).sum().item()
            else:
                data, labels = batch
                data, labels = data.to(device), labels.to(device)
                confidence = None
            
            optimizer.zero_grad()
            
            # Forward pass with label conditioning
            return_seq = config.get('temporal_consistency_weight', 0) > 0
            outputs = model(data, labels, return_sequences=return_seq)
            
            # Compute losses with confidence if available and running means
            losses = compute_cvae_losses(outputs, data, labels, config, epoch, 
                                       confidence, kl_weight, running_means, 
                                       centroid_weight=effective_centroid_weight)
            
            # Check training health
            if not check_training_health(model, losses, batch_idx):
                print(f"\nSkipping batch {batch_idx} due to unhealthy training state")
                optimizer.zero_grad()
                continue
            
            # Backward pass
            losses['total'].backward()
            
            # Safe gradient clipping with extra aggressive settings
            clip_gradients_safe(model, clip_value=0.1)
            
            optimizer.step()
            
            # Check if embeddings or priors have become NaN and reset if needed
            for name, param in model.named_parameters():
                if ('embedding' in name or 'risk_prior' in name) and (torch.isnan(param).any() or torch.isinf(param).any()):
                    print(f"\nWarning: NaN in {name}, reinitializing...")
                    if 'embedding' in name:
                        nn.init.normal_(param.data, mean=0.0, std=0.001)
                    else:  # risk_prior
                        if 'mu' in name:
                            # Reinitialize prior means with separation
                            angles = torch.linspace(0, 2 * np.pi, 6)[:-1]
                            if param.shape[1] >= 2:
                                param.data[:5, 0] = torch.cos(angles) * 2.0
                                param.data[:5, 1] = torch.sin(angles) * 2.0
                                if param.shape[1] > 2:
                                    param.data[:5, 2:] = torch.randn(5, param.shape[1] - 2) * 0.1
                            param.data[5, :] = 0  # Unknown at origin
                        else:  # log_sigma
                            param.data.zero_()
                            param.data[5, :] = 0.5  # Larger variance for unknown
            
            # Track losses
            for k, v in losses.items():
                if k not in train_losses:
                    train_losses[k] = 0
                if torch.is_tensor(v):
                    train_losses[k] += v.item()
                else:
                    train_losses[k] += float(v)
            train_batches += 1
            
            # Collect latents for monitoring (subsample to avoid memory issues)
            if batch_idx % 10 == 0:
                z_detached = outputs['z_all'].detach().cpu()
                if not (torch.isnan(z_detached).any() or torch.isinf(z_detached).any()):
                    all_train_z.append(z_detached)
            
            # Update progress bar
            postfix = {
                'loss': f"{losses['total'].item():.4f}",
                'recon': f"{losses['recon'].item():.4f}",
                'kl': f"{sum(losses.get(f'kl_{l}', torch.tensor(0.0)).item() for l in ['short', 'mid', 'long']):.4f}"
            }
            if 'diversity' in losses:
                postfix['div'] = f"{losses['diversity'].item():.4f}"
            if 'contrastive' in losses:
                postfix['cont'] = f"{losses['contrastive'].item():.4f}"
            if 'centroid' in losses:
                postfix['cent'] = f"{losses['centroid'].item():.4f}"
            if 'classification' in losses:
                postfix['cls'] = f"{losses['classification'].item():.4f}"
            if confidence_count > 0:
                postfix['conf'] = f"{total_confidence / confidence_count:.3f}"
            train_pbar.set_postfix(postfix)
        
        # Average training losses
        for k in train_losses:
            train_losses[k] /= train_batches
        
        # Track average confidence
        avg_train_conf = total_confidence / max(confidence_count, 1) if confidence_count > 0 else 0
        
        # Calculate training PR and cosine similarity
        train_pr = 0.0
        avg_cosine_sim = 0.0
        if all_train_z:
            all_train_z = torch.cat(all_train_z, dim=0).numpy()
            train_pr = calculate_participation_ratio(all_train_z)
            
            # Calculate average pairwise cosine similarity
            norms = np.linalg.norm(all_train_z, axis=1, keepdims=True)
            normalized = all_train_z / (norms + 1e-10)
            n_samples = min(1000, len(normalized))
            if n_samples > 1:
                idx = np.random.choice(len(normalized), n_samples, replace=False)
                sample_norm = normalized[idx]
                cos_sim = np.abs(sample_norm @ sample_norm.T)
                avg_cosine_sim = (cos_sim.sum() - n_samples) / (n_samples * (n_samples - 1))
        
        # Calculate prior separation (average distance between risk prior means)
        with torch.no_grad():
            prior_means_short = model.risk_prior_short.get_all_means()[:5]  # Exclude unknown
            prior_means_mid = model.risk_prior_mid.get_all_means()[:5]
            prior_means_long = model.risk_prior_long.get_all_means()[:5]
            
            # Average pairwise distance between risk priors
            def avg_pairwise_dist(means):
                n = len(means)
                if n < 2:
                    return 0
                dists = []
                for i in range(n):
                    for j in range(i+1, n):
                        dists.append(torch.norm(means[i] - means[j]).item())
                return np.mean(dists)
            
            prior_sep = (avg_pairwise_dist(prior_means_short) + 
                        avg_pairwise_dist(prior_means_mid) + 
                        avg_pairwise_dist(prior_means_long)) / 3
        
        # Validation with progress bar
        model.eval()
        val_losses = {}
        val_batches = 0
        all_val_z = []
        val_confidence_total = 0
        val_confidence_count = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, 
                           desc=f"Epoch {epoch+1}/{config['num_epochs']} [Val]",
                           position=1,
                           leave=False)
            
            for batch in val_pbar:
                # Handle both enhanced and regular datasets
                if len(batch) == 3:
                    data, labels, confidence = batch
                    data, labels, confidence = data.to(device), labels.to(device), confidence.to(device)
                    val_confidence_total += confidence[labels >= 0].sum().item()
                    val_confidence_count += (labels >= 0).sum().item()
                else:
                    data, labels = batch
                    data, labels = data.to(device), labels.to(device)
                    confidence = None
                
                outputs = model(data, labels)
                # Don't apply centroid regularization during validation
                losses = compute_cvae_losses(outputs, data, labels, config, epoch, 
                                           confidence, kl_weight, running_means,
                                           centroid_weight=0.0)
                
                # Track losses
                for k, v in losses.items():
                    if k not in val_losses:
                        val_losses[k] = 0
                    if torch.is_tensor(v):
                        val_losses[k] += v.item()
                    else:
                        val_losses[k] += float(v)
                val_batches += 1
                
                # Collect latents for PR calculation
                all_val_z.append(outputs['z_all'].cpu())
                
                # Update progress bar
                val_pbar.set_postfix({
                    'loss': f"{losses['total'].item():.4f}"
                })
        
        # Average validation losses
        for k in val_losses:
            val_losses[k] /= val_batches
        
        avg_val_conf = val_confidence_total / max(val_confidence_count, 1) if val_confidence_count > 0 else 0
        
        # Calculate validation participation ratio
        val_pr = 0.0
        if all_val_z:
            all_val_z = torch.cat(all_val_z).numpy()
            val_pr = calculate_participation_ratio(all_val_z)
        
        # Update learning rate
        scheduler.step()
        
        # Epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Check for collapse
        collapse_warning = ""
        if train_pr < 2.0:
            collapse_warning = " ⚠️ SEVERE COLLAPSE!"
        elif train_pr < 5.0:
            collapse_warning = " ⚠️ Collapse warning"
        elif train_pr < config['min_participation_ratio']:
            collapse_warning = " ⚠️ Low PR"
        
        # Print epoch summary
        tqdm.write(f"\nEpoch {epoch+1}/{config['num_epochs']} ({epoch_time:.1f}s):{collapse_warning}")
        tqdm.write(f"  Train: loss={train_losses['total']:.4f}, PR={train_pr:.2f}, cos_sim={avg_cosine_sim:.3f}")
        tqdm.write(f"  Val: loss={val_losses['total']:.4f}, PR={val_pr:.2f}")
        tqdm.write(f"  Components: recon={train_losses['recon']:.4f}, "
              f"kl={train_losses.get('kl_short', 0) + train_losses.get('kl_mid', 0) + train_losses.get('kl_long', 0):.4f}, "
              f"div={train_losses.get('diversity', 0):.4f}, "
              f"cent={train_losses.get('centroid', 0):.4f}, "
              f"cls={train_losses.get('classification', 0):.4f}")
        tqdm.write(f"  MI/TC: MI={-train_losses.get('mi', 0):.4f}, TC={train_losses.get('tc', 0):.4f}")
        tqdm.write(f"  Prior separation: {prior_sep:.3f}")
        if using_enhanced:
            tqdm.write(f"  Avg confidence: train={avg_train_conf:.3f}, val={avg_val_conf:.3f}")
        tqdm.write(f"  Settings: kl_weight={kl_weight:.3f}, centroid_weight={centroid_weight_schedule:.3f}, lr={optimizer.param_groups[0]['lr']:.6f}")
        
        # Track history
        history['train_loss'].append(train_losses['total'])
        history['val_loss'].append(val_losses['total'])
        history['train_pr'].append(train_pr)
        history['val_pr'].append(val_pr)
        history['train_kl'].append(train_losses.get('kl_short', 0) + train_losses.get('kl_mid', 0) + train_losses.get('kl_long', 0))
        history['train_diversity'].append(train_losses.get('diversity', 0))
        history['train_centroid'].append(train_losses.get('centroid', 0))
        history['train_classification'].append(train_losses.get('classification', 0))
        history['train_cosine_sim'].append(avg_cosine_sim)
        history['train_mi'].append(-train_losses.get('mi', 0))  # Store positive MI
        history['train_tc'].append(train_losses.get('tc', 0))
        history['train_temporal'].append(train_losses.get('temporal', 0))
        history['prior_separation'].append(prior_sep)
        if using_enhanced:
            history['avg_confidence'].append(avg_val_conf)
        
        # Save training history after each epoch
        history_path = os.path.join(output_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, default=lambda x: float(x) if isinstance(x, np.floating) else x)
        
        # Model selection based on PR and loss
        save_model = False
        save_reason = ""
        
        # Priority 1: Save if PR is good and loss improved
        if (val_pr > config['min_participation_ratio'] and 
            val_losses['total'] < best_val_loss):
            best_val_loss = val_losses['total']
            best_participation_ratio = val_pr
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            save_model = True
            save_reason = f"Best model (PR={val_pr:.2f}, loss={val_losses['total']:.4f})"
            patience_counter = 0
            pr_patience_counter = 0
        
        # Priority 2: Save if PR significantly improved even if loss is worse
        elif val_pr > best_participation_ratio * 1.2:  # 20% improvement
            best_participation_ratio = val_pr
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            save_model = True
            save_reason = f"Better PR (PR={val_pr:.2f})"
            pr_patience_counter = 0
        
        # Priority 3: Save periodic backup if PR is good
        elif val_pr > 15.0 and epoch % 10 == 0:
            save_model = True
            save_reason = f"Good PR backup (PR={val_pr:.2f})"
        else:
            patience_counter += 1
            pr_patience_counter += 1
        
        # Save checkpoint if needed (with enhanced information for resuming)
        if save_model:
            checkpoint_path = os.path.join(output_dir, 
                f'checkpoint_epoch{epoch}_pr{val_pr:.1f}_psep{prior_sep:.2f}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_losses['total'],
                'participation_ratio': val_pr,
                'val_pr': val_pr,  # Explicit validation PR
                'train_pr': train_pr,  # Also save training PR
                'prior_separation': prior_sep,
                'config': config,
                'data_type': model.data_type,
                'using_enhanced_labels': using_enhanced,
                'running_means': running_means,
                'patience_counter': patience_counter,
                'pr_patience_counter': pr_patience_counter,
                'best_val_loss': best_val_loss,
                'best_participation_ratio': best_participation_ratio
            }, checkpoint_path)
            tqdm.write(f"  → Saved: {save_reason}")
        
        # Always save a "latest" checkpoint for easy resuming
        latest_checkpoint_path = os.path.join(output_dir, 'checkpoint_latest.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_losses['total'],
            'participation_ratio': val_pr,
            'val_pr': val_pr,  # Explicit validation PR
            'train_pr': train_pr,  # Also save training PR
            'prior_separation': prior_sep,
            'config': config,
            'data_type': model.data_type,
            'using_enhanced_labels': using_enhanced,
            'running_means': running_means,
            'patience_counter': patience_counter,
            'pr_patience_counter': pr_patience_counter,
            'best_val_loss': best_val_loss,
            'best_participation_ratio': best_participation_ratio
        }, latest_checkpoint_path)
        
        # Update main progress bar
        epoch_pbar.set_postfix({
            'best_loss': f"{best_val_loss:.4f}",
            'best_pr': f"{best_participation_ratio:.2f}",
            'patience': f"{patience_counter}/{max_patience}"
        })
        
        # Early stopping based on PR
        if pr_patience_counter >= pr_patience and val_pr < config['min_participation_ratio']:
            tqdm.write(f"\n❌ Stopping due to persistent low PR ({val_pr:.2f})")
            break
        
        # Regular early stopping
        if patience_counter >= max_patience and epoch > config['num_epochs'] // 2:
            tqdm.write(f"\n✓ Early stopping triggered")
            break
    
    epoch_pbar.close()
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n✅ Restored best model: PR={best_participation_ratio:.2f}, loss={best_val_loss:.4f}")
    else:
        print("\n⚠️ WARNING: No good model found! Consider adjusting hyperparameters.")
    
    return model, history


# ============================================================================
# Main Training Script
# ============================================================================

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train Conditional Hierarchical VAE with Risk-Specific Priors')
    parser.add_argument('--data-type', type=str, default='sessions',
                       choices=['sessions', 'bets', 'payments', 'transactions'],
                       help='Type of gambling data to train on')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-5,
                       help='Learning rate')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: models/{data_type}_cvae_risk_priors)')
    parser.add_argument('--data-dir', type=str, default='processed_data_selected',
                       help='Root directory for processed data')
    parser.add_argument('--enhanced-labels-dir', type=str, default='enhanced_labels',
                       help='Directory containing enhanced labels')
    parser.add_argument('--force-original-labels', action='store_true',
                       help='Force using original labels even if enhanced are available')
    parser.add_argument('--use-class-balancing', action='store_true',
                       help='Enable class balancing for imbalanced datasets')
    
    # Checkpoint resuming arguments (NEW!)
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    parser.add_argument('--resume-latest', action='store_true',
                       help='Resume from latest checkpoint in output directory')
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        output_dir = os.path.join('models', f'{args.data_type}_cvae_risk_priors')
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Handle checkpoint resuming (NEW SECTION!)
    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = args.resume
        if not os.path.exists(resume_checkpoint):
            print(f"❌ Error: Checkpoint file not found: {resume_checkpoint}")
            return
    elif args.resume_latest:
        # Look for latest checkpoint in output directory
        latest_checkpoint = os.path.join(output_dir, 'checkpoint_latest.pt')
        if os.path.exists(latest_checkpoint):
            resume_checkpoint = latest_checkpoint
            print(f"Found latest checkpoint: {latest_checkpoint}")
        else:
            # Look for any checkpoint files
            checkpoint_files = [f for f in os.listdir(output_dir) if f.startswith('checkpoint_') and f.endswith('.pt')]
            if checkpoint_files:
                # Sort by modification time and get the most recent
                checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(output_dir, x)), reverse=True)
                resume_checkpoint = os.path.join(output_dir, checkpoint_files[0])
                print(f"Found most recent checkpoint: {resume_checkpoint}")
            else:
                print("❌ Error: No checkpoint files found in output directory")
                return
    
    # Initialize class weights (will be updated if class balancing is enabled)
    class_weights = [1.0, 1.0, 1.0, 1.0, 1.0]
    
    # Configuration with risk-specific priors
    config = {
        # Model architecture
        'latent_dims': [32, 24, 16],  # short, mid, long
        'hidden_dim': 256,
        'label_embed_dim': 16,
        
        # Risk-specific priors
        'prior_init_spread': 2.0,  # Initial separation between risk priors
        
        # Classification head
        'use_classification_head': True,
        'classification_weight': 0.3,
        
        # Anti-collapse features
        'use_spectral_norm': True,
        'diversity_weight': 0.8,
        'min_encoding_capacity': 20.0,
        'min_participation_ratio': 20.0,
        'pr_patience': 35,
        
        # Centroid loss (new!)
        'centroid_weight': 0.1,  # Will be scheduled during training
        
        # Mutual Information and disentanglement
        'use_mi_estimation': False,  # Start with off for stability
        'mi_weight': 0.0,
        'tc_weight': 0.05,
        
        # Temporal consistency for time series
        'temporal_consistency_weight': 0.1,
        'use_predictive_diversity': True,
        'pred_diversity_weight': 0.3,
        
        # Class balancing
        'use_class_balancing': args.use_class_balancing,
        'class_weights': class_weights,
        
        # Hierarchical KL weights (very small to prevent collapse)
        'kl_weights': [0.00001, 0.0001, 0.0005],
        'free_bits': [5.0, 3.0, 2.0],
        
        # KL annealing with warmup - CHANGED TO MONOTONIC!
        'annealing_type': 'monotonic',  # No more dangerous cycles!
        'annealing_cycles': 1,  # Not used with monotonic
        'warmup_epochs': 10,
        
        # Optional: Beta-VAE for stronger bottleneck
        'use_beta_vae': False,
        'beta': 4.0,
        
        # Contrastive learning
        'use_contrastive': True,
        'contrastive_weight': 0.5,
        
        # Training
        'batch_size': args.batch_size,
        'lr': args.lr,
        'num_epochs': args.epochs,
        'early_stopping_patience': 20,
        'dropout': 0.3,
        
        # Data
        'num_workers': 0,
        'data_type': args.data_type
    }
    
    # If resuming, potentially override config with checkpoint config
    if resume_checkpoint:
        checkpoint = torch.load(resume_checkpoint, map_location='cpu')
        if 'config' in checkpoint:
            print("\n⚠ Found config in checkpoint. Merging with current config...")
            # Keep certain command-line arguments
            checkpoint_config = checkpoint['config']
            checkpoint_config['num_epochs'] = args.epochs  # Allow extending training
            checkpoint_config['batch_size'] = args.batch_size  # Allow changing batch size
            checkpoint_config['lr'] = args.lr  # Allow changing learning rate
            checkpoint_config['num_workers'] = config['num_workers']  # Keep current
            # Ensure classification head settings are preserved
            checkpoint_config['use_classification_head'] = config['use_classification_head']
            checkpoint_config['classification_weight'] = config['classification_weight']
            config = checkpoint_config
            print("✓ Using configuration from checkpoint (with some overrides)")
            
            # Get start epoch and PR from checkpoint
            start_epoch = checkpoint.get('epoch', 0)
            checkpoint_pr = checkpoint.get('participation_ratio', 100.0)
            checkpoint_val_pr = checkpoint.get('val_pr', checkpoint_pr)  # Try to get validation PR
            
            # Detect PR collapse: activate anti-collapse if PR < 15 OR if resuming from later epochs
            pr_is_collapsed = checkpoint_val_pr < 15.0 or checkpoint_pr < 15.0
            
            # Anti-collapse override settings when resuming
            if pr_is_collapsed or start_epoch >= 10:
                if pr_is_collapsed:
                    print(f"\n🚨 PR COLLAPSE DETECTED! (PR={min(checkpoint_pr, checkpoint_val_pr):.2f}) 🚨")
                    print("🚨 AGGRESSIVE ANTI-COLLAPSE MODE ACTIVATED 🚨")
                    
                    
                else:
                    print("\n🚨 LATE-STAGE RESUME: ANTI-COLLAPSE MODE ACTIVATED 🚨")
                
                # NUCLEAR OPTION: Extreme settings to prevent any collapse
                config['diversity_weight'] = 5.0  # Was 0.8, then 2.0, now EXTREME
                config['kl_weights'] = [1e-10, 1e-9, 1e-8]  # Basically turn off KL
                config['min_encoding_capacity'] = 5.0  # Very low capacity requirement
                config['contrastive_weight'] = 0.01  # Almost off
                config['centroid_weight'] = 0.001  # Minimal centroid pull
                config['tc_weight'] = 0.001  # Minimal TC penalty
                
                # Force monotonic annealing to prevent future cycles
                config['annealing_type'] = 'monotonic'
                config['annealing_cycles'] = 1
                
                # Extend warmup way past current epoch
                config['warmup_epochs'] = start_epoch + 50
                
                config['use_mi_estimation'] = False
                config['mi_weight'] = 0.0  
                
                # High learning rate to escape collapse
                config['lr'] = 1e-3  # 100x boost from original
                
                # Turn off dropout to let information flow
                config['dropout'] = 0.0
                
                # Force high free bits to prevent KL from crushing latents
                config['free_bits'] = [20.0, 15.0, 10.0]  # Much higher
                
                print(f"  💥 Diversity weight: {config['diversity_weight']} (EXTREME)")
                print(f"  💥 KL weights: {config['kl_weights']} (NEARLY OFF)")
                print(f"  💥 Learning rate: {config['lr']} (100x BOOST)")
                print(f"  💥 Extended warmup to epoch {config['warmup_epochs']}")
                print(f"  💥 Dropout: DISABLED")
                print(f"  💥 Free bits: {config['free_bits']} (VERY HIGH)")
                print(f"  💥 Annealing: FORCED TO MONOTONIC (no more cycles!)")
                
            elif checkpoint_pr < 10.0:  # Moderate intervention for mild collapse
                print("\n⚠️ MODERATE COLLAPSE DETECTED - Applying gentler fixes")
                
                # Override to linear annealing
                config['annealing_type'] = 'linear'
                
                # Smaller KL weights
                config['kl_weights'] = [0.00001, 0.0001, 0.001]
                
                # Increase diversity a bit
                config['diversity_weight'] = 1.5  # Not as extreme as 5.0
                
                # Keep other settings moderate
                config['centroid_weight'] = 0.05
                config['warmup_epochs'] = 30
                
                print(f"  ⚠️ Switched to linear annealing")
                print(f"  ⚠️ Reduced KL weights: {config['kl_weights']}")
                print(f"  ⚠️ Moderate diversity weight: {config['diversity_weight']}")
    
    # Data paths
    data_dir = os.path.join(args.data_dir, f'{args.data_type}_windows')
    
    print(f"\n" + "="*60)
    print(f"Training Conditional Hierarchical VAE with Risk Priors for {args.data_type.upper()} data")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Enhanced labels directory: {args.enhanced_labels_dir}")
    if resume_checkpoint:  # NEW!
        print(f"Resuming from checkpoint: {resume_checkpoint}")
    print("="*60)
    
    # Load datasets with progress tracking
    print("\nLoading datasets...")
    
    # Overall dataset loading progress
    dataset_pbar = tqdm(total=3, desc="Loading all datasets", position=0)
    
    train_dataset = GamblingDataset(
        os.path.join(data_dir, 'X_train.npy'),
        os.path.join(data_dir, 'L_train.npy'),
        os.path.join(data_dir, 'feature_names.txt'),
        data_type=args.data_type,
        use_enhanced_labels=not args.force_original_labels,
        enhanced_labels_dir=args.enhanced_labels_dir
    )
    dataset_pbar.update(1)
    
    val_dataset = GamblingDataset(
        os.path.join(data_dir, 'X_val.npy'),
        os.path.join(data_dir, 'L_val.npy'),
        data_type=args.data_type,
        use_enhanced_labels=not args.force_original_labels,
        enhanced_labels_dir=args.enhanced_labels_dir
    )
    dataset_pbar.update(1)
    
    test_dataset = GamblingDataset(
        os.path.join(data_dir, 'X_test.npy'),
        os.path.join(data_dir, 'L_test.npy'),
        data_type=args.data_type,
        use_enhanced_labels=not args.force_original_labels,
        enhanced_labels_dir=args.enhanced_labels_dir
    )
    dataset_pbar.update(1)
    dataset_pbar.close()
    
    # Get feature groups
    feature_groups = train_dataset.get_feature_groups()
    
    # Calculate class weights if using class balancing
    if args.use_class_balancing:
        # Calculate inverse frequency weights from training data
        train_labels = train_dataset.labels
        labeled_mask = train_labels >= 0
        if labeled_mask.any():
            unique, counts = np.unique(train_labels[labeled_mask], return_counts=True)
            total_labeled = labeled_mask.sum()
            
            # Calculate inverse frequency weights with smoothing
            new_weights = [1.0, 1.0, 1.0, 1.0, 1.0]
            for class_idx, count in zip(unique, counts):
                if 0 <= class_idx < 5:
                    # Inverse frequency with square root for less extreme weights
                    new_weights[int(class_idx)] = np.sqrt(total_labeled / (5.0 * count))
            
            # Normalize weights to have mean of 1.0 for stability
            mean_weight = np.mean(new_weights)
            config['class_weights'] = [w / mean_weight for w in new_weights]
            
            # Clip weights to reasonable range
            config['class_weights'] = [np.clip(w, 0.2, 5.0) for w in config['class_weights']]
            
            print(f"\nCalculated class weights for balancing:")
            class_names = ['low_risk', 'low_medium_risk', 'medium_risk', 'medium_high_risk', 'high_risk']
            for name, weight in zip(class_names, config['class_weights']):
                print(f"  {name}: {weight:.2f}")
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=False
    )
    
    # Create model
    input_dim = train_dataset.X.shape[2]
    seq_len = train_dataset.X.shape[1]
    
    print(f"\nCreating conditional VAE model with risk-specific priors...")
    print(f"  Input dim: {input_dim}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Latent dims: {config['latent_dims']}")
    print(f"  Label embedding dim: {config['label_embed_dim']}")
    print(f"  Prior init spread: {config['prior_init_spread']}")
    print(f"  Classification head: {config['use_classification_head']}")
    
    model = ConditionalHierarchicalVAE(
        input_dim=input_dim,
        seq_len=seq_len,
        latent_dims=config['latent_dims'],
        hidden_dim=config['hidden_dim'],
        num_classes=5,
        feature_groups=feature_groups,
        data_type=args.data_type,
        label_embed_dim=config['label_embed_dim'],
        use_spectral_norm=config['use_spectral_norm'],
        diversity_weight=config['diversity_weight'],
        min_encoding_capacity=config['min_encoding_capacity'],
        use_mi_estimation=config['use_mi_estimation'],
        mi_weight=config['mi_weight'],
        temporal_consistency_weight=config['temporal_consistency_weight'],
        tc_weight=config['tc_weight'],
        centroid_weight=config['centroid_weight'],
        prior_init_spread=config['prior_init_spread'],
        dropout=config['dropout'],
        use_classification_head=config['use_classification_head'],
        classification_weight=config['classification_weight']
    ).to(device)
    
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model (with checkpoint resuming support) - CHANGED!
    model, history = train_model(model, train_loader, val_loader, config, device, output_dir, 
                                resume_checkpoint=resume_checkpoint)
    
    # Save final model
    model_path = os.path.join(output_dir, 'best_conditional_hvae_risk_priors.pt')
    torch.save({
        'epoch': len(history['train_loss']),
        'model_state_dict': model.state_dict(),
        'val_loss': min(history['val_loss']) if history['val_loss'] else float('inf'),
        'participation_ratio': max(history['val_pr']) if history['val_pr'] else 0,
        'prior_separation': history['prior_separation'][-1] if history['prior_separation'] else 0,
        'config': config,
        'data_type': args.data_type,
        'using_enhanced_labels': hasattr(train_dataset, 'using_enhanced') and train_dataset.using_enhanced
    }, model_path)
    
    # Save training history
    history_path = os.path.join(output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        # Fix: Handle numpy types
        json.dump(history, f, default=lambda x: float(x) if isinstance(x, np.floating) else x)
        
    # Plot training curves with PR monitoring and prior separation
    print("\nGenerating training curves...")
    fig, axes = plt.subplots(3, 4, figsize=(24, 18))
    
    # Loss curves
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Val')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title(f'Training Loss - {args.data_type.capitalize()} (Risk-Prior CVAE)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Participation Ratio
    axes[0, 1].plot(history['train_pr'], 'b-', label='Train PR')
    axes[0, 1].plot(history['val_pr'], 'r-', label='Val PR')
    axes[0, 1].axhline(y=15, color='g', linestyle='--', label='Good PR')
    axes[0, 1].axhline(y=10, color='orange', linestyle='--', label='Min acceptable')
    axes[0, 1].axhline(y=5, color='red', linestyle='--', label='Collapse threshold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Participation Ratio')
    axes[0, 1].set_title('PR Monitoring (Higher is Better)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # KL Loss
    axes[0, 2].plot(history['train_kl'])
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('KL Loss')
    axes[0, 2].set_title('KL Divergence')
    axes[0, 2].grid(True)
    
    # Prior Separation (NEW!)
    axes[0, 3].plot(history['prior_separation'])
    axes[0, 3].set_xlabel('Epoch')
    axes[0, 3].set_ylabel('Average Distance')
    axes[0, 3].set_title('Risk Prior Separation (Higher = Better Separation)')
    axes[0, 3].grid(True)
    
    # Diversity Loss
    axes[1, 0].plot(history['train_diversity'])
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Diversity Loss')
    axes[1, 0].set_title('Diversity Loss')
    axes[1, 0].grid(True)
    
    # Cosine Similarity
    axes[1, 1].plot(history['train_cosine_sim'])
    axes[1, 1].axhline(y=0.9, color='r', linestyle='--', label='High similarity (bad)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Cosine Similarity')
    axes[1, 1].set_title('Average Cosine Similarity (Lower is Better)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Centroid Loss
    axes[1, 2].plot(history['train_centroid'])
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Centroid Loss')
    axes[1, 2].set_title('Centroid Loss (Pulls to Risk Centers)')
    axes[1, 2].grid(True)
    
    # Classification Loss
    axes[1, 3].plot(history['train_classification'])
    axes[1, 3].set_xlabel('Epoch')
    axes[1, 3].set_ylabel('Classification Loss')
    axes[1, 3].set_title('Risk Classification Loss')
    axes[1, 3].grid(True)
    
    # Total Correlation
    axes[2, 0].plot(history['train_tc'])
    axes[2, 0].set_xlabel('Epoch')
    axes[2, 0].set_ylabel('Total Correlation')
    axes[2, 0].set_title('Total Correlation (Lower = More Disentangled)')
    axes[2, 0].grid(True)
    
    # Temporal Consistency
    axes[2, 1].plot(history['train_temporal'])
    axes[2, 1].set_xlabel('Epoch')
    axes[2, 1].set_ylabel('Temporal Loss')
    axes[2, 1].set_title('Temporal Consistency Loss')
    axes[2, 1].grid(True)
    
    # Confidence (if using enhanced labels)
    if 'avg_confidence' in history and history['avg_confidence']:
        axes[2, 2].plot(history['avg_confidence'])
        axes[2, 2].set_xlabel('Epoch')
        axes[2, 2].set_ylabel('Average Confidence')
        axes[2, 2].set_title('Label Confidence During Training')
        axes[2, 2].grid(True)
    else:
        axes[2, 2].axis('off')
    
    # Visualize final prior means
    with torch.no_grad():
        risk_labels = ['Low', 'L-M', 'Med', 'M-H', 'High']
        colors = ['green', 'yellowgreen', 'yellow', 'orange', 'red']
        markers = ['o', '^', 's']  # circle, triangle, square
        
        for idx, (level, prior) in enumerate([('short', model.risk_prior_short),
                                              ('mid', model.risk_prior_mid),
                                              ('long', model.risk_prior_long)]):
            means = prior.get_all_means()[:5].cpu().numpy()  # Exclude unknown
            if means.shape[1] >= 2:
                scatter = axes[2, 3].scatter(means[:, 0], means[:, 1], 
                                           label=f'{level} priors', 
                                           s=100, alpha=0.7, marker=markers[idx])
                
                # Add risk labels for this level
                for i, label in enumerate(risk_labels):
                    axes[2, 3].annotate(f'{label[0]}-{level[0]}', 
                                      (means[i, 0], means[i, 1]), 
                                      xytext=(5, 5), textcoords='offset points',
                                      fontsize=8)
        
        axes[2, 3].set_xlabel('Dimension 1')
        axes[2, 3].set_ylabel('Dimension 2')
        axes[2, 3].set_title('Learned Risk Prior Means (2D Projection)')
        axes[2, 3].legend()
        axes[2, 3].grid(True)
    
    plt.tight_layout()
    curves_path = os.path.join(output_dir, 'training_curves_risk_priors.png')
    plt.savefig(curves_path, dpi=150)
    plt.close()
    
    # Extract and save embeddings
    print("\n" + "="*60)
    print("Extracting Embeddings with Risk-Specific Structure")
    print("="*60)
    
    model.eval()
    
    # Progress bar for embedding extraction
    embed_pbar = tqdm(total=3, desc="Extracting embeddings for all splits")
    
    for split, loader in [('train', train_loader), ('val', val_loader), ('test', test_loader)]:
        embeddings = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Extracting {split} embeddings", leave=False):
                if len(batch) == 3:
                    data, labels, _ = batch  # data, labels, confidence
                else:
                    data, labels = batch  # data, labels
                data = data.to(device)
                labels = labels.to(device)
                emb = model.get_embeddings(data, labels)
                embeddings.append(emb.cpu().numpy())
        
        embeddings = np.vstack(embeddings)
        
        # Calculate final PR for this split
        final_pr = calculate_participation_ratio(embeddings)
        
        save_path = os.path.join(data_dir, f'embeddings_{split}_cvae_risk_priors.npy')
        np.save(save_path, embeddings)
        print(f"  ✓ Saved {split} embeddings: {embeddings.shape}, PR={final_pr:.2f} to {save_path}")
        embed_pbar.update(1)
    
    embed_pbar.close()
    
    # Save summary
    summary = {
        'data_type': args.data_type,
        'model_type': 'conditional_hierarchical_vae_risk_priors',
        'model_path': model_path,
        'best_val_loss': min(history['val_loss']) if history['val_loss'] else float('inf'),
        'best_participation_ratio': max(history['val_pr']) if history['val_pr'] else 0,
        'final_prior_separation': history['prior_separation'][-1] if history['prior_separation'] else 0,
        'using_enhanced_labels': hasattr(train_dataset, 'using_enhanced') and train_dataset.using_enhanced,
        'resumed_from_checkpoint': resume_checkpoint is not None,  # NEW!
        'embeddings': {
            'train': f'embeddings_train_cvae_risk_priors.npy',
            'val': f'embeddings_val_cvae_risk_priors.npy',
            'test': f'embeddings_test_cvae_risk_priors.npy'
        },
        'feature_groups': {
            'short': len(feature_groups['short']),
            'mid': len(feature_groups['mid']),
            'long': len(feature_groups['long'])
        },
        'anti_collapse_features': {
            'risk_specific_priors': True,
            'prior_init_spread': config['prior_init_spread'],
            'spectral_norm': config['use_spectral_norm'],
            'diversity_weight': config['diversity_weight'],
            'centroid_weight': config['centroid_weight'],
            'min_encoding_capacity': config['min_encoding_capacity'],
            'min_pr_threshold': config['min_participation_ratio'],
            'annealing_type': config.get('annealing_type', 'cyclical'),
            'annealing_cycles': config['annealing_cycles'],
            'warmup_epochs': config.get('warmup_epochs', 5),
            'use_mi_estimation': config.get('use_mi_estimation', True),
            'mi_weight': config.get('mi_weight', 0.1),
            'temporal_consistency_weight': config.get('temporal_consistency_weight', 0.1),
            'tc_weight': config.get('tc_weight', 0.05),
            'use_classification_head': config.get('use_classification_head', True),
            'classification_weight': config.get('classification_weight', 0.3)
        },
        'config': config
    }
    
    summary_path = os.path.join(output_dir, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n" + "="*60)
    print(f"✓ Training complete for {args.data_type} Conditional VAE with Risk-Specific Priors!")
    print(f"✓ Best validation loss: {summary['best_val_loss']:.4f}")
    print(f"✓ Best participation ratio: {summary['best_participation_ratio']:.2f}")
    print(f"✓ Final prior separation: {summary['final_prior_separation']:.3f}")
    print(f"✓ Using enhanced labels: {summary['using_enhanced_labels']}")
    print(f"✓ Classification head enabled: {config.get('use_classification_head', True)}")
    if resume_checkpoint:  # NEW!
        print(f"✓ Resumed from checkpoint: {resume_checkpoint}")
    print(f"\n✓ All outputs saved to: {output_dir}")
    print(f"✓ Embeddings saved with suffix '_cvae_risk_priors.npy'")
    print(f"\n✓ Next step: Run DBN model with the risk-aware conditional VAE embeddings")
    print("✓ The risk-specific priors should eliminate the permutation problem!")
    print("✓ The classification head provides additional risk prediction capability!")
    print("="*60)


if __name__ == "__main__":
    main()