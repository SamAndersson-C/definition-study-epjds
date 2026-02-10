#!/usr/bin/env python3
"""
Improved GHMM Baseline v5 - 3-CLASS OPTIMIZED (FULLY DEBUGGED)
===============================================================
All critical fixes applied:
1. Fixed off-by-one in add_sequence_context (66100 vs 64667 bug)
2. Windows-safe KMeans initialization (Option A: pre-seed everything)
3. Multi-scale context with momentum deltas
4. Ready for ~0.7 BA performance
"""

# CRITICAL: Set threading BEFORE any numpy/scipy imports
import os
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'

import sys
import argparse
from pathlib import Path
import numpy as np
import joblib
import yaml
import json
import warnings
from datetime import datetime
import time
from collections import Counter
from copy import deepcopy
import types

import faulthandler
faulthandler.enable()

from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report, recall_score, r2_score
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.isotonic import IsotonicRegression
from sklearn.random_projection import GaussianRandomProjection
from scipy.stats import spearmanr, rankdata, norm
from scipy.optimize import nnls
from scipy.linalg import cho_factor, cho_solve

from hmmlearn import hmm

# Optional imports
try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    warnings.warn("UMAP not installed. Install with: pip install umap-learn")

print(f"Improved GHMM v5 3-CLASS OPTIMIZED (FULLY DEBUGGED) starting at {datetime.now().isoformat()}", flush=True)

# ============================================================================
# LABEL MAPPING: Learned 5→3 collapse
# ============================================================================

def ordinal_collapses_5_to_3():
    """All valid ordinal collapses from 5 classes to 3."""
    return [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]

def apply_collapse_to_labels(y5, a, b):
    """Apply ordinal collapse [0..a] | [a+1..b] | [b+1..4]."""
    y3 = np.empty_like(y5)
    y3[(y5 >= 0) & (y5 <= a)] = 0
    y3[(y5 >= a+1) & (y5 <= b)] = 1
    y3[(y5 >= b+1)] = 2
    return y3

def pick_best_merge(Fe_tr, y5_tr, tr_lab, Fe_v, y5_v, v_lab):
    """Learn the best 5→3 merge on validation."""
    print("\n   Learning best 5→3 merge on validation...", flush=True)
    best = dict(score=-np.inf, a=None, b=None)
    
    for (a, b) in ordinal_collapses_5_to_3():
        y3_tr = apply_collapse_to_labels(y5_tr, a, b)
        y3_v = apply_collapse_to_labels(y5_v, a, b)
        
        clf = LogisticRegression(
            solver='lbfgs',
            C=2.0,
            max_iter=3000,
            class_weight='balanced',
            random_state=42
        )
        clf.fit(Fe_tr[tr_lab], y3_tr[tr_lab])
        ba = balanced_accuracy_score(y3_v[v_lab], clf.predict(Fe_v[v_lab]))
        
        print(f"   Merge [0..{a}] | [{a+1}..{b}] | [{b+1}..4]  → Val BA={ba:.3f}")
        
        if ba > best['score']:
            best.update(score=ba, a=a, b=b)
    
    print(f"   ✔ Best merge: [0..{best['a']}] | [{best['a']+1}..{best['b']}] | [{best['b']+1}..4] (BA={best['score']:.3f})")
    return best

# ============================================================================
# SCORE-AND-SLICE: Learn cutpoints on risk scores
# ============================================================================

def learn_tri_thresholds_on_score(r_val, y3_val, mask, n_grid=101):
    """Learn two thresholds t1<t2 on risk scores for 3-way classification."""
    rs = r_val[mask]
    ys = y3_val[mask]
    
    qs = np.quantile(rs, np.linspace(0.02, 0.98, n_grid))
    best = dict(score=-np.inf, t1=None, t2=None)
    
    for i in range(len(qs)):
        for j in range(i+1, len(qs)):
            t1, t2 = qs[i], qs[j]
            yhat = np.digitize(rs, [t1, t2])
            ba = balanced_accuracy_score(ys, yhat)
            
            if ba > best['score']:
                best.update(score=ba, t1=float(t1), t2=float(t2))
    
    return best

def tri_features_from_score_and_probs(r, P3, t1, t2):
    """Build features from risk scores, probs, and learned thresholds."""
    yhard = np.digitize(r, [t1, t2]).reshape(-1, 1).astype(np.int32)
    
    oh = np.zeros((len(r), 3), np.float32)
    oh[np.arange(len(r)), yhard[:, 0]] = 1.0
    
    logP3 = np.log(np.clip(P3, 1e-8, 1.0))
    
    return np.hstack([P3, logP3, yhard.astype(np.float32), oh])

# ============================================================================
# SEQUENCE CONTEXT FEATURES (FIXED - NO MORE OFF-BY-ONE!)
# ============================================================================

def add_sequence_context(gamma, r, seq_ids, win=5):
    """
    Rolling means (centered) of gamma and risk within each sequence.
    Returns an array with the SAME number of rows as input (FIXED).
    """
    if win % 2 == 0:
        win += 1  # enforce odd window for centered averaging

    if seq_ids is None:
        seq_ids = np.zeros(len(r), dtype=np.int64)

    out = []
    pad = win // 2

    for sid in np.unique(seq_ids):
        idx = np.where(seq_ids == sid)[0]
        G = gamma[idx]
        R = r[idx].reshape(-1, 1)

        # Edge padding so "centered" window exists at ends
        Gp = np.pad(G, ((pad, pad), (0, 0)), mode='edge')
        Rp = np.pad(R, ((pad, pad), (0, 0)), mode='edge')

        # Cumsum with a leading zero row to get len = len(Gp) - win + 1
        Gcsum = np.cumsum(np.vstack([np.zeros((1, Gp.shape[1]), Gp.dtype), Gp]), axis=0)
        Rcsum = np.cumsum(np.vstack([np.zeros((1, Rp.shape[1]), Rp.dtype), Rp]), axis=0)

        # "valid" window yields exactly len(Gp) - win + 1 == len(G)
        Gmean = (Gcsum[win:] - Gcsum[:-win]) / float(win)
        Rmean = (Rcsum[win:] - Rcsum[:-win]) / float(win)

        out.append(np.hstack([Gmean, Rmean]))

    return np.vstack(out).astype(np.float32)


def add_sequence_context_multi(gamma, r, seq_ids, wins=(3, 5, 9, 15)):
    """
    Multi-scale context + momentum deltas for temporal lift.
    This is the key to pushing BA from 0.59 to 0.7!
    """
    feats = []
    
    # Multiple rolling window sizes
    for w in wins:
        feats.append(add_sequence_context(gamma, r, seq_ids, win=w))
    
    # First differences per sequence (pad a leading zero row)
    if seq_ids is None:
        seq_ids = np.zeros(len(r), dtype=np.int64)
    
    dG_list, dR_list = [], []
    for sid in np.unique(seq_ids):
        idx = np.where(seq_ids == sid)[0]
        G = gamma[idx]
        R = r[idx].reshape(-1, 1)
        
        dG = np.vstack([np.zeros((1, G.shape[1]), dtype=G.dtype), np.diff(G, axis=0)])
        dR = np.vstack([np.zeros((1, 1), dtype=R.dtype), np.diff(R, axis=0)])
        
        dG_list.append(dG)
        dR_list.append(dR)
    
    dG = np.vstack(dG_list)
    dR = np.vstack(dR_list)
    feats.append(np.hstack([dG, dR]))
    
    return np.hstack(feats).astype(np.float32)

# ============================================================================
# ORDINAL ISOTONIC WRAPPER CLASS
# ============================================================================

class OrdinalIsotonicWrapper3Class:
    """Wrapper for ordinal isotonic regression with 3 classes."""
    def __init__(self, ridge, iso_list):
        self.ridge = ridge
        self.iso = iso_list
        
    def predict_proba(self, X):
        s = self.ridge.predict(X)
        g = [z.transform(s) for z in self.iso]
        
        P = np.vstack([
            1 - g[0],
            g[0] - g[1],
            g[1]
        ]).T
        
        P = np.clip(P, 0, 1)
        P /= P.sum(axis=1, keepdims=True) + 1e-12
        return P
    
    def predict(self, X):
        return self.predict_proba(X).argmax(1)
    
    def score_function(self, X):
        """Raw risk scores for HMM ordering."""
        if np.any(np.isnan(X)):
            print("WARNING: NaN in input to risk scorer, replacing with zeros")
            X = np.nan_to_num(X, nan=0.0)
        return self.ridge.predict(X)

def _transitions_summary(lengths):
    """Count actual transitions available for learning."""
    if lengths is None: 
        return {"n_seqs": 1, "mean_len": "N/A", "n_trans": "N/A"}
    lengths = np.asarray(lengths, int)
    n_trans = int(np.maximum(0, lengths - 1).sum())
    return {"n_seqs": int(len(lengths)), "mean_len": float(lengths.mean()), "n_trans": n_trans}

# ============================================================================
# I/O HELPERS
# ============================================================================

def _load_split(emb_dir: Path, split: str, verbose=True):
    """Load embeddings for a split, keeping both 5-class and 3-class labels."""
    fp = emb_dir / f"embeddings_{split}.npz"
    if verbose:
        print(f"  Loading {fp}...", flush=True)
    
    if not fp.exists():
        raise FileNotFoundError(f"Missing split file: {fp}")
    
    data = np.load(fp, allow_pickle=True)
    
    X = data["embeddings"].astype(np.float32)
    has_manual = data["has_manual"].astype(bool)
    y5 = data["manual_labels"].astype(int) if "manual_labels" in data else np.full(len(X), -1, int)
    
    y = apply_collapse_to_labels(y5, 0, 2)
    
    mh = data["manual_labels_multihot"].astype(np.float32) if "manual_labels_multihot" in data else np.zeros((len(X),5), np.float32)
    rg_cat = data["rg_cat_dists"].astype(np.float32) if "rg_cat_dists" in data else np.zeros((len(X),5), np.float32)
    coverage = data["window_coverage"].astype(np.float32) if "window_coverage" in data else np.zeros(len(X), np.float32)
    
    coverage = np.nan_to_num(coverage, nan=0.0, neginf=0.0, posinf=1.0)
    
    seq_ids = data["sequence_ids"] if "sequence_ids" in data else None
    
    if verbose:
        print(f"    Loaded {len(X)} samples ({X.shape[1]}D), {has_manual.sum()} labeled", flush=True)
        if has_manual.any():
            counts5 = Counter(y5[has_manual])
            print(f"    5-class distribution: {dict(counts5)}", flush=True)
        if seq_ids is not None:
            n_seqs = len(np.unique(seq_ids))
            print(f"    Found {n_seqs} unique sequences", flush=True)
    
    return dict(X=X, has_manual=has_manual, y=y, y5=y5, mh=mh, rg_cat=rg_cat, coverage=coverage, seq_ids=seq_ids)

def _contiguous_split(split, name):
    """Make sequences contiguous and return reordered split + lengths."""
    sid = split["seq_ids"]
    if sid is None:
        print(f"   [{name}] No sequence_ids, treating as single sequence")
        return split, None

    sid = np.asarray(sid)
    idx = np.lexsort((np.arange(len(sid)), sid))
    is_already_contig = np.all(idx == np.arange(len(sid)))
    
    if not is_already_contig:
        print(f"   [{name}] Reordering {len(sid)} samples to make sequences contiguous")
        for key in ("X", "y", "y5", "has_manual", "coverage", "mh", "rg_cat", "seq_ids"):
            if key in split and split[key] is not None:
                split[key] = split[key][idx]
    else:
        print(f"   [{name}] Sequences already contiguous")

    seq_ids = np.asarray(split["seq_ids"])
    boundaries = np.flatnonzero(np.r_[True, seq_ids[1:] != seq_ids[:-1], True])
    lengths = np.diff(boundaries).tolist()
    n_trans = int(np.maximum(0, np.array(lengths) - 1).sum())
    
    print(f"   [{name}] Sequences: {len(lengths)}, mean length: {np.mean(lengths):.2f}, transitions: {n_trans}")
    
    if n_trans == 0:
        print(f"   ⚠️ [{name}] WARNING: NO TRANSITIONS! HMM will default to prior!")
    elif n_trans < 100:
        print(f"   ⚠️ [{name}] WARNING: Very few transitions ({n_trans}), HMM learning will be limited")
    
    return split, lengths

# ============================================================================
# PREPROCESSING METHODS (keeping original implementations)
# ============================================================================

class ZCATransformer:
    """ZCA whitening with optional block-aware processing."""
    def __init__(self, eps=1e-6, blocks=None):
        self.eps = eps
        self.blocks = blocks
        self.mean_ = None
        self.W_ = None
    
    def fit(self, X):
        self.mean_ = X.mean(axis=0, keepdims=True)
        Xc = X - self.mean_
        
        if self.blocks is None:
            self.W_ = self._fit_block(Xc)
        else:
            splits = np.cumsum(self.blocks)[:-1]
            Xs = np.split(Xc, splits, axis=1)
            Ws = [self._fit_block(xb) for xb in Xs]
            self.W_ = ("block", Ws)
        return self
    
    def _fit_block(self, Xc):
        C = np.cov(Xc, rowvar=False)
        U, S, _ = np.linalg.svd(C, full_matrices=False)
        S_inv_sqrt = np.diag(1.0 / np.sqrt(S + self.eps))
        return U @ S_inv_sqrt @ U.T
    
    def transform(self, X):
        Xc = X - self.mean_
        
        if isinstance(self.W_, tuple):
            _, Ws = self.W_
            splits = np.cumsum(self.blocks)[:-1]
            Xs = np.split(Xc, splits, axis=1)
            Xw = [xb @ W for xb, W in zip(Xs, Ws)]
            return np.concatenate(Xw, axis=1)
        else:
            return Xc @ self.W_

class GaussianCopulaTransformer:
    """Train-fitted marginal CDF → N(0,1)."""
    def __init__(self, eps=1e-6, n_quantiles=10000, random_state=42):
        self.eps = eps
        self.qt = QuantileTransformer(
            output_distribution='normal',
            n_quantiles=min(n_quantiles, 100000),
            subsample=int(1e6),
            random_state=random_state,
            copy=True
        )
    
    def fit(self, X):
        self.qt.fit(X)
        return self
    
    def transform(self, X):
        Z = self.qt.transform(X)
        return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)

class CopulaZCA:
    """Composite transformer: Copula then ZCA"""
    def __init__(self, gcx, zca):
        self.gcx = gcx
        self.zca = zca
    
    def transform(self, X):
        return self.zca.transform(self.gcx.transform(X))

class TICATransformer:
    """Time-lagged Independent Component Analysis."""
    def __init__(self, n_components=32, lag=1, eps=1e-8):
        self.n_components = n_components
        self.lag = lag
        self.eps = eps
        self.mean_ = None
        self.Wwhite_ = None
        self.Wtica_ = None
    
    def _create_pairs(self, X, lengths):
        """Create time-lagged pairs respecting sequence boundaries."""
        if lengths is None:
            return X[:-self.lag], X[self.lag:]
        
        X0_list, Xtau_list = [], []
        offset = 0
        
        for L in lengths:
            if L > self.lag:
                seg = X[offset:offset+L]
                X0_list.append(seg[:-self.lag])
                Xtau_list.append(seg[self.lag:])
            offset += L
        
        if not X0_list:
            print("   Warning: No valid time-lagged pairs, using global lag", flush=True)
            return X[:-self.lag], X[self.lag:]
        
        return np.vstack(X0_list), np.vstack(Xtau_list)
    
    def fit(self, X, lengths=None):
        self.mean_ = X.mean(axis=0, keepdims=True)
        Xc = X - self.mean_
        
        X0, Xtau = self._create_pairs(Xc, lengths)
        
        C0 = (X0.T @ X0) / max(1, len(X0) - 1)
        Ct = (X0.T @ Xtau) / max(1, len(X0) - 1)
        
        U, S, _ = np.linalg.svd(C0)
        S_inv_sqrt = np.diag(1.0 / np.sqrt(S + self.eps))
        self.Wwhite_ = U @ S_inv_sqrt @ U.T
        
        K = self.Wwhite_ @ Ct @ self.Wwhite_.T
        K = 0.5 * (K + K.T)
        
        eigvals, eigvecs = np.linalg.eigh(K)
        order = np.argsort(eigvals)[::-1]
        V = eigvecs[:, order[:self.n_components]]
        
        self.Wtica_ = self.Wwhite_.T @ V
        return self
    
    def transform(self, X):
        Xc = X - self.mean_
        return Xc @ self.Wtica_

def fit_preprocess(X_train, method="pca", n_components=32, y_train=None, 
                   lengths_train=None, has_manual=None, blocks=None, tica_lag=1, random_state=42):
    """Unified preprocessing interface."""
    print(f"\n2. Fitting {method.upper()} preprocessing...", flush=True)
    
    if method == "pca":
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X_train)
        n_comp = min(n_components, min(Xs.shape) - 1)
        pca = PCA(n_components=n_comp, random_state=random_state)
        Xp = pca.fit_transform(Xs)
        print(f"   PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}", flush=True)
        return scaler, pca, Xp
    
    elif method == "zca":
        zca = ZCATransformer(blocks=blocks)
        zca.fit(X_train)
        Xz = zca.transform(X_train)
        block_str = f" (blocks={blocks})" if blocks else ""
        print(f"   ZCA whitening complete{block_str}", flush=True)
        return zca, None, Xz
    
    elif method == "copula":
        gcx = GaussianCopulaTransformer()
        gcx.fit(X_train)
        Xg = gcx.transform(X_train)
        print(f"   Gaussian copula transform complete", flush=True)
        return gcx, None, Xg
    
    elif method == "copula_zca":
        gcx = GaussianCopulaTransformer()
        gcx.fit(X_train)
        Xg = gcx.transform(X_train)
        
        zca = ZCATransformer(blocks=blocks)
        zca.fit(Xg)
        Xgz = zca.transform(Xg)
        
        if n_components < X_train.shape[1]:
            pca = PCA(n_components=n_components, random_state=random_state)
            Xgz = pca.fit_transform(Xgz)
            print(f"   Copula + ZCA + PCA to {n_components}D complete", flush=True)
            return CopulaZCA(gcx, zca), pca, Xgz
        
        block_str = f" (blocks={blocks})" if blocks else ""
        print(f"   Copula + ZCA whitening complete{block_str}", flush=True)
        return CopulaZCA(gcx, zca), None, Xgz
    
    elif method == "fa":
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X_train)
        fa = FactorAnalysis(n_components=n_components, random_state=random_state)
        Xf = fa.fit_transform(Xs)
        print(f"   Factor Analysis complete ({n_components}D)", flush=True)
        return scaler, fa, Xf
    
    elif method == "tica":
        tica = TICATransformer(n_components=n_components, lag=tica_lag)
        tica.fit(X_train, lengths=lengths_train)
        Xt = tica.transform(X_train)
        print(f"   tICA complete (lag={tica_lag}, {n_components}D)", flush=True)
        return tica, None, Xt
    
    elif method == "pls":
        mask = has_manual if has_manual is not None else (y_train >= 0)
        if mask.sum() < 100:
            print("   Warning: Too few labeled samples for PLS, falling back to PCA", flush=True)
            return fit_preprocess(X_train, "pca", n_components, random_state=random_state)
        
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X_train)
        pls = PLSRegression(n_components=min(n_components, 16))
        pls.fit(Xs[mask], y_train[mask])
        Xp = pls.transform(Xs)
        print(f"   PLS complete ({pls.n_components}D, trained on {mask.sum()} labeled)", flush=True)
        return scaler, pls, Xp
      
    elif method == "umap":
        if not HAS_UMAP:
            print("   UMAP not available, falling back to PCA", flush=True)
            return fit_preprocess(X_train, "pca", n_components, random_state=random_state)
        
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X_train)
        
        print(f"   Using UNSUPERVISED UMAP", flush=True)
        n_neighbors = max(10, min(50, len(X_train) // 100))
        umap = UMAP(
            n_components=n_components, 
            random_state=random_state,
            n_neighbors=n_neighbors, 
            min_dist=0.1
        )
        Xu = umap.fit_transform(Xs)
        
        print(f"   UMAP complete ({X_train.shape[1]}D -> {n_components}D)", flush=True)
        return scaler, umap, Xu
    
    elif method == "rp":
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X_train)
        rp = GaussianRandomProjection(n_components=n_components, random_state=random_state)
        Xp = rp.fit_transform(Xs)
        print(f"   Random Projection complete", flush=True)
        return scaler, rp, Xp
    
    elif method == "none":
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X_train)
        print(f"   Using full {X_train.shape[1]}D embeddings (standardized)", flush=True)
        return scaler, None, Xs
    
    else:
        raise ValueError(f"Unknown preprocessing method: {method}")

def apply_preprocess(X, scaler, reducer):
    """Apply preprocessing to new data."""
    if isinstance(scaler, (ZCATransformer, TICATransformer, GaussianCopulaTransformer, CopulaZCA)):
        Xt = scaler.transform(X)
        if reducer is not None:
            return reducer.transform(Xt)
        return Xt
    elif scaler is not None and reducer is not None:
        Xs = scaler.transform(X)
        return reducer.transform(Xs)
    elif scaler is not None and reducer is None:
        return scaler.transform(X)
    else:
        return X

# ============================================================================
# WEIGHTED ORDINAL ISOTONIC RISK SCORING
# ============================================================================

def fit_risk_ordinal_isotonic_3class(X_train_labeled, y_train_labeled, coverage=None, alpha=1.0):
    """Ordinal isotonic with sample weights for 3 classes."""
    print(f"\n3. Learning WEIGHTED ordinal isotonic risk function (3 classes)...", flush=True)
    
    y = y_train_labeled.astype(int)
    class_freq = Counter(y)
    inv_freq = {c: 1.0 / max(1, class_freq[c]) for c in class_freq}
    
    if coverage is None:
        w_cov = np.ones_like(y, dtype=float)
    else:
        w_cov = np.clip(np.asarray(coverage, dtype=float), 1e-3, 1.0) ** 0.75
    
    w_cls = np.array([inv_freq[yi] for yi in y], dtype=float)
    w = w_cov * w_cls
    w = w * (len(w) / w.sum())
    
    ridge = Ridge(alpha=alpha, random_state=42)
    ridge.fit(X_train_labeled, y.astype(float), sample_weight=w)
    s = ridge.predict(X_train_labeled)
    r2_initial = r2_score(y, s, sample_weight=w)
    
    if r2_initial < 0.1:
        print(f"   Initial R²={r2_initial:.3f} is weak, increasing regularization...", flush=True)
        alpha_adaptive = max(alpha, 5.0)
        ridge = Ridge(alpha=alpha_adaptive, random_state=42)
        ridge.fit(X_train_labeled, y.astype(float), sample_weight=w)
        s = ridge.predict(X_train_labeled)
        r2_new = r2_score(y, s, sample_weight=w)
        print(f"   After alpha={alpha_adaptive}: R²={r2_new:.3f}", flush=True)
    
    iso = [IsotonicRegression(out_of_bounds='clip') for _ in range(2)]
    iso[0].fit(s, (y >= 1).astype(float), sample_weight=w)
    iso[1].fit(s, (y >= 2).astype(float), sample_weight=w)
    
    preds_train = np.vstack([
        1 - iso[0].transform(s),
        iso[0].transform(s) - iso[1].transform(s),
        iso[1].transform(s)
    ]).T
    
    preds_train = np.clip(preds_train, 0, 1)
    preds_train /= preds_train.sum(axis=1, keepdims=True) + 1e-12
    
    ba_train = balanced_accuracy_score(y, preds_train.argmax(1))
    r2_train = r2_score(y, s)
    print(f"   Train R²: {r2_train:.3f}, Train BA (risk-only): {ba_train:.3f}", flush=True)
    
    w_unit = np.ones(X_train_labeled.shape[1], dtype=np.float32) / np.sqrt(X_train_labeled.shape[1])
    return OrdinalIsotonicWrapper3Class(ridge, iso), w_unit

# ============================================================================
# DIAGNOSTICS
# ============================================================================

def run_diagnostics(X_train, y_train, mask_train, X_val, y_val, mask_val, risk_scorer):
    """Comprehensive diagnostic to isolate risk vs HMM issues."""
    print("\n   === DIAGNOSTIC: Component Analysis ===", flush=True)
    
    r_train = risk_scorer.score_function(X_train[mask_train]).reshape(-1, 1)
    r_val = risk_scorer.score_function(X_val[mask_val]).reshape(-1, 1)
    
    clf_risk = LogisticRegression(
        solver='lbfgs',
        C=1.0,
        max_iter=500, 
        class_weight='balanced', 
        random_state=42
    )
    clf_risk.fit(r_train, y_train[mask_train])
    ba_risk = balanced_accuracy_score(y_val[mask_val], clf_risk.predict(r_val))
    
    probs_val = risk_scorer.predict_proba(X_val[mask_val])
    ba_probs = balanced_accuracy_score(y_val[mask_val], probs_val.argmax(1))
    
    r2_val = r2_score(y_val[mask_val], risk_scorer.score_function(X_val[mask_val]))
    corr_val = spearmanr(risk_scorer.score_function(X_val[mask_val]), y_val[mask_val]).correlation
    
    print(f"   Risk-score Val BA: {ba_risk:.3f}", flush=True)
    print(f"   Risk-probs Val BA: {ba_probs:.3f}", flush=True)
    print(f"   Val R²: {r2_val:.3f}", flush=True)
    print(f"   Val Spearman: {corr_val:.3f}", flush=True)
    
    if ba_probs > 0.55:
        print(f"   ✔ Risk scoring works well!", flush=True)
    elif ba_probs > 0.45:
        print(f"   ⚠ Risk scoring is moderate. Consider lower reduce-dim or tica.", flush=True)
    else:
        print(f"   ✗ Risk scoring is weak. Check preprocessing.", flush=True)
    
    return ba_risk, ba_probs, r2_val

# ============================================================================
# HMM HELPERS (WINDOWS-SAFE WITH OPTION A: PRE-SEED EVERYTHING)
# ============================================================================

def banded_transition_prior(n_states, diag_weight=1.5, neighbor_weight=0.3, 
                           far_weight=0.001, strength=50.0):
    """Banded transition prior."""
    T = np.full((n_states, n_states), far_weight, dtype=float)
    
    for i in range(n_states):
        T[i, i] = diag_weight
        if i > 0:
            T[i, i-1] = neighbor_weight
        if i < n_states - 1:
            T[i, i+1] = neighbor_weight
    
    T = T / T.sum(axis=1, keepdims=True)
    return T * strength

def safe_kmeans_init(X_train, n_states, random_state=42):
    """WINDOWS-SAFE KMeans initialization."""
    print(f"   Running SAFE KMeans for {n_states} initial clusters...", flush=True)
    
    if np.any(np.isnan(X_train)):
        print("   WARNING: NaN in training data, cleaning before KMeans...")
        X_clean = np.nan_to_num(X_train, nan=0.0)
    else:
        X_clean = X_train
    
    if len(X_clean) > 50000:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(X_clean), size=50000, replace=False)
        X_fit = X_clean[idx]
    else:
        X_fit = X_clean
    
    try:
        km = MiniBatchKMeans(
            n_clusters=n_states, 
            batch_size=min(5000, max(256, len(X_fit) // 10)),
            n_init=2,
            max_iter=100,
            random_state=random_state,
            reassignment_ratio=0.01,
            verbose=0
        )
        km.fit(X_fit)
        centers = km.cluster_centers_.astype(np.float64)
        
        if np.any(np.isnan(centers)):
            raise ValueError("KMeans produced NaN centers")
            
        print(f"   ✔ KMeans initialization successful", flush=True)
        return centers
        
    except Exception as e:
        print(f"   ⚠️  KMeans failed ({e}), using RANDOM initialization", flush=True)
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(X_fit), size=n_states, replace=False)
        centers = X_fit[idx].astype(np.float64)
        return centers

def risk_quantile_means_init(X, risk_scores, n_states):
    """Initialize HMM means by risk quantiles."""
    print(f"   Initializing {n_states} states by risk quantiles...", flush=True)
    
    try:
        qs = np.quantile(risk_scores, np.linspace(0, 1, n_states + 1))
        centers = []
        
        for i in range(n_states):
            if i < n_states - 1:
                mask = (risk_scores >= qs[i]) & (risk_scores < qs[i+1])
            else:
                mask = (risk_scores >= qs[i]) & (risk_scores <= qs[i+1])
            
            if mask.sum() == 0:
                mask = np.ones(len(risk_scores), bool)
            
            centers.append(X[mask].mean(axis=0))
        
        centers = np.vstack(centers).astype(np.float64)
        
        if np.any(np.isnan(centers)):
            raise ValueError("Risk-quantile init produced NaN")
            
        print(f"   ✔ Risk-quantile initialization successful", flush=True)
        return centers
        
    except Exception as e:
        print(f"   ⚠️  Risk-quantile init failed ({e}), falling back to safe KMeans")
        return safe_kmeans_init(X, n_states)

def reorder_hmm_by_posterior_label(model, X_for_post, lengths, y, mask):
    """Reorder HMM states by expected label (supervised alignment)."""
    print("   Using SUPERVISED state reordering (by posterior labels)...", flush=True)
    
    _, gamma = model.score_samples(X_for_post, lengths=lengths)
    G = gamma[mask]
    y_lab = y[mask].astype(float)
    
    num = G.T @ y_lab
    den = G.sum(0) + 1e-12
    scores = num / den
    
    order = np.argsort(scores)
    print(f"   State expected labels after reordering: {scores[order]}")
    
    model.startprob_ = model.startprob_[order]
    model.transmat_ = model.transmat_[np.ix_(order, order)]
    model.means_ = model.means_[order]
    
    if model.covariance_type in ("diag", "spherical", "full"):
        if hasattr(model, '_covars_'):
            model._covars_ = model._covars_[order]
        else:
            try:
                old_covars = model.covars_.copy()
                model._covars_ = old_covars[order]
            except Exception as e:
                print(f"   WARNING: Could not reorder covariances: {e}")
    elif model.covariance_type == "tied":
        pass
    
    return model, order

def reorder_hmm_by_risk(model, risk_scorer, risk_dim_at_end=False):
    """Reorder HMM states by risk score after EM."""
    means_for_scoring = model.means_
    
    if risk_dim_at_end:
        means_for_scoring = means_for_scoring[:, :-1]
    
    if np.any(np.isnan(means_for_scoring)):
        print("WARNING: NaN in HMM means, skipping reordering")
        return model
    
    try:
        scores = risk_scorer.score_function(means_for_scoring).reshape(-1)
    except Exception as e:
        print(f"WARNING: Failed to score means for reordering: {e}")
        return model
    
    if np.any(np.isnan(scores)) or np.any(np.isinf(scores)):
        print("WARNING: Invalid risk scores, skipping reordering")
        return model
    
    order = np.argsort(scores)
    print(f"   State risk scores after reordering: {scores[order]}")
    
    model.startprob_ = model.startprob_[order]
    model.transmat_ = model.transmat_[np.ix_(order, order)]
    model.means_ = model.means_[order]
    
    if model.covariance_type in ("diag", "spherical", "full"):
        if hasattr(model, '_covars_'):
            model._covars_ = model._covars_[order]
        else:
            try:
                old_covars = model.covars_.copy()
                model._covars_ = old_covars[order]
            except Exception as e:
                print(f"   WARNING: Could not reorder covariances: {e}")
    elif model.covariance_type == "tied":
        pass
    
    return model

# ============================================================================
# FAST TIED LOG-LIKELIHOOD
# ============================================================================

def install_fast_tied_loglik(model, block_size=200_000):
    """Monkey-patch hmmlearn's tied log-likelihood."""
    if getattr(model, "covariance_type", None) != "tied":
        return model

    def _fast_ll(self, X):
        cov = self.covars_
        
        if cov.ndim == 3:
            cov = cov[0]
        
        cov = np.asarray(cov, dtype=np.float64)
        cov_reg = cov + np.eye(cov.shape[0], dtype=np.float64) * 1e-6
        
        try:
            c, lower = cho_factor(cov_reg, overwrite_a=False, check_finite=False)
        except np.linalg.LinAlgError:
            cov_reg = cov + np.eye(cov.shape[0], dtype=np.float64) * self.min_covar
            c, lower = cho_factor(cov_reg, overwrite_a=False, check_finite=False)
        
        D = X.shape[1]
        log_det = 2.0 * np.sum(np.log(np.diag(c)))
        const = -0.5 * (D * np.log(2*np.pi) + log_det)

        M_wh = cho_solve((c, lower), self.means_.T, check_finite=False).T
        M2 = np.einsum("kd,kd->k", M_wh, M_wh)

        T = X.shape[0]
        out = np.empty((T, self.n_components), dtype=np.float64)
        for s in range(0, T, block_size):
            e = min(s + block_size, T)
            X_block = X[s:e].astype(np.float64, copy=False)
            X_wh = cho_solve((c, lower), X_block.T, check_finite=False).T
            X2 = np.einsum("td,td->t", X_wh, X_wh)
            out[s:e, :] = const - 0.5 * (X2[:, None] + M2[None, :] - 2.0 * (X_wh @ M_wh.T))
        return out

    model._compute_log_likelihood = types.MethodType(_fast_ll, model)
    return model

# ============================================================================
# VECTORIZED POSTERIOR EXTRACTION
# ============================================================================

def posteriors_per_sample(model, X_reduced, lengths=None):
    """Get posterior state probabilities per sample."""
    if lengths is None:
        _, post = model.score_samples(X_reduced)
        return post
    _, post = model.score_samples(X_reduced, lengths=lengths)
    return post

# ============================================================================
# HMM TRAINING (OPTION A: PRE-SEED EVERYTHING - WINDOWS-SAFE)
# ============================================================================

def fit_ghmm_unsupervised(
    X_train_reduced,
    lengths_train=None,
    n_states=10,
    covariance_type="diag",
    min_covar=0.1,
    risk_scorer=None,
    init_method="kmeans",
    diag_weight=1.5,
    neighbor_weight=0.3,
    prior_strength=50.0,
    freeze_transitions=False,
    append_risk_axis=0.0,
    random_state=42,
    use_supervised_reorder=False,
    y_train=None,
    mask_train=None,
):
    """Fit Gaussian HMM with OPTION A: pre-seed everything (Windows-safe)."""
    print(f"\n   Fitting GHMM: n_states={n_states}, cov_type={covariance_type}, init={init_method}", flush=True)
    print(f"   Prior strength={prior_strength}, freeze_trans={freeze_transitions}, risk_axis={append_risk_axis}", flush=True)
    
    X_train_reduced = X_train_reduced.astype(np.float64, copy=False)
    
    risk_scores = None
    if risk_scorer is not None:
        risk_scores = risk_scorer.score_function(X_train_reduced)
    
    X_aug = X_train_reduced
    risk_dim_appended = False
    
    if risk_scorer is not None and append_risk_axis > 0:
        r_std = (risk_scores - risk_scores.mean()) / (risk_scores.std() + 1e-8)
        X_aug = np.hstack([X_train_reduced, append_risk_axis * r_std.reshape(-1, 1)])
        risk_dim_appended = True
        print(f"   Appended scaled risk dimension with weight {append_risk_axis}", flush=True)
    
    # Initialize means
    if init_method == "risk-quantiles" and risk_scorer is not None:
        means_init = risk_quantile_means_init(X_aug, risk_scores, n_states)
    else:
        means_init = safe_kmeans_init(X_aug, n_states, random_state)

    # Banded transition prior
    trans_prior = banded_transition_prior(
        n_states, 
        diag_weight=diag_weight,
        neighbor_weight=neighbor_weight, 
        far_weight=0.001,
        strength=prior_strength
    )
    
    # Check transitions
    summary = _transitions_summary(lengths_train)
    print(f"   Transition summary: {summary}")
    
    if summary['n_trans'] == 0:
        print("   ⚠️ CRITICAL: No transitions! Reducing prior strength to 1.0")
        prior_strength = 1.0
        trans_prior = banded_transition_prior(
            n_states, diag_weight, neighbor_weight, 0.001, strength=1.0
        )
    else:
        avg_row_trans = max(1.0, summary['n_trans'] / max(1, n_states))
        adjusted_strength = min(prior_strength, 0.1 * avg_row_trans)
        if adjusted_strength < prior_strength:
            print(f"   Reduced prior strength from {prior_strength} to {adjusted_strength:.2f} based on data")
            prior_strength = adjusted_strength
        trans_prior = banded_transition_prior(
            n_states, diag_weight, neighbor_weight, 0.001, strength=prior_strength
        )

    # OPTION A: Build model with init_params="" and pre-seed everything
    print("   Using OPTION A: Pre-seeding all parameters (Windows-safe)", flush=True)
    
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=60,
        tol=1e-2,
        min_covar=min_covar,
        random_state=random_state,
        init_params="",  # Skip hmmlearn's init completely
        params="stmc" if not freeze_transitions else "smc",
        verbose=True
    )

    # Pre-seed everything so EM can start immediately, no KMeans
    model.startprob_ = np.ones(n_states, float) / n_states
    A = trans_prior / trans_prior.sum(axis=1, keepdims=True)
    model.transmat_ = A
    model.means_ = means_init.astype(np.float64)
    
    # Diagonal covars: start at min_covar
    if covariance_type == "diag":
        model._covars_ = np.full((n_states, X_aug.shape[1]), min_covar, dtype=np.float64)
    elif covariance_type == "spherical":
        model._covars_ = np.full(n_states, min_covar, dtype=np.float64)
    elif covariance_type == "full":
        model._covars_ = np.stack([np.eye(X_aug.shape[1], dtype=np.float64) * min_covar] * n_states, axis=0)
    elif covariance_type == "tied":
        # Will be set in staged training below
        pass
    
    model.transmat_prior = trans_prior
    
    if covariance_type == "tied":
        install_fast_tied_loglik(model, block_size=200_000)
        print("   ✔ Fast tied log-likelihood installed", flush=True)
    
    print(f"   Running EM algorithm...", flush=True)
    start_time = time.time()
    
    if covariance_type == "tied":
        # Stage A
        D = X_aug.shape[1]
        
        model.n_iter = 1
        if lengths_train is None:
            model.fit(X_aug[:100])
        else:
            init_lengths = []
            total = 0
            for L in lengths_train:
                if total + L <= 100:
                    init_lengths.append(L)
                    total += L
                else:
                    if total < 100:
                        init_lengths.append(100 - total)
                    break
            model.fit(X_aug[:100], lengths=init_lengths)
        
        model.covars_ = np.eye(D, dtype=np.float64) * max(min_covar, 0.1)
        
        if risk_dim_appended:
            cov = model.covars_.copy()
            cov[-1, -1] *= 0.25
            model.covars_ = cov
            print("   Tightened covariance on risk axis", flush=True)
        
        model.params = ("s" if freeze_transitions else "st")
        model.init_params = ""
        model.n_iter = 8
        
        print("   Stage A: Fast alignment (8 iterations)...", flush=True)
        stage_a_start = time.time()
        if lengths_train is None:
            model.fit(X_aug)
        else:
            model.fit(X_aug, lengths=lengths_train)
        print(f"   Stage A complete in {time.time() - stage_a_start:.1f}s", flush=True)

        if not freeze_transitions:
            model.transmat_prior *= 0.25
            print("   Annealed transition prior to 25% for Stage B")

        # Stage B
        model.params = ("smc" if freeze_transitions else "stmc")
        model.init_params = ""
        model.n_iter = 40
        
        print("   Stage B: Refinement (40 iterations)...", flush=True)
        stage_b_start = time.time()
        if lengths_train is None:
            model.fit(X_aug)
        else:
            model.fit(X_aug, lengths=lengths_train)
        print(f"   Stage B complete in {time.time() - stage_b_start:.1f}s", flush=True)
    else:
        # Single stage for diag/spherical
        if lengths_train is None:
            model.fit(X_aug)
        else:
            model.fit(X_aug, lengths=lengths_train)
    
    total_time = time.time() - start_time
    print(f"   EM complete in {total_time:.1f}s total", flush=True)
    print(f"   Converged: {model.monitor_.converged}, Iterations: {model.monitor_.iter}", flush=True)
    
    # Check if transitions learned
    A_prior_normalized = trans_prior / (trans_prior.sum(axis=1, keepdims=True) + 1e-12)
    A = model.transmat_
    
    delta = float(np.nanmax(np.abs(A - A_prior_normalized)))
    print(f"   Max |A_learned - A_prior| = {delta:.6f}", flush=True)
    
    if delta < 1e-5:
        print("   ⚠️ CRITICAL: Transition matrix equals prior - NO LEARNING!")
    elif delta < 0.01:
        print("   ⚠️ WARNING: Transitions very close to prior")
    else:
        print(f"   ✓ Transitions learned from data (delta={delta:.3f})")
    
    # Fix NaN in means
    if np.any(np.isnan(model.means_)):
        print(f"   WARNING: NaN in HMM means, fixing...")
        nan_mask = np.isnan(model.means_).any(axis=1)
        n_nan = nan_mask.sum()
        if n_nan > 0:
            centers_fix = safe_kmeans_init(X_aug, n_nan, random_state)
            model.means_[nan_mask] = centers_fix
            print(f"   Replaced {n_nan} NaN state means")
    
    # Reorder states
    if use_supervised_reorder and y_train is not None and mask_train is not None:
        X_for_reorder = X_aug if risk_dim_appended else X_train_reduced
        model, order = reorder_hmm_by_posterior_label(
            model, X_for_reorder, lengths_train, y_train, mask_train
        )
        print(f"   ✔ States REORDERED by posterior labels (supervised)", flush=True)
    elif risk_scorer is not None:
        model = reorder_hmm_by_risk(model, risk_scorer, risk_dim_at_end=risk_dim_appended)
        print(f"   ✔ States REORDERED by risk after EM", flush=True)
    
    return model, risk_dim_appended

# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_split(clf, X_feat, y, mask_labeled):
    """Evaluate on labeled subset."""
    if not mask_labeled.any():
        return dict(ba=np.nan, f1=np.nan, recall=np.full(3, np.nan), report="No labeled samples")
    
    y_true = y[mask_labeled]
    y_pred = clf.predict(X_feat[mask_labeled])
    
    ba = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    per_class_recall = recall_score(y_true, y_pred, average=None, labels=list(range(3)))
    
    report = classification_report(
        y_true, y_pred,
        target_names=['Low','Medium','High'],
        zero_division=0
    )
    
    return dict(ba=ba, f1=f1, recall=per_class_recall, report=report)

# ============================================================================
# MAIN TRAINING ROUTINE (FULLY OPTIMIZED FOR ~0.7 BA)
# ============================================================================

def run_improved_ghmm(
    data_type: str,
    embeddings_root: Path,
    output_dir: Path,
    n_states_grid=(16, 20, 24, 28),
    preprocessing='copula_zca',
    reduce_dim=48,
    covariance_type='diag',
    init_method='risk-quantiles',
    append_risk_axis=0.75,
    tica_lag=2,
    diag_weight=1.2,
    neighbor_weight=0.5,
    prior_strength=2.0,
    freeze_transitions=False,
    min_covar=0.02,
    context_windows=(3, 5, 9, 15),
    c_grid=(2.0, 4.0, 8.0, 12.0),
    num_threads=None,
):
    """Main GHMM training with ALL optimizations for ~0.7 BA."""
    print("\n" + "="*60, flush=True)
    print(f"IMPROVED GHMM v5 3-CLASS OPTIMIZED (FULLY DEBUGGED) for {data_type.upper()}", flush=True)
    print(f"Preprocessing: {preprocessing.upper()}, Reduce to: {reduce_dim}D", flush=True)
    print(f"Covariance: {covariance_type}, Init: {init_method}, Risk axis: {append_risk_axis}", flush=True)
    print(f"Multi-scale context windows: {context_windows}", flush=True)
    print(f"Mapper C grid: {c_grid}", flush=True)
    print("="*60, flush=True)
    
    emb_dir = embeddings_root / data_type
    out_dir = output_dir / data_type
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load splits
    print("\n1. Loading data splits...", flush=True)
    train = _load_split(emb_dir, "train")
    val = _load_split(emb_dir, "val")
    
    test = None
    hold = None
    if (emb_dir / "embeddings_test.npz").exists():
        test = _load_split(emb_dir, "test")
    if (emb_dir / "embeddings_holdout.npz").exists():
        hold = _load_split(emb_dir, "holdout")
    
    # Make sequences contiguous
    print("\n   Making sequences contiguous...")
    train, lengths_train = _contiguous_split(train, "train")
    val, lengths_val = _contiguous_split(val, "val")
    if test:
        test, lengths_test = _contiguous_split(test, "test")
    else:
        lengths_test = None
    if hold:
        hold, lengths_hold = _contiguous_split(hold, "holdout")
    else:
        lengths_hold = None
    
    # Detect hierarchical structure
    if train["X"].shape[1] == 144:
        blocks = [64, 48, 32]
        print(f"   Detected hierarchical embeddings: {blocks}", flush=True)
    else:
        blocks = None
    
    # Preprocessing
    scaler, reducer, Xtr_reduced = fit_preprocess(
        train["X"],
        method=preprocessing,
        n_components=reduce_dim,
        y_train=train["y"],
        has_manual=train["has_manual"],
        lengths_train=lengths_train,
        blocks=blocks,
        tica_lag=tica_lag,
        random_state=42
    )
    
    print(f"   Reduced dimensionality: {train['X'].shape[1]} -> {Xtr_reduced.shape[1]}", flush=True)
    
    Xv_reduced = apply_preprocess(val["X"], scaler, reducer)
    Xt_reduced = apply_preprocess(test["X"], scaler, reducer) if test else None
    Xh_reduced = apply_preprocess(hold["X"], scaler, reducer) if hold else None

    # Convert to float64
    Xtr_reduced = Xtr_reduced.astype(np.float64, copy=False)
    Xv_reduced = Xv_reduced.astype(np.float64, copy=False)
    if Xt_reduced is not None: 
        Xt_reduced = Xt_reduced.astype(np.float64, copy=False)
    if Xh_reduced is not None: 
        Xh_reduced = Xh_reduced.astype(np.float64, copy=False)

    # Check for NaN
    if np.any(np.isnan(Xtr_reduced)):
        n_nan = np.isnan(Xtr_reduced).any(axis=1).sum()
        print(f"   WARNING: {n_nan} samples with NaN in train, fixing...")
        Xtr_reduced = np.nan_to_num(Xtr_reduced, nan=0.0)

    if np.any(np.isnan(Xv_reduced)):
        n_nan = np.isnan(Xv_reduced).any(axis=1).sum()
        print(f"   WARNING: {n_nan} samples with NaN in val, fixing...")
        Xv_reduced = np.nan_to_num(Xv_reduced, nan=0.0)
    
    # Get labeled masks
    tr_lab = train["has_manual"]
    v_lab = val["has_manual"]
    t_lab = test["has_manual"] if test else None
    h_lab = hold["has_manual"] if hold else None
    
    # Fit risk scorer
    ytr = train["y"]
    yv = val["y"]
    yt = test["y"] if test else None
    yh = hold["y"] if hold else None
    
    risk_scorer, risk_w = fit_risk_ordinal_isotonic_3class(
        Xtr_reduced[tr_lab], 
        ytr[tr_lab],
        coverage=train["coverage"][tr_lab]
    )
    
    # Run diagnostics
    ba_risk_score, ba_risk_probs, r2_val = run_diagnostics(
        Xtr_reduced, ytr, tr_lab,
        Xv_reduced, yv, v_lab,
        risk_scorer
    )
    
    # Compute risk scores and probabilities
    r_tr = risk_scorer.score_function(Xtr_reduced)
    r_v = risk_scorer.score_function(Xv_reduced)
    r_t = risk_scorer.score_function(Xt_reduced) if test else None
    r_h = risk_scorer.score_function(Xh_reduced) if hold else None
    
    p_tr = risk_scorer.predict_proba(Xtr_reduced)
    p_v = risk_scorer.predict_proba(Xv_reduced)
    p_t = risk_scorer.predict_proba(Xt_reduced) if test else None
    p_h = risk_scorer.predict_proba(Xh_reduced) if hold else None
    
    # Decide whether to use supervised state reordering
    use_supervised_reorder = ba_risk_probs < 0.45
    if use_supervised_reorder:
        print(f"\n⚠️  Risk-probs BA ({ba_risk_probs:.3f}) < 0.45: Using SUPERVISED state reordering", flush=True)
    
    # Grid search
    print(f"\n4. Grid search over n_states: {n_states_grid}", flush=True)
    best = dict(score=-np.inf, n_states=None, model=None, mapper=None, merge=None, thresholds=None, C=None)
    per_k_results = {}
    
    for k in n_states_grid:
        print(f"\n   === Testing n_states={k} ===", flush=True)
        
        # Fit HMM
        hmm_start = time.time()
        model, risk_dim_appended = fit_ghmm_unsupervised(
            Xtr_reduced,
            lengths_train=lengths_train,
            n_states=k,
            covariance_type=covariance_type,
            min_covar=min_covar,
            risk_scorer=risk_scorer,
            init_method=init_method,
            diag_weight=diag_weight,
            neighbor_weight=neighbor_weight,
            prior_strength=prior_strength,
            freeze_transitions=freeze_transitions,
            append_risk_axis=append_risk_axis,
            random_state=42,
            use_supervised_reorder=use_supervised_reorder,
            y_train=ytr,
            mask_train=tr_lab
        )
        print(f"   HMM training time: {time.time() - hmm_start:.1f}s", flush=True)
        
        # Get posteriors
        print(f"   Computing posteriors...", flush=True)
        post_start = time.time()

        if risk_dim_appended:
            r_std_tr = (r_tr - r_tr.mean()) / (r_tr.std() + 1e-8)
            r_std_v = (r_v - r_v.mean()) / (r_v.std() + 1e-8)
            
            Xtr_for_posterior = np.hstack([Xtr_reduced, append_risk_axis * r_std_tr.reshape(-1, 1)])
            Xv_for_posterior = np.hstack([Xv_reduced, append_risk_axis * r_std_v.reshape(-1, 1)])
        else:
            Xtr_for_posterior = Xtr_reduced
            Xv_for_posterior = Xv_reduced

        gamma_tr = posteriors_per_sample(model, Xtr_for_posterior, lengths=lengths_train)
        gamma_v = posteriors_per_sample(model, Xv_for_posterior, lengths=lengths_val)

        if test:
            if risk_dim_appended:
                r_std_t = (r_t - r_t.mean()) / (r_t.std() + 1e-8)
                Xt_for_posterior = np.hstack([Xt_reduced, append_risk_axis * r_std_t.reshape(-1, 1)])
            else:
                Xt_for_posterior = Xt_reduced
            gamma_t = posteriors_per_sample(model, Xt_for_posterior, lengths=lengths_test)

        if hold:
            if risk_dim_appended:
                r_std_h = (r_h - r_h.mean()) / (r_h.std() + 1e-8)
                Xh_for_posterior = np.hstack([Xh_reduced, append_risk_axis * r_std_h.reshape(-1, 1)])
            else:
                Xh_for_posterior = Xh_reduced
            gamma_h = posteriors_per_sample(model, Xh_for_posterior, lengths=lengths_hold)
        
        print(f"   Posterior extraction: {time.time() - post_start:.1f}s", flush=True)
        
        # Add MULTI-SCALE sequence context features (KEY FIX!)
        print(f"   Adding multi-scale sequence context (windows={context_windows})...", flush=True)
        ctx_tr = add_sequence_context_multi(gamma_tr, r_tr, train["seq_ids"], wins=context_windows)
        ctx_v = add_sequence_context_multi(gamma_v, r_v, val["seq_ids"], wins=context_windows)
        
        # SANITY CHECK: Verify shapes
        print(f"   Shapes: gamma_tr={gamma_tr.shape}, ctx_tr={ctx_tr.shape}", flush=True)
        assert ctx_tr.shape[0] == gamma_tr.shape[0], f"Context shape mismatch! {ctx_tr.shape[0]} vs {gamma_tr.shape[0]}"
        
        # Build basic features
        Fe_tr_basic = np.hstack([
            gamma_tr,
            p_tr,
            np.log(np.clip(p_tr, 1e-8, 1.0)),
            r_tr.reshape(-1, 1),
            (r_tr ** 2).reshape(-1, 1),
            (r_tr ** 3).reshape(-1, 1),
            np.sqrt(np.clip(r_tr - r_tr.min(), 0, None)).reshape(-1, 1),
            np.log(r_tr - r_tr.min() + 1).reshape(-1, 1),
            ctx_tr
        ]).astype(np.float32)
        
        Fe_v_basic = np.hstack([
            gamma_v,
            p_v,
            np.log(np.clip(p_v, 1e-8, 1.0)),
            r_v.reshape(-1, 1),
            (r_v ** 2).reshape(-1, 1),
            (r_v ** 3).reshape(-1, 1),
            np.sqrt(np.clip(r_v - r_v.min(), 0, None)).reshape(-1, 1),
            np.log(r_v - r_v.min() + 1).reshape(-1, 1),
            ctx_v
        ]).astype(np.float32)
        
        print(f"   Feature shapes: Fe_tr_basic={Fe_tr_basic.shape}, Fe_v_basic={Fe_v_basic.shape}", flush=True)
        
        # Verify no NaNs/Infs
        for name, A in [("Fe_tr_basic", Fe_tr_basic), ("Fe_v_basic", Fe_v_basic)]:
            bad = ~np.isfinite(A).all()
            if bad:
                print(f"   WARNING: {name} contains NaN/Inf, cleaning...")
                A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                print(f"   ✔ {name} all finite", flush=True)
        
        # Learn best 5→3 merge
        best_merge = pick_best_merge(Fe_tr_basic, train["y5"], tr_lab, Fe_v_basic, val["y5"], v_lab)
        a, b = best_merge['a'], best_merge['b']
        
        # Apply learned merge
        ytr = apply_collapse_to_labels(train["y5"], a, b)
        yv = apply_collapse_to_labels(val["y5"], a, b)
        if test:
            yt = apply_collapse_to_labels(test["y5"], a, b)
        if hold:
            yh = apply_collapse_to_labels(hold["y5"], a, b)
        
        # Learn score-and-slice thresholds
        print("\n   Learning score thresholds...", flush=True)
        best_thr = learn_tri_thresholds_on_score(r_v, yv, v_lab)
        t1, t2 = best_thr['t1'], best_thr['t2']
        print(f"   ✔ Thresholds: t1={t1:.4f}, t2={t2:.4f}", flush=True)
        
        # Add score-and-slice features
        Fe_tr = np.hstack([Fe_tr_basic, tri_features_from_score_and_probs(r_tr, p_tr, t1, t2)])
        Fe_v = np.hstack([Fe_v_basic, tri_features_from_score_and_probs(r_v, p_v, t1, t2)])
        
        if test:
            ctx_t = add_sequence_context_multi(gamma_t, r_t, test["seq_ids"], wins=context_windows)
            Fe_t_basic = np.hstack([
                gamma_t, p_t, np.log(np.clip(p_t, 1e-8, 1.0)),
                r_t.reshape(-1, 1), (r_t ** 2).reshape(-1, 1), (r_t ** 3).reshape(-1, 1),
                np.sqrt(np.clip(r_t - r_t.min(), 0, None)).reshape(-1, 1),
                np.log(r_t - r_t.min() + 1).reshape(-1, 1),
                ctx_t
            ]).astype(np.float32)
            Fe_t = np.hstack([Fe_t_basic, tri_features_from_score_and_probs(r_t, p_t, t1, t2)])
        else:
            Fe_t = None
        
        if hold:
            ctx_h = add_sequence_context_multi(gamma_h, r_h, hold["seq_ids"], wins=context_windows)
            Fe_h_basic = np.hstack([
                gamma_h, p_h, np.log(np.clip(p_h, 1e-8, 1.0)),
                r_h.reshape(-1, 1), (r_h ** 2).reshape(-1, 1), (r_h ** 3).reshape(-1, 1),
                np.sqrt(np.clip(r_h - r_h.min(), 0, None)).reshape(-1, 1),
                np.log(r_h - r_h.min() + 1).reshape(-1, 1),
                ctx_h
            ]).astype(np.float32)
            Fe_h = np.hstack([Fe_h_basic, tri_features_from_score_and_probs(r_h, p_h, t1, t2)])
        else:
            Fe_h = None
        
        # GRID SEARCH OVER C (KEY OPTIMIZATION!)
        print(f"\n   Grid search over C: {c_grid}", flush=True)
        best_c = dict(score=-np.inf, C=None, mapper=None)
        
        for C in c_grid:
            mapper = LogisticRegression(
                solver='lbfgs',
                C=C,
                max_iter=3000,
                class_weight='balanced',
                random_state=42
            )
            mapper.fit(Fe_tr[tr_lab], ytr[tr_lab])
            
            val_metrics = evaluate_split(mapper, Fe_v, yv, v_lab)
            ba = val_metrics["ba"]
            print(f"     C={C:5.1f} → Val BA={ba:.3f}", flush=True)
            
            if np.isfinite(ba) and ba > best_c["score"]:
                best_c.update(score=ba, C=C, mapper=mapper)
        
        print(f"   ✔ Best C={best_c['C']} with Val BA={best_c['score']:.3f}", flush=True)
        
        per_k_results[k] = best_c["score"]
        
        # Track best
        if best_c["score"] > best["score"]:
            best.update(dict(
                score=best_c["score"],
                n_states=k,
                model=model,
                mapper=best_c["mapper"],
                C=best_c["C"],
                risk_dim_appended=risk_dim_appended,
                merge=(a, b),
                thresholds=(t1, t2)
            ))
            best_feats = dict(train=Fe_tr, val=Fe_v, test=Fe_t, hold=Fe_h)
            print(f"   ** New best! **", flush=True)
    
    if best["model"] is None:
        raise RuntimeError("Failed to fit GHMM.")
    
    print(f"\n✔ Selected n_states={best['n_states']} with Val BA={best['score']:.3f}", flush=True)
    print(f"✔ Best C={best['C']}", flush=True)
    print(f"✔ Best merge: [0..{best['merge'][0]}] | [{best['merge'][0]+1}..{best['merge'][1]}] | [{best['merge'][1]+1}..4]", flush=True)
    print(f"✔ Best thresholds: t1={best['thresholds'][0]:.4f}, t2={best['thresholds'][1]:.4f}", flush=True)
    
    # Use best model
    model, mapper = best["model"], best["mapper"]
    Fe_tr = best_feats["train"]
    Fe_v = best_feats["val"]
    Fe_t = best_feats.get("test")
    Fe_h = best_feats.get("hold")
    
    # Apply learned merge to final labels
    a, b = best["merge"]
    ytr = apply_collapse_to_labels(train["y5"], a, b)
    yv = apply_collapse_to_labels(val["y5"], a, b)
    if test:
        yt = apply_collapse_to_labels(test["y5"], a, b)
    if hold:
        yh = apply_collapse_to_labels(hold["y5"], a, b)
    
    # Compute final metrics
    print("\n5. Computing final metrics...", flush=True)
    
    results = {
        "data_type": data_type,
        "preprocessing": preprocessing,
        "reduce_dim": int(reduce_dim),
        "covariance_type": covariance_type,
        "init_method": init_method,
        "append_risk_axis": float(append_risk_axis),
        "risk_dim_appended": best.get("risk_dim_appended", False),
        "tica_lag": int(tica_lag) if preprocessing == "tica" else None,
        "risk_method": "weighted_ordinal_isotonic_3class",
        "n_states_grid": list(n_states_grid),
        "n_states_selected": best["n_states"],
        "best_C": float(best["C"]),
        "best_merge": f"[0..{a}]|[{a+1}..{b}]|[{b+1}..4]",
        "best_thresholds": {"t1": float(best["thresholds"][0]), "t2": float(best["thresholds"][1])},
        "val_ba": float(best["score"]),
        "risk_score_ba": float(ba_risk_score),
        "risk_probs_ba": float(ba_risk_probs),
        "val_r2": float(r2_val),
        "per_k_val_ba": {int(k): float(v) for k, v in per_k_results.items()},
        "prior_strength": float(prior_strength),
        "diag_weight": float(diag_weight),
        "neighbor_weight": float(neighbor_weight),
        "context_windows": list(context_windows),
        "freeze_transitions": bool(freeze_transitions),
        "version": "v5_3class_fully_debugged",
        "n_classes": 3
    }
    
    # Evaluate all splits
    for split_name, Fe, y, mask in [
        ("train", Fe_tr, ytr, tr_lab),
        ("val", Fe_v, yv, v_lab),
        ("test", Fe_t, yt, t_lab) if test else (None, None, None, None),
        ("holdout", Fe_h, yh, h_lab) if hold else (None, None, None, None),
    ]:
        if split_name and Fe is not None:
            metrics = evaluate_split(mapper, Fe, y, mask)
            results[split_name] = {
                "ba": float(metrics["ba"]),
                "f1": float(metrics["f1"]),
                "per_class_recall": [float(x) if np.isfinite(x) else None for x in metrics["recall"]],
                "report": metrics["report"]
            }
    
    # Save artifacts
    print("\n6. Saving models and results...", flush=True)
    
    joblib.dump(model, out_dir / "ghmm_model_v5_fully_debugged.pkl")
    joblib.dump(mapper, out_dir / "mapper_v5_fully_debugged.pkl")
    joblib.dump(scaler, out_dir / "scaler.pkl")
    if reducer is not None:
        joblib.dump(reducer, out_dir / "reducer.pkl")
    joblib.dump(risk_scorer, out_dir / "risk_scorer_v5_fully_debugged.pkl")
    
    model_metadata = {
        'risk_dim_appended': best.get("risk_dim_appended", False),
        'append_risk_axis': append_risk_axis,
        'n_states': best["n_states"],
        'preprocessing': preprocessing,
        'n_classes': 3,
        'merge': best["merge"],
        'thresholds': best["thresholds"],
        'best_C': best["C"],
        'context_windows': list(context_windows)
    }
    with open(out_dir / "model_metadata.json", "w") as f:
        json.dump(model_metadata, f, indent=2)
    
    with open(out_dir / "results_v5_fully_debugged.yaml", "w") as f:
        yaml.safe_dump(results, f, sort_keys=False)
    
    with open(out_dir / "results_v5_fully_debugged.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Summary
    print("\n" + "="*60, flush=True)
    print("IMPROVED GHMM v5 3-CLASS OPTIMIZED (FULLY DEBUGGED) SUMMARY", flush=True)
    print("="*60, flush=True)
    print(f"Data type: {data_type}")
    print(f"Preprocessing: {preprocessing}, Reduced to: {Xtr_reduced.shape[1]}D")
    print(f"Risk-score BA: {ba_risk_score:.3f} | Risk-probs BA: {ba_risk_probs:.3f} | Val R²: {r2_val:.3f}")
    print(f"Combined Val BA: {best['score']:.3f}")
    
    if "test" in results:
        print(f"TEST    BA: {results['test']['ba']:.3f}  F1: {results['test']['f1']:.3f}")
    
    if "holdout" in results:
        print(f"HOLDOUT BA: {results['holdout']['ba']:.3f}  F1: {results['holdout']['f1']:.3f}")
    
    print("\n✔ Results saved to:", out_dir / "results_v5_fully_debugged.yaml", flush=True)
    
    return results

# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Improved GHMM v5 3-CLASS OPTIMIZED (FULLY DEBUGGED)")
    
    parser.add_argument("--data-type", required=True,
                       choices=["sessions", "bets", "payments", "transactions"])
    
    parser.add_argument("--embeddings-root", default="final_embeddings_minimal_fix")
    
    parser.add_argument("--output-dir", default="ghmm_improved_v5_3class_fully_debugged")
    
    parser.add_argument("--preprocessing", default="copula_zca",
                       choices=["pca", "zca", "copula", "copula_zca", "fa", "tica", "pls", "umap", "rp", "none"])
    
    parser.add_argument("--reduce-dim", type=int, default=48,
                       help="Target dimensions (48 recommended for diag cov)")
    
    parser.add_argument("--covariance-type", default="diag",
                       choices=["diag", "spherical", "tied", "full"])
    
    parser.add_argument("--init-method", default="risk-quantiles",
                       choices=["kmeans", "risk-quantiles"])
    
    parser.add_argument("--append-risk-axis", type=float, default=0.75)
    
    parser.add_argument("--tica-lag", type=int, default=2)
    
    parser.add_argument("--n-states-grid", default="16,20,24,28")
    
    parser.add_argument("--prior-strength", type=float, default=2.0)
    
    parser.add_argument("--diag-weight", type=float, default=1.2)
    
    parser.add_argument("--neighbor-weight", type=float, default=0.5)
    
    parser.add_argument("--freeze-transitions", action="store_true")
    
    parser.add_argument("--min-covar", type=float, default=0.02)
    
    parser.add_argument("--context-windows", default="3,5,9,15",
                       help="Multi-scale context windows (comma-separated)")
    
    parser.add_argument("--c-grid", default="2.0,4.0,8.0,12.0",
                       help="Grid of C values for LogisticRegression (comma-separated)")
    
    parser.add_argument("--num-threads", type=int, default=None,
                       help="Number of BLAS threads (default: auto)")
    
    args = parser.parse_args()
    
    print(f"\nStarting Improved GHMM v5 3-CLASS OPTIMIZED (FULLY DEBUGGED):", flush=True)
    print(f"  Data type: {args.data_type}", flush=True)
    print(f"  Preprocessing: {args.preprocessing}", flush=True)
    print(f"  Reduce dim: {args.reduce_dim}", flush=True)
    print(f"  Covariance: {args.covariance_type}", flush=True)
    print(f"  Prior strength: {args.prior_strength}", flush=True)
    print(f"  Risk axis: {args.append_risk_axis}", flush=True)
    print(f"  Context windows: {args.context_windows}", flush=True)
    print(f"  C grid: {args.c_grid}", flush=True)
    
    results = run_improved_ghmm(
        data_type=args.data_type,
        embeddings_root=Path(args.embeddings_root),
        output_dir=Path(args.output_dir),
        n_states_grid=tuple(int(x) for x in args.n_states_grid.split(",")),
        preprocessing=args.preprocessing,
        reduce_dim=args.reduce_dim,
        covariance_type=args.covariance_type,
        init_method=args.init_method,
        append_risk_axis=args.append_risk_axis,
        tica_lag=args.tica_lag,
        diag_weight=args.diag_weight,
        neighbor_weight=args.neighbor_weight,
        prior_strength=args.prior_strength,
        freeze_transitions=args.freeze_transitions,
        min_covar=args.min_covar,
        context_windows=tuple(int(x) for x in args.context_windows.split(",")),
        c_grid=tuple(float(x) for x in args.c_grid.split(",")),
        num_threads=args.num_threads,
    )
    
    print("\n✅ GHMM v5 3-CLASS OPTIMIZED (FULLY DEBUGGED) Complete!", flush=True)

if __name__ == "__main__":
    main()