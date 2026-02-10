#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BALI for Bets Dataset v3 - Better Risk Models
==============================================

V3 CHANGES:
- Multiple classifier options for risk model (RF, XGBoost, MLP, Ensemble)
- Auto-selection based on validation BA
- Calibrated probability outputs

Usage:
    python bali_for_bets_v3.py --emb-root final_embeddings_minimal_fix_seq_id_nov_10 --data-type bets --risk-model auto
"""

import numpy as np
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV

# Try to import XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("Note: XGBoost not available. Install with: pip install xgboost")

# Try to import LightGBM
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


class BacklogEstimator:
    """Estimate queue backlog B_t using Lindley recursion."""
    
    def __init__(self, use_simple_estimate=True):
        self.use_simple_estimate = use_simple_estimate
        self.mean_backlog = None
        
    def estimate_backlog_simple(self, has_manual, window_size=100):
        n = len(has_manual)
        backlog = np.zeros(n, dtype=np.float32)
        for i in range(n):
            start = max(0, i - window_size)
            end = i + 1
            backlog[i] = (~has_manual[start:end]).sum()
        self.mean_backlog = backlog.mean()
        return backlog
    
    def estimate_backlog_features(self, has_manual, sequence_ids=None):
        n = len(has_manual)
        features = {}
        features['backlog'] = self.estimate_backlog_simple(has_manual, window_size=100)
        
        if sequence_ids is not None:
            seq_labeled_frac = np.zeros(n, dtype=np.float32)
            seq_position = np.zeros(n, dtype=np.float32)
            for seq_id in np.unique(sequence_ids):
                mask = sequence_ids == seq_id
                n_seq = mask.sum()
                seq_labeled_frac[mask] = has_manual[mask].sum() / n_seq if n_seq > 0 else 0
                seq_position[mask] = np.arange(n_seq) / max(n_seq - 1, 1)
            features['seq_labeled_frac'] = seq_labeled_frac
            features['seq_position'] = seq_position
        else:
            features['seq_labeled_frac'] = np.zeros(n, dtype=np.float32)
            features['seq_position'] = np.zeros(n, dtype=np.float32)
        
        features['local_label_density'] = self._local_label_density(has_manual, window=50)
        return features
    
    def _local_label_density(self, has_manual, window=50):
        n = len(has_manual)
        density = np.zeros(n, dtype=np.float32)
        for i in range(n):
            start = max(0, i - window // 2)
            end = min(n, i + window // 2)
            density[i] = has_manual[start:end].mean()
        return density


class LabelingHazardModel:
    """Model P(labeled | X, B_t)."""
    
    def __init__(self, use_embeddings=True):
        self.model = None
        self.scaler = StandardScaler()
        self.use_embeddings = use_embeddings
        
    def prepare_features(self, X, backlog_features):
        feature_list = []
        if self.use_embeddings and X is not None:
            if X.shape[1] > 50:
                feature_list.append(X[:, :50])
            else:
                feature_list.append(X)
        for key in ['backlog', 'seq_labeled_frac', 'seq_position', 'local_label_density']:
            if key in backlog_features:
                feature_list.append(backlog_features[key].reshape(-1, 1))
        return np.hstack(feature_list)
    
    def fit(self, X, backlog_features, has_manual, verbose=True):
        features = self.prepare_features(X, backlog_features)
        if verbose:
            print("\n" + "=" * 70)
            print("FITTING HAZARD MODEL")
            print("=" * 70)
            print(f"  Feature dimensions: {features.shape}")
            print(f"  Labeled: {has_manual.sum()} ({100*has_manual.mean():.1f}%)")
        
        X_scaled = self.scaler.fit_transform(features)
        self.model = LogisticRegression(penalty='l2', C=1.0, class_weight='balanced', max_iter=1000, random_state=42)
        self.model.fit(X_scaled, has_manual)
        
        probs = self.model.predict_proba(X_scaled)[:, 1]
        try:
            auc = roc_auc_score(has_manual, probs)
            print(f"  Hazard model AUC: {auc:.3f}")
        except Exception as e:
            print(f"  Could not compute AUC: {e}")
        return self
    
    def predict_censoring_proba(self, X, backlog_features):
        features = self.prepare_features(X, backlog_features)
        X_scaled = self.scaler.transform(features)
        p_labeled = self.model.predict_proba(X_scaled)[:, 1]
        return 1 - p_labeled
    
    def compute_ipcw_weights(self, X, backlog_features, has_manual):
        p_unlabeled = self.predict_censoring_proba(X, backlog_features)
        p_labeled = np.clip(1 - p_unlabeled, 0.01, 0.99)
        weights = np.ones(len(has_manual), dtype=np.float32)
        weights[has_manual] = 1.0 / p_labeled[has_manual]
        weights = weights / weights[has_manual].mean()
        print(f"\n  IPCW weight range: [{weights[has_manual].min():.2f}, {weights[has_manual].max():.2f}]")
        return weights


def create_risk_classifier(model_type, n_classes=5, random_state=42, **kwargs):
    """
    Factory function to create different risk classifiers.
    
    Returns a classifier with predict_proba method.
    """
    if model_type == 'logreg':
        return LogisticRegression(
            #multi_class='multinomial',
            penalty='l2',
            C=kwargs.get('C', 1.0),
            max_iter=1000,
            random_state=random_state
        )
    
    elif model_type == 'rf':
        # Regularization to prevent overfitting
        return RandomForestClassifier(
            n_estimators=kwargs.get('n_estimators', 200),
            max_depth=kwargs.get('max_depth', 10),  # Reduced from 15
            min_samples_leaf=kwargs.get('min_samples_leaf', 20),  # Increased from 5
            min_samples_split=kwargs.get('min_samples_split', 40),  # Added
            max_features=kwargs.get('max_features', 'sqrt'),  # Limit features per split
            class_weight='balanced',
            oob_score=True,  # Out-of-bag score for honest evaluation
            n_jobs=-1,
            random_state=random_state
        )
    
    elif model_type == 'xgb':
        if not HAS_XGB:
            raise ImportError("XGBoost not installed")
        return XGBClassifier(
            n_estimators=kwargs.get('n_estimators', 200),
            max_depth=kwargs.get('max_depth', 5),  # Reduced
            learning_rate=kwargs.get('learning_rate', 0.05),  # Slower
            subsample=0.8,
            colsample_bytree=0.6,  # More regularization
            reg_alpha=kwargs.get('reg_alpha', 0.1),  # L1 regularization
            reg_lambda=kwargs.get('reg_lambda', 1.0),  # L2 regularization
            objective='multi:softprob',
            num_class=n_classes,
            use_label_encoder=False,
            eval_metric='mlogloss',
            n_jobs=-1,
            random_state=random_state
        )
    
    elif model_type == 'lgbm':
        if not HAS_LGBM:
            raise ImportError("LightGBM not installed")
        return LGBMClassifier(
            n_estimators=kwargs.get('n_estimators', 200),
            max_depth=kwargs.get('max_depth', 5),  # Reduced
            learning_rate=kwargs.get('learning_rate', 0.05),  # Slower
            subsample=0.8,
            colsample_bytree=0.6,
            reg_alpha=kwargs.get('reg_alpha', 0.1),
            reg_lambda=kwargs.get('reg_lambda', 1.0),
            class_weight='balanced',
            n_jobs=-1,
            random_state=random_state,
            verbose=-1
        )
    
    elif model_type == 'mlp':
        return MLPClassifier(
            hidden_layer_sizes=kwargs.get('hidden_layers', (128, 64)),  # Smaller
            activation='relu',
            solver='adam',
            alpha=kwargs.get('alpha', 0.01),  # More regularization
            batch_size=64,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=random_state
        )
    
    elif model_type == 'gb':
        return GradientBoostingClassifier(
            n_estimators=kwargs.get('n_estimators', 150),
            max_depth=kwargs.get('max_depth', 4),  # Reduced
            learning_rate=kwargs.get('learning_rate', 0.05),  # Slower
            subsample=0.8,
            min_samples_leaf=kwargs.get('min_samples_leaf', 20),
            random_state=random_state
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def evaluate_classifier_cv(X, y, weights, model_type, n_splits=3, verbose=True, **kwargs):
    """
    Evaluate a classifier using stratified cross-validation.
    Returns mean balanced accuracy.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    bas = []
    
    scaler = StandardScaler()
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        w_tr = weights[train_idx] if weights is not None else None
        
        # Scale
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)
        
        # Fit
        try:
            clf = create_risk_classifier(model_type, **kwargs)
            if w_tr is not None and hasattr(clf, 'fit'):
                # Not all classifiers support sample_weight
                try:
                    clf.fit(X_tr_s, y_tr, sample_weight=w_tr)
                except TypeError:
                    clf.fit(X_tr_s, y_tr)
            else:
                clf.fit(X_tr_s, y_tr)
            
            y_pred = clf.predict(X_val_s)
            ba = balanced_accuracy_score(y_val, y_pred)
            bas.append(ba)
        except Exception as e:
            if verbose:
                print(f"    {model_type} fold {fold+1} failed: {e}")
            bas.append(0.0)
    
    mean_ba = np.mean(bas)
    std_ba = np.std(bas)
    
    if verbose:
        print(f"    {model_type:8s}: CV BA = {mean_ba:.3f} ± {std_ba:.3f}")
    
    return mean_ba


def auto_select_model(X, y, weights, candidates=None, verbose=True, **kwargs):
    """
    Try multiple classifiers and pick the best one.
    """
    if candidates is None:
        candidates = ['logreg', 'rf', 'mlp']
        if HAS_XGB:
            candidates.append('xgb')
        if HAS_LGBM:
            candidates.append('lgbm')
    
    if verbose:
        print("\n" + "=" * 70)
        print("AUTO-SELECTING BEST RISK MODEL (3-fold CV)")
        print("=" * 70)
    
    results = {}
    for model_type in candidates:
        try:
            ba = evaluate_classifier_cv(X, y, weights, model_type, n_splits=3, verbose=verbose, **kwargs)
            results[model_type] = ba
        except Exception as e:
            if verbose:
                print(f"    {model_type}: FAILED ({e})")
    
    if not results:
        raise RuntimeError("All classifiers failed!")
    
    best_model = max(results, key=results.get)
    best_ba = results[best_model]
    
    if verbose:
        print(f"\n  ✓ Best model: {best_model} (CV BA = {best_ba:.3f})")
    
    return best_model, best_ba


class BALIRiskModelV3:
    """
    Risk model with flexible classifier backend.
    """
    
    def __init__(self, n_classes=5, allowed_classes=None, risk_factors=None, 
                 model_type='auto', calibrate=True, model_kwargs=None):
        self.n_classes = n_classes
        self.risk_model = None
        self.hazard_model = None
        self.scaler = StandardScaler()
        self.model_type = model_type
        self.calibrate = calibrate
        self.selected_model_type = None
        self.model_kwargs = model_kwargs or {}
        
        if allowed_classes is None:
            self.allowed_classes = np.array([0, 1, 2], dtype=int)
        else:
            self.allowed_classes = np.array(allowed_classes, dtype=int)
        
        if risk_factors is None:
            self.risk_factors = np.array([1.0, 1.0, 1.0, 0.05, 0.01], dtype=float)
        else:
            rf = np.array(risk_factors, dtype=float)
            if len(rf) != n_classes:
                raise ValueError(f"risk_factors must have length {n_classes}")
            self.risk_factors = rf
        
    def fit(self, X, y, weights=None, verbose=True):
        labeled = y >= 0
        X_labeled = X[labeled]
        y_labeled = y[labeled]

        # Compute combined weights (IPCW × class balance)
        if weights is not None:
            base_w = weights[labeled].astype(float)
        else:
            base_w = np.ones_like(y_labeled, dtype=float)
        
        class_counts = np.bincount(y_labeled, minlength=self.n_classes)
        inv_freq = np.zeros(self.n_classes, dtype=float)
        for c in range(self.n_classes):
            inv_freq[c] = 1.0 / max(1, class_counts[c])
        cls_weight_vec = inv_freq / (inv_freq.mean() if inv_freq.mean() > 0 else 1.0)
        w_cls = cls_weight_vec[y_labeled]
        w_labeled = base_w * w_cls
        
        if verbose:
            print("\n" + "=" * 70)
            print("FITTING RISK MODEL V3")
            print("=" * 70)
            print(f"  Training samples: {len(y_labeled)}")
            print(f"  Class distribution:")
            for c in range(self.n_classes):
                count = (y_labeled == c).sum()
                pct = 100 * count / len(y_labeled)
                print(f"    Class {c}: {count:4d} ({pct:5.1f}%)")
        
        # Auto-select model if requested
        if self.model_type == 'auto':
            self.selected_model_type, cv_ba = auto_select_model(
                X_labeled, y_labeled, w_labeled, verbose=verbose, **self.model_kwargs
            )
        else:
            self.selected_model_type = self.model_type
            if verbose:
                print(f"  Using specified model: {self.model_type}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_labeled)
        
        # Create and fit the selected model
        base_clf = create_risk_classifier(self.selected_model_type, self.n_classes, **self.model_kwargs)
        
        # Fit with sample weights if supported
        try:
            base_clf.fit(X_scaled, y_labeled, sample_weight=w_labeled)
        except TypeError:
            if verbose:
                print(f"  Note: {self.selected_model_type} doesn't support sample_weight, using unweighted fit")
            base_clf.fit(X_scaled, y_labeled)
        
        # Optionally calibrate probabilities
        if self.calibrate and self.selected_model_type not in ['logreg']:
            if verbose:
                print(f"  Calibrating probabilities (isotonic)...")
            self.risk_model = CalibratedClassifierCV(
                base_clf, method='isotonic', cv='prefit'
            )
            self.risk_model.fit(X_scaled, y_labeled)
        else:
            self.risk_model = base_clf
        
        # Report training performance
        if verbose:
            y_pred = self.risk_model.predict(X_scaled)
            ba = balanced_accuracy_score(y_labeled, y_pred)
            print(f"\n  Training balanced accuracy: {ba:.3f}")
            
            # For RF, also report OOB score (honest estimate)
            if self.selected_model_type == 'rf' and hasattr(base_clf, 'oob_score_'):
                print(f"  RF Out-of-Bag score: {base_clf.oob_score_:.3f} (honest estimate)")
            
            print(f"  Per-class accuracy:")
            for c in range(self.n_classes):
                mask = y_labeled == c
                if mask.any():
                    acc = (y_pred[mask] == c).mean()
                    print(f"    Class {c}: {acc:.3f}")
        
        return self
    
    def predict_risk_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.risk_model.predict_proba(X_scaled)
    
    def predict_with_bayes_adjustment(self, X, backlog_features, is_unknown, 
                                     hazard_model, verbose=True):
        p_risk = self.predict_risk_proba(X)
        
        if not is_unknown.any():
            return p_risk, p_risk
        
        if verbose:
            print("\n" + "=" * 70)
            print("BAYES ADJUSTMENT FOR UNKNOWNS")
            print("=" * 70)
        
        p_adjusted = p_risk.copy()
        p_censored = hazard_model.predict_censoring_proba(
            X[is_unknown], 
            {k: v[is_unknown] for k, v in backlog_features.items()}
        )
        
        if verbose:
            print(f"  Unknowns to adjust: {is_unknown.sum()}")
            print(f"  Mean P(unlabeled | X, B_t): {p_censored.mean():.3f}")
            print(f"  Using risk factors: {self.risk_factors}")
        
        for y in range(self.n_classes):
            p_adjusted[is_unknown, y] *= p_censored * self.risk_factors[y]
        
        row_sums = p_adjusted[is_unknown].sum(axis=1, keepdims=True)
        p_adjusted[is_unknown] /= np.maximum(row_sums, 1e-10)
        
        return p_adjusted, p_risk
    
    def generate_pseudo_labels(self, X, backlog_features, y_true, hazard_model,
                               confidence_threshold=0.90, verbose=True):
        is_unknown = y_true < 0
        
        probs_adjusted, probs_base = self.predict_with_bayes_adjustment(
            X, backlog_features, is_unknown, hazard_model, verbose=verbose
        )
        
        y_pred = np.argmax(probs_adjusted, axis=1)
        conf_adjusted = np.max(probs_adjusted, axis=1)
        
        allowed_mask = np.isin(y_pred, self.allowed_classes)
        
        y_pseudo = np.where(
            is_unknown & allowed_mask & (conf_adjusted >= confidence_threshold),
            y_pred,
            -1
        )
        
        n_accepted = (y_pseudo >= 0).sum()
        n_unknown = is_unknown.sum()
        n_blocked = (is_unknown & ~allowed_mask & (conf_adjusted >= confidence_threshold)).sum()
        
        if verbose:
            print(f"\n  Confidence threshold: {confidence_threshold:.2f}")
            print(f"  Allowed classes: {list(self.allowed_classes)}")
            print(f"  Pseudo-labels accepted: {n_accepted} / {n_unknown} ({100*n_accepted/max(1,n_unknown):.1f}%)")
            print(f"  Blocked by class restriction: {n_blocked}")
            
            if n_accepted > 0:
                print(f"  Mean confidence (accepted): {conf_adjusted[y_pseudo >= 0].mean():.3f}")
            
            print(f"  Pseudo-label distribution:")
            for c in range(self.n_classes):
                count = (y_pseudo == c).sum()
                if count > 0:
                    pct = 100 * count / max(1, n_accepted)
                    print(f"    Class {c}: {count:4d} ({pct:5.1f}%)")
        
        return y_pseudo, conf_adjusted, probs_adjusted


def evaluate_on_masked_test(bali_model, X, y, backlog_features, hazard_model,
                            mask_frac=0.25, confidence_threshold=0.90):
    print("\n" + "=" * 70)
    print("MASKED TEST EVALUATION")
    print("=" * 70)
    
    labeled = y >= 0
    labeled_idx = np.where(labeled)[0]
    
    mask_idx = []
    for c in range(bali_model.n_classes):
        class_idx = labeled_idx[y[labeled_idx] == c]
        if len(class_idx) > 0:
            n_mask = max(1, int(mask_frac * len(class_idx)))
            mask_idx.extend(np.random.choice(class_idx, n_mask, replace=False))
    
    mask_idx = np.array(mask_idx)
    
    y_masked = y.copy()
    y_masked[mask_idx] = -1
    
    print(f"  Masked {len(mask_idx)} samples as fake unknowns")
    print(f"  True class distribution of masked:")
    for c in range(bali_model.n_classes):
        count = (y[mask_idx] == c).sum()
        pct = 100 * count / len(mask_idx)
        allowed = "✓" if c in bali_model.allowed_classes else "✗"
        print(f"    Class {c}: {count:4d} ({pct:5.1f}%) {allowed}")
    
    y_pseudo, conf, probs = bali_model.generate_pseudo_labels(
        X, backlog_features, y_masked, hazard_model,
        confidence_threshold=confidence_threshold,
        verbose=False
    )
    
    recovered = y_pseudo[mask_idx] >= 0
    print(f"\n  Recovery rate: {recovered.sum()} / {len(mask_idx)} ({100*recovered.mean():.1f}%)")
    
    if recovered.any():
        ba = balanced_accuracy_score(y[mask_idx][recovered], y_pseudo[mask_idx][recovered])
        print(f"  Balanced accuracy (recovered): {ba:.3f}")
        
        print(f"  Per-class recall:")
        for c in range(bali_model.n_classes):
            class_mask = y[mask_idx] == c
            if class_mask.any():
                correct = (y_pseudo[mask_idx][class_mask] == c).sum()
                total = class_mask.sum()
                recall = correct / total
                allowed = "✓" if c in bali_model.allowed_classes else "✗"
                print(f"    Class {c}: {recall:.3f} ({correct}/{total}) {allowed}")
        return ba
    else:
        print("  No samples recovered")
        return 0.0


def main():
    ap = argparse.ArgumentParser(description="BALI v3 - Better Risk Models")
    ap.add_argument("--emb-root", required=True, help="Root directory with embeddings")
    ap.add_argument("--data-type", default="bets", help="Data type")
    ap.add_argument("--confidence-threshold", type=float, default=0.70)
    ap.add_argument("--mask-frac", type=float, default=0.25)
    ap.add_argument("--out-suffix", default="_bali")
    ap.add_argument("--seed", type=int, default=42)
    
    ap.add_argument("--allowed-classes", default="0,1,2",
                   help="Comma-separated classes to pseudo-label")
    ap.add_argument("--risk-factors", default="1.0,1.0,1.0,0.05,0.01",
                   help="Comma-separated risk factors for Bayes adjustment")
    
    # V3: Model selection
    ap.add_argument("--risk-model", default="auto",
                   choices=['auto', 'logreg', 'rf', 'xgb', 'lgbm', 'mlp', 'gb'],
                   help="Risk model type ('auto' tries multiple and picks best)")
    ap.add_argument("--no-calibrate", action="store_true",
                   help="Skip probability calibration")
    
    # RF regularization options
    ap.add_argument("--max-depth", type=int, default=10,
                   help="Max tree depth for RF/XGB/LGBM (default: 10)")
    ap.add_argument("--min-samples-leaf", type=int, default=20,
                   help="Min samples per leaf for RF (default: 20)")
    
    args = ap.parse_args()
    np.random.seed(args.seed)
    
    allowed_classes = [int(x) for x in args.allowed_classes.split(",") if x.strip()]
    risk_factors = [float(x) for x in args.risk_factors.split(",") if x.strip()]
    
    if len(risk_factors) != 5:
        raise ValueError(f"--risk-factors must have 5 values, got {len(risk_factors)}")
    
    print("=" * 70)
    print("BACKLOG-AWARE LABEL INFERENCE (BALI) v3")
    print("=" * 70)
    print(f"  Risk model: {args.risk_model}")
    print(f"  Allowed classes: {allowed_classes}")
    print(f"  Risk factors: {risk_factors}")
    print(f"  Confidence threshold: {args.confidence_threshold}")
    
    # Load data
    print(f"\nLoading embeddings from {args.emb_root}/{args.data_type}...")
    train_path = Path(args.emb_root) / args.data_type / "embeddings_train.npz"
    data = np.load(train_path, allow_pickle=True)
    
    X = data['embeddings'].astype(np.float32)
    has_manual = data['has_manual'].astype(bool)
    y = data['manual_labels'].astype(int)
    y[~has_manual] = -1
    seq_ids = data['sequence_ids'] if 'sequence_ids' in data else None
    
    print(f"  Total samples: {len(X)}")
    print(f"  Labeled: {has_manual.sum()} ({100*has_manual.mean():.1f}%)")
    print(f"  Embedding dim: {X.shape[1]}")
    
    print(f"  Class distribution:")
    for c in range(5):
        count = (y == c).sum()
        pct = 100 * count / has_manual.sum()
        allowed = "✓" if c in allowed_classes else "✗"
        print(f"    Class {c}: {count:4d} ({pct:5.1f}%) {allowed}")
    
    # Backlog estimation
    print("\n" + "=" * 70)
    print("STEP 1: BACKLOG ESTIMATION")
    print("=" * 70)
    
    backlog_estimator = BacklogEstimator()
    backlog_features = backlog_estimator.estimate_backlog_features(has_manual, seq_ids)
    
    for key, vals in backlog_features.items():
        print(f"    {key}: mean={vals.mean():.3f}, std={vals.std():.3f}")
    
    # Hazard model
    hazard_model = LabelingHazardModel(use_embeddings=True)
    hazard_model.fit(X, backlog_features, has_manual, verbose=True)
    
    # IPCW weights
    ipcw_weights = hazard_model.compute_ipcw_weights(X, backlog_features, has_manual)
    
    # Risk model V3
    model_kwargs = {
        'max_depth': args.max_depth,
        'min_samples_leaf': args.min_samples_leaf,
    }
    
    bali_model = BALIRiskModelV3(
        n_classes=5,
        allowed_classes=allowed_classes,
        risk_factors=risk_factors,
        model_type=args.risk_model,
        calibrate=not args.no_calibrate,
        model_kwargs=model_kwargs
    )
    bali_model.fit(X, y, weights=ipcw_weights, verbose=True)
    bali_model.hazard_model = hazard_model
    
    # Evaluate
    ba_masked = evaluate_on_masked_test(
        bali_model, X, y, backlog_features, hazard_model,
        mask_frac=args.mask_frac,
        confidence_threshold=args.confidence_threshold
    )
    
    # Generate pseudo-labels
    print("\n" + "=" * 70)
    print("GENERATING PSEUDO-LABELS FOR REAL UNKNOWNS")
    print("=" * 70)
    
    y_pseudo, conf, probs = bali_model.generate_pseudo_labels(
        X, backlog_features, y, hazard_model,
        confidence_threshold=args.confidence_threshold,
        verbose=True
    )
    
    # Save
    out_path = Path(args.emb_root) / args.data_type / f"embeddings_train{args.out_suffix}.npz"
    
    has_manual_updated = has_manual.copy()
    y_updated = y.copy()
    
    accepted = y_pseudo >= 0
    has_manual_updated[accepted] = True
    y_updated[accepted] = y_pseudo[accepted]
    
    print(f"\nSaving to {out_path}...")
    np.savez(
        out_path,
        embeddings=X,
        has_manual=has_manual_updated,
        manual_labels=y_updated,
        sequence_ids=seq_ids,
        bali_confidence=conf,
        bali_probs=probs,
        bali_model_type=bali_model.selected_model_type,
        bali_allowed_classes=np.array(allowed_classes),
        bali_risk_factors=np.array(risk_factors),
        **{k: v for k, v in data.items() if k not in 
           ['embeddings', 'has_manual', 'manual_labels', 'sequence_ids']}
    )
    
    print(f"  Original labeled: {has_manual.sum()}")
    print(f"  Pseudo-labels added: {accepted.sum()}")
    print(f"  Total labeled: {has_manual_updated.sum()}")
    print(f"  Coverage: {100 * has_manual_updated.mean():.1f}%")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Selected model: {bali_model.selected_model_type}")
    print(f"  Masked-test BA: {ba_masked:.3f}")
    print(f"  Pseudo-labels added: {accepted.sum()} / {(~has_manual).sum()}")
    print(f"  Final coverage: {100 * has_manual_updated.mean():.1f}%")
    
    # Distribution sanity check
    print("\n  Distribution comparison:")
    print(f"  {'Class':<8} {'Original':<12} {'Pseudo':<12} {'Combined':<12}")
    print(f"  {'-'*44}")
    
    orig_dist = np.bincount(y[has_manual], minlength=5) / has_manual.sum()
    pseudo_dist = np.bincount(y_pseudo[y_pseudo >= 0], minlength=5) / max(1, (y_pseudo >= 0).sum())
    combined_dist = np.bincount(y_updated[has_manual_updated], minlength=5) / has_manual_updated.sum()
    
    max_shift = 0
    for c in range(5):
        o = 100 * orig_dist[c]
        p = 100 * pseudo_dist[c] if (y_pseudo >= 0).any() else 0
        cb = 100 * combined_dist[c]
        shift = abs(cb - o)
        max_shift = max(max_shift, shift)
        print(f"  {c:<8} {o:>10.1f}%  {p:>10.1f}%  {cb:>10.1f}%")
    
    print("=" * 70)
    
    if ba_masked >= 0.40:
        print("\n✅ BA ≥ 0.40 - pseudo-labels are trustworthy!")
    else:
        print(f"\n⚠️  BA = {ba_masked:.3f} < 0.40 - consider using these cautiously")
    
    if max_shift > 20:
        print(f"⚠️  Large distribution shift detected ({max_shift:.1f}% max)")
        print("   This may be valid (MNAR assumption) or indicate model bias.")


if __name__ == "__main__":
    main()