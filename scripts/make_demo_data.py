#!/usr/bin/env python3
"""Generate synthetic demo embeddings for the public repository.

This script creates *.npz files compatible with:
- bali_for_bets_v2.py
- ghmm_improved_v5_3class_optimized.py

The outputs contain NO real data and are intended only to smoke-test the pipeline.

Usage:
    python scripts/make_demo_data.py --out-root data/demo --seed 20260210
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def make_transition_matrix(n: int = 5, stay: float = 0.92, step: float = 0.04) -> np.ndarray:
    T = np.zeros((n, n), dtype=float)
    for i in range(n):
        T[i, i] = stay
        if i > 0:
            T[i, i - 1] += step
        else:
            T[i, i] += step
        if i < n - 1:
            T[i, i + 1] += step
        else:
            T[i, i] += step
    return T / T.sum(axis=1, keepdims=True)


def simulate_markov_states(length: int, T: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = T.shape[0]
    p0 = np.array([0.55, 0.25, 0.12, 0.06, 0.02], dtype=float)
    p0 = p0 / p0.sum()
    states = np.empty(length, dtype=int)
    states[0] = rng.choice(n, p=p0)
    for t in range(1, length):
        states[t] = rng.choice(n, p=T[states[t - 1]])
    return states


def sample_embeddings(states: np.ndarray, dim: int, rng: np.random.Generator) -> np.ndarray:
    n = len(states)
    means = np.zeros((5, dim), dtype=np.float32)
    for c in range(5):
        means[c, 0] = (c - 2) * 1.2
        means[c, 1] = (c - 2) * 0.7
        means[c, 2] = (c - 2) * 0.4

    # Heavy-tailed noise: Student-t with lognormal scaling.
    noise = rng.standard_t(df=3, size=(n, dim)).astype(np.float32)
    scale = rng.lognormal(mean=0.0, sigma=0.35, size=(n, 1)).astype(np.float32)

    return means[states] + noise * scale


def rg_dirichlet(states: np.ndarray, rng: np.random.Generator, alpha_base: float = 0.3) -> np.ndarray:
    n = len(states)
    out = np.zeros((n, 5), dtype=np.float32)
    for i, s in enumerate(states):
        alpha = np.full(5, alpha_base, dtype=np.float32)
        alpha[s] += 2.0
        out[i] = rng.dirichlet(alpha).astype(np.float32)
    return out


def label_propensity(states: np.ndarray, base: float, slope: float) -> np.ndarray:
    p = base + slope * states
    return np.clip(p, 0.01, 0.95)


def build_split(
    customers: list[int],
    seq_len_range: tuple[int, int],
    dim: int,
    base_label: float,
    slope_label: float,
    rng: np.random.Generator,
) -> dict:
    T = make_transition_matrix()
    embeddings_list = []
    has_manual_list = []
    manual_labels_list = []
    true_labels_list = []
    seq_ids_list = []
    rg_list = []
    multihot_list = []
    coverage_list = []

    for cid in customers:
        L = int(rng.integers(seq_len_range[0], seq_len_range[1] + 1))
        states = simulate_markov_states(L, T, rng)
        X = sample_embeddings(states, dim, rng)

        p_label = label_propensity(states, base_label, slope_label)
        has_manual = rng.random(L) < p_label

        # manual_labels is only meaningful when has_manual=True.
        # For unlabeled rows we write a placeholder (0) and provide synthetic_true_labels for debugging.
        manual = states.copy()
        manual[~has_manual] = 0

        embeddings_list.append(X)
        has_manual_list.append(has_manual.astype(bool))
        manual_labels_list.append(manual.astype(int))
        true_labels_list.append(states.astype(int))
        seq_ids_list.append(np.full(L, cid, dtype=int))
        rg_list.append(rg_dirichlet(states, rng))

        mh = np.zeros((L, 5), dtype=np.float32)
        mh[np.arange(L), states] = 1.0
        mh[~has_manual] = 0.0
        multihot_list.append(mh)

        coverage_list.append(rng.uniform(0.7, 1.0, size=L).astype(np.float32))

    return dict(
        embeddings=np.vstack(embeddings_list).astype(np.float32),
        has_manual=np.concatenate(has_manual_list),
        manual_labels=np.concatenate(manual_labels_list),
        synthetic_true_labels=np.concatenate(true_labels_list),
        sequence_ids=np.concatenate(seq_ids_list),
        rg_cat_dists=np.vstack(rg_list).astype(np.float32),
        manual_labels_multihot=np.vstack(multihot_list).astype(np.float32),
        window_coverage=np.concatenate(coverage_list).astype(np.float32),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=str, default="data/demo", help="Output root directory.")
    ap.add_argument("--seed", type=int, default=20260210, help="Random seed.")
    ap.add_argument("--dim", type=int, default=32, help="Embedding dimension.")
    ap.add_argument("--min-len", type=int, default=20, help="Min sequence length.")
    ap.add_argument("--max-len", type=int, default=45, help="Max sequence length.")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    streams = ["transactions", "bets", "sessions", "payments"]
    stream_params = {
        "transactions": dict(base=0.03, slope=0.03),
        "sessions": dict(base=0.35, slope=0.07),
        "payments": dict(base=0.55, slope=0.08),
        "bets": dict(base=0.50, slope=0.08),
    }
    split_customers = {
        "train": range(1000, 1080),  # 80 customers
        "val": range(2000, 2020),  # 20 customers
        "test": range(3000, 3020),  # 20 customers
    }
    seq_len_range = (args.min_len, args.max_len)

    for stream in streams:
        stream_dir = out_root / stream
        stream_dir.mkdir(parents=True, exist_ok=True)
        params = stream_params[stream]

        for split, customers in split_customers.items():
            d = build_split(
                customers=list(customers),
                seq_len_range=seq_len_range,
                dim=args.dim,
                base_label=params["base"],
                slope_label=params["slope"],
                rng=rng,
            )
            out_path = stream_dir / f"embeddings_{split}.npz"
            np.savez(out_path, **d)
            print(f"Wrote {out_path} ({len(d['embeddings'])} windows, d={d['embeddings'].shape[1]})")

    print("Done.")


if __name__ == "__main__":
    main()
