#!/usr/bin/env python3
"""Run the public synthetic demo pipeline.

This is a convenience wrapper that runs:
1) BALI on the demo train split
2) GHMM on demo train/val/test splits

Example:
    python scripts/run_demo_pipeline.py --stream bets
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", default="bets",
                    choices=["transactions", "bets", "sessions", "payments"],
                    help="Which synthetic stream to run.")
    ap.add_argument("--emb-root", default="data/demo", help="Embeddings root (synthetic).")
    ap.add_argument("--out-dir", default="outputs/demo_ghmm", help="Output directory for GHMM artefacts.")
    ap.add_argument("--n-states-grid", default="6,8", help="Comma-separated grid of n_states values for GHMM.")
    ap.add_argument("--reduce-dim", default="12", help="Dimensionality reduction target.")
    ap.add_argument("--risk-model", default="logreg", help="BALI risk model (e.g., logreg, rf, auto).")
    args = ap.parse_args()

    emb_root = Path(args.emb_root)
    if not (emb_root / args.stream / "embeddings_train.npz").exists():
        raise FileNotFoundError(
            f"Missing demo embeddings. Expected: {emb_root / args.stream / 'embeddings_train.npz'}\n"
            "Regenerate with: python scripts/make_demo_data.py"
        )

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # 1) BALI
    run([
        sys.executable, "pipelines/bali_for_bets_v2.py",
        "--emb-root", str(emb_root),
        "--data-type", args.stream,
        "--risk-model", args.risk_model,
        "--out-suffix", "_bali",
    ])

    # 2) GHMM
    run([
        sys.executable, "pipelines/ghmm_improved_v5_3class_optimized.py",
        "--data-type", args.stream,
        "--embeddings-root", str(emb_root),
        "--output-dir", args.out_dir,
        "--n-states-grid", args.n_states_grid,
        "--reduce-dim", args.reduce_dim,
    ])

    print("\nDemo complete. Outputs written to:", args.out_dir)


if __name__ == "__main__":
    main()
