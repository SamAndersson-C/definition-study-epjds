# Demo pipeline (synthetic data)

This demo is included so that reviewers and readers can **run the code** without access to restricted operator data.

The demo uses synthetic embeddings located in `data/demo/` and runs:

1. Backlog-aware label inference (BALI)
2. Gaussian HMM regime modelling (GHMM)

## 1) Install (pip)

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate  # Windows PowerShell

pip install -U pip
pip install -r requirements-demo.txt
```

## 2) Run BALI (pseudo-labelling)

Example (bets stream):

```bash
python bali_for_bets_v2.py \
  --emb-root data/demo \
  --data-type bets \
  --risk-model logreg \
  --confidence-threshold 0.7 \
  --out-suffix _bali
```

This writes:

- `data/demo/bets/embeddings_train_bali.npz`

## 3) Run GHMM

```bash
python ghmm_improved_v5_3class_optimized.py \
  --data-type bets \
  --embeddings-root data/demo \
  --output-dir outputs/demo_ghmm \
  --n-states-grid 6,8 \
  --reduce-dim 12 \
  --preprocessing copula_zca \
  --covariance-type diag
```

## Notes

- The GHMM script will perform model selection over the specified `--n-states-grid`. Keep this small for quick runs.
- Any metrics printed by the scripts are **only for smoke-testing** and should not be interpreted.
