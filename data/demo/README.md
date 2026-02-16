# Demo data (synthetic)

This folder contains **synthetic** embeddings and labels that are safe to publish and are intended purely to let others run the public code end-to-end.

- **These files contain no real customer data.**
- **Any metrics produced on this demo data are meaningless** and must not be compared to the paper’s results.

## File format

For each stream (`transactions`, `bets`, `sessions`, `payments`) and split (`train`, `val`, `test`), we provide:

- `embeddings_<split>.npz`

Each `.npz` file contains:

- `embeddings` (float32, shape `[n_windows, d]`): synthetic window embeddings.
- `sequence_ids` (int, shape `[n_windows]`): synthetic customer / sequence identifier.
- `has_manual` (bool, shape `[n_windows]`): whether a manual label is observed (MNAR-style selective labelling).
- `manual_labels` (int, shape `[n_windows]`): *observed* manual label in `{0,1,2,3,4}` when `has_manual=True`.  
  When `has_manual=False`, the value is a **placeholder** (set to `0`) and should be ignored.
- `synthetic_true_labels` (int, shape `[n_windows]`): the underlying synthetic class used to generate the embeddings (for debugging only).
- `rg_cat_dists` (float32, shape `[n_windows, 5]`): synthetic proxy-category distributions.
- `manual_labels_multihot` (float32, shape `[n_windows, 5]`): one-hot labels for labelled windows, zeros otherwise.
- `window_coverage` (float32, shape `[n_windows]`): synthetic coverage fraction.

## Intended demo pipeline

1. Run BALI on the training split to generate pseudo-labels (writes `embeddings_train_bali.npz`).
2. Run the GHMM script using `--embeddings-root data/demo`.

See `docs/DEMO_PIPELINE.md`.
