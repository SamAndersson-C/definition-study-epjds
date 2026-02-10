# Definition Study (EPJ Data Science) — Reproducibility Repository

This repository contains code for the EPJ Data Science Regular Article:

> *Heavy-tail-aware representation learning and dynamic Bayesian state modelling to derive an operational proxy definition of problem gambling risk*

The project is designed for **operational monitoring / triage evaluation**, not clinical diagnosis.

## Data access

Raw behavioural telemetry and manual analyst assessments used in the paper are **restricted** due to privacy and contractual constraints.

This repository therefore provides:

- the end-to-end pipeline code (preprocessing → feature engineering → representation learning → BALI → GHMM),
- configuration templates,
- and a **synthetic demo dataset** sufficient to run a smoke-test of the public pipeline.

See `data/README.md` and `data/demo/README.md`.

## Quick start (synthetic demo)

This demo runs **BALI + GHMM** on the synthetic embeddings in `data/demo/` so that others can execute the code without restricted data.

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate  # Windows PowerShell

pip install -U pip
pip install -r requirements-demo.txt

# Example: run BALI on demo bets embeddings
python bali_for_bets_v2.py --emb-root data/demo --data-type bets --risk-model logreg --out-suffix _bali

# Example: run GHMM on demo bets embeddings
python ghmm_improved_v5_3class_optimized.py --data-type bets --embeddings-root data/demo --output-dir outputs/demo_ghmm --n-states-grid 6,8 --reduce-dim 12
```

Full instructions: `docs/DEMO_PIPELINE.md`.

**Important:** any metrics produced on the demo data are meaningless and must not be compared to the paper’s results.

## Full pipeline (restricted data)

The scripts in this repository were used in an internal pipeline that starts from operator exports / a database and ends in model artefacts, tables and figures.

See `docs/REPRODUCIBILITY.md` for the intended order of operations.

## Citation

Add/update `CITATION.cff` once an archival DOI is available (e.g., Zenodo).

## Licence

**TODO:** choose a licence consistent with organisational and co-author requirements.

