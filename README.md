# Definition Study (EPJ Data Science) — Reproducibility Repository

This repository contains code for the manuscript entitled:

> *Heavy-tail-aware representation learning and dynamic Bayesian state modelling to derive an operational proxy definition of problem gambling risk*

submitted to EPJ Data Science.

The project is designed for **operational monitoring / triage evaluation**, not clinical diagnosis.

## Data access

Raw behavioural telemetry and manual analyst assessments used in the paper are **restricted** due to privacy and contractual constraints.

This repository therefore provides:

- the end-to-end pipeline implementation (preprocessing → feature engineering → representation learning → BALI → GHMM),
- configuration templates,
- and a **synthetic demo dataset** sufficient to run a smoke-test of the public pipeline.

See `data/README.md` and `data/demo/README.md`.

## Quick start (synthetic demo)

This repository includes a **fully runnable synthetic demo** that exercises the public pipeline (**BALI → GHMM**) without requiring restricted data.

The demo uses pre-generated synthetic embeddings in `data/demo/` and is intended solely as a **smoke test** of the code structure and execution flow.

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows PowerShell

pip install -U pip
pip install -r requirements-demo.txt

# One-command demo: runs BALI → GHMM on synthetic bets data
python scripts/run_demo_pipeline.py --stream bets
```

Supported demo streams are:

- `bets`
- `transactions`
- `sessions`
- `payments`

Full step-by-step instructions are provided in `docs/DEMO_PIPELINE.md`.

**Important:** Any metrics produced on the synthetic demo data are not meaningful and must not be compared to the paper’s results.

## Repository structure

The repository is organised to separate **entrypoints**, **implementation**, **documentation**, and **data**.

```
.
├── scripts/                 # User-facing entrypoints (CLI wrappers)
│   ├── run_demo_pipeline.py # One-command synthetic demo (BALI → GHMM)
│   └── make_demo_data.py    # Deterministic synthetic data generator
│
├── pipelines/               # Core pipeline implementations (not user API)
│   ├── bali_for_bets_v2.py
│   ├── ghmm_improved_v5_3class_optimized.py
│   ├── preprocessing_final_script_dates_thread_safe_final.py
│   ├── enhanced_feature_engineering_final_script_final_use.py
│   ├── train_hierarchical_risk_priors_check_v3.py
│   └── prototype_teacher_student_version_october_27_final.py
│
├── experiments/             # Prototypes / exploratory scripts (not required for demo)
│   └── feature_selection_prototype.py
│
├── data/
│   ├── demo/                # Synthetic demo data (public)
│   └── README.md            # Data access and restrictions
│
├── docs/
│   ├── DEMO_PIPELINE.md     # Detailed demo instructions (copy/paste)
│   └── REPRODUCIBILITY.md   # Intended full pipeline order (restricted data context)
│
├── configs/                 # Configuration templates (YAML)
├── outputs/                 # Runtime outputs (ignored by git)
├── env.example              # Environment variable template (no secrets)
├── utils.py                 # Shared helpers
├── requirements-demo.txt    # Minimal dependencies for demo
├── requirements.in          # Full internal environment (reference)
├── CITATION.cff             # Citation metadata (update DOI once archived)
├── LICENSE                  # GNU GPL-3.0 license
└── README.md
```

Notes:

- Only files in `scripts/` are intended to be executed directly.
- Files in `pipelines/` implement the methodological components described in the paper.
- Real data and operator-specific integrations are not included.

## Full pipeline (restricted data)

The scripts in this repository were used in an internal pipeline that starts from operator exports / a database and ends in model artefacts, tables and figures.

See `docs/REPRODUCIBILITY.md` for the intended order of operations.

## Citation


Please cite this repository using the Zenodo DOI associated with the archived release (see DOI badge above) 
and the metadata provided in CITATION.cff.



## Licence

This project is licensed under the **GNU General Public License v3.0 (GPL-3.0)**.

If you use, modify, or redistribute this code (or derivative works), you must do so under
GPL-3.0-compatible terms and provide appropriate attribution. Proprietary redistribution
is not permitted without explicit re-licensing by the copyright holder(s).
