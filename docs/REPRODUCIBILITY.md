# Reproducibility notes

This project is implemented as a sequence of scripts that persist intermediate artefacts between stages.

## Pipeline order (high level)

1. `preprocessing_final_script_dates_thread_safe_final.py`
   - data cleaning
   - split construction (including leakage audits)
   - window construction

2. `enhanced_feature_engineering_final_script_final_use.py`
   - heavy-tail-aware exceedance features
   - tail dependence interactions
   - consistency artefacts (thresholds, feature order, etc.)

3. `feature_selection_prototype.py`
   - unsupervised + weakly-supervised (RG) feature selection

4. `train_hierarchical_risk_priors_check_v3.py`
   - hierarchical conditional VAE with risk-specific priors

5. `prototype_teacher_student_version_october_27_final.py`
   - teacher–student training and embedding export

6. `bali_for_bets_v2.py`
   - backlog-aware label inference (BALI)

7. `ghmm_improved_v5_3class_optimized.py`
   - Gaussian HMM regime modelling + risk mapping

## Determinism / seeds

Where possible, set and report:
- Python version
- package versions
- random seeds
- GPU / CUDA versions (if used)

## Expected outputs

The scripts write outputs into local folders (e.g., `processed_data/`, embeddings folders, plots, results JSON).
For publication release, add a clear directory layout table here once final paths are fixed.


---

## Synthetic demo

A small synthetic dataset and a runnable smoke-test pipeline (BALI + GHMM) are provided in `data/demo/`.

See `docs/DEMO_PIPELINE.md`.
