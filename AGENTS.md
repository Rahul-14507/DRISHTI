# AGENTS.md — SECOM-ML-MODEL

Flat repo of standalone Python ML scripts. No package, no test suite, no lint/typecheck, no CI.

## Setup / run

- `pip install -r requirements.txt` then `python <script>.py` (PowerShell on Windows).
- All training scripts fetch UCI SECOM (id=179) live via `ucimlrepo` — network required, no local dataset/cache.
- Runs are long: `production_pipeline_v2.py` does 5-fold `cross_val_predict`; `train_optimized.py` uses 300 trees.

## Which script matters

- `production_pipeline_v2.py` — canonical final pipeline. Prefer it over `production_pipeline.py` (older; optimizes threshold on in-sample train probas = leaky. V2 fixes this with `StratifiedKFold` + `cross_val_predict`).
- `train_classifier.py` — baseline; the ONLY script that writes `secom_defect_model.pkl`.
- `test_defective.py`, `predict_random_samples.py` — inference demos; require `secom_defect_model.pkl` to exist. Run `train_classifier.py` first if missing.
- `secom_analysis.py` — unrelated exploratory prototype (lot-aware MAD outliers + drift regressor). Ignore when touching production flow.
- `qa_dashboard.py` — hackathon pitch app (Streamlit QA inspector dashboard). Self-contained: synthetic burn-in data (0/24/96/168h), no UCI fetch. Run with `streamlit run qa_dashboard.py`. LLM explanations via local Ollama `llama3.1:latest`; works offline via rule-based fallback.
- `secom_defect_model.pkl` is committed to git; don't re-commit it unless the training pipeline intentionally changed.

## Invariants (do not "improve")

- Labels: `-1 → 0 (Pass)`, `1 → 1 (Fail)`. Defects are ~6.6% — never judge by accuracy or default 0.5 threshold.
- Split: `train_test_split(test_size=0.2, stratify=y, random_state=42)`. Keep exact for reproducibility; inference scripts recreate the same split.
- Cleaning: drop `class` + `timestamp` from features, drop cols with >40% NaN, drop zero-variance cols, `SimpleImputer(median)` → `StandardScaler`.
- Decision threshold: maximize F2 via `precision_recall_curve` (`5*p*r/(4*p+r)`); guard the `best_idx == len(thresholds)` edge case as V2 does.
- Column-naming quirk: `train_classifier.py` / `test_defective.py` / `predict_random_samples.py` rename features to `str(i)`; production pipelines keep original names. Inference input must match the naming its model trained with.
