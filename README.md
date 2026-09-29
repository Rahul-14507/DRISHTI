# 🔬 SECOM: Semiconductor Predictive Defect & Burn-In QA Intelligence

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.51-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-Cost--Sensitive-EB5424)](https://xgboost.readthedocs.io)
[![Dataset](https://img.shields.io/badge/UCI%20Dataset-SECOM%20(ID:179)-0288D1)](https://archive.ics.uci.edu/dataset/179/secom)

An end-to-end industrial machine learning system and QA command center designed for **early defect detection, lot-relative anomaly screening, and burn-in drift forecasting** in semiconductor fabrication and test packaging.

---

## 📌 Executive Summary & Problem Context

In semiconductor manufacturing, wafers pass through hundreds of complex chemical, lithographic, and thermal steps. Modern fabrication facilities monitor these processes with dense sensor arrays. However, automated quality control faces three critical challenges:

1. **Severe Class Imbalance (~14:1)**: Defective chips represent only **~6.6%** (104 of 1,567) of all runs. A naive model that classifies 100% of chips as "PASS" achieves **93.4% accuracy** while catching **0% of defects**.
2. **Asymmetric Failure Cost ($10\times$–$100\times$)**: Shipping a defective chip (False Negative) results in catastrophic downstream failures, recalls, or warranty claims. Conversely, flagging a questionable chip for secondary testing (False Positive) carries a low, bounded inspection cost.
3. **Static Limits vs. Lot Drift**: Absolute datasheet limits miss subtle batch anomalies. A wafer leakage current of $45\,\mu\text{A}$ is technically within a $50\,\mu\text{A}$ limit, but if the rest of its production lot averages $10\,\mu\text{A}$, it indicates severe localized process drift.

This project delivers **leakage-free machine learning pipelines** and a **live Streamlit QA Inspector Dashboard** to address these exact industrial pain points.

---

## 🏗️ Architecture & Core Components

```
SECOM-ML-MODEL/
├── qa_dashboard.py              # Interactive Streamlit QA Inspector (Pitch & Demo App)
├── production_pipeline_v2.py    # Canonical leakage-free ML pipeline (Stratified CV + F2 optimization)
├── train_classifier.py          # Baseline SMOTE + XGBoost trainer (exports secom_defect_model.pkl)
├── train_optimized.py           # Feature-selected benchmark (Top 30 sensors)
├── test_defective.py            # Targeted validation on known defective test samples
├── predict_random_samples.py    # Factory-floor inference simulator
├── secom_analysis.py            # Lot-aware MAD outlier & drift research script
├── requirements.txt             # Project dependencies
├── .streamlit/config.toml       # Custom light-theme dashboard design tokens
└── secom_defect_model.pkl       # Serialized production-ready model artifact
```

---

## ⚙️ Mathematical & Algorithmic Rationale

### 1. Robust Preprocessing & Sensor Hygiene
- **Missing-Value Filtering**: Sensors with $>40\%$ missing entries are dropped as defunct or disconnected channels.
- **Zero-Variance Pruning**: Invariant sensors carrying zero predictive signal are eliminated to prevent matrix singularity.
- **Median Imputation & Standardization**: `SimpleImputer(strategy='median')` followed by `StandardScaler` to handle extreme sensor outliers without introducing distribution bias.

### 2. Imbalance Remediation & Cost-Sensitive Learning
- **Class-Weighted XGBoost**: Uses `scale_pos_weight = N_neg / N_pos` ($\approx 14.15$) to penalize missed defects during gradient boosting iterations.
- **Synthetic Minority Over-Sampling (SMOTE)**: Synthesizes minority feature representations along $k$-nearest neighbor feature manifolds.

### 3. Out-of-Fold $F_2$-Score Threshold Optimization
Standard models make binary decisions at probability $p \ge 0.5$. In this project, decision thresholds are dynamically optimized using out-of-fold probabilities from `StratifiedKFold(n_splits=5)`:
$$\text{Best Threshold} = \arg\max_T F_2(T) \quad \text{where} \quad F_2 = \frac{5 \cdot \text{Precision} \cdot \text{Recall}}{4 \cdot \text{Precision} + \text{Recall}}$$
- By weighting Recall twice as heavily as Precision, the model operates at the calibrated industrial operating point (typically $p^* \approx 0.20 \text{--} 0.25$).

### 4. Lot-Relative Dynamic Anomaly Detection (MAD Modified Z-Score)
Rather than relying on static upper/lower spec limits, each die is evaluated against its manufacturing lot distribution:
$$M_i = \frac{0.6745 \cdot |x_i - \tilde{x}|}{\text{MAD}}$$
where $\tilde{x} = \text{median}(X_{\text{lot}})$ and $\text{MAD} = \text{median}(|X_{\text{lot}} - \tilde{x}|)$.
- Identifies outliers that are nominally compliant but statistically deviant within their batch.

### 5. Early Burn-In Degradation Forecasting
Accelerated life testing (Burn-in) lasts up to $168$ hours per lot. Using early checkpoint measurements at $0\,\text{h}$ and $24\,\text{h}$, a penalized Ridge regression model projects degradation at $168\,\text{h}$, enabling **$85\%$ chamber time savings** by aborting defective parts early.

---

## 🚀 Quickstart & Setup Guide

### 1. Prerequisites
- Python 3.10 or higher
- PowerShell (Windows) or Bash (macOS/Linux)
- *(Optional)* [Ollama](https://ollama.ai) with `llama3.1:latest` for local LLM diagnostics (the app runs seamlessly offline with deterministic rule-based analysis).

### 2. Installation

Clone the repository and install all dependencies:
```bash
git clone https://github.com/phanikumar822/SECOM-ML-MODEL.git
cd SECOM-ML-MODEL
pip install -r requirements.txt
```

---

## 🖥️ Running the Applications

### 1. Launch the QA Inspector Dashboard (Primary Pitch Demo)
```powershell
streamlit run qa_dashboard.py
```
- Open your browser at **`http://localhost:8501`**.
- **What to explore in the dashboard:**
  - **Lot-Relative Screening**: Inspect chips flagged for abnormal lot divergence even when passing datasheet limits.
  - **Burn-In Trajectory Charts**: View multi-stage drift curves ($0\,\text{h} \to 24\,\text{h} \to 96\,\text{h} \to 168\,\text{h}$) with predicted vs. actual safety bounds.
  - **Asymmetric Cost Matrix**: Observe how false negatives are penalized $10\times$ harder than false alarms.
  - **Root-Cause Analysis Engine**: View diagnostic reasoning per part.

### 2. Run the Full Production Pipeline (Leakage-Free Cross-Validation)
```powershell
python production_pipeline_v2.py
```
- Dynamically fetches the live UCI SECOM dataset.
- Executes automated cleaning, 5-fold out-of-fold probability estimation, optimal $F_2$ thresholding, and prints the full confusion matrix and classification report.

### 3. Real-Time Factory Floor Inference Simulator
```powershell
python predict_random_samples.py
```
- Simulates real-time wafer testing by randomly drawing chips from the held-out test split, displaying raw multi-sensor telemetry, and predicting pass/fail status with calibrated risk probabilities.

### 4. Targeted Defective-Chip Stress Test
```powershell
python test_defective.py
```
- Filters specifically for true defective chips in the test partition to evaluate model sensitivity and recall under challenging boundary conditions.

---

## 📊 Evaluation & Verification Benchmark

| Metric / Stage | Baseline Default Model ($T = 0.50$) | Production Pipeline ($F_2$-Optimized $T \approx 0.23$) |
| :--- | :--- | :--- |
| **Defect Detection (Recall)** | Low (~10% - 15%) | **High (~50% on raw test split)** |
| **Class Weighting** | Uniform ($1:1$) | **Cost-Sensitive ($14.15:1$)** |
| **Decision Boundary** | Rigid $0.5$ threshold | **Calibrated Out-Of-Fold $F_2$ Curve** |
| **Validation Strategy** | Single holdout (prone to leak) | **Stratified 5-Fold Cross-Validation** |
| **Lot-Level Screening** | Static bounds only | **Dynamic MAD Modified Z-Score** |

---

## 🔍 Hackathon Jury Evaluation Checklist

When reviewing this implementation, examine the following engineering highlights:

1. **Zero Data Leakage**: In `production_pipeline_v2.py`, imputers and scalers are fitted strictly inside training loops, and threshold tuning uses out-of-fold cross-validated predictions.
2. **Domain-Specific Metric Choice**: We explicitly reject raw accuracy in favor of $F_2$ score, precision-recall AUC, and asymmetric financial loss matrices.
3. **Graceful Fallbacks**: In `qa_dashboard.py`, the LLM root cause explainer gracefully falls back to deterministic rule-based explanations if local Ollama service is not running.
4. **Reproducibility**: All random seeds (`random_state=42`) and stratification splits are locked across training and inference scripts.

---

## 📜 License

This project is open-source under the [MIT License](LICENSE).
