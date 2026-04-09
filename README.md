#  Traffic Prediction — PRML Course Project

A data-driven, end-to-end machine learning system that forecasts urban traffic volume and classifies congestion levels (Low / Medium / High) from historical traffic data. Built as a team project for a **Pattern Recognition and Machine Learning (PRML)** course, this system covers the full ML pipeline — from raw data cleaning to ensemble modeling — and includes an interactive prediction frontend.

---

##  Team & Module Ownership

| Member | Module | Core Responsibility |
| **Radhika** | Data Pipeline + Statistical Analysis + Baseline Models | Data preprocessing, EDA, Linear Regression from scratch |
| **Riddhi** | Advanced EDA + Feature Engineering + Regularization | Lag features, rolling averages, Ridge & Lasso Regression |
| **Utkarsha** | Non-Linear Models + Decision Systems | Decision Tree, Random Forest, congestion classification |
| **Akshaya** | Advanced Modeling + Time-Series + Deployment | XGBoost, Gradient Boosting, ensemble pipeline, frontend |

---

##  What This Project Does

The system solves two complementary prediction tasks simultaneously:

- **Regression** — Predicts the exact number of vehicles at a given time and junction.
- **Classification** — Labels traffic conditions as **Low**, **Medium**, or **High** congestion using quantile-based thresholds.

The user  provides input features such as Hour , day of week, month , year , isWeekend. The pipeline returns both a predicted vehicle count and a congestion category.

---

##  PRML Concepts Covered

This project is intentionally comprehensive to reflect core PRML topics:

- **Linear Regression from scratch** — implemented using gradient descent and matrix operations 
- **Bias–Variance Tradeoff** — explored during baseline vs. regularized model comparison
- **Ridge & Lasso Regression** — regularization effects on coefficient shrinkage and feature selection
- **Multicollinearity Analysis** — VIF-based feature diagnosis before model training
- **Decision Trees & Random Forests** — non-linear pattern recognition with hyperparameter tuning
- **Gradient Boosting / XGBoost** — high-performance sequential ensemble learning
- **Feature Engineering** — lag features, rolling averages, time decomposition
- **Time-Series Cross-Validation** — prevents data leakage; respects temporal ordering during splits
- **Ensemble Learning** — combines predictions from multiple models for improved robustness
- **Traffic Congestion Classification** — transforms regression output into actionable categories

---

##  Repository Structure

```
Traffic-Prediction/
│
├── data/
│   ├── raw/
│   │   └── traffic.csv                     # Original, unmodified dataset
│   └── processed/
│       ├── cleaned_traffic.csv             # After preprocessing
│       └── feature_engineered_traffic.csv  # After feature engineering
│
├── notebooks/
│   ├── 01_eda_and_baselines.ipynb          #  EDA, stats, Linear Regression
│   ├── 02_feature_engineering.ipynb        # lag features, Ridge, Lasso
│   ├── 03_classification_models.ipynb      #  Decision Tree, Random Forest, classification
│   └── 04_timeseries_ensemble.ipynb        #  XGBoost, ensemble, time-series CV
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py               # Data loading, cleaning, time feature extraction
│   ├── feature_extraction.py               # Lag features, rolling averages, scaling
│   ├── eda.py                              # Advanced EDA functions and visualizations
│   ├── models.py                           # Linear, Ridge, Lasso (from scratch + sklearn)
│   ├── advanced_models.py                  # XGBoost, Gradient Boosting, ensemble logic
│   └── pipeline.py                         # End-to-end execution script
│
│
├── requirements.txt                        # Full dependency list with pinned versions
├── .gitignore
└── README.md
```

---

##  Pipeline Overview

```
Raw CSV Data  (data/raw/traffic.csv)
      ↓
Data Preprocessing
  → Handle missing values, parse datetime, extract time features
      ↓
Exploratory Data Analysis
  → Mean/variance of traffic, peak hour detection, correlation heatmaps,
    traffic vs. weather, traffic vs. holidays, weekly behaviour
      ↓
Feature Engineering
  → Lag features (t-1, t-2, ...), rolling averages, external features
      ↓
Baseline Model
  → Linear Regression (from scratch)
      ↓
Regularized Models
  → Ridge Regression  |  Lasso Regression
  → Compare coefficient shrinkage and generalisation
      ↓
Tree-Based Models
  → Decision Tree  →  Random Forest
  → Hyperparameter tuning, feature importance
      ↓
Ensemble Models
  → XGBoost / Gradient Boosting
  → Time-based cross-validation
  → Ensemble prediction (combine all models)
      ↓
Dual Output
  → Predicted vehicle count  (regression)
  → Congestion label: Low / Medium / High  (classification)
      ↓
  →→ Model generates predictions on test data via pipeline execution
```

---

##  Setup & Installation

### Prerequisites

- Python 3.10 or higher
- `pip` package manager

### Step 1 — Clone the Repository

```bash
git clone https://github.com/hackerriddhi/Traffic-Prediction
cd Traffic-Prediction
```


### Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

The key packages used in this project are:

| Package | Version | Purpose |
|---|---|---|
| `numpy` | 2.4.4 | Numerical operations, Linear Regression from scratch |
| `pandas` | 3.0.1 | Data loading, cleaning, feature engineering |
| `scikit-learn` | 1.8.0 | Ridge, Lasso, Random Forest, metrics, cross-validation |
| `xgboost` | 3.2.0 | Gradient Boosting / XGBoost model |
| `matplotlib` | 3.10.8 | Plotting, 3D model comparison graphs |
| `seaborn` | 0.13.2 | Correlation heatmaps, distribution plots |
| `scipy` | 1.17.1 | Statistical analysis utilities |
| `jupyterlab` | 4.5.6 | Running the `.ipynb` notebooks |

> **Note:** The `requirements.txt` contains pinned versions of all transitive dependencies for full reproducibility. If you encounter conflicts, install only the core packages listed above.

---

##  How to Run

### Option A — Run the Full Pipeline (recommended first run)

This executes the entire ML pipeline end-to-end: preprocessing → feature engineering → EDA → all models → predictions.

```bash
cd Traffic-Prediction
python -m src.pipeline
```

Make sure you are in the `Traffic-Prediction/` root directory when running this so that relative data paths (`data/processed/...`) resolve correctly.

### Option B — Explore the Notebooks

Launch JupyterLab and open the notebooks in order:

```bash
jupyter lab
```

Then open the notebooks in sequence:

1. `notebooks/01_eda_and_baselines.ipynb` — Start here for data understanding and Linear Regression
2. `notebooks/02_feature_engineering.ipynb` — Feature engineering and regularization
3. `notebooks/03_classification_models.ipynb` — Tree-based models and congestion classification
4. `notebooks/04_timeseries_ensemble.ipynb` — XGBoost, ensemble, and time-series evaluation



---

##  Dataset

**Source:** `data/raw/traffic.csv`

The dataset contains hourly traffic observations across multiple junctions. Key columns include:

- `DateTime` — Timestamp of the observation
- `Junction` — Junction identifier (1–4)
- `Vehicles` — Number of vehicles recorded (prediction target)
- `ID` — Row identifier

During preprocessing, `DateTime` is parsed into time features (hour, day of week, month, is_weekend, is_holiday). `Vehicles` is renamed to `traffic` internally for consistency across modules.

---

##  Models Used


| Model             | Purpose             |
| ----------------- | ------------------- |
| Linear Regression | Baseline            |
| Ridge / Lasso     | Regularization      |
| Decision Tree     | Non-linear patterns |
| Random Forest     | Ensemble learning   |
| XGBoost           | Advanced boosting   |
| Ensemble          | Final prediction    |
---

## 🌐 Real-World Applications

The system is designed to support:

- **Smart traffic signal control** — dynamically adjust signal timing based on predicted congestion
- **Navigation & routing** — feed congestion forecasts into GPS route recommendations
- **Ride-sharing demand planning** — predict surge zones ahead of time
- **Urban infrastructure planning** — identify chronically congested junctions for long-term investment

---

