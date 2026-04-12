# Traffic Prediction — PRML Course Project

A data-driven, end-to-end machine learning system that forecasts urban traffic volume and classifies congestion levels (Low / Medium / High) from historical traffic data. Built as a team project for a **Pattern Recognition and Machine Learning (PRML)** course, this system covers the full ML pipeline — from raw data cleaning to ensemble modeling — and includes an interactive prediction frontend.

---

## Team & Module Ownership

| Member | Module | Core Responsibility |
| **Radhika** | Data Pipeline + Statistical Analysis + Baseline Models | Data preprocessing, EDA, Linear Regression from scratch |
| **Riddhi** | Advanced EDA + Feature Engineering + Regularization | Lag features, rolling averages, Ridge & Lasso Regression |
| **Utkarsha** | Non-Linear Models + Decision Systems | Decision Tree, Random Forest, congestion classification |
| **Akshaya** | Advanced Modeling + Time-Series + Deployment | XGBoost, Gradient Boosting, ensemble pipeline, frontend |

---

## What This Project Does

* **Regression** — Predicts number of vehicles
* **Classification** — Labels traffic as Low / Medium / High

---

## PRML Concepts Covered

* Linear Regression (from scratch)
* Bias–Variance Tradeoff
* Ridge & Lasso
* Multicollinearity
* Decision Trees & Random Forest
* Gradient Boosting / XGBoost
* Feature Engineering
* Time-Series Cross-Validation
* Ensemble Learning

---

## Repository Structure

```
Traffic-Prediction/
│
├── data/
│   ├── raw/
│   │   └── traffic.csv
│   └── processed/
│       ├── cleaned_traffic.csv
│       └── feature_engineered_traffic.csv
│
├── notebooks/
│   ├── 01_eda_and_baselines.ipynb
│   └── 02_feature_engineering.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_extraction.py
│   ├── eda.py
│   ├── models.py
│   ├── advanced_models.py
│   └── pipeline.py
│
├── requirements.txt
└── README.md
```

---

## How to Run

```bash
git clone https://github.com/hackerriddhi/Traffic-Prediction
cd Traffic-Prediction
pip install -r requirements.txt
python -m src.pipeline
```

---

## Dataset

* `DateTime` — timestamp
* `Junction` — location
* `Vehicles` — traffic count

---

## Models Used

* Linear Regression
* Ridge / Lasso
* Decision Tree
* Random Forest
* XGBoost
* Ensemble

---

## 🌐 Applications

* Smart traffic signals
* Route optimization
* Demand prediction
* Urban planning

---

