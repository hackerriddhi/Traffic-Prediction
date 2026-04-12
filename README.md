# Traffic Prediction — PRML Course Project

A data-driven, end-to-end machine learning system that forecasts urban traffic volume and classifies congestion levels (**Low / Medium / High**) from historical traffic data. Built as part of a **Pattern Recognition and Machine Learning (PRML)** course, this project covers the full ML pipeline — from preprocessing to ensemble modeling.

---

## Team & Module Ownership

| Member   | Module                               | Core Responsibility                                       |
| -------- | ------------------------------------ | --------------------------------------------------------- |
| Radhika  | Data Pipeline + Statistical Analysis | Data preprocessing, EDA, Linear Regression (from scratch) |
| Riddhi   | Feature Engineering + Regularization | Lag features, rolling averages, Ridge & Lasso             |
| Utkarsha | Non-Linear Models                    | Decision Tree, Random Forest, classification              |
| Akshaya  | Advanced Models + Deployment         | XGBoost, Gradient Boosting, ensemble pipeline             |

---

## What This Project Does

* **Regression** — Predicts number of vehicles
* **Classification** — Labels traffic as Low / Medium / High

---

## PRML Concepts Covered

* Linear Regression (from scratch)
* Bias–Variance Tradeoff
* Ridge & Lasso Regression
* Multicollinearity
* Decision Trees & Random Forest
* Gradient Boosting / XGBoost
* Feature Engineering (lag, rolling features)
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

## ⚙️ Installation

```bash
git clone https://github.com/hackerriddhi/Traffic-Prediction
cd Traffic-Prediction
pip install -r requirements.txt
```

---

## ▶️ Run the Project

```bash
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
