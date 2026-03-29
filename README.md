# Traffic-Prediction

A data-driven traffic prediction system developed using ML concepts to forecast traffic conditions based on various features. The project aims to improve urban mobility by predicting congestion levels and enabling smarter route planning.

This project was developed as part of a Pattern Recognition and Machine Learning (PRML) course. It emphasizes a rigorous, end-to-end machine learning pipeline, ranging from mathematical implementations from scratch to advanced time-series ensemble modeling.

##  Project Overview

The system addresses the traffic prediction problem through a dual-task formulation:
1. **Regression:** Predicting exact traffic volume and density.
2. **Classification:** Categorizing traffic congestion into Low, Medium, and High states.

### Key PRML Concepts Covered:
* **Linear Regression:** Implemented purely from scratch using mathematical operations (matrix/gradient descent).
* **Advanced Feature Engineering:** Lag features, rolling averages, and time-based feature extraction.
* **Regularization Techniques:** Ridge and Lasso Regression for controlling model complexity.
* **Non-Linear Decision Systems:** Decision Trees and Random Forests for complex pattern recognition.
* **Advanced Ensemble Learning:** Gradient Boosting (XGBoost) for high-performance forecasting.
* **Time-Series Validation:** Time-based Cross-Validation to prevent data leakage during model training.

---

##  Repository Structure

```text
TRAFFIC-PREDICTION/
│
├── data/                      # Local data storage (Ignored in Git)
│   ├── raw/                   # Unmodified original dataset
│   └── processed/             # Cleaned data ready for modeling
│
├── notebooks/                 # Jupyter notebooks for module experimentation
│   ├── 01_eda_and_baselines.ipynb       
│   ├── 02_feature_engineering.ipynb 
│   ├── 03_classification_models.ipynb     
│   └── 04_timeseries_ensemble.ipynb       
│
├── src/                       # Reusable Python scripts for the final pipeline
│   ├── __init__.py
│   ├── data_preprocessing.py  # Loading, cleaning, and time feature extraction
│   ├── feature_extraction.py  # Lag features, rolling averages, scaling
│   ├── models.py              # Model implementations (from scratch and sklearn)
│   └── pipeline.py            # End-to-end execution script
│
├── requirements.txt           # Project dependencies
├── .gitignore                 # Untracked files and directories (e.g., __pycache__, data/)
└── README.md                  # Project documentation

