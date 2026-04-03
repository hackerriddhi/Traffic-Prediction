# src/pipeline.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from src.feature_extraction import create_advanced_features
from src.advanced_models import (
    time_series_split,
    train_gradient_boosting,
    ensemble_predict
)
from src.models import evaluate
import matplotlib.pyplot as plt
from src.advanced_models import train_xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ======================
# LOAD + PREPROCESS
# ======================
df = pd.read_csv("data/processed/cleaned_traffic.csv")

df.columns = df.columns.str.strip()

if 'Vehicles' in df.columns:
    df.rename(columns={'Vehicles': 'traffic'}, inplace=True)

df = create_advanced_features(df)
df = df.dropna()

# Drop non-numeric
for col in ['DateTime', 'Junction', 'ID']:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)

# ======================
# FEATURES
# ======================
X = df.drop(columns=['traffic'])
y = df['traffic']

X = X.select_dtypes(include=['number'])


# ======================
# TRAINING LOOP (TIME SERIES)
# ======================
all_results = []

def classify_traffic(y):
    return pd.cut(y, bins=3, labels=["Low", "Medium", "High"])

for X_train, X_test, y_train, y_test in time_series_split(X, y):
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Models
    lr = LinearRegression().fit(X_train, y_train)
    rf = RandomForestRegressor(n_estimators=100).fit(X_train, y_train)
    gb = train_gradient_boosting(X_train, y_train)
    xgb = train_xgb(X_train, y_train)   # NEW
    
    # Convert to classification labels
    y_train_cls = classify_traffic(y_train)
    y_test_cls = classify_traffic(y_test)

# Train classifier
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train_cls)

# Predict
    y_pred_cls = clf.predict(X_test)

    print("Classification Sample:", y_pred_cls[:10])

    # Predictions
    lr_pred = lr.predict(X_test)
    rf_pred = rf.predict(X_test)
    gb_pred = gb.predict(X_test)
    xgb_pred = xgb.predict(X_test)      # NEW

    # Ensemble
    ensemble_pred = ensemble_predict([lr, rf, gb, xgb], X_test)
    print("Classification Accuracy:",
      accuracy_score(y_test_cls, y_pred_cls))
    # Evaluation
    print("\n--- Fold Results ---")
    print("Linear:", evaluate(y_test, lr_pred))
    print("Random Forest:", evaluate(y_test, rf_pred))
    print("Gradient Boosting:", evaluate(y_test, gb_pred))
    print("XGBoost:", evaluate(y_test, xgb_pred))   # NEW
    print("Ensemble:", evaluate(y_test, ensemble_pred))

    all_results.append(evaluate(y_test, ensemble_pred))
    
    # 🔥 VISUALIZATION (REQUIRED)
    plt.figure()
    plt.plot(y_test.values, label="Actual")
    plt.plot(ensemble_pred, label="Predicted")
    plt.legend()
    plt.title("Traffic Prediction (Fold)")
    plt.show()

print("\nFinal Results:", all_results)