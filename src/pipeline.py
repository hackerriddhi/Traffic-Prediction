# src/pipeline.py

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error

from src.feature_extraction import create_advanced_features
from src.advanced_models import (
    time_series_split,
    train_gradient_boosting,
    ensemble_predict,
    train_xgb
)
from src.models import RidgeRegression, LassoRegression, evaluate



# LOAD + PREPROCESS

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



# FEATURES

X = df.drop(columns=['traffic'])
y = df['traffic']

X = X.select_dtypes(include=['number'])



# HELPER FUNCTIONS

def classify_traffic(y):
    return pd.cut(y, bins=3, labels=["Low", "Medium", "High"])


def plot_weights(model, feature_names, title):
    plt.figure()
    plt.bar(feature_names, model.W)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(2)
    plt.close()



# TRAINING LOOP 

all_results = []

for X_train, X_test, y_train, y_test in time_series_split(X, y):


    # SCALING 

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    # BASELINE MODELS 

    ridge = RidgeRegression()
    lasso = LassoRegression()

    ridge.fit(X_train_scaled, y_train)
    lasso.fit(X_train_scaled, y_train)

    ridge_pred = ridge.predict(X_test_scaled)
    lasso_pred = lasso.predict(X_test_scaled)

    print("\n--- Baseline Models ---")
    print("Ridge MSE:", mean_squared_error(y_test, ridge_pred))
    print("Lasso MSE:", mean_squared_error(y_test, lasso_pred))

    # Feature importance
    plot_weights(ridge, X.columns, "Ridge Importance")
    plot_weights(lasso, X.columns, "Lasso Importance")


    # ADVANCED MODELS 

    lr = LinearRegression().fit(X_train_scaled, y_train)
    rf = RandomForestRegressor(n_estimators=100).fit(X_train_scaled, y_train)
    gb = train_gradient_boosting(X_train_scaled, y_train)
    xgb = train_xgb(X_train_scaled, y_train)

    # Predictions
    lr_pred = lr.predict(X_test_scaled)
    rf_pred = rf.predict(X_test_scaled)
    gb_pred = gb.predict(X_test_scaled)
    xgb_pred = xgb.predict(X_test_scaled)

    # Ensemble
    ensemble_pred = ensemble_predict([lr, rf, gb, xgb], X_test_scaled)

    print("\n--- Advanced Models ---")
    print("Linear:", evaluate(y_test, lr_pred))
    print("Random Forest:", evaluate(y_test, rf_pred))
    print("Gradient Boosting:", evaluate(y_test, gb_pred))
    print("XGBoost:", evaluate(y_test, xgb_pred))
    print("Ensemble:", evaluate(y_test, ensemble_pred))


    # CLASSIFICATION
  
    y_train_cls = classify_traffic(y_train)
    y_test_cls = classify_traffic(y_test)

    clf = RandomForestClassifier()
    clf.fit(X_train_scaled, y_train_cls)

    y_pred_cls = clf.predict(X_test_scaled)

    print("Classification Accuracy:",
          accuracy_score(y_test_cls, y_pred_cls))

    # VISUALIZATION

    plt.figure()
    plt.plot(y_test.values, label="Actual")
    plt.plot(ensemble_pred, label="Predicted")
    plt.legend()
    plt.title("Traffic Prediction (Fold)")
    plt.show(block=False)
    plt.pause(2)
    plt.close()

    # Store results
    all_results.append(evaluate(y_test, ensemble_pred))


print("\nFinal Results:", all_results)