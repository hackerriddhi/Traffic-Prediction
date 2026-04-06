import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error

from src.eda import advanced_eda
from src.models import (
    RidgeRegression,
    LassoRegression,
    evaluate,
    compare_regularization,
    check_multicollinearity
)
from src.feature_extraction import create_advanced_features, add_external_features
from src.advanced_models import (
    time_series_split,
    train_gradient_boosting,
    ensemble_predict,
    train_xgb
)

# LOAD + PREPROCESS

df = pd.read_csv("data/processed/cleaned_traffic.csv")

df.columns = df.columns.str.strip()

# Safe rename
if 'traffic' not in df.columns and 'Vehicles' in df.columns:
    df.rename(columns={'Vehicles': 'traffic'}, inplace=True)

# Feature engineering
df = create_advanced_features(df)

# Add weather + holiday features
df = add_external_features(df)

# Drop missing
df = df.dropna()

# Drop non-numeric columns
for col in ['DateTime', 'Junction', 'ID']:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)

# EDA

advanced_eda(df)

# TREND ANALYSIS 

if 'Hour' in df.columns:
    peak_hours = df.groupby('Hour')['traffic'].mean().sort_values(ascending=False).head(3)
    print("\nPeak Traffic Hours:\n", peak_hours)

if 'DayOfWeek' in df.columns:
    weekly_pattern = df.groupby('DayOfWeek')['traffic'].mean()
    print("\nWeekly Traffic Pattern:\n", weekly_pattern)

# FEATURE

X = df.drop(columns=['traffic'])
y = df['traffic']

X = X.select_dtypes(include=['number'])

# Multicollinearity check
check_multicollinearity(X)

# HELPER FUNCTIONS

def classify_traffic(y):
    return pd.cut(y, bins=3, labels=["Low", "Medium", "High"])


def plot_weights(model, feature_names, title):
    weights = getattr(model, 'W', None)

    if weights is None:
        print(f"{title}: No weights available")
        return

    plt.figure()
    plt.bar(feature_names, weights)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(2)
    plt.close()


def select_features_lasso(model, feature_names, threshold=0.01):
    important_features = []

    for coef, name in zip(model.W, feature_names):
        if abs(coef) > threshold:
            important_features.append(name)

    print("\nSelected Features:", important_features)
    return important_features

# TRAINING LOOP

all_results = []

for X_train, X_test, y_train, y_test in time_series_split(X, y):

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # REGULARIZATION MODELS

    ridge = RidgeRegression()
    lasso = LassoRegression()

    ridge.fit(X_train_scaled, y_train)
    lasso.fit(X_train_scaled, y_train)

    ridge_pred = ridge.predict(X_test_scaled)
    lasso_pred = lasso.predict(X_test_scaled)

    print("\n--- Regularization Models ---")
    print("Ridge MSE:", mean_squared_error(y_test, ridge_pred))
    print("Lasso MSE:", mean_squared_error(y_test, lasso_pred))

    # Feature importance
    plot_weights(ridge, X.columns, "Ridge Importance")
    plot_weights(lasso, X.columns, "Lasso Importance")

    # Feature selection
    selected_features = select_features_lasso(lasso, X.columns)
    print("Number of selected features:", len(selected_features))

    # Compare models
    compare_regularization(y_test, ridge_pred, lasso_pred)

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
    print("Linear ->", evaluate(y_test, lr_pred))
    print("Random Forest ->", evaluate(y_test, rf_pred))
    print("Gradient Boosting ->", evaluate(y_test, gb_pred))
    print("XGBoost ->", evaluate(y_test, xgb_pred))
    print("Ensemble ->", evaluate(y_test, ensemble_pred))
    
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