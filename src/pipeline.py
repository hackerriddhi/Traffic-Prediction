import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from src.models import RidgeRegression, LassoRegression, evaluate
from src.advanced_models import time_series_split, train_gradient_boosting, train_xgb

# LOAD DATA

df = pd.read_csv("data/processed/feature_engineered_traffic.csv")
df.columns = df.columns.str.strip()

if 'Vehicles' in df.columns:
    df.rename(columns={'Vehicles': 'traffic'}, inplace=True)

# Drop unnecessary columns
for col in ['DateTime', 'Junction', 'ID']:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)

df = df.dropna()

# FEATURES

X = df.drop(columns=['traffic'])
y = df['traffic']

X = X.select_dtypes(include=['number'])
feature_columns = X.columns

# TRAINING + EVALUATION

all_results = []

for X_train, X_test, y_train, y_test in time_series_split(X, y):

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Models
    lr = LinearRegression().fit(X_train, y_train)
    rf = RandomForestRegressor(n_estimators=100).fit(X_train, y_train)
    gb = train_gradient_boosting(X_train, y_train)
    xgb = train_xgb(X_train, y_train)

    ridge = RidgeRegression()
    ridge.fit(X_train, y_train)

    lasso = LassoRegression()
    lasso.fit(X_train, y_train)

    # Predictions
    lr_pred = lr.predict(X_test)
    rf_pred = rf.predict(X_test)
    gb_pred = gb.predict(X_test)
    xgb_pred = xgb.predict(X_test)
    ridge_pred = ridge.predict(X_test)
    lasso_pred = lasso.predict(X_test)

    # Ensemble
    ensemble_pred = (
        lr_pred + rf_pred + gb_pred + xgb_pred + ridge_pred + lasso_pred
    ) / 6

    # Store results
    fold_results = {
        "Linear": evaluate(y_test, lr_pred),
        "RF": evaluate(y_test, rf_pred),
        "GB": evaluate(y_test, gb_pred),
        "XGB": evaluate(y_test, xgb_pred),
        "Ridge": evaluate(y_test, ridge_pred),
        "Lasso": evaluate(y_test, lasso_pred),
        "Ensemble": evaluate(y_test, ensemble_pred)
    }

    all_results.append(fold_results)

# FINAL RESULTS

results_df = pd.DataFrame()

for i, fold in enumerate(all_results):
    temp = pd.DataFrame(fold).T
    temp["Fold"] = i + 1
    results_df = pd.concat([results_df, temp])

avg_results = results_df.groupby(results_df.index).mean()

print("\n MODEL COMPARISON")
print(avg_results)

# Plot RMSE
avg_results["RMSE"].plot(kind="bar", title="Model RMSE Comparison")
plt.ylabel("RMSE")
plt.show()

# BEST MODEL

best_model_name = avg_results["R2"].idxmax()


# FINAL TRAIN ALL MODELS

scaler_final = StandardScaler()
X_scaled = scaler_final.fit_transform(X)

models = {}

models["Linear"] = LinearRegression().fit(X_scaled, y)
models["RF"] = RandomForestRegressor(n_estimators=100).fit(X_scaled, y)
models["GB"] = train_gradient_boosting(X_scaled, y)
models["XGB"] = train_xgb(X_scaled, y)

ridge_final = RidgeRegression()
ridge_final.fit(X_scaled, y)
models["Ridge"] = ridge_final

lasso_final = LassoRegression()
lasso_final.fit(X_scaled, y)
models["Lasso"] = lasso_final

print(" All models trained")

# USER INPUT

def get_user_input():
    print("\n====== ENTER TRAFFIC DETAILS ======")

    hour = int(input("Hour (0-23): "))
    day = int(input("Day (1-31): "))
    dayofweek = int(input("DayOfWeek (0=Mon,6=Sun): "))
    month = int(input("Month (1-12): "))
    year = int(input("Year: "))
    is_weekend = int(input("Is Weekend (0/1): "))
    is_peak = int(input("Is Peak Hour (0/1): "))

    lag_1 = df["traffic"].iloc[-1]
    rolling_mean_3 = df["traffic"].iloc[-3:].mean()

    user_data = pd.DataFrame([{
        "Hour": hour,
        "Day": day,
        "DayOfWeek": dayofweek,
        "Month": month,
        "Year": year,
        "Is_Weekend": is_weekend,
        "Is_Peak_Hour": is_peak,
        "lag_1": lag_1,
        "rolling_mean_3": rolling_mean_3
    }])

    user_data = user_data[feature_columns]

    return user_data

# PREDICT ALL MODELS

def predict_all_models(user_data):

    user_scaled = scaler_final.transform(user_data)

    predictions = {}

    for name, model in models.items():
        predictions[name] = model.predict(user_scaled)[0]

    predictions["Ensemble"] = np.mean(list(predictions.values()))

    return predictions

# TRAFFIC CATEGORY

def traffic_category(value):
    if value < 10:
        return "Low Traffic "
    elif value < 20:
        return "Medium Traffic "
    else:
        return "High Traffic "


while True:
    user_data = get_user_input()

    preds = predict_all_models(user_data)

    print("\n====== MODEL PREDICTIONS ======")

    for model_name, value in preds.items():
        category = traffic_category(value)
        print(f"{model_name}: {round(value,2)} → {category}")

    cont = input("\nAgain? (y/n): ")
    if cont.lower() != 'y':
        break