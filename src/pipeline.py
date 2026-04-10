import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.lstm_model import train_lstm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from src.models import RidgeRegression, LassoRegression, evaluate
from src.advanced_models import time_series_split, train_gradient_boosting, train_xgb

# =========================
# LOAD DATA
# =========================

df = pd.read_csv("data/processed/feature_engineered_traffic.csv")
df.columns = df.columns.str.strip()

if 'Vehicles' in df.columns:
    df.rename(columns={'Vehicles': 'traffic'}, inplace=True)

# =========================
# FEATURE ENHANCEMENT
# =========================

if "hour_sin" not in df.columns:
    df["hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)

if "dow_sin" not in df.columns:
    df["dow_sin"] = np.sin(2 * np.pi * df["DayOfWeek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["DayOfWeek"] / 7)

df["weekend_peak"] = df["Is_Weekend"] * df["Is_Peak_Hour"]

for col in ['DateTime', 'Junction', 'ID']:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)

df = df.dropna()

# =========================
# FEATURES
# =========================

X = df.drop(columns=['traffic'])
y = df['traffic']

X = X.select_dtypes(include=['number'])
feature_columns = X.columns

# =========================
# TRAINING + EVALUATION
# =========================

all_results = []

for X_train, X_test, y_train, y_test in time_series_split(X, y):

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    lr = LinearRegression().fit(X_train, y_train)
    rf = RandomForestRegressor(n_estimators=200).fit(X_train, y_train)
    gb = train_gradient_boosting(X_train, y_train)
    xgb = train_xgb(X_train, y_train)

    ridge = RidgeRegression()
    ridge.fit(X_train, y_train)

    lasso = LassoRegression()
    lasso.fit(X_train, y_train)

    lr_pred = lr.predict(X_test)
    rf_pred = rf.predict(X_test)
    gb_pred = gb.predict(X_test)
    xgb_pred = xgb.predict(X_test)
    ridge_pred = ridge.predict(X_test)
    lasso_pred = lasso.predict(X_test)

    ensemble_pred = (xgb_pred * 0.5 + rf_pred * 0.3 + gb_pred * 0.2)

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

results_df = pd.DataFrame()

for i, fold in enumerate(all_results):
    temp = pd.DataFrame(fold).T
    temp["Fold"] = i + 1
    results_df = pd.concat([results_df, temp])

avg_results = results_df.groupby(results_df.index).mean()

print("\n MODEL COMPARISON ")
print(avg_results)

avg_results["RMSE"].plot(kind="bar")
plt.show()

best_model_name = avg_results["R2"].idxmax()

# =========================
# FINAL TRAINING
# =========================

scaler_final = StandardScaler()
X_scaled = scaler_final.fit_transform(X)

models = {}

models["Linear"] = LinearRegression().fit(X_scaled, y)
models["RF"] = RandomForestRegressor(n_estimators=200).fit(X_scaled, y)
models["GB"] = train_gradient_boosting(X_scaled, y)
models["XGB"] = train_xgb(X_scaled, y)

ridge_final = RidgeRegression()
ridge_final.fit(X_scaled, y)
models["Ridge"] = ridge_final

lasso_final = LassoRegression()
lasso_final.fit(X_scaled, y)
models["Lasso"] = lasso_final

print("All models trained")

# =========================
# TRAIN LSTM
# =========================

from src.lstm_model import train_lstm

lstm_model, lstm_scaler, last_sequence = train_lstm(df)

print("LSTM model trained")

# =========================
# USER INPUT
# =========================

def get_user_input():

    hour = int(input("Hour: "))
    day = int(input("Day: "))
    dow = int(input("DayOfWeek: "))
    month = int(input("Month: "))
    year = int(input("Year: "))
    is_weekend = int(input("Is Weekend: "))
    is_peak = int(input("Is Peak Hour: "))

    lag_1 = df["traffic"].iloc[-1]
    rolling_mean_3 = df["traffic"].iloc[-3:].mean()

    data = pd.DataFrame([{
        "Hour": hour,
        "Day": day,
        "DayOfWeek": dow,
        "Month": month,
        "Year": year,
        "Is_Weekend": is_weekend,
        "Is_Peak_Hour": is_peak,
        "lag_1": lag_1,
        "rolling_mean_3": rolling_mean_3
    }])

    data["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    data["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    data["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    data["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    data["weekend_peak"] = is_weekend * is_peak

    return data[feature_columns]

# =========================
# PREDICTION FUNCTION
# =========================
def predict_all_models(user_data):

    user_scaled = scaler_final.transform(user_data)

    predictions = {}

    for name, model in models.items():
        predictions[name] = model.predict(user_scaled)[0]

    # =========================
    # LSTM PREDICTION (FIXED)
    # =========================

    # Prepare same feature structure as training
    lstm_input = user_data.copy()

    lstm_input["traffic"] = df["traffic"].iloc[-1]

    lstm_features = lstm_input[[
        "traffic", "Hour", "DayOfWeek",
        "Is_Weekend", "Is_Peak_Hour",
        "lag_1", "rolling_mean_3",
        "hour_sin", "hour_cos",
        "dow_sin", "dow_cos",
        "weekend_peak"
    ]].values

    # scale input
    lstm_features_scaled = lstm_scaler.transform(lstm_features)

    # append to last sequence
    new_seq = np.vstack([last_sequence[1:], lstm_features_scaled])

    lstm_tensor = torch.tensor(new_seq, dtype=torch.float32).unsqueeze(0)

    lstm_scaled_pred = lstm_model(lstm_tensor).item()

    # inverse scale (only traffic column)
    temp = np.zeros((1, lstm_features.shape[1]))
    temp[0, 0] = lstm_scaled_pred

    lstm_pred = lstm_scaler.inverse_transform(temp)[0][0]

    predictions["LSTM"] = lstm_pred

    # Hybrid
    predictions["Hybrid"] = 0.6 * predictions["XGB"] + 0.4 * lstm_pred

    return predictions
# =========================
# RUN
# =========================

while True:

    user_data = get_user_input()
    preds = predict_all_models(user_data)

    print("\n===== RESULTS =====")

    for k, v in preds.items():
        print(f"{k}: {round(v,2)}")

    if input("Again? (y/n): ") != "y":
        break