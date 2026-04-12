import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

def time_series_split(X, y, splits=5):
    tscv = TimeSeriesSplit(n_splits=splits)

    for train_idx, test_idx in tscv.split(X):
        yield X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]


def train_gradient_boosting(X_train, y_train):
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3
    )
    model.fit(X_train, y_train)
    return model


def ensemble_predict(models, X):
    preds = [model.predict(X) for model in models]
    return np.mean(preds, axis=0)

def train_xgb(X_train, y_train):
    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5
    )
    model.fit(X_train, y_train)
    return model