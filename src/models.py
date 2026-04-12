import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor


class RidgeRegression:
    def __init__(self, lr=0.001, epochs=1000, alpha=1):
        self.lr = lr
        self.epochs = epochs
        self.alpha = alpha

    def fit(self, X, y):
        self.m, self.n = X.shape
        self.W = np.zeros(self.n)
        self.b = 0

        for _ in range(self.epochs):
            y_pred = X.dot(self.W) + self.b

            dW = (1/self.m)*(X.T.dot(y_pred - y)) + (self.alpha/self.m)*self.W
            db = (1/self.m)*np.sum(y_pred - y)

            self.W -= self.lr * dW
            self.b -= self.lr * db

    def predict(self, X):
        return X.dot(self.W) + self.b


class LassoRegression:
    def __init__(self, lr=0.01, epochs=1000, alpha=1):
        self.lr = lr
        self.epochs = epochs
        self.alpha = alpha

    def fit(self, X, y):
        self.m, self.n = X.shape
        self.W = np.zeros(self.n)
        self.b = 0

        for _ in range(self.epochs):
            y_pred = X.dot(self.W) + self.b

            dW = (1/self.m)*(X.T.dot(y_pred - y)) + (self.alpha/self.m)*np.sign(self.W)
            db = (1/self.m)*np.sum(y_pred - y)

            self.W -= self.lr * dW
            self.b -= self.lr * db

    def predict(self, X):
        return X.dot(self.W) + self.b


def evaluate(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred)
    }


def compare_regularization(y_test, ridge_pred, lasso_pred):
    ridge_mse = mean_squared_error(y_test, ridge_pred)
    lasso_mse = mean_squared_error(y_test, lasso_pred)

    print("\n--- Regularization Comparison ---")
    print("Ridge MSE:", ridge_mse)
    print("Lasso MSE:", lasso_mse)

    if ridge_mse < lasso_mse:
        print("Ridge performs better (handles multicollinearity)")
    else:
        print("Lasso performs better (feature selection effect)")


def check_multicollinearity(X):
    X = X.copy()

    # Remove constant columns
    X = X.loc[:, X.nunique() > 1]

    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X.values, i)
        for i in range(X.shape[1])
    ]

    print("\n--- Multicollinearity (VIF) ---")
    print(vif_data)