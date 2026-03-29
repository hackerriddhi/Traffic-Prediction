import numpy as np

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

            dW = (1/self.m)*(X.T.dot(y_pred - y)) + self.alpha*self.W
            db = (1/self.m)*np.sum(y_pred - y)

            self.W -= self.lr*dW
            self.b -= self.lr*db

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

            dW = (1/self.m)*(X.T.dot(y_pred - y)) + self.alpha*np.sign(self.W)
            db = (1/self.m)*np.sum(y_pred - y)

            self.W -= self.lr*dW
            self.b -= self.lr*db

    def predict(self, X):
        return X.dot(self.W) + self.b