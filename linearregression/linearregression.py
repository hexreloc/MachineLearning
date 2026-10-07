import numpy as np
from numpy import ndarray


class LinearModel:
    def __init__(self):
        self.iterations = 1000
        self.L = 0.01

        self.w = 0.0
        self.b = 0.0

        self.x_mean = 0.0
        self.x_std = 1.0

    # normalization of data
    def fit(self, x_train: ndarray, y_train: ndarray, iterations=10000):
        x_train = np.asarray(x_train, dtype=float).flatten()
        y_train = np.asarray(y_train, dtype=float).flatten()

        self.iterations = iterations

        self.x_mean = np.mean(x_train)
        self.x_std = np.std(x_train)

        x = (x_train - self.x_mean) / self.x_std

        self.x = x
        self.y = y_train

        self.train()

        # Convert parameters back to original x scale
        self.w = self.w / self.x_std
        self.b = self.b - self.w * self.x_mean

    def gradient_descent(self):
        m = len(self.x)

        prediction = self.w * self.x + self.b
        error = prediction - self.y

        w_gradient = (2 / m) * np.sum(self.x * error)
        b_gradient = (2 / m) * np.sum(error)

        self.w -= self.L * w_gradient
        self.b -= self.L * b_gradient

    def train(self):
        for _ in range(self.iterations):
            self.gradient_descent()

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        return self.w * x + self.b
