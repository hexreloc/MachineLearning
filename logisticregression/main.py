import numpy as np
def sigmoid(z):
    return 1 / ( 1 + np.exp(-z))

def gradient_step(X, y, theta):
    m = y.shape[0]
    return (X.T @ (sigmoid(X @ theta) - y)) / m

def gradient_descent(X, y, iters=1000, alpha = 0.01):

    # X_b will be [1, x1, x2]
    # theta will be [b, w1, w2]
    # so in gradient_step X_b and theta will do matrix multiplication and we get X @ theta = y and we squish that with sigmoid subtract with y and find error and we do matrix multiplication with X to find gradient

    x_b = np.c_[np.ones((X.shape[0],1)), X]
    theta = np.zeros(x_b.shape[1])
    print(theta.shape)
    
    for i in range(iters):
        theta -= alpha * gradient_step(x_b, y, theta)

        if i % 100 == 0:
            print(i, theta)

    return theta


def predict_proba(X, theta):

    x_b = np.c_[np.ones((X.shape[0],1)), X]
    return sigmoid(x_b @ theta)

def predict(X,theta):
    y_hat = predict_proba(X, theta)
    y_pred = (y_hat >= 0.5).astype(int)
    return y_pred

# training



import numpy as np

X = np.array([
    [1, 2, 3],
    [2, 1, 4],
    [3, 4, 1],
    [4, 2, 5],
    [5, 3, 2],
    [6, 5, 4],
    [7, 4, 3],
    [8, 6, 5],
    [9, 5, 6],
    [10, 7, 4]
])

y = np.array([
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    1
])

# theta = gradient_descent(X, y)
# print(theta)
#
# y_hat = predict(X, theta)
#
# print("Y_hat");
# print(y_hat)
# print("y_real")
# print(y)

def calculate_accuracy(y, y_hat):
    accuracy = np.mean((y_hat ==  y )*100)
    print("Accuracy", accuracy);


## Real data
from sklearn.datasets import load_breast_cancer
import pandas
data = load_breast_cancer(return_X_y = True)
X, y = data
m = y.shape[0]
train_size = int((m*0.8))
X_train = X[:train_size]
y_train = y[:train_size]

X_test = X[train_size:]
y_test = y[train_size:]

theta = gradient_descent(X_train, y_train)
y_train_hat = predict(X_train,theta)
y_test_hat = predict(X_test, theta)

calculate_accuracy(y_train, y_train_hat)
calculate_accuracy(y_test, y_test_hat)


