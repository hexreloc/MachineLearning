import numpy as np
from numpy import ndarray

class LinearModel:
    def __init__(self):
        self.iterations = 0
        self.x  = []
        self.y  = []
        self.L = 0.001
        self.w = 0
        self.b = 0
        self.m = 0

    def fit(self, x_train:ndarray, y_train:ndarray,iterations=1000):
        self.x = x_train
        self.y = y_train
        self.iterations = iterations

    def gradient_descent(self):
        w_now = self.w
        b_now = self.b
        m = len(self.x)
        sum_w = 0
        sum_b = 0
        for i in range(m):
            sum_w += (self.x[i])*(self.y[i] - w_now*self.x[i] - b_now)
            sum_b += (self.y[i] - w_now*self.x[i] - b_now)
        w_gradient = -(1/m)*sum_w 
        b_gradient = -(1/m)*sum_b
        self.w = w_now - self.L*w_gradient
        self.b = b_now - self.L*b_gradient
        return self.w, self.b

    def train(self, w =0, b=0):
        for i in range(self.iterations):
            w,b = self.gradient_descent()
        self.w = w
        self.b = b
    
    def predict(self,x):
        y = self.w * x + self.b
        return y


model = LinearModel()
x_train = np.array([1.0, 2.0])   
y_train = np.array([300.0, 500.0])
model.fit(x_train, y_train, iterations=10000)
model.train()
print(model.predict(2))

