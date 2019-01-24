import numpy as np

class LinearRegression(object):
    def __init__(self):
        self.b = []

    def fit(self, x: list, y: list):
        point_num, future_num = np.shape(x)
        tmpx = np.ones(shape=(point_num, future_num + 1))
        tmpx[:,1 :] = x

        x_mat = np.mat(tmpx)
        y_mat = np.mat(y).T
        xT = x_mat.T
        self.b = (xT * x_mat).I * xT * y_mat

    def predict(self, x):
        return np.mat([1] + x) * self.b