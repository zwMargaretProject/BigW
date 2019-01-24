import numpy as np

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

class LogisticRegression(object):
    def __init__(self):
        pass
    def fit(self, x, y, learn_rate=0.0005):
        point_num, future_num = np.shape(x)
        new_x = np.ones(shape=(point_num, future_num + 1)) # 多一列x0，全部设为1
        new_x[:, 1:] = x
        self.theta = np.ones(shape=(future_num + 1, 1))

        x_mat = np.mat(new_x)
        y_mat = np.mat(y).T
        J = []
        for i in range(800):
            h = sigmoid(np.dot(x_mat, self.theta))
            cost = np.sum([ a * -np.log(b) + (1 - a) * -np.log(1 - b)  for a, b in zip(y_mat, h)])
            J.append(cost)
            self.theta -= learn_rate * x_mat.T * (h - y_mat)
        plt.plot(J)
        plt.show()

    def predict(self, row):
        row = np.array([1] + row)
        result = sigmoid(np.dot(row, self.theta))
        return 1 if result > 0.5 else 0