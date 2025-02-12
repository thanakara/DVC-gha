import numpy as np

class LinearRegression(object):
    def __init__(self, name=None):
        self.is_built = False

    def __call__(self, X):
        if self.is_built:
            Y_pred = np.hstack([np.ones([len(X), 1]), X]) @ self.best_W
            return Y_pred

    def fit(self, X, y):
        if not self.is_built:
            self.X_s = np.hstack([np.ones([len(X), 1]), X])
            self.W_s = np.zeros([X.shape[-1] + 1, 1])
            self.best_W = np.linalg.pinv(self.X_s) @ y
            self.is_built = True