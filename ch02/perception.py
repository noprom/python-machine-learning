# coding=utf-8
'''
@Title:神经元预测

@Author: tyee.noprom@qq.com
@Time: 4/12/16 3:13 PM
'''
import numpy as np


class Perception(object):
    """Perception classifier
        Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.
    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.
    errors_ : list
        Number of misclassifications in every epoch.
    """

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """Fit training data.
       Parameters
       ----------
        X : {array-like}, shape = [n_samples, n_features]
           Training vectors, where n_samples is the number of samples and
           n_features is the number of features.
        y : array-like, shape = [n_samples] Target values.
        Returns
        -------
        self : object
        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 1.0)
            self.errors_.append(errors)
        return self
