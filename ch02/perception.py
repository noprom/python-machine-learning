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
