# coding=utf-8
'''
@Title:Predicting Continuous Target Variables with Regression Analysis

@Author: tyee.noprom@qq.com
@Time: 4/11/16
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('./housing.data', header=None, sep='\s+')
cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']


def look_data():
    '''
    查看数据
    :return:
    '''
    df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT',
                  'MEDV']
    print(df.head())


def visualize_data():
    '''
    可视化数据
    '''
    sns.set(style='whitegrid', context='notebook')
    sns.pairplot(df[cols], size=2.5)
    plt.show()


def corelation_data():
    '''
    发现数据之间的关联关系
    '''
    cm = np.corrcoef(df[cols].values.T)
    sns.set(font_scale=1.5)
    hm = sns.heatmap(cm, cbar=True, annot=True, fmt='.2f', annot_kws={'size': 15}, yticklabels=cols, xticklabels=cols)
    plt.show()

look_data()
visualize_data()
corelation_data()


class LinearRegressionGD(object):
    '''
    线性回归模型
    使用最小二乘法
    '''

    def __init__(self, eta=0.001, n_iter=20):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return self.net_input(X)
