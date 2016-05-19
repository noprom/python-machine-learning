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
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split


class LinearRegressionGD(object):
    '''
    线性回归模型,梯度下降
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


df = pd.read_csv('./housing.data', header=None, sep='\s+')
cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT',
              'MEDV']
X = df[['RM']].values
y = df['MEDV'].values


def look_data():
    '''
    查看数据
    :return:
    '''
    df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT',
                  'MEDV']
    # print(df.head())


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
    # 热力图
    hm = sns.heatmap(cm, cbar=True, annot=True, fmt='.2f', annot_kws={'size': 15}, yticklabels=cols, xticklabels=cols)
    plt.show()


def linear_model():
    '''
    线性回归
    :return:
    '''
    from sklearn.preprocessing import StandardScaler
    sc_x = StandardScaler()
    sc_y = StandardScaler()
    X_std = sc_x.fit_transform(X)
    y_std = sc_y.fit_transform(y)
    lr = LinearRegressionGD()
    lr.fit(X_std, y_std)
    plt.plot(range(1, lr.n_iter + 1), lr.cost_)
    plt.ylabel('SSE')
    plt.xlabel('Epoch')
    plt.show()
    # RM与MEDV关系图
    lin_regplot(X_std, y_std, lr)
    plt.xlabel('Average number of rooms [RM] (standardized)')
    plt.ylabel('Price in $1000\'s [MEDV] (standardized)')
    plt.show()


def lin_regplot(X, y, model):
    '''
    画线性关系图
    :param X:
    :param y:
    :param model:
    :return:
    '''
    plt.scatter(X, y, c='blue')
    plt.plot(X, model.predict(X), color='red')
    return None


def ransac_fit(X, y):
    '''
    一个强健的fit
    :return:
    '''
    from sklearn.linear_model import RANSACRegressor
    ransac = RANSACRegressor(LinearRegression(),
                             max_trials=100,
                             min_samples=50,
                             residual_metric=lambda x: np.sum(np.abs(x), axis=1),
                             residual_threshold=5.0,
                             random_state=0)
    ransac.fit(X, y)
    # 输出斜率|截距等数据
    print('Slope: %.3f' % ransac.estimator_.coef_[0])
    print('Intercept: %.3f' % ransac.estimator_.intercept_)
    # plot
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)
    line_X = np.arange(3, 10, 1)
    line_y_ransac = ransac.predict(line_X[:, np.newaxis])
    plt.scatter(X[inlier_mask], y[inlier_mask],
                c='blue', marker='o', label='Inliers')
    plt.scatter(X[outlier_mask], y[outlier_mask],
                c='lightgreen', marker='s', label='Outliers')
    plt.plot(line_X, line_y_ransac, color='red')
    plt.xlabel('Average number of rooms [RM]')
    plt.ylabel('Price in $1000\'s [MEDV]')
    plt.legend(loc='upper left')
    plt.show()


def multi_regression():
    '''
    多元回归
    :return:
    '''
    from sklearn.cross_validation import train_test_split
    X = df.iloc[:, :-1].values
    y = df['MEDV'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    slr = LinearRegression()
    slr.fit(X_train, y_train)
    y_train_pred = slr.predict(X_train)
    y_test_pred = slr.predict(X_test)
    # 计算Mean Squared Error (MSE)
    from sklearn.metrics import mean_squared_error
    print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
    # MSE train: 19.958, test: 27.196 => over fitting

    # 计算R*R
    # If R*R =1, the model  ts the data perfectly with a corresponding MSE = 0 .
    from sklearn.metrics import r2_score
    print('R^2 train: %.3f, test: %.3f' % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)))


    # plot
    plt.scatter(y_train_pred, y_train_pred - y_train, c='blue', marker='o', label='Training data')
    plt.scatter(y_test_pred, y_test_pred - y_test, c='lightgreen', marker='s', label='Test data')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red')
    plt.xlim([-10, 50])
    plt.show()


# look_data()
# visualize_data()
# corelation_data()
# linear_model()
# ransac_fit(X, y)
multi_regression()
