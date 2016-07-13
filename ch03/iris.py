# coding=utf-8
'''
@Title:Training a perceptron via scikit-learn

@Author: tyee.noprom@qq.com
@Time: 7/13/16 9:48 PM
'''

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

# 交叉验证
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# feature scaling for optimal performance
sc = StandardScaler()
# standardized the training data
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# 用神经元训练
ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
ppn.fit(X_train_std, y_train)

# 预测
y_pred = ppn.predict(X_test_std)
# 与真实值对比
print('Misclassified samples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))