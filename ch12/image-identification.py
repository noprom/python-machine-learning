# coding=utf-8
'''
@Title:image-identification

@Author: tyee.noprom@qq.com
@Time: 5/21/16 4:16 PM
'''
import os
import struct
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
    return images, labels


def save_bin2csv(X_train, y_train, X_test, y_test):
    '''
    将读取到的二进制转化为csv文件
    :return:
    '''
    np.savetxt('train_img.csv', X_train, fmt='%i', delimiter=',')
    np.savetxt('train_labels.csv', y_train, fmt='%i', delimiter=',')
    np.savetxt('test_img.csv', X_test, fmt='%i', delimiter=',')
    np.savetxt('test_labels.csv', y_test, fmt='%i', delimiter=',')


def load_from_csv():
    '''
    从csv文件读取
    :return:
    '''
    X_train = np.genfromtxt('train_img.csv', dtype=int, delimiter=',')
    y_train = np.genfromtxt('train_labels.csv', dtype=int, delimiter=',')
    X_test = np.genfromtxt('test_img.csv', dtype=int, delimiter=',')
    y_test = np.genfromtxt('test_labels.csv', dtype=int, delimiter=',')


mnist_path = '../data/mnist'
X_train, y_train = load_mnist(mnist_path, kind='train')
print('Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))
# Rows: 60000, columns: 784
X_test, y_test = load_mnist(mnist_path, kind='t10k')
print('Rows: %d, columns: %d' % (X_test.shape[0], X_test.shape[1]))
# Rows: 10000, columns: 784

# plot the number
fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True, )
ax = ax.flatten()
for i in range(10):
    img = X_train[y_train == i][0].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

# In addition, let's also plot multiple examples of the same digit to see how different those handwriting examples really are:

fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True, )
ax = ax.flatten()
for i in range(25):
    img = X_train[y_train == 7][i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()
