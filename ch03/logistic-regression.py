# coding=utf-8
'''
@Title:logistic-regression

@Author: tyee.noprom@qq.com
@Time: 7/14/16 9:47 PM
'''
import matplotlib.pyplot as plt
import numpy as np

def sigmoid(z):
    '''
    sigmoid函数
    :param z: z
    :return: sigmoid的值
    '''
    return 1.0 / (1 + np.exp(-z))


def plot_sigmoid():
    '''
    绘制sigmoid函数图像
    :return:无
    '''
    # 创建一个从[-7, 7), 步长为0.1的数组
    z = np.arange(-7, 7, 0.1)
    phi_z = sigmoid(z)
    # 绘图
    plt.plot(z, phi_z)
    plt.axvline(0.0, color='k')
    plt.axhspan(0.0, 1.0, facecolor='1.0', alpha=1.0, ls='dotted')
    plt.axhline(y=0.5, ls='dotted', color='k')
    plt.yticks([0.0, 0.5, 1.0])
    plt.ylim(-0.1, 1.1)
    plt.xlabel('z')
    plt.ylabel('$\phi (z)$')
    plt.show()
