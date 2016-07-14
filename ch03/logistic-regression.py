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
