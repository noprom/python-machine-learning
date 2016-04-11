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
    :return:
    '''
    sns.set(style='whitegrid', context='notebook')
    sns.pairplot(df[cols], size=2.5)
    plt.show()


def corelation_data():
    '''
    发现数据之间的关联关系
    :return:
    '''
    cm = np.corrcoef(df[cols].values.T)
    sns.set(font_scale=1.5)
    hm = sns.heatmap(cm, cbar=True, annot=True, fmt='.2f', annot_kws={'size': 15}, yticklabels=cols, xticklabels=cols)

look_data()
visualize_data()
corelation_data()