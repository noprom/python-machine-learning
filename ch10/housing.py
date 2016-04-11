# coding=utf-8
'''
@title:Predicting Continuous Target Variables with Regression Analysis

@author: tyee.noprom@qq.com
Created on 4/11/16
'''
import pandas as pd
df = pd.read_csv('./housing.data', header = None, sep = '\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df.head()