# coding=utf-8
'''
@Title: config theano

@Author: tyee.noprom@qq.com
@Time: 7/25/16 11:07 PM
'''

import theano
from theano import scalar as T

print(theano.config.floatX)

theano.config.floatX = 'float32'

print(theano.config.device)
