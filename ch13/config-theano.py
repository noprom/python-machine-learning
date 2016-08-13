# coding=utf-8
'''
@Title: config theano

@Author: tyee.noprom@qq.com
@Time: 7/25/16 11:07 PM
'''

import theano
import numpy as np
from theano import scalar as T


def config():
    '''
    config theano
    :return:
    '''
    print(theano.config.floatX)
    theano.config.floatX = 'float32'
    print(theano.config.device)


def arr():
    '''
    working with array in
    :return:
    '''
    # initialize
    x = T.fmatrix(name='x')
    x_sum = T.sum(x, axis=0)

    # compile
    calc_sum = theano.function(inputs=[x], outputs=x_sum)

    # execute (Python list)
    ary = [[1, 2, 3], [1, 2, 3]]
    print('Column sum:', calc_sum(ary))

    # execute (NumPy array)
    ary = np.array([[1, 2, 3], [1, 2, 3]], dtype=theano.config.floatX)
    print('Column sum:', calc_sum(ary))


arr()