# coding=utf-8
'''
@Title:scalars for tensor demo

@Author: tyee.noprom@qq.com
@Time: 7/25/16 10:51 PM
'''
import theano
from theano import tensor as T

# initialize
x1 = T.scalar()
w1 = T.scalar()
w0 = T.scalar()
z1 = x1 * w1 + w0

# compile
net_input = theano.function(inputs=[w1, x1, w0], outputs=z1)

# execute
print ('Net input: %.2f' % net_input(2.0, 1.0, 0.5))
