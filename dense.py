# -*-coding:utf-8 -*-

"""
# Time       ：2022/12/28 16:26
# Author     ：pei xiaopeng
# version    ：python 3.9
# Description： 简单的Dense类
"""


import tensorflow as tf


class NaiveDense:
    def __init__(self, input_size, output_size, activation):
        self.activation = activation
        # 创建矩阵W
        w_shape = (input_size, output_size)
        # 初始化矩阵
        w_initial_value = tf.random.uniform(w_shape, minval=0, maxval=1e-1)
        self.W = tf.Variable(w_initial_value)
        # 创建零向量b
        b_shape = (output_size,)
        b_initial_value = tf.zeros(b_shape)
        self.b = tf.Variable(b_initial_value)

    def __call__(self, inputs):
        """
        向前传播
        """
        return self.activation(tf.matmul(inputs, self.W) + self.b)

    @property
    def weights(self):
        """
        获取该层权重
        """
        return [self.W, self.b]
