# -*-coding:utf-8 -*-

"""
# Time       ：2022/12/30 14:58
# Author     ：pei xiaopeng
# version    ：python 3.9
# Description：
"""

from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers


class SimpleDense(keras.layers.Layer):  # keras的所有层都继承自Layer基类
    def __init__(self, units, activation=None):
        super().__init__()
        self.units = units
        self.activation = activation

    def build(self, input_shape):
        """
        在build方法中创建权重
        """
        input_dim = input_shape[-1]
        # add_weight()是创建权重的快捷方法
        self.W = self.add_weight(shape=(input_dim, self.units)
                                 , initializer="random_normal"
                                 )
        self.b = self.add_weight(shape=(self.units, )
                                 , initializer="zeros"
                                 )

    def call(self, inputs):
        """
        在call()方法中定义前向传播计算
        :param inputs:
        :return:
        """
        y = tf.matmul(inputs, self.W) + self.b
        if self.activation is not None:
            y = self.activation(y)
        return y


my_dense = SimpleDense(units=32, activation=tf.nn.relu)  # 实例化层
input_tensor = tf.ones(shape=(2, 784))  # 创建测试输入
output_tensor = my_dense(input_tensor)  # 对输入调用层
print(output_tensor.shape)

