# -*-coding:utf-8 -*-

"""
# Time       ：2022/12/29 13:59
# Author     ：pei xiaopeng
# version    ：python 3.9
# Description： 简单的Sequential类
"""


class NaiveSequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x

    @property
    def weights(self):
        weights = []
        for layer in self.layers:
            weights += layer.weights
        return weights
