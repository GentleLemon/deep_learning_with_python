# -*-coding:utf-8 -*-

"""
# Time       ：2022/12/30 10:56
# Author     ：pei xiaopeng
# version    ：python 3.9
# Description：
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# **************************************************************
# 首先，生成一些线性可分的数据
num_samples_per_class = 1000
# 第一个类
negative_samples = np.random.multivariate_normal(
    mean=[0, 3]
    , cov=[[1, 0.5], [0.5, 1]]
    , size=num_samples_per_class
)
# 第二个类
positive_samples = np.random.multivariate_normal(
    mean=[3, 0]
    , cov=[[1, 0.5], [0.5, 1]]
    , size=num_samples_per_class
)
# 合并生成样本
inputs = np.vstack((negative_samples, positive_samples)).astype(np.float32)
# 生成目标标签
targets = np.vstack((np.zeros((num_samples_per_class, 1), dtype="float32")
                    , np.ones((num_samples_per_class, 1), dtype="float32")))
# 绘制图像
plt.scatter(inputs[:, 0], inputs[:, 1], c=targets[:, 0])
plt.show()
# **************************************************************
# 创建线性分类器
input_dim = 2
output_dim = 1
W = tf.Variable(initial_value=tf.random.uniform(shape=(input_dim, output_dim)))
b = tf.Variable(initial_value=tf.zeros(shape=(output_dim, )))

# 向前传播函数
def model(inputs):
    return tf.matmul(inputs, W) + b
# 均方误差损失函数
def square_loss(targets, predictions):
    per_samples_losses = tf.square(targets - predictions)
    return tf.reduce_mean(per_samples_losses)
# 训练步骤函数
learning_rate = 0.1
def training_step(inputs, targets):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = square_loss(targets, predictions)
    grad_loss_wrt_W, grad_loss_wrt_b = tape.gradient(loss, [W, b])  # 检索损失相对于权重的梯度
    W.assign_sub(grad_loss_wrt_W * learning_rate)  # 更新权重
    b.assign_sub(grad_loss_wrt_b * learning_rate)
    return loss
# 训练循环
for step in range(40):
    loss = training_step(inputs, targets)
    print(f"Loss as step {step}: {loss:.4f}")
# 线性模型对训练数据点分类
predictions = model(inputs)
plt.scatter(inputs[:, 0], inputs[:, 1], c=predictions[:, 0] > 0.5)
plt.show()
# 绘制直线
x = np.linspace(-1, 4, 100)  # -1 和 4之间生成100个等间距的数字
y = - W[0] / W[1] * x + (0.5 - b) / W[1]
plt.plot(x, y, "-r")
plt.scatter(inputs[:, 0], inputs[:, 1], c=predictions[:, 0] > 0.5)
plt.show()
