# -*-coding:utf-8 -*-

"""
# Time       ：2023/1/10 18:28
# Author     ：pei xiaopeng
# version    ：python 3.9
# Description： 训练过程可视化
"""

from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os


path = r'E:/desk/my_projects/GitHUb/deep_learning_with_python'

# 多输入、多输出 -- 输入工单标题、正文、用户添加的标签，输出工单的优先级分数和处理工单的部门
vocabulary_size = 10000
num_tags = 100
num_departments = 4
# 定义模型输入
title = keras.Input(shape=(vocabulary_size, ), name="title")  # 文本输入
text_body = keras.Input(shape=(vocabulary_size, ), name="text_body")  # 文本输入
tags = keras.Input(shape=(num_tags, ), name="tags")  # 分类输入
features = layers.Concatenate()([title, text_body, tags])  # 拼接特征
features = layers.Dense(64, activation="relu")(features)
# 定义模型输出
priority = layers.Dense(1, activation="sigmoid", name="priority")(features)  # 工单的优先级，二分类
department = layers.Dense(
    num_departments, activation="softmax", name="department"  # 处理工单的部门，多分类
)(features)
# 实例化模型
model = keras.Model(
    inputs=[title, text_body, tags]
    , outputs=[priority, department]
)
# 生成训练数据
num_samples = 1280
title_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))
text_body_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))
tags_data = np.random.randint(0, 2, size=(num_samples, num_tags))
# 生成目标数据
priority_data = np.random.randint(0, 2, size=(num_samples, 1))
department_data = np.random.randint(0, 2, size=(num_samples, num_departments))
# 编译模型
model.compile(
    optimizer="rmsprop"
    , loss=["mean_squared_error", "categorical_crossentropy"]  # 损失函数 均方误差 MSE + 分类交叉熵
    , metrics=[["mean_absolute_error"], ["accuracy"]]  # 监控指标 平均绝对误差 MAE + 精度
)

# **************** 记录监控日志 ****************
tensorboard = keras.callbacks.TensorBoard(
    log_dir=os.path.join(path, "log_tensorboard")
)

# 训练模型
model.fit(
    [title_data, text_body_data, tags_data]
    , [priority_data, department_data]
    , validation_data=[
        [np.random.randint(0, 2, size=(num_samples, vocabulary_size))
         , np.random.randint(0, 2, size=(num_samples, vocabulary_size))
         , np.random.randint(0, 2, size=(num_samples, num_tags))]
        , [np.random.randint(0, 2, size=(num_samples, 1))
           , np.random.randint(0, 2, size=(num_samples, num_departments))]
    ]
    , epochs=10
    , callbacks=[tensorboard]
)
