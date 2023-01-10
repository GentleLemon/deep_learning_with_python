# -*-coding:utf-8 -*-

"""
# Time       ：2023/1/10 10:25
# Author     ：pei xiaopeng
# version    ：python 3.9
# Description： 构建Keras模型的3种方法
"""

from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


# **************** 序贯模型 ****************
# 序贯模型 sequential model -- 简单堆积原则
# 此时的模型还没有权重
model = keras.Sequential([
    layers.Dense(64, activation="relu")
    , layers.Dense(10, activation="softmax")
])

# 通过第一次调用模型来完成构建
model.build(input_shape=(None, 3))  # None表示批量可以是任意大小，模型样本形状应该是（3，）
print('检索模型权重：', model.weights)

# summary()方法显示模型内容
print(model.summary())

# 用name参数命名模型和层
# Input类可以实时构建sequential模型
model = keras.Sequential(name="my_test_model")
model.add(keras.Input(shape=(3, )))  # shape是单个样本形状
model.add(layers.Dense(64, activation='relu'))
print(model.summary())
model.add(layers.Dense(10, activation='softmax'))
print(model.summary())


# **************** 函数式API ****************
# 当输入数据是一维的时，例如多层感知器，必须显示的预留最后一维的空间，以便在训练网络时分割数据时使用的mini_batch大小的形状
# 因此，当输入是一维（3，）时，shape tuple总是用用逗号指示预留一个维度
# shape=(3,)表示接收一个含有 3 个整数的序列，这些整数在 1 到 10,000 之间。
inputs = keras.Input(shape=(3, ), name="my_input")  # 输入层
features = layers.Dense(64, activation="relu")(inputs)  # 非线性的几何变换
outputs = layers.Dense(10, activation="softmax")(features)  # 输出层，10个类别，softmax计算概率分布
model = keras.Model(inputs=inputs, outputs=outputs)  # 实例化模型

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
print(model.summary())
# 训练模型
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
# 训练模型
model.fit(
    [title_data, text_body_data, tags_data]
    , [priority_data, department_data]
    , epochs=1
)
# 评估模型
model.evaluate(
    [title_data, text_body_data, tags_data]
    , [priority_data, department_data]
)
# 预测
priority_perds, department_perds = model.predict(
    [title_data, text_body_data, tags_data]
)
print(priority_perds[0], department_perds[0])

# 一般按输入和目标组成的字典训练模型 priority关键字为Input对象和输出层名称
# model.compile(
#     optimizer="rmsprop"
#     , loss={"priority": "mean_squared_error", "department": "categorical_crossentropy"}  # 损失函数 均方误差 MSE + 分类交叉熵
#     , metrics={"priority": ["mean_absolute_error"], "department": ["accuracy"]}  # 监控指标 平均绝对误差 MAE + 精度
# )
# model.fit(
#     {"title": title_data, "text_body": text_body_data, "tags": tags_data}
#     , {"priority": priority_data, "department": department_data}
#     , epochs=1
# )
# 模型可视化
keras.utils.plot_model(model, "ticket_classifier.png")
# 将输入和输出形状添加到图中
keras.utils.plot_model(
    model, "ticket_classifier_with_shape.png", show_shapes=True
)
# 查看所有层
print(model.layers)
# 增加一个输出
num_difficulty = 3  # 解决难度 快、中、慢
features = model.layers[4].output  # Dense层
difficulty = layers.Dense(num_difficulty, activation="softmax", name="difficulty")(features)
new_model = keras.Model(
    inputs=[title, text_body, tags]
    , outputs=[priority, department, difficulty]
)
# 将输入和输出形状添加到图中
keras.utils.plot_model(
    new_model, "updated_ticket_classifier_with_shape.png", show_shapes=True
)


# **************** 模型子类化 ****************
# 使用Model子类重新实现客户支持工单管理模型
class CustomerTicketModel(keras.Model):

    def __init__(self, num_departments):
        """
        1. 在__init__()方法中，定义模型将使用的层
        Args:
            num_departments:
        """
        super().__init__()
        self.concat_layer = layers.Concatenate()
        self.mixing_layer = layers.Dense(64, activation="relu")
        self.priority_scorer = layers.Dense(1, activation="sigmoid")
        self.department_classifier = layers.Dense(num_departments, activation="softmax")

    def call(self, inputs):
        """
        2. 在call()方法中，定义模型前向传播，重复使用之前创建的层
        Args:
            inputs:

        Returns:

        """
        title = inputs["title"]
        text_body = inputs["text_body"]
        tags = inputs["tags"]

        features = self.concat_layer([title, text_body, tags])
        features = self.mixing_layer(features)
        priority = self.priority_scorer(features)
        department = self.department_classifier(features)
        return priority, department

# 实例化模型
model = CustomerTicketModel(num_departments=4)
priority, department = model(
    {"title": title_data, "text_body": text_body_data, "tags": tags_data}
)
# 编译模型
model.compile(
    optimizer="rmsprop"
    , loss=["mean_squared_error", "categorical_crossentropy"]  # 损失函数 均方误差 MSE + 分类交叉熵
    , metrics=[["mean_absolute_error"], ["accuracy"]]  # 监控指标 平均绝对误差 MAE + 精度
)
# 训练模型
model.fit(
    {"title": title_data, "text_body": text_body_data, "tags": tags_data}
    , [priority_data, department_data]
    , epochs=1
)
# 评估模型
model.evaluate(
    {"title": title_data, "text_body": text_body_data, "tags": tags_data}
    , [priority_data, department_data]
)
# 预测
priority_perds, department_perds = model.predict(
    {"title": title_data, "text_body": text_body_data, "tags": tags_data}
)
print(priority_perds[0], department_perds[0])



