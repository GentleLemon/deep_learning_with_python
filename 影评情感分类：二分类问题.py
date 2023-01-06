# -*-coding:utf-8 -*-

"""
# Time       ：2022/12/30 16:55
# Author     ：pei xiaopeng
# version    ：python 3.9
# Description：
"""

from tensorflow.keras.datasets import imdb
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
# 将评论解码为文本
word_index = imdb.get_word_index()
reverse_word_index = dict(
    [(value, key) for (key, value) in word_index.items()]
)
decoded_review = " ".join(
    [reverse_word_index.get(i - 3, "?") for i in train_data[0]]
)


# 准备数据，用multi-hot编码对整数序列进行编码(目的：将列表转化为张量)
# 什么是multi-hot编码？ 把列表转换为由0和1组成的向量。例如[8, 5]转换为10000维向量，只有索引8和5为1，其余元素全为0
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequences in enumerate(sequences):
        for j in sequences:
            results[i, j] = 1.
    return results


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
# 将标签向量化
y_train = np.asarray(train_labels).astype("float32")
y_test = np.asarray(test_labels).astype("float32")
# 建模 两个中间层 + 第三层输出标量预测值
model = keras.Sequential([
    # 16是该层的单元(unit)个数，即该层表示空间的维度
    layers.Dense(16, activation="relu")
    , layers.Dense(16, activation="relu")
    # sigmoid函数，输出一个0和1之间的概率
    , layers.Dense(1, activation="sigmoid")
])
# 编译模型（使用优化器和损失函数来配置模型）
model.compile(optimizer="rmsprop"
              , loss="binary_crossentropy"
              , metrics=["accuracy"])
# 在训练集中留出验证集
x_val = x_train[:10000]  # 验证集
partial_x_train = x_train[10000:]  # 训练集
y_val = y_train[:10000]
partial_y_train = y_train[10000:]
# 训练模型
history = model.fit(partial_x_train
                    , partial_y_train
                    , epochs=20
                    , batch_size=512
                    , validation_data=(x_val, y_val))
# 训练的模型有一个history对象，它是一个字典，包含训练过程的全部数据
history_dict = history.history
# print(history_dict.keys())
# 绘制训练损失和验证损失
loss_value = history_dict["loss"]
val_value = history_dict["val_loss"]
epochs = range(1, len(loss_value) + 1)
plt.plot(epochs, loss_value, "bo", label="Training loss")
plt.plot(epochs, val_value, "b", label="Validation loss")
plt.title("Training and Validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
# 绘制训练精度和验证精度
plt.clf()
acc = history_dict["accuracy"]
val_acc = history_dict["val_accuracy"]
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, "bo", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.title("Training and Validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
# 画图后，发现过拟合。看图只需训练4轮
# 重新建模
model = keras.Sequential([
    # 16是该层的单元(unit)个数，即该层表示空间的维度
    layers.Dense(16, activation="relu")
    , layers.Dense(16, activation="relu")
    # sigmoid函数，输出一个0和1之间的概率
    , layers.Dense(1, activation="sigmoid")
])
# 编译模型（使用优化器和损失函数来配置模型）
model.compile(optimizer="rmsprop"
              , loss="binary_crossentropy"
              , metrics=["accuracy"])
model.fit(x_train
          , y_train
          , epochs=4
          , batch_size=512)
results = model.evaluate(x_test, y_test)
print(results)  # 损失0.28， 精度0.88
# 预测
pred = model.predict(x_test)
print(pred)
