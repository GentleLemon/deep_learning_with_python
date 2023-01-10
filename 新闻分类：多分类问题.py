# -*-coding:utf-8 -*-

"""
# Time       ：2023/1/6 17:38
# Author     ：pei xiaopeng
# version    ：python 3.9
# Description： 路透社数据集, 单标签多分类
"""

from tensorflow.keras.datasets import reuters
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


# 加载数据
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
# 解码
word_index = reuters.get_word_index()
reverse_word_index = dict(
    [(value, key) for (key, value) in word_index.items()]
)
decoded_review = " ".join(
    [reverse_word_index.get(i - 3, "?") for i in train_data[0]]
)


# 准备数据
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequences in enumerate(sequences):
        for j in sequences:
            results[i, j] = 1.
    return results


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)


# one-hot分类编码
def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results


y_train = to_one_hot(train_labels)
y_test = to_one_hot(test_labels)

# 建模
model = keras.Sequential([
    # 46个分类，需要更大的空间维度
    layers.Dense(64, activation="relu")
    , layers.Dense(64, activation="relu")
    # softmax函数，概率分布：每个类别都有一个概率
    , layers.Dense(46, activation="softmax")
])

# 编译模型（使用优化器和损失函数来配置模型）
model.compile(optimizer="rmsprop"
              , loss="categorical_crossentropy"
              , metrics=["accuracy"])

# 在训练集中留出验证集
x_val = x_train[:1000]  # 验证集
partial_x_train = x_train[1000:]  # 训练集
y_val = y_train[:1000]
partial_y_train = y_train[1000:]

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

# 第9轮开始过拟合，重新建模
model = keras.Sequential([
    # 46个分类，需要更大的空间维度
    layers.Dense(64, activation="relu")
    , layers.Dense(64, activation="relu")
    # softmax函数，概率分布：每个类别都有一个概率
    , layers.Dense(46, activation="softmax")
])

# 编译模型（使用优化器和损失函数来配置模型）
model.compile(optimizer="rmsprop"
              , loss="categorical_crossentropy"
              , metrics=["accuracy"])
# 训练
model.fit(x_train
          , y_train
          , epochs=9
          , batch_size=512)
results = model.evaluate(x_test, y_test)
print(results)  # 损失0.94， 精度0.78


# 46个类别，看下随机分类器的精度
import copy

test_labels_copy = copy.copy(test_labels)
np.random.shuffle(test_labels_copy)  # 随机打乱
hist_array = np.array(test_labels) == np.array(test_labels_copy)
print(hist_array.mean())  # 0.19

# 预测
pred = model.predict(x_test)
y_pred = [np.argmax(pred[i]) for i in range(len(x_test))]
print(y_pred)

# 为什么要用64个单元的中间层， 可以测试下4个单元
model = keras.Sequential([
    # 第二层用4个单元测试
    layers.Dense(64, activation="relu")
    , layers.Dense(4, activation="relu")
    # softmax函数，概率分布：每个类别都有一个概率
    , layers.Dense(46, activation="softmax")
])

# 编译模型（使用优化器和损失函数来配置模型）
model.compile(optimizer="rmsprop"
              , loss="categorical_crossentropy"
              , metrics=["accuracy"])
# 训练
model.fit(x_train
          , y_train
          , epochs=9
          , batch_size=512)
results = model.evaluate(x_test, y_test)
print(results)  # 精度只有0.64

# 精度下降8%， 主要原因：试图将大量的信息压缩到维度过小的中间层。
# 总结：单元维度要大于分类个数，过大没有关系，小了精度会降低
