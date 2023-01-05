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
# 建模
model = keras.Sequential([
    # 16是该层的单元(unit)个数，即该层表示空间的维度
    layers.Dense(16, activation="relu")
    , layers.Dense(16, activation="relu")
    # sigmoid函数，输出一个0和1之间的概率
    , layers.Dense(1, activation="sigmoid")
])
# 编译模型
model.compile(optimizer="rmsprop"
              , loss="binary_crossentropy"
              , metrics=["accuracy"])
