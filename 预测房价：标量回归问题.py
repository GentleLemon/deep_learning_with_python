# -*-coding:utf-8 -*-

"""
# Time       ：2023/1/6 18:51
# Author     ：pei xiaopeng
# version    ：python 3.9
# Description：
"""

from tensorflow.keras.datasets import boston_housing
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

(train_data, train_targets), (test_data, test_targets) = (boston_housing.load_data())
# 准备数据
# 数据标准化
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean  # 重要！！！ 对测试数据进行标准化的平均值和标准差都是在训练数据上计算得到的
test_data /= std


# 构建模型
def build_model():
    """
    需要将同一个模型多次实例化
    :return:
    """
    model = keras.Sequential([
        layers.Dense(64, activation="relu")
        , layers.Dense(64, activation="relu")
        # 没有激活函数，是一个线性层，学会预测任意范围的值
        , layers.Dense(1)
    ])
    # mse为均方误差损失函数，mae为平均绝对误差，是要监控的指标
    model.compile(optimizer="rmsprop"
                  , loss="mse"
                  , metrics=["mae"])
    return model


# K折交叉验证 （验证数据集过小时用，通常将数据化为K（4或5）个分区，实例化K个相同模型，每个模型在K-1个分区上训练，在剩下一个分区上评估）
# 训练100轮
k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []
for i in range(k):
    print(f"Processing fold #{i}")
    # 验证数据
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    # 训练数据
    partial_train_data = np.concatenate(
        [train_data[: i * num_val_samples]
            , train_data[(i + 1) * num_val_samples:]]
        , axis=0
    )
    partial_train_targets = np.concatenate(
        [train_targets[: i * num_val_samples]
            , train_targets[(i + 1) * num_val_samples:]]
        , axis=0
    )
    # 实例化模型
    model = build_model()
    model.fit(partial_train_data
              , partial_train_targets
              , epochs=num_epochs
              , batch_size=16
              , verbose=0)  # 当verbose=0时，就是不输出日志信息 ，进度条、loss、acc这些都不输出
    # 在验证集评估模型
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)  # 监控的为平均绝对误差

# 100轮误差太大，改为训练500轮
num_epochs = 500
all_mae_histories = []
for i in range(k):
    print(f"Processing fold #{i}")
    # 验证数据
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    # 训练数据
    partial_train_data = np.concatenate(
        [train_data[: i * num_val_samples]
            , train_data[(i + 1) * num_val_samples:]]
        , axis=0
    )
    partial_train_targets = np.concatenate(
        [train_targets[: i * num_val_samples]
            , train_targets[(i + 1) * num_val_samples:]]
        , axis=0
    )
    # 实例化模型
    model = build_model()
    history = model.fit(partial_train_data
                        , partial_train_targets
                        , validation_data=(val_data, val_targets)
                        , epochs=num_epochs
                        , batch_size=16
                        , verbose=0)  # 当verbose=0时，就是不输出日志信息 ，进度条、loss、acc这些都不输出
    # 保存没折的验证分数
    mae_histories = history.history["val_mae"]
    all_mae_histories.append(mae_histories)  # 监控的为平均绝对误差

# 计算每轮的K折验证分数平均值
average_mae_history = [
    np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)
]
# 绘制MAE曲线
plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel("Epochs")
plt.ylabel("Validation MAE")
plt.show()
# 剔除前10个点
truncated_mae_history = average_mae_history[10:]
plt.plot(range(1, len(truncated_mae_history) + 1), truncated_mae_history)
plt.xlabel("Epochs")
plt.ylabel("Validation MAE")
plt.show()

# 120轮后，误差不在显著降低
model = build_model()
model.fit(train_data
          , train_targets
          , epochs=120
          , batch_size=16
          , verbose=1)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)

# 预测
pred = model.predict(test_data)
print(pred[0])
