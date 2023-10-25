from itertools import cycle
import pickle
import sys
import warnings
import time
from typing import List, Tuple
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import LSTM, Bidirectional, BatchNormalization, Convolution1D, MaxPooling1D, Reshape
from tensorflow.keras.models import Sequential
from keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer

cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
tf.config.experimental.set_visible_devices(devices=cpus[0], device_type='CPU')

# data = pd.read_csv('web_attacks_balanced.csv', delimiter=',')
train_data = pd.read_csv(r"D:\AutoEncoder\selfdataset\train.csv")
test_data = pd.read_csv(r"D:\AutoEncoder\selfdataset\test.csv")

# 加载训练的数据
y = train_data['Label']
X = train_data.drop(
    labels=['Label',
            'Flow ID',
            'Src IP',
            'Dst IP'], axis=1)

scaler = MinMaxScaler()

# 构建自定义空间注意力层
class SpatialAttentionLayer1D(layers.Layer):
    def __init__(self, **kwargs):
        super(SpatialAttentionLayer1D, self).__init__(**kwargs)
        self.conv1 = layers.Conv1D(1, kernel_size=1, padding='same', activation='sigmoid')

    def call(self, inputs):
        attention_weights = self.conv1(inputs)
        return layers.multiply([inputs, attention_weights])

    def get_config(self):
        return super().get_config().copy()

# 搭建网络架构
model = Sequential()
model.add(Convolution1D(64, kernel_size=32, padding="same", activation="relu", input_shape=(79, 1)))
model.add(SpatialAttentionLayer1D())
model.add(MaxPooling1D(pool_size=(5)))
model.add(BatchNormalization())
model.add(Bidirectional(LSTM(64, return_sequences=False)))
model.add(Reshape((128, 1), input_shape=(128,)))

model.add(MaxPooling1D(pool_size=(5)))
model.add(BatchNormalization())
model.add(Bidirectional(LSTM(128, return_sequences=False)))

model.add(Dropout(0.5))
# model.add(Dense(10))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.add(Dense(5, activation='softmax'))

# 打印网络信息
model.summary(line_length=100)

# 使用分层K-Folds交叉验证
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 存储每折的评估指标
scores = pd.DataFrame(0, columns=[1, 2, 3, 4, 5],
                      index=['Accuracy', 'Precision', 'Recall', 'F1'])

# 打印评估指标
def print_metrics(y_eval: np.ndarray, y_pred: np.ndarray, average: str = None) -> List[float]:
    accuracy = metrics.accuracy_score(y_eval, y_pred, average=average)
    precision = metrics.precision_score(y_eval, y_pred, average=average, zero_division=0)
    recall = metrics.recall_score(y_eval, y_pred, average=average)
    f1 = metrics.f1_score(y_eval, y_pred, average=average)

    print('Accuracy =', accuracy)
    print('Precision =', precision)
    print('Recall =', recall)
    print('F1 =', f1)

    return [accuracy, precision, recall, f1]


# 在交叉验证的每个步骤中存储所有测试和预测的标签，用于计算混淆矩阵
actual_targets = np.empty([0], dtype=float)
predicted_targets = np.empty([0], dtype=float)


# 每折的训练和验证函数
def train_and_validate(model: Sequential,
                       X: pd.DataFrame,
                       y: pd.Series,
                       train_index: np.ndarray,
                       test_index: np.ndarray,
                       scaler: MinMaxScaler) -> Tuple[np.ndarray, np.ndarray]:
    # 获取当前折叠的测试数据和验证数据
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    print("Train index:", train_index)
    print("Test index:", test_index)

    # 数据特征缩放，归一化
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    # 重塑特征向量，将其用作神经网络输入
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # 标签One-Hot编码
    y_train_encoded = pd.get_dummies(y_train).values
    y_test_encoded = pd.get_dummies(y_test).values

    # y_train_encoded = to_categorical(y_train)
    # y_test_encoded = to_categorical(y_test)

    # 神经网络训练
    model.fit(X_train, y_train_encoded, validation_data=(X_test, y_test_encoded), batch_size=128, epochs=5)

    # 将神经网络模型应用于验证数据并获得该预测的准确性
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_eval = np.argmax(y_test_encoded, axis=1)

    return y_eval, y_pred

start_time = time.time()
i = 1
# 分层K-Folds交叉验证
for train_index, test_index in cv.split(X, y):
    y_eval, y_pred = train_and_validate(model, X, y, train_index, test_index, scaler)
    # 保存测试和预测标签
    actual_targets = np.append(actual_targets, y_eval)
    predicted_targets = np.append(predicted_targets, y_pred)
    # 获取并打印Validation scores
    print("Validation scores:")
    scores[i] = print_metrics(y_eval, y_pred, average='weighted')
    i += 1

# 打印训练时间
print('Total operation time: ', (time.time() - start_time) * 0.2, 'seconds')

# 模型评估
scores['Mean'] = (scores[1] + scores[2] + scores[3] + scores[4] + scores[5]) / 5
print('scores:', scores)

# 可视化所有数据折叠的混淆矩阵以评估分类器的性能
'''
def plot_confusion_matrix(y_eval: np.ndarray, y_pred: np.ndarray, labels: List[str]) -> None:
    # 获取最后一折数据的混淆矩阵
    confusion_matrix = metrics.confusion_matrix(y_eval, y_pred)

    # 准确率和错误率
    accuracy = np.trace(confusion_matrix) / float(np.sum(confusion_matrix))
    misclass = 1 - accuracy

    # 绘制混淆矩阵
    metrics.ConfusionMatrixDisplay.from_predictions(y_eval, y_pred, display_labels=labels,
                                                    xticks_rotation='vertical', cmap='Blues')
    plt.xlabel('Predicted label\n\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()

plot_confusion_matrix(actual_targets, predicted_targets, ['Benign',
                                                          'Brute_Force',
                                                          'Port_Scan',
                                                          'SQL_Injection',
                                                          'Web_Attack'
                                                          ])
'''