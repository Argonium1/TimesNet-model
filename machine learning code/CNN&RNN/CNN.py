import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import numpy as np
import tensorflow as tf
import random
import os
import matplotlib
matplotlib.use('TkAgg')  # 或者 'Qt5Agg'
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES'] = '0'#设置NVIDIA GPU

# 随机数种子，保证结果可以复现
random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)

# 加载数据
data = sm.datasets.sunspots.load_pandas().data

# 准备数据
data['YEAR'] = pd.to_datetime(data['YEAR'], format='%Y')
data.set_index('YEAR', inplace=True)

# 选择特征
sunspots = data['SUNACTIVITY'].values

# 正规化数据
scaler = MinMaxScaler(feature_range=(0, 1))
sunspots = scaler.fit_transform(sunspots.reshape(-1, 1))

# 使用22个过去的数据预测下一个值
def create_dataset(dataset, look_back):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)
look_back=5

X, Y = create_dataset(sunspots, look_back)

# 分割训练集和测试集
train_size = len(X) - 20
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

# 重塑输入为 [样本, 时间步, 特征]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# 构建CNN模型
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(look_back, 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(75, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# 训练模型
model.fit(X_train, Y_train, epochs=300, batch_size=11, verbose=2)

# 预测
predictions = model.predict(X_test)

# 反归一化数据
predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
Y_test_actual = scaler.inverse_transform(Y_test.reshape(-1, 1))

# 绘制结果
plt.plot(Y_test_actual, label='Actual')
plt.plot(predictions, label='Predicted')
plt.title('Sunspots Prediction')
plt.xlabel('Time Step')
plt.ylabel('Sunspot Activity')
plt.legend()
plt.show()

# 计算MSE
mse = mean_squared_error(Y_test_actual, predictions)
print('Mean Squared Error:', mse)
