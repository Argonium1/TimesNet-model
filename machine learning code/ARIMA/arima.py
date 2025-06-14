# 导入所需库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.datasets import sunspots
import matplotlib
matplotlib.use('TkAgg')  # 或者 'Qt5Agg'
import matplotlib.pyplot as plt
# 加载数据集
data = sunspots.load_pandas().data
data['YEAR'] = pd.to_datetime(data['YEAR'], format='%Y')
data.set_index('YEAR', inplace=True)
data = data.asfreq('YS')  # 设置频率为年度,

# 可视化数据
plt.figure(figsize=(10, 6))
plt.plot(data, label='Sunspot Data')
plt.title('Sunspot Data Over Time')
plt.xlabel('Year')
plt.ylabel('Sunspots')
plt.legend()
plt.show()

# 拆分训练集和测试集
train_size = len(data) - 20
train, test = data.iloc[:train_size], data.iloc[train_size:]

# 构建和拟合ARIMA模型
model = ARIMA(train, order=(5, 1, 1))
model_fit = model.fit()

# 进行预测
predictions = model_fit.forecast(steps=len(test))
test_index = test.index

# 计算预测的误差
mse = mean_squared_error(test, predictions)
print(f'Test MSE: {mse}')

# 可视化预测结果
plt.figure(figsize=(10, 6))
plt.plot(train.index, train, label='Train')
plt.plot(test_index, test, label='Test')
plt.plot(test_index, predictions, label='Predictions', color='red')
plt.title('Sunspot Forecast')
plt.xlabel('Year')
plt.ylabel('Sunspots')
plt.legend()
plt.show()
