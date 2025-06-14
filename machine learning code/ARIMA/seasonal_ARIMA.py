import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller, kpss
import pmdarima as pm
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 或者 'Qt5Agg'
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
# 载入sunspot.year数据集
data = sm.datasets.sunspots.load_pandas().data

# 将年份设置为时间索引
data.index = pd.Index(sm.tsa.datetools.dates_from_range('1700', '2008'))
data = data['SUNACTIVITY']

# 使用移动平均消除11年周期的季节性
moving_avg = data.rolling(window=0, center=True).mean()

# 画出原始数据和移动平均结果
plt.figure(figsize=(14, 7))
plt.plot(data, label='Original Data')
plt.plot(moving_avg, color='red', label='Moving Average')
plt.title('Time Series with Moving Average')
plt.legend()
plt.show()

# ADF检验
moving_avg_no_NA = moving_avg.dropna()  # 去掉NaN值
print('11年周期差分数据集')
result_adf = adfuller(moving_avg)
print('ADF Statistic:', result_adf[0])
print('p-value:', result_adf[1])
for key, value in result_adf[4].items():
    print(f'Critical Values {key}: {value}')

# 绘制ACF图
plt.figure(figsize=(10, 6))
plot_acf(moving_avg_no_NA, lags=50, alpha=0.05)
plt.title('ACF Plot for seasonal_difference')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.show()

# 对数据进行一次差分
data_diff = moving_avg_no_NA.diff().dropna()

# ADF检验
print('差分后')
result_adf = adfuller(data_diff)
print('ADF Statistic:', result_adf[0])
print('p-value:', result_adf[1])
for key, value in result_adf[4].items():
    print(f'Critical Values {key}: {value}')

# 绘制ACF图
plt.figure(figsize=(10, 6))
plot_acf(data_diff, lags=50, alpha=0.05)
plt.title('ACF Plot for seasonal_difference')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.show()

plt.show()

# 执行 auto_arima
model = pm.auto_arima(data,
                      seasonal=True,  # 假设非季节性数据
                      stepwise=True,  # 使用逐步选法来快速获得结果
                      suppress_warnings=True,  # 忽略模型拟合期间的警告
                      trace=True)  # 设为True以查看每次尝试的结果

# 拟合最佳模型
model.fit(data)

# 输出最佳模型摘要
print(model.summary())

# 预测未来20个数据点
n_periods = 20
forecast, conf_int = model.predict(n_periods=n_periods, return_conf_int=True)

# 计算均方误差 (MSE)
train_size = len(data)
test_size = n_periods
predicted = forecast
true_values = data[-test_size:]
mse = mean_squared_error(true_values, predicted)
print(f'Mean Squared Error (MSE): {mse}')

# 绘制残差图
residuals = model.resid()
plt.figure(figsize=(10, 4))
plt.plot(residuals)
plt.title('Residuals')
plt.xlabel('Time')
plt.ylabel('Residuals')
plt.show()

# 绘制预测结果
plt.figure(figsize=(10, 6))
plt.plot(range(train_size), data, label='Original Data')
plt.plot(range(train_size, train_size + n_periods), forecast, label='Forecast', color='r')
plt.fill_between(range(train_size, train_size + n_periods),
                 conf_int[:, 0], conf_int[:, 1], color='pink', alpha=0.3)
plt.title('Forecast vs Actuals')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()