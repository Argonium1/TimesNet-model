import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # 或者 'Qt5Agg'
import matplotlib.pyplot as plt
# 加载数据集
data = sm.datasets.sunspots.load_pandas().data

# 设置训练集的大小（例如，倒数20个点作为测试集）
train_size = len(data) - 20

# 训练集数据
train_data = data['SUNACTIVITY'][:train_size]

# # 绘制训练集的ACF图
# plt.figure(figsize=(10, 6))
# plot_acf(train_data, lags=11, alpha=0.05)
# plt.title('ACF Plot for Training Data original')
# plt.xlabel('Lag')
# plt.ylabel('Autocorrelation')
# plt.show()
#
# plot_acf(train_data, lags=50, alpha=0.05)
# plt.title('ACF Plot for Training Data original')
# plt.xlabel('Lag')
# plt.ylabel('Autocorrelation')
# plt.show()

# 对数据进行一次差分
data_diff = train_data.diff().dropna()

# 差分后acf
plt.figure(figsize=(10, 6))
plot_acf(data_diff, lags=11, alpha=0.05)
plt.title('ACF Plot for Training Data diff')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.show()

plot_acf(data_diff, lags=50, alpha=0.05)
plt.title('ACF Plot for Training Data diff')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.show()