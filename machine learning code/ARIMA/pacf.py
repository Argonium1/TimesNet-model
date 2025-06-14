import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_pacf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # 或者 'Qt5Agg'
import matplotlib.pyplot as plt
# 加载数据集
data = sm.datasets.sunspots.load_pandas().data

# 指定训练集大小
train_size = len(data) - 20

# 提取训练集
train_data = data['SUNACTIVITY'][:train_size]

# 绘制PACF图
# plt.figure(figsize=(10, 6))
# plot_pacf(train_data, lags=11, alpha=0.05)
# plt.title('PACF Plot for Training Data original')
# plt.xlabel('Lag')
# plt.ylabel('Partial Autocorrelation')
# plt.show()
#
# plot_pacf(train_data, lags=50, alpha=0.05)
# plt.title('PACF Plot for Training Data original')
# plt.xlabel('Lag')
# plt.ylabel('Partial Autocorrelation')
# plt.show()

# 对数据进行一次差分
data_diff = train_data.diff().dropna()
# 画图
plt.figure(figsize=(10, 6))
plot_pacf(data_diff, lags=11, alpha=0.05)
plt.title('PACF Plot for Training Data')
plt.xlabel('Lag')
plt.ylabel('Partial Autocorrelation')
plt.show()

plot_pacf(data_diff, lags=50, alpha=0.05)
plt.title('PACF Plot for Training Data')
plt.xlabel('Lag')
plt.ylabel('Partial Autocorrelation')
plt.show()