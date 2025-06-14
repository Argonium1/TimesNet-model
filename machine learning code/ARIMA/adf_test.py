import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.datasets import sunspots

# 加载数据集
data = sunspots.load_pandas().data
data['YEAR'] = pd.to_datetime(data['YEAR'], format='%Y')
data.set_index('YEAR', inplace=True)

# # 时序图
# plt.figure(figsize=(10, 6))
# plt.plot(data, label='Sunspot Data')
# plt.title('Sunspot Data Over Time')
# plt.xlabel('Year')
# plt.ylabel('Sunspots')
# plt.show()

print('原数据集')
# ADF检验
result_adf = adfuller(data['SUNACTIVITY'])
print('ADF Statistic:', result_adf[0])
print('p-value:', result_adf[1])
for key, value in result_adf[4].items():
    print(f'Critical Values {key}: {value}')

# KPSS检验
result_kpss = kpss(data['SUNACTIVITY'], regression='c')
print('KPSS Statistic:', result_kpss[0])
print('p-value:', result_kpss[1])
for key, value in result_kpss[3].items():
    print(f'Critical Values {key}: {value}')

# 数据集不平稳
# 做一次差分

# 对数据进行一次差分
data_diff = data.diff().dropna()

# 查看差分后的数据
# plt.figure(figsize=(10, 6))
# plt.plot(data_diff['SUNACTIVITY'], label='Differenced Data')
# plt.title('Differenced Sunspot Data')
# plt.xlabel('Year')
# plt.ylabel('Differenced Sunspots')
# plt.legend()
# plt.show()

print('差分后的数据')
# ADF检验
result_adf = adfuller(data_diff['SUNACTIVITY'])
print('ADF Statistic:', result_adf[0])
print('p-value:', result_adf[1])
for key, value in result_adf[4].items():
    print(f'Critical Values {key}: {value}')

# KPSS检验
result_kpss = kpss(data_diff['SUNACTIVITY'], regression='c')
print('KPSS Statistic:', result_kpss[0])
print('p-value:', result_kpss[1])
for key, value in result_kpss[3].items():
    print(f'Critical Values {key}: {value}')