import statsmodels.api as sm
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

# 加载 sunspot.year 数据集，309个数据点
data = sm.datasets.sunspots.load_pandas().data

# 查看数据集的前几行
print(data.head())

# 绘制数据集
plt.figure(figsize=(10, 6))
plt.plot(data['YEAR'], data['SUNACTIVITY'])
plt.title('Yearly Sunspot Data')
plt.xlabel('Year')
plt.ylabel('Sunspot Activity')
plt.show()

# 设置训练集和测试集的分割点，将最后20年的数据作为测试集
train_size = len(data) - 20

# 分割训练集和测试集
train, test = data.iloc[:train_size], data.iloc[train_size:]

# 输出训练集和测试集的大小
print(f"Training set size: {len(train)}")
print(f"Test set size: {len(test)}")

# 可视化训练集和测试集
plt.figure(figsize=(12, 6))
plt.plot(train['YEAR'], train['SUNACTIVITY'], label='Train')
plt.plot(test['YEAR'], test['SUNACTIVITY'], label='Test')
plt.title('Train-Test Split for Sunspot Activity')
plt.xlabel('Year')
plt.ylabel('Sunspot Activity')
plt.legend()#图例
plt.show()

# 43-66行都是转换数据格式用的
# 创建特征：通过滞后特征将时间序列转换为监督学习问题
def create_features(data, n_lags): # 把一列数与他们之前的数组成数组，变成一列数组
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(n_lags + 1)]
    df = pd.concat(columns, axis=1)
    df.columns = ['target'] + [f'lag_{i}' for i in range(1, n_lags + 1)]
    df.dropna(inplace=True)
    return df

# 根据需求设置滞后的阶数
n_lags = 5  # 选择合适的滞后阶数（看acf图？）
dataset = create_features(data['SUNACTIVITY'].values, n_lags)

# 调整训练集和测试集
train, test = dataset.iloc[:train_size-n_lags], dataset.iloc[train_size-n_lags:]

# 准备训练和测试数据
X_train, y_train = train.drop('target', axis=1), train['target']
X_test, y_test = test.drop('target', axis=1), test['target']


# 将数据转换为DMatrix格式
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# 设置XGBoost参数
params = {
    'objective': 'reg:squarederror',#定义回归的目标函数，指定回归问题使用平方误差作为损失函数
    'max_depth': 3,#树深度
    'eta': 0.1,#学习率（步长缩小），较小的值可以防止过拟合。
    'subsample': 0.9,#训练实例的子样本比率（防止过拟合用的）
    'colsample_bytree': 0.8,#构造每棵树时柱的子采样率（增加模型多样性）
    'seed': 42 # 随机数种子，保证实验可重复性
}
# XGBoosting参数怎么调整呢，不能一个个试吧

# 训练模型
model = xgb.train(params, dtrain, num_boost_round=100)#指定boosting轮数的训练模型，round=100

# 用测试集进行预测
y_pred = model.predict(dtest)

# 计算和输出均方误差
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# 可视化结果
plt.figure(figsize=(10, 6))
plt.plot(data['YEAR'].iloc[train_size:], y_test, label='Actual')
plt.plot(data['YEAR'].iloc[train_size:], y_pred, label='Predicted')
plt.title('XGBoost Predictions for Sunspot Activity')
plt.xlabel('Year')
plt.ylabel('Sunspot Activity')
plt.legend()
plt.show()
