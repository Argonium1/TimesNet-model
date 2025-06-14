import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use('TkAgg')  # 或者 'Qt5Agg'
import matplotlib.pyplot as plt
# 检查CUDA是否可用，并选择设备
device = torch.device("cuda" )#if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

if device.type == 'cuda':
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")

# 加载sunspots数据
data = sm.datasets.sunspots.load_pandas().data
data['YEAR'] = data['YEAR'].astype(int)

# 数据归一化
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data[['SUNACTIVITY']])

# 将数据分成训练和测试集
train_size = int(len(data_scaled) - 20) # 308-20=288,最后一项2008年
train_data = data_scaled[:train_size]
test_data = data_scaled[train_size:]

# 创建序列
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 10
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

# 定义模型
class TransformerTimeSeriesModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, dropout=0.1):
        super(TransformerTimeSeriesModel, self).__init__()
        self.input_projector = nn.Linear(input_dim, model_dim)
        self.transformer = nn.Transformer(d_model=model_dim,
                                          nhead=num_heads,
                                          num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers,
                                          dropout=dropout,
                                          batch_first=True)
        self.fc_out = nn.Linear(model_dim, 1)

    def forward(self, src):
        src = self.input_projector(src)
        transformer_out = self.transformer(src, src)
        out = self.fc_out(transformer_out)
        return out[:, -1, :]  # 只取最后一个时间步

# 初始化模型到设备
input_dim = 1
model_dim = 128
num_heads = 8
num_layers = 3
model = TransformerTimeSeriesModel(input_dim, model_dim, num_heads, num_layers).to(device)

# 转换数据以适应PyTorch张量并移动到设备
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).view(-1, seq_length, 1).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).view(-1, seq_length, 1).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

# 定义损失函数和优化器
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 训练模型
def train(model, optimizer, loss_fn, X_train, y_train, epochs=10000):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_train)
        loss = loss_fn(output, y_train)
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

train(model, optimizer, loss_fn, X_train_tensor, y_train_tensor)

# 模型评估
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor).cpu().numpy().squeeze()  # 转回CPU以便使用CPU上的库
    true_values = y_test_tensor.cpu().numpy()

    # 反归一化以恢复原值
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
    true_values = scaler.inverse_transform(true_values.reshape(-1, 1))

    # 打印评估结果
    mse = mean_squared_error(true_values, predictions)
    print(f'Test MSE: {mse}')



# 绘制结果

import matplotlib.pyplot as plt

# 绘制原始数据与预测结果
def plot_results(original_data, train_predictions, test_predictions, train_size, seq_length):
    # 反归一化以获取原始尺度
    original_data = scaler.inverse_transform(original_data)
    train_predictions = scaler.inverse_transform(train_predictions.reshape(-1, 1))
    test_predictions = scaler.inverse_transform(test_predictions.reshape(-1, 1))

    # 创建图形
    plt.figure(figsize=(10, 6))

    # 绘制原始数据
    plt.plot(original_data, label='Original Data', color='black')

    # 绘制训练数据的预测结果
    train_predict_plot = np.empty_like(original_data)
    train_predict_plot[:, :] = np.nan
    train_predict_plot[seq_length:train_size, :] = train_predictions
    plt.plot(train_predict_plot, label='Train Prediction', color='blue')

    # 绘制测试数据的预测结果
    test_predict_plot = np.empty_like(original_data)
    test_predict_plot[:, :] = np.nan
    test_predict_plot[train_size + seq_length:, :] = test_predictions
    plt.plot(test_predict_plot, label='Test Prediction', color='red')

    # 添加图例和标签
    plt.xlabel('Time')
    plt.ylabel('Sunspots Activity')
    plt.legend()
    plt.title('Sunspots Prediction Using Transformer')

    # 显示图形
    plt.show()


# 计算预测结果
model.eval()
with torch.no_grad():
    train_predictions = model(X_train_tensor).cpu().numpy().squeeze()
    test_predictions = model(X_test_tensor).cpu().numpy().squeeze()

# 绘制结果
plot_results(data_scaled, train_predictions, test_predictions, train_size, seq_length)





#输出预测结果
import pandas as pd

# 开发一个方法来反归一化并输出结果
def output_predictions(true_values, predictions, start_index):
    # 反归一化
    true_values = scaler.inverse_transform(true_values.reshape(-1, 1))
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1))

    # 创建 DataFrame 来存储实际值和预测值
    results_df = pd.DataFrame({
        'Year': data['YEAR'][start_index + seq_length:start_index + seq_length + len(true_values)],
        'Actual': true_values.flatten(),
        'Predicted': predictions.flatten()
    })

    # 输出前几条结果
    print("Prediction Results:")
    print(results_df)

# 计算测试集预测
model.eval()
with torch.no_grad():
    test_predictions = model(X_test_tensor).cpu().numpy().squeeze()

# 输出测试集的预测结果
output_predictions(y_test_tensor.cpu().numpy(), test_predictions, train_size)
