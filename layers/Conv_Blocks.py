import torch
import torch.nn as nn
#一个卷积模块

class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels  # 输入通道数
        self.out_channels = out_channels  # 输出通道数
        self.num_kernels = num_kernels  # 卷积核数量
        kernels = []
        # 创建多个不同大小的卷积核
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)  # 将卷积核列表转换为 ModuleList
        if init_weight:
            self._initialize_weights()  # 初始化权重

    def _initialize_weights(self):
        # 初始化卷积层的权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        # 对输入分别应用不同的卷积核
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        # 将结果堆叠并在最后一个维度上取平均
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res
'''
功能：实现 Inception 模块的 V1 版本，使用多个不同大小的卷积核提取特征。

输入：

in_channels：输入特征图的通道数。

out_channels：输出特征图的通道数。

num_kernels：卷积核的数量（默认 6 个）。

init_weight：是否初始化权重。

输出：经过多尺度卷积后的特征图，形状为 [batch_size, out_channels, height, width]。
'''

class Inception_Block_V2(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V2, self).__init__()
        self.in_channels = in_channels  # 输入通道数
        self.out_channels = out_channels  # 输出通道数
        self.num_kernels = num_kernels  # 卷积核数量
        kernels = []
        # 创建多个不同大小的卷积核
        for i in range(self.num_kernels // 2):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=[1, 2 * i + 3], padding=[0, i + 1]))
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=[2 * i + 3, 1], padding=[i + 1, 0]))
        # 添加一个 1x1 的卷积核
        kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
        self.kernels = nn.ModuleList(kernels)  # 将卷积核列表转换为 ModuleList
        if init_weight:
            self._initialize_weights()  # 初始化权重

    def _initialize_weights(self):
        # 初始化卷积层的权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        # 对输入分别应用不同的卷积核
        for i in range(self.num_kernels // 2 * 2 + 1):
            res_list.append(self.kernels[i](x))
        # 将结果堆叠并在最后一个维度上取平均
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res
'''
功能：实现 Inception 模块的 V2 版本，使用多个不同大小的卷积核（包括 1x1 卷积）提取特征。

输入：

in_channels：输入特征图的通道数。

out_channels：输出特征图的通道数。

num_kernels：卷积核的数量（默认 6 个）。

init_weight：是否初始化权重。

输出：经过多尺度卷积后的特征图，形状为 [batch_size, out_channels, height, width]。
'''

'''
Inception_Block_V1 和 Inception_Block_V2 是两种不同版本的 Inception 模块，用于提取多尺度特征。

V1 版本：使用不同大小的方形卷积核（如 1x1、3x3、5x5 等）。

V2 版本：使用不同大小的矩形卷积核（如 1x3、3x1、1x5、5x1 等）以及 1x1 卷积核。

两种模块都通过堆叠多个卷积结果并取平均来融合多尺度特征。
'''