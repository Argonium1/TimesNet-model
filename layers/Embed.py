import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math

#一个嵌入模块
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # 预先计算位置编码
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置使用正弦函数
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置使用余弦函数

        pe = pe.unsqueeze(0)  # 增加批次维度
        self.register_buffer('pe', pe)  # 将位置编码注册为缓冲区

    def forward(self, x):
        return self.pe[:, :x.size(1)]  # 返回与输入序列长度匹配的位置编码

'''
功能：生成位置编码，用于为序列中的每个位置提供位置信息。

输入：

d_model：嵌入维度。

max_len：最大序列长度。

输出：位置编码，形状为 [1, seq_len, d_model]。
'''

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        # 初始化卷积层权重
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)  # 转置后卷积，再转置回来
        return x
'''
功能：将输入序列的每个 token 映射到高维空间。

输入：

c_in：输入特征维度。

d_model：输出嵌入维度。

输出：嵌入后的序列，形状为 [batch_size, seq_len, d_model]。
'''

class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)  # 偶数位置使用正弦函数
        w[:, 1::2] = torch.cos(position * div_term)  # 奇数位置使用余弦函数

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)  # 固定权重

    def forward(self, x):
        return self.emb(x).detach()  # 返回嵌入结果，不计算梯度
'''
功能：生成固定的嵌入，用于时间特征编码。

输入：

c_in：输入特征维度。

d_model：输出嵌入维度。

输出：嵌入后的序列，形状为 [batch_size, seq_len, d_model]。
'''

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x
'''
功能：将时间特征（如小时、分钟、星期几等）映射到高维空间。

输入：

d_model：输出嵌入维度。

embed_type：嵌入类型（固定或可学习）。

freq：时间频率（如小时、分钟等）。

输出：时间特征嵌入，形状为 [batch_size, seq_len, d_model]。
'''

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)
'''
功能：将时间特征映射到高维空间。

输入：

d_model：输出嵌入维度。

embed_type：嵌入类型。

freq：时间频率。

输出：时间特征嵌入，形状为 [batch_size, seq_len, d_model]。
'''

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)
'''
功能：将输入数据和位置编码、时间特征编码结合。

输入：

c_in：输入特征维度。

d_model：输出嵌入维度。

embed_type：嵌入类型。

freq：时间频率。

dropout：Dropout 概率。

输出：嵌入后的序列，形状为 [batch_size, seq_len, d_model]。
'''

class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        return self.dropout(x)
'''
功能：将输入数据映射到高维空间（倒置版本）。

输入：

c_in：输入特征维度。

d_model：输出嵌入维度。

embed_type：嵌入类型。

freq：时间频率。

dropout：Dropout 概率。

输出：嵌入后的序列，形状为 [batch_size, seq_len, d_model]。
'''

class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)
'''
功能：将输入数据和时间特征编码结合（不包含位置编码）。

输入：

c_in：输入特征维度。

d_model：输出嵌入维度。

embed_type：嵌入类型。

freq：时间频率。

dropout：Dropout 概率。

输出：嵌入后的序列，形状为 [batch_size, seq_len, d_model]。
'''

class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars
'''
功能：将输入序列分块并嵌入到高维空间。

输入：

d_model：输出嵌入维度。

patch_len：每个块的长度。

stride：滑动步长。

padding：填充大小。

dropout：Dropout 概率。

输出：嵌入后的序列和变量数量。
'''