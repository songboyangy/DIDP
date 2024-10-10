import time
import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import math
import random
def set_config(args):
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    # param['prefix'] = f'{args.prefix}_{args.dataset}_CTCP'
    args.prefix = f'{args.prefix}_{args.data_name}_DDiff_{timestamp}'
    args.model_path = f"save_models/{args.prefix}"

    args.log_path = f"log/{args.prefix}.log"
    return args


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_size, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # 创建一个长为 max_len 的位置编码矩阵
        pe = torch.zeros(max_len, embedding_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)

        # 计算 10000^(2i/d_model)
        div_term = torch.exp(torch.arange(0, embedding_size, 2).float() * (-math.log(10000.0) / embedding_size))

        # 偶数维度使用 sin，奇数维度使用 cos
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置

        pe = pe.unsqueeze(0)  # 扩展维度为 (1, max_len, embedding_size)
        self.register_buffer('pe', pe)  # 将 pe 存储为模型的缓冲区，不会更新

    def forward(self, x):
        # x 的形状是 (batch_size, sequence_length, embedding_size)
        # 因此我们需要对 pe 切片，选择相同的 sequence_length
        x = x + self.pe[:, :x.size(1), :].to(x.device)  # 将位置编码加到输入的 embedding 上
        return x


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, embedding_size, max_len=5000):
        super(LearnedPositionalEncoding, self).__init__()
        # 定义可学习的位置编码
        self.position_embeddings = nn.Embedding(max_len, embedding_size)

    def forward(self, x):
        batch_size, sequence_length, embedding_size = x.shape
        # 创建位置序列 [0, 1, 2, ..., sequence_length-1]
        position_ids = torch.arange(sequence_length, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size,
                                                        sequence_length)  # 扩展为 (batch_size, sequence_length)

        # 获取可学习的位置编码并加到嵌入上
        position_embeddings = self.position_embeddings(position_ids)
        x = x + position_embeddings.to(x.device)
        return x


class TimeDifferenceEncoder(torch.nn.Module):
    def __init__(self, dimension: int):
        super(TimeDifferenceEncoder, self).__init__()
        self.dimension = dimension
        self.w = torch.nn.Linear(1, dimension)  # 线性层，从 1 维映射到 dimension 维

        # 初始化权重
        self.w.weight = torch.nn.Parameter(
            (torch.from_numpy(1 / 10 ** np.linspace(0, 9, dimension)))
            .float().reshape(dimension, -1)
        )
        self.w.bias = torch.nn.Parameter(torch.zeros(dimension).float())

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        将时间差映射到向量空间
        :param t: 时间差张量，形状为 (batch_size, seq_len)
        :return: 编码后的时间差向量，形状为 (batch_size, seq_len, dimension)
        """
        # t 的形状为 (batch_size, seq_len)，需要扩展到 (batch_size, seq_len, 1)
        t = t.unsqueeze(dim=-1)  # 添加一维，将其变为 (batch_size, seq_len, 1)

        # 线性变换并应用余弦函数，输出形状为 (batch_size, seq_len, dimension)
        output = torch.cos(self.w(t))
        return output


class TimeAttention(nn.Module):
    def __init__(self, time_size, in_features1):
        super(TimeAttention, self).__init__()
        self.time_embedding = nn.Embedding(time_size, in_features1)
        init.xavier_normal_(self.time_embedding.weight)
        self.dropout = nn.Dropout(0.1)

    def forward(self, T_idx, Dy_U_embed, mask=None, episilon=1e-6):
        '''
            T_idx: (bsz, user_len)
            Dy_U_embed: (bsz, user_len, time_len, d) # uid 从动态embedding lookup 之后的节点向量
            output: (bsz, user_len, d)
        '''
        temperature = Dy_U_embed.size(-1) ** 0.5 + episilon
        T_embed = self.time_embedding(T_idx)  # (bsz, user_len, d)

        # print(T_embed.size())
        # print(Dy_U_embed.size())

        affine = torch.einsum("bud,butd->but", T_embed, Dy_U_embed)  # (bsz, user_len, time_len)
        score = affine / temperature

        if mask is None:
            mask = torch.triu(torch.ones(score.size()), diagonal=1).bool().cuda()
            score = score.masked_fill(mask, -2 ** 32 + 1)

        alpha = F.softmax(score, dim=1)  # (bsz, user_len, time_len)
        # alpha = self.dropout(alpha)
        alpha = alpha.unsqueeze(dim=-1)  # (bsz, user_len, time_len, 1)

        att = (alpha * Dy_U_embed).sum(dim=2)  # (bsz, user_len, d)
        return att
def cumulative_average(x):
    cumulative_sum = torch.cumsum(x, dim=1)
    count = torch.arange(1, x.size(1) + 1, device=x.device).view(1, -1, 1)
    cumulative_avg = cumulative_sum / count
    return cumulative_avg


def mse_loss(y_pred, y_true):
    # 1. 计算两个张量之间的差值
    diff = y_pred - y_true

    # 2. 计算差值的平方
    squared_diff = diff ** 2

    # 3. 计算所有平方项的平均
    mse = squared_diff.mean()  # 这里 .mean() 默认会对所有维度求平均

    return mse

def add_random_edges_to_file(input_filename, output_filename, noise_ratio):
    # 读取原始数据
    with open(input_filename, 'r') as file:
        edges = [line.strip().split(',') for line in file]

    # 记录所有用户
    users = set(user for edge in edges for user in edge)
    new_edges = edges.copy()
    num_edges = len(edges)
    num_noise_edges = int(noise_ratio * num_edges)

    # 添加随机连接
    for _ in range(num_noise_edges):
        user1 = random.choice(list(users))
        user2 = random.choice(list(users))
        while user1 == user2 or [user1, user2] in new_edges or [user2, user1] in new_edges:
            user1 = random.choice(list(users))
            user2 = random.choice(list(users))
        new_edges.append([user1, user2])

    # 将结果写回文件
    with open(output_filename, 'w') as file:
        for edge in new_edges:
            file.write(','.join(edge) + '\n')

    print(f"随机连接已成功添加，并保存到 '{output_filename}' 文件中。")