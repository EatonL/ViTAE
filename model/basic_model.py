import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from einops import repeat

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, cls=False, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.cls = cls
        if cls:
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数列
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数列
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        if self.cls:
            b, n, _ = x.shape
            cls_token = repeat(self.cls_token, '1 n d -> b n d', b=b)
            x = torch.cat([cls_token, x], dim=1)
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value):
        nbatches = query.size(0)

        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

# class SublayerConnection(nn.Module):
#     def __init__(self, size, dropout):
#         super(SublayerConnection, self).__init__()
#         self.norm = LayerNorm(size)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x, sublayer):
#         return x + self.dropout(sublayer(self.norm(x)))

# class embedding(nn.Module):
#     def __init__(self, patch_len, dim, channels=2):
#         super(embedding, self).__init__()

#         patch_dim = patch_len * channels
#         self.to_patch_embedding = nn.Sequential(
#             Rearrange('b c (h p) -> b h (p c)', p=patch_len),
#             nn.Linear(patch_dim, dim)
#         )

#     def forward(self, x):
#         x = self.to_patch_embedding(x)
#         return x

class mlpHead(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# class Encoder(nn.Module):
#     def __init__(self, layer, N):
#         super(Encoder, self).__init__()
#         self.layers = clones(layer, N)
#         self.norm = LayerNorm(layer.size)

#     def forward(self, x):
#         for layer in self.layers:
#             x = layer(x)
#         return self.norm(x)


# class EncoderLayer(nn.Module):
#     def __init__(self, size, self_attn, feed_forward, dropout):
#         super(EncoderLayer, self).__init__()
#         self.self_attn = self_attn
#         self.feed_forward = feed_forward
#         self.sublayer = clones(SublayerConnection(size, dropout), 2)
#         self.size = size

#     def forward(self, x):
#         x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x))
#         return self.sublayer[1](x, self.feed_forward)


# class Transformer(nn.Module):
#     def __init__(self, encoder, cls=None):
#         super(Transformer, self).__init__()
#         self.cls = cls
#         self.encoder = encoder

#     def forward(self, src):
#         out = self.encoder(src)
#         if self.cls == 'cls':
#             out = out[:, 0]
#         elif self.cls == 'pool':
#             out.mean(dim=1)

#         return out

