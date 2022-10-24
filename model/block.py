import torch
import torch.nn as nn
import math
from einops.layers.torch import Rearrange
from einops import rearrange
import numpy as np
from basic_model import MultiHeadedAttention, LayerNorm, mlpHead, PositionwiseFeedForward

class skipConnectionForm(nn.Module):
    def __init__(self, in_dim=256, embed_dim=64, num_head=1, dropout=0.1, qkv_bias=False):
        super().__init__()

        self.LN = LayerNorm(in_dim)
        self.qkv = nn.Linear(in_dim, embed_dim*3, bias=qkv_bias)
        self.MHSA = MultiHeadedAttention(num_head, embed_dim, dropout)
        self.mlp = mlpHead(in_dim, embed_dim, embed_dim)
        
    def forward(self, x):
        out = self.LN(x)
        out = self.qkv(out)
        out = rearrange(out, 'B N (n C) -> n B N C', n=3)
        out_q, out_k, out_v = out[0], out[1], out[2]
        out = self.MHSA(out_q, out_k, out_v)
        x = self.mlp(x)
        return x + out
    
class PRM(nn.Module):
    def __init__(self, in_size=224, kernel_size=4 ,downsample_ratio=4, dilation_ratio=[1,6,12], in_channel=3, embed_dim=64):
        super().__init__()

        self.out_size = in_size // downsample_ratio
        self.stride = downsample_ratio
        self.kernel_size = kernel_size
        self.dilation_ratio = dilation_ratio
        self.dilated_conv = nn.ModuleList()

        for i in range(len(self.dilation_ratio)):
            # padding = ((kernel-1)*dilation+1-stride)/2
            padding = math.ceil(((self.kernel_size-1)*self.dilation_ratio[i] + 1 - self.stride) / 2)
            self.dilated_conv.append(
                nn.Sequential(nn.Conv2d(in_channels=in_channel, out_channels=embed_dim,\
                    kernel_size=self.kernel_size, dilation=self.dilation_ratio[i],\
                        stride=self.stride, padding=padding), nn.GELU()))

        self.img2seq = Rearrange('b n c w h -> b (w h) (n c)')
        self.out_dim = embed_dim * len(self.dilation_ratio)
        
    def forward(self, x):
        PR_out = self.dilated_conv[0](x).unsqueeze(dim=1)
        for i in range(1, len(self.dilation_ratio)):
            PR_out_ = self.dilated_conv[i](x).unsqueeze(dim=1)
            PR_out = torch.cat((PR_out, PR_out_), dim=1)
        out = self.img2seq(PR_out)
        return out

class ReductionCell(nn.Module):
    def __init__(self, in_size=224, kernel_size=4, in_channel=3, embed_dim=64, token_dim=64,\
                 downsample_ratio=4, dilations=[1,2,3,4], num_head=1, dropout=0, qkv_bias=False):
        super().__init__()

        PCMStride = []
        residual = downsample_ratio // 2
        for _ in range(3):
            PCMStride.append((residual > 0) + 1)
            residual = residual // 2
        assert residual == 0
        
        self.PRM = PRM(in_size, kernel_size, downsample_ratio, dilations, in_channel, embed_dim)
        self.PCM = nn.Sequential(
                nn.Conv2d(in_channel, embed_dim, kernel_size=(3, 3), stride=PCMStride[0], padding=(1, 1)),
                nn.BatchNorm2d(embed_dim),
                nn.SiLU(inplace=True),
                nn.Conv2d(embed_dim, embed_dim, kernel_size=(3, 3), stride=PCMStride[1], padding=(1, 1)),
                nn.BatchNorm2d(embed_dim),
                nn.SiLU(inplace=True),
                nn.Conv2d(embed_dim, token_dim, kernel_size=(3, 3), stride=PCMStride[2], padding=(1, 1)),
                nn.SiLU(inplace=True),
                Rearrange('b c w h -> b (w h) c')
            )
        in_dim = self.PRM.out_dim
        self.skipConnectionForm = skipConnectionForm(in_dim, embed_dim, num_head, dropout, qkv_bias)
        
    def forward(self, x):
        if len(x.shape) < 4:
            B, N, C  = x.shape
            n = int(np.sqrt(N))
            x = rearrange(x, 'b (n1 n2) c -> b c n1 n2', n1=n, n2=n)
        PCM_x = self.PCM(x)
        MHSA_x = self.skipConnectionForm(self.PRM(x))
        out = PCM_x + MHSA_x
        return out

class NormalCell(nn.Module):
    def __init__(self, embed_dim=64, mlp_ratio=4.0, num_head=1, dropout=0.1, qkv_bias=False, class_token=True):
        super().__init__()

        mlp_dim = int(embed_dim * mlp_ratio)
        self.class_token = class_token
        self.PCM = nn.Sequential(
                nn.Conv2d(embed_dim, mlp_dim, kernel_size=(3, 3), stride=1, padding=(1, 1)),
                nn.BatchNorm2d(mlp_dim),
                nn.SiLU(inplace=True),
                nn.Conv2d(mlp_dim, embed_dim, kernel_size=(3, 3), stride=1, padding=(1, 1)),
                nn.BatchNorm2d(embed_dim),
                nn.SiLU(inplace=True),
                nn.Conv2d(embed_dim, embed_dim, kernel_size=(3, 3), stride=1, padding=(1, 1)),
                nn.SiLU(inplace=True),
                Rearrange('b c w h -> b (w h) c')
            )
        self.skipConnectionForm = skipConnectionForm(embed_dim, embed_dim, num_head, dropout, qkv_bias)
        self.lnMlp = nn.Sequential(
                LayerNorm(embed_dim),
                PositionwiseFeedForward(embed_dim, mlp_dim, dropout)
            )
        
    def forward(self, x):
        B, N, C = x.shape
        if self.class_token:
            N = N - 1
            n = int(math.sqrt(N))
            x_nocls = rearrange(x[:, 1:, :], 'b (n1 n2) c -> b c n1 n2', n1=n, n2=n)
            PCM_x = self.PCM(x_nocls)
            x = self.skipConnectionForm(x)
            x[:, 1:, :] = x[:, 1:, :] + PCM_x
        else:
            n = int(math.sqrt(N))
            x_nocls = rearrange(x, 'b (n1 n2) c -> b c n1 n2', n1=n, n2=n)
            PCM_x = self.PCM(x_nocls)
            x = self.skipConnectionForm(x)
            x = x + PCM_x

        mlp_x = self.lnMlp(x)
        x = x + mlp_x
        return x

if __name__ == '__main__':
    # net1 = ReductionCell()
    # input = torch.ones((16,224*224,3))
    # output1 = net1(input)
    # print(output1.shape)
    
    net2 = NormalCell()
    input = torch.ones((16,3137,64))
    output2 = net2(input)
    print(output2.shape)

        