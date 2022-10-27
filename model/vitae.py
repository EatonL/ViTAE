import torch
import torch.nn as nn
from block import ReductionCell, NormalCell
from basic_model import PositionalEncoding

class ViTAE(nn.Module):
    def __init__(self, img_size=224, in_channel=3, embed_dim=64,\
        class_token=True, R_layers=3, N_layers=3, class_num=10):
        super().__init__()

        self.cls = class_token
        
        self.Reductionlayers = nn.Sequential()

        for i in range(R_layers):
            in_size = img_size/2**(i+1)
            if i == 0:
                downsample_ratio = 4
                R_channel = in_channel
            else:
                downsample_ratio = 2
                R_channel = embed_dim
                
            self.Reductionlayers.add_module('{}'.format(i),\
                ReductionCell(in_size=in_size, in_channel=R_channel, embed_dim=embed_dim,\
                    downsample_ratio=downsample_ratio))

        self.PE = PositionalEncoding(d_model=embed_dim, dropout=0.1, cls=class_token)

        self.Normallayers = nn.Sequential()
        for i in range(N_layers):
            self.Normallayers.add_module('{}'.format(i),\
                NormalCell(embed_dim=embed_dim))

        self.head = nn.Sequential(
            nn.Linear(embed_dim, int((embed_dim + class_num)/2)),
            nn.Linear(int((embed_dim + class_num)/2), int((embed_dim + class_num)/2)),
            nn.Linear(int((embed_dim + class_num)/2),class_num)
        )
        
    def forward(self, x):
        x = self.Reductionlayers(x)
        x = self.PE(x)
        x = self.Normallayers(x)
        if self.cls:
            x = x[:, 0]
        else:
            x.mean(dim=1)
        x = self.head(x)
        return x

if __name__ == '__main__':
    net = ViTAE()
    input = torch.ones((8,3,224,224))
    out = net(input)
    print(out.shape)

    