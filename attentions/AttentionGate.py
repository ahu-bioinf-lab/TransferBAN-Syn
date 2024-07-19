import torch
from torch import nn

from attentions.ca import CoordAttn
from attentions.cbam import CBAM
from attentions.eca import ECAttn
from attentions.se_net import SENet
from attentions.simAM import SimAM


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()

        '''
         MPA
        self.front_attn = MultiPathAttn(channels=F_g, reduction_ratio=32)
        self.down_attn = MultiPathAttn(channels=F_g, reduction_ratio=16, num_paths=4)
        '''
        '''SEnet
        # self.front_attn = SENet(channel=F_g)
        # self.down_attn = SENet(channel=F_l)
        '''
        ''' ECA
        self.front_attn = ECAttn(channels=F_g)
        self.down_attn = ECAttn(channels=F_l)
        '''
        # CA
        self.front_attn = CoordAttn(channel = F_g)
        self.down_attn = CoordAttn(channel = F_l)
        '''
        self.front_attn = SimAM()
        self.down_attn = SimAM()
        '''
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int))

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int))

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1), nn.Sigmoid())

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g = self.front_attn(g)
        x = self.down_attn(x)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return torch.cat([x * psi, g], 1)