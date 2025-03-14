import torch
from torch import nn


class ECAttn(nn.Module):
    """Constructs a ECA module.
    Args:
        channels: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channels, k_size=3):
        super(ECAttn, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 256, 416, 416)
    m = ECAttn(channels=256)
    output_tensor = m(input_tensor)
    print(output_tensor.size())
