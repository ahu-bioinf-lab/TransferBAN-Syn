from torch import nn
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
class SENet(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SENet, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
#
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
class SENETLayer(nn.Module):
    def __init__(self, reduction_ratio=1):
        super(SENETLayer, self).__init__()
        self.reduction_ratio = reduction_ratio

    def forward(self, inputs):
            # Squeeze
        Z = torch.mean(inputs, dim=-1, keepdim=True)

            # Excitation
        reduction_size = max(1, inputs.size(1) // self.reduction_ratio)

        W_1 = nn.Linear(inputs.size(1), reduction_size).to(device)
        A_1 = F.relu(W_1(Z.view(Z.size(0), -1)).to(device))

        W_2 = nn.Linear(reduction_size, inputs.size(1)).to(device)
        A_2 = F.relu(W_2(A_1)).to(device)

            # Re-weight
        V = inputs * A_2.view(A_2.size(0), A_2.size(1), 1)

        return V

    # Example usage:
    # Assuming inputs is a torch.Tensor with shape (batch_size, channels, embedding_size)

# inputs = torch.randn(32, 64, 128)
# senet_layer = SENETLayer(reduction_ratio=3)
# output = senet_layer(inputs)
# print(output)

# import torch
# import torch.nn.functional as F
# from itertools import combinations
#
# class BilinearInteraction(torch.nn.Module):
#     def __init__(self, type, input_size, weight_size=None):
#         super(BilinearInteraction, self).__init__()
#         self.type = type
#         self.input_size = input_size
#
#         if weight_size is None:
#             weight_size = input_size
#
#         if type == "all":
#             self.W = torch.nn.Parameter(torch.randn(128, 128))
#         elif type == "each":
#             self.W_list = [torch.nn.Parameter(torch.randn(128, input_size)) for _ in range(input_size)]
#         elif type == "interaction":
#             self.W_list = [torch.nn.Parameter(torch.randn(weight_size, input_size)) for _ in range(len(combinations(range(input_size), 2)))]
#
#     def forward(self, inputs_1,inputs_2):
#         p = []
#
#         if self.type == "all":
#             #@#p = [torch.mul(torch.tensordot(v_i, self.W, dims=(-1, 0)), v_j) for v_i, v_j in combinations(inputs, 2)]
#             # p = [torch.mul(torch.matmul(v_i, self.W), v_j) for v_i, v_j in combinations(inputs, 2)]
#             p = result = torch.matmul(inputs_1, self.W)
# # 与 V_D2 进行点积
#             final_result = torch.mul(result, inputs_2)
#         # elif self.type == "each":
#         #     p = [torch.mul(torch.tensordot(inputs[i], self.W_list[i], dims=(-1, 0)), inputs[j]) for i, j in combinations(range(len(inputs)), 2)]
#         # elif self.type == "interaction":
#         #     p = [torch.mul(torch.tensordot(v[0], w, dims=(-1, 0)), v[1]) for v, w in zip(combinations(inputs, 2), self.W_list)]
#
#         return final_result
#
# # Example usage:
# # Assuming inputs is a list of torch.Tensors
# inputs_1 = torch.randn(32, 64,128)
# inputs_2 = torch.randn(32, 64,128)
# type_option = "all"
# bilinear_layer = BilinearInteraction(type_option, input_size=64, weight_size=64)
# output = bilinear_layer(inputs_1,inputs_2)
# print(output)
