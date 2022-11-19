# -*- codeing = utf-8 -*-
# @Time : 2022/11/2 20:17
# @Author : 唐寅
# @File : test.py
# @Software : PyCharm

import torch
import torch.nn as nn


# def windows2img(img_splits_hw, H_sp, W_sp, H, W):
#     """
#     img_splits_hw: B*n, C, W, W
#     """
#     B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))  # batch
#     C = int(img_splits_hw.shape[1])    # C
#
#     img = img_splits_hw.view(B, H // H_sp, W // W_sp, C, H_sp, W_sp)
#     img = img.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, C, H, W)
#     return img
#
# def img2windows(img, H_sp, W_sp):
#     """
#     img: B, C, H, W
#     img_perm: B*num_windows, C, W, W
#     """
#     B, C, H, W = img.shape
#     img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
#     # B,C,num_h,h,num_w,w -> B,num_h,num_w,h,w,C ->B*num_windows, L, C
#     img_perm = img_reshape.permute(0, 2, 4, 1, 3, 5,).contiguous().reshape(-1, C, H_sp, W_sp)
#     return img_perm
#
# data = torch.randn(2, 2, 4, 4)
# print(data, '\n')
# data_partition = img2windows(data, 2, 2)
# print(data_partition, '\n')
# data_reverse = windows2img(data_partition,2,2,4,4)
# print(data_reverse==data)

# =================================================================================================================

class test_GeLU(nn.Module):
    def __init__(self, dim=768, act_layer=nn.SELU):
        super(test_GeLU, self).__init__()
        self.act = act_layer()
        self.layer = nn.Conv2d(64, 64 , 1)

    def forward(self, x):
        x = self.act(self.layer(x))
        return x

class test_StarReLU(nn.Module):
    def __init__(self, dim=768, act_layer=nn.ReLU):
        super(test_StarReLU, self).__init__()
        self.act = act_layer()
        self.layer = nn.Conv2d(64, 64 , 1)
        self.scale = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        x = self.layer(x)
        x = self.scale * torch.pow(self.act(x), 2) + self.bias
        return x

data = torch.randn(1, 64, 56, 56)
model_gelu = test_GeLU()
model_gelu.eval()
model_gelu(data)
from fvcore.nn import FlopCountAnalysis, ActivationCountAnalysis

flops = FlopCountAnalysis(model_gelu, data)
acts = ActivationCountAnalysis(model_gelu, data)

print(f"total flops : {(flops.total()).__round__(2)}G")
print(f"total activations: {acts.total()}")
