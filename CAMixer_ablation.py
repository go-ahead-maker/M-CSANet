# --------------------------------------------------------------------------------
# CA-Mixer: CA-Mixer: On The Integration Of Convolution And
# Self-Attention Under Multi-Scale Token Embedding

# Copyright (c) 2022 Department of CST, Nanjing Tech University
# All Rights Reserved.
# Written by Yin Tang
# --------------------------------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# MPViT: https://git.io/MPViT
# --------------------------------------------------------------------------------


import math
from functools import partial

import numpy as np
import torch
from einops import rearrange
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from torch import einsum, nn
import torch.nn.functional as F

import torch.utils.checkpoint as checkpoint


__all__ = [
    "ca_tiny",
    "ca_small",
    "ca_base",
]

def _cfg_camixer(url="", **kwargs):
    """configuration of mpvit."""
    return {
        "url": url,
        "num_classes": 100,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "crop_pct": 0.9,
        "interpolation": "bicubic",
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "first_conv": "patch_embed.proj",
        "classifier": "head",
        **kwargs,
    }


class Mlp(nn.Module):
    """Feed-forward network (FFN, a.k.a.

    MLP) class.
    """
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        self.hidden_features = hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        # self.dwconv = nn.Conv2d(in_channels=hidden_features, out_channels=hidden_features,
        #                         kernel_size=3, padding=1, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, size):
        # H, W = size
        # B, C = x.shape[0], x.shape[2]

        x = self.fc1(x)
        # x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        # x = x + self.dwconv(x)
        # x = x.reshape(B, self.hidden_features, -1).transpose(2, 1)
        x = self.drop(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class CMlp(nn.Module):
    """
    convolutional mlp
    """
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.norm1 = nn.BatchNorm2d(hidden_features)
        self.norm2 = nn.BatchNorm2d(out_features)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        """foward function"""
        x = self.norm1(self.fc1(x))
        x = self.act(x)
        x = self.drop(x)
        x = self.norm2(self.fc2(x))
        x = self.drop(x)
        return x

# 做普通Conv用
class Conv2d_BN(nn.Module):
    """Convolution with BN module."""
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size=1,
        stride=1,
        pad=0,
        dilation=1,
        groups=1,
        bn_weight_init=1,
        norm_layer=nn.BatchNorm2d,
        act_layer=None,
    ):
        super().__init__()

        self.conv = torch.nn.Conv2d(in_ch, out_ch, kernel_size, stride, pad, dilation, groups,bias=False)
        self.bn = norm_layer(out_ch)
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)
        for m in self.modules():
            # 对于卷积的一种weight初始化方法
            if isinstance(m, nn.Conv2d):
                # Note that there is no bias due to BN
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))

        self.act_layer = act_layer() if act_layer is not None else nn.Identity(
        )

    def forward(self, x):
        """foward function"""
        x = self.conv(x)
        x = self.bn(x)
        x = self.act_layer(x)

        return x


# 做DWconv用
class DWConv2d_BN(nn.Module):
    """Depthwise Separable Convolution with BN module."""
    def __init__(
        self,in_ch,out_ch,kernel_size=1,stride=1,norm_layer=nn.BatchNorm2d,
            act_layer=nn.GELU,bn_weight_init=1,
    ):
        super().__init__()

        # dw
        self.dwconv = nn.Conv2d(in_ch,out_ch,kernel_size,stride,(kernel_size - 1) // 2,groups=out_ch,bias=False)
        # pw-linear
        self.pwconv = nn.Conv2d(out_ch, out_ch, 1, 1, 0, bias=False)
        self.bn = norm_layer(out_ch)
        self.act = act_layer() if act_layer is not None else nn.Identity()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(bn_weight_init)
                m.bias.data.zero_()

    def forward(self, x):
        """
        foward function
        """
        x = self.dwconv(x)
        x = self.pwconv(x)
        x = self.act(x)
        x = self.bn(x)

        return x



class DWCPatchEmbed(nn.Module):
    """Depthwise Convolutional Patch Embedding layer Image to Patch
    Embedding."""
    def __init__(self,
                 in_chans=3,
                 embed_dim=768,
                 patch_size=16,
                 stride=1,
                 act_layer=nn.GELU):
        super().__init__()

        self.patch_conv = DWConv2d_BN(in_chans,embed_dim,kernel_size=patch_size,stride=stride,act_layer=act_layer)

    def forward(self, x):
        """foward function"""
        x = self.patch_conv(x)

        return x


class Patch_Embed_stage(nn.Module):
    """Depthwise Convolutional Patch Embedding stage comprised of
    `DWCPatchEmbed` layers."""
    def __init__(self, embed_dim, num_path=4, isPool=False, is_attn=False):
        # isPool:是否是需要进行下采样
        # num_path:当前stage采用几条分支
        super(Patch_Embed_stage, self).__init__()
        self.num_path = num_path
        self.is_attn = is_attn
        if is_attn:
            self.patch_embeds = DWCPatchEmbed(
                in_chans=embed_dim,
                embed_dim=embed_dim,
                patch_size=3,
                stride=2 if isPool else 1,
                )
        else:
            self.patch_embeds = nn.ModuleList([
                DWCPatchEmbed(
                    in_chans=embed_dim,
                    embed_dim=embed_dim,
                    patch_size=3,
                    # 对于stage之间的embed，ispool为True,且对于n个path来说，只需第一个path上下采样两倍
                    # 后续的n-1个path都是在第一个path的输出上继续做conv,因此不需要进行下采样了
                    # 对于两路分支：两个path的输出对应的感受野为3x3,5x5
                    # 对与三路分支: 三个path的输出对应的感受野为3x3,5x5,7x7
                    stride=2 if isPool and idx == 0 else 1,
                ) for idx in range(num_path)
            ])


    def forward(self, x):
        if self.is_attn:  # sa模块全部为3x3感受野
            x = self.patch_embeds(x)
            sa_inputs = [x] * self.num_path
            return sa_inputs
        else:             # conv模块为依次增加
            att_inputs = []
            for pe in self.patch_embeds:
                x = pe(x)
                att_inputs.append(x)
            return att_inputs


# condition position encoding——————————factorized attn中的第一步
class ConvPosEnc(nn.Module):
    """Convolutional Position Encoding.

    Note: This module is similar to the conditional position encoding in CPVT.
    """
    def __init__(self, dim, k=3):
        """init function"""
        super(ConvPosEnc, self).__init__()

        self.proj = nn.Conv2d(dim, dim, k, 1, k // 2, groups=dim)

    def forward(self, x, size):  # B,N,C / B,C,H,W
        """foward function"""
        if len(x.shape) == 3:   # for sa
            B, N, C = x.shape
            H, W = size

            feat = x.transpose(1, 2).view(B, C, H, W)
            x = self.proj(feat) + feat
            x = x.flatten(2).transpose(1, 2).contiguous()
        else:
            x = x + self.proj(x)    # for lca

        return x  # B,N,C / B,C,H,W


class ConvRelPosEnc(nn.Module):
    """Convolutional relative position encoding."""
    def __init__(self, Ch, h, window):
        """Initialization.

        Ch: Channels per head.
        h: Number of heads.
        window: Window size(s) in convolutional relative positional encoding.
                It can have two forms:
                1. An integer of window size, which assigns all attention heads
                   with the same window size in ConvRelPosEnc.
                2. A dict mapping window size to #attention head splits
                   (e.g. {window size 1: #attention head split 1,
                          window size 2: #attention head split 2})
                   It will apply different window size to
                   the attention head splits.

        crpe_window={
            3: 2,
            5: 3,
            7: 3
        },
        """
        super().__init__()

        if isinstance(window, int):
            # Set the same window size for all attention heads.
            window = {window: h}
            self.window = window
        elif isinstance(window, dict):
            self.window = window
        else:
            raise ValueError()

        self.conv_list = nn.ModuleList()
        self.head_splits = []
        for cur_window, cur_head_split in window.items():
            dilation = 1  # Use dilation=1 at default.
            padding_size = (cur_window + (cur_window - 1) *
                            (dilation - 1)) // 2
            # 多尺度DWConv,对于每个path进来的输入都有三种大小的kernel的卷积
            cur_conv = nn.Conv2d(
                cur_head_split * Ch,
                cur_head_split * Ch,
                kernel_size=(cur_window, cur_window),
                padding=(padding_size, padding_size),
                dilation=(dilation, dilation),
                groups=cur_head_split * Ch,
            )
            self.conv_list.append(cur_conv)  # 记录三种尺度的DWConv
            self.head_splits.append(cur_head_split)
        self.channel_splits = [x * Ch for x in self.head_splits]  # [2*Ch,3*Ch,3*Ch]

    def forward(self, q, v, size):
        """foward function"""
        B, h, N, Ch = q.shape
        H, W = size
        assert N == H * W, 'wrong tokens number'
        # We don't use CLS_TOKEN
        q_img = q
        v_img = v

        # 还原成卷积的输入格式，并且合并了所有heads的channel(因为这里实际上不是做MHSA,而是多尺度的卷积）
        v_img = rearrange(v_img, "B h (H W) Ch -> B (h Ch) H W", H=H, W=W)
        # Split according to channels.
        # 将v按照channel比例重新划分 -> 2:3:3
        v_img_list = torch.split(v_img, self.channel_splits, dim=1)
        # 将conv_list(不同kernel的conv算子)与划分出来的v进行一一对应,然后执行DWconv
        conv_v_img_list = [
            conv(x) for conv, x in zip(self.conv_list, v_img_list)
        ]
        # 做完三种尺度的DWConv后再在channel维度合并起来
        conv_v_img = torch.cat(conv_v_img_list, dim=1)   # B,C,H,W
        # Shape: [B, h*Ch, H, W] -> [B, h, H*W, Ch].
        conv_v_img = rearrange(conv_v_img, "B (h Ch) H W -> B h (H W) Ch", h=h)

        # q与经过Dwconv后的v做张量逐元素乘积(哈达玛积)
        EV_hat_img = q_img * conv_v_img
        EV_hat = EV_hat_img
        return EV_hat          # B,h,N,C_h


class StandardAttention(nn.Module):
    """Factorized attention with convolutional relative position encoding
    class."""
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.,
        proj_drop=0.,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        assert int(head_dim) - float(head_dim) == 0, "wrong scale of num_heads!"
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """foward function"""
        B, N, C = x.shape

        # Generate Q, K, V.
        qkv = (self.qkv(x).reshape(B, N, 3, self.num_heads,
                                   C // self.num_heads).permute(2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]  # B,num_heads,N,C_heads
        q = q * self.scale
        attn = k.transpose(-2, -1) @ q
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (v @ attn).transpose(1, 2).reshape(B, N, C)
        # Output projection.
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class FactorAtt_ConvRelPosEnc(nn.Module):
    """Factorized attention with convolutional relative position encoding
    class."""
    def __init__(
        self,
        dim,
        num_heads=8,
        num_path=3,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.,
        proj_drop=0.,
        shared_crpe=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_path = num_path
        head_dim = dim // num_heads
        assert int(head_dim) - float(head_dim) == 0, "wrong scale of num_heads!"
        self.scale = qk_scale or head_dim**-0.5

        # qkv分开映射
        self.q = nn.Linear(dim, dim, qkv_bias)
        self.k = nn.Linear(dim, dim, qkv_bias)
        self.v = nn.Linear(dim, dim, qkv_bias)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Shared convolutional relative position encoding.
        self.crpe = shared_crpe

        # 在3x3基础上继续扩大感受野
        self.patch_embed = nn.ModuleList([
            DWCPatchEmbed(in_chans=dim,
                          embed_dim=dim,
                          patch_size=3,
                          stride=1
        ) for idx in range(num_path - 1)
        ])

    def forward(self, x, size, idx):  #B,N,C
        """foward function"""
        B, N, C = x.shape
        attn_inputs=[x]
        x = x.transpose(-2, -1).reshape(B, C, size[0], size[1])
        for pe in self.patch_embed:
            x = pe(x)
            attn_inputs.append(x.flatten(2).transpose(-2, -1))

        q = attn_inputs[idx]

        if idx == 0:
            k, v = (attn_inputs[idx + 1], attn_inputs[idx + 2])
        elif idx > 0 and idx < self.num_path - 1:
            k, v = (attn_inputs[idx - 1], attn_inputs[idx + 1])
        else:
            k, v = (attn_inputs[idx - 2], attn_inputs[idx - 1])

        # Generate Q, K, V.
        q = self.q(q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(k).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(v).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # Factorized attention:Q @ (softmax(k).T @ V) / sqrt(C)
        k_softmax = k.softmax(dim=2)
        dd_trans = k_softmax.transpose(-2, -1) @ v   # B, h, C_h, C_h
        factor_attn = q @ dd_trans

        # Convolutional relative position encoding.
        crpe = self.crpe(q, v, size=size)

        # Merge and reshape.
        # B,h,N,C_h -> B,N,h,C_h -> B,N,C
        x = self.scale * factor_attn + crpe
        x = x.transpose(1, 2).reshape(B, N, C)

        # Output projection.
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


# stage3/4 transformer blocks pipeline
class MHCABlock(nn.Module):
    """Multi-Head Convolutional self-Attention block."""
    def __init__(
        self,
        dim,
        num_heads,
        num_path=3,
        mlp_ratio=3,
        drop_path=0.0,
        qkv_bias=True,
        qk_scale=None,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        shared_cpe=None,
        shared_crpe=None,
        attn_category='standard'
    ):
        super().__init__()

        self.cpe = shared_cpe
        self.attn_category = attn_category
        # 根据选择的attn方式构建mhsa
        self.MHSA = ChannelWiseAttention(
            dim,
            qkv_bias=True,
            qk_scale=qk_scale,
        ) if attn_category == 'standard' else \
            FactorAtt_ConvRelPosEnc(
                dim,
                num_heads=num_heads,
                num_path=num_path,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                shared_crpe=shared_crpe,
            )

        self.mlp = Mlp(in_features=dim, hidden_features=dim * mlp_ratio,)
        self.drop_path = DropPath(drop_prob=drop_path) if drop_path > 0 else nn.Identity()

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x, size, idx):   # B,N,C
        """foward function"""
        # 先过一层cep,然后进入transformer block
        x = self.cpe(x, size)
        if self.attn_category == 'standard':
            x = x + self.drop_path(self.norm1(self.MHSA(x)))
            x = x + self.drop_path(self.norm2(self.mlp(x, size)))
        else:
            x_norm = self.norm1(x)
            x = x + self.drop_path(self.MHSA(x_norm, size, idx))
            x_norm = self.norm2(x)
            x = x + self.drop_path(self.mlp(x_norm, size))
        return x  # B, N, C


class MHCAEncoder(nn.Module):
    """Multi-Head Convolutional self-Attention Encoder comprised of `MHCA`
    blocks."""
    def __init__(
        self,
        dim,
        num_layers=1,
        num_heads=8,
        num_path=3,
        mlp_ratio=3,
        drop_path_list=[],
        qk_scale=None,
        crpe_window={
            3: 2,
            5: 3,
            7: 3
        },
        attn_category='standard',
    ):
        super().__init__()

        self.cpe = ConvPosEnc(dim, k=3)
        self.crpe = ConvRelPosEnc(Ch=dim // num_heads,
                                  h=num_heads,
                                  window=crpe_window)
        # 对于每个path的encoder都有若干个block堆叠
        self.MHCA_layers = nn.ModuleList([
            MHCABlock(
                dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path_list[idx],
                qk_scale=qk_scale,
                shared_cpe=self.cpe,
                shared_crpe=self.crpe,
                attn_category=attn_category   # 使用标准attn还是factorized attn
            ) for idx in range(num_layers)
        ])


    def forward(self, x, size, idx):  # B,N,C
        """foward function"""
        H, W = size
        B = x.shape[0]

        for layer in self.MHCA_layers:
            x = layer(x, (H, W), idx)
        # return x's shape : [B, N, C] -> [B, C, H, W]
        # X返回后需要进行concat以及1x1conv，因此返回卷积形式的维度
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x



class MHCA_stage(nn.Module):
    """Multi-Head Convolutional self-Attention stage comprised of `MHCAEncoder`
    layers."""
    def __init__(
        self,
        embed_dim,
        out_embed_dim,
        num_layers=1,
        num_heads=8,
        mlp_ratio=3,
        num_path=4,
        drop_path_list=[],
        attn_category='standard',
    ):
        super().__init__()

        # MP-Transformer Block分为两个模块：multi-path conv and encoders + G-L feature interaction
        # 根据当前stage中的num_path生成若干个encoder
        crpe_window={
            3: 2,
            5: 2,
        } if num_heads == 4 else {
            3: 2,
            5: 3,
            7: 3
        }
        self.mhca_blks = nn.ModuleList([
            MHCAEncoder(
                embed_dim,
                num_layers,   # 每个Transformer encoder中又堆叠了若干个encoder blocks
                num_heads,
                num_path,
                mlp_ratio,
                drop_path_list=drop_path_list,
                attn_category=attn_category,
                crpe_window=crpe_window,
            ) for idx in range(num_path)
        ])

        self.aggregate = Conv2d_BN(embed_dim * num_path,
                                   out_embed_dim,
                                   act_layer=nn.GELU)

    def forward(self, inputs):        # inputs:由前面的MS-patchembed送进来的若干个输入特征图  B,C,H,W
        """foward function"""
        att_outputs = []
        for idx, (x, encoder) in enumerate(zip(inputs, self.mhca_blks), 0):
            # [B, C, H, W] -> [B, N, C]
            _, _, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)
            # 每个特征图进入各自的encoder中进行正向传播,得到num_path个输出,添加到attn_outputs中
            att_outputs.append(encoder(x, size=(H, W), idx=idx))  # idx用于标识encoder

        out_concat = torch.cat(att_outputs, dim=1)      # 将所有输出在channel维度进行拼接
        out = self.aggregate(out_concat)   # 然后用1x1 conv将channel映射到指定维度(下一个stage的维度)

        return out   # B,C_next_stage,H,W


# conv attention block
class LCAModel(nn.Module):
    def __init__(self,
                 dim,
                 dw_kernel=5,
                 dw_d_kernel=7):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, dw_kernel, padding= dw_kernel // 2, groups=dim)  # DWconv
        # to fix the H and W,we need to pad the fmap with padding = (K + (dilation-1)*(K-1)) // 2
        self.dw_d_conv = nn.Conv2d(dim, dim, dw_d_kernel, stride=1,
                                   padding=(dw_d_kernel + 2 * (dw_d_kernel-1)) // 2,
                                   groups=dim, dilation=3)
        self.pwconv = nn.Conv2d(dim, dim, 1)
        self.norm = nn.BatchNorm2d(dim)

        self.proj1 = nn.Conv2d(dim, dim // 4, 1)
        self.proj2 = nn.Conv2d(dim // 4, dim, 1)
        self.act = nn.ReLU()
        self.sig = torch.sigmoid

    def forward(self, x):
        adaptive = x
        attn = self.dwconv(x)
        attn = self.pwconv(attn)
        attn = self.norm(attn)
        attn = self.dw_d_conv(attn)

        adaptive = self.act(self.proj1(adaptive))
        adaptive = self.proj2(adaptive)
        adaptive = self.sig(adaptive)

        return adaptive * attn        # B,C,H,W


# channel-wise attention
class ChannelWiseAttention(nn.Module):
    """Channel-wise attention for image-level global view"""

    def __init__(
        self,
        num_channel_tokens, # 原channel数
        num_groups = 8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.,
        proj_drop=0.,
    ):
        super().__init__()
        self.num_groups = num_groups
        self.channels_per_groups = channels_per_groups = num_channel_tokens // num_groups      # Cg
        self.num_heads = 1          # 默认使用head为1的token-wise dim 划分
        self.scale = qk_scale or channels_per_groups**-0.5
        # trainable tau for cosine attention scaling
        # self.scale = nn.Parameter(torch.log(10 * torch.ones((1, 1, 1))), requires_grad=True) # num_heads = 1

        self.qkv = nn.Linear(num_channel_tokens, num_channel_tokens * 3, bias=qkv_bias)
        self.proj = nn.Linear(num_channel_tokens, num_channel_tokens)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x):
        """foward function"""
        B, N, C = x.shape   # 未经过transpose的输入
        # setattr(self, 'proj', nn.Linear(C, C))
        # Generate Q, K, V.
        qkv = self.qkv(x).transpose(-2, -1).\
            reshape(B, 3, self.num_groups, self.channels_per_groups, self.num_heads,
                                   N // self.num_heads).permute(1, 0, 2, 4, 3, 5).\
            reshape(3, -1, self.num_heads, self.channels_per_groups, N // self.num_heads)

        q, k, v = qkv[0], qkv[1], qkv[2]  # B*g, 1, Cg, N

        # channel-wise attention
        attn = q @ k.transpose(-2, -1)
        attn = attn * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).reshape(B, self.num_groups, self.num_heads, self.channels_per_groups, -1).\
                       permute(0, 1, 3, 2, 4).reshape(B, C, N).transpose(-2, -1)

        # Output projection.
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class DynamicDWConv(nn.Module):
    def __init__(self,
                 dim=96,
                 act_layer=nn.GELU,
                 dw_kernel=5,
                 dw_d_kernel=7,):

        super().__init__()

        self.conv_proj1 = nn.Conv2d(dim, dim , 1)
        self.conv_proj2 = nn.Conv2d(dim, dim , 1)
        self.act = act_layer()

        # dynamic depth-wise convolution
        self.lca = LCAModel(dim=dim,
                       dw_kernel=dw_kernel,
                       dw_d_kernel=dw_d_kernel,
                       )

    def forward(self, x):
        shotcut = x
        x = self.conv_proj1(x)
        x = self.act(x)
        x = self.lca(x)
        x = self.conv_proj2(x)
        x = x + shotcut

        return x


# stage1/2 local attn block pipline
class LocalConvAttnention(nn.Module):
    def __init__(self, dim,
                 drop_rate = 0.,
                 mlp_ratio = 4,
                 dw_kernel=5,
                 dw_d_kernel=7,):
        super().__init__()

        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.dynamicdwconv = DynamicDWConv(dim=dim,
                                           dw_kernel=dw_kernel,
                                           dw_d_kernel=dw_d_kernel)

        # layer scale
        layer_scale_init_value = 1e-4
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

        # stochastic depth
        self.drop_path = DropPath(drop_rate) if drop_rate > 0.0 else nn.Identity()
        self.CMLP = CMlp(dim, dim * mlp_ratio, dim,drop = drop_rate)
        self.CPE = ConvPosEnc(dim=dim, k=3)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self,x):  # B,C,H,W
        _, _, H, W = x.shape
        x = self.CPE(x,(H,W))

        x_norm = self.norm1(x)
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.dynamicdwconv(x_norm))

        x_norm = self.norm2(x)
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.CMLP(x_norm))
        return x

# local attn encoder(include multiple conv attn blocks)#
class LocalConvAttnEncoder(nn.Module):
    def __init__(self,
                 embed_dim,
                 drop_path_list=[],
                 num_layers=1,      # 每个encoder中包含多少个block
                 mlp_ratio=4,
                 dw_kernel=5,
                 dw_d_kernel=7,
                 ):
        super().__init__()

        # 构建多个blk的堆叠
        self.localconv_blks = nn.ModuleList([
            LocalConvAttnention(dim = embed_dim,
                           drop_rate=drop_path_list[idx],
                           mlp_ratio=mlp_ratio,
                           dw_d_kernel=dw_d_kernel,
                           dw_kernel=dw_kernel,
                           ) for idx in range(num_layers)
        ])


    def forward(self, x):  # B,C,H,W
        for blk in self.localconv_blks:
            x = blk(x)
        return x


# stage1/2 local conv pipline
class MPLocalConvAttnStage(nn.Module):
    def __init__(self,
                 embed_dim,
                 out_embed_dim,
                 num_groups=8,
                 num_layers=1,
                 mlp_ratio=3,
                 num_path=2,
                 drop_path_list=[],
                 norm_layer = partial(nn.LayerNorm, eps=1e-6),
                 dw_kernel=5,
                 dw_d_kernel=7,
                 ):
        super().__init__()
        # 根据path数构建对应个数的encoder
        self.convattn_encoders = nn.ModuleList([
            LocalConvAttnEncoder(embed_dim = embed_dim,
                                 drop_path_list = drop_path_list,
                                 num_layers = num_layers,
                                 mlp_ratio = mlp_ratio,
                                 dw_kernel=dw_kernel,
                                 dw_d_kernel=dw_d_kernel,
                                 ) for idx in range(num_path)
        ])
        self.channel_attn_norm1 = nn.LayerNorm(embed_dim)
        self.channel_attn_norm2 = nn.LayerNorm(embed_dim)
        self.mlp = Mlp(embed_dim, embed_dim * mlp_ratio,)
        self.ChannelWiseAttention = ChannelWiseAttention(num_channel_tokens=embed_dim,
                                                         num_groups=num_groups,
                                                         qkv_bias=True,
                                                         )   # 输出: B,N,C
        self.out_embed_dim = out_embed_dim
        self.norm = norm_layer(out_embed_dim)   # 1x1 conv后dim为下一个stage的维度
        self.aggregation = Conv2d_BN(in_ch=embed_dim * (num_path + 1),
                                     out_ch=out_embed_dim,
                                     act_layer=nn.GELU)
        self.drop_path = DropPath(drop_prob=drop_path_list[num_layers // 2]) if drop_path_list[0] > 0.0 else nn.Identity()

    def forward(self, inputs):  # 由patch embed送进来的若干个输入 B,C,H,W
        B, C, H, W = inputs[0].shape
        channel_attn_input = inputs[0].reshape(B, C, -1).transpose(-2, -1)  # B,N,C
        channel_attn_input = channel_attn_input + self.drop_path(self.ChannelWiseAttention(self.channel_attn_norm1(channel_attn_input)))
        channel_attn_input = channel_attn_input + self.drop_path(self.mlp(self.channel_attn_norm2(channel_attn_input), (H,W)))
        attn_outputs = [channel_attn_input.transpose(-2, -1).reshape(B, C, H, W)]

        for input, encoder in zip(inputs, self.convattn_encoders):
            attn_outputs.append(encoder(input))

        output = torch.cat(attn_outputs, dim=1)
        output = self.aggregation(output)   # B,C_next, H , W
        output = output.reshape(B, self.out_embed_dim, -1).permute(0,2,1).contiguous()
        output = self.norm(output).permute(0, 2, 1).reshape(B, self.out_embed_dim, H, W)

        return output



class Cls_head(nn.Module):
    """a linear layer for classification."""
    def __init__(self, embed_dim, num_classes):
        """initialization"""
        super().__init__()

        self.cls = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """foward function"""
        # (B, C, H, W) -> (B, C, 1)

        x = nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        # Shape : [B, C]
        out = self.cls(x)
        return out


def dpr_generator(drop_path_rate, num_layers, num_stages):
    """Generate drop path rate list following linear decay rule."""
    dpr_list = [x.item() for x in torch.linspace(0, drop_path_rate, sum(num_layers))]
    dpr = []
    cur = 0
    for i in range(num_stages):
        dpr_per_stage = dpr_list[cur:cur + num_layers[i]]
        dpr.append(dpr_per_stage)
        cur += num_layers[i]

    return dpr


class CAMixer(nn.Module):
    """Multi-Path ViT class."""
    def __init__(
        self,
        img_size=224,
        num_stages=4,
        num_path=[2, 3, 2, 3],         # 每个stage对应的path数
        num_layers=[1, 1, 1, 1],       # 每个path对应的encoder中含有多少个block
        embed_dims=[64, 128, 256, 512],
        mlp_ratios=[8, 8, 4, 4],
        num_heads=[8, 8, 8, 8],
        drop_path_rate=0.,
        in_chans=3,
        num_classes=1000,
        attn_category = [-1, 'standard', -1, 'standard'],
        dw_kernel = [5, -1, 5, -1],
        dw_d_kernel = [7, -1, 3, -1],
        num_groups=[2, -1, 4, -1 ],
        use_chk=False,
        **kwargs,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_stages = num_stages
        self.use_chk = use_chk

        # drop path ratio :[[dpr_stage],[],...]_
        dpr = dpr_generator(drop_path_rate, num_layers, num_stages)

        # conv stem
        self.stem = nn.Sequential(
            Conv2d_BN(
                in_chans,
                embed_dims[0] // 2,
                kernel_size=3,
                stride=2,
                pad=1,
                act_layer=nn.GELU,
            ),
            Conv2d_BN(
                embed_dims[0] // 2,
                embed_dims[0],
                kernel_size=3,
                stride=2,
                pad=1,
                act_layer=nn.GELU,
            ),
        )

        # Patch embeddings.只改变特征图大小，不改变通道数(通道由MPTB block的1x1 conv来改变)
        self.patch_embed_stages = nn.ModuleList([
            Patch_Embed_stage(
                embed_dims[idx],
                num_path=num_path[idx],
                isPool=False if idx == 0 else True,  # stem后的第一个embed层不需要进行下采样
                is_attn=True if idx % 2 == 1 else False
            ) for idx in range(self.num_stages)
        ])

        # Multi-Head Convolutional Self-Attention (MHCA)
        self.mhca_stages = nn.ModuleList([])
        for idx in range(self.num_stages):
            if idx % 2 == 0:
                self.mhca_stages.append(
                    MPLocalConvAttnStage(embed_dim=embed_dims[idx],
                                         out_embed_dim=embed_dims[idx + 1],
                                         num_layers=num_layers[idx],
                                         mlp_ratio=mlp_ratios[idx],
                                         num_path=num_path[idx],
                                         drop_path_list=dpr[idx],   # list
                                         dw_kernel=dw_kernel[idx],
                                         dw_d_kernel=dw_d_kernel[idx],
                                         num_groups=num_groups[idx],
                                         ))
            else:
                self.mhca_stages.append(
                    MHCA_stage(
                                embed_dims[idx],     # cur_dim
                                embed_dims[idx + 1]  # next_stage_dim(由最后的1x1来改变)
                                if not (idx + 1) == self.num_stages else embed_dims[idx],  # 最后一个stage不改变dim

                                num_layers[idx],  # 当前stage多少个block
                                num_heads[idx],
                                mlp_ratios[idx],
                                num_path[idx],    # 由MS-PE喂进来多少个fmaps(3-4)
                                drop_path_list=dpr[idx],
                                attn_category=attn_category[idx],    # attention类型
                            ))


        # Classification head.
        self.cls_head = Cls_head(embed_dims[-1], num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """initialization"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_classifier(self):
        """get classifier function"""
        return self.head

    def frezze_stem(self):
        """frezzing stem parameters"""
        self.stem.requires_grad_(False)

    def forward_features(self, x):
        """forward feature function"""

        # x's shape : [B, C, H, W]

        x = self.stem(x) # Shape : [B, C, H/4, W/4]

        for idx in range(self.num_stages):
            att_inputs = self.patch_embed_stages[idx](x)

            if self.use_chk:
                x = checkpoint.checkpoint(self.mhca_stages[idx], att_inputs)
            else:
                x = self.mhca_stages[idx](att_inputs)
        return x

    def forward(self, x):
        """foward function"""
        x = self.forward_features(x)

        # cls head
        out = self.cls_head(x)
        return out

@register_model
def ca_tiny(**kwargs):
    model = CAMixer(
        img_size=224,
        num_stages=4,
        num_path=[2, 3, 2, 3],
        num_layers=[1, 2, 5, 2],
        embed_dims=[64, 128, 176, 232],
        mlp_ratios=[4, 4, 4, 4],
        num_heads=[-1, 4, -1, 8],
        num_groups=[4, -1, 8, -1],
        **kwargs,
    )
    model.default_cfg = _cfg_camixer()
    return model


@register_model
def ca_small(**kwargs):
    model = CAMixer(
        img_size=224,
        num_stages=4,
        num_path=[2, 3, 2, 3],
        num_layers=[1, 3, 6, 3],
        embed_dims=[96, 128, 224, 320],
        mlp_ratios=[4, 4, 4, 4],
        num_heads=[-1, 4, -1, 8],
        num_groups=[4, -1, 8, -1],
        **kwargs,
    )
    model.default_cfg = _cfg_camixer()
    return model


@register_model
def ca_base(**kwargs):
    model = CAMixer(
        img_size=224,
        num_stages=4,
        num_path=[2, 3, 2, 3],
        num_layers=[1, 3, 8, 3],
        embed_dims=[96, 192, 288, 360],
        mlp_ratios=[4, 4, 4, 4],
        num_heads=[-1, 4, -1, 8],
        num_groups=[4, -1, 8, -1],
        **kwargs,
    )
    model.default_cfg = _cfg_camixer()
    return model

if __name__ == "__main__":
    model = ca_small()
    model.eval()
    inputs = torch.randn(1, 3, 224, 224)
    model(inputs)

    from fvcore.nn import FlopCountAnalysis, ActivationCountAnalysis

    flops = FlopCountAnalysis(model, inputs)
    param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    acts = ActivationCountAnalysis(model, inputs)

    print(f"total flops : {flops.total()}")
    print(f"total activations: {acts.total()}")
    print(f"number of parameter: {param}")