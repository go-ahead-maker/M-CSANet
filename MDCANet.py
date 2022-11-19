# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
from collections import OrderedDict
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
import math
from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
from einops import rearrange
from torch import einsum
import numpy as np
from torch.utils.checkpoint import checkpoint

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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

class InvertMLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.act = act_layer()
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1, groups=hidden_features)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = x + self.dwconv(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = rearrange(x, "B C H W -> B (H W) C")
        return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class Remerge(nn.Module):
    def __init__(self, dim=768, norm_layer=nn.LayerNorm):
        super(Remerge, self).__init__()

        self.remerge_dw = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.remerge_pw = nn.Conv2d(dim, dim, kernel_size=1)
        self.norm = LayerNorm(normalized_shape=dim, eps=1e-6, data_format="channels_first")

    def forward(self, x):
        x = self.remerge_dw(x)
        x = self.remerge_pw(x)
        x = self.norm(x)

        return x

class ConvPosEnc(nn.Module):
    """Convolutional Position Encoding.
    """
    def __init__(self, dim, k=3):
        """init function"""
        super(ConvPosEnc, self).__init__()

        self.proj = nn.Conv2d(dim, dim, k, 1, k // 2, groups=dim)

    def forward(self, x):
        """foward function"""
        x = x + self.proj(x)

        return x  # B,N,C / B,C,H,W

class SpatialAttention(nn.Module):

    "standard spatial dot product self-attention module"
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.,
        proj_drop=0.,
        reso=(1,1),
        norm_layer=nn.LayerNorm,
        idx=0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        assert int(head_dim) - float(head_dim) == 0, "wrong scale of num_heads!"
        self.scale = qk_scale or head_dim**-0.5
        self.H, self.W = reso
        self.idx = idx
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)

        if idx:
            self.remerge = Remerge(dim=dim, norm_layer=norm_layer)

        self.attn_drop = nn.Dropout(attn_drop)
        # self.proj = nn.Linear(dim, dim)
        # self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """foward function"""
        B, N, C = x.shape

        if self.idx:
            x = rearrange(x, "B (H W) C -> B C H W", H=self.H, W=self.W)
            for i in range(self.idx):
                x = self.remerge(x)
            x = rearrange(x, "B C H W -> B (H W) C")

        # Generate Q, K, V.
        qkv = (self.qkv(x).reshape(B, N, 3, self.num_heads,
                                   C // self.num_heads).permute(2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]  # B,num_heads,N,C_heads

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        return x


def img2windows(img, H_sp, W_sp):
    """
    img: B, C, H, W
    img_perm: B*num_windows, C, W, W
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    # B,C,num_h,h,num_w,w -> B,num_h,num_w,h,w,C ->B*num_windows, L, C
    img_perm = img_reshape.permute(0, 2, 4, 1, 3, 5,).contiguous().reshape(-1, C, H_sp, W_sp)
    return img_perm


#还原成feature map的形状，类似swin的window_reverse
def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    img_splits_hw: B*n, C, W, W
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))  # batch
    C = int(img_splits_hw.shape[1])    # C

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, C, H_sp, W_sp)
    img = img.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, C, H, W)
    return img


class FactorAtt_ConvRelPosEnc(nn.Module):
    """Factorized attention with convolutional relative position encoding
    class."""
    def __init__(
            self,
            dim,
            resolution,
            num_heads=8,
            qkv_bias=False,
            qk_scale=None,
            proj_drop=0.0,
            norm_layer=nn.LayerNorm,
            idx=0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        assert isinstance(resolution, tuple)
        self.H, self.W = resolution
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.idx = idx
        # self.proj = nn.Linear(dim, dim)
        # self.proj_drop = nn.Dropout(proj_drop)

        # Shared convolutional relative position encoding.
        self.crpe = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        if idx:
            self.remerge = Remerge(dim=dim, norm_layer=norm_layer)
        # self.softmax = nn.Softmax(dim=2)
        self.apply(self._init_weights)

    def get_crpe(self, input, func):
        q, v = input[0], input[1]
        h = q.shape[1]
        v = rearrange(v, "B h (H W) Ch -> B (h Ch) H W", H=self.H, W=self.W)
        v = func(v)
        v = rearrange(v, "B (h Ch) H W -> B h (H W) Ch", h=h)
        crpe = q * v
        return crpe

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
        B, N, C = x.shape
        assert (self.H * self.W) == N, "wrong token size"

        if self.idx:
            x = rearrange(x, "B (H W) C -> B C H W", H=self.H, W=self.W)
            for i in range(self.idx):
                x = self.remerge(x)
            x = rearrange(x, "B C H W -> B (H W) C")

        # Generate Q, K, V.
        qkv = (self.qkv(x).reshape(B, N, 3, self.num_heads,
                                   C // self.num_heads).permute(2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]  # B,num_heads,N,C_heads

        # Factorized attention:Q @ (softmax(k).T @ V) / sqrt(C)
        k_softmax = k.softmax(dim=2)
        k_softmax_T_dot_v = einsum("b h n k, b h n v -> b h k v", k_softmax, v)
        factor_att = einsum("b h n k, b h k v -> b h n v", q,
                            k_softmax_T_dot_v)

        # Convolutional relative position encoding.
        crpe = self.get_crpe(input=[q, v], func=self.crpe)

        # Merge and reshape.
        x = self.scale * factor_att + crpe
        x = x.transpose(1, 2).reshape(B, N, C).contiguous()

        return x


class MSConv_SE(nn.Module):
    def __init__(self, dim, resolution, expand_factor=4, sse_ws=14):
        super().__init__()

        self.proj_1 = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.proj_2 = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

        self.conv_head1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )
        self.conv_head2 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, bias=False)

        self.conv_head3 = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=False)

        self.conv_head4 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=(1, 11), padding=(0, 5), groups=dim, bias=False),
            nn.Conv2d(dim, dim, kernel_size=(11, 1), padding=(5, 0), groups=dim, bias=False),
        )

        self.H, self.W = resolution
        self.window_size = sse_ws

        # SE layer
        self.se_layer_c = nn.Sequential(
            nn.Conv2d(dim, dim // expand_factor, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim // expand_factor, dim, kernel_size=1),
        )
        self.se_layer_s = nn.Sequential(
            nn.Linear(sse_ws*sse_ws, (sse_ws*sse_ws) // 2),
            nn.GELU(),
            nn.Linear((sse_ws*sse_ws) // 2, sse_ws*sse_ws),
        )

        self.sigmoid = torch.sigmoid
        self.apply(self._init_weights)

    def window_partition(self, x):  # x: B, C, H, W
        x = img2windows(x, self.window_size, self.window_size)
        return x

    def window_reverse(self, x):    # x: B*n, C, wh, ww
        x = windows2img(x, self.window_size, self.window_size, self.H, self.W)
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):  # B, L, C
        B, N, C = x.shape
        assert self.H * self.W == N, "wrong token size"
        x = rearrange(x, "B (H W) C -> B C H W", H=self.H, W=self.W) # B,C,H,W
        x = self.proj_1(x)
        _x1 = self.conv_head1(x)

        _x2 = self.conv_head2(_x1)  # 5x5 br
        _x3 = self.conv_head3(_x1)  # 7x7 br
        _x4 = self.conv_head4(_x1)  # 11 x 11 br

        x = _x1 + _x2 + _x3 + _x4
        x = self.proj_2(x)
        # x = self.conv_head1[-1](x)
        # channel se
        se_branch = self.sigmoid(self.se_layer_c(
            F.adaptive_avg_pool2d(x, 1)  # B, C, 1, 1
        ))
        x = x * se_branch   # B, C, H, W

        # local spatial se
        x_window = self.window_partition(x)  # B*n, C, wh, ww
        x_spatial_mean = torch.mean(x_window, dim=1, keepdim=True).flatten(2) # B*n, 1, N'
        se_branch = self.sigmoid(
            self.se_layer_s(x_spatial_mean)
        ).reshape(-1, 1, self.window_size, self.window_size)
        x_window = x_window * se_branch        # (B*n, C, wh, ww) x (B*n, 1, wh, ww)
        x = self.window_reverse(x_window)   # B, C, H, W
        x = rearrange(x, "B C H W -> B (H W) C  ")  # B,N,C
        return x


class MixiTBlock(nn.Module):
    '''
        Args:
            dim:dimmension of features
            reso:resolution of feature map
            split_size:sw of local windows

    '''

    def __init__(self, dim, reso, num_heads, mlp_ratio=4, qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., expand_ratio=4,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, initial_value=5e-1, sign=0, g_attn_ctg='channel',
                 heads_sp=[1, 1, 1]):

        super().__init__()
        # self.dim = dim
        self.H, self.W = reso  # H,W of feature map
        self.norm1 = norm_layer(dim)
        self.sign = sign

        assert sum(heads_sp) == num_heads, "mismatch of heads_sp and num_heads"

        hidden_dim = mlp_ratio * dim
        if sign == 0:   # local
            self.attn = MSConv_SE(
                dim=dim, resolution=(self.H, self.W), expand_factor=expand_ratio,)
        else:   # global
            temp = []
            dim_sp = dim // num_heads
            self.dims = [dim_sp*heads_sp[0], dim_sp*heads_sp[1], dim_sp*heads_sp[2]]
            if g_attn_ctg == 'channel':
                temp.extend(
                    [FactorAtt_ConvRelPosEnc(dim=dim_sp*heads_sp[i], resolution=reso, num_heads=heads_sp[i], qkv_bias=qkv_bias,
                                                qk_scale=qk_scale, proj_drop=drop, norm_layer=norm_layer, idx=i)
                     for i in range(len(heads_sp))
                     ])
            elif g_attn_ctg == 'spatial':
                temp.extend(
                    [SpatialAttention(dim=dim_sp*heads_sp[i], reso=reso, num_heads=heads_sp[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
                                             attn_drop=attn_drop, proj_drop=drop, norm_layer=norm_layer, idx=i)
                     for i in range(len(heads_sp))
                     ])
            else:
                raise ValueError('wrong category')

            self.CPE = ConvPosEnc(dim=dim, k=3)
            self.attns = nn.ModuleList(temp)
            self.proj = nn.Linear(dim, dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=hidden_dim, out_features=dim,
                       act_layer=act_layer, drop=drop) if sign == 1 else InvertMLP(
            in_features=dim, hidden_features=hidden_dim, out_features=dim,
            act_layer=act_layer, drop=drop
        )
        self.norm2 = norm_layer(dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        """#
        x: B, H*W, C
        """
        B, N, C = x.shape
        assert N == self.H * self.W, "flatten img_tokens has wrong size"

        if self.sign == 0:
            x_norm = self.norm1(x)
            attn_out = self.attn(x_norm)
        else:
            x = rearrange(x, "B (H W) C -> B C H W", H=self.H, W=self.W)  # B, C, H, W
            # conditional position encoding (CPvT)
            x = self.CPE(x)
            x = rearrange(x, "B C H W -> B (H W) C")

            x_norm = self.norm1(x)
            out1 = self.attns[0](x_norm[:, :, :self.dims[0]])
            out2 = self.attns[1](x_norm[:, :, self.dims[0]:self.dims[0]+self.dims[1]])
            out3 = self.attns[2](x_norm[:, :, sum(self.dims[:2]):])
            attn_out = torch.cat([out1, out2, out3], dim=-1)   # B, N, C
            attn_out = self.proj(attn_out)

        x = x + self.drop_path(attn_out)

        if self.sign == 0:
            x_norm = self.norm2(x)
            x_norm = rearrange(x_norm, "B (H W) C -> B C H W", H=self.H, W=self.W)  # reshape to [B, C, H, W]
        else:
            x_norm = self.norm2(x)
        x = x + self.drop_path(self.mlp(x_norm))

        return x    # B, L, C

class Merge_layer(nn.Module):
    """ Merging layer for downsampeling
    """

    def __init__(self, dim, dim_out=768, norm_layer=nn.LayerNorm):
        super().__init__()

        self.norm = norm_layer(dim_out)
        self.embed_conv = nn.Conv2d(dim, dim_out, kernel_size=3, stride=2, padding=1)

    def forward(self, x, size):
        B, N, C = x.shape
        H, W = size

        x = x.transpose(-2, -1).reshape(B, C, H, W).contiguous()
        x = self.embed_conv(x)
        C_new, H_new, W_new = x.shape[1:]  # record resolution
        x = x.reshape(B, C_new, -1).transpose(-2, -1).contiguous()  # B,L,C
        x = self.norm(x)
        return x, H_new, W_new


class MDCANet(nn.Module):
    """ MixiT: Mixed Vision Transformer for Efficient Local-global Representations Learning
    """

    def __init__(self, depth=[3, 4, 8, 3], img_size=224, in_chans=3, num_classes=1000, embed_dim=[64, 128, 320, 512],
                 num_heads=[2, 4, 8, 16],heads_sp = dict(stage1=[1, 1, 2],
                                                         stage2=[1, 1, 2],
                                                         stage3=[2, 3, 3],
                                                         stage4=[4, 6, 6]), mlp_ratio=[8, 8, 8, 8], qkv_bias=True, qk_scale=None, expand_ratio = 4,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, use_chk=False, use_mlp_cls_head=False):
        """
        Args:
            depth (list): depth of each stage
            img_size (int, tuple): input image size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (list): embedding dimension of each stage
            num_heads: heads for global part (transformer part) of each stage
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim and expand-shrink conv1x1
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer (nn.Module): normalization layer
            conv_stem (bool): whether use overlapped patch stem
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]  # stochastic depth decay rule
        self.use_chk = use_chk
        if use_chk:
            print("launching checkpoint for saving GPU memory")

        # stem cell
        self.convolutional_stem = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim[0], kernel_size=7, stride=4, padding=2),
            nn.BatchNorm2d(embed_dim[0]),
            nn.GELU(),

            nn.Conv2d(embed_dim[0], embed_dim[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embed_dim[0]),
            nn.GELU(),
        )
        self.convolutional_stem.add_module('downsample layer', nn.Conv2d(embed_dim[0], embed_dim[0], kernel_size=3, stride=1, padding=1))

        cur_index = 0   # stage index
        self.stage1 = nn.ModuleList([
            MixiTBlock(dim=embed_dim[cur_index], reso=to_2tuple(img_size//4 * 2**cur_index), num_heads=num_heads[cur_index],
                       mlp_ratio=mlp_ratio[cur_index], qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                       drop_path=dpr[i], act_layer=nn.GELU, norm_layer=norm_layer, initial_value=1e-3, expand_ratio=expand_ratio,
                       sign=0 if (i % 2 == 0) else 1, g_attn_ctg='channel', heads_sp=heads_sp['stage1'])
            for i in range(depth[cur_index])
        ])
        self.merge_layer1 = Merge_layer(dim=embed_dim[cur_index], dim_out=embed_dim[cur_index+1], norm_layer=norm_layer)

        cur_index += 1
        self.stage2 = nn.ModuleList([
            MixiTBlock(dim=embed_dim[cur_index], reso=to_2tuple(img_size//(4 * 2**cur_index)), num_heads=num_heads[cur_index],
                       mlp_ratio=mlp_ratio[cur_index], qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                       attn_drop=attn_drop_rate, drop_path=dpr[np.sum(depth[:cur_index])+i],
                       act_layer=nn.GELU, norm_layer=norm_layer, initial_value=1e-3, expand_ratio=expand_ratio,
                       sign=0 if (i % 2 == 0) else 1, g_attn_ctg='channel', heads_sp=heads_sp['stage2'])
            for i in range(depth[cur_index])
        ])
        self.merge_layer2 = Merge_layer(dim=embed_dim[cur_index], dim_out=embed_dim[cur_index+1], norm_layer=norm_layer)

        cur_index += 1
        self.stage3 = nn.ModuleList([
            MixiTBlock(dim=embed_dim[cur_index], reso=to_2tuple(img_size//(4 * 2**cur_index)), num_heads=num_heads[cur_index],
                       mlp_ratio=mlp_ratio[cur_index], qkv_bias=qkv_bias, qk_scale=qk_scale,
                       drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[np.sum(depth[:cur_index])+i],
                       act_layer=nn.GELU, norm_layer=norm_layer, initial_value=1e-3, expand_ratio=expand_ratio,
                       sign=0 if (i % 2 == 0) else 1, g_attn_ctg='channel', heads_sp=heads_sp['stage3'])
            for i in range(depth[cur_index])
        ])
        self.merge_layer3 = Merge_layer(dim=embed_dim[cur_index], dim_out=embed_dim[cur_index+1], norm_layer=norm_layer)

        cur_index += 1
        self.stage4 = nn.ModuleList([
            MixiTBlock(dim=embed_dim[cur_index], reso=to_2tuple(img_size//(4 * 2**cur_index)), num_heads=num_heads[cur_index],
                       mlp_ratio=mlp_ratio[cur_index], qkv_bias=qkv_bias, qk_scale=qk_scale,
                       drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[np.sum(depth[:cur_index])+i],
                       act_layer=nn.GELU, norm_layer=norm_layer, initial_value=1e-3, expand_ratio=expand_ratio,
                       sign=1, g_attn_ctg='spatial', heads_sp=heads_sp['stage4'])
            for i in range(depth[cur_index])
        ])

        self.last_norm = norm_layer(embed_dim[-1])
        self.post_norm = norm_layer(embed_dim[0])

        # Classifier head
        if use_mlp_cls_head:
            self.head = nn.Sequential(
                nn.Linear(embed_dim[-1], 2048),
                nn.LayerNorm(2048),
                nn.GELU(),
                nn.Linear(2048, num_classes)
            ) if num_classes > 0 else nn.Identity()
        else:
            self.head = nn.Linear(embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def get_classifier(self):
        return self.head

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'CPE'}

    def forward_features(self, x):  # B, C, H, W
        H, W = x.shape[2:]
        x = self.convolutional_stem(x)
        x = rearrange(x, "B C H W -> B (H W) C")
        x = self.post_norm(x)
        H, W = H//4, W//4

        # stage1 with no merge layer
        for blk in self.stage1:
            if self.use_chk:
                x = checkpoint(blk, x)
            else:
                x = blk(x)

        # combining layers of merge and mixit blocks
        for merge_layer, blks in zip([self.merge_layer1, self.merge_layer2, self.merge_layer3],   # merge
                                     [self.stage2,       self.stage3,       self.stage4]):        # blocks
            x, H, W = merge_layer(x, (H, W))  # downsample
            for blk in blks:
                if self.use_chk:
                    x = checkpoint(blk, x)
                else:
                    x = blk(x)
        x = self.last_norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)  # B, L, C
        x = torch.mean(x, dim=1, keepdim=True).flatten(1)   # B,1,C -> B, C
        x = self.head(x)
        return x


@register_model
def dca_xsmall(pretrained=True, **kwargs):
    model = MDCANet(
        depth=[2, 2, 6, 3],
        embed_dim=[64, 128, 256, 512], num_heads=[4, 4, 8, 16],
        heads_sp = dict(stage1=[1, 1, 2],stage2=[1, 1, 2],stage3=[2, 3, 3],stage4=[4, 6, 6]), mlp_ratio=[4,4,4,4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), expand_ratio=4, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def dca_base(pretrained=True, **kwargs):
    model = MDCANet(
        depth=[2, 3, 12, 4],
        embed_dim=[64, 128, 320, 512], num_heads=[4, 4, 8, 16],
        heads_sp = dict(stage1=[1, 1, 2],stage2=[1, 1, 2],stage3=[3, 3, 2],stage4=[6, 6, 4]), mlp_ratio=[4,4,4,4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), expand_ratio=4, **kwargs)
    model.default_cfg = _cfg()
    return model


if __name__ == "__main__":
    model = dca_base()
    model.eval()
    # print(model)
    inputs = torch.randn(1, 3, 224, 224)
    model(inputs)

    from fvcore.nn import FlopCountAnalysis, ActivationCountAnalysis

    flops = FlopCountAnalysis(model, inputs)
    param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    acts = ActivationCountAnalysis(model, inputs)

    print(f"total flops : {(flops.total()/1e9).__round__(2)}G")
    print(f"total activations: {acts.total()}")
    print(f"number of parameter: {param}")
