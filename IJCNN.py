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
import numpy as np


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
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
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        assert isinstance(resolution, tuple)
        self.H, self.W = resolution
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Shared convolutional relative position encoding.
        self.crpe = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.softmax = nn.Softmax(dim=2)
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
        # Generate Q, K, V.
        qkv = (self.qkv(x).reshape(B, N, 3, self.num_heads,
                                   C // self.num_heads).permute(2, 0, 3, 1, 4)).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # B h N C_h

        # Factorized attention:Q @ (softmax(k).T @ V) / sqrt(C)
        k_softmax = self.softmax(k)
        dd_trans = k_softmax.transpose(-2, -1) @ v   # B, h, C_h, C_h
        factor_attn = q @ dd_trans

        # Convolutional relative position encoding.
        crpe = self.get_crpe(input=[q, v], func=self.crpe)

        # Merge and reshape.
        x = self.scale * factor_attn + crpe
        x = x.transpose(1, 2).reshape(B, N, C).contiguous()

        # Output projection.
        # x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Local_Conv_Attention(nn.Module):
    def __init__(self, dim, resolution, expand_factor=4,):
        super().__init__()

        # 7x7 dwconv
        self.conv7 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim),
            nn.BatchNorm2d(dim),
        )

        # SE layer
        self.se_layer = nn.Sequential(
            nn.Conv2d(dim, dim//2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim//2, dim, kernel_size=1),
        )

        # Inverted Bottleneck
        self.InvBotNeck = nn.Sequential(
            nn.Conv2d(dim, dim*expand_factor, kernel_size=1),
            nn.BatchNorm2d(dim*expand_factor),
            nn.GELU(),

            nn.Conv2d(dim*expand_factor, dim*expand_factor, kernel_size=3, padding=1, groups=dim*expand_factor),
            nn.BatchNorm2d(dim*expand_factor),
            nn.GELU(),

            nn.Conv2d(dim*expand_factor, dim, kernel_size=1),
        )
        self.H, self.W = resolution
        self.sigmoid = torch.sigmoid
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

    def forward(self, x):  # B, L, C
        B, N, C = x.shape
        assert self.H * self.W == N, "wrong token size"
        x = rearrange(x, "B (H W) C -> B C H W", H=self.H, W=self.W) # B,C,H,W
        shotcut = x
        x = self.conv7(x)
        se_branch = self.sigmoid(self.se_layer(
            F.adaptive_avg_pool2d(x, 1)  # B, C, 1, 1
        ))
        x = x * se_branch
        x = self.InvBotNeck(x)
        x = shotcut + x
        return x


# local + global
class MixAttention(nn.Module):
    def __init__(self, dim, resolution, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
                 proj_drop=0., expand_factor=4, equilibrium_factor_init_value = 5e-1, sparse_sample=0):
        super().__init__()
        assert isinstance(resolution, tuple), "reso must be tuple"

        # self.tau_local = nn.Parameter(equilibrium_factor_init_value * torch.ones((1)), requires_grad=True)
        # self.tau_global = nn.Parameter(equilibrium_factor_init_value * torch.ones((1)), requires_grad=True)
        self.partition_se = nn.Sequential(
            nn.Linear(2, 8),    # 2: mean of local + global
            nn.GELU(),
            nn.Linear(8, 2)
        )

        self.sparse_sample = sparse_sample
        # projection layer for heads of local part, so dose for global part
        self.proj_local = nn.Linear(dim // 2, dim // 2)
        # final norm layer for local part, so dose global part
        self.final_norm_local = nn.BatchNorm2d(dim//2)

        self.proj_global = nn.Linear(dim // 2, dim // 2)
        self.final_norm_global = nn.LayerNorm(dim//2)

        # global part -> factorized attn
        self.mix_global = FactorAtt_ConvRelPosEnc(dim=dim//2, num_heads=num_heads, qkv_bias=qkv_bias,
                                                 qk_scale=qk_scale, proj_drop=proj_drop, resolution=resolution)
        # local part -> dwconv-based convolutional modules
        self.mix_local = Local_Conv_Attention(dim=dim//2, resolution=resolution, expand_factor=expand_factor)

        # # final projection layer for features mixing
        self.final_proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sigmoid = torch.sigmoid
        self.apply(self._init_weights)

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

    def forward(self, x):  # B,L,C
        C = x.shape[2]

        # generate x_local and x_global by heads partitioning
        if self.sparse_sample == 0: # dense sample
            x_local = self.proj_local(x[:, :, :C//2])
            x_global = self.proj_global(x[:, :, C//2:])
        else:   # sparse sample
            x_local = self.proj_local(x[:, :, ::2])
            x_global = self.proj_global(x[:, :, 1::2])

        x_local = self.mix_local(x_local)   # B, C//2, H, W
        x_local = self.final_norm_local(x_local)
        x_local_mean = torch.mean(x_local, dim=(1, 2, 3), keepdim=True)  # B, 1, 1, 1

        x_global = self.mix_global(x_global)    # B, L ,C//2
        x_global = self.final_norm_global(x_global)
        x_global_mean = torch.mean(x_global, dim=(1, 2), keepdim=True)  # B, 1, 1

        block_wise_se_branch = self.partition_se(torch.cat([x_local_mean, x_global_mean.unsqueeze(-1)], dim=-1)) # B, 1, 1, 2
        block_wise_se_branch = self.sigmoid(block_wise_se_branch) # B, 1, 1, 2
        x_local = block_wise_se_branch[..., 0].unsqueeze(-1) * x_local
        x_global = block_wise_se_branch[..., 1] * x_global

        x = torch.cat([rearrange(x_local, "B C_half H W -> B (H W) C_half"), x_global], dim=2)  # B, L, C//2 -> B, L ,C
        x = self.final_proj(x)
        x = self.proj_drop(x)
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
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, initial_value=5e-1, sparse_sample=0):

        super().__init__()
        self.dim = dim
        self.H, self.W = reso  # H,W of feature map
        self.norm1 = norm_layer(dim)

        self.CPE = ConvPosEnc(dim=dim, k=3)
        self.attn = MixAttention(
            dim=dim, resolution=(self.H, self.W), num_heads=num_heads,qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, expand_factor=expand_ratio, equilibrium_factor_init_value=initial_value,
            sparse_sample=sparse_sample
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        hidden_dim = mlp_ratio * dim
        self.mlp = Mlp(in_features=dim, hidden_features=hidden_dim, out_features=dim, act_layer=act_layer,
                       drop=drop)
        self.norm2 = norm_layer(dim)
        self.apply(self._init_weights)

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
        """
        x: B, H*W, C
        """
        B, N, C = x.shape
        assert N == self.H * self.W, "flatten img_tokens has wrong size"
        x = rearrange(x, "B (H W) C -> B C H W", H=self.H, W=self.W)  # B, C, H, W

        # conditional position encoding (CPvT)
        x = self.CPE(x)
        x = rearrange(x, "B C H W -> B (H W) C")
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

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


class MixiT(nn.Module):
    """ MixiT: Mixed Vision Transformer for Efficient Local-global Representations Learning
    """

    def __init__(self, depth=[3, 4, 8, 3], img_size=224, in_chans=3, num_classes=1000, embed_dim=[64, 128, 320, 512],
                 num_heads=[2, 4, 8, 16], mlp_ratio=[8, 8, 8, 8], qkv_bias=True, qk_scale=None, expand_ratio = 4,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm):
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

        # stem cell
        self.convolutional_stem = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim[0], kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(embed_dim[0]),
            nn.GELU(),

            nn.Conv2d(embed_dim[0], embed_dim[0] // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim[0] // 2),
            nn.GELU(),

            nn.Conv2d(embed_dim[0] // 2, embed_dim[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim[0]),
            nn.GELU(),
        )
        self.convolutional_stem.add_module('downsample layer', nn.Conv2d(embed_dim[0], embed_dim[0], kernel_size=2, stride=2))
        self.stem_norm = nn.LayerNorm(embed_dim[0])

        cur_index = 0   # stage index
        self.stage1 = nn.ModuleList([
            MixiTBlock(dim=embed_dim[cur_index], reso=to_2tuple(img_size//4 * 2**cur_index), num_heads=num_heads[cur_index],
                       mlp_ratio=mlp_ratio[cur_index], qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                       drop_path=dpr[i], act_layer=nn.GELU, norm_layer=norm_layer, initial_value=1e-3, expand_ratio=expand_ratio,
                       sparse_sample=0 if (i % 2 == 0) else 1)
            for i in range(depth[cur_index])
        ])
        self.merge_layer1 = Merge_layer(dim=embed_dim[cur_index], dim_out=embed_dim[cur_index+1], norm_layer=norm_layer)

        cur_index += 1
        self.stage2 = nn.ModuleList([
            MixiTBlock(dim=embed_dim[cur_index], reso=to_2tuple(img_size//(4 * 2**cur_index)), num_heads=num_heads[cur_index],
                       mlp_ratio=mlp_ratio[cur_index], qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                       attn_drop=attn_drop_rate, drop_path=dpr[np.sum(depth[:cur_index])+i],
                       act_layer=nn.GELU, norm_layer=norm_layer, initial_value=1e-3, expand_ratio=expand_ratio,
                       sparse_sample=0 if (i % 2 == 0) else 1)
            for i in range(depth[cur_index])
        ])
        self.merge_layer2 = Merge_layer(dim=embed_dim[cur_index], dim_out=embed_dim[cur_index+1], norm_layer=norm_layer)

        cur_index += 1
        self.stage3 = nn.ModuleList([
            MixiTBlock(dim=embed_dim[cur_index], reso=to_2tuple(img_size//(4 * 2**cur_index)), num_heads=num_heads[cur_index],
                       mlp_ratio=mlp_ratio[cur_index], qkv_bias=qkv_bias, qk_scale=qk_scale,
                       drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[np.sum(depth[:cur_index])+i],
                       act_layer=nn.GELU, norm_layer=norm_layer, initial_value=1e-3, expand_ratio=expand_ratio,
                       sparse_sample=0 if (i % 2 == 0) else 1)
            for i in range(depth[cur_index])
        ])
        self.merge_layer3 = Merge_layer(dim=embed_dim[cur_index], dim_out=embed_dim[cur_index+1], norm_layer=norm_layer)

        cur_index += 1
        self.stage4 = nn.ModuleList([
            MixiTBlock(dim=embed_dim[cur_index], reso=to_2tuple(img_size//(4 * 2**cur_index)), num_heads=num_heads[cur_index],
                       mlp_ratio=mlp_ratio[cur_index], qkv_bias=qkv_bias, qk_scale=qk_scale,
                       drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[np.sum(depth[:cur_index])+i],
                       act_layer=nn.GELU, norm_layer=norm_layer, initial_value=1e-3, expand_ratio=expand_ratio,
                       sparse_sample=0 if (i % 2 == 0) else 1)
            for i in range(depth[cur_index])
        ])

        self.last_norm = nn.LayerNorm(embed_dim[-1])

        # Classifier head
        self.head = nn.Sequential(
            nn.Linear(embed_dim[-1], 1280),
            norm_layer(1280),
            nn.GELU(),
            nn.Linear(1280, num_classes)
        ) if num_classes > 0 else nn.Identity()

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

    def forward_features(self, x):  # B, C, H, W
        H, W = x.shape[2:]
        x = self.convolutional_stem(x)
        x = rearrange(x, "B C H W -> B (H W) C")
        x = self.stem_norm(x)   # B, L, C
        H, W = H//4, W//4

        # stage1 with no merge layer
        for blk in self.stage1:
            x = blk(x)

        # combining layers of merge and mixit blocks
        for merge_layer, blks in zip([self.merge_layer1, self.merge_layer2, self.merge_layer3],   # merge
                                     [self.stage2,       self.stage3,       self.stage4]):        # blocks
            x, H, W = merge_layer(x, (H, W))  # downsample
            for blk in blks:
                x = blk(x)

        x = self.last_norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)  # B, L, C
        x = torch.mean(x, dim=1, keepdim=True).flatten(1)   # B,1,C -> B, C
        x = self.head(x)
        return x


@register_model
def mixit_xsmall(pretrained=True, **kwargs):
    model = MixiT(
        depth=[1, 2, 6, 2],
        embed_dim=[64, 128, 256, 512], num_heads=[1, 2, 4, 8], mlp_ratio=[8,8,4,4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), expand_ratio=4, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def mixit_small(pretrained=True, **kwargs):
    model = MixiT(
        depth=[2, 2, 8, 4],
        embed_dim=[64, 128, 320, 512], num_heads=[1, 2, 4, 8], mlp_ratio=[8,8,4,4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), expand_ratio=4, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def mixit_base(pretrained=True, **kwargs):
    model = MixiT(
        depth=[2, 2, 14, 4],
        embed_dim=[96, 192, 384, 768], num_heads=[1, 2, 4, 8], mlp_ratio=[8,8,4,4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), expand_ratio=4, **kwargs)
    model.default_cfg = _cfg()
    return model

if __name__ == "__main__":
    model = mixit_small()
    model.eval()
    inputs = torch.randn(1, 3, 224, 224)
    out = model(inputs)
    print(out.shape)
    from fvcore.nn import FlopCountAnalysis, ActivationCountAnalysis

    flops = FlopCountAnalysis(model, inputs)
    param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    acts = ActivationCountAnalysis(model, inputs)

    print(f"total flops : {(flops.total()/1e9).__round__(2)}G")
    print(f"total activations: {acts.total()}")
    print(f"number of parameter: {param}")
