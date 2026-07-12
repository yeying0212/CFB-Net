import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """Truncated normal initialization (same as timm)."""
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0:
            random_tensor.div_(keep_prob)
        return x * random_tensor


# ==================== Attention Modules ====================

class PA(nn.Module):
    """Point Attention: 1x1 conv channel attention with sigmoid gate."""
    def __init__(self, dim):
        super().__init__()
        self.p_conv = nn.Sequential(
            nn.Conv2d(dim, dim * 4, 1, bias=False),
            nn.BatchNorm2d(dim * 4),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim, 1, bias=False))
        self.gate_fn = nn.Sigmoid()

    def forward(self, x):
        return x * self.gate_fn(self.p_conv(x))


class LA(nn.Module):
    """Local Attention: 3x3 conv."""
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU())

    def forward(self, x):
        return self.conv(x)


class MRA(nn.Module):
    """Medium-Range Attention: strip convolutions (horizontal + vertical + diagonal)."""
    def __init__(self, channel, att_kernel=11):
        super().__init__()
        self.channel = channel
        att_padding = att_kernel // 2
        self.gate_fn = nn.Sigmoid()
        self.max_m1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.max_m2 = nn.MaxPool2d(kernel_size=3, stride=3, padding=0)  # simplified BlurPool
        self.H_att1 = nn.Conv2d(channel, channel, (att_kernel, 3), 1,
                                 (att_padding, 1), groups=channel, bias=False)
        self.V_att1 = nn.Conv2d(channel, channel, (3, att_kernel), 1,
                                 (1, att_padding), groups=channel, bias=False)
        self.H_att2 = nn.Conv2d(channel, channel, (att_kernel, 3), 1,
                                 (att_padding, 1), groups=channel, bias=False)
        self.V_att2 = nn.Conv2d(channel, channel, (3, att_kernel), 1,
                                 (1, att_padding), groups=channel, bias=False)
        self.norm = nn.BatchNorm2d(channel)

    def forward(self, x):
        x_tem = self.max_m1(x)
        x_tem = self.max_m2(x_tem)
        x_h1 = self.H_att1(x_tem)
        x_w1 = self.V_att1(x_tem)
        x_h2 = self.inv_h_transform(self.H_att2(self.h_transform(x_tem)))
        x_w2 = self.inv_v_transform(self.V_att2(self.v_transform(x_tem)))
        att = self.norm(x_h1 + x_w1 + x_h2 + x_w2)
        out = x[:, :self.channel, :, :] * F.interpolate(
            self.gate_fn(att), size=(x.shape[-2], x.shape[-1]), mode='nearest')
        return out

    def h_transform(self, x):
        shape = x.size()
        x = F.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2 * shape[3] - 1)
        return x

    def inv_h_transform(self, x):
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1)
        x = F.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2 * shape[-2])
        x = x[..., 0: shape[-2]]
        return x

    def v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = F.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2 * shape[3] - 1)
        return x.permute(0, 1, 3, 2)

    def inv_v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1)
        x = F.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2 * shape[-2])
        x = x[..., 0: shape[-2]]
        return x.permute(0, 1, 3, 2)


class GA12(nn.Module):
    """Global Attention for stages 0 and 1: pool → attention → unpool."""
    def __init__(self, dim):
        super().__init__()
        self.downpool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.uppool = nn.MaxUnpool2d((2, 2), 2)
        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.act = nn.GELU()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9,
                                       groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)
        self.proj_2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        x_, idx = self.downpool(x)
        x_ = self.act(self.proj_1(x_))
        attn1 = self.conv0(x_)
        attn2 = self.conv_spatial(attn1)
        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)
        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0:1, :, :] + attn2 * sig[:, 1:2, :, :]
        attn = self.conv(attn)
        x_ = x_ * attn
        x_ = self.proj_2(x_)
        x = self.uppool(x_, indices=idx)
        return x


class D_GA(nn.Module):
    """Global Attention for stage 2: pool → self-attn → norm → unpool."""
    def __init__(self, dim):
        super().__init__()
        self.attn = GA(dim)
        self.downpool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.uppool = nn.MaxUnpool2d((2, 2), 2)
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x):
        x_, idx = self.downpool(x)
        x_ = self.norm(self.attn(x_))
        x = self.uppool(x_, indices=idx)
        return x


class GA(nn.Module):
    """Global Attention for stage 3: standard multi-head self-attention."""
    def __init__(self, dim, head_dim=4, num_heads=None):
        super().__init__()
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.num_heads = num_heads if num_heads else dim // head_dim
        if self.num_heads == 0:
            self.num_heads = 1
        self.attention_dim = self.num_heads * self.head_dim
        self.qkv = nn.Linear(dim, self.attention_dim * 3, bias=False)
        self.proj = nn.Linear(self.attention_dim, dim, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # B, H, W, C
        N = H * W
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.attention_dim)
        x = self.proj(x)
        x = x.permute(0, 3, 1, 2)
        return x


# ==================== Downsampling (DRFD) ====================

class DRFD(nn.Module):
    """Dual-path Residual Feature Downsampling: depthwise conv + maxpool fusion."""
    def __init__(self, dim):
        super().__init__()
        self.outdim = dim * 2
        self.conv = nn.Conv2d(dim, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim)
        self.conv_c = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=2, padding=1, groups=dim * 2)
        self.act_c = nn.GELU()
        self.norm_c = nn.BatchNorm2d(dim * 2)
        self.max_m = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.norm_m = nn.BatchNorm2d(dim * 2)
        self.fusion = nn.Conv2d(dim * 4, dim * 2, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv(x)
        conv = self.norm_c(self.act_c(self.conv_c(x)))
        maxp = self.norm_m(self.max_m(x))
        x = torch.cat([conv, maxp], dim=1)
        x = self.fusion(x)
        return x


# ==================== LWGA Block ====================

class LWGA_Block(nn.Module):
    """LightWeight Global-Aware Block: 4-path attention (PA + LA + MRA + GA)."""
    def __init__(self, dim, stage, att_kernel=11, mlp_ratio=2., drop_path=0.):
        super().__init__()
        self.stage = stage
        self.dim_split = dim // 4
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
            nn.BatchNorm2d(mlp_hidden_dim),
            nn.GELU(),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False))

        self.PA = PA(self.dim_split)
        self.LA = LA(self.dim_split)
        self.MRA = MRA(self.dim_split, att_kernel)

        if stage == 2:
            self.GA = D_GA(self.dim_split)
        elif stage == 3:
            self.GA = GA(self.dim_split)
            self.norm_ga = nn.BatchNorm2d(self.dim_split)
        else:
            self.GA = GA12(self.dim_split)
            self.norm_ga = nn.BatchNorm2d(self.dim_split)

        self.norm1 = nn.BatchNorm2d(dim)

    def forward(self, x):
        shortcut = x.clone()
        x1, x2, x3, x4 = torch.split(x, [self.dim_split] * 4, dim=1)

        x1 = x1 + self.PA(x1)
        x2 = self.LA(x2)
        x3 = self.MRA(x3)
        if self.stage in [0, 1]:
            x4 = self.norm_ga(x4 + self.GA(x4))
        elif self.stage == 2:
            x4 = x4 + self.GA(x4)
        else:  # stage 3
            x4 = self.norm_ga(x4 + self.GA(x4))

        x_att = torch.cat((x1, x2, x3, x4), 1)
        x = shortcut + self.norm1(self.drop_path(self.mlp(x_att)))
        return x


# ==================== LWGANet Backbone ====================

class BasicStage(nn.Module):
    def __init__(self, dim, stage, depth, att_kernel, mlp_ratio, drop_paths):
        super().__init__()
        blocks = [LWGA_Block(dim=dim, stage=stage, att_kernel=att_kernel,
                             mlp_ratio=mlp_ratio, drop_path=drop_paths[i])
                  for i in range(depth)]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class Stem(nn.Module):
    def __init__(self, in_chans, stem_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, stem_dim, kernel_size=4, stride=4, bias=False),
            nn.BatchNorm2d(stem_dim))

    def forward(self, x):
        return self.proj(x)


class LWGANet(nn.Module):
    """LWGANet backbone with multi-range attention.

    LightWeight Global-Aware Network (AAAI 2026)
    https://github.com/AeroVILab-AHU/LWGANet

    Variants:
    - L0: stem_dim=32,  depths=(1,2,4,2), att_kernel=11, act=GELU  (output: 32,64,128,256)
    - L1: stem_dim=64,  depths=(1,2,4,2), att_kernel=11, act=GELU  (output: 64,128,256,512)
    - L2: stem_dim=96,  depths=(1,4,4,2), att_kernel=11, act=ReLU  (output: 96,192,384,768)

    Returns 4 multi-scale feature maps for dense prediction.
    """
    def __init__(self, in_chans=3, stem_dim=32, depths=(1, 2, 4, 2),
                 att_kernel=(11, 11, 11, 11), mlp_ratio=2., drop_path_rate=0.1):
        super().__init__()
        self.num_stages = len(depths)
        self.num_features = int(stem_dim * 2 ** (self.num_stages - 1))
        self.stem_dim = stem_dim

        self.Stem = Stem(in_chans, stem_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        stages_list = []
        for i_stage in range(self.num_stages):
            stage_dim = int(stem_dim * 2 ** i_stage)
            stage = BasicStage(
                dim=stage_dim, stage=i_stage, depth=depths[i_stage],
                att_kernel=att_kernel[i_stage], mlp_ratio=mlp_ratio,
                drop_paths=dpr[sum(depths[:i_stage]):sum(depths[:i_stage + 1])])
            stages_list.append(stage)
            if i_stage < self.num_stages - 1:
                stages_list.append(DRFD(stage_dim))

        self.stages = nn.Sequential(*stages_list)

        # Norm layers for each output stage
        self.out_indices = [0, 2, 4, 6]
        for i_emb, i_layer in enumerate(self.out_indices):
            out_dim = int(stem_dim * 2 ** i_emb)
            self.add_module(f'norm{i_layer}', nn.BatchNorm2d(out_dim))

        # Channel list for downstream use
        self.channels = [int(stem_dim * 2 ** i) for i in range(self.num_stages)]

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.Stem(x)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                outs.append(norm_layer(x))
        return outs


# ==================== Builder Functions ====================

def lwganet_l0(pretrained=False):
    """LWGANet-L0: stem_dim=32, depths=(1,2,4,2)."""
    model = LWGANet(in_chans=3, stem_dim=32, depths=(1, 2, 4, 2),
                    att_kernel=(11, 11, 11, 11), mlp_ratio=2., drop_path_rate=0.0)
    return model


def lwganet_l1(pretrained=False):
    """LWGANet-L1: stem_dim=64, depths=(1,2,4,2)."""
    model = LWGANet(in_chans=3, stem_dim=64, depths=(1, 2, 4, 2),
                    att_kernel=(11, 11, 11, 11), mlp_ratio=2., drop_path_rate=0.0)
    return model


def lwganet_l2(pretrained=False):
    """LWGANet-L2: stem_dim=96, depths=(1,4,4,2)."""
    model = LWGANet(in_chans=3, stem_dim=96, depths=(1, 4, 4, 2),
                    att_kernel=(11, 11, 11, 11), mlp_ratio=2., drop_path_rate=0.1)
    return model
