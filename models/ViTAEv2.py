"""
ViTAEv2-S backbone for CFB-Net framework.

ViTAEv2: Vision Transformer Advanced by Exploring Inductive Bias
for Image Recognition and Beyond (IJCV 2022 / NeurIPS 2021)

Self-contained implementation — no timm dependency.
Used by LiST-Net as the Siamese encoder backbone.

Config (ViTAEv2-S):
  embed_dims=[64, 64, 128, 256]
  token_dims=[64, 128, 256, 512]
  downsample_ratios=[4, 2, 2, 2]
  NC_depth=[2, 2, 8, 2]
  window_size=7

Returns 4 multi-scale feature maps for dense prediction tasks.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial


# ==================== Helpers (replace timm) ====================

def _trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
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


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _trunc_normal_(tensor, mean, std, a, b)


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


# ==================== Window operations ====================

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# ==================== MLP ====================

class ViTMlp(nn.Module):
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


# ==================== Attention modules ====================

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class AttentionPerformer(nn.Module):
    """Linear-complexity attention via positive random features (Performer)."""
    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., kernel_ratio=0.5):
        super().__init__()
        self.head_dim = dim // num_heads
        self.emb = dim
        self.kqv = nn.Linear(dim, 3 * self.emb)
        self.dp = nn.Dropout(proj_drop)
        self.proj = nn.Linear(self.emb, self.emb)
        self.head_cnt = num_heads
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.epsilon = 1e-8
        self.drop_path = nn.Identity()
        self.m = int(self.head_dim * kernel_ratio)
        self.w = torch.randn(self.head_cnt, self.m, self.head_dim)
        for i in range(self.head_cnt):
            self.w[i] = nn.Parameter(nn.init.orthogonal_(self.w[i]) * math.sqrt(self.m), requires_grad=False)
        self.w.requires_grad_(False)

    def prm_exp(self, x):
        xd = ((x * x).sum(dim=-1, keepdim=True)).repeat(1, 1, 1, self.m) / 2
        wtx = torch.einsum('bhti,hmi->bhtm', x.float(), self.w.to(x.device))
        return torch.exp(wtx - xd) / math.sqrt(self.m)

    def attn(self, x):
        B, N, C = x.shape
        kqv = self.kqv(x).reshape(B, N, 3, self.head_cnt, self.head_dim).permute(2, 0, 3, 1, 4)
        k, q, v = kqv[0], kqv[1], kqv[2]
        kp, qp = self.prm_exp(k), self.prm_exp(q)
        D = torch.einsum('bhti,bhi->bht', qp, kp.sum(dim=2)).unsqueeze(dim=-1)
        kptv = torch.einsum('bhin,bhim->bhnm', v.float(), kp)
        y = torch.einsum('bhti,bhni->bhtn', qp, kptv) / (D.repeat(1, 1, 1, self.head_dim) + self.epsilon)
        y = y.permute(0, 2, 1, 3).reshape(B, N, self.emb)
        v = v.permute(0, 2, 1, 3).reshape(B, N, self.emb)
        return v + self.dp(self.proj(y))

    def forward(self, x):
        return self.attn(x)


class WindowAttention(nn.Module):
    """Window-based multi-head self-attention (from Swin Transformer)."""
    def __init__(self, in_dim, out_dim, window_size, num_heads, qkv_bias=True,
                 qk_scale=None, attn_drop=0., proj_drop=0., relative_pos=False):
        super().__init__()
        self.in_dim = in_dim
        self.dim = out_dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = out_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_pos = relative_pos

        if self.relative_pos:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
            trunc_normal_(self.relative_position_bias_table, std=.02)

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(in_dim, out_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(out_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        if self.relative_pos:
            rpb = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1], -1)
            rpb = rpb.permute(2, 0, 1).contiguous()
            attn = attn + rpb.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ==================== Token Transformer (ReductionCell helper) ====================

class TokenAttention(nn.Module):
    def __init__(self, dim, num_heads=8, in_dim=None, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.in_dim = in_dim
        head_dim = in_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, in_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(in_dim, in_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.in_dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.in_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        v = v.permute(0, 2, 1, 3).reshape(B, N, self.in_dim).contiguous()
        return v + x


class TokenTransformer(nn.Module):
    def __init__(self, dim, in_dim, num_heads, mlp_ratio=1., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = TokenAttention(dim, in_dim=in_dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                   qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(in_dim)
        self.mlp = ViTMlp(in_features=in_dim, hidden_features=int(in_dim * mlp_ratio),
                          out_features=in_dim, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.mlp(self.norm2(self.attn(self.norm1(x)))))
        return x


class TokenPerformer(nn.Module):
    def __init__(self, dim, in_dim, head_cnt=1, kernel_ratio=0.5, dp1=0.1, dp2=0.1):
        super().__init__()
        self.head_dim = in_dim // head_cnt
        self.emb = in_dim
        self.kqv = nn.Linear(dim, 3 * self.emb)
        self.dp = nn.Dropout(dp1)
        self.proj = nn.Linear(self.emb, self.emb)
        self.head_cnt = head_cnt
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(self.emb)
        self.epsilon = 1e-8
        self.drop_path = nn.Identity()
        self.mlp = nn.Sequential(
            nn.Linear(self.emb, self.emb), nn.GELU(),
            nn.Linear(self.emb, self.emb), nn.Dropout(dp2))
        self.m = int(self.head_dim * kernel_ratio)
        self.w = torch.randn(head_cnt, self.m, self.head_dim)
        for i in range(head_cnt):
            self.w[i] = nn.Parameter(nn.init.orthogonal_(self.w[i]) * math.sqrt(self.m), requires_grad=False)
        self.w.requires_grad_(False)

    def prm_exp(self, x):
        xd = ((x * x).sum(dim=-1, keepdim=True)).repeat(1, 1, 1, self.m) / 2
        wtx = torch.einsum('bhti,hmi->bhtm', x.float(), self.w.to(x.device))
        return torch.exp(wtx - xd) / math.sqrt(self.m)

    def attn(self, x):
        B, N, C = x.shape
        kqv = self.kqv(x).reshape(B, N, 3, self.head_cnt, self.head_dim).permute(2, 0, 3, 1, 4)
        k, q, v = kqv[0], kqv[1], kqv[2]
        kp, qp = self.prm_exp(k), self.prm_exp(q)
        D = torch.einsum('bhti,bhi->bht', qp, kp.sum(dim=2)).unsqueeze(dim=-1)
        kptv = torch.einsum('bhin,bhim->bhnm', v.float(), kp)
        y = torch.einsum('bhti,bhni->bhtn', qp, kptv) / (D.repeat(1, 1, 1, self.head_dim) + self.epsilon)
        y = y.permute(0, 2, 1, 3).reshape(B, N, self.emb)
        v = v.permute(0, 2, 1, 3).reshape(B, N, self.emb)
        return v + self.dp(self.proj(y))

    def forward(self, x):
        x = self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ==================== NormalCell ====================

class NormalCell(nn.Module):
    """ViTAEv2 Normal Cell: attention + PCM conv + MLP."""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., group=64, tokens_type='transformer',
                 shift_size=0, window_size=0, img_size=224, relative_pos=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.img_size = img_size
        self.window_size = window_size
        self.tokens_type = tokens_type
        self.shift_size = shift_size

        if tokens_type == 'transformer':
            self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                   qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        elif tokens_type == 'performer':
            self.attn = AttentionPerformer(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        elif tokens_type == 'window':
            self.attn = WindowAttention(in_dim=dim, out_dim=dim, window_size=(self.window_size, self.window_size),
                                         num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         attn_drop=attn_drop, proj_drop=drop, relative_pos=relative_pos)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ViTMlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        self.PCM = nn.Sequential(
            nn.Conv2d(dim, mlp_hidden_dim, 3, 1, 1, 1, group),
            nn.BatchNorm2d(mlp_hidden_dim), nn.SiLU(inplace=True),
            nn.Conv2d(mlp_hidden_dim, dim, 3, 1, 1, 1, group),
            nn.BatchNorm2d(dim), nn.SiLU(inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1, 1, group))

    def _get_attn_mask(self, H, W, device):
        """Compute window attention mask for given spatial dimensions."""
        if self.shift_size > 0:
            img_mask = torch.zeros((1, H, W, 1), device=device)
            h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            return attn_mask
        return None

    def forward(self, x):
        b, n, c = x.shape
        shortcut = x

        if self.tokens_type == 'window':
            H = W = int(math.sqrt(n))
            assert n == H * W, "input feature has wrong size or is not square"
            x = self.norm1(x).view(b, H, W, c)
            padding_td = (self.window_size - H % self.window_size) % self.window_size
            padding_top = padding_td // 2
            padding_down = padding_td - padding_top
            padding_lr = (self.window_size - W % self.window_size) % self.window_size
            padding_left = padding_lr // 2
            padding_right = padding_lr - padding_left
            if padding_td + padding_lr > 0:
                x = x.permute(0, 3, 1, 2)
                x = F.pad(x, (padding_left, padding_right, padding_top, padding_down))
                x = x.permute(0, 2, 3, 1).contiguous()
            # Compute mask for padded spatial size
            padH, padW = H + padding_td, W + padding_lr
            attn_mask = self._get_attn_mask(padH, padW, x.device)
            if self.shift_size > 0:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            else:
                shifted_x = x
            x_windows = window_partition(shifted_x, self.window_size)
            x_windows = x_windows.view(-1, self.window_size * self.window_size, c)
            attn_windows = self.attn(x_windows, mask=attn_mask)
            attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
            shifted_x = window_reverse(attn_windows, self.window_size, padH, padW)
            if self.shift_size > 0:
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x = shifted_x
            x = x[:, padding_top:padding_top + H, padding_left:padding_left + W, :]
            x = x.reshape(b, H * W, c)
        else:
            x = self.attn(self.norm1(x))

        wh = int(math.sqrt(n))
        convX = self.drop_path(self.PCM(shortcut.view(b, wh, wh, c).permute(0, 3, 1, 2).contiguous())
                               .permute(0, 2, 3, 1).contiguous().view(b, n, c))
        x = shortcut + self.drop_path(x) + convX
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ==================== ReductionCell (PRM + RC) ====================

class PRM(nn.Module):
    """Pyramid Reduction Module: multi-scale dilated conv for downsampling."""
    def __init__(self, img_size=224, kernel_size=4, downsample_ratio=4, dilations=[1, 6, 12],
                 in_chans=3, embed_dim=64, share_weights=False, op='cat'):
        super().__init__()
        self.dilations = dilations
        self.embed_dim = embed_dim
        self.downsample_ratio = downsample_ratio
        self.op = op
        self.kernel_size = kernel_size
        self.stride = downsample_ratio
        self.share_weights = share_weights
        self.outSize = img_size // downsample_ratio

        if share_weights:
            self.convolution = nn.Conv2d(in_channels=in_chans, out_channels=embed_dim,
                                          kernel_size=self.kernel_size, stride=self.stride,
                                          padding=3 * dilations[0] // 2, dilation=dilations[0])
        else:
            self.convs = nn.ModuleList()
            for dilation in self.dilations:
                padding = math.ceil(((self.kernel_size - 1) * dilation + 1 - self.stride) / 2)
                self.convs.append(nn.Sequential(
                    nn.Conv2d(in_channels=in_chans, out_channels=embed_dim,
                              kernel_size=self.kernel_size, stride=self.stride,
                              padding=padding, dilation=dilation),
                    nn.GELU()))

        self.out_chans = embed_dim if op == 'sum' else embed_dim * len(self.dilations)

    def forward(self, x):
        B, C, W, H = x.shape
        if self.share_weights:
            padding = math.ceil(((self.kernel_size - 1) * self.dilations[0] + 1 - self.stride) / 2)
            y = F.conv2d(x, weight=self.convolution.weight, bias=self.convolution.bias,
                         stride=self.downsample_ratio, padding=padding,
                         dilation=self.dilations[0]).unsqueeze(dim=-1)
            for i in range(1, len(self.dilations)):
                padding = math.ceil(((self.kernel_size - 1) * self.dilations[i] + 1 - self.stride) / 2)
                _y = F.conv2d(x, weight=self.convolution.weight, bias=self.convolution.bias,
                              stride=self.downsample_ratio, padding=padding,
                              dilation=self.dilations[i]).unsqueeze(dim=-1)
                y = torch.cat((y, _y), dim=-1)
        else:
            y = self.convs[0](x).unsqueeze(dim=-1)
            for i in range(1, len(self.dilations)):
                _y = self.convs[i](x).unsqueeze(dim=-1)
                y = torch.cat((y, _y), dim=-1)
        B, C, W, H, N = y.shape
        if self.op == 'sum':
            y = y.sum(dim=-1).flatten(2).permute(0, 2, 1).contiguous()
        elif self.op == 'cat':
            y = y.permute(0, 4, 1, 2, 3).flatten(3).reshape(B, N * C, W * H).permute(0, 2, 1).contiguous()
        return y, (W, H)


class ReductionCell(nn.Module):
    """ViTAEv2 Reduction Cell: PRM + token attention + PCM."""
    def __init__(self, img_size=224, in_chans=3, embed_dims=64, token_dims=64,
                 downsample_ratios=4, kernel_size=7, num_heads=1, dilations=[1, 2, 3, 4],
                 share_weights=False, op='cat', tokens_type='performer', group=1,
                 relative_pos=False, drop=0., attn_drop=0., drop_path=0.,
                 mlp_ratio=1.0, window_size=7):
        super().__init__()
        self.img_size = img_size
        self.window_size = window_size
        self.op = op
        self.tokens_type = tokens_type
        self.dilations = dilations
        self.num_heads = num_heads
        self.embed_dims = embed_dims
        self.token_dims = token_dims
        self.in_chans = in_chans
        self.downsample_ratios = downsample_ratios
        self.kernel_size = kernel_size
        self.outSize = img_size
        self.relative_pos = relative_pos

        PCMStride = []
        residual = downsample_ratios // 2
        for _ in range(3):
            PCMStride.append((residual > 0) + 1)
            residual = residual // 2
        assert residual == 0
        self.pool = None

        self.PCM = nn.Sequential(
            nn.Conv2d(in_chans, embed_dims, 3, PCMStride[0], 1, groups=group),
            nn.BatchNorm2d(embed_dims), nn.SiLU(inplace=True),
            nn.Conv2d(embed_dims, embed_dims, 3, PCMStride[1], 1, groups=group),
            nn.BatchNorm2d(embed_dims), nn.SiLU(inplace=True),
            nn.Conv2d(embed_dims, token_dims, 3, PCMStride[2], 1, groups=group))

        self.PRM = PRM(img_size=img_size, kernel_size=kernel_size, downsample_ratio=downsample_ratios,
                       dilations=self.dilations, in_chans=in_chans, embed_dim=embed_dims,
                       share_weights=share_weights, op=op)
        self.outSize = self.outSize // downsample_ratios

        in_chans = self.PRM.out_chans
        if tokens_type == 'performer':
            self.attn = TokenPerformer(dim=in_chans, in_dim=token_dims, head_cnt=num_heads, kernel_ratio=0.5)
        elif tokens_type == 'performer_less':
            self.attn = None
            self.PCM = None
        elif tokens_type == 'transformer':
            self.attn = TokenTransformer(dim=in_chans, in_dim=token_dims, num_heads=num_heads,
                                          mlp_ratio=mlp_ratio, drop=drop, attn_drop=attn_drop,
                                          drop_path=drop_path)
        elif tokens_type == 'window':
            self.reduce_proj = nn.Linear(in_chans, token_dims)

    def forward(self, x):
        if len(x.shape) < 4:
            B, N, C = x.shape
            n = int(np.sqrt(N))
            x = x.view(B, n, n, C).contiguous().permute(0, 3, 1, 2)
        if self.pool is not None:
            x = self.pool(x)
        shortcut = x
        PRM_x, _ = self.PRM(x)

        if self.tokens_type in ('performer', 'transformer'):
            if self.attn is None:
                return PRM_x
            convX = self.PCM(shortcut)
            x = self.attn.attn(TokenTransformer.__dict__.get('norm1', nn.Identity())(PRM_x)
                               if hasattr(self.attn, 'norm1') else self.attn.norm1(PRM_x))
            convX = convX.permute(0, 2, 3, 1).view(*x.shape).contiguous()
            x = x + DropPath(0.)(convX)
            x = x + DropPath(0.)(self.attn.mlp(self.attn.norm2(x)))
            return x
        if self.tokens_type == 'window':
            return self.reduce_proj(PRM_x)
        return PRM_x


# ==================== ViTAEv2 Basic Layer ====================

class BasicLayer(nn.Module):
    def __init__(self, img_size=224, in_chans=3, embed_dims=64, token_dims=64,
                 downsample_ratios=4, kernel_size=7, RC_heads=1, NC_heads=6,
                 dilations=[1, 2, 3, 4], RC_op='cat', RC_tokens_type='performer',
                 NC_tokens_type='transformer', RC_group=1, NC_group=64, NC_depth=2,
                 dpr=0.1, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0,
                 attn_drop=0., window_size=7, relative_pos=False):
        super().__init__()
        self.img_size = img_size
        self.in_chans = in_chans
        self.embed_dims = embed_dims
        self.token_dims = token_dims
        self.downsample_ratios = downsample_ratios
        self.out_size = self.img_size // self.downsample_ratios

        if RC_tokens_type == 'stem':
            self.RC = PatchEmbedding(inter_channel=token_dims // 2, out_channels=token_dims, img_size=img_size)
        elif downsample_ratios > 1:
            self.RC = ReductionCell(img_size, in_chans, embed_dims, token_dims, downsample_ratios,
                                     kernel_size, RC_heads, dilations, op=RC_op,
                                     tokens_type=RC_tokens_type, group=RC_group,
                                     relative_pos=relative_pos, window_size=window_size)
        else:
            self.RC = nn.Identity()

        self.NC = nn.ModuleList([
            NormalCell(token_dims, NC_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                       qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                       drop_path=dpr[i] if isinstance(dpr, list) else dpr,
                       group=NC_group, tokens_type=NC_tokens_type,
                       img_size=img_size // downsample_ratios,
                       window_size=window_size, shift_size=0 if i % 2 == 0 else window_size // 2,
                       relative_pos=relative_pos)
            for i in range(NC_depth)])

    def forward(self, x):
        x = self.RC(x)
        for nc in self.NC:
            x = nc(x)
        return x


class PatchEmbedding(nn.Module):
    """Stem: conv-based patch embedding."""
    def __init__(self, inter_channel=32, out_channels=48, img_size=None):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, inter_channel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(inter_channel), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(inter_channel, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv3(self.conv2(self.conv1(x)))
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        return x


# ==================== Main ViTAEv2 Backbone ====================

class ViTAEv2(nn.Module):
    """ViTAEv2-S backbone for dense prediction.

    Returns 4 multi-scale feature maps from each stage.
    Config: ViTAEv2-S (the variant used in LiST-Net).

    Params: ~18.8M (classification), ~5-7M (backbone only, 4 stages)
    """
    def __init__(self, img_size=224, in_chans=3, stages=4,
                 embed_dims=(64, 64, 128, 256), token_dims=(64, 128, 256, 512),
                 downsample_ratios=(4, 2, 2, 2), kernel_size=(7, 3, 3, 3),
                 RC_heads=(1, 1, 2, 4), NC_heads=(1, 2, 4, 8),
                 dilations=([1, 2, 3, 4], [1, 2, 3], [1, 2], [1, 2]),
                 RC_tokens_type=('window', 'window', 'transformer', 'transformer'),
                 NC_tokens_type=('window', 'window', 'transformer', 'transformer'),
                 NC_depth=(2, 2, 8, 2), mlp_ratio=4., drop_path_rate=0.1,
                 window_size=7, relative_pos=False):
        super().__init__()
        self.stages = stages
        self.token_dims = token_dims
        self.channels = list(token_dims)  # [64, 128, 256, 512]

        depth = sum(NC_depth)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        Layers = []
        cur_img_size = img_size
        cur_in_chans = in_chans
        for i in range(stages):
            startDpr = 0 if i == 0 else sum(NC_depth[:i])
            endDpr = sum(NC_depth[:i + 1])
            Layers.append(BasicLayer(
                img_size=cur_img_size, in_chans=cur_in_chans,
                embed_dims=embed_dims[i], token_dims=token_dims[i],
                downsample_ratios=downsample_ratios[i],
                kernel_size=kernel_size[i], RC_heads=RC_heads[i], NC_heads=NC_heads[i],
                dilations=dilations[i], RC_op='cat',
                RC_tokens_type=RC_tokens_type[i], NC_tokens_type=NC_tokens_type[i],
                RC_group=1, NC_group=[1, 32, 64, 128][i],
                NC_depth=NC_depth[i], dpr=dpr[startDpr:endDpr],
                mlp_ratio=mlp_ratio, window_size=window_size, relative_pos=relative_pos))
            cur_img_size = cur_img_size // downsample_ratios[i]
            cur_in_chans = token_dims[i]

        self.layers = nn.ModuleList(Layers)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """Return 4 multi-scale feature maps.

        For 256×256 input:
          stage1: [B, 64, 64, 64]
          stage2: [B, 128, 32, 32]
          stage3: [B, 256, 16, 16]
          stage4: [B, 512, 8, 8]
        """
        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            B, N, C = x.shape
            H = W = int(math.sqrt(N))
            feat = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
            outs.append(feat)
        return outs


def vitaev2_s(pretrained=False, img_size=224):
    """ViTAEv2-S backbone.

    Returns 4 feature maps:
      [B, 64, H/4, W/4], [B, 128, H/8, W/8],
      [B, 256, H/16, W/16], [B, 512, H/32, W/32]
    """
    return ViTAEv2(
        img_size=img_size,
        stages=4, embed_dims=[64, 64, 128, 256], token_dims=[64, 128, 256, 512],
        downsample_ratios=[4, 2, 2, 2], kernel_size=[7, 3, 3, 3],
        RC_heads=[1, 1, 2, 4], NC_heads=[1, 2, 4, 8],
        dilations=[[1, 2, 3, 4], [1, 2, 3], [1, 2], [1, 2]],
        RC_tokens_type=['window', 'window', 'transformer', 'transformer'],
        NC_tokens_type=['window', 'window', 'transformer', 'transformer'],
        NC_depth=[2, 2, 8, 2], drop_path_rate=0.1,
        window_size=7)
