import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial
import warnings


# ==================== MiT Backbone (SegFormer) ====================

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.", stacklevel=2)
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
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None: m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim; self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None: m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None, scale_by_keep=True):
        super().__init__()
        self.drop_prob = drop_prob; self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None: m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None: m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class MixVisionTransformer(nn.Module):
    def __init__(self, in_chans=3, embed_dims=[32, 64, 160, 256],
                 num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.depths = depths
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.patch_embed1 = OverlapPatchEmbed(7, 4, in_chans, embed_dims[0])
        cur = 0
        self.block1 = nn.ModuleList([
            Block(embed_dims[0], num_heads[0], mlp_ratios[0], qkv_bias=qkv_bias,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                  norm_layer=norm_layer, sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        self.patch_embed2 = OverlapPatchEmbed(3, 2, embed_dims[0], embed_dims[1])
        cur += depths[0]
        self.block2 = nn.ModuleList([
            Block(embed_dims[1], num_heads[1], mlp_ratios[1], qkv_bias=qkv_bias,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                  norm_layer=norm_layer, sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        self.patch_embed3 = OverlapPatchEmbed(3, 2, embed_dims[1], embed_dims[2])
        cur += depths[1]
        self.block3 = nn.ModuleList([
            Block(embed_dims[2], num_heads[2], mlp_ratios[2], qkv_bias=qkv_bias,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                  norm_layer=norm_layer, sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        self.patch_embed4 = OverlapPatchEmbed(3, 2, embed_dims[2], embed_dims[3])
        cur += depths[2]
        self.block4 = nn.ModuleList([
            Block(embed_dims[3], num_heads[3], mlp_ratios[3], qkv_bias=qkv_bias,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                  norm_layer=norm_layer, sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None: m.bias.data.zero_()

    def forward(self, x):
        B = x.shape[0]
        outs = []
        x, H, W = self.patch_embed1(x)
        for blk in self.block1: x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        x, H, W = self.patch_embed2(x)
        for blk in self.block2: x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        x, H, W = self.patch_embed3(x)
        for blk in self.block3: x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        x, H, W = self.patch_embed4(x)
        for blk in self.block4: x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        return outs


# ==================== CBAM ====================

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv1(x))


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        return x * self.sa(x * self.ca(x))


# ==================== Semantic Flow ====================

class SemanticFlow(nn.Module):
    def __init__(self, inchannel, outchannel):
        super().__init__()
        self.down_h = nn.Conv2d(inchannel, outchannel, 1, bias=False)
        self.down_l = nn.Conv2d(outchannel, outchannel, 1, bias=False)
        self.flow_make = nn.Conv2d(outchannel * 2, 2, kernel_size=3, padding=1, bias=False)

    def forward(self, h_feature, low_feature):
        low_feature, h_feature = low_feature, h_feature
        h_feature_orign = h_feature
        h, w = low_feature.size()[2:]
        size = (h, w)
        low_feature = self.down_l(low_feature)
        h_feature = self.down_h(h_feature)
        h_feature_up = F.interpolate(h_feature, size=size, mode="bilinear", align_corners=False)
        flow = self.flow_make(torch.cat([h_feature_up, low_feature], 1))
        h_feature = self.flow_warp(h_feature_orign, flow, size=size)
        return h_feature

    @staticmethod
    def flow_warp(inputs, flow, size):
        out_h, out_w = size
        n, c, h, w = inputs.size()
        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(inputs).to(inputs.device)
        w_coord = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h_coord = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h_coord.unsqueeze(2), w_coord.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(inputs).to(inputs.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm
        return F.grid_sample(inputs, grid, align_corners=False)


# ==================== Pyramid Module ====================

class PyramidExtraction(nn.Module):
    def __init__(self, channel, rate=1, bn_mom=0.1):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, rate, dilation=rate, groups=channel, bias=True),
            nn.BatchNorm2d(channel, momentum=bn_mom), nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 1, bias=True), nn.BatchNorm2d(channel, momentum=bn_mom), nn.ReLU(inplace=True))
        self.branch2 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1, groups=channel, bias=False),
            nn.BatchNorm2d(channel, momentum=bn_mom), nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 1, bias=False), nn.BatchNorm2d(channel, momentum=bn_mom), nn.ReLU(inplace=True))
        self.branch3 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 4 * rate, dilation=4 * rate, groups=channel, bias=False),
            nn.BatchNorm2d(channel, momentum=bn_mom), nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 1, bias=False), nn.BatchNorm2d(channel, momentum=bn_mom), nn.ReLU(inplace=True))
        self.branch4 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 8 * rate, dilation=8 * rate, groups=channel, bias=False),
            nn.BatchNorm2d(channel, momentum=bn_mom), nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 1, bias=False), nn.BatchNorm2d(channel, momentum=bn_mom), nn.ReLU(inplace=True))
        self.branch5_conv = nn.Conv2d(channel, channel, 1, bias=True)
        self.branch5_bn = nn.BatchNorm2d(channel, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(channel * 5, channel * 5, 1, groups=channel * 5, bias=False),
            nn.BatchNorm2d(channel * 5, momentum=bn_mom), nn.ReLU(inplace=True),
            nn.Conv2d(channel * 5, channel, 1, bias=False),
            nn.BatchNorm2d(channel, momentum=bn_mom), nn.ReLU(inplace=True))

    def forward(self, x):
        b, c, row, col = x.size()
        conv1_1 = self.branch1(x)
        conv3_1 = self.branch2(x)
        conv3_2 = self.branch3(x)
        conv3_3 = self.branch4(x)
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_relu(self.branch5_bn(self.branch5_conv(global_feature)))
        global_feature = F.interpolate(global_feature, (row, col), mode='bilinear', align_corners=True)
        feature_cat = torch.cat([conv1_1, conv3_1, conv3_2, conv3_3, global_feature], dim=1)
        return self.conv_cat(feature_cat)


class PyramidMerge(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.cbam = CBAM(channel)
        self.pe = PyramidExtraction(channel)
        self.conv = nn.Sequential(
            nn.Conv2d(channel * 2, channel * 2, 1, groups=channel * 2), nn.BatchNorm2d(channel * 2), nn.ReLU(),
            nn.Conv2d(channel * 2, channel, 1), nn.BatchNorm2d(channel), nn.ReLU())

    def forward(self, input1, input2):
        input_cat = torch.cat([input1, input2], dim=1)
        input_abs = torch.abs(input1 - input2)
        input_cat_conv = self.cbam(self.conv(input_cat))
        input_abs_py = self.pe(input_abs)
        return input_abs_py + input_abs + input_cat_conv


# ==================== Edge-Aware Refinement ====================

class DoubleConv(nn.Module):
    def __init__(self, input_channels, num_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, 3, 1, 1, groups=input_channels),
            nn.BatchNorm2d(input_channels), nn.ReLU(),
            nn.Conv2d(input_channels, num_channels, 1), nn.BatchNorm2d(num_channels), nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, 3, 1, 1, groups=num_channels),
            nn.BatchNorm2d(num_channels), nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, 1), nn.BatchNorm2d(num_channels), nn.ReLU())

    def forward(self, X):
        return self.conv2(self.conv1(X))


class EdgeAware(nn.Module):
    def __init__(self, channel1, channel2):
        super().__init__()
        self.conv_channel1 = nn.Sequential(
            nn.Conv2d(channel1, channel1, 3, 1, 1, groups=channel1), nn.BatchNorm2d(channel1), nn.ReLU(),
            nn.Conv2d(channel1, channel1, 1), nn.BatchNorm2d(channel1), nn.ReLU())
        self.conv_channel2 = nn.Sequential(
            nn.Conv2d(channel2, channel2, 3, 1, 1, groups=channel2), nn.BatchNorm2d(channel2), nn.ReLU(),
            nn.Conv2d(channel2, channel2, 1), nn.BatchNorm2d(channel2), nn.ReLU())
        self.doubleconv = DoubleConv(channel1 + channel2, (channel1 + channel2) * 2)
        self.conv1 = nn.Sequential(nn.Conv2d((channel1 + channel2) * 2, 2, 1), nn.BatchNorm2d(2))
        self.sef = SemanticFlow(channel2, channel1)

    def forward(self, input1, input4):
        input1 = self.conv_channel1(input1)
        input4 = self.conv_channel2(input4)
        input4_sef = self.sef(input4, input1)
        input4_big = F.interpolate(input4, size=input1.size()[2:], mode='bilinear', align_corners=False) + input4_sef
        inp = torch.cat([input4_big, input1], dim=1)
        return self.conv1(self.doubleconv(inp))


class EdgeGuidance(nn.Module):
    def __init__(self, channels, embed_dim):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.maxpool3 = nn.MaxPool2d(2, 2)
        self.ea = EdgeAware(c1, c4)
        self.conv2_1 = nn.Sequential(nn.Conv2d(2, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.ca1 = CBAM(c1)
        self.ca2 = CBAM(c2)
        self.ca3 = CBAM(c3)
        self.ca4 = CBAM(c4)

    def forward(self, inputs):
        input1, input2, input3, input4 = inputs
        edge_2 = self.ea(input1, input4)
        edge_1 = self.conv2_1(edge_2)
        edge_32 = self.maxpool1(edge_1)
        edge_16 = self.maxpool2(edge_32)
        edge_8 = self.maxpool3(edge_16)

        f1 = self.ca1(input1 * edge_1 + input1)
        f2 = self.ca2(input2 * edge_32 + input2)
        f3 = self.ca3(input3 * edge_16 + input3)
        f4 = self.ca4(input4 * edge_8 + input4)
        return edge_2, f1, f2, f3, f4


# ==================== Decoder Head ====================

class MLP(nn.Module):
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        return self.proj(x)


class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SegFormerHead(nn.Module):
    def __init__(self, num_classes=2, in_channels=[32, 64, 160, 256], embedding_dim=256):
        super().__init__()
        c1_in, c2_in, c3_in, c4_in = in_channels
        self.linear_c4 = MLP(c4_in, embedding_dim)
        self.linear_c3 = MLP(c3_in, embedding_dim)
        self.linear_c2 = MLP(c2_in, embedding_dim)
        self.linear_c1 = MLP(c1_in, embedding_dim)
        self.linear_fuse = ConvModule(embedding_dim * 4, embedding_dim, k=1)
        self.sef1 = SemanticFlow(embedding_dim, c1_in)
        self.sef2 = SemanticFlow(embedding_dim, c1_in)
        self.sef3 = SemanticFlow(embedding_dim, c1_in)
        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, 1)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = self.sef1(_c4, c1) + F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = self.sef2(_c3, c1) + F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = self.sef3(_c2, c1) + F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        return self.linear_pred(self.dropout(_c))


class Resampler(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, output_size), torch.linspace(-1, 1, output_size), indexing='ij')
        self.register_buffer('grid', torch.stack((grid_x, grid_y), 2).unsqueeze(0))

    def forward(self, x):
        grid = self.grid.repeat(x.size(0), 1, 1, 1)
        return F.grid_sample(x, grid, align_corners=True)


# ==================== Main SFEARNet ====================

class SFEARNet(nn.Module):
    """SFEARNet adapted for CFB-Net framework.

    SFEARNet: Semantic Flow and Edge-Aware Refinement Network for
    Highly Efficient Remote Sensing Image Change Detection (TGRS 2025)
    https://github.com/miao-0417/SFEARNet

    Uses MiT (SegFormer-style) backbone with Semantic Flow Information
    Transfer Module (SFITM), Pyramid Feature Enhancement Module (PFEM),
    and Edge-Aware Refinement Module (EARM).
    """
    def __init__(self, input_nc=3, output_nc=1, phi='b0', pretrained=False):
        super().__init__()
        self.phi = phi
        if phi == 'b1':
            embed_dims = [64, 128, 320, 512]
            num_heads = [1, 2, 5, 8]
            embedding_dim = 256
        else:  # b0 (default)
            embed_dims = [32, 64, 160, 256]
            num_heads = [1, 2, 5, 8]
            embedding_dim = 256

        self.backbone = MixVisionTransformer(
            in_chans=input_nc, embed_dims=embed_dims, num_heads=num_heads,
            mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)

        self.py1 = PyramidMerge(embed_dims[0])
        self.py2 = PyramidMerge(embed_dims[1])
        self.py3 = PyramidMerge(embed_dims[2])
        self.py4 = PyramidMerge(embed_dims[3])

        self.eg = EdgeGuidance(embed_dims, embedding_dim)
        self.decode_head = SegFormerHead(output_nc, embed_dims, embedding_dim)
        self.re = Resampler(embed_dims[0], embedding_dim)
        self.re2 = Resampler(embed_dims[0], embedding_dim)

    def forward(self, x1, x2):
        H, W = x1.size(2), x1.size(3)

        feat1 = self.backbone(x1)
        feat2 = self.backbone(x2)

        # Pyramid merge at each scale
        f0 = self.py1(feat1[0], feat2[0])
        f1 = self.py2(feat1[1], feat2[1])
        f2 = self.py3(feat1[2], feat2[2])
        f3 = self.py4(feat1[3], feat2[3])

        x = [f0, f1, f2, f3]

        # Edge guidance
        eg_out = self.eg(x)
        edge = eg_out[0]
        enhanced_feats = eg_out[1:]

        # Decode
        out = self.decode_head(enhanced_feats)
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=True)

        # Multi-scale outputs for framework compatibility
        m1 = torch.sigmoid(out)
        m2 = torch.sigmoid(F.interpolate(
            nn.Conv2d(f1.size(1), 1, 1, device=f1.device)(f1),
            size=(H, W), mode='bilinear', align_corners=True))
        m3 = torch.sigmoid(F.interpolate(
            nn.Conv2d(f2.size(1), 1, 1, device=f2.device)(f2),
            size=(H, W), mode='bilinear', align_corners=True))
        m4 = torch.sigmoid(F.interpolate(
            nn.Conv2d(f3.size(1), 1, 1, device=f3.device)(f3),
            size=(H, W), mode='bilinear', align_corners=True))

        return m1, m2, m3, m4
