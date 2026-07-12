import torch
import torch.nn as nn
import torch.nn.functional as F
from . import MobileNetV2
from einops import rearrange
import math


def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class LayerNorm_channel(nn.Module):
    def __init__(self, channel):
        super(LayerNorm_channel, self).__init__()
        self.norm = nn.LayerNorm(channel, eps=1e-5, elementwise_affine=True)

    def forward(self, x):
        h, w = x.shape[-2:]
        x = to_3d(x)
        x = self.norm(x)
        x = to_4d(x, h, w)
        return x


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.activation = nn.ReLU6()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.activation(x)


class DilatedBranch(nn.Module):
    def __init__(self, c, dw_expand, dilation=1):
        super().__init__()
        dw_channel = dw_expand * c
        self.branch = nn.Sequential(
            nn.Conv2d(dw_channel, dw_channel, kernel_size=3, padding=dilation,
                      stride=1, groups=dw_channel, bias=True, dilation=dilation)
        )

    def forward(self, x):
        return self.branch(x)


class MSAFusion(nn.Module):
    def __init__(self, fuse_d, id_d, out_d):
        super(MSAFusion, self).__init__()
        self.fuse_d = fuse_d
        self.id_d = id_d
        self.out_d = out_d
        self.up_d = 128

        self.conv_fuse1 = nn.Sequential(
            nn.Conv2d(self.fuse_d, self.up_d, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.up_d),
            nn.ReLU(inplace=True))
        self.conv_fuse2_1 = nn.Sequential(
            DilatedBranch(self.up_d, dw_expand=1, dilation=1),
            nn.BatchNorm2d(self.up_d),
            nn.ReLU(inplace=True))
        self.conv_fuse2_2 = nn.Sequential(
            DilatedBranch(self.up_d, dw_expand=1, dilation=4),
            nn.BatchNorm2d(self.up_d),
            nn.ReLU(inplace=True))
        self.conv_fuse2_3 = nn.Sequential(
            DilatedBranch(self.up_d, dw_expand=1, dilation=9),
            nn.BatchNorm2d(self.up_d),
            nn.ReLU(inplace=True))
        self.conv_fuse3 = nn.Sequential(
            nn.Conv2d(self.up_d, self.out_d, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.out_d),
        )
        self.conv_identity = nn.Conv2d(self.id_d, self.out_d, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, c_fuse, c):
        c_fuse = self.conv_fuse1(c_fuse)
        c_fuse = self.conv_fuse2_1(c_fuse) + self.conv_fuse2_2(c_fuse) + self.conv_fuse2_3(c_fuse)
        dout = channel_shuffle(c_fuse, gcd(self.up_d, self.out_d))
        c_fuse = self.conv_fuse3(dout)
        c_out = self.relu(c_fuse + self.conv_identity(c))
        return c_out


class MSAB(nn.Module):
    """Multi-Scale Semantic-Aware Bottleneck: unifies multi-level features and applies
    parallel dilated convolutions for multi-scale flood perception."""

    def __init__(self, in_d=None, out_d=64):
        super(MSAB, self).__init__()
        if in_d is None:
            in_d = [16, 24, 32, 96, 320]
        self.in_d = in_d
        self.mid_d = out_d
        self.out_d = out_d

        # --- scale 1 (highest resolution) ---
        self.conv_scale1_c1 = nn.Sequential(
            nn.Conv2d(self.in_d[0], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True))
        self.conv_scale2_c1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.in_d[0], self.mid_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.mid_d), nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, groups=self.mid_d),
            nn.BatchNorm2d(self.mid_d), nn.ReLU(inplace=True))
        self.conv_scale3_c1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(self.in_d[0], self.mid_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.mid_d), nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, groups=self.mid_d),
            nn.BatchNorm2d(self.mid_d), nn.ReLU(inplace=True))
        self.conv_scale4_c1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=8, stride=8),
            nn.Conv2d(self.in_d[0], self.mid_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.mid_d), nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, groups=self.mid_d),
            nn.BatchNorm2d(self.mid_d), nn.ReLU(inplace=True))
        self.conv_scale5_c1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=16, stride=16),
            nn.Conv2d(self.in_d[0], self.mid_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.mid_d), nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, groups=self.mid_d),
            nn.BatchNorm2d(self.mid_d), nn.ReLU(inplace=True))

        # --- scale 2 ---
        self.conv_scale1_c2 = nn.Sequential(
            nn.Conv2d(self.in_d[1], self.mid_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.mid_d), nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, groups=self.mid_d),
            nn.BatchNorm2d(self.mid_d), nn.ReLU(inplace=True))
        self.conv_scale2_c2 = nn.Sequential(
            nn.Conv2d(self.in_d[1], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True))
        self.conv_scale3_c2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.in_d[1], self.mid_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.mid_d), nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, groups=self.mid_d),
            nn.BatchNorm2d(self.mid_d), nn.ReLU(inplace=True))
        self.conv_scale4_c2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(self.in_d[1], self.mid_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.mid_d), nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, groups=self.mid_d),
            nn.BatchNorm2d(self.mid_d), nn.ReLU(inplace=True))
        self.conv_scale5_c2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=8, stride=8),
            nn.Conv2d(self.in_d[1], self.mid_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.mid_d), nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, groups=self.mid_d),
            nn.BatchNorm2d(self.mid_d), nn.ReLU(inplace=True))

        # --- scale 3 ---
        self.conv_scale1_c3 = nn.Sequential(
            nn.Conv2d(self.in_d[2], self.mid_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.mid_d), nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, groups=self.mid_d),
            nn.BatchNorm2d(self.mid_d), nn.ReLU(inplace=True))
        self.conv_scale2_c3 = nn.Sequential(
            nn.Conv2d(self.in_d[2], self.mid_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.mid_d), nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, groups=self.mid_d),
            nn.BatchNorm2d(self.mid_d), nn.ReLU(inplace=True))
        self.conv_scale3_c3 = nn.Sequential(
            nn.Conv2d(self.in_d[2], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True))
        self.conv_scale4_c3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.in_d[2], self.mid_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.mid_d), nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, groups=self.mid_d),
            nn.BatchNorm2d(self.mid_d), nn.ReLU(inplace=True))
        self.conv_scale5_c3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(self.in_d[2], self.mid_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.mid_d), nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, groups=self.mid_d),
            nn.BatchNorm2d(self.mid_d), nn.ReLU(inplace=True))

        # --- scale 4 ---
        self.conv_scale1_c4 = nn.Sequential(
            nn.Conv2d(self.in_d[3], self.mid_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.mid_d), nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, groups=self.mid_d),
            nn.BatchNorm2d(self.mid_d), nn.ReLU(inplace=True))
        self.conv_scale2_c4 = nn.Sequential(
            nn.Conv2d(self.in_d[3], self.mid_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.mid_d), nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, groups=self.mid_d),
            nn.BatchNorm2d(self.mid_d), nn.ReLU(inplace=True))
        self.conv_scale3_c4 = nn.Sequential(
            nn.Conv2d(self.in_d[3], self.mid_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.mid_d), nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, groups=self.mid_d),
            nn.BatchNorm2d(self.mid_d), nn.ReLU(inplace=True))
        self.conv_scale4_c4 = nn.Sequential(
            nn.Conv2d(self.in_d[3], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True))
        self.conv_scale5_c4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.in_d[3], self.mid_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.mid_d), nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, groups=self.mid_d),
            nn.BatchNorm2d(self.mid_d), nn.ReLU(inplace=True))

        # --- scale 5 (lowest resolution) ---
        self.conv_scale1_c5 = nn.Sequential(
            nn.Conv2d(self.in_d[4], self.mid_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.mid_d), nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, groups=self.mid_d),
            nn.BatchNorm2d(self.mid_d), nn.ReLU(inplace=True))
        self.conv_scale2_c5 = nn.Sequential(
            nn.Conv2d(self.in_d[4], self.mid_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.mid_d), nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, groups=self.mid_d),
            nn.BatchNorm2d(self.mid_d), nn.ReLU(inplace=True))
        self.conv_scale3_c5 = nn.Sequential(
            nn.Conv2d(self.in_d[4], self.mid_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.mid_d), nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, groups=self.mid_d),
            nn.BatchNorm2d(self.mid_d), nn.ReLU(inplace=True))
        self.conv_scale4_c5 = nn.Sequential(
            nn.Conv2d(self.in_d[4], self.mid_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.mid_d), nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, groups=self.mid_d),
            nn.BatchNorm2d(self.mid_d), nn.ReLU(inplace=True))
        self.conv_scale5_c5 = nn.Sequential(
            nn.Conv2d(self.in_d[4], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True))

        # per-scale fusion blocks
        self.agg_s1 = MSAFusion(self.mid_d, self.in_d[0], self.out_d)
        self.agg_s2 = MSAFusion(self.mid_d, self.in_d[1], self.out_d)
        self.agg_s3 = MSAFusion(self.mid_d, self.in_d[2], self.out_d)
        self.agg_s4 = MSAFusion(self.mid_d, self.in_d[3], self.out_d)
        self.agg_s5 = MSAFusion(self.mid_d, self.in_d[4], self.out_d)

    def forward(self, c1, c2, c3, c4, c5):
        # c1 -> 5 scales
        c1_s1 = self.conv_scale1_c1(c1)
        c1_s2 = self.conv_scale2_c1(c1)
        c1_s3 = self.conv_scale3_c1(c1)
        c1_s4 = self.conv_scale4_c1(c1)
        c1_s5 = self.conv_scale5_c1(c1)

        # c2 -> 5 scales
        c2_s1 = F.interpolate(self.conv_scale1_c2(c2), scale_factor=(2, 2), mode='bilinear')
        c2_s2 = self.conv_scale2_c2(c2)
        c2_s3 = self.conv_scale3_c2(c2)
        c2_s4 = self.conv_scale4_c2(c2)
        c2_s5 = self.conv_scale5_c2(c2)

        # c3 -> 5 scales
        c3_s1 = F.interpolate(self.conv_scale1_c3(c3), scale_factor=(4, 4), mode='bilinear')
        c3_s2 = F.interpolate(self.conv_scale2_c3(c3), scale_factor=(2, 2), mode='bilinear')
        c3_s3 = self.conv_scale3_c3(c3)
        c3_s4 = self.conv_scale4_c3(c3)
        c3_s5 = self.conv_scale5_c3(c3)

        # c4 -> 5 scales
        c4_s1 = F.interpolate(self.conv_scale1_c4(c4), scale_factor=(8, 8), mode='bilinear')
        c4_s2 = F.interpolate(self.conv_scale2_c4(c4), scale_factor=(4, 4), mode='bilinear')
        c4_s3 = F.interpolate(self.conv_scale3_c4(c4), scale_factor=(2, 2), mode='bilinear')
        c4_s4 = self.conv_scale4_c4(c4)
        c4_s5 = self.conv_scale5_c4(c4)

        # c5 -> 5 scales
        c5_s1 = F.interpolate(self.conv_scale1_c5(c5), scale_factor=(16, 16), mode='bilinear')
        c5_s2 = F.interpolate(self.conv_scale2_c5(c5), scale_factor=(8, 8), mode='bilinear')
        c5_s3 = F.interpolate(self.conv_scale3_c5(c5), scale_factor=(4, 4), mode='bilinear')
        c5_s4 = F.interpolate(self.conv_scale4_c5(c5), scale_factor=(2, 2), mode='bilinear')
        c5_s5 = self.conv_scale5_c5(c5)

        s1 = self.agg_s1(c1_s1 + c2_s1 + c3_s1 + c4_s1 + c5_s1, c1)
        s2 = self.agg_s2(c1_s2 + c2_s2 + c3_s2 + c4_s2 + c5_s2, c2)
        s3 = self.agg_s3(c1_s3 + c2_s3 + c3_s3 + c4_s3 + c5_s3, c3)
        s4 = self.agg_s4(c1_s4 + c2_s4 + c3_s4 + c4_s4 + c5_s4, c4)
        s5 = self.agg_s5(c1_s5 + c2_s5 + c3_s5 + c4_s5 + c5_s5, c5)

        return s1, s2, s3, s4, s5


def dsconv_3x3(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, groups=in_channel),
        nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True))


class TFF(nn.Module):
    """Temporal Feature Fusion: learns spatial attention weights to re-weight
    bi-temporal features before computing their absolute difference."""

    def __init__(self, in_channel):
        super(TFF, self).__init__()
        self.catconvB = dsconv_3x3(in_channel * 2, in_channel)
        self.catconvA = dsconv_3x3(in_channel * 2, in_channel)
        self.convA = nn.Conv2d(in_channel, 1, 1)
        self.convB = nn.Conv2d(in_channel, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, xA, xB):
        x_diff = xA - xB

        x_diffA = self.catconvA(torch.cat([x_diff, xA], dim=1))
        x_diffB = self.catconvB(torch.cat([x_diff, xB], dim=1))

        A_weight = self.sigmoid(self.convA(x_diffA))
        B_weight = self.sigmoid(self.convB(x_diffB))

        xA = A_weight * xA
        xB = B_weight * xB
        x = torch.abs(xA - xB)
        return x


class ChangeFusion(nn.Module):
    def __init__(self, in_d, out_d):
        super(ChangeFusion, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_d, self.out_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.out_d),
            nn.ReLU(inplace=True))
        self.TFF = TFF(in_channel=64)

    def forward(self, x1, x2):
        x = self.TFF(x1, x2)
        x = self.conv(x)
        return x


class TemporalFusion(nn.Module):
    def __init__(self, in_d=64, out_d=64):
        super(TemporalFusion, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.gf_x1 = ChangeFusion(self.in_d, self.out_d)
        self.gf_x2 = ChangeFusion(self.in_d, self.out_d)
        self.gf_x3 = ChangeFusion(self.in_d, self.out_d)
        self.gf_x4 = ChangeFusion(self.in_d, self.out_d)
        self.gf_x5 = ChangeFusion(self.in_d, self.out_d)

    def forward(self, x1_1, x1_2, x1_3, x1_4, x1_5, x2_1, x2_2, x2_3, x2_4, x2_5):
        c1 = self.gf_x1(x1_1, x2_1)
        c2 = self.gf_x2(x1_2, x2_2)
        c3 = self.gf_x3(x1_3, x2_3)
        c4 = self.gf_x4(x1_4, x2_4)
        c5 = self.gf_x5(x1_5, x2_5)
        return c1, c2, c3, c4, c5


class CSFB(nn.Module):
    """Cross-Level Semantic-Guided Frequency Boundary-Aware module.
    Computes frequency-domain correlation between deep and shallow features via FFT,
    then adaptively gates the shallow features for boundary-aware fusion."""

    def __init__(self, channel, num_heads):
        super().__init__()
        assert channel % num_heads == 0

        self.LayerNorm_high = LayerNorm_channel(channel=channel)
        self.LayerNorm_low = LayerNorm_channel(channel=channel)
        self.norm_img2 = LayerNorm_channel(channel=channel)
        self.norm_fft = LayerNorm_channel(channel=channel)

        self.high_conv = nn.Conv2d(channel, channel, kernel_size=1)
        self.low_conv = nn.Conv2d(channel, channel, kernel_size=1)

        self.num_heads = num_heads

        self.conv_q = DepthwiseSeparableConv(channel, channel)
        self.conv_k = DepthwiseSeparableConv(channel, channel)
        self.conv_v0 = DepthwiseSeparableConv(channel, channel)
        self.conv_v1 = DepthwiseSeparableConv(channel, channel)

        self.ds_conv_out0 = DepthwiseSeparableConv(channel, channel)
        self.ds_conv_out1 = DepthwiseSeparableConv(channel, channel)

        self.img_upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, low_f, high_f):
        image = self.img_upsample(high_f)
        b, c, h, w = image.shape

        if low_f.shape[-2:] != (h, w):
            low_f = F.interpolate(low_f, size=(h, w), mode="bilinear", align_corners=False)

        h_feat = self.high_conv(self.LayerNorm_high(image))
        l_feat = self.low_conv(self.LayerNorm_low(low_f))

        q = self.conv_q(h_feat)
        k = self.conv_k(l_feat)
        v0 = self.conv_v0(l_feat)
        v1 = self.conv_v1(l_feat)

        # frequency-domain correlation
        q_f = torch.fft.rfft2(q.float(), dim=(-2, -1))
        k_f = torch.fft.rfft2(k.float(), dim=(-2, -1))
        out_f = q_f * k_f.conj()
        out = torch.fft.irfft2(out_f, s=(h, w), dim=(-2, -1))
        out = out.to(dtype=q.dtype)
        out = self.norm_fft(out)

        # multi-head gating
        v0_m = rearrange(v0, 'b (head ch) h w -> b head ch (h w)', head=self.num_heads)
        v1_m = rearrange(v1, 'b (head ch) h w -> b head ch (h w)', head=self.num_heads)
        out_m = rearrange(out, 'b (head ch) h w -> b head ch (h w)', head=self.num_heads)

        out0 = out_m * v0_m
        out0 = rearrange(out0, 'b head ch (h w) -> b (head ch) h w', head=self.num_heads, h=h, w=w)
        out0 = self.ds_conv_out0(out0)

        out1 = out_m * v1_m
        out1 = rearrange(out1, 'b head ch (h w) -> b (head ch) h w', head=self.num_heads, h=h, w=w)
        out1 = self.ds_conv_out1(out1)

        fused = self.norm_img2(image) * out0 + out1

        if not self.training and getattr(self, "save_cache", True):
            q_spec = torch.fft.fftshift(torch.fft.fft2(q.float(), dim=(-2, -1)), dim=(-2, -1))
            k_spec = torch.fft.fftshift(torch.fft.fft2(k.float(), dim=(-2, -1)), dim=(-2, -1))
            self._fft_cache = {
                "q": q.detach(),
                "k": k.detach(),
                "q_fft": q_spec.detach(),
                "k_fft": k_spec.detach(),
                "out": out.detach(),
                "fused": fused.detach(),
            }

        return fused


class DecoderStage(nn.Module):
    def __init__(self, mid_d):
        super(DecoderStage, self).__init__()
        self.mid_d = mid_d
        self.conv_high = nn.Conv2d(self.mid_d, self.mid_d, kernel_size=1, stride=1)
        self.fusion = nn.Sequential(
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, groups=self.mid_d),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True))
        self.cls = nn.Conv2d(self.mid_d, 1, kernel_size=1)

    def forward(self, x_low, x_high):
        batch, channels, height, width = x_low.shape
        x_high = F.interpolate(self.conv_high(x_high), size=(height, width), mode="bilinear")
        x_fused = x_low + x_high
        x_fused = self.fusion(x_fused)
        mask = self.cls(x_fused)
        return x_fused, mask


class Decoder(nn.Module):
    def __init__(self, mid_d=320):
        super(Decoder, self).__init__()
        self.mid_d = mid_d
        self.csfb4 = CSFB(self.mid_d, num_heads=2)
        self.csfb3 = CSFB(self.mid_d, num_heads=2)
        self.csfb2 = CSFB(self.mid_d, num_heads=2)
        self.csfb1 = CSFB(self.mid_d, num_heads=2)
        self.db_p4 = DecoderStage(self.mid_d)
        self.db_p3 = DecoderStage(self.mid_d)
        self.db_p2 = DecoderStage(self.mid_d)
        self.db_p1 = DecoderStage(self.mid_d)
        self.cls = nn.Conv2d(self.mid_d, 1, kernel_size=1)
        self.conv_p5 = nn.Conv2d(self.mid_d, self.mid_d, kernel_size=1, stride=1)

    def forward(self, d1, d2, d3, d4, d5):
        p4 = self.csfb4(d4, d5)
        p4, mask_p4 = self.db_p4(p4, d5)
        p3 = self.csfb3(d3, p4)
        p3, mask_p3 = self.db_p3(p3, p4)
        p2 = self.csfb2(d2, p3)
        p2, mask_p2 = self.db_p2(p2, p3)
        p1 = self.csfb1(d1, p2)
        p1, mask_p1 = self.db_p1(p1, p2)
        return mask_p1, mask_p2, mask_p3, mask_p4


class CFBNet(nn.Module):
    def __init__(self, input_nc=3, output_nc=1):
        super(CFBNet, self).__init__()
        self.backbone = MobileNetV2.mobilenet_v2(pretrained=True)
        channels = [16, 24, 32, 96, 320]
        self.en_d = 32
        self.mid_d = self.en_d * 2
        self.msab = MSAB(channels, self.mid_d)
        self.temporal_fusion = TemporalFusion(self.mid_d, self.mid_d)
        self.decoder = Decoder(self.en_d * 2)

    def forward(self, x1, x2):
        x1_1, x1_2, x1_3, x1_4, x1_5 = self.backbone(x1)
        x2_1, x2_2, x2_3, x2_4, x2_5 = self.backbone(x2)

        x1_1, x1_2, x1_3, x1_4, x1_5 = self.msab(x1_1, x1_2, x1_3, x1_4, x1_5)
        x2_1, x2_2, x2_3, x2_4, x2_5 = self.msab(x2_1, x2_2, x2_3, x2_4, x2_5)

        c1, c2, c3, c4, c5 = self.temporal_fusion(
            x1_1, x1_2, x1_3, x1_4, x1_5,
            x2_1, x2_2, x2_3, x2_4, x2_5)

        mask_p1, mask_p2, mask_p3, mask_p4 = self.decoder(c1, c2, c3, c4, c5)

        mask_p1 = F.interpolate(mask_p1, scale_factor=(2, 2), mode='bilinear')
        mask_p1 = torch.sigmoid(mask_p1)
        mask_p2 = F.interpolate(mask_p2, scale_factor=(4, 4), mode='bilinear')
        mask_p2 = torch.sigmoid(mask_p2)
        mask_p3 = F.interpolate(mask_p3, scale_factor=(8, 8), mode='bilinear')
        mask_p3 = torch.sigmoid(mask_p3)
        mask_p4 = F.interpolate(mask_p4, scale_factor=(16, 16), mode='bilinear')
        mask_p4 = torch.sigmoid(mask_p4)

        return mask_p1, mask_p2, mask_p3, mask_p4
