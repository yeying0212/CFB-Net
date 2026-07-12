import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ==================== ACAHNet Helpers ====================

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, padding=padding, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class ConvNormAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, norm=nn.BatchNorm2d, act=nn.GELU, preact=True):
        super().__init__()
        self.norm = norm(in_ch) if preact else nn.Identity()
        self.act = act() if preact else nn.Identity()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        return self.conv(self.act(self.norm(x)))


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm=nn.BatchNorm2d, act=nn.GELU):
        super().__init__()
        self.conv1 = ConvNormAct(in_ch, out_ch, 3, 1, norm, act)
        self.conv2 = ConvNormAct(out_ch, out_ch, 3, 1, norm, act)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.conv2(self.conv1(x)) + self.skip(x)


class MBConv(nn.Module):
    def __init__(self, in_ch, out_ch, expansion=4, kernel_size=3, act=nn.GELU, norm=nn.BatchNorm2d, p=0.):
        super().__init__()
        hidden = in_ch * expansion
        self.expand = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1, bias=False),
            norm(hidden), act()) if expansion > 1 else nn.Identity()
        self.dw = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size, 1, kernel_size//2, groups=hidden, bias=False),
            norm(hidden), act())
        self.project = nn.Sequential(
            nn.Conv2d(hidden, out_ch, 1, bias=False), norm(out_ch))
        self.dropout = nn.Dropout2d(p) if p > 0 else nn.Identity()
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        out = self.expand(x)
        out = self.dw(out)
        out = self.dropout(out)
        out = self.project(out)
        return out + self.skip(x)


# ==================== Transformer Components ====================

class BidirectionAttention(nn.Module):
    def __init__(self, feat_dim, map_dim, out_dim, heads=4, dim_head=64,
                 attn_drop=0., proj_drop=0., map_size=8, proj_type='depthwise'):
        super().__init__()
        self.inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** (-0.5)
        self.dim_head = dim_head
        self.map_size = map_size

        if proj_type == 'linear':
            self.feat_qv = nn.Conv2d(feat_dim, self.inner_dim * 2, 1, bias=False)
            self.feat_out = nn.Conv2d(self.inner_dim, out_dim, 1, bias=False)
        else:
            self.feat_qv = DepthwiseSeparableConv(feat_dim, self.inner_dim * 2)
            self.feat_out = DepthwiseSeparableConv(self.inner_dim, out_dim)

        self.map_qv = nn.Conv2d(map_dim, self.inner_dim * 2, 1, bias=False)
        self.map_out = nn.Conv2d(self.inner_dim, map_dim, 1, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, feat, semantic_map):
        B, C, H, W = feat.shape

        feat_q, feat_v = self.feat_qv(feat).chunk(2, dim=1)
        map_q, map_v = self.map_qv(semantic_map).chunk(2, dim=1)

        feat_q = rearrange(feat_q, 'b (h d) x y -> b h (x y) d', h=self.heads, d=self.dim_head)
        feat_v = rearrange(feat_v, 'b (h d) x y -> b h (x y) d', h=self.heads, d=self.dim_head)
        map_q = rearrange(map_q, 'b (h d) x y -> b h (x y) d', h=self.heads, d=self.dim_head)
        map_v = rearrange(map_v, 'b (h d) x y -> b h (x y) d', h=self.heads, d=self.dim_head)

        attn = torch.einsum('bhid,bhjd->bhij', feat_q, map_q) * self.scale
        feat_map_attn = F.softmax(attn, dim=-1)
        map_feat_attn = self.attn_drop(F.softmax(attn, dim=-2))

        feat_out = torch.einsum('bhij,bhjd->bhid', feat_map_attn, map_v)
        feat_out = rearrange(feat_out, 'b h (x y) d -> b (h d) x y', h=self.heads, x=H, y=W, d=self.dim_head)

        map_out = torch.einsum('bhji,bhjd->bhid', map_feat_attn, feat_v)
        map_out = rearrange(map_out, 'b h (x y) d -> b (h d) x y', h=self.heads, x=self.map_size, y=self.map_size, d=self.dim_head)

        feat_out = self.proj_drop(self.feat_out(feat_out))
        map_out = self.proj_drop(self.map_out(map_out))
        return feat_out, map_out


class BidirectionAttentionBlock(nn.Module):
    def __init__(self, feat_dim, map_dim, out_dim, heads, dim_head,
                 expansion=4, attn_drop=0., proj_drop=0., map_size=8, proj_type='depthwise',
                 norm=nn.BatchNorm2d, act=nn.GELU):
        super().__init__()
        self.norm1 = norm(feat_dim) if norm else nn.Identity()
        self.norm2 = norm(map_dim) if norm else nn.Identity()
        self.attn = BidirectionAttention(feat_dim, map_dim, out_dim, heads=heads, dim_head=dim_head,
                                          attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size, proj_type=proj_type)
        self.shortcut = nn.Sequential()
        if feat_dim != out_dim:
            self.shortcut = ConvNormAct(feat_dim, out_dim, 1, 0, norm, act, preact=True)
        self.feedforward = MBConv(out_dim, out_dim, expansion=expansion, kernel_size=3, act=act, norm=norm)

    def forward(self, x, semantic_map):
        feat, mapp = self.norm1(x), self.norm2(semantic_map)
        out, mapp = self.attn(feat, mapp)
        out = out + self.shortcut(x)
        out = self.feedforward(out)
        mapp = mapp + semantic_map
        return out, mapp


class SemanticMapGeneration(nn.Module):
    def __init__(self, feat_dim, map_dim, map_size):
        super().__init__()
        self.map_size = map_size
        self.map_dim = map_dim
        self.map_code_num = map_size * map_size
        self.base_proj = nn.Conv2d(feat_dim, map_dim, 3, 1, bias=False)
        self.semantic_proj = nn.Conv2d(feat_dim, self.map_code_num, 3, 1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        feat = self.base_proj(x)
        weight_map = self.semantic_proj(x).view(B, self.map_code_num, -1)
        weight_map = F.softmax(weight_map, dim=2)
        feat = feat.view(B, self.map_dim, -1)
        semantic_map = torch.einsum('bij,bkj->bik', feat, weight_map)
        return semantic_map.view(B, self.map_dim, self.map_size, self.map_size)


# ==================== Main ACAHNet ====================

class ACAHNet(nn.Module):
    """ACAHNet adapted for CFB-Net framework.

    ACAHNet: Attentive Cross-Attention Hybrid Network for Change Detection (TGRS 2023)
    https://github.com/CCRG-XJU/ChangeDetection_ACAHNet_TGRS2023
    """
    def __init__(self, input_nc=3, output_nc=1):
        super().__init__()
        base_chan = 16
        map_size = 8
        conv_block = BasicBlock
        chan_num = [2*base_chan, 4*base_chan, 8*base_chan, 16*base_chan,
                    8*base_chan, 4*base_chan, 2*base_chan, base_chan]
        conv_num = [2, 1, 0, 0, 0, 1, 2, 2]
        trans_num = [0, 1, 2, 2, 2, 1, 0, 0]
        num_heads = [1, 4, 8, 16, 8, 4, 1, 1]
        dim_head = [chan_num[i] // num_heads[i] for i in range(8)]

        # Encoder stem
        self.inc = nn.Sequential(
            nn.Conv2d(input_nc, base_chan, 3, 1, 1, bias=False),
            BasicBlock(base_chan, base_chan))

        # Down-sampling blocks
        self.down1 = DownBlock(base_chan, chan_num[0], conv_num[0], trans_num[0],
                               conv_block, heads=num_heads[0], dim_head=dim_head[0],
                               map_size=map_size, map_generate=False)
        self.down2 = DownBlock(chan_num[0], chan_num[1], conv_num[1], trans_num[1],
                               conv_block, heads=num_heads[1], dim_head=dim_head[1],
                               map_size=map_size, map_generate=True)
        self.down3 = DownBlock(chan_num[1], chan_num[2], conv_num[2], trans_num[2],
                               conv_block, heads=num_heads[2], dim_head=dim_head[2],
                               map_size=map_size, map_generate=False, map_proj=True)
        self.down4 = DownBlock(chan_num[2], chan_num[3], conv_num[3], trans_num[3],
                               conv_block, heads=num_heads[3], dim_head=dim_head[3],
                               map_size=map_size, map_generate=False, map_proj=True)

        # Fusion blocks for bi-temporal features
        self.fusion0 = nn.Conv2d(base_chan * 2, base_chan, 1)
        self.fusion1 = nn.Conv2d(chan_num[0] * 2, chan_num[0], 1)
        self.fusion2 = nn.Conv2d(chan_num[1] * 2, chan_num[1], 1)
        self.fusion3 = nn.Conv2d(chan_num[2] * 2, chan_num[2], 1)
        self.fusion4 = nn.Conv2d(chan_num[3] * 2, chan_num[3], 1)

        # Decoder
        self.up1 = UpBlock(chan_num[3], chan_num[4], conv_num[4], trans_num[4],
                           conv_block, heads=num_heads[4], dim_head=dim_head[4],
                           map_size=map_size, map_shortcut=True)
        self.up2 = UpBlock(chan_num[4], chan_num[5], conv_num[5], trans_num[5],
                           conv_block, heads=num_heads[5], dim_head=dim_head[5],
                           map_size=map_size, map_shortcut=True)
        self.up3 = UpBlock(chan_num[5], chan_num[6], conv_num[6], trans_num[6],
                           conv_block, map_shortcut=False)
        self.up4 = UpBlock(chan_num[6], chan_num[7], conv_num[7], trans_num[7],
                           conv_block, map_shortcut=False)

        self.outc = nn.Conv2d(chan_num[7], output_nc, 1)

        # Auxiliary classifiers for multi-scale output (apply to final decoder features)
        self.aux_cls2 = nn.Conv2d(chan_num[7], output_nc, 1)
        self.aux_cls3 = nn.Conv2d(chan_num[7], output_nc, 1)
        self.aux_cls4 = nn.Conv2d(chan_num[7], output_nc, 1)

    def forward(self, x1, x2):
        # Encoder stream 1
        x01 = self.inc(x1)
        x11, _ = self.down1(x01)
        x12, map12 = self.down2(x11, None)

        # Encoder stream 2
        x02 = self.inc(x2)
        x21, _ = self.down1(x02)
        x22, map22 = self.down2(x21, None)

        # Fuse bi-temporal features
        x0 = self.fusion0(torch.cat([x01, x02], 1))
        x1 = self.fusion1(torch.cat([x11, x21], 1))
        x2 = self.fusion2(torch.cat([x12, x22], 1))
        map2 = self.fusion2(torch.cat([map12, map22], 1))

        x3, map3 = self.down3(x2, map2)
        x4, map4 = self.down4(x3, map3)

        map_list = [map2, map3, map4]

        out, sem_map = self.up1(x4, x3, map_list[2], map_list[1])
        out, sem_map = self.up2(out, x2, sem_map, map_list[0])
        out, sem_map = self.up3(out, x1, sem_map, None)
        out, _ = self.up4(out, x0, sem_map, None)

        out_main = self.outc(out)

        # Multi-scale outputs
        m1 = torch.sigmoid(F.interpolate(out_main, scale_factor=2, mode='bilinear'))

        size = m1.shape[2:]
        aux2 = self.aux_cls2(out) if hasattr(self, 'aux_cls2') else out[:, :1]
        m2 = torch.sigmoid(F.interpolate(aux2, size=size, mode='bilinear'))
        aux3 = self.aux_cls3(out) if hasattr(self, 'aux_cls3') else out[:, :1]
        m3 = torch.sigmoid(F.interpolate(aux3, size=size, mode='bilinear'))
        aux4 = self.aux_cls4(out) if hasattr(self, 'aux_cls4') else out[:, :1]
        m4 = torch.sigmoid(F.interpolate(aux4, size=size, mode='bilinear'))

        return m1, m2, m3, m4


class BasicLayer(nn.Module):
    """A basic transformer layer."""
    def __init__(self, feat_dim, map_dim, out_dim, num_blocks, heads=4, dim_head=64,
                 expansion=1, attn_drop=0., proj_drop=0., map_size=8,
                 proj_type='depthwise', norm=nn.BatchNorm2d, act=nn.GELU):
        super().__init__()
        self.blocks = nn.ModuleList([])
        dim1 = feat_dim
        for _ in range(num_blocks):
            self.blocks.append(BidirectionAttentionBlock(
                dim1, map_dim, out_dim, heads, dim_head,
                expansion=expansion, attn_drop=attn_drop, proj_drop=proj_drop,
                map_size=map_size, proj_type=proj_type, norm=norm, act=act))
            dim1 = out_dim

    def forward(self, x, semantic_map):
        for block in self.blocks:
            x, semantic_map = block(x, semantic_map)
        return x, semantic_map


class PatchMerging(nn.Module):
    """Patch merging with optional semantic map projection."""
    def __init__(self, dim, out_dim, proj_type='depthwise', map_proj=True, norm=nn.BatchNorm2d):
        super().__init__()
        if proj_type == 'linear':
            self.reduction = nn.Conv2d(4 * dim, out_dim, 1, bias=False)
        else:
            self.reduction = DepthwiseSeparableConv(4 * dim, out_dim)
        self.norm = norm(4 * dim)
        self.map_projection = nn.Conv2d(dim, out_dim, 1, bias=False) if map_proj else None

    def forward(self, x, semantic_map=None):
        x0, x1 = x[:, :, 0::2, 0::2], x[:, :, 1::2, 0::2]
        x2, x3 = x[:, :, 0::2, 1::2], x[:, :, 1::2, 1::2]
        x = torch.cat([x0, x1, x2, x3], 1)
        x = self.reduction(self.norm(x))
        if semantic_map is not None and self.map_projection is not None:
            semantic_map = self.map_projection(semantic_map)
        return x, semantic_map


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, conv_num, trans_num, conv_block=BasicBlock,
                 heads=4, dim_head=64, expansion=4, attn_drop=0., proj_drop=0.,
                 map_size=8, proj_type='depthwise', norm=nn.BatchNorm2d, act=nn.GELU,
                 map_generate=False, map_proj=True, map_dim=None):
        super().__init__()
        map_dim = out_ch if map_dim is None else map_dim
        self.map_generate = map_generate
        if map_generate:
            self.map_gen = SemanticMapGeneration(out_ch, map_dim, map_size)
        self.patch_merging = PatchMerging(in_ch, out_ch, proj_type=proj_type,
                                           map_proj=map_proj, norm=norm)
        self.conv_blocks = nn.Sequential(*[conv_block(out_ch, out_ch, norm=norm, act=act)
                                           for _ in range(conv_num)])
        self.trans_blocks = BasicLayer(out_ch, map_dim, out_ch, trans_num,
                                        heads=heads, dim_head=dim_head,
                                        expansion=expansion, attn_drop=attn_drop,
                                        proj_drop=proj_drop, map_size=map_size,
                                        proj_type=proj_type, norm=norm, act=act)

    def forward(self, x, semantic_map=None):
        x, semantic_map = self.patch_merging(x, semantic_map)
        out = self.conv_blocks(x)
        if self.map_generate:
            semantic_map = self.map_gen(out)
        out, semantic_map = self.trans_blocks(out, semantic_map)
        return out, semantic_map


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, conv_num, trans_num, conv_block=BasicBlock,
                 heads=4, dim_head=64, expansion=1, attn_drop=0., proj_drop=0.,
                 map_size=8, proj_type='depthwise', norm=nn.BatchNorm2d, act=nn.GELU,
                 map_shortcut=False, map_dim=None):
        super().__init__()
        self.reduction = nn.Conv2d(in_ch + out_ch, out_ch, 1, bias=False)
        self.norm = norm(in_ch + out_ch)
        self.map_shortcut = map_shortcut
        map_dim = out_ch if map_dim is None else map_dim
        if map_shortcut:
            self.map_reduction = nn.Conv2d(in_ch + out_ch, map_dim, 1, bias=False)
        else:
            self.map_reduction = nn.Conv2d(in_ch, map_dim, 1, bias=False)
        self.trans_blocks = BasicLayer(out_ch, map_dim, out_ch, trans_num,
                                        heads=heads, dim_head=dim_head,
                                        expansion=expansion, attn_drop=attn_drop,
                                        proj_drop=proj_drop, map_size=map_size,
                                        proj_type=proj_type, norm=norm, act=act)
        self.conv_blocks = nn.Sequential(*[conv_block(out_ch, out_ch, norm=norm, act=act)
                                           for _ in range(conv_num)])

    def forward(self, x1, x2, map1, map2=None):
        x1 = F.interpolate(x1, size=x2.shape[-2:], mode='bilinear', align_corners=True)
        feat = torch.cat([x1, x2], 1)
        out = self.reduction(self.norm(feat))
        if self.map_shortcut and map2 is not None:
            semantic_map = self.map_reduction(torch.cat([map1, map2], 1))
        else:
            semantic_map = self.map_reduction(map1)
        out, semantic_map = self.trans_blocks(out, semantic_map)
        out = self.conv_blocks(out)
        return out, semantic_map
