import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
from . import ResNet


# ==================== HFAB ====================

class HFAB(nn.Module):
    """Hybrid Feature Attention Block — channel + spatial attention.

    From EGENet: edge_block.py attention_block.
    """
    def __init__(self, input_channel, input_size, ratio=0.5):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(input_channel, max(1, int(input_channel * ratio)), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, int(input_channel * ratio)), input_channel, 1),
            nn.Sigmoid())
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(input_channel, 1, 3, 1, 1), nn.Sigmoid())
        self.conv_out = nn.Sequential(
            nn.Conv2d(input_channel, input_channel, 3, 1, 1),
            nn.BatchNorm2d(input_channel), nn.ReLU(inplace=True))

    def forward(self, x):
        ca = self.channel_attention(x)
        sa = self.spatial_attention(x)
        x = x * ca * sa
        return self.conv_out(x) + x


# ==================== Edge Encoder / Decoder ====================

class Mlp(nn.Module):
    """MLP with DWConv, from EGENet edge_block."""
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


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


class EdgeEncoder(nn.Module):
    """Multi-scale edge encoder: MLP each level + upsample + fuse.

    Exactly follows EGENet's Edge_Encoder.
    """
    def __init__(self):
        super().__init__()
        self.mlp_c0 = Mlp(64, 128, 64)
        self.mlp_c1 = Mlp(64, 128, 64)
        self.mlp_c2 = Mlp(128, 256, 128)
        self.mlp_c3 = Mlp(256, 512, 256)

        self.up_sample_c1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.up_sample_c2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=4)
        self.up_sample_c3 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=4)

        self.linear_fuse = nn.Conv2d(64 * 4, 32, kernel_size=1)

    def forward(self, c0, c1, c2, c3):
        B, C, H, W = c0.shape
        c0 = c0.view(B, C, -1).transpose(1, 2)
        c0 = self.mlp_c0(c0, H, W)
        c0 = c0.transpose(1, 2).view(B, C, H, W)

        B, C, H, W = c1.shape
        c1 = c1.view(B, C, -1).transpose(1, 2)
        c1 = self.mlp_c1(c1, H, W)
        c1 = c1.transpose(1, 2).view(B, C, H, W)

        B, C, H, W = c2.shape
        c2 = c2.view(B, C, -1).transpose(1, 2)
        c2 = self.mlp_c2(c2, H, W)
        c2 = c2.transpose(1, 2).view(B, C, H, W)

        B, C, H, W = c3.shape
        c3 = c3.view(B, C, -1).transpose(1, 2)
        c3 = self.mlp_c3(c3, H, W)
        c3 = c3.transpose(1, 2).view(B, C, H, W)

        c1_up = self.up_sample_c1(c1)
        c2_up = self.up_sample_c2(c2)
        c3_up = self.up_sample_c3(c3)

        return self.linear_fuse(torch.cat([c3_up, c2_up, c1_up, c0], dim=1))


class EdgeDecoder(nn.Module):
    """Edge decoder: MLP → upsample → classify.

    Exactly follows EGENet's Edge_Decoder.
    """
    def __init__(self, in_channels=32):
        super().__init__()
        self.linear = Mlp(in_channels, 256, in_channels * 8)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, -1).transpose(1, 2)
        x = self.linear(x, H, W)
        x = x.transpose(1, 2).view(B, C * 8, H, W)
        x = self.upsample(x)
        return x


# ==================== BIT Transformer ====================

class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim, eps=1e-5),
                nn.MultiheadAttention(dim, heads, dropout, batch_first=True),
                nn.LayerNorm(dim, eps=1e-5),
                nn.Sequential(
                    nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(mlp_dim, dim), nn.Dropout(dropout)),
            ]))

    def forward(self, x):
        for norm1, attn, norm2, mlp in self.layers:
            xn = norm1(x)
            x = x + attn(xn, xn, xn)[0]
            x = x + mlp(norm2(x))
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim, eps=1e-5),
                nn.MultiheadAttention(dim, heads, dropout, batch_first=True),
                nn.LayerNorm(dim, eps=1e-5),
                nn.MultiheadAttention(dim, heads, dropout, batch_first=True),
                nn.LayerNorm(dim, eps=1e-5),
                nn.Sequential(
                    nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(mlp_dim, dim), nn.Dropout(dropout)),
            ]))

    def forward(self, x, m):
        for norm1, self_attn, norm2, cross_attn, norm3, mlp in self.layers:
            xn = norm1(x)
            x = x + self_attn(xn, xn, xn)[0]
            xn = norm2(x)
            x = x + cross_attn(xn, m, m)[0]
            x = x + mlp(norm3(x))
        return x


# ==================== Main EGENet ====================

class EGENet(nn.Module):
    """EGENet for CFB-Net framework.

    EGENet: Edge-Guided Enhancement Network for Change Detection (IGARSS 2024)
    https://github.com/Jnmz/EGENet-IG24

    Full implementation with:
    - ResNet18 backbone
    - HFAB multi-scale feature refinement
    - Edge_Encoder + Edge_Decoder for edge detection
    - Dual BIT transformer decoders (main + edge branch)
    """
    def __init__(self, input_nc=3, output_nc=1, backbone='resnet18',
                 token_len=4, enc_depth=1, dec_depth=8,
                 dim_head=64, decoder_dim_head=64):
        super().__init__()
        if backbone == 'resnet18':
            self.backbone = ResNet.resnet18(pretrained=True)
            resnet_channels = self.backbone.channels  # [64, 64, 128, 256, 512]
        elif backbone == 'resnet50':
            self.backbone = ResNet.resnet50(pretrained=True)
            resnet_channels = self.backbone.channels
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        proj_dim = 32
        self.token_len = token_len

        # Projection: 512 → 32 (for resnet18)
        self.conv_pred = nn.Conv2d(resnet_channels[4], proj_dim, kernel_size=3, padding=1)

        # HFAB for multi-scale features (on c1,c2,c3,c4)
        self.hfab1 = HFAB(resnet_channels[1], 128)
        self.hfab2 = HFAB(resnet_channels[2], 64)
        self.hfab3 = HFAB(resnet_channels[3], 32)
        self.hfab4 = HFAB(resnet_channels[4], 32)

        # Edge encoder/decoder (from original EGENet)
        self.edge_encoder = EdgeEncoder()
        self.edge_decoder = EdgeDecoder(in_channels=proj_dim)
        self.HF = HFAB(proj_dim, 128, ratio=0.5)       # HFAB for edge features (before decoder)
        self.HF_ = HFAB(proj_dim, 128, ratio=0.5)      # HFAB for edge features (after transformer)
        self.HF__ = HFAB(256, 256, ratio=0.5)            # HFAB for edge final (256ch)
        self.edge_classifier = nn.Conv2d(512, output_nc, 3, 1, 1)

        # BIT transformer
        bit_dim = proj_dim
        mlp_dim = 2 * bit_dim
        self.conv_a = nn.Conv2d(proj_dim, token_len, 1, bias=False)
        self.pos_embedding = nn.Parameter(torch.randn(1, token_len * 2, bit_dim))
        self.pos_embedding_decoder = nn.Parameter(torch.randn(1, bit_dim, 64, 64))

        self.transformer = TransformerEncoder(bit_dim, enc_depth, 8, dim_head, mlp_dim)

        # Main branch transformer decoder
        self.transformer_decoder = TransformerDecoder(
            bit_dim, dec_depth, 8, decoder_dim_head, mlp_dim)

        # Edge branch transformer decoder (THE MISSING PIECE)
        self.transformer_decoder_edge = TransformerDecoder(
            bit_dim, dec_depth, 8, decoder_dim_head, mlp_dim)

        # Output heads
        self.classifier = nn.Sequential(
            nn.Conv2d(bit_dim * 2, proj_dim, 3, 1, 1),
            nn.BatchNorm2d(proj_dim), nn.ReLU(inplace=True),
            nn.Conv2d(proj_dim, output_nc, 3, 1, 1))

        self.upsample_x2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def _forward_single(self, x):
        """Extract ResNet features, apply HFAB, project deep features."""
        c0, c1, c2, c3, c4 = self.backbone.base_forward(x)
        # Apply HFAB at each level
        hf1 = self.hfab1(c1)
        hf2 = self.hfab2(c2)
        hf3 = self.hfab3(c3)
        hf4 = self.hfab4(c4)
        # Main feature from deepest level
        main_feat = self.conv_pred(c4)
        main_feat = self.upsample_x2(main_feat)
        return main_feat, (c0, c1, c2, c3, hf1, hf2, hf3, hf4)

    def _forward_semantic_tokens(self, x):
        b, c, h, w = x.shape
        sa = self.conv_a(x).view(b, self.token_len, -1)
        sa = torch.softmax(sa, dim=-1)
        x = x.view(b, c, -1)
        return torch.einsum('bln,bcn->blc', sa, x)

    def _forward_transformer_decoder(self, x, m, decoder):
        b, c, h, w = x.shape
        pe = F.interpolate(self.pos_embedding_decoder, size=(h, w), mode='bilinear', align_corners=True) \
            if self.pos_embedding_decoder.shape[-2:] != (h, w) else self.pos_embedding_decoder
        x = x + pe
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = decoder(x, m)
        return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

    def forward(self, x1, x2):
        H, W = x1.size(2), x1.size(3)

        # Extract features
        f1, (c0_1, c1_1, c2_1, c3_1, hf1_1, hf2_1, hf3_1, hf4_1) = self._forward_single(x1)
        f2, (c0_2, c1_2, c2_2, c3_2, hf1_2, hf2_2, hf3_2, hf4_2) = self._forward_single(x2)

        # === Edge branch ===
        # Edge encoder: MLP each level + fuse (uses raw ResNet features)
        edge1 = self.edge_encoder(c0_1, c1_1, c2_1, c3_1)
        edge2 = self.edge_encoder(c0_2, c1_2, c2_2, c3_2)

        edge1 = self.HF(edge1)
        edge2 = self.HF(edge2)

        # Edge decoder: MLP + upsample
        edge_final1 = self.edge_decoder(edge1)
        edge_final2 = self.edge_decoder(edge2)
        edge_final1 = self.HF__(edge_final1)
        edge_final2 = self.HF__(edge_final2)
        edge_map = self.edge_classifier(torch.cat([edge_final1, edge_final2], dim=1))
        edge_map = F.interpolate(edge_map, size=(H, W), mode='bilinear', align_corners=True)

        # === BIT transformer ===
        token1 = self._forward_semantic_tokens(f1)
        token2 = self._forward_semantic_tokens(f2)
        tokens = torch.cat([token1, token2], 1) + self.pos_embedding
        tokens = self.transformer(tokens)
        token1, token2 = tokens.chunk(2, 1)

        # Main branch: transformer decode
        f1 = self._forward_transformer_decoder(f1, token1, self.transformer_decoder)
        f2 = self._forward_transformer_decoder(f2, token2, self.transformer_decoder)

        # Edge branch: transformer decode (THE FIX)
        x_edge1 = self._forward_transformer_decoder(edge1, token1, self.transformer_decoder_edge)
        x_edge2 = self._forward_transformer_decoder(edge2, token2, self.transformer_decoder_edge)

        x_edge1 = self.HF_(x_edge1)
        x_edge2 = self.HF_(x_edge2)

        # Fuse main + edge features
        f1 = self.upsample_x2(f1)
        f2 = self.upsample_x2(f2)
        f1 = torch.cat([f1, x_edge1], dim=1)
        f2 = torch.cat([f2, x_edge2], dim=1)

        # Difference + classify
        x = torch.abs(f1 - f2)
        x = self.upsample_x2(x)
        main_out = self.classifier(x)
        main_out = F.interpolate(main_out, size=(H, W), mode='bilinear', align_corners=True)

        m1 = torch.sigmoid(main_out)
        m2 = torch.sigmoid(edge_map)

        # Auxiliary multi-scale outputs
        diff_c2 = torch.abs(hf2_1 - hf2_2)
        m3 = torch.sigmoid(F.interpolate(
            nn.Conv2d(diff_c2.size(1), 1, 1, device=diff_c2.device)(diff_c2),
            size=(H, W), mode='bilinear', align_corners=True))

        diff_c3 = torch.abs(hf3_1 - hf3_2)
        m4 = torch.sigmoid(F.interpolate(
            nn.Conv2d(diff_c3.size(1), 1, 1, device=diff_c3.device)(diff_c3),
            size=(H, W), mode='bilinear', align_corners=True))

        return m1, m2, m3, m4
