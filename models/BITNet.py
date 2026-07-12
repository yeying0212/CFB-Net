import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from . import ResNet


# ==================== Transformer Components ====================

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
                    nn.Linear(dim, mlp_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(mlp_dim, dim),
                    nn.Dropout(dropout),
                )
            ]))

    def forward(self, x):
        for norm1, attn, norm2, mlp in self.layers:
            xn = norm1(x)
            x = x + attn(xn, xn, xn)[0]
            x = x + mlp(norm2(x))
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., softmax=True):
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
                    nn.Linear(dim, mlp_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(mlp_dim, dim),
                    nn.Dropout(dropout),
                )
            ]))
        self.softmax = softmax

    def forward(self, x, m):
        for norm1, self_attn, norm2, cross_attn, norm3, mlp in self.layers:
            xn = norm1(x)
            x = x + self_attn(xn, xn, xn)[0]
            xn = norm2(x)
            x = x + cross_attn(xn, m, m)[0]
            x = x + mlp(norm3(x))
        return x


# ==================== BITNet ====================

class BITNet(nn.Module):
    """BIT_CD for CFB-Net framework.

    BIT: Remote Sensing Image Change Detection with Transformers (TGRS 2021)
    https://github.com/justchenhao/BIT_CD

    Uses ResNet18 backbone with BIT (Bitemporal Image Transformer).
    """
    def __init__(self, input_nc=3, output_nc=1, backbone='resnet18',
                 token_len=4, enc_depth=1, dec_depth=8,
                 dim_head=64, decoder_dim_head=64):
        super().__init__()
        # Backbone: ResNet18 with dilation on layer3+4 (output stride=8)
        if backbone == 'resnet18':
            self.backbone = ResNet.resnet18(pretrained=True)
            resnet_channels = self.backbone.channels  # [64, 64, 128, 256, 512]
        elif backbone == 'resnet50':
            self.backbone = ResNet.resnet50(pretrained=True)
            resnet_channels = self.backbone.channels  # [64, 256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.resnet_stages_num = 4  # use layer1~layer4 features
        self.token_len = token_len

        # Projection from ResNet output to transformer dim
        last_channel = resnet_channels[self.resnet_stages_num]  # 512 for r18
        proj_dim = 32
        self.conv_pred = nn.Conv2d(last_channel, proj_dim, kernel_size=3, padding=1)

        # Semantic tokenizer
        self.conv_a = nn.Conv2d(proj_dim, token_len, kernel_size=1, bias=False)

        # BIT transformer
        bit_dim = proj_dim
        mlp_dim = 2 * bit_dim
        self.pos_embedding = nn.Parameter(torch.randn(1, token_len * 2, bit_dim))
        self.transformer = TransformerEncoder(
            dim=bit_dim, depth=enc_depth, heads=8, dim_head=dim_head, mlp_dim=mlp_dim)
        self.transformer_decoder = TransformerDecoder(
            dim=bit_dim, depth=dec_depth, heads=8, dim_head=decoder_dim_head,
            mlp_dim=mlp_dim, softmax=True)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Conv2d(bit_dim, bit_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(bit_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(bit_dim, output_nc, kernel_size=3, padding=1),
        )

        # Upsample
        self.upsample_x2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_x4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        # Auxiliary heads for multi-scale outputs
        self.aux_head2 = nn.Conv2d(resnet_channels[2], output_nc, kernel_size=1)  # 128ch
        self.aux_head3 = nn.Conv2d(resnet_channels[3], output_nc, kernel_size=1)  # 256ch
        self.aux_head4 = nn.Conv2d(resnet_channels[4], output_nc, kernel_size=1)  # 512ch

    def _forward_single(self, x):
        """Extract ResNet features and project to 32-dim."""
        _, c1, c2, c3, c4 = self.backbone.base_forward(x)
        # Use c4 (layer4 output) as the feature map
        x = self.conv_pred(c4)
        x = self.upsample_x2(x)
        return x, (c1, c2, c3, c4)

    def _forward_semantic_tokens(self, x):
        b, c, h, w = x.shape
        spatial_attention = self.conv_a(x)
        spatial_attention = spatial_attention.view(b, self.token_len, -1)
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view(b, c, -1)
        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x)
        return tokens

    def _forward_transformer_decoder(self, x, m):
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer_decoder(x, m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return x

    def forward(self, x1, x2):
        # Extract features from both images
        f1, feats1 = self._forward_single(x1)
        f2, feats2 = self._forward_single(x2)

        # Semantic tokenization
        token1 = self._forward_semantic_tokens(f1)
        token2 = self._forward_semantic_tokens(f2)

        # BIT transformer
        tokens = torch.cat([token1, token2], dim=1)
        tokens = tokens + self.pos_embedding
        tokens = self.transformer(tokens)
        token1, token2 = tokens.chunk(2, dim=1)

        # Transformer decoder on feature maps
        f1 = self._forward_transformer_decoder(f1, token1)
        f2 = self._forward_transformer_decoder(f2, token2)

        # Difference + classify (main output)
        x = torch.abs(f1 - f2)
        x = self.upsample_x4(x)
        main_out = self.classifier(x)

        m1 = torch.sigmoid(main_out)

        # Auxiliary multi-scale outputs
        size = m1.shape[2:]
        c1, c2, c3, c4 = feats1
        d1, d2, d3, d4 = feats2

        aux2 = torch.abs(self.upsample_x4(c2) - self.upsample_x4(d2))
        m2 = torch.sigmoid(self.aux_head2(aux2))
        m2 = F.interpolate(m2, size=size, mode='bilinear', align_corners=True)

        aux3 = torch.abs(self.upsample_x4(c3) - self.upsample_x4(d3))
        m3 = torch.sigmoid(self.aux_head3(aux3))
        m3 = F.interpolate(m3, size=size, mode='bilinear', align_corners=True)

        aux4 = torch.abs(c4 - d4)
        aux4 = self.upsample_x4(aux4)
        m4 = torch.sigmoid(self.aux_head4(aux4))
        m4 = F.interpolate(m4, size=size, mode='bilinear', align_corners=True)

        return m1, m2, m3, m4
