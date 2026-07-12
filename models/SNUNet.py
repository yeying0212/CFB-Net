import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.padding import ReplicationPad2d


class ConvBlock(nn.Module):
    """Conv + BN + ReLU + Dropout block."""
    def __init__(self, in_ch, out_ch, dropout=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
        )

    def forward(self, x):
        return self.conv(x)


class ConvBlock2(nn.Module):
    """Double conv block."""
    def __init__(self, in_ch, out_ch, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True), nn.Dropout2d(p=dropout))
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True), nn.Dropout2d(p=dropout))

    def forward(self, x):
        return self.conv2(self.conv1(x))


class NestedUNetEncoder(nn.Module):
    """NestedUNet encoder — single stream."""
    def __init__(self, input_nc, filters):
        super().__init__()
        self.conv11 = ConvBlock2(input_nc, filters[0])
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv21 = ConvBlock2(filters[0], filters[1])
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv31 = ConvBlock2(filters[1], filters[2])
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv41 = ConvBlock2(filters[2], filters[3])
        self.pool4 = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x1 = self.conv11(x)
        x2 = self.conv21(self.pool1(x1))
        x3 = self.conv31(self.pool2(x2))
        x4 = self.conv41(self.pool3(x3))
        x5 = self.pool4(x4)
        return x1, x2, x3, x4, x5


class NestedUNetDecoder(nn.Module):
    """NestedUNet decoder with dense skip connections."""
    def __init__(self, filters, output_nc):
        super().__init__()
        n1 = filters[0]
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.upconv4 = nn.ConvTranspose2d(filters[3], filters[3], 3, padding=1, stride=2, output_padding=1)
        self.conv43d = nn.Conv2d(filters[4], filters[3], 3, padding=1)
        self.bn43d = nn.BatchNorm2d(filters[3])
        self.conv42d = nn.Conv2d(filters[3], filters[3], 3, padding=1)
        self.bn42d = nn.BatchNorm2d(filters[3])
        self.conv41d = nn.Conv2d(filters[3], filters[2], 3, padding=1)
        self.bn41d = nn.BatchNorm2d(filters[2])

        self.upconv3 = nn.ConvTranspose2d(filters[2], filters[2], 3, padding=1, stride=2, output_padding=1)
        self.conv33d = nn.Conv2d(filters[3], filters[2], 3, padding=1)
        self.bn33d = nn.BatchNorm2d(filters[2])
        self.conv32d = nn.Conv2d(filters[2], filters[2], 3, padding=1)
        self.bn32d = nn.BatchNorm2d(filters[2])
        self.conv31d = nn.Conv2d(filters[2], filters[1], 3, padding=1)
        self.bn31d = nn.BatchNorm2d(filters[1])

        self.upconv2 = nn.ConvTranspose2d(filters[1], filters[1], 3, padding=1, stride=2, output_padding=1)
        self.conv22d = nn.Conv2d(filters[2], filters[1], 3, padding=1)
        self.bn22d = nn.BatchNorm2d(filters[1])
        self.conv21d = nn.Conv2d(filters[1], filters[0], 3, padding=1)
        self.bn21d = nn.BatchNorm2d(filters[0])

        self.upconv1 = nn.ConvTranspose2d(filters[0], filters[0], 3, padding=1, stride=2, output_padding=1)
        self.conv12d = nn.Conv2d(filters[1], filters[0], 3, padding=1)
        self.bn12d = nn.BatchNorm2d(filters[0])
        self.conv11d = nn.Conv2d(filters[0], output_nc, 3, padding=1)

        self.drop = nn.Dropout2d(p=0.2)

    def forward(self, x1_1, x2_1, x3_1, x4_1, x5_1, x1_2, x2_2, x3_2, x4_2, x5_2):
        # Stage 4d
        x4d = self.upconv4(x5_1)
        pad4 = ReplicationPad2d((0, x4_1.size(3) - x4d.size(3), 0, x4_1.size(2) - x4d.size(2)))
        x4d = torch.cat((pad4(x4d), torch.abs(x4_1 - x4_2)), 1)
        x43d = self.drop(F.relu(self.bn43d(self.conv43d(x4d))))
        x42d = self.drop(F.relu(self.bn42d(self.conv42d(x43d))))
        x41d = self.drop(F.relu(self.bn41d(self.conv41d(x42d))))

        # Stage 3d
        x3d = self.upconv3(x41d)
        pad3 = ReplicationPad2d((0, x3_1.size(3) - x3d.size(3), 0, x3_1.size(2) - x3d.size(2)))
        x3d = torch.cat((pad3(x3d), torch.abs(x3_1 - x3_2)), 1)
        x33d = self.drop(F.relu(self.bn33d(self.conv33d(x3d))))
        x32d = self.drop(F.relu(self.bn32d(self.conv32d(x33d))))
        x31d = self.drop(F.relu(self.bn31d(self.conv31d(x32d))))

        # Stage 2d
        x2d = self.upconv2(x31d)
        pad2 = ReplicationPad2d((0, x2_1.size(3) - x2d.size(3), 0, x2_1.size(2) - x2d.size(2)))
        x2d = torch.cat((pad2(x2d), torch.abs(x2_1 - x2_2)), 1)
        x22d = self.drop(F.relu(self.bn22d(self.conv22d(x2d))))
        x21d = self.drop(F.relu(self.bn21d(self.conv21d(x22d))))

        # Stage 1d
        x1d = self.upconv1(x21d)
        pad1 = ReplicationPad2d((0, x1_1.size(3) - x1d.size(3), 0, x1_1.size(2) - x1d.size(2)))
        x1d = torch.cat((pad1(x1d), torch.abs(x1_1 - x1_2)), 1)
        x12d = self.drop(F.relu(self.bn12d(self.conv12d(x1d))))
        x11d = self.conv11d(x12d)

        # Generate multi-scale outputs for framework compatibility
        out_main = x11d
        # Use aux classifiers to project intermediate features to 1-channel
        out_p2 = torch.sigmoid(F.interpolate(
            self.aux_cls2(x22d) if hasattr(self, 'aux_cls2') else
            nn.Conv2d(x22d.size(1), 1, 1, device=x22d.device)(x22d),
            scale_factor=4, mode='bilinear'))
        out_p3 = torch.sigmoid(F.interpolate(
            self.aux_cls3(x32d) if hasattr(self, 'aux_cls3') else
            nn.Conv2d(x32d.size(1), 1, 1, device=x32d.device)(x32d),
            scale_factor=8, mode='bilinear'))
        out_p4 = torch.sigmoid(F.interpolate(
            self.aux_cls4(x42d) if hasattr(self, 'aux_cls4') else
            nn.Conv2d(x42d.size(1), 1, 1, device=x42d.device)(x42d),
            scale_factor=16, mode='bilinear'))

        return out_main, out_p2, out_p3, out_p4


class SNUNet(nn.Module):
    """SNUNet-CD adapted for CFB-Net framework.

    SNUNet-CD: A Densely Connected Siamese Network for Change Detection of VHR Images (GRSL 2021)
    https://github.com/likyoo/Siam-NestedUNet
    """
    def __init__(self, input_nc=3, output_nc=1):
        super().__init__()
        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        self.encoder = NestedUNetEncoder(input_nc, filters)
        self.decoder = NestedUNetDecoder(filters, output_nc)

        # Auxiliary classifiers for multi-scale output compatibility
        self.aux_cls1 = nn.Conv2d(n1, 1, 1)
        self.aux_cls2 = nn.Conv2d(n1 * 2, 1, 1)
        self.aux_cls3 = nn.Conv2d(n1 * 4, 1, 1)

    def forward(self, x1, x2):
        x1_1, x1_2, x1_3, x1_4, x1_5 = self.encoder(x1)
        x2_1, x2_2, x2_3, x2_4, x2_5 = self.encoder(x2)

        out_main, out_d2, out_d3, out_d4 = self.decoder(
            x1_1, x1_2, x1_3, x1_4, x1_5,
            x2_1, x2_2, x2_3, x2_4, x2_5)

        # Generate 4-scale output (decoder already produces 1ch sigmoid aux outputs)
        m1 = torch.sigmoid(out_main)
        m2 = F.interpolate(out_d2, size=m1.shape[2:], mode='bilinear')
        m3 = F.interpolate(out_d3, size=m1.shape[2:], mode='bilinear')
        m4 = F.interpolate(out_d4, size=m1.shape[2:], mode='bilinear')

        return m1, m2, m3, m4
