import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url


model_urls = {
    "shufflenetv2_0.5x": "https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth",
    "shufflenetv2_1.0x": "https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth",
    "shufflenetv2_1.5x": None,
    "shufflenetv2_2.0x": None,
}


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x


class ShuffleV2Block(nn.Module):
    def __init__(self, inp, oup, stride):
        super().__init__()
        if not (1 <= stride <= 3):
            raise ValueError("illegal stride value")
        self.stride = stride
        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3,
                                stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        out = channel_shuffle(out, 2)
        return out


class ShuffleNetV2(nn.Module):
    """Multi-scale ShuffleNetV2 backbone for change detection.

    Returns 5 feature maps at different resolutions.
    Supports model sizes: 0.5x, 1.0x, 1.5x, 2.0x.

    Channels by model size:
    - 0.5x: [24, 24, 48, 96, 192]
    - 1.0x: [24, 24, 116, 232, 464]
    - 1.5x: [24, 24, 176, 352, 704]
    - 2.0x: [24, 24, 244, 488, 976]
    """

    def __init__(self, model_size="1.0x", pretrain=True):
        super().__init__()
        self.model_size = model_size
        self.stage_repeats = [4, 8, 4]

        if model_size == "0.5x":
            self._stage_out_channels = [24, 48, 96, 192, 1024]
        elif model_size == "1.0x":
            self._stage_out_channels = [24, 116, 232, 464, 1024]
        elif model_size == "1.5x":
            self._stage_out_channels = [24, 176, 352, 704, 1024]
        elif model_size == "2.0x":
            self._stage_out_channels = [24, 244, 488, 976, 2048]
        else:
            raise NotImplementedError

        # Channels returned: [conv1_out, maxpool_out, stage2, stage3, stage4]
        # We list them explicitly for the change detection models
        if model_size == "0.5x":
            self.channels = [24, 24, 48, 96, 192]
        elif model_size == "1.0x":
            self.channels = [24, 24, 116, 232, 464]
        elif model_size == "1.5x":
            self.channels = [24, 24, 176, 352, 704]
        elif model_size == "2.0x":
            self.channels = [24, 24, 244, 488, 976]

        # first conv
        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # stages 2, 3, 4
        for name, repeats, output_channels in zip(
            ["stage2", "stage3", "stage4"],
            self.stage_repeats,
            self._stage_out_channels[1:],
        ):
            seq = [ShuffleV2Block(input_channels, output_channels, 2)]
            for _ in range(repeats - 1):
                seq.append(ShuffleV2Block(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        self._initialize_weights(pretrain)

    def forward(self, x):
        """Return 5 multi-scale features: c0, c1, c2, c3, c4."""
        c0 = self.conv1(x)           # 1/2
        c1 = self.maxpool(c0)        # 1/4
        c2 = self.stage2(c1)         # 1/8
        c3 = self.stage3(c2)         # 1/16
        c4 = self.stage4(c3)         # 1/32
        return c0, c1, c2, c3, c4

    def _initialize_weights(self, pretrain=True):
        print("Initializing ShuffleNetV2 weights...")
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if "first" in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
        if pretrain:
            url = model_urls[f"shufflenetv2_{self.model_size}"]
            if url is not None:
                pretrained_state_dict = load_state_dict_from_url(url)
                print(f"=> loading pretrained ShuffleNetV2 {self.model_size}")
                self.load_state_dict(pretrained_state_dict, strict=False)


if __name__ == '__main__':
    model = ShuffleNetV2(model_size='1.0x')
    x = torch.randn(1, 3, 256, 256)
    feats = model(x)
    for i, f in enumerate(feats):
        print(f"c{i}: {f.shape}")
