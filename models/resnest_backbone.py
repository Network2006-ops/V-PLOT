"""
resnest_backbone.py
--------------------
ResNeSt backbone with Split-Attention blocks used to extract rich
hierarchical, multi-scale structural features from keyframes (manuscript
Section: ResNeSt backbone, Eq. 5):

    Z = sum_{k=1..K} a_k * U_k

where the input feature map is split into K cardinal groups U_k and
aggregated using softmax-normalized attention weights a_k.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SplitAttentionConv2d(nn.Module):
    """
    Split-Attention block (ResNeSt core module). The input channels are
    divided into `radix` groups (cardinal splits); each group is convolved
    independently, a channel-wise attention vector is learned per group via
    global pooling + a small FC bottleneck, and the groups are aggregated
    with softmax-normalized attention weights (Eq. 5).
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, radix=2, cardinality=1, reduction_factor=4):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality
        self.out_channels = out_channels

        inter_channels = max(out_channels * radix // reduction_factor, 32)

        self.conv = nn.Conv2d(
            in_channels, out_channels * radix, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=cardinality * radix, bias=False
        )
        self.bn0 = nn.BatchNorm2d(out_channels * radix)
        self.relu = nn.ReLU(inplace=True)

        self.fc1 = nn.Conv2d(out_channels, inter_channels, kernel_size=1, groups=cardinality)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.fc2 = nn.Conv2d(inter_channels, out_channels * radix, kernel_size=1, groups=cardinality)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn0(x)
        x = self.relu(x)

        batch, rchannel = x.shape[0], x.shape[1]
        if self.radix > 1:
            splits = torch.split(x, rchannel // self.radix, dim=1)   # U_1..U_K
            gap = sum(splits)                                        # elementwise sum for pooling context
        else:
            gap = x
            splits = [x]

        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.relu(self.bn1(self.fc1(gap)))
        attn = self.fc2(gap)                                          # (B, out_channels*radix, 1, 1)
        attn = attn.view(batch, self.radix, self.out_channels, 1, 1)
        attn = F.softmax(attn, dim=1)                                 # a_k = softmax(.) per Eq. 5

        out = sum(attn[:, i] * splits[i] for i in range(self.radix))  # Z = sum a_k * U_k
        return out.contiguous()


class SplitAttentionBottleneck(nn.Module):
    """Residual bottleneck block using SplitAttentionConv2d as its 3x3 stage,
    analogous to the standard ResNet bottleneck but with split-attention
    replacing the plain convolution."""

    expansion = 4

    def __init__(self, in_channels, channels, stride=1, radix=2, cardinality=1,
                 downsample=None):
        super().__init__()
        out_channels = channels * self.expansion

        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.split_attn = SplitAttentionConv2d(
            channels, channels, kernel_size=3, stride=stride, padding=1,
            radix=radix, cardinality=cardinality
        )

        self.conv3 = nn.Conv2d(channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.split_attn(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


class ResNeStBackbone(nn.Module):
    """
    ResNeSt-style backbone producing a multi-stage feature pyramid
    (C2, C3, C4, C5) for downstream DDConv / RPN / ROIAlign stages.
    """

    def __init__(self, layers=(3, 4, 6, 3), widths=(64, 128, 256, 512),
                 radix=2, cardinality=1, in_channels=3):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality
        self.in_planes = 64

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self._make_layer(widths[0], layers[0], stride=1)
        self.layer2 = self._make_layer(widths[1], layers[1], stride=2)
        self.layer3 = self._make_layer(widths[2], layers[2], stride=2)
        self.layer4 = self._make_layer(widths[3], layers[3], stride=2)

        self.out_channels = {
            "C2": widths[0] * SplitAttentionBottleneck.expansion,
            "C3": widths[1] * SplitAttentionBottleneck.expansion,
            "C4": widths[2] * SplitAttentionBottleneck.expansion,
            "C5": widths[3] * SplitAttentionBottleneck.expansion,
        }

    def _make_layer(self, channels, blocks, stride):
        downsample = None
        out_channels = channels * SplitAttentionBottleneck.expansion
        if stride != 1 or self.in_planes != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = [SplitAttentionBottleneck(self.in_planes, channels, stride=stride,
                                            radix=self.radix, cardinality=self.cardinality,
                                            downsample=downsample)]
        self.in_planes = out_channels
        for _ in range(1, blocks):
            layers.append(SplitAttentionBottleneck(self.in_planes, channels,
                                                     radix=self.radix,
                                                     cardinality=self.cardinality))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return {"C2": c2, "C3": c3, "C4": c4, "C5": c5}


def resnest50(**kwargs):
    """ResNeSt-50-style backbone (3,4,6,3 blocks) used in the V-PLOT DD-RCNN."""
    return ResNeStBackbone(layers=(3, 4, 6, 3), **kwargs)


if __name__ == "__main__":
    model = resnest50()
    dummy = torch.randn(2, 3, 512, 512)  # batch > 1 required by BatchNorm during training-mode smoke test
    feats = model(dummy)
    for name, f in feats.items():
        print(name, f.shape)
