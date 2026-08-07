"""
ddconv.py
---------
Dilated-Deformable Convolution (DDConv) module (manuscript Section:
DD Convolution Module, Eqs. 6-7).

Pipeline: 3x3 standard conv -> max-pool -> dilated conv (Eq. 6, enlarges the
receptive field) -> deformable conv with learnable offsets (Eq. 7, adapts
spatial sampling locations to irregular graphical structures).

Eq. 6 (dilated convolution):
    Y(p) = sum_k W_k * X(p + p_k * d)

Eq. 7 (dilated-deformable convolution):
    Y(p) = sum_k W_k * X(p + p_k * d + delta_p_k)

where delta_p_k are learnable offsets predicted by an auxiliary conv layer.
"""

import torch
import torch.nn as nn

try:
    from torchvision.ops import deform_conv2d
    _HAS_DEFORM = True
except ImportError:
    _HAS_DEFORM = False


class DeformableDilatedConv2d(nn.Module):
    """
    Combines dilation (enlarged receptive field, Eq. 6) with deformable
    sampling offsets (adaptive spatial sampling, Eq. 7) in a single
    convolutional layer, using torchvision's deform_conv2d kernel.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, dilation=2, groups=1):
        super().__init__()
        if not _HAS_DEFORM:
            raise ImportError("torchvision.ops.deform_conv2d is required for DDConv. "
                               "Install torchvision>=0.9.")

        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = dilation * (kernel_size - 1) // 2

        # Learnable offsets delta_p_k: 2 values (dx, dy) per kernel sampling point.
        offset_channels = 2 * kernel_size * kernel_size
        self.offset_conv = nn.Conv2d(
            in_channels, offset_channels, kernel_size=kernel_size,
            stride=stride, padding=self.padding, dilation=dilation
        )
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)

        # Learnable per-sample modulation mask (modulated deformable conv, DCNv2-style).
        mask_channels = kernel_size * kernel_size
        self.mask_conv = nn.Conv2d(
            in_channels, mask_channels, kernel_size=kernel_size,
            stride=stride, padding=self.padding, dilation=dilation
        )
        nn.init.zeros_(self.mask_conv.weight)
        nn.init.zeros_(self.mask_conv.bias)

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.kaiming_uniform_(self.weight, a=1)
        self.groups = groups

    def forward(self, x):
        offset = self.offset_conv(x)               # delta_p_k (Eq. 7)
        mask = torch.sigmoid(self.mask_conv(x))     # modulation in [0, 1]
        out = deform_conv2d(
            x, offset, self.weight, bias=self.bias,
            stride=self.stride, padding=self.padding,
            dilation=self.dilation, mask=mask
        )
        return out


class DDConvModule(nn.Module):
    """
    Full Dilated-Deformable Convolution module as described in the
    manuscript: standard 3x3 conv -> max-pool -> dilated conv -> deformable
    conv, producing an enriched feature map with both global context and
    fine-grained structural detail, ready to be consumed by the RPN.
    """

    def __init__(self, in_channels, out_channels=256, dilation_rate=2,
                 deform_kernel_size=3):
        super().__init__()

        self.stem_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dilated_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                      padding=dilation_rate, dilation=dilation_rate),  # Eq. 6
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.deformable_conv = DeformableDilatedConv2d(
            out_channels, out_channels, kernel_size=deform_kernel_size,
            stride=1, dilation=1
        )  # Eq. 7, adaptive sampling offsets applied after dilation
        self.bn_deform = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.stem_conv(x)
        x = self.pool(x)
        x = self.dilated_conv(x)              # enlarged receptive field
        x = self.deformable_conv(x)           # adaptive spatial sampling
        x = self.relu(self.bn_deform(x))
        return x                              # enriched feature map F_enriched


if __name__ == "__main__":
    module = DDConvModule(in_channels=512, out_channels=256)
    dummy = torch.randn(1, 512, 32, 32)
    out = module(dummy)
    print("DDConv output shape:", out.shape)
