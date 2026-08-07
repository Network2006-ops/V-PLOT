"""
stn.py
------
Spatial Transformer Network (STN) used to geometrically normalize keyframe
feature maps before dilated-deformable feature extraction (manuscript
Section: Spatial Transformer Network, Eqs. 2-4).

Pipeline:
    1. Localisation network predicts affine transform parameters theta (Eq. 2).
    2. Grid generator maps output coordinates to source coordinates (Eq. 3).
    3. Bilinear sampler resamples the input feature map on the generated
       grid to produce the geometrically normalized output (Eq. 4).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalisationNetwork(nn.Module):
    """Predicts the 2x3 affine transformation matrix theta from an input
    feature map (Eq. 2: theta = f_loc(X))."""

    def __init__(self, in_channels: int, hidden_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, padding=3),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 6),
        )
        # Initialize as identity transform so training starts stable.
        self.fc[-1].weight.data.zero_()
        self.fc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        feat = self.conv(x)
        feat = feat.view(b, -1)
        theta = self.fc(feat)
        return theta.view(b, 2, 3)


class SpatialTransformerNetwork(nn.Module):
    """
    Full STN module: localisation network + affine grid generator + bilinear
    sampler. Applied to intermediate feature maps prior to DDConv so that
    graphical objects distorted by rotation, translation, or scale are
    geometrically normalized (manuscript, Spatial Transformer Network section).
    """

    def __init__(self, in_channels: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.localisation = LocalisationNetwork(in_channels, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input feature map, shape (B, C, H, W)
        Returns:
            Geometrically normalized feature map of the same shape as `x`.
        """
        theta = self.localisation(x)                       # Eq. 2
        grid = F.affine_grid(theta, x.size(), align_corners=False)   # Eq. 3
        x_transformed = F.grid_sample(x, grid, mode="bilinear",
                                       align_corners=False)  # Eq. 4 (bilinear sampler)
        return x_transformed

    def get_theta(self, x: torch.Tensor) -> torch.Tensor:
        """Expose the predicted affine parameters for inspection/analysis."""
        return self.localisation(x)


if __name__ == "__main__":
    stn = SpatialTransformerNetwork(in_channels=64)
    dummy = torch.randn(2, 64, 128, 128)
    out = stn(dummy)
    print("STN output shape:", out.shape)
