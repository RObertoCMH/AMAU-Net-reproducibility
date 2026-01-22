# src/amaunet/models/unet.py
from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import Conv2dBlock
from .attention import CBAM, SelfAttentionBlock


class UNetWithAttention(nn.Module):
    """
    Single configurable U-Net used for all variants (baselines + AMAU-Net).

    Baselines (paper ablations):
      - U-Net (amplitude-only): in_channels=1, cbam_in_layers=[], use_self_attention=False
      - U-Net + attributes:     in_channels=4, cbam_in_layers=[], use_self_attention=False

    AMAU-Net (manuscript setting):
      - in_channels=4
      - cbam_in_layers=[2, 4, 8]
      - use_self_attention=True (bottleneck)

    CBAM insertion IDs implemented here:
      Encoder: 2, 3, 4  (after conv2/conv3/conv4)
      Decoder: 6, 7, 8  (after conv6/conv7/conv8)
    """
    ALLOWED_CBAM_LAYERS = {2, 3, 4, 6, 7, 8}

    def __init__(
        self,
        n_classes: int,
        n_filters: int = 80,
        dropout: float = 0.5,
        batchnorm: bool = True,
        in_channels: int = 1,
        cbam_in_layers: list[int] | None = None,
        use_self_attention: bool = False,
    ):
        super().__init__()

        self.cbam_in_layers = list(cbam_in_layers or [])
        unknown = set(self.cbam_in_layers) - self.ALLOWED_CBAM_LAYERS
        if unknown:
            raise ValueError(
                f"Unsupported cbam_in_layers={sorted(list(unknown))}. "
                f"Allowed: {sorted(list(self.ALLOWED_CBAM_LAYERS))}"
            )

        self.use_self_attention = bool(use_self_attention)

        # Channel widths at each U-Net stage
        c1_ch = n_filters
        c2_ch = n_filters * 2
        c3_ch = n_filters * 4
        c4_ch = n_filters * 8
        c5_ch = n_filters * 16

        # Build CBAM blocks only for requested insertion points
        # IDs map to feature-map channel widths at those points.
        cbam_channels = {
            2: c2_ch,
            3: c3_ch,
            4: c4_ch,
            6: c4_ch,  # conv6 output channels
            7: c3_ch,  # conv7 output channels
            8: c2_ch,  # conv8 output channels
        }
        self.cbam_blocks = nn.ModuleDict()
        for lid in self.cbam_in_layers:
            self.cbam_blocks[f"cbam{lid}"] = CBAM(cbam_channels[lid])

        # -----------------------
        # Encoder
        # -----------------------
        self.conv1 = Conv2dBlock(in_channels, c1_ch, batchnorm)
        self.pool1 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout2d(dropout * 0.5)

        self.conv2 = Conv2dBlock(c1_ch, c2_ch, batchnorm)
        self.pool2 = nn.MaxPool2d(2)
        self.drop2 = nn.Dropout2d(dropout)

        self.conv3 = Conv2dBlock(c2_ch, c3_ch, batchnorm)
        self.pool3 = nn.MaxPool2d(2)
        self.drop3 = nn.Dropout2d(dropout)

        self.conv4 = Conv2dBlock(c3_ch, c4_ch, batchnorm)
        self.pool4 = nn.MaxPool2d(2)
        self.drop4 = nn.Dropout2d(dropout)

        # -----------------------
        # Bottleneck
        # -----------------------
        self.conv5 = Conv2dBlock(c4_ch, c5_ch, batchnorm)
        if self.use_self_attention:
            self.att5 = SelfAttentionBlock(c5_ch)

        # -----------------------
        # Decoder
        # -----------------------
        self.upconv6 = nn.ConvTranspose2d(c5_ch, c4_ch, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv6 = Conv2dBlock(c4_ch + c4_ch, c4_ch, batchnorm)  # concat with c4

        self.upconv7 = nn.ConvTranspose2d(c4_ch, c3_ch, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv7 = Conv2dBlock(c3_ch + c3_ch, c3_ch, batchnorm)  # concat with c3

        self.upconv8 = nn.ConvTranspose2d(c3_ch, c2_ch, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv8 = Conv2dBlock(c2_ch + c2_ch, c2_ch, batchnorm)  # concat with c2

        self.upconv9 = nn.ConvTranspose2d(c2_ch, c1_ch, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv9 = Conv2dBlock(c1_ch + c1_ch, c1_ch, batchnorm)  # concat with c1

        self.final_conv = nn.Conv2d(c1_ch, n_classes, kernel_size=1)

    def _apply_cbam(self, layer_id: int, x: torch.Tensor) -> torch.Tensor:
        key = f"cbam{layer_id}"
        if key in self.cbam_blocks:
            return self.cbam_blocks[key](x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        c1 = self.conv1(x)
        p1 = self.drop1(self.pool1(c1))

        c2 = self.conv2(p1)
        c2 = self._apply_cbam(2, c2)
        p2 = self.drop2(self.pool2(c2))

        c3 = self.conv3(p2)
        c3 = self._apply_cbam(3, c3)
        p3 = self.drop3(self.pool3(c3))

        c4 = self.conv4(p3)
        c4 = self._apply_cbam(4, c4)
        p4 = self.drop4(self.pool4(c4))

        # Bottleneck
        c5 = self.conv5(p4)
        if self.use_self_attention:
            c5 = self.att5(c5)

        # Decoder
        u6 = self.upconv6(c5)
        u6 = torch.cat([u6, c4], dim=1)
        c6 = self.conv6(u6)
        c6 = self._apply_cbam(6, c6)

        u7 = self.upconv7(c6)
        u7 = torch.cat([u7, c3], dim=1)
        c7 = self.conv7(u7)
        c7 = self._apply_cbam(7, c7)

        u8 = self.upconv8(c7)
        u8 = torch.cat([u8, c2], dim=1)
        c8 = self.conv8(u8)
        c8 = self._apply_cbam(8, c8)

        u9 = self.upconv9(c8)
        u9 = torch.cat([u9, c1], dim=1)
        c9 = self.conv9(u9)

        return self.final_conv(c9)
