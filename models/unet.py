# models/unet.py
# =========================================================
# U-Net architecture with optional auxiliary feature extractors
#
# FIXES:
#   1. BinaryUNet clean binary path, raw logits only
#   2. No double sigmoid
#   3. BinaryUNet accepts n_classes for backward compatibility
#   4. UNet64 supports 64x64 / 128x128 tiles
#   5. FIXED UNet64 channel mismatch:
#          down3: ch2 -> ch3
#      NOT:
#          down3: ch2 -> ch3 // factor
#   6. BinaryUNet / BinaryUNet64 preserve architecture config
#   7. get_config() contains base_channels / num_levels / bilinear
#   8. Standard UNet supports configurable base_channels/num_levels
#   9. Optional ResNet/MobileNet/EfficientNet feature extractors
#  10. Binary path remains pure U-Net when backbone == "unet"
# =========================================================

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


# =========================================================
# BUILDING BLOCKS
# =========================================================

class DoubleConv(nn.Module):
    """
    Two consecutive 3x3 convolutions.

    Conv -> BatchNorm -> ReLU
    Conv -> BatchNorm -> ReLU
    """

    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()

        if mid_ch is None:
            mid_ch = out_ch

        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_ch,
                mid_ch,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                mid_ch,
                out_ch,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    MaxPool followed by DoubleConv.
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upsampling followed by concatenation with the skip connection
    and DoubleConv.

    IMPORTANT:
        in_ch is the TOTAL number of channels AFTER concatenation.

    Example:

        decoder = 128 channels
        skip    = 128 channels

        concatenated tensor = 256 channels

        therefore:

        Up(256, 64, bilinear=True)
    """

    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2,
                mode="bilinear",
                align_corners=True,
            )

            self.conv = DoubleConv(
                in_ch,
                out_ch,
                in_ch // 2,
            )

        else:
            self.up = nn.ConvTranspose2d(
                in_ch,
                in_ch // 2,
                kernel_size=2,
                stride=2,
            )

            self.conv = DoubleConv(
                in_ch,
                out_ch,
            )

    def forward(self, x1, x2):

        # x1 = decoder feature
        # x2 = encoder skip connection

        x1 = self.up(x1)

        # Handle possible one-pixel spatial differences.
        dy = x2.size(2) - x1.size(2)
        dx = x2.size(3) - x1.size(3)

        x1 = F.pad(
            x1,
            [
                dx // 2,
                dx - dx // 2,
                dy // 2,
                dy - dy // 2,
            ],
        )

        # Concatenate skip + decoder features.
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


class OutConv(nn.Module):
    """
    Final 1x1 convolution.

    For binary segmentation:
        out_ch = 1

    The layer outputs RAW LOGITS.

    Sigmoid must be applied outside the model.
    """

    def __init__(
        self,
        in_ch,
        out_ch,
        bias_init=None,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=1,
        )

        if bias_init is not None:
            with torch.no_grad():
                self.conv.bias.fill_(float(bias_init))

    def forward(self, x):
        return self.conv(x)


# =========================================================
# STANDARD U-NET
# =========================================================

class UNet(nn.Module):
    """
    Standard U-Net.

    Default architecture:

        base_channels = 64
        num_levels = 4

    Channel schedule:

        64
        128
        256
        512
        1024

    For bilinear=True, the bottleneck is reduced by factor=2,
    following the conventional U-Net implementation.

    This class also supports configurable architecture sizes.
    """

    def __init__(
        self,
        n_channels=3,
        n_classes=2,
        bilinear=True,
        bias_init=None,
        base_channels=64,
        num_levels=4,
    ):
        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.base_channels = int(base_channels)
        self.num_levels = int(num_levels)
        self.architecture = "UNet"

        if self.num_levels not in (3, 4):
            raise ValueError(
                "UNet currently supports num_levels=3 or num_levels=4"
            )

        # Channel schedule.
        #
        # num_levels=4:
        #
        # 64, 128, 256, 512, 1024
        #
        # num_levels=3:
        #
        # 64, 128, 256, 512
        ch = [
            self.base_channels * (2 ** i)
            for i in range(self.num_levels + 1)
        ]

        factor = 2 if bilinear else 1

        # ---------------------------------------------------------
        # Encoder
        # ---------------------------------------------------------

        self.inc = DoubleConv(
            n_channels,
            ch[0],
        )

        self.down1 = Down(
            ch[0],
            ch[1],
        )

        self.down2 = Down(
            ch[1],
            ch[2],
        )

        self.down3 = Down(
            ch[2],
            ch[3],
        )

        if self.num_levels >= 4:

            # Standard U-Net bottleneck.
            self.down4 = Down(
                ch[3],
                ch[4] // factor,
            )

        else:

            self.down4 = None

        # ---------------------------------------------------------
        # Decoder
        # ---------------------------------------------------------

        if self.num_levels >= 4:

            self.up1 = Up(
                ch[4],
                ch[3] // factor,
                bilinear,
            )

            self.up2 = Up(
                ch[3],
                ch[2] // factor,
                bilinear,
            )

            self.up3 = Up(
                ch[2],
                ch[1] // factor,
                bilinear,
            )

            self.up4 = Up(
                ch[1],
                ch[0],
                bilinear,
            )

        else:

            # Three-level U-Net.
            # With bilinear upsampling, Up() does not change the number
            # of decoder channels before concatenation, so in_ch must be
            # decoder_channels + skip_channels.

            self.up1 = Up(
                ch[3] + ch[2],
                ch[2],
                bilinear,
            )

            self.up2 = Up(
                ch[2] + ch[1],
                ch[1],
                bilinear,
            )

            self.up3 = Up(
                ch[1] + ch[0],
                ch[0],
                bilinear,
            )

            self.up4 = None

        # ---------------------------------------------------------
        # Output
        # ---------------------------------------------------------

        self.outc = OutConv(
            ch[0],
            n_classes,
            bias_init=bias_init,
        )

    def forward(self, x):

        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        if self.down4 is not None:

            x5 = self.down4(x4)

            # Decoder
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)

        else:

            # Three-level decoder
            x = self.up1(x4, x3)
            x = self.up2(x, x2)
            x = self.up3(x, x1)

        return self.outc(x)

    def get_config(self):

        return {
            "arch": "unet",
            "n_channels": self.n_channels,
            "n_classes": self.n_classes,
            "base_channels": self.base_channels,
            "num_levels": self.num_levels,
            "bilinear": self.bilinear,
        }


# =========================================================
# BINARY U-NET
# =========================================================

class BinaryUNet(nn.Module):
    """
    Binary U-Net.

    Output:
        [B, 1, H, W]

    The output is RAW LOGITS.

    IMPORTANT:
        No sigmoid is applied inside forward().
    """

    def __init__(
        self,
        n_channels=3,
        n_classes=1,
        bilinear=True,
        bias_init=None,
        base_channels=64,
        num_levels=4,
    ):
        super().__init__()

        self._declared_classes = n_classes

        self.n_channels = n_channels
        self.base_channels = int(base_channels)
        self.num_levels = int(num_levels)
        self.bilinear = bilinear
        self.architecture = "BinaryUNet"

        # Binary model ALWAYS has exactly one output channel.
        self.model = UNet(
            n_channels=n_channels,
            n_classes=1,
            bilinear=bilinear,
            bias_init=bias_init,
            base_channels=base_channels,
            num_levels=num_levels,
        )

    def forward(self, x):
        return self.model(x)

    def get_config(self):

        return {
            "arch": "binary_unet",
            "n_channels": self.n_channels,
            "n_classes": 1,
            "base_channels": self.base_channels,
            "num_levels": self.num_levels,
            "bilinear": self.bilinear,
        }


# =========================================================
# SMALL U-NET
# =========================================================

class UNet64(nn.Module):
    """
    Small U-Net designed for 64x64 and 128x128 tiles.

    Default channel architecture:

        Input
          |
          v
        3 -> 32
          |
        32 -> 64
          |
        64 -> 128
          |
        128 -> 128
          |
        bottleneck
          |
        128 + 128 -> 64
          |
        64 + 64 -> 32
          |
        32 + 32 -> 32
          |
        32 -> n_classes

    For 64x64 input:

        64x64
          ↓
        32x32
          ↓
        16x16
          ↓
         8x8
          ↓
        16x16
          ↓
        32x32
          ↓
        64x64

    IMPORTANT FIX:

        The previous implementation used:

            down3 = Down(ch2, ch3 // factor)

        With bilinear=True this changed:

            128 -> 64

        while up1 expected:

            128 + 128 = 256

        causing:

            expected 256 channels
            received 192 channels

        The correct implementation is:

            down3 = Down(ch2, ch3)

        so the bottleneck remains 128 channels.
    """

    def __init__(
        self,
        n_channels=3,
        n_classes=2,
        bilinear=True,
        bias_init=None,
        base_channels=32,
        num_levels=3,
    ):
        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.base_channels = int(base_channels)
        self.num_levels = int(num_levels)
        self.architecture = "UNet64"

        if self.num_levels != 3:
            raise ValueError(
                "UNet64 is specifically designed for num_levels=3"
            )

        b = self.base_channels

        ch0 = b
        ch1 = b * 2
        ch2 = b * 4

        # Bottleneck is kept at ch2.
        #
        # For base_channels=32:
        #
        # ch0 = 32
        # ch1 = 64
        # ch2 = 128
        #
        ch3 = ch2

        # ---------------------------------------------------------
        # Encoder
        # ---------------------------------------------------------

        self.inc = DoubleConv(
            n_channels,
            ch0,
        )

        self.down1 = Down(
            ch0,
            ch1,
        )

        self.down2 = Down(
            ch1,
            ch2,
        )

        # =========================================================
        # CRITICAL FIX
        # =========================================================
        #
        # WRONG:
        #
        # self.down3 = Down(ch2, ch3 // factor)
        #
        # With bilinear=True:
        #
        # 128 -> 64
        #
        # Then:
        #
        # x4 = 64 channels
        # x3 = 128 channels
        #
        # concat = 192 channels
        #
        # But up1 expects 256.
        #
        # CORRECT:
        #
        # 128 -> 128
        #
        # =========================================================

        self.down3 = Down(
            ch2,
            ch3,
        )

        # ---------------------------------------------------------
        # Decoder
        # ---------------------------------------------------------

        # x4 = 128
        # x3 = 128
        # concat = 256
        #
        # 256 -> 64

        self.up1 = Up(
            ch3 + ch2,
            ch1,
            bilinear,
        )

        # Decoder output = 64
        # x2            = 64
        # concat        = 128
        #
        # 128 -> 32

        self.up2 = Up(
            ch1 + ch1,
            ch0,
            bilinear,
        )

        # Decoder output = 32
        # x1            = 32
        # concat        = 64
        #
        # 64 -> 32

        self.up3 = Up(
            ch0 + ch0,
            ch0,
            bilinear,
        )

        # Output

        self.outc = OutConv(
            ch0,
            n_classes,
            bias_init=bias_init,
        )

    def forward(self, x):

        # ---------------------------------------------------------
        # Encoder
        # ---------------------------------------------------------

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # ---------------------------------------------------------
        # Decoder
        # ---------------------------------------------------------

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        return self.outc(x)

    def get_config(self):

        return {
            "arch": "unet64",
            "n_channels": self.n_channels,
            "n_classes": self.n_classes,
            "base_channels": self.base_channels,
            "num_levels": self.num_levels,
            "bilinear": self.bilinear,
        }


# =========================================================
# BINARY SMALL U-NET
# =========================================================

class BinaryUNet64(nn.Module):
    """
    Binary U-Net for 64x64 / 128x128 tiles.

    Output:
        [B, 1, H, W]

    Output is RAW LOGITS.
    Sigmoid is applied outside the model.

    Default architecture:

        BinaryUNet64
            |
            +-- UNet64
                    |
                    +-- 32
                    +-- 64
                    +-- 128
                    +-- 128 bottleneck
    """

    def __init__(
        self,
        n_channels=3,
        n_classes=1,
        bilinear=True,
        bias_init=None,
        base_channels=32,
        num_levels=3,
    ):
        super().__init__()

        self._declared_classes = n_classes

        self.n_channels = n_channels
        self.base_channels = int(base_channels)
        self.num_levels = int(num_levels)
        self.bilinear = bilinear
        self.architecture = "BinaryUNet64"

        # Binary model ALWAYS produces one logit channel.

        self.model = UNet64(
            n_channels=n_channels,
            n_classes=1,
            bilinear=bilinear,
            bias_init=bias_init,
            base_channels=base_channels,
            num_levels=num_levels,
        )

    def forward(self, x):
        return self.model(x)

    def get_config(self):

        return {
            "arch": "binary_unet64",
            "n_channels": self.n_channels,
            "n_classes": 1,
            "base_channels": self.base_channels,
            "num_levels": self.num_levels,
            "bilinear": self.bilinear,
        }


# =========================================================
# BACKBONE ENCODERS
# =========================================================

def get_backbone_encoder(
    name,
    pretrained=True,
):

    name = name.lower()

    # ---------------------------------------------------------
    # ResNet
    # ---------------------------------------------------------

    if name.startswith("resnet"):

        builders = {
            "resnet18": (
                tv_models.resnet18,
                tv_models.ResNet18_Weights.DEFAULT,
            ),

            "resnet34": (
                tv_models.resnet34,
                tv_models.ResNet34_Weights.DEFAULT,
            ),

            "resnet50": (
                tv_models.resnet50,
                tv_models.ResNet50_Weights.IMAGENET1K_V2,
            ),

            "resnet101": (
                tv_models.resnet101,
                tv_models.ResNet101_Weights.DEFAULT,
            ),
        }

        if name not in builders:
            name = "resnet50"

        fn, weights = builders[name]

        try:

            model = fn(
                weights=weights if pretrained else None
            )

        except Exception:

            model = fn(
                weights=None
            )

        class ResNetFeatures(nn.Module):

            def __init__(self, model):

                super().__init__()

                self.stem = nn.Sequential(
                    model.conv1,
                    model.bn1,
                    model.relu,
                    model.maxpool,
                )

                self.layer1 = model.layer1
                self.layer2 = model.layer2
                self.layer3 = model.layer3
                self.layer4 = model.layer4

            def forward(self, x):

                x = self.stem(x)

                c1 = self.layer1(x)
                c2 = self.layer2(c1)
                c3 = self.layer3(c2)
                c4 = self.layer4(c3)

                return [
                    c1,
                    c2,
                    c3,
                    c4,
                ]

        return ResNetFeatures(model)

    # ---------------------------------------------------------
    # MobileNet V2
    # ---------------------------------------------------------

    if name.startswith("mobilenet"):

        try:

            model = tv_models.mobilenet_v2(
                weights=(
                    tv_models.MobileNet_V2_Weights.DEFAULT
                    if pretrained
                    else None
                )
            )

        except Exception:

            model = tv_models.mobilenet_v2(
                weights=None
            )

        class MobileNetFeatures(nn.Module):

            def __init__(self, features):

                super().__init__()

                self.feats = features

            def forward(self, x):

                target_scales = [
                    4,
                    8,
                    16,
                    32,
                ]

                collected = {}

                current = x

                for block in self.feats:

                    current = block(current)

                    for scale in target_scales:

                        if (
                            scale not in collected
                            and current.shape[-2]
                            <= x.shape[-2] / scale * 1.2
                        ):
                            collected[scale] = current

                result = []

                last = None

                for scale in target_scales:

                    if scale in collected:
                        last = collected[scale]

                    result.append(
                        last
                        if last is not None
                        else current
                    )

                first = next(
                    (
                        r
                        for r in result
                        if r is not None
                    ),
                    current,
                )

                return [
                    r
                    if r is not None
                    else first
                    for r in result
                ]

        return MobileNetFeatures(
            model.features
        )

    # ---------------------------------------------------------
    # EfficientNet B0
    # ---------------------------------------------------------

    if name.startswith("efficientnet"):

        try:

            model = tv_models.efficientnet_b0(
                weights=(
                    tv_models.EfficientNet_B0_Weights.DEFAULT
                    if pretrained
                    else None
                )
            )

        except Exception:

            model = tv_models.efficientnet_b0(
                weights=None
            )

        class EffNetFeatures(nn.Module):

            def __init__(self, features):

                super().__init__()

                self.feats = features

            def forward(self, x):

                target_scales = [
                    4,
                    8,
                    16,
                    32,
                ]

                collected = {}

                current = x

                for block in self.feats:

                    current = block(current)

                    for scale in target_scales:

                        if (
                            scale not in collected
                            and current.shape[-2]
                            <= x.shape[-2] / scale * 1.2
                        ):
                            collected[scale] = current

                result = []

                last = None

                for scale in target_scales:

                    if scale in collected:
                        last = collected[scale]

                    result.append(
                        last
                        if last is not None
                        else current
                    )

                first = next(
                    (
                        r
                        for r in result
                        if r is not None
                    ),
                    current,
                )

                return [
                    r
                    if r is not None
                    else first
                    for r in result
                ]

        return EffNetFeatures(
            model.features
        )

    raise ValueError(
        f"Unknown feature extractor: {name}"
    )


# =========================================================
# BACKBONE CHANNEL DEFINITIONS
# =========================================================

BACKBONE_CHANNELS = {

    "resnet18": [
        64,
        128,
        128,
        512,
    ],

    "resnet34": [
        64,
        128,
        128,
        512,
    ],

    "resnet50": [
        128,
        512,
        1024,
        2048,
    ],

    "resnet101": [
        128,
        512,
        1024,
        2048,
    ],

    "mobilenet_v2": [
        24,
        32,
        96,
        1280,
    ],

    "efficientnet_b0": [
        24,
        40,
        112,
        320,
    ],
}


# =========================================================
# HYBRID U-NET + BACKBONE
# =========================================================

class UNetWithFeatureExtractor(nn.Module):
    """
    Standard U-Net with optional backbone support.

    Used for multiclass / feature-extractor models.

    When feature_extractor == "unet":

        No external backbone is used.

    When feature_extractor is:

        resnet18
        resnet34
        resnet50
        resnet101
        mobilenet_v2
        efficientnet_b0

    the backbone is fused into the U-Net bottleneck.
    """

    def __init__(
        self,
        n_channels=3,
        n_classes=2,
        feature_extractor="unet",
        bilinear=True,
        pretrained=True,
        fuse_at="bottleneck",
        support_weight=0.3,
        backbone=None,
        base_channels=None,
        num_levels=None,
    ):
        super().__init__()

        # Some callers use backbone= instead of
        # feature_extractor=.

        if (
            backbone is not None
            and feature_extractor == "unet"
        ):
            feature_extractor = backbone

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.feature_extractor = feature_extractor
        self.fuse_at = fuse_at
        self.support_weight = support_weight

        self.use_backbone = (
            feature_extractor != "unet"
        )

        # ---------------------------------------------------------
        # PRIMARY U-NET
        # ---------------------------------------------------------

        self.inc = DoubleConv(
            n_channels,
            64,
        )

        self.down1 = Down(
            64,
            128,
        )

        self.down2 = Down(
            128,
            256,
        )

        self.down3 = Down(
            256,
            512,
        )

        factor = 2 if bilinear else 1

        self.down4 = Down(
            512,
            1024 // factor,
        )

        # Decoder

        self.up1 = Up(
            1024,
            512 // factor,
            bilinear,
        )

        self.up2 = Up(
            512,
            256 // factor,
            bilinear,
        )

        self.up3 = Up(
            256,
            128 // factor,
            bilinear,
        )

        self.up4 = Up(
            128,
            64,
            bilinear,
        )

        self.outc = OutConv(
            64,
            n_classes,
        )

        # ---------------------------------------------------------
        # BACKBONE SUPPORT
        # ---------------------------------------------------------

        if self.use_backbone:

            self.encoder = get_backbone_encoder(
                feature_extractor,
                pretrained=pretrained,
            )

            chs = BACKBONE_CHANNELS.get(
                feature_extractor,
                [
                    128,
                    512,
                    1024,
                    2048,
                ],
            )

            def make_adapter(
                cin,
                cout,
            ):

                return nn.Sequential(
                    nn.Conv2d(
                        cin,
                        cout,
                        kernel_size=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(cout),
                    nn.GELU(),
                )

            self.adapter3 = make_adapter(
                chs[2],
                1024 // factor,
            )

            self.adapter4 = make_adapter(
                chs[3],
                1024 // factor,
            )

            self.alpha3 = nn.Parameter(
                torch.tensor(
                    float(support_weight)
                )
            )

            self.alpha4 = nn.Parameter(
                torch.tensor(
                    float(support_weight)
                )
            )

            if fuse_at == "all_skips":

                self.adapter1 = make_adapter(
                    chs[0],
                    128,
                )

                self.adapter2 = make_adapter(
                    chs[1],
                    512,
                )

                self.alpha1 = nn.Parameter(
                    torch.tensor(
                        float(support_weight)
                    )
                )

                self.alpha2 = nn.Parameter(
                    torch.tensor(
                        float(support_weight)
                    )
                )

    def forward(self, x):

        # ---------------------------------------------------------
        # PRIMARY U-NET
        # ---------------------------------------------------------

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # ---------------------------------------------------------
        # BACKBONE
        # ---------------------------------------------------------

        if self.use_backbone:

            bf = self.encoder(x)

            b3 = bf[2]
            b4 = bf[3]

            s3 = self.adapter3(b3)
            s4 = self.adapter4(b4)

            if s3.shape[-2:] != x5.shape[-2:]:

                s3 = F.interpolate(
                    s3,
                    size=x5.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            if s4.shape[-2:] != x5.shape[-2:]:

                s4 = F.interpolate(
                    s4,
                    size=x5.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            x5 = (
                x5
                + self.alpha3 * s3
                + self.alpha4 * s4
            )

            if self.fuse_at == "all_skips":

                s1 = self.adapter1(
                    bf[0]
                )

                s2 = self.adapter2(
                    bf[1]
                )

                if s1.shape[-2:] != x3.shape[-2:]:

                    s1 = F.interpolate(
                        s1,
                        size=x3.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )

                if s2.shape[-2:] != x4.shape[-2:]:

                    s2 = F.interpolate(
                        s2,
                        size=x4.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )

                x3 = (
                    x3
                    + self.alpha1 * s1
                )

                x4 = (
                    x4
                    + self.alpha2 * s2
                )

        # ---------------------------------------------------------
        # DECODER
        # ---------------------------------------------------------

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.outc(x)

    def get_config(self):

        return {
            "arch": "unet_with_feature_extractor",
            "n_channels": self.n_channels,
            "n_classes": self.n_classes,
            "feature_extractor": self.feature_extractor,
            "fuse_at": self.fuse_at,
            "support_weight": self.support_weight,
            "bilinear": self.bilinear,
        }


# Compatibility aliases

BackboneUNet = UNetWithFeatureExtractor

UNetSegmentation = UNetWithFeatureExtractor


# =========================================================
# SMALL U-NET + BACKBONE
# =========================================================

class UNet64WithFeatureExtractor(nn.Module):
    """
    Small-tile U-Net with optional backbone support.

    Intended for 64x64 / 128x128 tiles.

    Primary U-Net:

        3 -> 32
        32 -> 64
        64 -> 128
        128 -> 128
    """

    def __init__(
        self,
        n_channels=3,
        n_classes=2,
        feature_extractor="unet",
        bilinear=True,
        pretrained=True,
        fuse_at="bottleneck",
        support_weight=0.3,
        backbone=None,
        base_channels=None,
        num_levels=None,
    ):
        super().__init__()

        if (
            backbone is not None
            and feature_extractor == "unet"
        ):
            feature_extractor = backbone

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.feature_extractor = feature_extractor
        self.fuse_at = fuse_at
        self.support_weight = support_weight

        self.use_backbone = (
            feature_extractor != "unet"
        )

        # ---------------------------------------------------------
        # PRIMARY SMALL U-NET
        # ---------------------------------------------------------

        self.inc = DoubleConv(
            n_channels,
            32,
        )

        self.down1 = Down(
            32,
            64,
        )

        self.down2 = Down(
            64,
            128,
        )

        # IMPORTANT:
        # Keep bottleneck at 128 channels.
        self.down3 = Down(
            128,
            128,
        )

        self.up1 = Up(
            128 + 128,
            64,
            bilinear,
        )

        self.up2 = Up(
            64 + 64,
            32,
            bilinear,
        )

        self.up3 = Up(
            32 + 32,
            32,
            bilinear,
        )

        self.outc = OutConv(
            32,
            n_classes,
        )

        # ---------------------------------------------------------
        # BACKBONE SUPPORT
        # ---------------------------------------------------------

        if self.use_backbone:

            self.encoder = get_backbone_encoder(
                feature_extractor,
                pretrained=pretrained,
            )

            chs = BACKBONE_CHANNELS.get(
                feature_extractor,
                [
                    128,
                    512,
                    1024,
                    2048,
                ],
            )

            def make_adapter(
                cin,
                cout,
            ):

                return nn.Sequential(
                    nn.Conv2d(
                        cin,
                        cout,
                        kernel_size=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(cout),
                    nn.GELU(),
                )

            self.adapter_b = make_adapter(
                chs[3],
                128,
            )

            self.alpha_b = nn.Parameter(
                torch.tensor(
                    float(support_weight)
                )
            )

    def forward(self, x):

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        if self.use_backbone:

            bf = self.encoder(x)

            sb = self.adapter_b(
                bf[3]
            )

            if sb.shape[-2:] != x4.shape[-2:]:

                sb = F.interpolate(
                    sb,
                    size=x4.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            x4 = (
                x4
                + self.alpha_b * sb
            )

        x = self.up1(
            x4,
            x3,
        )

        x = self.up2(
            x,
            x2,
        )

        x = self.up3(
            x,
            x1,
        )

        return self.outc(x)

    def get_config(self):

        return {
            "arch": "unet64_with_feature_extractor",
            "n_channels": self.n_channels,
            "n_classes": self.n_classes,
            "feature_extractor": self.feature_extractor,
            "fuse_at": self.fuse_at,
            "support_weight": self.support_weight,
            "bilinear": self.bilinear,
        }


# =========================================================
# LOSSES
# =========================================================

def binary_dice_loss(
    logits,
    target,
    eps=1e-6,
):
    """
    Binary Dice loss.

    Input:
        raw logits

    Sigmoid is applied exactly once here.
    """

    if (
        logits.dim() == 4
        and logits.size(1) == 1
    ):
        logits = logits[:, 0]

    if (
        target.dim() == 4
        and target.size(1) == 1
    ):
        target = target[:, 0]

    target = target.float()

    probs = torch.sigmoid(logits)

    inter = (
        probs * target
    ).sum(
        dim=(1, 2)
    )

    denom = (
        probs.sum(dim=(1, 2))
        + target.sum(dim=(1, 2))
    )

    dice = (
        2.0 * inter + eps
    ) / (
        denom + eps
    )

    return 1.0 - dice.mean()


def dice_loss(
    pred,
    target,
    eps=1e-6,
    include_background=False,
):
    """
    Multiclass Dice loss.

    If pred has one channel, use binary Dice.
    """

    if (
        pred.dim() == 4
        and pred.size(1) == 1
    ):
        return binary_dice_loss(
            pred,
            target,
            eps=eps,
        )

    num_classes = pred.size(1)

    probs = torch.softmax(
        pred,
        dim=1,
    )

    target_1h = F.one_hot(
        target.long(),
        num_classes,
    ).permute(
        0,
        3,
        1,
        2,
    ).float()

    dims = (
        0,
        2,
        3,
    )

    inter = (
        probs * target_1h
    ).sum(dims)

    union = (
        probs.sum(dims)
        + target_1h.sum(dims)
    )

    dice = (
        2 * inter + eps
    ) / (
        union + eps
    )

    if (
        not include_background
        and num_classes > 1
    ):
        dice = dice[1:]

    return 1 - dice.mean()


def combined_loss(
    pred,
    target,
    bce_weight=0.5,
    dice_weight=0.5,
    ignore_index=None,
    include_background=False,
    pos_weight=5.0,
):
    """
    Binary:
        BCEWithLogits + Dice

    Multiclass:
        CrossEntropy + foreground Dice
    """

    # ---------------------------------------------------------
    # BINARY
    # ---------------------------------------------------------

    if (
        pred.dim() == 4
        and pred.size(1) == 1
    ):

        if (
            target.dim() == 4
            and target.size(1) == 1
        ):
            target = target[:, 0]

        target = target.float()

        if pos_weight is not None:

            pw = torch.tensor(
                float(pos_weight),
                device=pred.device,
                dtype=pred.dtype,
            )

        else:

            pw = None

        bce = F.binary_cross_entropy_with_logits(
            pred[:, 0],
            target,
            pos_weight=pw,
        )

        dice = binary_dice_loss(
            pred,
            target,
        )

        return (
            bce_weight * bce
            + dice_weight * dice
        )

    # ---------------------------------------------------------
    # MULTICLASS
    # ---------------------------------------------------------

    ce = F.cross_entropy(
        pred,
        target.long(),
        ignore_index=(
            ignore_index
            if ignore_index is not None
            else -100
        ),
    )

    dice = dice_loss(
        pred,
        target,
        include_background=include_background,
    )

    return (
        bce_weight * ce
        + dice_weight * dice
    )


class DiceLoss(nn.Module):

    def __init__(
        self,
        include_background=False,
    ):
        super().__init__()

        self.include_background = (
            include_background
        )

    def forward(
        self,
        pred,
        target,
    ):

        return dice_loss(
            pred,
            target,
            include_background=(
                self.include_background
            ),
        )


class CombinedLoss(nn.Module):

    def __init__(
        self,
        bce_weight=0.5,
        dice_weight=0.5,
        include_background=False,
        pos_weight=5.0,
    ):
        super().__init__()

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.include_background = (
            include_background
        )
        self.pos_weight = pos_weight

    def forward(
        self,
        pred,
        target,
    ):

        return combined_loss(
            pred,
            target,
            bce_weight=self.bce_weight,
            dice_weight=self.dice_weight,
            include_background=(
                self.include_background
            ),
            pos_weight=self.pos_weight,
        )


# =========================================================
# BINARY METRICS
# =========================================================

def binary_metrics(
    logits,
    target,
    threshold=0.5,
    eps=1e-7,
):
    """
    Binary segmentation metrics.

    logits:
        raw model output

    target:
        binary ground truth

    threshold:
        sigmoid probability threshold
    """

    if logits.dim() == 4:
        logits = logits[:, 0]

    if target.dim() == 4:
        target = target[:, 0]

    probs = torch.sigmoid(logits)

    pred = (
        probs >= threshold
    ).long()

    target = target.long()

    tp = (
        (pred == 1)
        & (target == 1)
    ).sum().item()

    fp = (
        (pred == 1)
        & (target == 0)
    ).sum().item()

    fn = (
        (pred == 0)
        & (target == 1)
    ).sum().item()

    iou = (
        tp
        / (tp + fp + fn + eps)
    )

    precision = (
        tp
        / (tp + fp + eps)
    )

    recall = (
        tp
        / (tp + fn + eps)
    )

    f1 = (
        2 * precision * recall
        / (
            precision
            + recall
            + eps
        )
    )

    return {
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


# =========================================================
# NUMPY CONVERSION
# =========================================================

def _to_numpy(x):

    if isinstance(x, np.ndarray):
        return x

    if torch.is_tensor(x):
        return (
            x.detach()
            .cpu()
            .numpy()
        )

    raise TypeError(
        f"Unsupported type: {type(x)}"
    )


# =========================================================
# GENERAL METRICS
# =========================================================

def compute_metrics(
    pred,
    target,
    num_classes=2,
    eps=1e-6,
    foreground_only=False,
    is_logits=True,
):
    """
    Compute IoU, F1, precision and recall.

    Supports:
        binary predictions
        multiclass predictions
    """

    with torch.no_grad():

        p = _to_numpy(pred)
        t = _to_numpy(target)

        # -----------------------------------------------------
        # Multiclass channel-first prediction
        # -----------------------------------------------------

        if (
            p.ndim == 3
            and p.shape[0] == num_classes
        ):

            p = np.argmax(
                p,
                axis=0,
            )

        # -----------------------------------------------------
        # Binary floating prediction
        # -----------------------------------------------------

        elif (
            p.ndim == 2
            and p.dtype
            in (
                np.float32,
                np.float64,
            )
        ):

            if is_logits:

                p = (
                    1.0
                    / (
                        1.0
                        + np.exp(-p)
                    )
                )

            p = (
                p > 0.5
            ).astype(
                np.int64
            )

        p = p.astype(
            np.int64
        )

        t = t.astype(
            np.int64
        )

        per_iou = []
        per_f1 = []
        per_prec = []
        per_rec = []

        # -----------------------------------------------------
        # Per-class metrics
        # -----------------------------------------------------

        for c in range(
            num_classes
        ):

            pc = (
                p == c
            )

            tc = (
                t == c
            )

            tp = int(
                (
                    pc & tc
                ).sum()
            )

            fp = int(
                (
                    pc & ~tc
                ).sum()
            )

            fn = int(
                (
                    ~pc & tc
                ).sum()
            )

            union = (
                tp
                + fp
                + fn
            )

            if union > 0:

                iou = (
                    tp
                    / (
                        union
                        + eps
                    )
                )

            else:

                iou = 0.0

            if (
                tp + fp
            ) > 0:

                precision = (
                    tp
                    / (
                        tp
                        + fp
                        + eps
                    )
                )

            else:

                precision = 0.0

            if (
                tp + fn
            ) > 0:

                recall = (
                    tp
                    / (
                        tp
                        + fn
                        + eps
                    )
                )

            else:

                recall = 0.0

            if (
                precision
                + recall
            ) > 0:

                f1 = (
                    2
                    * precision
                    * recall
                    / (
                        precision
                        + recall
                        + eps
                    )
                )

            else:

                f1 = 0.0

            per_iou.append(iou)
            per_f1.append(f1)
            per_prec.append(precision)
            per_rec.append(recall)

        # -----------------------------------------------------
        # Aggregate
        # -----------------------------------------------------

        if (
            foreground_only
            and num_classes > 1
        ):

            mean_iou = float(
                np.mean(
                    per_iou[1:]
                )
            )

            mean_f1 = float(
                np.mean(
                    per_f1[1:]
                )
            )

            mean_prec = float(
                np.mean(
                    per_prec[1:]
                )
            )

            mean_rec = float(
                np.mean(
                    per_rec[1:]
                )
            )

        else:

            mean_iou = float(
                np.mean(per_iou)
            )

            mean_f1 = float(
                np.mean(per_f1)
            )

            mean_prec = float(
                np.mean(per_prec)
            )

            mean_rec = float(
                np.mean(per_rec)
            )

        return {
            "iou": mean_iou,
            "f1": mean_f1,
            "precision": mean_prec,
            "recall": mean_rec,

            "per_class_iou": per_iou,
            "per_class_f1": per_f1,
            "per_class_precision": per_prec,
            "per_class_recall": per_rec,
        }


# =========================================================
# OPTIONAL SELF-TEST
# =========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("U-Net architecture self-test")
    print("=" * 70)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    # ---------------------------------------------------------
    # Test 64x64 binary U-Net
    # ---------------------------------------------------------

    print()
    print("Testing BinaryUNet64...")
    
    model = BinaryUNet64(
        n_channels=3,
        n_classes=1,
        bilinear=True,
        base_channels=32,
        num_levels=3,
    ).to(device)

    x = torch.randn(
        2,
        3,
        64,
        64,
        device=device,
    )

    with torch.no_grad():

        y = model(x)

    print(
        f"Input :  {tuple(x.shape)}"
    )

    print(
        f"Output:  {tuple(y.shape)}"
    )

    assert y.shape == (
        2,
        1,
        64,
        64,
    ), (
        f"Unexpected output shape: "
        f"{tuple(y.shape)}"
    )

    print(
        "✅ BinaryUNet64 forward pass OK"
    )

    # ---------------------------------------------------------
    # Test 128x128 binary U-Net
    # ---------------------------------------------------------

    print()
    print("Testing BinaryUNet64 with 128x128...")

    x128 = torch.randn(
        2,
        3,
        128,
        128,
        device=device,
    )

    with torch.no_grad():

        y128 = model(x128)

    print(
        f"Input :  {tuple(x128.shape)}"
    )

    print(
        f"Output:  {tuple(y128.shape)}"
    )

    assert y128.shape == (
        2,
        1,
        128,
        128,
    ), (
        f"Unexpected output shape: "
        f"{tuple(y128.shape)}"
    )

    print(
        "✅ 128x128 forward pass OK"
    )

    # ---------------------------------------------------------
    # Test standard binary U-Net
    # ---------------------------------------------------------

    print()
    print("Testing standard BinaryUNet...")

    standard = BinaryUNet(
        n_channels=3,
        n_classes=1,
        bilinear=True,
        base_channels=64,
        num_levels=4,
    ).to(device)

    x256 = torch.randn(
        1,
        3,
        256,
        256,
        device=device,
    )

    with torch.no_grad():

        y256 = standard(x256)

    print(
        f"Input :  {tuple(x256.shape)}"
    )

    print(
        f"Output:  {tuple(y256.shape)}"
    )

    assert y256.shape == (
        1,
        1,
        256,
        256,
    ), (
        f"Unexpected output shape: "
        f"{tuple(y256.shape)}"
    )

    print(
        "✅ Standard BinaryUNet forward pass OK"
    )

    # ---------------------------------------------------------
    # Test sigmoid
    # ---------------------------------------------------------

    probs = torch.sigmoid(y)

    print()
    print(
        "Probability range:"
    )

    print(
        f"min={probs.min().item():.6f} "
        f"max={probs.max().item():.6f} "
        f"mean={probs.mean().item():.6f}"
    )

    # ---------------------------------------------------------
    # Test loss
    # ---------------------------------------------------------

    target = torch.randint(
        0,
        2,
        (
            2,
            64,
            64,
        ),
        device=device,
    ).float()

    loss = combined_loss(
        y,
        target,
        bce_weight=0.7,
        dice_weight=0.3,
        pos_weight=8.0,
    )

    print()
    print(
        f"Test loss: {loss.item():.6f}"
    )

    assert torch.isfinite(loss)

    print(
        "✅ Loss calculation OK"
    )

    print()
    print("=" * 70)
    print("ALL U-NET SELF-TESTS PASSED")
    print("=" * 70)

