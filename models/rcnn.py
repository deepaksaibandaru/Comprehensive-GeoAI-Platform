"""
Region-based R-CNN style detector.

This module provides:

    RCNN
    FastRCNN

Both models operate on detection targets rather than semantic masks.

Training input:
    images, targets

where each target contains:

    boxes
    labels

Inference output:
    boxes
    labels
    scores

This is intentionally separate from U-Net semantic segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

from torchvision.ops import roi_align


# ==============================================================
# BACKBONE
# ==============================================================

def _get_backbone(
    name="resnet50",
    pretrained=True,
):
    """
    Returns:

        backbone
        output_channels
        feature_stride
    """

    name = (name or "resnet50").lower()

    # ----------------------------------------------------------
    # RESNET
    # ----------------------------------------------------------

    if name in (
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
    ):

        if pretrained:
            try:
                weights_enum = getattr(
                    torchvision.models,
                    f"{name.capitalize()}_Weights",
                )
                weights = weights_enum.DEFAULT
            except Exception:
                weights = "DEFAULT"
        else:
            weights = None

        model_fn = getattr(
            torchvision.models,
            name,
        )

        model = model_fn(weights=weights)

        backbone = nn.Sequential(
            *list(model.children())[:-2]
        )

        if name in ("resnet18", "resnet34"):
            out_channels = 512
        else:
            out_channels = 2048

        return backbone, out_channels, 32

    # ----------------------------------------------------------
    # MOBILENET V2
    # ----------------------------------------------------------

    if name == "mobilenet_v2":

        weights = (
            torchvision.models.MobileNet_V2_Weights.DEFAULT
            if pretrained
            else None
        )

        model = torchvision.models.mobilenet_v2(
            weights=weights
        )

        return model.features, 1280, 32

    # ----------------------------------------------------------
    # EFFICIENTNET B0
    # ----------------------------------------------------------

    if name == "efficientnet_b0":

        weights = (
            torchvision.models.EfficientNet_B0_Weights.DEFAULT
            if pretrained
            else None
        )

        model = torchvision.models.efficientnet_b0(
            weights=weights
        )

        return model.features, 1280, 32

    # ----------------------------------------------------------
    # VGG16
    # ----------------------------------------------------------

    if name == "vgg16":

        weights = (
            torchvision.models.VGG16_Weights.DEFAULT
            if pretrained
            else None
        )

        model = torchvision.models.vgg16(
            weights=weights
        )

        return model.features, 512, 16

    raise ValueError(
        f"Unsupported R-CNN backbone: {name}"
    )


# ==============================================================
# RCNN
# ==============================================================

class RCNN(nn.Module):
    """
    Basic R-CNN style detector.

    This is a custom educational/simple implementation.

    It expects proposals supplied by the dataset/training pipeline.

    Parameters
    ----------
    n_classes:
        Number of object classes including background.

    Example for one object class:

        n_classes = 2

        0 = background
        1 = storage tank
    """

    def __init__(
        self,
        n_classes=2,
        backbone="resnet50",
        pretrained=True,
        roi_pool_size=7,
        hidden_dim=1024,
    ):
        super().__init__()

        self.n_classes = n_classes
        self.roi_pool_size = roi_pool_size

        # ------------------------------------------------------
        # Backbone
        # ------------------------------------------------------

        (
            self.backbone,
            self.feat_c,
            self.stride,
        ) = _get_backbone(
            backbone,
            pretrained=pretrained,
        )

        # ------------------------------------------------------
        # ROI classifier
        # ------------------------------------------------------

        flat_dim = (
            self.feat_c
            * roi_pool_size
            * roi_pool_size
        )

        self.fc1 = nn.Linear(
            flat_dim,
            hidden_dim,
        )

        self.fc2 = nn.Linear(
            hidden_dim,
            hidden_dim,
        )

        # Classification
        self.cls_head = nn.Linear(
            hidden_dim,
            n_classes,
        )

        # Bounding box regression
        self.reg_head = nn.Linear(
            hidden_dim,
            n_classes * 4,
        )

    # ==========================================================
    # FEATURE EXTRACTION
    # ==========================================================

    def extract_features(self, x):
        return self.backbone(x)

    # ==========================================================
    # ROI PROCESSING
    # ==========================================================

    def roi_features(
        self,
        features,
        proposals,
    ):

        pooled = roi_align(
            features,
            proposals,
            output_size=self.roi_pool_size,
            spatial_scale=1.0 / self.stride,
            sampling_ratio=2,
            aligned=True,
        )

        pooled = pooled.flatten(1)

        h = F.relu(
            self.fc1(pooled)
        )

        h = F.relu(
            self.fc2(h)
        )

        return h

    # ==========================================================
    # FORWARD
    # ==========================================================

    def forward(
        self,
        images,
        proposals,
    ):
        """
        Parameters
        ----------
        images:
            Tensor [B, 3, H, W]

        proposals:
            Tensor [N, 5]

            Format:

                [image_index, x1, y1, x2, y2]

        Returns
        -------
        cls_logits:
            [N, n_classes]

        bbox_deltas:
            [N, n_classes * 4]
        """

        features = self.extract_features(images)

        h = self.roi_features(
            features,
            proposals,
        )

        cls_logits = self.cls_head(h)

        bbox_deltas = self.reg_head(h)

        return cls_logits, bbox_deltas
        

# ==============================================================
# FAST R-CNN
# ==============================================================

class FastRCNN(RCNN):
    """
    Fast R-CNN style detector.

    Adds convolutional refinement before ROI pooling.
    """

    def __init__(
        self,
        n_classes=2,
        backbone="resnet50",
        pretrained=True,
        roi_pool_size=7,
        hidden_dim=1024,
    ):

        super().__init__(
            n_classes=n_classes,
            backbone=backbone,
            pretrained=pretrained,
            roi_pool_size=roi_pool_size,
            hidden_dim=hidden_dim,
        )

        self.roi_conv = nn.Sequential(
            nn.Conv2d(
                self.feat_c,
                self.feat_c,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                self.feat_c
            ),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        images,
        proposals,
    ):

        features = self.extract_features(
            images
        )

        # ------------------------------------------------------
        # ROI pooling
        # ------------------------------------------------------

        pooled = roi_align(
            features,
            proposals,
            output_size=self.roi_pool_size,
            spatial_scale=1.0 / self.stride,
            sampling_ratio=2,
            aligned=True,
        )

        # ------------------------------------------------------
        # ROI refinement
        # ------------------------------------------------------

        pooled = self.roi_conv(
            pooled
        )

        pooled = pooled.flatten(1)

        h = F.relu(
            self.fc1(pooled)
        )

        h = F.relu(
            self.fc2(h)
        )

        cls_logits = self.cls_head(h)

        bbox_deltas = self.reg_head(h)

        return cls_logits, bbox_deltas