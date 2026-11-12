"""
Real Mask R-CNN model based on torchvision.

Input during training:

    images = list[Tensor]

    targets = list[dict]

Each target contains:

    boxes
    labels
    masks
    image_id
    area
    iscrowd

Inference returns:

    boxes
    labels
    scores
    masks
"""

import torch
import torchvision

from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_Weights,
)


def _get_weights(pretrained=True):
    if pretrained:
        return MaskRCNN_ResNet50_FPN_Weights.DEFAULT

    return None


def MaskRCNN(
    n_classes=2,
    backbone="resnet50",
    pretrained=True,
    min_size=64,
    max_size=1024,
):
    """
    Build a real torchvision Mask R-CNN.

    Parameters
    ----------
    n_classes : int
        Number of classes including background.

        For one object class:

            0 = background
            1 = object

        therefore:

            n_classes = 2

    backbone : str
        Currently uses torchvision's ResNet-50-FPN implementation.

    pretrained : bool
        Use COCO pretrained weights.

    min_size / max_size:
        Image resizing constraints.
    """

    backbone = (
        backbone or "resnet50"
    ).lower()

    if backbone != "resnet50":
        raise ValueError(
            "The torchvision MaskRCNN implementation in this "
            "file currently supports backbone='resnet50'. "
            "Use resnet50 for Mask R-CNN."
        )

    weights = _get_weights(
        pretrained=pretrained
    )

    # ----------------------------------------------------------
    # IMPORTANT:
    #
    # num_classes includes background.
    #
    # For one class:
    #
    #   background = 0
    #   Mask_ST    = 1
    #
    # therefore num_classes=2
    # ----------------------------------------------------------

    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        weights=weights,
        num_classes=n_classes,
        min_size=min_size,
        max_size=max_size,
    )

    return model