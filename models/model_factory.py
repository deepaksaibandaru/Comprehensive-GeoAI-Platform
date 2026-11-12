# models/model_factory.py
# ============================================================
# Unified GeoAI model factory
#
# Model families:
#   semantic segmentation : U-Net, ClassicCNN
#   object detection      : Faster R-CNN (RCNN / FastRCNN UI types)
#   instance segmentation : Mask R-CNN
#
# IMPORTANT:
#   Detection and instance-segmentation models are NEVER wrapped in a
#   dense segmentation fallback. They are trained natively with the
#   torchvision detection API: model(images, targets) -> loss dict.
# ============================================================

import warnings

import torch
import torch.nn as nn
import torchvision

from .unet import (
    UNet,
    BinaryUNet,
    BinaryUNet64,
    UNet64,
    UNetWithFeatureExtractor,
    UNet64WithFeatureExtractor,
)


# ============================================================
# MODEL TYPES
# ============================================================

SEGMENTATION_MODELS = {
    "unet",
}

CLASSIFICATION_MODELS = {
    "classic_cnn",
}

DETECTION_MODELS = {
    "rcnn",
    "fast_rcnn",
    "resnet_detector",
}

INSTANCE_MODELS = {
    "mask_rcnn",
}

YOLO_MODELS = {"yolo"}

ALL_MODELS = (
    SEGMENTATION_MODELS
    | CLASSIFICATION_MODELS
    | DETECTION_MODELS
    | INSTANCE_MODELS
    | YOLO_MODELS
)

SUPPORTED_BACKBONES = {
    "unet",
    "none",
    "resnet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "mobilenet",
    "mobilenet_v2",
    "efficientnet",
    "efficientnet_b0",
    "vgg",
    "vgg16",
}


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_model_type(model_type):
    name = str(model_type or "unet").strip().lower()

    aliases = {
        "u-net": "unet",
        "u_net": "unet",
        "cnn": "classic_cnn",
        "classic": "classic_cnn",
        "r-cnn": "rcnn",
        "r_cnn": "rcnn",
        "fast-r-cnn": "fast_rcnn",
        "fast_r_cnn": "fast_rcnn",
        "fast rcnn": "fast_rcnn",
        "mask-r-cnn": "mask_rcnn",
        "mask_r_cnn": "mask_rcnn",
        "maskrcnn": "mask_rcnn",
        "resnet detector": "resnet_detector",
        "resnet-detector": "resnet_detector",
        "resnet_detector": "resnet_detector",
        "faster_rcnn": "rcnn",
        "yolov8": "yolo",
        "yolov9": "yolo",
        "yolov10": "yolo",
        "yolov11": "yolo",
        "yolov12": "yolo",
        "faster-r-cnn": "rcnn",
        "faster rcnn": "rcnn",
        "mask rcnn": "mask_rcnn",
        "yolo": "yolo",
    }

    return aliases.get(name, name)


def normalize_backbone(backbone):
    if backbone is None:
        return "unet"

    name = str(backbone).strip().lower()

    aliases = {
        "none": "unet",
        "standalone": "unet",
        "resnet": "resnet50",
        "resnet-18": "resnet18",
        "resnet-34": "resnet34",
        "resnet-50": "resnet50",
        "resnet-101": "resnet101",
        "mobilenet": "mobilenet_v2",
        "mobilenet-v2": "mobilenet_v2",
        "efficientnet": "efficientnet_b0",
        "efficientnet-b0": "efficientnet_b0",
        "vgg": "vgg16",
    }

    return aliases.get(name, name)


# ============================================================
# DETECTION HELPERS
# ============================================================

def _build_faster_rcnn(
    n_classes=2,
    backbone="resnet50",
    pretrained=True,
    min_size=256,
    max_size=1024,
):
    """Build a native torchvision Faster R-CNN detector."""

    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    backbone = normalize_backbone(backbone)

    # --------------------------------------------------------
    # ResNet-50 FPN
    # --------------------------------------------------------
    if backbone == "resnet50":
        try:
            weights = (
                torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
                if pretrained
                else None
            )

            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                weights=weights,
                weights_backbone=None,
                min_size=min_size,
                max_size=max_size,
            )

            in_features = (
                model.roi_heads.box_predictor.cls_score.in_features
            )

            model.roi_heads.box_predictor = FastRCNNPredictor(
                in_features,
                n_classes,
            )

        except Exception as e:
            print(
                f"⚠️ Faster R-CNN ResNet50 pretrained build failed: {e}"
            )

            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                weights=None,
                weights_backbone=None,
                min_size=min_size,
                max_size=max_size,
            )

            in_features = (
                model.roi_heads.box_predictor.cls_score.in_features
            )

            model.roi_heads.box_predictor = FastRCNNPredictor(
                in_features,
                n_classes,
            )

        return model

    # --------------------------------------------------------
    # ResNet 18 / 34 / 101
    # --------------------------------------------------------
    if backbone in {"resnet18", "resnet34", "resnet101"}:
        builder = getattr(torchvision.models, backbone)
        weights_enum = {
            "resnet18": torchvision.models.ResNet18_Weights,
            "resnet34": torchvision.models.ResNet34_Weights,
            "resnet101": torchvision.models.ResNet101_Weights,
        }[backbone]

        try:
            weights = weights_enum.DEFAULT if pretrained else None
            resnet = builder(weights=weights)
        except Exception as e:
            print(
                f"⚠️ {backbone} pretrained load failed: {e}"
            )
            resnet = builder(weights=None)

        return _make_custom_fpn_detector(
            resnet,
            n_classes=n_classes,
            min_size=min_size,
            max_size=max_size,
        )

    # --------------------------------------------------------
    # MobileNet V2
    # --------------------------------------------------------
    if backbone == "mobilenet_v2":
        try:
            from torchvision.models.detection.backbone_utils import (
                mobilenet_backbone,
            )

            backbone_module = mobilenet_backbone(
                "mobilenet_v2",
                weights=(
                    torchvision.models.MobileNet_V2_Weights.DEFAULT
                    if pretrained
                    else None
                ),
                fpn=True,
            )

            model = torchvision.models.detection.FasterRCNN(
                backbone_module,
                num_classes=n_classes,
                min_size=min_size,
                max_size=max_size,
            )

            return model

        except Exception as e:
            print(
                f"⚠️ MobileNetV2 detector failed: {e}"
            )

    print(
        f"⚠️ Detection backbone '{backbone}' is not supported directly; "
        f"using ResNet50 FPN."
    )

    return _build_faster_rcnn(
        n_classes=n_classes,
        backbone="resnet50",
        pretrained=pretrained,
        min_size=min_size,
        max_size=max_size,
    )


def _make_custom_fpn_detector(
    resnet,
    n_classes=2,
    min_size=256,
    max_size=1024,
):
    """Create Faster R-CNN on top of a ResNet + FPN backbone."""

    from torchvision.models._utils import IntermediateLayerGetter
    from torchvision.models.detection.backbone_utils import BackboneWithFPN
    from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

    return_layers = {
        "layer1": "0",
        "layer2": "1",
        "layer3": "2",
        "layer4": "3",
    }

    def block_out_channels(layer):
        block = layer[-1]

        if hasattr(block, "conv3"):
            return block.conv3.out_channels

        if hasattr(block, "conv2"):
            return block.conv2.out_channels

        raise RuntimeError(
            f"Could not determine output channels for {type(block).__name__}"
        )

    in_channels_list = [
        block_out_channels(resnet.layer1),
        block_out_channels(resnet.layer2),
        block_out_channels(resnet.layer3),
        block_out_channels(resnet.layer4),
    ]

    body = IntermediateLayerGetter(
        resnet,
        return_layers=return_layers,
    )

    backbone = BackboneWithFPN(
        body,
        return_layers=return_layers,
        in_channels_list=in_channels_list,
        out_channels=256,
        extra_blocks=LastLevelMaxPool(),
    )

    return torchvision.models.detection.FasterRCNN(
        backbone,
        num_classes=n_classes,
        min_size=min_size,
        max_size=max_size,
    )


def _configure_small_detector(model, tile_size):
    """
    Tune native RPN anchors for small GeoAI chips.

    The default COCO anchors are large for 64x64 chips. For small tiles
    use a compact anchor pyramid while keeping the native detector intact.
    """

    if not hasattr(model, "rpn"):
        return model

    if int(tile_size) <= 128:
        try:
            from torchvision.models.detection.rpn import AnchorGenerator

            anchor_sizes = (
                (8,),
                (16,),
                (32,),
                (64,),
                (128,),
            )

            aspect_ratios = (
                (0.5, 1.0, 2.0),
                (0.5, 1.0, 2.0),
                (0.5, 1.0, 2.0),
                (0.5, 1.0, 2.0),
                (0.5, 1.0, 2.0),
            )

            model.rpn.anchor_generator = AnchorGenerator(
                sizes=anchor_sizes,
                aspect_ratios=aspect_ratios,
            )

            print(
                f"[MODEL_FACTORY] Small-chip RPN anchors enabled: "
                f"{anchor_sizes}"
            )
        except Exception as e:
            print(
                f"⚠️ Small-chip RPN anchor configuration skipped: {e}"
            )

    return model


def _build_mask_rcnn(
    n_classes=2,
    backbone="resnet50",
    pretrained=True,
    min_size=64,
    max_size=128,
    tile_size=64,
):
    """Build a native torchvision Mask R-CNN without a dense fallback."""

    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    backbone = normalize_backbone(backbone)

    if backbone != "resnet50":
        # Build a Faster R-CNN FPN backbone for the requested ResNet and
        # attach a Mask R-CNN head to it.
        backbone_detector = _build_faster_rcnn(
            n_classes=n_classes,
            backbone=backbone,
            pretrained=pretrained,
            min_size=min_size,
            max_size=max_size,
        )

        backbone_module = backbone_detector.backbone
        model = torchvision.models.detection.MaskRCNN(
            backbone_module,
            num_classes=n_classes,
            min_size=min_size,
            max_size=max_size,
        )
        return _configure_small_detector(model, tile_size)

    # For pretrained Mask R-CNN, first load the native COCO model with its
    # native class count, then replace both task-specific heads with the
    # user's class count. This avoids the COCO num_classes mismatch.
    try:
        weights = (
            torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
            if pretrained
            else None
        )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*num_classes.*",
            )
            model = torchvision.models.detection.maskrcnn_resnet50_fpn(
                weights=weights,
                weights_backbone=None,
                min_size=min_size,
                max_size=max_size,
            )

        in_features_box = (
            model.roi_heads.box_predictor.cls_score.in_features
        )
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features_box,
            n_classes,
        )

        in_features_mask = (
            model.roi_heads.mask_predictor.conv5_mask.in_channels
        )
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask,
            256,
            n_classes,
        )

    except Exception as e:
        print(
            f"⚠️ Mask R-CNN pretrained build failed: {e}. "
            "Building from scratch."
        )

        model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            weights=None,
            weights_backbone=None,
            min_size=min_size,
            max_size=max_size,
        )

        in_features_box = (
            model.roi_heads.box_predictor.cls_score.in_features
        )
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features_box,
            n_classes,
        )

        in_features_mask = (
            model.roi_heads.mask_predictor.conv5_mask.in_channels
        )
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask,
            256,
            n_classes,
        )

    return _configure_small_detector(
        model,
        tile_size,
    )



# ============================================================
# LEGACY CLASSIC CNN CHIP CLASSIFIER (backward compatibility)
# ============================================================
class LegacyClassicCNNClassifier(nn.Module):
    """Exact module layout for old chip-classification checkpoints."""
    def __init__(self, n_channels=3, n_classes=2, base=32, dropout=0.0):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(n_channels, base, 3, padding=1),
            nn.BatchNorm2d(base), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(base, base*2, 3, padding=1),
            nn.BatchNorm2d(base*2), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(base*2, base*4, 3, padding=1),
            nn.BatchNorm2d(base*4), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(base*4, base*8, 3, padding=1),
            nn.BatchNorm2d(base*8), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.classifier = nn.Linear(base*8, max(2, int(n_classes)))
        self.geoai_model_type = "classic_cnn"
        self.geoai_architecture = "ClassicCNNClassifier"
        self.geoai_task = "classification"

    def forward(self, x):
        return self.classifier(self.dropout(self.features(x).flatten(1)))


# ============================================================
# CLASSIC CNN DENSE SEGMENTER
# ============================================================
class ClassicCNNClassifier(nn.Module):
    """
    Classic CNN implemented as a fully-convolutional semantic segmenter.
    Output is a dense logit map (B, C, H, W), never one class per chip.
    Kept under the legacy class name so imports remain compatible; new
    checkpoints record architecture=ClassicCNNSegmenter.
    """
    def __init__(self, n_classes=2, dropout=0.0, mode="binary"):
        super().__init__()
        self.mode = str(mode or "binary").lower()
        out_channels = 1 if self.mode == "binary" else max(2, int(n_classes))
        p = float(dropout or 0.0)
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 96, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(96), nn.ReLU(inplace=True),
            nn.Dropout2d(p) if p > 0 else nn.Identity(),
            nn.Conv2d(96, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(32 + 64 + 64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 1),
        )
        self.geoai_model_type = "classic_cnn"
        self.geoai_architecture = "ClassicCNNSegmenter"
        self.geoai_task = "semantic_segmentation"
        self.geoai_mode = self.mode

    def forward(self, x):
        # Keep every feature map at the input resolution so the output pixel
        # grid aligns exactly with the target mask.
        a = self.enc1(x)
        b = self.enc2(a)
        c = self.enc3(b)
        return self.fuse(torch.cat([a, b, c], dim=1))


# ============================================================
# YOLO BACKEND
# ============================================================
def build_yolo_model(model_name="yolo11n.pt", pretrained=True, **kwargs):
    """Return a real Ultralytics YOLO model when the optional backend is installed."""
    # AGPL-3.0 gate: ultralytics is network copyleft. Routed through
    # deployment_config.require_yolo(), which refuses unless ENABLE_YOLO=true.
    try:
        import deployment_config as depcfg
        YOLO = depcfg.require_yolo()
    except ImportError:
        raise RuntimeError(
            "YOLO is gated by deployment_config.require_yolo(); "
            "deployment_config.py was not importable."
        )
    except Exception as exc:
        raise RuntimeError(
            "YOLO backend unavailable: " + str(exc)
        ) from exc
    model = YOLO(model_name)
    try:
        model.geoai_model_type = "yolo"
        model.geoai_architecture = "UltralyticsYOLO"
    except Exception:
        pass
    return model

# ============================================================
# ARCHITECTURE METADATA
# ============================================================

def get_model_config(model):
    """Return a JSON-safe architecture description for checkpoints."""

    if hasattr(model, "get_config"):
        try:
            cfg = dict(model.get_config())
            cfg["model_class"] = type(model).__name__
            return cfg
        except Exception:
            pass

    model_type = normalize_model_type(
        getattr(model, "geoai_model_type", "unet")
    )

    architecture = getattr(
        model,
        "geoai_architecture",
        type(model).__name__,
    )

    return {
        "model_type": model_type,
        "architecture": architecture,
        "model_class": type(model).__name__,
    }


# ============================================================
# MAIN FACTORY
# ============================================================

def build_any_model(
    model_type="unet",
    backbone="unet",
    n_classes=1,
    mode="binary",
    pretrained=True,
    bias_init=None,
    fuse_at="bottleneck",
    support_weight=0.3,
    tile_size=256,
    dropout=0.0,
    detection_min_size=None,
    detection_max_size=None,
    real_detector=False,
    base_channels=None,
    num_levels=None,
    **kwargs,
):
    """
    Unified model constructor.

    Semantic segmentation:
        model(images) -> logits

    Native detection / instance segmentation:
        model(images, targets) -> loss dictionary during training
        model(images) -> prediction dictionaries during inference

    No detection model is converted into a dense segmentation model.
    """

    model_type = normalize_model_type(model_type)
    backbone = normalize_backbone(backbone)
    mode = str(mode or "binary").strip().lower()
    tile_size = int(tile_size or 256)

    if model_type not in ALL_MODELS:
        raise ValueError(
            f"Unknown model_type='{model_type}'. "
            f"Supported: {sorted(ALL_MODELS)}"
        )

    print(
        f"[MODEL_FACTORY] type={model_type} "
        f"backbone={backbone} "
        f"mode={mode} "
        f"classes={n_classes} "
        f"tile={tile_size} "
        f"base_channels={base_channels if base_channels is not None else 'default'} "
        f"num_levels={num_levels if num_levels is not None else 'default'}"
    )

    # ========================================================
    # U-NET
    # ========================================================

    if model_type == "unet":

        if tile_size <= 128:
            # UNet64 is intentionally fixed at three levels.
            effective_base = 32 if base_channels is None else int(base_channels)
            effective_levels = 3 if num_levels is None else int(num_levels)

            if effective_levels != 3:
                raise ValueError(
                    "UNet64 is fixed to num_levels=3 for tiles <= 128. "
                    f"Received num_levels={effective_levels}."
                )

            if mode == "binary":
                model = BinaryUNet64(
                    n_channels=3,
                    n_classes=1,
                    bilinear=True,
                    bias_init=bias_init,
                    base_channels=effective_base,
                    num_levels=3,
                )
            else:
                model = UNet64(
                    n_channels=3,
                    n_classes=n_classes,
                    bilinear=True,
                    bias_init=bias_init,
                    base_channels=effective_base,
                    num_levels=3,
                )

        else:
            effective_base = 64 if base_channels is None else int(base_channels)
            effective_levels = 4 if num_levels is None else int(num_levels)

            if effective_levels not in (3, 4):
                raise ValueError(
                    "Standard U-Net supports num_levels=3 or 4. "
                    f"Received {effective_levels}."
                )

            if mode == "binary":
                model = BinaryUNet(
                    n_channels=3,
                    n_classes=1,
                    bilinear=True,
                    bias_init=bias_init,
                    base_channels=effective_base,
                    num_levels=effective_levels,
                )
            elif backbone == "unet":
                model = UNet(
                    n_channels=3,
                    n_classes=n_classes,
                    bilinear=True,
                    bias_init=bias_init,
                    base_channels=effective_base,
                    num_levels=effective_levels,
                )
            else:
                # Existing feature-extractor implementation is kept for
                # compatibility. Its current architecture is fixed at the
                # standard 4-level topology.
                model = UNetWithFeatureExtractor(
                    n_channels=3,
                    n_classes=n_classes,
                    feature_extractor=backbone,
                    backbone=backbone,
                    bilinear=True,
                    pretrained=pretrained,
                    fuse_at=fuse_at,
                    support_weight=support_weight,
                )

        model.geoai_model_type = "unet"
        model.geoai_architecture = type(model).__name__
        return model

    # ========================================================
    # CLASSIC CNN
    # ========================================================

    if model_type == "classic_cnn":
        model = ClassicCNNClassifier(
            n_classes=max(2, int(n_classes)),
            dropout=dropout,
            mode=mode,
        )
        model.geoai_model_type = "classic_cnn"
        model.geoai_architecture = "ClassicCNNSegmenter"
        return model

    if model_type == "yolo":
        return build_yolo_model(
            model_name=kwargs.get("yolo_model", "yolo11n.pt"),
            pretrained=pretrained,
        )

    # ========================================================
    # DETECTION / INSTANCE SEGMENTATION
    # ========================================================

    if model_type in DETECTION_MODELS or model_type in INSTANCE_MODELS:

        if n_classes < 2:
            n_classes = 2

        if detection_min_size is None:
            detection_min_size = tile_size if tile_size <= 256 else 256

        if detection_max_size is None:
            detection_max_size = max(
                detection_min_size,
                tile_size * 2,
            )

        if model_type == "mask_rcnn":
            model = _build_mask_rcnn(
                n_classes=n_classes,
                backbone=backbone,
                pretrained=pretrained,
                min_size=int(detection_min_size),
                max_size=int(detection_max_size),
                tile_size=tile_size,
            )
            architecture = "MaskRCNN"

        else:
            detector_backbone = "resnet50" if model_type == "resnet_detector" else backbone
            model = _build_faster_rcnn(
                n_classes=n_classes,
                backbone=detector_backbone,
                pretrained=pretrained,
                min_size=int(detection_min_size),
                max_size=int(detection_max_size),
            )
            model = _configure_small_detector(
                model,
                tile_size,
            )
            architecture = "FasterRCNN"

        try:
            model.geoai_model_type = model_type
            model.geoai_architecture = architecture
            model.geoai_n_classes = int(n_classes)
            model.geoai_tile_size = int(tile_size)
        except Exception:
            pass

        print(
            f"[MODEL_FACTORY] native {architecture} returned; "
            f"no dense fallback"
        )

        return model

    raise RuntimeError(
        f"Unhandled model type: {model_type}"
    )


# ============================================================
# MODEL KIND HELPERS
# ============================================================

def get_model_kind(model_type):
    model_type = normalize_model_type(model_type)

    if model_type in SEGMENTATION_MODELS:
        return "segmentation"

    if model_type in CLASSIFICATION_MODELS:
        return "classification"

    if model_type in DETECTION_MODELS:
        return "detection"

    if model_type in INSTANCE_MODELS:
        return "instance_segmentation"

    if model_type in YOLO_MODELS:
        return "detection"

    raise ValueError(
        f"Unknown model type: {model_type}"
    )


def is_detection_model(model_type):
    return normalize_model_type(model_type) in DETECTION_MODELS or normalize_model_type(model_type) in YOLO_MODELS


def is_classification_model(model_type):
    return normalize_model_type(model_type) in CLASSIFICATION_MODELS


def is_instance_segmentation_model(model_type):
    return normalize_model_type(model_type) in INSTANCE_MODELS


def is_segmentation_model(model_type):
    return normalize_model_type(model_type) in SEGMENTATION_MODELS


__all__ = [
    "build_any_model",
    "get_model_config",
    "normalize_model_type",
    "normalize_backbone",
    "get_model_kind",
    "is_detection_model",
    "is_instance_segmentation_model",
    "is_segmentation_model",
    "SEGMENTATION_MODELS",
    "CLASSIFICATION_MODELS",
    "DETECTION_MODELS",
    "INSTANCE_MODELS",
    "ALL_MODELS",
    "SUPPORTED_BACKBONES",
    "YOLO_MODELS",
    "build_yolo_model",
    "ClassicCNNClassifier",
    "is_classification_model",
]
