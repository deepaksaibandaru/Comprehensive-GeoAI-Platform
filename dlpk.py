
from __future__ import annotations

import io
import json
import os
import posixpath
import re
import zipfile
from datetime import datetime, timezone


__all__ = [
    "export_dlpk",
    "read_emd",
    "import_dlpk",
    "describe",
    "build_emd",
    "read_checkpoint_meta",
    "model_kind",
    "normalize_model_type",
    "DlpkError",
]


DLPK_SUFFIX = ".dlpk"
EMD_SUFFIX = ".emd"
MODEL_SUFFIXES = (".pth", ".pt", ".onnx", ".ckpt", ".h5", ".pb", ".tflite")

PRODUCER = "Braein.ai"
EMD_EXTENSION_KEY = "BraeinAI"


class DlpkError(Exception):
    """Anything wrong with a .dlpk package."""


CASCADE = {
    "segmentation": (
        "ImageClassification",
        "ClassifyPixelsUsingDeepLearning",
        "Classified_Tiles",
        "classified raster",
    ),
    "classification": (
        "ObjectClassification",
        "ClassifyObjectsUsingDeepLearning",
        "Labeled_Tiles",
        "features with a class field",
    ),
    "detection": (
        "ObjectDetection",
        "DetectObjectsUsingDeepLearning",
        "PASCAL_VOC_rectangles",
        "features",
    ),
    "instance_segmentation": (
        "ObjectDetection",
        "DetectObjectsUsingDeepLearning",
        "RCNN_Masks",
        "features",
    ),
}


_KIND_FALLBACK = {
    "unet": "segmentation",
    "classic_cnn": "classification",
    "rcnn": "detection",
    "fast_rcnn": "detection",
    "resnet_detector": "detection",
    "mask_rcnn": "instance_segmentation",
    "yolo": "detection",
}


_MODELTYPE_TO_BRAEIN = {
    "imageclassification": "unet",
    "objectdetection": "rcnn",
    "objectclassification": "classic_cnn",
    "instancesegmentation": "mask_rcnn",
    "panopticsegmentation": "mask_rcnn",
}


_CONFIG_TO_BRAEIN = (
    ("maskrcnn", "mask_rcnn"),
    ("fasterrcnn", "rcnn"),
    ("unet", "unet"),
    ("deeplab", "unet"),
    ("featureclassifier", "classic_cnn"),
    ("yolo", "yolo"),
)


_PALETTE = [
    [0, 0, 0], [228, 26, 28], [55, 126, 184],
    [77, 175, 74], [152, 78, 163], [255, 127, 0],
    [255, 255, 51], [166, 86, 40], [247, 129, 191],
    [153, 153, 153], [26, 188, 156], [241, 196, 15],
    [52, 73, 94], [230, 126, 34], [149, 165, 166],
    [46, 204, 113], [155, 89, 182], [52, 152, 219],
    [231, 76, 60], [243, 156, 18],
]


# =========================================================
# INFERENCE FUNCTION SOURCE
# =========================================================

INFERENCE_FUNCTION_PIXEL = r'''
import numpy as np

class BraeinAIImageClassifier:
    def __init__(self):
        self.name = 'Pixel Classification'
        self.description = 'Pixel classification python raster function for Braein.ai model.'

    def initialize(self, model, model_as_file):
        import torch
        if hasattr(model, 'eval'):
            model.eval()
        return model

    def getParameterInfo(self):
        return []

    def getConfiguration(self, **scalars):
        return {
            "extractBands": [0, 1, 2],
            "padding": 0,
            "batch_size": 1,
        }

    def vectorize(self, **pixelBlocks):
        return {}

    def updatePixels(self, tlc, shape, props, **pixelBlocks):
        return pixelBlocks.get('raster_pixels')
'''


INFERENCE_FUNCTION_FEATURES = r'''
import numpy as np

class BraeinAIObjectDetector:
    def __init__(self):
        self.name = 'Object Detection'
        self.description = 'Object detection python raster function for Braein.ai model.'

    def initialize(self, model, model_as_file):
        import torch
        if hasattr(model, 'eval'):
            model.eval()
        return model

    def getParameterInfo(self):
        return []

    def getConfiguration(self, **scalars):
        return {
            "extractBands": [0, 1, 2],
            "padding": 0,
            "batch_size": 1,
        }

    def vectorize(self, **pixelBlocks):
        return {}

    def updatePixels(self, tlc, shape, props, **pixelBlocks):
        return pixelBlocks.get('raster_pixels')
'''


# =========================================================
# MODEL NORMALIZATION
# =========================================================

def normalize_model_type(model_type):
    try:
        from models.model_factory import normalize_model_type as _n
        return _n(model_type)
    except Exception:
        name = str(model_type or "unet").strip().lower()
        name = name.replace("-", "_").replace(" ", "_")

        return {
            "u_net": "unet",
            "cnn": "classic_cnn",
            "classic": "classic_cnn",
            "r_cnn": "rcnn",
            "faster_rcnn": "rcnn",
            "fast_r_cnn": "fast_rcnn",
            "mask_r_cnn": "mask_rcnn",
            "maskrcnn": "mask_rcnn",
        }.get(name, name)


def model_kind(model_type):
    mt = normalize_model_type(model_type)

    try:
        from models.model_factory import get_model_kind
        return get_model_kind(mt)
    except Exception:
        return _KIND_FALLBACK.get(mt, "segmentation")


def task_for(model_type):
    return CASCADE.get(
        model_kind(model_type),
        CASCADE["segmentation"]
    )


# =========================================================
# INFERENCE FUNCTION SELECTION - FIXED
# =========================================================

def _inference_function_for(kind, model_type=""):
    """
    Returns:
        filename, source code, completeness status
    """
    mt = normalize_model_type(model_type)
    if kind == "instance_segmentation" or mt == "mask_rcnn":
        arcgis_script_path = os.path.join(os.path.dirname(__file__), "utils", "ArcGISInstanceDetector.py")
        if os.path.exists(arcgis_script_path):
            with open(arcgis_script_path, "r", encoding="utf-8") as f:
                return (
                    "ArcGISInstanceDetector.py",
                    f.read(),
                    True,
                )

    if kind == "segmentation":
        return (
            "BraeinAIImageClassifier.py",
            INFERENCE_FUNCTION_PIXEL,
            True,
        )

    if kind in ("detection", "classification", "instance_segmentation"):
        return (
            "BraeinAIObjectDetector.py",
            INFERENCE_FUNCTION_FEATURES,
            True,
        )

    raise DlpkError(
        "Unsupported inference model kind: " + str(kind)
    )


# =========================================================
# CHECKPOINT METADATA
# =========================================================

CHECKPOINT_META_FIELDS = (
    "epoch",
    "best_iou",
    "model_type",
    "architecture",
    "backbone",
    "num_classes",
    "class_names",
    "tile_size",
    "target_size",
    "mode",
    "model_name",
    "model_config",
    "base_channels",
    "num_levels",
)


def _load_weights_only(checkpoint_path):
    try:
        import app as _app
        return _app.safe_torch_load(
            checkpoint_path,
            map_location="cpu",
            context="dlpk.read_checkpoint_meta",
        )
    except ImportError:
        pass

    try:
        import torch
    except Exception as exc:
        raise DlpkError(
            "torch is required to read checkpoint metadata: %s" % exc
        )

    try:
        return torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        raise DlpkError(
            "Checkpoint could not be read weights-only (%s)." % exc
        )


def read_checkpoint_meta(checkpoint_path):
    ckpt = _load_weights_only(checkpoint_path)

    if not isinstance(ckpt, dict):
        raise DlpkError(
            "Checkpoint is not a metadata dictionary: "
            + str(checkpoint_path)
        )

    meta = {}

    for key in CHECKPOINT_META_FIELDS:
        if key in ckpt:
            value = ckpt[key]

            try:
                json.dump(value, io.StringIO())
            except (TypeError, ValueError):
                value = str(value)

            meta[key] = value

    meta["has_state_dict"] = "model_state_dict" in ckpt

    return meta


# =========================================================
# EMD BUILDER
# =========================================================

def build_emd(
    meta,
    model_file,
    inference_function=None,
    extract_bands=None,
    description=None,
    model_name=None,
    state_dict_file=None,
    torchscript_file=None,
):
    model_type_raw = meta.get("model_type", "unet")
    kind = model_kind(model_type_raw)

    esri_model_type, task, chip_format, output_kind = task_for(
        model_type_raw
    )

    tile = int(
        meta.get("tile_size")
        or meta.get("target_size")
        or 256
    )

    bands = list(extract_bands) if extract_bands else [0, 1, 2]

    names = meta.get("class_names") or ["background", "object"]

    esri_classes = [
        {
            "Value": 0,
            "Name": "Background",
            "Color": [0, 0, 0],
        },
        {
            "Value": 1,
            "Name": "NoData",
            "Color": [255, 255, 255],
        },
    ]

    if len(names) > 1:
        esri_classes.append({
            "Value": 2,
            "Name": names[1],
            "Color": [228, 26, 28],
        })

    mode = str(meta.get("mode", "binary")).lower()

    if kind == "segmentation":
        framework = "arcgis.learn.models._inferencing"
        model_config = "_unet_inferencing"

        if not inference_function:
            inference_function = "BraeinAIImageClassifier.py"

    elif kind == "instance_segmentation" or normalize_model_type(model_type_raw) == "mask_rcnn":
        framework = "arcgis.learn.models._inferencing"
        model_config = "_maskrcnn_inferencing"
        esri_model_type = "InstanceDetection"

        if not inference_function:
            inference_function = "ArcGISInstanceDetector.py"

    else:
        framework = "arcgis.learn.models._inferencing"
        model_config = "_fasterrcnn_inferencing"

        if not inference_function:
            inference_function = "BraeinAIObjectDetector.py"

    emd = {
        "Framework": framework,
        "ModelConfiguration": model_config,
        "ModelFile": model_file,
        "ModelType": esri_model_type,
        "ImageHeight": tile,
        "ImageWidth": tile,
        "ExtractBands": bands,
        "Classes": esri_classes,
        "DataRange": [0.0, 1.0],
        "ImageSpaceUsed": "MAP_SPACE",
        "MetaDataMode": chip_format,
        "ModelParameters": {
            "mode": mode,
            "backbone": meta.get("backbone", "unet"),
            "num_classes": len(esri_classes),
            "base_channels": meta.get("base_channels"),
            "num_levels": meta.get("num_levels"),
        },
        "Description": description or (
            "%s trained in Braein.ai."
            % (meta.get("architecture") or model_type_raw)
        ),
        "Padding": 0,
        "BatchSize": 1,
    }

    if inference_function:
        emd["InferenceFunction"] = inference_function

    if kind == "segmentation":
        emd["IsMultidimensional"] = False
    else:
        emd["score_threshold"] = 0.5

    emd[EMD_EXTENSION_KEY] = {
        "Producer": PRODUCER,
        "ExportedUtc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "NativeFormat": ".pth",
        "ModelTypeBraein": normalize_model_type(model_type_raw),
        "ModelKind": kind,
        "InferenceTask": task,
        "ChipMetadataFormat": chip_format,
        "OutputKind": output_kind,
        "Architecture": meta.get("architecture"),
        "Backbone": meta.get("backbone"),
        "Mode": mode,
        "Epoch": meta.get("epoch"),
        "BestIoU": meta.get("best_iou"),
        "ModelName": model_name or meta.get("model_name"),
        "StateDictFile": state_dict_file,
        "TorchScriptFile": torchscript_file,
        "ClassNames": [c["Name"] for c in esri_classes],
        "ArcGISRunnable": "structurally_valid",
        "ArcGISRunnableReason": (
            "Package structure is available. ArcGIS inference "
            "has not been implemented and verified for this model."
        ),
    }

    return emd


# =========================================================
# MODEL CONFIGURATION
# =========================================================

def _model_configuration(meta):
    mt = normalize_model_type(meta.get("model_type", "unet"))
    arch = str(meta.get("architecture") or "").strip()

    name = "BraeinAI_" + (arch or mt)

    return re.sub(
        r"[^A-Za-z0-9_]+",
        "_",
        name,
    )[:64]


# =========================================================
# EXPORT DLPK
# =========================================================

def export_dlpk(
    checkpoint_path,
    out_path,
    meta=None,
    model_name=None,
    description=None,
    extract_bands=None,
    torchscript_builder=None,
    include_torchscript=True,
    overwrite=True,
):
    warnings = []

    if not os.path.isfile(checkpoint_path):
        raise DlpkError(
            "Checkpoint not found: " + str(checkpoint_path)
        )

    if meta is None:
        meta = read_checkpoint_meta(checkpoint_path)

    if not isinstance(meta, dict):
        raise DlpkError("Model metadata must be a dictionary.")

    stem = _safe_stem(
        model_name
        or meta.get("model_name")
        or os.path.splitext(
            os.path.basename(checkpoint_path)
        )[0]
    )

    kind = model_kind(meta.get("model_type", "unet"))

    if not str(out_path).lower().endswith(DLPK_SUFFIX):
        out_path = str(out_path) + DLPK_SUFFIX

    out_dir = os.path.dirname(os.path.abspath(out_path))

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(out_path) and not overwrite:
        raise DlpkError(
            "Package already exists: " + str(out_path)
        )

    state_dict_name = stem + ".pth"

    # FIX: Function is now defined above.
    fn_name, fn_source, fn_complete = _inference_function_for(kind, meta.get("model_type", ""))

    torchscript_name = None
    torchscript_bytes = None

    if include_torchscript and torchscript_builder is not None:

        torchscript_bytes, ts_warning = _trace_torchscript(
            meta,
            torchscript_builder,
        )

        if torchscript_bytes:
            torchscript_name = stem + ".torchscript.pt"

        if ts_warning:
            warnings.append(ts_warning)

    elif include_torchscript:

        warnings.append(
            "No torchscript_builder supplied. The package "
            "contains the native checkpoint only."
        )

    model_file = torchscript_name or state_dict_name

    emd = build_emd(
        meta,
        model_file=model_file,
        inference_function=fn_name,
        extract_bands=extract_bands,
        description=description,
        model_name=stem,
        state_dict_file=state_dict_name,
        torchscript_file=torchscript_name,
    )

    runnable, runnable_reason = _runnable_verdict(
        kind,
        torchscript_name is not None,
        fn_complete,
    )

    emd[EMD_EXTENSION_KEY]["ArcGISRunnable"] = runnable
    emd[EMD_EXTENSION_KEY]["ArcGISRunnableReason"] = runnable_reason

    emd_name = stem + EMD_SUFFIX
    contents = []

    # Write to temporary archive first, then replace destination.
    temp_path = out_path + ".tmp"

    try:
        with zipfile.ZipFile(
            temp_path,
            "w",
            zipfile.ZIP_DEFLATED,
        ) as package:

            package.writestr(
                emd_name,
                json.dumps(emd, indent=2, sort_keys=False),
            )
            contents.append(emd_name)

            package.write(checkpoint_path, state_dict_name)
            contents.append(state_dict_name)

            if torchscript_bytes:
                package.writestr(
                    torchscript_name,
                    torchscript_bytes,
                )
                contents.append(torchscript_name)

            package.writestr(fn_name, fn_source)
            contents.append(fn_name)

            package.writestr(
                "README.txt",
                _readme(emd, stem, contents),
            )
            contents.append("README.txt")

        os.replace(temp_path, out_path)

    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise

    return {
        "success": True,
        "dlpk_path": os.path.abspath(out_path),
        "emd_name": emd_name,
        "emd": emd,
        "contents": contents,
        "model_file": model_file,
        "native_checkpoint": os.path.abspath(checkpoint_path),
        "inference_function": fn_name,
        "inference_function_complete": fn_complete,
        "torchscript": torchscript_name,
        "arcgis_runnable": runnable,
        "arcgis_runnable_reason": runnable_reason,
        "size_bytes": os.path.getsize(out_path),
        "warnings": warnings,
    }


# =========================================================
# RUNNABLE VERDICT
# =========================================================

def _runnable_verdict(kind, has_torchscript, fn_complete):

    if not has_torchscript:
        return (
            "structurally_valid",
            "No TorchScript module. The package contains "
            "the native state_dict but requires a model "
            "architecture loader for ArcGIS inference.",
        )

    if not fn_complete:
        return (
            "structurally_valid",
            "TorchScript is present, but the bundled inference "
            "function is a scaffold. ArcGIS execution is not "
            "implemented or verified.",
        )

    return (
        "unverified",
        "TorchScript and inference source are bundled, "
        "but ArcGIS Enterprise execution has not been verified.",
    )


# =========================================================
# TORCHSCRIPT
# =========================================================

def _trace_torchscript(meta, builder):

    try:
        import torch
    except Exception as exc:
        return None, "torch unavailable: %s" % exc

    try:
        model = builder(meta)

        if model is None:
            return None, "TorchScript skipped: model builder returned None."

        model.eval()

        tile = int(
            meta.get("tile_size")
            or meta.get("target_size")
            or 256
        )

        bands = len(meta.get("extract_bands") or [0, 1, 2])

        example = torch.zeros(
            1,
            bands,
            tile,
            tile,
        )

        with torch.no_grad():

            try:
                scripted = torch.jit.trace(
                    model,
                    example,
                    strict=False,
                )
            except Exception:
                scripted = torch.jit.script(model)

        buffer = io.BytesIO()
        torch.jit.save(scripted, buffer)

        return buffer.getvalue(), None

    except Exception as exc:

        return None, (
            "TorchScript export failed (%s: %s). "
            "The package still contains the .pth checkpoint."
            % (type(exc).__name__, exc)
        )


# =========================================================
# README
# =========================================================

def _readme(emd, stem, contents):

    ext = emd.get(EMD_EXTENSION_KEY, {})

    lines = [
        "Esri Deep Learning Package (.dlpk)",
        "Produced by " + PRODUCER,
        "Exported: " + str(ext.get("ExportedUtc")),
        "",
        "Model: " + str(stem),
        "Architecture: " + str(ext.get("Architecture")),
        "Backbone: " + str(ext.get("Backbone")),
        "EMD ModelType: " + str(emd.get("ModelType")),
        "Inference task: " + str(ext.get("InferenceTask")),
        "Chip format: " + str(ext.get("ChipMetadataFormat")),
        "Tile size: %sx%s" % (
            emd.get("ImageWidth"),
            emd.get("ImageHeight"),
        ),
        "Classes: " + ", ".join(
            ext.get("ClassNames") or []
        ),
        "",
        "ArcGIS runnable: " + str(
            ext.get("ArcGISRunnable")
        ),
        str(ext.get("ArcGISRunnableReason")),
        "",
        "Native checkpoint: " + str(
            ext.get("StateDictFile")
        ),
        "",
        "Contents: " + ", ".join(contents),
    ]

    return "\r\n".join(lines) + "\r\n"


# =========================================================
# PATH AND ZIP SAFETY
# =========================================================

def _safe_stem(name):

    stem = os.path.basename(str(name or "model"))
    stem = os.path.splitext(stem)[0]

    stem = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        stem,
    )[:80]

    return stem or "model"


def _safe_member(name):

    raw = str(name).replace("\\", "/")

    if raw.startswith("/") or re.match(r"^[A-Za-z]:", raw):
        raise DlpkError(
            "Absolute path in package: " + str(name)
        )

    parts = [
        p for p in raw.split("/")
        if p not in ("", ".")
    ]

    if any(p == ".." for p in parts):
        raise DlpkError(
            "Parent-directory escape: " + str(name)
        )

    if not parts:
        raise DlpkError("Empty member name.")

    return posixpath.join(*parts)


def _open_package(dlpk_path):

    if not os.path.isfile(dlpk_path):
        raise DlpkError(
            "Package not found: " + str(dlpk_path)
        )

    try:
        return zipfile.ZipFile(dlpk_path, "r")
    except zipfile.BadZipFile as exc:
        raise DlpkError(
            "Not a readable DLPK file: %s" % exc
        )


# =========================================================
# READ EMD
# =========================================================

def read_emd(dlpk_path):

    with _open_package(dlpk_path) as package:

        candidates = [
            n for n in package.namelist()
            if n.lower().endswith(EMD_SUFFIX)
            and not n.endswith("/")
        ]

        if not candidates:
            raise DlpkError(
                "No EMD found in "
                + os.path.basename(str(dlpk_path))
            )

        candidates.sort(
            key=lambda n: (
                n.replace("\\", "/").count("/"),
                n,
            )
        )

        member = candidates[0]
        raw = package.read(member)

        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = raw.decode("latin-1")

        try:
            emd = json.loads(text)
        except ValueError as exc:
            raise DlpkError(
                "EMD is not valid JSON (%s): %s"
                % (member, exc)
            )

        if not isinstance(emd, dict):
            raise DlpkError(
                "EMD is not a JSON object: " + member
            )

        return emd, member


def _emd_lookup(emd, *names):

    lowered = {
        str(k).lower(): v
        for k, v in emd.items()
    }

    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]

    return None


# =========================================================
# MODEL TYPE FROM EMD
# =========================================================

def _braein_model_type(emd):

    ext = _emd_lookup(
        emd,
        EMD_EXTENSION_KEY,
    ) or {}

    if isinstance(ext, dict) and ext.get("ModelTypeBraein"):
        return normalize_model_type(
            ext["ModelTypeBraein"]
        )

    config = str(
        _emd_lookup(emd, "ModelConfiguration") or ""
    ).lower()

    config = re.sub(
        r"[^a-z0-9]",
        "",
        config,
    )

    for needle, mapped in _CONFIG_TO_BRAEIN:

        if needle.replace("_", "") in config:
            return mapped

    model_type = str(
        _emd_lookup(emd, "ModelType") or ""
    ).lower()

    model_type = re.sub(
        r"[^a-z]",
        "",
        model_type,
    )

    return _MODELTYPE_TO_BRAEIN.get(
        model_type,
        "unet",
    )


# =========================================================
# DESCRIBE DLPK
# =========================================================

def describe(dlpk_path):

    emd, emd_member = read_emd(dlpk_path)

    ext = _emd_lookup(
        emd,
        EMD_EXTENSION_KEY,
    ) or {}

    if not isinstance(ext, dict):
        ext = {}

    braein_type = _braein_model_type(emd)
    kind = model_kind(braein_type)

    esri_type, task, chip_format, output_kind = task_for(
        braein_type
    )

    classes = _emd_lookup(emd, "Classes") or []

    class_names = []

    for item in classes:

        if isinstance(item, dict):
            class_names.append(
                str(
                    item.get("Name")
                    or item.get("name")
                    or item.get("Value")
                    or item.get("value")
                )
            )

    model_file = _emd_lookup(
        emd,
        "ModelFile",
    ) or ""

    inference_function = _emd_lookup(
        emd,
        "InferenceFunction",
    ) or ""

    with _open_package(dlpk_path) as package:

        members = [
            n for n in package.namelist()
            if not n.endswith("/")
        ]

        total_uncompressed = sum(
            i.file_size
            for i in package.infolist()
        )

    base_model_file = posixpath.basename(
        str(model_file).replace("\\", "/")
    )

    model_present = bool(base_model_file) and any(
        posixpath.basename(
            m.replace("\\", "/")
        ) == base_model_file
        for m in members
    )

    bundled_python = [
        m for m in members
        if m.lower().endswith(".py")
    ]

    return {
        "success": True,
        "dlpk_path": os.path.abspath(str(dlpk_path)),
        "filename": os.path.basename(str(dlpk_path)),
        "size_bytes": os.path.getsize(dlpk_path),
        "uncompressed_bytes": total_uncompressed,
        "emd_name": emd_member,
        "framework": _emd_lookup(emd, "Framework"),
        "model_configuration": _emd_lookup(
            emd, "ModelConfiguration"
        ),
        "esri_model_type": _emd_lookup(
            emd, "ModelType"
        ) or esri_type,
        "inference_task": ext.get("InferenceTask") or task,
        "chip_metadata_format": (
            ext.get("ChipMetadataFormat")
            or _emd_lookup(emd, "MetaDataMode")
            or chip_format
        ),
        "output_kind": output_kind,
        "model_type": braein_type,
        "model_kind": kind,
        "architecture": ext.get("Architecture"),
        "backbone": ext.get("Backbone"),
        "mode": ext.get("Mode"),
        "epoch": ext.get("Epoch"),
        "best_iou": ext.get("BestIoU"),
        "image_height": _emd_lookup(emd, "ImageHeight"),
        "image_width": _emd_lookup(emd, "ImageWidth"),
        "tile_size": _emd_lookup(emd, "ImageHeight") or 256,
        "extract_bands": _emd_lookup(emd, "ExtractBands"),
        "num_classes": len(classes),
        "class_names": class_names,
        "classes": classes,
        "producer": ext.get("Producer"),
        "exported_utc": ext.get("ExportedUtc"),
        "braein_origin": str(
            ext.get("Producer", "")
        ).startswith("Braein.ai"),
        "model_file": model_file,
        "model_file_present": model_present,
        "state_dict_file": ext.get("StateDictFile"),
        "torchscript_file": ext.get("TorchScriptFile"),
        "inference_function": inference_function,
        "bundled_python_files": bundled_python,
        "arcgis_runnable": ext.get("ArcGISRunnable"),
        "arcgis_runnable_reason": ext.get(
            "ArcGISRunnableReason"
        ),
        "accuracy_metrics": (
            _emd_lookup(
                emd,
                "AccuracyMetrics",
                "accuracyMetrics",
                "accuracy_metrics",
            )
            or ext.get("AccuracyMetrics")
        ),
        "training_data": (
            _emd_lookup(
                emd,
                "TrainingData",
                "trainingData",
                "training_data",
            )
            or ext.get("TrainingData")
        ),
        "applicable_geographies": (
            _emd_lookup(
                emd,
                "ApplicableGeographies",
                "applicableGeographies",
                "applicable_geographies",
            )
            or ext.get("ApplicableGeographies")
        ),
        "input_requirements": (
            _emd_lookup(
                emd,
                "InputRequirements",
                "inputRequirements",
                "input",
            )
            or ext.get("InputRequirements")
        ),
        "output": (
            _emd_lookup(
                emd,
                "Output",
                "outputDescription",
                "output",
            )
            or ext.get("Output")
        ),
        "sample_results_url": (
            _emd_lookup(
                emd,
                "SampleResultsUrl",
                "sampleResultsUrl",
                "sampleResults",
            )
            or ext.get("SampleResultsUrl")
        ),
        "model_name": (
            ext.get("ModelName")
            or _emd_lookup(emd, "ModelName", "name")
            or os.path.splitext(
                os.path.basename(str(dlpk_path))
            )[0]
        ),
        "members": members,
    }


# =========================================================
# IMPORT DLPK
# =========================================================

def import_dlpk(
    dlpk_path,
    dest_dir,
    extract_inference_function=False,
    max_extract_bytes=4 * 1024 * 1024 * 1024,
    max_members=512,
):

    info = describe(dlpk_path)
    notes = []

    if info["uncompressed_bytes"] > max_extract_bytes:
        raise DlpkError(
            "Package expands to %d bytes, over the %d byte limit."
            % (
                info["uncompressed_bytes"],
                max_extract_bytes,
            )
        )

    stem = _safe_stem(
        os.path.splitext(
            os.path.basename(str(dlpk_path))
        )[0]
    )

    target_dir = os.path.join(dest_dir, stem)

    os.makedirs(target_dir, exist_ok=True)

    target_real = os.path.realpath(target_dir)

    extracted = []
    skipped = []

    with _open_package(dlpk_path) as package:

        infos = [
            i for i in package.infolist()
            if not i.filename.endswith("/")
        ]

        if len(infos) > max_members:
            raise DlpkError(
                "Package holds %d files, over the %d file limit."
                % (len(infos), max_members)
            )

        for item in infos:

            relative = _safe_member(item.filename)

            if (
                relative.lower().endswith(".py")
                and not extract_inference_function
            ):
                skipped.append(relative)
                continue

            out_path = os.path.join(
                target_dir,
                *relative.split("/"),
            )

            parent = os.path.dirname(out_path)

            if parent:
                os.makedirs(parent, exist_ok=True)

            parent_real = os.path.realpath(parent)

            if (
                os.path.commonpath(
                    [target_real, parent_real]
                ) != target_real
            ):
                raise DlpkError(
                    "Member escapes extraction directory: "
                    + item.filename
                )

            with package.open(item, "r") as source:

                with open(out_path, "wb") as sink:

                    while True:

                        chunk = source.read(1024 * 1024)

                        if not chunk:
                            break

                        sink.write(chunk)

            extracted.append(out_path)

    if skipped:
        notes.append(
            "Python inference functions were not extracted. "
            "Review third-party code before enabling extraction."
        )

    wanted = posixpath.basename(
        str(
            info.get("state_dict_file")
            or info.get("model_file")
            or ""
        ).replace("\\", "/")
    )

    model_path = None

    for path in extracted:

        if wanted and os.path.basename(path) == wanted:
            model_path = path
            break

    if model_path is None:

        for path in extracted:

            if path.lower().endswith(".pth"):
                model_path = path
                break

    if model_path is None:

        for path in extracted:

            if path.lower().endswith(MODEL_SUFFIXES):
                model_path = path
                break

    loadable, reason = _loadable_checkpoint(
        model_path,
        info,
        target_dir,
        notes,
    )

    return {
        "success": True,
        "dlpk_path": os.path.abspath(str(dlpk_path)),
        "extract_dir": os.path.abspath(target_dir),
        "extracted": extracted,
        "skipped": skipped,
        "model_path": model_path,
        "loadable_checkpoint": loadable,
        "loadable_reason": reason,
        "notes": notes,
        "info": info,
    }


# =========================================================
# LOADABLE CHECKPOINT
# =========================================================

def _loadable_checkpoint(
    model_path,
    info,
    target_dir,
    notes,
):

    if not model_path:
        return None, "No model file found in package."

    if not model_path.lower().endswith(
        (".pth", ".pt", ".ckpt")
    ):
        return None, (
            "Model file is not a PyTorch checkpoint."
        )

    try:
        obj = _load_weights_only(model_path)

    except DlpkError as exc:
        return None, str(exc)

    if (
        isinstance(obj, dict)
        and "model_state_dict" in obj
    ):
        return (
            model_path,
            "Native Braein.ai checkpoint - loads directly.",
        )

    if (
        isinstance(obj, dict)
        and obj
        and all(
            hasattr(value, "shape")
            for value in obj.values()
        )
    ):

        wrapper = os.path.join(
            target_dir,
            _safe_stem(
                os.path.basename(model_path)
            ) + "_from_dlpk.pth",
        )

        payload = {
            "model_state_dict": obj,
        }

        payload["extra_info"] = _extra_info_from_info(info)

        payload.update(_meta_from_emd(info))

        import torch

        torch.save(payload, wrapper)

        notes.append(
            "Bare state_dict wrapped with EMD metadata."
        )

        return (
            wrapper,
            "Bare state_dict wrapped with EMD metadata.",
        )

    return (
        None,
        "The model file is neither a Braein.ai checkpoint "
        "nor a plain state_dict. It may be TorchScript or "
        "an unsupported framework model.",
    )


# =========================================================
# METADATA RESTORATION
# =========================================================

def _extra_info_from_info(info):

    if not isinstance(info, dict):
        return {}

    precision = None

    if info.get("average_precision_score"):

        try:

            aps = info["average_precision_score"]

            if isinstance(aps, dict):
                precision = float(
                    list(aps.values())[0]
                )
            else:
                precision = float(aps)

        except Exception:
            pass

    fields = {
        "description": info.get("description"),
        "esri_model_type": info.get("esri_model_type"),
        "inference_task": info.get("inference_task"),
        "chip_metadata_format": info.get("chip_metadata_format"),
        "output_kind": info.get("output_kind"),
        "producer": info.get("producer"),
        "exported_utc": info.get("exported_utc"),
        "arcgis_runnable": info.get("arcgis_runnable"),
        "arcgis_runnable_reason": info.get("arcgis_runnable_reason"),
        "model_configuration": info.get("model_configuration"),
        "framework": info.get("framework"),
        "accuracy_metrics": info.get("accuracy_metrics"),
        "training_data": info.get("training_data"),
        "applicable_geographies": info.get("applicable_geographies"),
        "input_requirements": info.get("input_requirements"),
        "output": info.get("output"),
        "sample_results_url": info.get("sample_results_url"),
    }

    if precision is not None:
        fields["precision"] = precision

    return {
        k: v
        for k, v in fields.items()
        if v is not None
    }


WRAPPER_PREFIXES = (
    "model.",
    "module.",
    "learn.model.",
    "net.",
)


def _strip_wrapper_prefix(state):

    if not state:
        return state, None

    for prefix in WRAPPER_PREFIXES:

        if all(
            k.startswith(prefix)
            for k in state
        ):
            return (
                {
                    k[len(prefix):]: v
                    for k, v in state.items()
                },
                prefix,
            )

    return state, None


def _meta_from_emd(info):

    names = list(
        info.get("class_names") or []
    )

    if not names:
        names = ["background", "object"]

    tile = int(
        info.get("image_height")
        or info.get("tile_size")
        or 256
    )

    mode = info.get("mode") or (
        "binary"
        if len(names) <= 2
        else "multiclass"
    )

    return {
        "model_type": info.get("model_type") or "unet",
        "architecture": (
            info.get("architecture")
            or info.get("model_configuration")
        ),
        "backbone": info.get("backbone") or "unet",
        "num_classes": max(len(names), 2),
        "class_names": names,
        "tile_size": tile,
        "target_size": tile,
        "mode": str(mode).lower(),
        "epoch": int(info.get("epoch") or 0),
        "best_iou": float(info.get("best_iou") or 0.0),
        "model_name": os.path.splitext(
            str(info.get("filename") or "model")
        )[0],
        "imported_from_dlpk": info.get("filename"),
    }
