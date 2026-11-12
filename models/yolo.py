"""YOLO backend compatibility module for Braein.ai.

GOVERNANCE NOTE (IGA / NCSC)
----------------------------
ultralytics is AGPL-3.0. AGPL is network copyleft: serving software that
links it over a network can oblige disclosure of the ENTIRE application
source to its users. The import is therefore routed through
deployment_config.require_yolo(), which refuses unless ENABLE_YOLO=true is
set deliberately.

This does not affect the U-Net workflow - models are .pth checkpoints
and never enter this path.
"""


def build_yolo(model_name="yolo11n.pt", pretrained=True, **kwargs):
    try:
        import deployment_config as depcfg
        YOLO = depcfg.require_yolo()
    except ImportError:
        # Running outside the app (e.g. a standalone script). Fail closed
        # rather than silently pulling in an AGPL dependency.
        raise RuntimeError(
            "YOLO is gated by deployment_config.require_yolo(); "
            "deployment_config.py was not importable."
        )

    model = YOLO(model_name)
    try:
        model.geoai_model_type = "yolo"
        model.geoai_architecture = "UltralyticsYOLO"
    except Exception:
        pass
    return model


__all__ = ["build_yolo"]
