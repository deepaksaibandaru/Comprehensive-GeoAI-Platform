import os, sys, json, math, time, random, uuid, shutil, zipfile, traceback, asyncio
import base64
import re
from urllib.parse import quote, unquote
from io import BytesIO
from datetime import datetime
from typing import Optional, Dict, List, Any

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageEnhance
import requests
import uvicorn

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms

from fastapi import (
    FastAPI, File, UploadFile, HTTPException, WebSocket,
    WebSocketDisconnect, Form, Request, Depends,
)
from fastapi.responses import (
    HTMLResponse, JSONResponse, FileResponse, RedirectResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import geopandas as gpd
from shapely.geometry import Polygon as ShapelyPolygon, box, mapping
from shapely.ops import unary_union

try:
    import rasterio
    from rasterio.transform import from_bounds as rio_from_bounds
    HAS_RASTERIO = True
except Exception:
    HAS_RASTERIO = False
    print("⚠️ rasterio not installed — GeoTIFF output disabled.")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.unet import (
    DoubleConv, Down, Up, OutConv,
    UNet, BinaryUNet,
    UNetWithFeatureExtractor,
    BackboneUNet, UNetSegmentation,
    get_backbone_encoder,
    DiceLoss, CombinedLoss,
    dice_loss, combined_loss,
    compute_metrics,
)

try:
    from models.model_factory import (
        build_any_model,
        get_model_config,
        normalize_model_type,
        normalize_backbone,
        get_model_kind,
        is_detection_model,
        is_instance_segmentation_model,
        SEGMENTATION_MODELS,
        DETECTION_MODELS,
        INSTANCE_MODELS,
        ALL_MODELS,
        SUPPORTED_BACKBONES,
    )
    print("✅ Loaded unified model_factory: native segmentation + native detection/instance models")
except ImportError as _mf_err:
    print(f"❌ models.model_factory import failed: {_mf_err}")
    raise

try:
    from torchmetrics.detection import MeanAveragePrecision
    HAS_TORCHMETRICS_MAP = True
except Exception:
    HAS_TORCHMETRICS_MAP = False
    print("ℹ️  torchmetrics[detection] not installed — Mask R-CNN evaluation will use IoU-based metric instead of mAP.")

import deployment_config as depcfg
import dlpk
import auth

depcfg.install_pretrained_guard()

app = FastAPI(title="GeoAI - Hybrid Segmentation", **depcfg.docs_urls())

print(depcfg.summary())

_bootstrap = auth.ensure_bootstrap_admin()
print(auth.summary())
if _bootstrap:
    _u, _p = _bootstrap
    print("\n" + "=" * 64)
    print("  FIRST RUN - administrator account created")
    print("    username: " + _u)
    print("    password: " + _p)
    print("  Shown once. Change it after signing in.")
    print("=" * 64 + "\n")

LOGIN_PUBLIC_PATHS = frozenset({"/login", "/logout", "/favicon.ico", "/api/health"})
API_KEY_ROLE = os.environ.get("API_KEY_ROLE", "admin").strip().lower()

def _wants_html(request: Request) -> bool:
    if request.method != "GET": return False
    if request.headers.get("X-Requested-With") == "XMLHttpRequest": return False
    return "text/html" in request.headers.get("accept", "")

def _machine_user(request: Request):
    if not depcfg.auth_enabled(): return None
    presented = (request.headers.get(depcfg.API_KEY_HEADER) or request.query_params.get(depcfg.API_KEY_QUERY_PARAM))
    if presented and depcfg.check_api_key(presented):
        role = auth.normalize_role(API_KEY_ROLE)
        return {"username": "api-key", "full_name": "API key client", "role": role, "role_title": auth.ROLES[role]["title"], "source": "api_key", "capabilities": auth.capabilities_for(role)}
    return None

@app.middleware("http")
async def session_auth_middleware(request: Request, call_next):
    path = request.url.path
    user = auth.read_session(request.cookies.get(auth.SESSION_COOKIE))
    if user is None: user = _machine_user(request)
    request.state.user = user

    if path in LOGIN_PUBLIC_PATHS or request.method == "OPTIONS":
        return await call_next(request)

    if user is None:
        if _wants_html(request):
            nxt = request.url.path
            if request.url.query: nxt += "?" + request.url.query
            return RedirectResponse("/login?next=" + quote(nxt, safe=""), status_code=303)
        return JSONResponse({"success": False, "error": "Not signed in", "login_url": "/login"}, status_code=401)

    return await call_next(request)

app.add_middleware(
    CORSMiddleware,
    allow_origins=depcfg.CORS_ALLOW_ORIGINS,
    allow_credentials=depcfg.CORS_ALLOW_CREDENTIALS,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

async def require_api_key(request: Request):
    user = getattr(request.state, "user", None)
    if user: return user
    if not depcfg.auth_enabled(): return None
    raise HTTPException(status_code=401, detail="Not signed in.")

def require_capability(capability: str):
    async def _check(request: Request):
        user = getattr(request.state, "user", None)
        if user is None and not depcfg.auth_enabled(): return None
        if not auth.has_capability(user, capability):
            raise HTTPException(status_code=403, detail=(f"Your role ({(user or {}).get('role_title', 'unknown')}) does not carry {capability}."))
        return user
    return _check

async def websocket_api_key_ok(ws: WebSocket) -> bool:
    if not depcfg.auth_enabled(): return True
    if auth.read_session(ws.cookies.get(auth.SESSION_COOKIE)): return True
    presented = ws.query_params.get(depcfg.API_KEY_QUERY_PARAM) or ws.headers.get(depcfg.API_KEY_HEADER)
    if depcfg.check_api_key(presented): return True
    await ws.close(code=1008, reason="Invalid or missing API key")
    return False

def _safe_path(user_path, allowed_roots=None, what="path"):
    roots = allowed_roots if allowed_roots is not None else depcfg.allowed_roots()
    try: return depcfg.resolve_within_roots(user_path, roots)
    except ValueError as exc: raise HTTPException(status_code=400, detail=str(exc) + " (" + what + ")")

def _register_checkpoint_safe_globals():
    allow = []
    for modname in ("numpy._core.multiarray", "numpy.core.multiarray"):
        try: mod = __import__(modname, fromlist=["scalar"])
        except Exception: continue
        for attr in ("scalar", "_reconstruct"):
            obj = getattr(mod, attr, None)
            if obj is not None: allow.append(obj)
        break
    allow.extend([np.ndarray, np.dtype])
    dtypes_mod = getattr(np, "dtypes", None)
    if dtypes_mod is not None: allow.extend(getattr(dtypes_mod, n) for n in dir(dtypes_mod) if n.endswith("DType"))
    try: torch.serialization.add_safe_globals(allow)
    except Exception as exc: print(f"\u2139\ufe0f  torch.serialization.add_safe_globals unavailable: {exc}")

_register_checkpoint_safe_globals()

def safe_torch_load(path, map_location="cpu", allow_full_object=None, context=""):
    try: return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as weights_only_exc: first_error = weights_only_exc
    trusted = depcfg.is_trusted_checkpoint(path)
    permitted = depcfg.ALLOW_UNSAFE_CHECKPOINT_LOAD if allow_full_object is None else bool(allow_full_object) and depcfg.ALLOW_UNSAFE_CHECKPOINT_LOAD
    if not (permitted and trusted):
        reason = []
        if not depcfg.ALLOW_UNSAFE_CHECKPOINT_LOAD: reason.append("ALLOW_UNSAFE_CHECKPOINT_LOAD is false")
        if not trusted: reason.append("file is outside TRUSTED_CHECKPOINT_ROOTS")
        msg = ("Refused to unpickle checkpoint " + str(path) + (" [" + context + "]" if context else "") + ": it is not a weights-only checkpoint and the full-object fallback is not permitted (" + "; ".join(reason) + "). Underlying error: " + str(first_error))
        try: add_log("error", "\u26d4 " + msg)
        except Exception: pass
        raise ValueError(msg)
    warning = ("UNSAFE CHECKPOINT LOAD: unpickling full-object checkpoint " + str(path) + (" [" + context + "]" if context else "") + " - this executes code contained in the file. Permitted because ALLOW_UNSAFE_CHECKPOINT_LOAD=true and the file is under TRUSTED_CHECKPOINT_ROOTS.")
    print("[SECURITY] " + warning)
    try: add_log("warning", warning)
    except Exception: pass
    return torch.load(path, map_location=map_location, weights_only=False)

for d in ["uploads", "uploads/datasets", "outputs", "models/checkpoints", "models/saved", "static/css", "static/js", "templates", "annotations", "projects", "labels", "chips", "evaluation", "dataset"]:
    os.makedirs(d, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/chips", StaticFiles(directory="chips"), name="chips")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
templates = Jinja2Templates(directory="templates")

MAP_SERVICES = depcfg.build_map_services()
DEFAULT_MAP_SERVICE = depcfg.DEFAULT_MAP_SERVICE

@app.get("/api/basemap/tile/{z}/{x}/{y}")
async def api_basemap_tile(z: int, x: int, y: int):
    from fastapi.responses import Response
    import arcgis_service
    if not arcgis_service.is_configured(): raise HTTPException(status_code=503, detail="No IGA imagery service configured (IGA_IMAGE_SERVER).")
    try: data, ctype = arcgis_service.tile(z, x, y)
    except Exception as exc: raise HTTPException(status_code=502, detail=str(exc))
    return Response(content=data, media_type=ctype, headers={"Cache-Control": "private, max-age=86400"})

@app.get("/api/arcgis/diagnostic", dependencies=[Depends(require_api_key)])
async def api_arcgis_diagnostic():
    import arcgis_service
    return JSONResponse(arcgis_service.tile_diagnostic())

@app.get("/api/arcgis/health")
async def api_arcgis_health():
    import arcgis_service
    return JSONResponse(arcgis_service.health())

@app.get("/api/arcgis/export", dependencies=[Depends(require_api_key)])
async def api_arcgis_export(xmin: float, ymin: float, xmax: float, ymax: float, px: int = 512, fmt: str = "tiff", bands: str = ""):
    import arcgis_service
    if not arcgis_service.is_configured(): return JSONResponse({"error": "ArcGIS image service not configured. Set ARCGIS_IMAGE_SERVICE."}, status_code=503)
    try:
        band_ids = [int(b) for b in bands.split(",") if b.strip()] if bands else None
        data = arcgis_service.export_chip(xmin, ymin, xmax, ymax, px=px, image_format=fmt, band_ids=band_ids)
        media = "image/tiff" if fmt.lower().startswith("tif") else "image/png"
        from fastapi.responses import Response
        return Response(content=data, media_type=media)
    except Exception as e: return JSONResponse({"error": str(e)}, status_code=502)

@app.get("/api/map-services")
async def api_map_services():
    return JSONResponse({"services": MAP_SERVICES, "default": DEFAULT_MAP_SERVICE, "residency": depcfg.basemap_residency()})

@app.get("/api/geocode", dependencies=[Depends(require_api_key)])
async def api_geocode(q: str):
    url = getattr(depcfg, "GEOCODER_URL", "")
    if not url: return JSONResponse({"error": "Geocoding is disabled in this deployment. Set GEOCODER_URL to an approved service."}, status_code=503)
    try:
        r = requests.get(url, params={"format": "json", "q": q, "limit": 1}, headers={"User-Agent": "BraeinAI/1.0"}, timeout=10)
        r.raise_for_status()
        return JSONResponse(r.json())
    except Exception as e: return JSONResponse({"error": str(e)}, status_code=502)

def compute_gsd_m_per_px(bounds, image_width_px, image_height_px):
    try: w, e, s, n = float(bounds.get("west")), float(bounds.get("east")), float(bounds.get("south")), float(bounds.get("north"))
    except Exception: return None
    if not (abs(w) < 180 and abs(e) < 180 and abs(s) < 90 and abs(n) < 90): return None
    if image_width_px <= 0 or image_height_px <= 0: return None
    mid_lat = (n + s) / 2.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians(mid_lat))
    m_per_deg_lat = 110540.0
    width_m, height_m = abs(e - w) * m_per_deg_lon, abs(n - s) * m_per_deg_lat
    gsd_x, gsd_y = width_m / float(image_width_px), height_m / float(image_height_px)
    gsd = (gsd_x + gsd_y) / 2.0
    if not (gsd > 0 and gsd < 100000): return None
    return float(gsd)

PROJECTS_FILE = "projects/projects.json"
LABELS_INDEX_FILE = "labels/index.json"
UPLOADED_DATASETS_FILE = "uploads/datasets/index.json"

def load_projects():
    if os.path.exists(PROJECTS_FILE):
        try:
            with open(PROJECTS_FILE, "r") as f: return json.load(f)
        except Exception: return []
    return []

def save_projects(projects):
    with open(PROJECTS_FILE, "w") as f: json.dump(projects, f, indent=2)

def get_project(pid):
    for p in load_projects():
        if p["id"] == pid: return p
    return None

def update_project(pid, updates):
    projects = load_projects()
    for i, p in enumerate(projects):
        if p["id"] == pid:
            projects[i].update(updates)
            projects[i]["updated"] = datetime.now().strftime("%m/%d/%Y, %I:%M:%S %p")
            save_projects(projects)
            return projects[i]
    return None

def load_labels_index():
    if os.path.exists(LABELS_INDEX_FILE):
        try:
            with open(LABELS_INDEX_FILE, "r") as f: return json.load(f)
        except Exception: return {"labels": []}
    return {"labels": []}

def save_labels_index(idx):
    with open(LABELS_INDEX_FILE, "w") as f: json.dump(idx, f, indent=2)

def get_label_path(label_name): return os.path.join("labels", label_name.lower().replace(" ", "_"))

def ensure_label_folders(label_name):
    safe = label_name.lower().replace(" ", "_")
    lp = get_label_path(safe)
    for sub in ["annotations", "chips/images", "chips/masks", "chips/masks_vis", "chips/metadata", "models", "evaluation/geojson", "evaluation/shp", "evaluation/masks", "evaluation/epoch_snapshots", "evaluation/val_images", "evaluation/val_masks", "evaluation/val_overlay", "dataset/train/images", "dataset/train/masks", "dataset/train/masks_vis", "dataset/train/masks_overlay", "dataset/val/images", "dataset/val/masks", "dataset/val/masks_vis", "dataset/val/masks_overlay", "dataset/test/images", "dataset/test/masks", "dataset/test/masks_vis", "dataset/test/masks_overlay"]:
        os.makedirs(os.path.join(lp, sub), exist_ok=True)
    return lp

def get_all_labels(): return load_labels_index().get("labels", [])

def load_uploaded_datasets():
    if os.path.exists(UPLOADED_DATASETS_FILE):
        try:
            with open(UPLOADED_DATASETS_FILE, "r") as f: return json.load(f)
        except Exception: return {"sessions": []}
    return {"sessions": []}

def save_uploaded_datasets(data):
    with open(UPLOADED_DATASETS_FILE, "w") as f: json.dump(data, f, indent=2)

def dedupe_labels(labels):
    if isinstance(labels, str): labels = [l.strip() for l in labels.split(",") if l.strip()]
    if not labels: labels = ["background"]
    seen = set(); out = []
    for l in labels:
        key = str(l).lower().replace(" ", "_")
        if key not in seen:
            seen.add(key); out.append(l)
    if "background" not in [l.lower() for l in out]: out = ["background"] + out
    else:
        bg_idx = [l.lower() for l in out].index("background")
        if bg_idx != 0:
            out.pop(bg_idx); out.insert(0, "background")
    return out

class TrainingState:
    def __init__(self):
        self.is_training, self.stop_training = False, False
        self.model_instance, self.optimizer, self.scheduler = None, None, None
        self.epoch, self.total_epochs = 0, 0
        self.train_losses, self.val_losses, self.ious = [], [], []
        self.best_map, self.best_epoch = 0.0, 0
        self.image_path, self.image_url, self.image_bounds_latlon = None, None, None
        self.image_crs, self.image_width, self.image_height = "EPSG:4326", None, None
        self.learning_rate, self.training_start_time = 0.0001, None
        self.total_annotations, self.threshold, self.annotation_data = 0, 0.5, None
        self.annotation_file_path, self.status, self.progress_percent = None, "IDLE", 0
        self.selected_label, self.current_label = None, None
        self.current_map_service, self.total_chips = "satellite", 0
        self.current_project_id, self.model_path, self.evaluation_metrics = None, None, None
        self.class_names, self.num_classes = ["background"], 1
        self.current_location, self.current_zoom = None, 2
        self.dataset_dir, self.tile_size, self.batch_size = None, 256, 4
        self.overlap, self.model_backbone, self.model_type = 0.2, "unet", "unet"
        self.model_name, self.fuse_at, self.support_weight = "", "bottleneck", 0.3
        self.mode, self.training_completed, self.readiness = "binary", False, None
        self.split_mode, self.architecture = "spatial", None

training_state = TrainingState()
active_connections, training_logs, training_tasks = [], [], {}

def add_log(level, message):
    ts = datetime.now().strftime("%H:%M:%S")
    training_logs.append({"timestamp": ts, "level": level, "message": message})
    print(f"[{ts}] [{level.upper()}] {message}")
    if len(training_logs) > 200: training_logs.pop(0)

def build_model(mode, num_classes, backbone="unet", pretrained=True, fuse_at="bottleneck", support_weight=0.3, bias_init=None, model_type="unet", tile_size=256, dropout=0.0, real_detector=False, base_channels=None, num_levels=None, detection_min_size=None, detection_max_size=None):
    model_type = normalize_model_type(model_type)
    backbone = normalize_backbone(backbone)
    tile_size = int(tile_size or 256)
    if model_type == "unet":
        if tile_size <= 128:
            if num_levels is None: num_levels = 3
            if base_channels is None: base_channels = 32
            if int(num_levels) != 3: raise ValueError("Small U-Net requires num_levels=3 for tile_size<=128.")
        else:
            if num_levels is None: num_levels = 4
            if base_channels is None: base_channels = 64
    if model_type in DETECTION_MODELS or model_type in INSTANCE_MODELS:
        real_detector = True
        if num_classes < 2: num_classes = 2
        if detection_min_size is None: detection_min_size = int(tile_size)
        if detection_max_size is None: detection_max_size = max(int(detection_min_size), int(tile_size * 2))
    model = build_any_model(model_type=model_type, backbone=backbone, n_classes=num_classes, mode=mode, pretrained=pretrained, bias_init=bias_init, fuse_at=fuse_at, support_weight=support_weight, tile_size=tile_size, dropout=dropout, real_detector=real_detector, base_channels=base_channels, num_levels=num_levels, detection_min_size=detection_min_size, detection_max_size=detection_max_size)
    return model

def compute_bias_init(fg_ratio):
    p = max(min(float(fg_ratio), 0.999), 0.001)
    return float(np.log(p / (1.0 - p)))

def debug_model_bias(model, label=""):
    try:
        if isinstance(model, BinaryUNet):
            bias = model.model.outc.conv.bias.detach().cpu()
            sig = torch.sigmoid(bias).detach().cpu()
            print(f"🔍 DEBUG [{label}] final bias = {bias.tolist()} | sigmoid = {sig.tolist()}")
            return float(bias.item())
        elif hasattr(model, "outc"):
            bias = model.outc.conv.bias.detach().cpu()
            print(f"🔍 DEBUG [{label}] final bias (multi) = {bias.tolist()}")
            return bias.tolist()
    except Exception as e: print(f"⚠️ debug_model_bias failed: {e}")
    return None

def infer_unet_arch_from_state_dict(state_dict):
    result = {"base_channels": 32, "num_levels": 3, "in_channels": 3, "out_channels": 1, "is_small_unet": True, "suggested_tile_size": 64}
    if not isinstance(state_dict, dict) or not state_dict: return result
    keys = list(state_dict.keys())
    for k in keys:
        if k.endswith("inc.double_conv.0.weight"):
            try:
                shape = tuple(state_dict[k].shape)
                result["base_channels"] = int(shape[0]); result["in_channels"] = int(shape[1])
            except Exception: pass
            break
    for k in keys:
        if k.endswith("outc.conv.weight"):
            try: result["out_channels"] = int(state_dict[k].shape[0])
            except Exception: pass
            break
        if k.endswith("outc.weight"):
            try: result["out_channels"] = int(state_dict[k].shape[0])
            except Exception: pass
            break
    has_down4 = any(".down4." in k for k in keys)
    has_up4 = any(".up4." in k for k in keys)
    result["num_levels"] = 4 if (has_down4 or has_up4) else 3
    result["is_small_unet"] = bool((not has_down4) and (not has_up4) and result["base_channels"] <= 32)
    result["suggested_tile_size"] = 64 if result["is_small_unet"] else 256
    return result

def _normalise_state_dict_keys(state_dict):
    if not isinstance(state_dict, dict): return state_dict
    out = state_dict
    while out and all(str(k).startswith("module.") for k in out.keys()): out = {str(k)[7:]: v for k, v in out.items()}
    return out

def _read_adjacent_emd_metadata(model_path):
    """Read an Esri EMD beside an extracted DLPK checkpoint, without guessing across packages."""
    parent = os.path.dirname(os.path.abspath(model_path))
    stem = os.path.splitext(os.path.basename(model_path))[0].lower()
    directories = [parent, os.path.dirname(parent)]
    for directory_index, directory in enumerate(directories):
        if not directory or not os.path.isdir(directory):
            continue
        try:
            emd_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(".emd")]
        except OSError:
            continue
        exact = [p for p in emd_files if os.path.splitext(os.path.basename(p))[0].lower() == stem]
        # In the extracted package directory, one unambiguous EMD is also safe.
        candidates = exact or (emd_files if directory_index == 0 and len(emd_files) == 1 else [])
        for path in candidates:
            try:
                with open(path, "r", encoding="utf-8-sig") as handle:
                    data = json.load(handle)
                if isinstance(data, dict):
                    return data
            except Exception:
                continue
    return {}


def _class_names_from_emd(emd):
    raw = emd.get("ClassNames") or emd.get("class_names") or emd.get("Classes") or emd.get("classes")
    if isinstance(raw, dict):
        raw = list(raw.values())
    if isinstance(raw, (list, tuple)):
        if raw and isinstance(raw[0], dict):
            rows = []
            for item in raw:
                name = item.get("Name") or item.get("name") or item.get("ClassName") or item.get("class_name")
                value = item.get("Value", item.get("value", item.get("ClassValue", item.get("class_id"))))
                if name:
                    try: rows.append((int(value), str(name)))
                    except Exception: rows.append((len(rows) + 1, str(name)))
            if rows:
                rows.sort(key=lambda x: x[0])
                names = [name for _, name in rows]
                if names and names[0].strip().lower() in ("background", "__background__"):
                    return names
                return ["background"] + names
        else:
            names = [str(v).strip() for v in raw if str(v).strip()]
            if names:
                if names[0].lower() not in ("background", "__background__"):
                    names.insert(0, "background")
                return names
    return None


_CHECKPOINT_META_CACHE = {}

def _read_dlpk_metadata(dlpk_path):
    """Read EMD metadata directly from a .dlpk ZIP archive without full extraction."""
    try:
        import zipfile
        with zipfile.ZipFile(dlpk_path, "r") as z:
            emd_files = [n for n in z.namelist() if n.lower().endswith(".emd")]
            if emd_files:
                data = json.loads(z.read(emd_files[0]).decode("utf-8-sig"))
                if isinstance(data, dict):
                    return data
    except Exception:
        pass
    return {}


def load_checkpoint(model_path, device, requested_num_classes=None):
    model_path = depcfg.resolve_within_roots(model_path, depcfg.MODEL_ROOTS + depcfg.EXPORT_ROOTS)
    if not model_path or not os.path.exists(model_path): 
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    
    if str(model_path).lower().endswith(".dlpk"):
        import dlpk
        extract_dir = os.path.join(depcfg.DLPK_IMPORT_DIR, dlpk._safe_stem(os.path.basename(model_path)))
        if os.path.isdir(extract_dir):
            pths = [os.path.join(extract_dir, f) for f in os.listdir(extract_dir) if f.lower().endswith(".pth")]
            if pths:
                model_path = pths[0]
            else:
                res = dlpk.import_dlpk(model_path, depcfg.DLPK_IMPORT_DIR)
                model_path = res["model_path"]
        else:
            res = dlpk.import_dlpk(model_path, depcfg.DLPK_IMPORT_DIR)
            model_path = res["model_path"]
    
    try: 
        ckpt = safe_torch_load(model_path, map_location="cpu", context="load_checkpoint")
    except Exception: 
        ckpt = None
        
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        if isinstance(ckpt, dict) and any(isinstance(v, torch.Tensor) for v in ckpt.values()):
            ckpt = {"model_state_dict": ckpt}
        elif str(model_path).lower().endswith(".pt"):
            try:
                YOLO = depcfg.require_yolo()
                yolo_model = YOLO(model_path)
                meta = {"num_classes": 2, "class_names": ["background", "object"], "best_iou": 0.0, "backbone": "yolo", "model_name": os.path.basename(model_path), "mode": "detection", "target_size": 640, "tile_size": 640, "fuse_at": "none", "support_weight": 0.0, "model_type": "yolo", "architecture": "UltralyticsYOLO", "model_config": {}}
                yolo_model.to(device)
                return yolo_model, meta
            except Exception as e: 
                raise ValueError(f"Not a GeoAI checkpoint and YOLO load failed: {e}") from e
        else:
            raise ValueError(f"Checkpoint {model_path} missing 'model_state_dict'")
        
    state_dict = _normalise_state_dict_keys(ckpt["model_state_dict"])
    model_type = normalize_model_type(ckpt.get("model_type", "unet"))
    
    meta = {
        "epoch": int(ckpt.get("epoch", 0) or 0), "num_classes": int(ckpt.get("num_classes", 2)),
        "class_names": ckpt.get("class_names", ["background", "object"]), "best_iou": float(ckpt.get("best_iou", 0.0)),
        "backbone": normalize_backbone(ckpt.get("backbone", "unet")), "model_name": ckpt.get("model_name", "model"),
        "mode": str(ckpt.get("mode", "binary")).lower(), "target_size": int(ckpt.get("target_size", ckpt.get("tile_size", 256))),
        "tile_size": int(ckpt.get("tile_size", 256)), "fuse_at": ckpt.get("fuse_at", "bottleneck"),
        "support_weight": float(ckpt.get("support_weight", 0.3)), "model_type": model_type,
        "architecture": ckpt.get("architecture", None), "model_config": ckpt.get("model_config", {}),
        "base_channels": ckpt.get("base_channels", None), "num_levels": ckpt.get("num_levels", None),
        "extra_info": ckpt.get("extra_info", {}), "fine_tuned_from": ckpt.get("fine_tuned_from", None)
    }
    emd_meta = _read_adjacent_emd_metadata(model_path)
    if emd_meta:
        emd_classes = _class_names_from_emd(emd_meta)
        if emd_classes:
            # EMD provides names only. Keep num_classes from the checkpoint/head
            # because it controls architecture construction and strict state loading.
            meta["class_names"] = emd_classes
        meta["extra_info"] = dict(meta.get("extra_info") or {})
        meta["extra_info"]["emd_metadata_available"] = True
        meta["extra_info"]["emd_model_type"] = emd_meta.get("ModelType") or emd_meta.get("model_type")

    has_roi_heads = any(("roi_heads." in k) or ("detector.roi_heads." in k) for k in state_dict.keys())
    has_mask_head = any(("mask_predictor." in k) or ("detector.roi_heads.mask_predictor." in k) for k in state_dict.keys())
    
    # NEW: Detect Esri RetinaNet / SSD structure based on the head keys
    has_retinanet_head = any("head.classification_head" in k for k in state_dict.keys())
    has_ssd_head = any("head.regression_head" in k for k in state_dict.keys()) and not has_retinanet_head

    is_detection_checkpoint = (model_type in DETECTION_MODELS or model_type in INSTANCE_MODELS or has_roi_heads or has_retinanet_head or has_ssd_head)

    if is_detection_checkpoint:
        if any(k.startswith("detector.") for k in state_dict.keys()):
            raise RuntimeError("This checkpoint was saved with the old _DenseFallback wrapper. Retrain.")

        # Rename so we never clobber the string 'backbone' with a module
        backbone_name = normalize_backbone(meta.get("backbone", "resnet50"))
        if backbone_name == "unet":
            backbone_name = "resnet50"

        if has_mask_head:
            model_type = "mask_rcnn"
        elif has_retinanet_head:
            model_type = "retinanet"
        elif has_ssd_head:
            model_type = "ssd"
        else:
            model_type = "rcnn"
        meta["model_type"] = model_type
        meta["architecture"] = model_type.capitalize()

        # Determine how many classes the checkpoint was trained for
        ckpt_num_classes = int(meta.get("num_classes") or 2)
        
        # NEW: If the user added new labels, we must expand the head.
        if requested_num_classes is not None and requested_num_classes > ckpt_num_classes:
            print(f"🔧 Expanding model head from {ckpt_num_classes} to {requested_num_classes} classes for fine-tuning.")
            ckpt_num_classes = requested_num_classes
            meta["num_classes"] = requested_num_classes

        # ---- RetinaNet branch ---------------------------------------
        if model_type == "retinanet":
            import torchvision
            from torchvision.models.detection.retinanet import RetinaNet
            from torchvision.models.detection.anchor_utils import AnchorGenerator
            from torchvision.models.detection.backbone_utils import BackboneWithFPN
            from torchvision.models.detection.fcos import LastLevelP6P7
            from torchvision.models import resnet50

            with depcfg.allow_pretrained_download():
                resnet_body = resnet50(weights=None, norm_layer=torch.nn.BatchNorm2d)

            ckpt_cls_weight = state_dict.get("head.classification_head.cls_logits.weight")
            if ckpt_cls_weight is not None and ckpt_cls_weight.ndim == 4:
                inferred = int(ckpt_cls_weight.shape[0]) // 9
                if inferred >= 1:
                    ckpt_num_classes = inferred
                    meta["num_classes"] = ckpt_num_classes
                    print(f"[retinanet] inferred num_classes={ckpt_num_classes} from checkpoint head")

            model_backbone = BackboneWithFPN(
                backbone=resnet_body,
                return_layers={"layer1": "0", "layer2": "1", "layer3": "2"},
                in_channels_list=[256, 512, 1024],
                out_channels=256,
                extra_blocks=LastLevelP6P7(256, 256),
            )

            anchor_generator = AnchorGenerator(
                sizes=((32, 40, 48), (64, 80, 96), (128, 160, 192), (256, 320, 384), (512, 640, 768)),
                aspect_ratios=((0.5, 1.0, 2.0),) * 5,
            )

            model = RetinaNet(
                backbone=model_backbone,
                num_classes=ckpt_num_classes,
                anchor_generator=anchor_generator,
                min_size=meta.get("tile_size", 224),
                max_size=meta.get("tile_size", 224) * 2,
            )
            model.to(device)
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            return model, meta

        # ---- R-CNN / Mask R-CNN branch ------------------------------
        model = build_model(
            mode=meta["mode"],
            num_classes=ckpt_num_classes,
            backbone=backbone_name,
            pretrained=False,
            fuse_at=meta["fuse_at"],
            support_weight=meta["support_weight"],
            bias_init=None,
            model_type=model_type,
            tile_size=meta["tile_size"],
            real_detector=True,
        )
    else:
        # ... [Keep the existing U-Net/CNN logic exactly as it is] ...
        if model_type == "classic_cnn":
            model = build_model(mode=meta["mode"], num_classes=meta["num_classes"], backbone=meta["backbone"], pretrained=False, model_type="classic_cnn", tile_size=meta["tile_size"], dropout=0.0, real_detector=False)
            meta["architecture"] = meta.get("architecture") or getattr(model, "geoai_architecture", type(model).__name__)
        else:
            detected = infer_unet_arch_from_state_dict(state_dict)
            architecture_text = str(meta.get("architecture") or "").lower()
            config = meta.get("model_config") or {}
            base_channels = config.get("base_channels", meta.get("base_channels") or detected["base_channels"])
            num_levels = config.get("num_levels", meta.get("num_levels") or detected["num_levels"])
            base_channels, num_levels = int(base_channels), int(num_levels)
            arch_is_small = ("unet64" in architecture_text or "binaryunet64" in architecture_text)
            if arch_is_small: is_small = True; num_levels = 3; base_channels = max(base_channels, 32)
            else: is_small = bool(detected["is_small_unet"] and meta["tile_size"] <= 128)
            effective_tile = 64 if is_small else meta["tile_size"]
            if effective_tile <= 128: num_levels = 3
            elif num_levels not in (3, 4): num_levels = 4
            meta.update({"base_channels": base_channels, "num_levels": num_levels, "tile_size": effective_tile, "target_size": effective_tile})
            if not meta.get("architecture"): meta["architecture"] = "BinaryUNet64" if is_small and meta["mode"] == "binary" else "UNet64" if is_small else "BinaryUNet" if meta["mode"] == "binary" else "UNet"
            model = build_model(mode=meta["mode"], num_classes=meta["num_classes"], backbone=meta["backbone"], pretrained=False, fuse_at=meta["fuse_at"], support_weight=meta["support_weight"], bias_init=None, model_type=model_type, tile_size=effective_tile, real_detector=False, base_channels=base_channels, num_levels=num_levels)
            
    # NEW: Only use strict=True if the classes match exactly
    is_strict = (requested_num_classes is None or requested_num_classes == int(ckpt.get("num_classes", 2)))
    
    # Strip common foreign prefixes if present
    for prefix in ("model.", "module.", "learn.model."):
        if all(k.startswith(prefix) for k in state_dict):
            state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
            break

    if not is_strict:
        model_state = model.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if k in model_state and model_state[k].shape == v.shape}

    try: 
        model.load_state_dict(state_dict, strict=is_strict)
    except RuntimeError as e:
        raise RuntimeError(f"Checkpoint architecture mismatch for {os.path.basename(model_path)}: {e}") from e
            
    model.to(device); model.eval()
    return model, meta

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, target_size=256, num_classes=2, strict=True):
        self.image_dir, self.mask_dir, self.target_size, self.num_classes = image_dir, mask_dir, target_size, num_classes
        candidates = [f for f in os.listdir(image_dir) if f.endswith((".png", ".jpg", ".jpeg")) and not f.endswith("_vis.png") and not f.endswith("_ov.png") and not f.endswith("_ids.png")]
        self.images = [f for f in candidates if os.path.exists(os.path.join(mask_dir, f))] if strict else candidates
    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        name = self.images[idx]
        image = Image.open(os.path.join(self.image_dir, name)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, name)).convert("L") if os.path.exists(os.path.join(self.mask_dir, name)) else Image.new("L", image.size, 0)
        image = image.resize((self.target_size, self.target_size), Image.BILINEAR)
        mask = mask.resize((self.target_size, self.target_size), Image.NEAREST)
        image = (np.array(image).astype(np.float32) / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        mask = np.array(mask).astype(np.int64)
        if self.num_classes == 2: mask = (mask > (0 if int(mask.max()) <= 1 else 127)).astype(np.int64)
        else:
            if mask.max() >= 255: mask = np.where(mask >= 255, 1, mask)
            if mask.max() >= self.num_classes: mask = np.where(mask >= self.num_classes, 0, mask)
        return torch.from_numpy(image).permute(2, 0, 1).float(), torch.from_numpy(mask).long()

class BinaryChipDataset(Dataset):
    def __init__(self, img_dir, mask_dir, target_size=256, strict=True, augment=False):
        self.img_dir, self.mask_dir, self.target_size, self.augment = img_dir, mask_dir, target_size, augment
        candidates = [f for f in os.listdir(img_dir) if f.endswith((".png", ".jpg", ".jpeg")) and not f.endswith("_vis.png") and not f.endswith("_ov.png") and not f.endswith("_ids.png")]
        self.files = [f for f in candidates if os.path.exists(os.path.join(mask_dir, f))] if strict else candidates
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        name = self.files[i]
        img = Image.open(os.path.join(self.img_dir, name)).convert("RGB").resize((self.target_size, self.target_size), Image.BILINEAR)
        mp = os.path.join(self.mask_dir, name)
        if os.path.exists(mp):
            raw = np.asarray(Image.open(mp).convert("L"))
            binary = binary_mask_foreground(raw).astype(np.uint8)
            if self.target_size < max(binary.shape):
                coverage = cv2.resize(binary.astype(np.float32), (self.target_size, self.target_size), interpolation=cv2.INTER_AREA)
                m = (coverage > 0.0).astype(np.uint8) * 255
            else: m = cv2.resize(binary, (self.target_size, self.target_size), interpolation=cv2.INTER_NEAREST).astype(np.uint8) * 255
        else: m = np.zeros((self.target_size, self.target_size), dtype=np.uint8)
        if self.augment:
            if random.random() < 0.5: img = img.transpose(Image.FLIP_LEFT_RIGHT); m = cv2.flip(m, 1)
            if random.random() < 0.5: img = img.transpose(Image.FLIP_TOP_BOTTOM); m = cv2.flip(m, 0)
            r = random.choice([0, 90, 180, 270])
            if r: img = img.rotate(r, expand=False); m = np.rot90(m, r // 90)
            if random.random() < 0.85:
                img = ImageEnhance.Brightness(img).enhance(random.uniform(0.78, 1.22))
                img = ImageEnhance.Contrast(img).enhance(random.uniform(0.78, 1.25))
                img = ImageEnhance.Color(img).enhance(random.uniform(0.75, 1.25))
                if random.random() < 0.35:
                    arr = np.asarray(img, dtype=np.float32)
                    gains = np.asarray([random.uniform(0.90, 1.10) for _ in range(3)], dtype=np.float32)
                    img = Image.fromarray(np.clip(arr * gains.reshape(1, 1, 3), 0, 255).astype(np.uint8), mode="RGB")
        m = np.ascontiguousarray(m)
        m = (m > 127).astype(np.float32)
        img = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(img).permute(2, 0, 1).float(), torch.from_numpy(m).unsqueeze(0).float()

class MaskRCNNChipDataset(Dataset):
    def __init__(self, split_dir, tile_size=256, num_classes=2):
        self.split_dir, self.tile_size, self.num_classes = split_dir, int(tile_size), int(max(2, num_classes))
        self.img_dir, self.tgt_dir, self.msk_dir = os.path.join(split_dir, "images"), os.path.join(split_dir, "instance_targets"), os.path.join(split_dir, "instance_masks")
        if not os.path.exists(self.img_dir): raise FileNotFoundError(f"MaskRCNNChipDataset: no images in {self.img_dir}")
        all_imgs = [f for f in os.listdir(self.img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")) and not f.endswith("_vis.png") and not f.endswith("_ov.png") and not f.endswith("_ids.png")]
        self.stems = [os.path.splitext(img_name)[0] for img_name in sorted(all_imgs) if os.path.exists(os.path.join(self.tgt_dir, f"{os.path.splitext(img_name)[0]}.json"))]
    def __len__(self): return len(self.stems)
    def __getitem__(self, i):
        stem = self.stems[i]
        img_path = os.path.join(self.img_dir, f"{stem}.png")
        if not os.path.exists(img_path):
            for ext in (".jpg", ".jpeg"):
                alt = os.path.join(self.img_dir, f"{stem}{ext}")
                if os.path.exists(alt): img_path = alt; break
        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size
        img_t = transforms.functional.to_tensor(img)
        with open(os.path.join(self.tgt_dir, f"{stem}.json"), "r", encoding="utf-8") as f: tgt_json = json.load(f)
        boxes, labels, masks, areas, iscrowd = [], [], [], [], []
        for inst in tgt_json.get("instances", []) or []:
            mask_file = inst.get("mask_file")
            if not mask_file: continue
            mf = str(mask_file).replace("\\", "/")
            mask_path = os.path.join(self.split_dir, mf) if mf.startswith("instance_masks/") else os.path.join(self.msk_dir, stem, os.path.basename(mf))
            if not os.path.exists(mask_path): continue
            mask_img = Image.open(mask_path).convert("L")
            if mask_img.size != (img_w, img_h): mask_img = mask_img.resize((img_w, img_h), Image.NEAREST)
            mask_np = (np.asarray(mask_img) > 127).astype(np.uint8)
            ys, xs = np.where(mask_np > 0)
            if len(xs) == 0: continue
            x1, y1, x2, y2 = max(0, min(img_w - 1, int(xs.min()))), max(0, min(img_h - 1, int(ys.min()))), max(int(xs.min()) + 1, min(img_w, int(xs.max()) + 1)), max(int(ys.min()) + 1, min(img_h, int(ys.max()) + 1))
            
            cls_id = int(inst.get("class_id", 1))
            cls_id = max(1, min(cls_id, self.num_classes - 1))
            
            boxes.append([float(x1), float(y1), float(x2), float(y2)])
            labels.append(cls_id); masks.append(mask_np); areas.append(float(np.count_nonzero(mask_np))); iscrowd.append(int(inst.get("iscrowd", 0)))
        if masks:
            target = {"boxes": torch.as_tensor(boxes, dtype=torch.float32), "labels": torch.as_tensor(labels, dtype=torch.int64), "masks": torch.as_tensor(np.stack(masks, axis=0), dtype=torch.uint8), "image_id": torch.tensor(i, dtype=torch.int64), "area": torch.as_tensor(areas, dtype=torch.float32), "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64)}
        else:
            target = {"boxes": torch.zeros((0, 4), dtype=torch.float32), "labels": torch.zeros((0,), dtype=torch.int64), "masks": torch.zeros((0, img_h, img_w), dtype=torch.uint8), "image_id": torch.tensor(i, dtype=torch.int64), "area": torch.zeros((0,), dtype=torch.float32), "iscrowd": torch.zeros((0,), dtype=torch.int64)}
        return img_t, target

def detection_collate(batch): return [b[0] for b in batch], [b[1] for b in batch]

def get_eval_transform(mode): return transforms.Compose([transforms.ToTensor()]) if mode == "binary" else transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def binary_mask_foreground(raw):
    arr = np.asarray(raw)
    if arr.size == 0: return np.zeros(arr.shape, dtype=bool)
    return arr > (0 if int(np.max(arr)) <= 32 else 127)

def compute_dataset_pos_weight(mask_dir, max_samples=200):
    if not os.path.exists(mask_dir): return 5.0, 0.0
    files = [f for f in os.listdir(mask_dir) if f.endswith((".png", ".jpg", ".jpeg")) and not f.endswith("_vis.png") and not f.endswith("_ov.png") and not f.endswith("_ids.png")]
    if not files: return 5.0, 0.0
    random.seed(42); random.shuffle(files); files = files[:max_samples]
    total_px, fg_px = 0, 0
    for fn in files:
        try:
            raw = np.array(Image.open(os.path.join(mask_dir, fn)).convert("L"))
            fg_px += int(binary_mask_foreground(raw).sum()); total_px += int(raw.size)
        except Exception: continue
    if total_px == 0: return 5.0, 0.0
    fg_ratio = fg_px / total_px
    return (8.0, 0.0) if fg_ratio <= 0 else (float(min(8.0, max(1.0, (1.0 - fg_ratio) / max(fg_ratio, 1e-6)))), float(fg_ratio))

def compute_all_chip_bounds(label_path: str) -> Dict[str, Dict[str, float]]:
    sidecar = os.path.join(label_path, "dataset", "chip_bounds.json")
    if not os.path.exists(sidecar): return {}
    try:
        with open(sidecar, "r") as f: data = json.load(f)
        if not isinstance(data, dict): return {}
        out = {}
        for name, b in data.items():
            try: out[name] = {"west": float(b["west"]), "east": float(b["east"]), "south": float(b["south"]), "north": float(b["north"]), "px": int(b.get("px", 0)), "py": int(b.get("py", 0))}
            except Exception: continue
        return out
    except Exception: return {}

def check_training_readiness(project_id, label, tile_size=256, min_objects=3, min_pos=3, min_neg=2, max_imbalance=0.95):
    checks = {}
    def mark(name, ok, level, detail): checks[name] = {"ok": ok, "level": level, "detail": detail}
    project = get_project(project_id)
    if not project:
        mark("Project", False, "fail", "Not found"); return {"ready": False, "checks": checks, "summary": {}}
    mark("Project", True, "ok", project.get("name", "unnamed"))
    bounds = training_state.image_bounds_latlon or project.get("bounds")
    mark("AOI", bool(bounds and all(k in bounds for k in ("west", "south", "east", "north"))), "ok" if bounds else "fail", f"({bounds['west']:.4f},{bounds['south']:.4f}) -> ({bounds['east']:.4f},{bounds['north']:.4f})" if bounds else "No bounds")
    img = training_state.image_path or project.get("image_path")
    mark("Imagery", bool(img and os.path.exists(img)), "ok" if img and os.path.exists(img) else "fail", f"{Image.open(img).size[0]}x{Image.open(img).size[1]} px" if img and os.path.exists(img) else "No raster")
    safe = label.lower().replace(" ", "_")
    ds = os.path.join(get_label_path(safe), "dataset")
    stats = {}
    for split in ["train", "val", "test"]:
        idir, mdir = os.path.join(ds, split, "images"), os.path.join(ds, split, "masks")
        if not os.path.exists(idir):
            stats[split] = {"total": 0, "positive": 0, "negative": 0, "fg_px": 0, "total_px": 0, "paired": 0, "unpaired": 0}; continue
        imgs = [f for f in os.listdir(idir) if f.endswith((".png", ".jpg", ".jpeg")) and not f.endswith("_vis.png") and not f.endswith("_ov.png") and not f.endswith("_ids.png")]
        msks = ([f for f in os.listdir(mdir) if f.endswith((".png", ".jpg", ".jpeg")) and not f.endswith("_vis.png") and not f.endswith("_ov.png") and not f.endswith("_ids.png")] if os.path.exists(mdir) else [])
        paired_names = set(imgs) & set(msks)
        pos = neg = fg_px = total_px = 0
        for fn in paired_names:
            m = np.array(Image.open(os.path.join(mdir, fn)).convert("L"))
            f = int(binary_mask_foreground(m).sum())
            if f > 0: pos += 1
            else: neg += 1
            fg_px += f; total_px += m.size
        stats[split] = {"total": len(paired_names), "positive": pos, "negative": neg, "fg_px": fg_px, "total_px": total_px, "paired": len(paired_names), "unpaired": len(set(imgs) ^ set(msks))}
    tr, va, te = stats["train"]["total"], stats["val"]["total"], stats["test"]["total"]
    pos_total = sum(stats[s]["positive"] for s in stats)
    mark("Training chips", tr >= 5, "ok" if tr >= 5 else "warn" if tr >= 1 else "fail", str(tr))
    mark("Positive chips", pos_total >= min_pos, "ok" if pos_total >= min_pos else "warn" if pos_total > 0 else "fail", str(pos_total))
    total_px, fg_px = sum(stats[s]["total_px"] for s in stats), sum(stats[s]["fg_px"] for s in stats)
    fg_ratio = fg_px / total_px if total_px > 0 else 0.0
    mark("Class balance", fg_ratio > 0 and fg_ratio <= max_imbalance, "ok" if fg_ratio > 0 and fg_ratio <= max_imbalance else "fail", f"{fg_ratio*100:.2f}% foreground")
    any_fail, any_warn = any(c["level"] == "fail" for c in checks.values()), any(c["level"] == "warn" for c in checks.values())
    return {"ready": not any_fail, "checks": checks, "summary": {"train": tr, "val": va, "test": te, "pos": pos_total, "neg": sum(stats[s]["negative"] for s in stats), "fg_ratio": fg_ratio, "any_warn": any_warn, "unpaired": sum(stats[s]["unpaired"] for s in stats)}}

def _client_of(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    return fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "?")

def _safe_next(raw: str) -> str:
    nxt = unquote(raw or "/")
    return "/" if not nxt.startswith("/") or nxt.startswith("//") or "\\" in nxt else nxt

def _login_page(request: Request, error: str = "", next_url: str = "/", status_code: int = 200):
    backend = auth.AUTH_BACKEND
    subtitle, note = ("Sign in with your ArcGIS Portal account", "Authenticated by ArcGIS Enterprise") if backend == "arcgis" else ("Sign in", "ArcGIS Portal account, or a local account") if backend == "both" else ("Sign in", "Local account")
    ctx = {"error": error, "next_url": next_url, "subtitle": subtitle, "backend_note": note, "first_run_user": (auth.FIRST_RUN_PASSWORD and os.environ.get("BOOTSTRAP_ADMIN", "admin")) or ""}
    try: return templates.TemplateResponse(request, "login.html", ctx, status_code=status_code)
    except TypeError:
        ctx["request"] = request; return templates.TemplateResponse("login.html", ctx, status_code=status_code)

@app.get("/login", response_class=HTMLResponse)
async def login_form(request: Request, next: str = "/"):
    if auth.read_session(request.cookies.get(auth.SESSION_COOKIE)): return RedirectResponse(_safe_next(next), status_code=303)
    return _login_page(request, next_url=_safe_next(next))

@app.post("/login")
async def login_submit(request: Request, username: str = Form(...), password: str = Form(...), next: str = Form("/")):
    client, nxt = _client_of(request), _safe_next(next)
    wait = auth.lockout_remaining(username, client)
    if wait: return _login_page(request, error=f"Too many failed attempts. Try again in {wait} seconds.", next_url=nxt, status_code=429)
    try: user = auth.authenticate(username, password)
    except RuntimeError as exc: return _login_page(request, error=str(exc), next_url=nxt, status_code=503)
    if not user:
        auth.record_failure(username, client)
        left = max(0, auth.MAX_LOGIN_ATTEMPTS - auth._attempts.get(auth._throttle_key(username, client), (0, 0))[0])
        msg = "Incorrect username or password." + (f" {left} attempt{'s' if left != 1 else ''} left." if left <= 2 else "")
        return _login_page(request, error=msg, next_url=nxt, status_code=401)
    auth.clear_failures(username, client)
    token = auth.issue_session(user["username"], user["role"], user.get("source", "local"), user.get("full_name", ""))
    if auth.FIRST_RUN_PASSWORD and user.get("source") == "local": auth.FIRST_RUN_PASSWORD = None
    resp = RedirectResponse(nxt, status_code=303)
    resp.set_cookie(auth.SESSION_COOKIE, token, max_age=auth.SESSION_HOURS * 3600, httponly=True, samesite="lax", secure=auth.SESSION_COOKIE_SECURE, path="/")
    return resp

@app.get("/logout")
async def logout(request: Request):
    resp = RedirectResponse("/login", status_code=303)
    resp.delete_cookie(auth.SESSION_COOKIE, path="/")
    return resp

@app.get("/api/whoami")
async def whoami(request: Request):
    user = getattr(request.state, "user", None)
    if not user: return JSONResponse({"authenticated": False}, status_code=401)
    return JSONResponse({"authenticated": True, "username": user["username"], "full_name": user.get("full_name"), "role": user["role"], "role_title": user.get("role_title"), "source": user.get("source"), "capabilities": user.get("capabilities", []), "expires": user.get("expires")})

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    user = getattr(request.state, "user", None) or {}
    ctx = {"api_key": "", "api_key_header": depcfg.API_KEY_HEADER, "api_key_param": depcfg.API_KEY_QUERY_PARAM, "current_user": user.get("username", ""), "current_role": user.get("role", ""), "current_role_title": user.get("role_title", ""), "capabilities": " ".join(user.get("capabilities", []))}
    try: return templates.TemplateResponse(request, "index.html", ctx)
    except TypeError:
        ctx["request"] = request; return templates.TemplateResponse("index.html", ctx)

@app.get("/api/projects", dependencies=[Depends(require_api_key)])
async def get_projects():
    try: return JSONResponse({"success": True, "projects": load_projects(), "count": len(load_projects())})
    except Exception as e: return JSONResponse({"success": False, "projects": [], "count": 0, "error": str(e)})

@app.get("/api/projects/{project_id}", dependencies=[Depends(require_api_key)])
async def get_project_by_id(project_id: str):
    p = get_project(project_id)
    return JSONResponse({"success": True, "project": p}) if p else JSONResponse({"success": False, "error": "Project not found"}, status_code=404)

@app.post("/api/projects", dependencies=[Depends(require_api_key)])
async def create_project(data: Dict):
    try:
        projects = load_projects()
        name = data.get("name", "Untitled Project")
        project_id = f"proj_{uuid.uuid4().hex[:8]}"
        labels = dedupe_labels(data.get("labels", ["background"]))
        raw_model_type = data.get("model_type_key") or data.get("model_type") or "unet"
        canonical_model_type = normalize_model_type(str(raw_model_type).strip().lower().replace(" ", "_"))
        display_model_type = data.get("model_type") or {"unet": "U-Net", "classic_cnn": "Classic CNN", "rcnn": "R-CNN", "fast_rcnn": "Fast R-CNN", "mask_rcnn": "Mask R-CNN", "yolo": "YOLO"}.get(canonical_model_type, canonical_model_type)
        model_default_task = {"unet": "Semantic Segmentation", "classic_cnn": "Classification", "rcnn": "Object Detection", "fast_rcnn": "Object Detection", "mask_rcnn": "Instance Segmentation", "yolo": "Object Detection"}.get(canonical_model_type, "Semantic Segmentation")
        new_project = {"id": project_id, "name": name, "created": datetime.now().strftime("%m/%d/%Y, %I:%M:%S %p"), "updated": datetime.now().strftime("%m/%d/%Y, %I:%M:%S %p"), "labels": labels, "annotations_count": 0, "chips_count": 0, "has_model": False, "model_path": None, "evaluation_score": None, "image_path": None, "annotation_path": None, "dataset_dir": None, "task_type": data.get("task_type", model_default_task), "dataset_type": data.get("dataset_type", "Optical Satellite Imagery"), "model_type": display_model_type, "model_type_key": canonical_model_type, "feature_extractor": data.get("feature_extractor", "unet"), "fuse_at": data.get("fuse_at", "bottleneck"), "support_weight": float(data.get("support_weight", 0.3)), "description": data.get("description", ""), "mode": data.get("mode", "binary"), "crs": data.get("crs", "EPSG:4326")}
        projects.insert(0, new_project); save_projects(projects)
        training_state.current_project_id = new_project["id"]; training_state.class_names = labels; training_state.num_classes = len(labels); training_state.mode = new_project["mode"]; training_state.model_backbone = new_project["feature_extractor"]; training_state.model_type = canonical_model_type
        return JSONResponse({"success": True, "project": new_project})
    except Exception as e:
        traceback.print_exc(); return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.delete("/api/projects/{project_id}", dependencies=[Depends(require_api_key)])
async def delete_project(project_id: str):
    projects = load_projects()
    if not next((x for x in projects if x["id"] == project_id), None): return JSONResponse({"success": False, "error": "Not found"}, status_code=404)
    save_projects([x for x in projects if x["id"] != project_id])
    if training_state.current_project_id == project_id: training_state.current_project_id = None
    return JSONResponse({"success": True})

@app.post("/api/projects/{project_id}/select", dependencies=[Depends(require_api_key)])
async def select_project(project_id: str):
    try:
        project = get_project(project_id)
        if not project: return JSONResponse({"success": False, "error": "Project not found"}, status_code=404)
        training_state.current_project_id = project_id; training_state.class_names = dedupe_labels(project.get("labels", ["background"])); training_state.num_classes = len(training_state.class_names); training_state.mode = project.get("mode", "binary"); training_state.model_backbone = project.get("feature_extractor", "unet"); training_state.model_type = project.get("model_type_key", "unet"); training_state.fuse_at = project.get("fuse_at", "bottleneck"); training_state.support_weight = project.get("support_weight", 0.3); training_state.image_crs = project.get("crs", "EPSG:4326")
        if project.get("image_path") and os.path.exists(project["image_path"]):
            training_state.image_path = project["image_path"]; training_state.image_url = f"/uploads/{os.path.basename(project['image_path'])}"
            w, h = Image.open(project["image_path"]).size; training_state.image_width, training_state.image_height = w, h
            if project.get("bounds"): training_state.image_bounds_latlon = project["bounds"]
        all_annotations = []
        if os.path.exists("annotations"):
            for fname in os.listdir("annotations"):
                if not fname.endswith(".geojson"): continue
                fp = os.path.join("annotations", fname)
                try:
                    with open(fp, "r") as f: d = json.load(f)
                    data_pid = d.get("project_id")
                    if data_pid and data_pid != project_id: continue
                    lbl = d.get("label")
                    if not lbl or lbl == "unknown":
                        feats = d.get("features", [])
                        if feats: lbl = feats[0].get("properties", {}).get("class_name") or feats[0].get("properties", {}).get("label")
                    all_annotations.append({"filename": fname, "filepath": fp, "label": lbl or "unlabeled", "features": d.get("features", []), "count": len(d.get("features", [])), "modified": datetime.fromtimestamp(os.path.getmtime(fp)).strftime("%Y-%m-%d %H:%M:%S"), "project_id": data_pid or "", "bounds": d.get("bounds")})
                except Exception: pass
        if all_annotations:
            all_annotations.sort(key=lambda x: x["count"], reverse=True)
            best = all_annotations[0]
            training_state.annotation_data = {"features": best["features"]}; training_state.annotation_file_path = best["filepath"]; training_state.total_annotations = best["count"]
            training_state.current_label = best["label"] if best["label"] != "unlabeled" else (next((l for l in training_state.class_names if l != "background"), "st"))
            training_state.selected_label = training_state.current_label
        return JSONResponse({"success": True, "project": project, "annotations_count": training_state.total_annotations, "current_project_id": project_id, "model_type_key": project.get("model_type_key", "unet"), "available_annotations": [{"filename": a["filename"], "count": a["count"], "label": a["label"]} for a in all_annotations]})
    except Exception as e:
        traceback.print_exc(); return JSONResponse({"success": False, "error": str(e)}, status_code=500)

def _annotation_model_key(project, data=None):
    data = data or {}
    return normalize_model_type(str(project.get("model_type_key") or project.get("model_type") or data.get("model_type_key") or data.get("model_type") or training_state.model_type or "unet").strip().lower().replace(" ", "_"))

def _geojson_bbox_feature(feature):
    geometry = (feature or {}).get("geometry") or {}; coords = []
    def walk(v):
        if not isinstance(v, (list, tuple)): return
        if len(v) >= 2:
            try:
                x, y = float(v[0]), float(v[1])
                if math.isfinite(x) and math.isfinite(y): coords.append((x, y)); return
            except Exception: pass
        for item in v: walk(item)
    walk(geometry.get("coordinates", []))
    if not coords: return None
    xs, ys = [p[0] for p in coords], [p[1] for p in coords]
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    if maxx <= minx or maxy <= miny: return None
    props = dict(feature.get("properties") or {}); props.update({"target_type": "boxes_labels", "annotation_geometry": "bounding_box", "shape_type": "rectangle", "bbox_lonlat": [minx, miny, maxx, maxy], "source_geometry_type": geometry.get("type")})
    return {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[[minx,miny],[maxx,miny],[maxx,maxy],[minx,maxy],[minx,miny]]]}, "properties": props}

def _canonicalize_annotation_features(features, model_type):
    if normalize_model_type(model_type) not in {"rcnn", "fast_rcnn", "yolo"}: return features
    return [c for c in (_geojson_bbox_feature(f) for f in features or []) if c is not None]

@app.post("/api/projects/{project_id}/annotations", dependencies=[Depends(require_api_key)])
async def save_project_annotations(project_id: str, data: Dict):
    try:
        features, label = data.get("features", []), data.get("label", training_state.current_label or "object")
        if not features: return JSONResponse({"success": False, "error": "No features"}, status_code=400)
        project = get_project(project_id)
        if not project: return JSONResponse({"success": False, "error": "Project not found"}, status_code=404)
        model_key = _annotation_model_key(project, data)
        features = _canonicalize_annotation_features(features, model_key)
        if not features: return JSONResponse({"success": False, "error": "No valid annotation geometries after model-specific normalization"}, status_code=400)
        safe_label = label.lower().replace(" ", "_"); labels = project.get("labels", ["background"])
        if safe_label not in {l.lower().replace(" ", "_") for l in labels}: labels.append(safe_label); labels = dedupe_labels(labels); update_project(project_id, {"labels": labels})
        training_state.class_names, training_state.num_classes = labels, len(labels)
        timestamp, date_str = datetime.now().strftime("%Y%m%d_%H%M%S"), datetime.now().strftime("%Y-%m-%d")
        filename = f"{safe_label}_{project_id}_{timestamp}.geojson"
        class_id = training_state.class_names.index(safe_label) if safe_label in training_state.class_names else 1
        for feat in features:
            feat.setdefault("properties", {}).update({"class_id": class_id, "class_name": safe_label, "project_id": project_id, "label": safe_label, "saved_date": date_str, "saved_time": datetime.now().strftime("%H:%M:%S")})
            if model_key == "unet":
                if (feat.get("geometry") or {}).get("type") not in {"Polygon", "MultiPolygon"}: raise ValueError(f"U-Net annotations must be Polygon/MultiPolygon segments; received geometry type {feat.get('geometry', {}).get('type')!r}.")
                feat["properties"]["annotation_geometry"] = "polygon_segment"; feat["properties"]["target_type"] = "semantic_mask"
        annotation_target = "boxes_labels" if model_key in {"rcnn", "fast_rcnn", "yolo"} else "boxes_labels_masks" if model_key == "mask_rcnn" else "class_label" if model_key == "classic_cnn" else "semantic_mask"
        geojson_data = {"type": "FeatureCollection", "features": features, "bounds": training_state.image_bounds_latlon, "crs": training_state.image_crs, "project_id": project_id, "label": safe_label, "created": datetime.now().isoformat(), "total_features": len(features), "source": "manual_annotation", "model_type": model_key, "annotation_target": annotation_target}
        os.makedirs("annotations", exist_ok=True)
        root_path = os.path.join("annotations", filename)
        with open(root_path, "w") as f: json.dump(geojson_data, f, indent=2)
        lp = ensure_label_folders(safe_label)
        with open(os.path.join(lp, "annotations", filename), "w") as f: json.dump(geojson_data, f, indent=2)
        proj_ann = os.path.join("projects", project_id, "annotations"); os.makedirs(proj_ann, exist_ok=True)
        with open(os.path.join(proj_ann, filename), "w") as f: json.dump(geojson_data, f, indent=2)
        training_state.annotation_data, training_state.annotation_file_path, training_state.total_annotations = geojson_data, root_path, len(features)
        update_project(project_id, {"annotations_count": len(features), "annotation_path": root_path, "labels": labels})
        return JSONResponse({"success": True, "filename": filename, "geojson_path": root_path, "num_features": len(features), "label": safe_label, "project_id": project_id, "model_type": model_key, "annotation_target": annotation_target})
    except Exception as e:
        traceback.print_exc(); return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.post("/api/labels/create", dependencies=[Depends(require_api_key)])
async def create_label(data: Dict):
    try:
        label_name, project_id = (data.get("label_name") or "").strip(), data.get("project_id", training_state.current_project_id)
        if not label_name: return JSONResponse({"success": False, "error": "Label name required"}, status_code=400)
        if not project_id: return JSONResponse({"success": False, "error": "No project"}, status_code=400)
        safe = label_name.lower().replace(" ", "_"); project = get_project(project_id)
        if not project: return JSONResponse({"success": False, "error": "Project not found"}, status_code=404)
        labels = project.get("labels", ["background"])
        if safe not in {l.lower().replace(" ", "_") for l in labels}: labels.append(safe); labels = dedupe_labels(labels); update_project(project_id, {"labels": labels})
        idx = load_labels_index()
        if safe not in idx["labels"]: idx["labels"].append(safe); save_labels_index(idx)
        training_state.class_names, training_state.num_classes = labels, len(labels)
        ensure_label_folders(safe)
        return JSONResponse({"success": True, "label": safe, "all_labels": labels, "num_classes": len(labels)})
    except Exception as e:
        traceback.print_exc(); return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.get("/api/labels", dependencies=[Depends(require_api_key)])
async def list_labels_api(): return JSONResponse({"success": True, "labels": get_all_labels()})

@app.post("/api/labels/select", dependencies=[Depends(require_api_key)])
async def select_label(data: Dict):
    safe = (data.get("label_name") or "").strip().lower().replace(" ", "_")
    if not safe: return JSONResponse({"success": False, "error": "Label required"}, status_code=400)
    training_state.current_label, training_state.selected_label = safe, safe
    return JSONResponse({"success": True, "label": safe})

# --- Replace the /api/status endpoint in app.py ---
@app.get("/api/status", dependencies=[Depends(require_api_key)])
async def get_status():
    project_name = get_project(training_state.current_project_id).get("name") if training_state.current_project_id and get_project(training_state.current_project_id) else None
    projects = load_projects()
    
    total_annotations_all = 0
    if os.path.exists("annotations"):
        for fn in os.listdir("annotations"):
            if fn.endswith(".geojson"):
                try:
                    with open(os.path.join("annotations", fn), "r") as f:
                        total_annotations_all += len(json.load(f).get("features", []))
                except Exception:
                    pass

    total_chips_all = 0
    if os.path.exists("labels"):
        for label in os.listdir("labels"):
            for split in ("train", "val", "test"):
                idir = os.path.join("labels", label, "dataset", split, "images")
                if os.path.exists(idir):
                    for f in os.listdir(idir):
                        if f.endswith((".png", ".jpg", ".jpeg")) and not f.endswith(("_vis.png", "_ov.png", "_ids.png")):
                            total_chips_all += 1

    models_data = await list_models()
    models_json = json.loads(models_data.body)
    total_models_all = models_json.get("count", 0)
    
    best_iou_all = 0.0
    for m in models_json.get("models", []):
        is_dlpk = m.get("is_dlpk", False)
        # For DLPKs, try to grab precision. Otherwise use IoU.
        score = 0.0
        if is_dlpk:
            extra = m.get("extra_info", {})
            ap = extra.get("average_precision_score") or extra.get("precision")
            if isinstance(ap, dict):
                try: score = float(list(ap.values())[0])
                except: pass
            elif ap is not None:
                try: score = float(ap)
                except: pass
        if score == 0.0:
            score = float(m.get("best_iou", 0.0))
            
        best_iou_all = max(best_iou_all, score)

    return JSONResponse({
        "status": training_state.status, "is_training": training_state.is_training, 
        "epoch": training_state.epoch, "total_epochs": training_state.total_epochs, 
        "best_map": training_state.best_map, "best_iou": training_state.best_map, 
        "best_epoch": training_state.best_epoch, "image_path": training_state.image_path, 
        "annotation_file": training_state.annotation_file_path, "progress_percent": training_state.progress_percent, 
        "selected_label": training_state.selected_label, "current_label": training_state.current_label, 
        "total_chips": training_state.total_chips, "current_project_id": training_state.current_project_id, 
        "total_annotations": training_state.total_annotations, "train_losses": training_state.train_losses, 
        "losses": training_state.train_losses, "val_losses": training_state.val_losses, 
        "ious": training_state.ious, "num_classes": training_state.num_classes, 
        "class_names": training_state.class_names, "project_name": project_name, 
        "model_path": training_state.model_path, "model_name": training_state.model_name, 
        "training_completed": training_state.training_completed, "model_backbone": training_state.model_backbone, 
        "model_type": training_state.model_type, "architecture": training_state.architecture, 
        "tile_size": training_state.tile_size, "mode": training_state.mode, 
        "bounds": training_state.image_bounds_latlon, "crs": training_state.image_crs, 
        "fuse_at": training_state.fuse_at, "support_weight": training_state.support_weight, 
        "total_annotations_all": total_annotations_all, "total_chips_all": total_chips_all, 
        "total_models_all": total_models_all, "total_projects": len(projects), 
        "total_labels_all": len(set(l.lower().replace(" ", "_") for p in projects for l in p.get("labels", []) if l and l.lower() != "background")), 
        "best_map_all": best_iou_all, "projects_count": len(projects), "models_count": total_models_all
    })


def capture_map_image(bounds: Dict, zoom: int, service_id: str = None, project_id: str = None) -> Dict:
    service = MAP_SERVICES.get(service_id) or MAP_SERVICES.get(training_state.current_map_service) or MAP_SERVICES.get(DEFAULT_MAP_SERVICE)
    if not service: raise RuntimeError("No basemap service available.")
    
    tile_url, north, south, east, west = service["url"], float(bounds["north"]), float(bounds["south"]), float(bounds["east"]), float(bounds["west"])
    
    def lon_to_tile_x(lon, z): return (lon + 180.0) / 360.0 * (2 ** z)
    def lat_to_tile_y(lat, z): return (1.0 - math.log(math.tan(math.radians(lat)) + 1.0 / math.cos(math.radians(lat))) / math.pi) / 2.0 * (2 ** z)
    def tile_x_to_lon(tx, z): return tx / (2 ** z) * 360.0 - 180.0
    def tile_y_to_lat(ty, z): return math.degrees(math.atan(math.sinh(math.pi - 2.0 * math.pi * ty / (2 ** z))))
    
    use_arcgis = bool(service.get("upstream")) or str(tile_url).startswith("/api/basemap/")
    if use_arcgis:
        import arcgis_service
        if not arcgis_service.is_configured(): raise RuntimeError("IGA imagery proxy not configured.")
        px_per_deg = 256.0 * (2 ** zoom) / 360.0
        req_w, req_h = max(16, int(round((east - west) * px_per_deg))), max(16, int(round((north - south) * px_per_deg * math.cos(math.radians((north + south) / 2.0)))))
        if max(req_w, req_h) > 4096:
            k = 4096 / float(max(req_w, req_h)); req_w, req_h = max(16, int(req_w * k)), max(16, int(req_h * k))
        try:
            raw = arcgis_service.export_image((west, south, east, north), size=(req_w, req_h), bbox_sr=4326, image_sr=4326, image_format="png")
            stitched = Image.open(BytesIO(raw)).convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"ArcGIS exportImage failed: {exc}")
        w, h, snapped_west, snapped_east, snapped_south, snapped_north = stitched.size[0], stitched.size[1], west, east, south, north
    else:
        x_start, x_end, y_start, y_end = int(math.floor(lon_to_tile_x(west, zoom))), int(math.floor(lon_to_tile_x(east, zoom))), int(math.floor(lat_to_tile_y(north, zoom))), int(math.floor(lat_to_tile_y(south, zoom)))
        tile_count = (x_end - x_start + 1) * (y_end - y_start + 1)
        if tile_count > 400: raise RuntimeError(f"Capture area too large ({tile_count} tiles). Zoom in or draw a smaller AOI.")
        snapped_west, snapped_east, snapped_north, snapped_south = tile_x_to_lon(x_start, zoom), tile_x_to_lon(x_end + 1, zoom), tile_y_to_lat(y_start, zoom), tile_y_to_lat(y_end + 1, zoom)
        tiles = []
        for x in range(x_start, x_end + 1):
            for y in range(y_start, y_end + 1):
                url = tile_url.replace("{s}", "a").replace("{z}", str(zoom)).replace("{x}", str(x)).replace("{y}", str(y))
                try:
                    r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
                    if r.status_code == 200: tiles.append({"image": Image.open(BytesIO(r.content)), "x": x, "y": y})
                except Exception: pass
        if not tiles: raise RuntimeError("No tiles downloaded")
        ts, w, h = 256, (x_end - x_start + 1) * 256, (y_end - y_start + 1) * 256
        stitched = Image.new("RGB", (w, h))
        for t in tiles: stitched.paste(t["image"], ((t["x"] - x_start) * ts, (t["y"] - y_start) * ts))
        crop_box = (max(0, min(w - 1, int(math.floor(lon_to_tile_x(west, zoom) * ts - x_start * ts)))), max(0, min(h - 1, int(math.floor(lat_to_tile_y(north, zoom) * ts - y_start * ts)))), max(1, min(w, int(math.ceil(lon_to_tile_x(east, zoom) * ts - x_start * ts)))), max(1, min(h, int(math.ceil(lat_to_tile_y(south, zoom) * ts - y_start * ts)))))
        if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]: raise RuntimeError("AOI crop empty.")
        stitched = stitched.crop(crop_box); w, h = stitched.size
        snapped_west, snapped_east, snapped_south, snapped_north = west, east, south, north
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
    filename = f"map_capture_{timestamp}.png"
    filepath = os.path.join("uploads", filename)
    os.makedirs("uploads", exist_ok=True)
    stitched.save(filepath)
    snapped_bounds = {"west": float(snapped_west), "south": float(snapped_south), "east": float(snapped_east), "north": float(snapped_north), "crs": "EPSG:4326", "zoom": int(zoom), "service": service_id, "image_width": int(w), "image_height": int(h)}
    
    training_state.image_path, training_state.image_url, training_state.image_width, training_state.image_height, training_state.image_bounds_latlon, training_state.image_crs = filepath, f"/uploads/{filename}", w, h, snapped_bounds, "EPSG:4326"
    if project_id: update_project(project_id, {"image_path": filepath, "bounds": snapped_bounds, "crs": "EPSG:4326"})
    return {"success": True, "filepath": filepath, "image_url": training_state.image_url, "width": w, "height": h, "bounds": snapped_bounds, "crs": "EPSG:4326", "zoom": zoom}

@app.post("/api/map/capture", dependencies=[Depends(require_api_key)])
async def capture_map_area(data: Dict):
    try:
        bounds, zoom, service_id, project_id = data.get("bounds"), max(15, int(data.get("zoom", 12))), data.get("service_id", training_state.current_map_service), data.get("project_id", training_state.current_project_id)
        if not bounds: return JSONResponse({"error": "No bounds"}, status_code=400)
        res = capture_map_image(bounds, zoom, service_id, project_id)
        return JSONResponse(res)
    except Exception as e:
        traceback.print_exc(); return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/map/location", dependencies=[Depends(require_api_key)])
async def save_location(data: Dict):
    training_state.current_location, training_state.current_zoom = {"lat": data.get("lat"), "lng": data.get("lng")}, data.get("zoom", 2)
    return JSONResponse({"success": True})

@app.post("/api/dataset/plan-aoi", dependencies=[Depends(require_api_key)])
async def plan_label_cluster_aois(data: Dict):
    try:
        from shapely.geometry import shape, box, mapping, MultiPolygon, Polygon
        from shapely.ops import unary_union, transform as shp_transform
        from pyproj import CRS, Transformer
        features, active_label = data.get("features") or [], str(data.get("active_label") or "").strip()
        if not features: return JSONResponse({"success": False, "error": "No label geometries supplied."}, status_code=400)
        
        do_cluster = bool(data.get("cluster", False))
        cluster_distance_m = float(data.get("cluster_distance_m", 150.0))
        aoi_buffer_m = float(data.get("aoi_buffer_m", 25.0))
        min_cluster_ann = int(data.get("min_cluster_annotations", 5))
        bounds_clip = data.get("bounds")

        valid, available_labels = [], set()
        for feature in features:
            geom_json, props = feature.get("geometry"), feature.get("properties") or {}
            feature_label = str(props.get("class_name") or props.get("label") or props.get("class") or props.get("label_name") or props.get("object_label") or "").strip()
            if feature_label: available_labels.add(feature_label)
            if active_label and feature_label and feature_label.casefold() != active_label.casefold(): continue
            if not geom_json: continue
            try:
                geom = shape(geom_json)
                if geom.is_empty: continue
                if not geom.is_valid: geom = geom.buffer(0)
                if not geom.is_empty: valid.append((geom, feature))
            except Exception: continue
        if not valid and features:
            for feature in features:
                geom_json = feature.get("geometry")
                if not geom_json: continue
                try:
                    geom = shape(geom_json)
                    if geom.is_empty: continue
                    if not geom.is_valid: geom = geom.buffer(0)
                    if not geom.is_empty: valid.append((geom, feature))
                except Exception: continue
        if not valid: return JSONResponse({"success": False, "error": f"No valid annotations found for active label '{active_label}'. Available: {', '.join(sorted(available_labels)) or 'none'}"}, status_code=400)

        # Optional bounds clipping (e.g. current map extent or user selected boundary)
        if bounds_clip:
            try:
                clip_box = box(float(bounds_clip["west"]), float(bounds_clip["south"]), float(bounds_clip["east"]), float(bounds_clip["north"]))
                valid = [(g, f) for g, f in valid if g.intersects(clip_box)]
            except Exception: pass
        if not valid: return JSONResponse({"success": False, "error": "No annotations found within the specified boundary/view."}, status_code=400)

        geoms = [g for g, _ in valid]
        bounds = [g.bounds for g in geoms]
        lon0 = sum((b[0]+b[2])/2.0 for b in bounds)/len(bounds)
        lat0 = sum((b[1]+b[3])/2.0 for b in bounds)/len(bounds)
        metric = CRS.from_proj4(f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs")
        forward = Transformer.from_crs("EPSG:4326", metric, always_xy=True).transform
        inverse = Transformer.from_crs(metric, "EPSG:4326", always_xy=True).transform

        projected = [(shp_transform(forward, g), f) for g, f in valid]

        if do_cluster:
            # Fast spatial clustering using buffer-merge
            buffered = [pg.buffer(cluster_distance_m / 2.0) for pg, _ in projected]
            merged = unary_union(buffered)

            if isinstance(merged, Polygon):
                cluster_geoms = [merged]
            elif isinstance(merged, MultiPolygon):
                cluster_geoms = list(merged.geoms)
            else:
                cluster_geoms = []

            aois = []
            for idx, cg in enumerate(cluster_geoms):
                matching = [f for pg, f in projected if cg.intersects(pg)]
                if len(matching) < min_cluster_ann:
                    continue
                matching_pgs = [pg for pg, f in projected if cg.intersects(pg)]
                m_bounds = [mg.bounds for mg in matching_pgs]
                left = min(b[0] for b in m_bounds) - aoi_buffer_m
                bottom = min(b[1] for b in m_bounds) - aoi_buffer_m
                right = max(b[2] for b in m_bounds) + aoi_buffer_m
                top = max(b[3] for b in m_bounds) + aoi_buffer_m
                
                aoi_box = box(left, bottom, right, top)
                aoi_wgs84 = shp_transform(inverse, aoi_box)
                w, s, e, n = aoi_wgs84.bounds
                aois.append({
                    "id": f"aoi_cluster_{idx+1}",
                    "name": f"AOI #{idx+1}",
                    "bounds": {"west": float(w), "south": float(s), "east": float(e), "north": float(n)},
                    "geometry": mapping(aoi_wgs84),
                    "annotation_count": len(matching),
                    "label": active_label or "annotation",
                    "is_cluster": True
                })

            aois.sort(key=lambda a: a["annotation_count"], reverse=True)
            if not aois:
                left, bottom, right, top = min(g.bounds[0] for g, _ in projected), min(g.bounds[1] for g, _ in projected), max(g.bounds[2] for g, _ in projected), max(g.bounds[3] for g, _ in projected)
                label_extent = box(left - aoi_buffer_m, bottom - aoi_buffer_m, right + aoi_buffer_m, top + aoi_buffer_m)
                aoi_wgs84 = shp_transform(inverse, label_extent)
                w, s, e, n = aoi_wgs84.bounds
                aois = [{
                    "id": "aoi_cluster_1",
                    "name": "AOI #1",
                    "bounds": {"west": float(w), "south": float(s), "east": float(e), "north": float(n)},
                    "geometry": mapping(aoi_wgs84),
                    "annotation_count": len(valid),
                    "label": active_label or "annotation"
                }]

            return JSONResponse({
                "success": True,
                "aoi_count": len(aois),
                "aois": aois,
                "active_label": active_label or "annotation",
                "cluster_mode": True,
                "total_annotations_clustered": sum(a["annotation_count"] for a in aois)
            })
        else:
            left, bottom, right, top = min(g.bounds[0] for g, _ in projected), min(g.bounds[1] for g, _ in projected), max(g.bounds[2] for g, _ in projected), max(g.bounds[3] for g, _ in projected)
            label_extent = box(left, bottom, right, top)
            aoi_wgs84, extent_wgs84 = shp_transform(inverse, label_extent.buffer(1.0, cap_style=3, join_style=2)), shp_transform(inverse, label_extent)
            west, south, east, north = aoi_wgs84.bounds
            return JSONResponse({
                "success": True,
                "aoi_count": 1,
                "annotation_count": len(valid),
                "active_label": active_label or "annotation",
                "aoi_buffer_m": 1.0,
                "extent_method": "selected_label_extreme_bounds",
                "label_extrema_m": {"left": float(left), "bottom": float(bottom), "right": float(right), "top": float(top)},
                "aois": [{
                    "geometry": mapping(aoi_wgs84),
                    "bounds": {"west": west, "south": south, "east": east, "north": north},
                    "annotation_count": len(valid),
                    "label_extent": mapping(extent_wgs84)
                }]
            })
    except Exception as exc:
        traceback.print_exc(); return JSONResponse({"success": False, "error": str(exc)}, status_code=400)

@app.post("/api/dataset/build", dependencies=[Depends(require_api_key)])
async def build_dataset_api(data: Dict):
    try:
        project_id, tile_size, overlap = data.get("project_id") or training_state.current_project_id, int(data.get("tile_size", 256)), float(data.get("overlap", 0.2))
        label, features = data.get("label") or training_state.current_label or training_state.selected_label, data.get("features", [])
        if not project_id or not label: return JSONResponse({"success": False, "error": "Project ID and Label required"}, status_code=400)
        project = get_project(project_id)
        if not project: return JSONResponse({"success": False, "error": "Project not found"}, status_code=404)
        dataset_model_type = normalize_model_type(str(data.get("model_type_key") or data.get("model_type") or project.get("model_type_key") or project.get("model_type") or "unet").strip().lower().replace(" ", "_"))
        training_state.model_type = dataset_model_type
        
        if not features:
            ap = training_state.annotation_file_path or project.get("annotation_path")
            if ap and os.path.exists(ap):
                with open(ap, "r") as f: d = json.load(f)
                features = d.get("features", [])
        if not features: return JSONResponse({"success": False, "error": "No annotations"}, status_code=400)

        safe_label = label.lower().replace(" ", "_")
        class_names = dedupe_labels(project.get("labels", ["background"]))
        if safe_label not in {c.lower().replace(" ", "_") for c in class_names}: class_names.append(safe_label); class_names = dedupe_labels(class_names)
        from utils.dataset_builder import DatasetBuilder
        from shapely.geometry import shape, box, mapping

        aois = data.get("aois")
        # --- MULTI-AOI WORKFLOW ---
        if isinstance(aois, list) and len(aois) > 0:
            capture_zoom = max(15, int(data.get("capture_zoom", training_state.current_zoom or 17)))
            service_id = data.get("service_id", training_state.current_map_service)
            total_chips = 0
            all_stats = {}
            last_metadata = {}
            aoi_success_count = 0
            active_name = str(label).strip().casefold()
            
            # Adopt features if none match active_name
            matching_features = [f for f in features if str((f.get("properties") or {}).get("class_name") or (f.get("properties") or {}).get("label") or (f.get("properties") or {}).get("class") or "").strip().casefold() == active_name]
            if not matching_features and features:
                print(f"ℹ️ Auto-adopting {len(features)} features to active label '{label}'")
                for f in features:
                    props = f.setdefault("properties", {})
                    props["class_name"] = str(label).strip()
                    props["label"] = str(label).strip()

            for idx, aoi in enumerate(aois):
                aoi_bounds = aoi.get("bounds")
                if not aoi_bounds: continue
                
                # 1. Capture imagery for this specific AOI
                try:
                    cap_res = capture_map_image(aoi_bounds, capture_zoom, service_id, project_id)
                    aoi_img_path = cap_res["filepath"]
                    aoi_snapped_bounds = cap_res["bounds"]
                except Exception as cap_err:
                    print(f"⚠️ AOI {idx+1} capture failed: {cap_err}")
                    continue
                
                # 2. Filter features intersecting this AOI
                try:
                    aoi_geom = aoi.get("geometry")
                    aoi_shape = shape(aoi_geom) if aoi_geom else box(float(aoi_bounds["west"]), float(aoi_bounds["south"]), float(aoi_bounds["east"]), float(aoi_bounds["north"]))
                    active_name = str(label).strip().casefold()
                    filtered_features = []
                    for f in features:
                        props = f.get("properties") or {}
                        if str(props.get("class_name") or props.get("label") or props.get("class") or active_name).strip().casefold() != active_name: continue
                        g_json = f.get("geometry")
                        if not g_json: continue
                        geom = shape(g_json)
                        if not geom.is_valid: geom = geom.buffer(0)
                        clipped = geom.intersection(aoi_shape)
                        if clipped.is_empty: continue
                        out_f = dict(f); out_f["geometry"] = mapping(clipped); filtered_features.append(out_f)
                except Exception as clip_err:
                    print(f"⚠️ AOI {idx+1} clipping failed: {clip_err}")
                    continue
                
                if not filtered_features:
                    print(f"ℹ️ AOI {idx+1} has 0 intersecting features. Skipping.")
                    continue
                
                # 3. Build chips with unique prefix and append=(aoi_success_count > 0)
                try:
                    builder = DatasetBuilder(
                        source_image_path=aoi_img_path,
                        features=filtered_features,
                        class_names=class_names,
                        tile_size=tile_size,
                        overlap=overlap,
                        bounds=aoi_snapped_bounds,
                        label=safe_label,
                        model_type=dataset_model_type,
                        task_type=project.get("task_type"),
                        chip_prefix=f"aoi{idx+1:02d}_"
                    )
                    dataset_dir, last_metadata, last_stats = builder.build_dataset(
                        output_dir=ensure_label_folders(safe_label),
                        append=(aoi_success_count > 0)
                    )
                    total_chips = last_metadata.get("total_chips", total_chips)
                    all_stats = last_stats
                    aoi_success_count += 1
                except Exception as build_err:
                    print(f"⚠️ AOI {idx+1} dataset builder error: {build_err}")
                    continue

            if aoi_success_count == 0:
                return JSONResponse({"success": False, "error": "No chips could be generated from the selected AOIs. Ensure annotations intersect the AOI boundaries."}, status_code=400)
            
            training_state.total_chips, training_state.class_names, training_state.num_classes, training_state.dataset_dir, training_state.tile_size = total_chips, class_names, len(class_names), os.path.join(ensure_label_folders(safe_label), "dataset"), tile_size
            if not training_state.current_label: training_state.current_label, training_state.selected_label = safe_label, safe_label
            update_project(project_id, {"chips_count": total_chips, "labels": class_names, "dataset_dir": training_state.dataset_dir, "model_type": project.get("model_type", dataset_model_type), "model_type_key": dataset_model_type})
            training_state.readiness = check_training_readiness(project_id, safe_label, tile_size)
            
            return JSONResponse({
                "success": True,
                "dataset_dir": training_state.dataset_dir,
                "total_chips": total_chips,
                "num_aois": aoi_success_count,
                "stats": all_stats,
                "class_names": class_names,
                "num_classes": len(class_names),
                "tile_size": tile_size,
                "model_type": dataset_model_type,
                "model_family": last_metadata.get("model_family"),
                "primary_target": last_metadata.get("primary_target"),
                "target_manifest": last_metadata.get("target_manifest_file"),
                "chip_previews": last_metadata.get("chip_previews", []),
                "readiness": training_state.readiness
            })

        # --- SINGLE AOI / EXISTING WORKFLOW ---
        bounds, image_path = data.get("bounds") or training_state.image_bounds_latlon, data.get("image_path") or training_state.image_path
        aoi_geometry = data.get("aoi_geometry")
        active_name = str(label).strip().casefold()
        matching_features = [f for f in features if str((f.get("properties") or {}).get("class_name") or (f.get("properties") or {}).get("label") or (f.get("properties") or {}).get("class") or "").strip().casefold() == active_name]
        if not matching_features and features:
            print(f"ℹ️ Auto-adopting {len(features)} features to active label '{label}'")
            for f in features:
                props = f.setdefault("properties", {})
                props["class_name"] = str(label).strip()
                props["label"] = str(label).strip()

        if data.get("is_aoi") and aoi_geometry:
            try:
                aoi_shape, filtered_features = shape(aoi_geometry), []
                for feature in features:
                    props = feature.get("properties") or {}
                    if str(props.get("class_name") or props.get("label") or props.get("class") or active_name).strip().casefold() != active_name: continue
                    geom_json = feature.get("geometry")
                    if not geom_json: continue
                    geom = shape(geom_json)
                    if not geom.is_valid: geom = geom.buffer(0)
                    clipped = geom.intersection(aoi_shape)
                    if clipped.is_empty: continue
                    out_feature = dict(feature); out_feature["geometry"] = mapping(clipped); filtered_features.append(out_feature)
                features = filtered_features
                if not features: return JSONResponse({"success": False, "error": "No selected-label annotations intersect the AOI boundary."}, status_code=400)
            except Exception as clip_exc: return JSONResponse({"success": False, "error": f"AOI annotation clipping failed: {clip_exc}"}, status_code=400)
        img_path = image_path or project.get("image_path")
        if not img_path or not os.path.exists(img_path): return JSONResponse({"success": False, "error": "No image. Capture a map area first."}, status_code=400)
        if not bounds: w, h = Image.open(img_path).size; bounds = {"west": 0, "south": h, "east": w, "north": 0}
        builder = DatasetBuilder(source_image_path=img_path, features=features, class_names=class_names, tile_size=tile_size, overlap=overlap, bounds=bounds, label=safe_label, model_type=dataset_model_type, task_type=project.get("task_type"))
        dataset_dir, metadata, chip_stats = builder.build_dataset(output_dir=ensure_label_folders(safe_label))
        training_state.total_chips, training_state.class_names, training_state.num_classes, training_state.dataset_dir, training_state.tile_size = metadata.get("total_chips", 0), class_names, len(class_names), dataset_dir, tile_size
        if not training_state.current_label: training_state.current_label, training_state.selected_label = safe_label, safe_label
        update_project(project_id, {"chips_count": metadata.get("total_chips", 0), "labels": class_names, "dataset_dir": dataset_dir, "model_type": project.get("model_type", dataset_model_type), "model_type_key": dataset_model_type})
        training_state.readiness = check_training_readiness(project_id, safe_label, tile_size)
        return JSONResponse({"success": True, "dataset_dir": dataset_dir, "total_chips": metadata.get("total_chips", 0), "stats": chip_stats, "class_names": class_names, "num_classes": len(class_names), "tile_size": tile_size, "model_type": dataset_model_type, "model_family": metadata.get("model_family"), "primary_target": metadata.get("primary_target"), "target_manifest": metadata.get("target_manifest_file"), "chip_previews": metadata.get("chip_previews", []), "readiness": training_state.readiness})
    except Exception as e:
        traceback.print_exc(); return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.get("/api/training/readiness", dependencies=[Depends(require_api_key)])
async def training_readiness(project_id: Optional[str] = None, label: Optional[str] = None, tile_size: int = 256):
    pid, lbl = project_id or training_state.current_project_id, label or training_state.current_label or training_state.selected_label
    if not pid or not lbl: return JSONResponse({"success": False, "error": "Project and Label required"}, status_code=400)
    result = check_training_readiness(pid, lbl, tile_size); training_state.readiness = result
    return JSONResponse({"success": True, **result})

@app.post("/train", dependencies=[Depends(require_api_key), Depends(require_capability("can_train"))])
async def train_model(request: Request):
    try:
        data = await request.json()
        if training_state.is_training: return JSONResponse({"success": False, "error": "Training already running. Press STOP first."}, status_code=400)
        epochs, batch_size, threshold = int(data.get("epochs", 100)), max(1, int(data.get("batch_size", 2))), float(data.get("threshold", 0.5))
        label, backbone = data.get("label", ""), normalize_backbone(data.get("backbone", "unet"))
        model_name, fine_tune_model_path = data.get("model_name", ""), str(data.get("fine_tune_model_path") or "").strip()
        if fine_tune_model_path and not os.path.isfile(fine_tune_model_path): return JSONResponse({"success": False, "error": "Selected fine-tune checkpoint does not exist on the server"}, status_code=400)
        mode, project_id, force = str(data.get("mode", training_state.mode)).lower(), data.get("project_id") or training_state.current_project_id, bool(data.get("force", False))
        project_for_model = get_project(project_id) if project_id else None
        raw_model_type = data.get("model_type") or data.get("model_type_key") or (project_for_model or {}).get("model_type_key") or (project_for_model or {}).get("model_type") or training_state.model_type or "unet"
        model_type = normalize_model_type(str(raw_model_type).strip().lower().replace(" ", "_"))
        weight_decay, dropout, grad_clip = float(data.get("weight_decay", 0.0001)), float(data.get("dropout", 0.0)), float(data.get("grad_clip", 1.0))
        requested_early_stop, full_epochs = int(data.get("early_stop", 0) or 0), bool(data.get("full_epochs", True))
        early_stop = 0 if (model_type in SEGMENTATION_MODELS and full_epochs) else max(0, requested_early_stop)
        scheduler_kind, augment_kind = str(data.get("scheduler", "plateau")).lower(), str(data.get("augment", "basic")).lower()
        raw_lr = float(data.get("learning_rate", 0.0001)); learning_rate = min(raw_lr, 0.0002)
        if model_type in DETECTION_MODELS or model_type in INSTANCE_MODELS:
            if backbone not in {"resnet18", "resnet34", "resnet50", "resnet101", "mobilenet_v2"}: backbone = "resnet50"
        if project_id and project_id != training_state.current_project_id:
            project = get_project(project_id)
            if project:
                training_state.current_project_id, training_state.class_names, training_state.num_classes, training_state.mode = project_id, dedupe_labels(project.get("labels", ["background"])), len(dedupe_labels(project.get("labels", ["background"]))), project.get("mode", "binary")
                if project.get("image_path") and os.path.exists(project["image_path"]): training_state.image_path = project["image_path"]
                if project.get("bounds"): training_state.image_bounds_latlon = project["bounds"]
        use_label = label or training_state.current_label or training_state.selected_label
        if not use_label or use_label == "unlabeled": return JSONResponse({"success": False, "error": "No valid label selected. Pick a label in step 2."}, status_code=400)
        safe_label, label_path = use_label.lower().replace(" ", "_"), ensure_label_folders(use_label.lower().replace(" ", "_"))
        readiness = check_training_readiness(training_state.current_project_id, safe_label, int(training_state.tile_size or 256))
        auto_force = readiness.get("summary", {}).get("train", 0) < 25
        if not readiness["ready"] and not (force or auto_force): return JSONResponse({"success": False, "error": f"Training not ready. Failing checks: {', '.join(k for k, v in readiness['checks'].items() if v['level'] == 'fail')}", "readiness": readiness}, status_code=400)
        dataset_dir = os.path.join(label_path, "dataset"); train_images_dir, train_masks_dir = os.path.join(dataset_dir, "train", "images"), os.path.join(dataset_dir, "train", "masks")
        if not os.path.exists(train_images_dir) or not any(f.lower().endswith((".png", ".jpg", ".jpeg")) and not f.endswith("_vis.png") for f in os.listdir(train_images_dir)): return JSONResponse({"success": False, "error": "No training images found. Build dataset first."}, status_code=400)
        metadata_path = os.path.join(dataset_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f: meta_ds = json.load(f)
            class_names, num_classes, tile_size = dedupe_labels(meta_ds.get("class_names", ["background", safe_label])), len(dedupe_labels(meta_ds.get("class_names", ["background", safe_label]))), int(meta_ds.get("tile_size", 256))
        else: class_names, num_classes, tile_size = ["background", safe_label], 2, 256
        fine_tune_source_model, fine_tune_source_meta = None, None
        if fine_tune_model_path and model_type != "yolo":
            # 1. Load metadata first to check architecture
            _, fine_tune_source_meta = load_checkpoint(fine_tune_model_path, device=torch.device("cpu"))
            
            source_type = normalize_model_type(fine_tune_source_meta.get("model_type", "unet"))
            source_mode = str(fine_tune_source_meta.get("mode", "binary")).lower()
            source_tile = int(fine_tune_source_meta.get("tile_size", tile_size))
            
            # 2. Strict Enforcement
            if source_type != model_type: 
                raise ValueError(f"You selected a {model_type} model to fine-tune, but the .dlpk is a {source_type}. Please select {source_type} in Step 5.")
            if source_mode != mode: 
                raise ValueError(f"You selected {mode} mode, but the .dlpk is {source_mode}.")
            if source_tile != int(tile_size): 
                raise ValueError(f"The .dlpk expects {source_tile}x{source_tile} tiles, but your dataset is {tile_size}x{tile_size}. Rebuild the dataset with tile size {source_tile}.")
            
            # 3. Detect if the user added new classes
            dataset_classes = len(class_names)
            dlpk_classes = int(fine_tune_source_meta.get("num_classes", 2))
            
            if dataset_classes < dlpk_classes:
                raise ValueError(f"The .dlpk expects {dlpk_classes} classes, but your dataset only has {dataset_classes}. Ensure your labels match.")
            
            # 4. Load the actual model with the expanded class count
            fine_tune_source_model, fine_tune_source_meta = load_checkpoint(
                fine_tune_model_path, 
                device=torch.device("cpu"), 
                requested_num_classes=dataset_classes
            )
            
            backbone = normalize_backbone(fine_tune_source_meta.get("backbone", backbone))
            num_classes = dataset_classes
        if model_type in DETECTION_MODELS or model_type in INSTANCE_MODELS or model_type == "yolo": num_classes = max(2, num_classes)
        training_state.class_names, training_state.num_classes, training_state.tile_size = class_names, num_classes, tile_size
        training_state.stop_training, training_state.total_epochs, training_state.learning_rate, training_state.threshold = False, epochs, learning_rate, threshold
        training_state.training_start_time, training_state.status, training_state.progress_percent = time.time(), "Training", 0
        training_state.train_losses, training_state.val_losses, training_state.ious, training_state.best_map = [], [], [], 0.0
        training_state.model_name, training_state.training_completed, training_state.is_training, training_state.mode = model_name or safe_label, False, True, mode
        training_state.model_backbone, training_state.model_type = backbone, model_type
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pos_weight_auto, fg_ratio_mask = compute_dataset_pos_weight(train_masks_dir, max_samples=100)
        bias_init = compute_bias_init(fg_ratio_mask) if model_type in SEGMENTATION_MODELS and mode == "binary" else None
        if model_type in DETECTION_MODELS or model_type in INSTANCE_MODELS:
            detection_min_size, detection_max_size = int(tile_size), max(int(tile_size), tile_size * 2)
        else: detection_min_size, detection_max_size = None, None
        if model_type == "yolo" and fine_tune_model_path:
            YOLO = depcfg.require_yolo(); model = YOLO(fine_tune_model_path)
        elif fine_tune_source_model is not None:
            model = fine_tune_source_model.to(device); model.train()
        else:
            model = build_model(mode=mode, num_classes=num_classes, backbone=backbone, pretrained=True, fuse_at="bottleneck", support_weight=0.3, bias_init=bias_init, model_type=model_type, tile_size=tile_size, dropout=dropout, real_detector=(model_type in DETECTION_MODELS or model_type in INSTANCE_MODELS), detection_min_size=detection_min_size, detection_max_size=detection_max_size)
        training_state.architecture = getattr(model, "geoai_architecture", type(model).__name__)
        if model_type == "yolo":
            training_state.model_instance, training_state.optimizer, training_state.scheduler = model, None, None
            task = asyncio.create_task(actual_yolo_training_loop(epochs, batch_size, learning_rate, label_path, safe_label, class_names, num_classes, model_name, model, device, model_type, weight_decay=weight_decay, early_stop=early_stop))
            training_tasks["current"] = task; task.add_done_callback(lambda t: training_tasks.pop("current", None))
            return JSONResponse({"success": True, "message": "YOLO training started"})
        model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        if scheduler_kind == "cosine": scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=1e-6)
        elif scheduler_kind == "step": scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(5, epochs // 10), gamma=0.5)
        elif scheduler_kind == "none": scheduler = None
        else: scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=10, factor=0.5, min_lr=1e-6)
        training_state.model_instance, training_state.optimizer, training_state.scheduler = model, optimizer, scheduler
        if model_type == "classic_cnn": task = asyncio.create_task(actual_classification_training_loop(epochs, batch_size, learning_rate, label_path, safe_label, class_names, num_classes, backbone, model_name, model, optimizer, scheduler, device, mode, pos_weight_auto, grad_clip=grad_clip, early_stop=early_stop, scheduler_kind=scheduler_kind))
        elif model_type == "mask_rcnn": task = asyncio.create_task(real_maskrcnn_training_loop(epochs, batch_size, learning_rate, label_path, safe_label, class_names, num_classes, backbone, model_name, model, optimizer, scheduler, device, mode, pos_weight_auto, model_type, grad_clip=grad_clip, early_stop=early_stop, scheduler_kind=scheduler_kind))
        elif model_type in DETECTION_MODELS: task = asyncio.create_task(actual_detection_training_loop(epochs, batch_size, learning_rate, label_path, safe_label, class_names, num_classes, backbone, model_name, model, optimizer, scheduler, device, mode, pos_weight_auto, model_type, grad_clip=grad_clip, early_stop=early_stop, scheduler_kind=scheduler_kind))
        else: task = asyncio.create_task(actual_training_loop(epochs, batch_size, learning_rate, label_path, safe_label, class_names, num_classes, backbone, model_name, model, optimizer, scheduler, device, mode, pos_weight_auto, threshold=threshold, grad_clip=grad_clip, early_stop=early_stop, scheduler_kind=scheduler_kind, fine_tune_source_path=fine_tune_model_path or None, fine_tune_source_epoch=(int(fine_tune_source_meta.get("epoch", 0)) if fine_tune_source_meta else None)))
        training_tasks["current"] = task; task.add_done_callback(lambda t: training_tasks.pop("current", None))
        return JSONResponse({"success": True, "message": "Training started", "mode": mode, "backbone": backbone, "model_type": model_type, "architecture": training_state.architecture, "pos_weight": pos_weight_auto, "bias_init": bias_init, "weight_decay": weight_decay, "dropout": dropout, "grad_clip": grad_clip, "early_stop": early_stop, "requested_early_stop": requested_early_stop, "full_epochs": full_epochs, "scheduler": scheduler_kind, "augment": augment_kind})
    except depcfg.PretrainedWeightsUnavailable as e:
        training_state.is_training = False; training_state.status = "Failed"; print(str(e)); return JSONResponse({"success": False, "error": str(e), "finding": "air-gap/pretrained-weights"}, status_code=503)
    except Exception as e:
        traceback.print_exc(); training_state.is_training = False; training_state.status = "Failed"; return JSONResponse({"success": False, "error": str(e)}, status_code=500)

class _StopTraining(Exception): pass

class DetectionChipDataset(Dataset):
    def __init__(self, split_dir, num_classes=2):
        self.split_dir, self.num_classes = split_dir, max(2, int(num_classes))
        self.img_dir, self.tgt_dir, self.stems = os.path.join(split_dir, "images"), os.path.join(split_dir, "detection_targets"), []
        if not os.path.isdir(self.img_dir) or not os.path.isdir(self.tgt_dir): return
        for f in sorted(os.listdir(self.tgt_dir)):
            if not f.endswith(".json"): continue
            stem = os.path.splitext(f)[0]
            if os.path.exists(os.path.join(self.img_dir, stem + ".png")) or os.path.exists(os.path.join(self.img_dir, stem + ".jpg")) or os.path.exists(os.path.join(self.img_dir, stem + ".jpeg")): self.stems.append(stem)
    def __len__(self): return len(self.stems)
    def __getitem__(self, i):
        stem = self.stems[i]; img_path = next((os.path.join(self.img_dir, stem + ext) for ext in (".png", ".jpg", ".jpeg") if os.path.exists(os.path.join(self.img_dir, stem + ext))), None)
        if img_path is None: raise FileNotFoundError(f"Image not found for chip {stem}")
        img = Image.open(img_path).convert("RGB"); img_w, img_h = img.size; img_t = transforms.functional.to_tensor(img)
        with open(os.path.join(self.tgt_dir, stem + ".json"), "r", encoding="utf-8") as f: data = json.load(f)
        boxes, labels, areas, iscrowd = [], [], [], []
        for inst in data.get("instances", []) or []:
            b = inst.get("bbox")
            if not isinstance(b, (list, tuple)) or len(b) != 4: continue
            x1, y1, x2, y2 = [float(v) for v in b]
            x1 = max(0.0, min(float(img_w - 1), x1)); y1 = max(0.0, min(float(img_h - 1), y1))
            x2 = max(x1 + 1.0, min(float(img_w), x2)); y2 = max(y1 + 1.0, min(float(img_h), y2))
            cls = int(inst.get("class_id", 2))
            cls = max(1, min(cls, self.num_classes - 1))
            boxes.append([x1, y1, x2, y2]); labels.append(cls); areas.append((x2 - x1) * (y2 - y1)); iscrowd.append(int(inst.get("iscrowd", 0)))
        target = {"boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4), "labels": torch.as_tensor(labels, dtype=torch.int64), "image_id": torch.tensor(i, dtype=torch.int64), "area": torch.as_tensor(areas, dtype=torch.float32), "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64)}
        return img_t, target

class ClassificationChipDataset(Dataset):
    def __init__(self, split_dir, num_classes=2):
        self.split_dir, self.num_classes = split_dir, max(2, int(num_classes))
        self.img_dir, self.label_dir, self.samples = os.path.join(split_dir, "images"), os.path.join(split_dir, "class_labels"), []
        if not os.path.isdir(self.img_dir): return
        for f in sorted(os.listdir(self.img_dir)):
            if not f.lower().endswith((".png", ".jpg", ".jpeg")): continue
            stem = os.path.splitext(f)[0]; lp = os.path.join(self.label_dir, stem + ".json")
            if os.path.exists(lp): self.samples.append((f, lp))
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        fn, lp = self.samples[i]; img = Image.open(os.path.join(self.img_dir, fn)).convert("RGB")
        with open(lp, "r", encoding="utf-8") as f: target = json.load(f)
        return transforms.functional.to_tensor(img), torch.tensor(max(0, min(self.num_classes - 1, int(target.get("primary_class_id", 0)))), dtype=torch.long)

async def actual_classification_training_loop(epochs, batch_size, learning_rate, label_path, safe_label, class_names, num_classes, backbone, model_name, model, optimizer, scheduler, device, mode, pos_weight_auto, grad_clip=1.0, early_stop=0, scheduler_kind="plateau"):
    best_acc, best_epoch, best_model_path = -1.0, 0, None
    training_state.is_training, training_state.stop_training, training_state.training_completed = True, False, False
    try:
        dataset_dir = os.path.join(label_path, "dataset"); train_ds, val_ds = ClassificationChipDataset(os.path.join(dataset_dir, "train"), num_classes), ClassificationChipDataset(os.path.join(dataset_dir, "val"), num_classes)
        if len(train_ds) == 0: raise RuntimeError("Classic CNN training dataset empty.")
        train_loader, val_loader = DataLoader(train_ds, batch_size=max(1, batch_size), shuffle=True, num_workers=0), DataLoader(val_ds, batch_size=max(1, batch_size), shuffle=False, num_workers=0) if len(val_ds) else None
        criterion, models_dir = nn.CrossEntropyLoss(), os.path.join(label_path, "models"); os.makedirs(models_dir, exist_ok=True)
        for epoch in range(epochs):
            if training_state.stop_training: raise _StopTraining()
            training_state.epoch, training_state.progress_percent = epoch + 1, (epoch + 1) / max(epochs, 1) * 100.0
            model.train(); total_loss, n_batches = 0.0, 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device); optimizer.zero_grad(set_to_none=True)
                logits = model(images); logits = logits[0] if isinstance(logits, (list, tuple)) else logits
                loss = criterion(logits, labels)
                if not torch.isfinite(loss): raise RuntimeError("Non-finite Classic CNN loss")
                loss.backward()
                if grad_clip > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step(); total_loss += float(loss.item()); n_batches += 1
            avg_loss = total_loss / max(n_batches, 1); correct = total = 0
            if val_loader:
                model.eval()
                with torch.no_grad():
                    for images, labels in val_loader:
                        logits = model(images.to(device)); logits = logits[0] if isinstance(logits, (list, tuple)) else logits
                        pred = torch.argmax(logits, dim=1)
                        correct += int((pred == labels.to(device)).sum().item()); total += int(labels.numel())
            acc = correct / total if total else 0.0
            training_state.train_losses.append(avg_loss); training_state.val_losses.append(avg_loss); training_state.ious.append(acc); training_state.best_map = max(training_state.best_map, acc)
            await broadcast_training_update({"status": "training", "epoch": epoch + 1, "total_epochs": epochs, "train_loss": avg_loss, "val_loss": avg_loss, "iou": acc, "best_iou": training_state.best_map, "best_epoch": best_epoch, "progress": training_state.progress_percent, "progress_percent": training_state.progress_percent, "train_losses": training_state.train_losses, "losses": training_state.train_losses, "val_losses": training_state.val_losses, "ious": training_state.ious, "accuracy": acc, "metric_name": "accuracy"})
            if scheduler is not None:
                try: scheduler.step(avg_loss) if scheduler_kind == "plateau" else scheduler.step()
                except Exception: pass
            if acc > best_acc:
                best_acc, best_epoch = acc, epoch + 1; best_model_path = os.path.join(models_dir, f"{safe_label}_best.pth")
                torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict() if scheduler else None, "epoch": epoch + 1, "best_iou": best_acc, "model_type": "classic_cnn", "architecture": getattr(model, "geoai_architecture", type(model).__name__), "backbone": backbone, "num_classes": num_classes, "class_names": class_names, "tile_size": training_state.tile_size, "mode": mode}, best_model_path)
            if early_stop > 0 and (epoch + 1 - best_epoch) >= early_stop: break
        training_state.training_completed, training_state.status, training_state.progress_percent, training_state.model_path, training_state.is_training = True, "Completed", 100.0, best_model_path, False
        if training_state.current_project_id: update_project(training_state.current_project_id, {"has_model": True, "model_path": best_model_path, "evaluation_score": best_acc, "chips_count": len(train_ds)})
        await broadcast_training_update({"status": "completed", "epoch": training_state.epoch, "best_iou": best_acc, "best_epoch": best_epoch, "model_path": best_model_path, "metric_name": "accuracy"})
    except _StopTraining: training_state.is_training, training_state.status = False, "Stopped"
    except Exception as e:
        training_state.is_training, training_state.status = False, "Failed"; traceback.print_exc(); await broadcast_training_update({"status": "failed", "error": str(e)})

async def actual_yolo_training_loop(epochs, batch_size, learning_rate, label_path, safe_label, class_names, num_classes, model_name, model, device, model_type, weight_decay=0.0001, early_stop=0):
    training_state.is_training, training_state.stop_training, training_state.training_completed = True, False, False
    try:
        dataset_dir = os.path.join(label_path, "dataset"); yaml_path = os.path.join(dataset_dir, "data.yaml")
        if not os.path.exists(yaml_path): raise FileNotFoundError(f"YOLO data.yaml not found.")
        project_dir = os.path.join(label_path, "models", "yolo_runs"); os.makedirs(project_dir, exist_ok=True)
        results = await asyncio.to_thread(model.train, data=yaml_path, epochs=int(epochs), batch=int(batch_size), imgsz=int(training_state.tile_size or 640), lr0=float(learning_rate), weight_decay=float(weight_decay), device=(0 if torch.cuda.is_available() else "cpu"), project=project_dir, name=re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name or safe_label or "yolo"), exist_ok=True, verbose=True, patience=int(early_stop) if early_stop > 0 else 100)
        save_dir, best_path = getattr(getattr(model, "trainer", None), "save_dir", None), getattr(getattr(model, "trainer", None), "best", None)
        if best_path is None and save_dir is not None:
            candidate = os.path.join(str(save_dir), "weights", "best.pt")
            if os.path.exists(candidate): best_path = candidate
        training_state.model_path, training_state.training_completed, training_state.status, training_state.is_training, training_state.progress_percent, training_state.architecture = str(best_path), True, "Completed", False, 100.0, "UltralyticsYOLO"
        if training_state.current_project_id: update_project(training_state.current_project_id, {"has_model": bool(best_path), "model_path": best_path, "evaluation_score": None, "chips_count": len(os.listdir(os.path.join(dataset_dir, "train", "images"))) if os.path.isdir(os.path.join(dataset_dir, "train", "images")) else 0})
        await broadcast_training_update({"status": "completed", "epoch": epochs, "total_epochs": epochs, "progress": 100.0, "model_path": best_path, "metric_name": "YOLO mAP"})
    except Exception as e:
        training_state.is_training, training_state.status = False, "Failed"; traceback.print_exc(); await broadcast_training_update({"status": "failed", "error": str(e)})

async def actual_training_loop(epochs, batch_size, learning_rate, label_path, safe_label, class_names, num_classes, backbone, model_name, model, optimizer, scheduler, device, mode, pos_weight_auto, threshold=0.5, grad_clip=1.0, early_stop=0, scheduler_kind="plateau", fine_tune_source_path=None, fine_tune_source_epoch=None):
    threshold, best_iou, best_epoch, best_model_path, epoch_snap_epoch, stuck_count, pred_max_init = float(max(0.0, min(1.0, threshold))), 0.0, 0, None, -1, 0, 0.0
    training_state.is_training, training_state.stop_training, training_state.epoch, training_state.training_completed = True, False, 0, False
    try:
        models_dir = os.path.join(label_path, "models"); os.makedirs(models_dir, exist_ok=True)
        run_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(model_name or safe_label)).strip("._") or safe_label
        run_checkpoint_path = os.path.join(models_dir, f"{run_stem}_best.pth")
        if os.path.exists(run_checkpoint_path) or (fine_tune_source_path and os.path.abspath(run_checkpoint_path) == os.path.abspath(fine_tune_source_path)): run_checkpoint_path = os.path.join(models_dir, f"{run_stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_best.pth")
        dataset_dir = os.path.join(label_path, "dataset"); train_dir, val_dir = os.path.join(dataset_dir, "train"), os.path.join(dataset_dir, "val")
        snapshots_dir = os.path.join(label_path, "evaluation", "epoch_snapshots"); os.makedirs(snapshots_dir, exist_ok=True)
        train_images_dir, train_masks_dir = os.path.join(train_dir, "images"), os.path.join(train_dir, "masks")
        metadata_path = os.path.join(dataset_dir, "metadata.json"); tile_size = 256
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f: tile_size = int(json.load(f).get("tile_size", 256))
        if mode == "binary": train_dataset = BinaryChipDataset(train_images_dir, train_masks_dir, target_size=tile_size, augment=True)
        else: train_dataset = SegmentationDataset(train_images_dir, train_masks_dir, target_size=tile_size, num_classes=num_classes)
        vdir, val_dataset = os.path.join(val_dir, "images"), None
        if os.path.exists(vdir) and any(f.lower().endswith((".png", ".jpg", ".jpeg")) for f in os.listdir(vdir)):
            val_dataset = BinaryChipDataset(vdir, os.path.join(val_dir, "masks"), target_size=tile_size, augment=False) if mode == "binary" else SegmentationDataset(vdir, os.path.join(val_dir, "masks"), target_size=tile_size, num_classes=num_classes)
        if len(train_dataset) == 0: raise RuntimeError("Empty training dataset")
        train_sampler = None
        if mode == "binary" and isinstance(train_dataset, BinaryChipDataset):
            positive_flags = [bool(binary_mask_foreground(np.asarray(Image.open(os.path.join(train_dataset.mask_dir, chip_name)).convert("L"))).any()) for chip_name in train_dataset.files]
            n_pos, n_neg = int(sum(positive_flags)), len(positive_flags) - int(sum(positive_flags))
            if n_pos > 0 and n_neg > 0: train_sampler = WeightedRandomSampler(torch.as_tensor([(0.5 / n_pos) if is_pos else (0.5 / n_neg) for is_pos in positive_flags], dtype=torch.double), num_samples=len(positive_flags), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0) if val_dataset and len(val_dataset) > 0 else None
        regional_val_loaders = [("current_region", val_loader)] if val_loader is not None else []
        regions_root = os.path.join(dataset_dir, "regions")
        if os.path.isdir(regions_root):
            for region_name in sorted(os.listdir(regions_root)):
                region_val = os.path.join(regions_root, region_name, "val"); region_images, region_masks = os.path.join(region_val, "images"), os.path.join(region_val, "masks")
                if not (os.path.isdir(region_images) and os.path.isdir(region_masks)): continue
                try:
                    region_ds = BinaryChipDataset(region_images, region_masks, target_size=tile_size, augment=False) if mode == "binary" else SegmentationDataset(region_images, region_masks, target_size=tile_size, num_classes=num_classes)
                    if len(region_ds) == 0: continue
                    regional_val_loaders.append((str(region_name), DataLoader(region_ds, batch_size=batch_size, shuffle=False, num_workers=0)))
                except Exception: pass
        criterion = CombinedLoss(bce_weight=0.3, dice_weight=0.7, pos_weight=pos_weight_auto) if mode == "binary" else CombinedLoss(bce_weight=0.5, dice_weight=0.5, include_background=False)
        source_baseline_regional = {}; source_baseline_global_iou = None
        if fine_tune_source_path and regional_val_loaders:
            model.eval()
            with torch.no_grad():
                for region_name, region_loader in regional_val_loaders:
                    btp = bfp = bfn = 0
                    for b_images, b_masks in region_loader:
                        b_images, b_masks = b_images.to(device), b_masks.to(device)
                        b_outputs = model(b_images); b_outputs = b_outputs[0] if isinstance(b_outputs, (list, tuple)) else b_outputs
                        b_pred = torch.sigmoid(b_outputs)[:, 0] > threshold if mode == "binary" else torch.argmax(b_outputs, dim=1) > 0
                        b_tgt = b_masks[:, 0] > 0.5 if mode == "binary" else b_masks > 0
                        btp += int((b_pred & b_tgt).sum().item()); bfp += int((b_pred & ~b_tgt).sum().item()); bfn += int((~b_pred & b_tgt).sum().item())
                    source_baseline_regional[region_name] = {"tp": btp, "fp": bfp, "fn": bfn, "iou": float(btp / (btp + bfp + bfn + 1e-6)) if (btp + bfp + bfn) else 0.0}
            btp_all, bfp_all, bfn_all = sum(v["tp"] for v in source_baseline_regional.values()), sum(v["fp"] for v in source_baseline_regional.values()), sum(v["fn"] for v in source_baseline_regional.values())
            source_baseline_global_iou = btp_all / (btp_all + bfp_all + bfn_all + 1e-6) if (btp_all + bfp_all + bfn_all) else 0.0
        model.train()
        for epoch in range(epochs):
            if training_state.stop_training: raise _StopTraining()
            training_state.epoch, training_state.progress_percent = epoch + 1, (epoch + 1) / max(epochs, 1) * 100.0
            model.train(); running, nb = 0.0, 0
            for images, masks in train_loader:
                if training_state.stop_training: raise _StopTraining()
                images, masks = images.to(device), masks.to(device); optimizer.zero_grad(set_to_none=True)
                outputs = model(images); outputs = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
                loss = criterion(outputs, masks); loss.backward()
                if grad_clip > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step(); running += float(loss.item()); nb += 1
            avg_train_loss, val_loss, diag = running / max(nb, 1), running / max(nb, 1), {"pred_min": 0.0, "pred_max": 0.0, "pred_mean": 0.0, "pred_fg_px": 0, "gt_fg_px": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0}
            if val_loader is not None:
                model.eval(); vl_sum, vb, pred_min_running, pred_max_running, pred_sum, pred_count, tp_total, fp_total, fn_total, tn_total = 0.0, 0, float("inf"), float("-inf"), 0.0, 0, 0, 0, 0, 0
                with torch.no_grad():
                    for images, masks in val_loader:
                        images, masks = images.to(device), masks.to(device)
                        outputs = model(images); outputs = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
                        loss_v = criterion(outputs, masks)
                        if mode == "binary": prob, pred, tgt = torch.sigmoid(outputs)[:, 0], torch.sigmoid(outputs)[:, 0] > threshold, masks[:, 0] > 0.5
                        else: probs = torch.softmax(outputs, dim=1); prob, pred, tgt = 1.0 - probs[:, 0], torch.argmax(outputs, dim=1) > 0, masks > 0
                        pred_min_running, pred_max_running = min(pred_min_running, float(prob.min().item())), max(pred_max_running, float(prob.max().item()))
                        pred_sum += float(prob.sum().item()); pred_count += int(prob.numel())
                        tp_total += int((pred & tgt).sum().item()); fp_total += int((pred & ~tgt).sum().item()); fn_total += int((~pred & tgt).sum().item()); tn_total += int((~pred & ~tgt).sum().item())
                        vl_sum += float(loss_v.item()); vb += 1
                val_loss = vl_sum / max(vb, 1)
                diag = {"pred_min": pred_min_running if pred_min_running != float("inf") else 0.0, "pred_max": pred_max_running if pred_max_running != float("-inf") else 0.0, "pred_mean": pred_sum / max(pred_count, 1), "pred_fg_px": int(tp_total + fp_total), "gt_fg_px": int(tp_total + fn_total), "tp": tp_total, "fp": fp_total, "fn": fn_total, "tn": tn_total}
            tp, fp, fn = diag["tp"], diag["fp"], diag["fn"]
            st_iou = tp / (tp + fp + fn + 1e-6) if (tp + fp + fn) > 0 else 0.0
            st_prec = tp / (tp + fp + 1e-6) if (tp + fp) > 0 else 0.0
            st_rec = tp / (tp + fn + 1e-6) if (tp + fn) > 0 else 0.0
            st_f1 = 2 * st_prec * st_rec / (st_prec + st_rec + 1e-6) if (st_prec + st_rec) > 0 else 0.0
            iou, regional_metrics, global_tp, global_fp, global_fn = float(st_iou), {}, int(tp), int(fp), int(fn)
            for region_name, region_loader in regional_val_loaders:
                if region_name == "current_region": regional_metrics[region_name] = {"tp": int(tp), "fp": int(fp), "fn": int(fn), "iou": float(st_iou)}; continue
                rtp = rfp = rfn = 0; model.eval()
                with torch.no_grad():
                    for r_images, r_masks in region_loader:
                        r_images, r_masks = r_images.to(device), r_masks.to(device)
                        r_outputs = model(r_images); r_outputs = r_outputs[0] if isinstance(r_outputs, (list, tuple)) else r_outputs
                        r_pred = torch.sigmoid(r_outputs)[:, 0] > threshold if mode == "binary" else torch.argmax(r_outputs, dim=1) > 0
                        r_tgt = r_masks[:, 0] > 0.5 if mode == "binary" else r_masks > 0
                        rtp += int((r_pred & r_tgt).sum().item()); rfp += int((r_pred & ~r_tgt).sum().item()); rfn += int((~r_pred & r_tgt).sum().item())
                regional_metrics[region_name] = {"tp": rtp, "fp": rfp, "fn": rfn, "iou": float(rtp / (rtp + rfp + rfn + 1e-6)) if (rtp + rfp + rfn) else 0.0}
                global_tp += rtp; global_fp += rfp; global_fn += rfn
            global_iou = global_tp / (global_tp + global_fp + global_fn + 1e-6) if (global_tp + global_fp + global_fn) else 0.0
            selection_iou = float(global_iou) if regional_metrics else float(iou)
            if scheduler is not None:
                try: scheduler.step(val_loss) if scheduler_kind == "plateau" else scheduler.step()
                except Exception: pass
            training_state.train_losses.append(avg_train_loss); training_state.val_losses.append(val_loss); training_state.ious.append(iou)
            await broadcast_training_update({"status": "training", "epoch": epoch + 1, "total_epochs": epochs, "train_loss": avg_train_loss, "val_loss": val_loss, "iou": iou, "best_iou": max(best_iou, iou), "best_epoch": best_epoch, "progress": training_state.progress_percent, "progress_percent": training_state.progress_percent, "train_losses": training_state.train_losses, "losses": training_state.train_losses, "val_losses": training_state.val_losses, "ious": training_state.ious, "diag": diag, "st_iou": iou, "st_precision": st_prec, "st_recall": st_rec, "st_f1": st_f1})
            await asyncio.sleep(0)
            if mode == "binary" and val_loader is not None and (epoch < 3 or (epoch + 1) % 5 == 0) and epoch_snap_epoch != epoch:
                try:
                    model.eval()
                    with torch.no_grad():
                        for images, masks in val_loader:
                            images_d = images.to(device); out_s = model(images_d); out_s = out_s[0] if isinstance(out_s, (list, tuple)) else out_s
                            pred_s = torch.sigmoid(out_s)[:, 0] > 0.5
                            img_np = np.clip(images[0].cpu().numpy().transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)
                            gt_np = masks[0]; gt_np = (gt_np[0] if gt_np.ndim == 3 else gt_np).cpu().numpy() > 0.5; gt_np = gt_np.astype(np.uint8) * 255
                            pred_np = pred_s[0].cpu().numpy().astype(np.uint8) * 255
                            overlay = img_np.copy(); overlay[pred_np > 0] = [255, 0, 0]; overlay = cv2.addWeighted(img_np, 0.5, overlay, 0.5, 0)
                            snap = np.concatenate([img_np, cv2.cvtColor(gt_np, cv2.COLOR_GRAY2RGB), cv2.cvtColor(pred_np, cv2.COLOR_GRAY2RGB), overlay], axis=1)
                            Image.fromarray(snap).save(os.path.join(snapshots_dir, f"epoch_{epoch+1:04d}.png")); epoch_snap_epoch = epoch; break
                except Exception: pass
            if selection_iou > best_iou:
                best_iou, best_epoch, training_state.best_map, best_model_path = selection_iou, epoch + 1, selection_iou, run_checkpoint_path
                config = get_model_config(model); config.update({"model_type": "unet" if "unet" in type(model).__name__.lower() else normalize_model_type(getattr(model, "geoai_model_type", "unet")), "tile_size": tile_size, "target_size": tile_size})
                torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": (scheduler.state_dict() if scheduler is not None else None), "epoch": epoch + 1, "best_iou": best_iou, "best_global_iou": best_iou, "current_region_iou": float(iou), "regional_validation": regional_metrics, "source_baseline_regional_validation": source_baseline_regional, "source_baseline_global_iou": source_baseline_global_iou, "global_validation_metric": "micro_iou_from_pooled_tp_fp_fn", "best_epoch": best_epoch, "num_classes": num_classes, "class_names": class_names, "backbone": backbone, "model_name": model_name, "mode": mode, "tile_size": tile_size, "target_size": tile_size, "fuse_at": "bottleneck", "support_weight": 0.3, "model_type": "unet", "architecture": type(model).__name__, "model_config": config, "base_channels": getattr(model, "base_channels", None), "num_levels": getattr(model, "num_levels", None), "checkpoint_version": 2, "fine_tuned_from": os.path.abspath(fine_tune_source_path) if fine_tune_source_path else None, "fine_tuned_from_epoch": int(fine_tune_source_epoch) if fine_tune_source_epoch is not None else None}, best_model_path)
            if early_stop > 0 and (epoch + 1 - best_epoch) >= early_stop: break
        training_state.training_completed, training_state.status, training_state.progress_percent, training_state.model_path, training_state.is_training = True, "Completed", 100.0, best_model_path, False
        if training_state.current_project_id: update_project(training_state.current_project_id, {"has_model": True, "model_path": best_model_path, "evaluation_score": best_iou, "chips_count": len(train_dataset)})
        await broadcast_training_update({"status": "completed", "epoch": training_state.epoch, "best_iou": best_iou, "best_epoch": best_epoch, "model_path": best_model_path})
    except _StopTraining:
        training_state.is_training, training_state.status = False, "Stopped"; await broadcast_training_update({"status": "stopped", "epoch": training_state.epoch, "best_iou": training_state.best_map})
    except Exception as e:
        training_state.is_training, training_state.status = False, "Failed"; traceback.print_exc(); await broadcast_training_update({"status": "failed", "error": str(e)})

def _match_instance_masks(pred_masks, gt_masks, iou_threshold=0.5):
    pred_masks = np.asarray(pred_masks if pred_masks is not None else np.zeros((0, 1, 1))).astype(bool)
    gt_masks = np.asarray(gt_masks if gt_masks is not None else np.zeros((0, 1, 1))).astype(bool)
    used_gt, tp, fp, matched_iou = set(), 0, 0, []
    for pm in pred_masks:
        best_iou, best_j = 0.0, None
        for j, gm in enumerate(gt_masks):
            if j in used_gt: continue
            inter, union = np.logical_and(pm, gm).sum(), np.logical_or(pm, gm).sum()
            iou = float(inter) / float(union) if union > 0 else 0.0
            if iou > best_iou: best_iou, best_j = iou, j
        if best_j is not None and best_iou >= iou_threshold: used_gt.add(best_j); tp += 1; matched_iou.append(best_iou)
        else: fp += 1
    fn = max(0, len(gt_masks) - len(used_gt)); mean_iou = float(np.mean(matched_iou)) if matched_iou else 0.0
    precision = tp / (tp + fp + 1e-6) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn + 1e-6) if (tp + fn) > 0 else 0.0
    return {"iou": mean_iou, "precision": precision, "recall": recall, "f1": (2 * precision * recall / (precision + recall + 1e-6)) if precision + recall > 0 else 0.0, "dice": (2 * tp / (2 * tp + fp + fn + 1e-6)) if 2 * tp + fp + fn > 0 else 0.0, "tp": tp, "fp": fp, "fn": fn}

def _match_boxes(pred_boxes, gt_boxes, iou_threshold=0.5, pred_labels=None, gt_labels=None):
    pred_boxes = np.asarray(pred_boxes if pred_boxes is not None else [], dtype=np.float32).reshape(-1, 4)
    gt_boxes = np.asarray(gt_boxes if gt_boxes is not None else [], dtype=np.float32).reshape(-1, 4)
    pred_labels = np.asarray(pred_labels if pred_labels is not None else np.zeros((len(pred_boxes),), dtype=np.int64)).reshape(-1)
    gt_labels = np.asarray(gt_labels if gt_labels is not None else np.zeros((len(gt_boxes),), dtype=np.int64)).reshape(-1)
    used, tp, fp, matched = set(), 0, 0, []
    for pi, pb in enumerate(pred_boxes):
        px1, py1, px2, py2 = pb; best, best_j = 0.0, None
        for j, gb in enumerate(gt_boxes):
            if j in used or (pi < len(pred_labels) and j < len(gt_labels) and int(pred_labels[pi]) != int(gt_labels[j])): continue
            gx1, gy1, gx2, gy2 = gb; ix1, iy1, ix2, iy2 = max(px1, gx1), max(py1, gy1), min(px2, gx2), min(py2, gy2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1); pa, ga = max(0.0, px2 - px1) * max(0.0, py2 - py1), max(0.0, gx2 - gx1) * max(0.0, gy2 - gy1)
            union = pa + ga - inter; iou = inter / union if union > 0 else 0.0
            if iou > best: best, best_j = iou, j
        if best_j is not None and best >= iou_threshold: used.add(best_j); tp += 1; matched.append(best)
        else: fp += 1
    fn = max(0, len(gt_boxes) - len(used)); mean_iou = float(np.mean(matched)) if matched else 0.0
    precision = tp / (tp + fp + 1e-6) if tp + fp > 0 else 0.0; recall = tp / (tp + fn + 1e-6) if tp + fn > 0 else 0.0
    return {"iou": mean_iou, "precision": precision, "recall": recall, "f1": (2 * precision * recall / (precision + recall + 1e-6)) if precision + recall > 0 else 0.0, "dice": (2 * tp / (2 * tp + fp + fn + 1e-6)) if 2 * tp + fp + fn > 0 else 0.0, "tp": tp, "fp": fp, "fn": fn}

async def _native_detection_training_loop(epochs, batch_size, learning_rate, label_path, safe_label, class_names, num_classes, backbone, model_name, model, optimizer, scheduler, device, mode, pos_weight_auto, model_type, grad_clip=1.0, early_stop=0, scheduler_kind="plateau"):
    best_metric, best_epoch, best_model_path = 0.0, 0, None
    training_state.is_training, training_state.stop_training, training_state.epoch, training_state.training_completed = True, False, 0, False
    try:
        models_dir = os.path.join(label_path, "models"); os.makedirs(models_dir, exist_ok=True)
        dataset_dir = os.path.join(label_path, "dataset"); train_dir, val_dir = os.path.join(dataset_dir, "train"), os.path.join(dataset_dir, "val")
        metadata_path = os.path.join(dataset_dir, "metadata.json"); tile_size = 64
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f: tile_size = int(json.load(f).get("tile_size", 64))
        train_dataset = MaskRCNNChipDataset(train_dir, tile_size=tile_size, num_classes=num_classes) if model_type == "mask_rcnn" else DetectionChipDataset(train_dir, num_classes=num_classes)
        val_dataset = None
        if os.path.exists(val_dir):
            try:
                candidate = MaskRCNNChipDataset(val_dir, tile_size=tile_size, num_classes=num_classes) if model_type == "mask_rcnn" else DetectionChipDataset(val_dir, num_classes=num_classes)
                if len(candidate) > 0: val_dataset = candidate
            except Exception: val_dataset = None
        if len(train_dataset) == 0: raise RuntimeError("Empty native detection dataset.")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=detection_collate)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=detection_collate) if val_dataset is not None else None
        instance_mode, metric_title = model_type == "mask_rcnn", "mask-IoU" if model_type == "mask_rcnn" else "box-IoU"
        await asyncio.sleep(0.1)
        for epoch in range(epochs):
            if training_state.stop_training: raise _StopTraining()
            await asyncio.sleep(0.02)
            training_state.epoch, training_state.progress_percent = epoch + 1, (epoch + 1) / max(epochs, 1) * 100.0
            model.train(); epoch_loss, nb = 0.0, 0
            for images, targets in train_loader:
                if training_state.stop_training: raise _StopTraining()
                await asyncio.sleep(0.01)
                images = [img.to(device) for img in images]
                targets = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in t.items()} for t in targets]
                optimizer.zero_grad(set_to_none=True)
                loss_dict = model(images, targets); losses = sum(loss for loss in loss_dict.values())
                if not torch.isfinite(losses): raise RuntimeError(f"Non-finite detector loss at epoch {epoch+1}: {losses.item()}")
                losses.backward()
                if grad_clip > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step(); epoch_loss += float(losses.item()); nb += 1
            avg_loss, val_metric, aggregate, per_sample_iou = epoch_loss / max(nb, 1), 0.0, {"tp": 0, "fp": 0, "fn": 0}, []
            if val_loader is not None:
                model.eval()
                with torch.no_grad():
                    for images, targets in val_loader:
                        await asyncio.sleep(0.01)
                        images_d = [img.to(device) for img in images]; predictions = model(images_d)
                        pred, tgt = predictions[0], targets[0]
                        if instance_mode:
                            scores, pm = pred.get("scores"), pred.get("masks")
                            if pm is None: pm_np, score_np = np.zeros((0, tile_size, tile_size), dtype=np.uint8), np.zeros((0,), dtype=np.float32)
                            else:
                                pm_np = pm.detach().cpu().numpy(); pm_np = pm_np[:, 0] if pm_np.ndim == 4 else pm_np
                                score_np = scores.detach().cpu().numpy() if scores is not None else np.ones((len(pm_np),), dtype=np.float32)
                                pm_np = pm_np[score_np >= 0.5] > 0.5
                            m = _match_instance_masks(pm_np, tgt["masks"].cpu().numpy() > 0, iou_threshold=0.5)
                        else:
                            pb, gb, pl, gl = pred.get("boxes"), tgt.get("boxes"), pred.get("labels"), tgt.get("labels")
                            pb_np = pb.detach().cpu().numpy() if pb is not None else np.zeros((0, 4)); pl_np = pl.detach().cpu().numpy() if pl is not None else np.zeros((len(pb_np),), dtype=np.int64)
                            ps = pred.get("scores")
                            if ps is not None:
                                ps_np = ps.detach().cpu().numpy(); keep = ps_np >= 0.5; pb_np, pl_np = pb_np[keep], pl_np[keep]
                            gb_np = gb.detach().cpu().numpy() if gb is not None else np.zeros((0, 4)); gl_np = gl.detach().cpu().numpy() if gl is not None else np.zeros((len(gb_np),), dtype=np.int64)
                            m = _match_boxes(pb_np, gb_np, iou_threshold=0.5, pred_labels=pl_np, gt_labels=gl_np)
                        per_sample_iou.append(float(m["iou"])); aggregate["tp"] += m["tp"]; aggregate["fp"] += m["fp"]; aggregate["fn"] += m["fn"]
            val_metric = float(np.mean(per_sample_iou)) if per_sample_iou else 0.0
            tp, fp, fn = aggregate["tp"], aggregate["fp"], aggregate["fn"]
            precision = tp / (tp + fp + 1e-6) if tp + fp > 0 else 0.0; recall = tp / (tp + fn + 1e-6) if tp + fn > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall + 1e-6) if precision + recall > 0 else 0.0
            training_state.train_losses.append(avg_loss); training_state.val_losses.append(avg_loss); training_state.ious.append(val_metric); training_state.best_map = max(training_state.best_map, val_metric)
            await broadcast_training_update({"status": "training", "epoch": epoch + 1, "total_epochs": epochs, "train_loss": avg_loss, "val_loss": avg_loss, "iou": val_metric, "best_iou": training_state.best_map, "best_epoch": best_epoch, "progress": training_state.progress_percent, "progress_percent": training_state.progress_percent, "train_losses": training_state.train_losses, "losses": training_state.train_losses, "val_losses": training_state.val_losses, "ious": training_state.ious, "precision": precision, "recall": recall, "f1": f1, "metric_name": metric_title})
            if scheduler is not None:
                try: scheduler.step(avg_loss) if scheduler_kind == "plateau" else scheduler.step()
                except Exception: pass
            if val_metric > best_metric or epoch == 0:
                best_metric, best_epoch, training_state.best_map = val_metric, epoch + 1, val_metric; best_model_path = os.path.join(models_dir, f"{safe_label}_best.pth")
                config = {"model_type": model_type, "architecture": type(model).__name__, "backbone": backbone, "n_classes": num_classes, "tile_size": tile_size, "detection_min_size": getattr(model.transform, "min_size", None), "detection_max_size": getattr(model.transform, "max_size", None)}
                torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": (scheduler.state_dict() if scheduler is not None else None), "epoch": epoch + 1, "best_iou": best_metric, "best_epoch": best_epoch, "num_classes": num_classes, "class_names": class_names, "backbone": backbone, "model_name": model_name, "mode": mode, "tile_size": tile_size, "target_size": tile_size, "model_type": model_type, "architecture": type(model).__name__, "model_config": config, "checkpoint_version": 2, "task": ("instance_segmentation" if instance_mode else "object_detection")}, best_model_path)
            if early_stop > 0 and (epoch + 1 - best_epoch) >= early_stop: break
        training_state.training_completed, training_state.status, training_state.progress_percent, training_state.model_path, training_state.is_training = True, "Completed", 100.0, best_model_path, False
        if training_state.current_project_id: update_project(training_state.current_project_id, {"has_model": True, "model_path": best_model_path, "evaluation_score": best_metric, "chips_count": len(train_dataset)})
        await broadcast_training_update({"status": "completed", "epoch": training_state.epoch, "best_iou": best_metric, "best_epoch": best_epoch, "model_path": best_model_path, "metric_name": metric_title})
    except _StopTraining: training_state.is_training, training_state.status = False, "Stopped"; await broadcast_training_update({"status": "stopped", "epoch": training_state.epoch, "best_iou": training_state.best_map})
    except Exception as e: training_state.is_training, training_state.status = False, "Failed"; traceback.print_exc(); await broadcast_training_update({"status": "failed", "error": str(e)})

async def real_maskrcnn_training_loop(epochs, batch_size, learning_rate, label_path, safe_label, class_names, num_classes, backbone, model_name, model, optimizer, scheduler, device, mode, pos_weight_auto, model_type, grad_clip=1.0, early_stop=0, scheduler_kind="plateau"):
    await _native_detection_training_loop(epochs, batch_size, learning_rate, label_path, safe_label, class_names, num_classes, backbone, model_name, model, optimizer, scheduler, device, mode, pos_weight_auto, "mask_rcnn", grad_clip=grad_clip, early_stop=early_stop, scheduler_kind=scheduler_kind)

async def actual_detection_training_loop(epochs, batch_size, learning_rate, label_path, safe_label, class_names, num_classes, backbone, model_name, model, optimizer, scheduler, device, mode, pos_weight_auto, model_type, grad_clip=1.0, early_stop=0, scheduler_kind="plateau"):
    await _native_detection_training_loop(epochs, batch_size, learning_rate, label_path, safe_label, class_names, num_classes, backbone, model_name, model, optimizer, scheduler, device, mode, pos_weight_auto, model_type, grad_clip=grad_clip, early_stop=early_stop, scheduler_kind=scheduler_kind)

async def broadcast_training_update(payload: dict):
    dead = []
    for ws in active_connections:
        try: await ws.send_json(payload)
        except Exception: dead.append(ws)
    for ws in dead:
        try: active_connections.remove(ws)
        except ValueError: pass

@app.websocket("/ws/training")
async def websocket_training(ws: WebSocket):
    await ws.accept()
    if not await websocket_api_key_ok(ws): return
    active_connections.append(ws)
    try:
        await ws.send_json({"status": "completed" if training_state.training_completed else ("training" if training_state.is_training else str(training_state.status).lower()), "is_training": training_state.is_training, "training_completed": training_state.training_completed, "epoch": training_state.epoch, "total_epochs": training_state.total_epochs, "progress": training_state.progress_percent, "progress_percent": training_state.progress_percent, "best_iou": training_state.best_map, "best_epoch": getattr(training_state, "best_epoch", None), "model_path": training_state.model_path, "train_losses": training_state.train_losses, "losses": training_state.train_losses, "val_losses": training_state.val_losses, "ious": training_state.ious})
        while True:
            try:
                msg = await asyncio.wait_for(ws.receive_text(), timeout=30.0)
                if msg == "ping": await ws.send_json({"status": "pong", "is_training": training_state.is_training, "epoch": training_state.epoch, "total_epochs": training_state.total_epochs})
            except asyncio.TimeoutError: await ws.send_json({"status": "heartbeat", "is_training": training_state.is_training, "epoch": training_state.epoch, "total_epochs": training_state.total_epochs})
    except WebSocketDisconnect: pass
    except Exception as e: pass
    finally:
        if ws in active_connections: active_connections.remove(ws)

@app.websocket("/ws")
async def websocket_ws_compat(ws: WebSocket): await websocket_training(ws)

@app.post("/api/train/stop", dependencies=[Depends(require_api_key), Depends(require_capability("can_train"))])
async def stop_training():
    if not training_state.is_training: return JSONResponse({"success": False, "error": "Not training"})
    training_state.stop_training, training_state.status = True, "Stopping"
    task = training_tasks.get("current")
    if task and not task.done():
        try: task.cancel()
        except Exception: pass
    return JSONResponse({"success": True, "message": "Stop signal sent"})

@app.get("/api/logs", dependencies=[Depends(require_api_key)])
async def get_logs(): return JSONResponse({"success": True, "logs": training_logs})

@app.post("/api/logs/clear", dependencies=[Depends(require_api_key)])
async def clear_logs():
    training_logs.clear(); return JSONResponse({"success": True})

def _evaluate_single_native_prediction(pred, target, mask_model=True, score_threshold=0.5):
    if mask_model:
        masks, scores = pred.get("masks") if isinstance(pred, dict) else None, pred.get("scores") if isinstance(pred, dict) else None
        if masks is None: pred_masks = np.zeros((0, target["masks"].shape[-2], target["masks"].shape[-1]), dtype=np.uint8)
        else:
            pm = masks.detach().cpu().numpy(); pm = pm[:, 0] if pm.ndim == 4 else pm
            if scores is not None: pm = pm[scores.detach().cpu().numpy() >= score_threshold]
            pred_masks = pm > 0.5
        return _match_instance_masks(pred_masks, target["masks"].detach().cpu().numpy() > 0, iou_threshold=0.5)
    boxes, scores = pred.get("boxes") if isinstance(pred, dict) else None, pred.get("scores") if isinstance(pred, dict) else None
    pb = boxes.detach().cpu().numpy() if boxes is not None else np.zeros((0, 4), dtype=np.float32)
    if scores is not None: pb = pb[scores.detach().cpu().numpy() >= score_threshold]
    gb_t = target.get("boxes"); gb = gb_t.detach().cpu().numpy() if gb_t is not None else np.zeros((0, 4), dtype=np.float32)
    return _match_boxes(pb, gb, iou_threshold=0.5)

@app.post("/api/evaluate", dependencies=[Depends(require_api_key)])
async def evaluate_model(data: Dict):
    try:
        project_id, label = data.get("project_id") or training_state.current_project_id, data.get("label") or training_state.current_label or training_state.selected_label
        model_path, threshold, split = data.get("model_path") or training_state.model_path, float(data.get("threshold", 0.5)), data.get("split", "test")
        if not project_id or not label or not model_path or not os.path.exists(model_path): return JSONResponse({"success": False, "error": "Project ID, Label, and Model checkpoint required"}, status_code=400)
        safe, lp, dataset_dir = label.lower().replace(" ", "_"), get_label_path(label.lower().replace(" ", "_")), os.path.join(get_label_path(label.lower().replace(" ", "_")), "dataset")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model, meta = load_checkpoint(model_path, device)
        model_type = normalize_model_type(meta.get("model_type", "unet")); is_native_detector = (hasattr(model, "roi_heads") or model_type in DETECTION_MODELS or model_type in INSTANCE_MODELS or meta.get("architecture") in ("MaskRCNN", "FasterRCNN"))
        sample_outputs = []
        if model_type == "yolo":
            split_dir = os.path.join(dataset_dir, split) if os.path.isdir(os.path.join(dataset_dir, split)) else os.path.join(dataset_dir, "val")
            img_dir, lbl_dir = os.path.join(split_dir, "images"), os.path.join(split_dir, "yolo_labels")
            if not os.path.isdir(img_dir) or not os.path.isdir(lbl_dir): return JSONResponse({"success": False, "error": "No YOLO evaluation split/labels found"}, status_code=400)
            tp = fp = fn = count = 0; ious = []
            for fname in sorted(os.listdir(img_dir)):
                if not fname.lower().endswith((".png", ".jpg", ".jpeg")): continue
                stem = os.path.splitext(fname)[0]; lp_txt = os.path.join(lbl_dir, stem + ".txt"); gt_boxes, gt_labels = [], []
                if os.path.exists(lp_txt):
                    for line in open(lp_txt, "r", encoding="utf-8"):
                        vals = line.strip().split()
                        if len(vals) != 5: continue
                        cid, cx, cy, w, h = map(float, vals); W = H = float(meta.get("tile_size", training_state.tile_size or 640))
                        gt_boxes.append([(cx-w/2)*W, (cy-h/2)*H, (cx+w/2)*W, (cy+h/2)*H]); gt_labels.append(int(cid)+1)
                try:
                    results = await asyncio.to_thread(model.predict, source=os.path.join(img_dir, fname), conf=threshold, verbose=False, device=(0 if torch.cuda.is_available() else "cpu"))
                    r, pb, pl = results[0], [], []
                    if getattr(r, "boxes", None) is not None:
                        xyxy = r.boxes.xyxy.detach().cpu().numpy() if hasattr(r.boxes.xyxy, "detach") else np.asarray(r.boxes.xyxy)
                        cls = r.boxes.cls.detach().cpu().numpy() if hasattr(r.boxes.cls, "detach") else np.asarray(r.boxes.cls)
                        pb, pl = xyxy.tolist(), (cls.astype(int)+1).tolist()
                    m = _match_boxes(pb, gt_boxes, 0.5, np.asarray(pl), np.asarray(gt_labels))
                    tp += m["tp"]; fp += m["fp"]; fn += m["fn"]
                    if m["iou"] > 0: ious.append(m["iou"])
                    count += 1
                    if len(sample_outputs) < 12: sample_outputs.append({"filename": fname, "iou": float(m["iou"]), "tp": int(m["tp"]), "fp": int(m["fp"]), "fn": int(m["fn"])})
                except Exception as e: pass
            precision = tp / (tp + fp + 1e-6) if tp + fp else 0.0; recall = tp / (tp + fn + 1e-6) if tp + fn else 0.0
            summary = {"iou": float(np.mean(ious)) if ious else 0.0, "precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall + 1e-6) if precision + recall else 0.0, "dice": 2 * tp / (2 * tp + fp + fn + 1e-6) if 2 * tp + fp + fn else 0.0, "tp": tp, "fp": fp, "fn": fn, "num_samples": count, "threshold": threshold, "split": split, "model_type": "yolo", "architecture": "UltralyticsYOLO", "native": True, "metric": "box_iou"}
        elif model_type == "classic_cnn":
            split_dir = os.path.join(dataset_dir, split) if os.path.isdir(os.path.join(dataset_dir, split)) else os.path.join(dataset_dir, "val")
            ds = ClassificationChipDataset(split_dir, max(2, int(meta.get("num_classes", 2)))); loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
            correct = total = 0; model.eval()
            with torch.no_grad():
                for idx, (images, labels) in enumerate(loader):
                    logits = model(images.to(device)); logits = logits[0] if isinstance(logits, (list, tuple)) else logits
                    pred = torch.argmax(logits, dim=1).cpu().numpy().tolist()[0]; gt = int(labels.item())
                    total += 1; correct += int(pred == gt)
                    if idx < 12: sample_outputs.append({"filename": ds.samples[idx][0], "true_class": gt, "predicted_class": pred, "correct": bool(pred == gt)})
            acc = correct / max(total, 1); summary = {"accuracy": acc, "iou": acc, "precision": acc, "recall": acc, "f1": acc, "dice": acc, "correct": correct, "total": total, "threshold": threshold, "split": split, "model_type": "classic_cnn", "architecture": meta.get("architecture"), "native": True, "metric": "classification_accuracy"}
        elif is_native_detector:
            split_dir = os.path.join(dataset_dir, split) if os.path.exists(os.path.join(dataset_dir, split)) else os.path.join(dataset_dir, "val")
            if not os.path.exists(split_dir): return JSONResponse({"success": False, "error": "No detection evaluation split found"}, status_code=400)
            metadata_path = os.path.join(dataset_dir, "metadata.json"); tile_size = int(json.load(open(metadata_path, "r", encoding="utf-8")).get("tile_size", 64)) if os.path.exists(metadata_path) else meta.get("tile_size", 64)
            ds = MaskRCNNChipDataset(split_dir, tile_size=tile_size, num_classes=max(2, int(meta["num_classes"]))) if model_type == "mask_rcnn" else DetectionChipDataset(split_dir, num_classes=max(2, int(meta["num_classes"])))
            loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=detection_collate)
            mask_model = model_type == "mask_rcnn" or meta.get("architecture") == "MaskRCNN"; sum_iou = sum_p = sum_r = sum_f1 = sum_dice = 0.0; total_tp = total_fp = total_fn = count = 0
            model.eval()
            with torch.no_grad():
                for idx, (images, targets) in enumerate(loader):
                    images_d = [img.to(device) for img in images]; preds = model(images_d)
                    m = _evaluate_single_native_prediction(preds[0], targets[0], mask_model=mask_model, score_threshold=threshold)
                    sum_iou += m["iou"]; sum_p += m["precision"]; sum_r += m["recall"]; sum_f1 += m["f1"]; sum_dice += m["dice"]
                    total_tp += m["tp"]; total_fp += m["fp"]; total_fn += m["fn"]; count += 1
                    if idx < 12: sample_outputs.append({"filename": ds.stems[idx], "iou": float(m["iou"]), "tp": int(m["tp"]), "fp": int(m["fp"]), "fn": int(m["fn"])})
            summary = {"iou": sum_iou / max(count, 1), "precision": sum_p / max(count, 1), "recall": sum_r / max(count, 1), "f1": sum_f1 / max(count, 1), "dice": sum_dice / max(count, 1), "tp": total_tp, "fp": total_fp, "fn": total_fn, "num_samples": count, "threshold": threshold, "split": split, "model_type": model_type, "architecture": meta.get("architecture"), "native": True, "metric": "mask_iou" if mask_model else "box_iou"}
        else:
            eval_images_dir, eval_masks_dir = os.path.join(dataset_dir, split, "images"), os.path.join(dataset_dir, split, "masks")
            if not os.path.exists(eval_images_dir): eval_images_dir, eval_masks_dir = os.path.join(dataset_dir, "val", "images"), os.path.join(dataset_dir, "val", "masks")
            if not os.path.exists(eval_images_dir): return JSONResponse({"success": False, "error": "No evaluation split found"}, status_code=400)
            mode, tile_size, eval_tf = meta["mode"], int(meta["target_size"]), get_eval_transform(meta["mode"])
            eval_files = [f for f in os.listdir(eval_images_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")) and not f.endswith("_vis.png") and not f.endswith("_ov.png") and not f.endswith("_ids.png")]
            metrics = {"iou": [], "precision": [], "recall": [], "f1": [], "dice": []}; model.eval()
            for i, fname in enumerate(eval_files):
                img_pil = Image.open(os.path.join(eval_images_dir, fname)).convert("RGB").resize((tile_size, tile_size), Image.BILINEAR)
                img_t = eval_tf(img_pil).unsqueeze(0).to(device)
                with torch.no_grad():
                    out = model(img_t); out = out[0] if isinstance(out, (list, tuple)) else out
                    prob = torch.sigmoid(out)[0, 0].cpu().numpy() if mode == "binary" else torch.softmax(out, dim=1)[0, 1].cpu().numpy()
                pred = (prob > threshold).astype(np.uint8)
                mp = os.path.join(eval_masks_dir, fname)
                if os.path.exists(mp): gt = (np.asarray(Image.open(mp).convert("L").resize((tile_size, tile_size), Image.NEAREST)) > 127).astype(np.uint8)
                else: gt = np.zeros_like(pred)
                tp = int(((pred == 1) & (gt == 1)).sum()); fp = int(((pred == 1) & (gt == 0)).sum()); fn = int(((pred == 0) & (gt == 1)).sum())
                iou = tp / (tp + fp + fn + 1e-6) if tp + fp + fn > 0 else 0.0; precision = tp / (tp + fp + 1e-6) if tp + fp > 0 else 0.0; recall = tp / (tp + fn + 1e-6) if tp + fn > 0 else 0.0
                metrics["iou"].append(iou); metrics["precision"].append(precision); metrics["recall"].append(recall); metrics["f1"].append(2 * precision * recall / (precision + recall + 1e-6) if precision + recall > 0 else 0.0); metrics["dice"].append(2 * tp / (2 * tp + fp + fn + 1e-6) if 2 * tp + fp + fn > 0 else 0.0)
                if i < 12: sample_outputs.append({"filename": fname, "iou": float(iou)})
            mean = lambda v: float(np.mean(v)) if v else 0.0
            summary = {"iou": mean(metrics["iou"]), "precision": mean(metrics["precision"]), "recall": mean(metrics["recall"]), "f1": mean(metrics["f1"]), "dice": mean(metrics["dice"]), "num_samples": len(eval_files), "threshold": threshold, "split": split, "model_type": model_type, "architecture": meta.get("architecture"), "native": False, "metric": "pixel_iou"}
        training_state.evaluation_metrics, training_state.best_map = summary, float(summary["iou"])
        eval_json_path = os.path.join(lp, "evaluation", "metrics.json"); os.makedirs(os.path.dirname(eval_json_path), exist_ok=True)
        with open(eval_json_path, "w", encoding="utf-8") as f: json.dump({"summary": summary, "samples": sample_outputs}, f, indent=2)
        update_project(project_id, {"evaluation_score": summary["iou"]})
        return JSONResponse({"success": True, "metrics": summary, "samples": sample_outputs})
    except Exception as e:
        traceback.print_exc(); return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.post("/api/export/geojson", dependencies=[Depends(require_api_key)])
async def export_geojson(data: Dict):
    try:
        label, features = data.get("label") or training_state.current_label, data.get("features")
        if not label or not features: return JSONResponse({"success": False, "error": "Label and features required"}, status_code=400)
        safe, lp = label.lower().replace(" ", "_"), get_label_path(label.lower().replace(" ", "_"))
        out_dir = os.path.join(lp, "evaluation", "geojson"); os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{safe}_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.geojson")
        with open(out_path, "w") as f: json.dump({"type": "FeatureCollection", "features": features, "bounds": training_state.image_bounds_latlon or data.get("bounds"), "crs": training_state.image_crs, "label": safe, "created": datetime.now().isoformat()}, f, indent=2)
        return JSONResponse({"success": True, "filepath": out_path, "filename": os.path.basename(out_path), "count": len(features)})
    except Exception as e:
        traceback.print_exc(); return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.post("/api/export/shapefile", dependencies=[Depends(require_api_key)])
async def export_shapefile(data: Dict):
    try:
        label, features = data.get("label") or training_state.current_label, data.get("features")
        if not label or not features: return JSONResponse({"success": False, "error": "Label and features required"}, status_code=400)
        geoms = []
        for feat in features:
            g = feat.get("geometry")
            if g and g.get("type") == "Polygon" and g.get("coordinates"):
                try: geoms.append(ShapelyPolygon(g["coordinates"][0]))
                except Exception: continue
        if not geoms: return JSONResponse({"success": False, "error": "No valid polygons"}, status_code=400)
        safe, lp = label.lower().replace(" ", "_"), get_label_path(label.lower().replace(" ", "_"))
        out_dir = os.path.join(lp, "evaluation", "shp"); os.makedirs(out_dir, exist_ok=True)
        shp_path = os.path.join(out_dir, f"{safe}_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.shp")
        gpd.GeoDataFrame({"class": [safe] * len(geoms)}, geometry=geoms, crs=training_state.image_crs or "EPSG:4326").to_file(shp_path)
        zip_path = shp_path.replace(".shp", ".zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            base = shp_path.replace(".shp", "")
            for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
                if os.path.exists(base + ext): zf.write(base + ext, os.path.basename(base + ext))
        return JSONResponse({"success": True, "filepath": shp_path, "zip_path": zip_path, "filename": os.path.basename(zip_path), "count": len(geoms)})
    except Exception as e:
        traceback.print_exc(); return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.post("/api/export/geotiff", dependencies=[Depends(require_api_key)])
async def export_geotiff(data: Dict):
    try:
        if not HAS_RASTERIO: return JSONResponse({"success": False, "error": "rasterio not installed"}, status_code=400)
        mask_path, bounds, label = data.get("mask_path"), data.get("bounds") or training_state.image_bounds_latlon, data.get("label") or training_state.current_label or "output"
        if not mask_path or not os.path.exists(mask_path): return JSONResponse({"success": False, "error": "Mask not found"}, status_code=400)
        if not bounds: return JSONResponse({"success": False, "error": "Bounds required"}, status_code=400)
        mask = np.array(Image.open(mask_path).convert("L")); h, w = mask.shape
        out_dir = os.path.join(get_label_path(label.lower().replace(" ", "_")), "evaluation", "masks"); os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{label.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tif")
        with rasterio.open(out_path, "w", driver="GTiff", height=h, width=w, count=1, dtype=mask.dtype, crs="EPSG:4326", transform=rio_from_bounds(float(bounds["west"]), float(bounds["south"]), float(bounds["east"]), float(bounds["north"]), w, h), compress="lzw") as dst: dst.write(mask, 1)
        return JSONResponse({"success": True, "filepath": out_path, "filename": os.path.basename(out_path)})
    except Exception as e:
        traceback.print_exc(); return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.get("/api/dataset/{label}/preview", dependencies=[Depends(require_api_key)])
async def dataset_preview(label: str, split: str = "train", limit: int = 20):
    try:
        safe, lp = label.lower().replace(" ", "_"), get_label_path(label.lower().replace(" ", "_"))
        ds, images_dir, masks_dir, overlay_dir = os.path.join(lp, "dataset", split), os.path.join(lp, "dataset", split, "images"), os.path.join(lp, "dataset", split, "masks"), os.path.join(lp, "dataset", split, "masks_overlay")
        if not os.path.exists(images_dir): return JSONResponse({"success": False, "error": "Split not found"}, status_code=404)
        items = [{"filename": f, "image": f"/labels/{safe}/dataset/{split}/images/{f}", "mask": f"/labels/{safe}/dataset/{split}/masks/{f}" if os.path.exists(os.path.join(masks_dir, f)) else None, "overlay": f"/labels/{safe}/dataset/{split}/masks_overlay/{f}" if os.path.exists(os.path.join(overlay_dir, f)) else None} for f in sorted([f for f in os.listdir(images_dir) if f.endswith((".png", ".jpg", ".jpeg")) and not f.endswith("_vis.png") and not f.endswith("_ov.png") and not f.endswith("_ids.png")])[:limit]]
        return JSONResponse({"success": True, "items": items, "count": len(items), "split": split, "label": safe})
    except Exception as e:
        traceback.print_exc(); return JSONResponse({"success": False, "error": str(e)}, status_code=500)

app.mount("/labels", StaticFiles(directory="labels"), name="labels")

@app.get("/api/health")
async def health(): return JSONResponse({"success": True, "status": "ok", "training": training_state.is_training, "device": "cuda" if torch.cuda.is_available() else "cpu", "has_rasterio": HAS_RASTERIO, "has_torchmetrics_map": HAS_TORCHMETRICS_MAP})

@app.get("/api/annotations/files", dependencies=[Depends(require_api_key)])
async def api_annotation_files(project_id: Optional[str] = None):
    try:
        files, scan_dirs = [], ["annotations"] if os.path.exists("annotations") else []
        if project_id:
            proj_ann = os.path.join("projects", project_id, "annotations")
            if os.path.exists(proj_ann): scan_dirs.append(proj_ann)
        seen = set()
        for d in scan_dirs:
            for fname in sorted(os.listdir(d)):
                if not fname.endswith(".geojson"): continue
                fp = os.path.join(d, fname); key = os.path.abspath(fp)
                if key in seen: continue
                seen.add(key)
                try:
                    with open(fp, "r") as f: data = json.load(f)
                except Exception: continue
                data_pid = data.get("project_id")
                if project_id and data_pid and data_pid != project_id: continue
                feats = data.get("features", []) or []; lbl = data.get("label")
                if not lbl or lbl == "unknown":
                    if feats: lbl = feats[0].get("properties", {}).get("class_name") or feats[0].get("properties", {}).get("label")
                files.append({"filename": fname, "filepath": fp, "label": lbl or "unlabeled", "features_count": len(feats), "count": len(feats), "project_id": data_pid or "", "bounds": data.get("bounds"), "crs": data.get("crs", "EPSG:4326"), "modified": datetime.fromtimestamp(os.path.getmtime(fp)).strftime("%Y-%m-%d %H:%M:%S")})
        files.sort(key=lambda x: x.get("modified", ""), reverse=True)
        return JSONResponse({"success": True, "files": files, "count": len(files)})
    except Exception as e:
        traceback.print_exc(); return JSONResponse({"success": False, "files": [], "count": 0, "error": str(e)}, status_code=500)

@app.post("/api/annotations/load_file", dependencies=[Depends(require_api_key)])
async def api_load_annotation_file(data: Dict):
    try:
        filename, filepath, project_id = data.get("filename"), data.get("filepath"), data.get("project_id")
        candidates = [filepath] if filepath else []
        if filename:
            candidates.append(os.path.join("annotations", filename))
            if project_id: candidates.append(os.path.join("projects", project_id, "annotations", filename))
        safe_candidates = []
        for c in candidates:
            try: safe_candidates.append(depcfg.resolve_within_roots(c, depcfg.allowed_roots()))
            except ValueError: continue
        if candidates and not safe_candidates: return JSONResponse({"success": False, "error": "Annotation path is outside the permitted directories."}, status_code=400)
        target = next((p for p in safe_candidates if p and os.path.exists(p)), None)
        if not target: return JSONResponse({"success": False, "error": f"File not found. Tried: {candidates}"}, status_code=404)
        with open(target, "r") as f: geojson = json.load(f)
        features = geojson.get("features", []) or []; lbl = geojson.get("label")
        if not lbl or lbl == "unknown":
            if features: lbl = features[0].get("properties", {}).get("class_name") or features[0].get("properties", {}).get("label")
        training_state.annotation_data, training_state.annotation_file_path, training_state.total_annotations, training_state.current_label, training_state.selected_label = geojson, target, len(features), lbl or "unlabeled", lbl or "unlabeled"
        if geojson.get("bounds"): training_state.image_bounds_latlon = geojson["bounds"]
        return JSONResponse({"success": True, "filename": os.path.basename(target), "filepath": target, "label": lbl or "unlabeled", "features": features, "count": len(features), "bounds": geojson.get("bounds")})
    except Exception as e:
        traceback.print_exc(); return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.post("/api/annotations/merge", dependencies=[Depends(require_api_key)])
async def api_merge_annotations(data: Dict):
    try:
        project_id, label = data.get("project_id") or training_state.current_project_id, data.get("label") or training_state.current_label or "st"
        if not project_id: return JSONResponse({"success": False, "error": "Project ID required"}, status_code=400)
        safe, all_features = label.lower().replace(" ", "_"), []
        if os.path.exists("annotations"):
            for fname in os.listdir("annotations"):
                if not fname.endswith(".geojson"): continue
                try:
                    with open(os.path.join("annotations", fname), "r") as f: d = json.load(f)
                except Exception: continue
                pid = d.get("project_id")
                if pid and pid != project_id: continue
                for feat in d.get("features", []) or []:
                    feat.setdefault("properties", {})["class_name"] = feat["properties"].get("class_name", safe)
                    all_features.append(feat)
        if not all_features: return JSONResponse({"success": False, "error": "No annotations to merge"}, status_code=400)
        out_dir = os.path.join("labels", safe, "evaluation", "geojson"); os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"merged_{project_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.geojson")
        with open(out_path, "w") as f: json.dump({"type": "FeatureCollection", "features": all_features, "bounds": training_state.image_bounds_latlon, "crs": training_state.image_crs, "label": safe, "project_id": project_id, "created": datetime.now().isoformat()}, f, indent=2)
        return JSONResponse({"success": True, "num_features": len(all_features), "geojson_path": out_path, "download_geojson_url": f"/labels/{safe}/evaluation/geojson/{os.path.basename(out_path)}", "download_shp_url": None})
    except Exception as e:
        traceback.print_exc(); return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.post("/api/map/set_training_zoom", dependencies=[Depends(require_api_key)])
async def api_set_training_zoom(data: Dict):
    try:
        zoom, service_id, lat = int(data.get("zoom", 12)), data.get("service_id", "satellite"), float(data.get("lat", 0.0))
        gsd = 156543.03392 * math.cos(math.radians(lat)) / (2 ** zoom) / 256.0
        training_state.current_zoom, training_state.current_map_service, training_state.image_crs, training_state.training_zoom, training_state.training_gsd, training_state.training_service = zoom, service_id, "EPSG:4326", zoom, gsd, service_id
        if training_state.current_project_id: update_project(training_state.current_project_id, {"training_zoom": zoom, "training_gsd": gsd, "training_service": service_id})
        return JSONResponse({"success": True, "training_zoom": zoom, "training_gsd": float(gsd), "training_service": service_id})
    except Exception as e:
        traceback.print_exc(); return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.post("/api/train/overfit_test", dependencies=[Depends(require_api_key), Depends(require_capability("can_train"))])
async def api_overfit_test(data: Dict):
    try:
        project_id, label = data.get("project_id") or training_state.current_project_id, data.get("label") or training_state.current_label
        if not project_id or not label: return JSONResponse({"success": False, "error": "Project and Label required"}, status_code=400)
        lp, train_dir, img_dir, msk_dir = get_label_path(label.lower().replace(" ", "_")), os.path.join(get_label_path(label.lower().replace(" ", "_")), "dataset", "train"), os.path.join(get_label_path(label.lower().replace(" ", "_")), "dataset", "train", "images"), os.path.join(get_label_path(label.lower().replace(" ", "_")), "dataset", "train", "masks")
        files = [f for f in sorted(os.listdir(img_dir)) if f.endswith(".png") and not f.endswith("_vis.png") and not f.endswith("_ov.png") and not f.endswith("_ids.png") and os.path.exists(os.path.join(msk_dir, f))][:2]
        if len(files) < 1: return JSONResponse({"success": False, "error": "Need at least 1 paired chip"}, status_code=400)
        pos_w, fg_ratio = compute_dataset_pos_weight(msk_dir, max_samples=50); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ds = BinaryChipDataset(img_dir, msk_dir, target_size=256, augment=False); ds.files = files; loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
        model = build_model(mode="binary", num_classes=2, backbone="unet", pretrained=False, bias_init=compute_bias_init(fg_ratio), model_type=data.get("model_type", "unet"), tile_size=256, real_detector=False).to(device)
        optimizer, criterion, history = optim.AdamW(model.parameters(), lr=1e-3), CombinedLoss(bce_weight=0.3, dice_weight=0.7, pos_weight=pos_w), []
        model.train()
        for step in range(1, 21):
            total, nb = 0.0, 0
            for imgs, msks in loader:
                imgs, msks = imgs.to(device), msks.to(device); optimizer.zero_grad()
                out = model(imgs); loss = criterion(out, msks); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
                total += loss.item(); nb += 1
            if step % 5 == 0 or step == 1:
                model.eval()
                with torch.no_grad():
                    imgs, msks = next(iter(loader)); imgs, msks = imgs.to(device), msks.to(device)
                    out = model(imgs); prob = torch.sigmoid(out); pred, tgt = (prob > 0.5).float(), (msks > 0.5).float()
                    inter, union = ((pred == 1) & (tgt == 1)).sum().item(), ((pred == 1) | (tgt == 1)).sum().item()
                    history.append({"step": step, "loss": float(total / max(nb, 1)), "pred_mean": float(prob.mean().item()), "pred_fg": int((pred > 0).sum().item()), "gt_fg": int((tgt > 0).sum().item()), "iou": float(inter / union if union > 0 else 0.0)})
                model.train()
        verdict = "✅ Model can overfit — pipeline looks healthy." if history[-1]["iou"] > 0.5 else "⚠️ Model did NOT overfit 2 chips — check data / loss / lr."
        return JSONResponse({"success": True, "chips": files, "pos_weight": float(pos_w), "fg_ratio": float(fg_ratio), "history": history, "verdict": verdict})
    except Exception as e:
        traceback.print_exc(); return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.get("/api/uploaded_datasets", dependencies=[Depends(require_api_key)])
async def api_uploaded_datasets(): return JSONResponse({"success": True, "sessions": load_uploaded_datasets().get("sessions", [])})

@app.delete("/api/uploaded_datasets/{sid}", dependencies=[Depends(require_api_key)])
async def api_delete_uploaded_dataset(sid: str):
    data = load_uploaded_datasets(); before = len(data.get("sessions", []))
    data["sessions"] = [s for s in data.get("sessions", []) if s.get("session_id") != sid]
    save_uploaded_datasets(data)
    return JSONResponse({"success": True, "deleted": before - len(data["sessions"])})

# =========================================================
# MODELS LISTING & EXPORT
# =========================================================

@app.get("/api/models/all", dependencies=[Depends(require_api_key)])
async def list_models_all():
    return await list_models()

@app.get("/api/models", dependencies=[Depends(require_api_key)])
async def list_models():
    """
    Return a deterministic model registry for the Load Trained Model dialog.

    A checkpoint may be reachable through more than one configured discovery
    root. Collect candidates first, canonicalise paths, and apply a fixed
    source precedence before reading metadata. Never rely on os.listdir(),
    os.walk(), or set iteration order for the dropdown order.
    """
    try:
        print("--> Inside list_models() !!", flush=True)
        candidates = {}
        source_priority = {"project": 0, "imported": 1, "dlpk": 2, "saved": 3}

        def add_candidate(fp, source):
            if not fp:
                return
            # IMMEDIATELY IGNORE TorchScript artifacts so we don't try to unpickle them
            if ".torchscript" in fp.lower():
                return
            # Ignore vendored library weight files and internal python packages
            fp_norm = fp.lower().replace("\\", "/")
            if any(ign in fp_norm for ign in ("/torchmetrics/", "/segment-anything/", "/segment_anything/", "/site-packages/", "/__pycache__/", "/.git/")):
                return
            try:
                if not os.path.isfile(fp):
                    return
                if not fp.lower().endswith((".pth", ".pt", ".ckpt", ".dlpk")):
                    return
                canonical = os.path.normcase(os.path.realpath(os.path.abspath(fp)))
            except (OSError, TypeError, ValueError):
                return

            current = candidates.get(canonical)
            candidate = {
                "path": os.path.abspath(fp),
                "source": source,
                "priority": source_priority.get(source, 99),
            }
            if current is None or candidate["priority"] < current["priority"]:
                candidates[canonical] = candidate

        # Project checkpoints: deterministic label and filename traversal.
        labels_root = "labels"
        if os.path.isdir(labels_root):
            for label_name in sorted(os.listdir(labels_root), key=str.casefold):
                model_dir = os.path.join(labels_root, label_name, "models")
                if not os.path.isdir(model_dir):
                    continue
                for filename in sorted(os.listdir(model_dir), key=str.casefold):
                    add_candidate(os.path.join(model_dir, filename), "project")

        # Imported native checkpoints.
        imported_dir = os.path.join("models", "imported")
        if os.path.isdir(imported_dir):
            for filename in sorted(os.listdir(imported_dir), key=str.casefold):
                add_candidate(os.path.join(imported_dir, filename), "imported")

        # DLPK extraction roots. Scan extracted model packages (1 directory deep).
        dlpk_roots = [depcfg.DLPK_IMPORT_DIR, imported_dir]
        for dlpk_root in dlpk_roots:
            if not dlpk_root or not os.path.isdir(dlpk_root):
                continue
            for entry in sorted(os.listdir(dlpk_root), key=str.casefold):
                if entry.startswith("_") or entry.startswith("."):
                    continue
                entry_path = os.path.join(dlpk_root, entry)
                if os.path.isfile(entry_path):
                    add_candidate(entry_path, "dlpk")
                elif os.path.isdir(entry_path):
                    for filename in sorted(os.listdir(entry_path), key=str.casefold):
                        add_candidate(os.path.join(entry_path, filename), "dlpk")

        # User-saved copies.
        standard_saved = os.path.join("models", "saved")
        if os.path.isdir(standard_saved):
            for filename in sorted(os.listdir(standard_saved), key=str.casefold):
                add_candidate(os.path.join(standard_saved, filename), "saved")

        print(f"--> [list_models] CANDIDATES: {len(candidates)}", flush=True)
        models = []
        for canonical_path in sorted(candidates, key=str.casefold):
            candidate = candidates[canonical_path]
            fp, source = candidate["path"], candidate["source"]
            filename = os.path.basename(fp)
            print(f"--> [list_models] {source} {filename}", flush=True)

            is_dlpk = fp.lower().endswith(".dlpk") or source == "dlpk"
            ckpt = {}
            if fp.lower().endswith(".dlpk"):
                emd_data = _read_dlpk_metadata(fp)
                if emd_data:
                    b_info = emd_data.get("BraeinAI", {}) if isinstance(emd_data.get("BraeinAI"), dict) else {}
                    params = emd_data.get("ModelParameters", {}) if isinstance(emd_data.get("ModelParameters"), dict) else {}
                    ckpt = {
                        "model_name": b_info.get("ModelName") or os.path.splitext(filename)[0],
                        "model_type": b_info.get("ModelTypeBraein", "mask_rcnn"),
                        "architecture": b_info.get("Architecture", emd_data.get("ModelConfiguration", "MaskRCNN")),
                        "backbone": b_info.get("Backbone", params.get("backbone", "resnet50")),
                        "mode": b_info.get("Mode", params.get("mode", "binary")),
                        "epoch": b_info.get("Epoch", 0),
                        "best_iou": float(b_info.get("BestIoU", 0.0) or 0.0),
                        "tile_size": int(emd_data.get("ImageHeight", 256)),
                        "class_names": _class_names_from_emd(emd_data),
                        "is_dlpk": True,
                        "extra_info": b_info,
                    }
                else:
                    if source == "dlpk":
                        continue
                    ckpt = {"model_name": os.path.splitext(filename)[0], "is_dlpk": True}
            elif source == "dlpk" or _read_adjacent_emd_metadata(fp):
                emd_data = _read_adjacent_emd_metadata(fp)
                if emd_data:
                    b_info = emd_data.get("BraeinAI", {}) if isinstance(emd_data.get("BraeinAI"), dict) else {}
                    params = emd_data.get("ModelParameters", {}) if isinstance(emd_data.get("ModelParameters"), dict) else {}
                    ckpt = {
                        "model_name": b_info.get("ModelName") or os.path.splitext(filename)[0],
                        "model_type": b_info.get("ModelTypeBraein", "mask_rcnn"),
                        "architecture": b_info.get("Architecture", emd_data.get("ModelConfiguration", "MaskRCNN")),
                        "backbone": b_info.get("Backbone", params.get("backbone", "resnet50")),
                        "mode": b_info.get("Mode", params.get("mode", "binary")),
                        "epoch": b_info.get("Epoch", 0),
                        "best_iou": float(b_info.get("BestIoU", 0.0) or 0.0),
                        "tile_size": int(emd_data.get("ImageHeight", 256)),
                        "class_names": _class_names_from_emd(emd_data),
                        "is_dlpk": True,
                        "extra_info": b_info,
                    }
                else:
                    if source == "dlpk":
                        continue
                    ckpt = {"model_name": os.path.splitext(filename)[0], "is_dlpk": True}
            else:
                try:
                    mtime = os.path.getmtime(fp)
                except OSError:
                    mtime = 0
                cache_key = (canonical_path, mtime)
                if cache_key in _CHECKPOINT_META_CACHE:
                    ckpt = _CHECKPOINT_META_CACHE[cache_key]
                else:
                    try:
                        ckpt = safe_torch_load(fp, map_location="cpu", context="list_models")
                    except Exception:
                        ckpt = {}
                    if not isinstance(ckpt, dict):
                        ckpt = {}
                    if ckpt and "model_state_dict" not in ckpt:
                        values = list(ckpt.values())[:8]
                        if values and all(hasattr(value, "shape") for value in values):
                            continue
                    _CHECKPOINT_META_CACHE[cache_key] = ckpt

            extra_info = ckpt.get("extra_info", {})
            if not isinstance(extra_info, dict):
                extra_info = {}

            class_meta = {
                "class_names": ckpt.get("class_names") or ckpt.get("classes"),
                "extra_info": extra_info,
            }
            if is_dlpk:
                emd_meta = _read_dlpk_metadata(fp) if fp.lower().endswith(".dlpk") else _read_adjacent_emd_metadata(fp)
                emd_classes = _class_names_from_emd(emd_meta) if emd_meta else None
                if emd_classes:
                    class_meta["class_names"] = emd_classes

            model_classes = _normalise_model_class_names(class_meta)
            try:
                modified_ts = os.path.getmtime(fp)
                size_mb = round(os.path.getsize(fp) / (1024 * 1024), 2)
                modified = datetime.fromtimestamp(modified_ts).strftime("%Y-%m-%d %H:%M:%S")
            except OSError:
                modified_ts, size_mb, modified = 0, 0.0, ""

            model_name = str(ckpt.get("model_name") or os.path.splitext(filename)[0])
            label_value = str(
                ckpt.get("label")
                or os.path.basename(os.path.dirname(os.path.dirname(fp)))
            )

            models.append({
                # Stable identity for clients that need a persistent option key.
                # Keep filepath as the actual value sent to /api/models/load.
                "model_id": uuid.uuid5(uuid.NAMESPACE_URL, canonical_path).hex,
                "label": label_value,
                "filename": filename,
                "filepath": fp,
                "epoch": ckpt.get("epoch", 0),
                "best_iou": float(ckpt.get("best_iou", 0.0) or 0.0),
                "backbone": ckpt.get("backbone", "unet"),
                "model_type": ckpt.get("model_type", "unet"),
                "mode": str(ckpt.get("mode", "binary")).lower(),
                "class_names": model_classes,
                "tile_size": int(ckpt.get("tile_size", ckpt.get("target_size", 256)) or 256),
                "architecture": ckpt.get("architecture", ""),
                "model_name": model_name,
                "source": "dlpk" if is_dlpk else source,
                "size_mb": size_mb,
                "modified": modified,
                "modified_timestamp": modified_ts,
                "extra_info": extra_info,
                "is_dlpk": is_dlpk,
                "is_finetuned": bool(ckpt.get("fine_tuned_from")),
            })

        # Stable alphabetical order. Modification time is metadata only and
        # must not cause the dropdown options to jump after saves/copies.
        models.sort(key=lambda item: (
            str(item.get("label", "")).casefold(),
            str(item.get("model_type", "")).casefold(),
            str(item.get("model_name", "")).casefold(),
            str(item.get("filename", "")).casefold(),
            os.path.normcase(os.path.realpath(item.get("filepath", ""))),
        ))

        return JSONResponse({"success": True, "models": models, "count": len(models)})
    except Exception as exc:
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "models": [],
            "count": 0,
            "error": str(exc),
        }, status_code=500)

@app.post("/api/models/export", dependencies=[Depends(require_api_key)])
async def api_models_export(data: Dict):
    """Export a .pth model to various formats."""
    try:
        src = str(data.get("model_path") or "").strip()
        fmt = str(data.get("format") or ".pth").lower()
        save_location = str(data.get("save_location") or depcfg.DLPK_EXPORT_DIR).strip()
        model_name = str(data.get("model_name") or os.path.splitext(os.path.basename(src))[0]).strip()
        
        src = _safe_path(src, depcfg.MODEL_ROOTS + depcfg.EXPORT_ROOTS, "model_path")
        save_location = _safe_path(save_location, depcfg.MODEL_ROOTS + depcfg.EXPORT_ROOTS, "save_location")
        os.makedirs(save_location, exist_ok=True)
        if not os.path.exists(src): return JSONResponse({"success": False, "error": "Source model not found"}, status_code=404)

        if fmt == ".dlpk":
            import dlpk as _dlpk
            meta = _dlpk.read_checkpoint_meta(src)
            out_path = os.path.join(save_location, model_name + ".dlpk")
            builder = _dlpk_torchscript_builder(src) if data.get("include_torchscript", True) else None
            result = _dlpk.export_dlpk(src, out_path, meta=meta, model_name=model_name, torchscript_builder=builder, include_torchscript=data.get("include_torchscript", True))
            return JSONResponse(result)

        elif fmt in [".pth", ".pt", ".ckpt"]:
            out_path = os.path.join(save_location, model_name + fmt)
            shutil.copy2(src, out_path)
            return JSONResponse({"success": True, "model_path": out_path})

        elif fmt == ".onnx":
            device = torch.device("cpu")
            model, meta = load_checkpoint(src, device)
            tile = int(meta.get("tile_size", 64))
            dummy_input = torch.randn(1, 3, tile, tile, device=device)
            out_path = os.path.join(save_location, model_name + ".onnx")
            torch.onnx.export(model, dummy_input, out_path, export_params=True, opset_version=12, do_constant_folding=True, input_names=['input'], output_names=['output'], dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
            return JSONResponse({"success": True, "model_path": out_path})

        elif fmt in [".h5", ".pb", ".tflite"]:
            try:
                import tensorflow as tf
                import onnx
                from onnx_tf.backend import prepare
            except ImportError:
                return JSONResponse({"success": False, "error": "tensorflow and onnx-tf are required for this format."}, status_code=400)

            device = torch.device("cpu")
            model, meta = load_checkpoint(src, device)
            tile = int(meta.get("tile_size", 64))
            dummy_input = torch.randn(1, 3, tile, tile, device=device)
            temp_onnx = os.path.join(save_location, model_name + "_temp.onnx")
            torch.onnx.export(model, dummy_input, temp_onnx, opset_version=12, input_names=['input'], output_names=['output'])
            onnx_model = onnx.load(temp_onnx); tf_rep = prepare(onnx_model)
            out_path = os.path.join(save_location, model_name + fmt)
            if fmt == ".h5": tf_rep.export_graph(out_path)
            elif fmt == ".pb": tf_rep.export_graph(out_path.replace(".pb", "")); out_path = out_path.replace(".pb", "")
            elif fmt == ".tflite":
                converter = tf.lite.TFLiteConverter.from_saved_model(tf_rep.export_graph(out_path.replace(".tflite", "")))
                tflite_model = converter.convert()
                with open(out_path, 'wb') as f: f.write(tflite_model)
            if os.path.exists(temp_onnx): os.remove(temp_onnx)
            return JSONResponse({"success": True, "model_path": out_path})

        else:
            return JSONResponse({"success": False, "error": f"Unsupported format: {fmt}"}, status_code=400)
    except Exception as e:
        traceback.print_exc(); return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.post("/api/models/load", dependencies=[Depends(require_api_key)])
async def api_models_load(data: Dict):
    try:
        model_path, label = data.get("model_path"), data.get("label")
        model_path = _safe_path(model_path, depcfg.MODEL_ROOTS + depcfg.EXPORT_ROOTS, "model_path")
        if not model_path or not os.path.exists(model_path): return JSONResponse({"success": False, "error": "Model file not found"}, status_code=404)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, meta = load_checkpoint(model_path, device)
        resolved_classes = _normalise_model_class_names(meta)
        # Commit only after model construction and strict checkpoint validation succeed.
        training_state.model_instance, training_state.model_path, training_state.model_name = model, model_path, meta.get("model_name", os.path.splitext(os.path.basename(model_path))[0])
        training_state.model_backbone, training_state.model_type, training_state.architecture = meta.get("backbone", "unet"), meta.get("model_type", "unet"), meta.get("architecture") or type(model).__name__
        training_state.mode, training_state.num_classes, training_state.class_names = meta.get("mode", "binary"), max(int(meta.get("num_classes", len(resolved_classes))), len(resolved_classes)), resolved_classes
        training_state.tile_size, training_state.best_map, training_state.training_completed, training_state.is_training = int(meta.get("tile_size", meta.get("target_size", 256))), float(meta.get("best_iou", 0.0)), True, False
        return JSONResponse({"success": True, "model_path": model_path, "model_name": training_state.model_name, "backbone": training_state.model_backbone, "model_type": training_state.model_type, "architecture": training_state.architecture, "mode": training_state.mode, "class_names": training_state.class_names, "num_classes": training_state.num_classes, "tile_size": training_state.tile_size, "best_iou": training_state.best_map, "base_channels": meta.get("base_channels"), "num_levels": meta.get("num_levels"), "model_config": meta.get("model_config", {}), "extra_info": meta.get("extra_info", {}), "epoch": meta.get("epoch"), "bound_label": label, "bound_project_id": training_state.current_project_id})
    except HTTPException: raise
    except Exception as e:
        traceback.print_exc(); return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.post("/api/models/save", dependencies=[Depends(require_api_key)])
async def api_models_save(data: Dict):
    try:
        src, name, save_location = str(data.get("model_path") or "").strip(), str(data.get("model_name") or "").strip(), str(data.get("save_location") or "./models/saved").strip()
        src = _safe_path(src, depcfg.MODEL_ROOTS + depcfg.EXPORT_ROOTS, "model_path")
        if not os.path.exists(src): return JSONResponse({"success": False, "error": f"Source model not found: {src}"}, status_code=404)
        name, filename = os.path.basename(name), f"{os.path.basename(name)}.pth" if not name.lower().endswith(".pth") else os.path.basename(name)
        save_location = save_location.replace("\\", os.sep).replace("/", os.sep)
        if not save_location or save_location.lower() == "custom": save_location = os.path.join(".", "models", "saved")
        if not os.path.isabs(save_location) and not (len(save_location) >= 2 and save_location[1] == ":"): save_location = os.path.normpath(save_location)
        save_location = _safe_path(save_location, depcfg.MODEL_ROOTS + depcfg.EXPORT_ROOTS, "save_location")
        os.makedirs(save_location, exist_ok=True)
        dst = os.path.join(save_location, filename); shutil.copy2(src, dst)
        return JSONResponse({"success": True, "saved_path": dst, "filename": filename, "source_model": src, "model_name": os.path.splitext(filename)[0], "save_location": save_location})
    except HTTPException: raise
    except Exception as e:
        traceback.print_exc(); return JSONResponse({"success": False, "error": str(e)}, status_code=500)

def _dlpk_resolve_file(name, suffixes=(".pth", ".ckpt", ".pt"), what="model"):
    raw = str(name or "").strip().strip("/").replace("\\", "/")
    if not raw: raise HTTPException(status_code=400, detail=what + " name is required")
    try:
        candidate = _safe_path(raw, depcfg.MODEL_ROOTS + depcfg.EXPORT_ROOTS, what)
        if os.path.isfile(candidate): return candidate
    except HTTPException: pass
    wanted, stem = os.path.basename(raw), os.path.splitext(os.path.basename(raw))[0]
    for root in list(depcfg.MODEL_ROOTS) + list(depcfg.EXPORT_ROOTS):
        if not os.path.isdir(root): continue
        for dirpath, _dirnames, filenames in os.walk(root):
            for filename in filenames:
                if filename.lower().endswith(tuple(suffixes)) and (filename == wanted or os.path.splitext(filename)[0] == stem): return os.path.join(dirpath, filename)
    raise HTTPException(status_code=404, detail="No %s named '%s' under the configured model roots" % (what, wanted))

def _dlpk_torchscript_builder(model_path):
    def _build(_meta):
        model, _meta_out = load_checkpoint(model_path, torch.device("cpu"))
        return model
    return _build

@app.post("/api/models/{name:path}/export_dlpk", dependencies=[Depends(require_api_key), Depends(require_capability("can_publish_model"))])
async def api_models_export_dlpk(name: str, data: Optional[Dict] = None):
    try:
        data = data or {}; src = _dlpk_resolve_file(name)
        if src.lower().endswith(".pt"): return JSONResponse({"success": False, "error": "Ultralytics .pt checkpoints are not exported to .dlpk."}, status_code=400)
        meta = dlpk.read_checkpoint_meta(src)
        if not meta.get("has_state_dict"): return JSONResponse({"success": False, "error": "That file has no model_state_dict."}, status_code=400)
        save_location = _safe_path(str(data.get("save_location") or depcfg.DLPK_EXPORT_DIR).strip(), depcfg.MODEL_ROOTS + depcfg.EXPORT_ROOTS, "save_location")
        os.makedirs(save_location, exist_ok=True)
        model_name = str(data.get("model_name") or "").strip() or meta.get("model_name") or os.path.splitext(os.path.basename(src))[0]
        out_path = os.path.join(save_location, dlpk._safe_stem(model_name) + ".dlpk")
        include_ts = bool(data.get("include_torchscript", depcfg.DLPK_INCLUDE_TORCHSCRIPT))
        builder = _dlpk_torchscript_builder(src) if include_ts else None
        bands = data.get("extract_bands")
        if isinstance(bands, list) and bands:
            try: bands = [int(b) for b in bands]
            except (TypeError, ValueError): bands = None
        else: bands = None
        result = dlpk.export_dlpk(src, out_path, meta=meta, model_name=dlpk._safe_stem(model_name), description=data.get("description"), extract_bands=bands, torchscript_builder=builder, include_torchscript=include_ts)
        return JSONResponse(result)
    except HTTPException: raise
    except dlpk.DlpkError as exc: return JSONResponse({"success": False, "error": str(exc)}, status_code=400)
    except Exception as exc:
        traceback.print_exc(); return JSONResponse({"success": False, "error": str(exc)}, status_code=500)

@app.post("/api/models/import_dlpk", dependencies=[Depends(require_api_key), Depends(require_capability("can_train"))])
async def api_models_import_dlpk(file: UploadFile = File(...)):
    target = None
    try:
        filename = os.path.basename(file.filename or "").strip()
        if not filename.lower().endswith(".dlpk"): return JSONResponse({"success": False, "error": "Upload an Esri Deep Learning Package (.dlpk)"}, status_code=400)
        incoming = os.path.join(depcfg.DLPK_IMPORT_DIR, "_incoming"); os.makedirs(incoming, exist_ok=True)
        stem = dlpk._safe_stem(os.path.splitext(filename)[0]); target = os.path.join(incoming, stem + ".dlpk")
        if os.path.exists(target): target = os.path.join(incoming, "%s_%s.dlpk" % (stem, datetime.now().strftime("%Y%m%d_%H%M%S")))
        limit = int(depcfg.DLPK_MAX_UPLOAD_BYTES); written = 0
        try:
            with open(target, "wb") as handle:
                while True:
                    chunk = await file.read(8 * 1024 * 1024)
                    if not chunk: break
                    written += len(chunk)
                    if written > limit: raise ValueError("over_limit")
                    handle.write(chunk)
        except ValueError:
            _quiet_remove(target); return JSONResponse({"success": False, "error": "Package exceeds the %d byte upload limit." % limit}, status_code=400)
        except Exception:
            _quiet_remove(target); raise
        if not written:
            _quiet_remove(target); return JSONResponse({"success": False, "error": "Empty upload"}, status_code=400)
        result = dlpk.import_dlpk(target, depcfg.DLPK_IMPORT_DIR, extract_inference_function=depcfg.DLPK_EXTRACT_INFERENCE_FUNCTION, max_extract_bytes=depcfg.DLPK_MAX_EXTRACT_BYTES, max_members=10000)
        return JSONResponse({"success": True, "dlpk_path": result["dlpk_path"], "extract_dir": result["extract_dir"], "model_path": result["model_path"], "loadable_checkpoint": result["loadable_checkpoint"], "loadable_reason": result["loadable_reason"], "skipped_python_files": result["skipped"], "notes": result["notes"], "info": result["info"]})
    except dlpk.DlpkError as exc:
        _quiet_remove(target); return JSONResponse({"success": False, "error": str(exc)}, status_code=400)
    except HTTPException: raise
    except Exception as exc:
        traceback.print_exc(); _quiet_remove(target); return JSONResponse({"success": False, "error": str(exc)}, status_code=500)

def _quiet_remove(path):
    try:
        if path and os.path.isfile(path): os.remove(path)
    except OSError: pass

@app.get("/api/models/download", dependencies=[Depends(require_api_key), Depends(require_capability("can_publish_model"))])
async def api_models_download(path: str):
    try:
        resolved = _safe_path(path, depcfg.MODEL_ROOTS + depcfg.EXPORT_ROOTS, "path")
        if not resolved or not os.path.isfile(resolved): return JSONResponse({"success": False, "error": "File not found"}, status_code=404)
        if not resolved.lower().endswith((".pth", ".pt", ".ckpt", ".dlpk")): return JSONResponse({"success": False, "error": "Only .pth, .pt, .ckpt and .dlpk may be downloaded."}, status_code=400)
        return FileResponse(resolved, media_type="application/octet-stream", filename=os.path.basename(resolved))
    except HTTPException: raise
    except Exception as exc:
        traceback.print_exc(); return JSONResponse({"success": False, "error": str(exc)}, status_code=500)

@app.delete("/api/models/{path:path}", dependencies=[Depends(require_api_key)])
async def api_models_delete(path: str):
    try:
        path = _safe_path(path, depcfg.MODEL_ROOTS + depcfg.EXPORT_ROOTS, "model path")
        if not path.lower().endswith((".pth", ".pt", ".ckpt")): return JSONResponse({"success": False, "error": "Only .pth, .pt or .ckpt model files can be deleted"}, status_code=400)
        if not os.path.exists(path): return JSONResponse({"success": False, "error": "Not found"}, status_code=404)
        os.remove(path)
        return JSONResponse({"success": True})
    except HTTPException: raise
    except Exception as e:
        traceback.print_exc(); return JSONResponse({"success": False, "error": str(e)}, status_code=500)

def _separate_building_contours(binary_mask):
    mask = (np.asarray(binary_mask) > 0).astype(np.uint8)
    if mask.ndim != 2 or not np.any(mask): return []
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8); output = []
    for component_id in range(1, count):
        x, y, w, h, area = [int(v) for v in stats[component_id]]
        if area <= 0: continue
        component = (labels[y:y+h, x:x+w] == component_id).astype(np.uint8)
        dist = cv2.distanceTransform(component, cv2.DIST_L2, 5)
        local_max = (dist == cv2.dilate(dist, np.ones((7, 7), np.uint8))) & (dist >= max(2.0, float(dist.max()) * 0.35))
        seed_count, seeds = cv2.connectedComponents(local_max.astype(np.uint8), 8)
        if seed_count <= 2 or area < 80:
            contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            output.extend([c + np.array([[[x, y]]], dtype=c.dtype) for c in contours]); continue
        markers = np.zeros(component.shape, dtype=np.int32); markers[component == 0] = 1
        for seed_id in range(1, seed_count): markers[seeds == seed_id] = seed_id + 1
        surface = 255 - cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.watershed(cv2.cvtColor(surface, cv2.COLOR_GRAY2BGR), markers)
        for marker_id in np.unique(markers):
            if marker_id <= 1: continue
            part = ((markers == marker_id) & (component > 0)).astype(np.uint8)
            contours, _ = cv2.findContours(part, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            output.extend([c + np.array([[[x, y]]], dtype=c.dtype) for c in contours])
    return output

def _feature_from_polygon_contour(contour, prob_map, bounds, W, H, safe_label, area_geo_factor=None, confidence=0.5, model_iou=0.0, min_area=20, simplify_tolerance=0.008):
    area_px = int(cv2.contourArea(contour))
    if area_px < min_area: return None
    eps = max(0.5, float(simplify_tolerance) * cv2.arcLength(contour, True))
    approx = cv2.approxPolyDP(contour, eps, True)
    if len(approx) < 3: return None
    west, east, south, north = float(bounds["west"]), float(bounds["east"]), float(bounds["south"]), float(bounds["north"])
    ring = []
    for pt in approx:
        arr = np.asarray(pt).reshape(-1)
        if arr.size < 2: continue
        px, py = float(arr[0]), float(arr[1])
        lon, lat = west + (px / max(W, 1)) * (east - west), north - (py / max(H, 1)) * (north - south)
        if -180 <= lon <= 180 and -90 <= lat <= 90: ring.append([lon, lat])
    if len(ring) < 3: return None
    if ring[0] != ring[-1]: ring.append(ring[0])
    return {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring]}, "properties": {"class_name": safe_label, "label": safe_label, "confidence": round(float(confidence), 4), "area": area_px, "area_px": area_px, "area_m2": (area_px * area_geo_factor if area_geo_factor is not None else None), "model_iou": float(model_iou)}}

def _normalise_model_class_names(meta):
    """Return an index-stable class list from checkpoint/EMD metadata."""
    candidates = meta.get("class_names")
    extra = meta.get("extra_info") or {}
    if not candidates and isinstance(extra, dict):
        for key in ("class_names", "ClassNames", "classes", "Classes", "class_mapping", "ClassMapping"):
            if extra.get(key):
                candidates = extra[key]
                break
    if isinstance(candidates, dict):
        try:
            candidates = [v for _, v in sorted(candidates.items(), key=lambda item: int(item[0]))]
        except Exception:
            candidates = list(candidates.values())
    if isinstance(candidates, (list, tuple)) and candidates and isinstance(candidates[0], dict):
        rows = []
        for item in candidates:
            name = item.get("Name") or item.get("name") or item.get("ClassName") or item.get("class_name")
            value = item.get("Value", item.get("value", item.get("ClassValue", item.get("class_id"))))
            if name:
                try: rows.append((int(value), str(name)))
                except Exception: rows.append((len(rows) + 1, str(name)))
        if rows:
            rows.sort(key=lambda x: x[0])
            candidates = ["background"] + [name for _, name in rows]
    if isinstance(candidates, str):
        candidates = [v.strip() for v in candidates.split(",") if v.strip()]
    names = [str(v).strip() for v in candidates] if isinstance(candidates, (list, tuple)) else []
    names = [v for v in names if v]
    if not names:
        names = ["background", "object"]
    if len(names) == 1:
        names.insert(0, "background")
    return names


def _prediction_class_name(class_id, class_names):
    """Torchvision detector labels are normally 1-based (0 is background)."""
    try:
        idx = int(class_id)
    except (TypeError, ValueError):
        return "object"
    has_background = bool(class_names) and str(class_names[0]).strip().lower() in ("background", "__background__")
    name_index = idx if has_background else idx - 1
    if 0 <= name_index < len(class_names):
        name = str(class_names[name_index]).strip()
        if name and name.lower() not in ("background", "__background__"):
            return name
    return f"class_{idx}"


@app.post("/predict", dependencies=[Depends(require_api_key)])
async def predict_compat(data: Dict):
    prediction_started_at = time.perf_counter()
    try:
        model_path, image_path = data.get("model_path") or training_state.model_path, data.get("image_path") or training_state.image_path
        bounds = data.get("bounds") or training_state.image_bounds_latlon; threshold = float(data.get("threshold", 0.5)); min_area = int(data.get("min_area", 20))
        smooth_sigma = float(np.clip(data.get("smooth_sigma", 0.6), 0.0, 1.5)); simplify_tolerance = float(np.clip(data.get("simplify_tolerance", 0.008), 0.002, 0.03)); morph_close = bool(data.get("morph_close", False))
        label = data.get("label") or training_state.current_label or "object"; safe_label = label.lower().replace(" ", "_"); aoi_bounds = data.get("aoi_bounds")
        
        if not model_path or not os.path.exists(model_path): return JSONResponse({"success": False, "error": "Model not loaded"}, status_code=400)
        if image_path: image_path = _safe_path(image_path, what="image_path")
        if not image_path or not os.path.exists(image_path): return JSONResponse({"success": False, "error": "Image not found"}, status_code=400)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model, meta = load_checkpoint(model_path, device)
        model_type, mode, tile_size, eval_tf = normalize_model_type(meta.get("model_type", "unet")), meta["mode"], int(meta["target_size"]), get_eval_transform(meta["mode"])
        model_class_names = _normalise_model_class_names(meta)
        is_detector_model = hasattr(model, "roi_heads") or model_type in DETECTION_MODELS or model_type in INSTANCE_MODELS
        model_foreground_name = next((n for n in model_class_names if n.lower() not in ("background", "__background__")), "object")
        prediction_label = model_foreground_name if (is_detector_model or len(model_class_names) > 2) else safe_label
        threshold, min_area = float(np.clip(threshold, 0.0, 1.0)), max(1, int(min_area))
        
        img_pil = Image.open(image_path).convert("RGB"); W, H = img_pil.size
        if not bounds or not all(k in bounds for k in ("west", "south", "east", "north")): bounds = {"west": 0.0, "south": float(H), "east": float(W), "north": 0.0}
        gsd = compute_gsd_m_per_px(bounds, W, H)
        is_detector = hasattr(model, "roi_heads") or model_type in DETECTION_MODELS or model_type in INSTANCE_MODELS or meta.get("architecture") in ("MaskRCNN", "FasterRCNN")
        
        # FIX 2: Initialize these variables so they don't throw NameError
        features, raw_count = [], 0
        foreground_pixels = 0
        foreground_ratio = 0.0
        tiles_processed = 0
        components_kept = 0
        components_removed = 0
        
        model.eval()
        
        if is_detector:
            try: detection_scale = float(data.get("detection_scale") or 1.0)
            except (TypeError, ValueError): detection_scale = 1.0
            detection_scale = max(0.25, min(4.0, detection_scale)); tf = getattr(model, "transform", None)
            if tf is not None and abs(detection_scale - 1.0) > 1e-6:
                base = int(meta.get("tile_size") or meta.get("target_size") or 400); scaled = int(round(base * detection_scale))
                tf.min_size, tf.max_size = (scaled,), scaled
            
            with torch.no_grad():
                img_t = eval_tf(img_pil).to(device); preds = model([img_t])
            p0 = preds[0] if preds else {}; scores, boxes, masks, pred_labels = p0.get("scores"), p0.get("boxes"), p0.get("masks"), p0.get("labels")
            
            score_np = scores.detach().cpu().numpy() if scores is not None else np.zeros((0,), dtype=np.float32); box_np = boxes.detach().cpu().numpy() if boxes is not None else np.zeros((0, 4), dtype=np.float32)
            label_np = pred_labels.detach().cpu().numpy().astype(int) if pred_labels is not None else np.ones((len(score_np),), dtype=np.int64)
            mask_np = None
            if masks is not None:
                mask_np = masks.detach().cpu().numpy(); mask_np = mask_np[:, 0] if mask_np.ndim == 4 else mask_np
            
            if model_type == "mask_rcnn" or meta.get("architecture") == "MaskRCNN":
                valid_indices = [i for i in range(len(score_np)) if float(score_np[i]) >= threshold and mask_np is not None and i < len(mask_np)]
                if valid_indices:
                    valid_indices.sort(key=lambda idx: float(score_np[idx]), reverse=True)
                    combined_mask = np.zeros((H, W), dtype=np.uint8)
                    conf_map = np.zeros((H, W), dtype=np.float32)

                    for i in valid_indices:
                        mk = mask_np[i]
                        if mk.shape != (H, W): mk = cv2.resize(mk.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
                        binary = (mk > 0.5).astype(np.uint8)
                        combined_mask = np.maximum(combined_mask, binary)
                        conf_map = np.maximum(conf_map, binary.astype(np.float32) * float(score_np[i]))

                    if morph_close:
                        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)
                    if smooth_sigma > 0:
                        blurred = cv2.GaussianBlur(combined_mask.astype(np.float32), (5, 5), sigmaX=smooth_sigma, sigmaY=smooth_sigma)
                        combined_mask = (blurred > 0.4).astype(np.uint8)

                    num_labels, comp_labels, comp_stats, _ = cv2.connectedComponentsWithStats(combined_mask, connectivity=8)
                    for comp_id in range(1, num_labels):
                        c_area = int(comp_stats[comp_id, cv2.CC_STAT_AREA])
                        if c_area < min_area: continue
                        comp_mask = (comp_labels == comp_id).astype(np.uint8)
                        contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if not contours: continue
                        max_cnt = max(contours, key=cv2.contourArea)
                        c_conf = float(np.max(conf_map[comp_labels == comp_id])) if np.any(comp_labels == comp_id) else float(threshold)

                        feat = _feature_from_polygon_contour(
                            max_cnt, comp_mask, bounds, W, H, prediction_label,
                            area_geo_factor=(gsd * gsd if gsd else None),
                            confidence=c_conf,
                            model_iou=training_state.best_map,
                            min_area=min_area,
                            simplify_tolerance=simplify_tolerance
                        )
                        if feat is not None:
                            feat["properties"]["instance_id"] = len(features) + 1
                            feat["properties"]["class_id"] = 1
                            feat["properties"]["class_name"] = prediction_label
                            feat["properties"]["label"] = prediction_label
                            try:
                                poly = ShapelyPolygon(feat["geometry"]["coordinates"][0])
                                b = poly.bounds
                                feat["bbox"] = [round(b[0], 7), round(b[1], 7), round(b[2], 7), round(b[3], 7)]
                            except Exception: pass
                            features.append(feat); raw_count += 1
            else:
                for i, score in enumerate(score_np):
                    score = float(score)
                    if score < threshold or i >= len(box_np): continue
                    class_id = int(label_np[i]) if i < len(label_np) else 1
                    class_name = _prediction_class_name(class_id, model_class_names)
                    if class_name.lower() in ("nodata", "no_data", "background", "__background__"):
                        class_name = prediction_label if prediction_label.lower() not in ("nodata", "no_data", "background") else "Building"
                    x1, y1, x2, y2 = box_np[i]
                    ring = [[float(bounds["west"]) + x1 / max(W, 1) * (float(bounds["east"]) - float(bounds["west"])), float(bounds["north"]) - y1 / max(H, 1) * (float(bounds["north"]) - float(bounds["south"]))], [float(bounds["west"]) + x2 / max(W, 1) * (float(bounds["east"]) - float(bounds["west"])), float(bounds["north"]) - y1 / max(H, 1) * (float(bounds["north"]) - float(bounds["south"]))], [float(bounds["west"]) + x2 / max(W, 1) * (float(bounds["east"]) - float(bounds["west"])), float(bounds["north"]) - y2 / max(H, 1) * (float(bounds["north"]) - float(bounds["south"]))], [float(bounds["west"]) + x1 / max(W, 1) * (float(bounds["east"]) - float(bounds["west"])), float(bounds["north"]) - y2 / max(H, 1) * (float(bounds["north"]) - float(bounds["south"]))]]; ring.append(ring[0])
                    poly = ShapelyPolygon(ring)
                    b = poly.bounds
                    features.append({"type": "Feature", "bbox": [round(b[0], 7), round(b[1], 7), round(b[2], 7), round(b[3], 7)], "geometry": {"type": "Polygon", "coordinates": [ring]}, "properties": {"class_name": class_name, "label": class_name, "class_id": class_id, "confidence": round(score, 4), "model_iou": float(meta.get("best_iou", training_state.best_map)), "instance_id": len(features) + 1, "pix_bbox": f"{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}"}}); raw_count += 1
        else:
            stride, prob_sum, prob_weight = max(1, int(tile_size * 0.8)), np.zeros((H, W), dtype=np.float32), np.zeros((H, W), dtype=np.float32)
            ys, xs = list(range(0, max(H - tile_size, 0) + 1, stride)) or [0], list(range(0, max(W - tile_size, 0) + 1, stride)) or [0]
            if ys[-1] != max(H - tile_size, 0): ys.append(max(H - tile_size, 0))
            if xs[-1] != max(W - tile_size, 0): xs.append(max(W - tile_size, 0))
            
            with torch.inference_mode():
                for y in ys:
                    for x in xs:
                        x2, y2 = min(x + tile_size, W), min(y + tile_size, H); tw, th = x2 - x, y2 - y
                        tile = img_pil.crop((x, y, x2, y2))
                        if tw != tile_size or th != tile_size:
                            arr = np.asarray(tile.convert("RGB")); pad = np.pad(arr, ((0, tile_size - th), (0, tile_size - tw), (0, 0)), mode="edge")
                            tile = Image.fromarray(pad.astype(np.uint8), mode="RGB")
                        tile_t = eval_tf(tile).unsqueeze(0).to(device); out = model(tile_t); out = out[0] if isinstance(out, (list, tuple)) else out
                        tile_prob = torch.sigmoid(out)[0, 0].float().cpu().numpy() if mode == "binary" else torch.softmax(out, dim=1)[0, 1].float().cpu().numpy()
                        tile_prob = np.asarray(tile_prob[:th, :tw], dtype=np.float32)
                        if tile_prob.shape != (th, tw): tile_prob = cv2.resize(tile_prob, (tw, th), interpolation=cv2.INTER_LINEAR)
                        prob_sum[y:y2, x:x2] += tile_prob; prob_weight[y:y2, x:x2] += 1.0; tiles_processed += 1
            
            prob_map = np.divide(prob_sum, np.maximum(prob_weight, 1e-6), out=np.zeros_like(prob_sum), where=prob_weight > 0)
            prob_for_mask = cv2.GaussianBlur(prob_map, (3, 3), sigmaX=smooth_sigma, sigmaY=smooth_sigma, borderType=cv2.BORDER_REPLICATE) if smooth_sigma > 0 else prob_map
            mask_bool = prob_for_mask >= threshold
            if morph_close: mask_bool = cv2.morphologyEx(mask_bool.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1).astype(bool)
            
            foreground_pixels = int(np.count_nonzero(mask_bool))
            foreground_ratio = float(foreground_pixels / max(H * W, 1))
            
            n_components, component_labels, component_stats, _ = cv2.connectedComponentsWithStats(mask_bool.astype(np.uint8), connectivity=8)
            for component_id in range(1, n_components):
                pixel_area = int(component_stats[component_id, cv2.CC_STAT_AREA])
                if pixel_area < min_area: components_removed += 1; continue
                component_mask = (component_labels == component_id).astype(np.uint8); contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours: continue
                feat = _feature_from_polygon_contour(max(contours, key=cv2.contourArea), prob_map, bounds, W, H, prediction_label, area_geo_factor=(gsd * gsd if gsd else None), confidence=float(np.mean(prob_map[component_labels == component_id])), model_iou=training_state.best_map, min_area=1, simplify_tolerance=simplify_tolerance)
                if feat is not None: feat["properties"]["pixel_count"] = pixel_area; feat["properties"]["confidence"] = round(float(np.mean(prob_map[component_labels == component_id])), 4); features.append(feat); components_kept += 1

        dropped = 0
        if aoi_bounds and isinstance(aoi_bounds, dict):
            try:
                aoi_geom = box(float(aoi_bounds["west"]), float(aoi_bounds["south"]), float(aoi_bounds["east"]), float(aoi_bounds["north"])); kept = []
                for feat in features:
                    try:
                        geom = feat["geometry"]
                        if geom.get("type") != "Polygon": kept.append(feat); continue
                        poly = ShapelyPolygon(geom["coordinates"][0])
                        if not poly.is_valid: poly = poly.buffer(0)
                        if not poly.intersects(aoi_geom): dropped += 1; continue
                        inter = poly.intersection(aoi_geom)
                        if inter.is_empty: dropped += 1; continue
                        feat["geometry"] = mapping(inter); kept.append(feat)
                    except Exception: dropped += 1
                features = kept
            except Exception: pass
            
        # Clean RFC 7946 GeoJSON: no top-level crs property (standard requires EPSG:4326 without crs tag)
        geojson = {"type": "FeatureCollection", "features": features}
        out_dir = os.path.join("labels", safe_label, "evaluation", "geojson"); os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"predict_{datetime.now().strftime('%Y%m%d_%H%M%S')}.geojson")
        with open(out_path, "w", encoding="utf-8") as f: json.dump(geojson, f, indent=2)

        # Automatic ESRI Shapefile export
        shp_url = None
        if features:
            try:
                shp_out_dir = os.path.join("labels", safe_label, "evaluation", "shp")
                os.makedirs(shp_out_dir, exist_ok=True)
                stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                shp_base = f"predict_{stamp}"
                shp_file = os.path.join(shp_out_dir, f"{shp_base}.shp")
                shp_zip = os.path.join(shp_out_dir, f"{shp_base}.zip")
                geoms, records = [], []
                for feat in features:
                    g = feat.get("geometry")
                    if g and g.get("type") == "Polygon" and g.get("coordinates"):
                        poly = ShapelyPolygon(g["coordinates"][0])
                        if not poly.is_valid: poly = poly.buffer(0)
                        if not poly.is_empty:
                            geoms.append(poly)
                            p = dict(feat.get("properties", {}))
                            p.pop("bbox", None)
                            records.append(p)
                if geoms:
                    gdf = gpd.GeoDataFrame(records, geometry=geoms, crs="EPSG:4326")
                    gdf.to_file(shp_file)
                    with zipfile.ZipFile(shp_zip, "w", zipfile.ZIP_DEFLATED) as zf:
                        for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
                            fn = os.path.join(shp_out_dir, f"{shp_base}{ext}")
                            if os.path.exists(fn): zf.write(fn, os.path.basename(fn))
                    shp_url = f"/labels/{safe_label}/evaluation/shp/{shp_base}.zip"
            except Exception as shp_err:
                print(f"[predict] Shapefile export warning: {shp_err}")
        
        return JSONResponse({"success": True, "geojson": geojson, "num_objects": len(features), "duration_seconds": round(time.perf_counter() - prediction_started_at, 2), "shp_url": shp_url, "model": {"checkpoint": os.path.basename(os.path.abspath(model_path)), "epoch": meta.get("epoch", "unknown"), "architecture": meta.get("architecture", model_type), "model_type": model_type, "mode": mode, "tile_size": tile_size, "threshold": threshold, "min_area": min_area, "smooth_sigma": smooth_sigma, "simplify_tolerance": simplify_tolerance, "morph_close": morph_close}, "diagnostics": {"native_detector": is_detector, "semantic_tiles_processed": tiles_processed if not is_detector else None, "semantic_foreground_pixels": foreground_pixels if not is_detector else None, "semantic_foreground_ratio": foreground_ratio if not is_detector else None, "semantic_components_kept": components_kept if not is_detector else None, "semantic_components_removed": components_removed if not is_detector else None, "postprocessing": ({"smooth_sigma": smooth_sigma, "simplify_tolerance": simplify_tolerance, "morph_close": morph_close, "min_area": min_area} if not is_detector else None), "raw_detector_instances": raw_count if is_detector else None, "aoi_clip": ({"kept": len(features), "dropped": dropped} if dropped or aoi_bounds else None)}})
    except HTTPException: raise
    except Exception as e:
        traceback.print_exc(); return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.post("/api/predict_folder", dependencies=[Depends(require_api_key)])
async def api_predict_folder(data: Dict):
    try:
        folder, model_path, threshold, min_area, label = data.get("folder"), data.get("model_path") or training_state.model_path, float(data.get("threshold", 0.5)), int(data.get("min_area", 20)), (data.get("label") or training_state.current_label or "object")
        safe_label = label.lower().replace(" ", "_")
        folder = _safe_path(folder, what="folder")
        if not folder or not os.path.isdir(folder): return JSONResponse({"success": False, "error": f"Folder not found: {folder}"}, status_code=400)
        if not model_path or not os.path.exists(model_path): return JSONResponse({"success": False, "error": "Model not loaded"}, status_code=400)
        files = [f for f in sorted(os.listdir(folder)) if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))]
        if not files: return JSONResponse({"success": False, "error": "No images in folder"}, status_code=400)
        total_objects, per_file = 0, []
        for fname in files:
            img_path = os.path.join(folder, fname)
            resp = await predict_compat({"model_path": model_path, "image_path": img_path, "threshold": threshold, "min_area": min_area, "label": safe_label, "detection_scale": data.get("detection_scale"), "bounds": {"west": -180, "south": -90, "east": 180, "north": 90}})
            try:
                j = json.loads(resp.body.decode())
                if j.get("success"): n = len(j["geojson"]["features"]); total_objects += n; per_file.append({"file": fname, "objects": n})
            except Exception: pass
        return JSONResponse({"success": True, "n_files": len(files), "total_objects": total_objects, "per_file": per_file, "label": safe_label})
    except HTTPException: raise
    except Exception as e:
        traceback.print_exc(); return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.post("/api/evaluate_geojson", dependencies=[Depends(require_api_key)])
async def api_evaluate_geojson(data: Dict):
    VALID_SPLITS = ("train", "val", "test")
    def _safe_feature_from_mask(mask, bounds, label_name, chip_name, layer_name, min_area_px, confidence=1.0, instance_id=None):
        mask = np.asarray(mask)
        if mask.ndim != 2: return []
        binary = (mask > 0).astype(np.uint8)
        if int(np.count_nonzero(binary)) < int(min_area_px): return []
        contours, _ = cv2.findContours((binary * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        w, e, s, n = float(bounds["west"]), float(bounds["east"]), float(bounds["south"]), float(bounds["north"]); h, ww = binary.shape; gsd = compute_gsd_m_per_px(bounds, ww, h); features = []
        for contour in contours:
            if len(contour) < 3: continue
            area_px = float(cv2.contourArea(contour))
            if area_px < float(min_area_px): continue
            approx = cv2.approxPolyDP(contour, 0.005 * float(cv2.arcLength(contour, True)), True)
            if len(approx) < 3: continue
            ring = [[w + (float(point[0]) / max(float(ww), 1.0)) * (e - w), n - (float(point[1]) / max(float(h), 1.0)) * (n - s)] for point in approx.reshape(-1, 2)]
            if len(ring) < 3: continue
            if ring[0] != ring[-1]: ring.append(ring[0])
            try:
                geom = ShapelyPolygon(ring)
                if not geom.is_valid: geom = geom.buffer(0)
                if geom.is_empty: continue
                aoi_feature = data.get("aoi_bounds")
                if aoi_feature:
                    try:
                        aoi_geom = box(float(aoi_feature["west"]), float(aoi_feature["south"]), float(aoi_feature["east"]), float(aoi_feature["north"]))
                        geom = geom.intersection(aoi_geom)
                        if geom.is_empty: continue
                    except Exception: pass
                props = {"class_name": label_name, "layer": layer_name, "chip": chip_name, "confidence": round(float(confidence), 4), "area_px": round(area_px, 2), "area_m2": (round(area_px * gsd * gsd, 3) if gsd is not None else None)}
                if instance_id is not None: props["instance_id"] = int(instance_id)
                features.append({"type": "Feature", "geometry": mapping(geom), "properties": props})
            except Exception: pass
        return features
    def _safe_feature_from_box(box_xyxy, bounds, label_name, chip_name, confidence, instance_id=None):
        try: x1, y1, x2, y2 = [float(v) for v in box_xyxy]
        except Exception: return None
        w, e, s, n, ww, hh = float(bounds["west"]), float(bounds["east"]), float(bounds["south"]), float(bounds["north"]), float(bounds.get("px") or 1), float(bounds.get("py") or 1)
        ring = [[w + (x1 / max(ww, 1.0)) * (e - w), n - (y1 / max(hh, 1.0)) * (n - s)], [w + (x2 / max(ww, 1.0)) * (e - w), n - (y1 / max(hh, 1.0)) * (n - s)], [w + (x2 / max(ww, 1.0)) * (e - w), n - (y2 / max(hh, 1.0)) * (n - s)], [w + (x1 / max(ww, 1.0)) * (e - w), n - (y2 / max(hh, 1.0)) * (n - s)]]; ring.append(ring[0])
        try:
            geom = ShapelyPolygon(ring)
            if not geom.is_valid: geom = geom.buffer(0)
            if geom.is_empty: return None
            aoi_feature = data.get("aoi_bounds")
            if aoi_feature:
                try:
                    aoi_geom = box(float(aoi_feature["west"]), float(aoi_feature["south"]), float(aoi_feature["east"]), float(aoi_feature["north"]))
                    geom = geom.intersection(aoi_geom)
                    if geom.is_empty: return None
                except Exception: pass
            props = {"class_name": label_name, "layer": "prediction", "chip": chip_name, "confidence": round(float(confidence), 4)}
            if instance_id is not None: props["instance_id"] = int(instance_id)
            return {"type": "Feature", "geometry": mapping(geom), "properties": props}
        except Exception: return None
    def _binary_components(mask, min_area_px):
        arr = (np.asarray(mask) > 0).astype(np.uint8)
        if arr.ndim != 2: return []
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(arr, connectivity=8)
        return [labels == idx for idx in range(1, int(num_labels)) if int(stats[idx, cv2.CC_STAT_AREA]) >= int(min_area_px)]
    def _match_binary_components(pred_components, gt_components, iou_threshold=0.5):
        pairs = []
        for pi, pm in enumerate(pred_components):
            for gi, gm in enumerate(gt_components):
                inter = int(np.logical_and(pm, gm).sum())
                if inter <= 0: continue
                union = int(np.logical_or(pm, gm).sum())
                if union <= 0: continue
                iou = float(inter / union)
                if iou >= float(iou_threshold): pairs.append((iou, pi, gi))
        pairs.sort(reverse=True, key=lambda x: x[0]); matched_pred, matched_gt, matched_ious = set(), set(), []
        for iou, pi, gi in pairs:
            if pi in matched_pred or gi in matched_gt: continue
            matched_pred.add(pi); matched_gt.add(gi); matched_ious.append(float(iou))
        tp = len(matched_ious); fp = max(len(pred_components) - tp, 0); fn = max(len(gt_components) - tp, 0)
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0; recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "precision": float(precision), "recall": float(recall), "f1": float(2.0 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0, "mean_iou": float(np.mean(matched_ious)) if matched_ious else 0.0}
    try:
        project_id, model_path = data.get("project_id") or training_state.current_project_id, data.get("model_path") or training_state.model_path
        label = data.get("label") or training_state.current_label or training_state.selected_label; threshold, min_area = float(data.get("threshold", 0.5)), int(data.get("min_area", 8))
        requested_split = str(data.get("eval_split") or data.get("split") or "test").strip().lower()
        if requested_split not in VALID_SPLITS: return JSONResponse({"success": False, "error": f"Invalid evaluation split '{requested_split}'."}, status_code=400)
        if not project_id or not model_path or not os.path.exists(model_path) or not label: return JSONResponse({"success": False, "error": "Project, Label, and Model required"}, status_code=400)
        safe, lp = label.lower().replace(" ", "_"), get_label_path(label.lower().replace(" ", "_")); dataset_dir = os.path.join(lp, "dataset"); split_name = requested_split
        img_dir, msk_dir = os.path.join(dataset_dir, split_name, "images"), os.path.join(dataset_dir, split_name, "masks")
        if not os.path.isdir(img_dir): return JSONResponse({"success": False, "error": f"Evaluation split '{split_name}' is missing."}, status_code=400)
        chip_bounds_all = compute_all_chip_bounds(lp)
        if not chip_bounds_all: return JSONResponse({"success": False, "error": "chip_bounds.json is missing or empty. Rebuild the dataset before evaluation."}, status_code=400)
        img_files = [f for f in sorted(os.listdir(img_dir)) if f.lower().endswith((".png", ".jpg", ".jpeg")) and not f.endswith("_vis.png") and not f.endswith("_ov.png") and not f.endswith("_ids.png")]
        if not img_files: return JSONResponse({"success": False, "error": f"No image chips found in {img_dir}"}, status_code=400)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model, meta = load_checkpoint(model_path, device)
        mode, tile_size, model_type, eval_tf = meta.get("mode", "binary"), int(meta.get("target_size") or meta.get("tile_size") or 64), normalize_model_type(meta.get("model_type", "unet")), get_eval_transform(meta.get("mode", "binary"))
        gt_features, pred_features, chips_skipped_no_bounds, processed_chip_names = [], [], [], []
        inst_tp = inst_fp = inst_fn = 0; inst_ious, iou_list, f1_list, prec_list, rec_list = [], [], [], [], []
        tp_total = fp_total = fn_total = tn_total = 0; positive_chip_iou, positive_chip_precision, positive_chip_recall, positive_chip_f1 = [], [], [], []
        object_tp_total = object_fp_total = object_fn_total = 0; object_ious, object_macro_iou, object_macro_precision, object_macro_recall, object_macro_f1 = [], [], [], [], []
        semantic_gt_object_count = semantic_pred_object_count = semantic_positive_gt_chips = semantic_positive_pred_chips = 0
        is_native_detector = (model_type in DETECTION_MODELS or model_type in INSTANCE_MODELS or hasattr(model, "roi_heads") or meta.get("architecture") in ("MaskRCNN", "FasterRCNN"))
        is_instance_model = (model_type == "mask_rcnn" or meta.get("architecture") == "MaskRCNN")
        model.eval()
        with torch.no_grad():
            for fname in img_files:
                ip, mp, cb = os.path.join(img_dir, fname), os.path.join(msk_dir, fname), chip_bounds_all.get(fname)
                if not cb: chips_skipped_no_bounds.append(fname); continue
                processed_chip_names.append(fname); img_pil = Image.open(ip).convert("RGB").resize((tile_size, tile_size), Image.BILINEAR); img_t = eval_tf(img_pil).unsqueeze(0).to(device)
                if is_native_detector:
                    preds = model([img_t[0]]); p0 = preds[0] if preds else {}; gt_instance_masks, gt_instance_boxes = [], []
                    tgt_dir, inst_mask_root = os.path.join(dataset_dir, split_name, "instance_targets"), os.path.join(dataset_dir, split_name, "instance_masks")
                    stem = os.path.splitext(fname)[0]; tgt_path = os.path.join(tgt_dir, f"{stem}.json")
                    if os.path.exists(tgt_path):
                        try:
                            with open(tgt_path, "r", encoding="utf-8") as f: target_json = json.load(f)
                            for inst_idx, inst in enumerate(target_json.get("instances", []) or [], start=1):
                                mask_file = inst.get("mask_file")
                                if not mask_file: continue
                                mask_rel = str(mask_file).replace("\\", "/")
                                mask_path = os.path.join(dataset_dir, split_name, mask_rel) if mask_rel.startswith("instance_masks/") else os.path.join(inst_mask_root, stem, os.path.basename(mask_rel))
                                if not os.path.exists(mask_path): continue
                                gm = np.asarray(Image.open(mask_path).convert("L").resize((tile_size, tile_size), Image.NEAREST)) > 127
                                if int(np.count_nonzero(gm)) < min_area: continue
                                gt_instance_masks.append(gm.astype(np.uint8))
                                b = inst.get("bbox")
                                if b and len(b) == 4: gt_instance_boxes.append(np.asarray(b, dtype=np.float32))
                                else:
                                    ys, xs = np.where(gm)
                                    if len(xs): gt_instance_boxes.append(np.asarray([float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)], dtype=np.float32))
                                gt_features.extend(_safe_feature_from_mask(gm, cb, safe, fname, "ground_truth", min_area, confidence=1.0, instance_id=inst_idx))
                        except Exception: pass
                    pred_masks, pred_boxes = [], []
                    scores, boxes_pred, masks_pred = p0.get("scores"), p0.get("boxes"), p0.get("masks")
                    scores_np = scores.detach().cpu().numpy().astype(np.float32) if scores is not None else np.zeros((0,), dtype=np.float32)
                    boxes_np = boxes_pred.detach().cpu().numpy().astype(np.float32) if boxes_pred is not None else np.zeros((0, 4), dtype=np.float32)
                    masks_np = masks_pred.detach().cpu().numpy() if masks_pred is not None else None
                    if masks_np is not None and masks_np.ndim == 4: masks_np = masks_np[:, 0]
                    if is_instance_model and masks_np is not None:
                        for i, score_val in enumerate(scores_np):
                            score_val = float(score_val)
                            if score_val < threshold or i >= len(masks_np): continue
                            pm = masks_np[i]
                            if pm.shape != (tile_size, tile_size): pm = cv2.resize(pm.astype(np.float32), (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)
                            pm_binary = (pm > 0.5).astype(np.uint8)
                            if int(np.count_nonzero(pm_binary)) < min_area: continue
                            pred_masks.append(pm_binary)
                            if i < len(boxes_np): pred_boxes.append(boxes_np[i])
                            pred_features.extend(_safe_feature_from_mask(pm_binary, cb, safe, fname, "prediction", min_area, confidence=score_val, instance_id=len(pred_masks)))
                        match = _match_instance_masks(np.asarray(pred_masks, dtype=np.uint8), np.asarray(gt_instance_masks, dtype=np.uint8), iou_threshold=0.5)
                    else:
                        for i, score_val in enumerate(scores_np):
                            score_val = float(score_val)
                            if score_val < threshold or i >= len(boxes_np): continue
                            pred_boxes.append(boxes_np[i]); feat = _safe_feature_from_box(boxes_np[i], {**cb, "px": tile_size, "py": tile_size}, safe, fname, score_val, instance_id=len(pred_boxes))
                            if feat is not None: pred_features.append(feat)
                        match = _match_boxes(np.asarray(pred_boxes, dtype=np.float32), np.asarray(gt_instance_boxes, dtype=np.float32), iou_threshold=0.5)
                    inst_tp += int(match["tp"]); inst_fp += int(match["fp"]); inst_fn += int(match["fn"])
                    if match.get("iou", 0.0) > 0: inst_ious.append(float(match["iou"]))
                    iou_list.append(float(match["iou"])); prec_list.append(float(match["precision"])); rec_list.append(float(match["recall"])); f1_list.append(float(match["f1"]))
                    continue
                out = model(img_t); out = out[0] if isinstance(out, (list, tuple)) else out
                prob = torch.sigmoid(out)[0, 0].cpu().numpy() if mode == "binary" else torch.softmax(out, dim=1)[0, 1].cpu().numpy()
                pred_mask = (prob > threshold).astype(np.uint8)
                if os.path.exists(mp): gt = np.asarray(Image.open(mp).convert("L").resize((tile_size, tile_size), Image.NEAREST)); gt_mask = (gt > 127).astype(np.uint8)
                else: gt_mask = np.zeros_like(pred_mask, dtype=np.uint8)
                tp = int(np.logical_and(pred_mask == 1, gt_mask == 1).sum()); fp = int(np.logical_and(pred_mask == 1, gt_mask == 0).sum()); fn = int(np.logical_and(pred_mask == 0, gt_mask == 1).sum()); tn = int(np.logical_and(pred_mask == 0, gt_mask == 0).sum())
                tp_total += tp; fp_total += fp; fn_total += fn; tn_total += tn
                chip_iou = tp / (tp + fp + fn) if tp + fp + fn > 0 else 0.0; chip_precision = tp / (tp + fp) if tp + fp > 0 else 0.0; chip_recall = tp / (tp + fn) if tp + fn > 0 else 0.0
                chip_f1 = 2.0 * chip_precision * chip_recall / (chip_precision + chip_recall) if chip_precision + chip_recall > 0 else 0.0
                has_gt, has_pred = bool(np.any(gt_mask == 1)), bool(np.any(pred_mask == 1))
                if has_gt or has_pred: positive_chip_iou.append(float(chip_iou)); positive_chip_precision.append(float(chip_precision)); positive_chip_recall.append(float(chip_recall)); positive_chip_f1.append(float(chip_f1))
                if has_gt: semantic_positive_gt_chips += 1
                if has_pred: semantic_positive_pred_chips += 1
                gt_components, pred_components = _binary_components(gt_mask, min_area), _binary_components(pred_mask, min_area)
                semantic_gt_object_count += len(gt_components); semantic_pred_object_count += len(pred_components)
                object_match = _match_binary_components(pred_components, gt_components, iou_threshold=0.5)
                object_tp_total += object_match["tp"]; object_fp_total += object_match["fp"]; object_fn_total += object_match["fn"]
                if object_match["mean_iou"] > 0: object_ious.append(object_match["mean_iou"])
                if gt_components or pred_components: object_macro_iou.append(object_match["mean_iou"]); object_macro_precision.append(object_match["precision"]); object_macro_recall.append(object_match["recall"]); object_macro_f1.append(object_match["f1"])
                gt_features.extend(_safe_feature_from_mask(gt_mask, cb, safe, fname, "ground_truth", min_area, confidence=1.0))
                mean_conf = (float(np.mean(prob[pred_mask == 1])) if has_pred else 0.0)
                pred_features.extend(_safe_feature_from_mask(pred_mask, cb, safe, fname, "prediction", min_area, confidence=mean_conf))
        if is_native_detector:
            metric_iou = float(np.mean(inst_ious)) if inst_ious else 0.0; metric_precision = inst_tp / (inst_tp + inst_fp) if inst_tp + inst_fp > 0 else 0.0; metric_recall = inst_tp / (inst_tp + inst_fn) if inst_tp + inst_fn > 0 else 0.0
            metric_f1 = 2.0 * metric_precision * metric_recall / (metric_precision + metric_recall) if metric_precision + metric_recall > 0 else 0.0
            macro_iou = float(np.mean(iou_list)) if iou_list else 0.0; macro_precision = float(np.mean(prec_list)) if prec_list else 0.0; macro_recall = float(np.mean(rec_list)) if rec_list else 0.0; macro_f1 = float(np.mean(f1_list)) if f1_list else 0.0
            metrics = {"mean_iou": metric_iou, "mean_precision": metric_precision, "mean_recall": metric_recall, "mean_f1": metric_f1, "macro_iou": macro_iou, "macro_precision": macro_precision, "macro_recall": macro_recall, "macro_f1": macro_f1, "tp": int(inst_tp), "fp": int(inst_fp), "fn": int(inst_fn), "tn": 0, "num_samples": len(processed_chip_names), "threshold": threshold, "metric_type": ("instance_mask_iou" if is_instance_model else "box_iou"), "evaluation_split": split_name}
        else:
            micro_iou = tp_total / (tp_total + fp_total + fn_total) if tp_total + fp_total + fn_total > 0 else 0.0; micro_precision = tp_total / (tp_total + fp_total) if tp_total + fp_total > 0 else 0.0; micro_recall = tp_total / (tp_total + fn_total) if tp_total + fn_total > 0 else 0.0
            micro_f1 = 2.0 * micro_precision * micro_recall / (micro_precision + micro_recall) if micro_precision + micro_recall > 0 else 0.0
            mean_positive = lambda vals: (float(np.mean(vals)) if vals else 0.0)
            object_iou = float(np.mean(object_ious)) if object_ious else 0.0; object_precision = object_tp_total / (object_tp_total + object_fp_total) if object_tp_total + object_fp_total > 0 else 0.0; object_recall = object_tp_total / (object_tp_total + object_fn_total) if object_tp_total + object_fn_total > 0 else 0.0
            object_f1 = 2.0 * object_precision * object_recall / (object_precision + object_recall) if object_precision + object_recall > 0 else 0.0
            metrics = {"mean_iou": float(micro_iou), "mean_precision": float(micro_precision), "mean_recall": float(micro_recall), "mean_f1": float(micro_f1), "macro_positive_iou": mean_positive(positive_chip_iou), "macro_positive_precision": mean_positive(positive_chip_precision), "macro_positive_recall": mean_positive(positive_chip_recall), "macro_positive_f1": mean_positive(positive_chip_f1), "object_iou": float(object_iou), "object_precision": float(object_precision), "object_recall": float(object_recall), "object_f1": float(object_f1), "object_tp": int(object_tp_total), "object_fp": int(object_fp_total), "object_fn": int(object_fn_total), "object_macro_iou": mean_positive(object_macro_iou), "object_macro_precision": mean_positive(object_macro_precision), "object_macro_recall": mean_positive(object_macro_recall), "object_macro_f1": mean_positive(object_macro_f1), "tp": int(tp_total), "fp": int(fp_total), "fn": int(fn_total), "tn": int(tn_total), "num_samples": len(processed_chip_names), "threshold": threshold, "metric_type": "micro_pixel_iou", "evaluation_split": split_name, "positive_gt_chips": int(semantic_positive_gt_chips), "positive_pred_chips": int(semantic_positive_pred_chips), "gt_objects": int(semantic_gt_object_count), "pred_objects": int(semantic_pred_object_count)}
        training_state.evaluation_metrics = metrics
        eval_dir = os.path.join(lp, "evaluation"); os.makedirs(eval_dir, exist_ok=True)
        with open(os.path.join(eval_dir, "metrics.json"), "w", encoding="utf-8") as f: json.dump({"metrics": metrics, "model_type": model_type, "architecture": meta.get("architecture"), "split": split_name, "requested_split": requested_split, "num_requested_chips": len(img_files), "num_processed_chips": len(processed_chip_names), "num_gt_polygons": len(gt_features), "num_pred_polygons": len(pred_features), "chips_skipped_no_bounds": chips_skipped_no_bounds, "processed_chip_names": processed_chip_names}, f, indent=2)
        return JSONResponse({"success": True, "metrics": metrics, "gt_geojson": {"type": "FeatureCollection", "features": gt_features}, "pred_geojson": {"type": "FeatureCollection", "features": pred_features}, "num_gt_polygons": len(gt_features), "num_pred_polygons": len(pred_features), "chips_skipped_no_bounds": chips_skipped_no_bounds, "processed_chip_names": processed_chip_names, "requested_chip_count": len(img_files), "processed_chip_count": len(processed_chip_names), "val_chip_bounds_list": [chip_bounds_all.get(name) for name in processed_chip_names if chip_bounds_all.get(name)], "model_type": model_type, "architecture": meta.get("architecture"), "split": split_name})
    except Exception as e:
        traceback.print_exc(); return JSONResponse({"success": False, "error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host=depcfg.HOST, port=depcfg.PORT, log_level="info")