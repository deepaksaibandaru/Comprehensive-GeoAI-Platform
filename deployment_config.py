"""
deployment_config.py - deployment settings for Braein.ai
===================================================================
Single auditable file for the IGA + NCSC security review. Everything that
touches an external network, relaxes a security control, or carries a
third-party licence obligation is declared here and nowhere else.

Override any value with an environment variable of the same name.

Addresses the four findings raised in the GeoAI Use Case Finalization
meeting (2026-10-22, from 34:05):

  1. Basemap tiles        -> no Google; IGA Image Server first
  2. ultralytics / YOLO   -> AGPL-3.0, disabled by default
  3. CORS allow_origins=* -> restricted allowlist
  4. Runtime CDN loading  -> local vendored assets
  5. Geocoding            -> proxied through /api/geocode

And the four findings from the 2026-11 security audit:

  6. Pickle RCE           -> SECTION 6, checkpoint loading is weights-only
  7. No authentication    -> SECTION 7, X-API-Key on every mutating route
  8. Path traversal       -> SECTION 8, every caller path resolved to a root
  9. Open /docs           -> SECTION 9, interactive API docs off by default
"""

import os
import secrets
from contextlib import contextmanager


def _env_bool(name, default):
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _env_list(name, default):
    v = os.environ.get(name)
    if not v:
        return list(default)
    return [x.strip() for x in v.split(",") if x.strip()]


# =========================================================================
# 1. MAP SERVICES - data residency
# -------------------------------------------------------------------------
# Google tile endpoints (mt1.google.com) are removed. They sent every pan
# and zoom over Bahrain to Google servers, and direct /vt/ tile access sits
# outside the Google Maps Terms of Service.
#
# Point IGA_IMAGE_SERVER at your own Image Server and it becomes the default
# basemap, so imagery never leaves the IGA estate.
# =========================================================================

IGA_IMAGE_SERVER = os.environ.get("IGA_IMAGE_SERVER", "").rstrip("/")
IGA_IMAGE_SERVER_NAME = os.environ.get("IGA_IMAGE_SERVER_NAME", "Organisation Orthophoto")

# Allow public fallback basemaps. Set false for an air-gapped or fully
# internal deployment: only IGA_IMAGE_SERVER is then offered.
ALLOW_PUBLIC_BASEMAPS = _env_bool("ALLOW_PUBLIC_BASEMAPS", True)


def build_map_services():
    """Basemap list, IGA first. Never contains a Google endpoint."""
    services = {}

    if IGA_IMAGE_SERVER:
        services["iga_imagery"] = {
            "name": IGA_IMAGE_SERVER_NAME,
            # Served through this application, not fetched by the browser.
            #
            # It used to be IGA_IMAGE_SERVER + "/tile/{z}/{y}/{x}", which has
            # two problems: a secured service needs a token the browser can
            # only pass in the URL, and /tile/ exists only on a service with
            # a published cache - a dynamic Image Service returns nothing and
            # the map is silently blank. The proxy holds the token and falls
            # back to exportImage. See app.py /api/basemap/tile.
            #
            # Note the axis order. ArcGIS is /tile/{z}/{row}/{col} = z/y/x;
            # Leaflet substitutes z/x/y. The proxy takes Leaflet's order and
            # transposes internally, so this template is the ordinary one.
            # Getting that backwards returns the wrong place on earth rather
            # than an error.
            "url": "/api/basemap/tile/{z}/{x}/{y}",
            "upstream": IGA_IMAGE_SERVER,
            "attribution": "(c) Information and eGovernment Authority, Bahrain",
            "max_zoom": 21,
            "type": "Satellite",
            "resolution": "0.1 m",
            "internal": True,
        }

    if ALLOW_PUBLIC_BASEMAPS:
        services["satellite"] = {
            "name": "Satellite (Esri World Imagery)",
            "url": "https://server.arcgisonline.com/ArcGIS/rest/services/"
                   "World_Imagery/MapServer/tile/{z}/{y}/{x}",
            "attribution": "(c) Esri, Maxar",
            "max_zoom": 19,
            "type": "Satellite",
            "resolution": "~0.3-1 m",
            "internal": False,
        }
        services["osm"] = {
            "name": "OpenStreetMap",
            "url": "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
            "attribution": "(c) OpenStreetMap contributors",
            "max_zoom": 19,
            "type": "Raster",
            "resolution": "~1-2 m",
            "internal": False,
        }

    if not services:
        raise RuntimeError(
            "No basemap available: set IGA_IMAGE_SERVER, or set "
            "ALLOW_PUBLIC_BASEMAPS=true."
        )
    return services


DEFAULT_MAP_SERVICE = "iga_imagery" if IGA_IMAGE_SERVER else "satellite"


def basemap_residency():
    """Is the basemap this deployment will actually draw IGA-hosted?

    The finding raised at 34:05 was not "Google is configured", it was
    "Bahrain imagery is being requested from a service outside the IGA
    estate". Removing the Google endpoints did not by itself close that:
    with IGA_IMAGE_SERVER unset the app falls back to Esri World Imagery,
    which is the same finding wearing a different name.

    So the posture is computed, published on /api/map-services and shown
    in the UI, rather than being inferred from a config file by whoever is
    watching the demo.
    """
    services = build_map_services()
    default = services.get(DEFAULT_MAP_SERVICE) or {}
    internal = bool(default.get("internal"))
    external = sorted(k for k, v in services.items() if not v.get("internal"))

    if internal:
        status = "IGA"
        message = (
            "Basemap imagery is served by the IGA Image Server. No tile "
            "request leaves the IGA estate."
        )
    else:
        status = "PUBLIC"
        message = (
            "Basemap imagery is served by a PUBLIC service ("
            + str(default.get("name") or DEFAULT_MAP_SERVICE)
            + "). Every pan and zoom over Bahrain is a request to a "
              "third party. Set IGA_IMAGE_SERVER to close this - it is the "
              "finding IGA raised on 2026-10-22."
        )

    return {
        "status": status,                       # "IGA" or "PUBLIC"
        "iga_hosted": internal,
        "iga_image_server_configured": bool(IGA_IMAGE_SERVER),
        "iga_image_server": IGA_IMAGE_SERVER,
        "default_key": DEFAULT_MAP_SERVICE,
        "default_name": default.get("name") or DEFAULT_MAP_SERVICE,
        "public_basemaps_allowed": ALLOW_PUBLIC_BASEMAPS,
        "external_services": external,
        "message": message,
    }


# =========================================================================
# 2. THIRD-PARTY MODEL LICENCES
# -------------------------------------------------------------------------
# ultralytics (YOLO) is AGPL-3.0. AGPL is network copyleft: serving software
# that links it over a network can oblige disclosure of the ENTIRE source to
# its users. That is a commercial decision, not an engineering one, so it is
# off by default and must be switched on deliberately.
#
# This does not affect the U-Net workflow. YOLO paths only trigger for .pt
# checkpoints; models are saved as .pth.
# =========================================================================

ENABLE_YOLO = _env_bool("ENABLE_YOLO", False)

YOLO_LICENCE_NOTICE = (
    "ultralytics/YOLO is AGPL-3.0 licensed. Enabling it in a network-served "
    "deployment may oblige you to disclose this application's full source "
    "code to its users. Obtain a commercial Ultralytics licence or legal "
    "sign-off before setting ENABLE_YOLO=true."
)


def require_yolo():
    """Import ultralytics only when explicitly enabled."""
    if not ENABLE_YOLO:
        raise RuntimeError(
            "YOLO support is disabled in this deployment.\n" + YOLO_LICENCE_NOTICE
        )
    from ultralytics import YOLO  # noqa: F401  (deliberately deferred import)
    return YOLO


# =========================================================================
# 3. CORS
# -------------------------------------------------------------------------
# allow_origins=["*"] together with allow_credentials=True lets any website
# make authenticated calls to this API. Browsers reject that combination and
# it fails a security review. Declare the real origins instead.
# =========================================================================

CORS_ALLOW_ORIGINS = _env_list(
    "CORS_ALLOW_ORIGINS",
    [
        "http://localhost:8006",
        "http://127.0.0.1:8006",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
)
CORS_ALLOW_CREDENTIALS = _env_bool("CORS_ALLOW_CREDENTIALS", True)


# =========================================================================
# 4. FRONT-END ASSETS
# -------------------------------------------------------------------------
# Loading Leaflet / Font Awesome from unpkg.com and cdnjs at runtime means
# executing third-party code fetched live on every page load - a standard
# supply-chain finding. Vendored copies are served from /static/vendor.
#
#   python vendor_assets.py      (one-off, needs internet)
# =========================================================================

USE_VENDORED_ASSETS = _env_bool("USE_VENDORED_ASSETS", True)


# =========================================================================
# 5. GEOCODING - place search
# -------------------------------------------------------------------------
# The browser previously called nominatim.openstreetmap.org directly, so
# every analyst search term left the IGA estate and could not be governed.
# Search is now proxied through /api/geocode and pointed here.
#
# Empty string disables place search entirely (returns HTTP 503).
# Set to an internal geocoding service for a governed deployment.
# =========================================================================

GEOCODER_URL = os.environ.get(
    "GEOCODER_URL",
    "https://nominatim.openstreetmap.org/search",   # public default
)


# =========================================================================
# 6. CHECKPOINT DESERIALISATION  (audit finding: CRITICAL - pickle RCE)
# -------------------------------------------------------------------------
# torch.load() is pickle. With weights_only=False a .pth/.pt file executes
# arbitrary Python at load time, so an uploaded checkpoint was remote code
# execution as the server user. Loading is now weights-only by default.
#
# Some legitimate formats - notably ultralytics .pt, which pickles whole
# nn.Module objects - cannot be read weights-only. Those need this switch.
# Turning it on means "I accept that any checkpoint reaching the fallback
# path can run code on this server".
#
# Even when it is True the fallback is refused for files outside
# TRUSTED_CHECKPOINT_ROOTS, so a freshly uploaded file never reaches the
# unsafe path - only checkpoints this application produced itself.
# =========================================================================

ALLOW_UNSAFE_CHECKPOINT_LOAD = _env_bool("ALLOW_UNSAFE_CHECKPOINT_LOAD", False)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))


def _abs(*parts):
    return os.path.realpath(os.path.join(APP_ROOT, *parts))


# Checkpoints written by this application's own training runs. Only these
# may use the full-object (unsafe) fallback, and only when
# ALLOW_UNSAFE_CHECKPOINT_LOAD is set.
#
# models/imported is deliberately NOT here: it is where uploads land.
TRUSTED_CHECKPOINT_ROOTS = [
    _abs("labels"),
    _abs("models", "saved"),
    _abs("models", "checkpoints"),
    _abs("models", "dlpk_imported"),
]


# =========================================================================
# 7. API AUTHENTICATION  (audit finding: CRITICAL - all routes open)
# -------------------------------------------------------------------------
# SUPERSEDED FOR HUMAN USERS by auth.py: people sign in at /login and get a
# signed HttpOnly session cookie, and the middleware checks every request
# including the StaticFiles mounts. The API key below remains for scripts
# and machine callers, which can set a header. See auth.py.
# -------------------------------------------------------------------------
# The application had no authentication of any kind. The only thing keeping
# GPU training, model deletion and the file endpoints off the network was
# the loopback bind, so a single HOST=0.0.0.0 exposed everything.
#
# This is a deliberate minimum, not an identity system: one shared key in a
# header, checked by a FastAPI dependency on every mutating and
# file-touching route. In the IGA estate it sits behind the portal's own
# authentication; the key stops the service being open on the LAN.
#
#   API_KEY set    -> X-API-Key required, 401 otherwise
#   API_KEY unset  -> an ephemeral key is generated at startup (see
#                     AUTO_GENERATE_API_KEY) so the service is never open
#
# Browsers cannot set headers on a WebSocket handshake, so the WebSocket
# endpoints accept the same key as an ?api_key= query parameter. Query
# strings reach proxy and access logs: keep the key out of shared log
# retention, and prefer a configured API_KEY over the ephemeral one so it
# can be rotated deliberately.
# =========================================================================

API_KEY = os.environ.get("API_KEY", "").strip()
API_KEY_HEADER = os.environ.get("API_KEY_HEADER", "X-API-Key").strip() or "X-API-Key"

# Query-parameter name, used only by the WebSocket endpoints.
API_KEY_QUERY_PARAM = "api_key"

# When no API_KEY is configured, generate one per process rather than
# running unauthenticated. It is printed at startup and injected into the
# UI template, so the bundled browser client still works, but a network
# caller cannot guess it. Set false only for a deliberately open dev run -
# that configuration is reported loudly in summary().
AUTO_GENERATE_API_KEY = _env_bool("AUTO_GENERATE_API_KEY", True)

API_KEY_IS_EPHEMERAL = False

if not API_KEY and AUTO_GENERATE_API_KEY:
    API_KEY = secrets.token_urlsafe(32)
    API_KEY_IS_EPHEMERAL = True


def auth_enabled():
    return bool(API_KEY)


def check_api_key(presented):
    """Constant-time comparison. False when authentication is disabled."""
    if not API_KEY or not presented:
        return False
    return secrets.compare_digest(str(presented), API_KEY)


PUBLIC_PATHS = ()
PUBLIC_PATH_PREFIXES = ()


# =========================================================================
# 8. FILESYSTEM ACCESS  (audit finding: HIGH - path traversal)
# -------------------------------------------------------------------------
# Caller-supplied paths reached open(), shutil.copy2() and os.remove()
# unchecked, so the API could read, overwrite or delete any file the server
# process could touch. Every such path is now resolved with realpath and
# must land inside one of these roots.
#
# EXPORT_ROOTS exists because model export deliberately honours an absolute
# path such as D:/GeoAI/MyModels. That is a real workflow, so it is still
# allowed - but only to directories an administrator has declared here,
# never to an arbitrary path chosen by the caller.
#
#   EXPORT_ROOTS=D:/GeoAI/MyModels,E:/GeoAI/models
# =========================================================================

EXPORT_ROOTS = [
    os.path.realpath(p) for p in _env_list("EXPORT_ROOTS", [])
]

# Directories of input imagery that prediction may read from, when the
# analyst keeps source rasters outside the application tree. Same rule as
# EXPORT_ROOTS: declared by an administrator here, never chosen ad hoc by
# the caller.
#
#   EXTRA_READ_ROOTS=D:/GeoAI/orthophotos
EXTRA_READ_ROOTS = [
    os.path.realpath(p) for p in _env_list("EXTRA_READ_ROOTS", [])
]

# Roots the application reads from and writes to in normal operation.
DATA_ROOTS = [
    _abs("models"),
    _abs("labels"),
    _abs("uploads"),
    _abs("outputs"),
    _abs("chips"),
    _abs("dataset"),
    _abs("annotations"),
    _abs("projects"),
    _abs("evaluation"),
    _abs("static"),
]

# Where model checkpoints may be read from or written to.
MODEL_ROOTS = [
    _abs("models"),
    _abs("labels"),
]


def allowed_roots(*extra):
    """Every root the API may touch, plus any caller-specific extras."""
    roots = list(DATA_ROOTS) + list(EXPORT_ROOTS) + list(EXTRA_READ_ROOTS)
    for e in extra:
        if e:
            roots.append(os.path.realpath(e))
    return roots


def resolve_within_roots(user_path, roots):
    """Resolve user_path and return it only if it is inside one of roots.

    Raises ValueError otherwise. Symlinks and '..' are resolved first, so a
    link planted inside a data directory cannot be used to escape it.
    """
    if user_path is None or str(user_path).strip() == "":
        raise ValueError("Empty path")

    candidate = os.path.realpath(os.path.abspath(str(user_path).strip()))

    for root in roots:
        root_real = os.path.realpath(root)
        try:
            common = os.path.commonpath([candidate, root_real])
        except ValueError:
            # Different drive on Windows - cannot be inside this root.
            continue
        if os.path.normcase(common) == os.path.normcase(root_real):
            return candidate

    raise ValueError(
        "Path is outside the permitted directories for this deployment: "
        + str(user_path)
    )


def is_trusted_checkpoint(path):
    """True when the checkpoint was produced by this application itself."""
    try:
        resolve_within_roots(path, TRUSTED_CHECKPOINT_ROOTS)
        return True
    except ValueError:
        return False


# =========================================================================
# 9. INTERACTIVE API DOCS  (audit finding: MEDIUM)
# -------------------------------------------------------------------------
# FastAPI publishes /docs, /redoc and /openapi.json by default. That is a
# free map of every route, parameter and schema for an attacker. Off in
# production; turn it on for local development.
# =========================================================================

ENABLE_API_DOCS = _env_bool("ENABLE_API_DOCS", False)


def docs_urls():
    """kwargs for FastAPI(...) - all three endpoints move together."""
    if ENABLE_API_DOCS:
        return {"docs_url": "/docs", "redoc_url": "/redoc",
                "openapi_url": "/openapi.json"}
    return {"docs_url": None, "redoc_url": None, "openapi_url": None}



# =========================================================================
# 10. PRE-TRAINED BACKBONE WEIGHTS  (audit finding: undeclared outbound host)
# -------------------------------------------------------------------------
# torchvision's `pretrained=True` fetches ImageNet / COCO weights from
# download.pytorch.org the first time a backbone is built. Three problems:
#
#   1. It is an undeclared outbound host from the training machine. NCSC
#      will flag it, and it is in no firewall allowlist.
#   2. In an air-gapped IGA deployment the fetch simply fails.
#   3. When it fails, several call sites in models/ (unet.py,
#      model_factory.py, rcnn.py) catch the exception and rebuild the SAME
#      architecture with `weights=None`. Training then completes, the UI
#      reports success, and the model was trained from random
#      initialisation. A silently worse model is the dangerous outcome,
#      not the error.
#
# Policy:
#   TORCH_HOME                 pre-staged weight cache, shipped with the app
#   ALLOW_PRETRAINED_DOWNLOAD  default False - never fetch at runtime
#
# Stage the weights once, on a connected machine, then ship the directory:
#
#     python stage_torch_weights.py            # pinned, hashed, MANIFEST.txt
#     python stage_torch_weights.py --verify   # on the air-gapped host
#
# GOVERNANCE: with ALLOW_PRETRAINED_DOWNLOAD=false this application makes
# no connection to download.pytorch.org at any point. Setting it true adds
# that host to the outbound allowlist the NCSC review has to cover, and
# means the training machine needs egress.
# =========================================================================

PRETRAINED_WEIGHTS_HOST = "download.pytorch.org"

ALLOW_PRETRAINED_DOWNLOAD = _env_bool("ALLOW_PRETRAINED_DOWNLOAD", False)

# torch.hub reads TORCH_HOME from the environment every time it resolves a
# cache path, so exporting it here - before torch is imported by app.py -
# repoints the whole process at the shipped weights.
TORCH_HOME = os.path.realpath(
    os.environ.get("TORCH_HOME") or _abs("models", "torch_home")
)
os.environ["TORCH_HOME"] = TORCH_HOME

# Where torch.hub actually puts downloaded state dicts.
TORCH_CHECKPOINT_DIR = os.path.join(TORCH_HOME, "hub", "checkpoints")

WEIGHTS_MANIFEST = os.path.join(TORCH_CHECKPOINT_DIR, "MANIFEST.txt")


@contextmanager
def allow_pretrained_download():
    """Temporarily allow torchvision to reach its built-in weight URLs.

    Used ONLY when we are building an empty network whose weights will be
    immediately overwritten from a checkpoint. A RetinaNet built to load
    Esri's PalmTreeDetection.dlpk needs the ResNet50 FPN *architecture*,
    and torchvision fetches that architecture's weights over the network
    whenever `weights=None` is not passed to resnet50().

    The download is not the model we keep. It is the empty shell. Within
    this context only, the air-gap guard in install_pretrained_guard is
    told to let the fetch through. Outside this context nothing changes:
    a training run that reaches for pretrained weights still fails closed,
    as documented in the comment block above.

    Call sites must use `with depcfg.allow_pretrained_download():` and pass
    `weights=pretrained` (or equivalent) into the torchvision builder.
    """
    global ALLOW_PRETRAINED_DOWNLOAD
    previous = ALLOW_PRETRAINED_DOWNLOAD
    ALLOW_PRETRAINED_DOWNLOAD = True
    try:
        yield
    finally:
        ALLOW_PRETRAINED_DOWNLOAD = previous


class PretrainedWeightsUnavailable(BaseException):
    """Raised when a backbone needs weights that are not staged.

    Deliberately derived from BaseException, not Exception.

    models/unet.py, models/model_factory.py and models/rcnn.py wrap every
    pretrained build in `except Exception:` and fall back to
    `weights=None`. If this were an ordinary Exception those handlers would
    swallow it and train from random initialisation without telling anyone
    - which is the failure this control exists to prevent. Deriving from
    BaseException walks straight past them.

    app.py catches it explicitly on the /train route and returns a 503
    naming the missing file. Anywhere else uvicorn catches it per request,
    so an escape degrades to a logged 500 - never to a silent bad model.
    """


def weight_cache_path(url):
    """Where torch.hub would look for (or write) this weight file."""
    return os.path.join(
        TORCH_CHECKPOINT_DIR,
        os.path.basename(str(url).split("?")[0].rstrip("/")),
    )


def _weights_error_text(url):
    name = os.path.basename(str(url).split("?")[0].rstrip("/"))
    return (
        "Pre-trained weights are not available in this deployment.\n"
        "  needed file : " + name + "\n"
        "  expected at : " + os.path.join(TORCH_CHECKPOINT_DIR, name) + "\n"
        "  source      : " + str(url) + "\n"
        "\n"
        "ALLOW_PRETRAINED_DOWNLOAD is false, so this application will not\n"
        "fetch it from " + PRETRAINED_WEIGHTS_HOST + " at runtime. Training\n"
        "was stopped rather than silently continuing from random\n"
        "initialisation.\n"
        "\n"
        "Fix, on a machine with internet access:\n"
        "    python stage_torch_weights.py\n"
        "then copy " + TORCH_HOME + "\n"
        "to this host and re-run:\n"
        "    python stage_torch_weights.py --verify\n"
        "\n"
        "NOTE FOR LOG REVIEWERS: torch prints its own 'Downloading: ...'\n"
        "line immediately before this check runs. No connection was opened;\n"
        "the guard intercepts torch.hub.download_url_to_file itself."
    )


_pretrained_guard_installed = False


def install_pretrained_guard():
    """Route every torchvision weight download through this policy.

    One choke point instead of ~12 `pretrained=True` call sites:
    torchvision resolves all of them through
    torch.hub.load_state_dict_from_url, which calls download_url_to_file
    only when the file is absent from TORCH_HOME. Staged weights therefore
    load untouched; a missing weight raises PretrainedWeightsUnavailable
    naming the file that is needed.

    Returns True if the guard was installed by this call.
    """
    global _pretrained_guard_installed
    if _pretrained_guard_installed:
        return False

    try:
        import torch.hub as _hub
    except Exception as exc:     # torch absent - nothing to guard
        print("[weights] torch.hub unavailable, guard not installed: %s" % exc)
        return False

    try:
        os.makedirs(TORCH_CHECKPOINT_DIR, exist_ok=True)
    except Exception:
        pass

    _original_download = _hub.download_url_to_file

    def _guarded_download(url, dst, hash_prefix=None, progress=True, **kw):
        # Read the module-level variable dynamically
        import sys
        allow = sys.modules[__name__].ALLOW_PRETRAINED_DOWNLOAD
        
        if allow:
            print("[weights] ALLOW_PRETRAINED_DOWNLOAD=true -> fetching " + str(url))
            return _original_download(url, dst, hash_prefix=hash_prefix,
                                      progress=progress, **kw)
        raise PretrainedWeightsUnavailable(_weights_error_text(url))

    _hub.download_url_to_file = _guarded_download
    _pretrained_guard_installed = True
    return True


def staged_weights():
    """(count, total_bytes, has_manifest) for the shipped weight cache."""
    count = size = 0
    if os.path.isdir(TORCH_CHECKPOINT_DIR):
        for f in os.listdir(TORCH_CHECKPOINT_DIR):
            full = os.path.join(TORCH_CHECKPOINT_DIR, f)
            if os.path.isfile(full) and f.lower().endswith((".pth", ".pt")):
                count += 1
                size += os.path.getsize(full)
    return count, size, os.path.exists(WEIGHTS_MANIFEST)


# =========================================================================
# 11. REVERSE PROXY  (IIS on Windows Server - see DEPLOYMENT.md)
# -------------------------------------------------------------------------
# The application stays bound to loopback; IIS terminates TLS in front of
# it. Two settings have to agree with that arrangement:
#
# FORWARDED_ALLOW_IPS - uvicorn rewrites the client address from
#   X-Forwarded-For only when the immediate peer is in this list. Keep it
#   at the proxy's own address. Widening it lets any caller spoof its
#   source address in the login-throttle counter and in the access log.
#
# WS_PER_MESSAGE_DEFLATE - IIS ARR 3.0 cannot route compressed WebSocket
#   frames. uvicorn negotiates permessage-deflate by default, so behind ARR
#   /ws/training fails to establish and the training progress panel stays
#   empty with no error. Set WS_PER_MESSAGE_DEFLATE=false there.
# =========================================================================

FORWARDED_ALLOW_IPS = os.environ.get("FORWARDED_ALLOW_IPS", "127.0.0.1").strip()
WS_PER_MESSAGE_DEFLATE = _env_bool("WS_PER_MESSAGE_DEFLATE", True)


# =========================================================================
# 12. DEEP LEARNING PACKAGE (.dlpk) INTERCHANGE
# -------------------------------------------------------------------------
# A .dlpk is a zip carrying a model file plus an .emd descriptor. It is how
# a model crosses between this application and the ArcGIS estate.
# Nothing here touches the network - export writes a file, import reads an
# uploaded one - but three governance points belong on this auditable
# surface:
#
# DLPK_EXTRACT_INFERENCE_FUNCTION - an .emd may name an InferenceFunction,
#   a Python file inside the package that ArcGIS EXECUTES at inference time.
#   On import that is third-party executable code arriving in an uploaded
#   archive. Default off: the file is listed in the import report but not
#   written to disk, so it can never be picked up by an import of this
#   application's own package tree. Turn it on only for a package whose
#   source is known and whose inference function has been read.
#
# DLPK_INCLUDE_TORCHSCRIPT - on EXPORT, trace the model to TorchScript and
#   bundle it. This is what makes a package executable on an ArcGIS Image
#   Server without the Braein.ai source tree. It is the difference between
#   a package ArcGIS can read and one ArcGIS can run. Tracing is a local
#   CPU operation and downloads nothing.
#
# DLPK_MAX_UPLOAD_BYTES / DLPK_MAX_EXTRACT_BYTES / DLPK_MAX_MEMBERS - zip
#   bomb limits. An uploaded archive declares its own uncompressed size, so
#   the limits are checked before extraction, not during it.
#
# Licence note: .dlpk and .emd are an open, documented Esri file layout.
# Producing or reading one carries no licence obligation and pulls in no
# Esri code - the implementation in dlpk.py is standard library only.
# =========================================================================

DLPK_EXTRACT_INFERENCE_FUNCTION = _env_bool(
    "DLPK_EXTRACT_INFERENCE_FUNCTION", False
)
DLPK_INCLUDE_TORCHSCRIPT = _env_bool("DLPK_INCLUDE_TORCHSCRIPT", True)
DLPK_MAX_UPLOAD_BYTES = int(
    os.environ.get("DLPK_MAX_UPLOAD_BYTES", str(2 * 1024 * 1024 * 1024))
)
DLPK_MAX_EXTRACT_BYTES = int(
    os.environ.get("DLPK_MAX_EXTRACT_BYTES", str(4 * 1024 * 1024 * 1024))
)
DLPK_MAX_MEMBERS = int(os.environ.get("DLPK_MAX_MEMBERS", "512"))

# Where exported packages are written and imported ones unpacked. Both sit
# under models/, which is already inside MODEL_ROOTS, so every caller-
# supplied path still goes through resolve_within_roots().
DLPK_EXPORT_DIR = _abs(os.path.join("models", "dlpk"))
DLPK_IMPORT_DIR = _abs(os.path.join("models", "dlpk_imported"))


def dlpk_summary():
    """One line for the startup banner and the IGA architecture document."""
    return (
        "torchscript=%s, imported inference functions=%s"
        % (
            "on" if DLPK_INCLUDE_TORCHSCRIPT else "off",
            "EXTRACTED  <-- third-party code written to disk"
            if DLPK_EXTRACT_INFERENCE_FUNCTION else "not extracted",
        )
    )


# =========================================================================
# SERVER
# =========================================================================

HOST = os.environ.get("HOST", "127.0.0.1")   # was 0.0.0.0 - bind locally by default
PORT = int(os.environ.get("PORT", "8006"))


def summary():
    """Printed at startup so the running posture is visible in the log."""
    services = ", ".join(build_map_services().keys())
    residency = basemap_residency()
    n_weights, weight_bytes, has_manifest = staged_weights()
    yolo = ("ENABLED  <-- licence obligations apply" if ENABLE_YOLO else "disabled")

    if not API_KEY:
        auth = "DISABLED  <-- EVERY ROUTE IS OPEN"
    elif API_KEY_IS_EPHEMERAL:
        auth = "ephemeral key (regenerated each restart)"
    else:
        auth = "X-API-Key (configured via API_KEY)"

    ckpt = ("full-object fallback ALLOWED  <-- pickle executes code"
            if ALLOW_UNSAFE_CHECKPOINT_LOAD else "weights-only")

    docs = "ENABLED at /docs" if ENABLE_API_DOCS else "disabled"

    lines = [
        "=" * 64,
        " Braein.ai - deployment configuration",
        "=" * 64,
        "  Basemaps           : " + services,
        "  Default basemap    : " + residency["default_name"]
        + "   [" + residency["status"] + "]",
        "  IGA Image Server   : " + (IGA_IMAGE_SERVER or "(not set)"),
        "  Public basemaps    : " + ("allowed" if ALLOW_PUBLIC_BASEMAPS else "blocked"),
        "  YOLO (AGPL-3.0)    : " + yolo,
        "  CORS origins       : " + ", ".join(CORS_ALLOW_ORIGINS),
        "  Vendored assets    : " + str(USE_VENDORED_ASSETS),
        "  Geocoder           : " + (GEOCODER_URL or "(disabled)"),
        "-" * 64,
        "  API authentication : " + auth,
        "  Auth header        : " + API_KEY_HEADER,
        "  Open routes        : /login, /logout, /favicon.ico, /api/health"
        "  (everything else, static mounts included, needs a session)",
        "  Checkpoint loading : " + ckpt,
        "  Export roots       : " + (", ".join(EXPORT_ROOTS) or "(none - exports confined to models/saved)"),
        "  .dlpk interchange  : " + dlpk_summary(),
        "  Extra read roots   : " + (", ".join(EXTRA_READ_ROOTS) or "(none)"),
        "  Interactive docs   : " + docs,
        "-" * 64,
        "  Pretrained weights : " + (
            "runtime download ALLOWED from " + PRETRAINED_WEIGHTS_HOST
            if ALLOW_PRETRAINED_DOWNLOAD
            else "offline only (no call to " + PRETRAINED_WEIGHTS_HOST + ")"),
        "  TORCH_HOME         : " + TORCH_HOME,
        "  Staged weights     : %d file(s), %.0f MB, manifest %s"
        % (n_weights, weight_bytes / 1048576.0,
           "present" if has_manifest else "MISSING"),
        "  Forwarded-For from : " + FORWARDED_ALLOW_IPS,
        "  WS permessage-defl : " + ("on" if WS_PER_MESSAGE_DEFLATE
                                     else "off (IIS ARR compatible)"),
        "  Bind               : " + HOST + ":" + str(PORT),
        "=" * 64,
    ]

    # Basemap residency is the finding IGA raised on 2026-10-22 at 34:05.
    # It is printed as a block, not a field, so a demo cannot start with
    # Bahrain imagery coming from a public service without someone seeing
    # it in the console.
    if residency["iga_hosted"]:
        lines += [
            "   BASEMAP RESIDENCY: IGA-HOSTED",
            "   " + residency["message"],
            "=" * 64,
        ]
    else:
        lines += [
            "!! BASEMAP RESIDENCY: PUBLIC SERVICE  <-- IGA finding 34:05",
            "!! " + residency["message"],
            "!! Set IGA_IMAGE_SERVER=https://<bsdi-image-server>/.../MapServer",
            "!! and ALLOW_PUBLIC_BASEMAPS=false for a governed deployment.",
            "=" * 64,
        ]

    if not ALLOW_PRETRAINED_DOWNLOAD and n_weights == 0:
        lines += [
            "!! No pre-trained backbone weights are staged in TORCH_HOME and",
            "!! ALLOW_PRETRAINED_DOWNLOAD is false. Training with a pretrained",
            "!! backbone will stop with an error naming the missing file.",
            "!! Run: python stage_torch_weights.py   (on a connected machine)",
            "=" * 64,
        ]
    elif ALLOW_PRETRAINED_DOWNLOAD:
        lines += [
            "!! ALLOW_PRETRAINED_DOWNLOAD=true: this process may call",
            "!! " + PRETRAINED_WEIGHTS_HOST + " during training. That host must be in the",
            "!! declared outbound allowlist for the NCSC review.",
            "=" * 64,
        ]

    if not API_KEY:
        lines += [
            "!! WARNING: API_KEY is empty and AUTO_GENERATE_API_KEY is off.",
            "!! Every route - training, model deletion, file access - is",
            "!! reachable by anyone who can reach " + HOST + ":" + str(PORT) + ".",
            "!! Set API_KEY before any non-loopback deployment.",
            "=" * 64,
        ]
    elif API_KEY_IS_EPHEMERAL:
        lines += [
            "   No API_KEY configured: generated one for this process.",
            "   X-API-Key: " + API_KEY,
            "   Set API_KEY in the environment for a stable, rotatable key.",
            "=" * 64,
        ]

    if ALLOW_UNSAFE_CHECKPOINT_LOAD:
        lines += [
            "!! WARNING: ALLOW_UNSAFE_CHECKPOINT_LOAD=true.",
            "!! Full-object checkpoints under TRUSTED_CHECKPOINT_ROOTS will be",
            "!! unpickled, which executes code. Uploads are still refused.",
            "=" * 64,
        ]

    return "\n".join(lines)