"""
arcgis_service.py - native ArcGIS Enterprise integration for Braein.ai
=======================================================================
Talks to an ArcGIS Image Server / Portal directly instead of scraping web
map tiles.

WHY THIS EXISTS
---------------
Tile scraping was never the right way to get training imagery:

  * Tiles are rendered JPEG/PNG at fixed zoom levels. They are lossy,
    8-bit, and already colour-balanced for display - not the source pixels
    the model should learn from.
  * You cannot request a precise AOI, cell size or band combination.
  * Multiband imagery (NIR, NDVI inputs) is impossible through tiles.
  * It sent Bahrain AOIs to a public US-hosted service.

ArcGIS Image Server exposes `exportImage`, which returns exactly the pixels
for a requested extent, size and band selection - straight from the IGA
raster store, inside the estate. That is the native path, and it is also
the one the chip exporter should use.

SECURED SERVICES
----------------
IGA imagery will be secured. `generateToken` against the portal, then pass
the token on every request. Tokens are cached until shortly before expiry.

Configure in deployment_config.py / environment:

    ARCGIS_PORTAL_URL      https://bsdi-portal.gov.bh/portal
    ARCGIS_IMAGE_SERVICE   https://.../Imagery/Ortho/ImageServer
    ARCGIS_USERNAME        service account
    ARCGIS_PASSWORD        service account password (use a secret store)
"""

import os
import time
import math
import requests

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

PORTAL_URL    = os.environ.get("ARCGIS_PORTAL_URL", "").rstrip("/")
IMAGE_SERVICE = os.environ.get("ARCGIS_IMAGE_SERVICE", "").rstrip("/")
USERNAME      = os.environ.get("ARCGIS_USERNAME", "")
PASSWORD      = os.environ.get("ARCGIS_PASSWORD", "")
VERIFY_TLS    = os.environ.get("ARCGIS_VERIFY_TLS", "true").lower() not in ("0", "false", "no")
TIMEOUT       = int(os.environ.get("ARCGIS_TIMEOUT", "60"))

_token_cache = {"token": None, "expires": 0.0}


def is_configured() -> bool:
    """True when an ArcGIS image service has been configured."""
    return bool(IMAGE_SERVICE)


def requires_auth() -> bool:
    return bool(USERNAME and PASSWORD and PORTAL_URL)


# --------------------------------------------------------------------------
# Token handling
# --------------------------------------------------------------------------

def get_token(force: bool = False) -> str:
    """Portal token for secured services. Cached until 60s before expiry.

    Returns "" when the service is public or credentials are not set, which
    callers can pass through harmlessly.
    """
    if not requires_auth():
        return ""

    now = time.time()
    if not force and _token_cache["token"] and now < _token_cache["expires"] - 60:
        return _token_cache["token"]

    url = PORTAL_URL + "/sharing/rest/generateToken"
    try:
        r = requests.post(
            url,
            data={
                "username": USERNAME,
                "password": PASSWORD,
                "client": "referer",
                "referer": os.environ.get("ARCGIS_REFERER", "https://localhost"),
                "expiration": 60,          # minutes
                "f": "json",
            },
            timeout=TIMEOUT,
            verify=VERIFY_TLS,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        raise RuntimeError("ArcGIS token request failed: %s" % exc) from exc

    if "token" not in data:
        raise RuntimeError("ArcGIS token rejected: %s" % data.get("error", data))

    _token_cache["token"] = data["token"]
    # 'expires' is epoch milliseconds
    _token_cache["expires"] = float(data.get("expires", 0)) / 1000.0 or (time.time() + 3300)
    return _token_cache["token"]


def _auth_params() -> dict:
    t = get_token()
    return {"token": t} if t else {}


# --------------------------------------------------------------------------
# Service metadata
# --------------------------------------------------------------------------

def service_info(service_url: str = "") -> dict:
    """Raw ImageServer metadata - bands, pixel type, extent, cell size."""
    url = (service_url or IMAGE_SERVICE).rstrip("/")
    if not url:
        raise RuntimeError("No ArcGIS image service configured (ARCGIS_IMAGE_SERVICE).")
    r = requests.get(url, params={"f": "json", **_auth_params()},
                     timeout=TIMEOUT, verify=VERIFY_TLS)
    r.raise_for_status()
    info = r.json()
    if "error" in info:
        raise RuntimeError("ArcGIS service error: %s" % info["error"])
    return info


def describe(service_url: str = "") -> dict:
    """Condensed summary for the UI and for the IGA architecture document."""
    info = service_info(service_url)
    ext = info.get("extent", {}) or {}
    return {
        "name": info.get("name"),
        "description": (info.get("description") or "")[:300],
        "bandCount": info.get("bandCount"),
        "pixelType": info.get("pixelType"),
        "pixelSizeX": info.get("pixelSizeX"),
        "pixelSizeY": info.get("pixelSizeY"),
        "spatialReference": (info.get("spatialReference") or {}).get("latestWkid")
                             or (info.get("spatialReference") or {}).get("wkid"),
        "extent": {k: ext.get(k) for k in ("xmin", "ymin", "xmax", "ymax")},
        "capabilities": info.get("capabilities"),
        "secured": requires_auth(),
    }


# --------------------------------------------------------------------------
# exportImage - the native way to fetch training imagery
# --------------------------------------------------------------------------

def export_image(
    bbox,
    size=(512, 512),
    bbox_sr=4326,
    image_sr=3857,
    service_url: str = "",
    image_format: str = "tiff",
    band_ids=None,
    no_data=None,
    interpolation: str = "RSP_BilinearInterpolation",
):
    """Return raw image bytes for an exact extent.

    bbox      (xmin, ymin, xmax, ymax) in bbox_sr
    size      (width, height) in pixels
    band_ids  e.g. [0,1,2] for RGB, or [3,2,1] for false colour. None = all.

    Prefer 'tiff' over 'jpgpng': it preserves bit depth and multiple bands,
    which is what a model should train on. JPEG is display output.
    """
    url = (service_url or IMAGE_SERVICE).rstrip("/")
    if not url:
        raise RuntimeError("No ArcGIS image service configured (ARCGIS_IMAGE_SERVICE).")

    params = {
        "bbox": ",".join(str(v) for v in bbox),
        "bboxSR": bbox_sr,
        "imageSR": image_sr,
        "size": "%d,%d" % (size[0], size[1]),
        "format": image_format,
        "interpolation": interpolation,
        "f": "image",
        **_auth_params(),
    }
    if band_ids:
        params["bandIds"] = ",".join(str(b) for b in band_ids)
    if no_data is not None:
        params["noData"] = no_data

    r = requests.get(url + "/exportImage", params=params,
                     timeout=TIMEOUT, verify=VERIFY_TLS)
    r.raise_for_status()

    # On failure the server returns JSON with an image content-type request.
    if r.headers.get("Content-Type", "").startswith("application/json"):
        raise RuntimeError("ArcGIS exportImage failed: %s" % r.text[:400])
    return r.content


def export_chip(lon_min, lat_min, lon_max, lat_max, px=512, **kw):
    """Convenience wrapper: one training chip from a WGS84 bounding box."""
    return export_image((lon_min, lat_min, lon_max, lat_max),
                        size=(px, px), bbox_sr=4326, **kw)


# --------------------------------------------------------------------------
# Tiles - only for interactive display, never for training data
# --------------------------------------------------------------------------

def tile_url_template(service_url: str = "") -> str:
    """Leaflet-compatible template for the map background.

    Display only. Training imagery must come from export_image(), which
    gives exact extents and real pixel values.
    """
    url = (service_url or IMAGE_SERVICE).rstrip("/")
    if not url:
        raise RuntimeError("No ArcGIS image service configured.")
    t = get_token()
    return url + "/tile/{z}/{y}/{x}" + ("?token=" + t if t else "")


# --------------------------------------------------------------------------
# Geographic helpers
# --------------------------------------------------------------------------

# Manama. The diagnostic probe needs somewhere the IGA imagery actually
# covers; probing outside a service's extent returns no image and looks
# exactly like a broken service.
TEST_LON, TEST_LAT = 50.5860, 26.2285


def tile_for_lonlat(lon: float, lat: float, z: int):
    """XYZ tile containing a coordinate - the inverse of bbox_for_tile."""
    n = 2.0 ** z
    x = int((lon + 180.0) / 360.0 * n)
    lat_r = math.radians(lat)
    y = int((1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi)
            / 2.0 * n)
    return max(0, min(int(n) - 1, x)), max(0, min(int(n) - 1, y))


def bbox_for_tile(z: int, x: int, y: int):
    """WGS84 bbox of an XYZ tile - lets tile-indexed code move to exportImage."""
    n = 2.0 ** z
    lon1 = x / n * 360.0 - 180.0
    lon2 = (x + 1) / n * 360.0 - 180.0
    lat1 = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    lat2 = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    return (lon1, min(lat1, lat2), lon2, max(lat1, lat2))


def health() -> dict:
    """Connectivity check for the Data/diagnostics screen."""
    if not is_configured():
        return {"ok": False, "reason": "ARCGIS_IMAGE_SERVICE is not set"}
    try:
        d = describe()
        return {"ok": True, "service": d}
    except Exception as exc:
        return {"ok": False, "reason": str(exc)}


# --------------------------------------------------------------------------
# Service capabilities - what this service can actually serve
# --------------------------------------------------------------------------
# A hard-won distinction. `/tile/{z}/{y}/{x}` only works on a service with a
# published TILE CACHE. Many Image Services are dynamic: they render on
# request and have no cache at all. Point a Leaflet tileLayer at one of those
# and every tile 404s or returns a JSON error that the map draws as nothing -
# a blank grey map with no console error naming the cause.
#
# So never assume. Ask the service, cache the answer, and fall back to
# exportImage when there is no cache.

_caps_cache = {}


def service_kind(service_url: str = "") -> dict:
    """What kind of service this is and how to fetch pixels from it.

    Returns {kind, tiled, tile_info, bands, pixel_type, error}.
    Cached per URL - this is three round trips per deployment, not per tile.
    """
    url = (service_url or IMAGE_SERVICE).rstrip("/")
    if not url:
        return {"kind": None, "tiled": False, "error": "no service configured"}
    if url in _caps_cache:
        return _caps_cache[url]

    out = {"url": url, "kind": None, "tiled": False, "tile_info": None,
           "bands": None, "pixel_type": None, "error": None,
           "single_fused_cache": None}
    try:
        info = service_info(url)
    except Exception as exc:
        out["error"] = str(exc)
        # Do NOT cache a failure: the service may simply be starting up, and
        # caching it would keep the map blank until the process restarts.
        return out

    low = url.lower()
    out["kind"] = ("ImageServer" if low.endswith("/imageserver")
                   else "MapServer" if low.endswith("/mapserver")
                   else "unknown")
    out["bands"] = info.get("bandCount")
    out["pixel_type"] = info.get("pixelType")

    # ImageServer announces a cache through tileInfo + cacheType/capabilities;
    # MapServer through singleFusedMapCache.
    tile_info = info.get("tileInfo")
    out["tile_info"] = bool(tile_info)
    fused = info.get("singleFusedMapCache")
    out["single_fused_cache"] = fused

    if out["kind"] == "MapServer":
        out["tiled"] = bool(fused)
    else:
        caps = (info.get("capabilities") or "")
        out["tiled"] = bool(tile_info) and ("Tilemap" in caps or "Image" in caps
                                            or info.get("cacheType") == "Map")
        # An ImageServer with tileInfo but cacheType 'Elevation'/none still
        # usually serves /tile. Trust tileInfo, and the tile() fallback will
        # cover us if it does not.
        if tile_info and not out["tiled"]:
            out["tiled"] = True

    _caps_cache[url] = out
    return out


def reset_capability_cache():
    """Forget cached service capabilities (after a config change)."""
    _caps_cache.clear()


# --------------------------------------------------------------------------
# Server-side tiles - the browser never talks to the imagery host
# --------------------------------------------------------------------------

def tile(z: int, x: int, y: int, service_url: str = "",
         size: int = 256, image_format: str = "png"):
    """Return the bytes of one display tile, fetched by the SERVER.

    Two reasons this does not happen in the browser:

      1. SECURED SERVICES. IGA imagery will need a token. A browser cannot
         add a header to a Leaflet tile request, so the only way to pass one
         client-side is ?token= in the URL - which lands in IIS logs, ARR
         logs and the browser history. This is the same pattern that was
         removed from the WebSocket handshake. Fetching here keeps the token
         in the process.

      2. DYNAMIC SERVICES. If there is no tile cache, /tile/ does not exist
         and exportImage must be called on the tile's own bounding box
         instead. The caller should not have to know which.

    Returns (bytes, content_type). Raises RuntimeError with a message that
    names the cause - a blank map should never be a mystery.
    """
    url = (service_url or IMAGE_SERVICE).rstrip("/")
    if not url:
        raise RuntimeError("No ArcGIS image service configured (ARCGIS_IMAGE_SERVICE).")

    caps = service_kind(url)
    if caps.get("error"):
        raise RuntimeError("Imagery service unreachable: %s" % caps["error"])

    if caps.get("tiled"):
        # Note the axis order: ArcGIS is /tile/{z}/{row}/{col} = z/y/x, which
        # is NOT Leaflet's {z}/{x}/{y}. Transposing these silently returns
        # the wrong place on earth rather than an error.
        r = requests.get("%s/tile/%d/%d/%d" % (url, z, y, x),
                         params=_auth_params(), timeout=TIMEOUT,
                         verify=VERIFY_TLS)
        ctype = r.headers.get("Content-Type", "")
        if r.status_code == 200 and not ctype.startswith("application/json"):
            return r.content, (ctype or "image/png")
        # A cached service can still miss a level or an area. Fall through
        # to exportImage rather than returning a broken tile.

    # No cache, or the cache had nothing here: render the tile's extent.
    bbox = bbox_for_tile(z, x, y)
    data = export_image(
        bbox,
        size=(size, size),
        bbox_sr=4326,
        image_sr=3857,
        service_url=url,
        image_format="png" if image_format == "png" else image_format,
    )
    return data, "image/png"


def tile_diagnostic(service_url: str = "") -> dict:
    """Why is the map blank? Answer in one call.

    Written because a blank basemap has four different causes that look
    identical on screen: no service configured, wrong URL, service secured
    and credentials missing, or a dynamic service behind a /tile/ URL.
    """
    url = (service_url or IMAGE_SERVICE).rstrip("/")
    if not url:
        return {"ok": False, "cause": "not-configured",
                "detail": "ARCGIS_IMAGE_SERVICE / IGA_IMAGE_SERVER is empty.",
                "fix": "Set IGA_IMAGE_SERVER to the ImageServer or MapServer URL."}

    caps = service_kind(url)
    if caps.get("error"):
        err = caps["error"].lower()
        if "token" in err or "401" in err or "403" in err:
            return {"ok": False, "cause": "unauthorised", "detail": caps["error"],
                    "fix": "Set ARCGIS_USERNAME / ARCGIS_PASSWORD, or share the "
                           "service with the service account."}
        return {"ok": False, "cause": "unreachable", "detail": caps["error"],
                "fix": "Check the URL and that the server is reachable from "
                       "this host (not just from your browser)."}

    try:
        # Probe with a tile that really is over Manama. Hard-coding tile
        # indices by hand is how you end up testing a service's coverage of
        # somewhere it was never meant to cover and calling it a failure, so
        # derive them from the coordinates.
        z = 12
        tx, ty = tile_for_lonlat(TEST_LON, TEST_LAT, z)
        data, ctype = tile(z, tx, ty, url)
        return {"ok": True, "cause": None,
                "probe_tile": "%d/%d/%d" % (z, tx, ty),
                "probe_bbox": [round(v, 4) for v in bbox_for_tile(z, tx, ty)],
                "kind": caps["kind"],
                "served_by": "tile cache" if caps["tiled"] else "exportImage (dynamic)",
                "bytes": len(data), "content_type": ctype,
                "bands": caps["bands"], "pixel_type": caps["pixel_type"]}
    except Exception as exc:
        return {"ok": False, "cause": "no-pixels", "detail": str(exc),
                "kind": caps["kind"], "tiled": caps["tiled"],
                "fix": "The service responded but returned no image for a test "
                       "extent over Bahrain. Check the service's extent and "
                       "that it publishes exportImage."}
