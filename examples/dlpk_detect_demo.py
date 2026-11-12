"""
dlpk_detect_demo.py - run an Esri .dlpk model end to end, reproducibly.
=======================================================================
This is the script behind the car-detection result. It takes an Esri
.dlpk, puts it through Braein.ai's OWN import path, loads the weights,
runs tiled inference over imagery, and writes an annotated PNG plus a
georeferenced GeoJSON.

    python -X utf8 examples/dlpk_detect_demo.py \
        --dlpk "../../CarDetection_USA.dlpk" \
        --lat 42.0450 --lon -88.0370 --zoom 19 \
        --out out/cars

    # or against your own image instead of fetching tiles:
    python -X utf8 examples/dlpk_detect_demo.py \
        --dlpk CarDetection_USA.dlpk --image my_ortho.tif --out out/cars

-----------------------------------------------------------------------
THE TWO SETTINGS THAT DECIDE WHETHER THIS WORKS AT ALL
-----------------------------------------------------------------------
Both were found by measurement, and getting either wrong gives you
*zero* detections rather than poor ones - which reads like a broken
model when it is actually a mismatched input.

1. SCALE.  Esri's CarDetection_USA was trained on imagery finer than
   Esri World Imagery at z=19. Feeding a 400 px chip at its native size
   gave 8 cars at low confidence; upscaling that same chip to 800 px
   gave 109 at 0.51-0.94. The model has a fixed idea of how many pixels
   a car is. --upscale 2.0 is the default for that reason.

   Measured on one chip of a full car park:
       min_size=400  ->   6 detections above 0.5
       min_size=800  ->  30 detections above 0.5, peak 0.952

2. NORMALISATION.  torchvision's detection transform applies ImageNet
   mean/std internally. Pass raw 0-1 pixels and let it do that. If you
   normalise yourself first you double-normalise and detections
   collapse to near zero.

-----------------------------------------------------------------------
WHY THE WEIGHTS JUST LOAD
-----------------------------------------------------------------------
A real Esri *inferencing* package stores a bare torchvision state_dict:

    backbone.body.conv1.weight          <- real Esri, 307 tensors

so torchvision's maskrcnn_resnet50_fpn(num_classes=N+1) accepts it with
strict=True and no key remapping at all. An arcgis.learn TRAINING
checkpoint may instead nest everything under 'model.', which
app.load_checkpoint() handles with a strict-retry.
"""

import argparse
import io
import json
import math
import os
import sys
import urllib.request

# --- make the app package importable however this is invoked ----------
HERE = os.path.dirname(os.path.abspath(__file__))
APPDIR = os.path.dirname(HERE)
if APPDIR not in sys.path:
    sys.path.insert(0, APPDIR)

DEFAULT_SERVICE = ("https://services.arcgisonline.com/ArcGIS/rest/services/"
                   "World_Imagery/MapServer/tile/{z}/{y}/{x}")


# =====================================================================
#  Imagery
# =====================================================================
def deg2tile(lat, lon, z):
    """Lon/lat -> XYZ tile indices (Web Mercator)."""
    n = 2.0 ** z
    xt = (lon + 180.0) / 360.0 * n
    yt = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    return int(xt), int(yt)


def tile2deg(xt, yt, z):
    """Tile indices -> lon/lat of the tile's NW corner."""
    n = 2.0 ** z
    lon = xt / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * yt / n))))
    return lon, lat


def fetch_mosaic(lat, lon, z, n_tiles, service, timeout=30):
    """Stitch an n x n tile mosaic centred on lat/lon.

    NOTE the URL order. ArcGIS serves /tile/{z}/{row}/{col} = z/y/x,
    while the XYZ convention everyone writes is z/x/y. Swapping these
    returns a valid image of the WRONG PLACE - no error, just somewhere
    else on Earth. That is why the template names {y} before {x}.
    """
    from PIL import Image
    x0, y0 = deg2tile(lat, lon, z)
    x0 -= n_tiles // 2
    y0 -= n_tiles // 2
    canvas = Image.new("RGB", (256 * n_tiles, 256 * n_tiles))
    for dx in range(n_tiles):
        for dy in range(n_tiles):
            url = service.format(z=z, x=x0 + dx, y=y0 + dy)
            req = urllib.request.Request(url, headers={"User-Agent": "BraeinAI/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                tile = Image.open(io.BytesIO(r.read())).convert("RGB")
            canvas.paste(tile, (dx * 256, dy * 256))
    # Geo-reference: NW corner of the mosaic + metres per pixel.
    nw_lon, nw_lat = tile2deg(x0, y0, z)
    se_lon, se_lat = tile2deg(x0 + n_tiles, y0 + n_tiles, z)
    return canvas, (nw_lon, nw_lat, se_lon, se_lat)


# =====================================================================
#  Model
# =====================================================================
def read_package(dlpk_path):
    """Braein.ai's own reader - the same code the web UI calls."""
    import dlpk as dlpk_mod
    info = dlpk_mod.describe(dlpk_path)
    print("=" * 70)
    print(" 1. READ THE PACKAGE   (dlpk.describe)")
    print("=" * 70)
    for k in ("framework", "model_configuration", "esri_model_type", "model_type",
              "model_kind", "inference_task", "chip_metadata_format",
              "num_classes", "class_names", "tile_size", "extract_bands",
              "inference_function", "model_file"):
        if info.get(k) is not None:
            print("   %-22s %s" % (k, str(info[k])[:70]))
    return info, dlpk_mod


def import_package(dlpk_mod, dlpk_path, dest):
    print()
    print("=" * 70)
    print(" 2. IMPORT IT        (dlpk.import_dlpk)")
    print("=" * 70)
    res = dlpk_mod.import_dlpk(dlpk_path, dest_dir=dest)
    print("   checkpoint   %s" % os.path.basename(str(res.get("loadable_checkpoint") or "")))
    for note in (res.get("notes") or []):
        print("   NOTE         %s" % str(note)[:150])
    for skipped in (res.get("skipped") or []):
        print("   NOT EXTRACTED %s   <- third-party executable Python" % skipped)
    return res


def build_model(num_classes, chip, upscale):
    """Stock torchvision Mask R-CNN sized for this package's classes."""
    import torch  # noqa: F401  (imported for side effects / availability check)
    from torchvision.models.detection import maskrcnn_resnet50_fpn
    size = int(round(chip * upscale))
    return maskrcnn_resnet50_fpn(
        weights=None, weights_backbone=None,
        num_classes=num_classes + 1,          # + background
        min_size=size, max_size=size,         # <- THE SCALE SETTING
        box_score_thresh=0.05,
        box_detections_per_img=400,
    ), size


# =====================================================================
#  Inference
# =====================================================================
def detect(model, img, chip, stride, threshold):
    """Tiled inference with chip-overlap de-duplication."""
    import numpy as np
    import torch
    from torchvision.ops import nms

    W, H = img.size

    def offsets(total):
        """Chip origins covering `total` px, with the LAST chip flush to
        the edge. A plain range(0, total-chip+1, stride) silently drops
        the right and bottom strips whenever the remainder is smaller
        than one stride - on a 1024 px mosaic with chip 400 / stride 360
        it processed 4 chips instead of 9 and missed a third of the
        image. Objects there are not low-confidence, they are unseen."""
        if total <= chip:
            return [0]
        pos = list(range(0, total - chip + 1, stride))
        if pos[-1] != total - chip:
            pos.append(total - chip)
        return pos

    boxes, scores = [], []
    n_chips = 0
    for oy in offsets(H):
        for ox in offsets(W):
            sub = img.crop((ox, oy, ox + chip, oy + chip))
            # Raw 0-1. torchvision's own transform applies ImageNet
            # mean/std - normalising here as well collapses detections.
            t = torch.from_numpy(np.array(sub)).permute(2, 0, 1).float() / 255.0
            with torch.no_grad():
                d = model([t])[0]
            n_chips += 1
            keep = d["scores"] > threshold
            for b, s in zip(d["boxes"][keep].tolist(), d["scores"][keep].tolist()):
                boxes.append([b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy])
                scores.append(s)

    print("   chips processed      %d" % n_chips)
    print("   raw detections       %d" % len(boxes))
    if not boxes:
        return [], []
    bt = torch.tensor(boxes)
    st = torch.tensor(scores)
    keep = nms(bt, st, 0.3)          # merge duplicates from overlapping chips
    bt, st = bt[keep], st[keep]
    print("   after NMS            %d" % len(bt))
    return bt.tolist(), st.tolist()


def to_geojson(boxes, scores, bounds, size, class_name):
    """Pixel boxes -> WGS84 polygons. Linear in Web Mercator at one zoom."""
    nw_lon, nw_lat, se_lon, se_lat = bounds
    W, H = size

    def px2ll(px, py):
        return (nw_lon + (se_lon - nw_lon) * (px / W),
                nw_lat + (se_lat - nw_lat) * (py / H))

    feats = []
    for (x1, y1, x2, y2), s in zip(boxes, scores):
        a, b = px2ll(x1, y1)
        c, d = px2ll(x2, y2)
        feats.append({
            "type": "Feature",
            "properties": {"class": class_name, "score": round(float(s), 4)},
            "geometry": {"type": "Polygon",
                         "coordinates": [[[a, b], [c, b], [c, d], [a, d], [a, b]]]},
        })
    return {"type": "FeatureCollection", "features": feats}


# =====================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dlpk", required=True, help="path to the .dlpk")
    ap.add_argument("--image", help="local image instead of fetching tiles")
    ap.add_argument("--lat", type=float, default=42.0450)
    ap.add_argument("--lon", type=float, default=-88.0370)
    ap.add_argument("--zoom", type=int, default=19)
    ap.add_argument("--tiles", type=int, default=4, help="n x n 256px tiles")
    ap.add_argument("--service", default=DEFAULT_SERVICE,
                    help="tile URL template using {z} {x} {y}")
    ap.add_argument("--chip", type=int, default=0, help="0 = use the EMD tile size")
    ap.add_argument("--upscale", type=float, default=2.0,
                    help="chip is resampled to chip*upscale before inference")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default="out/detections")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    # ---- 1 & 2: read and import through Braein's own code ------------
    info, dlpk_mod = read_package(args.dlpk)
    res = import_package(dlpk_mod, args.dlpk,
                         os.path.join(APPDIR, "models", "imported"))

    chip = args.chip or int(info.get("tile_size") or 400)
    n_classes = int(info.get("num_classes") or 1)
    class_name = (info.get("class_names") or ["object"])[0]

    # ---- 3: load the weights ----------------------------------------
    import torch
    print()
    print("=" * 70)
    print(" 3. LOAD THE WEIGHTS")
    print("=" * 70)
    model, size = build_model(n_classes, chip, args.upscale)
    pth = res.get("model_path") or res.get("loadable_checkpoint")
    sd = torch.load(pth, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    try:
        model.load_state_dict(sd, strict=True)
        print("   STRICT load OK       %d tensors, zero adaptation" % len(sd))
    except RuntimeError:
        stripped = {k.split("model.", 1)[1]: v for k, v in sd.items()
                    if k.startswith("model.")}
        model.load_state_dict(stripped, strict=True)
        print("   STRICT load OK after stripping 'model.'  (%d tensors)" % len(stripped))
    model.eval()
    print("   inference size       %d px  (chip %d x %.2f upscale)" % (size, chip, args.upscale))

    # ---- 4: imagery --------------------------------------------------
    from PIL import Image, ImageDraw
    print()
    print("=" * 70)
    print(" 4. IMAGERY")
    print("=" * 70)
    if args.image:
        img = Image.open(args.image).convert("RGB")
        bounds = None
        print("   local image          %s  %dx%d" % (args.image, img.width, img.height))
    else:
        img, bounds = fetch_mosaic(args.lat, args.lon, args.zoom, args.tiles, args.service)
        print("   %dx%d tiles at z=%d around %.4f, %.4f -> %dx%d px"
              % (args.tiles, args.tiles, args.zoom, args.lat, args.lon, img.width, img.height))

    # ---- 5: detect ---------------------------------------------------
    print()
    print("=" * 70)
    print(" 5. DETECT")
    print("=" * 70)
    boxes, scores = detect(model, img, chip, max(chip - 40, 1), args.threshold)
    if scores:
        print("   score range          %.3f - %.3f" % (min(scores), max(scores)))

    # ---- 6: write ----------------------------------------------------
    annotated = img.copy()
    dr = ImageDraw.Draw(annotated)
    for b in boxes:
        dr.rectangle(b, outline=(0, 255, 0), width=2)
    png = args.out + ".png"
    annotated.save(png)
    print()
    print("=" * 70)
    print(" 6. OUTPUT")
    print("=" * 70)
    print("   %-20s %s" % ("annotated PNG", os.path.abspath(png)))
    if bounds:
        gj = args.out + ".geojson"
        with open(gj, "w", encoding="utf-8") as f:
            json.dump(to_geojson(boxes, scores, bounds, img.size, class_name), f, indent=1)
        print("   %-20s %s" % ("GeoJSON (WGS84)", os.path.abspath(gj)))
    print()
    print("   %d %s detected above %.2f" % (len(boxes), class_name, args.threshold))
    if not boxes:
        print()
        print("   NOTHING FOUND. Before concluding the model is wrong, try")
        print("   --upscale 1.0 and --upscale 3.0. Scale mismatch produces")
        print("   zero detections, not poor ones.")


if __name__ == "__main__":
    main()
