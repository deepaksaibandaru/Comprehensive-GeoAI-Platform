"""GeoAI chip dataset builder with pixel-aligned polygon masks.

The source image is treated as a Web-Mercator slippy-map capture when its
bounds contain a zoom value (as produced by /api/map/capture). GeoJSON
Polygon/MultiPolygon coordinates are projected onto that exact image pixel
grid, clipped to each chip, then rasterized from their real rings. Bounding
rectangles are never used as semantic-segmentation masks.
"""
from __future__ import annotations

import json
import math
import os
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import shape, box, Polygon, MultiPolygon, GeometryCollection, mapping
from shapely.ops import transform as shp_transform


def _mercator_y(lat):
    lat = max(-85.05112878, min(85.05112878, float(lat)))
    r = math.radians(lat)
    return (1.0 - math.asinh(math.tan(r)) / math.pi) / 2.0


def _geo_to_pixel_transform(bounds, width, height):
    """Return lon/lat -> image-pixel transform respecting slippy-map scale."""
    zoom = bounds.get("zoom")
    west, south, east, north = (float(bounds[k]) for k in ("west", "south", "east", "north"))
    if zoom is not None:
        nw = (west + 180.0) / 360.0
        ne = (east + 180.0) / 360.0
        yn = _mercator_y(north)
        ys = _mercator_y(south)
        dx, dy = ne - nw, ys - yn
        if abs(dx) < 1e-15 or abs(dy) < 1e-15:
            raise ValueError("Capture bounds have zero Web-Mercator extent")
        def convert(x, y, z=None):
            xa = np.asarray(x, dtype=float)
            ya = np.asarray(y, dtype=float)
            px = (xa + 180.0) / 360.0
            # vectorized Web-Mercator normalized world Y
            yc = np.clip(ya, -85.05112878, 85.05112878)
            yr = np.radians(yc)
            py = (1.0 - np.arcsinh(np.tan(yr)) / np.pi) / 2.0
            outx = (px - nw) / dx * width
            outy = (py - yn) / dy * height
            if np.isscalar(x) and np.isscalar(y):
                outx, outy = float(outx), float(outy)
            return (outx, outy) if z is None else (outx, outy, z)
        return convert
    # Legacy non-georeferenced image bounds: linear fallback.
    dx, dy = east - west, north - south
    if abs(dx) < 1e-15 or abs(dy) < 1e-15:
        raise ValueError("Capture bounds have zero extent")
    def convert_linear(x, y, z=None):
        xa, ya = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        px = (xa - west) / dx * width
        py = (north - ya) / dy * height
        if np.isscalar(x) and np.isscalar(y):
            px, py = float(px), float(py)
        return (px, py) if z is None else (px, py, z)
    return convert_linear


def _draw_polygon(draw, polygon, offset_x, offset_y, fill=255):
    if polygon.is_empty:
        return
    outer = [(round(x - offset_x), round(y - offset_y)) for x, y in polygon.exterior.coords]
    if len(outer) >= 3:
        draw.polygon(outer, fill=fill)
    # Restore interior holes as background, rather than filling the bbox.
    for ring in polygon.interiors:
        hole = [(round(x - offset_x), round(y - offset_y)) for x, y in ring.coords]
        if len(hole) >= 3:
            draw.polygon(hole, fill=0)


def _rasterize_geometry(mask_image, geom_px, chip_x, chip_y, chip_size):
    draw = ImageDraw.Draw(mask_image)
    clip = box(chip_x, chip_y, chip_x + chip_size, chip_y + chip_size)
    try:
        clipped = geom_px.intersection(clip)
    except Exception:
        clipped = geom_px.buffer(0).intersection(clip)
    if clipped.is_empty:
        return
    geoms = []
    if isinstance(clipped, Polygon):
        geoms = [clipped]
    elif isinstance(clipped, MultiPolygon):
        geoms = list(clipped.geoms)
    elif isinstance(clipped, GeometryCollection):
        for g in clipped.geoms:
            if isinstance(g, Polygon): geoms.append(g)
            elif isinstance(g, MultiPolygon): geoms.extend(g.geoms)
    for poly in geoms:
        _draw_polygon(draw, poly, chip_x, chip_y)


def _geometry_for_feature(feature, transform):
    geom = (feature or {}).get("geometry")
    if not geom or geom.get("type") not in ("Polygon", "MultiPolygon"):
        return None
    try:
        g = shape(geom)
        if g.is_empty:
            return None
        if not g.is_valid:
            g = g.buffer(0)
        if g.is_empty:
            return None
        return shp_transform(transform, g)
    except Exception:
        return None


def _write_mask_visuals(mask, image, vis_path, overlay_path):
    # Binary black/white mask and a translucent red overlay for visual QA.
    mask.save(vis_path)
    rgb = image.convert("RGB").convert("RGBA")
    alpha = Image.new("RGBA", image.size, (255, 0, 0, 0))
    alpha.putalpha(mask.point(lambda p: 105 if p > 0 else 0))
    Image.alpha_composite(rgb, alpha).convert("RGB").save(overlay_path)


class DatasetBuilder:
    """Build train/val/test chips from one georeferenced captured map image."""
    def __init__(self, source_image_path, features, class_names, tile_size=256,
                 overlap=0.2, bounds=None, label="object", model_type="unet", task_type=None,
                 label_padding_tiles=0.75):
        self.source_image_path = str(source_image_path)
        self.features = list(features or [])
        self.class_names = list(class_names or ["background", label])
        self.tile_size = int(tile_size)
        self.overlap = float(overlap)
        self.bounds = bounds or {}
        self.label = str(label or "object")
        self.model_type = str(model_type or "unet").lower()
        self.task_type = task_type
        # Keep chips close to annotated samples. A fractional tile margin creates
        # a small ring of useful background examples around the label boundary.
        self.label_padding_tiles = max(0.0, float(label_padding_tiles))
        if self.tile_size < 16:
            raise ValueError("tile_size must be at least 16 pixels")
        if not (0 <= self.overlap < 1):
            raise ValueError("overlap must be in [0, 1)")

    def build_dataset(self, output_dir):
        image_path = Path(self.source_image_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"Source image not found: {image_path}")
        source = Image.open(image_path).convert("RGB")
        width, height = source.size
        required = ("west", "south", "east", "north")
        if not all(k in self.bounds for k in required):
            raise ValueError("Geographic bounds west/south/east/north are required for mask alignment")
        transform = _geo_to_pixel_transform(self.bounds, width, height)
        pixel_geometries = []
        for feat in self.features:
            g = _geometry_for_feature(feat, transform)
            if g is not None:
                pixel_geometries.append(g)
        if not pixel_geometries:
            raise ValueError("No valid Polygon/MultiPolygon annotations could be projected onto the captured image")

        base = Path(output_dir)
        ds_root = base / "dataset"
        splits = ("train", "val", "test")
        for split in splits:
            for sub in ("images", "masks", "masks_vis", "masks_overlay"):
                (ds_root / split / sub).mkdir(parents=True, exist_ok=True)
        (ds_root / "metadata").mkdir(parents=True, exist_ok=True)

        step = max(1, int(round(self.tile_size * (1.0 - self.overlap))))
        xs = list(range(0, max(1, width - self.tile_size + 1), step))
        ys = list(range(0, max(1, height - self.tile_size + 1), step))
        if not xs or xs[-1] + self.tile_size < width:
            xs.append(max(0, width - self.tile_size))
        if not ys or ys[-1] + self.tile_size < height:
            ys.append(max(0, height - self.tile_size))
        xs, ys = sorted(set(xs)), sorted(set(ys))

        # Instead of tiling the whole capture (which can produce hundreds of
        # empty/black chips when labels occupy only a small part), build a spatial
        # neighbourhood around the actual label shapes. This is a geometry buffer,
        # not a bounding-box mask: rasterization below still uses original rings.
        from shapely.ops import unary_union
        label_union = unary_union(pixel_geometries)
        padding_px = self.tile_size * self.label_padding_tiles
        neighbourhood = label_union.buffer(padding_px, cap_style=3, join_style=2)

        # Rasterize once on the full source-image pixel grid, then crop the mask
        # with the exact same (x, y) window as the image chip. This avoids subtle
        # per-chip clipping/rounding differences at tile boundaries.
        full_mask = Image.new("L", (width, height), 0)
        _rasterize_geometry(full_mask, label_union, 0, 0, max(width, height))

        candidate_tiles = []
        for yy in ys:
            for xx in xs:
                tile_box = box(xx, yy, xx + self.tile_size, yy + self.tile_size)
                if tile_box.intersects(neighbourhood):
                    candidate_tiles.append((xx, yy))
        if not candidate_tiles:
            # Safety fallback: include the tile containing the label centroid.
            cx, cy = label_union.centroid.x, label_union.centroid.y
            xx = min(max(0, int(cx // step) * step), max(0, width - self.tile_size))
            yy = min(max(0, int(cy // step) * step), max(0, height - self.tile_size))
            candidate_tiles = [(xx, yy)]

        # Remove stale chips from a previous, larger capture before writing this run.
        # Otherwise old far-away negative chips would remain on disk and be trained.
        for split in splits:
            for sub in ("images", "masks", "masks_vis", "masks_overlay"):
                folder = ds_root / split / sub
                for old_file in folder.glob("*.png"):
                    old_file.unlink()

        chips = []
        chip_bounds = {}
        for x, y in candidate_tiles:
                # Crop-pad at image edges so every model chip has exactly tile_size pixels.
                chip = source.crop((x, y, min(width, x + self.tile_size), min(height, y + self.tile_size)))
                if chip.size != (self.tile_size, self.tile_size):
                    padded = Image.new("RGB", (self.tile_size, self.tile_size), (0, 0, 0))
                    padded.paste(chip, (0, 0)); chip = padded
                # Crop the full-resolution mask using the exact same chip origin.
                # Pad at image edges exactly as the image chip is padded.
                mask = full_mask.crop((x, y, min(width, x + self.tile_size), min(height, y + self.tile_size)))
                if mask.size != (self.tile_size, self.tile_size):
                    padded_mask = Image.new("L", (self.tile_size, self.tile_size), 0)
                    padded_mask.paste(mask, (0, 0))
                    mask = padded_mask
                positive = int(np.count_nonzero(np.asarray(mask) > 0)) > 0
                # Keep every chip: negative chips are important training examples.
                stem = f"chip_{x:06d}_{y:06d}.png"
                chips.append({"name": stem, "x": x, "y": y, "image": chip, "mask": mask, "positive": positive,
                              "fg_px": int(np.count_nonzero(np.asarray(mask) > 0))})
                # Store bounds for the chip in geographic coordinates.
                inv_bounds = self.bounds
                if inv_bounds.get("zoom") is not None:
                    zoom = int(inv_bounds["zoom"])
                    def px_to_lon(px): return (px / width) * (float(inv_bounds["east"]) - float(inv_bounds["west"])) + float(inv_bounds["west"])
                    # Pixel x uses normalized Mercator world X, which is linear in longitude.
                    def px_to_lat(py):
                        yn = _mercator_y(float(inv_bounds["north"])) + (py / height) * (_mercator_y(float(inv_bounds["south"])) - _mercator_y(float(inv_bounds["north"])))
                        return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * yn))))
                    cb = {"west": px_to_lon(x), "east": px_to_lon(min(width, x + self.tile_size)),
                          "north": px_to_lat(y), "south": px_to_lat(min(height, y + self.tile_size)),
                          "zoom": zoom, "crs": "EPSG:4326"}
                else:
                    cb = {"west": float(inv_bounds["west"]) + x / width * (float(inv_bounds["east"]) - float(inv_bounds["west"])),
                          "east": float(inv_bounds["west"]) + min(width, x+self.tile_size) / width * (float(inv_bounds["east"]) - float(inv_bounds["west"])),
                          "north": float(inv_bounds["north"]) - y / height * (float(inv_bounds["north"]) - float(inv_bounds["south"])),
                          "south": float(inv_bounds["north"]) - min(height, y+self.tile_size) / height * (float(inv_bounds["north"]) - float(inv_bounds["south"])),
                          "crs": "EPSG:4326"}
                chip_bounds[stem] = cb

        # Reproducible stratified split, preserving positive chips in each split where possible.
        rng = random.Random(42)
        positives = [c for c in chips if c["positive"]]
        negatives = [c for c in chips if not c["positive"]]
        rng.shuffle(positives); rng.shuffle(negatives)
        def split_group(group):
            n = len(group)
            a = int(n * 0.70); b = int(n * 0.20)
            return {"train": group[:a], "val": group[a:a+b], "test": group[a+b:]}
        pos_split, neg_split = split_group(positives), split_group(negatives)
        split_chips = {s: pos_split[s] + neg_split[s] for s in splits}
        for split in splits:
            rng.shuffle(split_chips[split])

        chip_stats = {}
        for split in splits:
            stats = {"total": 0, "positive": 0, "negative": 0, "fg_px": 0, "total_px": 0, "unpaired": 0}
            for chip in split_chips[split]:
                name = chip["name"]
                img_out = ds_root / split / "images" / name
                mask_out = ds_root / split / "masks" / name
                chip["image"].save(img_out)
                chip["mask"].save(mask_out)
                _write_mask_visuals(chip["mask"], chip["image"],
                    ds_root / split / "masks_vis" / name,
                    ds_root / split / "masks_overlay" / name)
                stats["total"] += 1
                stats["positive"] += int(chip["positive"])
                stats["negative"] += int(not chip["positive"])
                stats["fg_px"] += chip["fg_px"]
                stats["total_px"] += self.tile_size * self.tile_size
            chip_stats[split] = stats

        with open(ds_root / "chip_bounds.json", "w", encoding="utf-8") as f:
            json.dump(chip_bounds, f, indent=2)
        target_manifest = {
            "created": datetime.now().isoformat(), "source_image": str(image_path),
            "source_image_size": [width, height], "source_bounds": self.bounds,
            "tile_size": self.tile_size, "overlap": self.overlap,
            "label_padding_tiles": self.label_padding_tiles,
            "label_padding_pixels": padding_px,
            "chip_selection": "tiles intersecting label geometry buffered by label_padding_pixels",
            "model_type": self.model_type, "primary_target": "semantic_mask",
            "mask_encoding": "0=background, 255=foreground", "rasterization": "original Polygon/MultiPolygon rings; no bbox fill",
            "feature_count": len(self.features), "valid_polygon_count": len(pixel_geometries),
            "total_chips": len(chips), "split_counts": {s: len(split_chips[s]) for s in splits},
        }
        manifest_path = ds_root / "target_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(target_manifest, f, indent=2)
        metadata = {
            "total_chips": len(chips), "model_family": "semantic_segmentation" if self.model_type == "unet" else self.model_type,
            "primary_target": "semantic_mask", "target_manifest_file": str(manifest_path),
            "chip_previews": [f"/labels/{self.label}/dataset/train/images/{c['name']}" for c in split_chips["train"][:8]],
            "tile_size": self.tile_size, "zoom": self.bounds.get("zoom"),
        }
        print(f"[DATASET_BUILDER] source={width}x{height} zoom={self.bounds.get('zoom', 'unspecified')} tile={self.tile_size} step={step}")
        print(f"[DATASET_BUILDER] polygon geometries={len(pixel_geometries)} chips={len(chips)} padding_px={padding_px:.1f} foreground_px={sum(c['fg_px'] for c in chips)}")
        return str(ds_root), metadata, chip_stats


def build_dataset(*args, **kwargs):
    """Compatibility helper for older callers."""
    return DatasetBuilder(*args, **kwargs).build_dataset(kwargs.get("output_dir", "labels/object"))
