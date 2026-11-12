"""AOI-based geospatial image chip + mask builder.

Chips are read directly from the source raster's native pixel grid. No imagery
resampling or synthetic imagery is performed. Labels are transformed to the
raster CRS and rasterized with each chip's exact affine transform.
"""
from __future__ import annotations
import json, math, random
from pathlib import Path
from typing import List
import numpy as np
from PIL import Image
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window, bounds as window_bounds
from rasterio.warp import transform_geom
from shapely.geometry import shape, Point, box
from shapely.ops import transform as shp_transform
from pyproj import CRS, Transformer


def _geojson_features(path: str, raster_crs) -> List[dict]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    feats = obj.get("features", []) if isinstance(obj, dict) else []
    source_crs = (obj.get("crs", {}).get("properties", {}).get("name") or "EPSG:4326") if isinstance(obj, dict) else "EPSG:4326"
    src, dst = CRS.from_user_input(source_crs), CRS.from_user_input(raster_crs)
    out = []
    for f in feats:
        geom = f.get("geometry")
        if not geom:
            continue
        if src != dst:
            geom = transform_geom(src, dst, geom)
        try:
            sg = shape(geom)
            if sg.is_empty:
                continue
            if not sg.is_valid:
                sg = sg.buffer(0)
            if not sg.is_empty:
                out.append({"geometry": sg, "properties": f.get("properties") or {}})
        except Exception:
            continue
    return out


def _class_id(props, label_property, default_class_id):
    try:
        value = int(props.get(label_property, default_class_id))
    except (TypeError, ValueError):
        value = int(default_class_id)
    return value if value > 0 else max(1, int(default_class_id))


def _to_rgb(src, window, source_percentiles):
    """Read bands without spatial resampling; robustly map non-uint8 values to RGB."""
    count = min(3, src.count)
    if count == 0:
        return None
    arr = src.read(indexes=list(range(1, count + 1)), window=window, masked=True)
    if count == 1:
        arr = np.repeat(arr, 3, axis=0)
    elif count == 2:
        arr = np.concatenate([arr, arr[1:2]], axis=0)
    arr = np.moveaxis(arr[:3], 0, -1)
    if np.issubdtype(arr.dtype, np.integer) and arr.dtype == np.uint8:
        return np.asarray(arr, dtype=np.uint8)
    arr = np.asarray(arr.filled(0) if np.ma.isMaskedArray(arr) else arr, dtype=np.float32)
    lo, hi = source_percentiles
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    return np.clip((arr - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)


def build_aoi_dataset(raster_path: str, labels_geojson: str, output_dir: str,
                      sample_lon: float, sample_lat: float, radius_km: float = 5,
                      aoi_spacing_km: float = 5, chips_per_aoi: int = 100,
                      chip_size: int = 256, stride: int = 128,
                      train_ratio: float = .7, val_ratio: float = .2, seed: int = 42,
                      label_property: str = "class_id", default_class_id: int = 1,
                      max_chips: int = 10000, aoi_buffer_m: float = 1.0) -> dict:
    """Create paired RGB chips and class-ID masks around label-aware AOIs.

    The radius defines the overall sample search area; AOI spacing determines
    center spacing. Candidate windows are anchored to the raster pixel grid,
    kept only when their rasterized mask contains labelled pixels, and split by
    AOI group. `max_chips` is enforced globally.
    """
    chip_size, stride = int(chip_size), int(stride)
    chips_per_aoi, max_chips = int(chips_per_aoi), int(max_chips)
    if chip_size < 16 or stride < 1:
        raise ValueError("chip_size must be >=16 and stride >=1")
    if chips_per_aoi < 1 or max_chips < 1:
        raise ValueError("chips_per_aoi and max_chips must be >=1")
    if radius_km <= 0 or aoi_spacing_km <= 0:
        raise ValueError("radius_km and aoi_spacing_km must be positive")
    if not (0 < train_ratio < 1 and 0 <= val_ratio < 1 and train_ratio + val_ratio < 1):
        raise ValueError("train_ratio and val_ratio must leave a positive test ratio")
    rng = random.Random(int(seed))
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "masks").mkdir(parents=True, exist_ok=True)

    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise ValueError("Source raster has no CRS; georeference it first.")
        features = _geojson_features(labels_geojson, src.crs)
        if not features:
            raise ValueError("No valid features found in labels GeoJSON.")

        # Use a local metric projection for radius and spacing.
        local_crs = CRS.from_proj4(f"+proj=aeqd +lat_0={sample_lat} +lon_0={sample_lon} +datum=WGS84 +units=m +no_defs")
        to_local = Transformer.from_crs("EPSG:4326", local_crs, always_xy=True).transform
        to_wgs = Transformer.from_crs(local_crs, "EPSG:4326", always_xy=True).transform
        sample = Point(*to_local(float(sample_lon), float(sample_lat)))
        radius_m, spacing_m = float(radius_km) * 1000, max(25.0, float(aoi_spacing_km) * 1000)

        # Derive AOI sampling centers from label distribution, never from the
        # full map/raster canvas. Retain only labels inside the requested radius.
        to_local_raster = Transformer.from_crs(src.crs, local_crs, always_xy=True).transform
        label_geometries = []
        for feature in features:
            local_geom = shp_transform(to_local_raster, feature["geometry"])
            if local_geom.distance(sample) <= radius_m:
                label_geometries.append(local_geom)
        if not label_geometries:
            raise ValueError("No annotations fall inside the requested sample radius_km.")
        from shapely.ops import unary_union
        label_union = unary_union(label_geometries)
        min_x, min_y, max_x, max_y = label_union.bounds
        label_extent = box(min_x, min_y, max_x, max_y)
        label_aoi = label_extent.buffer(float(aoi_buffer_m), cap_style=3, join_style=2)
        centers_local = []
        x = min_x + spacing_m / 2.0
        while x <= max_x + 1e-6:
            y = min_y + spacing_m / 2.0
            while y <= max_y + 1e-6:
                point = Point(x, y)
                if label_aoi.covers(point):
                    centers_local.append(point)
                y += spacing_m
            x += spacing_m
        if not centers_local:
            centers_local.append(label_union.representative_point())
        # Ensure sparse edge components are sampled as well.
        centers_local.extend(g.representative_point() for g in label_geometries)
        # Deduplicate nearby centers while preserving stable order.
        unique = []
        for p in centers_local:
            if not any(p.distance(q) < min(25.0, spacing_m * 0.1) for q in unique):
                unique.append(p)
        to_raster = Transformer.from_crs(local_crs, src.crs, always_xy=True).transform
        raster_extent = box(src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
        centers = []
        for p in unique:
            rp = shp_transform(to_raster, p)
            if raster_extent.covers(rp):
                centers.append((rp, p))
        if not centers:
            raise ValueError("No generated AOIs fall inside the supplied raster extent. Use imagery covering the sample radius.")

        shapes = [(f["geometry"], _class_id(f["properties"], label_property, default_class_id)) for f in features]
        # Raster-aligned candidate grid; never resize or reproject chip pixels.
        h, w = src.height, src.width
        inv = ~src.transform
        # Pixel dimensions can be rotated; pixel-space windows still preserve source pixels.
        px_per_meter_x = 1.0 / max(abs(src.transform.a), abs(src.transform.b), 1e-12)
        px_per_meter_y = 1.0 / max(abs(src.transform.d), abs(src.transform.e), 1e-12)
        # AOI local radius is half the spacing with a minimum one-chip footprint.
        half_extent_m = max(float(aoi_spacing_km) * 500.0, float(chip_size) * max(abs(src.transform.a), abs(src.transform.e)))
        rx = max(chip_size, int(math.ceil(half_extent_m * px_per_meter_x)))
        ry = max(chip_size, int(math.ceil(half_extent_m * px_per_meter_y)))
        groups = []
        total_saved_candidates = 0
        # Use feature spatial index-ish bounds prefilter, avoiding repeated geometry parsing.
        geom_bounds = [(geom, cid, geom.bounds) for geom, cid in shapes]
        for ai, (center, center_local) in enumerate(centers):
            row_c, col_c = src.index(center.x, center.y)
            col0, row0 = max(0, col_c - rx), max(0, row_c - ry)
            col1, row1 = min(w, col_c + rx), min(h, row_c + ry)
            # Anchor each window to global raster stride; prevents gaps from off-grid AOI starts.
            first_col = (col0 // stride) * stride
            first_row = (row0 // stride) * stride
            candidates, seen_windows = [], set()
            for row in range(first_row, max(first_row, row1 - chip_size) + 1, stride):
                for col in range(first_col, max(first_col, col1 - chip_size) + 1, stride):
                    if col < 0 or row < 0 or col + chip_size > w or row + chip_size > h:
                        continue
                    key = (row, col)
                    if key in seen_windows:
                        continue
                    seen_windows.add(key)
                    win = Window(col, row, chip_size, chip_size)
                    left, bottom, right, top = window_bounds(win, src.transform)
                    wb = box(min(left, right), min(bottom, top), max(left, right), max(bottom, top))
                    intersect = [(geom, cid) for geom, cid, bounds in geom_bounds
                                 if bounds[0] <= wb.bounds[2] and bounds[2] >= wb.bounds[0]
                                 and bounds[1] <= wb.bounds[3] and bounds[3] >= wb.bounds[1]
                                 and geom.intersects(wb)]
                    if not intersect:
                        continue
                    mask = rasterize(intersect, out_shape=(chip_size, chip_size),
                                     transform=src.window_transform(win), fill=0, dtype="uint8", all_touched=False)
                    if not np.any(mask):
                        continue
                    rgb = _to_rgb(src, win, (np.nan, np.nan))
                    if rgb is None:
                        continue
                    candidates.append({"rgb": rgb, "mask": mask, "row": int(row), "col": int(col),
                                       "transform": list(src.window_transform(win))[:6]})
                    if len(candidates) >= chips_per_aoi:
                        break
                if len(candidates) >= chips_per_aoi:
                    break
            if candidates:
                groups.append({"aoi_id": ai, "chips": candidates,
                               "center_wgs84": [float(v) for v in to_wgs(center_local.x, center_local.y)]})
                total_saved_candidates += len(candidates)
            if total_saved_candidates >= max_chips:
                break

        if not groups:
            raise ValueError("No labelled chips were found. Check raster/label alignment and label density.")
        # Enforce max_chips exactly and maintain grouped split boundaries.
        remain = max_chips
        for group in groups:
            group["chips"] = group["chips"][:remain]
            remain -= len(group["chips"])
        groups = [g for g in groups if g["chips"]]
        rng.shuffle(groups)
        n = len(groups)
        if n >= 3:
            ntrain = max(1, min(n - 2, int(n * train_ratio)))
            nval = max(1, min(n - ntrain - 1, int(n * val_ratio)))
        elif n == 2:
            ntrain, nval = 1, 0
        else:
            ntrain, nval = 1, 0
        split_groups = {"train": groups[:ntrain], "val": groups[ntrain:ntrain+nval], "test": groups[ntrain+nval:]}
        counts = {s: 0 for s in split_groups}
        manifest, index = [], 0
        for split, split_list in split_groups.items():
            for group in split_list:
                for chip in group["chips"]:
                    name = f"aoi{group['aoi_id']:04d}_chip{index:07d}"
                    Image.fromarray(chip["rgb"], mode="RGB").save(root / split / "images" / f"{name}.png")
                    Image.fromarray(chip["mask"].astype(np.uint8), mode="L").save(root / split / "masks" / f"{name}.png")
                    vals, val_counts = np.unique(chip["mask"], return_counts=True)
                    manifest.append({"chip": name, "split": split, "aoi_id": int(group["aoi_id"]),
                                     "row": chip["row"], "col": chip["col"], "width": chip_size, "height": chip_size,
                                     "source_transform": chip["transform"],
                                     "mask_values": [int(x) for x in vals],
                                     "foreground_pixels": int(np.count_nonzero(chip["mask"]))})
                    counts[split] += 1
                    index += 1
        meta = {"source_raster": str(Path(raster_path).resolve()), "labels_geojson": str(Path(labels_geojson).resolve()),
                "source_crs": str(src.crs), "source_transform": list(src.transform)[:6],
                "source_resolution": [float(src.res[0]), float(src.res[1])],
                "source_width": int(src.width), "source_height": int(src.height),
                "sample_lon": float(sample_lon), "sample_lat": float(sample_lat), "radius_km": float(radius_km),
                "aoi_spacing_km": float(aoi_spacing_km), "aoi_buffer_m": float(aoi_buffer_m),
                "chip_size": chip_size, "stride": stride, "chips_per_aoi_limit": chips_per_aoi,
                "aoi_count": len(groups), "chips_per_split": counts, "total_chips": sum(counts.values()),
                "split_strategy": "grouped_by_AOI", "seed": int(seed), "class_id_property": label_property,
                "default_class_id": int(default_class_id), "max_chips": max_chips,
                "pixel_resampling": False, "mask_encoding": "class_id_uint8", "native_source_pixels": True}
        (root / "dataset_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        (root / "chip_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return meta
