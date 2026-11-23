"""
BraeinAIMaskRCNN.py - Mask R-CNN Python Raster Function for ArcGIS Pro.

Runs inference using the EXACT reference pipeline from Braein.ai (app.py):
1. Native scale without artificial distortion
2. Direct binary mask extraction (c_masks > 0.50)
3. Contour extraction with cv2.findContours / approxPolyDP (area >= 6)
   with pure skimage/RDP fallback if cv2 is not available
4. Physical size filtering matching app.py (max_size_m = 65m, min_size_m = 2m)
5. Overlap Padding with Centroid Filtering:
   When padding > 0, buildings crossing tile boundaries are detected whole
   in the padded overlap and kept in the inner tile, eliminating broken seam cuts!
6. NO fallback to bounding boxes (only valid polygon contours are emitted)
"""

import json
import os
import sys
import re
import numpy as np

# ---------------------------------------------------------------------------
# Geometry type enum for ArcGIS Pro
# ---------------------------------------------------------------------------

class GeometryType:
    Point = 1
    Multipoint = 2
    Polyline = 3
    Polygon = 4


_FIELDS = {
    'fields': [
        {'name': 'OID',        'type': 'esriFieldTypeOID',      'alias': 'OID'},
        {'name': 'Class',      'type': 'esriFieldTypeString',   'alias': 'Class'},
        {'name': 'Confidence', 'type': 'esriFieldTypeDouble',   'alias': 'Confidence'},
        {'name': 'Shape',      'type': 'esriFieldTypeGeometry', 'alias': 'Shape'},
    ]
}


# ---------------------------------------------------------------------------
# Key conversion: Esri / arcgis.learn naming -> torchvision naming
# ---------------------------------------------------------------------------

def _esri_to_torchvision_key(k):
    m = re.match(r'(.*roi_heads\.mask_head\.)mask_fcn(\d+)\.(.*)', k)
    if m:
        return f'{m.group(1)}{int(m.group(2))-1}.0.{m.group(3)}'
    k = re.sub(r'(backbone\.fpn\.inner_blocks\.\d+)\.(weight|bias)', r'\1.0.\2', k)
    k = re.sub(r'(backbone\.fpn\.layer_blocks\.\d+)\.(weight|bias)', r'\1.0.\2', k)
    k = re.sub(r'rpn\.head\.conv\.(weight|bias)', r'rpn.head.conv.0.0.\1', k)
    return k


def _convert_state_dict(esri_sd):
    return {_esri_to_torchvision_key(k): v for k, v in esri_sd.items()}


# ---------------------------------------------------------------------------
# Pure Python RDP simplification (fallback when cv2 is not present)
# ---------------------------------------------------------------------------

def _rdp_simplify(points, epsilon=1.0):
    if len(points) <= 2:
        return points
    p1 = np.array(points[0], dtype=float)
    p2 = np.array(points[-1], dtype=float)
    line_vec = p2 - p1
    line_len = float(np.linalg.norm(line_vec))
    pts_arr = np.array(points[1:-1], dtype=float)

    if line_len == 0.0:
        dists = np.linalg.norm(pts_arr - p1, axis=1)
    else:
        unit = line_vec / line_len
        diffs = pts_arr - p1
        projs = np.dot(diffs, unit)
        perp = pts_arr - (p1 + np.outer(projs, unit))
        dists = np.linalg.norm(perp, axis=1)

    max_idx = int(np.argmax(dists))
    max_dist = float(dists[max_idx])

    if max_dist > epsilon:
        left = _rdp_simplify(points[:max_idx + 2], epsilon)
        right = _rdp_simplify(points[max_idx + 1:], epsilon)
        return left[:-1] + right
    else:
        return [points[0], points[-1]]


def _polygon_area(ring):
    n = len(ring)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += ring[i][0] * ring[j][1] - ring[j][0] * ring[i][1]
    return abs(area) * 0.5


def _is_full_tile_artifact(ring, tile_size=256, tol=2):
    """
    Returns True if a polygon ring coincides with the full chip boundary
    (spans all 4 tile edges: x=0..tile_size and y=0..tile_size, covering >= 60% of tile area).
    Real buildings never form a solid square encompassing all 4 tile borders.
    """
    if not ring or len(ring) < 4:
        return False
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    if (min(xs) <= tol and max(xs) >= (tile_size - 1 - tol) and
        min(ys) <= tol and max(ys) >= (tile_size - 1 - tol)):
        if _polygon_area(ring) >= 0.60 * (tile_size * tile_size):
            return True
    return False


# ---------------------------------------------------------------------------
# Exact Contour Extraction (reference from app.py lines 3483-3505)
# ---------------------------------------------------------------------------

def _extract_rings_from_mask(m_binary, epsilon=1.0, min_area=6.0):
    """
    Extract closed polygon rings from a 2D binary uint8 mask.
    Matches app.py:
      cv2.findContours -> contourArea >= 6 -> approxPolyDP(epsilon=1.0) -> closed ring.
    If cv2 is not available, falls back to skimage.measure.find_contours.
    NEVER returns bounding boxes.
    """
    rings = []

    # 1. Primary: OpenCV (fastest, matches app.py exactly)
    try:
        import cv2
        contours, _ = cv2.findContours(m_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            for c in contours:
                if cv2.contourArea(c) >= min_area:
                    c_approx = cv2.approxPolyDP(c, epsilon=epsilon, closed=True)
                    pts = [[float(pt[0][0]), float(pt[0][1])] for pt in c_approx]
                    if len(pts) >= 3:
                        if pts[0] != pts[-1]:
                            pts.append(pts[0])
                        rings.append(pts)
            if rings:
                return rings
    except ImportError:
        pass

    # 2. Secondary fallback: skimage.measure (standard in ArcGIS Pro deep learning)
    try:
        import skimage.measure
        contours = skimage.measure.find_contours(m_binary, 0.5)
        for c in contours:
            pts = [[float(p[1]), float(p[0])] for p in c]
            if len(pts) >= 3:
                if pts[0] != pts[-1]:
                    pts.append(pts[0])
                if _polygon_area(pts) >= min_area:
                    simplified = _rdp_simplify(pts, epsilon=epsilon)
                    if len(simplified) >= 4 and simplified[0] == simplified[-1]:
                        rings.append(simplified)
        if rings:
            return rings
    except ImportError:
        pass

    return rings


# ---------------------------------------------------------------------------
# Python Raster Function
# ---------------------------------------------------------------------------

class BraeinAIMaskRCNN:
    def __init__(self):
        self.name = 'Instance Segmentation'
        self.description = 'Braein.ai Mask R-CNN instance segmentation (torchvision reference pipeline).'
        self.model = None
        self.json_info = None
        self.device = None
        self.score_threshold = 0.40
        self.batch_size = 1
        self.padding = 56
        self.tytx = 256
        self.max_size_m = 65.0   # Reference from templates/index.html (predictMaxSize=65) and app.py
        self.min_size_m = 2.0    # Reference from app.py line 3304 (min_size_m=2.0)

    def initialize(self, **kwargs):
        import torch
        from torchvision.models.detection import maskrcnn_resnet50_fpn

        if 'model' not in kwargs:
            return

        model_arg = kwargs['model']
        model_as_file = True
        try:
            with open(model_arg, 'r') as f:
                self.json_info = json.load(f)
        except FileNotFoundError:
            try:
                self.json_info = json.loads(model_arg)
                model_as_file = False
            except Exception:
                raise Exception("Invalid model argument")

        # Device setup
        device_id = kwargs.get('device', None)
        if device_id is not None and int(device_id) >= 0:
            try:
                if torch.cuda.is_available():
                    self.device = torch.device(f'cuda:{int(device_id)}')
                    torch.cuda.set_device(int(device_id))
                else:
                    self.device = torch.device('cpu')
            except Exception:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.tytx = int(self.json_info.get('ImageHeight', 256))
        self.score_threshold = float(self.json_info.get('score_threshold', 0.40))
        self.max_size_m = float(self.json_info.get('max_size_m', 65.0))
        self.min_size_m = float(self.json_info.get('min_size_m', 2.0))

        classes = self.json_info.get('Classes', [])
        num_classes = len(classes) + 1  # foreground + 1 background

        # Construct torchvision model
        self.model = maskrcnn_resnet50_fpn(
            weights=None,
            weights_backbone=None,
            num_classes=num_classes,
        )

        if hasattr(self.model, "transform"):
            self.model.transform.min_size = (self.tytx,)
            self.model.transform.max_size = max(self.tytx * 2, 1024)

        # Locate weights
        model_file = self.json_info.get('ModelFile', '')
        if model_as_file:
            pth_path = os.path.join(os.path.dirname(model_arg), model_file)
        else:
            pth_path = model_file

        raw_sd = torch.load(pth_path, map_location='cpu')
        if isinstance(raw_sd, dict) and 'model_state_dict' in raw_sd:
            raw_sd = raw_sd['model_state_dict']

        tv_sd = _convert_state_dict(raw_sd)
        self.model.load_state_dict(tv_sd, strict=False)

        self.model.to(self.device)
        self.model.eval()

    def getParameterInfo(self, *args):
        required = [
            {'name': 'raster',  'dataType': 'raster',  'required': True,
             'displayName': 'Raster', 'description': 'Input Raster'},
            {'name': 'model',   'dataType': 'string',  'required': True,
             'displayName': 'Input Model Definition (EMD) File',
             'description': 'Input model definition (EMD) JSON file'},
            {'name': 'device',  'dataType': 'numeric', 'required': False,
             'displayName': 'Device ID', 'description': 'Device ID'},
            {'name': 'confidence_threshold', 'dataType': 'numeric', 'required': False,
             'value': 0.40, 'displayName': 'Confidence Threshold',
             'description': 'Confidence score threshold (0.0 - 1.0, default 0.40)'},
            {'name': 'max_size', 'dataType': 'numeric', 'required': False,
             'value': 65.0, 'displayName': 'Max Building Size (meters)',
             'description': 'Maximum building dimension in meters (matches web app 65m; at 0.5m GSD = 130px). Set 0 to disable.'},
            {'name': 'min_size', 'dataType': 'numeric', 'required': False,
             'value': 2.0, 'displayName': 'Min Building Size (meters)',
             'description': 'Minimum building dimension in meters (matches web app min_size 2m = 4px). Set 0 to disable.'},
        ]
        if args:
            return args[0] + required[2:]
        return required

    def getConfiguration(self, **scalars):
        # Dynamically accept arguments passed from ArcGIS Pro UI
        if 'confidence_threshold' in scalars:
            self.score_threshold = float(scalars['confidence_threshold'])
        elif 'threshold' in scalars:
            self.score_threshold = float(scalars['threshold'])

        if 'max_size' in scalars:
            self.max_size_m = float(scalars['max_size'])
        if 'min_size' in scalars:
            self.min_size_m = float(scalars['min_size'])

        if 'padding' in scalars:
            self.padding = int(scalars['padding'])
        if 'tile_size' in scalars:
            self.tytx = int(scalars['tile_size'])
        elif 'tx' in scalars:
            self.tytx = int(scalars['tx'])

        return {
            'extractBands':      self.json_info.get('ExtractBands', [0, 1, 2]),
            'padding':           self.padding,
            'batch_size':        1,
            'tx':                self.tytx,
            'ty':                self.tytx,
            'fixedTileSize':     0,
            'inheritProperties': 2 | 4 | 8,
            'inputMask':         True,
        }

    def getFields(self):
        return json.dumps(_FIELDS)

    def getGeometryType(self):
        return GeometryType.Polygon

    def vectorize(self, **pixelBlocks):
        import torch

        raster_mask = pixelBlocks.get('raster_mask')
        raster_pixels = pixelBlocks['raster_pixels'].copy()
        if raster_mask is not None:
            raster_pixels[np.where(raster_mask == 0)] = 0

        # Uniform empty / flat chip skip (matches app.py line 3551: if chip_t.std() < 0.005: continue)
        img_f = raster_pixels.astype(np.float32) / 255.0
        if float(np.std(img_f)) < 0.005:
            return {'output_vectors': json.dumps({'displayFieldName': '', 'fields': [], 'features': []})}

        chip_h, chip_w = raster_pixels.shape[1], raster_pixels.shape[2]

        # Determine meter-per-pixel resolution (matching app.py lines 3377-3378, 3417, 3434)
        pixel_size = pixelBlocks.get('pixel_size')
        if pixel_size is not None and len(pixel_size) >= 2 and pixel_size[0] > 0:
            m_per_px = (abs(float(pixel_size[0])) + abs(float(pixel_size[1]))) / 2.0
        else:
            # Default typical satellite GSD (0.50m) if not provided by ArcGIS
            m_per_px = 0.50

        # Ensure native 1:1 scale for this block size
        if hasattr(self.model, "transform"):
            self.model.transform.min_size = (min(chip_h, chip_w),)
            self.model.transform.max_size = max(chip_h, chip_w, 1024)

        img_tensor = torch.from_numpy(img_f).to(self.device)

        with torch.inference_mode():
            outputs = self.model([img_tensor])

        out = outputs[0]
        c_boxes  = out['boxes'].cpu().numpy()
        c_scores = out['scores'].cpu().numpy()
        c_masks  = out['masks'].cpu().numpy() if 'masks' in out else None
        c_labels = out['labels'].cpu().numpy()

        features = {
            'displayFieldName': '',
            'fieldAliases': {'FID': 'FID', 'Class': 'Class', 'Confidence': 'Confidence'},
            'geometryType': 'esriGeometryPolygon',
            'fields': [
                {'name': 'FID',        'type': 'esriFieldTypeOID',    'alias': 'FID'},
                {'name': 'Class',      'type': 'esriFieldTypeString', 'alias': 'Class'},
                {'name': 'Confidence', 'type': 'esriFieldTypeDouble', 'alias': 'Confidence'},
            ],
            'features': []
        }

        classes = self.json_info.get('Classes', [])
        oid = 1

        # EXACT REFERENCE LOOP from app.py lines 3465-3505 and 3565-3600
        for i in range(len(c_scores)):
            s = float(c_scores[i])
            if s < self.score_threshold:
                continue

            b = c_boxes[i]
            bw = float(b[2] - b[0])
            bh = float(b[3] - b[1])
            # Matches app.py line 3471 / 3570: if min(bw, bh) < 4: continue
            if min(bw, bh) < 4:
                continue

            # Matches app.py lines 3474-3479 / 3572-3576:
            # Physical size filtering in meters (predictMaxSize = 65m in templates/index.html)
            w_m = bw * m_per_px
            h_m = bh * m_per_px
            if self.min_size_m > 0 and (w_m < self.min_size_m or h_m < self.min_size_m):
                continue
            if self.max_size_m > 0 and (w_m > self.max_size_m or h_m > self.max_size_m):
                continue

            rings = None
            if c_masks is not None and i < len(c_masks):
                # Matches app.py line 3485 / 3581: m_binary = (c_masks[i, 0] > 0.50).astype(np.uint8)
                m_binary = (c_masks[i, 0] > 0.50).astype(np.uint8)
                rings = _extract_rings_from_mask(m_binary, epsilon=1.0, min_area=6.0)

            # Matches app.py line 3504 / 3600: if poly_shp is None: continue
            # NEVER emit a bounding-box fallback! Only real polygon masks!
            if not rings:
                continue

            # Filter whole-tile border artifacts
            valid_rings = []
            tile_area = float(chip_h * chip_w)
            for r in rings:
                r_area = _polygon_area(r)
                if r_area < 6.0:
                    continue
                if r_area >= 0.60 * tile_area:
                    continue
                if _is_full_tile_artifact(r, max(chip_h, chip_w)):
                    continue
                valid_rings.append(r)

            if not valid_rings:
                continue
            rings = valid_rings

            # OVERLAP PADDING & CENTROID FILTERING:
            # If padding > 0, only keep detections whose centroid falls within the inner tile.
            # Detections whose centroid is in the padded margin belong to the neighbor tile.
            # Then shift the polygon coordinates by -padding so that the inner tile origin is (0, 0).
            # This ensures boundary buildings are kept 100% complete with ZERO seam cuts!
            if self.padding > 0:
                xs = [p[0] for p in rings[0]]
                ys = [p[1] for p in rings[0]]
                cx = float(np.mean(xs))
                cy = float(np.mean(ys))

                if (cx < self.padding or cx >= (chip_w - self.padding) or
                    cy < self.padding or cy >= (chip_h - self.padding)):
                    # Centroid is in the overlap margin -> neighbor tile owns it
                    continue

                # Shift by -padding
                shifted_rings = []
                for r in rings:
                    shifted_rings.append([[float(p[0] - self.padding), float(p[1] - self.padding)] for p in r])
                rings = shifted_rings

            lbl = int(c_labels[i]) - 1
            if 0 <= lbl < len(classes):
                cls_name = classes[lbl].get('Name', str(c_labels[i]))
            else:
                cls_name = "building"

            features['features'].append({
                'attributes': {
                    'OID': oid,
                    'Class': cls_name,
                    'Confidence': s,
                },
                'geometry': {'rings': rings},
            })
            oid += 1

        return {'output_vectors': json.dumps(features)}
