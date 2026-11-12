# polygonizer.py

import numpy as np
from shapely.geometry import Polygon, MultiPolygon, mapping
from shapely.validation import make_valid
import json
import geopandas as gpd
import os
import cv2
from datetime import datetime


class Polygonizer:
    """
    Convert segmentation masks to GIS polygons with georeferencing.

    Important:
        The segmentation model is binary semantic segmentation.
        Therefore, several touching objects can appear as one connected
        foreground region.

    This class performs additional instance-separation post-processing:

        1. Small morphological cleanup
        2. Connected-component detection
        3. Distance-transform based object-center detection
        4. Watershed separation of touching/merged objects
        5. Individual polygon generation

    Coordinate convention:
        x=0          -> west
        x=img_width  -> east
        y=0          -> north
        y=img_height -> south

    The pixel -> geo mapping deliberately uses img_width / img_height
    to remain consistent with DatasetBuilder._pixel_to_geo().
    """

    def __init__(
        self,
        bounds,
        crs='EPSG:4326',
        img_width=None,
        img_height=None,
        binary_mode=False,

        # ---------------------------------------------------------
        # Instance-separation parameters
        # ---------------------------------------------------------
        split_touching=True,

        # Remove very thin bridges/noise.
        # CHANGED: 3 -> 5 (removes more noise spurs)
        morphology_open=5,

        # Only run watershed on components larger than this.
        # CHANGED: 80 -> 120 (don't split small tanks)
        split_min_area=120,

        # Peak must reach this fraction of the component's maximum
        # distance-transform value.
        # CHANGED: 0.30 -> 0.40 (stricter peak detection)
        distance_ratio=0.40,

        # Minimum approximate distance between object centers.
        # CHANGED: 4 -> 10 (peaks must be far apart to split)
        min_peak_distance=10,

        # Prevent pathological over-segmentation.
        # CHANGED: 50 -> 80
        max_splits_per_component=80,
    ):

        if isinstance(bounds, dict):
            self.west = float(bounds.get('west', 0))
            self.south = float(bounds.get('south', 0))
            self.east = float(bounds.get('east', 1))
            self.north = float(bounds.get('north', 1))
        else:
            self.west, self.south, self.east, self.north = bounds

        self.crs = crs
        self.img_width = img_width
        self.img_height = img_height
        self.binary_mode = bool(binary_mode)

        # ---------------------------------------------------------
        # Post-processing configuration
        # ---------------------------------------------------------
        self.split_touching = bool(split_touching)

        self.morphology_open = int(morphology_open)

        self.split_min_area = int(split_min_area)

        self.distance_ratio = float(distance_ratio)

        self.min_peak_distance = int(min_peak_distance)

        self.max_splits_per_component = int(
            max_splits_per_component
        )

        self.is_geo = (
            abs(self.west) < 180
            and abs(self.east) < 180
            and abs(self.south) < 90
            and abs(self.north) < 90
        )

        print(
            f"📍 Bounds: "
            f"W={self.west}, "
            f"S={self.south}, "
            f"E={self.east}, "
            f"N={self.north}"
        )

        print(f"🌍 Geographic bounds: {self.is_geo}")

        print(
            f"🖼️ img_width={self.img_width}, "
            f"img_height={self.img_height}, "
            f"binary_mode={self.binary_mode}"
        )

        print(
            f"🔧 Instance separation: "
            f"{self.split_touching}"
        )

        print(
            f"🔧 Morphological opening: "
            f"{self.morphology_open}"
        )

        print(
            f"🔧 Split minimum area: "
            f"{self.split_min_area}px"
        )

        print(
            f"🔧 Distance ratio: "
            f"{self.distance_ratio}"
        )

        print(
            f"🔧 Minimum peak distance: "
            f"{self.min_peak_distance}px"
        )

    # =============================================================
    # PIXEL -> GEO
    # =============================================================

    def pixel_to_geo(self, x, y):
        width = self.east - self.west
        height = self.north - self.south

        if (
            self.is_geo
            and width > 0
            and height > 0
            and self.img_width
            and self.img_height
        ):
            lon = (
                self.west
                + (float(x) / self.img_width) * width
            )

            lat = (
                self.north
                - (float(y) / self.img_height) * height
            )

        else:
            lon = float(x)
            lat = float(y)

        return lon, lat

    # =============================================================
    # POLYGON VALIDATION
    # =============================================================

    def _safe_polygon(self, polygon):
        """
        Validate a Shapely polygon.
        """

        if polygon is None:
            return None

        if polygon.is_empty:
            return None

        try:

            if not polygon.is_valid:
                polygon = make_valid(polygon)

            if polygon.is_empty:
                return None

            return polygon

        except Exception:
            return None

    # =============================================================
    # POLYGON -> GEOJSON
    # =============================================================

    def polygon_to_geojson(
        self,
        polygon,
        properties=None
    ):

        polygon = self._safe_polygon(polygon)

        if polygon is None:
            return None

        properties = properties or {}

        if polygon.geom_type == 'MultiPolygon':

            features = []

            for poly in polygon.geoms:

                poly = self._safe_polygon(poly)

                if poly is None:
                    continue

                features.append(
                    {
                        'type': 'Feature',
                        'geometry': mapping(poly),
                        'properties': properties
                    }
                )

            return features

        return {
            'type': 'Feature',
            'geometry': mapping(polygon),
            'properties': properties
        }

    # =============================================================
    # MORPHOLOGICAL CLEANUP
    # =============================================================

    def _morphological_cleanup(self, binary):
        """
        Remove thin bridges and tiny segmentation artifacts.
        """

        binary = (
            np.asarray(binary)
            .astype(np.uint8)
        )

        if self.morphology_open <= 1:
            return binary

        size = int(self.morphology_open)

        # Always use odd kernel size.
        if size % 2 == 0:
            size += 1

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (size, size)
        )

        cleaned = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            kernel,
            iterations=1
        )

        return cleaned

    # =============================================================
    # CONNECTED COMPONENTS
    # =============================================================

    def _connected_components(self, binary):
        """
        Return individual connected foreground components.
        """

        binary = (
            np.asarray(binary)
            .astype(np.uint8)
        )

        count, labels, stats, centroids = (
            cv2.connectedComponentsWithStats(
                binary,
                connectivity=8
            )
        )

        components = []

        for label_id in range(1, count):

            area = int(
                stats[label_id, cv2.CC_STAT_AREA]
            )

            if area <= 0:
                continue

            component = (
                labels == label_id
            ).astype(np.uint8)

            components.append(
                {
                    'label': label_id,
                    'area': area,
                    'mask': component,
                    'bbox': (
                        int(stats[label_id, cv2.CC_STAT_LEFT]),
                        int(stats[label_id, cv2.CC_STAT_TOP]),
                        int(stats[label_id, cv2.CC_STAT_WIDTH]),
                        int(stats[label_id, cv2.CC_STAT_HEIGHT])
                    ),
                    'centroid': (
                        float(centroids[label_id][0]),
                        float(centroids[label_id][1])
                    )
                }
            )

        return components

    # =============================================================
    # DISTANCE TRANSFORM
    # =============================================================

    def _distance_transform(self, component):
        """
        Calculate distance from every foreground pixel to the
        nearest background pixel.
        """

        component = (
            np.asarray(component)
            .astype(np.uint8)
        )

        distance = cv2.distanceTransform(
            component,
            cv2.DIST_L2,
            5
        )

        return distance

    # =============================================================
    # FIND OBJECT CENTERS
    # =============================================================

    def _find_distance_peaks(self, distance, component):
        """
        Find likely centers of individual objects.

        For storage tanks, each tank normally has its own
        distance-transform maximum.
        """

        # ---------------------------------------------------------
        # NEW: Smooth distance map to avoid spurious pixel-level peaks
        # ---------------------------------------------------------
        distance = cv2.GaussianBlur(
            distance,
            (5, 5),
            0
        )

        max_distance = float(
            distance.max()
        )

        if max_distance <= 1.0:
            return np.zeros_like(
                component,
                dtype=np.uint8
            )

        # ---------------------------------------------------------
        # Relative threshold
        # ---------------------------------------------------------

        threshold = (
            max_distance
            * self.distance_ratio
        )

        candidate = (
            distance >= threshold
        ).astype(np.uint8)

        # ---------------------------------------------------------
        # Local maxima
        # ---------------------------------------------------------

        radius = max(
            1,
            int(self.min_peak_distance)
        )

        kernel_size = (
            2 * radius + 1
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                kernel_size,
                kernel_size
            )
        )

        local_max = (
            distance
            >= cv2.dilate(
                distance,
                kernel
            )
        )

        peaks = (
            candidate
            & local_max.astype(np.uint8)
            & component.astype(np.uint8)
        ).astype(np.uint8)

        # ---------------------------------------------------------
        # Clean very tiny peak regions
        # ---------------------------------------------------------

        n, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                peaks,
                connectivity=8
            )
        )

        clean_peaks = np.zeros_like(
            peaks,
            dtype=np.uint8
        )

        peak_count = 0

        for i in range(1, n):

            area = int(
                stats[i, cv2.CC_STAT_AREA]
            )

            if area < 1:
                continue

            clean_peaks[
                labels == i
            ] = 1

            peak_count += 1

        print(
            f"      🔹 Distance peaks: "
            f"{peak_count} "
            f"(max_dist={max_distance:.2f}, "
            f"threshold={threshold:.2f})"
        )

        return clean_peaks

    # =============================================================
    # WATERSHED SPLIT
    # =============================================================

    def _watershed_split(self, component):
        """
        Split one connected foreground component into multiple
        object instances using distance-transform watershed.
        """

        component = (
            np.asarray(component)
            .astype(np.uint8)
        )

        area = int(
            np.count_nonzero(component)
        )

        # ---------------------------------------------------------
        # Small component -> do not split
        # ---------------------------------------------------------

        if area < self.split_min_area:

            return [
                component
            ]

        distance = (
            self._distance_transform(
                component
            )
        )

        peaks = (
            self._find_distance_peaks(
                distance,
                component
            )
        )

        # ---------------------------------------------------------
        # Create markers
        # ---------------------------------------------------------

        peak_count, peak_labels = (
            cv2.connectedComponents(
                peaks,
                connectivity=8
            )
        )

        number_of_peaks = (
            peak_count - 1
        )

        # ---------------------------------------------------------
        # One peak = one object
        # ---------------------------------------------------------

        if number_of_peaks <= 1:

            return [
                component
            ]

        # ---------------------------------------------------------
        # Safety against pathological splitting
        # ---------------------------------------------------------

        if (
            number_of_peaks
            > self.max_splits_per_component
        ):

            print(
                f"      ⚠️ Too many peaks "
                f"({number_of_peaks}), "
                f"keeping component intact"
            )

            return [
                component
            ]

        print(
            f"      ✂️ Watershed splitting "
            f"component area={area}px "
            f"into {number_of_peaks} objects"
        )

        # ---------------------------------------------------------
        # Watershed markers
        # ---------------------------------------------------------

        markers = np.ones(
            component.shape,
            dtype=np.int32
        )

        markers[
            component > 0
        ] = 0

        markers[
            peak_labels > 0
        ] = (
            peak_labels[
                peak_labels > 0
            ] + 1
        )

        # ---------------------------------------------------------
        # Watershed surface
        # ---------------------------------------------------------

        max_distance = max(
            float(distance.max()),
            1.0
        )

        surface = (
            max_distance
            - distance
        )

        surface = cv2.normalize(
            surface,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

        # Outside component should be high.
        surface[
            component == 0
        ] = 255

        surface_rgb = cv2.cvtColor(
            surface,
            cv2.COLOR_GRAY2BGR
        )

        try:

            cv2.watershed(
                surface_rgb,
                markers
            )

        except Exception as e:

            print(
                f"      ⚠️ Watershed failed: {e}"
            )

            return [
                component
            ]

        # ---------------------------------------------------------
        # Extract individual watershed regions
        # ---------------------------------------------------------

        split_masks = []

        unique_labels = np.unique(
            markers
        )

        for label_id in unique_labels:

            if label_id <= 1:
                continue

            segment = (
                markers == label_id
            ).astype(np.uint8)

            segment_area = int(
                np.count_nonzero(segment)
            )

            if segment_area < 3:
                continue

            split_masks.append(
                segment
            )

        # ---------------------------------------------------------
        # Watershed did not actually produce multiple objects
        # ---------------------------------------------------------

        if len(split_masks) <= 1:

            return [
                component
            ]

        print(
            f"      ✅ Watershed produced "
            f"{len(split_masks)} individual regions"
        )

        return split_masks

    # =============================================================
    # CONVERT ONE MASK TO POLYGON
    # =============================================================

    def _mask_to_polygon_features(
        self,
        binary_mask,
        class_name='object',
        class_id=1,
        min_area=20,
        simplify_tolerance=1.2,   # CHANGED: 0.5 -> 1.2
        confidence=0.85
    ):
        """
        Convert one isolated binary object mask into one or more
        GeoJSON polygon features.

        Improvements:
          1. Gaussian smoothing before contour extraction
          2. Morphological close to fill small holes
          3. Correct epsilon formula (percentage of perimeter)
        """

        binary_mask = (
            np.asarray(binary_mask)
            .astype(np.uint8)
        )

        pixel_area = int(
            np.count_nonzero(
                binary_mask
            )
        )

        if pixel_area < min_area:

            return []

        # ---------------------------------------------------------
        # 1. Smooth mask edges before contouring
        # ---------------------------------------------------------

        contour_mask = (
            binary_mask * 255
        ).astype(np.uint8)

        # Gaussian blur to reduce pixel-level waviness
        blurred = cv2.GaussianBlur(
            contour_mask,
            (5, 5),
            0
        )

        # Re-threshold to get a clean binary mask
        smoothed = (
            blurred > 127
        ).astype(np.uint8)

        # ---------------------------------------------------------
        # 2. Fill small holes inside the object
        # ---------------------------------------------------------

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3)
        )

        smoothed = cv2.morphologyEx(
            smoothed,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=1
        )

        contour_mask = (
            smoothed * 255
        ).astype(np.uint8)

        # ---------------------------------------------------------
        # 3. Extract contours
        # ---------------------------------------------------------

        contours, _ = cv2.findContours(
            contour_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        features = []

        for contour in contours:

            if len(contour) < 3:
                continue

            area_px = float(
                cv2.contourArea(
                    contour
                )
            )

            if area_px < min_area:
                continue

            # -----------------------------------------------------
            # 4. Simplify contour
            #
            # FIXED:
            # epsilon is now a PERCENTAGE of perimeter,
            # not a raw multiplier.
            # -----------------------------------------------------

            perimeter = float(
                cv2.arcLength(
                    contour,
                    True
                )
            )

            if perimeter <= 0:
                continue

            epsilon = (
                float(simplify_tolerance)
                * perimeter
                / 100.0
            )

            approx = cv2.approxPolyDP(
                contour,
                epsilon,
                True
            )

            if len(approx) < 3:
                approx = contour

            # -----------------------------------------------------
            # Pixel -> geographic coordinates
            # -----------------------------------------------------

            geo_points = []

            for point in approx.reshape(
                -1,
                2
            ):

                x = int(
                    point[0]
                )

                y = int(
                    point[1]
                )

                lon, lat = (
                    self.pixel_to_geo(
                        x,
                        y
                    )
                )

                geo_points.append(
                    (
                        lon,
                        lat
                    )
                )

            if len(geo_points) < 3:
                continue

            if (
                geo_points[0]
                != geo_points[-1]
            ):

                geo_points.append(
                    geo_points[0]
                )

            try:

                polygon = Polygon(
                    geo_points
                )

                polygon = (
                    self._safe_polygon(
                        polygon
                    )
                )

                if polygon is None:
                    continue

                # -------------------------------------------------
                # MultiPolygon handling
                # -------------------------------------------------

                polygons = []

                if (
                    polygon.geom_type
                    == 'MultiPolygon'
                ):

                    polygons = list(
                        polygon.geoms
                    )

                else:

                    polygons = [
                        polygon
                    ]

                for poly in polygons:

                    if poly.is_empty:
                        continue

                    area_geo = (
                        float(poly.area)
                    )

                    features.append(
                        {
                            'type': 'Feature',
                            'geometry': mapping(
                                poly
                            ),
                            'properties': {
                                'class_id': int(
                                    class_id
                                ),
                                'class_name': (
                                    class_name
                                ),
                                'area_pixels': float(
                                    area_px
                                ),
                                'area_geo': (
                                    area_geo
                                ),
                                'num_points': len(
                                    list(
                                        poly.exterior.coords
                                    )
                                ),
                                'confidence': float(
                                    confidence
                                ),
                                'crs': self.crs,
                                'timestamp': (
                                    datetime.now()
                                    .isoformat()
                                )
                            }
                        }
                    )

            except Exception as e:

                print(
                    f"      ⚠️ Polygon creation "
                    f"failed: {e}"
                )

        return features

    # =============================================================
    # MASK -> GEOJSON
    # =============================================================

    def mask_to_geojson(
        self,
        mask,
        class_mapping=None,
        min_area=20,
        simplify_tolerance=1.2   # CHANGED: 0.5 -> 1.2
    ):

        features = []

        if mask is None:

            return self._empty_collection()

        mask = np.asarray(mask)

        if mask.size == 0:

            return self._empty_collection()

        if mask.dtype == bool:

            mask = mask.astype(
                np.uint8
            )

        classes = np.unique(
            mask
        )

        print(
            f"📊 mask_to_geojson: "
            f"dtype={mask.dtype}, "
            f"shape={mask.shape}, "
            f"classes={classes.tolist()}, "
            f"binary_mode={self.binary_mode}"
        )

        # =========================================================
        # BINARY MODE
        # =========================================================

        if self.binary_mode:

            binary = (
                mask > 0
            ).astype(np.uint8)

            raw_pixels = int(
                np.count_nonzero(
                    binary
                )
            )

            print(
                f"📊 Raw foreground pixels: "
                f"{raw_pixels}"
            )

            if raw_pixels < min_area:

                print(
                    f"⚠️ Foreground below "
                    f"min_area "
                    f"({raw_pixels} < {min_area})"
                )

                return self._empty_collection()

            # -----------------------------------------------------
            # Morphological cleanup
            # -----------------------------------------------------

            cleaned = (
                self._morphological_cleanup(
                    binary
                )
            )

            cleaned_pixels = int(
                np.count_nonzero(
                    cleaned
                )
            )

            print(
                f"🧹 After morphology: "
                f"{cleaned_pixels} foreground pixels"
            )

            if cleaned_pixels < min_area:

                return self._empty_collection()

            # -----------------------------------------------------
            # Class name
            # -----------------------------------------------------

            class_name = 'object'

            if (
                class_mapping
                and 1 in class_mapping
            ):

                class_name = (
                    class_mapping[1]
                )

            # -----------------------------------------------------
            # Find connected groups
            # -----------------------------------------------------

            components = (
                self._connected_components(
                    cleaned
                )
            )

            print(
                f"🔗 Connected foreground "
                f"components: "
                f"{len(components)}"
            )

            # -----------------------------------------------------
            # Process each component
            # -----------------------------------------------------

            total_instances = 0

            for index, component_info in enumerate(
                components,
                start=1
            ):

                component = (
                    component_info['mask']
                )

                area = (
                    component_info['area']
                )

                print(
                    f"   Component {index}: "
                    f"{area}px"
                )

                # -------------------------------------------------
                # Instance separation
                # -------------------------------------------------

                if (
                    self.split_touching
                    and area
                    >= self.split_min_area
                ):

                    instance_masks = (
                        self._watershed_split(
                            component
                        )
                    )

                else:

                    instance_masks = [
                        component
                    ]

                # -------------------------------------------------
                # Convert each separated instance
                # -------------------------------------------------

                for instance_index, instance_mask in enumerate(
                    instance_masks,
                    start=1
                ):

                    instance_area = int(
                        np.count_nonzero(
                            instance_mask
                        )
                    )

                    if (
                        instance_area
                        < min_area
                    ):
                        continue

                    instance_features = (
                        self._mask_to_polygon_features(
                            instance_mask,
                            class_name=class_name,
                            class_id=1,
                            min_area=min_area,
                            simplify_tolerance=(
                                simplify_tolerance
                            ),
                            confidence=0.85
                        )
                    )

                    features.extend(
                        instance_features
                    )

                    total_instances += (
                        len(instance_features)
                    )

            print(
                f"📊 FINAL individual "
                f"objects: "
                f"{total_instances}"
            )

            return {
                'type': 'FeatureCollection',
                'features': features,
                'crs': {
                    'type': 'name',
                    'properties': {
                        'name': self.crs
                    }
                }
            }

        # =========================================================
        # MULTICLASS MODE
        # =========================================================

        for class_id in classes:

            cid = int(
                class_id
            )

            if cid == 0:
                continue

            class_mask = (
                mask == cid
            ).astype(np.uint8)

            pixel_count = int(
                np.count_nonzero(
                    class_mask
                )
            )

            if pixel_count < min_area:
                continue

            if (
                class_mapping
                and cid in class_mapping
            ):

                class_name = (
                    class_mapping[cid]
                )

            else:

                class_name = (
                    f'class_{cid}'
                )

            components = (
                self._connected_components(
                    class_mask
                )
            )

            for component_info in components:

                component = (
                    component_info['mask']
                )

                area = (
                    component_info['area']
                )

                if (
                    self.split_touching
                    and area
                    >= self.split_min_area
                ):

                    instance_masks = (
                        self._watershed_split(
                            component
                        )
                    )

                else:

                    instance_masks = [
                        component
                    ]

                for instance_mask in instance_masks:

                    instance_features = (
                        self._mask_to_polygon_features(
                            instance_mask,
                            class_name=class_name,
                            class_id=cid,
                            min_area=min_area,
                            simplify_tolerance=(
                                simplify_tolerance
                            ),
                            confidence=0.85
                        )
                    )

                    features.extend(
                        instance_features
                    )

        print(
            f"📊 multiclass_mask_to_geojson: "
            f"{len(features)} features"
        )

        return {
            'type': 'FeatureCollection',
            'features': features,
            'crs': {
                'type': 'name',
                'properties': {
                    'name': self.crs
                }
            }
        }

    # =============================================================
    # EMPTY COLLECTION
    # =============================================================

    def _empty_collection(self):

        return {
            'type': 'FeatureCollection',
            'features': [],
            'crs': {
                'type': 'name',
                'properties': {
                    'name': self.crs
                }
            }
        }

    # =============================================================
    # PREDICTION POLYGONS -> GEOJSON
    # =============================================================

    def prediction_to_geojson(
        self,
        prediction_polygons,
        class_mapping=None
    ):

        features = []

        if prediction_polygons is None:
            return self._empty_collection()

        for pred in prediction_polygons:

            polygon = pred.get(
                'polygon'
            )

            if (
                polygon is None
                or polygon.is_empty
            ):
                continue

            polygon = (
                self._safe_polygon(
                    polygon
                )
            )

            if polygon is None:
                continue

            class_id = pred.get(
                'class_id',
                1
            )

            class_name = pred.get(
                'class_name',
                f'class_{class_id}'
            )

            confidence = pred.get(
                'confidence',
                0.5
            )

            area = pred.get(
                'area',
                polygon.area
            )

            # -----------------------------------------------------
            # Pixel polygon -> geographic polygon
            # -----------------------------------------------------

            if (
                self.is_geo
                and self.img_width is not None
                and self.img_height is not None
            ):

                try:

                    if polygon.geom_type == 'Polygon':

                        exterior = list(
                            polygon.exterior.coords
                        )

                        geo_exterior = [
                            self.pixel_to_geo(
                                x,
                                y
                            )
                            for x, y in exterior
                        ]

                        if (
                            geo_exterior
                            and geo_exterior[0]
                            != geo_exterior[-1]
                        ):

                            geo_exterior.append(
                                geo_exterior[0]
                            )

                        # Preserve holes.
                        geo_interiors = []

                        for ring in polygon.interiors:

                            ring_coords = list(
                                ring.coords
                            )

                            geo_ring = [
                                self.pixel_to_geo(
                                    x,
                                    y
                                )
                                for x, y
                                in ring_coords
                            ]

                            geo_interiors.append(
                                geo_ring
                            )

                        polygon = Polygon(
                            geo_exterior,
                            geo_interiors
                        )

                    elif (
                        polygon.geom_type
                        == 'MultiPolygon'
                    ):

                        converted_polygons = []

                        for poly in polygon.geoms:

                            exterior = list(
                                poly.exterior.coords
                            )

                            geo_exterior = [
                                self.pixel_to_geo(
                                    x,
                                    y
                                )
                                for x, y
                                in exterior
                            ]

                            interiors = []

                            for ring in poly.interiors:

                                ring_coords = list(
                                    ring.coords
                                )

                                geo_ring = [
                                    self.pixel_to_geo(
                                        x,
                                        y
                                    )
                                    for x, y
                                    in ring_coords
                                ]

                                interiors.append(
                                    geo_ring
                                )

                            converted_polygons.append(
                                Polygon(
                                    geo_exterior,
                                    interiors
                                )
                            )

                        polygon = MultiPolygon(
                            converted_polygons
                        )

                    polygon = (
                        self._safe_polygon(
                            polygon
                        )
                    )

                    if polygon is None:
                        continue

                except Exception as e:

                    print(
                        f"⚠️ prediction_to_geojson "
                        f"projection failed: {e}"
                    )

                    continue

            # -----------------------------------------------------
            # MultiPolygon -> individual features
            # -----------------------------------------------------

            polygons = []

            if (
                polygon.geom_type
                == 'MultiPolygon'
            ):

                polygons = list(
                    polygon.geoms
                )

            else:

                polygons = [
                    polygon
                ]

            for poly in polygons:

                if poly.is_empty:
                    continue

                features.append(
                    {
                        'type': 'Feature',
                        'geometry': mapping(
                            poly
                        ),
                        'properties': {
                            'class_id': int(
                                class_id
                            ),
                            'class_name': (
                                class_name
                            ),
                            'confidence': float(
                                confidence
                            ),
                            'area': float(
                                area
                            ),
                            'crs': self.crs,
                            'timestamp': (
                                datetime.now()
                                .isoformat()
                            )
                        }
                    }
                )

        return {
            'type': 'FeatureCollection',
            'features': features,
            'crs': {
                'type': 'name',
                'properties': {
                    'name': self.crs
                }
            }
        }

    # =============================================================
    # SAVE GEOJSON
    # =============================================================

    def save_geojson(
        self,
        geojson_data,
        output_path
    ):

        os.makedirs(
            os.path.dirname(
                output_path
            ) or '.',
            exist_ok=True
        )

        with open(
            output_path,
            'w',
            encoding='utf-8'
        ) as f:

            json.dump(
                geojson_data,
                f,
                indent=2
            )

        print(
            f"✅ GeoJSON saved to: "
            f"{output_path}"
        )

        return output_path

    # =============================================================
    # SAVE SHAPEFILE
    # =============================================================

    def save_shapefile(
        self,
        geojson_data,
        output_path
    ):

        if not geojson_data.get(
            'features'
        ):

            print(
                "⚠️ No features to "
                "save as shapefile"
            )

            return None

        try:

            gdf = (
                gpd.GeoDataFrame.from_features(
                    geojson_data['features']
                )
            )

            if gdf.empty:

                print(
                    "⚠️ Empty GeoDataFrame"
                )

                return None

            gdf.crs = self.crs

            gdf['geometry'] = (
                gdf.geometry.apply(
                    lambda g:
                    make_valid(g)
                    if not g.is_valid
                    else g
                )
            )

            gdf = gdf[
                ~gdf.geometry.is_empty
            ]

            if gdf.empty:

                print(
                    "⚠️ No valid geometries "
                    "after validation"
                )

                return None

            os.makedirs(
                os.path.dirname(
                    output_path
                ) or '.',
                exist_ok=True
            )

            gdf.to_file(
                output_path,
                driver='ESRI Shapefile'
            )

            prj_path = (
                output_path
                .replace(
                    '.shp',
                    '.prj'
                )
            )

            if not os.path.exists(
                prj_path
            ):

                with open(
                    prj_path,
                    'w',
                    encoding='utf-8'
                ) as f:

                    f.write(
                        'GEOGCS["GCS_WGS_1984",'
                        'DATUM["D_WGS_1984",'
                        'SPHEROID["WGS_1984",'
                        '6378137,298.257223563]],'
                        'PRIMEM["Greenwich",0],'
                        'UNIT["Degree",'
                        '0.017453292519943295]]'
                    )

            print(
                f"✅ Shapefile saved to: "
                f"{output_path}"
            )

            return output_path

        except Exception as e:

            print(
                f"⚠️ Error saving shapefile: "
                f"{e}"
            )

            return None

    # =============================================================
    # CREATE GEOJSON
    # =============================================================

    def create_geojson_from_features(
        self,
        features,
        metadata=None
    ):

        geojson = {
            'type': 'FeatureCollection',
            'features': features,
            'crs': {
                'type': 'name',
                'properties': {
                    'name': self.crs
                }
            }
        }

        if metadata:
            geojson['metadata'] = metadata

        return geojson