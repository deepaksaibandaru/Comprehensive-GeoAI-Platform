"""
Breain.ai GeoAI - Prediction Post-processing

Generic post-processing for semantic-segmentation probability maps.

Pipeline
--------
1. Validate / sanitize probability map
2. Smooth probabilities
3. Threshold probabilities
4. Morphological closing to repair small boundary gaps
5. Morphological opening to remove isolated noise
6. Fill internal holes
7. Remove very small connected components
8. Smooth polygon contours
9. Return clean binary mask + individual contours

This module is intentionally GENERIC.

It does not assume that objects are:
    - circles
    - storage tanks
    - buildings
    - boats
    - roads

The geometry comes entirely from the model prediction.
"""

import cv2
import numpy as np


# ============================================================
# BASIC HELPERS
# ============================================================

def _odd_kernel_size(value, minimum=1, maximum=15):
    """
    Convert a requested kernel size into a safe odd integer.
    """
    value = int(value)

    value = max(minimum, min(maximum, value))

    if value % 2 == 0:
        value += 1

    return value


def _ellipse_kernel(size):
    """
    Create an elliptical morphology kernel.
    """
    size = _odd_kernel_size(size)

    return cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (size, size),
    )


def _fill_holes(binary):
    """
    Fill internal holes in a binary foreground mask.

    Input:
        uint8 mask containing 0 / 255

    Output:
        uint8 mask containing 0 / 255
    """
    binary = (binary > 0).astype(np.uint8) * 255

    if binary.size == 0:
        return binary

    # Flood-fill background from the image border.
    flood = binary.copy()

    h, w = flood.shape[:2]

    flood_mask = np.zeros(
        (h + 2, w + 2),
        dtype=np.uint8,
    )

    cv2.floodFill(
        flood,
        flood_mask,
        (0, 0),
        255,
    )

    # Everything not reachable from the outside is a hole.
    flood_inv = cv2.bitwise_not(flood)

    # Combine original foreground and filled holes.
    filled = binary | flood_inv

    return filled


def _remove_small_components(binary, min_area):
    """
    Remove connected components smaller than min_area.

    Returns
    -------
    cleaned_mask
    contours
    diagnostics
    """
    binary = (binary > 0).astype(np.uint8) * 255

    min_area = max(1, int(min_area))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    cleaned = np.zeros_like(binary)

    retained_contours = []

    detected_components = max(0, count - 1)
    removed_components = 0

    for component_id in range(1, count):

        area = int(
            stats[
                component_id,
                cv2.CC_STAT_AREA,
            ]
        )

        if area < min_area:
            removed_components += 1
            continue

        component_mask = np.zeros_like(binary)

        component_mask[
            labels == component_id
        ] = 255

        contours, _ = cv2.findContours(
            component_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )

        if not contours:
            removed_components += 1
            continue

        # Keep component.
        cleaned[
            labels == component_id
        ] = 255

        # Usually there is one external contour.
        for contour in contours:

            contour_area = float(
                cv2.contourArea(contour)
            )

            if contour_area >= min_area:
                retained_contours.append(contour)

    diagnostics = {
        "detected_components": int(
            detected_components
        ),
        "removed_components": int(
            removed_components
        ),
        "retained_components": int(
            len(retained_contours)
        ),
    }

    return cleaned, retained_contours, diagnostics


# ============================================================
# PROBABILITY SMOOTHING
# ============================================================

def smooth_probability_map(
    probability_map,
    blur_kernel=5,
    sigma=0.0,
):
    """
    Smooth a segmentation probability map before thresholding.

    Gaussian smoothing is performed on probabilities rather than
    on the binary mask. This produces cleaner object boundaries.

    Parameters
    ----------
    probability_map : numpy.ndarray
        2D floating-point probability map.

    blur_kernel : int
        Odd Gaussian kernel size.

    sigma : float
        Gaussian sigma. 0 lets OpenCV determine it automatically.

    Returns
    -------
    numpy.ndarray
        Smoothed probability map.
    """

    probability_map = np.asarray(
        probability_map,
        dtype=np.float32,
    )

    if probability_map.ndim != 2:
        raise ValueError(
            "Probability map must be a 2D array."
        )

    if probability_map.size == 0:
        raise ValueError(
            "Probability map is empty."
        )

    probability_map = np.nan_to_num(
        probability_map,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    probability_map = np.clip(
        probability_map,
        0.0,
        1.0,
    )

    blur_kernel = _odd_kernel_size(
        blur_kernel,
        minimum=1,
        maximum=15,
    )

    if blur_kernel <= 1:
        return probability_map

    smoothed = cv2.GaussianBlur(
        probability_map,
        (
            blur_kernel,
            blur_kernel,
        ),
        sigmaX=float(sigma),
        sigmaY=float(sigma),
        borderType=cv2.BORDER_REFLECT,
    )

    return np.clip(
        smoothed,
        0.0,
        1.0,
    ).astype(np.float32)


# ============================================================
# CONTOUR SMOOTHING
# ============================================================

def smooth_contour(
    contour,
    epsilon_ratio=0.0025,
    min_points=4,
):
    """
    Simplify a contour while preserving its overall shape.

    A very small epsilon is deliberately used so that genuine
    object geometry is not destroyed.

    Parameters
    ----------
    contour : numpy.ndarray
        OpenCV contour.

    epsilon_ratio : float
        Fraction of contour perimeter used for approximation.

    min_points : int
        Minimum number of contour points to accept.

    Returns
    -------
    numpy.ndarray
        Smoothed/simplified contour.
    """

    if contour is None:
        return None

    contour = np.asarray(
        contour,
        dtype=np.int32,
    )

    if contour.ndim != 3:
        return contour

    if len(contour) < min_points:
        return contour

    perimeter = cv2.arcLength(
        contour,
        True,
    )

    if perimeter <= 0:
        return contour

    epsilon = max(
        0.5,
        float(perimeter) * float(
            epsilon_ratio
        ),
    )

    simplified = cv2.approxPolyDP(
        contour,
        epsilon,
        True,
    )

    if simplified is None:
        return contour

    if len(simplified) < 3:
        return contour

    return simplified


# ============================================================
# MAIN POST-PROCESSOR
# ============================================================

def postprocess_probability_map(
    probability_map,
    threshold=0.5,
    min_area=20,
    enabled=True,

    # Probability smoothing.
    blur_kernel=5,
    blur_sigma=0.0,

    # Morphological cleanup.
    kernel_size=3,
    open_iterations=1,
    close_iterations=1,

    # Hole filling.
    fill_holes=True,

    # Polygon smoothing.
    contour_smoothing=True,
    contour_epsilon=0.0025,
):
    """
    Convert a probability map into clean individual objects.

    This is the main function used by /predict.

    Parameters
    ----------
    probability_map : numpy.ndarray
        2D probability map with values normally between 0 and 1.

    threshold : float
        Segmentation threshold.

    min_area : int
        Minimum connected component area in pixels.

    enabled : bool
        Enable morphological cleanup.

    blur_kernel : int
        Gaussian probability smoothing kernel.

    blur_sigma : float
        Gaussian sigma.

    kernel_size : int
        Morphological kernel.

    open_iterations : int
        Opening iterations.

    close_iterations : int
        Closing iterations.

    fill_holes : bool
        Fill holes inside detected objects.

    contour_smoothing : bool
        Apply small contour simplification.

    contour_epsilon : float
        Contour approximation factor.

    Returns
    -------
    cleaned_mask : numpy.ndarray
        uint8 binary mask containing 0 / 255.

    contours : list
        Clean external contours.

    diagnostics : dict
        Processing statistics.
    """

    # --------------------------------------------------------
    # Validate probability map
    # --------------------------------------------------------

    probability_map = np.asarray(
        probability_map,
        dtype=np.float32,
    )

    if probability_map.ndim != 2:
        raise ValueError(
            "Probability map must be a 2D array."
        )

    if probability_map.size == 0:
        raise ValueError(
            "Probability map is empty."
        )

    probability_map = np.nan_to_num(
        probability_map,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    probability_map = np.clip(
        probability_map,
        0.0,
        1.0,
    )

    threshold = float(
        np.clip(
            threshold,
            0.01,
            0.99,
        )
    )

    min_area = max(
        1,
        int(min_area),
    )

    # --------------------------------------------------------
    # 1. Probability smoothing
    # --------------------------------------------------------

    if enabled and blur_kernel > 1:

        smoothed_probability = (
            smooth_probability_map(
                probability_map,
                blur_kernel=blur_kernel,
                sigma=blur_sigma,
            )
        )

    else:

        smoothed_probability = (
            probability_map.copy()
        )

    # --------------------------------------------------------
    # 2. Threshold
    # --------------------------------------------------------

    binary = (
        smoothed_probability >= threshold
    ).astype(np.uint8) * 255

    raw_foreground_pixels = int(
        np.count_nonzero(binary)
    )

    # --------------------------------------------------------
    # 3. Morphological cleanup
    # --------------------------------------------------------

    morphology_kernel = _ellipse_kernel(
        kernel_size
    )

    if enabled:

        # CLOSE first.
        #
        # This is important for your case because it repairs
        # small gaps / broken boundaries before OPEN removes noise.
        if int(close_iterations) > 0:

            binary = cv2.morphologyEx(
                binary,
                cv2.MORPH_CLOSE,
                morphology_kernel,
                iterations=min(
                    5,
                    max(
                        0,
                        int(close_iterations),
                    ),
                ),
            )

        # OPEN second.
        #
        # Removes isolated specks without destroying as much
        # of the object boundary as a large opening would.
        if int(open_iterations) > 0:

            binary = cv2.morphologyEx(
                binary,
                cv2.MORPH_OPEN,
                morphology_kernel,
                iterations=min(
                    5,
                    max(
                        0,
                        int(open_iterations),
                    ),
                )
            )

    # --------------------------------------------------------
    # 4. Fill internal holes
    # --------------------------------------------------------

    if fill_holes:

        binary = _fill_holes(
            binary
        )

    # --------------------------------------------------------
    # 5. Remove small connected components
    # --------------------------------------------------------

    cleaned_mask, contours, component_diag = (
        _remove_small_components(
            binary,
            min_area=min_area,
        )
    )

    # --------------------------------------------------------
    # 6. Final contour extraction
    #
    # Re-extract from the final cleaned mask.
    # This guarantees that contours and mask correspond exactly.
    # --------------------------------------------------------

    contours, hierarchy = cv2.findContours(
        cleaned_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    final_contours = []

    for contour in contours:

        area = float(
            cv2.contourArea(
                contour
            )
        )

        if area < float(min_area):
            continue

        if contour_smoothing:

            contour = smooth_contour(
                contour,
                epsilon_ratio=contour_epsilon,
            )

        if contour is None:
            continue

        if len(contour) < 3:
            continue

        final_contours.append(
            contour
        )

    # --------------------------------------------------------
    # Sort largest first
    # --------------------------------------------------------

    final_contours.sort(
        key=lambda c: float(
            cv2.contourArea(c)
        ),
        reverse=True,
    )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    processed_foreground_pixels = int(
        np.count_nonzero(
            cleaned_mask
        )
    )

    diagnostics = {
        "enabled": bool(enabled),

        "threshold": float(
            threshold
        ),

        "min_area_px": int(
            min_area
        ),

        "blur_kernel": int(
            blur_kernel
        ),

        "blur_sigma": float(
            blur_sigma
        ),

        "morphology_kernel": int(
            kernel_size
        ),

        "open_iterations": int(
            open_iterations
        ),

        "close_iterations": int(
            close_iterations
        ),

        "fill_holes": bool(
            fill_holes
        ),

        "contour_smoothing": bool(
            contour_smoothing
        ),

        "contour_epsilon": float(
            contour_epsilon
        ),

        "raw_foreground_pixels": int(
            raw_foreground_pixels
        ),

        "processed_foreground_pixels": int(
            processed_foreground_pixels
        ),

        "detected_components": int(
            component_diag[
                "detected_components"
            ]
        ),

        "removed_components": int(
            component_diag[
                "removed_components"
            ]
        ),

        "retained_components": int(
            len(final_contours)
        ),
    }

    return (
        cleaned_mask,
        final_contours,
        diagnostics,
    )


# ============================================================
# PROBABILITY TILE STITCHING
# ============================================================

def create_blend_weight(
    height,
    width,
    edge_fraction=0.15,
):
    """
    Create a smooth spatial weighting window for overlapping
    segmentation tiles.

    Tile centers receive greater weight than tile edges.

    This reduces visible seams between overlapping tiles.
    """

    height = int(height)
    width = int(width)

    if height <= 0 or width <= 0:
        raise ValueError(
            "Invalid blend-window dimensions."
        )

    edge_fraction = float(
        np.clip(
            edge_fraction,
            0.01,
            0.45,
        )
    )

    y = np.linspace(
        -1.0,
        1.0,
        height,
        dtype=np.float32,
    )

    x = np.linspace(
        -1.0,
        1.0,
        width,
        dtype=np.float32,
    )

    # Distance from center.
    yy, xx = np.meshgrid(
        y,
        x,
        indexing="ij",
    )

    distance = np.maximum(
        np.abs(xx),
        np.abs(yy),
    )

    # Center weighting.
    weight = 1.0 - distance

    # Prevent weights from becoming zero.
    minimum_weight = max(
        0.05,
        edge_fraction,
    )

    weight = np.clip(
        weight,
        minimum_weight,
        1.0,
    )

    return weight.astype(
        np.float32
    )


def weighted_accumulate(
    probability_sum,
    weight_sum,
    probability_tile,
    x,
    y,
):
    """
    Add one probability tile to a weighted accumulation buffer.

    Parameters
    ----------
    probability_sum : ndarray
        Accumulated weighted probabilities.

    weight_sum : ndarray
        Accumulated weights.

    probability_tile : ndarray
        Tile probability map.

    x, y : int
        Top-left pixel location.
    """

    probability_tile = np.asarray(
        probability_tile,
        dtype=np.float32,
    )

    th, tw = probability_tile.shape

    H, W = probability_sum.shape

    x1 = max(
        0,
        int(x),
    )

    y1 = max(
        0,
        int(y),
    )

    x2 = min(
        W,
        x1 + tw,
    )

    y2 = min(
        H,
        y1 + th,
    )

    if x2 <= x1 or y2 <= y1:
        return

    tile_x1 = 0
    tile_y1 = 0

    tile_x2 = x2 - x1
    tile_y2 = y2 - y1

    tile = probability_tile[
        tile_y1:tile_y2,
        tile_x1:tile_x2,
    ]

    weight = create_blend_weight(
        th,
        tw,
    )[
        tile_y1:tile_y2,
        tile_x1:tile_x2,
    ]

    probability_sum[
        y1:y2,
        x1:x2,
    ] += tile * weight

    weight_sum[
        y1:y2,
        x1:x2,
    ] += weight


def finalize_weighted_probability(
    probability_sum,
    weight_sum,
):
    """
    Convert weighted accumulation buffers into the final
    probability map.
    """

    probability_sum = np.asarray(
        probability_sum,
        dtype=np.float32,
    )

    weight_sum = np.asarray(
        weight_sum,
        dtype=np.float32,
    )

    probability = np.zeros_like(
        probability_sum,
        dtype=np.float32,
    )

    valid = weight_sum > 1e-8

    probability[valid] = (
        probability_sum[valid]
        /
        weight_sum[valid]
    )

    return np.clip(
        probability,
        0.0,
        1.0,
    ).astype(np.float32)


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================

def clean_probability_to_contours(
    probability_map,
    threshold=0.5,
    min_area=20,
):
    """
    Convenience wrapper for callers that only need clean contours.
    """

    mask, contours, diagnostics = (
        postprocess_probability_map(
            probability_map,
            threshold=threshold,
            min_area=min_area,
        )
    )

    return (
        mask,
        contours,
        diagnostics,
    )