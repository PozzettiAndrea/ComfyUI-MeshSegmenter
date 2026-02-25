# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-MeshSegmenter Contributors

"""
Mask utilities extracted from sam.py for modular use.

These functions handle binary mask operations, artifact removal,
colorization, and point grid sampling.
"""

import cv2
import numpy as np
from PIL import Image
from numpy.random import RandomState

from ..data.common import NumpyTensor


def combine_bmasks(masks: NumpyTensor['n h w'], sort: bool = False) -> NumpyTensor['h w']:
    """
    Combine multiple binary masks into a single labeled mask.

    Args:
        masks: Array of binary masks (N, H, W)
        sort: If True, sort masks by area (largest first) before combining

    Returns:
        Combined mask (H, W) with integer labels 1..N, 0 for background
    """
    mask_combined = np.zeros_like(masks[0], dtype=int)
    if sort:
        masks = sorted(masks, key=lambda x: x.sum(), reverse=True)
    for i, mask in enumerate(masks):
        mask_combined[mask] = i + 1
    return mask_combined


def decompose_mask(mask: NumpyTensor['h w'], background: int = 0) -> NumpyTensor['n h w']:
    """
    Decompose a labeled mask into individual binary masks.

    Args:
        mask: Labeled mask (H, W) with integer labels
        background: Label to treat as background (excluded from output)

    Returns:
        Array of binary masks (N, H, W)
    """
    labels = np.unique(mask)
    labels = labels[labels != background]
    return mask == labels[:, None, None]


def remove_artifacts(mask: NumpyTensor['h w'], mode: str, min_area: int = 128) -> NumpyTensor['h w']:
    """
    Remove small islands or fill small holes from a mask.

    Args:
        mask: Labeled mask (H, W)
        mode: 'holes' to fill small holes, 'islands' to remove small islands
        min_area: Minimum area threshold - regions smaller than this are removed/filled

    Returns:
        Cleaned mask (H, W)
    """
    assert mode in ['holes', 'islands']
    mode_holes = (mode == 'holes')

    def remove_helper(bmask):
        # opencv connected components operates on binary masks only
        bmask = (mode_holes ^ bmask).astype(np.uint8)
        nregions, regions, stats, _ = cv2.connectedComponentsWithStats(bmask, 8)
        sizes = stats[:, -1][1:]  # Row 0 corresponds to 0 pixels
        fill = [i + 1 for i, s in enumerate(sizes) if s < min_area] + [0]
        if not mode_holes:
            fill = [i for i in range(nregions) if i not in fill]
        return np.isin(regions, fill)

    mask_combined = np.zeros_like(mask)
    for label in np.unique(mask):  # also process background
        mask_combined[remove_helper(mask == label)] = label
    return mask_combined


def colormap_mask(
    mask: NumpyTensor['h w'],
    image: NumpyTensor['h w 3'] = None,
    background: np.ndarray = None,
    foreground: np.ndarray = None,
    blend: float = 0.25,
    seed: int = 42
) -> Image.Image:
    """
    Colorize a labeled mask with random colors.

    Args:
        mask: Labeled mask (H, W) with integer labels
        image: Optional background image to blend with
        background: RGB color for background (label 0), default white
        foreground: If set, use this color for all labels (ignores random)
        blend: Blend factor when combining with image
        seed: Random seed for reproducible colors

    Returns:
        Colorized PIL Image
    """
    if background is None:
        background = np.array([255, 255, 255])

    rng = RandomState(seed)
    palette = rng.randint(0, 255, (int(np.max(mask)) + 1, 3))
    palette[0] = background

    if foreground is not None:
        for i in range(1, len(palette)):
            palette[i] = foreground

    image_mask = palette[mask.astype(int)]  # type conversion for boolean masks
    image_blend = image_mask if image is None else image_mask * (1 - blend) + image * blend
    image_blend = np.clip(image_blend, 0, 255).astype(np.uint8)
    return Image.fromarray(image_blend)


def colormap_bmask(bmask: NumpyTensor['h w']) -> Image.Image:
    """
    Colorize a single binary mask (white on black).
    """
    return colormap_mask(
        bmask,
        background=np.array([0, 0, 0]),
        foreground=np.array([255, 255, 255])
    )


def colormap_bmasks(
    masks: NumpyTensor['n h w'],
    image: NumpyTensor['h w 3'] = None,
    background: np.ndarray = None,
    blend: float = 0.25,
    seed: int = 42
) -> Image.Image:
    """
    Combine and colorize multiple binary masks.
    """
    if background is None:
        background = np.array([255, 255, 255])
    mask = combine_bmasks(masks)
    return colormap_mask(mask, image, background=background, blend=blend, seed=seed)


def point_grid_from_mask(mask: NumpyTensor['h w'], n: int) -> NumpyTensor['n 2']:
    """
    Sample points within valid mask region, normalized to [0, 1] x [0, 1].

    Args:
        mask: Binary mask (H, W) where True/1 indicates valid pixels
        n: Number of points to sample

    Returns:
        Array of (x, y) normalized coordinates, shape (n, 2)

    Raises:
        ValueError: If no valid points exist in mask
    """
    valid = np.argwhere(mask)
    if len(valid) == 0:
        raise ValueError('No valid points in mask')

    h, w = mask.shape
    n = min(n, len(valid))
    indices = np.random.choice(len(valid), n, replace=False)
    samples = valid[indices].astype(float)
    samples[:, 0] /= h - 1
    samples[:, 1] /= w - 1
    samples = samples[:, [1, 0]]  # swap to (x, y) order
    samples = samples[np.lexsort((samples[:, 1], samples[:, 0]))]
    return samples


def visualize_points(
    image: np.ndarray,
    valid_points: list,
    invalid_points: list,
    radius: int = 4,
    valid_color: tuple = (0, 255, 0),
    invalid_color: tuple = (255, 0, 0)
) -> np.ndarray:
    """
    Draw point markers on an image showing valid (green) and invalid (red) SAM points.

    Args:
        image: Input image (H, W, 3) uint8
        valid_points: List of (x, y) pixel coordinates that produced masks
        invalid_points: List of (x, y) pixel coordinates that didn't produce masks
        radius: Circle radius in pixels
        valid_color: BGR color for valid points (default green)
        invalid_color: BGR color for invalid points (default red)

    Returns:
        Image with points drawn (H, W, 3) uint8
    """
    result = image.copy()

    # Draw invalid points first (so valid ones are on top)
    for x, y in invalid_points:
        cv2.circle(result, (int(x), int(y)), radius, invalid_color, -1)
        cv2.circle(result, (int(x), int(y)), radius, (0, 0, 0), 1)  # black outline

    # Draw valid points
    for x, y in valid_points:
        cv2.circle(result, (int(x), int(y)), radius, valid_color, -1)
        cv2.circle(result, (int(x), int(y)), radius, (0, 0, 0), 1)  # black outline

    return result


def colormap_face_ids(face_ids: np.ndarray, seed: int = 42) -> np.ndarray:
    """
    Colorize face ID render for visualization.

    Args:
        face_ids: Face ID array (H, W) with -1 for background
        seed: Random seed for reproducible colors

    Returns:
        RGB image (H, W, 3) uint8
    """
    rng = RandomState(seed)

    # Get unique face IDs (excluding -1 background)
    max_id = int(face_ids.max()) + 1

    # Create palette - background is black
    palette = rng.randint(50, 255, (max_id + 1, 3)).astype(np.uint8)

    # Handle background (-1)
    output = np.zeros((*face_ids.shape, 3), dtype=np.uint8)
    valid_mask = face_ids >= 0
    output[valid_mask] = palette[face_ids[valid_mask]]

    return output


def colormap_depth(depth: np.ndarray) -> np.ndarray:
    """
    Colorize depth map for visualization.

    Args:
        depth: Depth array (H, W)

    Returns:
        RGB image (H, W, 3) uint8
    """
    # Normalize to 0-1, handling invalid values
    valid_mask = np.isfinite(depth) & (depth > 0)
    if not valid_mask.any():
        return np.zeros((*depth.shape, 3), dtype=np.uint8)

    depth_normalized = np.zeros_like(depth)
    depth_min = depth[valid_mask].min()
    depth_max = depth[valid_mask].max()

    if depth_max > depth_min:
        depth_normalized[valid_mask] = (depth[valid_mask] - depth_min) / (depth_max - depth_min)

    # Apply colormap (viridis-like)
    depth_uint8 = (depth_normalized * 255).astype(np.uint8)
    colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_VIRIDIS)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    # Set invalid pixels to black
    colored[~valid_mask] = 0

    return colored
