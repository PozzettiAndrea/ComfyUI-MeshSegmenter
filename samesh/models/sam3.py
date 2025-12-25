# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-MeshSegmenter Contributors

"""
SAM3 Model Wrapper with Custom AutomaticMaskGenerator

SAM3 doesn't have a built-in AutomaticMaskGenerator like SAM2,
so we implement one by sampling a grid of points and calling predict_inst.
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from omegaconf import OmegaConf
from tqdm import tqdm


def point_grid(points_per_side: int) -> np.ndarray:
    """
    Generate a grid of points normalized to [0, 1].

    Returns:
        (N, 2) array of (x, y) coordinates
    """
    offset = 1 / (2 * points_per_side)
    points_one_side = np.linspace(offset, 1 - offset, points_per_side)
    points_x = np.tile(points_one_side[None, :], (points_per_side, 1)).flatten()
    points_y = np.tile(points_one_side[:, None], (1, points_per_side)).flatten()
    return np.stack([points_x, points_y], axis=-1)


def calculate_stability_score(
    masks: np.ndarray,
    mask_threshold: float = 0.0,
    stability_score_offset: float = 1.0
) -> np.ndarray:
    """
    Calculate stability score for masks.

    Stability is measured as IoU between masks thresholded at different levels.
    """
    # Compute IoU between mask at threshold and mask at threshold+offset
    intersections = (masks > mask_threshold).sum(axis=(-2, -1))
    unions = (masks > mask_threshold - stability_score_offset).sum(axis=(-2, -1))

    # Avoid division by zero
    stability = np.where(unions > 0, intersections / unions, 0.0)
    return stability


def box_area(boxes: np.ndarray) -> np.ndarray:
    """Compute area of boxes."""
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Compute IoU between two sets of boxes."""
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    # Intersection
    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]

    # Union
    union = area1[:, None] + area2[None, :] - inter

    return inter / np.clip(union, 1e-8, None)


def mask_to_box(mask: np.ndarray) -> np.ndarray:
    """Convert binary mask to bounding box [x1, y1, x2, y2]."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any() or not cols.any():
        return np.array([0, 0, 0, 0])

    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]

    return np.array([x1, y1, x2 + 1, y2 + 1])


def nms_masks(
    masks: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.7
) -> np.ndarray:
    """
    Non-maximum suppression for masks using bounding box IoU.

    Returns:
        Indices of masks to keep
    """
    if len(masks) == 0:
        return np.array([], dtype=int)

    # Get bounding boxes
    boxes = np.array([mask_to_box(m) for m in masks])

    # Sort by score
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        # Compute IoU with remaining boxes
        ious = box_iou(boxes[i:i+1], boxes[order[1:]])[0]

        # Keep boxes with IoU below threshold
        inds = np.where(ious <= iou_threshold)[0]
        order = order[inds + 1]

    return np.array(keep)


class SAM3AutomaticMaskGenerator:
    """
    Automatic mask generator for SAM3 using point grid sampling.

    This mimics SAM2's AutomaticMaskGenerator by:
    1. Sampling a grid of points across the image
    2. Running predict_inst for each point
    3. Filtering by score, stability, and area
    4. Applying NMS to remove duplicates
    """

    def __init__(
        self,
        model,
        processor,
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.5,
        stability_score_thresh: float = 0.7,
        stability_score_offset: float = 1.0,
        min_mask_region_area: int = 100,
        box_nms_thresh: float = 0.7,
        crop_n_layers: int = 0,  # Not implemented yet
    ):
        """
        Args:
            model: SAM3 model (Sam3Image)
            processor: SAM3 processor
            points_per_side: Grid density for point sampling
            pred_iou_thresh: Minimum predicted IoU score
            stability_score_thresh: Minimum stability score
            stability_score_offset: Offset for stability calculation
            min_mask_region_area: Minimum mask area in pixels
            box_nms_thresh: IoU threshold for NMS
            crop_n_layers: Number of crop layers (0 = no crops)
        """
        self.model = model
        self.processor = processor
        self.points_per_side = points_per_side
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.min_mask_region_area = min_mask_region_area
        self.box_nms_thresh = box_nms_thresh
        self.crop_n_layers = crop_n_layers

        # Generate point grid
        self._points = point_grid(points_per_side)
        # SAM2 compatibility: point_grids is a list (one per crop layer)
        self._point_grids = None  # When set externally, overrides grid sampling

    @property
    def point_grids(self) -> List[np.ndarray]:
        """SAM2-compatible property for external point grid setting."""
        return self._point_grids

    @point_grids.setter
    def point_grids(self, grids: List[np.ndarray]):
        """Allow external code to set custom point grids (e.g., from mask sampling)."""
        self._point_grids = grids

    def generate(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Generate masks for an image.

        Args:
            image: (H, W, 3) numpy array

        Returns:
            List of annotation dicts with keys:
            - segmentation: (H, W) binary mask
            - area: mask area in pixels
            - bbox: [x, y, w, h] bounding box
            - predicted_iou: model's IoU prediction
            - stability_score: stability score
        """
        h, w = image.shape[:2]

        # Convert to PIL
        pil_image = Image.fromarray(image)

        # Set image (extract features once)
        state = self.processor.set_image(pil_image)

        # Use custom point_grids if set (SAM2 compatibility), otherwise use default grid
        if self._point_grids is not None and len(self._point_grids) > 0:
            # point_grids is already in normalized [0,1] coords
            points_normalized = self._point_grids[0]  # Use first grid (no crop layers for now)
            points_pixel = points_normalized.copy()
            points_pixel[:, 0] *= w
            points_pixel[:, 1] *= h
        else:
            # Use default grid
            points_pixel = self._points.copy()
            points_pixel[:, 0] *= w
            points_pixel[:, 1] *= h

        all_masks = []
        all_scores = []
        all_stability = []

        # Process each point
        for point in tqdm(points_pixel, desc="SAM3 generating masks", leave=False):
            try:
                # Call predict_inst with single point
                masks_np, scores_np, low_res_masks = self.model.predict_inst(
                    state,
                    point_coords=np.array([[point]]),
                    point_labels=np.array([1]),  # foreground
                    multimask_output=True,
                    normalize_coords=True,
                )

                # Get best mask (or all 3 if multimask)
                for i in range(len(masks_np)):
                    mask = masks_np[i]
                    score = scores_np[i]

                    # Filter by score
                    if score < self.pred_iou_thresh:
                        continue

                    # Filter by area
                    area = mask.sum()
                    if area < self.min_mask_region_area:
                        continue

                    all_masks.append(mask)
                    all_scores.append(score)

            except Exception as e:
                # Skip failed points
                continue

        if len(all_masks) == 0:
            return []

        all_masks = np.array(all_masks)
        all_scores = np.array(all_scores)

        # Apply NMS
        keep_indices = nms_masks(all_masks, all_scores, self.box_nms_thresh)

        # Build output annotations
        annotations = []
        for idx in keep_indices:
            mask = all_masks[idx]
            score = all_scores[idx]
            box = mask_to_box(mask)

            annotations.append({
                'segmentation': mask.astype(bool),
                'area': int(mask.sum()),
                'bbox': [int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1])],
                'predicted_iou': float(score),
                'stability_score': float(score),  # Use score as stability for now
            })

        return annotations


class Sam3Model(nn.Module):
    """
    SAM3 Model wrapper compatible with SAM2Model interface.

    This allows GenerateMasks node to use either SAM2 or SAM3 interchangeably.
    """

    def __init__(self, config: OmegaConf, device='cuda'):
        """
        Args:
            config: OmegaConf with sam3.checkpoint path
            device: torch device
        """
        super().__init__()
        self.config = config
        self.device = device

        self._load_model()
        self._setup_engine()

    def _load_model(self):
        """Load SAM3 model from checkpoint."""
        from ..sam3_lib.sam3_video_predictor import Sam3VideoPredictor
        from ..sam3_lib.model.sam3_image_processor import Sam3Processor

        checkpoint_path = self.config.sam3.checkpoint

        # BPE tokenizer path (relative to samesh/models/sam3.py -> samesh/sam3_lib/)
        bpe_path = Path(__file__).parent.parent / "sam3_lib" / "bpe_simple_vocab_16e6.txt.gz"
        if not bpe_path.exists():
            # Fallback to absolute path
            import importlib.util
            spec = importlib.util.find_spec("samesh.sam3_lib")
            if spec and spec.origin:
                bpe_path = Path(spec.origin).parent / "bpe_simple_vocab_16e6.txt.gz"

        print(f"[SAM3] Loading model from: {checkpoint_path}")

        # Build video predictor (contains image model)
        self._video_predictor = Sam3VideoPredictor(
            checkpoint_path=str(checkpoint_path),
            bpe_path=str(bpe_path),
            enable_inst_interactivity=True,
        )

        # Get the detector model for image segmentation
        self._detector = self._video_predictor.model.detector

        # Create processor
        self._processor = Sam3Processor(
            model=self._detector,
            resolution=1008,
            device=str(self.device),
            confidence_threshold=0.2
        )

        print(f"[SAM3] Model loaded successfully")

    def _setup_engine(self):
        """Setup automatic mask generator."""
        engine_config = self.config.sam3.get('engine_config', {})

        self.engine = SAM3AutomaticMaskGenerator(
            model=self._detector,
            processor=self._processor,
            points_per_side=engine_config.get('points_per_side', 32),
            pred_iou_thresh=engine_config.get('pred_iou_thresh', 0.5),
            stability_score_thresh=engine_config.get('stability_score_thresh', 0.7),
            stability_score_offset=engine_config.get('stability_score_offset', 1.0),
            min_mask_region_area=engine_config.get('min_mask_region_area', 100),
            box_nms_thresh=engine_config.get('box_nms_thresh', 0.7),
        )

    def process_image(self, image: Image.Image, prompt: dict = None) -> np.ndarray:
        """
        Process image and generate masks.

        Args:
            image: PIL Image
            prompt: Not used for automatic mode

        Returns:
            (N, H, W) numpy array of binary masks
        """
        image_np = np.array(image)

        annotations = self.engine.generate(image_np)

        if len(annotations) == 0:
            h, w = image_np.shape[:2]
            return np.zeros((1, h, w), dtype=bool)

        # Sort by area (largest first)
        annotations = sorted(annotations, key=lambda x: x['area'], reverse=True)
        masks = np.stack([anno['segmentation'] for anno in annotations])

        return masks

    def forward(self, image: Image.Image) -> np.ndarray:
        """
        Generate masks for image.

        Args:
            image: PIL Image

        Returns:
            (N, H, W) numpy array of binary masks
        """
        return self.process_image(image)

    def __call__(self, image: Image.Image) -> np.ndarray:
        """Allow calling model directly."""
        return self.forward(image)
